"""T23.4 streaming, interactive, promptlab, and trace-archival matrix."""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import create_engine, text

from mlflow.genai.scorers import Guidelines
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from rust.bench.genai.equivalence import compare_runs
from rust.bench.genai.metrics import (
    AsyncBenchClient,
    MetricsCollector,
    ResourceMonitor,
    percentile,
)
from rust.bench.genai.mock_provider import provider_server
from rust.bench.genai.runner import (
    ASSISTANT_PREFIX,
    DB_URI,
    FAKE_API_KEY,
    HERE,
    RUST_ROOT,
    compose_args,
    install_claude_stub,
    launch_server,
    postgres_sample,
    recreate_database,
    run_command,
    stop_server,
    sync_request,
    validate_raw_metrics,
    write_raw_metrics,
)
from rust.bench.genai.t23_2 import DB_POOL_CONFIG, machine_state
from rust.bench.genai.t23_3 import resource_summary

SCHEMA_VERSION = "1.3.0"
SEED = 2340
CANONICAL_REQUESTS = 1_000
CANONICAL_SMALL_TRACES = 10_000
CANONICAL_LARGE_TRACES = 1_000
WARMUP_REQUESTS = 10
FRAME_GAP_MS = 1.0
ARCHIVE_NOW_TAG = "mlflow.trace.archiveNow"
SPANS_LOCATION_TAG = "mlflow.trace.spansLocation"
ARCHIVE_LOCATION_TAG = "mlflow.trace.archiveLocation"
ARCHIVE_REPO = "ARCHIVE_REPO"


@dataclass(frozen=True)
class Cell:
    family: str
    slug: str
    kind: str
    concurrency: int
    count: int
    canonical_count: int
    stream_variant: str = "none"
    provider_mode: str = "none"


@dataclass(frozen=True)
class RequestSpec:
    sequence: int
    endpoint: str
    method: str
    path: str
    json_body: Any | None
    sse: bool = False


def cell_matrix(
    requests_per_cell: int = CANONICAL_REQUESTS,
    small_traces: int = CANONICAL_SMALL_TRACES,
    large_traces: int = CANONICAL_LARGE_TRACES,
) -> list[Cell]:
    gateway = [
        Cell(
            "gateway",
            "chat-small-c1",
            "stream-chat",
            1,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
        ),
        Cell(
            "gateway",
            "chat-small-c16",
            "stream-chat",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
        ),
        Cell(
            "gateway",
            "chat-small-c64",
            "stream-chat",
            64,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
        ),
        Cell(
            "gateway",
            "chat-large-c16",
            "stream-chat",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "large",
        ),
        Cell(
            "gateway",
            "passthrough-large-c64",
            "stream-passthrough",
            64,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "large",
        ),
        Cell(
            "gateway",
            "nonstream-mixed-c16",
            "nonstream-mixed",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
        ),
    ]
    assistant = [
        Cell(
            "assistant",
            "cli-c1",
            "assistant-stream",
            1,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
            "scripted-cli",
        ),
        Cell(
            "assistant",
            "cli-c16",
            "assistant-stream",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
            "scripted-cli",
        ),
        Cell(
            "assistant",
            "cli-c64",
            "assistant-stream",
            64,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
            "scripted-cli",
        ),
        Cell(
            "assistant",
            "openai-c16",
            "assistant-stream",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
            "openai-compatible",
        ),
    ]
    promptlab = [
        Cell(
            "promptlab", "small-c1", "promptlab", 1, requests_per_cell, CANONICAL_REQUESTS, "small"
        ),
        Cell(
            "promptlab",
            "small-c16",
            "promptlab",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
        ),
        Cell(
            "promptlab",
            "small-c64",
            "promptlab",
            64,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "small",
        ),
        Cell(
            "promptlab",
            "large-c16",
            "promptlab",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
            "large",
        ),
    ]
    archival = [
        Cell(
            "archival",
            "pass-small",
            "archive-pass",
            1,
            small_traces,
            CANONICAL_SMALL_TRACES,
            "small",
        ),
        Cell(
            "archival",
            "pass-large",
            "archive-pass",
            1,
            large_traces,
            CANONICAL_LARGE_TRACES,
            "large",
        ),
        Cell(
            "archival",
            "get-trace-c1",
            "archive-get-trace",
            1,
            requests_per_cell,
            CANONICAL_REQUESTS,
        ),
        Cell(
            "archival",
            "get-trace-c16",
            "archive-get-trace",
            16,
            requests_per_cell,
            CANONICAL_REQUESTS,
        ),
        Cell(
            "archival",
            "artifact-c64",
            "archive-artifact",
            64,
            requests_per_cell,
            CANONICAL_REQUESTS,
        ),
        Cell(
            "archival", "mixed-read-c16", "archive-mixed", 16, requests_per_cell, CANONICAL_REQUESTS
        ),
    ]
    return gateway + assistant + promptlab + archival


def _sample_sequences(count: int, seed: int, label: str) -> set[int]:
    ranked = sorted(
        range(count),
        key=lambda index: hashlib.sha256(f"{seed}:{label}:{index}".encode()).digest(),
    )
    return set(ranked[: min(16, count)])


def _fixed_hex(seed: int, *parts: object, length: int) -> str:
    return hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).hexdigest()[:length]


def _guardrail_scorer(judge_endpoint_name: str) -> str:
    scorer = Guidelines(
        guidelines="The response must be present.",
        model=f"gateway:/{judge_endpoint_name}",
    )
    return json.dumps(scorer.model_dump(), sort_keys=True, separators=(",", ":"))


def setup_target(base_url: str, provider_url: str, seed: int) -> dict[str, Any]:
    with requests.Session() as session:
        experiment_id = sync_request(
            session,
            base_url,
            "POST",
            "/api/2.0/mlflow/experiments/create",
            json={
                "artifact_location": "mlflow-artifacts://localhost/t23-4/promptlab",
                "name": f"t23-4-{seed}-promptlab",
            },
        )["experiment_id"]
        secret = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "auth_config": {"api_base": f"{provider_url}/v1"},
                "created_by": "t23-bench",
                "provider": "openai",
                "secret_name": f"t23-4-fake-secret-{seed}",
                "secret_value": {"api_key": FAKE_API_KEY},
            },
        )["secret"]
        model = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "created_by": "t23-bench",
                "model_name": "genai-bench-model",
                "name": f"t23-4-fake-model-{seed}",
                "provider": "openai",
                "secret_id": secret["secret_id"],
            },
        )["model_definition"]
        endpoint_name = f"t23-4-runtime-{seed}"
        endpoint = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "created_by": "t23-bench",
                "model_configs": [
                    {
                        "linkage_type": "PRIMARY",
                        "model_definition_id": model["model_definition_id"],
                        "weight": 1.0,
                    }
                ],
                "name": endpoint_name,
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "usage_tracking": True,
            },
        )["endpoint"]
        judge_endpoint_name = f"t23-4-guardrail-judge-{seed}"
        sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "created_by": "t23-bench",
                "model_configs": [
                    {
                        "linkage_type": "PRIMARY",
                        "model_definition_id": model["model_definition_id"],
                        "weight": 1.0,
                    }
                ],
                "name": judge_endpoint_name,
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "usage_tracking": False,
            },
        )
        scorer = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/scorers/register",
            json={
                "experiment_id": experiment_id,
                "name": f"t23-4-after-guidelines-{seed}",
                "serialized_scorer": _guardrail_scorer(judge_endpoint_name),
            },
        )
        guardrail = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/guardrails/create",
            json={
                "action": "VALIDATION",
                "name": f"t23-4-after-guidelines-{seed}",
                "scorer_id": scorer["scorer_id"],
                "scorer_version": scorer["version"],
                "stage": "AFTER",
            },
        )["guardrail"]
        sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/guardrails/add-to-endpoint",
            json={
                "endpoint_id": endpoint["endpoint_id"],
                "execution_order": 1,
                "guardrail_id": guardrail["guardrail_id"],
            },
        )
        budget = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/budgets/create",
            json={
                "budget_action": "ALERT",
                "budget_amount": 1_000_000.0,
                "budget_unit": "USD",
                "created_by": "t23-bench",
                "duration": {"unit": "DAYS", "value": 30},
                "target_scope": "GLOBAL",
            },
        )["budget_policy"]
    return {
        "budget_policy_id": budget["budget_policy_id"],
        "endpoint_id": endpoint["endpoint_id"],
        "endpoint_name": endpoint_name,
        "experiment_id": experiment_id,
        "guardrail_id": guardrail["guardrail_id"],
    }


def _gateway_specs(cell: Cell, setup: dict[str, Any], seed: int) -> list[RequestSpec]:
    rng = random.Random(f"{seed}:{cell.slug}")
    specs = []
    max_tokens = 32 if cell.stream_variant == "small" else 512
    for index in range(cell.count):
        prompt = f"t23-4-{seed}-{cell.slug}-{index}-{rng.randrange(1_000_000)}"
        if cell.kind == "stream-chat":
            path = f"/gateway/{setup['endpoint_name']}/mlflow/invocations"
            body = {
                "max_tokens": max_tokens,
                "messages": [{"content": prompt, "role": "user"}],
                "stream": True,
            }
            endpoint = "gateway_chat_stream"
            sse = True
        elif cell.kind == "stream-passthrough":
            path = "/gateway/openai/v1/chat/completions"
            body = {
                "max_tokens": max_tokens,
                "messages": [{"content": prompt, "role": "user"}],
                "model": setup["endpoint_name"],
                "stream": True,
                "user": prompt,
            }
            endpoint = "gateway_passthrough_stream"
            sse = True
        else:
            choice = index % 3
            if choice == 0:
                path = f"/gateway/{setup['endpoint_name']}/mlflow/invocations"
                body = {"messages": [{"content": prompt, "role": "user"}], "stream": False}
                endpoint = "gateway_chat"
            elif choice == 1:
                path = "/gateway/openai/v1/embeddings"
                body = {"input": [prompt, f"embedding-{index}"], "model": setup["endpoint_name"]}
                endpoint = "gateway_embeddings"
            else:
                path = "/gateway/openai/v1/chat/completions"
                body = {
                    "messages": [{"content": prompt, "role": "user"}],
                    "model": setup["endpoint_name"],
                    "user": prompt,
                }
                endpoint = "gateway_passthrough"
            sse = False
        specs.append(RequestSpec(index, endpoint, "POST", path, body, sse))
    return specs


def _promptlab_specs(cell: Cell, setup: dict[str, Any], seed: int) -> list[RequestSpec]:
    template_pad = "x" * (4_000 if cell.stream_variant == "large" else 256)
    return [
        RequestSpec(
            index,
            "promptlab_create_run",
            "POST",
            "/ajax-api/2.0/mlflow/runs/create-promptlab-run",
            {
                "experiment_id": setup["experiment_id"],
                "mlflow_version": "3.12.0",
                "model_input": f"input-{seed}-{index}",
                "model_output": f"output-{seed}-{index}",
                "model_parameters": [{"key": "temperature", "value": "0"}],
                "model_route": setup["endpoint_name"],
                "prompt_parameters": [{"key": "question", "value": f"question-{index}"}],
                "prompt_template": f"Answer {{question}}. {template_pad}",
                "run_name": f"t23-4-{cell.slug}-{seed}-{index}",
                "start_time": 1_750_000_000_000 + index,
                "user_id": "t23-bench",
            },
        )
        for index in range(cell.count)
    ]


async def _execute_specs(
    base_url: str,
    specs: list[RequestSpec],
    concurrency: int,
    sample: set[int],
    *,
    warmup: RequestSpec | None = None,
    timeout_seconds: float = 300,
) -> MetricsCollector:
    collector = MetricsCollector()
    client = AsyncBenchClient(base_url, concurrency, collector, timeout_seconds=timeout_seconds)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        if warmup is not None:
            for _ in range(WARMUP_REQUESTS):
                await client.request(
                    warmup.endpoint,
                    warmup.method,
                    warmup.path,
                    measured=False,
                    sse=warmup.sse,
                    json=warmup.json_body,
                )
        collector.started = time.perf_counter()

        async def one(spec: RequestSpec) -> None:
            async with semaphore:
                await client.request(
                    spec.endpoint,
                    spec.method,
                    spec.path,
                    capture_response=(
                        spec.sequence in sample or spec.endpoint == "promptlab_create_run"
                    ),
                    json=spec.json_body,
                    sequence=spec.sequence,
                    sse=spec.sse,
                )

        await asyncio.gather(*(one(spec) for spec in specs))
        collector.close()
        return collector
    finally:
        await client.close()


async def _execute_assistant(
    base_url: str,
    cell: Cell,
    seed: int,
    sample: set[int],
) -> MetricsCollector:
    collector = MetricsCollector()
    client = AsyncBenchClient(base_url, cell.concurrency, collector, timeout_seconds=300)
    semaphore = asyncio.Semaphore(cell.concurrency)
    try:
        collector.started = time.perf_counter()

        async def one(index: int) -> None:
            async with semaphore:
                status, message, _ = await client.request(
                    "assistant_session",
                    "POST",
                    f"{ASSISTANT_PREFIX}/message",
                    capture_response=index in sample,
                    json={
                        "context": {"phase": "23.4", "sequence": index},
                        "message": f"assistant seed {seed} turn {index}",
                    },
                    sequence=index * 2,
                )
                if status != 200 or not isinstance(message, dict) or "session_id" not in message:
                    return
                await client.request(
                    "assistant_stream",
                    "GET",
                    f"{ASSISTANT_PREFIX}/sessions/{message['session_id']}/stream",
                    capture_response=True,
                    sequence=index * 2 + 1,
                    sse=True,
                )

        await asyncio.gather(*(one(index) for index in range(cell.count)))
        collector.close()
        return collector
    finally:
        await client.close()


def _check_stream_completion(collector: MetricsCollector, family: str) -> None:
    for record in collector.records:
        if record.sse is None or record.error is not None:
            continue
        if not record.sse.frames:
            record.error = "stream completed without SSE frames"
            continue
        last = record.sse.frames[-1]
        if family == "assistant" and not last.startswith("event: done"):
            record.error = f"assistant stream missing terminal done event: {last[:120]}"
        elif family == "gateway" and not any(
            '"finish_reason":"stop"' in frame for frame in record.sse.frames
        ):
            record.error = f"gateway stream missing terminal stop frame: {last[:120]}"


def _build_equivalence(
    records: list[dict[str, Any]], sample: set[int], family: str
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        logical = record["sequence"] // 2 if family == "assistant" else record["sequence"]
        if logical not in sample:
            continue
        response = record["response"]
        if (
            family == "archival"
            and record["endpoint"] == "get_trace"
            and isinstance(response, dict)
        ):
            # The benchmark owns archive payload/read equivalence, not known
            # target-specific TraceInfo preview/artifact-location decoration.
            trace = response.get("trace", {})
            response = {"trace": {"spans": trace.get("spans", [])}}
        selected.append({
            "endpoint": record["endpoint"],
            "method": record["method"],
            "path": record["path"],
            "response": response,
            "sequence": record["sequence"],
            "sse_frames": record["sse"]["frames"] if record["sse"] else None,
            "status": record["status"],
        })
    return sorted(selected, key=lambda item: item["sequence"])


def _raw_value(
    target: str,
    cell: Cell,
    seed: int,
    collector: MetricsCollector,
    monitor: ResourceMonitor,
    state: dict[str, Any],
    provider: Any,
    provider_url: str,
    provider_start: int,
    sample: set[int],
    started_at: dt.datetime,
) -> dict[str, Any]:
    endpoints, overall = collector.summary()
    overall["resources"] = resource_summary(monitor.samples)
    records = collector.raw_records()
    proof_samples = _build_equivalence(records, sample, cell.family)
    for record in records:
        if sse := record.get("sse"):
            record["response"] = {"sse_frame_count": sse["frame_count"]}
            sse["frames"] = []
        elif cell.family == "promptlab":
            record["response"] = {"run_status": record["response"]["run"]["info"]["status"]}
    with provider.state.lock:
        observations = provider.state.observations[provider_start:]
    measured = cell.count * 2 if cell.family == "assistant" else cell.count
    trimmed = cell.count != cell.canonical_count
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "canonical_requests": cell.canonical_count,
            "concurrency": cell.concurrency,
            "db_pool_config": DB_POOL_CONFIG,
            "family": cell.family,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "measured_requests": measured,
            "provider_mode": cell.provider_mode,
            "seed": seed,
            "started_at": started_at.isoformat(),
            "stream_variant": cell.stream_variant,
            "target": target,
            "trim_note": (
                f"measured {cell.count}; canonical design is {cell.canonical_count}"
                if trimmed
                else None
            ),
            "trimmed": trimmed,
            "warmup_requests": WARMUP_REQUESTS if cell.family == "gateway" else 0,
            "workload": f"t23_4/{cell.family}/{cell.slug}",
        },
        "summary": {"endpoints": endpoints, "overall": overall},
        "requests": records,
        "jobs": [],
        "resources": {
            "method": "1 s /proc whole-process-tree VmRSS, threads, processes, utime/stime sum",
            "samples": monitor.samples,
        },
        "db_pool": {
            "method": "pg_stat_activity (server-side pool occupancy proxy)",
            "samples": monitor.pool_samples,
        },
        "provider": {
            "observations": [observation.__dict__ for observation in observations],
            "route_latency_ms": provider.state.route_latency_ms,
            "seed": seed,
            "url": provider_url,
        },
        "machine_state": state,
        "equivalence": {
            "jobs": [],
            "sample_seed": seed,
            "samples": proof_samples,
            "verdict": "PENDING",
        },
    }


def run_request_cell(
    handle: Any,
    target: str,
    cell: Cell,
    setup: dict[str, Any],
    provider: Any,
    provider_url: str,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    state = machine_state(handle)
    sample = _sample_sequences(cell.count, seed, cell.slug)
    if cell.family == "assistant":
        selected = "mlflow_gateway" if cell.provider_mode == "openai-compatible" else "claude_code"
        sync_request(
            requests.Session(),
            handle.url,
            "PUT",
            f"{ASSISTANT_PREFIX}/config",
            json={"providers": {selected: {"model": setup["endpoint_name"], "selected": True}}},
        )
        specs = None
    elif cell.family == "gateway":
        specs = _gateway_specs(cell, setup, seed)
    else:
        specs = _promptlab_specs(cell, setup, seed)
    with provider.state.lock:
        provider_start = len(provider.state.observations)
    monitor = ResourceMonitor(handle.process.pid, postgres_sample)
    started_at = dt.datetime.now(dt.timezone.utc)
    monitor.started = time.monotonic()
    monitor.start()
    try:
        if cell.family == "assistant":
            collector = asyncio.run(_execute_assistant(handle.url, cell, seed, sample))
        else:
            assert specs is not None
            collector = asyncio.run(
                _execute_specs(
                    handle.url,
                    specs,
                    cell.concurrency,
                    sample,
                    warmup=specs[0] if cell.family == "gateway" else None,
                )
            )
        if cell.family in {"assistant", "gateway"}:
            _check_stream_completion(collector, cell.family)
        elif cell.family == "promptlab":
            for record in collector.records:
                run = record.response.get("run", {}) if isinstance(record.response, dict) else {}
                status = run.get("info", {}).get("status")
                if record.error is None and status != "FINISHED":
                    record.error = f"promptlab run ended in {status!r}"
    finally:
        monitor.close()
    value = _raw_value(
        target,
        cell,
        seed,
        collector,
        monitor,
        state,
        provider,
        provider_url,
        provider_start,
        sample,
        started_at,
    )
    write_raw_metrics(value, output)
    return value


def _trace_span(seed: int, label: str, index: int, payload_bytes: int) -> dict[str, Any]:
    trace_hex = _fixed_hex(seed, label, index, length=32)
    span_hex = _fixed_hex(seed, label, index, "span", length=16)
    output = f"seeded-{label}-{index}-" + "x" * max(0, payload_bytes - 64)
    start_ns = 1_700_000_000_000_000_000 + index * 2_000_000
    return {
        "attributes": [
            {
                "key": "mlflow.spanInputs",
                "value": {"stringValue": json.dumps({"index": index, "label": label})},
            },
            {"key": "mlflow.spanOutputs", "value": {"stringValue": json.dumps(output)}},
            {"key": "mlflow.spanType", "value": {"stringValue": "CHAIN"}},
        ],
        "endTimeUnixNano": str(start_ns + 1_000_000),
        "name": f"archive-{label}-{index}",
        "spanId": span_hex,
        "startTimeUnixNano": str(start_ns),
        "status": {"code": 1},
        "traceId": trace_hex,
    }


def seed_archive_cell(base_url: str, cell: Cell, seed: int) -> tuple[str, list[str]]:
    experiment_id = sync_request(
        requests.Session(),
        base_url,
        "POST",
        "/api/2.0/mlflow/experiments/create",
        json={"name": f"t23-4-{seed}-{cell.slug}"},
    )["experiment_id"]
    payload_bytes = 256 if cell.stream_variant == "small" else 64 * 1024
    batch_size = 500 if cell.stream_variant == "small" else 10
    trace_ids = [
        f"tr-{_fixed_hex(seed, cell.slug, index, length=32)}" for index in range(cell.count)
    ]
    with requests.Session() as session:
        for start in range(0, cell.count, batch_size):
            spans = [
                _trace_span(seed, cell.slug, index, payload_bytes)
                for index in range(start, min(cell.count, start + batch_size))
            ]
            response = session.post(
                base_url + "/v1/traces",
                headers={
                    "content-type": "application/json",
                    "x-mlflow-experiment-id": experiment_id,
                },
                json={
                    "resourceSpans": [
                        {
                            "resource": {"attributes": []},
                            "scopeSpans": [{"scope": {"name": "t23-4-archive"}, "spans": spans}],
                        }
                    ]
                },
                timeout=300,
            )
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"archive seed failed at {start}: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
        sync_request(
            session,
            base_url,
            "POST",
            "/api/2.0/mlflow/experiments/set-experiment-tag",
            json={
                "experiment_id": experiment_id,
                "key": ARCHIVE_NOW_TAG,
                "value": json.dumps({"older_than": None}, separators=(",", ":")),
            },
        )
    return experiment_id, trace_ids


def write_archive_config(path: Path, location: str, enabled: bool, limit: int) -> None:
    path.write_text(
        "trace_archival:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        f"  location: {location}\n"
        "  retention: 1d\n"
        "  interval_seconds: 1\n"
        f"  max_traces_per_pass: {limit}\n"
    )


def _archived_rows(connection: Any, experiment_id: str) -> list[tuple[str, str]]:
    rows = connection.execute(
        text(
            "SELECT l.request_id, l.value FROM trace_tags l "
            "JOIN trace_tags s ON s.request_id=l.request_id "
            "JOIN trace_info i ON i.request_id=l.request_id "
            "WHERE l.key=:location_key AND s.key=:spans_key "
            "AND s.value=:archive_repo AND i.experiment_id=:experiment_id"
        ),
        {
            "archive_repo": ARCHIVE_REPO,
            "experiment_id": experiment_id,
            "location_key": ARCHIVE_LOCATION_TAG,
            "spans_key": SPANS_LOCATION_TAG,
        },
    ).all()
    return [(str(trace_id), str(uri)) for trace_id, uri in rows]


def wait_for_archive(
    experiment_id: str, expected_count: int, timeout_seconds: float
) -> tuple[list[tuple[str, str]], list[float], float]:
    deadline = time.monotonic() + timeout_seconds
    first_visible: float | None = None
    previous_time: float | None = None
    previous_count = 0
    cadence_ms: list[float] = []
    engine = create_engine(DB_URI, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            while time.monotonic() < deadline:
                rows = _archived_rows(connection, experiment_id)
                now = time.monotonic()
                count = len(rows)
                if count > previous_count:
                    first_visible = first_visible or now
                    if previous_time is not None:
                        cadence_ms.extend(
                            [(now - previous_time) * 1000 / (count - previous_count)]
                            * (count - previous_count)
                        )
                    previous_time = now
                    previous_count = count
                if count == expected_count:
                    first_visible = first_visible or now
                    duration = max(1e-9, now - first_visible)
                    return sorted(rows), cadence_ms, duration
                time.sleep(0.05)
    finally:
        engine.dispose()
    raise RuntimeError(f"archive pass reached {previous_count}/{expected_count} traces")


def _archive_payload(uri: str) -> bytes:
    local_path = get_artifact_repository(uri).download_artifacts("traces.pb")
    return Path(local_path).read_bytes()


def run_archive_pass(
    handle: Any,
    target: str,
    cell: Cell,
    config_path: Path,
    archive_location: str,
    provider_url: str,
    output: Path,
    seed: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    state = machine_state(handle)
    experiment_id, trace_ids = seed_archive_cell(handle.url, cell, seed)
    monitor = ResourceMonitor(handle.process.pid, postgres_sample)
    started_at = dt.datetime.now(dt.timezone.utc)
    monitor.started = time.monotonic()
    monitor.start()
    try:
        write_archive_config(config_path, archive_location, True, cell.count)
        rows, cadence_ms, duration = wait_for_archive(
            experiment_id, len(trace_ids), timeout_seconds
        )
    finally:
        monitor.close()
    payload = _archive_payload(rows[0][1])
    throughput = cell.count / duration
    finalize = {
        "max": max(cadence_ms) if cadence_ms else duration * 1000 / cell.count,
        "p50": percentile(cadence_ms, 50) or duration * 1000 / cell.count,
        "p95": percentile(cadence_ms, 95) or duration * 1000 / cell.count,
        "p99": percentile(cadence_ms, 99) or duration * 1000 / cell.count,
    }
    trimmed = cell.count != cell.canonical_count
    value = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "canonical_traces": cell.canonical_count,
            "concurrency": 1,
            "db_pool_config": DB_POOL_CONFIG,
            "family": cell.family,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "measured_traces": cell.count,
            "provider_mode": "none",
            "seed": seed,
            "started_at": started_at.isoformat(),
            "stream_variant": cell.stream_variant,
            "target": target,
            "trim_note": (
                f"measured {cell.count}; canonical design is {cell.canonical_count}"
                if trimmed
                else None
            ),
            "trimmed": trimmed,
            "warmup_requests": 0,
            "workload": f"t23_4/{cell.family}/{cell.slug}",
        },
        "summary": {
            "endpoints": {},
            "overall": {
                "duration_seconds": duration,
                "error_rate": 0.0,
                "errors": 0,
                "latency_ms": {"max": None, "p50": None, "p95": None, "p99": None},
                "requests": 0,
                "rps": 0.0,
                "resources": resource_summary(monitor.samples),
                "archival": {
                    "archived_traces": cell.count,
                    "experiment_id": experiment_id,
                    "finalize_transaction_latency_ms": finalize,
                    "finalize_transaction_measurement": (
                        "consecutive ARCHIVE_REPO commit-visibility cadence sampled at 50 ms; "
                        "because the pass is sequential, gaps include the next trace upload"
                    ),
                    "traces_per_second": throughput,
                },
            },
        },
        "requests": [],
        "jobs": [],
        "resources": {
            "method": "1 s /proc whole-process-tree VmRSS, threads, processes, utime/stime sum",
            "samples": monitor.samples,
        },
        "db_pool": {
            "method": "pg_stat_activity (server-side pool occupancy proxy)",
            "samples": monitor.pool_samples,
        },
        "provider": {"observations": [], "route_latency_ms": {}, "seed": seed, "url": provider_url},
        "machine_state": state,
        "equivalence": {
            "jobs": [],
            "sample_seed": seed,
            "samples": [
                {
                    "archive_payload_b64": base64.b64encode(payload).decode(),
                    "archive_payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "archive_payload_bytes": len(payload),
                    "endpoint": "archive_pass",
                    "sequence": 0,
                    "status": 200,
                    "trace_id": rows[0][0],
                }
            ],
            "verdict": "PENDING",
        },
    }
    write_raw_metrics(value, output)
    return value, trace_ids


def _archive_read_specs(cell: Cell, trace_ids: list[str]) -> list[RequestSpec]:
    specs = []
    for index in range(cell.count):
        trace_id = trace_ids[index % len(trace_ids)]
        artifact = cell.kind == "archive-artifact" or (
            cell.kind == "archive-mixed" and index % 2 == 1
        )
        if artifact:
            specs.append(
                RequestSpec(
                    index,
                    "get_trace_artifact",
                    "GET",
                    f"/ajax-api/3.0/mlflow/get-trace-artifact?request_id={trace_id}",
                    None,
                )
            )
        else:
            specs.append(
                RequestSpec(
                    index,
                    "get_trace",
                    "GET",
                    "/api/3.0/mlflow/traces/get",
                    {"allow_partial": True, "trace_id": trace_id},
                )
            )
    return specs


def run_archive_read(
    handle: Any,
    target: str,
    cell: Cell,
    trace_ids: list[str],
    provider: Any,
    provider_url: str,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    state = machine_state(handle)
    specs = _archive_read_specs(cell, trace_ids)
    sample = _sample_sequences(cell.count, seed, cell.slug)
    collector = MetricsCollector()

    async def execute() -> None:
        client = AsyncBenchClient(handle.url, cell.concurrency, collector, timeout_seconds=300)
        semaphore = asyncio.Semaphore(cell.concurrency)
        collector.started = time.perf_counter()

        async def one(spec: RequestSpec) -> None:
            async with semaphore:
                kwargs = {"json": spec.json_body} if spec.json_body is not None else {}
                await client.request(
                    spec.endpoint,
                    spec.method,
                    spec.path,
                    capture_response=spec.sequence in sample,
                    sequence=spec.sequence,
                    **kwargs,
                )

        try:
            await asyncio.gather(*(one(spec) for spec in specs))
            collector.close()
        finally:
            await client.close()

    with provider.state.lock:
        provider_start = len(provider.state.observations)
    monitor = ResourceMonitor(handle.process.pid, postgres_sample)
    started_at = dt.datetime.now(dt.timezone.utc)
    monitor.started = time.monotonic()
    monitor.start()
    try:
        asyncio.run(execute())
    finally:
        monitor.close()
    value = _raw_value(
        target,
        cell,
        seed,
        collector,
        monitor,
        state,
        provider,
        provider_url,
        provider_start,
        sample,
        started_at,
    )
    write_raw_metrics(value, output)
    return value


def mark_verdict(path: Path, verdict: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    value["equivalence"]["verdict"] = verdict
    write_raw_metrics(value, path)
    return value


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _resource(value: dict[str, Any]) -> tuple[float, float]:
    resources = value["summary"]["overall"]["resources"]
    return resources["peak_rss_mib"], resources["cpu_seconds"]


def summary_markdown(output_dir: Path, cells: list[Cell]) -> str:
    lines = [
        "# T23.4 streaming + archival benchmark summary",
        "",
        "This is raw material for T23.5, not the final Phase 23 report. Targets ran",
        "serially on PostgreSQL 16 + MinIO with a fresh DB and artifact prefix per target.",
        "Trace payloads used a fresh local file:// ARCHIVE_REPO per target because the Rust",
        "artifact factory does not currently wire S3. Promptlab used the",
        "`mlflow-artifacts://localhost/` proxy URI: Python proxied it to MinIO and Rust",
        "used its fresh local proxy destination.",
        "All upstream traffic used the loopback deterministic provider, fake Claude CLI,",
        "or the assistant's OpenAI-compatible gateway stub; no live provider was reachable.",
        "An AFTER Guidelines guardrail backed by the deterministic mock provider and a",
        "global ALERT budget were attached/enabled on",
        "the measured gateway endpoint. Per contract, post-LLM guardrails are loaded but",
        "not executed on streams. Usage tracking remained enabled so budget accounting ran.",
        "",
        "## Chosen matrix",
        "",
        "The fractional design keeps 4-6 cells per family while covering 1/16/64 stream",
        "concurrency, both ~10 and 100+ frame gateway variants, both assistant stub modes,",
        "promptlab payload/concurrency pressure, two archive payload sizes, and both read APIs.",
        "No volumes were trimmed; every cell ran at its canonical count.",
        "",
        "| Family | Cell | Kind | C | Count | Canonical | Rationale |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    rationale = {
        "chat-small-c1": "single-stream baseline",
        "chat-small-c16": "ordinary multiplexing",
        "chat-small-c64": "high stream fan-out",
        "chat-large-c16": "100+ frame stream cost",
        "passthrough-large-c64": "high-fanout 100+ frame passthrough",
        "nonstream-mixed-c16": "chat, embeddings, passthrough baseline",
        "cli-c1": "scripted CLI baseline",
        "cli-c16": "CLI multiplexing",
        "cli-c64": "CLI process fan-out",
        "openai-c16": "OpenAI-compatible assistant path",
        "small-c1": "artifact writer baseline",
        "small-c16": "artifact writer multiplexing",
        "small-c64": "artifact writer saturation",
        "large-c16": "large prompt artifact pressure",
        "pass-small": "10k small-trace pass when untrimmed",
        "pass-large": "1k 64-KiB-trace pass",
        "get-trace-c1": "archived getTrace baseline",
        "get-trace-c16": "archived getTrace multiplexing",
        "artifact-c64": "archived artifact high concurrency",
        "mixed-read-c16": "balanced archived read APIs",
    }
    lines.extend(
        (
            f"| {cell.family} | `{cell.slug}` | {cell.kind} | {cell.concurrency} | "
            f"{cell.count:,} | {cell.canonical_count:,} | {rationale[cell.slug]} |"
        )
        for cell in cells
    )

    stream_cells = [
        cell for cell in cells if cell.family == "assistant" or cell.kind.startswith("stream-")
    ]
    lines.extend([
        "",
        "## Streaming and interactive cells",
        "",
        "| Family/cell | N streams | Py TTFE p50/p95 ms | Rust TTFE p50/p95 ms | "
        "Py/Rust gap p95 ms | Py/Rust frames/s | Py/Rust completion errors | "
        "Py/Rust RSS MiB | Py/Rust CPU-s | Eq |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    slower = []
    for cell in stream_cells:
        py = json.loads((output_dir / f"{cell.family}-{cell.slug}-python.json").read_text())
        rust = json.loads((output_dir / f"{cell.family}-{cell.slug}-rust.json").read_text())
        endpoint = (
            "assistant_stream"
            if cell.family == "assistant"
            else next(name for name in py["summary"]["endpoints"] if "stream" in name)
        )
        pys = py["summary"]["endpoints"][endpoint]["sse"]
        rss = rust["summary"]["endpoints"][endpoint]["sse"]
        pyres = _resource(py)
        rures = _resource(rust)
        if rss["time_to_first_event_ms"]["p95"] > pys["time_to_first_event_ms"]["p95"]:
            slower.append((
                f"{cell.family}/{cell.slug} TTFE p95",
                pys["time_to_first_event_ms"]["p95"],
                rss["time_to_first_event_ms"]["p95"],
            ))
        py_ttfe = pys["time_to_first_event_ms"]
        rust_ttfe = rss["time_to_first_event_ms"]
        lines.append(
            f"| `{cell.family}/{cell.slug}` | {cell.count:,} | "
            f"{_fmt(py_ttfe['p50'])}/{_fmt(py_ttfe['p95'])} | "
            f"{_fmt(rust_ttfe['p50'])}/{_fmt(rust_ttfe['p95'])} | "
            f"{_fmt(pys['inter_frame_gap_ms']['p95'])}/{_fmt(rss['inter_frame_gap_ms']['p95'])} | "
            f"{pys['frames_per_second']:.1f}/{rss['frames_per_second']:.1f} | "
            f"{pys['completion_errors']}/{rss['completion_errors']} | "
            f"{pyres[0]:.1f}/{rures[0]:.1f} | {pyres[1]:.2f}/{rures[1]:.2f} | "
            f"{rust['equivalence']['verdict']} |"
        )

    nonstream = [
        cell for cell in cells if cell.family == "promptlab" or cell.kind == "nonstream-mixed"
    ]
    lines.extend([
        "",
        "## Non-streaming gateway + promptlab",
        "",
        "| Family/cell | N | Py p50/p95 ms | Rust p50/p95 ms | Py/Rust RPS | "
        "Py/Rust errors | Py/Rust RSS MiB | Py/Rust CPU-s | Eq |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for cell in nonstream:
        py = json.loads((output_dir / f"{cell.family}-{cell.slug}-python.json").read_text())
        rust = json.loads((output_dir / f"{cell.family}-{cell.slug}-rust.json").read_text())
        pyl = py["summary"]["overall"]["latency_ms"]
        rul = rust["summary"]["overall"]["latency_ms"]
        pyres = _resource(py)
        rures = _resource(rust)
        if rul["p95"] > pyl["p95"]:
            slower.append((f"{cell.family}/{cell.slug} p95", pyl["p95"], rul["p95"]))
        lines.append(
            f"| `{cell.family}/{cell.slug}` | {cell.count:,} | "
            f"{_fmt(pyl['p50'])}/{_fmt(pyl['p95'])} | "
            f"{_fmt(rul['p50'])}/{_fmt(rul['p95'])} | "
            f"{py['summary']['overall']['rps']:.1f}/{rust['summary']['overall']['rps']:.1f} | "
            f"{py['summary']['overall']['errors']}/{rust['summary']['overall']['errors']} | "
            f"{pyres[0]:.1f}/{rures[0]:.1f} | {pyres[1]:.2f}/{rures[1]:.2f} | "
            f"{rust['equivalence']['verdict']} |"
        )

    archival = [cell for cell in cells if cell.family == "archival"]
    lines.extend(["", "## Trace archival", ""])
    for cell in archival:
        py = json.loads((output_dir / f"archival-{cell.slug}-python.json").read_text())
        rust = json.loads((output_dir / f"archival-{cell.slug}-rust.json").read_text())
        if cell.kind == "archive-pass":
            pya = py["summary"]["overall"]["archival"]
            rsa = rust["summary"]["overall"]["archival"]
            lines.extend([
                f"### {cell.slug}",
                "",
                "| Target | Traces | traces/s | finalize visibility p50/p95 ms | "
                "RSS MiB | CPU-s | Eq |",
                "| --- | ---: | ---: | --- | ---: | ---: | --- |",
                f"| Python | {cell.count:,} | {pya['traces_per_second']:.1f} | "
                f"{pya['finalize_transaction_latency_ms']['p50']:.2f}/"
                f"{pya['finalize_transaction_latency_ms']['p95']:.2f} | "
                f"{_resource(py)[0]:.1f} | {_resource(py)[1]:.2f} | "
                f"{py['equivalence']['verdict']} |",
                f"| Rust | {cell.count:,} | {rsa['traces_per_second']:.1f} | "
                f"{rsa['finalize_transaction_latency_ms']['p50']:.2f}/"
                f"{rsa['finalize_transaction_latency_ms']['p95']:.2f} | "
                f"{_resource(rust)[0]:.1f} | {_resource(rust)[1]:.2f} | "
                f"{rust['equivalence']['verdict']} |",
                "",
            ])
            if rsa["traces_per_second"] < pya["traces_per_second"]:
                slower.append((
                    f"archival/{cell.slug} seconds/trace",
                    1 / pya["traces_per_second"],
                    1 / rsa["traces_per_second"],
                ))
        else:
            pyl = py["summary"]["overall"]["latency_ms"]
            rul = rust["summary"]["overall"]["latency_ms"]
            lines.append(
                f"- `{cell.slug}` ({cell.count:,} reads, c{cell.concurrency}): p50/p95 ms "
                f"Python {_fmt(pyl['p50'])}/{_fmt(pyl['p95'])}, Rust "
                f"{_fmt(rul['p50'])}/{_fmt(rul['p95'])}; RPS "
                f"{py['summary']['overall']['rps']:.1f}/{rust['summary']['overall']['rps']:.1f}; "
                f"errors {py['summary']['overall']['errors']}/"
                f"{rust['summary']['overall']['errors']}; "
                f"equivalence {rust['equivalence']['verdict']}."
            )
            if rul["p95"] > pyl["p95"]:
                slower.append((f"archival/{cell.slug} p95", pyl["p95"], rul["p95"]))
    lines.extend([
        "",
        "Archive `traces.pb` equivalence uses the T21 byte-parity payload itself: one",
        "deterministic payload per pass is stored base64 + SHA-256 in both raw files.",
        "Archived getTrace proof compares its complete ordered spans, excluding known",
        "target-specific TraceInfo preview and artifact-location decoration.",
        "SSE equivalence strips IDs/timing through the shared recorder normalizer and",
        "compares the complete ordered frame payload sequence for 16 seeded streams/cell.",
        "A cell counts only after both raw files are marked PASS.",
        "",
        "Finalize latency is a 50 ms poll of consecutive ARCHIVE_REPO tag-commit visibility.",
        "The pass is sequential, so each visibility gap includes the next trace's upload;",
        "it is an operational finalize-cadence proxy, not isolated SQL COMMIT duration.",
        "",
        "## Rust-slower cells and anomalies",
        "",
    ])
    if slower:
        lines.extend(
            f"- `{name}`: Python {_fmt(python)}, Rust {_fmt(rust)}."
            for name, python, rust in slower
        )
    else:
        lines.append("No measured Rust p95/TTFE/seconds-per-trace value exceeded Python.")
    lines.extend([
        "- Parsed SSE frames delivered in one socket read share a timestamp, so some",
        "  client-observed inter-frame p95 values round to 0.00 ms despite the provider's",
        "  fixed 1 ms write gap.",
        "- RSS is whole process-tree RSS: Python includes four uvicorn workers plus its job",
        "  runtime, while Rust includes its server and any native workers.",
    ])
    lines.extend(["", "## Raw result inventory", ""])
    lines.extend(f"- `{path.name}`" for path in sorted(output_dir.glob("*.json")))
    return "\n".join(lines) + "\n"


def _validate_cell(value: dict[str, Any], cell: Cell) -> None:
    errors = value["summary"]["overall"]["errors"]
    denominator = cell.count
    if errors / denominator >= 0.0001:
        raise RuntimeError(
            f"{value['run']['target']} {cell.slug} had {errors}/{denominator} errors"
        )


def _assert_hygiene() -> None:
    try:
        machine_state(None)
    except RuntimeError:
        run_command(
            ["cargo", "run", "-p", "mlflow-test-support", "--bin", "reap-reference-servers"],
            cwd=RUST_ROOT,
        )
        machine_state(None)


def matrix(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    all_cells = cell_matrix(args.requests, args.small_traces, args.large_traces)
    cells = [cell for cell in all_cells if not args.cells or cell.slug in args.cells]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        (output_dir / "t23_4_summary.md").write_text(summary_markdown(output_dir, all_cells))
        return 0
    stub_path, cleanup_path = install_claude_stub()
    aws = {
        "AWS_ACCESS_KEY_ID": "minio-fake-access",
        "AWS_SECRET_ACCESS_KEY": "minio-fake-secret",
        "AWS_DEFAULT_REGION": "us-east-1",
        "MLFLOW_S3_ENDPOINT_URL": "http://127.0.0.1:59092",
    }
    previous_env = {key: os.environ.get(key) for key in aws}
    os.environ.update(aws)
    try:
        if "python" in args.targets and importlib.util.find_spec("boto3") is None:
            raise RuntimeError(
                "T23.4 trace archival requires MLflow's locked S3 dependencies; "
                "run uv with '--extra extras'."
            )
        _assert_hygiene()
        run_command(compose_args("up", "-d", "--wait", "postgres", "minio"))
        run_command(compose_args("run", "--rm", "minio-init"))
        if not args.skip_build:
            run_command(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "mlflow-server",
                    "-p",
                    "mlflow-genai-worker",
                ],
                cwd=RUST_ROOT,
            )
        with provider_server(args.seed, frame_gap_ms=FRAME_GAP_MS) as provider:
            provider_url = f"http://127.0.0.1:{provider.server_port}"
            for target in args.targets:
                recreate_database()
                with tempfile.TemporaryDirectory(prefix=f"mlflow-t23-4-{target}-") as temporary:
                    workdir = Path(temporary)
                    archive_location = (workdir / "trace-archive").as_uri()
                    config_path = workdir / "trace-archival.yaml"
                    write_archive_config(
                        config_path,
                        archive_location,
                        False,
                        max(args.small_traces, args.large_traces),
                    )
                    handle = launch_server(
                        target,
                        workdir,
                        provider_url,
                        f"t23-4/{target}-{args.seed}",
                        stub_path,
                        {
                            "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW": str(
                                DB_POOL_CONFIG["max_overflow"]
                            ),
                            "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE": str(DB_POOL_CONFIG["pool_size"]),
                            "MLFLOW_TRACE_ARCHIVAL_CONFIG": str(config_path),
                            "OPENAI_API_BASE": f"{provider_url}/v1",
                            "OPENAI_API_KEY": FAKE_API_KEY,
                        },
                    )
                    try:
                        setup = setup_target(handle.url, provider_url, args.seed)
                        target_archive_ids = []
                        for cell in cells:
                            path = output_dir / f"{cell.family}-{cell.slug}-{target}.json"
                            print(
                                f"[{target}] {cell.family}/{cell.slug}: {cell.count:,}",
                                flush=True,
                            )
                            if cell.kind == "archive-pass":
                                value, trace_ids = run_archive_pass(
                                    handle,
                                    target,
                                    cell,
                                    config_path,
                                    archive_location,
                                    provider_url,
                                    path,
                                    args.seed,
                                    args.archive_timeout_seconds,
                                )
                                target_archive_ids.extend(trace_ids)
                            elif cell.family == "archival":
                                if not target_archive_ids:
                                    raise RuntimeError(
                                        "archived-read cells require a selected archive-pass cell"
                                    )
                                value = run_archive_read(
                                    handle,
                                    target,
                                    cell,
                                    target_archive_ids,
                                    provider,
                                    provider_url,
                                    path,
                                    args.seed,
                                )
                            else:
                                value = run_request_cell(
                                    handle,
                                    target,
                                    cell,
                                    setup,
                                    provider,
                                    provider_url,
                                    path,
                                    args.seed,
                                )
                            _validate_cell(value, cell)
                            python_path = output_dir / f"{cell.family}-{cell.slug}-python.json"
                            if target == "rust" and python_path.exists():
                                differences = compare_runs(
                                    json.loads(python_path.read_text()), value
                                )
                                verdict = "FAIL" if differences else "PASS"
                                mark_verdict(python_path, verdict)
                                mark_verdict(path, verdict)
                                if differences:
                                    raise RuntimeError(
                                        f"equivalence failed for {cell.family}/{cell.slug}:\n"
                                        + "\n".join(differences)
                                    )
                    finally:
                        stop_server(handle)
            if set(args.targets) == {"python", "rust"}:
                (output_dir / "t23_4_summary.md").write_text(summary_markdown(output_dir, cells))
        for path in output_dir.glob("*.json"):
            validate_raw_metrics(json.loads(path.read_text()))
        return 0
    finally:
        run_command(compose_args("down", "-v", "--remove-orphans"), check=False)
        shutil.rmtree(cleanup_path, ignore_errors=True)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "t23_4")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--requests", type=int, default=CANONICAL_REQUESTS)
    parser.add_argument("--small-traces", type=int, default=CANONICAL_SMALL_TRACES)
    parser.add_argument("--large-traces", type=int, default=CANONICAL_LARGE_TRACES)
    parser.add_argument("--archive-timeout-seconds", type=float, default=3_600)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--cells", nargs="+", default=[])
    parser.add_argument(
        "--targets", nargs="+", choices=("python", "rust"), default=["python", "rust"]
    )
    parser.set_defaults(func=matrix)
