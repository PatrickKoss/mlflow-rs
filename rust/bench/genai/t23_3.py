"""T23.3 async jobs and native-engine benchmark matrix."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.util
import json
import math
import random
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from jsonschema.exceptions import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mlflow import MlflowClient
from mlflow.genai.scorers.builtin_scorers import Completeness, ConversationCompleteness
from mlflow.store.tracking.dbmodels.models import SqlJob
from rust.bench.genai.equivalence import compare_runs
from rust.bench.genai.metrics import (
    AsyncBenchClient,
    MetricsCollector,
    ResourceMonitor,
    percentile,
)
from rust.bench.genai.mock_provider import provider_server
from rust.bench.genai.runner import (
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
    validate_raw_metrics,
    write_raw_metrics,
)
from rust.bench.genai.t23_2 import machine_state

SCHEMA_VERSION = "1.2.0"
SEED = 2330
CANONICAL_FANOUT_JOBS = 1_000
CANONICAL_LARGE_JOBS = 10
CANONICAL_BURST_JOBS = 100
CANONICAL_DRIP_JOBS = 20
CANONICAL_LARGE_ROWS = 1_000
POLL_INTERVAL_SECONDS = 0.20
SETTLE_SECONDS = 5.0
LEAK_SETTLE_TIMEOUT_SECONDS = 60.0
JOB_TIMEOUT_SECONDS = 3_600.0
DB_POOL_CONFIG = {"max_overflow": 8, "pool_size": 32, "postgres_max_connections": 400}

EVALUATE = "invoke_genai_evaluate"
SCORER = "invoke_scorer"
ONLINE_TRACE = "run_online_trace_scorer"
ONLINE_SESSION = "run_online_session_scorer"
ISSUES = "invoke_issue_detection"
OPTIMIZE = "optimize_prompts"
JOB_KINDS = (EVALUATE, SCORER, ONLINE_TRACE, ONLINE_SESSION, ISSUES, OPTIMIZE)
DIRECT_KINDS = (EVALUATE, SCORER, ISSUES, OPTIMIZE)


@dataclass(frozen=True)
class Cell:
    slug: str
    shape: str
    kinds: tuple[str, ...]
    jobs_by_kind: dict[str, int]
    canonical_by_kind: dict[str, int]
    rows: int
    concurrency: int
    drip_seconds: float = 0.0

    @property
    def jobs(self) -> int:
        return sum(self.jobs_by_kind.values())

    @property
    def canonical_jobs(self) -> int:
        return sum(self.canonical_by_kind.values())


@dataclass
class SubmittedJob:
    job_id: str
    job_kind: str
    sequence: int
    submitted: float


def cell_matrix(
    fanout_jobs: int,
    large_jobs: int,
    burst_jobs: int,
    drip_jobs: int,
    large_rows: int,
    issue_large_rows: int | None = None,
) -> list[Cell]:
    issue_large_rows = issue_large_rows or large_rows
    cells = []
    for kind, label in (
        (EVALUATE, "evaluation"),
        (SCORER, "scorer"),
        (ISSUES, "issue-discovery"),
        (OPTIMIZE, "prompt-optimization"),
    ):
        cells.extend([
            Cell(
                f"{label}-high-fanout",
                "high-fanout",
                (kind,),
                {kind: fanout_jobs},
                {kind: CANONICAL_FANOUT_JOBS},
                1,
                3,
            ),
            Cell(
                f"{label}-large-payload",
                "large-payload",
                (kind,),
                {kind: large_jobs},
                {kind: CANONICAL_LARGE_JOBS},
                issue_large_rows if kind == ISSUES else large_rows,
                3,
            ),
        ])
    cells.extend([
        Cell(
            "online-high-fanout",
            "high-fanout",
            (ONLINE_TRACE, ONLINE_SESSION),
            {ONLINE_TRACE: fanout_jobs, ONLINE_SESSION: fanout_jobs},
            {
                ONLINE_TRACE: CANONICAL_FANOUT_JOBS,
                ONLINE_SESSION: CANONICAL_FANOUT_JOBS,
            },
            1,
            3,
        ),
        Cell(
            "online-large-payload",
            "large-payload",
            (ONLINE_TRACE, ONLINE_SESSION),
            {ONLINE_TRACE: large_jobs, ONLINE_SESSION: large_jobs},
            {ONLINE_TRACE: CANONICAL_LARGE_JOBS, ONLINE_SESSION: CANONICAL_LARGE_JOBS},
            large_rows,
            3,
        ),
        Cell(
            "mixed-burst",
            "burst",
            JOB_KINDS,
            dict.fromkeys(JOB_KINDS, burst_jobs),
            dict.fromkeys(JOB_KINDS, CANONICAL_BURST_JOBS),
            1,
            3,
        ),
        Cell(
            "mixed-steady-drip",
            "steady-drip",
            JOB_KINDS,
            {
                **dict.fromkeys(DIRECT_KINDS, drip_jobs),
                ONLINE_TRACE: 2,
                ONLINE_SESSION: 2,
            },
            {
                **dict.fromkeys(DIRECT_KINDS, CANONICAL_DRIP_JOBS),
                ONLINE_TRACE: 2,
                ONLINE_SESSION: 2,
            },
            1,
            3,
            65.0,
        ),
    ])
    return cells


def _fixed_hex(seed: int, *parts: object, length: int) -> str:
    return hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).hexdigest()[:length]


def _parallel(items: list[Any], function: Callable[[Any], Any], workers: int = 64) -> list[Any]:
    if not items:
        return []
    results: list[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        futures = {executor.submit(function, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _request_json(base_url: str, method: str, path: str, body: Any | None = None) -> Any:
    response = requests.request(method, base_url + path, json=body, timeout=120)
    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"setup {method} {path}: HTTP {response.status_code}: {response.text[:500]}"
        )
    return response.json() if response.content else {}


def _create_experiment(base_url: str, name: str) -> str:
    return str(
        _request_json(base_url, "POST", "/api/2.0/mlflow/experiments/create", {"name": name})[
            "experiment_id"
        ]
    )


def _span(trace_hex: str, index: int, *, session_id: str | None = None) -> dict[str, Any]:
    attributes = [
        {
            "key": "mlflow.spanInputs",
            "value": {"stringValue": json.dumps({"question": f"seeded question {index}"})},
        },
        {
            "key": "mlflow.spanOutputs",
            "value": {"stringValue": json.dumps(f"seeded answer {index}")},
        },
        {"key": "mlflow.spanType", "value": {"stringValue": "CHAIN"}},
    ]
    if session_id is not None:
        attributes.append({"key": "session.id", "value": {"stringValue": session_id}})
    now_ns = 1_750_000_000_000_000_000 + index * 2_000_000
    return {
        "attributes": attributes,
        "endTimeUnixNano": str(now_ns + 1_000_000),
        "name": f"t23-3-root-{index}",
        "spanId": trace_hex[:16],
        "startTimeUnixNano": str(now_ns),
        "status": {"code": 1},
        "traceId": trace_hex,
    }


def create_traces(
    base_url: str,
    experiment_id: str,
    seed: int,
    label: str,
    count: int,
    *,
    session_size: int | None = None,
) -> list[str]:
    trace_hexes = [_fixed_hex(seed, label, index, length=32) for index in range(count)]
    if session_size is not None:

        def start(index: int) -> None:
            trace_id = f"tr-{trace_hexes[index]}"
            session_id = f"session-{label}-{index // session_size:06d}"
            _request_json(
                base_url,
                "POST",
                "/api/3.0/mlflow/traces",
                {
                    "trace": {
                        "trace_info": {
                            "trace_id": trace_id,
                            "trace_location": {
                                "type": "MLFLOW_EXPERIMENT",
                                "mlflow_experiment": {"experiment_id": experiment_id},
                            },
                            "request_time": "2025-06-15T15:06:40Z",
                            "execution_duration": "0.001s",
                            "state": "OK",
                            "request_preview": json.dumps({"question": f"seeded question {index}"}),
                            "response_preview": json.dumps(f"seeded answer {index}"),
                            "trace_metadata": {"mlflow.trace.session": session_id},
                            "tags": {"phase": "23.3"},
                        }
                    }
                },
            )

        _parallel(list(range(count)), start, workers=128)
    for offset in range(0, count, 500):
        spans = [
            _span(
                trace_hexes[index],
                index,
                session_id=(
                    f"session-{label}-{index // session_size:06d}" if session_size else None
                ),
            )
            for index in range(offset, min(offset + 500, count))
        ]
        response = requests.post(
            base_url + "/v1/traces",
            headers={
                "content-type": "application/json",
                "x-mlflow-experiment-id": experiment_id,
            },
            json={
                "resourceSpans": [
                    {
                        "resource": {"attributes": []},
                        "scopeSpans": [{"scope": {"name": "t23-3"}, "spans": spans}],
                    }
                ]
            },
            timeout=120,
        )
        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                f"OTLP trace setup failed: {response.status_code}: {response.text[:500]}"
            )
    return [f"tr-{value}" for value in trace_hexes]


def _create_dataset(base_url: str, experiment_id: str, seed: int, label: str, rows: int) -> str:
    dataset = _request_json(
        base_url,
        "POST",
        "/api/3.0/mlflow/datasets/create",
        {
            "created_by": "t23-bench",
            "experiment_ids": [experiment_id],
            "name": f"t23-3-{seed}-{label}",
            "source_type": "HUMAN",
            "tags": json.dumps({"phase": "23.3", "rows": rows}),
        },
    )["dataset"]
    records = [
        {
            "inputs": {"question": f"seeded question {index}"},
            "outputs": f"seeded answer {index}",
            "expectations": {"expected_response": f"seeded answer {index}"},
            "tags": {"row": str(index)},
        }
        for index in range(rows)
    ]
    _request_json(
        base_url,
        "POST",
        f"/api/3.0/mlflow/datasets/{dataset['dataset_id']}/records",
        {"records": json.dumps(records, sort_keys=True), "updated_by": "t23-bench"},
    )
    return str(dataset["dataset_id"])


def _create_gateway(base_url: str, provider_url: str, seed: int) -> dict[str, str]:
    secret = _request_json(
        base_url,
        "POST",
        "/api/3.0/mlflow/gateway/secrets/create",
        {
            "auth_config": {"api_base": f"{provider_url}/v1"},
            "created_by": "t23-bench",
            "provider": "openai",
            "secret_name": f"t23-3-fake-secret-{seed}",
            "secret_value": {"api_key": FAKE_API_KEY},
        },
    )["secret"]
    model = _request_json(
        base_url,
        "POST",
        "/api/3.0/mlflow/gateway/model-definitions/create",
        {
            "created_by": "t23-bench",
            "model_name": "genai-bench-model",
            "name": f"t23-3-fake-model-{seed}",
            "provider": "openai",
            "secret_id": secret["secret_id"],
        },
    )["model_definition"]

    def endpoint(name: str) -> None:
        _request_json(
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/endpoints/create",
            {
                "created_by": "t23-bench",
                "model_configs": [
                    {
                        "linkage_type": "PRIMARY",
                        "model_definition_id": model["model_definition_id"],
                        "weight": 1.0,
                    }
                ],
                "name": name,
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "usage_tracking": False,
            },
        )

    judge_endpoint = f"t23-3-judge-{seed}"
    endpoint(judge_endpoint)
    return {"judge": judge_endpoint}


def _instructions_scorer(endpoint_name: str) -> str:
    value = json.loads(
        (
            RUST_ROOT
            / "crates"
            / "mlflow-genai"
            / "tests"
            / "fixtures"
            / "instructions_judge_scorer.json"
        ).read_text()
    )
    value["instructions_judge_pydantic_data"]["model"] = f"gateway:/{endpoint_name}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def setup_target(
    base_url: str,
    provider_url: str,
    seed: int,
    large_rows: int,
    issue_rows: int,
    large_jobs: int,
) -> dict[str, Any]:
    experiments = {
        kind: _create_experiment(base_url, f"t23-3-{seed}-{kind}") for kind in DIRECT_KINDS
    }
    endpoints = _create_gateway(base_url, provider_url, seed)
    trace_ids = create_traces(base_url, experiments[EVALUATE], seed, "direct-large", large_rows)
    # All three trace-consuming direct kinds use the same deterministic IDs;
    # log the identical corpus into each experiment container.
    create_traces(base_url, experiments[SCORER], seed, "direct-large", large_rows)
    issue_trace_ids = create_traces(
        base_url,
        experiments[ISSUES],
        seed,
        "direct-issues-disjoint",
        max(large_rows, issue_rows * large_jobs),
    )
    small_dataset = _create_dataset(base_url, experiments[OPTIMIZE], seed, "small", 1)
    large_dataset = _create_dataset(base_url, experiments[OPTIMIZE], seed, "large", large_rows)
    registry = MlflowClient(tracking_uri=base_url, registry_uri=base_url)
    prompt = registry.register_prompt(
        name=f"t23_3_prompt_{seed}",
        template="Answer this question concisely: {{question}}",
        model_config={"provider": "openai", "model_name": "genai-bench-model"},
        tags={"phase": "23.3"},
    )
    return {
        "datasets": {"small": small_dataset, "large": large_dataset},
        "endpoints": endpoints,
        "experiments": experiments,
        "judge_scorer": _instructions_scorer(endpoints["judge"]),
        "prompt_uri": f"prompts:/{prompt.name}/{prompt.version}",
        "issue_trace_ids": issue_trace_ids,
        "trace_ids": trace_ids,
    }


def _serialized_builtin(scorer: Any) -> str:
    return json.dumps(scorer.model_dump(), sort_keys=True, separators=(",", ":"))


def prepare_online(
    base_url: str,
    endpoint_name: str,
    seed: int,
    label: str,
    experiments: int,
    rows: int,
) -> list[dict[str, str]]:
    def create(index: int) -> dict[str, str]:
        experiment_id = _create_experiment(base_url, f"t23-3-{seed}-{label}-{index:05d}")
        trace_name = f"online-trace-{label}-{index}"
        session_name = f"online-session-{label}-{index}"
        trace_scorer = _serialized_builtin(
            Completeness(name=trace_name, model=f"gateway:/{endpoint_name}")
        )
        session_scorer = _serialized_builtin(
            ConversationCompleteness(name=session_name, model=f"gateway:/{endpoint_name}")
        )
        for name, serialized in ((trace_name, trace_scorer), (session_name, session_scorer)):
            _request_json(
                base_url,
                "POST",
                "/api/3.0/mlflow/scorers/register",
                {"experiment_id": experiment_id, "name": name, "serialized_scorer": serialized},
            )
            _request_json(
                base_url,
                "PUT",
                "/api/3.0/mlflow/scorers/online-config",
                {"experiment_id": experiment_id, "name": name, "sample_rate": 0.0},
            )
        return {
            "experiment_id": experiment_id,
            "session_name": session_name,
            "trace_name": trace_name,
        }

    configs = _parallel(list(range(experiments)), create, workers=64)

    def traces(item: tuple[int, dict[str, str]]) -> None:
        index, config = item
        create_traces(
            base_url,
            config["experiment_id"],
            seed,
            f"{label}-{index:05d}",
            rows,
            session_size=max(1, min(10, rows)),
        )

    _parallel(list(enumerate(configs)), traces, workers=min(32, experiments))
    return configs


def set_online_rate(base_url: str, configs: list[dict[str, str]], rate: float) -> None:
    def update(item: dict[str, str]) -> None:
        for name in (item["trace_name"], item["session_name"]):
            _request_json(
                base_url,
                "PUT",
                "/api/3.0/mlflow/scorers/online-config",
                {"experiment_id": item["experiment_id"], "name": name, "sample_rate": rate},
            )

    _parallel(configs, update, workers=128)


def _wait_before_minute(seconds_before: float = 3.0) -> None:
    now = time.time()
    delay = 60.0 - (now % 60.0) - seconds_before
    if delay < 0:
        delay += 60.0
    if delay > 0:
        time.sleep(delay)


def discover_online_jobs(
    configs: list[dict[str, str]],
    expected_each: int,
    after_ms: int,
    submitted_perf: float,
    timeout: float,
) -> list[SubmittedJob]:
    experiment_ids = {item["experiment_id"] for item in configs}
    deadline = time.monotonic() + timeout
    engine = create_engine(DB_URI)
    try:
        while time.monotonic() < deadline:
            found: dict[str, list[SqlJob]] = {ONLINE_TRACE: [], ONLINE_SESSION: []}
            with Session(engine) as session:
                rows = session.scalars(
                    select(SqlJob)
                    .where(SqlJob.creation_time >= after_ms)
                    .where(SqlJob.job_name.in_((ONLINE_TRACE, ONLINE_SESSION)))
                    .order_by(SqlJob.creation_time, SqlJob.id)
                ).all()
            for row in rows:
                try:
                    experiment_id = str(json.loads(row.params)["experiment_id"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                if experiment_id in experiment_ids:
                    found[row.job_name].append(row)
            if all(len(found[kind]) >= expected_each for kind in found):
                result = []
                sequence = 0
                for kind in (ONLINE_TRACE, ONLINE_SESSION):
                    for row in found[kind][:expected_each]:
                        result.append(SubmittedJob(row.id, kind, sequence, submitted_perf))
                        sequence += 1
                return result
            time.sleep(0.25)
        raise TimeoutError(
            f"online scheduler did not expose {expected_each} jobs per kind after {after_ms}"
        )
    finally:
        engine.dispose()


def _submission_payload(
    kind: str, setup: dict[str, Any], cell: Cell, index: int
) -> tuple[str, Any]:
    source_trace_ids = setup["issue_trace_ids"] if kind == ISSUES else setup["trace_ids"]
    if cell.shape == "large-payload":
        if kind == ISSUES:
            start = index * cell.rows
            trace_ids = source_trace_ids[start : start + cell.rows]
        else:
            trace_ids = source_trace_ids[: cell.rows]
            rotation = index % len(trace_ids)
            trace_ids = trace_ids[rotation:] + trace_ids[:rotation]
    else:
        trace_ids = [source_trace_ids[index % len(source_trace_ids)]]
    experiment_id = setup["experiments"].get(kind, setup["experiments"][EVALUATE])
    if kind == EVALUATE:
        return "/ajax-api/3.0/mlflow/genai/evaluate/invoke", {
            "experiment_id": experiment_id,
            "trace_ids": trace_ids,
            "serialized_scorers": [setup["judge_scorer"]],
        }
    if kind == SCORER:
        return "/ajax-api/3.0/mlflow/scorer/invoke", {
            "experiment_id": experiment_id,
            "log_assessments": False,
            "serialized_scorer": setup["judge_scorer"],
            "trace_ids": trace_ids,
        }
    if kind == ISSUES:
        return "/ajax-api/3.0/mlflow/issues/invoke", {
            "categories": ["quality"],
            "endpoint_name": setup["endpoints"]["judge"],
            "experiment_id": experiment_id,
            "provider": "openai",
            "trace_ids": trace_ids,
        }
    if kind == OPTIMIZE:
        dataset = setup["datasets"]["large" if cell.shape == "large-payload" else "small"]
        return "/api/3.0/mlflow/prompt-optimization/jobs", {
            "config": {
                "dataset_id": dataset,
                "optimizer_config_json": json.dumps(
                    {
                        "reflection_model": "openai:/genai-bench-model",
                        "guidelines": "Prefer concise deterministic answers.",
                    },
                    sort_keys=True,
                ),
                "optimizer_type": "OPTIMIZER_TYPE_METAPROMPT",
                "scorers": ["Correctness"],
            },
            "experiment_id": experiment_id,
            "source_prompt_uri": setup["prompt_uri"],
            "tags": [{"key": "phase", "value": "23.3"}],
        }
    raise AssertionError(f"unsupported direct job kind: {kind}")


def _ordered_direct_specs(
    cell: Cell, specs: list[tuple[str, int, float]]
) -> list[tuple[str, int, float]]:
    if cell.shape == "steady-drip":
        return sorted(specs, key=lambda spec: (spec[2], JOB_KINDS.index(spec[0]), spec[1]))
    if len(cell.kinds) > 1:
        return [spec for spec in specs if spec[0] == OPTIMIZE] + [
            spec for spec in specs if spec[0] != OPTIMIZE
        ]
    return specs


async def _poll_job(
    client: AsyncBenchClient,
    job: SubmittedJob,
    cell_origin: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = job.submitted + timeout_seconds
    first_running = None
    polls = 0
    while time.perf_counter() < deadline:
        status_code, body, _ = await client.request(
            "job_poll",
            "GET",
            f"/ajax-api/3.0/mlflow/jobs/{job.job_id}",
            measured=False,
        )
        observed = time.perf_counter()
        polls += 1
        if status_code == 200 and isinstance(body, dict):
            status = str(body.get("status", "")).upper()
            if status == "RUNNING" and first_running is None:
                first_running = observed
            if status in {"SUCCEEDED", "FAILED", "CANCELED", "TIMEOUT"}:
                queue_end = first_running or observed
                return {
                    "api_creation_time_ms": body.get("creation_time"),
                    "api_last_update_time_ms": body.get("last_update_time"),
                    "execution_seconds": max(0.0, observed - queue_end),
                    "job_id": job.job_id,
                    "job_kind": job.job_kind,
                    "polls": polls,
                    "queue_wait_seconds": max(0.0, queue_end - job.submitted),
                    "result": body.get("result"),
                    "sequence": job.sequence,
                    "status": status,
                    "status_details": body.get("status_details"),
                    "submit_to_terminal_seconds": max(0.0, observed - job.submitted),
                    "submitted_offset_seconds": max(0.0, job.submitted - cell_origin),
                    "terminal_offset_seconds": max(0.0, observed - cell_origin),
                    "timed_out": False,
                }
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    observed = time.perf_counter()
    return {
        "api_creation_time_ms": None,
        "api_last_update_time_ms": None,
        "execution_seconds": 0.0,
        "job_id": job.job_id,
        "job_kind": job.job_kind,
        "polls": max(1, polls),
        "queue_wait_seconds": max(0.0, observed - job.submitted),
        "result": None,
        "sequence": job.sequence,
        "status": "TIMEOUT",
        "status_details": f"deterministic {timeout_seconds:.0f}s client timeout",
        "submit_to_terminal_seconds": max(0.0, observed - job.submitted),
        "submitted_offset_seconds": max(0.0, job.submitted - cell_origin),
        "terminal_offset_seconds": max(0.0, observed - cell_origin),
        "timed_out": True,
    }


async def submit_direct_jobs(
    base_url: str,
    setup: dict[str, Any],
    cell: Cell,
    collector: MetricsCollector,
    origin: float,
    seed: int,
    timeout_seconds: float,
) -> tuple[list[SubmittedJob], list[dict[str, Any]]]:
    specs: list[tuple[str, int, float]] = []
    rng = random.Random(f"{seed}:{cell.slug}")
    for kind in DIRECT_KINDS:
        count = cell.jobs_by_kind.get(kind, 0)
        # One scorer invocation over 1,000 traces is publicly split into ten
        # 100-trace worker jobs by the established API batch size.
        requests_for_kind = (
            1 if count and kind == SCORER and cell.shape == "large-payload" else count
        )
        for index in range(requests_for_kind):
            scheduled = 0.0
            if cell.shape == "steady-drip":
                scheduled = (index + DIRECT_KINDS.index(kind) / len(DIRECT_KINDS)) * (
                    cell.drip_seconds / max(1, count)
                )
            specs.append((kind, index, scheduled))
    if cell.shape == "burst":
        rng.shuffle(specs)
    client = AsyncBenchClient(base_url, cell.concurrency, collector, timeout_seconds=1800)
    poll_client = AsyncBenchClient(base_url, 128, MetricsCollector(), timeout_seconds=60)
    semaphore = asyncio.Semaphore(cell.concurrency)
    prompt_submission = asyncio.Semaphore(1)
    issue_submission = asyncio.Semaphore(1)
    submitted: list[SubmittedJob] = []
    terminal_tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:

        async def one(kind: str, index: int, scheduled: float) -> None:
            delay = origin + scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            path, payload = _submission_payload(kind, setup, cell, index)

            async def submit() -> tuple[int | None, Any]:
                async with semaphore:
                    began = time.perf_counter()
                    status, body, _ = await client.request(
                        f"{kind}_submit", "POST", path, json=payload
                    )
                return status, body, began

            if kind == OPTIMIZE:
                async with prompt_submission:
                    status, body, began = await submit()
            elif kind == ISSUES:
                async with issue_submission:
                    status, body, began = await submit()
            else:
                status, body, began = await submit()
            if status != 200 or not isinstance(body, dict):
                return
            if kind == SCORER:
                job_ids = [job["job_id"] for job in body.get("jobs", [])]
            elif kind == OPTIMIZE:
                job_ids = [body.get("job", {}).get("job_id")]
            else:
                job_ids = [body.get("job_id")]
            if any(not job_id for job_id in job_ids):
                return
            base_sequence = JOB_KINDS.index(kind) * 1_000_000 + index * 100
            for batch_index, job_id in enumerate(job_ids):
                job = SubmittedJob(str(job_id), kind, base_sequence + batch_index, began)
                submitted.append(job)
                terminal_tasks.append(
                    asyncio.create_task(_poll_job(poll_client, job, origin, timeout_seconds))
                )

        if len(cell.kinds) > 1:
            for spec in _ordered_direct_specs(cell, specs):
                await one(*spec)
        else:
            await asyncio.gather(*(one(*spec) for spec in specs))
        terminal = await asyncio.gather(*terminal_tasks)
        return submitted, sorted(terminal, key=lambda job: (job["job_kind"], job["sequence"]))
    finally:
        await client.close()
        await poll_client.close()


async def poll_jobs(
    base_url: str,
    submissions: list[SubmittedJob],
    cell_origin: float,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    collector = MetricsCollector()
    client = AsyncBenchClient(base_url, 128, collector, timeout_seconds=60)
    try:
        terminal = await asyncio.gather(
            *(_poll_job(client, job, cell_origin, timeout_seconds) for job in submissions)
        )
        return sorted(terminal, key=lambda job: (job["job_kind"], job["sequence"]))
    finally:
        await client.close()


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def job_summary(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    if not jobs:
        return {
            "duration_seconds": 0.0,
            "error_rate": 1.0,
            "errors": 1,
            "fairness": None,
            "job_kinds": {},
            "jobs_per_minute": 0.0,
            "requests": 0,
            "rps": 0.0,
        }
    duration = max(job["terminal_offset_seconds"] for job in jobs) - min(
        job["submitted_offset_seconds"] for job in jobs
    )
    duration = max(duration, 1e-9)
    kinds = {}
    for kind in sorted({job["job_kind"] for job in jobs}):
        selected = [job for job in jobs if job["job_kind"] == kind]
        errors = sum(job["status"] != "SUCCEEDED" for job in selected)
        kinds[kind] = {
            "completed": len(selected),
            "error_rate": errors / len(selected),
            "errors": errors,
            "execution_seconds": _percentiles([job["execution_seconds"] for job in selected]),
            "jobs_per_minute": len(selected) * 60 / duration,
            "queue_wait_seconds": _percentiles([job["queue_wait_seconds"] for job in selected]),
            "wall_seconds": _percentiles([job["submit_to_terminal_seconds"] for job in selected]),
        }
    errors = sum(job["status"] != "SUCCEEDED" for job in jobs)
    fairness = None
    if len(kinds) > 1:
        half = max(1, len(jobs) // 2)
        first_half = sorted(jobs, key=lambda job: job["terminal_offset_seconds"])[:half]
        fairness = {
            "first_half_completion_share": {
                kind: sum(job["job_kind"] == kind for job in first_half) / half for kind in kinds
            },
            "max_to_min_p95_queue_ratio": _fairness_ratio(kinds),
        }
    return {
        "duration_seconds": duration,
        "error_rate": errors / len(jobs),
        "errors": errors,
        "fairness": fairness,
        "job_kinds": kinds,
        "jobs_per_minute": len(jobs) * 60 / duration,
        "requests": len(jobs),
        "rps": len(jobs) / duration,
    }


def _fairness_ratio(kinds: dict[str, Any]) -> float | None:
    values = [value["queue_wait_seconds"]["p95"] for value in kinds.values()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values) / max(1e-9, min(values))


def leak_check(samples: list[dict[str, Any]], completed: int) -> dict[str, Any]:
    applicable = completed >= 1_000 and len(samples) >= 4
    if not samples:
        return {"applicable": applicable, "completed_jobs": completed, "verdict": "FAIL"}
    width = min(3, max(1, len(samples) // 3))

    def mean(name: str, selected: list[dict[str, Any]]) -> float:
        return statistics.fmean(float(sample.get(name, 0)) for sample in selected)

    first = samples[:width]
    last = samples[-width:]
    start_rss = mean("rss_bytes", first)
    end_rss = mean("rss_bytes", last)
    elapsed_minutes = max(
        1e-9, (samples[-1]["elapsed_seconds"] - samples[0]["elapsed_seconds"]) / 60
    )
    rss_growth_mib = (end_rss - start_rss) / 1024 / 1024
    rss_values = [sample["rss_bytes"] for sample in samples]
    monotonic_rss = all(left <= right for left, right in zip(rss_values, rss_values[1:])) and (
        rss_values[-1] > rss_values[0]
    )
    process_values = [sample["process_count"] for sample in samples]
    thread_values = [sample["thread_count"] for sample in samples]

    def monotonic_growth(values: list[int]) -> bool:
        return all(left <= right for left, right in zip(values, values[1:])) and (
            values[-1] > values[0]
        )

    start_processes = mean("process_count", first)
    end_processes = mean("process_count", last)
    start_threads = mean("thread_count", first)
    end_threads = mean("thread_count", last)
    settled_rss_spread_mib = (max(rss_values[-width:]) - min(rss_values[-width:])) / 1024 / 1024
    settled_process_spread = max(process_values[-width:]) - min(process_values[-width:])
    settled_thread_spread = max(thread_values[-width:]) - min(thread_values[-width:])
    settled_thread_allowance = max(2, math.ceil(end_threads * 0.01))
    monotonic_processes = monotonic_growth(process_values)
    monotonic_threads = monotonic_growth(thread_values)
    flat = (
        not monotonic_rss
        and not monotonic_processes
        and not monotonic_threads
        and end_rss <= start_rss * 1.15 + 64 * 1024 * 1024
        and end_processes <= start_processes + 0.5
        and end_threads <= start_threads * 1.10 + 8
        and settled_rss_spread_mib <= 16
        and settled_process_spread <= 0
        and settled_thread_spread <= settled_thread_allowance
    )
    return {
        "applicable": applicable,
        "completed_jobs": completed,
        "end_process_count": end_processes,
        "end_rss_mib": end_rss / 1024 / 1024,
        "end_thread_count": end_threads,
        "monotonic_process_growth": monotonic_processes,
        "monotonic_rss_growth": monotonic_rss,
        "monotonic_thread_growth": monotonic_threads,
        "rss_growth_mib": rss_growth_mib,
        "rss_growth_mib_per_minute": rss_growth_mib / elapsed_minutes,
        "start_process_count": start_processes,
        "start_rss_mib": start_rss / 1024 / 1024,
        "start_thread_count": start_threads,
        "settled_process_spread": settled_process_spread,
        "settled_rss_spread_mib": settled_rss_spread_mib,
        "settled_thread_allowance": settled_thread_allowance,
        "settled_thread_spread": settled_thread_spread,
        "verdict": "PASS" if flat else "FAIL",
    }


def wait_for_leak_settle(monitor: ResourceMonitor, completed: int) -> float:
    started = time.monotonic()
    time.sleep(SETTLE_SECONDS)
    if completed < 1_000:
        return time.monotonic() - started
    deadline = started + LEAK_SETTLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if leak_check(list(monitor.samples), completed)["verdict"] == "PASS":
            break
        time.sleep(1.0)
    return time.monotonic() - started


def _sample_sequences(jobs: list[dict[str, Any]], seed: int) -> set[tuple[str, int]]:
    ranked = sorted(
        jobs,
        key=lambda job: hashlib.sha256(
            f"{seed}:{job['job_kind']}:{job['sequence']}".encode()
        ).digest(),
    )
    selected = {(job["job_kind"], job["sequence"]) for job in ranked[:16]}
    for kind in {job["job_kind"] for job in jobs}:
        first = next(job for job in jobs if job["job_kind"] == kind)
        selected.add((kind, first["sequence"]))
    return selected


def compact_jobs(
    jobs: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampled = _sample_sequences(jobs, seed)
    raw = []
    proof = []
    for job in jobs:
        key = (job["job_kind"], job["sequence"])
        result = job["result"]
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str).encode()
        raw_job = {
            **job,
            "result": (
                result
                if key in sampled
                else {
                    "body_bytes": len(encoded),
                    "body_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            ),
            "result_sampled": key in sampled,
        }
        raw.append(raw_job)
        proof.append({
            **job,
            "result": result if key in sampled else None,
            "result_sampled": key in sampled,
        })
    return raw, proof


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {"cpu_seconds": 0.0, "mean_rss_mib": 0.0, "peak_rss_mib": 0.0}
    cpu_first = samples[0]["utime_seconds"] + samples[0]["stime_seconds"]
    cpu_last = samples[-1]["utime_seconds"] + samples[-1]["stime_seconds"]
    rss = [sample["rss_bytes"] / 1024 / 1024 for sample in samples]
    return {
        "cpu_seconds": max(0.0, cpu_last - cpu_first),
        "mean_rss_mib": statistics.fmean(rss),
        "peak_rss_mib": max(rss),
    }


def run_cell(
    handle: Any,
    target: str,
    cell: Cell,
    setup: dict[str, Any],
    provider: Any,
    provider_url: str,
    output: Path,
    online_configs: list[dict[str, str]] | None,
    timeout_seconds: float,
    seed: int,
) -> dict[str, Any]:
    state = machine_state(handle)
    with provider.state.lock:
        provider_start = len(provider.state.observations)
    monitor = ResourceMonitor(handle.process.pid, postgres_sample)
    collector = MetricsCollector()
    started_at = dt.datetime.now(dt.timezone.utc)
    monitor.started = time.monotonic()
    monitor.start()
    time.sleep(2.0)
    origin_perf = time.perf_counter()
    submissions: list[SubmittedJob] = []
    online_executor: ThreadPoolExecutor | None = None
    online_future = None
    try:
        expected_online = cell.jobs_by_kind.get(ONLINE_TRACE, 0)
        if online_configs and expected_online:
            if cell.shape in {"high-fanout", "large-payload", "burst"}:
                _wait_before_minute()
            activation_ms = int(time.time() * 1000)
            activation_perf = time.perf_counter()
            set_online_rate(handle.url, online_configs, 1.0)

            def discover_poll_and_deactivate() -> tuple[
                list[SubmittedJob], list[dict[str, Any]]
            ]:
                try:
                    online_submissions = discover_online_jobs(
                        online_configs,
                        expected_online,
                        activation_ms,
                        activation_perf,
                        max(timeout_seconds, 180),
                    )
                finally:
                    set_online_rate(handle.url, online_configs, 0.0)
                return online_submissions, asyncio.run(
                    poll_jobs(handle.url, online_submissions, origin_perf, timeout_seconds)
                )

            online_executor = ThreadPoolExecutor(max_workers=1)
            online_future = online_executor.submit(discover_poll_and_deactivate)
        else:
            activation_ms = 0
            activation_perf = origin_perf
        direct_submissions, direct_jobs = asyncio.run(
            submit_direct_jobs(
                handle.url,
                setup,
                cell,
                collector,
                origin_perf,
                seed,
                timeout_seconds,
            )
        )
        submissions.extend(direct_submissions)
        online_submissions, online_jobs = (
            online_future.result() if online_future is not None else ([], [])
        )
        submissions.extend(online_submissions)
        jobs = sorted(
            direct_jobs + online_jobs, key=lambda job: (job["job_kind"], job["sequence"])
        )
        if len(submissions) != cell.jobs:
            raise RuntimeError(
                f"{target} {cell.slug} created {len(submissions)}/{cell.jobs} expected jobs; "
                "see submission request records"
            )
        collector.close()
        completed = sum(job["status"] == "SUCCEEDED" for job in jobs)
        settle_seconds = wait_for_leak_settle(monitor, completed)
    finally:
        if online_executor is not None:
            online_executor.shutdown(wait=True, cancel_futures=True)
        if online_configs:
            set_online_rate(handle.url, online_configs, 0.0)
        monitor.close()
    endpoints, request_overall = collector.summary()
    summary = job_summary(jobs)
    summary["latency_ms"] = request_overall["latency_ms"]
    resources = resource_summary(monitor.samples)
    summary["resources"] = resources
    leak = leak_check(monitor.samples, completed)
    raw_jobs, proof_jobs = compact_jobs(jobs, seed)
    with provider.state.lock:
        observations = provider.state.observations[provider_start:]
    trimmed = cell.jobs != cell.canonical_jobs or cell.rows != (
        CANONICAL_LARGE_ROWS if cell.shape == "large-payload" else 1
    )
    trim_note = None
    if trimmed:
        trim_note = (
            f"measured {cell.jobs} jobs and {cell.rows} rows; canonical design is "
            f"{cell.canonical_jobs} jobs and "
            f"{CANONICAL_LARGE_ROWS if cell.shape == 'large-payload' else 1} rows"
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "canonical_jobs": cell.canonical_jobs,
            "concurrency": cell.concurrency,
            "db_pool_config": DB_POOL_CONFIG,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_kinds": list(cell.kinds),
            "issue_disjoint_corpus": True,
            "measured_jobs": cell.jobs,
            "post_completion_settle_seconds": settle_seconds,
            "rows_per_job": cell.rows,
            "seed": seed,
            "shape": cell.shape,
            "concurrent_terminal_polling": True,
            "steady_drip_interleaved": cell.shape == "steady-drip",
            "started_at": started_at.isoformat(),
            "target": target,
            "trim_note": trim_note,
            "trimmed": trimmed,
            "warmup_requests": 0,
            "workload": f"t23_3/{cell.slug}",
        },
        "summary": {"endpoints": endpoints, "overall": summary},
        "requests": collector.raw_records(),
        "jobs": raw_jobs,
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
        "leak_check": leak,
        "equivalence": {
            "jobs": proof_jobs,
            "sample_seed": seed,
            "samples": [
                {
                    "endpoint": record["endpoint"],
                    "method": record["method"],
                    "response": record["response"],
                    "sequence": record["sequence"],
                    "status": record["status"],
                }
                for record in sorted(
                    collector.raw_records(), key=lambda item: (item["endpoint"], item["sequence"])
                )[:16]
            ],
            "verdict": "PENDING",
        },
    }
    write_raw_metrics(value, output)
    return value


def mark_verdict(path: Path, verdict: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    value["equivalence"]["verdict"] = verdict
    write_raw_metrics(value, path)
    return value


def _fmt(values: dict[str, Any]) -> str:
    return "/".join(
        "-" if values[key] is None else f"{values[key]:.2f}" for key in ("p50", "p95", "p99", "max")
    )


def _leak_label(leak: dict[str, Any]) -> str:
    return leak["verdict"] if leak["applicable"] else "N/A"


def summary_markdown(output_dir: Path, cells: list[Cell]) -> str:
    lines = [
        "# T23.3 jobs + native-engine benchmark summary",
        "",
        "This is raw material for T23.5, not the final Phase 23 report. Python and Rust ran",
        "serially on PostgreSQL 16 + MinIO with a fresh database and artifact prefix per",
        "target. Python used four uvicorn workers and its real Huey subprocess runtime; Rust",
        "used the release server and one native worker subprocess per claimed job. Every model",
        "call went through the loopback deterministic provider; no live provider was reachable.",
        "RSS, CPU, process, and thread samples cover the server's whole process tree at one-second",
        "intervals, including Python job-runtime children and Rust native-worker subprocesses.",
        "Online jobs were activated only through registered scorer + public online-config APIs",
        "and the real minute scheduler; a read-only jobs-table query discovered their IDs so the",
        "public GET jobs API could measure them. It did not create or mutate jobs.",
        "Each online config was deactivated immediately after the first expected scheduler wave",
        "was discovered. Public terminal polling started as soon as each job ID became available.",
        "Leak-applicable cells sampled a five-to-60-second post-completion tail until the whole",
        "process tree met the bounded flat-tail rule; reaching 60 seconds still failing was fatal.",
        "",
        "## Chosen matrix",
        "",
        "| Cell | Shape | Kinds | Jobs by kind | Rows/job | Rationale |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    rationale = {
        "high-fanout": "subprocess churn and leak pressure over a small corpus",
        "large-payload": "about ten jobs processing a 1,000-row corpus",
        "burst": "all pools receive much more work than worker concurrency",
        "steady-drip": "submission rate stays at or below the smallest pool capacity",
    }
    for cell in cells:
        counts = ", ".join(f"{kind}={count:,}" for kind, count in cell.jobs_by_kind.items())
        rows = f"{cell.rows:,}"
        lines.append(
            f"| `{cell.slug}` | {cell.shape} | {', '.join(cell.kinds)} | {counts} | {rows} | "
            f"{rationale[cell.shape]} |"
        )
    trims = [cell for cell in cells if cell.rows not in (1, CANONICAL_LARGE_ROWS)]
    if trims:
        lines.extend(["", "Measured volume trims:"])
        lines.extend(
            (
                f"- `{cell.slug}` retained {cell.jobs} jobs but used {cell.rows:,} rows/job "
                f"instead of the canonical {CANONICAL_LARGE_ROWS:,}."
            )
            for cell in trims
        )
    for kind in JOB_KINDS:
        lines.extend([
            "",
            f"## {kind}",
            "",
            "| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | "
            "Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | "
            "Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for cell in cells:
            if kind not in cell.kinds:
                continue
            values = {
                target: json.loads((output_dir / f"{cell.slug}-{target}.json").read_text())
                for target in ("python", "rust")
            }
            py = values["python"]
            rust = values["rust"]
            pyk = py["summary"]["overall"]["job_kinds"][kind]
            ruk = rust["summary"]["overall"]["job_kinds"][kind]
            pyr = py["summary"]["overall"]["resources"]
            rur = rust["summary"]["overall"]["resources"]
            lines.append(
                f"| `{cell.slug}` | {pyk['completed']:,} | "
                f"{pyk['jobs_per_minute']:.1f}/{ruk['jobs_per_minute']:.1f} | "
                f"{_fmt(pyk['wall_seconds'])} | {_fmt(ruk['wall_seconds'])} | "
                f"{pyk['queue_wait_seconds']['p95']:.2f}/{ruk['queue_wait_seconds']['p95']:.2f} | "
                f"{pyk['execution_seconds']['p95']:.2f}/{ruk['execution_seconds']['p95']:.2f} | "
                f"{pyr['peak_rss_mib']:.1f}/{rur['peak_rss_mib']:.1f} | "
                f"{pyr['cpu_seconds']:.2f}/{rur['cpu_seconds']:.2f} | "
                f"{pyk['errors']}/{ruk['errors']} | {rust['equivalence']['verdict']} | "
                f"{_leak_label(py['leak_check'])}/{_leak_label(rust['leak_check'])} |"
            )
    burst = next((cell for cell in cells if cell.shape == "burst"), None)
    if burst is not None:
        lines.extend(["", "## Burst queueing and fairness", ""])
        for target in ("python", "rust"):
            value = json.loads((output_dir / f"{burst.slug}-{target}.json").read_text())
            fairness = value["summary"]["overall"]["fairness"]
            lines.append(
                f"- {target}: max/min per-kind queue-p95 ratio "
                f"{fairness['max_to_min_p95_queue_ratio']:.2f}; first-half completion shares "
                f"{json.dumps(fairness['first_half_completion_share'], sort_keys=True)}."
            )
            for kind, metrics in value["summary"]["overall"]["job_kinds"].items():
                lines.append(
                    f"  - `{kind}` queue p95 {metrics['queue_wait_seconds']['p95']:.2f}s; "
                    f"execution p95 {metrics['execution_seconds']['p95']:.2f}s."
                )
    lines.extend(["", "## Leak checks", ""])
    applicable = False
    for cell in cells:
        for target in ("python", "rust"):
            value = json.loads((output_dir / f"{cell.slug}-{target}.json").read_text())
            leak = value["leak_check"]
            if not leak["applicable"]:
                continue
            applicable = True
            lines.append(
                f"- `{cell.slug}/{target}`: {leak['verdict']}; RSS "
                f"{leak['start_rss_mib']:.1f}->{leak['end_rss_mib']:.1f} MiB "
                f"({leak['rss_growth_mib_per_minute']:.2f} MiB/min), processes "
                f"{leak['start_process_count']:.1f}->{leak['end_process_count']:.1f}, threads "
                f"{leak['start_thread_count']:.1f}->{leak['end_thread_count']:.1f}; monotonic "
                f"RSS/process/thread growth "
                f"{leak['monotonic_rss_growth']}/{leak['monotonic_process_growth']}/"
                f"{leak['monotonic_thread_growth']}; settled RSS/process/thread spread "
                f"{leak['settled_rss_spread_mib']:.1f} MiB/{leak['settled_process_spread']}/"
                f"{leak['settled_thread_spread']} (thread allowance "
                f"{leak['settled_thread_allowance']})."
            )
    if not applicable:
        lines.append("No selected cell reached the 1,000-completion leak threshold.")
    lines.extend(["", "## Rust-slower cells and anomalies", ""])
    slower = []
    for cell in cells:
        py = json.loads((output_dir / f"{cell.slug}-python.json").read_text())
        rust = json.loads((output_dir / f"{cell.slug}-rust.json").read_text())
        for kind in cell.kinds:
            p95 = py["summary"]["overall"]["job_kinds"][kind]["wall_seconds"]["p95"]
            r95 = rust["summary"]["overall"]["job_kinds"][kind]["wall_seconds"]["p95"]
            if r95 > p95:
                slower.append((cell.slug, kind, p95, r95))
    if slower:
        for cell, kind, python_p95, rust_p95 in slower:
            lines.append(
                f"- `{cell}/{kind}`: Rust p95 {rust_p95:.2f}s vs Python {python_p95:.2f}s."
            )
    else:
        lines.append("No Rust p95 wall time exceeded Python in the measured matrix.")
    lines.extend(["", "## Raw result inventory", ""])
    lines.extend(f"- `{path.name}`" for path in sorted(output_dir.glob("*.json")))
    return "\n".join(lines) + "\n"


def _can_resume(path: Path, target: str, cell: Cell, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
        validate_raw_metrics(value)
        run = value["run"]
        expected_trimmed = cell.jobs != cell.canonical_jobs or cell.rows != (
            CANONICAL_LARGE_ROWS if cell.shape == "large-payload" else 1
        )
        if not (
            run["target"] == target
            and run["workload"] == f"t23_3/{cell.slug}"
            and run["seed"] == seed
            and run.get("concurrent_terminal_polling") is True
            and run["measured_jobs"] == cell.jobs
            and run["canonical_jobs"] == cell.canonical_jobs
            and run["shape"] == cell.shape
            and (cell.shape != "steady-drip" or run.get("steady_drip_interleaved") is True)
            and run["trimmed"] == expected_trimmed
            and run.get("rows_per_job", cell.rows) == cell.rows
        ):
            return False
        if ISSUES in cell.kinds and not run.get("issue_disjoint_corpus", False):
            return False
        jobs = value["jobs"]
        if len(jobs) != cell.jobs or any(job["status"] != "SUCCEEDED" for job in jobs):
            return False
        if value["summary"]["overall"]["errors"] != 0:
            return False
        completed = sum(job["status"] == "SUCCEEDED" for job in jobs)
        leak = leak_check(value["resources"]["samples"], completed)
        if leak != value["leak_check"]:
            value["leak_check"] = leak
            write_raw_metrics(value, path)
        if leak["applicable"] and leak["verdict"] != "PASS":
            return False
        return target == "python" or value["equivalence"]["verdict"] == "PASS"
    except (KeyError, TypeError, ValueError, ValidationError):
        return False


def matrix(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    all_cells = cell_matrix(
        args.fanout_jobs,
        args.large_jobs,
        args.burst_jobs,
        args.drip_jobs,
        args.large_rows,
        args.issue_large_rows,
    )
    cells = [cell for cell in all_cells if not args.cells or cell.slug in args.cells]
    output_dir.mkdir(parents=True, exist_ok=True)
    if "python" in args.targets and importlib.util.find_spec("litellm") is None:
        raise RuntimeError(
            "T23.3's real Python prompt runtime requires the locked litellm dependency; "
            "run uv with '--with litellm'."
        )
    stub_path, cleanup_path = install_claude_stub()
    try:
        machine_state(None)
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
        with provider_server(args.seed) as provider:
            provider_url = f"http://127.0.0.1:{provider.server_port}"
            for target in args.targets:
                recreate_database()
                with tempfile.TemporaryDirectory(prefix=f"mlflow-t23-3-{target}-") as temporary:
                    workdir = Path(temporary)
                    handle = launch_server(
                        target,
                        workdir,
                        provider_url,
                        f"t23-3/{target}-{time.time_ns()}",
                        stub_path,
                        {
                            "MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS": "0",
                            "MLFLOW_GATEWAY_URI": provider_url,
                            "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW": str(
                                DB_POOL_CONFIG["max_overflow"]
                            ),
                            "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE": str(DB_POOL_CONFIG["pool_size"]),
                            "OPENAI_API_BASE": f"{provider_url}/v1",
                            "OPENAI_API_KEY": FAKE_API_KEY,
                        },
                    )
                    try:
                        print(f"[{target}] setting up deterministic core corpora", flush=True)
                        setup = setup_target(
                            handle.url,
                            provider_url,
                            args.seed,
                            args.large_rows,
                            args.issue_large_rows,
                            args.large_jobs,
                        )
                        for cell in cells:
                            path = output_dir / f"{cell.slug}-{target}.json"
                            if args.resume and _can_resume(path, target, cell, args.seed):
                                print(
                                    f"[{target}] {cell.slug}: reusing validated result",
                                    flush=True,
                                )
                                continue
                            online_configs = None
                            online_count = cell.jobs_by_kind.get(ONLINE_TRACE, 0)
                            if online_count:
                                online_groups = (
                                    1 if cell.shape == "steady-drip" else online_count
                                )
                                print(
                                    f"[{target}] {cell.slug}: preparing {online_groups} online "
                                    f"experiment groups x {cell.rows} rows",
                                    flush=True,
                                )
                                online_configs = prepare_online(
                                    handle.url,
                                    setup["endpoints"]["judge"],
                                    args.seed,
                                    cell.slug,
                                    online_groups,
                                    cell.rows,
                                )
                            print(
                                f"[{target}] {cell.slug}: {cell.jobs} jobs ({cell.shape})",
                                flush=True,
                            )
                            value = run_cell(
                                handle,
                                target,
                                cell,
                                setup,
                                provider,
                                provider_url,
                                path,
                                online_configs,
                                args.timeout_seconds,
                                args.seed,
                            )
                            errors = value["summary"]["overall"]["errors"]
                            if value["summary"]["overall"]["error_rate"] >= 0.0001:
                                raise RuntimeError(
                                    f"{target} {cell.slug} had {errors}/{cell.jobs} job errors"
                                )
                            if (
                                value["leak_check"]["applicable"]
                                and value["leak_check"]["verdict"] != "PASS"
                            ):
                                raise RuntimeError(
                                    f"{target} {cell.slug} leak check failed: {value['leak_check']}"
                                )
                            python_path = output_dir / f"{cell.slug}-python.json"
                            if target == "rust" and python_path.exists():
                                python_value = json.loads(python_path.read_text())
                                differences = compare_runs(python_value, value)
                                verdict = "FAIL" if differences else "PASS"
                                mark_verdict(python_path, verdict)
                                mark_verdict(path, verdict)
                                if differences:
                                    raise RuntimeError(
                                        f"equivalence failed for {cell.slug}:\n"
                                        + "\n".join(differences)
                                    )
                            print(
                                f"[{target}] {cell.slug}: "
                                f"{value['summary']['overall']['jobs_per_minute']:.1f} jobs/min, "
                                f"{errors} errors, leak={value['leak_check']['verdict']}",
                                flush=True,
                            )
                    finally:
                        stop_server(handle)
            if set(args.targets) == {"python", "rust"}:
                (output_dir / "t23_3_summary.md").write_text(summary_markdown(output_dir, cells))
        for path in output_dir.glob("*.json"):
            validate_raw_metrics(json.loads(path.read_text()))
        return 0
    finally:
        run_command(compose_args("down", "-v", "--remove-orphans"), check=False)
        shutil.rmtree(cleanup_path, ignore_errors=True)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "t23_3")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fanout-jobs", type=int, default=CANONICAL_FANOUT_JOBS)
    parser.add_argument("--large-jobs", type=int, default=CANONICAL_LARGE_JOBS)
    parser.add_argument("--burst-jobs", type=int, default=CANONICAL_BURST_JOBS)
    parser.add_argument("--drip-jobs", type=int, default=CANONICAL_DRIP_JOBS)
    parser.add_argument("--large-rows", type=int, default=CANONICAL_LARGE_ROWS)
    parser.add_argument(
        "--issue-large-rows", type=int, default=CANONICAL_LARGE_ROWS
    )
    parser.add_argument("--timeout-seconds", type=float, default=JOB_TIMEOUT_SECONDS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cells", nargs="+", default=[])
    parser.add_argument(
        "--targets", nargs="+", choices=("python", "rust"), default=["python", "rust"]
    )
    parser.set_defaults(func=matrix)
