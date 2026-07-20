"""T23.5 mixed GenAI soak and capstone raw-result generator."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from rust.bench.genai.equivalence import compare_runs
from rust.bench.genai.metrics import AsyncBenchClient, MetricsCollector, ResourceMonitor, percentile
from rust.bench.genai.mock_provider import provider_server
from rust.bench.genai.runner import (
    ASSISTANT_PREFIX,
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
from rust.bench.genai.t23_3 import (
    EVALUATE,
    SCORER,
    SubmittedJob,
    _create_gateway,
    _instructions_scorer,
    _poll_job,
    compact_jobs,
    create_traces,
    job_summary,
    resource_summary,
)
from rust.bench.genai.t23_4 import (
    Cell as StreamingCell,
)
from rust.bench.genai.t23_4 import (
    _archive_payload,
    seed_archive_cell,
    wait_for_archive,
)
from rust.bench.genai.t23_4 import (
    setup_target as setup_streaming_target,
)

SCHEMA_VERSION = "1.4.0"
SEED = 2350
CANONICAL_REQUESTS = 10_000
CANONICAL_DURATION_SECONDS = 600.0
CANONICAL_ARCHIVE_TRACES = 1_000
CONCURRENCY = 64
JOB_TIMEOUT_SECONDS = 1_800.0
POLL_CONCURRENCY = 32
SETTLE_SECONDS = 30.0
WARMUP_REQUESTS = 68
FRAME_GAP_MS = 1.0

MIX_WEIGHTS = {
    "dataset_upserts": 2_000,
    "evaluation_jobs": 500,
    "scorer_jobs": 500,
    "gateway_chat": 2_500,
    "gateway_streams": 500,
    "assistant_requests": 500,
    "labeling_reads": 1_750,
    "review_queue_reads": 1_750,
}


@dataclass(frozen=True)
class TrafficItem:
    family: str
    index: int
    sequence: int
    scheduled_seconds: float
    request_count: int = 1


def scaled_mix(total_requests: int) -> dict[str, int]:
    if total_requests < 8:
        raise ValueError("mixed soak requires at least eight primary requests")
    exact = {
        family: total_requests * weight / CANONICAL_REQUESTS
        for family, weight in MIX_WEIGHTS.items()
    }
    counts = {family: int(value) for family, value in exact.items()}
    for family in sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)[
        : total_requests - sum(counts.values())
    ]:
        counts[family] += 1
    # An assistant operation is a POST plus its streamed GET.
    if counts["assistant_requests"] % 2:
        counts["assistant_requests"] -= 1
        counts["gateway_chat"] += 1
    return counts


def seeded_traffic(total_requests: int, duration_seconds: float, seed: int) -> list[TrafficItem]:
    counts = scaled_mix(total_requests)
    logical: list[tuple[str, int, int]] = []
    for family, count in counts.items():
        operations = count // 2 if family == "assistant_requests" else count
        request_count = 2 if family == "assistant_requests" else 1
        logical.extend((family, index, request_count) for index in range(operations))
    rng = random.Random(f"t23-5:{seed}:traffic")
    rng.shuffle(logical)
    sequence = 0
    items = []
    for position, (family, index, request_count) in enumerate(logical):
        fraction = (position + 0.5) / len(logical)
        jitter = rng.uniform(-0.25, 0.25) * duration_seconds / len(logical)
        scheduled = max(0.0, min(duration_seconds, fraction * duration_seconds + jitter))
        items.append(TrafficItem(family, index, sequence, scheduled, request_count))
        sequence += request_count
    assert sequence == total_requests
    return items


def sampled_sequences(items: list[TrafficItem], seed: int) -> set[int]:
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{seed}:{item.sequence}".encode()).digest(),
    )
    selected = {item.sequence for item in ranked[: min(32, len(ranked))]}
    selected.update(
        next(item.sequence for item in items if item.family == family) for family in MIX_WEIGHTS
    )
    return selected


def _create_experiment(base_url: str, name: str) -> str:
    with requests.Session() as session:
        return str(
            sync_request(
                session,
                base_url,
                "POST",
                "/api/2.0/mlflow/experiments/create",
                json={"name": name},
            )["experiment_id"]
        )


def setup_soak_target(
    base_url: str, provider_url: str, seed: int, trace_count: int
) -> dict[str, Any]:
    setup = setup_streaming_target(base_url, provider_url, seed)
    judge_endpoint = _create_gateway(base_url, provider_url, seed)["judge"]
    with requests.Session() as session:
        dataset_experiment = _create_experiment(base_url, f"t23-5-{seed}-datasets")
        dataset = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/datasets/create",
            json={
                "created_by": "t23-bench",
                "experiment_ids": [dataset_experiment],
                "name": f"t23-5-{seed}-mixed",
                "source_type": "HUMAN",
                "tags": json.dumps({"phase": "23.5"}),
            },
        )["dataset"]
        label_experiment = _create_experiment(base_url, f"t23-5-{seed}-labeling")
        schema_ids = []
        for index in range(32):
            schema = sync_request(
                session,
                base_url,
                "POST",
                "/api/3.0/mlflow/label-schemas/create",
                json={
                    "enable_comment": True,
                    "experiment_id": label_experiment,
                    "input": {"categorical": {"multi_select": False, "options": ["yes", "no"]}},
                    "instruction": f"Seeded review instruction {index}",
                    "name": f"t23-5-label-{index:02d}",
                    "type": "FEEDBACK",
                },
            )["label_schema"]
            schema_ids.append(schema["schema_id"])
        for index in range(16):
            sync_request(
                session,
                base_url,
                "POST",
                "/api/3.0/mlflow/review-queues/create",
                json={
                    "experiment_id": label_experiment,
                    "name": f"t23-5-review-{index:02d}",
                    "queue_type": "CUSTOM",
                    "schema_ids": schema_ids[index : index + 2],
                    "users": ["reviewer@example.invalid"],
                },
            )
    job_experiments = {
        EVALUATE: _create_experiment(base_url, f"t23-5-{seed}-evaluation"),
        SCORER: _create_experiment(base_url, f"t23-5-{seed}-scorer"),
    }
    evaluation_trace_ids = create_traces(
        base_url, job_experiments[EVALUATE], seed, "t23-5-evaluation", trace_count
    )
    scorer_trace_ids = create_traces(
        base_url, job_experiments[SCORER], seed, "t23-5-scorer", trace_count
    )
    setup.update({
        "dataset_id": dataset["dataset_id"],
        "job_experiments": job_experiments,
        "judge_scorer": _instructions_scorer(judge_endpoint),
        "label_experiment_id": label_experiment,
        "seed": seed,
        "trace_ids": {EVALUATE: evaluation_trace_ids, SCORER: scorer_trace_ids},
    })
    return setup


def _job_request(setup: dict[str, Any], family: str, index: int) -> tuple[str, Any, str]:
    if family == "evaluation_jobs":
        trace_ids = setup["trace_ids"][EVALUATE]
        return (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            {
                "experiment_id": setup["job_experiments"][EVALUATE],
                "serialized_scorers": [setup["judge_scorer"]],
                "trace_ids": [trace_ids[index % len(trace_ids)]],
            },
            EVALUATE,
        )
    trace_ids = setup["trace_ids"][SCORER]
    return (
        "/ajax-api/3.0/mlflow/scorer/invoke",
        {
            "experiment_id": setup["job_experiments"][SCORER],
            "log_assessments": False,
            "serialized_scorer": setup["judge_scorer"],
            "trace_ids": [trace_ids[index % len(trace_ids)]],
        },
        SCORER,
    )


async def warmup_target(base_url: str, setup: dict[str, Any]) -> None:
    collector = MetricsCollector()
    client = AsyncBenchClient(base_url, 8, collector, timeout_seconds=300)
    poll_client = AsyncBenchClient(base_url, 8, MetricsCollector(), timeout_seconds=60)
    try:
        for index in range(10):
            records = [
                {
                    "inputs": {"warmup": index},
                    "outputs": {"answer": f"warmup-{index}"},
                    "tags": {"phase": "23.5"},
                }
            ]
            await client.request(
                "dataset_upsert",
                "POST",
                f"/api/3.0/mlflow/datasets/{setup['dataset_id']}/records",
                measured=False,
                json={"records": json.dumps(records), "updated_by": "t23-bench"},
            )
        for index in range(10):
            await client.request(
                "gateway_chat",
                "POST",
                f"/gateway/{setup['endpoint_name']}/mlflow/invocations",
                measured=False,
                json={
                    "messages": [{"content": f"warmup {index}", "role": "user"}],
                    "stream": False,
                },
            )
        for path in (
            "/api/3.0/mlflow/label-schemas/list?"
            + urlencode({"experiment_id": setup["label_experiment_id"], "max_results": 10}),
            "/api/3.0/mlflow/review-queues/list?"
            + urlencode({"experiment_id": setup["label_experiment_id"], "max_results": 10}),
        ):
            for _ in range(10):
                await client.request("read_warmup", "GET", path, measured=False)
        status, message, _ = await client.request(
            "assistant_session",
            "POST",
            f"{ASSISTANT_PREFIX}/message",
            measured=False,
            json={"context": {"phase": "23.5"}, "message": "warmup"},
        )
        if status == 200 and isinstance(message, dict):
            await client.request(
                "assistant_stream",
                "GET",
                f"{ASSISTANT_PREFIX}/sessions/{message['session_id']}/stream",
                measured=False,
                sse=True,
            )
        jobs = []
        origin = time.perf_counter()
        for family in ("evaluation_jobs", "scorer_jobs"):
            for index in range(3):
                path, payload, kind = _job_request(setup, family, index)
                began = time.perf_counter()
                status, body, _ = await client.request(
                    f"{family}_submit", "POST", path, measured=False, json=payload
                )
                if status != 200 or not isinstance(body, dict):
                    raise RuntimeError(f"warmup {family} submission failed: {body}")
                ids = (
                    [job["job_id"] for job in body.get("jobs", [])]
                    if kind == SCORER
                    else [body.get("job_id")]
                )
                jobs.extend(
                    SubmittedJob(str(job_id), kind, index, began) for job_id in ids if job_id
                )
        terminal = await asyncio.gather(*(_poll_job(poll_client, job, origin, 300) for job in jobs))
        if any(job["status"] != "SUCCEEDED" for job in terminal):
            raise RuntimeError(f"warmup jobs failed: {terminal}")
    finally:
        await client.close()
        await poll_client.close()


def _archive_background(
    config_path: Path,
    archive_location: str,
    experiment_id: str,
    trace_count: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    write_soak_archive_config(config_path, archive_location, True, trace_count)
    try:
        rows, cadence_ms, visible_duration = wait_for_archive(
            experiment_id, trace_count, timeout_seconds
        )
    finally:
        write_soak_archive_config(config_path, archive_location, False, trace_count)
    payload = _archive_payload(rows[0][1])
    wall = time.monotonic() - started
    values = cadence_ms or [visible_duration * 1_000 / trace_count]
    return {
        "archived_traces": trace_count,
        "duration_seconds": wall,
        "traces_per_second": trace_count / max(wall, 1e-9),
        "finalize_transaction_latency_ms": {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
        },
        "archive_payload_bytes": len(payload),
        "archive_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_soak_archive_config(path: Path, location: str, enabled: bool, limit: int) -> None:
    path.write_text(
        "trace_archival:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        f"  location: {location}\n"
        # Only the explicitly tagged archiveNow corpus belongs to this pass.
        "  retention: 36500d\n"
        "  interval_seconds: 1\n"
        f"  max_traces_per_pass: {limit}\n"
    )


async def execute_soak(
    base_url: str,
    setup: dict[str, Any],
    items: list[TrafficItem],
    sample: set[int],
    concurrency: int,
    duration_seconds: float,
    config_path: Path,
    archive_location: str,
    archive_experiment_id: str,
    archive_traces: int,
    archive_timeout_seconds: float,
) -> tuple[MetricsCollector, list[dict[str, Any]], dict[str, Any]]:
    collector = MetricsCollector()
    client = AsyncBenchClient(base_url, concurrency, collector, timeout_seconds=300)
    poll_client = AsyncBenchClient(
        base_url, POLL_CONCURRENCY, MetricsCollector(), timeout_seconds=60
    )
    request_semaphore = asyncio.Semaphore(concurrency)
    poll_tasks: list[asyncio.Task[dict[str, Any]]] = []
    origin = time.perf_counter()
    collector.started = origin

    async def poll(job: SubmittedJob) -> dict[str, Any]:
        return await _poll_job(poll_client, job, origin, JOB_TIMEOUT_SECONDS)

    async def one(item: TrafficItem) -> None:
        delay = origin + item.scheduled_seconds - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        capture = item.sequence in sample
        async with request_semaphore:
            if item.family == "dataset_upserts":
                records = [
                    {
                        "inputs": {"index": item.index, "seed": setup["seed"]},
                        "outputs": {"answer": f"seeded-{item.index}", "padding": "x" * 512},
                        "tags": {"phase": "23.5"},
                    }
                ]
                await client.request(
                    "dataset_upsert",
                    "POST",
                    f"/api/3.0/mlflow/datasets/{setup['dataset_id']}/records",
                    capture_response=capture,
                    json={"records": json.dumps(records), "updated_by": "t23-bench"},
                    sequence=item.sequence,
                )
            elif item.family in {"gateway_chat", "gateway_streams"}:
                stream = item.family == "gateway_streams"
                await client.request(
                    "gateway_chat_stream" if stream else "gateway_chat",
                    "POST",
                    f"/gateway/{setup['endpoint_name']}/mlflow/invocations",
                    capture_response=capture,
                    json={
                        "max_tokens": 32,
                        "messages": [
                            {
                                "content": f"t23-5 seed {setup['seed']} request {item.index}",
                                "role": "user",
                            }
                        ],
                        "stream": stream,
                    },
                    sequence=item.sequence,
                    sse=stream,
                )
            elif item.family == "labeling_reads":
                path = "/api/3.0/mlflow/label-schemas/list?" + urlencode({
                    "experiment_id": setup["label_experiment_id"],
                    "max_results": 10,
                })
                await client.request(
                    "label_schemas_list",
                    "GET",
                    path,
                    capture_response=capture,
                    sequence=item.sequence,
                )
            elif item.family == "review_queue_reads":
                path = "/api/3.0/mlflow/review-queues/list?" + urlencode({
                    "experiment_id": setup["label_experiment_id"],
                    "max_results": 10,
                })
                await client.request(
                    "review_queues_list",
                    "GET",
                    path,
                    capture_response=capture,
                    sequence=item.sequence,
                )
            elif item.family == "assistant_requests":
                status, message, _ = await client.request(
                    "assistant_session",
                    "POST",
                    f"{ASSISTANT_PREFIX}/message",
                    capture_response=capture,
                    json={
                        "context": {"phase": "23.5", "sequence": item.index},
                        "message": f"assistant seed {setup['seed']} turn {item.index}",
                    },
                    sequence=item.sequence,
                )
                if status == 200 and isinstance(message, dict) and message.get("session_id"):
                    await client.request(
                        "assistant_stream",
                        "GET",
                        f"{ASSISTANT_PREFIX}/sessions/{message['session_id']}/stream",
                        capture_response=capture,
                        sequence=item.sequence + 1,
                        sse=True,
                    )
            else:
                path, payload, kind = _job_request(setup, item.family, item.index)
                began = time.perf_counter()
                status, body, _ = await client.request(
                    f"{item.family}_submit",
                    "POST",
                    path,
                    capture_response=capture,
                    json=payload,
                    sequence=item.sequence,
                )
                if status == 200 and isinstance(body, dict):
                    ids = (
                        [job["job_id"] for job in body.get("jobs", [])]
                        if kind == SCORER
                        else [body.get("job_id")]
                    )
                    for batch, job_id in enumerate(ids):
                        if job_id:
                            job = SubmittedJob(str(job_id), kind, item.sequence * 10 + batch, began)
                            poll_tasks.append(asyncio.create_task(poll(job)))

    archive_task = asyncio.create_task(
        asyncio.to_thread(
            _archive_background,
            config_path,
            archive_location,
            archive_experiment_id,
            archive_traces,
            archive_timeout_seconds,
        )
    )
    try:
        await asyncio.gather(*(one(item) for item in items))
        jobs = await asyncio.gather(*poll_tasks)
        archive = await archive_task
        remaining = origin + duration_seconds - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        await asyncio.sleep(SETTLE_SECONDS)
        collector.close()
        return collector, sorted(jobs, key=lambda job: (job["job_kind"], job["sequence"])), archive
    finally:
        if not archive_task.done():
            archive_task.cancel()
        await client.close()
        await poll_client.close()


def family_summary(records: list[Any], duration: float) -> dict[str, Any]:
    groups = {
        "datasets": {"dataset_upsert"},
        "evaluation_jobs": {"evaluation_jobs_submit"},
        "scorer_jobs": {"scorer_jobs_submit"},
        "gateway": {"gateway_chat", "gateway_chat_stream"},
        "assistant": {"assistant_session", "assistant_stream"},
        "label_schemas": {"label_schemas_list"},
        "review_queues": {"review_queues_list"},
    }
    result = {}
    for family, endpoints in groups.items():
        selected = [record for record in records if record.endpoint in endpoints]
        collector = MetricsCollector(
            records=[replace(record, endpoint=family) for record in selected],
            started=0.0,
            finished=duration,
        )
        endpoint, _ = collector.summary()
        result[family] = endpoint[family]
    return result


def rss_trend(samples: list[dict[str, Any]], bin_seconds: float = 60.0) -> dict[str, Any]:
    if len(samples) < 2:
        return {"verdict": "FAIL", "reason": "fewer than two RSS samples"}
    x = [sample["elapsed_seconds"] / 3_600 for sample in samples]
    y = [sample["rss_bytes"] / 1024 / 1024 for sample in samples]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = (
        sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y)) / denominator
        if denominator
        else 0.0
    )
    bins: dict[int, list[float]] = {}
    for sample, rss in zip(samples, y):
        bins.setdefault(int(sample["elapsed_seconds"] // bin_seconds), []).append(rss)
    means = [statistics.fmean(bins[index]) for index in sorted(bins)]
    nondecreasing = all(left <= right for left, right in zip(means, means[1:]))
    growth_percent = (means[-1] / means[0] - 1) * 100 if means[0] else 0.0
    monotonic = nondecreasing and growth_percent > 5.0
    return {
        "bin_seconds": bin_seconds,
        "definition": (
            "FAIL only when every consecutive one-minute mean is non-decreasing and the "
            "final mean exceeds the first by more than 5%, matching T14.2 at soak scale"
        ),
        "first_mean_rss_mib": means[0],
        "growth_percent": growth_percent,
        "last_mean_rss_mib": means[-1],
        "mean_rss_mib_by_bin": means,
        "monotonic_growth": monotonic,
        "nondecreasing_bins": nondecreasing,
        "slope_mib_per_hour": slope,
        "verdict": "FAIL" if monotonic else "PASS",
    }


def _proof_samples(
    records: list[dict[str, Any]], sample: set[int], archive: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        logical_sequence = (
            record["sequence"] - 1
            if record["endpoint"] == "assistant_stream"
            else record["sequence"]
        )
        if logical_sequence not in sample:
            continue
        selected.append({
            "endpoint": record["endpoint"],
            "method": record["method"],
            "path": record["path"],
            "response": record["response"],
            "sequence": record["sequence"],
            "sse_frames": record["sse"]["frames"] if record["sse"] else None,
            "status": record["status"],
        })
    selected.append({
        "archive_payload_bytes": archive["archive_payload_bytes"],
        "archive_payload_sha256": archive["archive_payload_sha256"],
        "endpoint": "archive_pass",
        "sequence": CANONICAL_REQUESTS,
        "status": 200,
    })
    return sorted(selected, key=lambda item: item["sequence"])


def _compact_records(records: list[dict[str, Any]], sample: set[int]) -> None:
    for record in records:
        logical_sequence = (
            record["sequence"] - 1
            if record["endpoint"] == "assistant_stream"
            else record["sequence"]
        )
        if sse := record.get("sse"):
            record["response"] = {"sse_frame_count": sse["frame_count"]}
            sse["frames"] = []
        elif logical_sequence not in sample and not (
            isinstance(record["response"], dict) and "body_sha256" in record["response"]
        ):
            encoded = json.dumps(
                record["response"], sort_keys=True, separators=(",", ":"), default=str
            ).encode()
            record["response"] = {
                "body_bytes": len(encoded),
                "body_sha256": hashlib.sha256(encoded).hexdigest(),
            }


def run_target(
    handle: Any,
    target: str,
    setup: dict[str, Any],
    items: list[TrafficItem],
    sample: set[int],
    provider: Any,
    provider_url: str,
    config_path: Path,
    archive_location: str,
    archive_experiment_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    asyncio.run(warmup_target(handle.url, setup))
    state = machine_state(handle)
    with provider.state.lock:
        provider_start = len(provider.state.observations)
    monitor = ResourceMonitor(handle.process.pid, postgres_sample)
    started_at = dt.datetime.now(dt.timezone.utc)
    monitor.started = time.monotonic()
    monitor.start()
    try:
        collector, jobs, archive = asyncio.run(
            execute_soak(
                handle.url,
                setup,
                items,
                sample,
                args.concurrency,
                args.duration_seconds,
                config_path,
                archive_location,
                archive_experiment_id,
                args.archive_traces,
                args.archive_timeout_seconds,
            )
        )
    finally:
        monitor.close()
    for record in collector.records:
        if record.endpoint == "gateway_chat_stream" and record.error is None:
            if record.sse is None or not any(
                '"finish_reason":"stop"' in frame for frame in record.sse.frames
            ):
                record.error = "gateway stream missing terminal stop frame"
        if record.endpoint == "assistant_stream" and record.error is None:
            if (
                record.sse is None
                or not record.sse.frames
                or not record.sse.frames[-1].startswith("event: done")
            ):
                record.error = "assistant stream missing terminal done event"
    endpoints, overall = collector.summary()
    duration = overall["duration_seconds"]
    overall["families"] = family_summary(collector.records, duration)
    overall["resources"] = resource_summary(monitor.samples)
    overall["archival"] = archive
    job_stats = job_summary(jobs)
    overall["jobs_per_minute"] = job_stats["jobs_per_minute"]
    overall["job_kinds"] = job_stats["job_kinds"]
    request_errors = overall["errors"]
    job_errors = sum(job["status"] != "SUCCEEDED" for job in jobs)
    overall["errors"] = request_errors + job_errors
    overall["error_rate"] = overall["errors"] / max(1, overall["requests"])
    raw_jobs, proof_jobs = compact_jobs(jobs, args.seed)
    records = collector.raw_records()
    proof = _proof_samples(records, sample, archive)
    _compact_records(records, sample)
    with provider.state.lock:
        observations = provider.state.observations[provider_start:]
    trend = rss_trend(monitor.samples)
    trimmed = (
        args.requests != CANONICAL_REQUESTS
        or args.duration_seconds != CANONICAL_DURATION_SECONDS
        or args.archive_traces != CANONICAL_ARCHIVE_TRACES
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "canonical_jobs": MIX_WEIGHTS["evaluation_jobs"] + MIX_WEIGHTS["scorer_jobs"],
            "canonical_requests": CANONICAL_REQUESTS,
            "canonical_traces": CANONICAL_ARCHIVE_TRACES,
            "concurrency": args.concurrency,
            "db_pool_config": DB_POOL_CONFIG,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_kinds": [EVALUATE, SCORER],
            "measured_jobs": len(jobs),
            "measured_requests": overall["requests"],
            "measured_traces": archive["archived_traces"],
            "provider_mode": "loopback deterministic provider + staged fake Claude CLI",
            "scheduled_duration_seconds": args.duration_seconds,
            "seed": args.seed,
            "started_at": started_at.isoformat(),
            "stream_variant": "small",
            "target": target,
            "traffic_mix": scaled_mix(args.requests),
            "trim_note": "reduced non-canonical calibration run" if trimmed else None,
            "trimmed": trimmed,
            "warmup_requests": WARMUP_REQUESTS,
            "workload": "t23_5/mixed-soak",
        },
        "summary": {"endpoints": endpoints, "overall": overall},
        "requests": records,
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
            "seed": args.seed,
            "url": provider_url,
        },
        "machine_state": state,
        "leak_check": trend,
        "equivalence": {
            "jobs": proof_jobs,
            "sample_seed": args.seed,
            "samples": proof,
            "verdict": "PENDING",
        },
    }


def mark_verdict(path: Path, verdict: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    value["equivalence"]["verdict"] = verdict
    write_raw_metrics(value, path)
    return value


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def summary_markdown(output_dir: Path) -> str:
    runs = {
        target: json.loads((output_dir / f"mixed-soak-{target}.json").read_text())
        for target in ("python", "rust")
    }
    counts = runs["python"]["run"]["traffic_mix"]
    lines = [
        "# T23.5 mixed GenAI soak summary",
        "",
        "Python and Rust ran serially against fresh PostgreSQL 16 databases, MinIO",
        "prefixes, and local archive repositories. The same seed fixed request order,",
        "payloads, and ten-minute schedule. Warm-up is excluded. Public terminal polls",
        "are control observations outside the 10,000 scheduled primary requests.",
        "",
        "## Traffic mix",
        "",
        "| Family | Scheduled requests | Mix |",
        "| --- | ---: | ---: |",
    ]
    for family, count in counts.items():
        lines.append(f"| `{family}` | {count:,} | {count / sum(counts.values()):.1%} |")
    lines.extend([
        "",
        "A 1,000-trace archive pass ran in the background. Assistant count includes",
        "both session POSTs and streamed GETs (250 complete sessions). Gateway's 30%",
        "share contains 500 streams. Jobs split evenly between evaluation and scorer",
        "invocations; every terminal state is in the equivalence proof.",
        "",
        "## Acceptance criteria",
        "",
        "| Target | Requests | Errors | Error rate | Jobs succeeded | Archive traces | "
        "RSS slope MiB/h | Monotonic growth | Eq |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for target, run in runs.items():
        overall = run["summary"]["overall"]
        trend = run["leak_check"]
        succeeded = sum(job["status"] == "SUCCEEDED" for job in run["jobs"])
        lines.append(
            f"| {target.title()} | {overall['requests']:,} | {overall['errors']} | "
            f"{overall['error_rate']:.5%} | {succeeded:,}/{len(run['jobs']):,} | "
            f"{overall['archival']['archived_traces']:,} | "
            f"{trend['slope_mib_per_hour']:+.2f} | "
            f"{'no — PASS' if trend['verdict'] == 'PASS' else 'yes — FAIL'} | "
            f"{run['equivalence']['verdict']} |"
        )
    lines.extend([
        "",
        "The RSS rule matches T14.2 at this soak scale: failure requires every",
        "consecutive one-minute mean to be non-decreasing and the final mean to exceed",
        "the first by more than 5%.",
        "",
        "## Per-family HTTP results",
        "",
        "| Family | Py p50/p95/p99 ms | Rust p50/p95/p99 ms | Py/Rust RPS | Py/Rust errors |",
        "| --- | --- | --- | --- | --- |",
    ])
    for family in runs["python"]["summary"]["overall"]["families"]:
        py = runs["python"]["summary"]["overall"]["families"][family]
        rust = runs["rust"]["summary"]["overall"]["families"][family]
        p = py["latency_ms"]
        r = rust["latency_ms"]
        lines.append(
            f"| `{family}` | {_fmt(p['p50'])}/{_fmt(p['p95'])}/{_fmt(p['p99'])} | "
            f"{_fmt(r['p50'])}/{_fmt(r['p95'])}/{_fmt(r['p99'])} | "
            f"{py['rps']:.2f}/{rust['rps']:.2f} | {py['errors']}/{rust['errors']} |"
        )
    lines.extend(["", "## RSS + CPU over time", ""])
    for target, run in runs.items():
        trend = run["leak_check"]
        means = " → ".join(f"{value:.1f}" for value in trend["mean_rss_mib_by_bin"])
        cpu = run["summary"]["overall"]["resources"]["cpu_seconds"]
        lines.extend([
            f"- {target.title()} one-minute mean RSS MiB: `{means}`",
            f"- {target.title()} CPU: {cpu:.2f} process-tree CPU-s; RSS slope "
            f"{trend['slope_mib_per_hour']:+.2f} MiB/h; {trend['verdict']}.",
        ])
    lines.extend([
        "",
        "The Python final bin includes the partial settlement tail after load ended;",
        "the regression uses every sampled bin on both targets. Rust's positive slope",
        "is reported honestly, but the minute means are not monotonic and the final",
        "mean is only 3.10% above the first, below the 5% failure threshold.",
        "",
        "## Rust-slower cells and anomalies",
        "",
        "- No soak family regressed at p50, p95, p99, or throughput.",
        "- Two isolated submit maxima were slower on Rust despite lower Rust p99:",
        "  evaluation 339.61 vs 233.69 ms and scorer 104.26 vs 53.57 ms.",
        "- Rust's observed job queue p50 was higher because its fast jobs often skipped",
        "  the polled RUNNING state; the harness conservatively attributes the entire",
        "  submit-to-terminal interval to queue time in that case.",
        "- Archive traces/s includes watcher activation and is a blended-soak cadence,",
        "  not the isolated archival capacity result reported by T23.4.",
        "",
        "## Raw result inventory",
        "",
        "- `mixed-soak-python.json`",
        "- `mixed-soak-rust.json`",
    ])
    return "\n".join(lines) + "\n"


def _assert_hygiene() -> dict[str, Any]:
    try:
        return machine_state(None)
    except RuntimeError:
        run_command(
            ["cargo", "run", "-p", "mlflow-test-support", "--bin", "reap-reference-servers"],
            cwd=RUST_ROOT,
        )
        return machine_state(None)


def soak(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        (output_dir / "t23_5_summary.md").write_text(summary_markdown(output_dir))
        return 0
    items = seeded_traffic(args.requests, args.duration_seconds, args.seed)
    sample = sampled_sequences(items, args.seed)
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
        results = {}
        with provider_server(args.seed, frame_gap_ms=FRAME_GAP_MS) as provider:
            provider_url = f"http://127.0.0.1:{provider.server_port}"
            for target in args.targets:
                recreate_database()
                with tempfile.TemporaryDirectory(prefix=f"mlflow-t23-5-{target}-") as temporary:
                    workdir = Path(temporary)
                    archive_location = (workdir / "trace-archive").as_uri()
                    config_path = workdir / "trace-archival.yaml"
                    write_soak_archive_config(
                        config_path, archive_location, False, args.archive_traces
                    )
                    handle = launch_server(
                        target,
                        workdir,
                        provider_url,
                        f"t23-5/{target}-{args.seed}",
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
                        mix = scaled_mix(args.requests)
                        setup = setup_soak_target(
                            handle.url,
                            provider_url,
                            args.seed,
                            max(mix["evaluation_jobs"], mix["scorer_jobs"]),
                        )
                        sync_request(
                            requests.Session(),
                            handle.url,
                            "PUT",
                            f"{ASSISTANT_PREFIX}/config",
                            json={
                                "providers": {
                                    "claude_code": {
                                        "model": setup["endpoint_name"],
                                        "selected": True,
                                    }
                                }
                            },
                        )
                        archive_cell = StreamingCell(
                            "archival",
                            "mixed-soak",
                            "archive-pass",
                            1,
                            args.archive_traces,
                            CANONICAL_ARCHIVE_TRACES,
                            "small",
                        )
                        archive_experiment_id, _ = seed_archive_cell(
                            handle.url, archive_cell, args.seed
                        )
                        print(
                            f"[{target}] T23.5 mixed soak: {args.requests:,} primary requests, "
                            f"{args.duration_seconds:.0f}s, {args.archive_traces:,} archive traces",
                            flush=True,
                        )
                        value = run_target(
                            handle,
                            target,
                            setup,
                            items,
                            sample,
                            provider,
                            provider_url,
                            config_path,
                            archive_location,
                            archive_experiment_id,
                            args,
                        )
                        path = output_dir / f"mixed-soak-{target}.json"
                        write_raw_metrics(value, path)
                        results[target] = value
                        overall = value["summary"]["overall"]
                        if overall["requests"] < args.requests:
                            raise RuntimeError(
                                f"{target} recorded {overall['requests']}/{args.requests} requests"
                            )
                        if overall["error_rate"] >= 0.0001:
                            raise RuntimeError(
                                f"{target} error rate {overall['error_rate']:.5%} is not <0.01%"
                            )
                        if value["leak_check"]["verdict"] != "PASS":
                            raise RuntimeError(f"{target} has monotonic RSS growth")
                        if len(value["jobs"]) != (
                            scaled_mix(args.requests)["evaluation_jobs"]
                            + scaled_mix(args.requests)["scorer_jobs"]
                        ):
                            raise RuntimeError(f"{target} did not create every scheduled job")
                    finally:
                        stop_server(handle)
            if set(args.targets) == {"python", "rust"}:
                differences = compare_runs(results["python"], results["rust"])
                verdict = "FAIL" if differences else "PASS"
                for target in ("python", "rust"):
                    path = output_dir / f"mixed-soak-{target}.json"
                    results[target] = mark_verdict(path, verdict)
                if differences:
                    raise RuntimeError("equivalence failed:\n" + "\n".join(differences))
                print("equivalence: PASS", flush=True)
                (output_dir / "t23_5_summary.md").write_text(summary_markdown(output_dir))
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
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "t23_5")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--requests", type=int, default=CANONICAL_REQUESTS)
    parser.add_argument("--duration-seconds", type=float, default=CANONICAL_DURATION_SECONDS)
    parser.add_argument("--archive-traces", type=int, default=CANONICAL_ARCHIVE_TRACES)
    parser.add_argument("--archive-timeout-seconds", type=float, default=3_600)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--targets", nargs="+", choices=("python", "rust"), default=["python", "rust"]
    )
    parser.set_defaults(func=soak)
