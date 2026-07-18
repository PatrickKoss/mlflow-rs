"""T13.3 benchmark runner: Python vs Rust MLflow server, same seeded DB.

Boots the Python FastAPI wrapper and Rust
(``rust/target/release/mlflow-server``) tracking servers sequentially on
*identical-bytes* copies of a seeded DB (see ``seed.py``), then replays each
benchmark scenario ``--iterations`` times against each server, recording
per-request wall time and reporting p50/p95/p99 per scenario per server.

Reuses the compliance harness' process/environment plumbing. The FastAPI wrapper
is required on Python because the Flask app alone does not expose ``/v1/traces``.

Scenarios (plan T13.3 wording):

* ``run_search_metric_filter`` -- runs/search with a metric filter + ordering.
* ``run_search_deep_pagination`` -- walk many pages of runs/search, reporting
  per-page latency so O(1) vs O(page) is visible.
* ``metric_history_bulk_interval`` -- get-history-bulk-interval over a run set.
* ``trace_search_span_filter`` -- traces/search with a span-attribute LIKE.
* ``otlp_ingest_throughput`` -- POST OTLP spans, measured in spans/sec.
* ``registry_search_prompt_antijoin`` -- registered-models/search (excludes
  prompts).

Usage (from repo root)::

    uv run python rust/bench/seed.py --db /tmp/bench.db [scale flags...]
    uv run python rust/bench/bench.py --db /tmp/bench.db --iterations 30 \\
        --results rust/bench/RESULTS.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_COMPLIANCE = _REPO_ROOT / "rust" / "compliance"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_COMPLIANCE))

from replay import LOCALHOST, DualServers, ServerHandle, _free_port

from tests.tracking.integration_test_utils import _rust_server_cmd


class SequentialServers(DualServers):
    """Boot Python and Rust one at a time against equivalent database copies.

    The compliance harness launches Python through the pure-flask
    ``mlflow.server:app``. That app does not serve the OTLP ``/v1/traces``
    endpoint -- it lives only on the FastAPI wrapper
    (``mlflow.server.fastapi_app:app``), which mounts the flask app at ``/`` and
    adds the OTLP router on top. Since T13.3 benchmarks OTLP ingest, the Python
    server must be the FastAPI app so both servers expose the same routes.
    """

    def __init__(
        self,
        workdir: Path,
        seed_db: Path | None,
        artifact_root: Path,
        rust_bin: Path,
        postgres_uris: dict[str, str] | None = None,
    ) -> None:
        super().__init__(workdir, seed_db or Path("."), artifact_root)
        self.rust_bin = rust_bin
        if seed_db is not None:
            py_db = workdir / "python.db"
            rust_db = workdir / "rust.db"
            shutil.copy(seed_db, py_db)
            shutil.copy(seed_db, rust_db)
            self.db_uris = {
                "python": f"sqlite:///{py_db}",
                "rust": f"sqlite:///{rust_db}",
            }
        else:
            if not postgres_uris:
                raise ValueError("Postgres runs require separate Python and Rust database URIs")
            self.db_uris = postgres_uris

    @contextlib.contextmanager
    def server(self, name: str):
        art = self.artifact_root / name
        art.mkdir(parents=True, exist_ok=True)
        port = _free_port()
        if name == "python":
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "mlflow.server.fastapi_app:app",
                "--host",
                LOCALHOST,
                "--port",
                str(port),
                "--log-level",
                "warning",
            ]
        else:
            cmd = _rust_server_cmd(self.rust_bin, port, self.db_uris[name], str(art))
        handle = self._boot(name, cmd, self.db_uris[name], art)
        try:
            yield handle
        finally:
            handle.proc.terminate()
            try:
                handle.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                handle.proc.kill()


@dataclass
class Timing:
    server: str
    scenario: str
    samples_ms: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def pct(self, p: float) -> float:
        if not self.samples_ms:
            return float("nan")
        ordered = sorted(self.samples_ms)
        k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
        return ordered[k]

    @property
    def p50(self) -> float:
        return self.pct(50)

    @property
    def p95(self) -> float:
        return self.pct(95)

    @property
    def p99(self) -> float:
        return self.pct(99)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples_ms) if self.samples_ms else float("nan")


def _timed(fn: Callable[[], requests.Response]) -> tuple[float, requests.Response]:
    t0 = time.perf_counter()
    resp = fn()
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, resp


def _pick_experiment_and_run(handle: ServerHandle) -> tuple[list[str], list[str]]:
    """Find a populated experiment id and a handful of run ids for scenarios."""
    resp = requests.post(
        f"{handle.url}/api/2.0/mlflow/experiments/search",
        json={"max_results": 1000, "order_by": ["name ASC"]},
        timeout=30,
    )
    resp.raise_for_status()
    exps = [
        e["experiment_id"]
        for e in resp.json().get("experiments", [])
        if e.get("name", "").startswith("bench_exp_")
    ]
    exp_id = exps[0] if exps else "1"
    search = requests.post(
        f"{handle.url}/api/2.0/mlflow/runs/search",
        json={"experiment_ids": exps or [exp_id], "max_results": 10},
        timeout=30,
    )
    search.raise_for_status()
    run_ids = [r["info"]["run_id"] for r in search.json().get("runs", [])]
    return (exps or [exp_id]), run_ids


# --------------------------------------------------------------------------
# Scenarios. Each takes a server handle + context, returns per-iteration ms.
# --------------------------------------------------------------------------


def scenario_run_search_metric_filter(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    t = Timing(handle.name, "run_search_metric_filter")
    body = {
        "experiment_ids": ctx["exp_ids"],
        "filter": "metrics.loss < 0.5",
        "order_by": ["metrics.accuracy DESC"],
        "max_results": 100,
    }
    for _ in range(iters):
        dt, resp = _timed(
            lambda: requests.post(f"{handle.url}/api/2.0/mlflow/runs/search", json=body, timeout=60)
        )
        resp.raise_for_status()
        if not resp.json().get("runs"):
            raise RuntimeError("run-search metric filter returned no rows")
        t.samples_ms.append(dt)
    return t


def scenario_run_search_deep_pagination(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    """Walk many pages; record per-page latency so O(page) blow-up is visible.

    ``iters`` here caps the number of pages walked (deep-pagination is a single
    walk, not a repeated point query). Reports p50/p95 across per-page latencies
    plus first-vs-last page timing in ``extra``.
    """
    t = Timing(handle.name, "run_search_deep_pagination")
    max_pages = ctx.get("deep_pages", max(20, iters))
    body = {
        "experiment_ids": ctx["exp_ids"],
        "order_by": ["attributes.start_time ASC"],
        "max_results": 50,
    }
    per_page: list[float] = []
    token = None
    for page in range(max_pages):
        payload = dict(body)
        if token:
            payload["page_token"] = token
        dt, resp = _timed(
            lambda p=payload: requests.post(
                f"{handle.url}/api/2.0/mlflow/runs/search", json=p, timeout=60
            )
        )
        resp.raise_for_status()
        if not resp.json().get("runs"):
            raise RuntimeError(f"deep-pagination page {page + 1} returned no rows")
        per_page.append(dt)
        token = resp.json().get("next_page_token")
        if not token:
            break
    t.samples_ms = per_page
    bucket_size = max(1, min(5, len(per_page) // 3))
    first = statistics.median(per_page[:bucket_size]) if per_page else None
    middle_start = max(0, (len(per_page) - bucket_size) // 2)
    middle = (
        statistics.median(per_page[middle_start : middle_start + bucket_size]) if per_page else None
    )
    last = statistics.median(per_page[-bucket_size:]) if per_page else None
    late_to_early = last / first if first and last else None
    t.extra = {
        "pages_walked": len(per_page),
        "bucket_size": bucket_size,
        "first_pages_median_ms": round(first, 2) if first is not None else None,
        "middle_pages_median_ms": round(middle, 2) if middle is not None else None,
        "last_pages_median_ms": round(last, 2) if last is not None else None,
        "last_to_first_ratio": round(late_to_early, 3) if late_to_early is not None else None,
        "o1_verdict": "consistent with O(1)"
        if late_to_early and late_to_early <= 2
        else "not O(1)",
    }
    return t


def scenario_metric_history_bulk_interval(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    t = Timing(handle.name, "metric_history_bulk_interval")
    run_ids = ctx["run_ids"][:5] or ctx["run_ids"]
    params = [("run_ids", r) for r in run_ids]
    max_step = max(0, ctx.get("history_points", 100) - 1)
    start_step = max_step // 4
    end_step = max(start_step, (max_step * 3) // 4)
    params += [
        ("metric_key", "loss"),
        ("start_step", str(start_step)),
        ("end_step", str(end_step)),
        ("max_results", "100"),
    ]
    for _ in range(iters):
        dt, resp = _timed(
            lambda: requests.get(
                f"{handle.url}/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
                params=params,
                timeout=60,
            )
        )
        resp.raise_for_status()
        if not resp.json().get("metrics"):
            raise RuntimeError("bulk-interval metric history returned no rows")
        t.samples_ms.append(dt)
    return t


def scenario_trace_search_span_filter(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    t = Timing(handle.name, "trace_search_span_filter")
    body = {
        "locations": [
            {"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": e}}
            for e in ctx["exp_ids"]
        ],
        "filter": "span.attributes.`gen_ai.request.model` LIKE '%gpt%'",
        "max_results": 100,
    }
    for _ in range(iters):
        dt, resp = _timed(
            lambda: requests.post(
                f"{handle.url}/api/3.0/mlflow/traces/search", json=body, timeout=60
            )
        )
        resp.raise_for_status()
        if not resp.json().get("traces"):
            raise RuntimeError("trace span-attribute filter returned no rows")
        t.samples_ms.append(dt)
    return t


def _otlp_payload(n_spans: int, batch: int) -> bytes:
    """Serialize a minimal binary OTLP export containing new traces and spans."""
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    service_name = resource_spans.resource.attributes.add()
    service_name.key = "service.name"
    service_name.value.string_value = "bench"
    scope_spans = resource_spans.scope_spans.add()
    for i in range(n_spans):
        trace_number = batch * n_spans + i + 1
        span = scope_spans.spans.add()
        span.trace_id = trace_number.to_bytes(16, "big")
        span.span_id = trace_number.to_bytes(8, "big")
        span.name = f"bench_span_{i}"
        span.start_time_unix_nano = 1_000_000_000 + i * 1000
        span.end_time_unix_nano = span.start_time_unix_nano + 500
        span.status.code = 1
        span_type = span.attributes.add()
        span_type.key = "mlflow.spanType"
        span_type.value.string_value = "LLM"
        model = span.attributes.add()
        model.key = "gen_ai.request.model"
        model.value.string_value = "gpt-4o-mini"
    return request.SerializeToString()


def scenario_otlp_ingest_throughput(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    """POST OTLP span batches; report per-batch latency + aggregate spans/sec."""
    t = Timing(handle.name, "otlp_ingest_throughput")
    exp_id = ctx["exp_ids"][0]
    spans_per_batch = ctx.get("otlp_spans_per_batch", 100)
    headers = {
        "x-mlflow-experiment-id": exp_id,
        "Content-Type": "application/x-protobuf",
    }
    total_spans = 0
    sequence = ctx.setdefault("otlp_sequence", 0)
    for b in range(iters):
        payload = _otlp_payload(spans_per_batch, sequence + b)
        dt, resp = _timed(
            lambda p=payload: requests.post(
                f"{handle.url}/v1/traces", data=p, headers=headers, timeout=120
            )
        )
        resp.raise_for_status()
        t.samples_ms.append(dt)
        total_spans += spans_per_batch
    ctx["otlp_sequence"] = sequence + iters
    wall = sum(t.samples_ms) / 1000
    t.extra = {
        "spans_ingested": total_spans,
        "wall_s": round(wall, 3),
        "spans_per_sec": round(total_spans / wall, 1) if wall > 0 else None,
    }
    return t


def scenario_registry_search_prompt_antijoin(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    t = Timing(handle.name, "registry_search_prompt_antijoin")
    for _ in range(iters):
        dt, resp = _timed(
            lambda: requests.get(
                f"{handle.url}/api/2.0/mlflow/registered-models/search",
                params={"max_results": 100, "order_by": "name ASC"},
                timeout=60,
            )
        )
        resp.raise_for_status()
        registered_models = resp.json().get("registered_models", [])
        if not registered_models:
            raise RuntimeError("registered-model search returned no rows")
        if any(model["name"].startswith("prompt_") for model in registered_models):
            raise RuntimeError("registered-model search did not exclude seeded prompts")
        t.samples_ms.append(dt)
    return t


SCENARIOS: dict[str, Callable[[ServerHandle, dict, int], Timing]] = {
    "run_search_metric_filter": scenario_run_search_metric_filter,
    "run_search_deep_pagination": scenario_run_search_deep_pagination,
    "metric_history_bulk_interval": scenario_metric_history_bulk_interval,
    "trace_search_span_filter": scenario_trace_search_span_filter,
    "otlp_ingest_throughput": scenario_otlp_ingest_throughput,
    "registry_search_prompt_antijoin": scenario_registry_search_prompt_antijoin,
}


# --------------------------------------------------------------------------
# Hardware notes.
# --------------------------------------------------------------------------


def _hardware_notes() -> dict[str, str]:
    notes: dict[str, str] = {"platform": platform.platform()}
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        model = next(
            (ln.split(":", 1)[1].strip() for ln in cpu.splitlines() if "Model name" in ln), "?"
        )
        cores = next(
            (ln.split(":", 1)[1].strip() for ln in cpu.splitlines() if ln.startswith("CPU(s):")),
            "?",
        )
        notes["cpu"] = f"{model} ({cores} logical CPUs)"
    except Exception as exc:
        notes["cpu"] = f"unavailable: {exc}"
    try:
        mem = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10).stdout
        notes["memory"] = next(
            (ln for ln in mem.splitlines() if ln.startswith("Mem:")), "?"
        ).strip()
    except Exception as exc:
        notes["memory"] = f"unavailable: {exc}"
    return notes


# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------


def _warmup(handle: ServerHandle, ctx: dict, names: list[str], iterations: int) -> None:
    warm_ctx = {**ctx, "deep_pages": iterations}
    for name in names:
        SCENARIOS[name](handle, warm_ctx, iterations)
    ctx["otlp_sequence"] = warm_ctx.get("otlp_sequence", 0)


def run_all(
    servers: SequentialServers,
    iterations: int,
    warmup_iterations: int,
    only: list[str] | None,
    history_points: int,
    deep_pages: int,
    otlp_spans_per_batch: int,
) -> dict[str, dict[str, Timing]]:
    names = [name for name in SCENARIOS if not only or name in only]
    results: dict[str, dict[str, Timing]] = {name: {} for name in names}
    for server_name in ("python", "rust"):
        with servers.server(server_name) as handle:
            print(f"{server_name}: {handle.url}")
            exp_ids, run_ids = _pick_experiment_and_run(handle)
            ctx = {
                "exp_ids": exp_ids,
                "run_ids": run_ids,
                "history_points": history_points,
                "deep_pages": deep_pages,
                "otlp_spans_per_batch": otlp_spans_per_batch,
                "otlp_sequence": 0,
            }
            _warmup(handle, ctx, names, warmup_iterations)
            for name in names:
                print(f"  {name}")
                results[name][server_name] = SCENARIOS[name](handle, ctx, iterations)
    return results


def write_results(
    results: dict[str, dict[str, Timing]],
    hw: dict[str, str],
    scale: dict[str, int],
    iterations: int,
    warmup_iterations: int,
    deep_pages: int,
    out: Path,
    counts: dict[str, int] | None,
    seed_metadata: dict[str, Any] | None,
    backend: str,
    database_bytes: int | None,
) -> None:
    lines: list[str] = []
    lines.append("# T13.3 Benchmark Results - Python vs Rust MLflow Server")
    lines.append("")
    lines.append(
        "Auto-generated by `rust/bench/bench.py`. Numbers below were **measured on the "
        "scale and hardware documented here** -- they are NOT the 100 GB targets. See "
        "*Reproducing at full scale* for the rig configuration."
    )
    lines.append("")
    lines.append("## Hardware / environment")
    lines.append("")
    lines.append(f"- CPU: {hw.get('cpu')}")
    lines.append(f"- Memory: {hw.get('memory')}")
    lines.append(f"- Platform: {hw.get('platform')}")
    lines.append(
        "- **WSL2 caveat:** this was run inside WSL2 on a shared developer laptop, not a "
        "dedicated benchmark host. Absolute latencies include VM/filesystem overhead and "
        "background noise; treat the Python-vs-Rust *ratio* as the signal, not the raw ms."
    )
    lines.append(f"- Backend store: {backend}")
    lines.append("- Server scheduling: sequential; only the measured server process was running.")
    lines.append("- Rust binary: release profile (`cargo build --release`).")
    lines.append("")
    lines.append("## Scale actually run")
    lines.append("")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    requested_scale = seed_metadata.get("scale", {}) if seed_metadata else {}
    for k, v in (requested_scale or scale).items():
        lines.append(f"| {k} | {v:,} |")
    lines.append(f"| iterations/scenario | {iterations:,} |")
    lines.append(f"| unmeasured warmups/scenario | {warmup_iterations:,} |")
    lines.append(f"| deep-pagination pages | {deep_pages:,} |")
    if seed_metadata:
        lines.append(f"| seed total wall seconds | {seed_metadata['total_seconds']:.3f} |")
    if database_bytes is not None:
        lines.append(f"| SQLite seed file bytes | {database_bytes:,} |")
    lines.append("")
    if counts:
        lines.append("Seeded row counts:")
        lines.append("")
        lines.append("| table | rows |")
        lines.append("|---|---|")
        for k, v in sorted(counts.items()):
            lines.append(f"| {k} | {v:,} |")
        lines.append("")

    lines.append("## Latency: p50 / p95 / p99 (ms), Python vs Rust")
    lines.append("")
    lines.append(
        "| scenario | py p50 | py p95 | py p99 | rust p50 | rust p95 | rust p99 | p95 speedup |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, by_server in results.items():
        py = by_server.get("python")
        rust = by_server.get("rust")
        if not py or not rust:
            continue
        speedup = (py.p95 / rust.p95) if rust.p95 else float("nan")
        lines.append(
            f"| {name} | {py.p50:.1f} | {py.p95:.1f} | {py.p99:.1f} | "
            f"{rust.p50:.1f} | {rust.p95:.1f} | {rust.p99:.1f} | {speedup:.2f}x |"
        )
    lines.append("")

    lines.append("## Scenario detail")
    lines.append("")
    for name, by_server in results.items():
        lines.append(f"### {name}")
        lines.append("")
        for server, t in by_server.items():
            extra = f"  extra={json.dumps(t.extra)}" if t.extra else ""
            lines.append(
                f"- **{server}**: n={len(t.samples_ms)} mean={t.mean:.1f}ms "
                f"p50={t.p50:.1f} p95={t.p95:.1f} p99={t.p99:.1f}{extra}"
            )
        lines.append("")

    lines.append("## Targets (plan AC)")
    lines.append("")
    lines.append("| target | status at this scale |")
    lines.append("|---|---|")
    rsm = results.get("run_search_metric_filter", {}).get("rust")
    if rsm:
        ok = rsm.p95 < 500
        lines.append(
            f"| p95 run-search < 500 ms | rust p95 = {rsm.p95:.1f} ms -> "
            f"{'MET' if ok else 'NOT MET'} |"
        )
    dp = results.get("run_search_deep_pagination", {}).get("rust")
    if dp and dp.extra.get("first_pages_median_ms") is not None:
        ratio = dp.extra["last_to_first_ratio"]
        verdict = dp.extra["o1_verdict"]
        lines.append(f"| deep-page O(1) | rust late/early bucket ratio {ratio:.2f}x -> {verdict} |")
    otlp_py = results.get("otlp_ingest_throughput", {}).get("python")
    otlp_rust = results.get("otlp_ingest_throughput", {}).get("rust")
    if otlp_py and otlp_rust:
        py_sps = otlp_py.extra.get("spans_per_sec") or 0
        rust_sps = otlp_rust.extra.get("spans_per_sec") or 0
        ratio = (rust_sps / py_sps) if py_sps else float("nan")
        ok = ratio >= 5.0
        lines.append(
            f"| OTLP ingest >= 5x Python | py={py_sps} rust={rust_sps} spans/s "
            f"({ratio:.2f}x) -> {'MET' if ok else 'NOT MET'} |"
        )
    lines.append("")

    lines.append("## Deep-pagination curve")
    lines.append("")
    lines.append(
        "The operational verdict is `consistent with O(1)` when the median of the last "
        "five pages is no more than 2x the median of the first five pages. This is an "
        "empirical bounded-depth verdict, not an asymptotic proof."
    )
    lines.append("")
    lines.append("| server | first pages | middle pages | last pages | late/early | verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for server, timing in results.get("run_search_deep_pagination", {}).items():
        extra = timing.extra
        lines.append(
            f"| {server} | {extra['first_pages_median_ms']:.2f} ms | "
            f"{extra['middle_pages_median_ms']:.2f} ms | "
            f"{extra['last_pages_median_ms']:.2f} ms | "
            f"{extra['last_to_first_ratio']:.2f}x | {extra['o1_verdict']} |"
        )
    lines.append("")

    lines.append("## OTLP protobuf ingest throughput")
    lines.append("")
    lines.append("| server | spans | measured request wall | spans/second |")
    lines.append("|---|---:|---:|---:|")
    for server, timing in results.get("otlp_ingest_throughput", {}).items():
        extra = timing.extra
        lines.append(
            f"| {server} | {extra['spans_ingested']:,} | {extra['wall_s']:.3f} s | "
            f"{extra['spans_per_sec']:,.1f} |"
        )
    lines.append("")

    lines.append("## Reproducing at full (100 GB) scale")
    lines.append("")
    lines.append(
        "Use fresh databases on a dedicated host (>=16 physical cores, >=64 GB RAM, NVMe). "
        "The following starts with a high-volume candidate scale; database size varies by "
        "Postgres version and index/storage settings, so verify it with `pg_database_size` "
        "and adjust the scale instead of treating the flags as an exact byte estimate."
    )
    lines.append("")
    lines.append("```bash")
    lines.append("createdb mlflow_bench_seed")
    lines.append("export BENCH_SEED='postgresql://mlflow:mlflow@localhost/mlflow_bench_seed'")
    lines.append('uv run python rust/bench/seed.py --db "$BENCH_SEED" \\')
    lines.append("    --runs 750000 --metrics-per-run 5 --history-points 100 \\")
    lines.append("    --traces 2000000 --spans-per-trace 5 --model-versions 100000 \\")
    lines.append("    --experiments 500 --seed 42 --metadata /tmp/t133-seed.json")
    lines.append('psql "$BENCH_SEED" -Atc \\')
    lines.append("    \"SELECT pg_size_pretty(pg_database_size('mlflow_bench_seed'));\"")
    lines.append("createdb mlflow_bench_python")
    lines.append("createdb mlflow_bench_rust")
    lines.append('pg_dump --format=custom "$BENCH_SEED" --file=/tmp/mlflow-bench.dump')
    lines.append("pg_restore --dbname=mlflow_bench_python --jobs=8 /tmp/mlflow-bench.dump")
    lines.append("pg_restore --dbname=mlflow_bench_rust --jobs=8 /tmp/mlflow-bench.dump")
    lines.append("psql postgresql://mlflow:mlflow@localhost/mlflow_bench_python -c 'ANALYZE;'")
    lines.append("psql postgresql://mlflow:mlflow@localhost/mlflow_bench_rust -c 'ANALYZE;'")
    lines.append("cargo build --manifest-path rust/Cargo.toml --release")
    lines.append("uv run python rust/bench/bench.py \\")
    lines.append("    --python-db-uri postgresql://mlflow:mlflow@localhost/mlflow_bench_python \\")
    lines.append("    --rust-db-uri postgresql://mlflow:mlflow@localhost/mlflow_bench_rust \\")
    lines.append("    --seed-metadata /tmp/t133-seed.json --iterations 200 --deep-pages 100 \\")
    lines.append("    --results rust/bench/RESULTS.md")
    lines.append("```")
    lines.append("")
    lines.append(
        "Do not point both servers at one database: OTLP is a write benchmark. Restoring the "
        "same dump into two databases gives both servers equivalent starting rows while keeping "
        "their writes isolated. Run `ANALYZE` on both restored databases if autovacuum has not."
    )
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    faster_rust = []
    faster_python = []
    anomalies = []
    for name, by_server in results.items():
        py, rust = by_server.get("python"), by_server.get("rust")
        if not py or not rust:
            continue
        (faster_rust if rust.p95 < py.p95 else faster_python).append(name)
        for timing in (py, rust):
            tail_ratio = timing.p99 / timing.p50 if timing.p50 else 0
            if tail_ratio >= 2:
                anomalies.append(f"{timing.server} {name} p99/p50 was {tail_ratio:.2f}x")
    otlp_py = results.get("otlp_ingest_throughput", {}).get("python")
    otlp_rust = results.get("otlp_ingest_throughput", {}).get("rust")
    otlp_note = ""
    if otlp_py and otlp_rust:
        otlp_ratio = otlp_rust.extra["spans_per_sec"] / otlp_py.extra["spans_per_sec"]
        otlp_note = (
            f"OTLP was {otlp_ratio:.2f}x faster in Rust, "
            f"{'meeting' if otlp_ratio >= 5 else 'below'} the 5x target."
        )
    anomaly_note = "; ".join(anomalies) if anomalies else "none by the 2x p99/p50 rule"
    lines.append(
        f"Rust had the lower measured p95 in {', '.join(faster_rust) or 'no scenarios'}; "
        f"Python had the lower or equal p95 in {', '.join(faster_python) or 'no scenarios'}. "
        f"{otlp_note} "
        "The deep-page table reports the observed curve independently for each server. "
        f"Observed long-tail anomalies: {anomaly_note}. "
        "These SQLite/WSL2 results are sensitive to cache state and VM noise, and the OTLP "
        "measurement is sequential batch ingest rather than a concurrent saturation test."
    )
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", help="seeded SQLite file (copied byte-for-byte per server)")
    source.add_argument("--python-db-uri", help="Python server Postgres URI")
    p.add_argument("--rust-db-uri", help="Rust server Postgres URI; required with --python-db-uri")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--warmup-iterations", type=int, default=3)
    p.add_argument("--deep-pages", type=int, default=25)
    p.add_argument("--otlp-spans-per-batch", type=int, default=100)
    p.add_argument(
        "--rust-bin", default=str(_REPO_ROOT / "rust" / "target" / "release" / "mlflow-server")
    )
    p.add_argument("--results", default=str(_HERE / "RESULTS.md"))
    p.add_argument("-k", dest="only", action="append", help="only run matching scenario(s)")
    p.add_argument("--seed-metadata", help="JSON written by seed.py --metadata")
    args = p.parse_args()

    if args.iterations < 1 or args.warmup_iterations < 1:
        p.error("iterations and warmup iterations must be positive")
    if args.deep_pages < 20:
        p.error("--deep-pages must be at least 20")
    if args.otlp_spans_per_batch < 1:
        p.error("--otlp-spans-per-batch must be positive")
    unknown = set(args.only or []) - set(SCENARIOS)
    if unknown:
        p.error(f"unknown scenarios: {', '.join(sorted(unknown))}")

    db_path = Path(args.db).resolve() if args.db else None
    if db_path is not None and not db_path.exists():
        print(f"Seeded DB not found: {db_path}. Run seed.py first.", file=sys.stderr)
        return 2
    if bool(args.python_db_uri) != bool(args.rust_db_uri):
        p.error("--python-db-uri and --rust-db-uri must be supplied together")
    rust_bin = Path(args.rust_bin).resolve()
    if not rust_bin.exists():
        print(
            f"Release Rust server not found at {rust_bin}. Run `cargo build --release` in rust/.",
            file=sys.stderr,
        )
        return 2

    seed_metadata = None
    if args.seed_metadata:
        seed_metadata = json.loads(Path(args.seed_metadata).read_text())

    inspect_db = args.db or args.python_db_uri
    scale = _infer_scale(inspect_db)
    counts = seed_metadata.get("counts") if seed_metadata else scale

    import tempfile

    with tempfile.TemporaryDirectory(prefix="t133-bench-") as td:
        workdir = Path(td)
        artifact_root = workdir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        postgres_uris = (
            {"python": args.python_db_uri, "rust": args.rust_db_uri} if args.python_db_uri else None
        )
        servers = SequentialServers(
            workdir,
            db_path,
            artifact_root,
            rust_bin,
            postgres_uris=postgres_uris,
        )
        results = run_all(
            servers,
            args.iterations,
            args.warmup_iterations,
            args.only,
            scale.get("history_points", 100),
            args.deep_pages,
            args.otlp_spans_per_batch,
        )

    hw = _hardware_notes()
    write_results(
        results,
        hw,
        scale,
        args.iterations,
        args.warmup_iterations,
        args.deep_pages,
        Path(args.results),
        counts,
        seed_metadata,
        "SQLite (byte-identical per-server copies)" if db_path else "Postgres (dump clones)",
        db_path.stat().st_size if db_path else None,
    )
    return 0


def _infer_scale(db: str) -> dict[str, int]:
    """Best-effort row counts from the seeded DB for the report's scale table."""
    from sqlalchemy import create_engine

    uri = db if "://" in db else f"sqlite:///{Path(db).resolve()}"
    engine = create_engine(uri)
    scale: dict[str, int] = {}
    queries = {
        "runs": "SELECT COUNT(*) FROM runs",
        "metrics": "SELECT COUNT(*) FROM metrics",
        "traces": "SELECT COUNT(*) FROM trace_info",
        "spans": "SELECT COUNT(*) FROM spans",
        "model_versions": "SELECT COUNT(*) FROM model_versions",
        "experiments": "SELECT COUNT(*) FROM experiments",
        "span_attributes": "SELECT COUNT(*) FROM span_attributes",
        "registered_models": "SELECT COUNT(*) FROM registered_models",
        "history_points": "SELECT COALESCE(MAX(step), -1) + 1 FROM metrics",
    }
    with engine.begin() as conn:
        for k, q in queries.items():
            try:
                scale[k] = int(conn.exec_driver_sql(q).scalar() or 0)
            except Exception:
                scale[k] = -1
    engine.dispose()
    return scale


if __name__ == "__main__":
    raise SystemExit(main())
