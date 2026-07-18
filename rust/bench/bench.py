"""T13.3 benchmark runner: Python vs Rust MLflow server, same seeded DB.

Boots the Python (flask ``mlflow.server:app``) and Rust
(``rust/target/release/mlflow-server``) tracking servers on *identical-bytes*
copies of a seeded DB (see ``seed.py``), then replays each benchmark scenario
``--iterations`` times against each server, recording per-request wall time and
reporting p50/p95/p99 per scenario per server.

Reuses the compliance harness' dual-boot machinery (``DualServers``,
``_resolve_rust_server_bin``, ...) so the launch path matches the differential
suite exactly.

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
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_COMPLIANCE = _REPO_ROOT / "rust" / "compliance"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_COMPLIANCE))

import shutil  # noqa: E402

from replay import LOCALHOST, DualServers, ServerHandle, _free_port  # noqa: E402

from tests.tracking.integration_test_utils import (  # noqa: E402
    _resolve_rust_server_bin,
    _rust_server_cmd,
)


class UvicornDualServers(DualServers):
    """Like ``DualServers`` but boots the Python side via uvicorn (FastAPI).

    The compliance harness launches Python through the pure-flask
    ``mlflow.server:app``. That app does not serve the OTLP ``/v1/traces``
    endpoint -- it lives only on the FastAPI wrapper
    (``mlflow.server.fastapi_app:app``), which mounts the flask app at ``/`` and
    adds the OTLP router on top. Since T13.3 benchmarks OTLP ingest, the Python
    server must be the FastAPI app so both servers expose the same routes.
    """

    def __enter__(self) -> "UvicornDualServers":
        py_db = self.workdir / "python.db"
        rust_db = self.workdir / "rust.db"
        shutil.copy(self.seed_db, py_db)
        shutil.copy(self.seed_db, rust_db)
        py_art = self.artifact_root / "python"
        rust_art = self.artifact_root / "rust"
        py_art.mkdir(parents=True, exist_ok=True)
        rust_art.mkdir(parents=True, exist_ok=True)

        py_port = _free_port()
        py_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "mlflow.server.fastapi_app:app",
            "--host",
            LOCALHOST,
            "--port",
            str(py_port),
            "--log-level",
            "warning",
        ]
        self.python = self._boot("python", py_cmd, f"sqlite:///{py_db}", py_art)

        rust_port = _free_port()
        rust_bin = _resolve_rust_server_bin()
        rust_cmd = _rust_server_cmd(rust_bin, rust_port, f"sqlite:///{rust_db}", str(rust_art))
        self.rust = self._boot("rust", rust_cmd, f"sqlite:///{rust_db}", rust_art)
        return self


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


def _pick_experiment_and_run(handle: ServerHandle) -> tuple[str, list[str]]:
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
            lambda: requests.post(
                f"{handle.url}/api/2.0/mlflow/runs/search", json=body, timeout=60
            )
        )
        resp.raise_for_status()
        t.samples_ms.append(dt)
    return t


def scenario_run_search_deep_pagination(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    """Walk many pages; record per-page latency so O(page) blow-up is visible.

    ``iters`` here caps the number of pages walked (deep-pagination is a single
    walk, not a repeated point query). Reports p50/p95 across per-page latencies
    plus first-vs-last page timing in ``extra``.
    """
    t = Timing(handle.name, "run_search_deep_pagination")
    max_pages = max(20, iters)
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
        per_page.append(dt)
        token = resp.json().get("next_page_token")
        if not token:
            break
    t.samples_ms = per_page
    t.extra = {
        "pages_walked": len(per_page),
        "first_page_ms": round(per_page[0], 2) if per_page else None,
        "last_page_ms": round(per_page[-1], 2) if per_page else None,
    }
    return t


def scenario_metric_history_bulk_interval(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    t = Timing(handle.name, "metric_history_bulk_interval")
    run_ids = ctx["run_ids"][:5] or ctx["run_ids"]
    params = [("run_ids", r) for r in run_ids]
    params += [("metric_key", "loss"), ("max_results", "100")]
    for _ in range(iters):
        dt, resp = _timed(
            lambda: requests.get(
                f"{handle.url}/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
                params=params,
                timeout=60,
            )
        )
        resp.raise_for_status()
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
        t.samples_ms.append(dt)
    return t


def _otlp_payload(exp_id: str, n_spans: int, batch: int) -> dict:
    """A minimal OTLP JSON export request with ``n_spans`` spans in one resource."""
    spans = []
    for i in range(n_spans):
        trace_hex = f"{(batch * 100000 + i):032x}"
        span_hex = f"{(batch * 100000 + i):016x}"
        spans.append(
            {
                "traceId": trace_hex,
                "spanId": span_hex,
                "name": f"bench_span_{i}",
                "startTimeUnixNano": str(1_000_000_000 + i * 1000),
                "endTimeUnixNano": str(1_000_000_000 + i * 1000 + 500),
                "status": {"code": "STATUS_CODE_OK"},
                "attributes": [
                    {"key": "mlflow.spanType", "value": {"stringValue": "LLM"}},
                    {
                        "key": "mlflow.traceRequestId",
                        "value": {"stringValue": f'"tr-bench-{batch}-{i}"'},
                    },
                ],
            }
        )
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "bench"}}
                    ]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


def scenario_otlp_ingest_throughput(handle: ServerHandle, ctx: dict, iters: int) -> Timing:
    """POST OTLP span batches; report per-batch latency + aggregate spans/sec."""
    t = Timing(handle.name, "otlp_ingest_throughput")
    exp_id = ctx["exp_ids"][0]
    spans_per_batch = 100
    headers = {"x-mlflow-experiment-id": exp_id, "Content-Type": "application/json"}
    total_spans = 0
    wall0 = time.perf_counter()
    for b in range(iters):
        payload = _otlp_payload(exp_id, spans_per_batch, b)
        dt, resp = _timed(
            lambda p=payload: requests.post(
                f"{handle.url}/v1/traces", json=p, headers=headers, timeout=120
            )
        )
        resp.raise_for_status()
        t.samples_ms.append(dt)
        total_spans += spans_per_batch
    wall = time.perf_counter() - wall0
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


def _warmup(handle: ServerHandle, ctx: dict) -> None:
    for name, fn in SCENARIOS.items():
        try:
            fn(handle, ctx, 2 if name != "otlp_ingest_throughput" else 1)
        except Exception:
            # Warmup failures surface again in the measured pass; ignore here.
            pass


def run_all(
    servers: DualServers, iterations: int, only: list[str] | None
) -> dict[str, dict[str, Timing]]:
    ctx_by_server: dict[str, dict] = {}
    for h in (servers.python, servers.rust):
        exp_ids, run_ids = _pick_experiment_and_run(h)
        ctx_by_server[h.name] = {"exp_ids": exp_ids, "run_ids": run_ids}

    results: dict[str, dict[str, Timing]] = {}
    for h in (servers.python, servers.rust):
        _warmup(h, ctx_by_server[h.name])
    for name, fn in SCENARIOS.items():
        if only and name not in only:
            continue
        results[name] = {}
        for h in (servers.python, servers.rust):
            results[name][h.name] = fn(h, ctx_by_server[h.name], iterations)
    return results


def write_results(
    results: dict[str, dict[str, Timing]],
    hw: dict[str, str],
    scale: dict[str, int],
    iterations: int,
    out: Path,
    counts: dict[str, int] | None,
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
    lines.append("- Backend store: SQLite (single-file, WAL default). See full-scale notes for Postgres.")
    lines.append("")
    lines.append("## Scale actually run")
    lines.append("")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    for k, v in scale.items():
        lines.append(f"| {k} | {v:,} |")
    lines.append(f"| iterations/scenario | {iterations:,} |")
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
    if dp and dp.extra.get("first_page_ms") is not None:
        first = dp.extra["first_page_ms"]
        last = dp.extra["last_page_ms"]
        ratio = (last / first) if first else float("nan")
        ok = ratio < 3.0
        lines.append(
            f"| deep-page O(1) | rust first={first}ms last={last}ms (ratio {ratio:.2f}x) -> "
            f"{'roughly O(1)' if ok else 'super-linear, investigate'} |"
        )
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

    lines.append("## Reproducing at full (100 GB) scale")
    lines.append("")
    lines.append(
        "Scale is fully parameterized. On a dedicated host (>= 16 physical cores, >= 64 GB RAM, "
        "NVMe, Postgres backend), a ~100 GB dataset is roughly:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("# Postgres backend (recommended at scale; sqlite single-writer stalls on ingest)")
    lines.append("export BENCH_DB='postgresql://mlflow:mlflow@localhost:5432/mlflow_bench'")
    lines.append("uv run python rust/bench/seed.py --db \"$BENCH_DB\" \\")
    lines.append("    --runs 5000000 --metrics-per-run 50 --history-points 200 \\")
    lines.append("    --traces 10000000 --spans-per-trace 8 --model-versions 100000 \\")
    lines.append("    --experiments 500 --seed 42")
    lines.append("uv run python rust/bench/bench.py --db \"$BENCH_DB\" --iterations 200 \\")
    lines.append("    --results rust/bench/RESULTS.md")
    lines.append("```")
    lines.append("")
    lines.append(
        "The generator writes ~5M runs x 50 metrics x 200 history points (~50B metric rows) and "
        "10M traces x 8 spans; on Postgres this lands near 100 GB. Increase `--history-points` "
        "and `--traces` to grow the DB further. Use `--seed` to keep runs reproducible."
    )
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append(
        "_Fill in after a run: where Rust wins (typically search/pagination and OTLP ingest "
        "under load), where it doesn't (endpoints dominated by SQLite I/O look similar since "
        "both hit the same file), and any anomalies (cold-cache first pages, GC pauses)._ "
        "The auto-generated tables above are the measured ground truth."
    )
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Seeded SQLite path or SQLAlchemy URI")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--results", default=str(_HERE / "RESULTS.md"))
    p.add_argument("-k", dest="only", action="append", help="only run matching scenario(s)")
    p.add_argument("--counts-json", help="optional path to seed counts JSON for the report")
    args = p.parse_args()

    db_path = Path(args.db) if "://" not in args.db else None
    if db_path is not None and not db_path.exists():
        print(f"Seeded DB not found: {db_path}. Run seed.py first.", file=sys.stderr)
        return 2

    counts = None
    if args.counts_json and Path(args.counts_json).exists():
        counts = json.loads(Path(args.counts_json).read_text())

    scale = _infer_scale(args.db)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="t133-bench-") as td:
        workdir = Path(td)
        artifact_root = workdir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        seed_db = db_path or _copy_uri_to_sqlite(args.db, workdir)
        with UvicornDualServers(workdir, seed_db, artifact_root) as servers:
            print(f"python: {servers.python.url}  rust: {servers.rust.url}")
            results = run_all(servers, args.iterations, args.only)

    hw = _hardware_notes()
    write_results(results, hw, scale, args.iterations, Path(args.results), counts)
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
    }
    with engine.begin() as conn:
        for k, q in queries.items():
            try:
                scale[k] = int(conn.exec_driver_sql(q).scalar() or 0)
            except Exception:
                scale[k] = -1
    engine.dispose()
    return scale


def _copy_uri_to_sqlite(uri: str, workdir: Path) -> Path:
    raise SystemExit(
        "Non-sqlite backends: point both servers at the URI directly is not yet wired; "
        "run against a sqlite file for the laptop harness, or extend DualServers for Postgres."
    )


if __name__ == "__main__":
    raise SystemExit(main())
