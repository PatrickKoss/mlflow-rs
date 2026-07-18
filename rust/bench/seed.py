"""T13.3 seeded dataset generator.

Populates a *migrated* MLflow tracking + model-registry SQLite (or Postgres) DB
with synthetic-but-realistically-shaped data, sized entirely by CLI flags so the
same script drives a laptop-scale smoke run and a 100 GB benchmark rig.

The schema is created via the real store (``_seed_tracking_db`` runs the full
Alembic chain so the Rust server, which refuses an unmigrated DB, accepts the
result). Bulk rows are then written with SQLAlchemy Core ``executemany`` against
the ORM tables' ``__table__`` objects (column names therefore always match the
live schema) in autocommitted batches -- orders of magnitude faster than issuing
one REST/ORM call per entity.

Shapes mirror production MLflow:

* runs carry lifecycle_stage (mostly active, a slice deleted), a status mix
  (FINISHED heavy, some RUNNING/FAILED), params, tags (incl. mlflow.runName),
  a dense per-metric history (``metrics`` rows) plus the derived
  ``latest_metrics`` row the search path reads.
* traces get request/response previews and a status mix; each fans out into a
  root + child spans whose ``content`` JSON carries an ``attributes`` map so the
  span-attribute LIKE filter has something to bite on.
* the registry gets registered models + versions across stages, and a
  configurable slice of *prompts* (``mlflow.prompt.is_prompt`` tag) so the
  registered-models anti-join actually excludes rows.

Deterministic: everything derives from ``--seed``.

Usage (from repo root)::

    uv run python rust/bench/seed.py --db /tmp/bench.db \\
        --runs 50000 --metrics-per-run 4 --history-points 100 \\
        --traces 100000 --spans-per-trace 5 --model-versions 10000 \\
        --experiments 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, insert

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mlflow.prompt.constants import IS_PROMPT_TAG_KEY, PROMPT_TYPE_TAG_KEY
from mlflow.store.model_registry.dbmodels.models import (
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelTag,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlExperiment,
    SqlLatestMetric,
    SqlMetric,
    SqlParam,
    SqlRun,
    SqlSpan,
    SqlSpanAttribute,
    SqlTag,
    SqlTraceInfo,
    SqlTraceTag,
)

# The batch size for executemany. Large enough to amortize round-trips, small
# enough that a single INSERT statement's parameter list stays sane on sqlite
# (SQLITE_MAX_VARIABLE_NUMBER) and memory stays bounded on the big scale.
_BATCH = 5000

_RUN_STATUSES = ["FINISHED", "FINISHED", "FINISHED", "RUNNING", "FAILED"]
_TRACE_STATES = ["OK", "OK", "OK", "ERROR", "IN_PROGRESS"]
_SPAN_TYPES = ["LLM", "CHAIN", "TOOL", "RETRIEVER", "AGENT"]
_MODEL_FAMILIES = ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "llama-3"]
_STAGES = ["None", "Staging", "Production", "Archived"]
_METRIC_KEYS = ["loss", "accuracy", "f1", "precision", "recall", "auc", "val_loss", "lr"]
_PARAM_KEYS = ["optimizer", "batch_size", "epochs", "lr", "dropout", "model", "seed", "dataset"]


@dataclass
class Scale:
    runs: int
    metrics_per_run: int
    history_points: int
    traces: int
    spans_per_trace: int
    model_versions: int
    experiments: int
    prompt_fraction: float
    seed: int


def _seed_schema(db_uri: str) -> None:
    """Create a migrated tracking + registry schema (runs the full Alembic chain).

    Reuses the compliance harness' seeding so the Rust server accepts the DB.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=db_uri)
    # Creating an experiment triggers ``mlflow db upgrade`` internally.
    client.create_experiment("_bench_bootstrap")


def _batched(engine, table, rows: list[dict]) -> None:
    if not rows:
        return
    stmt = insert(table)
    with engine.begin() as conn:
        for i in range(0, len(rows), _BATCH):
            conn.execute(stmt, rows[i : i + _BATCH])


def _gen_experiments(rng: random.Random, scale: Scale, now_ms: int) -> list[dict]:
    return [
        {
            "name": f"bench_exp_{i}",
            "artifact_location": f"/tmp/bench-artifacts/exp_{i}",
            "lifecycle_stage": "active",
            "creation_time": now_ms - rng.randint(0, 10_000_000),
            "last_update_time": now_ms,
        }
        for i in range(scale.experiments)
    ]


def _span_content(
    rng: random.Random, span_id: str, name: str, span_type: str
) -> tuple[str, dict[str, str]]:
    """Build the same double-encoded attribute shape produced by ``Span.to_dict``."""
    model = rng.choice(_MODEL_FAMILIES)
    attributes = {
        "mlflow.spanType": json.dumps(span_type),
        "gen_ai.request.model": json.dumps(model),
        "gen_ai.usage.input_tokens": json.dumps(rng.randint(10, 4000)),
        "gen_ai.usage.output_tokens": json.dumps(rng.randint(10, 2000)),
        "llm.temperature": json.dumps(round(rng.uniform(0.0, 1.5), 2)),
    }
    content = json.dumps({
        "span_id": span_id,
        "name": name,
        "span_type": span_type,
        "attributes": attributes,
    })
    return content, attributes


def generate(db_uri: str, scale: Scale) -> dict[str, int]:
    rng = random.Random(scale.seed)
    now_ms = int(time.time() * 1000)
    engine = create_engine(db_uri)
    counts: dict[str, int] = {}

    exp_rows = _gen_experiments(rng, scale, now_ms)
    _batched(engine, SqlExperiment.__table__, exp_rows)
    # The bootstrap experiment is id 0; ours start at 1..N. Reflect ids back.
    with engine.begin() as conn:
        exp_ids = [
            r[0]
            for r in conn.exec_driver_sql(
                "SELECT experiment_id FROM experiments WHERE name LIKE 'bench_exp_%'"
            ).fetchall()
        ]
    counts["experiments"] = len(exp_ids)

    _gen_runs(engine, rng, scale, exp_ids, now_ms, counts)
    _gen_traces(engine, rng, scale, exp_ids, now_ms, counts)
    _gen_registry(engine, rng, scale, now_ms, counts)

    engine.dispose()
    return counts


def _gen_runs(engine, rng, scale: Scale, exp_ids, now_ms, counts) -> None:
    runs, params, tags, metrics, latest = [], [], [], [], []
    n_runs = n_params = n_tags = n_metrics = n_latest = 0

    def flush() -> None:
        _batched(engine, SqlRun.__table__, runs)
        _batched(engine, SqlParam.__table__, params)
        _batched(engine, SqlTag.__table__, tags)
        _batched(engine, SqlMetric.__table__, metrics)
        _batched(engine, SqlLatestMetric.__table__, latest)
        runs.clear()
        params.clear()
        tags.clear()
        metrics.clear()
        latest.clear()

    for i in range(scale.runs):
        run_uuid = f"{rng.getrandbits(128):032x}"
        exp_id = rng.choice(exp_ids)
        start = now_ms - rng.randint(0, 30 * 24 * 3600 * 1000)
        status = rng.choice(_RUN_STATUSES)
        deleted = i % 20 == 0  # ~5% deleted, exercising lifecycle filtering
        end = start + rng.randint(1000, 3_600_000) if status != "RUNNING" else None
        runs.append({
            "run_uuid": run_uuid,
            "name": f"run_{i}",
            "source_type": "LOCAL",
            "source_name": "bench",
            "entry_point_name": "",
            "user_id": rng.choice(["alice", "bob", "carol"]),
            "status": status,
            "start_time": start,
            "end_time": end,
            "deleted_time": now_ms if deleted else None,
            "source_version": "",
            "lifecycle_stage": "deleted" if deleted else "active",
            "artifact_uri": f"/tmp/bench-artifacts/{run_uuid}/artifacts",
            "experiment_id": exp_id,
        })
        tags.append({"key": "mlflow.runName", "value": f"run_{i}", "run_uuid": run_uuid})
        tags.append({
            "key": "phase",
            "value": rng.choice(["train", "eval", "tune"]),
            "run_uuid": run_uuid,
        })
        n_tags += 2
        for pk in rng.sample(_PARAM_KEYS, k=min(len(_PARAM_KEYS), 5)):
            params.append({"key": pk, "value": str(rng.randint(1, 512)), "run_uuid": run_uuid})
            n_params += 1

        for mk in (
            _METRIC_KEYS[: scale.metrics_per_run]
            if scale.metrics_per_run <= len(_METRIC_KEYS)
            else _synth_metric_keys(scale.metrics_per_run)
        ):
            last_val, last_ts, last_step = None, None, None
            for step in range(scale.history_points):
                val = round(rng.uniform(0.0, 1.0), 6)
                ts = start + step * 1000
                metrics.append({
                    "key": mk,
                    "value": val,
                    "timestamp": ts,
                    "step": step,
                    "is_nan": False,
                    "run_uuid": run_uuid,
                })
                n_metrics += 1
                last_val, last_ts, last_step = val, ts, step
            if last_val is not None:
                latest.append({
                    "key": mk,
                    "value": last_val,
                    "timestamp": last_ts,
                    "step": last_step,
                    "is_nan": False,
                    "run_uuid": run_uuid,
                })
                n_latest += 1
        n_runs += 1
        if len(runs) >= _BATCH or len(metrics) >= _BATCH * 4:
            flush()
    flush()
    counts.update(
        runs=n_runs, params=n_params, tags=n_tags, metrics=n_metrics, latest_metrics=n_latest
    )


def _synth_metric_keys(n: int) -> list[str]:
    keys = list(_METRIC_KEYS)
    i = 0
    while len(keys) < n:
        keys.append(f"metric_{i}")
        i += 1
    return keys[:n]


def _gen_traces(engine, rng, scale: Scale, exp_ids, now_ms, counts) -> None:
    infos, ttags, spans, span_attributes = [], [], [], []
    n_traces = n_spans = n_span_attributes = 0

    def flush() -> None:
        _batched(engine, SqlTraceInfo.__table__, infos)
        _batched(engine, SqlTraceTag.__table__, ttags)
        _batched(engine, SqlSpan.__table__, spans)
        _batched(engine, SqlSpanAttribute.__table__, span_attributes)
        infos.clear()
        ttags.clear()
        spans.clear()
        span_attributes.clear()

    for i in range(scale.traces):
        request_id = f"tr-{rng.getrandbits(96):024x}"
        exp_id = rng.choice(exp_ids)
        ts = now_ms - rng.randint(0, 30 * 24 * 3600 * 1000)
        state = rng.choice(_TRACE_STATES)
        request_preview = json.dumps({"messages": [{"role": "user", "content": f"q{i}"}]})
        infos.append({
            "request_id": request_id,
            "experiment_id": exp_id,
            "timestamp_ms": ts,
            "execution_time_ms": rng.randint(5, 60_000),
            "status": state,
            "client_request_id": None,
            "request_preview": request_preview[:1000],
            "response_preview": f'{{"content": "answer {i}"}}'[:1000],
            "db_payload_generation": 0,
        })
        ttags.append({
            "key": "mlflow.traceName",
            "value": rng.choice(["chat", "rag", "agent_loop"]),
            "request_id": request_id,
        })
        n_traces += 1
        start_nano = ts * 1_000_000
        parent = None
        for s in range(scale.spans_per_trace):
            span_id = f"{rng.getrandbits(64):016x}"
            span_type = rng.choice(_SPAN_TYPES)
            name = f"{span_type.lower()}_span_{s}"
            span_start = start_nano + s * 1_000_000
            content, attributes = _span_content(rng, span_id, name, span_type)
            spans.append({
                "trace_id": request_id,
                "experiment_id": exp_id,
                "span_id": span_id,
                "parent_span_id": parent,
                "name": name,
                "type": span_type,
                "status": "OK" if state != "ERROR" else "ERROR",
                "start_time_unix_nano": span_start,
                "end_time_unix_nano": span_start + rng.randint(1000, 5_000_000),
                "content": content,
                "dimension_attributes": None,
            })
            span_attributes.extend(
                {
                    "trace_id": request_id,
                    "span_id": span_id,
                    "key": key,
                    "value": value[:500],
                    "value_truncated": len(value) > 500,
                }
                for key, value in attributes.items()
            )
            n_spans += 1
            n_span_attributes += len(attributes)
            parent = span_id
        if len(infos) >= _BATCH or len(spans) >= _BATCH * 2:
            flush()
    flush()
    counts.update(traces=n_traces, spans=n_spans, span_attributes=n_span_attributes)


def _gen_registry(engine, rng, scale: Scale, now_ms, counts) -> None:
    """Registered models + versions, with a prompt slice for the anti-join.

    Model *count* is derived so that versions/model averages ~10; a
    configurable fraction of the models are prompts (tagged
    ``mlflow.prompt.is_prompt``), which the registered-models search must
    exclude.
    """
    n_models = max(1, (scale.model_versions + 9) // 10)
    models, rm_tags, mvs, mv_tags = [], [], [], []
    n_mv = 0
    prompt_count = round(n_models * scale.prompt_fraction)
    if 0 < scale.prompt_fraction < 1 and n_models > 1:
        prompt_count = max(1, min(n_models - 1, prompt_count))
    prompt_models = set(rng.sample(range(n_models), prompt_count))
    base_versions, extra_versions = divmod(scale.model_versions, n_models)

    for m in range(n_models):
        is_prompt = m in prompt_models
        name = f"{'prompt' if is_prompt else 'model'}_{m}"
        created = now_ms - rng.randint(0, 10_000_000)
        models.append({
            "workspace": "default",
            "name": name,
            "creation_time": created,
            "last_updated_time": now_ms,
            "description": f"bench {'prompt' if is_prompt else 'model'} {m}",
        })
        if is_prompt:
            rm_tags.append({
                "workspace": "default",
                "name": name,
                "key": IS_PROMPT_TAG_KEY,
                "value": "true",
            })
        else:
            rm_tags.append({
                "workspace": "default",
                "name": name,
                "key": "owner",
                "value": rng.choice(["alice", "bob"]),
            })

        n_versions = base_versions + (m < extra_versions)
        for v in range(1, n_versions + 1):
            mvs.append({
                "workspace": "default",
                "name": name,
                "version": v,
                "creation_time": created + v,
                "last_updated_time": now_ms,
                "description": f"v{v}",
                "user_id": rng.choice(["alice", "bob"]),
                "current_stage": rng.choice(_STAGES),
                "source": f"/tmp/bench-artifacts/{name}/{v}",
                "storage_location": None,
                "run_id": None,
                "run_link": None,
                "status": "READY",
                "status_message": None,
            })
            if is_prompt:
                mv_tags.append({
                    "workspace": "default",
                    "name": name,
                    "version": v,
                    "key": IS_PROMPT_TAG_KEY,
                    "value": "true",
                })
                mv_tags.append({
                    "workspace": "default",
                    "name": name,
                    "version": v,
                    "key": PROMPT_TYPE_TAG_KEY,
                    "value": "text",
                })
            n_mv += 1
    _batched(engine, SqlRegisteredModel.__table__, models)
    _batched(engine, SqlRegisteredModelTag.__table__, rm_tags)
    _batched(engine, SqlModelVersion.__table__, mvs)
    _batched(engine, SqlModelVersionTag.__table__, mv_tags)
    n_prompts = sum(1 for t in rm_tags if t["key"] == IS_PROMPT_TAG_KEY)
    counts.update(registered_models=len(models), model_versions=n_mv, prompts=n_prompts)


def build_scale(args: argparse.Namespace) -> Scale:
    return Scale(
        runs=args.runs,
        metrics_per_run=args.metrics_per_run,
        history_points=args.history_points,
        traces=args.traces,
        spans_per_trace=args.spans_per_trace,
        model_versions=args.model_versions,
        experiments=args.experiments,
        prompt_fraction=args.prompt_fraction,
        seed=args.seed,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="SQLite path or full SQLAlchemy URI")
    p.add_argument("--runs", type=int, default=50_000)
    p.add_argument("--metrics-per-run", type=int, default=4)
    p.add_argument("--history-points", type=int, default=100)
    p.add_argument("--traces", type=int, default=100_000)
    p.add_argument("--spans-per-trace", type=int, default=5)
    p.add_argument("--model-versions", type=int, default=10_000)
    p.add_argument("--experiments", type=int, default=50)
    p.add_argument("--prompt-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metadata", help="write scale, counts, and seed timing as JSON")
    args = p.parse_args()

    if (
        min(
            args.runs,
            args.metrics_per_run,
            args.history_points,
            args.traces,
            args.spans_per_trace,
            args.model_versions,
            args.experiments,
        )
        < 1
    ):
        p.error("all scale counts must be positive")
    if not 0 <= args.prompt_fraction <= 1:
        p.error("--prompt-fraction must be between 0 and 1")

    db_uri = args.db if "://" in args.db else f"sqlite:///{Path(args.db).resolve()}"
    scale = build_scale(args)

    print(f"Seeding {db_uri}")
    print(f"  scale: {scale}")
    t0 = time.time()
    _seed_schema(db_uri)
    print(f"  schema migrated in {time.time() - t0:.1f}s")
    t1 = time.time()
    counts = generate(db_uri, scale)
    data_seconds = time.time() - t1
    total_seconds = time.time() - t0
    print(f"  data generated in {data_seconds:.1f}s")
    print(f"  total wall: {total_seconds:.1f}s")
    print("  counts:")
    for k, v in sorted(counts.items()):
        print(f"    {k}: {v:,}")
    if args.metadata:
        metadata = {
            "scale": vars(scale),
            "counts": counts,
            "schema_seconds": round(t1 - t0, 3),
            "data_seconds": round(data_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }
        Path(args.metadata).write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"  metadata: {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
