"""Generate a differential corpus for metric-history interval sampling (T2.7).

Run from the repo root:

    uv run --frozen python rust/tools/gen_sampling_cases.py

Produces:
    rust/crates/mlflow-store/tests/corpus/sampling/sampling.db      (a real
        Alembic-migrated SQLite DB seeded with dense metric histories)
    rust/crates/mlflow-store/tests/corpus/sampling/cases.json       (query cases
        + the EXACT output of the genuine Python
        ``SqlAlchemyStore.get_metric_history_bulk_interval`` for each)

The Rust test (``tests/sampling_corpus.rs``) copies ``sampling.db`` to a temp
file, calls the Rust ``get_metric_history_bulk_interval`` with each case's
params, and asserts byte-for-byte equality against ``cases.json`` — this is the
"identical sampled point sets vs Python on dense histories" acceptance check.

Why seed a real store instead of importing a pure helper: the sampling algorithm
lives in ``SqlAlchemyStore.get_metric_history_bulk_interval`` (and its SQL-based
step discovery), NOT in a standalone function. Running the actual store end to
end is the only faithful oracle.

Edge cases covered: dense (>2500 points), multiple runs, sparse/irregular steps,
duplicate steps across runs, single point, exactly-at-cap, explicit
start/end_step windows (including a window that clamps out most data), and a
window with start_step == end_step.
"""

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "rust" / "crates" / "mlflow-store" / "tests" / "corpus" / "sampling"


def _metric_to_dict(m):
    """Serialize a MetricWithRunId the way the Rust test compares it.

    ``MetricWithRunId`` is a ``Metric`` subclass exposing ``.run_id``.
    """
    value = m.value
    # NaN/Inf are not JSON-representable; encode sentinels the Rust side maps back.
    if value != value:  # noqa: PLR0124  (NaN check)
        value = "NaN"
    elif value == float("inf"):
        value = "Infinity"
    elif value == float("-inf"):
        value = "-Infinity"
    return {
        "run_id": m.run_id,
        "key": m.key,
        "value": value,
        "timestamp": m.timestamp,
        "step": m.step,
    }


def build_corpus(out_dir: Path) -> int:
    import mlflow
    from mlflow.entities import Metric
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "sampling.db"
    if db_path.exists():
        db_path.unlink()

    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    store = SqlAlchemyStore(uri, default_artifact_root=str(out_dir / "artifacts"))

    exp_id = store.create_experiment("sampling")

    # --- Seed several runs with distinct step distributions. ---
    # run_dense: 3000 contiguous steps 0..2999 (well over MAX_RESULTS_PER_RUN).
    # run_dense2: 3000 steps but shifted values, to exercise multi-run merge.
    # run_sparse: irregular gaps.
    # run_single: one point.
    # run_dupsteps: repeated steps (multiple metrics at same step).
    runs = {}

    def make_run(name):
        r = store.create_run(exp_id, user_id="u", start_time=0, tags=[], run_name=name)
        runs[name] = r.info.run_id
        return r.info.run_id

    def log_metrics(run_id, metrics):
        # log_batch caps at 1000 metrics per request; chunk to stay under it.
        for i in range(0, len(metrics), 1000):
            store.log_batch(run_id, metrics=metrics[i : i + 1000], params=[], tags=[])

    key = "loss"

    rid_dense = make_run("dense")
    log_metrics(rid_dense, [Metric(key, float(s) * 0.5, 1000 + s, s) for s in range(3000)])

    rid_dense2 = make_run("dense2")
    log_metrics(rid_dense2, [Metric(key, 100.0 - float(s) * 0.1, 2000 + s, s) for s in range(3000)])

    rid_sparse = make_run("sparse")
    sparse_steps = [0, 1, 5, 9, 10, 50, 51, 100, 250, 999, 2500, 2999]
    log_metrics(rid_sparse, [Metric(key, float(s), 500 + s, s) for s in sparse_steps])

    rid_single = make_run("single")
    log_metrics(rid_single, [Metric(key, 42.0, 7, 7)])

    rid_dup = make_run("dupsteps")
    # Two metrics per step for steps 0..99, different timestamps/values.
    dup_metrics = []
    for s in range(100):
        dup_metrics.append(Metric(key, float(s), 10 + s, s))
        dup_metrics.append(Metric(key, float(s) + 0.25, 20 + s, s))
    log_metrics(rid_dup, dup_metrics)

    rid_atcap = make_run("atcap")
    # Exactly MAX_RESULTS_PER_RUN (2500) distinct steps.
    log_metrics(rid_atcap, [Metric(key, float(s), s, s) for s in range(2500)])

    # --- Query cases. ---
    raw_cases = [
        # single dense run, default range, default max_results (2500)
        {"run_names": ["dense"], "max_results": 2500, "start_step": None, "end_step": None},
        # dense run, small max_results forces heavy sampling
        {"run_names": ["dense"], "max_results": 10, "start_step": None, "end_step": None},
        {"run_names": ["dense"], "max_results": 100, "start_step": None, "end_step": None},
        {"run_names": ["dense"], "max_results": 1, "start_step": None, "end_step": None},
        # explicit window inside the dense range
        {"run_names": ["dense"], "max_results": 50, "start_step": 500, "end_step": 1500},
        # window that clamps out almost everything
        {"run_names": ["dense"], "max_results": 50, "start_step": 2990, "end_step": 3000},
        # start == end window
        {"run_names": ["dense"], "max_results": 50, "start_step": 100, "end_step": 100},
        # multi-run merge, dense + dense2
        {"run_names": ["dense", "dense2"], "max_results": 20, "start_step": None, "end_step": None},
        # multi-run with differing distributions incl. sparse & single
        {
            "run_names": ["dense", "sparse", "single"],
            "max_results": 30,
            "start_step": None,
            "end_step": None,
        },
        # sparse only
        {"run_names": ["sparse"], "max_results": 5, "start_step": None, "end_step": None},
        {"run_names": ["sparse"], "max_results": 100, "start_step": None, "end_step": None},
        # single point
        {"run_names": ["single"], "max_results": 2500, "start_step": None, "end_step": None},
        # duplicate steps (2 metrics per step): sampling on distinct steps
        {"run_names": ["dupsteps"], "max_results": 10, "start_step": None, "end_step": None},
        {"run_names": ["dupsteps"], "max_results": 2500, "start_step": None, "end_step": None},
        # exactly at cap
        {"run_names": ["atcap"], "max_results": 2500, "start_step": None, "end_step": None},
        {"run_names": ["atcap"], "max_results": 2499, "start_step": None, "end_step": None},
        # explicit window on multi-run
        {
            "run_names": ["dense", "dense2"],
            "max_results": 40,
            "start_step": 0,
            "end_step": 200,
        },
    ]

    cases = []
    for c in raw_cases:
        run_ids = [runs[n] for n in c["run_names"]]
        result = store.get_metric_history_bulk_interval(
            run_ids=run_ids,
            metric_key=key,
            max_results=c["max_results"],
            start_step=c["start_step"],
            end_step=c["end_step"],
        )
        cases.append(
            {
                "run_ids": run_ids,
                "metric_key": key,
                "max_results": c["max_results"],
                "start_step": c["start_step"],
                "end_step": c["end_step"],
                "expected": [_metric_to_dict(m) for m in result],
            }
        )

    (out_dir / "cases.json").write_text(json.dumps(cases, indent=2))

    # Sanity: report DB head so the Rust connect step (head verification) works.
    with sqlite3.connect(db_path) as conn:
        (head,) = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    print(f"alembic head: {head}")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    n = build_corpus(args.out_dir)
    print(f"Wrote {n} sampling cases + sampling.db to {args.out_dir}")


if __name__ == "__main__":
    main()
