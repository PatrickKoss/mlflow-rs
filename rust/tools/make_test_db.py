"""Generate a real Alembic-migrated SQLite tracking DB fixture for mlflow-store tests.

Run from the repo root:

    uv run --frozen python rust/tools/make_test_db.py

Produces (by default):
    rust/crates/mlflow-store/tests/fixtures/tracking.db

The fixture is a fully migrated MLflow tracking database at the current Alembic
head (see RUST_TRACKING_SERVER_PLAN.md §5.4, head ``c4a9b7d3e812``). It is
created by pointing the MLflow client at a fresh SQLite file and creating an
experiment, which triggers ``_initialize_tables`` -> ``_upgrade_db`` and runs
the whole migration chain. A little data is written so the Rust tests exercise
reads against non-empty tables.

The Rust integration test (``tests/connect_verify.rs``) checks in the generated
file and uses it directly; regenerate with the command above whenever the
Alembic head changes.
"""

import argparse
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "rust" / "crates" / "mlflow-store" / "tests" / "fixtures" / "tracking.db"


def build_db(out_path: Path) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    uri = f"sqlite:///{out_path}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # Creating an experiment runs the migrations and populates a few tables.
    exp_id = client.create_experiment("rust_store_fixture")
    client.set_experiment_tag(exp_id, "team", "rust")
    run = client.create_run(exp_id, tags={"mlflow.runName": "fixture-run"})
    client.log_param(run.info.run_id, "alpha", "0.1")
    client.log_metric(run.info.run_id, "accuracy", 0.99, step=1)
    client.log_metric(run.info.run_id, "accuracy", 0.995, step=2)
    client.set_tag(run.info.run_id, "phase", "test")
    client.set_terminated(run.info.run_id)

    with sqlite3.connect(out_path) as conn:
        (head,) = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    head = build_db(args.out)
    print(f"Wrote migrated SQLite fixture to {args.out} (alembic head: {head})")


if __name__ == "__main__":
    main()
