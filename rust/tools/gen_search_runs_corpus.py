"""Generate a differential corpus for `search_runs` ordering/filtering/paging (T2.6).

Run from the repo root:

    uv run --frozen python rust/tools/gen_search_runs_corpus.py

Produces:
    rust/crates/mlflow-store/tests/corpus/search_runs/search.db    (a real
        Alembic-migrated SQLite DB seeded with runs/metrics/params/tags/datasets
        designed to exercise NULL-metric orderings, ties on start_time, multi-page
        walks, and filters on every entity type)
    rust/crates/mlflow-store/tests/corpus/search_runs/cases.json   (query cases +
        the EXACT ordered run_id pages the genuine Python
        `SqlAlchemyStore._search_runs` returns, walking every page)

The Rust test (`tests/search_runs_corpus.rs`) copies `search.db` to a temp file,
runs the Rust `search_runs` over each case (walking pages via its opaque keyset
tokens), and asserts the ordered run_id sequence and the per-page boundaries match
Python's. Pagination *token contents* differ by design (Rust uses keyset tokens,
Python uses offset tokens — plan decision D3); only the page *contents* must match.

Why seed a real store: the ordering/NULLS-LAST/tiebreak/DISTINCT semantics live in
`SqlAlchemyStore._search_runs` + `_get_orderby_clauses` + `_get_sqlalchemy_filter_clauses`.
Running the genuine store end to end is the only faithful oracle.
"""

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "rust" / "crates" / "mlflow-store" / "tests" / "corpus" / "search_runs"


def build_corpus(out_dir: Path) -> int:
    import mlflow
    from mlflow.entities import (
        Dataset,
        DatasetInput,
        InputTag,
        Metric,
        Param,
        RunTag,
        ViewType,
    )
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "search.db"
    if db_path.exists():
        db_path.unlink()

    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    store = SqlAlchemyStore(uri, default_artifact_root=str(out_dir / "artifacts"))

    exp_a = store.create_experiment("search_a")
    exp_b = store.create_experiment("search_b")

    # Deterministic seed data. We deliberately create:
    #  * ties on start_time (several runs share the same start_time) so the
    #    run_uuid ASC tiebreak is exercised;
    #  * runs MISSING the "acc" metric (NULL-metric ordering -> rank 2);
    #  * a run whose "acc" metric is NaN (rank 1);
    #  * params/tags present on some runs only (NULL param/tag ordering);
    #  * dataset inputs with name/digest/context tags for dataset filters.
    #
    # Each entry: (exp, name, start_time, {metrics}, {params}, {tags}, dataset_or_None)
    # metric value None -> metric absent; float("nan") -> NaN metric.
    Row = tuple
    specs: list[Row] = [
        (exp_a, "r00", 100, {"acc": 0.90, "loss": 0.5}, {"model": "lr"}, {"phase": "train"}, ("d1", "abc", "train")),
        (exp_a, "r01", 100, {"acc": 0.95, "loss": 0.4}, {"model": "rf"}, {"phase": "train"}, ("d1", "abc", "train")),
        (exp_a, "r02", 100, {"acc": 0.95, "loss": 0.3}, {"model": "rf"}, {"phase": "eval"}, ("d2", "def", "eval")),
        (exp_a, "r03", 200, {"acc": None, "loss": 0.6}, {"model": "lr"}, {}, None),
        (exp_a, "r04", 200, {"acc": float("nan"), "loss": 0.2}, {"model": "xgb"}, {"phase": "eval"}, ("d2", "def", "eval")),
        (exp_a, "r05", 300, {"acc": 0.80}, {}, {"phase": "train"}, ("d3", "ghi", "train")),
        (exp_a, "r06", 300, {"acc": 0.85, "loss": 0.7}, {"model": "lr"}, {"phase": "prod"}, None),
        (exp_a, "r07", 300, {"loss": 0.9}, {"model": "rf"}, {}, None),
        (exp_b, "r08", 150, {"acc": 0.70, "loss": 0.1}, {"model": "lr"}, {"phase": "train"}, ("d1", "abc", "train")),
        (exp_b, "r09", 150, {"acc": 0.99}, {"model": "rf"}, {"phase": "prod"}, ("d4", "jkl", "prod")),
        (exp_b, "r10", 400, {"acc": 0.60, "loss": 0.8}, {}, {"phase": "eval"}, None),
        (exp_b, "r11", 400, {"acc": None}, {"model": "xgb"}, {"phase": "eval"}, ("d2", "def", "eval")),
    ]

    # name -> run_id (also used to delete some to exercise view_type).
    name_to_id: dict[str, str] = {}
    for exp, name, start_time, metrics, params, tags, dataset in specs:
        run = store.create_run(exp, user_id="u", start_time=start_time, tags=[RunTag("mlflow.runName", name)], run_name=name)
        rid = run.info.run_id
        name_to_id[name] = rid
        ms = []
        for k, v in metrics.items():
            if v is None:
                continue
            ms.append(Metric(k, v, timestamp=start_time, step=0))
        ps = [Param(k, v) for k, v in params.items()]
        ts = [RunTag(k, v) for k, v in tags.items()]
        if ms or ps or ts:
            store.log_batch(rid, metrics=ms, params=ps, tags=ts)
        if dataset is not None:
            dname, ddigest, dcontext = dataset
            ds = Dataset(name=dname, digest=ddigest, source_type="local", source="path")
            di = DatasetInput(dataset=ds, tags=[InputTag("mlflow.data.context", dcontext)])
            store.log_inputs(rid, [di])

    # Soft-delete two runs to exercise view_type filtering.
    store.delete_run(name_to_id["r06"])
    store.delete_run(name_to_id["r11"])

    exp_all = [exp_a, exp_b]

    view_map = {
        "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
        "DELETED_ONLY": ViewType.DELETED_ONLY,
        "ALL": ViewType.ALL,
    }

    # (label, experiment_ids, filter, order_by, view, page_size)
    raw_cases = [
        ("default_order", exp_all, "", [], "ACTIVE_ONLY", 3),
        ("default_order_all_exps_full", exp_all, "", [], "ACTIVE_ONLY", 100),
        ("order_metric_acc_asc", exp_all, "", ["metrics.acc ASC"], "ACTIVE_ONLY", 4),
        ("order_metric_acc_desc", exp_all, "", ["metrics.acc DESC"], "ACTIVE_ONLY", 4),
        ("order_metric_loss_asc", exp_all, "", ["metrics.loss ASC"], "ACTIVE_ONLY", 5),
        ("order_start_time_asc", exp_all, "", ["attribute.start_time ASC"], "ACTIVE_ONLY", 3),
        ("order_start_time_desc", exp_all, "", ["attribute.start_time DESC"], "ACTIVE_ONLY", 3),
        ("order_param_model", exp_all, "", ["params.model ASC"], "ACTIVE_ONLY", 3),
        ("order_tag_phase", exp_all, "", ["tags.phase DESC"], "ACTIVE_ONLY", 3),
        ("order_name_asc", exp_all, "", ["attribute.run_name ASC"], "ACTIVE_ONLY", 4),
        ("order_multi_metric_then_start", exp_all, "", ["metrics.acc DESC", "attribute.start_time ASC"], "ACTIVE_ONLY", 2),
        ("filter_metric_acc_ge", exp_all, "metrics.acc >= 0.85", [], "ACTIVE_ONLY", 3),
        ("filter_metric_acc_ge_order", exp_all, "metrics.acc >= 0.85", ["metrics.acc DESC"], "ACTIVE_ONLY", 2),
        ("filter_param_model_eq", exp_all, "params.model = 'lr'", [], "ACTIVE_ONLY", 2),
        ("filter_param_model_like", exp_all, "params.model LIKE 'r%'", [], "ACTIVE_ONLY", 3),
        ("filter_tag_phase_eq", exp_all, "tags.phase = 'train'", [], "ACTIVE_ONLY", 3),
        ("filter_tag_is_null", exp_all, "tags.phase IS NULL", [], "ACTIVE_ONLY", 3),
        ("filter_param_is_not_null", exp_all, "params.model IS NOT NULL", [], "ACTIVE_ONLY", 4),
        ("filter_attr_run_name_like", exp_all, "attribute.run_name LIKE 'r0%'", [], "ACTIVE_ONLY", 3),
        ("filter_attr_start_time_gt", exp_all, "attribute.start_time > 150", [], "ACTIVE_ONLY", 3),
        ("filter_multi_and", exp_all, "metrics.loss < 0.7 and params.model = 'lr'", [], "ACTIVE_ONLY", 2),
        ("filter_dataset_name", exp_all, "dataset.name = 'd1'", [], "ACTIVE_ONLY", 3),
        ("filter_dataset_digest", exp_all, "dataset.digest = 'def'", [], "ACTIVE_ONLY", 3),
        ("filter_dataset_context", exp_all, "dataset.context = 'train'", [], "ACTIVE_ONLY", 3),
        ("filter_dataset_name_in", exp_all, "dataset.name IN ('d1', 'd4')", [], "ACTIVE_ONLY", 3),
        ("view_deleted_only", exp_all, "", [], "DELETED_ONLY", 5),
        ("view_all_order_start", exp_all, "", ["attribute.start_time DESC"], "ALL", 4),
        ("single_exp_a", [exp_a], "", ["metrics.acc ASC"], "ACTIVE_ONLY", 3),
        ("filter_metric_and_order_tie", exp_all, "metrics.loss IS NOT NULL", ["metrics.acc DESC"], "ACTIVE_ONLY", 3),
        ("empty_result", exp_all, "metrics.acc > 100", [], "ACTIVE_ONLY", 5),
        ("order_end_time_desc", exp_all, "", ["attribute.end_time DESC"], "ALL", 4),
    ]

    cases = []
    for label, exp_ids, filt, order_by, view, page_size in raw_cases:
        vt = view_map[view]
        # Full pagination walk: collect ordered run_ids page by page.
        pages: list[list[str]] = []
        all_ids: list[str] = []
        token = None
        # Guard against runaway loops.
        for _ in range(1000):
            runs, token = store._search_runs(
                experiment_ids=[str(e) for e in exp_ids],
                filter_string=filt,
                run_view_type=vt,
                max_results=page_size,
                order_by=order_by,
                page_token=token,
            )
            page_ids = [r.info.run_id for r in runs]
            pages.append(page_ids)
            all_ids.extend(page_ids)
            if not token:
                break
        cases.append(
            {
                "label": label,
                "experiment_ids": [str(e) for e in exp_ids],
                "filter": filt,
                "order_by": order_by,
                "view_type": view,
                "max_results": page_size,
                "pages": pages,
                "ordered_run_ids": all_ids,
            }
        )

    (out_dir / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")

    with sqlite3.connect(db_path) as conn:
        (head,) = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    print(f"alembic head: {head}")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    n = build_corpus(args.out_dir)
    print(f"Wrote {n} search_runs cases + search.db to {args.out_dir}")


if __name__ == "__main__":
    main()
