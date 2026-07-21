"""Generate a differential corpus for registry search (T7.3).

Run from the repo root:

    uv run python rust/tools/gen_registry_search_corpus.py

Produces:
    rust/crates/mlflow-registry/tests/corpus/search/registry.db   (a real
        Alembic-migrated SQLite DB seeded — through the GENUINE Python
        model-registry `SqlAlchemyStore` — with registered models, model
        versions, tags, aliases, and prompt-tagged rows designed to exercise
        name/tag filters, MV version_number/run_id-IN/source aliases, AND-of-tags,
        prompt inclusion/exclusion, order_by variants + tiebreaks, deleted-MV
        visibility, and multi-page walks.)
    rust/crates/mlflow-registry/tests/corpus/search/cases.json    (query cases +
        the EXACT ordered result pages the genuine Python store returns, walking
        every page via its offset page tokens.)

The Rust test (`tests/search_corpus.rs`) copies `registry.db` to a temp file,
runs the Rust `search_registered_models` / `search_model_versions` over each case
(walking pages via its offset tokens), and asserts the ordered identifier
sequence, the per-page boundaries, AND the page-token contents match Python's.
Registry search keeps Python's offset tokens (plan T7.3), so tokens must match
byte-for-byte too.

Why seed a real store: the filter → SQL, prompt anti-join, AND-of-tags
HAVING-count subquery, order_by tiebreaks, and offset pagination all live in the
genuine `SqlAlchemyStore` (model-registry). Running it end to end is the only
faithful oracle.
"""

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "rust" / "crates" / "mlflow-registry" / "tests" / "corpus" / "search"

IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"


def build_corpus(out_dir: Path) -> tuple[int, int]:
    from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
    from mlflow.store.db.utils import _initialize_tables
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "registry.db"
    if db_path.exists():
        db_path.unlink()

    uri = f"sqlite:///{db_path}"
    store = SqlAlchemyStore(uri)
    # `SqlAlchemyStore.__init__` already runs the migrations, but be explicit.
    _initialize_tables(store.engine)

    # ---- Seed registered models (some prompts) --------------------------------
    # Each: (name, {tags}, is_prompt)
    rm_specs = [
        ("alpha", {"team": "ml", "stage": "prod"}, False),
        ("beta", {"team": "ml", "stage": "dev"}, False),
        ("gamma", {"team": "data"}, False),
        ("delta", {"team": "ml", "stage": "prod", "extra": "x"}, False),
        ("epsilon", {}, False),
        ("prompt_one", {"team": "ml"}, True),
        ("prompt_two", {"stage": "prod"}, True),
    ]

    for name, tags, is_prompt in rm_specs:
        rm_tags = [RegisteredModelTag(k, v) for k, v in tags.items()]
        if is_prompt:
            rm_tags.append(RegisteredModelTag(IS_PROMPT_TAG_KEY, "true"))
        store.create_registered_model(name, tags=rm_tags)

    # ---- Seed model versions --------------------------------------------------
    # Each: (rm_name, source, run_id, {tags})
    mv_specs = [
        ("alpha", "path/to/alpha/1", "run_alpha_1", {"quality": "high"}),
        ("alpha", "path/to/alpha/2", "run_alpha_2", {"quality": "low"}),
        ("alpha", "s3://bucket/alpha/3", "run_alpha_3", {"quality": "high", "vip": "yes"}),
        ("beta", "path/to/beta/1", "run_beta_1", {"quality": "high"}),
        ("beta", "path/to/beta/2", None, {}),
        ("gamma", "path/to/gamma/1", "run_gamma_1", {"quality": "low"}),
        ("delta", "s3://bucket/delta/1", "run_delta_1", {"quality": "high"}),
        ("epsilon", "path/to/epsilon/1", "run_epsilon_1", {}),
        ("prompt_one", "path/to/prompt_one/1", "run_prompt_1", {}),
        ("prompt_two", "path/to/prompt_two/1", "run_prompt_2", {}),
    ]

    for name, source, run_id, tags in mv_specs:
        mv_tags = [ModelVersionTag(k, v) for k, v in tags.items()]
        store.create_model_version(name, source, run_id=run_id, tags=mv_tags)

    # Soft-delete one version to exercise deleted-MV invisibility.
    store.delete_model_version("beta", "2")

    # ---- Registered-model search cases ---------------------------------------
    # (label, filter, order_by, max_results)
    rm_cases = [
        ("default_all", None, None, 3),
        ("default_all_full", None, None, 100),
        ("name_eq", "name = 'alpha'", None, 5),
        ("name_like", "name LIKE 'a%'", None, 5),
        ("name_ilike", "name ILIKE 'A%'", None, 5),
        ("name_neq", "name != 'alpha'", None, 3),
        ("tag_eq", "tags.team = 'ml'", None, 3),
        ("tag_like", "tags.stage LIKE 'pro%'", None, 3),
        ("tag_and", "tags.team = 'ml' AND tags.stage = 'prod'", None, 3),
        ("tag_and_three", "tags.team = 'ml' AND tags.stage = 'prod' AND tags.extra = 'x'", None, 3),
        ("order_name_desc", None, ["name DESC"], 3),
        ("order_last_updated_asc", None, ["last_updated_timestamp ASC"], 3),
        ("order_timestamp_desc", None, ["timestamp DESC"], 4),
        # Prompt handling: default excludes prompts; explicit query includes them.
        ("prompt_default_excluded", None, None, 100),
        ("prompt_is_true", f"tags.`{IS_PROMPT_TAG_KEY}` = 'true'", None, 100),
        ("prompt_neq_false", f"tags.`{IS_PROMPT_TAG_KEY}` != 'false'", None, 100),
        ("prompt_eq_false", f"tags.`{IS_PROMPT_TAG_KEY}` = 'false'", None, 100),
        ("prompt_neq_true", f"tags.`{IS_PROMPT_TAG_KEY}` != 'true'", None, 100),
        ("empty_result", "name = 'nope'", None, 5),
    ]

    rm_out = []
    for label, filt, order_by, max_results in rm_cases:
        pages, tokens, ordered = walk_rm(store, filt, order_by, max_results)
        rm_out.append({
            "label": label,
            "filter": filt,
            "order_by": order_by or [],
            "max_results": max_results,
            "pages": pages,
            "page_tokens": tokens,
            "ordered_names": ordered,
        })

    # ---- Model-version search cases ------------------------------------------
    mv_cases = [
        ("default_all", None, None, 4),
        ("default_all_full", None, None, 100),
        ("name_eq", "name = 'alpha'", None, 5),
        ("name_like", "name LIKE 'a%'", None, 5),
        ("version_eq", "version_number = 1", None, 100),
        ("version_gt", "version_number > 1", None, 100),
        ("run_id_eq", "run_id = 'run_alpha_1'", None, 5),
        ("run_id_in", "run_id IN ('run_alpha_1', 'run_beta_1')", None, 5),
        ("source_path_eq", "source_path = 'path/to/alpha/1'", None, 5),
        ("source_path_like", "source_path LIKE 's3://%'", None, 5),
        ("tag_eq", "tags.quality = 'high'", None, 3),
        ("tag_and", "tags.quality = 'high' AND tags.vip = 'yes'", None, 3),
        ("order_name_version", None, ["name ASC", "version_number ASC"], 3),
        ("order_version_desc", None, ["version_number DESC"], 4),
        ("order_creation_asc", None, ["creation_timestamp ASC"], 4),
        ("name_and_version", "name = 'alpha' AND version_number >= 2", None, 5),
        # Deleted MV must not appear (beta/2 soft-deleted).
        ("beta_versions", "name = 'beta'", None, 5),
        # Prompt handling on MVs.
        ("prompt_default_excluded", None, None, 100),
        ("prompt_is_true", f"tags.`{IS_PROMPT_TAG_KEY}` = 'true'", None, 100),
        ("prompt_neq_false", f"tags.`{IS_PROMPT_TAG_KEY}` != 'false'", None, 100),
        ("empty_result", "name = 'nope'", None, 5),
    ]

    mv_out = []
    for label, filt, order_by, max_results in mv_cases:
        pages, tokens, ordered = walk_mv(store, filt, order_by, max_results)
        mv_out.append({
            "label": label,
            "filter": filt,
            "order_by": order_by or [],
            "max_results": max_results,
            "pages": pages,
            "page_tokens": tokens,
            "ordered_ids": ordered,
        })

    (out_dir / "cases.json").write_text(
        json.dumps({"registered_models": rm_out, "model_versions": mv_out}, indent=2) + "\n"
    )

    with sqlite3.connect(db_path) as conn:
        (head,) = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    print(f"alembic head: {head}")
    return len(rm_out), len(mv_out)


def walk_rm(store, filt, order_by, max_results):
    """Full pagination walk of search_registered_models; record names + tokens."""
    pages: list[list[str]] = []
    tokens: list[str | None] = []
    ordered: list[str] = []
    token = None
    for _ in range(1000):
        res = store.search_registered_models(
            filter_string=filt,
            max_results=max_results,
            order_by=order_by,
            page_token=token,
        )
        page = [rm.name for rm in res]
        pages.append(page)
        ordered.extend(page)
        token = res.token
        # `create_page_token` returns bytes; normalize to a str for JSON + the
        # Rust comparison (Rust's `create_page_token` yields the same ASCII).
        tokens.append(token.decode("ascii") if isinstance(token, bytes) else token)
        if not token:
            break
    return pages, tokens, ordered


def walk_mv(store, filt, order_by, max_results):
    """Full pagination walk of search_model_versions; record name/version + tokens."""
    pages: list[list[str]] = []
    tokens: list[str | None] = []
    ordered: list[str] = []
    token = None
    for _ in range(1000):
        res = store.search_model_versions(
            filter_string=filt,
            max_results=max_results,
            order_by=order_by,
            page_token=token,
        )
        page = [f"{mv.name}/{mv.version}" for mv in res]
        pages.append(page)
        ordered.extend(page)
        token = res.token
        # `create_page_token` returns bytes; normalize to a str for JSON + the
        # Rust comparison (Rust's `create_page_token` yields the same ASCII).
        tokens.append(token.decode("ascii") if isinstance(token, bytes) else token)
        if not token:
            break
    return pages, tokens, ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    n_rm, n_mv = build_corpus(args.out_dir)
    print(f"Wrote {n_rm} RM + {n_mv} MV search cases + registry.db to {args.out_dir}")


if __name__ == "__main__":
    main()
