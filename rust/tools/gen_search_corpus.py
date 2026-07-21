"""Generate the search-filter/order-by parity corpus for `mlflow-search`.

Run with:

    uv run --frozen python rust/tools/gen_search_corpus.py

This imports the *actual* MLflow Python parser classes from
``mlflow/utils/search_utils.py`` and feeds a curated list of inputs (valid and
invalid, covering every in-scope domain) through ``parse_search_filter`` /
``parse_order_by_*``. Each result is recorded as either the parsed structure or
the exact ``MlflowException`` message + error code. The output JSON files under
``rust/crates/mlflow-search/tests/corpus/`` are the source of truth that the
Rust ``tests/corpus_replay.rs`` test replays and asserts byte-for-byte equal.

Determinism note: Python ``set`` reprs are order-dependent, and a handful of
MLflow error messages interpolate a ``set``/``dict_keys`` directly (e.g.
"Valid values are {...}"). We normalize those to a stable, sorted form on
*both* sides (here and in the Rust replay) so the corpus is reproducible; see
``_normalize`` below and the matching normalization in the Rust test.
"""

import base64
import json
import re
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import (
    SearchExperimentsUtils,
    SearchModelUtils,
    SearchModelVersionUtils,
    SearchTraceUtils,
    SearchUtils,
)

CORPUS_DIR = Path(__file__).resolve().parents[1] / "crates" / "mlflow-search" / "tests" / "corpus"


def _normalize(msg: str) -> str:
    """Normalize Python set/dict_keys reprs embedded in error messages.

    Any ``{...}`` or ``dict_keys([...])`` blob is replaced with a token whose
    contents are sorted, so the corpus does not depend on Python set-iteration
    order. The Rust side applies the identical normalization before comparing.
    """

    def sort_set(m: str) -> str:
        inner = m.group(1)
        # Split on commas at the top level (the elements are simple quoted
        # strings or bare words, no nested braces in these messages).
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        parts.sort()
        return "{" + ", ".join(parts) + "}"

    # dict_keys([...]) -> sorted {|...|}
    def sort_dict_keys(m: str) -> str:
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        parts.sort()
        return "dict_keys([" + ", ".join(parts) + "])"

    msg = re.sub(r"dict_keys\(\[(.*?)\]\)", sort_dict_keys, msg)
    return re.sub(r"\{(.*?)\}", sort_set, msg)


def _run(fn, arg):
    try:
        result = fn(arg)
    except MlflowException as e:
        return {
            "ok": False,
            "error_code": e.error_code,
            "message": _normalize(str(e.message)),
        }
    except Exception as e:  # pragma: no cover - flags unexpected non-MLflow errors
        return {
            "ok": False,
            "error_code": "PYTHON_" + type(e).__name__,
            "message": _normalize(str(e)),
        }
    return {"ok": True, "value": result}


def filter_case(fn):
    return lambda s: _run(fn, s)


def order_by_case(fn):
    # order_by parsers return a tuple; JSON-encode as a list for stable compare.
    def wrapped(s):
        r = _run(fn, s)
        if r["ok"] and isinstance(r["value"], tuple):
            r["value"] = list(r["value"])
        return r

    return wrapped


# ---------------------------------------------------------------------------
# Curated input lists.
# ---------------------------------------------------------------------------

# Shared filter inputs exercised against every domain's parse_search_filter.
COMMON_FILTERS = [
    "",
    "metric.acc >= 0.94",
    "metric.acc>=100",
    "metrics.acc >= 0.94",
    "params.m != 'tf'",
    "params.m!='tf'",
    'params."m" != "tf"',
    'metric."legit name" >= 0.243',
    "metrics.XYZ = 3",
    'params."cat dog" = "pets"',
    'metrics."X-Y-Z" = 3',
    'metrics."X//Y#$$@&Z" = 3',
    "params.model = 'LinearRegression'",
    "metrics.rmse < 1 and params.model_class = 'LR'",
    "`metric`.a >= 0.1",
    "`params`.model >= 'LR'",
    "tags.version = 'commit-hash'",
    "`tags`.source_name = 'a notebook'",
    'metrics."accuracy.2.0" > 5',
    "metrics.`spacey name` > 5",
    'params."p.a.r.a.m" != "a"',
    'tags."t.a.g" = "a"',
    "attribute.artifact_uri = '1/23/4'",
    "attribute.start_time >= 1234",
    "run.status = 'RUNNING'",
    "dataset.name = 'my_dataset'",
    "dataset.digest = 'abc123'",
    "dataset.context = 'train'",
    "tags.version IS NULL",
    "tags.version IS NOT NULL",
    "params.lr IS NULL",
    "params.lr IS NOT NULL",
    "tags.a IS NULL AND params.b = 'val'",
    "params.m = 'LR'",
    'params.m = "LR"',
    'params.m = "L\'Hosp"',
    "name = 'foo'",
    "name != 'foo'",
    "name LIKE '%foo%'",
    "name ILIKE '%foo%'",
    "version_number > 5",
    "version_number = 3",
    "run_id = 'abc123'",
    "run_id IN ('abc123', 'def456')",
    "run_id NOT IN ('abc123')",
    "attribute.run_id IN ('abc123def456abc123def456abc1234f', 'bcd')",
    "attribute.run_id IN ('ABCUPPER', 'lower123')",
    "source_path = '/tmp/x'",
    "status = 'READY'",
    "creation_timestamp > 100",
    "last_updated_timestamp < 999",
    "creation_time > 100",
    "last_update_time < 999",
    "model_id = 'm-123'",
    "model_type = 'agent'",
    "source_run_id = 'r-1'",
    "timestamp > 1000",
    "timestamp_ms >= 5",
    "execution_time < 500",
    "end_time = 123",
    "attributes.status = 'OK'",
    "trace.status = 'OK'",
    "metadata.mlflow.sourceRun = 'r1'",
    "feedback.quality > 0.8",
    "feedback.correctness = 'yes'",
    "expectation.expected = 'foo'",
    "span.name = 'llm'",
    "span.type = 'LLM'",
    "span.content LIKE '%hello%'",
    "span.attributes.model = 'gpt'",
    "issue.id = 'i-1'",
    # Invalid / error inputs
    "metric.acc >= 0.94; metrics.rmse < 1",
    "m.acc >= 0.94",
    "acc >= 0.94",
    "p.model >= 'LR'",
    "attri.x != 1",
    "a.x != 1",
    "model >= 'LR'",
    "metrics.A > 0.1 OR params.B = 'LR'",
    "metrics.A > 0.1 NAND params.B = 'LR'",
    "metrics.A > 0.1 AND (params.B = 'LR')",
    "`metrics.A > 0.1",
    "param`.A > 0.1",
    "`dummy.A > 0.1",
    "dummy`.A > 0.1",
    "attribute.start != 1",
    "attribute.experiment_id != 1",
    "attribute.lifecycle_stage = 'ACTIVE'",
    "attribute.name != 1",
    "attribute.time != 1",
    "attribute._status != 'RUNNING'",
    "attribute.status = true",
    "dataset.status = 'true'",
    "dataset.profile = 'num_rows: 10'",
    "metrics.acc IS NULL",
    "attribute.status IS NULL",
    "metric.model = 'LR'",
    "metric.model = '5'",
    "params.acc = 5",
    "tags.acc = 5",
    "metrics.acc != metrics.acc",
    "1.0 > metrics.acc",
    "attribute.status = 1",
    "params.acc = LR",
    "tags.acc = LR",
    "params.acc = `LR`",
    "params.'acc = LR",
    "params.acc = 'LR",
    "params.acc = LR'",
    "params.acc = \"LR'",
    "tags.acc = \"LR'",
    "tags.acc = = 'LR'",
    "attribute.status IS 'RUNNING'",
    "params.acc LR !=",
    "params.acc LR",
    "metric.acc !=",
    "acc != 1.0",
    "foo is null",
    "1=1",
    "1==2",
    "metrics.foo > 1 extra",
    "foo bar",
    "metrics.foo >",
    "attribute.run_id IN ()",
    "attribute.run_id IN ('a', 5)",
    "metrics.日本 = 3",
    "tags.日本 = 'ok'",
    "params.`a b` = 'v'",
    "metrics.foo = 1e10",
    "metrics.foo = -5",
    "metrics.foo = .5",
    "metrics.foo = 0.0",
    "metrics.m <= 1.5",
    "metrics.foo LIKE '%x%'",
    "metrics.foo IN ('a')",
    "params.p RLIKE 'x'",
    "tags.t RLIKE 'x'",
    "attribute.status RLIKE 'x'",
    # --- Additional breadth: quoting variants ---
    "params.`key with spaces` = 'v'",
    "params.\"double.quoted.key\" = 'v'",
    "tags.`mlflow.prompt.is_prompt` = 'true'",
    "tags.`mlflow.prompt.is_prompt` != 'true'",
    "tags.`mlflow.prompt.is_prompt` = 'false'",
    "params.k = 'a\\'b'",
    'params.k = "a\\"b"',
    "params.k = ''",
    'params.k = ""',
    "params.k = 'with, comma'",
    "params.k = 'trailing space '",
    # --- Unicode keys/values ---
    "params.café = 'value'",
    "tags.emoji = '🚀'",
    "params.Ω = 'ω'",
    'metrics."日本 語" = 3',
    # --- Long keys / values ---
    "metrics." + ("a" * 200) + " = 1",
    "params.k = '" + ("x" * 500) + "'",
    # --- Numeric edge values ---
    "metrics.m = 0",
    "metrics.m = -0.0",
    "metrics.m = 123456789012345",
    "metrics.m = 3.14159265358979",
    "metrics.m = 1E10",
    "metrics.m = -1.5e-3",
    "metrics.m = 0x1F",
    "metrics.m = .0",
    "metrics.m = 100.",
    "version_number = 0",
    "version_number = -1",
    "version_number = 1.5",
    "timestamp = 0",
    "timestamp_ms = -999",
    # --- IN / NOT IN lists ---
    "run_id IN ('a')",
    "run_id IN ('a', 'b', 'c', 'd', 'e')",
    "run_id NOT IN ('a', 'b')",
    "attribute.run_id IN ('MixedCase', 'lower')",
    "attribute.run_id NOT IN ('UPPER')",
    "run_id IN ()",
    "run_id IN ('a' 'b')",
    "run_id IN (1, 2)",
    "run_id IN ('a',)",
    "status IN ('READY', 'FAILED')",
    "name IN ('a', 'b')",
    "dataset.name IN ('x', 'y')",
    "dataset.digest IN ('d1')",
    "dataset.context IN ('train')",
    # --- Every alias (entity-type) ---
    "metric.a = 1",
    "metrics.a = 1",
    "parameter.a = 'x'",
    "param.a = 'x'",
    "parameters.a = 'x'",
    "tag.a = 'x'",
    "attr.a = 'x'",
    "attributes.status = 'x'",
    "run.status = 'x'",
    "dataset.name = 'x'",
    "datasets.name = 'x'",
    "trace.status = 'OK'",
    "metadata.foo = 'bar'",
    # --- Invalid-comparator combinations (parse-time vs filter-time differ) ---
    "params.p > 'x'",
    "params.p < 'x'",
    "params.p >= 'x'",
    "tags.t > 'x'",
    "metrics.m LIKE 'x'",
    "metrics.m ILIKE 'x'",
    "attribute.status > 'x'",
    "attribute.start_time LIKE '5'",
    "version_number LIKE '5'",
    "feedback.q = 'yes'",
    "feedback.q != 'no'",
    "feedback.q > 'x'",
    "feedback.q > 5",
    "feedback.q < 0.5",
    "expectation.e = 'foo'",
    "expectation.e >= 3",
    "span.name != 'x'",
    "span.type IN ('LLM', 'CHAIN')",
    "span.status = 'OK'",
    "span.content ILIKE '%err%'",
    "span.content = 'x'",
    "span.badattr = 'x'",
    "span.attributes. = 'x'",
    "issue.id = 'i1'",
    "issue.id != 'i1'",
    "issue.badattr = 'x'",
    # --- IS NULL / IS NOT NULL across types ---
    "tags.t IS NULL",
    "tags.t IS NOT NULL",
    "params.p IS NULL",
    "attribute.status IS NOT NULL",
    "metadata.foo IS NULL",
    "feedback.q IS NULL",
    "timestamp IS NULL",
    # --- Whitespace / structural oddities ---
    "  metrics.a  >  1  ",
    "metrics.a>1 AND params.b='x'",
    "metrics.a > 1 and params.b = 'x' and tags.c = 'y'",
    "metrics.a > 1  AND  metrics.b < 2",
    "AND metrics.a > 1",
    "metrics.a > 1 AND",
    "metrics.a = = 1",
    "= 'x'",
    "'x' = 'y'",
    "5 = 5",
]

# order_by inputs shared across domains
COMMON_ORDER_BYS = [
    "",
    "metrics.foo",
    "metrics.foo ASC",
    "metrics.foo DESC",
    "metrics.foo asc",
    "metrics.foo desc",
    "metrics.`Mean Square Error`",
    "metrics.`Mean Square Error` ASC",
    "metrics.`Mean Square Error` DESC",
    "attribute.start_time",
    "attribute.start_time DESC",
    "params.p",
    "tags.t",
    "name",
    "name ASC",
    "name DESC",
    "name aCs",
    "timestamp",
    "timestamp DESC",
    "timestamp ASC",
    "creation_timestamp",
    "creation_timestamp DESC",
    "last_updated_timestamp DESC",
    "last_update_time",
    "version_number",
    "version_number DESC",
    "run_id",
    "status",
    "experiment_id",
    "execution_time",
    "end_time",
    # invalid
    "m.acc",
    "acc",
    "attri.x",
    "`metrics.A",
    "`metrics.A`",
    "attribute.start",
    "attribute.experiment_id",
    "metrics.A != 1",
    "params.my_param ",
    "attribute.run_id ACS",
    "attribute.run_id decs",
    "last_updated_timestamp DESC blah",
    "timestamp somerandomstuff ASC",
    "timestamp somerandomstuff",
    "timestamp decs",
    "timestamp ACS",
    "foo",
    "foo DESC",
    # --- Additional order_by breadth ---
    "metrics.foo ASC DESC",
    "metrics.`a b c`",
    'metrics."quoted key" DESC',
    "attribute.run_id",
    "attribute.run_id ASC",
    "attribute.run_name DESC",
    "artifact_uri",
    "user_id ASC",
    "creation_time",
    "creation_time DESC",
    "last_update_time ASC",
    "version_number ASC",
    "model_id",
    "source_run_id",
    "model_type DESC",
    "  name  ",
    "  name  DESC  ",
    "metric.acc",
    "params.foo bar",
    "tags.`x`",
    "`name`",
    "`name` DESC",
    "日本",
]

DOMAINS = {
    "runs": {
        "filter": SearchUtils.parse_search_filter,
        "order_by": SearchUtils.parse_order_by_for_search_runs,
    },
    "experiments": {
        "filter": SearchExperimentsUtils.parse_search_filter,
        "order_by": SearchExperimentsUtils.parse_order_by_for_search_experiments,
    },
    "registered_models": {
        "filter": SearchModelUtils.parse_search_filter,
        "order_by": SearchModelUtils.parse_order_by_for_search_registered_models,
    },
    "model_versions": {
        "filter": SearchModelVersionUtils.parse_search_filter,
        "order_by": SearchModelVersionUtils.parse_order_by_for_search_model_versions,
    },
    "traces": {
        "filter": SearchTraceUtils.parse_search_filter_for_search_traces,
        "order_by": SearchTraceUtils.parse_order_by_for_search_traces,
    },
    "logged_models": {
        # SearchLoggedModelsUtils shares parse_search_filter with SearchUtils
        # semantics but different valid keys; order_by uses a dict-based API
        # (parse_order_by_for_logged_models) handled separately below.
        "filter": __import__(
            "mlflow.utils.search_utils", fromlist=["SearchLoggedModelsUtils"]
        ).SearchLoggedModelsUtils.parse_search_filter,
    },
}


def build():
    corpus = {}
    for domain, fns in DOMAINS.items():
        entries = {"filter": [], "order_by": []}
        f = filter_case(fns["filter"])
        for s in COMMON_FILTERS:
            entries["filter"].append({"input": s, "result": f(s)})
        if "order_by" in fns:
            o = order_by_case(fns["order_by"])
            for s in COMMON_ORDER_BYS:
                entries["order_by"].append({"input": s, "result": o(s)})
        corpus[domain] = entries

    # Logged models order_by: dict-based API.
    from mlflow.utils.search_utils import SearchLoggedModelsUtils

    def lm_order(order_by_dict):
        try:
            ob = SearchLoggedModelsUtils.parse_order_by_for_logged_models(order_by_dict)
            return {
                "ok": True,
                "value": {
                    "field_name": ob.field_name,
                    "ascending": ob.ascending,
                    "dataset_name": ob.dataset_name,
                    "dataset_digest": ob.dataset_digest,
                },
            }
        except MlflowException as e:
            return {
                "ok": False,
                "error_code": e.error_code,
                "message": _normalize(str(e.message)),
            }

    lm_order_inputs = [
        {"field_name": "name"},
        {"field_name": "name", "ascending": False},
        {"field_name": "creation_time"},
        {"field_name": "creation_timestamp", "ascending": True},
        {"field_name": "model_id"},
        {"field_name": "status"},
        {"field_name": "metrics.accuracy"},
        {"field_name": "metrics.accuracy", "ascending": False},
        {"field_name": "metrics.rmse", "dataset_name": "train"},
        {"field_name": "metrics.rmse", "dataset_name": "train", "dataset_digest": "d1"},
        {"field_name": "metrics.rmse", "dataset_digest": "d1"},
        {"ascending": True},
        {"field_name": "invalid_field"},
        {"field_name": "params.foo"},
        {"field_name": "name", "ascending": "yes"},
    ]
    corpus["logged_models"]["order_by_dict"] = [
        {"input": inp, "result": lm_order(inp)} for inp in lm_order_inputs
    ]

    # Page-token round-trip cases (SearchUtils.parse_start_offset_from_page_token).
    def page_token_case(token):
        try:
            return {"ok": True, "value": SearchUtils.parse_start_offset_from_page_token(token)}
        except MlflowException as e:
            return {
                "ok": False,
                "error_code": e.error_code,
                "message": _normalize(str(e.message)),
            }

    def b64(obj):
        return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("utf-8")

    page_tokens = [
        {"input": None, "result": page_token_case(None)},
        {"input": "", "result": page_token_case("")},
        {"input": b64({"offset": 5}), "result": page_token_case(b64({"offset": 5}))},
        {"input": b64({"offset": 0}), "result": page_token_case(b64({"offset": 0}))},
        {"input": b64({}), "result": page_token_case(b64({}))},
        {"input": b64({"offset": "a"}), "result": page_token_case(b64({"offset": "a"}))},
        {"input": b64({"offsoot": 7}), "result": page_token_case(b64({"offsoot": 7}))},
    ]
    corpus["page_tokens"] = page_tokens

    return corpus


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = build()
    for name, data in corpus.items():
        path = CORPUS_DIR / f"{name}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        counts = {k: len(v) for k, v in data.items()} if isinstance(data, dict) else len(data)
        print(f"wrote {path.name}: {counts}")


if __name__ == "__main__":
    main()
