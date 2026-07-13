//! Replays the Python-generated parity corpus against the Rust parsers.
//!
//! The corpus under `tests/corpus/*.json` is produced by
//! `rust/tools/gen_search_corpus.py`, which runs each input through the real
//! MLflow Python `Search*Utils` classes and records the parsed structure or the
//! exact `MlflowException` (error_code + message). This test parses the same
//! inputs with `mlflow-search` and asserts byte-for-byte equality of the
//! normalized JSON result.
//!
//! Set-repr normalization: a few MLflow error messages interpolate a Python
//! `set`/`dict_keys`, whose iteration order is not stable. The generator sorts
//! those blobs; we apply the identical normalization to the Rust error messages
//! before comparing (see `normalize`).

use mlflow_search::{parse, AscendingValue, LoggedModelOrderByInput, SearchError};
use serde_json::{json, Value};

/// Normalize `{...}` / `dict_keys([...])` blobs by sorting their comma-split
/// contents, matching `_normalize` in `gen_search_corpus.py`.
fn normalize(msg: &str) -> String {
    let mut s = sort_braced(msg, "dict_keys([", "])");
    s = sort_braced(&s, "{", "}");
    s
}

fn sort_braced(msg: &str, open: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = msg;
    while let Some(start) = rest.find(open) {
        let after_open = &rest[start + open.len()..];
        let Some(end_rel) = after_open.find(close) else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..start]);
        let inner = &after_open[..end_rel];
        let mut parts: Vec<&str> = inner
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        parts.sort();
        out.push_str(open);
        out.push_str(&parts.join(", "));
        out.push_str(close);
        rest = &after_open[end_rel + close.len()..];
    }
    out.push_str(rest);
    out
}

/// Convert a Rust parse result into the corpus's `{"ok": ...}` JSON shape.
fn to_result_json<T: serde::Serialize>(r: Result<T, SearchError>) -> Value {
    match r {
        Ok(v) => json!({"ok": true, "value": serde_json::to_value(v).unwrap()}),
        Err(e) => json!({
            "ok": false,
            "error_code": e.error_code.as_str(),
            "message": normalize(&e.message),
        }),
    }
}

fn load(name: &str) -> Value {
    let path = format!("{}/tests/corpus/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

/// Compare a Rust result JSON against the expected (Python) result JSON.
/// Both are `serde_json::Value` maps, so key order is irrelevant.
fn assert_case(domain: &str, kind: &str, input: &Value, expected: &Value, actual: Value) {
    if &actual != expected {
        panic!(
            "MISMATCH [{domain}/{kind}] input={input}\n  expected={}\n  actual  ={}",
            serde_json::to_string(expected).unwrap(),
            serde_json::to_string(&actual).unwrap(),
        );
    }
}

fn run_filter(
    domain: &str,
    parse_fn: impl Fn(&str) -> Result<Vec<mlflow_search::Comparison>, SearchError>,
) {
    let data = load(domain);
    for case in data["filter"].as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let expected = &case["result"];
        let actual = to_result_json(parse_fn(input));
        assert_case(domain, "filter", &case["input"], expected, actual);
    }
}

fn run_order_by(
    domain: &str,
    parse_fn: impl Fn(&str) -> Result<mlflow_search::OrderBy, SearchError>,
) {
    let data = load(domain);
    let Some(cases) = data["order_by"].as_array() else {
        return;
    };
    for case in cases {
        let input = case["input"].as_str().unwrap();
        let expected = &case["result"];
        // order_by parsers return (type, key, ascending); the corpus stores a
        // 3-element list. Map our OrderBy into that shape.
        let actual = match parse_fn(input) {
            Ok(ob) => json!({
                "ok": true,
                "value": [ob.entity_type, ob.key, ob.ascending],
            }),
            Err(e) => json!({
                "ok": false,
                "error_code": e.error_code.as_str(),
                "message": normalize(&e.message),
            }),
        };
        assert_case(domain, "order_by", &case["input"], expected, actual);
    }
}

#[test]
fn runs_corpus() {
    run_filter("runs", parse::runs_filter);
    run_order_by("runs", parse::runs_order_by);
}

#[test]
fn experiments_corpus() {
    run_filter("experiments", parse::experiments_filter);
    run_order_by("experiments", parse::experiments_order_by);
}

#[test]
fn registered_models_corpus() {
    run_filter("registered_models", parse::registered_models_filter);
    run_order_by("registered_models", parse::registered_models_order_by);
}

#[test]
fn model_versions_corpus() {
    run_filter("model_versions", parse::model_versions_filter);
    run_order_by("model_versions", parse::model_versions_order_by);
}

#[test]
fn traces_corpus() {
    run_filter("traces", parse::traces_filter);
    run_order_by("traces", parse::traces_order_by);
}

#[test]
fn logged_models_corpus() {
    run_filter("logged_models", parse::logged_models_filter);
    // logged models use a dict-based order_by API.
    let data = load("logged_models");
    for case in data["order_by_dict"].as_array().unwrap() {
        let input = &case["input"];
        let expected = &case["result"];
        let order_by = LoggedModelOrderByInput {
            field_name: input
                .get("field_name")
                .and_then(|v| v.as_str())
                .map(String::from),
            ascending: input.get("ascending").map(ascending_value),
            dataset_name: input
                .get("dataset_name")
                .and_then(|v| v.as_str())
                .map(String::from),
            dataset_digest: input
                .get("dataset_digest")
                .and_then(|v| v.as_str())
                .map(String::from),
        };
        let actual = match parse::logged_models_order_by(&order_by) {
            Ok(ob) => json!({"ok": true, "value": serde_json::to_value(ob).unwrap()}),
            Err(e) => json!({
                "ok": false,
                "error_code": e.error_code.as_str(),
                "message": normalize(&e.message),
            }),
        };
        assert_case("logged_models", "order_by_dict", input, expected, actual);
    }
}

#[test]
fn page_tokens_corpus() {
    let cases: Vec<Value> = serde_json::from_value(load("page_tokens")).unwrap();
    for case in &cases {
        let input = case["input"].as_str();
        let expected = &case["result"];
        let actual = match mlflow_search::parse_start_offset_from_page_token(input) {
            Ok(offset) => json!({"ok": true, "value": offset}),
            Err(e) => json!({
                "ok": false,
                "error_code": e.error_code.as_str(),
                "message": normalize(&e.message),
            }),
        };
        assert_case(
            "page_tokens",
            "page_token",
            &case["input"],
            expected,
            actual,
        );
    }
}

fn ascending_value(v: &Value) -> AscendingValue {
    match v {
        Value::Bool(b) => AscendingValue::Bool(*b),
        Value::String(_) => AscendingValue::Other("str"),
        Value::Number(n) if n.is_i64() || n.is_u64() => AscendingValue::Other("int"),
        Value::Number(_) => AscendingValue::Other("float"),
        Value::Null => AscendingValue::Other("NoneType"),
        Value::Array(_) => AscendingValue::Other("list"),
        Value::Object(_) => AscendingValue::Other("dict"),
    }
}
