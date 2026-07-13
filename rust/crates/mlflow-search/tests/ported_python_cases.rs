//! Hand-ported cases from `tests/utils/test_search_utils.py`.
//!
//! These mirror the Python `@pytest.mark.parametrize` bodies directly (rather
//! than via the generated corpus) so the intent of each case is legible and so
//! representative assertions survive even if the corpus is regenerated. The
//! exhaustive breadth lives in `corpus_replay.rs`.

use mlflow_search::{parse, Comparison, Value};

fn comp(entity_type: &str, key: &str, comparator: &str, value: Value) -> Comparison {
    Comparison {
        entity_type: entity_type.to_string(),
        key: key.to_string(),
        comparator: comparator.to_string(),
        value,
    }
}

fn s(v: &str) -> Value {
    Value::Str(v.to_string())
}

// ----- test_filter (SearchUtils.parse_search_filter, runs) -----

#[test]
fn runs_filter_basic() {
    let cases: &[(&str, Vec<Comparison>)] = &[
        (
            "metric.acc >= 0.94",
            vec![comp("metric", "acc", ">=", s("0.94"))],
        ),
        (
            "metric.acc>=100",
            vec![comp("metric", "acc", ">=", s("100"))],
        ),
        (
            "params.m!='tf'",
            vec![comp("parameter", "m", "!=", s("tf"))],
        ),
        (
            r#"params."m"!="tf""#,
            vec![comp("parameter", "m", "!=", s("tf"))],
        ),
        (
            r#"metric."legit name" >= 0.243"#,
            vec![comp("metric", "legit name", ">=", s("0.243"))],
        ),
        ("metrics.XYZ = 3", vec![comp("metric", "XYZ", "=", s("3"))]),
        (
            r#"params."cat dog" = "pets""#,
            vec![comp("parameter", "cat dog", "=", s("pets"))],
        ),
        (
            r#"metrics."X-Y-Z" = 3"#,
            vec![comp("metric", "X-Y-Z", "=", s("3"))],
        ),
        (
            r#"metrics."X//Y#$$@&Z" = 3"#,
            vec![comp("metric", "X//Y#$$@&Z", "=", s("3"))],
        ),
        ("", vec![]),
        (
            "`metric`.a >= 0.1",
            vec![comp("metric", "a", ">=", s("0.1"))],
        ),
        (
            "`params`.model >= 'LR'",
            vec![comp("parameter", "model", ">=", s("LR"))],
        ),
        (
            "tags.version = 'commit-hash'",
            vec![comp("tag", "version", "=", s("commit-hash"))],
        ),
        (
            "`tags`.source_name = 'a notebook'",
            vec![comp("tag", "source_name", "=", s("a notebook"))],
        ),
        (
            r#"metrics."accuracy.2.0" > 5"#,
            vec![comp("metric", "accuracy.2.0", ">", s("5"))],
        ),
        (
            "metrics.`spacey name` > 5",
            vec![comp("metric", "spacey name", ">", s("5"))],
        ),
        (
            "attribute.artifact_uri = '1/23/4'",
            vec![comp("attribute", "artifact_uri", "=", s("1/23/4"))],
        ),
        (
            "attribute.start_time >= 1234",
            vec![comp("attribute", "start_time", ">=", s("1234"))],
        ),
        (
            "run.status = 'RUNNING'",
            vec![comp("attribute", "status", "=", s("RUNNING"))],
        ),
        (
            "dataset.name = 'my_dataset'",
            vec![comp("dataset", "name", "=", s("my_dataset"))],
        ),
        (
            "tags.version IS NULL",
            vec![comp("tag", "version", "IS NULL", Value::Null)],
        ),
        (
            "params.lr IS NOT NULL",
            vec![comp("parameter", "lr", "IS NOT NULL", Value::Null)],
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(
            &parse::runs_filter(input).unwrap(),
            expected,
            "input={input:?}"
        );
    }
}

#[test]
fn runs_filter_quote_trimming() {
    assert_eq!(
        parse::runs_filter(r#"params.m = "L'Hosp""#).unwrap(),
        vec![comp("parameter", "m", "=", s("L'Hosp"))]
    );
}

#[test]
fn runs_filter_conjunction() {
    assert_eq!(
        parse::runs_filter("metrics.rmse < 1 and params.model_class = 'LR'").unwrap(),
        vec![
            comp("metric", "rmse", "<", s("1")),
            comp("parameter", "model_class", "=", s("LR")),
        ]
    );
}

// ----- test_error_filter -----

#[test]
fn runs_filter_errors() {
    let cases: &[(&str, &str)] = &[
        (
            "metric.acc >= 0.94; metrics.rmse < 1",
            "Search filter contained multiple expression",
        ),
        ("m.acc >= 0.94", "Invalid entity type"),
        ("acc >= 0.94", "Invalid attribute key"),
        (
            "metrics.A > 0.1 OR params.B = 'LR'",
            "Invalid clause(s) in filter string",
        ),
        ("`metrics.A > 0.1", "Invalid clause(s) in filter string"),
        (
            "attribute.status = true",
            "Invalid clause(s) in filter string",
        ),
        ("dataset.status = 'true'", "Invalid dataset key"),
        (
            "metrics.acc IS NULL",
            "IS NULL / IS NOT NULL is only supported for tags and params",
        ),
        (
            "metric.model = 'LR'",
            "Expected numeric value type for metric",
        ),
        ("params.acc = 5", "Expected a quoted string value for param"),
        (
            "attribute.status = 1",
            "Expected a quoted string value for attributes",
        ),
        (
            "params.acc = LR",
            "value is either not quoted or unidentified quote types",
        ),
        ("1=1", "Expected 'Identifier' found"),
    ];
    for (input, needle) in cases {
        let err = parse::runs_filter(input).unwrap_err();
        assert!(
            err.message.contains(needle),
            "input={input:?}: expected message to contain {needle:?}, got {:?}",
            err.message
        );
    }
}

// ----- test_space_order_by_search_runs -----

#[test]
fn runs_order_by_spaces_and_dir() {
    for (input, ascending) in [
        ("metrics.`Mean Square Error`", true),
        ("metrics.`Mean Square Error` ASC", true),
        ("metrics.`Mean Square Error` DESC", false),
    ] {
        let ob = parse::runs_order_by(input).unwrap();
        assert_eq!(ob.entity_type, "metric");
        assert_eq!(ob.key, "Mean Square Error");
        assert_eq!(ob.ascending, ascending, "input={input:?}");
    }
}

// ----- test_invalid_order_by_search_runs -----

#[test]
fn runs_order_by_errors() {
    let cases: &[(&str, &str)] = &[
        ("m.acc", "Invalid entity type"),
        ("acc", "Invalid attribute key"),
        ("`metrics.A", "Invalid order_by clause"),
        ("`metrics.A`", "Invalid entity type"),
        ("metrics.A != 1", "Invalid order_by clause"),
        ("attribute.run_id ACS", "Invalid ordering key"),
    ];
    for (input, needle) in cases {
        let err = parse::runs_order_by(input).unwrap_err();
        assert!(
            err.message.contains(needle),
            "input={input:?}: expected {needle:?}, got {:?}",
            err.message
        );
    }
}

// ----- registered-models order_by errors (test_invalid_order_by_search_registered_models) -----

#[test]
fn registered_models_order_by() {
    // SearchModelUtils.parse_order_by_for_search_registered_models (NOT the
    // base SearchUtils version): `creation_timestamp` is a *valid* attribute
    // key here, so `creation_timestamp DESC` parses rather than erroring.
    let ob = parse::registered_models_order_by("creation_timestamp DESC").unwrap();
    assert_eq!(
        (ob.entity_type.as_str(), ob.key.as_str(), ob.ascending),
        ("attribute", "creation_timestamp", false)
    );

    let cases: &[(&str, &str)] = &[
        (
            "last_updated_timestamp DESC blah",
            "Invalid order_by clause",
        ),
        ("", "Invalid order_by clause"),
        ("timestamp decs", "Invalid order_by clause"),
        ("name aCs", "Invalid ordering key"),
    ];
    for (input, needle) in cases {
        let err = parse::registered_models_order_by(input).unwrap_err();
        assert!(
            err.message.contains(needle),
            "input={input:?}: expected {needle:?}, got {:?}",
            err.message
        );
    }
}

// ----- trace key remapping (SearchTraceUtils) -----

#[test]
fn traces_key_remapping() {
    assert_eq!(
        parse::traces_filter("name = 'foo'").unwrap(),
        vec![comp("tag", "mlflow.traceName", "=", s("foo"))]
    );
    assert_eq!(
        parse::traces_filter("timestamp > 1000").unwrap(),
        vec![comp("attribute", "timestamp_ms", ">", Value::Int(1000))]
    );
    assert_eq!(
        parse::traces_filter("run_id = 'r1'").unwrap(),
        vec![comp("request_metadata", "mlflow.sourceRun", "=", s("r1"))]
    );
}

// ----- model versions IN + numeric (SearchModelVersionUtils) -----

#[test]
fn model_versions_in_and_numeric() {
    assert_eq!(
        parse::model_versions_filter("run_id IN ('abc', 'def')").unwrap(),
        vec![comp(
            "attribute",
            "run_id",
            "IN",
            Value::List(vec!["abc".into(), "def".into()])
        )]
    );
    assert_eq!(
        parse::model_versions_filter("version_number > 5").unwrap(),
        vec![comp("attribute", "version_number", ">", Value::Int(5))]
    );
}
