//! `search_traces` behavioral tests (plan T2.10), ported from the order-by /
//! filter / pagination / run_id / span cases in
//! `tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py`.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

#![allow(clippy::too_many_arguments, clippy::cloned_ref_to_slice_refs)]

use mlflow_store::{SpanInput, StartTraceInput, TraceInfo, TraceTimeRange, TrackingStore};
use mlflow_test_support::TempDb;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
}

#[allow(clippy::too_many_arguments)]
async fn create_trace(
    s: &TrackingStore,
    trace_id: &str,
    exp: &str,
    request_time: i64,
    execution_duration: Option<i64>,
    state: &str,
    tags: &[(&str, &str)],
    metadata: &[(&str, &str)],
) {
    let input = StartTraceInput {
        trace_id: trace_id.to_string(),
        experiment_id: exp.to_string(),
        request_time,
        execution_duration,
        state: state.to_string(),
        client_request_id: None,
        request_preview: None,
        response_preview: None,
        tags: tags
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        trace_metadata: metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        trace_metrics: vec![],
    };
    s.start_trace(WS, &input).await.unwrap();
}

fn ids(page: &[TraceInfo]) -> Vec<String> {
    page.iter().map(|t| t.trace_id.clone()).collect()
}

/// Reproduce the Python `store_with_traces` fixture: 5 traces across exp1/exp2.
async fn seed(s: &TrackingStore) -> (String, String) {
    let exp1 = s.create_experiment(WS, "exp1", None, &[]).await.unwrap();
    let exp2 = s.create_experiment(WS, "exp2", None, &[]).await.unwrap();

    create_trace(
        s,
        "tr-0",
        &exp2,
        0,
        Some(6),
        "OK",
        &[("mlflow.traceName", "ddd")],
        &[("mlflow.sourceRun", "run0")],
    )
    .await;
    create_trace(
        s,
        "tr-1",
        &exp2,
        1,
        Some(2),
        "ERROR",
        &[
            ("mlflow.traceName", "aaa"),
            ("fruit", "apple"),
            ("color", "red"),
        ],
        &[("mlflow.sourceRun", "run1")],
    )
    .await;
    create_trace(
        s,
        "tr-2",
        &exp1,
        2,
        Some(4),
        "STATE_UNSPECIFIED",
        &[
            ("mlflow.traceName", "bbb"),
            ("fruit", "apple"),
            ("color", "green"),
        ],
        &[],
    )
    .await;
    create_trace(
        s,
        "tr-3",
        &exp1,
        3,
        Some(10),
        "OK",
        &[("mlflow.traceName", "ccc"), ("fruit", "orange")],
        &[],
    )
    .await;
    create_trace(
        s,
        "tr-4",
        &exp1,
        4,
        Some(10),
        "OK",
        &[("mlflow.traceName", "ddd"), ("color", "blue")],
        &[],
    )
    .await;
    (exp1, exp2)
}

async fn search_ids(
    s: &TrackingStore,
    exps: &[String],
    filter: Option<&str>,
    order_by: &[String],
) -> Vec<String> {
    let page = s
        .search_traces(WS, exps, filter, 5, order_by, None)
        .await
        .unwrap();
    ids(&page.trace_infos)
}

// ---------------------------------------------------------------------------
// order-by parity (from test_search_traces_order_by)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn order_by_cases() {
    let tmp = TempDb::new("order").await;
    let s = store(&tmp).await;
    let (exp1, exp2) = seed(&s).await;
    let exps = [exp1, exp2];
    let ob = |v: &str| vec![v.to_string()];

    // Default: timestamp DESC.
    assert_eq!(
        search_ids(&s, &exps, None, &[]).await,
        vec!["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]
    );
    assert_eq!(
        search_ids(&s, &exps, None, &ob("timestamp")).await,
        vec!["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"]
    );
    assert_eq!(
        search_ids(&s, &exps, None, &ob("timestamp DESC")).await,
        vec!["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]
    );
    assert_eq!(
        search_ids(
            &s,
            &exps,
            None,
            &["execution_time DESC".into(), "timestamp ASC".into()]
        )
        .await,
        vec!["tr-3", "tr-4", "tr-0", "tr-2", "tr-1"]
    );
    assert_eq!(
        search_ids(&s, &exps, None, &ob("status")).await,
        vec!["tr-1", "tr-4", "tr-3", "tr-0", "tr-2"]
    );
    // Order by name (tag mlflow.traceName). tr-0 & tr-4 share "ddd"; the default
    // tiebreak is `timestamp_ms DESC` (appended before request_id ASC), so
    // tr-4 (ts=4) sorts before tr-0 (ts=0).
    assert_eq!(
        search_ids(&s, &exps, None, &ob("name")).await,
        vec!["tr-1", "tr-2", "tr-3", "tr-4", "tr-0"]
    );
    // Order by tag (null last).
    assert_eq!(
        search_ids(&s, &exps, None, &ob("tag.fruit")).await,
        vec!["tr-2", "tr-1", "tr-3", "tr-4", "tr-0"]
    );
    // Order by non-existent tag → default order.
    assert_eq!(
        search_ids(&s, &exps, None, &ob("tag.nonexistent")).await,
        vec!["tr-4", "tr-3", "tr-2", "tr-1", "tr-0"]
    );
    // Order by run_id (metadata mlflow.sourceRun, null last).
    assert_eq!(
        search_ids(&s, &exps, None, &ob("run_id")).await,
        vec!["tr-0", "tr-1", "tr-4", "tr-3", "tr-2"]
    );
}

// ---------------------------------------------------------------------------
// filter parity (from test_search_traces_with_filter)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn filter_cases() {
    let tmp = TempDb::new("filter").await;
    let s = store(&tmp).await;
    let (exp1, exp2) = seed(&s).await;
    let exps = [exp1, exp2];

    let cases: &[(&str, &[&str])] = &[
        ("name = 'aaa'", &["tr-1"]),
        ("name != 'aaa'", &["tr-4", "tr-3", "tr-2", "tr-0"]),
        ("status = 'OK'", &["tr-4", "tr-3", "tr-0"]),
        ("status != 'OK'", &["tr-2", "tr-1"]),
        ("attributes.status = 'OK'", &["tr-4", "tr-3", "tr-0"]),
        ("trace.status = 'OK'", &["tr-4", "tr-3", "tr-0"]),
        (
            "`timestamp` >= 1 AND execution_time < 10",
            &["tr-2", "tr-1"],
        ),
        ("tag.fruit = 'apple'", &["tr-2", "tr-1"]),
        ("tags.fruit = 'apple' and tags.color != 'red'", &["tr-2"]),
        ("run_id = 'run0'", &["tr-0"]),
        ("request_metadata.mlflow.sourceRun = 'run0'", &["tr-0"]),
        ("metadata.mlflow.sourceRun != 'run0'", &["tr-1"]),
    ];
    for (q, expected) in cases {
        let got = search_ids(&s, &exps, Some(q), &[]).await;
        let want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(got, want, "query: {q}");
    }
}

#[tokio::test]
async fn filter_invalid_errors() {
    let tmp = TempDb::new("filter_invalid").await;
    let s = store(&tmp).await;
    let (exp1, exp2) = seed(&s).await;
    let exps = [exp1, exp2];
    for (q, needle) in [
        ("foo.bar = 'baz'", "Invalid entity type 'foo'"),
        ("invalid = 'foo'", "Invalid attribute key 'invalid'"),
        ("trace.status < 'OK'", "Invalid comparator '<'"),
        ("name IN ('foo', 'bar')", "Invalid comparator 'IN'"),
    ] {
        let err = s
            .search_traces(WS, &exps, Some(q), 5, &[], None)
            .await
            .unwrap_err();
        assert!(err.message.contains(needle), "query {q}: {}", err.message);
    }
}

// ---------------------------------------------------------------------------
// run_id filter (linked via entity_associations OR metadata)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_id_filter_links_and_metadata() {
    let tmp = TempDb::new("runid").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let run = s
        .create_run(WS, &exp, Some("u"), Some(0), Some("r"), &[])
        .await
        .unwrap();
    let run_id = run.info.run_id.clone();

    // Direct metadata association.
    create_trace(
        &s,
        "tr-direct",
        &exp,
        1,
        Some(1),
        "OK",
        &[],
        &[("mlflow.sourceRun", &run_id)],
    )
    .await;
    // Linked via entity association.
    create_trace(&s, "tr-linked", &exp, 2, Some(1), "OK", &[], &[]).await;
    s.link_traces_to_run(WS, &["tr-linked".into()], &run_id)
        .await
        .unwrap();
    // Both.
    create_trace(
        &s,
        "tr-both",
        &exp,
        3,
        Some(1),
        "OK",
        &[],
        &[("mlflow.sourceRun", &run_id)],
    )
    .await;
    s.link_traces_to_run(WS, &["tr-both".into()], &run_id)
        .await
        .unwrap();
    // Unrelated.
    create_trace(&s, "tr-unrelated", &exp, 4, Some(1), "OK", &[], &[]).await;

    let got: std::collections::HashSet<String> = search_ids(
        &s,
        &[exp.clone()],
        Some(&format!("attributes.run_id = \"{run_id}\"")),
        &[],
    )
    .await
    .into_iter()
    .collect();
    assert_eq!(
        got,
        ["tr-direct", "tr-linked", "tr-both"]
            .map(String::from)
            .into_iter()
            .collect()
    );
}

#[tokio::test]
async fn run_id_filter_combined_with_tag() {
    let tmp = TempDb::new("runid_tag").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let run = s
        .create_run(WS, &exp, Some("u"), Some(0), Some("r"), &[])
        .await
        .unwrap();
    let run_id = run.info.run_id.clone();

    create_trace(
        &s,
        "t1",
        &exp,
        1,
        Some(1),
        "OK",
        &[("type", "training")],
        &[],
    )
    .await;
    s.link_traces_to_run(WS, &["t1".into()], &run_id)
        .await
        .unwrap();
    create_trace(
        &s,
        "t2",
        &exp,
        2,
        Some(1),
        "OK",
        &[("type", "inference")],
        &[],
    )
    .await;
    s.link_traces_to_run(WS, &["t2".into()], &run_id)
        .await
        .unwrap();
    create_trace(
        &s,
        "t3",
        &exp,
        3,
        Some(1),
        "OK",
        &[("type", "training")],
        &[],
    )
    .await;

    let got = search_ids(
        &s,
        &[exp.clone()],
        Some(&format!(
            "run_id = \"{run_id}\" AND tags.type = \"training\""
        )),
        &[],
    )
    .await;
    assert_eq!(got, vec!["t1"]);
}

// ---------------------------------------------------------------------------
// span filters
// ---------------------------------------------------------------------------

fn span(trace_id: &str, span_id: &str, name: &str, span_type: &str) -> SpanInput {
    SpanInput {
        trace_id: trace_id.to_string(),
        span_id: span_id.to_string(),
        parent_span_id: None,
        name: Some(name.to_string()),
        span_type: Some(span_type.to_string()),
        status: "OK".to_string(),
        start_time_unix_nano: 1_000_000_000,
        end_time_unix_nano: Some(2_000_000_000),
        content: format!("{{\"name\":\"{name}\"}}"),
        dimension_attributes: None,
    }
}

fn range(trace_id: &str) -> TraceTimeRange {
    TraceTimeRange {
        trace_id: trace_id.to_string(),
        min_start_ms: 1_000,
        max_end_ms: Some(2_000),
        root_span_status: Some("OK".to_string()),
    }
}

#[tokio::test]
async fn span_name_and_type_filters() {
    let tmp = TempDb::new("span_filter").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    for id in ["trace1", "trace2", "trace3"] {
        create_trace(&s, id, &exp, 1, Some(1), "OK", &[], &[]).await;
    }
    let sp1 = span("trace1", "111", "database_query", "FUNCTION");
    let sp2 = span("trace2", "222", "api_call", "LLM");
    let sp3 = span("trace3", "333", "database_update", "RETRIEVER");
    s.log_spans(WS, &exp, &[sp1], &[], &[range("trace1")])
        .await
        .unwrap();
    s.log_spans(WS, &exp, &[sp2], &[], &[range("trace2")])
        .await
        .unwrap();
    s.log_spans(WS, &exp, &[sp3], &[], &[range("trace3")])
        .await
        .unwrap();

    assert_eq!(
        search_ids(
            &s,
            &[exp.clone()],
            Some("span.name = \"database_query\""),
            &[]
        )
        .await,
        vec!["trace1"]
    );
    let like: std::collections::HashSet<_> = search_ids(
        &s,
        &[exp.clone()],
        Some("span.name LIKE \"database%\""),
        &[],
    )
    .await
    .into_iter()
    .collect();
    assert_eq!(
        like,
        ["trace1", "trace3"].map(String::from).into_iter().collect()
    );
    let neq: std::collections::HashSet<_> =
        search_ids(&s, &[exp.clone()], Some("span.name != \"api_call\""), &[])
            .await
            .into_iter()
            .collect();
    assert_eq!(
        neq,
        ["trace1", "trace3"].map(String::from).into_iter().collect()
    );
    assert!(
        search_ids(&s, &[exp.clone()], Some("span.name = \"nonexistent\""), &[])
            .await
            .is_empty()
    );
    // Type filter with IN.
    let in_types: std::collections::HashSet<_> = search_ids(
        &s,
        &[exp.clone()],
        Some("span.type IN (\"LLM\", \"RETRIEVER\")"),
        &[],
    )
    .await
    .into_iter()
    .collect();
    assert_eq!(
        in_types,
        ["trace2", "trace3"].map(String::from).into_iter().collect()
    );
}

#[tokio::test]
async fn span_multiple_predicates_match_same_span() {
    let tmp = TempDb::new("span_same").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    create_trace(&s, "t", &exp, 1, Some(1), "OK", &[], &[]).await;
    // Two spans: one named foo/OK, one named bar/ERROR.
    let mut foo = span("t", "1", "foo", "FUNCTION");
    foo.status = "OK".into();
    let mut bar = span("t", "2", "bar", "FUNCTION");
    bar.status = "ERROR".into();
    s.log_spans(WS, &exp, &[foo, bar], &[], &[range("t")])
        .await
        .unwrap();

    // name=bar AND status=OK must NOT match (no single span satisfies both).
    assert!(search_ids(
        &s,
        &[exp.clone()],
        Some("span.name = \"bar\" AND span.status = \"OK\""),
        &[]
    )
    .await
    .is_empty());
    // name=foo AND status=OK matches.
    assert_eq!(
        search_ids(
            &s,
            &[exp.clone()],
            Some("span.name = \"foo\" AND span.status = \"OK\""),
            &[]
        )
        .await,
        vec!["t"]
    );
}

// ---------------------------------------------------------------------------
// pagination
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pagination_walks_all_pages() {
    let tmp = TempDb::new("pagination").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    for i in 0..7 {
        create_trace(&s, &format!("tr-{i}"), &exp, i, Some(1), "OK", &[], &[]).await;
    }
    let mut seen = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let page = s
            .search_traces(
                WS,
                &[exp.clone()],
                None,
                3,
                &["timestamp".into()],
                token.as_deref(),
            )
            .await
            .unwrap();
        seen.extend(ids(&page.trace_infos));
        match page.next_page_token {
            Some(t) => token = Some(t),
            None => break,
        }
    }
    assert_eq!(seen, (0..7).map(|i| format!("tr-{i}")).collect::<Vec<_>>());
}

#[tokio::test]
async fn max_results_validation() {
    let tmp = TempDb::new("maxres").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let err = s
        .search_traces(WS, &[exp.clone()], None, 50001, &[], None)
        .await
        .unwrap_err();
    assert!(err.message.contains("at most 50000"), "{}", err.message);
    let err = s
        .search_traces(WS, &[exp.clone()], None, -1, &[], None)
        .await
        .unwrap_err();
    assert!(err.message.contains("positive integer"), "{}", err.message);
}
