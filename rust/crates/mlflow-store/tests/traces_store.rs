//! Behavioral integration tests for the tracing V3 store (plan T2.10), ported
//! from `tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py`.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

#![allow(clippy::too_many_arguments, clippy::cloned_ref_to_slice_refs)]

use mlflow_store::{
    AssessmentSource, AssessmentValue, NewAssessment, StartTraceInput, TrackingStore,
};
use mlflow_test_support::TempDb;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
}

/// Build a `StartTraceInput` with the given fields; tags/metadata as `(k, v)`.
fn trace_input(
    trace_id: &str,
    experiment_id: &str,
    request_time: i64,
    execution_duration: Option<i64>,
    state: &str,
    tags: &[(&str, &str)],
    metadata: &[(&str, &str)],
) -> StartTraceInput {
    StartTraceInput {
        trace_id: trace_id.to_string(),
        experiment_id: experiment_id.to_string(),
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
        assessments: vec![],
    }
}

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
    let input = trace_input(
        trace_id,
        exp,
        request_time,
        execution_duration,
        state,
        tags,
        metadata,
    );
    s.start_trace(WS, &input).await.unwrap();
}

fn ids(page: &[mlflow_store::TraceInfo]) -> Vec<String> {
    page.iter().map(|t| t.trace_id.clone()).collect()
}

// ---------------------------------------------------------------------------
// start_trace / get_trace_info
// ---------------------------------------------------------------------------

#[tokio::test]
async fn start_trace_and_get_info() {
    let tmp = TempDb::new("start").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let input = trace_input(
        "tr-123",
        &exp,
        1234,
        Some(100),
        "OK",
        &[("tag1", "apple"), ("tag2", "orange")],
        &[("rq1", "foo"), ("rq2", "bar")],
    );
    let ti = s.start_trace(WS, &input).await.unwrap();

    assert_eq!(ti.trace_id, "tr-123");
    assert_eq!(ti.experiment_id, exp);
    assert_eq!(ti.request_time, 1234);
    assert_eq!(ti.execution_duration, Some(100));
    assert_eq!(ti.state, "OK");
    // Caller metadata present + the FINALIZED flag.
    assert_eq!(ti.metadata("rq1"), Some("foo"));
    assert_eq!(ti.metadata("rq2"), Some("bar"));
    assert_eq!(ti.metadata("mlflow.trace.infoFinalized"), Some("true"));
    // Artifact-location tag was added.
    let art = ti.tag("mlflow.artifactLocation").unwrap();
    assert!(
        art.ends_with(&format!("/{exp}/traces/tr-123/artifacts")),
        "{art}"
    );
    assert_eq!(ti.tag("tag1"), Some("apple"));
    assert_eq!(ti.tag("tag2"), Some("orange"));

    // Round-trips through get_trace_info.
    let got = s.get_trace_info(WS, "tr-123").await.unwrap();
    assert_eq!(got, ti);
}

#[tokio::test]
async fn start_trace_persists_embedded_assessments_atomically() {
    let tmp = TempDb::new("start_assessments").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let mut input = trace_input("tr-with-assessment", &exp, 1234, Some(100), "OK", &[], &[]);
    input.assessments.push(NewAssessment {
        trace_id: input.trace_id.clone(),
        name: "expected_answer".to_string(),
        value: AssessmentValue::Expectation {
            value_json: "\"MLflow\"".to_string(),
        },
        source: AssessmentSource {
            source_type: "CODE".to_string(),
            source_id: Some("store-regression".to_string()),
        },
        run_id: None,
        span_id: Some("span-1".to_string()),
        rationale: Some("known answer".to_string()),
        metadata: None,
        create_time_ms: Some(1_700_000_000_000),
        last_update_time_ms: Some(1_700_000_000_001),
        assessment_id: Some("a-embedded-store".to_string()),
        overrides: None,
    });

    let started = s.start_trace(WS, &input).await.unwrap();
    assert_eq!(started.assessments.len(), 1);
    let assessment = &started.assessments[0];
    assert_eq!(assessment.assessment_id, "a-embedded-store");
    assert_eq!(assessment.trace_id, "tr-with-assessment");
    assert_eq!(assessment.name, "expected_answer");
    assert_eq!(assessment.assessment_type, "expectation");
    assert_eq!(assessment.value, "\"MLflow\"");

    let got = s.get_trace_info(WS, "tr-with-assessment").await.unwrap();
    assert_eq!(got.assessments, started.assessments);
}

#[tokio::test]
async fn get_trace_info_missing_errors() {
    let tmp = TempDb::new("get_missing").await;
    let s = store(&tmp).await;
    let err = s.get_trace_info(WS, "nope").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert!(err.message.contains("not found"), "{}", err.message);
}

#[tokio::test]
async fn start_trace_idempotent_overwrites() {
    let tmp = TempDb::new("start_idem").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    create_trace(&s, "tr", &exp, 1, Some(1), "OK", &[("a", "1")], &[]).await;
    // Re-start with new values overwrites top-level fields + upserts children.
    create_trace(
        &s,
        "tr",
        &exp,
        5,
        Some(9),
        "ERROR",
        &[("a", "2"), ("b", "3")],
        &[],
    )
    .await;
    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!(ti.request_time, 5);
    assert_eq!(ti.execution_duration, Some(9));
    assert_eq!(ti.state, "ERROR");
    assert_eq!(ti.tag("a"), Some("2"));
    assert_eq!(ti.tag("b"), Some("3"));
}

#[tokio::test]
async fn start_trace_preserves_existing_preview_on_conflict() {
    // Python guard: on the conflict path, a None request/response preview does
    // NOT clear an existing one. Simulate log_spans having backfilled a preview
    // by first starting a trace with previews, then re-starting with None.
    let tmp = TempDb::new("preview").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let mut input = trace_input("tr", &exp, 1, Some(1), "OK", &[], &[]);
    input.request_preview = Some("req".to_string());
    input.response_preview = Some("resp".to_string());
    s.start_trace(WS, &input).await.unwrap();

    // Re-start with None previews — existing values must survive.
    let input2 = trace_input("tr", &exp, 2, Some(2), "OK", &[], &[]);
    s.start_trace(WS, &input2).await.unwrap();
    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!(ti.request_preview.as_deref(), Some("req"));
    assert_eq!(ti.response_preview.as_deref(), Some("resp"));
    // Non-preview fields were overwritten.
    assert_eq!(ti.request_time, 2);
}

// ---------------------------------------------------------------------------
// batch-get
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batch_get_trace_infos_preserves_order_and_skips_missing() {
    let tmp = TempDb::new("batch").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    for id in ["a", "b", "c"] {
        create_trace(&s, id, &exp, 1, Some(1), "OK", &[], &[]).await;
    }
    let got = s
        .batch_get_trace_infos(WS, &["c".into(), "missing".into(), "a".into()])
        .await
        .unwrap();
    assert_eq!(ids(&got), vec!["c", "a"]);
}

// ---------------------------------------------------------------------------
// trace tag CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn trace_tag_set_delete() {
    let tmp = TempDb::new("tag").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    create_trace(&s, "tr", &exp, 1, Some(1), "OK", &[], &[]).await;

    s.set_trace_tag(WS, "tr", "k", "v1").await.unwrap();
    assert_eq!(
        s.get_trace_info(WS, "tr").await.unwrap().tag("k"),
        Some("v1")
    );
    // Upsert overwrites.
    s.set_trace_tag(WS, "tr", "k", "v2").await.unwrap();
    assert_eq!(
        s.get_trace_info(WS, "tr").await.unwrap().tag("k"),
        Some("v2")
    );

    s.delete_trace_tag(WS, "tr", "k").await.unwrap();
    assert_eq!(s.get_trace_info(WS, "tr").await.unwrap().tag("k"), None);

    // Deleting a missing tag errors.
    let err = s.delete_trace_tag(WS, "tr", "k").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert!(
        err.message.contains("No trace tag with key 'k'"),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// link_traces_to_run
// ---------------------------------------------------------------------------

#[tokio::test]
async fn link_traces_to_run_dedups_and_limits() {
    let tmp = TempDb::new("link").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let run = s
        .create_run(WS, &exp, Some("u"), Some(0), Some("r"), &[])
        .await
        .unwrap();
    let run_id = run.info.run_id.clone();
    create_trace(&s, "t1", &exp, 1, Some(1), "OK", &[], &[]).await;

    // Empty is a no-op.
    s.link_traces_to_run(WS, &[], &run_id).await.unwrap();
    // Link, then re-link (dedup — no duplicate row / no error).
    s.link_traces_to_run(WS, &["t1".into()], &run_id)
        .await
        .unwrap();
    s.link_traces_to_run(WS, &["t1".into()], &run_id)
        .await
        .unwrap();

    // Over the limit errors.
    let too_many: Vec<String> = (0..=100).map(|i| format!("x{i}")).collect();
    let err = s
        .link_traces_to_run(WS, &too_many, &run_id)
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::InvalidParameterValue
    );
    assert!(
        err.message.contains("Cannot link more than 100"),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// delete_traces — both modes + HasField edges
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_traces_by_max_timestamp_inclusive() {
    let tmp = TempDb::new("del_ts").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    for i in 0..10 {
        create_trace(&s, &format!("tr-{i}"), &exp, i, Some(1), "OK", &[], &[]).await;
    }
    let deleted = s
        .delete_traces(WS, &exp, Some(3), None, None)
        .await
        .unwrap();
    assert_eq!(deleted, 4); // inclusive: 0,1,2,3
    let page = s
        .search_traces(WS, &[exp.clone()], None, 100, &[], None)
        .await
        .unwrap();
    assert_eq!(page.trace_infos.len(), 6);
    for t in &page.trace_infos {
        assert!(t.request_time >= 4);
    }
}

#[tokio::test]
async fn delete_traces_max_count_oldest_first() {
    let tmp = TempDb::new("del_count").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    for i in 0..10 {
        create_trace(&s, &format!("tr-{i}"), &exp, i, Some(1), "OK", &[], &[]).await;
    }
    let deleted = s
        .delete_traces(WS, &exp, Some(10), Some(4), None)
        .await
        .unwrap();
    assert_eq!(deleted, 4);
    let page = s
        .search_traces(WS, &[exp.clone()], None, 100, &[], None)
        .await
        .unwrap();
    assert_eq!(page.trace_infos.len(), 6);
    for t in &page.trace_infos {
        assert!(t.request_time >= 4);
    }
}

#[tokio::test]
async fn delete_traces_by_ids() {
    let tmp = TempDb::new("del_ids").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    for i in 0..10 {
        create_trace(&s, &format!("tr-{i}"), &exp, i, Some(1), "OK", &[], &[]).await;
    }
    let to_del: Vec<String> = (0..8).map(|i| format!("tr-{i}")).collect();
    let deleted = s
        .delete_traces(WS, &exp, None, None, Some(&to_del))
        .await
        .unwrap();
    assert_eq!(deleted, 8);
    let page = s
        .search_traces(WS, &[exp.clone()], None, 100, &["timestamp".into()], None)
        .await
        .unwrap();
    assert_eq!(ids(&page.trace_infos), vec!["tr-8", "tr-9"]);
}

#[tokio::test]
async fn delete_traces_cascades_children() {
    let tmp = TempDb::new("del_cascade").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    create_trace(
        &s,
        "tr",
        &exp,
        1,
        Some(1),
        "OK",
        &[("tg", "v")],
        &[("md", "w")],
    )
    .await;
    s.delete_traces(WS, &exp, None, None, Some(&["tr".into()]))
        .await
        .unwrap();
    // Trace gone (and its children cascaded — no error, empty search).
    let err = s.get_trace_info(WS, "tr").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
}

#[tokio::test]
async fn delete_traces_hasfield_validation() {
    let tmp = TempDb::new("del_validate").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();

    // Neither specified.
    let err = s
        .delete_traces(WS, &exp, None, None, None)
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("Either `max_timestamp_millis` or `trace_ids`"),
        "{}",
        err.message
    );
    // Both specified.
    let err = s
        .delete_traces(WS, &exp, Some(100), None, Some(&["x".into()]))
        .await
        .unwrap_err();
    assert!(err.message.contains("Only one of"), "{}", err.message);
    // max_traces with trace_ids.
    let err = s
        .delete_traces(WS, &exp, None, Some(2), Some(&["x".into()]))
        .await
        .unwrap_err();
    assert!(
        err.message.contains("can't be specified if `trace_ids`"),
        "{}",
        err.message
    );
    // max_traces <= 0.
    let err = s
        .delete_traces(WS, &exp, Some(100), Some(0), None)
        .await
        .unwrap_err();
    assert!(
        err.message.contains("must be a positive integer"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn delete_traces_max_timestamp_zero_is_set_not_unset() {
    // HasField edge: Some(0) is a real filter (delete traces at/before ts 0),
    // distinct from None (unset → validation error).
    let tmp = TempDb::new("del_zero").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    create_trace(&s, "at-zero", &exp, 0, Some(1), "OK", &[], &[]).await;
    create_trace(&s, "at-one", &exp, 1, Some(1), "OK", &[], &[]).await;
    let deleted = s
        .delete_traces(WS, &exp, Some(0), None, None)
        .await
        .unwrap();
    assert_eq!(deleted, 1);
    assert!(s.get_trace_info(WS, "at-zero").await.is_err());
    assert!(s.get_trace_info(WS, "at-one").await.is_ok());
}

// ---------------------------------------------------------------------------
// workspace isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workspace_isolation() {
    let tmp = TempDb::new("ws").await;
    let s = store(&tmp).await;
    // Two experiments in two workspaces.
    let exp_a = s.create_experiment("wsA", "ea", None, &[]).await.unwrap();
    let exp_b = s.create_experiment("wsB", "eb", None, &[]).await.unwrap();
    // Start a trace in each workspace.
    let ia = trace_input("ta", &exp_a, 1, Some(1), "OK", &[], &[]);
    let ib = trace_input("tb", &exp_b, 1, Some(1), "OK", &[], &[]);
    s.start_trace("wsA", &ia).await.unwrap();
    s.start_trace("wsB", &ib).await.unwrap();

    // get_trace_info is scoped: wsB cannot see ta.
    assert!(s.get_trace_info("wsB", "ta").await.is_err());
    assert!(s.get_trace_info("wsA", "ta").await.is_ok());

    // search is scoped by experiment workspace.
    let page = s
        .search_traces("wsB", &[exp_a.clone()], None, 100, &[], None)
        .await
        .unwrap();
    assert!(page.trace_infos.is_empty());

    // set_trace_tag on a foreign-workspace trace errors.
    let err = s.set_trace_tag("wsB", "ta", "k", "v").await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
}
