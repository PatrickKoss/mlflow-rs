//! Behavioral integration tests for assessments (plan T2.12), ported from
//! `tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_assessments.py`.
//!
//! Each test copies the checked-in SQLite fixture into a temp file (same
//! pattern as `tracking_store.rs`/`datasets_store.rs`), so the committed
//! fixture is never mutated.
//!
//! T2.10 (the traces store) is being implemented in parallel and isn't landed
//! in this tree yet, so traces are inserted directly into `trace_info` via raw
//! SQL against the fixture's already-migrated schema, rather than through a
//! store method.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlflow_error::MlflowError;
use mlflow_store::{
    Assessment, AssessmentError, AssessmentSource, AssessmentUpdate, AssessmentValue, Db,
    FeedbackUpdate, NewAssessment, PoolConfig, TrackingStore,
};

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_assessments_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixture_path(), &path).expect("copy fixture");
        TempDb { path }
    }

    fn uri(&self) -> String {
        format!("sqlite:///{}", self.path.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn store(temp: &TempDb) -> TrackingStore {
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect temp fixture");
    TrackingStore::new(db, ART_ROOT)
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}_{}",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    )
}

async fn new_experiment(s: &TrackingStore) -> String {
    s.create_experiment(WS, &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap()
}

/// Insert a minimal `trace_info` row directly (T2.10 isn't landed in this
/// tree yet). Mirrors the columns `start_trace` would populate.
async fn insert_trace(s: &TrackingStore, experiment_id: &str, trace_id: &str) {
    let Db::Sqlite(pool) = s.db() else {
        panic!("expected sqlite pool in tests");
    };
    sqlx::query(
        "INSERT INTO trace_info \
         (request_id, experiment_id, timestamp_ms, execution_time_ms, status, \
          client_request_id, request_preview, response_preview, db_payload_generation) \
         VALUES (?, ?, ?, ?, 'OK', ?, NULL, NULL, 0)",
    )
    .bind(trace_id)
    .bind(experiment_id.parse::<i64>().unwrap())
    .bind(0i64)
    .bind(0i64)
    .bind(trace_id)
    .execute(pool)
    .await
    .expect("insert trace_info");
}

async fn new_trace(s: &TrackingStore, experiment_id: &str) -> String {
    let trace_id = format!("tr-{}", uuid_like());
    insert_trace(s, experiment_id, &trace_id).await;
    trace_id
}

fn human_source(id: &str) -> AssessmentSource {
    AssessmentSource {
        source_type: "HUMAN".to_string(),
        source_id: Some(id.to_string()),
    }
}

fn code_source() -> AssessmentSource {
    AssessmentSource {
        source_type: "CODE".to_string(),
        source_id: None,
    }
}

fn llm_judge_source() -> AssessmentSource {
    AssessmentSource {
        source_type: "LLM_JUDGE".to_string(),
        source_id: None,
    }
}

fn metadata(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn feedback_value(value: &serde_json::Value) -> AssessmentValue {
    AssessmentValue::Feedback {
        value_json: value.to_string(),
        error: None,
    }
}

fn expectation_value(value: &serde_json::Value) -> AssessmentValue {
    AssessmentValue::Expectation {
        value_json: value.to_string(),
    }
}

fn new_feedback(trace_id: &str, name: &str, value: &serde_json::Value) -> NewAssessment {
    NewAssessment {
        trace_id: trace_id.to_string(),
        name: name.to_string(),
        value: feedback_value(value),
        source: code_source(),
        run_id: None,
        span_id: None,
        rationale: None,
        metadata: None,
        create_time_ms: None,
        last_update_time_ms: None,
        assessment_id: None,
        overrides: None,
    }
}

fn feedback_bool(value: bool) -> serde_json::Value {
    serde_json::Value::Bool(value)
}

fn feedback_str(value: &str) -> serde_json::Value {
    serde_json::Value::String(value.to_string())
}

// ---------------------------------------------------------------------------
// create + get
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_feedback_and_expectation() {
    let tmp = TempDb::new("create_get");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let feedback = NewAssessment {
        trace_id: trace_id.clone(),
        name: "correctness".to_string(),
        value: feedback_value(&feedback_bool(true)),
        source: human_source("evaluator@company.com"),
        run_id: None,
        span_id: Some("span-123".to_string()),
        rationale: Some("The response is correct and well-formatted".to_string()),
        metadata: Some(metadata(&[("project", "test-project"), ("version", "1.0")])),
        create_time_ms: None,
        last_update_time_ms: None,
        assessment_id: None,
        overrides: None,
    };
    let created_feedback = s.create_assessment(WS, feedback).await.unwrap();
    assert!(created_feedback.assessment_id.starts_with("a-"));
    assert_eq!(created_feedback.trace_id, trace_id);
    assert_eq!(created_feedback.name, "correctness");
    assert_eq!(
        as_feedback_json(&created_feedback.value),
        feedback_bool(true)
    );
    assert_eq!(
        created_feedback.rationale.as_deref(),
        Some("The response is correct and well-formatted")
    );
    assert_eq!(
        created_feedback.metadata,
        Some(metadata(&[("project", "test-project"), ("version", "1.0")]))
    );
    assert_eq!(created_feedback.span_id.as_deref(), Some("span-123"));
    assert!(created_feedback.valid);

    let expectation = NewAssessment {
        trace_id: trace_id.clone(),
        name: "expected_response".to_string(),
        value: expectation_value(&feedback_str("The capital of France is Paris.")),
        source: human_source("annotator@company.com"),
        run_id: None,
        span_id: Some("span-456".to_string()),
        rationale: None,
        metadata: Some(metadata(&[
            ("context", "geography-qa"),
            ("difficulty", "easy"),
        ])),
        create_time_ms: None,
        last_update_time_ms: None,
        assessment_id: None,
        overrides: None,
    };
    let created_expectation = s.create_assessment(WS, expectation).await.unwrap();
    assert_ne!(
        created_expectation.assessment_id,
        created_feedback.assessment_id
    );
    assert_eq!(
        as_expectation_json(&created_expectation.value),
        feedback_str("The capital of France is Paris.")
    );
    assert!(created_expectation.valid);

    let retrieved_feedback = s
        .get_assessment(WS, &trace_id, &created_feedback.assessment_id)
        .await
        .unwrap();
    assert_eq!(retrieved_feedback.name, "correctness");
    assert_eq!(
        as_feedback_json(&retrieved_feedback.value),
        feedback_bool(true)
    );
    assert_eq!(
        retrieved_feedback.rationale.as_deref(),
        Some("The response is correct and well-formatted")
    );
    assert!(retrieved_feedback.valid);

    let retrieved_expectation = s
        .get_assessment(WS, &trace_id, &created_expectation.assessment_id)
        .await
        .unwrap();
    assert_eq!(
        as_expectation_json(&retrieved_expectation.value),
        feedback_str("The capital of France is Paris.")
    );
    assert!(retrieved_expectation.valid);
}

fn as_feedback_json(value: &AssessmentValue) -> serde_json::Value {
    match value {
        AssessmentValue::Feedback { value_json, .. } => serde_json::from_str(value_json).unwrap(),
        other => panic!("expected feedback, got {other:?}"),
    }
}

fn as_expectation_json(value: &AssessmentValue) -> serde_json::Value {
    match value {
        AssessmentValue::Expectation { value_json } => serde_json::from_str(value_json).unwrap(),
        other => panic!("expected expectation, got {other:?}"),
    }
}

#[tokio::test]
async fn get_assessment_errors() {
    let tmp = TempDb::new("get_errors");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let err = s
        .get_assessment(WS, "fake_trace", "fake_assessment")
        .await
        .unwrap_err();
    assert!(err.message.contains("fake_trace"), "{}", err.message);
    assert!(err.message.contains("not found"), "{}", err.message);

    let err = s
        .get_assessment(WS, &trace_id, "fake_assessment")
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("Assessment with ID 'fake_assessment' not found for trace"),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_assessment_feedback() {
    let tmp = TempDb::new("update_feedback");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = NewAssessment {
        rationale: Some("Original rationale".to_string()),
        metadata: Some(metadata(&[("project", "test-project"), ("version", "1.0")])),
        span_id: Some("span-123".to_string()),
        source: human_source("evaluator@company.com"),
        ..new_feedback(&trace_id, "correctness", &feedback_bool(true))
    };
    let created = s.create_assessment(WS, original).await.unwrap();

    let update = AssessmentUpdate {
        name: Some("correctness_updated".to_string()),
        feedback: Some(FeedbackUpdate {
            value_json: feedback_bool(false).to_string(),
            error: None,
        }),
        rationale: Some("Updated rationale".to_string()),
        metadata: Some(metadata(&[
            ("project", "test-project"),
            ("version", "2.0"),
            ("new_field", "added"),
        ])),
        ..Default::default()
    };
    let updated = s
        .update_assessment(WS, &trace_id, &created.assessment_id, update)
        .await
        .unwrap();

    assert_eq!(updated.assessment_id, created.assessment_id);
    assert_eq!(updated.name, "correctness_updated");
    assert_eq!(as_feedback_json(&updated.value), feedback_bool(false));
    assert_eq!(updated.rationale.as_deref(), Some("Updated rationale"));
    assert_eq!(
        updated.metadata,
        Some(metadata(&[
            ("project", "test-project"),
            ("version", "2.0"),
            ("new_field", "added"),
        ]))
    );
    assert_eq!(updated.span_id.as_deref(), Some("span-123"));
    assert_eq!(
        updated.source.source_id.as_deref(),
        Some("evaluator@company.com")
    );
    assert!(updated.valid);

    let retrieved = s
        .get_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();
    assert_eq!(as_feedback_json(&retrieved.value), feedback_bool(false));
    assert_eq!(retrieved.name, "correctness_updated");
    assert_eq!(retrieved.rationale.as_deref(), Some("Updated rationale"));
}

/// `feedback.value` and `feedback.error` travel together as one `FeedbackValue`
/// (mirrors Python's `new_value, new_error = feedback.value, feedback.error`):
/// supplying a new feedback value with no error clears a previously-recorded
/// error rather than preserving it independently.
#[tokio::test]
async fn update_assessment_feedback_clears_prior_error() {
    let tmp = TempDb::new("update_feedback_clears_error");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = NewAssessment {
        value: AssessmentValue::Feedback {
            value_json: "null".to_string(),
            error: Some(AssessmentError {
                error_code: "ValueError".to_string(),
                error_message: Some("boom".to_string()),
                stack_trace: None,
            }),
        },
        ..new_feedback(&trace_id, "with_error", &serde_json::Value::Null)
    };
    let created = s.create_assessment(WS, original).await.unwrap();
    assert!(as_error(&created.value).is_some());

    let updated = s
        .update_assessment(
            WS,
            &trace_id,
            &created.assessment_id,
            AssessmentUpdate {
                feedback: Some(FeedbackUpdate {
                    value_json: feedback_str("recovered").to_string(),
                    error: None,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(as_feedback_json(&updated.value), feedback_str("recovered"));
    assert!(as_error(&updated.value).is_none());

    let retrieved = s
        .get_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();
    assert!(as_error(&retrieved.value).is_none());
}

#[tokio::test]
async fn update_assessment_expectation() {
    let tmp = TempDb::new("update_expectation");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = NewAssessment {
        trace_id: trace_id.clone(),
        name: "expected_response".to_string(),
        value: expectation_value(&feedback_str("The capital of France is Paris.")),
        source: human_source("annotator@company.com"),
        run_id: None,
        span_id: Some("span-456".to_string()),
        rationale: None,
        metadata: Some(metadata(&[("context", "geography-qa")])),
        create_time_ms: None,
        last_update_time_ms: None,
        assessment_id: None,
        overrides: None,
    };
    let created = s.create_assessment(WS, original).await.unwrap();

    let update = AssessmentUpdate {
        expectation_value_json: Some(
            feedback_str("The capital and largest city of France is Paris.").to_string(),
        ),
        metadata: Some(metadata(&[
            ("context", "geography-qa"),
            ("updated", "true"),
        ])),
        ..Default::default()
    };
    let updated = s
        .update_assessment(WS, &trace_id, &created.assessment_id, update)
        .await
        .unwrap();

    assert_eq!(updated.assessment_id, created.assessment_id);
    assert_eq!(updated.name, "expected_response");
    assert_eq!(
        as_expectation_json(&updated.value),
        feedback_str("The capital and largest city of France is Paris.")
    );
    assert_eq!(
        updated.metadata,
        Some(metadata(&[
            ("context", "geography-qa"),
            ("updated", "true")
        ]))
    );
    assert_eq!(updated.span_id.as_deref(), Some("span-456"));
    assert_eq!(
        updated.source.source_id.as_deref(),
        Some("annotator@company.com")
    );
}

#[tokio::test]
async fn update_assessment_partial_fields_preserves_others() {
    let tmp = TempDb::new("update_partial");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = NewAssessment {
        rationale: Some("Original rationale".to_string()),
        metadata: Some(metadata(&[("scorer", "automated")])),
        source: code_source(),
        ..new_feedback(&trace_id, "quality", &serde_json::json!(5))
    };
    let created = s.create_assessment(WS, original).await.unwrap();

    let update = AssessmentUpdate {
        rationale: Some("Updated rationale only".to_string()),
        ..Default::default()
    };
    let updated = s
        .update_assessment(WS, &trace_id, &created.assessment_id, update)
        .await
        .unwrap();

    assert_eq!(updated.name, "quality");
    assert_eq!(as_feedback_json(&updated.value), serde_json::json!(5));
    assert_eq!(updated.rationale.as_deref(), Some("Updated rationale only"));
    assert_eq!(updated.metadata, Some(metadata(&[("scorer", "automated")])));
}

#[tokio::test]
async fn update_assessment_type_validation() {
    let tmp = TempDb::new("update_type_validation");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let feedback = new_feedback(&trace_id, "test_feedback", &feedback_str("original"));
    let created_feedback = s.create_assessment(WS, feedback).await.unwrap();

    let err = s
        .update_assessment(
            WS,
            &trace_id,
            &created_feedback.assessment_id,
            AssessmentUpdate {
                expectation_value_json: Some(feedback_str("This should fail").to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("Cannot update expectation value on a Feedback assessment"),
        "{}",
        err.message
    );

    let expectation = NewAssessment {
        trace_id: trace_id.clone(),
        name: "test_expectation".to_string(),
        value: expectation_value(&feedback_str("original_expected")),
        source: human_source("default"),
        run_id: None,
        span_id: None,
        rationale: None,
        metadata: None,
        create_time_ms: None,
        last_update_time_ms: None,
        assessment_id: None,
        overrides: None,
    };
    let created_expectation = s.create_assessment(WS, expectation).await.unwrap();

    let err = s
        .update_assessment(
            WS,
            &trace_id,
            &created_expectation.assessment_id,
            AssessmentUpdate {
                feedback: Some(FeedbackUpdate {
                    value_json: feedback_str("This should fail").to_string(),
                    error: None,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("Cannot update feedback value on an Expectation assessment"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn update_assessment_errors() {
    let tmp = TempDb::new("update_errors");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let err = s
        .update_assessment(
            WS,
            "fake_trace",
            "fake_assessment",
            AssessmentUpdate {
                rationale: Some("This should fail".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(err.message.contains("fake_trace"));
    assert!(err.message.contains("not found"));

    let err = s
        .update_assessment(
            WS,
            &trace_id,
            "fake_assessment",
            AssessmentUpdate {
                rationale: Some("This should fail".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("Assessment with ID 'fake_assessment' not found for trace"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn update_assessment_metadata_merging() {
    let tmp = TempDb::new("update_metadata_merge");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = NewAssessment {
        metadata: Some(metadata(&[
            ("keep", "this"),
            ("override", "old_value"),
            ("remove_me", "will_stay"),
        ])),
        ..new_feedback(&trace_id, "test", &feedback_str("original"))
    };
    let created = s.create_assessment(WS, original).await.unwrap();

    let update = AssessmentUpdate {
        metadata: Some(metadata(&[
            ("override", "new_value"),
            ("new_key", "new_value"),
        ])),
        ..Default::default()
    };
    let updated = s
        .update_assessment(WS, &trace_id, &created.assessment_id, update)
        .await
        .unwrap();

    assert_eq!(
        updated.metadata,
        Some(metadata(&[
            ("keep", "this"),
            ("override", "new_value"),
            ("remove_me", "will_stay"),
            ("new_key", "new_value"),
        ]))
    );
}

#[tokio::test]
async fn update_assessment_timestamps() {
    let tmp = TempDb::new("update_timestamps");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let created = s
        .create_assessment(
            WS,
            new_feedback(&trace_id, "test", &feedback_str("original")),
        )
        .await
        .unwrap();
    let original_create_time = created.create_time_ms;
    let original_update_time = created.last_update_time_ms;

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let updated = s
        .update_assessment(
            WS,
            &trace_id,
            &created.assessment_id,
            AssessmentUpdate {
                name: Some("updated_name".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(updated.create_time_ms, original_create_time);
    assert!(updated.last_update_time_ms > original_update_time);
}

// ---------------------------------------------------------------------------
// overrides
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_assessment_with_overrides() {
    let tmp = TempDb::new("overrides");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = NewAssessment {
        source: llm_judge_source(),
        ..new_feedback(&trace_id, "quality", &feedback_str("poor"))
    };
    let created_original = s.create_assessment(WS, original).await.unwrap();

    let overriding = NewAssessment {
        source: human_source("default"),
        overrides: Some(created_original.assessment_id.clone()),
        ..new_feedback(&trace_id, "quality", &feedback_str("excellent"))
    };
    let created_override = s.create_assessment(WS, overriding).await.unwrap();

    assert_eq!(
        created_override.overrides.as_deref(),
        Some(created_original.assessment_id.as_str())
    );
    assert_eq!(
        as_feedback_json(&created_override.value),
        feedback_str("excellent")
    );
    assert!(created_override.valid);

    let retrieved_original = s
        .get_assessment(WS, &trace_id, &created_original.assessment_id)
        .await
        .unwrap();
    assert!(!retrieved_original.valid);
    assert_eq!(
        as_feedback_json(&retrieved_original.value),
        feedback_str("poor")
    );
}

#[tokio::test]
async fn create_assessment_override_nonexistent() {
    let tmp = TempDb::new("override_nonexistent");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let overriding = NewAssessment {
        source: human_source("default"),
        overrides: Some("nonexistent-assessment-id".to_string()),
        ..new_feedback(&trace_id, "quality", &feedback_str("excellent"))
    };
    let err = s.create_assessment(WS, overriding).await.unwrap_err();
    assert!(
        err.message
            .contains("Assessment with ID 'nonexistent-assessment-id' not found"),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_assessment_idempotent() {
    let tmp = TempDb::new("delete_idempotent");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let created = s
        .create_assessment(
            WS,
            new_feedback(&trace_id, "test", &feedback_str("test_value")),
        )
        .await
        .unwrap();

    let retrieved = s
        .get_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();
    assert_eq!(retrieved.assessment_id, created.assessment_id);

    s.delete_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();

    let err = s
        .get_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap_err();
    assert!(
        err.message.contains(&format!(
            "Assessment with ID '{}' not found for trace",
            created.assessment_id
        )),
        "{}",
        err.message
    );

    // Idempotent: deleting an already-deleted or never-existing assessment is a no-op.
    s.delete_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();
    s.delete_assessment(WS, &trace_id, "fake_assessment_id")
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_assessment_restores_overridden_validity() {
    let tmp = TempDb::new("delete_override_restore");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let original = s
        .create_assessment(
            WS,
            NewAssessment {
                source: code_source(),
                ..new_feedback(&trace_id, "original", &feedback_str("original_value"))
            },
        )
        .await
        .unwrap();

    let overriding = NewAssessment {
        source: human_source("default"),
        overrides: Some(original.assessment_id.clone()),
        ..new_feedback(&trace_id, "override", &feedback_str("override_value"))
    };
    let overriding = s.create_assessment(WS, overriding).await.unwrap();

    assert!(
        !s.get_assessment(WS, &trace_id, &original.assessment_id)
            .await
            .unwrap()
            .valid
    );
    assert!(
        s.get_assessment(WS, &trace_id, &overriding.assessment_id)
            .await
            .unwrap()
            .valid
    );

    s.delete_assessment(WS, &trace_id, &overriding.assessment_id)
        .await
        .unwrap();

    let err = s
        .get_assessment(WS, &trace_id, &overriding.assessment_id)
        .await
        .unwrap_err();
    assert!(err.message.contains("not found"));
    assert!(
        s.get_assessment(WS, &trace_id, &original.assessment_id)
            .await
            .unwrap()
            .valid
    );
}

// ---------------------------------------------------------------------------
// run_id / error payloads
// ---------------------------------------------------------------------------

#[tokio::test]
async fn assessment_with_run_id() {
    let tmp = TempDb::new("run_id");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;
    let run_id = s
        .create_run(WS, &exp, None, Some(0), Some("test_run"), &[])
        .await
        .unwrap()
        .info
        .run_id;

    let feedback = NewAssessment {
        run_id: Some(run_id.clone()),
        ..new_feedback(&trace_id, "run_feedback", &feedback_str("excellent"))
    };
    let created = s.create_assessment(WS, feedback).await.unwrap();
    assert_eq!(created.run_id.as_deref(), Some(run_id.as_str()));

    let retrieved = s
        .get_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();
    assert_eq!(retrieved.run_id.as_deref(), Some(run_id.as_str()));
}

#[tokio::test]
async fn assessment_with_error_round_trips() {
    let tmp = TempDb::new("with_error");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let feedback = NewAssessment {
        value: AssessmentValue::Feedback {
            value_json: "null".to_string(),
            error: Some(AssessmentError {
                error_code: "ValueError".to_string(),
                error_message: Some("Test error message".to_string()),
                stack_trace: Some("Traceback...\nValueError: Test error message".to_string()),
            }),
        },
        ..new_feedback(&trace_id, "error_feedback", &serde_json::Value::Null)
    };
    let created = s.create_assessment(WS, feedback).await.unwrap();
    let created_error = as_error(&created.value).expect("error present");
    assert_eq!(
        created_error.error_message.as_deref(),
        Some("Test error message")
    );
    assert_eq!(created_error.error_code, "ValueError");
    assert!(created_error
        .stack_trace
        .as_deref()
        .unwrap()
        .contains("ValueError: Test error message"));

    let retrieved = s
        .get_assessment(WS, &trace_id, &created.assessment_id)
        .await
        .unwrap();
    let retrieved_error = as_error(&retrieved.value).expect("error present");
    assert_eq!(
        retrieved_error.error_message.as_deref(),
        Some("Test error message")
    );
    assert_eq!(retrieved_error.error_code, "ValueError");
    assert_eq!(retrieved_error.stack_trace, created_error.stack_trace);
}

fn as_error(value: &AssessmentValue) -> Option<AssessmentError> {
    match value {
        AssessmentValue::Feedback { error, .. } => error.clone(),
        other => panic!("expected feedback, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// missing-trace / workspace isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_assessment_for_missing_trace_returns_not_found() {
    let tmp = TempDb::new("missing_trace");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    // Do not insert a trace row: "tr-doomed" never existed.
    let _ = &exp;

    let feedback = new_feedback("tr-doomed", "quality", &feedback_str("looks good"));
    let err = s.create_assessment(WS, feedback).await.unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );
    assert!(err.message.contains("not found"), "{}", err.message);
    assert!(!err.message.contains("IntegrityError"));
    assert!(!err.message.contains("INSERT INTO"));
}

#[tokio::test]
async fn create_assessment_duplicate_id_is_constraint_violation() {
    let tmp = TempDb::new("dup_id");
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let trace_id = new_trace(&s, &exp).await;

    let first = NewAssessment {
        assessment_id: Some("a-duplicate-id".to_string()),
        source: human_source("reviewer"),
        ..new_feedback(&trace_id, "quality", &feedback_str("a"))
    };
    s.create_assessment(WS, first).await.unwrap();

    let second = NewAssessment {
        assessment_id: Some("a-duplicate-id".to_string()),
        source: human_source("reviewer"),
        ..new_feedback(&trace_id, "quality", &feedback_str("b"))
    };
    let err: MlflowError = s.create_assessment(WS, second).await.unwrap_err();
    assert_eq!(err.error_code, mlflow_error::ErrorCode::InternalError);
    assert!(
        err.message.contains("constraint violation"),
        "{}",
        err.message
    );
    assert!(!err.message.contains("not found"));
    assert!(!err.message.contains("IntegrityError"));
    assert!(!err.message.contains("INSERT INTO"));
}

#[tokio::test]
async fn assessment_workspace_isolation() {
    let tmp = TempDb::new("workspace_isolation");
    let s = store(&tmp).await;

    let exp_id = s
        .create_experiment("ws-a", &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap();
    let trace_id = format!("tr-{}", uuid_like());
    insert_trace(&s, &exp_id, &trace_id).await;

    let created = s
        .create_assessment(
            "ws-a",
            new_feedback(&trace_id, "quality", &feedback_str("value")),
        )
        .await
        .unwrap();

    // Same trace/assessment id, wrong workspace: not found.
    let err = s
        .get_assessment("ws-b", &trace_id, &created.assessment_id)
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );

    let err = s
        .create_assessment(
            "ws-b",
            new_feedback(&trace_id, "quality", &feedback_str("value2")),
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.error_code,
        mlflow_error::ErrorCode::ResourceDoesNotExist
    );

    // The right workspace still works.
    let fetched: Assessment = s
        .get_assessment("ws-a", &trace_id, &created.assessment_id)
        .await
        .unwrap();
    assert_eq!(fetched.assessment_id, created.assessment_id);
}
