//! HTTP integration tests for the assessment endpoints (plan T4.4, §3.9).
//!
//! Boots the axum app on a real ephemeral socket (same pattern as
//! `logged_models_http.rs`) against a fresh copy of the committed
//! Alembic-migrated SQLite fixture. Traces are created directly through
//! [`TrackingStore::start_trace`] (T2.10, landed in this tree) rather than
//! over HTTP, since the trace-creation endpoints are T4.1's territory (a
//! parallel task) and out of scope here; the assessment endpoints under test
//! only need an existing `trace_id` to hang off.
//!
//! Ported scenarios mirror `test_assessments_end_to_end`
//! (`tests/tracking/test_rest_tracking.py`): create feedback + expectation,
//! get, update (each `update_mask` path individually, plus `valid` and an
//! unknown path), override/supersede + un-invalidation on delete, and the
//! missing-trace/missing-assessment error bodies.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, StartTraceInput, TrackingStore};
use serde_json::Value;
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
/// Experiment "0" ("Default") from the committed fixture.
const EXP_ID: &str = "0";
const WORKSPACE: &str = "default";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

/// Copy the committed fixture to a unique temp file; removed on drop.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_server_assessments_{}_{}_{}.db",
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

/// A running test server with a base URL and a handle to the same store
/// (used to create traces to hang assessments off, bypassing the trace HTTP
/// endpoints which are out of scope for this task).
struct TestServer {
    base: String,
    store: TrackingStore,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    _db: TempDb,
}

impl TestServer {
    async fn start(tag: &str) -> Self {
        let db_file = TempDb::new(tag);
        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        let store = TrackingStore::new(db, ART_ROOT);
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            static_prefix: None,
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_destination: None,
            allowed_hosts: None,
            cors_allowed_origins: None,
            x_frame_options: "SAMEORIGIN".to_string(),
            ..Default::default()
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(&config, recorder, Some(AppState::new(store.clone())));

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("server error");
        });

        TestServer {
            base: format!("http://{addr}"),
            store,
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
            _db: db_file,
        }
    }

    /// Create a trace directly through the store (T2.10), returning its
    /// `trace_id`. Assessment endpoints require an existing trace; creating
    /// one is T4.1's territory over HTTP, so this bypasses HTTP entirely.
    async fn new_trace(&self) -> String {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let trace_id = format!("tr-assessments-http-{}-{n}", std::process::id());
        let input = StartTraceInput {
            trace_id: trace_id.clone(),
            experiment_id: EXP_ID.to_string(),
            request_time: 0,
            execution_duration: Some(0),
            state: "OK".to_string(),
            client_request_id: None,
            request_preview: None,
            response_preview: None,
            tags: Vec::new(),
            trace_metadata: Vec::new(),
            trace_metrics: Vec::new(),
        };
        self.store
            .start_trace(WORKSPACE, &input)
            .await
            .expect("start_trace");
        trace_id
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

struct HttpResponse {
    status: StatusCode,
    body: String,
}

impl HttpResponse {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON: {e}: {}", self.body))
    }
}

async fn send(server: &TestServer, method: Method, path: &str, body: Option<&str>) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{path}", server.base);
    let mut builder = Request::builder().method(method).uri(&url);
    let request = match body {
        Some(b) => {
            builder = builder.header("content-type", "application/json");
            builder.body(Full::<Bytes>::new(Bytes::from(b.to_string())))
        }
        None => builder.body(Full::<Bytes>::new(Bytes::new())),
    }
    .unwrap();

    let mut last = None;
    for _ in 0..50 {
        match client.request(clone_request(&request)).await {
            Ok(res) => {
                let status = res.status();
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return HttpResponse {
                    status,
                    body: String::from_utf8_lossy(&bytes).into_owned(),
                };
            }
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }
    panic!("failed to connect: {last:?}");
}

fn clone_request(req: &Request<Full<Bytes>>) -> Request<Full<Bytes>> {
    let mut builder = Request::builder()
        .method(req.method().clone())
        .uri(req.uri().clone());
    for (k, v) in req.headers() {
        builder = builder.header(k, v);
    }
    let body = req.body().clone();
    builder.body(body).unwrap()
}

const PREFIXES: [&str; 2] = ["/api/3.0", "/ajax-api/3.0"];

async fn post(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    send(
        server,
        Method::POST,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

async fn get(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::GET, &format!("{prefix}{endpoint}"), None).await
}

async fn patch(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    send(
        server,
        Method::PATCH,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

async fn delete(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::DELETE, &format!("{prefix}{endpoint}"), None).await
}

async fn create_feedback(server: &TestServer, prefix: &str, trace_id: &str) -> Value {
    let body = serde_json::json!({
        "assessment": {
            "assessment_name": "quality_score",
            "feedback": {"value": {"rating": 4, "comments": "Good response"}},
            "source": {"source_type": "HUMAN", "source_id": "evaluator@company.com"},
            "rationale": "Response was accurate and helpful",
            "metadata": {"model": "gpt-4", "version": "1.0"},
        }
    });
    let res = post(
        server,
        prefix,
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    res.json()["assessment"].clone()
}

// ---------------------------------------------------------------------------
// Create / get
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_feedback_assessment_on_both_prefixes() {
    let server = TestServer::start("crud_feedback").await;

    for (i, prefix) in PREFIXES.iter().enumerate() {
        let trace_id = server.new_trace().await;
        let assessment = create_feedback(&server, prefix, &trace_id).await;

        assert_eq!(assessment["assessment_name"], "quality_score");
        assert_eq!(assessment["feedback"]["value"]["rating"], 4.0);
        assert_eq!(assessment["source"]["source_type"], "HUMAN");
        assert_eq!(assessment["source"]["source_id"], "evaluator@company.com");
        assert_eq!(assessment["trace_id"], trace_id);
        assert_eq!(assessment["valid"], true);
        assert_eq!(assessment["metadata"]["model"], "gpt-4");
        let assessment_id = assessment["assessment_id"].as_str().unwrap().to_string();
        assert!(assessment_id.starts_with("a-"), "iter {i}: {assessment_id}");

        let get_res = get(
            &server,
            prefix,
            &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        )
        .await;
        assert_eq!(get_res.status, StatusCode::OK, "{}", get_res.body);
        let got = &get_res.json()["assessment"];
        assert_eq!(got["assessment_id"], assessment_id);
        assert_eq!(got["feedback"]["value"]["rating"], 4.0);
    }
}

#[tokio::test]
async fn create_and_get_expectation_assessment() {
    let server = TestServer::start("crud_expectation").await;
    let trace_id = server.new_trace().await;

    let body = serde_json::json!({
        "assessment": {
            "assessment_name": "response_time_check",
            "expectation": {"value": "under 2 seconds"},
            "source": {"source_type": "HUMAN", "source_id": "evaluator@company.com"},
        }
    });
    let create = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &body.to_string(),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);
    let assessment = &create.json()["assessment"];
    assert_eq!(assessment["expectation"]["value"], "under 2 seconds");
    let assessment_id = assessment["assessment_id"].as_str().unwrap().to_string();

    let got = get(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
    )
    .await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert_eq!(
        got.json()["assessment"]["expectation"]["value"],
        "under 2 seconds"
    );
}

#[tokio::test]
async fn create_assessment_without_a_value_variant_is_invalid_parameter_value() {
    // `Assessment.from_proto`'s `WhichOneof("value")` dispatch requires
    // exactly one of expectation/feedback/issue; none set is
    // `"Unknown assessment type: None"` (`assessment.py:157-168`).
    let server = TestServer::start("no_value_variant").await;
    let trace_id = server.new_trace().await;

    let body = serde_json::json!({
        "assessment": {
            "assessment_name": "quality_score",
            "source": {"source_type": "CODE"},
        }
    });
    let res = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(res.json()["message"], "Unknown assessment type: None");
}

#[tokio::test]
async fn create_assessment_missing_body_field_is_invalid_parameter_value() {
    let server = TestServer::start("missing_assessment").await;
    let trace_id = server.new_trace().await;

    let res = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        "{}",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(res.json()["message"]
        .as_str()
        .unwrap()
        .contains("Missing value for required parameter 'assessment'"));
}

#[tokio::test]
async fn create_assessment_for_missing_trace_is_resource_does_not_exist() {
    let server = TestServer::start("missing_trace").await;

    let body = serde_json::json!({
        "assessment": {
            "assessment_name": "quality_score",
            "feedback": {"value": true},
            "source": {"source_type": "CODE"},
        }
    });
    let res = post(
        &server,
        "/api/3.0",
        "/mlflow/traces/tr-does-not-exist/assessments",
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn get_missing_assessment_is_resource_does_not_exist() {
    let server = TestServer::start("get_missing").await;
    let trace_id = server.new_trace().await;

    let res = get(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/a-doesnotexist"),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn get_assessment_for_missing_trace_is_resource_does_not_exist() {
    let server = TestServer::start("get_missing_trace").await;

    let res = get(
        &server,
        "/api/3.0",
        "/mlflow/traces/tr-does-not-exist/assessments/a-whatever",
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

// ---------------------------------------------------------------------------
// Update: one test per FieldMask path, plus valid + unknown path handling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_assessment_name_rationale_feedback_metadata() {
    let server = TestServer::start("update_full").await;
    let trace_id = server.new_trace().await;
    let assessment = create_feedback(&server, "/api/3.0", &trace_id).await;
    let assessment_id = assessment["assessment_id"].as_str().unwrap();

    let body = serde_json::json!({
        "assessment": {
            "assessment_id": assessment_id,
            "trace_id": trace_id,
            "assessment_name": "updated_quality_score",
            "feedback": {"value": {"rating": 5, "comments": "Excellent response"}},
            "rationale": "Actually, the response was excellent",
            "metadata": {"model": "gpt-4", "version": "2.0"},
        },
        "update_mask": "assessmentName,feedback,rationale,metadata",
    });
    let res = patch(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let updated = &res.json()["assessment"];
    assert_eq!(updated["assessment_name"], "updated_quality_score");
    assert_eq!(updated["feedback"]["value"]["rating"], 5.0);
    assert_eq!(
        updated["feedback"]["value"]["comments"],
        "Excellent response"
    );
    assert_eq!(updated["rationale"], "Actually, the response was excellent");
    // Metadata merges (existing "model" key retained, "version" overwritten).
    assert_eq!(updated["metadata"]["model"], "gpt-4");
    assert_eq!(updated["metadata"]["version"], "2.0");
    // Source/span/create_time are immutable.
    assert_eq!(updated["source"]["source_type"], "HUMAN");
}

#[tokio::test]
async fn update_assessment_expectation_path() {
    let server = TestServer::start("update_expectation").await;
    let trace_id = server.new_trace().await;

    let create_body = serde_json::json!({
        "assessment": {
            "assessment_name": "response_time_check",
            "expectation": {"value": "under 2 seconds"},
            "source": {"source_type": "CODE"},
        }
    });
    let create = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &create_body.to_string(),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);
    let assessment_id = create.json()["assessment"]["assessment_id"]
        .as_str()
        .unwrap()
        .to_string();

    let update_body = serde_json::json!({
        "assessment": {
            "expectation": {"value": "under 1 second"},
        },
        "update_mask": "expectation",
    });
    let res = patch(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        &update_body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(
        res.json()["assessment"]["expectation"]["value"],
        "under 1 second"
    );
}

#[tokio::test]
async fn update_assessment_valid_path_is_internal_error() {
    // Python's `_update_assessment` builds `kwargs["valid"] =
    // assessment_proto.valid` for this FieldMask path and passes it to
    // `_get_tracking_store().update_assessment(...)`, which has no `valid`
    // parameter — an uncaught `TypeError` (not an `MlflowException`), so
    // `catch_mlflow_exception` does not convert it to a clean 4xx. This test
    // asserts the Rust port's faithful reproduction: an internal-error 500.
    let server = TestServer::start("update_valid").await;
    let trace_id = server.new_trace().await;
    let assessment = create_feedback(&server, "/api/3.0", &trace_id).await;
    let assessment_id = assessment["assessment_id"].as_str().unwrap();

    let body = serde_json::json!({
        "assessment": {"valid": false},
        "update_mask": "valid",
    });
    let res = patch(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        &body.to_string(),
    )
    .await;
    assert_eq!(
        res.status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        res.body
    );
    assert_eq!(res.json()["error_code"], "INTERNAL_ERROR");
}

#[tokio::test]
async fn update_assessment_unknown_mask_path_is_ignored() {
    // Python's `for path in update_mask.paths: if path == ...` chain has no
    // `else`/default branch, so an unrecognized path is silently ignored
    // (the assessment is still "updated" — `last_update_time` advances, but
    // no field actually changes).
    let server = TestServer::start("update_unknown_path").await;
    let trace_id = server.new_trace().await;
    let assessment = create_feedback(&server, "/api/3.0", &trace_id).await;
    let assessment_id = assessment["assessment_id"].as_str().unwrap();

    let body = serde_json::json!({
        "assessment": {"assessment_name": "should_not_apply"},
        "update_mask": "bogusUnknownPath",
    });
    let res = patch(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let updated = &res.json()["assessment"];
    assert_eq!(updated["assessment_name"], "quality_score");
}

#[tokio::test]
async fn update_assessment_missing_update_mask_is_invalid_parameter_value() {
    let server = TestServer::start("update_missing_mask").await;
    let trace_id = server.new_trace().await;
    let assessment = create_feedback(&server, "/api/3.0", &trace_id).await;
    let assessment_id = assessment["assessment_id"].as_str().unwrap();

    let body = serde_json::json!({
        "assessment": {"assessment_name": "x"},
    });
    let res = patch(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(res.json()["message"]
        .as_str()
        .unwrap()
        .contains("Missing value for required parameter 'update_mask'"));
}

#[tokio::test]
async fn update_feedback_value_on_expectation_is_invalid_parameter_value() {
    let server = TestServer::start("update_type_mismatch").await;
    let trace_id = server.new_trace().await;

    let create_body = serde_json::json!({
        "assessment": {
            "assessment_name": "response_time_check",
            "expectation": {"value": "under 2 seconds"},
            "source": {"source_type": "CODE"},
        }
    });
    let create = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &create_body.to_string(),
    )
    .await;
    let assessment_id = create.json()["assessment"]["assessment_id"]
        .as_str()
        .unwrap()
        .to_string();

    let update_body = serde_json::json!({
        "assessment": {"feedback": {"value": true}},
        "update_mask": "feedback",
    });
    let res = patch(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
        &update_body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// Delete + override/supersede + un-invalidation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn override_marks_original_invalid_and_delete_restores_it() {
    let server = TestServer::start("override_restore").await;
    let trace_id = server.new_trace().await;
    let original = create_feedback(&server, "/api/3.0", &trace_id).await;
    let original_id = original["assessment_id"].as_str().unwrap().to_string();

    let override_body = serde_json::json!({
        "assessment": {
            "assessment_name": "corrected_quality_score",
            "feedback": {"value": {"rating": 3, "comments": "Actually needs improvement"}},
            "source": {"source_type": "HUMAN", "source_id": "senior_evaluator@company.com"},
            "overrides": original_id,
        }
    });
    let override_res = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &override_body.to_string(),
    )
    .await;
    assert_eq!(override_res.status, StatusCode::OK, "{}", override_res.body);
    let override_assessment = &override_res.json()["assessment"];
    assert_eq!(override_assessment["valid"], true);
    assert_eq!(override_assessment["overrides"], original_id);
    let override_id = override_assessment["assessment_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Original is now invalid.
    let get_original = get(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{original_id}"),
    )
    .await;
    assert_eq!(get_original.status, StatusCode::OK, "{}", get_original.body);
    assert_eq!(get_original.json()["assessment"]["valid"], false);

    // Deleting the override restores the original.
    let del = delete(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{override_id}"),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);

    let get_deleted = get(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{override_id}"),
    )
    .await;
    assert_eq!(
        get_deleted.status,
        StatusCode::NOT_FOUND,
        "{}",
        get_deleted.body
    );

    let get_restored = get(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{original_id}"),
    )
    .await;
    assert_eq!(get_restored.status, StatusCode::OK, "{}", get_restored.body);
    assert_eq!(get_restored.json()["assessment"]["valid"], true);
}

#[tokio::test]
async fn override_missing_target_is_resource_does_not_exist() {
    let server = TestServer::start("override_missing").await;
    let trace_id = server.new_trace().await;

    let body = serde_json::json!({
        "assessment": {
            "assessment_name": "x",
            "feedback": {"value": true},
            "source": {"source_type": "CODE"},
            "overrides": "a-does-not-exist",
        }
    });
    let res = post(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments"),
        &body.to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn delete_missing_assessment_is_idempotent_no_op() {
    let server = TestServer::start("delete_missing").await;
    let trace_id = server.new_trace().await;

    let res = delete(
        &server,
        "/api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/a-does-not-exist"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
}

#[tokio::test]
async fn ajax_prefix_serves_assessments_too() {
    let server = TestServer::start("ajax_prefix").await;
    let trace_id = server.new_trace().await;
    let assessment = create_feedback(&server, "/ajax-api/3.0", &trace_id).await;
    let assessment_id = assessment["assessment_id"].as_str().unwrap();

    let got = get(
        &server,
        "/ajax-api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
    )
    .await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);

    let del = delete(
        &server,
        "/ajax-api/3.0",
        &format!("/mlflow/traces/{trace_id}/assessments/{assessment_id}"),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);
}
