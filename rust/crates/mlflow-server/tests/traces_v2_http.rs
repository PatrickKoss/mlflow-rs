//! HTTP integration tests for the Tracing V2 endpoints (plan T4.2, §3.7): the
//! 7 deprecated-but-still-served V2 trace RPCs under `/api/2.0/mlflow/traces...`.
//!
//! Mirrors the `traces_http.rs` (V3) harness: boots the axum app on a real
//! ephemeral socket against a fresh copy of the committed Alembic-migrated
//! SQLite fixture. Covers the happy path per endpoint, V2<->V3 store
//! consistency (a trace created via one API surface is visible through the
//! other), the GET search's repeated `experiment_ids` + filter + pagination,
//! both delete modes, tag CRUD, and verbatim error bodies.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const EXP_ID: &str = "0";

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
            "mlflow_rust_server_traces_v2_{}_{}_{}.db",
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

struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    #[allow(dead_code)]
    store: TrackingStore,
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
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
            store,
            _db: db_file,
        }
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

const V2_PREFIXES: [&str; 2] = ["/api/2.0", "/ajax-api/2.0"];

async fn post(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    send(
        server,
        Method::POST,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

async fn get_q(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::GET, &format!("{prefix}{endpoint}"), None).await
}

/// Start a V2 trace via `POST .../mlflow/traces` and return the parsed
/// response body (`{"trace_info": {...}}`).
async fn start_trace_v2(server: &TestServer, prefix: &str, exp_id: &str) -> Value {
    let body = json!({
        "experiment_id": exp_id,
        "timestamp_ms": 1_700_000_000_000i64,
        "request_metadata": [{"key": "mlflow.sourceType", "value": "NOTEBOOK"}],
        "tags": [{"key": "team", "value": "rust"}]
    })
    .to_string();
    let res = post(server, prefix, "/mlflow/traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    res.json()
}

/// Start a V3 trace (reused from the T4.1 fixture pattern) so V2<->V3 cross
/// visibility tests have a known trace id to work with.
async fn start_trace_v3(server: &TestServer, trace_id: &str, exp_id: &str, state: &str) -> Value {
    let body = json!({
        "trace": {
            "trace_info": {
                "trace_id": trace_id,
                "trace_location": {
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": exp_id}
                },
                "request_time": "2024-01-01T00:00:00Z",
                "execution_duration": "1.500s",
                "state": state,
                "tags": {"team": "rust"},
                "trace_metadata": {"mlflow.traceName": "my-trace"}
            }
        }
    })
    .to_string();
    let res = post(server, "/api/3.0", "/mlflow/traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    res.json()
}

// ---------------------------------------------------------------------------
// startTrace / endTrace (V2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn start_trace_v2_on_both_prefixes() {
    for prefix in V2_PREFIXES {
        let server = TestServer::start("start_v2").await;
        let started = start_trace_v2(&server, prefix, EXP_ID).await;
        let info = &started["trace_info"];
        assert_eq!(info["experiment_id"], EXP_ID);
        assert_eq!(info["status"], "IN_PROGRESS");
        // execution_time_ms substitutes 0 for an unset duration.
        assert_eq!(info["execution_time_ms"], 0);
        let tags = info["tags"].as_array().unwrap();
        assert!(tags
            .iter()
            .any(|t| t["key"] == "team" && t["value"] == "rust"));
        let metadata = info["request_metadata"].as_array().unwrap();
        assert!(metadata.iter().any(|m| m["key"] == "mlflow.sourceType"));
        // request_id is a plain uuid4 hex (32 lowercase hex chars, no "tr-" prefix).
        let request_id = info["request_id"].as_str().expect("request_id");
        assert_eq!(request_id.len(), 32);
        assert!(request_id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

#[tokio::test]
async fn start_trace_v2_then_end_trace_v2() {
    let server = TestServer::start("start_end_v2").await;
    let started = start_trace_v2(&server, "/api/2.0", EXP_ID).await;
    let request_id = started["trace_info"]["request_id"]
        .as_str()
        .unwrap()
        .to_string();

    let end_body = json!({
        "timestamp_ms": 1_700_000_005_000i64,
        "status": "OK",
        "request_metadata": [{"key": "mlflow.sourceRun", "value": "run-123"}],
        "tags": [{"key": "outcome", "value": "success"}]
    })
    .to_string();
    let res = send(
        &server,
        Method::PATCH,
        &format!("/api/2.0/mlflow/traces/{request_id}"),
        Some(&end_body),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let info = &res.json()["trace_info"];
    assert_eq!(info["status"], "OK");
    // 1_700_000_005_000 - 1_700_000_000_000 = 5000ms.
    assert_eq!(info["execution_time_ms"], 5000);
    let tags = info["tags"].as_array().unwrap();
    assert!(tags
        .iter()
        .any(|t| t["key"] == "outcome" && t["value"] == "success"));

    // endTrace merges metadata/tags on top of what startTrace wrote (both
    // present afterward).
    let keys: Vec<&str> = info["request_metadata"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["key"].as_str().unwrap())
        .collect();
    assert!(keys.contains(&"mlflow.sourceType"));
    assert!(keys.contains(&"mlflow.sourceRun"));
}

#[tokio::test]
async fn end_trace_v2_missing_trace_is_resource_does_not_exist() {
    let server = TestServer::start("end_missing").await;
    let res = send(
        &server,
        Method::PATCH,
        "/api/2.0/mlflow/traces/does-not-exist",
        Some(&json!({"timestamp_ms": 1, "status": "OK"}).to_string()),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn start_trace_v2_rejects_metadata_entry_without_key() {
    let server = TestServer::start("start_v2_bad_metadata").await;
    let body = json!({
        "experiment_id": EXP_ID,
        "timestamp_ms": 1,
        "request_metadata": [{"value": "orphaned"}]
    })
    .to_string();
    let res = post(&server, "/api/2.0", "/mlflow/traces", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(
        res.body.contains("for parameter 'request_metadata'"),
        "{}",
        res.body
    );
}

// ---------------------------------------------------------------------------
// getTraceInfo (V2) — V2<->V3 store consistency
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_trace_info_v2_on_both_prefixes() {
    for prefix in V2_PREFIXES {
        let server = TestServer::start("get_info_v2").await;
        let started = start_trace_v2(&server, "/api/2.0", EXP_ID).await;
        let request_id = started["trace_info"]["request_id"].as_str().unwrap();

        let res = get_q(
            &server,
            prefix,
            &format!("/mlflow/traces/{request_id}/info"),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        assert_eq!(res.json()["trace_info"]["request_id"], request_id);
    }
}

#[tokio::test]
async fn get_trace_info_v2_missing_is_resource_does_not_exist() {
    let server = TestServer::start("get_info_missing").await;
    let res = get_q(&server, "/api/2.0", "/mlflow/traces/nope/info").await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn trace_started_via_v3_is_visible_via_v2_get_trace_info() {
    let server = TestServer::start("v3_then_v2").await;
    start_trace_v3(&server, "tr-cross-1", EXP_ID, "OK").await;

    let res = get_q(&server, "/api/2.0", "/mlflow/traces/tr-cross-1/info").await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["trace_info"]["request_id"], "tr-cross-1");
    assert_eq!(res.json()["trace_info"]["status"], "OK");
}

#[tokio::test]
async fn trace_started_via_v2_is_visible_via_v3_get_trace_info() {
    let server = TestServer::start("v2_then_v3").await;
    let started = start_trace_v2(&server, "/api/2.0", EXP_ID).await;
    let request_id = started["trace_info"]["request_id"]
        .as_str()
        .unwrap()
        .to_string();

    let res = get_q(&server, "/api/3.0", &format!("/mlflow/traces/{request_id}")).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["trace"]["trace_info"]["trace_id"], request_id);
    assert_eq!(res.json()["trace"]["trace_info"]["state"], "IN_PROGRESS");
}

// ---------------------------------------------------------------------------
// searchTraces (V2, GET) — also the UI's "contains traces" ajax call shape.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_traces_v2_get_with_repeated_experiment_ids_filter_and_pagination() {
    let server = TestServer::start("search_v2").await;
    for i in 0..3 {
        start_trace_v3(&server, &format!("sv2-{i}"), EXP_ID, "OK").await;
    }

    // Mirrors the UI's contains-traces call shape exactly: `GET
    // /ajax-api/2.0/mlflow/traces?experiment_ids=...&order_by=...&max_results=...&filter=...`.
    let res = get_q(
        &server,
        "/ajax-api/2.0",
        &format!(
            "/mlflow/traces?experiment_ids={EXP_ID}&order_by=timestamp_ms+DESC&max_results=2&filter="
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let page1 = res.json();
    let traces1 = page1["traces"].as_array().unwrap();
    assert_eq!(traces1.len(), 2);
    let token = page1["next_page_token"].as_str().expect("next page token");

    let res2 = get_q(
        &server,
        "/api/2.0",
        &format!("/mlflow/traces?experiment_ids={EXP_ID}&max_results=2&page_token={token}"),
    )
    .await;
    assert_eq!(res2.status, StatusCode::OK, "{}", res2.body);
    assert_eq!(res2.json()["traces"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn search_traces_v2_get_repeated_experiment_ids_multiple_values() {
    let server = TestServer::start("search_v2_multi_exp").await;
    start_trace_v3(&server, "sv2m-1", EXP_ID, "OK").await;

    // Repeated query param: one real experiment id + one bogus one that
    // doesn't exist. The bogus id is silently dropped (matches V3's
    // `_filter_experiment_ids`), so only the real experiment's trace shows up.
    let res = get_q(
        &server,
        "/api/2.0",
        &format!("/mlflow/traces?experiment_ids={EXP_ID}&experiment_ids=99999"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["traces"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn search_traces_v2_max_results_violation() {
    let server = TestServer::start("search_v2_max").await;
    let res = get_q(
        &server,
        "/api/2.0",
        &format!("/mlflow/traces?experiment_ids={EXP_ID}&max_results=501"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Invalid value 501 for parameter 'max_results' supplied. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn search_traces_v2_requires_experiment_ids() {
    let server = TestServer::start("search_v2_no_exp").await;
    let res = get_q(&server, "/api/2.0", "/mlflow/traces").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'experiment_ids'. \
         See the API docs for more information about request parameters."
    );
}

// ---------------------------------------------------------------------------
// deleteTraces (V2) — same handler/proto shape as V3, separate URL prefix.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_traces_v2_by_ids() {
    let server = TestServer::start("delete_v2_ids").await;
    start_trace_v3(&server, "dv2-1", EXP_ID, "OK").await;
    start_trace_v3(&server, "dv2-2", EXP_ID, "OK").await;

    let body = json!({"experiment_id": EXP_ID, "request_ids": ["dv2-1", "dv2-2"]}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/traces/delete-traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["traces_deleted"], 2);

    // Confirm gone via V3 get too (shared underlying rows).
    let got = get_q(&server, "/api/3.0", "/mlflow/traces/dv2-1").await;
    assert_eq!(got.status, StatusCode::NOT_FOUND, "{}", got.body);
}

#[tokio::test]
async fn delete_traces_v2_by_timestamp() {
    let server = TestServer::start("delete_v2_ts").await;
    start_trace_v3(&server, "dtv2-1", EXP_ID, "OK").await;

    let body = json!({
        "experiment_id": EXP_ID,
        "max_timestamp_millis": 9999999999999i64,
        "max_traces": 10
    })
    .to_string();
    let res = post(&server, "/api/2.0", "/mlflow/traces/delete-traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["traces_deleted"], 1);
}

#[tokio::test]
async fn delete_traces_v2_rejects_both_modes() {
    let server = TestServer::start("delete_v2_both").await;
    let body = json!({
        "experiment_id": EXP_ID,
        "max_timestamp_millis": 1,
        "request_ids": ["x"]
    })
    .to_string();
    let res = post(&server, "/api/2.0", "/mlflow/traces/delete-traces", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

#[tokio::test]
async fn delete_traces_v2_requires_experiment_id() {
    let server = TestServer::start("delete_v2_no_exp").await;
    let res = post(&server, "/api/2.0", "/mlflow/traces/delete-traces", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'experiment_id'. \
         See the API docs for more information about request parameters."
    );
}

// ---------------------------------------------------------------------------
// setTraceTag / deleteTraceTag (V2)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn trace_tag_v2_set_and_delete_on_both_prefixes() {
    for prefix in V2_PREFIXES {
        let server = TestServer::start("tag_crud_v2").await;
        let trace_id = "tagv2-1";
        start_trace_v3(&server, trace_id, EXP_ID, "OK").await;

        let set = send(
            &server,
            Method::PATCH,
            &format!("{prefix}/mlflow/traces/{trace_id}/tags"),
            Some(&json!({"key": "topic", "value": "billing"}).to_string()),
        )
        .await;
        assert_eq!(set.status, StatusCode::OK, "{}", set.body);

        // Verify via V2 getTraceInfo.
        let info = get_q(&server, prefix, &format!("/mlflow/traces/{trace_id}/info")).await;
        let tags = info.json()["trace_info"]["tags"]
            .as_array()
            .unwrap()
            .clone();
        assert!(tags
            .iter()
            .any(|t| t["key"] == "topic" && t["value"] == "billing"));

        let del = send(
            &server,
            Method::DELETE,
            &format!("{prefix}/mlflow/traces/{trace_id}/tags"),
            Some(&json!({"key": "topic"}).to_string()),
        )
        .await;
        assert_eq!(del.status, StatusCode::OK, "{}", del.body);

        // Deleting a missing tag is RESOURCE_DOES_NOT_EXIST.
        let del2 = send(
            &server,
            Method::DELETE,
            &format!("{prefix}/mlflow/traces/{trace_id}/tags"),
            Some(&json!({"key": "topic"}).to_string()),
        )
        .await;
        assert_eq!(del2.status, StatusCode::NOT_FOUND, "{}", del2.body);
        assert_eq!(del2.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
    }
}

#[tokio::test]
async fn trace_tag_v2_set_requires_key() {
    let server = TestServer::start("tag_v2_no_key").await;
    let trace_id = "tagv2-nokey";
    start_trace_v3(&server, trace_id, EXP_ID, "OK").await;

    let res = send(
        &server,
        Method::PATCH,
        &format!("/api/2.0/mlflow/traces/{trace_id}/tags"),
        Some(&json!({"value": "billing"}).to_string()),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'key'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn trace_tag_v2_set_visible_via_v3() {
    let server = TestServer::start("tag_v2_then_v3").await;
    let trace_id = "tagv2-cross";
    start_trace_v3(&server, trace_id, EXP_ID, "OK").await;

    let set = send(
        &server,
        Method::PATCH,
        &format!("/api/2.0/mlflow/traces/{trace_id}/tags"),
        Some(&json!({"key": "topic", "value": "billing"}).to_string()),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "{}", set.body);

    let got = get_q(&server, "/api/3.0", &format!("/mlflow/traces/{trace_id}")).await;
    assert_eq!(
        got.json()["trace"]["trace_info"]["tags"]["topic"],
        "billing"
    );
}

// ---------------------------------------------------------------------------
// UI "contains traces" ajax call shape (plan §3.7, T3.5 GET re-check).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ui_contains_traces_ajax_get_shape() {
    let server = TestServer::start("contains_traces").await;
    start_trace_v3(&server, "ct-1", EXP_ID, "OK").await;

    // Exact shape MlflowService.ts's `getExperimentTraces` builds via
    // `qs.stringify({ arrayFormat: 'repeat' })`: `order_by`/`page_token`/
    // `filter` are omitted entirely when unset (an empty `repeated` field or
    // `undefined` scalar serializes to nothing, not an empty-string param).
    let res = get_q(
        &server,
        "/ajax-api/2.0",
        &format!("/mlflow/traces?experiment_ids={EXP_ID}&max_results=1"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let traces = res.json()["traces"].as_array().unwrap().clone();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0]["request_id"], "ct-1");
}
