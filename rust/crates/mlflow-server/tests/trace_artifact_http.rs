//! HTTP integration tests for `GET /ajax-api/{2,3}.0/mlflow/get-trace-artifact`
//! (plan T4.5, §3.10). Boots the axum app on a real ephemeral socket (same
//! pattern as `traces_http.rs`) against a fresh copy of the committed
//! Alembic-migrated SQLite fixture, then drives:
//!
//! * TRACKING_STORE-backed spans JSON (body + headers).
//! * ARTIFACT_REPO-backed spans JSON (`traces.json` on local FS).
//! * attachment fetch via `path=` (a canonical-UUID attachment file).
//! * traversal `path=` → 400.
//! * missing `request_id` → 400.
//! * unknown trace → 404.
//! * ARCHIVE_REPO → `NOT_IMPLEMENTED`.
//! * both `/ajax-api/2.0` and `/ajax-api/3.0` prefixes serve the route.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, SpanInput, StartTraceInput, TraceTimeRange, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const EXP_ID: &str = "0";
/// The `TrackingStore`'s configured default artifact root. Per-trace artifact
/// locations (used by the artifact-repo tests) are overridden with a
/// per-test `TempDir` instead, so this value is never actually read from.
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
            "mlflow_rust_server_trace_artifact_{}_{}_{}.db",
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
            serve_artifacts: true,
            artifacts_destination: None,
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
    headers: hyper::HeaderMap,
    body: Vec<u8>,
}

impl HttpResponse {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| {
            panic!(
                "body is not JSON: {e}: {}",
                String::from_utf8_lossy(&self.body)
            )
        })
    }

    fn body_str(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn header(&self, name: &str) -> Option<String> {
        self.headers
            .get(name)
            .map(|v| v.to_str().unwrap().to_string())
    }
}

async fn get(server: &TestServer, prefix: &str, query: &str) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{prefix}/mlflow/get-trace-artifact?{query}", server.base);
    let request = Request::builder()
        .method(Method::GET)
        .uri(&url)
        .body(Full::<Bytes>::new(Bytes::new()))
        .unwrap();

    let mut last = None;
    for _ in 0..50 {
        match client.request(clone_request(&request)).await {
            Ok(res) => {
                let status = res.status();
                let headers = res.headers().clone();
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return HttpResponse {
                    status,
                    headers,
                    body: bytes.to_vec(),
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

/// Start a trace directly via the store (cheaper than round-tripping HTTP for
/// these tests, which only need a valid trace_id + experiment linkage).
async fn start_trace(store: &TrackingStore, trace_id: &str) {
    store
        .start_trace(
            "default",
            &StartTraceInput {
                trace_id: trace_id.to_string(),
                experiment_id: EXP_ID.to_string(),
                request_time: 1_700_000_000_000,
                execution_duration: Some(100),
                state: "OK".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: None,
                tags: Vec::new(),
                trace_metadata: Vec::new(),
                trace_metrics: Vec::new(),
            },
        )
        .await
        .expect("start_trace");
}

async fn insert_span(store: &TrackingStore, trace_id: &str, span_id: &str, content: &str) {
    let span = SpanInput {
        trace_id: trace_id.to_string(),
        span_id: span_id.to_string(),
        parent_span_id: None,
        name: Some("root-span".to_string()),
        span_type: Some("LLM".to_string()),
        status: "OK".to_string(),
        start_time_unix_nano: 1000,
        end_time_unix_nano: Some(2000),
        content: content.to_string(),
        dimension_attributes: None,
    };
    let range = TraceTimeRange {
        trace_id: trace_id.to_string(),
        min_start_ms: 0,
        max_end_ms: Some(0),
        root_span_status: Some("OK".to_string()),
    };
    store
        .log_spans("default", EXP_ID, &[span], &[], &[range])
        .await
        .expect("log_spans");
}

fn span_content(span_id_b64: &str) -> String {
    json!({
        "trace_id": "AAAAAAAAAAAAAAAAAAAAAA==",
        "span_id": span_id_b64,
        "parent_span_id": null,
        "name": "root-span",
        "start_time_unix_nano": 1000,
        "end_time_unix_nano": 2000,
        "events": [],
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {"mlflow.spanType": "\"LLM\""},
        "links": []
    })
    .to_string()
}

const PREFIXES: [&str; 2] = ["/ajax-api/2.0", "/ajax-api/3.0"];

// ---------------------------------------------------------------------------
// TRACKING_STORE: spans JSON built from the DB
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tracking_store_spans_json_on_both_ajax_prefixes() {
    let server = TestServer::start("tracking_store").await;

    for (i, prefix) in PREFIXES.iter().enumerate() {
        let trace_id = format!("tr-ts-{i}");
        start_trace(&server.store, &trace_id).await;
        server
            .store
            .set_trace_tag(
                "default",
                &trace_id,
                "mlflow.trace.spansLocation",
                "TRACKING_STORE",
            )
            .await
            .unwrap();
        insert_span(
            &server.store,
            &trace_id,
            "0000000000000001",
            &span_content("AAAAAAAAAAE="),
        )
        .await;

        let res = get(&server, prefix, &format!("request_id={trace_id}")).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body_str());
        assert_eq!(res.header("content-type").as_deref(), Some("text/plain"));
        assert_eq!(
            res.header("content-disposition").as_deref(),
            Some("attachment; filename=\"traces.json\"")
        );
        assert_eq!(
            res.header("x-content-type-options").as_deref(),
            Some("nosniff")
        );

        let body = res.json();
        let spans = body["spans"].as_array().expect("spans array");
        assert_eq!(spans.len(), 1, "{}", res.body_str());
        assert_eq!(spans[0]["name"], "root-span");
        // Default `json.dumps` separators: ", " / ": ", not the proto codec's
        // `indent=2` pretty-printer.
        assert!(!res.body_str().contains('\n'), "{}", res.body_str());
    }
}

#[tokio::test]
async fn tracking_store_partial_trace_returns_available_spans() {
    let server = TestServer::start("tracking_store_partial").await;
    let trace_id = "tr-partial";
    start_trace(&server.store, trace_id).await;
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.trace.spansLocation",
            "TRACKING_STORE",
        )
        .await
        .unwrap();
    // No spans logged yet — `allow_partial=True` still returns 200 with an
    // empty spans array (mirrors "allow partial so the frontend can render
    // in-progress traces").
    let res = get(&server, "/ajax-api/3.0", &format!("request_id={trace_id}")).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body_str());
    assert_eq!(res.json()["spans"].as_array().unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// ARTIFACT_REPO: traces.json read from the artifact store
// ---------------------------------------------------------------------------

#[tokio::test]
async fn artifact_repo_spans_json_from_local_fs() {
    let server = TestServer::start("artifact_repo").await;
    let trace_id = "tr-artifact-repo";
    start_trace(&server.store, trace_id).await;

    let dir = tempfile::TempDir::new().unwrap();
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.artifactLocation",
            &dir.path().display().to_string(),
        )
        .await
        .unwrap();
    // No spansLocation tag at all — Python's `_fetch_trace_data_from_store`
    // falls through to the artifact repo for anything that isn't
    // TRACKING_STORE/ARCHIVE_REPO, including an absent tag.
    let trace_data = json!({"spans": [{"name": "from-artifact-repo"}]});
    std::fs::write(
        dir.path().join("traces.json"),
        serde_json::to_vec(&trace_data).unwrap(),
    )
    .unwrap();

    let res = get(&server, "/ajax-api/2.0", &format!("request_id={trace_id}")).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body_str());
    assert_eq!(res.header("content-type").as_deref(), Some("text/plain"));
    assert_eq!(
        res.header("content-disposition").as_deref(),
        Some("attachment; filename=\"traces.json\"")
    );
    assert_eq!(
        res.json()["spans"][0]["name"],
        "from-artifact-repo",
        "{}",
        res.body_str()
    );
}

#[tokio::test]
async fn artifact_repo_explicit_tag_also_falls_back() {
    let server = TestServer::start("artifact_repo_tag").await;
    let trace_id = "tr-artifact-repo-tag";
    start_trace(&server.store, trace_id).await;

    let dir = tempfile::TempDir::new().unwrap();
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.artifactLocation",
            &dir.path().display().to_string(),
        )
        .await
        .unwrap();
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.trace.spansLocation",
            "ARTIFACT_REPO",
        )
        .await
        .unwrap();
    std::fs::write(
        dir.path().join("traces.json"),
        serde_json::to_vec(&json!({"spans": []})).unwrap(),
    )
    .unwrap();

    let res = get(&server, "/ajax-api/2.0", &format!("request_id={trace_id}")).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body_str());
    assert_eq!(res.json()["spans"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn artifact_repo_missing_traces_json_is_404() {
    let server = TestServer::start("artifact_repo_missing").await;
    let trace_id = "tr-artifact-repo-missing";
    start_trace(&server.store, trace_id).await;

    let dir = tempfile::TempDir::new().unwrap();
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.artifactLocation",
            &dir.path().display().to_string(),
        )
        .await
        .unwrap();

    let res = get(&server, "/ajax-api/2.0", &format!("request_id={trace_id}")).await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body_str());
    assert_eq!(res.json()["error_code"], "NOT_FOUND");
}

// ---------------------------------------------------------------------------
// Attachments via `path=`
// ---------------------------------------------------------------------------

#[tokio::test]
async fn attachment_fetch_via_path_uuid() {
    let server = TestServer::start("attachment").await;
    let trace_id = "tr-attachment";
    start_trace(&server.store, trace_id).await;

    let dir = tempfile::TempDir::new().unwrap();
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.artifactLocation",
            &dir.path().display().to_string(),
        )
        .await
        .unwrap();
    std::fs::create_dir_all(dir.path().join("attachments")).unwrap();
    let attachment_id = "550e8400-e29b-41d4-a716-446655440000";
    std::fs::write(
        dir.path().join("attachments").join(attachment_id),
        b"binary-attachment-bytes",
    )
    .unwrap();

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("request_id={trace_id}&path={attachment_id}"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body_str());
    assert_eq!(res.body, b"binary-attachment-bytes");
    assert_eq!(
        res.header("content-type").as_deref(),
        Some("application/octet-stream")
    );
    assert_eq!(
        res.header("content-disposition").as_deref(),
        Some(format!("attachment; filename=\"{attachment_id}\"").as_str())
    );
    assert_eq!(
        res.header("x-content-type-options").as_deref(),
        Some("nosniff")
    );
}

#[tokio::test]
async fn attachment_non_uuid_path_is_400() {
    let server = TestServer::start("attachment_bad_uuid").await;
    let trace_id = "tr-attachment-bad-uuid";
    start_trace(&server.store, trace_id).await;
    let dir = tempfile::TempDir::new().unwrap();
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.artifactLocation",
            &dir.path().display().to_string(),
        )
        .await
        .unwrap();

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("request_id={trace_id}&path=not-a-uuid"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body_str());
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

#[tokio::test]
async fn attachment_path_traversal_is_400_with_python_exact_body() {
    let server = TestServer::start("attachment_traversal").await;
    let trace_id = "tr-attachment-traversal";
    start_trace(&server.store, trace_id).await;

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("request_id={trace_id}&path=../../etc/passwd"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body_str());
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(res.json()["message"], "Invalid path");
}

// ---------------------------------------------------------------------------
// Required params / not-found / ARCHIVE_REPO
// ---------------------------------------------------------------------------

#[tokio::test]
async fn missing_request_id_is_400_with_python_exact_body() {
    let server = TestServer::start("missing_request_id").await;
    let res = get(&server, "/ajax-api/2.0", "").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body_str());
    assert_eq!(res.json()["error_code"], "BAD_REQUEST");
    assert_eq!(
        res.json()["message"],
        "Request must include the \"request_id\" query parameter."
    );
}

#[tokio::test]
async fn unknown_trace_is_404_with_python_exact_body() {
    let server = TestServer::start("unknown_trace").await;
    let res = get(&server, "/ajax-api/2.0", "request_id=tr-does-not-exist").await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body_str());
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
    assert_eq!(
        res.json()["message"],
        "Trace with ID 'tr-does-not-exist' not found."
    );
}

#[tokio::test]
async fn archive_repo_trace_is_not_implemented() {
    let server = TestServer::start("archive_repo").await;
    let trace_id = "tr-archive-repo";
    start_trace(&server.store, trace_id).await;
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.trace.spansLocation",
            "ARCHIVE_REPO",
        )
        .await
        .unwrap();

    let res = get(&server, "/ajax-api/2.0", &format!("request_id={trace_id}")).await;
    assert_eq!(
        res.status,
        StatusCode::NOT_IMPLEMENTED,
        "{}",
        res.body_str()
    );
    assert_eq!(res.json()["error_code"], "NOT_IMPLEMENTED");
}
