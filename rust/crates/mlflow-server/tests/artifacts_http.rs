//! HTTP integration tests for the artifact plane (plan T5.1-T5.3, §3.11).
//!
//! Boots the axum app on a real ephemeral socket (same harness as the other
//! `*_http.rs` suites) with a **local** default-artifact root and a local
//! `--artifacts-destination`, then drives:
//!
//! * T5.1 `GET /get-artifact` (happy / traversal → 400 / missing-run / missing
//!   params);
//! * T5.2 the `mlflow-artifacts` proxy (list / upload / download / delete
//!   round-trip, multipart NOT_IMPLEMENTED parity, disabled-mode 503);
//! * T5.3 ajax `upload-artifact`, `listLoggedModelArtifacts`, and the logged-model
//!   artifact file download;
//! * a bounded-memory streaming download of a large file.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::Value;
use tempfile::TempDir;
use tokio::net::TcpListener;

/// Create a fresh experiment whose artifact root is under the server's local
/// default-artifact root (the store derives `file://<art_dir>/<exp_id>` when no
/// explicit location is given), and return its id.
async fn create_local_experiment(server: &TestServer, name: &str) -> String {
    let res = post(
        server,
        "/api/2.0/mlflow/experiments/create",
        &format!(r#"{{"name": "{name}"}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    res.json()["experiment_id"].as_str().unwrap().to_string()
}

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
            "mlflow_rust_server_artifacts_{}_{}_{}.db",
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

/// A running test server. Owns the temp artifact dirs so they outlive the app.
struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    _db: TempDb,
    /// The `--artifacts-destination` root (proxy repo storage).
    dest_dir: TempDir,
    /// The default experiment/run artifact root (local FS).
    _art_dir: TempDir,
}

impl TestServer {
    /// Start with `serve_artifacts` on/off. Both the default-artifact root and
    /// the `--artifacts-destination` are local temp dirs.
    async fn start(tag: &str, serve_artifacts: bool) -> Self {
        let db_file = TempDb::new(tag);
        let art_dir = TempDir::new().expect("art dir");
        let dest_dir = TempDir::new().expect("dest dir");
        let art_root = format!("file://{}", art_dir.path().display());

        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        let store = TrackingStore::new(db, art_root);

        let dest_uri = format!("file://{}", dest_dir.path().display());
        let proxy_repo = mlflow_artifacts::factory::repo_from_uri(&dest_uri).expect("proxy repo");

        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            static_prefix: None,
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts,
            artifacts_destination: Some(dest_uri),
            allowed_hosts: None,
            cors_allowed_origins: None,
            x_frame_options: "SAMEORIGIN".to_string(),
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app_state = AppState::with_artifacts(
            store,
            serve_artifacts,
            Some(proxy_repo),
            config.artifacts_destination.clone(),
        );
        let app = build_app_with_recorder(&config, recorder, Some(app_state));

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
            _db: db_file,
            dest_dir,
            _art_dir: art_dir,
        }
    }

    /// Path to a file under the `--artifacts-destination` root, for direct
    /// filesystem setup/inspection in proxy tests.
    fn dest_file(&self, rel: &str) -> PathBuf {
        self.dest_dir.path().join(rel)
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
    body: Vec<u8>,
    content_type: Option<String>,
    content_disposition: Option<String>,
}

impl HttpResponse {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON: {e}: {}", self.text()))
    }
}

async fn send_bytes(
    server: &TestServer,
    method: Method,
    path: &str,
    body: Option<Vec<u8>>,
) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{path}", server.base);
    let mut builder = Request::builder().method(method).uri(&url);
    let request = match body {
        Some(b) => {
            builder = builder.header("content-type", "application/json");
            builder.body(Full::<Bytes>::new(Bytes::from(b)))
        }
        None => builder.body(Full::<Bytes>::new(Bytes::new())),
    }
    .unwrap();

    let mut last = None;
    for _ in 0..50 {
        match client.request(clone_request(&request)).await {
            Ok(res) => {
                let status = res.status();
                let content_type = res
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let content_disposition = res
                    .headers()
                    .get("content-disposition")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return HttpResponse {
                    status,
                    body: bytes.to_vec(),
                    content_type,
                    content_disposition,
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

async fn get(server: &TestServer, path: &str) -> HttpResponse {
    send_bytes(server, Method::GET, path, None).await
}

async fn post(server: &TestServer, path: &str, body: &str) -> HttpResponse {
    send_bytes(server, Method::POST, path, Some(body.as_bytes().to_vec())).await
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

/// Create a run in a fresh local-FS experiment and return
/// `(run_id, artifact_uri)`.
async fn create_run(server: &TestServer) -> (String, String) {
    let exp_id = create_local_experiment(server, &format!("exp_{}", uniq())).await;
    let res = post(
        server,
        "/api/2.0/mlflow/runs/create",
        &format!(r#"{{"experiment_id": "{exp_id}", "start_time": 1}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let json = res.json();
    let info = &json["run"]["info"];
    (
        info["run_id"].as_str().unwrap().to_string(),
        info["artifact_uri"].as_str().unwrap().to_string(),
    )
}

/// A process-unique counter for hermetic experiment names.
fn uniq() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Strip a `file://` prefix to a local path.
fn local_path(uri: &str) -> PathBuf {
    PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri))
}

/// Write a file (creating parents) at `<artifact_uri>/<rel>`.
fn write_run_artifact(artifact_uri: &str, rel: &str, contents: &[u8]) {
    let full = local_path(artifact_uri).join(rel);
    std::fs::create_dir_all(full.parent().unwrap()).unwrap();
    std::fs::write(full, contents).unwrap();
}

// ===========================================================================
// T5.1 — /get-artifact
// ===========================================================================

#[tokio::test]
async fn get_artifact_streams_run_file() {
    let server = TestServer::start("get_happy", true).await;
    let (run_id, artifact_uri) = create_run(&server).await;
    write_run_artifact(&artifact_uri, "logs/output.txt", b"hello artifact");

    let res = get(
        &server,
        &format!("/get-artifact?run_id={run_id}&path=logs/output.txt"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.text(), "hello artifact");
    assert_eq!(res.content_type.as_deref(), Some("text/plain"));
    assert_eq!(
        res.content_disposition.as_deref(),
        Some("attachment; filename=\"output.txt\"")
    );
}

#[tokio::test]
async fn get_artifact_traversal_is_400() {
    let server = TestServer::start("get_traversal", true).await;
    let (run_id, _) = create_run(&server).await;
    let res = get(
        &server,
        &format!("/get-artifact?run_id={run_id}&path=..%2Fsecret"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text().contains("INVALID_PARAMETER_VALUE"),
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn get_artifact_missing_run_is_404() {
    let server = TestServer::start("get_missing_run", true).await;
    let res = get(&server, "/get-artifact?run_id=does-not-exist&path=f.txt").await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());
}

#[tokio::test]
async fn get_artifact_missing_params_is_400() {
    let server = TestServer::start("get_missing_params", true).await;
    // No path.
    let res = get(&server, "/get-artifact?run_id=abc").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    // No run_id.
    let res = get(&server, "/get-artifact?path=f.txt").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
}

// ===========================================================================
// T5.2 — mlflow-artifacts proxy
// ===========================================================================

#[tokio::test]
async fn proxy_upload_list_download_delete_roundtrip() {
    let server = TestServer::start("proxy_roundtrip", true).await;

    for prefix in ["/api/2.0", "/ajax-api/2.0"] {
        let key = prefix.trim_start_matches('/').replace('/', "_");
        let art = format!("exp/run/{key}.txt");

        // Upload (PUT streams into the dest repo).
        let res = send_bytes(
            &server,
            Method::PUT,
            &format!("{prefix}/mlflow-artifacts/artifacts/{art}"),
            Some(b"payload-bytes".to_vec()),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());
        assert_eq!(res.text(), "{}");
        assert!(server.dest_file(&art).exists());

        // List the parent dir (basenames).
        let res = get(
            &server,
            &format!("{prefix}/mlflow-artifacts/artifacts?path=exp/run"),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());
        let listing = res.text();
        assert!(listing.contains(&format!("\"{key}.txt\"")), "{listing}");

        // Download.
        let res = get(
            &server,
            &format!("{prefix}/mlflow-artifacts/artifacts/{art}"),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());
        assert_eq!(res.text(), "payload-bytes");
        assert_eq!(res.content_type.as_deref(), Some("text/plain"));

        // Delete.
        let res = send_bytes(
            &server,
            Method::DELETE,
            &format!("{prefix}/mlflow-artifacts/artifacts/{art}"),
            None,
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());

        // Gone.
        let res = get(
            &server,
            &format!("{prefix}/mlflow-artifacts/artifacts/{art}"),
        )
        .await;
        assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());
    }
}

#[tokio::test]
async fn proxy_multipart_create_is_not_implemented() {
    let server = TestServer::start("proxy_mpu", true).await;
    let res = post(
        &server,
        "/api/2.0/mlflow-artifacts/mpu/create/big.bin",
        r#"{"path": "big.bin", "num_parts": 3}"#,
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_IMPLEMENTED, "{}", res.text());
    assert!(res.text().contains("NOT_IMPLEMENTED"), "{}", res.text());
}

#[tokio::test]
async fn proxy_presigned_is_not_implemented() {
    let server = TestServer::start("proxy_presigned", true).await;
    let res = get(&server, "/api/2.0/mlflow-artifacts/presigned/x.bin").await;
    assert_eq!(res.status, StatusCode::NOT_IMPLEMENTED, "{}", res.text());
}

#[tokio::test]
async fn proxy_traversal_is_400() {
    let server = TestServer::start("proxy_traversal", true).await;
    let res = get(&server, "/api/2.0/mlflow-artifacts/artifacts/%2e%2e/secret").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(res.text().contains("Invalid path"), "{}", res.text());
}

#[tokio::test]
async fn proxy_disabled_returns_503() {
    let server = TestServer::start("proxy_disabled", false).await;
    // List.
    let res = get(&server, "/api/2.0/mlflow-artifacts/artifacts").await;
    assert_eq!(
        res.status,
        StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        res.text()
    );
    assert!(
        res.text().contains("--no-serve-artifacts"),
        "{}",
        res.text()
    );
    // Upload.
    let res = send_bytes(
        &server,
        Method::PUT,
        "/api/2.0/mlflow-artifacts/artifacts/x.txt",
        Some(b"x".to_vec()),
    )
    .await;
    assert_eq!(
        res.status,
        StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        res.text()
    );
}

// ===========================================================================
// T5.3 — ajax upload-artifact + logged-model artifacts
// ===========================================================================

#[tokio::test]
async fn ajax_upload_artifact_writes_to_run_root() {
    let server = TestServer::start("ajax_upload", true).await;
    let (run_id, artifact_uri) = create_run(&server).await;

    let res = send_bytes(
        &server,
        Method::POST,
        &format!("/ajax-api/2.0/mlflow/upload-artifact?run_uuid={run_id}&path=sub/data.bin"),
        Some(b"uploaded".to_vec()),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // The file landed at <run artifact root>/sub/data.bin.
    let written = std::fs::read(local_path(&artifact_uri).join("sub/data.bin")).unwrap();
    assert_eq!(written, b"uploaded");
}

#[tokio::test]
async fn ajax_upload_artifact_requires_run_uuid_and_path() {
    let server = TestServer::start("ajax_upload_validation", true).await;
    // Missing run_uuid.
    let res = send_bytes(
        &server,
        Method::POST,
        "/ajax-api/2.0/mlflow/upload-artifact?path=f",
        Some(b"x".to_vec()),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(res.text().contains("run_uuid"), "{}", res.text());

    // Missing data (empty body).
    let (run_id, _) = create_run(&server).await;
    let res = send_bytes(
        &server,
        Method::POST,
        &format!("/ajax-api/2.0/mlflow/upload-artifact?run_uuid={run_id}&path=f"),
        Some(Vec::new()),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(res.text().contains("data"), "{}", res.text());
}

#[tokio::test]
async fn list_logged_model_artifacts_and_download() {
    let server = TestServer::start("lm_artifacts", true).await;

    // Create a logged model (its artifact_location is under the local art root).
    let exp_id = create_local_experiment(&server, "lm_exp").await;
    let res = post(
        &server,
        "/api/2.0/mlflow/logged-models",
        &format!(r#"{{"experiment_id": "{exp_id}", "name": "m"}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let json = res.json();
    let model_id = json["model"]["info"]["model_id"]
        .as_str()
        .unwrap()
        .to_string();
    let art_loc = json["model"]["info"]["artifact_uri"]
        .as_str()
        .unwrap()
        .to_string();

    // Write two files into the model's artifact dir.
    write_run_artifact(&art_loc, "MLmodel", b"flavors: {}");
    write_run_artifact(&art_loc, "data/model.pkl", b"binary");

    // listLoggedModelArtifacts (proto route, GET).
    let res = get(
        &server,
        &format!("/api/2.0/mlflow/logged-models/{model_id}/artifacts/directories"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let listing = res.json();
    let files = listing["files"].as_array().unwrap();
    let names: Vec<&str> = files.iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert!(names.contains(&"MLmodel"), "{listing}");
    assert!(names.contains(&"data"), "{listing}");
    assert_eq!(listing["root_uri"].as_str().unwrap(), art_loc);

    // Download a logged-model artifact file (ajax-only).
    let res = get(
        &server,
        &format!("/ajax-api/2.0/mlflow/logged-models/{model_id}/artifacts/files?artifact_file_path=MLmodel"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.text(), "flavors: {}");
}

// ===========================================================================
// Bounded-memory streaming download of a large file
// ===========================================================================

#[cfg(target_os = "linux")]
fn rss_bytes() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
    let resident_pages: usize = statm.split_whitespace().nth(1).unwrap().parse().unwrap();
    resident_pages * 4096
}

#[tokio::test]
async fn proxy_download_large_file_is_memory_bounded() {
    let server = TestServer::start("proxy_large", true).await;

    // 256 MiB file written directly to the dest repo.
    const SIZE: usize = 256 * 1024 * 1024;
    let path = server.dest_file("big/blob.bin");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        let chunk = vec![7u8; 8 * 1024 * 1024];
        let mut written = 0;
        while written < SIZE {
            f.write_all(&chunk).unwrap();
            written += chunk.len();
        }
    }

    #[cfg(target_os = "linux")]
    let before = rss_bytes();

    // Stream the download, discarding chunks — never materialize the whole body.
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!(
        "{}/api/2.0/mlflow-artifacts/artifacts/big/blob.bin",
        server.base
    );
    let res = client
        .request(
            Request::builder()
                .method(Method::GET)
                .uri(&url)
                .body(Full::<Bytes>::new(Bytes::new()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let mut body = res.into_body();
    let mut total = 0usize;
    #[cfg(target_os = "linux")]
    let mut peak = before;
    while let Some(frame) = body.frame().await {
        let frame = frame.unwrap();
        if let Some(data) = frame.data_ref() {
            total += data.len();
        }
        #[cfg(target_os = "linux")]
        {
            peak = peak.max(rss_bytes());
        }
    }
    assert_eq!(total, SIZE);

    #[cfg(target_os = "linux")]
    {
        let growth = peak.saturating_sub(before);
        // A buffering implementation would grow by ~256 MiB; assert the peak
        // stays well under the payload size.
        assert!(
            growth < 128 * 1024 * 1024,
            "download RSS growth {growth} bytes exceeded 128 MiB bound"
        );
    }
}
