//! HTTP integration tests for `GET /model-versions/get-artifact` (plan T5.4,
//! §3.11): `get_model_version_artifact_handler` (`handlers.py:3033`).
//!
//! Boots the axum app on a real ephemeral socket (same harness as
//! `artifacts_http.rs`), seeding registry rows directly via
//! [`mlflow_registry::RegistryStore`] (sharing the same fixture DB as the
//! tracking store) and artifacts on the local filesystem, then drives:
//!
//! * direct-sourced version download (`storage_location` absent → falls back
//!   to `source`, a local `file://` URI);
//! * `models:/name/version`-sourced version download (the registry store
//!   resolves `storage_location` to the referenced version's download URI at
//!   creation time, so this exercises the same code path end-to-end);
//! * proxied `mlflow-artifacts://` locations resolving through
//!   `--artifacts-destination`;
//! * traversal → 400 exact body;
//! * missing `path`/`name`/`version` and unknown model/version errors.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::Value;
use tempfile::TempDir;
use tokio::net::TcpListener;

const WS: &str = "default";

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
            "mlflow_rust_server_mv_artifacts_{}_{}_{}.db",
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

/// A running test server. Owns the temp dirs so they outlive the app, and the
/// [`RegistryStore`] handle so tests can seed registered models / versions
/// directly (bypassing the not-yet-implemented registry HTTP endpoints).
struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    _db: TempDb,
    registry: RegistryStore,
    /// The `--artifacts-destination` root (proxy repo storage).
    dest_dir: TempDir,
    /// A local FS directory used as a direct (non-proxied) artifact source.
    direct_dir: TempDir,
}

impl TestServer {
    async fn start(tag: &str, serve_artifacts: bool) -> Self {
        let db_file = TempDb::new(tag);
        let dest_dir = TempDir::new().expect("dest dir");
        let direct_dir = TempDir::new().expect("direct dir");

        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        let store = TrackingStore::new(db.clone(), "file:///unused".to_string());
        let registry = RegistryStore::new(db);

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
            ..Default::default()
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app_state = AppState::with_registry(
            store,
            registry.clone(),
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
            registry,
            dest_dir,
            direct_dir,
        }
    }

    /// A `file://` URI for a subdirectory under the direct (non-proxied)
    /// source root.
    fn direct_source_uri(&self, rel: &str) -> String {
        format!("file://{}/{rel}", self.direct_dir.path().display())
    }

    /// Write a file under the direct source root at `<rel_dir>/<file>`.
    fn write_direct_file(&self, rel_dir: &str, file: &str, contents: &[u8]) {
        let dir = self.direct_dir.path().join(rel_dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), contents).unwrap();
    }

    /// Path to a file under the `--artifacts-destination` root, for direct
    /// filesystem setup in proxied-source tests.
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
    content_disposition: Option<String>,
}

impl HttpResponse {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    #[allow(dead_code)]
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON: {e}: {}", self.text()))
    }
}

async fn send_bytes(server: &TestServer, method: Method, path: &str) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{path}", server.base);
    let request = Request::builder()
        .method(method)
        .uri(&url)
        .body(Full::<Bytes>::new(Bytes::new()))
        .unwrap();

    let mut last = None;
    for _ in 0..50 {
        match client.request(clone_request(&request)).await {
            Ok(res) => {
                let status = res.status();
                let content_disposition = res
                    .headers()
                    .get("content-disposition")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return HttpResponse {
                    status,
                    body: bytes.to_vec(),
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
    send_bytes(server, Method::GET, path).await
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

/// A process-unique counter for hermetic model names.
fn uniq() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ===========================================================================
// Happy paths
// ===========================================================================

#[tokio::test]
async fn direct_sourced_version_downloads() {
    let server = TestServer::start("direct", true).await;
    let name = format!("model_{}", uniq());
    server
        .registry
        .create_registered_model(WS, &name, &[], None)
        .await
        .unwrap();

    server.write_direct_file("m", "MLmodel", b"flavors: {}");
    let source = server.direct_source_uri("m");
    server
        .registry
        .create_model_version(WS, &name, &source, None, &[], None, None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("/model-versions/get-artifact?name={name}&version=1&path=MLmodel"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.text(), "flavors: {}");
    assert_eq!(
        res.content_disposition.as_deref(),
        Some("attachment; filename=MLmodel")
    );
}

#[tokio::test]
async fn models_sourced_version_downloads_end_to_end() {
    let server = TestServer::start("models_source", true).await;
    let base_name = format!("base_{}", uniq());
    let alias_name = format!("alias_{}", uniq());

    server
        .registry
        .create_registered_model(WS, &base_name, &[], None)
        .await
        .unwrap();
    server
        .registry
        .create_registered_model(WS, &alias_name, &[], None)
        .await
        .unwrap();

    server.write_direct_file("base_m", "weights.bin", b"binary-weights");
    let base_source = server.direct_source_uri("base_m");
    server
        .registry
        .create_model_version(WS, &base_name, &base_source, None, &[], None, None)
        .await
        .unwrap();

    // The second registered model's version sources from the first via
    // `models:/<name>/<version>` — the registry store resolves this to the
    // base version's download URI (`storage_location`) at creation time.
    let models_source = format!("models:/{base_name}/1");
    server
        .registry
        .create_model_version(WS, &alias_name, &models_source, None, &[], None, None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("/model-versions/get-artifact?name={alias_name}&version=1&path=weights.bin"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.text(), "binary-weights");
}

#[tokio::test]
async fn proxied_mlflow_artifacts_source_resolves_through_destination() {
    let server = TestServer::start("proxied", true).await;
    let name = format!("proxied_{}", uniq());
    server
        .registry
        .create_registered_model(WS, &name, &[], None)
        .await
        .unwrap();

    // Write the file directly under the `--artifacts-destination` root, as if
    // it had been uploaded through the mlflow-artifacts proxy.
    let dest_path = server.dest_file("m/1/artifacts/model.pkl");
    std::fs::create_dir_all(dest_path.parent().unwrap()).unwrap();
    std::fs::write(&dest_path, b"pickled-model").unwrap();

    let source = "mlflow-artifacts://host/m/1/artifacts";
    server
        .registry
        .create_model_version(WS, &name, source, None, &[], None, None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("/model-versions/get-artifact?name={name}&version=1&path=model.pkl"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.text(), "pickled-model");
}

// ===========================================================================
// Errors
// ===========================================================================

#[tokio::test]
async fn traversal_path_is_400() {
    let server = TestServer::start("traversal", true).await;
    let name = format!("trav_{}", uniq());
    server
        .registry
        .create_registered_model(WS, &name, &[], None)
        .await
        .unwrap();
    let source = server.direct_source_uri("m");
    server
        .registry
        .create_model_version(WS, &name, &source, None, &[], None, None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("/model-versions/get-artifact?name={name}&version=1&path=..%2Fsecret"),
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
async fn missing_path_is_400() {
    let server = TestServer::start("missing_path", true).await;
    let res = get(
        &server,
        "/model-versions/get-artifact?name=whatever&version=1",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
}

#[tokio::test]
async fn missing_name_is_400() {
    let server = TestServer::start("missing_name", true).await;

    let res = get(&server, "/model-versions/get-artifact?version=1&path=f").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text()
            .contains("Missing value for required parameter 'name'"),
        "{}",
        res.text()
    );
}

/// Python's `_validate_model_version` (`mlflow/utils/validation.py:684`) has
/// no explicit `is None` check: it calls `int(model_version)` inside a
/// `try/except ValueError`. `int(None)` raises `TypeError`, NOT `ValueError`,
/// so the exception escapes uncaught, propagates out of the SQLAlchemy
/// store's `ManagedSessionMaker`, whose blanket `except Exception` handler
/// (`mlflow/store/db/utils.py:188`) wraps it into a 500 `INTERNAL_ERROR` —
/// NOT the 400 `INVALID_PARAMETER_VALUE` a present-but-non-numeric `version`
/// gets. Verified directly against the real Python handler.
#[tokio::test]
async fn missing_version_is_500_matching_python_typeerror_quirk() {
    let server = TestServer::start("missing_version", true).await;

    let res = get(&server, "/model-versions/get-artifact?name=m&path=f").await;
    assert_eq!(
        res.status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        res.text()
    );
    assert!(res.text().contains("INTERNAL_ERROR"), "{}", res.text());
}

#[tokio::test]
async fn non_numeric_version_is_400() {
    let server = TestServer::start("non_numeric_version", true).await;

    let res = get(
        &server,
        "/model-versions/get-artifact?name=m&version=abc&path=f",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text()
            .contains("Parameter 'version' must be an integer, got 'abc'."),
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn unknown_model_is_404() {
    let server = TestServer::start("unknown_model", true).await;
    let res = get(
        &server,
        "/model-versions/get-artifact?name=does-not-exist&version=1&path=f.txt",
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());
}

#[tokio::test]
async fn unknown_version_is_404() {
    let server = TestServer::start("unknown_version", true).await;
    let name = format!("nv_{}", uniq());
    server
        .registry
        .create_registered_model(WS, &name, &[], None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("/model-versions/get-artifact?name={name}&version=99&path=f.txt"),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());
    assert!(res.text().contains("Model Version"), "{}", res.text());
}
