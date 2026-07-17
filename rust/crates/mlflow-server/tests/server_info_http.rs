//! HTTP integration tests for `GET /(api|ajax-api)/3.0/mlflow/server-info`
//! (plan T11.5, D5).
//!
//! Byte-checks the response shape (`_get_server_info`,
//! `mlflow/server/handlers.py:6586-6616`) across the deployment-flag matrix:
//! plain (both off), auth-enabled, workspaces-enabled, and both-enabled —
//! plus both URL prefixes and the exact `Content-Type`.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use serde_json::Value;
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";

fn tracking_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

fn auth_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("mlflow-auth")
        .join("tests")
        .join("fixtures")
        .join("basic_auth.db")
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str, source: &Path) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_server_info_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(source, &path).expect("copy fixture");
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
    _tracking_db: TempDb,
    _auth_db: Option<TempDb>,
}

impl TestServer {
    /// Boots the app with a tracking store always present (server-info is
    /// only reachable when `register_proto_routes` runs, i.e. `state: Some`),
    /// optionally wiring an `AuthStore` and/or `WorkspaceStore` to exercise
    /// the deployment-flag matrix.
    async fn start(tag: &str, with_auth: bool, with_workspaces: bool) -> Self {
        let tracking_db = TempDb::new(&format!("{tag}_track"), &tracking_fixture_path());
        let db = Db::connect(&tracking_db.uri(), PoolConfig::default())
            .await
            .expect("connect tracking fixture");
        let store = TrackingStore::new(db.clone(), ART_ROOT);

        let mut state = AppState::new(store);

        let auth_db_file = if with_auth {
            let auth_db_file = TempDb::new(&format!("{tag}_auth"), &auth_fixture_path());
            let auth_db =
                AuthDb::connect_and_verify_with(&auth_db_file.uri(), None, PoolConfig::default())
                    .await
                    .expect("connect + verify auth fixture");
            state = state.with_auth_store(AuthStore::new(auth_db));
            Some(auth_db_file)
        } else {
            None
        };

        if with_workspaces {
            state = state.with_workspace_store(WorkspaceStore::new(db.clone(), tracking_db.uri()));
        }

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
        let app = build_app_with_recorder(&config, recorder, Some(state));

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
            _tracking_db: tracking_db,
            _auth_db: auth_db_file,
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
    content_type: Option<String>,
    body: String,
    json: Value,
}

/// The auth fixture's admin (`auth_fixture.json`), for the auth-enabled cases:
/// like every route, `server-info` sits behind Python's `_before_request`
/// authentication gate when the basic-auth app is on (it has no *authorization*
/// validator, so any authenticated caller may read it).
const ADMIN: (&str, &str) = ("alice_scrypt", "alice-password-123");

async fn get(base: &str, path: &str) -> HttpResponse {
    get_with(base, path, None).await
}

async fn get_with(base: &str, path: &str, creds: Option<(&str, &str)>) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{base}{path}");
    let mut builder = Request::builder().method(Method::GET).uri(uri);
    if let Some((user, password)) = creds {
        let encoded = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            format!("{user}:{password}"),
        );
        builder = builder.header("Authorization", format!("Basic {encoded}"));
    }
    let req = builder.body(Full::new(Bytes::new())).unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(hyper::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    HttpResponse {
        status,
        content_type,
        body,
        json,
    }
}

const PATHS: [&str; 2] = [
    "/api/3.0/mlflow/server-info",
    "/ajax-api/3.0/mlflow/server-info",
];

fn assert_shape(resp: &HttpResponse, workspaces_enabled: bool) {
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert_eq!(resp.content_type.as_deref(), Some("application/json"));
    assert_eq!(resp.json["store_type"], "SqlStore");
    assert_eq!(resp.json["workspaces_enabled"], workspaces_enabled);
    assert_eq!(resp.json["trace_archival_enabled"], false);
    // Exactly three keys — no `auth_enabled`, no extras.
    let obj = resp.json.as_object().unwrap();
    assert_eq!(obj.len(), 3);
    assert!(obj.contains_key("store_type"));
    assert!(obj.contains_key("workspaces_enabled"));
    assert!(obj.contains_key("trace_archival_enabled"));
}

#[tokio::test]
async fn plain_deployment_both_flags_off() {
    let srv = TestServer::start("plain", false, false).await;
    for path in PATHS {
        let resp = get(&srv.base, path).await;
        assert_shape(&resp, false);
    }
}

#[tokio::test]
async fn auth_enabled_workspaces_disabled() {
    let srv = TestServer::start("auth_only", true, false).await;
    for path in PATHS {
        let resp = get_with(&srv.base, path, Some(ADMIN)).await;
        // `server-info` reports no `auth_enabled` field at all (Python parity):
        // enabling auth must not change the response shape or its other values.
        assert_shape(&resp, false);
    }
}

#[tokio::test]
async fn workspaces_enabled_auth_disabled() {
    let srv = TestServer::start("workspaces_only", false, true).await;
    for path in PATHS {
        let resp = get(&srv.base, path).await;
        assert_shape(&resp, true);
    }
}

#[tokio::test]
async fn auth_and_workspaces_both_enabled() {
    let srv = TestServer::start("both", true, true).await;
    for path in PATHS {
        let resp = get_with(&srv.base, path, Some(ADMIN)).await;
        assert_shape(&resp, true);
    }
}

#[tokio::test]
async fn server_info_requires_authentication_under_auth() {
    // Under the basic-auth app, `server-info` goes through `_before_request`
    // like every route: no credentials → the 401 Basic challenge. (It has no
    // authorization validator, so any authenticated user gets the payload —
    // covered above. The exemption at `auth/__init__.py:3638` is for
    // *after-request* handlers only; `workspace_helpers.py:103-105` carves it
    // out of the workspace-header gate, not out of authentication.)
    let srv = TestServer::start("no_creds", true, false).await;
    let resp = get(&srv.base, "/ajax-api/3.0/mlflow/server-info").await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED, "{}", resp.body);
    // Any authenticated (non-admin would also do) caller succeeds; the admin
    // fixture user keeps it simple.
    let resp = get_with(&srv.base, "/ajax-api/3.0/mlflow/server-info", Some(ADMIN)).await;
    assert_shape(&resp, false);
}

#[tokio::test]
async fn exact_byte_body_plain_deployment() {
    let srv = TestServer::start("byte_check", false, false).await;
    let resp = get(&srv.base, "/api/3.0/mlflow/server-info").await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(
        resp.body,
        r#"{"store_type":"SqlStore","workspaces_enabled":false,"trace_archival_enabled":false}"#
    );
}

#[tokio::test]
async fn only_get_is_registered() {
    let srv = TestServer::start("method_check", false, false).await;
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{}/api/3.0/mlflow/server-info", srv.base);
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    // axum returns 405 for a registered path hit with an unregistered method.
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}
