//! Auth-middleware HTTP tests that require `default_permission = NO_PERMISSIONS`
//! (plan T9.4). These mirror the Python cases that use the
//! `fixtures/no_permission_auth.ini` config: the MV-create source-READ deny and
//! the default-permission deny fallback.
//!
//! Since T9.8, `default_permission` is threaded through the parsed
//! [`mlflow_auth::AuthConfig`] carried by the [`AuthStore`], so these tests
//! build the store with `AuthConfig { default_permission: "NO_PERMISSIONS", .. }`
//! instead of the retired `MLFLOW_AUTH_DEFAULT_PERMISSION` env var. That makes
//! the config per-store rather than process-global, so no cross-test env race
//! remains.

use std::path::{Path, PathBuf};

use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthConfig, AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const WS: &str = "default";

fn no_permission_config() -> AuthConfig {
    AuthConfig {
        default_permission: "NO_PERMISSIONS".to_string(),
        ..AuthConfig::default()
    }
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

fn tracking_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str, source: &Path) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_authmw_nd_{}_{}_{}.db",
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
    tracking: TrackingStore,
    auth: AuthStore,
    _tracking_db: TempDb,
    _auth_db: TempDb,
}

impl TestServer {
    async fn start(tag: &str) -> Self {
        let tracking_db = TempDb::new(&format!("{tag}_track"), &tracking_fixture_path());
        let db = Db::connect(&tracking_db.uri(), PoolConfig::default())
            .await
            .expect("connect tracking fixture");
        let tracking = TrackingStore::new(db, ART_ROOT);

        let auth_db_file = TempDb::new(&format!("{tag}_auth"), &auth_fixture_path());
        let auth_db =
            AuthDb::connect_and_verify_with(&auth_db_file.uri(), None, PoolConfig::default())
                .await
                .expect("connect + verify auth fixture");
        let auth = AuthStore::with_config(auth_db, no_permission_config());

        let state = AppState::new(tracking.clone()).with_auth_store(auth.clone());
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
            tracking,
            auth,
            _tracking_db: tracking_db,
            _auth_db: auth_db_file,
        }
    }

    async fn create_user(&self, username: &str) -> (String, String) {
        let password = format!("{username}-password-1");
        self.auth
            .create_user(username, &password, false)
            .await
            .expect("create user");
        (username.to_string(), password)
    }

    async fn grant(
        &self,
        username: &str,
        resource_type: &str,
        resource_id: &str,
        permission: &str,
    ) {
        self.auth
            .grant_user_permission(username, resource_type, resource_id, permission, WS)
            .await
            .expect("grant");
    }

    async fn create_experiment(&self, name: &str) -> String {
        self.tracking
            .create_experiment(WS, name, None, &[])
            .await
            .expect("create experiment")
    }

    async fn create_run(&self, experiment_id: &str) -> String {
        self.tracking
            .create_run(WS, experiment_id, None, Some(1), Some("r"), &[])
            .await
            .expect("create run")
            .info
            .run_id
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

fn basic_header(user: &str, pass: &str) -> String {
    let raw = format!("{user}:{pass}");
    let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
    format!("Basic {encoded}")
}

struct HttpResponse {
    status: StatusCode,
    body: String,
}

async fn send(
    base: &str,
    method: Method,
    path: &str,
    auth: Option<(&str, &str)>,
    body: Option<Value>,
) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{base}{path}");
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some((u, p)) = auth {
        builder = builder.header("Authorization", basic_header(u, p));
    }
    let body_bytes = match body {
        Some(v) => {
            builder = builder.header("Content-Type", "application/json");
            Bytes::from(v.to_string())
        }
        None => Bytes::new(),
    };
    let req = builder.body(Full::new(body_bytes)).unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    HttpResponse { status, body }
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_no_permission_denies_read_without_grant() {
    // With `default_permission = NO_PERMISSIONS`, an ungranted user cannot even
    // read an experiment → 403 (the resolver folds `None` → NO_PERMISSIONS is
    // wrong; `None` folds to the *default*, which here is NO_PERMISSIONS).
    let srv = TestServer::start("nd_read").await;
    let exp = srv.create_experiment("nd-read-exp").await;
    let (u, pw) = srv.create_user("nate_nd").await;
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/experiments/get?experiment_id={exp}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN, "{}", resp.body);
    assert_eq!(resp.body, "Permission denied");

    // A READ grant lifts the deny.
    srv.grant(&u, "experiment", &exp, "READ").await;
    let ok = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/experiments/get?experiment_id={exp}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(ok.status, StatusCode::FORBIDDEN, "{}", ok.body);
}

#[tokio::test]
async fn model_version_create_source_read_denied_with_no_default() {
    // The MV-create dual requirement's source-READ half: user has MANAGE on the
    // target model but no READ on the source run's experiment (and the default
    // is NO_PERMISSIONS), so anchoring a model version at that run is denied.
    let srv = TestServer::start("nd_mv").await;
    let exp = srv.create_experiment("nd-mv-exp").await;
    let run = srv.create_run(&exp).await;
    let (u, pw) = srv.create_user("olga_nd").await;
    srv.grant(&u, "registered_model", "nd-model", "MANAGE")
        .await;

    let denied = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/model-versions/create",
        Some((&u, &pw)),
        Some(json!({"name": "nd-model", "source": "s3://x", "run_id": run})),
    )
    .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
    assert_eq!(denied.body, "Permission denied");

    // Grant READ on the source experiment → the source-read half now passes
    // (gate reaches the handler, which may 400 on the source URI but not 403).
    srv.grant(&u, "experiment", &exp, "READ").await;
    let allowed = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/model-versions/create",
        Some((&u, &pw)),
        Some(json!({"name": "nd-model", "source": "s3://x", "run_id": run})),
    )
    .await;
    assert_ne!(allowed.status, StatusCode::FORBIDDEN, "{}", allowed.body);
}

#[tokio::test]
async fn mcp_server_permissions_follow_read_edit_manage_lattice() {
    let srv = TestServer::start("nd_mcp_permissions").await;
    let name = "com.example/auth-existing";
    srv.tracking
        .create_mcp_server(WS, name, None, None, None)
        .await
        .unwrap();
    let (user, password) = srv.create_user("mcp_reader").await;
    let path = format!("/api/3.0/mlflow/mcp-servers/{name}");

    let denied = send(
        &srv.base,
        Method::GET,
        &path,
        Some((&user, &password)),
        None,
    )
    .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);

    srv.grant(&user, "mcp_server", name, "READ").await;
    let readable = send(
        &srv.base,
        Method::GET,
        &path,
        Some((&user, &password)),
        None,
    )
    .await;
    assert_eq!(readable.status, StatusCode::OK, "{}", readable.body);
    let not_editable = send(
        &srv.base,
        Method::PATCH,
        &path,
        Some((&user, &password)),
        Some(json!({"description": "denied"})),
    )
    .await;
    assert_eq!(
        not_editable.status,
        StatusCode::FORBIDDEN,
        "{}",
        not_editable.body
    );

    srv.grant(&user, "mcp_server", name, "EDIT").await;
    let editable = send(
        &srv.base,
        Method::PATCH,
        &path,
        Some((&user, &password)),
        Some(json!({"description": "allowed"})),
    )
    .await;
    assert_eq!(editable.status, StatusCode::OK, "{}", editable.body);
    let not_deletable = send(
        &srv.base,
        Method::DELETE,
        &path,
        Some((&user, &password)),
        None,
    )
    .await;
    assert_eq!(
        not_deletable.status,
        StatusCode::FORBIDDEN,
        "{}",
        not_deletable.body
    );

    srv.grant(&user, "mcp_server", name, "MANAGE").await;
    let deleted = send(
        &srv.base,
        Method::DELETE,
        &path,
        Some((&user, &password)),
        None,
    )
    .await;
    assert_eq!(deleted.status, StatusCode::OK, "{}", deleted.body);
}

#[tokio::test]
async fn mcp_server_creator_gets_manage_for_root_and_auto_created_parent() {
    let srv = TestServer::start("nd_mcp_creator").await;
    let (creator, password) = srv.create_user("mcp_creator").await;
    let (other, other_password) = srv.create_user("mcp_other").await;

    let created = send(
        &srv.base,
        Method::POST,
        "/api/3.0/mlflow/mcp-servers",
        Some((&creator, &password)),
        Some(json!({"name": "com.example/auth-created"})),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{}", created.body);
    let creator_read = send(
        &srv.base,
        Method::GET,
        "/api/3.0/mlflow/mcp-servers/com.example/auth-created",
        Some((&creator, &password)),
        None,
    )
    .await;
    assert_eq!(creator_read.status, StatusCode::OK, "{}", creator_read.body);
    let other_denied = send(
        &srv.base,
        Method::GET,
        "/api/3.0/mlflow/mcp-servers/com.example/auth-created",
        Some((&other, &other_password)),
        None,
    )
    .await;
    assert_eq!(
        other_denied.status,
        StatusCode::FORBIDDEN,
        "{}",
        other_denied.body
    );

    let deleted = send(
        &srv.base,
        Method::DELETE,
        "/api/3.0/mlflow/mcp-servers/com.example/auth-created",
        Some((&creator, &password)),
        None,
    )
    .await;
    assert_eq!(deleted.status, StatusCode::OK, "{}", deleted.body);
    srv.tracking
        .create_mcp_server(WS, "com.example/auth-created", None, None, None)
        .await
        .unwrap();
    let stale_grant_denied = send(
        &srv.base,
        Method::GET,
        "/api/3.0/mlflow/mcp-servers/com.example/auth-created",
        Some((&creator, &password)),
        None,
    )
    .await;
    assert_eq!(
        stale_grant_denied.status,
        StatusCode::FORBIDDEN,
        "{}",
        stale_grant_denied.body
    );

    let auto_created = send(
        &srv.base,
        Method::POST,
        "/api/3.0/mlflow/mcp-servers/com.example/auto-parent/versions",
        Some((&creator, &password)),
        Some(json!({
            "server_json": {"name": "com.example/auto-parent", "version": "1.0.0"}
        })),
    )
    .await;
    assert_eq!(auto_created.status, StatusCode::OK, "{}", auto_created.body);
    let auto_parent_read = send(
        &srv.base,
        Method::GET,
        "/api/3.0/mlflow/mcp-servers/com.example/auto-parent",
        Some((&creator, &password)),
        None,
    )
    .await;
    assert_eq!(
        auto_parent_read.status,
        StatusCode::OK,
        "{}",
        auto_parent_read.body
    );
}
