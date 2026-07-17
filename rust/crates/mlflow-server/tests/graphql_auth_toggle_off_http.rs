//! GraphQL auth toggle-OFF test (plan T9.6). With
//! `MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH=false`, per-field authorization is
//! disabled entirely — `get_graphql_authorization_middleware` returns `[]` — so
//! any *authenticated* user resolves every field, exactly like Python.
//!
//! This lives in its own binary (separate process) because the toggle is a
//! process-global env var; keeping it here means it can't race the checks-on
//! tests in `graphql_auth_http.rs`. We also set `MLFLOW_AUTH_DEFAULT_PERMISSION
//! = NO_PERMISSIONS` to prove the toggle short-circuits *before* any
//! permission lookup: even with the strictest default and no grant, the
//! non-admin still sees the experiment.

use std::path::{Path, PathBuf};

use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const WS: &str = "default";

fn set_env() {
    // Safe: every test in this binary sets the identical values.
    std::env::set_var("MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH", "false");
    std::env::set_var("MLFLOW_AUTH_DEFAULT_PERMISSION", "NO_PERMISSIONS");
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
            "mlflow_rust_gqlauth_off_{}_{}_{}.db",
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
        set_env();

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
        let auth = AuthStore::new(auth_db);

        let state = AppState::new(tracking.clone()).with_auth_store(auth.clone());

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

    async fn create_experiment(&self, name: &str) -> String {
        self.tracking
            .create_experiment(WS, name, None, &[])
            .await
            .expect("create experiment")
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

async fn gql(server: &TestServer, auth: (&str, &str), query: &str, variables: Value) -> String {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{}/graphql", server.base);
    let body = json!({"query": query, "variables": variables});
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("Authorization", basic_header(auth.0, auth.1))
        .body(Full::new(Bytes::from(serde_json::to_vec(&body).unwrap())))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[tokio::test]
async fn toggle_off_non_admin_sees_experiment_without_grant() {
    let srv = TestServer::start("toggle_off").await;
    let exp = srv.create_experiment("gql-toggle-off").await;
    let (u, pw) = srv.create_user("gql_toggle_off").await;
    // The user has no grant and the default is NO_PERMISSIONS, but the toggle is
    // off so no per-field authorization runs — the field resolves normally.
    let body: Value = serde_json::from_str(
        &gql(
            &srv,
            (&u, &pw),
            "query Q($input: MlflowGetExperimentInput!) { \
             mlflowGetExperiment(input: $input) { experiment { experimentId } apiError } }",
            json!({"input": {"experimentId": exp}}),
        )
        .await,
    )
    .unwrap();
    assert_eq!(
        body["data"]["mlflowGetExperiment"]["experiment"]["experimentId"],
        exp
    );
    assert!(body["errors"].is_null());
}
