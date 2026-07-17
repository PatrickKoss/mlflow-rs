//! HTTP integration tests for GraphQL per-field authorization (plan T9.6,
//! §3.16). Ports the behaviors of
//! `mlflow/server/auth/__init__.py::GraphQLAuthorizationMiddleware`:
//!
//! * admin sees everything (bypass);
//! * a non-admin denied on a protected field gets Python's in-band shape —
//!   the field is `null` in `data` with **no** `errors` entry;
//! * `mlflowSearchRuns` narrows `experimentIds` to the readable subset (mixed
//!   readable/unreadable ids), and denies (null) when none are readable;
//! * `mlflowSearchModelVersions` post-filters out unreadable model versions;
//! * unauthenticated → 401 (from the T9.4 layer, before GraphQL runs).
//!
//! The toggle-off case (`MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH=false`) lives in its
//! own binary (`graphql_auth_toggle_off_http.rs`) because the toggle is a
//! process-global env var and would race the checks-on tests here.
//!
//! Every test in this binary builds its `AuthStore` with `AuthConfig
//! { default_permission: NO_PERMISSIONS, .. }` (T9.8; same pattern as
//! `auth_middleware_no_default_http.rs`): the packaged default is `READ`, which
//! would grant every non-admin read access to everything and make denial
//! impossible. With `NO_PERMISSIONS` the floor, a non-admin sees only what an
//! explicit grant allows — exactly the matrix the middleware gates.

use std::path::{Path, PathBuf};

use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const WS: &str = "default";
const ALICE: (&str, &str) = ("alice_scrypt", "alice-password-123");
// The fixture's `bob_pbkdf2` carries an `experiment/*/EDIT` role grant, so he is
// *not* a clean non-admin. Denial tests use freshly created users (via
// `create_user`) with no grants and the NO_PERMISSIONS default floor.

/// T9.8: the permission floor comes from the parsed [`AuthConfig`] carried by
/// the `AuthStore` (the env-var seam is retired), so each test server builds
/// its store with `default_permission: NO_PERMISSIONS` directly.
fn no_permission_config() -> mlflow_auth::AuthConfig {
    mlflow_auth::AuthConfig {
        default_permission: "NO_PERMISSIONS".to_string(),
        ..mlflow_auth::AuthConfig::default()
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
            "mlflow_rust_gqlauth_{}_{}_{}.db",
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
    registry: RegistryStore,
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
        let tracking = TrackingStore::new(db.clone(), ART_ROOT);
        let registry = RegistryStore::new(db);

        let auth_db_file = TempDb::new(&format!("{tag}_auth"), &auth_fixture_path());
        let auth_db =
            AuthDb::connect_and_verify_with(&auth_db_file.uri(), None, PoolConfig::default())
                .await
                .expect("connect + verify auth fixture");
        let auth = AuthStore::with_config(auth_db, no_permission_config());

        let state = AppState::with_registry(tracking.clone(), registry.clone(), true, None, None)
            .with_auth_store(auth.clone());

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
            registry,
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

    async fn create_model_version(&self, model_name: &str) {
        self.registry
            .create_registered_model(WS, model_name, &[], None)
            .await
            .expect("create registered model");
        self.registry
            .create_model_version(
                WS,
                model_name,
                "mlflow-artifacts:/m/1",
                None,
                &[],
                None,
                None,
            )
            .await
            .expect("create model version");
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

fn basic_header(user: &str, pass: &str) -> String {
    let raw = format!("{user}:{pass}");
    let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
    format!("Basic {encoded}")
}

/// POST a `{query, variables}` GraphQL request. `auth` = optional Basic creds.
async fn gql(
    server: &TestServer,
    auth: Option<(&str, &str)>,
    query: &str,
    variables: Value,
) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{}/graphql", server.base);
    let body = json!({"query": query, "variables": variables});
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some((u, p)) = auth {
        builder = builder.header("Authorization", basic_header(u, p));
    }
    let req = builder
        .body(Full::new(Bytes::from(serde_json::to_vec(&body).unwrap())))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    HttpResponse {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
    }
}

// ===========================================================================
// Unauthenticated → 401 (T9.4 layer, before GraphQL executes)
// ===========================================================================

#[tokio::test]
async fn unauthenticated_graphql_is_401() {
    let srv = TestServer::start("unauth").await;
    let exp = srv.create_experiment("gql-unauth").await;
    let res = gql(
        &srv,
        None,
        "query Q($input: MlflowGetExperimentInput!) { \
         mlflowGetExperiment(input: $input) { experiment { experimentId } } }",
        json!({"input": {"experimentId": exp}}),
    )
    .await;
    assert_eq!(res.status, StatusCode::UNAUTHORIZED, "{}", res.body);
    assert!(res.body.starts_with("You are not authenticated."));
}

// ===========================================================================
// Admin bypass — sees everything
// ===========================================================================

#[tokio::test]
async fn admin_reads_experiment_without_grant() {
    let srv = TestServer::start("admin_exp").await;
    let exp = srv.create_experiment("gql-admin-exp").await;
    let res = gql(
        &srv,
        Some(ALICE),
        "query Q($input: MlflowGetExperimentInput!) { \
         mlflowGetExperiment(input: $input) { experiment { experimentId } apiError } }",
        json!({"input": {"experimentId": exp}}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let body = res.json();
    assert_eq!(
        body["data"]["mlflowGetExperiment"]["experiment"]["experimentId"],
        exp
    );
    assert!(body["errors"].is_null());
}

// ===========================================================================
// Non-admin denied field → null in `data`, no `errors` entry
// ===========================================================================

#[tokio::test]
async fn non_admin_denied_experiment_is_null_no_error() {
    let srv = TestServer::start("deny_exp").await;
    let exp = srv.create_experiment("gql-deny-exp").await;
    let (u, pw) = srv.create_user("gql_deny_exp").await;
    // Fresh user has no grant; default is NO_PERMISSIONS → cannot read.
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "query Q($input: MlflowGetExperimentInput!) { \
         mlflowGetExperiment(input: $input) { experiment { experimentId } apiError } }",
        json!({"input": {"experimentId": exp}}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let body = res.json();
    // Denied field → `null` (Python `resolve` returns None), and NO error string.
    assert!(
        body["data"]["mlflowGetExperiment"].is_null(),
        "expected null, got {}",
        body["data"]
    );
    assert!(
        body["errors"].is_null(),
        "no error expected: {}",
        body["errors"]
    );
}

#[tokio::test]
async fn non_admin_with_read_grant_sees_experiment() {
    let srv = TestServer::start("allow_exp").await;
    let exp = srv.create_experiment("gql-allow-exp").await;
    let (u, pw) = srv.create_user("gql_allow_exp").await;
    srv.grant(&u, "experiment", &exp, "READ").await;
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "query Q($input: MlflowGetExperimentInput!) { \
         mlflowGetExperiment(input: $input) { experiment { experimentId } apiError } }",
        json!({"input": {"experimentId": exp}}),
    )
    .await;
    let body = res.json();
    assert_eq!(
        body["data"]["mlflowGetExperiment"]["experiment"]["experimentId"],
        exp
    );
    assert!(body["errors"].is_null());
}

#[tokio::test]
async fn non_admin_denied_run_inherits_experiment_deny() {
    let srv = TestServer::start("deny_run").await;
    let exp = srv.create_experiment("gql-deny-run").await;
    let run = srv.create_run(&exp).await;
    let (u, pw) = srv.create_user("gql_deny_run").await;
    // No grant on the parent experiment → run READ denied.
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "query Q($input: MlflowGetRunInput!) { \
         mlflowGetRun(input: $input) { run { info { runId } } } }",
        json!({"input": {"runId": run}}),
    )
    .await;
    let body = res.json();
    assert!(body["data"]["mlflowGetRun"].is_null());
    assert!(body["errors"].is_null());

    // Granting READ on the parent experiment unblocks the run.
    srv.grant(&u, "experiment", &exp, "READ").await;
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "query Q($input: MlflowGetRunInput!) { \
         mlflowGetRun(input: $input) { run { info { runId experimentId } } } }",
        json!({"input": {"runId": run}}),
    )
    .await;
    let body = res.json();
    assert_eq!(body["data"]["mlflowGetRun"]["run"]["info"]["runId"], run);
}

// ===========================================================================
// searchRuns experiment-id narrowing
// ===========================================================================

#[tokio::test]
async fn search_runs_narrows_to_readable_experiments() {
    let srv = TestServer::start("search_narrow").await;
    let readable = srv.create_experiment("gql-readable").await;
    let unreadable = srv.create_experiment("gql-unreadable").await;
    let run_r = srv.create_run(&readable).await;
    let _run_u = srv.create_run(&unreadable).await;
    let (u, pw) = srv.create_user("gql_narrow").await;
    // The user can read only `readable`.
    srv.grant(&u, "experiment", &readable, "READ").await;

    // Mixed ids → narrowed to the readable one; only its run comes back.
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "mutation M($input: MlflowSearchRunsInput!) { \
         mlflowSearchRuns(input: $input) { runs { info { runId experimentId } } } }",
        json!({"input": {"experimentIds": [readable, unreadable]}}),
    )
    .await;
    let body = res.json();
    assert!(body["errors"].is_null(), "{}", body["errors"]);
    let runs = body["data"]["mlflowSearchRuns"]["runs"]
        .as_array()
        .expect("runs array");
    assert_eq!(runs.len(), 1, "expected only the readable run: {runs:?}");
    assert_eq!(runs[0]["info"]["runId"], run_r);
    assert_eq!(runs[0]["info"]["experimentId"], readable);
}

#[tokio::test]
async fn search_runs_all_unreadable_is_denied_null() {
    let srv = TestServer::start("search_denyall").await;
    let e1 = srv.create_experiment("gql-e1").await;
    let e2 = srv.create_experiment("gql-e2").await;
    srv.create_run(&e1).await;
    srv.create_run(&e2).await;
    let (u, pw) = srv.create_user("gql_denyall").await;
    // No grant on either → none readable → the field is denied (null).
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "mutation M($input: MlflowSearchRunsInput!) { \
         mlflowSearchRuns(input: $input) { runs { info { runId } } } }",
        json!({"input": {"experimentIds": [e1, e2]}}),
    )
    .await;
    let body = res.json();
    assert!(body["data"]["mlflowSearchRuns"].is_null());
    assert!(body["errors"].is_null());
}

// ===========================================================================
// searchModelVersions post-filter
// ===========================================================================

#[tokio::test]
async fn search_model_versions_post_filters_unreadable() {
    let srv = TestServer::start("mv_filter").await;
    srv.create_model_version("gql_readable_model").await;
    srv.create_model_version("gql_hidden_model").await;
    let (u, pw) = srv.create_user("gql_mv_filter").await;
    // The user can read only one of the two models.
    srv.grant(&u, "registered_model", "gql_readable_model", "READ")
        .await;

    let res = gql(
        &srv,
        Some((&u, &pw)),
        "query Q($input: MlflowSearchModelVersionsInput!) { \
         mlflowSearchModelVersions(input: $input) { modelVersions { name version } } }",
        json!({"input": {}}),
    )
    .await;
    let body = res.json();
    assert!(body["errors"].is_null(), "{}", body["errors"]);
    let mvs = body["data"]["mlflowSearchModelVersions"]["modelVersions"]
        .as_array()
        .expect("modelVersions array");
    // Only the readable model's version survives the post-filter.
    assert_eq!(mvs.len(), 1, "expected 1 readable mv: {mvs:?}");
    assert_eq!(mvs[0]["name"], "gql_readable_model");
}

#[tokio::test]
async fn admin_search_model_versions_sees_all() {
    let srv = TestServer::start("mv_admin").await;
    srv.create_model_version("gql_admin_a").await;
    srv.create_model_version("gql_admin_b").await;
    let res = gql(
        &srv,
        Some(ALICE),
        "query Q($input: MlflowSearchModelVersionsInput!) { \
         mlflowSearchModelVersions(input: $input) { modelVersions { name } } }",
        json!({"input": {}}),
    )
    .await;
    let body = res.json();
    let mvs = body["data"]["mlflowSearchModelVersions"]["modelVersions"]
        .as_array()
        .expect("modelVersions array");
    assert_eq!(mvs.len(), 2, "admin sees every mv: {mvs:?}");
}

// ===========================================================================
// Unprotected field (`test`) is never gated
// ===========================================================================

#[tokio::test]
async fn unprotected_field_resolves_for_non_admin() {
    let srv = TestServer::start("unprotected").await;
    let (u, pw) = srv.create_user("gql_unprotected").await;
    let res = gql(
        &srv,
        Some((&u, &pw)),
        "query Q { test(inputString: \"hi\") { output } }",
        json!({}),
    )
    .await;
    let body = res.json();
    assert_eq!(body["data"]["test"]["output"], "hi");
    assert!(body["errors"].is_null());
}
