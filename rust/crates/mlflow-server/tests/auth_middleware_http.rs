//! HTTP integration tests for the tower auth middleware (plan T9.4, §3.16).
//!
//! Ports the *authorization behaviors* of `tests/server/auth/test_auth.py`
//! (permission matrix, admin bypass, 401/403 shapes, unprotected routes, OTLP
//! experiment-UPDATE from header, artifact-proxy experiment-id extraction, the
//! MV-create dual requirement, webhooks admin-only, unknown-traces fail-closed,
//! default_permission fallback). The lattice-only cases of
//! `tests/server/auth/test_permissions.py` are already covered by
//! `mlflow-auth`'s `permissions` unit tests, so they are not re-ported here.
//!
//! The Python suite boots a full server subprocess and drives a real
//! `MlflowClient`; here we boot the axum app on an ephemeral socket with an
//! `AuthStore` (a copy of the committed fixture) + a `TrackingStore` (a copy of
//! the tracking fixture), seed users and per-user grants directly through the
//! store, create the experiments/runs/logged-models/traces the matrix needs,
//! then drive the gated endpoints over HTTP.
//!
//! Auth fixture users: `alice_scrypt` (admin) and `bob_pbkdf2` (non-admin, with
//! an `editors` role granting `experiment/*/EDIT`). We create additional users
//! and grants per test.

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
use mlflow_store::{Db, PoolConfig, StartTraceInput, TrackingStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const WS: &str = "default";
const ALICE: (&str, &str) = ("alice_scrypt", "alice-password-123");
const BOB: (&str, &str) = ("bob_pbkdf2", "bob-password-4567");

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
            "mlflow_rust_authmw_{}_{}_{}.db",
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

    /// Create a fresh non-admin user, returning `(username, password)`.
    async fn create_user(&self, username: &str) -> (String, String) {
        let password = format!("{username}-password-1");
        self.auth
            .create_user(username, &password, false)
            .await
            .expect("create user");
        (username.to_string(), password)
    }

    /// Grant a per-user permission on a resource, mirroring
    /// `grant_role_permission` in the Python test utils.
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

    async fn create_logged_model(&self, experiment_id: &str) -> String {
        self.tracking
            .create_logged_model(WS, experiment_id, Some("m"), None, &[], &[], None)
            .await
            .expect("create logged model")
            .model_id
    }

    async fn create_trace(&self, experiment_id: &str, trace_id: &str) {
        self.tracking
            .start_trace(
                WS,
                &StartTraceInput {
                    trace_id: trace_id.to_string(),
                    experiment_id: experiment_id.to_string(),
                    request_time: 1,
                    execution_duration: None,
                    state: "OK".to_string(),
                    client_request_id: None,
                    request_preview: None,
                    response_preview: None,
                    tags: vec![],
                    trace_metadata: vec![],
                    trace_metrics: vec![],
                    assessments: vec![],
                },
            )
            .await
            .expect("start trace");
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
    headers: hyper::HeaderMap,
}

fn basic_header(user: &str, pass: &str) -> String {
    let raw = format!("{user}:{pass}");
    let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
    format!("Basic {encoded}")
}

async fn send(
    base: &str,
    method: Method,
    path: &str,
    auth: Option<(&str, &str)>,
    body: Option<Value>,
) -> HttpResponse {
    send_with_headers(base, method, path, auth, body, &[]).await
}

async fn send_with_headers(
    base: &str,
    method: Method,
    path: &str,
    auth: Option<(&str, &str)>,
    body: Option<Value>,
    extra_headers: &[(&str, &str)],
) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{base}{path}");
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some((u, p)) = auth {
        builder = builder.header("Authorization", basic_header(u, p));
    }
    for (k, v) in extra_headers {
        builder = builder.header(*k, *v);
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
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    HttpResponse {
        status,
        body,
        headers,
    }
}

// ---------------------------------------------------------------------------
// 401 authentication (no / bad creds + challenge header)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_credentials_is_401_with_challenge() {
    let srv = TestServer::start("no_creds").await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/search",
        None,
        Some(json!({"max_results": 100})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED);
    assert!(resp.body.starts_with("You are not authenticated."));
    assert_eq!(
        resp.headers.get("WWW-Authenticate").unwrap(),
        "Basic realm=\"mlflow\""
    );
}

#[tokio::test]
async fn bad_password_is_401_with_challenge() {
    let srv = TestServer::start("bad_pw").await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/experiments/get?experiment_id=0",
        Some((BOB.0, "wrong-password")),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        resp.headers.get("WWW-Authenticate").unwrap(),
        "Basic realm=\"mlflow\""
    );
}

// ---------------------------------------------------------------------------
// Admin bypass
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_bypasses_all_validators() {
    let srv = TestServer::start("admin_bypass").await;
    let exp = srv.create_experiment("admin-bypass-exp").await;
    // Admin (alice) can read/update/delete without any grant.
    for (method, path) in [
        (
            Method::GET,
            format!("/api/2.0/mlflow/experiments/get?experiment_id={exp}"),
        ),
        (
            Method::POST,
            "/api/2.0/mlflow/experiments/update".to_string(),
        ),
    ] {
        let body = if method == Method::POST {
            Some(json!({"experiment_id": exp, "new_name": "renamed"}))
        } else {
            None
        };
        let resp = send(&srv.base, method, &path, Some(ALICE), body).await;
        assert_ne!(
            resp.status,
            StatusCode::FORBIDDEN,
            "admin denied: {}",
            resp.body
        );
        assert_ne!(resp.status, StatusCode::UNAUTHORIZED);
    }
}

// ---------------------------------------------------------------------------
// Experiment CRUD permission levels
// ---------------------------------------------------------------------------

#[tokio::test]
async fn experiment_read_requires_read_grant() {
    let srv = TestServer::start("exp_read").await;
    let exp = srv.create_experiment("exp-read").await;
    let (carol, carol_pw) = srv.create_user("carol_read").await;

    // No grant + default_permission READ → can read (default is READ).
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/experiments/get?experiment_id={exp}"),
        Some((&carol, &carol_pw)),
        None,
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN);

    // Update requires can_update — default READ does NOT grant it → 403.
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/update",
        Some((&carol, &carol_pw)),
        Some(json!({"experiment_id": exp, "new_name": "x"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(resp.body, "Permission denied");
}

#[tokio::test]
async fn experiment_update_and_delete_levels() {
    let srv = TestServer::start("exp_levels").await;
    let exp = srv.create_experiment("exp-levels").await;
    let (u, pw) = srv.create_user("dave_levels").await;

    // EDIT → can_update true, can_delete false.
    srv.grant(&u, "experiment", &exp, "EDIT").await;
    let update = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/set-experiment-tag",
        Some((&u, &pw)),
        Some(json!({"experiment_id": exp, "key": "k", "value": "v"})),
    )
    .await;
    assert_ne!(update.status, StatusCode::FORBIDDEN, "{}", update.body);
    let delete = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/delete",
        Some((&u, &pw)),
        Some(json!({"experiment_id": exp})),
    )
    .await;
    assert_eq!(delete.status, StatusCode::FORBIDDEN);

    // MANAGE → can_delete true.
    srv.grant(&u, "experiment", &exp, "MANAGE").await;
    let delete2 = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/delete",
        Some((&u, &pw)),
        Some(json!({"experiment_id": exp})),
    )
    .await;
    assert_ne!(delete2.status, StatusCode::FORBIDDEN, "{}", delete2.body);
}

// ---------------------------------------------------------------------------
// Runs inherit experiment permission
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_inherits_experiment_permission() {
    let srv = TestServer::start("run_inherit").await;
    let exp = srv.create_experiment("run-inherit-exp").await;
    let run = srv.create_run(&exp).await;
    let (u, pw) = srv.create_user("erin_run").await;

    // No grant, default READ → get-run allowed; log-metric (update) denied.
    let get = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/runs/get?run_id={run}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(get.status, StatusCode::FORBIDDEN, "{}", get.body);

    let log = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/runs/log-metric",
        Some((&u, &pw)),
        Some(json!({"run_id": run, "key": "m", "value": 1.0, "timestamp": 1, "step": 0})),
    )
    .await;
    assert_eq!(log.status, StatusCode::FORBIDDEN);

    // Grant EDIT on the parent experiment → log-metric now allowed.
    srv.grant(&u, "experiment", &exp, "EDIT").await;
    let log2 = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/runs/log-metric",
        Some((&u, &pw)),
        Some(json!({"run_id": run, "key": "m", "value": 1.0, "timestamp": 1, "step": 0})),
    )
    .await;
    assert_ne!(log2.status, StatusCode::FORBIDDEN, "{}", log2.body);
}

// ---------------------------------------------------------------------------
// Logged models inherit experiment permission
// ---------------------------------------------------------------------------

#[tokio::test]
async fn logged_model_inherits_experiment_permission() {
    let srv = TestServer::start("lm_inherit").await;
    let exp = srv.create_experiment("lm-inherit-exp").await;
    let model = srv.create_logged_model(&exp).await;
    let (u, pw) = srv.create_user("frank_lm").await;

    // Default READ → get allowed; delete (needs can_delete) denied.
    let get = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/logged-models/{model}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(get.status, StatusCode::FORBIDDEN, "{}", get.body);

    let del = send(
        &srv.base,
        Method::DELETE,
        &format!("/api/2.0/mlflow/logged-models/{model}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_eq!(del.status, StatusCode::FORBIDDEN);

    srv.grant(&u, "experiment", &exp, "MANAGE").await;
    let del2 = send(
        &srv.base,
        Method::DELETE,
        &format!("/api/2.0/mlflow/logged-models/{model}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(del2.status, StatusCode::FORBIDDEN, "{}", del2.body);
}

// ---------------------------------------------------------------------------
// MV create dual requirement: model UPDATE + source READ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_version_create_requires_model_update() {
    // The MV-create dual requirement's model-UPDATE half: without an UPDATE-level
    // grant on the registered model, create is denied *before* the source-read
    // check. This half is deterministic under the default `READ` permission
    // (READ lacks `can_update`). The source-READ-deny half requires a
    // `NO_PERMISSIONS` default (Python's `no_permission_auth.ini`); with the env
    // default of READ every experiment reads succeed, so that half is exercised
    // by the dedicated `model_version_create_source_read_denied_with_no_default`
    // test which flips the default.
    let srv = TestServer::start("mv_create").await;
    let exp = srv.create_experiment("mv-src-exp").await;
    let run = srv.create_run(&exp).await;
    let (u, pw) = srv.create_user("grace_mv").await;

    // Default READ on the registered model → `can_update` false → 403.
    let denied = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/model-versions/create",
        Some((&u, &pw)),
        Some(json!({"name": "mv-model", "source": "s3://x", "run_id": run})),
    )
    .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
    assert_eq!(denied.body, "Permission denied");

    // Grant MANAGE on the model + READ on the source experiment → the gate
    // passes (both dual halves satisfied); it may 400 downstream on the invalid
    // source URI, but must not be 403.
    srv.grant(&u, "registered_model", "mv-model", "MANAGE")
        .await;
    srv.grant(&u, "experiment", &exp, "READ").await;
    let allowed = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/model-versions/create",
        Some((&u, &pw)),
        Some(json!({"name": "mv-model", "source": "s3://x", "run_id": run})),
    )
    .await;
    assert_ne!(allowed.status, StatusCode::FORBIDDEN, "{}", allowed.body);
}

#[tokio::test]
async fn model_version_create_empty_source_id_does_not_bypass() {
    let srv = TestServer::start("mv_empty").await;
    let (u, pw) = srv.create_user("heidi_mv").await;
    srv.grant(&u, "registered_model", "mv-model2", "MANAGE")
        .await;
    // Explicit empty run_id → denied (guard on presence, not truthiness).
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/model-versions/create",
        Some((&u, &pw)),
        Some(json!({"name": "mv-model2", "source": "s3://x", "run_id": ""})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
}

// ---------------------------------------------------------------------------
// Webhooks admin-only
// ---------------------------------------------------------------------------

#[tokio::test]
async fn webhooks_are_admin_only() {
    let srv = TestServer::start("webhooks").await;
    // Non-admin (bob) is forbidden on every webhook route.
    for (method, path, body) in [
        (
            Method::POST,
            "/api/2.0/mlflow/webhooks",
            Some(json!({"name": "w"})),
        ),
        (Method::GET, "/api/2.0/mlflow/webhooks", None),
        (Method::GET, "/api/2.0/mlflow/webhooks/wh-1", None),
        (
            Method::PATCH,
            "/api/2.0/mlflow/webhooks/wh-1",
            Some(json!({"name": "n"})),
        ),
        (
            Method::POST,
            "/api/2.0/mlflow/webhooks/wh-1/test",
            Some(json!({})),
        ),
        (Method::DELETE, "/api/2.0/mlflow/webhooks/wh-1", None),
    ] {
        let resp = send(&srv.base, method.clone(), path, Some(BOB), body).await;
        assert_eq!(
            resp.status,
            StatusCode::FORBIDDEN,
            "{method} {path}: {}",
            resp.body
        );
    }
    // Admin (alice) passes the gate (create returns non-403).
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/webhooks",
        Some(ALICE),
        Some(json!({
            "name": "admin-webhook",
            "url": "https://example.com/webhook",
            "events": [{"entity": "REGISTERED_MODEL", "action": "CREATED"}]
        })),
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN, "{}", resp.body);
    assert_ne!(resp.status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// OTLP: experiment UPDATE required, from X-Mlflow-Experiment-Id header
// ---------------------------------------------------------------------------

#[tokio::test]
async fn otlp_unauthenticated_is_401() {
    let srv = TestServer::start("otlp_401").await;
    let resp = send_with_headers(
        &srv.base,
        Method::POST,
        "/v1/traces",
        None,
        None,
        &[
            ("Content-Type", "application/x-protobuf"),
            ("X-Mlflow-Experiment-Id", "0"),
        ],
    )
    .await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn otlp_requires_experiment_update_from_header() {
    let srv = TestServer::start("otlp_perm").await;
    let exp = srv.create_experiment("otel-perm-exp").await;
    let (u, pw) = srv.create_user("ivan_otel").await;

    // READ only → cannot write traces (needs can_update) → 403.
    srv.grant(&u, "experiment", &exp, "READ").await;
    let denied = send_with_headers(
        &srv.base,
        Method::POST,
        "/v1/traces",
        Some((&u, &pw)),
        None,
        &[
            ("Content-Type", "application/x-protobuf"),
            ("X-Mlflow-Experiment-Id", &exp),
        ],
    )
    .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);

    // EDIT → can_update → passes the permission gate (may fail downstream on
    // the empty protobuf body, but not 403).
    srv.grant(&u, "experiment", &exp, "EDIT").await;
    let allowed = send_with_headers(
        &srv.base,
        Method::POST,
        "/v1/traces",
        Some((&u, &pw)),
        None,
        &[
            ("Content-Type", "application/x-protobuf"),
            ("X-Mlflow-Experiment-Id", &exp),
        ],
    )
    .await;
    assert_ne!(allowed.status, StatusCode::FORBIDDEN, "{}", allowed.body);
}

// ---------------------------------------------------------------------------
// Artifact-proxy path inspection + experiment-id extraction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn artifact_proxy_authorization_by_experiment() {
    let srv = TestServer::start("art_proxy").await;
    let exp = srv.create_experiment("art-proxy-exp").await;
    let (u, pw) = srv.create_user("judy_art").await;

    // PUT upload requires can_update on the experiment from the artifact path;
    // default READ is not enough → 403.
    let denied = send(
        &srv.base,
        Method::PUT,
        &format!("/ajax-api/2.0/mlflow-artifacts/artifacts/{exp}/test.txt"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);

    // Grant EDIT → upload passes the gate.
    srv.grant(&u, "experiment", &exp, "EDIT").await;
    let allowed = send(
        &srv.base,
        Method::PUT,
        &format!("/ajax-api/2.0/mlflow-artifacts/artifacts/{exp}/test.txt"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(allowed.status, StatusCode::FORBIDDEN, "{}", allowed.body);
}

#[tokio::test]
async fn artifact_proxy_list_uses_query_param_experiment() {
    // The bare list route reads the experiment id from `?path=<experiment_id>/...`
    // (Python's `view_args is None` List case), including the optional
    // `workspaces/<ws>/` prefix form. A user with MANAGE on the experiment lists;
    // the (default-READ-sensitive) deny path is exercised in the NO_PERMISSIONS
    // suite.
    let srv = TestServer::start("art_list").await;
    let exp = srv.create_experiment("art-list-exp").await;
    let (owner, owner_pw) = srv.create_user("owner_art").await;
    srv.grant(&owner, "experiment", &exp, "MANAGE").await;

    // Plain `<experiment_id>/...` tail.
    let owner_resp = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow-artifacts/artifacts?path={exp}/models/m-abc/artifacts"),
        Some((&owner, &owner_pw)),
        None,
    )
    .await;
    assert_ne!(
        owner_resp.status,
        StatusCode::FORBIDDEN,
        "{}",
        owner_resp.body
    );

    // `workspaces/<ws>/<experiment_id>/...` prefixed tail resolves the same id.
    let prefixed = send(
        &srv.base,
        Method::GET,
        &format!(
            "/api/2.0/mlflow-artifacts/artifacts?path=workspaces/{WS}/{exp}/models/m/artifacts"
        ),
        Some((&owner, &owner_pw)),
        None,
    )
    .await;
    assert_ne!(prefixed.status, StatusCode::FORBIDDEN, "{}", prefixed.body);
}

// ---------------------------------------------------------------------------
// Unknown /mlflow/traces/ subpath fails closed
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_traces_subpath_fails_closed() {
    let srv = TestServer::start("trace_failclosed").await;
    // A non-admin hitting an unknown `/mlflow/traces/<id>/...` subpath is denied
    // (fail-closed), even though the route doesn't exist as a handler.
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/traces/some-id/unknown-subpath",
        Some(BOB),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(resp.body, "Permission denied");
}

#[tokio::test]
async fn trace_permission_inherits_experiment() {
    let srv = TestServer::start("trace_inherit").await;
    let exp = srv.create_experiment("trace-exp").await;
    srv.create_trace(&exp, "tr-abc").await;
    let (u, pw) = srv.create_user("karl_trace").await;

    // Read trace by trace id (v3 path param) — default READ allows.
    let get = send(
        &srv.base,
        Method::GET,
        "/api/3.0/mlflow/traces/tr-abc",
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(get.status, StatusCode::FORBIDDEN, "{}", get.body);

    // Set a trace tag (v3, needs update) — default READ denies.
    let tag = send(
        &srv.base,
        Method::PATCH,
        "/api/3.0/mlflow/traces/tr-abc/tags",
        Some((&u, &pw)),
        Some(json!({"key": "k", "value": "v"})),
    )
    .await;
    assert_eq!(tag.status, StatusCode::FORBIDDEN);

    srv.grant(&u, "experiment", &exp, "EDIT").await;
    let tag2 = send(
        &srv.base,
        Method::PATCH,
        "/api/3.0/mlflow/traces/tr-abc/tags",
        Some((&u, &pw)),
        Some(json!({"key": "k", "value": "v"})),
    )
    .await;
    assert_ne!(tag2.status, StatusCode::FORBIDDEN, "{}", tag2.body);
}

// ---------------------------------------------------------------------------
// Unprotected routes reachable unauthenticated
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unprotected_routes_reachable_without_auth() {
    let srv = TestServer::start("unprotected").await;
    for path in ["/health", "/version"] {
        let resp = send(&srv.base, Method::GET, path, None, None).await;
        assert_eq!(resp.status, StatusCode::OK, "{path}: {}", resp.body);
    }
}

// ---------------------------------------------------------------------------
// User endpoints: read/update-password gated to self; create/delete/admin
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_user_is_self_only_for_non_admin() {
    let srv = TestServer::start("user_self").await;
    // Bob reading his own record: allowed.
    let own = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/users/get?username={}", BOB.0),
        Some(BOB),
        None,
    )
    .await;
    assert_ne!(own.status, StatusCode::FORBIDDEN, "{}", own.body);

    // Bob reading alice's record: forbidden (username_is_sender false).
    let other = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/users/get?username={}", ALICE.0),
        Some(BOB),
        None,
    )
    .await;
    assert_eq!(other.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn update_admin_and_delete_user_forbidden_for_non_admin() {
    let srv = TestServer::start("user_admin_only").await;
    let update_admin = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-admin",
        Some(BOB),
        Some(json!({"username": BOB.0, "is_admin": true})),
    )
    .await;
    assert_eq!(update_admin.status, StatusCode::FORBIDDEN);

    let delete = send(
        &srv.base,
        Method::DELETE,
        "/api/2.0/mlflow/users/delete",
        Some(BOB),
        Some(json!({"username": BOB.0})),
    )
    .await;
    assert_eq!(delete.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn create_user_forbidden_for_non_workspace_admin() {
    let srv = TestServer::start("create_forbidden").await;
    // Bob is not a workspace admin (no `(workspace,*,MANAGE)` grant) → 403.
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(BOB),
        Some(json!({"username": "newuser", "password": "newuser-pass-12"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn current_user_allowed_for_any_authenticated() {
    let srv = TestServer::start("current_ok").await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        Some(BOB),
        None,
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN);
    assert_ne!(resp.status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// default_permission fallback
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_permission_read_allows_read_without_grant() {
    let srv = TestServer::start("default_read").await;
    let exp = srv.create_experiment("default-read-exp").await;
    let (u, pw) = srv.create_user("liam_default").await;
    // No grant at all; default_permission=READ → get-experiment (read) allowed.
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/experiments/get?experiment_id={exp}"),
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_ne!(resp.status, StatusCode::FORBIDDEN, "{}", resp.body);
}

// ---------------------------------------------------------------------------
// ajax prefix parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ajax_prefix_is_gated_too() {
    let srv = TestServer::start("ajax_gate").await;
    let exp = srv.create_experiment("ajax-exp").await;
    let (u, pw) = srv.create_user("mia_ajax").await;
    // Update on the ajax prefix is gated identically → default READ denies.
    let resp = send(
        &srv.base,
        Method::POST,
        "/ajax-api/2.0/mlflow/experiments/update",
        Some((&u, &pw)),
        Some(json!({"experiment_id": exp, "new_name": "x"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
}
