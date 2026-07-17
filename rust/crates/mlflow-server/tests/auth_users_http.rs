//! HTTP integration tests for the 8 user-management endpoints (plan T9.2,
//! §3.16).
//!
//! Boots the axum app on a real ephemeral socket with the basic-auth app
//! enabled (an [`AuthStore`] over a fresh copy of the committed
//! `mlflow-auth` `basic_auth.db` fixture), then drives every user endpoint over
//! HTTP: create/get/current/list/update-password/update-admin/delete, the
//! param-missing / duplicate-user / cannot-delete-self / password-rule error
//! paths, response-shape byte checks, and the auth-disabled case where the
//! routes are absent.
//!
//! The fixture seeds two users (`rust/crates/mlflow-auth/tests/fixtures`):
//! `alice_scrypt` (admin, `alice-password-123`) and `bob_pbkdf2` (non-admin,
//! `bob-password-4567`). We authenticate as these to exercise the handler-level
//! self checks; the authorization middleware (admin-only gating) is T9.4, so
//! these tests assert only the handler behaviors T9.2 owns.

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
const ALICE: (&str, &str) = ("alice_scrypt", "alice-password-123");
const BOB: (&str, &str) = ("bob_pbkdf2", "bob-password-4567");

fn auth_fixture_path() -> PathBuf {
    // The fixture lives in the sibling `mlflow-auth` crate.
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
            "mlflow_rust_auth_users_{}_{}_{}.db",
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
    async fn start(tag: &str, with_auth: bool) -> Self {
        let tracking_db = TempDb::new(&format!("{tag}_track"), &tracking_fixture_path());
        let db = Db::connect(&tracking_db.uri(), PoolConfig::default())
            .await
            .expect("connect tracking fixture");
        let store = TrackingStore::new(db, ART_ROOT);

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
    body: String,
    json: Value,
}

fn basic_header(user: &str, pass: &str) -> String {
    let raw = format!("{user}:{pass}");
    let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
    format!("Basic {encoded}")
}

/// Send a request. `auth` is `Some((user, pass))` for HTTP Basic; `body` is a
/// JSON body (adds `Content-Type: application/json`).
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
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    HttpResponse { status, body, json }
}

/// Options for [`send_raw_with_cookie`]: the pieces beyond method/path/body
/// that vary per call (auth, content type, and the T9.7 CSRF carriers).
/// Bundled into a struct rather than positional params since the CSRF cases
/// need most of these independently toggled.
#[derive(Default)]
struct RawRequestOpts<'a> {
    auth: Option<(&'a str, &'a str)>,
    content_type: Option<&'a str>,
    cookie: Option<&'a str>,
    csrf_header: Option<&'a str>,
}

/// Send a raw (non-JSON) body with the given options, optionally carrying a
/// `Cookie` and/or `X-CSRFToken` header (the `create-ui` CSRF flow needs the
/// cookie always, and the header when the body isn't form-urlencoded).
async fn send_raw_with_cookie(
    base: &str,
    method: Method,
    path: &str,
    opts: RawRequestOpts<'_>,
    body: &str,
) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{base}{path}");
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some((u, p)) = opts.auth {
        builder = builder.header("Authorization", basic_header(u, p));
    }
    if let Some(ct) = opts.content_type {
        builder = builder.header("Content-Type", ct);
    }
    if let Some(c) = opts.cookie {
        builder = builder.header("Cookie", c);
    }
    if let Some(t) = opts.csrf_header {
        builder = builder.header("X-CSRFToken", t);
    }
    let req = builder
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    HttpResponse { status, body, json }
}

/// Send a raw (non-JSON) body with an explicit content type.
async fn send_raw(
    base: &str,
    method: Method,
    path: &str,
    auth: Option<(&str, &str)>,
    content_type: Option<&str>,
    body: &str,
) -> HttpResponse {
    send_raw_with_cookie(
        base,
        method,
        path,
        RawRequestOpts {
            auth,
            content_type,
            ..Default::default()
        },
        body,
    )
    .await
}

/// `GET /signup`, returning the `Set-Cookie` value and the `csrf_token` form
/// field embedded in the rendered HTML — the pair a browser-driven signup
/// POST needs to pass CSRF (T9.7).
async fn fetch_csrf_pair(base: &str) -> (String, String) {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    // `/signup` sits behind `_before_request` like every route (T9.4): it
    // needs an authenticated caller, and the admin bypasses its
    // `validate_can_create_user` gate.
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("{base}/signup"))
        .header("Authorization", basic_header(ALICE.0, ALICE.1))
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    assert_eq!(resp.status(), StatusCode::OK);
    let cookie = resp
        .headers()
        .get("set-cookie")
        .expect("signup sets a cookie")
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let html = String::from_utf8_lossy(&bytes).into_owned();
    let marker = "name=\"csrf_token\" value=\"";
    let start = html.find(marker).expect("csrf field present") + marker.len();
    let end = html[start..].find('"').expect("closing quote") + start;
    (cookie, html[start..end].to_string())
}

// ---------------------------------------------------------------------------
// create-user
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_user_returns_user_shape() {
    let srv = TestServer::start("create", true).await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some(json!({"username": "carol", "password": "carol-password-1"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    // `{"user": {"id", "username", "is_admin"}}` — no password hash, no
    // permission arrays.
    let user = &resp.json["user"];
    assert_eq!(user["username"], "carol");
    assert_eq!(user["is_admin"], false);
    assert!(user["id"].is_number());
    assert!(user.get("password_hash").is_none());
    assert!(user.get("experiment_permissions").is_none());
    assert!(user.get("registered_model_permissions").is_none());
    // Exactly three keys.
    assert_eq!(user.as_object().unwrap().len(), 3);
}

#[tokio::test]
async fn create_user_requires_json_content_type() {
    let srv = TestServer::start("create_ct", true).await;
    let resp = send_raw(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some("text/plain"),
        "username=x&password=y",
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.body, "Invalid content type. Must be application/json");
}

#[tokio::test]
async fn create_user_rejects_empty_username_or_password() {
    let srv = TestServer::start("create_empty", true).await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some(json!({"username": "", "password": "carol-password-1"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.body, "Username and password cannot be empty.");
}

#[tokio::test]
async fn create_user_missing_param_is_400_with_message() {
    let srv = TestServer::start("create_missing", true).await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some(json!({"username": "carol"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Missing value for required parameter 'password'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn create_user_null_body_is_400() {
    // A JSON-typed POST whose body is literal `null` coerces to `{}`, so the
    // missing-param 400 fires (not a 500).
    let srv = TestServer::start("create_null", true).await;
    let resp = send_raw(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some("application/json"),
        "null",
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
}

#[tokio::test]
async fn create_user_short_password_is_rejected() {
    let srv = TestServer::start("create_short", true).await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some(json!({"username": "carol", "password": "short"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Password must be a string longer than 12 characters."
    );
}

#[tokio::test]
async fn create_user_duplicate_is_rejected() {
    let srv = TestServer::start("create_dup", true).await;
    // `alice_scrypt` already exists in the fixture.
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some(json!({"username": ALICE.0, "password": "some-long-password"})),
    )
    .await;
    // `RESOURCE_ALREADY_EXISTS` maps to HTTP 400 (Python's
    // `ERROR_CODE_TO_HTTP_STATUS`).
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.body);
    assert_eq!(resp.json["error_code"], "RESOURCE_ALREADY_EXISTS");
}

// ---------------------------------------------------------------------------
// get-user / current / list
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_user_returns_user_shape() {
    let srv = TestServer::start("get", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/users/get?username={}", BOB.0),
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert_eq!(resp.json["user"]["username"], BOB.0);
    assert_eq!(resp.json["user"]["is_admin"], false);
}

#[tokio::test]
async fn get_user_missing_username_is_400() {
    let srv = TestServer::start("get_missing", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/get",
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Missing value for required parameter 'username'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn get_user_not_found_is_404() {
    let srv = TestServer::start("get_404", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/get?username=nobody",
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
    assert_eq!(resp.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
    assert_eq!(resp.json["message"], "User with username=nobody not found");
}

#[tokio::test]
async fn get_current_user_returns_identity_and_is_basic_auth() {
    let srv = TestServer::start("current", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        Some(BOB),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert_eq!(resp.json["user"]["username"], BOB.0);
    assert_eq!(resp.json["user"]["is_admin"], false);
    assert_eq!(resp.json["is_basic_auth"], true);
    // The admin caller reports is_admin true.
    let admin = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(admin.json["user"]["username"], ALICE.0);
    assert_eq!(admin.json["user"]["is_admin"], true);
}

#[tokio::test]
async fn get_current_user_unauthenticated_is_401() {
    let srv = TestServer::start("current_401", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        None,
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED);
    assert!(resp.body.starts_with("You are not authenticated."));
}

#[tokio::test]
async fn list_users_returns_users_with_roles() {
    let srv = TestServer::start("list", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/list",
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    let users = resp.json["users"].as_array().unwrap();
    // Fixture seeds both users; each row has {id, username, is_admin, roles}.
    let names: Vec<&str> = users
        .iter()
        .map(|u| u["username"].as_str().unwrap())
        .collect();
    assert!(names.contains(&ALICE.0));
    assert!(names.contains(&BOB.0));
    let alice = users
        .iter()
        .find(|u| u["username"] == ALICE.0)
        .expect("alice row");
    assert_eq!(alice["is_admin"], true);
    assert!(alice["roles"].is_array());
    assert_eq!(alice.as_object().unwrap().len(), 4);
}

// ---------------------------------------------------------------------------
// update-password (admin + self-service rules)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_update_password_no_current_password_needed() {
    let srv = TestServer::start("pw_admin", true).await;
    // Admin (alice) changing bob's password: no current_password required.
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-password",
        Some(ALICE),
        Some(json!({"username": BOB.0, "password": "bob-new-password-1"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    // Empty JSON object body.
    assert_eq!(resp.json, json!({}));
    // The new password now authenticates (login as bob with it).
    let check = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        Some((BOB.0, "bob-new-password-1")),
        None,
    )
    .await;
    assert_eq!(check.status, StatusCode::OK);
}

#[tokio::test]
async fn self_service_password_requires_current_password() {
    let srv = TestServer::start("pw_self_missing", true).await;
    // Bob changing his own password without current_password: rejected.
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-password",
        Some(BOB),
        Some(json!({"username": BOB.0, "password": "bob-new-password-1"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Current password is required when changing your own password."
    );
}

#[tokio::test]
async fn self_service_password_wrong_current_password_rejected() {
    let srv = TestServer::start("pw_self_wrong", true).await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-password",
        Some(BOB),
        Some(json!({
            "username": BOB.0,
            "password": "bob-new-password-1",
            "current_password": "not-the-current-password"
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["message"], "Current password does not match.");
}

#[tokio::test]
async fn self_service_password_same_as_current_rejected() {
    let srv = TestServer::start("pw_self_same", true).await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-password",
        Some(BOB),
        Some(json!({
            "username": BOB.0,
            "password": BOB.1,
            "current_password": BOB.1
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp.json["message"],
        "New password must differ from the current password."
    );
}

#[tokio::test]
async fn self_service_password_correct_current_password_accepted() {
    let srv = TestServer::start("pw_self_ok", true).await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-password",
        Some(BOB),
        Some(json!({
            "username": BOB.0,
            "password": "bob-brand-new-pass",
            "current_password": BOB.1
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    // Old password no longer works, new one does.
    let old = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        Some(BOB),
        None,
    )
    .await;
    assert_eq!(old.status, StatusCode::UNAUTHORIZED);
    let new = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/current",
        Some((BOB.0, "bob-brand-new-pass")),
        None,
    )
    .await;
    assert_eq!(new.status, StatusCode::OK);
}

#[tokio::test]
async fn self_service_password_null_body_is_400() {
    let srv = TestServer::start("pw_null", true).await;
    // `null` body for a self-service change coerces to `{}`; username is then
    // missing so the standard missing-param 400 fires (not a 500).
    let resp = send_raw(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-password",
        Some(BOB),
        Some("application/json"),
        "null",
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// update-admin
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_admin_promotes_user() {
    let srv = TestServer::start("admin", true).await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-admin",
        Some(ALICE),
        Some(json!({"username": BOB.0, "is_admin": true})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert_eq!(resp.json, json!({}));
    // Reflected in get-user.
    let got = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/users/get?username={}", BOB.0),
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(got.json["user"]["is_admin"], true);
}

#[tokio::test]
async fn update_admin_non_bool_is_400() {
    let srv = TestServer::start("admin_nonbool", true).await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        "/api/2.0/mlflow/users/update-admin",
        Some(ALICE),
        Some(json!({"username": BOB.0, "is_admin": "yes"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// delete-user (incl. cannot-delete-self)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_user_removes_user() {
    let srv = TestServer::start("delete", true).await;
    // Create a throwaway user, then delete it as admin.
    let created = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create",
        Some(ALICE),
        Some(json!({"username": "todelete", "password": "delete-me-please"})),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{}", created.body);

    let resp = send(
        &srv.base,
        Method::DELETE,
        "/api/2.0/mlflow/users/delete",
        Some(ALICE),
        Some(json!({"username": "todelete"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert_eq!(resp.json, json!({}));

    // Gone.
    let got = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/get?username=todelete",
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(got.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_user_rejects_self_delete() {
    let srv = TestServer::start("delete_self", true).await;
    // Alice (admin) tries to delete herself.
    let resp = send(
        &srv.base,
        Method::DELETE,
        "/api/2.0/mlflow/users/delete",
        Some(ALICE),
        Some(json!({"username": ALICE.0})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "BAD_REQUEST");
    assert!(resp.json["message"]
        .as_str()
        .unwrap()
        .contains("cannot delete their own account"));
    // Alice still exists and is usable.
    let got = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/users/get?username={}", ALICE.0),
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(got.status, StatusCode::OK);
}

#[tokio::test]
async fn delete_user_missing_username_is_400() {
    let srv = TestServer::start("delete_missing", true).await;
    let resp = send(
        &srv.base,
        Method::DELETE,
        "/api/2.0/mlflow/users/delete",
        Some(ALICE),
        Some(json!({})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// ajax prefix + auth-disabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn endpoints_also_served_under_ajax_prefix() {
    let srv = TestServer::start("ajax", true).await;
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("/ajax-api/2.0/mlflow/users/get?username={}", BOB.0),
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert_eq!(resp.json["user"]["username"], BOB.0);
}

#[tokio::test]
async fn routes_absent_when_auth_disabled() {
    let srv = TestServer::start("no_auth", false).await;
    // Every user endpoint 404s when the basic-auth app is not enabled.
    for (method, path) in [
        (Method::POST, "/api/2.0/mlflow/users/create"),
        (Method::GET, "/api/2.0/mlflow/users/get?username=x"),
        (Method::GET, "/api/2.0/mlflow/users/current"),
        (Method::GET, "/api/2.0/mlflow/users/list"),
        (Method::PATCH, "/api/2.0/mlflow/users/update-password"),
        (Method::PATCH, "/api/2.0/mlflow/users/update-admin"),
        (Method::DELETE, "/api/2.0/mlflow/users/delete"),
        (Method::POST, "/api/2.0/mlflow/users/create-ui"),
        (Method::GET, "/signup"),
    ] {
        let resp = send(&srv.base, method.clone(), path, Some(ALICE), None).await;
        assert_eq!(
            resp.status,
            StatusCode::NOT_FOUND,
            "{method} {path} should 404 when auth disabled"
        );
    }
}

#[tokio::test]
async fn create_user_ui_rejects_wrong_content_type() {
    // T9.7: CSRF is checked *before* content type, so a JSON-content-typed
    // POST needs a valid CSRF pair to actually reach the content-type
    // branch this test exercises (see `auth_signup_http.rs` for the
    // CSRF-rejection cases themselves).
    let srv = TestServer::start("ui_ct", true).await;
    let (cookie, csrf_token) = fetch_csrf_pair(&srv.base).await;
    // A JSON body has no `csrf_token` form field, so the token travels via
    // the `X-CSRFToken` header instead — Python's `_get_csrf_token` fallback.
    let resp = send_raw_with_cookie(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create-ui",
        RawRequestOpts {
            auth: Some(ALICE),
            content_type: Some("application/json"),
            cookie: Some(&cookie),
            csrf_header: Some(&csrf_token),
        },
        "{}",
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp.body,
        "Invalid content type. Must be application/x-www-form-urlencoded"
    );
}

#[tokio::test]
async fn create_user_ui_form_creates_user() {
    let srv = TestServer::start("ui_ok", true).await;
    let (cookie, csrf_token) = fetch_csrf_pair(&srv.base).await;
    let body = format!("username=dave&password=dave-password-12&csrf_token={csrf_token}");
    let resp = send_raw_with_cookie(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/users/create-ui",
        RawRequestOpts {
            auth: Some(ALICE),
            content_type: Some("application/x-www-form-urlencoded"),
            cookie: Some(&cookie),
            ..Default::default()
        },
        &body,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    // The user now exists.
    let got = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/users/get?username=dave",
        Some(ALICE),
        None,
    )
    .await;
    assert_eq!(got.status, StatusCode::OK);
    assert_eq!(got.json["user"]["username"], "dave");
}
