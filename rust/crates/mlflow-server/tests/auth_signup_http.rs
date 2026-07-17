//! HTTP integration tests for `/signup` + CSRF (plan T9.7, §3.16 "Signup UI").
//!
//! Boots the axum app on a real ephemeral socket with the basic-auth app
//! enabled, then drives the full browser-style signup flow: `GET /signup`
//! renders the form with an embedded CSRF token and issues a session cookie,
//! `POST create-user-ui` with the valid pair creates the user, and each CSRF
//! rejection path (missing token, missing cookie, invalid token, cookie/token
//! mismatch) gets its own case asserting the exact 400 status + Python
//! message text (`flask_wtf.csrf.validate_csrf`). Finally, the auth-disabled
//! case confirms `/signup` 404s like every other auth-app route.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthDb, AuthStore};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use tokio::net::TcpListener;

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
            "mlflow_rust_auth_signup_{}_{}_{}.db",
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
        let store = TrackingStore::new(db, "s3://bucket/mlruns");

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
    set_cookie: Option<String>,
}

async fn get(base: &str, path: &str) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!("{base}{path}"))
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let set_cookie = resp
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    HttpResponse {
        status,
        body,
        set_cookie,
    }
}

/// `POST create-user-ui` with an explicit `Content-Type`, optional `Cookie`,
/// form-urlencoded body.
async fn post_create_ui(
    base: &str,
    content_type: &str,
    cookie: Option<&str>,
    body: &str,
) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(format!("{base}/api/2.0/mlflow/users/create-ui"))
        .header("Content-Type", content_type);
    if let Some(c) = cookie {
        builder = builder.header("Cookie", c);
    }
    let req = builder
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    let status = resp.status();
    let set_cookie = resp
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).into_owned();
    HttpResponse {
        status,
        body,
        set_cookie,
    }
}

/// `GET /signup`, returning the cookie (first `;`-delimited attribute of
/// `Set-Cookie`) and the embedded `csrf_token` field value.
async fn fetch_csrf_pair(base: &str) -> (String, String) {
    let resp = get(base, "/signup").await;
    assert_eq!(resp.status, StatusCode::OK);
    let cookie = resp
        .set_cookie
        .expect("signup sets a cookie")
        .split(';')
        .next()
        .unwrap()
        .to_string();
    let marker = "name=\"csrf_token\" value=\"";
    let start = resp.body.find(marker).expect("csrf field present") + marker.len();
    let end = resp.body[start..].find('"').expect("closing quote") + start;
    (cookie, resp.body[start..end].to_string())
}

// ---------------------------------------------------------------------------
// GET /signup
// ---------------------------------------------------------------------------

#[tokio::test]
async fn signup_page_renders_form_with_token_and_cookie() {
    let srv = TestServer::start("page", true).await;
    let resp = get(&srv.base, "/signup").await;
    assert_eq!(resp.status, StatusCode::OK);
    assert!(resp.set_cookie.is_some(), "signup must Set-Cookie");
    assert!(resp.body.contains(r#"name="username""#));
    assert!(resp.body.contains(r#"name="password""#));
    assert!(resp.body.contains(r#"name="csrf_token""#));
    assert!(resp
        .body
        .contains(r#"action="/api/2.0/mlflow/users/create-ui""#));
    assert!(resp.body.contains("<svg"));
}

#[tokio::test]
async fn signup_page_issues_a_fresh_pair_each_time() {
    let srv = TestServer::start("fresh", true).await;
    let (cookie_a, token_a) = fetch_csrf_pair(&srv.base).await;
    let (cookie_b, token_b) = fetch_csrf_pair(&srv.base).await;
    assert_ne!(cookie_a, cookie_b);
    assert_ne!(token_a, token_b);
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_signup_flow_creates_user() {
    let srv = TestServer::start("happy", true).await;
    let (cookie, csrf_token) = fetch_csrf_pair(&srv.base).await;
    let body = format!("username=carol&password=carol-password-99&csrf_token={csrf_token}");
    let resp = post_create_ui(
        &srv.base,
        "application/x-www-form-urlencoded",
        Some(&cookie),
        &body,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert!(resp.body.contains("Successfully signed up user: carol"));
    assert!(resp.body.contains(r#"window.location.href = "/""#));

    // The user now exists and can authenticate.
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let encoded = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        "carol:carol-password-99",
    );
    let req = Request::builder()
        .method(Method::GET)
        .uri(format!(
            "{}/api/2.0/mlflow/users/get?username=carol",
            srv.base
        ))
        .header("Authorization", format!("Basic {encoded}"))
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = client.request(req).await.expect("request");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn duplicate_username_flashes_and_redirects_to_signup() {
    let srv = TestServer::start("dup", true).await;
    let (cookie, csrf_token) = fetch_csrf_pair(&srv.base).await;
    // `alice_scrypt` already exists in the fixture.
    let body =
        format!("username=alice_scrypt&password=whatever-password-1&csrf_token={csrf_token}");
    let resp = post_create_ui(
        &srv.base,
        "application/x-www-form-urlencoded",
        Some(&cookie),
        &body,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    assert!(resp
        .body
        .contains("Username has already been taken: alice_scrypt"));
    assert!(resp.body.contains(r#"window.location.href = "/signup""#));
}

// ---------------------------------------------------------------------------
// CSRF rejections
// ---------------------------------------------------------------------------

#[tokio::test]
async fn csrf_less_post_is_rejected() {
    let srv = TestServer::start("no_csrf", true).await;
    let resp = post_create_ui(
        &srv.base,
        "application/x-www-form-urlencoded",
        None,
        "username=eve&password=eve-password-1234",
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.body, "The CSRF token is missing.");
}

#[tokio::test]
async fn missing_cookie_is_rejected() {
    let srv = TestServer::start("no_cookie", true).await;
    let (_cookie, csrf_token) = fetch_csrf_pair(&srv.base).await;
    let body = format!("username=eve&password=eve-password-1234&csrf_token={csrf_token}");
    // Valid token, but no session cookie presented alongside it.
    let resp = post_create_ui(&srv.base, "application/x-www-form-urlencoded", None, &body).await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.body, "The CSRF session token is missing.");
}

#[tokio::test]
async fn invalid_token_is_rejected() {
    let srv = TestServer::start("bad_token", true).await;
    let (cookie, csrf_token) = fetch_csrf_pair(&srv.base).await;
    let tampered = format!("{csrf_token}tampered");
    let body = format!("username=eve&password=eve-password-1234&csrf_token={tampered}");
    let resp = post_create_ui(
        &srv.base,
        "application/x-www-form-urlencoded",
        Some(&cookie),
        &body,
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.body, "The CSRF token is invalid.");
}

#[tokio::test]
async fn token_cookie_mismatch_is_rejected() {
    let srv = TestServer::start("mismatch", true).await;
    // Two independently issued pairs; presenting session A's cookie with
    // session B's token is a well-signed token that just doesn't belong to
    // the presented session.
    let (cookie_a, _token_a) = fetch_csrf_pair(&srv.base).await;
    let (_cookie_b, token_b) = fetch_csrf_pair(&srv.base).await;
    let body = format!("username=eve&password=eve-password-1234&csrf_token={token_b}");
    let resp = post_create_ui(
        &srv.base,
        "application/x-www-form-urlencoded",
        Some(&cookie_a),
        &body,
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.body, "The CSRF tokens do not match.");
}

// ---------------------------------------------------------------------------
// Auth-disabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn signup_404s_when_auth_disabled() {
    let srv = TestServer::start("disabled", false).await;
    let resp = get(&srv.base, "/signup").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
}
