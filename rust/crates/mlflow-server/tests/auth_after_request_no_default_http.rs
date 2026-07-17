//! Auth after-request *filtering* tests that require `default_permission =
//! NO_PERMISSIONS` (plan T9.5). Under the packaged default (`READ`) every row is
//! readable, so filtering is a no-op and unobservable; these mirror Python's
//! `no_permission_auth.ini` posture — the only way to exercise the search
//! response filters and their page-fill deterministically.
//!
//! The floor comes from the parsed [`mlflow_auth::AuthConfig`] carried by the
//! `AuthStore` (T9.8), so each server here is built with `AuthConfig
//! { default_permission: NO_PERMISSIONS, .. }`. They still live in their own
//! test binary to keep the strict-floor posture isolated from the default-`READ`
//! after-request tests in `auth_after_request_http.rs`.

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
            "mlflow_rust_authafter_nd_{}_{}_{}.db",
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

    async fn create_logged_model(&self, experiment_id: &str, name: &str) -> String {
        self.tracking
            .create_logged_model(WS, experiment_id, Some(name), None, &[], &[], None)
            .await
            .expect("create logged model")
            .model_id
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
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON: {e}: {}", self.body))
    }
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
    HttpResponse {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
    }
}

fn experiment_ids(resp: &Value) -> Vec<String> {
    resp["experiments"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| e["experiment_id"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Search-experiments filtering + page-fill across tokens
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_experiments_filters_and_refills_page_for_non_admin() {
    let srv = TestServer::start("search_exp_fill").await;
    // Admin seeds N experiments; grant READ to the user on a subset.
    let mut ids = Vec::new();
    for i in 0..6 {
        ids.push(srv.create_experiment(&format!("filt-exp-{i}")).await);
    }
    let (u, pw) = srv.create_user("frank_search").await;
    // READ on exactly the even-indexed experiments (3 of 6).
    let readable: Vec<String> = ids
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, id)| id.clone())
        .collect();
    for id in &readable {
        srv.grant(&u, "experiment", id, "READ").await;
    }

    // Walk the pages with a small max_results, collecting the filtered stream.
    // Each page must come back *refilled* to readable rows (page-fill parity).
    let mut seen = Vec::new();
    let mut page_token: Option<String> = None;
    for _ in 0..30 {
        let mut body = json!({"max_results": 2, "view_type": 1});
        if let Some(t) = &page_token {
            body["page_token"] = json!(t);
        }
        let resp = send(
            &srv.base,
            Method::POST,
            "/api/2.0/mlflow/experiments/search",
            Some((&u, &pw)),
            Some(body),
        )
        .await;
        assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
        let j = resp.json();
        for id in experiment_ids(&j) {
            assert!(readable.contains(&id), "leaked unreadable experiment {id}");
            seen.push(id);
        }
        match j["next_page_token"].as_str() {
            Some(t) if !t.is_empty() => page_token = Some(t.to_string()),
            _ => break,
        }
    }

    // Exactly the readable experiments, once each.
    seen.sort();
    let mut expected = readable.clone();
    expected.sort();
    assert_eq!(seen, expected);
}

#[tokio::test]
async fn search_experiments_admin_sees_everything_unfiltered() {
    let srv = TestServer::start("search_exp_admin").await;
    let mut ids = Vec::new();
    for i in 0..4 {
        ids.push(srv.create_experiment(&format!("admin-exp-{i}")).await);
    }
    // Admin has no explicit grants but bypasses filtering (`sender_is_admin`).
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/search",
        Some(ALICE),
        Some(json!({"max_results": 100, "view_type": 1})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    let seen = experiment_ids(&resp.json());
    for id in &ids {
        assert!(seen.contains(id), "admin missing experiment {id}");
    }
}

// ---------------------------------------------------------------------------
// Search-registered-models filtering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_registered_models_filters_for_non_admin() {
    let srv = TestServer::start("search_rm").await;
    for i in 0..4 {
        srv.registry
            .create_registered_model(WS, &format!("filt-rm-{i}"), &[], None)
            .await
            .expect("create rm");
    }
    let (u, pw) = srv.create_user("grace_rm").await;
    srv.grant(&u, "registered_model", "filt-rm-0", "READ").await;
    srv.grant(&u, "registered_model", "filt-rm-2", "READ").await;

    let resp = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/registered-models/search?max_results=100",
        Some((&u, &pw)),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    let mut names: Vec<String> = resp.json()["registered_models"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| m["name"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    assert_eq!(
        names,
        vec!["filt-rm-0".to_string(), "filt-rm-2".to_string()]
    );
}

// ---------------------------------------------------------------------------
// Search-logged-models filtering (inherits experiment READ)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_logged_models_filters_by_parent_experiment() {
    let srv = TestServer::start("search_lm").await;
    let exp_readable = srv.create_experiment("lm-readable").await;
    let exp_hidden = srv.create_experiment("lm-hidden").await;
    let readable_model = srv.create_logged_model(&exp_readable, "m-readable").await;
    let _hidden_model = srv.create_logged_model(&exp_hidden, "m-hidden").await;

    let (u, pw) = srv.create_user("judy_lm").await;
    srv.grant(&u, "experiment", &exp_readable, "READ").await;

    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/logged-models/search",
        Some((&u, &pw)),
        Some(json!({"experiment_ids": [exp_readable, exp_hidden], "max_results": 100})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    let model_ids: Vec<String> = resp.json()["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| m["info"]["model_id"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(model_ids, vec![readable_model]);
}

#[tokio::test]
async fn search_logged_models_filters_and_refills_page_across_tokens() {
    let srv = TestServer::start("search_lm_fill").await;
    let exp_readable = srv.create_experiment("lm-fill-readable").await;
    let exp_hidden = srv.create_experiment("lm-fill-hidden").await;
    // Interleave readable and hidden models so a small page never fills from one
    // batch — forcing the token-driven refill loop to walk multiple pages. Names
    // are distinct and monotonically increasing so ordering by `name ASC` gives
    // a stable total order (the default `creation_timestamp_ms` tiebreak is not
    // unique when models are created in the same millisecond).
    let mut readable_models = Vec::new();
    for i in 0..4 {
        readable_models.push(
            srv.create_logged_model(&exp_readable, &format!("m-{:02}-readable", i * 2))
                .await,
        );
        srv.create_logged_model(&exp_hidden, &format!("m-{:02}-hidden", i * 2 + 1))
            .await;
    }

    let (u, pw) = srv.create_user("kate_lm").await;
    srv.grant(&u, "experiment", &exp_readable, "READ").await;

    let mut seen = Vec::new();
    let mut page_token: Option<String> = None;
    for _ in 0..30 {
        let mut body = json!({
            "experiment_ids": [exp_readable, exp_hidden],
            "max_results": 2,
            "order_by": [{"field_name": "name", "ascending": true}]
        });
        if let Some(t) = &page_token {
            body["page_token"] = json!(t);
        }
        let resp = send(
            &srv.base,
            Method::POST,
            "/api/2.0/mlflow/logged-models/search",
            Some((&u, &pw)),
            Some(body),
        )
        .await;
        assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
        let j = resp.json();
        if let Some(models) = j["models"].as_array() {
            for m in models {
                let id = m["info"]["model_id"].as_str().unwrap().to_string();
                assert!(
                    readable_models.contains(&id),
                    "leaked unreadable logged model {id}"
                );
                seen.push(id);
            }
        }
        match j["next_page_token"].as_str() {
            Some(t) if !t.is_empty() => page_token = Some(t.to_string()),
            _ => break,
        }
    }

    seen.sort();
    let mut expected = readable_models.clone();
    expected.sort();
    assert_eq!(seen, expected);
}
