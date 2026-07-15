//! HTTP integration tests for the 6 webhook endpoints (plan T8.2, §4.16).
//!
//! Boots the axum app on a real ephemeral socket against a fresh copy of the
//! committed Alembic-migrated SQLite fixture (which already contains the
//! `webhooks`/`webhook_events` tables), then drives every endpoint over HTTP:
//! create/get/list/update/delete CRUD, validation errors, list pagination with
//! the encoded token, and `/test` against a local listener asserting the
//! signature + the three `X-MLflow-*` headers.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use mlflow_webhooks::{SecretCipher, WebhookStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const FIXED_KEY: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";

/// Serializes tests that read or mutate the process-global webhook URL
/// validation env vars (`MLFLOW_WEBHOOK_ALLOWED_SCHEMES` /
/// `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS`), so the `/test` firing test's temporary
/// `http`/private-IP allowance never leaks into the scheme-rejection test. A
/// `tokio::sync::Mutex` so the guard can be safely held across `.await` points.
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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
            "mlflow_rust_server_webhooks_{}_{}_{}.db",
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

struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    _db: TempDb,
}

impl TestServer {
    async fn start(tag: &str) -> Self {
        let db_file = TempDb::new(tag);
        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        // Pin a fixed cipher so secrets round-trip deterministically.
        let cipher = SecretCipher::from_key(FIXED_KEY).unwrap();
        let webhook_store = WebhookStore::with_cipher(db.clone(), cipher);
        let store = TrackingStore::new(db, ART_ROOT);
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
        let state = AppState::new(store).with_webhook_store(webhook_store);
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
            _db: db_file,
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
    json: Value,
}

async fn send(base: &str, method: Method, path: &str, body: Option<Value>) -> HttpResponse {
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{base}{path}");
    let mut builder = Request::builder().method(method).uri(uri);
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
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    HttpResponse { status, json }
}

fn create_body(name: &str, url: &str) -> Value {
    json!({
        "name": name,
        "url": url,
        "events": [{"entity": "REGISTERED_MODEL", "action": "CREATED"}],
        "description": "a hook"
    })
}

#[tokio::test]
async fn create_get_list_update_delete_lifecycle() {
    let srv = TestServer::start("lifecycle").await;

    // Create.
    let created = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/webhooks",
        Some(create_body("hook-1", "https://example.com/h")),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{:?}", created.json);
    let webhook = &created.json["webhook"];
    let id = webhook["webhook_id"].as_str().unwrap().to_string();
    assert_eq!(webhook["name"], "hook-1");
    assert_eq!(webhook["status"], "ACTIVE");
    // Secret is never returned.
    assert!(webhook.get("secret").is_none());

    // Get.
    let got = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/webhooks/{id}"),
        None,
    )
    .await;
    assert_eq!(got.status, StatusCode::OK);
    assert_eq!(got.json["webhook"]["webhook_id"], id);

    // List.
    let listed = send(&srv.base, Method::GET, "/api/2.0/mlflow/webhooks", None).await;
    assert_eq!(listed.status, StatusCode::OK);
    let ids: Vec<&str> = listed.json["webhooks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["webhook_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&id.as_str()));

    // Update (PATCH) — rename + disable.
    let updated = send(
        &srv.base,
        Method::PATCH,
        &format!("/api/2.0/mlflow/webhooks/{id}"),
        Some(json!({"name": "hook-renamed", "status": "DISABLED"})),
    )
    .await;
    assert_eq!(updated.status, StatusCode::OK, "{:?}", updated.json);
    assert_eq!(updated.json["webhook"]["name"], "hook-renamed");
    assert_eq!(updated.json["webhook"]["status"], "DISABLED");

    // Delete (soft).
    let deleted = send(
        &srv.base,
        Method::DELETE,
        &format!("/api/2.0/mlflow/webhooks/{id}"),
        None,
    )
    .await;
    assert_eq!(deleted.status, StatusCode::OK);

    // Now gone.
    let gone = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/webhooks/{id}"),
        None,
    )
    .await;
    assert_eq!(gone.status, StatusCode::NOT_FOUND);
    assert_eq!(gone.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn create_missing_name_is_invalid_parameter() {
    let srv = TestServer::start("missing_name").await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/webhooks",
        Some(json!({"url": "https://example.com/h", "events": [{"entity":"REGISTERED_MODEL","action":"CREATED"}]})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Missing value for required parameter 'name'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn create_bad_scheme_is_invalid_parameter() {
    let _env = ENV_LOCK.lock().await;
    let srv = TestServer::start("bad_scheme").await;
    // Default allowed scheme is https; http is rejected.
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/webhooks",
        Some(create_body("hook", "http://example.com/h")),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Invalid webhook URL scheme: 'http'. Allowed schemes are: https."
    );
}

#[tokio::test]
async fn create_invalid_event_combination_is_invalid_parameter() {
    let srv = TestServer::start("bad_combo").await;
    let resp = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/webhooks",
        Some(json!({
            "name": "hook",
            "url": "https://example.com/h",
            "events": [{"entity": "REGISTERED_MODEL", "action": "DELETED"}]
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "Invalid action 'deleted' for entity 'registered_model'. Valid actions are: ['created']"
    );
}

#[tokio::test]
async fn list_paginates_with_next_page_token() {
    let srv = TestServer::start("pagination").await;
    for i in 0..3 {
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let r = send(
            &srv.base,
            Method::POST,
            "/api/2.0/mlflow/webhooks",
            Some(create_body(&format!("hook-{i}"), "https://example.com/h")),
        )
        .await;
        assert_eq!(r.status, StatusCode::OK);
    }

    let page1 = send(
        &srv.base,
        Method::GET,
        "/api/2.0/mlflow/webhooks?max_results=2",
        None,
    )
    .await;
    assert_eq!(page1.status, StatusCode::OK);
    assert_eq!(page1.json["webhooks"].as_array().unwrap().len(), 2);
    let token = page1.json["next_page_token"].as_str().unwrap().to_string();

    let page2 = send(
        &srv.base,
        Method::GET,
        &format!("/api/2.0/mlflow/webhooks?max_results=2&page_token={token}"),
        None,
    )
    .await;
    assert_eq!(page2.status, StatusCode::OK);
    assert!(!page2.json["webhooks"].as_array().unwrap().is_empty());
}

/// A tiny local HTTP listener that records the first request's headers + body.
struct CapturingListener {
    url: String,
    captured: Arc<Mutex<Option<CapturedRequest>>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Clone)]
struct CapturedRequest {
    headers: Vec<(String, String)>,
    body: String,
}

impl CapturingListener {
    async fn start() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured: Arc<Mutex<Option<CapturedRequest>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            loop {
                let accept = tokio::select! {
                    a = listener.accept() => a,
                    _ = &mut rx => break,
                };
                let Ok((stream, _)) = accept else { break };
                let cap = cap.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |req: Request<Incoming>| {
                        let cap = cap.clone();
                        async move {
                            let headers: Vec<(String, String)> = req
                                .headers()
                                .iter()
                                .map(|(k, v)| {
                                    (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())
                                })
                                .collect();
                            let body = req.into_body().collect().await.unwrap().to_bytes();
                            let body = String::from_utf8_lossy(&body).into_owned();
                            *cap.lock().unwrap() = Some(CapturedRequest { headers, body });
                            Ok::<_, std::convert::Infallible>(Response::new(Full::new(
                                Bytes::from_static(b"ok"),
                            )))
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .await;
                });
            }
        });
        CapturingListener {
            url: format!("http://{addr}/hook"),
            captured,
            shutdown: Some(tx),
            handle: Some(handle),
        }
    }

    fn captured(&self) -> Option<CapturedRequest> {
        self.captured.lock().unwrap().clone()
    }
}

impl Drop for CapturingListener {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

#[tokio::test]
async fn test_endpoint_fires_signed_request_with_headers() {
    let _env = ENV_LOCK.lock().await;
    // Allow the local sink's `http` scheme + localhost IP for the test path
    // (a dev server would set these to deliver to localhost).
    std::env::set_var("MLFLOW_WEBHOOK_ALLOWED_SCHEMES", "http,https");
    std::env::set_var("MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS", "true");

    let sink = CapturingListener::start().await;
    let srv = TestServer::start("test_fire").await;

    // Create a webhook with a secret, pointed at the local sink.
    let created = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/webhooks",
        Some(json!({
            "name": "test-hook",
            "url": sink.url,
            "events": [{"entity": "REGISTERED_MODEL", "action": "CREATED"}],
            "secret": "my-webhook-secret"
        })),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{:?}", created.json);
    let id = created.json["webhook"]["webhook_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Fire the test delivery.
    let result = send(
        &srv.base,
        Method::POST,
        &format!("/api/2.0/mlflow/webhooks/{id}/test"),
        Some(json!({})),
    )
    .await;
    assert_eq!(result.status, StatusCode::OK, "{:?}", result.json);
    let tr = &result.json["result"];
    assert_eq!(tr["success"], true, "{:?}", result.json);
    assert_eq!(tr["response_status"], 200);
    assert_eq!(tr["response_body"], "ok");

    // Give the sink a moment (the request completed before the response, so it
    // is already captured, but be defensive).
    let captured = sink.captured().expect("sink captured a request");
    let header = |name: &str| -> Option<String> {
        captured
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    };
    let delivery_id = header("x-mlflow-delivery-id").expect("delivery id header");
    let timestamp = header("x-mlflow-timestamp").expect("timestamp header");
    let signature = header("x-mlflow-signature").expect("signature header");
    assert_eq!(header("content-type").as_deref(), Some("application/json"));
    assert!(signature.starts_with("v1,"));

    // Recompute the signature and confirm it matches (proves the signed content
    // is `{delivery_id}.{timestamp}.{payload}` with the webhook's secret).
    let expected = mlflow_webhooks::signing::generate_hmac_signature(
        "my-webhook-secret",
        &delivery_id,
        &timestamp,
        &captured.body,
    );
    assert_eq!(signature, expected);

    // The payload wraps the example data with entity/action/timestamp/data.
    let payload: Value = serde_json::from_str(&captured.body).unwrap();
    assert_eq!(payload["entity"], "registered_model");
    assert_eq!(payload["action"], "created");
    assert_eq!(payload["data"]["name"], "example_model");

    std::env::remove_var("MLFLOW_WEBHOOK_ALLOWED_SCHEMES");
    std::env::remove_var("MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS");
}
