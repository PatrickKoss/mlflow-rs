//! HTTP trigger-matrix tests for the T8.4 registry webhook event triggers.
//!
//! Boots the axum app (same real-socket harness as `registry_http.rs`) wired
//! with a webhook store **and** a [`WebhookDispatcher`] pointed at a local
//! recording receiver, then drives each registry mutation over HTTP and asserts
//! the exact `(entity, action)` + payload keys Python fires
//! (`mlflow/server/handlers.py` `deliver_webhook(...)` sites; payload shapes in
//! `mlflow/webhooks/types.py`).
//!
//! ## The differential assertion
//!
//! One webhook subscribes to **all** 14 registry events. A single receiver
//! records every delivered envelope (`{entity, action, timestamp, data}`). For
//! each mutation we snapshot the receiver count, perform the HTTP call, then
//! poll until exactly one new envelope arrives (or time out) and assert its
//! entity/action/data. Mutations Python does *not* fire on (update RM/MV
//! description, rename, transition stage, delete, and — critically — a
//! **non-prompt** RM tag set/delete) are asserted to deliver **nothing**.
//!
//! The registry handlers call the true fire-and-forget `fire`, so delivery is a
//! detached task; the receiver poll (with a generous timeout) is the
//! determinism mechanism, mirroring the `webhook_delivery.rs` receiver pattern.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use mlflow_webhooks::http_send::SendConfig;
use mlflow_webhooks::{
    Resolver, SecretCipher, WebhookAction, WebhookDispatcher, WebhookEntity, WebhookEvent,
    WebhookStatus, WebhookStore,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const WS: &str = "default";
const FIXED_KEY: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
const API: &str = "/api/2.0/mlflow";

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
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_server_webhooktrig_{}_{}_{}.db",
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

// ---------------------------------------------------------------------------
// Recording receiver
// ---------------------------------------------------------------------------

/// A recorded delivery: the parsed `{entity, action, timestamp, data}` envelope.
#[derive(Debug, Clone)]
struct Delivery {
    entity: String,
    action: String,
    data: Value,
}

/// A minimal HTTP/1.1 server that records every inbound webhook envelope and
/// always replies 200. Runs for the lifetime of the test.
struct Receiver {
    addr: SocketAddr,
    deliveries: Arc<Mutex<Vec<Delivery>>>,
}

impl Receiver {
    async fn start() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().unwrap();
        let deliveries = Arc::new(Mutex::new(Vec::new()));
        let sink = deliveries.clone();
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let sink = sink.clone();
                tokio::spawn(async move {
                    handle_conn(stream, sink).await;
                });
            }
        });
        Receiver { addr, deliveries }
    }

    fn port(&self) -> u16 {
        self.addr.port()
    }

    fn count(&self) -> usize {
        self.deliveries.lock().unwrap().len()
    }

    fn snapshot(&self) -> Vec<Delivery> {
        self.deliveries.lock().unwrap().clone()
    }

    /// Poll until the recorded count exceeds `from` (a new delivery arrived) or
    /// ~2s elapses. Returns the new deliveries beyond `from`.
    async fn wait_for_new(&self, from: usize) -> Vec<Delivery> {
        for _ in 0..200 {
            if self.count() > from {
                // Small settle so a spurious extra (bug) would also be captured.
                tokio::time::sleep(Duration::from_millis(10)).await;
                return self.snapshot()[from..].to_vec();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Vec::new()
    }

    /// Assert that no new delivery arrives within a bounded window.
    async fn assert_no_new(&self, from: usize) {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let after = self.snapshot();
        assert_eq!(
            after.len(),
            from,
            "expected no new webhook delivery, got: {:?}",
            &after[from..]
        );
    }
}

async fn handle_conn(mut stream: TcpStream, sink: Arc<Mutex<Vec<Delivery>>>) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        buf.extend_from_slice(&tmp[..n]);
        if let Some(body) = try_parse_body(&buf) {
            if let Ok(env) = serde_json::from_str::<Value>(&body) {
                sink.lock().unwrap().push(Delivery {
                    entity: env["entity"].as_str().unwrap_or_default().to_string(),
                    action: env["action"].as_str().unwrap_or_default().to_string(),
                    data: env["data"].clone(),
                });
            }
            break;
        }
    }
    let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
    let _ = stream.write_all(resp.as_bytes()).await;
    let _ = stream.flush().await;
}

/// Return the body once the full request (headers + Content-Length) is buffered.
fn try_parse_body(buf: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(buf);
    let split = text.find("\r\n\r\n")?;
    let (head, rest) = text.split_at(split);
    let body_part = &rest[4..];
    let mut headers = HashMap::new();
    for line in head.split("\r\n").skip(1) {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if body_part.len() < content_length {
        return None;
    }
    Some(body_part[..content_length].to_string())
}

// ---------------------------------------------------------------------------
// Test app harness (registry + webhook store + dispatcher → receiver)
// ---------------------------------------------------------------------------

struct StaticResolver;
impl Resolver for StaticResolver {
    fn resolve(&self, _host: &str) -> Result<Vec<std::net::IpAddr>, String> {
        Ok(vec!["127.0.0.1".parse().unwrap()])
    }
}

/// Allow `http` webhook URLs + private IPs for `create_webhook`'s URL validation
/// (the dispatcher itself gets the escape hatch through `SendConfig`).
fn allow_http_scheme() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("MLFLOW_WEBHOOK_ALLOWED_SCHEMES", "http,https");
        std::env::set_var("MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS", "true");
    });
}

/// Every `(entity, action)` in the T8.4 matrix, so one webhook receives all of
/// them.
fn all_matrix_events() -> Vec<WebhookEvent> {
    use WebhookAction::*;
    use WebhookEntity::*;
    [
        (RegisteredModel, Created),
        (ModelVersion, Created),
        (ModelVersionTag, Set),
        (ModelVersionTag, Deleted),
        (ModelVersionAlias, Created),
        (ModelVersionAlias, Deleted),
        (Prompt, Created),
        (PromptVersion, Created),
        (PromptTag, Set),
        (PromptTag, Deleted),
        (PromptVersionTag, Set),
        (PromptVersionTag, Deleted),
        (PromptAlias, Created),
        (PromptAlias, Deleted),
    ]
    .into_iter()
    .map(|(e, a)| WebhookEvent::new(e, a))
    .collect()
}

fn test_send_config() -> SendConfig {
    SendConfig {
        timeout: Duration::from_secs(5),
        max_retries: 0,
        backoff_factor: 0.0,
        backoff_max: Duration::from_secs(0),
        backoff_jitter: 0.0,
        allow_private_ips: true,
    }
}

struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    _db: TempDb,
    receiver: Receiver,
}

impl TestServer {
    async fn start(tag: &str) -> Self {
        allow_http_scheme();
        let receiver = Receiver::start().await;

        let db_file = TempDb::new(tag);
        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        let tracking = TrackingStore::new(db.clone(), "file:///unused".to_string());
        let registry = RegistryStore::new(db.clone());
        let cipher = SecretCipher::from_key(FIXED_KEY).unwrap();
        let webhook_store = WebhookStore::with_cipher(db, cipher);

        // Subscribe one webhook to every matrix event, pointed at the receiver.
        webhook_store
            .create_webhook(
                WS,
                "trigger-matrix",
                &format!("http://webhook.test:{}/hook", receiver.port()),
                &all_matrix_events(),
                None,
                Some("topsecret"),
                Some(WebhookStatus::Active),
            )
            .await
            .expect("create webhook");

        let dispatcher = WebhookDispatcher::with_config(
            webhook_store.clone(),
            WS,
            Arc::new(StaticResolver),
            test_send_config(),
        );

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
        let app_state = AppState::with_registry(tracking, registry, true, None, None)
            .with_webhook_store(webhook_store, dispatcher);
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
            receiver,
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

async fn send(
    server: &TestServer,
    method: Method,
    path: &str,
    body: Option<&Value>,
) -> (StatusCode, Vec<u8>) {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{path}", server.base);
    let bytes = match body {
        Some(v) => Bytes::from(serde_json::to_vec(v).unwrap()),
        None => Bytes::new(),
    };
    let build = || {
        let mut b = Request::builder().method(method.clone()).uri(&url);
        if body.is_some() {
            b = b.header("content-type", "application/json");
        }
        b.body(Full::<Bytes>::new(bytes.clone())).unwrap()
    };
    let mut last = None;
    for _ in 0..50 {
        match client.request(build()).await {
            Ok(res) => {
                let status = res.status();
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return (status, bytes.to_vec());
            }
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
    panic!("failed to connect: {last:?}");
}

async fn post(server: &TestServer, path: &str, body: &Value) -> (StatusCode, Vec<u8>) {
    send(server, Method::POST, path, Some(body)).await
}
async fn patch(server: &TestServer, path: &str, body: &Value) -> (StatusCode, Vec<u8>) {
    send(server, Method::PATCH, path, Some(body)).await
}
async fn delete(server: &TestServer, path: &str, body: &Value) -> (StatusCode, Vec<u8>) {
    send(server, Method::DELETE, path, Some(body)).await
}

fn assert_ok(status: StatusCode, body: &[u8]) {
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(body));
}

fn uniq() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

const IS_PROMPT: &str = "mlflow.prompt.is_prompt";
const PROMPT_TEXT: &str = "mlflow.prompt.text";
const PROMPT_TYPE: &str = "_mlflow_prompt_type";

/// Perform a mutation and return the single new delivery, asserting exactly one.
async fn expect_one(
    server: &TestServer,
    mutate: impl std::future::Future<Output = ()>,
) -> Delivery {
    let before = server.receiver.count();
    mutate.await;
    let new = server.receiver.wait_for_new(before).await;
    assert_eq!(new.len(), 1, "expected exactly one delivery, got {new:?}");
    new.into_iter().next().unwrap()
}

// ===========================================================================
// Non-prompt matrix (regular registered models)
// ===========================================================================

#[tokio::test]
async fn model_mutation_matrix_fires_expected_events() {
    let server = TestServer::start("model_matrix").await;
    let name = format!("m_{}", uniq());

    // RM create → registered_model/created with all request tags + description.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/create"),
            &json!({"name": name, "description": "d", "tags": [{"key": "t", "value": "v"}]}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(d.entity, "registered_model");
    assert_eq!(d.action, "created");
    assert_eq!(d.data["name"], json!(name));
    assert_eq!(d.data["tags"], json!({"t": "v"}));
    assert_eq!(d.data["description"], json!("d"));

    // MV create → model_version/created with source, null run_id, tags, description.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/model-versions/create"),
            &json!({"name": name, "source": "mlflow-artifacts:/m/1", "tags": [{"key": "mt", "value": "mv"}]}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(d.entity, "model_version");
    assert_eq!(d.action, "created");
    assert_eq!(d.data["name"], json!(name));
    assert_eq!(d.data["version"], json!("1"));
    assert_eq!(d.data["source"], json!("mlflow-artifacts:/m/1"));
    assert_eq!(d.data["run_id"], Value::Null);
    assert_eq!(d.data["tags"], json!({"mt": "mv"}));
    assert_eq!(d.data["description"], Value::Null);

    // MV tag set → model_version_tag/set.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/model-versions/set-tag"),
            &json!({"name": name, "version": "1", "key": "k", "value": "vv"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("model_version_tag", "set")
    );
    assert_eq!(
        d.data,
        json!({"name": name, "version": "1", "key": "k", "value": "vv"})
    );

    // MV tag delete → model_version_tag/deleted.
    let d = expect_one(&server, async {
        let (s, b) = delete(
            &server,
            &format!("{API}/model-versions/delete-tag"),
            &json!({"name": name, "version": "1", "key": "k"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("model_version_tag", "deleted")
    );
    assert_eq!(d.data, json!({"name": name, "version": "1", "key": "k"}));

    // Alias set → model_version_alias/created.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/alias"),
            &json!({"name": name, "alias": "champion", "version": "1"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("model_version_alias", "created")
    );
    assert_eq!(
        d.data,
        json!({"name": name, "alias": "champion", "version": "1"})
    );

    // Alias delete → model_version_alias/deleted.
    let d = expect_one(&server, async {
        let (s, b) = delete(
            &server,
            &format!("{API}/registered-models/alias"),
            &json!({"name": name, "alias": "champion"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("model_version_alias", "deleted")
    );
    assert_eq!(d.data, json!({"name": name, "alias": "champion"}));
}

// ===========================================================================
// Prompt matrix (mirror events selected by is_prompt classification)
// ===========================================================================

#[tokio::test]
async fn prompt_mutation_matrix_fires_prompt_mirror_events() {
    let server = TestServer::start("prompt_matrix").await;
    let name = format!("p_{}", uniq());

    // RM create with is_prompt tag → prompt/created; internal tags stripped.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/create"),
            &json!({
                "name": name,
                "description": "pd",
                "tags": [
                    {"key": IS_PROMPT, "value": "true"},
                    {"key": PROMPT_TYPE, "value": "text"},
                    {"key": "user", "value": "uv"}
                ]
            }),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt", "created")
    );
    assert_eq!(d.data["name"], json!(name));
    assert_eq!(d.data["tags"], json!({"user": "uv"}));
    assert_eq!(d.data["description"], json!("pd"));

    // Prompt version create → prompt_version/created; template popped, tags stripped.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/model-versions/create"),
            &json!({
                "name": name,
                "source": "mlflow-artifacts:/p/1",
                "tags": [
                    {"key": IS_PROMPT, "value": "true"},
                    {"key": PROMPT_TEXT, "value": "Hello {{n}}!"},
                    {"key": PROMPT_TYPE, "value": "text"},
                    {"key": "user", "value": "uv"}
                ]
            }),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_version", "created")
    );
    assert_eq!(d.data["name"], json!(name));
    assert_eq!(d.data["version"], json!("1"));
    assert_eq!(d.data["template"], json!("Hello {{n}}!"));
    assert_eq!(d.data["tags"], json!({"user": "uv"}));
    assert_eq!(d.data["description"], Value::Null);
    // Non-prompt-version keys (source/run_id) must be absent for the prompt shape.
    assert!(
        d.data.get("source").is_none(),
        "prompt_version payload must not carry source"
    );
    assert!(
        d.data.get("run_id").is_none(),
        "prompt_version payload must not carry run_id"
    );

    // Prompt tag set (on the prompt RM) → prompt_tag/set.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/set-tag"),
            &json!({"name": name, "key": "pk", "value": "pv"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_tag", "set")
    );
    assert_eq!(d.data, json!({"name": name, "key": "pk", "value": "pv"}));

    // Prompt tag delete → prompt_tag/deleted.
    let d = expect_one(&server, async {
        let (s, b) = delete(
            &server,
            &format!("{API}/registered-models/delete-tag"),
            &json!({"name": name, "key": "pk"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_tag", "deleted")
    );
    assert_eq!(d.data, json!({"name": name, "key": "pk"}));

    // Prompt version tag set → prompt_version_tag/set.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/model-versions/set-tag"),
            &json!({"name": name, "version": "1", "key": "vk", "value": "vv"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_version_tag", "set")
    );
    assert_eq!(
        d.data,
        json!({"name": name, "version": "1", "key": "vk", "value": "vv"})
    );

    // Prompt version tag delete → prompt_version_tag/deleted.
    let d = expect_one(&server, async {
        let (s, b) = delete(
            &server,
            &format!("{API}/model-versions/delete-tag"),
            &json!({"name": name, "version": "1", "key": "vk"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_version_tag", "deleted")
    );
    assert_eq!(d.data, json!({"name": name, "version": "1", "key": "vk"}));

    // Prompt alias set → prompt_alias/created.
    let d = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/alias"),
            &json!({"name": name, "alias": "prod", "version": "1"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_alias", "created")
    );
    assert_eq!(
        d.data,
        json!({"name": name, "alias": "prod", "version": "1"})
    );

    // Prompt alias delete → prompt_alias/deleted.
    let d = expect_one(&server, async {
        let (s, b) = delete(
            &server,
            &format!("{API}/registered-models/alias"),
            &json!({"name": name, "alias": "prod"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    assert_eq!(
        (d.entity.as_str(), d.action.as_str()),
        ("prompt_alias", "deleted")
    );
    assert_eq!(d.data, json!({"name": name, "alias": "prod"}));
}

// ===========================================================================
// Negative cases: mutations Python does NOT fire on
// ===========================================================================

#[tokio::test]
async fn non_prompt_registered_model_tag_mutations_fire_nothing() {
    let server = TestServer::start("rm_tag_silent").await;
    let name = format!("m_{}", uniq());

    // Create (fires registered_model/created — drain it).
    let _ = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/create"),
            &json!({"name": name}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;

    // Non-prompt RM tag set: Python only fires for prompts → nothing here.
    let before = server.receiver.count();
    let (s, b) = post(
        &server,
        &format!("{API}/registered-models/set-tag"),
        &json!({"name": name, "key": "k", "value": "v"}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;

    // Non-prompt RM tag delete: likewise nothing.
    let before = server.receiver.count();
    let (s, b) = delete(
        &server,
        &format!("{API}/registered-models/delete-tag"),
        &json!({"name": name, "key": "k"}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;
}

#[tokio::test]
async fn non_triggering_mutations_fire_nothing() {
    let server = TestServer::start("silent_mutations").await;
    let name = format!("m_{}", uniq());

    // Seed: create RM (drain) + create MV (drain).
    let _ = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/registered-models/create"),
            &json!({"name": name}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;
    let _ = expect_one(&server, async {
        let (s, b) = post(
            &server,
            &format!("{API}/model-versions/create"),
            &json!({"name": name, "source": "mlflow-artifacts:/m/1"}),
        )
        .await;
        assert_ok(s, &b);
    })
    .await;

    // update RM description — no webhook (Python `_update_registered_model`).
    let before = server.receiver.count();
    let (s, b) = patch(
        &server,
        &format!("{API}/registered-models/update"),
        &json!({"name": name, "description": "new"}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;

    // update MV description — no webhook.
    let before = server.receiver.count();
    let (s, b) = patch(
        &server,
        &format!("{API}/model-versions/update"),
        &json!({"name": name, "version": "1", "description": "d"}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;

    // transition stage — no webhook.
    let before = server.receiver.count();
    let (s, b) = post(
        &server,
        &format!("{API}/model-versions/transition-stage"),
        &json!({"name": name, "version": "1", "stage": "Staging", "archive_existing_versions": false}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;

    // delete MV — no webhook.
    let before = server.receiver.count();
    let (s, b) = delete(
        &server,
        &format!("{API}/model-versions/delete"),
        &json!({"name": name, "version": "1"}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;

    // rename + delete RM — no webhook.
    let renamed = format!("{name}_r");
    let before = server.receiver.count();
    let (s, b) = post(
        &server,
        &format!("{API}/registered-models/rename"),
        &json!({"name": name, "new_name": renamed}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;

    let before = server.receiver.count();
    let (s, b) = delete(
        &server,
        &format!("{API}/registered-models/delete"),
        &json!({"name": renamed}),
    )
    .await;
    assert_ok(s, &b);
    server.receiver.assert_no_new(before).await;
}
