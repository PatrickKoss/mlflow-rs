//! HTTP integration tests for the auth *after-request* hooks (plan T9.5, §3.16).
//!
//! Ports the behaviors of `_after_request` from `mlflow/server/auth/__init__.py`
//! for the RPCs this Rust server serves:
//!
//! * Creator MANAGE grant on `createExperiment` / `createRegisteredModel`
//!   (incl. the prompt → `prompt` namespace classification from the response
//!   `mlflow.prompt.is_prompt` tag).
//! * `search` response filtering for experiments / registered-models /
//!   model-versions / logged-models — a non-admin sees only readable rows, and
//!   the paged searches (experiments / registered-models / logged-models) refill
//!   the page from the next token exactly as Python does. Admins skip filtering.
//! * Grant cascade on registered-model delete / rename.
//!
//! Boots the axum app on an ephemeral socket with an `AuthStore` (a copy of the
//! committed fixture) + a `TrackingStore`/`RegistryStore` (a copy of the tracking
//! fixture), seeds users/grants/resources through the stores, then drives the
//! gated endpoints over HTTP — the same style as `auth_middleware_http.rs`.

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
            "mlflow_rust_authafter_{}_{}_{}.db",
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
        // `tracking` is only needed to build the app state; the after-request
        // tests in this (default-`READ`) binary drive everything over HTTP.

        let auth_db_file = TempDb::new(&format!("{tag}_auth"), &auth_fixture_path());
        let auth_db =
            AuthDb::connect_and_verify_with(&auth_db_file.uri(), None, PoolConfig::default())
                .await
                .expect("connect + verify auth fixture");
        let auth = AuthStore::new(auth_db);

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

    /// Read the caller's effective grant for `(resource_type, resource_id)` —
    /// used to assert creator/cascade grants directly through the store.
    async fn effective_permission(
        &self,
        username: &str,
        resource_type: &str,
        resource_id: &str,
    ) -> Option<&'static str> {
        let user = self.auth.get_user(username).await.expect("user");
        self.auth
            .get_role_permission_for_resource(user.id, resource_type, resource_id, WS)
            .await
            .expect("resolve")
            .map(|p| p.name)
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

// ---------------------------------------------------------------------------
// Creator MANAGE grants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_experiment_grants_creator_manage() {
    let srv = TestServer::start("creator_exp").await;
    let (u, pw) = srv.create_user("carol_create").await;
    // Without MANAGE, the creator would not be able to delete. Create then delete.
    let create = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/create",
        Some((&u, &pw)),
        Some(json!({"name": "creator-owned-exp"})),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);
    let exp_id = create.json()["experiment_id"].as_str().unwrap().to_string();

    // The after-request hook granted (experiment, id, MANAGE).
    assert_eq!(
        srv.effective_permission(&u, "experiment", &exp_id).await,
        Some("MANAGE")
    );

    // And that MANAGE is live: the creator can delete the experiment.
    let delete = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/experiments/delete",
        Some((&u, &pw)),
        Some(json!({"experiment_id": exp_id})),
    )
    .await;
    assert_ne!(delete.status, StatusCode::FORBIDDEN, "{}", delete.body);
}

#[tokio::test]
async fn create_registered_model_grants_creator_manage_in_rm_namespace() {
    let srv = TestServer::start("creator_rm").await;
    let (u, pw) = srv.create_user("dave_rm").await;
    let create = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/registered-models/create",
        Some((&u, &pw)),
        Some(json!({"name": "creator-rm"})),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);

    assert_eq!(
        srv.effective_permission(&u, "registered_model", "creator-rm")
            .await,
        Some("MANAGE")
    );
    // The prompt namespace is untouched for a plain registered model.
    assert_eq!(
        srv.effective_permission(&u, "prompt", "creator-rm").await,
        None
    );
}

#[tokio::test]
async fn create_prompt_grants_creator_manage_in_prompt_namespace() {
    let srv = TestServer::start("creator_prompt").await;
    let (u, pw) = srv.create_user("erin_prompt").await;
    // A prompt is created via the shared registered-model surface with the
    // `mlflow.prompt.is_prompt` tag; the response carries the persisted tag, so
    // the creator grant lands in the `prompt` namespace.
    let create = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/registered-models/create",
        Some((&u, &pw)),
        Some(json!({
            "name": "creator-prompt",
            "tags": [{"key": "mlflow.prompt.is_prompt", "value": "true"}]
        })),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);

    assert_eq!(
        srv.effective_permission(&u, "prompt", "creator-prompt")
            .await,
        Some("MANAGE")
    );
    // Nothing granted in the registered_model namespace for a prompt.
    assert_eq!(
        srv.effective_permission(&u, "registered_model", "creator-prompt")
            .await,
        None
    );
}

// ---------------------------------------------------------------------------
// Grant cascade on registered-model delete / rename
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_registered_model_cascades_grants() {
    let srv = TestServer::start("rm_delete_cascade").await;
    srv.registry
        .create_registered_model(WS, "doomed-rm", &[], None)
        .await
        .expect("create rm");
    let (u, _pw) = srv.create_user("heidi_del").await;
    srv.grant(&u, "registered_model", "doomed-rm", "MANAGE")
        .await;
    // Also seed a prompt-namespace grant on the same name to prove both sweep.
    srv.grant(&u, "prompt", "doomed-rm", "MANAGE").await;

    // Admin deletes the model; the after-request cascade sweeps both namespaces.
    let del = send(
        &srv.base,
        Method::DELETE,
        "/api/2.0/mlflow/registered-models/delete",
        Some(ALICE),
        Some(json!({"name": "doomed-rm"})),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);

    assert_eq!(
        srv.effective_permission(&u, "registered_model", "doomed-rm")
            .await,
        None
    );
    assert_eq!(
        srv.effective_permission(&u, "prompt", "doomed-rm").await,
        None
    );
}

#[tokio::test]
async fn rename_registered_model_cascades_grants() {
    let srv = TestServer::start("rm_rename_cascade").await;
    srv.registry
        .create_registered_model(WS, "old-rm-name", &[], None)
        .await
        .expect("create rm");
    let (u, _pw) = srv.create_user("ivan_rename").await;
    srv.grant(&u, "registered_model", "old-rm-name", "MANAGE")
        .await;

    let rename = send(
        &srv.base,
        Method::POST,
        "/api/2.0/mlflow/registered-models/rename",
        Some(ALICE),
        Some(json!({"name": "old-rm-name", "new_name": "new-rm-name"})),
    )
    .await;
    assert_eq!(rename.status, StatusCode::OK, "{}", rename.body);

    // The grant moved from the old name to the new name.
    assert_eq!(
        srv.effective_permission(&u, "registered_model", "old-rm-name")
            .await,
        None
    );
    assert_eq!(
        srv.effective_permission(&u, "registered_model", "new-rm-name")
            .await,
        Some("MANAGE")
    );
}
