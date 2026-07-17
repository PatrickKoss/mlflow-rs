//! HTTP integration tests for T10.3 (request workspace context + scoping).
//!
//! Ports the behaviors of `tests/server/test_workspace_middleware.py`,
//! `tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py`,
//! and `tests/store/model_registry/test_sqlalchemy_workspace_store.py` to the
//! Rust server over real HTTP:
//!
//! * header resolution (present / absent → default / trimmed);
//! * server-info skip (a bogus header never breaks server-info);
//! * non-existent workspace → 404 with the byte-matched store message;
//! * workspaces-disabled ignores the header (no error, always `default`);
//! * cross-workspace isolation: experiment + run + registered model created in
//!   workspace A are invisible in B and in `default`; search is scoped;
//! * artifact-location prefixing `workspaces/<name>/<exp_id>` on create (for a
//!   non-default *and* the `default` workspace, since the server root has no
//!   per-workspace override);
//! * forbid explicit `artifact_location` on create when workspaces are enabled.
//!
//! The harness boots the axum app on an ephemeral socket against a fresh copy
//! of the committed Alembic-migrated fixture (which ships a `default` workspace
//! row), wiring the tracking store, registry store, and — for the enabled cases
//! — the workspace store (`with_workspace_store`).

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const WS_HEADER: &str = "X-MLFLOW-WORKSPACE";
const API: &str = "/api/2.0/mlflow";
const API3: &str = "/api/3.0/mlflow";
const WORKSPACES: &str = "/api/3.0/mlflow/workspaces";

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
            "mlflow_rust_ws_scoping_{}_{}_{}.db",
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
    async fn start(tag: &str, enable_workspaces: bool, artifact_root: &str) -> Self {
        let db_file = TempDb::new(tag);
        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        let tracking = TrackingStore::new(db.clone(), artifact_root.to_string());
        let registry = RegistryStore::new(db.clone());
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
        let mut state = AppState::with_registry(tracking, registry, true, None, None);
        if enable_workspaces {
            state = state.with_workspace_store(WorkspaceStore::new(db, db_file.uri()));
        }
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

    async fn enabled(tag: &str) -> Self {
        Self::start(tag, true, ART_ROOT).await
    }

    async fn disabled(tag: &str) -> Self {
        Self::start(tag, false, ART_ROOT).await
    }

    /// Create a workspace via the REST API (enabled server only).
    async fn create_workspace(&self, name: &str) {
        let resp = self
            .send(Method::POST, WORKSPACES, None, Some(json!({"name": name})))
            .await;
        assert_eq!(
            resp.status,
            StatusCode::CREATED,
            "create ws {name}: {}",
            resp.text
        );
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        workspace: Option<&str>,
        body: Option<Value>,
    ) -> HttpResponse {
        let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
        let uri = format!("{}{path}", self.base);
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(ws) = workspace {
            builder = builder.header(WS_HEADER, ws);
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
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let json = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        HttpResponse { status, text, json }
    }

    /// Create an experiment in `workspace`, returning its id.
    async fn create_experiment(&self, workspace: Option<&str>, name: &str) -> String {
        let resp = self
            .send(
                Method::POST,
                &format!("{API}/experiments/create"),
                workspace,
                Some(json!({ "name": name })),
            )
            .await;
        assert_eq!(resp.status, StatusCode::OK, "create exp: {}", resp.text);
        resp.json["experiment_id"].as_str().unwrap().to_string()
    }

    async fn create_registered_model(&self, workspace: Option<&str>, name: &str) -> HttpResponse {
        self.send(
            Method::POST,
            &format!("{API}/registered-models/create"),
            workspace,
            Some(json!({ "name": name })),
        )
        .await
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
    text: String,
    json: Value,
}

// ---------------------------------------------------------------------------
// (A) Header resolution / middleware
// ---------------------------------------------------------------------------

#[tokio::test]
async fn absent_header_resolves_to_default_when_enabled() {
    // The fixture ships a `default` workspace, so an absent header resolves to
    // it and the request succeeds.
    let srv = TestServer::enabled("absent-default").await;
    let id = srv.create_experiment(None, "no-header-exp").await;
    // Visible in `default` (explicit header) and absent-header (both `default`).
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("default"),
            None,
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.json["experiment"]["workspace"], "default");
}

#[tokio::test]
async fn present_header_resolves_to_named_workspace() {
    let srv = TestServer::enabled("present-named").await;
    srv.create_workspace("team-a").await;
    let id = srv.create_experiment(Some("team-a"), "a-exp").await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("team-a"),
            None,
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.json["experiment"]["workspace"], "team-a");
}

#[tokio::test]
async fn whitespace_header_is_trimmed() {
    let srv = TestServer::enabled("trim").await;
    srv.create_workspace("team-a").await;
    // A padded header must resolve to the trimmed name (a real workspace).
    let id = srv
        .create_experiment(Some("  team-a  "), "padded-exp")
        .await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("team-a"),
            None,
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.json["experiment"]["workspace"], "team-a");
}

#[tokio::test]
async fn nonexistent_workspace_header_is_404_byte_matched() {
    let srv = TestServer::enabled("missing-ws").await;
    let resp = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/create"),
            Some("ghost"),
            Some(json!({ "name": "x" })),
        )
        .await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
    assert_eq!(resp.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
    assert_eq!(resp.json["message"], "Workspace 'ghost' not found");
}

#[tokio::test]
async fn invalid_workspace_name_header_is_400() {
    let srv = TestServer::enabled("invalid-name").await;
    // An uppercase name fails `WorkspaceNameValidator` before any store lookup.
    let resp = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/create"),
            Some("Team-Uppercase"),
            Some(json!({ "name": "x" })),
        )
        .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// (F) server-info skip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_info_skips_resolution_with_bogus_header() {
    let srv = TestServer::enabled("server-info-skip").await;
    // A non-existent workspace header must NOT break server-info.
    let resp = srv
        .send(
            Method::GET,
            &format!("{API3}/server-info"),
            Some("missing"),
            None,
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.json["workspaces_enabled"], true);
}

#[tokio::test]
async fn server_info_reports_disabled_and_ignores_header() {
    let srv = TestServer::disabled("server-info-disabled").await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API3}/server-info"),
            Some("some-workspace"),
            None,
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK);
    assert_eq!(resp.json["workspaces_enabled"], false);
}

// ---------------------------------------------------------------------------
// workspaces disabled: header ignored, no error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn disabled_server_ignores_workspace_header() {
    let srv = TestServer::disabled("disabled-ignore").await;
    // A header naming a non-existent workspace is ignored → resolves to
    // `default`, and the create succeeds (single-tenant).
    let id = srv
        .create_experiment(Some("whatever-team"), "single-tenant-exp")
        .await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("another-team"),
            None,
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK);
    // Everything is `default` regardless of the header.
    assert_eq!(resp.json["experiment"]["workspace"], "default");
}

// ---------------------------------------------------------------------------
// (B) cross-workspace isolation — tracking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn experiment_created_in_a_invisible_in_b_and_default() {
    let srv = TestServer::enabled("exp-isolation").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    let id = srv.create_experiment(Some("team-a"), "secret-exp").await;

    // Visible in A.
    let in_a = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("team-a"),
            None,
        )
        .await;
    assert_eq!(in_a.status, StatusCode::OK);

    // Invisible (404) in B and in default.
    for ws in ["team-b", "default"] {
        let resp = srv
            .send(
                Method::GET,
                &format!("{API}/experiments/get?experiment_id={id}"),
                Some(ws),
                None,
            )
            .await;
        assert_eq!(resp.status, StatusCode::NOT_FOUND, "ws {ws}: {}", resp.text);
        assert_eq!(resp.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
    }

    // get-by-name is scoped too.
    let by_name_b = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get-by-name?experiment_name=secret-exp"),
            Some("team-b"),
            None,
        )
        .await;
    assert_eq!(by_name_b.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn same_experiment_name_allowed_across_workspaces_and_search_scoped() {
    let srv = TestServer::enabled("exp-dup-search").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    // Same name in both workspaces — no conflict.
    srv.create_experiment(Some("team-a"), "shared-name").await;
    srv.create_experiment(Some("team-b"), "shared-name").await;
    srv.create_experiment(Some("team-a"), "only-in-a").await;

    // Search in team-a sees `shared-name` + `only-in-a` (+ the seeded default
    // experiment for team-a, which is created lazily; assert the non-default
    // names are the team-a set and `only-in-a` is absent in team-b).
    let search_a = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/search"),
            Some("team-a"),
            Some(json!({ "max_results": 100, "view_type": "ALL" })),
        )
        .await;
    assert_eq!(search_a.status, StatusCode::OK, "{}", search_a.text);
    let names_a: Vec<&str> = search_a.json["experiments"]
        .as_array()
        .map(|a| a.iter().map(|e| e["name"].as_str().unwrap()).collect())
        .unwrap_or_default();
    assert!(names_a.contains(&"shared-name"));
    assert!(names_a.contains(&"only-in-a"));

    let search_b = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/search"),
            Some("team-b"),
            Some(json!({ "max_results": 100, "view_type": "ALL" })),
        )
        .await;
    let names_b: Vec<&str> = search_b.json["experiments"]
        .as_array()
        .map(|a| a.iter().map(|e| e["name"].as_str().unwrap()).collect())
        .unwrap_or_default();
    assert!(names_b.contains(&"shared-name"));
    assert!(!names_b.contains(&"only-in-a"));
}

#[tokio::test]
async fn run_created_in_a_invisible_in_b() {
    let srv = TestServer::enabled("run-isolation").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    let exp_id = srv.create_experiment(Some("team-a"), "run-exp").await;
    let create_run = srv
        .send(
            Method::POST,
            &format!("{API}/runs/create"),
            Some("team-a"),
            Some(json!({ "experiment_id": exp_id, "start_time": 0 })),
        )
        .await;
    assert_eq!(create_run.status, StatusCode::OK, "{}", create_run.text);
    let run_id = create_run.json["run"]["info"]["run_id"]
        .as_str()
        .unwrap()
        .to_string();

    // The run is not visible from team-b.
    let get_b = srv
        .send(
            Method::GET,
            &format!("{API}/runs/get?run_id={run_id}"),
            Some("team-b"),
            None,
        )
        .await;
    assert_eq!(get_b.status, StatusCode::NOT_FOUND, "{}", get_b.text);
    assert_eq!(get_b.json["error_code"], "RESOURCE_DOES_NOT_EXIST");

    // Creating a run against team-a's experiment from team-b fails (experiment
    // not found in team-b).
    let cross = srv
        .send(
            Method::POST,
            &format!("{API}/runs/create"),
            Some("team-b"),
            Some(json!({ "experiment_id": exp_id, "start_time": 0 })),
        )
        .await;
    assert_eq!(cross.status, StatusCode::NOT_FOUND, "{}", cross.text);
}

// ---------------------------------------------------------------------------
// (C) cross-workspace isolation — registry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn registered_model_created_in_a_invisible_in_b_and_default() {
    let srv = TestServer::enabled("rm-isolation").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    let created = srv.create_registered_model(Some("team-a"), "alpha").await;
    assert_eq!(created.status, StatusCode::OK, "{}", created.text);

    // Visible in A.
    let get_a = srv
        .send(
            Method::GET,
            &format!("{API}/registered-models/get?name=alpha"),
            Some("team-a"),
            None,
        )
        .await;
    assert_eq!(get_a.status, StatusCode::OK);

    // Invisible in B and default.
    for ws in ["team-b", "default"] {
        let resp = srv
            .send(
                Method::GET,
                &format!("{API}/registered-models/get?name=alpha"),
                Some(ws),
                None,
            )
            .await;
        assert_eq!(resp.status, StatusCode::NOT_FOUND, "ws {ws}: {}", resp.text);
        assert_eq!(resp.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
        assert_eq!(
            resp.json["message"],
            "Registered Model with name=alpha not found"
        );
    }
}

#[tokio::test]
async fn same_model_name_allowed_across_workspaces_and_search_scoped() {
    let srv = TestServer::enabled("rm-dup-search").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    assert_eq!(
        srv.create_registered_model(Some("team-a"), "shared")
            .await
            .status,
        StatusCode::OK
    );
    assert_eq!(
        srv.create_registered_model(Some("team-b"), "shared")
            .await
            .status,
        StatusCode::OK
    );
    assert_eq!(
        srv.create_registered_model(Some("team-a"), "only-a")
            .await
            .status,
        StatusCode::OK
    );

    let search_b = srv
        .send(
            Method::GET,
            &format!("{API}/registered-models/search?max_results=100"),
            Some("team-b"),
            None,
        )
        .await;
    assert_eq!(search_b.status, StatusCode::OK, "{}", search_b.text);
    let names_b: Vec<&str> = search_b.json["registered_models"]
        .as_array()
        .map(|a| a.iter().map(|m| m["name"].as_str().unwrap()).collect())
        .unwrap_or_default();
    assert!(names_b.contains(&"shared"));
    assert!(!names_b.contains(&"only-a"));
}

// ---------------------------------------------------------------------------
// (D) artifact-location prefixing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn artifact_location_prefixed_for_nondefault_workspace() {
    let srv = TestServer::enabled("art-nondefault").await;
    srv.create_workspace("team-a").await;
    let id = srv.create_experiment(Some("team-a"), "art-exp").await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("team-a"),
            None,
        )
        .await;
    let loc = resp.json["experiment"]["artifact_location"]
        .as_str()
        .unwrap();
    // `<server_root>/workspaces/team-a/<exp_id>`.
    assert_eq!(loc, format!("{ART_ROOT}/workspaces/team-a/{id}"));
}

#[tokio::test]
async fn artifact_location_prefixed_for_default_workspace_when_enabled() {
    let srv = TestServer::enabled("art-default").await;
    // The `default` workspace (no per-workspace root override) still gets the
    // `workspaces/default/` prefix when workspaces are enabled.
    let id = srv
        .create_experiment(Some("default"), "default-art-exp")
        .await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("default"),
            None,
        )
        .await;
    let loc = resp.json["experiment"]["artifact_location"]
        .as_str()
        .unwrap();
    assert_eq!(loc, format!("{ART_ROOT}/workspaces/default/{id}"));
}

#[tokio::test]
async fn artifact_location_uses_workspace_override_without_prefix() {
    let srv = TestServer::enabled("art-override").await;
    // Workspace with its own default_artifact_root → used verbatim, no
    // `workspaces/` prefix.
    let resp = srv
        .send(
            Method::POST,
            WORKSPACES,
            None,
            Some(json!({"name": "team-root", "default_artifact_root": "s3://team-bucket/root"})),
        )
        .await;
    assert_eq!(resp.status, StatusCode::CREATED, "{}", resp.text);

    let id = srv
        .create_experiment(Some("team-root"), "override-exp")
        .await;
    let get = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            Some("team-root"),
            None,
        )
        .await;
    let loc = get.json["experiment"]["artifact_location"]
        .as_str()
        .unwrap();
    assert_eq!(loc, format!("s3://team-bucket/root/{id}"));
    assert!(!loc.contains("/workspaces/"));
}

#[tokio::test]
async fn single_tenant_artifact_location_not_prefixed() {
    let srv = TestServer::disabled("art-single-tenant").await;
    let id = srv.create_experiment(None, "st-exp").await;
    let resp = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            None,
            None,
        )
        .await;
    let loc = resp.json["experiment"]["artifact_location"]
        .as_str()
        .unwrap();
    assert_eq!(loc, format!("{ART_ROOT}/{id}"));
    assert!(!loc.contains("/workspaces/"));
}

// ---------------------------------------------------------------------------
// (E) forbid explicit artifact_location when enabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn forbid_explicit_artifact_location_when_enabled() {
    let srv = TestServer::enabled("forbid-art").await;
    srv.create_workspace("team-a").await;
    let resp = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/create"),
            Some("team-a"),
            Some(json!({ "name": "custom", "artifact_location": "file:///tmp/custom" })),
        )
        .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(resp.json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        resp.json["message"],
        "artifact_location cannot be specified when workspaces are enabled"
    );
}

#[tokio::test]
async fn explicit_artifact_location_allowed_when_disabled() {
    let srv = TestServer::disabled("allow-art-disabled").await;
    let resp = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/create"),
            None,
            Some(json!({ "name": "custom", "artifact_location": "s3://custom/loc" })),
        )
        .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    let id = resp.json["experiment_id"].as_str().unwrap();
    let get = srv
        .send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={id}"),
            None,
            None,
        )
        .await;
    assert_eq!(
        get.json["experiment"]["artifact_location"],
        "s3://custom/loc"
    );
}
