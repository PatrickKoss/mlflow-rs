//! HTTP integration tests for the 5 workspace endpoints (plan T10.2, §3.17).
//!
//! Ports the behavioral spec in `tests/server/test_workspace_endpoints.py`
//! (Python's source of truth) to the Rust server: create (201 + shape),
//! duplicate, invalid names, get/list, update (incl. clear semantics), delete
//! (204), the `?mode=` matrix (RESTRICT / CASCADE / SET_DEFAULT, unknown mode,
//! RESTRICT-conflict, SET_DEFAULT-conflict), not-found, the reserved-`default`
//! guards, and the plain-text 503 when workspaces are disabled.
//!
//! Boots the axum app on a real ephemeral socket against a fresh copy of the
//! committed Alembic-migrated SQLite fixture (which already has the
//! `workspaces` table + a `default` row), with the [`WorkspaceStore`] wired in
//! (enabled) or omitted (disabled → 503).

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";

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
            "mlflow_rust_server_workspaces_{}_{}_{}.db",
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
    seed_pool: sqlx::SqlitePool,
    _db_file: TempDb,
}

impl TestServer {
    async fn start(tag: &str, enable_workspaces: bool, artifact_root: &str) -> Self {
        let db_file = TempDb::new(tag);
        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        // A separate sqlx pool over the same file for direct seed/count SQL —
        // the store's `Val`/`exec` helpers are crate-private.
        let seed_pool = sqlx::SqlitePool::connect(&db_file.uri())
            .await
            .expect("seed pool");
        let store = TrackingStore::new(db.clone(), artifact_root);
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
        let mut state = AppState::new(store);
        if enable_workspaces {
            state = state.with_workspace_store(WorkspaceStore::new(db.clone(), db_file.uri()));
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
            seed_pool,
            _db_file: db_file,
        }
    }

    /// Enabled server with a server-side default artifact root configured.
    async fn enabled(tag: &str) -> Self {
        Self::start(tag, true, ART_ROOT).await
    }

    /// Insert a root-table row (`experiments`) tagged with `workspace`, so the
    /// delete-mode matrix has resources to RESTRICT against / reassign / cascade.
    async fn seed_experiment(&self, experiment_id: i64, name: &str, workspace: &str) {
        sqlx::query(
            "INSERT INTO experiments \
             (experiment_id, name, artifact_location, lifecycle_stage, creation_time, \
              last_update_time, workspace) VALUES (?, ?, ?, 'active', 0, 0, ?)",
        )
        .bind(experiment_id)
        .bind(name)
        .bind(format!("s3://bucket/{experiment_id}"))
        .bind(workspace)
        .execute(&self.seed_pool)
        .await
        .expect("seed experiment");
    }

    async fn count_experiments(&self, workspace: &str) -> i64 {
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM experiments WHERE workspace = ?")
            .bind(workspace)
            .fetch_one(&self.seed_pool)
            .await
            .expect("count")
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
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    HttpResponse { status, text, json }
}

const WORKSPACES: &str = "/api/3.0/mlflow/workspaces";

// ---------------------------------------------------------------------------
// list / create / get / update
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_workspaces_returns_records_with_snake_case_shape() {
    let srv = TestServer::enabled("list").await;
    // The fixture ships a `default` row; add a second workspace.
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-a", "default_artifact_root": "s3://bucket/team-a"})),
    )
    .await;

    let resp = send(&srv.base, Method::GET, WORKSPACES, None).await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    let names: Vec<&str> = resp.json["workspaces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"default"));
    assert!(names.contains(&"team-a"));
    // A workspace with no description omits the key (proto-unset optional).
    let team_a = resp.json["workspaces"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"] == "team-a")
        .unwrap();
    assert!(team_a.get("description").is_none());
}

#[tokio::test]
async fn create_workspace_returns_201_and_full_shape() {
    let srv = TestServer::enabled("create").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({
            "name": "team-b",
            "description": "Team B",
            "trace_archival_config": {"location": "s3://archive/team-b", "retention": "30d"},
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::CREATED, "{}", resp.text);
    assert_eq!(
        resp.json,
        json!({
            "workspace": {
                "name": "team-b",
                "description": "Team B",
                "trace_archival_config": {
                    "location": "s3://archive/team-b",
                    "retention": "30d",
                },
            }
        })
    );
}

#[tokio::test]
async fn create_workspace_with_artifact_root_succeeds_without_server_default() {
    // Server started WITHOUT a default artifact root; a workspace-level root
    // makes creation valid (`_ensure_artifact_root_available`).
    let srv = TestServer::start("create-noserverroot", true, "").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-with-root", "default_artifact_root": "s3://bucket/path"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::CREATED, "{}", resp.text);
    assert_eq!(
        resp.json["workspace"]["default_artifact_root"],
        "s3://bucket/path"
    );
}

#[tokio::test]
async fn create_workspace_fails_without_any_artifact_root() {
    let srv = TestServer::start("create-noroot", true, "").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-no-root"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert!(
        resp.json["message"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("artifact root"),
        "{}",
        resp.text
    );
}

#[tokio::test]
async fn create_default_workspace_rejected() {
    let srv = TestServer::enabled("create-default").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "default"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert!(resp.json["message"].as_str().unwrap().contains("reserved"));
}

#[tokio::test]
async fn create_duplicate_workspace_conflicts() {
    let srv = TestServer::enabled("create-dup").await;
    let body = json!({"name": "team-dup", "default_artifact_root": "s3://bucket/d"});
    let first = send(&srv.base, Method::POST, WORKSPACES, Some(body.clone())).await;
    assert_eq!(first.status, StatusCode::CREATED, "{}", first.text);
    let second = send(&srv.base, Method::POST, WORKSPACES, Some(body)).await;
    // RESOURCE_ALREADY_EXISTS → HTTP 400.
    assert_eq!(second.status, StatusCode::BAD_REQUEST, "{}", second.text);
    assert_eq!(second.json["error_code"], "RESOURCE_ALREADY_EXISTS");
}

#[tokio::test]
async fn create_workspace_rejects_invalid_name() {
    let srv = TestServer::enabled("create-badname").await;
    for name in ["Team-A", "team_a", "team--a", "-team", "t"] {
        let resp = send(
            &srv.base,
            Method::POST,
            WORKSPACES,
            Some(json!({"name": name})),
        )
        .await;
        assert_eq!(
            resp.status,
            StatusCode::BAD_REQUEST,
            "name={name}: {}",
            resp.text
        );
        assert_eq!(
            resp.json["error_code"], "INVALID_PARAMETER_VALUE",
            "name={name}"
        );
    }
}

#[tokio::test]
async fn create_workspace_rejects_reserved_name() {
    let srv = TestServer::enabled("create-reserved").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "workspaces"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert!(resp.json["message"]
        .as_str()
        .unwrap()
        .contains("is reserved"));
}

#[tokio::test]
async fn create_workspace_rejects_invalid_trace_archival_retention() {
    let srv = TestServer::enabled("create-badretention").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-bad", "trace_archival_config": {"retention": "90days"}})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(
        resp.json["message"],
        "Invalid value for 'trace_archival_config.retention'. Expected a duration in the form \
         `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h')."
    );
}

#[tokio::test]
async fn create_workspace_rejects_trace_archival_retention_over_32_chars() {
    let srv = TestServer::enabled("create-longretention").await;
    let retention = format!("{}d", "1".repeat(32));
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-bad", "trace_archival_config": {"retention": retention}})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(
        resp.json["message"],
        "Invalid value for 'trace_archival_config.retention'. Maximum length is 32 characters."
    );
}

#[tokio::test]
async fn create_workspace_rejects_invalid_trace_archival_location() {
    let srv = TestServer::enabled("create-badloc").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({
            "name": "team-bad",
            "trace_archival_config": {"location": "s3://archive/team#fragment"},
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert!(
        resp.json["message"]
            .as_str()
            .unwrap()
            .contains("trace_archival_config.location"),
        "{}",
        resp.text
    );
}

#[tokio::test]
async fn create_workspace_rejects_proxy_only_trace_archival_location() {
    let srv = TestServer::enabled("create-proxyloc").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({
            "name": "team-proxy",
            "trace_archival_config": {"location": "mlflow-artifacts:/archive/team-proxy"},
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(
        resp.json["message"],
        "Invalid value for 'trace_archival_config.location'. Trace archival location cannot use \
         the proxy-only `mlflow-artifacts:` scheme."
    );
}

#[tokio::test]
async fn create_workspace_rejects_non_uri_trace_archival_location() {
    let srv = TestServer::enabled("create-nonuri").await;
    let resp = send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({
            "name": "team-local-path",
            "trace_archival_config": {"location": "archive/team-local-path"},
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(
        resp.json["message"],
        "Invalid value for 'trace_archival_config.location'. Expected a URI string."
    );
}

#[tokio::test]
async fn get_workspace_returns_metadata() {
    let srv = TestServer::enabled("get").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(
            json!({"name": "team-c", "description": "Team C", "default_artifact_root": "s3://b/c"}),
        ),
    )
    .await;
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("{WORKSPACES}/team-c"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    assert_eq!(resp.json["workspace"]["name"], "team-c");
    assert_eq!(resp.json["workspace"]["description"], "Team C");
}

#[tokio::test]
async fn get_workspace_not_found() {
    let srv = TestServer::enabled("get-404").await;
    let resp = send(
        &srv.base,
        Method::GET,
        &format!("{WORKSPACES}/team-missing"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "{}", resp.text);
    assert_eq!(resp.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn update_workspace_updates_fields() {
    let srv = TestServer::enabled("update").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-d", "description": "orig", "default_artifact_root": "s3://b/d"})),
    )
    .await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        &format!("{WORKSPACES}/team-d"),
        Some(json!({
            "description": "Updated",
            "trace_archival_config": {"location": "s3://archive/team-d", "retention": "14d"},
        })),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    assert_eq!(resp.json["workspace"]["description"], "Updated");
    assert_eq!(
        resp.json["workspace"]["trace_archival_config"],
        json!({"location": "s3://archive/team-d", "retention": "14d"})
    );
}

#[tokio::test]
async fn update_default_workspace_allows_reserved_name() {
    let srv = TestServer::enabled("update-default").await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        &format!("{WORKSPACES}/default"),
        Some(json!({"default_artifact_root": "s3://bucket/root"})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    assert_eq!(resp.json["workspace"]["name"], "default");
    assert_eq!(
        resp.json["workspace"]["default_artifact_root"],
        "s3://bucket/root"
    );
}

#[tokio::test]
async fn update_workspace_can_clear_default_artifact_root() {
    let srv = TestServer::enabled("update-clear-root").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-clear", "default_artifact_root": "s3://b/x"})),
    )
    .await;
    // Whitespace-only clears; a server default exists so it's allowed.
    let resp = send(
        &srv.base,
        Method::PATCH,
        &format!("{WORKSPACES}/team-clear"),
        Some(json!({"default_artifact_root": " "})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    assert!(
        resp.json["workspace"]
            .get("default_artifact_root")
            .is_none(),
        "{}",
        resp.text
    );
}

#[tokio::test]
async fn update_workspace_clear_artifact_root_fails_without_server_default() {
    let srv = TestServer::start("update-clear-noserver", true, "").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-clear", "default_artifact_root": "s3://b/x"})),
    )
    .await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        &format!("{WORKSPACES}/team-clear"),
        Some(json!({"default_artifact_root": ""})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert!(resp.json["message"]
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("artifact root"));
}

#[tokio::test]
async fn update_workspace_can_clear_trace_archival_location() {
    let srv = TestServer::enabled("update-clear-loc").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({
            "name": "team-clear",
            "default_artifact_root": "s3://b/x",
            "trace_archival_config": {"location": "s3://archive/x", "retention": "10d"},
        })),
    )
    .await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        &format!("{WORKSPACES}/team-clear"),
        Some(json!({"trace_archival_config": {"location": ""}})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.text);
    // Location cleared, retention retained.
    assert_eq!(
        resp.json["workspace"]["trace_archival_config"],
        json!({"retention": "10d"})
    );
}

#[tokio::test]
async fn update_workspace_rejects_invalid_trace_archival_retention() {
    let srv = TestServer::enabled("update-badretention").await;
    let resp = send(
        &srv.base,
        Method::PATCH,
        &format!("{WORKSPACES}/team-bad"),
        Some(json!({"trace_archival_config": {"retention": "5w"}})),
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(
        resp.json["message"],
        "Invalid value for 'trace_archival_config.retention'. Expected a duration in the form \
         `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h')."
    );
}

// ---------------------------------------------------------------------------
// delete + ?mode= matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_workspace_returns_204() {
    let srv = TestServer::enabled("delete").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-e", "default_artifact_root": "s3://b/e"})),
    )
    .await;
    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-e"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT, "{}", resp.text);
    assert!(resp.text.is_empty());
    // Really gone.
    let got = send(
        &srv.base,
        Method::GET,
        &format!("{WORKSPACES}/team-e"),
        None,
    )
    .await;
    assert_eq!(got.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn delete_default_workspace_rejected_by_validation() {
    let srv = TestServer::enabled("delete-default").await;
    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/default"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert!(resp.json["message"]
        .as_str()
        .unwrap()
        .contains("cannot be deleted"));
}

#[tokio::test]
async fn delete_workspace_unknown_mode_rejected() {
    let srv = TestServer::enabled("delete-badmode").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-mode", "default_artifact_root": "s3://b/m"})),
    )
    .await;
    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-mode?mode=BOGUS"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::BAD_REQUEST, "{}", resp.text);
    assert_eq!(
        resp.json["message"],
        "Invalid deletion mode 'BOGUS'. Must be one of: SET_DEFAULT, CASCADE, RESTRICT"
    );
}

#[tokio::test]
async fn delete_workspace_restrict_conflict_is_500() {
    // Default mode is RESTRICT; a workspace with a resource cannot be deleted.
    let srv = TestServer::enabled("delete-restrict").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-restrict", "default_artifact_root": "s3://b/r"})),
    )
    .await;
    srv.seed_experiment(9001, "exp-in-ws", "team-restrict")
        .await;

    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-restrict"),
        None,
    )
    .await;
    // INVALID_STATE → HTTP 500 (Python's `ERROR_CODE_TO_HTTP_STATUS`).
    assert_eq!(
        resp.status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        resp.text
    );
    assert_eq!(resp.json["error_code"], "INVALID_STATE");
    assert!(resp.json["message"]
        .as_str()
        .unwrap()
        .contains("still contains"));
    // Workspace survives.
    let got = send(
        &srv.base,
        Method::GET,
        &format!("{WORKSPACES}/team-restrict"),
        None,
    )
    .await;
    assert_eq!(got.status, StatusCode::OK);
}

#[tokio::test]
async fn delete_workspace_cascade_deletes_resources() {
    let srv = TestServer::enabled("delete-cascade").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-cascade", "default_artifact_root": "s3://b/c"})),
    )
    .await;
    srv.seed_experiment(9101, "exp-cascade", "team-cascade")
        .await;

    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-cascade?mode=CASCADE"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT, "{}", resp.text);
    assert_eq!(srv.count_experiments("team-cascade").await, 0);
}

#[tokio::test]
async fn delete_workspace_set_default_reassigns_resources() {
    let srv = TestServer::enabled("delete-setdefault").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-reassign", "default_artifact_root": "s3://b/re"})),
    )
    .await;
    srv.seed_experiment(9201, "exp-reassign", "team-reassign")
        .await;

    let before_default = srv.count_experiments("default").await;
    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-reassign?mode=SET_DEFAULT"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT, "{}", resp.text);
    assert_eq!(srv.count_experiments("team-reassign").await, 0);
    assert_eq!(srv.count_experiments("default").await, before_default + 1);
}

#[tokio::test]
async fn delete_workspace_set_default_conflict_is_500() {
    let srv = TestServer::enabled("delete-setdefault-conflict").await;
    send(
        &srv.base,
        Method::POST,
        WORKSPACES,
        Some(json!({"name": "team-conflict", "default_artifact_root": "s3://b/cf"})),
    )
    .await;
    // Same experiment name exists in BOTH default and the target workspace →
    // reassigning to default would violate the (workspace, name) uniqueness.
    srv.seed_experiment(9301, "dup-name", "default").await;
    srv.seed_experiment(9302, "dup-name", "team-conflict").await;

    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-conflict?mode=SET_DEFAULT"),
        None,
    )
    .await;
    assert_eq!(
        resp.status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        resp.text
    );
    assert_eq!(resp.json["error_code"], "INVALID_STATE");
    assert!(resp.json["message"]
        .as_str()
        .unwrap()
        .contains("already exist"));
}

#[tokio::test]
async fn delete_workspace_not_found() {
    let srv = TestServer::enabled("delete-404").await;
    let resp = send(
        &srv.base,
        Method::DELETE,
        &format!("{WORKSPACES}/team-missing"),
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "{}", resp.text);
    assert_eq!(resp.json["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

// ---------------------------------------------------------------------------
// disabled → 503 on every endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn all_endpoints_return_503_when_workspaces_disabled() {
    let srv = TestServer::start("disabled", false, ART_ROOT).await;
    let expected_body_contains =
        "disabled because the server is running without workspaces support";

    let cases: &[(Method, String, Option<Value>)] = &[
        (Method::GET, WORKSPACES.to_string(), None),
        (
            Method::POST,
            WORKSPACES.to_string(),
            Some(json!({"name": "team-x"})),
        ),
        (Method::GET, format!("{WORKSPACES}/team-x"), None),
        (
            Method::PATCH,
            format!("{WORKSPACES}/team-x"),
            Some(json!({"description": "y"})),
        ),
        (Method::DELETE, format!("{WORKSPACES}/team-x"), None),
    ];
    for (method, path, body) in cases {
        let resp = send(&srv.base, method.clone(), path, body.clone()).await;
        assert_eq!(
            resp.status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{method} {path}: {}",
            resp.text
        );
        assert!(
            resp.text.contains(expected_body_contains),
            "{method} {path}: body was {:?}",
            resp.text
        );
        assert!(
            resp.text.contains("--enable-workspaces"),
            "{method} {path}: body was {:?}",
            resp.text
        );
    }
}
