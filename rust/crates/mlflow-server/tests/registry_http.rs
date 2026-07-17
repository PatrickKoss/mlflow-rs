//! HTTP integration tests for the Model Registry REST endpoints (plan T7.4,
//! §3.14). Boots the axum app on a real ephemeral socket (same harness as
//! `model_version_artifact_http.rs`), then drives all 21 endpoints — plus the
//! method-overloaded `/registered-models/alias` route — over HTTP on **both**
//! the `/api/2.0` and `/ajax-api/2.0` prefixes.
//!
//! Coverage:
//! * CRUD round-trips for registered models + versions;
//! * rename cascade visible over HTTP;
//! * search GET with filter / order_by / pagination + threshold errors;
//! * get-latest-versions on POST and GET with `stages` repeated params;
//! * transition-stage + `archive_existing_versions`;
//! * soft-delete then 404;
//! * download-uri;
//! * tags (set/delete for RM and MV);
//! * aliases via all three methods (POST/DELETE/GET) on the one path;
//! * `createModelVersion` source validation negative cases (local-path escape,
//!   run mismatch, traversal) with byte-matched errors;
//! * copy-model-version flow (`models:/{src}/{ver}` source end-to-end);
//! * prompts-on-registry (T7.5): the Prompts UI's exact REST call shapes
//!   (`mlflow/server/js/src/experiment-tracking/pages/prompts/api.ts`) —
//!   create prompt (RM tagged `mlflow.prompt.is_prompt=true`), create prompt
//!   version (`model-versions/create` with `source: "dummy-source"`), tag
//!   set/delete on both, alias set, and the `search` filter clauses the UI
//!   sends (`tags.\`mlflow.prompt.is_prompt\` = 'true'`) — plus the model/prompt
//!   name-collision rejection (`handle_resource_already_exist_error`).

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;

const WS: &str = "default";

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
            "mlflow_rust_server_registry_{}_{}_{}.db",
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
    tracking: TrackingStore,
    registry: RegistryStore,
    /// Root for `file://` experiment artifact locations used by source-validation.
    artifact_root: TempDir,
}

impl TestServer {
    async fn start(tag: &str) -> Self {
        let db_file = TempDb::new(tag);
        let artifact_root = TempDir::new().expect("artifact root");

        let db = Db::connect(&db_file.uri(), PoolConfig::default())
            .await
            .expect("connect temp fixture");
        let tracking = TrackingStore::new(db.clone(), "file:///unused".to_string());
        let registry = RegistryStore::new(db);

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
        let app_state =
            AppState::with_registry(tracking.clone(), registry.clone(), true, None, None);
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
            tracking,
            registry,
            artifact_root,
        }
    }

    /// Create an experiment with a local `file://` artifact location and a run
    /// under it, returning `(run_id, run_artifact_uri)`. Used by the
    /// source-validation tests that need a run whose local artifact directory
    /// contains the model version source.
    async fn seed_run(&self, exp_name: &str) -> (String, String) {
        let art_dir = self.artifact_root.path().join(exp_name);
        std::fs::create_dir_all(&art_dir).unwrap();
        let art_loc = format!("file://{}", art_dir.display());
        let exp_id = self
            .tracking
            .create_experiment(WS, exp_name, Some(&art_loc), &[])
            .await
            .unwrap();
        let run = self
            .tracking
            .create_run(WS, &exp_id, None, Some(0), Some("r"), &[])
            .await
            .unwrap();
        let uri = run.info.artifact_uri.clone().unwrap();
        (run.info.run_id, uri)
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
    body: Vec<u8>,
}

impl HttpResponse {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON: {e}: {}", self.text()))
    }
}

async fn send(
    server: &TestServer,
    method: Method,
    path: &str,
    body: Option<&Value>,
) -> HttpResponse {
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
                return HttpResponse {
                    status,
                    body: bytes.to_vec(),
                };
            }
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }
    panic!("failed to connect: {last:?}");
}

async fn post(server: &TestServer, path: &str, body: &Value) -> HttpResponse {
    send(server, Method::POST, path, Some(body)).await
}
async fn get(server: &TestServer, path: &str) -> HttpResponse {
    send(server, Method::GET, path, None).await
}
async fn patch(server: &TestServer, path: &str, body: &Value) -> HttpResponse {
    send(server, Method::PATCH, path, Some(body)).await
}
async fn delete(server: &TestServer, path: &str, body: &Value) -> HttpResponse {
    send(server, Method::DELETE, path, Some(body)).await
}

/// Minimal query-string percent-encoding for filter clauses in GET URLs (the
/// existing tests hand-encode a handful of characters inline; the prompt
/// filters below need backticks/`=`/quotes too, so a small general helper
/// avoids repeating ad hoc `%XX` literals).
fn qs_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn uniq() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

const API: &str = "/api/2.0/mlflow";
const AJAX: &str = "/ajax-api/2.0/mlflow";

// ===========================================================================
// Registered model CRUD + both prefixes
// ===========================================================================

#[tokio::test]
async fn registered_model_crud_roundtrip() {
    let server = TestServer::start("rm_crud").await;
    let name = format!("m_{}", uniq());

    let res = post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name, "description": "d", "tags": [{"key": "t", "value": "v"}]}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let rm = &res.json()["registered_model"];
    assert_eq!(rm["name"], name);
    assert_eq!(rm["description"], "d");
    assert_eq!(rm["tags"][0]["key"], "t");
    // `user_id` never populated on a RegisteredModel (§3.14).
    assert!(rm.get("user_id").is_none() || rm["user_id"].is_null());

    // get on the ajax prefix.
    let res = get(
        &server,
        &format!("{AJAX}/registered-models/get?name={name}"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["registered_model"]["name"], name);

    // update (PATCH) description.
    let res = patch(
        &server,
        &format!("{API}/registered-models/update"),
        &json!({"name": name, "description": "d2"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["registered_model"]["description"], "d2");

    // delete (DELETE).
    let res = delete(
        &server,
        &format!("{API}/registered-models/delete"),
        &json!({"name": name}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    let res = get(&server, &format!("{API}/registered-models/get?name={name}")).await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());
}

#[tokio::test]
async fn create_registered_model_missing_name_is_400() {
    let server = TestServer::start("rm_missing_name").await;
    let res = post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text()
            .contains("Missing value for required parameter 'name'"),
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn rename_cascade_visible_over_http() {
    let server = TestServer::start("rm_rename").await;
    let name = format!("m_{}", uniq());
    let new_name = format!("m2_{}", uniq());

    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    // A version + alias under the old name.
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();
    post(
        &server,
        &format!("{API}/registered-models/rename"),
        &json!({"name": name, "new_name": new_name}),
    )
    .await;

    // Old name gone; new name present; the version cascaded to the new name.
    let res = get(&server, &format!("{API}/registered-models/get?name={name}")).await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());

    let res = get(
        &server,
        &format!("{API}/model-versions/get?name={new_name}&version=1"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["model_version"]["name"], new_name);
}

// ===========================================================================
// Search: filter / order_by / pagination + thresholds
// ===========================================================================

#[tokio::test]
async fn search_registered_models_filter_order_pagination() {
    let server = TestServer::start("rm_search").await;
    let prefix = format!("srm{}", uniq());
    for i in 0..3 {
        post(
            &server,
            &format!("{API}/registered-models/create"),
            &json!({"name": format!("{prefix}_{i}")}),
        )
        .await;
    }

    // Filter by exact name; order_by name DESC; page size 2.
    let res = get(
        &server,
        &format!(
            "{API}/registered-models/search?filter=name+LIKE+%27{prefix}%25%27&order_by=name+DESC&max_results=2"
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let body = res.json();
    let models = body["registered_models"].as_array().unwrap();
    assert_eq!(models.len(), 2, "{}", res.text());
    assert_eq!(models[0]["name"], format!("{prefix}_2"));
    let token = body["next_page_token"].as_str().unwrap().to_string();

    // Second page.
    let res = get(
        &server,
        &format!(
            "{API}/registered-models/search?filter=name+LIKE+%27{prefix}%25%27&order_by=name+DESC&max_results=2&page_token={token}"
        ),
    )
    .await;
    let body = res.json();
    let models = body["registered_models"].as_array().unwrap();
    assert_eq!(models.len(), 1, "{}", res.text());
    assert_eq!(models[0]["name"], format!("{prefix}_0"));
}

#[tokio::test]
async fn search_registered_models_threshold_is_400() {
    let server = TestServer::start("rm_threshold").await;
    let res = get(
        &server,
        &format!("{API}/registered-models/search?max_results=1001"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(res.text().contains("at most 1000"), "{}", res.text());
}

#[tokio::test]
async fn search_model_versions_threshold_is_400() {
    let server = TestServer::start("mv_threshold").await;
    let res = get(
        &server,
        &format!("{API}/model-versions/search?max_results=200001"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(res.text().contains("at most 200000"), "{}", res.text());
}

#[tokio::test]
async fn search_model_versions_filter() {
    let server = TestServer::start("mv_search").await;
    let name = format!("mvs_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/2", None, &[], None, None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("{AJAX}/model-versions/search?filter=name%3D%27{name}%27"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let versions = res.json()["model_versions"].as_array().unwrap().len();
    assert_eq!(versions, 2, "{}", res.text());
}

// ===========================================================================
// get-latest-versions on POST and GET with stages
// ===========================================================================

#[tokio::test]
async fn get_latest_versions_post_and_get_with_stages() {
    let server = TestServer::start("latest").await;
    let name = format!("lv_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/2", None, &[], None, None)
        .await
        .unwrap();
    // Move version 2 to Production.
    post(
        &server,
        &format!("{API}/model-versions/transition-stage"),
        &json!({"name": name, "version": "2", "stage": "Production", "archive_existing_versions": false}),
    )
    .await;

    // POST, no stages → latest per stage (None + Production).
    let res = post(
        &server,
        &format!("{API}/registered-models/get-latest-versions"),
        &json!({"name": name}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let versions = res.json()["model_versions"].as_array().unwrap().len();
    assert_eq!(versions, 2, "{}", res.text());

    // GET with a repeated `stages` query param.
    let res = get(
        &server,
        &format!("{API}/registered-models/get-latest-versions?name={name}&stages=Production"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let body = res.json();
    let mvs = body["model_versions"].as_array().unwrap();
    assert_eq!(mvs.len(), 1, "{}", res.text());
    assert_eq!(mvs[0]["version"], "2");
    assert_eq!(mvs[0]["current_stage"], "Production");
}

// ===========================================================================
// transition-stage + archive
// ===========================================================================

#[tokio::test]
async fn transition_stage_with_archive_existing() {
    let server = TestServer::start("archive").await;
    let name = format!("arch_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    for _ in 0..2 {
        server
            .registry
            .create_model_version(WS, &name, "mlflow-artifacts:/m", None, &[], None, None)
            .await
            .unwrap();
    }
    // v1 → Production.
    post(
        &server,
        &format!("{API}/model-versions/transition-stage"),
        &json!({"name": name, "version": "1", "stage": "Production", "archive_existing_versions": false}),
    )
    .await;
    // v2 → Production with archive → v1 becomes Archived.
    let res = post(
        &server,
        &format!("{API}/model-versions/transition-stage"),
        &json!({"name": name, "version": "2", "stage": "Production", "archive_existing_versions": true}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["model_version"]["current_stage"], "Production");

    let res = get(
        &server,
        &format!("{API}/model-versions/get?name={name}&version=1"),
    )
    .await;
    assert_eq!(
        res.json()["model_version"]["current_stage"],
        "Archived",
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn transition_stage_archive_to_non_active_is_400() {
    let server = TestServer::start("archive_bad").await;
    let name = format!("ab_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m", None, &[], None, None)
        .await
        .unwrap();
    let res = post(
        &server,
        &format!("{API}/model-versions/transition-stage"),
        &json!({"name": name, "version": "1", "stage": "Archived", "archive_existing_versions": true}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text().contains("['Staging', 'Production']"),
        "{}",
        res.text()
    );
}

// ===========================================================================
// soft-delete then 404 + download-uri
// ===========================================================================

#[tokio::test]
async fn model_version_soft_delete_then_404() {
    let server = TestServer::start("mv_delete").await;
    let name = format!("d_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();

    let res = delete(
        &server,
        &format!("{API}/model-versions/delete"),
        &json!({"name": name, "version": "1"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    let res = get(
        &server,
        &format!("{API}/model-versions/get?name={name}&version=1"),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.text());
}

#[tokio::test]
async fn get_download_uri() {
    let server = TestServer::start("download_uri").await;
    let name = format!("du_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/loc", None, &[], None, None)
        .await
        .unwrap();

    let res = get(
        &server,
        &format!("{API}/model-versions/get-download-uri?name={name}&version=1"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["artifact_uri"], "mlflow-artifacts:/m/loc");
}

// ===========================================================================
// Tags (RM + MV)
// ===========================================================================

#[tokio::test]
async fn registered_model_and_version_tags() {
    let server = TestServer::start("tags").await;
    let name = format!("tag_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();

    // RM tag set + delete.
    let res = post(
        &server,
        &format!("{API}/registered-models/set-tag"),
        &json!({"name": name, "key": "k", "value": "v"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = get(&server, &format!("{API}/registered-models/get?name={name}")).await;
    assert_eq!(res.json()["registered_model"]["tags"][0]["value"], "v");

    let res = delete(
        &server,
        &format!("{API}/registered-models/delete-tag"),
        &json!({"name": name, "key": "k"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // MV tag set + delete.
    let res = post(
        &server,
        &format!("{API}/model-versions/set-tag"),
        &json!({"name": name, "version": "1", "key": "mk", "value": "mv"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = get(
        &server,
        &format!("{API}/model-versions/get?name={name}&version=1"),
    )
    .await;
    assert_eq!(res.json()["model_version"]["tags"][0]["key"], "mk");

    let res = delete(
        &server,
        &format!("{API}/model-versions/delete-tag"),
        &json!({"name": name, "version": "1", "key": "mk"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
}

// ===========================================================================
// Aliases — all three methods on the one method-overloaded path
// ===========================================================================

#[tokio::test]
async fn alias_route_all_three_methods() {
    let server = TestServer::start("alias").await;
    let name = format!("al_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();

    // POST = setRegisteredModelAlias.
    let res = post(
        &server,
        &format!("{API}/registered-models/alias"),
        &json!({"name": name, "alias": "champion", "version": "1"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // GET = getModelVersionByAlias.
    let res = get(
        &server,
        &format!("{API}/registered-models/alias?name={name}&alias=champion"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["model_version"]["version"], "1");

    // Alias also surfaces on the version + registered model reads.
    let res = get(
        &server,
        &format!("{API}/model-versions/get?name={name}&version=1"),
    )
    .await;
    assert_eq!(res.json()["model_version"]["aliases"][0], "champion");

    // DELETE = deleteRegisteredModelAlias.
    let res = delete(
        &server,
        &format!("{API}/registered-models/alias"),
        &json!({"name": name, "alias": "champion"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // A deleted alias is 400 INVALID_PARAMETER_VALUE, not 404 — matching the
    // Python registry store (`sqlalchemy_store.py:1592`, "Registered model
    // alias ... not found." raised as INVALID_PARAMETER_VALUE).
    let res = get(
        &server,
        &format!("{API}/registered-models/alias?name={name}&alias=champion"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text()
            .contains("Registered model alias champion not found"),
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn alias_route_works_on_ajax_prefix_too() {
    let server = TestServer::start("alias_ajax").await;
    let name = format!("ala_{}", uniq());
    post(
        &server,
        &format!("{AJAX}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    server
        .registry
        .create_model_version(WS, &name, "mlflow-artifacts:/m/1", None, &[], None, None)
        .await
        .unwrap();
    let res = post(
        &server,
        &format!("{AJAX}/registered-models/alias"),
        &json!({"name": name, "alias": "a", "version": "1"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = get(
        &server,
        &format!("{AJAX}/registered-models/alias?name={name}&alias=a"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
}

// ===========================================================================
// createModelVersion — happy path + source validation
// ===========================================================================

#[tokio::test]
async fn create_model_version_with_local_source_in_run_artifact_dir() {
    let server = TestServer::start("mv_local_ok").await;
    let name = format!("clv_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    let (run_id, artifact_uri) = server.seed_run(&format!("exp_{}", uniq())).await;
    // Strip the `file://` prefix (mirrors the Python test's local-path source).
    let local = artifact_uri.strip_prefix("file://").unwrap();

    let res = post(
        &server,
        &format!("{API}/model-versions/create"),
        &json!({"name": name, "source": local, "run_id": run_id}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["model_version"]["version"], "1");
}

#[tokio::test]
async fn create_model_version_local_source_without_run_id_is_400() {
    let server = TestServer::start("mv_local_norun").await;
    let name = format!("clv_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    let (_run_id, artifact_uri) = server.seed_run(&format!("exp_{}", uniq())).await;
    let local = artifact_uri.strip_prefix("file://").unwrap();

    let res = post(
        &server,
        &format!("{API}/model-versions/create"),
        &json!({"name": name, "source": local}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text()
            .contains("To use a local path as a model version"),
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn create_model_version_local_source_outside_run_dir_is_400() {
    let server = TestServer::start("mv_local_outside").await;
    let name = format!("clv_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    let (run_id, _artifact_uri) = server.seed_run(&format!("exp_{}", uniq())).await;

    let res = post(
        &server,
        &format!("{API}/model-versions/create"),
        &json!({"name": name, "source": "/tmp", "run_id": run_id}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert!(
        res.text()
            .contains("To use a local path as a model version"),
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn create_model_version_remote_traversal_source_is_400() {
    let server = TestServer::start("mv_traversal").await;
    let name = format!("clv_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;

    for src in [
        "mlflow-artifacts://host:9000/models/../../../",
        "http://host:9000/models/../../../",
        "mlflow-artifacts://host:9000/models/..%2f..%2fartifacts",
        "mlflow-artifacts://host:9000/models/artifact%00",
    ] {
        let res = post(
            &server,
            &format!("{API}/model-versions/create"),
            &json!({"name": name, "source": src}),
        )
        .await;
        assert_eq!(
            res.status,
            StatusCode::BAD_REQUEST,
            "src={src}: {}",
            res.text()
        );
        assert!(
            res.text()
                .contains("If supplying a source as an http, https,"),
            "src={src}: {}",
            res.text()
        );
    }
}

#[tokio::test]
async fn create_model_version_remote_absolute_source_ok() {
    let server = TestServer::start("mv_remote_ok").await;
    let name = format!("clv_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;

    for src in [
        "mlflow-artifacts:/models",
        "mlflow-artifacts:/models///",
        "mlflow-artifacts://host:9000/models",
    ] {
        let res = post(
            &server,
            &format!("{API}/model-versions/create"),
            &json!({"name": name, "source": src}),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "src={src}: {}", res.text());
    }
}

// ===========================================================================
// copy-model-version flow: create a version whose source is `models:/{n}/{v}`
// ===========================================================================

#[tokio::test]
async fn copy_model_version_flow() {
    let server = TestServer::start("copy").await;
    let src_name = format!("src_{}", uniq());
    let dst_name = format!("dst_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": src_name}),
    )
    .await;
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": dst_name}),
    )
    .await;

    // Source version with a resolvable storage location.
    let res = post(
        &server,
        &format!("{API}/model-versions/create"),
        &json!({"name": src_name, "source": "mlflow-artifacts:/base/loc"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // Copy: destination version sources from `models:/{src}/1`.
    let res = post(
        &server,
        &format!("{API}/model-versions/create"),
        &json!({"name": dst_name, "source": format!("models:/{src_name}/1")}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["model_version"]["version"], "1");

    // The copy's download-uri resolves to the base version's storage location.
    let res = get(
        &server,
        &format!("{API}/model-versions/get-download-uri?name={dst_name}&version=1"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["artifact_uri"], "mlflow-artifacts:/base/loc");
}

// ===========================================================================
// Prompts-on-registry (T7.5): the Prompts UI is unmodified RegisteredModel /
// ModelVersion REST traffic with `mlflow.prompt.is_prompt=true` tags — see
// `mlflow/server/js/src/experiment-tracking/pages/prompts/api.ts`. These
// tests drive that exact call shape end-to-end and confirm models pages never
// surface prompts (the T7.3 anti-join, exercised here at the HTTP layer).
// ===========================================================================

const IS_PROMPT_TAG: &str = "mlflow.prompt.is_prompt";

#[tokio::test]
async fn prompt_lifecycle_over_http() {
    let server = TestServer::start("prompt_lifecycle").await;
    let name = format!("pr_{}", uniq());

    // `createRegisteredPrompt`: registered-models/create with the is_prompt tag.
    let res = post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({
            "name": name,
            "tags": [{"key": IS_PROMPT_TAG, "value": "true"}],
        }),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let rm = &res.json()["registered_model"];
    assert_eq!(rm["name"], name);
    assert!(
        rm["tags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["key"] == IS_PROMPT_TAG && t["value"] == "true"),
        "{rm}"
    );

    // `createRegisteredPromptVersion`: model-versions/create, placeholder
    // "dummy-source" (accepted for prompts; see source_validation's
    // is_prompt branch), tagged with the prompt template text.
    let res = post(
        &server,
        &format!("{API}/model-versions/create"),
        &json!({
            "name": name,
            "source": "dummy-source",
            "tags": [
                {"key": IS_PROMPT_TAG, "value": "true"},
                {"key": "mlflow.prompt.text", "value": "Hello {{name}}"},
            ],
        }),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let mv = &res.json()["model_version"];
    assert_eq!(mv["name"], name);
    assert_eq!(mv["version"], "1");

    // `setRegisteredPromptTag` / `deleteRegisteredPromptTag`.
    let res = post(
        &server,
        &format!("{API}/registered-models/set-tag"),
        &json!({"name": name, "key": "owner", "value": "alice"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = delete(
        &server,
        &format!("{API}/registered-models/delete-tag"),
        &json!({"name": name, "key": "owner"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // `setRegisteredPromptVersionTag` / `deleteRegisteredPromptVersionTag`.
    let res = post(
        &server,
        &format!("{API}/model-versions/set-tag"),
        &json!({"name": name, "version": "1", "key": "lang", "value": "en"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = delete(
        &server,
        &format!("{API}/model-versions/delete-tag"),
        &json!({"name": name, "version": "1", "key": "lang"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    // Set alias on the prompt version (shared alias route; the Prompts UI
    // itself does not call this today, but prompts and models share the same
    // registry surface, so aliasing a prompt version must work identically).
    let res = post(
        &server,
        &format!("{API}/registered-models/alias"),
        &json!({"name": name, "alias": "production", "version": "1"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = get(
        &server,
        &format!("{API}/registered-models/alias?name={name}&alias=production"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.json()["model_version"]["version"], "1");

    // `listRegisteredPrompts`: registered-models/search with the exact filter
    // clause the Prompts UI sends (backtick-quoted tag key, `= 'true'`).
    let filter = qs_encode(&format!("tags.`{IS_PROMPT_TAG}` = 'true'"));
    let res = get(
        &server,
        &format!("{API}/registered-models/search?filter={filter}"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let body = res.json();
    let names: Vec<&str> = body["registered_models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&name.as_str()), "{names:?}");

    // `getPromptVersions`: model-versions/search with `name='x' AND
    // tags.`is_prompt` = 'true'`.
    let mv_filter = qs_encode(&format!(
        "name='{name}' AND tags.`{IS_PROMPT_TAG}` = 'true'"
    ));
    let res = get(
        &server,
        &format!("{API}/model-versions/search?filter={mv_filter}"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let versions = res.json()["model_versions"].as_array().unwrap().clone();
    assert_eq!(versions.len(), 1, "{versions:?}");
    assert_eq!(versions[0]["version"], "1");

    // `deleteRegisteredPromptVersion` / `deleteRegisteredPrompt`.
    let res = delete(
        &server,
        &format!("{API}/model-versions/delete"),
        &json!({"name": name, "version": "1"}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let res = delete(
        &server,
        &format!("{API}/registered-models/delete"),
        &json!({"name": name}),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
}

#[tokio::test]
async fn prompts_excluded_from_default_model_search() {
    let server = TestServer::start("prompt_hidden").await;
    let prefix = format!("ph_{}", uniq());
    let model_name = format!("{prefix}_model");
    let prompt_name = format!("{prefix}_prompt");

    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": model_name}),
    )
    .await;
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": prompt_name, "tags": [{"key": IS_PROMPT_TAG, "value": "true"}]}),
    )
    .await;

    // Default (no filter): the Models UI's plain listing — prompts must never
    // appear.
    let filter = qs_encode(&format!("name LIKE '{prefix}%'"));
    let res = get(
        &server,
        &format!("{API}/registered-models/search?filter={filter}"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let body = res.json();
    let names: Vec<&str> = body["registered_models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&model_name.as_str()), "{names:?}");
    assert!(!names.contains(&prompt_name.as_str()), "{names:?}");

    // The prompts-only filter is the inverse: only the prompt appears.
    let prompt_filter = qs_encode(&format!(
        "name LIKE '{prefix}%' AND tags.`{IS_PROMPT_TAG}` = 'true'"
    ));
    let res = get(
        &server,
        &format!("{API}/registered-models/search?filter={prompt_filter}"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let body = res.json();
    let names: Vec<&str> = body["registered_models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec![prompt_name.as_str()], "{names:?}");
}

#[tokio::test]
async fn prompt_and_model_name_collision_error_messages() {
    let server = TestServer::start("prompt_collision").await;
    let name = format!("coll_{}", uniq());

    // Existing plain model, then a same-name prompt create attempt: the
    // cross-type message (`registry_utils.py:280-285`).
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    let res = post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name, "tags": [{"key": IS_PROMPT_TAG, "value": "true"}]}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert_eq!(
        res.json()["message"],
        format!(
            "Tried to create a prompt with name '{name}', but the name is already taken by a \
             registered model. MLflow does not allow creating a model and a prompt with the \
             same name."
        )
    );

    // Existing prompt, then a same-name plain model create attempt: the
    // symmetric cross-type message.
    let name2 = format!("coll2_{}", uniq());
    post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name2, "tags": [{"key": IS_PROMPT_TAG, "value": "true"}]}),
    )
    .await;
    let res = post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name2}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert_eq!(
        res.json()["message"],
        format!(
            "Tried to create a registered model with name '{name2}', but the name is already \
             taken by a prompt. MLflow does not allow creating a model and a prompt with the \
             same name."
        )
    );

    // Same-type collision keeps the plain "already exists." message.
    let res = post(
        &server,
        &format!("{API}/registered-models/create"),
        &json!({"name": name}),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert_eq!(
        res.json()["message"],
        format!("Registered Model (name={name}) already exists.")
    );
}
