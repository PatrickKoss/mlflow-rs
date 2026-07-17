//! HTTP integration tests for the logged-model + search-datasets endpoints
//! (plan T3.4, §3.4/§3.5).
//!
//! Boots the axum app on a real ephemeral socket (same pattern as
//! `experiments_http.rs`) against a fresh copy of the committed
//! Alembic-migrated SQLite fixture, then drives every endpoint over HTTP,
//! including the path-parameter mechanism (`{model_id}`, `{tag_key}`),
//! finalize (PATCH), search pagination with the encoded token, tag
//! set/delete, and the search-datasets route quirks (missing-leading-slash +
//! the hand-registered ajax route).

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use serde_json::Value;
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
/// Experiment "0" ("Default") from the committed fixture.
const EXP_ID: &str = "0";
/// Matches the `Workspace` extractor's fallback when no
/// `X-MLFLOW-WORKSPACE` header is sent (`workspace::DEFAULT_WORKSPACE_NAME`).
const WORKSPACE: &str = "default";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

/// Copy the committed fixture to a unique temp file; removed on drop.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_server_logged_models_{}_{}_{}.db",
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

/// A running test server with a base URL; shuts down on drop.
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
        let store = TrackingStore::new(db, ART_ROOT);
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
        let app = build_app_with_recorder(&config, recorder, Some(AppState::new(store)));

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
    body: String,
}

impl HttpResponse {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON: {e}: {}", self.body))
    }
}

async fn send(server: &TestServer, method: Method, path: &str, body: Option<&str>) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{path}", server.base);
    let mut builder = Request::builder().method(method).uri(&url);
    let request = match body {
        Some(b) => {
            builder = builder.header("content-type", "application/json");
            builder.body(Full::<Bytes>::new(Bytes::from(b.to_string())))
        }
        None => builder.body(Full::<Bytes>::new(Bytes::new())),
    }
    .unwrap();

    let mut last = None;
    for _ in 0..50 {
        match client.request(clone_request(&request)).await {
            Ok(res) => {
                let status = res.status();
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return HttpResponse {
                    status,
                    body: String::from_utf8_lossy(&bytes).into_owned(),
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

fn clone_request(req: &Request<Full<Bytes>>) -> Request<Full<Bytes>> {
    let mut builder = Request::builder()
        .method(req.method().clone())
        .uri(req.uri().clone());
    for (k, v) in req.headers() {
        builder = builder.header(k, v);
    }
    let body = req.body().clone();
    builder.body(body).unwrap()
}

const PREFIXES: [&str; 2] = ["/api/2.0", "/ajax-api/2.0"];

async fn post(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    send(
        server,
        Method::POST,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

async fn get(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::GET, &format!("{prefix}{endpoint}"), None).await
}

async fn patch(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    send(
        server,
        Method::PATCH,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

async fn delete(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::DELETE, &format!("{prefix}{endpoint}"), None).await
}

async fn create_model(server: &TestServer, prefix: &str, name: &str) -> String {
    let res = post(
        server,
        prefix,
        "/mlflow/logged-models",
        &format!(r#"{{"experiment_id": "{EXP_ID}", "name": "{name}"}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    res.json()["model"]["info"]["model_id"]
        .as_str()
        .unwrap()
        .to_string()
}

// ---------------------------------------------------------------------------
// Logged models: CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_logged_model_on_both_prefixes() {
    let server = TestServer::start("crud").await;

    for (i, prefix) in PREFIXES.iter().enumerate() {
        let name = format!("model_{i}");
        let create = post(
            &server,
            prefix,
            "/mlflow/logged-models",
            &format!(
                r#"{{"experiment_id": "{EXP_ID}", "name": "{name}", "model_type": "LLM",
                    "params": [{{"key": "alpha", "value": "0.5"}}],
                    "tags": [{{"key": "team", "value": "rust"}}]}}"#
            ),
        )
        .await;
        assert_eq!(create.status, StatusCode::OK, "{}", create.body);
        let created = create.json();
        let info = &created["model"]["info"];
        assert_eq!(info["name"], name);
        assert_eq!(info["experiment_id"], EXP_ID);
        assert_eq!(info["model_type"], "LLM");
        assert_eq!(info["status"], "LOGGED_MODEL_PENDING");
        let model_id = info["model_id"].as_str().unwrap().to_string();
        assert!(model_id.starts_with("m-"));
        assert_eq!(created["model"]["data"]["params"][0]["key"], "alpha");
        assert_eq!(info["tags"][0]["key"], "team");

        // GET via path param.
        let got = get(
            &server,
            prefix,
            &format!("/mlflow/logged-models/{model_id}"),
        )
        .await;
        assert_eq!(got.status, StatusCode::OK, "{}", got.body);
        assert_eq!(got.json()["model"]["info"]["model_id"], model_id);
    }
}

#[tokio::test]
async fn create_logged_model_generates_name_when_absent() {
    let server = TestServer::start("gen_name").await;
    let create = post(
        &server,
        "/api/2.0",
        "/mlflow/logged-models",
        &format!(r#"{{"experiment_id": "{EXP_ID}"}}"#),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);
    let name = create.json()["model"]["info"]["name"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(!name.is_empty());
}

#[tokio::test]
async fn get_missing_logged_model_is_resource_does_not_exist() {
    let server = TestServer::start("get_missing").await;
    let res = get(&server, "/api/2.0", "/mlflow/logged-models/m-doesnotexist").await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn create_missing_experiment_id_is_invalid_parameter_value() {
    let server = TestServer::start("create_missing_exp").await;
    let res = post(&server, "/api/2.0", "/mlflow/logged-models", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        json["message"],
        "Missing value for required parameter 'experiment_id'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn delete_logged_model_then_get_allow_deleted() {
    let server = TestServer::start("delete").await;
    let model_id = create_model(&server, "/api/2.0", "to_delete").await;

    let del = delete(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);
    assert_eq!(del.json(), serde_json::json!({}));

    // Default (`allow_deleted` absent/false) → not found.
    let not_found = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    assert_eq!(
        not_found.status,
        StatusCode::NOT_FOUND,
        "{}",
        not_found.body
    );

    // `allow_deleted=true` → still resolvable.
    let found = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}?allow_deleted=true"),
    )
    .await;
    assert_eq!(found.status, StatusCode::OK, "{}", found.body);
    assert_eq!(found.json()["model"]["info"]["model_id"], model_id);
}

// ---------------------------------------------------------------------------
// Finalize (PATCH)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_logged_model_updates_status() {
    let server = TestServer::start("finalize").await;
    let model_id = create_model(&server, "/api/2.0", "to_finalize").await;

    // Exact Python parity (T12.5): `_finalize_logged_model` requires `model_id`
    // in the BODY (its schema) and uses that body value for the store call, so
    // the client sends it in both the URL and the JSON.
    let finalize = patch(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
        &format!(r#"{{"model_id": "{model_id}", "status": "LOGGED_MODEL_READY"}}"#),
    )
    .await;
    assert_eq!(finalize.status, StatusCode::OK, "{}", finalize.body);
    assert_eq!(
        finalize.json()["model"]["info"]["status"],
        "LOGGED_MODEL_READY"
    );

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    assert_eq!(got.json()["model"]["info"]["status"], "LOGGED_MODEL_READY");
}

#[tokio::test]
async fn finalize_uses_body_model_id_not_path() {
    // Exact Python parity (T12.5): `_finalize_logged_model` reads
    // `request_message.model_id` — the BODY value — for the store call, not the
    // URL segment. So a body `model_id` that differs from the path is what the
    // store sees; here it names a nonexistent model, which Python surfaces as a
    // RESOURCE_DOES_NOT_EXIST (404), proving the body value drives the lookup.
    let server = TestServer::start("finalize_body_model_id").await;
    let model_id = create_model(&server, "/api/2.0", "body_wins").await;

    let finalize = patch(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
        r#"{"model_id": "m-doesnotexist", "status": "LOGGED_MODEL_READY"}"#,
    )
    .await;
    assert_eq!(finalize.status, StatusCode::NOT_FOUND, "{}", finalize.body);
    assert_eq!(finalize.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn finalize_empty_body_missing_required_model_id() {
    // With an empty body, `_finalize_logged_model`'s schema
    // (`model_id` then `status`, both required) fails on `model_id` first —
    // exact byte-parity with Python's `_assert_required` message (T12.5).
    let server = TestServer::start("finalize_empty_body").await;
    let model_id = create_model(&server, "/api/2.0", "no_status").await;

    let res = patch(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
        "{}",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'model_id'. \
         See the API docs for more information about request parameters."
    );
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_logged_model_params_via_path_param() {
    let server = TestServer::start("log_params").await;
    let model_id = create_model(&server, "/api/2.0", "params_model").await;

    // Exact Python parity (T12.5): `_log_logged_model_params` requires
    // `model_id` in the BODY (schema), then uses the URL path arg for the store
    // call. The client sends the id in both places.
    let res = post(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}/params"),
        &format!(r#"{{"model_id": "{model_id}", "params": [{{"key": "beta", "value": "1.0"}}]}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json(), serde_json::json!({}));

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    let params = got.json()["model"]["data"]["params"].clone();
    assert!(params
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p["key"] == "beta" && p["value"] == "1.0"));
}

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_and_delete_logged_model_tag_roundtrip() {
    let server = TestServer::start("tags").await;
    let model_id = create_model(&server, "/api/2.0", "tag_model").await;

    let set = patch(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}/tags"),
        r#"{"tags": [{"key": "k", "value": "v1"}]}"#,
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "{}", set.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    let tags = got.json()["model"]["info"]["tags"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["key"], "k");
    assert_eq!(tags[0]["value"], "v1");

    // Upsert.
    patch(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}/tags"),
        r#"{"tags": [{"key": "k", "value": "v2"}]}"#,
    )
    .await;
    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    assert_eq!(got.json()["model"]["info"]["tags"][0]["value"], "v2");

    // Delete via two path params (model_id + tag_key).
    let del = delete(
        &server,
        "/ajax-api/2.0",
        &format!("/mlflow/logged-models/{model_id}/tags/k"),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}"),
    )
    .await;
    assert_eq!(
        got.json()["model"]["info"]
            .get("tags")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0),
        0
    );
}

#[tokio::test]
async fn delete_missing_tag_is_resource_does_not_exist() {
    let server = TestServer::start("tag_missing").await;
    let model_id = create_model(&server, "/api/2.0", "tag_missing_model").await;

    let res = delete(
        &server,
        "/api/2.0",
        &format!("/mlflow/logged-models/{model_id}/tags/nope"),
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

// ---------------------------------------------------------------------------
// Search + pagination
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_logged_models_finds_created_model() {
    let server = TestServer::start("search").await;
    create_model(&server, "/api/2.0", "search_me").await;

    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/logged-models/search",
        &format!(r#"{{"experiment_ids": ["{EXP_ID}"]}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let names: Vec<String> = res.json()["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["info"]["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"search_me".to_string()));
}

#[tokio::test]
async fn search_logged_models_missing_experiment_ids_is_invalid_parameter_value() {
    let server = TestServer::start("search_missing").await;
    let res = post(&server, "/api/2.0", "/mlflow/logged-models/search", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

#[tokio::test]
async fn search_logged_models_pagination_walk_round_trips_token() {
    let server = TestServer::start("search_page").await;
    for i in 0..3 {
        create_model(&server, "/api/2.0", &format!("page_model_{i}")).await;
    }

    let mut names = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let body = match &token {
            Some(t) => format!(
                r#"{{"experiment_ids": ["{EXP_ID}"], "max_results": 1, "page_token": "{t}"}}"#
            ),
            None => format!(r#"{{"experiment_ids": ["{EXP_ID}"], "max_results": 1}}"#),
        };
        let res = post(&server, "/api/2.0", "/mlflow/logged-models/search", &body).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        let json = res.json();
        let page = json["models"].as_array().unwrap();
        assert!(page.len() <= 1);
        for m in page {
            names.push(m["info"]["name"].as_str().unwrap().to_string());
        }
        match json.get("next_page_token").and_then(Value::as_str) {
            Some(t) => token = Some(t.to_string()),
            None => break,
        }
    }
    for i in 0..3 {
        assert!(names.contains(&format!("page_model_{i}")));
    }
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len());
}

// ---------------------------------------------------------------------------
// search-datasets
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_datasets_missing_leading_slash_route_works() {
    let server = TestServer::start("search_datasets_quirk").await;
    // The route table quirk (§3.4): no slash between the version and "mlflow".
    let res = post(
        &server,
        "",
        "/api/2.0mlflow/experiments/search-datasets",
        &format!(r#"{{"experiment_ids": ["{EXP_ID}"]}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    // No datasets logged for this experiment in the fixture: an empty repeated
    // field is omitted from the JSON entirely (matches Python's proto->JSON).
    assert_eq!(res.json(), serde_json::json!({}));
}

#[tokio::test]
async fn search_datasets_hand_registered_ajax_route_works() {
    let server = TestServer::start("search_datasets_ajax").await;
    // The correctly-slashed ajax route hand-registered in
    // `mlflow/server/__init__.py:135`.
    let res = post(
        &server,
        "",
        "/ajax-api/2.0/mlflow/experiments/search-datasets",
        &format!(r#"{{"experiment_ids": ["{EXP_ID}"]}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json(), serde_json::json!({}));
}

#[tokio::test]
async fn search_datasets_ajax_route_from_route_table_also_works() {
    let server = TestServer::start("search_datasets_ajax_quirk").await;
    // The route-table-generated ajax twin of the missing-slash quirk path.
    let res = post(
        &server,
        "",
        "/ajax-api/2.0mlflow/experiments/search-datasets",
        &format!(r#"{{"experiment_ids": ["{EXP_ID}"]}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
}

#[tokio::test]
async fn search_datasets_empty_experiment_ids_is_invalid_parameter_value() {
    let server = TestServer::start("search_datasets_empty").await;
    let res = post(
        &server,
        "",
        "/api/2.0mlflow/experiments/search-datasets",
        "{}",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        json["message"],
        "SearchDatasets request must specify at least one experiment_id."
    );
}

#[tokio::test]
async fn search_datasets_too_many_experiment_ids_is_invalid_parameter_value() {
    let server = TestServer::start("search_datasets_too_many").await;
    let ids: Vec<String> = (0..21).map(|i| format!("\"{i}\"")).collect();
    let res = post(
        &server,
        "",
        "/api/2.0mlflow/experiments/search-datasets",
        &format!(r#"{{"experiment_ids": [{}]}}"#, ids.join(",")),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        json["message"],
        "SearchDatasets request cannot specify more than 20 experiment_ids. \
         Received 21 experiment_ids."
    );
}

#[tokio::test]
async fn search_datasets_finds_logged_run_dataset() {
    // The runs HTTP endpoints (T3.2) aren't implemented yet, so seed the run +
    // dataset input directly through the store (the same `TrackingStore` the
    // HTTP layer calls into) rather than over HTTP.
    let db_file = TempDb::new("search_datasets_found");
    let db = Db::connect(&db_file.uri(), PoolConfig::default())
        .await
        .expect("connect temp fixture");
    let store = TrackingStore::new(db, ART_ROOT);
    let run = store
        .create_run(WORKSPACE, EXP_ID, None, None, None, &[])
        .await
        .expect("create run");
    store
        .log_inputs(
            WORKSPACE,
            &run.info.run_id,
            &[mlflow_store::DatasetInputSpec {
                name: "ds1".to_string(),
                digest: "abc123".to_string(),
                source_type: "local".to_string(),
                source: "path".to_string(),
                schema: None,
                profile: None,
                tags: Vec::new(),
            }],
            &[],
        )
        .await
        .expect("log inputs");

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
    let app = build_app_with_recorder(&config, recorder, Some(AppState::new(store)));
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
    let server = TestServer {
        base: format!("http://{addr}"),
        shutdown: Some(shutdown_tx),
        handle: Some(handle),
        _db: db_file,
    };

    let res = post(
        &server,
        "",
        "/api/2.0mlflow/experiments/search-datasets",
        &format!(r#"{{"experiment_ids": ["{EXP_ID}"]}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let summaries = res.json()["dataset_summaries"].as_array().unwrap().clone();
    assert!(summaries.iter().any(|s| s["name"] == "ds1"));
}
