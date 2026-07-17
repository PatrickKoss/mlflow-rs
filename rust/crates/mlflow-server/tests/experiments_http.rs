//! HTTP integration tests for the experiment endpoints (plan T3.1 + T3.5).
//!
//! Boots the axum app on a real ephemeral socket (same pattern as
//! `real_socket.rs`) against a fresh copy of the committed Alembic-migrated
//! SQLite fixture, then drives every endpoint over HTTP on **both** the `/api/`
//! and `/ajax-api/` prefixes, including POST + GET search with a pagination
//! walk, tag round-trips, and the error cases (missing params,
//! RESOURCE_DOES_NOT_EXIST, RESOURCE_ALREADY_EXISTS, bad max_results).

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
            "mlflow_rust_server_{}_{}_{}.db",
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
        // Per-router recorder handle so multiple test servers can coexist in
        // one process (a global recorder can only be installed once).
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

    // Retry the connect a few times while the accept loop starts.
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
    // Re-read the body bytes: Full<Bytes> is cheaply cloneable via its data.
    let body = req.body().clone();
    builder.body(body).unwrap()
}

// The two URL prefixes every proto endpoint is served on.
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

#[tokio::test]
async fn create_get_and_by_name_roundtrip_on_both_prefixes() {
    let server = TestServer::start("crud").await;

    for (i, prefix) in PREFIXES.iter().enumerate() {
        let name = format!("exp_{i}");
        let create = post(
            &server,
            prefix,
            "/mlflow/experiments/create",
            &format!(r#"{{"name": "{name}"}}"#),
        )
        .await;
        assert_eq!(create.status, StatusCode::OK, "{}", create.body);
        let created = create.json();
        let exp_id = created["experiment_id"].as_str().unwrap().to_string();

        // get by id
        let got = get(
            &server,
            prefix,
            &format!("/mlflow/experiments/get?experiment_id={exp_id}"),
        )
        .await;
        assert_eq!(got.status, StatusCode::OK, "{}", got.body);
        let exp = &got.json()["experiment"];
        assert_eq!(exp["experiment_id"], exp_id);
        assert_eq!(exp["name"], name);
        assert_eq!(exp["lifecycle_stage"], "active");
        // int64 timestamps are JSON numbers (not strings).
        assert!(exp["creation_time"].is_number());

        // get by name (GET with query args)
        let by_name = get(
            &server,
            prefix,
            &format!("/mlflow/experiments/get-by-name?experiment_name={name}"),
        )
        .await;
        assert_eq!(by_name.status, StatusCode::OK, "{}", by_name.body);
        assert_eq!(by_name.json()["experiment"]["experiment_id"], exp_id);
    }
}

#[tokio::test]
async fn create_with_tags_and_artifact_location() {
    let server = TestServer::start("create_tags").await;
    let create = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/create",
        r#"{"name": "tagged", "artifact_location": "s3://custom/loc",
            "tags": [{"key": "team", "value": "rust"}]}"#,
    )
    .await;
    assert_eq!(create.status, StatusCode::OK, "{}", create.body);
    let id = create.json()["experiment_id"].as_str().unwrap().to_string();

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/experiments/get?experiment_id={id}"),
    )
    .await;
    let exp = &got.json()["experiment"];
    assert_eq!(exp["artifact_location"], "s3://custom/loc");
    assert_eq!(exp["tags"][0]["key"], "team");
    assert_eq!(exp["tags"][0]["value"], "rust");
}

#[tokio::test]
async fn experiment_response_always_carries_workspace() {
    // Python's `Experiment.to_proto` always emits `workspace` (proto field 9,
    // "Always `default` if workspace is not enabled") because
    // `resolve_entity_workspace_name` defaults it. The Rust response must too —
    // regression guard for the earlier `workspace: None` drop.
    let server = TestServer::start("ws_field").await;
    let id = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/create",
        r#"{"name": "ws-exp"}"#,
    )
    .await
    .json()["experiment_id"]
        .as_str()
        .unwrap()
        .to_string();

    // get
    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/experiments/get?experiment_id={id}"),
    )
    .await;
    assert_eq!(got.json()["experiment"]["workspace"], "default");

    // get-by-name
    let by_name = get(
        &server,
        "/api/2.0",
        "/mlflow/experiments/get-by-name?experiment_name=ws-exp",
    )
    .await;
    assert_eq!(by_name.json()["experiment"]["workspace"], "default");

    // search
    let search = get(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search?max_results=100&view_type=ALL",
    )
    .await;
    let experiments = search.json()["experiments"].as_array().unwrap().clone();
    assert!(
        experiments.iter().all(|e| e["workspace"] == "default"),
        "every searched experiment carries workspace=default"
    );
}

#[tokio::test]
async fn search_experiments_post_and_get_agree() {
    let server = TestServer::start("search_agree").await;

    let post_res = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search",
        r#"{"max_results": 100, "view_type": "ALL"}"#,
    )
    .await;
    assert_eq!(post_res.status, StatusCode::OK, "{}", post_res.body);
    let post_names: Vec<String> = post_res.json()["experiments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect();
    // Fixture has "Default" + "rust_store_fixture".
    assert!(post_names.contains(&"Default".to_string()));
    assert!(post_names.contains(&"rust_store_fixture".to_string()));

    // GET with query args (T3.5): view_type by enum name, max_results as string.
    let get_res = get(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search?max_results=100&view_type=ALL",
    )
    .await;
    assert_eq!(get_res.status, StatusCode::OK, "{}", get_res.body);
    let get_names: Vec<String> = get_res.json()["experiments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(post_names, get_names);
}

#[tokio::test]
async fn search_experiments_pagination_walk() {
    let server = TestServer::start("search_page").await;

    // Page size 1 over the two active fixture experiments (+ any created).
    let mut names = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let body = match &token {
            Some(t) => format!(r#"{{"max_results": 1, "view_type": "ALL", "page_token": "{t}"}}"#),
            None => r#"{"max_results": 1, "view_type": "ALL"}"#.to_string(),
        };
        let res = post(&server, "/api/2.0", "/mlflow/experiments/search", &body).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        let json = res.json();
        let page = json["experiments"].as_array().unwrap();
        assert!(page.len() <= 1);
        for e in page {
            names.push(e["name"].as_str().unwrap().to_string());
        }
        match json.get("next_page_token").and_then(Value::as_str) {
            Some(t) => token = Some(t.to_string()),
            None => break,
        }
    }
    assert!(names.contains(&"Default".to_string()));
    assert!(names.contains(&"rust_store_fixture".to_string()));
    // No duplicates across pages.
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len());
}

#[tokio::test]
async fn search_experiments_get_repeated_order_by() {
    let server = TestServer::start("search_orderby").await;
    // Repeated order_by query params (T3.5): a repeated proto field parsed from
    // multiple query occurrences.
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search?max_results=100&view_type=ALL&order_by=name+ASC&order_by=experiment_id+ASC",
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let names: Vec<String> = res.json()["experiments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect();
    // Ordered by name ASC: "Default" < "rust_store_fixture".
    let default_pos = names.iter().position(|n| n == "Default").unwrap();
    let fixture_pos = names
        .iter()
        .position(|n| n == "rust_store_fixture")
        .unwrap();
    assert!(default_pos < fixture_pos);
}

#[tokio::test]
async fn search_experiments_filter() {
    let server = TestServer::start("search_filter").await;
    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search",
        r#"{"max_results": 100, "view_type": "ALL", "filter": "name = 'Default'"}"#,
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let exps = res.json()["experiments"].as_array().unwrap().clone();
    assert_eq!(exps.len(), 1);
    assert_eq!(exps[0]["name"], "Default");
}

#[tokio::test]
async fn delete_and_restore_experiment() {
    let server = TestServer::start("del_restore").await;
    let id = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/create",
        r#"{"name": "to_delete"}"#,
    )
    .await
    .json()["experiment_id"]
        .as_str()
        .unwrap()
        .to_string();

    let del = post(
        &server,
        "/ajax-api/2.0",
        "/mlflow/experiments/delete",
        &format!(r#"{{"experiment_id": "{id}"}}"#),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);
    assert_eq!(del.json(), serde_json::json!({}));

    // Now DELETED_ONLY search should include it.
    let deleted = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search",
        r#"{"max_results": 100, "view_type": "DELETED_ONLY"}"#,
    )
    .await;
    let names: Vec<String> = deleted.json()["experiments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"to_delete".to_string()));

    let restore = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/restore",
        &format!(r#"{{"experiment_id": "{id}"}}"#),
    )
    .await;
    assert_eq!(restore.status, StatusCode::OK, "{}", restore.body);
}

#[tokio::test]
async fn update_experiment_renames() {
    let server = TestServer::start("update").await;
    let id = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/create",
        r#"{"name": "old_name"}"#,
    )
    .await
    .json()["experiment_id"]
        .as_str()
        .unwrap()
        .to_string();

    let update = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/update",
        &format!(r#"{{"experiment_id": "{id}", "new_name": "new_name"}}"#),
    )
    .await;
    assert_eq!(update.status, StatusCode::OK, "{}", update.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/experiments/get?experiment_id={id}"),
    )
    .await;
    assert_eq!(got.json()["experiment"]["name"], "new_name");
}

#[tokio::test]
async fn set_and_delete_experiment_tag_roundtrip() {
    let server = TestServer::start("tags").await;
    let id = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/create",
        r#"{"name": "tag_exp"}"#,
    )
    .await
    .json()["experiment_id"]
        .as_str()
        .unwrap()
        .to_string();

    let set = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/set-experiment-tag",
        &format!(r#"{{"experiment_id": "{id}", "key": "k", "value": "v1"}}"#),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "{}", set.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/experiments/get?experiment_id={id}"),
    )
    .await;
    let tags = got.json()["experiment"]["tags"].as_array().unwrap().clone();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["key"], "k");
    assert_eq!(tags[0]["value"], "v1");

    // Upsert
    post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/set-experiment-tag",
        &format!(r#"{{"experiment_id": "{id}", "key": "k", "value": "v2"}}"#),
    )
    .await;
    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/experiments/get?experiment_id={id}"),
    )
    .await;
    assert_eq!(got.json()["experiment"]["tags"][0]["value"], "v2");

    let del = post(
        &server,
        "/ajax-api/2.0",
        "/mlflow/experiments/delete-experiment-tag",
        &format!(r#"{{"experiment_id": "{id}", "key": "k"}}"#),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/experiments/get?experiment_id={id}"),
    )
    .await;
    assert_eq!(
        got.json()["experiment"]
            .get("tags")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0),
        0
    );
}

#[tokio::test]
async fn missing_required_param_is_invalid_parameter_value() {
    let server = TestServer::start("err_missing").await;
    let res = post(&server, "/api/2.0", "/mlflow/experiments/create", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        json["message"],
        "Missing value for required parameter 'name'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn get_missing_experiment_is_resource_does_not_exist() {
    let server = TestServer::start("err_missing_exp").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/experiments/get?experiment_id=999999",
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn get_by_name_missing_is_resource_does_not_exist() {
    let server = TestServer::start("err_missing_name").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/experiments/get-by-name?experiment_name=nope",
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "RESOURCE_DOES_NOT_EXIST");
    assert_eq!(
        json["message"],
        "Could not find experiment with name 'nope'"
    );
}

#[tokio::test]
async fn duplicate_name_is_resource_already_exists() {
    let server = TestServer::start("err_dup").await;
    // "Default" already exists in the fixture.
    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/create",
        r#"{"name": "Default"}"#,
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_ALREADY_EXISTS");
}

#[tokio::test]
async fn bad_max_results_is_invalid_parameter_value() {
    let server = TestServer::start("err_max").await;

    // Unset max_results (proto default 0) → rejected as non-positive.
    let zero = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search",
        r#"{"view_type": "ALL"}"#,
    )
    .await;
    assert_eq!(zero.status, StatusCode::BAD_REQUEST, "{}", zero.body);
    assert_eq!(zero.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(zero.json()["message"]
        .as_str()
        .unwrap()
        .contains("must be a positive integer"));

    // Over-threshold max_results.
    let over = post(
        &server,
        "/api/2.0",
        "/mlflow/experiments/search",
        r#"{"max_results": 999999, "view_type": "ALL"}"#,
    )
    .await;
    assert_eq!(over.status, StatusCode::BAD_REQUEST, "{}", over.body);
    assert!(over.json()["message"]
        .as_str()
        .unwrap()
        .contains("at most 50000"));
}

#[tokio::test]
async fn unimplemented_endpoint_returns_404() {
    let server = TestServer::start("unimpl").await;
    // A route-table RPC that `handler_for` does not yet wire up → 404 (no route
    // match). Label schemas (§3.x) are a later phase, so `getLabelSchema`
    // (`/api/3.0/mlflow/label-schemas/get`) falls through. Was previously
    // `listWorkspaces`, but workspaces landed in T10.2.
    let res = get(&server, "/api/3.0", "/mlflow/label-schemas/get").await;
    assert_eq!(res.status, StatusCode::NOT_FOUND);
    // Route miss, not a JSON error body from a matched handler.
    assert!(res.body.is_empty(), "{}", res.body);
}
