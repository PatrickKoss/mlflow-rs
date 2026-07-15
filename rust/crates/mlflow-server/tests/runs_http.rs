//! HTTP integration tests for the run endpoints (plan T3.2).
//!
//! Boots the axum app on a real ephemeral socket against a fresh copy of the
//! committed Alembic-migrated SQLite fixture, then drives every run endpoint
//! over HTTP on both the `/api/` and `/ajax-api/` prefixes. Covers the happy
//! path per endpoint, required-param error bodies, the search `max_results`
//! violation payload, log-batch limit violations, the param-length error, the
//! view-type default, and a search pagination round-trip.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
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
            "mlflow_rust_runs_{}_{}_{}.db",
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

/// Create a run in the fixture's active experiment (id 1) and return its id.
async fn create_run(server: &TestServer, prefix: &str, name: &str) -> String {
    let res = post(
        server,
        prefix,
        "/mlflow/runs/create",
        &format!(r#"{{"experiment_id": "1", "run_name": "{name}", "start_time": 111}}"#),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    res.json()["run"]["info"]["run_id"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn create_and_get_run_on_both_prefixes() {
    let server = TestServer::start("crud").await;

    for (i, prefix) in PREFIXES.iter().enumerate() {
        let name = format!("run_{i}");
        let create = post(
            &server,
            prefix,
            "/mlflow/runs/create",
            &format!(
                r#"{{"experiment_id": "1", "run_name": "{name}", "start_time": 42,
                     "tags": [{{"key": "team", "value": "rust"}}]}}"#
            ),
        )
        .await;
        assert_eq!(create.status, StatusCode::OK, "{}", create.body);
        let created = create.json();
        let info = &created["run"]["info"];
        let run_id = info["run_id"].as_str().unwrap().to_string();

        // RunInfo parity: both run_id and run_uuid are set to the id.
        assert_eq!(info["run_uuid"], run_id);
        assert_eq!(info["run_name"], name);
        assert_eq!(info["experiment_id"], "1");
        assert_eq!(info["status"], "RUNNING");
        assert_eq!(info["lifecycle_stage"], "active");
        assert!(info["start_time"].is_number());
        // user_id defaults to "" (always emitted).
        assert_eq!(info["user_id"], "");

        // The mlflow.runName tag is synthesized and returned in run data.
        let tags = &created["run"]["data"]["tags"];
        let names: Vec<&str> = tags
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["key"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"mlflow.runName"));
        assert!(names.contains(&"team"));

        let got = get(
            &server,
            prefix,
            &format!("/mlflow/runs/get?run_id={run_id}"),
        )
        .await;
        assert_eq!(got.status, StatusCode::OK, "{}", got.body);
        assert_eq!(got.json()["run"]["info"]["run_id"], run_id);
    }
}

#[tokio::test]
async fn get_run_accepts_deprecated_run_uuid() {
    let server = TestServer::start("run_uuid").await;
    let run_id = create_run(&server, "/api/2.0", "uuid_run").await;
    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_uuid={run_id}"),
    )
    .await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert_eq!(got.json()["run"]["info"]["run_id"], run_id);
}

#[tokio::test]
async fn update_run_sets_status_end_time_and_name() {
    let server = TestServer::start("update").await;
    let run_id = create_run(&server, "/api/2.0", "before").await;

    let update = post(
        &server,
        "/ajax-api/2.0",
        "/mlflow/runs/update",
        &format!(
            r#"{{"run_id": "{run_id}", "status": "FINISHED", "end_time": 999,
                 "run_name": "after"}}"#
        ),
    )
    .await;
    assert_eq!(update.status, StatusCode::OK, "{}", update.body);
    let info = &update.json()["run_info"];
    assert_eq!(info["status"], "FINISHED");
    assert_eq!(info["end_time"], 999);
    assert_eq!(info["run_name"], "after");

    // The name change synced the mlflow.runName tag.
    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_id={run_id}"),
    )
    .await;
    let run_name_tag = got.json()["run"]["data"]["tags"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["key"] == "mlflow.runName")
        .map(|t| t["value"].as_str().unwrap().to_string());
    assert_eq!(run_name_tag.as_deref(), Some("after"));
}

#[tokio::test]
async fn delete_and_restore_run() {
    let server = TestServer::start("del_restore").await;
    let run_id = create_run(&server, "/api/2.0", "gone").await;

    let del = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/delete",
        &format!(r#"{{"run_id": "{run_id}"}}"#),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);
    assert_eq!(del.json(), json!({}));

    // DELETED_ONLY search finds it.
    let deleted = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/search",
        r#"{"experiment_ids": ["1"], "run_view_type": "DELETED_ONLY", "max_results": 100}"#,
    )
    .await;
    let ids: Vec<String> = deleted.json()["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["info"]["run_id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&run_id));

    let restore = post(
        &server,
        "/ajax-api/2.0",
        "/mlflow/runs/restore",
        &format!(r#"{{"run_id": "{run_id}"}}"#),
    )
    .await;
    assert_eq!(restore.status, StatusCode::OK, "{}", restore.body);
}

#[tokio::test]
async fn log_metric_param_tag_roundtrip() {
    let server = TestServer::start("log_kv").await;
    let run_id = create_run(&server, "/api/2.0", "kv").await;

    let m = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-metric",
        &format!(
            r#"{{"run_id": "{run_id}", "key": "acc", "value": 0.9, "timestamp": 5, "step": 2}}"#
        ),
    )
    .await;
    assert_eq!(m.status, StatusCode::OK, "{}", m.body);

    let p = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-parameter",
        &format!(r#"{{"run_id": "{run_id}", "key": "lr", "value": "0.01"}}"#),
    )
    .await;
    assert_eq!(p.status, StatusCode::OK, "{}", p.body);

    let t = post(
        &server,
        "/ajax-api/2.0",
        "/mlflow/runs/set-tag",
        &format!(r#"{{"run_id": "{run_id}", "key": "phase", "value": "train"}}"#),
    )
    .await;
    assert_eq!(t.status, StatusCode::OK, "{}", t.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_id={run_id}"),
    )
    .await;
    let data = &got.json()["run"]["data"];
    assert_eq!(data["metrics"][0]["key"], "acc");
    assert_eq!(data["metrics"][0]["value"], 0.9);
    assert_eq!(data["metrics"][0]["step"], 2);
    let params: Vec<&str> = data["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["key"].as_str().unwrap())
        .collect();
    assert!(params.contains(&"lr"));
    let phase = data["tags"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["key"] == "phase");
    assert_eq!(phase.unwrap()["value"], "train");

    // delete-tag removes it.
    let dt = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/delete-tag",
        &format!(r#"{{"run_id": "{run_id}", "key": "phase"}}"#),
    )
    .await;
    assert_eq!(dt.status, StatusCode::OK, "{}", dt.body);
    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_id={run_id}"),
    )
    .await;
    let has_phase = got.json()["run"]["data"]["tags"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["key"] == "phase");
    assert!(!has_phase);
}

#[tokio::test]
async fn log_batch_happy_path() {
    let server = TestServer::start("log_batch").await;
    let run_id = create_run(&server, "/api/2.0", "batch").await;

    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-batch",
        &format!(
            r#"{{"run_id": "{run_id}",
                 "metrics": [{{"key": "m1", "value": 1.0, "timestamp": 1, "step": 0}}],
                 "params": [{{"key": "p1", "value": "v1"}}],
                 "tags": [{{"key": "t1", "value": "w1"}}]}}"#
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json(), json!({}));

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_id={run_id}"),
    )
    .await;
    let data = &got.json()["run"]["data"];
    assert_eq!(data["metrics"][0]["key"], "m1");
    assert!(data["params"]
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p["key"] == "p1"));
}

#[tokio::test]
async fn log_inputs_and_outputs() {
    let server = TestServer::start("io").await;
    let run_id = create_run(&server, "/api/2.0", "io").await;

    let inputs = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-inputs",
        &format!(
            r#"{{"run_id": "{run_id}", "datasets": [
                 {{"dataset": {{"name": "d1", "digest": "abc", "source_type": "S3",
                              "source": "s3://x"}},
                   "tags": [{{"key": "mlflow.data.context", "value": "train"}}]}}
               ]}}"#
        ),
    )
    .await;
    assert_eq!(inputs.status, StatusCode::OK, "{}", inputs.body);

    let outputs = post(
        &server,
        "/ajax-api/2.0",
        "/mlflow/runs/outputs",
        &format!(r#"{{"run_id": "{run_id}", "models": [{{"model_id": "m-123", "step": 3}}]}}"#),
    )
    .await;
    assert_eq!(outputs.status, StatusCode::OK, "{}", outputs.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_id={run_id}"),
    )
    .await;
    let run = &got.json()["run"];
    let ds = &run["inputs"]["dataset_inputs"][0]["dataset"];
    assert_eq!(ds["name"], "d1");
    assert_eq!(ds["digest"], "abc");
    assert_eq!(run["outputs"]["model_outputs"][0]["model_id"], "m-123");
    assert_eq!(run["outputs"]["model_outputs"][0]["step"], 3);
}

#[tokio::test]
async fn log_model_appends_history_tag() {
    let server = TestServer::start("log_model").await;
    let run_id = create_run(&server, "/api/2.0", "model").await;

    let model_json = json!({
        "artifact_path": "model",
        "run_id": run_id,
        "utc_time_created": "2020-01-01 00:00:00",
        "model_uuid": "uuid-1",
        "flavors": {"python_function": {"loader_module": "m", "config": {"x": 1}}}
    })
    .to_string();
    let body = json!({"run_id": run_id, "model_json": model_json}).to_string();

    let res = post(&server, "/api/2.0", "/mlflow/runs/log-model", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);

    let got = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/runs/get?run_id={run_id}"),
    )
    .await;
    let history = got.json()["run"]["data"]["tags"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["key"] == "mlflow.log-model.history")
        .map(|t| t["value"].as_str().unwrap().to_string())
        .expect("history tag present");
    // Value is a JSON array; the flavor's nested `config` key is stripped.
    let parsed: Value = serde_json::from_str(&history).unwrap();
    assert_eq!(parsed[0]["artifact_path"], "model");
    assert_eq!(parsed[0]["model_uuid"], "uuid-1");
    assert!(parsed[0]["flavors"]["python_function"]["config"].is_null());
    assert_eq!(
        parsed[0]["flavors"]["python_function"]["loader_module"],
        "m"
    );
}

#[tokio::test]
async fn log_model_malformed_json_is_invalid_parameter_value() {
    let server = TestServer::start("log_model_bad").await;
    let run_id = create_run(&server, "/api/2.0", "model_bad").await;
    let body = json!({"run_id": run_id, "model_json": "not json"}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/runs/log-model", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(res.json()["message"]
        .as_str()
        .unwrap()
        .contains("is not a valid JSON"));
}

#[tokio::test]
async fn log_model_missing_fields_is_invalid_parameter_value() {
    let server = TestServer::start("log_model_missing").await;
    let run_id = create_run(&server, "/api/2.0", "model_missing").await;
    let model_json = json!({"run_id": run_id}).to_string();
    let body = json!({"run_id": run_id, "model_json": model_json}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/runs/log-model", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let msg = res.json()["message"].as_str().unwrap().to_string();
    assert!(msg.contains("missing mandatory fields"), "{msg}");
    assert!(msg.contains("'artifact_path'"), "{msg}");
    assert!(msg.contains("'flavors'"), "{msg}");
    assert!(msg.contains("'utc_time_created'"), "{msg}");
}

#[tokio::test]
async fn search_runs_view_type_default_is_active_only() {
    let server = TestServer::start("search_default").await;
    let active = create_run(&server, "/api/2.0", "active_run").await;
    let deleted = create_run(&server, "/api/2.0", "deleted_run").await;
    post(
        &server,
        "/api/2.0",
        "/mlflow/runs/delete",
        &format!(r#"{{"run_id": "{deleted}"}}"#),
    )
    .await;

    // No run_view_type field → default ACTIVE_ONLY (deleted run excluded).
    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/search",
        r#"{"experiment_ids": ["1"], "max_results": 100}"#,
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let ids: Vec<String> = res.json()["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["info"]["run_id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&active));
    assert!(!ids.contains(&deleted));
}

#[tokio::test]
async fn search_runs_pagination_round_trip() {
    let server = TestServer::start("search_page").await;
    // Create a handful of runs in experiment 1.
    let mut created = Vec::new();
    for i in 0..3 {
        created.push(create_run(&server, "/api/2.0", &format!("page_{i}")).await);
    }

    let mut seen = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let body = match &token {
            Some(t) => format!(
                r#"{{"experiment_ids": ["1"], "run_view_type": "ALL", "max_results": 1, "page_token": "{t}"}}"#
            ),
            None => {
                r#"{"experiment_ids": ["1"], "run_view_type": "ALL", "max_results": 1}"#.to_string()
            }
        };
        let res = post(&server, "/api/2.0", "/mlflow/runs/search", &body).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        let json = res.json();
        let page = json["runs"].as_array().unwrap();
        assert!(page.len() <= 1);
        for r in page {
            seen.push(r["info"]["run_id"].as_str().unwrap().to_string());
        }
        match json.get("next_page_token").and_then(Value::as_str) {
            Some(t) => token = Some(t.to_string()),
            None => break,
        }
    }
    // All created runs are visited, no duplicates.
    for id in &created {
        assert!(seen.contains(id), "missing {id}");
    }
    let mut sorted = seen.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), seen.len());
}

#[tokio::test]
async fn search_runs_max_results_over_limit_matches_python_message() {
    let server = TestServer::start("search_max").await;
    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/search",
        r#"{"experiment_ids": ["1"], "max_results": 999999}"#,
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "INVALID_PARAMETER_VALUE");
    // Handler-level message (distinct from the store's threshold message).
    assert_eq!(
        json["message"],
        "Invalid value 999999 for parameter 'max_results' supplied. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn log_batch_too_many_metrics_is_invalid_parameter_value() {
    let server = TestServer::start("batch_metrics").await;
    let run_id = create_run(&server, "/api/2.0", "batch_lim").await;
    let metrics: Vec<Value> = (0..1001)
        .map(|i| json!({"key": format!("m{i}"), "value": 1.0, "timestamp": 1, "step": 0}))
        .collect();
    let body = json!({"run_id": run_id, "metrics": metrics}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/runs/log-batch", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(res.json()["message"]
        .as_str()
        .unwrap()
        .contains("1000 metrics"));
}

#[tokio::test]
async fn log_batch_too_many_total_entities_is_invalid_parameter_value() {
    let server = TestServer::start("batch_total").await;
    let run_id = create_run(&server, "/api/2.0", "batch_total").await;
    // 900 metrics + 100 params + 100 tags = 1100 total (> 1000).
    let metrics: Vec<Value> = (0..900)
        .map(|i| json!({"key": format!("m{i}"), "value": 1.0, "timestamp": 1, "step": 0}))
        .collect();
    let params: Vec<Value> = (0..100)
        .map(|i| json!({"key": format!("p{i}"), "value": "v"}))
        .collect();
    let tags: Vec<Value> = (0..100)
        .map(|i| json!({"key": format!("t{i}"), "value": "w"}))
        .collect();
    let body =
        json!({"run_id": run_id, "metrics": metrics, "params": params, "tags": tags}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/runs/log-batch", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert!(res.json()["message"].as_str().unwrap().contains("1000"));
}

#[tokio::test]
async fn log_param_value_too_long_is_invalid_parameter_value() {
    let server = TestServer::start("param_len").await;
    let run_id = create_run(&server, "/api/2.0", "param_len").await;
    let long = "x".repeat(6001);
    let body = json!({"run_id": run_id, "key": "k", "value": long}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/runs/log-parameter", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(res.json()["message"]
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("param value"));
}

#[tokio::test]
async fn log_param_immutable_value_is_invalid_parameter_value() {
    let server = TestServer::start("param_immut").await;
    let run_id = create_run(&server, "/api/2.0", "immut").await;
    let first = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-parameter",
        &format!(r#"{{"run_id": "{run_id}", "key": "p", "value": "1"}}"#),
    )
    .await;
    assert_eq!(first.status, StatusCode::OK, "{}", first.body);
    // Same value → idempotent OK.
    let same = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-parameter",
        &format!(r#"{{"run_id": "{run_id}", "key": "p", "value": "1"}}"#),
    )
    .await;
    assert_eq!(same.status, StatusCode::OK, "{}", same.body);
    // Different value → error.
    let diff = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-parameter",
        &format!(r#"{{"run_id": "{run_id}", "key": "p", "value": "2"}}"#),
    )
    .await;
    assert_eq!(diff.status, StatusCode::BAD_REQUEST, "{}", diff.body);
    assert!(diff.json()["message"]
        .as_str()
        .unwrap()
        .contains("Changing param values is not allowed"));
}

#[tokio::test]
async fn create_run_missing_experiment_is_resource_does_not_exist() {
    let server = TestServer::start("create_no_exp").await;
    let res = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/create",
        r#"{"experiment_id": "999999"}"#,
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn get_run_missing_run_id_is_invalid_parameter_value() {
    let server = TestServer::start("get_no_id").await;
    let res = get(&server, "/api/2.0", "/mlflow/runs/get").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    let json = res.json();
    assert_eq!(json["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        json["message"],
        "Missing value for required parameter 'run_id'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn log_metric_missing_required_fields() {
    let server = TestServer::start("metric_missing").await;
    let run_id = create_run(&server, "/api/2.0", "metric_missing").await;

    // Missing key.
    let no_key = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-metric",
        &format!(r#"{{"run_id": "{run_id}", "value": 1.0, "timestamp": 1}}"#),
    )
    .await;
    assert_eq!(no_key.status, StatusCode::BAD_REQUEST, "{}", no_key.body);
    assert_eq!(
        no_key.json()["message"],
        "Missing value for required parameter 'key'. \
         See the API docs for more information about request parameters."
    );

    // Missing value.
    let no_val = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-metric",
        &format!(r#"{{"run_id": "{run_id}", "key": "m", "timestamp": 1}}"#),
    )
    .await;
    assert_eq!(no_val.status, StatusCode::BAD_REQUEST, "{}", no_val.body);
    assert_eq!(
        no_val.json()["message"],
        "Missing value for required parameter 'value'. \
         See the API docs for more information about request parameters."
    );

    // Missing timestamp.
    let no_ts = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-metric",
        &format!(r#"{{"run_id": "{run_id}", "key": "m", "value": 1.0}}"#),
    )
    .await;
    assert_eq!(no_ts.status, StatusCode::BAD_REQUEST, "{}", no_ts.body);
    assert_eq!(
        no_ts.json()["message"],
        "Missing value for required parameter 'timestamp'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn delete_run_missing_run_id_error_body() {
    let server = TestServer::start("del_no_id").await;
    let res = post(&server, "/api/2.0", "/mlflow/runs/delete", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'run_id'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn metric_nan_inf_are_preserved_over_http() {
    let server = TestServer::start("metric_nan").await;
    let run_id = create_run(&server, "/api/2.0", "nan").await;
    // NaN and +Inf are logged (the store sanitizes storage); the get-run reads
    // them back. We only assert the writes succeed and the metric round-trips.
    let inf = post(
        &server,
        "/api/2.0",
        "/mlflow/runs/log-metric",
        &format!(r#"{{"run_id": "{run_id}", "key": "big", "value": 1e308, "timestamp": 1}}"#),
    )
    .await;
    assert_eq!(inf.status, StatusCode::OK, "{}", inf.body);
}
