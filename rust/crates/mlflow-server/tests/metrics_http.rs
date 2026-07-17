//! HTTP integration tests for the metric history endpoints (plan T3.3, §3.3).
//!
//! Same harness pattern as `experiments_http.rs` (real socket, copy of the
//! committed fixture DB). Metrics are logged through the store directly
//! (runs/log-batch HTTP endpoints are a later phase, T3.2), then fetched over
//! HTTP to exercise the three endpoints:
//!
//! * `GET /mlflow/metrics/get-history` (proto, both prefixes) — happy path +
//!   pagination + the `max_results<=0` error + the "no schema validator"
//!   tolerance for a non-numeric `max_results`.
//! * `GET /mlflow/metrics/get-history-bulk-interval` (proto, both prefixes) —
//!   multi-run fetch, interval sampling over >2500 points, and the
//!   run_ids/metric_key/max_results/start_step/end_step error cases.
//! * `GET /ajax-api/2.0/mlflow/metrics/get-history-bulk` (ajax-only,
//!   hand-rolled JSON) — exact response body byte-for-byte, run_id cap, and
//!   the metric_key-missing error.

use std::path::{Path, PathBuf};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, MetricInput, PoolConfig, TrackingStore};
use serde_json::Value;
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const DEFAULT_WORKSPACE: &str = "default";
const DEFAULT_EXPERIMENT_ID: &str = "0";

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
            "mlflow_rust_server_metrics_{}_{}_{}.db",
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

/// A running test server with a base URL and a handle to the store (so tests
/// can log metrics directly, bypassing the not-yet-implemented runs/log-batch
/// HTTP endpoints); shuts down on drop.
struct TestServer {
    base: String,
    store: TrackingStore,
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
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(&config, recorder, Some(AppState::new(store.clone())));

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
            store,
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
            _db: db_file,
        }
    }

    /// Create a run in the default experiment/workspace and return its id.
    async fn create_run(&self, name: &str) -> String {
        self.store
            .create_run(
                DEFAULT_WORKSPACE,
                DEFAULT_EXPERIMENT_ID,
                None,
                Some(0),
                Some(name),
                &[],
            )
            .await
            .expect("create run")
            .info
            .run_id
    }

    async fn log_metric(&self, run_id: &str, key: &str, value: f64, timestamp: i64, step: i64) {
        self.store
            .log_metric(
                DEFAULT_WORKSPACE,
                run_id,
                &MetricInput {
                    key: key.to_string(),
                    value,
                    timestamp,
                    step,
                    model_id: None,
                    dataset_name: None,
                    dataset_digest: None,
                },
            )
            .await
            .expect("log metric");
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
    content_type: Option<String>,
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
                let content_type = res
                    .headers()
                    .get(hyper::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                return HttpResponse {
                    status,
                    body: String::from_utf8_lossy(&bytes).into_owned(),
                    content_type,
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

async fn get(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::GET, &format!("{prefix}{endpoint}"), None).await
}

// ---------------------------------------------------------------------------
// get-history
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_history_happy_path_on_both_prefixes() {
    let server = TestServer::start("history_happy").await;
    let run_id = server.create_run("r1").await;
    server.log_metric(&run_id, "acc", 0.1, 100, 0).await;
    server.log_metric(&run_id, "acc", 0.2, 200, 1).await;
    server.log_metric(&run_id, "acc", 0.3, 300, 2).await;

    for prefix in PREFIXES {
        let res = get(
            &server,
            prefix,
            &format!("/mlflow/metrics/get-history?run_id={run_id}&metric_key=acc"),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        let json = res.json();
        let metrics = json["metrics"].as_array().unwrap();
        assert_eq!(metrics.len(), 3);
        // Ordered by (timestamp, step, value).
        assert_eq!(metrics[0]["value"], 0.1);
        assert_eq!(metrics[0]["timestamp"], 100);
        assert_eq!(metrics[0]["step"], 0);
        assert_eq!(metrics[2]["value"], 0.3);
        assert!(json.get("next_page_token").is_none());
    }
}

#[tokio::test]
async fn get_history_run_uuid_fallback() {
    let server = TestServer::start("history_run_uuid").await;
    let run_id = server.create_run("r_uuid").await;
    server.log_metric(&run_id, "loss", 1.0, 1, 0).await;

    // `run_uuid` is the deprecated alias for `run_id`; used when `run_id` is absent.
    let res = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/metrics/get-history?run_uuid={run_id}&metric_key=loss"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["metrics"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn get_history_pagination_walk() {
    let server = TestServer::start("history_page").await;
    let run_id = server.create_run("r_page").await;
    for i in 0..10 {
        server.log_metric(&run_id, "m", i as f64, i, i).await;
    }

    let mut values = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let url = match &token {
            Some(t) => format!(
                "/mlflow/metrics/get-history?run_id={run_id}&metric_key=m&max_results=3&page_token={t}"
            ),
            None => format!("/mlflow/metrics/get-history?run_id={run_id}&metric_key=m&max_results=3"),
        };
        let res = get(&server, "/api/2.0", &url).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        let json = res.json();
        let page = json["metrics"].as_array().unwrap();
        assert!(page.len() <= 3);
        for m in page {
            values.push(m["value"].as_f64().unwrap() as i64);
        }
        match json.get("next_page_token").and_then(Value::as_str) {
            Some(t) => token = Some(t.to_string()),
            None => break,
        }
    }
    assert_eq!(values, (0..10).collect::<Vec<_>>());
}

#[tokio::test]
async fn get_history_missing_required_params() {
    let server = TestServer::start("history_missing").await;

    let no_run_id = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history?metric_key=acc",
    )
    .await;
    assert_eq!(
        no_run_id.status,
        StatusCode::BAD_REQUEST,
        "{}",
        no_run_id.body
    );
    assert_eq!(no_run_id.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        no_run_id.json()["message"],
        "Missing value for required parameter 'run_id'. \
         See the API docs for more information about request parameters."
    );

    let no_metric_key = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history?run_id=abc",
    )
    .await;
    assert_eq!(
        no_metric_key.status,
        StatusCode::BAD_REQUEST,
        "{}",
        no_metric_key.body
    );
    assert_eq!(
        no_metric_key.json()["message"],
        "Missing value for required parameter 'metric_key'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn get_history_non_positive_max_results_is_invalid_parameter_value() {
    let server = TestServer::start("history_bad_max").await;
    let run_id = server.create_run("r_bad_max").await;
    server.log_metric(&run_id, "acc", 1.0, 1, 0).await;

    let res = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/metrics/get-history?run_id={run_id}&metric_key=acc&max_results=0"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Invalid value 0 for parameter 'max_results' supplied. It must be a positive integer."
    );
}

#[tokio::test]
async fn get_history_non_numeric_max_results_is_silently_ignored() {
    // `_get_metric_history`'s schema has no validator for `max_results`
    // (handlers.py:2070-2074): a non-numeric value fails protobuf's `parse_dict`
    // internally, but nothing re-checks it afterwards, so the field is simply
    // treated as absent (non-paginated) instead of erroring.
    let server = TestServer::start("history_nonnumeric_max").await;
    let run_id = server.create_run("r_nonnumeric").await;
    server.log_metric(&run_id, "acc", 1.0, 1, 0).await;
    server.log_metric(&run_id, "acc", 2.0, 2, 1).await;

    let res = get(
        &server,
        "/api/2.0",
        &format!(
            "/mlflow/metrics/get-history?run_id={run_id}&metric_key=acc&max_results=notanumber"
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let json = res.json();
    assert_eq!(json["metrics"].as_array().unwrap().len(), 2);
    assert!(json.get("next_page_token").is_none());
}

// ---------------------------------------------------------------------------
// get-history-bulk-interval
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bulk_interval_multi_run_on_both_prefixes() {
    let server = TestServer::start("interval_multi").await;
    let run_a = server.create_run("a").await;
    let run_b = server.create_run("b").await;
    for step in 0..5 {
        server
            .log_metric(&run_a, "acc", step as f64, step, step)
            .await;
        server
            .log_metric(&run_b, "acc", (step * 10) as f64, step, step)
            .await;
    }

    for prefix in PREFIXES {
        let res = get(
            &server,
            prefix,
            &format!(
                "/mlflow/metrics/get-history-bulk-interval?run_ids={run_a}&run_ids={run_b}&metric_key=acc"
            ),
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.body);
        let metrics = res.json()["metrics"].as_array().unwrap().clone();
        // 5 steps per run, both runs, all within max_results default (2500).
        assert_eq!(metrics.len(), 10);
        let run_ids: Vec<String> = metrics
            .iter()
            .map(|m| m["run_id"].as_str().unwrap().to_string())
            .collect();
        assert!(run_ids.contains(&run_a));
        assert!(run_ids.contains(&run_b));
    }
}

#[tokio::test]
async fn bulk_interval_sampling_over_max_results() {
    let server = TestServer::start("interval_sampling").await;
    let run_id = server.create_run("dense").await;
    // 3000 distinct steps, well above MAX_RESULTS_PER_RUN (2500), to force
    // interval sampling (not just "keep everything").
    for step in 0..3000i64 {
        server
            .log_metric(&run_id, "loss", step as f64, step, step)
            .await;
    }

    let res = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/metrics/get-history-bulk-interval?run_ids={run_id}&metric_key=loss"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let metrics = res.json()["metrics"].as_array().unwrap().clone();
    // Sampled down close to MAX_RESULTS_PER_RUN (2500) — the store docs note
    // the response can carry a few extra points beyond the cap (the forced
    // endpoint + the min/max-per-run set unioned back in), so allow a small
    // slack rather than an exact `<= 2500`. What matters here is that
    // sampling actually kicked in (well under the full 3000 steps) and the
    // final step is always retained (forced endpoint).
    assert!(metrics.len() <= 2510, "got {} metrics", metrics.len());
    assert!(metrics.len() < 3000);
    let steps: Vec<i64> = metrics
        .iter()
        .map(|m| m["step"].as_i64().unwrap())
        .collect();
    assert!(steps.contains(&2999));
}

#[tokio::test]
async fn bulk_interval_start_end_step_range() {
    let server = TestServer::start("interval_range").await;
    let run_id = server.create_run("ranged").await;
    for step in 0..10i64 {
        server
            .log_metric(&run_id, "m", step as f64, step, step)
            .await;
    }

    let res = get(
        &server,
        "/api/2.0",
        &format!(
            "/mlflow/metrics/get-history-bulk-interval?run_ids={run_id}&metric_key=m&start_step=3&end_step=6"
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let steps: Vec<i64> = res.json()["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["step"].as_i64().unwrap())
        .collect();
    assert!(steps.iter().all(|s| (3..=6).contains(s)));
    assert!(steps.contains(&3));
    assert!(steps.contains(&6));
}

#[tokio::test]
async fn bulk_interval_missing_run_ids_is_invalid_parameter_value() {
    let server = TestServer::start("interval_missing_runs").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history-bulk-interval?metric_key=acc",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'run_ids'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn bulk_interval_missing_metric_key_is_invalid_parameter_value() {
    let server = TestServer::start("interval_missing_key").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history-bulk-interval?run_ids=abc",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "Missing value for required parameter 'metric_key'. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn bulk_interval_over_100_run_ids_is_invalid_parameter_value() {
    let server = TestServer::start("interval_too_many_runs").await;
    let run_ids_query: String = (0..101)
        .map(|i| format!("run_ids=r{i}"))
        .collect::<Vec<_>>()
        .join("&");
    let res = get(
        &server,
        "/api/2.0",
        &format!("/mlflow/metrics/get-history-bulk-interval?{run_ids_query}&metric_key=acc"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "GetMetricHistoryBulkInterval request must specify at most 100 run_ids. \
         Received 101 run_ids."
    );
}

#[tokio::test]
async fn bulk_interval_bad_max_results_range() {
    let server = TestServer::start("interval_bad_max_range").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history-bulk-interval?run_ids=abc&metric_key=acc&max_results=0",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "max_results must be between 1 and 2500. \
         See the API docs for more information about request parameters."
    );

    let over = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history-bulk-interval?run_ids=abc&metric_key=acc&max_results=99999",
    )
    .await;
    assert_eq!(over.status, StatusCode::BAD_REQUEST, "{}", over.body);
    assert_eq!(
        over.json()["message"],
        "max_results must be between 1 and 2500. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn bulk_interval_non_numeric_max_results_matches_python_schema_message() {
    let server = TestServer::start("interval_nonnumeric_max").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history-bulk-interval?run_ids=abc&metric_key=acc&max_results=abc",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "Invalid value \"abc\" for parameter 'max_results' supplied:  Hint: Value was of type \
         'str'. See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn bulk_interval_start_step_without_end_step_errors() {
    let server = TestServer::start("interval_partial_range").await;
    let run_id = server.create_run("partial").await;
    server.log_metric(&run_id, "m", 1.0, 1, 0).await;

    let res = get(
        &server,
        "/api/2.0",
        &format!(
            "/mlflow/metrics/get-history-bulk-interval?run_ids={run_id}&metric_key=m&start_step=1"
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "If either start step or end step are specified, both must be specified."
    );
}

#[tokio::test]
async fn bulk_interval_start_greater_than_end_step_errors() {
    let server = TestServer::start("interval_reversed_range").await;
    let run_id = server.create_run("reversed").await;
    server.log_metric(&run_id, "m", 1.0, 1, 0).await;

    let res = get(
        &server,
        "/api/2.0",
        &format!(
            "/mlflow/metrics/get-history-bulk-interval?run_ids={run_id}&metric_key=m&start_step=5&end_step=2"
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "end_step must be greater than start_step. Found start_step=5 and end_step=2."
    );
}

// ---------------------------------------------------------------------------
// get-history-bulk (ajax-only, hand-rolled JSON)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bulk_ajax_only_is_not_registered_under_api_prefix() {
    let server = TestServer::start("bulk_ajax_only").await;
    let res = get(
        &server,
        "/api/2.0",
        "/mlflow/metrics/get-history-bulk?run_id=abc&metric_key=acc",
    )
    .await;
    assert_eq!(res.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn bulk_exact_json_body_shape() {
    let server = TestServer::start("bulk_exact_body").await;
    let run_id = server.create_run("shape").await;
    server
        .log_metric(&run_id, "acc", 0.95, 1700000000123, 1)
        .await;
    server
        .log_metric(&run_id, "acc", 1.0, 1700000000456, 2)
        .await;

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("/mlflow/metrics/get-history-bulk?run_id={run_id}&metric_key=acc"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.content_type.as_deref(), Some("application/json"));
    // Byte-for-byte: compact separators, keys sorted alphabetically
    // (key, run_id, step, timestamp, value), trailing newline — verified
    // against a live Flask `jsonify`/dict-return response.
    let expected = format!(
        "{{\"metrics\":[{{\"key\":\"acc\",\"run_id\":\"{run_id}\",\"step\":1,\"timestamp\":1700000000123,\"value\":0.95}},\
         {{\"key\":\"acc\",\"run_id\":\"{run_id}\",\"step\":2,\"timestamp\":1700000000456,\"value\":1.0}}]}}\n"
    );
    assert_eq!(res.body, expected);
}

#[tokio::test]
async fn bulk_empty_metrics_body_shape() {
    let server = TestServer::start("bulk_empty_body").await;
    let run_id = server.create_run("empty").await;

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("/mlflow/metrics/get-history-bulk?run_id={run_id}&metric_key=nonexistent"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.body, "{\"metrics\":[]}\n");
}

#[tokio::test]
async fn bulk_multi_run_sorted_by_run_id() {
    let server = TestServer::start("bulk_multi_run").await;
    let run_a = server.create_run("a").await;
    let run_b = server.create_run("b").await;
    server.log_metric(&run_a, "acc", 1.0, 1, 0).await;
    server.log_metric(&run_b, "acc", 2.0, 1, 0).await;

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("/mlflow/metrics/get-history-bulk?run_id={run_a}&run_id={run_b}&metric_key=acc"),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let json = res.json();
    let metrics = json["metrics"].as_array().unwrap();
    assert_eq!(metrics.len(), 2);
    let run_ids: Vec<String> = metrics
        .iter()
        .map(|m| m["run_id"].as_str().unwrap().to_string())
        .collect();
    assert!(run_ids.contains(&run_a));
    assert!(run_ids.contains(&run_b));
}

#[tokio::test]
async fn bulk_missing_run_id_is_invalid_parameter_value() {
    let server = TestServer::start("bulk_missing_run_id").await;
    let res = get(
        &server,
        "/ajax-api/2.0",
        "/mlflow/metrics/get-history-bulk?metric_key=acc",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "GetMetricHistoryBulk request must specify at least one run_id."
    );
}

#[tokio::test]
async fn bulk_missing_metric_key_is_invalid_parameter_value() {
    let server = TestServer::start("bulk_missing_metric_key").await;
    let res = get(
        &server,
        "/ajax-api/2.0",
        "/mlflow/metrics/get-history-bulk?run_id=abc",
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "GetMetricHistoryBulk request must specify a metric_key."
    );
}

#[tokio::test]
async fn bulk_over_100_run_ids_is_invalid_parameter_value() {
    let server = TestServer::start("bulk_too_many_runs").await;
    let run_id_query: String = (0..101)
        .map(|i| format!("run_id=r{i}"))
        .collect::<Vec<_>>()
        .join("&");
    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!("/mlflow/metrics/get-history-bulk?{run_id_query}&metric_key=acc"),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(
        res.json()["message"],
        "GetMetricHistoryBulk request cannot specify more than 100 run_ids. \
         Received 101 run_ids."
    );
}

#[tokio::test]
async fn bulk_non_numeric_max_results_is_generic_500_html() {
    // Python: `int(request.args.get("max_results", ...))` raises a bare
    // (uncaught) `ValueError` here — `catch_mlflow_exception` only catches
    // `MlflowException` — so Flask's default non-debug error handler returns
    // a generic HTML 500 with no exception detail. Verified against a live
    // Flask app.
    let server = TestServer::start("bulk_nonnumeric_max").await;
    let res = get(
        &server,
        "/ajax-api/2.0",
        "/mlflow/metrics/get-history-bulk?run_id=abc&metric_key=acc&max_results=notanumber",
    )
    .await;
    assert_eq!(
        res.status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        res.body
    );
    assert_eq!(
        res.content_type.as_deref(),
        Some("text/html; charset=utf-8")
    );
    assert!(res.body.contains("Internal Server Error"));
    assert!(!res.body.contains("notanumber"));
}

#[tokio::test]
async fn bulk_max_results_caps_at_25000() {
    let server = TestServer::start("bulk_max_cap").await;
    let run_id = server.create_run("cap").await;
    for step in 0..5i64 {
        server
            .log_metric(&run_id, "m", step as f64, step, step)
            .await;
    }

    let res = get(
        &server,
        "/ajax-api/2.0",
        &format!(
            "/mlflow/metrics/get-history-bulk?run_id={run_id}&metric_key=m&max_results=999999999"
        ),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    // Cap is only observable as "doesn't error"; all 5 logged points return.
    assert_eq!(res.json()["metrics"].as_array().unwrap().len(), 5);
}
