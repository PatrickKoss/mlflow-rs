//! HTTP integration tests for the Tracing V3 endpoints (plan T4.1, §3.6).
//!
//! Boots the axum app on a real ephemeral socket (same pattern as
//! `logged_models_http.rs`) against a fresh copy of the committed
//! Alembic-migrated SQLite fixture, then drives every V3 trace endpoint over
//! HTTP: start/get info, get-with-spans, batch get, search (+ pagination and
//! the max_results violation), delete (both modes), tag CRUD (path params),
//! link limits, correlation, and query-metrics.

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
const EXP_ID: &str = "0";

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
            "mlflow_rust_server_traces_{}_{}_{}.db",
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
    store: TrackingStore,
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
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
            store,
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

const PREFIXES: [&str; 2] = ["/api/3.0", "/ajax-api/3.0"];

async fn post(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    send(
        server,
        Method::POST,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

async fn get_q(server: &TestServer, prefix: &str, endpoint: &str) -> HttpResponse {
    send(server, Method::GET, &format!("{prefix}{endpoint}"), None).await
}

async fn get_body(server: &TestServer, prefix: &str, endpoint: &str, body: &str) -> HttpResponse {
    // GET-with-body helper (getTrace/batchGetTraces accept a JSON body).
    send(
        server,
        Method::GET,
        &format!("{prefix}{endpoint}"),
        Some(body),
    )
    .await
}

/// Start a trace via `startTraceV3` and return its trace_id.
async fn start_trace(server: &TestServer, trace_id: &str, exp_id: &str, state: &str) -> Value {
    let body = json!({
        "trace": {
            "trace_info": {
                "trace_id": trace_id,
                "trace_location": {
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": exp_id}
                },
                "request_time": "2024-01-01T00:00:00Z",
                "execution_duration": "1.500s",
                "state": state,
                "request_preview": "hello",
                "response_preview": "world",
                "tags": {"team": "rust"},
                "trace_metadata": {"mlflow.traceName": "my-trace"}
            }
        }
    })
    .to_string();
    let res = post(server, "/api/3.0", "/mlflow/traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    res.json()
}

// ---------------------------------------------------------------------------
// startTraceV3 / getTraceInfoV3
// ---------------------------------------------------------------------------

#[tokio::test]
async fn start_and_get_trace_info_on_both_prefixes() {
    let server = TestServer::start("start_get").await;

    for (i, prefix) in PREFIXES.iter().enumerate() {
        let trace_id = format!("tr-{i}");
        let started = start_trace(&server, &trace_id, EXP_ID, "OK").await;
        let info = &started["trace"]["trace_info"];
        assert_eq!(info["trace_id"], trace_id);
        assert_eq!(info["state"], "OK");
        assert_eq!(
            info["trace_location"]["mlflow_experiment"]["experiment_id"],
            EXP_ID
        );
        assert_eq!(info["request_preview"], "hello");
        assert_eq!(info["tags"]["team"], "rust");

        let got = get_q(&server, prefix, &format!("/mlflow/traces/{trace_id}")).await;
        assert_eq!(got.status, StatusCode::OK, "{}", got.body);
        assert_eq!(got.json()["trace"]["trace_info"]["trace_id"], trace_id);
        assert_eq!(got.json()["trace"]["trace_info"]["state"], "OK");
    }
}

#[tokio::test]
async fn get_missing_trace_info_is_resource_does_not_exist() {
    let server = TestServer::start("get_missing_info").await;
    let res = get_q(&server, "/api/3.0", "/mlflow/traces/tr-nope").await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

// ---------------------------------------------------------------------------
// getTrace with spans
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_trace_with_spans_builds_otel_spans() {
    let server = TestServer::start("get_trace_spans").await;
    let trace_id = "tr-spans";
    start_trace(&server, trace_id, EXP_ID, "OK").await;

    // Mark the trace as TRACKING_STORE-backed and insert one span row whose
    // content is the mlflow span dict (base64 ids, OTLP status-name string,
    // string attributes).
    server
        .store
        .set_trace_tag(
            "default",
            trace_id,
            "mlflow.trace.spansLocation",
            "TRACKING_STORE",
        )
        .await
        .unwrap();

    // trace_id (16 bytes) / span_id (8 bytes) base64 of arbitrary bytes.
    let content = json!({
        "trace_id": "AAAAAAAAAAAAAAAAAAAAAA==",
        "span_id": "AAAAAAAAAAE=",
        "parent_span_id": null,
        "name": "root-span",
        "start_time_unix_nano": 1000,
        "end_time_unix_nano": 2000,
        "events": [],
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {"mlflow.spanType": "\"LLM\""},
        "links": []
    })
    .to_string();
    insert_span(&server.store, trace_id, "0000000000000001", &content).await;

    let body = json!({"trace_id": trace_id, "allow_partial": true}).to_string();
    let res = get_body(&server, "/api/3.0", "/mlflow/traces/get", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let trace = &res.json()["trace"];
    assert_eq!(trace["trace_info"]["trace_id"], trace_id);
    let spans = trace["spans"].as_array().expect("spans array");
    assert_eq!(spans.len(), 1, "{}", res.body);
    assert_eq!(spans[0]["name"], "root-span");
    // OTLP status code serializes as its enum name in the JSON codec.
    assert_eq!(spans[0]["status"]["code"], "STATUS_CODE_OK");
}

#[tokio::test]
async fn get_trace_missing_is_resource_does_not_exist() {
    let server = TestServer::start("get_trace_missing").await;
    let body = json!({"trace_id": "tr-nope", "allow_partial": true}).to_string();
    let res = get_body(&server, "/api/3.0", "/mlflow/traces/get", &body).await;
    assert_eq!(res.status, StatusCode::NOT_FOUND, "{}", res.body);
    assert_eq!(res.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

#[tokio::test]
async fn get_trace_requires_trace_id() {
    let server = TestServer::start("get_trace_no_id").await;
    let res = get_body(&server, "/api/3.0", "/mlflow/traces/get", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// batchGetTraces / batchGetTraceInfos
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batch_get_trace_infos_preserves_order() {
    let server = TestServer::start("batch_infos").await;
    start_trace(&server, "b-1", EXP_ID, "OK").await;
    start_trace(&server, "b-2", EXP_ID, "ERROR").await;

    let body = json!({"trace_ids": ["b-2", "b-1"]}).to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/batchGetInfos", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let infos = res.json()["trace_infos"].as_array().unwrap().clone();
    assert_eq!(infos.len(), 2);
    assert_eq!(infos[0]["trace_id"], "b-2");
    assert_eq!(infos[1]["trace_id"], "b-1");
}

#[tokio::test]
async fn batch_get_trace_infos_requires_ids() {
    let server = TestServer::start("batch_infos_empty").await;
    let res = post(&server, "/api/3.0", "/mlflow/traces/batchGetInfos", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// searchTracesV3
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_traces_v3_filters_and_paginates() {
    let server = TestServer::start("search").await;
    for i in 0..3 {
        start_trace(&server, &format!("s-{i}"), EXP_ID, "OK").await;
    }

    let body = json!({
        "locations": [{"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": EXP_ID}}],
        "max_results": 2
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/search", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let page1 = res.json();
    assert_eq!(page1["traces"].as_array().unwrap().len(), 2);
    let token = page1["next_page_token"].as_str().expect("next page token");

    let body2 = json!({
        "locations": [{"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": EXP_ID}}],
        "max_results": 2,
        "page_token": token
    })
    .to_string();
    let res2 = post(&server, "/api/3.0", "/mlflow/traces/search", &body2).await;
    assert_eq!(res2.status, StatusCode::OK, "{}", res2.body);
    assert_eq!(res2.json()["traces"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn search_traces_v3_max_results_violation() {
    let server = TestServer::start("search_max").await;
    let body = json!({
        "locations": [{"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": EXP_ID}}],
        "max_results": 501
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/search", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert_eq!(
        res.json()["message"],
        "Invalid value 501 for parameter 'max_results' supplied. \
         See the API docs for more information about request parameters."
    );
}

#[tokio::test]
async fn search_traces_v3_requires_locations() {
    let server = TestServer::start("search_no_loc").await;
    let res = post(&server, "/api/3.0", "/mlflow/traces/search", "{}").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// deleteTracesV3
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_traces_by_ids() {
    let server = TestServer::start("delete_ids").await;
    start_trace(&server, "d-1", EXP_ID, "OK").await;
    start_trace(&server, "d-2", EXP_ID, "OK").await;

    let body = json!({"experiment_id": EXP_ID, "request_ids": ["d-1", "d-2"]}).to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/delete-traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["traces_deleted"], 2);
}

#[tokio::test]
async fn delete_traces_by_timestamp() {
    let server = TestServer::start("delete_ts").await;
    start_trace(&server, "dt-1", EXP_ID, "OK").await;

    // request_time was 2024-01-01 → ms far below this bound.
    let body = json!({
        "experiment_id": EXP_ID,
        "max_timestamp_millis": 9999999999999i64,
        "max_traces": 10
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/delete-traces", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    assert_eq!(res.json()["traces_deleted"], 1);
}

#[tokio::test]
async fn delete_traces_rejects_both_modes() {
    let server = TestServer::start("delete_both").await;
    let body = json!({
        "experiment_id": EXP_ID,
        "max_timestamp_millis": 1,
        "request_ids": ["x"]
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/delete-traces", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// setTraceTagV3 / deleteTraceTagV3 (path params)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn trace_tag_set_and_delete() {
    let server = TestServer::start("tag_crud").await;
    let trace_id = "tag-1";
    start_trace(&server, trace_id, EXP_ID, "OK").await;

    let set = send(
        &server,
        Method::PATCH,
        &format!("/api/3.0/mlflow/traces/{trace_id}/tags"),
        Some(&json!({"key": "topic", "value": "billing"}).to_string()),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "{}", set.body);

    // Verify the tag is present on the info.
    let info = get_q(&server, "/api/3.0", &format!("/mlflow/traces/{trace_id}")).await;
    assert_eq!(
        info.json()["trace"]["trace_info"]["tags"]["topic"],
        "billing"
    );

    // The client sends the key in the JSON body (message_to_json), even for DELETE.
    let del = send(
        &server,
        Method::DELETE,
        &format!("/api/3.0/mlflow/traces/{trace_id}/tags"),
        Some(&json!({"key": "topic"}).to_string()),
    )
    .await;
    assert_eq!(del.status, StatusCode::OK, "{}", del.body);

    // Deleting a missing tag is RESOURCE_DOES_NOT_EXIST.
    let del2 = send(
        &server,
        Method::DELETE,
        &format!("/api/3.0/mlflow/traces/{trace_id}/tags"),
        Some(&json!({"key": "topic"}).to_string()),
    )
    .await;
    assert_eq!(del2.status, StatusCode::NOT_FOUND, "{}", del2.body);
    assert_eq!(del2.json()["error_code"], "RESOURCE_DOES_NOT_EXIST");
}

// ---------------------------------------------------------------------------
// linkTracesToRun / linkPromptsToTrace
// ---------------------------------------------------------------------------

#[tokio::test]
async fn link_traces_to_run_limit_error() {
    let server = TestServer::start("link_limit").await;
    let trace_ids: Vec<String> = (0..101).map(|i| format!("lt-{i}")).collect();
    let body = json!({"trace_ids": trace_ids, "run_id": "some-run"}).to_string();
    let res = post(&server, "/api/2.0", "/mlflow/traces/link-to-run", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(
        res.body.contains("Cannot link more than 100"),
        "{}",
        res.body
    );
}

#[tokio::test]
async fn link_prompts_to_trace_ok() {
    let server = TestServer::start("link_prompts").await;
    let trace_id = "lp-1";
    start_trace(&server, trace_id, EXP_ID, "OK").await;

    let body = json!({
        "trace_id": trace_id,
        "prompt_versions": [{"name": "greeting", "version": "3"}]
    })
    .to_string();
    let res = post(&server, "/api/2.0", "/mlflow/traces/link-prompts", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
}

// ---------------------------------------------------------------------------
// calculateTraceFilterCorrelation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn calculate_correlation_basic() {
    let server = TestServer::start("correlation").await;
    // 4 OK traces, 2 with an "err" tag.
    for i in 0..4 {
        let trace_id = format!("c-{i}");
        start_trace(&server, &trace_id, EXP_ID, "OK").await;
        if i < 2 {
            server
                .store
                .set_trace_tag("default", &trace_id, "flag", "err")
                .await
                .unwrap();
        }
    }

    let body = json!({
        "experiment_ids": [EXP_ID],
        "filter_string1": "trace.status = 'OK'",
        "filter_string2": "tag.flag = 'err'"
    })
    .to_string();
    let res = post(
        &server,
        "/api/3.0",
        "/mlflow/traces/calculate-filter-correlation",
        &body,
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let j = res.json();
    assert_eq!(j["total_count"], 4);
    assert_eq!(j["filter1_count"], 4);
    assert_eq!(j["filter2_count"], 2);
    assert_eq!(j["joint_count"], 2);
}

#[tokio::test]
async fn calculate_correlation_requires_filters() {
    let server = TestServer::start("correlation_missing").await;
    let body =
        json!({"experiment_ids": [EXP_ID], "filter_string1": "trace.status = 'OK'"}).to_string();
    let res = post(
        &server,
        "/api/3.0",
        "/mlflow/traces/calculate-filter-correlation",
        &body,
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

// ---------------------------------------------------------------------------
// queryTraceMetrics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn query_metrics_trace_count_by_status() {
    let server = TestServer::start("metrics_count").await;
    start_trace(&server, "m-1", EXP_ID, "OK").await;
    start_trace(&server, "m-2", EXP_ID, "OK").await;
    start_trace(&server, "m-3", EXP_ID, "ERROR").await;

    let body = json!({
        "experiment_ids": [EXP_ID],
        "view_type": "TRACES",
        "metric_name": "trace_count",
        "aggregations": [{"aggregation_type": "COUNT"}],
        "dimensions": ["trace_status"]
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/metrics", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let points = res.json()["data_points"].as_array().unwrap().clone();
    // Two groups: OK (2) and ERROR (1).
    assert_eq!(points.len(), 2, "{}", res.body);
    let ok = points
        .iter()
        .find(|p| p["dimensions"]["trace_status"] == "OK")
        .unwrap();
    assert_eq!(ok["values"]["COUNT"], 2.0);
    let err = points
        .iter()
        .find(|p| p["dimensions"]["trace_status"] == "ERROR")
        .unwrap();
    assert_eq!(err["values"]["COUNT"], 1.0);
}

#[tokio::test]
async fn query_metrics_global_no_dimensions() {
    let server = TestServer::start("metrics_global").await;
    start_trace(&server, "g-1", EXP_ID, "OK").await;
    start_trace(&server, "g-2", EXP_ID, "OK").await;

    let body = json!({
        "experiment_ids": [EXP_ID],
        "view_type": "TRACES",
        "metric_name": "trace_count",
        "aggregations": [{"aggregation_type": "COUNT"}]
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/metrics", &body).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.body);
    let points = res.json()["data_points"].as_array().unwrap().clone();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0]["values"]["COUNT"], 2.0);
}

#[tokio::test]
async fn query_metrics_invalid_metric_name() {
    let server = TestServer::start("metrics_bad").await;
    let body = json!({
        "experiment_ids": [EXP_ID],
        "view_type": "TRACES",
        "metric_name": "nonsense",
        "aggregations": [{"aggregation_type": "COUNT"}]
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/metrics", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert_eq!(res.json()["error_code"], "INVALID_PARAMETER_VALUE");
}

#[tokio::test]
async fn query_metrics_percentile_deferred() {
    let server = TestServer::start("metrics_pct").await;
    start_trace(&server, "p-1", EXP_ID, "OK").await;
    let body = json!({
        "experiment_ids": [EXP_ID],
        "view_type": "TRACES",
        "metric_name": "latency",
        "aggregations": [{"aggregation_type": "PERCENTILE", "percentile_value": 95}]
    })
    .to_string();
    let res = post(&server, "/api/3.0", "/mlflow/traces/metrics", &body).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.body);
    assert!(res.body.contains("not yet supported"), "{}", res.body);
}

// ---------------------------------------------------------------------------
// Span insert helper
// ---------------------------------------------------------------------------

async fn insert_span(store: &TrackingStore, trace_id: &str, span_id: &str, content: &str) {
    use mlflow_store::{SpanInput, TraceTimeRange};
    let span = SpanInput {
        trace_id: trace_id.to_string(),
        span_id: span_id.to_string(),
        parent_span_id: None,
        name: Some("root-span".to_string()),
        span_type: Some("LLM".to_string()),
        status: "OK".to_string(),
        start_time_unix_nano: 1000,
        end_time_unix_nano: Some(2000),
        content: content.to_string(),
        dimension_attributes: None,
    };
    let range = TraceTimeRange {
        trace_id: trace_id.to_string(),
        min_start_ms: 0,
        max_end_ms: Some(0),
        root_span_status: Some("OK".to_string()),
    };
    store
        .log_spans("default", EXP_ID, &[span], &[], &[range])
        .await
        .expect("log_spans");
}
