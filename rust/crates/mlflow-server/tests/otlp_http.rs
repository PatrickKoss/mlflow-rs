//! HTTP integration tests for `POST /v1/traces` (plan T4.3, §3.8).
//!
//! Same real-socket harness pattern as `metrics_http.rs` / `runs_http.rs`
//! (temp copy of the committed fixture DB, a real `axum::serve` listener).
//! Covers: protobuf happy path, JSON happy path, gzip `Content-Encoding`,
//! missing/invalid `x-mlflow-experiment-id`, malformed payloads (400 vs 422
//! split per `mlflow/server/otel_api.py`), run-id linking, and a
//! differential-style assertion that the persisted span row matches what
//! `Span.to_dict()`/`log_spans` would produce for the same OTel span (base64
//! ids, `STATUS_CODE_*` in the JSON blob vs `OK`/`ERROR`/`UNSET` in the
//! `status` column, `tr-<hex>` trace id).

use std::io::Write;
use std::path::{Path, PathBuf};

use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_proto::opentelemetry::proto::collector::trace::v1::ExportTraceServiceRequest;
use mlflow_proto::opentelemetry::proto::common::v1::{any_value, AnyValue, KeyValue};
use mlflow_proto::opentelemetry::proto::resource::v1::Resource;
use mlflow_proto::opentelemetry::proto::trace::v1::{ResourceSpans, ScopeSpans, Span, Status};
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore};
use prost::Message;
use serde_json::Value;
use tokio::net::TcpListener;

const DEFAULT_WORKSPACE: &str = "default";
const DEFAULT_EXPERIMENT_ID: &str = "0";

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
            "mlflow_rust_server_otlp_{}_{}_{}.db",
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
        let store = TrackingStore::new(db, "s3://bucket/mlruns");
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            static_prefix: None,
            backend_store_uri: None,
            default_artifact_root: None,
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
    content_type: Option<String>,
}

impl HttpResponse {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| {
            panic!(
                "body is not JSON: {e}: {}",
                String::from_utf8_lossy(&self.body)
            )
        })
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

async fn send(
    server: &TestServer,
    method: Method,
    path: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> HttpResponse {
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url = format!("{}{path}", server.base);
    let mut builder = Request::builder().method(method.clone()).uri(&url);
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let request = builder
        .body(Full::<Bytes>::new(Bytes::from(body.clone())))
        .unwrap();

    let mut last = None;
    for _ in 0..50 {
        let mut retry_builder = Request::builder().method(method.clone()).uri(&url);
        for (k, v) in headers {
            retry_builder = retry_builder.header(*k, *v);
        }
        let retry_request = retry_builder
            .body(Full::<Bytes>::new(Bytes::from(body.clone())))
            .unwrap();
        match client.request(retry_request).await {
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
                    body: bytes.to_vec(),
                    content_type,
                };
            }
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }
    let _ = request;
    panic!("failed to connect: {last:?}");
}

// ---------------------------------------------------------------------------
// Test fixture builders
// ---------------------------------------------------------------------------

fn string_attr(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.to_string())),
        }),
    }
}

/// Build a minimal single-root-span `ExportTraceServiceRequest`.
fn build_request(trace_id: [u8; 16], span_id: [u8; 8], name: &str) -> ExportTraceServiceRequest {
    ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: Some(Resource {
                attributes: vec![string_attr("service.name", "claude-code")],
                ..Default::default()
            }),
            scope_spans: vec![ScopeSpans {
                scope: None,
                spans: vec![Span {
                    trace_id: trace_id.to_vec(),
                    span_id: span_id.to_vec(),
                    name: name.to_string(),
                    start_time_unix_nano: 1_700_000_000_000_000_000,
                    end_time_unix_nano: 1_700_000_000_500_000_000,
                    status: Some(Status {
                        code: 1, // STATUS_CODE_OK
                        message: String::new(),
                    }),
                    attributes: vec![string_attr("custom.key", "custom-value")],
                    ..Default::default()
                }],
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    }
}

fn expected_trace_id_hex(trace_id: [u8; 16]) -> String {
    trace_id.iter().map(|b| format!("{b:02x}")).collect()
}

fn expected_span_id_hex(span_id: [u8; 8]) -> String {
    span_id.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Happy paths
// ---------------------------------------------------------------------------

#[tokio::test]
async fn protobuf_happy_path_persists_spans() {
    let server = TestServer::start("pb_happy").await;
    let trace_id = [0xAA; 16];
    let span_id = [0xBB; 8];
    let request = build_request(trace_id, span_id, "root");
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        body,
    )
    .await;

    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.content_type.as_deref(), Some("application/x-protobuf"));

    let expected_trace_id = format!("tr-{}", expected_trace_id_hex(trace_id));
    let info = server
        .store
        .get_trace_info(DEFAULT_WORKSPACE, &expected_trace_id)
        .await
        .expect("trace should exist");
    assert_eq!(info.state, "OK");

    let traces = server
        .store
        .batch_get_traces(DEFAULT_WORKSPACE, std::slice::from_ref(&expected_trace_id))
        .await
        .unwrap();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].spans.len(), 1);
    let span_row = &traces[0].spans[0];
    assert_eq!(span_row.span_id, expected_span_id_hex(span_id));
    assert_eq!(span_row.status, "OK");
    assert!(span_row.parent_span_id.is_none());
}

#[tokio::test]
async fn json_happy_path_persists_spans() {
    let server = TestServer::start("json_happy").await;
    let trace_id_hex = "0102030405060708090a0b0c0d0e0f10";
    let span_id_hex = "0102030405060708";
    let body = serde_json::json!({
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": [{
                    "traceId": trace_id_hex,
                    "spanId": span_id_hex,
                    "name": "root",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000000500000000",
                    "status": {"code": "STATUS_CODE_OK"}
                }]
            }]
        }]
    })
    .to_string();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/json"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        body.into_bytes(),
    )
    .await;

    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(res.content_type.as_deref(), Some("application/json"));
    assert_eq!(res.json(), serde_json::json!({}));

    let expected_trace_id = format!("tr-{trace_id_hex}");
    let info = server
        .store
        .get_trace_info(DEFAULT_WORKSPACE, &expected_trace_id)
        .await
        .expect("trace should exist");
    assert_eq!(info.state, "OK");
}

#[tokio::test]
async fn gzip_content_encoding_is_decompressed() {
    let server = TestServer::start("gzip").await;
    let trace_id = [0xCC; 16];
    let span_id = [0xDD; 8];
    let request = build_request(trace_id, span_id, "root");
    let raw = request.encode_to_vec();

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&raw).unwrap();
    let compressed = encoder.finish().unwrap();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("content-encoding", "gzip"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        compressed,
    )
    .await;

    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let expected_trace_id = format!("tr-{}", expected_trace_id_hex(trace_id));
    server
        .store
        .get_trace_info(DEFAULT_WORKSPACE, &expected_trace_id)
        .await
        .expect("trace should exist after gzip decompression");
}

// ---------------------------------------------------------------------------
// Header / content-type validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn missing_experiment_id_header_is_422() {
    let server = TestServer::start("missing_exp_header").await;
    let request = build_request([1; 16], [2; 8], "root");
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[("content-type", "application/x-protobuf")],
        body,
    )
    .await;

    assert_eq!(
        res.status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "{}",
        res.text()
    );
}

#[tokio::test]
async fn invalid_experiment_id_returns_mlflow_error_shape() {
    let server = TestServer::start("invalid_exp").await;
    let request = build_request([1; 16], [2; 8], "root");
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", "does-not-exist"),
        ],
        body,
    )
    .await;

    // Non-numeric experiment id -> store rejects it as an `MlflowError`,
    // which takes the `except MlflowException` passthrough branch
    // (otel_api.py:221-225): MLflow-shaped JSON body, not `{"detail": ...}`.
    assert!(res.status.is_client_error(), "{}", res.text());
    let json = res.json();
    assert!(json.get("error_code").is_some(), "{}", res.text());
}

#[tokio::test]
async fn invalid_content_type_is_400_with_detail_body() {
    let server = TestServer::start("bad_content_type").await;
    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "text/plain"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        b"hello".to_vec(),
    )
    .await;

    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    assert_eq!(res.content_type.as_deref(), Some("application/json"));
    let json = res.json();
    assert!(json["detail"]
        .as_str()
        .unwrap()
        .starts_with("Invalid Content-Type"));
}

// ---------------------------------------------------------------------------
// Malformed payloads: 400 vs 422 split
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_protobuf_body_is_400() {
    let server = TestServer::start("malformed_pb").await;
    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
    )
    .await;

    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    let json = res.json();
    assert_eq!(json["detail"], "Invalid OpenTelemetry format");
}

#[tokio::test]
async fn malformed_json_body_is_400() {
    let server = TestServer::start("malformed_json").await;
    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/json"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        b"not json at all".to_vec(),
    )
    .await;

    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    let json = res.json();
    assert_eq!(json["detail"], "Invalid OpenTelemetry format");
}

#[tokio::test]
async fn empty_resource_spans_is_400_no_spans_found() {
    let server = TestServer::start("no_spans").await;
    let request = ExportTraceServiceRequest {
        resource_spans: vec![],
    };
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        body,
    )
    .await;

    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    let json = res.json();
    assert_eq!(
        json["detail"],
        "Invalid OpenTelemetry format - no spans found"
    );
}

#[tokio::test]
async fn span_with_empty_span_id_is_422() {
    let server = TestServer::start("bad_span").await;
    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: None,
            scope_spans: vec![ScopeSpans {
                scope: None,
                spans: vec![Span {
                    trace_id: vec![1; 16],
                    span_id: vec![], // invalid: empty span_id
                    name: "bad".to_string(),
                    start_time_unix_nano: 1,
                    ..Default::default()
                }],
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    };
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        body,
    )
    .await;

    assert_eq!(
        res.status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "{}",
        res.text()
    );
    let json = res.json();
    assert_eq!(
        json["detail"],
        "Cannot convert OpenTelemetry span to MLflow span"
    );
}

// ---------------------------------------------------------------------------
// Run-id linking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_id_header_links_completed_trace_to_run() {
    let server = TestServer::start("run_link").await;
    let run_id = server.create_run("otlp-run").await;

    let trace_id = [0x11; 16];
    let span_id = [0x22; 8];
    let request = build_request(trace_id, span_id, "root");
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
            ("x-mlflow-run-id", &run_id),
        ],
        body,
    )
    .await;

    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    let expected_trace_id = format!("tr-{}", expected_trace_id_hex(trace_id));
    // `link_traces_to_run` records the link in `entity_associations`, which
    // `search_traces`'s `run_id = '<run>'` filter resolves via a link-OR-
    // metadata predicate (`traces_search.rs:537-553`) — the store-level way
    // to observe the link without reaching into internal tables directly.
    let page = server
        .store
        .search_traces(
            DEFAULT_WORKSPACE,
            &[DEFAULT_EXPERIMENT_ID.to_string()],
            Some(&format!("run_id = '{run_id}'")),
            10,
            &[],
            None,
        )
        .await
        .unwrap();
    assert!(
        page.trace_infos
            .iter()
            .any(|t| t.trace_id == expected_trace_id),
        "trace should be linked to the run via entity_associations"
    );
}

#[tokio::test]
async fn run_id_header_does_not_link_non_root_only_batches() {
    // A batch whose only span is non-root (no root span present) has no
    // "completed" trace id, so linking is a no-op even with a run-id header.
    let server = TestServer::start("run_link_child_only").await;
    let run_id = server.create_run("otlp-run-child-only").await;

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: None,
            scope_spans: vec![ScopeSpans {
                scope: None,
                spans: vec![Span {
                    trace_id: vec![0x33; 16],
                    span_id: vec![0x44; 8],
                    parent_span_id: vec![0x55; 8],
                    name: "child".to_string(),
                    start_time_unix_nano: 1_700_000_000_000_000_000,
                    end_time_unix_nano: 1_700_000_000_500_000_000,
                    ..Default::default()
                }],
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    };
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
            ("x-mlflow-run-id", &run_id),
        ],
        body,
    )
    .await;

    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    let expected_trace_id = "tr-33333333333333333333333333333333".to_string();
    server
        .store
        .get_trace_info(DEFAULT_WORKSPACE, &expected_trace_id)
        .await
        .expect("trace should still be created even though it's not linked");

    let page = server
        .store
        .search_traces(
            DEFAULT_WORKSPACE,
            &[DEFAULT_EXPERIMENT_ID.to_string()],
            Some(&format!("run_id = '{run_id}'")),
            10,
            &[],
            None,
        )
        .await
        .unwrap();
    assert!(
        !page
            .trace_infos
            .iter()
            .any(|t| t.trace_id == expected_trace_id),
        "a batch with no root span must not be linked to the run"
    );
}

// ---------------------------------------------------------------------------
// Differential-style content assertions
// ---------------------------------------------------------------------------

/// Asserts the persisted span row's `content` JSON matches the exact shape
/// `Span.to_dict()` / `log_spans` would produce for the same OTel span:
/// base64 trace/span ids, `STATUS_CODE_OK` (not `OK`) inside the blob, while
/// the `status` DB column (surfaced indirectly via `TraceInfo.state` for the
/// root span) uses the plain `OK`/`ERROR`/`UNSET` form.
#[tokio::test]
async fn persisted_span_content_matches_span_to_dict_shape() {
    let server = TestServer::start("content_shape").await;
    let trace_id = [0x01; 16];
    let span_id = [0x02; 8];
    let request = build_request(trace_id, span_id, "root-op");
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        body,
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    let expected_trace_id = format!("tr-{}", expected_trace_id_hex(trace_id));
    let traces = server
        .store
        .batch_get_traces(DEFAULT_WORKSPACE, &[expected_trace_id])
        .await
        .unwrap();
    let span_row = &traces[0].spans[0];

    let content: Value = serde_json::from_str(&span_row.content).unwrap();
    // Base64(big-endian bytes), matching `Span.to_dict()` (span.py:306-337) —
    // NOT the "tr-<hex>"/hex form used for the DB column/FK.
    assert_eq!(
        content["trace_id"],
        base64::engine::general_purpose::STANDARD.encode(trace_id)
    );
    assert_eq!(
        content["span_id"],
        base64::engine::general_purpose::STANDARD.encode(span_id)
    );
    assert!(content["parent_span_id"].is_null());
    assert_eq!(content["name"], "root-op");
    // OTel-proto enum NAME in the JSON blob (span_status.py's
    // `to_otel_proto_status_code_name`), distinct from the plain "OK" stored
    // in `spans.status` (asserted separately below).
    assert_eq!(content["status"]["code"], "STATUS_CODE_OK");
    // Attribute values in `content.attributes` are JSON-*encoded* strings
    // (`dump_span_attribute_value`'s output, copied verbatim by
    // `Span.to_dict()`), so a plain string attribute is double-quoted here.
    assert_eq!(content["attributes"]["custom.key"], "\"custom-value\"");
    // service.name propagated onto the root span's attributes
    // (otel_api.py:192-201).
    assert_eq!(content["attributes"]["service.name"], "\"claude-code\"");

    // Plain SpanStatusCode DB column value.
    assert_eq!(span_row.status, "OK");
    assert_eq!(span_row.span_id, expected_span_id_hex(span_id));
}

#[tokio::test]
async fn multi_span_trace_persists_parent_child_relationship() {
    let server = TestServer::start("parent_child").await;
    let trace_id = [0x77; 16];
    let root_span_id = [0x01; 8];
    let child_span_id = [0x02; 8];

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: None,
            scope_spans: vec![ScopeSpans {
                scope: None,
                spans: vec![
                    Span {
                        trace_id: trace_id.to_vec(),
                        span_id: root_span_id.to_vec(),
                        name: "root".to_string(),
                        start_time_unix_nano: 1_700_000_000_000_000_000,
                        end_time_unix_nano: 1_700_000_000_900_000_000,
                        status: Some(Status {
                            code: 1,
                            message: String::new(),
                        }),
                        ..Default::default()
                    },
                    Span {
                        trace_id: trace_id.to_vec(),
                        span_id: child_span_id.to_vec(),
                        parent_span_id: root_span_id.to_vec(),
                        name: "child".to_string(),
                        start_time_unix_nano: 1_700_000_000_100_000_000,
                        end_time_unix_nano: 1_700_000_000_400_000_000,
                        status: Some(Status {
                            code: 2, // STATUS_CODE_ERROR
                            message: "child failed".to_string(),
                        }),
                        ..Default::default()
                    },
                ],
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    };
    let body = request.encode_to_vec();

    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        body,
    )
    .await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());

    let expected_trace_id = format!("tr-{}", expected_trace_id_hex(trace_id));
    // Trace status derives from the ROOT span only (OK), even though the
    // child errored (`_get_trace_status_from_root_span`).
    let info = server
        .store
        .get_trace_info(DEFAULT_WORKSPACE, &expected_trace_id)
        .await
        .unwrap();
    assert_eq!(info.state, "OK");

    let traces = server
        .store
        .batch_get_traces(DEFAULT_WORKSPACE, &[expected_trace_id])
        .await
        .unwrap();
    assert_eq!(traces[0].spans.len(), 2);
    let child = traces[0]
        .spans
        .iter()
        .find(|s| s.span_id == expected_span_id_hex(child_span_id))
        .unwrap();
    assert_eq!(
        child.parent_span_id.as_deref(),
        Some(expected_span_id_hex(root_span_id).as_str())
    );
    assert_eq!(child.status, "ERROR");
}

#[tokio::test]
async fn unsupported_content_encoding_is_400() {
    let server = TestServer::start("bad_encoding").await;
    let res = send(
        &server,
        Method::POST,
        "/v1/traces",
        &[
            ("content-type", "application/x-protobuf"),
            ("content-encoding", "br"),
            ("x-mlflow-experiment-id", DEFAULT_EXPERIMENT_ID),
        ],
        b"whatever".to_vec(),
    )
    .await;

    assert_eq!(res.status, StatusCode::BAD_REQUEST, "{}", res.text());
    let json = res.json();
    assert_eq!(json["detail"], "Unsupported Content-Encoding: br");
}
