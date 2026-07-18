//! Hermetic T18.3 gateway runtime coverage. Every provider call targets the
//! local mock below; no test or CI path can reach a live provider.

use std::collections::HashMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, Method, Response, StatusCode};
use axum::routing::post;
use axum::Router;
use futures::{stream, StreamExt};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::gateway_provider_matrix::token_cost;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{
    Db, EndpointModelConfig, FallbackConfig, PoolConfig, TrackingStore, WORKSPACE_DEFAULT_NAME,
};
use mlflow_webhooks::http_send::SendConfig;
use mlflow_webhooks::{
    WebhookAction, WebhookDispatcher, WebhookEntity, WebhookEvent, WebhookStatus, WebhookStore,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tower::ServiceExt;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

fn python_routing_oracle() -> &'static Value {
    static ORACLE: OnceLock<Value> = OnceLock::new();
    ORACLE.get_or_init(|| {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("repository root");
        let output = Command::new("uv")
            .args([
                "run",
                "--frozen",
                "python",
                "rust/tools/gateway_routing_oracle.py",
            ])
            .current_dir(repository)
            .output()
            .expect("run Python gateway routing oracle");
        assert!(
            output.status.success(),
            "Python oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("parse Python gateway routing oracle")
    })
}

struct Fixture {
    _directory: tempfile::TempDir,
    app: Router,
    store: TrackingStore,
    trace_experiment_id: String,
    webhook_store: WebhookStore,
    webhook_base: String,
    webhook_deliveries: Arc<Mutex<Vec<Value>>>,
    mock_base: String,
    attempts: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct MockState {
    attempts: Arc<Mutex<Vec<String>>>,
}

impl Fixture {
    async fn new() -> Self {
        static WEBHOOK_ENV: std::sync::Once = std::sync::Once::new();
        WEBHOOK_ENV.call_once(|| {
            std::env::set_var("MLFLOW_WEBHOOK_ALLOWED_SCHEMES", "http,https");
            std::env::set_var("MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS", "true");
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mock_base = format!("http://{}", listener.local_addr().unwrap());
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let mock_state = MockState {
            attempts: attempts.clone(),
        };
        tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .fallback(post(mock_provider))
                    .with_state(mock_state),
            )
            .await
            .unwrap();
        });

        let webhook_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let webhook_base = format!("http://{}", webhook_listener.local_addr().unwrap());
        let webhook_deliveries = Arc::new(Mutex::new(Vec::new()));
        let webhook_sink = webhook_deliveries.clone();
        tokio::spawn(async move {
            axum::serve(
                webhook_listener,
                Router::new()
                    .fallback(post(record_webhook))
                    .with_state(webhook_sink),
            )
            .await
            .unwrap();
        });

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gateway-runtime.db");
        std::fs::copy(fixture_path(), &path).unwrap();
        let db = Db::connect(
            &format!("sqlite:///{}", path.display()),
            PoolConfig::default(),
        )
        .await
        .unwrap();
        let store = TrackingStore::new(
            db.clone(),
            directory.path().join("artifacts").display().to_string(),
        );
        let webhook_store = WebhookStore::new(db).unwrap();
        let mut secret_ids = HashMap::new();
        for provider in ["openai", "azure", "anthropic", "gemini"] {
            secret_ids.insert(
                provider.to_string(),
                seed_endpoint(&store, provider, &mock_base).await,
            );
        }
        seed_routing_endpoints(&store, &secret_ids).await;
        let trace_experiment_id = store
            .create_experiment(WORKSPACE_DEFAULT_NAME, "gateway-runtime-traces", None, &[])
            .await
            .unwrap();
        let traced_model = store
            .create_gateway_model_definition(
                WORKSPACE_DEFAULT_NAME,
                "traced-openai-definition",
                &secret_ids["openai"],
                "openai",
                "gpt-4",
                Some("runtime-tracing-test"),
            )
            .await
            .unwrap();
        store
            .create_gateway_endpoint(
                WORKSPACE_DEFAULT_NAME,
                "traced-openai-endpoint",
                &[EndpointModelConfig {
                    model_definition_id: traced_model.model_definition_id,
                    linkage_type: "PRIMARY".to_string(),
                    weight: 1.0,
                    fallback_order: None,
                }],
                Some("runtime-tracing-test"),
                None,
                None,
                Some(&trace_experiment_id),
                true,
            )
            .await
            .unwrap();
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let dispatcher = WebhookDispatcher::with_config(
            webhook_store.clone(),
            WORKSPACE_DEFAULT_NAME,
            Arc::new(mlflow_webhooks::SystemResolver),
            SendConfig {
                timeout: std::time::Duration::from_secs(5),
                max_retries: 0,
                backoff_factor: 0.0,
                backoff_max: std::time::Duration::ZERO,
                backoff_jitter: 0.0,
                allow_private_ips: true,
            },
        );
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(
                AppState::new(store.clone()).with_webhook_store(webhook_store.clone(), dispatcher),
            ),
        );
        Self {
            _directory: directory,
            app,
            store,
            trace_experiment_id,
            webhook_store,
            webhook_base,
            webhook_deliveries,
            mock_base,
            attempts,
        }
    }

    async fn post(&self, endpoint: &str, body: Value) -> Response<Body> {
        self.app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/gateway/{endpoint}/mlflow/invocations"))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn take_attempts(&self) -> Vec<String> {
        std::mem::take(&mut *self.attempts.lock().unwrap())
    }
}

async fn seed_endpoint(store: &TrackingStore, provider: &str, base: &str) -> String {
    let secret = store
        .create_gateway_secret(
            WORKSPACE_DEFAULT_NAME,
            &format!("obvious-fake-{provider}-secret"),
            &HashMap::from([(
                "api_key".to_string(),
                format!("obvious-fake-{provider}-key"),
            )]),
            Some(provider),
            &auth_config(provider, base),
            Some("runtime-test"),
        )
        .await
        .unwrap();
    let secret_id = secret.secret_id.clone();
    let model = store
        .create_gateway_model_definition(
            WORKSPACE_DEFAULT_NAME,
            &format!("{provider}-definition"),
            &secret.secret_id,
            provider,
            &format!("{provider}-fixture-model"),
            Some("runtime-test"),
        )
        .await
        .unwrap();
    store
        .create_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            &format!("{provider}-endpoint"),
            &[EndpointModelConfig {
                model_definition_id: model.model_definition_id,
                linkage_type: "PRIMARY".to_string(),
                weight: 1.0,
                fallback_order: None,
            }],
            Some("runtime-test"),
            None,
            None,
            None,
            false,
        )
        .await
        .unwrap();
    secret_id
}

async fn seed_routing_endpoints(store: &TrackingStore, secret_ids: &HashMap<String, String>) {
    let zero = seed_script_model(store, "traffic-zero", "openai", secret_ids).await;
    let hundred = seed_script_model(store, "traffic-hundred", "openai", secret_ids).await;
    create_script_endpoint(
        store,
        "traffic-zero-hundred",
        &[
            (zero, "PRIMARY", 0.0, None),
            (hundred, "PRIMARY", 1.0, None),
        ],
        Some("REQUEST_BASED_TRAFFIC_SPLIT"),
        None,
    )
    .await;

    let single = seed_script_model(store, "traffic-single", "openai", secret_ids).await;
    create_script_endpoint(
        store,
        "traffic-single",
        &[(single, "PRIMARY", 0.01, None)],
        Some("REQUEST_BASED_TRAFFIC_SPLIT"),
        None,
    )
    .await;

    let primary_500 = seed_script_model(store, "fail-500", "openai", secret_ids).await;
    let fallback_success = seed_script_model(store, "fallback-success", "openai", secret_ids).await;
    create_script_endpoint(
        store,
        "fallback-first-500",
        &[
            (primary_500.clone(), "PRIMARY", 1.0, None),
            (fallback_success.clone(), "FALLBACK", 1.0, Some(0)),
        ],
        None,
        Some(1),
    )
    .await;

    let final_429 = seed_script_model(store, "fail-429", "openai", secret_ids).await;
    create_script_endpoint(
        store,
        "fallback-all-fail",
        &[
            (primary_500.clone(), "PRIMARY", 1.0, None),
            (final_429, "FALLBACK", 1.0, Some(0)),
        ],
        None,
        Some(1),
    )
    .await;

    let second_500 = seed_script_model(store, "fail-500-second", "openai", secret_ids).await;
    let excluded_success = seed_script_model(store, "excluded-success", "openai", secret_ids).await;
    create_script_endpoint(
        store,
        "fallback-max-attempts",
        &[
            (primary_500.clone(), "PRIMARY", 1.0, None),
            (second_500, "FALLBACK", 1.0, Some(0)),
            (excluded_success, "FALLBACK", 1.0, Some(1)),
        ],
        None,
        Some(1),
    )
    .await;

    let late = seed_script_model(store, "late-order", "openai", secret_ids).await;
    let early = seed_script_model(store, "early-order", "openai", secret_ids).await;
    create_script_endpoint(
        store,
        "fallback-order",
        &[
            (primary_500.clone(), "PRIMARY", 1.0, None),
            (late, "FALLBACK", 1.0, Some(2)),
            (early, "FALLBACK", 1.0, Some(1)),
        ],
        None,
        Some(2),
    )
    .await;

    let partial = seed_script_model(store, "partial-stream", "anthropic", secret_ids).await;
    create_script_endpoint(
        store,
        "fallback-partial-stream",
        &[
            (partial, "PRIMARY", 1.0, None),
            (fallback_success, "FALLBACK", 1.0, Some(0)),
        ],
        None,
        Some(1),
    )
    .await;
}

async fn seed_script_model(
    store: &TrackingStore,
    script: &str,
    provider: &str,
    secret_ids: &HashMap<String, String>,
) -> String {
    store
        .create_gateway_model_definition(
            WORKSPACE_DEFAULT_NAME,
            &format!("{script}-definition"),
            &secret_ids[provider],
            provider,
            &format!("{script}-model"),
            Some("runtime-routing-test"),
        )
        .await
        .unwrap()
        .model_definition_id
}

async fn create_script_endpoint(
    store: &TrackingStore,
    name: &str,
    configs: &[(String, &str, f64, Option<i32>)],
    routing_strategy: Option<&str>,
    max_attempts: Option<i32>,
) {
    let configs = configs
        .iter()
        .map(
            |(model_definition_id, linkage_type, weight, fallback_order)| EndpointModelConfig {
                model_definition_id: model_definition_id.clone(),
                linkage_type: (*linkage_type).to_string(),
                weight: *weight,
                fallback_order: *fallback_order,
            },
        )
        .collect::<Vec<_>>();
    let fallback = max_attempts.map(|max_attempts| FallbackConfig {
        strategy: Some("SEQUENTIAL".to_string()),
        max_attempts: Some(max_attempts),
    });
    store
        .create_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            name,
            &configs,
            Some("runtime-routing-test"),
            routing_strategy,
            fallback.as_ref(),
            None,
            false,
        )
        .await
        .unwrap();
}

fn auth_config(provider: &str, base: &str) -> HashMap<String, String> {
    match provider {
        "openai" => HashMap::from([("api_base".to_string(), format!("{base}/v1"))]),
        "azure" => HashMap::from([
            ("api_type".to_string(), "azure".to_string()),
            ("api_base".to_string(), base.to_string()),
            ("api_version".to_string(), "2025-01-01".to_string()),
        ]),
        "anthropic" => HashMap::from([("api_base".to_string(), format!("{base}/v1"))]),
        "gemini" => HashMap::from([("api_base".to_string(), format!("{base}/v1beta/models"))]),
        _ => unreachable!(),
    }
}

async fn mock_provider(State(state): State<MockState>, request: Request) -> Response<Body> {
    let path = request.uri().path().to_string();
    let headers = request.headers().clone();
    let body: Value =
        serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let scripted_model = body
        .get("model")
        .and_then(Value::as_str)
        .and_then(|model| model.strip_suffix("-model"));
    if let Some(script) = scripted_model {
        state.attempts.lock().unwrap().push(script.to_string());
    }
    assert_eq!(
        headers.get(header::ACCEPT_ENCODING),
        Some(&HeaderValue::from_static("gzip, deflate, identity"))
    );
    assert!(headers.get("x-mlflow-authorization").is_none());
    if path.contains("anthropic") || path.ends_with("/messages") {
        assert_eq!(headers["x-api-key"], "obvious-fake-anthropic-key");
    } else if path.contains("gemini") {
        assert_eq!(headers["x-goog-api-key"], "obvious-fake-gemini-key");
    } else if path.contains("deployments") {
        assert_eq!(headers["api-key"], "obvious-fake-azure-key");
    } else {
        assert_eq!(headers["authorization"], "Bearer obvious-fake-openai-key");
    }

    let content = find_text(&body);
    if scripted_model.is_some_and(|model| model.starts_with("fail-500")) {
        return json_provider_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":{"message":"scripted primary failure"}}),
        );
    }
    if scripted_model == Some("fail-429") {
        return json_provider_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error":{"message":"scripted final limit"}}),
        );
    }
    if content == "error-429" {
        return json_provider_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error":{"message":"fixture provider limit"}}),
        );
    }
    if content == "error-500" {
        return json_provider_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":{"message":"fixture provider failure"}}),
        );
    }
    let streaming = path.contains("streamGenerateContent")
        || body.get("stream").and_then(Value::as_bool) == Some(true);
    if streaming {
        return stream_provider_response(
            &path,
            content == "mid-stream-error" || scripted_model == Some("partial-stream"),
            scripted_model == Some("partial-stream"),
        );
    }
    if path.contains("embedContent") {
        json_provider_response(StatusCode::OK, json!({"embedding":{"values":[0.25,0.5]}}))
    } else if path.ends_with("/embeddings") {
        let model = body["model"].as_str().unwrap_or("azure-fixture-model");
        json_provider_response(
            StatusCode::OK,
            json!({
                "object":"list","data":[{"object":"embedding","embedding":[0.25,0.5],"index":0}],
                "model":model,"usage":{"prompt_tokens":2,"total_tokens":2}
            }),
        )
    } else if path.ends_with("/messages") {
        json_provider_response(
            StatusCode::OK,
            json!({
                "id":"anthropic-fixture-id","model":"anthropic-fixture-model","role":"assistant",
                "content":[{"type":"text","text":"fixture answer"}],"stop_reason":"end_turn",
                "usage":{"input_tokens":2,"output_tokens":3}
            }),
        )
    } else if path.contains("generateContent") {
        json_provider_response(
            StatusCode::OK,
            json!({
                "candidates":[{"content":{"parts":[{"text":"fixture answer"}]},"finishReason":"STOP"}],
                "usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}
            }),
        )
    } else {
        let model = body["model"].as_str().unwrap_or("azure-fixture-model");
        json_provider_response(
            StatusCode::OK,
            json!({
                "id":"openai-fixture-id","object":"chat.completion","created":7,"model":model,
                "choices":[{"index":0,"message":{"role":"assistant","content":"fixture answer"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
            }),
        )
    }
}

async fn record_webhook(
    State(deliveries): State<Arc<Mutex<Vec<Value>>>>,
    body: Bytes,
) -> Response<Body> {
    deliveries
        .lock()
        .unwrap()
        .push(serde_json::from_slice(&body).unwrap());
    Response::builder()
        .status(StatusCode::OK)
        .body(Body::from("ok"))
        .unwrap()
}

fn find_text(body: &Value) -> &str {
    body.pointer("/messages/0/content")
        .or_else(|| body.pointer("/contents/0/parts/0/text"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn json_provider_response(status: StatusCode, value: Value) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}

fn stream_provider_response(
    path: &str,
    fail_midstream: bool,
    emit_before_failure: bool,
) -> Response<Body> {
    let mut chunks = if path.ends_with("/messages") {
        vec![
            Bytes::from_static(b": keep-alive\n\nevent: message_start\n"),
            Bytes::from_static(b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"anthropic-stream-id\",\"model\":\"anthropic-fixture-model\",\"usage\":{\"input_tokens\":2}}}\n\n"),
            Bytes::from_static(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"fixture \"}}\n\n"),
            Bytes::from_static(b"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n"),
        ]
    } else if path.contains("streamGenerateContent") {
        vec![
            Bytes::from_static(b": keep-alive\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"fixture \"}]}}]}\n"),
            Bytes::from_static(b"\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"answer\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":3,\"totalTokenCount\":5}}\n\ndata: [DONE]\n\n"),
        ]
    } else {
        vec![
            Bytes::from_static(b": keep-alive\n\ndata: {\"id\":\"openai-stream-id\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"fixture \"},\"finish_reason\":null}]}\n"),
            Bytes::from_static(b"\ndata: {\"id\":\"openai-stream-id\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\ndata: [DONE]\n\n"),
        ]
    };
    if fail_midstream {
        chunks.truncate(if emit_before_failure { 3 } else { 1 });
        chunks.push(Bytes::from_static(b"data: not-json\n\n"));
    }
    let body = Body::from_stream(stream::iter(chunks.into_iter().map(Ok::<_, Infallible>)));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(body)
        .unwrap()
}

fn chat(content: &str, stream: bool) -> Value {
    json!({"messages":[{"role":"user","content":content}],"stream":stream})
}

#[tokio::test]
async fn usage_tracked_invocations_persist_gateway_and_cost_spans() {
    let fixture = Fixture::new().await;
    let response = fixture
        .post("traced-openai-endpoint", chat("trace me", false))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();

    let page = fixture
        .store
        .search_traces(
            WORKSPACE_DEFAULT_NAME,
            std::slice::from_ref(&fixture.trace_experiment_id),
            None,
            10,
            &[],
            None,
        )
        .await
        .unwrap();
    assert_eq!(page.trace_infos.len(), 1);
    let trace = fixture
        .store
        .batch_get_traces(
            WORKSPACE_DEFAULT_NAME,
            &[page.trace_infos[0].trace_id.clone()],
        )
        .await
        .unwrap()
        .remove(0);
    assert_eq!(trace.spans.len(), 2);
    assert_eq!(
        trace.spans[0].name.as_deref(),
        Some("gateway/traced-openai-endpoint")
    );
    assert_eq!(
        trace.spans[1].name.as_deref(),
        Some("provider/openai/gpt-4")
    );
    let child: Value = serde_json::from_str(&trace.spans[1].content).unwrap();
    assert_eq!(
        child["attributes"]["mlflow.chat.tokenUsage"],
        Value::String(
            "{\"input_tokens\": 2, \"output_tokens\": 3, \"total_tokens\": 5}".to_string()
        )
    );
    let cost: Value =
        serde_json::from_str(child["attributes"]["mlflow.llm.cost"].as_str().unwrap()).unwrap();
    assert!(cost["total_cost"].as_f64().unwrap() > 0.0);
    assert_eq!(
        fixture
            .store
            .sum_gateway_trace_cost(0, i64::MAX, None)
            .await
            .unwrap(),
        cost["total_cost"].as_f64().unwrap()
    );

    let response = fixture
        .post("traced-openai-endpoint", chat("stream trace", true))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = response.into_body().collect().await.unwrap().to_bytes();
    assert!(stream.windows(7).any(|window| window == b"fixture"));
    let page = fixture
        .store
        .search_traces(
            WORKSPACE_DEFAULT_NAME,
            std::slice::from_ref(&fixture.trace_experiment_id),
            None,
            10,
            &[],
            None,
        )
        .await
        .unwrap();
    assert_eq!(page.trace_infos.len(), 2);
    assert_eq!(
        fixture
            .store
            .sum_gateway_trace_cost(0, i64::MAX, None)
            .await
            .unwrap(),
        2.0 * cost["total_cost"].as_f64().unwrap()
    );

    let response = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/gateway/openai/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "traced-openai-endpoint",
                        "messages": [{"role": "user", "content": "passthrough trace"}],
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();
    let page = fixture
        .store
        .search_traces(
            WORKSPACE_DEFAULT_NAME,
            std::slice::from_ref(&fixture.trace_experiment_id),
            None,
            10,
            &[],
            None,
        )
        .await
        .unwrap();
    assert_eq!(page.trace_infos.len(), 3);
    assert!(page.trace_infos.iter().any(|trace| {
        trace.metadata("mlflow.gateway.requestType") == Some("passthrough/model/openai-chat")
    }));
    assert_eq!(
        fixture
            .store
            .sum_gateway_trace_cost(0, i64::MAX, None)
            .await
            .unwrap(),
        3.0 * cost["total_cost"].as_f64().unwrap()
    );
}

#[tokio::test]
async fn reject_policy_matches_python_boundary_and_429_body() {
    let fixture = Fixture::new().await;
    let invocation_cost = token_cost("openai", "gpt-4", 2, 3).unwrap();
    fixture
        .store
        .create_budget_policy(
            WORKSPACE_DEFAULT_NAME,
            "USD",
            invocation_cost,
            "DAYS",
            1,
            "GLOBAL",
            "REJECT",
            Some("budget-test"),
        )
        .await
        .unwrap();

    // Python admits the request that reaches the boundary; the following
    // request observes cumulative_spend >= budget_amount and is rejected.
    let first = fixture
        .post("traced-openai-endpoint", chat("at boundary", false))
        .await;
    assert_eq!(first.status(), StatusCode::OK);
    let second = fixture
        .post("traced-openai-endpoint", chat("must reject", false))
        .await;
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let window = fixture
        .store
        .list_budget_windows(WORKSPACE_DEFAULT_NAME)
        .await
        .unwrap()
        .remove(0);
    let reset = chrono::DateTime::from_timestamp_millis(window.window_end_ms).unwrap();
    let detail = format!(
        "Budget limit exceeded. Limit: ${invocation_cost:.2} USD per 1 day. Budget resets at {}. Request rejected.",
        reset.format("%Y-%m-%dT%H:%M:%SZ")
    );
    let bytes = second.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        bytes.as_ref(),
        serde_json::to_vec(&json!({"detail":detail}))
            .unwrap()
            .as_slice()
    );
    let traces = fixture
        .store
        .search_traces(
            WORKSPACE_DEFAULT_NAME,
            std::slice::from_ref(&fixture.trace_experiment_id),
            None,
            10,
            &[],
            None,
        )
        .await
        .unwrap();
    let rejected = traces
        .trace_infos
        .iter()
        .find(|trace| trace.state == "ERROR")
        .expect("Python-compatible budget rejection trace");
    let rejected = fixture
        .store
        .batch_get_traces(
            WORKSPACE_DEFAULT_NAME,
            std::slice::from_ref(&rejected.trace_id),
        )
        .await
        .unwrap()
        .remove(0);
    assert_eq!(rejected.spans.len(), 1);
    assert_eq!(rejected.spans[0].span_type.as_deref(), Some("LLM"));
    assert_eq!(rejected.spans[0].status, "ERROR");
}

#[tokio::test]
async fn alert_policy_dispatches_python_shaped_budget_webhook_once() {
    let fixture = Fixture::new().await;
    fixture
        .webhook_store
        .create_webhook(
            WORKSPACE_DEFAULT_NAME,
            "budget-alert-recorder",
            &format!("{}/budget", fixture.webhook_base),
            &[WebhookEvent::new(
                WebhookEntity::BudgetPolicy,
                WebhookAction::Exceeded,
            )],
            None,
            None,
            Some(WebhookStatus::Active),
        )
        .await
        .unwrap();
    let invocation_cost = token_cost("openai", "gpt-4", 2, 3).unwrap();
    let policy = fixture
        .store
        .create_budget_policy(
            WORKSPACE_DEFAULT_NAME,
            "USD",
            invocation_cost,
            "DAYS",
            1,
            "GLOBAL",
            "ALERT",
            Some("budget-test"),
        )
        .await
        .unwrap();

    let response = fixture
        .post("traced-openai-endpoint", chat("alert", false))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    for _ in 0..100 {
        if !fixture.webhook_deliveries.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let deliveries = fixture.webhook_deliveries.lock().unwrap();
    assert_eq!(deliveries.len(), 1);
    let envelope = &deliveries[0];
    assert_eq!(envelope["entity"], "budget_policy");
    assert_eq!(envelope["action"], "exceeded");
    assert_eq!(
        envelope["data"]["budget_policy_id"],
        policy.budget_policy_id
    );
    assert_eq!(envelope["data"]["budget_unit"], "USD");
    assert_eq!(envelope["data"]["budget_amount"], invocation_cost);
    assert_eq!(envelope["data"]["current_spend"], invocation_cost);
    assert_eq!(envelope["data"]["duration_unit"], "DAYS");
    assert_eq!(envelope["data"]["duration_value"], 1);
    assert_eq!(envelope["data"]["target_scope"], "GLOBAL");
    assert_eq!(envelope["data"]["workspace"], WORKSPACE_DEFAULT_NAME);
    assert!(envelope["data"]["window_start"].is_i64());
}

#[tokio::test]
async fn all_native_providers_match_unified_non_stream_and_error_contracts() {
    let fixture = Fixture::new().await;
    assert!(fixture.mock_base.starts_with("http://127.0.0.1:"));
    for provider in ["openai", "azure", "anthropic", "gemini"] {
        let response = fixture
            .post(&format!("{provider}-endpoint"), chat("hello", false))
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{provider}");
        assert!(response
            .headers()
            .contains_key("x-mlflow-gateway-duration-ms"));
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "fixture answer");
        assert_eq!(body["provider"], provider_name(provider));

        let response = fixture
            .post(&format!("{provider}-endpoint"), chat("error-429", false))
            .await;
        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "{provider}"
        );
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body, json!({"detail":"fixture provider limit"}));

        let response = fixture
            .post(&format!("{provider}-endpoint"), chat("error-500", false))
            .await;
        assert_eq!(
            response.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "{provider}"
        );
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body, json!({"detail":"fixture provider failure"}));
    }
}

#[tokio::test]
async fn all_native_provider_streams_have_exact_frames_and_midstream_errors() {
    let fixture = Fixture::new().await;
    for provider in ["openai", "azure", "anthropic", "gemini"] {
        let response = fixture
            .post(&format!("{provider}-endpoint"), chat("hello", true))
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{provider}");
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream; charset=utf-8"
        );
        assert!(response
            .headers()
            .contains_key("x-mlflow-gateway-duration-ms"));
        let frames = response
            .into_body()
            .into_data_stream()
            .map(|chunk| String::from_utf8(chunk.unwrap().to_vec()).unwrap())
            .collect::<Vec<_>>()
            .await;
        assert!(!frames.is_empty(), "{provider}");
        assert!(frames
            .iter()
            .all(|frame| frame.starts_with("data: {") && frame.ends_with("\n\n")));
        assert!(frames
            .iter()
            .all(|frame| !frame.contains("[DONE]") && !frame.contains("keep-alive")));
        let combined = frames.concat();
        assert!(combined.contains("fixture "), "{provider}: {combined}");
        assert!(
            combined.contains(provider_name(provider)),
            "{provider}: {combined}"
        );

        let response = fixture
            .post(
                &format!("{provider}-endpoint"),
                chat("mid-stream-error", true),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let stream = String::from_utf8(bytes.to_vec()).unwrap();
        if matches!(provider, "openai" | "azure") {
            assert!(!stream.contains("\"error\""), "{provider}: {stream}");
        } else {
            assert!(
                stream.contains("\"type\": \"JSONDecodeError\""),
                "{provider}: {stream}"
            );
        }
        assert!(!stream.contains("[DONE]"));
    }
}

#[tokio::test]
async fn model_selected_chat_route_removes_endpoint_model_before_provider_transform() {
    let fixture = Fixture::new().await;
    let response = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/gateway/mlflow/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model":"openai-endpoint","messages":[{"role":"user","content":"hello"}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn native_embedding_adapters_use_the_unified_input_branch() {
    let fixture = Fixture::new().await;
    for provider in ["openai", "azure", "gemini"] {
        let response = fixture
            .post(
                &format!("{provider}-endpoint"),
                json!({"input":"fixture embedding input"}),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{provider}");
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["embedding"], json!([0.25, 0.5]));
    }
}

#[tokio::test]
async fn passthrough_and_raw_proxy_routes_share_the_hermetic_transport() {
    let fixture = Fixture::new().await;
    let unary_cases = [
        (
            "/gateway/openai/v1/chat/completions",
            json!({"model":"openai-endpoint","messages":[{"role":"user","content":"hello"}]}),
        ),
        (
            "/gateway/openai/v1/embeddings",
            json!({"model":"openai-endpoint","input":"hello"}),
        ),
        (
            "/gateway/openai/v1/responses",
            json!({"model":"openai-endpoint","input":"hello"}),
        ),
        (
            "/gateway/openai/v1/responses/compact",
            json!({"model":"openai-endpoint","previous_response_id":"obvious-fake-response"}),
        ),
        (
            "/gateway/anthropic/v1/messages",
            json!({"model":"anthropic-endpoint","messages":[{"role":"user","content":"hello"}],"max_tokens":8}),
        ),
        (
            "/gateway/gemini/v1beta/models/gemini-endpoint:generateContent",
            json!({"contents":[{"role":"user","parts":[{"text":"hello"}]}]}),
        ),
    ];
    for (path, body) in unary_cases {
        let response = fixture
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("x-mlflow-authorization", "obvious-fake-rbac-token")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        assert!(response
            .headers()
            .contains_key("x-mlflow-gateway-duration-ms"));
        let value: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert!(value.is_object(), "{path}: {value}");
    }

    let response = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/gateway/gemini/v1beta/models/gemini-endpoint:streamGenerateContent")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"contents":[{"role":"user","parts":[{"text":"hello"}]}]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let stream = response.into_body().collect().await.unwrap().to_bytes();
    assert!(stream.windows(6).any(|window| window == b"[DONE]"));
    assert!(stream.windows(10).any(|window| window == b"keep-alive"));

    let response = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/gateway/proxy/openai-endpoint/v1/chat/completions?fixture=1")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model":"caller-selected-model","messages":[{"role":"user","content":"hello"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(value["model"], "caller-selected-model");

    let response = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/gateway/openai/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model":"openai-endpoint","messages":[{"role":"user","content":"error-429"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    let response = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/gateway/openai/v1/responses/compact")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model":"openai-endpoint","stream":true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

fn provider_name(provider: &str) -> &str {
    if provider == "azure" {
        "openai"
    } else {
        provider
    }
}

fn script_attempts(attempts: Vec<String>) -> Vec<String> {
    attempts
}

#[tokio::test]
async fn forced_traffic_weights_and_single_target_are_deterministic() {
    assert_eq!(
        python_routing_oracle()["weights"]["integer"],
        json!([0.0, 69.0, 30.0])
    );
    let fixture = Fixture::new().await;
    let response = fixture
        .post("traffic-zero-hundred", chat("hello", false))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["model"], "traffic-hundred-model");
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        ["traffic-hundred"]
    );

    let response = fixture.post("traffic-single", chat("hello", false)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["model"], "traffic-single-model");
    assert_eq!(script_attempts(fixture.take_attempts()), ["traffic-single"]);
}

#[tokio::test]
async fn fallback_success_order_limits_and_final_status_match_python() {
    let oracle = python_routing_oracle();
    let fixture = Fixture::new().await;
    let response = fixture
        .post("fallback-first-500", chat("hello", false))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["model"], "fallback-success-model");
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        serde_json::from_value::<Vec<String>>(oracle["first_500_then_success"]["attempts"].clone())
            .unwrap()
    );

    let response = fixture.post("fallback-order", chat("hello", false)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["model"], "early-order-model");
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        ["fail-500", "early-order"]
    );

    let response = fixture
        .post("fallback-all-fail", chat("hello", false))
        .await;
    assert_eq!(
        response.status().as_u16(),
        oracle["all_fail"]["status"].as_u64().unwrap() as u16
    );
    assert!(response
        .headers()
        .get("x-mlflow-gateway-attempts")
        .is_none());
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        bytes.as_ref(),
        serde_json::to_vec(&json!({"detail":oracle["all_fail"]["detail"]}))
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        serde_json::from_value::<Vec<String>>(oracle["all_fail"]["attempts"].clone()).unwrap()
    );

    let response = fixture
        .post("fallback-max-attempts", chat("hello", false))
        .await;
    assert_eq!(
        response.status().as_u16(),
        oracle["max_attempts"]["status"].as_u64().unwrap() as u16
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        bytes.as_ref(),
        serde_json::to_vec(&json!({"detail":oracle["max_attempts"]["detail"]}))
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        serde_json::from_value::<Vec<String>>(oracle["max_attempts"]["attempts"].clone()).unwrap()
    );
}

#[tokio::test]
async fn request_validation_propagates_before_any_fallback_attempt() {
    let fixture = Fixture::new().await;
    let response = fixture
        .post(
            "fallback-first-500",
            json!({"model":"client-model","messages":[{"role":"user","content":"hello"}]}),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(fixture.take_attempts().is_empty());
}

#[tokio::test]
async fn streaming_falls_back_before_and_after_the_first_emitted_chunk() {
    let oracle = python_routing_oracle();
    let fixture = Fixture::new().await;
    let response = fixture
        .post("fallback-first-500", chat("hello", true))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(stream.contains("fixture "));
    assert!(!stream.contains("scripted primary failure"));
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        ["fail-500", "fallback-success"]
    );

    let response = fixture
        .post("fallback-partial-stream", chat("hello", true))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(stream.contains("anthropic"), "{stream}");
    assert!(stream.contains("openai"), "{stream}");
    assert!(!stream.contains("JSONDecodeError"), "{stream}");
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        serde_json::from_value::<Vec<String>>(oracle["partial_stream"]["attempts"].clone())
            .unwrap()
    );

    let response = fixture.post("fallback-all-fail", chat("hello", true)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert_eq!(
        stream,
        "data: {\"error\": {\"message\": \"All 2 fallback attempts failed. Last error: 429: scripted final limit\", \"type\": \"AIGatewayException\"}}\n\n"
    );
    assert_eq!(
        script_attempts(fixture.take_attempts()),
        ["fail-500", "fail-429"]
    );
}
