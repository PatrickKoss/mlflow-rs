//! Hermetic T18.3 gateway runtime coverage. Every provider call targets the
//! local mock below; no test or CI path can reach a live provider.

use std::collections::HashMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};

use axum::body::{Body, Bytes};
use axum::extract::Request;
use axum::http::{header, HeaderValue, Method, Response, StatusCode};
use axum::routing::post;
use axum::Router;
use futures::{stream, StreamExt};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, EndpointModelConfig, PoolConfig, TrackingStore, WORKSPACE_DEFAULT_NAME};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tower::ServiceExt;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct Fixture {
    _directory: tempfile::TempDir,
    app: Router,
    mock_base: String,
}

impl Fixture {
    async fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mock_base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, Router::new().fallback(post(mock_provider)))
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
        let store =
            TrackingStore::new(db, directory.path().join("artifacts").display().to_string());
        for provider in ["openai", "azure", "anthropic", "gemini"] {
            seed_endpoint(&store, provider, &mock_base).await;
        }
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(
            &ServerConfig {
                disable_security_middleware: true,
                ..Default::default()
            },
            recorder,
            Some(AppState::new(store)),
        );
        Self {
            _directory: directory,
            app,
            mock_base,
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
}

async fn seed_endpoint(store: &TrackingStore, provider: &str, base: &str) {
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

async fn mock_provider(request: Request) -> Response<Body> {
    let path = request.uri().path().to_string();
    let headers = request.headers().clone();
    let body: Value =
        serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes()).unwrap();
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
        return stream_provider_response(&path, content == "mid-stream-error");
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

fn stream_provider_response(path: &str, fail_midstream: bool) -> Response<Body> {
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
        chunks.truncate(1);
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
