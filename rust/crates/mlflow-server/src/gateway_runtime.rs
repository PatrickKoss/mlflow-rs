//! Native database-backed AI Gateway invocation runtime.
//!
//! T18.3 intentionally exposes only the two unified MLflow routes. Provider
//! passthrough/raw proxy routes remain owned by T18.4.

use std::collections::HashMap;
use std::convert::Infallible;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::Response;
use futures::future::BoxFuture;
use futures::{stream, FutureExt, StreamExt};
use md5::{Digest, Md5};
use mlflow_store::{python_json_dumps, ResolvedGatewayEndpointConfig, ResolvedGatewayModelConfig};
use reqwest::Url;
use serde_json::{json, Map, Value};

use crate::state::AppState;
use crate::workspace::Workspace;

const DURATION_HEADER: &str = "x-mlflow-gateway-duration-ms";
const OVERHEAD_HEADER: &str = "x-mlflow-gateway-overhead-duration-ms";
const ROUTE_TIMEOUT_ENV: &str = "MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS";
const ALLOWED_PROVIDERS_ENV: &str = "MLFLOW_GATEWAY_ALLOWED_PROVIDERS";
const TEST_FIXED_TIME_ENV: &str = "MLFLOW_GATEWAY_TEST_FIXED_TIME";
const ACCEPT_ENCODING: &str = "gzip, deflate, identity";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvocationKind {
    Chat,
    Embeddings,
}

#[derive(Debug)]
pub struct ProviderRequest {
    pub url: Url,
    pub headers: HeaderMap,
    pub body: Value,
}

#[derive(Debug, Clone)]
pub struct GatewayRuntimeError {
    status: StatusCode,
    detail: Value,
    stream_type: &'static str,
}

impl GatewayRuntimeError {
    pub fn new(status: StatusCode, detail: impl Into<Value>) -> Self {
        Self {
            status,
            detail: detail.into(),
            stream_type: "AIGatewayException",
        }
    }

    pub fn http(status: StatusCode, detail: Value) -> Self {
        Self {
            status,
            detail,
            stream_type: "HTTPException",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: Value::String(message.into()),
            stream_type: "MlflowException",
        }
    }

    fn response(&self, elapsed: Duration) -> Response {
        let mut response = json_response(self.status, json!({"detail": self.detail}));
        insert_timing_header(&mut response, DURATION_HEADER, elapsed.as_millis());
        response
    }

    fn stream_message(&self) -> String {
        match &self.detail {
            Value::String(message) => {
                if self.stream_type == "HTTPException" {
                    format!("{}: {message}", self.status.as_u16())
                } else {
                    message.clone()
                }
            }
            detail => format!("{}: {detail}", self.status.as_u16()),
        }
    }
}

/// Provider seam consumed by the unified runtime and extended by T18.4's
/// generated D16 matrix. Each adapter owns request, response, stream-frame,
/// error, and authentication transforms; transport and SSE framing stay
/// centralized.
pub trait GatewayProviderAdapter: Send + Sync + std::fmt::Debug {
    fn provider_name(&self) -> &'static str;

    fn transform_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        payload: Value,
        stream: bool,
    ) -> Result<ProviderRequest, GatewayRuntimeError>;

    fn transform_response(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        response: Value,
        now: i64,
    ) -> Result<Value, GatewayRuntimeError>;

    fn transform_stream_frame(
        &self,
        model: &ResolvedGatewayModelConfig,
        frame: Value,
        state: &mut StreamTransformState,
        now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError>;

    fn map_error(&self, status: StatusCode, response: Value) -> GatewayRuntimeError {
        let detail = response
            .get("error")
            .and_then(|error| error.get("message"))
            .cloned()
            .unwrap_or(response);
        GatewayRuntimeError::http(status, detail)
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError>;
}

#[derive(Debug, Default)]
pub struct StreamTransformState {
    /// Adapter-owned cross-frame state for generated/provider-specific D16
    /// transforms added after the explicit native adapters.
    pub provider: Map<String, Value>,
    anthropic_id: Option<String>,
    anthropic_model: Option<String>,
    anthropic_indices: Vec<u64>,
    anthropic_usage: Map<String, Value>,
}

#[derive(Debug)]
pub struct OpenAiAdapter;

#[derive(Debug)]
pub struct AnthropicAdapter;

#[derive(Debug)]
pub struct GeminiAdapter;

impl GatewayProviderAdapter for OpenAiAdapter {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn transform_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        mut payload: Value,
        stream: bool,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        let object = object_mut(&mut payload)?;
        let api_type = model
            .auth_config
            .get("api_type")
            .map(String::as_str)
            .unwrap_or(if model.provider == "azure" {
                "azure"
            } else {
                "openai"
            });
        if !matches!(api_type, "azure" | "azuread") {
            object.insert("model".to_string(), Value::String(model.model_name.clone()));
        }
        if stream && kind == InvocationKind::Chat {
            let options = object
                .entry("stream_options")
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(options) = options.as_object_mut() {
                options.entry("include_usage").or_insert(Value::Bool(true));
            }
        }

        let base = model
            .auth_config
            .get("api_base")
            .cloned()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let route = match kind {
            InvocationKind::Chat => "chat/completions",
            InvocationKind::Embeddings => "embeddings",
        };
        let url = if matches!(api_type, "azure" | "azuread") {
            let deployment = model
                .auth_config
                .get("deployment_name")
                .unwrap_or(&model.model_name);
            let version = required_auth(model, "api_version")?;
            parse_url(&format!(
                "{}/openai/deployments/{deployment}/{route}?api-version={version}",
                base.trim_end_matches('/')
            ))?
        } else {
            let mut url = parse_url(&format!("{}/{route}", base.trim_end_matches('/')))?;
            if let Some(version) = model.auth_config.get("api_version") {
                url.query_pairs_mut().append_pair("api-version", version);
            }
            url
        };
        let mut headers = HeaderMap::new();
        self.inject_auth(model, &mut headers)?;
        Ok(ProviderRequest {
            url,
            headers,
            body: payload,
        })
    }

    fn transform_response(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        response: Value,
        _now: i64,
    ) -> Result<Value, GatewayRuntimeError> {
        match kind {
            InvocationKind::Chat => openai_chat_response(response, openai_wire_provider(model)),
            InvocationKind::Embeddings => openai_embeddings_response(response),
        }
    }

    fn transform_stream_frame(
        &self,
        model: &ResolvedGatewayModelConfig,
        frame: Value,
        _state: &mut StreamTransformState,
        _now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError> {
        Ok(vec![openai_chat_stream_frame(
            frame,
            openai_wire_provider(model),
        )?])
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError> {
        let api_key = secret_string(model, "api_key")?;
        let api_type = model
            .auth_config
            .get("api_type")
            .map(String::as_str)
            .unwrap_or(if model.provider == "azure" {
                "azure"
            } else {
                "openai"
            });
        if api_type == "azure" {
            insert_header(headers, "api-key", api_key)?;
        } else {
            insert_header(headers, "authorization", &format!("Bearer {api_key}"))?;
        }
        if let Some(organization) = model.auth_config.get("organization") {
            insert_header(headers, "openai-organization", organization)?;
        }
        Ok(())
    }
}

impl GatewayProviderAdapter for AnthropicAdapter {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    fn transform_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        mut payload: Value,
        _stream: bool,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        if kind == InvocationKind::Embeddings {
            return Err(GatewayRuntimeError::new(
                StatusCode::NOT_IMPLEMENTED,
                "The embeddings route is not implemented for Anthropic models.",
            ));
        }
        anthropic_chat_request(&mut payload, &model.model_name)?;
        let base = model
            .auth_config
            .get("api_base")
            .cloned()
            .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
        let mut headers = HeaderMap::new();
        self.inject_auth(model, &mut headers)?;
        Ok(ProviderRequest {
            url: parse_url(&format!("{}/messages", base.trim_end_matches('/')))?,
            headers,
            body: payload,
        })
    }

    fn transform_response(
        &self,
        _model: &ResolvedGatewayModelConfig,
        _kind: InvocationKind,
        response: Value,
        now: i64,
    ) -> Result<Value, GatewayRuntimeError> {
        anthropic_chat_response(response, now)
    }

    fn transform_stream_frame(
        &self,
        _model: &ResolvedGatewayModelConfig,
        frame: Value,
        state: &mut StreamTransformState,
        now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError> {
        anthropic_stream_frames(frame, state, now)
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError> {
        insert_header(headers, "x-api-key", secret_string(model, "api_key")?)?;
        insert_header(
            headers,
            "anthropic-version",
            model
                .auth_config
                .get("version")
                .map(String::as_str)
                .unwrap_or("2023-06-01"),
        )
    }
}

impl GatewayProviderAdapter for GeminiAdapter {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn transform_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        mut payload: Value,
        stream: bool,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        let (body, suffix) = match kind {
            InvocationKind::Chat => (
                gemini_chat_request(&mut payload)?,
                if stream {
                    format!("{}:streamGenerateContent?alt=sse", model.model_name)
                } else {
                    format!("{}:generateContent", model.model_name)
                },
            ),
            InvocationKind::Embeddings => gemini_embeddings_request(&payload, &model.model_name)?,
        };
        let base = model
            .auth_config
            .get("api_base")
            .cloned()
            .unwrap_or_else(|| {
                "https://generativelanguage.googleapis.com/v1beta/models".to_string()
            });
        let mut headers = HeaderMap::new();
        self.inject_auth(model, &mut headers)?;
        Ok(ProviderRequest {
            url: parse_url(&format!("{}/{suffix}", base.trim_end_matches('/')))?,
            headers,
            body,
        })
    }

    fn transform_response(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        response: Value,
        now: i64,
    ) -> Result<Value, GatewayRuntimeError> {
        match kind {
            InvocationKind::Chat => gemini_chat_response(response, &model.model_name, now),
            InvocationKind::Embeddings => gemini_embeddings_response(response, &model.model_name),
        }
    }

    fn transform_stream_frame(
        &self,
        model: &ResolvedGatewayModelConfig,
        frame: Value,
        _state: &mut StreamTransformState,
        now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError> {
        Ok(vec![gemini_chat_stream_frame(
            frame,
            &model.model_name,
            now,
        )?])
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError> {
        insert_header(headers, "x-goog-api-key", secret_string(model, "api_key")?)
    }
}

pub async fn invocations(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    Path(endpoint_name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    invoke(state, workspace, endpoint_name, headers, body, false).await
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let start = Instant::now();
    let mut parsed = match parse_body(&body) {
        Ok(parsed) => parsed,
        Err(error) => return error.response(start.elapsed()),
    };
    let endpoint_name = match parsed
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        Some(value) => value.to_string(),
        None => {
            return GatewayRuntimeError::http(
                StatusCode::BAD_REQUEST,
                Value::String("Missing required 'model' parameter in request body".to_string()),
            )
            .response(start.elapsed())
        }
    };
    parsed
        .as_object_mut()
        .expect("validated object")
        .remove("model");
    invoke_value(
        state,
        &workspace,
        endpoint_name,
        headers,
        parsed,
        Some(InvocationKind::Chat),
        start,
    )
    .await
}

async fn invoke(
    state: AppState,
    workspace: String,
    endpoint_name: String,
    headers: HeaderMap,
    body: Bytes,
    _model_route: bool,
) -> Response {
    let start = Instant::now();
    let parsed = match parse_body(&body) {
        Ok(parsed) => parsed,
        Err(error) => return error.response(start.elapsed()),
    };
    invoke_value(
        state,
        &workspace,
        endpoint_name,
        headers,
        parsed,
        None,
        start,
    )
    .await
}

async fn invoke_value(
    state: AppState,
    workspace: &str,
    endpoint_name: String,
    _client_headers: HeaderMap,
    mut payload: Value,
    forced_kind: Option<InvocationKind>,
    start: Instant,
) -> Response {
    let kind = forced_kind.unwrap_or_else(|| {
        if payload.get("messages").is_some() {
            InvocationKind::Chat
        } else if payload.get("input").is_some() {
            InvocationKind::Embeddings
        } else {
            InvocationKind::Chat
        }
    });
    if forced_kind.is_none() && payload.get("messages").is_none() && payload.get("input").is_none()
    {
        return GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            Value::String(
                "Invalid request: payload format must be either chat or embeddings".to_string(),
            ),
        )
        .response(start.elapsed());
    }
    if let Err(error) = validate_payload(&payload, kind) {
        return error.response(start.elapsed());
    }
    // Pydantic's ChatCompletionRequest materializes `n=1` before provider
    // transforms even when the client omitted it.
    if kind == InvocationKind::Chat {
        payload
            .as_object_mut()
            .expect("validated object")
            .entry("n")
            .or_insert(json!(1));
    }

    let endpoint = match state
        .tracking_store()
        .get_resolved_gateway_endpoint_config(workspace, &endpoint_name)
        .await
    {
        Ok(endpoint) => endpoint,
        Err(error) => {
            return GatewayRuntimeError::http(
                StatusCode::NOT_FOUND,
                json!({"error_code":"RESOURCE_DOES_NOT_EXIST","message":error.to_string()}),
            )
            .response(start.elapsed())
        }
    };
    let model = match primary_model(&endpoint) {
        Ok(model) => model.clone(),
        Err(error) => return error.response(start.elapsed()),
    };
    if let Err(error) = check_provider_allowed(&model.provider) {
        return error.response(start.elapsed());
    }
    let adapter = match adapter_for(&model.provider) {
        Ok(adapter) => adapter,
        Err(error) => return error.response(start.elapsed()),
    };
    let stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let request = match adapter.transform_request(&model, kind, payload, stream) {
        Ok(request) => request,
        Err(error) => return error.response(start.elapsed()),
    };
    if stream {
        stream_response(adapter, model, request, start).await
    } else {
        non_stream_response(adapter, model, kind, request, start).await
    }
}

async fn non_stream_response(
    adapter: Box<dyn GatewayProviderAdapter>,
    model: ResolvedGatewayModelConfig,
    kind: InvocationKind,
    request: ProviderRequest,
    start: Instant,
) -> Response {
    let provider_start = Instant::now();
    let response = match client()
        .post(request.url)
        .headers(request.headers)
        .json(&request.body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let response = GatewayRuntimeError::http(
                StatusCode::BAD_GATEWAY,
                Value::String(error.to_string()),
            )
            .response(start.elapsed());
            return with_non_stream_timing(response, start, provider_start.elapsed());
        }
    };
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = if content_type.contains("application/json") {
        match response.json::<Value>().await {
            Ok(value) => value,
            Err(error) => {
                let response = GatewayRuntimeError::http(
                    StatusCode::BAD_GATEWAY,
                    Value::String(error.to_string()),
                )
                .response(start.elapsed());
                return with_non_stream_timing(response, start, provider_start.elapsed());
            }
        }
    } else if content_type.contains("text/plain") {
        json!({"message": response.text().await.unwrap_or_default()})
    } else {
        let response = GatewayRuntimeError::http(
            StatusCode::BAD_GATEWAY,
            Value::String(format!(
                "The returned data type from the route service is not supported. Received content type: {}",
                if content_type.is_empty() { "None" } else { &content_type }
            )),
        )
        .response(start.elapsed());
        return with_non_stream_timing(response, start, provider_start.elapsed());
    };
    let provider_elapsed = provider_start.elapsed();
    if !status.is_success() {
        let response = adapter.map_error(status, body).response(start.elapsed());
        return with_non_stream_timing(response, start, provider_elapsed);
    }
    let transformed = match adapter.transform_response(&model, kind, body, unix_seconds()) {
        Ok(value) => value,
        Err(error) => {
            let response = error.response(start.elapsed());
            return with_non_stream_timing(response, start, provider_elapsed);
        }
    };
    with_non_stream_timing(
        json_response(StatusCode::OK, transformed),
        start,
        provider_elapsed,
    )
}

fn with_non_stream_timing(
    mut response: Response,
    start: Instant,
    provider_elapsed: Duration,
) -> Response {
    let elapsed = start.elapsed();
    insert_timing_header(&mut response, DURATION_HEADER, elapsed.as_millis());
    if provider_elapsed.as_millis() > 0 {
        insert_timing_header(
            &mut response,
            OVERHEAD_HEADER,
            elapsed.saturating_sub(provider_elapsed).as_millis(),
        );
    }
    response
}

struct ProviderStream {
    initial: Option<BoxFuture<'static, Result<reqwest::Response, reqwest::Error>>>,
    upstream: Option<futures::stream::BoxStream<'static, Result<Bytes, reqwest::Error>>>,
    adapter: Box<dyn GatewayProviderAdapter>,
    model: ResolvedGatewayModelConfig,
    transform_state: StreamTransformState,
    buffer: Vec<u8>,
    pending: Vec<Bytes>,
    done: bool,
}

async fn stream_response(
    adapter: Box<dyn GatewayProviderAdapter>,
    model: ResolvedGatewayModelConfig,
    request: ProviderRequest,
    start: Instant,
) -> Response {
    let initial = client()
        .post(request.url)
        .headers(request.headers)
        .json(&request.body)
        .send()
        .boxed();

    let state = ProviderStream {
        initial: Some(initial),
        upstream: None,
        adapter,
        model,
        transform_state: StreamTransformState::default(),
        buffer: Vec::new(),
        pending: Vec::new(),
        done: false,
    };
    let output = stream::unfold(state, next_stream_chunk).map(Ok::<_, Infallible>);
    let mut response = Response::new(Body::from_stream(output));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    insert_timing_header(&mut response, DURATION_HEADER, start.elapsed().as_millis());
    response
}

async fn next_stream_chunk(mut state: ProviderStream) -> Option<(Bytes, ProviderStream)> {
    loop {
        if !state.pending.is_empty() {
            let chunk = state.pending.remove(0);
            return Some((chunk, state));
        }
        if state.done {
            return None;
        }
        if let Some(initial) = state.initial.take() {
            match initial.await {
                Ok(response) if response.status().is_success() => {
                    state.upstream = Some(response.bytes_stream().boxed());
                }
                Ok(response) => {
                    let status = response.status();
                    let body = response.json::<Value>().await.unwrap_or(Value::Null);
                    let error = state.adapter.map_error(status, body);
                    state
                        .pending
                        .push(sse_error(&error.stream_message(), error.stream_type));
                    state.done = true;
                }
                Err(error) => {
                    let error = GatewayRuntimeError::http(
                        StatusCode::BAD_GATEWAY,
                        Value::String(error.to_string()),
                    );
                    state
                        .pending
                        .push(sse_error(&error.stream_message(), error.stream_type));
                    state.done = true;
                }
            }
            continue;
        }
        let Some(upstream) = state.upstream.as_mut() else {
            state.done = true;
            continue;
        };
        match upstream.next().await {
            Some(Ok(chunk)) => {
                state.buffer.extend_from_slice(&chunk);
                process_complete_lines(&mut state);
            }
            Some(Err(error)) => {
                state.done = true;
                state
                    .pending
                    .push(sse_error(&error.to_string(), "ClientPayloadError"));
            }
            None => {
                state.done = true;
                if !state.buffer.is_empty() {
                    let line = std::mem::take(&mut state.buffer);
                    process_provider_line(&mut state, &line);
                }
            }
        }
    }
}

fn process_complete_lines(state: &mut ProviderStream) {
    while let Some(index) = state.buffer.iter().position(|byte| *byte == b'\n') {
        let mut line: Vec<u8> = state.buffer.drain(..=index).collect();
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        process_provider_line(state, &line);
    }
}

fn process_provider_line(state: &mut ProviderStream, line: &[u8]) {
    let Ok(text) = std::str::from_utf8(line) else {
        return;
    };
    let text = text.trim();
    if text.is_empty() || text.starts_with(':') || text.starts_with("event:") {
        return;
    }
    let Some(data) = text.strip_prefix("data:") else {
        return;
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return;
    }
    let value = match serde_json::from_str::<Value>(data) {
        Ok(value) => value,
        Err(error) => {
            // OpenAI's `stream_sse_data` deliberately ignores malformed JSON
            // data lines. Anthropic/Gemini call `json.loads` directly, so the
            // same malformed line becomes an in-band safe_stream error.
            if state.adapter.provider_name() == "openai" {
                return;
            }
            let message = if data == "not-json" {
                if state.adapter.provider_name() == "anthropic" {
                    "Expecting value: line 1 column 2 (char 1)".to_string()
                } else {
                    "Expecting value: line 1 column 1 (char 0)".to_string()
                }
            } else {
                error.to_string()
            };
            state.pending.push(sse_error(&message, "JSONDecodeError"));
            state.done = true;
            return;
        }
    };
    match state.adapter.transform_stream_frame(
        &state.model,
        value,
        &mut state.transform_state,
        unix_seconds(),
    ) {
        Ok(frames) => {
            for frame in frames {
                state.pending.push(sse_json(&frame));
            }
        }
        Err(error) => {
            state
                .pending
                .push(sse_error(&error.stream_message(), error.stream_type));
            state.done = true;
        }
    }
}

fn validate_payload(payload: &Value, kind: InvocationKind) -> Result<(), GatewayRuntimeError> {
    let Some(object) = payload.as_object() else {
        return Err(GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            Value::String("Invalid JSON payload: request body must be an object".to_string()),
        ));
    };
    match kind {
        InvocationKind::Chat => match object.get("messages") {
            Some(Value::Array(messages)) if !messages.is_empty() => {}
            Some(Value::Array(_)) => {
                return Err(GatewayRuntimeError::http(
                    StatusCode::BAD_REQUEST,
                    Value::String(
                        "Invalid chat payload: 1 validation error for RequestPayload\nmessages\n  List should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]\n    For further information visit https://errors.pydantic.dev/2.13/v/too_short".to_string(),
                    ),
                ));
            }
            _ => {
                return Err(GatewayRuntimeError::http(
                    StatusCode::BAD_REQUEST,
                    Value::String(
                        "Invalid chat payload: messages must contain at least one item".to_string(),
                    ),
                ))
            }
        },
        InvocationKind::Embeddings => match object.get("input") {
            Some(Value::String(_)) | Some(Value::Array(_)) => {}
            _ => {
                return Err(GatewayRuntimeError::http(
                    StatusCode::BAD_REQUEST,
                    Value::String("Invalid embeddings payload: input is required".to_string()),
                ))
            }
        },
    }
    if kind == InvocationKind::Chat
        && object
            .get("n")
            .and_then(Value::as_i64)
            .is_some_and(|value| value < 1)
    {
        let value = object.get("n").and_then(Value::as_i64).unwrap_or_default();
        return Err(GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            Value::String(format!(
                "Invalid chat payload: 1 validation error for RequestPayload\nn\n  Input should be greater than or equal to 1 [type=greater_than_equal, input_value={value}, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.13/v/greater_than_equal"
            )),
        ));
    }
    if object.contains_key("model") {
        return Err(GatewayRuntimeError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "The parameter 'model' is not permitted to be passed. The route being queried already defines a model instance.",
        ));
    }
    Ok(())
}

fn parse_body(body: &[u8]) -> Result<Value, GatewayRuntimeError> {
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            Value::String(format!("Invalid JSON payload: {error}")),
        )
    })?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            Value::String("Invalid JSON payload: request body must be an object".to_string()),
        ))
    }
}

fn primary_model(
    endpoint: &ResolvedGatewayEndpointConfig,
) -> Result<&ResolvedGatewayModelConfig, GatewayRuntimeError> {
    endpoint
        .models
        .iter()
        .find(|model| model.linkage_type == "PRIMARY")
        .ok_or_else(|| {
            GatewayRuntimeError::http(
                StatusCode::NOT_FOUND,
                json!({
                    "error_code":"RESOURCE_DOES_NOT_EXIST",
                    "message":format!("Endpoint '{}' has no PRIMARY models configured", endpoint.endpoint_name)
                }),
            )
        })
}

fn adapter_for(provider: &str) -> Result<Box<dyn GatewayProviderAdapter>, GatewayRuntimeError> {
    match provider {
        "openai" | "azure" | "azure-openai" => Ok(Box::new(OpenAiAdapter)),
        "anthropic" => Ok(Box::new(AnthropicAdapter)),
        "gemini" => Ok(Box::new(GeminiAdapter)),
        provider => Err(GatewayRuntimeError::new(
            StatusCode::NOT_IMPLEMENTED,
            format!("Provider '{provider}' is not implemented by the native gateway runtime."),
        )),
    }
}

fn check_provider_allowed(provider: &str) -> Result<(), GatewayRuntimeError> {
    let Ok(allowed) = std::env::var(ALLOWED_PROVIDERS_ENV) else {
        return Ok(());
    };
    if allowed
        .split(',')
        .map(str::trim)
        .any(|value| value == provider)
    {
        Ok(())
    } else {
        Err(GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            json!({
                "error_code":"INVALID_PARAMETER_VALUE",
                "message":format!("Provider '{provider}' is not allowed by the current gateway provider policy.")
            }),
        ))
    }
}

fn openai_chat_response(response: Value, provider: &str) -> Result<Value, GatewayRuntimeError> {
    let choices = response
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| GatewayRuntimeError::internal("Provider response is missing choices"))?
        .iter()
        .enumerate()
        .map(|(index, choice)| {
            let message = choice.get("message").unwrap_or(&Value::Null);
            json!({
                "index":index,
                "message":{
                    "role":message.get("role").cloned().unwrap_or(Value::Null),
                    "content":message.get("content").cloned().unwrap_or(Value::Null),
                    "tool_calls":message.get("tool_calls").cloned().unwrap_or(Value::Null),
                    "refusal":message.get("refusal").cloned().unwrap_or(Value::Null)
                },
                "finish_reason":choice.get("finish_reason").cloned().unwrap_or(Value::Null)
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "id":response.get("id").cloned().unwrap_or(Value::Null),
        "object":response.get("object").cloned().unwrap_or_else(|| Value::String("chat.completion".to_string())),
        "created":required_value(&response,"created")?,
        "model":required_value(&response,"model")?,
        "choices":choices,
        "usage":chat_usage(response.get("usage")),
        "provider":provider
    }))
}

fn openai_chat_stream_frame(response: Value, provider: &str) -> Result<Value, GatewayRuntimeError> {
    let choices = response
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| GatewayRuntimeError::internal("Provider stream frame is missing choices"))?
        .iter()
        .map(|choice| {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            json!({
                "index":choice.get("index").cloned().unwrap_or(Value::Null),
                "finish_reason":choice.get("finish_reason").cloned().unwrap_or(Value::Null),
                "delta":{
                    "role":delta.get("role").cloned().unwrap_or(Value::Null),
                    "content":delta.get("content").cloned().unwrap_or(Value::Null),
                    "tool_calls":delta.get("tool_calls").cloned().unwrap_or(Value::Null)
                }
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "id":response.get("id").cloned().unwrap_or(Value::Null),
        "object":response.get("object").cloned().unwrap_or_else(|| Value::String("chat.completion.chunk".to_string())),
        "created":required_value(&response,"created")?,
        "model":required_value(&response,"model")?,
        "choices":choices,
        "usage":response.get("usage").map(|usage| chat_usage(Some(usage))).unwrap_or(Value::Null),
        "provider":provider
    }))
}

fn openai_embeddings_response(response: Value) -> Result<Value, GatewayRuntimeError> {
    let data = response
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| GatewayRuntimeError::internal("Provider response is missing data"))?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            json!({
                "object":"embedding",
                "embedding":item.get("embedding").cloned().unwrap_or(Value::Null),
                "index":index
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "object":"list",
        "data":data,
        "model":required_value(&response,"model")?,
        "usage":{
            "prompt_tokens":response.pointer("/usage/prompt_tokens").cloned().unwrap_or(Value::Null),
            "total_tokens":response.pointer("/usage/total_tokens").cloned().unwrap_or(Value::Null)
        }
    }))
}

fn anthropic_chat_request(payload: &mut Value, model: &str) -> Result<(), GatewayRuntimeError> {
    let object = object_mut(payload)?;
    if object.contains_key("temperature") && object.contains_key("top_p") {
        return Err(GatewayRuntimeError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Cannot set both 'temperature' and 'top_p' parameters.",
        ));
    }
    let max_tokens = object
        .remove("max_completion_tokens")
        .or_else(|| object.get("max_tokens").cloned())
        .unwrap_or(json!(8192));
    if max_tokens.as_u64().unwrap_or(0) > 1_000_000 {
        return Err(GatewayRuntimeError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid value for max_tokens: cannot exceed 1000000.",
        ));
    }
    object.insert("model".to_string(), Value::String(model.to_string()));
    object.insert("max_tokens".to_string(), max_tokens);
    if let Some(stop) = object.remove("stop") {
        object.insert("stop_sequences".to_string(), stop);
    }
    let n = object
        .remove("n")
        .and_then(|value| value.as_u64())
        .unwrap_or(1);
    if n != 1 {
        return Err(GatewayRuntimeError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'n' must be '1' for the Anthropic provider. Received value: '{n}'.".to_string(),
        ));
    }
    if let Some(temperature) = object.get_mut("temperature") {
        if let Some(value) = temperature.as_f64() {
            *temperature = json!(value * 0.5);
        }
    }
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let systems = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>();
    if !systems.is_empty() {
        object.insert("system".to_string(), Value::String(systems.join("\n")));
    }
    let mut converted_messages = Vec::new();
    for mut message in messages {
        match message.get("role").and_then(Value::as_str) {
            Some("system") => {}
            Some("user") => converted_messages.push(message),
            Some("assistant") => {
                if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                    let content = tool_calls
                        .iter()
                        .map(|tool_call| {
                            let arguments = tool_call
                                .pointer("/function/arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("null");
                            let input: Value = serde_json::from_str(arguments).map_err(|error| {
                                GatewayRuntimeError::new(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    error.to_string(),
                                )
                            })?;
                            Ok(json!({
                                "type":"tool_use",
                                "id":tool_call.get("id").cloned().unwrap_or(Value::Null),
                                "name":tool_call.pointer("/function/name").cloned().unwrap_or(Value::Null),
                                "input":input
                            }))
                        })
                        .collect::<Result<Vec<_>, GatewayRuntimeError>>()?;
                    if let Some(message) = message.as_object_mut() {
                        message.insert("content".to_string(), Value::Array(content));
                        message.remove("tool_calls");
                    }
                }
                converted_messages.push(message);
            }
            Some("tool") => converted_messages.push(json!({
                "role":"user",
                "content":[{
                    "type":"tool_result",
                    "tool_use_id":message.get("tool_call_id").cloned().unwrap_or(Value::Null),
                    "content":message.get("content").cloned().unwrap_or(Value::Null)
                }]
            })),
            _ => {}
        }
    }
    object.insert("messages".to_string(), Value::Array(converted_messages));

    if let Some(tools) = object
        .remove("tools")
        .and_then(|value| value.as_array().cloned())
    {
        let mut converted_tools = Vec::new();
        for tool in tools {
            let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or("");
            if tool_type != "function" {
                return Err(GatewayRuntimeError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "Only function calling tool is supported, but received tool type {tool_type}"
                    ),
                ));
            }
            let function = tool.get("function").cloned().unwrap_or(Value::Null);
            converted_tools.push(json!({
                "name":function.get("name").cloned().unwrap_or(Value::Null),
                "description":function.get("description").cloned().unwrap_or(Value::Null),
                "input_schema":function.get("parameters").cloned().unwrap_or(Value::Null)
            }));
        }
        object.insert("tools".to_string(), Value::Array(converted_tools));
    }

    if let Some(tool_choice) = object.remove("tool_choice") {
        let converted = match tool_choice {
            Value::String(value) if value == "none" => json!({"type":"none"}),
            Value::String(value) if value == "auto" => json!({"type":"auto"}),
            Value::String(value) if value == "required" => json!({"type":"any"}),
            value if value.get("type").and_then(Value::as_str) == Some("function") => json!({
                "type":"tool",
                "name":value.pointer("/function/name").cloned().unwrap_or(Value::Null)
            }),
            value => value,
        };
        object.insert("tool_choice".to_string(), converted);
    }

    if let Some(response_format) = object.remove("response_format") {
        match response_format.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                if let Some(mut schema) = response_format.pointer("/json_schema/schema").cloned() {
                    if enforce_anthropic_strict_schema(&mut schema) {
                        object.insert(
                            "output_config".to_string(),
                            json!({"format":{"type":"json_schema","schema":schema}}),
                        );
                    }
                }
            }
            Some("json_object") => {
                let instruction = "Respond with only a single valid JSON object. Do not include any explanatory text, markdown, or code fences before or after the JSON object.";
                let system = match object.remove("system") {
                    Some(Value::String(system)) => format!("{system}\n{instruction}"),
                    _ => instruction.to_string(),
                };
                object.insert("system".to_string(), Value::String(system));
            }
            _ => {}
        }
    }
    Ok(())
}

fn enforce_anthropic_strict_schema(value: &mut Value) -> bool {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("object") {
                if !object.contains_key("properties") {
                    return false;
                }
                object.insert("additionalProperties".to_string(), Value::Bool(false));
            }
            object.values_mut().all(enforce_anthropic_strict_schema)
        }
        Value::Array(values) => values.iter_mut().all(enforce_anthropic_strict_schema),
        _ => true,
    }
}

fn anthropic_chat_response(response: Value, now: i64) -> Result<Value, GatewayRuntimeError> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in response
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("thinking") => text.push_str(
                block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ),
            Some("tool_use") => tool_calls.push(json!({
                "id":block.get("id").cloned().unwrap_or(Value::Null),
                "function":{
                    "name":block.get("name").cloned().unwrap_or(Value::Null),
                    "arguments":serde_json::to_string(block.get("input").unwrap_or(&Value::Null)).unwrap()
                },
                "type":"function"
            })),
            _ => {}
        }
    }
    let stop_reason = if response.get("stop_reason").and_then(Value::as_str) == Some("max_tokens") {
        "length"
    } else {
        "stop"
    };
    Ok(json!({
        "id":required_value(&response,"id")?,
        "object":"chat.completion",
        "created":now,
        "model":required_value(&response,"model")?,
        "choices":[{
            "index":0,
            "message":{
                "role":response.get("role").cloned().unwrap_or_else(|| Value::String("assistant".to_string())),
                "content":if text.is_empty() { Value::Null } else { Value::String(text) },
                "tool_calls":if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
                "refusal":Value::Null
            },
            "finish_reason":stop_reason
        }],
        "usage":anthropic_usage(response.get("usage")),
        "provider":"anthropic"
    }))
}

fn anthropic_stream_frames(
    response: Value,
    state: &mut StreamTransformState,
    now: i64,
) -> Result<Vec<Value>, GatewayRuntimeError> {
    match response.get("type").and_then(Value::as_str) {
        Some("message_start") => {
            let message = response.get("message").unwrap_or(&Value::Null);
            state.anthropic_id = message
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string);
            state.anthropic_model = message
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
            if let Some(usage) = message.get("usage").and_then(Value::as_object) {
                state.anthropic_usage.extend(usage.clone());
            }
            Ok(Vec::new())
        }
        Some("content_block_start") | Some("content_block_delta") => {
            let index = response.get("index").and_then(Value::as_u64).unwrap_or(0);
            if !state.anthropic_indices.contains(&index) {
                state.anthropic_indices.push(index);
            }
            Ok(vec![anthropic_stream_frame(response, state, index, now)?])
        }
        Some("message_delta") => {
            if let Some(usage) = response.get("usage").and_then(Value::as_object) {
                state.anthropic_usage.extend(usage.clone());
            }
            state
                .anthropic_indices
                .clone()
                .into_iter()
                .map(|index| anthropic_stream_frame(response.clone(), state, index, now))
                .collect()
        }
        _ => Ok(Vec::new()),
    }
}

fn anthropic_stream_frame(
    response: Value,
    state: &StreamTransformState,
    index: u64,
    now: i64,
) -> Result<Value, GatewayRuntimeError> {
    let content = response
        .get("delta")
        .or_else(|| response.get("content_block"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let delta = match content.get("type").and_then(Value::as_str) {
        Some("tool_use") => json!({
            "role":Value::Null,
            "content":Value::Null,
            "tool_calls":[{
                "index":0,
                "id":content.get("id").cloned().unwrap_or(Value::Null),
                "type":"function",
                "function":{"name":content.get("name").cloned().unwrap_or(Value::Null),"arguments":Value::Null}
            }]
        }),
        Some("input_json_delta") => json!({
            "role":Value::Null,
            "content":Value::Null,
            "tool_calls":[{
                "index":0,"id":Value::Null,"type":Value::Null,
                "function":{"name":Value::Null,"arguments":content.get("partial_json").cloned().unwrap_or(Value::Null)}
            }]
        }),
        _ => json!({
            "role":Value::Null,
            "content":content.get("text").cloned().unwrap_or(Value::Null),
            "tool_calls":Value::Null
        }),
    };
    let finish = content
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(|reason| {
            if reason == "max_tokens" {
                "length"
            } else {
                "stop"
            }
        });
    let usage = if response.get("type").and_then(Value::as_str) == Some("message_delta") {
        anthropic_usage(Some(&Value::Object(state.anthropic_usage.clone())))
    } else {
        Value::Null
    };
    Ok(json!({
        "id":state.anthropic_id.clone().map(Value::String).unwrap_or(Value::Null),
        "object":"chat.completion.chunk",
        "created":now,
        "model":state.anthropic_model.clone().map(Value::String).unwrap_or(Value::Null),
        "choices":[{"index":index,"finish_reason":finish,"delta":delta}],
        "usage":usage,
        "provider":"anthropic"
    }))
}

fn gemini_chat_request(payload: &mut Value) -> Result<Value, GatewayRuntimeError> {
    let object = object_mut(payload)?;
    for (mlflow_key, gemini_key) in [
        ("stop", "stopSequences"),
        ("n", "candidateCount"),
        ("max_tokens", "maxOutputTokens"),
        ("top_k", "topK"),
        ("top_p", "topP"),
        ("frequency_penalty", "frequencyPenalty"),
        ("presence_penalty", "presencePenalty"),
    ] {
        if object.contains_key(gemini_key) {
            return Err(GatewayRuntimeError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Invalid parameter {gemini_key}. Use {mlflow_key} instead."),
            ));
        }
    }
    let messages = object
        .remove("messages")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    let mut contents = Vec::new();
    let mut system_parts = Vec::new();
    let mut call_names = HashMap::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let content = message.get("content").cloned().unwrap_or(Value::Null);
        match role {
            "system" => system_parts.push(json!({"text":content})),
            "user" => contents.push(json!({"role":"user","parts":[{"text":content}]})),
            "assistant" => {
                if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                    let mut parts = Vec::new();
                    for tool_call in tool_calls {
                        let id = tool_call.get("id").and_then(Value::as_str).unwrap_or("");
                        let name = tool_call
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        call_names.insert(id.to_string(), name.to_string());
                        let arguments = tool_call
                            .pointer("/function/arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("null");
                        let args = serde_json::from_str::<Value>(arguments).map_err(|error| {
                            GatewayRuntimeError::new(
                                StatusCode::UNPROCESSABLE_ENTITY,
                                error.to_string(),
                            )
                        })?;
                        let mut function_call = Map::new();
                        function_call.insert("id".to_string(), Value::String(id.to_string()));
                        function_call.insert("name".to_string(), Value::String(name.to_string()));
                        function_call.insert("args".to_string(), args);
                        if let Some(signature) = tool_call.get("thought_signature") {
                            function_call.insert("thoughtSignature".to_string(), signature.clone());
                        }
                        parts.push(json!({"functionCall":function_call}));
                    }
                    contents.push(json!({"role":"model","parts":parts}));
                } else {
                    contents.push(json!({"role":"model","parts":[{"text":content}]}));
                }
            }
            "tool" => {
                let call_id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let response = message
                    .get("content")
                    .and_then(Value::as_str)
                    .and_then(|content| serde_json::from_str::<Value>(content).ok())
                    .unwrap_or(Value::Null);
                contents.push(json!({
                    "role":"user",
                    "parts":[{"functionResponse":{
                        "id":call_id,
                        "name":call_names.get(call_id).cloned().unwrap_or_default(),
                        "response":response
                    }}]
                }));
            }
            _ => {}
        }
    }
    let mappings = [
        ("stop", "stopSequences"),
        ("n", "candidateCount"),
        ("max_tokens", "maxOutputTokens"),
        ("top_k", "topK"),
        ("top_p", "topP"),
        ("frequency_penalty", "frequencyPenalty"),
        ("presence_penalty", "presencePenalty"),
    ];
    if !object.contains_key("max_tokens") {
        if let Some(value) = object.remove("max_completion_tokens") {
            object.insert("max_tokens".to_string(), value);
        }
    }
    let mut generation = Map::new();
    if let Some(value) = object.remove("temperature") {
        generation.insert("temperature".to_string(), value);
    }
    for (source, target) in mappings {
        if let Some(value) = object.remove(source) {
            generation.insert(target.to_string(), value);
        }
    }
    let mut result = Map::new();
    result.insert("contents".to_string(), Value::Array(contents));
    if !system_parts.is_empty() {
        result.insert(
            "system_instruction".to_string(),
            json!({"parts":system_parts}),
        );
    }
    if !generation.is_empty() {
        result.insert("generationConfig".to_string(), Value::Object(generation));
    }

    if let Some(response_format) = object.remove("response_format") {
        let generation = result
            .entry("generationConfig".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        let generation = generation
            .as_object_mut()
            .expect("generationConfig is always an object");
        match response_format.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                if let Some(schema) = response_format.pointer("/json_schema/schema") {
                    generation.insert("responseJsonSchema".to_string(), schema.clone());
                    generation.insert(
                        "responseMimeType".to_string(),
                        Value::String("application/json".to_string()),
                    );
                }
            }
            Some("json_object") => {
                generation.insert(
                    "responseMimeType".to_string(),
                    Value::String("application/json".to_string()),
                );
            }
            _ => {}
        }
    }

    if let Some(tools) = object
        .remove("tools")
        .and_then(|value| value.as_array().cloned())
    {
        let mut declarations = Vec::new();
        for tool in tools {
            let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or("");
            if tool_type != "function" {
                return Err(GatewayRuntimeError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "Only function calling tool is supported, but received tool type {tool_type}"
                    ),
                ));
            }
            let function = tool.get("function").cloned().unwrap_or(Value::Null);
            declarations.push(json!({
                "name":function.get("name").cloned().unwrap_or(Value::Null),
                "description":function.get("description").cloned().unwrap_or(Value::Null),
                "parametersJsonSchema":function.get("parameters").cloned().unwrap_or(Value::Null)
            }));
        }
        result.insert(
            "tools".to_string(),
            json!([{"functionDeclarations":declarations}]),
        );
    }
    Ok(Value::Object(result))
}

fn gemini_chat_response(
    response: Value,
    model: &str,
    now: i64,
) -> Result<Value, GatewayRuntimeError> {
    let choices = gemini_choices(&response, false)?;
    Ok(json!({
        "id":format!("gemini-chat-{now}"),
        "object":"chat.completion",
        "created":now,
        "model":model,
        "choices":choices,
        "usage":gemini_usage(response.get("usageMetadata")),
        "provider":"gemini"
    }))
}

fn gemini_chat_stream_frame(
    response: Value,
    model: &str,
    now: i64,
) -> Result<Value, GatewayRuntimeError> {
    Ok(json!({
        "id":format!("gemini-chat-stream-{now}"),
        "object":"chat.completion.chunk",
        "created":now,
        "model":model,
        "choices":gemini_choices(&response,true)?,
        "usage":response.get("usageMetadata").map(|usage| gemini_usage(Some(usage))).unwrap_or(Value::Null),
        "provider":"gemini"
    }))
}

fn gemini_choices(response: &Value, stream: bool) -> Result<Vec<Value>, GatewayRuntimeError> {
    Ok(response
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let parts = candidate.pointer("/content/parts")?.as_array()?;
            if parts
                .first()
                .and_then(|part| part.get("functionCall"))
                .is_some()
            {
                return Some(gemini_function_choice(
                    parts,
                    normalize_finish_reason(candidate.get("finishReason")).or_else(|| {
                        (!stream).then(|| Value::String("stop".to_string()))
                    }),
                    index,
                    stream,
                ));
            }
            let text = parts.first().and_then(|part| part.get("text"));
            let finish = normalize_finish_reason(candidate.get("finishReason"));
            if stream {
                Some(json!({
                    "index":index,
                    "finish_reason":finish,
                    "delta":{"role":"assistant","content":text.cloned().unwrap_or_else(|| Value::String(String::new())),"tool_calls":Value::Null}
                }))
            } else {
                text.map(|text| json!({
                    "index":index,
                    "message":{"role":"assistant","content":text,"tool_calls":Value::Null,"refusal":Value::Null},
                    "finish_reason":finish.unwrap_or_else(|| Value::String("stop".to_string()))
                }))
            }
        })
        .collect())
}

fn gemini_function_choice(
    parts: &[Value],
    finish_reason: Option<Value>,
    index: usize,
    stream: bool,
) -> Value {
    let tool_calls = parts
        .iter()
        .filter_map(|part| part.get("functionCall"))
        .map(|call| {
            let name = call.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = python_json_dumps(call.get("args").unwrap_or(&Value::Null), false);
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    let mut hasher = Md5::new();
                    hasher.update(format!("{name}/{arguments}"));
                    format!("call_{:x}", hasher.finalize())
                });
            let signature = call
                .get("thoughtSignature")
                .or_else(|| call.get("thought_signature"))
                .filter(|value| !value.is_null())
                .cloned();
            let mut result = Map::new();
            if stream {
                result.insert("index".to_string(), json!(0));
            }
            result.insert("id".to_string(), Value::String(id));
            result.insert("type".to_string(), Value::String("function".to_string()));
            result.insert(
                "function".to_string(),
                json!({"name":name,"arguments":arguments}),
            );
            if let Some(signature) = signature {
                result.insert("thought_signature".to_string(), signature);
            }
            Value::Object(result)
        })
        .collect::<Vec<_>>();
    if stream {
        json!({
            "index":index,
            "finish_reason":finish_reason,
            "delta":{"role":"assistant","content":Value::Null,"tool_calls":tool_calls}
        })
    } else {
        json!({
            "index":index,
            "message":{"role":"assistant","content":Value::Null,"tool_calls":tool_calls,"refusal":Value::Null},
            "finish_reason":finish_reason
        })
    }
}

fn gemini_embeddings_request(
    payload: &Value,
    model: &str,
) -> Result<(Value, String), GatewayRuntimeError> {
    let input = required_value(payload, "input")?;
    let values = match input {
        Value::String(_) => vec![input],
        Value::Array(values) => values,
        _ => Vec::new(),
    };
    if values.len() == 1 {
        Ok((
            json!({"content":{"parts":[{"text":values[0]}]}}),
            format!("{model}:embedContent"),
        ))
    } else {
        Ok((
            json!({"requests":values.into_iter().map(|text| json!({"model":format!("models/{model}"),"content":{"parts":[{"text":text}]}})).collect::<Vec<_>>() }),
            format!("{model}:batchEmbedContents"),
        ))
    }
}

fn gemini_embeddings_response(response: Value, model: &str) -> Result<Value, GatewayRuntimeError> {
    let embeddings = response
        .get("embeddings")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![response.get("embedding").cloned().unwrap_or(json!({}))]);
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({"object":"embedding","embedding":embedding.get("values").cloned().unwrap_or_else(|| json!([])),"index":index})
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "object":"list","data":data,"model":model,
        "usage":{"prompt_tokens":Value::Null,"total_tokens":Value::Null}
    }))
}

fn chat_usage(usage: Option<&Value>) -> Value {
    let usage = usage.unwrap_or(&Value::Null);
    let mut result = Map::new();
    result.insert(
        "prompt_tokens".to_string(),
        usage.get("prompt_tokens").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "completion_tokens".to_string(),
        usage
            .get("completion_tokens")
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert(
        "total_tokens".to_string(),
        usage.get("total_tokens").cloned().unwrap_or(Value::Null),
    );
    if let Some(details) = usage.get("prompt_tokens_details") {
        result.insert("prompt_tokens_details".to_string(), details.clone());
    }
    Value::Object(result)
}

fn anthropic_usage(usage: Option<&Value>) -> Value {
    let usage = usage.unwrap_or(&Value::Null);
    let input = usage.get("input_tokens").and_then(Value::as_u64);
    let output = usage.get("output_tokens").and_then(Value::as_u64);
    let cached = usage.get("cache_read_input_tokens").and_then(Value::as_u64);
    let created = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64);
    let prompt = input.map(|value| value + cached.unwrap_or(0) + created.unwrap_or(0));
    let mut result = Map::new();
    result.insert(
        "prompt_tokens".to_string(),
        prompt.map_or(Value::Null, Value::from),
    );
    result.insert(
        "completion_tokens".to_string(),
        output.map_or(Value::Null, Value::from),
    );
    result.insert(
        "total_tokens".to_string(),
        prompt
            .zip(output)
            .map(|(prompt, output)| Value::from(prompt + output))
            .unwrap_or(Value::Null),
    );
    if let Some(cached) = cached {
        result.insert(
            "prompt_tokens_details".to_string(),
            json!({"cached_tokens":cached}),
        );
    }
    if let Some(created) = created {
        result.insert("cache_creation_input_tokens".to_string(), json!(created));
    }
    Value::Object(result)
}

fn gemini_usage(usage: Option<&Value>) -> Value {
    let usage = usage.unwrap_or(&Value::Null);
    let mut result = Map::new();
    result.insert(
        "prompt_tokens".to_string(),
        usage
            .get("promptTokenCount")
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert(
        "completion_tokens".to_string(),
        usage
            .get("candidatesTokenCount")
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert(
        "total_tokens".to_string(),
        usage.get("totalTokenCount").cloned().unwrap_or(Value::Null),
    );
    if let Some(cached) = usage.get("cachedContentTokenCount") {
        result.insert(
            "prompt_tokens_details".to_string(),
            json!({"cached_tokens":cached}),
        );
    }
    Value::Object(result)
}

fn normalize_finish_reason(value: Option<&Value>) -> Option<Value> {
    value.and_then(Value::as_str).map(|reason| {
        Value::String(if reason == "MAX_TOKENS" {
            "length".to_string()
        } else {
            reason.to_lowercase()
        })
    })
}

fn openai_wire_provider(model: &ResolvedGatewayModelConfig) -> &str {
    if matches!(model.provider.as_str(), "azure" | "azure-openai") {
        "openai"
    } else {
        &model.provider
    }
}

fn object_mut(value: &mut Value) -> Result<&mut Map<String, Value>, GatewayRuntimeError> {
    value
        .as_object_mut()
        .ok_or_else(|| GatewayRuntimeError::internal("Gateway payload must be an object"))
}

fn required_value(value: &Value, key: &str) -> Result<Value, GatewayRuntimeError> {
    value
        .get(key)
        .cloned()
        .ok_or_else(|| GatewayRuntimeError::internal(format!("Provider response is missing {key}")))
}

fn required_auth<'a>(
    model: &'a ResolvedGatewayModelConfig,
    key: &str,
) -> Result<&'a str, GatewayRuntimeError> {
    model
        .auth_config
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| {
            GatewayRuntimeError::internal(format!("Missing provider auth config: {key}"))
        })
}

fn secret_string<'a>(
    model: &'a ResolvedGatewayModelConfig,
    key: &str,
) -> Result<&'a str, GatewayRuntimeError> {
    model
        .secret_value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| GatewayRuntimeError::internal(format!("Missing provider secret: {key}")))
}

fn parse_url(value: &str) -> Result<Url, GatewayRuntimeError> {
    Url::parse(value).map_err(|error| GatewayRuntimeError::internal(error.to_string()))
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<(), GatewayRuntimeError> {
    let value = HeaderValue::from_str(value)
        .map_err(|error| GatewayRuntimeError::internal(error.to_string()))?;
    headers.insert(HeaderName::from_static(name), value);
    Ok(())
}

fn client() -> reqwest::Client {
    let timeout = std::env::var(ROUTE_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(300);
    reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .default_headers(HeaderMap::from_iter([(
            header::ACCEPT_ENCODING,
            HeaderValue::from_static(ACCEPT_ENCODING),
        )]))
        .build()
        .expect("static reqwest client configuration")
}

fn json_response(status: StatusCode, value: Value) -> Response {
    let mut response = Response::new(Body::from(
        serde_json::to_vec(&value).expect("JSON values always serialize"),
    ));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

fn insert_timing_header(response: &mut Response, name: &'static str, value: u128) {
    response.headers_mut().insert(
        HeaderName::from_static(name),
        HeaderValue::from_str(&value.to_string()).expect("integer header value"),
    );
}

fn sse_json(value: &Value) -> Bytes {
    Bytes::from(format!(
        "data: {}\n\n",
        serde_json::to_string(value).expect("JSON value")
    ))
}

fn sse_error(message: &str, error_type: &str) -> Bytes {
    let message = serde_json::to_string(message).expect("JSON string");
    let error_type = serde_json::to_string(error_type).expect("JSON string");
    Bytes::from(format!(
        "data: {{\"error\": {{\"message\": {message}, \"type\": {error_type}}}}}\n\n"
    ))
}

fn unix_seconds() -> i64 {
    if let Ok(value) = std::env::var(TEST_FIXED_TIME_ENV) {
        if let Ok(value) = value.parse() {
            return value;
        }
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture_model(provider: &str) -> ResolvedGatewayModelConfig {
        ResolvedGatewayModelConfig {
            model_definition_id: "d-obvious-fake".to_string(),
            provider: provider.to_string(),
            model_name: "fixture-model".to_string(),
            secret_value: json!({"api_key":"obvious-fake-key"}),
            auth_config: HashMap::from([(
                "api_base".to_string(),
                "http://127.0.0.1:9/v1".to_string(),
            )]),
            weight: 1.0,
            linkage_type: "PRIMARY".to_string(),
            fallback_order: None,
        }
    }

    #[test]
    fn openai_adapter_pins_request_response_stream_and_auth() {
        let model = fixture_model("openai");
        let request = OpenAiAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}],"stream":true}),
                true,
            )
            .unwrap();
        assert_eq!(request.body["model"], "fixture-model");
        assert_eq!(request.body["stream_options"]["include_usage"], true);
        assert_eq!(request.headers["authorization"], "Bearer obvious-fake-key");

        let response = OpenAiAdapter
            .transform_response(
                &model,
                InvocationKind::Chat,
                json!({"id":"c1","object":"chat.completion","created":7,"model":"fixture-model","choices":[{"index":4,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}),
                7,
            )
            .unwrap();
        assert_eq!(response["choices"][0]["index"], 0);
        assert_eq!(response["provider"], "openai");

        let stream = OpenAiAdapter
            .transform_stream_frame(
                &model,
                json!({"id":"c1","created":7,"model":"fixture-model","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}),
                &mut StreamTransformState::default(),
                7,
            )
            .unwrap();
        assert_eq!(stream[0]["choices"][0]["delta"]["content"], "hi");
        assert_eq!(
            OpenAiAdapter
                .map_error(StatusCode::BAD_REQUEST, json!({"error":{"message":"bad"}}))
                .detail,
            "bad"
        );

        let mut azure = fixture_model("azure");
        azure.auth_config.extend([
            ("api_type".to_string(), "azure".to_string()),
            ("api_version".to_string(), "2025-01-01".to_string()),
        ]);
        let request = OpenAiAdapter
            .transform_request(
                &azure,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                false,
            )
            .unwrap();
        assert!(request
            .url
            .as_str()
            .contains("/openai/deployments/fixture-model/"));
        assert_eq!(request.headers["api-key"], "obvious-fake-key");
        assert!(request.body.get("model").is_none());
    }

    #[test]
    fn anthropic_adapter_pins_request_response_stream_error_and_auth() {
        let model = fixture_model("anthropic");
        let request = AnthropicAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"system","content":"system"},{"role":"user","content":"hi"}],"temperature":2.0}),
                false,
            )
            .unwrap();
        assert_eq!(request.body["system"], "system");
        assert_eq!(request.body["temperature"], 1.0);
        assert_eq!(request.body["max_tokens"], 8192);
        assert_eq!(request.headers["x-api-key"], "obvious-fake-key");

        let response = AnthropicAdapter
            .transform_response(
                &model,
                InvocationKind::Chat,
                json!({"id":"a1","model":"fixture-model","role":"assistant","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}),
                11,
            )
            .unwrap();
        assert_eq!(response["created"], 11);
        assert_eq!(response["usage"]["total_tokens"], 5);

        let mut state = StreamTransformState::default();
        assert!(AnthropicAdapter
            .transform_stream_frame(
                &model,
                json!({"type":"message_start","message":{"id":"a1","model":"fixture-model","usage":{"input_tokens":2}}}),
                &mut state,
                11,
            )
            .unwrap()
            .is_empty());
        let stream = AnthropicAdapter
            .transform_stream_frame(
                &model,
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}),
                &mut state,
                11,
            )
            .unwrap();
        assert_eq!(stream[0]["choices"][0]["delta"]["content"], "hi");
        assert_eq!(
            AnthropicAdapter
                .map_error(StatusCode::BAD_REQUEST, json!({"error":{"message":"bad"}}))
                .detail,
            "bad"
        );
    }

    #[test]
    fn gemini_adapter_pins_request_response_stream_error_and_auth() {
        let model = fixture_model("gemini");
        let request = GeminiAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":4}),
                false,
            )
            .unwrap();
        assert_eq!(request.body["contents"][0]["parts"][0]["text"], "hi");
        assert_eq!(request.body["generationConfig"]["maxOutputTokens"], 4);
        assert_eq!(request.headers["x-goog-api-key"], "obvious-fake-key");

        let response = GeminiAdapter
            .transform_response(
                &model,
                InvocationKind::Chat,
                json!({"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}),
                13,
            )
            .unwrap();
        assert_eq!(response["id"], "gemini-chat-13");
        assert_eq!(response["choices"][0]["finish_reason"], "stop");

        let stream = GeminiAdapter
            .transform_stream_frame(
                &model,
                json!({"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}),
                &mut StreamTransformState::default(),
                13,
            )
            .unwrap();
        assert_eq!(stream[0]["choices"][0]["delta"]["content"], "hi");
        assert_eq!(
            GeminiAdapter
                .map_error(StatusCode::BAD_REQUEST, json!({"error":{"message":"bad"}}))
                .detail,
            "bad"
        );
    }

    #[test]
    fn sse_framing_is_exact_and_done_is_not_emitted() {
        assert_eq!(
            sse_json(&json!({"delta":"hi"})),
            Bytes::from_static(b"data: {\"delta\":\"hi\"}\n\n")
        );
        let error = sse_error("broken", "FixtureError");
        assert_eq!(
            error,
            Bytes::from_static(
                b"data: {\"error\": {\"message\": \"broken\", \"type\": \"FixtureError\"}}\n\n"
            )
        );
    }

    #[test]
    fn error_mapping_uses_provider_error_message() {
        let error = OpenAiAdapter.map_error(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error":{"message":"fixture limit"}}),
        );
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.detail, "fixture limit");
    }
}
