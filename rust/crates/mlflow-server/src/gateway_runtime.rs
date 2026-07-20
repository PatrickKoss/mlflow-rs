//! Native database-backed AI Gateway invocation runtime.
//!
//! The T18.3 unified runtime and T18.4 provider matrix deliberately share one
//! provider trait, transport choke point, and SSE implementation.

use std::collections::HashMap;
use std::convert::Infallible;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{OriginalUri, Path, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::Response;
use base64::Engine;
use chrono::Utc;
use futures::future::BoxFuture;
use futures::{stream, FutureExt, StreamExt};
use hmac::{Hmac, Mac};
use md5::{Digest, Md5};
use mlflow_store::{
    python_json_dumps, ResolvedGatewayEndpointConfig, ResolvedGatewayModelConfig, SpanInput,
    SpanMetricInput, StartTraceInput, TraceTimeRange,
};
use mlflow_webhooks::{WebhookAction, WebhookEntity, WebhookEvent};
use rand::Rng;
use reqwest::Url;
use serde_json::{json, Map, Value};
use sha2::Sha256;

use crate::budget::{exceeded_payload, refresh_from_store, reject_message};
use crate::gateway_guardrails::{
    load_guardrails, GuardrailExecutionError, GuardrailPayloadSchema, LoadedGuardrails,
};
use crate::gateway_provider_matrix::{
    default_api_base, is_supported_provider, model_accounting, normalize_provider,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthroughAction {
    OpenAiChat,
    OpenAiEmbeddings,
    OpenAiResponses,
    OpenAiResponsesCompact,
    AnthropicMessages,
    GeminiGenerateContent,
    GeminiStreamGenerateContent,
}

impl PassthroughAction {
    fn provider_path(self, model: &str) -> String {
        match self {
            Self::OpenAiChat => "chat/completions".to_string(),
            Self::OpenAiEmbeddings => "embeddings".to_string(),
            Self::OpenAiResponses => "responses".to_string(),
            Self::OpenAiResponsesCompact => "responses/compact".to_string(),
            Self::AnthropicMessages => "messages".to_string(),
            Self::GeminiGenerateContent => format!("{model}:generateContent"),
            Self::GeminiStreamGenerateContent => format!("{model}:streamGenerateContent"),
        }
    }

    fn streaming(self, payload: &Value) -> bool {
        self == Self::GeminiStreamGenerateContent
            || (self != Self::OpenAiEmbeddings
                && payload.get("stream").and_then(Value::as_bool) == Some(true))
    }
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

    fn propagates_fallback_status(&self) -> bool {
        matches!(self.stream_type, "AIGatewayException" | "HTTPException")
    }

    fn fallback_message(&self) -> String {
        match &self.detail {
            Value::String(message) if self.stream_type == "HTTPException" => {
                format!("{}: {message}", self.status.as_u16())
            }
            Value::String(message) => message.clone(),
            detail if self.stream_type == "HTTPException" => {
                format!("{}: {detail}", self.status.as_u16())
            }
            detail => detail.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
enum PrimaryRoute {
    Single(Box<ResolvedGatewayModelConfig>),
    TrafficSplit(Vec<ResolvedGatewayModelConfig>),
}

#[derive(Debug, Clone)]
struct RoutingPlan {
    primary: PrimaryRoute,
    fallbacks: Vec<ResolvedGatewayModelConfig>,
    fallback_attempt_label: Option<i64>,
    attempt_limit: usize,
}

#[derive(Clone)]
struct GatewayTraceContext {
    state: AppState,
    workspace: String,
    endpoint: ResolvedGatewayEndpointConfig,
    request: Value,
    request_type: &'static str,
    started_ns: i64,
}

#[derive(Debug, Clone, Copy)]
struct TokenUsage {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Clone, Copy)]
struct TokenCost {
    input_cost: f64,
    output_cost: f64,
    total_cost: f64,
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

    fn passthrough_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        action: PassthroughAction,
        mut payload: Value,
        client_headers: &HeaderMap,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        let object = object_mut(&mut payload)?;
        object.insert("model".to_string(), Value::String(model.model_name.clone()));
        let base = provider_api_base(model)?;
        let mut headers = merged_passthrough_headers(model, client_headers)?;
        if !has_provider_auth(&headers) {
            self.inject_auth(model, &mut headers)?;
        }
        Ok(ProviderRequest {
            url: append_provider_path(&base, &action.provider_path(&model.model_name))?,
            headers,
            body: payload,
        })
    }

    fn proxy_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        path: &str,
        payload: Value,
        client_headers: &HeaderMap,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        let base = proxy_root(&provider_api_base(model)?)?;
        let mut headers = merged_passthrough_headers(model, client_headers)?;
        if !has_provider_auth(&headers) {
            self.inject_auth(model, &mut headers)?;
        }
        Ok(ProviderRequest {
            url: append_provider_path(&base, path)?,
            headers,
            body: payload,
        })
    }
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

#[derive(Debug)]
pub struct OpenAiCompatibleAdapter {
    provider: String,
}

#[derive(Debug)]
pub struct BedrockAdapter;

#[derive(Debug)]
pub struct DatabricksAdapter;

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

    fn passthrough_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        action: PassthroughAction,
        payload: Value,
        client_headers: &HeaderMap,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        let mut headers = merged_passthrough_headers(model, client_headers)?;
        if !has_provider_auth(&headers) {
            self.inject_auth(model, &mut headers)?;
        }
        Ok(ProviderRequest {
            url: append_provider_path(
                &provider_api_base(model)?,
                &action.provider_path(&model.model_name),
            )?,
            headers,
            body: payload,
        })
    }
}

impl GatewayProviderAdapter for OpenAiCompatibleAdapter {
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
        object.insert("model".to_string(), Value::String(model.model_name.clone()));
        if stream && kind == InvocationKind::Chat {
            let options = object
                .entry("stream_options")
                .or_insert_with(|| Value::Object(Map::new()));
            if let Some(options) = options.as_object_mut() {
                options.entry("include_usage").or_insert(Value::Bool(true));
            }
        }
        let route = match kind {
            InvocationKind::Chat => "chat/completions",
            InvocationKind::Embeddings => "embeddings",
        };
        let mut headers = HeaderMap::new();
        self.inject_auth(model, &mut headers)?;
        Ok(ProviderRequest {
            url: append_provider_path(&provider_api_base(model)?, route)?,
            headers,
            body: payload,
        })
    }

    fn transform_response(
        &self,
        _model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        response: Value,
        _now: i64,
    ) -> Result<Value, GatewayRuntimeError> {
        match kind {
            InvocationKind::Chat => openai_chat_response(response, &self.provider),
            InvocationKind::Embeddings => openai_embeddings_response(response),
        }
    }

    fn transform_stream_frame(
        &self,
        _model: &ResolvedGatewayModelConfig,
        frame: Value,
        _state: &mut StreamTransformState,
        _now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError> {
        Ok(vec![openai_chat_stream_frame(frame, &self.provider)?])
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError> {
        match normalize_provider(&self.provider) {
            "ollama" => {
                if let Some(key) = secret_string_optional(model, "api_key") {
                    if key != "ollama" {
                        insert_header(headers, "authorization", &format!("Bearer {key}"))?;
                    }
                }
                Ok(())
            }
            "portkey" => insert_header(
                headers,
                "x-portkey-api-key",
                secret_string(model, "api_key")?,
            ),
            _ => insert_header(
                headers,
                "authorization",
                &format!("Bearer {}", secret_string(model, "api_key")?),
            ),
        }
    }
}

impl GatewayProviderAdapter for DatabricksAdapter {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn transform_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        payload: Value,
        stream: bool,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        OpenAiCompatibleAdapter {
            provider: "databricks".to_string(),
        }
        .transform_request(model, kind, payload, stream)
    }

    fn transform_response(
        &self,
        _model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        mut response: Value,
        _now: i64,
    ) -> Result<Value, GatewayRuntimeError> {
        if kind == InvocationKind::Chat {
            normalize_databricks_content(&mut response);
            openai_chat_response(response, "databricks")
        } else {
            openai_embeddings_response(response)
        }
    }

    fn transform_stream_frame(
        &self,
        _model: &ResolvedGatewayModelConfig,
        mut frame: Value,
        _state: &mut StreamTransformState,
        _now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError> {
        normalize_databricks_content(&mut frame);
        Ok(vec![openai_chat_stream_frame(frame, "databricks")?])
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError> {
        if model.auth_config.get("auth_mode").map(String::as_str) == Some("oauth_m2m") {
            return Ok(());
        }
        let environment_token = std::env::var("DATABRICKS_TOKEN").ok();
        let token = secret_string_optional(model, "api_key")
            .or(environment_token.as_deref())
            .ok_or_else(|| missing_auth("api_key"))?;
        insert_header(headers, "authorization", &format!("Bearer {token}"))
    }

    fn passthrough_request(
        &self,
        model: &ResolvedGatewayModelConfig,
        action: PassthroughAction,
        mut payload: Value,
        client_headers: &HeaderMap,
    ) -> Result<ProviderRequest, GatewayRuntimeError> {
        let path = match action {
            PassthroughAction::OpenAiChat => "chat/completions".to_string(),
            PassthroughAction::OpenAiEmbeddings => "embeddings".to_string(),
            PassthroughAction::OpenAiResponses => "responses".to_string(),
            PassthroughAction::OpenAiResponsesCompact => "responses/compact".to_string(),
            PassthroughAction::AnthropicMessages => "anthropic/v1/messages".to_string(),
            PassthroughAction::GeminiGenerateContent => {
                format!("gemini/v1beta/models/{}:generateContent", model.model_name)
            }
            PassthroughAction::GeminiStreamGenerateContent => format!(
                "gemini/v1beta/models/{}:streamGenerateContent",
                model.model_name
            ),
        };
        if !matches!(
            action,
            PassthroughAction::GeminiGenerateContent
                | PassthroughAction::GeminiStreamGenerateContent
        ) {
            object_mut(&mut payload)?
                .insert("model".to_string(), Value::String(model.model_name.clone()));
        }
        let mut headers = merged_passthrough_headers(model, client_headers)?;
        if !has_provider_auth(&headers) {
            self.inject_auth(model, &mut headers)?;
        }
        Ok(ProviderRequest {
            url: append_provider_path(&provider_api_base(model)?, &path)?,
            headers,
            body: payload,
        })
    }
}

impl GatewayProviderAdapter for BedrockAdapter {
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
        if kind == InvocationKind::Embeddings {
            let object = object_mut(&mut payload)?;
            let input = object.remove("input").unwrap_or(Value::Null);
            payload = json!({"inputText": input});
        } else if model.model_name.contains("anthropic") || model.model_name.contains("claude") {
            anthropic_chat_request(&mut payload, &model.model_name)?;
            let object = object_mut(&mut payload)?;
            object.remove("model");
            object.insert(
                "anthropic_version".to_string(),
                Value::String("bedrock-2023-05-31".to_string()),
            );
            if let Some(max_tokens) = object.get_mut("max_tokens") {
                if max_tokens.as_u64().is_some_and(|value| value > 8_191) {
                    *max_tokens = json!(8_191);
                }
            }
        }
        let region = bedrock_region(model);
        let base = model
            .auth_config
            .get("api_base")
            .cloned()
            .unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com"));
        let suffix = if stream {
            format!("model/{}/invoke-with-response-stream", model.model_name)
        } else {
            format!("model/{}/invoke", model.model_name)
        };
        let mut request = ProviderRequest {
            url: append_provider_path(&parse_url(&base)?, &suffix)?,
            headers: HeaderMap::new(),
            body: payload,
        };
        self.inject_auth(model, &mut request.headers)?;
        if !matches!(
            model.auth_config.get("auth_mode").map(String::as_str),
            Some("api_key" | "iam_role")
        ) {
            sign_bedrock_request(model, &mut request)?;
        }
        Ok(request)
    }

    fn transform_response(
        &self,
        model: &ResolvedGatewayModelConfig,
        kind: InvocationKind,
        response: Value,
        now: i64,
    ) -> Result<Value, GatewayRuntimeError> {
        if kind == InvocationKind::Embeddings {
            return gemini_embeddings_response(
                json!({"embedding":{"values":response.get("embedding").cloned().unwrap_or(Value::Null)}}),
                &model.model_name,
            );
        }
        if model.model_name.contains("anthropic") || model.model_name.contains("claude") {
            anthropic_chat_response(response, now)
        } else {
            openai_chat_response(response, "bedrock")
        }
    }

    fn transform_stream_frame(
        &self,
        _model: &ResolvedGatewayModelConfig,
        frame: Value,
        _state: &mut StreamTransformState,
        _now: i64,
    ) -> Result<Vec<Value>, GatewayRuntimeError> {
        Ok(vec![frame])
    }

    fn inject_auth(
        &self,
        model: &ResolvedGatewayModelConfig,
        headers: &mut HeaderMap,
    ) -> Result<(), GatewayRuntimeError> {
        if model
            .auth_config
            .get("auth_mode")
            .map(String::as_str)
            .unwrap_or("api_key")
            == "api_key"
        {
            insert_header(
                headers,
                "authorization",
                &format!("Bearer {}", secret_string(model, "api_key")?),
            )?;
        }
        Ok(())
    }
}

pub async fn openai_passthrough_chat(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_model_route(
        state,
        workspace,
        headers,
        body,
        PassthroughAction::OpenAiChat,
    )
    .await
}

pub async fn openai_passthrough_embeddings(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_model_route(
        state,
        workspace,
        headers,
        body,
        PassthroughAction::OpenAiEmbeddings,
    )
    .await
}

pub async fn openai_passthrough_responses(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_model_route(
        state,
        workspace,
        headers,
        body,
        PassthroughAction::OpenAiResponses,
    )
    .await
}

pub async fn openai_passthrough_responses_compact(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_model_route(
        state,
        workspace,
        headers,
        body,
        PassthroughAction::OpenAiResponsesCompact,
    )
    .await
}

pub async fn anthropic_passthrough_messages(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_model_route(
        state,
        workspace,
        headers,
        body,
        PassthroughAction::AnthropicMessages,
    )
    .await
}

pub async fn gemini_passthrough(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    Path(model_action): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let (endpoint_name, action) =
        if let Some(endpoint_name) = model_action.strip_suffix(":streamGenerateContent") {
            (
                endpoint_name,
                PassthroughAction::GeminiStreamGenerateContent,
            )
        } else if let Some(endpoint_name) = model_action.strip_suffix(":generateContent") {
            (endpoint_name, PassthroughAction::GeminiGenerateContent)
        } else {
            return json_response(StatusCode::NOT_FOUND, json!({"detail":"Not Found"}));
        };
    passthrough_path_route(
        state,
        workspace,
        endpoint_name.to_string(),
        headers,
        body,
        action,
    )
    .await
}

pub async fn gemini_passthrough_generate_content(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    Path(endpoint_name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_path_route(
        state,
        workspace,
        endpoint_name,
        headers,
        body,
        PassthroughAction::GeminiGenerateContent,
    )
    .await
}

pub async fn gemini_passthrough_stream_generate_content(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    Path(endpoint_name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    passthrough_path_route(
        state,
        workspace,
        endpoint_name,
        headers,
        body,
        PassthroughAction::GeminiStreamGenerateContent,
    )
    .await
}

async fn passthrough_model_route(
    state: AppState,
    workspace: String,
    headers: HeaderMap,
    body: Bytes,
    action: PassthroughAction,
) -> Response {
    let start = Instant::now();
    let mut payload = match parse_body(&body) {
        Ok(payload) => payload,
        Err(error) => return error.response(start.elapsed()),
    };
    if action == PassthroughAction::OpenAiResponsesCompact
        && payload.get("stream").and_then(Value::as_bool) == Some(true)
    {
        return GatewayRuntimeError::http(
            StatusCode::BAD_REQUEST,
            Value::String(
                "stream=true is not supported on /responses/compact; compaction is a unary request."
                    .to_string(),
            ),
        )
        .response(start.elapsed());
    }
    let endpoint_name = match payload
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
    payload
        .as_object_mut()
        .expect("validated object")
        .remove("model");
    passthrough_value_route(
        state,
        workspace,
        endpoint_name,
        headers,
        payload,
        action,
        start,
    )
    .await
}

async fn passthrough_path_route(
    state: AppState,
    workspace: String,
    endpoint_name: String,
    headers: HeaderMap,
    body: Bytes,
    action: PassthroughAction,
) -> Response {
    let start = Instant::now();
    let payload = match parse_body(&body) {
        Ok(payload) => payload,
        Err(error) => return error.response(start.elapsed()),
    };
    passthrough_value_route(
        state,
        workspace,
        endpoint_name,
        headers,
        payload,
        action,
        start,
    )
    .await
}

async fn passthrough_value_route(
    state: AppState,
    workspace: String,
    endpoint_name: String,
    headers: HeaderMap,
    payload: Value,
    action: PassthroughAction,
    start: Instant,
) -> Response {
    let (endpoint, model, adapter) =
        match resolve_runtime_endpoint_provider(&state, &workspace, &endpoint_name).await {
            Ok(value) => value,
            Err(error) => return error.response(start.elapsed()),
        };
    if let Some(response) = check_budget_limit(&state, &workspace, &endpoint, start).await {
        return response;
    }
    let trace_started_ns = unix_nanos();
    if !supports_passthrough(&model.provider, action) {
        return unsupported_passthrough(&model.provider, action).response(start.elapsed());
    }
    let streaming = action.streaming(&payload);
    // Traces record the client's original request even when a sanitizing
    // guardrail rewrites the payload before the provider call.
    let original_request = payload.clone();
    let guardrails = load_guardrails(&state, &workspace, &endpoint, &headers).await;
    let payload = match guardrails.before(payload, None).await {
        Ok(payload) => payload,
        Err(error) if streaming => {
            tracing::error!(detail = %error.detail, "Error during streaming response");
            return guardrail_stream_error_response(error, start);
        }
        Err(error) => return guardrail_error_response(error, start),
    };
    let trace = endpoint.usage_tracking.then(|| GatewayTraceContext {
        state: state.clone(),
        workspace: workspace.clone(),
        endpoint,
        request: original_request,
        request_type: passthrough_request_type(action),
        started_ns: trace_started_ns,
    });
    let mut request = match adapter.passthrough_request(&model, action, payload.clone(), &headers) {
        Ok(request) => request,
        Err(error) => return error.response(start.elapsed()),
    };
    if let Err(error) = prepare_dynamic_auth(&model, &mut request).await {
        return error.response(start.elapsed());
    }
    if streaming {
        raw_stream_response(
            request,
            trace.map(|trace| (trace, model, passthrough_method(action))),
            start,
        )
    } else {
        raw_non_stream_response(
            request,
            trace.map(|trace| (trace, model, passthrough_method(action))),
            start,
            guardrails,
            payload,
            action != PassthroughAction::OpenAiEmbeddings,
        )
        .await
    }
}

pub async fn raw_proxy(
    State(state): State<AppState>,
    Workspace(workspace): Workspace,
    Path((endpoint_name, path)): Path<(String, String)>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let start = Instant::now();
    let payload = match parse_body(&body) {
        Ok(payload) => payload,
        Err(error) => return error.response(start.elapsed()),
    };
    let (endpoint, model, adapter) =
        match resolve_runtime_endpoint_provider(&state, &workspace, &endpoint_name).await {
            Ok(value) => value,
            Err(error) => return error.response(start.elapsed()),
        };
    if let Some(response) = check_budget_limit(&state, &workspace, &endpoint, start).await {
        return response;
    }
    let path = match uri.query() {
        Some(query) => format!("{path}?{query}"),
        None => path,
    };
    let guardrails = load_guardrails(&state, &workspace, &endpoint, &headers).await;
    let payload = match guardrails.before(payload, None).await {
        Ok(payload) => payload,
        Err(error) => return guardrail_error_response(error, start),
    };
    let mut request = match adapter.proxy_request(&model, &path, payload.clone(), &headers) {
        Ok(request) => request,
        Err(error) => return error.response(start.elapsed()),
    };
    if let Err(error) = prepare_dynamic_auth(&model, &mut request).await {
        return error.response(start.elapsed());
    }
    raw_proxy_response(request, start, guardrails, payload).await
}

// Raw-proxy routes address the endpoint's primary model directly; traffic
// split and fallback apply only to the unified invocation paths (Python
// behaves the same, validated by the T18.4 differential).
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

async fn complete_gateway_trace(
    trace: &GatewayTraceContext,
    model: &ResolvedGatewayModelConfig,
    method: &str,
    output: &Value,
    status: &str,
) {
    let usage = extract_token_usage(output);
    let cost = usage.and_then(|usage| token_cost_for_model(model, usage));

    // Python runs the budget callback inside the active trace, before the
    // asynchronous trace exporter flushes this invocation to the store. Keep
    // that ordering so a refresh cannot backfill this call and then add it a
    // second time through record_cost.
    if let Some(cost) = cost {
        let now = Utc::now();
        if let Err(error) = refresh_from_store(
            trace.state.budget_tracker(),
            trace.state.tracking_store(),
            &trace.workspace,
            now,
        )
        .await
        {
            tracing::debug!(error = %error, "Failed to refresh budget policies");
        }
        match trace
            .state
            .budget_tracker()
            .record_cost(cost.total_cost, Some(&trace.workspace), now)
            .await
        {
            Ok(windows) => {
                if let Some(dispatcher) = trace.state.webhook_dispatcher() {
                    for window in windows
                        .into_iter()
                        .filter(|window| window.policy.budget_action == "ALERT")
                    {
                        dispatcher
                            .fire(
                                WebhookEvent::new(
                                    WebhookEntity::BudgetPolicy,
                                    WebhookAction::Exceeded,
                                ),
                                exceeded_payload(&window, Some(&trace.workspace)),
                            )
                            .await;
                    }
                }
            }
            Err(error) => tracing::debug!(error = %error, "Failed to record budget cost"),
        }
    }

    if let Err(error) =
        persist_gateway_trace(trace, model, method, output, usage, cost, status).await
    {
        tracing::debug!(error = %error, "Failed to persist gateway trace");
    }
}

async fn persist_gateway_trace(
    trace: &GatewayTraceContext,
    model: &ResolvedGatewayModelConfig,
    method: &str,
    output: &Value,
    usage: Option<TokenUsage>,
    cost: Option<TokenCost>,
    status: &str,
) -> Result<(), String> {
    let Some(experiment_id) = trace.endpoint.experiment_id.as_deref() else {
        return Ok(());
    };
    let trace_uuid = uuid::Uuid::new_v4();
    let trace_id = format!("tr-{}", trace_uuid.simple());
    let root_bytes = first_eight(uuid::Uuid::new_v4());
    let child_bytes = first_eight(uuid::Uuid::new_v4());
    let root_id = hex_lower(&root_bytes);
    let child_id = hex_lower(&child_bytes);
    let ended_ns = unix_nanos().max(trace.started_ns);
    let duration_ms = (ended_ns - trace.started_ns) / 1_000_000;

    let request_json = python_json_dumps(&trace.request, false);
    let output_json = python_json_dumps(output, false);
    let mut metadata = vec![
        (
            "mlflow.gateway.endpointId".to_string(),
            trace.endpoint.endpoint_id.clone(),
        ),
        (
            "mlflow.gateway.requestType".to_string(),
            trace.request_type.to_string(),
        ),
        ("mlflow.traceInputs".to_string(), request_json.clone()),
        ("mlflow.traceOutputs".to_string(), output_json.clone()),
        ("mlflow.trace_schema.version".to_string(), "3".to_string()),
    ];
    if let Some(usage) = usage {
        metadata.push((
            "mlflow.trace.tokenUsage".to_string(),
            python_json_dumps(&usage_value(usage), false),
        ));
    }
    if let Some(cost) = cost {
        metadata.push((
            "mlflow.trace.cost".to_string(),
            python_json_dumps(&cost_value(cost), false),
        ));
    }
    let start = StartTraceInput {
        trace_id: trace_id.clone(),
        experiment_id: experiment_id.to_string(),
        request_time: trace.started_ns / 1_000_000,
        execution_duration: Some(duration_ms),
        state: status.to_string(),
        client_request_id: None,
        request_preview: request_preview(&trace.request),
        response_preview: response_preview(output),
        tags: vec![(
            "mlflow.traceName".to_string(),
            format!("gateway/{}", trace.endpoint.endpoint_name),
        )],
        trace_metadata: metadata,
        trace_metrics: Vec::new(),
        assessments: Vec::new(),
    };
    trace
        .state
        .tracking_store()
        .start_trace(&trace.workspace, &start)
        .await
        .map_err(|error| error.to_string())?;

    let root_content = span_content(
        trace_uuid.as_bytes(),
        &root_bytes,
        None,
        &format!("gateway/{}", trace.endpoint.endpoint_name),
        trace.started_ns,
        ended_ns,
        status,
        root_attributes(trace, &trace_id, &request_json, &output_json),
    );
    let child_attributes = child_attributes(model, &trace_id, method, usage, cost);
    let child_content = span_content(
        trace_uuid.as_bytes(),
        &child_bytes,
        Some(&root_bytes),
        &format!(
            "provider/{}/{}",
            normalize_provider(&model.provider),
            model.model_name
        ),
        trace.started_ns,
        ended_ns,
        status,
        child_attributes,
    );
    let spans = vec![
        SpanInput {
            trace_id: trace_id.clone(),
            span_id: root_id.clone(),
            parent_span_id: None,
            name: Some(format!("gateway/{}", trace.endpoint.endpoint_name)),
            span_type: Some("UNKNOWN".to_string()),
            status: status.to_string(),
            start_time_unix_nano: trace.started_ns,
            end_time_unix_nano: Some(ended_ns),
            content: root_content,
            dimension_attributes: None,
        },
        SpanInput {
            trace_id: trace_id.clone(),
            span_id: child_id.clone(),
            parent_span_id: Some(root_id),
            name: Some(format!(
                "provider/{}/{}",
                normalize_provider(&model.provider),
                model.model_name
            )),
            span_type: Some("LLM".to_string()),
            status: status.to_string(),
            start_time_unix_nano: trace.started_ns,
            end_time_unix_nano: Some(ended_ns),
            content: child_content,
            dimension_attributes: Some(
                json!({
                    "mlflow.llm.model": model.model_name,
                    "mlflow.llm.provider": normalize_provider(&model.provider),
                })
                .to_string(),
            ),
        },
    ];
    let metrics = cost
        .map(|cost| {
            [
                ("input_cost", cost.input_cost),
                ("output_cost", cost.output_cost),
                ("total_cost", cost.total_cost),
            ]
            .into_iter()
            .map(|(key, value)| SpanMetricInput {
                trace_id: trace_id.clone(),
                span_id: child_id.clone(),
                key: key.to_string(),
                value,
            })
            .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    trace
        .state
        .tracking_store()
        .log_spans(
            &trace.workspace,
            experiment_id,
            &spans,
            &metrics,
            &[TraceTimeRange {
                trace_id,
                min_start_ms: trace.started_ns / 1_000_000,
                max_end_ms: Some(ended_ns / 1_000_000),
                root_span_status: Some(status.to_string()),
            }],
        )
        .await
        .map_err(|error| error.to_string())
}

fn extract_token_usage(output: &Value) -> Option<TokenUsage> {
    let usage = output
        .get("usage")
        .or_else(|| output.get("usageMetadata"))?;
    let input = first_u64(
        usage,
        &["prompt_tokens", "input_tokens", "promptTokenCount"],
    );
    let output_tokens = first_u64(
        usage,
        &["completion_tokens", "output_tokens", "candidatesTokenCount"],
    );
    let total = first_u64(usage, &["total_tokens", "totalTokenCount"]).or_else(|| {
        input
            .zip(output_tokens)
            .map(|(input, output)| input + output)
    });
    match (input, output_tokens, total) {
        (Some(input_tokens), Some(output_tokens), Some(total_tokens)) => Some(TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens,
        }),
        _ => None,
    }
}

fn stream_usage_output(bytes: &[u8]) -> Value {
    let mut input_tokens = None;
    let mut output_tokens = None;
    let mut total_tokens = None;

    for line in String::from_utf8_lossy(bytes).lines() {
        let line = line.trim();
        let payload = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        let Some(usage) = value
            .get("usage")
            .or_else(|| value.get("usageMetadata"))
            .or_else(|| value.pointer("/message/usage"))
        else {
            continue;
        };
        input_tokens = first_u64(
            usage,
            &["prompt_tokens", "input_tokens", "promptTokenCount"],
        )
        .or(input_tokens);
        output_tokens = first_u64(
            usage,
            &["completion_tokens", "output_tokens", "candidatesTokenCount"],
        )
        .or(output_tokens);
        total_tokens = first_u64(usage, &["total_tokens", "totalTokenCount"]).or(total_tokens);
    }

    let total_tokens = total_tokens.or_else(|| {
        input_tokens
            .zip(output_tokens)
            .map(|(input, output)| input + output)
    });
    match (input_tokens, output_tokens, total_tokens) {
        (Some(input), Some(output), Some(total)) => json!({
            "usage": {
                "prompt_tokens": input,
                "completion_tokens": output,
                "total_tokens": total,
            }
        }),
        _ => Value::Null,
    }
}

fn first_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| value.get(*key)?.as_u64())
}

fn token_cost_for_model(
    model: &ResolvedGatewayModelConfig,
    usage: TokenUsage,
) -> Option<TokenCost> {
    let accounting = model_accounting(&model.provider, &model.model_name)?;
    let input_rate = accounting
        .prices
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let output_rate = accounting
        .prices
        .get("output_cost_per_token")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    if input_rate == 0.0 && output_rate == 0.0 {
        return None;
    }
    let input_cost = input_rate * usage.input_tokens as f64;
    let output_cost = output_rate * usage.output_tokens as f64;
    Some(TokenCost {
        input_cost,
        output_cost,
        total_cost: input_cost + output_cost,
    })
}

fn usage_value(usage: TokenUsage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    })
}

fn cost_value(cost: TokenCost) -> Value {
    json!({
        "input_cost": cost.input_cost,
        "output_cost": cost.output_cost,
        "total_cost": cost.total_cost,
    })
}

fn root_attributes(
    trace: &GatewayTraceContext,
    trace_id: &str,
    request_json: &str,
    output_json: &str,
) -> Map<String, Value> {
    string_attributes([
        (
            "mlflow.experimentId",
            quoted(trace.endpoint.experiment_id.as_deref().unwrap_or("")),
        ),
        ("mlflow.traceRequestId", quoted(trace_id)),
        ("mlflow.spanType", quoted("UNKNOWN")),
        ("endpoint_id", quoted(&trace.endpoint.endpoint_id)),
        ("endpoint_name", quoted(&trace.endpoint.endpoint_name)),
        ("mlflow.spanInputs", request_json.to_string()),
        ("mlflow.spanOutputs", output_json.to_string()),
    ])
}

fn child_attributes(
    model: &ResolvedGatewayModelConfig,
    trace_id: &str,
    method: &str,
    usage: Option<TokenUsage>,
    cost: Option<TokenCost>,
) -> Map<String, Value> {
    let mut attributes = string_attributes([
        ("mlflow.traceRequestId", quoted(trace_id)),
        ("mlflow.spanType", quoted("LLM")),
        ("mlflow.llm.model", quoted(&model.model_name)),
        (
            "mlflow.llm.provider",
            quoted(normalize_provider(&model.provider)),
        ),
    ]);
    let action = if method.contains('_') {
        "action"
    } else {
        "method"
    };
    attributes.insert(action.to_string(), Value::String(quoted(method)));
    if let Some(usage) = usage {
        attributes.insert(
            "mlflow.chat.tokenUsage".to_string(),
            Value::String(python_json_dumps(&usage_value(usage), false)),
        );
    }
    if let Some(cost) = cost {
        attributes.insert(
            "mlflow.llm.cost".to_string(),
            Value::String(python_json_dumps(&cost_value(cost), false)),
        );
    }
    attributes
}

fn string_attributes<const N: usize>(entries: [(&str, String); N]) -> Map<String, Value> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), Value::String(value)))
        .collect()
}

fn quoted(value: &str) -> String {
    serde_json::to_string(value).expect("string JSON serialization cannot fail")
}

#[allow(clippy::too_many_arguments)]
fn span_content(
    trace_id: &[u8; 16],
    span_id: &[u8; 8],
    parent_span_id: Option<&[u8; 8]>,
    name: &str,
    start_ns: i64,
    end_ns: i64,
    status: &str,
    attributes: Map<String, Value>,
) -> String {
    let code = if status == "OK" {
        "STATUS_CODE_OK"
    } else {
        "STATUS_CODE_ERROR"
    };
    json!({
        "trace_id": base64::engine::general_purpose::STANDARD.encode(trace_id),
        "span_id": base64::engine::general_purpose::STANDARD.encode(span_id),
        "parent_span_id": parent_span_id.map(|id| base64::engine::general_purpose::STANDARD.encode(id)),
        "name": name,
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": end_ns,
        "events": [],
        "status": {"code": code, "message": ""},
        "attributes": attributes,
        "links": [],
    })
    .to_string()
}

fn first_eight(id: uuid::Uuid) -> [u8; 8] {
    id.as_bytes()[..8].try_into().expect("UUID has 16 bytes")
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn request_preview(request: &Value) -> Option<String> {
    request
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            request
                .get("input")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn response_preview(output: &Value) -> Option<String> {
    output
        .pointer("/choices/0/message/content")
        .or_else(|| output.pointer("/choices/0/delta/content"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn unix_nanos() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    )
    .unwrap_or(i64::MAX)
}

/// Python calls `check_budget_limit` after endpoint resolution and before
/// guardrail loading/execution. T18.7 inserts guardrails after this seam.
async fn check_budget_limit(
    state: &AppState,
    workspace: &str,
    endpoint: &ResolvedGatewayEndpointConfig,
    start: Instant,
) -> Option<Response> {
    let now = chrono::Utc::now();
    if let Err(error) = refresh_from_store(
        state.budget_tracker(),
        state.tracking_store(),
        workspace,
        now,
    )
    .await
    {
        tracing::debug!(error = %error, "Failed to refresh budget policies");
    }
    match state
        .budget_tracker()
        .should_reject_request(Some(workspace), now)
        .await
    {
        Ok(Some(window)) => {
            let detail = reject_message(&window);
            if let Err(error) =
                persist_budget_error_trace(state, workspace, endpoint, &detail).await
            {
                tracing::debug!(error = %error, "Failed to persist budget rejection trace");
            }
            Some(
                GatewayRuntimeError::http(StatusCode::TOO_MANY_REQUESTS, Value::String(detail))
                    .response(start.elapsed()),
            )
        }
        Ok(None) => None,
        Err(error) => {
            Some(GatewayRuntimeError::internal(error.to_string()).response(start.elapsed()))
        }
    }
}

async fn persist_budget_error_trace(
    state: &AppState,
    workspace: &str,
    endpoint: &ResolvedGatewayEndpointConfig,
    detail: &str,
) -> Result<(), String> {
    if !endpoint.usage_tracking {
        return Ok(());
    }
    let Some(experiment_id) = endpoint.experiment_id.as_deref() else {
        return Ok(());
    };
    let trace_uuid = uuid::Uuid::new_v4();
    let trace_id = format!("tr-{}", trace_uuid.simple());
    let root_bytes = first_eight(uuid::Uuid::new_v4());
    let root_id = hex_lower(&root_bytes);
    let started_ns = unix_nanos();
    let ended_ns = unix_nanos().max(started_ns);
    let name = format!("gateway/{}", endpoint.endpoint_name);
    state
        .tracking_store()
        .start_trace(
            workspace,
            &StartTraceInput {
                trace_id: trace_id.clone(),
                experiment_id: experiment_id.to_string(),
                request_time: started_ns / 1_000_000,
                execution_duration: Some((ended_ns - started_ns) / 1_000_000),
                state: "ERROR".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: Some(detail.to_string()),
                tags: vec![("mlflow.traceName".to_string(), name.clone())],
                trace_metadata: vec![("mlflow.trace_schema.version".to_string(), "3".to_string())],
                trace_metrics: Vec::new(),
                assessments: Vec::new(),
            },
        )
        .await
        .map_err(|error| error.to_string())?;
    let content = span_content(
        trace_uuid.as_bytes(),
        &root_bytes,
        None,
        &name,
        started_ns,
        ended_ns,
        "ERROR",
        string_attributes([
            ("mlflow.experimentId", quoted(experiment_id)),
            ("mlflow.traceRequestId", quoted(&trace_id)),
            ("mlflow.spanType", quoted("LLM")),
            ("endpoint_id", quoted(&endpoint.endpoint_id)),
            ("endpoint_name", quoted(&endpoint.endpoint_name)),
        ]),
    );
    state
        .tracking_store()
        .log_spans(
            workspace,
            experiment_id,
            &[SpanInput {
                trace_id: trace_id.clone(),
                span_id: root_id,
                parent_span_id: None,
                name: Some(name),
                span_type: Some("LLM".to_string()),
                status: "ERROR".to_string(),
                start_time_unix_nano: started_ns,
                end_time_unix_nano: Some(ended_ns),
                content,
                dimension_attributes: None,
            }],
            &[],
            &[TraceTimeRange {
                trace_id,
                min_start_ms: started_ns / 1_000_000,
                max_end_ms: Some(ended_ns / 1_000_000),
                root_span_status: Some("ERROR".to_string()),
            }],
        )
        .await
        .map_err(|error| error.to_string())
}

async fn resolve_runtime_endpoint_provider(
    state: &AppState,
    workspace: &str,
    endpoint_name: &str,
) -> Result<
    (
        ResolvedGatewayEndpointConfig,
        ResolvedGatewayModelConfig,
        Box<dyn GatewayProviderAdapter>,
    ),
    GatewayRuntimeError,
> {
    let endpoint = state
        .tracking_store()
        .get_resolved_gateway_endpoint_config(workspace, endpoint_name)
        .await
        .map_err(|error| {
            GatewayRuntimeError::http(
                StatusCode::NOT_FOUND,
                json!({"error_code":"RESOURCE_DOES_NOT_EXIST","message":error.to_string()}),
            )
        })?;
    let model = primary_model(&endpoint)?.clone();
    check_provider_allowed(&model.provider)?;
    let adapter = adapter_for(&model.provider)?;
    Ok((endpoint, model, adapter))
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
    client_headers: HeaderMap,
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
    if kind == InvocationKind::Chat {
        payload = materialize_chat_payload(payload);
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
    if let Some(response) = check_budget_limit(&state, workspace, &endpoint, start).await {
        return response;
    }
    let trace = endpoint.usage_tracking.then(|| GatewayTraceContext {
        state: state.clone(),
        workspace: workspace.to_string(),
        endpoint: endpoint.clone(),
        request: normalized_typed_trace_payload(payload.clone()),
        request_type: match kind {
            InvocationKind::Chat => "unified/chat",
            InvocationKind::Embeddings => "unified/embeddings",
        },
        started_ns: unix_nanos(),
    });
    let plan = match build_routing_plan(&endpoint) {
        Ok(plan) => plan,
        Err(error) => return error.response(start.elapsed()),
    };
    let stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let guardrails = if kind == InvocationKind::Chat {
        load_guardrails(&state, workspace, &endpoint, &client_headers).await
    } else {
        LoadedGuardrails::empty()
    };
    if stream {
        stream_response(plan, kind, payload, guardrails, trace, start).await
    } else {
        let payload = match guardrails
            .before(payload, Some(GuardrailPayloadSchema::ChatRequest))
            .await
        {
            Ok(payload) => payload,
            Err(error) => return guardrail_error_response(error, start),
        };
        non_stream_response(plan, kind, payload, guardrails, trace, start).await
    }
}

async fn non_stream_response(
    plan: RoutingPlan,
    kind: InvocationKind,
    payload: Value,
    guardrails: LoadedGuardrails,
    trace: Option<GatewayTraceContext>,
    start: Instant,
) -> Response {
    let mut provider_elapsed = Duration::ZERO;
    let mut last_error = None;
    for attempt_index in 0..plan.attempt_limit {
        let model = match plan.model_for_attempt(attempt_index) {
            Ok(model) => model,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        let result = execute_non_stream_attempt(model.clone(), kind, payload.clone()).await;
        provider_elapsed += result.1;
        match result.0 {
            Ok(value) => {
                // AFTER guardrails transform the response first so the trace
                // records what the client actually receives.
                let value = match guardrails
                    .after(&payload, value, Some(GuardrailPayloadSchema::ChatResponse))
                    .await
                {
                    Ok(value) => value,
                    Err(error) => return guardrail_error_response(error, start),
                };
                if let Some(trace) = trace.as_ref() {
                    complete_gateway_trace(
                        trace,
                        &model,
                        match kind {
                            InvocationKind::Chat => "chat",
                            InvocationKind::Embeddings => "embeddings",
                        },
                        &value,
                        "OK",
                    )
                    .await;
                }
                return with_non_stream_timing(
                    json_response(StatusCode::OK, value),
                    start,
                    provider_elapsed,
                );
            }
            Err(error) => last_error = Some(error),
        }
    }

    let error = if let Some(attempts) = plan.fallback_attempt_label {
        fallback_error(attempts, last_error.as_ref())
    } else {
        last_error.unwrap_or_else(|| GatewayRuntimeError::internal("No provider was selected"))
    };
    let response = error.response(start.elapsed());
    with_non_stream_timing(response, start, provider_elapsed)
}

async fn execute_non_stream_attempt(
    model: ResolvedGatewayModelConfig,
    kind: InvocationKind,
    mut payload: Value,
) -> (Result<Value, GatewayRuntimeError>, Duration) {
    let adapter = match adapter_for(&model.provider) {
        Ok(adapter) => adapter,
        Err(error) => return (Err(error), Duration::ZERO),
    };
    if kind == InvocationKind::Chat {
        compact_chat_payload(&mut payload);
    }
    let request = match adapter.transform_request(&model, kind, payload, false) {
        Ok(request) => request,
        Err(error) => return (Err(error), Duration::ZERO),
    };
    let provider_start = Instant::now();
    let response = match send_provider_request(request).await {
        Ok(response) => response,
        Err(error) => {
            return (
                Err(GatewayRuntimeError::http(
                    StatusCode::BAD_GATEWAY,
                    Value::String(error.to_string()),
                )),
                provider_start.elapsed(),
            )
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
                return (
                    Err(GatewayRuntimeError::http(
                        StatusCode::BAD_GATEWAY,
                        Value::String(error.to_string()),
                    )),
                    provider_start.elapsed(),
                )
            }
        }
    } else if content_type.contains("text/plain") {
        json!({"message": response.text().await.unwrap_or_default()})
    } else {
        return (
            Err(GatewayRuntimeError::http(
                StatusCode::BAD_GATEWAY,
                Value::String(format!(
                    "The returned data type from the route service is not supported. Received content type: {}",
                    if content_type.is_empty() { "None" } else { &content_type }
                )),
            )),
            provider_start.elapsed(),
        );
    };
    let provider_elapsed = provider_start.elapsed();
    if !status.is_success() {
        return (Err(adapter.map_error(status, body)), provider_elapsed);
    }
    let transformed = match adapter.transform_response(&model, kind, body, unix_seconds()) {
        Ok(value) => value,
        Err(error) => return (Err(error), provider_elapsed),
    };
    (Ok(transformed), provider_elapsed)
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

fn guardrail_error_response(error: GuardrailExecutionError, start: Instant) -> Response {
    GatewayRuntimeError {
        status: error.status,
        detail: Value::String(error.detail),
        stream_type: error.stream_type,
    }
    .response(start.elapsed())
}

fn guardrail_stream_error_response(error: GuardrailExecutionError, start: Instant) -> Response {
    let output =
        stream::once(
            async move { Ok::<_, Infallible>(sse_error(&error.detail, error.stream_type)) },
        );
    let mut response = Response::new(Body::from_stream(output));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    insert_timing_header(&mut response, DURATION_HEADER, start.elapsed().as_millis());
    response
}

fn materialize_chat_payload(payload: Value) -> Value {
    let source = payload
        .as_object()
        .expect("validated chat payload is an object");
    let value_or_null = |field: &str| source.get(field).cloned().unwrap_or(Value::Null);
    let messages = source
        .get("messages")
        .and_then(Value::as_array)
        .expect("validated chat messages are an array")
        .iter()
        .map(|message| {
            let message = message
                .as_object()
                .expect("validated chat message is an object");
            json!({
                "role": message.get("role").cloned().unwrap_or(Value::Null),
                "content": message.get("content").cloned().unwrap_or(Value::Null),
                "tool_calls": message.get("tool_calls").cloned().unwrap_or(Value::Null),
                "refusal": message.get("refusal").cloned().unwrap_or(Value::Null),
                "tool_call_id": message.get("tool_call_id").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "n": source.get("n").cloned().unwrap_or(json!(1)),
        "stop": value_or_null("stop"),
        "max_tokens": value_or_null("max_tokens"),
        "max_completion_tokens": value_or_null("max_completion_tokens"),
        "stream": value_or_null("stream"),
        "stream_options": value_or_null("stream_options"),
        "model": value_or_null("model"),
        "response_format": value_or_null("response_format"),
        "temperature": value_or_null("temperature"),
        "top_p": value_or_null("top_p"),
        "presence_penalty": value_or_null("presence_penalty"),
        "frequency_penalty": value_or_null("frequency_penalty"),
        "top_k": value_or_null("top_k"),
        "messages": messages,
        "tools": value_or_null("tools"),
        "tool_choice": value_or_null("tool_choice"),
    })
}

fn compact_chat_payload(payload: &mut Value) {
    let Some(object) = payload.as_object_mut() else {
        return;
    };
    object.retain(|_, value| !value.is_null());
    if let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if let Some(message) = message.as_object_mut() {
                message.retain(|_, value| !value.is_null());
            }
        }
    }
}

fn normalized_typed_trace_payload(mut payload: Value) -> Value {
    omit_none_fields(&mut payload);
    payload
}

fn omit_none_fields(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.retain(|_, value| !value.is_null());
            for value in object.values_mut() {
                omit_none_fields(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                omit_none_fields(value);
            }
        }
        _ => {}
    }
}

async fn send_provider_request(
    request: ProviderRequest,
) -> Result<reqwest::Response, reqwest::Error> {
    let body = serde_json::to_vec(&request.body).expect("JSON value serialization cannot fail");
    client()
        .post(request.url)
        .headers(request.headers)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
}

async fn raw_non_stream_response(
    request: ProviderRequest,
    trace: Option<(
        GatewayTraceContext,
        ResolvedGatewayModelConfig,
        &'static str,
    )>,
    start: Instant,
    guardrails: LoadedGuardrails,
    request_payload: Value,
    run_after: bool,
) -> Response {
    let provider_start = Instant::now();
    let response = match send_provider_request(request).await {
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
    let provider_elapsed = provider_start.elapsed();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let value = if content_type.contains("application/json") {
        response.json::<Value>().await.unwrap_or(Value::Null)
    } else if content_type.contains("text/plain") {
        json!({"message":response.text().await.unwrap_or_default()})
    } else {
        let response = GatewayRuntimeError::http(
            StatusCode::BAD_GATEWAY,
            Value::String(format!(
                "The returned data type from the route service is not supported. Received content type: {}",
                if content_type.is_empty() { "None" } else { &content_type }
            )),
        )
        .response(start.elapsed());
        return with_non_stream_timing(response, start, provider_elapsed);
    };
    let response = if status.is_success() {
        // AFTER guardrails transform the response first so the trace records
        // what the client actually receives.
        let value = if run_after {
            match guardrails.after(&request_payload, value, None).await {
                Ok(value) => value,
                Err(error) => return guardrail_error_response(error, start),
            }
        } else {
            value
        };
        if let Some((trace, model, method)) = trace.as_ref() {
            complete_gateway_trace(trace, model, method, &value, "OK").await;
        }
        json_response(status, value)
    } else {
        let detail = value.pointer("/error/message").cloned().unwrap_or(value);
        GatewayRuntimeError::http(status, detail).response(start.elapsed())
    };
    with_non_stream_timing(response, start, provider_elapsed)
}

struct RawProviderStream {
    initial: Option<BoxFuture<'static, Result<reqwest::Response, reqwest::Error>>>,
    upstream: Option<futures::stream::BoxStream<'static, Result<Bytes, reqwest::Error>>>,
    done: bool,
    trace: Option<(
        GatewayTraceContext,
        ResolvedGatewayModelConfig,
        &'static str,
    )>,
    trace_bytes: Vec<u8>,
}

fn raw_stream_response(
    request: ProviderRequest,
    trace: Option<(
        GatewayTraceContext,
        ResolvedGatewayModelConfig,
        &'static str,
    )>,
    start: Instant,
) -> Response {
    let state = RawProviderStream {
        initial: Some(send_provider_request(request).boxed()),
        upstream: None,
        done: false,
        trace,
        trace_bytes: Vec::new(),
    };
    let output = stream::unfold(state, next_raw_stream_chunk).map(Ok::<_, Infallible>);
    let mut response = Response::new(Body::from_stream(output));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    insert_timing_header(&mut response, DURATION_HEADER, start.elapsed().as_millis());
    response
}

async fn next_raw_stream_chunk(mut state: RawProviderStream) -> Option<(Bytes, RawProviderStream)> {
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
                let value = response.json::<Value>().await.unwrap_or(Value::Null);
                let message = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| {
                        status
                            .canonical_reason()
                            .unwrap_or("HTTP error")
                            .to_string()
                    });
                state.done = true;
                return Some((
                    sse_error(&format!("{}: {message}", status.as_u16()), "HTTPException"),
                    state,
                ));
            }
            Err(error) => {
                state.done = true;
                return Some((sse_error(&error.to_string(), "ClientError"), state));
            }
        }
    }
    match state.upstream.as_mut()?.next().await {
        Some(Ok(bytes)) => {
            state.trace_bytes.extend_from_slice(&bytes);
            Some((bytes, state))
        }
        Some(Err(error)) => {
            state.done = true;
            Some((sse_error(&error.to_string(), "ClientPayloadError"), state))
        }
        None => {
            if let Some((trace, model, method)) = state.trace.as_ref() {
                let output = stream_usage_output(&state.trace_bytes);
                complete_gateway_trace(trace, model, method, &output, "OK").await;
            }
            None
        }
    }
}

async fn raw_proxy_response(
    request: ProviderRequest,
    start: Instant,
    guardrails: LoadedGuardrails,
    request_payload: Value,
) -> Response {
    let provider_start = Instant::now();
    let response = match send_provider_request(request).await {
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
        .unwrap_or("application/json")
        .to_string();
    if !status.is_success() {
        let value = response.json::<Value>().await.unwrap_or(Value::Null);
        let detail = value
            .pointer("/error/message")
            .cloned()
            .unwrap_or_else(|| Value::String(value.to_string()));
        return GatewayRuntimeError::http(status, detail).response(start.elapsed());
    }
    let media_type = content_type.split(';').next().unwrap_or_default().trim();
    if matches!(media_type, "text/event-stream" | "application/x-ndjson") {
        let output = response.bytes_stream().map(|result| {
            Ok::<_, Infallible>(
                result.unwrap_or_else(|error| sse_error(&error.to_string(), "ClientPayloadError")),
            )
        });
        let mut result = Response::new(Body::from_stream(output));
        result.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        insert_timing_header(&mut result, DURATION_HEADER, start.elapsed().as_millis());
        return result;
    }
    if content_type.contains("application/json") {
        return match response.json::<Value>().await {
            Ok(value) => match guardrails.after(&request_payload, value, None).await {
                Ok(value) => with_non_stream_timing(
                    json_response(StatusCode::OK, value),
                    start,
                    provider_start.elapsed(),
                ),
                Err(error) => guardrail_error_response(error, start),
            },
            Err(error) => {
                GatewayRuntimeError::http(StatusCode::BAD_GATEWAY, Value::String(error.to_string()))
                    .response(start.elapsed())
            }
        };
    }
    if content_type.contains("text/plain") {
        let value = json!({"message":response.text().await.unwrap_or_default()});
        let value = match guardrails.after(&request_payload, value, None).await {
            Ok(value) => value,
            Err(error) => return guardrail_error_response(error, start),
        };
        return with_non_stream_timing(
            json_response(StatusCode::OK, value),
            start,
            provider_start.elapsed(),
        );
    }
    GatewayRuntimeError::http(
        StatusCode::BAD_GATEWAY,
        Value::String(format!(
            "Unsupported Content-Type from upstream proxy: {content_type}"
        )),
    )
    .response(start.elapsed())
}

struct ProviderStream {
    plan: RoutingPlan,
    kind: InvocationKind,
    payload: Value,
    guardrails: LoadedGuardrails,
    pre_guardrails_complete: bool,
    trace: Option<GatewayTraceContext>,
    trace_frames: Vec<Value>,
    next_attempt: usize,
    active: Option<ActiveProviderStream>,
    last_error: Option<GatewayRuntimeError>,
    pending: Vec<Bytes>,
    done: bool,
}

struct ActiveProviderStream {
    initial: Option<BoxFuture<'static, Result<reqwest::Response, reqwest::Error>>>,
    upstream: Option<futures::stream::BoxStream<'static, Result<Bytes, reqwest::Error>>>,
    adapter: Box<dyn GatewayProviderAdapter>,
    model: ResolvedGatewayModelConfig,
    transform_state: StreamTransformState,
    buffer: Vec<u8>,
}

async fn stream_response(
    plan: RoutingPlan,
    kind: InvocationKind,
    payload: Value,
    guardrails: LoadedGuardrails,
    trace: Option<GatewayTraceContext>,
    start: Instant,
) -> Response {
    let state = ProviderStream {
        plan,
        kind,
        payload,
        guardrails,
        pre_guardrails_complete: false,
        trace,
        trace_frames: Vec::new(),
        next_attempt: 0,
        active: None,
        last_error: None,
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
        if !state.pre_guardrails_complete {
            match state
                .guardrails
                .before(
                    state.payload.clone(),
                    Some(GuardrailPayloadSchema::ChatRequest),
                )
                .await
            {
                Ok(payload) => {
                    state.payload = payload;
                    state.pre_guardrails_complete = true;
                }
                Err(error) => {
                    tracing::error!(detail = %error.detail, "Error during streaming response");
                    state
                        .pending
                        .push(sse_error(&error.detail, error.stream_type));
                    state.done = true;
                }
            }
            continue;
        }
        if state.active.is_none() {
            start_stream_attempt(&mut state);
            continue;
        }
        let initial = state
            .active
            .as_mut()
            .and_then(|active| active.initial.take());
        if let Some(initial) = initial {
            match initial.await {
                Ok(response) if response.status().is_success() => {
                    state.active.as_mut().expect("active attempt").upstream =
                        Some(response.bytes_stream().boxed());
                }
                Ok(response) => {
                    let status = response.status();
                    let body = response.json::<Value>().await.unwrap_or(Value::Null);
                    let error = state
                        .active
                        .as_ref()
                        .expect("active attempt")
                        .adapter
                        .map_error(status, body);
                    fail_stream_attempt(&mut state, error);
                }
                Err(error) => {
                    let error = GatewayRuntimeError::http(
                        StatusCode::BAD_GATEWAY,
                        Value::String(error.to_string()),
                    );
                    fail_stream_attempt(&mut state, error);
                }
            }
            continue;
        }
        let next = state
            .active
            .as_mut()
            .and_then(|active| active.upstream.as_mut())
            .expect("connected active stream")
            .next()
            .await;
        match next {
            Some(Ok(chunk)) => {
                let active = state.active.as_mut().expect("active attempt");
                active.buffer.extend_from_slice(&chunk);
                if let Err(error) = process_complete_lines(&mut state) {
                    fail_stream_attempt(&mut state, error);
                }
            }
            Some(Err(error)) => {
                fail_stream_attempt(
                    &mut state,
                    GatewayRuntimeError {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        detail: Value::String(error.to_string()),
                        stream_type: "ClientPayloadError",
                    },
                );
            }
            None => {
                let line = state.active.as_mut().and_then(|active| {
                    (!active.buffer.is_empty()).then(|| std::mem::take(&mut active.buffer))
                });
                if let Some(line) = line {
                    if let Err(error) = process_provider_line(&mut state, &line) {
                        fail_stream_attempt(&mut state, error);
                        continue;
                    }
                }
                if let (Some(trace), Some(active)) = (state.trace.as_ref(), state.active.as_ref()) {
                    let output = state.trace_frames.last().cloned().unwrap_or(Value::Null);
                    complete_gateway_trace(
                        trace,
                        &active.model,
                        match state.kind {
                            InvocationKind::Chat => "chat_stream",
                            InvocationKind::Embeddings => "embeddings",
                        },
                        &output,
                        "OK",
                    )
                    .await;
                }
                // An empty stream or a clean EOF is success and ends the
                // fallback chain, exactly like Python's async-for completion.
                state.done = true;
            }
        }
    }
}

fn start_stream_attempt(state: &mut ProviderStream) {
    if state.next_attempt >= state.plan.attempt_limit {
        finish_failed_stream(state);
        return;
    }
    let attempt_index = state.next_attempt;
    state.next_attempt += 1;
    let model = match state.plan.model_for_attempt(attempt_index) {
        Ok(model) => model,
        Err(error) => {
            fail_stream_attempt(state, error);
            return;
        }
    };
    let adapter = match adapter_for(&model.provider) {
        Ok(adapter) => adapter,
        Err(error) => {
            fail_stream_attempt(state, error);
            return;
        }
    };
    let mut payload = state.payload.clone();
    if state.kind == InvocationKind::Chat {
        compact_chat_payload(&mut payload);
    }
    let request = match adapter.transform_request(&model, state.kind, payload, true) {
        Ok(request) => request,
        Err(error) => {
            fail_stream_attempt(state, error);
            return;
        }
    };
    let initial = client()
        .post(request.url)
        .headers(request.headers)
        .json(&request.body)
        .send()
        .boxed();
    state.active = Some(ActiveProviderStream {
        initial: Some(initial),
        upstream: None,
        adapter,
        model,
        transform_state: StreamTransformState::default(),
        buffer: Vec::new(),
    });
}

fn fail_stream_attempt(state: &mut ProviderStream, error: GatewayRuntimeError) {
    state.last_error = Some(error);
    state.active = None;
    if state.next_attempt >= state.plan.attempt_limit {
        finish_failed_stream(state);
    }
}

fn finish_failed_stream(state: &mut ProviderStream) {
    let error = if let Some(attempts) = state.plan.fallback_attempt_label {
        fallback_error(attempts, state.last_error.as_ref())
    } else {
        state
            .last_error
            .take()
            .unwrap_or_else(|| GatewayRuntimeError::internal("No provider was selected"))
    };
    state
        .pending
        .push(sse_error(&error.stream_message(), error.stream_type));
    state.done = true;
}

fn process_complete_lines(state: &mut ProviderStream) -> Result<(), GatewayRuntimeError> {
    loop {
        let index = state
            .active
            .as_ref()
            .and_then(|active| active.buffer.iter().position(|byte| *byte == b'\n'));
        let Some(index) = index else {
            return Ok(());
        };
        let mut line: Vec<u8> = state
            .active
            .as_mut()
            .expect("active attempt")
            .buffer
            .drain(..=index)
            .collect();
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        process_provider_line(state, &line)?;
    }
}

fn process_provider_line(
    state: &mut ProviderStream,
    line: &[u8],
) -> Result<(), GatewayRuntimeError> {
    let Ok(text) = std::str::from_utf8(line) else {
        return Ok(());
    };
    let text = text.trim();
    if text.is_empty() || text.starts_with(':') || text.starts_with("event:") {
        return Ok(());
    }
    let Some(data) = text.strip_prefix("data:") else {
        return Ok(());
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }
    let provider_name = state
        .active
        .as_ref()
        .expect("active attempt")
        .adapter
        .provider_name();
    let value = match serde_json::from_str::<Value>(data) {
        Ok(value) => value,
        Err(error) => {
            // OpenAI's `stream_sse_data` deliberately ignores malformed JSON
            // data lines. Anthropic/Gemini call `json.loads` directly, so the
            // same malformed line becomes an in-band safe_stream error.
            if provider_name == "openai" {
                return Ok(());
            }
            let message = if data == "not-json" {
                if provider_name == "anthropic" {
                    "Expecting value: line 1 column 2 (char 1)".to_string()
                } else {
                    "Expecting value: line 1 column 1 (char 0)".to_string()
                }
            } else {
                error.to_string()
            };
            return Err(GatewayRuntimeError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                detail: Value::String(message),
                stream_type: "JSONDecodeError",
            });
        }
    };
    let active = state.active.as_mut().expect("active attempt");
    match active.adapter.transform_stream_frame(
        &active.model,
        value,
        &mut active.transform_state,
        unix_seconds(),
    ) {
        Ok(frames) => {
            for frame in frames {
                state.trace_frames.push(frame.clone());
                state.pending.push(sse_json(&frame));
            }
            Ok(())
        }
        Err(error) => Err(error),
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

fn build_routing_plan(
    endpoint: &ResolvedGatewayEndpointConfig,
) -> Result<RoutingPlan, GatewayRuntimeError> {
    let primary_models = endpoint
        .models
        .iter()
        .filter(|model| model.linkage_type == "PRIMARY")
        .cloned()
        .collect::<Vec<_>>();
    if primary_models.is_empty() {
        return Err(GatewayRuntimeError::http(
            StatusCode::NOT_FOUND,
            json!({
                "error_code":"RESOURCE_DOES_NOT_EXIST",
                "message":format!("Endpoint '{}' has no PRIMARY models configured", endpoint.endpoint_name)
            }),
        ));
    }

    let primary = if endpoint.routing_strategy.as_deref() == Some("REQUEST_BASED_TRAFFIC_SPLIT") {
        validate_models(&primary_models)?;
        PrimaryRoute::TrafficSplit(primary_models)
    } else {
        let model = primary_models
            .into_iter()
            .next()
            .expect("checked non-empty");
        validate_models(std::slice::from_ref(&model))?;
        PrimaryRoute::Single(Box::new(model))
    };

    let mut fallbacks = endpoint
        .models
        .iter()
        .filter(|model| model.linkage_type == "FALLBACK")
        .cloned()
        .collect::<Vec<_>>();
    let Some(fallback_config) = endpoint
        .fallback_config
        .as_ref()
        .filter(|_| !fallbacks.is_empty())
    else {
        return Ok(RoutingPlan {
            primary,
            fallbacks: Vec::new(),
            fallback_attempt_label: None,
            attempt_limit: 1,
        });
    };

    // Python's stable sort leaves equal and missing fallback_order entries in
    // endpoint-model order, with missing values after every explicit order.
    fallbacks.sort_by_key(|model| {
        (
            model.fallback_order.is_none(),
            model.fallback_order.unwrap_or_default(),
        )
    });
    validate_models(&fallbacks)?;

    // `max_attempts` counts fallback destinations in GatewayEndpoint, while
    // FallbackProvider counts the primary too. Python also treats zero like
    // None (`configured or len(fallback_models)`) and caps at provider count.
    let configured_fallbacks = match fallback_config.max_attempts {
        Some(value) if value != 0 => i64::from(value),
        _ => i64::try_from(fallbacks.len()).expect("model count fits i64"),
    };
    let provider_count = 1_i64 + i64::try_from(fallbacks.len()).expect("model count fits i64");
    let attempt_label = (configured_fallbacks + 1).min(provider_count);
    let python_slice_len = if attempt_label >= 0 {
        usize::try_from(attempt_label)
            .unwrap_or(usize::MAX)
            .min(provider_count as usize)
    } else {
        (provider_count - (-attempt_label).min(provider_count)) as usize
    };
    // For negative values Python's `attempt < self._max_attempts` is false on
    // the first failure even when negative slicing selected several providers.
    let attempt_limit = if attempt_label > 0 {
        python_slice_len
    } else {
        python_slice_len.min(1)
    };
    Ok(RoutingPlan {
        primary,
        fallbacks,
        fallback_attempt_label: Some(attempt_label),
        attempt_limit,
    })
}

impl RoutingPlan {
    fn model_for_attempt(
        &self,
        attempt_index: usize,
    ) -> Result<ResolvedGatewayModelConfig, GatewayRuntimeError> {
        if attempt_index == 0 {
            return match &self.primary {
                PrimaryRoute::Single(model) => Ok((**model).clone()),
                PrimaryRoute::TrafficSplit(models) => {
                    let index = weighted_model_index(models)?;
                    Ok(models[index].clone())
                }
            };
        }
        self.fallbacks
            .get(attempt_index - 1)
            .cloned()
            .ok_or_else(|| GatewayRuntimeError::internal("Fallback attempt is out of range"))
    }
}

fn validate_models(models: &[ResolvedGatewayModelConfig]) -> Result<(), GatewayRuntimeError> {
    for model in models {
        check_provider_allowed(&model.provider)?;
        adapter_for(&model.provider)?;
    }
    Ok(())
}

fn weighted_model_index(
    models: &[ResolvedGatewayModelConfig],
) -> Result<usize, GatewayRuntimeError> {
    let weights = models
        .iter()
        .map(|model| python_integer_weight(model.weight))
        .collect::<Result<Vec<_>, _>>()?;
    // Python draws from NumPy's process-global MT19937 stream. Rust uses its
    // per-thread RNG: probabilities and independent-choice distribution are
    // identical, but the seeded cross-language request order is deliberately
    // not byte-for-byte reproducible (the same order-only precedent as T17.3).
    let mut rng = rand::thread_rng();
    weighted_index_for_draw(&weights, rng.gen::<f64>())
}

fn python_integer_weight(weight: f64) -> Result<f32, GatewayRuntimeError> {
    let scaled = weight * 100.0;
    if scaled.is_nan() {
        return Err(python_value_error("cannot convert float NaN to integer"));
    }
    if !scaled.is_finite() {
        return Err(python_value_error(
            "cannot convert float infinity to integer",
        ));
    }
    Ok((scaled.trunc() as i64) as f32)
}

fn weighted_index_for_draw(weights: &[f32], draw: f64) -> Result<usize, GatewayRuntimeError> {
    let sum = weights.iter().copied().sum::<f32>();
    let probabilities = weights
        .iter()
        .map(|weight| *weight / sum)
        .collect::<Vec<_>>();
    if probabilities.iter().any(|probability| probability.is_nan()) {
        return Err(python_value_error("probabilities contain NaN"));
    }
    if probabilities.iter().any(|probability| *probability < 0.0) {
        return Err(python_value_error("probabilities are not non-negative"));
    }
    let mut cumulative = 0.0_f64;
    let mut last_nonzero = None;
    for (index, probability) in probabilities.into_iter().enumerate() {
        if probability > 0.0 {
            last_nonzero = Some(index);
        }
        cumulative += f64::from(probability);
        if draw < cumulative {
            return Ok(index);
        }
    }
    last_nonzero.ok_or_else(|| python_value_error("probabilities contain NaN"))
}

fn python_value_error(message: &str) -> GatewayRuntimeError {
    GatewayRuntimeError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        detail: Value::String(message.to_string()),
        stream_type: "ValueError",
    }
}

fn fallback_error(attempts: i64, last_error: Option<&GatewayRuntimeError>) -> GatewayRuntimeError {
    let status = last_error
        .filter(|error| error.propagates_fallback_status())
        .map(|error| error.status)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let last_message = last_error
        .map(GatewayRuntimeError::fallback_message)
        .unwrap_or_else(|| "None".to_string());
    GatewayRuntimeError::new(
        status,
        format!("All {attempts} fallback attempts failed. Last error: {last_message}"),
    )
}

fn adapter_for(provider: &str) -> Result<Box<dyn GatewayProviderAdapter>, GatewayRuntimeError> {
    let normalized = normalize_provider(provider);
    match normalized {
        "openai" | "azure" | "azure-openai" => Ok(Box::new(OpenAiAdapter)),
        "anthropic" => Ok(Box::new(AnthropicAdapter)),
        "gemini" => Ok(Box::new(GeminiAdapter)),
        "bedrock" => Ok(Box::new(BedrockAdapter)),
        "databricks" => Ok(Box::new(DatabricksAdapter)),
        provider if is_supported_provider(provider) => Ok(Box::new(OpenAiCompatibleAdapter {
            provider: provider.to_string(),
        })),
        provider => Err(GatewayRuntimeError::new(
            StatusCode::BAD_REQUEST,
            format!("Provider '{provider}' is not present in the pinned native provider manifest."),
        )),
    }
}

fn check_provider_allowed(provider: &str) -> Result<(), GatewayRuntimeError> {
    let Ok(allowed) = std::env::var(ALLOWED_PROVIDERS_ENV) else {
        return Ok(());
    };
    let normalized = normalize_provider(provider);
    if allowed
        .split(',')
        .map(str::trim)
        .map(|value| value.to_ascii_lowercase())
        .any(|value| normalize_provider(&value) == normalized)
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

fn supports_passthrough(provider: &str, action: PassthroughAction) -> bool {
    match normalize_provider(provider) {
        "openai" | "azure" => matches!(
            action,
            PassthroughAction::OpenAiChat
                | PassthroughAction::OpenAiEmbeddings
                | PassthroughAction::OpenAiResponses
                | PassthroughAction::OpenAiResponsesCompact
        ),
        "anthropic" => action == PassthroughAction::AnthropicMessages,
        "gemini" => matches!(
            action,
            PassthroughAction::GeminiGenerateContent
                | PassthroughAction::GeminiStreamGenerateContent
        ),
        "databricks" => action != PassthroughAction::OpenAiResponsesCompact,
        "groq" | "deepseek" | "xai" | "openrouter" | "ollama" | "portkey" => matches!(
            action,
            PassthroughAction::OpenAiChat | PassthroughAction::OpenAiEmbeddings
        ),
        "bedrock" => false,
        provider => is_supported_provider(provider),
    }
}

fn unsupported_passthrough(provider: &str, action: PassthroughAction) -> GatewayRuntimeError {
    GatewayRuntimeError::new(
        StatusCode::BAD_REQUEST,
        format!(
            "Unsupported passthrough endpoint '{}' for {provider} provider.",
            passthrough_route(action)
        ),
    )
}

fn passthrough_route(action: PassthroughAction) -> &'static str {
    match action {
        PassthroughAction::OpenAiChat => "/openai/v1/chat/completions",
        PassthroughAction::OpenAiEmbeddings => "/openai/v1/embeddings",
        PassthroughAction::OpenAiResponses => "/openai/v1/responses",
        PassthroughAction::OpenAiResponsesCompact => "/openai/v1/responses/compact",
        PassthroughAction::AnthropicMessages => "/anthropic/v1/messages",
        PassthroughAction::GeminiGenerateContent => {
            "/gemini/v1beta/models/{endpoint_name}:generateContent"
        }
        PassthroughAction::GeminiStreamGenerateContent => {
            "/gemini/v1beta/models/{endpoint_name}:streamGenerateContent"
        }
    }
}

fn passthrough_request_type(action: PassthroughAction) -> &'static str {
    match action {
        PassthroughAction::OpenAiChat => "passthrough/model/openai-chat",
        PassthroughAction::OpenAiEmbeddings => "passthrough/model/openai-embeddings",
        PassthroughAction::OpenAiResponses | PassthroughAction::OpenAiResponsesCompact => {
            "passthrough/model/openai-responses"
        }
        PassthroughAction::AnthropicMessages => "passthrough/model/anthropic-messages",
        PassthroughAction::GeminiGenerateContent
        | PassthroughAction::GeminiStreamGenerateContent => {
            "passthrough/model/gemini-generateContent"
        }
    }
}

fn passthrough_method(action: PassthroughAction) -> &'static str {
    match action {
        PassthroughAction::OpenAiChat => "openai_chat",
        PassthroughAction::OpenAiEmbeddings => "openai_embeddings",
        PassthroughAction::OpenAiResponses => "openai_responses",
        PassthroughAction::OpenAiResponsesCompact => "openai_responses_compact",
        PassthroughAction::AnthropicMessages => "anthropic_messages",
        PassthroughAction::GeminiGenerateContent => "gemini_generate_content",
        PassthroughAction::GeminiStreamGenerateContent => "gemini_stream_generate_content",
    }
}

fn provider_api_base(model: &ResolvedGatewayModelConfig) -> Result<Url, GatewayRuntimeError> {
    let provider = normalize_provider(&model.provider);
    let configured = model.auth_config.get("api_base").cloned();
    if provider == "azure" {
        let base = configured.ok_or_else(|| missing_auth("api_base"))?;
        let version = required_auth(model, "api_version")?;
        let deployment = model
            .auth_config
            .get("deployment_name")
            .unwrap_or(&model.model_name);
        return parse_url(&format!(
            "{}/openai/deployments/{deployment}?api-version={version}",
            base.trim_end_matches('/')
        ));
    }
    if provider == "databricks" {
        let base = configured
            .or_else(|| std::env::var("DATABRICKS_HOST").ok())
            .ok_or_else(|| missing_auth("api_base"))?;
        let base = base.trim_end_matches('/');
        let normalized = if base.contains("/serving-endpoints") {
            base.to_string()
        } else {
            format!("{base}/serving-endpoints")
        };
        return parse_url(&normalized);
    }
    let base = configured
        .or_else(|| default_api_base(provider).map(str::to_string))
        .ok_or_else(|| {
            GatewayRuntimeError::new(
                StatusCode::BAD_REQUEST,
                format!(
                    "Provider '{provider}' requires 'api_base' in the pinned native runtime configuration."
                ),
            )
        })?;
    parse_url(&base)
}

fn append_provider_path(base: &Url, path: &str) -> Result<Url, GatewayRuntimeError> {
    let (path, query) = path.split_once('?').unwrap_or((path, ""));
    let mut url = base.clone();
    let joined = format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    url.set_path(&joined);
    if !query.is_empty() {
        url.set_query(Some(query));
    }
    Ok(url)
}

fn proxy_root(base: &Url) -> Result<Url, GatewayRuntimeError> {
    let mut url = base.clone();
    let path = url.path().trim_end_matches('/');
    let root = path
        .rsplit_once('/')
        .map(|(head, _)| head)
        .unwrap_or("")
        .to_string();
    url.set_path(&root);
    url.set_query(None);
    Ok(url)
}

fn merged_passthrough_headers(
    _model: &ResolvedGatewayModelConfig,
    client: &HeaderMap,
) -> Result<HeaderMap, GatewayRuntimeError> {
    let preserve_auth = client_provides_auth(client);
    let mut headers = HeaderMap::new();
    for (name, value) in client {
        let lower = name.as_str().to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "host" | "content-length" | "accept-encoding" | "x-mlflow-authorization"
        ) || (!preserve_auth && is_auth_header(&lower))
        {
            continue;
        }
        headers.insert(name.clone(), value.clone());
    }
    Ok(headers)
}

fn client_provides_auth(headers: &HeaderMap) -> bool {
    let agent = headers
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    ["claude-cli", "codex", "geminicli"]
        .iter()
        .any(|prefix| agent.contains(prefix))
        && headers.keys().any(|name| is_auth_header(name.as_str()))
}

fn is_auth_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization" | "x-api-key" | "x-goog-api-key" | "api-key"
    )
}

fn has_provider_auth(headers: &HeaderMap) -> bool {
    headers.keys().any(|name| is_auth_header(name.as_str()))
}

async fn prepare_dynamic_auth(
    model: &ResolvedGatewayModelConfig,
    request: &mut ProviderRequest,
) -> Result<(), GatewayRuntimeError> {
    if normalize_provider(&model.provider) == "bedrock"
        && model.auth_config.get("auth_mode").map(String::as_str) == Some("iam_role")
    {
        let credentials = assume_bedrock_role(model).await?;
        return sign_bedrock_with_credentials(model, request, &credentials);
    }
    if normalize_provider(&model.provider) != "databricks"
        || model.auth_config.get("auth_mode").map(String::as_str) != Some("oauth_m2m")
        || has_provider_auth(&request.headers)
    {
        return Ok(());
    }
    let client_id = required_auth(model, "client_id")?;
    let client_secret = secret_string(model, "client_secret")?;
    let mut token_url = provider_api_base(model)?;
    token_url.set_path("/oidc/v1/token");
    token_url.set_query(None);
    let response = client()
        .post(token_url)
        .basic_auth(client_id, Some(client_secret))
        .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body("grant_type=client_credentials&scope=all-apis")
        .send()
        .await
        .map_err(|error| {
            GatewayRuntimeError::http(StatusCode::BAD_GATEWAY, Value::String(error.to_string()))
        })?;
    let status = response.status();
    let value = response.json::<Value>().await.map_err(|error| {
        GatewayRuntimeError::http(StatusCode::BAD_GATEWAY, Value::String(error.to_string()))
    })?;
    if !status.is_success() {
        return Err(GatewayRuntimeError::http(status, value));
    }
    let token = value
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            GatewayRuntimeError::internal("Databricks OAuth response omitted access_token")
        })?;
    insert_header(
        &mut request.headers,
        "authorization",
        &format!("Bearer {token}"),
    )
}

#[derive(Debug)]
struct AwsCredentials {
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
}

async fn assume_bedrock_role(
    model: &ResolvedGatewayModelConfig,
) -> Result<AwsCredentials, GatewayRuntimeError> {
    let base = AwsCredentials {
        access_key: secret_string_optional(model, "aws_access_key_id")
            .map(str::to_string)
            .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
            .ok_or_else(|| missing_auth("aws_access_key_id"))?,
        secret_key: secret_string_optional(model, "aws_secret_access_key")
            .map(str::to_string)
            .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
            .ok_or_else(|| missing_auth("aws_secret_access_key"))?,
        session_token: secret_string_optional(model, "aws_session_token")
            .map(str::to_string)
            .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok()),
    };
    let region = bedrock_region(model);
    let url = parse_url(
        &model
            .auth_config
            .get("aws_sts_endpoint")
            .cloned()
            .unwrap_or_else(|| format!("https://sts.{region}.amazonaws.com")),
    )?;
    let role = required_auth(model, "aws_role_name")?;
    let session_name = model
        .auth_config
        .get("aws_session_name")
        .map(String::as_str)
        .unwrap_or("ai-gateway-bedrock");
    let body = format!(
        "Action=AssumeRole&Version=2011-06-15&RoleArn={}&RoleSessionName={}",
        form_encode(role),
        form_encode(session_name)
    );
    let headers = aws_sigv4_headers(&url, body.as_bytes(), &region, "sts", &base)?;
    let response = client()
        .post(url)
        .headers(headers)
        .header(
            header::CONTENT_TYPE,
            "application/x-www-form-urlencoded; charset=utf-8",
        )
        .body(body)
        .send()
        .await
        .map_err(|error| {
            GatewayRuntimeError::http(StatusCode::BAD_GATEWAY, Value::String(error.to_string()))
        })?;
    let status = response.status();
    let text = response.text().await.map_err(|error| {
        GatewayRuntimeError::http(StatusCode::BAD_GATEWAY, Value::String(error.to_string()))
    })?;
    if !status.is_success() {
        return Err(GatewayRuntimeError::http(status, Value::String(text)));
    }
    Ok(AwsCredentials {
        access_key: xml_value(&text, "AccessKeyId")?,
        secret_key: xml_value(&text, "SecretAccessKey")?,
        session_token: Some(xml_value(&text, "SessionToken")?),
    })
}

fn form_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn xml_value(xml: &str, tag: &str) -> Result<String, GatewayRuntimeError> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = xml
        .find(&start_tag)
        .map(|index| index + start_tag.len())
        .ok_or_else(|| GatewayRuntimeError::internal(format!("STS response omitted {tag}")))?;
    let end = xml[start..]
        .find(&end_tag)
        .map(|index| start + index)
        .ok_or_else(|| GatewayRuntimeError::internal(format!("STS response omitted {tag}")))?;
    Ok(xml[start..end].to_string())
}

fn bedrock_region(model: &ResolvedGatewayModelConfig) -> String {
    model
        .auth_config
        .get("aws_region_name")
        .cloned()
        .or_else(|| std::env::var("AWS_REGION").ok())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
        .unwrap_or_else(|| "us-east-1".to_string())
}

fn sign_bedrock_request(
    model: &ResolvedGatewayModelConfig,
    request: &mut ProviderRequest,
) -> Result<(), GatewayRuntimeError> {
    let credentials = AwsCredentials {
        access_key: secret_string_optional(model, "aws_access_key_id")
            .map(str::to_string)
            .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
            .ok_or_else(|| missing_auth("aws_access_key_id"))?,
        secret_key: secret_string_optional(model, "aws_secret_access_key")
            .map(str::to_string)
            .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
            .ok_or_else(|| missing_auth("aws_secret_access_key"))?,
        session_token: secret_string_optional(model, "aws_session_token")
            .map(str::to_string)
            .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok()),
    };
    sign_bedrock_with_credentials(model, request, &credentials)
}

fn sign_bedrock_with_credentials(
    model: &ResolvedGatewayModelConfig,
    request: &mut ProviderRequest,
    credentials: &AwsCredentials,
) -> Result<(), GatewayRuntimeError> {
    let payload = serde_json::to_vec(&request.body)
        .map_err(|error| GatewayRuntimeError::internal(error.to_string()))?;
    let headers = aws_sigv4_headers(
        &request.url,
        &payload,
        &bedrock_region(model),
        "bedrock",
        credentials,
    )?;
    request.headers.extend(headers);
    Ok(())
}

fn aws_sigv4_headers(
    url: &Url,
    payload: &[u8],
    region: &str,
    service: &str,
    credentials: &AwsCredentials,
) -> Result<HeaderMap, GatewayRuntimeError> {
    type HmacSha256 = Hmac<Sha256>;

    let amz_date = std::env::var("MLFLOW_GATEWAY_TEST_AWS_AMZ_DATE")
        .unwrap_or_else(|_| chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let date = amz_date
        .get(..8)
        .ok_or_else(|| GatewayRuntimeError::internal("Invalid MLFLOW_GATEWAY_TEST_AWS_AMZ_DATE"))?;
    let host = url
        .host_str()
        .ok_or_else(|| GatewayRuntimeError::internal("AWS URL has no host"))?;
    let payload_hash = sha256_hex(payload);
    let mut headers = HeaderMap::new();
    insert_header(&mut headers, "host", host)?;
    insert_header(&mut headers, "x-amz-date", &amz_date)?;
    insert_header(&mut headers, "x-amz-content-sha256", &payload_hash)?;
    if let Some(token) = &credentials.session_token {
        insert_header(&mut headers, "x-amz-security-token", token)?;
    }
    let mut canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
    let mut signed_headers = "host;x-amz-content-sha256;x-amz-date".to_string();
    if let Some(token) = &credentials.session_token {
        canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed_headers.push_str(";x-amz-security-token");
    }
    let canonical_request = format!(
        "POST\n{}\n{}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        url.path(),
        url.query().unwrap_or_default(),
    );
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );
    let date_key = hmac_bytes::<HmacSha256>(
        format!("AWS4{}", credentials.secret_key).as_bytes(),
        date.as_bytes(),
    )?;
    let region_key = hmac_bytes::<HmacSha256>(&date_key, region.as_bytes())?;
    let service_key = hmac_bytes::<HmacSha256>(&region_key, service.as_bytes())?;
    let signing_key = hmac_bytes::<HmacSha256>(&service_key, b"aws4_request")?;
    let signature = hex_bytes(&hmac_bytes::<HmacSha256>(
        &signing_key,
        string_to_sign.as_bytes(),
    )?);
    insert_header(
        &mut headers,
        "authorization",
        &format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            credentials.access_key
        ),
    )?;
    Ok(headers)
}

fn hmac_bytes<M>(key: &[u8], value: &[u8]) -> Result<Vec<u8>, GatewayRuntimeError>
where
    M: Mac + hmac::digest::KeyInit,
{
    let mut mac = <M as Mac>::new_from_slice(key)
        .map_err(|_| GatewayRuntimeError::internal("Invalid HMAC key"))?;
    mac.update(value);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn sha256_hex(value: &[u8]) -> String {
    use sha2::Digest as _;
    hex_bytes(&Sha256::digest(value))
}

fn hex_bytes(value: &[u8]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn normalize_databricks_content(response: &mut Value) {
    let Some(choices) = response.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        let message = if choice.get("message").is_some() {
            choice.get_mut("message")
        } else {
            choice.get_mut("delta")
        };
        let Some(content) = message.and_then(|value| value.get_mut("content")) else {
            continue;
        };
        let Some(parts) = content.as_array() else {
            continue;
        };
        let supported = parts
            .iter()
            .filter(|part| {
                matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("text" | "image_url" | "input_audio")
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        if supported
            .iter()
            .all(|part| part.get("type") == Some(&json!("text")))
        {
            let text = supported
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            *content = if text.is_empty() {
                Value::Null
            } else {
                Value::String(text)
            };
        } else {
            *content = Value::Array(supported);
        }
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

fn missing_auth(key: &str) -> GatewayRuntimeError {
    GatewayRuntimeError::new(
        StatusCode::BAD_REQUEST,
        format!("Missing required provider authentication field '{key}'."),
    )
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

fn secret_string_optional<'a>(model: &'a ResolvedGatewayModelConfig, key: &str) -> Option<&'a str> {
    model.secret_value.get(key).and_then(Value::as_str)
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
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::HashMap;

    #[test]
    fn typed_trace_payload_omits_none_fields_recursively() {
        let payload = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "unused": null},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": null}}
                ],
                "tool_calls": null
            }],
            "stop": ["done", null],
            "stream_options": null
        });

        assert_eq!(
            normalized_typed_trace_payload(payload),
            json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                    ]
                }],
                "stop": ["done", null]
            })
        );
    }

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
    fn traffic_split_truncates_to_percent_and_excludes_zero_weight() {
        assert_eq!(python_integer_weight(0.009).unwrap(), 0.0);
        assert_eq!(python_integer_weight(0.019).unwrap(), 1.0);
        assert_eq!(weighted_index_for_draw(&[0.0, 100.0], 0.0).unwrap(), 1);
        assert_eq!(weighted_index_for_draw(&[100.0, 0.0], 0.999).unwrap(), 0);
        assert_eq!(weighted_index_for_draw(&[25.0], 0.75).unwrap(), 0);
        assert_eq!(
            weighted_index_for_draw(&[0.0, 0.0], 0.5)
                .unwrap_err()
                .detail,
            "probabilities contain NaN"
        );
    }

    #[test]
    fn seeded_traffic_split_distribution_is_ci_stable() {
        let mut rng = StdRng::seed_from_u64(18_005);
        let mut counts = [0_usize; 2];
        for _ in 0..100_000 {
            let index = weighted_index_for_draw(&[69.0, 31.0], rng.gen()).unwrap();
            counts[index] += 1;
        }
        let first_share = counts[0] as f64 / 100_000.0;
        assert!(
            (first_share - 0.69).abs() < 0.01,
            "counts={counts:?}, share={first_share}"
        );
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

    #[test]
    fn every_pinned_provider_resolves_without_python_fallback() {
        let providers = crate::gateway_provider_matrix::supported_provider_names();
        assert_eq!(providers.len(), 191);
        for provider in providers {
            let adapter =
                adapter_for(provider).unwrap_or_else(|error| panic!("{provider}: {error:?}"));
            if crate::gateway_provider_matrix::provider_adapter_kind(provider)
                == Some("pinned_litellm_transform")
            {
                let mut model = fixture_model(provider);
                model
                    .auth_config
                    .insert("api_base".to_string(), "http://127.0.0.1:9/v1".to_string());
                let payload = json!({"messages":[{"role":"user","content":"fixture"}]});
                for stream in [false, true] {
                    let request = adapter
                        .transform_request(&model, InvocationKind::Chat, payload.clone(), stream)
                        .unwrap_or_else(|error| panic!("{provider} request: {error:?}"));
                    assert_eq!(request.body["model"], model.model_name, "{provider}");
                    assert!(
                        request.url.path().ends_with("/chat/completions"),
                        "{provider}"
                    );
                }

                let response = adapter
                    .transform_response(
                        &model,
                        InvocationKind::Chat,
                        json!({
                            "id":"fixture-id",
                            "object":"chat.completion",
                            "created":7,
                            "model":"fixture-model",
                            "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                            "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
                        }),
                        7,
                    )
                    .unwrap_or_else(|error| panic!("{provider} response: {error:?}"));
                assert_eq!(response["usage"]["total_tokens"], 5, "{provider}");

                let frames = adapter
                    .transform_stream_frame(
                        &model,
                        json!({
                            "id":"fixture-stream-id",
                            "object":"chat.completion.chunk",
                            "created":7,
                            "model":"fixture-model",
                            "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]
                        }),
                        &mut StreamTransformState::default(),
                        7,
                    )
                    .unwrap_or_else(|error| panic!("{provider} stream: {error:?}"));
                assert_eq!(frames.len(), 1, "{provider}");

                let error = adapter.map_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    json!({"error":{"message":"fixture limit"}}),
                );
                assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS, "{provider}");
                assert_eq!(
                    crate::gateway_provider_matrix::classify_retry(
                        error.status.as_u16(),
                        &json!({"error":{"message":"fixture limit"}})
                    ),
                    Some(crate::gateway_provider_matrix::RetryClass::RateLimitError),
                    "{provider}"
                );
            }
        }
        assert!(adapter_for("obvious-future-provider-not-in-pin").is_err());
    }

    #[test]
    fn openai_compatible_family_pins_base_url_auth_and_quirks() {
        let cases = [
            ("groq", "https://api.groq.com/openai/v1", "authorization"),
            ("deepseek", "https://api.deepseek.com/v1", "authorization"),
            ("xai", "https://api.x.ai/v1", "authorization"),
            (
                "openrouter",
                "https://openrouter.ai/api/v1",
                "authorization",
            ),
            ("ollama", "http://localhost:11434/v1", "authorization"),
            ("portkey", "http://127.0.0.1:9/v1", "x-portkey-api-key"),
        ];
        for (provider, expected_base, auth_header) in cases {
            let mut model = fixture_model(provider);
            if provider != "portkey" {
                model.auth_config.remove("api_base");
            }
            if provider == "ollama" {
                model.secret_value = json!({"api_key":"ollama"});
            }
            let adapter = OpenAiCompatibleAdapter {
                provider: provider.to_string(),
            };
            let request = adapter
                .transform_request(
                    &model,
                    InvocationKind::Chat,
                    json!({"messages":[{"role":"user","content":"hi"}]}),
                    false,
                )
                .unwrap();
            assert!(
                request.url.as_str().starts_with(expected_base),
                "{provider}"
            );
            if provider == "ollama" {
                assert!(request.headers.get("authorization").is_none());
            } else {
                assert!(request.headers.contains_key(auth_header), "{provider}");
            }
        }
    }

    #[test]
    fn subscription_clients_keep_their_own_auth_on_passthrough() {
        let model = fixture_model("openai");
        let headers = HeaderMap::from_iter([
            (
                header::USER_AGENT,
                HeaderValue::from_static("codex_cli_rs/1.0"),
            ),
            (
                header::AUTHORIZATION,
                HeaderValue::from_static("Bearer obvious-fake-client-subscription"),
            ),
            (
                HeaderName::from_static("x-mlflow-authorization"),
                HeaderValue::from_static("obvious-fake-mlflow-rbac"),
            ),
        ]);
        let request = OpenAiAdapter
            .passthrough_request(
                &model,
                PassthroughAction::OpenAiChat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                &headers,
            )
            .unwrap();
        assert_eq!(
            request.headers[header::AUTHORIZATION],
            "Bearer obvious-fake-client-subscription"
        );
        assert!(request.headers.get("x-mlflow-authorization").is_none());
    }

    #[test]
    fn bedrock_access_keys_are_sigv4_signed_without_live_credentials() {
        let mut model = fixture_model("bedrock");
        model.model_name = "anthropic.claude-fixture".to_string();
        model.auth_config.extend([
            ("auth_mode".to_string(), "access_keys".to_string()),
            ("aws_region_name".to_string(), "us-test-1".to_string()),
        ]);
        model.secret_value = json!({
            "aws_access_key_id":"OBVIOUSFAKEACCESSKEY",
            "aws_secret_access_key":"obvious-fake-secret-access-key",
            "aws_session_token":"obvious-fake-session-token"
        });
        let request = BedrockAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                false,
            )
            .unwrap();
        let authorization = request.headers[header::AUTHORIZATION].to_str().unwrap();
        assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=OBVIOUSFAKEACCESSKEY/"));
        assert!(authorization.contains("/us-test-1/bedrock/aws4_request"));
        assert!(authorization.contains("x-amz-security-token"));
        assert_eq!(
            request.headers["x-amz-security-token"],
            "obvious-fake-session-token"
        );
        assert_eq!(request.body["anthropic_version"], "bedrock-2023-05-31");
    }

    #[test]
    fn bedrock_api_key_and_default_chain_auth_modes_are_native() {
        let mut token_model = fixture_model("bedrock");
        token_model
            .auth_config
            .insert("auth_mode".to_string(), "api_key".to_string());
        let token_request = BedrockAdapter
            .transform_request(
                &token_model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                false,
            )
            .unwrap();
        assert_eq!(
            token_request.headers[header::AUTHORIZATION],
            "Bearer obvious-fake-key"
        );

        let mut model = fixture_model("bedrock");
        model.auth_config.extend([
            ("auth_mode".to_string(), "default_chain".to_string()),
            ("aws_region_name".to_string(), "us-test-1".to_string()),
        ]);
        // Recorded fake credentials stand in for default-chain resolution;
        // the runtime never contacts AWS in tests.
        model.secret_value = json!({
            "aws_access_key_id":"OBVIOUSFAKEDEFAULTKEY",
            "aws_secret_access_key":"obvious-fake-default-secret",
            "aws_session_token":"obvious-fake-default-session"
        });
        let request = BedrockAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                false,
            )
            .unwrap();
        assert!(request.headers[header::AUTHORIZATION]
            .to_str()
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256"));
    }

    #[tokio::test]
    async fn bedrock_iam_role_uses_mock_sts_then_signs_with_assumed_credentials() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let sts_base = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new().fallback(axum::routing::post(
            |headers: HeaderMap, body: Bytes| async move {
                let authorization = headers[header::AUTHORIZATION].to_str().unwrap();
                assert!(
                    authorization.starts_with("AWS4-HMAC-SHA256 Credential=OBVIOUSFAKEBASEKEY/")
                );
                assert!(authorization.contains("/sts/aws4_request"));
                assert!(String::from_utf8_lossy(&body).contains("Action=AssumeRole"));
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/xml")
                    .body(Body::from(
                        "<AssumeRoleResponse><AssumeRoleResult><Credentials>\
                         <AccessKeyId>OBVIOUSFAKEASSUMEDKEY</AccessKeyId>\
                         <SecretAccessKey>obvious-fake-assumed-secret</SecretAccessKey>\
                         <SessionToken>obvious-fake-assumed-session</SessionToken>\
                         </Credentials></AssumeRoleResult></AssumeRoleResponse>",
                    ))
                    .unwrap()
            },
        ));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut model = fixture_model("bedrock");
        model.auth_config.extend([
            ("auth_mode".to_string(), "iam_role".to_string()),
            ("aws_region_name".to_string(), "us-test-1".to_string()),
            (
                "aws_role_name".to_string(),
                "arn:aws:iam::000000000000:role/obvious-fake-role".to_string(),
            ),
            ("aws_sts_endpoint".to_string(), sts_base),
        ]);
        model.secret_value = json!({
            "aws_access_key_id":"OBVIOUSFAKEBASEKEY",
            "aws_secret_access_key":"obvious-fake-base-secret"
        });
        let mut request = BedrockAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                false,
            )
            .unwrap();
        assert!(request.headers.get(header::AUTHORIZATION).is_none());
        prepare_dynamic_auth(&model, &mut request).await.unwrap();
        let authorization = request.headers[header::AUTHORIZATION].to_str().unwrap();
        assert!(authorization.contains("Credential=OBVIOUSFAKEASSUMEDKEY/"));
        assert_eq!(
            request.headers["x-amz-security-token"],
            "obvious-fake-assumed-session"
        );
    }

    #[test]
    fn databricks_pat_uses_normalized_serving_base_and_bearer_auth() {
        let mut model = fixture_model("databricks");
        model
            .auth_config
            .insert("auth_mode".to_string(), "pat_token".to_string());
        let request = DatabricksAdapter
            .transform_request(
                &model,
                InvocationKind::Chat,
                json!({"messages":[{"role":"user","content":"hi"}]}),
                false,
            )
            .unwrap();
        assert!(request
            .url
            .path()
            .contains("/serving-endpoints/chat/completions"));
        assert_eq!(
            request.headers[header::AUTHORIZATION],
            "Bearer obvious-fake-key"
        );
    }

    #[tokio::test]
    async fn databricks_oauth_m2m_injects_mock_token_response() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new().route(
            "/oidc/v1/token",
            axum::routing::post(|headers: HeaderMap, body: Bytes| async move {
                assert!(headers[header::AUTHORIZATION]
                    .to_str()
                    .unwrap()
                    .starts_with("Basic "));
                assert_eq!(
                    body,
                    Bytes::from_static(b"grant_type=client_credentials&scope=all-apis")
                );
                json_response(
                    StatusCode::OK,
                    json!({"access_token":"obvious-fake-oauth-access-token","token_type":"Bearer"}),
                )
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut model = fixture_model("databricks");
        model.auth_config = HashMap::from([
            ("api_base".to_string(), base.clone()),
            ("auth_mode".to_string(), "oauth_m2m".to_string()),
            (
                "client_id".to_string(),
                "obvious-fake-client-id".to_string(),
            ),
        ]);
        model.secret_value = json!({"client_secret":"obvious-fake-client-secret"});
        let mut request = ProviderRequest {
            url: parse_url(&format!("{base}/serving-endpoints/chat/completions")).unwrap(),
            headers: HeaderMap::new(),
            body: json!({}),
        };
        prepare_dynamic_auth(&model, &mut request).await.unwrap();
        assert_eq!(
            request.headers[header::AUTHORIZATION],
            "Bearer obvious-fake-oauth-access-token"
        );
    }
}
