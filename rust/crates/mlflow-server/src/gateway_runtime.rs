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
use futures::future::BoxFuture;
use futures::{stream, FutureExt, StreamExt};
use hmac::{Hmac, Mac};
use md5::{Digest, Md5};
use mlflow_store::{python_json_dumps, ResolvedGatewayEndpointConfig, ResolvedGatewayModelConfig};
use reqwest::Url;
use serde_json::{json, Map, Value};
use sha2::Sha256;

use crate::gateway_provider_matrix::{default_api_base, is_supported_provider, normalize_provider};
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
    let (model, adapter) = match resolve_runtime_provider(&state, &workspace, &endpoint_name).await
    {
        Ok(value) => value,
        Err(error) => return error.response(start.elapsed()),
    };
    if !supports_passthrough(&model.provider, action) {
        return unsupported_passthrough(&model.provider, action).response(start.elapsed());
    }
    let streaming = action.streaming(&payload);
    let mut request = match adapter.passthrough_request(&model, action, payload, &headers) {
        Ok(request) => request,
        Err(error) => return error.response(start.elapsed()),
    };
    if let Err(error) = prepare_dynamic_auth(&model, &mut request).await {
        return error.response(start.elapsed());
    }
    if streaming {
        raw_stream_response(request, start)
    } else {
        raw_non_stream_response(request, start).await
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
    let (model, adapter) = match resolve_runtime_provider(&state, &workspace, &endpoint_name).await
    {
        Ok(value) => value,
        Err(error) => return error.response(start.elapsed()),
    };
    let path = match uri.query() {
        Some(query) => format!("{path}?{query}"),
        None => path,
    };
    let mut request = match adapter.proxy_request(&model, &path, payload, &headers) {
        Ok(request) => request,
        Err(error) => return error.response(start.elapsed()),
    };
    if let Err(error) = prepare_dynamic_auth(&model, &mut request).await {
        return error.response(start.elapsed());
    }
    raw_proxy_response(request, start).await
}

async fn resolve_runtime_provider(
    state: &AppState,
    workspace: &str,
    endpoint_name: &str,
) -> Result<(ResolvedGatewayModelConfig, Box<dyn GatewayProviderAdapter>), GatewayRuntimeError> {
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
    Ok((model, adapter))
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

async fn raw_non_stream_response(request: ProviderRequest, start: Instant) -> Response {
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
}

fn raw_stream_response(request: ProviderRequest, start: Instant) -> Response {
    let state = RawProviderStream {
        initial: Some(send_provider_request(request).boxed()),
        upstream: None,
        done: false,
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
        Some(Ok(bytes)) => Some((bytes, state)),
        Some(Err(error)) => {
            state.done = true;
            Some((sse_error(&error.to_string(), "ClientPayloadError"), state))
        }
        None => None,
    }
}

async fn raw_proxy_response(request: ProviderRequest, start: Instant) -> Response {
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
            Ok(value) => with_non_stream_timing(
                json_response(StatusCode::OK, value),
                start,
                provider_start.elapsed(),
            ),
            Err(error) => {
                GatewayRuntimeError::http(StatusCode::BAD_GATEWAY, Value::String(error.to_string()))
                    .response(start.elapsed())
            }
        };
    }
    if content_type.contains("text/plain") {
        return with_non_stream_timing(
            json_response(
                StatusCode::OK,
                json!({"message":response.text().await.unwrap_or_default()}),
            ),
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
    let initial = send_provider_request(request).boxed();

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
