//! AI Gateway CRUD, discovery, and legacy deployments-bridge handlers.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::http::{header, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_store::{
    BudgetPolicy, BudgetPolicyUpdate, Endpoint, EndpointBinding, EndpointModelConfig,
    EndpointModelMapping, EndpointUpdate, FallbackConfig, GatewayGuardrail, GatewayGuardrailConfig,
    GatewayModelDefinition, GatewaySecretInfo, ScorerVersion,
};

use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

include!(concat!(env!("OUT_DIR"), "/model_catalog.rs"));

const MODEL_CATALOG_URI_ENV: &str = "MLFLOW_MODEL_CATALOG_URI";
const MODEL_CATALOG_CACHE_TTL_ENV: &str = "MLFLOW_MODEL_CATALOG_CACHE_TTL";
const DEFAULT_MODEL_CATALOG_URI: &str =
    "https://github.com/mlflow/mlflow/releases/download/model-catalog%2Flatest";
const ALLOWED_PROVIDERS_ENV: &str = "MLFLOW_GATEWAY_ALLOWED_PROVIDERS";
const DEPLOYMENTS_TARGET_ENV: &str = "MLFLOW_DEPLOYMENTS_TARGET";
const KEK_PASSPHRASE_ENV: &str = "MLFLOW_CRYPTO_KEK_PASSPHRASE";

const BEDROCK_CONFIG: &str = r#"{"auth_modes":[{"config_fields":[{"description":"AWS Region","name":"aws_region_name","required":true,"type":"string"}],"description":"Use Amazon Bedrock API Key (bearer token)","display_name":"API Key","mode":"api_key","secret_fields":[{"description":"Amazon Bedrock API Key","name":"api_key","required":true,"type":"string"}]},{"config_fields":[{"description":"AWS Region (e.g., us-east-1)","name":"aws_region_name","required":false,"type":"string"}],"description":"Use AWS Access Key ID and Secret Access Key","display_name":"Access Keys","mode":"access_keys","secret_fields":[{"description":"AWS Access Key ID","name":"aws_access_key_id","required":true,"type":"string"},{"description":"AWS Secret Access Key","name":"aws_secret_access_key","required":true,"type":"string"}]},{"config_fields":[{"description":"IAM Role ARN to assume","name":"aws_role_name","required":true,"type":"string"},{"description":"AWS Region (e.g., us-east-1)","name":"aws_region_name","required":false,"type":"string"}],"description":"Assume an IAM role using the server's ambient credentials (instance profile, IRSA, ECS task role, ~/.aws/credentials, etc.)","display_name":"IAM Role Assumption","mode":"iam_role","secret_fields":[]},{"config_fields":[{"description":"IAM Role ARN to assume (optional, for cross-account access)","name":"aws_role_name","required":false,"type":"string"},{"description":"Session name for assumed role","name":"aws_session_name","required":false,"type":"string"},{"description":"AWS Region (e.g., us-east-1)","name":"aws_region_name","required":false,"type":"string"}],"description":"Use the server's default AWS credentials (instance profile, IRSA, ECS task role, ~/.aws/credentials, etc.)","display_name":"Default Credential Chain","mode":"default_chain","secret_fields":[]}],"default_mode":"api_key"}"#;
const AZURE_CONFIG: &str = r#"{"auth_modes":[{"config_fields":[{"description":"Azure OpenAI endpoint URL","name":"api_base","required":true,"type":"string"},{"description":"API version (e.g., 2024-02-01)","name":"api_version","required":true,"type":"string"}],"description":"Use Azure OpenAI API Key","display_name":"API Key","mode":"api_key","secret_fields":[{"description":"Azure OpenAI API Key","name":"api_key","required":true,"type":"string"}]}],"default_mode":"api_key"}"#;
const VERTEX_AI_CONFIG: &str = r#"{"auth_modes":[{"config_fields":[{"description":"GCP Project ID","name":"vertex_project","required":true,"type":"string"},{"default":"us-central1","description":"GCP Region (e.g., us-central1)","name":"vertex_location","required":false,"type":"string"}],"description":"Use GCP Service Account credentials (JSON key file contents)","display_name":"Service Account JSON","mode":"service_account_json","secret_fields":[{"description":"Service Account JSON key file contents","name":"vertex_credentials","required":true,"type":"string"}]},{"config_fields":[{"description":"GCP Project ID","name":"vertex_project","required":true,"type":"string"},{"default":"us-central1","description":"GCP Region (e.g., us-central1)","name":"vertex_location","required":false,"type":"string"}],"description":"Use the server's Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS, gcloud auth application-default login, or attached GCE/GKE/Cloud Run service account)","display_name":"Application Default Credentials","mode":"default_chain","secret_fields":[]}],"default_mode":"service_account_json"}"#;
const DATABRICKS_CONFIG: &str = r#"{"auth_modes":[{"config_fields":[{"description":"Databricks workspace URL","name":"api_base","required":true,"type":"string"}],"description":"Use Databricks Personal Access Token","display_name":"Personal Access Token","mode":"pat_token","secret_fields":[{"description":"Databricks Personal Access Token","name":"api_key","required":true,"type":"string"}]},{"config_fields":[{"description":"Databricks workspace URL","name":"api_base","required":true,"type":"string"},{"description":"OAuth Client ID","name":"client_id","required":true,"type":"string"}],"description":"Use OAuth machine-to-machine authentication","display_name":"OAuth M2M (Service Principal)","mode":"oauth_m2m","secret_fields":[{"description":"OAuth Client Secret","name":"client_secret","required":true,"type":"string"}]}],"default_mode":"pat_token"}"#;
const SAGEMAKER_CONFIG: &str = r#"{"auth_modes":[{"config_fields":[{"description":"AWS Region (e.g., us-east-1)","name":"aws_region_name","required":true,"type":"string"}],"description":"Use AWS Access Key ID and Secret Access Key","display_name":"Access Keys","mode":"access_keys","secret_fields":[{"description":"AWS Access Key ID","name":"aws_access_key_id","required":true,"type":"string"},{"description":"AWS Secret Access Key","name":"aws_secret_access_key","required":true,"type":"string"}]},{"config_fields":[{"description":"IAM Role ARN to assume","name":"aws_role_name","required":true,"type":"string"},{"description":"Session name for assumed role","name":"aws_session_name","required":false,"type":"string"},{"description":"AWS Region (e.g., us-east-1)","name":"aws_region_name","required":true,"type":"string"}],"description":"Assume an IAM role using base credentials (for cross-account access)","display_name":"IAM Role Assumption","mode":"iam_role","secret_fields":[{"description":"AWS Access Key ID (for assuming role)","name":"aws_access_key_id","required":true,"type":"string"},{"description":"AWS Secret Access Key","name":"aws_secret_access_key","required":true,"type":"string"}]},{"config_fields":[{"description":"IAM Role ARN to assume (optional, for cross-account access)","name":"aws_role_name","required":false,"type":"string"},{"description":"Session name for assumed role","name":"aws_session_name","required":false,"type":"string"},{"description":"AWS Region (e.g., us-east-1)","name":"aws_region_name","required":false,"type":"string"}],"description":"Use the server's default AWS credentials (instance profile, IRSA, ECS task role, ~/.aws/credentials, etc.)","display_name":"Default Credential Chain","mode":"default_chain","secret_fields":[]}],"default_mode":"access_keys"}"#;

type CatalogModels = serde_json::Map<String, serde_json::Value>;

#[derive(Clone)]
struct CachedCatalog {
    inserted: Instant,
    models: Option<CatalogModels>,
}

fn bundled_catalogs() -> &'static Vec<(&'static str, CatalogModels)> {
    static CATALOGS: OnceLock<Vec<(&'static str, CatalogModels)>> = OnceLock::new();
    CATALOGS.get_or_init(|| {
        BUNDLED_MODEL_CATALOGS
            .iter()
            .map(|(provider, raw)| {
                let value: serde_json::Value =
                    serde_json::from_str(raw).expect("bundled model catalog is valid JSON");
                let models = value
                    .get("models")
                    .and_then(serde_json::Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                (*provider, models)
            })
            .collect()
    })
}

fn normalize_provider_alias(provider: &str) -> &str {
    match provider {
        "amazon-bedrock" => "bedrock",
        "databricks-model-serving" => "databricks",
        _ => provider,
    }
}

fn consolidate_provider(provider: &str) -> &str {
    if provider == "vertex_ai" || provider.starts_with("vertex_ai-") {
        "vertex_ai"
    } else {
        provider
    }
}

fn allowed_providers() -> Option<HashSet<String>> {
    let raw = std::env::var(ALLOWED_PROVIDERS_ENV).ok()?;
    let providers = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| normalize_provider_alias(&value.to_ascii_lowercase()).to_string())
        .collect::<HashSet<_>>();
    (!providers.is_empty()).then_some(providers)
}

fn is_provider_allowed(provider: &str) -> bool {
    allowed_providers().is_none_or(|allowed| {
        allowed.contains(normalize_provider_alias(&provider.to_ascii_lowercase()))
    })
}

/// `GET /ajax-api/3.0/mlflow/gateway/supported-providers`.
pub async fn supported_providers(_workspace: Workspace) -> Response {
    let mut providers = bundled_catalogs()
        .iter()
        .filter_map(|(provider, _)| {
            (*provider != "bedrock_converse")
                .then(|| consolidate_provider(provider))
                .filter(|provider| is_provider_allowed(provider))
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    providers.sort_unstable();
    flask_json_response(&serde_json::json!({"providers": providers}))
}

/// `GET /ajax-api/3.0/mlflow/gateway/supported-models`.
pub async fn supported_models(_workspace: Workspace, parts: Parts) -> Response {
    let provider_filter = first_query_value(&parts, "provider");
    match model_list(provider_filter.as_deref()).await {
        Ok(models) => flask_json_response(&serde_json::json!({"models": models})),
        Err(()) => generic_500_response(),
    }
}

/// `GET /ajax-api/3.0/mlflow/gateway/provider-config`.
pub async fn provider_config(_workspace: Workspace, parts: Parts) -> Result<Response, MlflowError> {
    let provider = first_query_value(&parts, "provider").unwrap_or_default();
    if provider.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Provider parameter is required",
        ));
    }
    if !is_provider_allowed(&provider) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Provider '{provider}' is not allowed by the current gateway provider policy."
        )));
    }
    let normalized = normalize_provider_alias(&provider.to_ascii_lowercase()).to_string();
    let raw = match normalized.as_str() {
        "bedrock" | "bedrock_converse" => Some(BEDROCK_CONFIG),
        "azure" => Some(AZURE_CONFIG),
        "vertex_ai" => Some(VERTEX_AI_CONFIG),
        "databricks" => Some(DATABRICKS_CONFIG),
        "sagemaker" => Some(SAGEMAKER_CONFIG),
        _ => None,
    };
    let config = match raw {
        Some(raw) => serde_json::from_str(raw).expect("static provider config is valid JSON"),
        None => simple_provider_config(&normalized),
    };
    Ok(flask_json_response(&config))
}

/// `GET /ajax-api/3.0/mlflow/gateway/secrets/config`.
pub async fn secrets_config(_workspace: Workspace) -> Response {
    let using_default_passphrase = std::env::var(KEK_PASSPHRASE_ENV)
        .ok()
        .is_none_or(|value| value.is_empty());
    flask_json_response(&serde_json::json!({
        "secrets_available": true,
        "using_default_passphrase": using_default_passphrase,
    }))
}

/// GET/POST `/ajax-api/2.0/mlflow/gateway-proxy`.
pub async fn gateway_proxy(_workspace: Workspace, parts: Parts, body: Bytes) -> Response {
    let Some(target) = std::env::var(DEPLOYMENTS_TARGET_ENV)
        .ok()
        .filter(|value| !value.is_empty())
    else {
        // Python returns before reading request args/JSON or validating the path.
        return flask_json_response(&serde_json::json!({"endpoints": []}));
    };

    let (gateway_path, json_data) = if parts.method == Method::GET {
        (
            first_query_value(&parts, "gateway_path").map(serde_json::Value::String),
            first_query_value(&parts, "json_data").map(serde_json::Value::String),
        )
    } else {
        let Some(content_type) = parts
            .headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
        else {
            return unsupported_media_type_response();
        };
        let media_type = content_type.split(';').next().unwrap_or_default().trim();
        if media_type != "application/json"
            && !(media_type.starts_with("application/") && media_type.ends_with("+json"))
        {
            return unsupported_media_type_response();
        }
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body) else {
            return bad_request_html_response();
        };
        let Some(object) = value.as_object() else {
            return generic_500_response();
        };
        (
            object.get("gateway_path").cloned(),
            object.get("json_data").cloned(),
        )
    };

    let gateway_path = match gateway_path {
        None | Some(serde_json::Value::Null) => {
            return MlflowError::invalid_parameter_value(
                "Deployments proxy request must specify a gateway_path.",
            )
            .into_response();
        }
        Some(serde_json::Value::String(value)) if value.is_empty() => {
            return MlflowError::invalid_parameter_value(
                "Deployments proxy request must specify a gateway_path.",
            )
            .into_response();
        }
        Some(serde_json::Value::String(value)) => value,
        Some(value) if !python_truthy(&value) => {
            return MlflowError::invalid_parameter_value(
                "Deployments proxy request must specify a gateway_path.",
            )
            .into_response();
        }
        Some(_) => return generic_500_response(),
    };

    if !valid_gateway_path(&parts.method, &gateway_path) {
        return MlflowError::invalid_parameter_value(format!(
            "Invalid gateway_path: {gateway_path} for method: {}",
            parts.method
        ))
        .into_response();
    }

    let url = format!("{target}/{gateway_path}");
    let mut request = reqwest::Client::new().request(parts.method.clone(), url);
    if let Some(json_data) = json_data.filter(|value| !value.is_null()) {
        request = request
            .header(header::CONTENT_TYPE, "application/json")
            .body(python_json_dumps(&json_data, false, true));
    }
    let Ok(response) = request.send().await else {
        return generic_500_response();
    };
    let status = response.status();
    let Ok(text) = response.text().await else {
        return generic_500_response();
    };
    if status != StatusCode::OK {
        return MlflowError::internal_error(format!(
            "Deployments proxy request failed with error code {}. Error message: {text}",
            status.as_u16()
        ))
        .into_response();
    }
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(serde_json::Value::Object(value)) => {
            flask_json_response(&serde_json::Value::Object(value))
        }
        Ok(serde_json::Value::Array(value)) => {
            flask_json_response(&serde_json::Value::Array(value))
        }
        Ok(serde_json::Value::String(value)) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(axum::body::Body::from(value))
            .expect("valid string response"),
        Ok(_) => generic_500_response(),
        Err(_) => generic_500_response(),
    }
}

async fn model_list(provider_filter: Option<&str>) -> Result<Vec<serde_json::Value>, ()> {
    let mut models = Vec::new();
    let mut seen = HashSet::new();
    for (file_provider, bundled) in bundled_catalogs() {
        if *file_provider == "bedrock_converse" {
            continue;
        }
        let normalized = consolidate_provider(file_provider);
        if provider_filter.is_some_and(|filter| filter != normalized) {
            continue;
        }
        let entries = load_provider(file_provider, bundled).await?;
        for (raw_name, entry) in entries {
            let mode = entry
                .get("mode")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            if !matches!(mode.as_str(), Some("chat" | "completion" | "embedding"))
                && !mode.is_null()
            {
                continue;
            }
            if !is_provider_allowed(normalized) {
                continue;
            }
            let prefix = format!("{normalized}/");
            let model_name = raw_name.strip_prefix(&prefix).unwrap_or(&raw_name);
            if model_name.starts_with("ft:") {
                continue;
            }
            if !seen.insert((normalized.to_string(), model_name.to_string())) {
                continue;
            }
            models.push(model_response(model_name, normalized, &mode, &entry));
        }
    }
    Ok(models)
}

fn model_response(
    model_name: &str,
    provider: &str,
    mode: &serde_json::Value,
    entry: &serde_json::Value,
) -> serde_json::Value {
    let context = entry
        .get("context_window")
        .and_then(serde_json::Value::as_object);
    let pricing = entry.get("pricing").and_then(serde_json::Value::as_object);
    let capabilities = entry
        .get("capabilities")
        .and_then(serde_json::Value::as_object);
    let per_token = |field: &str| {
        pricing
            .and_then(|value| value.get(field))
            .and_then(serde_json::Value::as_f64)
            .and_then(|value| serde_json::Number::from_f64(value / 1_000_000.0))
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    };
    let capability = |field: &str| {
        capabilities
            .and_then(|value| value.get(field))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    serde_json::json!({
        "deprecation_date": entry.get("deprecation_date").cloned().unwrap_or(serde_json::Value::Null),
        "input_cost_per_token": per_token("input_per_million_tokens"),
        "last_updated_at": entry.get("last_updated_at").cloned().unwrap_or(serde_json::Value::Null),
        "max_input_tokens": context.and_then(|value| value.get("max_input")).cloned().unwrap_or(serde_json::Value::Null),
        "max_output_tokens": context.and_then(|value| value.get("max_output")).cloned().unwrap_or(serde_json::Value::Null),
        "modality": pricing.and_then(|value| value.get("modality")).cloned().unwrap_or(serde_json::Value::Null),
        "mode": mode,
        "model": model_name,
        "output_cost_per_token": per_token("output_per_million_tokens"),
        "provider": provider,
        "supports_function_calling": capability("function_calling"),
        "supports_prompt_caching": capability("prompt_caching"),
        "supports_reasoning": capability("reasoning"),
        "supports_response_schema": capability("response_schema"),
        "supports_vision": capability("vision"),
    })
}

async fn load_provider(provider: &str, bundled: &CatalogModels) -> Result<CatalogModels, ()> {
    let base = std::env::var(MODEL_CATALOG_URI_ENV)
        .unwrap_or_else(|_| DEFAULT_MODEL_CATALOG_URI.to_string());
    if base.is_empty() {
        return Ok(bundled.clone());
    }
    if let Some(models) = cached_remote_provider(provider).await? {
        if !models.is_empty() {
            return Ok(models);
        }
    }
    Ok(bundled.clone())
}

async fn cached_remote_provider(provider: &str) -> Result<Option<CatalogModels>, ()> {
    static CACHE: OnceLock<tokio::sync::Mutex<HashMap<String, CachedCatalog>>> = OnceLock::new();
    static TTL: OnceLock<Result<Duration, ()>> = OnceLock::new();
    let ttl = *TTL.get_or_init(|| {
        let seconds = match std::env::var(MODEL_CATALOG_CACHE_TTL_ENV) {
            Ok(value) => value.parse::<i64>().map_err(|_| ())?,
            Err(_) => 86_400,
        };
        Ok(Duration::from_secs(seconds.max(0) as u64))
    });
    let ttl = ttl?;
    let cache = CACHE.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()));
    if let Some(entry) = cache.lock().await.get(provider).cloned() {
        if entry.inserted.elapsed() < ttl {
            return Ok(entry.models);
        }
    }
    let models = fetch_remote_provider(provider).await?;
    cache.lock().await.insert(
        provider.to_string(),
        CachedCatalog {
            inserted: Instant::now(),
            models: models.clone(),
        },
    );
    Ok(models)
}

async fn fetch_remote_provider(provider: &str) -> Result<Option<CatalogModels>, ()> {
    let base = std::env::var(MODEL_CATALOG_URI_ENV)
        .unwrap_or_else(|_| DEFAULT_MODEL_CATALOG_URI.to_string());
    let url = format!("{}/{provider}.json", base.trim_end_matches('/'));
    let value = if let Some(path) = url.strip_prefix("file://") {
        let path = percent_decode_path(path);
        match tokio::fs::read_to_string(path).await {
            Ok(raw) => serde_json::from_str(&raw).ok(),
            Err(_) => None,
        }
    } else if url.starts_with("http://") || url.starts_with("https://") {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|_| ())?;
        let mut parsed = None;
        for attempt in 0..=3 {
            match client.get(&url).send().await {
                Ok(response) if response.status().is_success() => {
                    parsed = response.json::<serde_json::Value>().await.ok();
                    break;
                }
                Ok(response)
                    if matches!(
                        response.status().as_u16(),
                        404 | 408 | 429 | 500 | 502 | 503 | 504
                    ) && attempt < 3 =>
                {
                    tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
                }
                _ => break,
            }
        }
        parsed
    } else {
        return Err(());
    };
    Ok(value
        .and_then(|value| value.get("models").cloned())
        .and_then(|value| value.as_object().cloned()))
}

fn first_query_value(parts: &Parts, name: &str) -> Option<String> {
    parts
        .uri
        .query()
        .map(crate::proto_http::parse_query_pairs)
        .unwrap_or_default()
        .into_iter()
        .find_map(|(key, value)| (key == name).then_some(value))
}

fn simple_provider_config(provider: &str) -> serde_json::Value {
    let display = python_title(provider);
    serde_json::json!({
        "auth_modes": [{
            "config_fields": [{
                "description": format!("{display} API Base URL"),
                "name": "api_base",
                "required": false,
                "type": "string",
            }],
            "description": format!("Use {display} API Key"),
            "display_name": "API Key",
            "mode": "api_key",
            "secret_fields": [{
                "description": format!("{display} API Key"),
                "name": "api_key",
                "required": true,
                "type": "string",
            }],
        }],
        "default_mode": "api_key",
    })
}

fn python_title(value: &str) -> String {
    let mut start = true;
    value
        .chars()
        .map(|ch| {
            let out = if start {
                ch.to_ascii_uppercase()
            } else {
                ch.to_ascii_lowercase()
            };
            start = !ch.is_alphanumeric();
            out
        })
        .collect()
}

fn valid_gateway_path(method: &Method, path: &str) -> bool {
    let stripped = path.trim_matches('/');
    if *method == Method::GET {
        return stripped == "api/2.0/endpoints";
    }
    if *method != Method::POST {
        return true;
    }
    let Some(name) = stripped
        .strip_prefix("gateway/")
        .and_then(|value| value.strip_suffix("/invocations"))
    else {
        return false;
    };
    !name.is_empty() && !name.contains('/')
}

fn python_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value.as_f64() != Some(0.0),
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
    }
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = |value: u8| match value {
                b'0'..=b'9' => Some(value - b'0'),
                b'a'..=b'f' => Some(value - b'a' + 10),
                b'A'..=b'F' => Some(value - b'A' + 10),
                _ => None,
            };
            if let (Some(high), Some(low)) = (hex(bytes[index + 1]), hex(bytes[index + 2])) {
                out.push(high << 4 | low);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn flask_json_response(value: &serde_json::Value) -> Response {
    let mut body = python_json_dumps(value, true, false);
    body.push('\n');
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .expect("valid JSON response")
}

fn python_json_dumps(value: &serde_json::Value, sort_keys: bool, spaces: bool) -> String {
    fn write(out: &mut String, value: &serde_json::Value, sort_keys: bool, spaces: bool) {
        match value {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            serde_json::Value::Number(value) if value.is_f64() => {
                out.push_str(&mlflow_proto::python_float_repr(
                    value.as_f64().expect("f64 JSON number"),
                ));
            }
            serde_json::Value::Number(value) => out.push_str(&value.to_string()),
            serde_json::Value::String(value) => {
                out.push_str(&mlflow_proto::quote_json_string(value));
            }
            serde_json::Value::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                        if spaces {
                            out.push(' ');
                        }
                    }
                    write(out, value, sort_keys, spaces);
                }
                out.push(']');
            }
            serde_json::Value::Object(values) => {
                out.push('{');
                let mut entries = values.iter().collect::<Vec<_>>();
                if sort_keys {
                    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                }
                for (index, (key, value)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                        if spaces {
                            out.push(' ');
                        }
                    }
                    out.push_str(&mlflow_proto::quote_json_string(key));
                    out.push(':');
                    if spaces {
                        out.push(' ');
                    }
                    write(out, value, sort_keys, spaces);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    write(&mut out, value, sort_keys, spaces);
    out
}

fn generic_500_response() -> Response {
    const BODY: &str = "<!doctype html>\n<html lang=en>\n<title>500 Internal Server Error</title>\n<h1>Internal Server Error</h1>\n<p>The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application.</p>\n";
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        BODY,
    )
        .into_response()
}

fn unsupported_media_type_response() -> Response {
    const BODY: &str = "<!doctype html>\n<html lang=en>\n<title>415 Unsupported Media Type</title>\n<h1>Unsupported Media Type</h1>\n<p>Did not attempt to load JSON data because the request Content-Type was not &#39;application/json&#39;.</p>\n";
    (
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        BODY,
    )
        .into_response()
}

fn bad_request_html_response() -> Response {
    const BODY: &str = "<!doctype html>\n<html lang=en>\n<title>400 Bad Request</title>\n<h1>Bad Request</h1>\n<p>The browser (or proxy) sent a request that this server could not understand.</p>\n";
    (
        StatusCode::BAD_REQUEST,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        BODY,
    )
        .into_response()
}

macro_rules! parse {
    ($parts:expr, $body:expr, $ty:ty, $name:literal) => {
        parse_request::<$ty>($parts, $body, $name)?
    };
}

macro_rules! empty_response {
    ($ty:path, $name:literal) => {{
        let response: $ty = Default::default();
        proto_response(&response, $name)
    }};
}

pub async fn create_secret(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::CreateGatewaySecret,
        "mlflow.CreateGatewaySecret"
    );
    let secret_name = required(req.secret_name.as_deref(), "secret_name")?;
    let secret = state
        .tracking_store()
        .create_gateway_secret(
            workspace.name(),
            secret_name,
            &req.secret_value,
            req.provider.as_deref().filter(|v| !v.is_empty()),
            &req.auth_config,
            req.created_by.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::create_gateway_secret::Response {
            secret: Some(secret_proto(secret)),
        },
        "mlflow.CreateGatewaySecret.Response",
    )
}

pub async fn get_secret(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::GetGatewaySecretInfo,
        "mlflow.GetGatewaySecretInfo"
    );
    let id = required(req.secret_id.as_deref(), "secret_id")?;
    let secret = state
        .tracking_store()
        .get_gateway_secret_info(workspace.name(), Some(id), None)
        .await?;
    proto_response(
        &pb::get_gateway_secret_info::Response {
            secret: Some(secret_proto(secret)),
        },
        "mlflow.GetGatewaySecretInfo.Response",
    )
}

pub async fn update_secret(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::UpdateGatewaySecret,
        "mlflow.UpdateGatewaySecret"
    );
    let id = required(req.secret_id.as_deref(), "secret_id")?;
    let secret = state
        .tracking_store()
        .update_gateway_secret(
            workspace.name(),
            id,
            (!req.secret_value.is_empty()).then_some(&req.secret_value),
            (!req.auth_config.is_empty()).then_some(&req.auth_config),
            req.updated_by.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::update_gateway_secret::Response {
            secret: Some(secret_proto(secret)),
        },
        "mlflow.UpdateGatewaySecret.Response",
    )
}

pub async fn delete_secret(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewaySecret,
        "mlflow.DeleteGatewaySecret"
    );
    state
        .tracking_store()
        .delete_gateway_secret(
            workspace.name(),
            required(req.secret_id.as_deref(), "secret_id")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_secret::Response,
        "mlflow.DeleteGatewaySecret.Response"
    )
}

pub async fn list_secrets(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListGatewaySecretInfos,
        "mlflow.ListGatewaySecretInfos"
    );
    let secrets = state
        .tracking_store()
        .list_gateway_secret_infos(
            workspace.name(),
            req.provider.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::list_gateway_secret_infos::Response {
            secrets: secrets.into_iter().map(secret_proto).collect(),
        },
        "mlflow.ListGatewaySecretInfos.Response",
    )
}

pub async fn create_model_definition(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::CreateGatewayModelDefinition,
        "mlflow.CreateGatewayModelDefinition"
    );
    let model = state
        .tracking_store()
        .create_gateway_model_definition(
            workspace.name(),
            required(req.name.as_deref(), "name")?,
            required(req.secret_id.as_deref(), "secret_id")?,
            required(req.provider.as_deref(), "provider")?,
            required(req.model_name.as_deref(), "model_name")?,
            req.created_by.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::create_gateway_model_definition::Response {
            model_definition: Some(model_proto(model)),
        },
        "mlflow.CreateGatewayModelDefinition.Response",
    )
}

pub async fn get_model_definition(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::GetGatewayModelDefinition,
        "mlflow.GetGatewayModelDefinition"
    );
    let model = state
        .tracking_store()
        .get_gateway_model_definition(
            workspace.name(),
            Some(required(
                req.model_definition_id.as_deref(),
                "model_definition_id",
            )?),
            None,
        )
        .await?;
    proto_response(
        &pb::get_gateway_model_definition::Response {
            model_definition: Some(model_proto(model)),
        },
        "mlflow.GetGatewayModelDefinition.Response",
    )
}

pub async fn list_model_definitions(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListGatewayModelDefinitions,
        "mlflow.ListGatewayModelDefinitions"
    );
    let models = state
        .tracking_store()
        .list_gateway_model_definitions(
            workspace.name(),
            req.provider.as_deref().filter(|v| !v.is_empty()),
            req.secret_id.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::list_gateway_model_definitions::Response {
            model_definitions: models.into_iter().map(model_proto).collect(),
        },
        "mlflow.ListGatewayModelDefinitions.Response",
    )
}

pub async fn update_model_definition(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::UpdateGatewayModelDefinition,
        "mlflow.UpdateGatewayModelDefinition"
    );
    let model = state
        .tracking_store()
        .update_gateway_model_definition(
            workspace.name(),
            required(req.model_definition_id.as_deref(), "model_definition_id")?,
            req.name.as_deref().filter(|v| !v.is_empty()),
            req.secret_id.as_deref().filter(|v| !v.is_empty()),
            req.model_name.as_deref().filter(|v| !v.is_empty()),
            req.updated_by.as_deref().filter(|v| !v.is_empty()),
            req.provider.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::update_gateway_model_definition::Response {
            model_definition: Some(model_proto(model)),
        },
        "mlflow.UpdateGatewayModelDefinition.Response",
    )
}

pub async fn delete_model_definition(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewayModelDefinition,
        "mlflow.DeleteGatewayModelDefinition"
    );
    state
        .tracking_store()
        .delete_gateway_model_definition(
            workspace.name(),
            required(req.model_definition_id.as_deref(), "model_definition_id")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_model_definition::Response,
        "mlflow.DeleteGatewayModelDefinition.Response"
    )
}

pub async fn create_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::CreateGatewayEndpoint,
        "mlflow.CreateGatewayEndpoint"
    );
    let name = required(req.name.as_deref(), "name")?;
    validate_endpoint_name(name)?;
    let configs = req
        .model_configs
        .iter()
        .map(config_from_proto)
        .collect::<Result<Vec<_>, _>>()?;
    let fallback = req
        .fallback_config
        .as_ref()
        .map(fallback_from_proto)
        .transpose()?;
    let routing = req.routing_strategy.map(routing_name).transpose()?;
    let endpoint = state
        .tracking_store()
        .create_gateway_endpoint(
            workspace.name(),
            name,
            &configs,
            req.created_by.as_deref().filter(|v| !v.is_empty()),
            routing.as_deref(),
            fallback.as_ref(),
            req.experiment_id.as_deref(),
            req.usage_tracking.unwrap_or(true),
        )
        .await?;
    proto_response(
        &pb::create_gateway_endpoint::Response {
            endpoint: Some(endpoint_proto(endpoint)),
        },
        "mlflow.CreateGatewayEndpoint.Response",
    )
}

pub async fn get_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::GetGatewayEndpoint,
        "mlflow.GetGatewayEndpoint"
    );
    let endpoint = state
        .tracking_store()
        .get_gateway_endpoint(
            workspace.name(),
            req.endpoint_id.as_deref().filter(|v| !v.is_empty()),
            req.name.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::get_gateway_endpoint::Response {
            endpoint: Some(endpoint_proto(endpoint)),
        },
        "mlflow.GetGatewayEndpoint.Response",
    )
}

pub async fn list_endpoints(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListGatewayEndpoints,
        "mlflow.ListGatewayEndpoints"
    );
    let endpoints = state
        .tracking_store()
        .list_gateway_endpoints(
            workspace.name(),
            req.provider.as_deref().filter(|v| !v.is_empty()),
            None,
        )
        .await?;
    proto_response(
        &pb::list_gateway_endpoints::Response {
            endpoints: endpoints.into_iter().map(endpoint_proto).collect(),
        },
        "mlflow.ListGatewayEndpoints.Response",
    )
}

pub async fn update_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::UpdateGatewayEndpoint,
        "mlflow.UpdateGatewayEndpoint"
    );
    let endpoint_id = required(req.endpoint_id.as_deref(), "endpoint_id")?;
    if let Some(name) = req.name.as_deref().filter(|v| !v.is_empty()) {
        validate_endpoint_name(name)?;
    }
    let configs = (!req.model_configs.is_empty())
        .then(|| {
            req.model_configs
                .iter()
                .map(config_from_proto)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let fallback = req
        .fallback_config
        .as_ref()
        .map(fallback_from_proto)
        .transpose()?;
    let routing = req.routing_strategy.map(routing_name).transpose()?;
    let endpoint = state
        .tracking_store()
        .update_gateway_endpoint(
            workspace.name(),
            endpoint_id,
            EndpointUpdate {
                name: req.name.as_deref().filter(|v| !v.is_empty()),
                updated_by: req.updated_by.as_deref().filter(|v| !v.is_empty()),
                routing_strategy: routing.as_deref(),
                fallback_config: fallback.as_ref(),
                model_configs: configs.as_deref(),
                experiment_id: req.experiment_id.as_deref(),
                usage_tracking: req.usage_tracking,
            },
        )
        .await?;
    proto_response(
        &pb::update_gateway_endpoint::Response {
            endpoint: Some(endpoint_proto(endpoint)),
        },
        "mlflow.UpdateGatewayEndpoint.Response",
    )
}

pub async fn delete_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewayEndpoint,
        "mlflow.DeleteGatewayEndpoint"
    );
    state
        .tracking_store()
        .delete_gateway_endpoint(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_endpoint::Response,
        "mlflow.DeleteGatewayEndpoint.Response"
    )
}

pub async fn attach_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::AttachModelToGatewayEndpoint,
        "mlflow.AttachModelToGatewayEndpoint"
    );
    let config = req
        .model_config
        .as_ref()
        .ok_or_else(|| missing("model_config"))?;
    let mapping = state
        .tracking_store()
        .attach_model_to_gateway_endpoint(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            &config_from_proto(config)?,
            req.created_by.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::attach_model_to_gateway_endpoint::Response {
            mapping: Some(mapping_proto(mapping)),
        },
        "mlflow.AttachModelToGatewayEndpoint.Response",
    )
}

pub async fn detach_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DetachModelFromGatewayEndpoint,
        "mlflow.DetachModelFromGatewayEndpoint"
    );
    state
        .tracking_store()
        .detach_model_from_gateway_endpoint(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.model_definition_id.as_deref(), "model_definition_id")?,
        )
        .await?;
    empty_response!(
        pb::detach_model_from_gateway_endpoint::Response,
        "mlflow.DetachModelFromGatewayEndpoint.Response"
    )
}

pub async fn create_binding(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::CreateGatewayEndpointBinding,
        "mlflow.CreateGatewayEndpointBinding"
    );
    let binding = state
        .tracking_store()
        .create_gateway_endpoint_binding(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.resource_type.as_deref(), "resource_type")?,
            required(req.resource_id.as_deref(), "resource_id")?,
            req.created_by.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::create_gateway_endpoint_binding::Response {
            binding: Some(binding_proto(binding)),
        },
        "mlflow.CreateGatewayEndpointBinding.Response",
    )
}

pub async fn delete_binding(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewayEndpointBinding,
        "mlflow.DeleteGatewayEndpointBinding"
    );
    state
        .tracking_store()
        .delete_gateway_endpoint_binding(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.resource_type.as_deref(), "resource_type")?,
            required(req.resource_id.as_deref(), "resource_id")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_endpoint_binding::Response,
        "mlflow.DeleteGatewayEndpointBinding.Response"
    )
}

pub async fn list_bindings(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListGatewayEndpointBindings,
        "mlflow.ListGatewayEndpointBindings"
    );
    let bindings = state
        .tracking_store()
        .list_gateway_endpoint_bindings(
            workspace.name(),
            req.endpoint_id.as_deref().filter(|v| !v.is_empty()),
            req.resource_type.as_deref().filter(|v| !v.is_empty()),
            req.resource_id.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::list_gateway_endpoint_bindings::Response {
            bindings: bindings.into_iter().map(binding_proto).collect(),
        },
        "mlflow.ListGatewayEndpointBindings.Response",
    )
}

pub async fn set_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::SetGatewayEndpointTag,
        "mlflow.SetGatewayEndpointTag"
    );
    state
        .tracking_store()
        .set_gateway_endpoint_tag(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.key.as_deref(), "key")?,
            req.value.as_deref(),
        )
        .await?;
    empty_response!(
        pb::set_gateway_endpoint_tag::Response,
        "mlflow.SetGatewayEndpointTag.Response"
    )
}

pub async fn delete_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewayEndpointTag,
        "mlflow.DeleteGatewayEndpointTag"
    );
    state
        .tracking_store()
        .delete_gateway_endpoint_tag(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.key.as_deref(), "key")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_endpoint_tag::Response,
        "mlflow.DeleteGatewayEndpointTag.Response"
    )
}

pub async fn create_budget_policy(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::CreateGatewayBudgetPolicy,
        "mlflow.CreateGatewayBudgetPolicy"
    );
    let unit = budget_unit(required_enum(req.budget_unit, "budget_unit")?)?;
    let duration = req.duration.as_ref().ok_or_else(|| missing("duration"))?;
    let duration_unit = duration_unit(required_enum(duration.unit, "duration")?)?;
    let duration_value = required_positive(duration.value)?;
    let scope = target_scope(required_enum(req.target_scope, "target_scope")?)?;
    let action = budget_action(required_enum(req.budget_action, "budget_action")?)?;
    let amount = req.budget_amount.ok_or_else(|| missing("budget_amount"))?;
    let policy = state
        .tracking_store()
        .create_budget_policy(
            workspace.name(),
            unit,
            amount,
            duration_unit,
            duration_value,
            scope,
            action,
            req.created_by.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::create_gateway_budget_policy::Response {
            budget_policy: Some(budget_proto(policy)),
        },
        "mlflow.CreateGatewayBudgetPolicy.Response",
    )
}

pub async fn get_budget_policy(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::GetGatewayBudgetPolicy,
        "mlflow.GetGatewayBudgetPolicy"
    );
    let policy = state
        .tracking_store()
        .get_budget_policy(
            workspace.name(),
            required(req.budget_policy_id.as_deref(), "budget_policy_id")?,
        )
        .await?;
    proto_response(
        &pb::get_gateway_budget_policy::Response {
            budget_policy: Some(budget_proto(policy)),
        },
        "mlflow.GetGatewayBudgetPolicy.Response",
    )
}

pub async fn update_budget_policy(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::UpdateGatewayBudgetPolicy,
        "mlflow.UpdateGatewayBudgetPolicy"
    );
    let unit = req.budget_unit.map(budget_unit).transpose()?;
    let duration = req
        .duration
        .as_ref()
        .map(|v| {
            Ok((
                duration_unit(v.unit.unwrap_or_default())?,
                required_positive(v.value)?,
            ))
        })
        .transpose()?;
    let scope = req.target_scope.map(target_scope).transpose()?;
    let action = req.budget_action.map(budget_action).transpose()?;
    let policy = state
        .tracking_store()
        .update_budget_policy(
            workspace.name(),
            required(req.budget_policy_id.as_deref(), "budget_policy_id")?,
            BudgetPolicyUpdate {
                budget_unit: unit,
                budget_amount: req.budget_amount,
                duration,
                target_scope: scope,
                budget_action: action,
                updated_by: req.updated_by.as_deref().filter(|v| !v.is_empty()),
            },
        )
        .await?;
    proto_response(
        &pb::update_gateway_budget_policy::Response {
            budget_policy: Some(budget_proto(policy)),
        },
        "mlflow.UpdateGatewayBudgetPolicy.Response",
    )
}

pub async fn delete_budget_policy(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewayBudgetPolicy,
        "mlflow.DeleteGatewayBudgetPolicy"
    );
    state
        .tracking_store()
        .delete_budget_policy(
            workspace.name(),
            required(req.budget_policy_id.as_deref(), "budget_policy_id")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_budget_policy::Response,
        "mlflow.DeleteGatewayBudgetPolicy.Response"
    )
}

pub async fn list_budget_policies(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListGatewayBudgetPolicies,
        "mlflow.ListGatewayBudgetPolicies"
    );
    let page = state
        .tracking_store()
        .list_budget_policies(
            workspace.name(),
            req.max_results.unwrap_or(1000),
            req.page_token.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::list_gateway_budget_policies::Response {
            budget_policies: page.policies.into_iter().map(budget_proto).collect(),
            next_page_token: page.next_page_token,
        },
        "mlflow.ListGatewayBudgetPolicies.Response",
    )
}

pub async fn list_budget_windows(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let _: pb::ListGatewayBudgetWindows = parse!(
        &parts,
        &body,
        pb::ListGatewayBudgetWindows,
        "mlflow.ListGatewayBudgetWindows"
    );
    let windows = state
        .tracking_store()
        .list_budget_windows(workspace.name())
        .await?;
    proto_response(
        &pb::list_gateway_budget_windows::Response {
            windows: windows
                .into_iter()
                .map(|w| pb::list_gateway_budget_windows::BudgetWindow {
                    budget_policy_id: Some(w.budget_policy_id),
                    window_start_ms: Some(w.window_start_ms),
                    window_end_ms: Some(w.window_end_ms),
                    current_spend: Some(w.current_spend),
                })
                .collect(),
        },
        "mlflow.ListGatewayBudgetWindows.Response",
    )
}

pub async fn create_guardrail(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::CreateGatewayGuardrail,
        "mlflow.CreateGatewayGuardrail"
    );
    let stage = guardrail_stage(required_enum(req.stage, "stage")?)?;
    let action = guardrail_action(required_enum(req.action, "action")?)?;
    let version = req
        .scorer_version
        .ok_or_else(|| missing("scorer_version"))?;
    let version = i32::try_from(version).map_err(|_| {
        MlflowError::invalid_parameter_value(format!("Invalid scorer_version: {version}"))
    })?;
    let user = current_user();
    let guardrail = state
        .tracking_store()
        .create_gateway_guardrail(
            workspace.name(),
            required(req.name.as_deref(), "name")?,
            required(req.scorer_id.as_deref(), "scorer_id")?,
            version,
            stage,
            action,
            req.action_endpoint_id.as_deref().filter(|v| !v.is_empty()),
            Some(&user),
        )
        .await?;
    proto_response(
        &pb::create_gateway_guardrail::Response {
            guardrail: Some(guardrail_proto(guardrail)),
        },
        "mlflow.CreateGatewayGuardrail.Response",
    )
}

pub async fn get_guardrail(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::GetGatewayGuardrail,
        "mlflow.GetGatewayGuardrail"
    );
    let guardrail = state
        .tracking_store()
        .get_gateway_guardrail(
            workspace.name(),
            required(req.guardrail_id.as_deref(), "guardrail_id")?,
        )
        .await?;
    proto_response(
        &pb::get_gateway_guardrail::Response {
            guardrail: Some(guardrail_proto(guardrail)),
        },
        "mlflow.GetGatewayGuardrail.Response",
    )
}

pub async fn delete_guardrail(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::DeleteGatewayGuardrail,
        "mlflow.DeleteGatewayGuardrail"
    );
    state
        .tracking_store()
        .delete_gateway_guardrail(
            workspace.name(),
            required(req.guardrail_id.as_deref(), "guardrail_id")?,
        )
        .await?;
    empty_response!(
        pb::delete_gateway_guardrail::Response,
        "mlflow.DeleteGatewayGuardrail.Response"
    )
}

pub async fn list_guardrails(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListGatewayGuardrails,
        "mlflow.ListGatewayGuardrails"
    );
    let page = state
        .tracking_store()
        .list_gateway_guardrails(
            workspace.name(),
            req.max_results.unwrap_or(1000),
            req.page_token.as_deref().filter(|v| !v.is_empty()),
        )
        .await?;
    proto_response(
        &pb::list_gateway_guardrails::Response {
            guardrails: page.guardrails.into_iter().map(guardrail_proto).collect(),
            next_page_token: page.next_page_token,
        },
        "mlflow.ListGatewayGuardrails.Response",
    )
}

pub async fn add_guardrail_to_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::AddGuardrailToEndpoint,
        "mlflow.AddGuardrailToEndpoint"
    );
    let user = current_user();
    let config = state
        .tracking_store()
        .add_guardrail_to_endpoint(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.guardrail_id.as_deref(), "guardrail_id")?,
            req.execution_order,
            Some(&user),
        )
        .await?;
    proto_response(
        &pb::add_guardrail_to_endpoint::Response {
            config: Some(guardrail_config_proto(config)),
        },
        "mlflow.AddGuardrailToEndpoint.Response",
    )
}

pub async fn remove_guardrail_from_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::RemoveGuardrailFromEndpoint,
        "mlflow.RemoveGuardrailFromEndpoint"
    );
    state
        .tracking_store()
        .remove_guardrail_from_endpoint(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.guardrail_id.as_deref(), "guardrail_id")?,
        )
        .await?;
    empty_response!(
        pb::remove_guardrail_from_endpoint::Response,
        "mlflow.RemoveGuardrailFromEndpoint.Response"
    )
}

pub async fn list_endpoint_guardrail_configs(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::ListEndpointGuardrailConfigs,
        "mlflow.ListEndpointGuardrailConfigs"
    );
    let configs = state
        .tracking_store()
        .list_endpoint_guardrail_configs(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
        )
        .await?;
    proto_response(
        &pb::list_endpoint_guardrail_configs::Response {
            configs: configs.into_iter().map(guardrail_config_proto).collect(),
        },
        "mlflow.ListEndpointGuardrailConfigs.Response",
    )
}

pub async fn update_endpoint_guardrail_config(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req = parse!(
        &parts,
        &body,
        pb::UpdateEndpointGuardrailConfig,
        "mlflow.UpdateEndpointGuardrailConfig"
    );
    let config = state
        .tracking_store()
        .update_endpoint_guardrail_config(
            workspace.name(),
            required(req.endpoint_id.as_deref(), "endpoint_id")?,
            required(req.guardrail_id.as_deref(), "guardrail_id")?,
            req.execution_order,
        )
        .await?;
    proto_response(
        &pb::update_endpoint_guardrail_config::Response {
            config: Some(guardrail_config_proto(config)),
        },
        "mlflow.UpdateEndpointGuardrailConfig.Response",
    )
}

fn required<'a>(value: Option<&'a str>, name: &str) -> Result<&'a str, MlflowError> {
    value
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing(name))
}

fn missing(name: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!("Missing value for required parameter '{name}'."))
}

fn required_enum(value: Option<i32>, name: &str) -> Result<i32, MlflowError> {
    value.ok_or_else(|| missing(name))
}

fn validate_endpoint_name(name: &str) -> Result<(), MlflowError> {
    if name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        Ok(())
    } else {
        Err(MlflowError::invalid_parameter_value(format!("Invalid endpoint name '{name}'. Name can only contain letters, numbers, underscores, hyphens, and dots.")))
    }
}

fn linkage_name(value: i32) -> Result<&'static str, MlflowError> {
    pb::GatewayModelLinkageType::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid linkage_type: {value}"))
        })
}

fn routing_name(value: i32) -> Result<String, MlflowError> {
    pb::RoutingStrategy::try_from(value)
        .ok()
        .map(|v| v.as_str_name().to_string())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid routing_strategy: {value}"))
        })
}

fn fallback_name(value: i32) -> Result<String, MlflowError> {
    pb::FallbackStrategy::try_from(value)
        .ok()
        .map(|v| v.as_str_name().to_string())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid fallback strategy: {value}"))
        })
}

fn budget_unit(value: i32) -> Result<&'static str, MlflowError> {
    let name = pb::BudgetUnit::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid budget_unit: {value}"))
        })?;
    (name != "BUDGET_UNIT_UNSPECIFIED")
        .then_some(name)
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid budget_unit: {value}"))
        })
}

fn duration_unit(value: i32) -> Result<&'static str, MlflowError> {
    let name = pb::BudgetDurationUnit::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid duration.unit: {value}"))
        })?;
    (name != "DURATION_UNIT_UNSPECIFIED")
        .then_some(name)
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid duration.unit: {value}"))
        })
}

fn target_scope(value: i32) -> Result<&'static str, MlflowError> {
    let name = pb::BudgetTargetScope::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid target_scope: {value}"))
        })?;
    (name != "TARGET_SCOPE_UNSPECIFIED")
        .then_some(name)
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid target_scope: {value}"))
        })
}

fn budget_action(value: i32) -> Result<&'static str, MlflowError> {
    let name = pb::BudgetAction::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid budget_action: {value}"))
        })?;
    (name != "BUDGET_ACTION_UNSPECIFIED")
        .then_some(name)
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!("Invalid budget_action: {value}"))
        })
}

fn guardrail_stage(value: i32) -> Result<&'static str, MlflowError> {
    let name = pb::GuardrailStage::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| MlflowError::invalid_parameter_value(format!("Invalid stage: {value}")))?;
    (name != "GUARDRAIL_STAGE_UNSPECIFIED")
        .then_some(name)
        .ok_or_else(|| MlflowError::invalid_parameter_value(format!("Invalid stage: {value}")))
}

fn guardrail_action(value: i32) -> Result<&'static str, MlflowError> {
    let name = pb::GuardrailAction::try_from(value)
        .ok()
        .map(|v| v.as_str_name())
        .ok_or_else(|| MlflowError::invalid_parameter_value(format!("Invalid action: {value}")))?;
    (name != "GUARDRAIL_ACTION_UNSPECIFIED")
        .then_some(name)
        .ok_or_else(|| MlflowError::invalid_parameter_value(format!("Invalid action: {value}")))
}

fn required_positive(value: Option<i32>) -> Result<i32, MlflowError> {
    let value = value.unwrap_or_default();
    (value > 0).then_some(value).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "duration.value must be a positive integer, got {value}"
        ))
    })
}

fn current_user() -> String {
    ["LOGNAME", "USER", "LNAME", "USERNAME"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "unknown".to_string())
}

fn config_from_proto(
    config: &pb::GatewayEndpointModelConfig,
) -> Result<EndpointModelConfig, MlflowError> {
    Ok(EndpointModelConfig {
        model_definition_id: required(
            config.model_definition_id.as_deref(),
            "model_definition_id",
        )?
        .to_string(),
        linkage_type: linkage_name(config.linkage_type.unwrap_or_default())?.to_string(),
        weight: f64::from(config.weight.unwrap_or_default()),
        fallback_order: config.fallback_order,
    })
}

fn fallback_from_proto(config: &pb::FallbackConfig) -> Result<FallbackConfig, MlflowError> {
    Ok(FallbackConfig {
        strategy: config.strategy.map(fallback_name).transpose()?,
        max_attempts: config.max_attempts,
    })
}

fn secret_proto(secret: GatewaySecretInfo) -> pb::GatewaySecretInfo {
    pb::GatewaySecretInfo {
        secret_id: Some(secret.secret_id),
        secret_name: Some(secret.secret_name),
        masked_values: secret.masked_values,
        created_at: Some(secret.created_at),
        last_updated_at: Some(secret.last_updated_at),
        provider: secret.provider,
        created_by: Some(secret.created_by.unwrap_or_default()),
        last_updated_by: Some(secret.last_updated_by.unwrap_or_default()),
        auth_config: secret.auth_config,
    }
}

fn model_proto(model: GatewayModelDefinition) -> pb::GatewayModelDefinition {
    pb::GatewayModelDefinition {
        model_definition_id: Some(model.model_definition_id),
        name: Some(model.name),
        secret_id: model.secret_id,
        secret_name: model.secret_name,
        provider: Some(model.provider),
        model_name: Some(model.model_name),
        created_at: Some(model.created_at),
        last_updated_at: Some(model.last_updated_at),
        created_by: model.created_by,
        last_updated_by: model.last_updated_by,
    }
}

fn mapping_proto(mapping: EndpointModelMapping) -> pb::GatewayEndpointModelMapping {
    pb::GatewayEndpointModelMapping {
        mapping_id: Some(mapping.mapping_id),
        endpoint_id: Some(mapping.endpoint_id),
        model_definition_id: Some(mapping.model_definition_id),
        model_definition: mapping.model_definition.map(model_proto),
        weight: Some(mapping.weight as f32),
        created_at: Some(mapping.created_at),
        created_by: mapping.created_by,
        linkage_type: pb::GatewayModelLinkageType::from_str_name(&mapping.linkage_type)
            .map(|value| value as i32),
        fallback_order: mapping.fallback_order,
    }
}

fn endpoint_proto(endpoint: Endpoint) -> pb::GatewayEndpoint {
    pb::GatewayEndpoint {
        endpoint_id: Some(endpoint.endpoint_id),
        name: Some(endpoint.name.unwrap_or_default()),
        created_at: Some(endpoint.created_at),
        last_updated_at: Some(endpoint.last_updated_at),
        model_mappings: endpoint
            .model_mappings
            .into_iter()
            .map(mapping_proto)
            .collect(),
        created_by: Some(endpoint.created_by.unwrap_or_default()),
        last_updated_by: Some(endpoint.last_updated_by.unwrap_or_default()),
        tags: endpoint
            .tags
            .into_iter()
            .map(|tag| pb::GatewayEndpointTag {
                key: Some(tag.key),
                value: tag.value,
            })
            .collect(),
        routing_strategy: endpoint
            .routing_strategy
            .as_deref()
            .and_then(pb::RoutingStrategy::from_str_name)
            .map(|value| value as i32),
        fallback_config: endpoint.fallback_config.map(|config| pb::FallbackConfig {
            strategy: config
                .strategy
                .as_deref()
                .and_then(pb::FallbackStrategy::from_str_name)
                .map(|value| value as i32),
            max_attempts: config.max_attempts,
        }),
        experiment_id: endpoint.experiment_id,
        usage_tracking: Some(endpoint.usage_tracking),
    }
}

fn binding_proto(binding: EndpointBinding) -> pb::GatewayEndpointBinding {
    pb::GatewayEndpointBinding {
        endpoint_id: Some(binding.endpoint_id),
        resource_type: Some(binding.resource_type),
        resource_id: Some(binding.resource_id),
        created_at: Some(binding.created_at),
        last_updated_at: Some(binding.last_updated_at),
        created_by: binding.created_by,
        last_updated_by: binding.last_updated_by,
        display_name: binding.display_name,
    }
}

fn budget_proto(policy: BudgetPolicy) -> pb::GatewayBudgetPolicy {
    pb::GatewayBudgetPolicy {
        budget_policy_id: Some(policy.budget_policy_id),
        budget_unit: pb::BudgetUnit::from_str_name(&policy.budget_unit).map(|value| value as i32),
        budget_amount: Some(policy.budget_amount),
        duration: Some(pb::BudgetDuration {
            unit: pb::BudgetDurationUnit::from_str_name(&policy.duration_unit)
                .map(|value| value as i32),
            value: Some(policy.duration_value),
        }),
        target_scope: pb::BudgetTargetScope::from_str_name(&policy.target_scope)
            .map(|value| value as i32),
        budget_action: pb::BudgetAction::from_str_name(&policy.budget_action)
            .map(|value| value as i32),
        created_by: Some(policy.created_by.unwrap_or_default()),
        created_at: Some(policy.created_at),
        last_updated_by: Some(policy.last_updated_by.unwrap_or_default()),
        last_updated_at: Some(policy.last_updated_at),
    }
}

fn scorer_proto(scorer: ScorerVersion) -> pb::Scorer {
    pb::Scorer {
        experiment_id: scorer.experiment_id.parse().ok(),
        scorer_name: Some(scorer.scorer_name),
        scorer_version: Some(scorer.scorer_version),
        serialized_scorer: Some(scorer.serialized_scorer),
        creation_time: scorer.creation_time,
        scorer_id: Some(scorer.scorer_id),
    }
}

fn guardrail_proto(guardrail: GatewayGuardrail) -> pb::GatewayGuardrail {
    pb::GatewayGuardrail {
        guardrail_id: Some(guardrail.guardrail_id),
        name: Some(guardrail.name),
        scorer: Some(scorer_proto(guardrail.scorer)),
        stage: pb::GuardrailStage::from_str_name(&guardrail.stage).map(|value| value as i32),
        action: pb::GuardrailAction::from_str_name(&guardrail.action).map(|value| value as i32),
        action_endpoint_id: guardrail.action_endpoint_name,
        created_by: Some(guardrail.created_by.unwrap_or_default()),
        created_at: Some(guardrail.created_at),
        last_updated_by: Some(guardrail.last_updated_by.unwrap_or_default()),
        last_updated_at: Some(guardrail.last_updated_at),
    }
}

fn guardrail_config_proto(config: GatewayGuardrailConfig) -> pb::GatewayGuardrailConfig {
    pb::GatewayGuardrailConfig {
        endpoint_id: Some(config.endpoint_id),
        guardrail_id: Some(config.guardrail_id),
        execution_order: config.execution_order,
        created_by: Some(config.created_by.unwrap_or_default()),
        created_at: Some(config.created_at),
        guardrail: config.guardrail.map(guardrail_proto),
    }
}
