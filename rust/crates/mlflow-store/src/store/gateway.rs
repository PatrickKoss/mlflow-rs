//! Workspace-scoped CRUD for the nine shared AI Gateway tables.

use std::collections::HashMap;

use chrono::{Datelike, TimeZone, Utc};
use mlflow_error::MlflowError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::secrets::{decrypt_secret, encrypt_secret, mask_secret_value, Kek};

use super::dbutil::{RowLike, Val};
use super::evaluation_datasets::python_json_dumps;
use super::experiments::{internal, is_unique_violation, now_millis};
use super::scorers::ScorerVersion;
use super::search::{SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD};
use super::TrackingStore;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewaySecretInfo {
    pub secret_id: String,
    pub secret_name: String,
    pub masked_values: HashMap<String, String>,
    pub created_at: i64,
    pub last_updated_at: i64,
    pub provider: Option<String>,
    pub auth_config: HashMap<String, String>,
    pub workspace: String,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayModelDefinition {
    pub model_definition_id: String,
    pub name: String,
    pub secret_id: Option<String>,
    pub secret_name: Option<String>,
    pub provider: String,
    pub model_name: String,
    pub created_at: i64,
    pub last_updated_at: i64,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    pub workspace: String,
}

/// Privileged, invocation-ready model configuration. Unlike the public CRUD
/// entities this contains the decrypted secret and must never be serialized on
/// an HTTP response or logged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedGatewayModelConfig {
    pub model_definition_id: String,
    pub provider: String,
    pub model_name: String,
    pub secret_value: Value,
    pub auth_config: HashMap<String, String>,
    pub weight: f64,
    pub linkage_type: String,
    pub fallback_order: Option<i32>,
}

/// Python `GatewayEndpointConfig` equivalent used by the native runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedGatewayEndpointConfig {
    pub endpoint_id: String,
    pub endpoint_name: String,
    pub models: Vec<ResolvedGatewayModelConfig>,
    pub routing_strategy: Option<String>,
    pub fallback_config: Option<ResolvedGatewayFallbackConfig>,
    pub experiment_id: Option<String>,
    pub usage_tracking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedGatewayFallbackConfig {
    pub strategy: Option<String>,
    pub max_attempts: Option<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointModelConfig {
    pub model_definition_id: String,
    pub linkage_type: String,
    pub weight: f64,
    pub fallback_order: Option<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointModelMapping {
    pub mapping_id: String,
    pub endpoint_id: String,
    pub model_definition_id: String,
    pub model_definition: Option<GatewayModelDefinition>,
    pub weight: f64,
    pub linkage_type: String,
    pub fallback_order: Option<i32>,
    pub created_at: i64,
    pub created_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackConfig {
    pub strategy: Option<String>,
    pub max_attempts: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointTag {
    pub key: String,
    pub value: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Endpoint {
    pub endpoint_id: String,
    pub name: Option<String>,
    pub created_at: i64,
    pub last_updated_at: i64,
    pub model_mappings: Vec<EndpointModelMapping>,
    pub tags: Vec<EndpointTag>,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    pub routing_strategy: Option<String>,
    pub fallback_config: Option<FallbackConfig>,
    pub experiment_id: Option<String>,
    pub usage_tracking: bool,
    pub workspace: String,
}

#[derive(Debug, Clone, Default)]
pub struct EndpointUpdate<'a> {
    pub name: Option<&'a str>,
    pub updated_by: Option<&'a str>,
    pub routing_strategy: Option<&'a str>,
    pub fallback_config: Option<&'a FallbackConfig>,
    pub model_configs: Option<&'a [EndpointModelConfig]>,
    pub experiment_id: Option<&'a str>,
    pub usage_tracking: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointBinding {
    pub endpoint_id: String,
    pub resource_type: String,
    pub resource_id: String,
    pub created_at: i64,
    pub last_updated_at: i64,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BudgetPolicy {
    pub budget_policy_id: String,
    pub budget_unit: String,
    pub budget_amount: f64,
    pub duration_unit: String,
    pub duration_value: i32,
    pub target_scope: String,
    pub budget_action: String,
    pub created_by: Option<String>,
    pub created_at: i64,
    pub last_updated_by: Option<String>,
    pub last_updated_at: i64,
    pub workspace: String,
}

#[derive(Debug, Clone, Default)]
pub struct BudgetPolicyUpdate<'a> {
    pub budget_unit: Option<&'a str>,
    pub budget_amount: Option<f64>,
    pub duration: Option<(&'a str, i32)>,
    pub target_scope: Option<&'a str>,
    pub budget_action: Option<&'a str>,
    pub updated_by: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BudgetPoliciesPage {
    pub policies: Vec<BudgetPolicy>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BudgetWindow {
    pub budget_policy_id: String,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub current_spend: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayGuardrail {
    pub guardrail_id: String,
    pub name: String,
    pub scorer: ScorerVersion,
    pub stage: String,
    pub action: String,
    /// Python's entity names this `action_endpoint_name` and serializes the
    /// endpoint name into the proto field called `action_endpoint_id`.
    pub action_endpoint_name: Option<String>,
    pub created_by: Option<String>,
    pub created_at: i64,
    pub last_updated_by: Option<String>,
    pub last_updated_at: i64,
    pub workspace: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GatewayGuardrailConfig {
    pub endpoint_id: String,
    pub guardrail_id: String,
    pub execution_order: Option<i64>,
    pub created_by: Option<String>,
    pub created_at: i64,
    pub guardrail: Option<GatewayGuardrail>,
    pub workspace: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GuardrailsPage {
    pub guardrails: Vec<GatewayGuardrail>,
    pub next_page_token: Option<String>,
}

fn uuid_id(prefix: &str) -> String {
    format!("{prefix}{}", Uuid::new_v4().simple())
}

fn exact_one(
    first_name: &str,
    first: Option<&str>,
    second_name: &str,
    second: Option<&str>,
) -> Result<(), MlflowError> {
    if first.is_some() == second.is_some() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Exactly one of {first_name} or {second_name} must be provided"
        )));
    }
    Ok(())
}

fn json_string_map(value: Option<&str>) -> HashMap<String, String> {
    value
        .and_then(|value| serde_json::from_str(value).ok())
        .unwrap_or_default()
}

fn string_map_json(value: &HashMap<String, String>) -> Option<String> {
    (!value.is_empty()).then(|| {
        let object = value
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect::<Map<_, _>>();
        python_json_dumps(&Value::Object(object), false)
    })
}

fn secret_not_found(field: &str, value: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("GatewaySecret not found ({field}='{value}')"))
}

fn model_not_found(field: &str, value: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!(
        "GatewayModelDefinition not found ({field}='{value}')"
    ))
}

fn endpoint_not_found(field: &str, value: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("GatewayEndpoint not found ({field}='{value}')"))
}

impl TrackingStore {
    pub async fn create_gateway_secret(
        &self,
        workspace: &str,
        secret_name: &str,
        secret_value: &HashMap<String, String>,
        provider: Option<&str>,
        auth_config: &HashMap<String, String>,
        created_by: Option<&str>,
    ) -> Result<GatewaySecretInfo, MlflowError> {
        let secret_id = uuid_id("s-");
        let now = now_millis();
        let value = Value::Object(
            secret_value
                .iter()
                .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                .collect(),
        );
        let plaintext = python_json_dumps(&value, false);
        let masked = mask_secret_value(value.as_object().expect("constructed object"));
        let masked_json = python_json_dumps(&Value::Object(masked), false);
        let kek = Kek::from_environment().map_err(|error| {
            MlflowError::invalid_parameter_value(format!("Failed to initialize KEK: {error}"))
        })?;
        let encrypted = encrypt_secret(plaintext.as_bytes(), &kek, &secret_id, secret_name);
        let dialect = self.db().dialect();
        let placeholders = (1..=14)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        let result = self
            .db()
            .exec(
                &format!(
                    "INSERT INTO secrets (secret_id, secret_name, encrypted_value, wrapped_dek, \
                     kek_version, masked_value, provider, auth_config, description, created_by, \
                     created_at, last_updated_by, last_updated_at, workspace) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(secret_id.clone()),
                    Val::Text(secret_name.to_string()),
                    Val::Bytes(encrypted.encrypted_value),
                    Val::Bytes(encrypted.wrapped_dek),
                    Val::Int(i64::from(encrypted.kek_version)),
                    Val::Text(masked_json),
                    Val::OptText(provider.map(str::to_string)),
                    Val::OptText(string_map_json(auth_config)),
                    Val::OptText(None),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await;
        if let Err(error) = result {
            if is_unique_violation(&error) {
                return Err(MlflowError::resource_already_exists(format!(
                    "Secret with name '{secret_name}' already exists"
                )));
            }
            return Err(internal(error));
        }
        self.get_gateway_secret_info(workspace, Some(&secret_id), None)
            .await
    }

    pub async fn get_gateway_secret_info(
        &self,
        workspace: &str,
        secret_id: Option<&str>,
        secret_name: Option<&str>,
    ) -> Result<GatewaySecretInfo, MlflowError> {
        exact_one("secret_id", secret_id, "secret_name", secret_name)?;
        let (field, value) = secret_id
            .map(|value| ("secret_id", value))
            .unwrap_or_else(|| ("secret_name", secret_name.expect("validated")));
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT secret_id, secret_name, masked_value, provider, auth_config, \
                     created_at, last_updated_at, created_by, last_updated_by, workspace FROM \
                     secrets WHERE {field} = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(value.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_secret_info,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| secret_not_found(field, value))
    }

    pub async fn update_gateway_secret(
        &self,
        workspace: &str,
        secret_id: &str,
        secret_value: Option<&HashMap<String, String>>,
        auth_config: Option<&HashMap<String, String>>,
        updated_by: Option<&str>,
    ) -> Result<GatewaySecretInfo, MlflowError> {
        let current = self
            .get_gateway_secret_info(workspace, Some(secret_id), None)
            .await?;
        let dialect = self.db().dialect();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let Some(secret_value) = secret_value {
            let value = Value::Object(
                secret_value
                    .iter()
                    .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                    .collect(),
            );
            let plaintext = python_json_dumps(&value, false);
            let masked = python_json_dumps(
                &Value::Object(mask_secret_value(
                    value.as_object().expect("constructed object"),
                )),
                false,
            );
            let kek = Kek::from_environment().map_err(|error| {
                MlflowError::invalid_parameter_value(format!("Failed to initialize KEK: {error}"))
            })?;
            let encrypted = encrypt_secret(
                plaintext.as_bytes(),
                &kek,
                &current.secret_id,
                &current.secret_name,
            );
            for (column, value) in [
                ("encrypted_value", Val::Bytes(encrypted.encrypted_value)),
                ("wrapped_dek", Val::Bytes(encrypted.wrapped_dek)),
                ("kek_version", Val::Int(i64::from(encrypted.kek_version))),
                ("masked_value", Val::Text(masked)),
            ] {
                values.push(value);
                assignments.push(format!("{column} = {}", dialect.placeholder(values.len())));
            }
        }
        if let Some(auth_config) = auth_config {
            values.push(Val::OptText(string_map_json(auth_config)));
            assignments.push(format!(
                "auth_config = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::OptText(updated_by.map(str::to_string)));
        assignments.push(format!(
            "last_updated_by = {}",
            dialect.placeholder(values.len())
        ));
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_updated_at = {}",
            dialect.placeholder(values.len())
        ));
        values.push(Val::Text(secret_id.to_string()));
        let id_placeholder = dialect.placeholder(values.len());
        values.push(Val::Text(workspace.to_string()));
        let workspace_placeholder = dialect.placeholder(values.len());
        self.db()
            .exec(
                &format!(
                    "UPDATE secrets SET {} WHERE secret_id = {id_placeholder} AND workspace = \
                     {workspace_placeholder}",
                    assignments.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        self.get_gateway_secret_info(workspace, Some(secret_id), None)
            .await
    }

    pub async fn delete_gateway_secret(
        &self,
        workspace: &str,
        secret_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_gateway_secret_info(workspace, Some(secret_id), None)
            .await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM secrets WHERE secret_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(secret_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        Ok(())
    }

    pub async fn list_gateway_secret_infos(
        &self,
        workspace: &str,
        provider: Option<&str>,
    ) -> Result<Vec<GatewaySecretInfo>, MlflowError> {
        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(workspace.to_string())];
        let mut filter = format!("workspace = {}", dialect.placeholder(1));
        if let Some(provider) = provider {
            values.push(Val::Text(provider.to_string()));
            filter.push_str(&format!(" AND provider = {}", dialect.placeholder(2)));
        }
        self.db()
            .fetch_all(
                &format!(
                    "SELECT secret_id, secret_name, masked_value, provider, auth_config, \
                     created_at, last_updated_at, created_by, last_updated_by, workspace FROM \
                     secrets WHERE {filter}"
                ),
                &values,
                map_secret_info,
            )
            .await
            .map_err(internal)
    }

    /// Privileged runtime read. The cache contains only encrypted values and
    /// all gateway mutations clear it, matching Python's full-cache invalidation.
    pub async fn get_decrypted_gateway_secret(
        &self,
        workspace: &str,
        secret_id: &str,
    ) -> Result<Value, MlflowError> {
        let cache_key = format!("secret:{workspace}:{secret_id}");
        if let Some(value) = self.secret_cache()?.get(&cache_key) {
            return Ok(value);
        }
        let dialect = self.db().dialect();
        let row = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT secret_id, secret_name, encrypted_value, wrapped_dek, kek_version FROM \
                     secrets WHERE secret_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(secret_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| {
                    Ok((
                        row.get_string("secret_id")?,
                        row.get_string("secret_name")?,
                        row.get_bytes("encrypted_value")?,
                        row.get_bytes("wrapped_dek")?,
                        row.get_int("kek_version")?,
                    ))
                },
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| secret_not_found("secret_id", secret_id))?;
        let version = u32::try_from(row.4).map_err(|_| {
            MlflowError::invalid_parameter_value("Failed to decrypt secret. Check KEK passphrase, secret metadata, or database integrity.")
        })?;
        let kek = Kek::for_stored_version(version).map_err(|_| {
            MlflowError::invalid_parameter_value("Failed to decrypt secret. Check KEK passphrase, secret metadata, or database integrity.")
        })?;
        let plaintext = decrypt_secret(&row.2, &row.3, &kek, &row.0, &row.1).map_err(|_| {
            MlflowError::invalid_parameter_value("Failed to decrypt secret. Check KEK passphrase, secret metadata, or database integrity.")
        })?;
        let text = String::from_utf8(plaintext).map_err(|_| {
            MlflowError::invalid_parameter_value("Failed to decrypt secret. Check KEK passphrase, secret metadata, or database integrity.")
        })?;
        let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        self.secret_cache()?.set(&cache_key, &value);
        Ok(value)
    }

    pub async fn create_gateway_model_definition(
        &self,
        workspace: &str,
        name: &str,
        secret_id: &str,
        provider: &str,
        model_name: &str,
        created_by: Option<&str>,
    ) -> Result<GatewayModelDefinition, MlflowError> {
        self.get_gateway_secret_info(workspace, Some(secret_id), None)
            .await?;
        let model_definition_id = uuid_id("d-");
        let now = now_millis();
        let dialect = self.db().dialect();
        let placeholders = (1..=10)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        let result = self
            .db()
            .exec(
                &format!(
                    "INSERT INTO model_definitions (model_definition_id, name, secret_id, \
                     provider, model_name, created_by, created_at, last_updated_by, \
                     last_updated_at, workspace) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(model_definition_id.clone()),
                    Val::Text(name.to_string()),
                    Val::Text(secret_id.to_string()),
                    Val::Text(provider.to_string()),
                    Val::Text(model_name.to_string()),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await;
        if let Err(error) = result {
            if is_unique_violation(&error) {
                return Err(MlflowError::resource_already_exists(format!(
                    "Model definition with name '{name}' already exists"
                )));
            }
            return Err(internal(error));
        }
        self.get_gateway_model_definition(workspace, Some(&model_definition_id), None)
            .await
    }

    pub async fn get_gateway_model_definition(
        &self,
        workspace: &str,
        model_definition_id: Option<&str>,
        name: Option<&str>,
    ) -> Result<GatewayModelDefinition, MlflowError> {
        exact_one("model_definition_id", model_definition_id, "name", name)?;
        let (field, value) = model_definition_id
            .map(|value| ("model_definition_id", value))
            .unwrap_or_else(|| ("name", name.expect("validated")));
        self.find_gateway_model_definition(workspace, field, value)
            .await?
            .ok_or_else(|| model_not_found(field, value))
    }

    async fn find_gateway_model_definition(
        &self,
        workspace: &str,
        field: &str,
        value: &str,
    ) -> Result<Option<GatewayModelDefinition>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT d.model_definition_id, d.name, d.secret_id, s.secret_name, \
                     d.provider, d.model_name, d.created_at, d.last_updated_at, d.created_by, \
                     d.last_updated_by, d.workspace FROM model_definitions d LEFT JOIN secrets s \
                     ON s.secret_id = d.secret_id WHERE d.{field} = {} AND d.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(value.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_model_definition,
            )
            .await
            .map_err(internal)
    }

    pub async fn list_gateway_model_definitions(
        &self,
        workspace: &str,
        provider: Option<&str>,
        secret_id: Option<&str>,
    ) -> Result<Vec<GatewayModelDefinition>, MlflowError> {
        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(workspace.to_string())];
        let mut filters = vec![format!("d.workspace = {}", dialect.placeholder(1))];
        if let Some(provider) = provider {
            values.push(Val::Text(provider.to_string()));
            filters.push(format!(
                "d.provider = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(secret_id) = secret_id {
            values.push(Val::Text(secret_id.to_string()));
            filters.push(format!(
                "d.secret_id = {}",
                dialect.placeholder(values.len())
            ));
        }
        self.db()
            .fetch_all(
                &format!(
                    "SELECT d.model_definition_id, d.name, d.secret_id, s.secret_name, \
                     d.provider, d.model_name, d.created_at, d.last_updated_at, d.created_by, \
                     d.last_updated_by, d.workspace FROM model_definitions d LEFT JOIN secrets s \
                     ON s.secret_id = d.secret_id WHERE {}",
                    filters.join(" AND ")
                ),
                &values,
                map_model_definition,
            )
            .await
            .map_err(internal)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn update_gateway_model_definition(
        &self,
        workspace: &str,
        model_definition_id: &str,
        name: Option<&str>,
        secret_id: Option<&str>,
        model_name: Option<&str>,
        updated_by: Option<&str>,
        provider: Option<&str>,
    ) -> Result<GatewayModelDefinition, MlflowError> {
        self.get_gateway_model_definition(workspace, Some(model_definition_id), None)
            .await?;
        if let Some(secret_id) = secret_id {
            self.get_gateway_secret_info(workspace, Some(secret_id), None)
                .await?;
        }
        let dialect = self.db().dialect();
        let mut values = Vec::new();
        let mut assignments = Vec::new();
        for (column, value) in [
            ("name", name),
            ("secret_id", secret_id),
            ("model_name", model_name),
            ("provider", provider),
        ] {
            if let Some(value) = value {
                values.push(Val::Text(value.to_string()));
                assignments.push(format!("{column} = {}", dialect.placeholder(values.len())));
            }
        }
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_updated_at = {}",
            dialect.placeholder(values.len())
        ));
        if let Some(updated_by) = updated_by {
            values.push(Val::Text(updated_by.to_string()));
            assignments.push(format!(
                "last_updated_by = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Text(model_definition_id.to_string()));
        let id_placeholder = dialect.placeholder(values.len());
        values.push(Val::Text(workspace.to_string()));
        let workspace_placeholder = dialect.placeholder(values.len());
        let result = self
            .db()
            .exec(
                &format!(
                    "UPDATE model_definitions SET {} WHERE model_definition_id = \
                     {id_placeholder} AND workspace = {workspace_placeholder}",
                    assignments.join(", ")
                ),
                &values,
            )
            .await;
        if let Err(error) = result {
            if is_unique_violation(&error) {
                return Err(MlflowError::resource_already_exists(format!(
                    "Model definition with name '{}' already exists",
                    name.unwrap_or("")
                )));
            }
            return Err(internal(error));
        }
        self.invalidate_secret_cache();
        self.get_gateway_model_definition(workspace, Some(model_definition_id), None)
            .await
    }

    pub async fn delete_gateway_model_definition(
        &self,
        workspace: &str,
        model_definition_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_gateway_model_definition(workspace, Some(model_definition_id), None)
            .await?;
        let dialect = self.db().dialect();
        let result = self
            .db()
            .exec(
                &format!(
                    "DELETE FROM model_definitions WHERE model_definition_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(model_definition_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await;
        match result {
            Ok(_) => {
                self.invalidate_secret_cache();
                Ok(())
            }
            Err(error) if is_foreign_key_violation(&error) => Err(MlflowError::invalid_state(
                "Cannot delete model definition that is currently in use by endpoints. Detach it from all endpoints first.",
            )),
            Err(error) => Err(internal(error)),
        }
    }
}

impl TrackingStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn create_gateway_endpoint(
        &self,
        workspace: &str,
        name: &str,
        model_configs: &[EndpointModelConfig],
        created_by: Option<&str>,
        routing_strategy: Option<&str>,
        fallback_config: Option<&FallbackConfig>,
        experiment_id: Option<&str>,
        usage_tracking: bool,
    ) -> Result<Endpoint, MlflowError> {
        if model_configs.is_empty() {
            return Err(MlflowError::invalid_parameter_value(
                "Endpoint must have at least one model configuration",
            ));
        }
        self.validate_model_configs(workspace, model_configs)
            .await?;
        let endpoint_id = uuid_id("e-");
        let resolved_experiment = if usage_tracking && experiment_id.is_none() {
            Some(
                self.gateway_experiment_id(workspace, name, &endpoint_id)
                    .await?,
            )
        } else {
            experiment_id.map(str::to_string)
        };
        let fallback_json = build_fallback_json(fallback_config, model_configs, false);
        let now = now_millis();
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let placeholders = (1..=11)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        tx.exec(
            &format!(
                "INSERT INTO endpoints (endpoint_id, name, created_by, created_at, \
                 last_updated_by, last_updated_at, routing_strategy, fallback_config_json, \
                 experiment_id, usage_tracking, workspace) VALUES ({})",
                placeholders.join(", ")
            ),
            &[
                Val::Text(endpoint_id.clone()),
                Val::Text(name.to_string()),
                Val::OptText(created_by.map(str::to_string)),
                Val::Int(now),
                Val::OptText(created_by.map(str::to_string)),
                Val::Int(now),
                Val::OptText(routing_strategy.map(str::to_string)),
                Val::OptText(fallback_json),
                Val::OptInt(
                    resolved_experiment
                        .as_deref()
                        .map(parse_experiment_id)
                        .transpose()?,
                ),
                Val::Bool(usage_tracking),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        for config in model_configs {
            insert_mapping(&mut tx, dialect, &endpoint_id, config, created_by, now).await?;
        }
        tx.commit().await.map_err(internal)?;
        self.get_gateway_endpoint(workspace, Some(&endpoint_id), None)
            .await
    }

    pub async fn get_gateway_endpoint(
        &self,
        workspace: &str,
        endpoint_id: Option<&str>,
        name: Option<&str>,
    ) -> Result<Endpoint, MlflowError> {
        exact_one("endpoint_id", endpoint_id, "name", name)?;
        let (field, value) = endpoint_id
            .map(|value| ("endpoint_id", value))
            .unwrap_or_else(|| ("name", name.expect("validated")));
        let dialect = self.db().dialect();
        let root = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT endpoint_id, name, created_by, created_at, last_updated_by, \
                     last_updated_at, routing_strategy, fallback_config_json, experiment_id, \
                     usage_tracking, workspace FROM endpoints WHERE {field} = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[Val::Text(value.to_string()), Val::Text(workspace.to_string())],
                map_endpoint_root,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| endpoint_not_found(field, value))?;
        self.populate_endpoint(root).await
    }

    /// Resolve endpoint -> mappings -> model definitions -> decrypted secrets.
    ///
    /// The cache key and full-cache mutation invalidation deliberately match
    /// Python's `get_endpoint_config`: resolved plaintext is handed to the
    /// existing encrypted `SecretCache`, never a second plaintext cache.
    pub async fn get_resolved_gateway_endpoint_config(
        &self,
        workspace: &str,
        endpoint_name: &str,
    ) -> Result<ResolvedGatewayEndpointConfig, MlflowError> {
        let cache_key = format!("endpoint_config:{workspace}:{endpoint_name}");
        if let Some(value) = self.secret_cache()?.get(&cache_key) {
            return serde_json::from_value(value).map_err(|error| {
                MlflowError::internal_error(format!(
                    "Failed to deserialize cached gateway endpoint configuration: {error}"
                ))
            });
        }

        let endpoint = self
            .get_gateway_endpoint(workspace, None, Some(endpoint_name))
            .await?;
        let mut models = Vec::with_capacity(endpoint.model_mappings.len());
        for mapping in &endpoint.model_mappings {
            let definition = match &mapping.model_definition {
                Some(definition) => definition.clone(),
                None => {
                    self.get_gateway_model_definition(
                        workspace,
                        Some(&mapping.model_definition_id),
                        None,
                    )
                    .await?
                }
            };
            let Some(secret_id) = definition.secret_id.as_deref() else {
                continue;
            };
            let secret = self
                .get_gateway_secret_info(workspace, Some(secret_id), None)
                .await?;
            models.push(ResolvedGatewayModelConfig {
                model_definition_id: definition.model_definition_id,
                provider: definition.provider,
                model_name: definition.model_name,
                secret_value: self
                    .get_decrypted_gateway_secret(workspace, secret_id)
                    .await?,
                auth_config: secret.auth_config,
                weight: mapping.weight,
                linkage_type: mapping.linkage_type.clone(),
                fallback_order: mapping.fallback_order,
            });
        }

        let result = ResolvedGatewayEndpointConfig {
            endpoint_id: endpoint.endpoint_id,
            endpoint_name: endpoint.name.unwrap_or_else(|| endpoint_name.to_string()),
            models,
            routing_strategy: endpoint.routing_strategy,
            fallback_config: endpoint
                .fallback_config
                .map(|config| ResolvedGatewayFallbackConfig {
                    strategy: config.strategy,
                    max_attempts: config.max_attempts,
                }),
            experiment_id: endpoint.experiment_id.clone(),
            usage_tracking: endpoint.usage_tracking && endpoint.experiment_id.is_some(),
        };
        let value = serde_json::to_value(&result).map_err(|error| {
            MlflowError::internal_error(format!(
                "Failed to serialize gateway endpoint configuration: {error}"
            ))
        })?;
        self.secret_cache()?.set(&cache_key, &value);
        Ok(result)
    }

    /// Resolve every endpoint bound to a resource. Python does not cache this
    /// binding lookup, but each endpoint reached through it uses the same
    /// encrypted endpoint-config cache as direct invocation resolution.
    pub async fn get_resolved_gateway_resource_endpoint_configs(
        &self,
        workspace: &str,
        resource_type: &str,
        resource_id: &str,
    ) -> Result<Vec<ResolvedGatewayEndpointConfig>, MlflowError> {
        let dialect = self.db().dialect();
        let endpoint_names = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT e.name FROM endpoint_bindings b JOIN endpoints e ON e.endpoint_id = \
                     b.endpoint_id WHERE b.resource_type = {} AND b.resource_id = {} AND \
                     e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("name"),
            )
            .await
            .map_err(internal)?;
        let mut configs = Vec::with_capacity(endpoint_names.len());
        for endpoint_name in endpoint_names {
            configs.push(
                self.get_resolved_gateway_endpoint_config(workspace, &endpoint_name)
                    .await?,
            );
        }
        Ok(configs)
    }

    pub async fn list_gateway_endpoints(
        &self,
        workspace: &str,
        provider: Option<&str>,
        secret_id: Option<&str>,
    ) -> Result<Vec<Endpoint>, MlflowError> {
        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(workspace.to_string())];
        let mut filters = vec![format!("e.workspace = {}", dialect.placeholder(1))];
        let needs_definition = provider.is_some() || secret_id.is_some();
        if let Some(provider) = provider {
            values.push(Val::Text(provider.to_string()));
            filters.push(format!(
                "d.provider = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(secret_id) = secret_id {
            values.push(Val::Text(secret_id.to_string()));
            filters.push(format!(
                "d.secret_id = {}",
                dialect.placeholder(values.len())
            ));
        }
        let definition_join = if needs_definition {
            " JOIN model_definitions d ON d.model_definition_id = m.model_definition_id"
        } else {
            ""
        };
        let roots = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT DISTINCT e.endpoint_id, e.name, e.created_by, e.created_at, \
                     e.last_updated_by, e.last_updated_at, e.routing_strategy, \
                     e.fallback_config_json, e.experiment_id, e.usage_tracking, e.workspace FROM \
                     endpoints e JOIN endpoint_model_mappings m ON m.endpoint_id = e.endpoint_id \
                     {definition_join} WHERE {}",
                    filters.join(" AND ")
                ),
                &values,
                map_endpoint_root,
            )
            .await
            .map_err(internal)?;
        let mut endpoints = Vec::with_capacity(roots.len());
        for root in roots {
            endpoints.push(self.populate_endpoint(root).await?);
        }
        Ok(endpoints)
    }

    pub async fn update_gateway_endpoint(
        &self,
        workspace: &str,
        endpoint_id: &str,
        update: EndpointUpdate<'_>,
    ) -> Result<Endpoint, MlflowError> {
        let current = self
            .get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        if let Some(configs) = update.model_configs {
            self.validate_model_configs(workspace, configs).await?;
        }
        let mut experiment_id = update.experiment_id.map(str::to_string);
        if update.usage_tracking == Some(true)
            && experiment_id.is_none()
            && current.experiment_id.is_none()
        {
            let endpoint_name = update.name.or(current.name.as_deref()).unwrap_or("");
            experiment_id = Some(
                self.gateway_experiment_id(workspace, endpoint_name, endpoint_id)
                    .await?,
            );
        }
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let Some(name) = update.name {
            values.push(Val::Text(name.to_string()));
            assignments.push(format!("name = {}", dialect.placeholder(values.len())));
        }
        if let Some(usage_tracking) = update.usage_tracking {
            values.push(Val::Bool(usage_tracking));
            assignments.push(format!(
                "usage_tracking = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(experiment_id) = experiment_id.as_deref() {
            values.push(Val::Int(parse_experiment_id(experiment_id)?));
            assignments.push(format!(
                "experiment_id = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(routing_strategy) = update.routing_strategy {
            values.push(Val::Text(routing_strategy.to_string()));
            assignments.push(format!(
                "routing_strategy = {}",
                dialect.placeholder(values.len())
            ));
        }
        let fallback_json = if let Some(configs) = update.model_configs {
            Some(build_fallback_json(update.fallback_config, configs, true))
        } else if let Some(fallback) = update.fallback_config {
            let ids = fallback_ids_from_json(current_fallback_json(&current));
            Some(Some(fallback_json(fallback, &ids)))
        } else {
            None
        };
        if let Some(fallback_json) = fallback_json {
            values.push(Val::OptText(fallback_json));
            assignments.push(format!(
                "fallback_config_json = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_updated_at = {}",
            dialect.placeholder(values.len())
        ));
        if let Some(updated_by) = update.updated_by {
            values.push(Val::Text(updated_by.to_string()));
            assignments.push(format!(
                "last_updated_by = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Text(endpoint_id.to_string()));
        let id_placeholder = dialect.placeholder(values.len());
        values.push(Val::Text(workspace.to_string()));
        let workspace_placeholder = dialect.placeholder(values.len());
        tx.exec(
            &format!(
                "UPDATE endpoints SET {} WHERE endpoint_id = {id_placeholder} AND workspace = \
                 {workspace_placeholder}",
                assignments.join(", ")
            ),
            &values,
        )
        .await
        .map_err(internal)?;
        if let Some(configs) = update.model_configs {
            tx.exec(
                &format!(
                    "DELETE FROM endpoint_model_mappings WHERE endpoint_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(endpoint_id.to_string())],
            )
            .await
            .map_err(internal)?;
            let now = now_millis();
            for config in configs {
                insert_mapping(
                    &mut tx,
                    dialect,
                    endpoint_id,
                    config,
                    update.updated_by,
                    now,
                )
                .await?;
            }
        }
        tx.commit().await.map_err(internal)?;
        self.invalidate_secret_cache();
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await
    }

    pub async fn delete_gateway_endpoint(
        &self,
        workspace: &str,
        endpoint_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM endpoints WHERE endpoint_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        Ok(())
    }

    pub async fn attach_model_to_gateway_endpoint(
        &self,
        workspace: &str,
        endpoint_id: &str,
        model_config: &EndpointModelConfig,
        created_by: Option<&str>,
    ) -> Result<EndpointModelMapping, MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        self.get_gateway_model_definition(workspace, Some(&model_config.model_definition_id), None)
            .await?;
        let now = now_millis();
        let mapping_id = uuid_id("m-");
        let dialect = self.db().dialect();
        let placeholders = (1..=8)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        let result = self
            .db()
            .exec(
                &format!(
                    "INSERT INTO endpoint_model_mappings (mapping_id, endpoint_id, \
                     model_definition_id, weight, linkage_type, fallback_order, created_by, \
                     created_at) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(mapping_id.clone()),
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(model_config.model_definition_id.clone()),
                    Val::Float(model_config.weight),
                    Val::Text(model_config.linkage_type.clone()),
                    Val::OptInt(model_config.fallback_order.map(i64::from)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                ],
            )
            .await;
        if let Err(error) = result {
            if is_unique_violation(&error) {
                return Err(MlflowError::resource_already_exists(format!(
                    "Model definition '{}' is already attached to endpoint '{endpoint_id}'",
                    model_config.model_definition_id
                )));
            }
            return Err(internal(error));
        }
        self.db()
            .exec(
                &format!(
                    "UPDATE endpoints SET last_updated_at = {}, last_updated_by = {} WHERE \
                     endpoint_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                    dialect.placeholder(4)
                ),
                &[
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        self.find_mapping(workspace, &mapping_id)
            .await?
            .ok_or_else(|| MlflowError::internal_error("Created gateway mapping was not found"))
    }

    pub async fn detach_model_from_gateway_endpoint(
        &self,
        workspace: &str,
        endpoint_id: &str,
        model_definition_id: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let mapping = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT m.mapping_id FROM endpoint_model_mappings m JOIN endpoints e ON \
                     e.endpoint_id = m.endpoint_id WHERE m.endpoint_id = {} AND \
                     m.model_definition_id = {} AND e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(model_definition_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("mapping_id"),
            )
            .await
            .map_err(internal)?;
        let Some(mapping_id) = mapping else {
            if self
                .get_gateway_endpoint(workspace, Some(endpoint_id), None)
                .await
                .is_err()
            {
                return Err(endpoint_not_found("endpoint_id", endpoint_id));
            }
            return Err(MlflowError::resource_does_not_exist(format!(
                "Model definition '{model_definition_id}' is not attached to endpoint '{endpoint_id}'"
            )));
        };
        self.db()
            .exec(
                &format!(
                    "DELETE FROM endpoint_model_mappings WHERE mapping_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(mapping_id)],
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        Ok(())
    }

    pub async fn create_gateway_endpoint_binding(
        &self,
        workspace: &str,
        endpoint_id: &str,
        resource_type: &str,
        resource_id: &str,
        created_by: Option<&str>,
    ) -> Result<EndpointBinding, MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        let now = now_millis();
        let dialect = self.db().dialect();
        let placeholders = (1..=8)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        self.db()
            .exec(
                &format!(
                    "INSERT INTO endpoint_bindings (endpoint_id, resource_type, resource_id, \
                     created_at, created_by, last_updated_at, last_updated_by, display_name) \
                     VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_id.to_string()),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::OptText(None),
                ],
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        self.find_binding(workspace, endpoint_id, resource_type, resource_id)
            .await?
            .ok_or_else(|| MlflowError::internal_error("Created gateway binding was not found"))
    }

    pub async fn delete_gateway_endpoint_binding(
        &self,
        workspace: &str,
        endpoint_id: &str,
        resource_type: &str,
        resource_id: &str,
    ) -> Result<(), MlflowError> {
        if self
            .find_binding(workspace, endpoint_id, resource_type, resource_id)
            .await?
            .is_none()
        {
            return Err(MlflowError::resource_does_not_exist(format!(
                "GatewayEndpointBinding not found (endpoint_id='{endpoint_id}', resource_type='{resource_type}', resource_id='{resource_id}')"
            )));
        }
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM endpoint_bindings WHERE endpoint_id = {} AND resource_type = {} \
                     AND resource_id = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_id.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.invalidate_secret_cache();
        Ok(())
    }

    pub async fn list_gateway_endpoint_bindings(
        &self,
        workspace: &str,
        endpoint_id: Option<&str>,
        resource_type: Option<&str>,
        resource_id: Option<&str>,
    ) -> Result<Vec<EndpointBinding>, MlflowError> {
        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(workspace.to_string())];
        let mut filters = vec![format!("e.workspace = {}", dialect.placeholder(1))];
        for (column, value) in [
            ("b.endpoint_id", endpoint_id),
            ("b.resource_type", resource_type),
            ("b.resource_id", resource_id),
        ] {
            if let Some(value) = value {
                values.push(Val::Text(value.to_string()));
                filters.push(format!("{column} = {}", dialect.placeholder(values.len())));
            }
        }
        self.db()
            .fetch_all(
                &format!(
                    "SELECT b.endpoint_id, b.resource_type, b.resource_id, b.created_at, \
                     b.last_updated_at, b.created_by, b.last_updated_by, b.display_name FROM \
                     endpoint_bindings b JOIN endpoints e ON e.endpoint_id = b.endpoint_id WHERE {}",
                    filters.join(" AND ")
                ),
                &values,
                map_binding,
            )
            .await
            .map_err(internal)
    }

    pub async fn set_gateway_endpoint_tag(
        &self,
        workspace: &str,
        endpoint_id: &str,
        key: &str,
        value: Option<&str>,
    ) -> Result<(), MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        tx.exec(
            &format!(
                "DELETE FROM endpoint_tags WHERE endpoint_id = {} AND \"key\" = {}",
                dialect.placeholder(1),
                dialect.placeholder(2)
            ),
            &[
                Val::Text(endpoint_id.to_string()),
                Val::Text(key.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.exec(
            &format!(
                "INSERT INTO endpoint_tags (\"key\", value, endpoint_id) VALUES ({}, {}, {})",
                dialect.placeholder(1),
                dialect.placeholder(2),
                dialect.placeholder(3)
            ),
            &[
                Val::Text(key.to_string()),
                Val::OptText(value.map(str::to_string)),
                Val::Text(endpoint_id.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)
    }

    pub async fn delete_gateway_endpoint_tag(
        &self,
        workspace: &str,
        endpoint_id: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM endpoint_tags WHERE endpoint_id = {} AND \"key\" = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(key.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    async fn validate_model_configs(
        &self,
        workspace: &str,
        configs: &[EndpointModelConfig],
    ) -> Result<(), MlflowError> {
        let mut missing = Vec::new();
        for id in configs
            .iter()
            .map(|config| config.model_definition_id.as_str())
        {
            if self
                .find_gateway_model_definition(workspace, "model_definition_id", id)
                .await?
                .is_none()
                && !missing.iter().any(|value| value == id)
            {
                missing.push(id.to_string());
            }
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(MlflowError::resource_does_not_exist(format!(
                "Model definitions not found: {}",
                missing.join(", ")
            )))
        }
    }

    async fn gateway_experiment_id(
        &self,
        workspace: &str,
        endpoint_name: &str,
        endpoint_id: &str,
    ) -> Result<String, MlflowError> {
        let name = format!("gateway/{endpoint_name}");
        match self
            .create_experiment(
                workspace,
                &name,
                None,
                &[
                    ("mlflow.experiment.sourceType", "GATEWAY"),
                    ("mlflow.experiment.sourceId", endpoint_id),
                    ("mlflow.experiment.isGateway", "true"),
                ],
            )
            .await
        {
            Ok(id) => Ok(id),
            Err(error) if error.error_code == mlflow_error::ErrorCode::ResourceAlreadyExists => {
                self.get_experiment_by_name(workspace, &name)
                    .await?
                    .map(|experiment| experiment.experiment_id)
                    .ok_or(error)
            }
            Err(error) => Err(error),
        }
    }

    async fn populate_endpoint(&self, mut endpoint: Endpoint) -> Result<Endpoint, MlflowError> {
        endpoint.model_mappings = self.load_mappings(&endpoint.endpoint_id).await?;
        endpoint.tags = self.load_endpoint_tags(&endpoint.endpoint_id).await?;
        Ok(endpoint)
    }

    async fn load_mappings(
        &self,
        endpoint_id: &str,
    ) -> Result<Vec<EndpointModelMapping>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_all(
                &format!(
                    "SELECT m.mapping_id, m.endpoint_id, m.model_definition_id, m.weight, \
                     m.linkage_type, m.fallback_order, m.created_at, m.created_by, \
                     d.name, d.secret_id, s.secret_name, d.provider, d.model_name, \
                     d.created_at AS definition_created_at, d.last_updated_at AS \
                     definition_last_updated_at, d.created_by AS definition_created_by, \
                     d.last_updated_by AS definition_last_updated_by, d.workspace FROM \
                     endpoint_model_mappings m LEFT JOIN model_definitions d ON \
                     d.model_definition_id = m.model_definition_id LEFT JOIN secrets s ON \
                     s.secret_id = d.secret_id WHERE m.endpoint_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(endpoint_id.to_string())],
                map_mapping,
            )
            .await
            .map_err(internal)
    }

    async fn find_mapping(
        &self,
        workspace: &str,
        mapping_id: &str,
    ) -> Result<Option<EndpointModelMapping>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT m.mapping_id, m.endpoint_id, m.model_definition_id, m.weight, \
                     m.linkage_type, m.fallback_order, m.created_at, m.created_by, d.name, \
                     d.secret_id, s.secret_name, d.provider, d.model_name, d.created_at AS \
                     definition_created_at, d.last_updated_at AS definition_last_updated_at, \
                     d.created_by AS definition_created_by, d.last_updated_by AS \
                     definition_last_updated_by, d.workspace FROM endpoint_model_mappings m JOIN \
                     endpoints e ON e.endpoint_id = m.endpoint_id LEFT JOIN model_definitions d ON \
                     d.model_definition_id = m.model_definition_id LEFT JOIN secrets s ON \
                     s.secret_id = d.secret_id WHERE m.mapping_id = {} AND e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(mapping_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_mapping,
            )
            .await
            .map_err(internal)
    }

    async fn load_endpoint_tags(&self, endpoint_id: &str) -> Result<Vec<EndpointTag>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_all(
                &format!(
                    "SELECT \"key\", value FROM endpoint_tags WHERE endpoint_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(endpoint_id.to_string())],
                |row| {
                    Ok(EndpointTag {
                        key: row.get_string("key")?,
                        value: row.get_opt_string("value")?,
                    })
                },
            )
            .await
            .map_err(internal)
    }

    async fn find_binding(
        &self,
        workspace: &str,
        endpoint_id: &str,
        resource_type: &str,
        resource_id: &str,
    ) -> Result<Option<EndpointBinding>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT b.endpoint_id, b.resource_type, b.resource_id, b.created_at, \
                     b.last_updated_at, b.created_by, b.last_updated_by, b.display_name FROM \
                     endpoint_bindings b JOIN endpoints e ON e.endpoint_id = b.endpoint_id WHERE \
                     b.endpoint_id = {} AND b.resource_type = {} AND b.resource_id = {} AND \
                     e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                    dialect.placeholder(4)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_binding,
            )
            .await
            .map_err(internal)
    }
}

async fn insert_mapping(
    tx: &mut super::dbutil::Tx<'_>,
    dialect: crate::Dialect,
    endpoint_id: &str,
    config: &EndpointModelConfig,
    created_by: Option<&str>,
    now: i64,
) -> Result<(), MlflowError> {
    let placeholders = (1..=8)
        .map(|index| dialect.placeholder(index))
        .collect::<Vec<_>>();
    tx.exec(
        &format!(
            "INSERT INTO endpoint_model_mappings (mapping_id, endpoint_id, model_definition_id, \
             weight, linkage_type, fallback_order, created_by, created_at) VALUES ({})",
            placeholders.join(", ")
        ),
        &[
            Val::Text(uuid_id("m-")),
            Val::Text(endpoint_id.to_string()),
            Val::Text(config.model_definition_id.clone()),
            Val::Float(config.weight),
            Val::Text(config.linkage_type.clone()),
            Val::OptInt(config.fallback_order.map(i64::from)),
            Val::OptText(created_by.map(str::to_string)),
            Val::Int(now),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

fn build_fallback_json(
    config: Option<&FallbackConfig>,
    model_configs: &[EndpointModelConfig],
    force: bool,
) -> Option<String> {
    let ids = model_configs
        .iter()
        .filter(|config| config.linkage_type == "FALLBACK")
        .map(|config| config.model_definition_id.clone())
        .collect::<Vec<_>>();
    (force || config.is_some() || !ids.is_empty()).then(|| {
        fallback_json(
            config.unwrap_or(&FallbackConfig {
                strategy: None,
                max_attempts: None,
            }),
            &ids,
        )
    })
}

fn fallback_json(config: &FallbackConfig, ids: &[String]) -> String {
    let value = serde_json::json!({
        "strategy": config.strategy,
        "max_attempts": config.max_attempts,
        "model_definition_ids": ids,
    });
    python_json_dumps(&value, false)
}

fn current_fallback_json(endpoint: &Endpoint) -> Option<String> {
    endpoint.fallback_config.as_ref().map(|config| {
        let ids = endpoint
            .model_mappings
            .iter()
            .filter(|mapping| mapping.linkage_type == "FALLBACK")
            .map(|mapping| mapping.model_definition_id.clone())
            .collect::<Vec<_>>();
        fallback_json(config, &ids)
    })
}

fn fallback_ids_from_json(value: Option<String>) -> Vec<String> {
    value
        .as_deref()
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .and_then(|value| value.get("model_definition_ids").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

fn parse_experiment_id(value: &str) -> Result<i64, MlflowError> {
    value.parse().map_err(|_| {
        MlflowError::invalid_parameter_value(format!("Invalid experiment ID: '{value}'"))
    })
}

fn map_endpoint_root(row: &dyn RowLike) -> Result<Endpoint, sqlx::Error> {
    let fallback_config = row
        .get_opt_string("fallback_config_json")?
        .and_then(|value| serde_json::from_str::<Value>(&value).ok())
        .map(|value| FallbackConfig {
            strategy: value
                .get("strategy")
                .and_then(Value::as_str)
                .map(str::to_string),
            max_attempts: value
                .get("max_attempts")
                .and_then(Value::as_i64)
                .and_then(|value| i32::try_from(value).ok()),
        });
    Ok(Endpoint {
        endpoint_id: row.get_string("endpoint_id")?,
        name: row.get_opt_string("name")?,
        created_at: row.get_i64("created_at")?,
        last_updated_at: row.get_i64("last_updated_at")?,
        model_mappings: Vec::new(),
        tags: Vec::new(),
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        routing_strategy: row.get_opt_string("routing_strategy")?,
        fallback_config,
        experiment_id: row
            .get_opt_int("experiment_id")?
            .map(|value| value.to_string()),
        usage_tracking: row.get_bool("usage_tracking")?,
        workspace: row.get_string("workspace")?,
    })
}

fn map_mapping(row: &dyn RowLike) -> Result<EndpointModelMapping, sqlx::Error> {
    let model_definition_id = row.get_string("model_definition_id")?;
    let definition = match row.get_opt_string("name")? {
        Some(name) => Some(GatewayModelDefinition {
            model_definition_id: model_definition_id.clone(),
            name,
            secret_id: row.get_opt_string("secret_id")?,
            secret_name: row.get_opt_string("secret_name")?,
            provider: row.get_string("provider")?,
            model_name: row.get_string("model_name")?,
            created_at: row.get_i64("definition_created_at")?,
            last_updated_at: row.get_i64("definition_last_updated_at")?,
            created_by: row.get_opt_string("definition_created_by")?,
            last_updated_by: row.get_opt_string("definition_last_updated_by")?,
            workspace: row.get_string("workspace")?,
        }),
        None => None,
    };
    Ok(EndpointModelMapping {
        mapping_id: row.get_string("mapping_id")?,
        endpoint_id: row.get_string("endpoint_id")?,
        model_definition_id,
        model_definition: definition,
        weight: row.get_f64("weight")?,
        linkage_type: row.get_string("linkage_type")?,
        fallback_order: row
            .get_opt_int("fallback_order")?
            .and_then(|value| i32::try_from(value).ok()),
        created_at: row.get_i64("created_at")?,
        created_by: row.get_opt_string("created_by")?,
    })
}

fn map_binding(row: &dyn RowLike) -> Result<EndpointBinding, sqlx::Error> {
    Ok(EndpointBinding {
        endpoint_id: row.get_string("endpoint_id")?,
        resource_type: row.get_string("resource_type")?,
        resource_id: row.get_string("resource_id")?,
        created_at: row.get_i64("created_at")?,
        last_updated_at: row.get_i64("last_updated_at")?,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        display_name: row.get_opt_string("display_name")?,
    })
}

fn map_secret_info(row: &dyn RowLike) -> Result<GatewaySecretInfo, sqlx::Error> {
    let masked = json_string_map(row.get_opt_string("masked_value")?.as_deref());
    Ok(GatewaySecretInfo {
        secret_id: row.get_string("secret_id")?,
        secret_name: row.get_string("secret_name")?,
        masked_values: if masked.is_empty() {
            HashMap::from([("value".to_string(), "***".to_string())])
        } else {
            masked
        },
        provider: row.get_opt_string("provider")?,
        auth_config: json_string_map(row.get_opt_string("auth_config")?.as_deref()),
        created_at: row.get_i64("created_at")?,
        last_updated_at: row.get_i64("last_updated_at")?,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        workspace: row.get_string("workspace")?,
    })
}

fn map_model_definition(row: &dyn RowLike) -> Result<GatewayModelDefinition, sqlx::Error> {
    Ok(GatewayModelDefinition {
        model_definition_id: row.get_string("model_definition_id")?,
        name: row.get_string("name")?,
        secret_id: row.get_opt_string("secret_id")?,
        secret_name: row.get_opt_string("secret_name")?,
        provider: row.get_string("provider")?,
        model_name: row.get_string("model_name")?,
        created_at: row.get_i64("created_at")?,
        last_updated_at: row.get_i64("last_updated_at")?,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        workspace: row.get_string("workspace")?,
    })
}

fn is_foreign_key_violation(error: &sqlx::Error) -> bool {
    let Some(error) = error.as_database_error() else {
        return false;
    };
    if error
        .code()
        .is_some_and(|code| matches!(code.as_ref(), "23503" | "1451" | "787"))
    {
        return true;
    }
    error.message().to_ascii_lowercase().contains("foreign key")
}

impl TrackingStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn create_budget_policy(
        &self,
        workspace: &str,
        budget_unit: &str,
        budget_amount: f64,
        duration_unit: &str,
        duration_value: i32,
        target_scope: &str,
        budget_action: &str,
        created_by: Option<&str>,
    ) -> Result<BudgetPolicy, MlflowError> {
        let id = uuid_id("bp-");
        let now = now_millis();
        let dialect = self.db().dialect();
        let placeholders = (1..=12)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        self.db()
            .exec(
                &format!(
                    "INSERT INTO budget_policies (budget_policy_id, budget_unit, budget_amount, \
                     duration_unit, duration_value, target_scope, budget_action, created_by, \
                     created_at, last_updated_by, last_updated_at, workspace) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(id.clone()),
                    Val::Text(budget_unit.to_string()),
                    Val::Float(budget_amount),
                    Val::Text(duration_unit.to_string()),
                    Val::Int(i64::from(duration_value)),
                    Val::Text(target_scope.to_string()),
                    Val::Text(budget_action.to_string()),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_budget_policy(workspace, &id).await
    }

    pub async fn get_budget_policy(
        &self,
        workspace: &str,
        budget_policy_id: &str,
    ) -> Result<BudgetPolicy, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT budget_policy_id, budget_unit, budget_amount, duration_unit, \
                     duration_value, target_scope, budget_action, created_by, created_at, \
                     last_updated_by, last_updated_at, workspace FROM budget_policies WHERE \
                     budget_policy_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(budget_policy_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_budget_policy,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "BudgetPolicy not found (budget_policy_id='{budget_policy_id}')"
                ))
            })
    }

    pub async fn update_budget_policy(
        &self,
        workspace: &str,
        budget_policy_id: &str,
        update: BudgetPolicyUpdate<'_>,
    ) -> Result<BudgetPolicy, MlflowError> {
        self.get_budget_policy(workspace, budget_policy_id).await?;
        let dialect = self.db().dialect();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let Some(value) = update.budget_unit {
            values.push(Val::Text(value.to_string()));
            assignments.push(format!(
                "budget_unit = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(value) = update.budget_amount {
            values.push(Val::Float(value));
            assignments.push(format!(
                "budget_amount = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some((unit, value)) = update.duration {
            values.push(Val::Text(unit.to_string()));
            assignments.push(format!(
                "duration_unit = {}",
                dialect.placeholder(values.len())
            ));
            values.push(Val::Int(i64::from(value)));
            assignments.push(format!(
                "duration_value = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(value) = update.target_scope {
            values.push(Val::Text(value.to_string()));
            assignments.push(format!(
                "target_scope = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(value) = update.budget_action {
            values.push(Val::Text(value.to_string()));
            assignments.push(format!(
                "budget_action = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_updated_at = {}",
            dialect.placeholder(values.len())
        ));
        if let Some(value) = update.updated_by {
            values.push(Val::Text(value.to_string()));
            assignments.push(format!(
                "last_updated_by = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Text(budget_policy_id.to_string()));
        let id_placeholder = dialect.placeholder(values.len());
        values.push(Val::Text(workspace.to_string()));
        let workspace_placeholder = dialect.placeholder(values.len());
        self.db()
            .exec(
                &format!(
                    "UPDATE budget_policies SET {} WHERE budget_policy_id = {id_placeholder} AND \
                     workspace = {workspace_placeholder}",
                    assignments.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        self.get_budget_policy(workspace, budget_policy_id).await
    }

    pub async fn delete_budget_policy(
        &self,
        workspace: &str,
        budget_policy_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_budget_policy(workspace, budget_policy_id).await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM budget_policies WHERE budget_policy_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(budget_policy_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    pub async fn list_budget_policies(
        &self,
        workspace: &str,
        max_results: i64,
        page_token: Option<&str>,
    ) -> Result<BudgetPoliciesPage, MlflowError> {
        validate_max_results(max_results)?;
        let offset = mlflow_search::parse_start_offset_from_page_token(page_token)
            .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
        let dialect = self.db().dialect();
        let policies = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT budget_policy_id, budget_unit, budget_amount, duration_unit, \
                     duration_value, target_scope, budget_action, created_by, created_at, \
                     last_updated_by, last_updated_at, workspace FROM budget_policies WHERE \
                     workspace = {} ORDER BY budget_policy_id LIMIT {} OFFSET {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(workspace.to_string()),
                    Val::Int(max_results + 1),
                    Val::Int(offset),
                ],
                map_budget_policy,
            )
            .await
            .map_err(internal)?;
        let has_more = policies.len() > usize::try_from(max_results).unwrap_or(usize::MAX);
        Ok(BudgetPoliciesPage {
            policies: policies
                .into_iter()
                .take(usize::try_from(max_results).unwrap_or(usize::MAX))
                .collect(),
            next_page_token: has_more
                .then(|| mlflow_search::create_page_token(offset + max_results)),
        })
    }

    pub async fn list_budget_windows(
        &self,
        workspace: &str,
    ) -> Result<Vec<BudgetWindow>, MlflowError> {
        let policies = self
            .list_budget_policies(workspace, SEARCH_MAX_RESULTS_DEFAULT, None)
            .await?
            .policies;
        let now = Utc::now();
        let mut windows = Vec::with_capacity(policies.len());
        for policy in policies {
            let (start, end) = budget_window_bounds(&policy, now)?;
            let spend_workspace =
                (policy.target_scope == "WORKSPACE").then_some(policy.workspace.as_str());
            let current_spend = self
                .sum_gateway_trace_cost(
                    start.timestamp_millis(),
                    end.timestamp_millis(),
                    spend_workspace,
                )
                .await?;
            windows.push(BudgetWindow {
                budget_policy_id: policy.budget_policy_id,
                window_start_ms: start.timestamp_millis(),
                window_end_ms: end.timestamp_millis(),
                current_spend,
            });
        }
        Ok(windows)
    }

    async fn sum_gateway_trace_cost(
        &self,
        start_time_ms: i64,
        end_time_ms: i64,
        workspace: Option<&str>,
    ) -> Result<f64, MlflowError> {
        let dialect = self.db().dialect();
        let mut values = vec![
            Val::Text("total_cost".to_string()),
            Val::Text("mlflow.gateway.endpointId".to_string()),
            Val::Int(start_time_ms),
            Val::Int(end_time_ms),
        ];
        let mut workspace_join = "";
        let mut workspace_filter = String::new();
        if let Some(workspace) = workspace {
            workspace_join = " JOIN experiments e ON e.experiment_id = t.experiment_id";
            values.push(Val::Text(workspace.to_string()));
            workspace_filter = format!(" AND e.workspace = {}", dialect.placeholder(values.len()));
        }
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT COALESCE(SUM(sm.value), 0.0) AS total FROM span_metrics sm JOIN \
                     trace_info t ON t.request_id = sm.trace_id JOIN trace_request_metadata tm ON \
                     tm.request_id = t.request_id{workspace_join} WHERE sm.\"key\" = {} AND \
                     tm.\"key\" = {} AND t.timestamp_ms >= {} AND t.timestamp_ms < \
                     {}{workspace_filter}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                    dialect.placeholder(4),
                ),
                &values,
                |row| row.get_f64("total"),
            )
            .await
            .map_err(internal)
            .map(|value| value.unwrap_or(0.0))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_gateway_guardrail(
        &self,
        workspace: &str,
        name: &str,
        scorer_id: &str,
        scorer_version: i32,
        stage: &str,
        action: &str,
        action_endpoint_id: Option<&str>,
        created_by: Option<&str>,
    ) -> Result<GatewayGuardrail, MlflowError> {
        self.find_scorer_version(workspace, scorer_id, scorer_version)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Scorer version not found (scorer_id='{scorer_id}', scorer_version='{scorer_version}')"
                ))
            })?;
        if let Some(endpoint_id) = action_endpoint_id {
            self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
                .await?;
        }
        let id = uuid_id("gr-");
        let now = now_millis();
        let dialect = self.db().dialect();
        let placeholders = (1..=11)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        self.db()
            .exec(
                &format!(
                    "INSERT INTO guardrails (guardrail_id, name, scorer_id, scorer_version, stage, \
                     action, action_endpoint_id, created_by, created_at, last_updated_by, \
                     last_updated_at, workspace) VALUES ({}, {})",
                    placeholders.join(", "),
                    dialect.placeholder(12)
                ),
                &[
                    Val::Text(id.clone()),
                    Val::Text(name.to_string()),
                    Val::Text(scorer_id.to_string()),
                    Val::Int(i64::from(scorer_version)),
                    Val::Text(stage.to_string()),
                    Val::Text(action.to_string()),
                    Val::OptText(action_endpoint_id.map(str::to_string)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_gateway_guardrail(workspace, &id).await
    }

    pub async fn get_gateway_guardrail(
        &self,
        workspace: &str,
        guardrail_id: &str,
    ) -> Result<GatewayGuardrail, MlflowError> {
        self.find_gateway_guardrail(workspace, guardrail_id)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Guardrail not found (guardrail_id='{guardrail_id}')"
                ))
            })
    }

    pub async fn delete_gateway_guardrail(
        &self,
        workspace: &str,
        guardrail_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_gateway_guardrail(workspace, guardrail_id).await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM guardrails WHERE guardrail_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(guardrail_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    pub async fn list_gateway_guardrails(
        &self,
        workspace: &str,
        max_results: i64,
        page_token: Option<&str>,
    ) -> Result<GuardrailsPage, MlflowError> {
        validate_max_results(max_results)?;
        let offset = mlflow_search::parse_start_offset_from_page_token(page_token)
            .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
        let dialect = self.db().dialect();
        let guardrails = self
            .db()
            .fetch_all(
                &guardrail_select(
                    &format!("g.workspace = {}", dialect.placeholder(1)),
                    &format!(
                        " ORDER BY g.guardrail_id LIMIT {} OFFSET {}",
                        dialect.placeholder(2),
                        dialect.placeholder(3)
                    ),
                ),
                &[
                    Val::Text(workspace.to_string()),
                    Val::Int(max_results + 1),
                    Val::Int(offset),
                ],
                map_guardrail,
            )
            .await
            .map_err(internal)?;
        let has_more = guardrails.len() > usize::try_from(max_results).unwrap_or(usize::MAX);
        Ok(GuardrailsPage {
            guardrails: guardrails
                .into_iter()
                .take(usize::try_from(max_results).unwrap_or(usize::MAX))
                .collect(),
            next_page_token: has_more
                .then(|| mlflow_search::create_page_token(offset + max_results)),
        })
    }

    pub async fn add_guardrail_to_endpoint(
        &self,
        workspace: &str,
        endpoint_id: &str,
        guardrail_id: &str,
        execution_order: Option<i64>,
        created_by: Option<&str>,
    ) -> Result<GatewayGuardrailConfig, MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        self.get_gateway_guardrail(workspace, guardrail_id).await?;
        let dialect = self.db().dialect();
        let result = self
            .db()
            .exec(
                &format!(
                    "INSERT INTO guardrail_configs (endpoint_id, guardrail_id, execution_order, \
                     created_by, created_at, workspace) VALUES ({}, {}, {}, {}, {}, {})",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                    dialect.placeholder(4),
                    dialect.placeholder(5),
                    dialect.placeholder(6)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(guardrail_id.to_string()),
                    Val::OptInt(execution_order),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now_millis()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await;
        if let Err(error) = result {
            if is_unique_violation(&error) {
                return Err(MlflowError::resource_already_exists(format!(
                    "Guardrail '{guardrail_id}' is already added to endpoint '{endpoint_id}'"
                )));
            }
            return Err(internal(error));
        }
        self.get_guardrail_config(workspace, endpoint_id, guardrail_id, false)
            .await
    }

    pub async fn update_endpoint_guardrail_config(
        &self,
        workspace: &str,
        endpoint_id: &str,
        guardrail_id: &str,
        execution_order: Option<i64>,
    ) -> Result<GatewayGuardrailConfig, MlflowError> {
        self.get_guardrail_config(workspace, endpoint_id, guardrail_id, false)
            .await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "UPDATE guardrail_configs SET execution_order = {} WHERE endpoint_id = {} AND \
                     guardrail_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                    dialect.placeholder(4)
                ),
                &[
                    Val::OptInt(execution_order),
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(guardrail_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_guardrail_config(workspace, endpoint_id, guardrail_id, false)
            .await
    }

    pub async fn remove_guardrail_from_endpoint(
        &self,
        workspace: &str,
        endpoint_id: &str,
        guardrail_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_guardrail_config(workspace, endpoint_id, guardrail_id, false)
            .await?;
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM guardrail_configs WHERE endpoint_id = {} AND guardrail_id = {} \
                     AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(guardrail_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    pub async fn list_endpoint_guardrail_configs(
        &self,
        workspace: &str,
        endpoint_id: &str,
    ) -> Result<Vec<GatewayGuardrailConfig>, MlflowError> {
        self.get_gateway_endpoint(workspace, Some(endpoint_id), None)
            .await?;
        let dialect = self.db().dialect();
        let rows = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT endpoint_id, guardrail_id, execution_order, created_by, created_at, \
                     workspace FROM guardrail_configs WHERE endpoint_id = {} AND workspace = {} \
                     ORDER BY CASE WHEN execution_order IS NULL THEN 1 ELSE 0 END, execution_order, \
                     guardrail_id",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_guardrail_config,
            )
            .await
            .map_err(internal)?;
        let mut configs = Vec::with_capacity(rows.len());
        for mut config in rows {
            config.guardrail = Some(
                self.get_gateway_guardrail(workspace, &config.guardrail_id)
                    .await?,
            );
            configs.push(config);
        }
        Ok(configs)
    }

    async fn get_guardrail_config(
        &self,
        workspace: &str,
        endpoint_id: &str,
        guardrail_id: &str,
        with_guardrail: bool,
    ) -> Result<GatewayGuardrailConfig, MlflowError> {
        let dialect = self.db().dialect();
        let mut config = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT endpoint_id, guardrail_id, execution_order, created_by, created_at, \
                     workspace FROM guardrail_configs WHERE endpoint_id = {} AND guardrail_id = {} \
                     AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text(guardrail_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_guardrail_config,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "GuardrailConfig not found (endpoint_id='{endpoint_id}', guardrail_id='{guardrail_id}')"
                ))
            })?;
        if with_guardrail {
            config.guardrail = Some(self.get_gateway_guardrail(workspace, guardrail_id).await?);
        }
        Ok(config)
    }

    async fn find_gateway_guardrail(
        &self,
        workspace: &str,
        guardrail_id: &str,
    ) -> Result<Option<GatewayGuardrail>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &guardrail_select(
                    &format!(
                        "g.guardrail_id = {} AND g.workspace = {}",
                        dialect.placeholder(1),
                        dialect.placeholder(2)
                    ),
                    "",
                ),
                &[
                    Val::Text(guardrail_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_guardrail,
            )
            .await
            .map_err(internal)
    }

    async fn find_scorer_version(
        &self,
        workspace: &str,
        scorer_id: &str,
        scorer_version: i32,
    ) -> Result<Option<ScorerVersion>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT s.experiment_id, s.scorer_name, s.scorer_id, v.scorer_version, \
                     v.serialized_scorer, v.creation_time FROM scorers s JOIN scorer_versions v ON \
                     v.scorer_id = s.scorer_id JOIN experiments e ON e.experiment_id = \
                     s.experiment_id WHERE s.scorer_id = {} AND v.scorer_version = {} AND \
                     e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(scorer_id.to_string()),
                    Val::Int(i64::from(scorer_version)),
                    Val::Text(workspace.to_string()),
                ],
                map_scorer_version,
            )
            .await
            .map_err(internal)
    }
}

fn validate_max_results(max_results: i64) -> Result<(), MlflowError> {
    if max_results <= 0 || max_results > SEARCH_MAX_RESULTS_THRESHOLD {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value for request parameter max_results. It must be at least 1 and at most {SEARCH_MAX_RESULTS_THRESHOLD}, but got value {max_results}"
        )));
    }
    Ok(())
}

fn map_budget_policy(row: &dyn RowLike) -> Result<BudgetPolicy, sqlx::Error> {
    Ok(BudgetPolicy {
        budget_policy_id: row.get_string("budget_policy_id")?,
        budget_unit: row.get_string("budget_unit")?,
        budget_amount: row.get_f64("budget_amount")?,
        duration_unit: row.get_string("duration_unit")?,
        duration_value: i32::try_from(row.get_int("duration_value")?).unwrap_or_default(),
        target_scope: row.get_string("target_scope")?,
        budget_action: row.get_string("budget_action")?,
        created_by: row.get_opt_string("created_by")?,
        created_at: row.get_i64("created_at")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        last_updated_at: row.get_i64("last_updated_at")?,
        workspace: row.get_string("workspace")?,
    })
}

fn budget_window_bounds(
    policy: &BudgetPolicy,
    now: chrono::DateTime<Utc>,
) -> Result<(chrono::DateTime<Utc>, chrono::DateTime<Utc>), MlflowError> {
    let value = i64::from(policy.duration_value);
    let epoch = Utc.timestamp_opt(0, 0).single().expect("Unix epoch");
    let start = match policy.duration_unit.as_str() {
        "MINUTES" => {
            let elapsed = now.timestamp().div_euclid(60);
            epoch + chrono::Duration::minutes(elapsed.div_euclid(value) * value)
        }
        "HOURS" => {
            let elapsed = now.timestamp().div_euclid(3600);
            epoch + chrono::Duration::hours(elapsed.div_euclid(value) * value)
        }
        "DAYS" => {
            let elapsed = now.timestamp().div_euclid(86_400);
            epoch + chrono::Duration::days(elapsed.div_euclid(value) * value)
        }
        "WEEKS" => {
            let sunday = epoch - chrono::Duration::days(4);
            let elapsed = (now - sunday).num_days();
            sunday + chrono::Duration::days(elapsed.div_euclid(7 * value) * 7 * value)
        }
        "MONTHS" => {
            let total = i64::from(now.year() - 1970) * 12 + i64::from(now.month0());
            let start_month = total.div_euclid(value) * value;
            Utc.with_ymd_and_hms(
                1970 + i32::try_from(start_month.div_euclid(12)).unwrap_or_default(),
                u32::try_from(start_month.rem_euclid(12) + 1).unwrap_or(1),
                1,
                0,
                0,
                0,
            )
            .single()
            .ok_or_else(|| MlflowError::internal_error("Invalid monthly budget window"))?
        }
        unit => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Unknown duration type: {unit}"
            )))
        }
    };
    let end = match policy.duration_unit.as_str() {
        "MINUTES" => start + chrono::Duration::minutes(value),
        "HOURS" => start + chrono::Duration::hours(value),
        "DAYS" => start + chrono::Duration::days(value),
        "WEEKS" => start + chrono::Duration::weeks(value),
        "MONTHS" => {
            let total = i64::from(start.year()) * 12 + i64::from(start.month0()) + value;
            Utc.with_ymd_and_hms(
                i32::try_from(total.div_euclid(12)).unwrap_or_default(),
                u32::try_from(total.rem_euclid(12) + 1).unwrap_or(1),
                1,
                0,
                0,
                0,
            )
            .single()
            .ok_or_else(|| MlflowError::internal_error("Invalid monthly budget window"))?
        }
        _ => unreachable!(),
    };
    Ok((start, end))
}

fn guardrail_select(filter: &str, suffix: &str) -> String {
    format!(
        "SELECT g.guardrail_id, g.name, g.stage, g.action, a.name AS \
         action_endpoint_name, g.created_by, g.created_at, g.last_updated_by, \
         g.last_updated_at, g.workspace, s.experiment_id, s.scorer_name, s.scorer_id, \
         v.scorer_version, v.serialized_scorer, v.creation_time FROM guardrails g JOIN scorers s \
         ON s.scorer_id = g.scorer_id JOIN scorer_versions v ON v.scorer_id = g.scorer_id AND \
         v.scorer_version = g.scorer_version LEFT JOIN endpoints a ON a.endpoint_id = \
         g.action_endpoint_id WHERE {filter}{suffix}"
    )
}

fn map_guardrail(row: &dyn RowLike) -> Result<GatewayGuardrail, sqlx::Error> {
    Ok(GatewayGuardrail {
        guardrail_id: row.get_string("guardrail_id")?,
        name: row.get_string("name")?,
        scorer: map_scorer_version(row)?,
        stage: row.get_string("stage")?,
        action: row.get_string("action")?,
        action_endpoint_name: row.get_opt_string("action_endpoint_name")?,
        created_by: row.get_opt_string("created_by")?,
        created_at: row.get_i64("created_at")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        last_updated_at: row.get_i64("last_updated_at")?,
        workspace: row.get_string("workspace")?,
    })
}

fn map_scorer_version(row: &dyn RowLike) -> Result<ScorerVersion, sqlx::Error> {
    Ok(ScorerVersion {
        experiment_id: row.get_int("experiment_id")?.to_string(),
        scorer_name: row.get_string("scorer_name")?,
        scorer_version: i32::try_from(row.get_int("scorer_version")?).unwrap_or_default(),
        serialized_scorer: row.get_string("serialized_scorer")?,
        creation_time: row.get_opt_i64("creation_time")?,
        scorer_id: row.get_string("scorer_id")?,
    })
}

fn map_guardrail_config(row: &dyn RowLike) -> Result<GatewayGuardrailConfig, sqlx::Error> {
    Ok(GatewayGuardrailConfig {
        endpoint_id: row.get_string("endpoint_id")?,
        guardrail_id: row.get_string("guardrail_id")?,
        execution_order: row.get_opt_int("execution_order")?,
        created_by: row.get_opt_string("created_by")?,
        created_at: row.get_i64("created_at")?,
        guardrail: None,
        workspace: row.get_string("workspace")?,
    })
}
