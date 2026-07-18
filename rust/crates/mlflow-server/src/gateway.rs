//! AI Gateway CRUD handlers.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
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
