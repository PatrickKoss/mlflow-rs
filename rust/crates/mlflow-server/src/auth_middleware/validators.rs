//! The validators (`validate_can_*`) and the permission-resolution flow they
//! share (plan T9.4, §3.16), mirroring `mlflow/server/auth/__init__.py`.
//!
//! Each [`Validator`] variant maps 1:1 to a `validate_can_*` function. The
//! resolution flow is:
//!
//! * `_get_experiment_permission(experiment_id, username)` — the workhorse.
//!   Runs / logged models / traces all *inherit* their parent experiment's
//!   permission, so nearly every validator funnels through it after resolving
//!   the owning experiment id.
//! * `_get_role_permission_or_default` — folds the role-derived permission
//!   against the configured `default_permission` floor: `None` (no grant,
//!   workspaces disabled) → default; `NO_PERMISSIONS` → deny; otherwise
//!   `max(role, default)`.
//!
//! ## Workspace scoping (T10.4)
//!
//! Workspace partitioning (T10.4) threads the request's *resolved* workspace
//! (`RequestCtx::workspace`, stamped by the T10.3 layer) into every role lookup
//! and branches the "no grant matched" fold on whether workspaces are enabled
//! (`RequestCtx::workspaces_enabled`, `AppState::workspace_store().is_some()`):
//!
//! * **Workspaces disabled** (single-tenant): a missing grant folds to
//!   `default_permission` — the `_role_permission_for` "workspaces off" branch
//!   returns `None`, never `NO_PERMISSIONS` (`__init__.py:711-712`). Byte-identical
//!   to pre-T10.4.
//! * **Workspaces enabled**: a missing grant in the request's workspace is the
//!   `NO_PERMISSIONS` **boundary deny** (`__init__.py:715`) — NOT
//!   `default_permission`. A resource in a workspace the user has no role in
//!   denies, even where the user would have implicit `default_permission` READ
//!   in single-tenant. The one exception is the opt-in default-workspace
//!   auto-grant (`_user_inherits_default_workspace_grant`, `__init__.py:541`,
//!   `grant_default_workspace_access` off by default): when on **and** the
//!   request workspace is the default workspace, an ungranted user inherits
//!   `default_permission` (`__init__.py:713-714`).
//!
//! Grant lookups are scoped to `RequestCtx::workspace` by
//! `get_role_permission_for_resource(.., workspace)` (T9.3), which already folds
//! `(workspace, *)` MANAGE grants into concrete-resource reads (workspace admin)
//! and USE only for the workspace-tier create-gate — matching
//! `sqlalchemy_store.py:2031-2041`.
//!
//! ## default_permission (T9.8)
//!
//! `default_permission` comes from the parsed [`mlflow_auth::AuthConfig`] carried
//! by the [`AuthStore`] (`AuthStore::config().default_permission`), which the
//! same `basic_auth.ini` drives on both servers. The config validator rejects an
//! unknown permission name at startup, so lookups here are infallible. See
//! [`default_permission`].

use mlflow_auth::permissions::{
    get_permission, max_permission, Permission, ALL_PERMISSIONS, NO_PERMISSIONS,
};
use mlflow_auth::AuthStore;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_store::TrackingStore;
use serde_json::Value;

use crate::workspace::DEFAULT_WORKSPACE_NAME;

/// Everything a validator needs about the current request. Built once by the
/// middleware after buffering the body.
pub struct RequestCtx<'a> {
    pub username: &'a str,
    pub method: &'a str,
    pub workspace: &'a str,
    /// Whether the server runs with workspaces enabled
    /// (`AppState::workspace_store().is_some()` ≈ `MLFLOW_ENABLE_WORKSPACES`).
    /// Gates the `NO_PERMISSIONS` boundary deny (T10.4).
    pub workspaces_enabled: bool,
    /// Captured path parameters (`request.view_args`).
    pub path_params: &'a [(String, String)],
    /// Query string parameters (`request.args`).
    pub query: &'a [(String, String)],
    /// Parsed JSON body, if the body was JSON (`request.get_json(silent=True)`).
    pub json_body: Option<&'a Value>,
    /// Request headers relevant to auth (currently only OTLP experiment id).
    pub experiment_id_header: Option<&'a str>,
    pub auth_store: &'a AuthStore,
    pub tracking_store: &'a TrackingStore,
}

impl RequestCtx<'_> {
    /// `_get_request_param(param)` for the current method: GET reads query;
    /// POST/PATCH read the JSON body; DELETE reads body when JSON else query.
    /// Path params override, mirroring `args | (request.view_args or {})`.
    /// `run_id` falls back to `run_uuid`.
    fn get_param(&self, param: &str) -> Option<String> {
        if let Some((_, v)) = self.path_params.iter().find(|(k, _)| k == param) {
            return Some(v.clone());
        }
        let from_source = self.param_from_source(param);
        if from_source.is_some() {
            return from_source;
        }
        if param == "run_id" {
            return self.get_param("run_uuid");
        }
        None
    }

    fn param_from_source(&self, param: &str) -> Option<String> {
        match self.method {
            "GET" => self.query_param(param),
            "POST" | "PATCH" => self.body_param(param),
            "DELETE" => {
                if self.json_body.is_some() {
                    self.body_param(param)
                } else {
                    self.query_param(param)
                }
            }
            _ => None,
        }
    }

    fn query_param(&self, param: &str) -> Option<String> {
        self.query
            .iter()
            .find(|(k, _)| k == param)
            .map(|(_, v)| v.clone())
    }

    fn body_param(&self, param: &str) -> Option<String> {
        match self.json_body?.get(param)? {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            other => Some(other.to_string()),
        }
    }

    /// All query values for a repeated key (`request.args.to_dict(flat=False)`).
    fn query_multi(&self, param: &str) -> Vec<String> {
        self.query
            .iter()
            .filter(|(k, _)| k == param)
            .map(|(_, v)| v.clone())
            .collect()
    }
}

/// One `validate_can_*` function, dispatched by [`super::path_matchers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validator {
    /// `lambda: True` — authenticated access is enough (e.g. `/users/current`).
    Allow,
    /// `validate_can_update_user_admin` / `validate_can_delete_user` — always
    /// `False` for non-admins (admins bypass upstream).
    AdminOnlyFalse,
    // ---- Experiments ----
    CanCreateExperiment,
    ReadExperiment,
    ReadExperimentByName,
    UpdateExperiment,
    DeleteExperiment,
    // ---- Runs (inherit experiment) ----
    ReadRun,
    UpdateRun,
    DeleteRun,
    // ---- Logged models (inherit experiment) ----
    ReadLoggedModel,
    UpdateLoggedModel,
    DeleteLoggedModel,
    // ---- Registered models / prompts ----
    CanCreateRegisteredModel,
    ReadRegisteredModelOrPrompt,
    UpdateRegisteredModelOrPrompt,
    DeleteRegisteredModelOrPrompt,
    CreateModelVersion,
    // ---- Traces (inherit experiment) ----
    StartTraceV3,
    ReadTraceByRequestId,
    ReadTraceByTraceId,
    UpdateTraceByRequestId,
    UpdateTraceByTraceId,
    SearchTraces,
    SearchTracesV3,
    BatchGetTraces,
    DeleteTraces,
    LinkTracesToRun,
    ReadTracesByExperimentIds,
    ReadTraceArtifact,
    // ---- Artifact plane ----
    ReadRunArtifact,
    UpdateRunArtifact,
    ReadModelVersionArtifact,
    ReadExperimentArtifactProxy,
    UpdateExperimentArtifactProxy,
    DeleteExperimentArtifactProxy,
    // ---- Metrics / datasets ----
    ReadMetricHistoryBulk,
    ReadMetricHistoryBulkInterval,
    SearchDatasets,
    // ---- Label schemas (inherit experiment) ----
    CreateLabelSchema,
    ReadLabelSchema,
    ManageLabelSchema,
    // ---- OTLP ----
    OtlpExperimentUpdate,
    // ---- Users ----
    ReadUser,
    CanListUsers,
    CanCreateUser,
    UpdateUserPassword,
    // ---- RBAC role / per-user grant management ----
    ManageRoles,
    ViewRoles,
    ListRoles,
    ViewUserRoles,
    ManageResource,
    GetUserPermission,
    // ---- Webhooks (admin-only) ----
    SenderIsAdmin,
    // ---- Workspaces (T10.4) ----
    /// `validate_can_view_workspace` — GetWorkspace: workspaces off → True;
    /// on → the `workspace_name` view arg is in the caller's accessible set (or
    /// the default-workspace auto-grant).
    ViewWorkspace,
}

impl Validator {
    /// Run the validator: `true` authorizes, `false` denies (→ 403). A store
    /// error propagates so the middleware can surface the same HTTP status
    /// Python would (`catch_mlflow_exception`).
    pub async fn check(self, ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
        use Validator::*;
        match self {
            Allow => Ok(true),
            AdminOnlyFalse => Ok(false),
            SenderIsAdmin => sender_is_admin(ctx).await,
            // `validate_can_list_users` / `validate_can_create_experiment` /
            // `validate_can_create_registered_model` all delegate to
            // `_user_can_create_in_workspace` (`__init__.py:1208-1213`, `1663-1667`):
            // workspaces off → True; on → a workspace-wide USE/MANAGE grant in the
            // request workspace (or the default-workspace auto-grant).
            CanListUsers | CanCreateExperiment | CanCreateRegisteredModel => {
                user_can_create_in_workspace(ctx).await
            }
            ViewWorkspace => validate_can_view_workspace(ctx).await,
            // Experiments.
            ReadExperiment => Ok(experiment_perm_from_id_param(ctx).await?.can_read),
            ReadExperimentByName => Ok(experiment_perm_from_name(ctx).await?.can_read),
            UpdateExperiment => Ok(experiment_perm_from_id_param(ctx).await?.can_update),
            DeleteExperiment => Ok(experiment_perm_from_id_param(ctx).await?.can_delete),
            // Runs inherit experiment.
            ReadRun => Ok(experiment_perm_from_run(ctx).await?.can_read),
            UpdateRun => Ok(experiment_perm_from_run(ctx).await?.can_update),
            DeleteRun => Ok(experiment_perm_from_run(ctx).await?.can_delete),
            // Logged models inherit experiment.
            ReadLoggedModel => Ok(experiment_perm_from_model(ctx).await?.can_read),
            UpdateLoggedModel => Ok(experiment_perm_from_model(ctx).await?.can_update),
            DeleteLoggedModel => Ok(experiment_perm_from_model(ctx).await?.can_delete),
            // Registered models / prompts.
            ReadRegisteredModelOrPrompt => Ok(registered_model_perm(ctx).await?.can_read),
            UpdateRegisteredModelOrPrompt => Ok(registered_model_perm(ctx).await?.can_update),
            DeleteRegisteredModelOrPrompt => Ok(registered_model_perm(ctx).await?.can_delete),
            CreateModelVersion => validate_can_create_model_version(ctx).await,
            // Traces.
            StartTraceV3 => validate_start_trace_v3(ctx).await,
            ReadTraceByRequestId => Ok(trace_perm(ctx, "request_id").await?.can_read),
            ReadTraceByTraceId => Ok(trace_perm(ctx, "trace_id").await?.can_read),
            UpdateTraceByRequestId => Ok(trace_perm(ctx, "request_id").await?.can_update),
            UpdateTraceByTraceId => Ok(trace_perm(ctx, "trace_id").await?.can_update),
            SearchTraces => validate_search_traces(ctx).await,
            SearchTracesV3 => validate_search_traces_v3(ctx).await,
            BatchGetTraces => validate_batch_get_traces(ctx).await,
            DeleteTraces => Ok(experiment_perm_from_id_param(ctx).await?.can_delete),
            LinkTracesToRun => validate_link_traces_to_run(ctx).await,
            ReadTracesByExperimentIds => validate_read_traces_by_experiment_ids(ctx).await,
            ReadTraceArtifact => Ok(trace_perm_from_query(ctx).await?.can_read),
            // Artifact plane (inherit experiment via run / model version).
            ReadRunArtifact => Ok(experiment_perm_from_run(ctx).await?.can_read),
            UpdateRunArtifact => Ok(experiment_perm_from_run(ctx).await?.can_update),
            ReadModelVersionArtifact => Ok(registered_model_perm(ctx).await?.can_read),
            ReadExperimentArtifactProxy => Ok(artifact_proxy_perm(ctx).await?.can_read),
            UpdateExperimentArtifactProxy => Ok(artifact_proxy_perm(ctx).await?.can_update),
            DeleteExperimentArtifactProxy => Ok(artifact_proxy_perm(ctx).await?.can_manage),
            // Metrics / datasets (read the experiment/run).
            ReadMetricHistoryBulk => validate_metric_history_bulk(ctx, "run_id").await,
            ReadMetricHistoryBulkInterval => validate_metric_history_bulk(ctx, "run_ids").await,
            SearchDatasets => validate_search_datasets(ctx).await,
            CreateLabelSchema => Ok(experiment_perm_from_id_param(ctx).await?.can_manage),
            ReadLabelSchema => Ok(experiment_perm_from_label_schema(ctx).await?.can_read),
            ManageLabelSchema => Ok(experiment_perm_from_label_schema(ctx).await?.can_manage),
            // OTLP: experiment UPDATE from the X-Mlflow-Experiment-Id header.
            OtlpExperimentUpdate => validate_otlp(ctx).await,
            // Users.
            ReadUser => username_is_sender(ctx),
            CanCreateUser => validate_can_create_user(ctx).await,
            UpdateUserPassword => username_is_sender(ctx),
            ManageRoles => validate_can_manage_roles(ctx).await,
            ViewRoles => validate_can_view_roles(ctx).await,
            ListRoles => validate_can_list_roles(ctx).await,
            ViewUserRoles => validate_can_view_user_roles(ctx).await,
            ManageResource => validate_can_manage_resource(ctx).await,
            GetUserPermission => validate_can_get_user_permission(ctx).await,
        }
    }
}

// ---- Permission resolution ----

/// `default_permission` — the configured floor, read from the parsed
/// [`mlflow_auth::AuthConfig`] on the given [`AuthStore`]
/// (`AuthStore::config().default_permission`, T9.8). Store-based (not
/// [`RequestCtx`]-based) so the GraphQL auth gate (T9.6) resolves the same
/// floor through [`resolve_experiment_permission`] /
/// [`resolve_registered_model_permission`]. The config validator guarantees
/// the name is a known permission, so the lookup is defensively clamped to
/// `READ` only in the never-taken invalid branch.
///
/// `pub(crate)` so the after-request read predicate (T9.5) reads the same floor.
pub(crate) fn default_permission(auth_store: &AuthStore) -> &'static Permission {
    let name = auth_store.config().default_permission.as_str();
    if ALL_PERMISSIONS.iter().any(|p| p.name == name) {
        get_permission(name)
    } else {
        get_permission("READ")
    }
}

/// `_get_role_permission_or_default` (`__init__.py:556`): fold the role-derived
/// permission against `default_permission`. `None` → default; `NO_PERMISSIONS`
/// stays a deny; otherwise `max(role, default)`.
fn fold_default(
    auth_store: &AuthStore,
    role_perm: Option<&'static Permission>,
) -> &'static Permission {
    let default = default_permission(auth_store);
    match role_perm {
        None => default,
        Some(p) if p.name == NO_PERMISSIONS.name => p,
        Some(p) => get_permission(max_permission(p.name, default.name)),
    }
}

/// `_user_inherits_default_workspace_grant` (`__init__.py:541`): the opt-in
/// default-workspace auto-grant. True iff `grant_default_workspace_access` is on
/// **and** the request workspace is the default workspace. Python resolves the
/// live default workspace via `get_default_workspace_optional`; a workspace can
/// only be *made* default (via `?mode=SET_DEFAULT` on delete) but the auth
/// resolver's grant is keyed on that live default name — for the standard
/// deployment the default workspace is [`DEFAULT_WORKSPACE_NAME`], so we compare
/// against it. (The rename-to-default corner is not exercised by the ACs.)
fn user_inherits_default_workspace_grant(auth_store: &AuthStore, workspace: &str) -> bool {
    auth_store.config().grant_default_workspace_access && workspace == DEFAULT_WORKSPACE_NAME
}

/// The shared inner resolver (`_role_permission_for` closure + fold,
/// `__init__.py:672-717`, `750-760`): look up the role grant on
/// `(resource_type, resource_id)` scoped to `workspace`, then fold it against
/// `default_permission` with the workspaces-enabled boundary deny.
///
/// * grant found → `max(grant, default)` (default is a floor, never a downgrade;
///   `NO_PERMISSIONS` is preserved).
/// * no grant, workspaces **disabled** → `default_permission` (single-tenant).
/// * no grant, workspaces **enabled** → `NO_PERMISSIONS` boundary deny, unless
///   the default-workspace auto-grant applies (→ `default_permission`).
pub(crate) async fn resolve_role_permission(
    auth_store: &AuthStore,
    username: &str,
    workspace: &str,
    workspaces_enabled: bool,
    resource_type: &str,
    resource_id: &str,
) -> Result<&'static Permission, MlflowError> {
    let user = auth_store.get_user(username).await?;
    let role_perm = auth_store
        .get_role_permission_for_resource(user.id, resource_type, resource_id, workspace)
        .await?;
    let inner = match role_perm {
        Some(p) => Some(p),
        None if !workspaces_enabled => None,
        None if user_inherits_default_workspace_grant(auth_store, workspace) => {
            Some(default_permission(auth_store))
        }
        None => Some(&NO_PERMISSIONS),
    };
    Ok(fold_default(auth_store, inner))
}

// ---- RBAC role / per-user permission APIs ----

async fn role_workspace_from_request(ctx: &RequestCtx<'_>) -> Result<Option<String>, MlflowError> {
    if let Some(raw) = ctx.get_param("role_id") {
        let Ok(role_id) = raw.parse::<i64>() else {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {raw:?} for parameter 'role_id'."
            )));
        };
        return match ctx.auth_store.get_role(role_id).await {
            Ok(role) => Ok(Some(role.workspace)),
            Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => Ok(None),
            Err(e) => Err(e),
        };
    }
    if let Some(raw) = ctx.get_param("role_permission_id") {
        let Ok(permission_id) = raw.parse::<i64>() else {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {raw:?} for parameter 'role_permission_id'."
            )));
        };
        let permission = match ctx.auth_store.get_role_permission(permission_id).await {
            Ok(permission) => permission,
            Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => return Ok(None),
            Err(e) => return Err(e),
        };
        return match ctx.auth_store.get_role(permission.role_id).await {
            Ok(role) => Ok(Some(role.workspace)),
            Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => Ok(None),
            Err(e) => Err(e),
        };
    }
    if let Some(workspace) = ctx.get_param("workspace") {
        if workspace.trim().is_empty() {
            return Err(MlflowError::invalid_parameter_value(
                "Parameter 'workspace' must be a non-empty string.",
            ));
        }
        return Ok(Some(workspace));
    }
    Err(MlflowError::invalid_parameter_value(
        "Request must include one of: role_id, role_permission_id, workspace.",
    ))
}

async fn validate_can_manage_roles(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let Some(workspace) = role_workspace_from_request(ctx).await? else {
        return Ok(false);
    };
    let user = ctx.auth_store.get_user(ctx.username).await?;
    ctx.auth_store.is_workspace_admin(user.id, &workspace).await
}

async fn validate_can_view_roles(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let Some(workspace) = role_workspace_from_request(ctx).await? else {
        return Ok(false);
    };
    let user = ctx.auth_store.get_user(ctx.username).await?;
    Ok(!ctx
        .auth_store
        .list_user_roles_for_workspace(user.id, &workspace)
        .await?
        .is_empty())
}

async fn validate_can_list_roles(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let requested: std::collections::HashSet<String> = ctx
        .query_multi("workspace")
        .into_iter()
        .filter_map(|workspace| {
            let trimmed = workspace.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect();
    if requested.is_empty() {
        return Ok(false);
    }
    let user = ctx.auth_store.get_user(ctx.username).await?;
    for workspace in requested {
        if ctx
            .auth_store
            .list_user_roles_for_workspace(user.id, &workspace)
            .await?
            .is_empty()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn validate_can_view_user_roles(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let target_username = require_param(ctx, "username")?;
    if target_username == ctx.username {
        return Ok(true);
    }
    let requester = ctx.auth_store.get_user(ctx.username).await?;
    let target = match ctx.auth_store.get_user(&target_username).await {
        Ok(target) => target,
        Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => return Ok(false),
        Err(e) => return Err(e),
    };
    let admin_workspaces = ctx
        .auth_store
        .list_workspace_admin_workspaces(requester.id)
        .await?;
    if admin_workspaces.is_empty() {
        return Ok(false);
    }
    let target_roles = ctx.auth_store.list_user_roles(target.id).await?;
    Ok(target_roles
        .iter()
        .any(|role| admin_workspaces.contains(&role.workspace)))
}

fn reject_workspace_resource_type(resource_type: &str) -> Result<(), MlflowError> {
    if resource_type == "workspace" {
        return Err(MlflowError::invalid_parameter_value(
            "resource_type 'workspace' is not supported by the per-user permission convenience \
             APIs. Use set_workspace_permission / delete_workspace_permission for workspace-wide \
             grants.",
        ));
    }
    Ok(())
}

async fn validate_can_manage_resource(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let resource_type = require_param(ctx, "resource_type")?;
    let resource_id = require_param(ctx, "resource_id")?;
    reject_workspace_resource_type(&resource_type)?;
    mlflow_auth::permissions::validate_resource_type(&resource_type)?;
    if resource_type == "scorer" && !resource_id.contains('/') {
        return Err(MlflowError::invalid_parameter_value(
            "Invalid scorer resource_id. Expected '<experiment_id>/<scorer_name>'.",
        ));
    }
    Ok(resolve_role_permission(
        ctx.auth_store,
        ctx.username,
        ctx.workspace,
        ctx.workspaces_enabled,
        &resource_type,
        &resource_id,
    )
    .await?
    .can_manage)
}

async fn validate_can_get_user_permission(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let resource_type = require_param(ctx, "resource_type")?;
    reject_workspace_resource_type(&resource_type)?;
    let target_username = require_param(ctx, "username")?;
    if target_username == ctx.username {
        return Ok(true);
    }
    if let Err(e) = ctx.auth_store.get_user(&target_username).await {
        return if e.error_code == ErrorCode::ResourceDoesNotExist {
            Ok(false)
        } else {
            Err(e)
        };
    }
    let resource_id = require_param(ctx, "resource_id")?;
    let workspace = match resource_type.as_str() {
        "experiment" => match ctx
            .tracking_store
            .get_experiment(ctx.workspace, &resource_id)
            .await
        {
            Ok(_) => ctx.workspace,
            // `_workspace_for_resource(..., silent=True)` is fail-closed for
            // every lookup failure, including malformed experiment ids.
            Err(_) => return Ok(false),
        },
        "scorer" => {
            let Some((experiment_id, _)) = resource_id.split_once('/') else {
                return Ok(false);
            };
            match ctx
                .tracking_store
                .get_experiment(ctx.workspace, experiment_id)
                .await
            {
                Ok(_) => ctx.workspace,
                Err(_) => return Ok(false),
            }
        }
        _ => return Ok(false),
    };
    let requester = ctx.auth_store.get_user(ctx.username).await?;
    ctx.auth_store
        .is_workspace_admin(requester.id, workspace)
        .await
}

/// `_get_experiment_permission(experiment_id, username)` (`__init__.py:750`): the
/// role grant scoped to `workspace`, folded with the workspaces-enabled boundary
/// deny (see [`resolve_role_permission`]). Store-level primitive, decoupled from
/// [`RequestCtx`] so both the before-request validators and the GraphQL auth gate
/// (T9.6) call it.
pub(crate) async fn resolve_experiment_permission(
    auth_store: &AuthStore,
    username: &str,
    workspace: &str,
    workspaces_enabled: bool,
    experiment_id: &str,
) -> Result<&'static Permission, MlflowError> {
    resolve_role_permission(
        auth_store,
        username,
        workspace,
        workspaces_enabled,
        "experiment",
        experiment_id,
    )
    .await
}

/// `_graphql_get_permission_for_model` (`__init__.py:4114`): the
/// `registered_model` role grant scoped to `workspace`, folded with the
/// workspaces-enabled boundary deny. Shared with [`registered_model_perm`].
pub(crate) async fn resolve_registered_model_permission(
    auth_store: &AuthStore,
    username: &str,
    workspace: &str,
    workspaces_enabled: bool,
    model_name: &str,
) -> Result<&'static Permission, MlflowError> {
    resolve_role_permission(
        auth_store,
        username,
        workspace,
        workspaces_enabled,
        "registered_model",
        model_name,
    )
    .await
}

async fn experiment_permission(
    ctx: &RequestCtx<'_>,
    experiment_id: &str,
) -> Result<&'static Permission, MlflowError> {
    resolve_experiment_permission(
        ctx.auth_store,
        ctx.username,
        ctx.workspace,
        ctx.workspaces_enabled,
        experiment_id,
    )
    .await
}

async fn experiment_perm_from_id_param(
    ctx: &RequestCtx<'_>,
) -> Result<&'static Permission, MlflowError> {
    let experiment_id = require_param(ctx, "experiment_id")?;
    experiment_permission(ctx, &experiment_id).await
}

async fn experiment_perm_from_name(
    ctx: &RequestCtx<'_>,
) -> Result<&'static Permission, MlflowError> {
    let name = require_param(ctx, "experiment_name")?;
    let exp = ctx
        .tracking_store
        .get_experiment_by_name(ctx.workspace, &name)
        .await?
        .ok_or_else(|| {
            MlflowError::resource_does_not_exist(format!(
                "Could not find experiment with name {name}"
            ))
        })?;
    experiment_permission(ctx, &exp.experiment_id).await
}

async fn experiment_perm_from_run(
    ctx: &RequestCtx<'_>,
) -> Result<&'static Permission, MlflowError> {
    let run_id = require_param(ctx, "run_id")?;
    let run = ctx.tracking_store.get_run(ctx.workspace, &run_id).await?;
    experiment_permission(ctx, &run.info.experiment_id).await
}

async fn experiment_perm_from_model(
    ctx: &RequestCtx<'_>,
) -> Result<&'static Permission, MlflowError> {
    let model_id = require_param(ctx, "model_id")?;
    let model = ctx
        .tracking_store
        .get_logged_model(ctx.workspace, &model_id, true)
        .await?;
    experiment_permission(ctx, &model.experiment_id).await
}

async fn experiment_perm_from_label_schema(
    ctx: &RequestCtx<'_>,
) -> Result<&'static Permission, MlflowError> {
    let schema_id = require_param(ctx, "schema_id")?;
    let schema = ctx
        .tracking_store
        .get_label_schema(ctx.workspace, &schema_id)
        .await?;
    experiment_permission(ctx, &schema.experiment_id).await
}

/// `_get_permission_from_registered_model_or_prompt_name` on the
/// workspaces-disabled path. The registry store lands in a later phase; until
/// then the resolver falls to `default_permission` (a missing/unresolvable name
/// is Python's `RESOURCE_DOES_NOT_EXIST` → `workspace_name = None` →
/// `None` grant → default). This matches Python's behavior for a not-yet-created
/// model and is the conservative fallback for the model-registry routes.
async fn registered_model_perm(ctx: &RequestCtx<'_>) -> Result<&'static Permission, MlflowError> {
    let name = require_param(ctx, "name")?;
    // Namespaced under `registered_model`; a real prompt/model classification
    // needs the registry store (later phase). Resolve the grant directly.
    resolve_registered_model_permission(
        ctx.auth_store,
        ctx.username,
        ctx.workspace,
        ctx.workspaces_enabled,
        &name,
    )
    .await
}

/// `validate_can_create_model_version` (`__init__.py:1188`): require UPDATE on
/// the registered model, plus READ on the source `run_id`/`model_id` when
/// present (guard on **presence**, not truthiness — an explicit empty id is a
/// deny). This is the "MV create dual requirement".
async fn validate_can_create_model_version(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    if !registered_model_perm(ctx).await?.can_update {
        return Ok(false);
    }
    let body = ctx.json_body;
    // `run_id` present.
    if let Some(run_id_val) = body.and_then(|b| b.get("run_id")) {
        let run_id = run_id_val.as_str().unwrap_or("");
        if run_id.is_empty() {
            return Ok(false);
        }
        let run = ctx.tracking_store.get_run(ctx.workspace, run_id).await?;
        if !experiment_permission(ctx, &run.info.experiment_id)
            .await?
            .can_read
        {
            return Ok(false);
        }
    }
    // `model_id` present.
    if let Some(model_id_val) = body.and_then(|b| b.get("model_id")) {
        let model_id = model_id_val.as_str().unwrap_or("");
        if model_id.is_empty() {
            return Ok(false);
        }
        let model = ctx
            .tracking_store
            .get_logged_model(ctx.workspace, model_id, true)
            .await?;
        if !experiment_permission(ctx, &model.experiment_id)
            .await?
            .can_read
        {
            return Ok(false);
        }
    }
    Ok(true)
}

// ---- Traces ----

/// `_get_permission_from_trace(trace_id, username)` (`__init__.py:1958`): look
/// up the trace's experiment; a `RESOURCE_DOES_NOT_EXIST` becomes
/// `NO_PERMISSIONS` (deny), other errors propagate.
async fn trace_perm(
    ctx: &RequestCtx<'_>,
    id_param: &str,
) -> Result<&'static Permission, MlflowError> {
    let trace_id = require_param(ctx, id_param)?;
    resolve_trace_perm(ctx, &trace_id).await
}

/// `validate_can_read_trace_artifact`: the `request_id` comes from the query
/// (`request.args.get("request_id")`), and a missing one is a 400.
async fn trace_perm_from_query(ctx: &RequestCtx<'_>) -> Result<&'static Permission, MlflowError> {
    let request_id = ctx.query_param("request_id").ok_or_else(|| {
        MlflowError::invalid_parameter_value("Request must specify request_id parameter")
    })?;
    // Unlike `_get_permission_from_trace`, `_get_permission_from_trace_request_id`
    // lets a missing trace error propagate.
    let trace = ctx
        .tracking_store
        .get_trace_info(ctx.workspace, &request_id)
        .await?;
    experiment_permission(ctx, &trace.experiment_id).await
}

async fn resolve_trace_perm(
    ctx: &RequestCtx<'_>,
    trace_id: &str,
) -> Result<&'static Permission, MlflowError> {
    match ctx
        .tracking_store
        .get_trace_info(ctx.workspace, trace_id)
        .await
    {
        Ok(trace) => experiment_permission(ctx, &trace.experiment_id).await,
        Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => Ok(&NO_PERMISSIONS),
        Err(e) => Err(e),
    }
}

/// `validate_can_search_traces` (`__init__.py:1980`): `experiment_ids` from the
/// query; non-empty and READ on all.
async fn validate_search_traces(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let ids = ctx.query_multi("experiment_ids");
    all_can_read_experiments(ctx, &ids).await
}

/// `validate_can_search_traces_v3` (`__init__.py:1988`): experiment ids come
/// from `locations[].mlflow_experiment.experiment_id` in the body; fail-closed
/// when none map to a local experiment.
async fn validate_search_traces_v3(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let ids = ctx
        .json_body
        .and_then(|b| b.get("locations"))
        .and_then(Value::as_array)
        .map(|locs| {
            locs.iter()
                .filter_map(|loc| {
                    loc.get("mlflow_experiment")?
                        .get("experiment_id")?
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    all_can_read_experiments(ctx, &ids).await
}

/// `validate_can_batch_get_traces` (`__init__.py:2007`): resolve every
/// `trace_ids` entry to its experiment (GET query or body), READ on all.
async fn validate_batch_get_traces(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let trace_ids: Vec<String> = if ctx.method == "GET" {
        ctx.query_multi("trace_ids")
    } else {
        ctx.json_body
            .and_then(|b| b.get("trace_ids"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default()
    };
    let mut experiment_ids = Vec::new();
    for tid in &trace_ids {
        match ctx.tracking_store.get_trace_info(ctx.workspace, tid).await {
            Ok(t) => experiment_ids.push(t.experiment_id),
            Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => return Ok(false),
            Err(e) => return Err(e),
        }
    }
    all_can_read_experiments(ctx, &experiment_ids).await
}

/// `validate_can_link_traces_to_run` (`__init__.py:2064`): UPDATE on the
/// destination run's experiment and READ on every source trace's experiment.
/// A missing run/trace or an empty trace list is a deny, matching Python's
/// fail-closed validator.
async fn validate_link_traces_to_run(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let run_id = require_param(ctx, "run_id")?;
    let run = match ctx.tracking_store.get_run(ctx.workspace, &run_id).await {
        Ok(run) => run,
        Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => return Ok(false),
        Err(e) => return Err(e),
    };
    if !experiment_permission(ctx, &run.info.experiment_id)
        .await?
        .can_update
    {
        return Ok(false);
    }

    let trace_ids = ctx
        .json_body
        .and_then(|b| b.get("trace_ids"))
        .and_then(Value::as_array)
        .map(|ids| ids.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    if trace_ids.is_empty() {
        return Ok(false);
    }
    for trace_id in trace_ids {
        let trace = match ctx
            .tracking_store
            .get_trace_info(ctx.workspace, trace_id)
            .await
        {
            Ok(trace) => trace,
            Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => return Ok(false),
            Err(e) => return Err(e),
        };
        if !experiment_permission(ctx, &trace.experiment_id)
            .await?
            .can_read
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `validate_can_read_traces_by_experiment_ids` (`__init__.py:2043`):
/// `experiment_ids` from the body; non-empty and READ on all.
async fn validate_read_traces_by_experiment_ids(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let ids = ctx
        .json_body
        .and_then(|b| b.get("experiment_ids"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    all_can_read_experiments(ctx, &ids).await
}

/// `validate_can_start_trace_v3` (`__init__.py:2051`): the experiment id is
/// nested at `trace.trace_info.trace_location.mlflow_experiment.experiment_id`;
/// UPDATE on it, else deny.
async fn validate_start_trace_v3(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let eid = ctx
        .json_body
        .and_then(|b| b.get("trace"))
        .and_then(|t| t.get("trace_info"))
        .and_then(|ti| ti.get("trace_location"))
        .and_then(|tl| tl.get("mlflow_experiment"))
        .and_then(|me| me.get("experiment_id"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    match eid {
        Some(eid) => Ok(experiment_permission(ctx, eid).await?.can_update),
        None => Ok(false),
    }
}

/// `validate_can_read_metric_history_bulk` (`__init__.py:2090`): READ on every
/// requested run's experiment; non-empty required. The older bulk route uses
/// repeated `run_id`, while bulk-interval uses repeated `run_ids`.
async fn validate_metric_history_bulk(
    ctx: &RequestCtx<'_>,
    query_param: &str,
) -> Result<bool, MlflowError> {
    let run_ids = ctx.query_multi(query_param);
    if run_ids.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "GetMetricHistoryBulk request must specify at least one run_id.",
        ));
    }
    for run_id in &run_ids {
        let run = ctx.tracking_store.get_run(ctx.workspace, run_id).await?;
        if !experiment_permission(ctx, &run.info.experiment_id)
            .await?
            .can_read
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `validate_can_search_datasets` (`__init__.py:2138`): READ on every
/// `experiment_ids` entry (from the body).
async fn validate_search_datasets(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let ids = ctx
        .json_body
        .and_then(|b| b.get("experiment_ids"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    all_can_read_experiments(ctx, &ids).await
}

/// `_get_otel_validator` (`__init__.py:4441`): the experiment id comes from the
/// `X-Mlflow-Experiment-Id` header (a missing one is a 400), UPDATE required.
async fn validate_otlp(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let eid = ctx
        .experiment_id_header
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            MlflowError::new(
                "Missing required header: X-Mlflow-Experiment-Id",
                ErrorCode::BadRequest,
            )
        })?;
    Ok(experiment_permission(ctx, eid).await?.can_update)
}

/// `_get_permission_from_experiment_id_artifact_proxy` (`__init__.py:780-806`):
/// extract the experiment id from the artifact tail (path param or `?path=`
/// query) and resolve as an experiment permission. When no id is found:
///
/// * workspaces **disabled** → `default_permission`.
/// * workspaces **enabled** → the request-workspace `(workspace, *)` grant if
///   any (with the default-workspace auto-grant fallback), else `NO_PERMISSIONS`
///   — the coarse workspace-tier grant Python falls back to for a bare
///   proxy-artifact path.
async fn artifact_proxy_perm(ctx: &RequestCtx<'_>) -> Result<&'static Permission, MlflowError> {
    // `artifact_path` (view arg) or `path` (query), then `_EXPERIMENT_ID_PATTERN`.
    let artifact_path = ctx
        .path_params
        .iter()
        .find(|(k, _)| k == "artifact_path")
        .map(|(_, v)| v.clone())
        .or_else(|| ctx.query_param("path"));
    if let Some(ap) = artifact_path {
        if let Some(eid) = super::path_matchers::experiment_id_from_artifact_path(&ap) {
            return experiment_permission(ctx, &eid).await;
        }
    }
    if !ctx.workspaces_enabled {
        // No experiment id resolved: workspaces-disabled → default_permission.
        return Ok(default_permission(ctx.auth_store));
    }
    // Workspaces enabled: fall back to the coarse workspace-tier grant on
    // `(workspace, *)` in the request workspace.
    let user = ctx.auth_store.get_user(ctx.username).await?;
    let perm = ctx
        .auth_store
        .get_role_permission_for_resource(user.id, "workspace", "*", ctx.workspace)
        .await?;
    if let Some(p) = perm {
        return Ok(p);
    }
    if user_inherits_default_workspace_grant(ctx.auth_store, ctx.workspace) {
        return Ok(default_permission(ctx.auth_store));
    }
    Ok(&NO_PERMISSIONS)
}

async fn all_can_read_experiments(
    ctx: &RequestCtx<'_>,
    experiment_ids: &[String],
) -> Result<bool, MlflowError> {
    if experiment_ids.is_empty() {
        return Ok(false);
    }
    for eid in experiment_ids {
        if !experiment_permission(ctx, eid).await?.can_read {
            return Ok(false);
        }
    }
    Ok(true)
}

// ---- Users ----

/// `username_is_sender` (`__init__.py:1652`): reads the `username` param via
/// `_get_request_param`, which raises the missing-param 400 when absent (rather
/// than denying) — so a `null`/empty body yields 400, not 403.
fn username_is_sender(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    Ok(require_param(ctx, "username")? == ctx.username)
}

/// `validate_can_create_user` (`__init__.py:1670`): non-admins may create a
/// user only if they are a workspace admin somewhere. (Admins bypass upstream.)
async fn validate_can_create_user(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let user = ctx.auth_store.get_user(ctx.username).await?;
    Ok(!ctx
        .auth_store
        .list_workspace_admin_workspaces(user.id)
        .await?
        .is_empty())
}

/// `sender_is_admin` (`__init__.py:1260`).
async fn sender_is_admin(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    Ok(ctx.auth_store.get_user(ctx.username).await?.is_admin)
}

/// `_user_can_create_in_workspace` (`__init__.py:580-610`): workspaces off →
/// always True. On → a workspace-wide grant with `can_use` (USE/MANAGE stored on
/// `(workspace, *)`) in the request workspace; resource-specific grants don't
/// confer create. Honors the default-workspace auto-grant: an ungranted user in
/// the default workspace inherits `default_permission` and can create iff it
/// carries `can_use`.
async fn user_can_create_in_workspace(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    if !ctx.workspaces_enabled {
        return Ok(true);
    }
    let user = ctx.auth_store.get_user(ctx.username).await?;
    let perm = ctx
        .auth_store
        .get_role_permission_for_resource(user.id, "workspace", "*", ctx.workspace)
        .await?;
    if let Some(p) = perm {
        return Ok(p.can_use);
    }
    if user_inherits_default_workspace_grant(ctx.auth_store, ctx.workspace) {
        return Ok(default_permission(ctx.auth_store).can_use);
    }
    Ok(false)
}

/// `validate_can_view_workspace` (`__init__.py:1216-1236`): workspaces off →
/// True. On → the `workspace_name` view arg must be present and in the caller's
/// accessible-workspace set (or the default-workspace auto-grant when the target
/// is the default workspace). Admins bypass upstream in the middleware.
async fn validate_can_view_workspace(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    if !ctx.workspaces_enabled {
        return Ok(true);
    }
    let Some((_, workspace_name)) = ctx.path_params.iter().find(|(k, _)| k == "workspace_name")
    else {
        return Ok(false);
    };
    if user_inherits_default_workspace_grant(ctx.auth_store, workspace_name) {
        return Ok(true);
    }
    let names = ctx
        .auth_store
        .list_accessible_workspace_names(ctx.username)
        .await?;
    Ok(names.contains(workspace_name.as_str()))
}

/// `_get_request_param` that raises the standard missing-param 400.
fn require_param(ctx: &RequestCtx<'_>, param: &str) -> Result<String, MlflowError> {
    ctx.get_param(param).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "Missing value for required parameter '{param}'. \
             See the API docs for more information about request parameters."
        ))
    })
}
