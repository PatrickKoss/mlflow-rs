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
//! ## Workspace scoping (T10.4 seam)
//!
//! Workspace partitioning is T10.4. Pre-T10.4 the server is single-tenant, so
//! we resolve every role lookup in the `"default"` workspace — exactly what
//! Python does when `MLFLOW_ENABLE_WORKSPACES` is off (the `_role_permission_for`
//! "workspaces disabled" branch returns the raw grant or `None`, never
//! `NO_PERMISSIONS`). The request's `X-MLFLOW-WORKSPACE` header is carried
//! through so the wiring is ready, but it always resolves to `"default"` here.
//!
//! ## default_permission (T9.8 seam)
//!
//! The ini-file config layer is T9.8. Until then, `default_permission` is read
//! from the `MLFLOW_AUTH_DEFAULT_PERMISSION` env var, defaulting to `"READ"`
//! (the packaged `basic_auth.ini` default). See [`default_permission`].

use mlflow_auth::permissions::{
    get_permission, max_permission, Permission, ALL_PERMISSIONS, NO_PERMISSIONS,
};
use mlflow_auth::AuthStore;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_store::TrackingStore;
use serde_json::Value;

/// Everything a validator needs about the current request. Built once by the
/// middleware after buffering the body.
pub struct RequestCtx<'a> {
    pub username: &'a str,
    pub method: &'a str,
    pub workspace: &'a str,
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
    // ---- OTLP ----
    OtlpExperimentUpdate,
    // ---- Users ----
    ReadUser,
    CanListUsers,
    CanCreateUser,
    UpdateUserPassword,
    // ---- Webhooks (admin-only) ----
    SenderIsAdmin,
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
            CanListUsers => Ok(true), // `_user_can_create_in_workspace` (workspaces off → True).
            CanCreateExperiment | CanCreateRegisteredModel => Ok(true), // ditto.
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
            LinkTracesToRun => Ok(experiment_perm_from_run(ctx).await?.can_update),
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
            ReadMetricHistoryBulk => validate_metric_history_bulk(ctx).await,
            ReadMetricHistoryBulkInterval => Ok(experiment_perm_from_run(ctx).await?.can_read),
            SearchDatasets => validate_search_datasets(ctx).await,
            // OTLP: experiment UPDATE from the X-Mlflow-Experiment-Id header.
            OtlpExperimentUpdate => validate_otlp(ctx).await,
            // Users.
            ReadUser => username_is_sender(ctx),
            CanCreateUser => validate_can_create_user(ctx).await,
            UpdateUserPassword => username_is_sender(ctx),
        }
    }
}

// ---- Permission resolution ----

/// `default_permission` — T9.8 SEAM: reads `MLFLOW_AUTH_DEFAULT_PERMISSION`
/// (default `"READ"`, the packaged `basic_auth.ini` value). The full ini-file
/// config layer (`read_auth_config`) lands in T9.8.
///
/// Exposed `pub(crate)` so the GraphQL auth gate (T9.6) resolves the same
/// default-permission floor without re-deriving it.
pub(crate) fn default_permission() -> &'static Permission {
    // T9.8 SEAM: replace this env fallback with `AuthConfig.default_permission`.
    let name = std::env::var("MLFLOW_AUTH_DEFAULT_PERMISSION").unwrap_or_else(|_| "READ".into());
    // An invalid value falls back to READ rather than panicking (defensive; the
    // ini validator will reject bad values once T9.8 wires config).
    if ALL_PERMISSIONS.iter().any(|p| p.name == name) {
        get_permission(&name)
    } else {
        get_permission("READ")
    }
}

/// `_get_role_permission_or_default` (`__init__.py:556`): fold the role-derived
/// permission against `default_permission`. `None` → default; `NO_PERMISSIONS`
/// stays a deny; otherwise `max(role, default)`.
///
/// `pub(crate)` for reuse by the GraphQL auth gate (T9.6).
pub(crate) fn fold_default(role_perm: Option<&'static Permission>) -> &'static Permission {
    let default = default_permission();
    match role_perm {
        None => default,
        Some(p) if p.name == NO_PERMISSIONS.name => p,
        Some(p) => get_permission(max_permission(p.name, default.name)),
    }
}

/// `_get_experiment_permission(experiment_id, username)` (`__init__.py:750`),
/// on the workspaces-disabled path: the role grant if any, else default. The
/// store-level primitive, decoupled from [`RequestCtx`] so both the
/// before-request validators and the GraphQL auth gate (T9.6) call it.
pub(crate) async fn resolve_experiment_permission(
    auth_store: &AuthStore,
    username: &str,
    workspace: &str,
    experiment_id: &str,
) -> Result<&'static Permission, MlflowError> {
    let user = auth_store.get_user(username).await?;
    let role_perm = auth_store
        .get_role_permission_for_resource(user.id, "experiment", experiment_id, workspace)
        .await?;
    Ok(fold_default(role_perm))
}

/// `_graphql_get_permission_for_model` (`__init__.py:4114`) on the
/// workspaces-disabled path: the `registered_model` role grant if any, else
/// default. Shared with [`registered_model_perm`].
pub(crate) async fn resolve_registered_model_permission(
    auth_store: &AuthStore,
    username: &str,
    workspace: &str,
    model_name: &str,
) -> Result<&'static Permission, MlflowError> {
    let user = auth_store.get_user(username).await?;
    let role_perm = auth_store
        .get_role_permission_for_resource(user.id, "registered_model", model_name, workspace)
        .await?;
    Ok(fold_default(role_perm))
}

async fn experiment_permission(
    ctx: &RequestCtx<'_>,
    experiment_id: &str,
) -> Result<&'static Permission, MlflowError> {
    resolve_experiment_permission(ctx.auth_store, ctx.username, ctx.workspace, experiment_id).await
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
    resolve_registered_model_permission(ctx.auth_store, ctx.username, ctx.workspace, &name).await
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
/// `run_ids` entry's experiment; non-empty required.
async fn validate_metric_history_bulk(ctx: &RequestCtx<'_>) -> Result<bool, MlflowError> {
    let run_ids = ctx.query_multi("run_ids");
    if run_ids.is_empty() {
        return Ok(false);
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

/// `_get_permission_from_experiment_id_artifact_proxy` on the workspaces-disabled
/// path: extract the experiment id from the artifact tail (path param or `?path=`
/// query) and resolve; if no id is found, fall to `default_permission`.
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
    // No experiment id resolved: workspaces-disabled → default_permission.
    Ok(default_permission())
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

/// `_get_request_param` that raises the standard missing-param 400.
fn require_param(ctx: &RequestCtx<'_>, param: &str) -> Result<String, MlflowError> {
    ctx.get_param(param).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "Missing value for required parameter '{param}'. \
             See the API docs for more information about request parameters."
        ))
    })
}
