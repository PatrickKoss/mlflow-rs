//! Experiment endpoints (plan T3.1, §3.1): create, get, get-by-name, search
//! (POST + GET), delete, restore, update, set-tag, delete-tag.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`:
//! parse the request proto (via [`crate::proto_http`]), run the handler-level
//! schema validation (required fields — Python's `_validate_request_json_with_schema`),
//! call the workspace-scoped store method, then serialize the response proto.
//!
//! ## Required-field validation parity
//!
//! Python validates required fields *after* proto parsing via `_assert_required`,
//! whose failure yields the exact message
//! `"Missing value for required parameter '{param}'. See the API docs for more
//! information about request parameters."` (`handlers.py:904-944`). Because
//! proto2 optional string fields deserialize an absent field as `None` (prost
//! `Option`) — and Python treats both absent and empty-string as "missing" —
//! [`require_non_empty`] enforces the same non-empty-string requirement.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_store::{Experiment, ExperimentTag, ExperimentsPage, ViewType};

use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `_create_experiment` (`handlers.py:1550`).
pub async fn create_experiment(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateExperiment = parse_request(&parts, &body, "mlflow.CreateExperiment")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;

    let tags: Vec<(&str, &str)> = req
        .tags
        .iter()
        .map(|t| {
            (
                t.key.as_deref().unwrap_or(""),
                t.value.as_deref().unwrap_or(""),
            )
        })
        .collect();

    let experiment_id = state
        .tracking_store()
        .create_experiment(
            workspace.name(),
            name,
            req.artifact_location.as_deref().filter(|s| !s.is_empty()),
            &tags,
        )
        .await?;

    let resp = pb::create_experiment::Response {
        experiment_id: Some(experiment_id),
    };
    proto_response(&resp, "mlflow.CreateExperiment.Response")
}

/// `_get_experiment` (`handlers.py:1576`).
pub async fn get_experiment(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetExperiment = parse_request(&parts, &body, "mlflow.GetExperiment")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;

    let exp = state
        .tracking_store()
        .get_experiment(workspace.name(), experiment_id)
        .await?;

    let resp = pb::get_experiment::Response {
        experiment: Some(to_proto_experiment(exp, workspace.name())),
    };
    proto_response(&resp, "mlflow.GetExperiment.Response")
}

/// `_get_experiment_by_name` (`handlers.py:1595`).
pub async fn get_experiment_by_name(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetExperimentByName = parse_request(&parts, &body, "mlflow.GetExperimentByName")?;
    let name = require_non_empty(req.experiment_name.as_deref(), "experiment_name")?;

    let exp = state
        .tracking_store()
        .get_experiment_by_name(workspace.name(), name)
        .await?
        .ok_or_else(|| {
            MlflowError::new(
                format!("Could not find experiment with name '{name}'"),
                ErrorCode::ResourceDoesNotExist,
            )
        })?;

    let resp = pb::get_experiment_by_name::Response {
        experiment: Some(to_proto_experiment(exp, workspace.name())),
    };
    proto_response(&resp, "mlflow.GetExperimentByName.Response")
}

/// `_search_experiments` (`handlers.py:2482`), POST and GET.
pub async fn search_experiments(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchExperiments = parse_request(&parts, &body, "mlflow.SearchExperiments")?;

    // The handler passes the raw proto values straight to the store: `view_type`
    // defaults to the proto default (0 → interpreted as ACTIVE_ONLY), and
    // `max_results` defaults to 0 (no proto default), which the store validates.
    let view_type = view_type_from_proto(req.view_type);
    let max_results = req.max_results.unwrap_or(0);
    let filter = req.filter.as_deref().filter(|s| !s.is_empty());
    let page_token = req.page_token.as_deref().filter(|s| !s.is_empty());

    let ExperimentsPage {
        experiments,
        next_page_token,
    } = state
        .tracking_store()
        .search_experiments(
            workspace.name(),
            view_type,
            max_results,
            filter,
            &req.order_by,
            page_token,
        )
        .await?;

    let resp = pb::search_experiments::Response {
        experiments: experiments
            .into_iter()
            .map(|e| to_proto_experiment(e, workspace.name()))
            .collect(),
        next_page_token,
    };
    proto_response(&resp, "mlflow.SearchExperiments.Response")
}

/// `_delete_experiment` (`handlers.py:1616`).
pub async fn delete_experiment(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteExperiment = parse_request(&parts, &body, "mlflow.DeleteExperiment")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;

    state
        .tracking_store()
        .delete_experiment(workspace.name(), experiment_id)
        .await?;

    proto_response(
        &pb::delete_experiment::Response {},
        "mlflow.DeleteExperiment.Response",
    )
}

/// `_restore_experiment` (`handlers.py:1629`).
pub async fn restore_experiment(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::RestoreExperiment = parse_request(&parts, &body, "mlflow.RestoreExperiment")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;

    state
        .tracking_store()
        .restore_experiment(workspace.name(), experiment_id)
        .await?;

    proto_response(
        &pb::restore_experiment::Response {},
        "mlflow.RestoreExperiment.Response",
    )
}

/// `_update_experiment` (`handlers.py:1643`). Renames only when `new_name` is
/// present (Python: `if request_message.new_name:`), but `new_name` is a
/// required schema field, so an absent/empty value is rejected first.
pub async fn update_experiment(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateExperiment = parse_request(&parts, &body, "mlflow.UpdateExperiment")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;
    let new_name = require_non_empty(req.new_name.as_deref(), "new_name")?;

    state
        .tracking_store()
        .rename_experiment(workspace.name(), experiment_id, new_name)
        .await?;

    proto_response(
        &pb::update_experiment::Response {},
        "mlflow.UpdateExperiment.Response",
    )
}

/// `_set_experiment_tag` (`handlers.py:1842`).
pub async fn set_experiment_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetExperimentTag = parse_request(&parts, &body, "mlflow.SetExperimentTag")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;
    let value = req.value.as_deref().unwrap_or("");

    state
        .tracking_store()
        .set_experiment_tag(workspace.name(), experiment_id, key, value)
        .await?;

    proto_response(
        &pb::set_experiment_tag::Response {},
        "mlflow.SetExperimentTag.Response",
    )
}

/// `_delete_experiment_tag` (`handlers.py:1861`).
pub async fn delete_experiment_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteExperimentTag = parse_request(&parts, &body, "mlflow.DeleteExperimentTag")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;

    state
        .tracking_store()
        .delete_experiment_tag(workspace.name(), experiment_id, key)
        .await?;

    proto_response(
        &pb::delete_experiment_tag::Response {},
        "mlflow.DeleteExperimentTag.Response",
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Enforce a required, non-empty string field, matching `_assert_required`
/// (absent OR empty string is "missing") and its verbatim error message.
fn require_non_empty<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(MlflowError::new(
            format!(
                "Missing value for required parameter '{param}'. \
                 See the API docs for more information about request parameters."
            ),
            ErrorCode::InvalidParameterValue,
        )),
    }
}

/// Map a store [`Experiment`] entity to the proto message. The proto uses `""`
/// for an absent artifact location / tag value (proto2 string default), and
/// timestamps default to `0` when unset (matching `to_mlflow_entity`/`to_proto`).
/// Serialize a store [`Experiment`] to its proto, stamping the request's
/// `workspace` (proto field 9). Python's `Experiment.__init__` always runs
/// `resolve_entity_workspace_name`, so `to_proto` always sets `workspace` —
/// "Always `default` if workspace is not enabled" (`service.proto:3227`). The
/// store entity doesn't carry the column back, so the handler passes the
/// request-scoped workspace name here; pre-T10.4 that always resolves to
/// `default`, matching the single-tenant Python default.
pub(crate) fn to_proto_experiment(exp: Experiment, workspace: &str) -> pb::Experiment {
    pb::Experiment {
        experiment_id: Some(exp.experiment_id),
        name: Some(exp.name),
        artifact_location: Some(exp.artifact_location.unwrap_or_default()),
        lifecycle_stage: Some(exp.lifecycle_stage),
        last_update_time: exp.last_update_time,
        creation_time: exp.creation_time,
        tags: exp.tags.into_iter().map(to_proto_tag).collect(),
        effective_trace_archival_retention: None,
        workspace: Some(workspace.to_string()),
    }
}

fn to_proto_tag(tag: ExperimentTag) -> pb::ExperimentTag {
    pb::ExperimentTag {
        key: Some(tag.key),
        value: Some(tag.value.unwrap_or_default()),
    }
}

/// Interpret the proto `view_type` (`Option<i32>`) as a store [`ViewType`].
///
/// `ViewType` is a proto2 enum whose first value is `ACTIVE_ONLY = 1` (there is
/// no zero value). Per proto2 semantics an *absent* `optional ViewType` field
/// reads back as that first value, so Python's `request_message.view_type`
/// yields `ACTIVE_ONLY` for a request that omits it — which is why an
/// unfiltered search returns the active experiments rather than nothing. We
/// mirror that: `None` (field absent) → `ACTIVE_ONLY`.
///
/// A field that is *present but unrecognized* (e.g. an explicit `0`) matches
/// none of the store stages and maps to `None`, which the store treats as an
/// empty stage set (no rows) — byte-for-byte with Python's
/// `LifecycleStage.view_type_to_stages(0) == []`.
///
/// (Found by the T12.4 differential harness: an unfiltered `searchExperiments`
/// returned every active experiment on Python but nothing on Rust.)
fn view_type_from_proto(view_type: Option<i32>) -> Option<ViewType> {
    match view_type {
        None => Some(ViewType::ActiveOnly),
        Some(v) if v == pb::ViewType::ActiveOnly as i32 => Some(ViewType::ActiveOnly),
        Some(v) if v == pb::ViewType::DeletedOnly as i32 => Some(ViewType::DeletedOnly),
        Some(v) if v == pb::ViewType::All as i32 => Some(ViewType::All),
        _ => None,
    }
}
