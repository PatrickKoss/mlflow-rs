//! Logged model endpoints (plan T3.4, §3.5): create, get, finalize (PATCH),
//! delete, search, tags set/delete, log-params.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`
//! (`_create_logged_model`, `_get_logged_model`, `_finalize_logged_model`,
//! `_delete_logged_model`, `_search_logged_models`, `_set_logged_model_tags`,
//! `_delete_logged_model_tag`, `_log_logged_model_params`). See
//! [`crate::experiments`] for the general parse/validate/store/respond shape
//! this crate follows.
//!
//! ## Path parameters: `model_id` / `tag_key`
//!
//! Five of these routes carry `{model_id}` (and one also `{tag_key}`) as a
//! REST-style path segment (`/mlflow/logged-models/{model_id}`, `.../tags`,
//! `.../tags/{tag_key}`, `.../params`) — the first path-parameterized
//! proto endpoints in this server. Python's Flask view functions receive path
//! segments as plain function arguments, entirely separate from
//! `_get_request_message`'s body parsing; some of those Python handlers
//! (`_finalize_logged_model`, `_log_logged_model_params`) additionally read
//! `request_message.model_id` off the *parsed body*, because the schema also
//! requires it there (the real MLflow client sends the id in both the URL and
//! the JSON body for those endpoints).
//!
//! Rather than duplicate that split (URL segment for some fields, parsed body
//! for others, per-endpoint) we use one uniform mechanism, built as reusable
//! infrastructure in [`crate::proto_http::parse_request_with_path_params`]:
//! axum's `Path<HashMap<String, String>>` extractor captures every `{...}`
//! segment the route declares, and each handler overlays those onto the
//! parsed request *before* proto validation, as if the client had also sent
//! them as body/query fields. See that function's doc comment for why this is
//! safe (strictly more permissive than Python, never observably different for
//! real clients) and how later path-param phases (traces, registry, webhooks)
//! should reuse it.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_store::{
    DatasetFilter, LoggedModel, LoggedModelKv, LoggedModelOrderByInput, LoggedModelStatus,
};

use crate::proto_http::{parse_request, parse_request_with_path_params, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `_create_logged_model` (`handlers.py:5240`).
pub async fn create_logged_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateLoggedModel = parse_request(&parts, &body, "mlflow.CreateLoggedModel")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;

    let params: Vec<LoggedModelKv> = req.params.iter().map(to_store_kv_param).collect();
    let tags: Vec<LoggedModelKv> = req.tags.iter().map(to_store_kv_tag).collect();

    let model = state
        .tracking_store()
        .create_logged_model(
            workspace.name(),
            experiment_id,
            req.name.as_deref().filter(|s| !s.is_empty()),
            req.source_run_id.as_deref().filter(|s| !s.is_empty()),
            &tags,
            &params,
            req.model_type.as_deref().filter(|s| !s.is_empty()),
        )
        .await?;

    let resp = pb::create_logged_model::Response {
        model: Some(to_proto_logged_model(model)),
    };
    proto_response(&resp, "mlflow.CreateLoggedModel.Response")
}

/// `_log_logged_model_params` (`handlers.py:5275`), path: `POST
/// /mlflow/logged-models/{model_id}/params`.
pub async fn log_logged_model_params(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogLoggedModelParamsRequest = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.LogLoggedModelParamsRequest",
        &path_param_pairs(&path_params, &["model_id"]),
    )?;
    let model_id = require_non_empty(req.model_id.as_deref(), "model_id")?;
    let params: Vec<LoggedModelKv> = req.params.iter().map(to_store_kv_param).collect();

    state
        .tracking_store()
        .log_logged_model_params(workspace.name(), model_id, &params)
        .await?;

    proto_response(
        &pb::log_logged_model_params_request::Response {},
        "mlflow.LogLoggedModelParamsRequest.Response",
    )
}

/// `_get_logged_model` (`handlers.py:5294`), path: `GET
/// /mlflow/logged-models/{model_id}`. `allow_deleted` is a hand-rolled query
/// flag (not a proto field — `request.args.get("allow_deleted", "false")`),
/// matched verbatim: any value other than a case-insensitive `"true"` is
/// `false`.
pub async fn get_logged_model(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let model_id = path_params.get("model_id").cloned().unwrap_or_default();
    let allow_deleted = parts
        .uri
        .query()
        .and_then(|q| {
            url_query_pairs(q)
                .into_iter()
                .rev()
                .find(|(k, _)| k == "allow_deleted")
        })
        .map(|(_, v)| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let model = state
        .tracking_store()
        .get_logged_model(workspace.name(), &model_id, allow_deleted)
        .await?;

    let resp = pb::get_logged_model::Response {
        model: Some(to_proto_logged_model(model)),
    };
    proto_response(&resp, "mlflow.GetLoggedModel.Response")
}

/// `_finalize_logged_model` (`handlers.py:5303`), path: `PATCH
/// /mlflow/logged-models/{model_id}`.
pub async fn finalize_logged_model(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::FinalizeLoggedModel = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.FinalizeLoggedModel",
        &path_param_pairs(&path_params, &["model_id"]),
    )?;
    let model_id = require_non_empty(req.model_id.as_deref(), "model_id")?;
    let status_i32 = req.status.ok_or_else(|| missing_param("status"))?;
    let status = LoggedModelStatus::from_int(status_i32 as i64)?;

    let model = state
        .tracking_store()
        .finalize_logged_model(workspace.name(), model_id, status)
        .await?;

    let resp = pb::finalize_logged_model::Response {
        model: Some(to_proto_logged_model(model)),
    };
    proto_response(&resp, "mlflow.FinalizeLoggedModel.Response")
}

/// `_delete_logged_model` (`handlers.py:5320`), path: `DELETE
/// /mlflow/logged-models/{model_id}`. Python takes `model_id` purely as a URL
/// segment (no body schema at all), so this skips proto body parsing entirely.
pub async fn delete_logged_model(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
) -> Result<Response, MlflowError> {
    let model_id = path_params.get("model_id").cloned().unwrap_or_default();
    state
        .tracking_store()
        .delete_logged_model(workspace.name(), &model_id)
        .await?;

    proto_response(
        &pb::delete_logged_model::Response {},
        "mlflow.DeleteLoggedModel.Response",
    )
}

/// `_set_logged_model_tags` (`handlers.py:5327`), path: `PATCH
/// /mlflow/logged-models/{model_id}/tags`.
pub async fn set_logged_model_tags(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetLoggedModelTags = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.SetLoggedModelTags",
        &path_param_pairs(&path_params, &["model_id"]),
    )?;
    let model_id = require_non_empty(req.model_id.as_deref(), "model_id")?;
    let tags: Vec<LoggedModelKv> = req.tags.iter().map(to_store_kv_tag).collect();

    state
        .tracking_store()
        .set_logged_model_tags(workspace.name(), model_id, &tags)
        .await?;

    // Python's `_set_logged_model_tags` returns an empty `Response()` — the
    // `model` field, though present on the message, is never populated here.
    proto_response(
        &pb::set_logged_model_tags::Response { model: None },
        "mlflow.SetLoggedModelTags.Response",
    )
}

/// `_delete_logged_model_tag` (`handlers.py:5339`), path: `DELETE
/// /mlflow/logged-models/{model_id}/tags/{tag_key}`. Both segments are URL
/// path parameters in Python (no body schema), so this skips proto body
/// parsing entirely.
pub async fn delete_logged_model_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
) -> Result<Response, MlflowError> {
    let model_id = path_params.get("model_id").cloned().unwrap_or_default();
    let tag_key = path_params.get("tag_key").cloned().unwrap_or_default();

    state
        .tracking_store()
        .delete_logged_model_tag(workspace.name(), &model_id, &tag_key)
        .await?;

    proto_response(
        &pb::delete_logged_model_tag::Response {},
        "mlflow.DeleteLoggedModelTag.Response",
    )
}

/// `_search_logged_models` (`handlers.py:5346`), path: `POST
/// /mlflow/logged-models/search`.
pub async fn search_logged_models(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchLoggedModels = parse_request(&parts, &body, "mlflow.SearchLoggedModels")?;
    if req.experiment_ids.is_empty() {
        return Err(missing_param("experiment_ids"));
    }

    let datasets: Vec<DatasetFilter> = req
        .datasets
        .iter()
        .map(|d| DatasetFilter {
            dataset_name: d.dataset_name.clone().unwrap_or_default(),
            dataset_digest: d.dataset_digest.clone().filter(|s| !s.is_empty()),
        })
        .collect();
    let order_by: Vec<LoggedModelOrderByInput> = req
        .order_by
        .iter()
        .map(|ob| LoggedModelOrderByInput {
            field_name: ob.field_name.clone().unwrap_or_default(),
            ascending: ob.ascending.unwrap_or(true),
            dataset_name: ob.dataset_name.clone().filter(|s| !s.is_empty()),
            dataset_digest: ob.dataset_digest.clone().filter(|s| !s.is_empty()),
        })
        .collect();

    let page = state
        .tracking_store()
        .search_logged_models(
            workspace.name(),
            &req.experiment_ids,
            req.filter.as_deref().filter(|s| !s.is_empty()),
            &datasets,
            req.max_results.filter(|n| *n > 0).map(|n| n as usize),
            &order_by,
            req.page_token.as_deref().filter(|s| !s.is_empty()),
        )
        .await?;

    let resp = pb::search_logged_models::Response {
        models: page.models.into_iter().map(to_proto_logged_model).collect(),
        next_page_token: page.next_page_token,
    };
    proto_response(&resp, "mlflow.SearchLoggedModels.Response")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Same required/non-empty check as [`crate::experiments::require_non_empty`]
/// (duplicated locally to keep modules independent — see that function's doc
/// comment for the exact Python parity it reproduces).
fn require_non_empty<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(missing_param(param)),
    }
}

fn missing_param(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. \
         See the API docs for more information about request parameters."
    ))
}

/// Build the `path_params` overlay slice for
/// [`parse_request_with_path_params`] from the axum-captured path segments,
/// keeping only the names this endpoint's route declares.
fn path_param_pairs(
    path_params: &HashMap<String, String>,
    names: &[&'static str],
) -> Vec<(&'static str, String)> {
    names
        .iter()
        .filter_map(|name| path_params.get(*name).map(|v| (*name, v.clone())))
        .collect()
}

/// Minimal query-string parser for the one hand-rolled (non-proto)
/// `allow_deleted` flag `_get_logged_model` reads directly off
/// `request.args`. Percent-decoding is unnecessary here since `"true"`/
/// `"false"` never need it, but we still split on `&`/`=` the same way
/// [`crate::proto_http`]'s query parser does.
fn url_query_pairs(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (k.to_string(), v.to_string()),
            None => (pair.to_string(), String::new()),
        })
        .collect()
}

fn to_store_kv_param(p: &pb::LoggedModelParameter) -> LoggedModelKv {
    LoggedModelKv {
        key: p.key.clone().unwrap_or_default(),
        value: p.value.clone().unwrap_or_default(),
    }
}

fn to_store_kv_tag(t: &pb::LoggedModelTag) -> LoggedModelKv {
    LoggedModelKv {
        key: t.key.clone().unwrap_or_default(),
        value: t.value.clone().unwrap_or_default(),
    }
}

/// Map the store [`LoggedModel`] entity to the proto message
/// (`LoggedModelInfo` + `LoggedModelData`). `registrations` is always empty:
/// the store doesn't track Model Registry promotions on the logged-model row
/// (out of scope for this task; the Model Registry phase would populate it).
fn to_proto_logged_model(model: LoggedModel) -> pb::LoggedModel {
    let info = pb::LoggedModelInfo {
        model_id: Some(model.model_id),
        experiment_id: Some(model.experiment_id),
        name: Some(model.name),
        creation_timestamp_ms: Some(model.creation_timestamp),
        last_updated_timestamp_ms: Some(model.last_updated_timestamp),
        artifact_uri: Some(model.artifact_location),
        status: Some(model.status as i32),
        creator_id: None,
        model_type: model.model_type,
        source_run_id: model.source_run_id,
        status_message: model.status_message,
        tags: model.tags.into_iter().map(to_proto_tag).collect(),
        registrations: Vec::new(),
    };
    let data = pb::LoggedModelData {
        params: model.params.into_iter().map(to_proto_param).collect(),
        metrics: model.metrics.into_iter().map(to_proto_metric).collect(),
    };
    pb::LoggedModel {
        info: Some(info),
        data: Some(data),
    }
}

fn to_proto_tag(t: LoggedModelKv) -> pb::LoggedModelTag {
    pb::LoggedModelTag {
        key: Some(t.key),
        value: Some(t.value),
    }
}

fn to_proto_param(p: LoggedModelKv) -> pb::LoggedModelParameter {
    pb::LoggedModelParameter {
        key: Some(p.key),
        value: Some(p.value),
    }
}

fn to_proto_metric(m: mlflow_store::LoggedModelMetric) -> pb::Metric {
    pb::Metric {
        key: Some(m.key),
        value: m.value,
        timestamp: Some(m.timestamp),
        step: Some(m.step),
        dataset_name: m.dataset_name,
        dataset_digest: m.dataset_digest,
        model_id: None,
        run_id: Some(m.run_id),
    }
}
