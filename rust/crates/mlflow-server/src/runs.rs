//! Run endpoints (plan T3.2, §3.2): create, update, delete, restore, get,
//! search (POST), log-metric, log-parameter, set-tag, delete-tag, log-batch,
//! log-model (legacy), log-inputs, outputs (log-outputs).
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`:
//! parse the request proto (via [`crate::proto_http`]), enforce the handler-level
//! schema (required fields — Python's `_assert_required`), call the
//! workspace-scoped store method, then serialize the response proto. Nearly all
//! of the semantic validation (metric NaN/Inf, param immutability, batch limits,
//! run-name↔`mlflow.runName` sync, run active-state checks) already lives in the
//! store (`mlflow-store`), so these handlers deliberately do NOT re-validate
//! those — they only reproduce what Python's *handler* layer does before calling
//! the store.
//!
//! ## `run_id` / `run_uuid` precedence
//!
//! Several run messages carry both a modern `run_id` and a deprecated `run_uuid`.
//! Python resolves `run_id = request_message.run_id or request_message.run_uuid`
//! (`handlers.py:1701,1764,1784,1888,1927`), i.e. prefer a non-empty `run_id`,
//! else fall back to `run_uuid`. [`require_run_id`] reproduces that.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_store::{
    DatasetInputSpec, LoggedModelOutput, MetricInput, Run, RunData, RunInfo, RunInputs, RunOutputs,
    RunStatus, RunsPage, ViewType,
};

use crate::auth_middleware::{validators::resolve_experiment_permission, AuthContext};
use crate::proto_http::{parse_request, parse_request_lenient, proto_response};
use crate::schema_validation::{missing_required_error, SchemaEntry, Validator};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `_log_batch`'s per-element presence closures (`handlers.py:2537-2549`):
/// `_assert_metrics_fields_present` requires each metric's `key`/`value`/
/// `timestamp`; `_assert_params_fields_present` requires each param's `key`;
/// `_assert_tags_fields_present` requires each tag's `key`. Each mirrors
/// `_assert_required(elem.get(field), path=f"...[i].field")`.
fn assert_metrics_fields_present(value: &serde_json::Value) -> Result<(), MlflowError> {
    assert_elem_fields(value, "metrics", &["key", "value", "timestamp"])
}

fn assert_params_fields_present(value: &serde_json::Value) -> Result<(), MlflowError> {
    assert_elem_fields(value, "params", &["key"])
}

fn assert_tags_fields_present(value: &serde_json::Value) -> Result<(), MlflowError> {
    assert_elem_fields(value, "tags", &["key"])
}

/// For each element of `value` (a JSON array), require every field in `fields`
/// to be present and non-empty, matching `_assert_required(elem.get(field),
/// path=f"{list_name}[{i}].{field}")`. A non-array value has no elements to walk
/// (`_assert_array` would have failed earlier in the schema), so it is a no-op.
fn assert_elem_fields(
    value: &serde_json::Value,
    list_name: &str,
    fields: &[&str],
) -> Result<(), MlflowError> {
    let serde_json::Value::Array(items) = value else {
        return Ok(());
    };
    for (idx, item) in items.iter().enumerate() {
        for field in fields {
            // `_assert_required`: present, not null, and not the empty string.
            let ok = match item.get(field) {
                None | Some(serde_json::Value::Null) => false,
                Some(serde_json::Value::String(s)) => !s.is_empty(),
                Some(_) => true,
            };
            if !ok {
                return Err(missing_required_error(&format!(
                    "{list_name}[{idx}].{field}"
                )));
            }
        }
    }
    Ok(())
}

/// `_log_batch`'s schema (`handlers.py:2554-2559`).
const LOG_BATCH_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "run_id",
        validators: &[Validator::String, Validator::Required],
    },
    SchemaEntry {
        param: "metrics",
        validators: &[
            Validator::Array,
            Validator::Custom(assert_metrics_fields_present),
        ],
    },
    SchemaEntry {
        param: "params",
        validators: &[
            Validator::Array,
            Validator::Custom(assert_params_fields_present),
        ],
    },
    SchemaEntry {
        param: "tags",
        validators: &[
            Validator::Array,
            Validator::Custom(assert_tags_fields_present),
        ],
    },
];

/// `_log_metric`'s schema (`handlers.py:1743-1752`). Field order matches Python's
/// dict so the first failing param surfaces the same message.
const LOG_METRIC_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "run_id",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "key",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "value",
        validators: &[Validator::Required, Validator::FloatLike],
    },
    SchemaEntry {
        param: "timestamp",
        validators: &[Validator::IntLike, Validator::Required],
    },
    SchemaEntry {
        param: "step",
        validators: &[Validator::IntLike],
    },
    SchemaEntry {
        param: "model_id",
        validators: &[Validator::String],
    },
    SchemaEntry {
        param: "dataset_name",
        validators: &[Validator::String],
    },
    SchemaEntry {
        param: "dataset_digest",
        validators: &[Validator::String],
    },
];

/// `_log_param`'s schema (`handlers.py:1777-1781`).
const LOG_PARAM_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "run_id",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "key",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "value",
        validators: &[Validator::String],
    },
];

/// `_set_tag`'s schema (`handlers.py:1881-1885`).
const SET_TAG_SCHEMA: &[SchemaEntry] = &[
    SchemaEntry {
        param: "run_id",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "key",
        validators: &[Validator::Required, Validator::String],
    },
    SchemaEntry {
        param: "value",
        validators: &[Validator::String],
    },
];

/// `_create_run` (`handlers.py:1663`).
///
/// The deprecated `user_id` field is passed straight through, matching Python:
/// `request_message.user_id` yields the proto2 default `""` when absent, so the
/// run's `user_id` column is stored as `""` (not NULL). Likewise `start_time`
/// defaults to the proto2 `0`.
pub async fn create_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateRun = parse_request(&parts, &body, "mlflow.CreateRun")?;

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

    let run = state
        .tracking_store()
        .create_run(
            workspace.name(),
            req.experiment_id.as_deref().unwrap_or(""),
            Some(req.user_id.as_deref().unwrap_or("")),
            Some(req.start_time.unwrap_or(0)),
            req.run_name.as_deref(),
            &tags,
        )
        .await?;

    // Python's `create_run` returns `Run(run.info, run.data,
    // RunInputs(dataset_inputs=...))` with NO outputs argument
    // (`sqlalchemy_store.py:979`), so `Run.to_proto`'s `if self.outputs:`
    // omits the field on the create response — unlike `get_run`, which always
    // attaches a (possibly empty) `RunOutputs`. Found by the T12.4 harness.
    let mut run_proto = to_proto_run(run);
    run_proto.outputs = None;
    let resp = pb::create_run::Response {
        run: Some(run_proto),
    };
    proto_response(&resp, "mlflow.CreateRun.Response")
}

/// `_update_run` (`handlers.py:1691`). `run_name`/`end_time`/`status` are applied
/// only when present (`HasField`), which prost's `Option` already encodes.
pub async fn update_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateRun = parse_request(&parts, &body, "mlflow.UpdateRun")?;
    let run_id = require_run_id(req.run_id.as_deref(), req.run_uuid.as_deref())?;
    let status = req.status.map(run_status_to_string);

    let updated = state
        .tracking_store()
        .update_run_info(
            workspace.name(),
            run_id,
            status.as_deref(),
            req.end_time,
            req.run_name.as_deref(),
        )
        .await?;

    let resp = pb::update_run::Response {
        run_info: Some(to_proto_run_info(updated)),
    };
    proto_response(&resp, "mlflow.UpdateRun.Response")
}

/// `_delete_run` (`handlers.py:1714`).
pub async fn delete_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteRun = parse_request(&parts, &body, "mlflow.DeleteRun")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;

    state
        .tracking_store()
        .delete_run(workspace.name(), run_id)
        .await?;

    proto_response(&pb::delete_run::Response {}, "mlflow.DeleteRun.Response")
}

/// `_restore_run` (`handlers.py:1727`).
pub async fn restore_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::RestoreRun = parse_request(&parts, &body, "mlflow.RestoreRun")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;

    state
        .tracking_store()
        .restore_run(workspace.name(), run_id)
        .await?;

    proto_response(&pb::restore_run::Response {}, "mlflow.RestoreRun.Response")
}

/// `_get_run` / `get_run_impl` (`handlers.py:1915`), GET.
pub async fn get_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetRun = parse_request(&parts, &body, "mlflow.GetRun")?;
    // `run_id` is `[_assert_required, _assert_string]`; the `run_uuid` fallback
    // only applies once required-ness is satisfied via either field.
    let run_id = require_run_id(req.run_id.as_deref(), req.run_uuid.as_deref())?;

    let run = state
        .tracking_store()
        .get_run(workspace.name(), run_id)
        .await?;

    let resp = pb::get_run::Response {
        run: Some(to_proto_run(run)),
    };
    proto_response(&resp, "mlflow.GetRun.Response")
}

/// `_search_runs` / `search_runs_impl` (`handlers.py:1934`), POST.
///
/// Parity with the Python handler's schema validation: `max_results` is only
/// range-checked when the client explicitly sends it (`_validate_request_json`
/// iterates keys present in the JSON body). When sent and `> 50000`, Python's
/// `_assert_less_than_or_equal` raises a bare `AssertionError`, which becomes the
/// `invalid_value` message — distinct from the store's own threshold message.
/// When omitted, the proto default `1000` applies. `run_view_type` uses
/// `HasField` semantics: default `ACTIVE_ONLY`.
pub async fn search_runs(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchRuns = parse_request(&parts, &body, "mlflow.SearchRuns")?;

    // Handler-level max_results check (only when explicitly provided).
    if let Some(mr) = req.max_results {
        if mr > 50000 {
            return Err(MlflowError::new(
                format!(
                    "Invalid value {mr} for parameter 'max_results' supplied. \
                     See the API docs for more information about request parameters."
                ),
                ErrorCode::InvalidParameterValue,
            ));
        }
    }

    // Python passes `request_message.max_results` (proto default 1000) straight
    // to the store, so max_results is always a concrete value here.
    let max_results = req.max_results.unwrap_or(1000) as i64;

    // `run_view_type` HasField semantics: unset → ACTIVE_ONLY. An explicitly set
    // value is mapped by `ViewType::from_proto` in Python; here we mirror the
    // three recognized values and default anything else to ACTIVE_ONLY (Python's
    // `from_proto` would KeyError on an unknown, but real clients only send the
    // three enum values, and prost only surfaces those three from JSON).
    let run_view_type = view_type_from_proto(req.run_view_type);

    let filter = req.filter.as_deref().filter(|s| !s.is_empty());
    let page_token = req.page_token.as_deref().filter(|s| !s.is_empty());

    // `search_runs_impl` calls `auth.filter_experiment_ids` before querying the
    // store (`handlers.py:1961-1968`). Apply the same filter when basic auth is
    // active; filtering before pagination is essential so page tokens walk only
    // the readable run stream.
    let mut experiment_ids = req.experiment_ids.clone();
    if let (Some(auth), Some(auth_store)) =
        (parts.extensions.get::<AuthContext>(), state.auth_store())
    {
        if !auth.is_admin {
            let mut readable = Vec::with_capacity(experiment_ids.len());
            for experiment_id in experiment_ids {
                if resolve_experiment_permission(
                    auth_store,
                    &auth.username,
                    workspace.name(),
                    state.workspace_store().is_some(),
                    &experiment_id,
                )
                .await?
                .can_read
                {
                    readable.push(experiment_id);
                }
            }
            experiment_ids = readable;
        }
    }

    let RunsPage {
        runs,
        next_page_token,
    } = state
        .tracking_store()
        .search_runs(
            workspace.name(),
            &experiment_ids,
            filter,
            run_view_type,
            Some(max_results),
            &req.order_by,
            page_token,
        )
        .await?;

    let resp = pb::search_runs::Response {
        runs: runs.into_iter().map(to_proto_run).collect(),
        next_page_token,
    };
    proto_response(&resp, "mlflow.SearchRuns.Response")
}

/// `_log_metric` (`handlers.py:1740`).
pub async fn log_metric(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogMetric =
        parse_request_lenient(&parts, &body, "mlflow.LogMetric", LOG_METRIC_SCHEMA)?;
    let run_id = require_run_id(req.run_id.as_deref(), req.run_uuid.as_deref())?;

    let metric = MetricInput {
        key: req.key.clone().unwrap_or_default(),
        value: req.value.unwrap_or(0.0),
        timestamp: req.timestamp.unwrap_or(0),
        step: req.step.unwrap_or(0),
        model_id: req.model_id.filter(|s| !s.is_empty()),
        dataset_name: req.dataset_name.filter(|s| !s.is_empty()),
        dataset_digest: req.dataset_digest.filter(|s| !s.is_empty()),
    };

    state
        .tracking_store()
        .log_metric(workspace.name(), run_id, &metric)
        .await?;

    proto_response(&pb::log_metric::Response {}, "mlflow.LogMetric.Response")
}

/// `_log_param` (`handlers.py:1774`). `value` is a required schema field, so an
/// absent value is rejected before the store's own validation runs.
pub async fn log_param(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogParam =
        parse_request_lenient(&parts, &body, "mlflow.LogParam", LOG_PARAM_SCHEMA)?;
    let run_id = require_run_id(req.run_id.as_deref(), req.run_uuid.as_deref())?;
    let value = req.value.as_deref().unwrap_or("");

    state
        .tracking_store()
        .log_param(
            workspace.name(),
            run_id,
            req.key.as_deref().unwrap_or(""),
            value,
        )
        .await?;

    proto_response(&pb::log_param::Response {}, "mlflow.LogParam.Response")
}

/// `_set_tag` (`handlers.py:1878`).
pub async fn set_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetTag = parse_request_lenient(&parts, &body, "mlflow.SetTag", SET_TAG_SCHEMA)?;
    let run_id = require_run_id(req.run_id.as_deref(), req.run_uuid.as_deref())?;
    let value = req.value.as_deref().unwrap_or("");

    state
        .tracking_store()
        .set_tag(
            workspace.name(),
            run_id,
            req.key.as_deref().unwrap_or(""),
            value,
        )
        .await?;

    proto_response(&pb::set_tag::Response {}, "mlflow.SetTag.Response")
}

/// `_delete_tag` (`handlers.py:1898`). `run_id` here is `[_assert_required,
/// _assert_string]` with NO `run_uuid` fallback (the proto has no `run_uuid`).
pub async fn delete_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteTag = parse_request(&parts, &body, "mlflow.DeleteTag")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;

    state
        .tracking_store()
        .delete_tag(workspace.name(), run_id, key)
        .await?;

    proto_response(&pb::delete_tag::Response {}, "mlflow.DeleteTag.Response")
}

/// `_log_batch` (`handlers.py:2536`). Batch limits, param-value length, dup-param
/// keys, and per-entity metric/param/tag validation all live in the store's
/// `log_batch`; the handler only enforces `run_id` required-ness (the
/// `_assert_*_fields_present` checks in Python require `key`/`value`/`timestamp`
/// sub-fields, but proto parsing makes absent scalar sub-fields their defaults,
/// so those handler asserts never fire post-proto-parse — mirrored by passing
/// the proto values straight through).
///
/// NB: Python also calls `_validate_batch_log_api_req(_get_request_json())`,
/// which is `len(json_req) > 1e6`. But `_get_request_json` returns the *parsed
/// dict*, so `len(...)` counts top-level keys (≈4), never bytes — the check can
/// never fire. We deliberately omit this dead no-op rather than reproduce it
/// (there is no observable behavior to match).
pub async fn log_batch(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogBatch =
        parse_request_lenient(&parts, &body, "mlflow.LogBatch", LOG_BATCH_SCHEMA)?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;

    let metrics: Vec<MetricInput> = req
        .metrics
        .iter()
        .map(|m| MetricInput {
            key: m.key.clone().unwrap_or_default(),
            value: m.value.unwrap_or(0.0),
            timestamp: m.timestamp.unwrap_or(0),
            step: m.step.unwrap_or(0),
            model_id: m.model_id.clone().filter(|s| !s.is_empty()),
            dataset_name: m.dataset_name.clone().filter(|s| !s.is_empty()),
            dataset_digest: m.dataset_digest.clone().filter(|s| !s.is_empty()),
        })
        .collect();
    let params: Vec<(&str, &str)> = req
        .params
        .iter()
        .map(|p| {
            (
                p.key.as_deref().unwrap_or(""),
                p.value.as_deref().unwrap_or(""),
            )
        })
        .collect();
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

    state
        .tracking_store()
        .log_batch(workspace.name(), run_id, &metrics, &params, &tags)
        .await?;

    proto_response(&pb::log_batch::Response {}, "mlflow.LogBatch.Response")
}

/// `_log_model` (`handlers.py:2575`) — the legacy model-history API. Validates
/// the `model_json` is valid JSON and carries the mandatory fields, then appends
/// its tags dict to the run's `mlflow.log-model.history` tag.
pub async fn log_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogModel = parse_request(&parts, &body, "mlflow.LogModel")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;
    let model_json = require_non_empty(req.model_json.as_deref(), "model_json")?;

    let model: serde_json::Value = serde_json::from_str(model_json).map_err(|_| {
        MlflowError::invalid_parameter_value(format!(
            "Malformed model info. \n {model_json} \n is not a valid JSON."
        ))
    })?;

    // `{"artifact_path", "flavors", "utc_time_created", "run_id"} - set(model)`.
    let missing = missing_model_fields(&model);
    if !missing.is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Model json is missing mandatory fields: {}",
            render_field_set(&missing)
        )));
    }

    state
        .tracking_store()
        .record_logged_model(workspace.name(), run_id, &model)
        .await?;

    proto_response(&pb::log_model::Response {}, "mlflow.LogModel.Response")
}

/// `_log_inputs` (`handlers.py:1794`). `models` is optional; an empty list maps
/// to Python's `None` (no model-input edges).
pub async fn log_inputs(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogInputs = parse_request(&parts, &body, "mlflow.LogInputs")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;

    let datasets: Vec<DatasetInputSpec> = req
        .datasets
        .iter()
        .map(|d| {
            let ds = d.dataset.as_ref();
            DatasetInputSpec {
                name: ds.and_then(|x| x.name.clone()).unwrap_or_default(),
                digest: ds.and_then(|x| x.digest.clone()).unwrap_or_default(),
                source_type: ds.and_then(|x| x.source_type.clone()).unwrap_or_default(),
                source: ds.and_then(|x| x.source.clone()).unwrap_or_default(),
                schema: ds.and_then(|x| x.schema.clone()),
                profile: ds.and_then(|x| x.profile.clone()),
                tags: d
                    .tags
                    .iter()
                    .map(|t| {
                        (
                            t.key.clone().unwrap_or_default(),
                            t.value.clone().unwrap_or_default(),
                        )
                    })
                    .collect(),
            }
        })
        .collect();
    let model_inputs: Vec<&str> = req
        .models
        .iter()
        .map(|m| m.model_id.as_deref().unwrap_or(""))
        .collect();

    state
        .tracking_store()
        .log_inputs(workspace.name(), run_id, &datasets, &model_inputs)
        .await?;

    proto_response(&pb::log_inputs::Response {}, "mlflow.LogInputs.Response")
}

/// `_log_outputs` (`handlers.py:1826`). `models` is a required array.
pub async fn log_outputs(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LogOutputs = parse_request(&parts, &body, "mlflow.LogOutputs")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;

    let models: Vec<LoggedModelOutput> = req
        .models
        .iter()
        .map(|m| LoggedModelOutput {
            model_id: m.model_id.clone().unwrap_or_default(),
            step: m.step.unwrap_or(0),
        })
        .collect();

    state
        .tracking_store()
        .log_outputs(workspace.name(), run_id, &models)
        .await?;

    proto_response(&pb::log_outputs::Response {}, "mlflow.LogOutputs.Response")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Enforce a required, non-empty string field (`_assert_required` + verbatim
/// message).
fn require_non_empty<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(missing_required(param)),
    }
}

/// `run_id = request_message.run_id or request_message.run_uuid` with the
/// `_assert_required` on the resolved `run_id`. Prefers a non-empty `run_id`,
/// then a non-empty `run_uuid`, else the missing-`run_id` error.
fn require_run_id<'a>(
    run_id: Option<&'a str>,
    run_uuid: Option<&'a str>,
) -> Result<&'a str, MlflowError> {
    let resolved = run_id
        .filter(|s| !s.is_empty())
        .or(run_uuid.filter(|s| !s.is_empty()));
    resolved.ok_or_else(|| missing_required("run_id"))
}

fn missing_required(param: &str) -> MlflowError {
    MlflowError::new(
        format!(
            "Missing value for required parameter '{param}'. \
             See the API docs for more information about request parameters."
        ),
        ErrorCode::InvalidParameterValue,
    )
}

/// `{"artifact_path", "flavors", "utc_time_created", "run_id"} - set(model)` —
/// the mandatory fields absent from the model JSON, sorted (Python's `set` order
/// is nondeterministic; sorting gives a stable message).
fn missing_model_fields(model: &serde_json::Value) -> Vec<&'static str> {
    const REQUIRED: [&str; 4] = ["artifact_path", "flavors", "utc_time_created", "run_id"];
    let present = model.as_object();
    let mut missing: Vec<&'static str> = REQUIRED
        .iter()
        .copied()
        .filter(|f| present.map(|o| !o.contains_key(*f)).unwrap_or(true))
        .collect();
    missing.sort_unstable();
    missing
}

/// Render a field list as Python's `set` repr (`{'a', 'b'}`).
fn render_field_set(fields: &[&str]) -> String {
    let inner = fields
        .iter()
        .map(|f| format!("'{f}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{inner}}}")
}

/// Interpret the proto `run_view_type` (`Option<i32>`) as a store [`ViewType`],
/// defaulting an unset field to `ACTIVE_ONLY` (Python's `search_runs_impl`
/// default). Only the three real enum values occur on the wire.
fn view_type_from_proto(view_type: Option<i32>) -> ViewType {
    match view_type {
        Some(v) if v == pb::ViewType::DeletedOnly as i32 => ViewType::DeletedOnly,
        Some(v) if v == pb::ViewType::All as i32 => ViewType::All,
        _ => ViewType::ActiveOnly,
    }
}

/// `RunStatus.to_string(status_enum_int)` — map the proto enum to its persisted
/// name. An unrecognized value would be a client sending an out-of-range enum;
/// prost surfaces it as the raw int, which we render as `"RUNNING"` fallback
/// (the store then rejects any invalid status string, matching Python's
/// `RunStatus.to_string` KeyError surfacing an error).
fn run_status_to_string(status: i32) -> String {
    match pb::RunStatus::try_from(status) {
        Ok(s) => s.as_str_name().to_string(),
        Err(_) => RunStatus::RUNNING.to_string(),
    }
}

/// Map a store [`Run`] to the proto message (`Run.to_proto`): info + data +
/// inputs + outputs, each section always emitted (Python's `MergeFrom` on a
/// truthy sub-entity; the store always returns all four).
pub(crate) fn to_proto_run(run: Run) -> pb::Run {
    pb::Run {
        info: Some(to_proto_run_info(run.info)),
        data: Some(to_proto_run_data(run.data)),
        inputs: Some(to_proto_run_inputs(run.inputs)),
        outputs: Some(to_proto_run_outputs(run.outputs)),
    }
}

/// `RunInfo.to_proto` (`mlflow/entities/run_info.py:137`): sets both `run_uuid`
/// and `run_id` to the id; `user_id` and `start_time` are always emitted (proto2
/// defaults `""` / `0`); `end_time` and `artifact_uri` only when truthy;
/// `run_name` when not None (always, in practice).
fn to_proto_run_info(info: RunInfo) -> pb::RunInfo {
    let end_time = info.end_time.filter(|&t| t != 0);
    let artifact_uri = info.artifact_uri.filter(|s| !s.is_empty());
    pb::RunInfo {
        run_id: Some(info.run_id.clone()),
        run_uuid: Some(info.run_id),
        run_name: Some(info.run_name),
        experiment_id: Some(info.experiment_id),
        user_id: Some(info.user_id.unwrap_or_default()),
        status: Some(run_status_from_string(&info.status)),
        start_time: Some(info.start_time.unwrap_or(0)),
        end_time,
        artifact_uri,
        lifecycle_stage: Some(info.lifecycle_stage),
    }
}

/// `RunStatus.from_string(name)` → proto enum int.
fn run_status_from_string(status: &str) -> i32 {
    pb::RunStatus::from_str_name(status)
        .map(|s| s as i32)
        .unwrap_or(pb::RunStatus::Running as i32)
}

/// `RunData.to_proto`: the store's latest-metrics entities carry only
/// key/value/timestamp/step (no model_id/dataset/run_id), matching what
/// `latest_metrics` stores and what Python emits for run data.
fn to_proto_run_data(data: RunData) -> pb::RunData {
    pb::RunData {
        metrics: data
            .metrics
            .into_iter()
            .map(|m| pb::Metric {
                key: Some(m.key),
                value: Some(m.value),
                timestamp: Some(m.timestamp),
                step: Some(m.step),
                dataset_name: None,
                dataset_digest: None,
                model_id: None,
                run_id: None,
            })
            .collect(),
        params: data
            .params
            .into_iter()
            .map(|p| pb::Param {
                key: Some(p.key),
                value: Some(p.value),
            })
            .collect(),
        tags: data
            .tags
            .into_iter()
            .map(|t| pb::RunTag {
                key: Some(t.key),
                value: Some(t.value),
            })
            .collect(),
    }
}

/// `RunInputs.to_proto`: dataset inputs (each `Dataset.to_proto` omits an
/// empty schema/profile) + model inputs.
fn to_proto_run_inputs(inputs: RunInputs) -> pb::RunInputs {
    pb::RunInputs {
        dataset_inputs: inputs
            .dataset_inputs
            .into_iter()
            .map(|di| pb::DatasetInput {
                tags: di
                    .tags
                    .into_iter()
                    .map(|t| pb::InputTag {
                        key: Some(t.key),
                        value: Some(t.value),
                    })
                    .collect(),
                dataset: Some(pb::Dataset {
                    name: Some(di.dataset.name),
                    digest: Some(di.dataset.digest),
                    source_type: Some(di.dataset.source_type),
                    source: Some(di.dataset.source),
                    schema: di.dataset.schema.filter(|s| !s.is_empty()),
                    profile: di.dataset.profile.filter(|s| !s.is_empty()),
                }),
            })
            .collect(),
        model_inputs: inputs
            .model_inputs
            .into_iter()
            .map(|mi| pb::ModelInput {
                model_id: Some(mi.model_id),
            })
            .collect(),
    }
}

/// `RunOutputs.to_proto`.
fn to_proto_run_outputs(outputs: RunOutputs) -> pb::RunOutputs {
    pb::RunOutputs {
        model_outputs: outputs
            .model_outputs
            .into_iter()
            .map(|mo| pb::ModelOutput {
                model_id: Some(mo.model_id),
                step: Some(mo.step),
            })
            .collect(),
    }
}
