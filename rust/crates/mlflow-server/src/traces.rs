//! Tracing V3 endpoints (plan T4.1, §3.6): the 13 V3 trace RPCs.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`
//! (`_start_trace_v3`, `_get_trace_info_v3`, `_get_trace`, `_batch_get_traces`,
//! `_batch_get_trace_infos`, `_search_traces_v3`, `_delete_traces`,
//! `_set_trace_tag_v3`, `_delete_trace_tag_v3`, `_link_traces_to_run`,
//! `_link_prompts_to_trace`, `_calculate_trace_filter_correlation`,
//! `_query_trace_metrics`). See [`crate::logged_models`] for the path-param
//! mechanism the tag routes reuse (`/mlflow/traces/{trace_id}/tags`).
//!
//! ## Response assembly
//!
//! Trace responses build the V3 wire protos from the store entities:
//! [`to_proto_trace_info`] maps [`TraceInfo`] → `TraceInfoV3` (timestamp/duration
//! well-known types, `State` enum, `TraceLocation`, tag/metadata maps), and
//! [`to_otel_span`] reconstructs each stored span's JSON `content` (the
//! mlflow span dict) into an OTLP `Span` proto for `Trace.spans`, matching
//! `mlflow.entities.Span.to_otel_proto`.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_store::{
    Assessment, AssessmentError, AssessmentSource, AssessmentValue, MetricAggregation,
    MetricDataPoint, MetricViewType, TraceAssessment, TraceInfo, TraceState, TraceWithSpans,
    MAX_RESULTS_QUERY_TRACE_METRICS,
};

use crate::proto_http::{
    parse_request, parse_request_lenient, parse_request_with_path_params, proto_response,
};
use crate::schema_validation::{SchemaEntry, Validator};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `_start_trace_v3`'s schema (`handlers.py:3877`): `{"trace": [_assert_required]}`.
const START_TRACE_V3_SCHEMA: &[SchemaEntry] = &[SchemaEntry {
    param: "trace",
    validators: &[Validator::Required],
}];

/// `SEARCH_TRACES_V3_MAX_RESULTS` handler-level threshold (`handlers.py:3961`,
/// `_assert_less_than_or_equal(int(x), 500)`).
const SEARCH_TRACES_V3_MAX_RESULTS: i32 = 500;
/// Default `max_results` for `searchTracesV3` (proto default = 100).
const SEARCH_TRACES_V3_DEFAULT_MAX_RESULTS: i32 = 100;

// ---------------------------------------------------------------------------
// startTraceV3
// ---------------------------------------------------------------------------

/// `_start_trace_v3` (`handlers.py:3872`), path: `POST /mlflow/traces`.
pub async fn start_trace_v3(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::StartTraceV3 =
        parse_request_lenient(&parts, &body, "mlflow.StartTraceV3", START_TRACE_V3_SCHEMA)?;
    let trace = req.trace.ok_or_else(|| missing_param("trace"))?;
    let info = trace
        .trace_info
        .ok_or_else(|| missing_param("trace.trace_info"))?;

    let input = start_trace_input_from_proto(info)?;
    let stored = state
        .tracking_store()
        .start_trace(workspace.name(), &input)
        .await?;

    let resp = pb::start_trace_v3::Response {
        trace: Some(pb::Trace {
            trace_info: Some(to_proto_trace_info(&stored)),
            spans: Vec::new(),
        }),
    };
    proto_response(&resp, "mlflow.StartTraceV3.Response")
}

// ---------------------------------------------------------------------------
// getTraceInfoV3
// ---------------------------------------------------------------------------

/// `_get_trace_info_v3` (`handlers.py:3888`), path: `GET
/// /mlflow/traces/{trace_id}`.
pub async fn get_trace_info_v3(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
) -> Result<Response, MlflowError> {
    let trace_id = path_params.get("trace_id").cloned().unwrap_or_default();
    let info = state
        .tracking_store()
        .get_trace_info(workspace.name(), &trace_id)
        .await?;
    let resp = pb::get_trace_info_v3::Response {
        trace: Some(pb::Trace {
            trace_info: Some(to_proto_trace_info(&info)),
            spans: Vec::new(),
        }),
    };
    proto_response(&resp, "mlflow.GetTraceInfoV3.Response")
}

// ---------------------------------------------------------------------------
// getTrace
// ---------------------------------------------------------------------------

/// `_get_trace` (`handlers.py:3930`), path: `GET /mlflow/traces/get`
/// (`?trace_id=&allow_partial=`).
pub async fn get_trace(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetTrace = parse_request(&parts, &body, "mlflow.GetTrace")?;
    let trace_id = require_non_empty(req.trace_id.as_deref(), "trace_id")?;
    let allow_partial = req.allow_partial.unwrap_or(false);

    let info = state
        .tracking_store()
        .get_trace_info(workspace.name(), trace_id)
        .await?;
    let trace = if info.tag(mlflow_store::TRACE_TAG_SPANS_LOCATION)
        == Some(mlflow_store::SPANS_LOCATION_ARCHIVE_REPO)
    {
        pb::Trace {
            trace_info: Some(to_proto_trace_info(&info)),
            spans: crate::trace_archival::download_archived_spans(&info).await?,
        }
    } else {
        let trace = state
            .tracking_store()
            .get_trace(workspace.name(), trace_id, allow_partial)
            .await?;
        to_proto_trace(&trace)?
    };
    let resp = pb::get_trace::Response { trace: Some(trace) };
    proto_response(&resp, "mlflow.GetTrace.Response")
}

// ---------------------------------------------------------------------------
// batchGetTraces
// ---------------------------------------------------------------------------

/// `_batch_get_traces` (`handlers.py:3900`), path: `GET
/// /mlflow/traces/batchGet`.
pub async fn batch_get_traces(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::BatchGetTraces = parse_request(&parts, &body, "mlflow.BatchGetTraces")?;
    if req.trace_ids.is_empty() {
        return Err(missing_param("trace_ids"));
    }
    let traces = state
        .tracking_store()
        .batch_get_traces(workspace.name(), &req.trace_ids)
        .await?;
    let mut proto_traces = Vec::with_capacity(traces.len());
    for t in &traces {
        proto_traces.push(to_proto_trace(t)?);
    }
    let resp = pb::batch_get_traces::Response {
        traces: proto_traces,
    };
    proto_response(&resp, "mlflow.BatchGetTraces.Response")
}

// ---------------------------------------------------------------------------
// batchGetTraceInfos
// ---------------------------------------------------------------------------

/// `_batch_get_trace_infos` (`handlers.py:3917`), path: `POST
/// /mlflow/traces/batchGetInfos`.
pub async fn batch_get_trace_infos(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::BatchGetTraceInfos = parse_request(&parts, &body, "mlflow.BatchGetTraceInfos")?;
    if req.trace_ids.is_empty() {
        return Err(missing_param("trace_ids"));
    }
    let infos = state
        .tracking_store()
        .batch_get_trace_infos(workspace.name(), &req.trace_ids)
        .await?;
    let resp = pb::batch_get_trace_infos::Response {
        trace_infos: infos.iter().map(to_proto_trace_info).collect(),
    };
    proto_response(&resp, "mlflow.BatchGetTraceInfos.Response")
}

// ---------------------------------------------------------------------------
// searchTracesV3
// ---------------------------------------------------------------------------

/// `_search_traces_v3` (`handlers.py:3950`), path: `POST
/// /mlflow/traces/search`. `max_results` defaults to 100, capped at 500 at the
/// handler with a byte-matched error.
pub async fn search_traces_v3(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchTracesV3 = parse_request(&parts, &body, "mlflow.SearchTracesV3")?;
    if req.locations.is_empty() {
        return Err(missing_param("locations"));
    }
    let max_results = req
        .max_results
        .unwrap_or(SEARCH_TRACES_V3_DEFAULT_MAX_RESULTS);
    if max_results > SEARCH_TRACES_V3_MAX_RESULTS {
        // Byte-matched to `_assert_less_than_or_equal(..., 500)` (bare
        // AssertionError → `invalid_value`, `validation.py:113`): the value is
        // `json.dumps(max_results)` (an int, unquoted).
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. \
             See the API docs for more information about request parameters."
        )));
    }

    // Extract experiment ids from the MLFLOW_EXPERIMENT locations
    // (`location.mlflow_experiment.experiment_id` for HasField mlflow_experiment).
    let experiment_ids: Vec<String> = req
        .locations
        .iter()
        .filter_map(|loc| match &loc.identifier {
            Some(pb::trace_location::Identifier::MlflowExperiment(exp)) => {
                exp.experiment_id.clone()
            }
            _ => None,
        })
        .collect();

    let page = state
        .tracking_store()
        .search_traces(
            workspace.name(),
            &experiment_ids,
            req.filter.as_deref().filter(|s| !s.is_empty()),
            max_results as i64,
            &req.order_by,
            req.page_token.as_deref().filter(|s| !s.is_empty()),
        )
        .await?;

    let resp = pb::search_traces_v3::Response {
        traces: page.trace_infos.iter().map(to_proto_trace_info).collect(),
        next_page_token: page.next_page_token,
    };
    proto_response(&resp, "mlflow.SearchTracesV3.Response")
}

// ---------------------------------------------------------------------------
// deleteTracesV3
// ---------------------------------------------------------------------------

/// `_delete_traces` (`handlers.py:3989`), path: `POST
/// /mlflow/traces/delete-traces`. Bound to the `DeleteTracesV3` message (the V3
/// route), whose fields are identical to `DeleteTraces`. `HasField` semantics on
/// `max_timestamp_millis` (Some(0) ≠ None) are preserved via prost `Option`.
pub async fn delete_traces_v3(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteTracesV3 = parse_request(&parts, &body, "mlflow.DeleteTracesV3")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;

    // An empty `request_ids` list means "id-based deletion not requested"
    // (Python always passes the list; the store's `Option` contract treats an
    // empty list as `None` so time-based deletion still runs).
    let trace_ids = if req.request_ids.is_empty() {
        None
    } else {
        Some(req.request_ids.as_slice())
    };
    let deleted = state
        .tracking_store()
        .delete_traces(
            workspace.name(),
            experiment_id,
            req.max_timestamp_millis,
            req.max_traces.map(|n| n as i64),
            trace_ids,
        )
        .await?;

    let resp = pb::delete_traces_v3::Response {
        traces_deleted: Some(deleted as i32),
    };
    proto_response(&resp, "mlflow.DeleteTracesV3.Response")
}

// ---------------------------------------------------------------------------
// setTraceTagV3 / deleteTraceTagV3
// ---------------------------------------------------------------------------

/// `_set_trace_tag_v3` (`handlers.py:4073`), path: `PATCH
/// /mlflow/traces/{trace_id}/tags`.
pub async fn set_trace_tag_v3(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetTraceTagV3 = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.SetTraceTagV3",
        &path_param_pairs(&path_params, &["trace_id"]),
    )?;
    let trace_id = require_non_empty(req.trace_id.as_deref(), "trace_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;
    let value = req.value.unwrap_or_default();

    state
        .tracking_store()
        .set_trace_tag(workspace.name(), trace_id, key, &value)
        .await?;
    proto_response(
        &pb::set_trace_tag_v3::Response {},
        "mlflow.SetTraceTagV3.Response",
    )
}

/// `_delete_trace_tag_v3` (`handlers.py:4110`), path: `DELETE
/// /mlflow/traces/{trace_id}/tags` (`?key=`).
pub async fn delete_trace_tag_v3(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteTraceTagV3 = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.DeleteTraceTagV3",
        &path_param_pairs(&path_params, &["trace_id"]),
    )?;
    let trace_id = require_non_empty(req.trace_id.as_deref(), "trace_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;

    state
        .tracking_store()
        .delete_trace_tag(workspace.name(), trace_id, key)
        .await?;
    proto_response(
        &pb::delete_trace_tag_v3::Response {},
        "mlflow.DeleteTraceTagV3.Response",
    )
}

// ---------------------------------------------------------------------------
// linkTracesToRun / linkPromptsToTrace
// ---------------------------------------------------------------------------

/// `_link_traces_to_run` (`handlers.py:4129`), path: `POST
/// /mlflow/traces/link-to-run`. The store enforces the ≤100 limit with a
/// byte-matched error.
pub async fn link_traces_to_run(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LinkTracesToRun = parse_request(&parts, &body, "mlflow.LinkTracesToRun")?;
    let run_id = require_non_empty(req.run_id.as_deref(), "run_id")?;

    state
        .tracking_store()
        .link_traces_to_run(workspace.name(), &req.trace_ids, run_id)
        .await?;
    proto_response(
        &pb::link_traces_to_run::Response {},
        "mlflow.LinkTracesToRun.Response",
    )
}

/// `_link_prompts_to_trace` (`handlers.py:4149`), path: `POST
/// /mlflow/traces/link-prompts`. Stores the (name, version) link only.
pub async fn link_prompts_to_trace(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::LinkPromptsToTrace = parse_request(&parts, &body, "mlflow.LinkPromptsToTrace")?;
    let trace_id = require_non_empty(req.trace_id.as_deref(), "trace_id")?;

    let prompt_versions: Vec<(String, String)> = req
        .prompt_versions
        .iter()
        .map(|pv| {
            (
                pv.name.clone().unwrap_or_default(),
                pv.version.clone().unwrap_or_default(),
            )
        })
        .collect();

    state
        .tracking_store()
        .link_prompts_to_trace(workspace.name(), trace_id, &prompt_versions)
        .await?;
    proto_response(
        &pb::link_prompts_to_trace::Response {},
        "mlflow.LinkPromptsToTrace.Response",
    )
}

// ---------------------------------------------------------------------------
// calculateTraceFilterCorrelation
// ---------------------------------------------------------------------------

/// `_calculate_trace_filter_correlation` (`handlers.py:4026`), path: `POST
/// /mlflow/traces/calculate-filter-correlation`.
pub async fn calculate_trace_filter_correlation(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CalculateTraceFilterCorrelation =
        parse_request(&parts, &body, "mlflow.CalculateTraceFilterCorrelation")?;
    if req.experiment_ids.is_empty() {
        return Err(missing_param("experiment_ids"));
    }
    let filter1 = require_non_empty(req.filter_string1.as_deref(), "filter_string1")?;
    let filter2 = require_non_empty(req.filter_string2.as_deref(), "filter_string2")?;
    // `base_filter` uses HasField semantics (None when unset).
    let base_filter = req.base_filter.as_deref();

    let result = state
        .tracking_store()
        .calculate_trace_filter_correlation(
            workspace.name(),
            &req.experiment_ids,
            filter1,
            filter2,
            base_filter,
        )
        .await?;

    let resp = pb::calculate_trace_filter_correlation::Response {
        npmi: Some(result.npmi),
        npmi_smoothed: Some(result.npmi_smoothed),
        filter1_count: Some(result.filter1_count as i32),
        filter2_count: Some(result.filter2_count as i32),
        joint_count: Some(result.joint_count as i32),
        total_count: Some(result.total_count as i32),
    };
    proto_response(&resp, "mlflow.CalculateTraceFilterCorrelation.Response")
}

// ---------------------------------------------------------------------------
// queryTraceMetrics
// ---------------------------------------------------------------------------

/// `_query_trace_metrics` (`handlers.py:4283`), path: `POST
/// /mlflow/traces/metrics`.
pub async fn query_trace_metrics(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::QueryTraceMetrics = parse_request(&parts, &body, "mlflow.QueryTraceMetrics")?;
    if req.experiment_ids.is_empty() {
        return Err(missing_param("experiment_ids"));
    }
    let view_type = view_type_from_proto(req.view_type)?;
    let metric_name = require_non_empty(req.metric_name.as_deref(), "metric_name")?;
    if req.aggregations.is_empty() {
        return Err(missing_param("aggregations"));
    }
    let aggregations = req
        .aggregations
        .iter()
        .map(aggregation_from_proto)
        .collect::<Result<Vec<_>, _>>()?;

    let max_results = req
        .max_results
        .map(|n| n as i64)
        .unwrap_or(MAX_RESULTS_QUERY_TRACE_METRICS);

    let data_points = state
        .tracking_store()
        .query_trace_metrics(
            workspace.name(),
            &req.experiment_ids,
            view_type,
            metric_name,
            &aggregations,
            &req.dimensions,
            &req.filters,
            req.time_interval_seconds,
            req.start_time_ms,
            req.end_time_ms,
            max_results,
        )
        .await?;

    let resp = pb::query_trace_metrics::Response {
        data_points: data_points.iter().map(to_proto_data_point).collect(),
        // Python never sets next_page_token (pagination unimplemented).
        next_page_token: None,
    };
    proto_response(&resp, "mlflow.QueryTraceMetrics.Response")
}

// ===========================================================================
// Proto conversion helpers
// ===========================================================================

/// Build the store [`mlflow_store::StartTraceInput`] from the request's
/// `TraceInfoV3` proto. The caller-supplied tags/metadata are copied verbatim;
/// the store adds the artifact-location tag and finalized-metadata marker.
fn start_trace_input_from_proto(
    info: pb::TraceInfoV3,
) -> Result<mlflow_store::StartTraceInput, MlflowError> {
    let experiment_id = info
        .trace_location
        .as_ref()
        .and_then(|loc| match &loc.identifier {
            Some(pb::trace_location::Identifier::MlflowExperiment(exp)) => {
                exp.experiment_id.clone()
            }
            _ => None,
        })
        .ok_or_else(|| missing_param("trace.trace_info.trace_location.mlflow_experiment"))?;

    // Python's `TraceInfo.from_proto` reads `request_time` off the proto with no
    // presence check — an unset Timestamp yields `0` (verified live:
    // `TraceInfo.from_proto(TraceInfoV3()).request_time == 0`), never an error.
    // This matters for the lenient-parse path: `startTraceV3` with a
    // non-RFC3339 `request_time` string (e.g. `"1000"`) fails `parse_dict` on
    // that field, which is swallowed, leaving `request_time` at its default; the
    // handler must then proceed with `0` and return 200, not 400.
    let request_time = info.request_time.map(timestamp_to_millis).unwrap_or(0);
    let execution_duration = info.execution_duration.map(duration_to_millis);
    let state = state_from_proto(info.state);

    let tags: Vec<(String, String)> = info.tags.into_iter().collect();
    let trace_metadata: Vec<(String, String)> = info.trace_metadata.into_iter().collect();
    let trace_id = info.trace_id.unwrap_or_default();
    let assessments = info
        .assessments
        .into_iter()
        .map(|assessment| {
            let assessment_trace_id = assessment
                .trace_id
                .clone()
                .filter(|trace_id| !trace_id.is_empty())
                .unwrap_or_else(|| trace_id.clone());
            crate::assessments::new_assessment_from_proto(assessment, assessment_trace_id)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(mlflow_store::StartTraceInput {
        trace_id,
        experiment_id,
        request_time,
        execution_duration,
        state,
        client_request_id: info.client_request_id.filter(|s| !s.is_empty()),
        request_preview: info.request_preview.filter(|s| !s.is_empty()),
        response_preview: info.response_preview.filter(|s| !s.is_empty()),
        tags,
        trace_metadata,
        // Token-usage-derived metrics are computed from spans (log_spans), not
        // on start_trace; the V3 TraceInfo has no metrics field.
        trace_metrics: Vec::new(),
        assessments,
    })
}

/// Map a store [`TraceWithSpans`] to the wire `Trace` proto.
fn to_proto_trace(trace: &TraceWithSpans) -> Result<pb::Trace, MlflowError> {
    let mut spans = Vec::with_capacity(trace.spans.len());
    for span in &trace.spans {
        spans.push(
            crate::trace_archival::stored_span_to_otel_span(span)
                .map_err(|error| MlflowError::new(error.to_string(), ErrorCode::InternalError))?,
        );
    }
    Ok(pb::Trace {
        trace_info: Some(to_proto_trace_info(&trace.info)),
        spans,
    })
}

/// Map a store [`TraceInfo`] to the wire `TraceInfoV3` proto, mirroring
/// `mlflow.entities.TraceInfo.to_proto`.
fn to_proto_trace_info(info: &TraceInfo) -> pb::TraceInfoV3 {
    let trace_location = pb::TraceLocation {
        r#type: Some(pb::trace_location::TraceLocationType::MlflowExperiment as i32),
        identifier: Some(pb::trace_location::Identifier::MlflowExperiment(
            pb::trace_location::MlflowExperimentLocation {
                experiment_id: Some(info.experiment_id.clone()),
            },
        )),
    };

    pb::TraceInfoV3 {
        trace_id: Some(info.trace_id.clone()),
        client_request_id: info.client_request_id.clone(),
        trace_location: Some(trace_location),
        // Deprecated request/response are never set by `to_proto`.
        request: None,
        response: None,
        request_preview: info.request_preview.clone(),
        response_preview: info.response_preview.clone(),
        request_time: Some(millis_to_timestamp(info.request_time)),
        execution_duration: info.execution_duration.map(millis_to_duration),
        state: Some(state_to_proto(&info.state)),
        trace_metadata: info
            .trace_metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().unwrap_or_default()))
            .collect(),
        assessments: info
            .assessments
            .iter()
            .map(trace_assessment_to_entity)
            .map(crate::assessments::to_proto_assessment)
            .collect(),
        tags: info
            .tags
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().unwrap_or_default()))
            .collect(),
    }
}

fn trace_assessment_to_entity(assessment: &TraceAssessment) -> Assessment {
    let value = match assessment.assessment_type.as_str() {
        "feedback" => AssessmentValue::Feedback {
            value_json: assessment.value.clone(),
            error: assessment.error.as_deref().map(|value| {
                serde_json::from_str::<AssessmentError>(value).unwrap_or(AssessmentError {
                    error_code: "UNKNOWN".to_string(),
                    error_message: None,
                    stack_trace: None,
                })
            }),
        },
        "expectation" => AssessmentValue::Expectation {
            value_json: assessment.value.clone(),
        },
        _ => AssessmentValue::Issue {
            issue_name: serde_json::from_str::<serde_json::Value>(&assessment.value)
                .ok()
                .and_then(|value| {
                    value
                        .get("issue_name")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_default(),
        },
    };
    Assessment {
        assessment_id: assessment.assessment_id.clone(),
        trace_id: assessment.trace_id.clone(),
        name: assessment.name.clone(),
        value,
        source: AssessmentSource {
            source_type: assessment.source_type.clone(),
            source_id: assessment.source_id.clone(),
        },
        run_id: assessment.run_id.clone(),
        span_id: assessment.span_id.clone(),
        rationale: assessment.rationale.clone(),
        metadata: assessment
            .metadata
            .as_deref()
            .and_then(|value| serde_json::from_str(value).ok()),
        create_time_ms: assessment.created_timestamp,
        last_update_time_ms: assessment.last_updated_timestamp,
        overrides: assessment.overrides.clone(),
        valid: assessment.valid,
    }
}

/// Map a `MetricDataPoint` store entity to the wire proto.
fn to_proto_data_point(dp: &MetricDataPoint) -> pb::MetricDataPoint {
    pb::MetricDataPoint {
        metric_name: Some(dp.metric_name.clone()),
        dimensions: dp.dimensions.clone().into_iter().collect(),
        values: dp.values.clone().into_iter().collect(),
    }
}

// ---- primitive helpers ----

/// `Timestamp.FromMilliseconds`: ms → (seconds, nanos) flooring toward -inf so
/// nanos stays in `[0, 1e9)`.
fn millis_to_timestamp(ms: i64) -> prost_types::Timestamp {
    let seconds = ms.div_euclid(1000);
    let nanos = (ms.rem_euclid(1000) * 1_000_000) as i32;
    prost_types::Timestamp { seconds, nanos }
}

/// Inverse of [`millis_to_timestamp`] for parsing an incoming request time.
fn timestamp_to_millis(ts: prost_types::Timestamp) -> i64 {
    ts.seconds * 1000 + (ts.nanos as i64) / 1_000_000
}

/// `Duration.FromMilliseconds`: ms → (seconds, nanos) truncating toward zero
/// (same sign). Durations are non-negative in practice.
fn millis_to_duration(ms: i64) -> prost_types::Duration {
    let seconds = ms / 1000;
    let nanos = ((ms % 1000) * 1_000_000) as i32;
    prost_types::Duration { seconds, nanos }
}

fn duration_to_millis(d: prost_types::Duration) -> i64 {
    d.seconds * 1000 + (d.nanos as i64) / 1_000_000
}

/// Map the `TraceInfoV3.State` enum int to the store status string.
fn state_from_proto(state: Option<i32>) -> String {
    match state {
        Some(1) => TraceState::OK,
        Some(2) => TraceState::ERROR,
        Some(3) => TraceState::IN_PROGRESS,
        _ => TraceState::STATE_UNSPECIFIED,
    }
    .to_string()
}

/// Map the store status string to the `TraceInfoV3.State` enum int.
fn state_to_proto(state: &str) -> i32 {
    match state {
        TraceState::OK => pb::trace_info_v3::State::Ok as i32,
        TraceState::ERROR => pb::trace_info_v3::State::Error as i32,
        TraceState::IN_PROGRESS => pb::trace_info_v3::State::InProgress as i32,
        _ => pb::trace_info_v3::State::Unspecified as i32,
    }
}

fn view_type_from_proto(view_type: Option<i32>) -> Result<MetricViewType, MlflowError> {
    match view_type {
        Some(1) => Ok(MetricViewType::Traces),
        Some(2) => Ok(MetricViewType::Spans),
        Some(3) => Ok(MetricViewType::Assessments),
        _ => Err(missing_param("view_type")),
    }
}

fn aggregation_from_proto(agg: &pb::MetricAggregation) -> Result<MetricAggregation, MlflowError> {
    match agg.aggregation_type {
        Some(1) => Ok(MetricAggregation::Count),
        Some(2) => Ok(MetricAggregation::Sum),
        Some(3) => Ok(MetricAggregation::Avg),
        Some(4) => {
            // PERCENTILE: requires percentile_value in [0, 100].
            let v = agg.percentile_value.ok_or_else(|| {
                MlflowError::invalid_parameter_value(
                    "Percentile value is required for PERCENTILE aggregation",
                )
            })?;
            if !(0.0..=100.0).contains(&v) {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Percentile value must be between 0 and 100, got {v}"
                )));
            }
            Ok(MetricAggregation::Percentile(v))
        }
        Some(5) | Some(6) => Err(MlflowError::invalid_parameter_value(
            "MIN/MAX aggregations are not supported",
        )),
        _ => Err(missing_param("aggregation_type")),
    }
}

/// Same required/non-empty check as [`crate::experiments::require_non_empty`].
/// `pub(crate)` so the V2 adapters in [`crate::traces_v2`] (plan T4.2, §3.7)
/// share the exact same "Missing value for required parameter" wording.
pub(crate) fn require_non_empty<'a>(
    value: Option<&'a str>,
    param: &str,
) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(missing_param(param)),
    }
}

pub(crate) fn missing_param(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. \
         See the API docs for more information about request parameters."
    ))
}

/// Build the `path_params` overlay slice (see [`crate::logged_models`]).
/// `pub(crate)` so [`crate::traces_v2`]'s `{request_id}`-keyed path routes
/// (`setTraceTag`/`deleteTraceTag`/`endTrace`) reuse it.
pub(crate) fn path_param_pairs(
    path_params: &HashMap<String, String>,
    names: &[&'static str],
) -> Vec<(&'static str, String)> {
    names
        .iter()
        .filter_map(|name| path_params.get(*name).map(|v| (*name, v.clone())))
        .collect()
}
