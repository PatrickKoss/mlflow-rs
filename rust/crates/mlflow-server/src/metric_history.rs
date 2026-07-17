//! Metric history endpoints (plan T3.3, §3.3): `getMetricHistory` /
//! `getMetricHistoryBulkInterval` (proto-backed, both `/api` and `/ajax-api`
//! via the route table) and the ajax-only hand-rolled `get-history-bulk`.
//!
//! Store-side logic (pagination cap, bulk-interval sampling) was ported
//! byte-identically in T2.7 (`mlflow-store::store::metrics`/`metrics_bulk`).
//! This module is the handler layer only: request parsing, the
//! handler-level caps/validation Python applies on top of the store call,
//! and response shaping — mirroring `mlflow/server/handlers.py`.
//!
//! Named `metric_history` (not `metrics`) because [`crate::metrics`] is
//! already the Prometheus exposition module.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_proto::{python_float_repr, quote_json_string};
use mlflow_store::{Metric, MetricWithRunId, GET_METRIC_HISTORY_MAX_RESULTS};
use mlflow_store::{MAX_RESULTS_PER_RUN, MAX_RUNS_GET_METRIC_HISTORY_BULK};

use crate::proto_http::{codec_err, parse_query_pairs, parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

/// `_get_metric_history` (`handlers.py:2067`).
///
/// Python reads `request_message.run_id or request_message.run_uuid`
/// (`run_id` wins when both/either are set; `run_uuid` is the deprecated
/// alias) and treats `max_results` as *absent* (non-paginated) unless the
/// proto field is explicitly present (`HasField`) — see the store's
/// `GET_METRIC_HISTORY_MAX_RESULTS` doc comment for why this distinction
/// (vs. treating an absent field as `0`) matters.
///
/// Unlike `getMetricHistoryBulkInterval`, this endpoint's schema
/// (`handlers.py:2070-2074`) has **no validator at all for `max_results`** —
/// only `run_id`/`metric_key`/`page_token` are checked. So when
/// `parse_dict` fails to coerce a non-numeric `max_results` query value,
/// Python's `_get_request_message` swallows the `ParseError` into
/// `proto_parsing_succeeded=False` and moves on: nothing re-validates
/// `max_results` afterwards, `HasField("max_results")` reads `False`, and the
/// request proceeds as if the field were never sent (verified against a live
/// `_get_request_message` call). Reproduced here by scrubbing an
/// un-parseable `max_results` from the query pairs *before* handing off to
/// the proto codec, rather than by using [`parse_request`] directly (which
/// would fail the whole-message parse on this one field, unlike Python).
pub async fn get_metric_history(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetMetricHistory = parse_metric_history_request(&parts, &body)?;
    let run_id = req
        .run_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(req.run_uuid.as_deref().filter(|s| !s.is_empty()));
    let run_id = require_non_empty(run_id, "run_id")?;
    let metric_key = require_non_empty(req.metric_key.as_deref(), "metric_key")?;

    let max_results = req.max_results.map(|v| v as i64);
    if let Some(mr) = max_results {
        if mr <= 0 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {mr} for parameter 'max_results' supplied. \
                 It must be a positive integer."
            )));
        }
    }
    let page_token = req.page_token.as_deref().filter(|s| !s.is_empty());

    let result = state
        .tracking_store()
        .get_metric_history(
            workspace.name(),
            run_id,
            metric_key,
            max_results.map(|v| v as usize),
            page_token,
        )
        .await;

    // Python's `_validate_run_accessible` is a NO-OP on the single-tenant
    // store (`sqlalchemy_store.py:787` — "the database will raise appropriate
    // errors ... or empty query results"), so a nonexistent run yields 200
    // with an empty history; only the workspace-aware subclass raises 404
    // (`sqlalchemy_workspace_store.py:349`). The Rust store always validates
    // (its run lookup doubles as workspace isolation — the metrics table has
    // no workspace column), so mirror the single-tenant no-op here by
    // degrading the run-not-found error to the empty page when workspaces are
    // disabled. Found by the T12.4 harness.
    let (metrics, next_page_token) = match result {
        Err(e)
            if state.workspace_store().is_none()
                && e.error_code == ErrorCode::ResourceDoesNotExist =>
        {
            (Vec::new(), None)
        }
        other => other?,
    };

    let resp = pb::get_metric_history::Response {
        metrics: metrics.into_iter().map(to_proto_metric).collect(),
        next_page_token,
    };
    proto_response(&resp, "mlflow.GetMetricHistory.Response")
}

/// `get_metric_history_bulk_interval_handler` +
/// `get_metric_history_bulk_interval_impl` (`handlers.py:2184-2250`).
///
/// Registered via the route table on both `/api` and `/ajax-api` (T3.5's
/// `handler_for` extension below) — Python additionally registers the ajax
/// path a second time by hand in `mlflow/server/__init__.py:129`, but that is
/// a redundant re-registration of the *same* URL to the *same* handler
/// (harmless duplication on the Python side); nothing extra is needed here.
pub async fn get_metric_history_bulk_interval(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    // Pre-validate the int-typed query params (`max_results`, `start_step`,
    // `end_step`) with Python's exact schema-validator fallback message
    // *before* handing off to the generic proto codec. Context: this is a GET
    // endpoint, so these arrive as raw query strings; Python's
    // `_get_request_message` first tries `parse_dict` (protobuf JSON
    // parsing), and on failure falls back to running the schema's type
    // validators (`_assert_intlike`) against the raw string, whose
    // AssertionError becomes
    // `Invalid value "<v>" for parameter '<name>' supplied:  Hint: Value was
    // of type 'str'.` (note the double space before "Hint:", verified against
    // a live `_validate_param_against_schema` call). The generic codec here
    // instead surfaces a raw `serde_json`/prost-reflect parse error, so these
    // three fields are checked by hand first for parity.
    if let Some(query) = parts.uri.query() {
        for field in ["max_results", "start_step", "end_step"] {
            if let Some(raw) = query_param(query, field) {
                if parse_python_int(&raw).is_err() {
                    return Err(invalid_intlike(field, &raw));
                }
            }
        }
    }

    let req: pb::GetMetricHistoryBulkInterval =
        parse_request(&parts, &body, "mlflow.GetMetricHistoryBulkInterval")?;

    if req.run_ids.is_empty() {
        return Err(missing_required("run_ids"));
    }
    if req.run_ids.len() > MAX_RUNS_GET_METRIC_HISTORY_BULK {
        return Err(MlflowError::invalid_parameter_value(format!(
            "GetMetricHistoryBulkInterval request must specify at most \
             {MAX_RUNS_GET_METRIC_HISTORY_BULK} run_ids. Received {} run_ids.",
            req.run_ids.len()
        )));
    }
    let metric_key = require_non_empty(req.metric_key.as_deref(), "metric_key")?;

    // `max_results`: unset defaults to MAX_RESULTS_PER_RUN (Python:
    // `int(args.get("max_results", MAX_RESULTS_PER_RUN))`); when present it
    // must be within [1, MAX_RESULTS_PER_RUN] (schema `_assert_intlike_within_range`).
    let max_results = match req.max_results {
        Some(mr) => {
            if !(1..=MAX_RESULTS_PER_RUN as i32).contains(&mr) {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "max_results must be between 1 and {MAX_RESULTS_PER_RUN}. \
                     See the API docs for more information about request parameters."
                )));
            }
            mr as usize
        }
        None => MAX_RESULTS_PER_RUN,
    };

    let (start_step, end_step) = match (req.start_step, req.end_step) {
        (Some(s), Some(e)) => {
            if s > e {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "end_step must be greater than start_step. \
                     Found start_step={s} and end_step={e}."
                )));
            }
            (Some(i64::from(s)), Some(i64::from(e)))
        }
        (None, None) => (None, None),
        _ => {
            return Err(MlflowError::invalid_parameter_value(
                "If either start step or end step are specified, both must be specified."
                    .to_string(),
            ));
        }
    };

    let run_ids: Vec<&str> = req.run_ids.iter().map(String::as_str).collect();
    let metrics = state
        .tracking_store()
        .get_metric_history_bulk_interval(
            workspace.name(),
            &run_ids,
            metric_key,
            max_results,
            start_step,
            end_step,
        )
        .await?;

    let resp = pb::get_metric_history_bulk_interval::Response {
        metrics: metrics
            .into_iter()
            .map(to_proto_metric_with_run_id)
            .collect(),
    };
    proto_response(&resp, "mlflow.GetMetricHistoryBulkInterval.Response")
}

/// `get_metric_history_bulk_handler` (`handlers.py:2112`, ajax-only —
/// `mlflow/server/__init__.py:123`, not in the proto route table).
///
/// Reads straight from the query string (not a proto message, so
/// [`parse_request`] doesn't apply) via [`parse_query_pairs`] — axum's
/// built-in `Query` extractor can't collect repeated `run_id=a&run_id=b` into
/// a `Vec` (its `serde_urlencoded` backing doesn't support that), which is
/// exactly the shape this endpoint needs.
///
/// The response is a **hand-rolled JSON dict**, not a serialized proto:
/// Python `return {"metrics": [...]}`, which Flask turns into JSON via its
/// default `jsonify` provider — compact separators (no spaces), object keys
/// sorted alphabetically, a trailing `\n`, and `allow_nan=True` float
/// formatting (`NaN`/`Infinity`/`-Infinity` literals, not `null`). Verified
/// empirically against a live Flask test client; `serde_json` cannot
/// reproduce the NaN/Infinity behavior (it silently maps non-finite floats to
/// `null`), so the body is built by hand using the same float/string
/// formatting the proto JSON codec uses ([`python_float_repr`],
/// [`quote_json_string`]).
pub async fn get_metric_history_bulk(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
    let run_ids: Vec<&str> = pairs
        .iter()
        .filter(|(k, _)| k == "run_id")
        .map(|(_, v)| v.as_str())
        .collect();

    if run_ids.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "GetMetricHistoryBulk request must specify at least one run_id.".to_string(),
        ));
    }
    if run_ids.len() > MAX_RUNS_GET_METRIC_HISTORY_BULK {
        return Err(MlflowError::invalid_parameter_value(format!(
            "GetMetricHistoryBulk request cannot specify more than \
             {MAX_RUNS_GET_METRIC_HISTORY_BULK} run_ids. Received {} run_ids.",
            run_ids.len()
        )));
    }
    // Python: `request.args.get("metric_key")`; `None` (param absent) errors,
    // but an explicit empty string is accepted (no `_assert_required` here).
    let metric_key = pairs
        .iter()
        .find(|(k, _)| k == "metric_key")
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(
                "GetMetricHistoryBulk request must specify a metric_key.".to_string(),
            )
        })?;

    // Python: `max_results = int(request.args.get("max_results", MAX_HISTORY_RESULTS))`
    // then `min(max_results, MAX_HISTORY_RESULTS)`. A non-integer value raises
    // Python's bare `int(str)` `ValueError`, which `catch_mlflow_exception`
    // does NOT catch (it only catches `MlflowException`): the exception
    // propagates to Flask's default (non-debug) error handler, which returns
    // a generic HTML `500 Internal Server Error` page — no JSON, no
    // exception detail. Verified empirically against a live Flask app.
    let max_results_param = pairs
        .iter()
        .find(|(k, _)| k == "max_results")
        .map(|(_, v)| v.as_str());
    let max_results = match max_results_param {
        Some(s) => match parse_python_int(s) {
            Ok(v) => v.min(GET_METRIC_HISTORY_MAX_RESULTS as i64),
            Err(_) => return Ok(generic_500_response()),
        },
        None => GET_METRIC_HISTORY_MAX_RESULTS as i64,
    };
    let max_results = max_results.max(0) as usize;

    // `store.get_metric_history_bulk` already sorts by `run_uuid` (Python's
    // `sorted(run_ids)` + per-run `sorted(..., key=(timestamp, step, value))`
    // collapse into the same single global `ORDER BY run_uuid, timestamp,
    // step, value` the T2.7 store method uses), and applies the same global
    // `max_results` cap Python re-applies via `metrics_with_run_ids[:max_results]`.
    let metrics = state
        .tracking_store()
        .get_metric_history_bulk(workspace.name(), &run_ids, metric_key, max_results)
        .await?;

    Ok(hand_rolled_bulk_response(&metrics))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// [`parse_request`] for `GetMetricHistory`, except an un-parseable
/// `max_results` query value is dropped rather than failing the whole parse
/// — see the doc comment on [`get_metric_history`] for why (no schema
/// validator for that field in Python, so a coercion failure is silently
/// ignored there instead of surfacing an error).
fn parse_metric_history_request(
    parts: &Parts,
    body: &Bytes,
) -> Result<pb::GetMetricHistory, MlflowError> {
    use axum::http::Method;

    if parts.method != Method::GET {
        return parse_request(parts, body, "mlflow.GetMetricHistory");
    }
    let Some(query) = parts.uri.query().filter(|q| !q.is_empty()) else {
        return parse_request(parts, body, "mlflow.GetMetricHistory");
    };

    let mut pairs = parse_query_pairs(query);
    if let Some(raw) = query_param(query, "max_results") {
        if parse_python_int(&raw).is_err() {
            pairs.retain(|(k, _)| k != "max_results");
        }
    }
    mlflow_proto::from_query_pairs::<pb::GetMetricHistory>(&pairs, "mlflow.GetMetricHistory")
        .map_err(codec_err)
}

/// Enforce a required, non-empty string field (`_assert_required`'s verbatim
/// message) — same helper as `experiments.rs::require_non_empty`, duplicated
/// locally to keep each handler module self-contained (mirrors the existing
/// per-module pattern rather than introducing a shared-helpers module for a
/// two-line function).
fn require_non_empty<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(missing_required(param)),
    }
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

/// `int(s)` with Python's exact semantics for the subset that matters here:
/// optional leading/trailing ASCII whitespace, an optional sign, and decimal
/// digits (optionally `_`-separated, though query params never carry those in
/// practice). Rust's `str::parse::<i64>` already rejects the same malformed
/// inputs `int()` rejects; this exists only to give the call site a single
/// narrow error type to map to the Python `ValueError` text.
fn parse_python_int(s: &str) -> Result<i64, std::num::ParseIntError> {
    s.trim().parse::<i64>()
}

/// Look up the first raw (percent-decoded) value for `field` in a query
/// string, matching `flask_request.args.get(field)`.
fn query_param(query: &str, field: &str) -> Option<String> {
    parse_query_pairs(query)
        .into_iter()
        .find(|(k, _)| k == field)
        .map(|(_, v)| v)
}

/// `_validate_param_against_schema`'s fallback message for a failed
/// `_assert_intlike` (bare `AssertionError`, no args):
/// `invalid_value(param, value, f" Hint: Value was of type '{type(value).__name__}'.")`,
/// then suffixed with the schema-validation-failure tail. Every GET query
/// value starts life as a Python `str`, hence the hardcoded `'str'` hint.
fn invalid_intlike(param: &str, value: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Invalid value {} for parameter '{param}' supplied:  Hint: Value was of type 'str'. \
         See the API docs for more information about request parameters.",
        quote_json_string(value),
    ))
}

fn to_proto_metric(m: Metric) -> pb::Metric {
    pb::Metric {
        key: Some(m.key),
        value: Some(m.value),
        timestamp: Some(m.timestamp),
        step: Some(m.step),
        dataset_name: None,
        dataset_digest: None,
        model_id: None,
        run_id: None,
    }
}

fn to_proto_metric_with_run_id(m: MetricWithRunId) -> pb::MetricWithRunId {
    pb::MetricWithRunId {
        key: Some(m.metric.key),
        value: Some(m.metric.value),
        timestamp: Some(m.metric.timestamp),
        step: Some(m.metric.step),
        run_id: Some(m.run_id),
    }
}

/// Build the exact hand-rolled JSON body Flask's `jsonify` produces for
/// `{"metrics": [...]}`: compact (no spaces), object keys sorted
/// alphabetically (`key`, `run_id`, `step`, `timestamp`, `value`), a trailing
/// `\n`, `Content-Type: application/json`, HTTP 200.
fn hand_rolled_bulk_response(metrics: &[MetricWithRunId]) -> Response {
    let mut body = String::from("{\"metrics\":[");
    for (i, m) in metrics.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push('{');
        body.push_str("\"key\":");
        body.push_str(&quote_json_string(&m.metric.key));
        body.push_str(",\"run_id\":");
        body.push_str(&quote_json_string(&m.run_id));
        body.push_str(",\"step\":");
        body.push_str(&m.metric.step.to_string());
        body.push_str(",\"timestamp\":");
        body.push_str(&m.metric.timestamp.to_string());
        body.push_str(",\"value\":");
        body.push_str(&python_float_repr(m.metric.value));
        body.push('}');
    }
    body.push_str("]}\n");

    Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .expect("valid response")
}

/// Werkzeug/Flask's default (non-debug) response for an uncaught exception:
/// `500 Internal Server Error`, `text/html; charset=utf-8`, a fixed generic
/// body with no exception detail. Verified empirically against a live Flask
/// test client hitting a view that raises a bare (non-`MlflowException`)
/// error, matching the `int("abc")` `ValueError` path in
/// `get_metric_history_bulk_handler`'s `max_results` parsing.
fn generic_500_response() -> Response {
    const BODY: &str = "<!doctype html>\n\
        <html lang=en>\n\
        <title>500 Internal Server Error</title>\n\
        <h1>Internal Server Error</h1>\n\
        <p>The server encountered an internal error and was unable to complete your request. \
        Either the server is overloaded or there is an error in the application.</p>\n";
    Response::builder()
        .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
        .header(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(axum::body::Body::from(BODY))
        .expect("valid response")
}
