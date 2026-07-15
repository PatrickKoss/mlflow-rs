//! Tracing V2 endpoints (plan T4.2, §3.7): the 7 deprecated-but-still-served
//! V2 trace RPCs, thin adapters over the same store paths [`crate::traces`]
//! (V3) uses.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`:
//! `_deprecated_start_trace_v2` (`:5097`), `_deprecated_end_trace_v2`
//! (`:5125`), `_deprecated_get_trace_info_v2` (`:5159`),
//! `_deprecated_search_traces_v2` (`:5172`), `_delete_traces` (`:3989`, V2 and
//! V3 share this exact handler in Python — same request/response proto shape,
//! registered separately per version prefix), `_set_trace_tag` (`:4055`),
//! `_delete_trace_tag` (`:4092`).
//!
//! ## Routing note
//!
//! V2 endpoints have `since.major = 2`, so the route table registers them
//! only under `/api/2.0/...` + `/ajax-api/2.0/...` — the V3 twins
//! (`since.major = 3`) live at the `/api/3.0/...` prefix, so there is no path
//! collision even though several V2/V3 pairs share the same *tail* (e.g. both
//! `startTrace` and `startTraceV3` are `POST .../mlflow/traces`).
//!
//! ## `TraceInfoV2` response shape
//!
//! Unlike V3's `TraceInfoV3` (maps for tags/metadata), the V2 wire proto
//! (`mlflow.TraceInfo`) carries `tags`/`request_metadata` as repeated
//! `{key, value}` messages and truncates them on the way out
//! (`TraceInfoV2.to_proto`, `mlflow/entities/trace_info_v2.py:70-97`):
//! metadata keys/values to `MAX_CHARS_IN_TRACE_INFO_METADATA` (250 chars
//! each), tag keys to `MAX_CHARS_IN_TRACE_INFO_TAGS_KEY` (250), tag values to
//! `MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE` (4096). [`to_proto_trace_info_v2`]
//! reproduces this. (Python's V3 `TraceInfo.to_proto` truncates identically,
//! but T4.1's `traces::to_proto_trace_info` does not yet — out of scope here,
//! left as-is per the "keep shared-file changes additive" instruction.)

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_proto::quote_json_string;
use mlflow_store::TraceInfo;

use crate::proto_http::{parse_request, parse_request_with_path_params, proto_response};
use crate::state::AppState;
use crate::traces::{missing_param, path_param_pairs, require_non_empty};
use crate::workspace::Workspace;

/// `MAX_CHARS_IN_TRACE_INFO_METADATA` (`mlflow/tracing/constant.py:173`).
const MAX_CHARS_IN_TRACE_INFO_METADATA: usize = 250;
/// `MAX_CHARS_IN_TRACE_INFO_TAGS_KEY` (`mlflow/tracing/constant.py:176`).
const MAX_CHARS_IN_TRACE_INFO_TAGS_KEY: usize = 250;
/// `MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE` (`mlflow/tracing/constant.py:177`).
const MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE: usize = 4096;

/// `SEARCH_TRACES` handler-level `max_results` threshold, shared by V2 and V3
/// (`handlers.py:3961`/`:5187`, `_assert_less_than_or_equal(int(x), 500)`).
const SEARCH_TRACES_MAX_RESULTS: i32 = 500;
/// Default `max_results` for `searchTraces` V2 (proto default = 100).
const SEARCH_TRACES_DEFAULT_MAX_RESULTS: i32 = 100;

// ---------------------------------------------------------------------------
// startTrace (V2)
// ---------------------------------------------------------------------------

/// `_deprecated_start_trace_v2` (`handlers.py:5097`), path: `POST
/// /mlflow/traces`. Unlike V3, `experiment_id`/`timestamp_ms` have no
/// `_assert_required` in Python's schema — a missing `experiment_id` fails
/// later inside the store (`parse_experiment_id`/`get_experiment` on an
/// empty/None id), and a missing `timestamp_ms` is passed through as `0`
/// (proto3 scalar default, no `HasField` check in the handler).
pub async fn start_trace(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::StartTrace = parse_request(&parts, &body, "mlflow.StartTrace")?;
    validate_map_key_present_metadata(&req.request_metadata, "request_metadata")?;
    validate_map_key_present_tags(&req.tags, "tags")?;

    let experiment_id = req.experiment_id.unwrap_or_default();
    let timestamp_ms = req.timestamp_ms.unwrap_or(0);
    let request_metadata = kv_pairs_from_metadata(&req.request_metadata);
    let tags = kv_pairs_from_tags(&req.tags);

    let info = state
        .tracking_store()
        .deprecated_start_trace_v2(
            workspace.name(),
            &experiment_id,
            timestamp_ms,
            &request_metadata,
            &tags,
        )
        .await?;

    let resp = pb::start_trace::Response {
        trace_info: Some(to_proto_trace_info_v2(&info)),
    };
    proto_response(&resp, "mlflow.StartTrace.Response")
}

// ---------------------------------------------------------------------------
// endTrace (V2)
// ---------------------------------------------------------------------------

/// `_deprecated_end_trace_v2` (`handlers.py:5125`), path: `PATCH
/// /mlflow/traces/{request_id}`. Like `startTrace`, `timestamp_ms`/`status`
/// have no `_assert_required`: a missing `status` defaults to
/// `TRACE_STATUS_UNSPECIFIED` (proto enum default 0), a missing
/// `timestamp_ms` to `0`.
pub async fn end_trace(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::EndTrace = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.EndTrace",
        &path_param_pairs(&path_params, &["request_id"]),
    )?;
    validate_map_key_present_metadata(&req.request_metadata, "request_metadata")?;
    validate_map_key_present_tags(&req.tags, "tags")?;

    let request_id = req.request_id.unwrap_or_default();
    let timestamp_ms = req.timestamp_ms.unwrap_or(0);
    let status = trace_status_name(req.status.unwrap_or(0));
    let request_metadata = kv_pairs_from_metadata(&req.request_metadata);
    let tags = kv_pairs_from_tags(&req.tags);

    let info = state
        .tracking_store()
        .deprecated_end_trace_v2(
            workspace.name(),
            &request_id,
            timestamp_ms,
            status,
            &request_metadata,
            &tags,
        )
        .await?;

    let resp = pb::end_trace::Response {
        trace_info: Some(to_proto_trace_info_v2(&info)),
    };
    proto_response(&resp, "mlflow.EndTrace.Response")
}

// ---------------------------------------------------------------------------
// getTraceInfo (V2)
// ---------------------------------------------------------------------------

/// `_deprecated_get_trace_info_v2` (`handlers.py:5159`), path: `GET
/// /mlflow/traces/{request_id}/info`.
pub async fn get_trace_info(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
) -> Result<Response, MlflowError> {
    let request_id = path_params.get("request_id").cloned().unwrap_or_default();
    let info = state
        .tracking_store()
        .get_trace_info(workspace.name(), &request_id)
        .await?;
    let resp = pb::get_trace_info::Response {
        trace_info: Some(to_proto_trace_info_v2(&info)),
    };
    proto_response(&resp, "mlflow.GetTraceInfo.Response")
}

// ---------------------------------------------------------------------------
// searchTraces (V2, GET) — also the UI's "contains traces" check
// (`GET /ajax-api/2.0/mlflow/traces`).
// ---------------------------------------------------------------------------

/// `_deprecated_search_traces_v2` (`handlers.py:5172`), path: `GET
/// /mlflow/traces`.
pub async fn search_traces(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchTraces = parse_request(&parts, &body, "mlflow.SearchTraces")?;
    if req.experiment_ids.is_empty() {
        return Err(missing_param("experiment_ids"));
    }
    let max_results = req.max_results.unwrap_or(SEARCH_TRACES_DEFAULT_MAX_RESULTS);
    if max_results > SEARCH_TRACES_MAX_RESULTS {
        // Byte-matched to `_assert_less_than_or_equal(..., 500)` (bare
        // AssertionError → `invalid_value`, same shape as `searchTracesV3`).
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. \
             See the API docs for more information about request parameters."
        )));
    }

    let page = state
        .tracking_store()
        .search_traces(
            workspace.name(),
            &req.experiment_ids,
            req.filter.as_deref().filter(|s| !s.is_empty()),
            max_results as i64,
            &req.order_by,
            req.page_token.as_deref().filter(|s| !s.is_empty()),
        )
        .await?;

    let resp = pb::search_traces::Response {
        traces: page
            .trace_infos
            .iter()
            .map(to_proto_trace_info_v2)
            .collect(),
        next_page_token: page.next_page_token,
    };
    proto_response(&resp, "mlflow.SearchTraces.Response")
}

// ---------------------------------------------------------------------------
// deleteTraces (V2)
// ---------------------------------------------------------------------------

/// `_delete_traces` (`handlers.py:3989`) — Python registers the exact same
/// handler function for both `deleteTraces` (V2) and `deleteTracesV3`; only
/// the URL version prefix differs. Bound here to the `DeleteTraces` message
/// (identical fields to `DeleteTracesV3`), path: `POST
/// /mlflow/traces/delete-traces`. See
/// [`crate::traces::delete_traces_v3`] for the `HasField` semantics this
/// mirrors.
pub async fn delete_traces(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteTraces = parse_request(&parts, &body, "mlflow.DeleteTraces")?;
    let experiment_id = require_non_empty(req.experiment_id.as_deref(), "experiment_id")?;

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

    let resp = pb::delete_traces::Response {
        traces_deleted: Some(deleted as i32),
    };
    proto_response(&resp, "mlflow.DeleteTraces.Response")
}

// ---------------------------------------------------------------------------
// setTraceTag / deleteTraceTag (V2)
// ---------------------------------------------------------------------------

/// `_set_trace_tag` (`handlers.py:4055`), path: `PATCH
/// /mlflow/traces/{request_id}/tags`.
pub async fn set_trace_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetTraceTag = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.SetTraceTag",
        &path_param_pairs(&path_params, &["request_id"]),
    )?;
    let request_id = require_non_empty(req.request_id.as_deref(), "request_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;
    let value = req.value.unwrap_or_default();

    state
        .tracking_store()
        .set_trace_tag(workspace.name(), request_id, key, &value)
        .await?;
    proto_response(
        &pb::set_trace_tag::Response {},
        "mlflow.SetTraceTag.Response",
    )
}

/// `_delete_trace_tag` (`handlers.py:4092`), path: `DELETE
/// /mlflow/traces/{request_id}/tags` (`?key=`).
pub async fn delete_trace_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteTraceTag = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.DeleteTraceTag",
        &path_param_pairs(&path_params, &["request_id"]),
    )?;
    let request_id = require_non_empty(req.request_id.as_deref(), "request_id")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;

    state
        .tracking_store()
        .delete_trace_tag(workspace.name(), request_id, key)
        .await?;
    proto_response(
        &pb::delete_trace_tag::Response {},
        "mlflow.DeleteTraceTag.Response",
    )
}

// ===========================================================================
// Proto conversion helpers
// ===========================================================================

/// Map a store [`TraceInfo`] to the wire V2 `TraceInfo` proto, mirroring
/// `TraceInfoV2.from_v3(...).to_proto()`: `execution_time_ms` substitutes `0`
/// for `None` (Python's proto setter can't express nullable ints), and
/// metadata/tags are truncated per-field (see the module doc comment).
fn to_proto_trace_info_v2(info: &TraceInfo) -> pb::TraceInfo {
    pb::TraceInfo {
        request_id: Some(info.trace_id.clone()),
        experiment_id: Some(info.experiment_id.clone()),
        timestamp_ms: Some(info.request_time),
        execution_time_ms: Some(info.execution_duration.unwrap_or(0)),
        status: Some(trace_status_to_proto(&info.state)),
        request_metadata: info
            .trace_metadata
            .iter()
            .map(|(k, v)| pb::TraceRequestMetadata {
                key: Some(truncate(k, MAX_CHARS_IN_TRACE_INFO_METADATA)),
                value: Some(truncate(
                    v.as_deref().unwrap_or(""),
                    MAX_CHARS_IN_TRACE_INFO_METADATA,
                )),
            })
            .collect(),
        tags: info
            .tags
            .iter()
            .map(|(k, v)| pb::TraceTag {
                key: Some(truncate(k, MAX_CHARS_IN_TRACE_INFO_TAGS_KEY)),
                value: Some(truncate(
                    v.as_deref().unwrap_or(""),
                    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
                )),
            })
            .collect(),
    }
}

/// Truncate a `&str` to at most `max_chars` **characters** (not bytes),
/// matching Python's `s[:max_chars]` on a `str` (code-point slicing).
fn truncate(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// `TraceStatus.to_proto()`: `ProtoTraceStatus.Value(self)` — the store status
/// string is already the exact enum member name (`TraceState::OK` ==
/// `"OK"` etc.), and the V2 `TraceStatus` enum shares the same int values as
/// `TraceInfoV3.State` (`TRACE_STATUS_UNSPECIFIED=0, OK=1, ERROR=2,
/// IN_PROGRESS=3`).
fn trace_status_to_proto(state: &str) -> i32 {
    use mlflow_store::TraceState;
    match state {
        TraceState::OK => pb::TraceStatus::Ok as i32,
        TraceState::ERROR => pb::TraceStatus::Error as i32,
        TraceState::IN_PROGRESS => pb::TraceStatus::InProgress as i32,
        _ => pb::TraceStatus::Unspecified as i32,
    }
}

/// `TraceStatus.from_proto(proto_status)` ==
/// `TraceStatus(ProtoTraceStatus.Name(proto_status))`: map the wire int back
/// to the store's status string. Unknown ints fall back to
/// `STATE_UNSPECIFIED` (proto3 would reject an out-of-range enum value during
/// parsing in Python; prost is lenient, so this is a defensive default).
fn trace_status_name(proto_status: i32) -> &'static str {
    use mlflow_store::TraceState;
    match proto_status {
        1 => TraceState::OK,
        2 => TraceState::ERROR,
        3 => TraceState::IN_PROGRESS,
        _ => TraceState::STATE_UNSPECIFIED,
    }
}

/// Extract `(key, value)` pairs from parsed `TraceRequestMetadata` entries,
/// mirroring `{e.key: e.value for e in request_message.request_metadata}`
/// (later entries with a duplicate key win, matching Python dict-comprehension
/// semantics — last write wins).
fn kv_pairs_from_metadata(entries: &[pb::TraceRequestMetadata]) -> Vec<(String, String)> {
    dedup_last_wins(
        entries
            .iter()
            .map(|e| {
                (
                    e.key.clone().unwrap_or_default(),
                    e.value.clone().unwrap_or_default(),
                )
            })
            .collect(),
    )
}

/// Same as [`kv_pairs_from_metadata`] for `TraceTag` entries.
fn kv_pairs_from_tags(entries: &[pb::TraceTag]) -> Vec<(String, String)> {
    dedup_last_wins(
        entries
            .iter()
            .map(|e| {
                (
                    e.key.clone().unwrap_or_default(),
                    e.value.clone().unwrap_or_default(),
                )
            })
            .collect(),
    )
}

/// `{k: v for k, v in pairs}`: keep only the last value per key, preserving
/// first-seen key order (irrelevant for correctness here since callers only
/// use these as unordered KV writes, but keeps output deterministic for
/// tests).
fn dedup_last_wins(pairs: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, String> = HashMap::new();
    for (k, v) in pairs {
        if !map.contains_key(&k) {
            order.push(k.clone());
        }
        map.insert(k, v);
    }
    order
        .into_iter()
        .map(|k| {
            let v = map.remove(&k).unwrap_or_default();
            (k, v)
        })
        .collect()
}

/// `_assert_map_key_present`: each entry must carry a non-empty `key`
/// (`handlers.py:854`). Applies to `StartTrace`/`EndTrace`'s
/// `request_metadata`/`tags` fields — repeated `{key, value}` messages in the
/// V2 wire format (V3 uses `map<string,string>` fields instead, which cannot
/// have a missing key by construction, so this check has no V3 analog).
///
/// Reproduces `_validate_param_against_schema`'s fallback message for a bare
/// `AssertionError` from a non-`_assert_required` schema function:
/// `invalid_value(param, value, " Hint: Value was of type 'list'.")` — `value`
/// is the raw JSON array, serialized the way `json.dumps(..., sort_keys=True,
/// separators=(",", ":"))` would (though object-key sorting inside each
/// `{key, value}` entry is skipped here since both possible keys, `key` and
/// `value`, already sort alphabetically — `k` < `v` — so the common case
/// matches exactly).
fn validate_map_key_present_generic<T>(
    entries: &[T],
    param: &str,
    key_of: impl Fn(&T) -> &Option<String>,
    value_of: impl Fn(&T) -> &Option<String>,
) -> Result<(), MlflowError> {
    for entry in entries {
        let key = key_of(entry);
        if key.as_deref().unwrap_or("").is_empty() {
            let rendered = render_entries_json(entries, &key_of, &value_of);
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {rendered} for parameter '{param}' supplied:  Hint: Value was of \
                 type 'list'. See the API docs for more information about request parameters."
            )));
        }
    }
    Ok(())
}

fn validate_map_key_present_metadata(
    entries: &[pb::TraceRequestMetadata],
    param: &str,
) -> Result<(), MlflowError> {
    validate_map_key_present_generic(
        entries,
        param,
        |e: &pb::TraceRequestMetadata| &e.key,
        |e: &pb::TraceRequestMetadata| &e.value,
    )
}

fn validate_map_key_present_tags(entries: &[pb::TraceTag], param: &str) -> Result<(), MlflowError> {
    validate_map_key_present_generic(
        entries,
        param,
        |e: &pb::TraceTag| &e.key,
        |e: &pb::TraceTag| &e.value,
    )
}

/// `json.dumps([{"key": ..., "value": ...}, ...], sort_keys=True,
/// separators=(",", ":"))`: compact, alphabetically-sorted-key JSON. `key` <
/// `value` alphabetically, so sorting is a fixed `key` first then `value`
/// (both entries always present in the rendered dict — Python's raw JSON dict
/// may omit `value`, but `_assert_map_key_present` only inspects `key`, so a
/// `None` key/value here renders as JSON `null`, matching Python's dict
/// `.get()` default when a client truly omits the field).
fn render_entries_json<T>(
    entries: &[T],
    key_of: &impl Fn(&T) -> &Option<String>,
    value_of: &impl Fn(&T) -> &Option<String>,
) -> String {
    let mut out = String::from("[");
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str("\"key\":");
        match key_of(entry) {
            Some(k) => out.push_str(&quote_json_string(k)),
            None => out.push_str("null"),
        }
        out.push(',');
        out.push_str("\"value\":");
        match value_of(entry) {
            Some(v) => out.push_str(&quote_json_string(v)),
            None => out.push_str("null"),
        }
        out.push('}');
    }
    out.push(']');
    out
}
