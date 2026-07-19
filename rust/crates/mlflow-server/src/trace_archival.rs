//! Cross-language codec for trace-archive `traces.pb` payloads.
//!
//! Python's `mlflow.tracing.otel.otel_archival` stores one OTLP `TracesData`
//! message containing exactly one `ResourceSpans` and one `ScopeSpans`. The
//! instrumentation scope is intentionally absent. Spans are normalized to
//! root-first display order, then by start time and span ID.
//!
//! [`stored_spans_to_traces_pb`] is the T21.2-facing writer for DB-backed
//! [`StoredSpan`] entities. [`decode_traces_pb`] returns the resource and OTLP
//! span entities directly so archive reads do not discard resource metadata.

use std::collections::{BTreeSet, HashSet};

use axum::body::Bytes;
use base64::Engine;
use futures::stream::{self, StreamExt};
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::opentelemetry::proto::common::v1::{
    any_value, AnyValue, ArrayValue, KeyValue, KeyValueList,
};
use mlflow_proto::opentelemetry::proto::resource::v1::Resource;
use mlflow_proto::opentelemetry::proto::trace::v1::{
    span, status, ResourceSpans, ScopeSpans, Span, Status, TracesData,
};
use mlflow_store::{
    StoredSpan, TraceInfo, TrackingStore, WorkspaceStore, SPANS_LOCATION_ARCHIVE_REPO,
    TRACE_TAG_ARCHIVE_LOCATION, TRACE_TAG_SPANS_LOCATION,
};
use prost::Message;
use serde_json::Value;

/// Filename used inside every trace archive repository.
pub const TRACE_ARCHIVAL_FILENAME: &str = "traces.pb";

/// The canonical, single-resource archive entity represented by `traces.pb`.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceArchive {
    pub resource: Resource,
    pub spans: Vec<Span>,
}

/// Errors raised while translating or validating an archive payload.
#[derive(Debug, thiserror::Error)]
pub enum TraceArchivalCodecError {
    #[error("Archived trace payload must include at least one span.")]
    EmptySpans,
    #[error("Archived trace payload must be a non-empty OTLP TracesData protobuf.")]
    EmptyPayload,
    #[error("Archived trace payload must be a valid OTLP TracesData protobuf: {0}")]
    InvalidProtobuf(#[from] prost::DecodeError),
    #[error("Archived trace payload must contain exactly one ResourceSpans group.")]
    ResourceSpansCount,
    #[error("Archived trace payload must contain exactly one ScopeSpans group.")]
    ScopeSpansCount,
    #[error("Archived trace payload must contain spans for a single OTLP trace.")]
    MultipleTraceIds,
    #[error("Archived span is missing required field `{0}`.")]
    MissingField(&'static str),
    #[error("Archived span field `{field}` is invalid: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("Failed to parse stored span content: {0}")]
    InvalidStoredContent(#[from] serde_json::Error),
}

/// Serialize a canonical archive to Python-compatible OTLP protobuf bytes.
pub fn encode_traces_pb(archive: &TraceArchive) -> Result<Vec<u8>, TraceArchivalCodecError> {
    validate_single_trace(&archive.spans)?;
    let mut spans = archive.spans.clone();
    sort_spans_root_first(&mut spans);

    Ok(TracesData {
        resource_spans: vec![ResourceSpans {
            // Python uses `CopyFrom`, which preserves message presence even
            // for an empty resource (`resource {}`).
            resource: Some(archive.resource.clone()),
            scope_spans: vec![ScopeSpans {
                scope: None,
                spans,
                schema_url: String::new(),
            }],
            schema_url: String::new(),
        }],
    }
    .encode_to_vec())
}

/// Reconstruct stored MLflow span entities and serialize a `traces.pb` payload.
///
/// DB span JSON does not retain an OTel SDK resource, matching Python's
/// `Span.from_dict`, so this path emits a present but empty `Resource`.
pub fn stored_spans_to_traces_pb(spans: &[StoredSpan]) -> Result<Vec<u8>, TraceArchivalCodecError> {
    let spans = spans
        .iter()
        .map(stored_span_to_otel_span)
        .collect::<Result<Vec<_>, _>>()?;
    encode_traces_pb(&TraceArchive {
        resource: Resource::default(),
        spans,
    })
}

/// Parse and validate a Python- or Rust-written `traces.pb` payload.
pub fn decode_traces_pb(data: &[u8]) -> Result<TraceArchive, TraceArchivalCodecError> {
    if data.is_empty() {
        return Err(TraceArchivalCodecError::EmptyPayload);
    }
    let traces_data = TracesData::decode(data)?;
    if traces_data.resource_spans.len() != 1 {
        return Err(TraceArchivalCodecError::ResourceSpansCount);
    }
    let mut resource_spans = traces_data.resource_spans.into_iter().next().unwrap();
    if resource_spans.scope_spans.len() != 1 {
        return Err(TraceArchivalCodecError::ScopeSpansCount);
    }
    let mut spans = resource_spans.scope_spans.remove(0).spans;
    validate_single_trace(&spans)?;
    sort_spans_root_first(&mut spans);
    Ok(TraceArchive {
        resource: resource_spans.resource.unwrap_or_default(),
        spans,
    })
}

/// Scheduler-facing, single-workspace archival pass. The caller supplies the
/// already-resolved workspace location and retention plus the remaining
/// pass-level budget; no scheduling or interval gating happens here.
pub async fn archive_traces(
    store: &TrackingStore,
    workspace: &str,
    resolved_trace_archival_location: &str,
    broader_retention: &str,
    long_retention_allowlist: &[String],
    max_traces_per_pass: Option<usize>,
) -> Result<u64, MlflowError> {
    archive_traces_at(
        store,
        workspace,
        resolved_trace_archival_location,
        broader_retention,
        long_retention_allowlist,
        max_traces_per_pass,
        chrono::Utc::now().timestamp_millis(),
    )
    .await
}

/// Resolve server/workspace configuration and run one bounded archival pass.
/// This is the T21.4 hand-off: the scheduler owns timing, locking, fairness,
/// and its cross-workspace remaining budget; this function owns one workspace.
pub async fn archive_traces_for_workspace(
    store: &TrackingStore,
    workspace_store: Option<&WorkspaceStore>,
    workspace: &str,
    config: &crate::TraceArchivalServerConfig,
    remaining_budget: Option<usize>,
) -> Result<u64, MlflowError> {
    if !config.enabled {
        return Ok(0);
    }
    let (mut location, retention, append_workspace_prefix) =
        if let Some(workspace_store) = workspace_store {
            let resolved = workspace_store
                .resolve_trace_archival_config(&config.location, &config.retention, workspace)
                .await?;
            (
                resolved
                    .config
                    .location
                    .unwrap_or_else(|| config.location.clone()),
                resolved
                    .config
                    .retention
                    .unwrap_or_else(|| config.retention.clone()),
                resolved.append_workspace_prefix,
            )
        } else {
            (config.location.clone(), config.retention.clone(), false)
        };
    if append_workspace_prefix {
        location = format!(
            "{}/workspaces/{}",
            location.trim_end_matches('/'),
            workspace.trim_matches('/')
        );
    }
    let configured_budget = config
        .max_traces_per_pass
        .and_then(|value| usize::try_from(value).ok());
    let budget = match (remaining_budget, configured_budget) {
        (Some(remaining), Some(configured)) => Some(remaining.min(configured)),
        (Some(remaining), None) => Some(remaining),
        (None, configured) => configured,
    };
    archive_traces(
        store,
        workspace,
        &location,
        &retention,
        &config.long_retention_allowlist,
        budget,
    )
    .await
}

/// Deterministic-clock variant used by differential tests and T21.4.
pub async fn archive_traces_at(
    store: &TrackingStore,
    workspace: &str,
    resolved_trace_archival_location: &str,
    broader_retention: &str,
    long_retention_allowlist: &[String],
    max_traces_per_pass: Option<usize>,
    now_millis: i64,
) -> Result<u64, MlflowError> {
    if resolved_trace_archival_location.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "`resolved_trace_archival_location` must be provided.",
        ));
    }
    if broader_retention.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "`broader_retention` must be provided.",
        ));
    }
    if max_traces_per_pass == Some(0) {
        return Err(MlflowError::invalid_parameter_value(
            "`max_traces_per_pass` must be a positive integer, received 0.",
        ));
    }

    let allowlist = long_retention_allowlist
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let (archive_now_requests, candidates) = store
        .plan_trace_archival(
            workspace,
            now_millis,
            broader_retention,
            &allowlist,
            max_traces_per_pass,
        )
        .await?;
    let mut archived = 0;
    let archive_now_experiments = archive_now_requests
        .iter()
        .map(|request| request.experiment_id.as_str())
        .collect::<HashSet<_>>();
    let mut retryable_failure_experiment_ids = HashSet::new();
    for candidate in candidates {
        let data = match store
            .load_trace_archival_data(workspace, &candidate.trace_id)
            .await
        {
            Ok(Some(data)) => data,
            Ok(None) => continue,
            Err(error) => {
                if archive_now_experiments.contains(candidate.experiment_id.as_str()) {
                    retryable_failure_experiment_ids.insert(candidate.experiment_id.clone());
                }
                tracing::warn!(trace_id = %candidate.trace_id, %error, "failed to load trace archival snapshot");
                continue;
            }
        };
        let payload = match stored_spans_to_traces_pb(&data.spans) {
            Ok(payload) => payload,
            Err(error) => {
                tracing::warn!(trace_id = %candidate.trace_id, %error, "marking malformed trace during archival");
                if let Err(error) = store
                    .mark_trace_archival_malformed(
                        workspace,
                        &candidate.trace_id,
                        data.db_payload_generation,
                    )
                    .await
                {
                    if archive_now_experiments.contains(candidate.experiment_id.as_str()) {
                        retryable_failure_experiment_ids.insert(candidate.experiment_id.clone());
                    }
                    tracing::warn!(trace_id = %candidate.trace_id, %error, "failed to mark malformed trace");
                }
                continue;
            }
        };
        let artifact_uri = append_archive_artifact_uri(
            resolved_trace_archival_location,
            &candidate.experiment_id,
            &candidate.trace_id,
        );
        let repo = match mlflow_artifacts::factory::repo_from_uri(&artifact_uri) {
            Ok(repo) => repo,
            Err(error) => {
                if archive_now_experiments.contains(candidate.experiment_id.as_str()) {
                    retryable_failure_experiment_ids.insert(candidate.experiment_id.clone());
                }
                tracing::warn!(trace_id = %candidate.trace_id, %error, "failed to resolve trace archive repository");
                continue;
            }
        };
        let body = stream::once(async move { Ok(Bytes::from(payload)) }).boxed();
        if let Err(error) = repo.put(TRACE_ARCHIVAL_FILENAME, body).await {
            if archive_now_experiments.contains(candidate.experiment_id.as_str()) {
                retryable_failure_experiment_ids.insert(candidate.experiment_id.clone());
            }
            let _ = repo.delete(TRACE_ARCHIVAL_FILENAME).await;
            tracing::warn!(trace_id = %candidate.trace_id, %error, "trace archival upload failed");
            continue;
        }
        let finalized = match store
            .finalize_archived_trace(
                workspace,
                &candidate.trace_id,
                &artifact_uri,
                data.db_payload_generation,
            )
            .await
        {
            Ok(finalized) => finalized,
            Err(error) => {
                if archive_now_experiments.contains(candidate.experiment_id.as_str()) {
                    retryable_failure_experiment_ids.insert(candidate.experiment_id.clone());
                }
                delete_unreferenced_payload(
                    store,
                    workspace,
                    &candidate.trace_id,
                    &artifact_uri,
                    repo.as_ref(),
                )
                .await;
                tracing::warn!(trace_id = %candidate.trace_id, %error, "trace archival finalization failed");
                continue;
            }
        };
        if finalized {
            archived += 1;
        } else {
            delete_unreferenced_payload(
                store,
                workspace,
                &candidate.trace_id,
                &artifact_uri,
                repo.as_ref(),
            )
            .await;
        }
    }
    store
        .clear_completed_archive_now_requests(
            workspace,
            &archive_now_requests,
            now_millis,
            &retryable_failure_experiment_ids,
        )
        .await?;
    Ok(archived)
}

/// Download an archive payload. Repository failures (including missing files)
/// resolve to empty spans, while an empty or malformed payload is corruption,
/// matching `ArtifactRepository.download_archived_trace_data`.
pub async fn download_archived_spans(trace_info: &TraceInfo) -> Result<Vec<Span>, MlflowError> {
    let artifact_uri = trace_info.tag(TRACE_TAG_ARCHIVE_LOCATION).ok_or_else(|| {
        MlflowError::new(
            format!(
                "Trace data is corrupted for request_id={}",
                trace_info.trace_id
            ),
            ErrorCode::InvalidState,
        )
    })?;
    let repo = mlflow_artifacts::factory::repo_from_uri(artifact_uri)?;
    let download = match repo.get(TRACE_ARCHIVAL_FILENAME).await {
        Ok(download) => download,
        Err(_) => return Ok(Vec::new()),
    };
    let bytes = collect_download(download).await?;
    if bytes.is_empty() {
        return Err(archive_payload_corrupted());
    }
    decode_traces_pb(&bytes)
        .map(|archive| archive.spans)
        .map_err(|_| archive_payload_corrupted())
}

/// Archive-backed `get-trace-artifact` JSON reconstruction.
pub async fn download_archived_trace_json(
    trace_info: &TraceInfo,
) -> Result<serde_json::Value, MlflowError> {
    let artifact_uri = trace_info.tag(TRACE_TAG_ARCHIVE_LOCATION).ok_or_else(|| {
        MlflowError::new(
            format!(
                "Trace data is corrupted for request_id={}",
                trace_info.trace_id
            ),
            ErrorCode::InvalidState,
        )
    })?;
    let repo = mlflow_artifacts::factory::repo_from_uri(artifact_uri)?;
    let download = match repo.get(TRACE_ARCHIVAL_FILENAME).await {
        Ok(download) => download,
        Err(_) => return Ok(serde_json::json!({ "spans": [] })),
    };
    let bytes = collect_download(download).await?;
    if bytes.is_empty() {
        return Err(archive_payload_corrupted());
    }
    let archive = decode_traces_pb(&bytes).map_err(|_| archive_payload_corrupted())?;
    let spans = archive
        .spans
        .iter()
        .map(|span| {
            crate::otlp::translate::archived_span_content(span, &archive.resource)
                .map_err(|_| archive_payload_corrupted())
                .and_then(|content| {
                    serde_json::from_str(&content).map_err(|_| archive_payload_corrupted())
                })
        })
        .collect::<Result<Vec<serde_json::Value>, MlflowError>>()?;
    Ok(serde_json::json!({ "spans": spans }))
}

fn append_archive_artifact_uri(root: &str, experiment_id: &str, trace_id: &str) -> String {
    format!(
        "{}/{}/traces/{}/artifacts",
        root.trim_end_matches('/'),
        experiment_id.trim_matches('/'),
        trace_id.trim_matches('/')
    )
}

async fn delete_unreferenced_payload(
    store: &TrackingStore,
    workspace: &str,
    trace_id: &str,
    artifact_uri: &str,
    repo: &dyn mlflow_artifacts::ArtifactRepo,
) {
    if let Ok(info) = store.get_trace_info(workspace, trace_id).await {
        if info.tag(TRACE_TAG_SPANS_LOCATION) == Some(SPANS_LOCATION_ARCHIVE_REPO)
            && info.tag(TRACE_TAG_ARCHIVE_LOCATION) == Some(artifact_uri)
        {
            return;
        }
    }
    if let Err(error) = repo.delete(TRACE_ARCHIVAL_FILENAME).await {
        tracing::warn!(%trace_id, %error, "failed to delete unreferenced archived payload");
    }
}

async fn collect_download(
    download: mlflow_artifacts::ArtifactDownload,
) -> Result<Vec<u8>, MlflowError> {
    let mut bytes = Vec::with_capacity(download.size.max(0) as usize);
    let mut stream = download.stream;
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk?);
    }
    Ok(bytes)
}

fn archive_payload_corrupted() -> MlflowError {
    MlflowError::new(
        format!("Trace data is corrupted for path={TRACE_ARCHIVAL_FILENAME}"),
        ErrorCode::InvalidState,
    )
}

fn validate_single_trace(spans: &[Span]) -> Result<(), TraceArchivalCodecError> {
    if spans.is_empty() {
        return Err(TraceArchivalCodecError::EmptySpans);
    }
    let mut trace_ids = BTreeSet::new();
    for span in spans {
        if span.trace_id.is_empty() {
            return Err(TraceArchivalCodecError::MissingField("trace_id"));
        }
        if span.span_id.is_empty() {
            return Err(TraceArchivalCodecError::MissingField("span_id"));
        }
        trace_ids.insert(span.trace_id.as_slice());
    }
    if trace_ids.len() != 1 {
        return Err(TraceArchivalCodecError::MultipleTraceIds);
    }
    Ok(())
}

fn sort_spans_root_first(spans: &mut [Span]) {
    spans.sort_by(|left, right| {
        let left_root = left.parent_span_id.is_empty();
        let right_root = right.parent_span_id.is_empty();
        right_root
            .cmp(&left_root)
            .then_with(|| left.start_time_unix_nano.cmp(&right.start_time_unix_nano))
            .then_with(|| left.span_id.cmp(&right.span_id))
    });
}

/// Reconstruct one stored span's JSON entity as Python's
/// `Span.from_dict(...).to_otel_proto()` does.
pub(crate) fn stored_span_to_otel_span(
    stored: &StoredSpan,
) -> Result<Span, TraceArchivalCodecError> {
    let content: Value = serde_json::from_str(&stored.content)?;
    let object = content
        .as_object()
        .ok_or(TraceArchivalCodecError::InvalidField {
            field: "content",
            message: "expected a JSON object".to_string(),
        })?;

    let trace_id = decode_base64(required_str(object, "trace_id")?, "trace_id")?;
    let span_id = decode_base64(required_str(object, "span_id")?, "span_id")?;
    let parent_span_id = optional_str(object, "parent_span_id")?
        .map(|value| decode_base64(value, "parent_span_id"))
        .transpose()?
        .unwrap_or_default();
    let name = required_str(object, "name")?.to_string();
    let start_time_unix_nano = required_u64(object, "start_time_unix_nano")?;
    let end_time_unix_nano = optional_u64(object, "end_time_unix_nano")?.unwrap_or(0);

    let status_value = object
        .get("status")
        .and_then(Value::as_object)
        .ok_or(TraceArchivalCodecError::MissingField("status"))?;
    let status = Status {
        message: status_value
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        code: match status_value.get("code").and_then(Value::as_str) {
            Some("STATUS_CODE_OK") | Some("OK") => status::StatusCode::Ok as i32,
            Some("STATUS_CODE_ERROR") | Some("ERROR") => status::StatusCode::Error as i32,
            _ => status::StatusCode::Unset as i32,
        },
    };

    let attributes = object
        .get("attributes")
        .and_then(Value::as_object)
        .ok_or(TraceArchivalCodecError::MissingField("attributes"))?
        .iter()
        .map(|(key, value)| key_value_from_json(key, &decode_stored_attribute(value)))
        .collect();

    let events = object
        .get("events")
        .and_then(Value::as_array)
        .map(|events| events.iter().map(event_from_json).collect())
        .transpose()?
        .unwrap_or_default();
    let links = object
        .get("links")
        .and_then(Value::as_array)
        .map(|links| links.iter().map(link_from_json).collect())
        .transpose()?
        .unwrap_or_default();

    Ok(Span {
        trace_id,
        span_id,
        trace_state: String::new(),
        parent_span_id,
        flags: 0,
        name,
        kind: 0,
        start_time_unix_nano,
        end_time_unix_nano,
        attributes,
        dropped_attributes_count: 0,
        events,
        dropped_events_count: 0,
        links,
        dropped_links_count: 0,
        status: Some(status),
    })
}

fn event_from_json(value: &Value) -> Result<span::Event, TraceArchivalCodecError> {
    let object = value
        .as_object()
        .ok_or(TraceArchivalCodecError::InvalidField {
            field: "events",
            message: "expected an event object".to_string(),
        })?;
    let attributes = object
        .get("attributes")
        .and_then(Value::as_object)
        .map(|attrs| {
            attrs
                .iter()
                // Event attributes in Span.to_dict are logical values,
                // unlike the JSON-encoded top-level span attributes.
                .map(|(key, value)| key_value_from_json(key, value))
                .collect()
        })
        .unwrap_or_default();
    Ok(span::Event {
        time_unix_nano: required_u64(object, "time_unix_nano")?,
        name: required_str(object, "name")?.to_string(),
        attributes,
        dropped_attributes_count: 0,
    })
}

fn link_from_json(value: &Value) -> Result<span::Link, TraceArchivalCodecError> {
    let object = value
        .as_object()
        .ok_or(TraceArchivalCodecError::InvalidField {
            field: "links",
            message: "expected a link object".to_string(),
        })?;
    let trace_id = decode_link_trace_id(required_str(object, "trace_id")?)?;
    let span_id = decode_hex(required_str(object, "span_id")?, "links.span_id")?;
    let attributes = object
        .get("attributes")
        .and_then(Value::as_object)
        .map(|attrs| {
            attrs
                .iter()
                .map(|(key, value)| key_value_from_json(key, value))
                .collect()
        })
        .unwrap_or_default();
    Ok(span::Link {
        trace_id,
        span_id,
        trace_state: String::new(),
        attributes,
        dropped_attributes_count: 0,
        flags: 0,
    })
}

fn decode_link_trace_id(value: &str) -> Result<Vec<u8>, TraceArchivalCodecError> {
    let hex = if let Some(rest) = value.strip_prefix("trace:/") {
        rest.split_once('/').map(|(_, id)| id).unwrap_or(rest)
    } else {
        value.strip_prefix("tr-").unwrap_or(value)
    };
    decode_hex(hex, "links.trace_id")
}

fn decode_hex(value: &str, field: &'static str) -> Result<Vec<u8>, TraceArchivalCodecError> {
    if !value.len().is_multiple_of(2) {
        return Err(invalid_field(field, "hex value has odd length"));
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16)
                .map_err(|error| invalid_field(field, error.to_string()))
        })
        .collect()
}

fn decode_base64(value: &str, field: &'static str) -> Result<Vec<u8>, TraceArchivalCodecError> {
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|error| invalid_field(field, error.to_string()))
}

fn required_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, TraceArchivalCodecError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or(TraceArchivalCodecError::MissingField(field))
}

fn optional_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<Option<&'a str>, TraceArchivalCodecError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(invalid_field(field, "expected a string or null")),
    }
}

fn required_u64(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<u64, TraceArchivalCodecError> {
    optional_u64(object, field)?.ok_or(TraceArchivalCodecError::MissingField(field))
}

fn optional_u64(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<Option<u64>, TraceArchivalCodecError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| invalid_field(field, "expected a non-negative integer")),
        Some(Value::String(value)) => value
            .parse::<u64>()
            .map(Some)
            .map_err(|error| invalid_field(field, error.to_string())),
        Some(_) => Err(invalid_field(field, "expected an integer or null")),
    }
}

fn invalid_field(field: &'static str, message: impl Into<String>) -> TraceArchivalCodecError {
    TraceArchivalCodecError::InvalidField {
        field,
        message: message.into(),
    }
}

fn decode_stored_attribute(value: &Value) -> Value {
    match value {
        Value::String(serialized) => {
            serde_json::from_str(serialized).unwrap_or_else(|_| value.clone())
        }
        other => other.clone(),
    }
}

fn any_value_from_json(value: &Value) -> AnyValue {
    let value = match value {
        Value::Null => None,
        Value::Bool(value) => Some(any_value::Value::BoolValue(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(any_value::Value::IntValue)
            .or_else(|| value.as_f64().map(any_value::Value::DoubleValue)),
        Value::String(value) => Some(any_value::Value::StringValue(value.clone())),
        Value::Array(items) => Some(any_value::Value::ArrayValue(ArrayValue {
            values: items.iter().map(any_value_from_json).collect(),
        })),
        Value::Object(object) => Some(any_value::Value::KvlistValue(KeyValueList {
            values: object
                .iter()
                .map(|(key, value)| key_value_from_json(key, value))
                .collect(),
        })),
    };
    AnyValue { value }
}

fn key_value_from_json(key: &str, value: &Value) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        // Python merely accesses `kv.value` for `None`; protobuf does not mark
        // the submessage present until a oneof member is assigned.
        value: (!value.is_null()).then(|| any_value_from_json(value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(trace: u8, id: u8, parent: Option<u8>, start: u64) -> Span {
        Span {
            trace_id: vec![trace; 16],
            span_id: vec![id; 8],
            parent_span_id: parent.map(|value| vec![value; 8]).unwrap_or_default(),
            start_time_unix_nano: start,
            name: format!("span-{id}"),
            ..Default::default()
        }
    }

    #[test]
    fn writer_and_reader_sort_root_first_with_stable_ties() {
        let archive = TraceArchive {
            resource: Resource::default(),
            spans: vec![
                span(1, 4, Some(2), 10),
                span(1, 3, None, 20),
                span(1, 2, None, 10),
                span(1, 1, Some(2), 10),
            ],
        };
        let decoded = decode_traces_pb(&encode_traces_pb(&archive).unwrap()).unwrap();
        let ids: Vec<u8> = decoded.spans.iter().map(|span| span.span_id[0]).collect();
        assert_eq!(ids, vec![2, 3, 1, 4]);
    }

    #[test]
    fn rejects_noncanonical_wrapper_and_multiple_traces() {
        assert!(matches!(
            decode_traces_pb(&[]),
            Err(TraceArchivalCodecError::EmptyPayload)
        ));
        assert!(matches!(
            decode_traces_pb(
                &TracesData {
                    resource_spans: vec![ResourceSpans::default(), ResourceSpans::default()],
                }
                .encode_to_vec()
            ),
            Err(TraceArchivalCodecError::ResourceSpansCount)
        ));
        let archive = TraceArchive {
            resource: Resource::default(),
            spans: vec![span(1, 1, None, 1), span(2, 2, None, 2)],
        };
        assert!(matches!(
            encode_traces_pb(&archive),
            Err(TraceArchivalCodecError::MultipleTraceIds)
        ));
    }
}
