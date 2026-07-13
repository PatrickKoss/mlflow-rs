//! Tracing V3 tables: `trace_info`, `trace_tags`, `trace_request_metadata`,
//! `trace_metrics`, `spans`, `span_metrics`, `assessments`.
//!
//! Mirrors `SqlTraceInfo`, `SqlTraceTag`, `SqlTraceMetadata`, `SqlTraceMetrics`,
//! `SqlSpan`, `SqlSpanMetrics`, and `SqlAssessments`
//! (`mlflow/store/tracking/dbmodels/models.py`).

use sqlx::FromRow;

pub const TRACE_INFO: &str = "trace_info";
pub const TRACE_TAGS: &str = "trace_tags";
pub const TRACE_REQUEST_METADATA: &str = "trace_request_metadata";
pub const TRACE_METRICS: &str = "trace_metrics";
pub const SPANS: &str = "spans";
pub const SPAN_METRICS: &str = "span_metrics";
pub const ASSESSMENTS: &str = "assessments";

/// Row of the `trace_info` table (`SqlTraceInfo`). PK `request_id` (trace_id).
///
/// `db_payload_generation` is a DB `Integer` (server default `0`).
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct TraceInfo {
    pub request_id: String,
    pub experiment_id: i64,
    pub timestamp_ms: i64,
    pub execution_time_ms: Option<i64>,
    pub status: String,
    pub client_request_id: Option<String>,
    pub request_preview: Option<String>,
    pub response_preview: Option<String>,
    pub db_payload_generation: i64,
}

/// Row of the `trace_tags` table (`SqlTraceTag`). PK `(request_id, key)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct TraceTag {
    pub key: String,
    pub value: Option<String>,
    pub request_id: String,
}

/// Row of the `trace_request_metadata` table (`SqlTraceMetadata`).
///
/// PK `(request_id, key)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct TraceRequestMetadata {
    pub key: String,
    pub value: Option<String>,
    pub request_id: String,
}

/// Row of the `trace_metrics` table (`SqlTraceMetrics`). PK `(request_id, key)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct TraceMetric {
    pub request_id: String,
    pub key: String,
    pub value: Option<f64>,
}

/// Row of the `spans` table (`SqlSpan`). PK `(trace_id, span_id)`.
///
/// `duration_ns` is a *stored/persisted generated column*
/// (`end_time_unix_nano - start_time_unix_nano`, `models.py:2010`) — the store
/// never writes it; it is read-only and NULL for in-progress spans.
/// `dimension_attributes` is a JSON column stored here as raw JSON text.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Span {
    pub trace_id: String,
    pub experiment_id: i64,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: Option<String>,
    #[sqlx(rename = "type")]
    pub span_type: Option<String>,
    pub status: String,
    pub start_time_unix_nano: i64,
    pub end_time_unix_nano: Option<i64>,
    /// Read-only generated column; never written by the store.
    pub duration_ns: Option<i64>,
    pub content: String,
    pub dimension_attributes: Option<String>,
}

/// Row of the `span_metrics` table (`SqlSpanMetrics`).
///
/// PK `(trace_id, span_id, key)`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct SpanMetric {
    pub trace_id: String,
    pub span_id: String,
    pub key: String,
    pub value: Option<f64>,
}

/// Row of the `assessments` table (`SqlAssessments`). PK `assessment_id`.
///
/// `value`/`error`/`rationale`/`assessment_metadata` are `Text` (JSON payloads
/// for the JSON-typed ones). `valid` defaults to `true`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Assessment {
    pub assessment_id: String,
    pub trace_id: String,
    pub name: String,
    pub assessment_type: String,
    pub value: String,
    pub error: Option<String>,
    pub created_timestamp: i64,
    pub last_updated_timestamp: i64,
    pub source_type: String,
    pub source_id: Option<String>,
    pub run_id: Option<String>,
    pub span_id: Option<String>,
    pub rationale: Option<String>,
    pub overrides: Option<String>,
    pub valid: bool,
    pub assessment_metadata: Option<String>,
}
