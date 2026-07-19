//! `POST /v1/traces` — OTLP trace ingestion (plan T4.3, §3.8).
//!
//! Mirrors `mlflow/server/otel_api.py:95-259` exactly: this is NOT a
//! proto-route-table endpoint (OTLP is its own wire protocol, not MLflow's
//! `service.proto`), so it is hand-registered in `lib.rs` like the other
//! ajax-only routes.
//!
//! ## Request handling
//!
//! 1. **`Content-Type`** (`otel_api.py:128-136`): normalized by stripping
//!    parameters (`; charset=...`); must be `application/x-protobuf` or
//!    `application/json`, else 400.
//! 2. **`Content-Encoding`** (`otel_api.py:138-141`, `decompress_otlp_body`,
//!    `mlflow/tracing/utils/otlp.py:238-266`): `gzip` or `deflate` (RFC-compliant
//!    zlib-wrapped OR raw deflate), applied before parsing. Anything else, or a
//!    decompression failure, is 400.
//! 3. **Body parsing** (`otel_api.py:143-170`): protobuf via `ParseFromString`
//!    (never raises `DecodeError` in newer protobuf, so a garbage/empty parse
//!    is caught by the "no spans found" check below) or JSON (via
//!    [`json::parse_otlp_json`], our hand-rolled equivalent of
//!    `_convert_otlp_json_ids_to_base64` + `ParseJsonProto`). An empty
//!    `resource_spans` is 400 ("no spans found"); any parse failure is 400
//!    ("Invalid OpenTelemetry format").
//! 4. **Span translation** ([`translate::translate_request`]): any single span
//!    conversion failure is 422 for the WHOLE batch (`otel_api.py:204-208`) —
//!    matches Python's `try/except` inside the nested loop, which aborts on
//!    the first bad span rather than skipping it.
//! 5. **Persistence**: `store.log_spans(...)` (all-or-nothing, per trace
//!    aggregate) then, if `x-mlflow-run-id` is present, `link_traces_to_run`
//!    for every ROOT-span-completed trace id (`completed_trace_ids` —
//!    `otel_api.py:196-197, 229-233`; link failures are swallowed, matching
//!    Python's bare `except Exception: _logger.exception(...)`).
//! 6. **Response**: `200` with an empty `ExportTraceServiceResponse`, serialized
//!    per the request's content type (protobuf bytes for
//!    `application/x-protobuf`, `{}`-shaped JSON for `application/json` — OTLP
//!    JSON responses use the same camelCase/protobuf-JSON mapping as
//!    requests; an empty message serializes to `{}`).
//!
//! ## Error response shape (§ verified against FastAPI/Starlette defaults)
//!
//! None of `otel_api.py`'s six `raise HTTPException(...)` sites register a
//! custom exception handler, so FastAPI's default `http_exception_handler`
//! applies: `{"detail": "<message>"}`, compact separators (Starlette
//! `JSONResponse` uses `separators=(",", ":")`), `Content-Type:
//! application/json` (no charset). [`detail_error`] reproduces this exactly.
//! The one exception is the `MlflowException` branch
//! (`otel_api.py:221-225`), which bypasses `HTTPException` and returns
//! `JSONResponse(status_code=e.get_http_status_code(),
//! content=json.loads(e.serialize_as_json()))` directly — i.e. the
//! MLflow-shaped `{"error_code": ..., "message": ...}` body via
//! [`mlflow_error::MlflowError`]'s own `IntoResponse`.

mod json;
pub(crate) mod translate;

use std::collections::BTreeMap;
use std::io::Read;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, request::Parts, StatusCode};
use axum::response::{IntoResponse, Response};
use mlflow_error::MlflowError;
use mlflow_proto::opentelemetry::proto::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use mlflow_store::TraceTimeRange;
use prost::Message;

use crate::state::AppState;
use crate::workspace::Workspace;

/// `MLFLOW_EXPERIMENT_ID_HEADER` (`mlflow/tracing/utils/otlp.py:19`).
const EXPERIMENT_ID_HEADER: &str = "x-mlflow-experiment-id";
/// `MLFLOW_RUN_ID_HEADER` (`otlp.py:20`).
const RUN_ID_HEADER: &str = "x-mlflow-run-id";

/// `POST /v1/traces` handler.
pub async fn export_traces(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Response {
    match export_traces_impl(state, workspace, &parts, body).await {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}

/// Error paths, expressed as an enum so the handler can cleanly map each to
/// its own status/body shape (mirroring the distinct `raise` sites in
/// `otel_api.py`) without threading `Result<Response, Response>` everywhere.
#[derive(Debug)]
enum OtlpError {
    /// A plain FastAPI-`HTTPException`-shaped `{"detail": ...}` error.
    Detail { status: StatusCode, message: String },
    /// The `except MlflowException` passthrough (`otel_api.py:221-225`).
    Mlflow(MlflowError),
}

impl IntoResponse for OtlpError {
    fn into_response(self) -> Response {
        match self {
            OtlpError::Detail { status, message } => detail_error(status, &message),
            OtlpError::Mlflow(err) => err.into_response(),
        }
    }
}

fn bad_request(message: impl Into<String>) -> OtlpError {
    OtlpError::Detail {
        status: StatusCode::BAD_REQUEST,
        message: message.into(),
    }
}

fn unprocessable(message: impl Into<String>) -> OtlpError {
    OtlpError::Detail {
        status: StatusCode::UNPROCESSABLE_ENTITY,
        message: message.into(),
    }
}

/// FastAPI's default `http_exception_handler`: `{"detail": "<message>"}`,
/// Starlette `JSONResponse`'s compact `separators=(",", ":")`, no charset
/// suffix on the content-type.
fn detail_error(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({ "detail": message }).to_string();
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

async fn export_traces_impl(
    state: AppState,
    workspace: Workspace,
    parts: &Parts,
    body: Bytes,
) -> Result<Response, OtlpError> {
    let experiment_id = require_experiment_id(parts)?;
    let run_id = parts
        .headers
        .get(RUN_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let media_type = validate_content_type(parts)?;
    let decompressed = decompress_body(parts, &body)?;
    let request = parse_body(media_type, &decompressed)?;

    if request.resource_spans.is_empty() {
        return Err(bad_request("Invalid OpenTelemetry format - no spans found"));
    }

    let (spans, _service_names) =
        translate::translate_request(&request).map_err(|e| unprocessable(e.to_string()))?;

    if spans.is_empty() {
        return Ok(success_response(media_type));
    }

    let mut spans_by_trace: BTreeMap<&str, Vec<&translate::TranslatedSpan>> = BTreeMap::new();
    for span in &spans {
        spans_by_trace.entry(&span.trace_id).or_default().push(span);
    }
    let time_ranges: Vec<TraceTimeRange> = spans_by_trace
        .iter()
        .map(|(trace_id, group)| translate::compute_time_range(trace_id, group))
        .collect();
    let span_inputs: Vec<_> = spans.iter().map(translate::to_span_input).collect();
    let metric_inputs: Vec<_> = spans.iter().flat_map(|s| s.metrics.clone()).collect();

    let store = state.tracking_store();
    match store
        .log_spans(
            workspace.name(),
            &experiment_id,
            &span_inputs,
            &metric_inputs,
            &time_ranges,
        )
        .await
    {
        Ok(()) => {}
        Err(err) if err.error_code == mlflow_error::ErrorCode::NotImplemented => {
            // `except NotImplementedError` (`otel_api.py:215-220`): Python
            // reports the *store class name*; the Rust port has a single SQL
            // store implementation (`TrackingStore`), which never raises this
            // in practice — mirrored for parity should a future backend do so.
            return Err(OtlpError::Detail {
                status: StatusCode::NOT_IMPLEMENTED,
                message: format!(
                    "REST OTLP span logging is not supported by {}",
                    "TrackingStore"
                ),
            });
        }
        Err(err) => return Err(OtlpError::Mlflow(err)),
    }

    // `completed_trace_ids` (`otel_api.py:173,196-197`): traces that had a
    // root span in this batch.
    if let Some(run_id) = run_id {
        let completed_trace_ids: Vec<String> = spans_by_trace
            .iter()
            .filter(|(_, group)| group.iter().any(|s| s.is_root))
            .map(|(trace_id, _)| trace_id.to_string())
            .collect();
        if !completed_trace_ids.is_empty() {
            // Link failures are swallowed, matching Python's bare
            // `except Exception: _logger.exception(...)` (otel_api.py:229-233).
            if let Err(err) = store
                .link_traces_to_run(workspace.name(), &completed_trace_ids, &run_id)
                .await
            {
                tracing::warn!(
                    error = %err,
                    "Failed to link OpenTelemetry traces to MLflow run"
                );
            }
        }
    }

    Ok(success_response(media_type))
}

/// `x_mlflow_experiment_id: str = Header(...)` (required, `otel_api.py:98`).
/// FastAPI's dependency injection raises `RequestValidationError` (422,
/// `{"detail":[{"type":"missing",...}]}`) before the handler body runs when a
/// required header is absent; a present-but-empty header reaches the handler
/// (FastAPI only checks presence, not non-emptiness, for a plain `str`
/// header). We surface the same 422-on-missing behavior; the exact
/// `RequestValidationError` body shape is Pydantic-internal and not
/// reproduced byte-for-byte (no client relies on its structure — only the 422
/// status and the fact that no spans are ever persisted).
fn require_experiment_id(parts: &Parts) -> Result<String, OtlpError> {
    match parts.headers.get(EXPERIMENT_ID_HEADER).map(|v| v.to_str()) {
        Some(Ok(v)) => Ok(v.to_string()),
        _ => Err(OtlpError::Detail {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message: format!("Missing required header: {EXPERIMENT_ID_HEADER}"),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaType {
    Protobuf,
    Json,
}

/// `_validate content_type` inline check (`otel_api.py:128-136`): strip
/// `; charset=...` etc., accept only the two OTLP media types.
fn validate_content_type(parts: &Parts) -> Result<MediaType, OtlpError> {
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok());
    let media_type = content_type.map(|s| s.split(';').next().unwrap_or("").trim());
    match media_type {
        Some("application/x-protobuf") => Ok(MediaType::Protobuf),
        Some("application/json") => Ok(MediaType::Json),
        _ => Err(bad_request(format!(
            "Invalid Content-Type: {}. Expected: application/x-protobuf or application/json",
            content_type.unwrap_or("None")
        ))),
    }
}

/// `decompress_otlp_body` (`mlflow/tracing/utils/otlp.py:238-266`): `gzip`,
/// `deflate` (zlib-wrapped, falling back to raw deflate), else 400.
fn decompress_body(parts: &Parts, body: &[u8]) -> Result<Vec<u8>, OtlpError> {
    let Some(encoding) = parts
        .headers
        .get(header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
    else {
        return Ok(body.to_vec());
    };
    match encoding.to_ascii_lowercase().as_str() {
        "gzip" => {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(body)
                .read_to_end(&mut out)
                .map_err(|_| bad_request("Failed to decompress gzip payload"))?;
            Ok(out)
        }
        "deflate" => {
            let mut out = Vec::new();
            if flate2::read::ZlibDecoder::new(body)
                .read_to_end(&mut out)
                .is_ok()
            {
                return Ok(out);
            }
            out.clear();
            flate2::read::DeflateDecoder::new(body)
                .read_to_end(&mut out)
                .map_err(|_| bad_request("Failed to decompress deflate payload"))?;
            Ok(out)
        }
        other => Err(bad_request(format!(
            "Unsupported Content-Encoding: {other}"
        ))),
    }
}

/// Parse the (already decompressed) body per its media type
/// (`otel_api.py:143-170`).
fn parse_body(media_type: MediaType, body: &[u8]) -> Result<ExportTraceServiceRequest, OtlpError> {
    match media_type {
        MediaType::Json => {
            json::parse_otlp_json(body).map_err(|_| bad_request("Invalid OpenTelemetry format"))
        }
        MediaType::Protobuf => ExportTraceServiceRequest::decode(body)
            .map_err(|_| bad_request("Invalid OpenTelemetry format")),
    }
}

/// `200` with an empty `ExportTraceServiceResponse`
/// (`otel_api.py:252-259`), serialized per the request's content type.
fn success_response(media_type: MediaType) -> Response {
    let response = ExportTraceServiceResponse::default();
    match media_type {
        MediaType::Protobuf => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-protobuf")],
            response.encode_to_vec(),
        )
            .into_response(),
        MediaType::Json => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            // An unset `partial_success` (proto3 message-typed optional field)
            // is omitted, matching protobuf-JSON's presence rule -> `{}`.
            "{}".to_string(),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request as HttpRequest;

    fn parts_with_headers(headers: &[(&str, &str)]) -> Parts {
        let mut builder = HttpRequest::builder().uri("/v1/traces");
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        builder.body(()).unwrap().into_parts().0
    }

    #[test]
    fn validate_content_type_accepts_protobuf() {
        let parts = parts_with_headers(&[("content-type", "application/x-protobuf")]);
        assert_eq!(validate_content_type(&parts).unwrap(), MediaType::Protobuf);
    }

    #[test]
    fn validate_content_type_strips_charset_param() {
        let parts = parts_with_headers(&[("content-type", "application/json; charset=utf-8")]);
        assert_eq!(validate_content_type(&parts).unwrap(), MediaType::Json);
    }

    #[test]
    fn validate_content_type_rejects_other_types() {
        let parts = parts_with_headers(&[("content-type", "text/plain")]);
        assert!(validate_content_type(&parts).is_err());
    }

    #[test]
    fn require_experiment_id_reads_header() {
        let parts = parts_with_headers(&[(EXPERIMENT_ID_HEADER, "123")]);
        assert_eq!(require_experiment_id(&parts).unwrap(), "123");
    }

    #[test]
    fn require_experiment_id_missing_errors() {
        let parts = parts_with_headers(&[]);
        assert!(require_experiment_id(&parts).is_err());
    }

    #[test]
    fn decompress_body_passes_through_without_encoding_header() {
        let parts = parts_with_headers(&[]);
        assert_eq!(decompress_body(&parts, b"hello").unwrap(), b"hello");
    }

    #[test]
    fn decompress_body_gzip_round_trips() {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(b"hello world").unwrap();
        let compressed = encoder.finish().unwrap();

        let parts = parts_with_headers(&[("content-encoding", "gzip")]);
        assert_eq!(
            decompress_body(&parts, &compressed).unwrap(),
            b"hello world"
        );
    }

    #[test]
    fn decompress_body_rejects_unsupported_encoding() {
        let parts = parts_with_headers(&[("content-encoding", "br")]);
        assert!(decompress_body(&parts, b"x").is_err());
    }

    #[test]
    fn decompress_body_gzip_bad_payload_errors() {
        let parts = parts_with_headers(&[("content-encoding", "gzip")]);
        assert!(decompress_body(&parts, b"not gzip").is_err());
    }
}
