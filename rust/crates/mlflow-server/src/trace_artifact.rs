//! `GET /ajax-api/{2,3}.0/mlflow/get-trace-artifact` (plan T4.5, §3.10):
//! `get_trace_artifact_handler` (`handlers.py:4211-4278`) and everything it
//! calls (`_fetch_trace_data_from_store`, `_get_trace_artifact_repo`,
//! `ArtifactRepositoryBase.download_trace_data`/`download_trace_attachment`,
//! `_response_with_file_attachment_headers`).
//!
//! This is an ajax-only, non-proto endpoint (plain `request_id`/`path` query
//! params, not a proto message), so it is hand-registered in `lib.rs` like
//! `get-history-bulk` rather than driven by the route table.
//!
//! ## Dispatch (mirrors `_fetch_trace_data_from_store` + the handler's
//! fallback block exactly — NOT `TrackingStore::get_trace`, whose `getTrace`
//! V3-proto-endpoint semantics silently return empty spans for anything that
//! isn't `TRACKING_STORE`/`ARCHIVE_REPO`; this handler must instead *fall
//! through to the artifact repo* in that case)
//!
//! * `request_id` missing → 400 `BAD_REQUEST`.
//! * `path` present (attachment fetch) → `validate_path_is_safe`, then
//!   ALWAYS through the artifact repo (`_get_trace_artifact_repo`,
//!   regardless of `spansLocation`) — `download_trace_attachment` further
//!   requires `path` be a canonical UUID.
//! * No `path`: dispatch on the trace's `mlflow.trace.spansLocation` tag:
//!   - `TRACKING_STORE` → build `{"spans": [...]}` from the DB-backed spans
//!     (`Trace.data.to_dict()`).
//!   - `ARCHIVE_REPO` → decode `traces.pb` from `mlflow.trace.archiveLocation`.
//!   - anything else (including `ARTIFACT_REPO`, or no tag at all) →
//!     `download_trace_data()`: read `traces.json` from the artifact repo.
//!
//! ## Response shape
//!
//! Both the spans-JSON and attachment responses are `send_file(..., mimetype=
//! "application/octet-stream", as_attachment=True, ...)` immediately
//! overwritten by `_response_with_file_attachment_headers`, which re-guesses
//! the content type from the *download filename* (`_guess_mime_type`) and
//! always sets `Content-Disposition: attachment` + `X-Content-Type-Options:
//! nosniff`. For the spans-JSON path the filename is the constant
//! `TRACE_DATA_FILE_NAME` (`"traces.json"`, extension `json` →
//! `text/plain`); for the attachment path the filename is the (UUID) `path`
//! itself (no extension → `application/octet-stream`).
//!
//! The spans-JSON body is `json.dumps(trace_data)` — Python's *default*
//! separators (`", "` / `": "`), NOT the `indent=2` pretty-printer
//! `mlflow-proto`'s proto-JSON codec uses for normal 2xx bodies. See
//! [`compact_json_dumps`].

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::{python_float_repr, quote_json_string};
use mlflow_store::{
    StoredSpan, TraceInfo, MLFLOW_ARTIFACT_LOCATION, SPANS_LOCATION_ARCHIVE_REPO,
    SPANS_LOCATION_TRACKING_STORE, TRACE_TAG_SPANS_LOCATION,
};

use crate::proto_http::parse_query_pairs;
use crate::state::AppState;
use crate::workspace::Workspace;

/// `TRACE_DATA_FILE_NAME` (`mlflow/tracing/utils/artifact_utils.py:6`).
const TRACE_DATA_FILE_NAME: &str = "traces.json";

/// `get_trace_artifact_handler` (`handlers.py:4211`).
pub async fn get_trace_artifact(
    axum::extract::State(state): axum::extract::State<AppState>,
    workspace: Workspace,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let pairs = parts.uri.query().map(parse_query_pairs).unwrap_or_default();
    let request_id = query_param(&pairs, "request_id").filter(|s| !s.is_empty());
    let path = query_param(&pairs, "path").filter(|s| !s.is_empty());

    let Some(request_id) = request_id else {
        return Err(MlflowError::new(
            "Request must include the \"request_id\" query parameter.",
            ErrorCode::BadRequest,
        ));
    };

    let store = state.tracking_store();

    if let Some(path) = path {
        let safe_path = mlflow_artifacts::validate_path_is_safe(&path)?;
        let trace_info = store.get_trace_info(workspace.name(), &request_id).await?;
        let repo = repo_for_trace(&trace_info)?;
        let content_bytes = download_trace_attachment(repo.as_ref(), &safe_path).await?;
        return Ok(attachment_response(&safe_path, content_bytes));
    }

    let trace_data = fetch_trace_data_from_store(store, &workspace, &request_id).await?;
    let trace_data = match trace_data {
        Some(v) => v,
        None => {
            let trace_info = store.get_trace_info(workspace.name(), &request_id).await?;
            if trace_info.tag(TRACE_TAG_SPANS_LOCATION) == Some(SPANS_LOCATION_ARCHIVE_REPO) {
                crate::trace_archival::download_archived_trace_json(&trace_info).await?
            } else {
                let repo = repo_for_trace(&trace_info)?;
                download_trace_data(repo.as_ref()).await?
            }
        }
    };

    Ok(spans_json_response(&trace_data))
}

fn query_param(pairs: &[(String, String)], name: &str) -> Option<String> {
    pairs
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
}

/// `_fetch_trace_data_from_store` (`handlers.py:4175`): `Some(dict)` when the
/// trace is `TRACKING_STORE`-backed (`{"spans": [...]}`), `None` to signal
/// "fall back to the artifact repo" for everything else (including
/// `ARTIFACT_REPO` and an absent tag). `ARCHIVE_REPO` is also `None` here —
/// the caller checks the tag again to route it to the archive repository.
async fn fetch_trace_data_from_store(
    store: &mlflow_store::TrackingStore,
    workspace: &Workspace,
    trace_id: &str,
) -> Result<Option<serde_json::Value>, MlflowError> {
    let trace_info = store.get_trace_info(workspace.name(), trace_id).await?;
    if trace_info.tag(TRACE_TAG_SPANS_LOCATION) != Some(SPANS_LOCATION_TRACKING_STORE) {
        return Ok(None);
    }
    // `allow_partial=True` ("allow partial so the frontend can render
    // in-progress traces") — always returns spans for a TRACKING_STORE trace,
    // regardless of export completeness.
    let trace = store.get_trace(workspace.name(), trace_id, true).await?;
    Ok(Some(trace_data_to_json(&trace.spans)?))
}

/// `TraceData.to_dict()`: `{"spans": [span.to_dict() for span in spans]}`.
/// `LazySpan.to_dict()` on a TRACKING_STORE-sourced span returns the stored
/// dict essentially verbatim (see `handlers.py` module docs above) — the
/// stored `content` JSON *is* already the `span.to_dict()` shape, so this is
/// a parse-and-collect, not a reconstruction.
///
/// One documented gap: Python's `translate_loaded_span` additionally
/// backfills a `mlflow.spanType` attribute from third-party OTEL semantic
/// convention attributes (OpenInference/Traceloop span-kind hints) when it is
/// missing or `"UNKNOWN"`. That compatibility shim for externally-produced
/// OTEL spans is not ported here — spans logged through this server's own
/// `log_spans` already carry `mlflow.spanType`, so this only diverges for
/// spans ingested from a third-party OTEL SDK without an MLflow-aware
/// exporter, which is out of scope for v1 byte-parity.
fn trace_data_to_json(spans: &[StoredSpan]) -> Result<serde_json::Value, MlflowError> {
    let mut out = Vec::with_capacity(spans.len());
    for span in spans {
        let value: serde_json::Value = serde_json::from_str(&span.content).map_err(|e| {
            MlflowError::internal_error(format!("Failed to parse stored span content: {e}"))
        })?;
        out.push(value);
    }
    Ok(serde_json::json!({ "spans": out }))
}

/// Resolve the [`mlflow_artifacts::ArtifactRepo`] for a trace's artifact
/// location, mirroring `_get_trace_artifact_repo` /
/// `get_artifact_uri_for_trace` (`mlflow/tracing/utils/artifact_utils.py:13`):
/// the `mlflow.artifactLocation` tag IS the artifact URI (written by
/// `start_trace`/`log_spans`, plan T2.10). A missing tag mirrors Python's
/// `MlflowTraceDataCorrupted` (a trace should always carry this tag).
///
/// Unlike Python, this does not resolve `mlflow-artifacts://` proxy URIs to
/// the server's `--artifacts-destination` root (that resolution — and the
/// full run/registry artifact-URI plumbing generally — is Phase 5 (T5.1/T5.2)
/// server wiring, not yet landed); such schemes fall through to
/// [`mlflow_artifacts::factory::repo_from_uri`]'s own `NOT_IMPLEMENTED` for
/// unrecognized/unsupported schemes.
fn repo_for_trace(
    trace_info: &TraceInfo,
) -> Result<std::sync::Arc<dyn mlflow_artifacts::ArtifactRepo>, MlflowError> {
    let artifact_uri = trace_info.tag(MLFLOW_ARTIFACT_LOCATION).ok_or_else(|| {
        MlflowError::new(
            format!(
                "Trace data is corrupted for request_id={}",
                trace_info.trace_id
            ),
            ErrorCode::InvalidState,
        )
    })?;
    mlflow_artifacts::factory::repo_from_uri(artifact_uri)
}

/// `ArtifactRepositoryBase.download_trace_data` as overridden by
/// `LocalArtifactRepository.download_trace_data` (`local_artifact_repo.py:238`),
/// which delegates to `try_read_trace_data` (`artifact_repo.py:525`): read
/// `traces.json`, erroring `NOT_FOUND` (missing/empty) or `INVALID_STATE`
/// (bad JSON).
///
/// Deviation: Python's message embeds the *absolute local filesystem path* to
/// the temp/artifact-dir copy of `traces.json`, which is either a fresh temp
/// directory per call (the generic `ArtifactRepositoryBase` path) or the
/// artifact directory (the `LocalArtifactRepository` override) — neither is a
/// stable, backend-agnostic value the [`mlflow_artifacts::ArtifactRepo`]
/// trait exposes (it is generic over `object_store` backends, not just local
/// FS). We use the repo-relative artifact path (`"traces.json"`) instead,
/// matching the style `send_artifact`'s own not-found message already uses.
async fn download_trace_data(
    repo: &dyn mlflow_artifacts::ArtifactRepo,
) -> Result<serde_json::Value, MlflowError> {
    let bytes = match repo.get(TRACE_DATA_FILE_NAME).await {
        Ok(dl) => collect(dl).await?,
        Err(e) if e.error_code == ErrorCode::ResourceDoesNotExist => {
            return Err(trace_data_not_found());
        }
        Err(e) => return Err(e),
    };
    if bytes.is_empty() {
        return Err(trace_data_not_found());
    }
    serde_json::from_slice(&bytes).map_err(|_| {
        MlflowError::new(
            format!("Trace data is corrupted for path={TRACE_DATA_FILE_NAME}"),
            ErrorCode::InvalidState,
        )
    })
}

fn trace_data_not_found() -> MlflowError {
    MlflowError::new(
        format!("Trace data not found for path={TRACE_DATA_FILE_NAME}"),
        ErrorCode::NotFound,
    )
}

/// `ArtifactRepositoryBase.download_trace_attachment`
/// (`artifact_repo.py:450`) + `_validate_attachment_path` (`artifact_repo.py:684`):
/// the attachment lives at `attachments/{path}` within the repo, and `path`
/// must be a canonical (lowercase, hyphenated) UUID string.
async fn download_trace_attachment(
    repo: &dyn mlflow_artifacts::ArtifactRepo,
    path: &str,
) -> Result<Vec<u8>, MlflowError> {
    validate_attachment_path(path)?;
    let full = format!("attachments/{path}");
    let dl = repo.get(&full).await.map_err(|e| {
        // Mirrors the handler's `except Exception` catch-all around
        // `download_trace_attachment` (any non-`MlflowException` failure is
        // wrapped); a repo miss IS an `MlflowException`
        // (`RESOURCE_DOES_NOT_EXIST`) in Python too (`_download_file`), so it
        // passes through unchanged, matching `except MlflowException: raise`.
        if e.error_code == ErrorCode::ResourceDoesNotExist {
            e
        } else {
            MlflowError::internal_error(format!(
                "Failed to download attachment '{path}' for trace."
            ))
        }
    })?;
    collect(dl).await
}

/// Port of `_validate_attachment_path`: `path` must round-trip through
/// `uuid.UUID(path)` to the identical canonical string (rejects uppercase,
/// braces, missing hyphens, etc. — anything `str(UUID(path)) != path`).
fn validate_attachment_path(path: &str) -> Result<(), MlflowError> {
    match uuid::Uuid::parse_str(path) {
        Ok(parsed) if hyphenated_lowercase(&parsed) == path => Ok(()),
        _ => Err(MlflowError::new(
            format!("Invalid attachment path: '{path}'. Attachment path must be a valid UUID."),
            ErrorCode::InvalidParameterValue,
        )),
    }
}

fn hyphenated_lowercase(id: &uuid::Uuid) -> String {
    id.hyphenated().to_string()
}

async fn collect(dl: mlflow_artifacts::ArtifactDownload) -> Result<Vec<u8>, MlflowError> {
    use futures::stream::StreamExt;
    let mut out = Vec::with_capacity(dl.size.max(0) as usize);
    let mut stream = dl.stream;
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk?);
    }
    Ok(out)
}

/// Build the `traces.json` response: `send_file(..., mimetype=
/// "application/octet-stream")` then `_response_with_file_attachment_headers`
/// overwrites content-type via `_guess_mime_type("traces.json")` → `text/plain`
/// (the `json` extension is in MLflow's text-extension allowlist) and sets
/// `Content-Disposition: attachment; filename=traces.json` +
/// `X-Content-Type-Options: nosniff`. Body is `json.dumps(trace_data)` — the
/// *default* separator style, not `mlflow-proto`'s `indent=2` proto codec.
fn spans_json_response(trace_data: &serde_json::Value) -> Response {
    let body = compact_json_dumps(trace_data);
    let mime = mlflow_artifacts::mime::guess_mime_type(TRACE_DATA_FILE_NAME);
    let content_disposition =
        mlflow_artifacts::mime::content_disposition_attachment(TRACE_DATA_FILE_NAME);
    build_attachment_response(body.into_bytes(), &mime, &content_disposition)
}

/// Build the attachment response: same header shape as above, but the
/// filename (and therefore the guessed mime type / Content-Disposition) is
/// the requested `path` (`download_name=path`,
/// `_response_with_file_attachment_headers(path, ...)`).
fn attachment_response(path: &str, content_bytes: Vec<u8>) -> Response {
    let mime = mlflow_artifacts::mime::guess_mime_type(path);
    let content_disposition = mlflow_artifacts::mime::content_disposition_attachment(path);
    build_attachment_response(content_bytes, &mime, &content_disposition)
}

fn build_attachment_response(body: Vec<u8>, mime: &str, content_disposition: &str) -> Response {
    let len = body.len() as u64;
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(body))
        .unwrap_or_else(|_| {
            MlflowError::internal_error("Failed to build response").into_response()
        });
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(mime)
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(content_disposition)
            .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(len));
    headers.insert(
        "X-Content-Type-Options",
        HeaderValue::from_static("nosniff"),
    );
    response
}

// ---------------------------------------------------------------------------
// Python `json.dumps` default-separator serialization
// ---------------------------------------------------------------------------

/// Serialize a [`serde_json::Value`] exactly like Python's `json.dumps(value)`
/// with its *default* arguments: `", "` / `": "` separators, no indentation,
/// `ensure_ascii=True` string escaping, `allow_nan=True` float literals.
/// Distinct from `mlflow-proto`'s `to_mlflow_json` pretty-printer
/// (`indent=2`), which normal proto 2xx bodies use — this handler's body is
/// hand-built via plain `json.dumps`, not the proto JSON codec.
fn compact_json_dumps(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_compact(&mut out, value);
    out
}

fn write_compact(out: &mut String, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&python_float_repr(n.as_f64().unwrap_or(0.0)));
            }
        }
        serde_json::Value::String(s) => out.push_str(&quote_json_string(s)),
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_compact(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(entries) => {
            out.push('{');
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&quote_json_string(key));
                out.push_str(": ");
                write_compact(out, val);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_json_dumps_matches_python_default_separators() {
        let value = serde_json::json!({"spans": [{"a": 1, "b": [1, 2]}]});
        assert_eq!(
            compact_json_dumps(&value),
            r#"{"spans": [{"a": 1, "b": [1, 2]}]}"#
        );
    }

    #[test]
    fn compact_json_dumps_empty_containers() {
        assert_eq!(
            compact_json_dumps(&serde_json::json!({"spans": []})),
            r#"{"spans": []}"#
        );
    }

    #[test]
    fn validate_attachment_path_accepts_canonical_uuid_only() {
        assert!(validate_attachment_path("550e8400-e29b-41d4-a716-446655440000").is_ok());
        assert!(validate_attachment_path("550E8400-E29B-41D4-A716-446655440000").is_err());
        assert!(validate_attachment_path("not-a-uuid").is_err());
        assert!(validate_attachment_path("../etc/passwd").is_err());
    }
}
