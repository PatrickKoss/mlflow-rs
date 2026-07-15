//! The proto-over-HTTP adapter shared by every proto-backed endpoint.
//!
//! This is the reusable spine later phases (runs, metrics, traces, registry,
//! webhooks) follow. It reproduces the request→proto→store→response pipeline of
//! Python's handlers (`mlflow/server/handlers.py`):
//!
//! 1. **Request parsing** ([`parse_request`]): for `GET` with a query string,
//!    parse query params into the proto message (T3.5, via
//!    [`mlflow_proto::from_query_pairs`]); otherwise parse the JSON body
//!    (unknown-field tolerant, `HasField`/presence preserved by prost
//!    `Option<T>`). An empty/absent body parses as `{}`, matching
//!    `_get_normalized_request_json`. Content-Type is validated for `POST`/`PUT`
//!    exactly as `_validate_content_type`.
//! 2. **Store invocation**: the handler calls the store, which returns
//!    `Result<_, mlflow_error::MlflowError>`.
//! 3. **Response serialization** ([`proto_response`]): the response proto is
//!    serialized with the MLflow JSON codec (indent=2 wire parity) and returned
//!    with `Content-Type: application/json`, mirroring
//!    `Response(mimetype="application/json"); response.set_data(message_to_json(...))`.
//!
//! Errors from any stage are `MlflowError`, whose `IntoResponse` emits the
//! golden-parity `{"error_code": ..., "message": ...}` body with the mapped HTTP
//! status.

use axum::body::Bytes;
use axum::http::request::Parts;
use axum::http::{header, Method};
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::JsonCodecError;

/// Parse a request into the proto message `M`.
///
/// `parts` carries the method, headers, and (for GET) the query string; `body`
/// is the raw request body (empty for GET). `type_name` is the fully-qualified
/// proto message name (e.g. `"mlflow.CreateExperiment"`).
pub fn parse_request<M>(parts: &Parts, body: &Bytes, type_name: &str) -> Result<M, MlflowError>
where
    M: prost::Message + Default,
{
    if parts.method == Method::GET {
        if let Some(query) = parts.uri.query() {
            if !query.is_empty() {
                let pairs = parse_query_pairs(query);
                return mlflow_proto::from_query_pairs::<M>(&pairs, type_name).map_err(codec_err);
            }
        }
        // GET with no query args → empty message (Python parses `{}`).
        return mlflow_proto::from_mlflow_json::<M>("{}", type_name).map_err(codec_err);
    }

    validate_content_type(parts)?;

    // `get_json(force=True, silent=True)`: empty/whitespace body → treat as `{}`.
    let text = std::str::from_utf8(body).map_err(|_| {
        MlflowError::invalid_parameter_value("Request body is not valid UTF-8.".to_string())
    })?;
    let json = if text.trim().is_empty() { "{}" } else { text };
    mlflow_proto::from_mlflow_json::<M>(json, type_name).map_err(codec_err)
}

/// Build a `200 application/json` response by serializing `message` with the
/// MLflow JSON codec.
pub fn proto_response<M>(message: &M, type_name: &str) -> Result<Response, MlflowError>
where
    M: prost::Message,
{
    let body = mlflow_proto::to_mlflow_json(message, type_name).map_err(codec_err)?;
    Ok(Response::builder()
        .status(axum::http::StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .expect("valid response"))
}

/// `_validate_content_type(request, ["application/json"])`, applied to POST/PUT
/// only (`mlflow/server/validation.py:5-30`).
fn validate_content_type(parts: &Parts) -> Result<(), MlflowError> {
    if parts.method != Method::POST && parts.method != Method::PUT {
        return Ok(());
    }
    let content_type = parts.headers.get(header::CONTENT_TYPE);
    let Some(content_type) = content_type else {
        return Err(MlflowError::invalid_parameter_value(
            "Bad Request. Content-Type header is missing.".to_string(),
        ));
    };
    let value = content_type.to_str().unwrap_or("");
    // Strip parameters: "application/json; charset=utf-8" -> "application/json".
    let base = value.split(';').next().unwrap_or("").trim();
    if base != "application/json" {
        return Err(MlflowError::invalid_parameter_value(
            "Bad Request. Content-Type must be one of ['application/json'].".to_string(),
        ));
    }
    Ok(())
}

/// Map a codec error to an `MlflowError`. A malformed body (`ParseJson`) is an
/// `INVALID_PARAMETER_VALUE`; an unknown message type is a server bug
/// (`INTERNAL_ERROR`); the bool-query error carries Python's verbatim message.
///
/// `pub(crate)` so handlers that need to drop down to [`mlflow_proto::from_query_pairs`]
/// directly (bypassing [`parse_request`]'s all-or-nothing parse for a field
/// that needs special per-field tolerance, e.g. `get_metric_history`'s
/// `max_results`) can still map codec errors the same way.
pub(crate) fn codec_err(e: JsonCodecError) -> MlflowError {
    match e {
        JsonCodecError::InvalidBoolQueryValue(_) => {
            MlflowError::new(e.to_string(), ErrorCode::InvalidParameterValue)
        }
        JsonCodecError::ParseJson(_) | JsonCodecError::Decode(_) | JsonCodecError::Encode(_) => {
            MlflowError::new(e.to_string(), ErrorCode::InvalidParameterValue)
        }
        JsonCodecError::UnknownMessageType(_) => {
            MlflowError::new(e.to_string(), ErrorCode::InternalError)
        }
    }
}

/// Parse a URL query string into ordered `(key, value)` pairs, percent-decoding
/// both. Repeated keys are preserved in order (matching werkzeug's `MultiDict`),
/// which the repeated-field handling in [`mlflow_proto::from_query_pairs`] needs.
///
/// `pub(crate)` so non-proto-backed handlers (e.g. the ajax-only, hand-rolled
/// `get-history-bulk`, `handlers.py:2112`) can also read repeated query
/// params without going through a proto message — axum's `Query` extractor
/// (backed by `serde_urlencoded`) doesn't support collecting repeated
/// same-named keys into a `Vec`, so it can't be used there.
pub(crate) fn parse_query_pairs(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(pair), String::new()),
        })
        .collect()
}

/// Percent-decode a query component, treating `+` as a space (form-encoding).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_val(bytes[i + 1]);
                let lo = hex_val(bytes[i + 2]);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push(h << 4 | l);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_pairs_preserve_repeated_keys_in_order() {
        let pairs = parse_query_pairs("order_by=name+DESC&order_by=id&max_results=5");
        assert_eq!(
            pairs,
            vec![
                ("order_by".to_string(), "name DESC".to_string()),
                ("order_by".to_string(), "id".to_string()),
                ("max_results".to_string(), "5".to_string()),
            ]
        );
    }

    #[test]
    fn percent_decoding_handles_encoded_values() {
        let pairs = parse_query_pairs("filter=name%20LIKE%20%27a%25%27");
        assert_eq!(
            pairs,
            vec![("filter".to_string(), "name LIKE 'a%'".to_string())]
        );
    }

    #[test]
    fn empty_query_component_yields_empty_value() {
        let pairs = parse_query_pairs("experiment_id=");
        assert_eq!(pairs, vec![("experiment_id".to_string(), String::new())]);
    }
}
