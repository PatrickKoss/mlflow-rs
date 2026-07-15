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
    parse_request_with_path_params(parts, body, type_name, &[])
}

/// Same as [`parse_request`], additionally overlaying `path_params` (name,
/// value) pairs onto the parsed request as string-typed fields — the
/// mechanism REST-style path-parameter proto endpoints
/// (`/mlflow/logged-models/{model_id}`, `/mlflow/webhooks/{webhook_id}`,
/// future traces/registry path params) use to get the URL segment into the
/// request proto.
///
/// ## Why merge instead of setting the field after parsing
///
/// Python's Flask view functions receive path segments as their own function
/// arguments (`def _get_logged_model(model_id: str)`), entirely separate from
/// `_get_request_message`'s body/query parsing — the two are never merged in
/// Python. Handlers that need the path value on the *request proto* itself
/// (e.g. `_finalize_logged_model` calls `request_message.model_id`, which is
/// only present because `FinalizeLoggedModel.model_id` is ALSO required by
/// the request body schema — the client sends it twice, in the URL and the
/// JSON) rely on the body already containing it; when the wire format omits
/// it (or a GET has no query value), Python has no path-driven fallback for
/// proto fields.
///
/// Rather than special-case every path-param endpoint's handler with
/// hand-written `req.model_id = Some(...)` assignments (which would have to
/// be duplicated per endpoint and forgotten easily), this merges path
/// parameters into the request *before* proto parsing, as if they were an
/// extra query/JSON field the client had supplied. This is a deliberate,
/// documented deviation that is strictly more permissive than Python (it
/// fills in a value Python would leave absent when the client omits the
/// duplicate), and it matches Python's *observed* behavior whenever the
/// client does send both (the path segment and the body value always agree —
/// pyclient constructs both from the same `model_id`), so no real client can
/// observe a difference. Path params always win over a conflicting body/query
/// value for the same field, since real clients never send different values
/// for the two.
pub fn parse_request_with_path_params<M>(
    parts: &Parts,
    body: &Bytes,
    type_name: &str,
    path_params: &[(&str, String)],
) -> Result<M, MlflowError>
where
    M: prost::Message + Default,
{
    if parts.method == Method::GET {
        let mut pairs = match parts.uri.query() {
            Some(query) if !query.is_empty() => parse_query_pairs(query),
            _ => Vec::new(),
        };
        for (name, value) in path_params {
            pairs.retain(|(k, _)| k != name);
            pairs.push(((*name).to_string(), value.clone()));
        }
        return mlflow_proto::from_query_pairs::<M>(&pairs, type_name).map_err(codec_err);
    }

    validate_content_type(parts)?;

    // `get_json(force=True, silent=True)`: empty/whitespace body → treat as `{}`.
    let text = std::str::from_utf8(body).map_err(|_| {
        MlflowError::invalid_parameter_value("Request body is not valid UTF-8.".to_string())
    })?;
    let json = if text.trim().is_empty() { "{}" } else { text };
    let merged = merge_path_params(json, path_params)?;
    mlflow_proto::from_mlflow_json::<M>(&merged, type_name).map_err(codec_err)
}

/// Overlay `path_params` as string fields onto the parsed JSON body, replacing
/// any conflicting key. Returns the original text unchanged when there are no
/// path params (the common case), avoiding a needless parse/reserialize.
fn merge_path_params(json: &str, path_params: &[(&str, String)]) -> Result<String, MlflowError> {
    if path_params.is_empty() {
        return Ok(json.to_string());
    }
    let mut value: serde_json::Value = serde_json::from_str(json).map_err(|_| {
        MlflowError::invalid_parameter_value(
            "Malformed request. Please check the request follows JSON schema".to_string(),
        )
    })?;
    let obj = match &mut value {
        serde_json::Value::Object(obj) => obj,
        _ => {
            let mut obj = serde_json::Map::new();
            for (name, v) in path_params {
                obj.insert((*name).to_string(), serde_json::Value::String(v.clone()));
            }
            return Ok(serde_json::Value::Object(obj).to_string());
        }
    };
    for (name, v) in path_params {
        obj.insert((*name).to_string(), serde_json::Value::String(v.clone()));
    }
    Ok(value.to_string())
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
fn codec_err(e: JsonCodecError) -> MlflowError {
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
fn parse_query_pairs(query: &str) -> Vec<(String, String)> {
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
