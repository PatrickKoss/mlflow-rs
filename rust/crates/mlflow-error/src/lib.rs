//! `mlflow-error`: the MLflow error model, with exact wire parity against
//! `mlflow/exceptions.py` and the error-related bits of
//! `mlflow/server/handlers.py` / `mlflow/server/auth/__init__.py`.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` §4.6 / Phase 1 T1.4:
//!
//! - [`MlflowError`] mirrors `mlflow.exceptions.MlflowException`: an
//!   [`ErrorCode`] (re-exported from `mlflow-proto`) plus a message, with
//!   `sqlstate` / `error_class` auto-derived exactly like Python's
//!   `__init__` (see the `classification` module).
//! - [`MlflowError::http_status`] mirrors
//!   `MlflowException.get_http_status_code()` (`ERROR_CODE_TO_HTTP_STATUS`,
//!   default HTTP 500 for unmapped codes) — see the `status` module.
//! - `IntoResponse for MlflowError` mirrors
//!   `catch_mlflow_exception`'s `Response` construction exactly: the body is
//!   `serialize_as_json()` (**compact** `json.dumps`, *not* the
//!   pretty-printed proto JSON used for normal 2xx bodies), mimetype
//!   `application/json`.
//! - [`not_implemented_response`] mirrors `_not_implemented()`: empty body,
//!   HTTP 404, Flask's default `text/html; charset=utf-8` content type.
//! - [`unauthenticated_response`] / [`forbidden_response`] mirror
//!   `mlflow/server/auth/__init__.py`'s `make_basic_auth_response` /
//!   `make_forbidden_response`: plain-text bodies, `WWW-Authenticate: Basic
//!   realm="mlflow"` on the 401.
//!
//! ## Usage in future endpoint handlers
//!
//! Axum handlers should return `Result<T, MlflowError>` (or `MlflowError`
//! directly for infallible-but-erroring paths); `?` on any fallible store
//! call that yields an `MlflowError` composes naturally since `MlflowError`
//! implements [`axum::response::IntoResponse`]. Construct errors via the
//! `MlflowError::*` constructors below (mirroring
//! `MlflowException.invalid_parameter_value` etc.) rather than building the
//! JSON body by hand, so status/sqlstate/error_class stay centrally correct.

mod classification;
mod status;

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

pub use mlflow_proto::mlflow::ErrorCode;
pub use status::{http_status, DEFAULT_HTTP_STATUS, MAPPED_ERROR_CODE_COUNT};

/// Mirrors `mlflow.exceptions.MlflowException`.
///
/// `sqlstate` and `error_class` are auto-derived from `error_code` at
/// construction time (see `classification::derive`), exactly matching
/// Python's `__init__` when no explicit `sqlstate`/`error_class` override is
/// passed at the raise site — which covers every server-side raise (see the
/// module docs on why the `_CP_*` tables don't apply here).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{error_code:?}: {message}")]
pub struct MlflowError {
    pub error_code: ErrorCode,
    pub message: String,
    pub sqlstate: Option<&'static str>,
    pub error_class: Option<&'static str>,
}

impl MlflowError {
    /// Mirrors `MlflowException(message, error_code=...)`.
    pub fn new(message: impl Into<String>, error_code: ErrorCode) -> Self {
        let (error_class, sqlstate) = classification::derive(error_code.as_str_name());
        Self {
            error_code,
            message: message.into(),
            sqlstate,
            error_class,
        }
    }

    /// Mirrors `MlflowException.invalid_parameter_value`.
    pub fn invalid_parameter_value(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::InvalidParameterValue)
    }

    /// Mirrors the common `MlflowException(msg, error_code=RESOURCE_DOES_NOT_EXIST)`
    /// raise-site pattern (e.g. `mlflow/utils/validation.py`, store `get_*`
    /// lookups) — Python has no dedicated classmethod for this one, callers
    /// construct it directly, so this is a Rust-side convenience mirroring
    /// that idiom.
    pub fn resource_does_not_exist(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::ResourceDoesNotExist)
    }

    /// Mirrors the common `MlflowException(msg, error_code=RESOURCE_ALREADY_EXISTS)`
    /// raise-site pattern.
    pub fn resource_already_exists(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::ResourceAlreadyExists)
    }

    /// Mirrors `MlflowException(msg, error_code=INTERNAL_ERROR)`, which is
    /// also `MlflowException`'s own default `error_code`.
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::InternalError)
    }

    /// Mirrors `MlflowException(msg, error_code=INVALID_STATE)`.
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::InvalidState)
    }

    /// Mirrors `MlflowException(msg, error_code=PERMISSION_DENIED)`.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::PermissionDenied)
    }

    /// Mirrors `MlflowException(msg, error_code=RESOURCE_CONFLICT)`.
    pub fn resource_conflict(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::ResourceConflict)
    }

    /// Mirrors `MlflowNotImplementedException` /
    /// `MlflowException(msg, error_code=NOT_IMPLEMENTED)`.
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::new(message, ErrorCode::NotImplemented)
    }

    /// Mirrors `MlflowException.get_http_status_code()`.
    pub fn http_status(&self) -> StatusCode {
        status::http_status(self.error_code)
    }

    /// Mirrors `MlflowException.serialize_as_json()`: Python's
    /// `json.dumps(exception_dict)` with **default separators** — i.e.
    /// `", "` / `": "` (a space after every comma and colon), NOT the
    /// fully-compact `","`/`":"` `serde_json::to_string` produces, and NOT
    /// the `indent=2` pretty-printing used for normal 2xx proto bodies (§4
    /// item 3). Keys are in insertion order `error_code`, `message`, then
    /// `sqlstate` / `error_class` if present.
    pub fn serialize_as_json(&self) -> String {
        let body = SerializedError {
            error_code: self.error_code.as_str_name(),
            message: &self.message,
            sqlstate: self.sqlstate,
            error_class: self.error_class,
        };
        // `serde_json` has no built-in "default Python separators" formatter,
        // so serialize compactly (field order still matches, per struct decl
        // order + `skip_serializing_if`) and then widen the separators to
        // match `json.dumps`'s default `", "` / `": "` output byte-for-byte.
        let compact =
            serde_json::to_string(&body).expect("MlflowError fields are all valid UTF-8 strings");
        widen_json_separators(&compact)
    }
}

/// Field order matches `MlflowException.serialize_as_json`'s
/// `{"error_code": ..., "message": ...}` dict literal followed by conditional
/// `sqlstate` / `error_class` inserts.
#[derive(Serialize)]
struct SerializedError<'a> {
    error_code: &'a str,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    sqlstate: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_class: Option<&'static str>,
}

/// Rewrites compact-serde_json separators (`,` and `:` with no surrounding
/// space) to Python `json.dumps`'s default separators (`, ` and `: `),
/// without touching commas/colons that occur inside string values.
///
/// This is safe for [`SerializedError`] specifically because none of its
/// field values are attacker/user-controlled structural JSON — `message` is
/// always serialized as a JSON string, so any `,`/`:` inside it is already
/// escaped-or-quoted by `serde_json` and this function only ever sees
/// top-level structural separators between the fixed set of known field
/// values, EXCEPT that `message` content itself can legitimately contain
/// literal `,`/`:` characters inside the (already-quoted) string. We must
/// therefore only widen separators OUTSIDE of string literals.
fn widen_json_separators(compact: &str) -> String {
    let mut out = String::with_capacity(compact.len() + 8);
    let mut in_string = false;
    let mut escaped = false;
    for c in compact.chars() {
        match c {
            '"' if !escaped => {
                in_string = !in_string;
                out.push(c);
            }
            '\\' if in_string && !escaped => {
                escaped = true;
                out.push(c);
                continue;
            }
            ',' | ':' if !in_string => {
                out.push(c);
                out.push(' ');
            }
            _ => out.push(c),
        }
        escaped = false;
    }
    out
}

impl IntoResponse for MlflowError {
    /// Mirrors `catch_mlflow_exception`'s `Response` construction
    /// (`mlflow/server/handlers.py`): `mimetype="application/json"`, body
    /// `e.serialize_as_json()`, status `e.get_http_status_code()`. Note this
    /// is deliberately NOT the pretty-printed (`indent=2`) JSON used for
    /// normal proto responses (§4 item 3) — error bodies are compact.
    fn into_response(self) -> Response {
        let status = self.http_status();
        let body = self.serialize_as_json();
        (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
    }
}

/// Mirrors `mlflow/server/handlers.py::_not_implemented()`: an empty body,
/// HTTP 404, and Flask's default `text/html; charset=utf-8` content type
/// (Flask's bare `Response()` doesn't set an explicit mimetype, so it falls
/// back to the app's default).
pub fn not_implemented_response() -> Response {
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        String::new(),
    )
        .into_response()
}

/// Mirrors `mlflow/server/auth/__init__.py::make_basic_auth_response()`:
/// plain-text body, HTTP 401, `WWW-Authenticate: Basic realm="mlflow"`.
pub fn unauthenticated_response() -> Response {
    let mut response = (
        StatusCode::UNAUTHORIZED,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        "You are not authenticated. Please see \
         https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow \
         on how to authenticate.",
    )
        .into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Basic realm=\"mlflow\""),
    );
    response
}

/// Mirrors `mlflow/server/auth/__init__.py::make_forbidden_response()`:
/// plain-text body `"Permission denied"`, HTTP 403.
pub fn forbidden_response() -> Response {
    (
        StatusCode::FORBIDDEN,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        "Permission denied",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_as_json_matches_python_shape_with_extras() {
        let err = MlflowError::resource_does_not_exist("Run 'x' not found");
        assert_eq!(
            err.serialize_as_json(),
            r#"{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Run 'x' not found", "sqlstate": "KAM00", "error_class": "RESOURCE_NOT_FOUND"}"#
        );
    }

    #[test]
    fn serialize_as_json_omits_absent_fields() {
        let err = MlflowError::new("Not authenticated", ErrorCode::Unauthenticated);
        assert_eq!(
            err.serialize_as_json(),
            r#"{"error_code": "UNAUTHENTICATED", "message": "Not authenticated"}"#
        );
    }

    #[test]
    fn serialize_as_json_does_not_widen_separators_inside_message() {
        // Regression guard for `widen_json_separators`: commas/colons that
        // are part of the (user-supplied) message text must NOT gain extra
        // spaces, only the structural JSON separators may.
        let err = MlflowError::invalid_parameter_value("bad: value, expected: int");
        assert_eq!(
            err.serialize_as_json(),
            r#"{"error_code": "INVALID_PARAMETER_VALUE", "message": "bad: value, expected: int", "sqlstate": "KAM00", "error_class": "INVALID_PARAMETER_VALUE"}"#
        );
    }

    #[test]
    fn internal_error_is_default_error_code_status() {
        let err = MlflowError::internal_error("boom");
        assert_eq!(err.http_status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
