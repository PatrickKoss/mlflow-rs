//! Crate-local error type for the search DSL parsers.
//!
//! Every parse error raised by `mlflow/utils/search_utils.py` uses
//! `error_code=INVALID_PARAMETER_VALUE`, so [`SearchError`] carries the
//! `error_code`-equivalent as a plain enum plus the **verbatim** Python
//! message. Future callers in `mlflow-store` / `mlflow-registry` can convert
//! [`SearchError`] into `mlflow-error`'s `MlflowError` without this crate
//! having to depend on axum (which `mlflow-error` pulls in). Keeping the
//! parser dependency-light is deliberate — see the crate docs.

use std::fmt;

/// The `error_code`-equivalent for a [`SearchError`].
///
/// Only `InvalidParameterValue` is ever produced by the current parsers, but
/// the enum leaves room for the (unreachable-in-practice) internal-error
/// branch Python has in `SearchUtils._get_value`'s final `else`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Mirrors `INVALID_PARAMETER_VALUE`.
    InvalidParameterValue,
    /// Mirrors the bare `MlflowException(msg)` default (`INTERNAL_ERROR`).
    InternalError,
    /// Not a real MLflow error code: marks an input where the Python parser
    /// itself raises an *uncaught* `ValueError` (a bug), rather than an
    /// `MlflowException`. Reproduced so the parity corpus matches; the
    /// corpus generator records these as `PYTHON_ValueError`. See the crate
    /// report's "known gaps" for the affected inputs.
    PythonValueError,
}

impl ErrorCode {
    /// The protobuf enum *name*, matching `ErrorCode.Name(...)` in Python and
    /// the string MLflow stores on `MlflowException.error_code`.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidParameterValue => "INVALID_PARAMETER_VALUE",
            ErrorCode::InternalError => "INTERNAL_ERROR",
            ErrorCode::PythonValueError => "PYTHON_ValueError",
        }
    }
}

/// A parse error carrying the `error_code`-equivalent and a message that
/// matches the corresponding `MlflowException` message byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchError {
    pub error_code: ErrorCode,
    pub message: String,
}

impl SearchError {
    /// Mirrors `MlflowException(message, error_code=INVALID_PARAMETER_VALUE)`
    /// / `MlflowException.invalid_parameter_value(message)`.
    pub fn invalid_parameter_value(message: impl Into<String>) -> Self {
        Self {
            error_code: ErrorCode::InvalidParameterValue,
            message: message.into(),
        }
    }

    /// Mirrors the bare `MlflowException(message)` (defaults to
    /// `INTERNAL_ERROR`). Only used for Python's "Invalid identifier type"
    /// dead-end branches that never fire in the exercised grammar.
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            error_code: ErrorCode::InternalError,
            message: message.into(),
        }
    }

    /// Reproduces a Python-side uncaught `ValueError` (see [`ErrorCode::PythonValueError`]).
    pub fn python_value_error(message: impl Into<String>) -> Self {
        Self {
            error_code: ErrorCode::PythonValueError,
            message: message.into(),
        }
    }
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.error_code.as_str(), self.message)
    }
}

impl std::error::Error for SearchError {}

/// Convenience alias for parser results.
pub type Result<T> = std::result::Result<T, SearchError>;
