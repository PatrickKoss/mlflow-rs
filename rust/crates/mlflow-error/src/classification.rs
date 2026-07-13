//! Port of `mlflow/error_classification.py`.
//!
//! Maps `ErrorCode` enum names to `sqlstate` and `error_class` strings for
//! structured error classification, mirroring the derivation chain in
//! `MlflowException.__init__`:
//!
//! 1. `error_class`: derived from `error_code` via
//!    [`error_class_from_client_error_code`] (raise sites never pass an
//!    explicit `error_class` in the server, so Rust only needs the
//!    auto-derive path).
//! 2. `sqlstate`: derived from `error_class` via
//!    [`sqlstate_from_error_class`] if that mapping exists, otherwise from
//!    `error_code` via [`sqlstate_from_client_error_code`].
//!
//! Only the **client** tables (`_CLIENT_ERROR_CODE_TO_SQLSTATE` /
//! `_CLIENT_ERROR_CODE_TO_ERROR_CLASS` in the Python source) are ported here:
//! `MlflowException.__init__` always uses the `from_client_error_code` path
//! regardless of whether the exception originates on the server or the
//! client. The `_CP_*` ("control plane") tables are only consulted by
//! `RestException.__init__`, which runs in the Python *client* when parsing a
//! server response — out of scope for a server-side crate.

/// Auto-derive `error_class` from an `ErrorCode` name, matching
/// `ErrorClass.from_client_error_code` / `_CLIENT_ERROR_CODE_TO_ERROR_CLASS`.
/// Returns `None` for codes without an explicit client-side classification
/// (Python then leaves `error_class` unset, so the JSON body omits it).
pub(crate) fn error_class_from_client_error_code(error_code_name: &str) -> Option<&'static str> {
    match error_code_name {
        "BAD_REQUEST" => Some("INVALID_PARAMETER_VALUE"),
        "CUSTOMER_UNAUTHORIZED" => Some("PERMISSION_DENIED"),
        "ENDPOINT_NOT_FOUND" => Some("RESOURCE_NOT_FOUND"),
        "FEATURE_DISABLED" => Some("FEATURE_DISABLED"),
        "INTERNAL_ERROR" => Some("CLIENT_INTERNAL_ERROR"),
        "INVALID_PARAMETER_VALUE" => Some("INVALID_PARAMETER_VALUE"),
        "INVALID_STATE" => Some("CLIENT_INTERNAL_ERROR"),
        "NOT_FOUND" => Some("RESOURCE_NOT_FOUND"),
        "PERMISSION_DENIED" => Some("PERMISSION_DENIED"),
        "RESOURCE_ALREADY_EXISTS" => Some("RESOURCE_ALREADY_EXISTS"),
        "RESOURCE_DOES_NOT_EXIST" => Some("RESOURCE_NOT_FOUND"),
        "TEMPORARILY_UNAVAILABLE" => Some("CLIENT_INTERNAL_ERROR"),
        _ => None,
    }
}

/// Auto-derive `sqlstate` straight from an `ErrorCode` name, matching
/// `SqlState.from_client_error_code` / `_CLIENT_ERROR_CODE_TO_SQLSTATE`. Used
/// as the fallback when [`sqlstate_from_error_class`] has no override.
pub(crate) fn sqlstate_from_client_error_code(error_code_name: &str) -> Option<&'static str> {
    match error_code_name {
        "BAD_REQUEST" => Some("KAM00"),
        "CUSTOMER_UNAUTHORIZED" => Some("KAM00"),
        "ENDPOINT_NOT_FOUND" => Some("KAM00"),
        "FEATURE_DISABLED" => Some("KAM00"),
        "INTERNAL_ERROR" => Some("XXM00"),
        "INVALID_PARAMETER_VALUE" => Some("KAM00"),
        "INVALID_STATE" => Some("XXM00"),
        "NOT_FOUND" => Some("KAM00"),
        "PERMISSION_DENIED" => Some("KAM00"),
        "RESOURCE_ALREADY_EXISTS" => Some("KAM00"),
        "RESOURCE_DOES_NOT_EXIST" => Some("KAM00"),
        "TEMPORARILY_UNAVAILABLE" => Some("XXM00"),
        _ => None,
    }
}

/// `sqlstate` override keyed by `error_class`, matching
/// `_ERROR_CLASS_TO_SQLSTATE`. Only consulted when an explicit `error_class`
/// was passed at the raise site (finer-grained than the `error_code` alone);
/// [`MlflowError`](crate::MlflowError)'s constructors don't currently expose
/// that override, but the table is ported for completeness / future use.
pub(crate) fn sqlstate_from_error_class(error_class: &str) -> Option<&'static str> {
    match error_class {
        "ATTRIBUTE_NOT_FOUND" => Some("KAM04"),
        "MODEL_SERIALIZATION_FAILED" => Some("KAM03"),
        "PREDICTION_FUNCTION_FAILED" => Some("KAM02"),
        "SCHEMA_ENFORCEMENT_FAILED" => Some("KAM01"),
        _ => None,
    }
}

/// Full derivation chain from `MlflowException.__init__` for the
/// no-explicit-override case (the only path server raise sites use):
/// returns `(error_class, sqlstate)`, either of which may be absent — Python
/// omits absent fields from the serialized JSON body entirely.
pub(crate) fn derive(error_code_name: &str) -> (Option<&'static str>, Option<&'static str>) {
    let error_class = error_class_from_client_error_code(error_code_name);
    let sqlstate = error_class
        .and_then(sqlstate_from_error_class)
        .or_else(|| sqlstate_from_client_error_code(error_code_name));
    (error_class, sqlstate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_matches_python_resource_does_not_exist() {
        let (error_class, sqlstate) = derive("RESOURCE_DOES_NOT_EXIST");
        assert_eq!(error_class, Some("RESOURCE_NOT_FOUND"));
        assert_eq!(sqlstate, Some("KAM00"));
    }

    #[test]
    fn derive_matches_python_unmapped_code() {
        // UNAUTHENTICATED has no entry in the client tables in Python: both
        // fields are omitted from the wire body.
        let (error_class, sqlstate) = derive("UNAUTHENTICATED");
        assert_eq!(error_class, None);
        assert_eq!(sqlstate, None);
    }

    #[test]
    fn derive_matches_python_internal_error() {
        let (error_class, sqlstate) = derive("INTERNAL_ERROR");
        assert_eq!(error_class, Some("CLIENT_INTERNAL_ERROR"));
        assert_eq!(sqlstate, Some("XXM00"));
    }
}
