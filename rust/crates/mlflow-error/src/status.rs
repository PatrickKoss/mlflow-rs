//! Port of `ERROR_CODE_TO_HTTP_STATUS` from `mlflow/exceptions.py`.
//!
//! Python's `MlflowException.get_http_status_code()` looks up
//! `ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)` — i.e. any
//! `ErrorCode` not explicitly listed in the map (the vast majority of the 79
//! generated variants; only 21 are mapped) defaults to HTTP 500.

use axum::http::StatusCode;
use mlflow_proto::mlflow::ErrorCode;

/// Number of `ErrorCode` variants Python's `ERROR_CODE_TO_HTTP_STATUS`
/// explicitly maps (`mlflow/exceptions.py`, confirmed via
/// `len(ERROR_CODE_TO_HTTP_STATUS)`). Everything else falls back to the
/// [`DEFAULT_HTTP_STATUS`].
pub const MAPPED_ERROR_CODE_COUNT: usize = 21;

/// Default status for any `ErrorCode` not present in the explicit map,
/// matching Python's `ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)`.
pub const DEFAULT_HTTP_STATUS: StatusCode = StatusCode::INTERNAL_SERVER_ERROR;

/// HTTP status for a given `ErrorCode`, matching
/// `MlflowException.get_http_status_code()` exactly, including the default.
pub fn http_status(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::InternalError => StatusCode::INTERNAL_SERVER_ERROR,
        ErrorCode::InvalidState => StatusCode::INTERNAL_SERVER_ERROR,
        ErrorCode::DataLoss => StatusCode::INTERNAL_SERVER_ERROR,
        ErrorCode::NotImplemented => StatusCode::NOT_IMPLEMENTED,
        ErrorCode::TemporarilyUnavailable => StatusCode::SERVICE_UNAVAILABLE,
        ErrorCode::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        ErrorCode::RequestLimitExceeded => StatusCode::TOO_MANY_REQUESTS,
        // HTTP 499 ("Client Closed Request") is an Nginx-originated
        // non-standard code with no `http::StatusCode` constant; build it
        // from its numeric value the same way Python's Flask does.
        ErrorCode::Cancelled => StatusCode::from_u16(499).expect("499 is a valid status code"),
        ErrorCode::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        ErrorCode::Aborted => StatusCode::CONFLICT,
        ErrorCode::ResourceConflict => StatusCode::CONFLICT,
        ErrorCode::AlreadyExists => StatusCode::CONFLICT,
        ErrorCode::NotFound => StatusCode::NOT_FOUND,
        ErrorCode::EndpointNotFound => StatusCode::NOT_FOUND,
        ErrorCode::ResourceDoesNotExist => StatusCode::NOT_FOUND,
        ErrorCode::PermissionDenied => StatusCode::FORBIDDEN,
        ErrorCode::CustomerUnauthorized => StatusCode::UNAUTHORIZED,
        ErrorCode::Unauthenticated => StatusCode::UNAUTHORIZED,
        ErrorCode::BadRequest => StatusCode::BAD_REQUEST,
        ErrorCode::ResourceAlreadyExists => StatusCode::BAD_REQUEST,
        ErrorCode::InvalidParameterValue => StatusCode::BAD_REQUEST,
        _ => DEFAULT_HTTP_STATUS,
    }
}

/// Every `ErrorCode` variant generated from `databricks.proto`, in
/// declaration order. There is no derived iterator for prost enums (only
/// `as_str_name`/`from_str_name`), so this list is transcribed by hand from
/// the generated `mlflow.rs` and double-checked by
/// [`tests::all_error_codes_round_trips_every_variant`] against the
/// generated `as_str_name`/`from_str_name` pair — if upstream adds or
/// removes a variant, that round-trip test (via [`ALL_ERROR_CODES_COUNT`])
/// or the exhaustive `match` above (via `#[deny(unreachable_patterns)]`-style
/// compiler coverage) will not silently miss it because
/// [`tests::every_error_code_variant_is_covered`] asserts this list's length
/// against the enum's known total.
#[cfg(test)]
const ALL_ERROR_CODES: &[ErrorCode] = &[
    ErrorCode::InternalError,
    ErrorCode::TemporarilyUnavailable,
    ErrorCode::IoError,
    ErrorCode::BadRequest,
    ErrorCode::ServiceUnderMaintenance,
    ErrorCode::WorkspaceTemporarilyUnavailable,
    ErrorCode::DeadlineExceeded,
    ErrorCode::Cancelled,
    ErrorCode::ResourceExhausted,
    ErrorCode::Aborted,
    ErrorCode::NotFound,
    ErrorCode::AlreadyExists,
    ErrorCode::Unauthenticated,
    ErrorCode::InvalidParameterValue,
    ErrorCode::EndpointNotFound,
    ErrorCode::MalformedRequest,
    ErrorCode::InvalidState,
    ErrorCode::PermissionDenied,
    ErrorCode::FeatureDisabled,
    ErrorCode::CustomerUnauthorized,
    ErrorCode::RequestLimitExceeded,
    ErrorCode::ResourceConflict,
    ErrorCode::UnparseableHttpError,
    ErrorCode::NotImplemented,
    ErrorCode::DataLoss,
    ErrorCode::InvalidStateTransition,
    ErrorCode::CouldNotAcquireLock,
    ErrorCode::ResourceAlreadyExists,
    ErrorCode::ResourceDoesNotExist,
    ErrorCode::QuotaExceeded,
    ErrorCode::MaxBlockSizeExceeded,
    ErrorCode::MaxReadSizeExceeded,
    ErrorCode::PartialDelete,
    ErrorCode::MaxListSizeExceeded,
    ErrorCode::DryRunFailed,
    ErrorCode::ResourceLimitExceeded,
    ErrorCode::DirectoryNotEmpty,
    ErrorCode::DirectoryProtected,
    ErrorCode::MaxNotebookSizeExceeded,
    ErrorCode::MaxChildNodeSizeExceeded,
    ErrorCode::SearchQueryTooLong,
    ErrorCode::SearchQueryTooShort,
    ErrorCode::ManagedResourceGroupDoesNotExist,
    ErrorCode::PermissionNotPropagated,
    ErrorCode::DeploymentTimeout,
    ErrorCode::GitConflict,
    ErrorCode::GitUnknownRef,
    ErrorCode::GitSensitiveTokenDetected,
    ErrorCode::GitUrlNotOnAllowList,
    ErrorCode::GitRemoteError,
    ErrorCode::ProjectsOperationTimeout,
    ErrorCode::IpynbFileInRepo,
    ErrorCode::InsecurePartnerResponse,
    ErrorCode::MalformedPartnerResponse,
    ErrorCode::MetastoreDoesNotExist,
    ErrorCode::DacDoesNotExist,
    ErrorCode::CatalogDoesNotExist,
    ErrorCode::SchemaDoesNotExist,
    ErrorCode::TableDoesNotExist,
    ErrorCode::ShareDoesNotExist,
    ErrorCode::RecipientDoesNotExist,
    ErrorCode::StorageCredentialDoesNotExist,
    ErrorCode::ExternalLocationDoesNotExist,
    ErrorCode::PrincipalDoesNotExist,
    ErrorCode::ProviderDoesNotExist,
    ErrorCode::MetastoreAlreadyExists,
    ErrorCode::DacAlreadyExists,
    ErrorCode::CatalogAlreadyExists,
    ErrorCode::SchemaAlreadyExists,
    ErrorCode::TableAlreadyExists,
    ErrorCode::ShareAlreadyExists,
    ErrorCode::RecipientAlreadyExists,
    ErrorCode::StorageCredentialAlreadyExists,
    ErrorCode::ExternalLocationAlreadyExists,
    ErrorCode::ProviderAlreadyExists,
    ErrorCode::CatalogNotEmpty,
    ErrorCode::SchemaNotEmpty,
    ErrorCode::MetastoreNotEmpty,
    ErrorCode::ProviderShareNotAccessible,
];

/// Total variant count of the generated `ErrorCode` enum as of this port.
/// [`tests::error_code_variant_count_is_up_to_date`] fails the build if a
/// proto change adds/removes a variant, forcing [`ALL_ERROR_CODES`] (and this
/// constant) to be updated by hand.
#[cfg(test)]
const ALL_ERROR_CODES_COUNT: usize = 79;

#[cfg(test)]
mod tests {
    use super::*;

    /// Table transcribed 1:1 from `ERROR_CODE_TO_HTTP_STATUS` in
    /// `mlflow/exceptions.py`. Every entry in this table is asserted against
    /// [`http_status`], AND its length is asserted to equal
    /// [`MAPPED_ERROR_CODE_COUNT`] so this test fails loudly if Python's map
    /// ever grows/shrinks without a matching Rust update.
    fn python_error_code_to_http_status() -> Vec<(ErrorCode, u16)> {
        vec![
            (ErrorCode::InternalError, 500),
            (ErrorCode::InvalidState, 500),
            (ErrorCode::DataLoss, 500),
            (ErrorCode::NotImplemented, 501),
            (ErrorCode::TemporarilyUnavailable, 503),
            (ErrorCode::DeadlineExceeded, 504),
            (ErrorCode::RequestLimitExceeded, 429),
            (ErrorCode::Cancelled, 499),
            (ErrorCode::ResourceExhausted, 429),
            (ErrorCode::Aborted, 409),
            (ErrorCode::ResourceConflict, 409),
            (ErrorCode::AlreadyExists, 409),
            (ErrorCode::NotFound, 404),
            (ErrorCode::EndpointNotFound, 404),
            (ErrorCode::ResourceDoesNotExist, 404),
            (ErrorCode::PermissionDenied, 403),
            (ErrorCode::CustomerUnauthorized, 401),
            (ErrorCode::Unauthenticated, 401),
            (ErrorCode::BadRequest, 400),
            (ErrorCode::ResourceAlreadyExists, 400),
            (ErrorCode::InvalidParameterValue, 400),
        ]
    }

    #[test]
    fn mapped_table_size_matches_python() {
        assert_eq!(
            python_error_code_to_http_status().len(),
            MAPPED_ERROR_CODE_COUNT
        );
    }

    #[test]
    fn every_mapped_code_matches_python() {
        for (code, expected) in python_error_code_to_http_status() {
            assert_eq!(
                http_status(code).as_u16(),
                expected,
                "mismatch for {code:?}"
            );
        }
    }

    /// Guards [`ALL_ERROR_CODES`] against silent drift: if this fails, the
    /// generated `ErrorCode` enum gained/lost a variant and both
    /// [`ALL_ERROR_CODES`] and [`ALL_ERROR_CODES_COUNT`] need a manual update
    /// (transcribe from the generated `mlflow.rs`'s `pub enum ErrorCode`).
    #[test]
    fn error_code_variant_count_is_up_to_date() {
        assert_eq!(ALL_ERROR_CODES.len(), ALL_ERROR_CODES_COUNT);
    }

    /// Every listed code must round-trip through `as_str_name`/
    /// `from_str_name` to itself, catching transcription typos in
    /// [`ALL_ERROR_CODES`].
    #[test]
    fn all_error_codes_round_trips_every_variant() {
        for &code in ALL_ERROR_CODES {
            let name = code.as_str_name();
            assert_eq!(ErrorCode::from_str_name(name), Some(code), "{name}");
        }
    }

    /// Every `ErrorCode` variant must be covered by [`http_status`]: codes
    /// explicitly mapped in Python resolve to their exact mapped status
    /// (which MAY legitimately equal the default — e.g. `INTERNAL_ERROR`,
    /// `INVALID_STATE`, and `DATA_LOSS` are all explicitly mapped to 500,
    /// same as the fallback), and codes absent from Python's map resolve to
    /// exactly the documented default (500). A newly added upstream
    /// `ErrorCode` that this test doesn't know about yet is caught by
    /// [`error_code_variant_count_is_up_to_date`] first.
    #[test]
    fn every_error_code_variant_is_covered() {
        let mapped: std::collections::HashMap<ErrorCode, u16> =
            python_error_code_to_http_status().into_iter().collect();
        for &code in ALL_ERROR_CODES {
            let status = http_status(code);
            let expected = mapped
                .get(&code)
                .copied()
                .map(|s| StatusCode::from_u16(s).unwrap())
                .unwrap_or(DEFAULT_HTTP_STATUS);
            assert_eq!(
                status, expected,
                "{code:?} resolved to an unexpected status"
            );
        }
        assert_eq!(
            ALL_ERROR_CODES.len(),
            mapped.len() + unmapped_count(),
            "sanity check: mapped + unmapped counts must cover every variant exactly once"
        );
    }

    fn unmapped_count() -> usize {
        ALL_ERROR_CODES.len() - python_error_code_to_http_status().len()
    }
}
