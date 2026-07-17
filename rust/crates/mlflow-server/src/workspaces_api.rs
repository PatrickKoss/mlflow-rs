//! Workspace REST endpoints (plan T10.2, §3.17): the 5 `MlflowService`
//! workspace RPCs registered at `/api/3.0/mlflow/workspaces...`:
//!
//! * `GET  /mlflow/workspaces`                  — list (200)
//! * `POST /mlflow/workspaces`                  — create (**201**)
//! * `GET  /mlflow/workspaces/{workspace_name}` — get (200)
//! * `PATCH /mlflow/workspaces/{workspace_name}`— update (200)
//! * `DELETE /mlflow/workspaces/{workspace_name}?mode=RESTRICT|CASCADE|SET_DEFAULT`
//!   — delete (**204**)
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`
//! (`_list_workspaces_handler`..`_delete_workspace_handler`,
//! `handlers.py:1349-1515`): parse the request proto (via [`crate::proto_http`]),
//! run the same name / trace-archival / artifact-root validation, call the
//! [`mlflow_store::WorkspaceStore`], then serialize the response proto.
//!
//! ## Disabled → 503
//!
//! Every endpoint is gated by `_disable_if_workspaces_disabled`
//! (`handlers.py:1222`): when `MLFLOW_ENABLE_WORKSPACES` is off (no
//! [`WorkspaceStore`] wired into [`AppState`]), each returns a plain-text **503**
//! carrying Python's exact body. The `{path}` slot is Python's `request.url_rule`
//! (the matched route); we substitute the concrete request path, which for the
//! registered routes stringifies to the same value.
//!
//! ## Validation lives in the handler
//!
//! Python validates trace-archival config and the default-artifact-root at the
//! *handler* level (with `parameter_name`-carrying messages such as
//! `"Invalid value for 'trace_archival_config.retention'. ..."`) *before* calling
//! the store. The [`WorkspaceStore`] re-validates with its own (non-parameter)
//! messages, but those are never observed for bad input because the handler
//! rejects first. We reproduce the handler-level validators here so the error
//! bodies match byte-for-byte.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::header;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_store::{Workspace, WorkspaceDeletionMode};

use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;

/// `DEFAULT_WORKSPACE_NAME` (`mlflow/utils/workspace_utils.py`). Reserved and
/// undeletable / uncreatable.
const DEFAULT_WORKSPACE_NAME: &str = "default";

/// `MLFLOW_ARTIFACT_LOCATION_MAX_LENGTH` default
/// (`mlflow/environment_variables.py:1119`).
const ARTIFACT_LOCATION_MAX_LENGTH: usize = 2048;

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `_list_workspaces_handler` (`handlers.py:1350`), path: `GET /mlflow/workspaces`.
pub async fn list_workspaces(
    State(state): State<AppState>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, WorkspaceError> {
    let store = workspace_store(&state, &parts)?;
    let _req: pb::ListWorkspaces = parse_request(&parts, &body, "mlflow.ListWorkspaces")?;

    let workspaces = store.list_workspaces().await?;
    let resp = pb::list_workspaces::Response {
        workspaces: workspaces.into_iter().map(to_proto_workspace).collect(),
    };
    Ok(proto_response(&resp, "mlflow.ListWorkspaces.Response")?)
}

/// `_create_workspace_handler` (`handlers.py:1360`), path: `POST /mlflow/workspaces`.
/// Returns **201**.
pub async fn create_workspace(
    State(state): State<AppState>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, WorkspaceError> {
    let store = workspace_store(&state, &parts)?;
    let req: pb::CreateWorkspace = parse_request(&parts, &body, "mlflow.CreateWorkspace")?;

    let name = req.name.unwrap_or_default();
    if name == DEFAULT_WORKSPACE_NAME {
        return Err(MlflowError::invalid_parameter_value(format!(
            "The '{DEFAULT_WORKSPACE_NAME}' workspace is reserved and cannot be created"
        ))
        .into());
    }
    mlflow_store::WorkspaceNameValidator::validate(&name)?;

    // `description if HasField("description") else None` — prost `Option` is the
    // `HasField`; a present empty string stays `Some("")`.
    let description = req.description;
    let default_artifact_root = req.default_artifact_root;

    let (has_location, has_retention, raw_location, raw_retention) =
        split_trace_archival_config(req.trace_archival_config.as_ref());

    let default_artifact_root = validate_workspace_default_artifact_root(default_artifact_root)?;
    let trace_archival_location =
        validate_workspace_trace_archival_location(has_location, raw_location)?;
    let trace_archival_retention =
        validate_workspace_trace_archival_retention(has_retention, raw_retention)?;

    ensure_artifact_root_available(&state, default_artifact_root.as_deref())?;

    let workspace = store
        .create_workspace(Workspace {
            name,
            description,
            default_artifact_root,
            trace_archival_location,
            trace_archival_retention,
        })
        .await?;

    let resp = pb::create_workspace::Response {
        workspace: Some(to_proto_workspace(workspace)),
    };
    let mut response = proto_response(&resp, "mlflow.CreateWorkspace.Response")?;
    *response.status_mut() = axum::http::StatusCode::CREATED;
    Ok(response)
}

/// `_get_workspace_handler` (`handlers.py:1419`), path:
/// `GET /mlflow/workspaces/{workspace_name}`.
pub async fn get_workspace(
    State(state): State<AppState>,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
) -> Result<Response, WorkspaceError> {
    let store = workspace_store(&state, &parts)?;
    let workspace_name = path_param(&path_params, "workspace_name");
    if workspace_name != DEFAULT_WORKSPACE_NAME {
        mlflow_store::WorkspaceNameValidator::validate(&workspace_name)?;
    }

    let workspace = store.get_workspace(&workspace_name).await?;
    let resp = pb::get_workspace::Response {
        workspace: Some(to_proto_workspace(workspace)),
    };
    Ok(proto_response(&resp, "mlflow.GetWorkspace.Response")?)
}

/// `_update_workspace_handler` (`handlers.py:1430`), path:
/// `PATCH /mlflow/workspaces/{workspace_name}`.
pub async fn update_workspace(
    State(state): State<AppState>,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, WorkspaceError> {
    let store = workspace_store(&state, &parts)?;
    let workspace_name = path_param(&path_params, "workspace_name");
    if workspace_name != DEFAULT_WORKSPACE_NAME {
        mlflow_store::WorkspaceNameValidator::validate(&workspace_name)?;
    }

    let req: pb::UpdateWorkspace = parse_request(&parts, &body, "mlflow.UpdateWorkspace")?;

    let has_description = req.description.is_some();
    let has_artifact_root = req.default_artifact_root.is_some();
    let (has_location, has_retention, raw_location, raw_retention) =
        split_trace_archival_config(req.trace_archival_config.as_ref());

    if !has_description && !has_artifact_root && !has_location && !has_retention {
        return Err(MlflowError::invalid_parameter_value(
            "Workspace update must have at least one key",
        )
        .into());
    }

    let description = req.description;
    let default_artifact_root =
        validate_workspace_default_artifact_root(req.default_artifact_root)?;
    let trace_archival_location =
        validate_workspace_trace_archival_location(has_location, raw_location)?;
    let trace_archival_retention =
        validate_workspace_trace_archival_retention(has_retention, raw_retention)?;

    // Clearing the workspace artifact root (empty string) requires a server
    // default to fall back on.
    if default_artifact_root.as_deref() == Some("") {
        ensure_artifact_root_available(&state, default_artifact_root.as_deref())?;
    }

    let workspace = store
        .update_workspace(Workspace {
            name: workspace_name,
            description,
            default_artifact_root,
            trace_archival_location,
            trace_archival_retention,
        })
        .await?;

    let resp = pb::update_workspace::Response {
        workspace: Some(to_proto_workspace(workspace)),
    };
    Ok(proto_response(&resp, "mlflow.UpdateWorkspace.Response")?)
}

/// `_delete_workspace_handler` (`handlers.py:1495`), path:
/// `DELETE /mlflow/workspaces/{workspace_name}?mode=...`. Returns **204**.
pub async fn delete_workspace(
    State(state): State<AppState>,
    Path(path_params): Path<HashMap<String, String>>,
    parts: Parts,
) -> Result<Response, WorkspaceError> {
    let store = workspace_store(&state, &parts)?;
    let workspace_name = path_param(&path_params, "workspace_name");
    if workspace_name == DEFAULT_WORKSPACE_NAME {
        return Err(MlflowError::invalid_parameter_value(format!(
            "The '{DEFAULT_WORKSPACE_NAME}' workspace is reserved and cannot be deleted"
        ))
        .into());
    }
    mlflow_store::WorkspaceNameValidator::validate(&workspace_name)?;

    let mode = parse_mode(query_param(&parts, "mode"))?;

    store.delete_workspace(&workspace_name, mode).await?;

    Ok(Response::builder()
        .status(axum::http::StatusCode::NO_CONTENT)
        .body(axum::body::Body::empty())
        .expect("valid 204 response"))
}

// ---------------------------------------------------------------------------
// 503-when-disabled + store handle
// ---------------------------------------------------------------------------

/// `_disable_if_workspaces_disabled` (`handlers.py:1222`): resolve the wired
/// [`WorkspaceStore`], or short-circuit with the plain-text **503** Python emits
/// when `MLFLOW_ENABLE_WORKSPACES` is off. The `{path}` slot is Python's
/// `request.url_rule`; we use the concrete request path (same string for the
/// registered routes).
fn workspace_store<'a>(
    state: &'a AppState,
    parts: &Parts,
) -> Result<&'a mlflow_store::WorkspaceStore, WorkspaceError> {
    state
        .workspace_store()
        .ok_or_else(|| WorkspaceError::Disabled(parts.uri.path().to_string()))
}

/// The two error shapes a workspace handler produces: an ordinary
/// [`MlflowError`] (JSON body, mapped status) or the disabled-workspaces
/// short-circuit (plain-text 503). Keeping them distinct lets `?` compose while
/// the 503 renders Python's exact non-JSON body.
pub enum WorkspaceError {
    /// A normal MLflow error — rendered via [`MlflowError`]'s `IntoResponse`.
    Mlflow(MlflowError),
    /// Workspaces are disabled; carries the request path for the 503 body.
    Disabled(String),
}

impl From<MlflowError> for WorkspaceError {
    fn from(err: MlflowError) -> Self {
        WorkspaceError::Mlflow(err)
    }
}

impl axum::response::IntoResponse for WorkspaceError {
    fn into_response(self) -> Response {
        match self {
            WorkspaceError::Mlflow(err) => err.into_response(),
            WorkspaceError::Disabled(path) => {
                let body = format!(
                    "Endpoint: {path} disabled because the server is running without workspaces \
                     support. To enable workspace, run `mlflow server` with `--enable-workspaces`"
                );
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                    body,
                )
                    .into_response()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trace-archival / artifact-root validation (handler-level, parameter_name form)
// ---------------------------------------------------------------------------

/// Split the optional `trace_archival_config` into presence flags + raw values,
/// mirroring Python's `"location" in trace_archival_config_json` /
/// `"retention" in trace_archival_config_json` checks. A present empty string is
/// `Some("")` (key present, clears the value); an absent key is `None`.
fn split_trace_archival_config(
    config: Option<&pb::TraceArchivalConfig>,
) -> (bool, bool, Option<String>, Option<String>) {
    match config {
        None => (false, false, None, None),
        Some(c) => (
            c.location.is_some(),
            c.retention.is_some(),
            c.location.clone(),
            c.retention.clone(),
        ),
    }
}

/// `_validate_workspace_default_artifact_root` (`handlers.py:1268`):
/// `None` → `None`; trimmed-empty → `Some("")` (clear sentinel); otherwise the
/// validated storage-location URI.
fn validate_workspace_default_artifact_root(
    value: Option<String>,
) -> Result<Option<String>, MlflowError> {
    validate_optional_workspace_storage_location(value, "default_artifact_root")
}

/// `_validate_workspace_trace_archival_location` (`handlers.py:1272`):
/// runs `_validate_optional_workspace_storage_location` then, for a non-empty
/// value, the scheme/proxy check of `_validate_trace_archival_location`.
fn validate_workspace_trace_archival_location(
    has_location: bool,
    value: Option<String>,
) -> Result<Option<String>, MlflowError> {
    // The handler only forwards the field when the key was present.
    let value = if has_location { value } else { None };
    let validated =
        validate_optional_workspace_storage_location(value, "trace_archival_config.location")?;
    match validated.as_deref() {
        None | Some("") => Ok(validated),
        Some(loc) => Ok(Some(validate_trace_archival_location(loc)?)),
    }
}

/// `_validate_workspace_trace_archival_retention` (`handlers.py:1283`):
/// `None` → `None`; trimmed-empty → `Some("")`; otherwise the validated
/// retention string.
fn validate_workspace_trace_archival_retention(
    has_retention: bool,
    value: Option<String>,
) -> Result<Option<String>, MlflowError> {
    let value = if has_retention { value } else { None };
    let Some(value) = value else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Some(String::new()));
    }
    Ok(Some(validate_trace_archival_retention(trimmed)?))
}

/// `_validate_optional_workspace_storage_location` (`handlers.py:1257`):
/// `None` → `None`; trimmed-empty → `Some("")`; else `_validate_storage_location_uri`.
fn validate_optional_workspace_storage_location(
    value: Option<String>,
    field_name: &str,
) -> Result<Option<String>, MlflowError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Some(String::new()));
    }
    Ok(Some(validate_storage_location_uri(trimmed, field_name)?))
}

/// `_validate_storage_location_uri` (`handlers.py:1243`): reject fragments/params,
/// path-traversal query strings, `runs:` URIs, and over-length locations.
fn validate_storage_location_uri(value: &str, field_name: &str) -> Result<String, MlflowError> {
    let parsed = ParsedUri::parse(value);
    if !parsed.fragment.is_empty() || !parsed.params.is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "'{field_name}' URL can't include fragments or params."
        )));
    }
    // `validate_query_string`: block `..` traversal.
    if parsed.query.contains("..") {
        return Err(MlflowError::invalid_parameter_value("Invalid query string"));
    }
    // `_validate_experiment_artifact_location`: no `runs:` URIs.
    if value.starts_with("runs:") {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Artifact location cannot be a runs:/ URI. Given: '{value}'"
        )));
    }
    // `_validate_experiment_artifact_location_length`.
    if value.chars().count() > ARTIFACT_LOCATION_MAX_LENGTH {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid artifact path length. The length of the artifact path cannot be \
             greater than {ARTIFACT_LOCATION_MAX_LENGTH} characters. To configure this \
             limit, please set the MLFLOW_ARTIFACT_LOCATION_MAX_LENGTH environment variable."
        )));
    }
    Ok(value.to_string())
}

/// `_validate_trace_archival_location(value, parameter_name="trace_archival_config.location")`
/// (`mlflow/utils/validation.py:177`): require a URI scheme, reject the
/// proxy-only `mlflow-artifacts:` scheme.
fn validate_trace_archival_location(value: &str) -> Result<String, MlflowError> {
    let parsed = ParsedUri::parse(value);
    match parsed.scheme.as_deref() {
        None | Some("") => Err(MlflowError::invalid_parameter_value(
            "Invalid value for 'trace_archival_config.location'. Expected a URI string.",
        )),
        Some("mlflow-artifacts") => Err(MlflowError::invalid_parameter_value(
            "Invalid value for 'trace_archival_config.location'. Trace archival location cannot \
             use the proxy-only `mlflow-artifacts:` scheme.",
        )),
        Some(_) => Ok(value.to_string()),
    }
}

/// `_validate_trace_archival_retention_string(value,
/// parameter_name="trace_archival_config.retention")`
/// (`mlflow/utils/validation.py:137`): max length 32, then regex
/// `^[1-9][0-9]*[mhd]$`.
fn validate_trace_archival_retention(value: &str) -> Result<String, MlflowError> {
    let trimmed = value.trim();
    if trimmed.chars().count() > 32 {
        return Err(MlflowError::invalid_parameter_value(
            "Invalid value for 'trace_archival_config.retention'. Maximum length is 32 characters.",
        ));
    }
    if !retention_regex_matches(trimmed) {
        return Err(MlflowError::invalid_parameter_value(
            "Invalid value for 'trace_archival_config.retention'. Expected a duration in the form \
             `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h').",
        ));
    }
    Ok(trimmed.to_string())
}

/// `re.compile(r"^[1-9][0-9]*[mhd]$").fullmatch(value)`.
fn retention_regex_matches(value: &str) -> bool {
    match value.as_bytes() {
        [first, rest @ .., unit] => {
            (b'1'..=b'9').contains(first)
                && rest.iter().all(u8::is_ascii_digit)
                && matches!(unit, b'm' | b'h' | b'd')
        }
        _ => false,
    }
}

/// `_ensure_artifact_root_available` (`handlers.py:1323`): a non-empty workspace
/// artifact root is always valid; otherwise the server must have a default
/// artifact root configured.
fn ensure_artifact_root_available(
    state: &AppState,
    workspace_artifact_root: Option<&str>,
) -> Result<(), MlflowError> {
    if workspace_artifact_root.is_some_and(|r| !r.is_empty()) {
        return Ok(());
    }
    let server_artifact_root = state.tracking_store().artifact_root_uri();
    if server_artifact_root.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Cannot create or update workspace without an artifact root. Either specify \
             'default_artifact_root' for this workspace or start the server with \
             '--default-artifact-root'.",
        ));
    }
    Ok(())
}

/// `WorkspaceDeletionMode(mode_str)` with Python's fallback + error
/// (`handlers.py:1502`). Default (no `mode` query param) is `RESTRICT`.
fn parse_mode(mode: Option<String>) -> Result<WorkspaceDeletionMode, MlflowError> {
    let mode_str = mode.unwrap_or_else(|| WorkspaceDeletionMode::Restrict.value().to_string());
    match mode_str.as_str() {
        "SET_DEFAULT" => Ok(WorkspaceDeletionMode::SetDefault),
        "CASCADE" => Ok(WorkspaceDeletionMode::Cascade),
        "RESTRICT" => Ok(WorkspaceDeletionMode::Restrict),
        other => Err(MlflowError::invalid_parameter_value(format!(
            "Invalid deletion mode '{other}'. Must be one of: SET_DEFAULT, CASCADE, RESTRICT"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Conversions + small helpers
// ---------------------------------------------------------------------------

/// `Workspace.to_proto` (`mlflow/entities/workspace.py:76`): fields are set only
/// when `is not None`; the `trace_archival_config` submessage is present only
/// when at least one of its fields is set.
fn to_proto_workspace(workspace: Workspace) -> pb::Workspace {
    let trace_archival_config = if workspace.trace_archival_location.is_some()
        || workspace.trace_archival_retention.is_some()
    {
        Some(pb::TraceArchivalConfig {
            location: workspace.trace_archival_location,
            retention: workspace.trace_archival_retention,
        })
    } else {
        None
    };
    pb::Workspace {
        name: Some(workspace.name),
        description: workspace.description,
        default_artifact_root: workspace.default_artifact_root,
        trace_archival_config,
    }
}

fn path_param(path_params: &HashMap<String, String>, name: &str) -> String {
    path_params.get(name).cloned().unwrap_or_default()
}

fn query_param(parts: &Parts, key: &str) -> Option<String> {
    let query = parts.uri.query()?;
    crate::proto_http::parse_query_pairs(query)
        .into_iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
}

/// A minimal `urllib.parse.urlparse` for the fields the workspace validators
/// read: `scheme`, `params`, `query`, `fragment`. Enough to reproduce the
/// `_validate_storage_location_uri` / `_validate_trace_archival_location`
/// checks the tested inputs exercise (fragment `#...`, `;params`, `?query`,
/// and the URI scheme).
struct ParsedUri {
    scheme: Option<String>,
    params: String,
    query: String,
    fragment: String,
}

impl ParsedUri {
    fn parse(uri: &str) -> Self {
        // Split off the fragment first (`#`), then the query (`?`), matching
        // urlparse's precedence.
        let (before_fragment, fragment) = match uri.split_once('#') {
            Some((a, b)) => (a, b.to_string()),
            None => (uri, String::new()),
        };
        let (before_query, query) = match before_fragment.split_once('?') {
            Some((a, b)) => (a, b.to_string()),
            None => (before_fragment, String::new()),
        };
        // `;params` apply to the last path segment; urlparse extracts them from
        // the path portion after the scheme/authority.
        let params = match before_query.rsplit_once('/') {
            Some((_, last)) => last.split_once(';').map(|(_, p)| p.to_string()),
            None => before_query.split_once(';').map(|(_, p)| p.to_string()),
        }
        .unwrap_or_default();

        // Scheme: `[a-zA-Z][a-zA-Z0-9+.-]*` before the first `:` (urlparse only
        // treats it as a scheme when it starts with a letter).
        let scheme = before_query.split_once(':').and_then(|(s, _)| {
            let bytes = s.as_bytes();
            let valid = !s.is_empty()
                && bytes[0].is_ascii_alphabetic()
                && bytes
                    .iter()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'));
            valid.then(|| s.to_ascii_lowercase())
        });

        Self {
            scheme,
            params,
            query,
            fragment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_regex_matrix() {
        assert!(retention_regex_matches("30d"));
        assert!(retention_regex_matches("12h"));
        assert!(retention_regex_matches("1m"));
        assert!(!retention_regex_matches("0d"));
        assert!(!retention_regex_matches("90days"));
        assert!(!retention_regex_matches("5w"));
        assert!(!retention_regex_matches(""));
    }

    #[test]
    fn retention_too_long_message() {
        let long = format!("{}d", "1".repeat(32));
        let err = validate_trace_archival_retention(&long).unwrap_err();
        assert_eq!(
            err.message,
            "Invalid value for 'trace_archival_config.retention'. Maximum length is 32 characters."
        );
    }

    #[test]
    fn retention_bad_format_message() {
        let err = validate_trace_archival_retention("90days").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid value for 'trace_archival_config.retention'. Expected a duration in the form \
             `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h')."
        );
    }

    #[test]
    fn location_proxy_scheme_rejected() {
        let err =
            validate_trace_archival_location("mlflow-artifacts:/archive/team-proxy").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid value for 'trace_archival_config.location'. Trace archival location cannot \
             use the proxy-only `mlflow-artifacts:` scheme."
        );
    }

    #[test]
    fn location_non_uri_rejected() {
        let err = validate_trace_archival_location("archive/team-local-path").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid value for 'trace_archival_config.location'. Expected a URI string."
        );
    }

    #[test]
    fn location_with_fragment_rejected_by_storage_validator() {
        let err = validate_storage_location_uri(
            "s3://archive/team#fragment",
            "trace_archival_config.location",
        )
        .unwrap_err();
        assert!(
            err.message.contains("trace_archival_config.location"),
            "{}",
            err.message
        );
        assert!(
            err.message.contains("fragments or params"),
            "{}",
            err.message
        );
    }

    #[test]
    fn location_valid_uri_passes() {
        assert_eq!(
            validate_trace_archival_location("s3://archive/team-b").unwrap(),
            "s3://archive/team-b"
        );
    }

    #[test]
    fn parse_mode_matrix() {
        assert_eq!(parse_mode(None).unwrap(), WorkspaceDeletionMode::Restrict);
        assert_eq!(
            parse_mode(Some("CASCADE".into())).unwrap(),
            WorkspaceDeletionMode::Cascade
        );
        assert_eq!(
            parse_mode(Some("SET_DEFAULT".into())).unwrap(),
            WorkspaceDeletionMode::SetDefault
        );
        let err = parse_mode(Some("BOGUS".into())).unwrap_err();
        assert_eq!(
            err.message,
            "Invalid deletion mode 'BOGUS'. Must be one of: SET_DEFAULT, CASCADE, RESTRICT"
        );
    }

    #[test]
    fn parsed_uri_fields() {
        let p = ParsedUri::parse("s3://bucket/path");
        assert_eq!(p.scheme.as_deref(), Some("s3"));
        assert!(p.fragment.is_empty() && p.query.is_empty() && p.params.is_empty());

        let p = ParsedUri::parse("s3://archive/team#fragment");
        assert_eq!(p.fragment, "fragment");

        let p = ParsedUri::parse("archive/team-local-path");
        assert_eq!(p.scheme, None);
    }
}
