//! Workspace resolution for the request (plan §3.17, §4 item 12, T10.3).
//!
//! The wire contract is the `X-MLFLOW-WORKSPACE` request header. Resolution has
//! two modes, mirroring `mlflow/server/workspace_helpers.py`:
//!
//! * **Workspaces disabled** (no [`AppState::workspace_store`]): the header is
//!   *ignored* (never rejected) — every request resolves to
//!   [`DEFAULT_WORKSPACE_NAME`], preserving the single-tenant default. This is
//!   Python's `resolve_workspace_for_request_if_enabled` returning `None` when
//!   `MLFLOW_ENABLE_WORKSPACES` is off (`workspace_helpers.py:108-114`), which
//!   downstream code treats as `default` via `resolve_entity_workspace_name`.
//!   (Python raises `FEATURE_DISABLED` only when a *non-empty* header reaches
//!   the Flask `before_request` hook on a non-server-info route; the disabled
//!   Rust server has no workspace store to validate against and single-tenant
//!   scoping is always `default`, so it matches the observable "ignore"
//!   behavior the server-info test asserts.)
//!
//! * **Workspaces enabled** ([`AppState::workspace_store`] is `Some`): the
//!   header is normalized (`_normalize_workspace`: trim; empty → absent); when
//!   present it is validated (non-`default` names go through
//!   [`WorkspaceNameValidator`]) and looked up in the store — a missing
//!   workspace yields the store's `RESOURCE_DOES_NOT_EXIST` (404). When absent,
//!   the default workspace is resolved from the store
//!   (`get_default_workspace`). This is `resolve_workspace_from_header`
//!   (`workspace_helpers.py:30-45`).
//!
//! The [`workspace_middleware`] tower layer performs this once per request and
//! stamps the resolved name into request extensions as [`ResolvedWorkspace`];
//! the [`Workspace`] extractor (used by every handler) reads it back. Server-info
//! is skipped entirely (`workspace_helpers.py:105`), so a bogus header never
//! breaks that route. The layer sits *inside* the security layer but *outside*
//! the auth layer, matching Python installing `workspace_before_request_handler`
//! on the base app after `security.init_security_middleware` but before the auth
//! app's own `_before_request` (`mlflow/server/__init__.py:82-84`).
//!
//! ## T10.4 seam
//!
//! The resolved workspace is stamped into extensions *outside* the auth layer,
//! so the auth middleware reads the real resolved name (not a re-derived
//! `default`) via [`ResolvedWorkspace`]. Pre-T10.4 the auth validators still
//! resolve every grant in `default`; T10.4 consumes the stamped workspace for
//! grant partitioning without further plumbing here.

use axum::body::Body;
use axum::extract::{FromRequestParts, State};
use axum::http::request::Parts;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;

use mlflow_store::WorkspaceNameValidator;

use crate::state::AppState;

/// The workspace header name (`WORKSPACE_HEADER_NAME`).
pub const WORKSPACE_HEADER_NAME: &str = "X-MLFLOW-WORKSPACE";

/// The default workspace (`DEFAULT_WORKSPACE_NAME`).
pub const DEFAULT_WORKSPACE_NAME: &str = "default";

/// The resolved workspace name for the current request, stamped into request
/// extensions by [`workspace_middleware`]. The [`Workspace`] extractor reads it,
/// and the auth middleware reads it for the T10.4 grant-partitioning seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedWorkspace(pub String);

/// The resolved workspace name for the current request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace(pub String);

impl Workspace {
    /// The workspace name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// Normalize a raw header value, mirroring `_normalize_workspace`: trim ASCII
/// whitespace, and treat an empty/absent value as absent (`None`).
fn normalize_workspace(header_value: Option<&str>) -> Option<&str> {
    header_value.map(str::trim).filter(|s| !s.is_empty())
}

/// Resolve the workspace from a raw header value for the *disabled* (single
/// tenant) path: trim, empty/absent → `default`. Never consults a store.
pub fn resolve_workspace(header_value: Option<&str>) -> Workspace {
    Workspace(
        normalize_workspace(header_value)
            .unwrap_or(DEFAULT_WORKSPACE_NAME)
            .to_string(),
    )
}

/// Whether the request path is the server-info route, which is exempt from
/// workspace resolution (`workspace_helpers.py:105`:
/// `path.rstrip("/").endswith("/mlflow/server-info")`). The static prefix, if
/// any, precedes the matched path, so a suffix test mirrors Python.
fn is_server_info_path(path: &str) -> bool {
    path.trim_end_matches('/').ends_with("/mlflow/server-info")
}

/// The workspace-resolution tower middleware (`axum::middleware::from_fn_with_state`).
///
/// Resolves the active workspace for the request and stamps it into extensions
/// as [`ResolvedWorkspace`]. Short-circuits with an `MlflowError` response
/// (404 for a missing workspace, 400 for an invalid name) when enabled
/// resolution fails, matching Python's `_workspace_error_response`.
pub async fn workspace_middleware(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    // Server-info is always reachable, even with a bogus header — skip
    // resolution entirely (do not stamp; the handler on this route takes no
    // `Workspace`).
    if is_server_info_path(req.uri().path()) {
        return next.run(req).await;
    }

    let header = req
        .headers()
        .get(WORKSPACE_HEADER_NAME)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let resolved = match state.workspace_store() {
        // Enabled: validate + look up against the store.
        Some(store) => {
            let result = match normalize_workspace(header.as_deref()) {
                Some(name) => {
                    if name != DEFAULT_WORKSPACE_NAME {
                        if let Err(e) = WorkspaceNameValidator::validate(name) {
                            return e.into_response();
                        }
                    }
                    store.get_workspace(name).await
                }
                None => store.get_default_workspace().await,
            };
            match result {
                Ok(ws) => ws.name,
                Err(e) => return e.into_response(),
            }
        }
        // Disabled: ignore the header; always `default`.
        None => DEFAULT_WORKSPACE_NAME.to_string(),
    };

    req.extensions_mut().insert(ResolvedWorkspace(resolved));
    next.run(req).await
}

impl<S> FromRequestParts<S> for Workspace
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Prefer the middleware-resolved workspace (enabled mode validated it
        // against the store; disabled mode set `default`). Fall back to
        // header-based `default` resolution when the middleware did not run
        // (e.g. the ops-only app without the workspace layer, or a test app that
        // omits it), so handlers keep working single-tenant.
        if let Some(ResolvedWorkspace(name)) = parts.extensions.get::<ResolvedWorkspace>() {
            return Ok(Workspace(name.clone()));
        }
        let header = parts
            .headers
            .get(WORKSPACE_HEADER_NAME)
            .and_then(|v| v.to_str().ok());
        Ok(resolve_workspace(header))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_header_falls_back_to_default() {
        assert_eq!(resolve_workspace(None), Workspace("default".to_string()));
    }

    #[test]
    fn empty_header_falls_back_to_default() {
        assert_eq!(
            resolve_workspace(Some("")),
            Workspace("default".to_string())
        );
        assert_eq!(
            resolve_workspace(Some("   ")),
            Workspace("default".to_string())
        );
    }

    #[test]
    fn header_value_is_trimmed() {
        assert_eq!(
            resolve_workspace(Some("  team-a  ")),
            Workspace("team-a".to_string())
        );
    }

    #[test]
    fn header_value_passes_through() {
        assert_eq!(
            resolve_workspace(Some("team-a")),
            Workspace("team-a".to_string())
        );
    }

    #[test]
    fn server_info_path_detection() {
        assert!(is_server_info_path("/api/3.0/mlflow/server-info"));
        assert!(is_server_info_path("/ajax-api/3.0/mlflow/server-info/"));
        assert!(is_server_info_path("/prefix/api/3.0/mlflow/server-info"));
        assert!(!is_server_info_path("/api/3.0/mlflow/experiments/create"));
        assert!(!is_server_info_path("/api/3.0/mlflow/server-information"));
    }
}
