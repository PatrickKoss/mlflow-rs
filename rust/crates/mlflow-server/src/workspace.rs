//! Workspace resolution for the request (plan §3.17, §4 item 12).
//!
//! The wire contract is the `X-MLFLOW-WORKSPACE` request header. This module
//! provides an axum extractor that mirrors Python's request-workspace
//! resolution for the single-tenant / workspaces-disabled default:
//! `_normalize_workspace(header)` (trim whitespace; empty → absent) falling back
//! to `DEFAULT_WORKSPACE_NAME` (`mlflow/utils/workspace_utils.py:7,12-24,54`).
//!
//! Full workspace enablement (validating the name against the `workspaces`
//! table, pinning to `default` when disabled, the 503 on disabled) lands with
//! the workspaces phase; here we only extract the header value so every store
//! call is workspace-scoped from day one.

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use std::convert::Infallible;

/// The workspace header name (`WORKSPACE_HEADER_NAME`).
pub const WORKSPACE_HEADER_NAME: &str = "X-MLFLOW-WORKSPACE";

/// The default workspace (`DEFAULT_WORKSPACE_NAME`).
pub const DEFAULT_WORKSPACE_NAME: &str = "default";

/// The resolved workspace name for the current request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace(pub String);

impl Workspace {
    /// The workspace name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// Resolve the workspace from a raw header value, mirroring
/// `_normalize_workspace` + the `DEFAULT_WORKSPACE_NAME` fallback: trim ASCII
/// whitespace, and treat an empty/absent value as `default`.
pub fn resolve_workspace(header_value: Option<&str>) -> Workspace {
    let normalized = header_value.map(str::trim).filter(|s| !s.is_empty());
    Workspace(normalized.unwrap_or(DEFAULT_WORKSPACE_NAME).to_string())
}

impl<S> FromRequestParts<S> for Workspace
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
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
}
