//! The auth/RBAC HTTP API (plan Phase 9, §3.16).
//!
//! These are the hand-rolled JSON endpoints served by Python's
//! `mlflow.server.auth` app (`mlflow/server/auth/__init__.py`), **not** proto
//! `ROUTE_TABLE` routes — so they are registered separately from
//! `register_proto_routes` and only when the basic-auth app is enabled
//! ([`crate::state::AppState::auth_enabled`]).
//!
//! T9.2 ([`users`]) covers the 8 user-management endpoints. Roles/permissions
//! (T9.3) and the tower auth middleware (T9.4) land in sibling modules.
//!
//! ## Authentication vs authorization (the T9.4 seam)
//!
//! Python splits these two concerns: a *before-request* hook authorizes the
//! request (admin-only gating, self-service permission, `Permission denied`),
//! while the handler bodies call `authenticate_request()` purely to learn *who*
//! the caller is for handler-level self checks (self-service password change,
//! cannot-delete-self). This module implements only the second half — HTTP
//! Basic *authentication* (identity resolution against the [`AuthStore`]) — so
//! the handler-level checks the T9.2 spec calls for can run. The authorization
//! middleware (admin bypass, per-endpoint validators, the 401/403 challenge
//! responses) is T9.4; every place that middleware will gate is marked with an
//! `// AUTH SEAM (T9.4):` comment in [`users`].

use axum::http::request::Parts;
use base64::Engine;
use mlflow_auth::AuthStore;

pub mod roles;
pub mod signup;
pub mod users;

pub use roles::register_role_routes;

/// The HTTP Basic credentials on a request, if present and well-formed.
pub struct BasicCredentials {
    pub username: String,
    pub password: String,
}

/// Extract HTTP Basic credentials from the `Authorization` header, mirroring
/// werkzeug's `request.authorization` for the `Basic` scheme. Returns `None`
/// when the header is absent or not a decodable `Basic <base64(user:pass)>`.
///
/// Python reads `request.authorization.username` / `.password`; werkzeug splits
/// the decoded credentials on the **first** colon (the password may itself
/// contain colons), which this reproduces.
pub fn basic_credentials(parts: &Parts) -> Option<BasicCredentials> {
    let header = parts.headers.get(axum::http::header::AUTHORIZATION)?;
    let value = header.to_str().ok()?;
    let encoded = value
        .strip_prefix("Basic ")
        .or_else(|| value.strip_prefix("basic "))?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let (username, password) = decoded.split_once(':')?;
    Some(BasicCredentials {
        username: username.to_string(),
        password: password.to_string(),
    })
}

/// Resolve the authenticated caller's username, or `None` when the request
/// carries no valid Basic credential. This is the handler-level counterpart of
/// Python's `authenticate_request().username` used for the self checks
/// (`update_user_password`, `delete_user`). It verifies the credential against
/// the store exactly as `authenticate_request_basic_auth` does; a wrong
/// password (or missing/unknown user) yields `None`.
///
/// AUTH SEAM (T9.4): the full before-request authentication (returning a 401
/// `WWW-Authenticate: Basic realm="mlflow"` challenge on failure) and
/// authorization (admin bypass + per-endpoint validators) live in the middleware
/// task. Here we only need the caller's *identity* to run the handler-level self
/// checks; an absent/invalid credential simply means "no self match", which is
/// the correct handler-level behavior once the middleware has already
/// authenticated the request upstream.
pub async fn authenticated_username(store: &AuthStore, parts: &Parts) -> Option<String> {
    let creds = basic_credentials(parts)?;
    if store
        .authenticate_user(&creds.username, &creds.password)
        .await
    {
        Some(creds.username)
    } else {
        None
    }
}
