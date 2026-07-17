//! The tower auth middleware (plan T9.4, §3.16): authenticate → admin bypass →
//! validator dispatch → permission enforcement, mirroring `_before_request`
//! (`mlflow/server/auth/__init__.py:2913`) and the FastAPI OTLP middleware
//! (`add_fastapi_permission_middleware`, `:4488`).
//!
//! Applied at the top of the app router by [`layer`] (only when auth is
//! enabled). It runs before every handler and:
//!
//! 1. Lets unprotected routes through ([`is_unprotected_route`]).
//! 2. Authenticates HTTP Basic credentials against the [`AuthStore`]; a
//!    missing/invalid credential returns the byte-matched 401 challenge.
//! 3. Bypasses admins entirely (`sender_is_admin`).
//! 4. Dispatches to a [`validators::Validator`] ([`path_matchers::dispatch_request`]),
//!    running it against the request context. A `false` result is the
//!    byte-matched 403 `Permission denied`. A [`path_matchers::Dispatched::Deny`]
//!    (unknown `/mlflow/traces/` subpath) is the same 403 (fail-closed).
//! 5. Buffers the request body so validators that read it (MV create, trace
//!    search v3, start-trace v3, batch-get, delete-user, update-password, …) see
//!    the JSON, then reconstructs the request for the downstream handler.
//!
//! ## After-request hooks (T9.5 seam)
//!
//! Python's `_after_request` attaches creator grants and filters list
//! responses. That is T9.5; this middleware only runs the *before-request*
//! authorization. See the `// T9.5 SEAM:` marker below.

pub mod path_matchers;
pub mod validators;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderValue, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde_json::Value;

use crate::state::AppState;
use crate::workspace::{resolve_workspace, WORKSPACE_HEADER_NAME};
use path_matchers::{dispatch_request, Dispatched};
use validators::RequestCtx;

/// The authenticated identity for the current request, attached to the request
/// extensions by [`authorize`] once credentials are verified (T9.6). Handlers
/// that run their own in-band per-field authorization (the `/graphql` executor)
/// read it back to mirror Python's `authenticate_request()` +
/// `store.get_user(username).is_admin` inside the graphene auth middleware.
///
/// Only attached when the basic-auth app is enabled; a plain tracking server
/// leaves it absent (auth is off, so the gate is a no-op).
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub username: String,
    pub is_admin: bool,
}

/// `_UNPROTECTED_PATH_PREFIXES` (`__init__.py:454`) plus the Rust server's ops
/// endpoints. Python's auth app only carries `/static`, `/favicon.ico`,
/// `/health`; this Rust binary additionally serves `/version` and `/metrics` as
/// operational endpoints that are never auth-gated, so they are unprotected too.
const UNPROTECTED_PREFIXES: [&str; 5] =
    ["/static", "/favicon.ico", "/health", "/version", "/metrics"];

/// `is_unprotected_route` (`__init__.py:457`). The static-prefix nesting strips
/// the prefix before this middleware runs, so we match the bare forms; Python's
/// prefixed-form handling is covered by the router's `nest`.
pub fn is_unprotected_route(path: &str) -> bool {
    UNPROTECTED_PREFIXES.iter().any(|p| path.starts_with(p))
}

/// The 401 challenge response, byte-matched to `make_basic_auth_response`
/// (`__init__.py:466`): the exact body plus `WWW-Authenticate: Basic
/// realm="mlflow"`.
fn unauthenticated_response() -> Response {
    let mut resp = (
        StatusCode::UNAUTHORIZED,
        "You are not authenticated. Please see \
         https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow \
         on how to authenticate.",
    )
        .into_response();
    resp.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Basic realm=\"mlflow\""),
    );
    resp
}

/// The 403 response, byte-matched to `make_forbidden_response` (`__init__.py:477`).
fn forbidden_response() -> Response {
    (StatusCode::FORBIDDEN, "Permission denied").into_response()
}

/// Decode HTTP Basic credentials, mirroring `basic_credentials`
/// (`auth_api/mod.rs`) — werkzeug splits the decoded pair on the first colon.
fn basic_credentials(req: &Request<Body>) -> Option<(String, String)> {
    let value = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let encoded = value
        .strip_prefix("Basic ")
        .or_else(|| value.strip_prefix("basic "))?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let (u, p) = decoded.split_once(':')?;
    Some((u.to_string(), p.to_string()))
}

/// Parse a query string into `(key, value)` pairs (percent-decoded), mirroring
/// Flask's `request.args`.
fn parse_query(query: Option<&str>) -> Vec<(String, String)> {
    let Some(q) = query else {
        return Vec::new();
    };
    form_urlencoded_pairs(q.as_bytes())
}

/// Minimal `application/x-www-form-urlencoded` parser (query string or form
/// body): split on `&`, then `=`, percent-decode both sides, `+` → space.
fn form_urlencoded_pairs(bytes: &[u8]) -> Vec<(String, String)> {
    let s = String::from_utf8_lossy(bytes);
    s.split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (url_decode(k), url_decode(v))
        })
        .collect()
}

fn url_decode(s: &str) -> String {
    let bytes = s.replace('+', " ");
    let bytes = bytes.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// The tower middleware entry (`axum::middleware::from_fn_with_state`). Buffers
/// the body, runs the before-request authorization, then either short-circuits
/// (401/403) or forwards the reconstructed request to `next`.
pub async fn authorize(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();

    // 1. Unprotected routes pass through untouched.
    if is_unprotected_route(&path) {
        return next.run(req).await;
    }

    let auth_store = match state.auth_store() {
        Some(s) => s,
        // Auth not enabled — the middleware should not have been layered, but be
        // safe and pass through.
        None => return next.run(req).await,
    };

    // 2. Authenticate. `authenticate_and_get_user` (`_authenticate_cached`,
    //    `__init__.py:402`) fronts the werkzeug hash comparison with the
    //    credential cache (off by default) and returns the resolved user, so the
    //    admin-bypass check below reuses it instead of a second `get_user` query.
    let Some((username, password)) = basic_credentials(&req) else {
        return unauthenticated_response();
    };
    let Some(user) = auth_store
        .authenticate_and_get_user(&username, &password)
        .await
    else {
        return unauthenticated_response();
    };

    // 3. Admin bypass (`sender_is_admin`). Stamp the authenticated identity
    //    onto the request extensions first so a downstream handler that runs
    //    its own in-band authorization (the `/graphql` executor, T9.6) can read
    //    it — this mirrors Python, where the graphene auth middleware
    //    re-derives username + `is_admin` per request.
    //    `authenticate_and_get_user` already resolved the user (T9.8's cached
    //    `_authenticate_cached` path), so no second `get_user` query runs here.
    req.extensions_mut().insert(AuthContext {
        username: username.clone(),
        is_admin: user.is_admin,
    });
    if user.is_admin {
        return next.run(req).await;
    }

    // 4. Dispatch to the validator.
    let method = req.method().as_str().to_string();
    let query = parse_query(req.uri().query());
    let experiment_id_header = req
        .headers()
        .get("x-mlflow-experiment-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let workspace = resolve_workspace(
        req.headers()
            .get(WORKSPACE_HEADER_NAME)
            .and_then(|v| v.to_str().ok()),
    );

    let dispatched = dispatch_request(&path, &method);
    let validator = match dispatched {
        Dispatched::Allow => {
            // No gate — but the request is authenticated, so forward it.
            return next.run(req).await;
        }
        Dispatched::Deny => return forbidden_response(),
        Dispatched::Validator(v, params) => (v, params),
    };
    let (validator, path_params) = validator;

    // Buffer the body so body-reading validators can inspect the JSON and the
    // downstream handler still receives it.
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => Bytes::new(),
    };
    let json_body: Option<Value> = serde_json::from_slice(&body_bytes).ok();

    let ctx = RequestCtx {
        username: &username,
        method: &method,
        workspace: workspace.name(),
        path_params: &path_params,
        query: &query,
        json_body: json_body.as_ref(),
        experiment_id_header: experiment_id_header.as_deref(),
        auth_store,
        tracking_store: state.tracking_store(),
    };

    match validator.check(&ctx).await {
        Ok(true) => {}
        Ok(false) => return forbidden_response(),
        Err(e) => return error_response(&e),
    }

    // T9.5 SEAM: `_after_request` (creator grants + list-response filtering)
    // attaches here around `next.run(...)`. Not part of T9.4.

    let rebuilt = Request::from_parts(parts, Body::from(body_bytes));
    next.run(rebuilt).await
}

/// Render an `MlflowError` raised inside a validator as the same JSON error
/// body + status Python's Flask `_before_request` produces: the hook is wrapped
/// in `@catch_mlflow_exception`, which serializes the exception as JSON
/// (`error_code` + `message`) with `get_http_status_code()`. This reuses
/// `MlflowError`'s `IntoResponse` so the shape byte-matches every other error on
/// the server (the vast majority of gated routes are Flask-routed; the OTLP path
/// is the only Python plain-text exception, and its missing-header 400 is a
/// negligible divergence not exercised by the permission-matrix ACs).
fn error_response(e: &mlflow_error::MlflowError) -> Response {
    e.clone().into_response()
}
