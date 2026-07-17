//! The tower auth middleware (plan T9.4 + T9.5, §3.16): authenticate → admin
//! bypass → validator dispatch → permission enforcement → after-request hook,
//! mirroring `_before_request` (`mlflow/server/auth/__init__.py:2913`), the
//! FastAPI OTLP middleware (`add_fastapi_permission_middleware`, `:4488`), and
//! `_after_request` (`:3651`).
//!
//! Applied at the top of the app router by [`layer`] (only when auth is
//! enabled). It runs before *and* after every handler and:
//!
//! 1. Lets unprotected routes through ([`is_unprotected_route`]).
//! 2. Authenticates HTTP Basic credentials against the [`AuthStore`]; a
//!    missing/invalid credential returns the byte-matched 401 challenge.
//! 3. Bypasses the before-request validators for admins (`sender_is_admin`);
//!    the after-request hook still runs for them (creator grants apply, filters
//!    skip internally).
//! 4. Dispatches to a [`validators::Validator`] ([`path_matchers::dispatch_request`]),
//!    running it against the request context. A `false` result is the
//!    byte-matched 403 `Permission denied`. A [`path_matchers::Dispatched::Deny`]
//!    (unknown `/mlflow/traces/` subpath) is the same 403 (fail-closed).
//! 5. Buffers the request body so validators that read it (MV create, trace
//!    search v3, start-trace v3, batch-get, delete-user, update-password, …) see
//!    the JSON, then reconstructs the request for the downstream handler.
//! 6. After the handler runs, dispatches the after-request hook
//!    ([`path_matchers::dispatch_after_request`] → [`after_request::run`]) on a
//!    successful (`2xx`/`3xx`) response: creator MANAGE grants, search-response
//!    filtering (with page-fill), and the registered-model delete/rename grant
//!    cascade. See [`after_request`].

pub mod after_request;
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
use after_request::AfterCtx;
use path_matchers::{dispatch_after_request, dispatch_request, Dispatched};
use validators::RequestCtx;

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
pub async fn authorize(State(state): State<AppState>, req: Request<Body>, next: Next) -> Response {
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

    // 2. Authenticate.
    let Some((username, password)) = basic_credentials(&req) else {
        return unauthenticated_response();
    };
    if !auth_store.authenticate_user(&username, &password).await {
        return unauthenticated_response();
    }

    // 3. Admin flag (`sender_is_admin`). Admins bypass the before-request
    //    validators, but `_after_request` still runs for them (creator grants
    //    apply to admin-created resources; the filters short-circuit on
    //    `sender_is_admin` internally). A store error surfaces as the matching
    //    HTTP status (`catch_mlflow_exception`).
    let is_admin = match auth_store.get_user(&username).await {
        Ok(user) => user.is_admin,
        Err(e) => return error_response(&e),
    };

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

    // The after-request hook (if any) for this route. `_after_request` runs for
    // both admins and non-admins.
    let after_handler = dispatch_after_request(&path, &method);

    // 4. Before-request validator dispatch (skipped entirely for admins).
    let validator = if is_admin {
        None
    } else {
        match dispatch_request(&path, &method) {
            Dispatched::Allow => None,
            Dispatched::Deny => return forbidden_response(),
            Dispatched::Validator(v, params) => Some((v, params)),
        }
    };

    // Fast path: no validator and no after-request hook — forward untouched.
    if validator.is_none() && after_handler.is_none() {
        return next.run(req).await;
    }

    // Buffer the request body so body-reading validators can inspect the JSON,
    // the after-request hook can re-parse the search request, and the downstream
    // handler still receives it.
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => Bytes::new(),
    };
    let json_body: Option<Value> = serde_json::from_slice(&body_bytes).ok();

    if let Some((validator, path_params)) = validator {
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
    }

    // Capture the raw query string (owned) before rebuilding the request — the
    // after-request re-parse needs it, and `http::request::Parts` (its
    // `Extensions` field) is `!Send`, so we must not hold `parts` across the
    // downstream `await`.
    let request_query: Option<String> = parts.uri.query().map(str::to_string);

    // Run the downstream handler with the reconstructed request (consumes `parts`).
    let rebuilt = Request::from_parts(parts, Body::from(body_bytes.clone()));
    let resp = next.run(rebuilt).await;

    // 5. After-request hook (`_after_request`, T9.5). Only on a successful
    //    (`2xx`/`3xx`) response, mirroring the `400 <= status < 600` skip.
    let Some(after_handler) = after_handler else {
        return resp;
    };
    if resp.status().is_client_error() || resp.status().is_server_error() {
        return resp;
    }

    // Buffer the response body only when the handler needs it
    // ([`AfterRequestHandler::needs_response_body`]) — mirroring Python reading
    // `resp.json` only for the creator-grant / filter hooks.
    let (resp, resp_body) = if after_handler.needs_response_body() {
        let (rparts, rbody) = resp.into_parts();
        let bytes = match axum::body::to_bytes(rbody, usize::MAX).await {
            Ok(b) => b,
            Err(_) => Bytes::new(),
        };
        (
            Response::from_parts(rparts, Body::from(bytes.clone())),
            Some(bytes),
        )
    } else {
        (resp, None)
    };

    let ctx = AfterCtx {
        username: &username,
        workspace: workspace.name(),
        is_admin,
        method: &method,
        query: request_query.as_deref(),
        request_body: &body_bytes,
        request_json: json_body.as_ref(),
        state: &state,
    };
    after_request::run(after_handler, &ctx, resp, resp_body).await
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
