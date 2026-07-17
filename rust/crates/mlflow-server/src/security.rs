//! Security middleware parity (plan T11.2), mirroring
//! `mlflow/server/security.py` + `mlflow/server/security_utils.py`.
//!
//! Provides host-header validation (DNS-rebinding protection), CORS
//! (flask-cors-compatible response headers), cross-origin state-change
//! blocking, and the `X-Frame-Options` / `X-Content-Type-Options` security
//! headers. Implemented as a single tower middleware
//! ([`security_middleware`]) applied as the *outermost* layer in
//! `build_app_with_recorder`, so — like Python, where
//! `security.init_security_middleware(app)` registers its `before_request`
//! hooks before the auth app's `_before_request` — a disallowed Host is
//! rejected (403) before authentication ever runs (no 401 challenge).
//!
//! ## Semantics mirrored from Python
//!
//! * **Host allowlist** (`MLFLOW_SERVER_ALLOWED_HOSTS`, default
//!   [`default_allowed_hosts`]): `fnmatch`-style patterns; a wildcard `*`
//!   entry disables the check entirely (`security.py:73`). `/health` and
//!   `/version` are exempt (`HEALTH_ENDPOINTS`, `security_utils.py:24`). A
//!   disallowed Host yields the byte-exact 403 [`INVALID_HOST_MSG`],
//!   `Content-Type: text/plain; charset=utf-8`.
//! * **CORS** (`MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`, default none): the
//!   effective allowlist is the configured origins **plus** the localhost
//!   patterns (`LOCALHOST_ORIGIN_PATTERNS`, `security_utils.py:42`), matching
//!   `security.py:65`. A request whose `Origin` matches gets
//!   `Access-Control-Allow-Origin: <origin>`,
//!   `Access-Control-Allow-Credentials: true`, and `Vary: Origin` — flask-cors'
//!   reflected-origin behavior. Preflight (`OPTIONS` with
//!   `Access-Control-Request-Method`) additionally gets
//!   `Access-Control-Allow-Methods: DELETE, GET, OPTIONS, PATCH, POST, PUT`
//!   and echoes `Access-Control-Request-Headers` back as
//!   `Access-Control-Allow-Headers`, and short-circuits with a 204 empty body.
//!   Wildcard mode (`*`) reflects any origin but omits the credentials header
//!   (`supports_credentials=False`, `security.py:63`).
//! * **Cross-origin state-change block** (`security.py:96`): a state-changing
//!   method (POST/PUT/DELETE/PATCH) to an API endpoint with a non-localhost
//!   `Origin` not in the allowlist yields the byte-exact 403
//!   [`CORS_BLOCKED_MSG`]. Skipped entirely in wildcard mode.
//! * **Security headers** (`security.py:108` `after_request`):
//!   `X-Content-Type-Options: nosniff` on every response, and
//!   `X-Frame-Options` (`MLFLOW_SERVER_X_FRAME_OPTIONS`, default `SAMEORIGIN`;
//!   the value `NONE` disables the header) on every response — including 403
//!   rejections and 404s.
//!
//! The notebook-trace-renderer `X-Frame-Options` exemption
//! (`security.py:113`) is not mirrored: this Rust server serves no such HTML
//! asset route, so there is nothing to exempt.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// `INVALID_HOST_MSG` (`security_utils.py:17`).
pub const INVALID_HOST_MSG: &str = "Invalid Host header - possible DNS rebinding attack detected";
/// `CORS_BLOCKED_MSG` (`security_utils.py:18`).
pub const CORS_BLOCKED_MSG: &str = "Cross-origin request blocked";

/// `HEALTH_ENDPOINTS` (`security_utils.py:24`): exempt from host validation.
const HEALTH_ENDPOINTS: [&str; 2] = ["/health", "/version"];

/// `API_PATH_PREFIX` / `AJAX_API_PATH_PREFIX` (`security_utils.py:27-28`).
const API_PATH_PREFIX: &str = "/api/";
const AJAX_API_PATH_PREFIX: &str = "/ajax-api/";

/// `TEST_ENDPOINTS` (`security_utils.py:31`): excluded from `is_api_endpoint`.
const TEST_ENDPOINTS: [&str; 2] = ["/test", "/api/test"];

/// `STATE_CHANGING_METHODS` (`security_utils.py:21`).
const STATE_CHANGING_METHODS: [Method; 4] =
    [Method::POST, Method::PUT, Method::DELETE, Method::PATCH];

/// `CORS_LOCALHOST_HOSTS` (`security_utils.py:35`): hostnames treated as
/// localhost for the cross-origin state-change bypass.
const CORS_LOCALHOST_HOSTS: [&str; 4] = ["localhost", "127.0.0.1", "[::1]", "::1"];

/// `LOCALHOST_VARIANTS` (`security_utils.py:34`).
const LOCALHOST_VARIANTS: [&str; 4] = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"];

/// flask-cors' fixed `Access-Control-Allow-Methods` value for the configured
/// method set (`security.py:70`, sorted+joined by flask-cors).
const CORS_ALLOW_METHODS: &str = "DELETE, GET, OPTIONS, PATCH, POST, PUT";

/// The default `X-Frame-Options` value (`MLFLOW_SERVER_X_FRAME_OPTIONS`
/// default, `environment_variables.py:1123`).
pub const DEFAULT_X_FRAME_OPTIONS: &str = "SAMEORIGIN";

/// Resolved security configuration, threaded into the middleware as tower
/// state. Built from CLI/env by [`SecurityConfig::from_parts`].
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Effective host allowlist (defaults applied when unset). An entry `*`
    /// disables host validation.
    allowed_hosts: Vec<String>,
    /// Configured CORS origins (the localhost patterns are appended
    /// internally). Empty = no configured origins (localhost still allowed).
    /// An entry `*` enables wildcard mode.
    allowed_origins: Vec<String>,
    /// The `X-Frame-Options` header value; `None` when the header is disabled
    /// (configured value `NONE`, case-insensitive).
    x_frame_options: Option<String>,
    /// Whether `allowed_origins` contains `*` (wildcard mode: reflect any
    /// origin, no credentials, no state-change block).
    wildcard_cors: bool,
    /// Whether `allowed_hosts` contains `*` (host validation disabled).
    wildcard_hosts: bool,
}

impl SecurityConfig {
    /// Build the resolved config from the three raw inputs (already
    /// comma-split into lists; `x_frame_options` as the raw string).
    ///
    /// * `allowed_hosts` — `None` applies [`default_allowed_hosts`]
    ///   (`security.py:30`).
    /// * `allowed_origins` — `None`/empty means no configured origins
    ///   (`security.py:35`).
    /// * `x_frame_options` — the raw value; `NONE` (case-insensitive) disables
    ///   the header (`security.py:115`). Uppercased to match Python's
    ///   `.upper()`.
    pub fn from_parts(
        allowed_hosts: Option<Vec<String>>,
        allowed_origins: Option<Vec<String>>,
        x_frame_options: &str,
    ) -> Self {
        let allowed_hosts = allowed_hosts.unwrap_or_else(default_allowed_hosts);
        let allowed_origins = allowed_origins.unwrap_or_default();
        let wildcard_cors = allowed_origins.iter().any(|o| o == "*");
        let wildcard_hosts = allowed_hosts.iter().any(|h| h == "*");
        let x_frame_options = normalize_x_frame_options(x_frame_options);
        Self {
            allowed_hosts,
            allowed_origins,
            x_frame_options,
            wildcard_cors,
            wildcard_hosts,
        }
    }
}

/// `MLFLOW_SERVER_X_FRAME_OPTIONS` handling (`security.py:115`): the value is
/// uppercased; `NONE` disables the header (returns `None`).
fn normalize_x_frame_options(value: &str) -> Option<String> {
    let upper = value.to_uppercase();
    if upper.is_empty() || upper == "NONE" {
        None
    } else {
        Some(upper)
    }
}

/// `get_default_allowed_hosts` (`security_utils.py:149`): localhost variants,
/// their `:*` wildcard forms (with the IPv6 bracket escaped for `fnmatch`),
/// and the private-IP-range patterns.
pub fn default_allowed_hosts() -> Vec<String> {
    let mut hosts: Vec<String> = LOCALHOST_VARIANTS.iter().map(|h| h.to_string()).collect();
    for host in LOCALHOST_VARIANTS {
        if let Some(rest) = host.strip_prefix('[') {
            // IPv6: escape the opening bracket for fnmatch (`[` -> `[[]`),
            // matching Python's `host.replace("[", "[[]", 1)`.
            hosts.push(format!("[[]{rest}:*"));
        } else {
            hosts.push(format!("{host}:*"));
        }
    }
    hosts.extend(private_ip_patterns());
    hosts
}

/// `get_private_ip_patterns` (`security_utils.py:54`): RFC-1918/4193 wildcard
/// patterns. `172.16.*` .. `172.31.*` plus the class-A/B/C and IPv6 ULA forms.
fn private_ip_patterns() -> Vec<String> {
    let mut patterns = vec!["192.168.*".to_string(), "10.*".to_string()];
    for i in 16..32 {
        patterns.push(format!("172.{i}.*"));
    }
    patterns.push("fc00:*".to_string());
    patterns.push("fd00:*".to_string());
    patterns
}

/// `is_api_endpoint` (`security_utils.py:127`).
fn is_api_endpoint(path: &str) -> bool {
    (path.starts_with(API_PATH_PREFIX) || path.starts_with(AJAX_API_PATH_PREFIX))
        && !TEST_ENDPOINTS.contains(&path)
}

/// `fnmatch.fnmatch`-style match used by both host and origin allowlists
/// (`security_utils.py:120,144`): shell-glob semantics. We support the two
/// metacharacters Python's `fnmatch` uses in these patterns — `*` (any run,
/// including `.`/`/`, matching `fnmatch`'s non-path-aware behavior) and `?`
/// (any single char) — plus `[...]` character classes (used for the escaped
/// IPv6 bracket `[[]`). Case-insensitive, like `fnmatch` on the default
/// (case-normalizing) platforms MLflow targets.
fn fnmatch(name: &str, pattern: &str) -> bool {
    let name = name.to_lowercase();
    let pattern = pattern.to_lowercase();
    glob_match(name.as_bytes(), pattern.as_bytes())
}

/// Recursive shell-glob matcher supporting `*`, `?`, and `[...]` classes.
fn glob_match(name: &[u8], pattern: &[u8]) -> bool {
    match pattern.first() {
        None => name.is_empty(),
        Some(b'*') => {
            // `*` matches zero or more chars: try consuming nothing, else one.
            glob_match(name, &pattern[1..]) || (!name.is_empty() && glob_match(&name[1..], pattern))
        }
        Some(b'?') => !name.is_empty() && glob_match(&name[1..], &pattern[1..]),
        Some(b'[') => match_class(name, pattern),
        Some(&c) => !name.is_empty() && name[0] == c && glob_match(&name[1..], &pattern[1..]),
    }
}

/// Match a `[...]` character class at the head of `pattern`. Supports a leading
/// `!`/`^` negation and the literal-`]`-first convention. Falls back to a
/// literal `[` match when the class is unterminated (mirroring `fnmatch`).
fn match_class(name: &[u8], pattern: &[u8]) -> bool {
    let Some(close) = find_class_end(pattern) else {
        // Unterminated: treat `[` literally.
        return !name.is_empty() && name[0] == b'[' && glob_match(&name[1..], &pattern[1..]);
    };
    if name.is_empty() {
        return false;
    }
    let body = &pattern[1..close];
    let (negated, body) = match body.first() {
        Some(b'!') | Some(b'^') => (true, &body[1..]),
        _ => (false, body),
    };
    let matched = class_contains(body, name[0]);
    (matched != negated) && glob_match(&name[1..], &pattern[close + 1..])
}

/// Find the index of the `]` that closes a class starting at index 0. A `]`
/// immediately after `[` (or after a leading negation) is a literal member.
fn find_class_end(pattern: &[u8]) -> Option<usize> {
    let mut i = 1;
    if matches!(pattern.get(i), Some(b'!') | Some(b'^')) {
        i += 1;
    }
    if pattern.get(i) == Some(&b']') {
        i += 1;
    }
    while i < pattern.len() {
        if pattern[i] == b']' {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Whether a character class body (ranges + literals) contains `c`.
fn class_contains(body: &[u8], c: u8) -> bool {
    let mut i = 0;
    while i < body.len() {
        if i + 2 < body.len() && body[i + 1] == b'-' {
            if body[i] <= c && c <= body[i + 2] {
                return true;
            }
            i += 3;
        } else {
            if body[i] == c {
                return true;
            }
            i += 1;
        }
    }
    false
}

/// `is_allowed_host_header` (`security_utils.py:134`).
fn is_allowed_host(allowed_hosts: &[String], host: Option<&str>) -> bool {
    let Some(host) = host.filter(|h| !h.is_empty()) else {
        return false;
    };
    if allowed_hosts.iter().any(|h| h == "*") {
        return true;
    }
    // Python only switches to fnmatch when the pattern contains `*`
    // (`security_utils.py:144`); every other pattern (including the escaped
    // IPv6 form `[[]::1]:*` — which does contain `*` — and the bare `[::1]`
    // literal — which does not) is matched by exact string equality.
    allowed_hosts.iter().any(|allowed| {
        if allowed.contains('*') {
            fnmatch(host, allowed)
        } else {
            host == allowed
        }
    })
}

/// Parse the hostname out of an `Origin` header value for the localhost check
/// (`is_localhost_origin`, `security_utils.py:93` — Python uses
/// `urlparse(origin).hostname`).
fn origin_hostname(origin: &str) -> Option<String> {
    let after_scheme = origin.split_once("://").map(|(_, rest)| rest)?;
    // Strip path/query, then userinfo, then port.
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    // IPv6 literal: `[::1]` (keep the brackets — matches `urlparse` for the
    // CORS_LOCALHOST_HOSTS comparison against `[::1]`... but urlparse strips
    // brackets, yielding `::1`. Handle both bracketed and bare forms below).
    if let Some(rest) = host_port.strip_prefix('[') {
        let host = rest.split(']').next().unwrap_or(rest);
        return Some(host.to_lowercase());
    }
    let host = host_port.rsplit_once(':').map_or(host_port, |(h, _)| h);
    Some(host.to_lowercase())
}

/// `is_localhost_origin` (`security_utils.py:93`).
fn is_localhost_origin(origin: &str) -> bool {
    if origin.is_empty() {
        return false;
    }
    match origin_hostname(origin) {
        // urlparse strips IPv6 brackets, so `[::1]` -> `::1`; our parser keeps
        // the bare form for both `[::1]` and `::1`, and `CORS_LOCALHOST_HOSTS`
        // carries both `[::1]` and `::1`, so a straight membership test works.
        Some(host) => CORS_LOCALHOST_HOSTS.iter().any(|h| {
            let h = h.trim_start_matches('[').trim_end_matches(']');
            h == host
        }),
        None => false,
    }
}

/// Whether an origin is in the effective CORS allowlist (configured origins +
/// localhost). Mirrors flask-cors' per-origin match against
/// `cors_origins = allowed_origins + LOCALHOST_ORIGIN_PATTERNS`
/// (`security.py:65`). The localhost patterns are regexes in Python; we model
/// them via [`is_localhost_origin`] (same set of accepted origins).
fn origin_allowed(config: &SecurityConfig, origin: &str) -> bool {
    if config.wildcard_cors {
        return true;
    }
    if is_localhost_origin(origin) {
        return true;
    }
    config.allowed_origins.iter().any(|allowed| {
        if allowed.contains('*') {
            fnmatch(origin, allowed)
        } else {
            origin == allowed
        }
    })
}

/// `should_block_cors_request` (`security_utils.py:106`).
fn should_block_cors_request(
    config: &SecurityConfig,
    origin: Option<&str>,
    method: &Method,
) -> bool {
    let Some(origin) = origin.filter(|o| !o.is_empty()) else {
        return false;
    };
    if !STATE_CHANGING_METHODS.contains(method) {
        return false;
    }
    if is_localhost_origin(origin) {
        return false;
    }
    if !config.allowed_origins.is_empty() {
        if config.wildcard_cors {
            return false;
        }
        return !config.allowed_origins.iter().any(|allowed| {
            if allowed.contains('*') {
                fnmatch(origin, allowed)
            } else {
                origin == allowed
            }
        });
    }
    true
}

/// Byte-exact plain-text 403 (`Response(msg, status=FORBIDDEN,
/// mimetype="text/plain")`, `security.py:82`). Flask renders `text/plain`
/// with a `; charset=utf-8` suffix; we set the same.
fn plain_forbidden(msg: &'static str) -> Response {
    (
        StatusCode::FORBIDDEN,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        msg,
    )
        .into_response()
}

/// The single tower middleware implementing the full security pipeline. Runs
/// as the outermost layer (before auth), mirroring the Flask `before_request`
/// ordering: host validation → cross-origin state-change block → downstream,
/// then the `after_request` CORS + security-header decoration on the way out.
pub async fn security_middleware(
    State(config): State<SecurityConfig>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    let method = request.method().clone();
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let acr_headers = request
        .headers()
        .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let is_preflight = method == Method::OPTIONS
        && request
            .headers()
            .contains_key(header::ACCESS_CONTROL_REQUEST_METHOD);

    // 1. Host validation (`validate_host`, `security.py:75`). Skipped when a
    //    `*` host is configured, and for the health endpoints.
    if !config.wildcard_hosts
        && !HEALTH_ENDPOINTS.contains(&path.as_str())
        && !is_allowed_host(&config.allowed_hosts, host.as_deref())
    {
        return decorate(
            plain_forbidden(INVALID_HOST_MSG),
            &config,
            &path,
            origin.as_deref(),
        );
    }

    // 2. Cross-origin state-change block (`block_cross_origin_state_changes`,
    //    `security.py:96`). Only for API endpoints, skipped in wildcard mode.
    if !config.wildcard_cors
        && is_api_endpoint(&path)
        && should_block_cors_request(&config, origin.as_deref(), &method)
    {
        return decorate(
            plain_forbidden(CORS_BLOCKED_MSG),
            &config,
            &path,
            origin.as_deref(),
        );
    }

    // 3. flask-cors preflight short-circuit: an OPTIONS with
    //    `Access-Control-Request-Method` returns 204 with an empty body
    //    directly (flask-cors intercepts before the view). For a disallowed
    //    origin it still returns 204 but omits the CORS headers.
    if is_preflight {
        let mut resp = (StatusCode::NO_CONTENT, Body::empty()).into_response();
        if origin
            .as_deref()
            .is_some_and(|o| origin_allowed(&config, o))
        {
            add_preflight_cors_headers(
                &mut resp,
                &config,
                origin.as_deref().unwrap(),
                acr_headers.as_deref(),
            );
        }
        return decorate(resp, &config, &path, origin.as_deref());
    }

    let response = next.run(request).await;
    decorate(response, &config, &path, origin.as_deref())
}

/// Apply flask-cors' actual-request headers plus the `after_request` security
/// headers (`security.py:108`) to an outgoing response.
fn decorate(
    mut response: Response,
    config: &SecurityConfig,
    path: &str,
    origin: Option<&str>,
) -> Response {
    // CORS actual-request headers (non-preflight; preflight sets its own set
    // before reaching here and `add_actual_cors_headers` is idempotent on the
    // origin/credentials/vary triple).
    if let Some(origin) = origin.filter(|o| origin_allowed(config, o)) {
        add_actual_cors_headers(&mut response, config, origin);
    }
    add_security_headers(&mut response, config, path);
    response
}

fn add_actual_cors_headers(response: &mut Response, config: &SecurityConfig, origin: &str) {
    let headers = response.headers_mut();
    set_header(headers, header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    if !config.wildcard_cors {
        set_header(headers, header::ACCESS_CONTROL_ALLOW_CREDENTIALS, "true");
    }
    set_header(headers, header::VARY, "Origin");
}

fn add_preflight_cors_headers(
    response: &mut Response,
    config: &SecurityConfig,
    origin: &str,
    acr_headers: Option<&str>,
) {
    add_actual_cors_headers(response, config, origin);
    let headers = response.headers_mut();
    set_header(
        headers,
        header::ACCESS_CONTROL_ALLOW_METHODS,
        CORS_ALLOW_METHODS,
    );
    // flask-cors echoes the requested headers back verbatim; omits the header
    // entirely when the request carried none.
    if let Some(acr) = acr_headers {
        set_header(headers, header::ACCESS_CONTROL_ALLOW_HEADERS, acr);
    }
}

/// `add_security_headers` (`security.py:108`): `X-Content-Type-Options` always,
/// `X-Frame-Options` unless disabled.
fn add_security_headers(response: &mut Response, config: &SecurityConfig, _path: &str) {
    let headers = response.headers_mut();
    set_header(
        headers,
        HeaderName::from_static("x-content-type-options"),
        "nosniff",
    );
    if let Some(xfo) = &config.x_frame_options {
        set_header(headers, HeaderName::from_static("x-frame-options"), xfo);
    }
}

fn set_header(headers: &mut axum::http::HeaderMap, name: HeaderName, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        headers.insert(name, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hosts_include_localhost_and_private_ranges() {
        let hosts = default_allowed_hosts();
        for expected in [
            "localhost",
            "127.0.0.1",
            "[::1]",
            "localhost:*",
            "127.0.0.1:*",
            "[[]::1]:*",
            "192.168.*",
            "10.*",
            "172.16.*",
            "172.31.*",
        ] {
            assert!(hosts.contains(&expected.to_string()), "missing {expected}");
        }
        assert!(!hosts.contains(&"172.32.*".to_string()));
    }

    #[test]
    fn host_validation_matches_python() {
        let hosts = default_allowed_hosts();
        for (host, valid) in [
            ("192.168.1.1", true),
            ("10.0.0.1", true),
            ("172.16.0.1", true),
            ("127.0.0.1", true),
            ("localhost", true),
            ("[::1]", true),
            ("192.168.1.1:8080", true),
            ("[::1]:8080", true),
            ("evil.com", false),
        ] {
            assert_eq!(is_allowed_host(&hosts, Some(host)), valid, "host={host}");
        }
    }

    #[test]
    fn empty_or_missing_host_rejected() {
        let hosts = default_allowed_hosts();
        assert!(!is_allowed_host(&hosts, None));
        assert!(!is_allowed_host(&hosts, Some("")));
    }

    #[test]
    fn wildcard_host_allows_anything() {
        let hosts = vec!["*".to_string()];
        assert!(is_allowed_host(&hosts, Some("any.domain.com")));
    }

    #[test]
    fn wildcard_subdomain_host() {
        let hosts = vec!["*.example.com".to_string()];
        assert!(is_allowed_host(&hosts, Some("app.example.com")));
        assert!(is_allowed_host(&hosts, Some("sub.app.example.com")));
        assert!(!is_allowed_host(&hosts, Some("evil.com")));
    }

    #[test]
    fn is_api_endpoint_matches_python() {
        for (path, expected) in [
            ("/api/2.0/mlflow/experiments/list", true),
            ("/ajax-api/2.0/mlflow/experiments/list", true),
            ("/ajax-api/3.0/mlflow/runs/search", true),
            ("/api/test", false),
            ("/test", false),
            ("/health", false),
            ("/static/index.html", false),
        ] {
            assert_eq!(is_api_endpoint(path), expected, "path={path}");
        }
    }

    #[test]
    fn localhost_origins_recognized() {
        for origin in [
            "http://localhost:3000",
            "http://127.0.0.1:5000",
            "http://[::1]:8080",
            "http://localhost",
        ] {
            assert!(is_localhost_origin(origin), "origin={origin}");
        }
        assert!(!is_localhost_origin("http://evil.com"));
    }

    #[test]
    fn origin_allowed_with_configured_and_localhost() {
        let config = SecurityConfig::from_parts(
            None,
            Some(vec!["https://trusted.com".to_string()]),
            "SAMEORIGIN",
        );
        assert!(origin_allowed(&config, "https://trusted.com"));
        assert!(origin_allowed(&config, "http://localhost:3000"));
        assert!(!origin_allowed(&config, "http://evil.com"));
    }

    #[test]
    fn wildcard_origin_allows_any() {
        let config = SecurityConfig::from_parts(None, Some(vec!["*".to_string()]), "SAMEORIGIN");
        assert!(origin_allowed(&config, "http://any.domain.com"));
    }

    #[test]
    fn wildcard_origin_pattern() {
        let config = SecurityConfig::from_parts(
            None,
            Some(vec!["http://*.example.com".to_string()]),
            "SAMEORIGIN",
        );
        assert!(origin_allowed(&config, "http://app.example.com"));
        assert!(origin_allowed(&config, "http://sub.app.example.com"));
        assert!(!origin_allowed(&config, "http://evil.com"));
    }

    #[test]
    fn state_change_block_semantics() {
        let config = SecurityConfig::from_parts(
            None,
            Some(vec!["http://localhost:3000".to_string()]),
            "SAMEORIGIN",
        );
        // POST from disallowed origin -> block.
        assert!(should_block_cors_request(
            &config,
            Some("http://evil.com"),
            &Method::POST
        ));
        // GET never blocked.
        assert!(!should_block_cors_request(
            &config,
            Some("http://evil.com"),
            &Method::GET
        ));
        // No origin -> not blocked.
        assert!(!should_block_cors_request(&config, None, &Method::POST));
        // Localhost origin -> not blocked.
        assert!(!should_block_cors_request(
            &config,
            Some("http://localhost:9999"),
            &Method::POST
        ));
    }

    #[test]
    fn x_frame_options_normalization() {
        assert_eq!(
            normalize_x_frame_options("SAMEORIGIN"),
            Some("SAMEORIGIN".to_string())
        );
        assert_eq!(normalize_x_frame_options("deny"), Some("DENY".to_string()));
        assert_eq!(normalize_x_frame_options("NONE"), None);
        assert_eq!(normalize_x_frame_options("none"), None);
        assert_eq!(normalize_x_frame_options(""), None);
    }

    #[test]
    fn fnmatch_basic() {
        assert!(fnmatch("app.example.com", "*.example.com"));
        assert!(!fnmatch("evil.com", "*.example.com"));
        assert!(fnmatch("[::1]:8080", "[[]::1]:*"));
        assert!(fnmatch("127.0.0.1:5000", "127.0.0.1:*"));
    }
}
