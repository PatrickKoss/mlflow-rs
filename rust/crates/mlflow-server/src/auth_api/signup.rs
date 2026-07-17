//! The `/signup` server-rendered form + CSRF token mechanics (plan T9.2 seam,
//! §3.16 "Signup UI"; plan T9.7).
//!
//! ## Python source of truth
//!
//! * `signup()` (`mlflow/server/auth/__init__.py:3721`): renders an inline
//!   `render_template_string` HTML form (no separate template file) with a
//!   hidden `csrf_token` field and username/password inputs.
//! * `create_user_ui(csrf)` (`:3798`): `csrf.protect()` first, then the
//!   content-type / empty-field / duplicate-user checks (`users.rs`'s
//!   `create_user_ui` handles those; this module adds the CSRF gate in front).
//! * CSRF mechanics are `flask_wtf.csrf` (`flask_wtf/csrf.py`):
//!   - `generate_csrf`: a random session-bound value is stored in
//!     `session["csrf_token"]` (Flask's signed session cookie); the value
//!     embedded in the form is a **separately, timestamp-signed** derivation
//!     of that session value (`itsdangerous.URLSafeTimedSerializer`).
//!   - `validate_csrf`: re-derives + checks the signature/age of the submitted
//!     token, then `hmac.compare_digest`s it against the session value.
//!   - Errors (in check order), all raised as `ValidationError` and surfaced by
//!     `CSRFProtect._error_response` as a Werkzeug `BadRequest` — HTTP 400,
//!     plain-text body equal to the message, no MLflow JSON error envelope:
//!     - missing submitted token: `"The CSRF token is missing."`
//!     - no session token: `"The CSRF session token is missing."`
//!     - signature expired (> `WTF_CSRF_TIME_LIMIT`, default 3600s):
//!       `"The CSRF token has expired."`
//!     - signature malformed/wrong key: `"The CSRF token is invalid."`
//!     - signature valid but doesn't match the session value:
//!       `"The CSRF tokens do not match."`
//!
//! ## Rust reproduction (observable parity, not byte-identical cookies)
//!
//! Per the plan's D12, the Rust server does **not** read
//! `MLFLOW_FLASK_SERVER_SECRET_KEY` (that stays Python-side / is meaningless
//! here since this process never shares a session cookie with a Python
//! instance) — it holds its own HMAC-SHA256 secret
//! ([`CsrfSecret`], a random 32-byte key generated once per process; this
//! server has no multi-worker deployment mode, so the "must be static across
//! workers" constraint that forces Python's secret to come from an env var
//! doesn't apply here). The two-layer design is reproduced with HMAC in place
//! of itsdangerous's signer:
//!
//! * The "session" half becomes a random 256-bit value (`uuid::Uuid::new_v4`
//!   x2, analogous to Python's `sha1(os.urandom(64))` session value) carried
//!   in a `Set-Cookie` as its own signed envelope,
//!   `<value>.<hex(hmac(secret, "session." + value))>` — HttpOnly,
//!   `SameSite=Strict`, `Path=/`, so it can't be read or forged by script or
//!   cross-site requests.
//! * The form-embedded token is a **second**, timestamped envelope signing
//!   that *same* session value: `<ts>.<value>.<hex(hmac(secret, "csrf." + ts +
//!   "." + value))>` — the same shape as itsdangerous's
//!   `dumps`/`loads(max_age=...)`, where the payload travels inside the signed
//!   envelope rather than being re-derived from anything the verifier already
//!   has.
//! * Validation re-derives the session value from the cookie, checks the form
//!   token's own signature + age, then compares the two envelopes' payloads
//!   (`hmac.compare_digest`-equivalent, via `subtle::ConstantTimeEq`) — so a
//!   well-signed token carrying a value that doesn't match the cookie's
//!   session value is the reachable "tokens do not match" case, exactly like
//!   Python's. Check order and error strings match Python's `validate_csrf`
//!   exactly (see [`CsrfError`] and [`validate_csrf_request`]), so a
//!   CSRF-less POST, a missing/expired/invalid token, and a token/cookie
//!   mismatch are each distinguishable the same way a Python client would
//!   observe them — same 400 status, same message text.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::state::AppState;

/// `WTF_CSRF_TIME_LIMIT` default (`flask_wtf/csrf.py:222`): 60 minutes.
const CSRF_TIME_LIMIT_SECS: u64 = 3600;

/// `SIGNUP = "/signup"` (`auth/routes.py:4`).
pub const SIGNUP_PATH: &str = "/signup";
/// `HOME = "/"` (`auth/routes.py:3`).
pub const HOME_PATH: &str = "/";
/// `CREATE_USER_UI = _get_rest_path("/mlflow/users/create-ui")`
/// (`auth/routes.py:7`) — the `/api/2.0` REST path the signup form posts to.
pub const CREATE_USER_UI_PATH: &str = "/api/2.0/mlflow/users/create-ui";

/// The cookie carrying the signed session nonce (Python: the `session`
/// cookie's `csrf_token` key). Not a real Flask session — just the minimal
/// carrier this CSRF scheme needs.
const CSRF_COOKIE_NAME: &str = "mlflow_csrf_session";

/// The hidden form field name (`WTF_CSRF_FIELD_NAME` default, `csrf.py:219`).
pub const CSRF_FIELD_NAME: &str = "csrf_token";

/// This server's CSRF signing secret (plan D12: Rust owns its own secret,
/// distinct from Python's `MLFLOW_FLASK_SERVER_SECRET_KEY`). Generated once
/// per process from a UUIDv4 pair, which is plenty of entropy for an
/// HMAC key and needs no extra crate ( `uuid` is already a workspace dep and
/// its v4 generator draws from the OS CSPRNG via `getrandom`).
#[derive(Clone)]
pub struct CsrfSecret(Vec<u8>);

impl CsrfSecret {
    /// A fresh random secret. Called once at server startup; every `/signup`
    /// GET and `create-user-ui` POST on this process shares it.
    pub fn generate() -> Self {
        let combined = format!("{}{}", uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        Self(combined.into_bytes())
    }

    fn hmac_hex(&self, message: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.0).expect("HMAC accepts any key length");
        mac.update(message.as_bytes());
        hex_encode(&mac.finalize().into_bytes())
    }
}

/// A freshly issued CSRF pair for a `GET /signup` response: the cookie value
/// to set and the token value to embed in the hidden form field.
pub struct CsrfIssue {
    pub cookie_value: String,
    pub form_token: String,
}

/// Issue a new session value + form token pair (`generate_csrf`,
/// `csrf.py:25`).
///
/// Mirrors Python's two-envelope shape precisely: `session_value` is the raw
/// random value that would live in `session["csrf_token"]`, carried here in
/// the `Set-Cookie` inside its *own* HMAC envelope (`sign_payload`); the form
/// token is a *second*, timestamped HMAC envelope that signs that same raw
/// value (`itsdangerous.dumps(session[field_name])`). Validation re-derives
/// the session value from the cookie, then checks whether the form token's
/// signed payload equals it — so a well-signed token carrying a *different*
/// payload than the cookie's is the reachable "tokens do not match" case,
/// exactly like Python's `hmac.compare_digest(session[field_name], token)`.
pub fn issue_csrf(secret: &CsrfSecret) -> CsrfIssue {
    let session_value = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let cookie_value = sign_payload(secret, "session", &session_value);
    let ts = now_unix();
    let form_token = sign_timestamped_payload(secret, ts, &session_value);
    CsrfIssue {
        cookie_value,
        form_token,
    }
}

/// Sign an arbitrary payload as `<payload>.<hex(hmac(secret, domain + "." +
/// payload))>`. `domain` namespaces the two envelopes (`"session"` vs the
/// timestamped form-token domain) so a session cookie can never be replayed
/// as a form token or vice versa.
fn sign_payload(secret: &CsrfSecret, domain: &str, payload: &str) -> String {
    let sig = secret.hmac_hex(&format!("{domain}.{payload}"));
    format!("{payload}.{sig}")
}

/// Verify a [`sign_payload`] envelope and return the payload.
fn verify_payload(secret: &CsrfSecret, domain: &str, envelope: &str) -> Option<String> {
    let (payload, sig) = envelope.split_once('.')?;
    let expected = secret.hmac_hex(&format!("{domain}.{payload}"));
    bool::from(sig.as_bytes().ct_eq(expected.as_bytes())).then(|| payload.to_string())
}

/// Sign a payload together with an issue timestamp:
/// `<ts>.<payload>.<hex(hmac(secret, "csrf." + ts + "." + payload))>` —
/// analogous to `URLSafeTimedSerializer.dumps`, which prepends the timestamp
/// inside the signed envelope so `loads(max_age=...)` can reject stale
/// tokens without a second round trip.
fn sign_timestamped_payload(secret: &CsrfSecret, ts: u64, payload: &str) -> String {
    let sig = secret.hmac_hex(&format!("csrf.{ts}.{payload}"));
    format!("{ts}.{payload}.{sig}")
}

/// Verify a [`sign_timestamped_payload`] envelope's signature and freshness,
/// returning `(timestamp, payload)`. Signature/format problems are reported
/// separately from staleness so callers can distinguish `Expired` from
/// `Invalid` (Python: `SignatureExpired` vs `BadData`).
fn verify_timestamped_payload(
    secret: &CsrfSecret,
    envelope: &str,
) -> Result<(u64, String), CsrfError> {
    let mut parts = envelope.splitn(3, '.');
    let ts = parts.next().ok_or(CsrfError::Invalid)?;
    let payload = parts.next().ok_or(CsrfError::Invalid)?;
    let sig = parts.next().ok_or(CsrfError::Invalid)?;
    let ts: u64 = ts.parse().map_err(|_| CsrfError::Invalid)?;
    let expected = secret.hmac_hex(&format!("csrf.{ts}.{payload}"));
    if !bool::from(sig.as_bytes().ct_eq(expected.as_bytes())) {
        return Err(CsrfError::Invalid);
    }
    let age = now_unix().saturating_sub(ts);
    if age > CSRF_TIME_LIMIT_SECS {
        return Err(CsrfError::Expired);
    }
    Ok((ts, payload.to_string()))
}

/// Why CSRF validation failed, carrying Python's verbatim message text
/// (`flask_wtf.csrf.validate_csrf`, `csrf.py:101-117`) and the enclosing
/// `CSRFProtect._error_response` status: always 400.
#[derive(Debug, PartialEq, Eq)]
pub enum CsrfError {
    /// No `csrf_token` field on the request (`"The CSRF token is missing."`).
    TokenMissing,
    /// No session cookie / malformed session cookie
    /// (`"The CSRF session token is missing."`).
    SessionMissing,
    /// The submitted token's timestamp is older than the time limit
    /// (`"The CSRF token has expired."`).
    Expired,
    /// The submitted token doesn't parse or its signature doesn't verify
    /// (`"The CSRF token is invalid."`).
    Invalid,
    /// Both signatures verify but the derived values don't match
    /// (`"The CSRF tokens do not match."`).
    Mismatch,
}

impl CsrfError {
    /// The exact message Werkzeug's `BadRequest` renders as the plain-text
    /// body (`CSRFError.description` defaults to the `ValidationError`
    /// message passed to `_error_response`).
    pub fn message(&self) -> &'static str {
        match self {
            CsrfError::TokenMissing => "The CSRF token is missing.",
            CsrfError::SessionMissing => "The CSRF session token is missing.",
            CsrfError::Expired => "The CSRF token has expired.",
            CsrfError::Invalid => "The CSRF token is invalid.",
            CsrfError::Mismatch => "The CSRF tokens do not match.",
        }
    }

    /// `CSRFError` is a Werkzeug `BadRequest` subclass: HTTP 400.
    pub fn status(&self) -> StatusCode {
        StatusCode::BAD_REQUEST
    }

    /// The plain-text 400 response Python's `CSRFProtect._error_response`
    /// produces (Werkzeug's default `BadRequest` body is the description
    /// text; MLflow installs no custom `errorhandler` for `CSRFError`).
    pub fn into_response(self) -> Response {
        Response::builder()
            .status(self.status())
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(axum::body::Body::from(self.message()))
            .expect("valid response")
    }
}

/// `WTF_CSRF_HEADERS` default (`csrf.py:220`): headers checked when the
/// submitted token isn't in the form (`CSRFProtect._get_csrf_token`,
/// `csrf.py:241-264`) — the AJAX/JS-client fallback for requests that carry
/// no form body at all (e.g. a JSON POST).
const CSRF_HEADER_NAMES: [&str; 2] = ["x-csrftoken", "x-csrf-token"];

/// Find the submitted CSRF token per `CSRFProtect._get_csrf_token`'s exact
/// fallback chain: the `csrf_token` form field first (passed in as
/// `form_token`, since this crate's form parsing lives at the call site,
/// `users::create_user_ui`), then the `X-CSRFToken` / `X-CSRF-Token`
/// headers (in that order).
pub fn csrf_token_from_request(headers: &HeaderMap, form_token: Option<&str>) -> Option<String> {
    if let Some(t) = form_token.filter(|t| !t.is_empty()) {
        return Some(t.to_string());
    }
    CSRF_HEADER_NAMES.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    })
}

/// Validate a `create-user-ui` POST's CSRF cookie + form token pair against
/// Python's exact check order (`validate_csrf`, `csrf.py:68-117`):
/// 1. submitted token present,
/// 2. session (cookie) present and its own signature verifies,
/// 3. submitted token's signature verifies (distinguishing expired vs
///    otherwise-invalid),
/// 4. the two derived values match via a constant-time compare
///    (`hmac.compare_digest`).
pub fn validate_csrf_request(
    secret: &CsrfSecret,
    headers: &HeaderMap,
    form_token: Option<&str>,
) -> Result<(), CsrfError> {
    let form_token = form_token
        .filter(|t| !t.is_empty())
        .ok_or(CsrfError::TokenMissing)?;

    let cookie_value = read_cookie(headers, CSRF_COOKIE_NAME).ok_or(CsrfError::SessionMissing)?;
    let session_value =
        verify_payload(secret, "session", &cookie_value).ok_or(CsrfError::SessionMissing)?;

    let (_ts, token_payload) = verify_timestamped_payload(secret, form_token)?;

    if !bool::from(token_payload.as_bytes().ct_eq(session_value.as_bytes())) {
        return Err(CsrfError::Mismatch);
    }
    Ok(())
}

/// Read a single cookie value from the `Cookie` header (`Cookie: a=1; b=2`).
fn read_cookie(headers: &HeaderMap, name: &str) -> Option<String> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    raw.split(';').find_map(|pair| {
        let pair = pair.trim();
        let (k, v) = pair.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Build the `Set-Cookie` header value for a freshly issued session nonce.
/// `HttpOnly` + `SameSite=Strict` + `Path=/`; no `Secure` attribute so the
/// dev-server-typical plain-HTTP signup flow still works (Python's own
/// SSL-strict referrer check in `CSRFProtect.protect` is likewise gated on
/// `request.is_secure`, so it never fires over plain HTTP either).
pub fn csrf_cookie_header(value: &str) -> HeaderValue {
    HeaderValue::from_str(&format!(
        "{CSRF_COOKIE_NAME}={value}; HttpOnly; SameSite=Strict; Path=/"
    ))
    .expect("cookie value is header-safe")
}

/// `MLFLOW_LOGO` (`mlflow/server/auth/logo.py`) — the inline SVG embedded in
/// the signup page via `{% autoescape false %}{{ mlflow_logo }}{% endautoescape %}`.
pub const MLFLOW_LOGO: &str = include_str!("mlflow_logo.svg");

/// `signup()` (`auth/__init__.py:3721`): the `/signup` HTML, byte-matching
/// Python's inline template with the concrete `csrf_token`/`users_route`
/// values substituted (this server has no Jinja renderer, so the `{{ }}`
/// placeholders are filled in directly — the static markup around them is
/// copied verbatim from the Python template string).
pub fn signup_html(csrf_token: &str, users_route: &str) -> String {
    format!(
        r#"
<style>
  form {{
    background-color: #F5F5F5;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
    padding: 20px;
    max-width: 400px;
    margin: 0 auto;
    font-family: Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }}

  input[type=text], input[type=password] {{
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #CCCCCC;
    border-radius: 4px;
    box-sizing: border-box;
  }}
  input[type=submit] {{
    background-color: rgb(34, 114, 180);
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
  }}

  input[type=submit]:hover {{
    background-color: rgb(14, 83, 139);
  }}

  .logo-container {{
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
  }}

  .logo {{
    max-width: 150px;
    margin-right: 10px;
  }}
</style>

<form action="{users_route}" method="post">
  <input type="hidden" name="{CSRF_FIELD_NAME}" value="{csrf_token}"/>
  <div class="logo-container">
    {MLFLOW_LOGO}
  </div>
  <label for="username">Username:</label>
  <br>
  <input type="text" id="username" name="username" minlength="4">
  <br>
  <label for="password">Password:</label>
  <br>
  <input type="password" id="password" name="password" minlength="12">
  <br>
  <br>
  <input type="submit" value="Sign up">
</form>
"#
    )
}

/// `alert(href)` (`auth/__init__.py:3703`): the flash-message + redirect
/// script Python renders after a UI signup attempt. Reproduces the flashed
/// message inline (this server has no Flask-session-backed flash queue, so
/// the message is baked directly into the script rather than round-tripped
/// through a session cookie) and the same `window.location.href` redirect.
pub fn alert_html(message: &str, href: &str) -> String {
    let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        r#"<script type = "text/javascript">
      alert("{escaped}");
      window.location.href = "{href}";
</script>
"#
    )
}

/// `GET /signup` (`signup()`, `auth/__init__.py:3721`): issue a fresh CSRF
/// pair and render the form. `AppState::csrf_secret` is always `Some` here —
/// this route is only mounted when the basic-auth app (and therefore the
/// secret) is enabled, mirroring how Python's `/signup` route only exists in
/// the auth app.
pub async fn signup_page(State(state): State<AppState>) -> Response {
    // AUTH SEAM (T9.4): `/signup` is unprotected in Python (no
    // `BEFORE_REQUEST` validator covers it — it's the entry point for a user
    // who by definition isn't authenticated yet), so this handler never
    // gates on identity.
    let Some(secret) = state.csrf_secret() else {
        // Unreachable in practice: `lib.rs` only mounts this route when
        // `auth_enabled()` (which implies `csrf_secret().is_some()`).
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(axum::body::Body::empty())
            .expect("valid response");
    };
    let issue = issue_csrf(secret);
    let html = signup_html(&issue.form_token, CREATE_USER_UI_PATH);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::SET_COOKIE, csrf_cookie_header(&issue.cookie_value))
        .body(axum::body::Body::from(html))
        .expect("valid response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issued_token_round_trips() {
        let secret = CsrfSecret::generate();
        let issue = issue_csrf(&secret);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("mlflow_csrf_session={}", issue.cookie_value)).unwrap(),
        );
        assert!(validate_csrf_request(&secret, &headers, Some(&issue.form_token)).is_ok());
    }

    #[test]
    fn missing_token_is_rejected() {
        let secret = CsrfSecret::generate();
        let headers = HeaderMap::new();
        assert_eq!(
            validate_csrf_request(&secret, &headers, None),
            Err(CsrfError::TokenMissing)
        );
    }

    #[test]
    fn missing_cookie_is_rejected() {
        let secret = CsrfSecret::generate();
        let issue = issue_csrf(&secret);
        let headers = HeaderMap::new();
        assert_eq!(
            validate_csrf_request(&secret, &headers, Some(&issue.form_token)),
            Err(CsrfError::SessionMissing)
        );
    }

    #[test]
    fn tampered_token_is_invalid() {
        let secret = CsrfSecret::generate();
        let issue = issue_csrf(&secret);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("mlflow_csrf_session={}", issue.cookie_value)).unwrap(),
        );
        let tampered = format!("{}x", issue.form_token);
        assert_eq!(
            validate_csrf_request(&secret, &headers, Some(&tampered)),
            Err(CsrfError::Invalid)
        );
    }

    #[test]
    fn cookie_from_different_secret_is_session_missing() {
        let secret_a = CsrfSecret::generate();
        let secret_b = CsrfSecret::generate();
        let issue_a = issue_csrf(&secret_a);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("mlflow_csrf_session={}", issue_a.cookie_value))
                .unwrap(),
        );
        // Validating issue_a's form token under secret_b: the cookie's own
        // signature no longer verifies under secret_b, so this surfaces as
        // `SessionMissing` (Python: a session cookie signed under a
        // different `SECRET_KEY` fails Flask's own cookie-signature check
        // before CSRF ever inspects it, which likewise yields an empty
        // session and "The CSRF session token is missing.").
        assert_eq!(
            validate_csrf_request(&secret_b, &headers, Some(&issue_a.form_token)),
            Err(CsrfError::SessionMissing)
        );
    }

    #[test]
    fn token_for_a_different_session_mismatches() {
        // Two independently issued sessions under the same secret: a
        // well-signed, non-expired token from session B presented alongside
        // session A's cookie is the reachable "tokens do not match" case
        // (Python: `hmac.compare_digest(session[field_name], token)` fails
        // even though the token's own signature verifies).
        let secret = CsrfSecret::generate();
        let issue_a = issue_csrf(&secret);
        let issue_b = issue_csrf(&secret);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("mlflow_csrf_session={}", issue_a.cookie_value))
                .unwrap(),
        );
        assert_eq!(
            validate_csrf_request(&secret, &headers, Some(&issue_b.form_token)),
            Err(CsrfError::Mismatch)
        );
    }

    #[test]
    fn expired_token_is_rejected() {
        let secret = CsrfSecret::generate();
        let session_value = "deadbeefdeadbeef";
        let cookie_value = sign_payload(&secret, "session", session_value);
        let old_ts = now_unix() - CSRF_TIME_LIMIT_SECS - 10;
        let form_token = sign_timestamped_payload(&secret, old_ts, session_value);

        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("mlflow_csrf_session={cookie_value}")).unwrap(),
        );
        assert_eq!(
            validate_csrf_request(&secret, &headers, Some(&form_token)),
            Err(CsrfError::Expired)
        );
    }
}
