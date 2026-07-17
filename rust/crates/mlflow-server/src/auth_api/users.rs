//! The 8 user-management endpoints (plan T9.2, §3.16).
//!
//! Hand-rolled JSON shapes mirroring `mlflow/server/auth/__init__.py` — these
//! are NOT proto `ROUTE_TABLE` routes, so the request/response bodies are built
//! with `serde_json` rather than the proto codec. Each handler byte-matches its
//! Python counterpart's param extraction, error messages/codes, and response
//! shape.
//!
//! | Endpoint | Method | Python (`auth/__init__.py`) |
//! |----------|--------|------------------------------|
//! | `/users/create`          | POST   | `create_user` (`:3821`) |
//! | `/users/create-ui`       | POST   | `create_user_ui` (`:3798`) |
//! | `/users/get`             | GET    | `get_user` (`:3836`) |
//! | `/users/current`         | GET    | `get_current_user` (`:3873`) |
//! | `/users/list`            | GET    | `list_users` (`:3843`) |
//! | `/users/update-password` | PATCH  | `update_user_password` (`:3962`) |
//! | `/users/update-admin`    | PATCH  | `update_user_admin` (`:3995`) |
//! | `/users/delete`          | DELETE | `delete_user` (`:4004`) |
//!
//! Response `{"user": {...}}` uses the `User.to_json()` shape
//! (`auth/entities.py:39-44`): `{id, username, is_admin}` — no password hash, no
//! `experiment_permissions` / `registered_model_permissions` arrays (those
//! live on the legacy per-resource entities, which are dead at runtime).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::Response;
use mlflow_auth::{AuthStore, User};
use mlflow_error::{ErrorCode, MlflowError};
use serde_json::{json, Value};

use super::authenticated_username;
use crate::state::AppState;

/// `create_user` (`auth/__init__.py:3821`): `POST /mlflow/users/create`.
///
/// AUTH SEAM (T9.4): authorization is `validate_can_create_user` (super-admin
/// only, `__init__.py:1670`); the middleware gates it before this handler runs.
pub async fn create_user(
    State(state): State<AppState>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;

    // `if not request.is_json: return 400`.
    if !is_json(&parts) {
        return Ok(text_response(
            StatusCode::BAD_REQUEST,
            "Invalid content type. Must be application/json",
        ));
    }

    let args = json_body(&body)?;
    let username = get_request_param(&args, "username")?;
    let password = get_request_param(&args, "password")?;

    // `if not username or not password: return 400` — the empty-string guard is
    // separate from (and wins over) the store-level length/username validation.
    if username.is_empty() || password.is_empty() {
        return Ok(text_response(
            StatusCode::BAD_REQUEST,
            "Username and password cannot be empty.",
        ));
    }

    let user = store.create_user(&username, &password, false).await?;
    user_response(&user)
}

/// `create_user_ui` (`auth/__init__.py:3798`): `POST /mlflow/users/create-ui`.
///
/// The server-rendered signup form path. `csrf.protect()` runs **first**,
/// before the content-type check — a request with no valid CSRF pair is
/// rejected regardless of its content type or fields (T9.7 closes this seam;
/// T9.2 left it open, see the module-level history in the doc above). The
/// submitted token is looked up the way Python's `_get_csrf_token` does: the
/// `csrf_token` form field first, falling back to the `X-CSRFToken` /
/// `X-CSRF-Token` headers (relevant for non-form content types, whose bodies
/// this handler never parses as a form — see [`signup::csrf_token_from_request`]).
/// On a duplicate username Python flashes a message and redirects back to
/// `/signup`; on success it flashes and redirects to `/` (`HOME`) — both via
/// the `alert(href)` HTML/script response ([`signup::alert_html`]).
///
/// AUTH SEAM (T9.4): authorization is `validate_can_create_user`; the
/// middleware gates it. CSRF validation here is independent of that gate —
/// Python checks CSRF for every caller regardless of authorization.
pub async fn create_user_ui(
    State(state): State<AppState>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;

    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or("").trim());

    // `csrf.protect()`: Flask only populates `request.form` for
    // form-urlencoded/multipart bodies, so a non-form content type reaches
    // this check with no form field to find — same as an actually-missing
    // token. The CSRF gate therefore runs unconditionally, before the
    // content-type branch below, exactly mirroring Python's check order.
    let form = if content_type == Some("application/x-www-form-urlencoded") {
        parse_form(&body)
    } else {
        Vec::new()
    };
    // `AppState::csrf_secret()` is always `Some` here: this route is only
    // mounted when the basic-auth app (and therefore the secret) is enabled.
    let csrf_secret = state
        .csrf_secret()
        .ok_or_else(|| MlflowError::internal_error("CSRF secret is not configured"))?;
    let form_csrf_token = form_get(&form, super::signup::CSRF_FIELD_NAME);
    let csrf_token =
        super::signup::csrf_token_from_request(&parts.headers, form_csrf_token.as_deref());
    if let Err(csrf_err) =
        super::signup::validate_csrf_request(csrf_secret, &parts.headers, csrf_token.as_deref())
    {
        return Ok(csrf_err.into_response());
    }

    if content_type != Some("application/x-www-form-urlencoded") {
        return Ok(text_response(
            StatusCode::BAD_REQUEST,
            "Invalid content type. Must be application/x-www-form-urlencoded",
        ));
    }

    let username = form_get(&form, "username").unwrap_or_default();
    let password = form_get(&form, "password").unwrap_or_default();
    if username.is_empty() || password.is_empty() {
        return Ok(text_response(
            StatusCode::BAD_REQUEST,
            "Username and password cannot be empty.",
        ));
    }

    // `if store.has_user(username): flash(...); return alert(href=SIGNUP)`.
    if store.has_user(&username).await? {
        return Ok(text_response(
            StatusCode::OK,
            &super::signup::alert_html(
                &format!("Username has already been taken: {username}"),
                super::signup::SIGNUP_PATH,
            ),
        ));
    }

    store.create_user(&username, &password, false).await?;
    Ok(text_response(
        StatusCode::OK,
        &super::signup::alert_html(
            &format!("Successfully signed up user: {username}"),
            super::signup::HOME_PATH,
        ),
    ))
}

/// `get_user` (`auth/__init__.py:3836`): `GET /mlflow/users/get`.
///
/// AUTH SEAM (T9.4): authorization is `validate_can_read_user` (admin / self /
/// workspace-admin-of-target, `__init__.py:1654`).
pub async fn get_user(
    State(state): State<AppState>,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;
    let args = query_args(&parts);
    let username = get_request_param(&args, "username")?;
    let user = store.get_user(&username).await?;
    user_response(&user)
}

/// `get_current_user` (`auth/__init__.py:3873`): `GET /mlflow/users/current`.
///
/// Returns minimal identity for the authenticated caller plus `is_basic_auth`.
/// Response shape: `{"user": {id, username, is_admin}, "is_basic_auth": bool}`.
///
/// AUTH SEAM (T9.4): the BEFORE_REQUEST validator is `lambda: True`
/// (`__init__.py:2654`) — no authorization gate — but the request must still be
/// *authenticated* (the middleware's 401 challenge). Here we authenticate the
/// caller ourselves; an unauthenticated request yields 401 to mirror Python's
/// `resp.status_code == 401` behavior (`test_client.py:150-151`).
pub async fn get_current_user(
    State(state): State<AppState>,
    parts: Parts,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;
    let Some(username) = authenticated_username(store, &parts).await else {
        // AUTH SEAM (T9.4): the middleware owns this challenge globally; until it
        // lands, `/users/current` (whose validator is `lambda: True`) still needs
        // a caller identity, so we emit the same 401 challenge here.
        return Ok(unauthenticated_response());
    };
    let user = store.get_user(&username).await?;
    // `is_basic_auth = auth_config.authorization_function == DEFAULT` — always
    // true in v1 (custom authorization functions are out of scope, D9).
    let payload = json!({
        "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin},
        "is_basic_auth": true,
    });
    Ok(json_response(StatusCode::OK, &payload))
}

/// `list_users` (`auth/__init__.py:3843`): `GET /mlflow/users/list`.
///
/// Response: `{"users": [{id, username, is_admin, roles: [...]}, ...]}`.
///
/// AUTH SEAM (T9.4): authorization is `validate_can_list_users` (super-admin
/// only, `__init__.py:1663`). The per-user role *visibility* scoping in Python
/// (`__init__.py:3850-3869`: super-admin/self see all roles; workspace admins
/// see only roles in workspaces they administer) is an authorization concern
/// that depends on the caller identity + workspace-admin lookup — deferred to
/// T9.3/T9.4 with the roles layer. v1 returns every role for each user (the
/// super-admin view), which is what the AC (super-admin caller) exercises.
pub async fn list_users(State(state): State<AppState>) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;
    let users = store.list_users().await?;
    let mut rows = Vec::with_capacity(users.len());
    for user in users {
        // AUTH SEAM (T9.4): role-visibility scoping folds in with T9.3's roles
        // API; the super-admin view (all roles) is returned here.
        let roles = store.get_user_roles(&user.username).await?;
        let role_json: Vec<Value> = roles.iter().map(role_to_json).collect();
        rows.push(json!({
            "id": user.id,
            "username": user.username,
            "is_admin": user.is_admin,
            "roles": role_json,
        }));
    }
    Ok(json_response(StatusCode::OK, &json!({"users": rows})))
}

/// `update_user_password` (`auth/__init__.py:3962`): `PATCH
/// /mlflow/users/update-password`.
///
/// Handler-level self-service rules (ported verbatim): when the caller is
/// changing *their own* password, `current_password` is required, must match,
/// and the new password must differ. Admin paths (caller != target) skip all
/// three checks. On success returns 200 with an empty JSON object `{}`.
///
/// AUTH SEAM (T9.4): authorization is `validate_can_update_user_password`
/// (admin / self, `__init__.py:1679`); the middleware gates whether a non-admin
/// may target another user at all. The self checks below are the *handler's*
/// own — they run regardless of the middleware.
pub async fn update_user_password(
    State(state): State<AppState>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;
    let args = json_body(&body)?;
    let username = get_request_param(&args, "username")?;
    let password = get_request_param(&args, "password")?;

    // `sender_username == username` → self-service branch.
    let sender = authenticated_username(store, &parts).await;
    if sender.as_deref() == Some(username.as_str()) {
        let current_password = args
            .get("current_password")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let Some(current_password) = current_password else {
            return Err(MlflowError::invalid_parameter_value(
                "Current password is required when changing your own password.",
            ));
        };
        if !store.authenticate_user(&username, current_password).await {
            return Err(MlflowError::invalid_parameter_value(
                "Current password does not match.",
            ));
        }
        if password == current_password {
            return Err(MlflowError::invalid_parameter_value(
                "New password must differ from the current password.",
            ));
        }
    }

    store.update_user(&username, Some(&password), None).await?;
    // AUTH SEAM (T9.4): `_invalidate_user_auth_cache(username)` — the credential
    // cache lives in the middleware layer; nothing to invalidate until it lands.
    Ok(json_response(StatusCode::OK, &json!({})))
}

/// `update_user_admin` (`auth/__init__.py:3995`): `PATCH
/// /mlflow/users/update-admin`.
///
/// AUTH SEAM (T9.4): authorization is `validate_can_update_user_admin`
/// (super-admin only, `__init__.py:1683`).
pub async fn update_user_admin(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;
    let args = json_body(&body)?;
    let username = get_request_param(&args, "username")?;
    // Python passes the raw JSON value of `is_admin` to `store.update_user`,
    // which assigns it directly to `user.is_admin`. The client always sends a
    // JSON bool (`client.py`), so require a bool here.
    let is_admin = get_bool_request_param(&args, "is_admin")?;

    store.update_user(&username, None, Some(is_admin)).await?;
    Ok(json_response(StatusCode::OK, &json!({})))
}

/// `delete_user` (`auth/__init__.py:4004`): `DELETE /mlflow/users/delete`.
///
/// Handler-level rule (ported verbatim): a caller cannot delete their own
/// account (`BAD_REQUEST`), even as admin. On success returns 200 with `{}`.
///
/// AUTH SEAM (T9.4): authorization is `validate_can_delete_user` (super-admin
/// only, `__init__.py:1688`). The cannot-delete-self guard is the *handler's*
/// own and runs regardless.
pub async fn delete_user(
    State(state): State<AppState>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let store = auth_store(&state)?;
    // `_get_request_param` for DELETE: JSON body when the request is JSON, else
    // query args.
    let username = if is_json(&parts) {
        let args = json_body(&body)?;
        get_request_param(&args, "username")?
    } else {
        let args = query_args(&parts);
        get_request_param(&args, "username")?
    };

    let sender = authenticated_username(store, &parts).await;
    if sender.as_deref() == Some(username.as_str()) {
        return Err(MlflowError::new(
            "Users cannot delete their own account. Ask another admin to delete this user instead.",
            ErrorCode::BadRequest,
        ));
    }

    store.delete_user(&username).await?;
    // AUTH SEAM (T9.4): `_invalidate_user_auth_cache(username)` — see above.
    Ok(json_response(StatusCode::OK, &json!({})))
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// The auth store, or `INTERNAL_ERROR` if the route was somehow reached without
/// auth enabled (never happens: the routes are only mounted when it is).
fn auth_store(state: &AppState) -> Result<&AuthStore, MlflowError> {
    state
        .auth_store()
        .ok_or_else(|| MlflowError::internal_error("auth is not enabled on this server"))
}

/// `request.is_json`: the Content-Type is `application/json` (params ignored).
fn is_json(parts: &Parts) -> bool {
    parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or("").trim())
        == Some("application/json")
}

/// Parse the request body as a JSON object. Mirrors `_get_request_param`'s body
/// coercion: a `null`/empty/non-object body becomes `{}` (so a missing param
/// surfaces as the standard 400, not a 500).
fn json_body(body: &Bytes) -> Result<serde_json::Map<String, Value>, MlflowError> {
    let text = std::str::from_utf8(body).map_err(|_| {
        MlflowError::invalid_parameter_value("Request body is not valid UTF-8.".to_string())
    })?;
    if text.trim().is_empty() {
        return Ok(serde_json::Map::new());
    }
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(map)) => Ok(map),
        // `null`, arrays, scalars, and malformed JSON all coerce to `{}`, exactly
        // like Python's `body if isinstance(body, dict) else {}`.
        _ => Ok(serde_json::Map::new()),
    }
}

/// The GET query string parsed into a JSON object (last value wins per key).
fn query_args(parts: &Parts) -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    if let Some(query) = parts.uri.query() {
        for (k, v) in crate::proto_http::parse_query_pairs(query) {
            map.insert(k, Value::String(v));
        }
    }
    map
}

/// `_get_request_param(param)` (`auth/__init__.py:483`): return the string value
/// or raise `INVALID_PARAMETER_VALUE` with Python's verbatim message. JSON
/// values are stringified the way Flask's `args[param]` would surface them for
/// the string-typed user params (a JSON string stays a string).
fn get_request_param(
    args: &serde_json::Map<String, Value>,
    param: &str,
) -> Result<String, MlflowError> {
    match args.get(param) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Ok(other.to_string()),
        None => Err(missing_param(param)),
    }
}

/// `_get_request_param` for a field the client sends as a JSON bool
/// (`is_admin`). A missing key is the standard missing-param 400; a present but
/// non-bool value is an `INVALID_PARAMETER_VALUE`.
fn get_bool_request_param(
    args: &serde_json::Map<String, Value>,
    param: &str,
) -> Result<bool, MlflowError> {
    match args.get(param) {
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => Err(MlflowError::invalid_parameter_value(format!(
            "Parameter '{param}' must be a boolean. Got: {other}"
        ))),
        None => Err(missing_param(param)),
    }
}

/// `_get_request_param`'s missing-value error (`auth/__init__.py:508-512`).
fn missing_param(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. \
         See the API docs for more information about request parameters."
    ))
}

/// Parse an `application/x-www-form-urlencoded` body into `(key, value)` pairs.
fn parse_form(body: &Bytes) -> Vec<(String, String)> {
    match std::str::from_utf8(body) {
        Ok(text) => crate::proto_http::parse_query_pairs(text),
        Err(_) => Vec::new(),
    }
}

fn form_get(form: &[(String, String)], key: &str) -> Option<String> {
    form.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// `jsonify({"user": user.to_json()})` — the `{id, username, is_admin}` shape.
fn user_response(user: &User) -> Result<Response, MlflowError> {
    let payload = json!({
        "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin},
    });
    Ok(json_response(StatusCode::OK, &payload))
}

/// `Role.to_json()` (`auth/entities.py:358-...`): `{id, name, workspace,
/// description, permissions: [...]}`.
fn role_to_json(role: &mlflow_auth::Role) -> Value {
    let permissions: Vec<Value> = role
        .permissions
        .iter()
        .map(|p| {
            json!({
                "id": p.id,
                "role_id": p.role_id,
                "resource_type": p.resource_type,
                "resource_pattern": p.resource_pattern,
                "permission": p.permission,
            })
        })
        .collect();
    json!({
        "id": role.id,
        "name": role.name,
        "workspace": role.workspace,
        "description": role.description,
        "permissions": permissions,
    })
}

/// A `Content-Type: application/json` response with a compact JSON body.
fn json_response(status: StatusCode, value: &Value) -> Response {
    let body = serde_json::to_string(value).expect("JSON value serializes");
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .expect("valid response")
}

/// A plain-text response (matching Flask's `make_response(msg, code)` default
/// `text/html; charset=utf-8` mimetype).
fn text_response(status: StatusCode, msg: &str) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(axum::body::Body::from(msg.to_string()))
        .expect("valid response")
}

/// `make_basic_auth_response()` (`auth/__init__.py:466-474`): 401 with the
/// `WWW-Authenticate: Basic realm="mlflow"` challenge and Python's verbatim
/// "You are not authenticated…" body. See the AUTH SEAM note on
/// [`get_current_user`].
fn unauthenticated_response() -> Response {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::WWW_AUTHENTICATE, "Basic realm=\"mlflow\"")
        .body(axum::body::Body::from(
            "You are not authenticated. Please see \
             https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow \
             on how to authenticate.",
        ))
        .expect("valid response")
}
