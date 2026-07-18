//! RBAC role / permission / assignment REST endpoints (plan T9.3).
//!
//! Ports the hand-rolled-JSON RBAC handlers in `mlflow/server/auth/__init__.py`
//! (`create_role`..`get_user_permission`, `__init__.py:3001-4072`) and their
//! routes (`mlflow/server/auth/routes.py:51-76`, plus the per-user grant/revoke/
//! get at `:21-26`). Every route is registered on both `/api/3.0/...` and
//! `/ajax-api/3.0/...` (matching `handlers._get_paths`).
//!
//! ## What this task does NOT do
//!
//! Authorization (who may call these — super-admin vs workspace-admin vs the
//! caller themselves) is the T9.4 auth middleware's job. These handlers are the
//! endpoint surface + store wiring + Python-identical error shapes; they assume
//! the middleware has already authorized the request. `list_user_roles` also
//! uses the authenticated principal stamped by that middleware to apply
//! Python's workspace-admin response scoping (`__init__.py:3119-3128`).
//!
//! ## Self-contained wiring (avoids the contested `AppState.auth_store`)
//!
//! T9.2 owns `auth_api/mod.rs`, the `AppState.auth_store` field, and the lib.rs
//! registration block. To land in parallel without conflicts, the router here
//! is parameterized on [`mlflow_auth::AuthStore`] directly (its own axum state)
//! via [`register_role_routes`]. The orchestrator folds this into the shared
//! `AppState` when both tasks merge.
//!
//! ## Active workspace
//!
//! The per-user grant/revoke/get convenience APIs write to the caller's
//! synthetic role "in the active workspace" (`_get_active_workspace_name`,
//! `sqlalchemy_store.py:99`). With workspaces disabled that is the default
//! workspace; multi-tenant workspace routing is threaded through the middleware
//! (T9.4), so this layer uses [`mlflow_auth::DEFAULT_WORKSPACE_NAME`].

use axum::extract::{Extension, Query, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use mlflow_auth::{AuthStore, DEFAULT_WORKSPACE_NAME};
use mlflow_error::MlflowError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth_middleware::AuthContext;

// ---------------------------------------------------------------------------
// Route registration
// ---------------------------------------------------------------------------

// ---- role/permission routes (T9.3) ----

/// The 3.0 REST base paths for the RBAC routes (`routes.py:51-76`, `:21-26`),
/// registered under both the `/api` and `/ajax-api` prefixes.
const ROLE_ROUTE_BASES: &[(&str, RouteKind)] = &[
    ("/mlflow/roles/create", RouteKind::Post),
    ("/mlflow/roles/get", RouteKind::Get),
    ("/mlflow/roles/list", RouteKind::Get),
    ("/mlflow/roles/update", RouteKind::Patch),
    ("/mlflow/roles/delete", RouteKind::Delete),
    ("/mlflow/roles/permissions/add", RouteKind::Post),
    ("/mlflow/roles/permissions/remove", RouteKind::Delete),
    ("/mlflow/roles/permissions/list", RouteKind::Get),
    ("/mlflow/roles/permissions/update", RouteKind::Patch),
    ("/mlflow/roles/assign", RouteKind::Post),
    ("/mlflow/roles/unassign", RouteKind::Delete),
    ("/mlflow/users/roles/list", RouteKind::Get),
    ("/mlflow/roles/users/list", RouteKind::Get),
    ("/mlflow/users/permissions/grant", RouteKind::Post),
    ("/mlflow/users/permissions/revoke", RouteKind::Post),
    ("/mlflow/users/permissions/get", RouteKind::Get),
];

#[derive(Clone, Copy)]
enum RouteKind {
    Get,
    Post,
    Patch,
    Delete,
}

/// Register every RBAC route on `router` (both `/api/3.0` and `/ajax-api/3.0`
/// prefixes), with an [`AuthStore`] as the router state. Self-contained so it
/// composes into `build_app` without touching the contested `AppState` field.
pub fn register_role_routes(store: AuthStore) -> Router {
    let mut router: Router<AuthStore> = Router::new();
    for (base, _kind) in ROLE_ROUTE_BASES {
        for prefix in ["/api/3.0", "/ajax-api/3.0"] {
            let path = format!("{prefix}{base}");
            router = router.route(&path, method_router_for(base));
        }
    }
    router.with_state(store)
}

fn method_router_for(base: &str) -> axum::routing::MethodRouter<AuthStore> {
    match base {
        "/mlflow/roles/create" => post(create_role),
        "/mlflow/roles/get" => get(get_role),
        "/mlflow/roles/list" => get(list_roles),
        "/mlflow/roles/update" => patch(update_role),
        "/mlflow/roles/delete" => delete(delete_role),
        "/mlflow/roles/permissions/add" => post(add_role_permission),
        "/mlflow/roles/permissions/remove" => delete(remove_role_permission),
        "/mlflow/roles/permissions/list" => get(list_role_permissions),
        "/mlflow/roles/permissions/update" => patch(update_role_permission),
        "/mlflow/roles/assign" => post(assign_role),
        "/mlflow/roles/unassign" => delete(unassign_role),
        "/mlflow/users/roles/list" => get(list_user_roles),
        "/mlflow/roles/users/list" => get(list_role_users),
        "/mlflow/users/permissions/grant" => post(grant_user_permission),
        "/mlflow/users/permissions/revoke" => post(revoke_user_permission),
        "/mlflow/users/permissions/get" => get(get_user_permission),
        other => unreachable!("unregistered RBAC route {other}"),
    }
}

// ---------------------------------------------------------------------------
// Entity -> JSON (mirrors `entities.py` `to_json`)
// ---------------------------------------------------------------------------

fn role_json(role: &mlflow_auth::Role) -> Value {
    json!({
        "id": role.id,
        "name": role.name,
        "workspace": role.workspace,
        "description": role.description,
        "permissions": role.permissions.iter().map(role_permission_json).collect::<Vec<_>>(),
    })
}

fn role_permission_json(rp: &mlflow_auth::RolePermission) -> Value {
    json!({
        "id": rp.id,
        "role_id": rp.role_id,
        "resource_type": rp.resource_type,
        "resource_pattern": rp.resource_pattern,
        "permission": rp.permission,
    })
}

fn assignment_json(a: &mlflow_auth::UserRoleAssignment) -> Value {
    json!({ "id": a.id, "user_id": a.user_id, "role_id": a.role_id })
}

fn ok_json(value: Value) -> Response {
    Json(value).into_response()
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `create_role` (`__init__.py:3001`), `POST /mlflow/roles/create`.
async fn create_role(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let name = require_str(&body, "name")?;
    let workspace = require_str(&body, "workspace")?;
    if name.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Role name cannot be empty.",
        ));
    }
    if workspace.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Workspace cannot be empty.",
        ));
    }
    let description = optional_str(&body, "description")?;
    let role = store
        .create_role(&name, &workspace, description.as_deref())
        .await?;
    Ok(ok_json(json!({ "role": role_json(&role) })))
}

/// `get_role` (`__init__.py:3017`), `GET /mlflow/roles/get`.
async fn get_role(
    State(store): State<AuthStore>,
    Query(params): Query<RoleIdParam>,
) -> Result<Response, MlflowError> {
    let role_id = coerce_int("role_id", params.role_id)?;
    let role = store.get_role(role_id).await?;
    Ok(ok_json(json!({ "role": role_json(&role) })))
}

/// `list_roles` (`__init__.py:3024`), `GET /mlflow/roles/list`. Repeated
/// `workspace` scopes the listing; omitting it lists cross-workspace.
async fn list_roles(
    State(store): State<AuthStore>,
    Query(raw): Query<Vec<(String, String)>>,
) -> Result<Response, MlflowError> {
    let workspaces: Vec<String> = raw
        .into_iter()
        .filter(|(k, _)| k == "workspace")
        .map(|(_, v)| v)
        .collect();
    for w in &workspaces {
        if w.trim().is_empty() {
            return Err(MlflowError::invalid_parameter_value(
                "Parameter 'workspace' must be a non-empty string when provided.",
            ));
        }
    }
    let roles = if workspaces.is_empty() {
        store.list_roles(None).await?
    } else {
        store.list_roles(Some(&workspaces)).await?
    };
    Ok(ok_json(
        json!({ "roles": roles.iter().map(role_json).collect::<Vec<_>>() }),
    ))
}

/// `update_role` (`__init__.py:3038`), `PATCH /mlflow/roles/update`.
async fn update_role(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let role_id = coerce_int("role_id", require_value(&body, "role_id")?.clone())?;
    let name = optional_str(&body, "name")?;
    let description = optional_str(&body, "description")?;
    if name.is_none() && description.is_none() {
        return Err(MlflowError::invalid_parameter_value(
            "At least one of 'name' or 'description' must be provided to update a role.",
        ));
    }
    if let Some(n) = &name {
        if n.trim().is_empty() {
            return Err(MlflowError::invalid_parameter_value(
                "Role name cannot be empty.",
            ));
        }
    }
    let role = store
        .update_role(role_id, name.as_deref(), description.as_deref())
        .await?;
    Ok(ok_json(json!({ "role": role_json(&role) })))
}

/// `delete_role` (`__init__.py:3056`), `DELETE /mlflow/roles/delete`.
async fn delete_role(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let role_id = coerce_int("role_id", require_value(&body.0, "role_id")?.clone())?;
    store.delete_role(role_id).await?;
    Ok(ok_json(json!({})))
}

/// `add_role_permission` (`__init__.py:3063`), `POST /mlflow/roles/permissions/add`.
async fn add_role_permission(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let role_id = coerce_int("role_id", require_value(&body, "role_id")?.clone())?;
    let resource_type = require_str(&body, "resource_type")?;
    let resource_pattern = require_str(&body, "resource_pattern")?;
    let permission = require_str(&body, "permission")?;
    let rp = store
        .add_role_permission(role_id, &resource_type, &resource_pattern, &permission)
        .await?;
    Ok(ok_json(
        json!({ "role_permission": role_permission_json(&rp) }),
    ))
}

/// `remove_role_permission` (`__init__.py:3073`), `DELETE /mlflow/roles/permissions/remove`.
async fn remove_role_permission(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let id = coerce_int(
        "role_permission_id",
        require_value(&body.0, "role_permission_id")?.clone(),
    )?;
    store.remove_role_permission(id).await?;
    Ok(ok_json(json!({})))
}

/// `list_role_permissions` (`__init__.py:3080`), `GET /mlflow/roles/permissions/list`.
async fn list_role_permissions(
    State(store): State<AuthStore>,
    Query(params): Query<RoleIdParam>,
) -> Result<Response, MlflowError> {
    let role_id = coerce_int("role_id", params.role_id)?;
    let perms = store.list_permissions_of_role(role_id).await?;
    Ok(ok_json(json!({
        "role_permissions": perms.iter().map(role_permission_json).collect::<Vec<_>>(),
    })))
}

/// `update_role_permission` (`__init__.py:3087`), `PATCH /mlflow/roles/permissions/update`.
async fn update_role_permission(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let id = coerce_int(
        "role_permission_id",
        require_value(&body, "role_permission_id")?.clone(),
    )?;
    let permission = require_str(&body, "permission")?;
    let rp = store.update_role_permission(id, &permission).await?;
    Ok(ok_json(
        json!({ "role_permission": role_permission_json(&rp) }),
    ))
}

/// `assign_role` (`__init__.py:3095`), `POST /mlflow/roles/assign`.
async fn assign_role(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let username = require_str(&body, "username")?;
    let role_id = coerce_int("role_id", require_value(&body, "role_id")?.clone())?;
    let user = store.get_user(&username).await?;
    let assignment = store.assign_role_to_user(user.id, role_id).await?;
    Ok(ok_json(
        json!({ "assignment": assignment_json(&assignment) }),
    ))
}

/// `unassign_role` (`__init__.py:3104`), `DELETE /mlflow/roles/unassign`.
async fn unassign_role(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let username = require_str(&body, "username")?;
    let role_id = coerce_int("role_id", require_value(&body, "role_id")?.clone())?;
    let user = store.get_user(&username).await?;
    store.unassign_role_from_user(user.id, role_id).await?;
    Ok(ok_json(json!({})))
}

/// `list_user_roles` (`__init__.py:3113`), `GET /mlflow/users/roles/list`.
/// Self/super-admin callers see all roles; workspace admins see only roles in
/// workspaces they administer.
async fn list_user_roles(
    State(store): State<AuthStore>,
    auth: Option<Extension<AuthContext>>,
    Query(params): Query<UsernameParam>,
) -> Result<Response, MlflowError> {
    let user = store.get_user(&params.username).await?;
    let mut roles = store.list_user_roles(user.id).await?;
    if let Some(Extension(auth)) = auth {
        if !auth.is_admin && auth.username != params.username {
            let requester = store.get_user(&auth.username).await?;
            let admin_workspaces = store.list_workspace_admin_workspaces(requester.id).await?;
            roles.retain(|role| admin_workspaces.contains(&role.workspace));
        }
    }
    Ok(ok_json(
        json!({ "roles": roles.iter().map(role_json).collect::<Vec<_>>() }),
    ))
}

/// `list_role_users` (`__init__.py:3133`), `GET /mlflow/roles/users/list`.
async fn list_role_users(
    State(store): State<AuthStore>,
    Query(params): Query<RoleIdParam>,
) -> Result<Response, MlflowError> {
    let role_id = coerce_int("role_id", params.role_id)?;
    let assignments = store.list_role_users(role_id).await?;
    Ok(ok_json(json!({
        "assignments": assignments.iter().map(assignment_json).collect::<Vec<_>>(),
    })))
}

/// `grant_user_permission` (`__init__.py:4037`), `POST /mlflow/users/permissions/grant`.
async fn grant_user_permission(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let username = require_str(&body, "username")?;
    let resource_type = require_str(&body, "resource_type")?;
    let resource_id = require_str(&body, "resource_id")?;
    let permission = require_str(&body, "permission")?;
    store.get_user(&username).await?;
    store
        .grant_user_resource_permission(
            &username,
            &resource_type,
            &resource_id,
            &permission,
            DEFAULT_WORKSPACE_NAME,
        )
        .await?;
    Ok(ok_json(json!({})))
}

/// `revoke_user_permission` (`__init__.py:4048`), `POST /mlflow/users/permissions/revoke`.
async fn revoke_user_permission(
    State(store): State<AuthStore>,
    body: JsonBody,
) -> Result<Response, MlflowError> {
    let body = body.0;
    let username = require_str(&body, "username")?;
    let resource_type = require_str(&body, "resource_type")?;
    let resource_id = require_str(&body, "resource_id")?;
    store.get_user(&username).await?;
    store
        .revoke_user_resource_permission(
            &username,
            &resource_type,
            &resource_id,
            DEFAULT_WORKSPACE_NAME,
        )
        .await?;
    Ok(ok_json(json!({})))
}

/// `get_user_permission` (`__init__.py:4058`), `GET /mlflow/users/permissions/get`.
/// `allowed` mirrors `Permission.can_use`. Unknown *resources* return
/// `NO_PERMISSIONS` / `allowed=false` (deny-by-default); unknown *users* /
/// unsupported resource types raise.
async fn get_user_permission(
    State(store): State<AuthStore>,
    Query(params): Query<GetUserPermissionParams>,
) -> Result<Response, MlflowError> {
    let user = store.get_user(&params.username).await?;
    mlflow_auth::permissions::validate_resource_type(&params.resource_type)?;
    let permission = store
        .get_role_permission_for_resource(
            user.id,
            &params.resource_type,
            &params.resource_id,
            DEFAULT_WORKSPACE_NAME,
        )
        .await?
        .unwrap_or(&mlflow_auth::NO_PERMISSIONS);
    Ok(ok_json(json!({
        "allowed": permission.can_use,
        "permission": permission.name,
    })))
}

// ---------------------------------------------------------------------------
// Query-param structs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RoleIdParam {
    role_id: Value,
}

#[derive(Deserialize)]
struct UsernameParam {
    username: String,
}

#[derive(Deserialize)]
struct GetUserPermissionParams {
    username: String,
    resource_type: String,
    resource_id: String,
}

// ---------------------------------------------------------------------------
// JSON body extraction with Python-identical missing-param errors
// ---------------------------------------------------------------------------

/// A request JSON body coerced to an object (`{}` when absent/non-object), the
/// way `_get_request_param` treats POST/PATCH/DELETE bodies
/// (`__init__.py:486-494`).
struct JsonBody(serde_json::Map<String, Value>);

impl<S: Send + Sync> axum::extract::FromRequest<S> for JsonBody {
    type Rejection = Response;

    async fn from_request(
        req: axum::extract::Request,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        let bytes = axum::body::to_bytes(req.into_body(), usize::MAX)
            .await
            .map_err(|e| {
                MlflowError::invalid_parameter_value(format!("failed to read request body: {e}"))
                    .into_response()
            })?;
        let value: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        match value {
            Value::Object(map) => Ok(JsonBody(map)),
            _ => Ok(JsonBody(serde_json::Map::new())),
        }
    }
}

fn require_value<'a>(
    body: &'a serde_json::Map<String, Value>,
    param: &str,
) -> Result<&'a Value, MlflowError> {
    body.get(param).ok_or_else(|| missing_param(param))
}

fn require_str(body: &serde_json::Map<String, Value>, param: &str) -> Result<String, MlflowError> {
    match require_value(body, param)? {
        Value::String(s) => Ok(s.clone()),
        other => Ok(other.to_string()),
    }
}

/// An optional string field: absent -> `None`; a non-string, non-null value is
/// rejected (mirrors the `isinstance(..., str)` guards in `create_role` /
/// `update_role`).
fn optional_str(
    body: &serde_json::Map<String, Value>,
    param: &str,
) -> Result<Option<String>, MlflowError> {
    match body.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(MlflowError::invalid_parameter_value(format!(
            "Role {param} must be a string or null."
        ))),
    }
}

/// `_coerce_int_param` (`__init__.py:526`): accept a JSON number or a numeric
/// string; anything else is `INVALID_PARAMETER_VALUE`.
fn coerce_int(param: &str, value: Value) -> Result<i64, MlflowError> {
    match value {
        Value::Number(n) => n.as_i64().ok_or_else(|| bad_int(param, &n.to_string())),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| bad_int(param, &s)),
        other => Err(bad_int(param, &other.to_string())),
    }
}

fn missing_param(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. \
         See the API docs for more information about request parameters."
    ))
}

fn bad_int(param: &str, raw: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Invalid value {raw:?} for parameter '{param}': must be an integer."
    ))
}
