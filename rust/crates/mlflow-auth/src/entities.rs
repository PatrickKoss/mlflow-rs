//! Owned auth entities mirroring `mlflow/server/auth/entities.py`
//! (`User`, `Role`, `RolePermission`, `UserRoleAssignment`).
//!
//! These are the `to_mlflow_entity()` outputs of the ORM models in
//! `mlflow/server/auth/db/models.py:42-119`. Field names track the Python
//! entity attributes; `id` mirrors the Python `id_` attr (Rust `id` is fine, no
//! keyword clash).

/// A user row (`SqlUser.to_mlflow_entity`, `models.py:42-48`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct User {
    pub id: i64,
    pub username: String,
    /// The stored werkzeug password hash. Never emitted to API responses (the
    /// handler layer redacts it), but carried here for `authenticate_user`.
    pub password_hash: String,
    pub is_admin: bool,
}

/// A role row with its permissions (`SqlRole.to_mlflow_entity`,
/// `models.py:67-74`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    pub id: i64,
    pub name: String,
    pub workspace: String,
    pub description: Option<String>,
    pub permissions: Vec<RolePermission>,
}

/// A single grant on a role (`SqlRolePermission.to_mlflow_entity`,
/// `models.py:92-99`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolePermission {
    pub id: i64,
    pub role_id: i64,
    pub resource_type: String,
    pub resource_pattern: String,
    pub permission: String,
}

/// A user-to-role assignment (`SqlUserRoleAssignment.to_mlflow_entity`,
/// `models.py:114-119`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserRoleAssignment {
    pub id: i64,
    pub user_id: i64,
    pub role_id: i64,
}
