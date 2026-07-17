//! Table names for the four live auth RBAC tables (plan §5.3).
//!
//! Source of truth: `mlflow/server/auth/db/models.py:22-119`
//! (`SqlUser`, `SqlRole`, `SqlRolePermission`, `SqlUserRoleAssignment`).
//!
//! * `users` — `id` (int PK), `username` (varchar(255) unique), `password_hash`
//!   (varchar(255)), `is_admin` (bool, default false).
//! * `roles` — `id` (int PK), `name` (varchar(255) not null), `workspace`
//!   (varchar(63) not null), `description` (varchar(1024) null); unique
//!   `(workspace, name)`.
//! * `role_permissions` — `id` (int PK), `role_id` (int FK->roles.id),
//!   `resource_type` (varchar(64)), `resource_pattern` (varchar(255)),
//!   `permission` (varchar(255)); unique `(role_id, resource_type,
//!   resource_pattern)`.
//! * `user_role_assignments` — `id` (int PK), `user_id` (int FK->users.id),
//!   `role_id` (int FK->roles.id); unique `(user_id, role_id)`.
//!
//! The legacy per-resource permission tables (`experiment_permissions`,
//! `registered_model_permissions`, `scorer_permissions`, `gateway_*`,
//! `workspace_permissions`) still exist on disk for rollback but are dead at
//! runtime — Rust never reads or writes them (plan §5.3).

/// The `users` table.
pub const USERS: &str = "users";
/// The `roles` table.
pub const ROLES: &str = "roles";
/// The `role_permissions` table.
pub const ROLE_PERMISSIONS: &str = "role_permissions";
/// The `user_role_assignments` table.
pub const USER_ROLE_ASSIGNMENTS: &str = "user_role_assignments";

/// The four live auth tables owned by the Rust auth store (plan §5.3).
pub const AUTH_TABLES: &[&str] = &[USERS, ROLES, ROLE_PERMISSIONS, USER_ROLE_ASSIGNMENTS];
