//! `mlflow-auth`: RBAC authentication and authorization — the DB layer (T9.1).
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§3.16, §5.3, Phase 9), this crate owns
//! the auth database: the four live RBAC tables (`users`, `roles`,
//! `role_permissions`, `user_role_assignments`), werkzeug-compatible password
//! hash verification/generation (shared `basic_auth.db` with the Python
//! server), read-replica routing, and admin bootstrap. It reuses
//! `mlflow-store`'s `Db`/`Dialect` connection infrastructure.
//!
//! This module implements the **T9.1 DB-layer foundation**:
//!
//! * [`hash`] — werkzeug `generate_password_hash`/`check_password_hash` parity:
//!   verify any `scrypt:...`/`pbkdf2:...` salted hash and generate hashes
//!   werkzeug accepts (default `scrypt:32768:8:1`, salt length 16), constant-
//!   time compare. Byte-matched to `werkzeug/security.py` (3.1.8).
//! * [`schema`] — the four live auth table names (columns documented inline);
//!   the legacy per-resource permission tables are dead at runtime.
//! * [`entities`] — owned `User`/`Role`/`RolePermission`/`UserRoleAssignment`
//!   (the `to_mlflow_entity()` shapes from `auth/entities.py`).
//! * [`db`] — [`db::AuthDb`]: Alembic head verification against the
//!   `alembic_version_auth` table (head `f1a2b3c4d5e6`), refusing a stale or
//!   uninitialized DB with a Python-matching "run `mlflow db upgrade`" message,
//!   plus read-replica routing (`read_database_uri`).
//! * [`store::AuthStore`] — user create/get/list/update, `authenticate_user`,
//!   and RBAC grant reads (`get_user_roles` / `list_role_permissions`).
//! * [`bootstrap`] — `create_admin_user` + the default-admin-password warning,
//!   reproducing `auth/__init__.py`'s wording.
//!
//! The **T9.3 roles + permissions surface** builds on that foundation:
//!
//! * [`permissions`] — the `READ < USE < EDIT < MANAGE` (+ `NO_PERMISSIONS`)
//!   levels, the eight resource types, and their validators (byte-faithful to
//!   `auth/permissions.py`).
//! * [`roles`] — role / role-permission / user-role-assignment CRUD plus the
//!   role-based permission resolver and workspace-admin helpers.
//! * [`user_grants`] — per-user grant/revoke via synthetic `__user_<id>__`
//!   roles (SAVEPOINT-safe get-or-create) and scorer pattern-key encoding.
//!
//! The tower auth middleware (authenticate -> validator dispatch) is T9.4.
//!
//! ## Reuse of `mlflow-store`
//!
//! Like `mlflow-registry`/`mlflow-webhooks`, this crate depends on
//! `mlflow-store` for its connection/dialect infrastructure
//! ([`mlflow_store::Db`], [`mlflow_store::Dialect`]). The crate-private query
//! helpers there are copied into [`dbutil`] pending consolidation.

mod dbutil;

pub mod bootstrap;
pub mod config;
pub mod credential_cache;
pub mod db;
pub mod entities;
pub mod hash;
pub mod permissions;
pub mod roles;
pub mod schema;
pub mod store;
pub mod user_grants;
pub mod workspace_cache;

pub use bootstrap::{
    create_admin_user, warn_if_default_admin_password, DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
};
pub use config::{AuthConfig, MLFLOW_AUTH_CONFIG_PATH_ENV};
pub use credential_cache::CredentialCache;
pub use db::{
    AuthDb, AuthDbError, AuthSchemaError, ALEMBIC_VERSION_AUTH_TABLE, EXPECTED_AUTH_ALEMBIC_HEAD,
};
pub use entities::{Role, RolePermission, User, UserRoleAssignment};
pub use hash::{check_password_hash, generate_password_hash, HashError};
pub use permissions::{Permission, EDIT, MANAGE, NO_PERMISSIONS, READ, USE, VALID_RESOURCE_TYPES};
pub use store::AuthStore;
pub use user_grants::DEFAULT_WORKSPACE_NAME;
pub use workspace_cache::ResourceWorkspaceCache;
