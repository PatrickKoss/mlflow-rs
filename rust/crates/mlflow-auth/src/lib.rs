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
//! The permission-resolution model (`READ < USE < EDIT < MANAGE`, synthetic
//! `__user_<id>__` roles) and the tower auth middleware are later Phase 9 tasks,
//! not this one.
//!
//! ## Reuse of `mlflow-store`
//!
//! Like `mlflow-registry`/`mlflow-webhooks`, this crate depends on
//! `mlflow-store` for its connection/dialect infrastructure
//! ([`mlflow_store::Db`], [`mlflow_store::Dialect`]). The crate-private query
//! helpers there are copied into [`dbutil`] pending consolidation.

mod dbutil;

pub mod bootstrap;
pub mod db;
pub mod entities;
pub mod hash;
pub mod schema;
pub mod store;

pub use bootstrap::{
    create_admin_user, warn_if_default_admin_password, DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
};
pub use db::{
    AuthDb, AuthDbError, AuthSchemaError, ALEMBIC_VERSION_AUTH_TABLE, EXPECTED_AUTH_ALEMBIC_HEAD,
};
pub use entities::{Role, RolePermission, User, UserRoleAssignment};
pub use hash::{check_password_hash, generate_password_hash, HashError};
pub use store::AuthStore;
