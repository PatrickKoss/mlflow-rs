//! The [`AuthStore`]: user CRUD + authentication + role/permission reads,
//! mirroring `mlflow/server/auth/sqlalchemy_store.py`.
//!
//! Scope (plan T9.1): the DB layer needed for the cross-language auth AC — Rust
//! authenticates users Python created and vice versa on a shared
//! `basic_auth.db`. That means werkzeug-compatible password verify/generate
//! (see [`crate::hash`]), user create/get/list/update, and reading the RBAC
//! grants for a user (roles -> role_permissions via user_role_assignments). The
//! full permission-resolution model and role/permission mutation surface are
//! later tasks in Phase 9; this store carries the pieces the DB-layer AC needs.
//!
//! ## Read routing
//!
//! Reads use [`AuthDb::reader`] (the replica when configured); writes use
//! [`AuthDb::writer`]. This mirrors Python's `read_only` session routing
//! (`sqlalchemy_store.py:136-181` reads vs `:144-241` `read_only=False` writes).
//!
//! ## Validation
//!
//! `create_user` validates username/password exactly as Python
//! (`mlflow/utils/validation.py:824-833`): a non-empty username, and a password
//! that is a string longer than 12 characters.

use mlflow_error::MlflowError;

use crate::db::AuthDb;
use crate::dbutil::{DbExt, RowLike, Val};
use crate::entities::{Role, RolePermission, User};
use crate::hash::{check_password_hash, generate_password_hash};
use crate::schema::{ROLES, ROLE_PERMISSIONS, USERS, USER_ROLE_ASSIGNMENTS};

/// The auth store over a connected [`AuthDb`].
#[derive(Debug, Clone)]
pub struct AuthStore {
    db: AuthDb,
}

impl AuthStore {
    /// Build a store over an already-connected/verified [`AuthDb`].
    pub fn new(db: AuthDb) -> Self {
        Self { db }
    }

    /// The underlying auth database.
    pub fn db(&self) -> &AuthDb {
        &self.db
    }

    /// `authenticate_user` (`sqlalchemy_store.py:136-142`): `True` iff the user
    /// exists and the password verifies against its stored werkzeug hash. A
    /// missing user is `False`, not an error.
    pub async fn authenticate_user(&self, username: &str, password: &str) -> bool {
        match self.get_user(username).await {
            Ok(user) => check_password_hash(&user.password_hash, password),
            Err(_) => false,
        }
    }

    /// `create_user` (`sqlalchemy_store.py:144-158`): validate, hash the
    /// password with werkzeug's default (scrypt), insert, return the entity.
    /// A duplicate username surfaces as `RESOURCE_ALREADY_EXISTS`.
    pub async fn create_user(
        &self,
        username: &str,
        password: &str,
        is_admin: bool,
    ) -> Result<User, MlflowError> {
        validate_username(username)?;
        validate_password(password)?;
        let pwhash = generate_password_hash(password)
            .map_err(|e| MlflowError::internal_error(format!("password hashing failed: {e}")))?;

        let dialect = self.db.writer().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db.writer().begin_tx().await.map_err(internal)?;

        let cols = "(username, password_hash, is_admin)";
        let vals = vec![
            Val::Text(username.to_string()),
            Val::Text(pwhash.clone()),
            Val::Bool(is_admin),
        ];

        let id = if dialect.supports_returning() {
            let sql = format!(
                "INSERT INTO {USERS} {cols} VALUES ({}, {}, {}) RETURNING id",
                ph(1),
                ph(2),
                ph(3),
            );
            match tx.insert_returning_id(&sql, &vals).await {
                Ok(id) => id,
                Err(e) => {
                    let _ = tx.commit().await; // release; ignore
                    return Err(map_insert_error(e, username));
                }
            }
        } else {
            let sql = format!(
                "INSERT INTO {USERS} {cols} VALUES ({}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
            );
            if let Err(e) = tx.exec(&sql, &vals).await {
                return Err(map_insert_error(e, username));
            }
            tx.last_insert_id().await.map_err(internal)?
        };
        tx.commit().await.map_err(internal)?;

        Ok(User {
            id,
            username: username.to_string(),
            password_hash: pwhash,
            is_admin,
        })
    }

    /// `has_user` (`sqlalchemy_store.py:175-177`).
    pub async fn has_user(&self, username: &str) -> Result<bool, MlflowError> {
        Ok(self.fetch_user(username).await?.is_some())
    }

    /// `get_user` (`sqlalchemy_store.py:179-181`), erroring
    /// `RESOURCE_DOES_NOT_EXIST` when absent (matching `_get_user`'s
    /// `NoResultFound` mapping, `:164-168`).
    pub async fn get_user(&self, username: &str) -> Result<User, MlflowError> {
        self.fetch_user(username)
            .await?
            .ok_or_else(|| user_not_found(username))
    }

    /// `list_users` (`sqlalchemy_store.py:183-186`).
    pub async fn list_users(&self) -> Result<Vec<User>, MlflowError> {
        let sql = format!("SELECT id, username, password_hash, is_admin FROM {USERS} ORDER BY id");
        self.db
            .reader()
            .fetch_all(&sql, &[], map_user_row)
            .await
            .map_err(internal)
    }

    /// `update_user` (`sqlalchemy_store.py:210-220`): set a new password hash
    /// and/or `is_admin`. Errors `RESOURCE_DOES_NOT_EXIST` if the user is gone.
    pub async fn update_user(
        &self,
        username: &str,
        password: Option<&str>,
        is_admin: Option<bool>,
    ) -> Result<User, MlflowError> {
        let existing = self.get_user(username).await?;

        let dialect = self.db.writer().dialect();
        let mut sets: Vec<String> = Vec::new();
        let mut vals: Vec<Val> = Vec::new();
        let mut idx = 1usize;

        let new_hash = match password {
            Some(p) => {
                let h = generate_password_hash(p).map_err(|e| {
                    MlflowError::internal_error(format!("password hashing failed: {e}"))
                })?;
                sets.push(format!("password_hash = {}", dialect.placeholder(idx)));
                idx += 1;
                vals.push(Val::Text(h.clone()));
                Some(h)
            }
            None => None,
        };
        if let Some(admin) = is_admin {
            sets.push(format!("is_admin = {}", dialect.placeholder(idx)));
            idx += 1;
            vals.push(Val::Bool(admin));
        }

        if !sets.is_empty() {
            let sql = format!(
                "UPDATE {USERS} SET {} WHERE username = {}",
                sets.join(", "),
                dialect.placeholder(idx),
            );
            vals.push(Val::Text(username.to_string()));
            self.db.writer().exec(&sql, &vals).await.map_err(internal)?;
        }

        Ok(User {
            id: existing.id,
            username: existing.username,
            password_hash: new_hash.unwrap_or(existing.password_hash),
            is_admin: is_admin.unwrap_or(existing.is_admin),
        })
    }

    /// The roles assigned to a user (via `user_role_assignments`), each with its
    /// `role_permissions`. This is the RBAC grant surface for a user — the
    /// "list permissions" half of the T9.1 AC. Ordered by role id then
    /// permission id for a stable result.
    pub async fn get_user_roles(&self, username: &str) -> Result<Vec<Role>, MlflowError> {
        let user = self.get_user(username).await?;
        let dialect = self.db.reader().dialect();

        let roles_sql = format!(
            "SELECT r.id AS id, r.name AS name, r.workspace AS workspace, \
                    r.description AS description \
             FROM {ROLES} r \
             JOIN {USER_ROLE_ASSIGNMENTS} a ON a.role_id = r.id \
             WHERE a.user_id = {} \
             ORDER BY r.id",
            dialect.placeholder(1),
        );
        let role_rows = self
            .db
            .reader()
            .fetch_all(&roles_sql, &[Val::Int(user.id)], |row: &dyn RowLike| {
                Ok((
                    row.get_i64("id")?,
                    row.get_string("name")?,
                    row.get_string("workspace")?,
                    row.get_opt_string("description")?,
                ))
            })
            .await
            .map_err(internal)?;

        let mut roles = Vec::with_capacity(role_rows.len());
        for (id, name, workspace, description) in role_rows {
            let permissions = self.list_role_permissions(id).await?;
            roles.push(Role {
                id,
                name,
                workspace,
                description,
                permissions,
            });
        }
        Ok(roles)
    }

    /// All `role_permissions` rows for a role, ordered by id.
    pub async fn list_role_permissions(
        &self,
        role_id: i64,
    ) -> Result<Vec<RolePermission>, MlflowError> {
        let dialect = self.db.reader().dialect();
        let sql = format!(
            "SELECT id, role_id, resource_type, resource_pattern, permission \
             FROM {ROLE_PERMISSIONS} WHERE role_id = {} ORDER BY id",
            dialect.placeholder(1),
        );
        self.db
            .reader()
            .fetch_all(&sql, &[Val::Int(role_id)], |row: &dyn RowLike| {
                Ok(RolePermission {
                    id: row.get_i64("id")?,
                    role_id: row.get_i64("role_id")?,
                    resource_type: row.get_string("resource_type")?,
                    resource_pattern: row.get_string("resource_pattern")?,
                    permission: row.get_string("permission")?,
                })
            })
            .await
            .map_err(internal)
    }

    async fn fetch_user(&self, username: &str) -> Result<Option<User>, MlflowError> {
        let dialect = self.db.reader().dialect();
        let sql = format!(
            "SELECT id, username, password_hash, is_admin FROM {USERS} WHERE username = {}",
            dialect.placeholder(1),
        );
        self.db
            .reader()
            .fetch_optional(&sql, &[Val::Text(username.to_string())], map_user_row)
            .await
            .map_err(internal)
    }
}

fn map_user_row(row: &dyn RowLike) -> Result<User, sqlx::Error> {
    Ok(User {
        id: row.get_i64("id")?,
        username: row.get_string("username")?,
        password_hash: row.get_string("password_hash")?,
        is_admin: row.get_bool("is_admin")?,
    })
}

/// `_validate_username` (`mlflow/utils/validation.py:824-826`).
fn validate_username(username: &str) -> Result<(), MlflowError> {
    if username.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Username cannot be empty.".to_string(),
        ));
    }
    Ok(())
}

/// `_validate_password` (`mlflow/utils/validation.py:829-833`): longer than 12
/// characters. Python measures Unicode code points (`len(str)`); we match with
/// `chars().count()`.
fn validate_password(password: &str) -> Result<(), MlflowError> {
    if password.chars().count() < 12 {
        return Err(MlflowError::invalid_parameter_value(
            "Password must be a string longer than 12 characters.".to_string(),
        ));
    }
    Ok(())
}

fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("auth database error: {e}"))
}

/// `_get_user`'s `NoResultFound` mapping (`sqlalchemy_store.py:164-168`).
fn user_not_found(username: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("User with username={username} not found"))
}

/// Map an INSERT error, translating a uniqueness violation to the Python
/// duplicate-user message (`sqlalchemy_store.py:154-158`).
fn map_insert_error(e: sqlx::Error, username: &str) -> MlflowError {
    if is_unique_violation(&e) {
        MlflowError::resource_already_exists(format!(
            "User (username={username}) already exists. Error: {e}"
        ))
    } else {
        internal(e)
    }
}

/// Detect a unique-constraint violation across the three backends.
fn is_unique_violation(err: &sqlx::Error) -> bool {
    let Some(db_err) = err.as_database_error() else {
        return false;
    };
    if let Some(code) = db_err.code() {
        // Postgres 23505 unique_violation; MySQL 1062 ER_DUP_ENTRY -> 23000.
        if code == "23505" || code == "1062" || code == "23000" {
            return true;
        }
    }
    let msg = db_err.message().to_ascii_lowercase();
    msg.contains("unique") || msg.contains("duplicate")
}
