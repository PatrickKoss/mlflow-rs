//! Per-user permission grants via synthetic `__user_<id>__` roles (plan T9.3),
//! mirroring `mlflow/server/auth/sqlalchemy_store.py`.
//!
//! The auth model expresses per-user grants ("user X has permission P on
//! resource R") on top of role-based storage by maintaining a hidden per-user
//! role `__user_<user_id>__` in each workspace and attaching the user's grants
//! to it (`sqlalchemy_store.py:243-355`). This module ports:
//!
//! * `_synthetic_user_role_name` (`:262`) / `_get_or_create_synthetic_user_role`
//!   (`:285`) — the SAVEPOINT-safe get-or-create for the race where two
//!   concurrent mutations both try to create the same synthetic role/assignment.
//! * `grant_user_permission` (`:377`) — upsert one grant on the synthetic role.
//! * `grant_user_resource_permission` (`:426`) — insert, `RESOURCE_ALREADY_EXISTS`
//!   on a duplicate (the REST grant contract).
//! * `revoke_user_resource_permission` (`:475`) — delete, `RESOURCE_DOES_NOT_EXIST`
//!   when absent (the REST revoke contract).
//! * `_scorer_pattern` (`:1130`) / scorer-name url-encoding — the compound
//!   `<experiment_id>/<url_quote(name)>` resource_pattern key.
//!
//! ## SAVEPOINT strategy
//!
//! Python wraps the racing INSERT in `session.begin_nested()` (a SAVEPOINT) and
//! recovers by re-querying on `IntegrityError`. The `mlflow-store` `Tx` here
//! doesn't expose nested savepoints, so we get the same **observable** behavior
//! with the dialect-agnostic idiom: `INSERT ... ON CONFLICT DO NOTHING`
//! (`Dialect::upsert` with no update columns), then unconditionally re-select
//! the winner's row. A concurrent create can't error and can't duplicate.

use mlflow_error::MlflowError;
use mlflow_store::dialect::UpsertSpec;

use crate::dbutil::{DbExt, RowLike, Val};
use crate::permissions;
use crate::roles::SYNTHETIC_ROLE_PREFIX;
use crate::schema::{ROLES, ROLE_PERMISSIONS, USER_ROLE_ASSIGNMENTS};
use crate::store::AuthStore;

/// The default workspace name (`mlflow/utils/workspace_utils.py:7`). Rust's
/// single-tenant auth path resolves the active workspace to this when
/// workspaces are disabled (`_get_active_workspace_name`,
/// `sqlalchemy_store.py:99`).
pub const DEFAULT_WORKSPACE_NAME: &str = "default";

impl AuthStore {
    /// `_synthetic_user_role_name` (`sqlalchemy_store.py:262`).
    pub fn synthetic_user_role_name(user_id: i64) -> String {
        format!("{SYNTHETIC_ROLE_PREFIX}{user_id}__")
    }

    /// `grant_user_permission` (`sqlalchemy_store.py:377`): upsert a grant on
    /// `(resource_type, resource_pattern)` for `username` via their synthetic
    /// role in `workspace`. Existing grants are overwritten (no error).
    pub async fn grant_user_permission(
        &self,
        username: &str,
        resource_type: &str,
        resource_pattern: &str,
        permission: &str,
        workspace: &str,
    ) -> Result<(), MlflowError> {
        permissions::validate_permission_for_resource_type(permission, resource_type)?;
        let user = self.get_user(username).await?;
        let role_id = self
            .get_or_create_synthetic_user_role(user.id, workspace)
            .await?;
        self.upsert_grant(role_id, resource_type, resource_pattern, permission)
            .await
    }

    /// `grant_user_resource_permission` (`sqlalchemy_store.py:426`): insert one
    /// grant; `RESOURCE_ALREADY_EXISTS` if the row already exists. Rejects the
    /// `workspace` resource type (`_reject_workspace_resource_type`, `:416`).
    pub async fn grant_user_resource_permission(
        &self,
        username: &str,
        resource_type: &str,
        resource_pattern: &str,
        permission: &str,
        workspace: &str,
    ) -> Result<(), MlflowError> {
        reject_workspace_resource_type(resource_type)?;
        permissions::validate_permission_for_resource_type(permission, resource_type)?;
        let duplicate = format!(
            "Permission for user={username} on \
             resource_type={resource_type}, resource_id={resource_pattern} already exists."
        );
        let user = self.get_user(username).await?;
        let role_id = self
            .get_or_create_synthetic_user_role(user.id, workspace)
            .await?;
        if self
            .find_grant(role_id, resource_type, resource_pattern)
            .await?
            .is_some()
        {
            return Err(MlflowError::resource_already_exists(duplicate));
        }
        self.insert_grant(
            role_id,
            resource_type,
            resource_pattern,
            permission,
            &duplicate,
        )
        .await
    }

    /// `revoke_user_resource_permission` (`sqlalchemy_store.py:475`): delete one
    /// grant; `RESOURCE_DOES_NOT_EXIST` if no row matches. Rejects the
    /// `workspace` resource type.
    pub async fn revoke_user_resource_permission(
        &self,
        username: &str,
        resource_type: &str,
        resource_pattern: &str,
        workspace: &str,
    ) -> Result<(), MlflowError> {
        reject_workspace_resource_type(resource_type)?;
        permissions::validate_resource_type(resource_type)?;
        let not_found = format!(
            "Permission for user={username} on \
             resource_type={resource_type}, resource_id={resource_pattern} not found."
        );
        let user = self.get_user(username).await?;
        let role_id = match self.find_synthetic_user_role(user.id, workspace).await? {
            Some(id) => id,
            None => return Err(MlflowError::resource_does_not_exist(not_found)),
        };
        let grant = self
            .find_grant(role_id, resource_type, resource_pattern)
            .await?;
        match grant {
            None => Err(MlflowError::resource_does_not_exist(not_found)),
            Some(grant_id) => {
                let dialect = self.db().writer().dialect();
                let sql = format!(
                    "DELETE FROM {ROLE_PERMISSIONS} WHERE id = {}",
                    dialect.placeholder(1)
                );
                self.db()
                    .writer()
                    .exec(&sql, &[Val::Int(grant_id)])
                    .await
                    .map_err(internal)?;
                Ok(())
            }
        }
    }

    /// `_scorer_pattern` (`sqlalchemy_store.py:1130`): the compound
    /// `<experiment_id>/<url_quote(scorer_name)>` resource_pattern for a scorer
    /// grant. The name is percent-encoded with `safe=''` so a `/` in the name
    /// cannot be confused with the delimiter.
    pub fn scorer_pattern(experiment_id: &str, scorer_name: &str) -> String {
        format!("{experiment_id}/{}", url_quote(scorer_name))
    }

    /// Inverse of [`AuthStore::scorer_pattern`] (`list_scorer_permissions`,
    /// `sqlalchemy_store.py:1252`): split on the first `/`, url-decode the name.
    pub fn parse_scorer_pattern(pattern: &str) -> (String, String) {
        match pattern.split_once('/') {
            Some((exp_id, encoded)) => (exp_id.to_string(), url_unquote(encoded)),
            None => (pattern.to_string(), String::new()),
        }
    }

    // ---- Synthetic-role plumbing ----

    /// `_get_or_create_synthetic_user_role` (`sqlalchemy_store.py:285`): return
    /// the id of the user's `__user_<id>__` role in `workspace`, creating it
    /// (and the user->role assignment) if missing. Race-proof via
    /// `INSERT ... ON CONFLICT DO NOTHING` + re-select (see module docs).
    ///
    /// Also ports the defense-in-depth check (`:315-337`): if a role with the
    /// reserved synthetic name already exists but is assigned to a *different*
    /// user, refuse rather than leak grants across accounts.
    pub(crate) async fn get_or_create_synthetic_user_role(
        &self,
        user_id: i64,
        workspace: &str,
    ) -> Result<i64, MlflowError> {
        let name = Self::synthetic_user_role_name(user_id);
        let dialect = self.db().writer().dialect();

        // Race-proof role create: ON CONFLICT (workspace, name) DO NOTHING,
        // then re-select the winner.
        let role_upsert = dialect.upsert(&UpsertSpec {
            table: ROLES,
            columns: &["name", "workspace", "description"],
            pk_columns: &["workspace", "name"],
            update_columns: &[],
            json_columns: &[],
        });
        self.db()
            .writer()
            .exec(
                &role_upsert,
                &[
                    Val::Text(name.clone()),
                    Val::Text(workspace.to_string()),
                    Val::Null,
                ],
            )
            .await
            .map_err(internal)?;

        let role_id = self
            .find_synthetic_user_role(user_id, workspace)
            .await?
            .ok_or_else(|| internal_msg("synthetic role vanished after upsert"))?;

        // Defense-in-depth: a pre-existing role with this reserved name that is
        // assigned to another user would leak grants — refuse (`:315-337`).
        if let Some(other) = self.other_assignee(role_id, user_id).await? {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Role {name:?} in workspace {workspace:?} collides with the \
                 reserved '__user_<id>__' synthetic namespace but is already \
                 assigned to user_id={other}. Rename or delete the conflicting \
                 role before granting per-user permissions."
            )));
        }

        // Race-proof user->role assignment: ON CONFLICT (user_id, role_id) DO NOTHING.
        let assignment_upsert = dialect.upsert(&UpsertSpec {
            table: USER_ROLE_ASSIGNMENTS,
            columns: &["user_id", "role_id"],
            pk_columns: &["user_id", "role_id"],
            update_columns: &[],
            json_columns: &[],
        });
        self.db()
            .writer()
            .exec(&assignment_upsert, &[Val::Int(user_id), Val::Int(role_id)])
            .await
            .map_err(internal)?;

        Ok(role_id)
    }

    async fn find_synthetic_user_role(
        &self,
        user_id: i64,
        workspace: &str,
    ) -> Result<Option<i64>, MlflowError> {
        let name = Self::synthetic_user_role_name(user_id);
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id FROM {ROLES} WHERE workspace = {} AND name = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db()
            .reader()
            .fetch_optional(
                &sql,
                &[Val::Text(workspace.to_string()), Val::Text(name)],
                |row: &dyn RowLike| row.get_i64("id"),
            )
            .await
            .map_err(internal)
    }

    async fn other_assignee(&self, role_id: i64, user_id: i64) -> Result<Option<i64>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT user_id FROM {USER_ROLE_ASSIGNMENTS} \
             WHERE role_id = {} AND user_id != {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db()
            .reader()
            .fetch_optional(
                &sql,
                &[Val::Int(role_id), Val::Int(user_id)],
                |row: &dyn RowLike| row.get_i64("user_id"),
            )
            .await
            .map_err(internal)
    }

    async fn find_grant(
        &self,
        role_id: i64,
        resource_type: &str,
        resource_pattern: &str,
    ) -> Result<Option<i64>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id FROM {ROLE_PERMISSIONS} \
             WHERE role_id = {} AND resource_type = {} AND resource_pattern = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
        );
        self.db()
            .reader()
            .fetch_optional(
                &sql,
                &[
                    Val::Int(role_id),
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_pattern.to_string()),
                ],
                |row: &dyn RowLike| row.get_i64("id"),
            )
            .await
            .map_err(internal)
    }

    async fn upsert_grant(
        &self,
        role_id: i64,
        resource_type: &str,
        resource_pattern: &str,
        permission: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().writer().dialect();
        let sql = dialect.upsert(&UpsertSpec {
            table: ROLE_PERMISSIONS,
            columns: &["role_id", "resource_type", "resource_pattern", "permission"],
            pk_columns: &["role_id", "resource_type", "resource_pattern"],
            update_columns: &["permission"],
            json_columns: &[],
        });
        self.db()
            .writer()
            .exec(
                &sql,
                &[
                    Val::Int(role_id),
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_pattern.to_string()),
                    Val::Text(permission.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    async fn insert_grant(
        &self,
        role_id: i64,
        resource_type: &str,
        resource_pattern: &str,
        permission: &str,
        duplicate_message: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().writer().dialect();
        let ph = |i| dialect.placeholder(i);
        let sql = format!(
            "INSERT INTO {ROLE_PERMISSIONS} \
             (role_id, resource_type, resource_pattern, permission) \
             VALUES ({}, {}, {}, {})",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
        );
        let vals = [
            Val::Int(role_id),
            Val::Text(resource_type.to_string()),
            Val::Text(resource_pattern.to_string()),
            Val::Text(permission.to_string()),
        ];
        match self.db().writer().exec(&sql, &vals).await {
            Ok(_) => Ok(()),
            Err(e) if is_unique_violation(&e) => Err(MlflowError::resource_already_exists(
                duplicate_message.to_string(),
            )),
            Err(e) => Err(internal(e)),
        }
    }
}

/// `_reject_workspace_resource_type` (`sqlalchemy_store.py:416`).
fn reject_workspace_resource_type(resource_type: &str) -> Result<(), MlflowError> {
    if resource_type == permissions::RESOURCE_TYPE_WORKSPACE {
        return Err(MlflowError::invalid_parameter_value(
            "resource_type 'workspace' is not supported by the per-user permission \
             convenience APIs. Use set_workspace_permission / \
             delete_workspace_permission for workspace-wide grants.",
        ));
    }
    Ok(())
}

/// `urllib.parse.quote(s, safe='')`: percent-encode every byte except the
/// unreserved set `A-Z a-z 0-9 _ . - ~`. Matches Python's default `quote`
/// (with `safe=''` so `/` is also encoded).
fn url_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_upper(b >> 4));
            out.push(hex_upper(b & 0xf));
        }
    }
    out
}

/// `urllib.parse.unquote(s)`: decode `%XX` escapes back to UTF-8 bytes. Invalid
/// escapes are passed through literally, matching CPython's lenient decoder.
fn url_unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
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

fn hex_upper(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + (nibble - 10)) as char,
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("auth database error: {e}"))
}

fn internal_msg(msg: &str) -> MlflowError {
    MlflowError::internal_error(format!("auth database error: {msg}"))
}

fn is_unique_violation(err: &sqlx::Error) -> bool {
    let Some(db_err) = err.as_database_error() else {
        return false;
    };
    if let Some(code) = db_err.code() {
        if code == "23505" || code == "1062" || code == "23000" {
            return true;
        }
    }
    let msg = db_err.message().to_ascii_lowercase();
    msg.contains("unique") || msg.contains("duplicate")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_pattern_encodes_name() {
        assert_eq!(
            AuthStore::scorer_pattern("123", "my/scorer"),
            "123/my%2Fscorer"
        );
        assert_eq!(AuthStore::scorer_pattern("exp", "plain"), "exp/plain");
        assert_eq!(AuthStore::scorer_pattern("e1", "a b+c"), "e1/a%20b%2Bc");
    }

    #[test]
    fn scorer_pattern_round_trips() {
        for (exp, name) in [
            ("123", "my/scorer"),
            ("e1", "a b+c"),
            ("exp-9", "weird name/with/slashes"),
            ("0", "unicode-\u{00e9}"),
        ] {
            let pattern = AuthStore::scorer_pattern(exp, name);
            let (got_exp, got_name) = AuthStore::parse_scorer_pattern(&pattern);
            assert_eq!(got_exp, exp);
            assert_eq!(got_name, name);
        }
    }

    #[test]
    fn synthetic_role_name_matches_python() {
        assert_eq!(AuthStore::synthetic_user_role_name(1), "__user_1__");
        assert_eq!(AuthStore::synthetic_user_role_name(42), "__user_42__");
    }
}
