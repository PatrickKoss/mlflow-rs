//! Role / role-permission / assignment CRUD and the role-based permission
//! resolver (plan T9.3), mirroring `mlflow/server/auth/sqlalchemy_store.py`.
//!
//! This is the "roles + permissions" store surface: the four public CRUD
//! groups plus the resolution helpers the HTTP layer and the (later) auth
//! middleware read through.
//!
//! ## Method map (Python source of truth)
//!
//! * `create_role` (`sqlalchemy_store.py:1668`), `get_role`/`_get_role`
//!   (`:1691`/`:1726`), `get_role_by_name` (`:1730`), `list_roles` (`:1734`),
//!   `update_role` (`:1755`), `delete_role` (`:1787`),
//!   `delete_roles_for_workspace` (`:1792`).
//! * `add_role_permission` (`:1813`), `get_role_permission`/`_get_role_permission`
//!   (`:1867`/`:1847`), `remove_role_permission` (`:1871`),
//!   `list_role_permissions` (`:1876`), `update_role_permission` (`:1884`).
//! * `assign_role_to_user` (`:1893`), `unassign_role_from_user` (`:1923`),
//!   `list_user_roles` (`:1942`), `list_user_roles_for_workspace` (`:1954`),
//!   `list_role_users` (`:1997`).
//! * `get_role_permission_for_resource` (`:2010`) — the resolver;
//!   `_workspace_admin_workspaces` (`:2056`), `is_workspace_admin` (`:2100`),
//!   `list_workspace_admin_workspaces` (`:2150`),
//!   `list_role_grants_for_user_in_workspace` (`:2111`).
//!
//! Legacy per-resource tables are never touched — everything is the four live
//! tables (§5.3). The `to_mlflow_entity` shapes come from `entities.rs`.

use std::collections::BTreeSet;

use mlflow_error::MlflowError;

use crate::dbutil::{DbExt, RowLike, Val};
use crate::entities::{Role, RolePermission, UserRoleAssignment};
use crate::permissions::{self, max_permission, permission_priority};
use crate::schema::{ROLES, ROLE_PERMISSIONS, USERS, USER_ROLE_ASSIGNMENTS};
use crate::store::AuthStore;

/// The `__user_` prefix reserved for synthetic per-user roles
/// (`sqlalchemy_store.py:257`).
pub(crate) const SYNTHETIC_ROLE_PREFIX: &str = "__user_";

impl AuthStore {
    // ---- Role CRUD ----

    /// `create_role` (`sqlalchemy_store.py:1668`): reject the reserved
    /// `__user_` prefix, insert, return the entity. A `(workspace, name)`
    /// collision surfaces as `RESOURCE_ALREADY_EXISTS`.
    pub async fn create_role(
        &self,
        name: &str,
        workspace: &str,
        description: Option<&str>,
    ) -> Result<Role, MlflowError> {
        reject_synthetic_role_name(name)?;
        let dialect = self.db().writer().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut tx = self.db().writer().begin_tx().await.map_err(internal)?;

        let vals = vec![
            Val::Text(name.to_string()),
            Val::Text(workspace.to_string()),
            description.map_or(Val::Null, |d| Val::Text(d.to_string())),
        ];
        let cols = "(name, workspace, description)";
        let insert_res = if dialect.supports_returning() {
            let sql = format!(
                "INSERT INTO {ROLES} {cols} VALUES ({}, {}, {}) RETURNING id",
                ph(1),
                ph(2),
                ph(3),
            );
            tx.insert_returning_id(&sql, &vals).await
        } else {
            let sql = format!(
                "INSERT INTO {ROLES} {cols} VALUES ({}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
            );
            match tx.exec(&sql, &vals).await {
                Ok(_) => tx.last_insert_id().await,
                Err(e) => Err(e),
            }
        };
        let id = match insert_res {
            Ok(id) => id,
            Err(e) => {
                let _ = tx.commit().await;
                return Err(map_role_conflict(e, name, workspace));
            }
        };
        tx.commit().await.map_err(internal)?;

        Ok(Role {
            id,
            name: name.to_string(),
            workspace: workspace.to_string(),
            description: description.map(str::to_string),
            permissions: Vec::new(),
        })
    }

    /// `get_role` (`sqlalchemy_store.py:1726`), erroring
    /// `RESOURCE_DOES_NOT_EXIST` when absent (`_get_role`, `:1691`).
    pub async fn get_role(&self, role_id: i64) -> Result<Role, MlflowError> {
        let base = self.fetch_role_by_id(role_id).await?;
        self.hydrate_role(base).await
    }

    /// `get_role_by_name` (`sqlalchemy_store.py:1730`).
    pub async fn get_role_by_name(&self, workspace: &str, name: &str) -> Result<Role, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id, name, workspace, description FROM {ROLES} \
             WHERE workspace = {} AND name = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        let base = self
            .db()
            .reader()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
                map_role_base,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Role with name={name} in workspace={workspace} not found"
                ))
            })?;
        self.hydrate_role(base).await
    }

    /// `list_roles` (`sqlalchemy_store.py:1734`). `None` lists every role
    /// (admin path); a `Some(&[])` empty slice returns no roles; a non-empty
    /// slice scopes to those workspaces.
    pub async fn list_roles(
        &self,
        workspaces: Option<&[String]>,
    ) -> Result<Vec<Role>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let bases = match workspaces {
            None => {
                let sql =
                    format!("SELECT id, name, workspace, description FROM {ROLES} ORDER BY id");
                self.db()
                    .reader()
                    .fetch_all(&sql, &[], map_role_base)
                    .await
                    .map_err(internal)?
            }
            Some([]) => return Ok(Vec::new()),
            Some(names) => {
                let placeholders: Vec<String> =
                    (1..=names.len()).map(|i| dialect.placeholder(i)).collect();
                let sql = format!(
                    "SELECT id, name, workspace, description FROM {ROLES} \
                     WHERE workspace IN ({}) ORDER BY id",
                    placeholders.join(", "),
                );
                let vals: Vec<Val> = names.iter().map(|w| Val::Text(w.clone())).collect();
                self.db()
                    .reader()
                    .fetch_all(&sql, &vals, map_role_base)
                    .await
                    .map_err(internal)?
            }
        };
        let mut roles = Vec::with_capacity(bases.len());
        for base in bases {
            roles.push(self.hydrate_role(base).await?);
        }
        Ok(roles)
    }

    /// `update_role` (`sqlalchemy_store.py:1755`): rename (rejecting the
    /// reserved prefix and `(workspace, name)` collisions) and/or set the
    /// description. `RESOURCE_DOES_NOT_EXIST` if the role is gone.
    pub async fn update_role(
        &self,
        role_id: i64,
        name: Option<&str>,
        description: Option<&str>,
    ) -> Result<Role, MlflowError> {
        if let Some(n) = name {
            reject_synthetic_role_name(n)?;
        }
        let existing = self.fetch_role_by_id(role_id).await?;
        let dialect = self.db().writer().dialect();

        if let Some(n) = name {
            let conflict_sql = format!(
                "SELECT id, name, workspace, description FROM {ROLES} \
                 WHERE workspace = {} AND name = {} AND id != {}",
                dialect.placeholder(1),
                dialect.placeholder(2),
                dialect.placeholder(3),
            );
            let clash = self
                .db()
                .reader()
                .fetch_optional(
                    &conflict_sql,
                    &[
                        Val::Text(existing.workspace.clone()),
                        Val::Text(n.to_string()),
                        Val::Int(role_id),
                    ],
                    map_role_base,
                )
                .await
                .map_err(internal)?;
            if clash.is_some() {
                return Err(MlflowError::resource_already_exists(format!(
                    "Role with name={n} already exists in workspace={}",
                    existing.workspace
                )));
            }
        }

        let mut sets: Vec<String> = Vec::new();
        let mut vals: Vec<Val> = Vec::new();
        let mut idx = 1usize;
        let new_name = name.map(str::to_string);
        let new_desc = description.map(str::to_string);
        if let Some(n) = &new_name {
            sets.push(format!("name = {}", dialect.placeholder(idx)));
            idx += 1;
            vals.push(Val::Text(n.clone()));
        }
        if let Some(d) = &new_desc {
            sets.push(format!("description = {}", dialect.placeholder(idx)));
            idx += 1;
            vals.push(Val::Text(d.clone()));
        }
        if !sets.is_empty() {
            let sql = format!(
                "UPDATE {ROLES} SET {} WHERE id = {}",
                sets.join(", "),
                dialect.placeholder(idx),
            );
            vals.push(Val::Int(role_id));
            self.db()
                .writer()
                .exec(&sql, &vals)
                .await
                .map_err(internal)?;
        }

        let base = RoleBase {
            id: existing.id,
            name: new_name.unwrap_or(existing.name),
            workspace: existing.workspace,
            description: new_desc.or(existing.description),
        };
        self.hydrate_role(base).await
    }

    /// `delete_role` (`sqlalchemy_store.py:1787`). The ORM cascade
    /// (`delete-orphan`) removes child `role_permissions` +
    /// `user_role_assignments`; we replicate it with explicit child deletes
    /// (the FK is not `ON DELETE CASCADE` at the DB level, `:1794`).
    pub async fn delete_role(&self, role_id: i64) -> Result<(), MlflowError> {
        self.fetch_role_by_id(role_id).await?;
        let dialect = self.db().writer().dialect();
        let mut tx = self.db().writer().begin_tx().await.map_err(internal)?;
        for stmt in [
            format!(
                "DELETE FROM {ROLE_PERMISSIONS} WHERE role_id = {}",
                dialect.placeholder(1)
            ),
            format!(
                "DELETE FROM {USER_ROLE_ASSIGNMENTS} WHERE role_id = {}",
                dialect.placeholder(1)
            ),
            format!("DELETE FROM {ROLES} WHERE id = {}", dialect.placeholder(1)),
        ] {
            if let Err(e) = tx.exec(&stmt, &[Val::Int(role_id)]).await {
                let _ = tx.commit().await;
                return Err(internal(e));
            }
        }
        tx.commit().await.map_err(internal)?;
        Ok(())
    }

    /// `delete_roles_for_workspace` (`sqlalchemy_store.py:1792`): bulk delete of
    /// every role in a workspace, with the child rows removed first.
    pub async fn delete_roles_for_workspace(&self, workspace: &str) -> Result<(), MlflowError> {
        let dialect = self.db().writer().dialect();
        let subq = format!(
            "SELECT id FROM {ROLES} WHERE workspace = {}",
            dialect.placeholder(1),
        );
        let mut tx = self.db().writer().begin_tx().await.map_err(internal)?;
        for stmt in [
            format!("DELETE FROM {ROLE_PERMISSIONS} WHERE role_id IN ({subq})"),
            format!("DELETE FROM {USER_ROLE_ASSIGNMENTS} WHERE role_id IN ({subq})"),
            format!(
                "DELETE FROM {ROLES} WHERE workspace = {}",
                dialect.placeholder(1)
            ),
        ] {
            if let Err(e) = tx.exec(&stmt, &[Val::Text(workspace.to_string())]).await {
                let _ = tx.commit().await;
                return Err(internal(e));
            }
        }
        tx.commit().await.map_err(internal)?;
        Ok(())
    }

    /// `delete_grants_for_resource` (`sqlalchemy_store.py:519`): delete every
    /// synthetic-role grant matching `(resource_type, resource_pattern)`.
    /// `workspace_scoped` restricts the sweep to the given workspace — used for
    /// resources (e.g. registered-model names) whose pattern can collide across
    /// workspaces. Admin-created roles are never touched (synthetic roles only).
    pub async fn delete_grants_for_resource(
        &self,
        resource_type: &str,
        resource_pattern: &str,
        workspace: Option<&str>,
    ) -> Result<(), MlflowError> {
        let role_ids = self.synthetic_role_ids(workspace).await?;
        if role_ids.is_empty() {
            return Ok(());
        }
        let dialect = self.db().writer().dialect();
        let in_list = role_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "DELETE FROM {ROLE_PERMISSIONS} \
             WHERE role_id IN ({in_list}) AND resource_type = {} AND resource_pattern = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db()
            .writer()
            .exec(
                &sql,
                &[
                    Val::Text(resource_type.to_string()),
                    Val::Text(resource_pattern.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `rename_grants_for_resource` (`sqlalchemy_store.py:543`): rewrite every
    /// synthetic-role grant on `(resource_type, old_pattern)` to
    /// `(resource_type, new_pattern)`. Used for resources whose pattern is the
    /// primary key and can change (registered-model rename).
    pub async fn rename_grants_for_resource(
        &self,
        resource_type: &str,
        old_pattern: &str,
        new_pattern: &str,
        workspace: Option<&str>,
    ) -> Result<(), MlflowError> {
        let role_ids = self.synthetic_role_ids(workspace).await?;
        if role_ids.is_empty() {
            return Ok(());
        }
        let dialect = self.db().writer().dialect();
        let in_list = role_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "UPDATE {ROLE_PERMISSIONS} SET resource_pattern = {} \
             WHERE role_id IN ({in_list}) AND resource_type = {} AND resource_pattern = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
        );
        self.db()
            .writer()
            .exec(
                &sql,
                &[
                    Val::Text(new_pattern.to_string()),
                    Val::Text(resource_type.to_string()),
                    Val::Text(old_pattern.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `_synthetic_role_ids` (`sqlalchemy_store.py:357`): ids of synthetic
    /// `__user_<id>__` roles, optionally scoped to `workspace`. Bulk
    /// cleanup/rename helpers route through this so they never touch
    /// admin-created roles that happen to share a grant. The synthetic-name
    /// match is done in Rust (mirroring Python's regex filter) rather than a
    /// dialect-specific SQL `LIKE`, whose `_` is a wildcard.
    async fn synthetic_role_ids(&self, workspace: Option<&str>) -> Result<Vec<i64>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let (sql, vals) = match workspace {
            None => (format!("SELECT id, name FROM {ROLES}"), vec![]),
            Some(ws) => (
                format!(
                    "SELECT id, name FROM {ROLES} WHERE workspace = {}",
                    dialect.placeholder(1)
                ),
                vec![Val::Text(ws.to_string())],
            ),
        };
        let rows = self
            .db()
            .reader()
            .fetch_all(&sql, &vals, |row: &dyn RowLike| {
                Ok((row.get_i64("id")?, row.get_string("name")?))
            })
            .await
            .map_err(internal)?;
        Ok(rows
            .into_iter()
            .filter(|(_, name)| is_synthetic_role_name(name))
            .map(|(id, _)| id)
            .collect())
    }

    // ---- RolePermission CRUD ----

    /// `add_role_permission` (`sqlalchemy_store.py:1813`): validate the
    /// permission for the resource type, reject non-`*` patterns on the
    /// workspace type, verify the role exists, insert. A duplicate
    /// `(role_id, resource_type, resource_pattern)` is `RESOURCE_ALREADY_EXISTS`.
    pub async fn add_role_permission(
        &self,
        role_id: i64,
        resource_type: &str,
        resource_pattern: &str,
        permission: &str,
    ) -> Result<RolePermission, MlflowError> {
        permissions::validate_permission_for_resource_type(permission, resource_type)?;
        if resource_type == permissions::RESOURCE_TYPE_WORKSPACE && resource_pattern != "*" {
            return Err(MlflowError::invalid_parameter_value(format!(
                "resource_type='{resource_type}' requires resource_pattern='*'. \
                 Got resource_pattern='{resource_pattern}'."
            )));
        }
        self.fetch_role_by_id(role_id).await?;

        let dialect = self.db().writer().dialect();
        let ph = |i| dialect.placeholder(i);
        let cols = "(role_id, resource_type, resource_pattern, permission)";
        let vals = vec![
            Val::Int(role_id),
            Val::Text(resource_type.to_string()),
            Val::Text(resource_pattern.to_string()),
            Val::Text(permission.to_string()),
        ];
        let mut tx = self.db().writer().begin_tx().await.map_err(internal)?;
        let insert_res = if dialect.supports_returning() {
            let sql = format!(
                "INSERT INTO {ROLE_PERMISSIONS} {cols} VALUES ({}, {}, {}, {}) RETURNING id",
                ph(1),
                ph(2),
                ph(3),
                ph(4),
            );
            tx.insert_returning_id(&sql, &vals).await
        } else {
            let sql = format!(
                "INSERT INTO {ROLE_PERMISSIONS} {cols} VALUES ({}, {}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
                ph(4),
            );
            match tx.exec(&sql, &vals).await {
                Ok(_) => tx.last_insert_id().await,
                Err(e) => Err(e),
            }
        };
        let id = match insert_res {
            Ok(id) => id,
            Err(e) => {
                let _ = tx.commit().await;
                return Err(map_role_permission_conflict(
                    e,
                    role_id,
                    resource_type,
                    resource_pattern,
                ));
            }
        };
        tx.commit().await.map_err(internal)?;

        Ok(RolePermission {
            id,
            role_id,
            resource_type: resource_type.to_string(),
            resource_pattern: resource_pattern.to_string(),
            permission: permission.to_string(),
        })
    }

    /// `get_role_permission` (`sqlalchemy_store.py:1867`) via `_get_role_permission`.
    pub async fn get_role_permission(&self, id: i64) -> Result<RolePermission, MlflowError> {
        self.fetch_role_permission(id).await
    }

    /// `remove_role_permission` (`sqlalchemy_store.py:1871`).
    pub async fn remove_role_permission(&self, id: i64) -> Result<(), MlflowError> {
        self.fetch_role_permission(id).await?;
        let dialect = self.db().writer().dialect();
        let sql = format!(
            "DELETE FROM {ROLE_PERMISSIONS} WHERE id = {}",
            dialect.placeholder(1)
        );
        self.db()
            .writer()
            .exec(&sql, &[Val::Int(id)])
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `list_role_permissions` (`sqlalchemy_store.py:1876`): validates the role
    /// exists, then returns its permission rows. Named distinctly from the
    /// T9.1 read-path `list_role_permissions` (which does not validate the
    /// role) to avoid clobbering it.
    pub async fn list_permissions_of_role(
        &self,
        role_id: i64,
    ) -> Result<Vec<RolePermission>, MlflowError> {
        self.fetch_role_by_id(role_id).await?;
        self.list_role_permissions(role_id).await
    }

    /// `update_role_permission` (`sqlalchemy_store.py:1884`): validate the new
    /// permission against the row's own resource type, then update.
    pub async fn update_role_permission(
        &self,
        id: i64,
        permission: &str,
    ) -> Result<RolePermission, MlflowError> {
        let rp = self.fetch_role_permission(id).await?;
        permissions::validate_permission_for_resource_type(permission, &rp.resource_type)?;
        let dialect = self.db().writer().dialect();
        let sql = format!(
            "UPDATE {ROLE_PERMISSIONS} SET permission = {} WHERE id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db()
            .writer()
            .exec(&sql, &[Val::Text(permission.to_string()), Val::Int(id)])
            .await
            .map_err(internal)?;
        Ok(RolePermission {
            permission: permission.to_string(),
            ..rp
        })
    }

    // ---- UserRoleAssignment CRUD ----

    /// `assign_role_to_user` (`sqlalchemy_store.py:1893`): validate the user and
    /// role exist, reject a duplicate `(user_id, role_id)`, insert.
    pub async fn assign_role_to_user(
        &self,
        user_id: i64,
        role_id: i64,
    ) -> Result<UserRoleAssignment, MlflowError> {
        if !self.user_exists(user_id).await? {
            return Err(MlflowError::resource_does_not_exist(format!(
                "User with id={user_id} not found"
            )));
        }
        self.fetch_role_by_id(role_id).await?;
        if self.assignment_exists(user_id, role_id).await? {
            return Err(MlflowError::resource_already_exists(format!(
                "User role assignment (user_id={user_id}, role_id={role_id}) already exists"
            )));
        }

        let dialect = self.db().writer().dialect();
        let ph = |i| dialect.placeholder(i);
        let cols = "(user_id, role_id)";
        let vals = vec![Val::Int(user_id), Val::Int(role_id)];
        let mut tx = self.db().writer().begin_tx().await.map_err(internal)?;
        let insert_res = if dialect.supports_returning() {
            let sql = format!(
                "INSERT INTO {USER_ROLE_ASSIGNMENTS} {cols} VALUES ({}, {}) RETURNING id",
                ph(1),
                ph(2),
            );
            tx.insert_returning_id(&sql, &vals).await
        } else {
            let sql = format!(
                "INSERT INTO {USER_ROLE_ASSIGNMENTS} {cols} VALUES ({}, {})",
                ph(1),
                ph(2),
            );
            match tx.exec(&sql, &vals).await {
                Ok(_) => tx.last_insert_id().await,
                Err(e) => Err(e),
            }
        };
        let id = match insert_res {
            Ok(id) => id,
            Err(e) => {
                let _ = tx.commit().await;
                // A lost unique-constraint race surfaces as the same
                // already-exists message; anything else is internal.
                if is_unique_violation(&e) {
                    return Err(MlflowError::resource_already_exists(format!(
                        "User role assignment (user_id={user_id}, role_id={role_id}) already exists"
                    )));
                }
                return Err(internal(e));
            }
        };
        tx.commit().await.map_err(internal)?;
        Ok(UserRoleAssignment {
            id,
            user_id,
            role_id,
        })
    }

    /// `unassign_role_from_user` (`sqlalchemy_store.py:1923`): delete the
    /// assignment, erroring `RESOURCE_DOES_NOT_EXIST` if absent.
    pub async fn unassign_role_from_user(
        &self,
        user_id: i64,
        role_id: i64,
    ) -> Result<(), MlflowError> {
        if !self.assignment_exists(user_id, role_id).await? {
            return Err(MlflowError::resource_does_not_exist(format!(
                "User role assignment (user_id={user_id}, role_id={role_id}) not found"
            )));
        }
        let dialect = self.db().writer().dialect();
        let sql = format!(
            "DELETE FROM {USER_ROLE_ASSIGNMENTS} WHERE user_id = {} AND role_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db()
            .writer()
            .exec(&sql, &[Val::Int(user_id), Val::Int(role_id)])
            .await
            .map_err(internal)?;
        Ok(())
    }

    /// `list_user_roles` (`sqlalchemy_store.py:1942`): every role assigned to a
    /// user (across workspaces), each hydrated with its permissions.
    pub async fn list_user_roles(&self, user_id: i64) -> Result<Vec<Role>, MlflowError> {
        self.roles_assigned_to_user(user_id, None).await
    }

    /// `list_user_roles_for_workspace` (`sqlalchemy_store.py:1954`).
    pub async fn list_user_roles_for_workspace(
        &self,
        user_id: i64,
        workspace: &str,
    ) -> Result<Vec<Role>, MlflowError> {
        self.roles_assigned_to_user(user_id, Some(workspace)).await
    }

    /// `list_role_users` (`sqlalchemy_store.py:1997`): validate the role exists,
    /// then return its assignments.
    pub async fn list_role_users(
        &self,
        role_id: i64,
    ) -> Result<Vec<UserRoleAssignment>, MlflowError> {
        self.fetch_role_by_id(role_id).await?;
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id, user_id, role_id FROM {USER_ROLE_ASSIGNMENTS} \
             WHERE role_id = {} ORDER BY id",
            dialect.placeholder(1),
        );
        self.db()
            .reader()
            .fetch_all(&sql, &[Val::Int(role_id)], |row: &dyn RowLike| {
                Ok(UserRoleAssignment {
                    id: row.get_i64("id")?,
                    user_id: row.get_i64("user_id")?,
                    role_id: row.get_i64("role_id")?,
                })
            })
            .await
            .map_err(internal)
    }

    // ---- Role-based permission resolution ----

    /// `get_role_permission_for_resource` (`sqlalchemy_store.py:2010`): the
    /// permission name a user resolves to on `(resource_type, resource_id)` in
    /// `workspace`, or `None`. Folds `(workspace, *)` grants (only MANAGE folds
    /// into concrete resource lookups; USE folds only for workspace-tier
    /// queries), matches specific + wildcard resource patterns, and unions
    /// across every assigned role via `max_permission`.
    pub async fn get_role_permission_for_resource(
        &self,
        user_id: i64,
        resource_type: &str,
        resource_id: &str,
        workspace: &str,
    ) -> Result<Option<&'static permissions::Permission>, MlflowError> {
        let rows = self
            .role_permission_rows_for_user_in_workspace(user_id, workspace)
            .await?;
        let mut best: Option<&str> = None;
        for (rp_type, rp_pattern, rp_perm) in &rows {
            if rp_type == permissions::RESOURCE_TYPE_WORKSPACE && rp_pattern == "*" {
                if resource_type == permissions::RESOURCE_TYPE_WORKSPACE
                    || rp_perm == permissions::MANAGE.name
                {
                    best = Some(fold_max(best, rp_perm));
                }
                continue;
            }
            if rp_type != resource_type {
                continue;
            }
            if rp_pattern == "*" || rp_pattern == resource_id {
                best = Some(fold_max(best, rp_perm));
            }
        }
        Ok(best.map(permissions::get_permission))
    }

    /// `is_workspace_admin` (`sqlalchemy_store.py:2100`) via
    /// `_workspace_admin_workspaces` (`:2056`): a MANAGE grant on
    /// `(workspace, *)` in the role's workspace.
    pub async fn is_workspace_admin(
        &self,
        user_id: i64,
        workspace: &str,
    ) -> Result<bool, MlflowError> {
        Ok(self
            .workspace_admin_workspaces(user_id)
            .await?
            .contains(workspace))
    }

    /// `list_workspace_admin_workspaces` (`sqlalchemy_store.py:2150`).
    pub async fn list_workspace_admin_workspaces(
        &self,
        user_id: i64,
    ) -> Result<BTreeSet<String>, MlflowError> {
        self.workspace_admin_workspaces(user_id).await
    }

    /// `list_role_grants_for_user_in_workspace` (`sqlalchemy_store.py:2111`):
    /// the `(resource_pattern, permission)` grants applying to `resource_type`
    /// in `workspace` — grants on that type plus workspace-wide `(*)` grants.
    pub async fn list_role_grants_for_user_in_workspace(
        &self,
        user_id: i64,
        workspace: &str,
        resource_type: &str,
    ) -> Result<Vec<(String, String)>, MlflowError> {
        permissions::validate_resource_type(resource_type)?;
        let rows = self
            .role_permission_rows_for_user_in_workspace(user_id, workspace)
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|(rp_type, rp_pattern, rp_perm)| {
                let matches_type = rp_type == resource_type
                    || (rp_type == permissions::RESOURCE_TYPE_WORKSPACE && rp_pattern == "*");
                matches_type.then_some((rp_pattern, rp_perm))
            })
            .collect())
    }

    // ---- Private helpers ----

    async fn workspace_admin_workspaces(
        &self,
        user_id: i64,
    ) -> Result<BTreeSet<String>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT DISTINCT r.workspace AS workspace \
             FROM {ROLES} r \
             JOIN {ROLE_PERMISSIONS} rp ON rp.role_id = r.id \
             JOIN {USER_ROLE_ASSIGNMENTS} a ON a.role_id = r.id \
             WHERE a.user_id = {} AND rp.resource_type = {} \
               AND rp.resource_pattern = '*' AND rp.permission = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
        );
        let rows = self
            .db()
            .reader()
            .fetch_all(
                &sql,
                &[
                    Val::Int(user_id),
                    Val::Text(permissions::RESOURCE_TYPE_WORKSPACE.to_string()),
                    Val::Text(permissions::MANAGE.name.to_string()),
                ],
                |row: &dyn RowLike| row.get_string("workspace"),
            )
            .await
            .map_err(internal)?;
        Ok(rows.into_iter().collect())
    }

    /// Every `(resource_type, resource_pattern, permission)` grant on a role
    /// the user is assigned to in `workspace`.
    async fn role_permission_rows_for_user_in_workspace(
        &self,
        user_id: i64,
        workspace: &str,
    ) -> Result<Vec<(String, String, String)>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT rp.resource_type AS resource_type, rp.resource_pattern AS resource_pattern, \
                    rp.permission AS permission \
             FROM {ROLE_PERMISSIONS} rp \
             JOIN {ROLES} r ON r.id = rp.role_id \
             JOIN {USER_ROLE_ASSIGNMENTS} a ON a.role_id = r.id \
             WHERE a.user_id = {} AND r.workspace = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db()
            .reader()
            .fetch_all(
                &sql,
                &[Val::Int(user_id), Val::Text(workspace.to_string())],
                |row: &dyn RowLike| {
                    Ok((
                        row.get_string("resource_type")?,
                        row.get_string("resource_pattern")?,
                        row.get_string("permission")?,
                    ))
                },
            )
            .await
            .map_err(internal)
    }

    async fn roles_assigned_to_user(
        &self,
        user_id: i64,
        workspace: Option<&str>,
    ) -> Result<Vec<Role>, MlflowError> {
        let dialect = self.db().reader().dialect();
        let (sql, vals) = match workspace {
            None => (
                format!(
                    "SELECT r.id AS id, r.name AS name, r.workspace AS workspace, \
                            r.description AS description \
                     FROM {ROLES} r \
                     JOIN {USER_ROLE_ASSIGNMENTS} a ON a.role_id = r.id \
                     WHERE a.user_id = {} ORDER BY r.id",
                    dialect.placeholder(1),
                ),
                vec![Val::Int(user_id)],
            ),
            Some(ws) => (
                format!(
                    "SELECT r.id AS id, r.name AS name, r.workspace AS workspace, \
                            r.description AS description \
                     FROM {ROLES} r \
                     JOIN {USER_ROLE_ASSIGNMENTS} a ON a.role_id = r.id \
                     WHERE a.user_id = {} AND r.workspace = {} ORDER BY r.id",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                ),
                vec![Val::Int(user_id), Val::Text(ws.to_string())],
            ),
        };
        let bases = self
            .db()
            .reader()
            .fetch_all(&sql, &vals, map_role_base)
            .await
            .map_err(internal)?;
        let mut roles = Vec::with_capacity(bases.len());
        for base in bases {
            roles.push(self.hydrate_role(base).await?);
        }
        Ok(roles)
    }

    async fn fetch_role_by_id(&self, role_id: i64) -> Result<RoleBase, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id, name, workspace, description FROM {ROLES} WHERE id = {}",
            dialect.placeholder(1),
        );
        self.db()
            .reader()
            .fetch_optional(&sql, &[Val::Int(role_id)], map_role_base)
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!("Role with id={role_id} not found"))
            })
    }

    async fn fetch_role_permission(&self, id: i64) -> Result<RolePermission, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id, role_id, resource_type, resource_pattern, permission \
             FROM {ROLE_PERMISSIONS} WHERE id = {}",
            dialect.placeholder(1),
        );
        self.db()
            .reader()
            .fetch_optional(&sql, &[Val::Int(id)], |row: &dyn RowLike| {
                Ok(RolePermission {
                    id: row.get_i64("id")?,
                    role_id: row.get_i64("role_id")?,
                    resource_type: row.get_string("resource_type")?,
                    resource_pattern: row.get_string("resource_pattern")?,
                    permission: row.get_string("permission")?,
                })
            })
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Role permission with id={id} not found"
                ))
            })
    }

    async fn hydrate_role(&self, base: RoleBase) -> Result<Role, MlflowError> {
        let permissions = self.list_role_permissions(base.id).await?;
        Ok(Role {
            id: base.id,
            name: base.name,
            workspace: base.workspace,
            description: base.description,
            permissions,
        })
    }

    async fn user_exists(&self, user_id: i64) -> Result<bool, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id FROM {USERS} WHERE id = {}",
            dialect.placeholder(1)
        );
        Ok(self
            .db()
            .reader()
            .fetch_optional(&sql, &[Val::Int(user_id)], |row: &dyn RowLike| {
                row.get_i64("id")
            })
            .await
            .map_err(internal)?
            .is_some())
    }

    async fn assignment_exists(&self, user_id: i64, role_id: i64) -> Result<bool, MlflowError> {
        let dialect = self.db().reader().dialect();
        let sql = format!(
            "SELECT id FROM {USER_ROLE_ASSIGNMENTS} WHERE user_id = {} AND role_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        Ok(self
            .db()
            .reader()
            .fetch_optional(
                &sql,
                &[Val::Int(user_id), Val::Int(role_id)],
                |row: &dyn RowLike| row.get_i64("id"),
            )
            .await
            .map_err(internal)?
            .is_some())
    }
}

/// The bare role row (no permissions yet).
struct RoleBase {
    id: i64,
    name: String,
    workspace: String,
    description: Option<String>,
}

fn map_role_base(row: &dyn RowLike) -> Result<RoleBase, sqlx::Error> {
    Ok(RoleBase {
        id: row.get_i64("id")?,
        name: row.get_string("name")?,
        workspace: row.get_string("workspace")?,
        description: row.get_opt_string("description")?,
    })
}

/// `_reject_synthetic_role_name` (`sqlalchemy_store.py:270`): reject any name
/// starting with the reserved `__user_` prefix (not just the strict
/// `__user_<digits>__` synthetic pattern).
pub(crate) fn reject_synthetic_role_name(name: &str) -> Result<(), MlflowError> {
    if name.starts_with(SYNTHETIC_ROLE_PREFIX) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Role name {name:?} uses the reserved '__user_' prefix, which \
             is held for the per-user permission representation. Choose a \
             different name."
        )));
    }
    Ok(())
}

/// `_is_synthetic_role_name` (`sqlalchemy_store.py:265`): matches the strict
/// `^__user_\d+__$` synthetic pattern (prefix + all-digit id + `__` suffix).
fn is_synthetic_role_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix(SYNTHETIC_ROLE_PREFIX) else {
        return false;
    };
    let Some(digits) = rest.strip_suffix("__") else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

fn fold_max<'a>(best: Option<&'a str>, candidate: &'a str) -> &'a str {
    match best {
        Some(b) if permission_priority(b) >= permission_priority(candidate) => b,
        Some(b) => max_permission(b, candidate),
        None => candidate,
    }
}

fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("auth database error: {e}"))
}

fn map_role_conflict(e: sqlx::Error, name: &str, workspace: &str) -> MlflowError {
    if is_unique_violation(&e) {
        MlflowError::resource_already_exists(format!(
            "Role (name={name}, workspace={workspace}) already exists. Error: {e}"
        ))
    } else {
        internal(e)
    }
}

fn map_role_permission_conflict(
    e: sqlx::Error,
    role_id: i64,
    resource_type: &str,
    resource_pattern: &str,
) -> MlflowError {
    if is_unique_violation(&e) {
        MlflowError::resource_already_exists(format!(
            "Role permission (role_id={role_id}, resource_type={resource_type}, \
             resource_pattern={resource_pattern}) already exists. Error: {e}"
        ))
    } else {
        internal(e)
    }
}

/// Detect a unique-constraint violation across the three backends (mirrors
/// `store::is_unique_violation`, kept crate-local there).
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
