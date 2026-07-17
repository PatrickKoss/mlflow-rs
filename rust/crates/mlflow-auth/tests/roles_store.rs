//! Store-level RBAC behaviors (plan T9.3), porting the observable behaviors of
//! `tests/server/auth/test_sqlalchemy_store_rbac.py`: role CRUD, role-permission
//! CRUD, user-role assignment, the synthetic per-user role get-or-create
//! (including the concurrent race and the `__user_` prefix rejection), the
//! role-based permission resolver, workspace-admin helpers, and scorer encoding.
//!
//! Each test runs against a fresh SQLite database whose four auth tables are
//! created by [`fresh_store`] with the same column shapes the Rust store reads
//! (`schema.rs`) — the migrated fixture DB is only needed for the cross-language
//! parity test.

use std::sync::atomic::{AtomicU64, Ordering};

use mlflow_auth::db::AuthDb;
use mlflow_auth::{AuthStore, EDIT, MANAGE, READ, USE};
use mlflow_store::{Db, PoolConfig};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;

const WS: &str = "default";

struct TempDb {
    path: std::path::PathBuf,
}

impl TempDb {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("mlflow_rust_roles_{}_{}.db", std::process::id(), n));
        let _ = std::fs::remove_file(&path);
        TempDb { path }
    }

    fn uri(&self) -> String {
        format!("sqlite:///{}", self.path.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Create a fresh auth DB with the four live tables + a satisfied
/// `alembic_version_auth` row, and return a store over it. Keeps the `TempDb`
/// alive for the caller's lifetime via the returned tuple.
async fn fresh_store() -> (AuthStore, TempDb) {
    let db = TempDb::new();
    let opts = SqliteConnectOptions::new()
        .filename(&db.path)
        .create_if_missing(true);
    let pool = SqlitePool::connect_with(opts)
        .await
        .expect("connect sqlite");
    for ddl in SCHEMA_DDL {
        sqlx::query(ddl)
            .execute(&pool)
            .await
            .expect("create schema");
    }
    sqlx::query("INSERT INTO alembic_version_auth (version_num) VALUES ('f1a2b3c4d5e6')")
        .execute(&pool)
        .await
        .expect("seed alembic head");
    drop(pool);

    let write = Db::connect(&db.uri(), PoolConfig::default())
        .await
        .expect("connect Db");
    let auth_db = AuthDb::from_pools(write, None);
    (AuthStore::new(auth_db), db)
}

const SCHEMA_DDL: &[&str] = &[
    "CREATE TABLE alembic_version_auth (version_num VARCHAR(32) NOT NULL PRIMARY KEY)",
    "CREATE TABLE users ( \
        id INTEGER PRIMARY KEY AUTOINCREMENT, \
        username VARCHAR(255) NOT NULL UNIQUE, \
        password_hash VARCHAR(255) NOT NULL, \
        is_admin BOOLEAN NOT NULL DEFAULT 0)",
    "CREATE TABLE roles ( \
        id INTEGER PRIMARY KEY AUTOINCREMENT, \
        name VARCHAR(255) NOT NULL, \
        workspace VARCHAR(63) NOT NULL, \
        description VARCHAR(1024), \
        UNIQUE (workspace, name))",
    "CREATE TABLE role_permissions ( \
        id INTEGER PRIMARY KEY AUTOINCREMENT, \
        role_id INTEGER NOT NULL, \
        resource_type VARCHAR(64) NOT NULL, \
        resource_pattern VARCHAR(255) NOT NULL, \
        permission VARCHAR(255) NOT NULL, \
        UNIQUE (role_id, resource_type, resource_pattern))",
    "CREATE TABLE user_role_assignments ( \
        id INTEGER PRIMARY KEY AUTOINCREMENT, \
        user_id INTEGER NOT NULL, \
        role_id INTEGER NOT NULL, \
        UNIQUE (user_id, role_id))",
];

async fn make_user(store: &AuthStore, name: &str) -> i64 {
    store
        .create_user(name, "password12345", false)
        .await
        .expect("create user")
        .id
}

// ---- Role CRUD ----

#[tokio::test]
async fn create_role_returns_entity() {
    let (store, _db) = fresh_store().await;
    let role = store
        .create_role("viewer", "ws1", Some("Read-only access"))
        .await
        .unwrap();
    assert_eq!(role.name, "viewer");
    assert_eq!(role.workspace, "ws1");
    assert_eq!(role.description.as_deref(), Some("Read-only access"));
    assert!(role.permissions.is_empty());
}

#[tokio::test]
async fn create_role_duplicate_errors() {
    let (store, _db) = fresh_store().await;
    store.create_role("viewer", "ws1", None).await.unwrap();
    let err = store.create_role("viewer", "ws1", None).await.unwrap_err();
    assert!(err.message.contains("already exists"));
}

#[tokio::test]
async fn create_role_same_name_different_workspace() {
    let (store, _db) = fresh_store().await;
    let r1 = store.create_role("viewer", "ws1", None).await.unwrap();
    let r2 = store.create_role("viewer", "ws2", None).await.unwrap();
    assert_ne!(r1.id, r2.id);
}

#[tokio::test]
async fn create_role_rejects_reserved_user_prefix() {
    let (store, _db) = fresh_store().await;
    for name in [
        "__user_1__",
        "__user_42__",
        "__user_999999__",
        "__user_admin",
        "__user_alice",
        "__user_foo_bar",
        "__user_",
    ] {
        let err = store.create_role(name, "ws1", None).await.unwrap_err();
        assert!(
            err.message.contains("reserved '__user_' prefix"),
            "name {name}: {}",
            err.message
        );
    }
}

#[tokio::test]
async fn update_role_rejects_reserved_user_prefix() {
    let (store, _db) = fresh_store().await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    for name in ["__user_1__", "__user_alice", "__user_"] {
        let err = store
            .update_role(role.id, Some(name), None)
            .await
            .unwrap_err();
        assert!(err.message.contains("reserved '__user_' prefix"));
    }
}

#[tokio::test]
async fn get_role_and_not_found() {
    let (store, _db) = fresh_store().await;
    let created = store.create_role("editor", "ws1", None).await.unwrap();
    let fetched = store.get_role(created.id).await.unwrap();
    assert_eq!(fetched.id, created.id);
    assert_eq!(fetched.name, "editor");

    let err = store.get_role(99999).await.unwrap_err();
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn get_role_by_name_and_not_found() {
    let (store, _db) = fresh_store().await;
    let created = store.create_role("editor", "ws1", None).await.unwrap();
    let fetched = store.get_role_by_name("ws1", "editor").await.unwrap();
    assert_eq!(fetched.id, created.id);

    let err = store.get_role_by_name("ws1", "nope").await.unwrap_err();
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn list_roles_scopes_by_workspace() {
    let (store, _db) = fresh_store().await;
    store.create_role("viewer", "ws1", None).await.unwrap();
    store.create_role("editor", "ws1", None).await.unwrap();
    store.create_role("viewer", "ws2", None).await.unwrap();
    store.create_role("other", "ws3", None).await.unwrap();

    let ws1: Vec<_> = store
        .list_roles(Some(&["ws1".to_string()]))
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.name)
        .collect();
    assert_eq!(sorted(ws1), vec!["editor", "viewer"]);

    let two = store
        .list_roles(Some(&["ws1".to_string(), "ws2".to_string()]))
        .await
        .unwrap();
    assert_eq!(two.len(), 3);

    // Empty slice -> no roles.
    assert!(store.list_roles(Some(&[])).await.unwrap().is_empty());

    // None -> whole system.
    assert_eq!(store.list_roles(None).await.unwrap().len(), 4);
}

#[tokio::test]
async fn update_role_sets_name_and_description() {
    let (store, _db) = fresh_store().await;
    let role = store
        .create_role("old-name", "ws1", Some("old desc"))
        .await
        .unwrap();
    let updated = store
        .update_role(role.id, Some("new-name"), Some("new desc"))
        .await
        .unwrap();
    assert_eq!(updated.name, "new-name");
    assert_eq!(updated.description.as_deref(), Some("new desc"));
}

#[tokio::test]
async fn update_role_name_conflict() {
    let (store, _db) = fresh_store().await;
    store.create_role("existing", "ws1", None).await.unwrap();
    let role2 = store.create_role("other", "ws1", None).await.unwrap();
    let err = store
        .update_role(role2.id, Some("existing"), None)
        .await
        .unwrap_err();
    assert!(err.message.contains("already exists"));
}

#[tokio::test]
async fn delete_role_cascades_permissions_and_assignments() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("role1", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "*", "READ")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();

    store.delete_role(role.id).await.unwrap();
    assert!(store.get_role(role.id).await.is_err());
    assert!(store.list_user_roles(uid).await.unwrap().is_empty());
}

#[tokio::test]
async fn delete_roles_for_workspace_cascades() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("r1", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "*", "READ")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();
    store.create_role("r3", "ws2", None).await.unwrap();

    store.delete_roles_for_workspace("ws1").await.unwrap();
    assert!(store
        .list_roles(Some(&["ws1".to_string()]))
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        store
            .list_roles(Some(&["ws2".to_string()]))
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(store.list_user_roles(uid).await.unwrap().is_empty());
}

// ---- RolePermission CRUD ----

#[tokio::test]
async fn add_role_permission_basics() {
    let (store, _db) = fresh_store().await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    let rp = store
        .add_role_permission(role.id, "experiment", "123", "READ")
        .await
        .unwrap();
    assert_eq!(rp.role_id, role.id);
    assert_eq!(rp.resource_type, "experiment");
    assert_eq!(rp.resource_pattern, "123");
    assert_eq!(rp.permission, "READ");

    let wc = store
        .add_role_permission(role.id, "experiment", "*", "READ")
        .await
        .unwrap();
    assert_eq!(wc.resource_pattern, "*");
}

#[tokio::test]
async fn add_role_permission_duplicate_and_invalid() {
    let (store, _db) = fresh_store().await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "123", "READ")
        .await
        .unwrap();
    assert!(store
        .add_role_permission(role.id, "experiment", "123", "EDIT")
        .await
        .unwrap_err()
        .message
        .contains("already exists"));

    assert!(store
        .add_role_permission(role.id, "experiment", "123", "INVALID")
        .await
        .unwrap_err()
        .message
        .contains("Invalid permission"));
    assert!(store
        .add_role_permission(role.id, "invalid_type", "123", "READ")
        .await
        .unwrap_err()
        .message
        .contains("Invalid resource type"));
}

#[tokio::test]
async fn add_role_permission_workspace_requires_wildcard() {
    let (store, _db) = fresh_store().await;
    let role = store.create_role("ws-role", "ws1", None).await.unwrap();
    let err = store
        .add_role_permission(role.id, "workspace", "42", "MANAGE")
        .await
        .unwrap_err();
    assert!(err.message.contains("resource_type='workspace' requires"));
}

#[tokio::test]
async fn add_role_permission_nonexistent_role() {
    let (store, _db) = fresh_store().await;
    let err = store
        .add_role_permission(99999, "experiment", "123", "READ")
        .await
        .unwrap_err();
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn remove_and_list_role_permissions() {
    let (store, _db) = fresh_store().await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    let rp = store
        .add_role_permission(role.id, "experiment", "123", "READ")
        .await
        .unwrap();
    store.remove_role_permission(rp.id).await.unwrap();
    assert!(store
        .list_permissions_of_role(role.id)
        .await
        .unwrap()
        .is_empty());

    assert!(store
        .remove_role_permission(99999)
        .await
        .unwrap_err()
        .message
        .contains("not found"));

    store
        .add_role_permission(role.id, "experiment", "1", "READ")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "experiment", "2", "EDIT")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "registered_model", "*", "READ")
        .await
        .unwrap();
    assert_eq!(
        store.list_permissions_of_role(role.id).await.unwrap().len(),
        3
    );
}

#[tokio::test]
async fn update_role_permission_and_errors() {
    let (store, _db) = fresh_store().await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    let rp = store
        .add_role_permission(role.id, "experiment", "123", "READ")
        .await
        .unwrap();
    let updated = store.update_role_permission(rp.id, "EDIT").await.unwrap();
    assert_eq!(updated.permission, "EDIT");

    assert!(store
        .update_role_permission(99999, "READ")
        .await
        .unwrap_err()
        .message
        .contains("not found"));
    assert!(store
        .update_role_permission(rp.id, "INVALID")
        .await
        .unwrap_err()
        .message
        .contains("Invalid permission"));
}

// ---- UserRoleAssignment CRUD ----

#[tokio::test]
async fn assign_unassign_role() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    let assignment = store.assign_role_to_user(uid, role.id).await.unwrap();
    assert_eq!(assignment.user_id, uid);
    assert_eq!(assignment.role_id, role.id);

    store.unassign_role_from_user(uid, role.id).await.unwrap();
    assert!(store.list_user_roles(uid).await.unwrap().is_empty());
}

#[tokio::test]
async fn assign_role_errors() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();

    assert!(store
        .assign_role_to_user(99999, role.id)
        .await
        .unwrap_err()
        .message
        .contains("not found"));

    store.assign_role_to_user(uid, role.id).await.unwrap();
    assert!(store
        .assign_role_to_user(uid, role.id)
        .await
        .unwrap_err()
        .message
        .contains("already exists"));

    assert!(store
        .assign_role_to_user(uid, 99999)
        .await
        .unwrap_err()
        .message
        .contains("not found"));
}

#[tokio::test]
async fn unassign_role_not_found() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let err = store.unassign_role_from_user(uid, 99999).await.unwrap_err();
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn list_user_roles_and_for_workspace() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let r1 = store.create_role("viewer", "ws1", None).await.unwrap();
    let r2 = store.create_role("editor", "ws1", None).await.unwrap();
    let r3 = store.create_role("viewer", "ws2", None).await.unwrap();
    store.assign_role_to_user(uid, r1.id).await.unwrap();
    store.assign_role_to_user(uid, r2.id).await.unwrap();
    store.assign_role_to_user(uid, r3.id).await.unwrap();

    assert_eq!(store.list_user_roles(uid).await.unwrap().len(), 3);
    assert_eq!(
        store
            .list_user_roles_for_workspace(uid, "ws1")
            .await
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        store
            .list_user_roles_for_workspace(uid, "ws2")
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn list_role_users() {
    let (store, _db) = fresh_store().await;
    let u1 = make_user(&store, "alice").await;
    let u2 = make_user(&store, "bob").await;
    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    store.assign_role_to_user(u1, role.id).await.unwrap();
    store.assign_role_to_user(u2, role.id).await.unwrap();

    let users = store.list_role_users(role.id).await.unwrap();
    let ids = sorted(users.iter().map(|a| a.user_id).collect());
    assert_eq!(ids, sorted(vec![u1, u2]));
}

// ---- Role-based permission resolution ----

#[tokio::test]
async fn resolver_no_roles_and_specific_match() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    assert!(store
        .get_role_permission_for_resource(uid, "experiment", "1", "ws1")
        .await
        .unwrap()
        .is_none());

    let role = store.create_role("viewer", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "1", "READ")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "1", "ws1")
            .await
            .unwrap(),
        Some(&READ)
    );
    // Different id -> no match.
    assert!(store
        .get_role_permission_for_resource(uid, "experiment", "999", "ws1")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn resolver_wildcard_and_union_and_specificity() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("mixed", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "*", "READ")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "experiment", "1", "EDIT")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();

    // Best-grant: specific EDIT wins for exp 1, wildcard READ for others.
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "1", "ws1")
            .await
            .unwrap(),
        Some(&EDIT)
    );
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "999", "ws1")
            .await
            .unwrap(),
        Some(&READ)
    );
}

#[tokio::test]
async fn resolver_union_picks_max_across_roles() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    for (i, perm) in ["READ", "USE", "MANAGE"].iter().enumerate() {
        let r = store
            .create_role(&format!("r{i}"), "ws1", None)
            .await
            .unwrap();
        store
            .add_role_permission(r.id, "experiment", "*", perm)
            .await
            .unwrap();
        store.assign_role_to_user(uid, r.id).await.unwrap();
    }
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "1", "ws1")
            .await
            .unwrap(),
        Some(&MANAGE)
    );
}

#[tokio::test]
async fn resolver_workspace_manage_folds_use_does_not() {
    let (store, _db) = fresh_store().await;
    // MANAGE folds into every resource type.
    let uid = make_user(&store, "admin_u").await;
    let admin = store.create_role("ws-admin", "ws1", None).await.unwrap();
    store
        .add_role_permission(admin.id, "workspace", "*", "MANAGE")
        .await
        .unwrap();
    store.assign_role_to_user(uid, admin.id).await.unwrap();
    for rt in ["experiment", "registered_model", "gateway_endpoint"] {
        assert_eq!(
            store
                .get_role_permission_for_resource(uid, rt, "any", "ws1")
                .await
                .unwrap(),
            Some(&MANAGE)
        );
    }

    // USE does not fold into concrete resource lookups.
    let uid2 = make_user(&store, "member_u").await;
    let member = store.create_role("user", "ws1", None).await.unwrap();
    store
        .add_role_permission(member.id, "workspace", "*", "USE")
        .await
        .unwrap();
    store.assign_role_to_user(uid2, member.id).await.unwrap();
    assert!(store
        .get_role_permission_for_resource(uid2, "experiment", "1", "ws1")
        .await
        .unwrap()
        .is_none());
    // Workspace-tier query still finds USE.
    assert_eq!(
        store
            .get_role_permission_for_resource(uid2, "workspace", "*", "ws1")
            .await
            .unwrap(),
        Some(&USE)
    );
}

#[tokio::test]
async fn resolver_cross_workspace_isolation() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("editor", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "*", "EDIT")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();

    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "1", "ws1")
            .await
            .unwrap(),
        Some(&EDIT)
    );
    assert!(store
        .get_role_permission_for_resource(uid, "experiment", "1", "ws2")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn resolver_resource_type_isolation() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("grants", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "registered_model", "foo", "READ")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "prompt", "bar", "MANAGE")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();

    assert!(store
        .get_role_permission_for_resource(uid, "prompt", "foo", "ws1")
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "registered_model", "foo", "ws1")
            .await
            .unwrap(),
        Some(&READ)
    );
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "prompt", "bar", "ws1")
            .await
            .unwrap(),
        Some(&MANAGE)
    );
}

#[tokio::test]
async fn workspace_admin_helpers() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let admin_ws1 = store.create_role("wa1", "ws1", None).await.unwrap();
    store
        .add_role_permission(admin_ws1.id, "workspace", "*", "MANAGE")
        .await
        .unwrap();
    store.assign_role_to_user(uid, admin_ws1.id).await.unwrap();
    let admin_ws3 = store.create_role("wa3", "ws3", None).await.unwrap();
    store
        .add_role_permission(admin_ws3.id, "workspace", "*", "MANAGE")
        .await
        .unwrap();
    store.assign_role_to_user(uid, admin_ws3.id).await.unwrap();
    // Non-MANAGE workspace grant does not count.
    let member = store.create_role("mem", "ws2", None).await.unwrap();
    store
        .add_role_permission(member.id, "workspace", "*", "USE")
        .await
        .unwrap();
    store.assign_role_to_user(uid, member.id).await.unwrap();

    assert!(store.is_workspace_admin(uid, "ws1").await.unwrap());
    assert!(!store.is_workspace_admin(uid, "ws2").await.unwrap());
    let admin_ws: Vec<String> = store
        .list_workspace_admin_workspaces(uid)
        .await
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(admin_ws, vec!["ws1".to_string(), "ws3".to_string()]);
}

#[tokio::test]
async fn list_role_grants_for_user_in_workspace() {
    let (store, _db) = fresh_store().await;
    let uid = make_user(&store, "alice").await;
    let role = store.create_role("multi", "ws1", None).await.unwrap();
    store
        .add_role_permission(role.id, "experiment", "42", "EDIT")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "experiment", "*", "READ")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "workspace", "*", "USE")
        .await
        .unwrap();
    store
        .add_role_permission(role.id, "registered_model", "*", "MANAGE")
        .await
        .unwrap();
    store.assign_role_to_user(uid, role.id).await.unwrap();

    let mut grants = store
        .list_role_grants_for_user_in_workspace(uid, "ws1", "experiment")
        .await
        .unwrap();
    grants.sort();
    let mut expected = vec![
        ("42".to_string(), "EDIT".to_string()),
        ("*".to_string(), "READ".to_string()),
        ("*".to_string(), "USE".to_string()),
    ];
    expected.sort();
    assert_eq!(grants, expected);

    // Invalid resource type rejected.
    assert!(store
        .list_role_grants_for_user_in_workspace(uid, "ws1", "not_a_type")
        .await
        .unwrap_err()
        .message
        .contains("Invalid resource type"));

    // Cross-workspace / no roles -> empty.
    assert!(store
        .list_role_grants_for_user_in_workspace(uid, "ws2", "experiment")
        .await
        .unwrap()
        .is_empty());
}

// ---- Synthetic per-user role grants ----

#[tokio::test]
async fn grant_user_permission_upserts_via_synthetic_role() {
    let (store, _db) = fresh_store().await;
    make_user(&store, "alice").await;

    store
        .grant_user_permission("alice", "experiment", "42", "READ", WS)
        .await
        .unwrap();
    // Resolves via the synthetic role.
    let uid = store.get_user("alice").await.unwrap().id;
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "42", WS)
            .await
            .unwrap(),
        Some(&READ)
    );

    // Re-granting a higher permission upserts (no error).
    store
        .grant_user_permission("alice", "experiment", "42", "EDIT", WS)
        .await
        .unwrap();
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "experiment", "42", WS)
            .await
            .unwrap(),
        Some(&EDIT)
    );

    // The synthetic role carries the reserved name.
    let roles = store.list_user_roles(uid).await.unwrap();
    assert_eq!(roles.len(), 1);
    assert_eq!(roles[0].name, AuthStore::synthetic_user_role_name(uid));
}

#[tokio::test]
async fn grant_and_revoke_user_resource_permission() {
    let (store, _db) = fresh_store().await;
    make_user(&store, "alice").await;

    store
        .grant_user_resource_permission("alice", "experiment", "7", "READ", WS)
        .await
        .unwrap();
    // Duplicate insert -> already exists.
    assert!(store
        .grant_user_resource_permission("alice", "experiment", "7", "EDIT", WS)
        .await
        .unwrap_err()
        .message
        .contains("already exists"));

    store
        .revoke_user_resource_permission("alice", "experiment", "7", WS)
        .await
        .unwrap();
    // Revoking again -> not found.
    assert!(store
        .revoke_user_resource_permission("alice", "experiment", "7", WS)
        .await
        .unwrap_err()
        .message
        .contains("not found"));
}

#[tokio::test]
async fn grant_user_resource_permission_rejects_workspace_type() {
    let (store, _db) = fresh_store().await;
    make_user(&store, "alice").await;
    assert!(store
        .grant_user_resource_permission("alice", "workspace", "*", "MANAGE", WS)
        .await
        .unwrap_err()
        .message
        .contains("resource_type 'workspace' is not supported"));
}

#[tokio::test]
async fn synthetic_role_get_or_create_is_idempotent() {
    let (store, _db) = fresh_store().await;
    make_user(&store, "alice").await;

    // Two sequential grants for the same user must reuse one synthetic role and
    // one assignment (no duplicate-role / duplicate-assignment errors).
    store
        .grant_user_permission("alice", "experiment", "1", "READ", WS)
        .await
        .unwrap();
    store
        .grant_user_permission("alice", "experiment", "2", "READ", WS)
        .await
        .unwrap();

    let uid = store.get_user("alice").await.unwrap().id;
    let roles = store.list_user_roles(uid).await.unwrap();
    assert_eq!(roles.len(), 1);
    assert_eq!(
        store
            .list_permissions_of_role(roles[0].id)
            .await
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn synthetic_role_get_or_create_concurrent_race() {
    let (store, _db) = fresh_store().await;
    make_user(&store, "racer").await;
    let uid = store.get_user("racer").await.unwrap().id;

    // Fire many concurrent grants for the same (user, workspace). The
    // SAVEPOINT-safe get-or-create must produce exactly one synthetic role and
    // one assignment, with no task erroring on the race.
    let mut handles = Vec::new();
    for i in 0..12 {
        let s = store.clone();
        handles.push(tokio::spawn(async move {
            s.grant_user_permission("racer", "experiment", &format!("e{i}"), "READ", WS)
                .await
        }));
    }
    for h in handles {
        h.await.unwrap().expect("no task should error on the race");
    }

    let roles = store.list_user_roles(uid).await.unwrap();
    assert_eq!(roles.len(), 1, "exactly one synthetic role");
    let assignments = store.list_role_users(roles[0].id).await.unwrap();
    assert_eq!(assignments.len(), 1, "exactly one assignment");
    assert_eq!(
        store
            .list_permissions_of_role(roles[0].id)
            .await
            .unwrap()
            .len(),
        12,
        "all grants landed"
    );
}

#[tokio::test]
async fn scorer_grants_use_encoded_pattern() {
    let (store, _db) = fresh_store().await;
    make_user(&store, "alice").await;
    let uid = store.get_user("alice").await.unwrap().id;

    // A scorer name containing a slash must round-trip through the encoded
    // compound key without colliding with the delimiter.
    let pattern = AuthStore::scorer_pattern("exp1", "my/scorer");
    assert_eq!(pattern, "exp1/my%2Fscorer");

    store
        .grant_user_permission("alice", "scorer", &pattern, "EDIT", WS)
        .await
        .unwrap();
    assert_eq!(
        store
            .get_role_permission_for_resource(uid, "scorer", &pattern, WS)
            .await
            .unwrap(),
        Some(&EDIT)
    );

    let (exp, name) = AuthStore::parse_scorer_pattern(&pattern);
    assert_eq!(exp, "exp1");
    assert_eq!(name, "my/scorer");
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}
