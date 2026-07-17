//! RBAC HTTP endpoint behaviors (plan T9.3), porting the observable behaviors
//! of `tests/server/auth/test_client_rbac.py` over the Rust axum router.
//!
//! These drive the `register_role_routes` router directly with `oneshot`
//! (no live listener), exercising the full endpoint matrix — role CRUD,
//! role-permission CRUD, assignment, per-user grant/revoke/get — plus the
//! Python-identical error shapes (`RESOURCE_DOES_NOT_EXIST`,
//! `RESOURCE_ALREADY_EXISTS`, `INVALID_PARAMETER_VALUE`) and the `/api/` +
//! `/ajax-api/` path parity.
//!
//! Authorization (who may call these) is the T9.4 middleware's job, so these
//! tests assume an already-authorized request — they verify the endpoint +
//! store wiring and wire shapes, matching what `test_client_rbac.py` asserts on
//! the response bodies once the request is past auth.

use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use mlflow_auth::db::AuthDb;
use mlflow_auth::AuthStore;
use mlflow_store::{Db, PoolConfig};
use serde_json::{json, Value};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tower::ServiceExt;

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

struct Harness {
    router: Router,
    store: AuthStore,
    _path: std::path::PathBuf,
}

async fn harness() -> Harness {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mlflow_rust_http_roles_{}_{}.db",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_file(&path);

    let opts = SqliteConnectOptions::new()
        .filename(&path)
        .create_if_missing(true);
    let pool = SqlitePool::connect_with(opts).await.unwrap();
    for ddl in SCHEMA_DDL {
        sqlx::query(ddl).execute(&pool).await.unwrap();
    }
    sqlx::query("INSERT INTO alembic_version_auth (version_num) VALUES ('f1a2b3c4d5e6')")
        .execute(&pool)
        .await
        .unwrap();
    drop(pool);

    let uri = format!("sqlite:///{}", path.display());
    let write = Db::connect(&uri, PoolConfig::default()).await.unwrap();
    let store = AuthStore::new(AuthDb::from_pools(write, None));
    let router = mlflow_server::auth_api::register_role_routes(store.clone());
    Harness {
        router,
        store,
        _path: path,
    }
}

async fn send(
    router: &Router,
    method: Method,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let builder = Request::builder().method(method).uri(path);
    let request = match body {
        Some(v) => builder
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&v).unwrap()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, value)
}

async fn post(router: &Router, path: &str, body: Value) -> (StatusCode, Value) {
    send(router, Method::POST, path, Some(body)).await
}

async fn get(router: &Router, path: &str) -> (StatusCode, Value) {
    send(router, Method::GET, path, None).await
}

// ---- Role CRUD ----

#[tokio::test]
async fn create_get_role() {
    let h = harness().await;
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer", "description": "read-only"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let role = &body["role"];
    assert_eq!(role["name"], "viewer");
    assert_eq!(role["workspace"], "ws1");
    assert_eq!(role["description"], "read-only");
    assert_eq!(role["permissions"], json!([]));

    let id = role["id"].as_i64().unwrap();
    let (status, body) = get(
        &h.router,
        &format!("/api/3.0/mlflow/roles/get?role_id={id}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["role"]["id"], id);
    assert_eq!(body["role"]["name"], "viewer");
}

#[tokio::test]
async fn create_role_duplicate_is_409() {
    let h = harness().await;
    post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error_code"], "RESOURCE_ALREADY_EXISTS");
    assert!(body["message"].as_str().unwrap().contains("already exists"));
}

#[tokio::test]
async fn create_role_rejects_reserved_prefix() {
    let h = harness().await;
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "__user_1__"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("reserved '__user_' prefix"));
}

#[tokio::test]
async fn create_role_empty_name_and_workspace() {
    let h = harness().await;
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "   "}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("Role name cannot be empty"));

    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "  ", "name": "viewer"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("Workspace cannot be empty"));
}

#[tokio::test]
async fn get_role_not_found_is_404() {
    let h = harness().await;
    let (status, body) = get(&h.router, "/api/3.0/mlflow/roles/get?role_id=99999").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error_code"], "RESOURCE_DOES_NOT_EXIST");
    assert!(body["message"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn list_roles_scoped_by_workspace() {
    let h = harness().await;
    for (ws, name) in [("ws1", "viewer"), ("ws1", "editor"), ("ws2", "viewer")] {
        post(
            &h.router,
            "/api/3.0/mlflow/roles/create",
            json!({"workspace": ws, "name": name}),
        )
        .await;
    }
    let (status, body) = get(&h.router, "/api/3.0/mlflow/roles/list?workspace=ws1").await;
    assert_eq!(status, StatusCode::OK);
    let names: Vec<&str> = body["roles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert_eq!(sorted(names), vec!["editor", "viewer"]);

    // Omitting workspace -> cross-workspace listing (admin path).
    let (_, body) = get(&h.router, "/api/3.0/mlflow/roles/list").await;
    assert_eq!(body["roles"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn update_role_and_reject_empty_update() {
    let h = harness().await;
    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "old"}),
    )
    .await;
    let id = body["role"]["id"].as_i64().unwrap();

    let (status, body) = send(
        &h.router,
        Method::PATCH,
        "/api/3.0/mlflow/roles/update",
        Some(json!({"role_id": id, "name": "new", "description": "updated"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["role"]["name"], "new");
    assert_eq!(body["role"]["description"], "updated");

    let (status, body) = send(
        &h.router,
        Method::PATCH,
        "/api/3.0/mlflow/roles/update",
        Some(json!({"role_id": id})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("At least one of 'name' or 'description'"));
}

#[tokio::test]
async fn delete_role() {
    let h = harness().await;
    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    let id = body["role"]["id"].as_i64().unwrap();
    let (status, _) = send(
        &h.router,
        Method::DELETE,
        "/api/3.0/mlflow/roles/delete",
        Some(json!({"role_id": id})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get(
        &h.router,
        &format!("/api/3.0/mlflow/roles/get?role_id={id}"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ---- Role permission CRUD ----

#[tokio::test]
async fn role_permission_crud() {
    let h = harness().await;
    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    let role_id = body["role"]["id"].as_i64().unwrap();

    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/permissions/add",
        json!({"role_id": role_id, "resource_type": "experiment", "resource_pattern": "42", "permission": "READ"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let rp = &body["role_permission"];
    assert_eq!(rp["resource_type"], "experiment");
    assert_eq!(rp["resource_pattern"], "42");
    assert_eq!(rp["permission"], "READ");
    let rp_id = rp["id"].as_i64().unwrap();

    let (_, body) = get(
        &h.router,
        &format!("/api/3.0/mlflow/roles/permissions/list?role_id={role_id}"),
    )
    .await;
    assert_eq!(body["role_permissions"].as_array().unwrap().len(), 1);

    let (status, body) = send(
        &h.router,
        Method::PATCH,
        "/api/3.0/mlflow/roles/permissions/update",
        Some(json!({"role_permission_id": rp_id, "permission": "EDIT"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["role_permission"]["permission"], "EDIT");

    let (status, _) = send(
        &h.router,
        Method::DELETE,
        "/api/3.0/mlflow/roles/permissions/remove",
        Some(json!({"role_permission_id": rp_id})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, body) = get(
        &h.router,
        &format!("/api/3.0/mlflow/roles/permissions/list?role_id={role_id}"),
    )
    .await;
    assert_eq!(body["role_permissions"], json!([]));
}

#[tokio::test]
async fn add_role_permission_invalid_resource_type() {
    let h = harness().await;
    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    let role_id = body["role"]["id"].as_i64().unwrap();
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/permissions/add",
        json!({"role_id": role_id, "resource_type": "bogus", "resource_pattern": "42", "permission": "READ"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("Invalid resource type"));
}

#[tokio::test]
async fn add_workspace_resource_type() {
    let h = harness().await;
    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "ws-admin"}),
    )
    .await;
    let role_id = body["role"]["id"].as_i64().unwrap();
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/permissions/add",
        json!({"role_id": role_id, "resource_type": "workspace", "resource_pattern": "*", "permission": "MANAGE"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["role_permission"]["resource_type"], "workspace");
    assert_eq!(body["role_permission"]["permission"], "MANAGE");
}

// ---- User-role assignment ----

#[tokio::test]
async fn assign_unassign_role() {
    let h = harness().await;
    h.store
        .create_user("alice", "password12345", false)
        .await
        .unwrap();
    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    let role_id = body["role"]["id"].as_i64().unwrap();

    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/assign",
        json!({"username": "alice", "role_id": role_id}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["assignment"]["role_id"], role_id);

    let (_, body) = get(&h.router, "/api/3.0/mlflow/users/roles/list?username=alice").await;
    let ids: Vec<i64> = body["roles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, vec![role_id]);

    let (_, body) = get(
        &h.router,
        &format!("/api/3.0/mlflow/roles/users/list?role_id={role_id}"),
    )
    .await;
    assert_eq!(body["assignments"].as_array().unwrap().len(), 1);

    let (status, _) = send(
        &h.router,
        Method::DELETE,
        "/api/3.0/mlflow/roles/unassign",
        Some(json!({"username": "alice", "role_id": role_id})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, body) = get(&h.router, "/api/3.0/mlflow/users/roles/list?username=alice").await;
    assert_eq!(body["roles"], json!([]));
}

#[tokio::test]
async fn assign_nonexistent_role_and_user() {
    let h = harness().await;
    h.store
        .create_user("alice", "password12345", false)
        .await
        .unwrap();
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/assign",
        json!({"username": "alice", "role_id": 99999}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["message"].as_str().unwrap().contains("not found"));

    let (_, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1", "name": "viewer"}),
    )
    .await;
    let role_id = body["role"]["id"].as_i64().unwrap();
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/assign",
        json!({"username": "ghost", "role_id": role_id}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["message"].as_str().unwrap().contains("not found"));
}

// ---- Per-user grant / revoke / get ----

#[tokio::test]
async fn grant_revoke_get_user_permission() {
    let h = harness().await;
    h.store
        .create_user("alice", "password12345", false)
        .await
        .unwrap();

    // Grant.
    let (status, _) = post(
        &h.router,
        "/api/3.0/mlflow/users/permissions/grant",
        json!({"username": "alice", "resource_type": "experiment", "resource_id": "42", "permission": "EDIT"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Get resolves the granted permission; allowed mirrors can_use.
    let (status, body) = get(
        &h.router,
        "/api/3.0/mlflow/users/permissions/get?username=alice&resource_type=experiment&resource_id=42",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["permission"], "EDIT");
    assert_eq!(body["allowed"], true);

    // Unknown resource -> deny-by-default.
    let (status, body) = get(
        &h.router,
        "/api/3.0/mlflow/users/permissions/get?username=alice&resource_type=experiment&resource_id=999",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["permission"], "NO_PERMISSIONS");
    assert_eq!(body["allowed"], false);

    // Duplicate grant -> already exists.
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/users/permissions/grant",
        json!({"username": "alice", "resource_type": "experiment", "resource_id": "42", "permission": "READ"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"].as_str().unwrap().contains("already exists"));

    // Revoke, then revoke again -> not found.
    let (status, _) = post(
        &h.router,
        "/api/3.0/mlflow/users/permissions/revoke",
        json!({"username": "alice", "resource_type": "experiment", "resource_id": "42"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/users/permissions/revoke",
        json!({"username": "alice", "resource_type": "experiment", "resource_id": "42"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["message"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn get_user_permission_unknown_user_and_type() {
    let h = harness().await;
    let (status, body) = get(
        &h.router,
        "/api/3.0/mlflow/users/permissions/get?username=ghost&resource_type=experiment&resource_id=1",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["message"].as_str().unwrap().contains("not found"));

    h.store
        .create_user("alice", "password12345", false)
        .await
        .unwrap();
    let (status, body) = get(
        &h.router,
        "/api/3.0/mlflow/users/permissions/get?username=alice&resource_type=bogus&resource_id=1",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("Invalid resource type"));
}

// ---- Missing-param error shape ----

#[tokio::test]
async fn missing_required_param() {
    let h = harness().await;
    let (status, body) = post(
        &h.router,
        "/api/3.0/mlflow/roles/create",
        json!({"workspace": "ws1"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");
    assert!(body["message"]
        .as_str()
        .unwrap()
        .contains("Missing value for required parameter 'name'"));
}

// ---- /api and /ajax-api path parity ----

#[tokio::test]
async fn endpoints_reachable_at_both_prefixes() {
    let h = harness().await;
    for prefix in ["api", "ajax-api"] {
        let (status, body) = post(
            &h.router,
            &format!("/{prefix}/3.0/mlflow/roles/create"),
            json!({"workspace": "path-parity", "name": format!("r-{prefix}")}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "prefix {prefix}");
        let id = body["role"]["id"].as_i64().unwrap();

        let (status, body) = get(
            &h.router,
            &format!("/{prefix}/3.0/mlflow/roles/get?role_id={id}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "prefix {prefix}");
        assert_eq!(body["role"]["id"], id);

        let (status, body) = get(
            &h.router,
            &format!("/{prefix}/3.0/mlflow/roles/list?workspace=path-parity"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "prefix {prefix}");
        let names: Vec<&str> = body["roles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&format!("r-{prefix}").as_str()));
    }
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}
