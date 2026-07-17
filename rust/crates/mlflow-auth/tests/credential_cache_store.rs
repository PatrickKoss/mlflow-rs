//! Credential-cache integration behaviors (plan T9.8): `authenticate_and_get_user`
//! (`_authenticate_cached`) with the cache enabled must serve the second request
//! for the same `(user, password)` from the cache instead of re-querying /
//! re-hashing against the store, and must drop the entry on a mutation
//! (`_invalidate_user_auth_cache`).
//!
//! Observability trick: after the first (caching) authenticate, we mutate the
//! DB **out of band** (raw SQL / `delete_user` on a *separate* store handle that
//! doesn't share this store's cache). A subsequent authenticate that still
//! succeeds proves it never re-consulted the store — exactly the staleness
//! window `basic_auth.ini` documents for the cache.

use std::sync::atomic::{AtomicU64, Ordering};

use mlflow_auth::db::AuthDb;
use mlflow_auth::{AuthConfig, AuthStore};
use mlflow_store::{Db, PoolConfig};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;

const PW: &str = "password12345";

struct TempDb {
    path: std::path::PathBuf,
}

impl TempDb {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("mlflow_rust_credc_{}_{}.db", std::process::id(), n));
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

/// A store over a fresh auth DB with the credential cache enabled (ttl 60s).
/// Returns the store plus the `TempDb` (keep-alive) and its URI (so a second
/// handle can be opened for out-of-band mutation).
async fn cached_store(ttl_seconds: u64) -> (AuthStore, TempDb, String) {
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

    let uri = db.uri();
    let write = Db::connect(&uri, PoolConfig::default())
        .await
        .expect("connect Db");
    let auth_db = AuthDb::from_pools(write, None);
    let config = AuthConfig {
        auth_cache_ttl_seconds: ttl_seconds,
        ..AuthConfig::default()
    };
    (AuthStore::with_config(auth_db, config), db, uri)
}

/// A second store handle over the same DB with its own (independent) cache — a
/// stand-in for a different worker process doing an out-of-band mutation.
async fn other_handle(uri: &str) -> AuthStore {
    let write = Db::connect(uri, PoolConfig::default())
        .await
        .expect("connect Db");
    AuthStore::new(AuthDb::from_pools(write, None))
}

#[tokio::test]
async fn cache_enabled_serves_second_request_without_store() {
    let (store, _db, uri) = cached_store(60).await;
    store.create_user("alice", PW, false).await.expect("create");

    // First authenticate: cache miss → store query + hash → caches the user.
    assert!(store.credential_cache_len() == 0);
    let first = store.authenticate_and_get_user("alice", PW).await;
    assert!(first.is_some(), "first auth succeeds");
    assert_eq!(store.credential_cache_len(), 1, "entry cached after miss");

    // Delete the user out of band (via a separate handle that does NOT share
    // this store's cache), so the store no longer has the row.
    let other = other_handle(&uri).await;
    other.delete_user("alice").await.expect("delete");
    assert!(
        !store.authenticate_user("alice", PW).await,
        "uncached path reflects the deletion immediately"
    );

    // Second authenticate on the *cached* path still succeeds — proving it
    // served from the cache and never re-queried the (now empty) store.
    let second = store.authenticate_and_get_user("alice", PW).await;
    assert!(
        second.is_some(),
        "cache hit serves the user without a store query"
    );
}

#[tokio::test]
async fn cache_disabled_always_hits_store() {
    // ttl 0 → cache off; every authenticate reflects live store state.
    let (store, _db, uri) = cached_store(0).await;
    store.create_user("bob", PW, false).await.expect("create");
    assert!(!store.credential_cache_enabled());

    assert!(store.authenticate_and_get_user("bob", PW).await.is_some());
    assert_eq!(store.credential_cache_len(), 0, "nothing cached when off");

    let other = other_handle(&uri).await;
    other.delete_user("bob").await.expect("delete");
    assert!(
        store.authenticate_and_get_user("bob", PW).await.is_none(),
        "disabled cache always sees the live store (deletion honoured)"
    );
}

#[tokio::test]
async fn password_change_invalidates_cache_on_same_store() {
    let (store, _db, _uri) = cached_store(60).await;
    store.create_user("carol", PW, false).await.expect("create");

    assert!(store.authenticate_and_get_user("carol", PW).await.is_some());
    assert_eq!(store.credential_cache_len(), 1);

    // A password change through the same store must invalidate the cached
    // credential immediately (`_invalidate_user_auth_cache`).
    let new_pw = "brand-new-password-9";
    store
        .update_user("carol", Some(new_pw), None)
        .await
        .expect("update password");
    assert_eq!(
        store.credential_cache_len(),
        0,
        "password change drops the cached entry"
    );
    // The old password no longer authenticates; the new one does.
    assert!(store.authenticate_and_get_user("carol", PW).await.is_none());
    assert!(store
        .authenticate_and_get_user("carol", new_pw)
        .await
        .is_some());
}

#[tokio::test]
async fn wrong_password_is_not_cached() {
    let (store, _db, _uri) = cached_store(60).await;
    store.create_user("dave", PW, false).await.expect("create");
    assert!(store
        .authenticate_and_get_user("dave", "wrong-password-x")
        .await
        .is_none());
    assert_eq!(
        store.credential_cache_len(),
        0,
        "a failed check caches nothing"
    );
}
