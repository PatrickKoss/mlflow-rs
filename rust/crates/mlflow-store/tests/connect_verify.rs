//! Integration tests for [`mlflow_store::Db::connect_and_verify`] against a
//! real Alembic-migrated SQLite database.
//!
//! The fixture `tests/fixtures/tracking.db` is a fully migrated MLflow tracking
//! DB at Alembic head `a3f8c21d9b47`, produced by
//! `uv run --frozen python rust/tools/make_test_db.py`. Regenerate it with that
//! command whenever the head changes.
//!
//! Postgres/MySQL live-connect smoke tests are gated behind
//! `MLFLOW_RUST_TEST_PG_URI` / `MLFLOW_RUST_TEST_MYSQL_URI` so CI can opt in
//! later (plan §6 item 8); they are skipped when unset.

use std::path::{Path, PathBuf};

use mlflow_store::db::{Db, SchemaError, StoreError, EXPECTED_ALEMBIC_HEAD};
use mlflow_store::PoolConfig;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

fn sqlite_uri(path: &Path) -> String {
    format!("sqlite:///{}", path.display())
}

/// Copy the read-only fixture into a temp file so tampering tests don't mutate
/// the checked-in DB. Returns the temp path; the caller keeps it alive.
fn temp_copy_of_fixture(tag: &str) -> PathBuf {
    let dst = std::env::temp_dir().join(format!(
        "mlflow_rust_store_{}_{}.db",
        tag,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&dst);
    std::fs::copy(fixture_path(), &dst).expect("copy fixture");
    dst
}

#[tokio::test]
async fn connect_and_verify_succeeds_on_migrated_db() {
    let uri = sqlite_uri(&fixture_path());
    let db = Db::connect_and_verify(&uri)
        .await
        .expect("verify migrated fixture DB");
    assert_eq!(db.dialect(), mlflow_store::Dialect::Sqlite);
}

#[tokio::test]
async fn refuses_out_of_date_schema() {
    let path = temp_copy_of_fixture("stale");
    // Tamper: set the alembic head to an older/bogus revision.
    {
        let db = Db::connect(&sqlite_uri(&path), PoolConfig::default())
            .await
            .expect("connect temp copy");
        if let Db::Sqlite(pool) = &db {
            sqlx::query("UPDATE alembic_version SET version_num = '0000deadbeef'")
                .execute(pool)
                .await
                .expect("tamper alembic_version");
        } else {
            panic!("expected sqlite pool");
        }
    }

    let err = Db::connect_and_verify(&sqlite_uri(&path))
        .await
        .expect_err("stale schema must be refused");
    match err {
        StoreError::Schema(SchemaError::OutOfDate { found, expected }) => {
            assert_eq!(found, "0000deadbeef");
            assert_eq!(expected, EXPECTED_ALEMBIC_HEAD);
        }
        other => panic!("expected OutOfDate, got {other:?}"),
    }
    let msg = format!(
        "{}",
        Db::connect_and_verify(&sqlite_uri(&path))
            .await
            .unwrap_err()
    );
    assert!(msg.contains("mlflow db upgrade"), "message: {msg}");
    assert!(
        msg.contains("out-of-date database schema"),
        "message: {msg}"
    );

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn refuses_uninitialized_db() {
    let path = temp_copy_of_fixture("uninit");
    // Simulate an uninitialized DB by dropping the alembic bookkeeping table.
    {
        let db = Db::connect(&sqlite_uri(&path), PoolConfig::default())
            .await
            .expect("connect temp copy");
        if let Db::Sqlite(pool) = &db {
            sqlx::query("DROP TABLE alembic_version")
                .execute(pool)
                .await
                .expect("drop alembic_version");
        } else {
            panic!("expected sqlite pool");
        }
    }

    let err = Db::connect_and_verify(&sqlite_uri(&path))
        .await
        .expect_err("uninitialized DB must be refused");
    match err {
        StoreError::Schema(SchemaError::Uninitialized { table }) => {
            assert_eq!(table, "alembic_version");
        }
        other => panic!("expected Uninitialized, got {other:?}"),
    }

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn sqlite_session_pragmas_are_in_effect() {
    let db = Db::connect(&sqlite_uri(&fixture_path()), PoolConfig::default())
        .await
        .expect("connect fixture");
    let Db::Sqlite(pool) = &db else {
        panic!("expected sqlite pool");
    };

    let foreign_keys: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(pool)
        .await
        .expect("read foreign_keys pragma");
    assert_eq!(foreign_keys, 1, "foreign_keys should be ON");

    let busy_timeout: i64 = sqlx::query_scalar("PRAGMA busy_timeout")
        .fetch_one(pool)
        .await
        .expect("read busy_timeout pragma");
    assert_eq!(busy_timeout, 20_000);

    // `PRAGMA case_sensitive_like` is write-only in SQLite (querying it returns
    // no rows), so verify it behaviorally: with it ON, 'A' LIKE 'a' is false (0);
    // with the default (OFF) it would be true (1).
    let like_is_case_sensitive: i64 = sqlx::query_scalar("SELECT 'A' LIKE 'a'")
        .fetch_one(pool)
        .await
        .expect("evaluate LIKE case sensitivity");
    assert_eq!(
        like_is_case_sensitive, 0,
        "case_sensitive_like should be ON, so 'A' LIKE 'a' must be false"
    );
}

#[tokio::test]
async fn missing_sqlite_file_does_not_get_created() {
    let path = std::env::temp_dir().join(format!(
        "mlflow_rust_store_absent_{}.db",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let result = Db::connect(&sqlite_uri(&path), PoolConfig::default()).await;
    assert!(
        result.is_err(),
        "connecting to a missing DB file must error"
    );
    assert!(
        !path.exists(),
        "Rust must never create the schema/DB file (Python-owned)"
    );
}

#[tokio::test]
async fn connects_to_pg_when_configured() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_PG_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_PG_URI not set");
        return;
    };
    let db = Db::connect_and_verify(&uri)
        .await
        .expect("verify postgres DB");
    assert_eq!(db.dialect(), mlflow_store::Dialect::Postgres);
}

#[tokio::test]
async fn connects_to_mysql_when_configured() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_MYSQL_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_MYSQL_URI not set");
        return;
    };
    let db = Db::connect_and_verify(&uri).await.expect("verify mysql DB");
    assert_eq!(db.dialect(), mlflow_store::Dialect::MySql);
}
