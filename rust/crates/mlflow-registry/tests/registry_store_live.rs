//! Live Postgres/MySQL behavioral smoke for [`mlflow_registry::RegistryStore`],
//! gated behind `MLFLOW_RUST_TEST_PG_URI` / `MLFLOW_RUST_TEST_MYSQL_URI` (plan
//! §6 item 8), mirroring how `mlflow-store` gates its dialect matrix.
//!
//! These cover the dialect-sensitive parts of T7.1 — notably the rename
//! **cascade** (which depends on FK `ON UPDATE CASCADE` being honored, and on
//! MySQL/InnoDB the `version` `Integer` vs `BigInteger` decode) and the
//! `ROW_NUMBER` window used by `get_latest_versions`. Each run uses a unique
//! workspace so it is isolated from any existing rows and from concurrent runs.

use mlflow_error::ErrorCode;
use mlflow_registry::RegistryStore;
use mlflow_store::{Db, PoolConfig};

#[tokio::test]
async fn pg_registry_smoke() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_PG_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_PG_URI not set");
        return;
    };
    registry_smoke(&uri).await;
}

#[tokio::test]
async fn mysql_registry_smoke() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_MYSQL_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_MYSQL_URI not set");
        return;
    };
    registry_smoke(&uri).await;
}

async fn registry_smoke(uri: &str) {
    let db = Db::connect_and_verify_with(uri, PoolConfig::default())
        .await
        .expect("connect");
    let s = RegistryStore::new(db);
    let ws = format!("rust-reg-smoke-{}-{}", std::process::id(), unique());
    let name = format!("m-{}", unique());

    // Create with tags + description.
    let rm = s
        .create_registered_model(&ws, &name, &[("owner", "team")], Some("desc"))
        .await
        .unwrap();
    assert_eq!(rm.description.as_deref(), Some("desc"));
    assert_eq!(rm.tags.len(), 1);

    // Two versions + an alias on version 2.
    s.create_model_version(&ws, &name, "src/1", None, &[], None, None)
        .await
        .unwrap();
    s.create_model_version(&ws, &name, "src/2", None, &[], None, None)
        .await
        .unwrap();
    s.set_registered_model_alias(&ws, &name, "champion", "2")
        .await
        .unwrap();

    // latest_versions (ROW_NUMBER window) → highest version, stage "None".
    let latest = s.get_latest_versions(&ws, &name, None).await.unwrap();
    assert_eq!(latest.len(), 1);
    assert_eq!(latest[0].version, "2");
    assert_eq!(latest[0].current_stage.as_deref(), Some("None"));

    // Rename cascade: model + versions + tag + alias move to the new name.
    let new_name = format!("{name}-renamed");
    s.rename_registered_model(&ws, &name, &new_name)
        .await
        .unwrap();
    let renamed = s.get_registered_model(&ws, &new_name).await.unwrap();
    assert_eq!(renamed.tags.len(), 1);
    assert_eq!(renamed.aliases.len(), 1);
    assert_eq!(renamed.aliases[0].version, "2");
    assert_eq!(
        s.get_model_version(&ws, &new_name, "2").await.unwrap().name,
        new_name
    );
    // Old name gone.
    let err = s.get_registered_model(&ws, &name).await.unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);

    // get-by-alias resolves; unknown alias errors.
    let mv = s
        .get_model_version_by_alias(&ws, &new_name, "champion")
        .await
        .unwrap();
    assert_eq!(mv.version, "2");
    assert_eq!(mv.aliases, vec!["champion".to_string()]);

    // Cleanup so repeated CI runs stay isolated.
    s.delete_registered_model(&ws, &new_name).await.unwrap();
}

/// A cheap unique suffix without pulling in the `uuid`/`rand` crates.
fn unique() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
