//! Behavioral integration tests for [`mlflow_registry::RegistryStore`] (plan
//! T7.1), ported from the Python store suites
//! (`tests/store/model_registry/test_sqlalchemy_store.py` and
//! `test_sqlalchemy_workspace_store.py`).
//!
//! Each test copies the checked-in SQLite fixture (a real Alembic-migrated DB
//! at head `b7e4c1a90f23`, shared with `mlflow-store`) into a temp file, so the
//! committed fixture is never mutated. The registry tables live in the same
//! migrated DB as the tracking tables.
//!
//! Live Postgres/MySQL runs are gated behind `MLFLOW_RUST_TEST_PG_URI` /
//! `MLFLOW_RUST_TEST_MYSQL_URI` (see `registry_store_live.rs`), mirroring how
//! `mlflow-store` gates its dialect matrix (plan §6 item 8).

use std::path::{Path, PathBuf};

use mlflow_error::ErrorCode;
use mlflow_registry::RegistryStore;
use mlflow_store::{Db, PoolConfig};

const WS: &str = "default";

fn fixture_path() -> PathBuf {
    // Reuse the tracking-store fixture (registry tables are in the same DB).
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("mlflow-store")
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

/// Copy the fixture to a unique temp file; the guard removes it on drop.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_registrystore_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixture_path(), &path).expect("copy fixture");
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

async fn store(temp: &TempDb) -> RegistryStore {
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect temp fixture");
    RegistryStore::new(db)
}

fn tags_map(tags: &[mlflow_registry::RegisteredModelTag]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = tags
        .iter()
        .map(|t| (t.key.clone(), t.value.clone().unwrap_or_default()))
        .collect();
    v.sort();
    v
}

// ---------------------------------------------------------------------------
// create_registered_model
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_registered_model_defaults_and_timestamps() {
    let tmp = TempDb::new("create_rm");
    let s = store(&tmp).await;
    let rm = s
        .create_registered_model(WS, "model_1", &[], None)
        .await
        .unwrap();
    assert_eq!(rm.name, "model_1");
    assert!(rm.description.is_none());
    assert_eq!(rm.latest_versions, vec![]);
    assert!(rm.creation_timestamp.is_some());
    // creation == last_updated at creation.
    assert_eq!(rm.creation_timestamp, rm.last_updated_timestamp);
}

#[tokio::test]
async fn create_registered_model_with_tags_and_description() {
    let tmp = TempDb::new("create_rm_tags");
    let s = store(&tmp).await;
    let rm = s
        .create_registered_model(
            WS,
            "tagged",
            &[("key", "value"), ("anotherKey", "some other value")],
            Some("the best model ever"),
        )
        .await
        .unwrap();
    assert_eq!(rm.description.as_deref(), Some("the best model ever"));
    assert_eq!(
        tags_map(&rm.tags),
        vec![
            ("anotherKey".to_string(), "some other value".to_string()),
            ("key".to_string(), "value".to_string()),
        ]
    );
    // Round-trip via get.
    let fetched = s.get_registered_model(WS, "tagged").await.unwrap();
    assert_eq!(tags_map(&fetched.tags), tags_map(&rm.tags));
}

#[tokio::test]
async fn create_duplicate_name_conflicts() {
    let tmp = TempDb::new("create_dup");
    let s = store(&tmp).await;
    s.create_registered_model(WS, "dupe", &[], None)
        .await
        .unwrap();
    let err = s
        .create_registered_model(WS, "dupe", &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceAlreadyExists);
    assert_eq!(err.message, "Registered Model (name=dupe) already exists");
}

#[tokio::test]
async fn create_empty_name_is_missing_value() {
    let tmp = TempDb::new("create_empty");
    let s = store(&tmp).await;
    let err = s
        .create_registered_model(WS, "", &[], None)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert_eq!(err.message, "Missing value for required parameter 'name'.");
}

// ---------------------------------------------------------------------------
// get / update / delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_missing_model_errors() {
    let tmp = TempDb::new("get_missing");
    let s = store(&tmp).await;
    let err = s.get_registered_model(WS, "nope").await.unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert_eq!(err.message, "Registered Model with name=nope not found");
}

#[tokio::test]
async fn update_registered_model_description() {
    let tmp = TempDb::new("update_rm");
    let s = store(&tmp).await;
    s.create_registered_model(WS, "model_for_update_RM", &[], None)
        .await
        .unwrap();
    let updated = s
        .update_registered_model(WS, "model_for_update_RM", Some("test model"))
        .await
        .unwrap();
    assert_eq!(updated.name, "model_for_update_RM");
    let fetched = s
        .get_registered_model(WS, "model_for_update_RM")
        .await
        .unwrap();
    assert_eq!(fetched.description.as_deref(), Some("test model"));
}

#[tokio::test]
async fn delete_registered_model_cascades_and_errors_after() {
    let tmp = TempDb::new("delete_rm");
    let s = store(&tmp).await;
    let name = "model_for_delete_RM";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    s.create_model_version(WS, name, "path/to/source", None, &[], None, None)
        .await
        .unwrap();
    s.delete_registered_model(WS, name).await.unwrap();

    // get / update / delete all error not-found.
    for err in [
        s.get_registered_model(WS, name).await.unwrap_err(),
        s.update_registered_model(WS, name, Some("x"))
            .await
            .unwrap_err(),
        s.delete_registered_model(WS, name).await.unwrap_err(),
    ] {
        assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
        assert_eq!(
            err.message,
            format!("Registered Model with name={name} not found")
        );
    }
    // Cascade to versions.
    let mv_err = s.get_model_version(WS, name, "1").await.unwrap_err();
    assert_eq!(mv_err.error_code, ErrorCode::ResourceDoesNotExist);
    assert_eq!(
        mv_err.message,
        format!("Model Version (name={name}, version=1) not found")
    );
}

// ---------------------------------------------------------------------------
// rename (cascade)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rename_cascades_to_versions_tags_aliases() {
    let tmp = TempDb::new("rename_rm");
    let s = store(&tmp).await;
    let original = "original name";
    let new_name = "new name";
    s.create_registered_model(WS, original, &[("owner", "team-a")], None)
        .await
        .unwrap();
    s.create_model_version(WS, original, "path/1", None, &[], None, None)
        .await
        .unwrap();
    s.create_model_version(WS, original, "path/2", None, &[], None, None)
        .await
        .unwrap();
    s.set_registered_model_alias(WS, original, "champion", "2")
        .await
        .unwrap();

    s.rename_registered_model(WS, original, new_name)
        .await
        .unwrap();

    // Model + both versions moved to the new name.
    let renamed = s.get_registered_model(WS, new_name).await.unwrap();
    assert_eq!(renamed.name, new_name);
    // Tag cascaded.
    assert_eq!(
        tags_map(&renamed.tags),
        vec![("owner".to_string(), "team-a".to_string())]
    );
    // Alias cascaded (still points at version 2).
    assert_eq!(renamed.aliases.len(), 1);
    assert_eq!(renamed.aliases[0].alias, "champion");
    assert_eq!(renamed.aliases[0].version, "2");
    // Versions cascaded.
    assert_eq!(
        s.get_model_version(WS, new_name, "1").await.unwrap().name,
        new_name
    );
    assert_eq!(
        s.get_model_version(WS, new_name, "2").await.unwrap().name,
        new_name
    );

    // Old name gone.
    let err = s.get_registered_model(WS, original).await.unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert_eq!(
        err.message,
        format!("Registered Model with name={original} not found")
    );
}

#[tokio::test]
async fn rename_to_existing_name_conflicts() {
    let tmp = TempDb::new("rename_conflict");
    let s = store(&tmp).await;
    s.create_registered_model(WS, "new name", &[], None)
        .await
        .unwrap();
    s.create_registered_model(WS, "original name", &[], None)
        .await
        .unwrap();
    let err = s
        .rename_registered_model(WS, "new name", "original name")
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceAlreadyExists);
    assert!(
        err.message
            .starts_with("Registered Model (name=original name) already exists"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn rename_empty_new_name_is_missing_value() {
    let tmp = TempDb::new("rename_empty");
    let s = store(&tmp).await;
    s.create_registered_model(WS, "m", &[], None).await.unwrap();
    let err = s.rename_registered_model(WS, "m", "").await.unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert_eq!(
        err.message,
        "Missing value for required parameter 'new_name'."
    );
}

// ---------------------------------------------------------------------------
// get_latest_versions
// ---------------------------------------------------------------------------

fn latest_by_stage(mvs: &[mlflow_registry::ModelVersion]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = mvs
        .iter()
        .map(|mv| {
            (
                mv.current_stage.clone().unwrap_or_default(),
                mv.version.clone(),
            )
        })
        .collect();
    v.sort();
    v
}

#[tokio::test]
async fn get_latest_versions_default_stage_none() {
    let tmp = TempDb::new("latest_none");
    let s = store(&tmp).await;
    let name = "test_for_latest_versions";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    // Empty.
    assert_eq!(
        s.get_registered_model(WS, name)
            .await
            .unwrap()
            .latest_versions,
        vec![]
    );
    // One version → stage "None".
    s.create_model_version(WS, name, "path/1", None, &[], None, None)
        .await
        .unwrap();
    let rm = s.get_registered_model(WS, name).await.unwrap();
    assert_eq!(
        latest_by_stage(&rm.latest_versions),
        vec![("None".to_string(), "1".to_string())]
    );
    // get_latest_versions with None and empty stages behave the same.
    assert_eq!(
        latest_by_stage(&s.get_latest_versions(WS, name, None).await.unwrap()),
        vec![("None".to_string(), "1".to_string())]
    );
    assert_eq!(
        latest_by_stage(&s.get_latest_versions(WS, name, Some(&[])).await.unwrap()),
        vec![("None".to_string(), "1".to_string())]
    );
}

#[tokio::test]
async fn get_latest_versions_highest_per_stage() {
    let tmp = TempDb::new("latest_multi");
    let s = store(&tmp).await;
    let name = "multi";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    // Three versions, all default stage "None": latest is the highest (3).
    for src in ["a", "b", "c"] {
        s.create_model_version(WS, name, src, None, &[], None, None)
            .await
            .unwrap();
    }
    assert_eq!(
        latest_by_stage(&s.get_latest_versions(WS, name, None).await.unwrap()),
        vec![("None".to_string(), "3".to_string())]
    );
    // Filter to a specific (canonical + case-insensitive) stage.
    assert_eq!(
        latest_by_stage(
            &s.get_latest_versions(WS, name, Some(&["none"]))
                .await
                .unwrap()
        ),
        vec![("None".to_string(), "3".to_string())]
    );
    // A stage with no versions yields nothing.
    assert_eq!(
        s.get_latest_versions(WS, name, Some(&["Production"]))
            .await
            .unwrap(),
        vec![]
    );
    // Unknown stage errors.
    let err = s
        .get_latest_versions(WS, name, Some(&["bogus"]))
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err.message.contains("Invalid Model Version stage: bogus"));
}

// ---------------------------------------------------------------------------
// registered-model tags
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_and_overwrite_registered_model_tag() {
    let tmp = TempDb::new("set_rm_tag");
    let s = store(&tmp).await;
    let name = "SetRegisteredModelTag_TestMod";
    s.create_registered_model(WS, name, &[("key", "value")], None)
        .await
        .unwrap();
    s.set_registered_model_tag(WS, name, "randomTag", "not a random value")
        .await
        .unwrap();
    let rm = s.get_registered_model(WS, name).await.unwrap();
    assert_eq!(
        tags_map(&rm.tags),
        vec![
            ("key".to_string(), "value".to_string()),
            ("randomTag".to_string(), "not a random value".to_string()),
        ]
    );
    // Overwrite existing key.
    s.set_registered_model_tag(WS, name, "key", "overriding")
        .await
        .unwrap();
    let rm = s.get_registered_model(WS, name).await.unwrap();
    let key_val = rm
        .tags
        .iter()
        .find(|t| t.key == "key")
        .and_then(|t| t.value.clone());
    assert_eq!(key_val.as_deref(), Some("overriding"));
}

#[tokio::test]
async fn set_registered_model_tag_value_too_long() {
    let tmp = TempDb::new("set_rm_tag_long");
    let s = store(&tmp).await;
    s.create_registered_model(WS, "m", &[], None).await.unwrap();
    let long = "a".repeat(100_001);
    let err = s
        .set_registered_model_tag(WS, "m", "longTagKey", &long)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert_eq!(
        err.message,
        "'value' exceeds the maximum length of 100000 characters"
    );
    // 4999 is fine.
    assert!(s
        .set_registered_model_tag(WS, "m", "okKey", &"a".repeat(4999))
        .await
        .is_ok());
}

#[tokio::test]
async fn delete_registered_model_tag_and_missing_noop() {
    let tmp = TempDb::new("del_rm_tag");
    let s = store(&tmp).await;
    let name = "DeleteRegisteredModelTag_TestMod";
    s.create_registered_model(
        WS,
        name,
        &[("key", "value"), ("anotherKey", "some other value")],
        None,
    )
    .await
    .unwrap();
    s.delete_registered_model_tag(WS, name, "key")
        .await
        .unwrap();
    let rm = s.get_registered_model(WS, name).await.unwrap();
    assert_eq!(
        tags_map(&rm.tags),
        vec![("anotherKey".to_string(), "some other value".to_string())]
    );
    // Deleting a now-missing key is a silent no-op.
    s.delete_registered_model_tag(WS, name, "key")
        .await
        .unwrap();
    let rm = s.get_registered_model(WS, name).await.unwrap();
    assert_eq!(
        tags_map(&rm.tags),
        vec![("anotherKey".to_string(), "some other value".to_string())]
    );
    // On a deleted model, tag deletion errors not-found.
    s.delete_registered_model(WS, name).await.unwrap();
    let err = s
        .delete_registered_model_tag(WS, name, "anotherKey")
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
}

// ---------------------------------------------------------------------------
// aliases
// ---------------------------------------------------------------------------

async fn setup_aliases(s: &RegistryStore) -> &'static str {
    let name = "alias_model";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    s.create_model_version(WS, name, "v1", None, &[], None, None)
        .await
        .unwrap();
    s.create_model_version(WS, name, "v2", None, &[], None, None)
        .await
        .unwrap();
    // Version passed as a string, matching the wire.
    s.set_registered_model_alias(WS, name, "test_alias", "2")
        .await
        .unwrap();
    name
}

#[tokio::test]
async fn set_and_read_alias() {
    let tmp = TempDb::new("set_alias");
    let s = store(&tmp).await;
    let name = setup_aliases(&s).await;
    let rm = s.get_registered_model(WS, name).await.unwrap();
    assert_eq!(rm.aliases.len(), 1);
    assert_eq!(rm.aliases[0].alias, "test_alias");
    assert_eq!(rm.aliases[0].version, "2");
    // MV .aliases is a list of alias strings for that version.
    assert_eq!(
        s.get_model_version(WS, name, "1").await.unwrap().aliases,
        Vec::<String>::new()
    );
    assert_eq!(
        s.get_model_version(WS, name, "2").await.unwrap().aliases,
        vec!["test_alias".to_string()]
    );
}

#[tokio::test]
async fn alias_overwrite_repoints_version() {
    let tmp = TempDb::new("alias_overwrite");
    let s = store(&tmp).await;
    let name = setup_aliases(&s).await;
    // Repoint the existing alias to version 1.
    s.set_registered_model_alias(WS, name, "test_alias", "1")
        .await
        .unwrap();
    let rm = s.get_registered_model(WS, name).await.unwrap();
    assert_eq!(rm.aliases.len(), 1);
    assert_eq!(rm.aliases[0].version, "1");
}

#[tokio::test]
async fn set_alias_on_missing_version_errors() {
    let tmp = TempDb::new("alias_missing_ver");
    let s = store(&tmp).await;
    let name = "am";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    let err = s
        .set_registered_model_alias(WS, name, "a", "5")
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert_eq!(
        err.message,
        format!("Model Version (name={name}, version=5) not found")
    );
}

#[tokio::test]
async fn reserved_alias_names_rejected() {
    let tmp = TempDb::new("alias_reserved");
    let s = store(&tmp).await;
    let name = setup_aliases(&s).await;
    for (alias, frag) in [
        (
            "latest",
            "'latest' alias name (case insensitive) is reserved.",
        ),
        ("v3", "Version alias name 'v3' is reserved."),
    ] {
        let err = s
            .set_registered_model_alias(WS, name, alias, "2")
            .await
            .unwrap_err();
        assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
        assert_eq!(err.message, frag);
    }
}

#[tokio::test]
async fn delete_alias_and_missing_noop() {
    let tmp = TempDb::new("del_alias");
    let s = store(&tmp).await;
    let name = setup_aliases(&s).await;
    s.delete_registered_model_alias(WS, name, "test_alias")
        .await
        .unwrap();
    assert_eq!(
        s.get_registered_model(WS, name).await.unwrap().aliases,
        vec![]
    );
    assert_eq!(
        s.get_model_version(WS, name, "2").await.unwrap().aliases,
        Vec::<String>::new()
    );
    // Deleting a now-missing alias is a no-op.
    s.delete_registered_model_alias(WS, name, "test_alias")
        .await
        .unwrap();
}

#[tokio::test]
async fn get_model_version_by_alias_and_not_found() {
    let tmp = TempDb::new("get_by_alias");
    let s = store(&tmp).await;
    let name = setup_aliases(&s).await;
    let mv = s
        .get_model_version_by_alias(WS, name, "test_alias")
        .await
        .unwrap();
    assert_eq!(mv.version, "2");
    assert_eq!(mv.aliases, vec!["test_alias".to_string()]);
    // Unknown alias.
    let err = s
        .get_model_version_by_alias(WS, name, "ghost")
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert_eq!(err.message, "Registered model alias ghost not found.");
}

#[tokio::test]
async fn get_model_version_by_latest_alias() {
    let tmp = TempDb::new("get_latest_alias");
    let s = store(&tmp).await;
    let name = "latest_model";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    s.create_model_version(WS, name, "v1", None, &[], None, None)
        .await
        .unwrap();
    // 'latest' resolves to the newest version without a stored alias row.
    let mv = s
        .get_model_version_by_alias(WS, name, "latest")
        .await
        .unwrap();
    assert_eq!(mv.version, "1");
}

// ---------------------------------------------------------------------------
// download uri (storage_location or source)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn download_uri_returns_source() {
    let tmp = TempDb::new("dl_uri");
    let s = store(&tmp).await;
    let name = "dl";
    s.create_registered_model(WS, name, &[], None)
        .await
        .unwrap();
    s.create_model_version(WS, name, "s3://bucket/path", None, &[], None, None)
        .await
        .unwrap();
    let uri = s
        .get_model_version_download_uri(WS, name, "1")
        .await
        .unwrap();
    assert_eq!(uri, "s3://bucket/path");
}
