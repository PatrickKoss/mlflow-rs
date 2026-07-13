//! Workspace-isolation integration tests for [`mlflow_registry::RegistryStore`],
//! ported from `tests/store/model_registry/test_sqlalchemy_workspace_store.py`.
//!
//! Every registry table has a workspace-leading composite PK, so operations in
//! one workspace must never see or mutate rows in another. These tests exercise
//! the cross-workspace no-leak contract for get/rename/delete/tags/aliases/
//! latest-versions, plus same-name coexistence across workspaces.

use std::path::{Path, PathBuf};

use mlflow_error::ErrorCode;
use mlflow_registry::RegistryStore;
use mlflow_store::{Db, PoolConfig};

const TEAM_A: &str = "team-a";
const TEAM_B: &str = "team-b";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("mlflow-store")
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_registryws_{}_{}_{}.db",
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

fn assert_not_found(err: mlflow_error::MlflowError, name: &str) {
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert_eq!(
        err.message,
        format!("Registered Model with name={name} not found")
    );
}

#[tokio::test]
async fn registered_model_ops_are_workspace_scoped() {
    let tmp = TempDb::new("rm_scoped");
    let s = store(&tmp).await;

    // team-a owns "alpha" with a tag.
    s.create_registered_model(TEAM_A, "alpha", &[("owner", "team-a")], None)
        .await
        .unwrap();
    s.create_registered_model(TEAM_B, "beta", &[], None)
        .await
        .unwrap();

    // team-b cannot get/tag/rename/delete alpha.
    assert_not_found(
        s.get_registered_model(TEAM_B, "alpha").await.unwrap_err(),
        "alpha",
    );
    assert_not_found(
        s.set_registered_model_tag(TEAM_B, "alpha", "x", "y")
            .await
            .unwrap_err(),
        "alpha",
    );
    assert_not_found(
        s.rename_registered_model(TEAM_B, "alpha", "alpha-b")
            .await
            .unwrap_err(),
        "alpha",
    );
    assert_not_found(
        s.delete_registered_model(TEAM_B, "alpha")
            .await
            .unwrap_err(),
        "alpha",
    );

    // team-a still owns alpha with its tag intact.
    let alpha = s.get_registered_model(TEAM_A, "alpha").await.unwrap();
    assert_eq!(alpha.tags.len(), 1);
    assert_eq!(alpha.tags[0].key, "owner");
}

#[tokio::test]
async fn rename_is_workspace_scoped_and_preserves_tags() {
    let tmp = TempDb::new("rename_scoped");
    let s = store(&tmp).await;
    s.create_registered_model(TEAM_A, "alpha", &[("owner", "team-a")], None)
        .await
        .unwrap();
    s.create_registered_model(TEAM_B, "beta", &[], None)
        .await
        .unwrap();

    // Rename in team-a; tag preserved.
    s.rename_registered_model(TEAM_A, "alpha", "alpha-renamed")
        .await
        .unwrap();
    let renamed = s
        .get_registered_model(TEAM_A, "alpha-renamed")
        .await
        .unwrap();
    assert_eq!(renamed.name, "alpha-renamed");
    assert_eq!(renamed.tags.len(), 1);
    assert_eq!(renamed.tags[0].value.as_deref(), Some("team-a"));

    // team-b still cannot see the renamed model.
    assert_not_found(
        s.get_registered_model(TEAM_B, "alpha-renamed")
            .await
            .unwrap_err(),
        "alpha-renamed",
    );
}

#[tokio::test]
async fn same_name_allowed_in_different_workspaces() {
    let tmp = TempDb::new("same_name");
    let s = store(&tmp).await;
    s.create_registered_model(TEAM_A, "shared-name", &[("w", "a")], None)
        .await
        .unwrap();
    // No conflict creating the same name in another workspace.
    s.create_registered_model(TEAM_B, "shared-name", &[("w", "b")], None)
        .await
        .unwrap();
    let a = s.get_registered_model(TEAM_A, "shared-name").await.unwrap();
    let b = s.get_registered_model(TEAM_B, "shared-name").await.unwrap();
    assert_eq!(a.tags[0].value.as_deref(), Some("a"));
    assert_eq!(b.tags[0].value.as_deref(), Some("b"));
}

#[tokio::test]
async fn model_version_reads_are_workspace_scoped() {
    let tmp = TempDb::new("mv_scoped");
    let s = store(&tmp).await;
    s.create_registered_model(TEAM_A, "alpha", &[], None)
        .await
        .unwrap();
    s.create_model_version(TEAM_A, "alpha", "src", None, &[], None, None)
        .await
        .unwrap();

    // team-a sees version 1.
    let latest = s.get_latest_versions(TEAM_A, "alpha", None).await.unwrap();
    assert_eq!(latest.len(), 1);
    assert_eq!(latest[0].version, "1");

    // team-b sees nothing: get_model_version + get_latest_versions both error.
    let mv_err = s.get_model_version(TEAM_B, "alpha", "1").await.unwrap_err();
    assert_eq!(mv_err.error_code, ErrorCode::ResourceDoesNotExist);
    assert_eq!(
        mv_err.message,
        "Model Version (name=alpha, version=1) not found"
    );
    assert_not_found(
        s.get_latest_versions(TEAM_B, "alpha", None)
            .await
            .unwrap_err(),
        "alpha",
    );
}

#[tokio::test]
async fn alias_ops_are_workspace_scoped() {
    let tmp = TempDb::new("alias_scoped");
    let s = store(&tmp).await;
    s.create_registered_model(TEAM_A, "alpha", &[], None)
        .await
        .unwrap();
    s.create_model_version(TEAM_A, "alpha", "src", None, &[], None, None)
        .await
        .unwrap();
    s.set_registered_model_alias(TEAM_A, "alpha", "production", "1")
        .await
        .unwrap();
    let a = s.get_registered_model(TEAM_A, "alpha").await.unwrap();
    assert_eq!(a.aliases.len(), 1);
    assert_eq!(a.aliases[0].alias, "production");

    // team-b: setting an alias errors (version not found in this workspace),
    // deleting errors (registered model not found in this workspace).
    let set_err = s
        .set_registered_model_alias(TEAM_B, "alpha", "shadow", "1")
        .await
        .unwrap_err();
    assert_eq!(set_err.error_code, ErrorCode::ResourceDoesNotExist);
    let del_err = s
        .delete_registered_model_alias(TEAM_B, "alpha", "production")
        .await
        .unwrap_err();
    assert_not_found(del_err, "alpha");

    // team-a still owns the alias.
    let a = s.get_registered_model(TEAM_A, "alpha").await.unwrap();
    assert_eq!(a.aliases.len(), 1);
}
