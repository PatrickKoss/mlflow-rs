//! Workspace-isolation integration tests for [`mlflow_registry::RegistryStore`],
//! ported from `tests/store/model_registry/test_sqlalchemy_workspace_store.py`.
//!
//! Every registry table has a workspace-leading composite PK, so operations in
//! one workspace must never see or mutate rows in another. These tests exercise
//! the cross-workspace no-leak contract for get/rename/delete/tags/aliases/
//! latest-versions, plus same-name coexistence across workspaces.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

use mlflow_error::ErrorCode;
use mlflow_registry::RegistryStore;
use mlflow_test_support::TempDb;

const TEAM_A: &str = "team-a";
const TEAM_B: &str = "team-b";

async fn store(temp: &TempDb) -> RegistryStore {
    RegistryStore::new(temp.connect().await)
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
    let tmp = TempDb::new("rm_scoped").await;
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
    let tmp = TempDb::new("rename_scoped").await;
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
    let tmp = TempDb::new("same_name").await;
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
    let tmp = TempDb::new("mv_scoped").await;
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
    let tmp = TempDb::new("alias_scoped").await;
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

#[tokio::test]
async fn model_version_lifecycle_is_workspace_scoped() {
    let tmp = TempDb::new("mv_lifecycle_scoped").await;
    let s = store(&tmp).await;
    // Same model name + version in both workspaces.
    for ws in [TEAM_A, TEAM_B] {
        s.create_registered_model(ws, "alpha", &[], None)
            .await
            .unwrap();
        s.create_model_version(ws, "alpha", "src", None, &[], None, None)
            .await
            .unwrap();
    }

    // Transition team-a's version; team-b's stays "None".
    s.transition_model_version_stage(TEAM_A, "alpha", "1", "Production", false)
        .await
        .unwrap();
    assert_eq!(
        s.get_model_version(TEAM_A, "alpha", "1")
            .await
            .unwrap()
            .current_stage
            .as_deref(),
        Some("Production")
    );
    assert_eq!(
        s.get_model_version(TEAM_B, "alpha", "1")
            .await
            .unwrap()
            .current_stage
            .as_deref(),
        Some("None")
    );

    // team-b transition/update/delete on the OTHER workspace's version is bounded
    // by workspace: operating on team-b's own version never touches team-a's.
    s.update_model_version(TEAM_B, "alpha", "1", Some("b desc"))
        .await
        .unwrap();
    assert!(s
        .get_model_version(TEAM_A, "alpha", "1")
        .await
        .unwrap()
        .description
        .is_none());

    // Delete team-b's version; team-a's version survives.
    s.delete_model_version(TEAM_B, "alpha", "1").await.unwrap();
    let a1 = s.get_model_version(TEAM_A, "alpha", "1").await.unwrap();
    assert_eq!(a1.current_stage.as_deref(), Some("Production"));
    let b_err = s.get_model_version(TEAM_B, "alpha", "1").await.unwrap_err();
    assert_eq!(b_err.error_code, ErrorCode::ResourceDoesNotExist);
}
