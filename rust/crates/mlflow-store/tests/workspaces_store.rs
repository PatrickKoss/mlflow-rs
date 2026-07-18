//! Behavioral integration tests for [`mlflow_store::WorkspaceStore`] (plan
//! T10.1), ported from `tests/store/workspace/test_sqlalchemy_store.py`.
//!
//! Each test copies the checked-in SQLite fixture (a real Alembic-migrated DB at
//! head `c4a9b7d3e812`, which already contains the `workspaces` table and a
//! `default` row) into a temp file and operates on it, so the committed fixture
//! is never mutated.

use std::path::{Path, PathBuf};

use mlflow_error::ErrorCode;
use mlflow_store::{
    verify_single_tenant_data, Db, PoolConfig, Workspace, WorkspaceDeletionMode, WorkspaceStore,
};

const WS_URI: &str = "sqlite-workspaces";

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
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
            "mlflow_rust_wsstore_{}_{}_{}.db",
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

async fn store(temp: &TempDb) -> WorkspaceStore {
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect temp fixture");
    WorkspaceStore::new(db, WS_URI)
}

/// Insert an experiment row directly (mirrors the raw `INSERT INTO experiments`
/// the Python tests use). Returns the generated experiment id.
async fn insert_experiment(db: &Db, name: &str, workspace: &str) -> i64 {
    use sqlx::Row;
    match db {
        Db::Sqlite(p) => {
            let id: i64 = sqlx::query(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) \
                 VALUES (?, ?, 'active') RETURNING experiment_id",
            )
            .bind(name)
            .bind(workspace)
            .fetch_one(p)
            .await
            .expect("insert experiment")
            .get("experiment_id");
            id
        }
        _ => unreachable!("tests use sqlite"),
    }
}

async fn insert_experiment_with_id(db: &Db, id: i64, name: &str, workspace: &str) {
    match db {
        Db::Sqlite(p) => {
            sqlx::query(
                "INSERT INTO experiments (experiment_id, name, workspace, lifecycle_stage) \
                 VALUES (?, ?, ?, 'active')",
            )
            .bind(id)
            .bind(name)
            .bind(workspace)
            .execute(p)
            .await
            .expect("insert experiment");
        }
        _ => unreachable!(),
    }
}

async fn insert_run(db: &Db, run_id: &str, exp_id: i64) {
    match db {
        Db::Sqlite(p) => {
            sqlx::query(
                "INSERT INTO runs (run_uuid, name, experiment_id, lifecycle_stage, status, \
                 source_type, start_time, end_time) \
                 VALUES (?, ?, ?, 'active', 'FINISHED', 'LOCAL', 0, 0)",
            )
            .bind(run_id)
            .bind("test-run")
            .bind(exp_id)
            .execute(p)
            .await
            .expect("insert run");
        }
        _ => unreachable!(),
    }
}

async fn experiment_workspace(db: &Db, name: &str) -> Option<String> {
    use sqlx::Row;
    match db {
        Db::Sqlite(p) => sqlx::query("SELECT workspace FROM experiments WHERE name = ?")
            .bind(name)
            .fetch_optional(p)
            .await
            .expect("query")
            .map(|r| r.get::<String, _>("workspace")),
        _ => unreachable!(),
    }
}

async fn count_experiments(db: &Db, name: &str) -> i64 {
    use sqlx::Row;
    match db {
        Db::Sqlite(p) => sqlx::query("SELECT COUNT(*) AS c FROM experiments WHERE name = ?")
            .bind(name)
            .fetch_one(p)
            .await
            .expect("count")
            .get("c"),
        _ => unreachable!(),
    }
}

async fn count_experiments_in_workspace(db: &Db, name: &str, workspace: &str) -> i64 {
    use sqlx::Row;
    match db {
        Db::Sqlite(p) => {
            sqlx::query("SELECT COUNT(*) AS c FROM experiments WHERE name = ? AND workspace = ?")
                .bind(name)
                .bind(workspace)
                .fetch_one(p)
                .await
                .expect("count")
                .get("c")
        }
        _ => unreachable!(),
    }
}

async fn count_runs(db: &Db, run_id: &str) -> i64 {
    use sqlx::Row;
    match db {
        Db::Sqlite(p) => sqlx::query("SELECT COUNT(*) AS c FROM runs WHERE run_uuid = ?")
            .bind(run_id)
            .fetch_one(p)
            .await
            .expect("count")
            .get("c"),
        _ => unreachable!(),
    }
}

const DEFAULT: &str = "default";

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_workspaces_returns_all() {
    let temp = TempDb::new("list");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        description: Some("Team A".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    s.create_workspace(Workspace::named("team-b"))
        .await
        .unwrap();

    let names: Vec<(String, Option<String>)> = s
        .list_workspaces()
        .await
        .unwrap()
        .into_iter()
        .map(|w| (w.name, w.description))
        .collect();
    assert!(names.contains(&("team-a".to_string(), Some("Team A".to_string()))));
    assert!(names.contains(&("team-b".to_string(), None)));
    assert!(names.iter().any(|(n, _)| n == DEFAULT));
    // ordered by name ascending
    let sorted: Vec<_> = {
        let mut v = names.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        v.sort();
        v
    };
    assert_eq!(
        names.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        sorted
    );
}

#[tokio::test]
async fn get_workspace_success() {
    let temp = TempDb::new("get");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        description: Some("Team A".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    let ws = s.get_workspace("team-a").await.unwrap();
    assert_eq!(ws.name, "team-a");
    assert_eq!(ws.description.as_deref(), Some("Team A"));
}

#[tokio::test]
async fn get_workspace_not_found() {
    let temp = TempDb::new("getnf");
    let s = store(&temp).await;
    let err = s.get_workspace("unknown").await.unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert!(err.message.contains("Workspace 'unknown' not found"));
}

#[tokio::test]
async fn create_workspace_persists_all_fields() {
    let temp = TempDb::new("create");
    let s = store(&temp).await;
    let created = s
        .create_workspace(Workspace {
            name: "team-a".into(),
            description: Some("Team A".into()),
            default_artifact_root: Some("s3://root/team-a".into()),
            trace_archival_location: Some("s3://archive/team-a".into()),
            trace_archival_retention: Some("30d".into()),
        })
        .await
        .unwrap();
    assert_eq!(created.name, "team-a");
    assert_eq!(
        created.default_artifact_root.as_deref(),
        Some("s3://root/team-a")
    );
    assert_eq!(
        created.trace_archival_location.as_deref(),
        Some("s3://archive/team-a")
    );
    assert_eq!(created.trace_archival_retention.as_deref(), Some("30d"));

    let fetched = s.get_workspace("team-a").await.unwrap();
    assert_eq!(fetched, created);
}

#[tokio::test]
async fn create_workspace_duplicate_raises() {
    let temp = TempDb::new("dup");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    let err = s
        .create_workspace(Workspace::named("team-a"))
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceAlreadyExists);
    assert!(err.message.contains("Workspace 'team-a' already exists."));
}

#[tokio::test]
async fn create_workspace_invalid_name_raises() {
    let temp = TempDb::new("badname");
    let s = store(&temp).await;
    let err = s
        .create_workspace(Workspace::named("Team-A"))
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err
        .message
        .contains("Workspace name 'Team-A' must match the pattern"));
}

#[tokio::test]
async fn create_workspace_invalid_trace_archival_location_raises() {
    let temp = TempDb::new("badloc");
    let s = store(&temp).await;
    let err = s
        .create_workspace(Workspace {
            name: "team-a".into(),
            trace_archival_location: Some("mlflow-artifacts:/archive/team-a".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err
        .message
        .contains("proxy-only `mlflow-artifacts:` scheme"));
}

#[tokio::test]
async fn create_workspace_invalid_retention_raises() {
    let temp = TempDb::new("badret");
    let s = store(&temp).await;
    let err = s
        .create_workspace(Workspace {
            name: "team-a".into(),
            trace_archival_retention: Some("thirty-days".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err.message.contains("Trace archival retention must"));
}

#[tokio::test]
async fn update_workspace_changes_description() {
    let temp = TempDb::new("upddesc");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        description: Some("old".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    let updated = s
        .update_workspace(Workspace {
            name: "team-a".into(),
            description: Some("new description".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(updated.description.as_deref(), Some("new description"));
    let fetched = s.get_workspace("team-a").await.unwrap();
    assert_eq!(fetched.description.as_deref(), Some("new description"));
}

#[tokio::test]
async fn update_workspace_sets_and_clears_artifact_root() {
    let temp = TempDb::new("updroot");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        default_artifact_root: Some("s3://bucket/team-a".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    // Empty string clears.
    let cleared = s
        .update_workspace(Workspace {
            name: "team-a".into(),
            default_artifact_root: Some(String::new()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(cleared.default_artifact_root, None);
    assert_eq!(
        s.get_workspace("team-a")
            .await
            .unwrap()
            .default_artifact_root,
        None
    );
}

#[tokio::test]
async fn update_workspace_clears_trace_archival_fields() {
    let temp = TempDb::new("updclear");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        trace_archival_location: Some("s3://archive/team-a".into()),
        trace_archival_retention: Some("30d".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    let cleared = s
        .update_workspace(Workspace {
            name: "team-a".into(),
            trace_archival_location: Some(String::new()),
            trace_archival_retention: Some(String::new()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(cleared.trace_archival_location, None);
    assert_eq!(cleared.trace_archival_retention, None);
}

#[tokio::test]
async fn update_workspace_invalid_retention_raises() {
    let temp = TempDb::new("updbadret");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    let err = s
        .update_workspace(Workspace {
            name: "team-a".into(),
            trace_archival_retention: Some("thirty-days".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidParameterValue);
    assert!(err.message.contains("Trace archival retention must"));
}

#[tokio::test]
async fn update_workspace_not_found() {
    let temp = TempDb::new("updnf");
    let s = store(&temp).await;
    let err = s
        .update_workspace(Workspace {
            name: "unknown".into(),
            description: Some("x".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert!(err.message.contains("Workspace 'unknown' not found"));
}

#[tokio::test]
async fn get_default_workspace() {
    let temp = TempDb::new("default");
    let s = store(&temp).await;
    let ws = s.get_default_workspace().await.unwrap();
    assert_eq!(ws.name, DEFAULT);
    assert!(ws.description.is_some());
}

// ---------------------------------------------------------------------------
// Delete modes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_empty_workspace() {
    let temp = TempDb::new("delempty");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    s.delete_workspace("team-a", WorkspaceDeletionMode::Restrict)
        .await
        .unwrap();
    let err = s.get_workspace("team-a").await.unwrap_err();
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn delete_default_rejected() {
    let temp = TempDb::new("deldefault");
    let s = store(&temp).await;
    let err = s
        .delete_workspace(DEFAULT, WorkspaceDeletionMode::Restrict)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidState);
    assert!(err
        .message
        .contains("Cannot delete the reserved 'default' workspace"));
}

#[tokio::test]
async fn delete_not_found() {
    let temp = TempDb::new("delnf");
    let s = store(&temp).await;
    let err = s
        .delete_workspace("unknown", WorkspaceDeletionMode::Restrict)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::ResourceDoesNotExist);
    assert!(err.message.contains("Workspace 'unknown' not found"));
}

#[tokio::test]
async fn delete_restrict_blocks_when_resources_exist() {
    let temp = TempDb::new("restrictblock");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    insert_experiment(s.db(), "exp-in-team-a", "team-a").await;

    let err = s
        .delete_workspace("team-a", WorkspaceDeletionMode::Restrict)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidState);
    assert!(err.message.contains("still contains"));

    // Workspace and resources survive.
    assert_eq!(s.get_workspace("team-a").await.unwrap().name, "team-a");
    assert_eq!(
        experiment_workspace(s.db(), "exp-in-team-a")
            .await
            .as_deref(),
        Some("team-a")
    );
}

#[tokio::test]
async fn delete_restrict_allows_empty() {
    let temp = TempDb::new("restrictempty");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    s.delete_workspace("team-a", WorkspaceDeletionMode::Restrict)
        .await
        .unwrap();
    assert!(s.get_workspace("team-a").await.is_err());
}

#[tokio::test]
async fn delete_cascade_removes_resources() {
    let temp = TempDb::new("cascade");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    insert_experiment(s.db(), "exp-in-team-a", "team-a").await;

    s.delete_workspace("team-a", WorkspaceDeletionMode::Cascade)
        .await
        .unwrap();

    assert_eq!(count_experiments(s.db(), "exp-in-team-a").await, 0);
    assert!(s.get_workspace("team-a").await.is_err());
}

#[tokio::test]
async fn delete_cascade_removes_experiment_with_runs() {
    let temp = TempDb::new("cascaderuns");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    insert_experiment_with_id(s.db(), 999, "exp-with-runs", "team-a").await;
    insert_run(s.db(), "run-in-team-a", 999).await;

    s.delete_workspace("team-a", WorkspaceDeletionMode::Cascade)
        .await
        .unwrap();

    assert_eq!(count_experiments(s.db(), "exp-with-runs").await, 0);
    assert_eq!(count_runs(s.db(), "run-in-team-a").await, 0);
}

#[tokio::test]
async fn delete_set_default_reassigns_resources() {
    let temp = TempDb::new("setdefault");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    insert_experiment(s.db(), "exp-in-team-a", "team-a").await;

    s.delete_workspace("team-a", WorkspaceDeletionMode::SetDefault)
        .await
        .unwrap();

    assert_eq!(
        experiment_workspace(s.db(), "exp-in-team-a")
            .await
            .as_deref(),
        Some(DEFAULT)
    );
    assert!(s.get_workspace("team-a").await.is_err());
}

#[tokio::test]
async fn delete_set_default_fails_on_naming_conflict() {
    let temp = TempDb::new("conflict");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    insert_experiment(s.db(), "shared-exp", "team-a").await;
    insert_experiment(s.db(), "shared-exp", DEFAULT).await;

    let err = s
        .delete_workspace("team-a", WorkspaceDeletionMode::SetDefault)
        .await
        .unwrap_err();
    assert_eq!(err.error_code, ErrorCode::InvalidState);
    assert!(err
        .message
        .contains("already exist in the default workspace"));

    // Workspace still exists (transaction rolled back).
    assert_eq!(s.get_workspace("team-a").await.unwrap().name, "team-a");
    // The transaction rolled back: the team-a copy of shared-exp was not
    // reassigned, so exactly one shared-exp row remains in each workspace.
    assert_eq!(
        count_experiments_in_workspace(s.db(), "shared-exp", "team-a").await,
        1
    );
    assert_eq!(
        count_experiments_in_workspace(s.db(), "shared-exp", DEFAULT).await,
        1
    );
}

// ---------------------------------------------------------------------------
// resolve_artifact_root + TTL cache
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resolve_artifact_root_returns_default() {
    let temp = TempDb::new("resroot");
    let s = store(&temp).await;
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), DEFAULT)
            .await
            .unwrap(),
        (Some("/default/path".to_string()), true)
    );
    s.create_workspace(Workspace::named("team-a"))
        .await
        .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-a")
            .await
            .unwrap(),
        (Some("/default/path".to_string()), true)
    );
}

#[tokio::test]
async fn resolve_artifact_root_prefers_override() {
    let temp = TempDb::new("resoverride");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        default_artifact_root: Some("s3://team-a-artifacts".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-a")
            .await
            .unwrap(),
        (Some("s3://team-a-artifacts".to_string()), false)
    );
}

#[tokio::test]
async fn resolve_artifact_root_cache_updates_on_override_change() {
    let temp = TempDb::new("cacheupd");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-cache"))
        .await
        .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-cache")
            .await
            .unwrap(),
        (Some("/default/path".to_string()), true)
    );
    s.update_workspace(Workspace {
        name: "team-cache".into(),
        default_artifact_root: Some("s3://cache/team".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-cache")
            .await
            .unwrap(),
        (Some("s3://cache/team".to_string()), false)
    );
}

#[tokio::test]
async fn resolve_artifact_root_cache_handles_delete_and_recreate() {
    let temp = TempDb::new("cachedelrecreate");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-cache".into(),
        default_artifact_root: Some("s3://cache/a".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-cache")
            .await
            .unwrap(),
        (Some("s3://cache/a".to_string()), false)
    );
    s.delete_workspace("team-cache", WorkspaceDeletionMode::Restrict)
        .await
        .unwrap();
    s.create_workspace(Workspace {
        name: "team-cache".into(),
        default_artifact_root: Some("s3://cache/b".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-cache")
            .await
            .unwrap(),
        (Some("s3://cache/b".to_string()), false)
    );
}

#[tokio::test]
async fn resolve_artifact_root_cache_clears_when_override_removed() {
    let temp = TempDb::new("cacheclear");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-cache".into(),
        default_artifact_root: Some("s3://cache/a".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-cache")
            .await
            .unwrap(),
        (Some("s3://cache/a".to_string()), false)
    );
    s.update_workspace(Workspace {
        name: "team-cache".into(),
        default_artifact_root: Some(String::new()),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        s.resolve_artifact_root(Some("/default/path"), "team-cache")
            .await
            .unwrap(),
        (Some("/default/path".to_string()), true)
    );
}

// ---------------------------------------------------------------------------
// resolve_trace_archival_config + TTL cache
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resolve_trace_archival_config_returns_defaults() {
    let temp = TempDb::new("archdefault");
    let s = store(&temp).await;
    let cfg = s
        .resolve_trace_archival_config("s3://archive/default", "30d", DEFAULT)
        .await
        .unwrap();
    assert_eq!(cfg.config.location.as_deref(), Some("s3://archive/default"));
    assert_eq!(cfg.config.retention.as_deref(), Some("30d"));
    assert!(cfg.append_workspace_prefix);
}

#[tokio::test]
async fn resolve_trace_archival_config_prefers_overrides() {
    let temp = TempDb::new("archoverride");
    let s = store(&temp).await;
    s.create_workspace(Workspace {
        name: "team-a".into(),
        trace_archival_location: Some("s3://archive/team-a".into()),
        trace_archival_retention: Some("14d".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    let cfg = s
        .resolve_trace_archival_config("s3://archive/default", "30d", "team-a")
        .await
        .unwrap();
    assert_eq!(cfg.config.location.as_deref(), Some("s3://archive/team-a"));
    assert_eq!(cfg.config.retention.as_deref(), Some("14d"));
    assert!(!cfg.append_workspace_prefix);
}

#[tokio::test]
async fn resolve_trace_archival_config_cache_updates_on_override_change() {
    let temp = TempDb::new("archcacheupd");
    let s = store(&temp).await;
    s.create_workspace(Workspace::named("team-cache"))
        .await
        .unwrap();
    let initial = s
        .resolve_trace_archival_config("s3://archive/default", "30d", "team-cache")
        .await
        .unwrap();
    assert_eq!(
        initial.config.location.as_deref(),
        Some("s3://archive/default")
    );
    assert!(initial.append_workspace_prefix);

    s.update_workspace(Workspace {
        name: "team-cache".into(),
        trace_archival_location: Some("s3://archive/team-cache".into()),
        trace_archival_retention: Some("7d".into()),
        ..Default::default()
    })
    .await
    .unwrap();
    let updated = s
        .resolve_trace_archival_config("s3://archive/default", "30d", "team-cache")
        .await
        .unwrap();
    assert_eq!(
        updated.config.location.as_deref(),
        Some("s3://archive/team-cache")
    );
    assert_eq!(updated.config.retention.as_deref(), Some("7d"));
    assert!(!updated.append_workspace_prefix);
}

// ---------------------------------------------------------------------------
// Single-tenant startup guard (T10.3)
// ---------------------------------------------------------------------------

const EXPERIMENT_CHECKS: &[(&str, &str)] = &[("experiments", "experiments")];

#[tokio::test]
async fn single_tenant_guard_passes_with_only_default_workspace_rows() {
    let temp = TempDb::new("guard-ok");
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect");
    insert_experiment(&db, "in-default", "default").await;
    // The fixture's rows are all in `default`; the guard must pass.
    verify_single_tenant_data(&db, EXPERIMENT_CHECKS)
        .await
        .expect("guard should pass with only default-workspace rows");
}

#[tokio::test]
async fn single_tenant_guard_rejects_non_default_experiments() {
    let temp = TempDb::new("guard-exp");
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect");
    insert_experiment(&db, "team-exp", "team-startup").await;
    let err = verify_single_tenant_data(&db, EXPERIMENT_CHECKS)
        .await
        .expect_err("guard should reject non-default rows");
    assert_eq!(err.error_code, mlflow_error::ErrorCode::InvalidState);
    assert_eq!(
        err.message,
        "Cannot disable workspaces because experiments exist outside the default workspace"
    );
}

#[tokio::test]
async fn single_tenant_guard_skips_missing_tables() {
    let temp = TempDb::new("guard-missing");
    let db = Db::connect(&temp.uri(), PoolConfig::default())
        .await
        .expect("connect");
    // A table that does not exist is skipped (no violation).
    verify_single_tenant_data(&db, &[("no_such_table", "widgets")])
        .await
        .expect("missing table is not a guard violation");
}
