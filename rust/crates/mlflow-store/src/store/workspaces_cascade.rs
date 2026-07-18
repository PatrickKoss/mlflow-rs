// Included into `workspaces::cascade` via `include!`. See that module for the
// `use` imports (Dialect, Tx, Val, MlflowError, internal, DEFAULT_WORKSPACE_NAME).
//
// Mirrors `_WORKSPACE_ROOT_MODELS` and the SQLAlchemy ORM `cascade="all"` /
// `cascade="all, delete-orphan"` trees from
// `mlflow/store/workspace/sqlalchemy_store.py` +
// `mlflow/store/{tracking,model_registry}/dbmodels/models.py`.

/// The workspace-scoped root tables, in the exact order of
/// `_WORKSPACE_ROOT_MODELS`. RESTRICT counts rows here; SET_DEFAULT reassigns the
/// `workspace` column here; CASCADE deletes these rows (plus their cascade
/// children).
pub(crate) const ROOT_TABLES: &[&str] = &[
    "registered_models",
    "experiments",
    "evaluation_datasets",
    "webhooks",
    "secrets",
    "endpoints",
    "model_definitions",
    "budget_policies",
    "guardrails",
    "jobs",
];

/// Root tables that carry a unique `name` column, used by the SET_DEFAULT
/// preflight conflict check (`_check_set_default_conflicts` only inspects models
/// with a `name` attribute). `experiments` and `registered_models` are the
/// workspace-scoped roots whose SQLAlchemy model exposes `name`.
const NAMED_ROOT_TABLES: &[&str] = &["registered_models", "experiments"];

/// RESTRICT: refuse to delete if any root table still contains a row in the
/// workspace. Byte-matches Python's error message and table iteration order.
pub(crate) async fn restrict(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
) -> Result<(), MlflowError> {
    for table in ROOT_TABLES {
        let sql = format!(
            "SELECT COUNT(*) AS c FROM {table} WHERE workspace = {}",
            dialect.placeholder(1)
        );
        let count = tx
            .fetch_all(&sql, &[Val::Text(workspace.to_string())], |r| r.get_i64("c"))
            .await
            .map_err(internal)?
            .into_iter()
            .next()
            .unwrap_or(0);
        if count > 0 {
            return Err(MlflowError::invalid_state(format!(
                "Cannot delete workspace '{workspace}': table '{table}' still contains {count} \
                 resource(s). Remove or reassign them before deleting the workspace."
            )));
        }
    }
    Ok(())
}

/// SET_DEFAULT: reassign every root row's `workspace` column to `default`, after
/// a preflight name-conflict check that mirrors `_check_set_default_conflicts`.
pub(crate) async fn set_default(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
) -> Result<(), MlflowError> {
    check_set_default_conflicts(tx, dialect, workspace).await?;
    for table in ROOT_TABLES {
        let sql = format!(
            "UPDATE {table} SET workspace = {} WHERE workspace = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        tx.exec(
            &sql,
            &[
                Val::Text(DEFAULT_WORKSPACE_NAME.to_string()),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
    }
    Ok(())
}

/// Preflight: report all name conflicts that reassigning `workspace` -> `default`
/// would create, mirroring `_check_set_default_conflicts`. Conflicts are ordered
/// by root-table iteration then by the overlapping name, matching Python's list
/// construction (`f"  - {tablename}: {name!r}"`).
async fn check_set_default_conflicts(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
) -> Result<(), MlflowError> {
    let mut conflicts: Vec<String> = Vec::new();
    for table in NAMED_ROOT_TABLES {
        let sql = format!(
            "SELECT name FROM {table} WHERE workspace = {} AND name IN \
             (SELECT name FROM {table} WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        let names: Vec<String> = tx
            .fetch_all(
                &sql,
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(DEFAULT_WORKSPACE_NAME.to_string()),
                ],
                |r| r.get_string("name"),
            )
            .await
            .map_err(internal)?;
        for name in names {
            // Python renders the name via `{name!r}` (repr). For plain workspace
            // names this is `'name'` — single-quoted.
            conflicts.push(format!("  - {table}: '{name}'"));
        }
    }
    if !conflicts.is_empty() {
        let details = conflicts.join("\n");
        return Err(MlflowError::invalid_state(format!(
            "Cannot reassign resources from workspace '{workspace}' to \
             '{DEFAULT_WORKSPACE_NAME}': the following names already exist in the \
             default workspace and would cause conflicts:\n{details}\n\
             Rename or remove the conflicting resources before retrying."
        )));
    }
    Ok(())
}

/// CASCADE: delete every root row in the workspace plus its cascade descendants.
///
/// SQLAlchemy `session.delete(root)` fires ORM relationship cascades; this
/// reproduces the equivalent set of rows with explicit child-before-parent
/// `DELETE`s. Every workspace-scoped root table also carries the `workspace`
/// column, but children are keyed by their FK to the root, so we scope children
/// by a subquery selecting the in-workspace parent ids.
///
/// Quirks reproduced from the Python cascade (see the ORM relationship map):
/// * `inputs` / `input_tags` / `entity_associations` are NOT deleted — Python
///   leaves them orphaned for `mlflow gc`.
/// * `endpoints.experiment_id`, `secrets`<-`model_definitions.secret_id`,
///   `guardrails.action_endpoint_id`, `issues.source_run_id` are SET NULL FKs,
///   not deletes; those parents are themselves deleted by their own root-table
///   pass so we never rely on the null-out.
pub(crate) async fn cascade(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
) -> Result<(), MlflowError> {
    // 1. registered_models family.
    cascade_registered_models(tx, dialect, workspace).await?;
    // 2. experiments family (the deepest tree).
    cascade_experiments(tx, dialect, workspace).await?;
    // 3. evaluation_datasets.
    delete_children(
        tx,
        dialect,
        &["evaluation_dataset_records", "evaluation_dataset_tags"],
        "dataset_id",
        "evaluation_datasets",
        "dataset_id",
        workspace,
    )
    .await?;
    delete_root(tx, dialect, "evaluation_datasets", workspace).await?;
    // 4. webhooks.
    delete_children(
        tx,
        dialect,
        &["webhook_events"],
        "webhook_id",
        "webhooks",
        "webhook_id",
        workspace,
    )
    .await?;
    delete_root(tx, dialect, "webhooks", workspace).await?;
    // 5. secrets (no cascade children; model_definitions.secret_id is SET NULL).
    delete_root(tx, dialect, "secrets", workspace).await?;
    // 6. endpoints family.
    delete_children(
        tx,
        dialect,
        &[
            "endpoint_model_mappings",
            "endpoint_tags",
            "endpoint_bindings",
            "guardrail_configs",
        ],
        "endpoint_id",
        "endpoints",
        "endpoint_id",
        workspace,
    )
    .await?;
    delete_root(tx, dialect, "endpoints", workspace).await?;
    // 7. model_definitions (endpoint_model_mappings already gone in step 6).
    delete_root(tx, dialect, "model_definitions", workspace).await?;
    // 8. budget_policies.
    delete_root(tx, dialect, "budget_policies", workspace).await?;
    // 9. guardrails family (guardrail_configs may already be partly gone; idempotent).
    delete_children(
        tx,
        dialect,
        &["guardrail_configs"],
        "guardrail_id",
        "guardrails",
        "guardrail_id",
        workspace,
    )
    .await?;
    delete_root(tx, dialect, "guardrails", workspace).await?;
    // 10. jobs.
    delete_root(tx, dialect, "jobs", workspace).await?;
    Ok(())
}

async fn cascade_registered_models(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
) -> Result<(), MlflowError> {
    // All registered-model children carry the `workspace` column, so we can
    // delete by workspace directly (matches the (workspace, name[, version]) PKs).
    for table in [
        "model_version_tags",
        "registered_model_tags",
        "registered_model_aliases",
        "model_versions",
    ] {
        delete_by_workspace(tx, dialect, table, workspace).await?;
    }
    delete_root(tx, dialect, "registered_models", workspace).await?;
    Ok(())
}

async fn cascade_experiments(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
) -> Result<(), MlflowError> {
    // Children keyed by run_uuid of runs in in-workspace experiments.
    let runs_subq = format!(
        "SELECT run_uuid FROM runs WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    for table in ["metrics", "latest_metrics", "params", "tags"] {
        let sql = format!("DELETE FROM {table} WHERE run_uuid IN ({runs_subq})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }

    // Trace tree, keyed by request_id of trace_info in in-workspace experiments.
    let traces_subq = format!(
        "SELECT request_id FROM trace_info WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    // span_metrics/span_attributes are keyed by (trace_id, span_id) -> spans;
    // delete via spans. This explicit cleanup also supports deployments where
    // database-level FK enforcement is disabled.
    let spans_by_trace = format!("SELECT trace_id FROM spans WHERE trace_id IN ({traces_subq})");
    for table in ["span_metrics", "span_attributes"] {
        let sql = format!("DELETE FROM {table} WHERE trace_id IN ({spans_by_trace})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }
    for (table, col) in [
        ("spans", "trace_id"),
        ("trace_tags", "request_id"),
        ("trace_request_metadata", "request_id"),
        ("trace_metrics", "request_id"),
        ("assessments", "trace_id"),
    ] {
        let sql = format!("DELETE FROM {table} WHERE {col} IN ({traces_subq})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }
    let ti_sql = format!(
        "DELETE FROM trace_info WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    tx.exec(&ti_sql, &[Val::Text(workspace.to_string())])
        .await
        .map_err(internal)?;

    // Scorers tree.
    let scorers_subq = format!(
        "SELECT scorer_id FROM scorers WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    for table in ["online_scoring_configs", "scorer_versions"] {
        let sql = format!("DELETE FROM {table} WHERE scorer_id IN ({scorers_subq})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }
    delete_by_experiment(tx, dialect, "scorers", workspace).await?;

    // Logged models tree.
    let models_subq = format!(
        "SELECT model_id FROM logged_models WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    for table in ["logged_model_metrics", "logged_model_params", "logged_model_tags"] {
        let sql = format!("DELETE FROM {table} WHERE model_id IN ({models_subq})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }
    delete_by_experiment(tx, dialect, "logged_models", workspace).await?;

    // Review-queue + issues + label schemas tree.
    let queues_subq = format!(
        "SELECT queue_id FROM review_queues WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    for table in [
        "review_queue_users",
        "review_queue_items",
        "review_queue_label_schemas",
    ] {
        let sql = format!("DELETE FROM {table} WHERE queue_id IN ({queues_subq})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }
    delete_by_experiment(tx, dialect, "review_queues", workspace).await?;
    delete_by_experiment(tx, dialect, "issues", workspace).await?;
    delete_by_experiment(tx, dialect, "label_schemas", workspace).await?;

    // datasets (SqlDataset) — DB-cascade in Python, deleted explicitly here.
    delete_by_experiment(tx, dialect, "datasets", workspace).await?;

    // runs + experiment_tags (ORM-cascaded).
    delete_by_experiment(tx, dialect, "runs", workspace).await?;
    delete_by_experiment(tx, dialect, "experiment_tags", workspace).await?;

    delete_root(tx, dialect, "experiments", workspace).await?;
    Ok(())
}

/// Delete `table` rows whose `experiment_id` belongs to an in-workspace
/// experiment.
async fn delete_by_experiment(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    table: &str,
    workspace: &str,
) -> Result<(), MlflowError> {
    let sql = format!(
        "DELETE FROM {table} WHERE experiment_id IN \
         (SELECT experiment_id FROM experiments WHERE workspace = {})",
        dialect.placeholder(1)
    );
    tx.exec(&sql, &[Val::Text(workspace.to_string())])
        .await
        .map_err(internal)?;
    Ok(())
}

/// Delete `children` whose `child_fk` matches a parent id selected from
/// `parent_table.parent_pk` for the given workspace.
async fn delete_children(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    children: &[&str],
    child_fk: &str,
    parent_table: &str,
    parent_pk: &str,
    workspace: &str,
) -> Result<(), MlflowError> {
    let parent_subq = format!(
        "SELECT {parent_pk} FROM {parent_table} WHERE workspace = {}",
        dialect.placeholder(1)
    );
    for table in children {
        let sql = format!("DELETE FROM {table} WHERE {child_fk} IN ({parent_subq})");
        tx.exec(&sql, &[Val::Text(workspace.to_string())])
            .await
            .map_err(internal)?;
    }
    Ok(())
}

/// Delete every row of a table directly by its `workspace` column.
async fn delete_by_workspace(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    table: &str,
    workspace: &str,
) -> Result<(), MlflowError> {
    let sql = format!(
        "DELETE FROM {table} WHERE workspace = {}",
        dialect.placeholder(1)
    );
    tx.exec(&sql, &[Val::Text(workspace.to_string())])
        .await
        .map_err(internal)?;
    Ok(())
}

/// Delete a root table's rows for the workspace.
async fn delete_root(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    table: &str,
    workspace: &str,
) -> Result<(), MlflowError> {
    delete_by_workspace(tx, dialect, table, workspace).await
}
