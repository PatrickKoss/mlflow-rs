//! Assessment operations (plan T2.12), mirroring `create_assessment`,
//! `get_assessment`, `update_assessment`, `delete_assessment`, and
//! `_get_sql_assessment` in `mlflow/store/tracking/sqlalchemy_store.py` and
//! their workspace-aware overrides in
//! `mlflow/store/tracking/sqlalchemy_workspace_store.py`.
//!
//! ## ID generation
//!
//! `generate_assessment_id` (`mlflow/tracing/utils/__init__.py`): `a-<uuid4
//! hex>` (32 lowercase hex chars, no dashes) — see [`new_assessment_id`].
//!
//! ## Payload encoding (observable — read back through `traces/get`)
//!
//! `SqlAssessments.value`/`.error`/`.assessment_metadata` are `Text` columns
//! holding `json.dumps(...)` output:
//!
//! * `value`: `json.dumps(expectation.value)` / `json.dumps(feedback.value)` /
//!   `json.dumps({"issue_name": ...})` depending on assessment type.
//! * `error` (feedback only): `json.dumps(AssessmentError.to_dictionary())`,
//!   i.e. `{"error_code", "error_message", "stack_trace"}` — NULL when there is
//!   no error.
//! * `assessment_metadata`: `json.dumps(metadata)` when metadata is non-empty,
//!   else NULL (Python: `if assessment.metadata else None`, so `{}` also
//!   serializes as NULL — mirrored by [`metadata_json`]).
//!
//! Rust reproduces `serde_json::to_string` output for these instead of
//! reaching for Python's exact `json.dumps` whitespace; both are compact
//! (no extra whitespace) `{"key":"value"}`-style single-line JSON, which is
//! what matters for round-tripping (these are never diffed byte-for-byte
//! against Python's on-disk bytes, only read back through the same
//! `serde_json`/`json` decoders on each side).
//!
//! ## Workspace scoping (CRITICAL, plan §3.17)
//!
//! Assessments hang off `trace_id`, and traces hang off `experiment_id`, so
//! every method here re-validates the trace through a workspace-scoped
//! semi-join against `experiments`, exactly like
//! `WorkspaceAwareSqlAlchemyStore._validate_trace_accessible` /
//! `_get_sql_assessment`. `get_assessment`/`update_assessment` additionally
//! scope the assessment lookup itself to the workspace (via the same
//! semi-join), matching the workspace store's overridden `_get_sql_assessment`
//! — the base (single-tenant) `_get_sql_assessment` has no workspace concept
//! at all, so in single-tenant mode this is equivalent to an unscoped lookup.
//!
//! Note: `T2.10` (the traces store) is being implemented in parallel and will
//! own the canonical trace-existence/row-lookup helpers; the small
//! trace-scoping helpers here are private to this module to avoid merge
//! conflicts and should be reconciled with `T2.10`'s equivalents once both
//! land (see the final report's "known gaps" note).
//!
//! ## Override/supersede semantics
//!
//! Creating an assessment with `overrides = Some(id)` atomically marks the
//! overridden assessment `valid = false` (an `UPDATE ... WHERE assessment_id =
//! ? AND trace_id = ?`, matching Python's row-count check); zero rows updated
//! is a `RESOURCE_DOES_NOT_EXIST` error naming the missing override target.
//! Deleting an assessment that has `overrides` set restores the overridden
//! assessment's `valid` back to `true`. Both directions are exercised in
//! `tests/assessments_store.rs`.

use std::collections::BTreeMap;

use mlflow_error::MlflowError;
use uuid::Uuid;

use super::dbutil::{RowLike, Tx, Val};
use super::entities::{Assessment, AssessmentError, AssessmentSource, AssessmentValue};
use super::experiments::{internal, now_millis};
use super::TrackingStore;
use crate::dialect::Dialect;
use crate::schema::traces::{ASSESSMENTS, TRACE_INFO};

/// `generate_assessment_id`: `a-<uuid4 hex>`.
fn new_assessment_id() -> String {
    format!("a-{}", Uuid::new_v4().simple())
}

/// A newly created assessment's payload (the store fills in `assessment_id`,
/// `create_time_ms`/`last_update_time_ms` defaults, and `valid`).
#[derive(Debug, Clone)]
pub struct NewAssessment {
    pub trace_id: String,
    pub name: String,
    pub value: AssessmentValue,
    pub source: AssessmentSource,
    pub run_id: Option<String>,
    pub span_id: Option<String>,
    pub rationale: Option<String>,
    pub metadata: Option<BTreeMap<String, String>>,
    pub create_time_ms: Option<i64>,
    pub last_update_time_ms: Option<i64>,
    /// Caller-supplied assessment_id, if any (Python allows a caller-supplied
    /// id; a collision surfaces as a constraint-violation error, not silently
    /// overwritten).
    pub assessment_id: Option<String>,
    /// The assessment_id this one overrides/supersedes, if any.
    pub overrides: Option<String>,
}

/// The new value for a feedback assessment (mirrors Python's `FeedbackValue`:
/// `value` and `error` travel together as one object — supplying an updated
/// feedback value with no error clears any previously-recorded error, exactly
/// like passing `FeedbackValue(value=...)` with `error` defaulted to `None`).
#[derive(Debug, Clone)]
pub struct FeedbackUpdate {
    pub value_json: String,
    pub error: Option<AssessmentError>,
}

/// Partial update for `update_assessment`. Store-layer signature: explicit
/// optional fields (`name`, `expectation`, `feedback`, `rationale`,
/// `metadata`), matching Python's `update_assessment(...)` parameters
/// *exactly* — the HTTP layer's FieldMask (`assessment_name`, `expectation`,
/// `feedback`, `rationale`, `metadata`, `valid`) is translated into this
/// struct at the HTTP boundary (Phase 3/12), not here. Notably, **`valid` is
/// not settable through this store method** in Python either; it is only ever
/// flipped by the override/delete machinery in `create_assessment`/
/// `delete_assessment`. A FieldMask path of `valid` at the HTTP layer has no
/// store-level equivalent and is a known gap called out in the final report.
#[derive(Debug, Clone, Default)]
pub struct AssessmentUpdate {
    pub name: Option<String>,
    pub expectation_value_json: Option<String>,
    pub feedback: Option<FeedbackUpdate>,
    pub rationale: Option<String>,
    pub metadata: Option<BTreeMap<String, String>>,
}

/// Insert an assessment as part of an existing transaction. `start_trace`
/// uses the upsert form because Python's trace-conflict merge path calls
/// `session.merge` for embedded assessments; standalone create keeps strict
/// insert semantics so caller-supplied ID collisions remain errors.
pub(crate) async fn insert_assessment_tx(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    assessment: &NewAssessment,
    upsert: bool,
) -> Result<Assessment, sqlx::Error> {
    const COLUMNS: &[&str] = &[
        "assessment_id",
        "trace_id",
        "name",
        "assessment_type",
        "value",
        "error",
        "created_timestamp",
        "last_updated_timestamp",
        "source_type",
        "source_id",
        "run_id",
        "span_id",
        "rationale",
        "overrides",
        "valid",
        "assessment_metadata",
    ];
    const UPDATE_COLUMNS: &[&str] = &[
        "trace_id",
        "name",
        "assessment_type",
        "value",
        "error",
        "created_timestamp",
        "last_updated_timestamp",
        "source_type",
        "source_id",
        "run_id",
        "span_id",
        "rationale",
        "overrides",
        "valid",
        "assessment_metadata",
    ];

    let assessment_id = assessment
        .assessment_id
        .clone()
        .unwrap_or_else(new_assessment_id);
    let now = now_millis();
    let create_time_ms = assessment.create_time_ms.unwrap_or(now);
    let last_update_time_ms = assessment.last_update_time_ms.unwrap_or(now);
    let (assessment_type, value_json, error_json) = encode_value(&assessment.value);
    let assessment_metadata = metadata_json(assessment.metadata.as_ref());

    let sql = if upsert {
        dialect.upsert(&crate::dialect::UpsertSpec {
            table: ASSESSMENTS,
            columns: COLUMNS,
            pk_columns: &["assessment_id"],
            update_columns: UPDATE_COLUMNS,
            ..Default::default()
        })
    } else {
        let columns = COLUMNS
            .iter()
            .map(|column| dialect.quote_ident(column))
            .collect::<Vec<_>>()
            .join(", ");
        let values = (1..=COLUMNS.len())
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "INSERT INTO {} ({columns}) VALUES ({values})",
            dialect.quote_ident(ASSESSMENTS)
        )
    };
    tx.exec(
        &sql,
        &[
            Val::Text(assessment_id.clone()),
            Val::Text(assessment.trace_id.clone()),
            Val::Text(assessment.name.clone()),
            Val::Text(assessment_type.to_string()),
            Val::Text(value_json),
            Val::OptText(error_json),
            Val::Int(create_time_ms),
            Val::Int(last_update_time_ms),
            Val::Text(assessment.source.source_type.clone()),
            Val::OptText(assessment.source.source_id.clone()),
            Val::OptText(assessment.run_id.clone()),
            Val::OptText(assessment.span_id.clone()),
            Val::OptText(assessment.rationale.clone()),
            Val::OptText(assessment.overrides.clone()),
            Val::Bool(true),
            Val::OptText(assessment_metadata),
        ],
    )
    .await?;

    Ok(Assessment {
        assessment_id,
        trace_id: assessment.trace_id.clone(),
        name: assessment.name.clone(),
        value: assessment.value.clone(),
        source: assessment.source.clone(),
        run_id: assessment.run_id.clone(),
        span_id: assessment.span_id.clone(),
        rationale: assessment.rationale.clone(),
        metadata: assessment.metadata.clone(),
        create_time_ms,
        last_update_time_ms,
        overrides: assessment.overrides.clone(),
        valid: true,
    })
}

impl TrackingStore {
    /// `create_assessment`. Validates the trace is accessible in `workspace`,
    /// then (if `overrides` is set) marks the overridden assessment invalid,
    /// then inserts. A caller-supplied `assessment_id` that collides with an
    /// existing row surfaces as a generic constraint-violation `INTERNAL_ERROR`
    /// (Python: IntegrityError not attributable to the missing-trace FK).
    pub async fn create_assessment(
        &self,
        workspace: &str,
        assessment: NewAssessment,
    ) -> Result<Assessment, MlflowError> {
        self.validate_trace_accessible(workspace, &assessment.trace_id)
            .await?;

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        if let Some(overrides) = assessment.overrides.as_deref() {
            let sql = format!(
                "UPDATE {ASSESSMENTS} SET valid = {} \
                 WHERE trace_id = {} AND assessment_id = {}",
                Val::sql_bool(dialect, false),
                dialect.placeholder(1),
                dialect.placeholder(2),
            );
            let updated = tx
                .exec(
                    &sql,
                    &[
                        Val::Text(assessment.trace_id.clone()),
                        Val::Text(overrides.to_string()),
                    ],
                )
                .await
                .map_err(internal)?;
            if updated == 0 {
                // Mirrors Python: roll back (drop `tx`) and surface a clean
                // "not found" naming the missing override target.
                return Err(MlflowError::resource_does_not_exist(format!(
                    "Assessment with ID '{overrides}' not found for trace '{}'",
                    assessment.trace_id
                )));
            }
        }

        let insert_result = insert_assessment_tx(&mut tx, dialect, &assessment, false).await;

        // A missing trace (e.g. deleted between the accessibility check and
        // this insert) or a duplicate caller-supplied assessment_id both trip
        // constraints on flush. `validate_trace_accessible` above already
        // covers the ordinary "trace gone" case; this is the residual race +
        // duplicate-PK case Python's `except IntegrityError` handles.
        let created = match insert_result {
            Ok(created) => created,
            Err(_) => {
                drop(tx);
                if self
                    .validate_trace_accessible(workspace, &assessment.trace_id)
                    .await
                    .is_err()
                {
                    return Err(MlflowError::resource_does_not_exist(format!(
                        "Trace with ID '{}' not found. It may have been deleted.",
                        assessment.trace_id
                    )));
                }
                return Err(MlflowError::internal_error(format!(
                    "Failed to create assessment for trace '{}' due to a constraint violation.",
                    assessment.trace_id
                )));
            }
        };

        tx.commit().await.map_err(internal)?;

        Ok(created)
    }

    /// `get_assessment`.
    pub async fn get_assessment(
        &self,
        workspace: &str,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<Assessment, MlflowError> {
        self.fetch_assessment_row(workspace, trace_id, assessment_id)
            .await
            .map(row_to_assessment)
    }

    /// `update_assessment`. Only `source` and `span_id` are immutable;
    /// `last_update_time_ms` always advances to now; `metadata` is merged
    /// (existing keys kept, `update.metadata` keys take precedence) rather
    /// than replaced. Changing assessment type (feedback <-> expectation) is
    /// rejected with the exact Python error wording.
    pub async fn update_assessment(
        &self,
        workspace: &str,
        trace_id: &str,
        assessment_id: &str,
        update: AssessmentUpdate,
    ) -> Result<Assessment, MlflowError> {
        let existing_row = self
            .fetch_assessment_row(workspace, trace_id, assessment_id)
            .await?;
        let existing = row_to_assessment(existing_row);

        if update.expectation_value_json.is_some() && update.feedback.is_some() {
            return Err(MlflowError::invalid_parameter_value(
                "Cannot specify both `expectation` and `feedback` parameters.",
            ));
        }
        if update.expectation_value_json.is_some()
            && !matches!(existing.value, AssessmentValue::Expectation { .. })
        {
            return Err(MlflowError::invalid_parameter_value(
                "Cannot update expectation value on a Feedback assessment.",
            ));
        }
        if update.feedback.is_some() && !matches!(existing.value, AssessmentValue::Feedback { .. })
        {
            return Err(MlflowError::invalid_parameter_value(
                "Cannot update feedback value on an Expectation assessment.",
            ));
        }

        let merged_metadata = merge_metadata(existing.metadata.as_ref(), update.metadata.as_ref());
        let updated_timestamp = now_millis();

        let new_name = update.name.clone().unwrap_or_else(|| existing.name.clone());
        let new_rationale = update
            .rationale
            .clone()
            .or_else(|| existing.rationale.clone());

        let new_value = match &existing.value {
            AssessmentValue::Expectation { value_json } => AssessmentValue::Expectation {
                value_json: update
                    .expectation_value_json
                    .clone()
                    .unwrap_or_else(|| value_json.clone()),
            },
            AssessmentValue::Feedback { value_json, error } => match &update.feedback {
                // `feedback` supplied: value and error travel together, as one
                // `FeedbackValue` (Python: `new_value, new_error = feedback.value,
                // feedback.error`) — an update with no error clears a prior one.
                Some(f) => AssessmentValue::Feedback {
                    value_json: f.value_json.clone(),
                    error: f.error.clone(),
                },
                None => AssessmentValue::Feedback {
                    value_json: value_json.clone(),
                    error: error.clone(),
                },
            },
            // `update_assessment` never targets issue-type assessments in
            // Python (no code path constructs one); preserved as-is.
            AssessmentValue::Issue { issue_name } => AssessmentValue::Issue {
                issue_name: issue_name.clone(),
            },
        };

        let (_, value_json, error_json) = encode_value(&new_value);
        let metadata_json_str = metadata_json(merged_metadata.as_ref());

        let dialect = self.db().dialect();
        let sql = format!(
            "UPDATE {ASSESSMENTS} SET name = {}, value = {}, error = {}, \
             last_updated_timestamp = {}, rationale = {}, assessment_metadata = {} \
             WHERE trace_id = {} AND assessment_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
            dialect.placeholder(5),
            dialect.placeholder(6),
            dialect.placeholder(7),
            dialect.placeholder(8),
        );
        self.db()
            .exec(
                &sql,
                &[
                    Val::Text(new_name.clone()),
                    Val::Text(value_json),
                    Val::OptText(error_json),
                    Val::Int(updated_timestamp),
                    Val::OptText(new_rationale.clone()),
                    Val::OptText(metadata_json_str),
                    Val::Text(trace_id.to_string()),
                    Val::Text(assessment_id.to_string()),
                ],
            )
            .await
            .map_err(internal)?;

        Ok(Assessment {
            assessment_id: existing.assessment_id,
            trace_id: existing.trace_id,
            name: new_name,
            value: new_value,
            source: existing.source,
            run_id: existing.run_id,
            span_id: existing.span_id,
            rationale: new_rationale,
            metadata: merged_metadata,
            create_time_ms: existing.create_time_ms,
            last_update_time_ms: updated_timestamp,
            overrides: existing.overrides,
            valid: existing.valid,
        })
    }

    /// `delete_assessment`. Idempotent: deleting a missing assessment (or one
    /// under a missing/inaccessible trace) is a silent no-op, matching
    /// Python's `if assessment_to_delete is None: return`. If the deleted
    /// assessment had `overrides` set, restores the overridden assessment's
    /// `valid` back to `true`.
    pub async fn delete_assessment(
        &self,
        workspace: &str,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<(), MlflowError> {
        // Python's delete_assessment calls `_validate_trace_accessible`
        // (raising if the trace itself is gone/inaccessible) rather than the
        // assessment-scoped `_get_sql_assessment`; a missing *assessment* on
        // an accessible trace is the idempotent no-op case.
        if self
            .validate_trace_accessible(workspace, trace_id)
            .await
            .is_err()
        {
            return Ok(());
        }

        let dialect = self.db().dialect();
        let row = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT overrides FROM {ASSESSMENTS} WHERE trace_id = {} AND assessment_id = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                ),
                &[
                    Val::Text(trace_id.to_string()),
                    Val::Text(assessment_id.to_string()),
                ],
                |r| r.get_opt_string("overrides"),
            )
            .await
            .map_err(internal)?;

        let Some(overrides) = row else {
            return Ok(());
        };

        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        if let Some(overridden_id) = overrides {
            let restore_sql = format!(
                "UPDATE {ASSESSMENTS} SET valid = {} WHERE assessment_id = {}",
                Val::sql_bool(dialect, true),
                dialect.placeholder(1),
            );
            tx.exec(&restore_sql, &[Val::Text(overridden_id)])
                .await
                .map_err(internal)?;
        }

        let delete_sql = format!(
            "DELETE FROM {ASSESSMENTS} WHERE trace_id = {} AND assessment_id = {}",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        tx.exec(
            &delete_sql,
            &[
                Val::Text(trace_id.to_string()),
                Val::Text(assessment_id.to_string()),
            ],
        )
        .await
        .map_err(internal)?;

        tx.commit().await.map_err(internal)
    }

    // ---- internal helpers ----

    /// `_get_sql_assessment`: fetch one assessment row, scoped to the
    /// workspace via the trace it belongs to. Distinguishes "trace not found"
    /// from "assessment not found for [an accessible] trace", matching
    /// Python's two-branch error handling exactly.
    async fn fetch_assessment_row(
        &self,
        workspace: &str,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<AssessmentRow, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT a.{cols} FROM {ASSESSMENTS} a \
             JOIN {TRACE_INFO} t ON a.trace_id = t.request_id \
             WHERE a.trace_id = {} AND a.assessment_id = {} AND t.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            cols = AssessmentRow::select_cols_prefixed(),
        );
        let row = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(trace_id.to_string()),
                    Val::Text(assessment_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                AssessmentRow::from_row,
            )
            .await
            .map_err(internal)?;

        if let Some(row) = row {
            return Ok(row);
        }

        self.validate_trace_accessible(workspace, trace_id).await?;
        Err(MlflowError::resource_does_not_exist(format!(
            "Assessment with ID '{assessment_id}' not found for trace '{trace_id}'"
        )))
    }
}

impl Val {
    /// A dialect-safe boolean SQL literal for a `Boolean` column (SQLite has
    /// no native bool type but accepts `0`/`1`; Postgres/MySQL accept
    /// `TRUE`/`FALSE`, which SQLite also parses as synonyms for `1`/`0`).
    fn sql_bool(_dialect: Dialect, value: bool) -> &'static str {
        if value {
            "TRUE"
        } else {
            "FALSE"
        }
    }
}

/// The physical `assessments` row read back for entity assembly.
struct AssessmentRow {
    assessment_id: String,
    trace_id: String,
    name: String,
    assessment_type: String,
    value: String,
    error: Option<String>,
    created_timestamp: i64,
    last_updated_timestamp: i64,
    source_type: String,
    source_id: Option<String>,
    run_id: Option<String>,
    span_id: Option<String>,
    rationale: Option<String>,
    overrides: Option<String>,
    valid: bool,
    assessment_metadata: Option<String>,
}

impl AssessmentRow {
    const SELECT_COLS: &'static [&'static str] = &[
        "assessment_id",
        "trace_id",
        "name",
        "assessment_type",
        "value",
        "error",
        "created_timestamp",
        "last_updated_timestamp",
        "source_type",
        "source_id",
        "run_id",
        "span_id",
        "rationale",
        "overrides",
        "valid",
        "assessment_metadata",
    ];

    /// The `assessments` columns, comma-joined for interpolation after an
    /// `a.` table alias prefix in the `SELECT a.<cols>` clause.
    fn select_cols_prefixed() -> String {
        Self::SELECT_COLS.join(", a.")
    }

    fn from_row(r: &dyn RowLike) -> Result<Self, sqlx::Error> {
        Ok(AssessmentRow {
            assessment_id: r.get_string("assessment_id")?,
            trace_id: r.get_string("trace_id")?,
            name: r.get_string("name")?,
            assessment_type: r.get_string("assessment_type")?,
            value: r.get_string("value")?,
            error: r.get_opt_string("error")?,
            created_timestamp: r.get_i64("created_timestamp")?,
            last_updated_timestamp: r.get_i64("last_updated_timestamp")?,
            source_type: r.get_string("source_type")?,
            source_id: r.get_opt_string("source_id")?,
            run_id: r.get_opt_string("run_id")?,
            span_id: r.get_opt_string("span_id")?,
            rationale: r.get_opt_string("rationale")?,
            overrides: r.get_opt_string("overrides")?,
            valid: r.get_bool("valid")?,
            assessment_metadata: r.get_opt_string("assessment_metadata")?,
        })
    }
}

/// `SqlAssessments.to_mlflow_entity`.
fn row_to_assessment(row: AssessmentRow) -> Assessment {
    let value = match row.assessment_type.as_str() {
        "feedback" => AssessmentValue::Feedback {
            value_json: row.value,
            error: row.error.map(|e| {
                serde_json::from_str::<AssessmentError>(&e).unwrap_or(AssessmentError {
                    error_code: "UNKNOWN".to_string(),
                    error_message: None,
                    stack_trace: None,
                })
            }),
        },
        "expectation" => AssessmentValue::Expectation {
            value_json: row.value,
        },
        _ => {
            // "issue" or anything else: preserve the raw JSON's issue_name if
            // present (mirrors `parsed_value.get("issue_name")`), defaulting
            // to an empty string rather than panicking on malformed data.
            let issue_name = serde_json::from_str::<serde_json::Value>(&row.value)
                .ok()
                .and_then(|v| {
                    v.get("issue_name")
                        .and_then(|n| n.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_default();
            AssessmentValue::Issue { issue_name }
        }
    };

    Assessment {
        assessment_id: row.assessment_id,
        trace_id: row.trace_id,
        name: row.name,
        value,
        source: AssessmentSource {
            source_type: row.source_type,
            source_id: row.source_id,
        },
        run_id: row.run_id,
        span_id: row.span_id,
        rationale: row.rationale,
        metadata: row
            .assessment_metadata
            .and_then(|m| serde_json::from_str(&m).ok()),
        create_time_ms: row.created_timestamp,
        last_update_time_ms: row.last_updated_timestamp,
        overrides: row.overrides,
        valid: row.valid,
    }
}

/// `SqlAssessments.from_mlflow_entity`'s value/error JSON encoding, keyed on
/// the discriminated [`AssessmentValue`]. Returns `(assessment_type,
/// value_json, error_json)`.
fn encode_value(value: &AssessmentValue) -> (&'static str, String, Option<String>) {
    match value {
        AssessmentValue::Expectation { value_json } => ("expectation", value_json.clone(), None),
        AssessmentValue::Feedback { value_json, error } => (
            "feedback",
            value_json.clone(),
            error
                .as_ref()
                .map(|e| serde_json::to_string(e).expect("AssessmentError serializes")),
        ),
        AssessmentValue::Issue { issue_name } => (
            "issue",
            serde_json::json!({ "issue_name": issue_name }).to_string(),
            None,
        ),
    }
}

/// `json.dumps(metadata) if metadata else None`: an empty map serializes as
/// SQL NULL, not `"{}"`, matching Python's truthiness check on the dict.
fn metadata_json(metadata: Option<&BTreeMap<String, String>>) -> Option<String> {
    match metadata {
        Some(m) if !m.is_empty() => {
            Some(serde_json::to_string(m).expect("metadata map serializes"))
        }
        _ => None,
    }
}

/// Merge existing and incoming metadata, incoming taking precedence
/// (`update_assessment`'s `merged_metadata`). Returns `None` when both sides
/// are empty/absent (so it round-trips through [`metadata_json`] as NULL).
fn merge_metadata(
    existing: Option<&BTreeMap<String, String>>,
    incoming: Option<&BTreeMap<String, String>>,
) -> Option<BTreeMap<String, String>> {
    if existing.is_none_or(BTreeMap::is_empty) && incoming.is_none_or(BTreeMap::is_empty) {
        return None;
    }
    let mut merged = existing.cloned().unwrap_or_default();
    if let Some(incoming) = incoming {
        merged.extend(incoming.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    Some(merged)
}
