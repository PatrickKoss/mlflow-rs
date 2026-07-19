//! Database half of trace archival.
//!
//! Object-store I/O and OTLP encoding live in `mlflow-server`; this module owns
//! workspace-scoped candidate selection and the generation-guarded finalize
//! transaction so a scheduler pass can never archive a stale span snapshot.

use std::collections::{HashMap, HashSet};

use mlflow_error::MlflowError;
use serde_json::Value;

use super::dbutil::{Tx, Val};
use super::entities::{
    StoredSpan, TraceInfo, SPANS_LOCATION_ARCHIVE_REPO, SPANS_LOCATION_TRACKING_STORE,
    TRACE_EXPERIMENT_TAG_ARCHIVAL_RETENTION, TRACE_EXPERIMENT_TAG_ARCHIVE_NOW,
    TRACE_TAG_ARCHIVAL_FAILURE, TRACE_TAG_ARCHIVE_LOCATION, TRACE_TAG_SPANS_LOCATION,
};
use super::spans::load_spans_for_traces;
use super::traces::map_db_err_pub;
use super::TrackingStore;
use crate::dialect::Dialect;
use crate::schema::traces::{SPANS, SPAN_ATTRIBUTES, TRACE_INFO, TRACE_TAGS};

const MINUTE_MILLIS: i64 = 60_000;
const HOUR_MILLIS: i64 = 60 * MINUTE_MILLIS;
const DAY_MILLIS: i64 = 24 * HOUR_MILLIS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceArchiveCandidate {
    pub trace_id: String,
    pub experiment_id: String,
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceArchivalData {
    pub trace_info: TraceInfo,
    pub db_payload_generation: i64,
    pub spans: Vec<StoredSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveNowRequest {
    pub experiment_id: String,
    pub raw_value: String,
    pub older_than_millis: Option<i64>,
}

impl TrackingStore {
    /// Select Python-equivalent candidates for one workspace and one pass.
    /// Archive-now candidates have priority; both groups retain deterministic
    /// `(timestamp_ms, trace_id)` ordering and share the pass budget.
    pub async fn plan_trace_archival(
        &self,
        workspace: &str,
        now_millis: i64,
        broader_retention: &str,
        long_retention_allowlist: &HashSet<String>,
        max_traces_per_pass: Option<usize>,
    ) -> Result<(Vec<ArchiveNowRequest>, Vec<TraceArchiveCandidate>), MlflowError> {
        let broader_ms = parse_retention_millis(broader_retention).ok_or_else(|| {
            MlflowError::invalid_parameter_value(
                "Trace archival retention must be in the form `<int><unit>`, where unit is one of 'm', 'h', or 'd'.",
            )
        })?;
        let experiments = self.archival_experiments(workspace).await?;
        let mut archive_now_requests = Vec::new();
        let mut archive_now_cutoffs: HashMap<Option<i64>, Vec<i64>> = HashMap::new();
        let mut regular_cutoffs: HashMap<Option<i64>, Vec<i64>> = HashMap::new();

        for (experiment_id, tags) in experiments {
            let archive_now = tags.get(TRACE_EXPERIMENT_TAG_ARCHIVE_NOW).and_then(|raw| {
                parse_archive_now(raw).map(|older_than_millis| (raw, older_than_millis))
            });
            if let Some((raw, older_than_millis)) = archive_now {
                archive_now_requests.push(ArchiveNowRequest {
                    experiment_id: experiment_id.to_string(),
                    raw_value: raw.clone(),
                    older_than_millis,
                });
                archive_now_cutoffs
                    .entry(older_than_millis.map(|duration| now_millis - duration))
                    .or_default()
                    .push(experiment_id);
            }

            let retention_ms = effective_retention_millis(
                experiment_id,
                &tags,
                broader_ms,
                long_retention_allowlist,
            );
            let archive_now_covers_retention = archive_now
                .map(|(_, duration)| duration.is_none_or(|value| value <= retention_ms))
                .unwrap_or(false);
            if retention_ms != 0 && !archive_now_covers_retention {
                regular_cutoffs
                    .entry(Some(now_millis - retention_ms))
                    .or_default()
                    .push(experiment_id);
            }
        }

        let mut urgent = self
            .collect_candidates(workspace, &archive_now_cutoffs, max_traces_per_pass)
            .await?;
        let regular = if max_traces_per_pass.is_none_or(|limit| urgent.len() < limit) {
            self.collect_candidates(workspace, &regular_cutoffs, max_traces_per_pass)
                .await?
        } else {
            Vec::new()
        };
        let mut seen = HashSet::new();
        urgent.extend(regular);
        urgent.retain(|candidate| seen.insert(candidate.trace_id.clone()));
        if let Some(limit) = max_traces_per_pass {
            urgent.truncate(limit);
        }
        Ok((archive_now_requests, urgent))
    }

    /// Reload the complete DB-backed snapshot immediately before encoding.
    pub async fn load_trace_archival_data(
        &self,
        workspace: &str,
        trace_id: &str,
    ) -> Result<Option<TraceArchivalData>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT ti.db_payload_generation FROM {TRACE_INFO} ti WHERE ti.request_id = {} \
             AND ti.experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let generation = self
            .db()
            .fetch_optional(
                &sql,
                &[
                    Val::Text(trace_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_int("db_payload_generation"),
            )
            .await
            .map_err(map_db_err_pub)?;
        let Some(db_payload_generation) = generation else {
            return Ok(None);
        };
        let trace_info = self.get_trace_info(workspace, trace_id).await?;
        if !trace_actionable(&trace_info) {
            return Ok(None);
        }
        let ids = [trace_id.to_string()];
        let mut spans = load_spans_for_traces(self, &ids).await?;
        let spans = spans.remove(trace_id).unwrap_or_default();
        if spans.is_empty() {
            return Ok(None);
        }
        Ok(Some(TraceArchivalData {
            trace_info,
            db_payload_generation,
            spans,
        }))
    }

    /// Atomically publish an uploaded archive payload. A false result means the
    /// snapshot became stale; no DB state is changed and the caller must delete
    /// its unreferenced upload.
    pub async fn finalize_archived_trace(
        &self,
        workspace: &str,
        trace_id: &str,
        artifact_uri: &str,
        db_payload_generation: i64,
    ) -> Result<bool, MlflowError> {
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(map_db_err_pub)?;
        let lock = if dialect == Dialect::Sqlite {
            ""
        } else {
            " FOR UPDATE"
        };
        let sql = format!(
            "SELECT ti.db_payload_generation, ti.status FROM {TRACE_INFO} ti WHERE ti.request_id = {} \
             AND ti.experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = {}){lock}",
            dialect.placeholder(1), dialect.placeholder(2)
        );
        let row = tx
            .fetch_all(
                &sql,
                &[
                    Val::Text(trace_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| {
                    Ok((
                        row.get_int("db_payload_generation")?,
                        row.get_string("status")?,
                    ))
                },
            )
            .await
            .map_err(map_db_err_pub)?
            .into_iter()
            .next();
        let Some((generation, status)) = row else {
            return Ok(false);
        };
        if generation != db_payload_generation || status == "IN_PROGRESS" {
            return Ok(false);
        }
        let tags = trace_tags_in_tx(&mut tx, dialect, trace_id).await?;
        if !db_backed(tags.get(TRACE_TAG_SPANS_LOCATION).map(String::as_str))
            || tags.contains_key(TRACE_TAG_ARCHIVAL_FAILURE)
        {
            return Ok(false);
        }
        let nonempty_sql = format!(
            "SELECT span_id FROM {SPANS} WHERE trace_id = {} AND content <> '' LIMIT 1",
            dialect.placeholder(1)
        );
        if tx
            .fetch_all(&nonempty_sql, &[Val::Text(trace_id.to_string())], |row| {
                row.get_string("span_id")
            })
            .await
            .map_err(map_db_err_pub)?
            .is_empty()
        {
            return Ok(false);
        }

        let trace_bind = [Val::Text(trace_id.to_string())];
        tx.exec(
            &format!(
                "UPDATE {SPANS} SET content = '' WHERE trace_id = {}",
                dialect.placeholder(1)
            ),
            &trace_bind,
        )
        .await
        .map_err(map_db_err_pub)?;
        tx.exec(
            &format!(
                "DELETE FROM {SPAN_ATTRIBUTES} WHERE trace_id = {}",
                dialect.placeholder(1)
            ),
            &trace_bind,
        )
        .await
        .map_err(map_db_err_pub)?;
        upsert_trace_tag(
            &mut tx,
            dialect,
            trace_id,
            TRACE_TAG_SPANS_LOCATION,
            SPANS_LOCATION_ARCHIVE_REPO,
        )
        .await?;
        upsert_trace_tag(
            &mut tx,
            dialect,
            trace_id,
            TRACE_TAG_ARCHIVE_LOCATION,
            artifact_uri,
        )
        .await?;
        tx.exec(
            &format!(
                "DELETE FROM {TRACE_TAGS} WHERE request_id = {} AND \"key\" = {}",
                dialect.placeholder(1),
                dialect.placeholder(2)
            ),
            &[
                Val::Text(trace_id.to_string()),
                Val::Text(TRACE_TAG_ARCHIVAL_FAILURE.to_string()),
            ],
        )
        .await
        .map_err(map_db_err_pub)?;
        tx.commit().await.map_err(map_db_err_pub)?;
        Ok(true)
    }

    /// Generation-guarded terminal failure marking for malformed payloads.
    pub async fn mark_trace_archival_malformed(
        &self,
        workspace: &str,
        trace_id: &str,
        db_payload_generation: i64,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(map_db_err_pub)?;
        let sql = format!(
            "SELECT ti.db_payload_generation FROM {TRACE_INFO} ti WHERE ti.request_id = {} \
             AND ti.experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = {})",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let generation = tx
            .fetch_all(
                &sql,
                &[
                    Val::Text(trace_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_int("db_payload_generation"),
            )
            .await
            .map_err(map_db_err_pub)?
            .into_iter()
            .next();
        if generation != Some(db_payload_generation) {
            return Ok(());
        }
        let tags = trace_tags_in_tx(&mut tx, dialect, trace_id).await?;
        if !db_backed(tags.get(TRACE_TAG_SPANS_LOCATION).map(String::as_str)) {
            return Ok(());
        }
        upsert_trace_tag(
            &mut tx,
            dialect,
            trace_id,
            TRACE_TAG_ARCHIVAL_FAILURE,
            "MALFORMED_TRACE",
        )
        .await?;
        tx.commit().await.map_err(map_db_err_pub)
    }

    /// Clear archive-now tags whose matching traces have finished processing
    /// (or reached terminal failures), preserving requests with retryable or
    /// in-progress candidates and never deleting a value replaced mid-pass.
    pub async fn clear_completed_archive_now_requests(
        &self,
        workspace: &str,
        requests: &[ArchiveNowRequest],
        now_millis: i64,
        retryable_failure_experiment_ids: &HashSet<String>,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        for request in requests {
            if retryable_failure_experiment_ids.contains(&request.experiment_id) {
                continue;
            }
            let experiment_id = request.experiment_id.parse::<i64>().map_err(|error| {
                MlflowError::internal_error(format!("Invalid archival experiment ID: {error}"))
            })?;
            let mut binds = vec![Val::Int(experiment_id), Val::Text(workspace.to_string())];
            let mut cutoff = String::new();
            if let Some(older_than) = request.older_than_millis {
                binds.push(Val::Int(now_millis - older_than));
                cutoff = format!(" AND ti.timestamp_ms <= {}", dialect.placeholder(3));
            }
            let sql = format!(
                "SELECT ti.status, \
                 EXISTS (SELECT 1 FROM {TRACE_TAGS} failure WHERE failure.request_id = ti.request_id \
                 AND failure.\"key\" = '{TRACE_TAG_ARCHIVAL_FAILURE}') AS has_failure, \
                 EXISTS (SELECT 1 FROM {SPANS} span WHERE span.trace_id = ti.request_id \
                 AND span.content <> '') AS has_content FROM {TRACE_INFO} ti \
                 WHERE ti.experiment_id = {} AND ti.experiment_id IN \
                 (SELECT experiment_id FROM experiments WHERE workspace = {}) {cutoff} \
                 AND NOT EXISTS (SELECT 1 FROM {TRACE_TAGS} location \
                 WHERE location.request_id = ti.request_id AND location.\"key\" = '{TRACE_TAG_SPANS_LOCATION}' \
                 AND location.value <> '{SPANS_LOCATION_TRACKING_STORE}')",
                dialect.placeholder(1), dialect.placeholder(2)
            );
            let remaining = self
                .db()
                .fetch_all(&sql, &binds, |row| {
                    Ok((
                        row.get_string("status")?,
                        row.get_bool("has_failure")?,
                        row.get_bool("has_content")?,
                    ))
                })
                .await
                .map_err(map_db_err_pub)?;
            let retain = remaining.iter().any(|(status, failure, content)| {
                !failure && (*content || status == "IN_PROGRESS")
            });
            if retain {
                continue;
            }
            self.db()
                .exec(
                    &format!(
                        "DELETE FROM experiment_tags WHERE experiment_id = {} AND \"key\" = {} AND value = {}",
                        dialect.placeholder(1), dialect.placeholder(2), dialect.placeholder(3)
                    ),
                    &[
                        Val::Int(experiment_id),
                        Val::Text(TRACE_EXPERIMENT_TAG_ARCHIVE_NOW.to_string()),
                        Val::Text(request.raw_value.clone()),
                    ],
                )
                .await
                .map_err(map_db_err_pub)?;
        }
        Ok(())
    }

    async fn archival_experiments(
        &self,
        workspace: &str,
    ) -> Result<Vec<(i64, HashMap<String, String>)>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT e.experiment_id, et.\"key\", et.value FROM experiments e \
             LEFT JOIN experiment_tags et ON et.experiment_id = e.experiment_id \
             AND et.\"key\" IN ({}, {}) WHERE e.workspace = {} AND e.lifecycle_stage = 'active' \
             ORDER BY e.experiment_id, et.\"key\"",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        );
        let rows = self
            .db()
            .fetch_all(
                &sql,
                &[
                    Val::Text(TRACE_EXPERIMENT_TAG_ARCHIVE_NOW.to_string()),
                    Val::Text(TRACE_EXPERIMENT_TAG_ARCHIVAL_RETENTION.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| {
                    Ok((
                        row.get_int("experiment_id")?,
                        row.get_opt_string("key")?,
                        row.get_opt_string("value")?,
                    ))
                },
            )
            .await
            .map_err(map_db_err_pub)?;
        let mut experiments: Vec<(i64, HashMap<String, String>)> = Vec::new();
        for (experiment_id, key, value) in rows {
            if experiments.last().map(|entry| entry.0) != Some(experiment_id) {
                experiments.push((experiment_id, HashMap::new()));
            }
            if let (Some(key), Some(value)) = (key, value) {
                experiments.last_mut().unwrap().1.insert(key, value);
            }
        }
        Ok(experiments)
    }

    async fn collect_candidates(
        &self,
        workspace: &str,
        cutoff_groups: &HashMap<Option<i64>, Vec<i64>>,
        limit: Option<usize>,
    ) -> Result<Vec<TraceArchiveCandidate>, MlflowError> {
        let mut candidates = Vec::new();
        for (cutoff, experiment_ids) in cutoff_groups {
            let mut group = self
                .find_archivable_candidates(workspace, experiment_ids, *cutoff, limit)
                .await?;
            candidates.append(&mut group);
            candidates
                .sort_by(|a, b| (a.timestamp_ms, &a.trace_id).cmp(&(b.timestamp_ms, &b.trace_id)));
            if let Some(limit) = limit {
                candidates.truncate(limit);
            }
        }
        Ok(candidates)
    }

    async fn find_archivable_candidates(
        &self,
        workspace: &str,
        experiment_ids: &[i64],
        cutoff: Option<i64>,
        limit: Option<usize>,
    ) -> Result<Vec<TraceArchiveCandidate>, MlflowError> {
        if experiment_ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let mut binds = Vec::new();
        let exp_ph = experiment_ids
            .iter()
            .enumerate()
            .map(|(index, id)| {
                binds.push(Val::Int(*id));
                dialect.placeholder(index + 1)
            })
            .collect::<Vec<_>>();
        binds.push(Val::Text(workspace.to_string()));
        let workspace_ph = dialect.placeholder(binds.len());
        let mut cutoff_clause = String::new();
        if let Some(cutoff) = cutoff {
            binds.push(Val::Int(cutoff));
            cutoff_clause = format!(
                " AND ti.timestamp_ms <= {}",
                dialect.placeholder(binds.len())
            );
        }
        let mut sql = format!(
            "SELECT ti.request_id, ti.experiment_id, ti.timestamp_ms FROM {TRACE_INFO} ti \
             WHERE ti.experiment_id IN ({}) AND ti.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {workspace_ph}) \
             AND ti.status <> 'IN_PROGRESS' {cutoff_clause} \
             AND NOT EXISTS (SELECT 1 FROM {TRACE_TAGS} t WHERE t.request_id = ti.request_id \
             AND t.\"key\" = '{TRACE_TAG_SPANS_LOCATION}' AND t.value <> '{SPANS_LOCATION_TRACKING_STORE}') \
             AND NOT EXISTS (SELECT 1 FROM {TRACE_TAGS} t WHERE t.request_id = ti.request_id \
             AND t.\"key\" = '{TRACE_TAG_ARCHIVAL_FAILURE}') \
             AND EXISTS (SELECT 1 FROM {SPANS} s WHERE s.trace_id = ti.request_id AND s.content <> '') \
             ORDER BY ti.timestamp_ms, ti.request_id",
            exp_ph.join(", ")
        );
        if let Some(limit) = limit {
            sql.push_str(&format!(" LIMIT {limit}"));
        }
        self.db()
            .fetch_all(&sql, &binds, |row| {
                Ok(TraceArchiveCandidate {
                    trace_id: row.get_string("request_id")?,
                    experiment_id: row.get_int("experiment_id")?.to_string(),
                    timestamp_ms: row.get_i64("timestamp_ms")?,
                })
            })
            .await
            .map_err(map_db_err_pub)
    }
}

fn parse_retention_millis(value: &str) -> Option<i64> {
    let value = value.trim();
    if value.len() < 2 || value.len() > 32 {
        return None;
    }
    let (amount, unit) = value.split_at(value.len() - 1);
    if amount.starts_with('0') || !amount.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let amount = amount.parse::<i64>().ok()?;
    let multiplier = match unit {
        "m" => MINUTE_MILLIS,
        "h" => HOUR_MILLIS,
        "d" => DAY_MILLIS,
        _ => return None,
    };
    amount.checked_mul(multiplier)
}

fn parse_archive_now(raw: &str) -> Option<Option<i64>> {
    let payload: Value = serde_json::from_str(raw).ok()?;
    let object = payload.as_object()?;
    match object.get("older_than") {
        None | Some(Value::Null) => Some(None),
        Some(Value::String(value)) => parse_retention_millis(value).map(Some),
        Some(_) => None,
    }
}

fn effective_retention_millis(
    experiment_id: i64,
    tags: &HashMap<String, String>,
    broader_ms: i64,
    allowlist: &HashSet<String>,
) -> i64 {
    let Some(raw) = tags.get(TRACE_EXPERIMENT_TAG_ARCHIVAL_RETENTION) else {
        return broader_ms;
    };
    let experiment_ms = serde_json::from_str::<Value>(raw).ok().and_then(|value| {
        let object = value.as_object()?;
        if object.get("type")?.as_str()? != "duration" {
            return None;
        }
        parse_retention_millis(object.get("value")?.as_str()?)
    });
    match experiment_ms {
        Some(value) if value <= broader_ms => value,
        Some(value) if allowlist.contains(&experiment_id.to_string()) => value,
        _ => broader_ms,
    }
}

fn db_backed(location: Option<&str>) -> bool {
    location.is_none() || location == Some(SPANS_LOCATION_TRACKING_STORE)
}

fn trace_actionable(info: &TraceInfo) -> bool {
    db_backed(info.tag(TRACE_TAG_SPANS_LOCATION))
        && info.tag(TRACE_TAG_ARCHIVAL_FAILURE).is_none()
        && info.state != "IN_PROGRESS"
}

async fn trace_tags_in_tx(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_id: &str,
) -> Result<HashMap<String, String>, MlflowError> {
    let sql = format!(
        "SELECT \"key\", value FROM {TRACE_TAGS} WHERE request_id = {}",
        dialect.placeholder(1)
    );
    let rows = tx
        .fetch_all(&sql, &[Val::Text(trace_id.to_string())], |row| {
            Ok((row.get_string("key")?, row.get_opt_string("value")?))
        })
        .await
        .map_err(map_db_err_pub)?;
    Ok(rows
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| (key, value)))
        .collect())
}

async fn upsert_trace_tag(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    trace_id: &str,
    key: &str,
    value: &str,
) -> Result<(), MlflowError> {
    let sql = match dialect {
        Dialect::MySql => format!(
            "INSERT INTO {TRACE_TAGS} (request_id, \"key\", value) VALUES (?, ?, ?) \
             ON DUPLICATE KEY UPDATE value = VALUES(value)"
        ),
        _ => format!(
            "INSERT INTO {TRACE_TAGS} (request_id, \"key\", value) VALUES ({}, {}, {}) \
             ON CONFLICT (request_id, \"key\") DO UPDATE SET value = excluded.value",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3)
        ),
    };
    tx.exec(
        &sql,
        &[
            Val::Text(trace_id.to_string()),
            Val::Text(key.to_string()),
            Val::Text(value.to_string()),
        ],
    )
    .await
    .map_err(map_db_err_pub)?;
    Ok(())
}
