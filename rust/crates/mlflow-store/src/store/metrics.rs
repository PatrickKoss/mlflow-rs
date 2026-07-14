//! Metric operations: `log_metric`, `latest_metrics` maintenance,
//! `get_metric_history`, and `log_batch` (plan T2.5), mirroring
//! `sqlalchemy_store.py`.
//!
//! ## NaN / ±Inf storage (`sanitize_metric_value`)
//!
//! MLflow cannot store NaN/Inf as SQL floats, so:
//! * NaN  → stored `value = 0.0`, `is_nan = true`; read back as `f64::NAN`.
//! * +Inf → clamped to `f64::MAX` (`1.7976931348623157e308`), `is_nan = false`.
//! * -Inf → clamped to `-f64::MAX`, `is_nan = false`.
//!
//! ## `metrics` dedup (6-col PK)
//!
//! The `metrics` table PK is `(key, timestamp, step, run_uuid, value, is_nan)`.
//! Re-logging an identical row is silently OK, implemented with an
//! `ON CONFLICT DO NOTHING` (Python catches the IntegrityError and drops the
//! duplicate — same observable result). Because `is_nan` is part of the PK and
//! NaN is stored as `0.0`, two identical NaN entries dedup naturally here (an
//! edge Python's Python-side set-dedup does *not* collapse, but the DB does).
//!
//! ## `latest_metrics` atomic upsert (plan Q5)
//!
//! Python does select-for-update + compare `(step, timestamp, value)` and
//! overwrites when the new tuple is strictly greater. Rust does the same
//! comparison atomically in one statement:
//!
//! * SQLite / Postgres:
//!   `INSERT ... ON CONFLICT (key, run_uuid) DO UPDATE SET value=excluded.value,
//!    ... WHERE (excluded.step, excluded.timestamp, excluded.value) >
//!    (latest_metrics.step, latest_metrics.timestamp, latest_metrics.value)`
//!   using SQL row-value comparison, which orders lexicographically by
//!   `step`, then `timestamp`, then `value` — identical to Python's tuple `>`.
//! * MySQL (no `WHERE` on `ON DUPLICATE KEY UPDATE`, no row-value `>` there):
//!   `... ON DUPLICATE KEY UPDATE value = IF(<greater>, VALUES(value), value),
//!    ...` where `<greater>` is the expanded lexicographic comparison
//!   `VALUES(step) > step OR (VALUES(step)=step AND (VALUES(timestamp)>timestamp
//!    OR (VALUES(timestamp)=timestamp AND VALUES(value)>value)))`.
//!
//! Within one `log_batch`, metrics sharing a key are upserted in sequence in the
//! same transaction, so each comparison sees the running maximum — matching
//! Python's in-loop `new_latest_metric_dict` behavior.

use mlflow_error::MlflowError;

use super::dbutil::{Tx, Val};
use super::entities::Metric;
use super::experiments::internal;
use super::runs::check_run_active;
use super::validation;
use super::TrackingStore;
use crate::dialect::Dialect;

/// Store-side cap on `get_metric_history` results (plan §3.3).
pub const GET_METRIC_HISTORY_MAX_RESULTS: usize = 25_000;

/// A metric to be logged.
///
/// `model_id`/`dataset_name`/`dataset_digest` carry a metric into
/// `logged_model_metrics` in addition to the run's `metrics`/`latest_metrics`
/// tables — see [`TrackingStore::log_model_metrics_tx`] and the `log_batch`/
/// `log_metric` doc comments below for how Python routes these (`model_id`
/// metrics are written to BOTH the run metric tables and
/// `logged_model_metrics`; they are not mutually exclusive).
#[derive(Debug, Clone)]
pub struct MetricInput {
    pub key: String,
    pub value: f64,
    pub timestamp: i64,
    pub step: i64,
    pub model_id: Option<String>,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// `sanitize_metric_value`: returns `(is_nan, stored_value)`.
pub(crate) fn sanitize_metric_value(value: f64) -> (bool, f64) {
    if value.is_nan() {
        (true, 0.0)
    } else if value.is_infinite() {
        (false, if value > 0.0 { f64::MAX } else { -f64::MAX })
    } else {
        (false, value)
    }
}

impl TrackingStore {
    /// `log_metric` (single). Validates, then logs via the batch path.
    ///
    /// Python's `log_metric` (sqlalchemy_store.py:1183) does two independent
    /// things when `metric.model_id` is set: it logs the model metric via
    /// `_log_model_metrics` (its own session) AND unconditionally still logs
    /// the same metric into `metrics`/`latest_metrics` via `_log_metrics` — a
    /// `model_id` metric is not routed *instead of* the run metric tables, it
    /// is routed *in addition to* them. We do both in one transaction (Q6
    /// spirit: collapse Python's redundant separate sessions), rather than
    /// Python's two-or-three separate commits.
    pub async fn log_metric(
        &self,
        workspace: &str,
        run_id: &str,
        metric: &MetricInput,
    ) -> Result<(), MlflowError> {
        validation::validate_metric(
            &metric.key,
            metric.value,
            metric.timestamp,
            metric.step,
            None,
        )?;
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        if metric.model_id.is_some() {
            self.log_model_metrics_tx(
                &mut tx,
                workspace,
                row.experiment_id,
                run_id,
                None,
                std::slice::from_ref(metric),
            )
            .await?;
        }
        insert_metrics(
            &mut tx,
            self.db().dialect(),
            run_id,
            std::slice::from_ref(metric),
        )
        .await?;
        tx.commit().await.map_err(internal)
    }

    /// `get_metric_history` with offset pagination. Ordered by
    /// `(timestamp, step, value)` (Python's ORDER BY). `max_results` caps the
    /// page; a returned token means more rows follow.
    pub async fn get_metric_history(
        &self,
        workspace: &str,
        run_id: &str,
        metric_key: &str,
        max_results: Option<usize>,
        page_token: Option<&str>,
    ) -> Result<(Vec<Metric>, Option<String>), MlflowError> {
        // Workspace access check (mirrors `_validate_run_accessible`).
        self.resolve_run_row(workspace, run_id).await?;
        let offset = parse_page_token(page_token)?;
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);

        let mut sql = format!(
            "SELECT key, value, timestamp, step, is_nan FROM metrics \
             WHERE run_uuid = {} AND key = {} \
             ORDER BY timestamp, step, value",
            ph(1),
            ph(2)
        );
        if let Some(mr) = max_results {
            // Fetch one extra to detect a following page.
            sql.push_str(&format!(" LIMIT {} OFFSET {offset}", mr + 1));
        } else if offset > 0 {
            sql.push_str(&format!(" LIMIT -1 OFFSET {offset}"));
        }

        let mut metrics = self
            .db()
            .fetch_all(
                &sql,
                &[
                    Val::Text(run_id.to_string()),
                    Val::Text(metric_key.to_string()),
                ],
                metric_from_row,
            )
            .await
            .map_err(internal)?;

        let next_token = match max_results {
            Some(mr) if metrics.len() == mr + 1 => {
                metrics.truncate(mr);
                Some(encode_page_token(offset + mr))
            }
            _ => None,
        };
        Ok((metrics, next_token))
    }

    /// `log_batch`: validate limits + data, then log params, metrics, and tags
    /// in **one transaction** (plan Q6). Any failure rolls the whole batch back,
    /// so param immutability inside the batch aborts everything (matching the
    /// no-partial-data test).
    ///
    /// Python's `log_batch` (sqlalchemy_store.py:2098) calls, in order,
    /// `_log_params`, `_log_metrics` (ALL metrics, including any carrying a
    /// `model_id` — they still land in `metrics`/`latest_metrics`), then
    /// `_log_model_metrics(run_id, metrics, experiment_id=run.experiment_id)`
    /// (only the subset with `model_id is not None`, written again into
    /// `logged_model_metrics`), then `_set_tags`. Each of those Python calls
    /// opens its own session, so `log_batch` is *not* atomic across them
    /// there (plan Q6 calls this out as an intentional gap); we fold all four
    /// steps into the one transaction already used here for params/metrics/tags,
    /// consistent with Q6's "one transaction per log-batch" wire-invisible
    /// improvement.
    pub async fn log_batch(
        &self,
        workspace: &str,
        run_id: &str,
        metrics: &[MetricInput],
        params: &[(&str, &str)],
        tags: &[(&str, &str)],
    ) -> Result<(), MlflowError> {
        // Validate data (per-entity) then batch limits, then dup param keys —
        // same order as Python's `log_batch`.
        for (i, m) in metrics.iter().enumerate() {
            validation::validate_metric(
                &m.key,
                m.value,
                m.timestamp,
                m.step,
                Some(&format!("metrics[{i}]")),
            )?;
        }
        for (i, (k, v)) in params.iter().enumerate() {
            validation::validate_param(k, v).map_err(|e| prefix_param_err(e, i))?;
        }
        for (i, (k, v)) in tags.iter().enumerate() {
            validation::validate_tag(k, v, Some(&format!("tags[{i}]")))?;
        }
        validation::validate_batch_log_limits(metrics.len(), params.len(), tags.len())?;
        let param_keys: Vec<&str> = params.iter().map(|(k, _)| *k).collect();
        validation::validate_param_keys_unique(&param_keys)?;

        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        // Params first (matches Python: immutability check aborts before
        // metrics/tags are logged).
        self.log_params_tx(&mut tx, run_id, params).await?;
        // Metrics: run-scoped metrics/latest_metrics for every metric...
        if !metrics.is_empty() {
            insert_metrics(&mut tx, dialect, run_id, metrics).await?;
        }
        // ...and, for the subset carrying a model_id, also logged_model_metrics.
        // `metrics` is passed through unfiltered/ungrouped — `log_model_metrics_tx`
        // does its own `model_id is not None` filtering over the *whole* list,
        // exactly like Python's `_log_model_metrics(run_id, metrics, ...)` (see
        // its doc comment for why preserving the original list/indices matters).
        self.log_model_metrics_tx(&mut tx, workspace, row.experiment_id, run_id, None, metrics)
            .await?;
        // Tags.
        self.set_tags_tx(&mut tx, dialect, run_id, tags).await?;

        tx.commit().await.map_err(internal)
    }

    /// Load `latest_metrics` for a run into entities (used by `get_run`).
    pub(crate) async fn load_latest_metrics(
        &self,
        run_id: &str,
    ) -> Result<Vec<Metric>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT key, value, timestamp, step, is_nan FROM latest_metrics \
             WHERE run_uuid = {} ORDER BY key",
            dialect.placeholder(1)
        );
        self.db()
            .fetch_all(&sql, &[Val::Text(run_id.to_string())], metric_from_row)
            .await
            .map_err(internal)
    }

    /// `_log_params` (batch, immutability-checked) inside a transaction.
    async fn log_params_tx(
        &self,
        tx: &mut Tx<'_>,
        run_id: &str,
        params: &[(&str, &str)],
    ) -> Result<(), MlflowError> {
        if params.is_empty() {
            return Ok(());
        }
        let dialect = self.db().dialect();
        // Read existing params for the run once.
        let existing = tx
            .fetch_all(
                &format!(
                    "SELECT key, value FROM params WHERE run_uuid = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(run_id.to_string())],
                |r| Ok((r.get_string("key")?, r.get_string("value")?)),
            )
            .await
            .map_err(internal)?;

        let mut non_matching: Vec<(String, String, String)> = Vec::new();
        let mut to_insert: Vec<(&str, &str)> = Vec::new();
        for (k, v) in params {
            if let Some((_, old)) = existing.iter().find(|(ek, _)| ek == k) {
                if old != v {
                    non_matching.push((k.to_string(), old.clone(), v.to_string()));
                }
                continue;
            }
            to_insert.push((k, v));
        }

        if !non_matching.is_empty() {
            // Render list of dicts like Python's repr.
            let rendered = non_matching
                .iter()
                .map(|(k, old, new)| {
                    format!("{{'key': '{k}', 'old_value': '{old}', 'new_value': '{new}'}}")
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(MlflowError::invalid_parameter_value(format!(
                "Changing param values is not allowed. Params were already logged='[{rendered}]' \
                 for run ID='{run_id}'."
            )));
        }

        for (k, v) in to_insert {
            let sql = format!(
                "INSERT INTO params (key, value, run_uuid) VALUES ({}, {}, {})",
                dialect.placeholder(1),
                dialect.placeholder(2),
                dialect.placeholder(3)
            );
            tx.exec(
                &sql,
                &[
                    Val::Text(k.to_string()),
                    Val::Text(v.to_string()),
                    Val::Text(run_id.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        }
        Ok(())
    }

    /// `_set_tags` (batch) inside a transaction. Later duplicate keys win. The
    /// `mlflow.runName` tag also updates `runs.name`.
    async fn set_tags_tx(
        &self,
        tx: &mut Tx<'_>,
        dialect: Dialect,
        run_id: &str,
        tags: &[(&str, &str)],
    ) -> Result<(), MlflowError> {
        // Collapse duplicate keys, last value wins (Python's new_tag_dict).
        let mut resolved: Vec<(String, String)> = Vec::new();
        for (k, v) in tags {
            if let Some(slot) = resolved.iter_mut().find(|(rk, _)| rk == k) {
                slot.1 = v.to_string();
            } else {
                resolved.push((k.to_string(), v.to_string()));
            }
        }
        for (k, v) in &resolved {
            if k == super::MLFLOW_RUN_NAME {
                let sql = format!(
                    "UPDATE runs SET name = {} WHERE run_uuid = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                );
                tx.exec(&sql, &[Val::Text(v.clone()), Val::Text(run_id.to_string())])
                    .await
                    .map_err(internal)?;
                super::runs::sync_run_name_tag(tx, dialect, run_id, v).await?;
            } else {
                super::params_tags::upsert_tag(tx, dialect, run_id, k, v).await?;
            }
        }
        Ok(())
    }
}

/// Insert `metrics` rows (with dedup) and maintain `latest_metrics` atomically,
/// inside `tx`. Applies `sanitize_metric_value`.
async fn insert_metrics(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    run_id: &str,
    metrics: &[MetricInput],
) -> Result<(), MlflowError> {
    for m in metrics {
        let (is_nan, value) = sanitize_metric_value(m.value);
        // 1) Insert into metrics (dedup on the 6-col PK).
        let insert_sql = metrics_insert_sql(dialect);
        tx.exec(
            &insert_sql,
            &[
                Val::Text(m.key.clone()),
                Val::Float(value),
                Val::Int(m.timestamp),
                Val::Int(m.step),
                Val::Bool(is_nan),
                Val::Text(run_id.to_string()),
            ],
        )
        .await
        .map_err(internal)?;

        // 2) Maintain latest_metrics via the atomic conditional upsert.
        let upsert_sql = latest_metric_upsert_sql(dialect);
        tx.exec(
            &upsert_sql,
            &[
                Val::Text(m.key.clone()),
                Val::Float(value),
                Val::Int(m.timestamp),
                Val::Int(m.step),
                Val::Bool(is_nan),
                Val::Text(run_id.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
    }
    Ok(())
}

/// `INSERT INTO metrics ... ON CONFLICT/DUPLICATE DO NOTHING` (6-col PK dedup).
fn metrics_insert_sql(dialect: Dialect) -> String {
    let cols = "(key, value, timestamp, step, is_nan, run_uuid)";
    let values = match dialect {
        Dialect::Postgres => "($1, $2, $3, $4, $5, $6)",
        Dialect::Sqlite | Dialect::MySql => "(?, ?, ?, ?, ?, ?)",
    };
    match dialect {
        Dialect::Sqlite | Dialect::Postgres => {
            format!(
                "INSERT INTO metrics {cols} VALUES {values} \
                 ON CONFLICT (key, timestamp, step, run_uuid, value, is_nan) DO NOTHING"
            )
        }
        Dialect::MySql => {
            // Self-assign to make it a no-op on duplicate.
            format!(
                "INSERT INTO metrics {cols} VALUES {values} ON DUPLICATE KEY UPDATE `key` = `key`"
            )
        }
    }
}

/// The atomic `latest_metrics` upsert keyed on `(key, run_uuid)`, overwriting
/// only when the new `(step, timestamp, value)` tuple is strictly greater.
///
/// Bind order: key, value, timestamp, step, is_nan, run_uuid.
fn latest_metric_upsert_sql(dialect: Dialect) -> String {
    match dialect {
        Dialect::Sqlite | Dialect::Postgres => {
            let values = if dialect == Dialect::Postgres {
                "($1, $2, $3, $4, $5, $6)"
            } else {
                "(?, ?, ?, ?, ?, ?)"
            };
            format!(
                "INSERT INTO latest_metrics (key, value, timestamp, step, is_nan, run_uuid) \
                 VALUES {values} \
                 ON CONFLICT (key, run_uuid) DO UPDATE SET \
                 value = excluded.value, timestamp = excluded.timestamp, \
                 step = excluded.step, is_nan = excluded.is_nan \
                 WHERE (excluded.step, excluded.timestamp, excluded.value) > \
                 (latest_metrics.step, latest_metrics.timestamp, latest_metrics.value)"
            )
        }
        Dialect::MySql => {
            // Expand the lexicographic (step, timestamp, value) comparison; no
            // WHERE clause is available on ON DUPLICATE KEY UPDATE, and MySQL
            // lacks row-value `>` in this position, so guard each SET with IF().
            let greater = "(VALUES(step) > step OR \
                 (VALUES(step) = step AND \
                  (VALUES(timestamp) > timestamp OR \
                   (VALUES(timestamp) = timestamp AND VALUES(value) > value))))";
            format!(
                "INSERT INTO latest_metrics (key, value, timestamp, step, is_nan, run_uuid) \
                 VALUES (?, ?, ?, ?, ?, ?) \
                 ON DUPLICATE KEY UPDATE \
                 value = IF({greater}, VALUES(value), value), \
                 timestamp = IF({greater}, VALUES(timestamp), timestamp), \
                 step = IF({greater}, VALUES(step), step), \
                 is_nan = IF({greater}, VALUES(is_nan), is_nan)"
            )
        }
    }
}

/// Map a metric row: `is_nan` restores `f64::NAN` (Python `to_mlflow_entity`).
fn metric_from_row(r: &dyn super::dbutil::RowLike) -> Result<Metric, sqlx::Error> {
    let is_nan = r.get_bool("is_nan")?;
    let stored = r.get_f64("value")?;
    Ok(Metric {
        key: r.get_string("key")?,
        value: if is_nan { f64::NAN } else { stored },
        timestamp: r.get_opt_i64("timestamp")?.unwrap_or(0),
        step: r.get_i64("step")?,
    })
}

/// `SearchUtils.parse_start_offset_from_page_token`: base64(JSON `{"offset": N}`).
/// An unparseable token errors with "Invalid page token".
fn parse_page_token(token: Option<&str>) -> Result<usize, MlflowError> {
    let Some(t) = token else {
        return Ok(0);
    };
    decode_page_token(t)
        .ok_or_else(|| MlflowError::invalid_parameter_value(format!("Invalid page token: {t}")))
}

fn encode_page_token(offset: usize) -> String {
    use base64_lite::encode;
    encode(format!("{{\"offset\": {offset}}}").as_bytes())
}

fn decode_page_token(token: &str) -> Option<usize> {
    let bytes = base64_lite::decode(token)?;
    let s = String::from_utf8(bytes).ok()?;
    // Extract the integer after "offset".
    let idx = s.find("offset")?;
    let rest = &s[idx + "offset".len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<usize>().ok()
}

/// `Missing value ... 'metrics[i].value'` — but the batch metric-null path is
/// caught during validation of the request proto in Python; here f64 is always
/// present, so this only reshapes a param error path prefix.
fn prefix_param_err(e: MlflowError, _index: usize) -> MlflowError {
    e
}

/// Minimal standard-base64 (no external dep). MLflow page tokens are
/// base64(JSON); we only need round-trip and tolerant decode.
mod base64_lite {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[((n >> 6) & 63) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    pub fn decode(input: &str) -> Option<Vec<u8>> {
        fn val(c: u8) -> Option<u32> {
            match c {
                b'A'..=b'Z' => Some(u32::from(c - b'A')),
                b'a'..=b'z' => Some(u32::from(c - b'a' + 26)),
                b'0'..=b'9' => Some(u32::from(c - b'0' + 52)),
                b'+' => Some(62),
                b'/' => Some(63),
                _ => None,
            }
        }
        let cleaned: Vec<u8> = input.bytes().filter(|&c| c != b'=').collect();
        let mut out = Vec::new();
        for chunk in cleaned.chunks(4) {
            let mut n = 0u32;
            let mut bits = 0;
            for &c in chunk {
                n = (n << 6) | val(c)?;
                bits += 6;
            }
            // Align to byte boundary.
            let bytes = bits / 8;
            n <<= (4 - chunk.len()) as u32 * 6;
            let full = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
            out.extend_from_slice(&full[..bytes]);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize() {
        assert_eq!(sanitize_metric_value(1.5), (false, 1.5));
        assert_eq!(sanitize_metric_value(f64::INFINITY), (false, f64::MAX));
        assert_eq!(sanitize_metric_value(f64::NEG_INFINITY), (false, -f64::MAX));
        let (is_nan, v) = sanitize_metric_value(f64::NAN);
        assert!(is_nan);
        assert_eq!(v, 0.0);
        // Matches Python's exact clamp constant.
        assert_eq!(f64::MAX, 1.7976931348623157e308);
    }

    #[test]
    fn page_token_roundtrip() {
        let t = encode_page_token(42);
        assert_eq!(decode_page_token(&t), Some(42));
        assert_eq!(parse_page_token(Some(&t)).unwrap(), 42);
        assert_eq!(parse_page_token(None).unwrap(), 0);
        assert!(parse_page_token(Some("not base64 json !!")).is_err());
    }

    #[test]
    fn latest_upsert_sql_shapes() {
        let s = latest_metric_upsert_sql(Dialect::Sqlite);
        assert!(s.contains("ON CONFLICT (key, run_uuid) DO UPDATE"));
        assert!(s.contains(
            "(excluded.step, excluded.timestamp, excluded.value) > \
             (latest_metrics.step, latest_metrics.timestamp, latest_metrics.value)"
        ));
        let pg = latest_metric_upsert_sql(Dialect::Postgres);
        assert!(pg.contains("$6"));
        let my = latest_metric_upsert_sql(Dialect::MySql);
        assert!(my.contains("ON DUPLICATE KEY UPDATE"));
        assert!(my.contains("VALUES(step) > step"));
    }
}
