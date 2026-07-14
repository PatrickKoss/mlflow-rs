//! Bulk metric history: `get_metric_history_bulk` and
//! `get_metric_history_bulk_interval` (plan T2.7), ported exactly from
//! `mlflow/store/tracking/sqlalchemy_store.py` (the SQL-based override) and the
//! handler caps in `mlflow/server/handlers.py`.
//!
//! ## `get_metric_history_bulk`
//!
//! Returns all logged values for `metric_key` across `run_ids` (≤100, enforced
//! by the handler), ordered by `(run_uuid, timestamp, step, value)` and capped
//! at `max_results` (≤25000). The ordering is a single global order across runs,
//! and the cap is a single global `LIMIT` — matching the Python `.limit()` on
//! the combined query. Run ids are first filtered to those accessible in the
//! workspace (mirrors `_filter_entity_ids(RUN)`), so out-of-workspace ids are
//! silently dropped rather than erroring.
//!
//! ## `get_metric_history_bulk_interval` (interval sampling)
//!
//! Ported verbatim from the SqlAlchemyStore override:
//!
//! 1. `all_steps` = distinct steps across all runs for the key, ordered asc.
//!    Empty → return `[]`.
//! 2. `all_mins_and_maxes` = the per-run `MIN(step)`/`MAX(step)` set (grouped by
//!    run), unioned across runs.
//! 3. If both `start_step` and `end_step` are `None`: `start_step = 0`,
//!    `end_step = all_steps.last()`.
//! 4. Clamp `all_mins_and_maxes` to `[start_step, end_step]`.
//! 5. `start_idx = bisect_left(all_steps, start_step)`,
//!    `end_idx = bisect_right(all_steps, end_step)`.
//! 6. If `end_idx - start_idx <= max_results`: keep every step in the slice.
//!    Else: `interval = num_steps as f64 / max_results as f64`; for
//!    `i in 0..max_results`: `idx = start_idx + (i as f64 * interval) as i64`
//!    (truncate toward zero); if `idx < end_idx` keep `all_steps[idx]`. Always
//!    additionally keep `all_steps[end_idx - 1]`.
//! 7. `steps = sort(unique(sampled ∪ mins_and_maxes))`.
//! 8. For each run **in request order**, fetch rows where `step IN (steps)`,
//!    ordered `(run_uuid, step, timestamp, value)`, limited to 25000, and
//!    concatenate (no global re-sort).
//!
//! The `interval` division is IEEE-754 f64 and `int()` truncation is reproduced
//! with `as i64`. `sampled_steps` is a set, so duplicate index picks collapse.

use std::collections::BTreeSet;

use mlflow_error::MlflowError;

use super::dbutil::Val;
use super::entities::{Metric, MetricWithRunId};
use super::experiments::internal;
use super::metrics::GET_METRIC_HISTORY_MAX_RESULTS;
use super::TrackingStore;

/// Handler cap on run ids per bulk request (`MAX_RUN_IDS_PER_REQUEST` /
/// `MAX_RUNS_GET_METRIC_HISTORY_BULK`).
pub const MAX_RUNS_GET_METRIC_HISTORY_BULK: usize = 100;

/// Handler cap on sampled results per run for the interval API
/// (`MAX_RESULTS_PER_RUN`).
pub const MAX_RESULTS_PER_RUN: usize = 2500;

impl TrackingStore {
    /// `get_metric_history_bulk`: metrics for `metric_key` across `run_ids`,
    /// ordered by `(run_uuid, timestamp, step, value)`, globally capped at
    /// `max_results`. Run ids are filtered to the workspace first.
    pub async fn get_metric_history_bulk(
        &self,
        workspace: &str,
        run_ids: &[&str],
        metric_key: &str,
        max_results: usize,
    ) -> Result<Vec<MetricWithRunId>, MlflowError> {
        let accessible = self.filter_run_ids(workspace, run_ids).await?;
        if accessible.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);

        let mut vals: Vec<Val> = vec![Val::Text(metric_key.to_string())];
        let placeholders: Vec<String> = accessible
            .iter()
            .enumerate()
            .map(|(i, rid)| {
                vals.push(Val::Text(rid.clone()));
                ph(i + 2)
            })
            .collect();

        let sql = format!(
            "SELECT run_uuid, \"key\", value, timestamp, step, is_nan FROM metrics \
             WHERE \"key\" = {} AND run_uuid IN ({}) \
             ORDER BY run_uuid, timestamp, step, value \
             LIMIT {}",
            ph(1),
            placeholders.join(", "),
            max_results,
        );

        self.db()
            .fetch_all(&sql, &vals, metric_with_run_id_from_row)
            .await
            .map_err(internal)
    }

    /// `get_metric_history_bulk_interval`: interval-sampled metric history across
    /// `run_ids`. See the module docs for the exact sampling arithmetic.
    /// `start_step`/`end_step` are both-or-neither (the handler enforces this);
    /// passing `None`/`None` defaults the range to `[0, max_step]`.
    pub async fn get_metric_history_bulk_interval(
        &self,
        workspace: &str,
        run_ids: &[&str],
        metric_key: &str,
        max_results: usize,
        start_step: Option<i64>,
        end_step: Option<i64>,
    ) -> Result<Vec<MetricWithRunId>, MlflowError> {
        for rid in run_ids {
            // Workspace access check per run (mirrors `_validate_run_accessible`).
            self.resolve_run_row(workspace, rid).await?;
        }

        let all_steps = self.distinct_steps(run_ids, metric_key).await?;
        if all_steps.is_empty() {
            return Ok(Vec::new());
        }

        let mut mins_and_maxes: BTreeSet<i64> = BTreeSet::new();
        for (min_step, max_step) in self.per_run_min_max_steps(run_ids, metric_key).await? {
            mins_and_maxes.insert(min_step);
            mins_and_maxes.insert(max_step);
        }

        let (start_step, end_step) = match (start_step, end_step) {
            (None, None) => (0, *all_steps.last().unwrap()),
            (s, e) => (
                s.unwrap_or(0),
                e.unwrap_or_else(|| *all_steps.last().unwrap()),
            ),
        };

        let mins_and_maxes: BTreeSet<i64> = mins_and_maxes
            .into_iter()
            .filter(|s| start_step <= *s && *s <= end_step)
            .collect();

        let start_idx = bisect_left(&all_steps, start_step);
        let end_idx = bisect_right(&all_steps, end_step);

        let mut sampled: BTreeSet<i64> = BTreeSet::new();
        let window = end_idx.saturating_sub(start_idx);
        if window <= max_results {
            for &s in &all_steps[start_idx..end_idx] {
                sampled.insert(s);
            }
        } else {
            let num_steps = window as f64;
            let interval = num_steps / max_results as f64;
            for i in 0..max_results {
                let idx = start_idx + (i as f64 * interval) as usize;
                if idx < end_idx {
                    sampled.insert(all_steps[idx]);
                }
            }
            sampled.insert(all_steps[end_idx - 1]);
        }

        // steps = sorted(sampled ∪ mins_and_maxes). BTreeSet keeps them sorted.
        let mut steps: BTreeSet<i64> = sampled;
        steps.extend(mins_and_maxes);
        let steps: Vec<i64> = steps.into_iter().collect();

        // Concatenate per-run, in request order (no global re-sort).
        let mut out: Vec<MetricWithRunId> = Vec::new();
        for rid in run_ids {
            out.extend(
                self.metric_history_from_steps(rid, metric_key, &steps)
                    .await?,
            );
        }
        Ok(out)
    }

    /// `get_metric_history_bulk_interval_from_steps` for one run: rows where
    /// `step IN (steps)`, ordered `(run_uuid, step, timestamp, value)`, limited
    /// to `MAX_RESULTS_GET_METRIC_HISTORY` (25000).
    async fn metric_history_from_steps(
        &self,
        run_id: &str,
        metric_key: &str,
        steps: &[i64],
    ) -> Result<Vec<MetricWithRunId>, MlflowError> {
        if steps.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut vals: Vec<Val> = vec![
            Val::Text(metric_key.to_string()),
            Val::Text(run_id.to_string()),
        ];
        let step_ph: Vec<String> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                vals.push(Val::Int(*s));
                ph(i + 3)
            })
            .collect();
        let sql = format!(
            "SELECT run_uuid, \"key\", value, timestamp, step, is_nan FROM metrics \
             WHERE \"key\" = {} AND run_uuid = {} AND step IN ({}) \
             ORDER BY run_uuid, step, timestamp, value \
             LIMIT {}",
            ph(1),
            ph(2),
            step_ph.join(", "),
            GET_METRIC_HISTORY_MAX_RESULTS,
        );
        self.db()
            .fetch_all(&sql, &vals, metric_with_run_id_from_row)
            .await
            .map_err(internal)
    }

    /// Distinct steps across the runs for `metric_key`, ascending.
    async fn distinct_steps(
        &self,
        run_ids: &[&str],
        metric_key: &str,
    ) -> Result<Vec<i64>, MlflowError> {
        if run_ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut vals: Vec<Val> = vec![Val::Text(metric_key.to_string())];
        let placeholders: Vec<String> = run_ids
            .iter()
            .enumerate()
            .map(|(i, rid)| {
                vals.push(Val::Text(rid.to_string()));
                ph(i + 2)
            })
            .collect();
        let sql = format!(
            "SELECT DISTINCT step FROM metrics \
             WHERE \"key\" = {} AND run_uuid IN ({}) ORDER BY step",
            ph(1),
            placeholders.join(", "),
        );
        self.db()
            .fetch_all(&sql, &vals, |r| r.get_i64("step"))
            .await
            .map_err(internal)
    }

    /// Per-run `(MIN(step), MAX(step))` for `metric_key`.
    async fn per_run_min_max_steps(
        &self,
        run_ids: &[&str],
        metric_key: &str,
    ) -> Result<Vec<(i64, i64)>, MlflowError> {
        if run_ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut vals: Vec<Val> = vec![Val::Text(metric_key.to_string())];
        let placeholders: Vec<String> = run_ids
            .iter()
            .enumerate()
            .map(|(i, rid)| {
                vals.push(Val::Text(rid.to_string()));
                ph(i + 2)
            })
            .collect();
        let sql = format!(
            "SELECT MIN(step) AS min_step, MAX(step) AS max_step FROM metrics \
             WHERE \"key\" = {} AND run_uuid IN ({}) GROUP BY run_uuid",
            ph(1),
            placeholders.join(", "),
        );
        self.db()
            .fetch_all(&sql, &vals, |r| {
                Ok((r.get_i64("min_step")?, r.get_i64("max_step")?))
            })
            .await
            .map_err(internal)
    }

    /// Filter `run_ids` to those whose experiment is in `workspace`
    /// (`_filter_entity_ids(RUN)`), preserving input order and dropping others.
    async fn filter_run_ids(
        &self,
        workspace: &str,
        run_ids: &[&str],
    ) -> Result<Vec<String>, MlflowError> {
        if run_ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let ph = |i| dialect.placeholder(i);
        let mut vals: Vec<Val> = Vec::with_capacity(run_ids.len() + 1);
        let placeholders: Vec<String> = run_ids
            .iter()
            .enumerate()
            .map(|(i, rid)| {
                vals.push(Val::Text(rid.to_string()));
                ph(i + 1)
            })
            .collect();
        vals.push(Val::Text(workspace.to_string()));
        let sql = format!(
            "SELECT r.run_uuid AS run_uuid FROM runs r \
             WHERE r.run_uuid IN ({}) AND r.experiment_id IN \
             (SELECT experiment_id FROM experiments WHERE workspace = {})",
            placeholders.join(", "),
            ph(run_ids.len() + 1)
        );
        let found: Vec<String> = self
            .db()
            .fetch_all(&sql, &vals, |r| r.get_string("run_uuid"))
            .await
            .map_err(internal)?;
        Ok(run_ids
            .iter()
            .filter(|rid| found.iter().any(|f| f == *rid))
            .map(|rid| rid.to_string())
            .collect())
    }
}

fn metric_with_run_id_from_row(
    r: &dyn super::dbutil::RowLike,
) -> Result<MetricWithRunId, sqlx::Error> {
    let is_nan = r.get_bool("is_nan")?;
    let stored = r.get_f64("value")?;
    Ok(MetricWithRunId {
        run_id: r.get_string("run_uuid")?,
        metric: Metric {
            key: r.get_string("key")?,
            value: if is_nan { f64::NAN } else { stored },
            timestamp: r.get_opt_i64("timestamp")?.unwrap_or(0),
            step: r.get_i64("step")?,
        },
    })
}

/// `bisect.bisect_left`: first index `i` with `a[i] >= x`.
fn bisect_left(a: &[i64], x: i64) -> usize {
    a.partition_point(|&v| v < x)
}

/// `bisect.bisect_right`: first index `i` with `a[i] > x`.
fn bisect_right(a: &[i64], x: i64) -> usize {
    a.partition_point(|&v| v <= x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisect_matches_python() {
        let a = [1, 2, 2, 3, 5];
        // bisect_left
        assert_eq!(bisect_left(&a, 0), 0);
        assert_eq!(bisect_left(&a, 2), 1);
        assert_eq!(bisect_left(&a, 4), 4);
        assert_eq!(bisect_left(&a, 6), 5);
        // bisect_right
        assert_eq!(bisect_right(&a, 2), 3);
        assert_eq!(bisect_right(&a, 5), 5);
        assert_eq!(bisect_right(&a, 0), 0);
    }

    /// The exact index arithmetic of the sampling branch, isolated so it can be
    /// checked against Python's `int(i * interval)` semantics.
    fn sample_indices(start_idx: usize, end_idx: usize, max_results: usize) -> BTreeSet<usize> {
        let window = end_idx - start_idx;
        let mut out = BTreeSet::new();
        if window <= max_results {
            for idx in start_idx..end_idx {
                out.insert(idx);
            }
        } else {
            let interval = window as f64 / max_results as f64;
            for i in 0..max_results {
                let idx = start_idx + (i as f64 * interval) as usize;
                if idx < end_idx {
                    out.insert(idx);
                }
            }
            out.insert(end_idx - 1);
        }
        out
    }

    #[test]
    fn sampling_index_math() {
        // window <= max_results: keep everything.
        assert_eq!(sample_indices(0, 5, 10).len(), 5);
        // window > max_results: even spacing + forced endpoint.
        let idxs = sample_indices(0, 100, 10);
        assert!(idxs.contains(&0));
        assert!(idxs.contains(&99)); // forced endpoint = end_idx - 1
        assert!(idxs.len() <= 11);
    }
}
