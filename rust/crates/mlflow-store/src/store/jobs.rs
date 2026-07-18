//! Generic job persistence and D20 database-queue claiming.
//!
//! The schema and lifecycle mirror `SqlAlchemyJobStore`. Every public method
//! takes a workspace explicitly so the single-tenant and workspace-aware paths
//! share one implementation without weakening isolation.

use std::fmt;

use mlflow_error::MlflowError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::dbutil::{RowLike, Tx, Val};
use crate::schema::jobs::JOBS;
use crate::{Db, Dialect};

const PENDING: i64 = 0;
const RUNNING: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum JobStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Timeout,
    Canceled,
}

impl JobStatus {
    pub const fn to_int(self) -> i64 {
        match self {
            Self::Pending => 0,
            Self::Running => 1,
            Self::Succeeded => 2,
            Self::Failed => 3,
            Self::Timeout => 4,
            Self::Canceled => 5,
        }
    }

    pub fn from_int(value: i64) -> Result<Self, MlflowError> {
        match value {
            0 => Ok(Self::Pending),
            1 => Ok(Self::Running),
            2 => Ok(Self::Succeeded),
            3 => Ok(Self::Failed),
            4 => Ok(Self::Timeout),
            5 => Ok(Self::Canceled),
            _ => Err(MlflowError::invalid_parameter_value(format!(
                "The value {value} can't be converted to JobStatus enum value."
            ))),
        }
    }

    pub const fn is_finalized(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Timeout | Self::Canceled
        )
    }
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pending => "PENDING",
            Self::Running => "RUNNING",
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
            Self::Timeout => "TIMEOUT",
            Self::Canceled => "CANCELED",
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    pub job_id: String,
    pub creation_time: i64,
    pub job_name: String,
    pub params: String,
    pub workspace: String,
    pub timeout: Option<f64>,
    pub status: JobStatus,
    pub result: Option<String>,
    pub retry_count: i64,
    pub last_update_time: i64,
    pub status_details: Option<Value>,
}

impl Job {
    pub fn parsed_result(&self) -> Result<Option<Value>, MlflowError> {
        match (&self.status, &self.result) {
            (JobStatus::Succeeded, Some(result)) => serde_json::from_str(result)
                .map(Some)
                .map_err(|error| MlflowError::internal_error(error.to_string())),
            (_, Some(result)) => Ok(Some(Value::String(result.clone()))),
            (_, None) => Ok(None),
        }
    }
}

/// Store over the pre-existing Alembic-managed `jobs` table.
#[derive(Debug, Clone)]
pub struct JobStore {
    db: Db,
}

impl JobStore {
    pub fn new(db: Db) -> Self {
        Self { db }
    }

    pub fn db(&self) -> &Db {
        &self.db
    }

    pub async fn create_job(
        &self,
        workspace: &str,
        job_name: &str,
        params: &str,
        timeout: Option<f64>,
    ) -> Result<Job, MlflowError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_millis();
        let p = |index| self.db.dialect().placeholder(index);
        let sql = format!(
            "INSERT INTO {JOBS} (id, creation_time, job_name, params, workspace, timeout, \
             status, result, retry_count, last_update_time, status_details) \
             VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            p(1),
            p(2),
            p(3),
            p(4),
            p(5),
            p(6),
            p(7),
            p(8),
            p(9),
            p(10),
            p(11)
        );
        self.db
            .exec(
                &sql,
                &[
                    Val::Text(id.clone()),
                    Val::Int(now),
                    Val::Text(job_name.to_string()),
                    Val::Text(params.to_string()),
                    Val::Text(workspace.to_string()),
                    Val::OptFloat(timeout),
                    Val::Int(PENDING),
                    Val::OptText(None),
                    Val::Int(0),
                    Val::Int(now),
                    Val::OptJson(None),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_job(workspace, &id).await
    }

    pub async fn get_job(&self, workspace: &str, job_id: &str) -> Result<Job, MlflowError> {
        self.fetch_job(workspace, job_id).await?.ok_or_else(|| {
            MlflowError::resource_does_not_exist(format!("Job with ID {job_id} not found"))
        })
    }

    pub async fn list_jobs(
        &self,
        workspace: &str,
        job_name: Option<&str>,
        statuses: &[JobStatus],
        begin_timestamp: Option<i64>,
        end_timestamp: Option<i64>,
        params: Option<&Value>,
    ) -> Result<Vec<Job>, MlflowError> {
        let dialect = self.db.dialect();
        let mut vals = vec![Val::Text(workspace.to_string())];
        let mut clauses = vec![format!("workspace = {}", dialect.placeholder(1))];
        if let Some(job_name) = job_name {
            vals.push(Val::Text(job_name.to_string()));
            clauses.push(format!("job_name = {}", dialect.placeholder(vals.len())));
        }
        if !statuses.is_empty() {
            let mut placeholders = Vec::with_capacity(statuses.len());
            for status in statuses {
                vals.push(Val::Int(status.to_int()));
                placeholders.push(dialect.placeholder(vals.len()));
            }
            clauses.push(format!("status IN ({})", placeholders.join(", ")));
        }
        if let Some(begin) = begin_timestamp {
            vals.push(Val::Int(begin));
            clauses.push(format!(
                "creation_time >= {}",
                dialect.placeholder(vals.len())
            ));
        }
        if let Some(end) = end_timestamp {
            vals.push(Val::Int(end));
            clauses.push(format!(
                "creation_time <= {}",
                dialect.placeholder(vals.len())
            ));
        }
        let sql = format!(
            "SELECT {} FROM {JOBS} WHERE {} ORDER BY creation_time",
            job_columns(),
            clauses.join(" AND ")
        );
        let jobs = self
            .db
            .fetch_all(&sql, &vals, row_to_job)
            .await
            .map_err(internal)?;
        let Some(filter) = params
            .and_then(Value::as_object)
            .filter(|filter| !filter.is_empty())
        else {
            return Ok(jobs);
        };
        jobs.into_iter()
            .filter_map(|job| {
                let parsed = serde_json::from_str::<Value>(&job.params);
                match parsed {
                    Ok(Value::Object(values))
                        if filter
                            .iter()
                            .all(|(key, value)| values.get(key) == Some(value)) =>
                    {
                        Some(Ok(job))
                    }
                    Ok(_) => None,
                    Err(error) => Some(Err(MlflowError::internal_error(error.to_string()))),
                }
            })
            .collect()
    }

    pub async fn start_job(&self, workspace: &str, job_id: &str) -> Result<Job, MlflowError> {
        let changed = self
            .conditional_status_update(workspace, job_id, PENDING, JobStatus::Running)
            .await?;
        if changed == 0 {
            let job = self.get_job(workspace, job_id).await?;
            return Err(MlflowError::internal_error(format!(
                "Job {job_id} is in {} state, cannot start (must be PENDING)",
                job.status
            )));
        }
        self.get_job(workspace, job_id).await
    }

    pub async fn reset_job(&self, workspace: &str, job_id: &str) -> Result<Job, MlflowError> {
        self.transition_job(workspace, job_id, JobStatus::Pending, None)
            .await
    }

    pub async fn finish_job(
        &self,
        workspace: &str,
        job_id: &str,
        result: &str,
    ) -> Result<Job, MlflowError> {
        self.transition_job(
            workspace,
            job_id,
            JobStatus::Succeeded,
            Some(result.to_string()),
        )
        .await
    }

    pub async fn fail_job(
        &self,
        workspace: &str,
        job_id: &str,
        error: &str,
    ) -> Result<Job, MlflowError> {
        self.transition_job(
            workspace,
            job_id,
            JobStatus::Failed,
            Some(error.to_string()),
        )
        .await
    }

    pub async fn mark_job_timed_out(
        &self,
        workspace: &str,
        job_id: &str,
    ) -> Result<Job, MlflowError> {
        self.transition_job(workspace, job_id, JobStatus::Timeout, None)
            .await
    }

    pub async fn cancel_job(&self, workspace: &str, job_id: &str) -> Result<Job, MlflowError> {
        self.transition_job(workspace, job_id, JobStatus::Canceled, None)
            .await
    }

    pub async fn retry_or_fail_job(
        &self,
        workspace: &str,
        job_id: &str,
        error: &str,
        max_retries: i64,
    ) -> Result<Option<i64>, MlflowError> {
        let mut tx = self.db.begin_tx().await.map_err(internal)?;
        let Some(job) = fetch_job_tx(&mut tx, self.db.dialect(), workspace, job_id, true).await?
        else {
            return Err(MlflowError::resource_does_not_exist(format!(
                "Job with ID {job_id} not found"
            )));
        };
        reject_finalized(&job)?;
        let p = |index| self.db.dialect().placeholder(index);
        let (status, result, retry_count, last_update_time, return_value) =
            if job.retry_count >= max_retries {
                (
                    JobStatus::Failed,
                    Some(error.to_string()),
                    job.retry_count,
                    job.last_update_time,
                    None,
                )
            } else {
                let retry_count = job.retry_count + 1;
                (
                    JobStatus::Pending,
                    job.result,
                    retry_count,
                    now_millis(),
                    Some(retry_count),
                )
            };
        let sql =
            format!(
            "UPDATE {JOBS} SET status = {}, result = {}, retry_count = {}, last_update_time = {} \
             WHERE id = {} AND workspace = {} AND status IN (0, 1)",
            p(1), p(2), p(3), p(4), p(5), p(6)
        );
        let changed = tx
            .exec(
                &sql,
                &[
                    Val::Int(status.to_int()),
                    Val::OptText(result),
                    Val::Int(retry_count),
                    Val::Int(last_update_time),
                    Val::Text(job_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        if changed == 0 {
            return Err(finalized_error(job_id, job.status));
        }
        tx.commit().await.map_err(internal)?;
        Ok(return_value)
    }

    pub async fn update_status_details(
        &self,
        workspace: &str,
        job_id: &str,
        details: &Value,
    ) -> Result<Job, MlflowError> {
        let Some(new_details) = details.as_object() else {
            return Err(MlflowError::invalid_parameter_value(
                "Job status details must be a JSON object.",
            ));
        };
        let mut tx = self.db.begin_tx().await.map_err(internal)?;
        let Some(job) = fetch_job_tx(&mut tx, self.db.dialect(), workspace, job_id, true).await?
        else {
            return Err(MlflowError::resource_does_not_exist(format!(
                "Job with ID {job_id} not found"
            )));
        };
        let mut merged = job
            .status_details
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        merged.extend(new_details.clone());
        let p = |index| self.db.dialect().placeholder(index);
        let sql = format!(
            "UPDATE {JOBS} SET status_details = {}, last_update_time = {} \
             WHERE id = {} AND workspace = {}",
            p(1),
            p(2),
            p(3),
            p(4)
        );
        tx.exec(
            &sql,
            &[
                Val::OptJson(Some(Value::Object(merged))),
                Val::Int(now_millis()),
                Val::Text(job_id.to_string()),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)?;
        self.get_job(workspace, job_id).await
    }

    pub async fn delete_jobs(
        &self,
        workspace: &str,
        older_than: i64,
        job_ids: &[String],
    ) -> Result<Vec<String>, MlflowError> {
        let dialect = self.db.dialect();
        let mut vals = vec![Val::Text(workspace.to_string())];
        let mut clauses = vec![format!("workspace = {}", dialect.placeholder(1))];
        clauses.push("status IN (2, 3, 4, 5)".to_string());
        if !job_ids.is_empty() {
            let mut placeholders = Vec::with_capacity(job_ids.len());
            for id in job_ids {
                vals.push(Val::Text(id.clone()));
                placeholders.push(dialect.placeholder(vals.len()));
            }
            clauses.push(format!("id IN ({})", placeholders.join(", ")));
        }
        if older_than > 0 {
            vals.push(Val::Int(now_millis() - older_than));
            clauses.push(format!(
                "creation_time < {}",
                dialect.placeholder(vals.len())
            ));
        }
        let where_clause = clauses.join(" AND ");
        let select = format!("SELECT id FROM {JOBS} WHERE {where_clause}");
        let ids = self
            .db
            .fetch_all(&select, &vals, |row| row.get_string("id"))
            .await
            .map_err(internal)?;
        if !ids.is_empty() {
            let delete = format!("DELETE FROM {JOBS} WHERE {where_clause}");
            self.db.exec(&delete, &vals).await.map_err(internal)?;
        }
        Ok(ids)
    }

    /// Atomically claim the oldest pending job for a workspace/function.
    ///
    /// PostgreSQL uses one `FOR UPDATE SKIP LOCKED` CTE; MySQL uses the same
    /// row-lock clause inside a transaction because it has no `UPDATE ...
    /// RETURNING`; SQLite uses one conditional `UPDATE ... RETURNING`, whose
    /// writer serialization is the appropriate equivalent for its database-
    /// wide write lock.
    pub async fn claim_next_job(
        &self,
        workspace: &str,
        job_name: Option<&str>,
    ) -> Result<Option<Job>, MlflowError> {
        match self.db.dialect() {
            Dialect::Sqlite => self.claim_sqlite(workspace, job_name).await,
            Dialect::Postgres => self.claim_postgres(workspace, job_name).await,
            Dialect::MySql => self.claim_mysql(workspace, job_name).await,
        }
    }

    async fn claim_sqlite(
        &self,
        workspace: &str,
        job_name: Option<&str>,
    ) -> Result<Option<Job>, MlflowError> {
        let mut vals = vec![Val::Int(now_millis()), Val::Text(workspace.to_string())];
        let name_clause = if let Some(name) = job_name {
            vals.push(Val::Text(name.to_string()));
            format!(" AND job_name = {}", self.db.dialect().placeholder(3))
        } else {
            String::new()
        };
        let sql = format!(
            "UPDATE {JOBS} SET status = {RUNNING}, last_update_time = ? \
             WHERE id = (SELECT id FROM {JOBS} WHERE workspace = ? AND status = {PENDING}\
             {name_clause} ORDER BY creation_time, id LIMIT 1) AND status = {PENDING} \
             RETURNING {}",
            job_columns()
        );
        self.db
            .fetch_optional(&sql, &vals, row_to_job)
            .await
            .map_err(internal)
    }

    async fn claim_postgres(
        &self,
        workspace: &str,
        job_name: Option<&str>,
    ) -> Result<Option<Job>, MlflowError> {
        let mut vals = vec![Val::Text(workspace.to_string())];
        let name_clause = if let Some(name) = job_name {
            vals.push(Val::Text(name.to_string()));
            format!(" AND job_name = {}", self.db.dialect().placeholder(2))
        } else {
            String::new()
        };
        vals.push(Val::Int(now_millis()));
        let now_placeholder = self.db.dialect().placeholder(vals.len());
        let sql = format!(
            "WITH candidate AS (SELECT id FROM {JOBS} WHERE workspace = $1 AND status = {PENDING}\
             {name_clause} ORDER BY creation_time, id LIMIT 1 FOR UPDATE SKIP LOCKED) \
             UPDATE {JOBS} AS jobs SET status = {RUNNING}, last_update_time = {now_placeholder} \
             FROM candidate WHERE jobs.id = candidate.id AND jobs.status = {PENDING} \
             RETURNING {}",
            job_columns_qualified("jobs")
        );
        self.db
            .fetch_optional(&sql, &vals, row_to_job)
            .await
            .map_err(internal)
    }

    async fn claim_mysql(
        &self,
        workspace: &str,
        job_name: Option<&str>,
    ) -> Result<Option<Job>, MlflowError> {
        let mut tx = self.db.begin_tx().await.map_err(internal)?;
        let mut vals = vec![Val::Text(workspace.to_string())];
        let name_clause = if let Some(name) = job_name {
            vals.push(Val::Text(name.to_string()));
            " AND job_name = ?".to_string()
        } else {
            String::new()
        };
        let select = format!(
            "SELECT id FROM {JOBS} WHERE workspace = ? AND status = {PENDING}{name_clause} \
             ORDER BY creation_time, id LIMIT 1 FOR UPDATE SKIP LOCKED"
        );
        let Some(id) = tx
            .fetch_all(&select, &vals, |row| row.get_string("id"))
            .await
            .map_err(internal)?
            .into_iter()
            .next()
        else {
            tx.commit().await.map_err(internal)?;
            return Ok(None);
        };
        let update = format!(
            "UPDATE {JOBS} SET status = {RUNNING}, last_update_time = ? \
             WHERE id = ? AND workspace = ? AND status = {PENDING}"
        );
        let changed = tx
            .exec(
                &update,
                &[
                    Val::Int(now_millis()),
                    Val::Text(id.clone()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        if changed == 0 {
            tx.commit().await.map_err(internal)?;
            return Ok(None);
        }
        tx.commit().await.map_err(internal)?;
        self.fetch_job(workspace, &id).await
    }

    async fn transition_job(
        &self,
        workspace: &str,
        job_id: &str,
        status: JobStatus,
        result: Option<String>,
    ) -> Result<Job, MlflowError> {
        let p = |index| self.db.dialect().placeholder(index);
        let sql =
            format!(
            "UPDATE {JOBS} SET status = {}, result = COALESCE({}, result), last_update_time = {} \
             WHERE id = {} AND workspace = {} AND status IN (0, 1)",
            p(1), p(2), p(3), p(4), p(5)
        );
        let changed = self
            .db
            .exec(
                &sql,
                &[
                    Val::Int(status.to_int()),
                    Val::OptText(result),
                    Val::Int(now_millis()),
                    Val::Text(job_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        if changed == 0 {
            let job = self.get_job(workspace, job_id).await?;
            return Err(if job.status.is_finalized() {
                finalized_error(job_id, job.status)
            } else {
                MlflowError::internal_error(format!(
                    "Job {job_id} could not transition from {}",
                    job.status
                ))
            });
        }
        self.get_job(workspace, job_id).await
    }

    async fn conditional_status_update(
        &self,
        workspace: &str,
        job_id: &str,
        old_status: i64,
        new_status: JobStatus,
    ) -> Result<u64, MlflowError> {
        let p = |index| self.db.dialect().placeholder(index);
        let sql = format!(
            "UPDATE {JOBS} SET status = {}, last_update_time = {} \
             WHERE id = {} AND workspace = {} AND status = {}",
            p(1),
            p(2),
            p(3),
            p(4),
            p(5)
        );
        self.db
            .exec(
                &sql,
                &[
                    Val::Int(new_status.to_int()),
                    Val::Int(now_millis()),
                    Val::Text(job_id.to_string()),
                    Val::Text(workspace.to_string()),
                    Val::Int(old_status),
                ],
            )
            .await
            .map_err(internal)
    }

    async fn fetch_job(&self, workspace: &str, job_id: &str) -> Result<Option<Job>, MlflowError> {
        let p = |index| self.db.dialect().placeholder(index);
        let sql = format!(
            "SELECT {} FROM {JOBS} WHERE id = {} AND workspace = {}",
            job_columns(),
            p(1),
            p(2)
        );
        self.db
            .fetch_optional(
                &sql,
                &[
                    Val::Text(job_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                row_to_job,
            )
            .await
            .map_err(internal)
    }
}

async fn fetch_job_tx(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
    job_id: &str,
    lock: bool,
) -> Result<Option<Job>, MlflowError> {
    let lock_clause = if lock && dialect != Dialect::Sqlite {
        " FOR UPDATE"
    } else {
        ""
    };
    let sql = format!(
        "SELECT {} FROM {JOBS} WHERE id = {} AND workspace = {}{lock_clause}",
        job_columns(),
        dialect.placeholder(1),
        dialect.placeholder(2)
    );
    tx.fetch_all(
        &sql,
        &[
            Val::Text(job_id.to_string()),
            Val::Text(workspace.to_string()),
        ],
        row_to_job,
    )
    .await
    .map(|rows| rows.into_iter().next())
    .map_err(internal)
}

fn job_columns() -> &'static str {
    "id, creation_time, job_name, params, workspace, timeout, status, result, \
     retry_count, last_update_time, status_details"
}

fn job_columns_qualified(table: &str) -> String {
    job_columns()
        .split(", ")
        .map(|column| format!("{table}.{}", column.trim()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn row_to_job(row: &dyn RowLike) -> Result<Job, sqlx::Error> {
    let status = JobStatus::from_int(row.get_int("status")?).map_err(|error| {
        sqlx::Error::Decode(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            error.message,
        )))
    })?;
    Ok(Job {
        job_id: row.get_string("id")?,
        creation_time: row.get_i64("creation_time")?,
        job_name: row.get_string("job_name")?,
        params: row.get_string("params")?,
        workspace: row.get_string("workspace")?,
        timeout: row.get_opt_f64("timeout")?,
        status,
        result: row.get_opt_string("result")?,
        retry_count: row.get_int("retry_count")?,
        last_update_time: row.get_i64("last_update_time")?,
        status_details: row.get_opt_json("status_details")?,
    })
}

fn reject_finalized(job: &Job) -> Result<(), MlflowError> {
    if job.status.is_finalized() {
        Err(finalized_error(&job.job_id, job.status))
    } else {
        Ok(())
    }
}

fn finalized_error(job_id: &str, status: JobStatus) -> MlflowError {
    MlflowError::internal_error(format!(
        "The Job {job_id} is already finalized with status: {status}, it can't be updated."
    ))
}

fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn internal(error: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("Database error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_ordinals_match_python_enum_order() {
        let statuses = [
            JobStatus::Pending,
            JobStatus::Running,
            JobStatus::Succeeded,
            JobStatus::Failed,
            JobStatus::Timeout,
            JobStatus::Canceled,
        ];
        for (ordinal, status) in statuses.into_iter().enumerate() {
            assert_eq!(status.to_int(), ordinal as i64);
            assert_eq!(JobStatus::from_int(ordinal as i64).unwrap(), status);
        }
        assert_eq!(
            statuses
                .into_iter()
                .filter(|status| status.is_finalized())
                .map(JobStatus::to_int)
                .collect::<Vec<_>>(),
            [2, 3, 4, 5]
        );
    }
}
