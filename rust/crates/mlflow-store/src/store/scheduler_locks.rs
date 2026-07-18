//! Durable periodic-task locks backed by the shared jobs database.
//!
//! Python's `lock_task("online-scoring-scheduler-lock")` is a compare-and-set
//! key in Huey's SQLite storage. Rust keeps the same named database-lock
//! discipline in the shared backend DB so independently deployed servers
//! exclude one another without an in-process mutex.

use mlflow_error::MlflowError;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::dbutil::Val;
use super::experiments::{internal, is_unique_violation, now_millis};
use super::JobStore;
use crate::schema::jobs::JOBS;

const LOCK_JOB_NAME: &str = "__mlflow_periodic_scheduler_lock__";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodicSchedulerLock {
    id: String,
    owner: String,
}

impl JobStore {
    /// Atomically acquire a named cross-process scheduler lease.
    ///
    /// The row is `RUNNING`, so a job runner cannot claim it. A bounded lease
    /// lets another server recover after an ungraceful process exit; normal
    /// completion deletes the row using the owner token as a fencing check.
    pub async fn try_acquire_periodic_scheduler_lock(
        &self,
        name: &str,
        lease_ms: i64,
    ) -> Result<Option<PeriodicSchedulerLock>, MlflowError> {
        let id = lock_id(name);
        let owner = Uuid::new_v4().to_string();
        let now = now_millis();
        let dialect = self.db().dialect();
        let p = |index| dialect.placeholder(index);
        let insert = format!(
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
        let values = [
            Val::Text(id.clone()),
            Val::Int(now),
            Val::Text(LOCK_JOB_NAME.to_string()),
            Val::Text(owner.clone()),
            Val::Text("default".to_string()),
            Val::OptFloat(None),
            Val::Int(1),
            Val::OptText(None),
            Val::Int(0),
            Val::Int(now),
            Val::OptJson(None),
        ];
        match self.db().exec(&insert, &values).await {
            Ok(_) => {
                return Ok(Some(PeriodicSchedulerLock { id, owner }));
            }
            Err(error) if is_unique_violation(&error) => {}
            Err(error) => return Err(internal(error)),
        }

        let update = format!(
            "UPDATE {JOBS} SET creation_time = {}, params = {}, last_update_time = {} \
             WHERE id = {} AND job_name = {} AND last_update_time < {}",
            p(1),
            p(2),
            p(3),
            p(4),
            p(5),
            p(6)
        );
        let changed = self
            .db()
            .exec(
                &update,
                &[
                    Val::Int(now),
                    Val::Text(owner.clone()),
                    Val::Int(now),
                    Val::Text(id.clone()),
                    Val::Text(LOCK_JOB_NAME.to_string()),
                    Val::Int(now.saturating_sub(lease_ms.max(1))),
                ],
            )
            .await
            .map_err(internal)?;
        Ok((changed == 1).then_some(PeriodicSchedulerLock { id, owner }))
    }

    pub async fn release_periodic_scheduler_lock(
        &self,
        lock: &PeriodicSchedulerLock,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM {JOBS} WHERE id = {} AND job_name = {} AND params = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(lock.id.clone()),
                    Val::Text(LOCK_JOB_NAME.to_string()),
                    Val::Text(lock.owner.clone()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }
}

fn lock_id(name: &str) -> String {
    let digest = Sha256::digest(name.as_bytes());
    format!("{:x}", digest)[..32].to_string()
}
