//! Generic jobs table used as the native worker queue (plan D20).

use sqlx::FromRow;

pub const JOBS: &str = "jobs";

/// Physical row shape of `SqlJob` in
/// `mlflow/store/tracking/dbmodels/models.py`.
#[derive(Debug, Clone, PartialEq, FromRow)]
pub struct Job {
    pub id: String,
    pub creation_time: i64,
    pub job_name: String,
    pub params: String,
    pub workspace: String,
    pub timeout: Option<f64>,
    pub status: i64,
    pub result: Option<String>,
    pub retry_count: i64,
    pub last_update_time: i64,
    pub status_details: Option<serde_json::Value>,
}
