//! Scorer registration and online-scoring configuration tables.

pub const SCORERS: &str = "scorers";
pub const SCORER_VERSIONS: &str = "scorer_versions";
pub const ONLINE_SCORING_CONFIGS: &str = "online_scoring_configs";

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SqlScorer {
    pub experiment_id: i64,
    pub scorer_name: String,
    pub scorer_id: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SqlScorerVersion {
    pub scorer_id: String,
    pub scorer_version: i64,
    pub serialized_scorer: String,
    pub creation_time: Option<i64>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SqlOnlineScoringConfig {
    pub online_scoring_config_id: String,
    pub scorer_id: String,
    pub sample_rate: f64,
    pub experiment_id: i64,
    pub filter_string: Option<String>,
}
