//! Registered scorer versions and online-scoring configurations.

use mlflow_error::MlflowError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::dbutil::{RowLike, Val};
use super::experiments::{internal, is_unique_violation, now_millis, parse_experiment_id};
use super::{python_json_dumps, LifecycleStage, TrackingStore};
use crate::schema::scorers::{ONLINE_SCORING_CONFIGS, SCORERS, SCORER_VERSIONS};

const ENDPOINTS: &str = "endpoints";
const ENDPOINT_BINDINGS: &str = "endpoint_bindings";
const REGISTER_RETRIES: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ScorerVersion {
    pub experiment_id: String,
    pub scorer_name: String,
    pub scorer_version: i32,
    pub serialized_scorer: String,
    pub creation_time: Option<i64>,
    pub scorer_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineScoringConfig {
    pub online_scoring_config_id: String,
    pub scorer_id: String,
    pub sample_rate: f64,
    pub experiment_id: String,
    pub filter_string: Option<String>,
}

/// The latest serialized scorer version paired with an enabled online config.
/// Field order intentionally matches Python's `OnlineScorer` dataclass so
/// `python_json_dumps` produces submission parameters in the same order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineScorer {
    pub name: String,
    pub serialized_scorer: String,
    pub online_config: OnlineScoringConfig,
}

impl TrackingStore {
    /// Return active online scorers in a workspace (`sample_rate > 0`) using
    /// each scorer's latest registered version. Persisted gateway endpoint IDs
    /// are resolved back to endpoint names, matching Python's
    /// `SqlAlchemyStore.get_active_online_scorers`.
    pub async fn get_active_online_scorers(
        &self,
        workspace: &str,
    ) -> Result<Vec<OnlineScorer>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT c.online_scoring_config_id, c.scorer_id, c.sample_rate, \
             c.experiment_id, c.filter_string, s.scorer_name, v.scorer_version, \
             v.serialized_scorer, v.creation_time \
             FROM {ONLINE_SCORING_CONFIGS} c \
             JOIN experiments e ON e.experiment_id = c.experiment_id \
             JOIN {SCORERS} s ON s.scorer_id = c.scorer_id \
             JOIN {SCORER_VERSIONS} v ON v.scorer_id = s.scorer_id \
              AND v.scorer_version = (SELECT MAX(v2.scorer_version) \
               FROM {SCORER_VERSIONS} v2 WHERE v2.scorer_id = s.scorer_id) \
             WHERE e.workspace = {} AND c.sample_rate > 0",
            dialect.placeholder(1)
        );
        let rows = self
            .db()
            .fetch_all(&sql, &[Val::Text(workspace.to_string())], |row| {
                Ok((
                    OnlineScoringConfig {
                        online_scoring_config_id: row.get_string("online_scoring_config_id")?,
                        scorer_id: row.get_string("scorer_id")?,
                        sample_rate: row.get_f64("sample_rate")?,
                        experiment_id: row.get_int("experiment_id")?.to_string(),
                        filter_string: row.get_opt_string("filter_string")?,
                    },
                    ScorerVersion {
                        experiment_id: row.get_int("experiment_id")?.to_string(),
                        scorer_name: row.get_string("scorer_name")?,
                        scorer_version: row.get_int("scorer_version")? as i32,
                        serialized_scorer: row.get_string("serialized_scorer")?,
                        creation_time: row.get_opt_i64("creation_time")?,
                        scorer_id: row.get_string("scorer_id")?,
                    },
                ))
            })
            .await
            .map_err(internal)?;

        let mut active = Vec::with_capacity(rows.len());
        for (online_config, mut scorer) in rows {
            // Python drops configs whose latest scorer version no longer uses
            // a gateway model, even if an older version did.
            let serialized: Value =
                serde_json::from_str(&scorer.serialized_scorer).map_err(|error| {
                    MlflowError::internal_error(format!("stored scorer is not valid JSON: {error}"))
                })?;
            if extract_model(&serialized)?
                .as_deref()
                .and_then(gateway_endpoint_ref)
                .is_none()
            {
                continue;
            }
            self.resolve_scorer(workspace, &mut scorer).await?;
            active.push(OnlineScorer {
                name: scorer.scorer_name,
                serialized_scorer: scorer.serialized_scorer,
                online_config,
            });
        }
        Ok(active)
    }

    pub async fn register_scorer(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
        serialized_scorer: &str,
    ) -> Result<ScorerVersion, MlflowError> {
        validate_scorer_name(name)?;
        let experiment_id_num = self
            .require_active_scorer_experiment(workspace, experiment_id)
            .await?;
        let mut serialized_data: Value =
            serde_json::from_str(serialized_scorer).map_err(|error| {
                MlflowError::invalid_parameter_value(format!(
                    "serialized scorer is not valid JSON: {error}"
                ))
            })?;
        let model = extract_model(&serialized_data)?;
        validate_scorer_model(model.as_deref())?;

        let endpoint = if let Some(endpoint_ref) = model.as_deref().and_then(gateway_endpoint_ref) {
            let (endpoint_id, endpoint_name) = self
                .endpoint_by_name(workspace, endpoint_ref)
                .await?
                .ok_or_else(|| {
                    MlflowError::resource_does_not_exist(format!(
                        "GatewayEndpoint not found (name='{endpoint_ref}')"
                    ))
                })?;
            update_model(
                &mut serialized_data,
                Some(&format!("gateway:/{endpoint_id}")),
            );
            (Some(endpoint_id), Some(endpoint_name))
        } else {
            (None, None)
        };
        let persisted = if endpoint.0.is_some() {
            python_json_dumps(&serialized_data, false)
        } else {
            serialized_scorer.to_string()
        };

        for attempt in 0..REGISTER_RETRIES {
            match self
                .register_scorer_once(experiment_id_num, name, &persisted, endpoint.0.as_deref())
                .await
            {
                Ok(mut scorer) => {
                    if let Some(endpoint_name) = endpoint.1.as_deref() {
                        let mut returned: Value = serde_json::from_str(&scorer.serialized_scorer)
                            .map_err(|error| {
                            MlflowError::internal_error(format!(
                                "registered scorer JSON became invalid: {error}"
                            ))
                        })?;
                        update_model(&mut returned, Some(&format!("gateway:/{endpoint_name}")));
                        scorer.serialized_scorer = python_json_dumps(&returned, false);
                    }
                    return Ok(scorer);
                }
                Err(error) if attempt + 1 < REGISTER_RETRIES && retryable_registration(&error) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => return Err(internal(error)),
            }
        }
        unreachable!("registration retry loop always returns")
    }

    async fn register_scorer_once(
        &self,
        experiment_id: i64,
        name: &str,
        serialized_scorer: &str,
        endpoint_id: Option<&str>,
    ) -> Result<ScorerVersion, sqlx::Error> {
        let dialect = self.db().dialect();
        let ph = |index| dialect.placeholder(index);
        let mut tx = self.db().begin_tx().await?;
        let scorer_id = match tx
            .fetch_all(
                &format!(
                    "SELECT scorer_id FROM {SCORERS} WHERE experiment_id = {} AND scorer_name = {}",
                    ph(1),
                    ph(2)
                ),
                &[Val::Int(experiment_id), Val::Text(name.to_string())],
                |row| row.get_string("scorer_id"),
            )
            .await?
            .into_iter()
            .next()
        {
            Some(scorer_id) => scorer_id,
            None => {
                let scorer_id = Uuid::new_v4().to_string();
                tx.exec(
                    &format!(
                        "INSERT INTO {SCORERS} (experiment_id, scorer_name, scorer_id) VALUES ({}, {}, {})",
                        ph(1),
                        ph(2),
                        ph(3)
                    ),
                    &[
                        Val::Int(experiment_id),
                        Val::Text(name.to_string()),
                        Val::Text(scorer_id.clone()),
                    ],
                )
                .await?;
                scorer_id
            }
        };

        let max_version = tx
            .fetch_all(
                &format!(
                    "SELECT MAX(scorer_version) AS max_version FROM {SCORER_VERSIONS} WHERE scorer_id = {}",
                    ph(1)
                ),
                &[Val::Text(scorer_id.clone())],
                |row| row.get_opt_i64("max_version"),
            )
            .await?
            .into_iter()
            .next()
            .flatten();
        let version = max_version.unwrap_or(0) + 1;
        let creation_time = now_millis();
        tx.exec(
            &format!(
                "INSERT INTO {SCORER_VERSIONS} (scorer_id, scorer_version, serialized_scorer, creation_time) VALUES ({}, {}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
                ph(4)
            ),
            &[
                Val::Text(scorer_id.clone()),
                Val::Int(version),
                Val::Text(serialized_scorer.to_string()),
                Val::Int(creation_time),
            ],
        )
        .await?;

        if let Some(endpoint_id) = endpoint_id {
            tx.exec(
                &format!(
                    "DELETE FROM {ENDPOINT_BINDINGS} WHERE resource_type = {} AND resource_id = {}",
                    ph(1),
                    ph(2)
                ),
                &[
                    Val::Text("scorer".to_string()),
                    Val::Text(scorer_id.clone()),
                ],
            )
            .await?;
            tx.exec(
                &format!(
                    "INSERT INTO {ENDPOINT_BINDINGS} (endpoint_id, resource_type, resource_id, created_at, created_by, last_updated_at, last_updated_by, display_name) VALUES ({}, {}, {}, {}, {}, {}, {}, {})",
                    ph(1), ph(2), ph(3), ph(4), ph(5), ph(6), ph(7), ph(8)
                ),
                &[
                    Val::Text(endpoint_id.to_string()),
                    Val::Text("scorer".to_string()),
                    Val::Text(scorer_id.clone()),
                    Val::Int(creation_time),
                    Val::OptText(None),
                    Val::Int(creation_time),
                    Val::OptText(None),
                    Val::Text(name.to_string()),
                ],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(ScorerVersion {
            experiment_id: experiment_id.to_string(),
            scorer_name: name.to_string(),
            scorer_version: version as i32,
            serialized_scorer: serialized_scorer.to_string(),
            creation_time: Some(creation_time),
            scorer_id,
        })
    }

    pub async fn list_scorers(
        &self,
        workspace: &str,
        experiment_id: Option<&str>,
    ) -> Result<Vec<ScorerVersion>, MlflowError> {
        let experiment_id_num = match experiment_id.filter(|value| !value.is_empty()) {
            Some(experiment_id) => Some(
                self.require_active_scorer_experiment(workspace, experiment_id)
                    .await?,
            ),
            None => None,
        };
        let dialect = self.db().dialect();
        let ph = |index| dialect.placeholder(index);
        let mut values = vec![Val::Text(workspace.to_string())];
        let experiment_filter = if let Some(experiment_id) = experiment_id_num {
            values.push(Val::Int(experiment_id));
            format!(" AND s.experiment_id = {}", ph(2))
        } else {
            String::new()
        };
        let sql = format!(
            "SELECT s.experiment_id, s.scorer_name, s.scorer_id, v.scorer_version, v.serialized_scorer, v.creation_time \
             FROM {SCORERS} s JOIN experiments e ON e.experiment_id = s.experiment_id \
             JOIN {SCORER_VERSIONS} v ON v.scorer_id = s.scorer_id \
             JOIN (SELECT scorer_id, MAX(scorer_version) AS max_version FROM {SCORER_VERSIONS} GROUP BY scorer_id) latest \
               ON latest.scorer_id = v.scorer_id AND latest.max_version = v.scorer_version \
             WHERE e.workspace = {} AND e.lifecycle_stage = 'active'{experiment_filter} \
             ORDER BY s.experiment_id, s.scorer_name",
            ph(1)
        );
        let mut scorers = self
            .db()
            .fetch_all(&sql, &values, map_scorer)
            .await
            .map_err(internal)?;
        self.resolve_scorer_list(workspace, &mut scorers).await?;
        Ok(scorers)
    }

    pub async fn list_scorer_versions(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
    ) -> Result<Vec<ScorerVersion>, MlflowError> {
        let experiment_id_num = self
            .require_active_scorer_experiment(workspace, experiment_id)
            .await?;
        let mut scorers = self
            .fetch_named_scorers(experiment_id_num, name, None, true)
            .await?;
        if scorers.is_empty() {
            return Err(scorer_not_found(experiment_id, name, None));
        }
        self.resolve_scorer_list(workspace, &mut scorers).await?;
        Ok(scorers)
    }

    pub async fn get_scorer(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
        version: Option<i32>,
    ) -> Result<ScorerVersion, MlflowError> {
        let experiment_id_num = self
            .require_active_scorer_experiment(workspace, experiment_id)
            .await?;
        let mut scorers = self
            .fetch_named_scorers(experiment_id_num, name, version, false)
            .await?;
        let mut scorer = scorers
            .pop()
            .ok_or_else(|| scorer_not_found(experiment_id, name, version))?;
        self.resolve_scorer(workspace, &mut scorer).await?;
        Ok(scorer)
    }

    async fn fetch_named_scorers(
        &self,
        experiment_id: i64,
        name: &str,
        version: Option<i32>,
        all_versions: bool,
    ) -> Result<Vec<ScorerVersion>, MlflowError> {
        let dialect = self.db().dialect();
        let ph = |index| dialect.placeholder(index);
        let (version_filter, order, values) = if let Some(version) = version {
            (
                format!(" AND v.scorer_version = {}", ph(3)),
                "v.scorer_version ASC",
                vec![
                    Val::Int(experiment_id),
                    Val::Text(name.to_string()),
                    Val::Int(i64::from(version)),
                ],
            )
        } else {
            (
                String::new(),
                if all_versions {
                    "v.scorer_version ASC"
                } else {
                    "v.scorer_version DESC"
                },
                vec![Val::Int(experiment_id), Val::Text(name.to_string())],
            )
        };
        let limit = if all_versions { "" } else { " LIMIT 1" };
        let sql = format!(
            "SELECT s.experiment_id, s.scorer_name, s.scorer_id, v.scorer_version, v.serialized_scorer, v.creation_time \
             FROM {SCORERS} s JOIN {SCORER_VERSIONS} v ON v.scorer_id = s.scorer_id \
             WHERE s.experiment_id = {} AND s.scorer_name = {}{version_filter} \
             ORDER BY {order}{limit}",
            ph(1),
            ph(2)
        );
        self.db()
            .fetch_all(&sql, &values, map_scorer)
            .await
            .map_err(internal)
    }

    pub async fn delete_scorer(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
        version: Option<i32>,
    ) -> Result<(), MlflowError> {
        let experiment_id_num = self
            .require_active_scorer_experiment(workspace, experiment_id)
            .await?;
        let dialect = self.db().dialect();
        let ph = |index| dialect.placeholder(index);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let scorer_id = tx
            .fetch_all(
                &format!(
                    "SELECT scorer_id FROM {SCORERS} WHERE experiment_id = {} AND scorer_name = {}",
                    ph(1),
                    ph(2)
                ),
                &[Val::Int(experiment_id_num), Val::Text(name.to_string())],
                |row| row.get_string("scorer_id"),
            )
            .await
            .map_err(internal)?
            .into_iter()
            .next()
            .ok_or_else(|| scorer_not_found(experiment_id, name, None))?;

        if let Some(version) = version {
            let affected = tx
                .exec(
                    &format!(
                        "DELETE FROM {SCORER_VERSIONS} WHERE scorer_id = {} AND scorer_version = {}",
                        ph(1), ph(2)
                    ),
                    &[Val::Text(scorer_id), Val::Int(i64::from(version))],
                )
                .await
                .map_err(internal)?;
            if affected == 0 {
                return Err(scorer_not_found(experiment_id, name, Some(version)));
            }
        } else {
            tx.exec(
                &format!(
                    "DELETE FROM {ENDPOINT_BINDINGS} WHERE resource_type = {} AND resource_id = {}",
                    ph(1),
                    ph(2)
                ),
                &[
                    Val::Text("scorer".to_string()),
                    Val::Text(scorer_id.clone()),
                ],
            )
            .await
            .map_err(internal)?;
            tx.exec(
                &format!("DELETE FROM {SCORERS} WHERE scorer_id = {}", ph(1)),
                &[Val::Text(scorer_id)],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)
    }

    pub async fn get_online_scoring_configs(
        &self,
        workspace: &str,
        scorer_ids: &[String],
    ) -> Result<Vec<OnlineScoringConfig>, MlflowError> {
        if scorer_ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(workspace.to_string())];
        values.extend(scorer_ids.iter().cloned().map(Val::Text));
        let placeholders = (2..=scorer_ids.len() + 1)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>()
            .join(", ");
        self.db()
            .fetch_all(
                &format!(
                    "SELECT c.online_scoring_config_id, c.scorer_id, c.sample_rate, c.experiment_id, c.filter_string \
                     FROM {ONLINE_SCORING_CONFIGS} c JOIN experiments e ON e.experiment_id = c.experiment_id \
                     WHERE e.workspace = {} AND c.scorer_id IN ({placeholders})",
                    dialect.placeholder(1)
                ),
                &values,
                map_online_config,
            )
            .await
            .map_err(internal)
    }

    pub async fn upsert_online_scoring_config(
        &self,
        workspace: &str,
        experiment_id: &str,
        scorer_name: &str,
        sample_rate: f64,
        filter_string: Option<&str>,
    ) -> Result<OnlineScoringConfig, MlflowError> {
        if !(0.0..=1.0).contains(&sample_rate) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "sample_rate must be between 0.0 and 1.0, got {sample_rate}"
            )));
        }
        if let Some(filter_string) = filter_string.filter(|value| !value.is_empty()) {
            mlflow_search::parse::traces_filter(filter_string)
                .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
        }
        let experiment_id_num = self
            .require_active_scorer_experiment(workspace, experiment_id)
            .await?;
        let dialect = self.db().dialect();
        let ph = |index| dialect.placeholder(index);
        let scorer = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT s.scorer_id, v.serialized_scorer FROM {SCORERS} s \
                     LEFT JOIN {SCORER_VERSIONS} v ON v.scorer_id = s.scorer_id AND v.scorer_version = \
                       (SELECT MAX(v2.scorer_version) FROM {SCORER_VERSIONS} v2 WHERE v2.scorer_id = s.scorer_id) \
                     WHERE s.experiment_id = {} AND s.scorer_name = {}",
                    ph(1), ph(2)
                ),
                &[
                    Val::Int(experiment_id_num),
                    Val::Text(scorer_name.to_string()),
                ],
                |row| {
                    Ok((
                        row.get_string("scorer_id")?,
                        row.get_opt_string("serialized_scorer")?,
                    ))
                },
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| scorer_not_found(experiment_id, scorer_name, None))?;

        if sample_rate > 0.0 {
            if let Some(serialized) = scorer.1.as_deref() {
                let data: Value = serde_json::from_str(serialized).map_err(|error| {
                    MlflowError::internal_error(format!("stored scorer is not valid JSON: {error}"))
                })?;
                let model = extract_model(&data)?;
                if model.as_deref().and_then(gateway_endpoint_ref).is_none() {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "Scorer '{scorer_name}' does not use a gateway model. Automatic evaluation is only supported for scorers that use gateway models."
                    )));
                }
                if instructions_require_expectations(&data) {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "Scorer '{scorer_name}' requires expectations, but scorers with expectations are not currently supported for automatic evaluation."
                    )));
                }
            }
        }

        let config = OnlineScoringConfig {
            online_scoring_config_id: Uuid::new_v4().simple().to_string(),
            scorer_id: scorer.0,
            sample_rate,
            experiment_id: experiment_id_num.to_string(),
            filter_string: filter_string.map(str::to_string),
        };
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        tx.exec(
            &format!(
                "DELETE FROM {ONLINE_SCORING_CONFIGS} WHERE scorer_id = {}",
                ph(1)
            ),
            &[Val::Text(config.scorer_id.clone())],
        )
        .await
        .map_err(internal)?;
        tx.exec(
            &format!(
                "INSERT INTO {ONLINE_SCORING_CONFIGS} (online_scoring_config_id, scorer_id, sample_rate, experiment_id, filter_string) VALUES ({}, {}, {}, {}, {})",
                ph(1), ph(2), ph(3), ph(4), ph(5)
            ),
            &[
                Val::Text(config.online_scoring_config_id.clone()),
                Val::Text(config.scorer_id.clone()),
                Val::Float(config.sample_rate),
                Val::Int(experiment_id_num),
                Val::OptText(config.filter_string.clone()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)?;
        Ok(config)
    }

    async fn require_active_scorer_experiment(
        &self,
        workspace: &str,
        experiment_id: &str,
    ) -> Result<i64, MlflowError> {
        let experiment_id_num = parse_experiment_id(experiment_id)?;
        let experiment = self.get_experiment(workspace, experiment_id).await?;
        if experiment.lifecycle_stage != LifecycleStage::ACTIVE {
            return Err(MlflowError::invalid_parameter_value(format!(
                "The experiment {} must be in the 'active' state. Current state is {}.",
                experiment.experiment_id, experiment.lifecycle_stage
            )));
        }
        Ok(experiment_id_num)
    }

    async fn endpoint_by_name(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<Option<(String, String)>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT endpoint_id, name FROM {ENDPOINTS} WHERE workspace = {} AND name = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                ],
                |row| Ok((row.get_string("endpoint_id")?, row.get_string("name")?)),
            )
            .await
            .map_err(internal)
    }

    async fn endpoint_name_by_id(
        &self,
        workspace: &str,
        endpoint_id: &str,
    ) -> Result<Option<String>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT name FROM {ENDPOINTS} WHERE workspace = {} AND endpoint_id = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(endpoint_id.to_string()),
                ],
                |row| row.get_string("name"),
            )
            .await
            .map_err(internal)
    }

    async fn resolve_scorer(
        &self,
        workspace: &str,
        scorer: &mut ScorerVersion,
    ) -> Result<(), MlflowError> {
        let mut data: Value = serde_json::from_str(&scorer.serialized_scorer).map_err(|error| {
            MlflowError::internal_error(format!("stored scorer is not valid JSON: {error}"))
        })?;
        let endpoint_id = extract_model(&data)?
            .and_then(|model| gateway_endpoint_ref(&model).map(str::to_string));
        let Some(endpoint_id) = endpoint_id else {
            return Ok(());
        };
        let endpoint_name = self.endpoint_name_by_id(workspace, &endpoint_id).await?;
        let model = endpoint_name.map(|name| format!("gateway:/{name}"));
        update_model(&mut data, model.as_deref());
        scorer.serialized_scorer = python_json_dumps(&data, false);
        Ok(())
    }

    async fn resolve_scorer_list(
        &self,
        workspace: &str,
        scorers: &mut [ScorerVersion],
    ) -> Result<(), MlflowError> {
        for scorer in scorers {
            let mut data: Value =
                serde_json::from_str(&scorer.serialized_scorer).map_err(|error| {
                    MlflowError::internal_error(format!("stored scorer is not valid JSON: {error}"))
                })?;
            if let Some(endpoint_id) = extract_model(&data)?
                .and_then(|model| gateway_endpoint_ref(&model).map(str::to_string))
            {
                let endpoint_name = self.endpoint_name_by_id(workspace, &endpoint_id).await?;
                let model = endpoint_name.map(|name| format!("gateway:/{name}"));
                update_model(&mut data, model.as_deref());
            }
            scorer.serialized_scorer = python_json_dumps(&data, false);
        }
        Ok(())
    }
}

fn map_scorer(row: &dyn RowLike) -> Result<ScorerVersion, sqlx::Error> {
    Ok(ScorerVersion {
        experiment_id: row.get_int("experiment_id")?.to_string(),
        scorer_name: row.get_string("scorer_name")?,
        scorer_version: row.get_int("scorer_version")? as i32,
        serialized_scorer: row.get_string("serialized_scorer")?,
        creation_time: row.get_opt_i64("creation_time")?,
        scorer_id: row.get_string("scorer_id")?,
    })
}

fn map_online_config(row: &dyn RowLike) -> Result<OnlineScoringConfig, sqlx::Error> {
    Ok(OnlineScoringConfig {
        online_scoring_config_id: row.get_string("online_scoring_config_id")?,
        scorer_id: row.get_string("scorer_id")?,
        sample_rate: row.get_f64("sample_rate")?,
        experiment_id: row.get_int("experiment_id")?.to_string(),
        filter_string: row.get_opt_string("filter_string")?,
    })
}

fn validate_scorer_name(name: &str) -> Result<(), MlflowError> {
    if name.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Scorer name cannot be empty or contain only whitespace.",
        ));
    }
    Ok(())
}

fn validate_scorer_model(model: Option<&str>) -> Result<(), MlflowError> {
    if model.is_some_and(|model| model.trim().is_empty()) {
        return Err(MlflowError::invalid_parameter_value(
            "Scorer model cannot be empty or contain only whitespace.",
        ));
    }
    Ok(())
}

fn extract_model(data: &Value) -> Result<Option<String>, MlflowError> {
    let Some(object) = data.as_object() else {
        return Ok(None);
    };
    for key in [
        "instructions_judge_pydantic_data",
        "builtin_scorer_pydantic_data",
    ] {
        if let Some(value) = object.get(key).filter(|value| truthy(value)) {
            return model_value(value);
        }
    }
    if let Some(memory) = object
        .get("memory_augmented_judge_data")
        .filter(|value| truthy(value))
    {
        return extract_model(memory.get("base_judge").unwrap_or(&Value::Null));
    }
    if let Some(value) = object
        .get("third_party_scorer_data")
        .filter(|value| truthy(value))
    {
        return model_value(value);
    }
    Ok(None)
}

fn model_value(container: &Value) -> Result<Option<String>, MlflowError> {
    match container.get("model") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(model)) => Ok(Some(model.clone())),
        Some(value) => Err(MlflowError::invalid_parameter_value(format!(
            "Scorer model must be a string, got {}.",
            python_type_name(value)
        ))),
    }
}

fn update_model(data: &mut Value, new_model: Option<&str>) {
    let Some(object) = data.as_object_mut() else {
        return;
    };
    for key in [
        "instructions_judge_pydantic_data",
        "builtin_scorer_pydantic_data",
    ] {
        if object.get(key).is_some_and(truthy) {
            if let Some(inner) = object.get_mut(key).and_then(Value::as_object_mut) {
                inner.insert(
                    "model".to_string(),
                    new_model.map_or(Value::Null, |model| Value::String(model.to_string())),
                );
            }
            return;
        }
    }
    if object
        .get("memory_augmented_judge_data")
        .is_some_and(truthy)
    {
        if let Some(base) = object
            .get_mut("memory_augmented_judge_data")
            .and_then(Value::as_object_mut)
            .and_then(|memory| memory.get_mut("base_judge"))
        {
            update_model(base, new_model);
        }
        return;
    }
    if let Some(inner) = object
        .get_mut("third_party_scorer_data")
        .filter(|value| truthy(value))
        .and_then(Value::as_object_mut)
    {
        if inner.get("model").is_some_and(|value| !value.is_null()) {
            inner.insert(
                "model".to_string(),
                new_model.map_or(Value::Null, |model| Value::String(model.to_string())),
            );
        }
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn gateway_endpoint_ref(model: &str) -> Option<&str> {
    model.strip_prefix("gateway:/")
}

fn instructions_require_expectations(data: &Value) -> bool {
    data.get("instructions_judge_pydantic_data")
        .and_then(|value| value.get("instructions"))
        .and_then(Value::as_str)
        .is_some_and(|instructions| instructions.contains("{{ expectations }}"))
}

fn scorer_not_found(experiment_id: &str, name: &str, version: Option<i32>) -> MlflowError {
    let message = match version {
        Some(version) => format!(
            "Scorer with name '{name}' and version {version} not found for experiment {experiment_id}."
        ),
        None => format!("Scorer with name '{name}' not found for experiment {experiment_id}."),
    };
    MlflowError::resource_does_not_exist(message)
}

fn retryable_registration(error: &sqlx::Error) -> bool {
    if is_unique_violation(error) {
        return true;
    }
    let message = error.to_string().to_ascii_lowercase();
    message.contains("database is locked")
        || message.contains("deadlock")
        || message.contains("serialization failure")
}

fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_i64() || number.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}
