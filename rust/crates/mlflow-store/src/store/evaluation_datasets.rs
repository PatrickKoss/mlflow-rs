//! Evaluation dataset persistence (`SqlEvaluationDataset` and its tag/record
//! children), including Python-compatible offset and cursor pagination.

use base64::Engine;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_search::{Comparison, OrderBy, Value as SearchValue};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::dbutil::{RowLike, Tx, Val};
use super::experiments::{internal, now_millis};
use super::TrackingStore;
use crate::dialect::{Dialect, UpsertSpec};

const DATASET_ID_PREFIX: &str = "d-";
const RECORD_ID_PREFIX: &str = "dr-";
const ASSOCIATION_ID_PREFIX: &str = "a-";
const EVALUATION_DATASET: &str = "evaluation_dataset";
const EXPERIMENT: &str = "experiment";
const MLFLOW_USER: &str = "mlflow.user";
const WRAPPED_OUTPUT_KEY: &str = "mlflow_wrapped";
pub const SEARCH_EVALUATION_DATASETS_MAX_RESULTS: i32 = 1000;

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationDataset {
    pub dataset_id: String,
    pub name: String,
    pub tags: Map<String, Value>,
    pub schema: Option<String>,
    pub profile: Option<String>,
    pub digest: Option<String>,
    pub created_time: Option<i64>,
    pub last_update_time: Option<i64>,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    /// Python leaves this lazy (`None`) except on create. HTTP conversion only
    /// emits the repeated proto field when this is `Some`.
    pub experiment_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationRecord {
    pub dataset_record_id: String,
    pub dataset_id: String,
    pub inputs: Value,
    pub outputs: Option<Value>,
    pub expectations: Option<Value>,
    pub tags: Option<Value>,
    pub source: Option<Value>,
    pub source_id: Option<String>,
    pub source_type: Option<String>,
    pub created_time: Option<i64>,
    pub last_update_time: Option<i64>,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationDatasetsPage {
    pub datasets: Vec<EvaluationDataset>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationRecordsPage {
    pub records: Vec<EvaluationRecord>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpsertEvaluationRecordsResult {
    pub inserted: i32,
    pub updated: i32,
}

fn new_prefixed_id(prefix: &str) -> String {
    format!("{prefix}{}", Uuid::new_v4().simple())
}

fn dataset_digest(name: &str, last_update_time: i64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{name}:{last_update_time}"));
    format!("{:x}", hasher.finalize())[..8].to_string()
}

fn input_hash(inputs: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(python_json_dumps(inputs, true));
    format!("{:x}", hasher.finalize())
}

/// `json.dumps` for the JSON subset used by datasets. The default separators,
/// ASCII escaping, insertion ordering, and optional recursive key sorting are
/// observable both in record payload strings and Python-written input hashes.
pub fn python_json_dumps(value: &Value, sort_keys: bool) -> String {
    fn write_string(out: &mut String, value: &str) {
        out.push('"');
        for ch in value.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                ch if ch <= '\u{1f}' || ch as u32 > 0x7f => {
                    let code = ch as u32;
                    if code <= 0xffff {
                        out.push_str(&format!("\\u{code:04x}"));
                    } else {
                        let n = code - 0x10000;
                        let hi = 0xd800 + (n >> 10);
                        let lo = 0xdc00 + (n & 0x3ff);
                        out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
                    }
                }
                ch => out.push(ch),
            }
        }
        out.push('"');
    }

    fn write_value(out: &mut String, value: &Value, sort_keys: bool) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => out.push_str(&value.to_string()),
            Value::String(value) => write_string(out, value),
            Value::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    write_value(out, value, sort_keys);
                }
                out.push(']');
            }
            Value::Object(values) => {
                out.push('{');
                let mut entries: Vec<_> = values.iter().collect();
                if sort_keys {
                    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                }
                for (index, (key, value)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    write_string(out, key);
                    out.push_str(": ");
                    write_value(out, value, sort_keys);
                }
                out.push('}');
            }
        }
    }

    let mut output = String::new();
    write_value(&mut output, value, sort_keys);
    output
}

fn json_bind(dialect: Dialect, placeholder: String) -> String {
    if dialect == Dialect::Postgres {
        format!("CAST({placeholder} AS json)")
    } else {
        placeholder
    }
}

fn json_select(dialect: Dialect, column: &str) -> String {
    match dialect {
        Dialect::Postgres => format!("CAST({column} AS TEXT) AS {column}"),
        Dialect::MySql => format!("CAST({column} AS CHAR) AS {column}"),
        Dialect::Sqlite => column.to_string(),
    }
}

fn integer_as_text(dialect: Dialect, column: &str) -> String {
    match dialect {
        Dialect::Postgres => format!("CAST({column} AS TEXT)"),
        Dialect::Sqlite | Dialect::MySql => format!("CAST({column} AS CHAR)"),
    }
}

fn search_error(error: mlflow_search::SearchError) -> MlflowError {
    MlflowError::invalid_parameter_value(error.message)
}

impl TrackingStore {
    pub async fn create_evaluation_dataset(
        &self,
        workspace: &str,
        name: &str,
        tags: &Map<String, Value>,
        experiment_ids: &[String],
    ) -> Result<EvaluationDataset, MlflowError> {
        let dataset_id = new_prefixed_id(DATASET_ID_PREFIX);
        let now = now_millis();
        let digest = dataset_digest(name, now);
        let created_by = tags
            .get(MLFLOW_USER)
            .and_then(Value::as_str)
            .map(str::to_string);
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let ph = |index| dialect.placeholder(index);
        tx.exec(
            &format!(
                "INSERT INTO evaluation_datasets (dataset_id, workspace, name, schema, profile, \
                 digest, created_time, last_update_time, created_by, last_updated_by) VALUES \
                 ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                ph(1),
                ph(2),
                ph(3),
                ph(4),
                ph(5),
                ph(6),
                ph(7),
                ph(8),
                ph(9),
                ph(10)
            ),
            &[
                Val::Text(dataset_id.clone()),
                Val::Text(workspace.to_string()),
                Val::Text(name.to_string()),
                Val::OptText(None),
                Val::OptText(None),
                Val::Text(digest.clone()),
                Val::Int(now),
                Val::Int(now),
                Val::OptText(created_by.clone()),
                Val::OptText(created_by.clone()),
            ],
        )
        .await
        .map_err(internal)?;

        for (key, value) in tags {
            let Some(value) = tag_value(value) else {
                continue;
            };
            tx.exec(
                &format!(
                    "INSERT INTO evaluation_dataset_tags (dataset_id, key, value) VALUES ({}, {}, {})",
                    ph(1), ph(2), ph(3)
                ),
                &[
                    Val::Text(dataset_id.clone()),
                    Val::Text(key.clone()),
                    Val::Text(value),
                ],
            )
            .await
            .map_err(internal)?;
        }
        for experiment_id in experiment_ids {
            insert_association(&mut tx, dialect, &dataset_id, experiment_id, now).await?;
        }
        tx.commit().await.map_err(internal)?;

        Ok(EvaluationDataset {
            dataset_id,
            name: name.to_string(),
            tags: tags.clone(),
            schema: None,
            profile: None,
            digest: Some(digest),
            created_time: Some(now),
            last_update_time: Some(now),
            created_by: created_by.clone(),
            last_updated_by: created_by,
            experiment_ids: Some(experiment_ids.to_vec()),
        })
    }

    pub async fn get_evaluation_dataset(
        &self,
        workspace: &str,
        dataset_id: &str,
    ) -> Result<EvaluationDataset, MlflowError> {
        let dialect = self.db().dialect();
        let row = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT dataset_id, name, schema, profile, digest, created_time, \
                     last_update_time, created_by, last_updated_by FROM evaluation_datasets \
                     WHERE dataset_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_dataset,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::new(
                    format!("Evaluation dataset with id '{dataset_id}' not found"),
                    ErrorCode::ResourceDoesNotExist,
                )
            })?;
        self.load_dataset_tags(row).await
    }

    pub async fn delete_evaluation_dataset(
        &self,
        workspace: &str,
        dataset_id: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let ph = |index| dialect.placeholder(index);
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let exists = tx
            .fetch_all(
                &format!(
                    "SELECT dataset_id FROM evaluation_datasets WHERE dataset_id = {} AND workspace = {}",
                    ph(1), ph(2)
                ),
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("dataset_id"),
            )
            .await
            .map_err(internal)?;
        if exists.is_empty() {
            tx.commit().await.map_err(internal)?;
            return Ok(());
        }
        tx.exec(
            &format!(
                "DELETE FROM entity_associations WHERE \
                 (destination_type = {} AND destination_id = {}) OR \
                 (source_type = {} AND source_id = {})",
                ph(1),
                ph(2),
                ph(3),
                ph(4)
            ),
            &[
                Val::Text(EVALUATION_DATASET.to_string()),
                Val::Text(dataset_id.to_string()),
                Val::Text(EVALUATION_DATASET.to_string()),
                Val::Text(dataset_id.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.exec(
            &format!(
                "DELETE FROM evaluation_datasets WHERE dataset_id = {} AND workspace = {}",
                ph(1),
                ph(2)
            ),
            &[
                Val::Text(dataset_id.to_string()),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)
    }

    pub async fn search_evaluation_datasets(
        &self,
        workspace: &str,
        experiment_ids: &[String],
        filter_string: Option<&str>,
        max_results: i32,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<EvaluationDatasetsPage, MlflowError> {
        if max_results <= 0 || max_results > SEARCH_EVALUATION_DATASETS_MAX_RESULTS {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value for request parameter max_results. It must be at most {SEARCH_EVALUATION_DATASETS_MAX_RESULTS}, but got value {max_results}"
            )));
        }
        let offset =
            mlflow_search::parse_start_offset_from_page_token(page_token).map_err(search_error)?;
        let filters = filter_string
            .map(mlflow_search::parse::evaluation_datasets_filter)
            .transpose()
            .map_err(search_error)?
            .unwrap_or_default();
        let orders: Vec<OrderBy> = if order_by.is_empty() {
            vec![OrderBy {
                entity_type: "attribute".to_string(),
                key: "created_time".to_string(),
                ascending: false,
            }]
        } else {
            order_by
                .iter()
                .map(|value| mlflow_search::parse::evaluation_datasets_order_by(value))
                .collect::<Result<_, _>>()
                .map_err(search_error)?
        };
        if let Some(order) = orders.iter().find(|order| order.entity_type != "attribute") {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid order_by entity: {}",
                order.entity_type
            )));
        }

        let dialect = self.db().dialect();
        let mut vals = vec![Val::Text(workspace.to_string())];
        let mut sql = format!(
            "SELECT ed.dataset_id, ed.name, ed.schema, ed.profile, ed.digest, ed.created_time, \
             ed.last_update_time, ed.created_by, ed.last_updated_by FROM evaluation_datasets ed \
             WHERE ed.workspace = {}",
            dialect.placeholder(1)
        );
        if !experiment_ids.is_empty() {
            vals.push(Val::Text(workspace.to_string()));
            let experiment_workspace = dialect.placeholder(vals.len());
            let placeholders: Vec<_> = experiment_ids
                .iter()
                .map(|id| {
                    vals.push(Val::Text(id.clone()));
                    dialect.placeholder(vals.len())
                })
                .collect();
            sql.push_str(&format!(
                " AND EXISTS (SELECT 1 FROM entity_associations ea JOIN experiments e ON \
                 {} = ea.destination_id WHERE \
                 ea.source_type = 'evaluation_dataset' AND ea.source_id = ed.dataset_id AND \
                 ea.destination_type = 'experiment' AND e.workspace = {experiment_workspace} AND \
                 ea.destination_id IN ({}))",
                integer_as_text(dialect, "e.experiment_id"),
                placeholders.join(", ")
            ));
        }
        for filter in &filters {
            append_filter(&mut sql, &mut vals, dialect, filter)?;
        }
        sql.push_str(" ORDER BY ");
        let mut clauses: Vec<String> = orders
            .iter()
            .map(|order| {
                format!(
                    "ed.{} {}",
                    order.key,
                    if order.ascending { "ASC" } else { "DESC" }
                )
            })
            .collect();
        if !orders.iter().any(|order| order.key == "dataset_id") {
            clauses.push("ed.dataset_id DESC".to_string());
        }
        sql.push_str(&clauses.join(", "));
        sql.push_str(&format!(
            " LIMIT {} OFFSET {}",
            max_results as i64 + 1,
            offset
        ));

        let mut datasets = self
            .db()
            .fetch_all(&sql, &vals, map_dataset)
            .await
            .map_err(internal)?;
        let next_page_token = if datasets.len() > max_results as usize {
            datasets.truncate(max_results as usize);
            Some(mlflow_search::create_page_token(
                offset + max_results as i64,
            ))
        } else {
            None
        };
        let mut with_tags = Vec::with_capacity(datasets.len());
        for dataset in datasets {
            with_tags.push(self.load_dataset_tags(dataset).await?);
        }
        Ok(EvaluationDatasetsPage {
            datasets: with_tags,
            next_page_token,
        })
    }

    async fn load_dataset_tags(
        &self,
        mut dataset: EvaluationDataset,
    ) -> Result<EvaluationDataset, MlflowError> {
        let dialect = self.db().dialect();
        let tags = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT key, value FROM evaluation_dataset_tags WHERE dataset_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(dataset.dataset_id.clone())],
                |row| Ok((row.get_string("key")?, row.get_opt_string("value")?)),
            )
            .await
            .map_err(internal)?;
        dataset.tags = tags
            .into_iter()
            .map(|(key, value)| (key, value.map(Value::String).unwrap_or(Value::Null)))
            .collect();
        Ok(dataset)
    }

    async fn validate_evaluation_dataset(
        &self,
        workspace: &str,
        dataset_id: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let found = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT dataset_id FROM evaluation_datasets WHERE dataset_id = {} AND workspace = {}",
                    dialect.placeholder(1), dialect.placeholder(2)
                ),
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("dataset_id"),
            )
            .await
            .map_err(internal)?;
        if found.is_none() {
            return Err(MlflowError::new(
                format!("Dataset '{dataset_id}' not found."),
                ErrorCode::ResourceDoesNotExist,
            ));
        }
        Ok(())
    }

    pub async fn set_evaluation_dataset_tags(
        &self,
        workspace: &str,
        dataset_id: &str,
        tags: &Map<String, Value>,
    ) -> Result<(), MlflowError> {
        self.get_evaluation_dataset(workspace, dataset_id)
            .await
            .map_err(|_| {
                MlflowError::new(
                    format!("Could not find evaluation dataset with ID {dataset_id}"),
                    ErrorCode::ResourceDoesNotExist,
                )
            })?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for (key, value) in tags {
            let Some(value) = tag_value(value) else {
                continue;
            };
            let sql = dialect.upsert(&UpsertSpec {
                table: "evaluation_dataset_tags",
                columns: &["dataset_id", "key", "value"],
                pk_columns: &["dataset_id", "key"],
                update_columns: &["value"],
                json_columns: &[],
            });
            tx.exec(
                &sql,
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(key.clone()),
                    Val::Text(value),
                ],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)
    }

    pub async fn delete_evaluation_dataset_tag(
        &self,
        workspace: &str,
        dataset_id: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM evaluation_dataset_tags WHERE dataset_id = {} AND key = {} AND \
                     EXISTS (SELECT 1 FROM evaluation_datasets ed WHERE ed.dataset_id = \
                     evaluation_dataset_tags.dataset_id AND ed.workspace = {})",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(key.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    pub async fn upsert_evaluation_records(
        &self,
        workspace: &str,
        dataset_id: &str,
        records: &[Value],
    ) -> Result<UpsertEvaluationRecordsResult, MlflowError> {
        self.validate_evaluation_dataset(workspace, dataset_id)
            .await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let mut inserted = 0;
        let mut updated = 0;
        let now = now_millis();
        let mut updated_by = None;
        let mut existing_schema = self
            .get_evaluation_dataset(workspace, dataset_id)
            .await?
            .schema
            .and_then(|value| serde_json::from_str(&value).ok());

        for record in records {
            let object = record.as_object().ok_or_else(|| {
                MlflowError::invalid_parameter_value("Each dataset record must be a JSON object")
            })?;
            let inputs = object
                .get("inputs")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            let hash = input_hash(&inputs);
            let existing = find_record_by_hash(&mut tx, dialect, dataset_id, &hash).await?;
            let tags = object.get("tags").and_then(Value::as_object);
            if let Some(user) = tags
                .and_then(|tags| tags.get(MLFLOW_USER))
                .and_then(Value::as_str)
            {
                updated_by = Some(user.to_string());
            }
            if let Some(existing) = existing {
                merge_record(&mut tx, dialect, existing, object, now).await?;
                updated += 1;
            } else {
                insert_record(&mut tx, dialect, dataset_id, object, inputs, &hash, now).await?;
                inserted += 1;
            }
            update_schema(&mut existing_schema, object);
        }

        let profile_count = count_records(&mut tx, dialect, dataset_id).await?;
        let dataset_name = tx
            .fetch_all(
                &format!(
                    "SELECT name FROM evaluation_datasets WHERE dataset_id = {} AND workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("name"),
            )
            .await
            .map_err(internal)?
            .into_iter()
            .next()
            .unwrap_or_default();
        let schema = existing_schema.map(|value| python_json_dumps(&value, false));
        let profile = (profile_count > 0).then(|| format!("{{\"num_records\": {profile_count}}}"));
        let digest = dataset_digest(&dataset_name, now);
        tx.exec(
            &format!(
                "UPDATE evaluation_datasets SET schema = {}, profile = {}, digest = {}, \
                 last_update_time = {}, last_updated_by = {} WHERE dataset_id = {} AND workspace = {}",
                dialect.placeholder(1), dialect.placeholder(2), dialect.placeholder(3),
                dialect.placeholder(4), dialect.placeholder(5), dialect.placeholder(6),
                dialect.placeholder(7)
            ),
            &[
                Val::OptText(schema),
                Val::OptText(profile),
                Val::Text(digest),
                Val::Int(now),
                Val::OptText(updated_by),
                Val::Text(dataset_id.to_string()),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)?;
        Ok(UpsertEvaluationRecordsResult { inserted, updated })
    }

    pub async fn load_evaluation_records(
        &self,
        workspace: &str,
        dataset_id: &str,
        max_results: i32,
        page_token: Option<&str>,
    ) -> Result<EvaluationRecordsPage, MlflowError> {
        self.validate_evaluation_dataset(workspace, dataset_id)
            .await?;
        let dialect = self.db().dialect();
        let mut vals = vec![Val::Text(dataset_id.to_string())];
        let mut sql = format!(
            "SELECT dataset_record_id, dataset_id, {}, {}, {}, {}, {}, \
             source_id, source_type, created_time, last_update_time, created_by, last_updated_by \
             FROM evaluation_dataset_records WHERE dataset_id = {}",
            json_select(dialect, "inputs"),
            json_select(dialect, "outputs"),
            json_select(dialect, "expectations"),
            json_select(dialect, "tags"),
            json_select(dialect, "source"),
            dialect.placeholder(1)
        );
        let mut legacy_offset = None;
        if let Some(token) = page_token.filter(|value| !value.is_empty()) {
            match decode_cursor(token) {
                Ok((created_time, record_id)) => {
                    vals.push(Val::Int(created_time));
                    let greater_time = dialect.placeholder(vals.len());
                    let equal_time = if dialect == Dialect::Postgres {
                        greater_time.clone()
                    } else {
                        vals.push(Val::Int(created_time));
                        dialect.placeholder(vals.len())
                    };
                    vals.push(Val::Text(record_id));
                    let record = dialect.placeholder(vals.len());
                    sql.push_str(&format!(
                        " AND (created_time > {greater_time} OR (created_time = {equal_time} AND dataset_record_id > {record}))"
                    ));
                }
                Err(_) => {
                    legacy_offset = Some(token.parse::<i64>().map_err(|_| {
                        MlflowError::invalid_parameter_value(format!(
                            "invalid literal for int() with base 10: '{token}'"
                        ))
                    })?);
                }
            }
        }
        sql.push_str(" ORDER BY created_time, dataset_record_id");
        sql.push_str(&format!(" LIMIT {}", max_results as i64 + 1));
        if let Some(offset) = legacy_offset {
            sql.push_str(&format!(" OFFSET {offset}"));
        }
        let mut records = self
            .db()
            .fetch_all(&sql, &vals, map_record)
            .await
            .map_err(internal)?;
        let next_page_token = if records.len() > max_results as usize {
            records.truncate(max_results as usize);
            records.last().map(|record| {
                encode_cursor(
                    record.created_time.unwrap_or_default(),
                    &record.dataset_record_id,
                )
            })
        } else {
            None
        };
        Ok(EvaluationRecordsPage {
            records,
            next_page_token,
        })
    }

    pub async fn delete_evaluation_records(
        &self,
        workspace: &str,
        dataset_id: &str,
        record_ids: &[String],
    ) -> Result<i32, MlflowError> {
        self.validate_evaluation_dataset(workspace, dataset_id)
            .await?;
        if record_ids.is_empty() {
            return Ok(0);
        }
        let dialect = self.db().dialect();
        let mut vals = vec![Val::Text(dataset_id.to_string())];
        let placeholders: Vec<_> = record_ids
            .iter()
            .map(|id| {
                vals.push(Val::Text(id.clone()));
                dialect.placeholder(vals.len())
            })
            .collect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let deleted = tx
            .exec(
                &format!(
                    "DELETE FROM evaluation_dataset_records WHERE dataset_id = {} AND dataset_record_id IN ({})",
                    dialect.placeholder(1), placeholders.join(", ")
                ),
                &vals,
            )
            .await
            .map_err(internal)? as i32;
        if deleted > 0 {
            let remaining = count_records(&mut tx, dialect, dataset_id).await?;
            let profile = if remaining > 0 {
                Some(format!("{{\"num_records\": {remaining}}}"))
            } else {
                Some("null".to_string())
            };
            tx.exec(
                &format!(
                    "UPDATE evaluation_datasets SET profile = {} WHERE dataset_id = {} AND workspace = {}",
                    dialect.placeholder(1), dialect.placeholder(2), dialect.placeholder(3)
                ),
                &[
                    Val::OptText(profile),
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)?;
        Ok(deleted)
    }

    pub async fn get_evaluation_dataset_experiment_ids(
        &self,
        workspace: &str,
        dataset_id: &str,
    ) -> Result<Vec<String>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_all(
                &format!(
                    "SELECT ea.destination_id FROM entity_associations ea \
                     JOIN evaluation_datasets ed ON ed.dataset_id = ea.source_id \
                     JOIN experiments e ON {} = ea.destination_id \
                     WHERE ea.source_type = 'evaluation_dataset' AND \
                     ea.destination_type = 'experiment' AND ea.source_id = {} AND \
                     ed.workspace = {} AND e.workspace = {}",
                    integer_as_text(dialect, "e.experiment_id"),
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("destination_id"),
            )
            .await
            .map_err(internal)
    }

    pub async fn add_evaluation_dataset_to_experiments(
        &self,
        workspace: &str,
        dataset_id: &str,
        experiment_ids: &[String],
    ) -> Result<EvaluationDataset, MlflowError> {
        let mut dataset = self.dataset_for_association(workspace, dataset_id).await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for experiment_id in experiment_ids {
            validate_experiment(&mut tx, dialect, workspace, experiment_id).await?;
            insert_association(&mut tx, dialect, dataset_id, experiment_id, now_millis()).await?;
        }
        let now = now_millis();
        tx.exec(
            &format!(
                "UPDATE evaluation_datasets SET last_update_time = {} WHERE dataset_id = {} AND workspace = {}",
                dialect.placeholder(1), dialect.placeholder(2), dialect.placeholder(3)
            ),
            &[
                Val::Int(now),
                Val::Text(dataset_id.to_string()),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)?;
        dataset.last_update_time = Some(now);
        Ok(dataset)
    }

    pub async fn remove_evaluation_dataset_from_experiments(
        &self,
        workspace: &str,
        dataset_id: &str,
        experiment_ids: &[String],
    ) -> Result<EvaluationDataset, MlflowError> {
        let mut dataset = self.dataset_for_association(workspace, dataset_id).await?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let mut deleted = 0;
        for experiment_id in experiment_ids {
            deleted += tx
                .exec(
                    &format!(
                        "DELETE FROM entity_associations WHERE source_type = 'evaluation_dataset' AND \
                         source_id = {} AND destination_type = 'experiment' AND destination_id = {}",
                        dialect.placeholder(1), dialect.placeholder(2)
                    ),
                    &[
                        Val::Text(dataset_id.to_string()),
                        Val::Text(experiment_id.clone()),
                    ],
                )
                .await
                .map_err(internal)?;
        }
        if deleted > 0 {
            let now = now_millis();
            tx.exec(
                &format!(
                    "UPDATE evaluation_datasets SET last_update_time = {} WHERE dataset_id = {} AND workspace = {}",
                    dialect.placeholder(1), dialect.placeholder(2), dialect.placeholder(3)
                ),
                &[
                    Val::Int(now),
                    Val::Text(dataset_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
            dataset.last_update_time = Some(now);
        }
        tx.commit().await.map_err(internal)?;
        Ok(dataset)
    }

    async fn dataset_for_association(
        &self,
        workspace: &str,
        dataset_id: &str,
    ) -> Result<EvaluationDataset, MlflowError> {
        self.get_evaluation_dataset(workspace, dataset_id)
            .await
            .map_err(|_| {
                MlflowError::new(
                    format!("Dataset '{dataset_id}' not found"),
                    ErrorCode::ResourceDoesNotExist,
                )
            })
    }
}

fn tag_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => Some(value.clone()),
        Value::Bool(value) => Some(if *value { "True" } else { "False" }.to_string()),
        Value::Number(value) => Some(value.to_string()),
        other => Some(python_json_dumps(other, false)),
    }
}

fn map_dataset(row: &dyn RowLike) -> Result<EvaluationDataset, sqlx::Error> {
    Ok(EvaluationDataset {
        dataset_id: row.get_string("dataset_id")?,
        name: row.get_string("name")?,
        tags: Map::new(),
        schema: row.get_opt_string("schema")?,
        profile: row.get_opt_string("profile")?,
        digest: row.get_opt_string("digest")?,
        created_time: row.get_opt_i64("created_time")?,
        last_update_time: row.get_opt_i64("last_update_time")?,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        experiment_ids: None,
    })
}

fn parse_json_column(row: &dyn RowLike, column: &str) -> Result<Option<Value>, sqlx::Error> {
    row.get_opt_string(column)?.map_or(Ok(None), |value| {
        serde_json::from_str(&value)
            .map(Some)
            .map_err(|error| sqlx::Error::Decode(Box::new(error)))
    })
}

fn map_record(row: &dyn RowLike) -> Result<EvaluationRecord, sqlx::Error> {
    let outputs = parse_json_column(row, "outputs")?.and_then(|value| {
        value
            .as_object()
            .and_then(|value| value.get(WRAPPED_OUTPUT_KEY))
            .cloned()
    });
    Ok(EvaluationRecord {
        dataset_record_id: row.get_string("dataset_record_id")?,
        dataset_id: row.get_string("dataset_id")?,
        inputs: parse_json_column(row, "inputs")?.unwrap_or_else(|| Value::Object(Map::new())),
        outputs,
        expectations: parse_json_column(row, "expectations")?,
        tags: parse_json_column(row, "tags")?,
        source: parse_json_column(row, "source")?,
        source_id: row.get_opt_string("source_id")?,
        source_type: row.get_opt_string("source_type")?,
        created_time: row.get_opt_i64("created_time")?,
        last_update_time: row.get_opt_i64("last_update_time")?,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
    })
}

fn append_filter(
    sql: &mut String,
    vals: &mut Vec<Val>,
    dialect: Dialect,
    filter: &Comparison,
) -> Result<(), MlflowError> {
    // SearchEvaluationDatasetsUtils validates comparators case-sensitively at
    // SQL construction time. sqlparse preserves the user's spelling, so a
    // lowercase `like` must fail instead of being normalized to `LIKE`.
    let comparator = filter.comparator.as_str();
    let value = match &filter.value {
        SearchValue::Str(value) => value.clone(),
        _ => {
            return Err(MlflowError::invalid_parameter_value(
                "Invalid dataset filter value",
            ))
        }
    };
    vals.push(
        if matches!(filter.key.as_str(), "created_time" | "last_update_time") {
            Val::Int(value.parse().map_err(|_| {
                MlflowError::invalid_parameter_value(format!(
                    "Expected numeric value type for numeric attribute: {}. Found {value}",
                    filter.key
                ))
            })?)
        } else {
            Val::Text(value)
        },
    );
    let value_index = vals.len();
    if filter.entity_type == "attribute" {
        let valid = if matches!(filter.key.as_str(), "created_time" | "last_update_time") {
            ["=", "!=", "<", "<=", ">", ">="].as_slice()
        } else {
            ["=", "!=", "LIKE", "ILIKE"].as_slice()
        };
        if !valid.contains(&comparator) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator for {} attribute: {comparator}",
                if matches!(filter.key.as_str(), "created_time" | "last_update_time") {
                    "numeric"
                } else {
                    "string"
                }
            )));
        }
        let column = format!("ed.{}", filter.key);
        sql.push_str(" AND ");
        sql.push_str(&comparison_predicate(
            dialect,
            &column,
            comparator,
            value_index,
            vals,
            !matches!(filter.key.as_str(), "created_time" | "last_update_time"),
        ));
    } else if filter.entity_type == "tag" {
        if !["=", "!=", "LIKE", "ILIKE"].contains(&comparator) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid comparator for tag: {comparator}"
            )));
        }
        let predicate =
            comparison_predicate(dialect, "edt.value", comparator, value_index, vals, true);
        vals.push(Val::Text(filter.key.clone()));
        let key_ph = dialect.placeholder(vals.len());
        sql.push_str(&format!(
            " AND EXISTS (SELECT 1 FROM evaluation_dataset_tags edt WHERE edt.dataset_id = \
             ed.dataset_id AND {predicate} AND edt.key = {key_ph})",
        ));
    }
    Ok(())
}

fn comparison_predicate(
    dialect: Dialect,
    column: &str,
    comparator: &str,
    value_index: usize,
    vals: &mut Vec<Val>,
    case_sensitive_string: bool,
) -> String {
    match comparator {
        "ILIKE" => dialect.case_insensitive_like(column, value_index),
        "LIKE" => {
            if dialect == Dialect::MySql {
                vals.push(vals[value_index - 1].clone());
            }
            dialect.case_sensitive_like(column, value_index)
        }
        "=" | "!=" if dialect == Dialect::MySql && case_sensitive_string => {
            vals.push(vals[value_index - 1].clone());
            let left = dialect.placeholder(value_index);
            let binary = dialect.placeholder(value_index + 1);
            if comparator == "=" {
                format!("({column} = {left} AND BINARY {column} = {binary})")
            } else {
                format!("({column} != {left} OR BINARY {column} != {binary})")
            }
        }
        _ => format!("{column} {comparator} {}", dialect.placeholder(value_index)),
    }
}

async fn insert_association(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    dataset_id: &str,
    experiment_id: &str,
    created_time: i64,
) -> Result<(), MlflowError> {
    let sql = dialect.upsert(&UpsertSpec {
        table: "entity_associations",
        columns: &[
            "association_id",
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
            "created_time",
        ],
        pk_columns: &[
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
        ],
        update_columns: &[],
        json_columns: &[],
    });
    tx.exec(
        &sql,
        &[
            Val::Text(new_prefixed_id(ASSOCIATION_ID_PREFIX)),
            Val::Text(EVALUATION_DATASET.to_string()),
            Val::Text(dataset_id.to_string()),
            Val::Text(EXPERIMENT.to_string()),
            Val::Text(experiment_id.to_string()),
            Val::Int(created_time),
        ],
    )
    .await
    .map_err(internal)?;
    Ok(())
}

async fn validate_experiment(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    workspace: &str,
    experiment_id: &str,
) -> Result<(), MlflowError> {
    let experiment_id: i64 = experiment_id.parse().map_err(|_| {
        MlflowError::new(
            format!("No Experiment with id={experiment_id}"),
            ErrorCode::ResourceDoesNotExist,
        )
    })?;
    let rows = tx
        .fetch_all(
            &format!(
                "SELECT experiment_id FROM experiments WHERE experiment_id = {} AND workspace = {}",
                dialect.placeholder(1),
                dialect.placeholder(2)
            ),
            &[Val::Int(experiment_id), Val::Text(workspace.to_string())],
            |row| row.get_int("experiment_id"),
        )
        .await
        .map_err(internal)?;
    if rows.is_empty() {
        return Err(MlflowError::new(
            format!("No Experiment with id={experiment_id}"),
            ErrorCode::ResourceDoesNotExist,
        ));
    }
    Ok(())
}

async fn find_record_by_hash(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    dataset_id: &str,
    hash: &str,
) -> Result<Option<EvaluationRecord>, MlflowError> {
    Ok(tx
        .fetch_all(
            &format!(
                "SELECT dataset_record_id, dataset_id, {}, {}, {}, {}, {}, \
                 source_id, source_type, created_time, last_update_time, created_by, last_updated_by \
                 FROM evaluation_dataset_records WHERE dataset_id = {} AND input_hash = {}",
                json_select(dialect, "inputs"),
                json_select(dialect, "outputs"),
                json_select(dialect, "expectations"),
                json_select(dialect, "tags"),
                json_select(dialect, "source"),
                dialect.placeholder(1), dialect.placeholder(2)
            ),
            &[Val::Text(dataset_id.to_string()), Val::Text(hash.to_string())],
            map_record,
        )
        .await
        .map_err(internal)?
        .into_iter()
        .next())
}

async fn insert_record(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    dataset_id: &str,
    object: &Map<String, Value>,
    inputs: Value,
    hash: &str,
    now: i64,
) -> Result<(), MlflowError> {
    let tags = object.get("tags").cloned();
    let created_by = tags
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|tags| tags.get(MLFLOW_USER))
        .and_then(Value::as_str)
        .map(str::to_string);
    let source = normalize_source(object.get("source"))?;
    let source_type = source
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|source| source.get("source_type"))
        .and_then(Value::as_str)
        .map(|value| value.to_uppercase());
    let source_data = source
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|source| source.get("source_data"))
        .and_then(Value::as_object);
    let source_id = source_data
        .and_then(|data| {
            if source_type.as_deref() == Some("TRACE") {
                data.get("trace_id")
            } else {
                data.get("source_id")
            }
        })
        .and_then(Value::as_str)
        .map(str::to_string);
    let default_outputs = Value::Object(Map::new());
    let outputs = Some(object.get("outputs").unwrap_or(&default_outputs)).map(|value| {
        if value.is_null() {
            Value::Null
        } else {
            let mut wrapped = Map::new();
            wrapped.insert(WRAPPED_OUTPUT_KEY.to_string(), value.clone());
            Value::Object(wrapped)
        }
    });
    let fields = [
        Val::Text(new_prefixed_id(RECORD_ID_PREFIX)),
        Val::Text(dataset_id.to_string()),
        Val::Text(python_json_dumps(&inputs, false)),
        Val::OptText(
            outputs
                .filter(|value| !value.is_null())
                .map(|value| python_json_dumps(&value, false)),
        ),
        Val::OptText(
            object
                .get("expectations")
                .filter(|value| !value.is_null())
                .map(|value| python_json_dumps(value, false)),
        ),
        Val::OptText(
            tags.filter(|value| !value.is_null())
                .map(|value| python_json_dumps(&value, false)),
        ),
        Val::OptText(source.map(|value| python_json_dumps(&value, false))),
        Val::OptText(source_id),
        Val::OptText(source_type),
        Val::Int(now),
        Val::Int(now),
        Val::OptText(created_by.clone()),
        Val::OptText(created_by),
        Val::Text(hash.to_string()),
    ];
    let placeholders: Vec<_> = (1..=14)
        .map(|index| {
            let ph = dialect.placeholder(index);
            if [3, 4, 5, 6, 7].contains(&index) {
                json_bind(dialect, ph)
            } else {
                ph
            }
        })
        .collect();
    tx.exec(
        &format!(
            "INSERT INTO evaluation_dataset_records (dataset_record_id, dataset_id, inputs, \
             outputs, expectations, tags, source, source_id, source_type, created_time, \
             last_update_time, created_by, last_updated_by, input_hash) VALUES ({})",
            placeholders.join(", ")
        ),
        &fields,
    )
    .await
    .map_err(internal)?;
    Ok(())
}

fn normalize_source(source: Option<&Value>) -> Result<Option<Value>, MlflowError> {
    let Some(source) = source.filter(|value| match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Number(_) => true,
    }) else {
        return Ok(None);
    };
    let source = source.as_object().ok_or_else(|| {
        MlflowError::invalid_parameter_value("Dataset record source must be a JSON object")
    })?;
    let source_type = source
        .get("source_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value("Dataset record source_type is required")
        })?
        .to_uppercase();
    if !["UNSPECIFIED", "TRACE", "HUMAN", "DOCUMENT", "CODE"].contains(&source_type.as_str()) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid dataset record source type: {source_type}. Valid source types: ['UNSPECIFIED', 'TRACE', 'HUMAN', 'DOCUMENT', 'CODE']"
        )));
    }

    let mut normalized = Map::new();
    normalized.insert("source_type".to_string(), Value::String(source_type));
    normalized.insert(
        "source_data".to_string(),
        source
            .get("source_data")
            .filter(|value| !value.is_null())
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
    );
    Ok(Some(Value::Object(normalized)))
}

async fn merge_record(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    existing: EvaluationRecord,
    object: &Map<String, Value>,
    now: i64,
) -> Result<(), MlflowError> {
    let mut expectations = existing.expectations;
    if let Some(new_values) = object.get("expectations").and_then(Value::as_object) {
        if !new_values.is_empty() {
            let target = expectations
                .get_or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .expect("stored expectations are objects");
            target.extend(new_values.clone());
        }
    }
    let mut tags = existing.tags;
    if let Some(new_values) = object.get("tags").and_then(Value::as_object) {
        if !new_values.is_empty() {
            let target = tags
                .get_or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .expect("stored tags are objects");
            target.extend(new_values.clone());
        }
    }
    let last_updated_by = object
        .get("tags")
        .and_then(Value::as_object)
        .and_then(|tags| tags.get(MLFLOW_USER))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or(existing.last_updated_by);
    let outputs = object.get("outputs").map(|value| {
        if value.is_null() {
            None
        } else {
            let mut wrapped = Map::new();
            wrapped.insert(WRAPPED_OUTPUT_KEY.to_string(), value.clone());
            Some(Value::Object(wrapped))
        }
    });
    let mut sets = vec![
        format!(
            "expectations = {}",
            json_bind(dialect, dialect.placeholder(1))
        ),
        format!("tags = {}", json_bind(dialect, dialect.placeholder(2))),
        format!("last_update_time = {}", dialect.placeholder(3)),
        format!("last_updated_by = {}", dialect.placeholder(4)),
    ];
    let mut vals = vec![
        Val::OptText(expectations.map(|value| python_json_dumps(&value, false))),
        Val::OptText(tags.map(|value| python_json_dumps(&value, false))),
        Val::Int(now),
        Val::OptText(last_updated_by),
    ];
    if let Some(outputs) = outputs {
        vals.push(Val::OptText(
            outputs.map(|value| python_json_dumps(&value, false)),
        ));
        sets.push(format!(
            "outputs = {}",
            json_bind(dialect, dialect.placeholder(vals.len()))
        ));
    }
    vals.push(Val::Text(existing.dataset_record_id));
    let record_ph = dialect.placeholder(vals.len());
    tx.exec(
        &format!(
            "UPDATE evaluation_dataset_records SET {} WHERE dataset_record_id = {record_ph}",
            sets.join(", ")
        ),
        &vals,
    )
    .await
    .map_err(internal)?;
    Ok(())
}

fn update_schema(schema: &mut Option<Value>, record: &Map<String, Value>) {
    if schema.is_none() {
        let mut root = Map::new();
        root.insert("inputs".to_string(), Value::Object(Map::new()));
        root.insert("outputs".to_string(), Value::Object(Map::new()));
        root.insert("expectations".to_string(), Value::Object(Map::new()));
        root.insert("version".to_string(), Value::String("1.0".to_string()));
        *schema = Some(Value::Object(root));
    }
    let root = schema.as_mut().unwrap().as_object_mut().unwrap();
    for field in ["inputs", "expectations"] {
        let Some(values) = record.get(field).and_then(Value::as_object) else {
            continue;
        };
        let target = root.get_mut(field).unwrap().as_object_mut().unwrap();
        for (key, value) in values {
            target
                .entry(key.clone())
                .or_insert_with(|| Value::String(infer_type(value).to_string()));
        }
    }
    if let Some(outputs) = record.get("outputs").filter(|value| !value.is_null()) {
        if let Some(values) = outputs.as_object() {
            if let Some(target) = root.get_mut("outputs").and_then(Value::as_object_mut) {
                for (key, value) in values {
                    target
                        .entry(key.clone())
                        .or_insert_with(|| Value::String(infer_type(value).to_string()));
                }
            }
        } else if root
            .get("outputs")
            .and_then(Value::as_object)
            .is_some_and(Map::is_empty)
        {
            root.insert(
                "outputs".to_string(),
                Value::String(infer_type(outputs).to_string()),
            );
        }
    }
}

fn infer_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(value) if value.is_i64() || value.is_u64() => "integer",
        Value::Number(_) => "float",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

async fn count_records(
    tx: &mut Tx<'_>,
    dialect: Dialect,
    dataset_id: &str,
) -> Result<i64, MlflowError> {
    tx.fetch_all(
        &format!(
            "SELECT COUNT(*) AS count FROM evaluation_dataset_records WHERE dataset_id = {}",
            dialect.placeholder(1)
        ),
        &[Val::Text(dataset_id.to_string())],
        |row| row.get_i64("count"),
    )
    .await
    .map_err(internal)
    .map(|values| values[0])
}

fn encode_cursor(created_time: i64, record_id: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(format!("{created_time}:{record_id}"))
}

fn decode_cursor(token: &str) -> Result<(i64, String), ()> {
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(token)
        .map_err(|_| ())?;
    let decoded = String::from_utf8(decoded).map_err(|_| ())?;
    let (created_time, record_id) = decoded.split_once(':').ok_or(())?;
    Ok((created_time.parse().map_err(|_| ())?, record_id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_json_and_cursor_are_interoperable() {
        let value = serde_json::json!({"z": "café", "a": {"b": 1, "a": true}});
        assert_eq!(
            python_json_dumps(&value, true),
            r#"{"a": {"a": true, "b": 1}, "z": "caf\u00e9"}"#
        );
        let token = encode_cursor(123, "dr-abc");
        assert_eq!(token, "MTIzOmRyLWFiYw==");
        assert_eq!(decode_cursor(&token), Ok((123, "dr-abc".to_string())));
    }
}
