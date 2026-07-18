//! Workspace-scoped label-schema CRUD and validation.

use mlflow_error::MlflowError;
use serde_json::{Map, Number, Value};
use uuid::Uuid;

use super::dbutil::{RowLike, Val};
use super::evaluation_datasets::python_json_dumps;
use super::experiments::{internal, now_millis, parse_experiment_id};
use super::search::SEARCH_MAX_RESULTS_THRESHOLD;
use super::TrackingStore;

const SCHEMA_ID_PREFIX: &str = "ls-";
pub const DEFAULT_LABEL_SCHEMA_NAME: &str = "Feedback";
pub const DEFAULT_LABEL_SCHEMA_INSTRUCTION: &str = "Share any feedback on this trace.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSchemaType {
    Feedback,
    Expectation,
}

impl LabelSchemaType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Feedback => "feedback",
            Self::Expectation => "expectation",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LabelSchemaInput {
    PassFail {
        positive_label: String,
        negative_label: String,
    },
    Categorical {
        options: Vec<String>,
        multi_select: bool,
    },
    Numeric {
        min_value: Option<f64>,
        max_value: Option<f64>,
    },
    Text {
        max_length: Option<i64>,
    },
}

impl LabelSchemaInput {
    pub fn input_type(&self) -> &'static str {
        match self {
            Self::PassFail { .. } => "pass_fail",
            Self::Categorical { .. } => "categorical",
            Self::Numeric { .. } => "numeric",
            Self::Text { .. } => "text",
        }
    }

    fn python_class_name(&self) -> &'static str {
        match self {
            Self::PassFail { .. } => "InputPassFail",
            Self::Categorical { .. } => "InputCategorical",
            Self::Numeric { .. } => "InputNumeric",
            Self::Text { .. } => "InputText",
        }
    }

    fn config_json(&self) -> String {
        let mut config = Map::new();
        match self {
            Self::PassFail {
                positive_label,
                negative_label,
            } => {
                config.insert(
                    "positive_label".to_string(),
                    Value::String(positive_label.clone()),
                );
                config.insert(
                    "negative_label".to_string(),
                    Value::String(negative_label.clone()),
                );
            }
            Self::Categorical {
                options,
                multi_select,
            } => {
                config.insert(
                    "options".to_string(),
                    Value::Array(options.iter().cloned().map(Value::String).collect()),
                );
                config.insert("multi_select".to_string(), Value::Bool(*multi_select));
            }
            Self::Numeric {
                min_value,
                max_value,
            } => {
                config.insert("min_value".to_string(), optional_number(*min_value));
                config.insert("max_value".to_string(), optional_number(*max_value));
            }
            Self::Text { max_length } => {
                config.insert(
                    "max_length".to_string(),
                    max_length.map_or(Value::Null, |value| Value::Number(value.into())),
                );
            }
        }
        python_json_dumps(&Value::Object(config), false)
    }
}

fn optional_number(value: Option<f64>) -> Value {
    value
        .and_then(Number::from_f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LabelSchema {
    pub schema_id: String,
    pub experiment_id: String,
    pub name: String,
    pub schema_type: LabelSchemaType,
    pub input: LabelSchemaInput,
    pub instruction: Option<String>,
    pub enable_comment: bool,
    pub created_by: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub is_default: bool,
}

#[derive(Debug, Clone, Default)]
pub struct LabelSchemaUpdate<'a> {
    pub name: Option<&'a str>,
    pub instruction: Option<&'a str>,
    pub enable_comment: Option<bool>,
    pub input: Option<&'a LabelSchemaInput>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LabelSchemasPage {
    pub schemas: Vec<LabelSchema>,
    pub next_page_token: Option<String>,
}

impl TrackingStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn create_label_schema(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
        schema_type: LabelSchemaType,
        input: &LabelSchemaInput,
        instruction: Option<&str>,
        enable_comment: bool,
    ) -> Result<LabelSchema, MlflowError> {
        validate_create(name, input, instruction)?;
        let parsed_experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, parsed_experiment_id)
            .await?;
        if self
            .find_label_schema_by_name(workspace, parsed_experiment_id, name)
            .await?
            .is_some()
        {
            return Err(label_schema_exists(name, experiment_id));
        }

        let schema_id = format!("{SCHEMA_ID_PREFIX}{}", Uuid::new_v4().simple());
        let now = now_millis();
        let dialect = self.db().dialect();
        let placeholders = (1..=11)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        let result = self
            .db()
            .exec(
                &format!(
                    "INSERT INTO label_schemas (schema_id, experiment_id, name, type, instruction, \
                     enable_comment, input_type, input_config, created_by, created_time, \
                     last_update_time) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(schema_id.clone()),
                    Val::Int(parsed_experiment_id),
                    Val::Text(name.to_string()),
                    Val::Text(schema_type.as_str().to_string()),
                    Val::OptText(instruction.map(str::to_string)),
                    Val::Bool(enable_comment),
                    Val::Text(input.input_type().to_string()),
                    Val::Text(input.config_json()),
                    Val::OptText(None),
                    Val::Int(now),
                    Val::Int(now),
                ],
            )
            .await;
        if let Err(error) = result {
            if super::experiments::is_unique_violation(&error) {
                return Err(label_schema_exists(name, experiment_id));
            }
            return Err(internal(error));
        }
        self.get_label_schema(workspace, &schema_id).await
    }

    pub async fn get_label_schema(
        &self,
        workspace: &str,
        schema_id: &str,
    ) -> Result<LabelSchema, MlflowError> {
        self.find_label_schema_by_id(workspace, schema_id)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Label schema with id '{schema_id}' not found."
                ))
            })
    }

    pub async fn get_label_schema_by_name(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
    ) -> Result<LabelSchema, MlflowError> {
        let parsed_experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, parsed_experiment_id)
            .await?;
        self.find_label_schema_by_name(workspace, parsed_experiment_id, name)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Label schema with name '{name}' not found for experiment '{experiment_id}'."
                ))
            })
    }

    pub async fn list_label_schemas(
        &self,
        workspace: &str,
        experiment_id: &str,
        max_results: i32,
        page_token: Option<&str>,
    ) -> Result<LabelSchemasPage, MlflowError> {
        validate_max_results(max_results)?;
        let parsed_experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, parsed_experiment_id)
            .await?;
        self.ensure_default_label_schema(workspace, parsed_experiment_id)
            .await?;
        let offset = mlflow_search::parse_start_offset_from_page_token(page_token)
            .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
        let dialect = self.db().dialect();
        let mut schemas = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT ls.schema_id, ls.experiment_id, ls.name, ls.type, ls.instruction, \
                     ls.enable_comment, ls.input_type, ls.input_config, ls.created_by, \
                     ls.created_time, ls.last_update_time, ls.is_default FROM label_schemas ls \
                     JOIN experiments e ON e.experiment_id = ls.experiment_id WHERE \
                     ls.experiment_id = {} AND e.workspace = {} ORDER BY ls.created_time DESC, \
                     ls.schema_id ASC LIMIT {} OFFSET {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    i64::from(max_results) + 1,
                    offset
                ),
                &[
                    Val::Int(parsed_experiment_id),
                    Val::Text(workspace.to_string()),
                ],
                map_label_schema,
            )
            .await
            .map_err(internal)?;
        let next_page_token = if schemas.len() > max_results as usize {
            schemas.truncate(max_results as usize);
            Some(mlflow_search::create_page_token(
                offset + i64::from(max_results),
            ))
        } else {
            None
        };
        Ok(LabelSchemasPage {
            schemas,
            next_page_token,
        })
    }

    pub async fn update_label_schema(
        &self,
        workspace: &str,
        schema_id: &str,
        update: LabelSchemaUpdate<'_>,
    ) -> Result<LabelSchema, MlflowError> {
        let existing = self.get_label_schema(workspace, schema_id).await?;
        if existing.is_default {
            return Err(MlflowError::invalid_parameter_value(
                "The experiment's default question cannot be edited.",
            ));
        }
        if let Some(name) = update.name {
            validate_name(name)?;
            validate_not_reserved_name(name)?;
            if name != existing.name
                && self
                    .find_label_schema_by_name(
                        workspace,
                        parse_experiment_id(&existing.experiment_id)?,
                        name,
                    )
                    .await?
                    .is_some()
            {
                return Err(label_schema_exists(name, &existing.experiment_id));
            }
        }
        if let Some(instruction) = update.instruction {
            validate_instruction(Some(instruction))?;
        }
        if let Some(input) = update.input {
            validate_input(input)?;
            validate_input_immutable(&existing.input, input)?;
        }

        let dialect = self.db().dialect();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let Some(name) = update.name {
            values.push(Val::Text(name.to_string()));
            assignments.push(format!("name = {}", dialect.placeholder(values.len())));
        }
        if let Some(instruction) = update.instruction {
            values.push(Val::Text(instruction.to_string()));
            assignments.push(format!(
                "instruction = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(enable_comment) = update.enable_comment {
            values.push(Val::Bool(enable_comment));
            assignments.push(format!(
                "enable_comment = {}",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(input) = update.input {
            values.push(Val::Text(input.input_type().to_string()));
            assignments.push(format!(
                "input_type = {}",
                dialect.placeholder(values.len())
            ));
            values.push(Val::Text(input.config_json()));
            assignments.push(format!(
                "input_config = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_update_time = {}",
            dialect.placeholder(values.len())
        ));
        values.push(Val::Text(schema_id.to_string()));
        let schema_placeholder = dialect.placeholder(values.len());
        values.push(Val::Text(workspace.to_string()));
        let workspace_placeholder = dialect.placeholder(values.len());
        let result = self
            .db()
            .exec(
                &format!(
                    "UPDATE label_schemas SET {} WHERE schema_id = {schema_placeholder} AND \
                     experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = \
                     {workspace_placeholder})",
                    assignments.join(", ")
                ),
                &values,
            )
            .await;
        if let Err(error) = result {
            if super::experiments::is_unique_violation(&error) {
                return Err(label_schema_exists(
                    update.name.unwrap_or(&existing.name),
                    &existing.experiment_id,
                ));
            }
            return Err(internal(error));
        }
        self.get_label_schema(workspace, schema_id).await
    }

    pub async fn delete_label_schema(
        &self,
        workspace: &str,
        schema_id: &str,
    ) -> Result<(), MlflowError> {
        let Some(schema) = self.find_label_schema_by_id(workspace, schema_id).await? else {
            return Ok(());
        };
        if schema.is_default {
            return Err(MlflowError::invalid_parameter_value(
                "The experiment's default question cannot be deleted.",
            ));
        }
        let dialect = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM label_schemas WHERE schema_id = {} AND experiment_id IN \
                     (SELECT experiment_id FROM experiments WHERE workspace = {})",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(schema_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    async fn find_label_schema_by_id(
        &self,
        workspace: &str,
        schema_id: &str,
    ) -> Result<Option<LabelSchema>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT ls.schema_id, ls.experiment_id, ls.name, ls.type, ls.instruction, \
                     ls.enable_comment, ls.input_type, ls.input_config, ls.created_by, \
                     ls.created_time, ls.last_update_time, ls.is_default FROM label_schemas ls \
                     JOIN experiments e ON e.experiment_id = ls.experiment_id WHERE \
                     ls.schema_id = {} AND e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(schema_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_label_schema,
            )
            .await
            .map_err(internal)
    }

    async fn find_label_schema_by_name(
        &self,
        workspace: &str,
        experiment_id: i64,
        name: &str,
    ) -> Result<Option<LabelSchema>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT ls.schema_id, ls.experiment_id, ls.name, ls.type, ls.instruction, \
                     ls.enable_comment, ls.input_type, ls.input_config, ls.created_by, \
                     ls.created_time, ls.last_update_time, ls.is_default FROM label_schemas ls \
                     JOIN experiments e ON e.experiment_id = ls.experiment_id WHERE \
                     ls.experiment_id = {} AND ls.name = {} AND e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Int(experiment_id),
                    Val::Text(name.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_label_schema,
            )
            .await
            .map_err(internal)
    }

    async fn ensure_default_label_schema(
        &self,
        workspace: &str,
        experiment_id: i64,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let existing = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT ls.schema_id FROM label_schemas ls JOIN experiments e ON \
                     e.experiment_id = ls.experiment_id WHERE ls.experiment_id = {} AND \
                     ls.is_default = {} AND e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Int(experiment_id),
                    Val::Bool(true),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("schema_id"),
            )
            .await
            .map_err(internal)?;
        if existing.is_some() {
            return Ok(());
        }
        let now = now_millis();
        let input = LabelSchemaInput::Text { max_length: None };
        let placeholders = (1..=12)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        let result = self
            .db()
            .exec(
                &format!(
                    "INSERT INTO label_schemas (schema_id, experiment_id, name, type, instruction, \
                     enable_comment, input_type, input_config, created_by, created_time, \
                     last_update_time, is_default) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(format!("{SCHEMA_ID_PREFIX}{}", Uuid::new_v4().simple())),
                    Val::Int(experiment_id),
                    Val::Text(DEFAULT_LABEL_SCHEMA_NAME.to_string()),
                    Val::Text(LabelSchemaType::Feedback.as_str().to_string()),
                    Val::Text(DEFAULT_LABEL_SCHEMA_INSTRUCTION.to_string()),
                    Val::Bool(false),
                    Val::Text(input.input_type().to_string()),
                    Val::Text(input.config_json()),
                    Val::OptText(None),
                    Val::Int(now),
                    Val::Int(now),
                    Val::Bool(true),
                ],
            )
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(error) if super::experiments::is_unique_violation(&error) => Ok(()),
            Err(error) => Err(internal(error)),
        }
    }
}

fn map_label_schema(row: &dyn RowLike) -> Result<LabelSchema, sqlx::Error> {
    let schema_type = match row.get_string("type")?.as_str() {
        "feedback" => LabelSchemaType::Feedback,
        "expectation" => LabelSchemaType::Expectation,
        other => {
            return Err(sqlx::Error::Decode(
                format!("unknown label schema type {other:?}").into(),
            ))
        }
    };
    let input_type = row.get_string("input_type")?;
    let config: Value = serde_json::from_str(&row.get_string("input_config")?)
        .map_err(|error| sqlx::Error::Decode(Box::new(error)))?;
    let input = input_from_config(&input_type, &config)?;
    Ok(LabelSchema {
        schema_id: row.get_string("schema_id")?,
        experiment_id: row.get_int("experiment_id")?.to_string(),
        name: row.get_string("name")?,
        schema_type,
        input,
        instruction: row.get_opt_string("instruction")?,
        enable_comment: row.get_bool("enable_comment")?,
        created_by: row.get_opt_string("created_by")?,
        created_at: row.get_i64("created_time")?,
        updated_at: row.get_i64("last_update_time")?,
        is_default: row.get_bool("is_default")?,
    })
}

fn input_from_config(input_type: &str, config: &Value) -> Result<LabelSchemaInput, sqlx::Error> {
    let object = config
        .as_object()
        .ok_or_else(|| sqlx::Error::Decode("label schema input_config is not an object".into()))?;
    let string = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| sqlx::Error::Decode(format!("missing input field {key}").into()))
    };
    Ok(match input_type {
        "pass_fail" => LabelSchemaInput::PassFail {
            positive_label: string("positive_label")?,
            negative_label: string("negative_label")?,
        },
        "categorical" => LabelSchemaInput::Categorical {
            options: object
                .get("options")
                .and_then(Value::as_array)
                .ok_or_else(|| sqlx::Error::Decode("missing input field options".into()))?
                .iter()
                .map(|value| {
                    value.as_str().map(str::to_string).ok_or_else(|| {
                        sqlx::Error::Decode("categorical option is not a string".into())
                    })
                })
                .collect::<Result<_, _>>()?,
            multi_select: object
                .get("multi_select")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        "numeric" => LabelSchemaInput::Numeric {
            min_value: object.get("min_value").and_then(Value::as_f64),
            max_value: object.get("max_value").and_then(Value::as_f64),
        },
        "text" => LabelSchemaInput::Text {
            max_length: object.get("max_length").and_then(Value::as_i64),
        },
        other => {
            return Err(sqlx::Error::Decode(
                format!("unknown label schema input_type {other:?}").into(),
            ))
        }
    })
}

fn validate_create(
    name: &str,
    input: &LabelSchemaInput,
    instruction: Option<&str>,
) -> Result<(), MlflowError> {
    validate_name(name)?;
    validate_not_reserved_name(name)?;
    validate_instruction(instruction)?;
    validate_input(input)
}

fn validate_name(name: &str) -> Result<(), MlflowError> {
    if name.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Label schema `name` must be a non-empty string; got ''.",
        ));
    }
    if name.chars().count() > 250 {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Label schema `name` must be at most 250 characters; got {}.",
            name.chars().count()
        )));
    }
    Ok(())
}

fn validate_not_reserved_name(name: &str) -> Result<(), MlflowError> {
    if name.trim().eq_ignore_ascii_case(DEFAULT_LABEL_SCHEMA_NAME) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "`{name}` is reserved for the experiment's default question and cannot be used for \
             another label schema."
        )));
    }
    Ok(())
}

fn validate_instruction(instruction: Option<&str>) -> Result<(), MlflowError> {
    if let Some(instruction) = instruction {
        if instruction.chars().count() > 1000 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Label schema `instruction` must be at most 1000 characters; got {}.",
                instruction.chars().count()
            )));
        }
    }
    Ok(())
}

fn validate_input(input: &LabelSchemaInput) -> Result<(), MlflowError> {
    match input {
        LabelSchemaInput::PassFail {
            positive_label,
            negative_label,
        } => {
            for (field, label) in [
                ("positive_label", positive_label),
                ("negative_label", negative_label),
            ] {
                if label.is_empty() {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "`InputPassFail.{field}` must be a non-empty string; got ''."
                    )));
                }
                if label.chars().count() > 64 {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "`InputPassFail.{field}` must be at most 64 characters; got {}.",
                        label.chars().count()
                    )));
                }
            }
            if positive_label == negative_label {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "`InputPassFail.positive_label` and `negative_label` must be distinct; got \
                     '{positive_label}' for both."
                )));
            }
        }
        LabelSchemaInput::Categorical { options, .. } => {
            if options.is_empty() {
                return Err(MlflowError::invalid_parameter_value(
                    "`InputCategorical.options` must be a non-empty list; got [].",
                ));
            }
            if options.len() > 10 {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "`InputCategorical.options` must have at most 10 entries; got {}.",
                    options.len()
                )));
            }
            let mut seen = std::collections::HashSet::new();
            for option in options {
                if option.is_empty() {
                    return Err(MlflowError::invalid_parameter_value(
                        "`InputCategorical.options` entries must be non-empty strings; got ''.",
                    ));
                }
                if option.chars().count() > 64 {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "`InputCategorical.options` entries must be at most 64 characters; got {} \
                         for '{option}'.",
                        option.chars().count()
                    )));
                }
                if !seen.insert(option) {
                    return Err(MlflowError::invalid_parameter_value(format!(
                        "`InputCategorical.options` must be deduplicated; '{option}' appears twice."
                    )));
                }
            }
        }
        LabelSchemaInput::Numeric {
            min_value: Some(min),
            max_value: Some(max),
        } if min >= max => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "`InputNumeric.min_value` must be strictly less than `max_value`; got min={min}, \
                 max={max}."
            )));
        }
        LabelSchemaInput::Text {
            max_length: Some(max_length),
        } if *max_length < 1 => {
            return Err(MlflowError::invalid_parameter_value(format!(
                "`InputText.max_length` must be at least 1; got {max_length}."
            )));
        }
        _ => {}
    }
    Ok(())
}

fn validate_input_immutable(
    existing: &LabelSchemaInput,
    new: &LabelSchemaInput,
) -> Result<(), MlflowError> {
    if existing.input_type() != new.input_type() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "A label schema's input type cannot be changed after creation (existing: {}, got: \
             {}).",
            existing.python_class_name(),
            new.python_class_name()
        )));
    }
    if let (
        LabelSchemaInput::Categorical {
            multi_select: existing,
            ..
        },
        LabelSchemaInput::Categorical {
            multi_select: new, ..
        },
    ) = (existing, new)
    {
        if existing != new {
            return Err(MlflowError::invalid_parameter_value(
                "`InputCategorical.multi_select` cannot be changed after creation.",
            ));
        }
    }
    Ok(())
}

fn validate_max_results(max_results: i32) -> Result<(), MlflowError> {
    if max_results < 1 {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. It must be a \
             positive integer"
        )));
    }
    if i64::from(max_results) > SEARCH_MAX_RESULTS_THRESHOLD {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {max_results} for parameter 'max_results' supplied. It must be at most \
             {SEARCH_MAX_RESULTS_THRESHOLD}"
        )));
    }
    Ok(())
}

fn label_schema_exists(name: &str, experiment_id: &str) -> MlflowError {
    MlflowError::resource_already_exists(format!(
        "Label schema with name '{name}' already exists for experiment '{experiment_id}'."
    ))
}
