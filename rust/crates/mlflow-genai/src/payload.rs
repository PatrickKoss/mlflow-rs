use std::collections::BTreeMap;

use serde_json::{Map, Value};

const PINNED_MLFLOW_VERSION: &str = "3.14.1.dev0";

const BUILTIN_SCORERS: [&str; 24] = [
    "Completeness",
    "ConversationCompleteness",
    "ConversationalGuidelines",
    "ConversationalRoleAdherence",
    "ConversationalSafety",
    "ConversationalToolCallEfficiency",
    "Correctness",
    "Equivalence",
    "ExpectationsGuidelines",
    "Fluency",
    "Guidelines",
    "KnowledgeRetention",
    "PIIDetection",
    "RegexMatch",
    "RelevanceToQuery",
    "ResponseLength",
    "RetrievalGroundedness",
    "RetrievalRelevance",
    "RetrievalSufficiency",
    "Safety",
    "Summarization",
    "ToolCallCorrectness",
    "ToolCallEfficiency",
    "UserFrustration",
];

pub fn supported_builtin_scorers() -> &'static [&'static str] {
    &BUILTIN_SCORERS
}

const THIRD_PARTY_MODULES: [&str; 4] = [
    "mlflow.genai.scorers.deepeval",
    "mlflow.genai.scorers.phoenix",
    "mlflow.genai.scorers.ragas",
    "mlflow.genai.scorers.trulens",
];

/// Fields shared by every Python `SerializedScorer` representation.
#[derive(Debug, Clone, PartialEq)]
pub struct SerializedScorerCommon {
    pub name: String,
    pub aggregations: Option<Vec<String>>,
    pub description: Option<String>,
    pub is_session_level_scorer: bool,
    pub mlflow_version: String,
    pub serialization_version: u32,
    /// Additive fields from newer Python clients are retained for diagnostics.
    pub unknown_fields: BTreeMap<String, Value>,
    raw: Map<String, Value>,
}

/// A scorer from `mlflow.genai.scorers.builtin_scorers`.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltinScorerPayload {
    pub common: SerializedScorerCommon,
    pub class_name: String,
    pub pydantic_data: Map<String, Value>,
}

/// A scorer produced by Python's `InstructionsJudge.model_dump()`.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionsJudgePayload {
    pub common: SerializedScorerCommon,
    pub pydantic_data: Map<String, Value>,
}

/// The mutually exclusive persisted scorer forms from Python `SerializedScorer`.
#[derive(Debug, Clone, PartialEq)]
pub enum SerializedScorer {
    Builtin(BuiltinScorerPayload),
    Instructions(InstructionsJudgePayload),
    Decorator {
        common: SerializedScorerCommon,
        call_source: String,
        call_signature: Option<String>,
        original_func_name: Option<String>,
    },
    MemoryAugmented {
        common: SerializedScorerCommon,
        data: Map<String, Value>,
    },
    ThirdParty {
        common: SerializedScorerCommon,
        data: Map<String, Value>,
    },
}

impl SerializedScorer {
    /// Parse the JSON string stored in the scorer CRUD `serialized_scorer` column.
    pub fn from_json(json: &str) -> Result<Self, ScorerPayloadError> {
        let value: Value = serde_json::from_str(json)?;
        Self::from_value(value)
    }

    pub fn from_value(value: Value) -> Result<Self, ScorerPayloadError> {
        let raw = value.as_object().cloned().ok_or_else(|| {
            ScorerPayloadError::InvalidData(format!(
                "Invalid scorer data: expected a SerializedScorer object or dictionary, got {}. Scorer data must be either a SerializedScorer object or a dictionary containing serialized scorer information.",
                python_type_name(&value)
            ))
        })?;
        parse_raw(raw)
    }

    pub fn common(&self) -> &SerializedScorerCommon {
        match self {
            Self::Builtin(payload) => &payload.common,
            Self::Instructions(payload) => &payload.common,
            Self::Decorator { common, .. }
            | Self::MemoryAugmented { common, .. }
            | Self::ThirdParty { common, .. } => common,
        }
    }

    /// Return the input object unchanged, including additive fields from newer clients.
    pub fn to_json_value(&self) -> Value {
        Value::Object(self.common().raw.clone())
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.to_json_value())
    }

    /// Apply Python's server-side `Scorer.model_validate_json` execution checks.
    pub fn validate_for_oss_execution(&self) -> Result<(), ScorerPayloadError> {
        match self {
            Self::Decorator {
                common,
                call_source,
                call_signature,
                original_func_name,
            } => {
                let (Some(signature), Some(function_name)) =
                    (call_signature.as_deref(), original_func_name.as_deref())
                else {
                    return Err(unknown_format(common));
                };
                let mut snippet = format!(
                    "\n\nfrom mlflow.genai import scorer\n\n@scorer\ndef {function_name}{signature}:\n"
                );
                for line in call_source.split('\n') {
                    snippet.push_str("    ");
                    snippet.push_str(line);
                    snippet.push('\n');
                }
                Err(ScorerPayloadError::DecoratorRejected { snippet })
            }
            Self::ThirdParty { common, data } => validate_third_party(common, data),
            _ => Ok(()),
        }
    }

    pub fn is_third_party(&self) -> bool {
        matches!(self, Self::ThirdParty { .. })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScorerPayloadError {
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    InvalidData(String),
    #[error("{0}")]
    Validation(String),
    #[error(
        "Custom scorer registration (using @scorer decorator) is not supported outside of Databricks tracking environments due to security concerns. Custom scorers require arbitrary code execution during deserialization.\n\nTo use custom scorers:\n1. Configure MLflow to use a Databricks tracking URI, or\n2. Manage your custom scorer code in a source code repository (e.g., GitHub) and import it directly, or\n3. Use built-in scorers or make_judge() scorers instead.\nRegistered scorer code:\n{snippet}"
    )]
    DecoratorRejected { snippet: String },
    #[error(
        "Phoenix scorer metric '{metric}' is unavailable in the Rust server: arize-phoenix-evals is licensed under Elastic-2.0, which is incompatible with reimplementation in Apache-2.0 MLflow. Use the MLflow builtins Faithfulness (Hallucination), RelevanceToQuery (Relevance), Correctness (QA), or Safety (Toxicity); for Summarization and SQL, use a custom instructions judge."
    )]
    PhoenixLicense { metric: String },
}

fn parse_raw(raw: Map<String, Value>) -> Result<SerializedScorer, ScorerPayloadError> {
    let name = match raw.get("name") {
        None => {
            return Err(ScorerPayloadError::InvalidData(
                "Failed to parse serialized scorer data: SerializedScorer.__init__() missing 1 required positional argument: 'name'".to_string(),
            ));
        }
        Some(Value::String(name)) => name.clone(),
        Some(value) => {
            return Err(ScorerPayloadError::InvalidData(format!(
                "Failed to parse serialized scorer data: field 'name' must be str, got {}",
                python_type_name(value)
            )));
        }
    };

    let representation_fields = [
        "builtin_scorer_class",
        "call_source",
        "instructions_judge_pydantic_data",
        "memory_augmented_judge_data",
        "third_party_scorer_data",
    ];
    let present = representation_fields
        .iter()
        .filter(|field| raw.get(**field).is_some_and(|value| !value.is_null()))
        .copied()
        .collect::<Vec<_>>();
    match present.len() {
        0 => {
            return Err(ScorerPayloadError::InvalidData(
                "Failed to parse serialized scorer data: SerializedScorer must have either builtin scorer fields (builtin_scorer_class), decorator scorer fields (call_source), instructions judge fields (instructions_judge_pydantic_data), memory augmented judge fields (memory_augmented_judge_data), or third-party scorer fields (third_party_scorer_data) present".to_string(),
            ));
        }
        1 => {}
        _ => {
            return Err(ScorerPayloadError::InvalidData(
                "Failed to parse serialized scorer data: SerializedScorer cannot have multiple types of scorer fields present simultaneously".to_string(),
            ));
        }
    }

    let common = parse_common(&raw, name)?;
    match present[0] {
        "builtin_scorer_class" => {
            let class_name = string_field(&raw, "builtin_scorer_class")?;
            if !BUILTIN_SCORERS.contains(&class_name.as_str()) {
                return Err(ScorerPayloadError::Validation(format!(
                    "Unknown builtin scorer class: {class_name}"
                )));
            }
            let pydantic_data = optional_object(&raw, "builtin_scorer_pydantic_data")?;
            validate_builtin(&class_name, &pydantic_data)?;
            Ok(SerializedScorer::Builtin(BuiltinScorerPayload {
                common,
                class_name,
                pydantic_data,
            }))
        }
        "call_source" => Ok(SerializedScorer::Decorator {
            common,
            call_source: string_field(&raw, "call_source")?,
            call_signature: optional_string(&raw, "call_signature")?,
            original_func_name: optional_string(&raw, "original_func_name")?,
        }),
        "instructions_judge_pydantic_data" => {
            let pydantic_data = required_object(&raw, "instructions_judge_pydantic_data")?;
            validate_instructions(&common.name, &pydantic_data)?;
            Ok(SerializedScorer::Instructions(InstructionsJudgePayload {
                common,
                pydantic_data,
            }))
        }
        "memory_augmented_judge_data" => {
            let data = required_object(&raw, "memory_augmented_judge_data")?;
            validate_memory(&data)?;
            Ok(SerializedScorer::MemoryAugmented { common, data })
        }
        "third_party_scorer_data" => Ok(SerializedScorer::ThirdParty {
            common,
            data: required_object(&raw, "third_party_scorer_data")?,
        }),
        _ => unreachable!("representation field comes from closed list"),
    }
}

fn parse_common(
    raw: &Map<String, Value>,
    name: String,
) -> Result<SerializedScorerCommon, ScorerPayloadError> {
    let aggregations = match raw.get("aggregations") {
        None | Some(Value::Null) => None,
        Some(Value::Array(values)) => Some(
            values
                .iter()
                .map(|value| {
                    value.as_str().map(str::to_string).ok_or_else(|| {
                        ScorerPayloadError::InvalidData(format!(
                            "Failed to parse serialized scorer data: aggregation must be str, got {}",
                            python_type_name(value)
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Some(value) => {
            return Err(ScorerPayloadError::InvalidData(format!(
                "Failed to parse serialized scorer data: field 'aggregations' must be list, got {}",
                python_type_name(value)
            )));
        }
    };
    let description = optional_string(raw, "description")?;
    let is_session_level_scorer = match raw.get("is_session_level_scorer") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(value) => {
            return Err(ScorerPayloadError::InvalidData(format!(
                "Failed to parse serialized scorer data: field 'is_session_level_scorer' must be bool, got {}",
                python_type_name(value)
            )));
        }
    };
    let mlflow_version = match raw.get("mlflow_version") {
        None => PINNED_MLFLOW_VERSION.to_string(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Null) => String::new(),
        Some(value) => value.to_string(),
    };
    let serialization_version = match raw.get("serialization_version") {
        None | Some(Value::Null) => 1,
        Some(Value::Number(value)) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                ScorerPayloadError::InvalidData(
                    "Failed to parse serialized scorer data: serialization_version must be an integer".to_string(),
                )
            })?,
        Some(_) => {
            return Err(ScorerPayloadError::InvalidData(
                "Failed to parse serialized scorer data: serialization_version must be an integer".to_string(),
            ));
        }
    };
    let known = [
        "name",
        "aggregations",
        "description",
        "is_session_level_scorer",
        "mlflow_version",
        "serialization_version",
        "builtin_scorer_class",
        "builtin_scorer_pydantic_data",
        "call_source",
        "call_signature",
        "original_func_name",
        "instructions_judge_pydantic_data",
        "memory_augmented_judge_data",
        "third_party_scorer_data",
    ];
    let unknown_fields = raw
        .iter()
        .filter(|(key, _)| !known.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(SerializedScorerCommon {
        name,
        aggregations,
        description,
        is_session_level_scorer,
        mlflow_version,
        serialization_version,
        unknown_fields,
        raw: raw.clone(),
    })
}

fn validate_builtin(class_name: &str, data: &Map<String, Value>) -> Result<(), ScorerPayloadError> {
    match class_name {
        "ResponseLength" => {
            let min = optional_i64(data, "min_length")?;
            let max = optional_i64(data, "max_length")?;
            if min.is_none() && max.is_none() {
                return Err(pydantic_value_error(
                    class_name,
                    data,
                    "ResponseLength requires at least one of `min_length` or `max_length`.",
                ));
            }
            if let Some(value) = min.filter(|value| *value < 0) {
                return Err(pydantic_value_error(
                    class_name,
                    data,
                    &format!("`min_length` must be non-negative, got {value}."),
                ));
            }
            if let Some(value) = max.filter(|value| *value < 0) {
                return Err(pydantic_value_error(
                    class_name,
                    data,
                    &format!("`max_length` must be non-negative, got {value}."),
                ));
            }
            if let (Some(min), Some(max)) = (min, max) {
                if min > max {
                    return Err(pydantic_value_error(
                        class_name,
                        data,
                        &format!("`min_length` ({min}) must be <= `max_length` ({max})."),
                    ));
                }
            }
        }
        "RegexMatch" => {
            if !data.get("pattern").is_some_and(Value::is_string) {
                return Err(ScorerPayloadError::Validation(
                    "1 validation error for RegexMatch\npattern\n  Field required [type=missing, input_value={}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.13/v/missing".to_string(),
                ));
            }
        }
        "Guidelines" | "ConversationalGuidelines" => {
            if !data.get("guidelines").is_some_and(|value| {
                value.is_string()
                    || value
                        .as_array()
                        .is_some_and(|items| items.iter().all(Value::is_string))
            }) {
                return Err(ScorerPayloadError::Validation(format!(
                    "1 validation error for {class_name}\nguidelines\n  Field required [type=missing, input_value={{}}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.13/v/missing"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_instructions(
    scorer_name: &str,
    data: &Map<String, Value>,
) -> Result<(), ScorerPayloadError> {
    let mut errors = Vec::new();
    for field in ["instructions", "model"] {
        match data.get(field) {
            None => errors.push(format!("missing required field '{field}'")),
            Some(Value::String(_)) => {}
            Some(value) => errors.push(format!(
                "field '{field}' must be str, got {}",
                python_type_name(value)
            )),
        }
    }
    if !errors.is_empty() {
        return Err(ScorerPayloadError::Validation(format!(
            "Failed to deserialize InstructionsJudge scorer '{scorer_name}': {}",
            errors.join("; ")
        )));
    }
    if let Some(schema) = data.get("feedback_value_type") {
        validate_feedback_schema(schema)?;
    }
    let instructions = data["instructions"].as_str().expect("validated above");
    let variables = template_variables(instructions);
    if variables.is_empty() {
        return Err(ScorerPayloadError::Validation(format!(
            "Failed to create InstructionsJudge scorer '{scorer_name}': Instructions template must contain at least one variable (e.g., {{{{ inputs }}}}, {{{{ outputs }}}}, {{{{ trace }}}}, {{{{ expectations }}}}, or {{{{ conversation }}}})."
        )));
    }
    let allowed = ["inputs", "outputs", "trace", "expectations", "conversation"];
    let custom = variables
        .iter()
        .filter(|variable| !allowed.contains(&variable.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !custom.is_empty() {
        return Err(ScorerPayloadError::Validation(format!(
            "Failed to create InstructionsJudge scorer '{scorer_name}': Instructions template contains unsupported variables: {{{}}}. Only the following variables are allowed: inputs, outputs, trace, expectations, conversation",
            custom.join(", ")
        )));
    }
    if variables.contains(&"conversation".to_string())
        && variables
            .iter()
            .any(|variable| !["conversation", "expectations"].contains(&variable.as_str()))
    {
        return Err(ScorerPayloadError::Validation(format!(
            "Failed to create InstructionsJudge scorer '{scorer_name}': Instructions template must not contain any template variables other than {{{{ expectations }}}} if {{{{ conversation }}}} is provided."
        )));
    }
    Ok(())
}

fn validate_feedback_schema(schema: &Value) -> Result<(), ScorerPayloadError> {
    let Some(schema) = schema.as_object() else {
        return Err(ScorerPayloadError::Validation(format!(
            "Invalid feedback_value_type serialization: {}",
            python_repr(schema)
        )));
    };
    if schema.contains_key("anyOf") && !schema.contains_key("type") {
        let valid = schema
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                let nulls = items
                    .iter()
                    .filter(|item| item.get("type").and_then(Value::as_str) == Some("null"))
                    .count();
                let primitives = items
                    .iter()
                    .filter(|item| {
                        matches!(
                            item.get("type").and_then(Value::as_str),
                            Some("string" | "integer" | "number" | "boolean")
                        )
                    })
                    .count();
                items.len() == 2 && nulls == 1 && primitives == 1
            });
        if valid {
            return Ok(());
        }
        return Err(ScorerPayloadError::Validation(format!(
            "Invalid feedback_value_type serialization for anyOf-with-null. Expected exactly one null schema and one non-null schema with a supported primitive type: {}",
            python_repr(&Value::Object(schema.clone()))
        )));
    }
    let Some(schema_type) = schema.get("type").and_then(Value::as_str) else {
        return Err(ScorerPayloadError::Validation(format!(
            "Invalid feedback_value_type serialization: {}",
            python_repr(&Value::Object(schema.clone()))
        )));
    };
    if schema.contains_key("enum") {
        if schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        {
            return Ok(());
        }
        return Err(ScorerPayloadError::Validation(format!(
            "Enum must have at least one value: {}",
            python_repr(&Value::Object(schema.clone()))
        )));
    }
    match schema_type {
        "string" | "integer" | "number" | "boolean" => Ok(()),
        "object" => {
            let Some(inner) = schema.get("additionalProperties").and_then(Value::as_object) else {
                return Err(ScorerPayloadError::Validation(format!(
                    "Object type missing 'additionalProperties' field: {}",
                    python_repr(&Value::Object(schema.clone()))
                )));
            };
            validate_primitive_inner(inner, "additionalProperties")
        }
        "array" => {
            let Some(inner) = schema.get("items").and_then(Value::as_object) else {
                return Err(ScorerPayloadError::Validation(format!(
                    "Array type missing 'items' field: {}",
                    python_repr(&Value::Object(schema.clone()))
                )));
            };
            validate_primitive_inner(inner, "items")
        }
        other => Err(ScorerPayloadError::Validation(format!(
            "Unsupported JSON Schema type: {other}. Only string, integer, number, boolean, object, and array are supported."
        ))),
    }
}

fn validate_primitive_inner(
    schema: &Map<String, Value>,
    field: &str,
) -> Result<(), ScorerPayloadError> {
    let Some(inner_type) = schema.get("type").and_then(Value::as_str) else {
        return Err(ScorerPayloadError::Validation(format!(
            "{field} missing 'type' field: {}",
            python_repr(&Value::Object(schema.clone()))
        )));
    };
    if matches!(inner_type, "string" | "integer" | "number" | "boolean") {
        Ok(())
    } else {
        Err(ScorerPayloadError::Validation(format!(
            "Unsupported value type in {field}: {inner_type}"
        )))
    }
}

fn validate_third_party(
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
) -> Result<(), ScorerPayloadError> {
    let module = data.get("module").and_then(Value::as_str).unwrap_or("");
    if module.is_empty()
        || module == "mlflow.genai.scorers.phoenix"
        || module.starts_with("mlflow.genai.scorers.phoenix.")
    {
        if let Some(metric) = phoenix_metric(data) {
            return Err(ScorerPayloadError::PhoenixLicense {
                metric: metric.to_string(),
            });
        }
    }
    if !THIRD_PARTY_MODULES
        .iter()
        .any(|allowed| module == *allowed || module.starts_with(&format!("{allowed}.")))
    {
        return Err(ScorerPayloadError::Validation(format!(
            "Third-party scorer '{}': module '{}' is not in the allow-list ['mlflow.genai.scorers.deepeval', 'mlflow.genai.scorers.phoenix', 'mlflow.genai.scorers.ragas', 'mlflow.genai.scorers.trulens'].",
            common.name, module
        )));
    }
    let class_name = data.get("class").and_then(Value::as_str);
    let metric_name = data.get("metric_name").and_then(Value::as_str);
    if class_name.is_none() || metric_name.is_none() {
        return Err(ScorerPayloadError::Validation(format!(
            "Third-party scorer '{}': missing required fields in third_party_scorer_data (class, metric_name).",
            common.name
        )));
    }
    Ok(())
}

fn validate_memory(data: &Map<String, Value>) -> Result<(), ScorerPayloadError> {
    let base = data
        .get("base_judge")
        .cloned()
        .ok_or_else(|| ScorerPayloadError::Validation("'base_judge'".to_string()))?;
    SerializedScorer::from_value(base)?;
    for (field, default) in [("retrieval_k", 5), ("embedding_dim", 512)] {
        let value = data.get(field).and_then(Value::as_i64).unwrap_or(default);
        if value <= 0 {
            return Err(ScorerPayloadError::Validation(format!(
                "{field} must be a positive integer, got {value}"
            )));
        }
    }
    if let Some(memory) = data.get("semantic_memory") {
        if !memory.is_array() {
            return Err(ScorerPayloadError::Validation(
                "semantic_memory must be a list".to_string(),
            ));
        }
    }
    Ok(())
}

fn phoenix_metric(data: &Map<String, Value>) -> Option<&str> {
    const PHOENIX_METRICS: [&str; 6] = [
        "Hallucination",
        "QA",
        "Relevance",
        "SQL",
        "Summarization",
        "Toxicity",
    ];
    ["class", "metric_name"]
        .into_iter()
        .filter_map(|field| data.get(field).and_then(Value::as_str))
        .find(|metric| PHOENIX_METRICS.contains(metric))
}

fn unknown_format(common: &SerializedScorerCommon) -> ScorerPayloadError {
    ScorerPayloadError::Validation(format!(
        "Failed to load scorer '{}'. The scorer is serialized in an unknown format that cannot be deserialized. Please make sure you are using a compatible MLflow version or recreate the scorer. Scorer was created with MLflow version: {}, serialization version: {}, current MLflow version: {PINNED_MLFLOW_VERSION}.",
        common.name,
        if common.mlflow_version.is_empty() { "unknown" } else { &common.mlflow_version },
        common.serialization_version
    ))
}

fn required_object(
    raw: &Map<String, Value>,
    field: &'static str,
) -> Result<Map<String, Value>, ScorerPayloadError> {
    raw.get(field)
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            ScorerPayloadError::InvalidData(format!(
                "Failed to parse serialized scorer data: {field} must be a dictionary"
            ))
        })
}

fn optional_object(
    raw: &Map<String, Value>,
    field: &'static str,
) -> Result<Map<String, Value>, ScorerPayloadError> {
    match raw.get(field) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(value)) => Ok(value.clone()),
        Some(_) => Err(ScorerPayloadError::InvalidData(format!(
            "Failed to parse serialized scorer data: {field} must be a dictionary"
        ))),
    }
}

fn string_field(
    raw: &Map<String, Value>,
    field: &'static str,
) -> Result<String, ScorerPayloadError> {
    raw.get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            ScorerPayloadError::InvalidData(format!(
                "Failed to parse serialized scorer data: field '{field}' must be str"
            ))
        })
}

fn optional_string(
    raw: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<String>, ScorerPayloadError> {
    match raw.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(value) => Err(ScorerPayloadError::InvalidData(format!(
            "Failed to parse serialized scorer data: field '{field}' must be str, got {}",
            python_type_name(value)
        ))),
    }
}

fn optional_i64(
    data: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<i64>, ScorerPayloadError> {
    match data.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| ScorerPayloadError::Validation(format!("field '{field}' must be int"))),
        Some(_) => Err(ScorerPayloadError::Validation(format!(
            "field '{field}' must be int"
        ))),
    }
}

fn pydantic_value_error(
    class_name: &str,
    data: &Map<String, Value>,
    message: &str,
) -> ScorerPayloadError {
    ScorerPayloadError::Validation(format!(
        "1 validation error for {class_name}\n  Value error, {message} [type=value_error, input_value={}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.13/v/value_error",
        python_repr(&Value::Object(data.clone()))
    ))
}

pub(crate) fn template_variables(template: &str) -> Vec<String> {
    let mut variables = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let tail = &rest[start + 2..];
        let Some(end) = tail.find("}}") else {
            break;
        };
        let variable = tail[..end].trim().to_string();
        if !variables.contains(&variable) {
            variables.push(variable);
        }
        rest = &tail[end + 2..];
    }
    variables
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

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(value) => format!("'{value}'"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("'{key}': {}", python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
