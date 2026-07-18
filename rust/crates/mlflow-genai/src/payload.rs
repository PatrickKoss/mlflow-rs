use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

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
        let raw: RawSerializedScorer = serde_json::from_str(json)?;
        raw.try_into()
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
}

#[derive(Debug, thiserror::Error)]
pub enum ScorerPayloadError {
    #[error("serialized scorer is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("serialized scorer must contain exactly one scorer representation; found {0}")]
    RepresentationCount(usize),
    #[error("{0} must be a JSON object")]
    ExpectedObject(&'static str),
}

#[derive(Deserialize)]
struct RawSerializedScorer {
    name: String,
    #[serde(default)]
    aggregations: Option<Vec<String>>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    is_session_level_scorer: bool,
    #[serde(default)]
    mlflow_version: String,
    #[serde(default = "default_serialization_version")]
    serialization_version: u32,
    #[serde(default)]
    builtin_scorer_class: Option<String>,
    #[serde(default)]
    builtin_scorer_pydantic_data: Option<Value>,
    #[serde(default)]
    call_source: Option<String>,
    #[serde(default)]
    call_signature: Option<String>,
    #[serde(default)]
    original_func_name: Option<String>,
    #[serde(default)]
    instructions_judge_pydantic_data: Option<Value>,
    #[serde(default)]
    memory_augmented_judge_data: Option<Value>,
    #[serde(default)]
    third_party_scorer_data: Option<Value>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

const fn default_serialization_version() -> u32 {
    1
}

impl TryFrom<RawSerializedScorer> for SerializedScorer {
    type Error = ScorerPayloadError;

    fn try_from(raw: RawSerializedScorer) -> Result<Self, Self::Error> {
        let representation_count = usize::from(raw.builtin_scorer_class.is_some())
            + usize::from(raw.call_source.is_some())
            + usize::from(raw.instructions_judge_pydantic_data.is_some())
            + usize::from(raw.memory_augmented_judge_data.is_some())
            + usize::from(raw.third_party_scorer_data.is_some());
        if representation_count != 1 {
            return Err(ScorerPayloadError::RepresentationCount(
                representation_count,
            ));
        }

        let common = SerializedScorerCommon {
            name: raw.name,
            aggregations: raw.aggregations,
            description: raw.description,
            is_session_level_scorer: raw.is_session_level_scorer,
            mlflow_version: raw.mlflow_version,
            serialization_version: raw.serialization_version,
            unknown_fields: raw.unknown_fields,
        };

        if let Some(class_name) = raw.builtin_scorer_class {
            return Ok(Self::Builtin(BuiltinScorerPayload {
                common,
                class_name,
                pydantic_data: as_object(
                    raw.builtin_scorer_pydantic_data,
                    "builtin_scorer_pydantic_data",
                )?,
            }));
        }
        if let Some(call_source) = raw.call_source {
            return Ok(Self::Decorator {
                common,
                call_source,
                call_signature: raw.call_signature,
                original_func_name: raw.original_func_name,
            });
        }
        if let Some(data) = raw.instructions_judge_pydantic_data {
            return Ok(Self::Instructions(InstructionsJudgePayload {
                common,
                pydantic_data: as_object(Some(data), "instructions_judge_pydantic_data")?,
            }));
        }
        if let Some(data) = raw.memory_augmented_judge_data {
            return Ok(Self::MemoryAugmented {
                common,
                data: as_object(Some(data), "memory_augmented_judge_data")?,
            });
        }
        Ok(Self::ThirdParty {
            common,
            data: as_object(raw.third_party_scorer_data, "third_party_scorer_data")?,
        })
    }
}

fn as_object(
    value: Option<Value>,
    field: &'static str,
) -> Result<Map<String, Value>, ScorerPayloadError> {
    match value {
        Some(Value::Object(data)) => Ok(data),
        _ => Err(ScorerPayloadError::ExpectedObject(field)),
    }
}
