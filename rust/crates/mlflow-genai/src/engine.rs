use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::{JobKind, ScorerPayloadError, SerializedScorer};

const JUDGE_BASE_PROMPT: &str = "You are an expert judge tasked with evaluating the performance of an AI\nagent on a particular query. You will be given instructions that describe the criteria and\nmethodology for evaluating the agent's performance on the query.";
const RESULT_DESCRIPTION: &str = "The evaluation rating/result";
const RATIONALE_DESCRIPTION: &str = "Detailed explanation for the evaluation";

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvalItem {
    pub inputs: Option<Value>,
    pub outputs: Option<Value>,
    pub expectations: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AssessmentSource {
    pub source_type: String,
    pub source_id: String,
}

/// Canonical slice of Python `Feedback` used by the worker result envelope.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Feedback {
    pub name: String,
    pub value: Value,
    pub rationale: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<AssessmentSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    ScorerPayload(#[from] ScorerPayloadError),
    #[error("invalid invoke_scorer parameters: {0}")]
    InvalidParams(String),
    #[error("job kind {0} is not implemented by the T15.4 spike")]
    UnsupportedJobKind(JobKind),
    #[error("scorer form is recognized but not executable by the T15.4 spike")]
    UnsupportedScorer,
    #[error("builtin scorer class {0:?} is not implemented by the T15.4 spike")]
    UnsupportedBuiltin(String),
    #[error("missing or invalid scorer field {0:?}")]
    InvalidScorerField(&'static str),
    #[error("instructions judge requires a gateway URL")]
    MissingGatewayUrl,
    #[error("gateway request failed: {0}")]
    Gateway(String),
    #[error("gateway returned malformed completion: {0}")]
    MalformedGatewayResponse(String),
    #[error("result serialization failed: {0}")]
    Serialization(String),
}

/// Shared native scorer/judge execution surface used by workers and inline guardrails.
#[derive(Clone)]
pub struct ScorerExecutor {
    client: reqwest::Client,
}

impl ScorerExecutor {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("reqwest client configuration is static"),
        }
    }

    pub async fn execute(
        &self,
        scorer: &SerializedScorer,
        item: &EvalItem,
        gateway_url: Option<&str>,
    ) -> Result<Feedback, EngineError> {
        match scorer {
            SerializedScorer::Builtin(payload) => execute_builtin(payload, item),
            SerializedScorer::Instructions(payload) => {
                self.execute_instructions(payload, item, gateway_url).await
            }
            _ => Err(EngineError::UnsupportedScorer),
        }
    }

    async fn execute_instructions(
        &self,
        payload: &crate::InstructionsJudgePayload,
        item: &EvalItem,
        gateway_url: Option<&str>,
    ) -> Result<Feedback, EngineError> {
        let instructions = required_str(&payload.pydantic_data, "instructions")?;
        let model_uri = required_str(&payload.pydantic_data, "model")?;
        let model = model_uri
            .split_once(":/")
            .map(|(_, model)| model)
            .filter(|model| !model.is_empty())
            .ok_or(EngineError::InvalidScorerField("model"))?;
        let field_order = instruction_field_order(instructions);
        for field in &field_order {
            let missing = match *field {
                "inputs" => item.inputs.is_none(),
                "outputs" => item.outputs.is_none(),
                _ => false,
            };
            if missing {
                return Err(EngineError::InvalidScorerField(field));
            }
        }
        let system_content = format!(
            "{JUDGE_BASE_PROMPT}\n\nYour task: {instructions}.\n\nYou *must* format your evaluation rating as a JSON object with the following fields (no markdown). Pay close attention to the field type of the evaluation rating (string, boolean, numeric, etc.), and ensure that it conforms to the instructions.\n\n- result (str): {RESULT_DESCRIPTION}\n- rationale (str): {RATIONALE_DESCRIPTION}"
        );
        let user_content = field_order
            .iter()
            .filter_map(|field| match *field {
                "inputs" => item.inputs.as_ref().map(|value| (field, value)),
                "outputs" => item.outputs.as_ref().map(|value| (field, value)),
                _ => None,
            })
            .map(|(field, value)| format!("{field}: {}", pretty_json(value)))
            .collect::<Vec<_>>()
            .join("\n");
        let response_format = response_format(&payload.pydantic_data)?;
        let request = json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "response_format": response_format
        });

        let response = self
            .client
            .post(gateway_url.ok_or(EngineError::MissingGatewayUrl)?)
            .json(&request)
            .send()
            .await
            .map_err(|error| EngineError::Gateway(error.to_string()))?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .map_err(|error| EngineError::Gateway(error.to_string()))?;
        if !status.is_success() {
            return Err(EngineError::Gateway(format!("HTTP {status}: {body}")));
        }

        let completion = body
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| EngineError::MalformedGatewayResponse(body.to_string()))?;
        let completion: Value = serde_json::from_str(completion)
            .map_err(|error| EngineError::MalformedGatewayResponse(error.to_string()))?;
        let value = completion
            .get("result")
            .cloned()
            .ok_or(EngineError::InvalidScorerField("result"))?;
        let rationale = completion
            .get("rationale")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let mut metadata = BTreeMap::new();
        if let Some(tokens) = body.pointer("/usage/prompt_tokens").cloned() {
            metadata.insert("mlflow.assessment.judgeInputTokens".to_string(), tokens);
        }
        if let Some(tokens) = body.pointer("/usage/completion_tokens").cloned() {
            metadata.insert("mlflow.assessment.judgeOutputTokens".to_string(), tokens);
        }
        metadata.insert(
            "guideline".to_string(),
            Value::String(instructions.to_string()),
        );

        Ok(Feedback {
            name: payload.common.name.clone(),
            value,
            rationale,
            source: Some(AssessmentSource {
                source_type: "LLM_JUDGE".to_string(),
                source_id: model_uri.to_string(),
            }),
            metadata: Some(metadata),
        })
    }
}

fn instruction_field_order(instructions: &str) -> Vec<&'static str> {
    let mut fields = [("inputs", None), ("outputs", None)];
    for (field, position) in &mut fields {
        *position = instructions.match_indices("{{").find_map(|(start, _)| {
            let tail = &instructions[start + 2..];
            let end = tail.find("}}")?;
            (tail[..end].trim() == *field).then_some(start)
        });
    }
    let mut fields = fields
        .into_iter()
        .filter_map(|(field, position)| position.map(|position| (position, field)))
        .collect::<Vec<_>>();
    fields.sort_unstable_by_key(|(position, _)| *position);
    let fields = fields
        .into_iter()
        .map(|(_, field)| field)
        .collect::<Vec<_>>();
    if fields.is_empty() {
        vec!["outputs"]
    } else {
        fields
    }
}

fn pretty_json(value: &Value) -> String {
    let mut buffer = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"  ");
    let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, formatter);
    value
        .serialize(&mut serializer)
        .expect("serde_json::Value serialization cannot fail");
    String::from_utf8(buffer).expect("JSON serialization is UTF-8")
}

impl Default for ScorerExecutor {
    fn default() -> Self {
        Self::new()
    }
}

fn execute_builtin(
    payload: &crate::BuiltinScorerPayload,
    item: &EvalItem,
) -> Result<Feedback, EngineError> {
    if payload.class_name != "ResponseLength" {
        return Err(EngineError::UnsupportedBuiltin(payload.class_name.clone()));
    }
    let output = item
        .outputs
        .as_ref()
        .and_then(Value::as_str)
        .ok_or(EngineError::InvalidScorerField("outputs"))?;
    let unit = required_str(&payload.pydantic_data, "unit")?;
    let length = match unit {
        "words" => output.split_whitespace().count(),
        "chars" => output.chars().count(),
        _ => return Err(EngineError::InvalidScorerField("unit")),
    };
    let min_length = optional_usize(&payload.pydantic_data, "min_length")?;
    let max_length = optional_usize(&payload.pydantic_data, "max_length")?;

    let (value, rationale) = if min_length.is_some_and(|minimum| length < minimum) {
        let minimum = min_length.expect("checked above");
        (
            "no",
            format!("Output length ({length} {unit}) is below the minimum ({minimum} {unit})"),
        )
    } else if max_length.is_some_and(|maximum| length > maximum) {
        let maximum = max_length.expect("checked above");
        (
            "no",
            format!("Output length ({length} {unit}) exceeds the maximum ({maximum} {unit})"),
        )
    } else {
        (
            "yes",
            format!("Output length ({length} {unit}) is within bounds"),
        )
    };

    Ok(Feedback {
        name: payload.common.name.clone(),
        value: Value::String(value.to_string()),
        rationale,
        source: None,
        metadata: None,
    })
}

fn required_str<'a>(
    data: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, EngineError> {
    data.get(field)
        .and_then(Value::as_str)
        .ok_or(EngineError::InvalidScorerField(field))
}

fn optional_usize(
    data: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<usize>, EngineError> {
    match data.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or(EngineError::InvalidScorerField(field)),
    }
}

fn response_format(data: &Map<String, Value>) -> Result<Value, EngineError> {
    let mut result_schema = data
        .get("feedback_value_type")
        .and_then(Value::as_object)
        .cloned()
        .ok_or(EngineError::InvalidScorerField("feedback_value_type"))?;
    result_schema.insert(
        "description".to_string(),
        Value::String(RESULT_DESCRIPTION.to_string()),
    );
    let rationale_schema = json!({
        "description": RATIONALE_DESCRIPTION,
        "title": "Rationale",
        "type": "string"
    });
    Ok(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "ResponseFormat",
            "schema": {
                "properties": {
                    "result": Value::Object(result_schema),
                    "rationale": rationale_schema
                },
                "required": ["result", "rationale"],
                "title": "ResponseFormat",
                "type": "object",
                "additionalProperties": false
            },
            "strict": true
        }
    }))
}
