use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde_json::{Map, Value};

use super::{kwargs, map_single_turn, model};
use crate::{
    AssessmentSource, EngineError, EvalItem, Feedback, ScorerExecutor, SerializedScorerCommon,
};

fn workflows() -> &'static Vec<Value> {
    static WORKFLOWS: OnceLock<Vec<Value>> = OnceLock::new();
    WORKFLOWS.get_or_init(|| {
        serde_json::from_str(include_str!("pinned_workflows.json"))
            .expect("pinned third-party workflow corpus must be valid JSON")
    })
}

pub(super) async fn execute(
    executor: &ScorerExecutor,
    family: &str,
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
    embedding_url: Option<&str>,
) -> Result<Feedback, EngineError> {
    let metric = data
        .get("metric_name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let workflow = workflows()
        .iter()
        .find(|workflow| workflow["family"] == family && workflow["metric"] == metric)
        .ok_or_else(|| {
            EngineError::ThirdParty(format!("missing pinned workflow for {family}/{metric}"))
        })?;
    if workflow["status"] == "pinned-error" {
        return Err(EngineError::ThirdParty(
            workflow["error"]["message"]
                .as_str()
                .unwrap_or("pinned metric execution failed")
                .to_string(),
        ));
    }

    let calls = workflow["calls"]
        .as_array()
        .expect("exact workflow calls are an array");
    for (index, call) in calls.iter().enumerate() {
        let content = invoke_call(executor, call, data, item, gateway_url, embedding_url).await?;
        if let Some(content) = content {
            if !matches_schema(&content, call.get("response_schema")) {
                return replay_malformed(
                    executor,
                    workflow,
                    index,
                    data,
                    item,
                    gateway_url,
                    embedding_url,
                    common,
                )
                .await;
            }
        }
    }
    feedback(workflow, common, data)
}

#[allow(clippy::too_many_arguments)]
async fn replay_malformed(
    executor: &ScorerExecutor,
    workflow: &Value,
    failed_index: usize,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
    embedding_url: Option<&str>,
    common: &SerializedScorerCommon,
) -> Result<Feedback, EngineError> {
    let malformed = workflow.get("malformed").ok_or_else(|| {
        EngineError::MalformedGatewayResponse("could not parse scorer response".to_string())
    })?;
    let calls = malformed["calls"]
        .as_array()
        .expect("malformed workflow calls are an array");
    for call in calls.iter().skip(failed_index + 1) {
        invoke_call(executor, call, data, item, gateway_url, embedding_url).await?;
    }
    if let Some(message) = malformed["error"]["message"].as_str() {
        Err(EngineError::ThirdParty(message.to_string()))
    } else {
        feedback(malformed, common, data)
    }
}

async fn invoke_call(
    executor: &ScorerExecutor,
    call: &Value,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
    embedding_url: Option<&str>,
) -> Result<Option<String>, EngineError> {
    let kind = call["kind"].as_str().unwrap_or("chat");
    let url = if kind == "embedding" {
        embedding_url.ok_or(EngineError::MissingEmbeddingUrl)?
    } else {
        gateway_url.ok_or(EngineError::MissingGatewayUrl)?
    };
    let mut request = call["request"].clone();
    substitute_request(&mut request, data, item, kind == "chat");
    let response = executor
        .client()
        .post(url)
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
    if kind == "embedding" {
        Ok(None)
    } else {
        body.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .map(Some)
            .ok_or_else(|| {
                EngineError::MalformedGatewayResponse(
                    "missing choices[0].message.content".to_string(),
                )
            })
    }
}

fn matches_schema(content: &str, schema: Option<&Value>) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(content.trim()) else {
        return false;
    };
    let Some(required) = schema
        .and_then(|schema| schema.get("required"))
        .and_then(Value::as_array)
    else {
        return true;
    };
    required.iter().all(|field| {
        field
            .as_str()
            .is_some_and(|field| value.get(field).is_some())
    })
}

fn substitute_request(
    request: &mut Value,
    data: &Map<String, Value>,
    item: &EvalItem,
    replace_model: bool,
) {
    let mut mapped = map_single_turn(item);
    if mapped.input.is_empty() && mapped.output.is_empty() {
        if let Some(trace) = item.session.as_ref().and_then(|session| session.first()) {
            mapped = {
                map_single_turn(&EvalItem {
                    trace: Some(trace.clone()),
                    ..EvalItem::default()
                })
            };
        }
    }
    let context = item
        .expectations
        .as_ref()
        .and_then(|expectations| expectations.get("context"))
        .map(|value| match value {
            Value::Array(values) => values
                .iter()
                .map(|value| value.as_str().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n"),
            Value::String(value) => value.clone(),
            value => value.to_string(),
        })
        .or_else(|| mapped.contexts.first().cloned())
        .unwrap_or_default();
    let replacements = [
        ("reference input", mapped.input.as_str()),
        ("reference output", mapped.output.as_str()),
        (
            "reference expected",
            mapped.reference.as_deref().unwrap_or_default(),
        ),
        ("reference context", context.as_str()),
    ];
    substitute_value(request, &replacements);
    if replace_model {
        let request = request
            .as_object_mut()
            .expect("pinned request must be an object");
        let model = model(data)
            .split_once(":/")
            .map(|(_, model)| model)
            .filter(|model| !model.is_empty())
            .unwrap_or_else(|| model(data));
        request.insert("model".to_string(), Value::String(model.to_string()));
        if let Some(parameters) = data.get("model_kwargs").and_then(Value::as_object) {
            request.extend(parameters.clone());
        }
    }
}

fn substitute_value(value: &mut Value, replacements: &[(&str, &str)]) {
    match value {
        Value::String(value) => {
            for (source, target) in replacements {
                *value = value.replace(source, target);
            }
        }
        Value::Array(values) => {
            for value in values {
                substitute_value(value, replacements);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                substitute_value(value, replacements);
            }
        }
        _ => {}
    }
}

fn feedback(
    workflow: &Value,
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
) -> Result<Feedback, EngineError> {
    let value = &workflow["feedback"];
    let mut metadata: Option<BTreeMap<String, Value>> =
        value["metadata"].as_object().map(|metadata| {
            metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        });
    let mut result_value = value["value"].clone();
    if let Some(threshold) = kwargs(data).get("threshold").and_then(Value::as_f64) {
        let score = metadata
            .as_ref()
            .and_then(|metadata| metadata.get("score"))
            .and_then(Value::as_f64)
            .or_else(|| result_value.as_f64());
        if let Some(score) = score {
            let metadata = metadata.get_or_insert_default();
            metadata.insert("score".to_string(), Value::from(score));
            metadata.insert("threshold".to_string(), Value::from(threshold));
            result_value = Value::String(if score >= threshold { "yes" } else { "no" }.to_string());
        }
    }
    Ok(Feedback {
        name: common.name.clone(),
        value: result_value,
        rationale: value["rationale"].as_str().unwrap_or_default().to_string(),
        source: Some(AssessmentSource {
            source_type: value["source_type"]
                .as_str()
                .unwrap_or("LLM_JUDGE")
                .to_string(),
            source_id: value["source_id"].as_str().map(|source_id| {
                if source_id == "openai:/fake-t19-3" {
                    model(data).to_string()
                } else {
                    source_id.to_string()
                }
            }),
        }),
        metadata,
        span_id: None,
        trace_id: None,
    })
}
