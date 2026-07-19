//! Native compatibility layer for the pinned third-party scorer families.
//!
//! The family modules intentionally consume `third_party_scorer_data` rather
//! than introducing another serialized scorer representation. This preserves
//! the T19.1 wire contract while keeping upstream-specific behavior isolated.

mod deepeval;
mod ragas;
mod trulens;

use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::{
    trace::{parse_inputs_to_str, parse_outputs_to_str, python_str, TraceView},
    AssessmentSource, EngineError, EvalItem, Feedback, ScorerExecutor, SerializedScorerCommon,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThirdPartyFamily {
    DeepEval,
    Ragas,
    TruLens,
    Phoenix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ThirdPartyMetric {
    pub family: ThirdPartyFamily,
    pub name: &'static str,
    pub deterministic: bool,
}

pub fn supported_third_party_metrics() -> Vec<ThirdPartyMetric> {
    let mut metrics = Vec::with_capacity(112);
    metrics.extend(deepeval::metrics());
    metrics.extend(ragas::metrics());
    metrics.extend(trulens::metrics());
    for name in [
        "Hallucination",
        "QA",
        "Relevance",
        "SQL",
        "Summarization",
        "Toxicity",
    ] {
        metrics.push(ThirdPartyMetric {
            family: ThirdPartyFamily::Phoenix,
            name,
            deterministic: false,
        });
    }
    metrics
}

pub(crate) async fn execute(
    executor: &ScorerExecutor,
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
    embedding_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let module = data.get("module").and_then(Value::as_str).unwrap_or("");
    if module == "mlflow.genai.scorers.deepeval"
        || module.starts_with("mlflow.genai.scorers.deepeval.")
    {
        deepeval::execute(executor, common, data, item, gateway_url).await
    } else if module == "mlflow.genai.scorers.ragas"
        || module.starts_with("mlflow.genai.scorers.ragas.")
    {
        ragas::execute(executor, common, data, item, gateway_url, embedding_url).await
    } else if module == "mlflow.genai.scorers.trulens"
        || module.starts_with("mlflow.genai.scorers.trulens.")
    {
        trulens::execute(executor, common, data, item, gateway_url).await
    } else {
        Err(EngineError::ThirdParty(format!(
            "Third-party scorer '{}': module '{}' is not in the allow-list ['mlflow.genai.scorers.deepeval', 'mlflow.genai.scorers.phoenix', 'mlflow.genai.scorers.ragas', 'mlflow.genai.scorers.trulens'].",
            common.name, module
        )))
    }
}

pub(super) fn metric_name<'a>(
    common: &SerializedScorerCommon,
    data: &'a Map<String, Value>,
) -> Result<&'a str, EngineError> {
    data.get("metric_name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            EngineError::ThirdParty(format!(
                "Third-party scorer '{}': missing required fields in third_party_scorer_data (class, metric_name).",
                common.name
            ))
        })
}

pub(super) fn kwargs(data: &Map<String, Value>) -> Map<String, Value> {
    data.get("kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

pub(super) fn model(data: &Map<String, Value>) -> &str {
    data.get("model")
        .and_then(Value::as_str)
        .unwrap_or("databricks")
}

#[derive(Debug, Clone)]
pub(super) struct MappedItem {
    pub input: String,
    pub output: String,
    pub reference: Option<String>,
    pub contexts: Vec<String>,
}

pub(super) fn map_single_turn(item: &EvalItem) -> MappedItem {
    let view = item.trace.as_ref().map(TraceView::new);
    let inputs = item
        .inputs
        .clone()
        .or_else(|| view.as_ref().and_then(TraceView::inputs));
    let outputs = item
        .outputs
        .clone()
        .or_else(|| view.as_ref().and_then(TraceView::outputs));
    let reference = item.expectations.as_ref().and_then(|expectations| {
        let object = expectations.as_object()?;
        if let Some(value) = object.get("expected_output") {
            Some(parse_outputs_to_str(value))
        } else {
            let values = object
                .iter()
                .filter(|(key, _)| key.as_str() != "rubrics")
                .map(|(_, value)| python_scalar(value))
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| values.join(", "))
        }
    });
    let contexts = view
        .as_ref()
        .map(|view| {
            view.retrieval_contexts()
                .into_iter()
                .flat_map(|(_, contexts)| contexts)
                .map(|context| python_str(&context))
                .collect()
        })
        .unwrap_or_default();
    MappedItem {
        input: inputs.as_ref().map(parse_inputs_to_str).unwrap_or_default(),
        output: outputs
            .as_ref()
            .map(parse_outputs_to_str)
            .unwrap_or_default(),
        reference,
        contexts,
    }
}

fn python_scalar(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        other => python_str(other),
    }
}

pub(super) async fn invoke_messages(
    executor: &ScorerExecutor,
    model_uri: &str,
    messages: Vec<Value>,
    inference: Option<&Map<String, Value>>,
    response_format: Option<Value>,
    gateway_url: Option<&str>,
) -> Result<String, EngineError> {
    let model = model_uri
        .split_once(":/")
        .map(|(_, model)| model)
        .filter(|model| !model.is_empty())
        .unwrap_or(model_uri);
    let mut request = Map::new();
    request.insert("model".to_string(), Value::String(model.to_string()));
    request.insert("messages".to_string(), Value::Array(messages));
    if let Some(inference) = inference {
        request.extend(inference.clone());
    }
    if let Some(response_format) = response_format {
        request.insert("response_format".to_string(), response_format);
    }
    let response = executor
        .client()
        .post(gateway_url.ok_or(EngineError::MissingGatewayUrl)?)
        .json(&Value::Object(request))
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
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            EngineError::MalformedGatewayResponse("missing choices[0].message.content".to_string())
        })
}

pub(super) fn parse_score_reason(content: &str) -> Result<(f64, String), EngineError> {
    let cleaned = content
        .trim()
        .strip_prefix("```json")
        .or_else(|| content.trim().strip_prefix("```"))
        .unwrap_or(content.trim())
        .strip_suffix("```")
        .unwrap_or(content.trim())
        .trim();
    if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
        let score = ["score", "value", "verdict"]
            .into_iter()
            .find_map(|key| value.get(key))
            .and_then(json_number)
            .ok_or_else(|| EngineError::MalformedGatewayResponse("missing numeric score".into()))?;
        let reason = ["reason", "rationale", "feedback"]
            .into_iter()
            .find_map(|key| value.get(key).and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        return Ok((score, reason));
    }
    let score_regex = regex::Regex::new(r"(?im)^\s*(?:score|rating)\s*:\s*(-?\d+(?:\.\d+)?)")
        .expect("static score regex");
    let score = score_regex
        .captures(cleaned)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<f64>().ok())
        .ok_or_else(|| {
            EngineError::MalformedGatewayResponse("could not parse scorer response".into())
        })?;
    let reason = cleaned
        .lines()
        .filter(|line| !score_regex.is_match(line))
        .collect::<Vec<_>>()
        .join("\n");
    Ok((score, reason))
}

fn json_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        .or_else(|| value.as_bool().map(|value| if value { 1.0 } else { 0.0 }))
}

pub(super) struct FeedbackContext<'a> {
    pub source_type: &'a str,
    pub source_id: Option<String>,
    pub family: &'a str,
    pub score: Option<f64>,
    pub threshold: Option<f64>,
}

pub(super) fn feedback(
    common: &SerializedScorerCommon,
    value: Value,
    rationale: String,
    context: FeedbackContext<'_>,
) -> Feedback {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "mlflow.scorer.framework".to_string(),
        Value::String(context.family.to_string()),
    );
    if let Some(score) = context.score {
        metadata.insert("score".to_string(), json!(score));
    }
    if let Some(threshold) = context.threshold {
        metadata.insert("threshold".to_string(), json!(threshold));
    }
    Feedback {
        name: common.name.clone(),
        value,
        rationale,
        source: Some(AssessmentSource {
            source_type: context.source_type.to_string(),
            source_id: context.source_id,
        }),
        metadata: Some(metadata),
        span_id: None,
        trace_id: None,
    }
}
