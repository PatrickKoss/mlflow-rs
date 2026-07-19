use serde_json::{json, Map, Value};

use super::{
    feedback, invoke_messages, kwargs, map_single_turn, metric_name, model, parse_score_reason,
    FeedbackContext, ThirdPartyFamily, ThirdPartyMetric,
};
use crate::{EngineError, EvalItem, Feedback, ScorerExecutor, SerializedScorerCommon};

const METRICS: [&str; 25] = [
    "Coherence",
    "Comprehensiveness",
    "Conciseness",
    "ContextRelevance",
    "Controversiality",
    "Correctness",
    "Criminality",
    "ExecutionEfficiency",
    "Groundedness",
    "Harmfulness",
    "Helpfulness",
    "Insensitivity",
    "LogicalConsistency",
    "Maliciousness",
    "Misogyny",
    "PlanAdherence",
    "PlanQuality",
    "QsRelevance",
    "Relevance",
    "Sentiment",
    "Stereotypes",
    "Summarization",
    "ToolCalling",
    "ToolQuality",
    "ToolSelection",
];

pub(super) fn metrics() -> impl Iterator<Item = ThirdPartyMetric> {
    METRICS.into_iter().map(|name| ThirdPartyMetric {
        family: ThirdPartyFamily::TruLens,
        name,
        deterministic: false,
    })
}

pub(super) async fn execute(
    executor: &ScorerExecutor,
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let name = metric_name(common, data)?;
    if !METRICS.contains(&name) {
        let method = method_name(name);
        return Err(EngineError::ThirdParty(format!(
            "'GatewayProvider' object has no attribute '{method}'"
        )));
    }
    // MLflow's pinned generic wrapper only maps arguments for these three
    // manifest spellings. The other dynamic names reach getattr/call with no
    // arguments in Python; retain that observable failure behavior.
    if !matches!(name, "Coherence" | "ContextRelevance" | "Groundedness") {
        return Err(EngineError::ThirdParty(dynamic_call_error(name)));
    }
    llm_metric(executor, common, data, item, gateway_url)
        .await
        .map(|value| vec![value])
}

async fn llm_metric(
    executor: &ScorerExecutor,
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Feedback, EngineError> {
    let name = metric_name(common, data)?;
    let mapped = map_single_turn(item);
    let context = context(item, &mapped.contexts);
    let (criteria, user) = match name {
        "Coherence" => (
            "coherence and logical flow",
            format!("SUBMISSION:\n{}", mapped.output),
        ),
        "ContextRelevance" => (
            "the relevance of the context to the question",
            format!(
                "QUESTION: {}\nCONTEXT: {}\nRELEVANCE:",
                mapped.input, context
            ),
        ),
        "Groundedness" => (
            "whether each statement is supported by the source",
            format!("SOURCE: {context}\nSTATEMENT: {}", mapped.output),
        ),
        _ => unreachable!("mapped TruLens metric set"),
    };
    let system = format!(
        "You are a judge evaluating {criteria}. Give a score from 0 to 3. First provide the criteria used and supporting evidence, then the score."
    );
    let response_format = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "ChainOfThoughtResponse",
            "schema": {
                "properties": {
                    "criteria": {"type":"string"},
                    "supporting_evidence": {"type":"string"},
                    "score": {"type":"integer"}
                },
                "required": ["criteria", "supporting_evidence", "score"],
                "type": "object",
                "additionalProperties": false
            },
            "strict": true
        }
    });
    let mut inference = Map::new();
    inference.insert("temperature".to_string(), json!(0.0));
    let content = invoke_messages(
        executor,
        model(data),
        vec![
            json!({"role":"system", "content":system}),
            json!({"role":"user", "content":user}),
        ],
        Some(&inference),
        Some(response_format),
        gateway_url,
    )
    .await?;
    let (raw_score, mut rationale) = parse_score_reason(&content)?;
    if let Ok(parsed) = serde_json::from_str::<Value>(content.trim()) {
        if let (Some(criteria), Some(evidence)) = (
            parsed.get("criteria").and_then(Value::as_str),
            parsed.get("supporting_evidence").and_then(Value::as_str),
        ) {
            rationale = format!("reason: Criteria: {criteria}\nSupporting Evidence: {evidence}");
        }
    }
    let score = raw_score / 3.0;
    let threshold = kwargs(data)
        .get("threshold")
        .and_then(Value::as_f64)
        .unwrap_or(0.5);
    Ok(feedback(
        common,
        json!(if score >= threshold { "yes" } else { "no" }),
        rationale,
        FeedbackContext {
            source_type: "LLM_JUDGE",
            source_id: Some(model(data).to_string()),
            family: "trulens",
            score: Some(score),
            threshold: Some(threshold),
        },
    ))
}

fn context(item: &EvalItem, retrieval_contexts: &[String]) -> String {
    if let Some(expectations) = item.expectations.as_ref().and_then(Value::as_object) {
        for key in ["context", "reference", "expected_output"] {
            if let Some(value) = expectations.get(key).filter(|value| !value.is_null()) {
                return match value {
                    Value::Array(values) => values
                        .iter()
                        .map(|value| {
                            value
                                .as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| value.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Value::String(value) => value.clone(),
                    value => value.to_string(),
                };
            }
        }
    }
    retrieval_contexts.join("\n")
}

fn method_name(name: &str) -> String {
    let mut snake = String::new();
    for (index, character) in name.chars().enumerate() {
        if index > 0 && character.is_ascii_uppercase() {
            snake.push('_');
        }
        snake.push(character.to_ascii_lowercase());
    }
    format!("{snake}_with_cot_reasons")
}

fn dynamic_call_error(name: &str) -> String {
    let required = match name {
        "Comprehensiveness" | "Summarization" => "source, summary",
        "Conciseness" | "Controversiality" | "Correctness" | "Criminality" | "Harmfulness"
        | "Helpfulness" | "Insensitivity" | "Maliciousness" | "Misogyny" | "Sentiment"
        | "Stereotypes" => "text",
        "QsRelevance" | "Relevance" => "prompt, response",
        "ExecutionEfficiency"
        | "LogicalConsistency"
        | "PlanAdherence"
        | "PlanQuality"
        | "ToolCalling"
        | "ToolQuality"
        | "ToolSelection" => "trace",
        _ => "input",
    };
    let count = required.split(", ").count();
    if count == 1 {
        format!(
            "LLMProvider.{}() missing 1 required positional argument: '{required}'",
            method_name(name)
        )
    } else {
        format!(
            "LLMProvider.{}() missing {count} required positional arguments: '{}'",
            method_name(name),
            required.replace(", ", "' and '")
        )
    }
}
