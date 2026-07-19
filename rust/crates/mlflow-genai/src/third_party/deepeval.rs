use fancy_regex::Regex;
use serde_json::{json, Map, Value};

use super::{
    feedback, kwargs, map_single_turn, metric_name, workflow, FeedbackContext, ThirdPartyFamily,
    ThirdPartyMetric,
};
use crate::{EngineError, EvalItem, Feedback, ScorerExecutor, SerializedScorerCommon};

const METRICS: [&str; 44] = [
    "AnswerRelevancy",
    "ArgumentCorrectness",
    "Bias",
    "ContextualPrecision",
    "ContextualRecall",
    "ContextualRelevancy",
    "ConversationCompleteness",
    "ConversationalDAG",
    "DAG",
    "ExactMatch",
    "Faithfulness",
    "GoalAccuracy",
    "Hallucination",
    "ImageCoherence",
    "ImageEditing",
    "ImageHelpfulness",
    "ImageReference",
    "JsonCorrectness",
    "KnowledgeRetention",
    "MCPTaskCompletion",
    "MCPUse",
    "Misuse",
    "MultiTurnMCPUse",
    "NonAdvice",
    "PIILeakage",
    "PatternMatch",
    "PlanAdherence",
    "PlanQuality",
    "PromptAlignment",
    "RoleAdherence",
    "RoleViolation",
    "StepEfficiency",
    "Summarization",
    "TaskCompletion",
    "TextToImage",
    "ToolCorrectness",
    "ToolUse",
    "TopicAdherence",
    "Toxicity",
    "TurnContextualPrecision",
    "TurnContextualRecall",
    "TurnContextualRelevancy",
    "TurnFaithfulness",
    "TurnRelevancy",
];

const CONFIGURED: [&str; 30] = [
    "AnswerRelevancy",
    "ArgumentCorrectness",
    "Bias",
    "ContextualPrecision",
    "ContextualRecall",
    "ContextualRelevancy",
    "ConversationCompleteness",
    "ExactMatch",
    "Faithfulness",
    "GoalAccuracy",
    "Hallucination",
    "JsonCorrectness",
    "KnowledgeRetention",
    "Misuse",
    "NonAdvice",
    "PIILeakage",
    "PatternMatch",
    "PlanAdherence",
    "PlanQuality",
    "PromptAlignment",
    "RoleAdherence",
    "RoleViolation",
    "StepEfficiency",
    "Summarization",
    "TaskCompletion",
    "ToolCorrectness",
    "ToolUse",
    "TopicAdherence",
    "Toxicity",
    "TurnRelevancy",
];

const SESSION_METRICS: [&str; 13] = [
    "ConversationCompleteness",
    "ConversationalDAG",
    "GoalAccuracy",
    "KnowledgeRetention",
    "MCPTaskCompletion",
    "MultiTurnMCPUse",
    "RoleAdherence",
    "ToolUse",
    "TopicAdherence",
    "TurnContextualPrecision",
    "TurnContextualRecall",
    "TurnContextualRelevancy",
    "TurnRelevancy",
];

pub(super) fn metrics() -> impl Iterator<Item = ThirdPartyMetric> {
    METRICS.into_iter().map(|name| ThirdPartyMetric {
        family: ThirdPartyFamily::DeepEval,
        name,
        deterministic: matches!(name, "ExactMatch" | "PatternMatch"),
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
        let mut available = CONFIGURED;
        available.sort_unstable();
        return Err(EngineError::ThirdParty(format!(
            "Unknown metric: '{name}'. Could not import '{name}Metric' from 'deepeval.metrics'. Available pre-configured metrics: {}",
            available.join(", ")
        )));
    }
    let metric_kwargs = kwargs(data);
    match name {
        "ExactMatch" => Ok(vec![exact_match(common, item, &metric_kwargs)?]),
        "PatternMatch" => Ok(vec![pattern_match(common, item, &metric_kwargs)?]),
        _ => {
            validate_constructor(name, &metric_kwargs)?;
            if SESSION_METRICS.contains(&name) && item.session.is_none() {
                return Err(EngineError::ThirdParty(format!(
                    "Multi-turn scorer '{}' requires 'session' parameter containing a list of traces from the conversation.",
                    common.name
                )));
            }
            workflow::execute(executor, "deepeval", common, data, item, gateway_url, None)
                .await
                .map(|feedback| vec![feedback])
        }
    }
}

fn exact_match(
    common: &SerializedScorerCommon,
    item: &EvalItem,
    metric_kwargs: &Map<String, Value>,
) -> Result<Feedback, EngineError> {
    let mapped = map_single_turn(item);
    let expected = mapped.reference.ok_or_else(|| {
        EngineError::ThirdParty("'expected_output' cannot be None for metric 'Exact Match'".into())
    })?;
    let score = if expected.trim() == mapped.output.trim() {
        1.0
    } else {
        0.0
    };
    let threshold = number(metric_kwargs, "threshold").unwrap_or(1.0);
    let rationale = if score == 1.0 {
        "The actual and expected outputs are exact matches."
    } else {
        "The actual and expected outputs are different."
    };
    Ok(feedback(
        common,
        json!(if score >= threshold { "yes" } else { "no" }),
        rationale.to_string(),
        FeedbackContext {
            source_type: "CODE",
            source_id: None,
            family: "deepeval",
            score: Some(score),
            threshold: Some(threshold),
        },
    ))
}

fn pattern_match(
    common: &SerializedScorerCommon,
    item: &EvalItem,
    metric_kwargs: &Map<String, Value>,
) -> Result<Feedback, EngineError> {
    let pattern = metric_kwargs
        .get("pattern")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            EngineError::ThirdParty(
                "PatternMatchMetric.__init__() missing 1 required positional argument: 'pattern'"
                    .into(),
            )
        })?
        .trim();
    let body = if metric_kwargs
        .get("ignore_case")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        format!("(?i:{pattern})")
    } else {
        pattern.to_string()
    };
    let regex = Regex::new(&format!(r"\A(?:{body})\z")).map_err(|error| {
        EngineError::ThirdParty(format!("Invalid regex pattern: {pattern} — {error}"))
    })?;
    let mapped = map_single_turn(item);
    let score = if regex
        .is_match(mapped.output.trim())
        .map_err(|error| EngineError::ThirdParty(error.to_string()))?
    {
        1.0
    } else {
        0.0
    };
    let threshold = number(metric_kwargs, "threshold").unwrap_or(1.0);
    let rationale = if score == 1.0 {
        "The actual output fully matches the pattern."
    } else {
        "The actual output does not match the pattern."
    };
    Ok(feedback(
        common,
        json!(if score >= threshold { "yes" } else { "no" }),
        rationale.to_string(),
        FeedbackContext {
            source_type: "CODE",
            source_id: None,
            family: "deepeval",
            score: Some(score),
            threshold: Some(threshold),
        },
    ))
}

fn validate_constructor(name: &str, kwargs: &Map<String, Value>) -> Result<(), EngineError> {
    let required: &[&str] = match name {
        "ConversationalDAG" | "DAG" => &["name", "dag"],
        "JsonCorrectness" => &["expected_schema"],
        "Misuse" => &["domain"],
        "NonAdvice" => &["advice_types"],
        "PromptAlignment" => &["prompt_instructions"],
        "RoleViolation" => &["role"],
        "ToolUse" => &["available_tools"],
        "TopicAdherence" => &["relevant_topics"],
        _ => &[],
    };
    let missing = required
        .iter()
        .copied()
        .filter(|field| !kwargs.contains_key(*field))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    let class = format!("{name}Metric");
    if missing.len() == 1 {
        Err(EngineError::ThirdParty(format!(
            "{class}.__init__() missing 1 required positional argument: '{}'",
            missing[0]
        )))
    } else {
        Err(EngineError::ThirdParty(format!(
            "{class}.__init__() missing {} required positional arguments: '{}'",
            missing.len(),
            missing.join("' and '")
        )))
    }
}

fn number(values: &Map<String, Value>, key: &str) -> Option<f64> {
    values.get(key).and_then(Value::as_f64)
}
