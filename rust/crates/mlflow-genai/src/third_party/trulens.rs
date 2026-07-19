use serde_json::{Map, Value};

use super::{metric_name, workflow, ThirdPartyFamily, ThirdPartyMetric};
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
    workflow::execute(executor, "trulens", common, data, item, gateway_url, None)
        .await
        .map(|value| vec![value])
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
