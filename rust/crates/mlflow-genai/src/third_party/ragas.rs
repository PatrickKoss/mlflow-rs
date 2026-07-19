use std::collections::{BTreeMap, BTreeSet};

use regex::Regex;
use serde_json::{json, Map, Value};

use super::{
    feedback, kwargs, map_single_turn, metric_name, model, workflow, FeedbackContext,
    ThirdPartyFamily, ThirdPartyMetric,
};
use crate::{
    trace::{python_str, TraceView},
    EngineError, EvalItem, Feedback, ScorerExecutor, SerializedScorerCommon,
};

const METRICS: [&str; 37] = [
    "AgentGoalAccuracy",
    "AgentGoalAccuracyWithReference",
    "AgentGoalAccuracyWithoutReference",
    "AnswerAccuracy",
    "AnswerCorrectness",
    "AnswerRelevancy",
    "BleuScore",
    "CHRFScore",
    "ContextEntityRecall",
    "ContextPrecision",
    "ContextPrecisionWithReference",
    "ContextPrecisionWithoutReference",
    "ContextRecall",
    "ContextRelevance",
    "ContextUtilization",
    "DataCompyScore",
    "DomainSpecificRubrics",
    "ExactMatch",
    "FactualCorrectness",
    "Faithfulness",
    "InstanceSpecificRubrics",
    "MultiModalFaithfulness",
    "MultiModalRelevance",
    "NoiseSensitivity",
    "NonLLMStringSimilarity",
    "QuotedSpansAlignment",
    "ResponseGroundedness",
    "RougeScore",
    "RubricsScoreWithReference",
    "RubricsScoreWithoutReference",
    "SQLSemanticEquivalence",
    "SemanticSimilarity",
    "StringPresence",
    "SummaryScore",
    "ToolCallAccuracy",
    "ToolCallF1",
    "TopicAdherence",
];

const DETERMINISTIC: [&str; 10] = [
    "BleuScore",
    "CHRFScore",
    "DataCompyScore",
    "ExactMatch",
    "NonLLMStringSimilarity",
    "QuotedSpansAlignment",
    "RougeScore",
    "StringPresence",
    "ToolCallAccuracy",
    "ToolCallF1",
];

pub(super) fn metrics() -> impl Iterator<Item = ThirdPartyMetric> {
    METRICS.into_iter().map(|name| ThirdPartyMetric {
        family: ThirdPartyFamily::Ragas,
        name,
        deterministic: DETERMINISTIC.contains(&name),
    })
}

pub(super) async fn execute(
    executor: &ScorerExecutor,
    common: &SerializedScorerCommon,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
    embedding_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let name = metric_name(common, data)?;
    if !METRICS.contains(&name) {
        return Err(EngineError::ThirdParty(format!(
            "Unknown RAGAS metric: '{name}'. Could not find class '{name}' in module 'ragas.metrics.collections'."
        )));
    }
    let metric_kwargs = kwargs(data);
    if DETERMINISTIC.contains(&name) {
        return deterministic(common, name, data, &metric_kwargs, item).map(|value| vec![value]);
    }
    workflow::execute(
        executor,
        "ragas",
        common,
        data,
        item,
        gateway_url,
        embedding_url,
    )
    .await
    .map(|value| vec![value])
}

fn deterministic(
    common: &SerializedScorerCommon,
    name: &str,
    data: &Map<String, Value>,
    metric_kwargs: &Map<String, Value>,
    item: &EvalItem,
) -> Result<Feedback, EngineError> {
    let mapped = map_single_turn(item);
    let (score, rationale) = match name {
        "ExactMatch" => (
            bool_score(require_reference(name, &mapped.reference)? == mapped.output),
            String::new(),
        ),
        "StringPresence" => (
            bool_score(
                mapped
                    .output
                    .contains(require_reference(name, &mapped.reference)?),
            ),
            String::new(),
        ),
        "NonLLMStringSimilarity" => (
            string_similarity(
                require_reference(name, &mapped.reference)?,
                &mapped.output,
                metric_kwargs,
            )?,
            String::new(),
        ),
        "BleuScore" => (
            bleu(require_reference(name, &mapped.reference)?, &mapped.output),
            String::new(),
        ),
        "CHRFScore" => (
            chrf(require_reference(name, &mapped.reference)?, &mapped.output),
            String::new(),
        ),
        "RougeScore" => (
            rouge(
                require_reference(name, &mapped.reference)?,
                &mapped.output,
                metric_kwargs,
            ),
            String::new(),
        ),
        "QuotedSpansAlignment" => quoted_spans(&mapped.output, &mapped.contexts, metric_kwargs),
        "ToolCallAccuracy" => (tool_call_accuracy(item, metric_kwargs)?, String::new()),
        "ToolCallF1" => (tool_call_f1(item)?, String::new()),
        "DataCompyScore" => data_compare(
            require_reference(name, &mapped.reference)?,
            &mapped.output,
            metric_kwargs,
        )?,
        _ => unreachable!("closed deterministic metric set"),
    };
    let threshold = metric_kwargs.get("threshold").and_then(Value::as_f64);
    let value = threshold.map_or_else(
        || json!(score),
        |threshold| json!(if score >= threshold { "yes" } else { "no" }),
    );
    let wrapper_treats_as_llm = matches!(
        name,
        "CHRFScore" | "DataCompyScore" | "QuotedSpansAlignment"
    );
    Ok(feedback(
        common,
        value,
        rationale,
        FeedbackContext {
            source_type: if wrapper_treats_as_llm {
                "LLM_JUDGE"
            } else {
                "CODE"
            },
            source_id: Some(if wrapper_treats_as_llm {
                model(data).to_string()
            } else {
                name.to_string()
            }),
            family: "ragas",
            score: threshold.map(|_| score),
            threshold,
        },
    ))
}

fn require_reference<'a>(
    name: &str,
    reference: &'a Option<String>,
) -> Result<&'a str, EngineError> {
    reference.as_deref().ok_or_else(|| {
        EngineError::ThirdParty(format!(
            "RAGAS metric '{name}' requires 'expectations['expected_output']' parameter, which is missing.\nExample: judge(inputs='...', outputs='...', expectations={{'expected_output': ...}}) or log an expectation to the trace: mlflow.log_expectation(trace_id, name='expected_output', value=..., source=...)"
        ))
    })
}

fn bool_score(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn string_similarity(
    reference: &str,
    response: &str,
    metric_kwargs: &Map<String, Value>,
) -> Result<f64, EngineError> {
    let measure = metric_kwargs
        .get("distance_measure")
        .and_then(Value::as_str)
        .unwrap_or("levenshtein")
        .to_ascii_lowercase();
    Ok(match measure.as_str() {
        "levenshtein" => {
            let denominator = reference.chars().count().max(response.chars().count());
            if denominator == 0 {
                1.0
            } else {
                1.0 - strsim::levenshtein(reference, response) as f64 / denominator as f64
            }
        }
        "hamming" => {
            let denominator = reference.chars().count().max(response.chars().count());
            if denominator == 0 {
                1.0
            } else {
                let mismatches = reference
                    .chars()
                    .zip(response.chars())
                    .filter(|(left, right)| left != right)
                    .count()
                    + reference.chars().count().abs_diff(response.chars().count());
                1.0 - mismatches as f64 / denominator as f64
            }
        }
        "jaro" => strsim::jaro(reference, response),
        "jaro_winkler" => strsim::jaro_winkler(reference, response),
        other => {
            return Err(EngineError::ThirdParty(format!(
                "Unsupported distance measure: {other}"
            )))
        }
    })
}

fn tokenize_13a(value: &str) -> Vec<String> {
    let punctuation = Regex::new(r"([^\p{L}\p{N}\s])").expect("static punctuation regex");
    punctuation
        .replace_all(value, " $0 ")
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn bleu(reference: &str, response: &str) -> f64 {
    if reference == response {
        return 1.0000000000000004;
    }
    let reference = tokenize_13a(reference);
    let response = tokenize_13a(response);
    if response.is_empty() {
        return 0.0;
    }
    let mut precisions = Vec::with_capacity(4);
    let mut smooth = 1.0;
    for order in 1..=4 {
        let total = response.len().saturating_sub(order - 1);
        if total == 0 {
            return 0.0;
        }
        let reference_counts = ngram_counts(&reference, order);
        let response_counts = ngram_counts(&response, order);
        let correct: usize = response_counts
            .iter()
            .map(|(ngram, count)| count.min(reference_counts.get(ngram).unwrap_or(&0)))
            .sum();
        if correct == 0 {
            smooth *= 2.0;
            precisions.push(1.0 / (smooth * total as f64));
        } else {
            precisions.push(correct as f64 / total as f64);
        }
    }
    let brevity = if response.len() >= reference.len() {
        1.0
    } else {
        (1.0 - reference.len() as f64 / response.len() as f64).exp()
    };
    brevity * (precisions.iter().map(|value| value.ln()).sum::<f64>() / 4.0).exp()
}

fn ngram_counts(tokens: &[String], order: usize) -> BTreeMap<Vec<String>, usize> {
    let mut counts = BTreeMap::new();
    for values in tokens.windows(order) {
        *counts.entry(values.to_vec()).or_default() += 1;
    }
    counts
}

fn chrf(reference: &str, response: &str) -> f64 {
    let reference = reference
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect::<Vec<_>>();
    let response = response
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect::<Vec<_>>();
    if reference.is_empty() || response.is_empty() {
        return 0.0;
    }
    let mut precision = 0.0;
    let mut recall = 0.0;
    let mut orders = 0.0;
    for order in 1..=6 {
        if reference.len() < order || response.len() < order {
            continue;
        }
        let ref_counts = char_ngram_counts(&reference, order);
        let hyp_counts = char_ngram_counts(&response, order);
        let matches: usize = hyp_counts
            .iter()
            .map(|(gram, count)| count.min(ref_counts.get(gram).unwrap_or(&0)))
            .sum();
        precision += matches as f64 / hyp_counts.values().sum::<usize>() as f64;
        recall += matches as f64 / ref_counts.values().sum::<usize>() as f64;
        orders += 1.0;
    }
    if orders == 0.0 {
        return 0.0;
    }
    precision /= orders;
    recall /= orders;
    if precision + recall == 0.0 {
        0.0
    } else {
        (5.0 * precision * recall) / (4.0 * precision + recall)
    }
}

fn char_ngram_counts(tokens: &[char], order: usize) -> BTreeMap<Vec<char>, usize> {
    let mut counts = BTreeMap::new();
    for values in tokens.windows(order) {
        *counts.entry(values.to_vec()).or_default() += 1;
    }
    counts
}

fn rouge(reference: &str, response: &str, metric_kwargs: &Map<String, Value>) -> f64 {
    let reference = reference.split_whitespace().collect::<Vec<_>>();
    let response = response.split_whitespace().collect::<Vec<_>>();
    let rouge_type = metric_kwargs
        .get("rouge_type")
        .and_then(Value::as_str)
        .unwrap_or("rougeL");
    let matches = if rouge_type == "rouge1" {
        let mut available = reference.to_vec();
        response
            .iter()
            .filter(|token| {
                available
                    .iter()
                    .position(|candidate| candidate == *token)
                    .map(|index| available.remove(index))
                    .is_some()
            })
            .count()
    } else {
        lcs_len(&reference, &response)
    };
    let precision = matches as f64 / response.len().max(1) as f64;
    let recall = matches as f64 / reference.len().max(1) as f64;
    match metric_kwargs
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("fmeasure")
    {
        "precision" => precision,
        "recall" => recall,
        _ if precision + recall == 0.0 => 0.0,
        _ => 2.0 * precision * recall / (precision + recall),
    }
}

fn lcs_len(left: &[&str], right: &[&str]) -> usize {
    let mut row = vec![0; right.len() + 1];
    for left_value in left {
        let mut previous = 0;
        for (index, right_value) in right.iter().enumerate() {
            let saved = row[index + 1];
            row[index + 1] = if left_value == right_value {
                previous + 1
            } else {
                row[index + 1].max(row[index])
            };
            previous = saved;
        }
    }
    row[right.len()]
}

fn quoted_spans(
    response: &str,
    contexts: &[String],
    metric_kwargs: &Map<String, Value>,
) -> (f64, String) {
    let min_words = metric_kwargs
        .get("min_span_words")
        .and_then(Value::as_u64)
        .unwrap_or(3) as usize;
    let pattern = Regex::new(r#"[\"“”„‟'‘’`´](.*?)[\"“”„‟'‘’`´]"#).expect("static quote regex");
    let spans = pattern
        .captures_iter(response)
        .filter_map(|capture| capture.get(1).map(|value| value.as_str().trim()))
        .filter(|span| span.split_whitespace().count() >= min_words)
        .collect::<Vec<_>>();
    if spans.is_empty() {
        return (1.0, "No quoted spans found in response".to_string());
    }
    let casefold = metric_kwargs
        .get("casefold")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let normalize = |value: &str| {
        let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
        if casefold {
            collapsed.to_lowercase()
        } else {
            collapsed
        }
    };
    let joined = normalize(&contexts.join(" "));
    let matched = spans
        .iter()
        .filter(|span| joined.contains(&normalize(span)))
        .count();
    (
        matched as f64 / spans.len() as f64,
        format!("Matched {matched}/{} quoted spans", spans.len()),
    )
}

fn tool_call_accuracy(
    item: &EvalItem,
    metric_kwargs: &Map<String, Value>,
) -> Result<f64, EngineError> {
    let (predicted, expected) = tool_calls(item)?;
    if predicted.is_empty() && expected.is_empty() {
        return Ok(1.0);
    }
    if predicted.is_empty() || expected.is_empty() {
        return Ok(0.0);
    }
    let strict = metric_kwargs
        .get("strict_order")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mut predicted = predicted;
    let mut expected = expected;
    if !strict {
        predicted.sort_by_key(tool_sort_key);
        expected.sort_by_key(tool_sort_key);
    }
    let aligned = predicted
        .iter()
        .map(|call| &call.0)
        .eq(expected.iter().map(|call| &call.0));
    if !aligned {
        return Ok(0.0);
    }
    let compared = predicted.len().min(expected.len());
    let mut score = 0.0;
    for (predicted, expected) in predicted.iter().zip(&expected) {
        if predicted.0 == expected.0 {
            score += argument_accuracy(&predicted.1, &expected.1);
        }
    }
    score /= expected.len() as f64;
    if compared < expected.len() {
        score *= compared as f64 / expected.len() as f64;
    }
    Ok(score)
}

fn tool_call_f1(item: &EvalItem) -> Result<f64, EngineError> {
    let (predicted, expected) = tool_calls(item)?;
    let predicted = predicted
        .into_iter()
        .map(tool_hash)
        .collect::<BTreeSet<_>>();
    let expected = expected.into_iter().map(tool_hash).collect::<BTreeSet<_>>();
    let true_positive = predicted.intersection(&expected).count();
    let false_positive = predicted.difference(&expected).count();
    let false_negative = expected.difference(&predicted).count();
    let precision = true_positive as f64 / (true_positive + false_positive).max(1) as f64;
    let recall = true_positive as f64 / (true_positive + false_negative).max(1) as f64;
    let score = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok((score * 10_000.0).round() / 10_000.0)
}

type Tool = (String, Map<String, Value>);

fn tool_calls(item: &EvalItem) -> Result<(Vec<Tool>, Vec<Tool>), EngineError> {
    let predicted = item
        .trace
        .as_ref()
        .map(TraceView::new)
        .map(|view| {
            view.tools_called()
                .into_iter()
                .map(|call| {
                    let args = call
                        .arguments
                        .and_then(|value| value.pointer("/call/arguments").cloned().or(Some(value)))
                        .and_then(|value| value.as_object().cloned())
                        .unwrap_or_default();
                    (call.name, args)
                })
                .collect()
        })
        .unwrap_or_default();
    let expected_values = item
        .expectations
        .as_ref()
        .and_then(|value| value.get("expected_tool_calls"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let expected = expected_values
        .into_iter()
        .filter_map(|call| {
            let name = call.get("name")?.as_str()?.to_string();
            let args = call
                .get("arguments")
                .or_else(|| call.get("args"))
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            Some((name, args))
        })
        .collect();
    Ok((predicted, expected))
}

fn tool_sort_key(call: &Tool) -> Vec<String> {
    let mut key = vec![call.0.clone()];
    for (name, value) in &call.1 {
        key.push(name.clone());
        key.push(python_str(value));
    }
    key
}

fn argument_accuracy(predicted: &Map<String, Value>, expected: &Map<String, Value>) -> f64 {
    if expected.is_empty() {
        return bool_score(predicted.is_empty());
    }
    expected
        .iter()
        .filter(|(name, value)| {
            predicted
                .get(*name)
                .is_some_and(|candidate| python_str(candidate) == python_str(value))
        })
        .count() as f64
        / expected.len() as f64
}

fn tool_hash(call: Tool) -> String {
    format!("{}:{}", call.0, python_str(&Value::Object(call.1)))
}

fn data_compare(
    reference: &str,
    response: &str,
    metric_kwargs: &Map<String, Value>,
) -> Result<(f64, String), EngineError> {
    let parse = |value: &str| {
        value
            .lines()
            .map(|line| {
                line.split(',')
                    .map(|cell| cell.trim().to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let mut reference = parse(reference);
    let mut response = parse(response);
    if reference.is_empty() || response.is_empty() {
        return Ok((
            f64::NAN,
            "CSV parsing error: No columns to parse from file".to_string(),
        ));
    }
    let reference_header = reference.remove(0);
    let response_header = response.remove(0);
    let mode = metric_kwargs
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("rows");
    let (matched, reference_total, response_total) = if mode == "columns" {
        let matched = reference_header
            .iter()
            .enumerate()
            .filter(|(reference_column, name)| {
                let Some(response_column) = response_header.iter().position(|value| value == *name)
                else {
                    return false;
                };
                (0..reference.len().max(response.len())).all(|row| {
                    reference
                        .get(row)
                        .and_then(|values| values.get(*reference_column))
                        == response
                            .get(row)
                            .and_then(|values| values.get(response_column))
                })
            })
            .count();
        (matched, reference_header.len(), response_header.len())
    } else {
        let matched = reference
            .iter()
            .zip(&response)
            .filter(|(reference_row, response_row)| reference_row == response_row)
            .count();
        (matched, reference.len(), response.len())
    };
    let recall = matched as f64 / reference_total.max(1) as f64;
    let precision = matched as f64 / response_total.max(1) as f64;
    let score = match metric_kwargs
        .get("metric")
        .and_then(Value::as_str)
        .unwrap_or("f1")
    {
        "precision" => precision,
        "recall" => recall,
        _ if precision + recall == 0.0 => 0.0,
        _ => 2.0 * precision * recall / (precision + recall),
    };
    Ok((
        score,
        format!("Mode: {mode}, Precision: {precision:.4}, Recall: {recall:.4}"),
    ))
}
