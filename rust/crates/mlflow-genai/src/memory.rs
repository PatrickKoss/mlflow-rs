use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::trace::{parse_inputs_to_str, parse_outputs_to_str, python_str};
use crate::{
    EngineError, EvalItem, Feedback, MemoryExample, ScorerExecutor, SerializedScorer,
    SerializedScorerCommon,
};

pub(crate) async fn execute(
    executor: &ScorerExecutor,
    _common: &SerializedScorerCommon,
    data: &Map<String, Value>,
    item: &EvalItem,
    gateway_url: Option<&str>,
    embedding_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let base = data
        .get("base_judge")
        .cloned()
        .ok_or(EngineError::InvalidScorerField("base_judge"))?;
    let mut base = SerializedScorer::from_value(base)?;
    let examples = item.memory_examples.as_deref().unwrap_or_default();
    let k = data
        .get("retrieval_k")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(5);
    let query_field = query_field(&base)?;
    let query = item_field(item, query_field).unwrap_or_default();
    let eligible = examples
        .iter()
        .filter_map(|example| {
            example_field(example, query_field)
                .filter(|value| !value.is_empty())
                .map(|text| (example, text))
        })
        .collect::<Vec<_>>();
    let retrieved = if eligible.is_empty() || query.is_empty() {
        Vec::new()
    } else {
        retrieve(
            executor,
            data,
            &eligible,
            &query,
            k,
            embedding_url.ok_or(EngineError::MissingEmbeddingUrl)?,
        )
        .await?
    };
    augment_base(&mut base, data, &retrieved)?;
    let mut feedback =
        Box::pin(executor.execute_all(&base, item, gateway_url, embedding_url)).await?;
    let trace_ids = retrieved
        .iter()
        .map(|example| Value::String(example.trace_id.clone()))
        .collect::<Vec<_>>();
    for value in &mut feedback {
        let metadata = value.metadata.get_or_insert_with(BTreeMap::new);
        if !trace_ids.is_empty() {
            metadata.insert(
                "retrieved_example_trace_ids".to_string(),
                Value::Array(trace_ids.clone()),
            );
        }
        metadata.remove("guideline");
    }
    Ok(feedback)
}

async fn retrieve<'a>(
    executor: &ScorerExecutor,
    data: &Map<String, Value>,
    eligible: &[(&'a MemoryExample, String)],
    query: &str,
    k: usize,
    embedding_url: &str,
) -> Result<Vec<&'a MemoryExample>, EngineError> {
    let model_uri = data
        .get("embedding_model")
        .and_then(Value::as_str)
        .unwrap_or("openai:/text-embedding-3-small");
    let model = model_uri
        .split_once(":/")
        .map(|(_, model)| model)
        .unwrap_or(model_uri);
    let dimensions = data
        .get("embedding_dim")
        .and_then(Value::as_u64)
        .unwrap_or(512);
    let corpus = eligible
        .iter()
        .map(|(_, text)| Value::String(text.clone()))
        .collect::<Vec<_>>();
    let corpus_vectors = embedding_call(
        executor,
        embedding_url,
        json!({"model": model, "input": corpus, "dimensions": dimensions}),
    )
    .await?;
    let query_vectors = embedding_call(
        executor,
        embedding_url,
        json!({"model": model, "input": [query], "dimensions": dimensions}),
    )
    .await?;
    let query = query_vectors.first().ok_or_else(|| {
        EngineError::Embedding("embedding response did not contain a query vector".to_string())
    })?;
    let mut scored = eligible
        .iter()
        .zip(corpus_vectors.iter())
        .map(|((example, _), vector)| (*example, cosine(vector, query)))
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    Ok(scored
        .into_iter()
        .take(k.min(eligible.len()))
        .map(|(example, _)| example)
        .collect())
}

async fn embedding_call(
    executor: &ScorerExecutor,
    url: &str,
    request: Value,
) -> Result<Vec<Vec<f64>>, EngineError> {
    let response = executor
        .client()
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|error| EngineError::Embedding(error.to_string()))?;
    let status = response.status();
    let body: Value = response
        .json()
        .await
        .map_err(|error| EngineError::Embedding(error.to_string()))?;
    if !status.is_success() {
        return Err(EngineError::Embedding(format!("HTTP {status}: {body}")));
    }
    body.get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| EngineError::Embedding(body.to_string()))?
        .iter()
        .map(|entry| {
            entry
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| EngineError::Embedding(entry.to_string()))?
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .ok_or_else(|| EngineError::Embedding(value.to_string()))
                })
                .collect()
        })
        .collect()
}

fn augment_base(
    scorer: &mut SerializedScorer,
    data: &Map<String, Value>,
    examples: &[&MemoryExample],
) -> Result<(), EngineError> {
    let semantic = data
        .get("semantic_memory")
        .and_then(Value::as_array)
        .map(|guidelines| {
            guidelines
                .iter()
                .filter_map(|value| value.get("guideline_text").and_then(Value::as_str))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if semantic.is_empty() && examples.is_empty() {
        return Ok(());
    }
    let data = match scorer {
        SerializedScorer::Instructions(payload) => &mut payload.pydantic_data,
        SerializedScorer::Builtin(payload) => &mut payload.pydantic_data,
        _ => return Err(EngineError::UnsupportedScorer),
    };
    let instructions = data
        .get("instructions")
        .and_then(Value::as_str)
        .ok_or(EngineError::InvalidScorerField("instructions"))?;
    let mut augmented = instructions.to_string();
    if !semantic.is_empty() {
        augmented.push_str(&format!("\n\nDistilled Guidelines ({}):\n", semantic.len()));
        for guideline in semantic {
            augmented.push_str(&format!("  - {guideline}\n"));
        }
    }
    if !examples.is_empty() {
        augmented.push_str(
            "\nSome example judgements are provided below. Align the evaluation with them without referring to the examples directly:\n",
        );
        for example in examples {
            augmented.push_str(&format!(
                "- {}\n",
                serde_json::to_string(&json!({
                    "inputs": example.inputs,
                    "outputs": example.outputs,
                    "expectations": example.expectations,
                    "feedback": example.feedback,
                }))
                .map_err(|error| EngineError::Serialization(error.to_string()))?
            ));
        }
    }
    data.insert("instructions".to_string(), Value::String(augmented));
    Ok(())
}

fn query_field(scorer: &SerializedScorer) -> Result<&'static str, EngineError> {
    let data = match scorer {
        SerializedScorer::Instructions(payload) => &payload.pydantic_data,
        SerializedScorer::Builtin(payload) => &payload.pydantic_data,
        _ => return Err(EngineError::UnsupportedScorer),
    };
    let instructions = data
        .get("instructions")
        .and_then(Value::as_str)
        .ok_or(EngineError::InvalidScorerField("instructions"))?;
    ["inputs", "outputs", "expectations", "conversation", "trace"]
        .into_iter()
        .find(|field| {
            instructions.contains(&format!("{{{{ {field} }}}}"))
                || instructions.contains(&format!("{{{{{field}}}}}"))
        })
        .ok_or_else(|| {
            EngineError::InvalidParams(
                "Unable to build episodic memory: no suitable input field found in judge instructions. Please ensure the judge instructions reference at least one of the following fields: inputs, outputs, expectations, conversation, trace.".to_string(),
            )
        })
}

fn item_field(item: &EvalItem, field: &str) -> Option<String> {
    match field {
        "inputs" => item.inputs.as_ref().map(parse_inputs_to_str),
        "outputs" => item.outputs.as_ref().map(parse_outputs_to_str),
        "expectations" => item.expectations.as_ref().map(python_str),
        "conversation" => item.session.as_ref().map(|value| python_str(&json!(value))),
        "trace" => item.trace.as_ref().map(trace_text),
        _ => None,
    }
}

fn example_field(example: &MemoryExample, field: &str) -> Option<String> {
    match field {
        "inputs" => example.inputs.as_ref().map(parse_inputs_to_str),
        "outputs" => example.outputs.as_ref().map(parse_outputs_to_str),
        "expectations" => example.expectations.as_ref().map(python_str),
        "trace" => example.trace.as_ref().map(trace_text),
        _ => None,
    }
}

fn trace_text(trace: &Value) -> String {
    let view = crate::trace::TraceView::new(trace);
    [view.request(), view.response()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}
