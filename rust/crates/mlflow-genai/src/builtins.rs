use std::collections::BTreeMap;

use fancy_regex::RegexBuilder as FancyRegexBuilder;
use regex::RegexBuilder;
use serde_json::{json, Map, Value};

use crate::judge::{execute_instructions, invoke_prompt};
use crate::trace::{parse_inputs_to_str, parse_outputs_to_str, python_str, TraceView};
use crate::{
    AssessmentSource, BuiltinScorerPayload, EngineError, EvalItem, Feedback,
    InstructionsJudgePayload, ScorerExecutor,
};

const DEFAULT_MODEL: &str = "openai:/gpt-4.1-mini";

const RELEVANCE_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/relevance_to_query.py");
const CORRECTNESS_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/correctness.py");
const EQUIVALENCE_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/equivalence.py");
const GUIDELINES_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/guidelines.py");
const GROUNDEDNESS_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/groundedness.py");
const SUFFICIENCY_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/context_sufficiency.py");
const RETRIEVAL_RELEVANCE_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/retrieval_relevance.py");
const SAFETY_SOURCE: &str = include_str!("../../../../mlflow/genai/judges/prompts/safety.py");
const TOOL_EFFICIENCY_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/tool_call_efficiency.py");
const TOOL_CORRECTNESS_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/tool_call_correctness.py");
const KNOWLEDGE_SOURCE: &str =
    include_str!("../../../../mlflow/genai/judges/prompts/knowledge_retention.py");

pub(crate) async fn execute(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    match payload.class_name.as_str() {
        "PIIDetection" => Ok(vec![code_source(pii_detection(payload, item)?)]),
        "RegexMatch" => Ok(vec![code_source(regex_match(payload, item)?)]),
        "ResponseLength" => Ok(vec![code_source(response_length(payload, item)?)]),
        "Equivalence" => equivalence(executor, payload, item, gateway_url).await,
        "RetrievalRelevance" => retrieval_relevance(executor, payload, item, gateway_url).await,
        "RetrievalSufficiency" => retrieval_sufficiency(executor, payload, item, gateway_url).await,
        "RetrievalGroundedness" => {
            retrieval_groundedness(executor, payload, item, gateway_url).await
        }
        "Guidelines" | "ExpectationsGuidelines" => {
            guidelines(executor, payload, item, gateway_url).await
        }
        "RelevanceToQuery" => {
            let request = required_inputs(item)?;
            let response = required_outputs(item)?;
            custom_prompt(
                executor,
                payload,
                format!(
                    "{}{}",
                    format_prompt(
                        instructions(payload)?,
                        &[("input", request), ("output", response)]
                    ),
                    python_constant(RELEVANCE_SOURCE, "RELEVANCE_TO_QUERY_PROMPT_OUTPUT")?
                ),
                gateway_url,
            )
            .await
        }
        "Correctness" => correctness(executor, payload, item, gateway_url).await,
        "Safety" => {
            let output = required_outputs(item)?;
            let prompt = format_prompt(
                &python_constant(SAFETY_SOURCE, "SAFETY_PROMPT")?,
                &[("content", output)],
            );
            custom_prompt(executor, payload, prompt, gateway_url).await
        }
        "ToolCallEfficiency" => tool_efficiency(executor, payload, item, gateway_url).await,
        "ToolCallCorrectness" => tool_correctness(executor, payload, item, gateway_url).await,
        "KnowledgeRetention" => knowledge_retention(executor, payload, item, gateway_url).await,
        "Fluency"
        | "Completeness"
        | "Summarization"
        | "ConversationCompleteness"
        | "ConversationalGuidelines"
        | "ConversationalRoleAdherence"
        | "ConversationalSafety"
        | "ConversationalToolCallEfficiency"
        | "UserFrustration" => Ok(vec![
            execute_builtin_instructions(executor, payload, item, gateway_url).await?,
        ]),
        other => Err(EngineError::UnsupportedBuiltin(other.to_string())),
    }
}

fn response_length(
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
) -> Result<Feedback, EngineError> {
    let Some(output) = output_text(item) else {
        return Ok(Feedback::code(
            &payload.common.name,
            json!("no"),
            "No outputs provided to evaluate.",
        ));
    };
    let unit = payload
        .pydantic_data
        .get("unit")
        .and_then(Value::as_str)
        .unwrap_or("chars");
    let length = match unit {
        "words" => output.split_whitespace().count(),
        "chars" => output.chars().count(),
        _ => return Err(EngineError::InvalidScorerField("unit")),
    };
    let min = optional_usize(&payload.pydantic_data, "min_length")?;
    let max = optional_usize(&payload.pydantic_data, "max_length")?;
    if let Some(min) = min.filter(|min| length < *min) {
        return Ok(Feedback::code(
            &payload.common.name,
            json!("no"),
            format!("Output length ({length} {unit}) is below the minimum ({min} {unit})"),
        ));
    }
    if let Some(max) = max.filter(|max| length > *max) {
        return Ok(Feedback::code(
            &payload.common.name,
            json!("no"),
            format!("Output length ({length} {unit}) exceeds the maximum ({max} {unit})"),
        ));
    }
    Ok(Feedback::code(
        &payload.common.name,
        json!("yes"),
        format!("Output length ({length} {unit}) is within bounds"),
    ))
}

fn regex_match(payload: &BuiltinScorerPayload, item: &EvalItem) -> Result<Feedback, EngineError> {
    let Some(output) = output_text(item) else {
        return Ok(Feedback::code(
            &payload.common.name,
            json!("no"),
            "No outputs provided to evaluate.",
        ));
    };
    let pattern = required_str(&payload.pydantic_data, "pattern")?;
    let fullmatch = payload
        .pydantic_data
        .get("match_type")
        .and_then(Value::as_str)
        .unwrap_or("search")
        == "fullmatch";
    let expression = if fullmatch {
        format!(r"\A(?:{pattern})\z")
    } else {
        pattern.to_string()
    };
    let regex = FancyRegexBuilder::new(&expression)
        .case_insensitive(
            payload
                .pydantic_data
                .get("case_insensitive")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        )
        .build()
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    let matched = regex
        .is_match(&output)
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    Ok(Feedback::code(
        &payload.common.name,
        json!(if matched { "yes" } else { "no" }),
        format!(
            "Output {} pattern {}",
            if matched { "matches" } else { "does not match" },
            python_string_repr(pattern)
        ),
    ))
}

fn pii_detection(payload: &BuiltinScorerPayload, item: &EvalItem) -> Result<Feedback, EngineError> {
    let Some(output) = output_text(item) else {
        return Ok(Feedback::code(
            &payload.common.name,
            json!("no"),
            "No outputs provided to evaluate.",
        ));
    };
    const TYPES: [&str; 5] = ["email", "phone", "ssn", "credit_card", "ip_address"];
    const PATTERNS: [&str; 5] = [
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        r"(?:(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+[1-9]\d{1,14})",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:(?:4\d{3}|5[1-5]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5})\b",
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    ];
    let requested = payload
        .pydantic_data
        .get("pii_types")
        .and_then(Value::as_array)
        .map(|types| types.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_else(|| TYPES.to_vec());
    let mut detected = Vec::new();
    for pii_type in requested {
        let index = TYPES
            .iter()
            .position(|candidate| *candidate == pii_type)
            .ok_or(EngineError::InvalidScorerField("pii_types"))?;
        if RegexBuilder::new(PATTERNS[index])
            .build()
            .expect("PII regex is static")
            .is_match(&output)
        {
            detected.push(pii_type);
        }
    }
    if detected.is_empty() {
        Ok(Feedback::code(
            &payload.common.name,
            json!("yes"),
            "No PII detected",
        ))
    } else {
        Ok(Feedback::code(
            &payload.common.name,
            json!("no"),
            format!("Detected PII: {}", detected.join(", ")),
        ))
    }
}

async fn equivalence(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let actual = item
        .outputs
        .as_ref()
        .ok_or(EngineError::InvalidScorerField("outputs"))?;
    let expected = item
        .expectations
        .as_ref()
        .and_then(|value| value.get("expected_response"))
        .filter(|value| !value.is_null())
        .ok_or(EngineError::InvalidScorerField("expected_response"))?;
    if is_number_or_bool(actual) && is_number_or_bool(expected) {
        let matches = numeric_value(actual)
            .zip(numeric_value(expected))
            .is_some_and(|(actual, expected)| {
                (actual - expected).abs() <= 1e-9 * actual.abs().max(expected.abs())
            });
        return Ok(vec![code_source(Feedback::code(
            &payload.common.name,
            json!(if matches { "yes" } else { "no" }),
            if matches {
                "Exact numerical match".to_string()
            } else {
                format!(
                    "Values do not match: {} != {}",
                    python_str(actual),
                    python_str(expected)
                )
            },
        ))]);
    }
    let actual = python_scalar_str(actual);
    let expected = python_scalar_str(expected);
    if actual == expected {
        return Ok(vec![code_source(Feedback::code(
            &payload.common.name,
            json!("yes"),
            "Exact string match",
        ))]);
    }
    let prompt = format!(
        "{}{}",
        format_prompt(
            instructions(payload)?,
            &[("output", actual), ("expected_output", expected)]
        ),
        python_constant(EQUIVALENCE_SOURCE, "EQUIVALENCE_PROMPT_OUTPUT")?
    );
    custom_prompt(executor, payload, prompt, gateway_url).await
}

async fn correctness(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let request = required_inputs(item)?;
    let response = required_outputs(item)?;
    let expectations = item.expectations.as_ref().and_then(Value::as_object);
    let expected_response = expectations
        .and_then(|value| value.get("expected_response"))
        .filter(|value| !value.is_null())
        .map(python_scalar_str);
    let expected_facts = expectations
        .and_then(|value| value.get("expected_facts"))
        .and_then(Value::as_array);
    if expected_response.is_none() && expected_facts.is_none() {
        return Err(EngineError::InvalidParams(
            "Correctness scorer requires either `expected_response` or `expected_facts` in the `expectations` dictionary."
                .to_string(),
        ));
    }
    if expected_response.is_some() && expected_facts.is_some() {
        return Err(EngineError::InvalidParams(
            "Only one of expected_response or expected_facts should be provided, not both."
                .to_string(),
        ));
    }
    let ground_truth = expected_response.clone().unwrap_or_else(|| {
        expected_facts.map_or_else(String::new, |facts| {
            facts
                .iter()
                .map(python_scalar_str)
                .fold(String::new(), |mut output, fact| {
                    output.push_str("\n- ");
                    output.push_str(&fact);
                    output
                })
        })
    });
    let mut prompt = format!(
        "{}{}",
        format_prompt(
            instructions(payload)?,
            &[
                ("input", request),
                ("output", response),
                ("ground_truth", ground_truth)
            ]
        ),
        python_constant(CORRECTNESS_SOURCE, "CORRECTNESS_PROMPT_OUTPUT")?
    );
    if expected_response.is_none() && expected_facts.is_some_and(|facts| !facts.is_empty()) {
        prompt.push_str(&python_constant(
            CORRECTNESS_SOURCE,
            "CORRECTNESS_PROMPT_SUFFIX",
        )?);
    }
    custom_prompt(executor, payload, prompt, gateway_url).await
}

async fn guidelines(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let guidelines = if payload.class_name == "Guidelines" {
        payload.pydantic_data.get("guidelines")
    } else {
        item.expectations
            .as_ref()
            .and_then(|value| value.get("guidelines"))
    }
    .ok_or(EngineError::InvalidScorerField("guidelines"))?;
    let rendered = match guidelines {
        Value::String(value) => format!("<guideline>{value}</guideline>"),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(|value| format!("<guideline>{value}</guideline>"))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return Err(EngineError::InvalidScorerField("guidelines")),
    };
    let context = format!(
        "<request>{}</request>\n<response>{}</response>",
        required_inputs(item)?,
        required_outputs(item)?
    );
    let prefix = if payload.class_name == "Guidelines" {
        instructions(payload)?.to_string()
    } else {
        python_constant(GUIDELINES_SOURCE, "GUIDELINES_PROMPT_INSTRUCTIONS")?
    };
    let prompt = format!(
        "{}{}",
        format_prompt(
            &prefix,
            &[("guidelines", rendered), ("guidelines_context", context)]
        ),
        python_constant(GUIDELINES_SOURCE, "GUIDELINES_PROMPT_OUTPUT")?
    );
    let mut feedback = custom_prompt(executor, payload, prompt, gateway_url).await?;
    let guideline_text = match guidelines {
        Value::String(value) => value.clone(),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("\n"),
        _ => unreachable!(),
    };
    feedback[0]
        .metadata
        .get_or_insert_with(BTreeMap::new)
        .insert("guideline".to_string(), json!(guideline_text));
    Ok(feedback)
}

async fn retrieval_relevance(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let trace = item
        .trace
        .as_ref()
        .ok_or(EngineError::InvalidScorerField("trace"))?;
    let view = TraceView::new(trace);
    let request = view.request();
    let contexts = view.retrieval_contexts();
    if contexts.is_empty() {
        return Err(EngineError::InvalidParams(
            "No retrieval context found in the trace. The RetrievalRelevance scorer requires the trace to contain at least one span with type 'RETRIEVER'.".to_string(),
        ));
    }
    let template = python_constant(RETRIEVAL_RELEVANCE_SOURCE, "RETRIEVAL_RELEVANCE_PROMPT")?;
    let mut output = Vec::new();
    for (span_id, chunks) in contexts {
        let mut chunk_feedback = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let content = chunk
                .get("content")
                .map(python_scalar_str)
                .unwrap_or_default();
            let prompt = format_prompt(&template, &[("input", request.clone()), ("doc", content)]);
            let mut feedback = custom_prompt(executor, payload, prompt, gateway_url)
                .await?
                .remove(0);
            sanitize(&mut feedback);
            feedback
                .metadata
                .get_or_insert_with(BTreeMap::new)
                .insert("chunk_index".to_string(), json!(index));
            feedback.span_id = Some(span_id.clone());
            chunk_feedback.push(feedback);
        }
        if !chunk_feedback.is_empty() {
            let passing = chunk_feedback
                .iter()
                .filter(|feedback| feedback.value == json!("yes"))
                .count();
            let mut precision = Feedback::code(
                &format!("{}/precision", payload.common.name),
                json!(passing as f64 / chunk_feedback.len() as f64),
                "",
            );
            precision.rationale.clear();
            precision.source = chunk_feedback[0].source.clone();
            precision.span_id = Some(span_id);
            output.push(precision);
            output.extend(chunk_feedback);
        }
    }
    Ok(output)
}

async fn retrieval_sufficiency(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    retrieval_context_judge(
        executor,
        payload,
        item,
        gateway_url,
        SUFFICIENCY_SOURCE,
        "CONTEXT_SUFFICIENCY_PROMPT_OUTPUT",
        true,
    )
    .await
}

async fn retrieval_groundedness(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    retrieval_context_judge(
        executor,
        payload,
        item,
        gateway_url,
        GROUNDEDNESS_SOURCE,
        "GROUNDEDNESS_PROMPT_OUTPUT",
        false,
    )
    .await
}

async fn retrieval_context_judge(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
    source: &str,
    suffix: &str,
    sufficiency: bool,
) -> Result<Vec<Feedback>, EngineError> {
    let trace = item
        .trace
        .as_ref()
        .ok_or(EngineError::InvalidScorerField("trace"))?;
    let view = TraceView::new(trace);
    let contexts = view.retrieval_contexts();
    if contexts.is_empty() {
        return Err(EngineError::InvalidParams(format!(
            "No retrieval context found in the trace. The {} scorer requires the trace to contain at least one span with type 'RETRIEVER'.",
            payload.class_name
        )));
    }
    let request = view.request();
    let response = view.response();
    let expectations = item.expectations.as_ref();
    let expected_response = expectations
        .and_then(|value| value.get("expected_response"))
        .map(python_scalar_str)
        .unwrap_or_default();
    let expected_facts = expectations
        .and_then(|value| value.get("expected_facts"))
        .and_then(Value::as_array);
    let ground_truth = if expected_response.is_empty() {
        expected_facts.map_or_else(String::new, |facts| {
            format!(
                "  {}",
                facts
                    .iter()
                    .map(python_scalar_str)
                    .fold(String::new(), |mut value, fact| {
                        value.push_str("\n    - ");
                        value.push_str(&fact);
                        value
                    })
                    .trim()
            )
        })
    } else {
        expected_response
    };
    let mut feedbacks = Vec::new();
    for (span_id, chunks) in contexts {
        let context = python_str(&Value::Array(chunks));
        let values = if sufficiency {
            vec![
                ("input", request.clone()),
                ("ground_truth", ground_truth.clone()),
                ("retrieval_context", context),
            ]
        } else {
            vec![
                ("input", request.clone()),
                ("output", response.clone()),
                ("retrieval_context", context),
            ]
        };
        let prompt = format!(
            "{}{}",
            format_prompt(instructions(payload)?, &values),
            python_constant(source, suffix)?
        );
        let mut feedback = custom_prompt(executor, payload, prompt, gateway_url)
            .await?
            .remove(0);
        feedback.span_id = Some(span_id);
        feedbacks.push(feedback);
    }
    Ok(feedbacks)
}

async fn tool_efficiency(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let trace = item
        .trace
        .as_ref()
        .ok_or(EngineError::InvalidScorerField("trace"))?;
    let view = TraceView::new(trace);
    let tools_called = format_tools_called(&view);
    let available_tools = format_available_tools(&view);
    let prompt = format!(
        "{}{}",
        format_prompt(
            instructions(payload)?,
            &[
                ("request", view.request()),
                ("available_tools", available_tools),
                ("tools_called", tools_called)
            ]
        ),
        python_constant(TOOL_EFFICIENCY_SOURCE, "TOOL_CALL_EFFICIENCY_PROMPT_OUTPUT")?
    );
    custom_prompt(executor, payload, prompt, gateway_url).await
}

async fn tool_correctness(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let trace = item
        .trace
        .as_ref()
        .ok_or(EngineError::InvalidScorerField("trace"))?;
    let view = TraceView::new(trace);
    let expected = item
        .expectations
        .as_ref()
        .and_then(|value| value.get("expected_tool_calls"))
        .and_then(Value::as_array);
    let exact = payload
        .pydantic_data
        .get("should_exact_match")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let ordered = payload
        .pydantic_data
        .get("should_consider_ordering")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if exact {
        return Ok(vec![exact_tool_calls(payload, &view, expected, ordered)?]);
    }
    let ordering = if ordered {
        "3) Ordering\n- Consider whether the order of tool calls matches the expected order."
    } else {
        "Note: The order of tool calls does not need to match. You should not penalize the agent for calling tools in a different order than the expected order."
    };
    let (preamble_name, criteria_name, expected_section) = match expected {
        None => (
            "_GROUND_TRUTH_FREE_PREAMBLE",
            "_GROUND_TRUTH_FREE_CRITERIA",
            String::new(),
        ),
        Some(expected) => {
            let include_arguments = expected
                .iter()
                .all(|call| call.get("arguments").is_some_and(|value| !value.is_null()));
            let expected_calls = expected
                .iter()
                .enumerate()
                .flat_map(|(index, call)| {
                    let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
                    let mut lines = if include_arguments {
                        vec![format!("Expected Tool Call {}: {name}", index + 1)]
                    } else {
                        vec![format!("Expected Tool {}: {name}", index + 1)]
                    };
                    if include_arguments {
                        lines.push(
                            match call.get("arguments").filter(|value| {
                                value
                                    .as_object()
                                    .is_none_or(|arguments| !arguments.is_empty())
                            }) {
                                Some(arguments) => {
                                    format!("  Arguments: {}", python_json_dumps(arguments))
                                }
                                None => "  Arguments: empty".to_string(),
                            },
                        );
                    }
                    lines
                })
                .collect::<Vec<_>>()
                .join("\n");
            (
                if include_arguments {
                    "_FULL_EXPECTATIONS_PREAMBLE"
                } else {
                    "_PARTIAL_EXPECTATIONS_PREAMBLE"
                },
                if include_arguments {
                    "_FULL_EXPECTATIONS_CRITERIA"
                } else {
                    "_PARTIAL_EXPECTATIONS_CRITERIA"
                },
                format!("<expected_tool_calls>\n{expected_calls}\n</expected_tool_calls>\n\n"),
            )
        }
    };
    let preamble = python_constant(TOOL_CORRECTNESS_SOURCE, preamble_name)?;
    let criteria = python_constant(TOOL_CORRECTNESS_SOURCE, criteria_name)?
        .replace("{{ordering_instruction}}", ordering);
    let template = python_constant(TOOL_CORRECTNESS_SOURCE, "_PROMPT_TEMPLATE")?;
    let prompt = format!(
        "{}{}",
        format_prompt(
            &template,
            &[
                ("preamble", preamble),
                ("evaluation_criteria", criteria),
                ("expected_section", expected_section),
                ("request", view.request()),
                ("available_tools", format_available_tools(&view)),
                ("tools_called", format_tools_called(&view)),
            ]
        ),
        python_constant(TOOL_CORRECTNESS_SOURCE, "_OUTPUT_FORMAT")?
    );
    custom_prompt(executor, payload, prompt, gateway_url).await
}

fn exact_tool_calls(
    payload: &BuiltinScorerPayload,
    view: &TraceView<'_>,
    expected: Option<&Vec<Value>>,
    ordered: bool,
) -> Result<Feedback, EngineError> {
    let expected = expected.ok_or_else(|| {
        EngineError::InvalidParams(
            "should_exact_match=True requires expectations to be provided. Cannot perform exact matching without ground truth.".to_string(),
        )
    })?;
    let actual = view.tools_called();
    if actual.len() != expected.len() {
        return Ok(code_source(Feedback::code(
            &payload.common.name,
            json!("no"),
            format!(
                "Expected {} tool call(s), but got {} tool call(s).",
                expected.len(),
                actual.len()
            ),
        )));
    }
    let expected_arguments = expected
        .iter()
        .all(|call| call.get("arguments").is_some_and(|value| !value.is_null()));
    let signature = |name: &str, arguments: Option<&Value>| {
        if expected_arguments {
            format!(
                "{name}({})",
                python_json_inline(arguments.unwrap_or(&Value::Null).as_object())
            )
        } else {
            name.to_string()
        }
    };
    let actual_signatures = actual
        .iter()
        .map(|call| signature(&call.name, call.arguments.as_ref()))
        .collect::<Vec<_>>();
    let expected_signatures = expected
        .iter()
        .map(|call| {
            signature(
                call.get("name").and_then(Value::as_str).unwrap_or_default(),
                call.get("arguments"),
            )
        })
        .collect::<Vec<_>>();
    let (matches, rationale) = if ordered {
        let mismatches = actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter_map(|(index, (actual, expected))| {
                let expected_name = expected
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let expected_signature = signature(expected_name, expected.get("arguments"));
                let actual_signature = signature(&actual.name, actual.arguments.as_ref());
                (expected_signature != actual_signature).then(|| {
                    format!(
                        "Position {}: expected {expected_signature}, got {actual_signature}",
                        index + 1
                    )
                })
            })
            .collect::<Vec<_>>();
        if mismatches.is_empty() {
            (
                true,
                "All tool calls match expected sequence exactly.".to_string(),
            )
        } else {
            (
                false,
                format!(
                    "Tool calls do not match in order: {}",
                    mismatches.join("; ")
                ),
            )
        }
    } else {
        let actual = actual_signatures
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        let expected = expected_signatures
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        if actual == expected {
            (
                true,
                "All expected tool calls present (order ignored).".to_string(),
            )
        } else {
            let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
            let extra = actual.difference(&expected).cloned().collect::<Vec<_>>();
            let mut parts = Vec::new();
            if !missing.is_empty() {
                parts.push(format!("Missing: {}", python_set_repr(&missing)));
            }
            if !extra.is_empty() {
                parts.push(format!("Unexpected: {}", python_set_repr(&extra)));
            }
            (false, parts.join("; "))
        }
    };
    Ok(code_source(Feedback::code(
        &payload.common.name,
        json!(if matches { "yes" } else { "no" }),
        rationale,
    )))
}

async fn knowledge_retention(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let session = item
        .session
        .as_ref()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            EngineError::InvalidParams(
                "Must specify 'session' - cannot evaluate knowledge retention on empty session."
                    .to_string(),
            )
        })?;
    let mut per_turn = Vec::new();
    for index in 0..session.len() {
        let mut turn_payload = payload.clone();
        turn_payload.common.name = "last_turn_knowledge_retention".to_string();
        let feedback = execute_builtin_instructions(
            executor,
            &turn_payload,
            &EvalItem {
                session: Some(session[..=index].to_vec()),
                ..EvalItem::default()
            },
            gateway_url,
        )
        .await?;
        per_turn.push(feedback);
    }
    let failed = per_turn
        .iter()
        .filter(|feedback| feedback.value == json!("no"))
        .count();
    let total = per_turn.len();
    let mut lines = vec![format!(
        "Knowledge retention evaluation across {total} turn(s):"
    )];
    lines.extend(per_turn.iter().enumerate().map(|(index, feedback)| {
        format!(
            "- Turn {}: {} {}",
            index + 1,
            if feedback.value == json!("no") {
                "✗"
            } else {
                "✓"
            },
            feedback.rationale
        )
    }));
    if failed > 0 {
        lines.push(format!(
            "\nOverall: NO - Knowledge retention failed in {failed} out of {total} turn(s)."
        ));
    } else {
        lines.push(format!(
            "\nOverall: YES - Knowledge retention successful across all {total} turn(s)."
        ));
    }
    Ok(vec![Feedback {
        name: payload.common.name.clone(),
        value: json!(if failed > 0 { "no" } else { "yes" }),
        rationale: lines.join("\n"),
        source: Some(AssessmentSource {
            source_type: "LLM_JUDGE".to_string(),
            source_id: Some(model(payload).to_string()),
        }),
        metadata: None,
        span_id: None,
        trace_id: None,
    }])
}

async fn execute_builtin_instructions(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Feedback, EngineError> {
    let mut data = payload.pydantic_data.clone();
    if payload.class_name == "KnowledgeRetention" && !data.contains_key("instructions") {
        data.insert(
            "instructions".to_string(),
            Value::String(python_constant(
                KNOWLEDGE_SOURCE,
                "KNOWLEDGE_RETENTION_PROMPT",
            )?),
        );
    }
    data.insert("model".to_string(), json!(model(payload)));
    data.entry("feedback_value_type".to_string())
        .or_insert_with(|| json!({"enum": ["yes", "no"], "type": "string"}));
    if payload.class_name == "UserFrustration" {
        data.insert(
            "feedback_value_type".to_string(),
            json!({"enum": ["none", "resolved", "unresolved"], "type": "string"}),
        );
    }
    if payload.class_name == "ConversationalToolCallEfficiency" {
        data.insert(
            "include_tool_calls_in_conversation".to_string(),
            json!(true),
        );
    }
    if matches!(
        payload.class_name.as_str(),
        "ConversationCompleteness"
            | "ConversationalGuidelines"
            | "ConversationalRoleAdherence"
            | "ConversationalSafety"
            | "ConversationalToolCallEfficiency"
    ) {
        data.insert("generate_rationale_first".to_string(), json!(true));
    }
    execute_instructions(
        executor,
        &InstructionsJudgePayload {
            common: payload.common.clone(),
            pydantic_data: data,
        },
        item,
        gateway_url,
    )
    .await
}

async fn custom_prompt(
    executor: &ScorerExecutor,
    payload: &BuiltinScorerPayload,
    prompt: String,
    gateway_url: Option<&str>,
) -> Result<Vec<Feedback>, EngineError> {
    let mut feedback = invoke_prompt(
        executor,
        model(payload),
        &payload.common.name,
        prompt,
        payload
            .pydantic_data
            .get("inference_params")
            .and_then(Value::as_object),
        gateway_url,
    )
    .await?;
    sanitize(&mut feedback);
    Ok(vec![feedback])
}

fn sanitize(feedback: &mut Feedback) {
    let Some(value) = feedback.value.as_str() else {
        return;
    };
    let normalized = value.trim().to_ascii_lowercase();
    const YES: [&str; 8] = [
        "yes", "true", "1", "pass", "passed", "positive", "relevant", "safe",
    ];
    const NO: [&str; 8] = [
        "no",
        "false",
        "0",
        "fail",
        "failed",
        "negative",
        "irrelevant",
        "unsafe",
    ];
    feedback.value = if YES.contains(&normalized.as_str()) {
        json!("yes")
    } else if NO.contains(&normalized.as_str()) {
        json!("no")
    } else if matches!(normalized.as_str(), "yes" | "no" | "unknown") {
        json!(normalized)
    } else {
        json!("unknown")
    };
}

fn code_source(mut feedback: Feedback) -> Feedback {
    feedback.source = Some(AssessmentSource {
        source_type: "CODE".to_string(),
        source_id: Some("default".to_string()),
    });
    feedback
}

fn model(payload: &BuiltinScorerPayload) -> &str {
    payload
        .pydantic_data
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_MODEL)
}

fn instructions(payload: &BuiltinScorerPayload) -> Result<&str, EngineError> {
    required_str(&payload.pydantic_data, "instructions")
}

fn required_inputs(item: &EvalItem) -> Result<String, EngineError> {
    if let Some(value) = &item.inputs {
        return Ok(parse_inputs_to_str(value));
    }
    item.trace
        .as_ref()
        .and_then(|trace| TraceView::new(trace).inputs())
        .as_ref()
        .map(parse_inputs_to_str)
        .ok_or(EngineError::InvalidScorerField("inputs"))
}

fn required_outputs(item: &EvalItem) -> Result<String, EngineError> {
    if let Some(value) = &item.outputs {
        return Ok(parse_outputs_to_str(value));
    }
    item.trace
        .as_ref()
        .and_then(|trace| TraceView::new(trace).outputs())
        .as_ref()
        .map(parse_outputs_to_str)
        .ok_or(EngineError::InvalidScorerField("outputs"))
}

fn output_text(item: &EvalItem) -> Option<String> {
    item.outputs.as_ref().map(parse_outputs_to_str).or_else(|| {
        item.trace
            .as_ref()
            .map(TraceView::new)
            .map(|trace| trace.response())
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

fn python_constant(source: &str, name: &str) -> Result<String, EngineError> {
    let marker = format!("{name} = \"\"\"");
    let start = source
        .find(&marker)
        .map(|index| index + marker.len())
        .ok_or_else(|| EngineError::Serialization(format!("Python prompt {name} not found")))?;
    let tail = &source[start..];
    let end = tail.find("\"\"\"").ok_or_else(|| {
        EngineError::Serialization(format!("Python prompt {name} is unterminated"))
    })?;
    Ok(tail[..end].replace("\\\n", ""))
}

fn format_prompt(template: &str, values: &[(impl AsRef<str>, String)]) -> String {
    let mut prompt = template.to_string();
    for (key, value) in values {
        let key = key.as_ref();
        for marker in [format!("{{{{{key}}}}}"), format!("{{{{ {key} }}}}")] {
            prompt = prompt.replace(&marker, value);
        }
    }
    prompt
}

fn python_string_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn python_scalar_str(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => python_str(value),
    }
}

fn is_number_or_bool(value: &Value) -> bool {
    value.is_number() || value.is_boolean()
}

fn numeric_value(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_bool().map(|value| if value { 1.0 } else { 0.0 }))
}

fn format_tools_called(view: &TraceView<'_>) -> String {
    let calls = view.tools_called();
    if calls.is_empty() {
        return "No tools called".to_string();
    }
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let mut value = format!(
                "Tool Call {}: {}\n  Input Arguments: {}\n  Output: {}",
                index + 1,
                call.name,
                call.arguments
                    .as_ref()
                    .map(python_str)
                    .unwrap_or_else(|| "{}".to_string()),
                call.outputs
                    .as_ref()
                    .filter(|value| python_truthy(value))
                    .map(python_scalar_str)
                    .unwrap_or_else(|| "(no output)".to_string())
            );
            if let Some(exception) = &call.exception {
                value.push_str(&format!("\n  Exception: {exception}"));
            }
            value
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_available_tools(view: &TraceView<'_>) -> String {
    let tools = view.available_tools();
    if tools.is_empty() {
        "No tools available".to_string()
    } else {
        tools
            .iter()
            .map(|tool| {
                let function = tool.get("function").unwrap_or(tool);
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let description = function
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let mut rendered = if description.is_empty() {
                    format!("- {name}")
                } else {
                    format!("- {name}: {description}")
                };
                if let Some(properties) = function
                    .pointer("/parameters/properties")
                    .and_then(Value::as_object)
                {
                    let required = function
                        .pointer("/parameters/required")
                        .and_then(Value::as_array);
                    let parameters = properties
                        .iter()
                        .map(|(name, property)| {
                            let marker = if required.is_some_and(|required| {
                                required.iter().any(|value| value.as_str() == Some(name))
                            }) {
                                "required"
                            } else {
                                "optional"
                            };
                            let mut line = format!("    - {name} ({marker})");
                            if let Some(kind) = property.get("type").and_then(Value::as_str) {
                                line.push_str(&format!(": {kind}"));
                            }
                            if let Some(description) =
                                property.get("description").and_then(Value::as_str)
                            {
                                line.push_str(&format!(" - {description}"));
                            }
                            line
                        })
                        .collect::<Vec<_>>();
                    if !parameters.is_empty() {
                        rendered.push('\n');
                        rendered.push_str(&parameters.join("\n"));
                    }
                }
                rendered
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

fn python_json_dumps(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).unwrap_or_default(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json_dumps)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).unwrap_or_default(),
                    python_json_dumps(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_json_inline(value: Option<&Map<String, Value>>) -> String {
    fn render(value: &Value) -> String {
        match value {
            Value::Array(values) => {
                format!(
                    "[{}]",
                    values.iter().map(render).collect::<Vec<_>>().join(", ")
                )
            }
            Value::Object(values) => format!(
                "{{{}}}",
                values
                    .iter()
                    .map(|(key, value)| format!(
                        "{}: {}",
                        serde_json::to_string(key).expect("JSON object key serializes"),
                        render(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            value => serde_json::to_string(value).expect("JSON value serializes"),
        }
    }

    value.map_or_else(
        || "{}".to_string(),
        |value| render(&Value::Object(value.clone())),
    )
}

fn python_set_repr(values: &[String]) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(|value| python_string_repr(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Bool(true) => true,
    }
}
