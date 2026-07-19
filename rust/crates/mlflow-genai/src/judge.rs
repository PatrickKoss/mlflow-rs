use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::trace::{conversation, TraceView};
use crate::{
    AssessmentSource, EngineError, EvalItem, Feedback, InstructionsJudgePayload, ScorerExecutor,
};

const JUDGE_BASE_PROMPT: &str = "You are an expert judge tasked with evaluating the performance of an AI\nagent on a particular query. You will be given instructions that describe the criteria and\nmethodology for evaluating the agent's performance on the query.";
const RESULT_DESCRIPTION: &str = "The evaluation rating/result";
const RATIONALE_DESCRIPTION: &str = "Detailed explanation for the evaluation";
const EMPTY_TRACE_USER_MESSAGE: &str = "Use the tools to inspect the trace and return the JSON rating per the system message. This message and your tool calls in this chat are not the input or response being judged. The trace lives only behind the tools.";

const TRACE_PROMPT: &str = " Your job is to analyze a trace of the agent's execution on the
query and provide an evaluation rating in accordance with the instructions.

A *trace* is a step-by-step record of how the agent processed the query, including the input query
itself, all intermediate steps, decisions, and outputs. Each step in a trace is represented as a
*span*, which includes the inputs and outputs of that step, as well as latency information and
metadata.

The instructions containing the evaluation criteria and methodology are provided below, and they
refer to a placeholder called {{ trace }}. To read the actual trace, you will need to use the
tools provided to you. These tools enable you to 1. fetch trace metadata, timing, & execution
details, 2. list all spans in the trace with inputs and outputs, 3. search for specific text or
patterns across the entire trace, and much more. These tools do *not* require you to specify a
particular trace; the tools will select the relevant trace automatically (however, you *will* need
to specify *span* IDs when retrieving specific spans).

**Important: do not grade this conversation.** Your tool calls and their results in this chat
are how you inspect the trace; they are not actions the traced agent took. Inspect the trace via
the tools before producing a verdict.

In order to follow the instructions precisely and correctly, you must think methodically and act
step-by-step:

1. Thoroughly read the instructions to understand what information you need to gather from the trace
   in order to perform the evaluation, according to the criteria and methodology specified.
2. Look at the tools available to you, and use as many of them as necessary in order to gather the
   information you need from the trace.
3. Carefully read and analyze the information you gathered.
4. Think critically about whether you have enough information to produce an evaluation rating in
   accordance with the instructions. If you do not have enough information, or if you suspect that
   there is additional relevant information in the trace that you haven't gathered, then go back
   to steps 2 and 3.
5. Once you have gathered enough information, provide your evaluation rating in accordance with the
   instructions.

You *must* format your evaluation rating as a JSON object with the following fields. Pay close
attention to the field type of the evaluation rating (string, boolean, numeric, etc.), and ensure
that it conforms to the instructions.

Evaluation Rating Fields
------------------------
{evaluation_rating_fields}

Instructions
------------------------
{instructions}
";

pub(crate) async fn execute_instructions(
    executor: &ScorerExecutor,
    payload: &InstructionsJudgePayload,
    item: &EvalItem,
    gateway_url: Option<&str>,
) -> Result<Feedback, EngineError> {
    let instructions = required_str(&payload.pydantic_data, "instructions")?;
    let model_uri = required_str(&payload.pydantic_data, "model")?;
    let fields = instruction_field_order(instructions);
    validate_required_fields(&fields, item)?;
    let rationale_first = payload
        .pydantic_data
        .get("generate_rationale_first")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let schema = response_format(&payload.pydantic_data, rationale_first)?;
    let is_trace = fields.contains(&"trace");
    let system = if is_trace {
        let descriptions = output_field_descriptions(
            payload.pydantic_data.get("feedback_value_type"),
            rationale_first,
        );
        format!("{JUDGE_BASE_PROMPT}{TRACE_PROMPT}")
            .replace("{evaluation_rating_fields}", &descriptions)
            .replace("{instructions}", instructions)
    } else {
        add_output_format_instructions(
            &format!("{JUDGE_BASE_PROMPT}\n\nYour task: {instructions}."),
            payload.pydantic_data.get("feedback_value_type"),
            rationale_first,
        )
    };
    let user = build_user_message(
        &fields,
        item,
        payload
            .pydantic_data
            .get("include_tool_calls_in_conversation")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        payload
            .pydantic_data
            .get("include_timing_in_conversation")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    )?;
    let messages = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user}),
    ];
    let inference = payload
        .pydantic_data
        .get("inference_params")
        .and_then(Value::as_object);
    let completion = invoke(
        executor,
        model_uri,
        messages,
        schema,
        inference,
        if is_trace { item.trace.as_ref() } else { None },
        gateway_url,
    )
    .await?;
    let mut feedback = completion.feedback(&payload.common.name, model_uri)?;
    feedback.metadata.get_or_insert_with(BTreeMap::new).insert(
        "guideline".to_string(),
        Value::String(instructions.to_string()),
    );
    Ok(feedback)
}

pub(crate) async fn invoke_prompt(
    executor: &ScorerExecutor,
    model_uri: &str,
    name: &str,
    prompt: String,
    inference: Option<&Map<String, Value>>,
    gateway_url: Option<&str>,
) -> Result<Feedback, EngineError> {
    let completion = invoke(
        executor,
        model_uri,
        vec![json!({"role": "user", "content": prompt})],
        default_response_format(),
        inference,
        None,
        gateway_url,
    )
    .await?;
    completion.feedback(name, model_uri)
}

struct JudgeCompletion {
    body: Value,
    content: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    cost_usd: f64,
}

impl JudgeCompletion {
    fn feedback(self, name: &str, model_uri: &str) -> Result<Feedback, EngineError> {
        let cleaned = strip_markdown_code_blocks(&self.content);
        let parsed: Value = serde_json::from_str(&cleaned)
            .map_err(|error| EngineError::MalformedGatewayResponse(error.to_string()))?;
        let value = parsed
            .get("result")
            .cloned()
            .ok_or(EngineError::InvalidScorerField("result"))?;
        let rationale = parsed
            .get("rationale")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .replace("Let's think step by step. ", "");
        let mut metadata = BTreeMap::new();
        if self.prompt_tokens > 0 {
            metadata.insert(
                "mlflow.assessment.judgeInputTokens".to_string(),
                json!(self.prompt_tokens),
            );
        }
        if self.completion_tokens > 0 {
            metadata.insert(
                "mlflow.assessment.judgeOutputTokens".to_string(),
                json!(self.completion_tokens),
            );
        }
        if self.cost_usd != 0.0 {
            metadata.insert(
                "mlflow.assessment.judgeCost".to_string(),
                json!(self.cost_usd),
            );
        }
        let _ = self.body;
        Ok(Feedback {
            name: name.to_string(),
            value,
            rationale,
            source: Some(AssessmentSource {
                source_type: "LLM_JUDGE".to_string(),
                source_id: Some(model_uri.to_string()),
            }),
            metadata: (!metadata.is_empty()).then_some(metadata),
            span_id: None,
            trace_id: None,
        })
    }
}

async fn invoke(
    executor: &ScorerExecutor,
    model_uri: &str,
    mut messages: Vec<Value>,
    response_format: Value,
    inference: Option<&Map<String, Value>>,
    trace: Option<&Value>,
    gateway_url: Option<&str>,
) -> Result<JudgeCompletion, EngineError> {
    let model = model_uri
        .split_once(":/")
        .map(|(_, model)| model)
        .filter(|model| !model.is_empty())
        .unwrap_or(model_uri);
    let tools: Value = serde_json::from_str(include_str!("judge_tools.json"))
        .expect("judge tool definitions are valid JSON");
    let max_iterations = std::env::var("MLFLOW_JUDGE_MAX_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    let mut cost_usd = 0.0;
    for _ in 0..max_iterations {
        let mut request = Map::new();
        request.insert("model".to_string(), Value::String(model.to_string()));
        request.insert("messages".to_string(), Value::Array(messages.clone()));
        if trace.is_some() {
            request.insert("tools".to_string(), tools.clone());
            request.insert("tool_choice".to_string(), Value::String("auto".to_string()));
        }
        request.insert("response_format".to_string(), response_format.clone());
        if let Some(inference) = inference {
            request.extend(inference.clone());
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
        prompt_tokens += body
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        completion_tokens += body
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        cost_usd += body
            .pointer("/_hidden_params/response_cost")
            .or_else(|| body.get("response_cost"))
            .and_then(Value::as_f64)
            .unwrap_or_default();
        let message = body
            .pointer("/choices/0/message")
            .and_then(Value::as_object)
            .ok_or_else(|| EngineError::MalformedGatewayResponse(body.to_string()))?;
        let tool_calls = message.get("tool_calls").and_then(Value::as_array);
        if let Some(tool_calls) = tool_calls.filter(|calls| !calls.is_empty()) {
            let trace = trace.ok_or_else(|| {
                EngineError::MalformedGatewayResponse("tool call without trace".to_string())
            })?;
            messages.push(Value::Object(message.clone()));
            for call in tool_calls {
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| EngineError::MalformedGatewayResponse(call.to_string()))?;
                let arguments = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let arguments: Value = serde_json::from_str(arguments)
                    .map_err(|error| EngineError::Tool(error.to_string()))?;
                let content = TraceView::new(trace).invoke_tool(name, &arguments)?;
                messages.push(json!({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": call.get("id").cloned().unwrap_or(Value::Null),
                    "name": name,
                }));
            }
            continue;
        }
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| EngineError::MalformedGatewayResponse(body.to_string()))?
            .to_string();
        return Ok(JudgeCompletion {
            body,
            content,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        });
    }
    Err(EngineError::Gateway(format!(
        "Judge model exceeded maximum number of iterations ({max_iterations})"
    )))
}

fn build_user_message(
    fields: &[&str],
    item: &EvalItem,
    include_tool_calls: bool,
    include_timing: bool,
) -> Result<String, EngineError> {
    let mut parts = Vec::new();
    for field in fields.iter().filter(|field| **field != "trace") {
        let value = match *field {
            "inputs" => item.inputs.as_ref().map(pretty_json),
            "outputs" => item.outputs.as_ref().map(pretty_json),
            "expectations" => item.expectations.as_ref().map(pretty_json),
            "conversation" => item.session.as_ref().map(|session| {
                pretty_json(&Value::Array(conversation(
                    session,
                    include_tool_calls,
                    include_timing,
                )))
            }),
            _ => None,
        };
        if let Some(value) = value {
            parts.push(format!("{field}: {value}"));
        }
    }
    Ok(if parts.is_empty() {
        EMPTY_TRACE_USER_MESSAGE.to_string()
    } else {
        parts.join("\n")
    })
}

fn validate_required_fields(fields: &[&str], item: &EvalItem) -> Result<(), EngineError> {
    for field in fields {
        let missing = match *field {
            "inputs" => item.inputs.is_none(),
            "outputs" => item.outputs.is_none(),
            "expectations" => item.expectations.is_none(),
            "trace" => item.trace.is_none(),
            "conversation" => item.session.is_none(),
            _ => false,
        };
        if missing {
            let field = match *field {
                "inputs" => "inputs",
                "outputs" => "outputs",
                "expectations" => "expectations",
                "trace" => "trace",
                "conversation" => "session",
                _ => unreachable!("instruction fields are closed"),
            };
            return Err(EngineError::InvalidScorerField(field));
        }
    }
    Ok(())
}

fn instruction_field_order(instructions: &str) -> Vec<&'static str> {
    const FIELDS: [&str; 5] = ["inputs", "outputs", "trace", "expectations", "conversation"];
    let mut positions = FIELDS
        .into_iter()
        .filter_map(|field| {
            instructions
                .match_indices("{{")
                .find_map(|(start, _)| {
                    let tail = &instructions[start + 2..];
                    let end = tail.find("}}")?;
                    (tail[..end].trim() == field).then_some(start)
                })
                .map(|position| (position, field))
        })
        .collect::<Vec<_>>();
    positions.sort_unstable_by_key(|(position, _)| *position);
    positions.into_iter().map(|(_, field)| field).collect()
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).expect("serde_json::Value serialization cannot fail")
}

fn add_output_format_instructions(
    prompt: &str,
    schema: Option<&Value>,
    rationale_first: bool,
) -> String {
    format!(
        "{prompt}\n\nYou *must* format your evaluation rating as a JSON object with the following fields (no markdown). Pay close attention to the field type of the evaluation rating (string, boolean, numeric, etc.), and ensure that it conforms to the instructions.\n\n{}",
        output_field_descriptions(schema, rationale_first)
    )
}

fn output_field_descriptions(schema: Option<&Value>, rationale_first: bool) -> String {
    let result_type = schema.map(format_type).unwrap_or_else(|| "str".to_string());
    let result = format!("- result ({result_type}): {RESULT_DESCRIPTION}");
    let rationale = format!("- rationale (str): {RATIONALE_DESCRIPTION}");
    if rationale_first {
        format!("{rationale}\n{result}")
    } else {
        format!("{result}\n{rationale}")
    }
}

fn format_type(schema: &Value) -> String {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return format!(
            "Literal[{}]",
            values
                .iter()
                .map(|value| match value {
                    Value::String(value) => format!("'{value}'"),
                    value => value.to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    match schema
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("string")
    {
        "integer" => "int",
        "number" => "float",
        "boolean" => "bool",
        "object" => "<class 'dict'>",
        "array" => "<class 'list'>",
        _ => "str",
    }
    .to_string()
}

fn response_format(data: &Map<String, Value>, rationale_first: bool) -> Result<Value, EngineError> {
    let mut result = data
        .get("feedback_value_type")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| Map::from_iter([("type".to_string(), json!("string"))]));
    result.insert("description".to_string(), json!(RESULT_DESCRIPTION));
    let rationale = json!({
        "description": RATIONALE_DESCRIPTION,
        "title": "Rationale",
        "type": "string"
    });
    let mut properties = Map::new();
    let mut required = Vec::new();
    if rationale_first {
        properties.insert("rationale".to_string(), rationale.clone());
        properties.insert("result".to_string(), Value::Object(result));
        required.extend([json!("rationale"), json!("result")]);
    } else {
        properties.insert("result".to_string(), Value::Object(result));
        properties.insert("rationale".to_string(), rationale);
        required.extend([json!("result"), json!("rationale")]);
    }
    Ok(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "ResponseFormat",
            "schema": {
                "properties": properties,
                "required": required,
                "title": "ResponseFormat",
                "type": "object",
                "additionalProperties": false
            },
            "strict": true
        }
    }))
}

fn default_response_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "JudgeEvaluation",
            "schema": {
                "properties": {
                    "result": {"description": RESULT_DESCRIPTION, "title": "Result", "type": "string"},
                    "rationale": {"description": RATIONALE_DESCRIPTION, "title": "Rationale", "type": "string"}
                },
                "required": ["result", "rationale"],
                "title": "JudgeEvaluation",
                "type": "object",
                "additionalProperties": false
            },
            "strict": true
        }
    })
}

fn strip_markdown_code_blocks(response: &str) -> String {
    let cleaned = response.trim();
    if cleaned.starts_with("```") {
        let lines = cleaned.lines().collect::<Vec<_>>();
        let end = lines
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, line)| line.trim() == "```")
            .map(|(index, _)| index)
            .unwrap_or(lines.len());
        return lines[1..end].join("\n");
    }
    if let Some(start) = cleaned.to_ascii_lowercase().find("```json\n") {
        let body = &cleaned[start + 8..];
        if let Some(end) = body.find("\n```") {
            return body[..end].trim().to_string();
        }
    }
    cleaned.to_string()
}

fn required_str<'a>(
    data: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, EngineError> {
    data.get(field)
        .and_then(Value::as_str)
        .ok_or(EngineError::InvalidScorerField(field))
}
