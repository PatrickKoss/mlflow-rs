//! OpenAI-compatible Assistant provider and in-process tool loop.

use std::time::Duration;

use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::assistant::{AssistantEvent, AssistantProviderRequest};
use crate::assistant_providers::{format_system_prompt, python_json, PermissionsConfig};
use crate::assistant_tools::{execute_tool, static_permission_error, tools_schema};
use crate::gateway_provider_matrix::{model_accounting, supported_provider_names};

const SYSTEM_PROMPT: &str = include_str!("assistant_providers/codex_system_prompt.txt");
const MAX_SESSION_BYTES: usize = 500 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    MlflowGateway,
    Ollama,
}

impl Preset {
    pub const fn name(self) -> &'static str {
        match self {
            Self::MlflowGateway => "mlflow_gateway",
            Self::Ollama => "ollama",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::MlflowGateway => "MLflow AI Gateway",
            Self::Ollama => "Ollama",
        }
    }

    pub const fn connection_hint(self) -> &'static str {
        match self {
            Self::MlflowGateway => {
                "Configure an LLM chat endpoint on the MLflow AI Gateway and select it."
            }
            Self::Ollama => "Make sure Ollama is running: ollama serve",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub preset: Preset,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub permissions: PermissionsConfig,
}

pub fn stream(
    config: Config,
    request: AssistantProviderRequest,
) -> BoxStream<'static, AssistantEvent> {
    let (sender, receiver) = mpsc::channel(32);
    tokio::spawn(async move {
        if let Err(error) = run(config, request, &sender).await {
            let _ = sender.send(AssistantEvent::error(error)).await;
        }
    });
    stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|event| (event, receiver))
    })
    .boxed()
}

async fn run(
    config: Config,
    request: AssistantProviderRequest,
    sender: &mpsc::Sender<AssistantEvent>,
) -> Result<(), String> {
    let base_url = config
        .base_url
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_end_matches('/').to_string())
        .or_else(|| {
            (config.preset == Preset::Ollama).then(|| "http://localhost:11434".to_string())
        });
    let chat_url = match config.preset {
        Preset::MlflowGateway if !request.tracking_uri.is_empty() => format!(
            "{}/gateway/mlflow/v1/chat/completions",
            request.tracking_uri.trim_end_matches('/')
        ),
        Preset::MlflowGateway => {
            return send_provider_error(sender, &config, "chat URL could not be resolved").await
        }
        Preset::Ollama => format!(
            "{}/v1/chat/completions",
            base_url.as_deref().unwrap_or_default()
        ),
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|error| error.to_string())?;
    let model = if !config.model.is_empty() && config.model != "default" {
        config.model.clone()
    } else if config.preset == Preset::Ollama {
        let base = base_url.as_deref().unwrap_or_default();
        let models = list_ollama_models(&client, base, config.api_key.as_deref())
            .await
            .map_err(|error| {
                format!(
                    "Cannot connect to {} at {base}: {error}. {}",
                    config.preset.display_name(),
                    config.preset.connection_hint()
                )
            })?;
        match models.first() {
            Some(model) => model.clone(),
            None => {
                let _ = sender
                    .send(AssistantEvent::error(format!(
                        "No models available from {} at {base}.",
                        config.preset.display_name()
                    )))
                    .await;
                return Ok(());
            }
        }
    } else {
        let _ = sender
            .send(AssistantEvent::error(format!(
                "No model selected for {}. {}",
                config.preset.display_name(),
                config.preset.connection_hint()
            )))
            .await;
        return Ok(());
    };

    let mut messages = decode_session(request.session_id.as_deref());
    if messages.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": format_system_prompt(SYSTEM_PROMPT, &request.tracking_uri),
        }));
    }
    let pending = pending_tool_calls(&messages);
    let decisions = request
        .context
        .get("tool_decisions")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut is_resuming = !decisions.is_empty() && !pending.is_empty();
    if !is_resuming {
        for call in &pending {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call.get("id").cloned().unwrap_or(Value::Null),
                "content": "Tool call cancelled by user.",
            }));
        }
        let user_text = if request.context.is_empty() {
            request.prompt.clone()
        } else {
            format!(
                "<context>\n{}\n</context>\n\n{}",
                python_json(&Value::Object(request.context.clone())),
                request.prompt
            )
        };
        messages.push(json!({"role":"user", "content":user_text}));
    }

    loop {
        let calls = if is_resuming {
            is_resuming = false;
            pending.clone()
        } else {
            let turn = model_turn(
                &client,
                &chat_url,
                config.api_key.as_deref(),
                &model,
                &messages,
                config.preset.display_name(),
                sender,
            )
            .await?;
            if turn.tool_calls.is_empty() {
                if !turn.visible_text.is_empty() {
                    messages.push(json!({"role":"assistant", "content":turn.visible_text}));
                }
                break;
            }
            let calls = turn.tool_calls;
            messages.push(json!({
                "role":"assistant",
                "content": if turn.visible_text.is_empty() { Value::Null } else { Value::String(turn.visible_text) },
                "tool_calls": calls.clone(),
            }));
            calls
        };

        let mut paused = false;
        for call in &calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let function = call.get("function").and_then(Value::as_object);
            let tool_name = function
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let arguments = function
                .and_then(|function| function.get("arguments"))
                .cloned()
                .unwrap_or_else(|| Value::String("{}".to_string()));
            let input = match arguments {
                Value::String(arguments) => {
                    serde_json::from_str(&arguments).unwrap_or_else(|_| json!({}))
                }
                value => value,
            };
            let needs_prompt = static_permission_error(
                tool_name,
                &input,
                &config.permissions,
                request.cwd.as_deref(),
            )
            .is_some();
            let gated = !config.permissions.full_access
                && !request.mlflow_session_id.is_empty()
                && needs_prompt;
            let decision = decisions.get(id).and_then(Value::as_str);
            if !(gated && decision.is_some()) {
                send(
                    sender,
                    AssistantEvent::new(
                        "message",
                        json!({"message":{"role":"assistant","content":[{"id":id,"name":tool_name,"input":input}]}}),
                    ),
                )
                .await?;
            }
            if gated && decision.is_none() {
                send(
                    sender,
                    AssistantEvent::new(
                        "permission_request",
                        json!({"request_id":id,"tool_name":tool_name,"tool_input":input}),
                    ),
                )
                .await?;
                paused = true;
                break;
            }
            if gated && decision != Some("allow") {
                let denied = "Permission denied by user.";
                send_tool_result(sender, id, denied, true).await?;
                messages.push(json!({"role":"tool","tool_call_id":id,"content":denied}));
                continue;
            }
            let effective_permissions = if gated {
                PermissionsConfig {
                    full_access: true,
                    ..PermissionsConfig::default()
                }
            } else {
                config.permissions.clone()
            };
            let result = execute_tool(
                tool_name,
                &input,
                request.cwd.as_deref(),
                Some(&request.tracking_uri),
                &effective_permissions,
            )
            .await;
            send_tool_result(sender, id, &result.content, result.is_error).await?;
            messages.push(json!({
                "role":"tool",
                "tool_call_id":id,
                "content":result.content,
            }));
        }
        if paused {
            break;
        }
    }
    trim_session(&mut messages);
    send(
        sender,
        AssistantEvent::new(
            "done",
            json!({"result":Value::Null,"session_id":python_json(&Value::Array(messages))}),
        ),
    )
    .await
}

async fn send_provider_error(
    sender: &mpsc::Sender<AssistantEvent>,
    config: &Config,
    message: &str,
) -> Result<(), String> {
    send(
        sender,
        AssistantEvent::error(format!(
            "{} {}. {}",
            config.preset.display_name(),
            message,
            config.preset.connection_hint()
        )),
    )
    .await
}

async fn send(sender: &mpsc::Sender<AssistantEvent>, event: AssistantEvent) -> Result<(), String> {
    sender.send(event).await.map_err(|error| error.to_string())
}

async fn send_tool_result(
    sender: &mpsc::Sender<AssistantEvent>,
    id: &str,
    content: &str,
    is_error: bool,
) -> Result<(), String> {
    send(
        sender,
        AssistantEvent::new(
            "message",
            json!({"message":{"role":"user","content":[{"tool_use_id":id,"content":content,"is_error":is_error}]}}),
        ),
    )
    .await
}

async fn list_ollama_models(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut request = client.get(format!("{}/api/tags", base_url.trim_end_matches('/')));
    if let Some(api_key) = api_key.filter(|value| !value.is_empty()) {
        request = request.bearer_auth(api_key);
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    let response = response
        .error_for_status()
        .map_err(|error| error.to_string())?;
    let body: Value = response.json().await.map_err(|error| error.to_string())?;
    Ok(body
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| model.get("model").and_then(Value::as_str))
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect())
}

struct ModelTurn {
    visible_text: String,
    tool_calls: Vec<Value>,
}

async fn model_turn(
    client: &reqwest::Client,
    chat_url: &str,
    api_key: Option<&str>,
    model: &str,
    messages: &[Value],
    display_name: &str,
    sender: &mpsc::Sender<AssistantEvent>,
) -> Result<ModelTurn, String> {
    let mut request = client.post(chat_url).json(&json!({
        "model":model,
        "messages":messages,
        "tools":tools_schema(),
        "stream":true,
        "stream_options":{"include_usage":true},
    }));
    if let Some(api_key) = api_key.filter(|value| !value.is_empty()) {
        request = request.bearer_auth(api_key);
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    if response.status() != reqwest::StatusCode::OK {
        let status = response.status().as_u16();
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(format!("{display_name} error {status}: {body}"));
    }
    let mut bytes = response.bytes_stream();
    let mut line_buffer = Vec::new();
    let mut turn = ModelTurn {
        visible_text: String::new(),
        tool_calls: Vec::new(),
    };
    let mut think_buffer = String::new();
    let mut in_think = false;
    while let Some(chunk) = bytes.try_next().await.map_err(|error| error.to_string())? {
        line_buffer.extend_from_slice(&chunk);
        while let Some(index) = line_buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = line_buffer.drain(..=index).collect();
            process_line(
                &line,
                model,
                sender,
                &mut turn,
                &mut think_buffer,
                &mut in_think,
            )
            .await?;
        }
    }
    if !line_buffer.is_empty() {
        process_line(
            &line_buffer,
            model,
            sender,
            &mut turn,
            &mut think_buffer,
            &mut in_think,
        )
        .await?;
    }
    turn.tool_calls = turn
        .tool_calls
        .into_iter()
        .map(|call| {
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            json!({
                "id":id,
                "type":"function",
                "function":call.get("function").cloned().unwrap_or_else(|| json!({"name":"","arguments":""})),
            })
        })
        .collect();
    Ok(turn)
}

async fn process_line(
    raw: &[u8],
    model: &str,
    sender: &mpsc::Sender<AssistantEvent>,
    turn: &mut ModelTurn,
    think_buffer: &mut String,
    in_think: &mut bool,
) -> Result<(), String> {
    let mut line = std::str::from_utf8(raw)
        .map_err(|error| error.to_string())?
        .trim();
    if let Some(data) = line.strip_prefix("data:") {
        line = data.trim();
    }
    if line.is_empty() || line == "[DONE]" || line.starts_with(':') {
        return Ok(());
    }
    let Ok(chunk) = serde_json::from_str::<Value>(line) else {
        return Ok(());
    };
    if let Some(usage) = chunk.get("usage").filter(|usage| python_truthy(usage)) {
        send(sender, usage_event(usage, model)).await?;
    }
    let Some(delta) = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
    else {
        return Ok(());
    };
    if let Some(text) = delta
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        think_buffer.push_str(text);
        let (visible, remaining, state) = strip_think_blocks(think_buffer, *in_think);
        *think_buffer = remaining;
        *in_think = state;
        if !visible.is_empty() {
            turn.visible_text.push_str(&visible);
            send(
                sender,
                AssistantEvent::new(
                    "stream_event",
                    json!({"event":{"type":"content_delta","delta":{"text":visible}}}),
                ),
            )
            .await?;
        }
    }
    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            merge_tool_call(&mut turn.tool_calls, call);
        }
    }
    Ok(())
}

fn merge_tool_call(calls: &mut Vec<Value>, chunk: &Value) {
    let index = chunk.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    while calls.len() <= index {
        calls.push(json!({"id":"","function":{"name":"","arguments":""}}));
    }
    let call = calls[index].as_object_mut().expect("tool call object");
    if let Some(id) = chunk
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        call.insert("id".to_string(), Value::String(id.to_string()));
    }
    let function = call
        .get_mut("function")
        .and_then(Value::as_object_mut)
        .expect("function object");
    if let Some(name) = chunk
        .pointer("/function/name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    {
        function.insert("name".to_string(), Value::String(name.to_string()));
    }
    if let Some(arguments) = chunk
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .filter(|arguments| !arguments.is_empty())
    {
        let existing = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        function.insert(
            "arguments".to_string(),
            Value::String(format!("{existing}{arguments}")),
        );
    }
}

fn decode_session(session: Option<&str>) -> Vec<Value> {
    session
        .and_then(|session| serde_json::from_str::<Value>(session).ok())
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
}

fn pending_tool_calls(messages: &[Value]) -> Vec<Value> {
    let Some(calls) = messages.iter().rev().find_map(|message| {
        (message.get("role").and_then(Value::as_str) == Some("assistant"))
            .then(|| message.get("tool_calls").and_then(Value::as_array))
            .flatten()
    }) else {
        return Vec::new();
    };
    let resolved: Vec<&str> = messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("tool"))
        .filter_map(|message| message.get("tool_call_id").and_then(Value::as_str))
        .collect();
    calls
        .iter()
        .filter(|call| {
            call.get("id")
                .and_then(Value::as_str)
                .is_none_or(|id| !resolved.contains(&id))
        })
        .cloned()
        .collect()
}

fn trim_session(messages: &mut Vec<Value>) {
    while session_size(messages) > MAX_SESSION_BYTES && messages.len() > 2 {
        let mut end = 2;
        while end < messages.len()
            && messages[end].get("role").and_then(Value::as_str) != Some("user")
        {
            end += 1;
        }
        if end >= messages.len() {
            break;
        }
        messages.drain(1..end);
    }
}

fn session_size(messages: &[Value]) -> usize {
    2 + messages
        .iter()
        .map(|message| python_json(message).len())
        .sum::<usize>()
        + messages.len().saturating_sub(1) * 2
}

fn usage_event(usage: &Value, model: &str) -> AssistantEvent {
    let prompt = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64);
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64);
    AssistantEvent::new(
        "stream_event",
        json!({"event":{"type":"usage","usage":{
            "prompt_tokens":prompt,
            "completion_tokens":completion,
            "total_tokens":total,
            "total_cost_usd":token_cost(model,prompt,completion,cache_read,cache_creation),
        }}}),
    )
}

fn token_cost(
    model: &str,
    prompt: u64,
    completion: u64,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
) -> Option<f64> {
    if model.starts_with("gateway:/") || model.starts_with("endpoints:/") {
        return None;
    }
    let bare = model.split_once('/').map_or(model, |(_, bare)| bare);
    let accounting = supported_provider_names()
        .into_iter()
        .find_map(|provider| model_accounting(provider, bare))?;
    let input_rate = accounting
        .prices
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let output_rate = accounting
        .prices
        .get("output_cost_per_token")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let read = cache_read.unwrap_or(0);
    let creation = cache_creation.unwrap_or(0);
    let regular = prompt.saturating_sub(read).saturating_sub(creation);
    let read_rate = accounting
        .prices
        .get("cache_read_input_token_cost")
        .and_then(Value::as_f64)
        .unwrap_or(input_rate);
    let creation_rate = accounting
        .prices
        .get("cache_creation_input_token_cost")
        .and_then(Value::as_f64)
        .unwrap_or(input_rate);
    Some(
        regular as f64 * input_rate
            + read as f64 * read_rate
            + creation as f64 * creation_rate
            + completion as f64 * output_rate,
    )
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn trailing_partial_tag_len(buffer: &str, tag: &str) -> usize {
    let max = buffer.len().min(tag.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|length| tag.starts_with(&buffer[buffer.len() - length..]))
        .unwrap_or(0)
}

fn strip_think_blocks(mut buffer: &str, mut in_think: bool) -> (String, String, bool) {
    let mut visible = String::new();
    while !buffer.is_empty() {
        if in_think {
            let Some(end) = buffer.find("</think>") else {
                let hold = trailing_partial_tag_len(buffer, "</think>");
                return (
                    visible,
                    if hold == 0 {
                        String::new()
                    } else {
                        buffer[buffer.len() - hold..].to_string()
                    },
                    in_think,
                );
            };
            buffer = &buffer[end + "</think>".len()..];
            in_think = false;
        } else if let Some(start) = buffer.find("<think>") {
            visible.push_str(&buffer[..start]);
            buffer = &buffer[start + "<think>".len()..];
            in_think = true;
        } else {
            let hold = trailing_partial_tag_len(buffer, "<think>");
            if hold == 0 {
                visible.push_str(buffer);
                return (visible, String::new(), in_think);
            }
            visible.push_str(&buffer[..buffer.len() - hold]);
            return (visible, buffer[buffer.len() - hold..].to_string(), in_think);
        }
    }
    (visible, String::new(), in_think)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_complete_oldest_turn_group() {
        let big = "x".repeat(MAX_SESSION_BYTES / 3);
        let mut messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":format!("old-{big}")}),
            json!({"role":"assistant","content":format!("middle-{big}")}),
            json!({"role":"user","content":format!("new-{big}")}),
        ];
        trim_session(&mut messages);
        assert_eq!(messages[0]["role"], "system");
        assert!(messages.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .starts_with("new-"));
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn think_markers_split_across_frames_are_hidden() {
        let (first, remaining, state) = strip_think_blocks("foo<th", false);
        assert_eq!(first, "foo");
        let (second, remaining, state) =
            strip_think_blocks(&format!("{remaining}ink>secret</think>"), state);
        assert_eq!(
            (second, remaining, state),
            (String::new(), String::new(), false)
        );
    }
}
