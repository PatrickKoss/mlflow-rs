use std::cmp::Ordering;

use regex::RegexBuilder;
use serde_json::{json, Map, Value};

use crate::EngineError;

const SPAN_INPUTS: &str = "mlflow.spanInputs";
const SPAN_OUTPUTS: &str = "mlflow.spanOutputs";
const SPAN_TYPE: &str = "mlflow.spanType";
const CHAT_TOOLS: &str = "mlflow.chat.tools";

#[derive(Debug, Clone)]
pub(crate) struct TraceView<'a> {
    value: &'a Value,
}

impl<'a> TraceView<'a> {
    pub(crate) fn new(value: &'a Value) -> Self {
        Self { value }
    }

    pub(crate) fn root_span(&self) -> Option<&'a Value> {
        self.spans().iter().find(|span| parent_id(span).is_none())
    }

    pub(crate) fn spans(&self) -> &'a [Value] {
        self.value
            .pointer("/data/spans")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn inputs(&self) -> Option<Value> {
        self.root_span()
            .and_then(|span| span_attribute(span, SPAN_INPUTS))
    }

    pub(crate) fn outputs(&self) -> Option<Value> {
        self.root_span()
            .and_then(|span| span_attribute(span, SPAN_OUTPUTS))
    }

    pub(crate) fn request(&self) -> String {
        self.inputs()
            .as_ref()
            .map(parse_inputs_to_str)
            .unwrap_or_else(|| " ".to_string())
    }

    pub(crate) fn response(&self) -> String {
        self.outputs()
            .as_ref()
            .map(parse_outputs_to_str)
            .unwrap_or_else(|| " ".to_string())
    }

    pub(crate) fn retrieval_contexts(&self) -> Vec<(String, Vec<Value>)> {
        let spans = self.spans();
        spans
            .iter()
            .filter(|span| span_type(span).as_deref() == Some("RETRIEVER"))
            .filter(|span| !has_retriever_ancestor(span, spans))
            .filter_map(|span| {
                let span_id = span.get("span_id")?.as_str()?.to_string();
                let outputs = span_attribute(span, SPAN_OUTPUTS)?;
                let chunks = outputs
                    .as_array()?
                    .iter()
                    .filter_map(parse_chunk)
                    .collect::<Vec<_>>();
                Some((span_id, chunks))
            })
            .collect()
    }

    pub(crate) fn tools_called(&self) -> Vec<ToolCallRecord> {
        let mut spans = self
            .spans()
            .iter()
            .filter(|span| span_type(span).as_deref() == Some("TOOL"))
            .collect::<Vec<_>>();
        spans.sort_by_key(|span| {
            span.get("start_time_unix_nano")
                .and_then(Value::as_u64)
                .unwrap_or_default()
        });
        spans
            .into_iter()
            .map(|span| {
                let arguments = span_attribute(span, SPAN_INPUTS);
                let name = arguments
                    .as_ref()
                    .and_then(|inputs| inputs.pointer("/call/tool_name"))
                    .and_then(Value::as_str)
                    .or_else(|| span.get("name").and_then(Value::as_str))
                    .unwrap_or_default()
                    .to_string();
                ToolCallRecord {
                    name,
                    arguments,
                    outputs: span_attribute(span, SPAN_OUTPUTS),
                    exception: span
                        .get("events")
                        .and_then(Value::as_array)
                        .and_then(|events| events.iter().find_map(exception_event)),
                }
            })
            .collect()
    }

    pub(crate) fn available_tools(&self) -> Vec<Value> {
        let mut tools = Vec::new();
        for span in self
            .spans()
            .iter()
            .filter(|span| matches!(span_type(span).as_deref(), Some("LLM" | "CHAT_MODEL")))
        {
            let candidate = span_attribute(span, CHAT_TOOLS).or_else(|| {
                span_attribute(span, SPAN_INPUTS).and_then(|inputs| inputs.get("tools").cloned())
            });
            if let Some(Value::Array(values)) = candidate {
                for value in values {
                    if value.get("function").is_some() && !tools.contains(&value) {
                        tools.push(value);
                    }
                }
            }
        }
        tools
    }

    pub(crate) fn invoke_tool(&self, name: &str, arguments: &Value) -> Result<String, EngineError> {
        let result = match name {
            "get_trace_info" => self.trace_info(),
            "get_root_span" => match self.root_span() {
                Some(span) => self.span_result(span, arguments),
                None if self.spans().is_empty() => json!({
                    "span_id": null,
                    "content": null,
                    "content_size_bytes": 0,
                    "page_token": null,
                    "error": "Trace has no spans",
                }),
                None => json!({
                    "span_id": null,
                    "content": null,
                    "content_size_bytes": 0,
                    "page_token": null,
                    "error": "No root span found in trace",
                }),
            },
            "get_span" => {
                let span_id = arguments
                    .get("span_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| EngineError::Tool("get_span requires span_id".to_string()))?;
                match self
                    .spans()
                    .iter()
                    .find(|span| span.get("span_id").and_then(Value::as_str) == Some(span_id))
                {
                    Some(span) => self.span_result(span, arguments),
                    None if self.spans().is_empty() => json!({
                        "span_id": null,
                        "content": null,
                        "content_size_bytes": 0,
                        "page_token": null,
                        "error": "Trace has no spans",
                    }),
                    None => json!({
                        "span_id": null,
                        "content": null,
                        "content_size_bytes": 0,
                        "page_token": null,
                        "error": format!("Span with ID '{span_id}' not found in trace"),
                    }),
                }
            }
            "list_spans" => self.list_spans(arguments),
            "search_trace_regex" => self.search_regex(arguments)?,
            "get_span_performance_and_timing_report" => Value::String(self.performance_report()),
            other => {
                return Err(EngineError::Tool(format!(
                    "Tool '{other}' not found in registry"
                )))
            }
        };
        if let Value::String(value) = result {
            Ok(value)
        } else {
            serde_json::to_string(&result)
                .map_err(|error| EngineError::Serialization(error.to_string()))
        }
    }

    fn trace_info(&self) -> Value {
        self.value.get("info").cloned().unwrap_or(Value::Null)
    }

    fn span_result(&self, span: &Value, arguments: &Value) -> Value {
        let mut span = span.clone();
        if let Some(attributes) = arguments
            .get("attributes_to_fetch")
            .and_then(Value::as_array)
        {
            if let Some(current) = span.get_mut("attributes").and_then(Value::as_object_mut) {
                current.retain(|key, _| {
                    attributes
                        .iter()
                        .any(|attribute| attribute.as_str() == Some(key))
                });
            }
        }
        let content = serde_json::to_string_pretty(&span).unwrap_or_default();
        let total_size = content.len();
        let offset = arguments
            .get("page_token")
            .and_then(Value::as_str)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_default();
        let limit = arguments
            .get("max_content_length")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(100_000);
        let end = offset.saturating_add(limit).min(total_size);
        let chunk = content.get(offset..end).unwrap_or_default();
        json!({
            "span_id": span.get("span_id").cloned().unwrap_or(Value::Null),
            "content": chunk,
            "content_size_bytes": chunk.len(),
            "page_token": (end < total_size).then(|| end.to_string()),
            "error": null,
        })
    }

    fn list_spans(&self, arguments: &Value) -> Value {
        let offset = arguments
            .get("page_token")
            .and_then(Value::as_str)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_default();
        let limit = arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(100);
        let spans = self
            .spans()
            .iter()
            .skip(offset)
            .take(limit)
            .map(|span| {
                let start = span
                    .get("start_time_unix_nano")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                let end = span
                    .get("end_time_unix_nano")
                    .and_then(Value::as_u64)
                    .unwrap_or(start);
                let attribute_names: Vec<Value> = span
                    .get("attributes")
                    .and_then(Value::as_object)
                    .map(|attrs| attrs.keys().cloned().map(Value::String).collect())
                    .unwrap_or_default();
                json!({
                    "span_id": span.get("span_id").cloned().unwrap_or(Value::Null),
                    "name": span.get("name").cloned().unwrap_or(Value::Null),
                    "span_type": span_type(span),
                    "start_time_ms": start as f64 / 1_000_000.0,
                    "end_time_ms": end as f64 / 1_000_000.0,
                    "duration_ms": (end.saturating_sub(start)) as f64 / 1_000_000.0,
                    "parent_id": span.get("parent_span_id").or_else(|| span.get("parent_id")).cloned().unwrap_or(Value::Null),
                    "status": span.pointer("/status/code").cloned().unwrap_or(Value::Null),
                    "is_root": parent_id(span).is_none(),
                    "attribute_names": attribute_names,
                })
            })
            .collect::<Vec<_>>();
        let next = if offset + spans.len() < self.spans().len() {
            Value::String((offset + spans.len()).to_string())
        } else {
            Value::Null
        };
        json!({"spans": spans, "next_page_token": next})
    }

    fn search_regex(&self, arguments: &Value) -> Result<Value, EngineError> {
        let pattern = arguments
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| EngineError::Tool("search_trace_regex requires pattern".to_string()))?;
        let max_matches = arguments
            .get("max_matches")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(50);
        let surrounding = arguments
            .get("surrounding_content_length")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(100);
        let regex = match RegexBuilder::new(pattern).case_insensitive(true).build() {
            Ok(regex) => regex,
            Err(error) => {
                return Ok(json!({
                    "pattern": pattern,
                    "total_matches": 0,
                    "matches": [],
                    "error": format!("Invalid regex pattern: {error}"),
                }));
            }
        };
        let text = serde_json::to_string(self.value)
            .map_err(|error| EngineError::Serialization(error.to_string()))?;
        let matches = regex
            .find_iter(&text)
            .take(max_matches)
            .map(|found| {
                let start = found.start().saturating_sub(surrounding);
                let end = (found.end() + surrounding).min(text.len());
                let mut context = text[start..end].to_string();
                if start > 0 {
                    context.insert_str(0, "...");
                }
                if end < text.len() {
                    context.push_str("...");
                }
                json!({
                    "span_id": "trace",
                    "matched_text": found.as_str(),
                    "surrounding_text": context,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "pattern": pattern,
            "total_matches": matches.len(),
            "matches": matches,
        }))
    }

    fn performance_report(&self) -> String {
        if self.spans().is_empty() {
            return "No spans found in trace".to_string();
        }
        let mut spans = self
            .spans()
            .iter()
            .map(|span| {
                let start = span
                    .get("start_time_unix_nano")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                let end = span
                    .get("end_time_unix_nano")
                    .and_then(Value::as_u64)
                    .unwrap_or(start);
                (
                    span.get("name").and_then(Value::as_str).unwrap_or(""),
                    (end.saturating_sub(start)) as f64 / 1_000_000_000.0,
                )
            })
            .collect::<Vec<_>>();
        spans.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        let body = spans
            .iter()
            .enumerate()
            .map(|(index, (name, duration))| format!("{}. {}: {:.3}s", index + 1, name, duration))
            .collect::<Vec<_>>()
            .join("\n");
        format!("Span Performance and Timing Report\n\nLongest-running spans:\n{body}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCallRecord {
    pub(crate) name: String,
    pub(crate) arguments: Option<Value>,
    pub(crate) outputs: Option<Value>,
    pub(crate) exception: Option<String>,
}

pub(crate) fn conversation(
    session: &[Value],
    include_tool_calls: bool,
    include_timing: bool,
) -> Vec<Value> {
    let mut traces = session.iter().collect::<Vec<_>>();
    traces.sort_by_key(|trace| {
        trace
            .pointer("/info/request_time")
            .and_then(Value::as_str)
            .unwrap_or_default()
    });
    let mut messages = Vec::new();
    for trace in traces {
        let view = TraceView::new(trace);
        if let Some(inputs) = view.inputs() {
            let content = parse_inputs_to_str(&inputs);
            if !content.trim().is_empty() {
                messages.push(json!({"role": "user", "content": content}));
            }
        }
        if include_tool_calls {
            for tool in view.tools_called() {
                let mut content = format!("Tool: {}", tool.name);
                if let Some(arguments) = tool.arguments {
                    content.push_str(&format!("\nInputs: {}", python_str(&arguments)));
                }
                if let Some(outputs) = tool.outputs {
                    content.push_str(&format!("\nOutputs: {}", python_str(&outputs)));
                }
                if let Some(exception) = tool.exception {
                    content.push_str(&format!("\nException: {exception}"));
                }
                messages.push(json!({"role": "tool", "content": content}));
            }
        }
        if let Some(outputs) = view.outputs() {
            let mut content = parse_outputs_to_str(&outputs);
            if include_timing {
                if let Some(duration) = trace
                    .pointer("/info/execution_duration_ms")
                    .and_then(Value::as_f64)
                {
                    content.push_str(&format!("\n[Response duration: {:.2}s", duration / 1000.0));
                    if let Some(slowest) = slowest_spans(trace) {
                        content.push_str(", slowest spans: ");
                        content.push_str(&slowest);
                    }
                    content.push(']');
                }
            }
            if !content.trim().is_empty() {
                messages.push(json!({"role": "assistant", "content": content}));
            }
        }
    }
    messages
}

fn slowest_spans(trace: &Value) -> Option<String> {
    let spans = trace.pointer("/data/spans")?.as_array()?;
    let mut completed = spans
        .iter()
        .filter_map(|span| {
            let start = span_time(span.get("start_time_unix_nano")?)?;
            let end = span_time(span.get("end_time_unix_nano")?)?;
            Some((
                span.get("name").and_then(Value::as_str).unwrap_or(""),
                end - start,
            ))
        })
        .collect::<Vec<_>>();
    completed.sort_by(|left, right| right.1.cmp(&left.1));
    let formatted = completed
        .into_iter()
        .take(3)
        .map(|(name, duration)| format!("{name} ({:.2}s)", duration as f64 / 1_000_000_000.0))
        .collect::<Vec<_>>();
    (!formatted.is_empty()).then(|| formatted.join(", "))
}

fn span_time(value: &Value) -> Option<i128> {
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
        .or_else(|| value.as_str()?.parse().ok())
}

pub(crate) fn parse_inputs_to_str(value: &Value) -> String {
    match value {
        Value::Null => " ".to_string(),
        Value::String(value) => value.clone(),
        Value::Object(object) => {
            if let Some(messages) = object.get("messages").and_then(Value::as_array) {
                if !messages.is_empty() {
                    let contents = messages
                        .iter()
                        .map(|message| message.get("content").and_then(Value::as_str))
                        .collect::<Vec<_>>();
                    if contents.len() > 1 && contents.iter().all(|content| content.is_some()) {
                        return serde_json::to_string(messages).unwrap_or_default();
                    }
                    if let Some(Some(content)) = contents.last() {
                        return (*content).to_string();
                    }
                }
            }
            python_str(value)
        }
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

pub(crate) fn parse_outputs_to_str(value: &Value) -> String {
    match value {
        Value::Null => " ".to_string(),
        Value::String(value) => value.clone(),
        Value::Array(values) if !values.is_empty() => parse_outputs_to_str(&values[0]),
        Value::Object(object) => {
            if let Some(content) = value
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
            {
                return content.to_string();
            }
            if let Some(content) = object
                .get("messages")
                .and_then(Value::as_array)
                .and_then(|messages| messages.last())
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
            {
                return content.to_string();
            }
            serde_json::to_string(value).unwrap_or_default()
        }
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

pub(crate) fn python_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => format!("'{value}'"),
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(python_str).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("'{key}': {}", python_str(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn span_attribute(span: &Value, name: &str) -> Option<Value> {
    let value = span.pointer(&format!("/attributes/{name}"))?;
    match value {
        Value::String(value) => serde_json::from_str(value)
            .ok()
            .or_else(|| Some(Value::String(value.clone()))),
        value => Some(value.clone()),
    }
}

fn span_type(span: &Value) -> Option<String> {
    span_attribute(span, SPAN_TYPE).and_then(|value| value.as_str().map(str::to_string))
}

fn has_retriever_ancestor(span: &Value, spans: &[Value]) -> bool {
    let mut parent = parent_id(span);
    while let Some(current_parent_id) = parent {
        let Some(parent_span) = spans.iter().find(|candidate| {
            candidate.get("span_id").and_then(Value::as_str) == Some(current_parent_id)
        }) else {
            return false;
        };
        if span_type(parent_span).as_deref() == Some("RETRIEVER") {
            return true;
        }
        parent = parent_id(parent_span);
    }
    false
}

fn parent_id(span: &Value) -> Option<&str> {
    span.get("parent_span_id")
        .or_else(|| span.get("parent_id"))
        .and_then(Value::as_str)
}

fn parse_chunk(chunk: &Value) -> Option<Value> {
    let object = chunk.as_object()?;
    let content = ["page_content", "content", "text"]
        .into_iter()
        .find_map(|key| object.get(key))
        .cloned()
        .unwrap_or(Value::Null);
    let mut parsed = Map::new();
    parsed.insert("content".to_string(), content);
    if let Some(doc_uri) = object
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("doc_uri"))
        .filter(|value| !value.is_null())
    {
        parsed.insert("doc_uri".to_string(), doc_uri.clone());
    }
    Some(Value::Object(parsed))
}

fn exception_event(event: &Value) -> Option<String> {
    if event.get("name").and_then(Value::as_str) != Some("exception") {
        return None;
    }
    let attributes = event.get("attributes").and_then(Value::as_object)?;
    let kind = attributes
        .get("exception.type")
        .and_then(Value::as_str)
        .unwrap_or("Exception");
    Some(
        match attributes.get("exception.message").and_then(Value::as_str) {
            Some(message) => format!("{kind}: {message}"),
            None => kind.to_string(),
        },
    )
}
