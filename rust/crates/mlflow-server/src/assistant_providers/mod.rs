//! Native subprocess providers for the MLflow Assistant.
//!
//! This module deliberately has no dependency on the Assistant HTTP routes or
//! session store. Routes provide configuration and persist [`ProviderHandle::pid`]
//! if cross-request cancellation is needed; the provider owns argv construction,
//! environment setup, NDJSON translation, process lifetime, and health probes.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{ExitStatus, Stdio};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Number, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

use crate::gateway_provider_matrix::model_accounting;

const CLAUDE_SYSTEM_PROMPT: &str = include_str!("claude_system_prompt.txt");
const CODEX_SYSTEM_PROMPT: &str = include_str!("codex_system_prompt.txt");
const STREAM_LINE_LIMIT: usize = 100 * 1024 * 1024;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

const BASE_ALLOWED_TOOLS: &[&str] = &["Bash(mlflow:*)", "Skill"];
const FILE_EDIT_TOOLS: &[&str] = &[
    "Edit(*)",
    "Read(*)",
    "Write(*)",
    "Edit(//tmp/**)",
    "Read(//tmp/**)",
    "Write(//tmp/**)",
];
const DOCS_TOOLS: &[&str] = &["WebFetch(domain:mlflow.org)"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    ClaudeCode,
    Codex,
}

impl ProviderKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude_code",
            Self::Codex => "codex",
        }
    }

    const fn binary(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude",
            Self::Codex => "codex",
        }
    }

    fn stream_not_found_message(self) -> &'static str {
        match self {
            Self::ClaudeCode => {
                "Claude CLI not found. Please install Claude Code CLI and ensure it's in your PATH."
            }
            Self::Codex => {
                "codex CLI not found. Please install the OpenAI Codex CLI and ensure it's in your PATH."
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionsConfig {
    pub allow_edit_files: bool,
    pub allow_read_docs: bool,
    pub full_access: bool,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            allow_edit_files: true,
            allow_read_docs: true,
            full_access: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub model: String,
    #[serde(default)]
    pub permissions: PermissionsConfig,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            model: "default".to_string(),
            permissions: PermissionsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRequest {
    pub prompt: String,
    pub tracking_uri: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub context: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub stdin: Option<Vec<u8>>,
    pub environment: BTreeMap<OsString, OsString>,
    pub cwd: Option<PathBuf>,
}

impl Invocation {
    pub fn argv(&self) -> Vec<OsString> {
        std::iter::once(self.program.as_os_str().to_owned())
            .chain(self.args.iter().cloned())
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    Message,
    StreamEvent,
    Done,
    Error,
    Interrupted,
    PermissionRequest,
}

impl EventType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::StreamEvent => "stream_event",
            Self::Done => "done",
            Self::Error => "error",
            Self::Interrupted => "interrupted",
            Self::PermissionRequest => "permission_request",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    #[serde(rename = "type")]
    pub event_type: EventType,
    pub data: Value,
}

impl Event {
    pub fn from_error(error: impl Into<String>) -> Self {
        Self {
            event_type: EventType::Error,
            data: json!({"error": error.into()}),
        }
    }

    pub fn from_message(role: &str, content: Value) -> Self {
        Self {
            event_type: EventType::Message,
            data: json!({"message": {"role": role, "content": content}}),
        }
    }

    pub fn from_stream_event(event: Value) -> Self {
        Self {
            event_type: EventType::StreamEvent,
            data: json!({"event": event}),
        }
    }

    pub fn from_result(result: Value, session_id: impl Into<String>) -> Self {
        Self {
            event_type: EventType::Done,
            data: json!({"result": result, "session_id": session_id.into()}),
        }
    }

    pub fn from_interrupted() -> Self {
        Self {
            event_type: EventType::Interrupted,
            data: json!({"message": "Assistant was interrupted"}),
        }
    }

    /// Python's `Event.to_sse_event` uses default `json.dumps` formatting:
    /// spaces after separators and `ensure_ascii=True`.
    pub fn to_sse_frame(&self) -> String {
        format!(
            "event: {}\ndata: {}\n\n",
            self.event_type.as_str(),
            python_json(&self.data)
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("{0}")]
    CliNotFound(&'static str),
    #[error("{0}")]
    Io(#[from] io::Error),
}

impl SpawnError {
    fn into_event(self) -> Event {
        Event::from_error(self.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthError {
    NotImplemented(String),
    CliNotInstalled(String),
    NotAuthenticated(String),
}

impl HealthError {
    pub const fn status_code(&self) -> u16 {
        match self {
            Self::NotImplemented(_) => 501,
            Self::CliNotInstalled(_) => 412,
            Self::NotAuthenticated(_) => 401,
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Self::NotImplemented(detail)
            | Self::CliNotInstalled(detail)
            | Self::NotAuthenticated(detail) => detail,
        }
    }

    pub fn body(&self) -> Value {
        json!({"detail": self.detail()})
    }
}

#[derive(Debug, Clone)]
pub struct ProviderHandle {
    pid: u32,
}

impl ProviderHandle {
    pub const fn pid(&self) -> u32 {
        self.pid
    }

    /// Match `terminate_session_process`: send SIGTERM and let the provider
    /// translate the resulting process exit after stdout closes.
    #[cfg(unix)]
    pub fn cancel(&self) -> io::Result<()> {
        // SAFETY: `pid` comes directly from `tokio::process::Child::id`; no
        // pointer crosses the FFI boundary, and `kill` only reads its scalars.
        let result = unsafe { libc::kill(self.pid as libc::pid_t, libc::SIGTERM) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    #[cfg(not(unix))]
    pub fn cancel(&self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "assistant CLI cancellation requires Unix signals",
        ))
    }
}

pub struct ProviderEventStream {
    receiver: mpsc::Receiver<Event>,
    shutdown: Option<oneshot::Sender<()>>,
}

impl ProviderEventStream {
    pub async fn next(&mut self) -> Option<Event> {
        self.receiver.recv().await
    }

    fn single(event: Event) -> Self {
        let (sender, receiver) = mpsc::channel(1);
        let _ = sender.try_send(event);
        Self {
            receiver,
            shutdown: None,
        }
    }
}

impl Stream for ProviderEventStream {
    type Item = Event;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

impl Drop for ProviderEventStream {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

pub struct SpawnedProvider {
    pub handle: ProviderHandle,
    pub events: ProviderEventStream,
}

/// Build the complete subprocess invocation without spawning it. This is the
/// byte-level argv seam used by tests and by [`spawn`].
pub fn build_invocation(
    provider: ProviderKind,
    executable: impl Into<PathBuf>,
    config: &ProviderConfig,
    request: &StreamRequest,
) -> Invocation {
    let executable = executable.into();
    match provider {
        ProviderKind::ClaudeCode => build_claude_invocation(executable, config, request),
        ProviderKind::Codex => build_codex_invocation(executable, config, request),
    }
}

pub async fn spawn(
    provider: ProviderKind,
    config: ProviderConfig,
    request: StreamRequest,
) -> Result<SpawnedProvider, SpawnError> {
    let executable = find_executable(provider.binary())
        .ok_or_else(|| SpawnError::CliNotFound(provider.stream_not_found_message()))?;
    let invocation = build_invocation(provider, executable, &config, &request);
    let mut command = Command::new(&invocation.program);
    command
        .args(&invocation.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if invocation.stdin.is_some() {
        command.stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }
    if let Some(cwd) = &invocation.cwd {
        command.current_dir(cwd);
    }
    for (key, value) in &invocation.environment {
        command.env(key, value);
    }

    let mut child = command.spawn()?;
    let pid = child
        .id()
        .ok_or_else(|| io::Error::other("assistant CLI process has no PID"))?;
    if let Some(stdin_bytes) = invocation.stdin {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("assistant CLI stdin pipe was not created"))?;
        stdin.write_all(&stdin_bytes).await?;
        stdin.shutdown().await?;
    }

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("assistant CLI stdout pipe was not created"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("assistant CLI stderr pipe was not created"))?;
    let (sender, receiver) = mpsc::channel(32);
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    tokio::spawn(run_process(
        provider,
        config,
        child,
        stdout,
        stderr,
        sender,
        shutdown_receiver,
    ));

    Ok(SpawnedProvider {
        handle: ProviderHandle { pid },
        events: ProviderEventStream {
            receiver,
            shutdown: Some(shutdown_sender),
        },
    })
}

/// Convenience API for routes that only need the event stream. Spawn failures
/// become the same in-band `error` event yielded by Python's async generator.
pub async fn stream(
    provider: ProviderKind,
    config: ProviderConfig,
    request: StreamRequest,
) -> ProviderEventStream {
    match spawn(provider, config, request).await {
        Ok(spawned) => spawned.events,
        Err(error) => ProviderEventStream::single(error.into_event()),
    }
}

/// Probe CLI installation/authentication. Route code maps [`HealthError`] via
/// `status_code`, `detail`, or `body`; the values reproduce Python's
/// NotImplemented→501, CLINotInstalled→412, NotAuthenticated→401 contract.
pub async fn health(provider: ProviderKind) -> Result<(), HealthError> {
    let executable = find_executable(provider.binary()).ok_or_else(|| {
        HealthError::CliNotInstalled(match provider {
            ProviderKind::ClaudeCode => concat!(
                "Claude Code CLI is not installed. ",
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )
            .to_string(),
            ProviderKind::Codex => concat!(
                "OpenAI Codex CLI is not installed. ",
                "Install it with: npm install -g @openai/codex"
            )
            .to_string(),
        })
    })?;

    match provider {
        ProviderKind::ClaudeCode => probe_claude_health().await,
        ProviderKind::Codex => probe_codex_health(executable).await,
    }
}

fn build_claude_invocation(
    executable: PathBuf,
    config: &ProviderConfig,
    request: &StreamRequest,
) -> Invocation {
    let user_message = message_with_context(&request.prompt, request.context.as_ref());
    let system_prompt = format_system_prompt(CLAUDE_SYSTEM_PROMPT, &request.tracking_uri);
    let mut args = vec![
        OsString::from("-p"),
        OsString::from(user_message),
        OsString::from("--output-format"),
        OsString::from("stream-json"),
        OsString::from("--verbose"),
        OsString::from("--append-system-prompt"),
        OsString::from(system_prompt),
    ];

    if config.permissions.full_access {
        push_pair(&mut args, "--permission-mode", "bypassPermissions");
    } else {
        for tool in BASE_ALLOWED_TOOLS
            .iter()
            .chain(
                config
                    .permissions
                    .allow_edit_files
                    .then_some(FILE_EDIT_TOOLS)
                    .into_iter()
                    .flatten(),
            )
            .chain(
                config
                    .permissions
                    .allow_read_docs
                    .then_some(DOCS_TOOLS)
                    .into_iter()
                    .flatten(),
            )
        {
            push_pair(&mut args, "--allowed-tools", tool);
        }
    }
    if !config.model.is_empty() && config.model != "default" {
        push_pair(&mut args, "--model", &config.model);
    }
    if let Some(session_id) = request
        .session_id
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        push_pair(&mut args, "--resume", session_id);
    }

    Invocation {
        program: executable,
        args,
        stdin: None,
        environment: tracking_environment(&request.tracking_uri),
        cwd: request.cwd.clone(),
    }
}

fn build_codex_invocation(
    executable: PathBuf,
    config: &ProviderConfig,
    request: &StreamRequest,
) -> Invocation {
    let user_text = message_with_context(&request.prompt, request.context.as_ref());
    let user_message = if request
        .session_id
        .as_deref()
        .is_some_and(|id| !id.is_empty())
    {
        user_text
    } else {
        let system_prompt = format_system_prompt(CODEX_SYSTEM_PROMPT, &request.tracking_uri);
        format!("<system_instructions>\n{system_prompt}\n</system_instructions>\n\n{user_text}")
    };
    let mut args = vec![
        OsString::from("exec"),
        OsString::from("--json"),
        OsString::from("--sandbox"),
        OsString::from("danger-full-access"),
        OsString::from("--skip-git-repo-check"),
    ];
    if !config.model.is_empty() && config.model != "default" {
        push_pair(&mut args, "-m", &config.model);
    }
    if let Some(session_id) = request
        .session_id
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        args.push(OsString::from("resume"));
        args.push(OsString::from(session_id));
    }
    args.push(OsString::from("-"));

    Invocation {
        program: executable,
        args,
        stdin: Some(user_message.into_bytes()),
        environment: tracking_environment(&request.tracking_uri),
        cwd: request.cwd.clone(),
    }
}

fn tracking_environment(tracking_uri: &str) -> BTreeMap<OsString, OsString> {
    BTreeMap::from([(
        OsString::from("MLFLOW_TRACKING_URI"),
        OsString::from(tracking_uri),
    )])
}

fn push_pair(args: &mut Vec<OsString>, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
    args.push(key.as_ref().to_owned());
    args.push(value.as_ref().to_owned());
}

fn message_with_context(prompt: &str, context: Option<&Map<String, Value>>) -> String {
    match context.filter(|context| !context.is_empty()) {
        Some(context) => format!(
            "<context>\n{}\n</context>\n\n{prompt}",
            python_json(&Value::Object(context.clone()))
        ),
        None => prompt.to_string(),
    }
}

pub(crate) fn format_system_prompt(template: &str, tracking_uri: &str) -> String {
    // Both Python constants are passed through `str.format`. In addition to
    // replacing the named field, that turns literal `{{` / `}}` examples back
    // into single braces.
    template
        .replace("{tracking_uri}", tracking_uri)
        .replace("{{", "{")
        .replace("}}", "}")
}

async fn run_process<R, E>(
    provider: ProviderKind,
    config: ProviderConfig,
    mut child: Child,
    stdout: R,
    mut stderr: E,
    sender: mpsc::Sender<Event>,
    mut shutdown: oneshot::Receiver<()>,
) where
    R: AsyncRead + Unpin,
    E: AsyncRead + Unpin,
{
    let mut reader = BufReader::new(stdout);
    let mut thread_id = String::new();
    let mut buffer = Vec::new();
    let mut stream_error = None;

    loop {
        buffer.clear();
        tokio::select! {
            _ = &mut shutdown => {
                let _ = child.kill().await;
                return;
            }
            read = reader.read_until(b'\n', &mut buffer) => {
                match read {
                    Ok(0) => break,
                    Ok(_) if buffer.len() > STREAM_LINE_LIMIT => {
                        stream_error = Some(format!(
                            "assistant CLI output line exceeds {STREAM_LINE_LIMIT} byte limit"
                        ));
                        break;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        stream_error = Some(error.to_string());
                        break;
                    }
                }
            }
        }

        let line = match std::str::from_utf8(&buffer) {
            Ok(line) => line.trim(),
            Err(error) => {
                stream_error = Some(error.to_string());
                break;
            }
        };
        if line.is_empty() {
            continue;
        }

        let events = match provider {
            ProviderKind::ClaudeCode => parse_claude_line(line),
            ProviderKind::Codex => parse_codex_line(line, &config, &mut thread_id),
        };
        for event in events {
            if sender.send(event).await.is_err() {
                let _ = child.kill().await;
                return;
            }
        }
    }

    if let Some(error) = stream_error {
        let _ = child.kill().await;
        let _ = sender.send(Event::from_error(error)).await;
        return;
    }

    let status = match child.wait().await {
        Ok(status) => status,
        Err(error) => {
            let _ = sender.send(Event::from_error(error.to_string())).await;
            return;
        }
    };
    if exit_code(&status) == -9 {
        let _ = sender.send(Event::from_interrupted()).await;
        return;
    }
    if !status.success() {
        let mut stderr_bytes = Vec::new();
        let _ = stderr.read_to_end(&mut stderr_bytes).await;
        let detail = String::from_utf8_lossy(&stderr_bytes).trim().to_string();
        let detail = if detail.is_empty() {
            format!("Process exited with code {}", exit_code(&status))
        } else {
            detail
        };
        let _ = sender.send(Event::from_error(detail)).await;
    } else if provider == ProviderKind::Codex {
        let _ = sender
            .send(Event::from_result(Value::Null, thread_id))
            .await;
    }
}

fn parse_claude_line(line: &str) -> Vec<Event> {
    let data: Value = match serde_json::from_str(line) {
        Ok(data) => data,
        Err(_) => return vec![Event::from_message("user", Value::String(line.to_string()))],
    };
    if should_filter_claude_message(&data) {
        return Vec::new();
    }

    let mut events = Vec::with_capacity(2);
    if data.get("type").and_then(Value::as_str) == Some("result") {
        if let Some(usage) = data.get("usage").filter(|value| python_truthy(value)) {
            events.push(claude_usage_event(
                usage,
                data.get("total_cost_usd").cloned().unwrap_or(Value::Null),
            ));
        }
    }
    if let Some(event) = parse_claude_message(&data) {
        events.push(event);
    }
    events
}

fn should_filter_claude_message(data: &Value) -> bool {
    if data.get("type").and_then(Value::as_str) != Some("user") {
        return false;
    }
    data.pointer("/message/content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|block| {
                block.get("type").and_then(Value::as_str) == Some("text")
                    && block
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.starts_with("Base directory for this skill:"))
            })
        })
}

fn parse_claude_message(data: &Value) -> Option<Event> {
    let Some(message_type) = data.get("type").and_then(Value::as_str) else {
        return Some(Event::from_error("Message missing 'type' field"));
    };
    match message_type {
        "user" => Some(parse_claude_chat_message(data, "user")),
        "assistant" => {
            if let Some(error) = data
                .pointer("/message/error")
                .filter(|value| python_truthy(value))
            {
                return Some(Event::from_error(value_as_string(error)));
            }
            Some(parse_claude_chat_message(data, "assistant"))
        }
        "system" => None,
        "error" => {
            let error = data.get("error").cloned().unwrap_or(Value::Null);
            let detail = error
                .get("message")
                .map(value_as_string)
                .unwrap_or_else(|| python_string(&error));
            Some(Event::from_error(detail))
        }
        "result" => match data.get("session_id").and_then(Value::as_str) {
            Some(session_id) => Some(Event::from_result(
                data.get("result").cloned().unwrap_or(Value::Null),
                session_id,
            )),
            None => Some(Event::from_error(
                "Failed to parse result message: 'session_id'",
            )),
        },
        "stream_event" => match data.get("event") {
            Some(event) => Some(Event::from_stream_event(event.clone())),
            None => Some(Event::from_error(
                "Failed to parse stream_event message: 'event'",
            )),
        },
        "rate_limit_event" => parse_rate_limit_event(data),
        _ => None,
    }
}

fn parse_claude_chat_message(data: &Value, role: &str) -> Event {
    let Some(content) = data.pointer("/message/content") else {
        return Event::from_error(format!("Failed to parse {role} message: 'message'"));
    };
    let Some(blocks) = content.as_array() else {
        if role == "user" {
            return Event::from_message(role, content.clone());
        }
        return Event::from_error("Failed to parse assistant message: 'content'");
    };
    let mut parsed = Vec::new();
    for block in blocks {
        let Some(block_type) = block.get("type").and_then(Value::as_str) else {
            continue;
        };
        let value = match block_type {
            "text" => required_block(block, &["text"]),
            "thinking" if role == "assistant" => required_block(block, &["thinking", "signature"]),
            "tool_use" => required_block(block, &["id", "name", "input"]),
            "tool_result" => {
                let Some(tool_use_id) = block.get("tool_use_id") else {
                    return Event::from_error(format!(
                        "Failed to parse {role} message: 'tool_use_id'"
                    ));
                };
                Some(json!({
                    "tool_use_id": tool_use_id,
                    "content": block.get("content").cloned().unwrap_or(Value::Null),
                    "is_error": block.get("is_error").cloned().unwrap_or(Value::Null),
                }))
            }
            _ => continue,
        };
        match value {
            Some(value) => parsed.push(value),
            None => {
                let missing = match block_type {
                    "text" => "text",
                    "thinking" => ["thinking", "signature"]
                        .into_iter()
                        .find(|key| block.get(key).is_none())
                        .unwrap_or("thinking"),
                    "tool_use" => ["id", "name", "input"]
                        .into_iter()
                        .find(|key| block.get(key).is_none())
                        .unwrap_or("id"),
                    _ => block_type,
                };
                return Event::from_error(format!("Failed to parse {role} message: '{missing}'"));
            }
        }
    }
    Event::from_message(role, Value::Array(parsed))
}

fn required_block(block: &Value, keys: &[&str]) -> Option<Value> {
    let mut output = Map::new();
    for key in keys {
        output.insert((*key).to_string(), block.get(*key)?.clone());
    }
    Some(Value::Object(output))
}

fn parse_rate_limit_event(data: &Value) -> Option<Event> {
    let info = data.get("rate_limit_info")?;
    if info.get("status").and_then(Value::as_str) != Some("limited") {
        return None;
    }
    let mut message = "You've hit a rate limit — please wait a moment and try again.".to_string();
    if let Some(resets_at) = info.get("resetsAt").filter(|value| python_truthy(value)) {
        message.push_str(" Your limit resets at ");
        message.push_str(&value_as_string(resets_at));
        message.push('.');
    }
    Some(Event::from_message("assistant", json!([{"text": message}])))
}

fn claude_usage_event(usage: &Value, cost: Value) -> Event {
    let prompt_tokens = numeric_or_zero(usage.get("input_tokens"))
        + numeric_or_zero(usage.get("cache_creation_input_tokens"))
        + numeric_or_zero(usage.get("cache_read_input_tokens"));
    let completion_tokens = numeric_or_zero(usage.get("output_tokens"));
    Event::from_stream_event(json!({
        "type": "usage",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": cost,
        }
    }))
}

fn parse_codex_line(line: &str, config: &ProviderConfig, thread_id: &mut String) -> Vec<Event> {
    let data: Value = match serde_json::from_str(line) {
        Ok(data) => data,
        Err(_) => return Vec::new(),
    };
    match data.get("type").and_then(Value::as_str) {
        Some("thread.started") => {
            *thread_id = data
                .get("thread_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Vec::new()
        }
        Some("turn.completed") => data
            .get("usage")
            .filter(|value| python_truthy(value))
            .map(|usage| {
                let model = (!config.model.is_empty() && config.model != "default")
                    .then_some(config.model.as_str());
                vec![codex_usage_event(usage, model)]
            })
            .unwrap_or_default(),
        Some("item.completed") => {
            let item = data.get("item").unwrap_or(&Value::Null);
            if item.get("type").and_then(Value::as_str) != Some("agent_message") {
                return Vec::new();
            }
            item.get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| vec![Event::from_message("assistant", json!([{"text": text}]))])
                .unwrap_or_default()
        }
        _ => Vec::new(),
    }
}

fn codex_usage_event(usage: &Value, model: Option<&str>) -> Event {
    let prompt_tokens = numeric_or_zero(usage.get("input_tokens"));
    let completion_tokens = numeric_or_zero(usage.get("output_tokens"));
    let cached_tokens = usage.get("cached_input_tokens").and_then(Value::as_u64);
    let total_cost = model.and_then(|model| {
        codex_token_cost(
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens.unwrap_or(0),
        )
    });
    Event::from_stream_event(json!({
        "type": "usage",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": total_cost,
        }
    }))
}

fn codex_token_cost(
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
) -> Option<f64> {
    let accounting = model_accounting("openai", model)?;
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
    let cache_rate = accounting
        .prices
        .get("cache_read_input_token_cost")
        .and_then(Value::as_f64)
        .unwrap_or(input_rate);
    let regular_tokens = prompt_tokens.saturating_sub(cached_tokens);
    let input_cost = regular_tokens as f64 * input_rate + cached_tokens as f64 * cache_rate;
    let output_cost = completion_tokens as f64 * output_rate;
    Some(input_cost + output_cost)
}

async fn probe_claude_health() -> Result<(), HealthError> {
    // Python deliberately invokes the literal `claude` after `which`, while
    // Codex invokes the resolved path.
    let mut command = Command::new("claude");
    command
        .args(["-p", "hi", "--max-turns", "1", "--output-format", "json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_health_command(command, None, "Authentication check timed out").await?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lowered = stderr.to_lowercase();
    let detail = if ["auth", "login", "unauthorized"]
        .iter()
        .any(|needle| lowered.contains(needle))
    {
        "Not authenticated. Please run: claude login".to_string()
    } else if stderr.trim().is_empty() {
        format!("Process exited with code {}", exit_code(&output.status))
    } else {
        stderr.trim().to_string()
    };
    Err(HealthError::NotAuthenticated(detail))
}

async fn probe_codex_health(executable: PathBuf) -> Result<(), HealthError> {
    let mut command = Command::new(executable);
    command
        .args([
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            "--skip-git-repo-check",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_health_command(command, Some(b"say hi"), "Connection check timed out").await?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lowered = stderr.to_lowercase();
    let detail = if ["auth", "login", "unauthorized", "api key"]
        .iter()
        .any(|needle| lowered.contains(needle))
    {
        "Not authenticated. Please set OPENAI_API_KEY or run: codex login".to_string()
    } else if stderr.trim().is_empty() {
        format!("Process exited with code {}", exit_code(&output.status))
    } else {
        stderr.trim().to_string()
    };
    Err(HealthError::NotAuthenticated(detail))
}

async fn run_health_command(
    mut command: Command,
    input: Option<&[u8]>,
    timeout_message: &'static str,
) -> Result<std::process::Output, HealthError> {
    // `subprocess.run(..., timeout=30)` kills and reaps the Python child on
    // timeout. Make dropping Tokio's timed-out wait future do the same.
    command.kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|error| HealthError::NotAuthenticated(error.to_string()))?;
    if let Some(input) = input {
        let mut stdin = child.stdin.take().ok_or_else(|| {
            HealthError::NotAuthenticated("assistant CLI stdin pipe was not created".to_string())
        })?;
        stdin
            .write_all(input)
            .await
            .map_err(|error| HealthError::NotAuthenticated(error.to_string()))?;
        stdin
            .shutdown()
            .await
            .map_err(|error| HealthError::NotAuthenticated(error.to_string()))?;
    }
    match timeout(HEALTH_TIMEOUT, child.wait_with_output()).await {
        Ok(result) => result.map_err(|error| HealthError::NotAuthenticated(error.to_string())),
        Err(_) => Err(HealthError::NotAuthenticated(timeout_message.to_string())),
    }
}

fn find_executable(binary: &str) -> Option<PathBuf> {
    let candidate = Path::new(binary);
    if candidate.components().count() > 1 {
        return is_executable(candidate).then(|| candidate.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(binary))
        .find(|candidate| is_executable(candidate))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(unix)]
fn exit_code(status: &ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;

    status
        .code()
        .or_else(|| status.signal().map(|signal| -signal))
        .unwrap_or(-1)
}

#[cfg(not(unix))]
fn exit_code(status: &ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

fn numeric_or_zero(value: Option<&Value>) -> u64 {
    value.and_then(Value::as_u64).unwrap_or(0)
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

fn value_as_string(value: &Value) -> String {
    value
        .as_str()
        .map(ToString::to_string)
        .unwrap_or_else(|| python_string(value))
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(value) => value.clone(),
        _ => python_json(value),
    }
}

/// Serialize the JSON subset used by Assistant events like Python's
/// `json.dumps` defaults (`ensure_ascii=True`, `separators=(", ", ": ")`).
pub(crate) fn python_json(value: &Value) -> String {
    let mut output = String::new();
    write_python_json(value, &mut output);
    output
}

fn write_python_json(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&python_number(number)),
        Value::String(value) => write_python_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write_python_json(value, output);
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write_python_string(key, output);
                output.push_str(": ");
                write_python_json(value, output);
            }
            output.push('}');
        }
    }
}

fn write_python_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character if character.is_ascii() => output.push(character),
            character => {
                let codepoint = character as u32;
                if codepoint <= 0xffff {
                    output.push_str(&format!("\\u{codepoint:04x}"));
                } else {
                    let adjusted = codepoint - 0x1_0000;
                    let high = 0xd800 + (adjusted >> 10);
                    let low = 0xdc00 + (adjusted & 0x3ff);
                    output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    output.push('"');
}

fn python_number(number: &Number) -> String {
    if !number.is_f64() {
        return number.to_string();
    }
    let raw = number.to_string();
    let Ok(value) = raw.parse::<f64>() else {
        return raw;
    };
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let negative = value.is_sign_negative();
    let raw = raw.trim_start_matches('-');
    let (digits, decimal_exponent) = decimal_digits(raw);
    let scientific_exponent = decimal_exponent + digits.len() as i32 - 1;
    let mut rendered = if !(-4..16).contains(&scientific_exponent) {
        let mut rendered = String::new();
        rendered.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            rendered.push('.');
            rendered.push_str(&digits[1..]);
        }
        rendered.push('e');
        rendered.push(if scientific_exponent < 0 { '-' } else { '+' });
        rendered.push_str(&format!("{:02}", scientific_exponent.unsigned_abs()));
        rendered
    } else {
        let decimal_position = digits.len() as i32 + decimal_exponent;
        if decimal_position <= 0 {
            format!("0.{}{}", "0".repeat((-decimal_position) as usize), digits)
        } else if decimal_position < digits.len() as i32 {
            let split = decimal_position as usize;
            format!("{}.{}", &digits[..split], &digits[split..])
        } else {
            let mut rendered = digits;
            rendered
                .push_str(&"0".repeat((decimal_position as usize).saturating_sub(rendered.len())));
            rendered.push_str(".0");
            rendered
        }
    };
    if negative {
        rendered.insert(0, '-');
    }
    rendered
}

fn decimal_digits(raw: &str) -> (String, i32) {
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| (mantissa, exponent.parse::<i32>().unwrap_or(0)))
        .unwrap_or((raw, 0));
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let untrimmed = format!("{integer}{fraction}");
    let digits = untrimmed.trim_start_matches('0').to_string();
    (digits, exponent - fraction.len() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> StreamRequest {
        StreamRequest {
            prompt: "hello".to_string(),
            tracking_uri: "http://127.0.0.1:5000".to_string(),
            session_id: None,
            cwd: None,
            context: None,
        }
    }

    fn argv_strings(invocation: &Invocation) -> Vec<String> {
        invocation
            .argv()
            .into_iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn claude_restricted_permission_flag_table_is_exact() {
        let cases = [
            (
                PermissionsConfig {
                    allow_edit_files: false,
                    allow_read_docs: false,
                    full_access: false,
                },
                vec!["Bash(mlflow:*)", "Skill"],
            ),
            (
                PermissionsConfig {
                    allow_edit_files: true,
                    allow_read_docs: false,
                    full_access: false,
                },
                vec![
                    "Bash(mlflow:*)",
                    "Skill",
                    "Edit(*)",
                    "Read(*)",
                    "Write(*)",
                    "Edit(//tmp/**)",
                    "Read(//tmp/**)",
                    "Write(//tmp/**)",
                ],
            ),
            (
                PermissionsConfig::default(),
                vec![
                    "Bash(mlflow:*)",
                    "Skill",
                    "Edit(*)",
                    "Read(*)",
                    "Write(*)",
                    "Edit(//tmp/**)",
                    "Read(//tmp/**)",
                    "Write(//tmp/**)",
                    "WebFetch(domain:mlflow.org)",
                ],
            ),
        ];
        for (permissions, expected) in cases {
            let config = ProviderConfig {
                model: "default".to_string(),
                permissions,
            };
            let argv = argv_strings(&build_invocation(
                ProviderKind::ClaudeCode,
                "/fixture/claude",
                &config,
                &request(),
            ));
            let actual: Vec<&str> = argv
                .windows(2)
                .filter_map(|pair| (pair[0] == "--allowed-tools").then_some(pair[1].as_str()))
                .collect();
            assert_eq!(actual, expected);
            assert!(!argv.contains(&"--permission-mode".to_string()));
        }
    }

    #[test]
    fn claude_full_access_replaces_all_allowed_tools() {
        let config = ProviderConfig {
            model: "claude-test".to_string(),
            permissions: PermissionsConfig {
                full_access: true,
                ..PermissionsConfig::default()
            },
        };
        let mut request = request();
        request.session_id = Some("session-1".to_string());
        let argv = argv_strings(&build_invocation(
            ProviderKind::ClaudeCode,
            "/fixture/claude",
            &config,
            &request,
        ));
        assert_eq!(
            &argv[argv.len() - 6..],
            [
                "--permission-mode",
                "bypassPermissions",
                "--model",
                "claude-test",
                "--resume",
                "session-1"
            ]
        );
        assert!(!argv.contains(&"--allowed-tools".to_string()));
    }

    #[test]
    fn codex_stream_flags_do_not_change_with_permissions() {
        let restricted = build_invocation(
            ProviderKind::Codex,
            "/fixture/codex",
            &ProviderConfig::default(),
            &request(),
        );
        let full = build_invocation(
            ProviderKind::Codex,
            "/fixture/codex",
            &ProviderConfig {
                permissions: PermissionsConfig {
                    full_access: true,
                    ..PermissionsConfig::default()
                },
                ..ProviderConfig::default()
            },
            &request(),
        );
        assert_eq!(restricted.args, full.args);
        assert_eq!(
            argv_strings(&restricted)[..7],
            [
                "/fixture/codex",
                "exec",
                "--json",
                "--sandbox",
                "danger-full-access",
                "--skip-git-repo-check",
                "-"
            ]
        );
    }

    #[test]
    fn context_and_sse_json_match_python_defaults() {
        let mut request = request();
        request.prompt = "why?".to_string();
        request.context = Some(Map::from_iter([
            ("experimentId".to_string(), json!("12")),
            ("unicode".to_string(), json!("café 😀")),
        ]));
        let invocation = build_invocation(
            ProviderKind::ClaudeCode,
            "/fixture/claude",
            &ProviderConfig::default(),
            &request,
        );
        assert_eq!(
            invocation.args[1].to_string_lossy(),
            "<context>\n{\"experimentId\": \"12\", \"unicode\": \"caf\\u00e9 \\ud83d\\ude00\"}\n</context>\n\nwhy?"
        );
        assert_eq!(
            Event::from_message("assistant", json!([{"text": "café 😀"}])).to_sse_frame(),
            concat!(
                "event: message\n",
                "data: {\"message\": {\"role\": \"assistant\", ",
                "\"content\": [{\"text\": \"caf\\u00e9 \\ud83d\\ude00\"}]}}\n\n"
            )
        );
    }

    #[test]
    fn python_float_rendering_matches_json_dumps_thresholds() {
        let values = [
            (json!(0.0), "0.0"),
            (json!(0.0001), "0.0001"),
            (json!(0.0000297), "2.97e-05"),
            (json!(1.0e16), "1e+16"),
            (json!(0.1319), "0.1319"),
        ];
        for (value, expected) in values {
            assert_eq!(python_json(&value), expected);
        }
    }

    #[test]
    fn claude_parser_filters_skill_prompt_and_emits_usage_before_done() {
        assert!(parse_claude_line(
            r#"{"type":"user","message":{"content":[{"type":"text","text":"Base directory for this skill: /tmp/x"}]}}"#
        )
        .is_empty());
        let events = parse_claude_line(
            r#"{"type":"result","result":"ok","session_id":"s1","total_cost_usd":0.25,"usage":{"input_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"output_tokens":5}}"#,
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, EventType::StreamEvent);
        assert_eq!(events[0].data["event"]["usage"]["prompt_tokens"], 9);
        assert_eq!(events[1].event_type, EventType::Done);
    }

    #[test]
    fn claude_parser_covers_blocks_errors_rate_limits_and_plain_text() {
        let events = parse_claude_line(
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"},{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"mlflow --help"}},{"type":"tool_result","tool_use_id":"t1"}]}}"#,
        );
        assert_eq!(
            events[0].data["message"]["content"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert_eq!(
            parse_claude_line("plain output")[0].event_type,
            EventType::Message
        );
        assert_eq!(
            parse_claude_line(r#"{"type":"error","error":{"message":"bad"}}"#)[0].data,
            json!({"error": "bad"})
        );
        assert_eq!(
            parse_claude_line(
                r#"{"type":"rate_limit_event","rate_limit_info":{"status":"limited","resetsAt":"soon"}}"#
            )[0].data["message"]["content"][0]["text"],
            "You've hit a rate limit — please wait a moment and try again. Your limit resets at soon."
        );
    }

    #[test]
    fn codex_parser_filters_items_tracks_thread_and_prices_cache() {
        let config = ProviderConfig {
            model: "o4-mini".to_string(),
            ..ProviderConfig::default()
        };
        let mut thread = String::new();
        assert!(parse_codex_line(
            r#"{"type":"thread.started","thread_id":"thread-1"}"#,
            &config,
            &mut thread
        )
        .is_empty());
        assert_eq!(thread, "thread-1");
        assert!(parse_codex_line(
            r#"{"type":"item.completed","item":{"type":"mcp_tool_call","text":"hidden"}}"#,
            &config,
            &mut thread
        )
        .is_empty());
        let usage = parse_codex_line(
            r#"{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":4,"output_tokens":5}}"#,
            &config,
            &mut thread,
        );
        assert_eq!(
            usage[0].data["event"]["usage"]["total_cost_usd"],
            json!(2.97e-5)
        );
    }

    #[test]
    fn health_error_http_mapping_is_exact() {
        let cases = [
            (HealthError::NotImplemented("no probe".to_string()), 501),
            (
                HealthError::CliNotInstalled("CLI not installed".to_string()),
                412,
            ),
            (
                HealthError::NotAuthenticated("Not authenticated".to_string()),
                401,
            ),
        ];
        for (error, status) in cases {
            assert_eq!(error.status_code(), status);
            assert_eq!(error.body(), json!({"detail": error.detail()}));
        }
    }
}
