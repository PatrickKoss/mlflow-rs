//! Native MLflow Assistant HTTP surface (plan T20.1, §12.10).
//!
//! This module owns the localhost gate, file-backed session/config state, SSE
//! framing, and the provider integration seam. CLI process execution belongs
//! to T20.2 and plugs in through [`AssistantProvider`].

use std::convert::Infallible;
use std::fmt::Debug;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, Path, Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::Router;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};
use futures::{FutureExt, StreamExt};
use mlflow_store::python_json_dumps;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use crate::state::AppState;

const PREFIX: &str = "/ajax-api/3.0/mlflow/assistant";
const REMOTE_ACCESS_DETAIL: &str =
    "Assistant API is only accessible from the same host where the MLflow server is running.";
const NO_PROVIDER_DETAIL: &str = "No assistant provider is configured or available.";
const DEV_STUB_ENV: &str = "MLFLOW_ASSISTANT_DEV_STUB_PROVIDERS";
const DEV_STUB_REPLY: &str = "This is a synthetic reply from the MLflow dev stub Claude CLI. The real Claude Code provider is replaced so the Assistant chat panel can be reviewed without credentials or LLM calls. No model was invoked to produce this message.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssistantSession {
    #[serde(default)]
    pub context: Map<String, Value>,
    #[serde(default)]
    pub messages: Vec<AssistantMessage>,
    #[serde(default)]
    pub pending_message: Option<AssistantMessage>,
    #[serde(default)]
    pub provider_session_id: Option<String>,
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
    #[serde(default)]
    pub pending_tool_decisions: Map<String, Value>,
}

impl Default for AssistantSession {
    fn default() -> Self {
        Self {
            context: Map::new(),
            messages: Vec::new(),
            pending_message: None,
            provider_session_id: None,
            working_dir: None,
            pending_tool_decisions: Map::new(),
        }
    }
}

/// Python-compatible file store rooted at `$TMPDIR/mlflow-assistant-sessions`.
#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &FsPath {
        &self.root
    }

    pub fn validate_session_id(session_id: &str) -> Result<(), String> {
        Uuid::parse_str(session_id)
            .map(|_| ())
            .map_err(|_| "Invalid session ID format".to_string())
    }

    pub fn session_file(&self, session_id: &str) -> Result<PathBuf, String> {
        Self::validate_session_id(session_id)?;
        Ok(self.root.join(format!("{session_id}.json")))
    }

    pub fn process_file(&self, session_id: &str) -> Result<PathBuf, String> {
        Self::validate_session_id(session_id)?;
        Ok(self.root.join(format!("{session_id}.process.json")))
    }

    pub fn save(&self, session_id: &str, session: &AssistantSession) -> std::io::Result<()> {
        let destination = self
            .session_file(session_id)
            .map_err(std::io::Error::other)?;
        fs::create_dir_all(&self.root)?;
        let value = serde_json::to_value(session).map_err(std::io::Error::other)?;
        self.atomic_write(&destination, python_json_dumps(&value, false).as_bytes())
    }

    pub fn load(&self, session_id: &str) -> std::io::Result<Option<AssistantSession>> {
        let path = match self.session_file(session_id) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(std::io::Error::other)
    }

    pub fn save_process_pid(&self, session_id: &str, pid: i32) -> std::io::Result<()> {
        let path = self
            .process_file(session_id)
            .map_err(std::io::Error::other)?;
        fs::create_dir_all(&self.root)?;
        fs::write(path, format!("{{\"pid\": {pid}}}"))
    }

    pub fn process_pid(&self, session_id: &str) -> std::io::Result<Option<i32>> {
        let path = match self.process_file(session_id) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let value: Value = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        Ok(value
            .get("pid")
            .and_then(Value::as_i64)
            .and_then(|pid| i32::try_from(pid).ok()))
    }

    pub fn clear_process_pid(&self, session_id: &str) -> std::io::Result<()> {
        let path = match self.process_file(session_id) {
            Ok(path) => path,
            Err(_) => return Ok(()),
        };
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub fn terminate_process(&self, session_id: &str) -> std::io::Result<bool> {
        let Some(pid) = self.process_pid(session_id)? else {
            return Ok(false);
        };
        if pid == 0 {
            return Ok(false);
        }
        // SAFETY: `kill` receives a scalar PID read from the validated session's
        // process file and does not dereference memory.
        let result = unsafe { libc::kill(pid, libc::SIGTERM) };
        let error = std::io::Error::last_os_error();
        self.clear_process_pid(session_id)?;
        if result == 0 {
            Ok(true)
        } else if matches!(error.raw_os_error(), Some(libc::ESRCH | libc::EPERM)) {
            Ok(false)
        } else {
            Err(error)
        }
    }

    fn atomic_write(&self, destination: &FsPath, bytes: &[u8]) -> std::io::Result<()> {
        for _ in 0..100 {
            let temporary = self.root.join(format!("{}.tmp", Uuid::new_v4().simple()));
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary);
            let mut file = match file {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            };
            let result = (|| {
                file.write_all(bytes)?;
                file.flush()?;
                drop(file);
                fs::rename(&temporary, destination)
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            return result;
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate assistant session temporary file",
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantEvent {
    pub event_type: String,
    pub data: Value,
}

impl AssistantEvent {
    pub fn new(event_type: impl Into<String>, data: Value) -> Self {
        Self {
            event_type: event_type.into(),
            data,
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self::new("error", json!({"error": error.into()}))
    }

    pub fn to_sse(&self) -> Bytes {
        Bytes::from(format!(
            "event: {}\ndata: {}\n\n",
            self.event_type,
            python_json_dumps(&self.data, false)
        ))
    }
}

#[derive(Debug, Clone)]
pub struct AssistantProviderRequest {
    pub prompt: String,
    pub tracking_uri: String,
    pub session_id: Option<String>,
    pub mlflow_session_id: String,
    pub cwd: Option<PathBuf>,
    pub context: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantProviderError {
    NotImplemented(String),
    CliNotInstalled(String),
    NotAuthenticated(String),
    NotConfigured(String),
    Internal(String),
}

/// Minimal provider contract shared with T20.2. Implementations own provider
/// execution; this module retains HTTP status mapping and SSE framing.
pub trait AssistantProvider: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn resolve_skills_path(&self, base_directory: &FsPath) -> PathBuf;
    fn check_connection(
        &self,
        config: Option<Value>,
    ) -> BoxFuture<'static, Result<(), AssistantProviderError>>;
    fn list_models(
        &self,
        base_url: Option<String>,
        api_key: Option<String>,
        config: Option<Value>,
    ) -> BoxFuture<'static, Result<Vec<String>, AssistantProviderError>>;
    fn stream(&self, request: AssistantProviderRequest) -> BoxStream<'static, AssistantEvent>;
}

#[derive(Debug, Clone)]
pub struct AssistantRuntime {
    inner: Arc<AssistantRuntimeInner>,
}

#[derive(Debug)]
struct AssistantRuntimeInner {
    sessions: SessionStore,
    config_path: PathBuf,
    skills_source: PathBuf,
    home: PathBuf,
    providers: Vec<Arc<dyn AssistantProvider>>,
}

impl AssistantRuntime {
    pub fn new(
        session_root: PathBuf,
        config_path: PathBuf,
        skills_source: PathBuf,
        home: PathBuf,
        providers: Vec<Arc<dyn AssistantProvider>>,
    ) -> Self {
        Self {
            inner: Arc::new(AssistantRuntimeInner {
                sessions: SessionStore::new(session_root),
                config_path,
                skills_source,
                home,
                providers,
            }),
        }
    }

    pub fn from_env() -> Self {
        let home = home_dir();
        let session_root = std::env::temp_dir().join("mlflow-assistant-sessions");
        let config_path = home.join(".mlflow/assistant/config.json");
        let skills_source =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../mlflow/assistant/skills");
        let dev_stubs = std::env::var(DEV_STUB_ENV).unwrap_or_default();
        let stub_claude = dev_stubs
            .split(',')
            .map(str::trim)
            .any(|name| name == "claude");
        let providers: Vec<Arc<dyn AssistantProvider>> = vec![
            if stub_claude {
                Arc::new(DevClaudeProvider) as Arc<dyn AssistantProvider>
            } else {
                Arc::new(BuiltinProvider::claude()) as Arc<dyn AssistantProvider>
            },
            Arc::new(BuiltinProvider::codex()),
            Arc::new(BuiltinProvider::gateway()),
            Arc::new(BuiltinProvider::ollama()),
        ];
        Self::new(session_root, config_path, skills_source, home, providers)
    }

    pub fn sessions(&self) -> &SessionStore {
        &self.inner.sessions
    }

    fn provider(&self, name: &str) -> Option<Arc<dyn AssistantProvider>> {
        self.inner
            .providers
            .iter()
            .find(|provider| provider.name() == name)
            .cloned()
    }

    fn selected_provider(&self, config: &AssistantConfig) -> Option<Arc<dyn AssistantProvider>> {
        config.providers.iter().find_map(|(name, value)| {
            (value.get("selected").and_then(Value::as_bool) == Some(true))
                .then(|| self.provider(name))
                .flatten()
        })
    }

    fn load_config(&self) -> AssistantConfig {
        AssistantConfig::load(&self.inner.config_path)
    }

    fn save_config(&self, config: &AssistantConfig) -> std::io::Result<()> {
        config.save(&self.inner.config_path)
    }
}

impl Default for AssistantRuntime {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Clone, Default)]
struct AssistantConfig {
    projects: Map<String, Value>,
    providers: Map<String, Value>,
}

impl AssistantConfig {
    fn load(path: &FsPath) -> Self {
        let Ok(bytes) = fs::read(path) else {
            return Self::default();
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            return Self::default();
        };
        let Some(object) = value.as_object() else {
            return Self::default();
        };
        let projects = match object.get("projects") {
            None => Map::new(),
            Some(Value::Object(projects)) => projects.clone(),
            Some(_) => return Self::default(),
        };
        let providers = match object.get("providers") {
            None => Map::new(),
            Some(Value::Object(providers)) => providers.clone(),
            Some(_) => return Self::default(),
        };
        let mut normalized_projects = Map::new();
        for (name, project) in &projects {
            let Some(project) = normalize_project(project) else {
                return Self::default();
            };
            normalized_projects.insert(name.clone(), project);
        }
        let mut normalized_providers = Map::new();
        for (name, provider) in &providers {
            let Some(provider) = normalize_provider(provider) else {
                return Self::default();
            };
            normalized_providers.insert(name.clone(), provider);
        }
        Self {
            projects: normalized_projects,
            providers: normalized_providers,
        }
    }

    fn save(&self, path: &FsPath) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let value = json!({"projects": self.projects, "providers": self.providers});
        let body = serde_json::to_string_pretty(&value).map_err(std::io::Error::other)?;
        fs::write(path, body)
    }

    fn response_value(&self) -> Value {
        json!({"providers": self.providers, "projects": self.projects})
    }

    fn project_path(&self, experiment_id: &str) -> Option<PathBuf> {
        self.projects
            .get(experiment_id)
            .and_then(|project| project.get("location"))
            .and_then(Value::as_str)
            .map(PathBuf::from)
    }
}

#[derive(Debug, Clone, Copy)]
enum BuiltinKind {
    Claude,
    Codex,
    Gateway,
    Ollama,
}

#[derive(Debug)]
struct BuiltinProvider {
    name: &'static str,
    skills_dir: &'static str,
    kind: BuiltinKind,
}

impl BuiltinProvider {
    fn claude() -> Self {
        Self {
            name: "claude_code",
            skills_dir: ".claude",
            kind: BuiltinKind::Claude,
        }
    }

    fn codex() -> Self {
        Self {
            name: "codex",
            skills_dir: ".codex",
            kind: BuiltinKind::Codex,
        }
    }

    fn gateway() -> Self {
        Self {
            name: "mlflow_gateway",
            skills_dir: ".agent",
            kind: BuiltinKind::Gateway,
        }
    }

    fn ollama() -> Self {
        Self {
            name: "ollama",
            skills_dir: ".agent",
            kind: BuiltinKind::Ollama,
        }
    }
}

impl AssistantProvider for BuiltinProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn resolve_skills_path(&self, base_directory: &FsPath) -> PathBuf {
        base_directory.join(self.skills_dir).join("skills")
    }

    fn check_connection(
        &self,
        config: Option<Value>,
    ) -> BoxFuture<'static, Result<(), AssistantProviderError>> {
        let kind = self.kind;
        async move {
            match kind {
                BuiltinKind::Claude => Err(AssistantProviderError::CliNotInstalled(
                    "Claude Code CLI is not installed. Install it with: npm install -g @anthropic-ai/claude-code".to_string(),
                )),
                BuiltinKind::Codex => Err(AssistantProviderError::CliNotInstalled(
                    "OpenAI Codex CLI is not installed. Install it with: npm install -g @openai/codex".to_string(),
                )),
                BuiltinKind::Gateway => Err(AssistantProviderError::NotImplemented(
                    "MLflow AI Gateway connection is verified by the frontend; the assistant backend has no probe to run.".to_string(),
                )),
                BuiltinKind::Ollama => {
                    let base = config
                        .as_ref()
                        .and_then(|value| value.get("base_url"))
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .unwrap_or("http://localhost:11434");
                    let result = reqwest::Client::new()
                        .get(format!("{}/api/tags", base.trim_end_matches('/')))
                        .send()
                        .await;
                    match result {
                        Ok(response) if response.status().is_success() => Ok(()),
                        _ => Err(AssistantProviderError::NotAuthenticated(format!(
                            "Cannot connect to Ollama at {base}. Make sure Ollama is running: ollama serve"
                        ))),
                    }
                }
            }
        }
        .boxed()
    }

    fn list_models(
        &self,
        base_url: Option<String>,
        api_key: Option<String>,
        config: Option<Value>,
    ) -> BoxFuture<'static, Result<Vec<String>, AssistantProviderError>> {
        let kind = self.kind;
        async move {
            if !matches!(kind, BuiltinKind::Ollama) {
                return Err(AssistantProviderError::NotImplemented(String::new()));
            }
            let base = base_url
                .filter(|value| !value.is_empty())
                .or_else(|| {
                    config
                        .as_ref()
                        .and_then(|value| value.get("base_url"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            let mut request =
                reqwest::Client::new().get(format!("{}/api/tags", base.trim_end_matches('/')));
            if let Some(api_key) = api_key.filter(|value| !value.is_empty()) {
                request = request.bearer_auth(api_key);
            }
            let response = request.send().await.map_err(|error| {
                AssistantProviderError::NotConfigured(format!(
                    "Cannot connect to Ollama at {base}: {error}"
                ))
            })?;
            let response = response.error_for_status().map_err(|error| {
                AssistantProviderError::NotConfigured(format!(
                    "Cannot connect to Ollama at {base}: {error}"
                ))
            })?;
            let body: Value = response.json().await.map_err(|error| {
                AssistantProviderError::NotConfigured(format!(
                    "Cannot connect to Ollama at {base}: {error}"
                ))
            })?;
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
        .boxed()
    }

    fn stream(&self, _request: AssistantProviderRequest) -> BoxStream<'static, AssistantEvent> {
        let error = match self.kind {
            BuiltinKind::Claude => {
                "Claude CLI not found. Please install Claude Code CLI and ensure it's in your PATH."
            }
            BuiltinKind::Codex => {
                "codex CLI not found. Please install the OpenAI Codex CLI and ensure it's in your PATH."
            }
            BuiltinKind::Gateway => "MLflow AI Gateway provider execution is not installed.",
            BuiltinKind::Ollama => "Ollama provider execution is not installed.",
        };
        stream::once(async move { AssistantEvent::error(error) }).boxed()
    }
}

#[derive(Debug)]
struct DevClaudeProvider;

impl AssistantProvider for DevClaudeProvider {
    fn name(&self) -> &str {
        "claude_code"
    }

    fn resolve_skills_path(&self, base_directory: &FsPath) -> PathBuf {
        base_directory.join(".claude/skills")
    }

    fn check_connection(
        &self,
        _config: Option<Value>,
    ) -> BoxFuture<'static, Result<(), AssistantProviderError>> {
        async { Ok(()) }.boxed()
    }

    fn list_models(
        &self,
        _base_url: Option<String>,
        _api_key: Option<String>,
        _config: Option<Value>,
    ) -> BoxFuture<'static, Result<Vec<String>, AssistantProviderError>> {
        async { Err(AssistantProviderError::NotImplemented(String::new())) }.boxed()
    }

    fn stream(&self, request: AssistantProviderRequest) -> BoxStream<'static, AssistantEvent> {
        let session_id = request
            .session_id
            .unwrap_or_else(|| format!("mlflow-dev-stub-{}", Uuid::new_v4().simple()));
        stream::iter(vec![
            AssistantEvent::new(
                "message",
                json!({"message": {"role": "assistant", "content": [{"text": DEV_STUB_REPLY}]}}),
            ),
            AssistantEvent::new(
                "stream_event",
                json!({"event": {"type": "usage", "usage": {"prompt_tokens": 8, "completion_tokens": 24, "total_tokens": 32, "total_cost_usd": 0.0}}}),
            ),
            AssistantEvent::new(
                "done",
                json!({"result": DEV_STUB_REPLY, "session_id": session_id}),
            ),
        ])
        .boxed()
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(&format!("{PREFIX}/message"), post(send_message))
        .route(
            &format!("{PREFIX}/sessions/{{session_id}}/stream"),
            get(stream_response),
        )
        .route(
            &format!("{PREFIX}/sessions/{{session_id}}"),
            patch(patch_session),
        )
        .route(
            &format!("{PREFIX}/sessions/{{session_id}}/permission"),
            post(resolve_permission),
        )
        .route(
            &format!("{PREFIX}/providers/{{provider}}/health"),
            get(provider_health),
        )
        .route(
            &format!("{PREFIX}/config"),
            get(get_config).put(update_config),
        )
        .route(
            &format!("{PREFIX}/skills/install"),
            post(install_skills_endpoint),
        )
        .route(
            &format!("{PREFIX}/providers/{{provider}}/models"),
            get(list_provider_models),
        )
        .route_layer(middleware::from_fn(require_localhost))
}

async fn require_localhost(request: Request, next: Next) -> Response {
    let is_loopback = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(address)| address.ip().is_loopback())
        .unwrap_or(false);
    if !is_loopback {
        return detail_response(StatusCode::FORBIDDEN, REMOTE_ACCESS_DETAIL);
    }
    next.run(request).await
}

async fn send_message(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match parse_object_body(&body) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let message = match required_string(&request, "message") {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let session_id = match optional_string(&request, "session_id") {
        Ok(Some(value)) if !value.is_empty() => value,
        Ok(_) => Uuid::new_v4().to_string(),
        Err(response) => return *response,
    };
    let experiment_id = match optional_string(&request, "experiment_id") {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let context = match request.get("context") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(value)) => value.clone(),
        Some(value) => {
            return field_type_error(
                "context",
                "dict_type",
                "Input should be a valid dictionary",
                value.clone(),
            )
        }
    };
    let runtime = state.assistant_runtime();
    let config = runtime.load_config();
    let working_dir = experiment_id
        .as_deref()
        .and_then(|experiment_id| config.project_path(experiment_id));
    let mut session = match runtime.sessions().load(&session_id) {
        Ok(Some(session)) => session,
        Ok(None) => AssistantSession {
            context: context.clone(),
            working_dir,
            ..Default::default()
        },
        Err(_) => return internal_error(),
    };
    if !context.is_empty() && !session.context.is_empty() {
        session.context.extend(context);
    } else if !context.is_empty() {
        session.context = context;
    }
    let pending = AssistantMessage {
        role: "user".to_string(),
        content: Value::String(message),
    };
    session.pending_message = Some(pending.clone());
    session.messages.push(pending);
    if runtime.sessions().save(&session_id, &session).is_err() {
        return internal_error();
    }
    // D18: Python emits `/stream/{id}`, but the decided Rust contract returns
    // the actual route used by the frontend.
    json_response(
        StatusCode::OK,
        json!({"session_id": session_id, "stream_url": format!("{PREFIX}/sessions/{session_id}/stream")}),
    )
}

async fn stream_response(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let runtime = state.assistant_runtime().clone();
    let mut session = match runtime.sessions().load(&session_id) {
        Ok(Some(session)) => session,
        Ok(None) => return detail_response(StatusCode::NOT_FOUND, "Session not found"),
        Err(_) => return internal_error(),
    };
    let pending = session.pending_message.take();
    let decisions = std::mem::take(&mut session.pending_tool_decisions);
    if pending.is_none() && decisions.is_empty() {
        return detail_response(StatusCode::BAD_REQUEST, "No pending message to process");
    }
    if runtime.sessions().save(&session_id, &session).is_err() {
        return internal_error();
    }
    let prompt = pending
        .as_ref()
        .and_then(|message| message.content.as_str())
        .unwrap_or("")
        .to_string();
    let mut context = session.context.clone();
    if pending.is_none() && !decisions.is_empty() {
        context.insert("tool_decisions".to_string(), Value::Object(decisions));
    }
    let config = runtime.load_config();
    let provider = runtime.selected_provider(&config);
    let tracking_uri = tracking_uri(&headers);
    let source: BoxStream<'static, AssistantEvent> = match provider {
        Some(provider) => provider.stream(AssistantProviderRequest {
            prompt,
            tracking_uri,
            session_id: session.provider_session_id.clone(),
            mlflow_session_id: session_id.clone(),
            cwd: session.working_dir.clone(),
            context,
        }),
        None => stream::once(async { AssistantEvent::error(NO_PROVIDER_DETAIL) }).boxed(),
    };
    let output = source.then(move |event| {
        let runtime = runtime.clone();
        let session_id = session_id.clone();
        let mut session = session.clone();
        async move {
            if event.event_type == "done" {
                session.provider_session_id = event
                    .data
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let _ = runtime.sessions().save(&session_id, &session);
            }
            Ok::<_, Infallible>(event.to_sse())
        }
    });
    let mut response = Response::new(Body::from_stream(output));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

async fn patch_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    body: Bytes,
) -> Response {
    let request = match parse_object_body(&body) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    match request.get("status") {
        None => return missing_field("status", Value::Object(request)),
        Some(Value::String(status)) if status == "cancelled" => {}
        Some(value) => return literal_error("status", "'cancelled'", value.clone()),
    }
    let runtime = state.assistant_runtime();
    let mut session = match runtime.sessions().load(&session_id) {
        Ok(Some(session)) => session,
        Ok(None) => return detail_response(StatusCode::NOT_FOUND, "Session not found"),
        Err(_) => return internal_error(),
    };
    session.pending_tool_decisions.clear();
    if runtime.sessions().save(&session_id, &session).is_err() {
        return internal_error();
    }
    let terminated = runtime
        .sessions()
        .terminate_process(&session_id)
        .unwrap_or(false);
    let message = if terminated {
        "Session cancelled and process terminated"
    } else {
        "Session cancelled"
    };
    json_response(StatusCode::OK, json!({"message": message}))
}

async fn resolve_permission(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    body: Bytes,
) -> Response {
    let request = match parse_object_body(&body) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let request_id = match required_string(&request, "request_id") {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let decision = match request.get("decision") {
        None => return missing_field("decision", Value::Object(request)),
        Some(Value::String(value)) if matches!(value.as_str(), "allow" | "deny") => value.clone(),
        Some(value) => return literal_error("decision", "'allow' or 'deny'", value.clone()),
    };
    if let Err(detail) = SessionStore::validate_session_id(&session_id) {
        return detail_response(StatusCode::BAD_REQUEST, &detail);
    }
    let runtime = state.assistant_runtime();
    let mut session = match runtime.sessions().load(&session_id) {
        Ok(Some(session)) => session,
        Ok(None) => return detail_response(StatusCode::NOT_FOUND, "Session not found"),
        Err(_) => return internal_error(),
    };
    session.pending_tool_decisions = Map::from_iter([(request_id, Value::String(decision))]);
    if runtime.sessions().save(&session_id, &session).is_err() {
        return internal_error();
    }
    json_response(
        StatusCode::OK,
        json!({"session_id": session_id, "stream_url": format!("{PREFIX}/sessions/{session_id}/stream")}),
    )
}

async fn provider_health(State(state): State<AppState>, Path(provider): Path<String>) -> Response {
    let runtime = state.assistant_runtime();
    let Some(instance) = runtime.provider(&provider) else {
        return detail_response(
            StatusCode::NOT_FOUND,
            &format!("Provider '{provider}' not found"),
        );
    };
    let config = runtime.load_config();
    match instance
        .check_connection(config.providers.get(&provider).cloned())
        .await
    {
        Ok(()) => json_response(StatusCode::OK, json!({"status": "ok"})),
        Err(AssistantProviderError::NotImplemented(detail)) => {
            detail_response(StatusCode::NOT_IMPLEMENTED, &detail)
        }
        Err(AssistantProviderError::CliNotInstalled(detail)) => {
            detail_response(StatusCode::PRECONDITION_FAILED, &detail)
        }
        Err(AssistantProviderError::NotAuthenticated(detail)) => {
            detail_response(StatusCode::UNAUTHORIZED, &detail)
        }
        Err(_) => internal_error(),
    }
}

async fn get_config(State(state): State<AppState>) -> Response {
    json_response(
        StatusCode::OK,
        state.assistant_runtime().load_config().response_value(),
    )
}

async fn update_config(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match parse_object_body(&body) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    for field in ["providers", "projects"] {
        if let Some(value) = request.get(field) {
            if !value.is_null() && !value.is_object() {
                return field_type_error(
                    field,
                    "dict_type",
                    "Input should be a valid dictionary",
                    value.clone(),
                );
            }
        }
    }
    let runtime = state.assistant_runtime();
    let mut config = runtime.load_config();
    if let Some(providers) = request.get("providers").and_then(Value::as_object) {
        for (name, update) in providers {
            let Some(update) = update.as_object() else {
                return internal_error();
            };
            let existing = config.providers.get(name).cloned();
            let model = update
                .get("model")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .or_else(|| {
                    existing
                        .as_ref()
                        .and_then(|value| value.get("model"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "default".to_string());
            let mut provider = existing.unwrap_or_else(default_provider);
            let object = provider.as_object_mut().expect("normalized provider");
            object.insert("model".to_string(), Value::String(model));
            if let Some(Value::String(base_url)) = update.get("base_url") {
                object.insert("base_url".to_string(), Value::String(base_url.clone()));
            }
            if let Some(Value::String(api_key)) = update.get("api_key") {
                object.insert("api_key".to_string(), Value::String(api_key.clone()));
            }
            if let Some(permissions) = update.get("permissions") {
                let Some(permissions) = normalize_permissions_update(permissions) else {
                    return internal_error();
                };
                object.insert("permissions".to_string(), permissions);
            }
            let selected = update.get("selected").and_then(Value::as_bool) == Some(true);
            config.providers.insert(name.clone(), provider);
            if selected {
                for (provider_name, provider) in &mut config.providers {
                    provider
                        .as_object_mut()
                        .expect("normalized provider")
                        .insert("selected".to_string(), Value::Bool(provider_name == name));
                }
            }
        }
    }
    if let Some(projects) = request.get("projects").and_then(Value::as_object) {
        for (experiment_id, update) in projects {
            if update.is_null() {
                config.projects.shift_remove(experiment_id);
                continue;
            }
            let Some(update) = update.as_object() else {
                return internal_error();
            };
            let location = update.get("location").and_then(Value::as_str).unwrap_or("");
            let project_path = expand_user(location, &runtime.inner.home);
            if !project_path.exists() {
                return detail_response(
                    StatusCode::BAD_REQUEST,
                    &format!("Project path does not exist: {location}"),
                );
            }
            let project_type = update
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("local");
            if project_type != "local" {
                return internal_error();
            }
            config.projects.insert(
                experiment_id.clone(),
                json!({"type": "local", "location": project_path.to_string_lossy()}),
            );
        }
    }
    if runtime.save_config(&config).is_err() {
        return internal_error();
    }
    json_response(StatusCode::OK, config.response_value())
}

async fn install_skills_endpoint(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match parse_object_body(&body) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let install_type = match request.get("type") {
        None => "global".to_string(),
        Some(Value::String(value)) if matches!(value.as_str(), "global" | "project" | "custom") => {
            value.clone()
        }
        Some(value) => {
            return literal_error("type", "'global', 'project' or 'custom'", value.clone())
        }
    };
    let custom_path = match optional_string(&request, "custom_path") {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let experiment_id = match optional_string(&request, "experiment_id") {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let runtime = state.assistant_runtime();
    let config = runtime.load_config();
    let project_path = if install_type == "project" {
        let Some(experiment_id) = experiment_id else {
            return detail_response(
                StatusCode::BAD_REQUEST,
                "experiment_id required for 'project' type",
            );
        };
        let Some(path) = config.project_path(&experiment_id) else {
            return detail_response(
                StatusCode::BAD_REQUEST,
                &format!("No project path configured for experiment {experiment_id}"),
            );
        };
        Some(path)
    } else {
        None
    };
    let Some(provider) = runtime.selected_provider(&config) else {
        return detail_response(StatusCode::PRECONDITION_FAILED, NO_PROVIDER_DETAIL);
    };
    let destination = match install_type.as_str() {
        "global" => provider.resolve_skills_path(&runtime.inner.home),
        "project" => provider.resolve_skills_path(project_path.as_deref().expect("project path")),
        "custom" => {
            let Some(path) = custom_path.filter(|value| !value.is_empty()) else {
                return detail_response(
                    StatusCode::BAD_REQUEST,
                    "custom_path is required when type='custom'.",
                );
            };
            expand_user(&path, &runtime.inner.home)
        }
        _ => unreachable!(),
    };
    if destination.exists() {
        match list_installed_skills(&destination) {
            Ok(skills) if !skills.is_empty() => {
                return json_response(
                    StatusCode::OK,
                    json!({"installed_skills": skills, "skills_directory": destination.to_string_lossy()}),
                )
            }
            Ok(_) => {}
            Err(_) => return internal_error(),
        }
    }
    let installed = match install_skills(&runtime.inner.skills_source, &destination) {
        Ok(installed) => installed,
        Err(_) => return internal_error(),
    };
    json_response(
        StatusCode::OK,
        json!({"installed_skills": installed, "skills_directory": destination.to_string_lossy()}),
    )
}

#[derive(Debug, Deserialize)]
struct ModelsQuery {
    base_url: Option<String>,
}

async fn list_provider_models(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    Query(query): Query<ModelsQuery>,
    headers: HeaderMap,
) -> Response {
    let runtime = state.assistant_runtime();
    let Some(instance) = runtime.provider(&provider) else {
        return detail_response(
            StatusCode::NOT_FOUND,
            &format!("Provider '{provider}' not found"),
        );
    };
    let config = runtime.load_config();
    let api_key = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    match instance
        .list_models(
            query.base_url,
            api_key,
            config.providers.get(&provider).cloned(),
        )
        .await
    {
        Ok(models) => json_response(StatusCode::OK, json!({"models": models})),
        Err(AssistantProviderError::NotImplemented(_)) => detail_response(
            StatusCode::NOT_FOUND,
            &format!("Model listing is not supported for provider '{provider}'"),
        ),
        Err(AssistantProviderError::CliNotInstalled(detail)) => {
            detail_response(StatusCode::PRECONDITION_FAILED, &detail)
        }
        Err(AssistantProviderError::NotConfigured(detail))
        | Err(AssistantProviderError::NotAuthenticated(detail)) => {
            detail_response(StatusCode::SERVICE_UNAVAILABLE, &detail)
        }
        Err(AssistantProviderError::Internal(_)) => internal_error(),
    }
}

fn normalize_provider(value: &Value) -> Option<Value> {
    let value = value.as_object()?;
    let model = value
        .get("model")
        .map(Value::as_str)
        .transpose()?
        .unwrap_or("default");
    let selected = value
        .get("selected")
        .map(Value::as_bool)
        .transpose()?
        .unwrap_or(false);
    let base_url = nullable_string(value.get("base_url"))?;
    let api_key = nullable_string(value.get("api_key"))?;
    let permissions = match value.get("permissions") {
        Some(value) => normalize_permissions(value)?,
        None => default_permissions(),
    };
    let skills = match value.get("skills") {
        Some(Value::Object(skills)) => {
            let skill_type = skills
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("global");
            if !matches!(skill_type, "global" | "project" | "custom") {
                return None;
            }
            let custom_path = nullable_string(skills.get("custom_path"))?;
            json!({"type": skill_type, "custom_path": custom_path})
        }
        None => json!({"type": "global", "custom_path": null}),
        _ => return None,
    };
    Some(json!({
        "model": model,
        "selected": selected,
        "base_url": base_url,
        "api_key": api_key,
        "permissions": permissions,
        "skills": skills,
    }))
}

fn default_provider() -> Value {
    json!({
        "model": "default",
        "selected": false,
        "base_url": null,
        "api_key": null,
        "permissions": default_permissions(),
        "skills": {"type": "global", "custom_path": null},
    })
}

fn normalize_project(value: &Value) -> Option<Value> {
    let value = value.as_object()?;
    let project_type = value.get("type").and_then(Value::as_str).unwrap_or("local");
    if project_type != "local" {
        return None;
    }
    let location = value.get("location")?.as_str()?;
    Some(json!({"type": "local", "location": location}))
}

fn normalize_permissions(value: &Value) -> Option<Value> {
    let value = value.as_object()?;
    Some(json!({
        "allow_edit_files": optional_bool(value.get("allow_edit_files"), true)?,
        "allow_read_docs": optional_bool(value.get("allow_read_docs"), true)?,
        "full_access": optional_bool(value.get("full_access"), false)?,
    }))
}

fn normalize_permissions_update(value: &Value) -> Option<Value> {
    normalize_permissions(value)
}

fn default_permissions() -> Value {
    json!({"allow_edit_files": true, "allow_read_docs": true, "full_access": false})
}

fn optional_bool(value: Option<&Value>, default: bool) -> Option<bool> {
    value
        .map(Value::as_bool)
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn nullable_string(value: Option<&Value>) -> Option<Option<String>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(Value::String(value)) => Some(Some(value.clone())),
        _ => None,
    }
}

fn parse_object_body(body: &[u8]) -> Result<Map<String, Value>, Box<Response>> {
    if body.is_empty() {
        return Err(Box::new(validation_response(json!({
            "type": "missing", "loc": ["body"], "msg": "Field required", "input": null
        }))));
    }
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        Box::new(validation_response(json!({
            "type": "json_invalid",
            "loc": ["body", error.column().saturating_sub(1)],
            "msg": "JSON decode error",
            "input": {},
            "ctx": {"error": error.to_string()},
        })))
    })?;
    match value {
        Value::Object(value) => Ok(value),
        value => Err(Box::new(validation_response(json!({
            "type": "model_attributes_type",
            "loc": ["body"],
            "msg": "Input should be a valid dictionary or object to extract fields from",
            "input": value,
        })))),
    }
}

fn required_string(value: &Map<String, Value>, field: &str) -> Result<String, Box<Response>> {
    match value.get(field) {
        None => Err(Box::new(missing_field(field, Value::Object(value.clone())))),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(value) => Err(Box::new(field_type_error(
            field,
            "string_type",
            "Input should be a valid string",
            value.clone(),
        ))),
    }
}

fn optional_string(
    value: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, Box<Response>> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(value) => Err(Box::new(field_type_error(
            field,
            "string_type",
            "Input should be a valid string",
            value.clone(),
        ))),
    }
}

fn missing_field(field: &str, input: Value) -> Response {
    validation_response(json!({
        "type": "missing", "loc": ["body", field], "msg": "Field required", "input": input
    }))
}

fn field_type_error(field: &str, kind: &str, message: &str, input: Value) -> Response {
    validation_response(json!({
        "type": kind, "loc": ["body", field], "msg": message, "input": input
    }))
}

fn literal_error(field: &str, expected: &str, input: Value) -> Response {
    validation_response(json!({
        "type": "literal_error",
        "loc": ["body", field],
        "msg": format!("Input should be {expected}"),
        "input": input,
        "ctx": {"expected": expected},
    }))
}

fn validation_response(error: Value) -> Response {
    json_response(StatusCode::UNPROCESSABLE_ENTITY, json!({"detail": [error]}))
}

fn detail_response(status: StatusCode, detail: &str) -> Response {
    json_response(status, json!({"detail": detail}))
}

fn json_response(status: StatusCode, value: Value) -> Response {
    let mut response = Response::new(Body::from(
        serde_json::to_vec(&value).expect("JSON value serialization"),
    ));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

fn internal_error() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        "Internal Server Error",
    )
        .into_response()
}

fn tracking_uri(headers: &HeaderMap) -> String {
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("http");
    let host = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("localhost");
    format!("{scheme}://{host}")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn expand_user(path: &str, home: &FsPath) -> PathBuf {
    if path == "~" {
        home.to_path_buf()
    } else if let Some(rest) = path.strip_prefix("~/") {
        home.join(rest)
    } else {
        PathBuf::from(path)
    }
}

fn install_skills(source: &FsPath, destination: &FsPath) -> std::io::Result<Vec<String>> {
    let mut installed = Vec::new();
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || !path.join("SKILL.md").is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        copy_tree(&path, &destination.join(&name))?;
        installed.push(name);
    }
    installed.sort();
    Ok(installed)
}

fn copy_tree(source: &FsPath, destination: &FsPath) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &destination_path)?;
        } else {
            fs::copy(source_path, destination_path)?;
        }
    }
    Ok(())
}

fn list_installed_skills(destination: &FsPath) -> std::io::Result<Vec<String>> {
    fn visit(path: &FsPath, skills: &mut Vec<String>) -> std::io::Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let child = entry.path();
            if child.is_dir() {
                if child.join("SKILL.md").is_file() {
                    skills.push(entry.file_name().to_string_lossy().into_owned());
                }
                visit(&child, skills)?;
            }
        }
        Ok(())
    }
    let mut skills = Vec::new();
    visit(destination, &mut skills)?;
    skills.sort();
    Ok(skills)
}

trait OptionTranspose<T> {
    fn transpose(self) -> Option<Option<T>>;
}

impl<T> OptionTranspose<T> for Option<Option<T>> {
    fn transpose(self) -> Option<Option<T>> {
        match self {
            Some(Some(value)) => Some(Some(value)),
            Some(None) => None,
            None => Some(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_serialization_matches_python_json_dump() {
        let session = AssistantSession {
            context: Map::from_iter([("trace".to_string(), json!("café"))]),
            messages: vec![AssistantMessage {
                role: "user".to_string(),
                content: json!("hello"),
            }],
            pending_message: Some(AssistantMessage {
                role: "user".to_string(),
                content: json!("hello"),
            }),
            provider_session_id: None,
            working_dir: Some(PathBuf::from("/tmp/project")),
            pending_tool_decisions: Map::new(),
        };
        let value = serde_json::to_value(session).unwrap();
        assert_eq!(
            python_json_dumps(&value, false),
            "{\"context\": {\"trace\": \"caf\\u00e9\"}, \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"pending_message\": {\"role\": \"user\", \"content\": \"hello\"}, \"provider_session_id\": null, \"working_dir\": \"/tmp/project\", \"pending_tool_decisions\": {}}"
        );
    }

    #[test]
    fn uuid_validation_rejects_traversal() {
        assert!(SessionStore::validate_session_id("../../config").is_err());
        assert!(SessionStore::validate_session_id(&Uuid::new_v4().to_string()).is_ok());
    }

    #[test]
    fn session_save_atomically_replaces_and_leaves_no_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let store = SessionStore::new(directory.path().join("sessions"));
        let session_id = Uuid::new_v4().to_string();
        let mut session = AssistantSession::default();
        store.save(&session_id, &session).unwrap();
        session.provider_session_id = Some("second-write".to_string());
        store.save(&session_id, &session).unwrap();

        assert_eq!(store.load(&session_id).unwrap(), Some(session));
        let entries: Vec<_> = fs::read_dir(store.root())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec![format!("{session_id}.json")]);
    }
}
