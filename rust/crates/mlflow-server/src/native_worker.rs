//! Per-job native GenAI subprocess executor (plan §14.2 / D14).

use std::path::PathBuf;

use mlflow_genai::{
    JobKind, WorkerLaunchError, WorkerLauncher, WorkerRequest, NATIVE_WORKER_PROTOCOL_VERSION,
};
use serde_json::{json, Value};

use crate::job_runner::{
    Exclusive, JobExecutionFuture, JobExecutionRequest, JobExecutionResult, JobExecutor,
    JobFunction,
};

pub const DEFAULT_WORKER_CAPTURE_BYTES: usize = 4 * 1024 * 1024;
pub const MLFLOW_GENAI_WORKER_PATH: &str = "MLFLOW_GENAI_WORKER_PATH";

/// Resolve the co-installed worker. D14 treats absence as a deployment error,
/// never as permission to fall back to Python.
pub fn resolve_worker_program() -> std::io::Result<PathBuf> {
    let path = match std::env::var_os(MLFLOW_GENAI_WORKER_PATH) {
        Some(path) => PathBuf::from(path),
        None => std::env::current_exe()?
            .parent()
            .ok_or_else(|| std::io::Error::other("server executable has no parent directory"))?
            .join(if cfg!(windows) {
                "mlflow-genai-worker.exe"
            } else {
                "mlflow-genai-worker"
            }),
    };
    let metadata = std::fs::metadata(&path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::other(format!(
            "native GenAI worker is not a file: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("native GenAI worker is not executable: {}", path.display()),
            ));
        }
    }
    Ok(path)
}

/// Python decorator metadata for the closed six-kind allowlist. T17.1's
/// coordinators apply these caps before calling the launcher.
pub fn native_job_functions() -> Result<Vec<JobFunction>, mlflow_error::MlflowError> {
    let judge_workers = max_workers("MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS", 10)?;
    let online_workers = max_workers("MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS", 5)?;
    Ok(vec![
        JobFunction::new("invoke_scorer", judge_workers),
        JobFunction::new("run_online_trace_scorer", online_workers)
            .exclusive(Exclusive::Params(vec!["experiment_id".to_string()])),
        JobFunction::new("run_online_session_scorer", online_workers)
            .exclusive(Exclusive::Params(vec!["experiment_id".to_string()])),
        JobFunction::new("optimize_prompts", 2),
        JobFunction::new("invoke_issue_detection", judge_workers),
        JobFunction::new("invoke_genai_evaluate", judge_workers),
    ])
}

fn max_workers(name: &str, default: usize) -> Result<usize, mlflow_error::MlflowError> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    let parsed = value.to_string_lossy().parse::<usize>().map_err(|error| {
        mlflow_error::MlflowError::invalid_parameter_value(format!(
            "Failed to convert {value:?} for {name}: {error}"
        ))
    })?;
    if parsed == 0 {
        return Err(mlflow_error::MlflowError::invalid_parameter_value(format!(
            "{name} must be greater than zero."
        )));
    }
    Ok(parsed)
}

/// Converts runner requests into versioned stdin envelopes and supervises one
/// `mlflow-genai-worker` process for each call.
#[derive(Debug, Clone)]
pub struct NativeWorkerExecutor {
    launcher: WorkerLauncher,
    tracking_uri: Option<String>,
    gateway_uri: Option<String>,
    internal_gateway_token: Option<String>,
}

impl NativeWorkerExecutor {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self::from_launcher(
            WorkerLauncher::new(program)
                .without_timeout()
                .max_output_bytes(DEFAULT_WORKER_CAPTURE_BYTES),
        )
    }

    pub fn from_launcher(launcher: WorkerLauncher) -> Self {
        Self {
            launcher,
            tracking_uri: None,
            gateway_uri: None,
            internal_gateway_token: None,
        }
    }

    pub fn tracking_uri(mut self, uri: impl Into<String>) -> Self {
        self.tracking_uri = Some(uri.into());
        self
    }

    pub fn gateway_uri(mut self, uri: impl Into<String>) -> Self {
        self.gateway_uri = Some(uri.into());
        self
    }

    pub fn internal_gateway_token(mut self, token: impl Into<String>) -> Self {
        self.internal_gateway_token = Some(token.into());
        self
    }
}

impl JobExecutor for NativeWorkerExecutor {
    fn execute(&self, request: JobExecutionRequest) -> JobExecutionFuture {
        let job_kind = match request.job_name.parse::<JobKind>() {
            Ok(job_kind) => job_kind,
            Err(job_name) => {
                return Box::pin(async move {
                    JobExecutionResult::Failed {
                        error: python_runtime_error(&format!(
                            "Unknown native worker job kind: {job_name}"
                        )),
                        transient: false,
                        details: Some(json!({
                            "failure_type": "unknown_job_kind",
                            "job_kind": job_name,
                        })),
                    }
                });
            }
        };

        let mut launcher = self
            .launcher
            .clone()
            .env_remove("MLFLOW_WORKSPACE")
            .env_remove("MLFLOW_TRACKING_USERNAME")
            .env_remove("MLFLOW_TRACKING_PASSWORD");
        if let Some(uri) = &self.tracking_uri {
            launcher = launcher.env("MLFLOW_TRACKING_URI", uri);
        }
        if let Some(uri) = &self.gateway_uri {
            launcher = launcher.env("MLFLOW_GATEWAY_URI", uri);
        }
        if let Some(workspace) = &request.workspace {
            launcher = launcher.env("MLFLOW_WORKSPACE", workspace);
        }
        if let Some(username) = subject_username(&request.subject) {
            launcher = launcher.env("MLFLOW_TRACKING_USERNAME", username);
        }
        if let Some(token) = &self.internal_gateway_token {
            launcher = launcher
                .env("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", token)
                .env("MLFLOW_TRACKING_PASSWORD", token);
        }

        let envelope = WorkerRequest {
            protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
            job_id: request.job_id,
            job_kind,
            params: request.params,
            workspace: request.workspace,
            subject: request.subject,
        };
        Box::pin(async move {
            match launcher.run_with_status(&envelope).await {
                Ok(output) => match output.status_details {
                    Some(details) => JobExecutionResult::SucceededWithDetails {
                        result: output.result,
                        details,
                    },
                    None => JobExecutionResult::Succeeded(output.result),
                },
                Err(error) => map_launch_error(job_kind, error),
            }
        })
    }
}

fn subject_username(subject: &Value) -> Option<&str> {
    subject
        .as_str()
        .or_else(|| subject.get("username").and_then(Value::as_str))
}

fn map_launch_error(job_kind: JobKind, error: WorkerLaunchError) -> JobExecutionResult {
    let function = python_function_name(job_kind);
    let (message, details) = match error {
        WorkerLaunchError::NonZeroExit { code, stderr } => {
            let displayed_code = code.map_or_else(|| "None".to_string(), |code| code.to_string());
            (
                format!(
                    "The subprocess that executes job function {function} exists with error code {displayed_code}"
                ),
                json!({"failure_type": "non_zero_exit", "exit_code": code, "stderr": stderr}),
            )
        }
        WorkerLaunchError::Signal { signal, stderr } => (
            format!(
                "The subprocess that executes job function {function} exists with error code -{signal}"
            ),
            json!({"failure_type": "signal", "signal": signal, "stderr": stderr}),
        ),
        WorkerLaunchError::MalformedOutput { message, output } => (
            format!("Native worker returned malformed output: {message}"),
            json!({"failure_type": "malformed_output", "message": message, "stdout": output}),
        ),
        WorkerLaunchError::Timeout { timeout, stderr } => (
            format!("Native worker timed out after {} ms", timeout.as_millis()),
            json!({"failure_type": "timeout", "timeout_ms": timeout.as_millis(), "stderr": stderr}),
        ),
        WorkerLaunchError::Execution {
            code,
            message,
            status_details,
        } => {
            let failure_type = match code.as_str() {
                "UNSUPPORTED_PROTOCOL_VERSION" => "protocol_version_mismatch",
                "UNKNOWN_JOB_KIND" => "unknown_job_kind",
                _ => "execution",
            };
            (
                format!("Native worker execution failed ({code}): {message}"),
                status_details.unwrap_or_else(|| {
                    json!({"failure_type": failure_type, "code": code, "message": message})
                }),
            )
        }
        WorkerLaunchError::Spawn(message) => (
            format!("Failed to spawn native worker: {message}"),
            json!({"failure_type": "spawn", "message": message}),
        ),
        WorkerLaunchError::Io(message) => (
            format!("Native worker I/O failed: {message}"),
            json!({"failure_type": "io", "message": message}),
        ),
        WorkerLaunchError::Protocol(message) => (
            format!("Native worker protocol failed: {message}"),
            json!({"failure_type": "protocol", "message": message}),
        ),
        WorkerLaunchError::InputTooLarge(bytes) => (
            format!("Native worker request exceeded its bounded input ({bytes} bytes)"),
            json!({"failure_type": "input_too_large", "bytes": bytes}),
        ),
    };
    JobExecutionResult::Failed {
        error: python_runtime_error(&message),
        transient: false,
        details: Some(details),
    }
}

fn python_runtime_error(message: &str) -> String {
    let escaped = message.replace('\\', "\\\\").replace('\'', "\\'");
    format!("RuntimeError('{escaped}')")
}

fn python_function_name(job_kind: JobKind) -> &'static str {
    match job_kind {
        JobKind::InvokeScorer => "mlflow.genai.scorers.job.invoke_scorer_job",
        JobKind::RunOnlineTraceScorer => "mlflow.genai.scorers.job.run_online_trace_scorer_job",
        JobKind::RunOnlineSessionScorer => "mlflow.genai.scorers.job.run_online_session_scorer_job",
        JobKind::OptimizePrompts => "mlflow.genai.optimize.job.optimize_prompts_job",
        JobKind::InvokeIssueDetection => "mlflow.genai.discovery.job.invoke_issue_detection_job",
        JobKind::InvokeGenaiEvaluate => "mlflow.genai.evaluation.job.invoke_genai_evaluate_job",
    }
}
