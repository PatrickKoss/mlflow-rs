use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::{WorkerRequest, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION};

const DEFAULT_MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

/// Subprocess launcher shared by the future DB job runner.
#[derive(Debug, Clone)]
pub struct WorkerLauncher {
    program: PathBuf,
    timeout: Duration,
    max_input_bytes: usize,
    max_output_bytes: usize,
    clear_environment: bool,
    environment: BTreeMap<OsString, OsString>,
}

impl WorkerLauncher {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            timeout: Duration::from_secs(60),
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            clear_environment: false,
            environment: BTreeMap::new(),
        }
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Clear inherited variables before applying explicit propagation entries.
    pub fn clean_environment(mut self) -> Self {
        self.clear_environment = true;
        self
    }

    pub fn env(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.environment
            .insert(key.as_ref().to_os_string(), value.as_ref().to_os_string());
        self
    }

    pub async fn run(
        &self,
        request: &WorkerRequest,
    ) -> Result<serde_json::Value, WorkerLaunchError> {
        let encoded = serde_json::to_vec(request)
            .map_err(|error| WorkerLaunchError::Protocol(error.to_string()))?;
        if encoded.len() > self.max_input_bytes {
            return Err(WorkerLaunchError::InputTooLarge(encoded.len()));
        }

        let mut command = Command::new(&self.program);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if self.clear_environment {
            command.env_clear();
        }
        command.envs(&self.environment);
        configure_process_group(&mut command);

        let mut child = command
            .spawn()
            .map_err(|error| WorkerLaunchError::Spawn(error.to_string()))?;
        let process_id = child.id().ok_or_else(|| {
            WorkerLaunchError::Spawn("worker process has no process ID".to_string())
        })?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerLaunchError::Io("worker stdin was not piped".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| WorkerLaunchError::Io("worker stdout was not piped".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| WorkerLaunchError::Io("worker stderr was not piped".to_string()))?;

        let input_task = tokio::spawn(async move {
            stdin.write_all(&encoded).await?;
            stdin.shutdown().await
        });
        let stdout_task = tokio::spawn(read_bounded(stdout, self.max_output_bytes));
        let stderr_task = tokio::spawn(read_bounded(stderr, self.max_output_bytes));

        let status = match tokio::time::timeout(self.timeout, child.wait()).await {
            Ok(result) => result.map_err(|error| WorkerLaunchError::Io(error.to_string()))?,
            Err(_) => {
                kill_process_group(process_id, &mut child).await;
                let _ = child.wait().await;
                let (_, stderr, _) = join_output_tasks(stdout_task, stderr_task).await?;
                return Err(WorkerLaunchError::Timeout {
                    timeout: self.timeout,
                    stderr: String::from_utf8_lossy(&stderr).into_owned(),
                });
            }
        };

        let (stdout, stderr, output_overflow) = join_output_tasks(stdout_task, stderr_task).await?;
        if let Some(signal) = exit_signal(&status) {
            return Err(WorkerLaunchError::Signal {
                signal,
                stderr: String::from_utf8_lossy(&stderr).into_owned(),
            });
        }
        if !status.success() {
            return Err(WorkerLaunchError::NonZeroExit {
                code: status.code(),
                stderr: String::from_utf8_lossy(&stderr).into_owned(),
            });
        }
        input_task
            .await
            .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
            .map_err(|error| WorkerLaunchError::Io(error.to_string()))?;
        if output_overflow {
            return Err(WorkerLaunchError::OutputTooLarge(self.max_output_bytes));
        }

        let response: WorkerResponse = serde_json::from_slice(&stdout).map_err(|error| {
            WorkerLaunchError::MalformedOutput {
                message: error.to_string(),
                output: String::from_utf8_lossy(&stdout).into_owned(),
            }
        })?;
        validate_response(request, response)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerLaunchError {
    #[error("failed to spawn native worker: {0}")]
    Spawn(String),
    #[error("native worker I/O failed: {0}")]
    Io(String),
    #[error("native worker protocol error: {0}")]
    Protocol(String),
    #[error("native worker request exceeded the {0}-byte limit")]
    InputTooLarge(usize),
    #[error("native worker output exceeded the {0}-byte limit")]
    OutputTooLarge(usize),
    #[error("native worker exited non-zero (code {code:?}): {stderr}")]
    NonZeroExit { code: Option<i32>, stderr: String },
    #[error("native worker died from signal {signal}: {stderr}")]
    Signal { signal: i32, stderr: String },
    #[error("native worker output was malformed ({message}): {output}")]
    MalformedOutput { message: String, output: String },
    #[error("native worker timed out after {timeout:?}: {stderr}")]
    Timeout { timeout: Duration, stderr: String },
    #[error("native worker execution failed ({code}): {message}")]
    Execution { code: String, message: String },
}

fn validate_response(
    request: &WorkerRequest,
    response: WorkerResponse,
) -> Result<serde_json::Value, WorkerLaunchError> {
    let (protocol_version, job_id) = match &response {
        WorkerResponse::Succeeded {
            protocol_version,
            job_id,
            ..
        }
        | WorkerResponse::Failed {
            protocol_version,
            job_id,
            ..
        } => (*protocol_version, job_id),
    };
    if protocol_version != NATIVE_WORKER_PROTOCOL_VERSION {
        return Err(WorkerLaunchError::Protocol(format!(
            "response version {protocol_version} does not match {}",
            NATIVE_WORKER_PROTOCOL_VERSION
        )));
    }
    if job_id != &request.job_id {
        return Err(WorkerLaunchError::Protocol(format!(
            "response job ID {job_id:?} does not match request job ID {:?}",
            request.job_id
        )));
    }
    match response {
        WorkerResponse::Succeeded { result, .. } => Ok(result),
        WorkerResponse::Failed { error, .. } => Err(WorkerLaunchError::Execution {
            code: error.code,
            message: error.message,
        }),
    }
}

async fn read_bounded<R>(reader: R, limit: usize) -> Result<(Vec<u8>, bool), std::io::Error>
where
    R: AsyncRead + Unpin,
{
    let mut bytes = Vec::new();
    reader
        .take(u64::try_from(limit).unwrap_or(u64::MAX) + 1)
        .read_to_end(&mut bytes)
        .await?;
    let overflow = bytes.len() > limit;
    if overflow {
        bytes.truncate(limit);
    }
    Ok((bytes, overflow))
}

async fn join_output_tasks(
    stdout_task: tokio::task::JoinHandle<Result<(Vec<u8>, bool), std::io::Error>>,
    stderr_task: tokio::task::JoinHandle<Result<(Vec<u8>, bool), std::io::Error>>,
) -> Result<(Vec<u8>, Vec<u8>, bool), WorkerLaunchError> {
    let (stdout, stdout_overflow) = stdout_task
        .await
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?;
    let (stderr, stderr_overflow) = stderr_task
        .await
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?;
    Ok((stdout, stderr, stdout_overflow || stderr_overflow))
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
async fn kill_process_group(process_id: u32, child: &mut tokio::process::Child) {
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;

    if let Ok(process_id) = i32::try_from(process_id) {
        let _ = killpg(Pid::from_raw(process_id), Signal::SIGKILL);
    } else {
        let _ = child.kill().await;
    }
}

#[cfg(not(unix))]
async fn kill_process_group(_process_id: u32, child: &mut tokio::process::Child) {
    let _ = child.kill().await;
}

#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;

    status.signal()
}

#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}
