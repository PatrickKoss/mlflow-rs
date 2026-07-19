use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::{WorkerRequest, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION};

#[derive(Debug, Clone, PartialEq)]
pub struct WorkerOutput {
    pub result: serde_json::Value,
    pub status_details: Option<serde_json::Value>,
}

const DEFAULT_MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const TRUNCATION_MARKER: &str = "\n...[truncated]";

/// Subprocess launcher shared by the future DB job runner.
#[derive(Debug, Clone)]
pub struct WorkerLauncher {
    program: PathBuf,
    timeout: Option<Duration>,
    max_input_bytes: usize,
    max_output_bytes: usize,
    clear_environment: bool,
    environment: BTreeMap<OsString, OsString>,
    removed_environment: BTreeSet<OsString>,
}

impl WorkerLauncher {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            timeout: Some(Duration::from_secs(60)),
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            clear_environment: false,
            environment: BTreeMap::new(),
            removed_environment: BTreeSet::new(),
        }
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Let the DB runner own timeout policy and signal it by dropping this
    /// future. This is the production mode; a launcher-local timeout remains
    /// available for standalone supervision tests.
    pub fn without_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Clear inherited variables before applying explicit propagation entries.
    pub fn clean_environment(mut self) -> Self {
        self.clear_environment = true;
        self
    }

    pub fn env(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.removed_environment.remove(key.as_ref());
        self.environment
            .insert(key.as_ref().to_os_string(), value.as_ref().to_os_string());
        self
    }

    pub fn env_remove(mut self, key: impl AsRef<OsStr>) -> Self {
        self.environment.remove(key.as_ref());
        self.removed_environment.insert(key.as_ref().to_os_string());
        self
    }

    pub fn max_input_bytes(mut self, limit: usize) -> Self {
        self.max_input_bytes = limit;
        self
    }

    /// Set the independent stdout and stderr capture cap.
    pub fn max_output_bytes(mut self, limit: usize) -> Self {
        self.max_output_bytes = limit;
        self
    }

    pub async fn run(
        &self,
        request: &WorkerRequest,
    ) -> Result<serde_json::Value, WorkerLaunchError> {
        Ok(self.run_with_status(request).await?.result)
    }

    pub async fn run_with_status(
        &self,
        request: &WorkerRequest,
    ) -> Result<WorkerOutput, WorkerLaunchError> {
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
        for key in &self.removed_environment {
            command.env_remove(key);
        }
        command.envs(&self.environment);
        configure_process_group(&mut command);

        let mut child = command
            .spawn()
            .map_err(|error| WorkerLaunchError::Spawn(error.to_string()))?;
        let process_id = child.id().ok_or_else(|| {
            WorkerLaunchError::Spawn("worker process has no process ID".to_string())
        })?;
        let mut process_guard = ProcessGroupGuard::new(process_id);
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

        let status = if let Some(timeout) = self.timeout {
            match tokio::time::timeout(timeout, child.wait()).await {
                Ok(result) => result.map_err(|error| WorkerLaunchError::Io(error.to_string()))?,
                Err(_) => {
                    kill_process_group(process_id, &mut child).await;
                    let _ = child.wait().await;
                    process_guard.disarm();
                    let (_, stderr) = join_output_tasks(stdout_task, stderr_task).await?;
                    return Err(WorkerLaunchError::Timeout {
                        timeout,
                        stderr: stderr.render(),
                    });
                }
            }
        } else {
            child
                .wait()
                .await
                .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
        };
        process_guard.disarm();

        let (stdout, stderr) = join_output_tasks(stdout_task, stderr_task).await?;
        if let Some(signal) = exit_signal(&status) {
            return Err(WorkerLaunchError::Signal {
                signal,
                stderr: stderr.render(),
            });
        }
        if !status.success() {
            return Err(WorkerLaunchError::NonZeroExit {
                code: status.code(),
                stderr: stderr.render(),
            });
        }
        input_task
            .await
            .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
            .map_err(|error| WorkerLaunchError::Io(error.to_string()))?;
        if stdout.truncated {
            return Err(WorkerLaunchError::MalformedOutput {
                message: format!(
                    "stdout exceeded the {}-byte capture limit and was truncated",
                    self.max_output_bytes
                ),
                output: stdout.render(),
            });
        }

        let response: WorkerResponse = serde_json::from_slice(&stdout.bytes).map_err(|error| {
            WorkerLaunchError::MalformedOutput {
                message: error.to_string(),
                output: stdout.render(),
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
    #[error("native worker exited non-zero (code {code:?}): {stderr}")]
    NonZeroExit { code: Option<i32>, stderr: String },
    #[error("native worker died from signal {signal}: {stderr}")]
    Signal { signal: i32, stderr: String },
    #[error("native worker output was malformed ({message}): {output}")]
    MalformedOutput { message: String, output: String },
    #[error("native worker timed out after {timeout:?}: {stderr}")]
    Timeout { timeout: Duration, stderr: String },
    #[error("native worker execution failed ({code}): {message}")]
    Execution {
        code: String,
        message: String,
        status_details: Option<serde_json::Value>,
    },
}

fn validate_response(
    request: &WorkerRequest,
    response: WorkerResponse,
) -> Result<WorkerOutput, WorkerLaunchError> {
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
        WorkerResponse::Succeeded {
            result,
            status_details,
            ..
        } => Ok(WorkerOutput {
            result,
            status_details: status_details.map(|details| *details),
        }),
        WorkerResponse::Failed {
            error,
            status_details,
            ..
        } => Err(WorkerLaunchError::Execution {
            code: error.code,
            message: error.message,
            status_details: status_details.map(|details| *details),
        }),
    }
}

#[derive(Debug)]
struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

impl BoundedOutput {
    fn render(&self) -> String {
        let mut output = String::from_utf8_lossy(&self.bytes).into_owned();
        if self.truncated {
            output.push_str(TRUNCATION_MARKER);
        }
        output
    }
}

async fn read_bounded<R>(mut reader: R, limit: usize) -> Result<BoundedOutput, std::io::Error>
where
    R: AsyncRead + Unpin,
{
    let mut bytes = Vec::with_capacity(limit.min(8 * 1024));
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        let retained = read.min(remaining);
        bytes.extend_from_slice(&chunk[..retained]);
        truncated |= retained < read;
    }
    Ok(BoundedOutput { bytes, truncated })
}

async fn join_output_tasks(
    stdout_task: tokio::task::JoinHandle<Result<BoundedOutput, std::io::Error>>,
    stderr_task: tokio::task::JoinHandle<Result<BoundedOutput, std::io::Error>>,
) -> Result<(BoundedOutput, BoundedOutput), WorkerLaunchError> {
    let stdout = stdout_task
        .await
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?;
    let stderr = stderr_task
        .await
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?
        .map_err(|error| WorkerLaunchError::Io(error.to_string()))?;
    Ok((stdout, stderr))
}

/// The runner signals cancel and timeout by dropping the execution future.
/// Synchronously killing the process group here prevents grandchildren from
/// escaping even when no async cleanup can be awaited.
struct ProcessGroupGuard {
    process_id: Option<u32>,
}

impl ProcessGroupGuard {
    fn new(process_id: u32) -> Self {
        Self {
            process_id: Some(process_id),
        }
    }

    fn disarm(&mut self) {
        self.process_id = None;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if let Some(process_id) = self.process_id {
            kill_process_group_now(process_id);
        }
    }
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    command.as_std_mut().process_group(0);
    let max_fd = inherited_fd_limit();
    // SAFETY: the closure only invokes async-signal-safe fcntl/syscall
    // operations between fork and exec. Marking descriptors CLOEXEC preserves
    // std's exec-error pipe until exec while closing every inherited FD in the
    // worker image.
    unsafe {
        command
            .as_std_mut()
            .pre_exec(move || close_inherited_fds(max_fd));
    }
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn inherited_fd_limit() -> i32 {
    let mut limit = nix::libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is valid writable storage for getrlimit.
    if unsafe { nix::libc::getrlimit(nix::libc::RLIMIT_NOFILE, &mut limit) } == 0 {
        limit.rlim_cur.min(i32::MAX as nix::libc::rlim_t) as i32
    } else {
        65_536
    }
}

#[cfg(unix)]
fn close_inherited_fds(max_fd: i32) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        // CLOSE_RANGE_CLOEXEC avoids closing Rust's exec-error pipe before it
        // can report an execve failure to the parent.
        // SAFETY: close_range has no pointer arguments and is async-signal-safe.
        let result = unsafe {
            nix::libc::syscall(
                nix::libc::SYS_close_range,
                3_u32,
                u32::MAX,
                nix::libc::CLOSE_RANGE_CLOEXEC,
            )
        };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if !matches!(
            error.raw_os_error(),
            Some(code) if code == nix::libc::ENOSYS || code == nix::libc::EINVAL
        ) {
            return Err(error);
        }
    }

    for fd in 3..max_fd {
        // SAFETY: fcntl operates on the integer descriptor and does not retain
        // pointers. EBADF simply means this descriptor was already closed.
        let flags = unsafe { nix::libc::fcntl(fd, nix::libc::F_GETFD) };
        if flags >= 0 {
            let result =
                unsafe { nix::libc::fcntl(fd, nix::libc::F_SETFD, flags | nix::libc::FD_CLOEXEC) };
            if result == -1 {
                return Err(std::io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

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

#[cfg(unix)]
fn kill_process_group_now(process_id: u32) {
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;

    if let Ok(process_id) = i32::try_from(process_id) {
        let _ = killpg(Pid::from_raw(process_id), Signal::SIGKILL);
    }
}

#[cfg(not(unix))]
async fn kill_process_group(_process_id: u32, child: &mut tokio::process::Child) {
    let _ = child.kill().await;
}

#[cfg(not(unix))]
fn kill_process_group_now(_process_id: u32) {}

#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;

    status.signal()
}

#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}
