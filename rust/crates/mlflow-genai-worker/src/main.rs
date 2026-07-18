use std::io::Read;
use std::process::ExitCode;
#[cfg(debug_assertions)]
use std::time::Duration;

use mlflow_genai::{
    decode_worker_request, execute_worker_request, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION,
};

const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
#[cfg(debug_assertions)]
const SPIKE_MODE_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_MODE";
#[cfg(debug_assertions)]
const SPIKE_PID_FILE_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_PID_FILE";
#[cfg(debug_assertions)]
const SPIKE_FD_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_FD";

#[tokio::main]
async fn main() -> ExitCode {
    // Internal grandchild mode has no protocol input of its own. It is only
    // reachable from the debug-only, already-validated parent fault hook.
    #[cfg(debug_assertions)]
    if std::env::var(SPIKE_MODE_ENV).as_deref() == Ok("child-hang") {
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    let mut request_bytes = Vec::new();
    if let Err(error) = std::io::stdin()
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut request_bytes)
    {
        return write_response(WorkerResponse::failed(
            "<unknown>".to_string(),
            "REQUEST_IO_ERROR",
            error.to_string(),
        ));
    }
    if request_bytes.len() as u64 > MAX_REQUEST_BYTES {
        return write_response(WorkerResponse::failed(
            "<unknown>".to_string(),
            "REQUEST_TOO_LARGE",
            format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
        ));
    }

    let request = match decode_worker_request(&request_bytes) {
        Ok(request) => request,
        Err(response) => return write_response(response),
    };

    // Fault injection runs only after the version and closed kind allowlist
    // have been validated, proving negative envelopes cannot execute hooks.
    #[cfg(debug_assertions)]
    if let Some(exit) = run_spike_hook() {
        return exit;
    }

    write_response(execute_worker_request(&request).await)
}

fn write_response(response: WorkerResponse) -> ExitCode {
    match serde_json::to_writer(std::io::stdout(), &response) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("failed to write native worker response: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Debug-only fault injection used to prove T15.4's process supervision modes.
#[cfg(debug_assertions)]
fn run_spike_hook() -> Option<ExitCode> {
    match std::env::var(SPIKE_MODE_ENV).ok().as_deref() {
        None => None,
        Some("nonzero") => Some(ExitCode::from(23)),
        Some("signal") => std::process::abort(),
        Some("malformed") => {
            print!("not-a-{NATIVE_WORKER_PROTOCOL_VERSION}-envelope");
            Some(ExitCode::SUCCESS)
        }
        Some("stdout-large") => {
            print!("{}", "x".repeat(5 * 1024 * 1024));
            Some(ExitCode::SUCCESS)
        }
        Some("stderr-large-nonzero") => {
            eprint!("{}", "x".repeat(5 * 1024 * 1024));
            Some(ExitCode::from(24))
        }
        Some("delay") => {
            std::thread::sleep(Duration::from_millis(200));
            None
        }
        Some("assert-fd-closed") => {
            let fd = std::env::var(SPIKE_FD_ENV).expect("spike FD is configured");
            if std::path::Path::new(&format!("/proc/self/fd/{fd}")).exists() {
                eprintln!("inherited_fd={fd}");
                return Some(ExitCode::FAILURE);
            }
            None
        }
        Some("spawn-child-and-hang") => {
            let mut child = std::process::Command::new(
                std::env::current_exe().expect("current worker executable has a path"),
            )
            .env(SPIKE_MODE_ENV, "child-hang")
            .spawn()
            .expect("spike child worker starts");
            eprintln!("child_pid={}", child.id());
            if let Some(path) = std::env::var_os(SPIKE_PID_FILE_ENV) {
                std::fs::write(path, child.id().to_string()).expect("spike child PID file writes");
            }
            let _ = child.wait();
            Some(ExitCode::FAILURE)
        }
        Some("child-hang") => unreachable!("child mode is handled before protocol input"),
        Some(mode) => {
            eprintln!("unknown {SPIKE_MODE_ENV} value {mode:?}");
            Some(ExitCode::FAILURE)
        }
    }
}
