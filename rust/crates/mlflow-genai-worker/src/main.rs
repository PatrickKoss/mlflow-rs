use std::io::Read;
use std::process::ExitCode;
#[cfg(debug_assertions)]
use std::time::Duration;

use mlflow_genai::{
    execute_worker_request, WorkerRequest, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION,
};

const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
#[cfg(debug_assertions)]
const SPIKE_MODE_ENV: &str = "MLFLOW_GENAI_WORKER_SPIKE_MODE";

#[tokio::main]
async fn main() -> ExitCode {
    #[cfg(debug_assertions)]
    {
        if let Some(exit) = run_spike_hook() {
            return exit;
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

    let request: WorkerRequest = match serde_json::from_slice(&request_bytes) {
        Ok(request) => request,
        Err(error) => {
            let job_id = serde_json::from_slice::<serde_json::Value>(&request_bytes)
                .ok()
                .and_then(|value| value.get("job_id")?.as_str().map(str::to_string))
                .unwrap_or_else(|| "<unknown>".to_string());
            return write_response(WorkerResponse::failed(
                job_id,
                "INVALID_REQUEST_ENVELOPE",
                error.to_string(),
            ));
        }
    };
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
        Some("spawn-child-and-hang") => {
            let mut child = std::process::Command::new(
                std::env::current_exe().expect("current worker executable has a path"),
            )
            .env(SPIKE_MODE_ENV, "child-hang")
            .spawn()
            .expect("spike child worker starts");
            eprintln!("child_pid={}", child.id());
            let _ = child.wait();
            Some(ExitCode::FAILURE)
        }
        Some("child-hang") => loop {
            std::thread::sleep(Duration::from_secs(60));
        },
        Some(mode) => {
            eprintln!("unknown {SPIKE_MODE_ENV} value {mode:?}");
            Some(ExitCode::FAILURE)
        }
    }
}
