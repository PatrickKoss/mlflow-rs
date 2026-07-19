//! JSON-lines-free recorder entry point used by the Python/Rust differential.
//!
//! Input and output are single JSON documents on stdin/stdout so provider SSE
//! frames (which contain newlines) remain byte-preserving JSON strings.

use std::io::{self, Read};

use mlflow_server::assistant_providers::{
    health, spawn, HealthError, ProviderConfig, ProviderKind, StreamRequest,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct RecorderRequest {
    action: String,
    #[serde(default)]
    provider: Option<ProviderKind>,
    #[serde(default)]
    config: ProviderConfig,
    #[serde(default)]
    stream: Option<StreamRequest>,
    #[serde(default)]
    cancel_after_events: Option<usize>,
    #[serde(default)]
    health_error: Option<String>,
    #[serde(default)]
    detail: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: RecorderRequest = serde_json::from_str(&input)?;
    let output = match request.action.as_str() {
        "stream" => record_stream(request).await?,
        "health" => record_health(request).await?,
        "health_mapping" => record_health_mapping(request)?,
        action => anyhow::bail!("unknown recorder action: {action}"),
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

async fn record_stream(request: RecorderRequest) -> anyhow::Result<Value> {
    let provider = request
        .provider
        .ok_or_else(|| anyhow::anyhow!("stream action requires provider"))?;
    let stream = request
        .stream
        .ok_or_else(|| anyhow::anyhow!("stream action requires stream request"))?;
    let mut spawned = spawn(provider, request.config, stream).await?;
    let mut frames = Vec::new();
    while let Some(event) = spawned.events.next().await {
        frames.push(event.to_sse_frame());
        if request.cancel_after_events == Some(frames.len()) {
            spawned.handle.cancel()?;
        }
    }
    Ok(json!({"frames": frames}))
}

async fn record_health(request: RecorderRequest) -> anyhow::Result<Value> {
    let provider = request
        .provider
        .ok_or_else(|| anyhow::anyhow!("health action requires provider"))?;
    Ok(match health(provider).await {
        Ok(()) => json!({"status": 200, "body": {"status": "ok"}}),
        Err(error) => json!({"status": error.status_code(), "body": error.body()}),
    })
}

fn record_health_mapping(request: RecorderRequest) -> anyhow::Result<Value> {
    let detail = request.detail.unwrap_or_default();
    let error = match request.health_error.as_deref() {
        Some("not_implemented") => HealthError::NotImplemented(detail),
        Some("cli_not_installed") => HealthError::CliNotInstalled(detail),
        Some("not_authenticated") => HealthError::NotAuthenticated(detail),
        value => anyhow::bail!("unknown health error: {value:?}"),
    };
    Ok(json!({"status": error.status_code(), "body": error.body()}))
}
