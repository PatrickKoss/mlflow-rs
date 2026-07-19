use std::io::{self, Read};
use std::path::PathBuf;

use futures::StreamExt;
use mlflow_server::assistant::AssistantProviderRequest;
use mlflow_server::assistant_providers::PermissionsConfig;
use mlflow_server::openai_compatible::{self, Config, Preset};
use serde::Deserialize;
use serde_json::{json, Map, Value};

#[derive(Deserialize)]
struct Input {
    base_url: String,
    model: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    permissions: PermissionsConfig,
    prompt: String,
    tracking_uri: String,
    #[serde(default)]
    session_id: Option<String>,
    mlflow_session_id: String,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    context: Map<String, Value>,
}

#[tokio::main]
async fn main() {
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw).unwrap();
    let input: Input = serde_json::from_str(&raw).unwrap();
    let config = Config {
        preset: Preset::Ollama,
        model: input.model,
        base_url: Some(input.base_url),
        api_key: input.api_key,
        permissions: input.permissions,
    };
    let request = AssistantProviderRequest {
        prompt: input.prompt,
        tracking_uri: input.tracking_uri,
        session_id: input.session_id,
        mlflow_session_id: input.mlflow_session_id,
        cwd: input.cwd,
        context: input.context,
        config: None,
    };
    let frames = openai_compatible::stream(config, request)
        .map(|event| String::from_utf8(event.to_sse().to_vec()).unwrap())
        .collect::<Vec<_>>()
        .await;
    println!(
        "{}",
        serde_json::to_string(&json!({"frames":frames})).unwrap()
    );
}
