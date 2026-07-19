use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use mlflow_server::assistant::AssistantProviderRequest;
use mlflow_server::assistant_providers::PermissionsConfig;
use mlflow_server::openai_compatible::{self, Config, Preset};
use serde_json::{json, Map, Value};
use tempfile::TempDir;

#[derive(Clone)]
struct ScriptState {
    turns: Arc<Mutex<VecDeque<Vec<Value>>>>,
    requests: Arc<Mutex<Vec<Value>>>,
}

async fn chat(State(state): State<ScriptState>, Json(request): Json<Value>) -> Response<Body> {
    state.requests.lock().unwrap().push(request);
    let turn = state.turns.lock().unwrap().pop_front().unwrap();
    let body = turn
        .into_iter()
        .map(|chunk| format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()))
        .chain(std::iter::once("data: [DONE]\n\n".to_string()))
        .collect::<String>();
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body))
        .unwrap()
}

async fn scripted_server(turns: Vec<Vec<Value>>) -> (String, ScriptState) {
    let state = ScriptState {
        turns: Arc::new(Mutex::new(turns.into())),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), state)
}

fn config(base_url: String) -> Config {
    Config {
        preset: Preset::Ollama,
        model: "fixture-model".to_string(),
        base_url: Some(base_url),
        api_key: None,
        permissions: PermissionsConfig::default(),
    }
}

fn request(
    root: &std::path::Path,
    session_id: Option<String>,
    context: Map<String, Value>,
) -> AssistantProviderRequest {
    AssistantProviderRequest {
        prompt: "perform fixture".to_string(),
        tracking_uri: "http://127.0.0.1:5000".to_string(),
        session_id,
        mlflow_session_id: "00000000-0000-0000-0000-000000000020".to_string(),
        cwd: Some(root.to_path_buf()),
        context,
        config: None,
    }
}

fn tool_delta(index: usize, id: &str, name: &str, arguments: &str) -> Value {
    json!({"choices":[{"delta":{"role":"assistant","tool_calls":[{
        "index":index,"id":id,"function":{"name":name,"arguments":arguments}
    }]},"index":0}]})
}

fn text_delta(text: &str) -> Value {
    json!({"choices":[{"delta":{"role":"assistant","content":text},"index":0}]})
}

fn done_session(events: &[mlflow_server::assistant::AssistantEvent]) -> String {
    events
        .iter()
        .rev()
        .find(|event| event.event_type == "done")
        .and_then(|event| event.data.get("session_id"))
        .and_then(Value::as_str)
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn scripted_multi_tool_loop_persists_python_shaped_history() {
    let fixture = TempDir::new().unwrap();
    let (base, state) = scripted_server(vec![
        vec![
            tool_delta(
                0,
                "write-1",
                "Write",
                r#"{"file_path":"note.txt","content":"alpha"}"#,
            ),
            tool_delta(1, "read-1", "Read", r#"{"file_path":"note.txt"}"#),
        ],
        vec![text_delta("Done")],
    ])
    .await;
    let events: Vec<_> =
        openai_compatible::stream(config(base), request(fixture.path(), None, Map::new()))
            .collect()
            .await;
    assert_eq!(
        events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        [
            "message",
            "message",
            "message",
            "message",
            "stream_event",
            "done"
        ]
    );
    assert_eq!(
        std::fs::read_to_string(fixture.path().join("note.txt")).unwrap(),
        "alpha"
    );
    let history: Value = serde_json::from_str(&done_session(&events)).unwrap();
    assert_eq!(
        history.as_array().unwrap().last().unwrap()["content"],
        "Done"
    );
    let requests = state.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "tool")
            .count(),
        2
    );
}

#[tokio::test]
async fn permission_pause_and_resume_continue_without_duplicate_tool_use() {
    let fixture = TempDir::new().unwrap();
    let (base, _) = scripted_server(vec![vec![tool_delta(
        0,
        "bash-1",
        "Bash",
        r#"{"command":"printf resumed"}"#,
    )]])
    .await;
    let first: Vec<_> =
        openai_compatible::stream(config(base), request(fixture.path(), None, Map::new()))
            .collect()
            .await;
    assert_eq!(
        first
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        ["message", "permission_request", "done"]
    );

    let (base, _) = scripted_server(vec![vec![text_delta("Resumed")]]).await;
    let mut context = Map::new();
    context.insert("tool_decisions".to_string(), json!({"bash-1":"allow"}));
    let second: Vec<_> = openai_compatible::stream(
        config(base),
        request(fixture.path(), Some(done_session(&first)), context),
    )
    .collect()
    .await;
    assert_eq!(
        second
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        ["message", "stream_event", "done"]
    );
    assert_eq!(
        second[0].data["message"]["content"][0]["content"],
        "resumed"
    );
}

#[tokio::test]
async fn trim_boundary_drops_whole_oldest_turn_group() {
    let fixture = TempDir::new().unwrap();
    let big = "x".repeat(180_000);
    let history = json!([
        {"role":"system","content":"sys"},
        {"role":"user","content":format!("old-{big}")},
        {"role":"assistant","content":format!("old-answer-{big}")},
        {"role":"user","content":format!("new-{big}")}
    ]);
    let (base, _) = scripted_server(vec![vec![text_delta("final")]]).await;
    let events: Vec<_> = openai_compatible::stream(
        config(base),
        request(
            fixture.path(),
            Some(serde_json::to_string(&history).unwrap()),
            Map::new(),
        ),
    )
    .collect()
    .await;
    let final_history: Value = serde_json::from_str(&done_session(&events)).unwrap();
    let messages = final_history.as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert!(!messages.iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|content| content.starts_with("old-"))
    }));
    assert!(done_session(&events).len() <= 500 * 1024);
}
