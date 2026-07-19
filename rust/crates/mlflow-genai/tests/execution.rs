use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use mlflow_genai::{
    execute_worker_request, EvalItem, JobKind, MemoryExample, ScorerExecutor, SerializedScorer,
    WorkerRequest, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION,
};
use serde_json::{json, Value};

#[derive(Clone)]
struct Script {
    requests: Arc<Mutex<Vec<Value>>>,
    responses: Arc<Mutex<VecDeque<Value>>>,
}

async fn scripted(State(script): State<Script>, Json(request): Json<Value>) -> Json<Value> {
    script.requests.lock().unwrap().push(request);
    Json(script.responses.lock().unwrap().pop_front().unwrap())
}

async fn server(responses: Vec<Value>) -> (String, Script) {
    let script = Script {
        requests: Arc::new(Mutex::new(Vec::new())),
        responses: Arc::new(Mutex::new(responses.into())),
    };
    let app = Router::new()
        .route("/", post(scripted))
        .with_state(script.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}/"), script)
}

fn scorer(value: Value) -> SerializedScorer {
    SerializedScorer::from_value(value).unwrap()
}

fn builtin(class_name: &str, data: Value) -> SerializedScorer {
    scorer(json!({
        "name": class_name,
        "builtin_scorer_class": class_name,
        "builtin_scorer_pydantic_data": data,
    }))
}

fn completion(result: Value, rationale: &str) -> Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": json!({
            "result": result,
            "rationale": rationale,
        }).to_string()}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7}
    })
}

fn trace() -> Value {
    json!({
        "info": {"trace_id": "tr-1", "request_time": "2026-01-01T00:00:00Z"},
        "data": {"spans": [
            {
                "span_id": "root", "name": "agent", "start_time_unix_nano": 1,
                "attributes": {
                    "mlflow.spanInputs": "{\"question\":\"capital?\"}",
                    "mlflow.spanOutputs": "\"Paris\"",
                    "mlflow.spanType": "CHAIN"
                }
            },
            {
                "span_id": "ret", "parent_span_id": "root", "name": "retrieve",
                "start_time_unix_nano": 2,
                "attributes": {
                    "mlflow.spanInputs": "{\"query\":\"capital\"}",
                    "mlflow.spanOutputs": "[{\"page_content\":\"Paris is the capital of France.\",\"metadata\":{\"doc_uri\":\"fake://doc\"}}]",
                    "mlflow.spanType": "RETRIEVER"
                }
            },
            {
                "span_id": "llm", "parent_span_id": "root", "name": "chat",
                "start_time_unix_nano": 3,
                "attributes": {
                    "mlflow.spanInputs": "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"description\":\"Look up a city\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}",
                    "mlflow.spanType": "LLM"
                }
            },
            {
                "span_id": "tool", "parent_span_id": "root", "name": "lookup",
                "start_time_unix_nano": 4,
                "attributes": {
                    "mlflow.spanInputs": "{\"city\":\"Paris\"}",
                    "mlflow.spanOutputs": "\"France\"",
                    "mlflow.spanType": "TOOL"
                }
            }
        ]}
    })
}

#[tokio::test]
async fn deterministic_builtins_are_value_exact() {
    let executor = ScorerExecutor::new();
    let cases = [
        (
            builtin(
                "ResponseLength",
                json!({"min_length": 2, "max_length": 4, "unit": "words"}),
            ),
            Some(json!("two words")),
            "yes",
            "Output length (2 words) is within bounds",
        ),
        (
            builtin("ResponseLength", json!({"max_length": 2, "unit": "chars"})),
            Some(json!("é🙂x")),
            "no",
            "Output length (3 chars) exceeds the maximum (2 chars)",
        ),
        (
            builtin(
                "RegexMatch",
                json!({"pattern": "^answer:", "case_insensitive": true}),
            ),
            Some(json!("Answer: 42")),
            "yes",
            "Output matches pattern '^answer:'",
        ),
        (
            builtin("RegexMatch", json!({"pattern": r"^(\w+) \1$"})),
            Some(json!("echo echo")),
            "yes",
            r"Output matches pattern '^(\\w+) \\1$'",
        ),
        (
            builtin("PIIDetection", json!({"pii_types": ["email", "phone"]})),
            Some(json!("alice@example.com and 555-123-4567")),
            "no",
            "Detected PII: email, phone",
        ),
        (
            builtin("PIIDetection", json!({})),
            None,
            "no",
            "No outputs provided to evaluate.",
        ),
    ];
    for (scorer, outputs, expected, rationale) in cases {
        let feedback = executor
            .execute(
                &scorer,
                &EvalItem {
                    outputs,
                    ..EvalItem::default()
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(feedback.value, json!(expected));
        assert_eq!(feedback.rationale, rationale);
        let source = feedback.source.unwrap();
        assert_eq!(source.source_type, "CODE");
        assert_eq!(source.source_id.as_deref(), Some("default"));
    }
}

#[tokio::test]
async fn instructions_request_and_feedback_match_python_gateway_shape() {
    let (url, script) = server(vec![completion(
        json!("yes"),
        "Let's think step by step. concise",
    )])
    .await;
    let scorer = scorer(json!({
        "name": "concise",
        "instructions_judge_pydantic_data": {
            "instructions": "Return yes when {{ inputs }} is answered by {{ outputs }}",
            "model": "openai:/fake-chat",
            "feedback_value_type": {"enum": ["yes", "no"], "title": "Result", "type": "string"},
            "inference_params": {"temperature": 0.0, "max_tokens": 50}
        }
    }));
    let feedback = ScorerExecutor::new()
        .execute(
            &scorer,
            &EvalItem {
                inputs: Some(json!({"question": "capital?"})),
                outputs: Some(json!("Paris")),
                ..EvalItem::default()
            },
            Some(&url),
        )
        .await
        .unwrap();
    assert_eq!(feedback.value, json!("yes"));
    assert_eq!(feedback.rationale, "concise");
    assert_eq!(
        feedback.source.unwrap().source_id.as_deref(),
        Some("openai:/fake-chat")
    );
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/instructions_request.json")).unwrap();
    assert_eq!(requests[0], oracle);
    assert_eq!(requests[0]["model"], "fake-chat");
    assert_eq!(requests[0]["temperature"], 0.0);
    assert_eq!(requests[0]["max_tokens"], 50);
    assert_eq!(requests[0]["messages"][0]["role"], "system");
    assert_eq!(
        requests[0]["messages"][1]["content"],
        "inputs: {\n  \"question\": \"capital?\"\n}\noutputs: \"Paris\""
    );
    assert_eq!(
        requests[0]["response_format"]["json_schema"]["name"],
        "ResponseFormat"
    );
}

#[tokio::test]
async fn exact_tool_call_feedback_is_value_and_rationale_exact() {
    let executor = ScorerExecutor::new();
    let expectations = Some(json!({
        "expected_tool_calls": [{"name": "search", "arguments": {"query": "Paris"}}]
    }));
    let unordered = executor
        .execute(
            &builtin(
                "ToolCallCorrectness",
                json!({"should_exact_match": true, "should_consider_ordering": false}),
            ),
            &EvalItem {
                trace: Some(trace()),
                expectations: expectations.clone(),
                ..EvalItem::default()
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(unordered.value, json!("no"));
    assert_eq!(
        unordered.rationale,
        "Missing: {'search({\"query\": \"Paris\"})'}; Unexpected: {'lookup({\"city\": \"Paris\"})'}"
    );
    assert_eq!(
        unordered.source.unwrap().source_id.as_deref(),
        Some("default")
    );

    let ordered = executor
        .execute(
            &builtin(
                "ToolCallCorrectness",
                json!({"should_exact_match": true, "should_consider_ordering": true}),
            ),
            &EvalItem {
                trace: Some(trace()),
                expectations,
                ..EvalItem::default()
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(ordered.value, json!("no"));
    assert_eq!(
        ordered.rationale,
        "Tool calls do not match in order: Position 1: expected search({\"query\": \"Paris\"}), got lookup({\"city\": \"Paris\"})"
    );
}

#[tokio::test]
async fn trace_judge_runs_native_tool_loop() {
    let (url, script) = server(vec![
        json!({"choices": [{"message": {
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "get_root_span", "arguments": "{}"
            }}]
        }}]}),
        completion(json!(true), "inspected root"),
    ])
    .await;
    let scorer = scorer(json!({
        "name": "trace_check",
        "instructions_judge_pydantic_data": {
            "instructions": "Evaluate the execution in {{ trace }}",
            "model": "openai:/fake-chat",
            "feedback_value_type": {"type": "boolean"}
        }
    }));
    let feedback = ScorerExecutor::new()
        .execute(
            &scorer,
            &EvalItem {
                trace: Some(trace()),
                ..EvalItem::default()
            },
            Some(&url),
        )
        .await
        .unwrap();
    assert_eq!(feedback.value, json!(true));
    let requests = script.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 6);
    assert_eq!(requests[0]["tool_choice"], "auto");
    assert_eq!(requests[1]["messages"][2]["tool_calls"][0]["id"], "call-1");
    assert_eq!(requests[1]["messages"][3]["role"], "tool");
    assert_eq!(requests[1]["messages"][3]["tool_call_id"], "call-1");
}

#[tokio::test]
async fn memory_judge_retrieves_with_scripted_embeddings() {
    let (embedding_url, embedding_script) = server(vec![
        json!({"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]}),
        json!({"data": [{"embedding": [0.0, 1.0]}]}),
    ])
    .await;
    let (gateway_url, gateway_script) = server(vec![completion(json!("no"), "aligned")]).await;
    let scorer = scorer(json!({
        "name": "memory",
        "memory_augmented_judge_data": {
            "base_judge": {
                "name": "memory",
                "instructions_judge_pydantic_data": {
                    "instructions": "Evaluate {{ outputs }}",
                    "model": "openai:/fake-chat",
                    "feedback_value_type": {"enum": ["yes", "no"], "type": "string"}
                }
            },
            "episodic_trace_ids": ["ex-a", "ex-b"],
            "semantic_memory": [{"guideline_text": "Prefer exact facts"}],
            "retrieval_k": 1,
            "embedding_model": "openai:/fake-embedding",
            "embedding_dim": 2
        }
    }));
    let feedback = ScorerExecutor::new()
        .execute_all(
            &scorer,
            &EvalItem {
                outputs: Some(json!("new")),
                memory_examples: Some(vec![
                    MemoryExample {
                        trace_id: "ex-a".to_string(),
                        inputs: None,
                        outputs: Some(json!("first")),
                        expectations: None,
                        trace: None,
                        feedback: Some(json!({"result": "yes"})),
                    },
                    MemoryExample {
                        trace_id: "ex-b".to_string(),
                        inputs: None,
                        outputs: Some(json!("second")),
                        expectations: None,
                        trace: None,
                        feedback: Some(json!({"result": "no"})),
                    },
                ]),
                ..EvalItem::default()
            },
            Some(&gateway_url),
            Some(&embedding_url),
        )
        .await
        .unwrap();
    assert_eq!(feedback[0].value, json!("no"));
    assert_eq!(
        feedback[0].metadata.as_ref().unwrap()["retrieved_example_trace_ids"],
        json!(["ex-b"])
    );
    let embedding_requests = embedding_script.requests.lock().unwrap();
    assert_eq!(embedding_requests.len(), 2);
    assert_eq!(embedding_requests[0]["model"], "fake-embedding");
    assert_eq!(embedding_requests[0]["input"], json!(["first", "second"]));
    assert_eq!(embedding_requests[1]["input"], json!(["new"]));
    let gateway_requests = gateway_script.requests.lock().unwrap();
    let system = gateway_requests[0]["messages"][0]["content"]
        .as_str()
        .unwrap();
    assert!(system.contains("Prefer exact facts"));
    assert!(system.contains("ex-b") || system.contains("second"));
}

#[tokio::test]
async fn every_manifest_builtin_has_a_native_execution_path() {
    let responses = (0..40)
        .map(|_| completion(json!("yes"), "scripted"))
        .collect();
    let (url, _) = server(responses).await;
    let executor = ScorerExecutor::new();
    let item = EvalItem {
        inputs: Some(json!({"question": "capital?"})),
        outputs: Some(json!("Paris")),
        expectations: Some(json!({
            "expected_response": "The capital is Paris",
            "expected_facts": ["Paris is the capital"],
            "guidelines": ["Be concise"],
            "expected_tool_calls": [{"name": "lookup", "arguments": {"city": "Paris"}}]
        })),
        trace: Some(trace()),
        session: Some(vec![trace()]),
        memory_examples: None,
    };
    let classes = [
        "Completeness",
        "ConversationCompleteness",
        "ConversationalGuidelines",
        "ConversationalRoleAdherence",
        "ConversationalSafety",
        "ConversationalToolCallEfficiency",
        "Correctness",
        "Equivalence",
        "ExpectationsGuidelines",
        "Fluency",
        "Guidelines",
        "KnowledgeRetention",
        "RelevanceToQuery",
        "RetrievalGroundedness",
        "RetrievalRelevance",
        "RetrievalSufficiency",
        "Safety",
        "Summarization",
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "UserFrustration",
    ];
    for class_name in classes {
        let instructions = match class_name {
            "Fluency" => "Evaluate {{ outputs }}",
            "Completeness" | "Summarization" => "Evaluate {{ inputs }} and {{ outputs }}",
            "ConversationCompleteness"
            | "ConversationalGuidelines"
            | "ConversationalRoleAdherence"
            | "ConversationalSafety"
            | "ConversationalToolCallEfficiency"
            | "KnowledgeRetention"
            | "UserFrustration" => "Evaluate {{ conversation }}",
            "ToolCallCorrectness" => {
                "Evaluate {{ request }} {{ available_tools }} {{ tools_called }}"
            }
            "ToolCallEfficiency" => {
                "Evaluate {{ request }} {{ available_tools }} {{ tools_called }}"
            }
            "Correctness" => "Evaluate {{ input }} {{ output }} against {{ ground_truth }}",
            "Equivalence" => "Evaluate {{ output }} against {{ expected_output }}",
            "Guidelines" | "ExpectationsGuidelines" => {
                "Evaluate {{ guidelines }} with {{ guidelines_context }}"
            }
            "RelevanceToQuery" => "Evaluate {{ input }} and {{ output }}",
            "RetrievalGroundedness" => {
                "Evaluate {{ input }} {{ output }} using {{ retrieval_context }}"
            }
            "RetrievalSufficiency" => {
                "Evaluate {{ input }} {{ ground_truth }} using {{ retrieval_context }}"
            }
            "RetrievalRelevance" | "Safety" => "scripted prompt",
            _ => unreachable!(),
        };
        let mut data = json!({
            "instructions": instructions,
            "model": "openai:/fake-chat"
        });
        if class_name == "Guidelines" || class_name == "ConversationalGuidelines" {
            data["guidelines"] = json!(["Be concise"]);
        }
        let mut class_item = item.clone();
        if class_name == "Correctness" {
            class_item.expectations = Some(json!({
                "expected_response": "The capital is Paris",
                "guidelines": ["Be concise"],
                "expected_tool_calls": [{"name": "lookup", "arguments": {"city": "Paris"}}]
            }));
        }
        let feedback = executor
            .execute_all(&builtin(class_name, data), &class_item, Some(&url), None)
            .await
            .unwrap_or_else(|error| panic!("{class_name} did not execute: {error}"));
        assert!(!feedback.is_empty(), "{class_name} returned no feedback");
    }
}

#[tokio::test]
async fn non_fixture_worker_dispatches_to_the_real_scorer_executor() {
    let request = WorkerRequest {
        protocol_version: NATIVE_WORKER_PROTOCOL_VERSION,
        job_id: "native-scorer".to_string(),
        job_kind: JobKind::InvokeScorer,
        params: json!({
            "serialized_scorer": json!({
                "name": "length",
                "builtin_scorer_class": "ResponseLength",
                "builtin_scorer_pydantic_data": {"max_length": 5, "unit": "chars"}
            }).to_string(),
            "outputs": "four"
        }),
        workspace: Some("default".to_string()),
        subject: json!({"type": "user", "id": "fixture-user"}),
    };
    let response = execute_worker_request(&request).await;
    let WorkerResponse::Succeeded { result, .. } = response else {
        panic!("real worker execution failed: {response:?}");
    };
    assert_eq!(result["value"], "yes");
    assert_eq!(
        result["rationale"],
        "Output length (4 chars) is within bounds"
    );
}
