use mlflow_genai::{
    execute_worker_request, JobKind, WorkerRequest, WorkerResponse, NATIVE_WORKER_PROTOCOL_VERSION,
};

#[test]
fn unknown_job_kind_is_rejected_by_the_closed_enum() {
    let envelope = serde_json::json!({
        "protocol_version": NATIVE_WORKER_PROTOCOL_VERSION,
        "job_id": "job-unknown",
        "job_kind": "run_arbitrary_code",
        "params": {},
        "workspace": "default",
        "subject": {"username": "alice"}
    });
    assert!(serde_json::from_value::<WorkerRequest>(envelope).is_err());
}

#[tokio::test]
async fn unknown_protocol_version_fails_before_params_are_parsed() {
    let request = WorkerRequest {
        protocol_version: NATIVE_WORKER_PROTOCOL_VERSION + 1,
        job_id: "job-new-version".to_string(),
        job_kind: JobKind::InvokeScorer,
        params: serde_json::Value::String("deliberately not invoke_scorer params".to_string()),
        workspace: "default".to_string(),
        subject: serde_json::json!({"username": "alice"}),
    };
    let response = execute_worker_request(&request).await;
    let WorkerResponse::Failed { error, .. } = response else {
        panic!("unsupported version must fail");
    };
    assert_eq!(error.code, "UNSUPPORTED_PROTOCOL_VERSION");
}
