use mlflow_store::{
    AssessmentSource, AssessmentValue, IssueUpdate, LabelSchemaInput, LabelSchemaType,
    LabelSchemaUpdate, NewAssessment, StartTraceInput, TrackingStore,
};
use mlflow_test_support::TempDb;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
}

async fn experiment(store: &TrackingStore, name: &str) -> String {
    store.create_experiment(WS, name, None, &[]).await.unwrap()
}

async fn trace(store: &TrackingStore, experiment_id: &str, trace_id: &str) {
    store
        .start_trace(
            WS,
            &StartTraceInput {
                trace_id: trace_id.to_string(),
                experiment_id: experiment_id.to_string(),
                request_time: 1,
                execution_duration: Some(1),
                state: "OK".to_string(),
                client_request_id: Some(trace_id.to_string()),
                request_preview: None,
                response_preview: None,
                tags: vec![],
                trace_metadata: vec![],
                trace_metrics: vec![],
                assessments: vec![],
            },
        )
        .await
        .unwrap();
}

async fn assessment(store: &TrackingStore, trace_id: &str, name: &str, value: AssessmentValue) {
    store
        .create_assessment(
            WS,
            NewAssessment {
                trace_id: trace_id.to_string(),
                name: name.to_string(),
                value,
                source: AssessmentSource {
                    source_type: "CODE".to_string(),
                    source_id: None,
                },
                run_id: None,
                span_id: None,
                rationale: None,
                metadata: None,
                create_time_ms: None,
                last_update_time_ms: None,
                assessment_id: None,
                overrides: None,
            },
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn issue_crud_search_and_trace_count_join_semantics() {
    let temp = TempDb::new("issues-crud-count").await;
    let store = store(&temp).await;
    let experiment_id = experiment(&store, "issues-crud-count").await;
    let issue = store
        .create_issue(
            WS,
            &experiment_id,
            "Latency",
            "Slow responses",
            "pending",
            Some("high"),
            &["database".to_string()],
            None,
            &["quality".to_string()],
            Some("alice"),
        )
        .await
        .unwrap();
    assert!(issue.issue_id.starts_with("iss-"));
    assert_eq!(issue.root_causes, ["database"]);
    assert!(issue.trace_count.is_none());

    let updated = store
        .update_issue(
            WS,
            &issue.issue_id,
            IssueUpdate {
                name: Some("Latency updated"),
                status: Some("resolved"),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(updated.name, "Latency updated");
    assert_eq!(updated.status, "resolved");

    trace(&store, &experiment_id, "tr-issue-1").await;
    trace(&store, &experiment_id, "tr-issue-2").await;
    assessment(
        &store,
        "tr-issue-1",
        &issue.issue_id,
        AssessmentValue::Issue {
            issue_name: issue.name.clone(),
        },
    )
    .await;
    // A second matching issue assessment for the same trace is deduplicated by
    // COUNT(DISTINCT trace_id).
    assessment(
        &store,
        "tr-issue-1",
        &issue.issue_id,
        AssessmentValue::Issue {
            issue_name: issue.name.clone(),
        },
    )
    .await;
    assessment(
        &store,
        "tr-issue-2",
        &issue.issue_id,
        AssessmentValue::Issue {
            issue_name: issue.name.clone(),
        },
    )
    .await;
    // Same name but the wrong assessment type must not join.
    assessment(
        &store,
        "tr-issue-2",
        &issue.issue_id,
        AssessmentValue::Feedback {
            value_json: "true".to_string(),
            error: None,
        },
    )
    .await;

    let without_count = store
        .search_issues(
            WS,
            Some(&experiment_id),
            Some("status = 'resolved'"),
            None,
            None,
            false,
        )
        .await
        .unwrap();
    assert_eq!(without_count.issues.len(), 1);
    assert!(without_count.issues[0].trace_count.is_none());

    let with_count = store
        .search_issues(
            WS,
            Some(&experiment_id),
            Some("status = 'resolved'"),
            None,
            None,
            true,
        )
        .await
        .unwrap();
    assert_eq!(with_count.issues[0].trace_count, Some(2));
    assert!(store.get_issue("other", &issue.issue_id).await.is_err());
}

fn pass_fail() -> LabelSchemaInput {
    LabelSchemaInput::PassFail {
        positive_label: "Correct".to_string(),
        negative_label: "Incorrect".to_string(),
    }
}

#[tokio::test]
async fn label_schema_crud_uniqueness_scope_and_input_type_immutability() {
    let temp = TempDb::new("label-schema-crud").await;
    let store = store(&temp).await;
    let exp1 = experiment(&store, "label-schema-exp-1").await;
    let exp2 = experiment(&store, "label-schema-exp-2").await;

    let schema = store
        .create_label_schema(
            WS,
            &exp1,
            "Correctness",
            LabelSchemaType::Feedback,
            &pass_fail(),
            Some("Judge the response"),
            true,
        )
        .await
        .unwrap();
    assert!(schema.schema_id.starts_with("ls-"));
    assert_eq!(
        store
            .get_label_schema_by_name(WS, &exp1, "Correctness")
            .await
            .unwrap()
            .schema_id,
        schema.schema_id
    );

    let duplicate = store
        .create_label_schema(
            WS,
            &exp1,
            "Correctness",
            LabelSchemaType::Expectation,
            &pass_fail(),
            None,
            false,
        )
        .await
        .unwrap_err();
    assert_eq!(
        duplicate.message,
        format!("Label schema with name 'Correctness' already exists for experiment '{exp1}'.")
    );
    // Names are unique per experiment, not globally.
    store
        .create_label_schema(
            WS,
            &exp2,
            "Correctness",
            LabelSchemaType::Expectation,
            &pass_fail(),
            None,
            false,
        )
        .await
        .unwrap();

    let immutable = store
        .update_label_schema(
            WS,
            &schema.schema_id,
            LabelSchemaUpdate {
                input: Some(&LabelSchemaInput::Numeric {
                    min_value: Some(0.0),
                    max_value: Some(1.0),
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(
        immutable.message,
        "A label schema's input type cannot be changed after creation (existing: \
         InputPassFail, got: InputNumeric)."
    );

    let updated = store
        .update_label_schema(
            WS,
            &schema.schema_id,
            LabelSchemaUpdate {
                name: Some("Answer correctness"),
                enable_comment: Some(false),
                input: Some(&LabelSchemaInput::PassFail {
                    positive_label: "Good".to_string(),
                    negative_label: "Bad".to_string(),
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(updated.name, "Answer correctness");
    assert!(!updated.enable_comment);

    let listed = store
        .list_label_schemas(WS, &exp1, 100, None)
        .await
        .unwrap();
    assert_eq!(
        listed
            .schemas
            .iter()
            .filter(|schema| schema.is_default)
            .count(),
        1
    );
    assert_eq!(listed.schemas.len(), 2);

    store
        .delete_label_schema(WS, &schema.schema_id)
        .await
        .unwrap();
    assert!(store.get_label_schema(WS, &schema.schema_id).await.is_err());
    store.delete_label_schema(WS, "ls-missing").await.unwrap();
}
