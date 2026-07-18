use std::collections::HashMap;

use mlflow_store::{
    BudgetPolicyUpdate, EndpointModelConfig, EndpointUpdate, SpanInput, SpanMetricInput,
    StartTraceInput, TraceTimeRange, TrackingStore, WORKSPACE_DEFAULT_NAME,
};
use mlflow_test_support::TempDb;
use serde_json::json;

async fn store(tag: &str) -> (TempDb, TrackingStore) {
    let temp = TempDb::new(tag).await;
    let db = temp.connect().await;
    let store = TrackingStore::new(db, "/tmp/mlflow-gateway-test-artifacts");
    (temp, store)
}

fn fake_secret(value: &str) -> HashMap<String, String> {
    HashMap::from([("api_key".to_string(), value.to_string())])
}

async fn seed_gateway_cost(
    store: &TrackingStore,
    workspace: &str,
    experiment_id: &str,
    trace_id: &str,
    timestamp_ms: i64,
    cost: f64,
    gateway_tagged: bool,
) {
    store
        .start_trace(
            workspace,
            &StartTraceInput {
                trace_id: trace_id.to_string(),
                experiment_id: experiment_id.to_string(),
                request_time: timestamp_ms,
                execution_duration: Some(1),
                state: "OK".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: None,
                tags: Vec::new(),
                trace_metadata: if gateway_tagged {
                    vec![(
                        "mlflow.gateway.endpointId".to_string(),
                        "ep-fixture".to_string(),
                    )]
                } else {
                    Vec::new()
                },
                trace_metrics: Vec::new(),
            },
        )
        .await
        .unwrap();
    store
        .log_spans(
            workspace,
            experiment_id,
            &[SpanInput {
                trace_id: trace_id.to_string(),
                span_id: "0123456789abcdef".to_string(),
                parent_span_id: None,
                name: Some("provider/openai/gpt-4".to_string()),
                span_type: Some("LLM".to_string()),
                status: "OK".to_string(),
                start_time_unix_nano: timestamp_ms * 1_000_000,
                end_time_unix_nano: Some(timestamp_ms * 1_000_000 + 1),
                content: "{}".to_string(),
                dimension_attributes: None,
            }],
            &[SpanMetricInput {
                trace_id: trace_id.to_string(),
                span_id: "0123456789abcdef".to_string(),
                key: "total_cost".to_string(),
                value: cost,
            }],
            &[TraceTimeRange {
                trace_id: trace_id.to_string(),
                min_start_ms: timestamp_ms,
                max_end_ms: Some(timestamp_ms + 1),
                root_span_status: Some("OK".to_string()),
            }],
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn gateway_cost_sum_uses_total_cost_tag_time_bounds_and_workspace() {
    let (_temp, store) = store("gateway_cost_sum").await;
    let exp_default = store
        .create_experiment(WORKSPACE_DEFAULT_NAME, "gateway-cost-default", None, &[])
        .await
        .unwrap();
    let exp_other = store
        .create_experiment("other", "gateway-cost-other", None, &[])
        .await
        .unwrap();
    seed_gateway_cost(
        &store,
        WORKSPACE_DEFAULT_NAME,
        &exp_default,
        "tr-default",
        1_000,
        1.25,
        true,
    )
    .await;
    seed_gateway_cost(&store, "other", &exp_other, "tr-other", 1_500, 2.5, true).await;
    seed_gateway_cost(
        &store,
        WORKSPACE_DEFAULT_NAME,
        &exp_default,
        "tr-not-gateway",
        1_500,
        100.0,
        false,
    )
    .await;

    assert_eq!(
        store
            .sum_gateway_trace_cost(1_000, 2_000, None)
            .await
            .unwrap(),
        3.75
    );
    assert_eq!(
        store
            .sum_gateway_trace_cost(1_000, 2_000, Some(WORKSPACE_DEFAULT_NAME))
            .await
            .unwrap(),
        1.25
    );
    assert_eq!(
        store
            .sum_gateway_trace_cost(1_001, 1_500, None)
            .await
            .unwrap(),
        0.0
    );
    assert_eq!(
        store
            .sum_gateway_trace_cost(1_500, 1_501, None)
            .await
            .unwrap(),
        2.5
    );
}

#[tokio::test]
async fn secret_model_endpoint_crud_round_trip_and_cache_invalidation() {
    let (_temp, store) = store("gateway_crud").await;
    let secret = store
        .create_gateway_secret(
            WORKSPACE_DEFAULT_NAME,
            "obvious-fake-secret",
            &fake_secret("fake-value-123456"),
            Some("openai"),
            &HashMap::new(),
            Some("test-user"),
        )
        .await
        .unwrap();
    assert_eq!(secret.masked_values["api_key"], "fak...3456");
    assert_eq!(
        store
            .get_decrypted_gateway_secret(WORKSPACE_DEFAULT_NAME, &secret.secret_id)
            .await
            .unwrap(),
        json!({"api_key": "fake-value-123456"})
    );
    assert_eq!(store.secret_cache().unwrap().size(), 1);

    store
        .update_gateway_secret(
            WORKSPACE_DEFAULT_NAME,
            &secret.secret_id,
            Some(&fake_secret("rotated-fake-654321")),
            None,
            Some("test-user"),
        )
        .await
        .unwrap();
    assert_eq!(store.secret_cache().unwrap().size(), 0);

    let model = store
        .create_gateway_model_definition(
            WORKSPACE_DEFAULT_NAME,
            "fake-model-definition",
            &secret.secret_id,
            "openai",
            "fake-model",
            Some("test-user"),
        )
        .await
        .unwrap();
    let config = EndpointModelConfig {
        model_definition_id: model.model_definition_id.clone(),
        linkage_type: "PRIMARY".to_string(),
        weight: 1.0,
        fallback_order: None,
    };
    let endpoint = store
        .create_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            "fake-endpoint",
            &[config],
            Some("test-user"),
            Some("REQUEST_BASED_TRAFFIC_SPLIT"),
            None,
            None,
            true,
        )
        .await
        .unwrap();
    assert_eq!(endpoint.model_mappings.len(), 1);
    assert!(endpoint.experiment_id.is_some());
    store
        .create_gateway_endpoint_binding(
            WORKSPACE_DEFAULT_NAME,
            &endpoint.endpoint_id,
            "scorer",
            "scorer-obvious-fake",
            Some("test-user"),
        )
        .await
        .unwrap();

    let resolved = store
        .get_resolved_gateway_endpoint_config(WORKSPACE_DEFAULT_NAME, "fake-endpoint")
        .await
        .unwrap();
    assert_eq!(resolved.endpoint_id, endpoint.endpoint_id);
    assert_eq!(resolved.models.len(), 1);
    assert_eq!(
        resolved.models[0].secret_value,
        json!({"api_key": "rotated-fake-654321"})
    );
    // One encrypted entry for the secret and one for the complete endpoint
    // chain, using Python's exact endpoint_config cache key.
    assert_eq!(store.secret_cache().unwrap().size(), 2);
    let bound = store
        .get_resolved_gateway_resource_endpoint_configs(
            WORKSPACE_DEFAULT_NAME,
            "scorer",
            "scorer-obvious-fake",
        )
        .await
        .unwrap();
    assert_eq!(bound, vec![resolved]);

    store
        .update_gateway_secret(
            WORKSPACE_DEFAULT_NAME,
            &secret.secret_id,
            Some(&fake_secret("runtime-fake-112233")),
            None,
            Some("test-user"),
        )
        .await
        .unwrap();
    assert_eq!(store.secret_cache().unwrap().size(), 0);
    assert_eq!(
        store
            .get_resolved_gateway_endpoint_config(WORKSPACE_DEFAULT_NAME, "fake-endpoint")
            .await
            .unwrap()
            .models[0]
            .secret_value,
        json!({"api_key": "runtime-fake-112233"})
    );

    let updated = store
        .update_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            &endpoint.endpoint_id,
            EndpointUpdate {
                name: Some("fake-endpoint-updated"),
                usage_tracking: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(updated.name.as_deref(), Some("fake-endpoint-updated"));
    assert!(!updated.usage_tracking);
    // Python keeps the experiment ID when usage tracking is disabled.
    assert_eq!(updated.experiment_id, endpoint.experiment_id);
    assert_eq!(store.secret_cache().unwrap().size(), 0);
    assert!(store
        .get_resolved_gateway_endpoint_config(WORKSPACE_DEFAULT_NAME, "fake-endpoint")
        .await
        .is_err());
    let resolved = store
        .get_resolved_gateway_endpoint_config(WORKSPACE_DEFAULT_NAME, "fake-endpoint-updated")
        .await
        .unwrap();
    assert_eq!(resolved.endpoint_name, "fake-endpoint-updated");
    assert!(!resolved.usage_tracking);

    let scorer = store
        .register_scorer(
            WORKSPACE_DEFAULT_NAME,
            endpoint.experiment_id.as_deref().unwrap(),
            "fake-guardrail-scorer",
            r#"{"name":"obvious-fake-scorer"}"#,
        )
        .await
        .unwrap();
    let guardrail = store
        .create_gateway_guardrail(
            WORKSPACE_DEFAULT_NAME,
            "fake-guardrail",
            &scorer.scorer_id,
            scorer.scorer_version,
            "BEFORE",
            "VALIDATION",
            None,
            Some("test-user"),
        )
        .await
        .unwrap();
    store
        .add_guardrail_to_endpoint(
            WORKSPACE_DEFAULT_NAME,
            &endpoint.endpoint_id,
            &guardrail.guardrail_id,
            Some(1),
            Some("test-user"),
        )
        .await
        .unwrap();
    let config = store
        .update_endpoint_guardrail_config(
            WORKSPACE_DEFAULT_NAME,
            &endpoint.endpoint_id,
            &guardrail.guardrail_id,
            Some(2),
        )
        .await
        .unwrap();
    assert_eq!(config.execution_order, Some(2));
    assert_eq!(
        store
            .list_endpoint_guardrail_configs(WORKSPACE_DEFAULT_NAME, &endpoint.endpoint_id)
            .await
            .unwrap()
            .len(),
        1
    );
    store
        .remove_guardrail_from_endpoint(
            WORKSPACE_DEFAULT_NAME,
            &endpoint.endpoint_id,
            &guardrail.guardrail_id,
        )
        .await
        .unwrap();
    store
        .delete_gateway_guardrail(WORKSPACE_DEFAULT_NAME, &guardrail.guardrail_id)
        .await
        .unwrap();

    let budget = store
        .create_budget_policy(
            WORKSPACE_DEFAULT_NAME,
            "USD",
            25.0,
            "DAYS",
            1,
            "WORKSPACE",
            "ALERT",
            Some("test-user"),
        )
        .await
        .unwrap();
    let budget = store
        .update_budget_policy(
            WORKSPACE_DEFAULT_NAME,
            &budget.budget_policy_id,
            BudgetPolicyUpdate {
                budget_amount: Some(30.0),
                budget_action: Some("REJECT"),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(budget.budget_amount, 30.0);
    assert_eq!(budget.budget_action, "REJECT");
    assert_eq!(
        store
            .list_budget_windows(WORKSPACE_DEFAULT_NAME)
            .await
            .unwrap()
            .len(),
        1
    );
    store
        .delete_budget_policy(WORKSPACE_DEFAULT_NAME, &budget.budget_policy_id)
        .await
        .unwrap();

    store
        .delete_gateway_endpoint(WORKSPACE_DEFAULT_NAME, &endpoint.endpoint_id)
        .await
        .unwrap();
    store
        .delete_gateway_model_definition(WORKSPACE_DEFAULT_NAME, &model.model_definition_id)
        .await
        .unwrap();
    store
        .delete_gateway_secret(WORKSPACE_DEFAULT_NAME, &secret.secret_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn every_gateway_entity_read_is_workspace_scoped() {
    let (_temp, store) = store("gateway_workspace").await;
    let secret = store
        .create_gateway_secret(
            WORKSPACE_DEFAULT_NAME,
            "workspace-fake-secret",
            &fake_secret("workspace-fake-value"),
            None,
            &HashMap::new(),
            None,
        )
        .await
        .unwrap();
    assert!(store
        .get_gateway_secret_info("other-workspace", Some(&secret.secret_id), None)
        .await
        .is_err());
    assert!(store
        .get_decrypted_gateway_secret("other-workspace", &secret.secret_id)
        .await
        .is_err());

    let model = store
        .create_gateway_model_definition(
            WORKSPACE_DEFAULT_NAME,
            "workspace-fake-model",
            &secret.secret_id,
            "fake-provider",
            "fake-model",
            None,
        )
        .await
        .unwrap();
    assert!(store
        .get_gateway_model_definition("other-workspace", Some(&model.model_definition_id), None)
        .await
        .is_err());

    let endpoint = store
        .create_gateway_endpoint(
            WORKSPACE_DEFAULT_NAME,
            "workspace-fake-endpoint",
            &[EndpointModelConfig {
                model_definition_id: model.model_definition_id.clone(),
                linkage_type: "PRIMARY".to_string(),
                weight: 1.0,
                fallback_order: None,
            }],
            None,
            None,
            None,
            None,
            false,
        )
        .await
        .unwrap();
    assert!(store
        .get_gateway_endpoint("other-workspace", Some(&endpoint.endpoint_id), None)
        .await
        .is_err());
}
