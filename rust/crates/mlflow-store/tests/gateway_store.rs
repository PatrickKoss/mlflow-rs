use std::collections::HashMap;

use mlflow_store::{
    BudgetPolicyUpdate, EndpointModelConfig, EndpointUpdate, TrackingStore, WORKSPACE_DEFAULT_NAME,
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
