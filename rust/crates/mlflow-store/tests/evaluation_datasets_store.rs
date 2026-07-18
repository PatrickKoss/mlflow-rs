use base64::Engine;
use mlflow_store::{python_json_dumps, TrackingStore};
use mlflow_test_support::TempDb;
use serde_json::{json, Map, Value};

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
}

async fn experiment(store: &TrackingStore, name: &str) -> String {
    store.create_experiment(WS, name, None, &[]).await.unwrap()
}

fn tags(values: &[(&str, &str)]) -> Map<String, Value> {
    values
        .iter()
        .map(|(key, value)| ((*key).to_string(), Value::String((*value).to_string())))
        .collect()
}

#[tokio::test]
async fn evaluation_dataset_crud_search_and_associations() {
    let temp = TempDb::new("eval-dataset-crud").await;
    let store = store(&temp).await;
    let exp1 = experiment(&store, "eval-exp-1").await;
    let exp2 = experiment(&store, "eval-exp-2").await;
    let exp3 = experiment(&store, "eval-exp-3").await;

    let created = store
        .create_evaluation_dataset(
            WS,
            "eval-alpha",
            &tags(&[("priority", "high"), ("mlflow.user", "alice")]),
            std::slice::from_ref(&exp1),
        )
        .await
        .unwrap();
    assert!(created.dataset_id.starts_with("d-"));
    assert_eq!(created.created_by.as_deref(), Some("alice"));
    assert_eq!(
        created.experiment_ids.as_deref(),
        Some([exp1.clone()].as_slice())
    );

    let fetched = store
        .get_evaluation_dataset(WS, &created.dataset_id)
        .await
        .unwrap();
    assert_eq!(fetched.name, "eval-alpha");
    assert_eq!(fetched.tags["priority"], "high");
    assert!(fetched.experiment_ids.is_none());
    assert!(store
        .get_evaluation_dataset("other", &created.dataset_id)
        .await
        .is_err());
    assert!(store
        .upsert_evaluation_records(
            "other",
            &created.dataset_id,
            &[json!({"inputs": {"hidden": true}})],
        )
        .await
        .is_err());

    store
        .set_evaluation_dataset_tags(
            WS,
            &created.dataset_id,
            &tags(&[("priority", "low"), ("team", "genai")]),
        )
        .await
        .unwrap();
    let fetched = store
        .get_evaluation_dataset(WS, &created.dataset_id)
        .await
        .unwrap();
    assert_eq!(fetched.tags["priority"], "low");
    assert_eq!(fetched.tags["team"], "genai");

    let result = store
        .search_evaluation_datasets(
            WS,
            &[],
            Some("name LIKE 'eval-%' AND tags.priority = 'low'"),
            1000,
            &["name ASC".to_string()],
            None,
        )
        .await
        .unwrap();
    assert_eq!(result.datasets.len(), 1);
    assert_eq!(result.datasets[0].dataset_id, created.dataset_id);

    let lowercase = store
        .search_evaluation_datasets(WS, &[], Some("name like 'eval-%'"), 1000, &[], None)
        .await
        .unwrap_err();
    assert!(lowercase.message.contains("Invalid comparator"));

    let tag_order = store
        .search_evaluation_datasets(WS, &[], None, 1000, &["tags.priority".to_string()], None)
        .await
        .unwrap_err();
    assert_eq!(tag_order.message, "Invalid order_by entity: tag");

    store
        .add_evaluation_dataset_to_experiments(
            WS,
            &created.dataset_id,
            &[exp2.clone(), exp3.clone()],
        )
        .await
        .unwrap();
    let mut ids = store
        .get_evaluation_dataset_experiment_ids(WS, &created.dataset_id)
        .await
        .unwrap();
    ids.sort();
    let mut expected = vec![exp1.clone(), exp2.clone(), exp3.clone()];
    expected.sort();
    assert_eq!(ids, expected);

    store
        .remove_evaluation_dataset_from_experiments(
            WS,
            &created.dataset_id,
            std::slice::from_ref(&exp2),
        )
        .await
        .unwrap();
    assert!(!store
        .get_evaluation_dataset_experiment_ids(WS, &created.dataset_id)
        .await
        .unwrap()
        .contains(&exp2));

    store
        .delete_evaluation_dataset_tag(WS, &created.dataset_id, "team")
        .await
        .unwrap();
    assert!(!store
        .get_evaluation_dataset(WS, &created.dataset_id)
        .await
        .unwrap()
        .tags
        .contains_key("team"));

    store
        .delete_evaluation_dataset(WS, &created.dataset_id)
        .await
        .unwrap();
    assert!(store
        .get_evaluation_dataset(WS, &created.dataset_id)
        .await
        .is_err());
    store
        .delete_evaluation_dataset(WS, &created.dataset_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn evaluation_record_upsert_dedup_schema_digest_and_tokens() {
    let temp = TempDb::new("eval-records").await;
    let store = store(&temp).await;
    let dataset = store
        .create_evaluation_dataset(WS, "records", &Map::new(), &[])
        .await
        .unwrap();
    let initial_digest = dataset.digest.clone();

    let result = store
        .upsert_evaluation_records(
            WS,
            &dataset.dataset_id,
            &[
                json!({
                    "inputs": {"question": "MLflow?", "count": 1},
                    "outputs": "first",
                    "expectations": {"answer": "platform"},
                    "tags": {"version": "v1"},
                    "source": {"source_type": "trace", "source_data": {"trace_id": "tr-1"}}
                }),
                json!({
                    "inputs": {"count": 1, "question": "MLflow?"},
                    "outputs": "second",
                    "expectations": {"score": 1.0},
                    "tags": {"version": "v2"}
                }),
                json!({"inputs": {"question": "Rust?"}, "expectations": {"answer": true}}),
            ],
        )
        .await
        .unwrap();
    assert_eq!(result.inserted, 2);
    assert_eq!(result.updated, 1);

    let updated = store
        .get_evaluation_dataset(WS, &dataset.dataset_id)
        .await
        .unwrap();
    assert_ne!(updated.digest, initial_digest);
    assert_eq!(
        serde_json::from_str::<Value>(updated.profile.as_deref().unwrap()).unwrap(),
        json!({"num_records": 2})
    );
    let schema: Value = serde_json::from_str(updated.schema.as_deref().unwrap()).unwrap();
    assert_eq!(schema["inputs"]["count"], "integer");
    assert_eq!(schema["expectations"]["answer"], "string");
    assert_eq!(schema["outputs"], "string");

    let first = store
        .load_evaluation_records(WS, &dataset.dataset_id, 1, None)
        .await
        .unwrap();
    assert_eq!(first.records.len(), 1);
    let token = first.next_page_token.clone().expect("cursor token");
    let python_cursor = base64::engine::general_purpose::STANDARD.encode(format!(
        "{}:{}",
        first.records[0].created_time.unwrap(),
        first.records[0].dataset_record_id
    ));
    assert_eq!(token, python_cursor);
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&token)
        .unwrap();
    assert!(String::from_utf8(decoded).unwrap().contains(":dr-"));
    let second = store
        .load_evaluation_records(WS, &dataset.dataset_id, 1, Some(&token))
        .await
        .unwrap();
    assert_eq!(second.records.len(), 1);
    assert_ne!(
        first.records[0].dataset_record_id,
        second.records[0].dataset_record_id
    );

    // Python's legacy fallback passes a bare decimal offset token.
    let legacy = store
        .load_evaluation_records(WS, &dataset.dataset_id, 1, Some("1"))
        .await
        .unwrap();
    assert_eq!(
        legacy.records[0].dataset_record_id,
        second.records[0].dataset_record_id
    );

    let mlflow_record = [first.records[0].clone(), second.records[0].clone()]
        .into_iter()
        .find(|record| record.inputs["question"] == "MLflow?")
        .unwrap();
    assert_eq!(mlflow_record.outputs, Some(json!("second")));
    assert_eq!(
        mlflow_record.expectations.unwrap(),
        json!({"answer": "platform", "score": 1.0})
    );
    assert_eq!(mlflow_record.tags.unwrap(), json!({"version": "v2"}));
    assert_eq!(mlflow_record.source_id.as_deref(), Some("tr-1"));
    assert_eq!(
        mlflow_record.source.as_ref().unwrap()["source_type"],
        "TRACE"
    );

    let invalid_source = store
        .upsert_evaluation_records(
            WS,
            &dataset.dataset_id,
            &[json!({
                "inputs": {"question": "Invalid source"},
                "source": {"source_type": "database"}
            })],
        )
        .await
        .unwrap_err();
    assert!(invalid_source
        .message
        .contains("Invalid dataset record source type: DATABASE"));

    let deleted = store
        .delete_evaluation_records(
            WS,
            &dataset.dataset_id,
            &[first.records[0].dataset_record_id.clone()],
        )
        .await
        .unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(
        serde_json::from_str::<Value>(
            store
                .get_evaluation_dataset(WS, &dataset.dataset_id)
                .await
                .unwrap()
                .profile
                .as_deref()
                .unwrap()
        )
        .unwrap(),
        json!({"num_records": 1})
    );
}

#[tokio::test]
async fn evaluation_search_accepts_python_offset_tokens_and_scopes_workspace() {
    let temp = TempDb::new("eval-search-token").await;
    let store = store(&temp).await;
    for name in ["a", "b", "c"] {
        store
            .create_evaluation_dataset(WS, name, &Map::new(), &[])
            .await
            .unwrap();
    }
    store
        .create_evaluation_dataset("other", "hidden", &Map::new(), &[])
        .await
        .unwrap();

    // base64(json.dumps({"offset": 1})) written by Python.
    let python_token = "eyJvZmZzZXQiOiAxfQ==";
    let page = store
        .search_evaluation_datasets(
            WS,
            &[],
            None,
            1,
            &["name ASC".to_string()],
            Some(python_token),
        )
        .await
        .unwrap();
    assert_eq!(page.datasets[0].name, "b");
    assert_eq!(
        page.next_page_token.as_deref(),
        Some("eyJvZmZzZXQiOiAyfQ==")
    );

    assert_eq!(
        python_json_dumps(&json!({"b": "é", "a": 1}), true),
        r#"{"a": 1, "b": "\u00e9"}"#
    );
}
