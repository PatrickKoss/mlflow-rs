use mlflow_store::{Db, TrackingStore, WORKSPACE_DEFAULT_NAME};
use mlflow_test_support::TempDb;
use serde_json::Value;
use sqlx::Row;
use tokio::sync::Barrier;

const WS: &str = WORKSPACE_DEFAULT_NAME;

async fn store(tag: &str) -> (TempDb, TrackingStore) {
    let temp = TempDb::new(tag).await;
    let db = temp.connect().await;
    let store = TrackingStore::new(db, "/tmp/mlflow-artifacts");
    (temp, store)
}

#[tokio::test]
async fn concurrent_register_uses_max_plus_one_without_lost_versions() {
    let (_temp, store) = store("scorer_race").await;
    let experiment_id = store
        .create_experiment(WS, "scorer-race", None, &[])
        .await
        .unwrap();
    store
        .register_scorer(WS, &experiment_id, "judge", r#"{"seed": true}"#)
        .await
        .unwrap();

    let barrier = std::sync::Arc::new(Barrier::new(8));
    let mut tasks = Vec::new();
    for index in 0..8 {
        let store = store.clone();
        let barrier = barrier.clone();
        let experiment_id = experiment_id.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            store
                .register_scorer(
                    WS,
                    &experiment_id,
                    "judge",
                    &format!(r#"{{"index": {index}}}"#),
                )
                .await
                .unwrap()
                .scorer_version
        }));
    }
    let mut versions = Vec::new();
    for task in tasks {
        versions.push(task.await.unwrap());
    }
    versions.sort_unstable();
    assert_eq!(versions, (2..=9).collect::<Vec<_>>());
}

#[tokio::test]
async fn listing_returns_latest_version_per_name_in_name_order() {
    let (_temp, store) = store("scorer_latest").await;
    let experiment_id = store
        .create_experiment(WS, "scorer-latest", None, &[])
        .await
        .unwrap();
    for payload in [r#"{"v": 1}"#, r#"{"v": 2}"#] {
        store
            .register_scorer(WS, &experiment_id, "zeta", payload)
            .await
            .unwrap();
    }
    for payload in [r#"{"v": 1}"#, r#"{"v": 2}"#, r#"{"v": 3}"#] {
        store
            .register_scorer(WS, &experiment_id, "alpha", payload)
            .await
            .unwrap();
    }

    let scorers = store.list_scorers(WS, Some(&experiment_id)).await.unwrap();
    assert_eq!(
        scorers
            .iter()
            .map(|scorer| (scorer.scorer_name.as_str(), scorer.scorer_version))
            .collect::<Vec<_>>(),
        [("alpha", 3), ("zeta", 2)]
    );
}

#[tokio::test]
async fn gateway_name_is_stored_as_id_and_resolved_on_read() {
    let (_temp, store) = store("scorer_gateway_rewrite").await;
    let experiment_id = store
        .create_experiment(WS, "scorer-gateway", None, &[])
        .await
        .unwrap();
    seed_endpoint(store.db(), "endpoint-id", "friendly-name").await;
    let payload = serde_json::json!({
        "instructions_judge_pydantic_data": {
            "model": "gateway:/friendly-name",
            "instructions": "Judge it"
        }
    })
    .to_string();
    let scorer = store
        .register_scorer(WS, &experiment_id, "judge", &payload)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&scorer.serialized_scorer).unwrap()
            ["instructions_judge_pydantic_data"]["model"],
        "gateway:/friendly-name"
    );

    let stored = stored_payload(store.db(), &scorer.scorer_id).await;
    assert_eq!(
        serde_json::from_str::<Value>(&stored).unwrap()["instructions_judge_pydantic_data"]
            ["model"],
        "gateway:/endpoint-id"
    );
    let fetched = store
        .get_scorer(WS, &experiment_id, "judge", None)
        .await
        .unwrap();
    assert_eq!(fetched.serialized_scorer, scorer.serialized_scorer);
}

async fn seed_endpoint(db: &Db, endpoint_id: &str, name: &str) {
    let sql = "INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES (?, ?, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, ?)";
    match db {
        Db::Sqlite(pool) => {
            sqlx::query(sql)
                .bind(endpoint_id)
                .bind(name)
                .bind(WS)
                .execute(pool)
                .await
                .unwrap();
        }
        Db::Postgres(pool) => {
            sqlx::query(
                "INSERT INTO endpoints (endpoint_id, name, created_by, created_at, last_updated_by, last_updated_at, routing_strategy, fallback_config_json, experiment_id, usage_tracking, workspace) VALUES ($1, $2, NULL, 1, NULL, 1, NULL, NULL, NULL, 1, $3)",
            )
                .bind(endpoint_id)
                .bind(name)
                .bind(WS)
                .execute(pool)
                .await
                .unwrap();
        }
        Db::MySql(pool) => {
            sqlx::query(sql)
                .bind(endpoint_id)
                .bind(name)
                .bind(WS)
                .execute(pool)
                .await
                .unwrap();
        }
    }
}

async fn stored_payload(db: &Db, scorer_id: &str) -> String {
    let sql = "SELECT serialized_scorer FROM scorer_versions WHERE scorer_id = ?";
    match db {
        Db::Sqlite(pool) => sqlx::query(sql)
            .bind(scorer_id)
            .fetch_one(pool)
            .await
            .unwrap()
            .get("serialized_scorer"),
        Db::Postgres(pool) => sqlx::query(&sql.replace('?', "$1"))
            .bind(scorer_id)
            .fetch_one(pool)
            .await
            .unwrap()
            .get("serialized_scorer"),
        Db::MySql(pool) => sqlx::query(sql)
            .bind(scorer_id)
            .fetch_one(pool)
            .await
            .unwrap()
            .get("serialized_scorer"),
    }
}
