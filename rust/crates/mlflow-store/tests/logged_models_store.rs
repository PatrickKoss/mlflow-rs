//! Behavioral integration tests for logged models (plan T2.9): CRUD, the
//! finalize state machine, tags/params, `search_logged_models` (filters,
//! dataset-scoped ordering, pagination), and workspace isolation.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

use mlflow_store::{
    DatasetFilter, LoggedModelKv, LoggedModelMetricInput, LoggedModelOrderByInput,
    LoggedModelStatus, MetricInput, TrackingStore,
};
use mlflow_test_support::TempDb;

const WS: &str = "default";
const WS2: &str = "team-b";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}_{}",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    )
}

async fn new_experiment_ws(s: &TrackingStore, workspace: &str) -> String {
    s.create_experiment(workspace, &format!("e{}", uuid_like()), None, &[])
        .await
        .unwrap()
}

async fn new_experiment(s: &TrackingStore) -> String {
    new_experiment_ws(s, WS).await
}

async fn new_run_in_ws(s: &TrackingStore, workspace: &str, exp_id: &str) -> String {
    s.create_run(workspace, exp_id, None, Some(0), Some("run"), &[])
        .await
        .unwrap()
        .info
        .run_id
}

async fn new_run_in(s: &TrackingStore, exp_id: &str) -> String {
    new_run_in_ws(s, WS, exp_id).await
}

fn kv(key: &str, value: &str) -> LoggedModelKv {
    LoggedModelKv {
        key: key.to_string(),
        value: value.to_string(),
    }
}

fn metric(key: &str, value: f64, ts: i64, step: i64) -> LoggedModelMetricInput {
    LoggedModelMetricInput {
        key: key.to_string(),
        value,
        timestamp: ts,
        step,
        dataset_name: None,
        dataset_digest: None,
    }
}

fn metric_ds(
    key: &str,
    value: f64,
    ts: i64,
    step: i64,
    dataset_name: &str,
    dataset_digest: &str,
) -> LoggedModelMetricInput {
    LoggedModelMetricInput {
        key: key.to_string(),
        value,
        timestamp: ts,
        step,
        dataset_name: Some(dataset_name.to_string()),
        dataset_digest: Some(dataset_digest.to_string()),
    }
}

// ---------------------------------------------------------------------------
// create / get / delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_get_logged_model_defaults() {
    let tmp = TempDb::new("create").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    assert!(model.model_id.starts_with("m-"));
    assert_eq!(model.model_id.len(), 2 + 32); // "m-" + uuid4 hex (32 chars)
    assert_eq!(model.experiment_id, exp);
    assert!(!model.name.is_empty()); // random name generated
    assert!(model
        .artifact_location
        .ends_with(&format!("models/{}/artifacts", model.model_id)));
    assert_eq!(model.status, LoggedModelStatus::Pending.to_int());
    assert!(model.tags.is_empty());
    assert!(model.params.is_empty());
    assert!(model.metrics.is_empty());
    assert_eq!(model.creation_timestamp, model.last_updated_timestamp);

    let fetched = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(fetched, model);
}

#[tokio::test]
async fn create_logged_model_explicit_name_source_run_params_tags() {
    let tmp = TempDb::new("create2").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;

    let model = s
        .create_logged_model(
            WS,
            &exp,
            Some("my-model"),
            Some(&run_id),
            &[kv("owner", "alice")],
            &[kv("lr", "0.01")],
            Some("sklearn"),
        )
        .await
        .unwrap();

    assert_eq!(model.name, "my-model");
    assert_eq!(model.source_run_id.as_deref(), Some(run_id.as_str()));
    assert_eq!(model.model_type.as_deref(), Some("sklearn"));
    assert_eq!(model.tags.len(), 1);
    assert_eq!(model.tags[0].key, "owner");
    assert_eq!(model.tags[0].value, "alice");
    assert_eq!(model.params.len(), 1);
    assert_eq!(model.params[0].key, "lr");
    assert_eq!(model.params[0].value, "0.01");
}

#[tokio::test]
async fn create_logged_model_invalid_name_rejected() {
    let tmp = TempDb::new("badname").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    for bad in ["", "a/b", "a:b", "a.b", "a%b", "a\"b", "a'b"] {
        let err = s
            .create_logged_model(WS, &exp, Some(bad), None, &[], &[], None)
            .await
            .unwrap_err();
        assert!(err.message.contains("Invalid model name"), "{bad}: {err:?}");
    }
}

#[tokio::test]
async fn create_logged_model_requires_active_experiment() {
    let tmp = TempDb::new("inactive_exp").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    s.delete_experiment(WS, &exp).await.unwrap();

    let err = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap_err();
    assert!(err.message.contains("must be in the 'active' state"));
}

#[tokio::test]
async fn create_logged_model_missing_experiment() {
    let tmp = TempDb::new("missing_exp").await;
    let s = store(&tmp).await;

    let err = s
        .create_logged_model(WS, "999999", None, None, &[], &[], None)
        .await
        .unwrap_err();
    assert!(err.message.contains("No Experiment with id=999999 exists"));
}

#[tokio::test]
async fn get_logged_model_not_found() {
    let tmp = TempDb::new("get_missing").await;
    let s = store(&tmp).await;
    let err = s
        .get_logged_model(WS, "m-doesnotexist", false)
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        "Logged model with ID 'm-doesnotexist' not found."
    );
}

#[tokio::test]
async fn delete_logged_model_then_get_excludes_by_default() {
    let tmp = TempDb::new("delete").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.delete_logged_model(WS, &model.model_id).await.unwrap();

    let err = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap_err();
    assert!(err.message.contains("not found"));

    // allow_deleted=true still returns it.
    let still = s.get_logged_model(WS, &model.model_id, true).await.unwrap();
    assert_eq!(still.model_id, model.model_id);
}

#[tokio::test]
async fn delete_logged_model_missing() {
    let tmp = TempDb::new("delete_missing").await;
    let s = store(&tmp).await;
    let err = s.delete_logged_model(WS, "m-nope").await.unwrap_err();
    assert!(err.message.contains("not found"));
}

// ---------------------------------------------------------------------------
// finalize state machine
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_sets_status_and_updates_timestamp_no_state_guard() {
    let tmp = TempDb::new("finalize").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    assert_eq!(model.status, LoggedModelStatus::Pending.to_int());

    let ready = s
        .finalize_logged_model(WS, &model.model_id, LoggedModelStatus::Ready)
        .await
        .unwrap();
    assert_eq!(ready.status, LoggedModelStatus::Ready.to_int());
    assert!(ready.last_updated_timestamp >= model.last_updated_timestamp);

    // Python has no PENDING-only guard: re-finalizing an already-finalized
    // model to a different (or the same) status succeeds with no error.
    let failed = s
        .finalize_logged_model(WS, &model.model_id, LoggedModelStatus::Failed)
        .await
        .unwrap();
    assert_eq!(failed.status, LoggedModelStatus::Failed.to_int());

    let re_ready = s
        .finalize_logged_model(WS, &model.model_id, LoggedModelStatus::Ready)
        .await
        .unwrap();
    assert_eq!(re_ready.status, LoggedModelStatus::Ready.to_int());
}

#[tokio::test]
async fn finalize_unknown_model_not_found() {
    let tmp = TempDb::new("finalize_missing").await;
    let s = store(&tmp).await;
    let err = s
        .finalize_logged_model(WS, "m-nope", LoggedModelStatus::Ready)
        .await
        .unwrap_err();
    assert_eq!(err.message, "Logged model with ID 'm-nope' not found.");
}

// ---------------------------------------------------------------------------
// tags / params
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_and_delete_logged_model_tags() {
    let tmp = TempDb::new("tags").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.set_logged_model_tags(WS, &model.model_id, &[kv("k1", "v1"), kv("k2", "v2")])
        .await
        .unwrap();
    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(got.tags.len(), 2);

    // Upsert overwrites the value for an existing key.
    s.set_logged_model_tags(WS, &model.model_id, &[kv("k1", "updated")])
        .await
        .unwrap();
    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(got.tags.len(), 2);
    assert!(got
        .tags
        .iter()
        .any(|t| t.key == "k1" && t.value == "updated"));

    s.delete_logged_model_tag(WS, &model.model_id, "k1")
        .await
        .unwrap();
    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(got.tags.len(), 1);
    assert_eq!(got.tags[0].key, "k2");
}

#[tokio::test]
async fn delete_logged_model_tag_missing_key_errors() {
    let tmp = TempDb::new("tag_missing").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    let err = s
        .delete_logged_model_tag(WS, &model.model_id, "nope")
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        format!(
            "No tag with key 'nope' found for model with ID '{}'.",
            model.model_id
        )
    );
}

#[tokio::test]
async fn log_logged_model_params_appends() {
    let tmp = TempDb::new("params").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[kv("a", "1")], None)
        .await
        .unwrap();

    s.log_logged_model_params(WS, &model.model_id, &[kv("b", "2")])
        .await
        .unwrap();

    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(got.params.len(), 2);
    assert!(got.params.iter().any(|p| p.key == "a" && p.value == "1"));
    assert!(got.params.iter().any(|p| p.key == "b" && p.value == "2"));
}

#[tokio::test]
async fn log_logged_model_params_missing_model() {
    let tmp = TempDb::new("params_missing").await;
    let s = store(&tmp).await;
    let err = s
        .log_logged_model_params(WS, "m-nope", &[kv("a", "1")])
        .await
        .unwrap_err();
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn invalid_param_and_tag_are_rejected() {
    let tmp = TempDb::new("invalid_kv").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    let long_value = "x".repeat(6001);
    let err = s
        .log_logged_model_params(WS, &model.model_id, &[kv("k", &long_value)])
        .await
        .unwrap_err();
    assert!(err.message.contains("exceeds the maximum length"));

    let long_tag_value = "x".repeat(8001);
    let err = s
        .set_logged_model_tags(WS, &model.model_id, &[kv("k", &long_tag_value)])
        .await
        .unwrap_err();
    assert!(err.message.contains("exceeds the maximum length"));
}

// ---------------------------------------------------------------------------
// search_logged_models: filters
// ---------------------------------------------------------------------------

async fn search_ids(
    s: &TrackingStore,
    workspace: &str,
    exp_ids: &[String],
    filter: Option<&str>,
) -> Vec<String> {
    let page = s
        .search_logged_models(workspace, exp_ids, filter, &[], None, &[], None)
        .await
        .unwrap();
    page.models.into_iter().map(|m| m.model_id).collect()
}

#[tokio::test]
async fn search_by_attribute_filter() {
    let tmp = TempDb::new("search_attr").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let m1 = s
        .create_logged_model(WS, &exp, Some("alpha"), None, &[], &[], None)
        .await
        .unwrap();
    let _m2 = s
        .create_logged_model(WS, &exp, Some("beta"), None, &[], &[], None)
        .await
        .unwrap();

    let ids = search_ids(&s, WS, std::slice::from_ref(&exp), Some("name = 'alpha'")).await;
    assert_eq!(ids, vec![m1.model_id]);
}

#[tokio::test]
async fn search_by_param_and_tag_filter() {
    let tmp = TempDb::new("search_param_tag").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let m1 = s
        .create_logged_model(
            WS,
            &exp,
            None,
            None,
            &[kv("team", "ml")],
            &[kv("lr", "0.1")],
            None,
        )
        .await
        .unwrap();
    let _m2 = s
        .create_logged_model(
            WS,
            &exp,
            None,
            None,
            &[kv("team", "infra")],
            &[kv("lr", "0.2")],
            None,
        )
        .await
        .unwrap();

    let ids = search_ids(
        &s,
        WS,
        std::slice::from_ref(&exp),
        Some("params.lr = '0.1'"),
    )
    .await;
    assert_eq!(ids, vec![m1.model_id.clone()]);

    let ids = search_ids(&s, WS, std::slice::from_ref(&exp), Some("tags.team = 'ml'")).await;
    assert_eq!(ids, vec![m1.model_id]);
}

#[tokio::test]
async fn search_by_metric_filter() {
    let tmp = TempDb::new("search_metric").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();

    let m1 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    let m2 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_logged_model_metrics(
        &m1.model_id,
        exp_num,
        &run_id,
        None,
        &[metric("accuracy", 0.9, 100, 1)],
    )
    .await
    .unwrap();
    s.log_logged_model_metrics(
        &m2.model_id,
        exp_num,
        &run_id,
        None,
        &[metric("accuracy", 0.5, 100, 1)],
    )
    .await
    .unwrap();

    let ids = search_ids(
        &s,
        WS,
        std::slice::from_ref(&exp),
        Some("metrics.accuracy > 0.8"),
    )
    .await;
    assert_eq!(ids, vec![m1.model_id]);
}

#[tokio::test]
async fn search_filter_error_messages_match_python() {
    let tmp = TempDb::new("search_errors").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let err = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            Some("bogus.k = 'v'"),
            &[],
            None,
            &[],
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        "Invalid entity type: 'bogus'. Expected one of ['attributes', 'metrics', 'params', \
         'tags']."
    );

    let err = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            Some("status > 'v'"),
            &[],
            None,
            &[],
            None,
        )
        .await
        .unwrap_err();
    assert!(err.message.contains("Invalid comparison operator"));
}

#[tokio::test]
async fn search_datasets_clause_requires_dataset_name() {
    let tmp = TempDb::new("search_dataset_required").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let err = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            None,
            &[DatasetFilter {
                dataset_name: String::new(),
                dataset_digest: None,
            }],
            None,
            &[],
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        "`dataset_name` in the `datasets` clause must be specified."
    );
}

#[tokio::test]
async fn search_datasets_clause_without_metric_filter_requires_any_metric_on_dataset() {
    let tmp = TempDb::new("search_dataset_any_metric").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();

    let m1 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    let m2 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_logged_model_metrics(
        &m1.model_id,
        exp_num,
        &run_id,
        None,
        &[metric_ds("acc", 0.9, 100, 0, "ds1", "d1")],
    )
    .await
    .unwrap();
    s.log_logged_model_metrics(
        &m2.model_id,
        exp_num,
        &run_id,
        None,
        &[metric_ds("acc", 0.9, 100, 0, "ds2", "d2")],
    )
    .await
    .unwrap();

    let page = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            None,
            &[DatasetFilter {
                dataset_name: "ds1".to_string(),
                dataset_digest: None,
            }],
            None,
            &[],
            None,
        )
        .await
        .unwrap();
    let ids: Vec<String> = page.models.into_iter().map(|m| m.model_id).collect();
    assert_eq!(ids, vec![m1.model_id]);
}

// ---------------------------------------------------------------------------
// search_logged_models: dataset-scoped metric ordering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dataset_scoped_metric_ordering() {
    let tmp = TempDb::new("order_dataset_scoped").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();

    let m1 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    let m2 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    // On ds1, m1 has the higher accuracy; on ds2, m2 has the higher accuracy.
    // Ordering restricted to ds1 must rank m1 first; restricted to ds2, m2
    // first — proving the ordering is genuinely dataset-scoped, not global.
    s.log_logged_model_metrics(
        &m1.model_id,
        exp_num,
        &run_id,
        None,
        &[
            metric_ds("acc", 0.9, 100, 0, "ds1", "d1"),
            metric_ds("acc", 0.1, 100, 0, "ds2", "d2"),
        ],
    )
    .await
    .unwrap();
    s.log_logged_model_metrics(
        &m2.model_id,
        exp_num,
        &run_id,
        None,
        &[
            metric_ds("acc", 0.2, 100, 0, "ds1", "d1"),
            metric_ds("acc", 0.8, 100, 0, "ds2", "d2"),
        ],
    )
    .await
    .unwrap();

    let order_by_ds1 = vec![LoggedModelOrderByInput {
        field_name: "metrics.acc".to_string(),
        ascending: false,
        dataset_name: Some("ds1".to_string()),
        dataset_digest: Some("d1".to_string()),
    }];
    let page = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            None,
            &[],
            None,
            &order_by_ds1,
            None,
        )
        .await
        .unwrap();
    let ids: Vec<String> = page.models.iter().map(|m| m.model_id.clone()).collect();
    assert_eq!(ids[0], m1.model_id, "ds1-scoped order should rank m1 first");

    let order_by_ds2 = vec![LoggedModelOrderByInput {
        field_name: "metrics.acc".to_string(),
        ascending: false,
        dataset_name: Some("ds2".to_string()),
        dataset_digest: Some("d2".to_string()),
    }];
    let page = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            None,
            &[],
            None,
            &order_by_ds2,
            None,
        )
        .await
        .unwrap();
    let ids: Vec<String> = page.models.iter().map(|m| m.model_id.clone()).collect();
    assert_eq!(ids[0], m2.model_id, "ds2-scoped order should rank m2 first");
}

#[tokio::test]
async fn order_by_creation_timestamp_default_desc() {
    let tmp = TempDb::new("order_default").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let m1 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    let m2 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    let page = s
        .search_logged_models(WS, std::slice::from_ref(&exp), None, &[], None, &[], None)
        .await
        .unwrap();
    let ids: Vec<String> = page.models.into_iter().map(|m| m.model_id).collect();
    // Default order is creation_timestamp DESC -> most recently created first.
    assert_eq!(ids, vec![m2.model_id, m1.model_id]);
}

// ---------------------------------------------------------------------------
// search_logged_models: pagination
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pagination_full_walk_no_duplicates_no_gaps() {
    let tmp = TempDb::new("pagination").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;

    let mut created = Vec::new();
    for _ in 0..7 {
        let m = s
            .create_logged_model(WS, &exp, None, None, &[], &[], None)
            .await
            .unwrap();
        created.push(m.model_id);
    }

    let mut seen = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let page = s
            .search_logged_models(
                WS,
                std::slice::from_ref(&exp),
                None,
                &[],
                Some(3),
                &[],
                token.as_deref(),
            )
            .await
            .unwrap();
        assert!(page.models.len() <= 3);
        seen.extend(page.models.into_iter().map(|m| m.model_id));
        token = page.next_page_token;
        if token.is_none() {
            break;
        }
    }

    seen.sort();
    let mut expected = created.clone();
    expected.sort();
    assert_eq!(seen, expected);
}

#[tokio::test]
async fn pagination_token_validates_matching_request() {
    let tmp = TempDb::new("pagination_token_validate").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    for _ in 0..3 {
        s.create_logged_model(WS, &exp, None, None, &[], &[], None)
            .await
            .unwrap();
    }

    let page = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            None,
            &[],
            Some(1),
            &[],
            None,
        )
        .await
        .unwrap();
    let token = page.next_page_token.expect("expected a continuation token");

    // Same experiment_ids/filter/order_by: token is accepted.
    let page2 = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            None,
            &[],
            Some(1),
            &[],
            Some(&token),
        )
        .await
        .unwrap();
    assert_eq!(page2.models.len(), 1);

    // Different filter_string: rejected.
    let err = s
        .search_logged_models(
            WS,
            std::slice::from_ref(&exp),
            Some("name = 'x'"),
            &[],
            Some(1),
            &[],
            Some(&token),
        )
        .await
        .unwrap_err();
    assert!(err
        .message
        .contains("Filter string in the page token does not match"));

    // Different experiment_ids: rejected.
    let other_exp = new_experiment(&s).await;
    let err = s
        .search_logged_models(WS, &[other_exp], None, &[], Some(1), &[], Some(&token))
        .await
        .unwrap_err();
    assert!(err
        .message
        .contains("Experiment IDs in the page token do not match"));
}

#[tokio::test]
async fn search_default_max_results_is_100() {
    let tmp = TempDb::new("default_max_results").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    for _ in 0..5 {
        s.create_logged_model(WS, &exp, None, None, &[], &[], None)
            .await
            .unwrap();
    }
    let page = s
        .search_logged_models(WS, std::slice::from_ref(&exp), None, &[], None, &[], None)
        .await
        .unwrap();
    assert_eq!(page.models.len(), 5);
    assert!(page.next_page_token.is_none());
}

// ---------------------------------------------------------------------------
// workspace isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn logged_model_operations_are_workspace_scoped() {
    let tmp = TempDb::new("workspace").await;
    let s = store(&tmp).await;

    let exp_a = new_experiment_ws(&s, WS).await;
    let model_a = s
        .create_logged_model(WS, &exp_a, Some("model-a"), None, &[], &[], None)
        .await
        .unwrap();

    let _exp_b = new_experiment_ws(&s, WS2).await;

    for res in [
        s.get_logged_model(WS2, &model_a.model_id, false)
            .await
            .map(|_| ()),
        s.delete_logged_model(WS2, &model_a.model_id).await,
        s.finalize_logged_model(WS2, &model_a.model_id, LoggedModelStatus::Ready)
            .await
            .map(|_| ()),
        s.log_logged_model_params(WS2, &model_a.model_id, &[kv("k", "v")])
            .await,
        s.set_logged_model_tags(WS2, &model_a.model_id, &[kv("k", "v")])
            .await,
        s.delete_logged_model_tag(WS2, &model_a.model_id, "k").await,
    ] {
        let err = res.unwrap_err();
        assert!(err.message.contains("not found"), "{err:?}");
    }

    // Same-workspace access still works.
    let got = s
        .get_logged_model(WS, &model_a.model_id, false)
        .await
        .unwrap();
    assert_eq!(got.model_id, model_a.model_id);
}

#[tokio::test]
async fn search_logged_models_no_leakage_across_workspaces() {
    let tmp = TempDb::new("workspace_search").await;
    let s = store(&tmp).await;

    let exp_a = new_experiment_ws(&s, WS).await;
    let model_a = s
        .create_logged_model(WS, &exp_a, None, None, &[], &[], None)
        .await
        .unwrap();

    let exp_b = new_experiment_ws(&s, WS2).await;
    let model_b = s
        .create_logged_model(WS2, &exp_b, None, None, &[], &[], None)
        .await
        .unwrap();

    // Searching workspace B for experiment A's id: experiment_id isn't even
    // resolvable there, so no models leak across.
    let page = s
        .search_logged_models(
            WS2,
            std::slice::from_ref(&exp_a),
            None,
            &[],
            None,
            &[],
            None,
        )
        .await
        .unwrap();
    assert!(page.models.is_empty());

    let page = s
        .search_logged_models(WS, std::slice::from_ref(&exp_a), None, &[], None, &[], None)
        .await
        .unwrap();
    assert_eq!(page.models.len(), 1);
    assert_eq!(page.models[0].model_id, model_a.model_id);

    let page = s
        .search_logged_models(
            WS2,
            std::slice::from_ref(&exp_b),
            None,
            &[],
            None,
            &[],
            None,
        )
        .await
        .unwrap();
    assert_eq!(page.models.len(), 1);
    assert_eq!(page.models[0].model_id, model_b.model_id);
}

// ---------------------------------------------------------------------------
// log_logged_model_metrics / `_log_model_metrics` production port
// ---------------------------------------------------------------------------

fn run_metric(key: &str, value: f64, ts: i64, step: i64, model_id: &str) -> MetricInput {
    MetricInput {
        key: key.to_string(),
        value,
        timestamp: ts,
        step,
        model_id: Some(model_id.to_string()),
        dataset_name: None,
        dataset_digest: None,
    }
}

fn model_metrics_of<'a>(
    model: &'a mlflow_store::LoggedModel,
    key: &str,
) -> Vec<&'a mlflow_store::LoggedModelMetric> {
    model.metrics.iter().filter(|m| m.key == key).collect()
}

#[tokio::test]
async fn log_logged_model_metrics_single_and_batch() {
    let tmp = TempDb::new("model_metrics_single_batch").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    // Single write via the standalone entry point.
    s.log_logged_model_metrics(
        &model.model_id,
        exp_num,
        &run_id,
        None,
        &[metric("acc", 0.5, 100, 0)],
    )
    .await
    .unwrap();

    // Batch write of several distinct points.
    s.log_logged_model_metrics(
        &model.model_id,
        exp_num,
        &run_id,
        None,
        &[
            metric("acc", 0.6, 200, 1),
            metric("loss", 0.1, 100, 0),
            metric("loss", 0.05, 200, 1),
        ],
    )
    .await
    .unwrap();

    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(model_metrics_of(&got, "acc").len(), 2);
    assert_eq!(model_metrics_of(&got, "loss").len(), 2);
}

#[tokio::test]
async fn log_logged_model_metrics_duplicates_within_batch_are_deduped() {
    let tmp = TempDb::new("model_metrics_batch_dedup").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    // The exact same (key, value, timestamp, step, dataset) point three times
    // in one call: Python's per-call `seen` set (keyed on the full `Metric`
    // tuple) collapses these to one write before ever touching the DB.
    s.log_logged_model_metrics(
        &model.model_id,
        exp_num,
        &run_id,
        None,
        &[
            metric("acc", 0.5, 100, 0),
            metric("acc", 0.5, 100, 0),
            metric("acc", 0.5, 100, 0),
        ],
    )
    .await
    .unwrap();

    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(model_metrics_of(&got, "acc").len(), 1);
}

#[tokio::test]
async fn log_logged_model_metrics_conflict_with_existing_row_is_idempotent() {
    let tmp = TempDb::new("model_metrics_conflict_existing").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_logged_model_metrics(
        &model.model_id,
        exp_num,
        &run_id,
        None,
        &[metric("acc", 0.5, 100, 0)],
    )
    .await
    .unwrap();

    // Re-logging the identical PK (model_id, key, timestamp, step, run_id) in
    // a second, separate call is a silent no-op — matches Python's
    // IntegrityError-rollback-and-filter-existing outcome.
    s.log_logged_model_metrics(
        &model.model_id,
        exp_num,
        &run_id,
        None,
        &[metric("acc", 0.5, 100, 0)],
    )
    .await
    .unwrap();

    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(model_metrics_of(&got, "acc").len(), 1);
}

#[tokio::test]
async fn log_logged_model_metrics_nan_and_inf_are_sanitized() {
    let tmp = TempDb::new("model_metrics_nan_inf").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_logged_model_metrics(
        &model.model_id,
        exp_num,
        &run_id,
        None,
        &[
            metric("nan_metric", f64::NAN, 0, 0),
            metric("posinf_metric", f64::INFINITY, 0, 0),
            metric("neginf_metric", f64::NEG_INFINITY, 0, 0),
        ],
    )
    .await
    .unwrap();

    let got = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    // NaN is permanently lost to 0.0 (no is_nan column on this table).
    assert_eq!(model_metrics_of(&got, "nan_metric")[0].value, Some(0.0));
    assert_eq!(
        model_metrics_of(&got, "posinf_metric")[0].value,
        Some(f64::MAX)
    );
    assert_eq!(
        model_metrics_of(&got, "neginf_metric")[0].value,
        Some(-f64::MAX)
    );
}

#[tokio::test]
async fn log_logged_model_metrics_missing_model_errors() {
    let tmp = TempDb::new("model_metrics_missing_model").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let exp_num: i64 = exp.parse().unwrap();

    let err = s
        .log_logged_model_metrics(
            "m-doesnotexist",
            exp_num,
            &run_id,
            None,
            &[metric("acc", 0.5, 100, 0)],
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        "Logged model with ID 'm-doesnotexist' not found."
    );
}

#[tokio::test]
async fn log_logged_model_metrics_workspace_scoped_model_lookup() {
    let tmp = TempDb::new("model_metrics_workspace").await;
    let s = store(&tmp).await;
    let exp_a = new_experiment_ws(&s, WS).await;
    let run_a = new_run_in_ws(&s, WS, &exp_a).await;
    let exp_num: i64 = exp_a.parse().unwrap();
    let model_a = s
        .create_logged_model(WS, &exp_a, None, None, &[], &[], None)
        .await
        .unwrap();

    // A model created in WS must not be reachable from WS2's viewpoint, even
    // when the caller supplies the correct model_id/experiment_id/run_id.
    let err = s
        .log_logged_model_metrics(
            &model_a.model_id,
            exp_num,
            &run_a,
            None,
            &[metric("acc", 0.5, 100, 0)],
        )
        .await;
    // Same workspace: succeeds.
    assert!(err.is_ok());

    // Different run in a different workspace referencing model_a's id: since
    // `log_logged_model_metrics`'s workspace is derived from `experiment_id`
    // (the model's own experiment), simulate the cross-workspace case via
    // `log_batch`, which takes an explicit `workspace` argument independent of
    // the metric's model_id.
    let exp_b = new_experiment_ws(&s, WS2).await;
    let run_b = new_run_in_ws(&s, WS2, &exp_b).await;
    let err = s
        .log_batch(
            WS2,
            &run_b,
            &[run_metric("acc", 0.5, 100, 0, &model_a.model_id)],
            &[],
            &[],
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        format!("Logged model with ID '{}' not found.", model_a.model_id)
    );
}

#[tokio::test]
async fn log_batch_with_mixed_run_and_model_metrics() {
    let tmp = TempDb::new("log_batch_mixed").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    // A run-only metric (no model_id) and a model-carrying metric in the same
    // log_batch call. Python routes ALL metrics through `_log_metrics` (run
    // tables) and additionally routes the model_id subset through
    // `_log_model_metrics` — a model metric lands in BOTH places, not one or
    // the other.
    s.log_batch(
        WS,
        &run_id,
        &[
            MetricInput {
                key: "run_only".to_string(),
                value: 1.0,
                timestamp: 10,
                step: 0,
                model_id: None,
                dataset_name: None,
                dataset_digest: None,
            },
            run_metric("model_metric", 2.0, 10, 0, &model.model_id),
        ],
        &[],
        &[],
    )
    .await
    .unwrap();

    let run = s.get_run(WS, &run_id).await.unwrap();
    let run_metrics: Vec<&str> = run.data.metrics.iter().map(|m| m.key.as_str()).collect();
    assert!(run_metrics.contains(&"run_only"));
    // The model metric is ALSO present in the run's latest_metrics.
    assert!(run_metrics.contains(&"model_metric"));

    let got_model = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(model_metrics_of(&got_model, "model_metric").len(), 1);
    // The run-only metric never reaches logged_model_metrics.
    assert!(model_metrics_of(&got_model, "run_only").is_empty());
}

#[tokio::test]
async fn log_batch_with_metrics_for_two_different_models_in_one_call() {
    // Two metrics with the SAME key/value/timestamp/step but different
    // model_id must both be written — model_id is part of the dedup identity
    // (mirrors Python's `Metric.__eq__`, which includes model_id), so this
    // must not collapse to a single row.
    let tmp = TempDb::new("log_batch_two_models_same_point").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let m1 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    let m2 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_batch(
        WS,
        &run_id,
        &[
            run_metric("acc", 0.9, 100, 0, &m1.model_id),
            run_metric("acc", 0.9, 100, 0, &m2.model_id),
        ],
        &[],
        &[],
    )
    .await
    .unwrap();

    let got_m1 = s.get_logged_model(WS, &m1.model_id, false).await.unwrap();
    let got_m2 = s.get_logged_model(WS, &m2.model_id, false).await.unwrap();
    assert_eq!(model_metrics_of(&got_m1, "acc").len(), 1);
    assert_eq!(model_metrics_of(&got_m2, "acc").len(), 1);
}

#[tokio::test]
async fn log_batch_model_metrics_dataset_scoped_visible_in_search_ordering() {
    let tmp = TempDb::new("log_batch_model_metrics_search").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let m1 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();
    let m2 = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_batch(
        WS,
        &run_id,
        &[run_metric("accuracy", 0.9, 100, 1, &m1.model_id)],
        &[],
        &[],
    )
    .await
    .unwrap();
    s.log_batch(
        WS,
        &run_id,
        &[run_metric("accuracy", 0.5, 100, 1, &m2.model_id)],
        &[],
        &[],
    )
    .await
    .unwrap();

    let ids = search_ids(
        &s,
        WS,
        std::slice::from_ref(&exp),
        Some("metrics.accuracy > 0.8"),
    )
    .await;
    assert_eq!(ids, vec![m1.model_id]);
}

#[tokio::test]
async fn log_metric_single_with_model_id_writes_both_tables() {
    let tmp = TempDb::new("log_metric_model_id").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;
    let model = s
        .create_logged_model(WS, &exp, None, None, &[], &[], None)
        .await
        .unwrap();

    s.log_metric(
        WS,
        &run_id,
        &run_metric("acc", 0.75, 1000, 0, &model.model_id),
    )
    .await
    .unwrap();

    let run = s.get_run(WS, &run_id).await.unwrap();
    assert!(run
        .data
        .metrics
        .iter()
        .any(|m| m.key == "acc" && m.value == 0.75));

    let got_model = s
        .get_logged_model(WS, &model.model_id, false)
        .await
        .unwrap();
    assert_eq!(model_metrics_of(&got_model, "acc").len(), 1);
    assert_eq!(model_metrics_of(&got_model, "acc")[0].value, Some(0.75));
}

#[tokio::test]
async fn log_metric_missing_model_id_errors_and_writes_nothing() {
    let tmp = TempDb::new("log_metric_missing_model").await;
    let s = store(&tmp).await;
    let exp = new_experiment(&s).await;
    let run_id = new_run_in(&s, &exp).await;

    let err = s
        .log_metric(
            WS,
            &run_id,
            &run_metric("acc", 0.75, 1000, 0, "m-doesnotexist"),
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.message,
        "Logged model with ID 'm-doesnotexist' not found."
    );

    // The whole call is one transaction: the run metric must not have been
    // written either, since the model-metrics leg failed first.
    let run = s.get_run(WS, &run_id).await.unwrap();
    assert!(!run.data.metrics.iter().any(|m| m.key == "acc"));
}
