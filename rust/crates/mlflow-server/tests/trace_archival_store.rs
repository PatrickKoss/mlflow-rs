//! T21.2 archive → read → delete integration tests. `TempDb` makes the same
//! test run on SQLite by default and Postgres when the existing dialect test
//! environment variables are configured.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use base64::Engine;
use mlflow_server::trace_archival::{
    archive_traces_at, archive_traces_for_workspace, download_archived_spans,
    download_archived_trace_json, stored_spans_to_traces_pb, TRACE_ARCHIVAL_FILENAME,
};
use mlflow_server::TraceArchivalServerConfig;
use mlflow_store::{
    SpanInput, StartTraceInput, StoredSpan, TraceTimeRange, TrackingStore, Workspace,
    WorkspaceStore, SPANS_LOCATION_ARCHIVE_REPO, SPANS_LOCATION_TRACKING_STORE,
    TRACE_TAG_ARCHIVE_LOCATION, TRACE_TAG_SPANS_LOCATION,
};
use mlflow_test_support::TempDb;

const WORKSPACE: &str = "default";
const TRACE_ID: &str = "tr-00112233445566778899aabbccddeeff";

async fn test_store(tag: &str) -> (TempDb, TrackingStore, String) {
    let db = TempDb::new(tag).await;
    let store = TrackingStore::new(db.connect().await, "file:///tmp/mlruns-unused");
    let experiment_id = store
        .create_experiment(WORKSPACE, tag, None, &[])
        .await
        .unwrap();
    (db, store, experiment_id)
}

async fn seed_trace(store: &TrackingStore, experiment_id: &str, content: &str) {
    store
        .start_trace(
            WORKSPACE,
            &StartTraceInput {
                trace_id: TRACE_ID.to_string(),
                experiment_id: experiment_id.to_string(),
                request_time: 0,
                execution_duration: Some(1),
                state: "OK".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: None,
                tags: Vec::new(),
                trace_metadata: Vec::new(),
                trace_metrics: Vec::new(),
            },
        )
        .await
        .unwrap();
    store
        .log_spans(
            WORKSPACE,
            experiment_id,
            &[SpanInput {
                trace_id: TRACE_ID.to_string(),
                span_id: "1020304050607080".to_string(),
                parent_span_id: None,
                name: Some("root".to_string()),
                span_type: Some("CHAIN".to_string()),
                status: "OK".to_string(),
                start_time_unix_nano: 1,
                end_time_unix_nano: Some(2),
                content: content.to_string(),
                dimension_attributes: None,
            }],
            &[],
            &[TraceTimeRange {
                trace_id: TRACE_ID.to_string(),
                min_start_ms: 0,
                max_end_ms: Some(0),
                root_span_status: Some("OK".to_string()),
            }],
        )
        .await
        .unwrap();
}

fn root_content() -> &'static str {
    r#"{"trace_id":"ABEiM0RVZneImaq7zN3u/w==","span_id":"ECAwQFBgcIA=","parent_span_id":null,"name":"root","start_time_unix_nano":1,"end_time_unix_nano":2,"events":[],"status":{"code":"STATUS_CODE_OK","message":""},"attributes":{"mlflow.traceRequestId":"\"tr-00112233445566778899aabbccddeeff\"","mlflow.spanType":"\"CHAIN\""},"links":[]}"#
}

#[tokio::test]
async fn archive_read_delete_cycle_preserves_payload_and_state() {
    let (_db, store, experiment_id) = test_store("archive-cycle").await;
    seed_trace(&store, &experiment_id, root_content()).await;
    let before = store.get_trace(WORKSPACE, TRACE_ID, true).await.unwrap();
    let expected_payload = stored_spans_to_traces_pb(&before.spans).unwrap();
    let archive_dir = tempfile::tempdir().unwrap();
    let archive_root = format!("file://{}", archive_dir.path().display());

    let archived = archive_traces_at(
        &store,
        WORKSPACE,
        &archive_root,
        "1m",
        &[],
        Some(10),
        60_000,
    )
    .await
    .unwrap();
    assert_eq!(archived, 1);

    let info = store.get_trace_info(WORKSPACE, TRACE_ID).await.unwrap();
    assert_eq!(
        info.tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_ARCHIVE_REPO)
    );
    let archive_uri = info.tag(TRACE_TAG_ARCHIVE_LOCATION).unwrap();
    let payload_path =
        Path::new(archive_uri.strip_prefix("file://").unwrap()).join(TRACE_ARCHIVAL_FILENAME);
    assert_eq!(std::fs::read(&payload_path).unwrap(), expected_payload);
    let archived_spans = download_archived_spans(&info).await.unwrap();
    assert_eq!(archived_spans.len(), 1);
    assert_eq!(archived_spans[0].name, "root");
    assert_eq!(
        download_archived_trace_json(&info).await.unwrap(),
        serde_json::json!({
            "spans": [serde_json::from_str::<serde_json::Value>(root_content()).unwrap()]
        })
    );

    let deleted = store
        .delete_traces(
            WORKSPACE,
            &experiment_id,
            None,
            None,
            Some(&[TRACE_ID.to_string()]),
        )
        .await
        .unwrap();
    assert_eq!(deleted, 1);
    assert!(!payload_path.exists());
    assert!(store.get_trace_info(WORKSPACE, TRACE_ID).await.is_err());
}

#[test]
fn python_backend_archive_cycle_matches_rust_contract() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut command = Command::new("uv");
    command.args(["run", "--frozen"]);
    if std::env::var("MLFLOW_RUST_TEST_DIALECT").ok().as_deref() == Some("postgres") {
        command.args(["--extra", "db"]);
        command.env(
            "MLFLOW_TRACE_ARCHIVAL_PYTHON_BACKEND_URI",
            std::env::var("MLFLOW_RUST_TEST_PG_URI").expect("Postgres test URI"),
        );
    }
    command.args([
        "python",
        "rust/tools/trace_archival_store_differential.py",
        "--content",
        root_content(),
    ]);
    let output = command
        .current_dir(&root)
        .output()
        .expect("run Python archival store differential");
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["archived"], 1);
    assert_eq!(result["spans_location"], SPANS_LOCATION_ARCHIVE_REPO);
    assert_eq!(result["archive_uri_has_suffix"], true);
    let rust_payload = stored_spans_to_traces_pb(&[StoredSpan {
        trace_id: TRACE_ID.to_string(),
        experiment_id: 1,
        span_id: "1020304050607080".to_string(),
        parent_span_id: None,
        name: Some("root".to_string()),
        span_type: Some("CHAIN".to_string()),
        status: "OK".to_string(),
        start_time_unix_nano: 1,
        end_time_unix_nano: Some(2),
        duration_ns: Some(1),
        content: root_content().to_string(),
        dimension_attributes: None,
    }])
    .unwrap();
    assert_eq!(
        result["payload_b64"],
        base64::engine::general_purpose::STANDARD.encode(rust_payload)
    );
    assert_eq!(result["stored_content"], "");
    assert_eq!(result["read_json"]["spans"].as_array().unwrap().len(), 1);
    assert_eq!(result["deleted"], 1);
    assert_eq!(result["payload_exists_after_delete"], false);
}

#[tokio::test]
async fn finalize_generation_guard_rejects_concurrent_span_write() {
    let (_db, store, experiment_id) = test_store("archive-generation").await;
    seed_trace(&store, &experiment_id, root_content()).await;
    let snapshot = store
        .load_trace_archival_data(WORKSPACE, TRACE_ID)
        .await
        .unwrap()
        .unwrap();

    let mut replacement = root_content().to_string();
    replacement.push(' ');
    seed_trace(&store, &experiment_id, &replacement).await;
    assert!(!store
        .finalize_archived_trace(
            WORKSPACE,
            TRACE_ID,
            "file:///tmp/stale-archive",
            snapshot.db_payload_generation,
        )
        .await
        .unwrap());

    let info = store.get_trace_info(WORKSPACE, TRACE_ID).await.unwrap();
    assert_eq!(
        info.tag(TRACE_TAG_SPANS_LOCATION),
        Some(SPANS_LOCATION_TRACKING_STORE)
    );
    assert_eq!(
        store
            .get_trace(WORKSPACE, TRACE_ID, true)
            .await
            .unwrap()
            .spans[0]
            .content,
        replacement
    );
}

#[tokio::test]
async fn missing_archived_payload_is_tolerated_on_delete() {
    let (_db, store, experiment_id) = test_store("archive-missing-delete").await;
    seed_trace(&store, &experiment_id, root_content()).await;
    let archive_dir = tempfile::tempdir().unwrap();
    let archive_root = format!("file://{}", archive_dir.path().display());
    assert_eq!(
        archive_traces_at(&store, WORKSPACE, &archive_root, "1m", &[], None, 60_000,)
            .await
            .unwrap(),
        1
    );
    let info = store.get_trace_info(WORKSPACE, TRACE_ID).await.unwrap();
    let payload = Path::new(
        info.tag(TRACE_TAG_ARCHIVE_LOCATION)
            .unwrap()
            .strip_prefix("file://")
            .unwrap(),
    )
    .join(TRACE_ARCHIVAL_FILENAME);
    std::fs::remove_file(payload).unwrap();
    assert_eq!(
        store
            .delete_traces(
                WORKSPACE,
                &experiment_id,
                None,
                None,
                Some(&[TRACE_ID.to_string()]),
            )
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn retention_allowlist_and_archive_now_resolution_matches_python() {
    let (_db, store, experiment_id) = test_store("archive-retention").await;
    seed_trace(&store, &experiment_id, root_content()).await;
    store
        .set_experiment_tag(
            WORKSPACE,
            &experiment_id,
            "mlflow.trace.archivalRetention",
            r#"{"type":"duration","value":"30d"}"#,
        )
        .await
        .unwrap();
    let now = 2 * 24 * 60 * 60 * 1000;
    let (_, default_candidates) = store
        .plan_trace_archival(WORKSPACE, now, "1d", &HashSet::new(), Some(10))
        .await
        .unwrap();
    assert_eq!(default_candidates.len(), 1);

    let allowlist = HashSet::from([experiment_id.clone()]);
    let (_, allowlisted_candidates) = store
        .plan_trace_archival(WORKSPACE, now, "1d", &allowlist, Some(10))
        .await
        .unwrap();
    assert!(allowlisted_candidates.is_empty());

    store
        .set_experiment_tag(
            WORKSPACE,
            &experiment_id,
            "mlflow.trace.archiveNow",
            r#"{"older_than":"1m"}"#,
        )
        .await
        .unwrap();
    let (requests, urgent_candidates) = store
        .plan_trace_archival(WORKSPACE, now, "1d", &allowlist, Some(10))
        .await
        .unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(urgent_candidates.len(), 1);
}

#[tokio::test]
async fn inherited_workspace_config_scopes_archive_repository_path() {
    let db = TempDb::new("archive-workspace").await;
    let connected = db.connect().await;
    let store = TrackingStore::new(connected.clone(), "file:///tmp/mlruns-unused");
    let workspace_store = WorkspaceStore::new(connected, db.uri());
    let workspace = format!("team-a-{}", std::process::id());
    workspace_store
        .create_workspace(Workspace::named(&workspace))
        .await
        .unwrap();
    let experiment_id = store
        .create_experiment(&workspace, "archive-workspace", None, &[])
        .await
        .unwrap();
    store
        .start_trace(
            &workspace,
            &StartTraceInput {
                trace_id: TRACE_ID.to_string(),
                experiment_id: experiment_id.clone(),
                request_time: 0,
                execution_duration: Some(1),
                state: "OK".to_string(),
                client_request_id: None,
                request_preview: None,
                response_preview: None,
                tags: Vec::new(),
                trace_metadata: Vec::new(),
                trace_metrics: Vec::new(),
            },
        )
        .await
        .unwrap();
    store
        .log_spans(
            &workspace,
            &experiment_id,
            &[SpanInput {
                trace_id: TRACE_ID.to_string(),
                span_id: "1020304050607080".to_string(),
                parent_span_id: None,
                name: Some("root".to_string()),
                span_type: Some("CHAIN".to_string()),
                status: "OK".to_string(),
                start_time_unix_nano: 1,
                end_time_unix_nano: Some(2),
                content: root_content().to_string(),
                dimension_attributes: None,
            }],
            &[],
            &[TraceTimeRange {
                trace_id: TRACE_ID.to_string(),
                min_start_ms: 0,
                max_end_ms: Some(0),
                root_span_status: Some("OK".to_string()),
            }],
        )
        .await
        .unwrap();
    let archive = tempfile::tempdir().unwrap();
    let config = TraceArchivalServerConfig {
        enabled: true,
        location: format!("file://{}", archive.path().display()),
        retention: "1m".to_string(),
        long_retention_allowlist: Vec::new(),
        interval_seconds: 300,
        max_traces_per_pass: Some(10),
    };
    assert_eq!(
        archive_traces_for_workspace(&store, Some(&workspace_store), &workspace, &config, Some(1),)
            .await
            .unwrap(),
        1
    );
    let info = store.get_trace_info(&workspace, TRACE_ID).await.unwrap();
    assert!(info
        .tag(TRACE_TAG_ARCHIVE_LOCATION)
        .unwrap()
        .contains(&format!(
            "/workspaces/{workspace}/{experiment_id}/traces/{TRACE_ID}/artifacts"
        )));
    assert!(store.get_trace_info(WORKSPACE, TRACE_ID).await.is_err());
}
