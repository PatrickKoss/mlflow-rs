//! Concurrency stress test for the trace write path (plan T2.10/T2.11 AC):
//! parallel `start_trace` + `log_spans` on the same traces must not deadlock on
//! Postgres and must converge to a consistent trace state.
//!
//! Gated behind `MLFLOW_RUST_TEST_PG_URI` (and `MLFLOW_RUST_TEST_MYSQL_URI`),
//! like the other live-DB tests — skipped with a message when unset. The DB must
//! already be Alembic-migrated to head `a8b9c0d1e2f3` (Rust never migrates).
//!
//! The write-ordering discipline under test (plan §4 item 11, commit
//! `4c5548c39`): both writers emit `trace_request_metadata`/`trace_metrics` in
//! sorted key order and process trace ids in sorted order, and both wrap the
//! operation in the bounded deadlock-retry (2 retries, backoff). With that in
//! place, N concurrent `start_trace`+`log_spans` iterations complete with zero
//! deadlock failures.

#![allow(clippy::cloned_ref_to_slice_refs)]

use mlflow_store::{Db, SpanInput, StartTraceInput, TraceTimeRange, TrackingStore};

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

/// Iterations per DB. Kept moderate so CI stays fast; the AC's 1000-iteration
/// bar is met by raising this via the env when running the soak locally.
fn iterations() -> usize {
    std::env::var("MLFLOW_RUST_STRESS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200)
}

fn start_input(trace_id: &str, exp: &str) -> StartTraceInput {
    StartTraceInput {
        trace_id: trace_id.to_string(),
        experiment_id: exp.to_string(),
        request_time: 1_000,
        execution_duration: Some(500),
        state: "OK".to_string(),
        client_request_id: None,
        request_preview: None,
        response_preview: None,
        // Multiple metadata/metric keys so the sorted-key lock discipline is
        // actually exercised.
        tags: vec![("k1".into(), "v1".into()), ("k2".into(), "v2".into())],
        trace_metadata: vec![
            ("m_a".into(), "1".into()),
            ("m_b".into(), "2".into()),
            ("m_c".into(), "3".into()),
        ],
        trace_metrics: vec![
            ("t_a".into(), 1.0),
            ("t_b".into(), 2.0),
            ("t_c".into(), 3.0),
        ],
        assessments: vec![],
    }
}

fn span(trace_id: &str, span_id: &str) -> SpanInput {
    SpanInput {
        trace_id: trace_id.to_string(),
        span_id: span_id.to_string(),
        parent_span_id: None,
        name: Some("s".into()),
        span_type: Some("LLM".into()),
        status: "OK".into(),
        start_time_unix_nano: 900_000_000,
        end_time_unix_nano: Some(1_800_000_000),
        content: "{\"k\":1}".into(),
        dimension_attributes: None,
    }
}

fn range(trace_id: &str) -> TraceTimeRange {
    TraceTimeRange {
        trace_id: trace_id.to_string(),
        min_start_ms: 900,
        max_end_ms: Some(1_800),
        root_span_status: Some("OK".into()),
    }
}

/// Run the parallel start_trace/log_spans race against a live DB.
async fn run_stress(uri: &str) {
    let db = Db::connect_and_verify(uri).await.expect("connect + verify");
    let store = TrackingStore::new(db, ART_ROOT);
    let ws = WS;
    let suffix = std::process::id();
    let exp = store
        .create_experiment(ws, &format!("stress-{suffix}"), None, &[])
        .await
        .expect("create experiment");

    let n = iterations();
    let mut handles = Vec::with_capacity(n * 2);
    for i in 0..n {
        let trace_id = format!("stress-{suffix}-{i}");
        // Racer A: start_trace.
        let sa = store.clone();
        let ta = trace_id.clone();
        let ea = exp.clone();
        handles.push(tokio::spawn(async move {
            sa.start_trace(ws, &start_input(&ta, &ea)).await.map(|_| ())
        }));
        // Racer B: log_spans on the same trace id (may create the trace first).
        let sb = store.clone();
        let tb = trace_id.clone();
        let eb = exp.clone();
        handles.push(tokio::spawn(async move {
            sb.log_spans(ws, &eb, &[span(&tb, "1")], &[], &[range(&tb)])
                .await
        }));
    }

    for h in handles {
        // Any deadlock that escaped the bounded retry surfaces here.
        h.await
            .expect("task panicked")
            .expect("trace write failed (deadlock?)");
    }

    // Every trace must exist and be finalized (start_trace authoritative), with
    // its spans stored.
    for i in 0..n {
        let trace_id = format!("stress-{suffix}-{i}");
        let ti = store
            .get_trace_info(ws, &trace_id)
            .await
            .expect("trace exists");
        assert_eq!(ti.metadata("mlflow.trace.infoFinalized"), Some("true"));
        let traces = store
            .batch_get_traces(ws, &[trace_id.clone()])
            .await
            .unwrap();
        assert_eq!(traces.len(), 1, "{trace_id}");
        assert_eq!(traces[0].spans.len(), 1, "{trace_id}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn stress_start_trace_and_log_spans_postgres() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_PG_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_PG_URI not set");
        return;
    };
    run_stress(&uri).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn stress_start_trace_and_log_spans_mysql() {
    let Ok(uri) = std::env::var("MLFLOW_RUST_TEST_MYSQL_URI") else {
        eprintln!("skipping: MLFLOW_RUST_TEST_MYSQL_URI not set");
        return;
    };
    run_stress(&uri).await;
}
