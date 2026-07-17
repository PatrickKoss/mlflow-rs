//! Behavioral integration tests for the span store (plan T2.11): `log_spans`
//! upsert idempotency, trace time-range updates, `span_metrics`, cleared
//! content, lazy content reads, and `duration_ns` (generated column). Ported
//! from `test_sqlalchemy_store_traces.py` `test_log_spans*`.
//!
//! Each test gets a fresh [`TempDb`] (SQLite fixture copy, or a live
//! Postgres/MySQL database reset to a clean slate — see
//! `mlflow-test-support`), so the same test bodies run across all three
//! dialects (plan T2.2).

#![allow(clippy::too_many_arguments, clippy::cloned_ref_to_slice_refs)]

use mlflow_store::{SpanInput, SpanMetricInput, StartTraceInput, TraceTimeRange, TrackingStore};
use mlflow_test_support::TempDb;

const WS: &str = "default";
const ART_ROOT: &str = "s3://bucket/mlruns";

async fn store(temp: &TempDb) -> TrackingStore {
    TrackingStore::new(temp.connect().await, ART_ROOT)
}

fn span(
    trace_id: &str,
    span_id: &str,
    start_ns: i64,
    end_ns: Option<i64>,
    status: &str,
    content: &str,
) -> SpanInput {
    SpanInput {
        trace_id: trace_id.to_string(),
        span_id: span_id.to_string(),
        parent_span_id: None,
        name: Some("s".to_string()),
        span_type: Some("LLM".to_string()),
        status: status.to_string(),
        start_time_unix_nano: start_ns,
        end_time_unix_nano: end_ns,
        content: content.to_string(),
        dimension_attributes: None,
    }
}

/// A time-range aggregate derived from a single span (ns→ms floor division).
fn range_from(trace_id: &str, start_ns: i64, end_ns: Option<i64>, status: &str) -> TraceTimeRange {
    TraceTimeRange {
        trace_id: trace_id.to_string(),
        min_start_ms: start_ns / 1_000_000,
        max_end_ms: end_ns.map(|e| e / 1_000_000),
        root_span_status: Some(status.to_string()),
    }
}

// ---------------------------------------------------------------------------
// log_spans creates a trace + trace time range
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_spans_creates_trace_and_time_range() {
    let tmp = TempDb::new("create").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    // Span 1s..2s. Trace auto-created with 1000ms start, 1000ms duration.
    let sp = span(
        "tr",
        "1",
        1_000_000_000,
        Some(2_000_000_000),
        "OK",
        "{\"x\":1}",
    );
    s.log_spans(
        WS,
        &exp,
        &[sp],
        &[],
        &[range_from("tr", 1_000_000_000, Some(2_000_000_000), "OK")],
    )
    .await
    .unwrap();

    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!(ti.request_time, 1_000);
    assert_eq!(ti.execution_duration, Some(1_000));
    assert_eq!(ti.state, "OK");
    // SPANS_LOCATION tag set to TRACKING_STORE.
    assert_eq!(ti.tag("mlflow.trace.spansLocation"), Some("TRACKING_STORE"));
}

#[tokio::test]
async fn log_spans_updates_trace_time_range() {
    let tmp = TempDb::new("timerange").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();

    // First span 1s..2s → 1000ms start, 1000ms duration.
    s.log_spans(
        WS,
        &exp,
        &[span(
            "tr",
            "1",
            1_000_000_000,
            Some(2_000_000_000),
            "OK",
            "{}",
        )],
        &[],
        &[range_from("tr", 1_000_000_000, Some(2_000_000_000), "OK")],
    )
    .await
    .unwrap();
    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!(
        (ti.request_time, ti.execution_duration),
        (1_000, Some(1_000))
    );

    // Second span 0.5s..3s → start 500, duration 2500.
    s.log_spans(
        WS,
        &exp,
        &[span(
            "tr",
            "2",
            500_000_000,
            Some(3_000_000_000),
            "OK",
            "{}",
        )],
        &[],
        &[range_from("tr", 500_000_000, Some(3_000_000_000), "OK")],
    )
    .await
    .unwrap();
    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!((ti.request_time, ti.execution_duration), (500, Some(2_500)));

    // Third span 2.5s..4s → only extends end. start 500, duration 3500.
    s.log_spans(
        WS,
        &exp,
        &[span(
            "tr",
            "3",
            2_500_000_000,
            Some(4_000_000_000),
            "OK",
            "{}",
        )],
        &[],
        &[range_from("tr", 2_500_000_000, Some(4_000_000_000), "OK")],
    )
    .await
    .unwrap();
    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!((ti.request_time, ti.execution_duration), (500, Some(3_500)));
}

#[tokio::test]
async fn log_spans_no_end_time_leaves_duration_null() {
    let tmp = TempDb::new("noend").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    s.log_spans(
        WS,
        &exp,
        &[span("tr", "1", 1_000_000_000, None, "OK", "{}")],
        &[],
        &[range_from("tr", 1_000_000_000, None, "OK")],
    )
    .await
    .unwrap();
    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!(ti.request_time, 1_000);
    assert_eq!(ti.execution_duration, None);
}

#[tokio::test]
async fn log_spans_idempotent_upsert() {
    let tmp = TempDb::new("idem").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let range = range_from("tr", 1_000_000_000, Some(2_000_000_000), "OK");
    let sp = span(
        "tr",
        "1",
        1_000_000_000,
        Some(2_000_000_000),
        "OK",
        "{\"v\":1}",
    );
    s.log_spans(WS, &exp, &[sp.clone()], &[], &[range.clone()])
        .await
        .unwrap();
    // Re-log the same span with updated content — upsert overwrites, no dup row.
    let mut sp2 = sp.clone();
    sp2.content = "{\"v\":2}".to_string();
    s.log_spans(WS, &exp, &[sp2], &[], &[range]).await.unwrap();

    let traces = s.batch_get_traces(WS, &["tr".into()]).await.unwrap();
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].spans.len(), 1);
    assert_eq!(traces[0].spans[0].content, "{\"v\":2}");
}

// ---------------------------------------------------------------------------
// span_metrics + duration_ns (generated column)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_spans_writes_span_metrics_and_reads_duration() {
    let tmp = TempDb::new("metrics").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let sp = span("tr", "1", 1_000_000_000, Some(2_000_000_000), "OK", "{}");
    let m = SpanMetricInput {
        trace_id: "tr".into(),
        span_id: "1".into(),
        key: "cost".into(),
        value: 0.5,
    };
    s.log_spans(
        WS,
        &exp,
        &[sp],
        &[m],
        &[range_from("tr", 1_000_000_000, Some(2_000_000_000), "OK")],
    )
    .await
    .unwrap();

    let traces = s.batch_get_traces(WS, &["tr".into()]).await.unwrap();
    let read = &traces[0].spans[0];
    // duration_ns is the generated column (end - start = 1e9).
    assert_eq!(read.duration_ns, Some(1_000_000_000));
}

// ---------------------------------------------------------------------------
// lazy content + cleared payload
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batch_get_trace_infos_does_not_require_spans() {
    // A TraceInfo read returns even when spans exist; and content isn't needed.
    let tmp = TempDb::new("lazy").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    s.log_spans(
        WS,
        &exp,
        &[span(
            "tr",
            "1",
            1_000_000_000,
            Some(2_000_000_000),
            "OK",
            "{\"big\":\"payload\"}",
        )],
        &[],
        &[range_from("tr", 1_000_000_000, Some(2_000_000_000), "OK")],
    )
    .await
    .unwrap();
    let infos = s.batch_get_trace_infos(WS, &["tr".into()]).await.unwrap();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].trace_id, "tr");
}

#[tokio::test]
async fn cleared_content_span_is_skipped_on_read() {
    // content == "" means the payload was cleared (archival); reads skip it.
    let tmp = TempDb::new("cleared").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    // One real span, one with empty content (simulate a cleared payload).
    let good = span(
        "tr",
        "1",
        1_000_000_000,
        Some(2_000_000_000),
        "OK",
        "{\"k\":1}",
    );
    let cleared = span("tr", "2", 1_000_000_000, Some(2_000_000_000), "OK", "");
    s.log_spans(
        WS,
        &exp,
        &[good, cleared],
        &[],
        &[range_from("tr", 1_000_000_000, Some(2_000_000_000), "OK")],
    )
    .await
    .unwrap();

    let traces = s.batch_get_traces(WS, &["tr".into()]).await.unwrap();
    // The cleared-content span (span_id "2") is filtered out on read.
    assert_eq!(traces[0].spans.len(), 1);
    assert_eq!(traces[0].spans[0].span_id, "1");
}

// ---------------------------------------------------------------------------
// finalized trace (start_trace authoritative) is not overwritten by log_spans
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_spans_does_not_overwrite_finalized_trace() {
    let tmp = TempDb::new("finalized").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    // start_trace sets authoritative time + FINALIZED flag.
    let input = StartTraceInput {
        trace_id: "tr".to_string(),
        experiment_id: exp.clone(),
        request_time: 100,
        execution_duration: Some(50),
        state: "OK".to_string(),
        client_request_id: None,
        request_preview: None,
        response_preview: None,
        tags: vec![],
        trace_metadata: vec![],
        trace_metrics: vec![],
    };
    s.start_trace(WS, &input).await.unwrap();

    // log_spans with a wildly different time range must NOT change trace time.
    s.log_spans(
        WS,
        &exp,
        &[span(
            "tr",
            "1",
            999_000_000,
            Some(9_999_000_000),
            "ERROR",
            "{}",
        )],
        &[],
        &[range_from("tr", 999_000_000, Some(9_999_000_000), "ERROR")],
    )
    .await
    .unwrap();

    let ti = s.get_trace_info(WS, "tr").await.unwrap();
    assert_eq!(ti.request_time, 100);
    assert_eq!(ti.execution_duration, Some(50));
    assert_eq!(ti.state, "OK");
}

// ---------------------------------------------------------------------------
// multi-trace batch + empty
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_spans_multiple_traces_in_one_batch() {
    let tmp = TempDb::new("multi").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    let spans = vec![
        span("ta", "1", 1_000_000_000, Some(2_000_000_000), "OK", "{}"),
        span("tb", "1", 3_000_000_000, Some(5_000_000_000), "ERROR", "{}"),
    ];
    let ranges = vec![
        range_from("ta", 1_000_000_000, Some(2_000_000_000), "OK"),
        range_from("tb", 3_000_000_000, Some(5_000_000_000), "ERROR"),
    ];
    s.log_spans(WS, &exp, &spans, &[], &ranges).await.unwrap();

    let ta = s.get_trace_info(WS, "ta").await.unwrap();
    let tb = s.get_trace_info(WS, "tb").await.unwrap();
    assert_eq!(
        (ta.request_time, ta.execution_duration, ta.state.as_str()),
        (1_000, Some(1_000), "OK")
    );
    assert_eq!(
        (tb.request_time, tb.execution_duration, tb.state.as_str()),
        (3_000, Some(2_000), "ERROR")
    );
}

#[tokio::test]
async fn log_spans_empty_is_noop() {
    let tmp = TempDb::new("empty").await;
    let s = store(&tmp).await;
    let exp = s.create_experiment(WS, "e", None, &[]).await.unwrap();
    s.log_spans(WS, &exp, &[], &[], &[]).await.unwrap();
}
