//! Dumps the expanded Rust route table as JSON, one object per concrete
//! `(http_method, path)` pair (i.e. after `/api/` + `/ajax-api/` expansion).
//!
//! Consumed by `rust/tools/route_parity.py` to diff the Rust routes against
//! Python's `mlflow.server.handlers.get_endpoints()`. Honors `MLFLOW_STATIC_PREFIX`
//! so the dump matches Python's static-prefix behavior when set.
//!
//! Run: `cargo run -p mlflow-proto --example dump_routes`

use mlflow_proto::ROUTE_TABLE;

fn main() {
    let static_prefix = std::env::var("MLFLOW_STATIC_PREFIX").unwrap_or_default();
    let expanded: Vec<_> = ROUTE_TABLE
        .iter()
        .flat_map(|spec| spec.expand(&static_prefix))
        .collect();
    let json = serde_json::to_string_pretty(&expanded).expect("serialize routes");
    println!("{json}");
}
