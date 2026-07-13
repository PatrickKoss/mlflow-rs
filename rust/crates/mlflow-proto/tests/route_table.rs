//! Snapshot / spot-check tests for the build-time-generated route table (T1.2).
//!
//! Expected values were verified against `mlflow/protos/*.proto` and Python's
//! `mlflow.server.handlers.get_endpoints()` (see `rust/tools/route_parity.py`).

use mlflow_proto::{ExpandedRoute, RouteSpec, ROUTE_TABLE};

/// Every expanded route (both `/api/` and `/ajax-api/` forms), with no static
/// prefix — the shape `route_parity.py` compares against Python.
fn expanded() -> Vec<ExpandedRoute> {
    ROUTE_TABLE.iter().flat_map(|r| r.expand("")).collect()
}

fn has_route(http_method: &str, path: &str) -> bool {
    expanded()
        .iter()
        .any(|r| r.http_method == http_method && r.path == path)
}

#[test]
fn table_is_non_empty() {
    assert!(!ROUTE_TABLE.is_empty());
    // 186 raw endpoints at time of writing; guard against a regression that
    // silently empties the table without being exact/brittle.
    assert!(ROUTE_TABLE.len() > 150);
}

#[test]
fn covers_key_services() {
    let services: std::collections::HashSet<_> = ROUTE_TABLE.iter().map(|r| r.service).collect();
    for expected in [
        "MlflowService",
        "ModelRegistryService",
        "WebhookService",
        "MlflowArtifactsService",
    ] {
        assert!(services.contains(expected), "missing service {expected}");
    }
}

#[test]
fn every_raw_route_expands_to_two_prefixed_paths() {
    for spec in ROUTE_TABLE {
        let e = spec.expand("");
        assert_eq!(e.len(), 2, "route {spec:?} did not expand to 2 paths");
        assert!(e.iter().any(|r| r.path.starts_with("/api/")));
        assert!(e.iter().any(|r| r.path.starts_with("/ajax-api/")));
    }
}

#[test]
fn spot_check_experiments_create() {
    // service.proto: createExperiment, POST /mlflow/experiments/create, since 2.0
    assert!(has_route("POST", "/api/2.0/mlflow/experiments/create"));
    assert!(has_route("POST", "/ajax-api/2.0/mlflow/experiments/create"));
}

#[test]
fn spot_check_registered_model_alias_is_method_overloaded() {
    // model_registry.proto: the alias path is overloaded POST/DELETE/GET, since 2.0
    for method in ["POST", "DELETE", "GET"] {
        assert!(
            has_route(method, "/api/2.0/mlflow/registered-models/alias"),
            "missing {method} /api/2.0/mlflow/registered-models/alias"
        );
        assert!(
            has_route(method, "/ajax-api/2.0/mlflow/registered-models/alias"),
            "missing {method} /ajax-api/2.0/mlflow/registered-models/alias"
        );
    }
}

#[test]
fn spot_check_traces_v3_uses_v3_paths() {
    // service.proto: startTraceV3, POST /mlflow/traces, since 3.0
    assert!(has_route("POST", "/api/3.0/mlflow/traces"));
    assert!(has_route("POST", "/ajax-api/3.0/mlflow/traces"));
    // ...and the V2 start/search traces keep the 2.0 paths.
    assert!(has_route("POST", "/api/2.0/mlflow/traces"));
    assert!(has_route("GET", "/api/2.0/mlflow/traces"));
}

#[test]
fn spot_check_search_datasets_missing_leading_slash_quirk() {
    // service.proto path is "mlflow/experiments/search-datasets" (no leading
    // slash). Python concatenates f"/api/2.0{path}", producing a path with no
    // slash between the version and "mlflow". We must reproduce that verbatim.
    assert!(has_route(
        "POST",
        "/api/2.0mlflow/experiments/search-datasets"
    ));
    assert!(has_route(
        "POST",
        "/ajax-api/2.0mlflow/experiments/search-datasets"
    ));
}

#[test]
fn spot_check_webhook_and_artifacts_endpoints() {
    // webhooks.proto: createWebhook POST /mlflow/webhooks, since 2.0
    assert!(has_route("POST", "/api/2.0/mlflow/webhooks"));
    // mlflow_artifacts.proto: downloadArtifact GET /mlflow-artifacts/artifacts/<path>
    assert!(has_route(
        "GET",
        "/api/2.0/mlflow-artifacts/artifacts/<path:artifact_path>"
    ));
}

#[test]
fn static_prefix_is_prepended() {
    let spec = RouteSpec {
        service: "MlflowService",
        method: "createExperiment",
        http_method: "POST",
        path: "/mlflow/experiments/create",
        since_major: 2,
        since_minor: 0,
    };
    let e = spec.expand("/prefix/");
    assert!(e
        .iter()
        .any(|r| r.path == "/prefix/api/2.0/mlflow/experiments/create"));
}

#[test]
fn flask_path_parameters_are_converted() {
    // Trace-tag paths carry a {request_id}/{key} style parameter that Python
    // rewrites to Flask <...> syntax.
    let converted: Vec<_> = expanded()
        .into_iter()
        .filter(|r| r.path.contains("/mlflow/traces/") && r.path.contains('<'))
        .collect();
    assert!(
        !converted.is_empty(),
        "expected at least one parameterized trace route in Flask <...> form"
    );
    // No raw brace syntax should survive expansion.
    assert!(expanded().iter().all(|r| !r.path.contains('{')));
}
