//! `mlflow-server`: the Rust MLflow tracking server library.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§2, Phase 1 T1.5), this crate wires
//! together `mlflow-proto`, `mlflow-store`, `mlflow-registry`,
//! `mlflow-auth`, `mlflow-search`, `mlflow-artifacts`, and `mlflow-webhooks`
//! into an axum HTTP application serving the tracking, tracing, model
//! registry, webhooks, auth/RBAC, and workspaces API with byte-compatible
//! JSON wire behavior against the Python MLflow server. It sits behind
//! nginx, which routes everything except genai endpoints here (§2.2).
//!
//! `main.rs` is intentionally thin: all app construction lives here in
//! [`build_app`] so tests (and later tasks) can compose/exercise the
//! `Router` without booting a real listener.

pub mod artifacts;
pub mod assessments;
pub mod assistant;
pub mod assistant_providers;
pub mod assistant_tools;
pub mod auth_api;
pub mod auth_middleware;
pub mod budget;
pub mod config;
pub mod datasets;
pub mod demo;
pub mod experiments;
pub mod gateway;
mod gateway_guardrails;
pub mod gateway_provider_matrix;
pub mod gateway_runtime;
pub mod graphql;
pub mod invoke;
pub mod issues;
pub mod job_runner;
pub mod jobs;
pub mod label_schemas;
pub mod logged_models;
pub mod metric_history;
pub mod metrics;
pub mod native_worker;
pub mod online_scoring_scheduler;
pub mod openai_compatible;
pub mod otlp;
pub mod prompt_optimization;
pub mod promptlab;
pub mod proto_http;
pub mod registry;
pub mod review_queues;
pub mod routes;
pub mod runs;
pub mod schema_validation;
pub mod scorers;
pub mod security;
pub mod server_info;
pub mod state;
pub mod trace_archival;
pub mod trace_archival_config;
pub mod trace_archival_scheduler;
pub mod trace_artifact;
pub mod traces;
pub mod traces_v2;
pub mod webhooks;
pub mod workspace;
pub mod workspaces_api;

use axum::extract::MatchedPath;
use axum::http::Request;
use axum::middleware;
use axum::routing::{get, MethodRouter};
use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::trace::TraceLayer;
use tracing::info_span;

pub use config::{Cli, ServerConfig, StaticPrefixError};
pub use state::AppState;
pub use trace_archival_config::{
    load_trace_archival_server_config, SystemMonotonicClock, TraceArchivalConfigClock,
    TraceArchivalConfigError, TraceArchivalConfigProvider, TraceArchivalServerConfig,
    TRACE_ARCHIVAL_CONFIG_CACHE_TTL,
};

/// Builds the full application `Router` (ops endpoints only — no store).
/// Retained for the ops/skeleton tests that don't need a backend store.
///
/// Request logging and metrics middleware are applied, and everything is nested
/// under `config.static_prefix` when set (mirroring `_add_static_prefix`,
/// `mlflow/server/handlers.py:6731-6734`, which prepends the prefix to every
/// registered route).
pub fn build_app(config: &ServerConfig) -> Router {
    let metrics_handle = metrics::install_recorder();
    build_app_with_recorder(config, metrics_handle, None)
}

/// Builds the full application `Router` with a backend store, registering every
/// proto-backed endpoint implemented so far (Phase 3: experiments + runs) in
/// addition to the ops endpoints. `main` uses this; tests inject a store over a
/// temp DB.
pub fn build_app_with_state(config: &ServerConfig, state: AppState) -> Router {
    let metrics_handle = metrics::install_recorder();
    build_app_with_recorder(config, metrics_handle, Some(state))
}

/// Same as the builders above, but takes an already-installed
/// [`PrometheusHandle`] instead of installing the global recorder. Exists so
/// tests can build multiple `Router`s in the same process without hitting
/// "recorder already installed" panics from `metrics-exporter-prometheus`.
///
/// When `state` is `Some`, the proto-backed endpoints (experiments + runs) are
/// registered on both `/api/2.0/...` and `/ajax-api/2.0/...` (driven by the
/// `mlflow-proto` route table) honoring the static prefix. When `None`, only
/// the ops endpoints are served.
pub fn build_app_with_recorder(
    config: &ServerConfig,
    metrics_handle: PrometheusHandle,
    state: Option<AppState>,
) -> Router {
    let mut api = Router::new()
        .route("/health", get(routes::health))
        .route("/version", get(routes::version));

    // `/metrics` (Prometheus exporter) is gated on `--expose-prometheus`
    // (env `MLFLOW_EXPOSE_PROMETHEUS`), matching Python: the exporter is only
    // installed when `PROMETHEUS_EXPORTER_ENV_VAR` is set
    // (`mlflow/server/__init__.py:90`). When disabled, `/metrics` 404s.
    if config.expose_prometheus {
        api = api.route(
            "/metrics",
            get(move || routes::metrics(metrics_handle.clone())),
        );
    }

    // `/signup` (plan T9.7): registered with `app.add_url_rule(rule=SIGNUP,
    // ...)` in Python — a *raw* rule, unlike every other auth-app route,
    // which go through `_get_rest_path`/`_get_ajax_path` (both of which call
    // `_add_static_prefix`). So `/signup` alone is exempt from the
    // static-prefix nesting below; it is merged onto the outer router after
    // the `nest` rather than added to `api`. Only mounted when auth is
    // enabled, matching every other auth-app route.
    let signup_state = state.clone().filter(AppState::auth_enabled);
    // Kept for the workspace-resolution layer below, which needs `AppState`
    // (its workspace store presence is the enabled/disabled signal). `state`
    // itself is moved into `register_proto_routes`.
    let workspace_state = state.clone();

    if let Some(state) = state {
        api = api.merge(register_proto_routes(state, config.artifacts_only));
    }

    let api = api
        .layer(middleware::from_fn(metrics::track_metrics))
        .layer(
            TraceLayer::new_for_http().make_span_with(|request: &Request<_>| {
                let path = request
                    .extensions()
                    .get::<MatchedPath>()
                    .map(MatchedPath::as_str)
                    .unwrap_or_else(|| request.uri().path());
                info_span!(
                    "http_request",
                    method = %request.method(),
                    path,
                )
            }),
        );

    let app = match &config.static_prefix {
        Some(prefix) => Router::new().nest(prefix, api),
        None => api,
    };

    let app = match signup_state {
        // Registered via `.route()` (not `.merge()` of a fresh router): merging
        // would replace the app's fallback with the new router's default one,
        // dropping the T9.4 auth layer that wraps the fallback and provides the
        // fail-closed 403 on unmatched paths (e.g. unknown `/mlflow/traces/`
        // subpaths). Because this route is added after that layer, the auth
        // middleware is re-applied here directly: Python's `_before_request`
        // covers `/signup` like any other Flask route (`(SIGNUP, "GET"):
        // validate_can_create_user`, `__init__.py:2649`).
        Some(state) => app.route(
            auth_api::signup::SIGNUP_PATH,
            get(auth_api::signup::signup_page)
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    auth_middleware::authorize,
                ))
                .with_state(state),
        ),
        None => app,
    };

    // Workspace-resolution middleware (plan T10.3, §3.17). Applied *after* the
    // auth layer (which is wrapped inside `register_proto_routes`) so it is the
    // *outer* of the two, but *before* the security layer below so it stays
    // inner to security. This mirrors Python's install order: `security`'s
    // `before_request` runs first, then `workspace_before_request_handler`, then
    // the auth app's `_before_request` (`mlflow/server/__init__.py:82-84`). The
    // layer resolves the `X-MLFLOW-WORKSPACE` header (validating against the
    // workspace store when enabled, ignoring it → `default` when disabled),
    // skips server-info, and stamps `ResolvedWorkspace` into request extensions
    // for the `Workspace` extractor and the auth middleware's T10.4 seam. Only
    // layered when a state (backend store) is present — the ops-only app has no
    // workspace-scoped routes.
    let app = match workspace_state {
        Some(state) => app.layer(middleware::from_fn_with_state(
            state,
            workspace::workspace_middleware,
        )),
        None => app,
    };

    // Security middleware (plan T11.2) is applied *last* so it is the outermost
    // tower layer — it runs before every other layer (auth included) and
    // covers every route, including the `/signup` route above and the
    // fail-closed 404/403 fallbacks. This mirrors Python, where
    // `security.init_security_middleware(app)` registers its `before_request`
    // hooks on the base tracking app *before* the auth app's `_before_request`:
    // a disallowed Host is rejected with a 403 before authentication runs (no
    // 401 challenge). Uses `.layer()` on the composed router (not `.merge()`),
    // preserving the T9.4 fallback-wrapping lesson.
    if config.disable_security_middleware {
        app
    } else {
        let security_config = security::SecurityConfig::from_parts(
            config.allowed_hosts.clone(),
            config.cors_allowed_origins.clone(),
            &config.x_frame_options,
        );
        app.layer(middleware::from_fn_with_state(
            security_config,
            security::security_middleware,
        ))
    }
}

/// Build a `Router` of the implemented proto-backed endpoints, registered on
/// both URL prefixes, driving path/method from the `mlflow-proto` route table
/// (not hand-written paths) so later endpoints slot in by extending
/// [`handler_for`]. The static prefix is applied by the app-level `nest`, so we
/// register the bare `/api/…` + `/ajax-api/…` paths here (passing an empty
/// prefix to `expand`). `with_state` erases the state type so the result merges
/// into the ops router.
///
/// Route-table paths use Flask's `<param>` path-parameter syntax (T1.2); axum
/// (matchit) uses `{param}` instead, so [`to_axum_path`] converts before
/// registering.
///
/// A few routes are hand-registered alongside the route-table-driven ones
/// (same pre-static-prefix router, so `_add_static_prefix` nesting still
/// applies): the correctly-slashed `search-datasets` ajax route
/// (`mlflow/server/__init__.py:135` — the route table only produces the
/// leading-slash-missing form, §3.4 quirk) and the ajax-only,
/// non-proto-backed `get-history-bulk` (plan T3.3).
///
/// When `artifacts_only` is set (`--artifacts-only` / `MLFLOW_ARTIFACTS_ONLY`),
/// only the artifact-serving surface is registered: the `MlflowArtifactsService`
/// proxy routes plus the root `/get-artifact` and `/upload-artifact` endpoints
/// (which Python leaves enabled — they are NOT `@_disable_if_artifacts_only`,
/// `handlers.py:1519,2408`). Every tracking RPC and the artifacts-only-disabled
/// endpoints (e.g. `/model-versions/get-artifact`, `handlers.py:3033`) are
/// omitted, matching Python's 503-on-disabled semantics with an outright 404.
fn register_proto_routes(state: AppState, artifacts_only: bool) -> Router {
    use axum::routing::get;

    let mut router: Router<AppState> = Router::new();
    for spec in mlflow_proto::ROUTE_TABLE {
        // In artifacts-only mode, only the artifact proxy service is served.
        if artifacts_only && spec.service != "MlflowArtifactsService" {
            continue;
        }
        let Some(handler) = handler_for(spec.service, spec.method, spec.http_method) else {
            continue;
        };
        for route in spec.expand("") {
            router = router.route(&to_axum_path(&route.path), handler.clone());
        }
    }

    // `/get-artifact` and `/upload-artifact` are the two artifact-plane
    // endpoints Python leaves enabled in artifacts-only mode, so they are always
    // registered. Everything gated behind `!artifacts_only` below is a tracking
    // endpoint (or an artifacts-only-disabled artifact endpoint).
    router = router.route("/get-artifact", get(artifacts::get_artifact));
    // The `upload_artifact` handler enforces the 10 MB cap itself and returns a
    // 400 ("Artifact size is too large. ...") on overflow, mirroring
    // `handlers.py:2424-2439`. axum's `Bytes` extractor otherwise rejects bodies
    // over its 2 MB default with a bare 413 before the handler runs, so raise the
    // per-route body limit just past the cap: the handler still sees a 10 MB + 1
    // body and produces Python's 400, while absurdly larger bodies short-circuit.
    router = router.route(
        "/ajax-api/2.0/mlflow/upload-artifact",
        axum::routing::post(artifacts::upload_artifact).layer(
            axum::extract::DefaultBodyLimit::max(artifacts::MAX_UPLOAD_ARTIFACT_BYTES + 1024),
        ),
    );
    // Legacy deployments bridge (`server/__init__.py:146-148`). Unlike the
    // discovery handlers below, Python does not decorate this route with
    // `_disable_if_artifacts_only`, so keep it on the artifacts-only surface.
    router = router.route(
        "/ajax-api/2.0/mlflow/gateway-proxy",
        get(gateway::gateway_proxy).post(gateway::gateway_proxy),
    );
    if artifacts_only {
        return register_role_and_auth_layers(router, state);
    }

    router = router.route(
        "/ajax-api/2.0/mlflow/experiments/search-datasets",
        axum::routing::post(datasets::search_datasets),
    );
    router = router.route(
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        get(metric_history::get_metric_history_bulk),
    );
    router = router.route(
        "/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        axum::routing::post(promptlab::create_promptlab_run),
    );
    // `get-trace-artifact` (plan T4.5, §3.10) — ajax-only, served under both
    // the 2.0 and 3.0 ajax prefixes (`mlflow/server/__init__.py:159-161`);
    // not proto-route-table-driven (plain `request_id`/`path` query params).
    router = router.route(
        "/ajax-api/2.0/mlflow/get-trace-artifact",
        get(trace_artifact::get_trace_artifact),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/get-trace-artifact",
        get(trace_artifact::get_trace_artifact),
    );
    // OTLP trace ingestion (plan T4.3, §3.8) — not a proto-route-table
    // endpoint (its own wire protocol, `mlflow/server/otel_api.py`), so it is
    // hand-registered here like the routes above. `OTLP_TRACES_PATH` is
    // `/v1/traces` (`mlflow/tracing/utils/otlp.py:20`); the static-prefix
    // nesting in `build_app_with_recorder` still applies to it.
    router = router.route("/v1/traces", axum::routing::post(otlp::export_traces));
    // Artifact plane (plan T5.1/T5.3/T5.4, §3.11). `/get-artifact` is registered
    // above (kept in artifacts-only mode); `/model-versions/get-artifact` is a
    // tracking endpoint disabled in artifacts-only mode (`handlers.py:3033`,
    // served at the root, `__init__.py:117`), so it lives here.
    router = router.route(
        "/model-versions/get-artifact",
        get(artifacts::get_model_version_artifact),
    );
    // logged-model artifact file download (`__init__.py:166`); `/upload-artifact`
    // is registered above (kept in artifacts-only mode).
    router = router.route(
        "/ajax-api/2.0/mlflow/logged-models/{model_id}/artifacts/files",
        get(artifacts::get_logged_model_artifact),
    );
    // GraphQL (plan T6.1/T6.2, §3.12) — served at `/graphql` (root, under the
    // static prefix via the app-level `nest`), for GET **and** POST, exactly as
    // Python registers it (`handlers.py:6795`,
    // `_add_static_prefix("/graphql"), ["GET", "POST"]`). Both methods parse the
    // JSON body (content-type validated for POST only), so a single handler
    // covers both.
    router = router.route("/graphql", get(graphql::graphql).post(graphql::graphql));

    // ---- server-info (T11.5) ----
    // `GET /(api|ajax-api)/3.0/mlflow/server-info` — hand-registered like
    // `/graphql` above, mirroring Python's `_get_paths("/mlflow/server-info",
    // version=3)` (`handlers.py:6797-6802`), `["GET"]` only. See
    // `server_info.rs` for the response-shape rationale.
    router = router.route("/api/3.0/mlflow/server-info", get(server_info::server_info));
    router = router.route(
        "/ajax-api/3.0/mlflow/server-info",
        get(server_info::server_info),
    );
    // ---- end server-info (T11.5) ----

    // ---- AI Gateway discovery (T18.2, §12.8) ----
    // AUTH GAP (D21 posture): Python's basic-auth app authenticates these
    // ajax-only routes globally but gives them no resource-specific validator.
    // Workspace resolution still runs first, while the catalog/config payload
    // itself is intentionally workspace-independent just like Python.
    router = router.route(
        "/ajax-api/3.0/mlflow/gateway/supported-providers",
        get(gateway::supported_providers),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/gateway/supported-models",
        get(gateway::supported_models),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/gateway/provider-config",
        get(gateway::provider_config),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/gateway/secrets/config",
        get(gateway::secrets_config),
    );
    // ---- end AI Gateway discovery ----

    // ---- generic jobs (T16.5, §12.2/§12.13) ----
    // Python serves BOTH families. These `/mlflow/jobs` paths fall through the
    // FastAPI wrapper into Flask (`handlers.py:get_job_endpoints`), where the
    // basic-auth app's global `_before_request` authenticates them even though
    // no resource-specific validator exists. The shorter `/jobs` paths are a
    // separate native FastAPI router (`job_api.py`) explicitly matched by
    // `_find_fastapi_validator`. Preserve both prefixes and their different
    // response shapes.
    // AUTH GAP: both families are authenticated-only; Python applies no
    // per-job authorization validator to generic jobs.
    router = router.route(
        "/ajax-api/3.0/mlflow/jobs/{job_id}",
        get(jobs::flask_get_job),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/jobs/cancel/{job_id}",
        axum::routing::patch(jobs::flask_cancel_job),
    );
    router = router.route("/ajax-api/3.0/jobs/{job_id}", get(jobs::fastapi_get_job));
    router = router.route(
        "/ajax-api/3.0/jobs/cancel/{job_id}",
        axum::routing::patch(jobs::fastapi_cancel_job),
    );
    // ---- end generic jobs ----

    // ---- GenAI invoke submissions (T17.4, §12.2-§12.4) ----
    // AUTH GAP (D21): Python authenticates all three hand-written AJAX routes
    // globally but registers no experiment/trace-specific validator for them.
    router = router.route(
        "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
        axum::routing::post(invoke::invoke_genai_evaluate),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/scorer/invoke",
        axum::routing::post(invoke::invoke_scorer),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/issues/invoke",
        axum::routing::post(invoke::invoke_issue_detection),
    );
    // ---- end GenAI invoke submissions ----

    // ---- Demo data routes (T20.4, adjacent GenAI UI surface) ----
    router = router.route(
        "/ajax-api/3.0/mlflow/demo/generate",
        axum::routing::post(demo::generate),
    );
    router = router.route(
        "/ajax-api/3.0/mlflow/demo/delete",
        axum::routing::post(demo::delete),
    );
    // ---- end demo data routes ----

    // ---- Assistant (T20.1, §12.10) ----
    router = router.merge(assistant::routes());
    // ---- end Assistant ----

    // ---- Gateway runtime (T18.3/T18.4, §12.9) ----
    router = router.route(
        "/gateway/{endpoint_name}/mlflow/invocations",
        axum::routing::post(gateway_runtime::invocations),
    );
    router = router.route(
        "/gateway/mlflow/v1/chat/completions",
        axum::routing::post(gateway_runtime::chat_completions),
    );
    router = router.route(
        "/gateway/openai/v1/chat/completions",
        axum::routing::post(gateway_runtime::openai_passthrough_chat),
    );
    router = router.route(
        "/gateway/openai/v1/embeddings",
        axum::routing::post(gateway_runtime::openai_passthrough_embeddings),
    );
    router = router.route(
        "/gateway/openai/v1/responses",
        axum::routing::post(gateway_runtime::openai_passthrough_responses),
    );
    router = router.route(
        "/gateway/openai/v1/responses/compact",
        axum::routing::post(gateway_runtime::openai_passthrough_responses_compact),
    );
    router = router.route(
        "/gateway/anthropic/v1/messages",
        axum::routing::post(gateway_runtime::anthropic_passthrough_messages),
    );
    router = router.route(
        "/gateway/gemini/v1beta/models/{*model_action}",
        axum::routing::post(gateway_runtime::gemini_passthrough),
    );
    router = router.route(
        "/gateway/proxy/{endpoint_name}/{*path}",
        axum::routing::post(gateway_runtime::raw_proxy),
    );
    // ---- end Gateway runtime ----

    // AUTH GAP: online configs (D21) are authenticated-only in Python; no
    // experiment/scorer-specific validator is applied.
    for prefix in ["/api/3.0", "/ajax-api/3.0"] {
        router = router.route(
            &format!("{prefix}/mlflow/scorers/online-configs"),
            get(scorers::get_online_scoring_configs),
        );
        router = router.route(
            &format!("{prefix}/mlflow/scorers/online-config"),
            axum::routing::put(scorers::upsert_online_scoring_config),
        );
    }

    register_role_and_auth_layers(router, state)
}

/// Apply the auth-user routes (T9.2), role/permission routes (T9.3), and the
/// authorization middleware (T9.4) to `router`, then erase the state type.
/// Shared between the normal path and the `--artifacts-only` early return so the
/// auth surface is wired identically regardless of mode.
fn register_role_and_auth_layers(mut router: Router<AppState>, state: AppState) -> Router {
    // ---- auth API routes (T9.2) ----
    // Hand-rolled JSON endpoints from the `mlflow.server.auth` app
    // (`mlflow/server/auth/__init__.py`), NOT proto ROUTE_TABLE routes. Mounted
    // only when the basic-auth app is enabled (`state.auth_enabled()`), mirroring
    // Python: these endpoints exist solely in the auth app, so a plain tracking
    // server 404s on them. Each is served at both `/api/2.0/...` and
    // `/ajax-api/2.0/...` (`auth/routes.py` `_get_rest_path` + `_get_ajax_path`).
    // T9.3 (roles/permissions) and T9.4 (auth middleware) extend this block
    // additively.
    if state.auth_enabled() {
        router = register_auth_user_routes(router);
    }
    // ---- end auth API routes ----

    // ---- role/permission routes (T9.3) ----
    // `register_role_routes` is self-contained with its own `AuthStore` state
    // (see `auth_api/roles.rs`), so it merges after `with_state` rather than
    // joining the `Router<AppState>` block above.
    let auth_store = state.auth_store().cloned();
    let auth_enabled = state.auth_enabled();
    let layer_state = state.clone();
    let mut app = router.with_state(state);
    if let Some(store) = auth_store {
        app = app.merge(auth_api::register_role_routes(store));
    }
    // ---- auth middleware (T9.4) ----
    // The tower authorization layer (authenticate -> admin bypass -> validator
    // dispatch -> enforcement) wraps the *entire* app so it covers every route:
    // proto ROUTE_TABLE routes, hand-registered routes, and the merged role
    // router. Applied only when the basic-auth app is enabled, mirroring
    // Python's `before_request`/FastAPI middleware being installed solely by
    // `mlflow.server.auth:create_app`.
    if auth_enabled {
        app = app.layer(middleware::from_fn_with_state(
            layer_state,
            auth_middleware::authorize,
        ));
    }
    // ---- end auth middleware ----
    app
}

/// Register the T9.2 user-management routes (`/mlflow/users/*`) on both the
/// `/api/2.0` and `/ajax-api/2.0` prefixes. Split out so the conditional block
/// in [`register_proto_routes`] stays a single call and T9.3 can add a sibling
/// `register_auth_role_routes` without touching this one.
fn register_auth_user_routes(mut router: Router<AppState>) -> Router<AppState> {
    use auth_api::users;
    use axum::routing::{delete, get, patch, post};

    // (tail path, MethodRouter) for each of the 8 endpoints; registered under
    // both prefixes below.
    let routes: [(&str, MethodRouter<AppState>); 8] = [
        ("/mlflow/users/create", post(users::create_user)),
        ("/mlflow/users/create-ui", post(users::create_user_ui)),
        ("/mlflow/users/get", get(users::get_user)),
        ("/mlflow/users/current", get(users::get_current_user)),
        ("/mlflow/users/list", get(users::list_users)),
        (
            "/mlflow/users/update-password",
            patch(users::update_user_password),
        ),
        (
            "/mlflow/users/update-admin",
            patch(users::update_user_admin),
        ),
        ("/mlflow/users/delete", delete(users::delete_user)),
    ];
    for (tail, handler) in routes {
        router = router.route(&format!("/api/2.0{tail}"), handler.clone());
        router = router.route(&format!("/ajax-api/2.0{tail}"), handler);
    }
    router
}

/// Convert a Flask-style path to axum/matchit syntax:
///  * `<param>` → `{param}` (a single path segment);
///  * `<path:param>` → `{*param}` (Flask's `path` converter matches slashes, so
///    it becomes an axum wildcard capture — used by the `MlflowArtifactsService`
///    routes' `<path:artifact_path>`).
///
/// Non-parameterized paths pass through unchanged.
fn to_axum_path(path: &str) -> String {
    // Do the `path:` wildcard rewrite first so the generic `<`/`>` swap below
    // doesn't need to know about the converter prefix.
    path.replace("<path:", "{*")
        .replace('<', "{")
        .replace('>', "}")
}

/// Map a `(service, method, http_method)` route-table entry to its axum
/// handler. Returns `None` for endpoints not yet implemented (they fall through
/// to the 404 `_not_implemented` form). Extend this as later phases land.
fn handler_for(service: &str, method: &str, http_method: &str) -> Option<MethodRouter<AppState>> {
    use axum::routing::{delete, get, patch, post, put};
    // `MlflowArtifactsService` (plan T5.2, §3.11) is a distinct proto service —
    // its 8 endpoints live under `/(api|ajax-api)/2.0/mlflow-artifacts/...` and
    // route to the artifact-proxy handlers (gated by `--serve-artifacts` inside
    // each handler). Their `<path:artifact_path>` segments become axum wildcards
    // via `to_axum_path`.
    if service == "MlflowArtifactsService" {
        return Some(match (method, http_method) {
            ("downloadArtifact", "GET") => get(artifacts::proxy_download),
            ("uploadArtifact", "PUT") => put(artifacts::proxy_upload),
            ("listArtifacts", "GET") => get(artifacts::proxy_list),
            ("deleteArtifact", "DELETE") => delete(artifacts::proxy_delete),
            ("createMultipartUpload", "POST") => post(artifacts::proxy_create_multipart),
            ("completeMultipartUpload", "POST") => post(artifacts::proxy_complete_multipart),
            ("abortMultipartUpload", "POST") => post(artifacts::proxy_abort_multipart),
            ("getPresignedDownloadUrl", "GET") => get(artifacts::proxy_presigned_download),
            _ => return None,
        });
    }
    // `WebhookService` (plan T8.2, §4.16) is a distinct proto service — its 6
    // endpoints live under `/(api|ajax-api)/2.0/mlflow/webhooks[/{webhook_id}]`
    // and route to the webhook CRUD + test handlers. The `{webhook_id}` segment
    // becomes an axum path param via `to_axum_path`, overlaid onto the request
    // proto by `parse_request_with_path_params` (same mechanism as
    // logged-models).
    if service == "WebhookService" {
        return Some(match (method, http_method) {
            ("createWebhook", "POST") => post(webhooks::create_webhook),
            ("listWebhooks", "GET") => get(webhooks::list_webhooks),
            ("getWebhook", "GET") => get(webhooks::get_webhook),
            ("updateWebhook", "PATCH") => patch(webhooks::update_webhook),
            ("deleteWebhook", "DELETE") => delete(webhooks::delete_webhook),
            ("testWebhook", "POST") => post(webhooks::test_webhook),
            _ => return None,
        });
    }
    if service == "MlflowService" {
        let prompt_optimization = match (method, http_method) {
            ("createPromptOptimizationJob", "POST") => {
                Some(post(prompt_optimization::create_prompt_optimization_job))
            }
            ("getPromptOptimizationJob", "GET") => {
                Some(get(prompt_optimization::get_prompt_optimization_job))
            }
            ("searchPromptOptimizationJobs", "POST") => {
                Some(post(prompt_optimization::search_prompt_optimization_jobs))
            }
            ("searchPromptOptimizationJobs", "GET") => {
                Some(get(prompt_optimization::search_prompt_optimization_jobs))
            }
            ("cancelPromptOptimizationJob", "POST") => {
                Some(post(prompt_optimization::cancel_prompt_optimization_job))
            }
            ("deletePromptOptimizationJob", "DELETE") => {
                Some(delete(prompt_optimization::delete_prompt_optimization_job))
            }
            _ => None,
        };
        if prompt_optimization.is_some() {
            return prompt_optimization;
        }
    }
    // `ModelRegistryService` (plan T7.4, §3.14) — the 21 model-registry RPCs
    // under `/(api|ajax-api)/2.0/mlflow/{registered-models,model-versions}/...`.
    // The three alias RPCs share one path (`/mlflow/registered-models/alias`)
    // distinguished only by HTTP method; `register_proto_routes` calls
    // `Router::route` once per (path, method) route-table entry, and axum 0.8
    // merges the resulting `MethodRouter`s for a repeated path as long as the
    // methods are disjoint (POST/DELETE/GET here) — so the method-overloaded
    // alias route (and the POST+GET `get-latest-versions`) fall out naturally
    // from returning a distinct single-method `MethodRouter` per entry.
    if service == "ModelRegistryService" {
        return Some(match (method, http_method) {
            ("createRegisteredModel", "POST") => post(registry::create_registered_model),
            ("renameRegisteredModel", "POST") => post(registry::rename_registered_model),
            ("updateRegisteredModel", "PATCH") => patch(registry::update_registered_model),
            ("deleteRegisteredModel", "DELETE") => delete(registry::delete_registered_model),
            ("getRegisteredModel", "GET") => get(registry::get_registered_model),
            ("searchRegisteredModels", "GET") => get(registry::search_registered_models),
            ("getLatestVersions", "POST") => post(registry::get_latest_versions),
            ("getLatestVersions", "GET") => get(registry::get_latest_versions),
            ("setRegisteredModelTag", "POST") => post(registry::set_registered_model_tag),
            ("deleteRegisteredModelTag", "DELETE") => delete(registry::delete_registered_model_tag),
            ("createModelVersion", "POST") => post(registry::create_model_version),
            ("updateModelVersion", "PATCH") => patch(registry::update_model_version),
            ("transitionModelVersionStage", "POST") => {
                post(registry::transition_model_version_stage)
            }
            ("deleteModelVersion", "DELETE") => delete(registry::delete_model_version),
            ("getModelVersion", "GET") => get(registry::get_model_version),
            ("searchModelVersions", "GET") => get(registry::search_model_versions),
            ("getModelVersionDownloadUri", "GET") => get(registry::get_model_version_download_uri),
            ("setModelVersionTag", "POST") => post(registry::set_model_version_tag),
            ("deleteModelVersionTag", "DELETE") => delete(registry::delete_model_version_tag),
            ("setRegisteredModelAlias", "POST") => post(registry::set_registered_model_alias),
            ("deleteRegisteredModelAlias", "DELETE") => {
                delete(registry::delete_registered_model_alias)
            }
            ("getModelVersionByAlias", "GET") => get(registry::get_model_version_by_alias),
            _ => return None,
        });
    }
    if service != "MlflowService" {
        return None;
    }
    Some(match (method, http_method) {
        ("createGatewaySecret", "POST") => post(gateway::create_secret),
        ("getGatewaySecretInfo", "GET") => get(gateway::get_secret),
        ("updateGatewaySecret", "POST") => post(gateway::update_secret),
        ("deleteGatewaySecret", "DELETE") => delete(gateway::delete_secret),
        ("listGatewaySecretInfos", "GET") => get(gateway::list_secrets),
        ("createGatewayEndpoint", "POST") => post(gateway::create_endpoint),
        ("getGatewayEndpoint", "GET") => get(gateway::get_endpoint),
        ("updateGatewayEndpoint", "POST") => post(gateway::update_endpoint),
        ("deleteGatewayEndpoint", "DELETE") => delete(gateway::delete_endpoint),
        ("listGatewayEndpoints", "GET") => get(gateway::list_endpoints),
        ("createGatewayModelDefinition", "POST") => post(gateway::create_model_definition),
        ("getGatewayModelDefinition", "GET") => get(gateway::get_model_definition),
        ("listGatewayModelDefinitions", "GET") => get(gateway::list_model_definitions),
        ("updateGatewayModelDefinition", "POST") => post(gateway::update_model_definition),
        ("deleteGatewayModelDefinition", "DELETE") => delete(gateway::delete_model_definition),
        ("attachModelToEndpoint", "POST") => post(gateway::attach_model),
        ("detachModelFromEndpoint", "DELETE") => delete(gateway::detach_model),
        ("createEndpointBinding", "POST") => post(gateway::create_binding),
        ("deleteEndpointBinding", "DELETE") => delete(gateway::delete_binding),
        ("listEndpointBindings", "GET") => get(gateway::list_bindings),
        ("setGatewayEndpointTag", "POST") => post(gateway::set_tag),
        ("deleteGatewayEndpointTag", "DELETE") => delete(gateway::delete_tag),
        ("createBudgetPolicy", "POST") => post(gateway::create_budget_policy),
        ("getBudgetPolicy", "GET") => get(gateway::get_budget_policy),
        ("updateBudgetPolicy", "POST") => post(gateway::update_budget_policy),
        ("deleteBudgetPolicy", "DELETE") => delete(gateway::delete_budget_policy),
        ("listBudgetPolicies", "GET") => get(gateway::list_budget_policies),
        ("listBudgetWindows", "GET") => get(gateway::list_budget_windows),
        ("createGatewayGuardrail", "POST") => post(gateway::create_guardrail),
        ("getGatewayGuardrail", "GET") => get(gateway::get_guardrail),
        ("deleteGatewayGuardrail", "DELETE") => delete(gateway::delete_guardrail),
        ("listGatewayGuardrails", "GET") => get(gateway::list_guardrails),
        ("addGuardrailToEndpoint", "POST") => post(gateway::add_guardrail_to_endpoint),
        ("removeGuardrailFromEndpoint", "DELETE") => {
            delete(gateway::remove_guardrail_from_endpoint)
        }
        ("listEndpointGuardrailConfigs", "GET") => get(gateway::list_endpoint_guardrail_configs),
        ("updateEndpointGuardrailConfig", "PATCH") => {
            patch(gateway::update_endpoint_guardrail_config)
        }
        ("listLoggedModelArtifacts", "GET") => get(artifacts::list_logged_model_artifacts),
        ("createExperiment", "POST") => post(experiments::create_experiment),
        ("getExperiment", "GET") => get(experiments::get_experiment),
        ("getExperimentByName", "GET") => get(experiments::get_experiment_by_name),
        ("searchExperiments", "POST") => post(experiments::search_experiments),
        ("searchExperiments", "GET") => get(experiments::search_experiments),
        ("deleteExperiment", "POST") => post(experiments::delete_experiment),
        ("restoreExperiment", "POST") => post(experiments::restore_experiment),
        ("updateExperiment", "POST") => post(experiments::update_experiment),
        ("setExperimentTag", "POST") => post(experiments::set_experiment_tag),
        ("deleteExperimentTag", "POST") => post(experiments::delete_experiment_tag),
        ("searchDatasets", "POST") => post(datasets::search_datasets),
        ("createDataset", "POST") => post(datasets::create_evaluation_dataset),
        ("getDataset", "GET") => get(datasets::get_evaluation_dataset),
        ("deleteDataset", "DELETE") => delete(datasets::delete_evaluation_dataset),
        ("searchEvaluationDatasets", "POST") => post(datasets::search_evaluation_datasets),
        ("searchEvaluationDatasets", "GET") => get(datasets::search_evaluation_datasets),
        ("setDatasetTags", "PATCH") => patch(datasets::set_evaluation_dataset_tags),
        ("deleteDatasetTag", "DELETE") => delete(datasets::delete_evaluation_dataset_tag),
        ("upsertDatasetRecords", "POST") => post(datasets::upsert_evaluation_dataset_records),
        ("getDatasetExperimentIds", "GET") => get(datasets::get_evaluation_dataset_experiment_ids),
        ("getDatasetRecords", "GET") => get(datasets::get_evaluation_dataset_records),
        ("deleteDatasetRecords", "DELETE") => delete(datasets::delete_evaluation_dataset_records),
        ("addDatasetToExperiments", "POST") => {
            post(datasets::add_evaluation_dataset_to_experiments)
        }
        ("removeDatasetFromExperiments", "POST") => {
            post(datasets::remove_evaluation_dataset_from_experiments)
        }
        ("createIssue", "POST") => post(issues::create_issue),
        ("updateIssue", "PATCH") => patch(issues::update_issue),
        ("getIssue", "GET") => get(issues::get_issue),
        ("searchIssues", "POST") => post(issues::search_issues),
        ("createLabelSchema", "POST") => post(label_schemas::create_label_schema),
        ("getLabelSchema", "GET") => get(label_schemas::get_label_schema),
        ("getLabelSchemaByName", "GET") => get(label_schemas::get_label_schema_by_name),
        ("listLabelSchemas", "GET") => get(label_schemas::list_label_schemas),
        ("updateLabelSchema", "PATCH") => patch(label_schemas::update_label_schema),
        ("deleteLabelSchema", "DELETE") => delete(label_schemas::delete_label_schema),
        ("createReviewQueue", "POST") => post(review_queues::create_review_queue),
        ("getOrCreateUserQueue", "POST") => post(review_queues::get_or_create_user_queue),
        ("getReviewQueue", "GET") => get(review_queues::get_review_queue),
        ("getReviewQueueByName", "GET") => get(review_queues::get_review_queue_by_name),
        ("listReviewQueues", "GET") => get(review_queues::list_review_queues),
        ("updateReviewQueue", "POST") => post(review_queues::update_review_queue),
        ("deleteReviewQueue", "POST") => post(review_queues::delete_review_queue),
        ("addItemsToReviewQueue", "POST") => post(review_queues::add_items_to_review_queue),
        ("removeItemsFromReviewQueue", "POST") => {
            post(review_queues::remove_items_from_review_queue)
        }
        ("listReviewQueueItems", "GET") => get(review_queues::list_review_queue_items),
        ("setReviewQueueItemStatus", "POST") => post(review_queues::set_review_queue_item_status),
        ("registerScorer", "POST") => post(scorers::register_scorer),
        ("listScorers", "GET") => get(scorers::list_scorers),
        ("listScorerVersions", "GET") => get(scorers::list_scorer_versions),
        ("getScorer", "GET") => get(scorers::get_scorer),
        ("deleteScorer", "DELETE") => delete(scorers::delete_scorer),
        ("createLoggedModel", "POST") => post(logged_models::create_logged_model),
        ("finalizeLoggedModel", "PATCH") => patch(logged_models::finalize_logged_model),
        ("getLoggedModel", "GET") => get(logged_models::get_logged_model),
        ("deleteLoggedModel", "DELETE") => delete(logged_models::delete_logged_model),
        ("searchLoggedModels", "POST") => post(logged_models::search_logged_models),
        ("setLoggedModelTags", "PATCH") => patch(logged_models::set_logged_model_tags),
        ("deleteLoggedModelTag", "DELETE") => delete(logged_models::delete_logged_model_tag),
        ("LogLoggedModelParams", "POST") => post(logged_models::log_logged_model_params),
        ("createRun", "POST") => post(runs::create_run),
        ("updateRun", "POST") => post(runs::update_run),
        ("deleteRun", "POST") => post(runs::delete_run),
        ("restoreRun", "POST") => post(runs::restore_run),
        ("getRun", "GET") => get(runs::get_run),
        ("searchRuns", "POST") => post(runs::search_runs),
        ("logMetric", "POST") => post(runs::log_metric),
        ("logParam", "POST") => post(runs::log_param),
        ("setTag", "POST") => post(runs::set_tag),
        ("deleteTag", "POST") => post(runs::delete_tag),
        ("logBatch", "POST") => post(runs::log_batch),
        ("logModel", "POST") => post(runs::log_model),
        ("logInputs", "POST") => post(runs::log_inputs),
        ("logOutputs", "POST") => post(runs::log_outputs),
        ("getMetricHistory", "GET") => get(metric_history::get_metric_history),
        ("getMetricHistoryBulkInterval", "GET") => {
            get(metric_history::get_metric_history_bulk_interval)
        }
        ("createAssessment", "POST") => post(assessments::create_assessment),
        ("GetAssessment", "GET") => get(assessments::get_assessment),
        ("updateAssessment", "PATCH") => patch(assessments::update_assessment),
        ("deleteAssessment", "DELETE") => delete(assessments::delete_assessment),
        // Tracing V3 (T4.1, §3.6).
        ("startTraceV3", "POST") => post(traces::start_trace_v3),
        ("getTraceInfoV3", "GET") => get(traces::get_trace_info_v3),
        ("getTrace", "GET") => get(traces::get_trace),
        ("batchGetTraces", "GET") => get(traces::batch_get_traces),
        ("batchGetTraceInfos", "POST") => post(traces::batch_get_trace_infos),
        ("searchTracesV3", "POST") => post(traces::search_traces_v3),
        ("deleteTracesV3", "POST") => post(traces::delete_traces_v3),
        ("setTraceTagV3", "PATCH") => patch(traces::set_trace_tag_v3),
        ("deleteTraceTagV3", "DELETE") => delete(traces::delete_trace_tag_v3),
        ("linkTracesToRun", "POST") => post(traces::link_traces_to_run),
        ("linkPromptsToTrace", "POST") => post(traces::link_prompts_to_trace),
        ("calculateTraceFilterCorrelation", "POST") => {
            post(traces::calculate_trace_filter_correlation)
        }
        ("queryTraceMetrics", "POST") => post(traces::query_trace_metrics),
        // Tracing V2 (T4.2, §3.7) — deprecated adapters, registered only at
        // the `/api/2.0` prefix (`since.major = 2`), so they never collide
        // with their V3 twins above despite sharing tail paths.
        ("startTrace", "POST") => post(traces_v2::start_trace),
        ("endTrace", "PATCH") => patch(traces_v2::end_trace),
        ("getTraceInfo", "GET") => get(traces_v2::get_trace_info),
        ("searchTraces", "GET") => get(traces_v2::search_traces),
        ("deleteTraces", "POST") => post(traces_v2::delete_traces),
        ("setTraceTag", "PATCH") => patch(traces_v2::set_trace_tag),
        ("deleteTraceTag", "DELETE") => delete(traces_v2::delete_trace_tag),
        // ---- workspace routes (T10.2, §3.17) ----
        // The 5 workspace RPCs are part of `MlflowService`, registered at
        // `/api/3.0/mlflow/workspaces[/{workspace_name}]` (since.major = 3).
        // Create returns 201, delete returns 204 (`?mode=` on the query string);
        // when workspaces are disabled each returns a plain-text 503.
        ("listWorkspaces", "GET") => get(workspaces_api::list_workspaces),
        ("createWorkspace", "POST") => post(workspaces_api::create_workspace),
        ("getWorkspace", "GET") => get(workspaces_api::get_workspace),
        ("updateWorkspace", "PATCH") => patch(workspaces_api::update_workspace),
        ("deleteWorkspace", "DELETE") => delete(workspaces_api::delete_workspace),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use metrics_exporter_prometheus::PrometheusBuilder;
    use tower::ServiceExt;

    fn test_config(static_prefix: Option<&str>) -> ServerConfig {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            workers: None,
            static_prefix: static_prefix.map(str::to_string),
            backend_store_uri: None,
            read_replica_backend_store_uri: None,
            registry_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_only: false,
            trace_archival_config: TraceArchivalConfigProvider::default(),
            artifacts_destination: None,
            // These ops/routing unit tests issue raw axum requests without a
            // `Host` header (real HTTP clients always send one). Disable host
            // validation with a `*` allowlist so they exercise routing, not the
            // security layer — the security behavior itself is covered by the
            // `security` unit tests and `tests/security_http.rs`.
            allowed_hosts: Some(vec!["*".to_string()]),
            cors_allowed_origins: None,
            x_frame_options: security::DEFAULT_X_FRAME_OPTIONS.to_string(),
            disable_security_middleware: false,
            // `/metrics` is gated on this; the ops routing tests below include a
            // metrics-endpoint assertion, so enable it here.
            expose_prometheus: true,
            auth_enabled: false,
            job_execution_enabled: true,
            enable_workspaces: false,
            workspace_store_uri: None,
        }
    }

    fn test_app(static_prefix: Option<&str>) -> Router {
        let handle = PrometheusBuilder::new().build_recorder().handle();
        build_app_with_recorder(&test_config(static_prefix), handle, None)
    }

    async fn body_string(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok_without_prefix() {
        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/html; charset=utf-8"
        );
        assert_eq!(body_string(response).await, "OK");
    }

    #[tokio::test]
    async fn version_matches_mlflow_version_py() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(std::path::Path::parent)
            .and_then(std::path::Path::parent)
            .unwrap();
        let version_py = std::fs::read_to_string(repo_root.join("mlflow/version.py")).unwrap();
        let expected = version_py
            .lines()
            .find_map(|line| {
                let rest = line.trim().strip_prefix("VERSION")?.trim_start();
                let rest = rest.strip_prefix('=')?.trim_start();
                let rest = rest.strip_prefix('"')?;
                let end = rest.find('"')?;
                Some(rest[..end].to_string())
            })
            .expect("VERSION line in mlflow/version.py");

        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/version")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_string(response).await, expected);
    }

    #[tokio::test]
    async fn metrics_returns_prometheus_exposition_format() {
        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        // No assertions on specific metric names: the `metrics` crate's
        // recorder is per-Router here (fresh handle per test), so the body
        // may be empty until requests are recorded. We only care that the
        // endpoint responds successfully with the expected content type.
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/plain; version=0.0.4"
        );
    }

    #[tokio::test]
    async fn routes_are_nested_under_static_prefix() {
        let app = test_app(Some("/mlflow"));

        let prefixed = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/mlflow/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(prefixed.status(), StatusCode::OK);

        let unprefixed = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unprefixed.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_path_returns_404() {
        let response = test_app(None)
            .oneshot(
                HttpRequest::builder()
                    .uri("/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_path_returns_404_with_prefix() {
        let response = test_app(Some("/mlflow"))
            .oneshot(
                HttpRequest::builder()
                    .uri("/mlflow/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn to_axum_path_converts_flask_param_syntax() {
        assert_eq!(
            to_axum_path("/mlflow/logged-models/<model_id>"),
            "/mlflow/logged-models/{model_id}"
        );
        assert_eq!(
            to_axum_path("/mlflow/logged-models/<model_id>/tags/<tag_key>"),
            "/mlflow/logged-models/{model_id}/tags/{tag_key}"
        );
        assert_eq!(
            to_axum_path("/mlflow/experiments/create"),
            "/mlflow/experiments/create"
        );
    }
}
