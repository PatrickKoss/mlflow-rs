//! Validator dispatch: map an incoming `(path, method)` to the [`Validator`]
//! that authorizes it (plan T9.4, §3.16), mirroring `_find_validator`
//! (`mlflow/server/auth/__init__.py:2865`).
//!
//! Python builds several lookup structures and consults them in a fixed order:
//!
//! 1. `LOGGED_MODEL_BEFORE_REQUEST_VALIDATORS` — regex-matched, consulted first
//!    for any path containing `/mlflow/logged-models`.
//! 2. `WEBHOOK_BEFORE_REQUEST_VALIDATORS` — regex-matched, for `/mlflow/webhooks`.
//! 3. `BEFORE_REQUEST_VALIDATORS` — the exact `(path, method)` map.
//! 4. `TRACE_PARAMETERIZED_BEFORE_REQUEST_VALIDATORS` — regex-matched, for
//!    `/mlflow/traces/` **with fail-closed** on an unknown subpath.
//! 5. Otherwise: proxy-artifact path inspection (`_is_proxy_artifact_path` +
//!    `_get_proxy_artifact_validator`), else no validator (allow).
//!
//! We reproduce the same order and the same matching. The exact-path and
//! regex maps are derived once from the proto `ROUTE_TABLE` keyed on
//! `(service, method)` — exactly as Python keys `BEFORE_REQUEST_HANDLERS` on the
//! proto request class — plus the hand-registered auth/artifact/trace routes.
//! Only routes actually served by this Rust server are wired; unimplemented
//! Python-only surfaces (gateway, review queues, prompt-optimization, workspaces)
//! are intentionally absent (they never route here, so their validators would be
//! dead code).

use std::sync::OnceLock;

use crate::auth_middleware::after_request::{handler_for, AfterRequestHandler};
use crate::auth_middleware::validators::Validator;

/// Whether a template segment is a path parameter placeholder (`<name>`),
/// matching Python's `_re_compile_path` (`<...>` → `([^/]+)`).
fn is_param_segment(seg: &str) -> bool {
    seg.starts_with('<') && seg.ends_with('>') && seg.len() >= 2
}

/// A compiled template path: literal segments plus `<param>` wildcards, matched
/// segment-by-segment. This is the segment-wise equivalent of Python's
/// `_re_compile_path` full-match (`re.sub(r"<([^>]+)>", r"([^/]+)", path)`):
/// each `<param>` matches exactly one non-empty, slash-free segment.
#[derive(Debug, Clone)]
struct TemplateMatcher {
    segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
enum Segment {
    Literal(String),
    Param(String),
}

impl TemplateMatcher {
    fn compile(template: &str) -> Self {
        let segments = template
            .split('/')
            .map(|seg| {
                if is_param_segment(seg) {
                    Segment::Param(seg[1..seg.len() - 1].to_string())
                } else {
                    Segment::Literal(seg.to_string())
                }
            })
            .collect();
        Self { segments }
    }

    /// Full-match `path` against this template (`pat.fullmatch(req.path)`),
    /// returning the captured `<param>` values by name on a match.
    fn matches(&self, path: &str) -> Option<Vec<(String, String)>> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != self.segments.len() {
            return None;
        }
        let mut params = Vec::new();
        for (seg, part) in self.segments.iter().zip(parts.iter()) {
            match seg {
                Segment::Literal(lit) if lit == part => {}
                Segment::Literal(_) => return None,
                // `([^/]+)` — one non-empty segment.
                Segment::Param(_) if part.is_empty() => return None,
                Segment::Param(name) => params.push((name.clone(), (*part).to_string())),
            }
        }
        Some(params)
    }

    fn is_parameterized(&self) -> bool {
        self.segments.iter().any(|s| matches!(s, Segment::Param(_)))
    }
}

/// One `(template, method) -> validator` entry.
struct Route {
    matcher: TemplateMatcher,
    method: &'static str,
    validator: Validator,
}

/// The four dispatch groups, mirroring Python's four maps.
struct Dispatch {
    logged_model: Vec<Route>,
    webhook: Vec<Route>,
    exact: Vec<Route>,
    trace_parameterized: Vec<Route>,
}

fn dispatch() -> &'static Dispatch {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH.get_or_init(build_dispatch)
}

/// `(service, method)` -> the validator for that proto RPC, mirroring
/// `BEFORE_REQUEST_HANDLERS` / `LOGGED_MODEL_BEFORE_REQUEST_HANDLERS` /
/// `WEBHOOK_BEFORE_REQUEST_HANDLERS`. Returns `None` for RPCs this Rust server
/// does not serve or that Python leaves ungated.
fn proto_validator(service: &str, method: &str) -> Option<Validator> {
    use Validator::*;
    let v = match (service, method) {
        // ---- Experiments (BEFORE_REQUEST_HANDLERS) ----
        ("MlflowService", "createExperiment") => CanCreateExperiment,
        ("MlflowService", "getExperiment") => ReadExperiment,
        ("MlflowService", "getExperimentByName") => ReadExperimentByName,
        ("MlflowService", "deleteExperiment") => DeleteExperiment,
        ("MlflowService", "restoreExperiment") => DeleteExperiment,
        ("MlflowService", "updateExperiment") => UpdateExperiment,
        ("MlflowService", "setExperimentTag") => UpdateExperiment,
        ("MlflowService", "deleteExperimentTag") => UpdateExperiment,
        // ---- Runs (inherit experiment) ----
        ("MlflowService", "createRun") => UpdateExperiment,
        ("MlflowService", "getRun") => ReadRun,
        ("MlflowService", "deleteRun") => DeleteRun,
        ("MlflowService", "restoreRun") => DeleteRun,
        ("MlflowService", "updateRun") => UpdateRun,
        ("MlflowService", "logMetric") => UpdateRun,
        ("MlflowService", "logBatch") => UpdateRun,
        ("MlflowService", "logInputs") => UpdateRun,
        ("MlflowService", "logModel") => UpdateRun,
        ("MlflowService", "logOutputs") => UpdateRun,
        ("MlflowService", "setTag") => UpdateRun,
        ("MlflowService", "deleteTag") => UpdateRun,
        ("MlflowService", "logParam") => UpdateRun,
        ("MlflowService", "getMetricHistory") => ReadRun,
        // ---- Model registry (shared with prompts) ----
        ("ModelRegistryService", "createRegisteredModel") => CanCreateRegisteredModel,
        ("ModelRegistryService", "getRegisteredModel") => ReadRegisteredModelOrPrompt,
        ("ModelRegistryService", "deleteRegisteredModel") => DeleteRegisteredModelOrPrompt,
        ("ModelRegistryService", "updateRegisteredModel") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "renameRegisteredModel") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "getLatestVersions") => ReadRegisteredModelOrPrompt,
        ("ModelRegistryService", "createModelVersion") => CreateModelVersion,
        ("ModelRegistryService", "getModelVersion") => ReadRegisteredModelOrPrompt,
        ("ModelRegistryService", "deleteModelVersion") => DeleteRegisteredModelOrPrompt,
        ("ModelRegistryService", "updateModelVersion") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "transitionModelVersionStage") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "getModelVersionDownloadUri") => ReadRegisteredModelOrPrompt,
        ("ModelRegistryService", "setRegisteredModelTag") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "deleteRegisteredModelTag") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "setModelVersionTag") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "deleteModelVersionTag") => DeleteRegisteredModelOrPrompt,
        ("ModelRegistryService", "setRegisteredModelAlias") => UpdateRegisteredModelOrPrompt,
        ("ModelRegistryService", "deleteRegisteredModelAlias") => DeleteRegisteredModelOrPrompt,
        ("ModelRegistryService", "getModelVersionByAlias") => ReadRegisteredModelOrPrompt,
        // ---- Traces (BEFORE_REQUEST_HANDLERS) ----
        ("MlflowService", "startTrace") => UpdateExperiment,
        ("MlflowService", "startTraceV3") => StartTraceV3,
        ("MlflowService", "endTrace") => UpdateTraceByRequestId,
        ("MlflowService", "getTraceInfo") => ReadTraceByRequestId,
        ("MlflowService", "getTraceInfoV3") => ReadTraceByTraceId,
        ("MlflowService", "getTrace") => ReadTraceByTraceId,
        ("MlflowService", "searchTraces") => SearchTraces,
        ("MlflowService", "searchTracesV3") => SearchTracesV3,
        ("MlflowService", "batchGetTraces") => BatchGetTraces,
        ("MlflowService", "batchGetTraceInfos") => BatchGetTraces,
        ("MlflowService", "deleteTraces") => DeleteTraces,
        ("MlflowService", "deleteTracesV3") => DeleteTraces,
        ("MlflowService", "setTraceTag") => UpdateTraceByRequestId,
        ("MlflowService", "setTraceTagV3") => UpdateTraceByTraceId,
        ("MlflowService", "deleteTraceTag") => UpdateTraceByRequestId,
        ("MlflowService", "deleteTraceTagV3") => UpdateTraceByTraceId,
        ("MlflowService", "linkTracesToRun") => LinkTracesToRun,
        ("MlflowService", "linkPromptsToTrace") => UpdateTraceByTraceId,
        ("MlflowService", "calculateTraceFilterCorrelation") => ReadTracesByExperimentIds,
        ("MlflowService", "queryTraceMetrics") => ReadTracesByExperimentIds,
        ("MlflowService", "createAssessment") => UpdateTraceByTraceId,
        ("MlflowService", "GetAssessment") => ReadTraceByTraceId,
        ("MlflowService", "updateAssessment") => UpdateTraceByTraceId,
        ("MlflowService", "deleteAssessment") => UpdateTraceByTraceId,
        // AUTH GAP: datasets (D21) — all evaluation-dataset RPCs intentionally
        // have no before-request validator in Python. Authentication still runs
        // before this dispatch, so leaving them unmatched is exact parity.
        // ---- Logged models (LOGGED_MODEL_BEFORE_REQUEST_HANDLERS) ----
        ("MlflowService", "createLoggedModel") => UpdateExperiment,
        ("MlflowService", "getLoggedModel") => ReadLoggedModel,
        ("MlflowService", "deleteLoggedModel") => DeleteLoggedModel,
        ("MlflowService", "finalizeLoggedModel") => UpdateLoggedModel,
        ("MlflowService", "deleteLoggedModelTag") => DeleteLoggedModel,
        ("MlflowService", "setLoggedModelTags") => UpdateLoggedModel,
        ("MlflowService", "listLoggedModelArtifacts") => ReadLoggedModel,
        ("MlflowService", "LogLoggedModelParams") => UpdateLoggedModel,
        // ---- Workspaces (BEFORE_REQUEST_HANDLERS, T10.4 — `__init__.py:2611`) ----
        // `ListWorkspaces: None` — no before-request gate (any authenticated
        // caller); the after-request `filter_list_workspaces` filters rows. So
        // `listWorkspaces` intentionally returns `None` here (falls through to
        // Allow), while its after-request hook is registered separately.
        ("MlflowService", "createWorkspace") => SenderIsAdmin,
        ("MlflowService", "getWorkspace") => ViewWorkspace,
        ("MlflowService", "updateWorkspace") => SenderIsAdmin,
        ("MlflowService", "deleteWorkspace") => SenderIsAdmin,
        // ---- Webhooks (WEBHOOK_BEFORE_REQUEST_HANDLERS — admin-only) ----
        ("WebhookService", "createWebhook") => SenderIsAdmin,
        ("WebhookService", "getWebhook") => SenderIsAdmin,
        ("WebhookService", "listWebhooks") => SenderIsAdmin,
        ("WebhookService", "updateWebhook") => SenderIsAdmin,
        ("WebhookService", "deleteWebhook") => SenderIsAdmin,
        ("WebhookService", "testWebhook") => SenderIsAdmin,
        _ => return None,
    };
    Some(v)
}

/// Which regex group a route falls into, mirroring Python's `_find_validator`
/// prefix checks (`"/mlflow/logged-models"` / `"/mlflow/webhooks"` /
/// `"/mlflow/traces/"`).
fn classify(path: &str, parameterized: bool) -> Group {
    if path.contains("/mlflow/logged-models") {
        Group::LoggedModel
    } else if path.contains("/mlflow/webhooks") {
        Group::Webhook
    } else if parameterized && path.contains("/mlflow/traces/") {
        Group::TraceParameterized
    } else {
        Group::Exact
    }
}

enum Group {
    LoggedModel,
    Webhook,
    Exact,
    TraceParameterized,
}

fn build_dispatch() -> Dispatch {
    let mut d = Dispatch {
        logged_model: Vec::new(),
        webhook: Vec::new(),
        exact: Vec::new(),
        trace_parameterized: Vec::new(),
    };

    // Proto ROUTE_TABLE routes, keyed on (service, method) like Python.
    for spec in mlflow_proto::ROUTE_TABLE {
        let Some(validator) = proto_validator(spec.service, spec.method) else {
            continue;
        };
        for route in spec.expand("") {
            let matcher = TemplateMatcher::compile(&route.path);
            let parameterized = matcher.is_parameterized();
            let entry = Route {
                matcher,
                method: spec.http_method,
                validator,
            };
            match classify(&route.path, parameterized) {
                Group::LoggedModel => d.logged_model.push(entry),
                Group::Webhook => d.webhook.push(entry),
                Group::TraceParameterized => d.trace_parameterized.push(entry),
                Group::Exact => d.exact.push(entry),
            }
        }
    }

    // Hand-registered auth-app routes not in the proto table (mirroring the
    // `BEFORE_REQUEST_VALIDATORS.update({...})` blocks). Each on both prefixes.
    for (tail, method, validator) in [
        // Auth user routes.
        ("/mlflow/users/create", "POST", Validator::CanCreateUser),
        ("/mlflow/users/get", "GET", Validator::ReadUser),
        ("/mlflow/users/current", "GET", Validator::Allow),
        ("/mlflow/users/list", "GET", Validator::CanListUsers),
        (
            "/mlflow/users/update-password",
            "PATCH",
            Validator::UpdateUserPassword,
        ),
        (
            "/mlflow/users/update-admin",
            "PATCH",
            Validator::AdminOnlyFalse,
        ),
        ("/mlflow/users/delete", "DELETE", Validator::AdminOnlyFalse),
        // Flask (non-proto) artifact + metric-history routes served here.
        ("/mlflow/get-artifact", "GET", Validator::ReadRunArtifact),
        (
            "/mlflow/upload-artifact",
            "POST",
            Validator::UpdateRunArtifact,
        ),
        (
            "/mlflow/model-versions/get-artifact",
            "GET",
            Validator::ReadModelVersionArtifact,
        ),
        (
            "/mlflow/get-trace-artifact",
            "GET",
            Validator::ReadTraceArtifact,
        ),
        (
            "/mlflow/metrics/get-history-bulk",
            "GET",
            Validator::ReadMetricHistoryBulk,
        ),
        (
            "/mlflow/metrics/get-history-bulk-interval",
            "GET",
            Validator::ReadMetricHistoryBulkInterval,
        ),
        (
            "/mlflow/experiments/search-datasets",
            "POST",
            Validator::SearchDatasets,
        ),
    ] {
        for prefix in ["/api/2.0", "/ajax-api/2.0"] {
            d.exact.push(Route {
                matcher: TemplateMatcher::compile(&format!("{prefix}{tail}")),
                method,
                validator,
            });
        }
    }

    // Hand-registered RBAC routes (`BEFORE_REQUEST_VALIDATORS.update`,
    // `auth/__init__.py:2669-2720`). Super admins bypass these upstream; the
    // validators below implement workspace-admin, self, and resource-MANAGE
    // delegation for non-admin callers.
    for (tail, method, validator) in [
        ("/mlflow/roles/create", "POST", Validator::ManageRoles),
        ("/mlflow/roles/get", "GET", Validator::ViewRoles),
        ("/mlflow/roles/list", "GET", Validator::ListRoles),
        ("/mlflow/roles/update", "PATCH", Validator::ManageRoles),
        ("/mlflow/roles/delete", "DELETE", Validator::ManageRoles),
        (
            "/mlflow/roles/permissions/add",
            "POST",
            Validator::ManageRoles,
        ),
        (
            "/mlflow/roles/permissions/remove",
            "DELETE",
            Validator::ManageRoles,
        ),
        (
            "/mlflow/roles/permissions/list",
            "GET",
            Validator::ViewRoles,
        ),
        (
            "/mlflow/roles/permissions/update",
            "PATCH",
            Validator::ManageRoles,
        ),
        ("/mlflow/roles/assign", "POST", Validator::ManageRoles),
        ("/mlflow/roles/unassign", "DELETE", Validator::ManageRoles),
        ("/mlflow/users/roles/list", "GET", Validator::ViewUserRoles),
        ("/mlflow/roles/users/list", "GET", Validator::ManageRoles),
        (
            "/mlflow/users/permissions/grant",
            "POST",
            Validator::ManageResource,
        ),
        (
            "/mlflow/users/permissions/revoke",
            "POST",
            Validator::ManageResource,
        ),
        (
            "/mlflow/users/permissions/get",
            "GET",
            Validator::GetUserPermission,
        ),
    ] {
        for prefix in ["/api/3.0", "/ajax-api/3.0"] {
            d.exact.push(Route {
                matcher: TemplateMatcher::compile(&format!("{prefix}{tail}")),
                method,
                validator,
            });
        }
    }

    // The logged-model AJAX artifact-file download is a plain route with a path
    // parameter, mirroring the extra `LOGGED_MODEL_BEFORE_REQUEST_VALIDATORS`
    // entry Python adds (`__init__.py:2765`). Only the ajax prefix.
    d.logged_model.push(Route {
        matcher: TemplateMatcher::compile(
            "/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files",
        ),
        method: "GET",
        validator: Validator::ReadLoggedModel,
    });

    // `/mlflow/get-trace-artifact` is also served under the v3 ajax prefix
    // (`GET_TRACE_ARTIFACT_V3`).
    d.exact.push(Route {
        matcher: TemplateMatcher::compile("/ajax-api/3.0/mlflow/get-trace-artifact"),
        method: "GET",
        validator: Validator::ReadTraceArtifact,
    });

    // `/signup` (T9.7) — Python's `(SIGNUP, "GET"): validate_can_create_user`
    // (`__init__.py:2649`). Registered at the root (raw `add_url_rule`, outside
    // the api prefixes and the static-prefix nest), but still gated by
    // `_before_request` like every Flask route.
    d.exact.push(Route {
        matcher: TemplateMatcher::compile("/signup"),
        method: "GET",
        validator: Validator::CanCreateUser,
    });

    // OTLP trace ingestion (`/v1/traces`). Python routes this through the
    // FastAPI middleware (`_find_fastapi_validator` prefix `/v1/traces` →
    // `_get_otel_validator`): experiment UPDATE from the `X-Mlflow-Experiment-Id`
    // header. Served at the root here (no api/ajax prefix).
    d.exact.push(Route {
        matcher: TemplateMatcher::compile("/v1/traces"),
        method: "POST",
        validator: Validator::OtlpExperimentUpdate,
    });

    d
}

/// The result of dispatching a request, mirroring `_find_validator`'s return
/// plus the proxy-artifact fallback in `_before_request`.
pub enum Dispatched {
    /// A validator to run, with any captured path parameters (Flask's
    /// `request.view_args`).
    Validator(Validator, Vec<(String, String)>),
    /// No validator matched and the path is not gated — allow (Python returns
    /// `None` and `_before_request` falls through).
    Allow,
    /// Fail-closed deny (unknown `/mlflow/traces/` subpath).
    Deny,
}

/// Mirror `_find_validator` + the proxy-artifact fallback in `_before_request`.
pub fn dispatch_request(path: &str, method: &str) -> Dispatched {
    let d = dispatch();

    // 1. Logged-model routes (checked first, before the exact map).
    if path.contains("/mlflow/logged-models") {
        return match find(&d.logged_model, path, method) {
            Some((v, p)) => Dispatched::Validator(v, p),
            None => Dispatched::Allow,
        };
    }

    // 2. Webhook routes.
    if path.contains("/mlflow/webhooks") {
        return match find(&d.webhook, path, method) {
            Some((v, p)) => Dispatched::Validator(v, p),
            None => Dispatched::Allow,
        };
    }

    // 3. Exact `(path, method)` map.
    if let Some((v, p)) = find(&d.exact, path, method) {
        return Dispatched::Validator(v, p);
    }

    // 4. Trace parameterized routes — fail-closed on an unknown subpath.
    if path.contains("/mlflow/traces/") {
        return match find(&d.trace_parameterized, path, method) {
            Some((v, p)) => Dispatched::Validator(v, p),
            None => Dispatched::Deny,
        };
    }

    // 5. Proxy-artifact path inspection. The artifact tail (Flask's
    //    `<path:artifact_path>` view arg) is captured from the URL so the
    //    experiment-id extractor can read it, mirroring `request.view_args`.
    if is_proxy_artifact_path(path) {
        if let Some(v) = proxy_artifact_validator(path, method) {
            let params = artifact_path_tail(path)
                .map(|tail| vec![("artifact_path".to_string(), tail)])
                .unwrap_or_default();
            return Dispatched::Validator(v, params);
        }
    }

    Dispatched::Allow
}

// ---------------------------------------------------------------------------
// After-request dispatch (T9.5)
// ---------------------------------------------------------------------------

/// One `(template, method) -> after-request handler` entry.
struct AfterRoute {
    matcher: TemplateMatcher,
    method: &'static str,
    handler: AfterRequestHandler,
}

fn after_routes() -> &'static Vec<AfterRoute> {
    static ROUTES: OnceLock<Vec<AfterRoute>> = OnceLock::new();
    ROUTES.get_or_init(build_after_routes)
}

/// Build the after-request route table from the proto `ROUTE_TABLE`, mirroring
/// `AFTER_REQUEST_HANDLERS` (keyed on the proto request class → its HTTP paths).
/// Only RPCs with an after-request hook this server serves are wired.
fn build_after_routes() -> Vec<AfterRoute> {
    let mut routes = Vec::new();
    for spec in mlflow_proto::ROUTE_TABLE {
        let Some(handler) = handler_for(spec.service, spec.method) else {
            continue;
        };
        for route in spec.expand("") {
            routes.push(AfterRoute {
                matcher: TemplateMatcher::compile(&route.path),
                method: spec.http_method,
                handler,
            });
        }
    }
    routes
}

/// Map an incoming `(path, method)` to its after-request handler + captured path
/// params (Flask's `request.view_args`), mirroring
/// `AFTER_REQUEST_HANDLERS.get((request.path, request.method))`. Returns `None`
/// when the route has no after-request hook. The params feed the DeleteWorkspace
/// cleanup, which reads `workspace_name` from the path.
pub fn dispatch_after_request(
    path: &str,
    method: &str,
) -> Option<(AfterRequestHandler, Vec<(String, String)>)> {
    after_routes().iter().find_map(|r| {
        if r.method != method {
            return None;
        }
        r.matcher.matches(path).map(|params| (r.handler, params))
    })
}

/// The `<path:artifact_path>` tail of a proxy artifact URL: everything after the
/// `.../mlflow-artifacts/artifacts/` segment. `None` for the bare list route
/// (no tail) or the mpu routes (which the experiment-id extractor handles via
/// their own leading `<experiment_id>/` segment).
fn artifact_path_tail(path: &str) -> Option<String> {
    for prefix in [
        "/api/2.0/mlflow-artifacts/artifacts/",
        "/ajax-api/2.0/mlflow-artifacts/artifacts/",
        "/api/2.0/mlflow-artifacts/mpu/",
        "/ajax-api/2.0/mlflow-artifacts/mpu/",
    ] {
        if let Some(tail) = path.strip_prefix(prefix) {
            // mpu tails are `<action>/<experiment_id>/...`; drop the leading
            // action segment so the extractor sees `<experiment_id>/...`.
            if prefix.ends_with("/mpu/") {
                return tail.split_once('/').map(|(_action, rest)| rest.to_string());
            }
            return Some(tail.to_string());
        }
    }
    None
}

fn find(routes: &[Route], path: &str, method: &str) -> Option<(Validator, Vec<(String, String)>)> {
    routes.iter().find_map(|r| {
        if r.method != method {
            return None;
        }
        r.matcher.matches(path).map(|params| (r.validator, params))
    })
}

/// `_is_proxy_artifact_path` (`__init__.py:2801`): the artifact-proxy download/
/// upload/list/delete + multipart-upload paths, on both prefixes.
pub fn is_proxy_artifact_path(path: &str) -> bool {
    const PREFIXES: [&str; 4] = [
        "/api/2.0/mlflow-artifacts/artifacts",
        "/ajax-api/2.0/mlflow-artifacts/artifacts",
        "/api/2.0/mlflow-artifacts/mpu/",
        "/ajax-api/2.0/mlflow-artifacts/mpu/",
    ];
    PREFIXES.iter().any(|p| path.starts_with(p))
}

/// `_get_proxy_artifact_validator` (`__init__.py:2813`): the list endpoint (no
/// `artifact_path` view arg) always reads; otherwise map by HTTP method. We
/// approximate Python's `view_args is None` (List) test with "the path has no
/// artifact-path tail after `.../artifacts`" — the bare list route.
fn proxy_artifact_validator(path: &str, method: &str) -> Option<Validator> {
    let is_list = is_artifact_list_path(path);
    if is_list {
        // List: read (Python returns `validate_can_read_experiment_artifact_proxy`
        // regardless of method for the no-view-args case).
        return Some(Validator::ReadExperimentArtifactProxy);
    }
    Some(match method {
        "GET" => Validator::ReadExperimentArtifactProxy,
        "PUT" => Validator::UpdateExperimentArtifactProxy,
        "DELETE" => Validator::DeleteExperimentArtifactProxy,
        "POST" => Validator::UpdateExperimentArtifactProxy,
        _ => return None,
    })
}

/// The bare list route (`.../mlflow-artifacts/artifacts` with no `<artifact_path>`
/// tail): Python's `view_args is None` List case, which reads the experiment id
/// from the `?path=` query instead.
fn is_artifact_list_path(path: &str) -> bool {
    path == "/api/2.0/mlflow-artifacts/artifacts"
        || path == "/ajax-api/2.0/mlflow-artifacts/artifacts"
}

/// `_get_experiment_id_from_view_args` experiment-id extraction
/// (`_EXPERIMENT_ID_PATTERN`, `__init__.py:768`): the artifact tail is
/// `[workspaces/<ws>/]<experiment_id>/...` — an optional `workspaces/<name>/`
/// prefix then a numeric experiment id followed by `/`. Returns the id.
pub fn experiment_id_from_artifact_path(artifact_path: &str) -> Option<String> {
    // Strip an optional leading `workspaces/<name>/`.
    let rest = match artifact_path.strip_prefix("workspaces/") {
        Some(after) => {
            let (_ws, tail) = after.split_once('/')?;
            tail
        }
        None => artifact_path,
    };
    // Then `(\d+)/`.
    let (id, _tail) = rest.split_once('/')?;
    if !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()) {
        Some(id.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_id_extraction_matches_python_pattern() {
        assert_eq!(
            experiment_id_from_artifact_path("123/run-id/artifacts/model"),
            Some("123".to_string())
        );
        // workspaces/<ws>/ prefix.
        assert_eq!(
            experiment_id_from_artifact_path("workspaces/team-a/456/models/m-x/artifacts"),
            Some("456".to_string())
        );
        // Non-numeric id -> no match.
        assert_eq!(experiment_id_from_artifact_path("abc/run/artifacts"), None);
        // No trailing slash -> no match (pattern requires `\d+/`).
        assert_eq!(experiment_id_from_artifact_path("123"), None);
    }

    fn validator_of(d: Dispatched) -> Validator {
        match d {
            Dispatched::Validator(v, _) => v,
            Dispatched::Allow => panic!("expected a validator, got Allow"),
            Dispatched::Deny => panic!("expected a validator, got Deny"),
        }
    }

    #[test]
    fn template_matcher_full_match_and_params() {
        let m = TemplateMatcher::compile("/api/2.0/mlflow/logged-models/<model_id>");
        assert_eq!(
            m.matches("/api/2.0/mlflow/logged-models/m-123"),
            Some(vec![("model_id".to_string(), "m-123".to_string())])
        );
        assert!(m
            .matches("/api/2.0/mlflow/logged-models/m-123/tags")
            .is_none());
        assert!(m.matches("/api/2.0/mlflow/logged-models/").is_none());
        assert!(m.is_parameterized());
    }

    #[test]
    fn unknown_trace_subpath_fails_closed() {
        // A `/mlflow/traces/` path that matches no parameterized trace route.
        match dispatch_request("/api/2.0/mlflow/traces/xyz/bogus", "GET") {
            Dispatched::Deny => {}
            _ => panic!("expected fail-closed deny"),
        }
    }

    #[test]
    fn known_trace_tag_route_dispatches() {
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow/traces/req-1/tags",
                "PATCH"
            )),
            Validator::UpdateTraceByRequestId
        );
    }

    #[test]
    fn webhook_routes_admin_only() {
        assert_eq!(
            validator_of(dispatch_request("/api/2.0/mlflow/webhooks", "POST")),
            Validator::SenderIsAdmin
        );
        assert_eq!(
            validator_of(dispatch_request("/api/2.0/mlflow/webhooks/wh-1", "GET")),
            Validator::SenderIsAdmin
        );
    }

    #[test]
    fn logged_model_inherits_experiment() {
        assert_eq!(
            validator_of(dispatch_request("/api/2.0/mlflow/logged-models/m-1", "GET")),
            Validator::ReadLoggedModel
        );
    }

    #[test]
    fn experiment_crud_levels() {
        assert_eq!(
            validator_of(dispatch_request("/api/2.0/mlflow/experiments/get", "GET")),
            Validator::ReadExperiment
        );
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow/experiments/update",
                "POST"
            )),
            Validator::UpdateExperiment
        );
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow/experiments/delete",
                "POST"
            )),
            Validator::DeleteExperiment
        );
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow/experiments/create",
                "POST"
            )),
            Validator::CanCreateExperiment
        );
    }

    #[test]
    fn runs_inherit_experiment() {
        assert_eq!(
            validator_of(dispatch_request("/api/2.0/mlflow/runs/get", "GET")),
            Validator::ReadRun
        );
        assert_eq!(
            validator_of(dispatch_request("/api/2.0/mlflow/runs/log-metric", "POST")),
            Validator::UpdateRun
        );
    }

    #[test]
    fn model_version_create_dual() {
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow/model-versions/create",
                "POST"
            )),
            Validator::CreateModelVersion
        );
    }

    #[test]
    fn proxy_artifact_paths() {
        assert!(is_proxy_artifact_path(
            "/api/2.0/mlflow-artifacts/artifacts/1/test.txt"
        ));
        assert!(is_proxy_artifact_path(
            "/ajax-api/2.0/mlflow-artifacts/mpu/create/1/run/artifacts/model"
        ));
        assert!(!is_proxy_artifact_path("/api/2.0/mlflow/experiments/get"));
        // PUT upload dispatches to the update validator.
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow-artifacts/artifacts/1/test.txt",
                "PUT"
            )),
            Validator::UpdateExperimentArtifactProxy
        );
        // Bare list route reads via query param.
        assert_eq!(
            validator_of(dispatch_request(
                "/api/2.0/mlflow-artifacts/artifacts",
                "GET"
            )),
            Validator::ReadExperimentArtifactProxy
        );
    }
}
