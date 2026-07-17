//! The after-request hooks (`_after_request`, plan T9.5, §3.16), mirroring
//! `mlflow/server/auth/__init__.py:3651`.
//!
//! Python attaches an `after_request` handler to the auth Flask app that, on a
//! successful (`2xx`/`3xx`) response, either:
//!
//! * **Creator grants**: grant the caller `MANAGE` on the resource they just
//!   created (`set_can_manage_experiment_permission`,
//!   `set_can_manage_registered_model_permission` — the latter classifies the
//!   entity as `registered_model` vs `prompt` from the persisted
//!   `mlflow.prompt.is_prompt` tag in the response).
//! * **Response filtering**: drop rows the caller cannot read from a `search`
//!   response and — for the paged experiment / registered-model / logged-model
//!   searches — re-fetch from the next token to refill the page, rewriting the
//!   token so the client walks the *filtered* stream exactly as Python does
//!   (`filter_search_experiments`, `filter_search_registered_models`,
//!   `filter_search_model_versions`, `filter_search_logged_models`). Admins
//!   skip filtering entirely.
//! * **Grant cascade**: sweep / rewrite the synthetic-role grants when a
//!   registered model is deleted or renamed
//!   (`delete_can_manage_registered_model_permission`,
//!   `rename_registered_model_permission`) — across both the `registered_model`
//!   and `prompt` namespaces, workspace-scoped, since the registry PK is the
//!   name and can collide / change.
//!
//! ## Which hooks are ported (and which are seams)
//!
//! Python's `AFTER_REQUEST_PATH_HANDLERS` also carries scorer, gateway,
//! review-queue, and workspace-listing hooks. Those REST surfaces are **not
//! served by this Rust binary** (see `path_matchers` — the before-request
//! dispatch omits them for the same reason), so their after-request handlers
//! would be dead code and are intentionally absent. The workspace
//! seed/cleanup hooks (`_seed_default_workspace_roles`,
//! `_cleanup_workspace_permissions`) are gated on the multi-tenant workspace
//! feature (T10.4) and its RBAC-seed flag; the workspace routes exist (T10.2)
//! but pre-T10.4 the server is single-tenant, so seeding default roles into a
//! freshly-created workspace and their cleanup are wired here as a documented
//! `// T10.4 SEAM` and left out of the single-tenant matrix.
//!
//! ## Query-integrated filtering (plan Q10)
//!
//! The plan floats pushing the permission predicate into the search SQL (a
//! semi-join on the grants). Parity — identical page-fill behavior — is the
//! ship gate, so we implement the **Python-identical refetch** form and make it
//! the default. The query-integrated form is left as a future optimization
//! behind `MLFLOW_RUST_AUTH_QUERY_INTEGRATED_FILTERING` (unset today); the seam
//! is documented at [`query_integrated_filtering_enabled`] but the flag has no
//! effect yet, so the refetch path always runs.

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use mlflow_auth::permissions::get_permission;
use mlflow_auth::AuthStore;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_registry::RegisteredModelsPage;
use mlflow_search::{create_page_token, parse_start_offset_from_page_token};
use mlflow_store::{DatasetFilter, ExperimentsPage, LoggedModelOrderByInput, LoggedModelsPage};

use crate::experiments::to_proto_experiment;
use crate::logged_models::to_proto_logged_model;
use crate::registry::to_proto_registered_model;
use crate::state::AppState;

/// `mlflow.prompt.is_prompt` (`mlflow/prompt/constants.py:4`).
const IS_PROMPT_TAG_KEY: &str = "mlflow.prompt.is_prompt";

/// One `_after_request` handler, dispatched by the request's `(service, method)`
/// exactly as [`super::path_matchers`] dispatches the before-request validators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterRequestHandler {
    /// `set_can_manage_experiment_permission` — grant creator MANAGE on the new
    /// experiment (reads `experiment_id` from the response).
    CreatorGrantExperiment,
    /// `set_can_manage_registered_model_permission` — grant creator MANAGE on the
    /// new registered model / prompt (namespace from the response tag).
    CreatorGrantRegisteredModel,
    /// `delete_can_manage_registered_model_permission` — sweep grants on delete.
    DeleteGrantsRegisteredModel,
    /// `rename_registered_model_permission` — rewrite grants on rename.
    RenameGrantsRegisteredModel,
    /// `filter_search_experiments`.
    FilterSearchExperiments,
    /// `filter_search_registered_models`.
    FilterSearchRegisteredModels,
    /// `filter_search_model_versions`.
    FilterSearchModelVersions,
    /// `filter_search_logged_models`.
    FilterSearchLoggedModels,
}

impl AfterRequestHandler {
    /// Whether this handler needs the response body buffered. Creator grants and
    /// the search filters read `resp.json`; the delete/rename cascades read only
    /// the request, so they do not need the response body (matching Python,
    /// which reads `request.get_json` there).
    pub fn needs_response_body(self) -> bool {
        !matches!(
            self,
            AfterRequestHandler::DeleteGrantsRegisteredModel
                | AfterRequestHandler::RenameGrantsRegisteredModel
        )
    }
}

/// Map a request `(service, method)` to its after-request handler, mirroring
/// `AFTER_REQUEST_PATH_HANDLERS` keyed on the proto request class. Returns
/// `None` for RPCs with no after-request hook (the overwhelming majority).
///
/// Only RPCs this Rust server actually serves are wired (see the module docs on
/// the omitted scorer/gateway/review-queue surfaces).
pub fn handler_for(service: &str, method: &str) -> Option<AfterRequestHandler> {
    use AfterRequestHandler::*;
    let h = match (service, method) {
        ("MlflowService", "createExperiment") => CreatorGrantExperiment,
        ("MlflowService", "searchExperiments") => FilterSearchExperiments,
        ("MlflowService", "searchLoggedModels") => FilterSearchLoggedModels,
        ("ModelRegistryService", "createRegisteredModel") => CreatorGrantRegisteredModel,
        ("ModelRegistryService", "deleteRegisteredModel") => DeleteGrantsRegisteredModel,
        ("ModelRegistryService", "renameRegisteredModel") => RenameGrantsRegisteredModel,
        ("ModelRegistryService", "searchRegisteredModels") => FilterSearchRegisteredModels,
        ("ModelRegistryService", "searchModelVersions") => FilterSearchModelVersions,
        _ => return None,
    };
    Some(h)
}

/// Everything the after-request handlers need. Built by the middleware after the
/// downstream handler ran and (when required) the response body was buffered.
///
/// The search filters re-parse the *request* proto (mirroring the handler's
/// `_get_request_message`) to recover `max_results` / `filter` / `order_by` /
/// `page_token`, so the ctx carries the buffered request `parts` + `body`.
pub struct AfterCtx<'a> {
    pub username: &'a str,
    pub workspace: &'a str,
    pub is_admin: bool,
    /// The request method (`"GET"`/`"POST"`/...), for re-parsing the search
    /// request exactly as the handler did.
    pub method: &'a str,
    /// The raw request query string (`None` when empty), the GET search input.
    pub query: Option<&'a str>,
    /// The buffered raw request body (the POST search input).
    pub request_body: &'a Bytes,
    /// The parsed request JSON body (for the delete/rename cascades), if any.
    pub request_json: Option<&'a serde_json::Value>,
    pub state: &'a AppState,
}

impl AfterCtx<'_> {
    /// Re-parse the buffered request into a proto message exactly as the search
    /// handler's `parse_request` did (`_get_request_message`): GET with a
    /// non-empty query parses from the query pairs; otherwise parse the JSON
    /// body (empty body → `{}`). Content-Type validation is skipped — the
    /// request was already accepted by the handler, so re-validating is moot.
    ///
    /// This mirrors [`crate::proto_http::parse_request`] but takes the request
    /// pieces by value/ref rather than a `&Parts` (whose `Extensions` field is
    /// `!Sync`, which would make the middleware future non-`Send`).
    fn parse_request<M: prost::Message + Default>(
        &self,
        type_name: &str,
    ) -> Result<M, MlflowError> {
        if self.method == "GET" {
            if let Some(query) = self.query.filter(|q| !q.is_empty()) {
                let pairs = crate::proto_http::parse_query_pairs(query);
                return mlflow_proto::from_query_pairs::<M>(&pairs, type_name)
                    .map_err(crate::proto_http::codec_err);
            }
        }
        let text = std::str::from_utf8(self.request_body).map_err(|_| {
            MlflowError::invalid_parameter_value("Request body is not valid UTF-8.")
        })?;
        let json = if text.trim().is_empty() { "{}" } else { text };
        mlflow_proto::from_mlflow_json::<M>(json, type_name).map_err(crate::proto_http::codec_err)
    }
}

/// `_after_request` (`__init__.py:3651`): run the handler for a successful
/// response. `resp_body` is the buffered response body (only present when
/// [`AfterRequestHandler::needs_response_body`]). On the filter/grant paths this
/// may rewrite `resp` (a new body) or leave it untouched; errors surface as the
/// same JSON error `MlflowError` produces (matching Python's
/// `@catch_mlflow_exception` around `_after_request`).
pub async fn run(
    handler: AfterRequestHandler,
    ctx: &AfterCtx<'_>,
    resp: Response,
    resp_body: Option<Bytes>,
) -> Response {
    match run_inner(handler, ctx, resp_body).await {
        Ok(Some(new_body)) => rebuild_body(resp, new_body),
        Ok(None) => resp,
        Err(e) => e.into_response(),
    }
}

/// The fallible core. Returns `Some(new_body)` when the response body was
/// rewritten (filters), `None` when the response is unchanged (creator grants /
/// cascades / admin skips).
async fn run_inner(
    handler: AfterRequestHandler,
    ctx: &AfterCtx<'_>,
    resp_body: Option<Bytes>,
) -> Result<Option<Vec<u8>>, MlflowError> {
    use AfterRequestHandler::*;
    match handler {
        CreatorGrantExperiment => {
            grant_creator_experiment(ctx, body_json(resp_body)?).await?;
            Ok(None)
        }
        CreatorGrantRegisteredModel => {
            grant_creator_registered_model(ctx, body_json(resp_body)?).await?;
            Ok(None)
        }
        DeleteGrantsRegisteredModel => {
            delete_grants_registered_model(ctx).await?;
            Ok(None)
        }
        RenameGrantsRegisteredModel => {
            rename_grants_registered_model(ctx).await?;
            Ok(None)
        }
        FilterSearchExperiments => filter_search_experiments(ctx, body_json(resp_body)?).await,
        FilterSearchRegisteredModels => {
            filter_search_registered_models(ctx, body_json(resp_body)?).await
        }
        FilterSearchModelVersions => filter_search_model_versions(ctx, body_json(resp_body)?).await,
        FilterSearchLoggedModels => filter_search_logged_models(ctx, body_json(resp_body)?).await,
    }
}

// ---------------------------------------------------------------------------
// Creator grants
// ---------------------------------------------------------------------------

/// `set_can_manage_experiment_permission` (`__init__.py:2948`): parse the
/// created experiment's id from the response and grant the caller MANAGE.
async fn grant_creator_experiment(
    ctx: &AfterCtx<'_>,
    resp_json: serde_json::Value,
) -> Result<(), MlflowError> {
    let experiment_id = resp_json
        .get("experiment_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    ctx.state
        .auth_store()
        .expect("auth enabled")
        .grant_user_permission(
            ctx.username,
            "experiment",
            experiment_id,
            "MANAGE",
            ctx.workspace,
        )
        .await
}

/// `set_can_manage_registered_model_permission` (`__init__.py:2956`): classify
/// the created entity as `prompt` vs `registered_model` from its persisted
/// `mlflow.prompt.is_prompt` tag (present in the response) and grant MANAGE in
/// the matching namespace, so a prompt creator gets `(prompt, name, MANAGE)`.
async fn grant_creator_registered_model(
    ctx: &AfterCtx<'_>,
    resp_json: serde_json::Value,
) -> Result<(), MlflowError> {
    let model = resp_json.get("registered_model");
    let name = model
        .and_then(|m| m.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let resource_type = if model.map(entity_json_is_prompt).unwrap_or(false) {
        "prompt"
    } else {
        "registered_model"
    };
    ctx.state
        .auth_store()
        .expect("auth enabled")
        .grant_user_permission(ctx.username, resource_type, name, "MANAGE", ctx.workspace)
        .await
}

// ---------------------------------------------------------------------------
// Grant cascade on delete / rename
// ---------------------------------------------------------------------------

/// `delete_can_manage_registered_model_permission` (`__init__.py:2973`): sweep
/// both the `registered_model` and `prompt` grant namespaces for `name`
/// (workspace-scoped). `name` comes from the request body; a missing one is the
/// same missing-param 400 Python raises.
async fn delete_grants_registered_model(ctx: &AfterCtx<'_>) -> Result<(), MlflowError> {
    let name = required_request_str(ctx, "name")?;
    let store = ctx.state.auth_store().expect("auth enabled");
    store
        .delete_grants_for_resource("registered_model", &name, Some(ctx.workspace))
        .await?;
    store
        .delete_grants_for_resource("prompt", &name, Some(ctx.workspace))
        .await
}

/// `rename_registered_model_permission` (`__init__.py:3477`): rewrite grants on
/// `(name -> new_name)` across both namespaces (workspace-scoped).
async fn rename_grants_registered_model(ctx: &AfterCtx<'_>) -> Result<(), MlflowError> {
    let old_name = request_str(ctx, "name");
    let new_name = request_str(ctx, "new_name");
    let (Some(old_name), Some(new_name)) = (old_name, new_name) else {
        return Err(MlflowError::invalid_parameter_value(
            "Missing value for required parameter 'name' or 'new_name'.",
        ));
    };
    let store = ctx.state.auth_store().expect("auth enabled");
    store
        .rename_grants_for_resource(
            "registered_model",
            &old_name,
            &new_name,
            Some(ctx.workspace),
        )
        .await?;
    store
        .rename_grants_for_resource("prompt", &old_name, &new_name, Some(ctx.workspace))
        .await
}

// ---------------------------------------------------------------------------
// Response filtering — experiments
// ---------------------------------------------------------------------------

/// `filter_search_experiments` (`__init__.py:3285`): drop unreadable rows, then
/// re-fetch from the next token to refill up to `max_results`, rewriting the
/// token. Admins skip.
async fn filter_search_experiments(
    ctx: &AfterCtx<'_>,
    resp_json: serde_json::Value,
) -> Result<Option<Vec<u8>>, MlflowError> {
    if ctx.is_admin {
        return Ok(None);
    }
    let mut resp: pb::search_experiments::Response =
        parse_response(&resp_json, "mlflow.SearchExperiments.Response")?;

    let readable = readable_set(ctx, "experiment").await?;
    let can_read =
        |exp: &pb::Experiment| readable.allows(exp.experiment_id.as_deref().unwrap_or(""));

    resp.experiments.retain(&can_read);

    // The request drives max_results / filter / order_by / view_type.
    let req = search_experiments_request(ctx)?;
    let max_results = req.max_results.unwrap_or(0);
    let view_type = view_type_from_proto(req.view_type);
    let filter = req.filter.as_deref().filter(|s| !s.is_empty());

    while (resp.experiments.len() as i64) < max_results
        && resp.next_page_token.as_deref().unwrap_or("") != ""
    {
        let token = resp.next_page_token.clone().unwrap_or_default();
        let ExperimentsPage {
            experiments,
            next_page_token: _,
        } = ctx
            .state
            .tracking_store()
            .search_experiments(
                ctx.workspace,
                view_type,
                max_results,
                filter,
                &req.order_by,
                Some(&token),
            )
            .await?;

        // `refetched[: max_results - len(kept)]` — Python truncates the page to
        // what's still needed *before* filtering, and advances the offset by the
        // (untruncated-then-truncated) fetched count.
        let need = (max_results - resp.experiments.len() as i64).max(0) as usize;
        let refetched: Vec<_> = experiments.into_iter().take(need).collect();
        if refetched.is_empty() {
            resp.next_page_token = Some(String::new());
            break;
        }
        let fetched_len = refetched.len() as i64;
        for exp in refetched {
            let proto = to_proto_experiment(exp);
            if can_read(&proto) {
                resp.experiments.push(proto);
            }
        }
        let start_offset = parse_start_offset_from_page_token(Some(&token)).map_err(offset_err)?;
        resp.next_page_token = Some(create_page_token(start_offset + fetched_len));
    }

    serialize_response(&resp, "mlflow.SearchExperiments.Response").map(Some)
}

// ---------------------------------------------------------------------------
// Response filtering — registered models
// ---------------------------------------------------------------------------

/// `filter_search_registered_models` (`__init__.py:3400`): drop unreadable rows
/// (classified by `mlflow.prompt.is_prompt` into the RM/prompt namespace), then
/// re-fetch to refill.
async fn filter_search_registered_models(
    ctx: &AfterCtx<'_>,
    resp_json: serde_json::Value,
) -> Result<Option<Vec<u8>>, MlflowError> {
    if ctx.is_admin {
        return Ok(None);
    }
    let mut resp: pb::search_registered_models::Response =
        parse_response(&resp_json, "mlflow.SearchRegisteredModels.Response")?;

    let rm_readable = readable_set(ctx, "registered_model").await?;
    let prompt_readable = readable_set(ctx, "prompt").await?;
    let can_read = |m: &pb::RegisteredModel| {
        let name = m.name.as_deref().unwrap_or("");
        if registered_model_is_prompt(m) {
            prompt_readable.allows(name)
        } else {
            rm_readable.allows(name)
        }
    };

    resp.registered_models.retain(&can_read);

    let req = search_registered_models_request(ctx)?;
    let max_results = req.max_results.unwrap_or(100);
    let filter = req.filter.as_deref().filter(|s| !s.is_empty());

    while (resp.registered_models.len() as i64) < max_results
        && resp.next_page_token.as_deref().unwrap_or("") != ""
    {
        let token = resp.next_page_token.clone().unwrap_or_default();
        let RegisteredModelsPage {
            registered_models, ..
        } = ctx
            .state
            .registry_store()?
            .search_registered_models(
                ctx.workspace,
                filter,
                max_results,
                &req.order_by,
                Some(&token),
            )
            .await?;
        let need = (max_results - resp.registered_models.len() as i64).max(0) as usize;
        let refetched: Vec<_> = registered_models.into_iter().take(need).collect();
        if refetched.is_empty() {
            resp.next_page_token = Some(String::new());
            break;
        }
        let fetched_len = refetched.len() as i64;
        for rm in refetched {
            let proto = to_proto_registered_model(rm);
            if can_read(&proto) {
                resp.registered_models.push(proto);
            }
        }
        let start_offset = parse_start_offset_from_page_token(Some(&token)).map_err(offset_err)?;
        resp.next_page_token = Some(create_page_token(start_offset + fetched_len));
    }

    serialize_response(&resp, "mlflow.SearchRegisteredModels.Response").map(Some)
}

// ---------------------------------------------------------------------------
// Response filtering — model versions (no refetch; drop-only)
// ---------------------------------------------------------------------------

/// `filter_search_model_versions` (`__init__.py:3456`): drop model versions
/// whose (RM/prompt-classified) parent model is unreadable. No page-fill — the
/// response is filtered in place and the token left untouched.
async fn filter_search_model_versions(
    ctx: &AfterCtx<'_>,
    resp_json: serde_json::Value,
) -> Result<Option<Vec<u8>>, MlflowError> {
    if ctx.is_admin {
        return Ok(None);
    }
    let mut resp: pb::search_model_versions::Response =
        parse_response(&resp_json, "mlflow.SearchModelVersions.Response")?;

    let rm_readable = readable_set(ctx, "registered_model").await?;
    let prompt_readable = readable_set(ctx, "prompt").await?;
    resp.model_versions.retain(|mv| {
        let name = mv.name.as_deref().unwrap_or("");
        if model_version_is_prompt(mv) {
            prompt_readable.allows(name)
        } else {
            rm_readable.allows(name)
        }
    });

    serialize_response(&resp, "mlflow.SearchModelVersions.Response").map(Some)
}

// ---------------------------------------------------------------------------
// Response filtering — logged models
// ---------------------------------------------------------------------------

/// `filter_search_logged_models` (`__init__.py:3330`): drop logged models whose
/// parent experiment is unreadable, then re-fetch to refill. The logged-model
/// token is the opaque `SearchLoggedModelsPaginationToken`, so page-fill is
/// driven by the store's own returned token, and the per-row token math mirrors
/// Python's index-based token recomputation.
async fn filter_search_logged_models(
    ctx: &AfterCtx<'_>,
    resp_json: serde_json::Value,
) -> Result<Option<Vec<u8>>, MlflowError> {
    if ctx.is_admin {
        return Ok(None);
    }
    let mut resp: pb::search_logged_models::Response =
        parse_response(&resp_json, "mlflow.SearchLoggedModels.Response")?;

    let readable = readable_set(ctx, "experiment").await?;
    let can_read = |m: &pb::LoggedModel| {
        readable.allows(
            m.info
                .as_ref()
                .and_then(|i| i.experiment_id.as_deref())
                .unwrap_or(""),
        )
    };
    resp.models.retain(&can_read);

    let req = search_logged_models_request(ctx)?;
    // Python's loop bound is `request_proto.max_results` (proto default 50). The
    // store call gets the handler's `filter(|n| *n > 0)` value so an explicit 0
    // still means "use the store default" (parity with the search handler).
    let max_results = req.max_results.filter(|n| *n > 0).map(|n| n as usize);
    let max = req.max_results.unwrap_or(50).max(0) as usize;
    let datasets: Vec<DatasetFilter> = req
        .datasets
        .iter()
        .map(|d| DatasetFilter {
            dataset_name: d.dataset_name.clone().unwrap_or_default(),
            dataset_digest: d.dataset_digest.clone().filter(|s| !s.is_empty()),
        })
        .collect();
    let order_by: Vec<LoggedModelOrderByInput> = req
        .order_by
        .iter()
        .map(|ob| LoggedModelOrderByInput {
            field_name: ob.field_name.clone().unwrap_or_default(),
            ascending: ob.ascending.unwrap_or(true),
            dataset_name: ob.dataset_name.clone().filter(|s| !s.is_empty()),
            dataset_digest: ob.dataset_digest.clone().filter(|s| !s.is_empty()),
        })
        .collect();
    let filter = req.filter.as_deref().filter(|s| !s.is_empty());

    // Mirror `filter_search_logged_models`'s exact index-based token math: for
    // each refetched batch, decode the current token's offset and — when the
    // page fills mid-batch — resume at `offset + index + 1` (or clear the token
    // if that was the last row of the last page); when the batch is exhausted
    // without filling, resume at `offset + max_results` (or clear on the last
    // page). The opaque logged-model token round-trips through the store's
    // [`logged_models_token_offset`] / [`logged_models_page_token`].
    let make_token = |offset: usize| {
        mlflow_store::logged_models_page_token(offset, &req.experiment_ids, filter, &order_by)
    };
    let mut next_page_token = resp.next_page_token.clone().filter(|t| !t.is_empty());

    while resp.models.len() < max && next_page_token.is_some() {
        let token = next_page_token.clone().unwrap();
        let LoggedModelsPage {
            models,
            next_page_token: store_next,
        } = ctx
            .state
            .tracking_store()
            .search_logged_models(
                ctx.workspace,
                &req.experiment_ids,
                filter,
                &datasets,
                max_results,
                &order_by,
                Some(&token),
            )
            .await?;

        let is_last_page = store_next.is_none();
        let offset = mlflow_store::logged_models_token_offset(&token)?;
        let last_index = models.len().saturating_sub(1);
        let mut filled_mid_batch = false;
        for (index, model) in models.into_iter().enumerate() {
            let proto = to_proto_logged_model(model);
            if !can_read(&proto) {
                continue;
            }
            resp.models.push(proto);
            if resp.models.len() >= max {
                next_page_token = if is_last_page && index == last_index {
                    None
                } else {
                    Some(make_token(offset + index + 1))
                };
                filled_mid_batch = true;
                break;
            }
        }
        if !filled_mid_batch {
            // Batch exhausted without filling the page (Python's `for/else`).
            next_page_token = if is_last_page {
                None
            } else {
                Some(make_token(offset + max))
            };
        }
    }

    // `if next_page_token: response_proto.next_page_token = next_page_token` —
    // Python only overwrites when the recomputed token is truthy, so a loop that
    // exhausted (token → None) leaves the *initial* response token in place.
    if let Some(token) = next_page_token {
        resp.next_page_token = Some(token);
    }
    serialize_response(&resp, "mlflow.SearchLoggedModels.Response").map(Some)
}

// ---------------------------------------------------------------------------
// Read predicate (`_role_based_read_predicate`, `__init__.py:1586`)
// ---------------------------------------------------------------------------

/// The set of resource patterns a user can read for a resource type in a
/// workspace, plus the wildcard/default fallbacks — the resolved form of
/// `_role_based_read_predicate`.
struct ReadableSet {
    readable: std::collections::HashSet<String>,
    wildcard: bool,
    fallback: bool,
}

impl ReadableSet {
    fn allows(&self, resource_id: &str) -> bool {
        self.readable.contains(resource_id) || self.wildcard || self.fallback
    }
}

/// Build the [`ReadableSet`] for `(user, resource_type, workspace)`, mirroring
/// `_role_based_read_predicate`: any positive (`can_read`) grant — specific or
/// wildcard — makes the resource readable; `NO_PERMISSIONS` rows are ignored.
/// Pre-T10.4 the server is single-tenant, so the fallback is
/// `default_permission.can_read` (Python's "workspaces disabled" branch).
async fn readable_set(ctx: &AfterCtx<'_>, resource_type: &str) -> Result<ReadableSet, MlflowError> {
    let store: &AuthStore = ctx.state.auth_store().expect("auth enabled");
    let user = store.get_user(ctx.username).await?;
    let grants = store
        .list_role_grants_for_user_in_workspace(user.id, ctx.workspace, resource_type)
        .await?;

    let mut readable = std::collections::HashSet::new();
    let mut wildcard = false;
    for (pattern, permission) in grants {
        if !get_permission(&permission).can_read {
            continue;
        }
        if pattern == "*" {
            wildcard = true;
        } else {
            readable.insert(pattern);
        }
    }
    // T10.4 SEAM: when workspaces are enabled the fallback is deny; pre-T10.4
    // (single-tenant) it is `default_permission.can_read`, matching Python's
    // `MLFLOW_ENABLE_WORKSPACES` off branch.
    let fallback = super::validators::default_permission().can_read;
    Ok(ReadableSet {
        readable,
        wildcard,
        fallback,
    })
}

// ---------------------------------------------------------------------------
// Prompt classification (`_entity_is_prompt`, `__init__.py:1558`)
// ---------------------------------------------------------------------------

/// True if a JSON entity object carries `mlflow.prompt.is_prompt = "true"`
/// (case-insensitive) in its `tags` array. Used for the creator-grant namespace.
fn entity_json_is_prompt(entity: &serde_json::Value) -> bool {
    entity
        .get("tags")
        .and_then(|t| t.as_array())
        .map(|tags| {
            tags.iter().any(|t| {
                t.get("key").and_then(|k| k.as_str()) == Some(IS_PROMPT_TAG_KEY)
                    && t.get("value")
                        .and_then(|v| v.as_str())
                        .map(|v| v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn registered_model_is_prompt(m: &pb::RegisteredModel) -> bool {
    m.tags.iter().any(|t| {
        t.key.as_deref() == Some(IS_PROMPT_TAG_KEY)
            && t.value
                .as_deref()
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    })
}

fn model_version_is_prompt(mv: &pb::ModelVersion) -> bool {
    mv.tags.iter().any(|t| {
        t.key.as_deref() == Some(IS_PROMPT_TAG_KEY)
            && t.value
                .as_deref()
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    })
}

// ---------------------------------------------------------------------------
// Request re-parsing (`_get_request_message(Search*())`)
// ---------------------------------------------------------------------------

fn search_experiments_request(ctx: &AfterCtx<'_>) -> Result<pb::SearchExperiments, MlflowError> {
    ctx.parse_request("mlflow.SearchExperiments")
}

fn search_registered_models_request(
    ctx: &AfterCtx<'_>,
) -> Result<pb::SearchRegisteredModels, MlflowError> {
    ctx.parse_request("mlflow.SearchRegisteredModels")
}

fn search_logged_models_request(ctx: &AfterCtx<'_>) -> Result<pb::SearchLoggedModels, MlflowError> {
    ctx.parse_request("mlflow.SearchLoggedModels")
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// `_get_request_param(param)` semantics for a *string* value from the request
/// body (POST/PATCH) or query (GET). Used by the delete/rename cascades which
/// read from the request, not the response.
fn request_str(ctx: &AfterCtx<'_>, param: &str) -> Option<String> {
    ctx.request_json
        .and_then(|b| b.get(param))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn required_request_str(ctx: &AfterCtx<'_>, param: &str) -> Result<String, MlflowError> {
    request_str(ctx, param).ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!(
            "Missing value for required parameter '{param}'."
        ))
    })
}

fn body_json(resp_body: Option<Bytes>) -> Result<serde_json::Value, MlflowError> {
    let bytes = resp_body.ok_or_else(|| {
        MlflowError::internal_error(
            "after-request handler needs the response body but none was buffered",
        )
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|e| MlflowError::internal_error(format!("could not parse response JSON: {e}")))
}

fn parse_response<M: prost::Message + Default>(
    resp_json: &serde_json::Value,
    type_name: &str,
) -> Result<M, MlflowError> {
    mlflow_proto::from_mlflow_json::<M>(&resp_json.to_string(), type_name)
        .map_err(crate::proto_http::codec_err)
}

fn serialize_response<M: prost::Message>(
    message: &M,
    type_name: &str,
) -> Result<Vec<u8>, MlflowError> {
    mlflow_proto::to_mlflow_json(message, type_name)
        .map(|s| s.into_bytes())
        .map_err(crate::proto_http::codec_err)
}

/// Map a `mlflow_search::SearchError` (only raised here by page-token decoding)
/// to the matching `MlflowError`, mirroring the stores' own `search_err`.
fn offset_err(e: mlflow_search::SearchError) -> MlflowError {
    use mlflow_error::ErrorCode;
    let code = match e.error_code {
        mlflow_search::ErrorCode::InvalidParameterValue => ErrorCode::InvalidParameterValue,
        _ => ErrorCode::InternalError,
    };
    MlflowError::new(e.message, code)
}

fn view_type_from_proto(view_type: Option<i32>) -> Option<mlflow_store::ViewType> {
    match view_type {
        Some(v) if v == pb::ViewType::ActiveOnly as i32 => Some(mlflow_store::ViewType::ActiveOnly),
        Some(v) if v == pb::ViewType::DeletedOnly as i32 => {
            Some(mlflow_store::ViewType::DeletedOnly)
        }
        Some(v) if v == pb::ViewType::All as i32 => Some(mlflow_store::ViewType::All),
        _ => None,
    }
}

/// `MLFLOW_RUST_AUTH_QUERY_INTEGRATED_FILTERING` (plan Q10). Unused today — the
/// Python-identical refetch is always the default (parity is the ship gate).
/// The seam is here so the optimization can be wired later without touching the
/// dispatch surface.
#[allow(dead_code)]
fn query_integrated_filtering_enabled() -> bool {
    std::env::var("MLFLOW_RUST_AUTH_QUERY_INTEGRATED_FILTERING")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false)
}

/// Rebuild `resp` with a new body, preserving the status and headers (the search
/// handlers always emit `200 application/json`, and the filtered body is still
/// JSON, so the `Content-Type` is unchanged; `Content-Length` is recomputed by
/// the body swap).
fn rebuild_body(resp: Response, new_body: Vec<u8>) -> Response {
    let (mut parts, _) = resp.into_parts();
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.status = StatusCode::OK;
    Response::from_parts(parts, Body::from(new_body))
}
