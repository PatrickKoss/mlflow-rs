//! Model Registry endpoints (plan T7.4, §3.14): 21 RPCs across registered
//! models, model versions, and aliases.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`
//! (`_create_registered_model` .. `_get_model_version_by_alias`,
//! ~L2618-3361): parse the request proto (via [`crate::proto_http`]), run the
//! handler-level required-field validation (`_assert_required` →
//! [`require_non_empty`]), apply any handler-level default/threshold on
//! `max_results`, call the workspace-scoped [`mlflow_registry::RegistryStore`]
//! method, then serialize the response proto.
//!
//! ## Handler-level `max_results` defaults/thresholds (§4.14)
//!
//! The store validates the *threshold* (RM 1000, MV 200000). The **default**
//! when `max_results` is absent is a handler concern, and — critically — the
//! store's default wins over the proto declaration for model versions:
//! `search_registered_models` defaults to 100, `search_model_versions` to
//! 10000 (NOT the proto's 200000). See [`SEARCH_REGISTERED_MODELS_DEFAULT`] /
//! [`SEARCH_MODEL_VERSIONS_DEFAULT`].
//!
//! ## `createModelVersion` source validation (security-relevant)
//!
//! Python validates the `source` at the handler level *before* touching the
//! store, rejecting local-path escapes and mismatched run/model sources with
//! exact messages (`_validate_source_run` / `_validate_source_model` /
//! `_validate_non_local_source_contains_relative_paths`,
//! `handlers.py:2829-2963`). [`source_validation`] ports these byte-for-byte —
//! this is the path-traversal defense for the registry.
//!
//! ## Webhook event triggers (T8.4)
//!
//! Python fires webhook events from several registry mutations: registered
//! model created; model version created; MV tag set/deleted; MV alias
//! set/deleted; plus the `PROMPT_*` mirrors selected by an `is_prompt`
//! classification. Each mutation calls [`fire_event`] after the store call
//! succeeds (post-commit, matching Python's ordering), building the exact
//! `(entity, action, data)` payload in [`webhook_events`]. Delivery is
//! fire-and-forget through [`WebhookDispatcher::fire`] and never errors, so no
//! failure ever surfaces to the mutation response — mirroring Python's
//! `deliver_webhook` top-level swallow. When the backend does not support
//! webhooks ([`AppState::webhook_dispatcher`] is `None`) nothing is fired.
//!
//! Prompt classification is *instead of*, not in addition to (see
//! [`webhook_events`]): RM/MV *create* classify from the request tags
//! (`_is_prompt_request`), while the tag/alias mutations do a fresh
//! post-mutation `get_registered_model` lookup (`_is_prompt(name)`) via
//! [`is_prompt_model`].

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow as pb;
use mlflow_registry::{
    ModelVersion, ModelVersionTag, RegisteredModel, RegisteredModelAlias, RegisteredModelTag,
};

use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

mod source_validation;
mod webhook_events;

use source_validation::validate_create_model_version_source;
use webhook_events::Tag;

/// Handler-level default for `search_registered_models` `max_results`
/// (`SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT`, applied via the proto
/// `default = 100`; the store validates the 1000 threshold).
const SEARCH_REGISTERED_MODELS_DEFAULT: i64 = 100;

/// Handler-level default for `search_model_versions` `max_results`
/// (`SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT` = 10000). The proto declares
/// `default = 200000`, but the store's default wins (§4.14) — so when the
/// client omits `max_results`, we resolve to 10000, NOT the proto default.
const SEARCH_MODEL_VERSIONS_DEFAULT: i64 = 10000;

/// `mlflow.prompt.is_prompt` (`mlflow/prompt/constants.py:4`).
const IS_PROMPT_TAG_KEY: &str = "mlflow.prompt.is_prompt";

// ===========================================================================
// Registered models
// ===========================================================================

/// `_create_registered_model` (`handlers.py:2618`).
pub async fn create_registered_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateRegisteredModel =
        parse_request(&parts, &body, "mlflow.CreateRegisteredModel")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;

    let tags: Vec<(&str, &str)> = req
        .tags
        .iter()
        .map(|t| {
            (
                t.key.as_deref().unwrap_or(""),
                t.value.as_deref().unwrap_or(""),
            )
        })
        .collect();

    let model = state
        .registry_store()?
        .create_registered_model(
            workspace.name(),
            name,
            &tags,
            req.description.as_deref().filter(|s| !s.is_empty()),
        )
        .await?;

    // Python fires REGISTERED_MODEL/CREATED (or PROMPT/CREATED when the request
    // carries the `mlflow.prompt.is_prompt` tag) here (`handlers.py:2636-2661`).
    // The payload uses the *request* tags/description (proto2 default `""`), not
    // the stored entity.
    let event_tags = request_tags(&req.tags);
    fire_event(
        &state,
        webhook_events::registered_model_created(
            name,
            &event_tags,
            req.description.as_deref().unwrap_or(""),
        ),
    )
    .await;

    let resp = pb::create_registered_model::Response {
        registered_model: Some(to_proto_registered_model(model)),
    };
    proto_response(&resp, "mlflow.CreateRegisteredModel.Response")
}

/// `_get_registered_model` (`handlers.py:2668`).
pub async fn get_registered_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetRegisteredModel = parse_request(&parts, &body, "mlflow.GetRegisteredModel")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;

    let model = state
        .registry_store()?
        .get_registered_model(workspace.name(), name)
        .await?;

    let resp = pb::get_registered_model::Response {
        registered_model: Some(to_proto_registered_model(model)),
    };
    proto_response(&resp, "mlflow.GetRegisteredModel.Response")
}

/// `_update_registered_model` (`handlers.py:2679`). PATCH.
pub async fn update_registered_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateRegisteredModel =
        parse_request(&parts, &body, "mlflow.UpdateRegisteredModel")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;

    // Python passes `request_message.description` straight through. The store
    // sets the new description; `None`/absent preserves the existing value.
    let model = state
        .registry_store()?
        .update_registered_model(workspace.name(), name, req.description.as_deref())
        .await?;

    let resp = pb::update_registered_model::Response {
        registered_model: Some(to_proto_registered_model(model)),
    };
    proto_response(&resp, "mlflow.UpdateRegisteredModel.Response")
}

/// `_rename_registered_model` (`handlers.py:2698`). POST.
pub async fn rename_registered_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::RenameRegisteredModel =
        parse_request(&parts, &body, "mlflow.RenameRegisteredModel")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let new_name = require_non_empty(req.new_name.as_deref(), "new_name")?;

    let model = state
        .registry_store()?
        .rename_registered_model(workspace.name(), name, new_name)
        .await?;

    let resp = pb::rename_registered_model::Response {
        registered_model: Some(to_proto_registered_model(model)),
    };
    proto_response(&resp, "mlflow.RenameRegisteredModel.Response")
}

/// `_delete_registered_model` (`handlers.py:2717`). DELETE.
pub async fn delete_registered_model(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteRegisteredModel =
        parse_request(&parts, &body, "mlflow.DeleteRegisteredModel")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;

    state
        .registry_store()?
        .delete_registered_model(workspace.name(), name)
        .await?;

    proto_response(
        &pb::delete_registered_model::Response {},
        "mlflow.DeleteRegisteredModel.Response",
    )
}

/// `_search_registered_models` (`handlers.py:2727`). GET only.
pub async fn search_registered_models(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchRegisteredModels =
        parse_request(&parts, &body, "mlflow.SearchRegisteredModels")?;

    // Proto declares `default = 100`, so an absent field decodes to Some(100);
    // an explicit 0 stays 0. Mirror `request_message.max_results` unchanged, but
    // fall back to the handler default if the codec ever yields None.
    let max_results = req.max_results.unwrap_or(SEARCH_REGISTERED_MODELS_DEFAULT);
    let filter = req.filter.as_deref().filter(|s| !s.is_empty());
    let page_token = req.page_token.as_deref().filter(|s| !s.is_empty());

    let page = state
        .registry_store()?
        .search_registered_models(
            workspace.name(),
            filter,
            max_results,
            &req.order_by,
            page_token,
        )
        .await?;

    let resp = pb::search_registered_models::Response {
        registered_models: page
            .registered_models
            .into_iter()
            .map(to_proto_registered_model)
            .collect(),
        next_page_token: page.next_page_token,
    };
    proto_response(&resp, "mlflow.SearchRegisteredModels.Response")
}

/// `_get_latest_versions` (`handlers.py:2756`). POST **and** GET; `stages` is a
/// repeated field (repeated query params on GET).
pub async fn get_latest_versions(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetLatestVersions = parse_request(&parts, &body, "mlflow.GetLatestVersions")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;

    // Python passes `request_message.stages` (possibly empty) to the store; an
    // empty list means "all default stages". Map an empty repeated field to
    // `None` so the store applies its default-stages behavior.
    let stage_refs: Vec<&str> = req.stages.iter().map(String::as_str).collect();
    let stages = if stage_refs.is_empty() {
        None
    } else {
        Some(stage_refs.as_slice())
    };

    let versions = state
        .registry_store()?
        .get_latest_versions(workspace.name(), name, stages)
        .await?;

    let resp = pb::get_latest_versions::Response {
        model_versions: versions.into_iter().map(to_proto_model_version).collect(),
    };
    proto_response(&resp, "mlflow.GetLatestVersions.Response")
}

/// `_set_registered_model_tag` (`handlers.py:2774`). POST.
pub async fn set_registered_model_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetRegisteredModelTag =
        parse_request(&parts, &body, "mlflow.SetRegisteredModelTag")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;
    // `value` carries `validate_required` in the proto but only `_assert_string`
    // in the handler schema — so an empty string is accepted (default "").
    let value = req.value.as_deref().unwrap_or("");

    state
        .registry_store()?
        .set_registered_model_tag(workspace.name(), name, key, value)
        .await?;

    // PROMPT_TAG/SET only when the model is a prompt (`handlers.py:2787`); a
    // non-prompt RM tag set fires nothing. `_is_prompt(name)` re-reads the model
    // post-mutation (same query, same timing as Python).
    if let Some(pair) = webhook_events::registered_model_tag_set(
        is_prompt_model(&state, &workspace, name).await,
        name,
        key,
        value,
    ) {
        fire_event(&state, pair).await;
    }

    proto_response(
        &pb::set_registered_model_tag::Response {},
        "mlflow.SetRegisteredModelTag.Response",
    )
}

/// `_delete_registered_model_tag` (`handlers.py:2804`). DELETE.
pub async fn delete_registered_model_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteRegisteredModelTag =
        parse_request(&parts, &body, "mlflow.DeleteRegisteredModelTag")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;

    state
        .registry_store()?
        .delete_registered_model_tag(workspace.name(), name, key)
        .await?;

    // PROMPT_TAG/DELETED only when the model is a prompt (`handlers.py:2815`).
    if let Some(pair) = webhook_events::registered_model_tag_deleted(
        is_prompt_model(&state, &workspace, name).await,
        name,
        key,
    ) {
        fire_event(&state, pair).await;
    }

    proto_response(
        &pb::delete_registered_model_tag::Response {},
        "mlflow.DeleteRegisteredModelTag.Response",
    )
}

// ===========================================================================
// Model versions
// ===========================================================================

/// `_create_model_version` (`handlers.py:2919`). POST. Runs the handler-level
/// source validation before touching the store (see module docs).
pub async fn create_model_version(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateModelVersion = parse_request(&parts, &body, "mlflow.CreateModelVersion")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let source = require_non_empty(req.source.as_deref(), "source")?;

    let run_id = req.run_id.as_deref().filter(|s| !s.is_empty());
    let model_id = req.model_id.as_deref().filter(|s| !s.is_empty());
    let is_prompt = req
        .tags
        .iter()
        .any(|t| t.key.as_deref() == Some(IS_PROMPT_TAG_KEY));

    validate_create_model_version_source(
        &state,
        workspace.name(),
        source,
        run_id,
        model_id,
        is_prompt,
    )
    .await?;

    let tags: Vec<(&str, &str)> = req
        .tags
        .iter()
        .map(|t| {
            (
                t.key.as_deref().unwrap_or(""),
                t.value.as_deref().unwrap_or(""),
            )
        })
        .collect();

    let version = state
        .registry_store()?
        .create_model_version(
            workspace.name(),
            name,
            source,
            run_id,
            &tags,
            req.run_link.as_deref().filter(|s| !s.is_empty()),
            req.description.as_deref(),
        )
        .await?;

    // NB (deviation, documented): Python additionally calls
    // `tracking_store.set_model_versions_tags(name, version, model_id=...)` when
    // a non-prompt `model_id` is supplied (`handlers.py:2975-2981`) to back-link
    // the logged model. The Rust registry store's `create_model_version` does
    // not accept `model_id`, and that cross-store back-link is a caller
    // responsibility deferred with the logged-model-id source resolution (see
    // `mlflow-registry` `store/model_versions.rs` module docs). The model
    // version is created identically; only the back-reference tag is not set.

    // Python fires MODEL_VERSION/CREATED (or PROMPT_VERSION/CREATED when the
    // request carries `mlflow.prompt.is_prompt`) here (`handlers.py:2984-3017`).
    // The payload uses the request `source`/`run_id`/tags and the *stored*
    // version number (`str(model_version.version)`); the prompt variant pops
    // `mlflow.prompt.text` into `template`.
    let event_tags = request_tags(&req.tags);
    fire_event(
        &state,
        webhook_events::model_version_created(
            name,
            &version.version,
            source,
            run_id,
            &event_tags,
            req.description.as_deref(),
        ),
    )
    .await;

    let resp = pb::create_model_version::Response {
        model_version: Some(to_proto_model_version(version)),
    };
    proto_response(&resp, "mlflow.CreateModelVersion.Response")
}

/// `_get_model_version` (`handlers.py:3055`). GET.
pub async fn get_model_version(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetModelVersion = parse_request(&parts, &body, "mlflow.GetModelVersion")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;

    let mv = state
        .registry_store()?
        .get_model_version(workspace.name(), name, version)
        .await?;

    let resp = pb::get_model_version::Response {
        model_version: Some(to_proto_model_version(mv)),
    };
    proto_response(&resp, "mlflow.GetModelVersion.Response")
}

/// `_update_model_version` (`handlers.py:3073`). PATCH. Only sets the
/// description when the field is present (`HasField("description")`).
pub async fn update_model_version(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateModelVersion = parse_request(&parts, &body, "mlflow.UpdateModelVersion")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;

    // Python: `new_description = request_message.description if HasField else None`.
    // prost `Option<String>` gives us `HasField` semantics directly.
    let mv = state
        .registry_store()?
        .update_model_version(workspace.name(), name, version, req.description.as_deref())
        .await?;

    let resp = pb::update_model_version::Response {
        model_version: Some(to_proto_model_version(mv)),
    };
    proto_response(&resp, "mlflow.UpdateModelVersion.Response")
}

/// `_transition_stage` (`handlers.py:3095`). POST.
pub async fn transition_model_version_stage(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::TransitionModelVersionStage =
        parse_request(&parts, &body, "mlflow.TransitionModelVersionStage")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;
    let stage = require_non_empty(req.stage.as_deref(), "stage")?;
    // Proto marks `archive_existing_versions` as required, but the handler schema
    // only `_assert_bool`s it, so an absent value defaults to `false` (proto2
    // bool default).
    let archive_existing_versions = req.archive_existing_versions.unwrap_or(false);

    let mv = state
        .registry_store()?
        .transition_model_version_stage(
            workspace.name(),
            name,
            version,
            stage,
            archive_existing_versions,
        )
        .await?;

    let resp = pb::transition_model_version_stage::Response {
        model_version: Some(to_proto_model_version(mv)),
    };
    proto_response(&resp, "mlflow.TransitionModelVersionStage.Response")
}

/// `_delete_model_version` (`handlers.py:3118`). DELETE.
pub async fn delete_model_version(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteModelVersion = parse_request(&parts, &body, "mlflow.DeleteModelVersion")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;

    state
        .registry_store()?
        .delete_model_version(workspace.name(), name, version)
        .await?;

    proto_response(
        &pb::delete_model_version::Response {},
        "mlflow.DeleteModelVersion.Response",
    )
}

/// `_search_model_versions` (`handlers.py:3145`). GET only. `max_results`
/// resolves to the store default 10000 when absent (§4.14).
pub async fn search_model_versions(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SearchModelVersions = parse_request(&parts, &body, "mlflow.SearchModelVersions")?;

    // Proto declares `default = 200000`, but the store default (10000) wins
    // (§4.14). Python's `search_model_versions_impl` passes
    // `request_message.max_results` to the store, whose default is 10000 — that
    // default only takes effect when the client omitted the field. We detect
    // omission by re-scanning the raw request; see the helper.
    let max_results = resolve_model_version_max_results(&parts, &body, &req)?;
    let filter = req.filter.as_deref().filter(|s| !s.is_empty());
    let page_token = req.page_token.as_deref().filter(|s| !s.is_empty());

    let page = state
        .registry_store()?
        .search_model_versions(
            workspace.name(),
            filter,
            max_results,
            &req.order_by,
            page_token,
        )
        .await?;

    let resp = pb::search_model_versions::Response {
        model_versions: page
            .model_versions
            .into_iter()
            .map(to_proto_model_version)
            .collect(),
        next_page_token: page.next_page_token,
    };
    proto_response(&resp, "mlflow.SearchModelVersions.Response")
}

/// `_get_model_version_download_uri` (`handlers.py:3134`). GET. The Python
/// handler passes **no schema**, so `name`/`version` are not required-validated
/// at the handler level (the store validates `version` numerically and a
/// missing name/version yields a store-level error). We replicate that by
/// forwarding the raw (possibly empty) values.
pub async fn get_model_version_download_uri(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetModelVersionDownloadUri =
        parse_request(&parts, &body, "mlflow.GetModelVersionDownloadUri")?;
    let name = req.name.as_deref().unwrap_or("");
    let version = req.version.as_deref().unwrap_or("");

    let uri = state
        .registry_store()?
        .get_model_version_download_uri(workspace.name(), name, version)
        .await?;

    let resp = pb::get_model_version_download_uri::Response {
        artifact_uri: Some(uri),
    };
    proto_response(&resp, "mlflow.GetModelVersionDownloadUri.Response")
}

/// `_set_model_version_tag` (`handlers.py:3179`). POST.
pub async fn set_model_version_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetModelVersionTag = parse_request(&parts, &body, "mlflow.SetModelVersionTag")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;
    let value = req.value.as_deref().unwrap_or("");

    state
        .registry_store()?
        .set_model_version_tag(workspace.name(), name, version, key, value)
        .await?;

    // MODEL_VERSION_TAG/SET, or PROMPT_VERSION_TAG/SET when the model is a prompt
    // (`handlers.py:3193-3216`). Always fires (unlike RM tags). Classification is
    // a post-mutation `_is_prompt(name)` lookup.
    fire_event(
        &state,
        webhook_events::model_version_tag_set(
            is_prompt_model(&state, &workspace, name).await,
            name,
            version,
            key,
            value,
        ),
    )
    .await;

    proto_response(
        &pb::set_model_version_tag::Response {},
        "mlflow.SetModelVersionTag.Response",
    )
}

/// `_delete_model_version_tag` (`handlers.py:3223`). DELETE.
pub async fn delete_model_version_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteModelVersionTag =
        parse_request(&parts, &body, "mlflow.DeleteModelVersionTag")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;
    let key = require_non_empty(req.key.as_deref(), "key")?;

    state
        .registry_store()?
        .delete_model_version_tag(workspace.name(), name, version, key)
        .await?;

    // MODEL_VERSION_TAG/DELETED, or PROMPT_VERSION_TAG/DELETED when the model is
    // a prompt (`handlers.py:3239-3260`).
    fire_event(
        &state,
        webhook_events::model_version_tag_deleted(
            is_prompt_model(&state, &workspace, name).await,
            name,
            version,
            key,
        ),
    )
    .await;

    proto_response(
        &pb::delete_model_version_tag::Response {},
        "mlflow.DeleteModelVersionTag.Response",
    )
}

// ===========================================================================
// Aliases (method-overloaded route `/mlflow/registered-models/alias`)
// ===========================================================================

/// `_set_registered_model_alias` (`handlers.py:3267`). POST on the shared alias
/// path.
pub async fn set_registered_model_alias(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::SetRegisteredModelAlias =
        parse_request(&parts, &body, "mlflow.SetRegisteredModelAlias")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let alias = require_non_empty(req.alias.as_deref(), "alias")?;
    let version = require_non_empty(req.version.as_deref(), "version")?;

    state
        .registry_store()?
        .set_registered_model_alias(workspace.name(), name, alias, version)
        .await?;

    // MODEL_VERSION_ALIAS/CREATED, or PROMPT_ALIAS/CREATED when the model is a
    // prompt (`handlers.py:3283-3304`).
    fire_event(
        &state,
        webhook_events::registered_model_alias_set(
            is_prompt_model(&state, &workspace, name).await,
            name,
            alias,
            version,
        ),
    )
    .await;

    proto_response(
        &pb::set_registered_model_alias::Response {},
        "mlflow.SetRegisteredModelAlias.Response",
    )
}

/// `_delete_registered_model_alias` (`handlers.py:3311`). DELETE on the shared
/// alias path.
pub async fn delete_registered_model_alias(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteRegisteredModelAlias =
        parse_request(&parts, &body, "mlflow.DeleteRegisteredModelAlias")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let alias = require_non_empty(req.alias.as_deref(), "alias")?;

    state
        .registry_store()?
        .delete_registered_model_alias(workspace.name(), name, alias)
        .await?;

    // MODEL_VERSION_ALIAS/DELETED, or PROMPT_ALIAS/DELETED when the model is a
    // prompt (`handlers.py:3322-3341`).
    fire_event(
        &state,
        webhook_events::registered_model_alias_deleted(
            is_prompt_model(&state, &workspace, name).await,
            name,
            alias,
        ),
    )
    .await;

    proto_response(
        &pb::delete_registered_model_alias::Response {},
        "mlflow.DeleteRegisteredModelAlias.Response",
    )
}

/// `_get_model_version_by_alias` (`handlers.py:3348`). GET on the shared alias
/// path.
pub async fn get_model_version_by_alias(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetModelVersionByAlias =
        parse_request(&parts, &body, "mlflow.GetModelVersionByAlias")?;
    let name = require_non_empty(req.name.as_deref(), "name")?;
    let alias = require_non_empty(req.alias.as_deref(), "alias")?;

    let mv = state
        .registry_store()?
        .get_model_version_by_alias(workspace.name(), name, alias)
        .await?;

    let resp = pb::get_model_version_by_alias::Response {
        model_version: Some(to_proto_model_version(mv)),
    };
    proto_response(&resp, "mlflow.GetModelVersionByAlias.Response")
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Fire a webhook event through the dispatcher, if this server is configured
/// with one. Fire-and-forget: [`WebhookDispatcher::fire`] returns immediately
/// after enqueuing and never errors (delivery failures are logged + swallowed,
/// matching Python's `deliver_webhook`). A `None` dispatcher (webhooks
/// unsupported by the backend) is a silent no-op.
async fn fire_event(
    state: &AppState,
    event_and_data: (mlflow_webhooks::WebhookEvent, serde_json::Value),
) {
    if let Some(dispatcher) = state.webhook_dispatcher() {
        let (event, data) = event_and_data;
        dispatcher.fire(event, data).await;
    }
}

/// `_is_prompt(name)` (`handlers.py:3026`): re-read the registered model and
/// check its stored `mlflow.prompt.is_prompt` tag (defaulting to `"false"`,
/// case-insensitive `"true"`). Performed post-mutation, the same query and
/// timing Python uses at each tag/alias trigger site.
///
/// A lookup failure is treated as "not a prompt" so classification never blocks
/// the (already-committed) mutation response — Python would raise, but the
/// mutation has succeeded and the webhook is a best-effort side effect; skipping
/// it is strictly safer than surfacing a 500 after a successful write. This only
/// arises if the model vanished between the mutation and the lookup.
async fn is_prompt_model(state: &AppState, workspace: &Workspace, name: &str) -> bool {
    let Ok(store) = state.registry_store() else {
        return false;
    };
    match store.get_registered_model(workspace.name(), name).await {
        Ok(model) => {
            let value = model
                .tags
                .iter()
                .find(|t| t.key == webhook_events::IS_PROMPT_TAG_KEY)
                .and_then(|t| t.value.as_deref());
            webhook_events::is_prompt_tag_true(value)
        }
        Err(_) => false,
    }
}

/// A request tag proto carrying an optional key/value — implemented by both
/// `RegisteredModelTag` (create-RM) and `ModelVersionTag` (create-MV), so
/// [`request_tags`] serves both create handlers.
trait RequestTag {
    fn key_str(&self) -> &str;
    fn value_str(&self) -> &str;
}

impl RequestTag for pb::RegisteredModelTag {
    fn key_str(&self) -> &str {
        self.key.as_deref().unwrap_or("")
    }
    fn value_str(&self) -> &str {
        self.value.as_deref().unwrap_or("")
    }
}

impl RequestTag for pb::ModelVersionTag {
    fn key_str(&self) -> &str {
        self.key.as_deref().unwrap_or("")
    }
    fn value_str(&self) -> &str {
        self.value.as_deref().unwrap_or("")
    }
}

/// Borrow the request's repeated tags as `(key, value)` pairs for the webhook
/// payload builders, mirroring Python's `{t.key: t.value for t in tags}` (an
/// absent proto value is the empty string, the proto2 scalar default).
fn request_tags<T: RequestTag>(tags: &[T]) -> Vec<Tag<'_>> {
    tags.iter()
        .map(|t| Tag {
            key: t.key_str(),
            value: t.value_str(),
        })
        .collect()
}

/// Enforce a required, non-empty string field, matching `_assert_required`
/// (absent OR empty string is "missing") and its verbatim error message.
fn require_non_empty<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    match value {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(MlflowError::new(
            format!(
                "Missing value for required parameter '{param}'. \
                 See the API docs for more information about request parameters."
            ),
            ErrorCode::InvalidParameterValue,
        )),
    }
}

/// Resolve `search_model_versions`'s `max_results`, applying the store default
/// (10000) when the client omitted the field.
///
/// The proto declares `default = 200000`, so the codec fills 200000 for an
/// omitted field — but the OSS store's default is 10000, and the store wins
/// (§4.14). To tell an *explicit* `max_results=200000` from an omitted one, we
/// re-scan the raw request for a `max_results` key: present → use the parsed
/// proto value verbatim; absent → substitute the store default.
fn resolve_model_version_max_results(
    parts: &Parts,
    body: &Bytes,
    req: &pb::SearchModelVersions,
) -> Result<i64, MlflowError> {
    if request_has_max_results(parts, body)? {
        Ok(req.max_results.unwrap_or(SEARCH_MODEL_VERSIONS_DEFAULT))
    } else {
        Ok(SEARCH_MODEL_VERSIONS_DEFAULT)
    }
}

/// True when the request explicitly carried a `max_results` field (as a GET
/// query param or a JSON body key). Used only to decide whether the proto
/// default should be overridden by the store default.
fn request_has_max_results(parts: &Parts, body: &Bytes) -> Result<bool, MlflowError> {
    if parts.method == axum::http::Method::GET {
        if let Some(query) = parts.uri.query().filter(|q| !q.is_empty()) {
            return Ok(query.split('&').any(|pair| {
                let key = pair.split('=').next().unwrap_or("");
                key == "max_results"
            }));
        }
    }
    let text = std::str::from_utf8(body).unwrap_or("");
    if text.trim().is_empty() {
        return Ok(false);
    }
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::Object(map)) => Ok(map.contains_key("max_results")),
        _ => Ok(false),
    }
}

/// Map a store [`RegisteredModel`] entity to its proto. `user_id` is never
/// populated (§3.14). `latest_versions`, `tags`, and `aliases` are carried
/// through exactly as the store populated them.
fn to_proto_registered_model(model: RegisteredModel) -> pb::RegisteredModel {
    pb::RegisteredModel {
        name: Some(model.name),
        creation_timestamp: model.creation_timestamp,
        last_updated_timestamp: model.last_updated_timestamp,
        user_id: None,
        description: Some(model.description.unwrap_or_default()),
        latest_versions: model
            .latest_versions
            .into_iter()
            .map(to_proto_model_version)
            .collect(),
        tags: model.tags.into_iter().map(to_proto_rm_tag).collect(),
        aliases: model.aliases.into_iter().map(to_proto_rm_alias).collect(),
        deployment_job_id: None,
        deployment_job_state: None,
    }
}

/// Map a store [`ModelVersion`] entity to its proto. `version` is already a
/// String in the entity (int-in-DB → string-in-proto). `status` maps the string
/// stage name to the proto enum.
fn to_proto_model_version(mv: ModelVersion) -> pb::ModelVersion {
    pb::ModelVersion {
        name: Some(mv.name),
        version: Some(mv.version),
        creation_timestamp: mv.creation_timestamp,
        last_updated_timestamp: mv.last_updated_timestamp,
        user_id: mv.user_id,
        current_stage: Some(mv.current_stage.unwrap_or_default()),
        description: Some(mv.description.unwrap_or_default()),
        source: Some(mv.source.unwrap_or_default()),
        run_id: Some(mv.run_id.unwrap_or_default()),
        status: mv.status.as_deref().map(status_to_proto),
        status_message: mv.status_message,
        tags: mv.tags.into_iter().map(to_proto_mv_tag).collect(),
        run_link: Some(mv.run_link.unwrap_or_default()),
        aliases: mv.aliases,
        model_id: None,
        model_params: Vec::new(),
        model_metrics: Vec::new(),
        deployment_job_state: None,
    }
}

/// Map a status string (`"READY"`, ...) to the `ModelVersionStatus` proto enum
/// value. Unknown values default to `READY` — the OSS store only ever writes
/// `READY`, so this only guards against corrupt rows.
fn status_to_proto(status: &str) -> i32 {
    pb::ModelVersionStatus::from_str_name(status).unwrap_or(pb::ModelVersionStatus::Ready) as i32
}

fn to_proto_rm_tag(tag: RegisteredModelTag) -> pb::RegisteredModelTag {
    pb::RegisteredModelTag {
        key: Some(tag.key),
        value: Some(tag.value.unwrap_or_default()),
    }
}

fn to_proto_rm_alias(alias: RegisteredModelAlias) -> pb::RegisteredModelAlias {
    pb::RegisteredModelAlias {
        alias: Some(alias.alias),
        version: Some(alias.version),
    }
}

fn to_proto_mv_tag(tag: ModelVersionTag) -> pb::ModelVersionTag {
    pb::ModelVersionTag {
        key: Some(tag.key),
        value: Some(tag.value.unwrap_or_default()),
    }
}
