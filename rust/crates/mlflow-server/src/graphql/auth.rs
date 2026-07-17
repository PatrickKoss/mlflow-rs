//! GraphQL per-field authorization (plan T9.6, §3.16), a faithful port of
//! `mlflow/server/auth/__init__.py::GraphQLAuthorizationMiddleware`.
//!
//! ## Where this sits
//!
//! The tower auth middleware (T9.4) authenticates every request, including
//! `/graphql`, but dispatches it as `Dispatched::Allow` (no REST validator) —
//! exactly Python's before-request treatment: `/graphql` is a protected route
//! (401 when unauthenticated), yet `_find_validator` returns `None`, so any
//! authenticated user passes the request-level gate. Fine-grained authorization
//! then happens *inside* GraphQL execution, in the graphene
//! `GraphQLAuthorizationMiddleware.resolve`. This module is the Rust analogue:
//! the `/graphql` handler builds a [`GraphQlAuthGate`] from the authenticated
//! identity ([`crate::auth_middleware::AuthContext`], stamped onto the request
//! extensions by T9.4) and consults it per protected root field.
//!
//! ## Toggle (`MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH`)
//!
//! `get_graphql_authorization_middleware` returns `[]` when the toggle is off
//! *or* auth is not enabled — i.e. no per-field checks at all. We mirror both:
//! [`GraphQlAuthGate::for_request`] yields `None` (the gate is a no-op) when the
//! env toggle is off or there is no [`AuthContext`] (auth app disabled). The env
//! var defaults to **`True`** (`environment_variables.py:1527`), so an unset
//! variable enables the checks; only `"false"`/`"0"` disable them.
//!
//! ## Semantics (byte-for-byte with graphene)
//!
//! * **Admin bypass.** An admin's gate authorizes every field
//!   (`store.get_user(username).is_admin` short-circuit).
//! * **Protected fields** (`PROTECTED_FIELDS`): `mlflowGetExperiment`,
//!   `mlflowGetRun`, `mlflowListArtifacts`, `mlflowGetMetricHistoryBulkInterval`,
//!   `mlflowSearchRuns`, `mlflowSearchDatasets`, `mlflowSearchModelVersions`.
//!   Every other field (e.g. `test`) is unguarded.
//! * **Denied field → `null`, no error.** `resolve` returns `None` for a denied
//!   protected field; graphene serializes that as `null` in `data` with no entry
//!   in `errors`. The executor's `Ok(None)` outcome reproduces this. A
//!   permission-resolution *exception* (`MlflowException`) is also swallowed to
//!   `None` (`except MlflowException: return None`).
//! * **searchRuns / searchDatasets narrowing.** The requested `experiment_ids`
//!   are filtered to the readable ones; if *none* are readable the field is
//!   denied (`null`), otherwise the resolver runs against the narrowed list.
//! * **searchModelVersions post-filter.** After resolution, model versions the
//!   user can't read are dropped from the result in place (a simple drop — no
//!   page-fill, matching `_filter_model_versions_result`).

use mlflow_auth::AuthStore;
use mlflow_error::MlflowError;
use serde_json::Value;

use crate::auth_middleware::validators::{
    resolve_experiment_permission, resolve_registered_model_permission,
};
use crate::auth_middleware::AuthContext;
use crate::state::AppState;
use crate::workspace::Workspace;

use super::value::GqlVal;

/// `MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH` (`environment_variables.py:1527`,
/// default `True`): truthy unless explicitly `"false"`/`"0"`.
fn graphql_auth_enabled() -> bool {
    match std::env::var("MLFLOW_SERVER_ENABLE_GRAPHQL_AUTH") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "false" | "0"),
        Err(_) => true,
    }
}

/// `GraphQLAuthorizationMiddleware.PROTECTED_FIELDS`.
fn is_protected_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "mlflowGetExperiment"
            | "mlflowGetRun"
            | "mlflowListArtifacts"
            | "mlflowGetMetricHistoryBulkInterval"
            | "mlflowSearchRuns"
            | "mlflowSearchDatasets"
            | "mlflowSearchModelVersions"
    )
}

/// The outcome of the pre-resolution authorization check for a root field.
pub enum FieldAuth {
    /// Authorized — resolve the field (the `input` may have been narrowed).
    Allow,
    /// Denied — the field serializes as `null` with no error (graphene's
    /// `resolve` returning `None`).
    Deny,
}

/// The per-request GraphQL authorization gate. Holds the authenticated identity
/// and the stores needed to resolve experiment / registered-model READ
/// permissions the same way the T9.4 before-request validators do.
pub struct GraphQlAuthGate<'a> {
    state: &'a AppState,
    workspace: &'a Workspace,
    auth_store: &'a AuthStore,
    username: String,
    is_admin: bool,
}

impl<'a> GraphQlAuthGate<'a> {
    /// Build the gate for the current request, or `None` when per-field auth is
    /// inactive: the toggle is off, the auth app is disabled (no [`AuthStore`]),
    /// or there is no authenticated [`AuthContext`]. `None` means "resolve every
    /// field unguarded", matching `get_graphql_authorization_middleware` → `[]`.
    pub fn for_request(
        state: &'a AppState,
        workspace: &'a Workspace,
        auth_context: Option<&AuthContext>,
    ) -> Option<Self> {
        if !graphql_auth_enabled() {
            return None;
        }
        let auth_store = state.auth_store()?;
        let ctx = auth_context?;
        Some(Self {
            state,
            workspace,
            auth_store,
            username: ctx.username.clone(),
            is_admin: ctx.is_admin,
        })
    }

    /// `GraphQLAuthorizationMiddleware.resolve` pre-resolution half
    /// (`_check_authorization`). Admins authorize everything; unprotected fields
    /// pass; protected fields consult per-resource READ. A permission-resolution
    /// error is swallowed to [`FieldAuth::Deny`] (`except MlflowException:
    /// return None`). `searchRuns`/`searchDatasets` narrow `experiment_ids` in
    /// place on the `input`.
    pub async fn authorize_field(
        &self,
        field_name: &str,
        input: &mut serde_json::Map<String, Value>,
    ) -> FieldAuth {
        if self.is_admin || !is_protected_field(field_name) {
            return FieldAuth::Allow;
        }
        match self.check_authorization(field_name, input).await {
            Ok(allowed) => allowed,
            Err(_) => FieldAuth::Deny,
        }
    }

    /// `_check_authorization`: `None`-input (no resource) authorizes; otherwise
    /// dispatch on the field.
    async fn check_authorization(
        &self,
        field_name: &str,
        input: &mut serde_json::Map<String, Value>,
    ) -> Result<FieldAuth, MlflowError> {
        // graphene guards on `args.get("input") is None`. Our executor flattens
        // the `input` object into the resolver map, so "no input" is an empty map.
        if input.is_empty() {
            return Ok(FieldAuth::Allow);
        }

        match field_name {
            "mlflowGetExperiment" => {
                // `experiment_id := getattr(input_obj, "experiment_id", None)`:
                // only checks when the (truthy) id is present.
                match nonempty_str(input, "experimentId") {
                    Some(experiment_id) => Ok(self.experiment_read(&experiment_id).await?.into()),
                    None => Ok(FieldAuth::Allow),
                }
            }
            "mlflowGetRun" | "mlflowListArtifacts" => {
                // `run_id or run_uuid`, then check the run's experiment.
                match nonempty_str(input, "runId").or_else(|| nonempty_str(input, "runUuid")) {
                    Some(run_id) => Ok(self.run_read(&run_id).await?.into()),
                    None => Ok(FieldAuth::Allow),
                }
            }
            "mlflowGetMetricHistoryBulkInterval" => {
                for run_id in string_list(input, "runIds") {
                    if !self.run_read(&run_id).await? {
                        return Ok(FieldAuth::Deny);
                    }
                }
                Ok(FieldAuth::Allow)
            }
            "mlflowSearchRuns" | "mlflowSearchDatasets" => self.narrow_experiment_ids(input).await,
            // The post-filter case (`mlflowSearchModelVersions`) authorizes
            // pre-resolution; filtering happens in `post_resolve`.
            _ => Ok(FieldAuth::Allow),
        }
    }

    /// searchRuns / searchDatasets: filter `experimentIds` to readable ids. An
    /// empty requested list authorizes (nothing to narrow — matches graphene's
    /// `if experiment_ids :=` truthiness guard). A non-empty list with no
    /// readable ids denies; otherwise the input is rewritten to the readable
    /// subset.
    async fn narrow_experiment_ids(
        &self,
        input: &mut serde_json::Map<String, Value>,
    ) -> Result<FieldAuth, MlflowError> {
        let requested = string_list(input, "experimentIds");
        if requested.is_empty() {
            return Ok(FieldAuth::Allow);
        }
        let mut readable = Vec::new();
        for eid in requested {
            if self.experiment_read(&eid).await? {
                readable.push(Value::String(eid));
            }
        }
        if readable.is_empty() {
            return Ok(FieldAuth::Deny);
        }
        input.insert("experimentIds".to_string(), Value::Array(readable));
        Ok(FieldAuth::Allow)
    }

    /// `GraphQLAuthorizationMiddleware._post_resolve` — only
    /// `mlflowSearchModelVersions` filters its result: drop model versions the
    /// user can't read (a plain drop, no page-fill).
    pub async fn post_resolve(&self, field_name: &str, value: &mut GqlVal) {
        if self.is_admin || field_name != "mlflowSearchModelVersions" {
            return;
        }
        let Some(GqlVal::List(mvs)) = value.get_mut("modelVersions") else {
            return;
        };
        let mut kept = Vec::with_capacity(mvs.len());
        for mv in std::mem::take(mvs) {
            let name = mv
                .get("name")
                .and_then(|n| match n {
                    GqlVal::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            // A resolution error drops the row (fail-closed); the Python
            // predicate can't raise here (grants are already loaded), so this
            // only guards the store-less edge and matches "user can't read".
            if self.model_read(&name).await.unwrap_or(false) {
                kept.push(mv);
            }
        }
        *mvs = kept;
    }

    // ---- READ resolvers (mirror the `_graphql_can_read_*` helpers) ----

    async fn experiment_read(&self, experiment_id: &str) -> Result<bool, MlflowError> {
        Ok(resolve_experiment_permission(
            self.auth_store,
            &self.username,
            self.workspace.name(),
            experiment_id,
        )
        .await?
        .can_read)
    }

    /// `_graphql_can_read_run`: resolve the run, then check its experiment.
    async fn run_read(&self, run_id: &str) -> Result<bool, MlflowError> {
        let run = self
            .state
            .tracking_store()
            .get_run(self.workspace.name(), run_id)
            .await?;
        self.experiment_read(&run.info.experiment_id).await
    }

    async fn model_read(&self, model_name: &str) -> Result<bool, MlflowError> {
        Ok(resolve_registered_model_permission(
            self.auth_store,
            &self.username,
            self.workspace.name(),
            model_name,
        )
        .await?
        .can_read)
    }
}

impl From<bool> for FieldAuth {
    fn from(allowed: bool) -> Self {
        if allowed {
            FieldAuth::Allow
        } else {
            FieldAuth::Deny
        }
    }
}

/// A truthy (present, non-empty) string field on the resolver input.
fn nonempty_str(input: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// A `[String]` input argument as a `Vec<String>` (absent → empty).
fn string_list(input: &serde_json::Map<String, Value>, key: &str) -> Vec<String> {
    input
        .get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}
