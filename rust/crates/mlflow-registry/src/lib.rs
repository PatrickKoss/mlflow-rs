//! `mlflow-registry`: the model registry store.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§3.14, §5.1, Phase 7), this crate owns
//! registered models, model versions, tags, and aliases, all with
//! **workspace-leading composite primary keys**.
//!
//! ## What T7.1 implements
//!
//! * [`schema`] — data structs for the 5 registry tables.
//! * [`entities`] — the owned `RegisteredModel` / `ModelVersion` entities
//!   returned by the store (proto-shaped: `version` is a `String`, `user_id`
//!   is absent on `RegisteredModel`, aliases populated separately).
//! * [`RegistryStore`] — registered-model CRUD incl. the rename cascade (FK
//!   `ON UPDATE CASCADE`), `get_latest_versions` (ROW_NUMBER window,
//!   READY-only per stage, `Deleted_Internal` excluded), registered-model and
//!   model-version tag set/delete (upsert + no-op delete), and alias
//!   set/delete/get-by-alias — every method workspace-scoped.
//! * A **minimal** `create_model_version` (plain-DB `MAX(version)+1` insert, no
//!   `models:/` source resolution) needed to exercise latest-versions and
//!   aliases in tests.
//!
//! ## Boundary with T7.2
//!
//! `models:/`/`runs:/` source resolution, stage transitions, soft-delete
//! redaction, `update_model_version`, and registry **search** (with the
//! prompt-exclusion anti-join) are T7.2/T7.3 — intentionally absent here.
//!
//! ## Reuse of `mlflow-store`
//!
//! This crate depends on `mlflow-store` for its connection/dialect
//! infrastructure ([`mlflow_store::Db`], [`mlflow_store::Dialect`]). The
//! query-execution helpers there (`Val`/`Tx`/`RowLike` and the `Db` query
//! methods) are `pub(crate)`, so a small behaviorally-identical copy lives in
//! [`dbutil`] pending consolidation (see that module's docs).

mod dbutil;
pub mod entities;
pub mod schema;
mod stages;
mod store;
mod validation;

pub use entities::{
    ModelVersion, ModelVersionTag, RegisteredModel, RegisteredModelAlias, RegisteredModelTag,
};
pub use stages::{ALL_STAGES, STAGE_ARCHIVED, STAGE_NONE, STAGE_PRODUCTION, STAGE_STAGING};
pub use store::RegistryStore;
