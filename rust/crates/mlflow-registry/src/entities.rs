//! Owned entity types returned by the registry store, mirroring
//! `SqlRegisteredModel.to_mlflow_entity` / `SqlModelVersion.to_mlflow_entity`.
//!
//! Like `mlflow-store`, this crate returns lightweight owned entities rather
//! than proto types; the HTTP layer (T7.4) maps them to protos. Field
//! semantics that matter for parity:
//!
//! * `RegisteredModel` carries `latest_versions`, `tags`, and `aliases`.
//!   `user_id` is intentionally **not** present (§3.14 — never returned on a
//!   registered model). Aliases are populated separately by the store (they are
//!   not a field on `SqlRegisteredModel.to_mlflow_entity`'s version list).
//! * `ModelVersion.version` is a **String** here even though the physical
//!   column is an `Integer` (§3.14) — matching the proto.
//! * `ModelVersion.aliases` is a list of alias strings, populated by the store
//!   from the parent registered model's aliases whose version matches.

/// A registered-model tag (`RegisteredModelTag`). Value may be `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredModelTag {
    pub key: String,
    pub value: Option<String>,
}

/// A registered-model alias (`RegisteredModelAlias`): alias name → version.
/// `version` is a String to match the proto.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredModelAlias {
    pub alias: String,
    pub version: String,
}

/// A model-version tag (`ModelVersionTag`). Value may be `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelVersionTag {
    pub key: String,
    pub value: Option<String>,
}

/// The `RegisteredModel` entity (`SqlRegisteredModel.to_mlflow_entity`).
///
/// `latest_versions` holds the READY... (all non-`Deleted_Internal`) latest
/// version per stage. `tags` and `aliases` are the model's tags/aliases.
/// `workspace` is carried through for workspace-aware callers.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredModel {
    pub name: String,
    pub creation_timestamp: Option<i64>,
    pub last_updated_timestamp: Option<i64>,
    pub description: Option<String>,
    pub latest_versions: Vec<ModelVersion>,
    pub tags: Vec<RegisteredModelTag>,
    pub aliases: Vec<RegisteredModelAlias>,
    pub workspace: String,
}

/// The `ModelVersion` entity (`SqlModelVersion.to_mlflow_entity`).
///
/// `version` is a String (proto semantics). `aliases` is a list of alias names
/// pointing at this version, populated by the store.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelVersion {
    pub name: String,
    pub version: String,
    pub creation_timestamp: Option<i64>,
    pub last_updated_timestamp: Option<i64>,
    pub description: Option<String>,
    pub user_id: Option<String>,
    pub current_stage: Option<String>,
    pub source: Option<String>,
    pub run_id: Option<String>,
    pub status: Option<String>,
    pub status_message: Option<String>,
    pub tags: Vec<ModelVersionTag>,
    pub run_link: Option<String>,
    pub aliases: Vec<String>,
    pub workspace: String,
}
