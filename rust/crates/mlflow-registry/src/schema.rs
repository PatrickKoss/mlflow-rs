//! Rust structs mirroring the 5 model-registry tables (plan §5.1).
//!
//! Source of truth: `mlflow/store/model_registry/dbmodels/models.py`. Each
//! struct mirrors the SQLAlchemy column names, types, and nullability exactly.
//! All five tables have **workspace-leading composite primary keys** (§3.14).
//!
//! Type mapping:
//!
//! | SQLAlchemy | Rust |
//! |---|---|
//! | `String(N)` / `Text` | `String` (or `Option<String>` when nullable) |
//! | `BigInteger` (timestamps) | `Option<i64>` (nullable) |
//! | `Integer` (`version`) | `i64` at the DB boundary, string in the proto |
//!
//! Note on `version`: the physical column is `Integer` (DB-only), but the
//! MLflow proto/entity carries it as a *string*. The store converts at the
//! entity boundary — [`crate::entities::ModelVersion::version`] is a `String`.
//!
//! Note on `storage_location`: it is a DB-only column (the resolved artifact
//! path), distinct from the proto `source`. It is not returned on the entity.

pub const REGISTERED_MODELS: &str = "registered_models";
pub const MODEL_VERSIONS: &str = "model_versions";
pub const REGISTERED_MODEL_TAGS: &str = "registered_model_tags";
pub const MODEL_VERSION_TAGS: &str = "model_version_tags";
pub const REGISTERED_MODEL_ALIASES: &str = "registered_model_aliases";

/// All model-registry table names owned by the Rust store (plan §5.1).
pub const REGISTRY_TABLES: &[&str] = &[
    REGISTERED_MODELS,
    MODEL_VERSIONS,
    REGISTERED_MODEL_TAGS,
    MODEL_VERSION_TAGS,
    REGISTERED_MODEL_ALIASES,
];

/// Row of the `registered_models` table (`SqlRegisteredModel`).
/// PK `(workspace, name)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredModel {
    pub workspace: String,
    pub name: String,
    pub creation_time: Option<i64>,
    pub last_updated_time: Option<i64>,
    pub description: Option<String>,
}

/// Row of the `model_versions` table (`SqlModelVersion`).
/// PK `(workspace, name, version)`; FK `(workspace, name)` → `registered_models`
/// `ON UPDATE CASCADE`.
///
/// `version` is a physical `Integer`. `storage_location` is DB-only (the
/// resolved artifact path). `current_stage` defaults to `"None"`, `status`
/// defaults to `"READY"`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelVersion {
    pub workspace: String,
    pub name: String,
    pub version: i64,
    pub creation_time: Option<i64>,
    pub last_updated_time: Option<i64>,
    pub description: Option<String>,
    pub user_id: Option<String>,
    pub current_stage: Option<String>,
    pub source: Option<String>,
    pub storage_location: Option<String>,
    pub run_id: Option<String>,
    pub run_link: Option<String>,
    pub status: Option<String>,
    pub status_message: Option<String>,
}

/// Row of the `registered_model_tags` table (`SqlRegisteredModelTag`).
/// PK `(workspace, key, name)`; FK `(workspace, name)` `ON UPDATE CASCADE`.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredModelTag {
    pub workspace: String,
    pub name: String,
    pub key: String,
    pub value: Option<String>,
}

/// Row of the `model_version_tags` table (`SqlModelVersionTag`).
/// PK `(workspace, key, name, version)`; FK `(workspace, name, version)`
/// `ON UPDATE CASCADE`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelVersionTag {
    pub workspace: String,
    pub name: String,
    pub version: i64,
    pub key: String,
    pub value: Option<String>,
}

/// Row of the `registered_model_aliases` table (`SqlRegisteredModelAlias`).
/// PK `(workspace, name, alias)`; FK `(workspace, name)` `ON UPDATE CASCADE`
/// **and** `ON DELETE CASCADE` (so deleting a registered model removes its
/// aliases at the DB level).
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredModelAlias {
    pub workspace: String,
    pub name: String,
    pub alias: String,
    pub version: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_table_count() {
        assert_eq!(REGISTRY_TABLES.len(), 5);
    }
}
