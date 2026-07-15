//! Model-version stage constants and canonicalization, mirroring
//! `mlflow/entities/model_registry/model_version_stages.py`.

use mlflow_error::MlflowError;

pub const STAGE_NONE: &str = "None";
pub const STAGE_STAGING: &str = "Staging";
pub const STAGE_PRODUCTION: &str = "Production";
pub const STAGE_ARCHIVED: &str = "Archived";
pub const STAGE_DELETED_INTERNAL: &str = "Deleted_Internal";

/// `ALL_STAGES` — the user-facing stages (excludes `Deleted_Internal`).
pub const ALL_STAGES: &[&str] = &[STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED];

/// `DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS` — the "active" stages. Only these
/// are valid targets for `archive_existing_versions=true` in a stage transition.
pub const DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS: &[&str] = &[STAGE_STAGING, STAGE_PRODUCTION];

/// `get_canonical_stage`: case-insensitive match against [`ALL_STAGES`],
/// returning the canonical spelling. Errors `INVALID_PARAMETER_VALUE` on an
/// unknown stage, with the exact Python message.
pub fn get_canonical_stage(stage: &str) -> Result<&'static str, MlflowError> {
    let key = stage.to_lowercase();
    ALL_STAGES
        .iter()
        .find(|s| s.to_lowercase() == key)
        .copied()
        .ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid Model Version stage: {stage}. Value must be one of {}.",
                ALL_STAGES.join(", ")
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_case_insensitively() {
        assert_eq!(get_canonical_stage("production").unwrap(), "Production");
        assert_eq!(get_canonical_stage("STAGING").unwrap(), "Staging");
        assert_eq!(get_canonical_stage("None").unwrap(), "None");
    }

    #[test]
    fn rejects_unknown_stage() {
        let err = get_canonical_stage("bogus").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid Model Version stage: bogus. Value must be one of None, Staging, Production, Archived."
        );
    }
}
