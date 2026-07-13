//! Registry-specific input validation, mirroring the relevant functions in
//! `mlflow/utils/validation.py` exactly (messages, error codes, length caps).
//!
//! The path-uniqueness helpers (`posix_normpath`, `path_not_unique`,
//! `bad_path_message`) duplicate logic that also exists (crate-private) in
//! `mlflow-store::store::validation`. They are re-derived here rather than
//! shared because that module is not public; consolidation into a shared
//! validation crate is a later cleanup.

use mlflow_error::MlflowError;

/// `MAX_REGISTERED_MODEL_ALIAS_LENGTH`.
const MAX_REGISTERED_MODEL_ALIAS_LENGTH: usize = 255;
/// `MAX_MODEL_REGISTRY_TAG_KEY_LENGTH`.
const MAX_MODEL_REGISTRY_TAG_KEY_LENGTH: usize = 250;
/// `MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH`.
const MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH: usize = 100_000;

/// `_BAD_ALIAS_CHARACTERS_MESSAGE`.
const BAD_ALIAS_CHARACTERS_MESSAGE: &str =
    "Names may only contain alphanumerics, underscores (_), and dashes (-).";

/// `PROMPT_TEXT_TAG_KEY` — mirrors
/// `mlflow.entities.model_registry.prompt_version.PROMPT_TEXT_TAG_KEY`.
const PROMPT_TEXT_TAG_KEY: &str = "mlflow.prompt.text";

// ---------------------------------------------------------------------------
// Shared message builders (mirroring mlflow/utils/validation.py)
// ---------------------------------------------------------------------------

/// `exceeds_maximum_length(path, limit)`.
fn exceeds_maximum_length(path: &str, limit: usize) -> String {
    format!("'{path}' exceeds the maximum length of {limit} characters")
}

/// `missing_value(path)`.
fn missing_value(path: &str) -> String {
    format!("Missing value for required parameter '{path}'.")
}

/// `not_integer_value(path, value)`.
fn not_integer_value(path: &str, value: &str) -> String {
    format!("Parameter '{path}' must be an integer, got '{value}'.")
}

/// `invalid_value(path, value, message)`. `value` is JSON-encoded with sorted
/// keys and compact separators; for the string values here that reduces to a
/// JSON string literal.
fn invalid_value(path: &str, value: &str, message: Option<&str>) -> String {
    let formatted = json_string(value);
    match message {
        Some(m) => format!("Invalid value {formatted} for parameter '{path}' supplied: {m}"),
        None => format!("Invalid value {formatted} for parameter '{path}' supplied."),
    }
}

/// Encode a string as a JSON string literal, matching Python `json.dumps`.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `bad_character_message()` (non-windows).
fn bad_character_message() -> &'static str {
    "Names may only contain alphanumerics, underscores (_), dashes (-), periods (.), \
     spaces ( ), colon(:) and slashes (/)."
}

/// `bad_path_message(name)`.
fn bad_path_message(name: &str) -> String {
    format!(
        "Names may be treated as files in certain cases, and must not resolve to other names \
         when treated as such. This name would resolve to {:?}",
        posix_normpath(name)
    )
}

/// `validate_param_and_metric_name`: regex `^[/\w.\- :]*$` with Unicode `\w`.
fn valid_param_and_metric_name(name: &str) -> bool {
    name.chars().all(|c| {
        c == '/' || c == '.' || c == '-' || c == ' ' || c == ':' || c == '_' || c.is_alphanumeric()
    })
}

/// `path_not_unique(name)`.
fn path_not_unique(name: &str) -> bool {
    let norm = posix_normpath(name);
    norm != name || norm == "." || norm.starts_with("..") || norm.starts_with('/')
}

/// A faithful port of `posixpath.normpath` for the inputs seen here.
fn posix_normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_string();
    }
    let is_abs = path.starts_with('/');
    let leading = if path.starts_with("//") && !path.starts_with("///") {
        "//"
    } else if is_abs {
        "/"
    } else {
        ""
    };
    let mut comps: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".." || (!is_abs && comps.is_empty()) || comps.last().is_some_and(|c| *c == "..")
        {
            comps.push(comp);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let joined = comps.join("/");
    let result = format!("{leading}{joined}");
    if result.is_empty() {
        ".".to_string()
    } else {
        result
    }
}

/// `_validate_length_limit(entity_name, limit, value)`.
fn validate_length_limit(entity_name: &str, limit: usize, value: &str) -> Result<(), MlflowError> {
    if value.chars().count() <= limit {
        Ok(())
    } else {
        Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length(entity_name, limit),
        ))
    }
}

// ---------------------------------------------------------------------------
// Registry validators
// ---------------------------------------------------------------------------

/// `_validate_model_name`.
pub(crate) fn validate_model_name(name: &str) -> Result<(), MlflowError> {
    if name.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(missing_value("name")));
    }
    if name.contains('/') || name.contains(':') {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid model name '{name}'. Names cannot contain '/' or ':'."
        )));
    }
    if path_not_unique(name) {
        return Err(MlflowError::invalid_parameter_value(invalid_value(
            "name",
            name,
            Some(&bad_path_message(name)),
        )));
    }
    Ok(())
}

/// `_validate_model_renaming`.
pub(crate) fn validate_model_renaming(new_name: &str) -> Result<(), MlflowError> {
    if new_name.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(missing_value(
            "new_name",
        )));
    }
    validate_model_name(new_name)
}

/// `_validate_model_version`: the version must parse as an integer.
pub(crate) fn validate_model_version(version: &str) -> Result<(), MlflowError> {
    if version.parse::<i64>().is_err() {
        return Err(MlflowError::invalid_parameter_value(not_integer_value(
            "version", version,
        )));
    }
    Ok(())
}

/// `_validate_tag_name`.
fn validate_tag_name(name: &str) -> Result<(), MlflowError> {
    if !valid_param_and_metric_name(name) {
        return Err(MlflowError::invalid_parameter_value(invalid_value(
            "key",
            name,
            Some(bad_character_message()),
        )));
    }
    if path_not_unique(name) {
        return Err(MlflowError::invalid_parameter_value(invalid_value(
            "key",
            name,
            Some(&bad_path_message(name)),
        )));
    }
    Ok(())
}

/// `_validate_registered_model_tag`.
pub(crate) fn validate_registered_model_tag(key: &str, value: &str) -> Result<(), MlflowError> {
    validate_tag_name(key)?;
    validate_length_limit("key", MAX_MODEL_REGISTRY_TAG_KEY_LENGTH, key)?;
    validate_length_limit("value", MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH, value)?;
    Ok(())
}

/// `_validate_model_version_tag`.
pub(crate) fn validate_model_version_tag(key: &str, value: &str) -> Result<(), MlflowError> {
    validate_tag_name(key)?;
    validate_length_limit("key", MAX_MODEL_REGISTRY_TAG_KEY_LENGTH, key)?;
    if key == PROMPT_TEXT_TAG_KEY && value.chars().count() > MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Prompt text exceeds max length of {MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH} characters."
        )));
    }
    validate_length_limit("value", MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH, value)?;
    Ok(())
}

/// `_validate_tag_name` used directly by delete-tag paths (`delete_*_tag`).
pub(crate) fn validate_tag_key(key: &str) -> Result<(), MlflowError> {
    validate_tag_name(key)
}

/// `_validate_model_alias_name`: non-empty, matches `^[\w\-]*$`, length cap.
pub(crate) fn validate_model_alias_name(alias: &str) -> Result<(), MlflowError> {
    if alias.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "Registered model alias name cannot be empty.",
        ));
    }
    // `^[\w\-]*$`: word chars (Unicode) or dashes.
    if !alias
        .chars()
        .all(|c| c == '-' || c == '_' || c.is_alphanumeric())
    {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid alias name: '{alias}'. {BAD_ALIAS_CHARACTERS_MESSAGE}"
        )));
    }
    validate_length_limit(
        "Registered model alias name",
        MAX_REGISTERED_MODEL_ALIAS_LENGTH,
        alias,
    )?;
    Ok(())
}

/// `_validate_model_alias_name_reserved`: rejects `latest` (case-insensitive)
/// and version-style aliases (`^[vV]\d+$`).
pub(crate) fn validate_model_alias_name_reserved(alias: &str) -> Result<(), MlflowError> {
    if alias.to_lowercase() == "latest" {
        return Err(MlflowError::invalid_parameter_value(
            "'latest' alias name (case insensitive) is reserved.",
        ));
    }
    if is_version_alias(alias) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Version alias name '{alias}' is reserved."
        )));
    }
    Ok(())
}

/// `^[vV]\d+$`: a `v`/`V` followed by one or more digits.
fn is_version_alias(alias: &str) -> bool {
    let mut chars = alias.chars();
    match chars.next() {
        Some('v') | Some('V') => {}
        _ => return false,
    }
    let rest: String = chars.collect();
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_empty_and_bad_chars() {
        assert_eq!(
            validate_model_name("").unwrap_err().message,
            "Missing value for required parameter 'name'."
        );
        assert_eq!(
            validate_model_name("a/b").unwrap_err().message,
            "Invalid model name 'a/b'. Names cannot contain '/' or ':'."
        );
        assert_eq!(
            validate_model_name("a:b").unwrap_err().message,
            "Invalid model name 'a:b'. Names cannot contain '/' or ':'."
        );
        assert!(validate_model_name("my-model_1.2").is_ok());
    }

    #[test]
    fn model_version_must_be_integer() {
        assert_eq!(
            validate_model_version("abc").unwrap_err().message,
            "Parameter 'version' must be an integer, got 'abc'."
        );
        assert!(validate_model_version("3").is_ok());
    }

    #[test]
    fn alias_char_and_reserved_rules() {
        assert_eq!(
            validate_model_alias_name("").unwrap_err().message,
            "Registered model alias name cannot be empty."
        );
        assert!(validate_model_alias_name("champion").is_ok());
        assert!(validate_model_alias_name("has space").is_err());
        assert_eq!(
            validate_model_alias_name_reserved("latest")
                .unwrap_err()
                .message,
            "'latest' alias name (case insensitive) is reserved."
        );
        assert_eq!(
            validate_model_alias_name_reserved("v3")
                .unwrap_err()
                .message,
            "Version alias name 'v3' is reserved."
        );
        assert!(validate_model_alias_name_reserved("V12").is_err());
        assert!(validate_model_alias_name_reserved("champion").is_ok());
    }

    #[test]
    fn tag_length_caps() {
        let long_key = "k".repeat(MAX_MODEL_REGISTRY_TAG_KEY_LENGTH + 1);
        assert_eq!(
            validate_registered_model_tag(&long_key, "v")
                .unwrap_err()
                .message,
            "'key' exceeds the maximum length of 250 characters"
        );
        assert!(validate_registered_model_tag("k", "v").is_ok());
    }
}
