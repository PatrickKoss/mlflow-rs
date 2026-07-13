//! Input validation, mirroring `mlflow/utils/validation.py` exactly (messages,
//! error codes, and length caps). Only the pieces the T2.4/T2.5 store path
//! touches are ported here; the rest arrive with later tasks.

use mlflow_error::MlflowError;

// Length caps (verbatim from `mlflow/utils/validation.py`).
pub(crate) const MAX_PARAMS_TAGS_PER_BATCH: usize = 100;
pub(crate) const MAX_METRICS_PER_BATCH: usize = 1000;
pub(crate) const MAX_ENTITIES_PER_BATCH: usize = 1000;
pub(crate) const MAX_PARAM_VAL_LENGTH: usize = 6000;
pub(crate) const MAX_TAG_VAL_LENGTH: usize = 8000;
pub(crate) const MAX_EXPERIMENT_NAME_LENGTH: usize = 500;
pub(crate) const MAX_EXPERIMENT_TAG_KEY_LENGTH: usize = 250;
pub(crate) const MAX_EXPERIMENT_TAG_VAL_LENGTH: usize = 5000;
pub(crate) const MAX_ENTITY_KEY_LENGTH: usize = 250;

/// `mlflow/utils/validation.py::exceeds_maximum_length`.
fn exceeds_maximum_length(path: &str, limit: usize) -> String {
    format!("'{path}' exceeds the maximum length of {limit} characters")
}

/// `invalid_value(path, value, message)`. `value` is JSON-encoded with sorted
/// keys and compact separators (`json.dumps(value, sort_keys=True,
/// separators=(",", ":"))`). For the scalar string/None values the store passes
/// here, that reduces to a JSON string literal or `null`.
fn invalid_value(path: &str, value: JsonValue<'_>, message: Option<&str>) -> String {
    let formatted = value.to_compact_json();
    match message {
        Some(m) => format!("Invalid value {formatted} for parameter '{path}' supplied: {m}"),
        None => format!("Invalid value {formatted} for parameter '{path}' supplied."),
    }
}

/// The subset of JSON values validation error messages need to render.
enum JsonValue<'a> {
    Str(&'a str),
}

impl JsonValue<'_> {
    fn to_compact_json(&self) -> String {
        match self {
            // Matches `json.dumps("<s>")` for a plain string.
            JsonValue::Str(s) => json_string(s),
        }
    }
}

/// Encode a string as a JSON string literal, matching Python's
/// `json.dumps("<s>")` (ensure_ascii defaults to True, but MLflow's
/// `invalid_value` passes `ensure_ascii` implicitly True for the message; here
/// names are ASCII in practice, and we escape the JSON-significant characters).
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

/// `bad_character_message()` (non-windows branch — the server target platform).
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

/// `validate_param_and_metric_name` (non-windows): only slashes, alphanumerics,
/// underscores, periods, dashes, colons, and spaces. Regex `^[/\w.\- :]*$`;
/// `\w` is `[A-Za-z0-9_]` plus Unicode word chars — Python's `re` with a `str`
/// treats `\w` as Unicode by default, so we accept any alphanumeric.
fn valid_param_and_metric_name(name: &str) -> bool {
    name.chars().all(|c| {
        c == '/' || c == '.' || c == '-' || c == ' ' || c == ':' || c == '_' || c.is_alphanumeric()
    })
}

/// `path_not_unique(name)`: the normalized POSIX path differs from the input, or
/// is `.`, or escapes/roots out (`..`/`/` prefix).
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
    // POSIX special case: exactly two leading slashes are preserved.
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

/// `_validate_length_limit` (non-truncating form — the store validates lengths
/// after any client-side truncation has happened).
fn validate_length_limit(entity_name: &str, limit: usize, value: &str) -> Result<(), MlflowError> {
    if value.chars().count() <= limit {
        Ok(())
    } else {
        Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length(entity_name, limit),
        ))
    }
}

/// `_validate_experiment_name`.
pub(crate) fn validate_experiment_name(name: &str) -> Result<(), MlflowError> {
    if name.is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid experiment name: '{name}'"
        )));
    }
    if name.chars().count() > MAX_EXPERIMENT_NAME_LENGTH {
        return Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length("name", MAX_EXPERIMENT_NAME_LENGTH),
        ));
    }
    Ok(())
}

/// `_validate_experiment_tag`.
pub(crate) fn validate_experiment_tag(key: &str, value: &str) -> Result<(), MlflowError> {
    validate_tag_name(key)?;
    validate_length_limit("key", MAX_EXPERIMENT_TAG_KEY_LENGTH, key)?;
    validate_length_limit("value", MAX_EXPERIMENT_TAG_VAL_LENGTH, value)?;
    Ok(())
}

/// `_validate_param_name` + length caps (`_validate_param` returns the
/// validated `Param`; here we just validate). Returns Ok on success.
pub(crate) fn validate_param(key: &str, value: &str) -> Result<(), MlflowError> {
    validate_param_name(key)?;
    validate_length_limit("Param key", MAX_ENTITY_KEY_LENGTH, key)?;
    validate_length_limit("Param value", MAX_PARAM_VAL_LENGTH, value)?;
    Ok(())
}

/// `_validate_tag` (key + value caps). `path` is the JSON path prefix used in
/// batch error messages (e.g. `tags[3]`); pass `None` for the single-tag path.
pub(crate) fn validate_tag(key: &str, value: &str, path: Option<&str>) -> Result<(), MlflowError> {
    let key_path = join_json_path(path, "key");
    let value_path = join_json_path(path, "value");
    validate_tag_name_with_path(key, &key_path)?;
    validate_length_limit(&key_path, MAX_ENTITY_KEY_LENGTH, key)?;
    validate_length_limit(&value_path, MAX_TAG_VAL_LENGTH, value)?;
    Ok(())
}

/// `_validate_metric` (name + numeric checks + name length). `path` is the batch
/// JSON path prefix (`metrics[3]`) or `None` for single-metric.
pub(crate) fn validate_metric(
    key: &str,
    value: f64,
    timestamp: i64,
    _step: i64,
    path: Option<&str>,
) -> Result<(), MlflowError> {
    validate_metric_name(key, &join_json_path(path, "name"))?;
    // value must be numeric (always true for an f64) — NaN/Inf are allowed and
    // sanitized downstream, matching Python (which only rejects non-`Number`s).
    if timestamp < 0 {
        let tpath = join_json_path(path, "timestamp");
        return Err(MlflowError::invalid_parameter_value(invalid_value(
            &tpath,
            JsonValue::Str(&timestamp.to_string()),
            Some(&format!(
                "metric '{key}' (value={value}). Timestamp must be a nonnegative long \
                 (64-bit integer) "
            )),
        )));
    }
    validate_length_limit("Metric name", MAX_ENTITY_KEY_LENGTH, key)?;
    Ok(())
}

pub(crate) const MAX_DATASET_NAME_SIZE: usize = 500;
pub(crate) const MAX_DATASET_DIGEST_SIZE: usize = 36;
pub(crate) const MAX_DATASET_SCHEMA_SIZE: usize = 1_048_575;
pub(crate) const MAX_DATASET_SOURCE_SIZE: usize = 65_535;
pub(crate) const MAX_DATASET_PROFILE_SIZE: usize = 16_777_215;
pub(crate) const MAX_INPUT_TAG_KEY_SIZE: usize = 255;
pub(crate) const MAX_INPUT_TAG_VALUE_SIZE: usize = 500;

/// `_validate_dataset`: length caps on a dataset's fields. `name`, `digest`, and
/// `source` are required (the entity layer guarantees they are non-null before
/// they reach here; the caller supplies owned strings).
pub(crate) fn validate_dataset(
    name: &str,
    digest: &str,
    source: &str,
    schema: Option<&str>,
    profile: Option<&str>,
) -> Result<(), MlflowError> {
    if name.chars().count() > MAX_DATASET_NAME_SIZE {
        return Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length("name", MAX_DATASET_NAME_SIZE),
        ));
    }
    if digest.chars().count() > MAX_DATASET_DIGEST_SIZE {
        return Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length("digest", MAX_DATASET_DIGEST_SIZE),
        ));
    }
    if source.chars().count() > MAX_DATASET_SOURCE_SIZE {
        return Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length("source", MAX_DATASET_SOURCE_SIZE),
        ));
    }
    if let Some(s) = schema {
        if s.chars().count() > MAX_DATASET_SCHEMA_SIZE {
            return Err(MlflowError::invalid_parameter_value(
                exceeds_maximum_length("schema", MAX_DATASET_SCHEMA_SIZE),
            ));
        }
    }
    if let Some(p) = profile {
        if p.chars().count() > MAX_DATASET_PROFILE_SIZE {
            return Err(MlflowError::invalid_parameter_value(
                exceeds_maximum_length("profile", MAX_DATASET_PROFILE_SIZE),
            ));
        }
    }
    Ok(())
}

/// `_validate_input_tag`: length caps on an input tag's key/value.
pub(crate) fn validate_input_tag(key: &str, value: &str) -> Result<(), MlflowError> {
    if key.chars().count() > MAX_INPUT_TAG_KEY_SIZE {
        return Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length("key", MAX_INPUT_TAG_KEY_SIZE),
        ));
    }
    if value.chars().count() > MAX_INPUT_TAG_VALUE_SIZE {
        return Err(MlflowError::invalid_parameter_value(
            exceeds_maximum_length("value", MAX_INPUT_TAG_VALUE_SIZE),
        ));
    }
    Ok(())
}

fn validate_param_name(name: &str) -> Result<(), MlflowError> {
    validate_name_common(name, "key")
}

fn validate_metric_name(name: &str, path: &str) -> Result<(), MlflowError> {
    validate_name_common(name, path)
}

fn validate_tag_name(name: &str) -> Result<(), MlflowError> {
    validate_tag_name_with_path(name, "key")
}

fn validate_tag_name_with_path(name: &str, path: &str) -> Result<(), MlflowError> {
    // `_validate_tag_name`: same char/path rules as params/metrics.
    validate_name_common(name, path)
}

/// The common name check shared by param/metric/tag names: character set and
/// path-uniqueness, with `invalid_value`-formatted messages.
fn validate_name_common(name: &str, path: &str) -> Result<(), MlflowError> {
    if !valid_param_and_metric_name(name) {
        return Err(MlflowError::invalid_parameter_value(invalid_value(
            path,
            JsonValue::Str(name),
            Some(bad_character_message()),
        )));
    }
    if path_not_unique(name) {
        return Err(MlflowError::invalid_parameter_value(invalid_value(
            path,
            JsonValue::Str(name),
            Some(&bad_path_message(name)),
        )));
    }
    Ok(())
}

/// `append_to_json_path(prefix, value)` for the leaf paths used here.
fn join_json_path(prefix: Option<&str>, leaf: &str) -> String {
    match prefix {
        Some(p) if !p.is_empty() => format!("{p}.{leaf}"),
        _ => leaf.to_string(),
    }
}

/// `_validate_batch_limit`.
fn validate_batch_limit(entity_name: &str, limit: usize, length: usize) -> Result<(), MlflowError> {
    if length > limit {
        return Err(MlflowError::invalid_parameter_value(format!(
            "A batch logging request can contain at most {limit} {entity_name}. Got {length} \
             {entity_name}. Please split up {entity_name} across multiple requests and try again."
        )));
    }
    Ok(())
}

/// `_validate_batch_log_limits`.
pub(crate) fn validate_batch_log_limits(
    n_metrics: usize,
    n_params: usize,
    n_tags: usize,
) -> Result<(), MlflowError> {
    validate_batch_limit("metrics", MAX_METRICS_PER_BATCH, n_metrics)?;
    validate_batch_limit("params", MAX_PARAMS_TAGS_PER_BATCH, n_params)?;
    validate_batch_limit("tags", MAX_PARAMS_TAGS_PER_BATCH, n_tags)?;
    validate_batch_limit(
        "metrics, params, and tags",
        MAX_ENTITIES_PER_BATCH,
        n_metrics + n_params + n_tags,
    )?;
    Ok(())
}

/// `_validate_param_keys_unique`: duplicate param keys in one batch are rejected.
pub(crate) fn validate_param_keys_unique(keys: &[&str]) -> Result<(), MlflowError> {
    let mut seen = Vec::new();
    let mut dupes = Vec::new();
    for k in keys {
        if seen.contains(k) {
            dupes.push(*k);
        } else {
            seen.push(*k);
        }
    }
    if !dupes.is_empty() {
        // Python renders the list as `['a', 'b']` (repr of a list of str).
        let rendered = format!(
            "[{}]",
            dupes
                .iter()
                .map(|k| format!("'{k}'"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Err(MlflowError::invalid_parameter_value(format!(
            "Duplicate parameter keys have been submitted: {rendered}. Please ensure the request \
             contains only one param value per param key."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_name_empty_and_too_long() {
        assert!(validate_experiment_name("").is_err());
        assert_eq!(
            validate_experiment_name("").unwrap_err().message,
            "Invalid experiment name: ''"
        );
        let long = "a".repeat(MAX_EXPERIMENT_NAME_LENGTH + 1);
        assert_eq!(
            validate_experiment_name(&long).unwrap_err().message,
            "'name' exceeds the maximum length of 500 characters"
        );
        assert!(validate_experiment_name("ok").is_ok());
    }

    #[test]
    fn param_value_length_cap() {
        let long = "x".repeat(MAX_PARAM_VAL_LENGTH + 1);
        let err = validate_param("k", &long).unwrap_err();
        assert_eq!(
            err.message,
            "'Param value' exceeds the maximum length of 6000 characters"
        );
        assert!(validate_param("k", &"x".repeat(MAX_PARAM_VAL_LENGTH)).is_ok());
    }

    #[test]
    fn bad_param_name_chars() {
        let err = validate_param("bad$name", "v").unwrap_err();
        assert!(err
            .message
            .contains("Invalid value \"bad$name\" for parameter 'key' supplied:"));
        assert!(err.message.contains("Names may only contain"));
    }

    #[test]
    fn path_traversal_name_rejected() {
        let err = validate_param("../escape", "v").unwrap_err();
        assert!(err.message.contains("would resolve to"));
    }

    #[test]
    fn normpath_matches_python() {
        assert_eq!(posix_normpath("a/b"), "a/b");
        assert_eq!(posix_normpath("a//b"), "a/b");
        assert_eq!(posix_normpath("a/./b"), "a/b");
        assert_eq!(posix_normpath("a/../b"), "b");
        assert_eq!(posix_normpath("../b"), "../b");
        assert_eq!(posix_normpath("/a/b"), "/a/b");
        assert_eq!(posix_normpath(""), ".");
        assert_eq!(posix_normpath("."), ".");
    }

    #[test]
    fn batch_limits() {
        assert!(validate_batch_log_limits(1000, 100, 100).is_err()); // total 1200 > 1000
        assert!(validate_batch_log_limits(500, 100, 100).is_ok());
        let err = validate_batch_log_limits(1001, 0, 0).unwrap_err();
        assert_eq!(
            err.message,
            "A batch logging request can contain at most 1000 metrics. Got 1001 metrics. \
             Please split up metrics across multiple requests and try again."
        );
        let err = validate_batch_log_limits(0, 101, 0).unwrap_err();
        assert!(err.message.contains("at most 100 params. Got 101 params"));
    }

    #[test]
    fn dup_param_keys() {
        let err = validate_param_keys_unique(&["a", "b", "a"]).unwrap_err();
        assert_eq!(
            err.message,
            "Duplicate parameter keys have been submitted: ['a']. Please ensure the request \
             contains only one param value per param key."
        );
        assert!(validate_param_keys_unique(&["a", "b"]).is_ok());
    }

    #[test]
    fn tag_value_length_cap_uses_path() {
        let long = "x".repeat(MAX_TAG_VAL_LENGTH + 1);
        let err = validate_tag("k", &long, Some("tags[2]")).unwrap_err();
        assert_eq!(
            err.message,
            "'tags[2].value' exceeds the maximum length of 8000 characters"
        );
    }

    #[test]
    fn negative_timestamp_rejected() {
        let err = validate_metric("acc", 0.5, -1, 0, None).unwrap_err();
        assert!(err.message.contains("Timestamp must be a nonnegative long"));
    }
}
