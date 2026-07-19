//! Server-owned trace-archival configuration.
//!
//! This is a native port of `mlflow/tracing/trace_archival_config.py`.  The
//! public [`TraceArchivalConfigProvider`] is the hand-off point for the store
//! and scheduler tasks: it owns the configured path and a process-local,
//! thread-safe five-second cache that reloads the YAML on demand.

use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value};

const TRACE_ARCHIVAL_CONFIG_KEY: &str = "trace_archival";
const INTERVAL_SECONDS_DEFAULT: u64 = 300;
const INTERVAL_SECONDS_MAX: u64 = 86_400;
pub const TRACE_ARCHIVAL_CONFIG_CACHE_TTL: Duration = Duration::from_secs(5);

/// Parsed `trace_archival` YAML settings.
///
/// Field types and defaults match `TraceArchivalServerConfig` in
/// `mlflow/tracing/trace_archival_config.py`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceArchivalServerConfig {
    pub enabled: bool,
    pub location: String,
    pub retention: String,
    pub long_retention_allowlist: Vec<String>,
    pub interval_seconds: u64,
    pub max_traces_per_pass: Option<u64>,
}

/// A Python-compatible trace-archival configuration error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TraceArchivalConfigError {
    message: String,
}

impl TraceArchivalConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Injectable monotonic clock used by the five-second config cache.
///
/// The public trait lets scheduler tests advance time deterministically without
/// sleeping. Production uses [`SystemMonotonicClock`].
pub trait TraceArchivalConfigClock: Send + Sync {
    fn now(&self) -> Duration;
}

/// Process-monotonic production clock.
#[derive(Debug, Default)]
pub struct SystemMonotonicClock;

impl TraceArchivalConfigClock for SystemMonotonicClock {
    fn now(&self) -> Duration {
        monotonic_now()
    }
}

#[cfg(unix)]
fn monotonic_now() -> Duration {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // `CLOCK_MONOTONIC` is Python `time.monotonic()`'s Unix clock source. It
    // also preserves Python's initial scheduler comparison against 0.0; an
    // `Instant` origin created on first use would incorrectly delay the first
    // archival pass by a full configured interval.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
    if result == 0 {
        Duration::new(value.tv_sec.max(0) as u64, value.tv_nsec.max(0) as u32)
    } else {
        Duration::ZERO
    }
}

#[cfg(not(unix))]
fn monotonic_now() -> Duration {
    use std::sync::OnceLock;
    use std::time::Instant;

    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed()
}

type Loader =
    dyn Fn(&Path) -> Result<TraceArchivalServerConfig, TraceArchivalConfigError> + Send + Sync;

#[derive(Debug, Clone)]
struct CacheEntry {
    config_path: PathBuf,
    config: TraceArchivalServerConfig,
    expires_at: Duration,
}

struct CacheInner {
    entry: Mutex<Option<CacheEntry>>,
    clock: Arc<dyn TraceArchivalConfigClock>,
    loader: Arc<Loader>,
}

/// A cloneable, task-safe source for the current trace-archival configuration.
///
/// Calls before the TTL expires return the cached value. At expiry exactly one
/// caller reloads while holding the cache lock. If a refresh of the same path
/// fails, the last valid value is returned and its TTL is extended by another
/// five seconds. A new path never inherits another path's stale value.
#[derive(Clone)]
pub struct TraceArchivalConfigProvider {
    config_path: Option<PathBuf>,
    inner: Arc<CacheInner>,
}

impl fmt::Debug for TraceArchivalConfigProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraceArchivalConfigProvider")
            .field("config_path", &self.config_path)
            .finish_non_exhaustive()
    }
}

impl PartialEq for TraceArchivalConfigProvider {
    fn eq(&self, other: &Self) -> bool {
        self.config_path == other.config_path
    }
}

impl Eq for TraceArchivalConfigProvider {}

impl Default for TraceArchivalConfigProvider {
    fn default() -> Self {
        Self::new(None)
    }
}

impl TraceArchivalConfigProvider {
    /// Construct a provider for the CLI/env-resolved path.
    pub fn new(config_path: Option<PathBuf>) -> Self {
        Self::with_components(
            config_path,
            Arc::new(SystemMonotonicClock),
            Arc::new(|path| load_trace_archival_server_config(path)),
        )
    }

    /// Construct a provider with an injected monotonic clock. This keeps cache
    /// and scheduler tests deterministic while retaining the production YAML
    /// loader and all validation behavior.
    pub fn with_clock(
        config_path: Option<PathBuf>,
        clock: Arc<dyn TraceArchivalConfigClock>,
    ) -> Self {
        Self::with_components(
            config_path,
            clock,
            Arc::new(|path| load_trace_archival_server_config(path)),
        )
    }

    fn with_components(
        config_path: Option<PathBuf>,
        clock: Arc<dyn TraceArchivalConfigClock>,
        loader: Arc<Loader>,
    ) -> Self {
        Self {
            config_path,
            inner: Arc::new(CacheInner {
                entry: Mutex::new(None),
                clock,
                loader,
            }),
        }
    }

    /// The resolved config path, or `None` when trace archival is unconfigured.
    pub fn path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    /// Parse once without populating the runtime cache. Used during startup so
    /// invalid configuration fails before the listener/store starts, matching
    /// Python's CLI validation followed by its independently cached reads.
    pub fn validate_at_startup(
        &self,
    ) -> Result<Option<TraceArchivalServerConfig>, TraceArchivalConfigError> {
        let Some(path) = normalized_nonempty_path(self.config_path.as_deref()) else {
            return Ok(None);
        };
        (self.inner.loader)(&path).map(Some)
    }

    /// Return the current parsed config with Python's five-second TTL and
    /// stale-on-refresh-error behavior.
    pub fn get(&self) -> Result<Option<TraceArchivalServerConfig>, TraceArchivalConfigError> {
        let Some(config_path) = normalized_nonempty_path(self.config_path.as_deref()) else {
            *lock_unpoisoned(&self.inner.entry) = None;
            return Ok(None);
        };

        // Python samples `time.monotonic()` immediately before taking its lock.
        let now = self.inner.clock.now();
        let mut cached = lock_unpoisoned(&self.inner.entry);
        if let Some(entry) = cached.as_ref() {
            if entry.config_path == config_path && now < entry.expires_at {
                return Ok(Some(entry.config.clone()));
            }
        }

        let previous_config = cached
            .as_ref()
            .filter(|entry| entry.config_path == config_path)
            .map(|entry| entry.config.clone());

        match (self.inner.loader)(&config_path) {
            Ok(config) => {
                if previous_config
                    .as_ref()
                    .is_some_and(|previous| previous != &config)
                {
                    tracing::info!(
                        "Trace archival config changed; refreshed cached server settings."
                    );
                }
                *cached = Some(CacheEntry {
                    config_path,
                    config: config.clone(),
                    expires_at: now + TRACE_ARCHIVAL_CONFIG_CACHE_TTL,
                });
                Ok(Some(config))
            }
            Err(error) => {
                if let Some(previous_config) = previous_config {
                    tracing::warn!(
                        error = %error,
                        "Failed to refresh trace archival config; continuing to use the last valid config."
                    );
                    *cached = Some(CacheEntry {
                        config_path,
                        config: previous_config.clone(),
                        expires_at: now + TRACE_ARCHIVAL_CONFIG_CACHE_TTL,
                    });
                    Ok(Some(previous_config))
                } else {
                    Err(error)
                }
            }
        }
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn normalized_nonempty_path(path: Option<&Path>) -> Option<PathBuf> {
    let path = path?;
    if path.as_os_str().is_empty() || path.to_string_lossy().trim().is_empty() {
        return None;
    }

    // `str(Path(value))` removes redundant current-directory components. Rust's
    // `PathBuf` preserves a leading `./`, so rebuild from components here.
    let normalized = path
        .components()
        .fold(PathBuf::new(), |mut result, component| {
            if component != Component::CurDir {
                result.push(component.as_os_str());
            }
            result
        });
    Some(normalized)
}

/// Load and validate a trace-archival YAML file with Python-compatible schema
/// semantics and error messages.
pub fn load_trace_archival_server_config(
    config_path: impl AsRef<Path>,
) -> Result<TraceArchivalServerConfig, TraceArchivalConfigError> {
    let path = config_path.as_ref();
    let yaml = fs::read_to_string(path).map_err(|error| {
        invalid_config(
            path,
            format!(
                "Failed to read config file: {}",
                python_io_error(&error, path)
            ),
        )
    })?;
    reject_unknown_yaml_tag_syntax(path, &yaml)?;
    let payload: Value = serde_yaml::from_str(&yaml).map_err(|error| {
        invalid_config(
            path,
            format!(
                "Failed to parse YAML: {}",
                python_yaml_error(&error, path, &yaml)
            ),
        )
    })?;
    let top_level = payload.as_object().ok_or_else(|| {
        invalid_config(
            path,
            "Top-level YAML value must be a mapping containing 'trace_archival'.",
        )
    })?;
    let trace_archival = mapping_value(top_level, TRACE_ARCHIVAL_CONFIG_KEY)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_config(path, "Missing required 'trace_archival' mapping."))?;

    let enabled = python_yaml_bool(required_value(trace_archival, path, "enabled")?)
        .ok_or_else(|| invalid_config(path, "'trace_archival.enabled' must be a boolean."))?;

    let location_value = required_value(trace_archival, path, "location")?;
    let location = validate_repository_support(location_value)?;

    let retention_value = required_value(trace_archival, path, "retention")?;
    let retention = validate_retention(retention_value)?;

    let long_retention_allowlist = parse_long_retention_allowlist(
        mapping_value(trace_archival, "long_retention_allowlist"),
        path,
    )?;
    let interval_seconds = optional_positive_integer(
        trace_archival,
        path,
        "interval_seconds",
        Some(INTERVAL_SECONDS_DEFAULT),
        Some(INTERVAL_SECONDS_MAX),
    )?
    .expect("interval_seconds has a default");
    let max_traces_per_pass =
        optional_positive_integer(trace_archival, path, "max_traces_per_pass", None, None)?;

    Ok(TraceArchivalServerConfig {
        enabled,
        location,
        retention,
        long_retention_allowlist,
        interval_seconds,
        max_traces_per_pass,
    })
}

fn reject_unknown_yaml_tag_syntax(path: &Path, yaml: &str) -> Result<(), TraceArchivalConfigError> {
    for (line_index, line) in yaml.lines().enumerate() {
        let mut single_quoted = false;
        let mut double_quoted = false;
        let chars: Vec<(usize, char)> = line.char_indices().collect();
        let mut index = 0;
        while index < chars.len() {
            let (byte_offset, ch) = chars[index];
            match ch {
                '\'' if !double_quoted => single_quoted = !single_quoted,
                '"' if !single_quoted => double_quoted = !double_quoted,
                '#' if !single_quoted && !double_quoted => break,
                '!' if !single_quoted && !double_quoted => {
                    let starts_token = index == 0
                        || chars[index - 1].1.is_whitespace()
                        || matches!(chars[index - 1].1, '[' | '{' | ':' | ',' | '-');
                    if !starts_token {
                        index += 1;
                        continue;
                    }
                    let tail = &line[byte_offset..];
                    let token_len = tail
                        .find(|ch: char| ch.is_whitespace() || matches!(ch, '[' | ']' | '{' | '}'))
                        .unwrap_or(tail.len());
                    let token = &tail[..token_len];
                    let canonical = if let Some(name) = token.strip_prefix("!!") {
                        format!("tag:yaml.org,2002:{name}")
                    } else if token.starts_with("!<") && token.ends_with('>') {
                        token[2..token.len() - 1].to_string()
                    } else {
                        token.to_string()
                    };
                    let standard = canonical
                        .strip_prefix("tag:yaml.org,2002:")
                        .is_some_and(is_standard_yaml_tag);
                    if !standard {
                        return Err(invalid_config(
                            path,
                            format!(
                                "Failed to parse YAML: could not determine a constructor for the \
                                 tag {}\n{}",
                                python_repr(&canonical),
                                yaml_mark(path, (line_index + 1, index + 1))
                            ),
                        ));
                    }
                }
                _ => {}
            }
            index += 1;
        }
    }
    Ok(())
}

fn is_standard_yaml_tag(name: &str) -> bool {
    matches!(
        name,
        "null"
            | "bool"
            | "int"
            | "float"
            | "binary"
            | "timestamp"
            | "omap"
            | "pairs"
            | "set"
            | "str"
            | "seq"
            | "map"
    )
}

fn mapping_value<'a>(mapping: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    mapping.get(key)
}

fn required_value<'a>(
    mapping: &'a Map<String, Value>,
    path: &Path,
    key: &str,
) -> Result<&'a Value, TraceArchivalConfigError> {
    mapping_value(mapping, key).ok_or_else(|| {
        invalid_config(
            path,
            format!("Missing required 'trace_archival.{key}' value."),
        )
    })
}

fn optional_positive_integer(
    mapping: &Map<String, Value>,
    path: &Path,
    key: &str,
    default: Option<u64>,
    maximum: Option<u64>,
) -> Result<Option<u64>, TraceArchivalConfigError> {
    let Some(value) = mapping_value(mapping, key) else {
        return Ok(default);
    };
    let Some(value) = value.as_u64().filter(|value| *value > 0) else {
        return Err(invalid_config(
            path,
            format!("'trace_archival.{key}' must be a positive integer."),
        ));
    };
    if let Some(maximum) = maximum {
        if value > maximum {
            return Err(invalid_config(
                path,
                format!("'trace_archival.{key}' must be <= {maximum}."),
            ));
        }
    }
    Ok(Some(value))
}

fn python_yaml_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        // PyYAML's SafeLoader uses YAML 1.1 implicit booleans, while
        // serde_yaml follows YAML 1.2. Normalize the additional 1.1 spellings.
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "yes" | "on" => Some(true),
            "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn validate_retention(value: &Value) -> Result<String, TraceArchivalConfigError> {
    let Some(value) = value.as_str() else {
        return Err(TraceArchivalConfigError::new(
            "Invalid value for 'trace_archival.retention'. Expected a duration in the form \
             `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h').",
        ));
    };
    let trimmed = value.trim();
    if trimmed.chars().count() > 32 {
        return Err(TraceArchivalConfigError::new(
            "Invalid value for 'trace_archival.retention'. Maximum length is 32 characters.",
        ));
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return Err(invalid_retention());
    };
    let rest: Vec<char> = chars.collect();
    let valid = first.is_ascii_digit()
        && first != '0'
        && !rest.is_empty()
        && rest[..rest.len() - 1].iter().all(char::is_ascii_digit)
        && matches!(rest.last(), Some('m' | 'h' | 'd'));
    if !valid {
        return Err(invalid_retention());
    }
    Ok(trimmed.to_string())
}

fn invalid_retention() -> TraceArchivalConfigError {
    TraceArchivalConfigError::new(
        "Invalid value for 'trace_archival.retention'. Expected a duration in the form \
         `<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h').",
    )
}

fn validate_repository_support(value: &Value) -> Result<String, TraceArchivalConfigError> {
    let Some(value) = value.as_str() else {
        return Err(invalid_location_uri());
    };
    let location = value.trim();
    let Some(scheme) = uri_scheme(location) else {
        return Err(invalid_location_uri());
    };
    if scheme == "mlflow-artifacts" {
        return Err(TraceArchivalConfigError::new(
            "Invalid value for 'trace_archival.location'. Trace archival location cannot use \
             the proxy-only `mlflow-artifacts:` scheme.",
        ));
    }

    // `dbfs:` resolves either to DatabricksArtifactRepository (ACL trace/run
    // paths) or DbfsRestArtifactRepository. Python rejects the former because
    // archived reads are unsupported and the latter because archived payloads
    // cannot be deleted. UC volume repos inherit the base no-op deletion API
    // and therefore take the deletion branch as well.
    if scheme == "dbfs" {
        if location
            .to_ascii_lowercase()
            .contains("/databricks/mlflow-tracking/")
        {
            return Err(TraceArchivalConfigError::new(format!(
                "Invalid value for 'trace_archival.location'. Trace archival location {} \
                 resolves to a Databricks trace artifact repository that does not support \
                 archived trace reads.",
                python_repr(location)
            )));
        }
        return Err(repository_cannot_delete(location));
    }

    // Built-in registry entries whose repository overrides `delete_artifacts`.
    // Python's class-level check also accepts the `runs:`/`models:` wrappers
    // once their underlying repository lookup succeeds.
    const DELETABLE_SCHEMES: &[&str] = &[
        "file",
        "s3",
        "r2",
        "b2",
        "gs",
        "wasbs",
        "ftp",
        "sftp",
        "hdfs",
        "viewfs",
        "http",
        "https",
        "abfss",
        "runs",
        "models",
        "file-plugin",
    ];
    if DELETABLE_SCHEMES.contains(&scheme.as_str()) {
        return Ok(location.to_string());
    }

    const REGISTERED_SCHEMES: &str = "['', 'file', 's3', 'r2', 'b2', 'gs', 'wasbs', 'ftp', \
         'sftp', 'dbfs', 'hdfs', 'viewfs', 'runs', 'models', 'http', 'https', \
         'mlflow-artifacts', 'abfss', 'file-plugin']";
    Err(TraceArchivalConfigError::new(format!(
        "Could not find a registered artifact repository for: {location}. Currently registered \
         schemes are: {REGISTERED_SCHEMES}"
    )))
}

fn invalid_location_uri() -> TraceArchivalConfigError {
    TraceArchivalConfigError::new(
        "Invalid value for 'trace_archival.location'. Expected a URI string.",
    )
}

fn repository_cannot_delete(location: &str) -> TraceArchivalConfigError {
    TraceArchivalConfigError::new(format!(
        "Invalid value for 'trace_archival.location'. Trace archival location {} resolves to an \
         artifact repository that does not support deleting archived payloads.",
        python_repr(location)
    ))
}

fn uri_scheme(value: &str) -> Option<String> {
    let colon = value.find(':')?;
    let candidate = &value[..colon];
    let mut chars = candidate.chars();
    if !chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        || !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
    {
        return None;
    }
    Some(candidate.to_ascii_lowercase())
}

fn parse_long_retention_allowlist(
    value: Option<&Value>,
    path: &Path,
) -> Result<Vec<String>, TraceArchivalConfigError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(entries) = value.as_array() else {
        return Err(invalid_config(
            path,
            "'trace_archival.long_retention_allowlist' must be a list.",
        ));
    };

    let mut scalar_entries = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        if entry.is_object() || entry.is_array() {
            return Err(invalid_config(
                path,
                format!(
                    "'trace_archival.long_retention_allowlist[{index}]' must be a scalar \
                     experiment ID."
                ),
            ));
        }
        scalar_entries.push(python_scalar_string(entry));
    }

    let mut result = Vec::new();
    for raw_id in scalar_entries.join(",").split(',') {
        let experiment_id = raw_id.trim();
        if experiment_id.is_empty() {
            continue;
        }
        if !valid_experiment_id(experiment_id) {
            return Err(invalid_config(
                path,
                format!("Invalid experiment ID: '{experiment_id}'"),
            ));
        }
        if !result.iter().any(|seen| seen == experiment_id) {
            result.push(experiment_id.to_string());
        }
    }
    Ok(result)
}

fn python_scalar_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => unreachable!("container rejected by caller"),
    }
}

fn valid_experiment_id(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphanumeric()
        && value.chars().count() <= 64
        && chars.all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-'))
}

fn invalid_config(path: &Path, message: impl fmt::Display) -> TraceArchivalConfigError {
    TraceArchivalConfigError::new(format!(
        "Invalid trace archival config file '{}': {message}",
        path.display()
    ))
}

fn python_io_error(error: &std::io::Error, path: &Path) -> String {
    let kind = match error.raw_os_error() {
        Some(2) => "No such file or directory",
        Some(13) => "Permission denied",
        Some(21) => "Is a directory",
        _ => return error.to_string(),
    };
    format!(
        "[Errno {}] {kind}: {}",
        error.raw_os_error().unwrap_or_default(),
        python_repr(&path.to_string_lossy())
    )
}

fn python_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn python_yaml_error(error: &serde_yaml::Error, path: &Path, yaml: &str) -> String {
    let display = error.to_string();
    let Some(location) = error.location() else {
        return display;
    };
    let mark = format!(
        "  in \"{}\", line {}, column {}",
        path.display(),
        location.line(),
        location.column()
    );

    // PyYAML and serde_yaml both use libyaml but format parser diagnostics
    // differently. Normalize the common malformed-flow case exercised by the
    // Python CLI and parity suite to PyYAML's exact two-line diagnostic.
    if display.contains("did not find expected node content")
        || display.contains("expected a node")
        || display.contains("while parsing a flow node")
    {
        return format!(
            "while parsing a flow node\nexpected the node content, but found '<stream end>'\n{mark}"
        );
    }
    if display.contains("mapping values are not allowed") {
        return format!("mapping values are not allowed here\n{mark}");
    }
    let marks = yaml_error_marks(&display);
    if display.contains("did not find expected ',' or '}'")
        && display.contains("while parsing a flow mapping")
        && marks.len() >= 2
    {
        return format!(
            "while parsing a flow mapping\n{}\nexpected ',' or '}}', but got '<stream end>'\n{}",
            yaml_mark(path, marks[1]),
            yaml_mark(path, marks[0])
        );
    }
    if display.contains("did not find expected ',' or ']'")
        && display.contains("while parsing a flow sequence")
        && marks.len() >= 2
    {
        return format!(
            "while parsing a flow sequence\n{}\nexpected ',' or ']', but got '<stream end>'\n{}",
            yaml_mark(path, marks[1]),
            yaml_mark(path, marks[0])
        );
    }
    if display.contains("found unexpected end of stream")
        && display.contains("while scanning a quoted scalar")
        && marks.len() >= 2
    {
        return format!(
            "while scanning a quoted scalar\n{}\nfound unexpected end of stream\n{}",
            yaml_mark(path, marks[1]),
            yaml_mark(path, marks[0])
        );
    }
    if display.contains("found character that cannot start any token") && !marks.is_empty() {
        let (line, column) = marks[0];
        let character = yaml
            .lines()
            .nth(line.saturating_sub(1))
            .and_then(|line| line.chars().nth(column.saturating_sub(1)))
            .unwrap_or('\t');
        return format!(
            "while scanning for the next token\nfound character '{}' that cannot start any token\n{}",
            character.escape_default(),
            yaml_mark(path, marks[0])
        );
    }
    format!("{display}\n{mark}")
}

fn yaml_error_marks(display: &str) -> Vec<(usize, usize)> {
    let mut marks = Vec::new();
    let mut remainder = display;
    while let Some(index) = remainder.find("at line ") {
        remainder = &remainder[index + "at line ".len()..];
        let Some((line, after_line)) = remainder.split_once(" column ") else {
            break;
        };
        let line = line.parse().ok();
        let column_len = after_line
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(after_line.len());
        let column = after_line[..column_len].parse().ok();
        if let (Some(line), Some(column)) = (line, column) {
            marks.push((line, column));
        }
        remainder = &after_line[column_len..];
    }
    marks
}

fn yaml_mark(path: &Path, (line, column): (usize, usize)) -> String {
    format!("  in \"{}\", line {line}, column {column}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use tempfile::TempDir;

    #[derive(Default)]
    struct TestClock(AtomicU64);

    impl TestClock {
        fn set(&self, seconds: u64) {
            self.0.store(seconds, Ordering::SeqCst);
        }
    }

    impl TraceArchivalConfigClock for TestClock {
        fn now(&self) -> Duration {
            Duration::from_secs(self.0.load(Ordering::SeqCst))
        }
    }

    fn config(retention: &str) -> TraceArchivalServerConfig {
        TraceArchivalServerConfig {
            enabled: true,
            location: "file:///tmp/archive".to_string(),
            retention: retention.to_string(),
            long_retention_allowlist: Vec::new(),
            interval_seconds: INTERVAL_SECONDS_DEFAULT,
            max_traces_per_pass: None,
        }
    }

    fn write(temp: &TempDir, payload: &str) -> PathBuf {
        let path = temp.path().join("trace-archival.yaml");
        fs::write(&path, payload).unwrap();
        path
    }

    fn valid_yaml(location: &str) -> String {
        format!(
            "unknown_top_level: accepted\ntrace_archival:\n  enabled: true\n  location: \
             {location}\n  retention: 30d\n  unknown_nested: accepted\n"
        )
    }

    #[test]
    fn parses_full_schema_defaults_and_ignores_unknown_keys_like_python() {
        let temp = TempDir::new().unwrap();
        let path = write(
            &temp,
            "unknown: value\ntrace_archival:\n  enabled: false\n  location: file:///tmp/a\n  \
             retention: ' 30d '\n  long_retention_allowlist: [1, '2', 1, '3, 4', '']\n  \
             interval_seconds: 42\n  max_traces_per_pass: 1000\n  extra: ignored\n",
        );
        assert_eq!(
            load_trace_archival_server_config(path).unwrap(),
            TraceArchivalServerConfig {
                enabled: false,
                location: "file:///tmp/a".to_string(),
                retention: "30d".to_string(),
                long_retention_allowlist: vec!["1", "2", "3", "4"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                interval_seconds: 42,
                max_traces_per_pass: Some(1000),
            }
        );
    }

    #[test]
    fn duplicate_yaml_keys_keep_the_last_value_like_pyyaml() {
        let temp = TempDir::new().unwrap();
        let path = write(
            &temp,
            "trace_archival:\n  enabled: true\n  enabled: false\n  location: file:///tmp/a\n  \
             retention: 30d\n",
        );
        assert!(!load_trace_archival_server_config(path).unwrap().enabled);
    }

    #[test]
    fn schema_error_messages_match_python_matrix() {
        let temp = TempDir::new().unwrap();
        let cases = [
            ("null\n", "Top-level YAML value must be a mapping containing 'trace_archival'."),
            ("{}\n", "Missing required 'trace_archival' mapping."),
            ("trace_archival: {}\n", "Missing required 'trace_archival.enabled' value."),
            ("trace_archival:\n  enabled: yes-ish\n", "'trace_archival.enabled' must be a boolean."),
            (
                "trace_archival:\n  enabled: true\n  location: file:///tmp/a\n  retention: 30d\n  interval_seconds: 0\n",
                "'trace_archival.interval_seconds' must be a positive integer.",
            ),
            (
                "trace_archival:\n  enabled: true\n  location: file:///tmp/a\n  retention: 30d\n  interval_seconds: 86401\n",
                "'trace_archival.interval_seconds' must be <= 86400.",
            ),
            (
                "trace_archival:\n  enabled: true\n  location: file:///tmp/a\n  retention: 30d\n  max_traces_per_pass: false\n",
                "'trace_archival.max_traces_per_pass' must be a positive integer.",
            ),
        ];
        for (index, (payload, suffix)) in cases.into_iter().enumerate() {
            let path = temp.path().join(format!("case-{index}.yaml"));
            fs::write(&path, payload).unwrap();
            let error = load_trace_archival_server_config(&path).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!(
                    "Invalid trace archival config file '{}': {suffix}",
                    path.display()
                )
            );
        }
    }

    #[test]
    fn value_and_repository_errors_match_python() {
        let temp = TempDir::new().unwrap();
        let cases = [
            (
                "archive",
                "Invalid value for 'trace_archival.location'. Expected a URI string.",
            ),
            (
                "mlflow-artifacts:/archive",
                "Invalid value for 'trace_archival.location'. Trace archival location cannot use the proxy-only `mlflow-artifacts:` scheme.",
            ),
            (
                "dbfs:/archive",
                "Invalid value for 'trace_archival.location'. Trace archival location 'dbfs:/archive' resolves to an artifact repository that does not support deleting archived payloads.",
            ),
            (
                "dbfs:/databricks/mlflow-tracking/1/run/artifacts",
                "Invalid value for 'trace_archival.location'. Trace archival location 'dbfs:/databricks/mlflow-tracking/1/run/artifacts' resolves to a Databricks trace artifact repository that does not support archived trace reads.",
            ),
        ];
        for (index, (location, expected)) in cases.into_iter().enumerate() {
            let path = temp.path().join(format!("location-{index}.yaml"));
            fs::write(&path, valid_yaml(location)).unwrap();
            assert_eq!(
                load_trace_archival_server_config(path)
                    .unwrap_err()
                    .to_string(),
                expected
            );
        }
    }

    #[test]
    fn retention_and_allowlist_validation_match_python() {
        let temp = TempDir::new().unwrap();
        let invalid_id = write(
            &temp,
            "trace_archival:\n  enabled: true\n  location: file:///tmp/a\n  retention: 30d\n  \
             long_retention_allowlist: ['1', 'invalid id']\n",
        );
        assert_eq!(
            load_trace_archival_server_config(&invalid_id)
                .unwrap_err()
                .to_string(),
            format!(
                "Invalid trace archival config file '{}': Invalid experiment ID: 'invalid id'",
                invalid_id.display()
            )
        );

        for retention in ["0d", "30D", "30days", "1.5d", ""] {
            let path = temp.path().join(format!("retention-{retention:?}.yaml"));
            fs::write(
                &path,
                format!(
                    "trace_archival:\n  enabled: true\n  location: file:///tmp/a\n  retention: \
                     '{retention}'\n"
                ),
            )
            .unwrap();
            assert_eq!(
                load_trace_archival_server_config(path)
                    .unwrap_err()
                    .to_string(),
                invalid_retention().to_string()
            );
        }
    }

    #[test]
    fn cache_ttl_reload_and_stale_tolerance_are_deterministic() {
        let clock = Arc::new(TestClock::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let current = Arc::new(Mutex::new(Ok(config("30d"))));
        let loader_calls = calls.clone();
        let loader_value = current.clone();
        let provider = TraceArchivalConfigProvider::with_components(
            Some(PathBuf::from("config.yaml")),
            clock.clone(),
            Arc::new(move |_| {
                loader_calls.fetch_add(1, Ordering::SeqCst);
                lock_unpoisoned(&loader_value).clone()
            }),
        );

        clock.set(10);
        assert_eq!(provider.get().unwrap(), Some(config("30d")));
        *lock_unpoisoned(&current) = Ok(config("7d"));
        clock.set(14);
        assert_eq!(provider.get().unwrap(), Some(config("30d")));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        clock.set(15);
        assert_eq!(provider.get().unwrap(), Some(config("7d")));
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        *lock_unpoisoned(&current) = Err(TraceArchivalConfigError::new("refresh failed"));
        clock.set(20);
        assert_eq!(provider.get().unwrap(), Some(config("7d")));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        clock.set(24);
        assert_eq!(provider.get().unwrap(), Some(config("7d")));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn cache_does_not_use_stale_value_for_a_different_path() {
        let clock = Arc::new(TestClock::default());
        let provider = TraceArchivalConfigProvider::with_components(
            Some(PathBuf::from("first.yaml")),
            clock,
            Arc::new(|path| {
                if path == Path::new("first.yaml") {
                    Ok(config("30d"))
                } else {
                    Err(TraceArchivalConfigError::new("new path invalid"))
                }
            }),
        );
        provider.get().unwrap();
        let other = TraceArchivalConfigProvider {
            config_path: Some(PathBuf::from("second.yaml")),
            inner: provider.inner.clone(),
        };
        assert_eq!(
            other.get().unwrap_err(),
            TraceArchivalConfigError::new("new path invalid")
        );
    }

    #[test]
    fn cache_serializes_concurrent_reload() {
        let calls = Arc::new(AtomicUsize::new(0));
        let loader_calls = calls.clone();
        let provider = TraceArchivalConfigProvider::with_components(
            Some(PathBuf::from("config.yaml")),
            Arc::new(TestClock::default()),
            Arc::new(move |_| {
                loader_calls.fetch_add(1, Ordering::SeqCst);
                Ok(config("30d"))
            }),
        );
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let provider = provider.clone();
                std::thread::spawn(move || provider.get().unwrap())
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), Some(config("30d")));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
