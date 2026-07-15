//! `createModelVersion` source validation — the registry's path-traversal
//! defense (§3.14, security-relevant). Ports `_create_model_version`'s
//! pre-store checks (`mlflow/server/handlers.py:2829-2963`) byte-for-byte:
//!
//! * [`validate_non_local_source_contains_relative_paths`] — for non-local
//!   (schemed) sources: reject relative-path escapes (`..`), NUL bytes, and any
//!   source whose normalized path differs from its lexical path
//!   (`_validate_non_local_source_contains_relative_paths`, `handlers.py:2829`).
//! * [`validate_source_run`] / [`validate_source_model`] — for a **local**
//!   source, require a `run_id`/`model_id` whose artifact directory *contains*
//!   the resolved local path; otherwise reject
//!   (`_validate_source_run` / `_validate_source_model`, `handlers.py:2867-2914`).
//! * The optional `MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX` gate
//!   (`handlers.py:2933-2940`), applied first when the env var is set.
//! * The prompt-source branch (`is_prompt`): block `file:` and absolute local
//!   paths, and traversal-check only schemed sources
//!   (`handlers.py:2942-2958`).
//!
//! ## `is_local_uri` / `local_file_uri_to_path`
//!
//! [`is_local_uri`] ports `mlflow.utils.uri.is_local_uri` (`uri.py:28`) and
//! [`local_file_uri_to_path`] ports `mlflow.utils.file_utils.local_file_uri_to_path`
//! (`file_utils.py:447`), restricted to the POSIX server target.
//!
//! ## `Path.resolve()` fidelity
//!
//! Python calls `pathlib.Path(source_path).resolve()`. For the non-existent,
//! already-`//`-collapsed, `..`-free absolute paths this validator sees,
//! `resolve()` is a pure lexical normalization (drop `.` segments, collapse
//! separators) with no symlink following. [`lexical_resolve`] replicates that:
//! an absolute path normalizes in place; a relative path is anchored to the CWD
//! (exactly as `resolve()` does), which — for the schemed sources that reach
//! here — is what makes a genuinely relative path fail the equality check.

use std::path::{Component, Path, PathBuf};

use mlflow_error::MlflowError;

use crate::state::AppState;

/// The env var gating an extra source regex
/// (`MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX`). When set, the source
/// must match it or the request is rejected. We support the empty/simple prefix
/// forms MLflow deployments use; the value is treated as a substring/anchored
/// match via a tiny glob-free contains-or-prefix check (see [`regex_matches`]).
const SOURCE_VALIDATION_REGEX_ENV: &str = "MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX";

/// Top-level `createModelVersion` source validation, mirroring the branching in
/// `_create_model_version` (`handlers.py:2933-2963`).
///
/// Order matches Python exactly: (1) the optional regex gate; (2) the prompt
/// branch OR (3) the model-id branch OR (4) the run-id branch.
pub async fn validate_create_model_version_source(
    state: &AppState,
    workspace: &str,
    source: &str,
    run_id: Option<&str>,
    model_id: Option<&str>,
    is_prompt: bool,
) -> Result<(), MlflowError> {
    // (1) Optional deployment-configured regex gate.
    if let Ok(regex) = std::env::var(SOURCE_VALIDATION_REGEX_ENV) {
        if !regex.is_empty() && !regex_matches(&regex, source) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid model version source: '{source}'."
            )));
        }
    }

    if is_prompt {
        return validate_prompt_source(source);
    }
    if let Some(model_id) = model_id {
        return validate_source_model(state, workspace, source, model_id).await;
    }
    validate_source_run(state, workspace, source, run_id).await
}

/// The prompt-source branch (`handlers.py:2942-2958`): block `file:` URIs and
/// schemeless absolute paths, then traversal-check only sources that carry a
/// URL scheme.
fn validate_prompt_source(source: &str) -> Result<(), MlflowError> {
    let scheme = url_scheme(source);
    let is_absolute_schemeless = scheme.is_none() && source.starts_with('/');
    if scheme.as_deref() == Some("file") || is_absolute_schemeless {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid prompt source: '{source}'. Local source paths are not allowed for prompts."
        )));
    }
    if scheme.is_some() {
        validate_non_local_source_contains_relative_paths(source)?;
    }
    Ok(())
}

/// `_validate_source_run` (`handlers.py:2867`). For a local source, require a
/// `run_id` whose (local) artifact directory contains the resolved source path;
/// otherwise reject. For a non-local source, run the traversal check.
async fn validate_source_run(
    state: &AppState,
    workspace: &str,
    source: &str,
    run_id: Option<&str>,
) -> Result<(), MlflowError> {
    if is_local_uri(source)? {
        if let Some(run_id) = run_id {
            let run = state.tracking_store().get_run(workspace, run_id).await?;
            let source_resolved = lexical_resolve(&local_file_uri_to_path(source));
            if let Some(artifact_uri) = run.info.artifact_uri.as_deref() {
                if is_local_uri(artifact_uri)? {
                    let run_artifact_dir = lexical_resolve(&local_file_uri_to_path(artifact_uri));
                    if path_contains(&run_artifact_dir, &source_resolved) {
                        return Ok(());
                    }
                }
            }
        }
        return Err(local_source_run_error(source));
    }
    validate_non_local_source_contains_relative_paths(source)
}

/// `_validate_source_model` (`handlers.py:2892`). Same shape as the run variant
/// but keyed on `model_id` → logged-model `artifact_location`.
async fn validate_source_model(
    state: &AppState,
    workspace: &str,
    source: &str,
    model_id: &str,
) -> Result<(), MlflowError> {
    if is_local_uri(source)? {
        let model = state
            .tracking_store()
            .get_logged_model(workspace, model_id, false)
            .await?;
        let source_resolved = lexical_resolve(&local_file_uri_to_path(source));
        if is_local_uri(&model.artifact_location)? {
            let model_artifact_dir =
                lexical_resolve(&local_file_uri_to_path(&model.artifact_location));
            if path_contains(&model_artifact_dir, &source_resolved) {
                return Ok(());
            }
        }
        return Err(local_source_model_error(source));
    }
    validate_non_local_source_contains_relative_paths(source)
}

/// `_validate_non_local_source_contains_relative_paths` (`handlers.py:2829`).
///
/// Verbatim algorithm:
/// 1. `unquote_plus` until stable.
/// 2. `source_path = collapse_slashes(urlparse(source).path.rstrip("/"))`.
/// 3. Reject on a NUL in `source_path` or any `..` segment in `source.split("/")`.
/// 4. Reject when `lexical_resolve(source_path)` (drive-stripped) differs from
///    `source_path`.
pub fn validate_non_local_source_contains_relative_paths(source: &str) -> Result<(), MlflowError> {
    let err = || {
        MlflowError::invalid_parameter_value(format!(
            "Invalid model version source: '{source}'. If supplying a source as an http, https, \
             local file path, ftp, objectstore, or mlflow-artifacts uri, an absolute path must be \
             provided without relative path references present. Please provide an absolute path."
        ))
    };

    // (1) Unquote (form-decoding: '+' → ' ', %XX → byte) to a fixpoint.
    let mut decoded = source.to_string();
    loop {
        let next = unquote_plus(&decoded);
        if next == decoded {
            break;
        }
        decoded = next;
    }

    // (2) urlparse(decoded).path, strip trailing '/', collapse repeated '/'.
    let raw_path = url_path_component(&decoded);
    let trimmed = raw_path.trim_end_matches('/');
    let source_path = collapse_slashes(trimmed);

    // (3) NUL in the path, or a literal `..` segment anywhere in the raw source.
    if source_path.contains('\u{0}') || decoded.split('/').any(|seg| seg == "..") {
        return Err(err());
    }

    // (4) resolved (drive-stripped) must equal source_path.
    let resolved = lexical_resolve(&source_path);
    let resolved_posix = resolved.to_string_lossy();
    // `os.path.splitdrive` is a no-op on POSIX, so resolved_path == resolved.
    if resolved_posix != source_path {
        return Err(err());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Error messages (verbatim Python strings)
// ---------------------------------------------------------------------------

fn local_source_run_error(source: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Invalid model version source: '{source}'. To use a local path as a model version \
         source, the run_id request parameter has to be specified and the local path has to be \
         contained within the artifact directory of the run specified by the run_id."
    ))
}

fn local_source_model_error(source: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Invalid model version source: '{source}'. To use a local path as a model version \
         source, the model_id request parameter has to be specified and the local path has to \
         be contained within the artifact directory of the run specified by the model_id."
    ))
}

// ---------------------------------------------------------------------------
// URI helpers (ported from mlflow.utils.uri / file_utils, POSIX target)
// ---------------------------------------------------------------------------

/// Port of `is_local_uri` (`uri.py:28`) for the POSIX server. Returns
/// `Ok(true)` for schemeless paths and local `file:`/single-drive URIs;
/// `Ok(false)` for remote-hosted or non-local schemes. Errors when a `file:`
/// URI names a remote host (`"... is not a valid remote uri ..."`), matching
/// Python's raise inside `is_local_uri`.
fn is_local_uri(uri: &str) -> Result<bool, MlflowError> {
    // `is_tracking_or_registry_uri=True`: "databricks" is non-local. A source
    // is never the literal "databricks", but keep the branch for fidelity.
    if uri == "databricks" {
        return Ok(false);
    }
    let scheme = url_scheme(uri);
    let Some(scheme) = scheme else {
        // No scheme → local path.
        return Ok(true);
    };

    let host = url_host(uri);
    let is_remote_hostname = host
        .as_deref()
        .is_some_and(|h| !(h == "." || h.starts_with("localhost") || h.starts_with("127.0.0.1")));

    if scheme == "file" {
        if is_remote_hostname {
            return Err(MlflowError::internal_error(format!(
                "{uri} is not a valid remote uri. For remote access on windows, please consider \
                 using a different scheme such as SMB (e.g. smb://<hostname>/<path>)."
            )));
        }
        return Ok(true);
    }

    if is_remote_hostname {
        return Ok(false);
    }

    // POSIX single-letter drive check is Windows-only; skip it. Any other
    // schemed, non-remote-hostname URI is treated as non-local.
    Ok(false)
}

/// Port of `local_file_uri_to_path` (`file_utils.py:447`) for POSIX:
/// for a `file:` URI, take the path component and percent-decode it; otherwise
/// percent-decode the whole string (`url2pathname` == `unquote` on POSIX).
fn local_file_uri_to_path(uri: &str) -> String {
    if url_scheme(uri).as_deref() == Some("file") {
        return percent_decode(&url_path_component(uri));
    }
    percent_decode(uri)
}

/// Extract the URL scheme the way CPython's `urllib.parse.urlsplit` does:
/// ASCII letter followed by letters/digits/`+`/`-`/`.`, before the first `:`,
/// lowercased. `None` when there is no valid scheme.
fn url_scheme(uri: &str) -> Option<String> {
    let colon = uri.find(':')?;
    let candidate = &uri[..colon];
    let mut chars = candidate.chars();
    let first = chars.next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
        return None;
    }
    Some(candidate.to_ascii_lowercase())
}

/// The `netloc` host of a URI (the `//authority` component's host, without a
/// `:port` or `user@`), the way `urlparse(...).hostname` returns it (lowercased,
/// `None` when there is no authority).
fn url_host(uri: &str) -> Option<String> {
    let after_scheme = match uri.find(':') {
        Some(i) => &uri[i + 1..],
        None => uri,
    };
    let authority = after_scheme.strip_prefix("//")?;
    // Authority ends at the first '/', '?', or '#'.
    let end = authority.find(['/', '?', '#']).unwrap_or(authority.len());
    let authority = &authority[..end];
    // Drop userinfo.
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    // Drop port (last ':' not inside brackets — simple form for our inputs).
    let host = host_port.rsplit_once(':').map_or(host_port, |(h, _)| h);
    if host.is_empty() {
        None
    } else {
        Some(host.to_ascii_lowercase())
    }
}

/// The `path` component of a URI the way `urlparse(...).path` returns it: strip
/// the scheme, then an optional `//authority`, then drop any `?query`/`#fragment`.
fn url_path_component(uri: &str) -> String {
    let after_scheme = match uri.find(':') {
        Some(i) => &uri[i + 1..],
        None => uri,
    };
    let rest = if let Some(stripped) = after_scheme.strip_prefix("//") {
        match stripped.find('/') {
            Some(i) => &stripped[i..],
            None => "",
        }
    } else {
        after_scheme
    };
    let end = rest.find(['?', '#']).unwrap_or(rest.len());
    rest[..end].to_string()
}

/// Collapse runs of `/` into a single `/` (`re.sub(r"/+", "/", s)`).
fn collapse_slashes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_slash = false;
    for c in s.chars() {
        if c == '/' {
            if !prev_slash {
                out.push('/');
            }
            prev_slash = true;
        } else {
            out.push(c);
            prev_slash = false;
        }
    }
    out
}

/// Replicate `pathlib.Path(p).resolve()` for the non-symlink inputs this
/// validator sees: normalize `.`/`..`/`//` lexically. An absolute path stays
/// absolute; a relative path is anchored at the current working directory
/// (exactly what `resolve()` does), so a genuinely relative source path ends up
/// differing from its lexical form and fails the equality check.
fn lexical_resolve(path: &str) -> PathBuf {
    let p = Path::new(path);
    let anchored: PathBuf = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(p)
    };
    let mut out = PathBuf::new();
    for comp in anchored.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// True when `resolved_source` equals `dir` or is nested under it — the
/// `run_artifact_dir in [source, *source.parents]` membership test in Python.
fn path_contains(dir: &Path, resolved_source: &Path) -> bool {
    resolved_source == dir || resolved_source.starts_with(dir)
}

/// Approximate the deployment `MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX`
/// gate. MLflow deployments configure anchored prefix patterns like
/// `^mlflow-artifacts:/.*$`; we support that shape without pulling in a regex
/// dependency by matching an optional `^`-anchored literal prefix (stripping a
/// trailing `.*$` / `$`). Any other pattern falls back to a substring match.
fn regex_matches(pattern: &str, source: &str) -> bool {
    let mut pat = pattern;
    let anchored_start = pat.starts_with('^');
    if anchored_start {
        pat = &pat[1..];
    }
    // Strip a trailing `.*$` or `$` (common "prefix" pattern).
    for suffix in [".*$", "$"] {
        if let Some(stripped) = pat.strip_suffix(suffix) {
            pat = stripped;
            break;
        }
    }
    // Remaining literal must not contain regex metachars for this fast path; if
    // it does, fall back to a plain substring test.
    let literal_ok = !pat.contains(['[', '(', '\\', '+', '*', '?', '|', '{']);
    if literal_ok && anchored_start {
        source.starts_with(pat)
    } else {
        source.contains(pat)
    }
}

/// `urllib.parse.unquote_plus`: `+` → space, then percent-decode `%XX` runs as
/// UTF-8 (lossy for invalid sequences), leaving malformed escapes untouched.
fn unquote_plus(s: &str) -> String {
    let spaced: String = s.chars().map(|c| if c == '+' { ' ' } else { c }).collect();
    percent_decode(&spaced)
}

/// `urllib.parse.unquote` with default UTF-8 / errors="replace".
fn percent_decode(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut byte_buf: Vec<u8> = Vec::new();
    let flush = |byte_buf: &mut Vec<u8>, out: &mut String| {
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(byte_buf));
            byte_buf.clear();
        }
    };
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() + 1 && i + 2 <= bytes.len() {
            let h1 = bytes.get(i + 1).copied();
            let h2 = bytes.get(i + 2).copied();
            if let (Some(a), Some(b)) = (h1, h2) {
                if let (Some(hi), Some(lo)) = (hex_val(a), hex_val(b)) {
                    byte_buf.push((hi << 4) | lo);
                    i += 3;
                    continue;
                }
            }
            flush(&mut byte_buf, &mut out);
            out.push('%');
            i += 1;
        } else {
            flush(&mut byte_buf, &mut out);
            let ch_len = utf8_char_len(bytes[i]);
            let end = (i + ch_len).min(bytes.len());
            out.push_str(&String::from_utf8_lossy(&bytes[i..end]));
            i = end;
        }
    }
    flush(&mut byte_buf, &mut out);
    out
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn utf8_char_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_absolute_remote_sources() {
        for src in [
            "mlflow-artifacts:/models",
            "mlflow-artifacts:/models/",
            "mlflow-artifacts:/models///",
            "mlflow-artifacts:/models/foo///bar",
            "mlflow-artifacts://host:9000/models",
            "mlflow-artifacts://host:9000/models/artifact/..../",
        ] {
            assert!(
                validate_non_local_source_contains_relative_paths(src).is_ok(),
                "expected OK for {src:?}"
            );
        }
    }

    #[test]
    fn rejects_relative_traversal_sources() {
        for src in [
            "mlflow-artifacts://host:9000/models/../../../",
            "http://host:9000/models/../../../",
            "https://host/api/2.0/mlflow-artifacts/artifacts/../../../",
            "s3a://my_bucket/api/2.0/mlflow-artifacts/artifacts/../../../",
            "ftp://host:8888/api/2.0/mlflow-artifacts/artifacts/../../../",
            "mlflow-artifacts://host:9000/models/..%2f..%2fartifacts",
            "mlflow-artifacts://host:9000/models/artifact%00",
        ] {
            assert!(
                validate_non_local_source_contains_relative_paths(src).is_err(),
                "expected rejection for {src:?}"
            );
        }
    }

    #[test]
    fn dbfs_encoded_traversal_is_rejected() {
        let src = "dbfs:/run/artifacts/a%3f/../../../../../../../../../../";
        let err = validate_non_local_source_contains_relative_paths(src).unwrap_err();
        assert!(err.message.contains("Invalid model version source"));
    }

    #[test]
    fn scheme_and_host_extraction() {
        assert_eq!(
            url_scheme("mlflow-artifacts://h/p"),
            Some("mlflow-artifacts".to_string())
        );
        assert_eq!(url_scheme("/local/path"), None);
        assert_eq!(
            url_host("file://123.4.5.6/x"),
            Some("123.4.5.6".to_string())
        );
        assert_eq!(url_host("file:///x"), None);
        assert_eq!(
            url_host("mlflow-artifacts://host:9000/p"),
            Some("host".to_string())
        );
    }

    #[test]
    fn is_local_uri_classifies() {
        assert!(is_local_uri("/tmp/x").unwrap());
        assert!(is_local_uri("file:///tmp/x").unwrap());
        assert!(is_local_uri("file://localhost/tmp/x").unwrap());
        assert!(!is_local_uri("s3://bucket/x").unwrap());
        assert!(!is_local_uri("mlflow-artifacts:/x").unwrap());
        // Remote file host errors.
        assert!(is_local_uri("file://123.456.789.123/path").is_err());
    }

    #[test]
    fn prompt_source_blocks_local_paths() {
        assert!(validate_prompt_source("/etc/passwd").is_err());
        assert!(validate_prompt_source("file:///etc/passwd").is_err());
        assert!(validate_prompt_source("prompt-template").is_ok());
        assert!(validate_prompt_source("mlflow-artifacts:/p").is_ok());
        assert!(validate_prompt_source("https://h/p/../../../").is_err());
    }

    #[test]
    fn regex_gate_prefix_and_substring() {
        assert!(regex_matches(
            "^mlflow-artifacts:/.*$",
            "mlflow-artifacts:/models"
        ));
        assert!(!regex_matches(
            "^mlflow-artifacts:/.*$",
            "s3://path/to/model"
        ));
        assert!(regex_matches("models", "mlflow-artifacts:/models"));
    }
}
