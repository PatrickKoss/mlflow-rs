//! Minimal URI path joining, mirroring `append_to_uri_path`
//! (`mlflow/utils/uri.py`) for the two shapes the store needs:
//! `<root>/<experiment_id>` (default experiment artifact location) and
//! `<experiment_artifact_location>/<run_id>/artifacts` (run artifact URI).
//!
//! MLflow's `append_to_uri_path` parses the URI, joins POSIX path components,
//! and reassembles. For the store we only ever append plain, single-segment
//! POSIX paths (an integer id, a uuid hex, the literal "artifacts") to an
//! already-well-formed URI, so a faithful subset suffices: split off any
//! `?query`/`#fragment`, join the path with `/` (collapsing duplicate slashes at
//! the seam), and reattach. This matches Python's observable output for these
//! inputs, including schemed URIs like `s3://bucket/root` and `file:///abs` and
//! bare POSIX roots like `/tmp/mlruns`.

/// Append POSIX `segments` to the path component of `uri`.
///
/// Each segment is a plain path chunk (no leading/trailing slashes expected,
/// though leading slashes are tolerated and collapsed at the seam).
pub(crate) fn append_to_uri_path(uri: &str, segments: &[&str]) -> String {
    // Preserve any query/fragment on the *base* URI (rare for artifact roots,
    // but `append_to_uri_path` keeps them). Split on the first '?' or '#'.
    let (base, suffix) = match uri.find(['?', '#']) {
        Some(i) => (&uri[..i], &uri[i..]),
        None => (uri, ""),
    };

    let mut path = base.trim_end_matches('/').to_string();
    for seg in segments {
        let seg = seg.trim_matches('/');
        if seg.is_empty() {
            continue;
        }
        path.push('/');
        path.push_str(seg);
    }
    format!("{path}{suffix}")
}

/// Split a URI into its scheme and the remainder (everything after `scheme:`),
/// mirroring Python's `urllib.parse.urlsplit` scheme detection. Returns
/// `("", uri)` when there is no valid scheme. A scheme must start with an ASCII
/// letter and contain only letters, digits, `+`, `-`, `.` before the `:`.
fn split_scheme(uri: &str) -> (&str, &str) {
    let Some(colon) = uri.find(':') else {
        return ("", uri);
    };
    let candidate = &uri[..colon];
    let mut chars = candidate.chars();
    let valid = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if valid {
        (candidate, &uri[colon + 1..])
    } else {
        ("", uri)
    }
}

/// Whether `uri` refers to a local filesystem path, mirroring
/// `mlflow.utils.uri.is_local_uri` for the non-Windows server case: an empty
/// scheme is local, a `file` scheme is local, everything else is remote.
fn is_local_uri(uri: &str) -> bool {
    matches!(split_scheme(uri).0, "" | "file")
}

/// Convert a `file:`-schemed URI to its filesystem path, mirroring
/// `local_file_uri_to_path` for the shapes an artifact-location argument takes:
/// `file:///abs`, `file://host/abs`, `file:rel`, `file:/abs`. Non-`file` inputs
/// are returned unchanged.
fn local_file_uri_to_path(uri: &str) -> &str {
    let (scheme, rest) = split_scheme(uri);
    if scheme != "file" {
        return uri;
    }
    // Strip a leading `//authority` if present, keeping the path.
    match rest.strip_prefix("//") {
        Some(after) => match after.find('/') {
            Some(slash) => &after[slash..],
            None => "",
        },
        None => rest,
    }
}

/// Resolve a relative local `artifact_location` to an absolute path relative to
/// the process working directory, mirroring `mlflow.utils.uri.resolve_uri_if_local`
/// for the non-Windows server path (`create_experiment` applies this before
/// persisting, `sqlalchemy_store.py:554`). Absolute paths, remote URIs, and
/// `None` pass through unchanged; a relative bare path becomes an absolute POSIX
/// path, and a relative `<scheme>:` URI keeps its scheme with an absolutized path.
pub(crate) fn resolve_uri_if_local(local_uri: Option<&str>) -> Option<String> {
    let uri = local_uri?;
    if !is_local_uri(uri) {
        return Some(uri.to_string());
    }
    let (scheme, _) = split_scheme(uri);
    let local_path = local_file_uri_to_path(uri);
    if std::path::Path::new(local_path).is_absolute() {
        return Some(uri.to_string());
    }
    let cwd = std::env::current_dir().ok()?;
    let joined = cwd.join(local_path);
    let joined = joined.to_string_lossy().replace('\\', "/");
    if scheme.is_empty() {
        Some(joined)
    } else {
        // `is_local_uri` only admits the empty scheme (handled above) or `file`,
        // which is in Python's `uses_netloc`, so `urlunsplit` emits `file://` +
        // the absolute path (`file:///abs`). Preserve any query/fragment.
        let suffix = match uri.find(['?', '#']) {
            Some(i) => &uri[i..],
            None => "",
        };
        Some(format!("{scheme}://{joined}{suffix}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_bare_posix_root() {
        assert_eq!(append_to_uri_path("/tmp/mlruns", &["0"]), "/tmp/mlruns/0");
        assert_eq!(
            append_to_uri_path("/tmp/mlruns/", &["abc123", "artifacts"]),
            "/tmp/mlruns/abc123/artifacts"
        );
    }

    #[test]
    fn joins_schemed_uri() {
        assert_eq!(
            append_to_uri_path("s3://bucket/root", &["5"]),
            "s3://bucket/root/5"
        );
        assert_eq!(
            append_to_uri_path("file:///abs/path", &["run", "artifacts"]),
            "file:///abs/path/run/artifacts"
        );
    }

    #[test]
    fn preserves_query() {
        assert_eq!(
            append_to_uri_path("s3://bucket/root?x=1", &["7"]),
            "s3://bucket/root/7?x=1"
        );
    }

    #[test]
    fn collapses_slashes_at_seam() {
        assert_eq!(
            append_to_uri_path("s3://bucket/root/", &["/7/"]),
            "s3://bucket/root/7"
        );
    }

    #[test]
    fn resolve_passthrough_for_absolute_and_remote() {
        assert_eq!(resolve_uri_if_local(None), None);
        assert_eq!(
            resolve_uri_if_local(Some("/abs/path")),
            Some("/abs/path".to_string())
        );
        assert_eq!(
            resolve_uri_if_local(Some("file:///abs/path")),
            Some("file:///abs/path".to_string())
        );
        assert_eq!(
            resolve_uri_if_local(Some("s3://bucket/root")),
            Some("s3://bucket/root".to_string())
        );
    }

    #[test]
    fn resolve_bare_relative_becomes_absolute() {
        let cwd = std::env::current_dir().unwrap();
        let expected = cwd.join("my_location").to_string_lossy().replace('\\', "/");
        assert_eq!(resolve_uri_if_local(Some("my_location")), Some(expected));
    }

    #[test]
    fn resolve_file_scheme_relative_becomes_absolute_file_uri() {
        let cwd = std::env::current_dir().unwrap();
        let joined = cwd.join("rel/loc").to_string_lossy().replace('\\', "/");
        assert_eq!(
            resolve_uri_if_local(Some("file:rel/loc")),
            Some(format!("file://{joined}"))
        );
    }
}
