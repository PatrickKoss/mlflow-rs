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
}
