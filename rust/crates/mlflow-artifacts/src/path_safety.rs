//! Exact port of `mlflow.utils.uri.validate_path_is_safe` (+ its helpers
//! `_decode`, `_escape_control_characters`, `is_file_uri`,
//! `local_file_uri_to_path`) — the server's first line of defense against
//! artifact path-traversal attacks (§3.11 / T5.1).
//!
//! Every traversal attempt (`..`, encoded `..`, backslashes, absolute paths,
//! Windows drive letters, `file:` URIs escaping the root) must fail with the
//! exact Python error: `INVALID_PARAMETER_VALUE` / message `"Invalid path"`.
//!
//! ## Fidelity notes
//!
//! * Python's `_decode` loops `urllib.parse.unquote` up to 10 times until the
//!   string stops changing, wrapping each pass in `urlunparse(urlparse(...))`.
//!   For the artifact paths this validates (no scheme except `file:`, no
//!   fragment except the `#` we reject) that round-trip is an identity, so we
//!   replicate only the percent-decode-to-fixpoint loop and keep the 10-pass
//!   cap + "Failed to decode url" error semantics. Double/triple percent
//!   encodings (`%252e` → `%2e` → `.`) therefore collapse just like Python.
//! * `unquote` decodes `%XX` as UTF-8, replacing invalid sequences (Python's
//!   default `errors="replace"`), and leaves malformed `%` escapes untouched —
//!   we match both behaviors.
//! * The Windows-specific checks (`os.path.altsep == '\\'`, `pathlib`
//!   `PureWindowsPath.is_absolute`, drive-letter `path[1] == ':'`) are gated on
//!   `is_windows()`. This server targets a POSIX deployment, so `is_windows()`
//!   is `false`; we still port the checks that are unconditional on POSIX
//!   (backslash is NOT in `_OS_ALT_SEPS` on POSIX, but `PureWindowsPath` is
//!   consulted unconditionally, which DOES reject `\\`-rooted and drive-letter
//!   paths — see the code below).

use mlflow_error::MlflowError;

/// Mirrors `is_windows()` (`mlflow/utils/os.py`): `os.name == "nt"`. The Rust
/// server is POSIX-only for v1, matching the plan's deployment target, so this
/// is a compile-time `false`. Kept as a named function so the ported Windows
/// branches stay visible and are trivially flippable if a Windows build is ever
/// needed.
const fn is_windows() -> bool {
    cfg!(windows)
}

/// The `MlflowException("Invalid path", INVALID_PARAMETER_VALUE)` that every
/// rejection raises, verbatim.
fn invalid_path() -> MlflowError {
    MlflowError::invalid_parameter_value("Invalid path")
}

/// Port of `validate_path_is_safe`. On success returns the decoded+escaped path
/// (exactly what Python returns and then joins with the trusted root); on any
/// traversal signal returns the `"Invalid path"` error.
///
/// The one non-`Invalid path` failure mode Python has is `_decode` raising a
/// bare `ValueError("Failed to decode url")` after 10 non-converging passes;
/// that propagates as an uncaught 500 in Flask. We surface it as an
/// `INTERNAL_ERROR` `MlflowError` so the caller renders a 500 the same way (the
/// message differs from Python's traceback text, but Python's is a raw
/// `ValueError` with no MLflow JSON body either — a differential-harness
/// allowlist entry, documented in the final report).
pub fn validate_path_is_safe(path: &str) -> Result<String, MlflowError> {
    // We must decode path before validating it.
    let path = decode(path).map_err(|_| MlflowError::internal_error("Failed to decode url"))?;
    // If control characters are included in the path, escape them.
    let path = escape_control_characters(&path);

    if path.contains('#') {
        return Err(invalid_path());
    }

    let path = if is_file_uri(&path) {
        local_file_uri_to_path(&path)
    } else {
        path
    };

    if os_alt_sep_present(&path)
        || path.split('/').any(|seg| seg == "..")
        || is_windows_absolute(&path)
        || is_posix_absolute(&path)
        || (is_windows() && path.len() >= 2 && path.as_bytes()[1] == b':')
    {
        return Err(invalid_path());
    }

    Ok(path)
}

/// Port of `_OS_ALT_SEPS` membership: `[sep for sep in [os.sep, os.path.altsep]
/// if sep is not None and sep != "/"]`. On POSIX `os.sep == "/"` and
/// `os.path.altsep is None`, so the list is EMPTY — backslashes are NOT caught
/// here on POSIX (they are caught by the Windows-path absolute check only when
/// they make the path absolute). On Windows `os.sep == "\\"`, so `"\\"` is a
/// forbidden separator anywhere in the path.
fn os_alt_sep_present(path: &str) -> bool {
    if is_windows() {
        path.contains('\\')
    } else {
        false
    }
}

/// Mirrors `pathlib.PurePosixPath(path).is_absolute()` — true iff the path
/// starts with `/`.
fn is_posix_absolute(path: &str) -> bool {
    path.starts_with('/')
}

/// Mirrors `pathlib.PureWindowsPath(path).is_absolute()`, consulted
/// UNCONDITIONALLY by Python (not gated on `is_windows()`). A Windows path is
/// absolute when it has BOTH a drive/root, i.e. it is one of a drive-absolute
/// path (`C:\foo` or `C:/foo` — drive letter + separator) or a UNC path
/// (`\\server\share` or `//server/share`), but NOT a drive-relative `C:foo` nor
/// a rootless-but-drived `C:`.
///
/// A leading-separator-only path (`\foo`, `/foo`) is NOT absolute to
/// `PureWindowsPath` (it lacks a drive) — so `/foo` is caught by the POSIX
/// check, not this one. This matches CPython's `ntpath.splitroot` semantics for
/// the inputs this validator sees.
fn is_windows_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();

    // UNC: two leading separators (either kind) followed by a non-separator
    // (the server name). `\\`, `//`, `\/`, `/\` all count as the UNC prefix.
    if bytes.len() >= 3 {
        let s0 = bytes[0] == b'\\' || bytes[0] == b'/';
        let s1 = bytes[1] == b'\\' || bytes[1] == b'/';
        let s2 = bytes[2] == b'\\' || bytes[2] == b'/';
        if s0 && s1 && !s2 {
            return true;
        }
    }

    // Drive-absolute: `X:` followed by a separator.
    if bytes.len() >= 3 {
        let drive = bytes[0];
        let is_drive_letter = drive.is_ascii_alphabetic();
        let colon = bytes[1] == b':';
        let sep = bytes[2] == b'\\' || bytes[2] == b'/';
        if is_drive_letter && colon && sep {
            return true;
        }
    }

    false
}

/// Port of `is_file_uri`: `urlparse(uri).scheme == "file"`. A URL scheme is
/// `ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )` followed by `:`, per RFC 3986
/// and CPython's `urlsplit`. We replicate that scheme extraction precisely so
/// e.g. `file:foo`, `file:///x` are recognized but `/file:x` is not.
fn is_file_uri(uri: &str) -> bool {
    url_scheme(uri).as_deref() == Some("file")
}

/// Extract the URL scheme the way CPython's `urllib.parse.urlsplit` does:
/// letters/digits/`+`/`-`/`.` before the first `:`, with a leading ASCII
/// letter, lowercased. Returns `None` if there is no valid scheme.
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

/// Port of `local_file_uri_to_path` for the non-Windows path (the server is
/// POSIX): `url2pathname(urlparse(uri).path)`. For a `file:` URI, take the path
/// component (after the scheme and any `//netloc`) and percent-decode it (that
/// is what `nturl2path`/`urllib.request.url2pathname` does on POSIX — it is a
/// straight `unquote`). Non-`file:` inputs never reach here.
fn local_file_uri_to_path(uri: &str) -> String {
    let path = url_path_component(uri);
    // POSIX `url2pathname` == `unquote` (single pass).
    percent_decode(&path)
}

/// Extract the `path` component of a URI the way `urlparse` does, for the
/// subset we care about (`file:` URIs). Strips the scheme, then an optional
/// `//authority`, and drops any `?query`/`#fragment` (there is no `#` here — it
/// was already rejected).
fn url_path_component(uri: &str) -> String {
    let after_scheme = match uri.find(':') {
        Some(i) => &uri[i + 1..],
        None => uri,
    };
    // Optional authority.
    let rest = if let Some(stripped) = after_scheme.strip_prefix("//") {
        // Path begins at the next '/', or is empty if none.
        match stripped.find('/') {
            Some(i) => &stripped[i..],
            None => "",
        }
    } else {
        after_scheme
    };
    // Trim query/fragment.
    let end = rest.find(['?', '#']).unwrap_or(rest.len());
    rest[..end].to_string()
}

/// Port of `_decode`: percent-decode to a fixpoint, at most 10 passes; error if
/// it never converges. (The `urlunparse(urlparse(...))` wrapper is an identity
/// for these inputs, as documented in the module header.)
fn decode(url: &str) -> Result<String, ()> {
    let mut url = url.to_string();
    for _ in 0..10 {
        let decoded = percent_decode(&url);
        if decoded == url {
            return Ok(url);
        }
        url = decoded;
    }
    Err(())
}

/// Port of `urllib.parse.unquote(url)` with the default `encoding="utf-8"`,
/// `errors="replace"`: split on `%`, decode each `%XX` run as UTF-8 bytes,
/// replacing invalid byte sequences with U+FFFD, and leave any `%` not followed
/// by two hex digits untouched.
fn percent_decode(s: &str) -> String {
    // Fast path: no `%` means nothing to decode.
    if !s.contains('%') {
        return s.to_string();
    }

    // CPython's unquote splits the string on '%'; the first chunk is literal,
    // then each subsequent chunk starts with 2 chars that (if hex) are a byte.
    // Consecutive %XX escapes are decoded as one UTF-8 byte sequence so that a
    // multi-byte char split across escapes (e.g. `%E2%80%A6`) round-trips.
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    // Buffer of raw decoded bytes to flush as one UTF-8 (lossy) run.
    let mut byte_buf: Vec<u8> = Vec::new();

    let flush = |byte_buf: &mut Vec<u8>, out: &mut String| {
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(byte_buf));
            byte_buf.clear();
        }
    };

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() + 1 && i + 2 <= bytes.len() {
            // Need two hex digits.
            if i + 2 < bytes.len() + 1 && i + 2 <= bytes.len() {
                let h1 = bytes.get(i + 1).copied();
                let h2 = bytes.get(i + 2).copied();
                if let (Some(a), Some(b)) = (h1, h2) {
                    if let (Some(hi), Some(lo)) = (hex_val(a), hex_val(b)) {
                        byte_buf.push((hi << 4) | lo);
                        i += 3;
                        continue;
                    }
                }
            }
            // Malformed escape: emit the '%' literally and continue.
            flush(&mut byte_buf, &mut out);
            out.push('%');
            i += 1;
        } else {
            flush(&mut byte_buf, &mut out);
            // Copy one UTF-8 char starting at i.
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

/// Port of `_escape_control_characters`: replace ASCII control chars (0-31, 127)
/// with `%xx` (lowercase hex), leave everything else untouched.
fn escape_control_characters(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let cp = c as u32;
        if cp <= 31 || cp == 127 {
            out.push_str(&format!("%{cp:02x}"));
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlflow_error::ErrorCode;

    fn assert_rejected(path: &str) {
        let err = validate_path_is_safe(path)
            .unwrap_err_or_else(|| panic!("expected rejection for {path:?}"));
        assert_eq!(
            err.error_code,
            ErrorCode::InvalidParameterValue,
            "path={path:?}"
        );
        assert_eq!(err.message, "Invalid path", "path={path:?}");
    }

    trait ResultExt<T> {
        fn unwrap_err_or_else(self, f: impl FnOnce() -> MlflowError) -> MlflowError;
    }
    impl<T> ResultExt<T> for Result<T, MlflowError> {
        fn unwrap_err_or_else(self, f: impl FnOnce() -> MlflowError) -> MlflowError {
            match self {
                Ok(_) => f(),
                Err(e) => e,
            }
        }
    }

    #[test]
    fn accepts_plain_relative_paths() {
        assert_eq!(validate_path_is_safe("a/b/c.txt").unwrap(), "a/b/c.txt");
        assert_eq!(
            validate_path_is_safe("model/MLmodel").unwrap(),
            "model/MLmodel"
        );
        assert_eq!(validate_path_is_safe("file.txt").unwrap(), "file.txt");
        assert_eq!(validate_path_is_safe("").unwrap(), "");
        // A dot segment other than `..` is allowed by this validator (Python
        // does not normalize here — normalization happens later at join time).
        assert_eq!(validate_path_is_safe("a/./b").unwrap(), "a/./b");
        assert_eq!(validate_path_is_safe("...foo").unwrap(), "...foo");
        assert_eq!(validate_path_is_safe("foo..bar").unwrap(), "foo..bar");
    }

    #[test]
    fn rejects_dotdot_traversal() {
        assert_rejected("..");
        assert_rejected("../etc/passwd");
        assert_rejected("a/../../b");
        assert_rejected("a/b/..");
        assert_rejected("../");
    }

    #[test]
    fn rejects_percent_encoded_dotdot() {
        // %2e%2e == ".."
        assert_rejected("%2e%2e/etc/passwd");
        assert_rejected("%2E%2E/x");
        assert_rejected("a/%2e%2e/b");
        // Encoded slash + dots: %2e%2e%2f == "../"
        assert_rejected("%2e%2e%2fetc");
    }

    #[test]
    fn rejects_double_encoded_dotdot() {
        // %252e -> %2e -> "." ; %252e%252e -> ".."
        assert_rejected("%252e%252e/etc/passwd");
        // Triple encoding collapses within the 10-pass loop.
        assert_rejected("%25252e%25252e/x");
    }

    #[test]
    fn rejects_absolute_posix_paths() {
        assert_rejected("/etc/passwd");
        assert_rejected("/");
        assert_rejected("//server/share");
    }

    #[test]
    fn rejects_windows_absolute_and_drive_letters() {
        // Drive-absolute (PureWindowsPath.is_absolute is consulted on POSIX too).
        assert_rejected("C:\\Windows\\System32");
        assert_rejected("C:/Windows");
        // UNC.
        assert_rejected("\\\\server\\share");
        // Backslash-encoded drive-absolute.
        assert_rejected("%43%3a%5cwindows");
    }

    #[test]
    fn rejects_fragment() {
        assert_rejected("a#b");
        assert_rejected("model#../etc");
    }

    #[test]
    fn rejects_file_uri_escaping_root() {
        // file: URI resolving to an absolute path.
        assert_rejected("file:///etc/passwd");
        // file: URI with traversal.
        assert_rejected("file:../../etc");
    }

    #[test]
    fn escapes_null_and_control_bytes_then_treats_as_relative() {
        // A raw NUL becomes "%00"; no traversal signal, so it's accepted and
        // returned escaped (matching Python, which escapes control chars before
        // validating — the join layer later rejects it if it hits the FS).
        let out = validate_path_is_safe("a\u{0}b").unwrap();
        assert_eq!(out, "a%00b");
        // A percent-encoded NUL decodes to a NUL then re-escapes to "%00".
        let out = validate_path_is_safe("a%00b").unwrap();
        assert_eq!(out, "a%00b");
    }

    #[test]
    fn backslash_on_posix_is_not_a_separator_but_absolute_forms_are_caught() {
        // On POSIX a bare backslash is a normal filename char (not in
        // _OS_ALT_SEPS), so a relative path containing one is accepted...
        assert_eq!(validate_path_is_safe("a\\b").unwrap(), "a\\b");
        // ...but a backslash that forms a Windows drive/UNC absolute path is
        // caught by the unconditional PureWindowsPath check.
        assert_rejected("\\\\evil\\share");
    }

    #[test]
    fn decode_collapses_encoded_slashes() {
        // %2f -> "/"; the path stays relative so it's accepted, decoded.
        assert_eq!(validate_path_is_safe("a%2fb").unwrap(), "a/b");
    }
}
