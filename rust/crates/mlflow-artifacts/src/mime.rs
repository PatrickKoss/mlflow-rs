//! Port of MLflow's artifact download content-type / content-disposition logic
//! (`mlflow/utils/mime_type_utils.py::_guess_mime_type`,
//! `handlers.py::_content_disposition_attachment` /
//! `_response_with_file_attachment_headers`).

use percent_encoding::{AsciiSet, CONTROLS};

/// The MLflow text-extension allowlist (`get_text_extensions`) that force
/// `text/plain`. Includes the extensionless `MLmodel` / `MLproject` sentinel
/// filenames (only present in the non-tracing-SDK build, which the server is).
const TEXT_EXTENSIONS: &[&str] = &[
    "txt",
    "log",
    "err",
    "cfg",
    "conf",
    "cnf",
    "cf",
    "ini",
    "properties",
    "prop",
    "hocon",
    "toml",
    "yaml",
    "yml",
    "xml",
    "json",
    "js",
    "py",
    "py3",
    "csv",
    "tsv",
    "md",
    "rst",
    "MLmodel",
    "MLproject",
];

/// Port of `_guess_mime_type(file_path)`:
///  1. Take the basename.
///  2. Extension = chars after the last `.` (empty if none) — then if empty,
///     fall back to the whole filename (handles extensionless `MLmodel`).
///  3. If that token is in the text-extension allowlist → `text/plain`.
///  4. Otherwise `mimetypes.guess_type(filename)`; if it returns nothing →
///     `application/octet-stream`.
///
/// The extension comparison is case-SENSITIVE, exactly like Python's `in`
/// against the list (so `FILE.TXT` does NOT match `txt` — mirrors CPython's
/// `os.path.splitext` + membership test, which is what MLflow relies on).
pub fn guess_mime_type(file_path: &str) -> String {
    let filename = basename(file_path);

    // `os.path.splitext(filename)[-1].replace(".", "")`: the extension is the
    // suffix from the last dot (splitext ignores a leading dot), sans the dot.
    let extension = splitext_ext(filename);
    let extension = if extension.is_empty() {
        filename
    } else {
        extension
    };

    if TEXT_EXTENSIONS.contains(&extension) {
        return "text/plain".to_string();
    }

    // `mimetypes.guess_type` keys off the lowercased extension. `mime_guess`
    // does the same and covers the stdlib's default type map.
    match mime_guess::from_path(filename).first_raw() {
        Some(mime) => mime.to_string(),
        None => "application/octet-stream".to_string(),
    }
}

/// Basename via the last `/` (paths here are always posix-normalized).
fn basename(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((_, name)) => name,
        None => path,
    }
}

/// Port of `os.path.splitext(name)[-1].replace(".", "")`: everything after the
/// last dot, but a leading dot (dotfile) is NOT treated as an extension
/// separator. Returns `""` when there is no extension.
fn splitext_ext(filename: &str) -> &str {
    // Find the last '.' that is not at position 0 and is preceded by a
    // non-dot / non-empty base (matching CPython's `genericpath._splitext`,
    // which skips leading dots).
    let bytes = filename.as_bytes();
    // Leading dots are part of the name, not an extension separator.
    let mut start = 0;
    while start < bytes.len() && bytes[start] == b'.' {
        start += 1;
    }
    match filename[start..].rfind('.') {
        Some(rel) => &filename[start + rel + 1..],
        None => "",
    }
}

// RFC 5987 `attr-char` set: the chars that may appear unescaped in `filename*`.
// Everything else is percent-encoded. `safe="!#$&+-.^_`|~"` in the Python code.
const RFC5987_SAFE: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'%')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*')
    .add(b',')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'<')
    .add(b'=')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'{')
    .add(b'}');

/// Whether `c` is a werkzeug HTTP token character (`werkzeug.http._token_chars`):
/// alphanumerics plus `!#$%&'*+-.^_`|~`. A value made entirely of these is
/// emitted by `quote_header_value(allow_token=True)` without surrounding quotes.
fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "!#$%&'*+-.^_`|~".contains(c)
}

/// `werkzeug.http.quote_header_value(value, allow_token=True)`: an all-token,
/// non-empty value is returned unchanged; otherwise it is wrapped in double
/// quotes with `\` and `"` backslash-escaped.
fn quote_header_value(value: &str) -> String {
    if !value.is_empty() && value.chars().all(is_token_char) {
        return value.to_string();
    }
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

/// Port of `_content_disposition_attachment(filename)` (`handlers.py:1093`). For
/// ASCII filenames, emits `attachment; filename=<quoted>` where `<quoted>` uses
/// `quote_header_value(allow_token=True)` — an unquoted token (e.g.
/// `traces.json`) or a `"..."` quoted string when the name has non-token chars.
/// For non-ASCII names, emits an ASCII fallback plus an RFC 5987
/// `filename*=UTF-8''<pct-encoded>` parameter.
pub fn content_disposition_attachment(filename: &str) -> String {
    if filename.is_ascii() {
        return format!("attachment; filename={}", quote_header_value(filename));
    }

    // Non-ASCII: NFKD-strip to an ASCII fallback (best-effort — we do a plain
    // ASCII filter since full Unicode NFKD is not needed for correctness here;
    // clients that understand `filename*` use it, others get a usable name).
    let ascii_fallback: String = filename.chars().filter(char::is_ascii).collect();
    let ascii_fallback = if ascii_fallback.is_empty() {
        "download".to_string()
    } else {
        ascii_fallback
    };

    let quoted = percent_encoding::utf8_percent_encode(filename, RFC5987_SAFE).to_string();
    format!(
        "attachment; filename={}; filename*=UTF-8''{quoted}",
        quote_header_value(&ascii_fallback)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_extensions_force_text_plain() {
        assert_eq!(guess_mime_type("foo.txt"), "text/plain");
        assert_eq!(guess_mime_type("a/b/config.yaml"), "text/plain");
        assert_eq!(guess_mime_type("model/MLmodel"), "text/plain");
        assert_eq!(guess_mime_type("MLproject"), "text/plain");
        assert_eq!(guess_mime_type("data.json"), "text/plain");
    }

    #[test]
    fn known_binary_types() {
        assert_eq!(guess_mime_type("image.png"), "image/png");
        assert_eq!(guess_mime_type("a/b/pic.jpg"), "image/jpeg");
        assert_eq!(guess_mime_type("doc.pdf"), "application/pdf");
    }

    #[test]
    fn unknown_extension_is_octet_stream() {
        assert_eq!(guess_mime_type("model.pkl"), "application/octet-stream");
        assert_eq!(guess_mime_type("weights.bin"), "application/octet-stream");
        assert_eq!(guess_mime_type("noext"), "application/octet-stream");
    }

    #[test]
    fn ascii_token_filename_is_unquoted() {
        // Matches `_content_disposition_attachment` (verified live): an all-token
        // filename is emitted without surrounding quotes.
        assert_eq!(
            content_disposition_attachment("report.txt"),
            "attachment; filename=report.txt"
        );
        assert_eq!(
            content_disposition_attachment("traces.json"),
            "attachment; filename=traces.json"
        );
    }

    #[test]
    fn ascii_non_token_filename_is_quoted_and_escaped() {
        assert_eq!(
            content_disposition_attachment("my file.txt"),
            "attachment; filename=\"my file.txt\""
        );
        assert_eq!(
            content_disposition_attachment("a\"b.txt"),
            "attachment; filename=\"a\\\"b.txt\""
        );
    }

    #[test]
    fn non_ascii_content_disposition_has_filename_star() {
        // Fallback `.txt` is all-token, so it is unquoted, matching Python.
        let cd = content_disposition_attachment("日本語.txt");
        assert_eq!(
            cd,
            "attachment; filename=.txt; filename*=UTF-8''%E6%97%A5%E6%9C%AC%E8%AA%9E.txt"
        );
    }
}
