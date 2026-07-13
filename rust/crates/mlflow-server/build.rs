//! Build script: bakes the MLflow version string into the binary at compile
//! time by parsing `VERSION = "..."` out of `mlflow/version.py` (the Python
//! package's single source of truth). This lets `GET /version` mirror
//! `mlflow/server/__init__.py`'s `version()` handler without any runtime
//! file access.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // rust/crates/mlflow-server -> repo root is three levels up.
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("mlflow-server crate should be nested under <repo_root>/rust/crates");
    let version_py = repo_root.join("mlflow").join("version.py");

    println!("cargo::rerun-if-changed={}", version_py.display());

    let contents = fs::read_to_string(&version_py)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", version_py.display()));

    let version = parse_version(&contents).unwrap_or_else(|| {
        panic!(
            "no `VERSION = \"...\"` line found in {}",
            version_py.display()
        )
    });

    println!("cargo::rustc-env=MLFLOW_VERSION={version}");
}

/// Parses the `VERSION = "3.14.1.dev0"` line out of `mlflow/version.py`'s
/// source text.
fn parse_version(contents: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("VERSION")?.trim_start();
        let rest = rest.strip_prefix('=')?.trim_start();
        let rest = rest.strip_prefix('"')?;
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::parse_version;

    #[test]
    fn parses_version_line() {
        let src = "import re\n\nVERSION = \"3.14.1.dev0\"\n\ndef foo():\n    pass\n";
        assert_eq!(parse_version(src).as_deref(), Some("3.14.1.dev0"));
    }

    #[test]
    fn returns_none_when_missing() {
        assert_eq!(parse_version("no version here\n"), None);
    }
}
