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

    generate_model_catalog(repo_root);
}

/// Generate a deterministically ordered table of bundled provider catalogs.
/// Filesystem directory iteration order varies between checkouts, while the
/// provider order is observable when all models are returned.
fn generate_model_catalog(repo_root: &Path) {
    let catalog_dir = repo_root.join("mlflow/utils/model_catalog");
    println!("cargo::rerun-if-changed={}", catalog_dir.display());

    let mut generated = String::from("pub static BUNDLED_MODEL_CATALOGS: &[(&str, &str)] = &[\n");
    let mut paths = fs::read_dir(&catalog_dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", catalog_dir.display()))
        .map(|entry| entry.expect("model catalog directory entry").path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort_unstable_by(|left, right| left.file_stem().cmp(&right.file_stem()));
    for path in paths {
        let provider = path
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("model catalog filename is UTF-8");
        println!("cargo::rerun-if-changed={}", path.display());
        generated.push_str(&format!(
            "    ({provider:?}, include_str!({:?})),\n",
            path.to_string_lossy()
        ));
    }
    generated.push_str("];\n");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let output = out_dir.join("model_catalog.rs");
    fs::write(&output, generated)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", output.display()));
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
