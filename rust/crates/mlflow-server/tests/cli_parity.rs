//! CLI/env parity integration tests (plan T11.1).
//!
//! Spawns the built `mlflow-server` binary and asserts its argument-parsing
//! behaviour: `--help` succeeds and advertises the parity flags, unknown flags
//! are rejected by clap (exit 2), and the fail-loud parity cases
//! (`--app-name` other than `basic-auth`, a mismatched `--registry-store-uri`)
//! exit non-zero with a message naming the flag. The pure config-resolution
//! logic is unit-tested in `src/config.rs`; this file covers the process-level
//! contract that deploy scripts see.

use std::process::Command;

/// Path to the `mlflow-server` binary under test (Cargo sets `CARGO_BIN_EXE_*`
/// for integration tests of a binary crate).
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_mlflow-server")
}

/// Run the binary with `args` and an empty environment (so ambient
/// `MLFLOW_*` vars from the developer shell can't perturb the assertions),
/// returning (exit_code, stdout, stderr).
fn run(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .env_clear()
        .output()
        .expect("spawn mlflow-server");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn help_exits_zero_and_lists_parity_flags() {
    let (code, stdout, _stderr) = run(&["--help"]);
    assert_eq!(code, 0);
    for flag in [
        "--backend-store-uri",
        "--read-replica-backend-store-uri",
        "--registry-store-uri",
        "--default-artifact-root",
        "--serve-artifacts",
        "--no-serve-artifacts",
        "--artifacts-destination",
        "--artifacts-only",
        "--host",
        "--port",
        "--workers",
        "--static-prefix",
        "--allowed-hosts",
        "--cors-allowed-origins",
        "--x-frame-options",
        "--expose-prometheus",
        "--app-name",
        "--workspace-store-uri",
        "--enable-workspaces",
        "--disable-workspaces",
    ] {
        assert!(stdout.contains(flag), "help output missing {flag}");
    }
}

#[test]
fn unknown_flag_is_rejected() {
    let (code, _stdout, stderr) = run(&["--not-a-real-flag"]);
    // clap exits 2 on argument errors.
    assert_eq!(code, 2);
    assert!(stderr.contains("--not-a-real-flag") || stderr.contains("unexpected"));
}

#[test]
fn unsupported_app_name_fails_loudly() {
    let (code, _stdout, stderr) = run(&["--app-name", "wsgi-magic"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("--app-name") && stderr.contains("wsgi-magic"),
        "stderr did not name the unsupported --app-name value: {stderr}"
    );
}

#[test]
fn mismatched_registry_store_uri_fails_loudly() {
    let (code, _stdout, stderr) = run(&[
        "--backend-store-uri",
        "sqlite:///a.db",
        "--registry-store-uri",
        "postgresql://other/db",
    ]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("--registry-store-uri"),
        "stderr did not name --registry-store-uri: {stderr}"
    );
}

#[test]
fn invalid_static_prefix_fails_loudly() {
    // Missing leading slash: `_validate_static_prefix` parity.
    let (code, _stdout, stderr) = run(&["--static-prefix", "no-leading-slash"]);
    assert_ne!(code, 0);
    assert!(
        stderr.contains("--static-prefix"),
        "stderr did not name --static-prefix: {stderr}"
    );
}
