"""Byte-compare Python and Rust trace-archival startup validation errors."""

from __future__ import annotations

import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from mlflow.cli import server
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository
from mlflow.tracing.trace_archival_config import load_trace_archival_server_config
from mlflow.utils.server_cli_utils import artifacts_only_config_validation

ROOT = Path(__file__).resolve().parents[2]
RUST_ROOT = ROOT / "rust"
BINARY = RUST_ROOT / "target" / "debug" / "mlflow-server"

BASE = """trace_archival:
  enabled: true
  location: file:///tmp/archive
  retention: 30d
"""

CASES = {
    "malformed_yaml": "trace_archival: [\n",
    "malformed_flow_mapping": (
        "trace_archival: {enabled: true, location: file:///tmp/archive, retention: 30d\n"
    ),
    "malformed_quoted_scalar": BASE.replace("30d", '"unterminated'),
    "malformed_tab": "trace_archival:\n\t enabled: true\n",
    "unknown_yaml_tag": "trace_archival: !!python/object:example {}\n",
    "top_level_not_mapping": "null\n",
    "missing_trace_archival": "{}\n",
    "bad_enabled": "trace_archival:\n  enabled: value\n",
    "null_location": BASE.replace("file:///tmp/archive", "null"),
    "proxy_location": BASE.replace("file:///tmp/archive", "mlflow-artifacts:/archive"),
    "bad_retention": BASE.replace("30d", "30days"),
    "bad_allowlist": BASE + "  long_retention_allowlist: [invalid id]\n",
    "bad_interval": BASE + "  interval_seconds: 0\n",
    "interval_too_large": BASE + "  interval_seconds: 86401\n",
    "bad_max_traces": BASE + "  max_traces_per_pass: false\n",
    "dbfs_no_delete": BASE.replace("file:///tmp/archive", "dbfs:/archive"),
    "databricks_no_reads": BASE.replace(
        "file:///tmp/archive", "dbfs:/databricks/mlflow-tracking/1/run/artifacts"
    ),
}


def _python_message(case: str, path: Path) -> str:
    patch = nullcontext()
    if case == "dbfs_no_delete":
        repo = object.__new__(DbfsRestArtifactRepository)
        patch = mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        )
    elif case == "databricks_no_reads":
        repo = object.__new__(DatabricksArtifactRepository)
        patch = mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        )

    with patch:
        try:
            load_trace_archival_server_config(path)
        except Exception as error:
            return error.message
    raise AssertionError(f"Python unexpectedly accepted {case}")


def _rust_message(*args: str) -> str:
    result = subprocess.run(
        [str(BINARY), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(f"Rust unexpectedly started for arguments {args!r}")
    prefix = "Error: "
    if not result.stderr.startswith(prefix):
        raise AssertionError(f"Unexpected Rust stderr: {result.stderr!r}")
    return result.stderr.removeprefix(prefix).removesuffix("\n")


def _python_cli_message(*args: str) -> str:
    result = CliRunner().invoke(server, list(args))
    if result.exit_code == 0 or "Error: " not in result.output:
        raise AssertionError(f"Unexpected Python CLI result: {result.output!r}")
    return result.output.split("Error: ", maxsplit=1)[1].removesuffix("\n")


def main() -> None:
    subprocess.run(
        ["cargo", "build", "-p", "mlflow-server", "--bin", "mlflow-server"],
        cwd=RUST_ROOT,
        check=True,
    )

    compared = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for case, payload in CASES.items():
            path = root / f"{case}.yaml"
            path.write_text(payload, encoding="utf-8")
            python_message = _python_message(case, path)
            rust_message = _rust_message("--trace-archival-config", str(path))
            if rust_message != python_message:
                raise AssertionError(
                    f"{case} mismatch:\nPython: {python_message!r}\nRust:   {rust_message!r}"
                )
            compared += 1

        missing = root / "missing.yaml"
        python_message = _python_cli_message("--trace-archival-config", str(missing))
        rust_message = _rust_message("--trace-archival-config", str(missing))
        if rust_message != python_message:
            raise AssertionError(
                f"missing_file mismatch:\nPython: {python_message!r}\nRust:   {rust_message!r}"
            )
        compared += 1

        python_message = _python_cli_message("--trace-archival-config", str(root))
        rust_message = _rust_message("--trace-archival-config", str(root))
        if rust_message != python_message:
            raise AssertionError(
                f"directory mismatch:\nPython: {python_message!r}\nRust:   {rust_message!r}"
            )
        compared += 1

        conflict = root / "valid.yaml"
        conflict.write_text(BASE, encoding="utf-8")
        try:
            artifacts_only_config_validation(
                True,
                "sqlite:///mlflow.db",
                trace_archival_config_path=str(conflict),
            )
        except Exception as error:
            python_message = error.message
        else:
            raise AssertionError("Python unexpectedly accepted artifacts-only conflict")
        rust_message = _rust_message("--artifacts-only", "--trace-archival-config", str(conflict))
        if rust_message != python_message:
            raise AssertionError(
                "artifacts_only_conflict mismatch:\n"
                f"Python: {python_message!r}\nRust:   {rust_message!r}"
            )
        compared += 1

    print(f"trace archival config error parity: {compared}/{compared} byte-identical")


if __name__ == "__main__":
    main()
