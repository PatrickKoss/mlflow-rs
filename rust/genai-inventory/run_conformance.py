"""Run the T15.5 Python test ledger against the Rust HTTP server.

The required profile is a small, dependency-light cross-section for gated CI.
The full profile runs every ledger-indexed suite and is intended for the nightly
and manual workflow. Both profiles use the same selector and report generator.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import requests

from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = HERE / "ledger.json"
DEFAULT_REPORT_DIR = ROOT / "rust" / "compliance" / "report"
CLASSIFICATIONS = ("server_reachable", "client_only")
BACKENDS = ("sqlite", "postgres")
_logger = logging.getLogger(__name__)

# These cases exercise the public SDK/store boundary without optional provider
# packages. Full reachability remains the nightly profile; keeping the required
# profile explicit makes additions reviewable and keeps PR latency predictable.
REQUIRED_TESTS = {
    "server_reachable": (
        "tests/genai/test_rust_http_conformance.py::test_evaluation_dataset_sdk_round_trip",
        "tests/genai/test_rust_http_conformance.py::test_label_schema_sdk_round_trip",
        "tests/genai/test_rust_http_conformance.py::test_review_queue_sdk_round_trip",
    ),
    "client_only": (
        "tests/genai/scorers/guardrails/test_utils.py::test_map_scorer_inputs_to_text",
        "tests/genai/scorers/guardrails/test_utils.py::test_map_scorer_inputs_to_text_requires_input_or_output",
        "tests/genai/simulators/test_utils.py::test_format_history",
        "tests/genai/simulators/test_utils.py::test_get_default_simulation_model_non_databricks",
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("required", "full"), default="required")
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--classification", action="append", choices=CLASSIFICATIONS)
    parser.add_argument(
        "--server-bin",
        type=Path,
        default=ROOT / "rust" / "target" / "release" / "mlflow-server",
    )
    parser.add_argument(
        "--postgres-uri",
        default=os.environ.get(
            "MLFLOW_RUST_CONFORMANCE_PG_URI",
            "postgresql://mlflow:mlflow@127.0.0.1:5432/mlflow",
        ),
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--pytest-arg", action="append", default=[])
    return parser.parse_args()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _provision(uri: str, artifact_root: Path) -> None:
    _run_checked(["uv", "run", "--no-sync", "mlflow", "db", "upgrade", uri])
    store = SqlAlchemyStore(uri, artifact_root.as_uri())
    store.engine.dispose()


@contextlib.contextmanager
def _server(server_bin: Path, backend: str, postgres_uri: str) -> Iterator[tuple[str, Path]]:
    with tempfile.TemporaryDirectory(prefix=f"mlflow-t22-2-{backend}-") as raw_tmp:
        tmp = Path(raw_tmp)
        backend_uri = f"sqlite:///{tmp / 'mlflow.db'}" if backend == "sqlite" else postgres_uri
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        artifact_root = tmp / "artifacts"
        artifact_root.mkdir()
        _provision(backend_uri, artifact_root)
        server_log = tmp / "server.log"
        env = {
            **os.environ,
            "MLFLOW_CRYPTO_KEK_PASSPHRASE": "t22-2-conformance-passphrase-at-least-32-characters",
        }
        command = [
            str(server_bin),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            backend_uri,
            "--default-artifact-root",
            artifact_root.as_uri(),
            "--serve-artifacts",
            "--artifacts-destination",
            artifact_root.as_uri(),
        ]
        with server_log.open("w") as output:
            process = subprocess.Popen(
                command,
                cwd=ROOT / "rust",
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Rust server exited with {process.returncode}:\n{server_log.read_text()}"
                    )
                try:
                    if requests.get(f"{url}/health", timeout=0.5).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError(f"Rust server did not become ready:\n{server_log.read_text()}")
            yield url, server_log
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def _selected_tests(
    ledger: dict[str, Any], profile: str, classification: str
) -> tuple[list[str], dict[str, int]]:
    indexed = [test for test in ledger["tests"] if test["classification"] == classification]
    indexed_ids = {test["id"] for test in indexed}
    if profile == "required":
        selected = list(REQUIRED_TESTS[classification])
        missing = set(selected) - indexed_ids
        if missing:
            raise RuntimeError(f"required selectors missing from ledger: {sorted(missing)}")
        counts = Counter(test_id.split("::", 1)[0] for test_id in selected)
        return selected, dict(counts)
    paths = sorted({test["path"] for test in indexed})
    counts = Counter(test["path"] for test in indexed)
    return paths, dict(counts)


def _tee_pytest(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def _junit_counts(
    path: Path, indexed_by_suite: dict[str, int]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    totals: Counter[str] = Counter()
    suites: dict[str, Counter[str]] = defaultdict(Counter)
    if not path.exists():
        return dict(totals), {}
    for case in ET.parse(path).iter("testcase"):
        result = "passed"
        if case.find("failure") is not None:
            result = "failed"
        elif case.find("error") is not None:
            result = "error"
        elif case.find("skipped") is not None:
            result = "skipped"
        suite = case.attrib.get("file")
        if suite is None:
            classname = case.attrib.get("classname", "unknown")
            suite = next(
                (
                    candidate
                    for candidate in indexed_by_suite
                    if classname == candidate.removesuffix(".py").replace("/", ".")
                    or classname.startswith(candidate.removesuffix(".py").replace("/", ".") + ".")
                ),
                classname,
            )
        totals[result] += 1
        suites[suite][result] += 1
    for counter in (totals, *suites.values()):
        for result in ("passed", "failed", "error", "skipped"):
            counter[result] += 0
    return dict(totals), {suite: dict(counts) for suite, counts in sorted(suites.items())}


def _run_one(
    *,
    args: argparse.Namespace,
    ledger: dict[str, Any],
    backend: str,
    classification: str,
    report_dir: Path,
) -> dict[str, Any]:
    selectors, indexed_by_suite = _selected_tests(ledger, args.profile, classification)
    stem = f"t22_2_{args.profile}_{classification}_{backend}"
    junit = report_dir / f"{stem}.xml"
    pytest_log = report_dir / f"{stem}.log"
    with _server(args.server_bin, backend, args.postgres_uri) as (url, server_log):
        env = {
            **os.environ,
            "MLFLOW_SERVER_TYPE": "rust",
            "MLFLOW_RUST_SERVER_BIN": str(args.server_bin),
            "MLFLOW_TRACKING_URI": url,
            "MLFLOW_REGISTRY_URI": url,
            "MLFLOW_CONFORMANCE_TRACKING_URI": url,
            "MLFLOW_CRYPTO_KEK_PASSPHRASE": "t22-2-conformance-passphrase-at-least-32-characters",
        }
        uv_args = ["uv", "run"]
        if args.profile == "full":
            uv_args.extend([
                "--with",
                "dspy==3.2.1,google-adk==2.4.0,google-cloud-aiplatform==1.160.0,"
                "guardrails-ai==0.10.2",
            ])
        command = [
            *uv_args,
            "--no-sync",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            f"--junitxml={junit}",
            *args.pytest_arg,
            *selectors,
        ]
        _logger.info(
            "T22.2 %s %s on %s: %d selectors",
            args.profile,
            classification,
            backend,
            len(selectors),
        )
        exit_code = _tee_pytest(command, env, pytest_log)
        server_log_copy = report_dir / f"{stem}.server.log"
        server_log_copy.write_text(server_log.read_text())
    totals, suites = _junit_counts(junit, indexed_by_suite)
    for suite, counts in suites.items():
        counts["ledger_test_definitions"] = indexed_by_suite.get(suite, 0)
    return {
        "backend": backend,
        "classification": classification,
        "exit_code": exit_code,
        "ledger_test_definitions": sum(indexed_by_suite.values()),
        "selector_count": len(selectors),
        "totals": totals,
        "suites": suites,
        "evidence": {
            "junit": junit.relative_to(ROOT).as_posix(),
            "pytest_log": pytest_log.relative_to(ROOT).as_posix(),
            "server_log": server_log_copy.relative_to(ROOT).as_posix(),
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# T22.2 Python-suite and reachability conformance",
        "",
        f"Profile: `{report['profile']}`. Ledger reference: `{report['ledger_reference']}`.",
        "",
        "The required profile is the dependency-light HTTP/SDK core used on every Rust CI run. "
        "The full repointable ledger matrix runs nightly and on manual dispatch; tests classified "
        "`python_internal` remain inventory evidence but cannot cross an HTTP process boundary.",
        "",
        "| Classification | Backend | Ledger tests | Passed | Failed | Errors | Skipped | Exit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        totals = run["totals"]
        lines.append(
            f"| {run['classification']} | {run['backend']} | "
            f"{run['ledger_test_definitions']} | {totals.get('passed', 0)} | "
            f"{totals.get('failed', 0)} | {totals.get('error', 0)} | "
            f"{totals.get('skipped', 0)} | {run['exit_code']} |"
        )
    lines.extend([
        "",
        "## Per-suite results",
        "",
        "| Suite | Classification | Backend | Ledger tests | Passed | Failed | Errors | Skipped |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in report["runs"]:
        for suite, counts in run["suites"].items():
            lines.append(
                f"| `{suite}` | {run['classification']} | {run['backend']} | "
                f"{counts['ledger_test_definitions']} | {counts['passed']} | "
                f"{counts['failed']} | {counts['error']} | {counts['skipped']} |"
            )
    lines.extend([
        "",
        "## Ledger invariants",
        "",
        f"- Server-reachable symbols/surfaces: {report['ledger_counts']['server_reachable']}",
        f"- Client-only symbols: {report['ledger_counts']['client_only']}",
        f"- Dead symbols: {report['ledger_counts']['dead']}",
        f"- Repointable server tests: {report['ledger_test_counts']['server_reachable']}",
        f"- Client-only SDK tests: {report['ledger_test_counts']['client_only']}",
        f"- Python-internal tests: {report['ledger_test_counts']['python_internal']}",
        "- Unclassified paths: 0",
        "- Server-reachable entries missing native owners: 0",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.server_bin.is_file():
        raise FileNotFoundError(f"release Rust server not found: {args.server_bin}")
    _run_checked(["uv", "run", "--no-sync", "python", str(HERE / "validate_ledger.py")])
    ledger = json.loads(LEDGER_PATH.read_text())
    args.report_dir.mkdir(parents=True, exist_ok=True)
    backends = args.backend or list(BACKENDS)
    classifications = args.classification or list(CLASSIFICATIONS)
    runs = [
        _run_one(
            args=args,
            ledger=ledger,
            backend=backend,
            classification=classification,
            report_dir=args.report_dir,
        )
        for backend in backends
        for classification in classifications
    ]
    report = {
        "schema_version": 1,
        "task": "T22.2",
        "profile": args.profile,
        "ledger_reference": ledger["reference"]["git_sha"],
        "ledger_counts": ledger["summary"]["classification_counts"],
        "ledger_test_counts": ledger["summary"]["test_classification_counts"],
        "runs": runs,
    }
    json_path = args.report_dir / f"t22_2_{args.profile}.json"
    md_path = args.report_dir / f"t22_2_{args.profile}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_render_markdown(report))
    _logger.info("Wrote %s and %s", json_path.relative_to(ROOT), md_path.relative_to(ROOT))
    return int(any(run["exit_code"] != 0 for run in runs))


if __name__ == "__main__":
    raise SystemExit(main())
