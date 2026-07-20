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
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import requests
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = HERE / "ledger.json"
DEFAULT_REPORT_DIR = ROOT / "rust" / "compliance" / "report"
CLASSIFICATIONS = ("server_reachable", "client_only")
BACKENDS = ("sqlite", "postgres")
SERVER_IMPLEMENTATIONS = ("python_http", "rust")
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
    parser.add_argument("--server-implementation", action="append", choices=SERVER_IMPLEMENTATIONS)
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
    parser.add_argument(
        "--append-report",
        action="store_true",
        help="Merge requested runs into an existing report for the same profile.",
    )
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
def _backend_uri(backend: str, postgres_uri: str, tmp: Path) -> Iterator[str]:
    if backend == "sqlite":
        yield f"sqlite:///{tmp / 'mlflow.db'}"
        return

    base_url = make_url(postgres_uri)
    database = f"mlflow_t22_2_{uuid.uuid4().hex}"
    admin_url = base_url.set(database="postgres")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    quoted_database = admin.dialect.identifier_preparer.quote(database)
    try:
        with admin.connect() as connection:
            connection.exec_driver_sql(f"CREATE DATABASE {quoted_database}")
        yield base_url.set(database=database).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.exec_driver_sql(f"DROP DATABASE IF EXISTS {quoted_database} WITH (FORCE)")
        admin.dispose()


@contextlib.contextmanager
def _server(
    server_bin: Path,
    implementation: str,
    backend: str,
    postgres_uri: str,
    tag: str,
) -> Iterator[tuple[str, Path]]:
    with tempfile.TemporaryDirectory(prefix=f"mlflow-t22-2-{backend}-") as raw_tmp:
        tmp = Path(raw_tmp)
        with _backend_uri(backend, postgres_uri, tmp) as backend_uri:
            port = _free_port()
            url = f"http://127.0.0.1:{port}"
            artifact_root = tmp / "artifacts"
            artifact_root.mkdir()
            _provision(backend_uri, artifact_root)
            server_log = tmp / f"{implementation}-{tag}.server.log"
            env = {
                **os.environ,
                "MLFLOW_CRYPTO_KEK_PASSPHRASE": (
                    "t22-2-conformance-passphrase-at-least-32-characters"
                ),
            }
            common_args = [
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
            command = (
                [str(server_bin), *common_args]
                if implementation == "rust"
                else ["uv", "run", "--no-sync", "mlflow", "server", *common_args]
            )
            with server_log.open("w") as output:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT if implementation == "python_http" else ROOT / "rust",
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
                            f"{implementation} server exited with {process.returncode}:\n"
                            f"{server_log.read_text()}"
                        )
                    try:
                        if requests.get(f"{url}/health", timeout=0.5).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError(
                        f"{implementation} server did not become ready:\n{server_log.read_text()}"
                    )
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
    paths = sorted(test["id"] for test in indexed)
    counts = Counter(test["path"] for test in indexed)
    return paths, dict(counts)


def _tee_pytest(
    command: list[str], env: dict[str, str], log_path: Path, *, append: bool = False
) -> int:
    with log_path.open("a" if append else "w") as log:
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


def _merge_junit(parts: list[Path], output: Path) -> None:
    root = ET.Element("testsuites")
    for part in parts:
        parsed = ET.parse(part).getroot()
        if parsed.tag == "testsuite":
            root.append(parsed)
        else:
            root.extend(parsed.findall("testsuite"))
        part.unlink()
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


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
    implementation: str,
    backend: str,
    classification: str,
    report_dir: Path,
) -> dict[str, Any]:
    selectors, indexed_by_suite = _selected_tests(ledger, args.profile, classification)
    stem = f"t22_2_{args.profile}_{classification}_{implementation}_{backend}"
    junit = report_dir / f"{stem}.xml"
    pytest_log = report_dir / f"{stem}.log"
    server_log_copy = report_dir / f"{stem}.server.log"
    for path in (pytest_log, server_log_copy):
        path.unlink(missing_ok=True)

    uv_args = ["uv", "run"]
    if args.profile == "full" and classification == "client_only":
        uv_args.extend([
            "--with",
            "dspy==3.2.1,google-adk==2.4.0,google-cloud-aiplatform==1.160.0,guardrails-ai==0.10.2",
        ])

    def run_pytest(url: str, selected: list[str], part_junit: Path, append: bool) -> int:
        env = {
            **os.environ,
            "MLFLOW_SERVER_TYPE": implementation,
            "MLFLOW_RUST_SERVER_BIN": str(args.server_bin),
            "MLFLOW_TRACKING_URI": url,
            "MLFLOW_REGISTRY_URI": url,
            "MLFLOW_CONFORMANCE_TRACKING_URI": url,
            "MLFLOW_CRYPTO_KEK_PASSPHRASE": "t22-2-conformance-passphrase-at-least-32-characters",
        }
        command = [
            *uv_args,
            "--no-sync",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            f"--junitxml={part_junit}",
            *args.pytest_arg,
            *selected,
        ]
        return _tee_pytest(command, env, pytest_log, append=append)

    exit_codes: list[int] = []
    junit_parts: list[Path] = []
    isolated = classification == "server_reachable"
    selector_groups = [[selector] for selector in selectors] if isolated else [selectors]
    for index, selected in enumerate(selector_groups):
        _logger.info(
            "T22.2 %s %s via %s on %s: selector %d/%d",
            args.profile,
            classification,
            implementation,
            backend,
            index + 1,
            len(selector_groups),
        )
        part_junit = report_dir / f".{stem}.{index}.xml"
        with _server(
            args.server_bin,
            implementation,
            backend,
            args.postgres_uri,
            str(index),
        ) as (url, server_log):
            exit_codes.append(run_pytest(url, selected, part_junit, append=index > 0))
            with server_log_copy.open("a") as combined_server_log:
                combined_server_log.write(f"\n## selector: {selected[0]}\n")
                combined_server_log.write(server_log.read_text())
        junit_parts.append(part_junit)

    _merge_junit(junit_parts, junit)
    totals, suites = _junit_counts(junit, indexed_by_suite)
    for suite, counts in suites.items():
        counts["ledger_test_definitions"] = indexed_by_suite.get(suite, 0)
    return {
        "backend": backend,
        "server_implementation": implementation,
        "classification": classification,
        "isolation": "fresh_server_and_database_per_test" if isolated else "shared_server",
        "exit_code": max(exit_codes, default=0),
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
        f"Coverage status: `{report.get('coverage_status', 'complete')}`. "
        f"{report.get('coverage_note', '')}".rstrip(),
        "",
        "The required profile is the dependency-light HTTP/SDK core used on every Rust CI run. "
        "The full repointable ledger matrix runs nightly and on manual dispatch; tests classified "
        "`python_internal` remain inventory evidence but cannot cross an HTTP process boundary.",
        "",
        "| Classification | Server | Backend | Isolation | Ledger tests | Passed | Failed | "
        "Errors | Skipped | Exit |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        totals = run["totals"]
        lines.append(
            f"| {run['classification']} | {run['server_implementation']} | "
            f"{run['backend']} | {run['isolation']} | "
            f"{run['ledger_test_definitions']} | {totals.get('passed', 0)} | "
            f"{totals.get('failed', 0)} | {totals.get('error', 0)} | "
            f"{totals.get('skipped', 0)} | {run['exit_code']} |"
        )
    lines.extend([
        "",
        "## Per-suite results",
        "",
        "| Suite | Classification | Server | Backend | Ledger tests | Passed | Failed | "
        "Errors | Skipped |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in report["runs"]:
        for suite, counts in run["suites"].items():
            lines.append(
                f"| `{suite}` | {run['classification']} | "
                f"{run['server_implementation']} | {run['backend']} | "
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
    implementations = args.server_implementation or (
        list(SERVER_IMPLEMENTATIONS) if args.profile == "full" else ["rust"]
    )
    if "rust" in implementations and not args.server_bin.is_file():
        raise FileNotFoundError(f"release Rust server not found: {args.server_bin}")
    _run_checked(["uv", "run", "--no-sync", "python", str(HERE / "validate_ledger.py")])
    ledger = json.loads(LEDGER_PATH.read_text())
    args.report_dir.mkdir(parents=True, exist_ok=True)
    backends = args.backend or list(BACKENDS)
    classifications = args.classification or list(CLASSIFICATIONS)
    runs = []
    for backend in backends:
        for classification in classifications:
            run_implementations = ["rust"] if classification == "client_only" else implementations
            runs.extend(
                _run_one(
                    args=args,
                    ledger=ledger,
                    implementation=implementation,
                    backend=backend,
                    classification=classification,
                    report_dir=args.report_dir,
                )
                for implementation in run_implementations
            )
    json_path = args.report_dir / f"t22_2_{args.profile}.json"
    md_path = args.report_dir / f"t22_2_{args.profile}.md"
    if args.append_report and json_path.exists():
        previous = json.loads(json_path.read_text())
        previous_runs = previous["runs"] if previous.get("profile") == args.profile else []
        run_keys = {
            (run["classification"], run["server_implementation"], run["backend"]) for run in runs
        }
        runs = [
            run
            for run in previous_runs
            if (run["classification"], run["server_implementation"], run["backend"]) not in run_keys
        ] + runs
    expected_implementations = SERVER_IMPLEMENTATIONS if args.profile == "full" else ("rust",)
    expected_runs = {
        ("server_reachable", implementation, backend)
        for implementation in expected_implementations
        for backend in BACKENDS
    } | {("client_only", "rust", backend) for backend in BACKENDS}
    actual_runs = {
        (run["classification"], run["server_implementation"], run["backend"]) for run in runs
    }
    coverage_complete = expected_runs <= actual_runs
    report = {
        "schema_version": 1,
        "task": "T22.2",
        "profile": args.profile,
        "coverage_status": "complete" if coverage_complete else "partial",
        "coverage_note": (
            (
                "Python-HTTP baseline, Rust repointable, and client-only runs are present on "
                "both backends; repointable tests used a fresh server and database per test."
                if args.profile == "full"
                else "Required Rust repointable and client-only runs are present on both "
                "backends; repointable tests used a fresh server and database per test."
            )
            if coverage_complete
            else "Only the requested subset is present; append the remaining server/backend runs."
        ),
        "server_implementations": sorted({run["server_implementation"] for run in runs}),
        "ledger_reference": ledger["reference"]["git_sha"],
        "ledger_counts": ledger["summary"]["classification_counts"],
        "ledger_test_counts": ledger["summary"]["test_classification_counts"],
        "test_classification_criterion": ledger["test_classification_criterion"],
        "runs": runs,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_render_markdown(report))
    _logger.info("Wrote %s and %s", json_path.relative_to(ROOT), md_path.relative_to(ROOT))
    return int(any(run["exit_code"] != 0 for run in runs))


if __name__ == "__main__":
    raise SystemExit(main())
