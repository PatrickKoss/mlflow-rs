"""T12.4 differential replay harness.

Seeds a fresh SQLite tracking DB via the real Python store, boots BOTH the
Python and Rust ``mlflow-server`` on their own identical-bytes copies of it, and
replays an ordered request corpus against each independently. Responses are
normalized (timestamps/ids/paths erased, tokens opacity-checked) and diffed
pairwise at each step. Diffs not covered by ``allowlist.yaml`` are failures.

Run from the repo root:

    uv run python rust/compliance/replay.py                 # full suite, sqlite
    uv run python rust/compliance/replay.py -k experiments  # only matching sections
    uv run python rust/compliance/replay.py --list          # enumerate sections

Exit code 0 iff zero non-allowlisted diffs (skipped sections do not fail the run
but are reported prominently). A Markdown + JSON report is written under
``rust/compliance/report/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

# The harness lives at rust/compliance/; the repo root is two levels up. Put
# both on the path: the repo root so the T12.1 launch plumbing under
# `tests/tracking/...` imports, and this dir so `engine` resolves without
# `rust/` needing to be a Python package.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

from engine import (
    Diff,
    NormalizeOptions,
    diff_normalized,
    json_path_get,
    normalize,
    substitute,
)

from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    ARTIFACTS_DESTINATION_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
)

# Reuse the T12.1 launch plumbing so seeding/booting matches the integration
# suite exactly (same env-var names, same rust-binary resolution).
from tests.tracking.integration_test_utils import (
    _await_server_up_or_die,
    _resolve_rust_server_bin,
    _rust_server_cmd,
)

CORPUS_DIR = _HERE / "corpus"
ALLOWLIST_PATH = _HERE / "allowlist.yaml"
REPORT_DIR = _HERE / "report"
LOCALHOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Server lifecycle.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOCALHOST, 0))
        return s.getsockname()[1]


def _seed_tracking_db(db_path: Path) -> None:
    """Create a migrated, lightly-populated SQLite tracking DB via the real store.

    Mirrors ``rust/tools/make_test_db.py``: creating an experiment runs the full
    Alembic chain so the Rust server (which refuses an unmigrated DB) accepts it.
    We seed one experiment/run so read endpoints have data on the very first case
    without depending on corpus mutations.
    """
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{db_path}"
    client = MlflowClient(tracking_uri=uri)
    exp_id = client.create_experiment("seed_experiment")
    client.set_experiment_tag(exp_id, "seeded", "true")
    run = client.create_run(exp_id, tags={"mlflow.runName": "seed-run"})
    client.log_param(run.info.run_id, "seed_param", "1")
    client.log_metric(run.info.run_id, "seed_metric", 0.5, step=0)
    client.set_terminated(run.info.run_id)


def _migrate_auth_db(auth_db_uri: str) -> None:
    from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

    store = SqlAlchemyStore()
    store.init_db(auth_db_uri)
    admin_user = os.environ.get("_COMPLIANCE_ADMIN_USER", "admin")
    admin_pass = os.environ.get("_COMPLIANCE_ADMIN_PASS", "password1234")
    if not store.has_user(admin_user):
        store.create_user(admin_user, admin_pass, is_admin=True)
    store.engine.dispose()


@dataclass
class ServerHandle:
    name: str
    url: str
    proc: subprocess.Popen
    log_path: Path


class DualServers:
    """Boots a Python + Rust server pair on identical-bytes DB copies."""

    def __init__(
        self,
        workdir: Path,
        seed_db: Path,
        artifact_root: Path,
        extra_env: dict[str, str] | None = None,
        python_app: str = "mlflow.server:app",
    ) -> None:
        self.workdir = workdir
        self.seed_db = seed_db
        self.artifact_root = artifact_root
        self.extra_env = extra_env or {}
        # The flask `--app` target. The plain sections use the base tracking
        # app; the auth section must use the auth app factory (the auth
        # endpoints exist only there), mirroring
        # `tests/server/auth/test_auth.py`'s `app="mlflow.server.auth:create_app"`.
        self.python_app = python_app
        self.python: ServerHandle | None = None
        self.rust: ServerHandle | None = None

    def _boot(self, name: str, cmd: list, backend_uri: str, art: Path) -> ServerHandle:
        port = int(cmd[cmd.index("--port") + 1])
        log_path = self.workdir / f"{name}.log"
        log = log_path.open("w")
        env = {
            **os.environ,
            BACKEND_STORE_URI_ENV_VAR: backend_uri,
            ARTIFACT_ROOT_ENV_VAR: str(art),
            # The Rust boot passes `--serve-artifacts --artifacts-destination`
            # (see `_rust_server_cmd`); the Python app enables its artifact
            # proxy from these env vars instead (`mlflow/server/__init__.py:76`).
            # Without them the Python side 404s every /mlflow-artifacts route
            # and the whole artifacts section reads as a status mismatch.
            # Harmless for the Rust process, which reads CLI flags only.
            SERVE_ARTIFACTS_ENV_VAR: "true",
            ARTIFACTS_DESTINATION_ENV_VAR: str(art),
            **self.extra_env,
        }
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        _await_server_up_or_die(port, timeout=40)
        return ServerHandle(
            name=name, url=f"http://{LOCALHOST}:{port}", proc=proc, log_path=log_path
        )

    def __enter__(self) -> "DualServers":
        py_db = self.workdir / "python.db"
        rust_db = self.workdir / "rust.db"
        shutil.copy(self.seed_db, py_db)
        shutil.copy(self.seed_db, rust_db)
        py_art = self.artifact_root / "python"
        rust_art = self.artifact_root / "rust"
        py_art.mkdir(parents=True, exist_ok=True)
        rust_art.mkdir(parents=True, exist_ok=True)

        py_port = _free_port()
        py_cmd = [
            sys.executable,
            "-m",
            "flask",
            "--app",
            self.python_app,
            "run",
            "--host",
            LOCALHOST,
            "--port",
            str(py_port),
        ]
        self.python = self._boot("python", py_cmd, f"sqlite:///{py_db}", py_art)

        rust_port = _free_port()
        rust_bin = _resolve_rust_server_bin()
        rust_cmd = _rust_server_cmd(rust_bin, rust_port, f"sqlite:///{rust_db}", str(rust_art))
        self.rust = self._boot("rust", rust_cmd, f"sqlite:///{rust_db}", rust_art)
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in (self.python, self.rust):
            if h is not None:
                h.proc.terminate()
                try:
                    h.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    h.proc.kill()


# ---------------------------------------------------------------------------
# Corpus model.
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    method: str
    path: str
    section: str
    headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    body: Any = None
    bind: list[dict[str, str]] = field(default_factory=list)
    normalize_opts: NormalizeOptions = field(default_factory=NormalizeOptions)
    auth: str | None = None  # "admin" | "user" | None
    expect_status: int | None = None  # optional exact-status assertion (both servers)
    walk: dict[str, Any] | None = None  # walk_pages descriptor
    compare_body: bool = True
    raw_body: str | None = None  # raw (non-JSON) request body, e.g. artifact upload


@dataclass
class CaseResult:
    name: str
    section: str
    method: str
    path: str
    py_status: int | None
    rust_status: int | None
    status_match: bool
    diffs: list[dict[str, Any]]
    allowlisted: list[dict[str, Any]]
    error: str | None = None
    note: str | None = None


def _parse_case(raw: dict[str, Any], section: str) -> Case:
    n = raw.get("normalize", {}) or {}
    opts = NormalizeOptions(
        drop_paths=n.get("drop_paths", []),
        extra_timestamp_fields=n.get("extra_timestamp_fields", []),
        normalize_ids=n.get("normalize_ids", True),
        normalize_paths=n.get("normalize_paths", True),
    )
    return Case(
        name=raw["name"],
        method=raw["method"].upper(),
        path=raw["path"],
        section=section,
        headers=raw.get("headers", {}) or {},
        query=raw.get("query", {}) or {},
        body=raw.get("body"),
        bind=raw.get("bind", []) or [],
        normalize_opts=opts,
        auth=raw.get("auth"),
        expect_status=raw.get("expect_status"),
        walk=raw.get("walk"),
        compare_body=raw.get("compare_body", True),
        raw_body=raw.get("raw_body"),
    )


def load_corpus(only: list[str] | None = None) -> dict[str, list[Case]]:
    """Load every ``corpus/*.yaml`` into an ordered {section: [Case,...]} map.

    Section = file stem. Files are processed in sorted filename order; within a
    file, cases keep authoring order (that ordering is load-bearing: later cases
    consume ids bound by earlier ones).
    """
    sections: dict[str, list[Case]] = {}
    for f in sorted(CORPUS_DIR.glob("*.yaml")):
        section = f.stem
        if only and not any(k in section for k in only):
            continue
        doc = yaml.safe_load(f.read_text()) or {}
        cases = [_parse_case(c, section) for c in doc.get("cases", [])]
        if cases:
            sections[section] = cases
    return sections


# ---------------------------------------------------------------------------
# Allowlist.
# ---------------------------------------------------------------------------


@dataclass
class AllowEntry:
    case_or_path: str
    json_pointer: str
    reason: str
    link: str = ""


def load_allowlist() -> list[AllowEntry]:
    if not ALLOWLIST_PATH.exists():
        return []
    doc = yaml.safe_load(ALLOWLIST_PATH.read_text()) or {}
    return [AllowEntry(**e) for e in doc.get("entries", [])]


def _is_allowlisted(case: Case, d: Diff, allow: list[AllowEntry]) -> AllowEntry | None:
    for e in allow:
        matches_case = e.case_or_path in (case.name, case.path, case.section, "*")
        matches_ptr = (
            e.json_pointer in ("*", d.json_pointer)
            or d.json_pointer.startswith(e.json_pointer.rstrip("*"))
            if e.json_pointer.endswith("*")
            else e.json_pointer in ("*", d.json_pointer)
        )
        if matches_case and matches_ptr:
            return e
    return None


# ---------------------------------------------------------------------------
# Request execution.
# ---------------------------------------------------------------------------


def _auth_tuple(kind: str | None, creds: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    if kind is None:
        return None
    return creds.get(kind)


def _do_request(
    handle: ServerHandle,
    case: Case,
    bindings: dict[str, Any],
    creds: dict[str, tuple[str, str]],
    path_override: str | None = None,
    query_override: dict[str, Any] | None = None,
) -> tuple[int, Any, dict[str, str]]:
    path = substitute(path_override or case.path, bindings)
    query = substitute(query_override if query_override is not None else case.query, bindings)
    headers = substitute(case.headers, bindings)
    body = substitute(case.body, bindings) if case.body is not None else None
    url = handle.url + path
    kwargs: dict[str, Any] = {"timeout": 30}
    if query:
        kwargs["params"] = query
    if headers:
        kwargs["headers"] = headers
    if case.raw_body is not None:
        kwargs["data"] = substitute(case.raw_body, bindings).encode()
    elif body is not None:
        kwargs["json"] = body
    auth = _auth_tuple(case.auth, creds)
    if auth:
        kwargs["auth"] = auth
    resp = requests.request(case.method, url, **kwargs)
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"__raw_text__": resp.text}
    return resp.status_code, decoded, dict(resp.headers)


def _apply_bindings(case: Case, body: Any, bindings: dict[str, Any]) -> None:
    for b in case.bind:
        val = json_path_get(body, b["from"])
        bindings[b["bind"]] = val


# ---------------------------------------------------------------------------
# walk_pages: opacity-checked pagination walk.
# ---------------------------------------------------------------------------


def _walk_pages(
    handle: ServerHandle,
    case: Case,
    bindings: dict[str, Any],
    creds: dict[str, tuple[str, str]],
) -> list[Any]:
    """Follow ``next_page_token`` on one server, returning the normalized pages.

    The walk descriptor: ``{token_field, page_size_param, max_pages}``. Each page
    request injects the previous page's opaque token; we never inspect its bytes,
    only feed it back. Returns the list of per-page normalized bodies.
    """
    w = case.walk or {}
    token_field = w.get("token_field", "next_page_token")
    token_param = w.get("token_param", "page_token")
    max_pages = w.get("max_pages", 10)
    pages: list[Any] = []
    query = dict(substitute(case.query, bindings))
    for _ in range(max_pages):
        status, body, _ = _do_request(handle, case, bindings, creds, query_override=query)
        pages.append((status, normalize(body, case.normalize_opts)))
        token = body.get(token_field) if isinstance(body, dict) else None
        if not token:
            break
        query = dict(query)
        query[token_param] = token
    return pages


# ---------------------------------------------------------------------------
# Case execution + comparison.
# ---------------------------------------------------------------------------


def run_case(
    case: Case,
    servers: DualServers,
    py_bindings: dict[str, Any],
    rust_bindings: dict[str, Any],
    allow: list[AllowEntry],
    creds: dict[str, tuple[str, str]],
) -> CaseResult:
    try:
        if case.walk:
            py_pages = _walk_pages(servers.python, case, py_bindings, creds)
            rust_pages = _walk_pages(servers.rust, case, rust_bindings, creds)
            py_status = py_pages[0][0] if py_pages else None
            rust_status = rust_pages[0][0] if rust_pages else None
            diffs: list[Diff] = []
            if len(py_pages) != len(rust_pages):
                diffs.append(Diff("/__page_count__", len(py_pages), len(rust_pages), "value"))
            for i, (pp, rp) in enumerate(zip(py_pages, rust_pages)):
                diffs.extend(diff_normalized(pp[1], rp[1], f"/page{i}"))
        else:
            py_status, py_body, _ = _do_request(servers.python, case, py_bindings, creds)
            rust_status, rust_body, _ = _do_request(servers.rust, case, rust_bindings, creds)
            _apply_bindings(case, py_body, py_bindings)
            _apply_bindings(case, rust_body, rust_bindings)
            diffs = []
            if case.compare_body:
                py_norm = normalize(py_body, case.normalize_opts)
                rust_norm = normalize(rust_body, case.normalize_opts)
                diffs = diff_normalized(py_norm, rust_norm)
    except Exception as exc:
        return CaseResult(
            name=case.name,
            section=case.section,
            method=case.method,
            path=case.path,
            py_status=None,
            rust_status=None,
            status_match=False,
            diffs=[],
            allowlisted=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    status_match = py_status == rust_status
    real_diffs: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []

    # A status mismatch is allowlistable only via an explicit `/__status__`
    # pointer (or `*`) on a matching entry — used for DELIBERATE deviations
    # (e.g. Python 500s on an unhandled exception where Rust returns a clean
    # 4xx). The mismatch is then reported under `allowlisted`, not as a
    # failure.
    if not status_match:
        status_diff = Diff(
            json_pointer="/__status__",
            python_value=py_status,
            rust_value=rust_status,
            kind="status",
        )
        entry = _is_allowlisted(case, status_diff, allow)
        if entry is not None:
            allowed.append({**asdict(status_diff), "reason": entry.reason})
            status_match = True

    for d in diffs:
        entry = _is_allowlisted(case, d, allow)
        rec = {**asdict(d)}
        if entry is not None:
            rec["reason"] = entry.reason
            allowed.append(rec)
        else:
            real_diffs.append(rec)

    note = None
    if case.expect_status is not None and py_status != case.expect_status:
        note = f"python status {py_status} != expected {case.expect_status}"

    return CaseResult(
        name=case.name,
        section=case.section,
        method=case.method,
        path=case.path,
        py_status=py_status,
        rust_status=rust_status,
        status_match=status_match,
        diffs=real_diffs,
        allowlisted=allowed,
        note=note,
    )


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def _section_counts(results: list[CaseResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        s = out.setdefault(
            r.section,
            {"cases": 0, "status_mismatch": 0, "diffs": 0, "allowlisted": 0, "errors": 0},
        )
        s["cases"] += 1
        if not r.status_match:
            s["status_mismatch"] += 1
        s["diffs"] += len(r.diffs)
        s["allowlisted"] += len(r.allowlisted)
        if r.error:
            s["errors"] += 1
    return out


def write_report(
    results: list[CaseResult],
    skipped: list[dict[str, str]],
    coverage_notes: str,
) -> tuple[int, int, int]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    total_diffs = sum(len(r.diffs) for r in results)
    total_allow = sum(len(r.allowlisted) for r in results)
    status_mismatches = sum(1 for r in results if not r.status_match)
    errors = sum(1 for r in results if r.error)
    counts = _section_counts(results)

    payload = {
        "summary": {
            "cases_run": len(results),
            "non_allowlisted_diffs": total_diffs,
            "allowlisted_diffs": total_allow,
            "status_mismatches": status_mismatches,
            "errors": errors,
            "skipped_sections": skipped,
        },
        "per_section": counts,
        "results": [asdict(r) for r in results],
    }
    (REPORT_DIR / "last_run.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = ["# T12.4 Differential Replay - Last Run", ""]
    lines.append(
        f"- Cases run: **{len(results)}**  |  Non-allowlisted diffs: **{total_diffs}**  |  "
        f"Allowlisted: **{total_allow}**  |  Status mismatches: **{status_mismatches}**  |  "
        f"Errors: **{errors}**"
    )
    lines.append("")
    lines.append("## Per-section")
    lines.append("")
    lines.append("| Section | Cases | Status mismatch | Diffs | Allowlisted | Errors |")
    lines.append("|---|---|---|---|---|---|")
    for sec, c in sorted(counts.items()):
        lines.append(
            f"| {sec} | {c['cases']} | {c['status_mismatch']} | {c['diffs']} | "
            f"{c['allowlisted']} | {c['errors']} |"
        )
    lines.append("")
    if skipped:
        lines.append("## Skipped sections")
        lines.append("")
        lines.extend(f"- **{s['section']}**: {s['reason']}" for s in skipped)
        lines.append("")
    fails = [r for r in results if r.diffs or not r.status_match or r.error]
    if fails:
        lines.append("## Failures (non-allowlisted diffs / status mismatch / errors)")
        lines.append("")
        for r in fails:
            lines.append(f"### {r.section} :: {r.name}")
            lines.append(f"`{r.method} {r.path}` py={r.py_status} rust={r.rust_status}")
            if r.error:
                lines.append(f"- ERROR: {r.error}")
            lines.extend(
                f"- `{d['json_pointer']}` ({d['kind']}): py={d['python_value']!r} "
                f"rust={d['rust_value']!r}"
                for d in r.diffs
            )
            lines.append("")
    if any(r.allowlisted for r in results):
        lines.append("## Allowlisted diffs (known, tolerated)")
        lines.append("")
        for r in results:
            lines.extend(
                f"- {r.section}::{r.name} `{d['json_pointer']}` - {d.get('reason', '')}"
                for d in r.allowlisted
            )
        lines.append("")
    lines.append("## Coverage notes")
    lines.append("")
    lines.append(coverage_notes)
    (REPORT_DIR / "last_run.md").write_text("\n".join(lines))
    return total_diffs, status_mismatches, errors


COVERAGE_NOTES = """\
Corpus sections map to plan section 3 as follows:

- experiments -> 3.1 (CRUD, search POST+GET, pagination-walk, tags, errors)
- runs -> 3.2 (CRUD, log-metric/param/tag, log-batch, search-walk, errors)
- metrics -> 3.3 (get-history, get-history-bulk-interval, bulk ajax)
- logged_models -> 3.5 (create/get/search-walk/tags/artifacts-list, datasets 3.4)
- traces -> 3.6/3.7 (startTraceV3/end/search-walk/tag, OTLP 3.8)
- registry -> 3.14 (RM+MV CRUD/search-walk/stages/aliases/download-uri/errors)
- webhooks -> 3.15 (CRUD/test; local receiver skipped if unavailable)
- graphql -> 3.12 (getExperiment/getRun/searchModelVersions)
- server_info -> 3.13 (health/version/server-info)
- artifacts -> 3.11 (upload/list/download via proxy)
- auth (separate boot) -> 3.16 (401/403/admin/non-admin)
- workspaces (separate boot) -> 3.17 (X-MLFLOW-WORKSPACE scoping)
- prompt_optimization -> 12.7 (queued create/get/search/cancel/delete + errors)

Deliberately deferred to follow-up (documented, not covered here): assessments
FieldMask update paths (3.9) beyond create/get; trace artifact fetch dispatch
on spansLocation (3.10); tracing V2 deprecated adapters (3.7) beyond the search
smoke; queryTraceMetrics / calculateTraceFilterCorrelation aggregations (3.6);
multipart artifact create/complete/abort + presigned URLs (3.11); full RBAC
role/permission matrix and after-request search filtering (3.16); workspace
delete modes RESTRICT/CASCADE/SET_DEFAULT (3.17). These are enumerated as the
extensibility backlog for the corpus.
"""


# ---------------------------------------------------------------------------
# Section runners.
# ---------------------------------------------------------------------------


def _run_sqlite_sections(
    sections: dict[str, list[Case]],
    allow: list[AllowEntry],
    workroot: Path,
) -> list[CaseResult]:
    """Run all non-auth/non-workspace sections against one Python+Rust pair."""
    seed_db = workroot / "seed.db"
    _seed_tracking_db(seed_db)
    artifact_root = workroot / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    creds: dict[str, tuple[str, str]] = {}
    extra_env = {}
    if "prompt_optimization" in sections:
        extra_env = {
            "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "true",
            "_MLFLOW_HUEY_STORAGE_PATH": str(workroot),
        }
    with DualServers(workroot, seed_db, artifact_root, extra_env=extra_env) as servers:
        py_bindings: dict[str, Any] = {}
        rust_bindings: dict[str, Any] = {}
        for section, cases in sections.items():
            results.extend(
                run_case(case, servers, py_bindings, rust_bindings, allow, creds) for case in cases
            )
    return results


def _run_auth_section(
    cases: list[Case],
    allow: list[AllowEntry],
    workroot: Path,
    skipped: list[dict[str, str]],
) -> list[CaseResult]:
    """Boot both servers with basic-auth enabled and run the auth corpus.

    Requires the ``auth`` extra (flask-wtf etc.). If either server refuses to
    boot with auth in this environment, the section is skipped with a reason and
    no fake results are produced.
    """
    admin_user, admin_pass = "admin", "password1234"
    user_user, user_pass = "alice", "alicepw123"
    creds = {"admin": (admin_user, admin_pass), "user": (user_user, user_pass)}

    seed_db = workroot / "seed.db"
    _seed_tracking_db(seed_db)
    auth_db = workroot / "auth.db"
    auth_db_uri = f"sqlite:///{auth_db}"
    ini = workroot / "basic_auth.ini"
    ini.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = {auth_db_uri}\n"
        f"admin_username = {admin_user}\n"
        f"admin_password = {admin_pass}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
        "grant_default_workspace_access = false\n"
    )
    try:
        os.environ["_COMPLIANCE_ADMIN_USER"] = admin_user
        os.environ["_COMPLIANCE_ADMIN_PASS"] = admin_pass
        _migrate_auth_db(auth_db_uri)
    except Exception as exc:
        skipped.append({"section": "auth", "reason": f"auth DB migration failed: {exc}"})
        return []

    artifact_root = workroot / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    extra_env = {
        # Both servers read the ini via MLFLOW_AUTH_CONFIG_PATH (its
        # `database_uri` names the auth DB — the old MLFLOW_AUTH_DATABASE_URI
        # env override was retired by T9.8 on the Rust side and never existed
        # on the Python side).
        "MLFLOW_AUTH_CONFIG_PATH": str(ini),
        # Python's `create_app` refuses to start without a static secret key
        # (CSRF); the Rust server owns its own per-process secret (plan D12)
        # and ignores this.
        "MLFLOW_FLASK_SERVER_SECRET_KEY": "t124-compliance-secret",
    }
    results: list[CaseResult] = []
    try:
        with DualServers(
            workroot,
            seed_db,
            artifact_root,
            extra_env=extra_env,
            # The auth endpoints exist only in the auth app factory; the base
            # `mlflow.server:app` 404s them all.
            python_app="mlflow.server.auth:create_app",
        ) as servers:
            # Create the non-admin user on both servers via the admin account.
            for h in (servers.python, servers.rust):
                requests.post(
                    f"{h.url}/api/2.0/mlflow/users/create",
                    json={"username": user_user, "password": user_pass},
                    auth=(admin_user, admin_pass),
                    timeout=30,
                )
            py_bindings: dict[str, Any] = {}
            rust_bindings: dict[str, Any] = {}
            results.extend(
                run_case(case, servers, py_bindings, rust_bindings, allow, creds) for case in cases
            )
    except Exception as exc:
        skipped.append({"section": "auth", "reason": f"auth-enabled server boot failed: {exc}"})
        return []
    return results


def _run_workspace_section(
    cases: list[Case],
    allow: list[AllowEntry],
    workroot: Path,
    skipped: list[dict[str, str]],
) -> list[CaseResult]:
    """Boot both servers with workspaces enabled and run the workspace corpus."""
    seed_db = workroot / "seed.db"
    _seed_tracking_db(seed_db)
    artifact_root = workroot / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    extra_env = {"MLFLOW_ENABLE_WORKSPACES": "true"}
    results: list[CaseResult] = []
    creds: dict[str, tuple[str, str]] = {}
    try:
        with DualServers(workroot, seed_db, artifact_root, extra_env=extra_env) as servers:
            py_bindings: dict[str, Any] = {}
            rust_bindings: dict[str, Any] = {}
            results.extend(
                run_case(case, servers, py_bindings, rust_bindings, allow, creds) for case in cases
            )
    except Exception as exc:
        skipped.append({"section": "workspaces", "reason": f"workspace-enabled boot failed: {exc}"})
        return []
    return results


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", dest="only", action="append", help="only run matching sections")
    parser.add_argument("--list", action="store_true", help="list sections and exit")
    args = parser.parse_args()

    sections = load_corpus(only=args.only)
    if args.list:
        for sec, cases in sections.items():
            print(f"{sec}: {len(cases)} case(s)")
        print(f"total: {sum(len(c) for c in sections.values())} case(s)")
        return 0

    allow = load_allowlist()
    skipped: list[dict[str, str]] = []

    auth_cases = sections.pop("auth", [])
    workspace_cases = sections.pop("workspaces", [])

    all_results: list[CaseResult] = []
    with tempfile.TemporaryDirectory(prefix="t124-sqlite-") as td:
        all_results.extend(_run_sqlite_sections(sections, allow, Path(td)))

    if auth_cases:
        with tempfile.TemporaryDirectory(prefix="t124-auth-") as td:
            all_results.extend(_run_auth_section(auth_cases, allow, Path(td), skipped))
    if workspace_cases:
        with tempfile.TemporaryDirectory(prefix="t124-ws-") as td:
            all_results.extend(_run_workspace_section(workspace_cases, allow, Path(td), skipped))

    total_diffs, status_mismatches, errors = write_report(all_results, skipped, COVERAGE_NOTES)

    for s in skipped:
        pass

    return 0 if (total_diffs == 0 and status_mismatches == 0 and errors == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
