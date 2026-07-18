# ruff: noqa: T201
"""Route parity check: Python routes vs Rust proto and planned-route accounting.

Two modes:

  * ``dump`` — print Python's endpoint list (from
    ``mlflow.server.handlers.get_endpoints()``) as JSON on stdout. Handy for
    eyeballing / debugging.
  * (default, no args) — build the Rust route table (via
    ``cargo run -p mlflow-proto --example dump_routes``), diff it against
    Python, and exit non-zero on any difference that is not covered by the
    allowlist.

Contract (see ``RUST_TRACKING_SERVER_PLAN.md`` §2.1, §3, T1.2):

  * Every PROTO-backed endpoint must match Python EXACTLY on ``(method, path)``
    for BOTH the ``/api/`` and ``/ajax-api/`` forms.
  * Implemented Python-only endpoints that are NOT backed by a
    ``databricks.rpc`` proto option (hand-crafted routes like ``/graphql`` and
    ``server-info``) are listed in ``route_parity_allowlist.txt``.
  * Part II routes are classified below as implemented or planned with their
    owning phase. This includes every section 12 route, including Flask/FastAPI
    routes that ``get_endpoints()`` cannot discover. Proto metadata is
    accounting only: it does not imply that the Rust server has registered a
    live route.
  * A Rust route missing from Python is a HARD failure (never allowlisted).

Run from anywhere; paths are resolved relative to this file / the repo root::

    uv run --frozen python rust/tools/route_parity.py            # full compare
    uv run --frozen python rust/tools/route_parity.py dump       # dump Python
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
RUST_DIR = TOOLS_DIR.parent
REPO_ROOT = RUST_DIR.parent
ALLOWLIST_PATH = TOOLS_DIR / "route_parity_allowlist.txt"

Route = tuple[str, str]


@dataclass(frozen=True)
class RouteInfo:
    section: str
    phase: str
    source: str


SECTION_TITLES = {
    "12.1": "Evaluation datasets",
    "12.2": "GenAI evaluate + jobs API",
    "12.3": "Scorers",
    "12.4": "Issues",
    "12.5": "Label schemas",
    "12.6": "Review queues",
    "12.7": "Prompt optimization",
    "12.8": "Gateway CRUD",
    "12.9": "Gateway runtime",
    "12.10": "Assistant",
    "12.11": "Promptlab",
    "12.12": "Trace archival (no routes)",
    "12.13": "Auth treatment (no additional routes)",
}

# Counts are concrete (method, path) pairs. Proto RPCs appear under both URL
# prefixes, and dual-method paths count once per method.
EXPECTED_SECTION_ROUTE_COUNTS = {
    "12.1": 26,
    "12.2": 3,
    "12.3": 15,
    "12.4": 9,
    "12.5": 12,
    "12.6": 22,
    "12.7": 12,
    "12.8": 78,
    "12.9": 10,
    "12.10": 9,
    "12.11": 1,
    "12.12": 0,
    "12.13": 0,
}

# The RPCs live on MlflowService. Every raw proto endpoint expands to both
# /api and /ajax-api, hence these concrete-route counts are twice the §12
# inventory (plus the extra GET forms for the two search RPCs).
PROTO_SECTIONS = {
    "12.1": ("/mlflow/datasets", "T16.1", 26),
    "12.3": ("/mlflow/scorers", "T16.2", 10),
    "12.4": ("/mlflow/issues", "T16.3", 8),
    "12.5": ("/mlflow/label-schemas", "T16.3", 12),
    "12.6": ("/mlflow/review-queues", "T16.4", 22),
    "12.7": ("/mlflow/prompt-optimization", "T16.6", 12),
    "12.8": ("/mlflow/gateway", "T18.1", 72),
}

# Proto sections that are live in `mlflow-server::handler_for`. Other Part II
# routes are generated but deliberately fall through to not-implemented.
IMPLEMENTED_PROTO_SECTIONS = {"12.1", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8"}


def route_info(section: str, phase: str, source: str, *routes: Route) -> dict[Route, RouteInfo]:
    return {route: RouteInfo(section, phase, source) for route in routes}


def planned(section: str, phase: str, source: str, *routes: Route) -> dict[Route, RouteInfo]:
    return route_info(section, phase, source, *routes)


# Live Rust hand routes which cannot appear in the generated proto route table.
# They remain in route_parity_allowlist.txt for the Python-vs-proto diff, while
# this classified inventory keeps §12 implemented/planned accounting accurate.
IMPLEMENTED_GET_ENDPOINT_ROUTES = route_info(
    "12.2",
    "T16.5",
    "get_endpoints",
    ("GET", "/ajax-api/3.0/mlflow/jobs/<job_id>"),
    ("PATCH", "/ajax-api/3.0/mlflow/jobs/cancel/<job_id>"),
)
IMPLEMENTED_GET_ENDPOINT_ROUTES.update(
    route_info(
        "12.3",
        "T16.2",
        "get_endpoints",
        ("GET", "/api/3.0/mlflow/scorers/online-configs"),
        ("GET", "/ajax-api/3.0/mlflow/scorers/online-configs"),
        ("PUT", "/api/3.0/mlflow/scorers/online-config"),
        ("PUT", "/ajax-api/3.0/mlflow/scorers/online-config"),
    )
)
IMPLEMENTED_GET_ENDPOINT_ROUTES.update(
    route_info(
        "12.2",
        "T17.4",
        "get_endpoints",
        ("POST", "/ajax-api/3.0/mlflow/genai/evaluate/invoke"),
    )
)
IMPLEMENTED_GET_ENDPOINT_ROUTES.update(
    route_info(
        "12.3",
        "T17.4",
        "get_endpoints",
        ("POST", "/ajax-api/3.0/mlflow/scorer/invoke"),
    )
)
IMPLEMENTED_GET_ENDPOINT_ROUTES.update(
    route_info(
        "12.4",
        "T17.4",
        "get_endpoints",
        ("POST", "/ajax-api/3.0/mlflow/issues/invoke"),
    )
)


# Non-proto routes returned by handlers.get_endpoints(). These are the 15
# entries that T15.2 moves out of the old "genai, out of scope" allowlist.
# Two demo-data routes are adjacent GenAI UI surface but are not listed in §12.
PLANNED_GET_ENDPOINT_ROUTES = {
    **planned(
        "12.8",
        "T18.2",
        "get_endpoints",
        ("GET", "/ajax-api/3.0/mlflow/gateway/provider-config"),
        ("GET", "/ajax-api/3.0/mlflow/gateway/secrets/config"),
        ("GET", "/ajax-api/3.0/mlflow/gateway/supported-models"),
        ("GET", "/ajax-api/3.0/mlflow/gateway/supported-providers"),
    ),
    **planned(
        "demo",
        "T20.4",
        "get_endpoints",
        ("POST", "/ajax-api/3.0/mlflow/demo/generate"),
        ("POST", "/ajax-api/3.0/mlflow/demo/delete"),
    ),
}

# §12 routes mounted outside handlers.get_endpoints(): two Flask routes in
# server/__init__.py and the FastAPI gateway/assistant routers. They cannot
# participate in that function's parity diff, but keeping the concrete
# (method, path) inventory here prevents them from disappearing from T15.2's
# accounting.
IMPLEMENTED_EXTERNAL_ROUTES = route_info(
    "12.9",
    "T18.3",
    "gateway_api.py",
    ("POST", "/gateway/{endpoint_name}/mlflow/invocations"),
    ("POST", "/gateway/mlflow/v1/chat/completions"),
)

PLANNED_EXTERNAL_ROUTES = {
    **planned(
        "12.8",
        "T18.2",
        "server/__init__.py",
        ("GET", "/ajax-api/2.0/mlflow/gateway-proxy"),
        ("POST", "/ajax-api/2.0/mlflow/gateway-proxy"),
    ),
    **planned(
        "12.9",
        "T18.4",
        "gateway_api.py",
        ("POST", "/gateway/openai/v1/chat/completions"),
        ("POST", "/gateway/openai/v1/embeddings"),
        ("POST", "/gateway/openai/v1/responses"),
        ("POST", "/gateway/openai/v1/responses/compact"),
        ("POST", "/gateway/anthropic/v1/messages"),
        ("POST", "/gateway/gemini/v1beta/models/{endpoint_name}:generateContent"),
        ("POST", "/gateway/gemini/v1beta/models/{endpoint_name}:streamGenerateContent"),
        ("POST", "/gateway/proxy/{endpoint_name}/{path:path}"),
    ),
    **planned(
        "12.10",
        "T20.1",
        "assistant/api.py",
        ("POST", "/ajax-api/3.0/mlflow/assistant/message"),
        ("GET", "/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream"),
        ("PATCH", "/ajax-api/3.0/mlflow/assistant/sessions/{session_id}"),
        ("POST", "/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/permission"),
        ("GET", "/ajax-api/3.0/mlflow/assistant/providers/{provider}/health"),
        ("GET", "/ajax-api/3.0/mlflow/assistant/config"),
        ("PUT", "/ajax-api/3.0/mlflow/assistant/config"),
        ("POST", "/ajax-api/3.0/mlflow/assistant/skills/install"),
        ("GET", "/ajax-api/3.0/mlflow/assistant/providers/{provider}/models"),
    ),
    **planned(
        "12.11",
        "T20.4",
        "server/__init__.py",
        ("POST", "/ajax-api/2.0/mlflow/runs/create-promptlab-run"),
    ),
}


def python_endpoints() -> set[tuple[str, str]]:
    """Return the full set of ``(http_method, path)`` Python serves.

    Uses ``get_endpoints()`` which is exactly what the Flask app registers
    (`mlflow/server/__init__.py`). Auth routes are registered by a separate app
    and are hand-rolled (not proto-backed), so they are intentionally not part
    of this proto-parity comparison; document any that matter in the allowlist.
    """
    from mlflow.server.handlers import get_endpoints

    out: set[tuple[str, str]] = set()
    for path, _handler, methods in get_endpoints():
        out.update((method, path) for method in methods)
    return out


def rust_routes() -> list[dict[str, str]]:
    """Return the expanded generated Rust proto-route records."""
    result = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "mlflow-proto", "--example", "dump_routes"],
        cwd=RUST_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def rust_endpoints(routes: list[dict[str, str]]) -> set[Route]:
    """Reduce expanded proto-route records to ``(http_method, path)``."""
    return {(route["http_method"], route["path"]) for route in routes}


def load_allowlist() -> set[Route]:
    """Parse the allowlist file into a set of ``(method, path)``.

    Format: ``METHOD<space>PATH`` per line. ``#`` comments and blank lines are
    ignored. A trailing ``  # reason`` on an entry line is also stripped.
    """
    allow: set[tuple[str, str]] = set()
    if not ALLOWLIST_PATH.exists():
        return allow
    for raw in ALLOWLIST_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip an inline "# reason" comment.
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match line.split(None, 1):
            case [method, path]:
                allow.add((method, path))
            case _:
                raise SystemExit(f"Malformed allowlist line: {raw!r}")
    return allow


def proto_section_routes(routes: list[dict[str, str]]) -> dict[str, set[Route]]:
    """Classify the generated Part II proto routes into their §12 sections."""
    out: dict[str, set[Route]] = {}
    for section, (path_fragment, _phase, _expected) in PROTO_SECTIONS.items():
        out[section] = {
            (route["http_method"], route["path"])
            for route in routes
            if route["service"] == "MlflowService" and path_fragment in route["path"]
        }
    return out


def print_section_accounting(proto_routes: dict[str, set[Route]]) -> None:
    print("§12 route accounting (proto metadata plus classified hand routes):")
    all_non_proto = (
        PLANNED_GET_ENDPOINT_ROUTES
        | IMPLEMENTED_GET_ENDPOINT_ROUTES
        | IMPLEMENTED_EXTERNAL_ROUTES
        | PLANNED_EXTERNAL_ROUTES
    )
    for section, title in SECTION_TITLES.items():
        generated = len(proto_routes.get(section, set()))
        hand = [info for info in all_non_proto.values() if info.section == section]
        phases = Counter(info.phase for info in hand)
        proto_implemented = generated if section in IMPLEMENTED_PROTO_SECTIONS else 0
        proto_planned = generated - proto_implemented
        if section in PROTO_SECTIONS and proto_planned:
            phases[PROTO_SECTIONS[section][1]] += generated
        phase_text = ", ".join(f"{phase}={count}" for phase, count in sorted(phases.items()))
        suffix = f"; {phase_text}" if phase_text else ""
        implemented = sum(
            info.section == section for info in IMPLEMENTED_GET_ENDPOINT_ROUTES.values()
        ) + sum(info.section == section for info in IMPLEMENTED_EXTERNAL_ROUTES.values())
        print(
            f"  §{section} {title}: implemented={proto_implemented + implemented}, "
            f"planned={proto_planned + len(hand) - implemented} "
            f"(proto metadata={generated}, hand-registered={len(hand)}{suffix})"
        )
    demo_count = sum(info.section == "demo" for info in PLANNED_GET_ENDPOINT_ROUTES.values())
    print(f"  adjacent demo routes: implemented=0, planned={demo_count} (T20.4)")


def section_route_counts(proto_routes: dict[str, set[Route]]) -> dict[str, int]:
    all_non_proto = (
        PLANNED_GET_ENDPOINT_ROUTES
        | IMPLEMENTED_GET_ENDPOINT_ROUTES
        | IMPLEMENTED_EXTERNAL_ROUTES
        | PLANNED_EXTERNAL_ROUTES
    )
    return {
        section: len(proto_routes.get(section, set()))
        + sum(info.section == section for info in all_non_proto.values())
        for section in SECTION_TITLES
    }


def dump_python() -> int:
    eps = sorted(python_endpoints())
    print(json.dumps([{"http_method": m, "path": p} for m, p in eps], indent=2))
    return 0


def compare() -> int:
    py = python_endpoints()
    route_records = rust_routes()
    rust = rust_endpoints(route_records)
    allow = load_allowlist()
    planned_get = set(PLANNED_GET_ENDPOINT_ROUTES)
    proto_routes = proto_section_routes(route_records)

    # Rust routes Python does not serve: always a hard failure.
    rust_only = sorted(rust - py)
    # Python routes not in the Rust proto table: allowed only if allowlisted.
    python_only = sorted((py - rust) - allow - planned_get)
    # Allowlist hygiene: entries that no longer apply (Python now serves them
    # via the Rust table, or Python dropped them).
    stale_allow = sorted(allow - (py - rust))
    stale_planned = sorted(planned_get - (py - rust))
    overlap = sorted(allow & planned_get)
    bad_proto_counts = [
        (section, expected, len(proto_routes[section]))
        for section, (_fragment, _phase, expected) in PROTO_SECTIONS.items()
        if len(proto_routes[section]) != expected
    ]
    actual_section_counts = section_route_counts(proto_routes)
    bad_section_counts = [
        (section, expected, actual_section_counts[section])
        for section, expected in EXPECTED_SECTION_ROUTE_COUNTS.items()
        if actual_section_counts[section] != expected
    ]
    bad_planned_get_count = len(planned_get) != 6

    ok = True
    if rust_only:
        ok = False
        print("HARD FAILURE: Rust route table has routes Python does not serve:")
        for method, path in rust_only:
            print(f"  + {method} {path}")

    if python_only:
        ok = False
        print(
            "FAILURE: Python endpoints missing from the Rust proto table and not in the allowlist:"
        )
        for method, path in python_only:
            print(f"  - {method} {path}")
        print(
            "\nAdd not-yet-implemented Part II routes to the phase-tagged planned "
            "inventory. Add only already-implemented, non-proto hand routes to "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}."
        )

    if stale_allow:
        ok = False
        print("FAILURE: stale allowlist entries (no longer Python-only differences); remove them:")
        for method, path in stale_allow:
            print(f"  ? {method} {path}")

    if stale_planned:
        ok = False
        print("FAILURE: stale planned entries (route landed or Python dropped it):")
        for method, path in stale_planned:
            print(f"  ? {method} {path}")

    if overlap:
        ok = False
        print("FAILURE: routes cannot be both implemented/allowlisted and planned:")
        for method, path in overlap:
            print(f"  ! {method} {path}")

    if bad_proto_counts:
        ok = False
        print("FAILURE: §12 proto route inventory count changed:")
        for section, expected, actual in bad_proto_counts:
            print(f"  ! §{section}: expected {expected}, found {actual}")

    if bad_section_counts:
        ok = False
        print("FAILURE: §12 total route accounting changed:")
        for section, expected, actual in bad_section_counts:
            print(f"  ! §{section}: expected {expected}, found {actual}")

    if bad_planned_get_count:
        ok = False
        print(
            "FAILURE: expected exactly 6 phase-tagged non-proto get_endpoints routes, "
            f"found {len(planned_get)}"
        )

    if ok:
        matched = len(py & rust)
        print(
            f"OK: {matched} proto-backed routes match exactly "
            f"(Python total {len(py)}, Rust total {len(rust)}, "
            f"implemented hand routes {len(allow)}, "
            f"planned get_endpoints routes {len(planned_get)})."
        )
        print_section_accounting(proto_routes)
        return 0
    return 1


def main(argv: list[str]) -> int:
    match argv:
        case ["dump"]:
            return dump_python()
        case []:
            return compare()
        case _:
            print(__doc__)
            return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
