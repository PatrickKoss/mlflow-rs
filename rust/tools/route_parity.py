#!/usr/bin/env python
"""Route parity check: Python `get_endpoints()` vs the Rust proto route table.

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
  * Python-only endpoints that are NOT backed by a ``databricks.rpc`` proto
    option (genai routes, hand-crafted Flask routes like ``/graphql`` and
    ``server-info``, auth routes) are expected and listed in
    ``route_parity_allowlist.txt``. Each allowlisted entry needs a reason.
  * A Rust route missing from Python is a HARD failure (never allowlisted).

Run from anywhere; paths are resolved relative to this file / the repo root::

    uv run --frozen python rust/tools/route_parity.py            # full compare
    uv run --frozen python rust/tools/route_parity.py dump       # dump Python
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
RUST_DIR = TOOLS_DIR.parent
REPO_ROOT = RUST_DIR.parent
ALLOWLIST_PATH = TOOLS_DIR / "route_parity_allowlist.txt"


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
        for method in methods:
            out.add((method, path))
    return out


def rust_endpoints() -> set[tuple[str, str]]:
    """Return the Rust route table (expanded) as ``(http_method, path)``."""
    result = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "mlflow-proto", "--example", "dump_routes"],
        cwd=RUST_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    routes = json.loads(result.stdout)
    return {(r["http_method"], r["path"]) for r in routes}


def load_allowlist() -> set[tuple[str, str]]:
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


def dump_python() -> int:
    eps = sorted(python_endpoints())
    print(json.dumps([{"http_method": m, "path": p} for m, p in eps], indent=2))
    return 0


def compare() -> int:
    py = python_endpoints()
    rust = rust_endpoints()
    allow = load_allowlist()

    # Rust routes Python does not serve: always a hard failure.
    rust_only = sorted(rust - py)
    # Python routes not in the Rust proto table: allowed only if allowlisted.
    python_only = sorted((py - rust) - allow)
    # Allowlist hygiene: entries that no longer apply (Python now serves them
    # via the Rust table, or Python dropped them).
    stale_allow = sorted(allow - (py - rust))

    ok = True
    if rust_only:
        ok = False
        print("HARD FAILURE: Rust route table has routes Python does not serve:")
        for method, path in rust_only:
            print(f"  + {method} {path}")

    if python_only:
        ok = False
        print("FAILURE: Python endpoints missing from the Rust proto table "
              "and not in the allowlist:")
        for method, path in python_only:
            print(f"  - {method} {path}")
        print(
            "\nIf these are intentionally Python-only (genai / hand-crafted "
            "Flask / auth routes), add them to "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} with a one-line reason."
        )

    if stale_allow:
        ok = False
        print("FAILURE: stale allowlist entries (no longer Python-only "
              "differences); remove them:")
        for method, path in stale_allow:
            print(f"  ? {method} {path}")

    if ok:
        matched = len(py & rust)
        print(
            f"OK: {matched} proto-backed routes match exactly "
            f"(Python total {len(py)}, Rust total {len(rust)}, "
            f"allowlisted Python-only {len(allow)})."
        )
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
