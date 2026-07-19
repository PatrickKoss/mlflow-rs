"""Normalize and compare T23 benchmark proof samples."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
TRACE_RE = re.compile(r"\btr-[0-9a-fA-F]{16,}\b")
STUB_SESSION_RE = re.compile(r"mlflow-dev-stub-[A-Za-z0-9-]+")
ID_KEYS = {
    "assessment_id",
    "digest",
    "id",
    "ids",
    "job_id",
    "run_id",
    "session_id",
    "trace_id",
}
VOLATILE_FRAGMENTS = (
    "timestamp",
    "duration",
    "latency",
    "elapsed",
    "created_time",
    "last_updated",
    "queue_wait",
    "execution_seconds",
    "submit_to_terminal",
)


def _normalize_string(value: str) -> str:
    value = UUID_RE.sub("<uuid>", value)
    value = TRACE_RE.sub("<trace-id>", value)
    value = STUB_SESSION_RE.sub("<provider-session-id>", value)
    return value.replace("/assistant/stream/<uuid>", "/assistant/sessions/<uuid>/stream")


def normalize(value: Any, key: str = "") -> Any:
    lower = key.lower()
    if lower == "polls":
        return "<polls>"
    if lower in ID_KEYS or lower.endswith("_id"):
        return "<id>" if value is not None else None
    if lower.endswith("_ids") and isinstance(value, list):
        return ["<id>" for _ in value]
    if lower.endswith("_time") or any(fragment in lower for fragment in VOLATILE_FRAGMENTS):
        return "<time>" if value is not None else None
    if isinstance(value, dict):
        normalized = {name: normalize(item, name) for name, item in sorted(value.items())}
        if normalized.get("metadata") == {}:
            normalized.pop("metadata")
        return normalized
    if isinstance(value, list):
        normalized = [normalize(item, key) for item in value]
        if all(isinstance(item, dict) and ("key" in item or "name" in item) for item in normalized):
            normalized.sort(key=lambda item: (str(item.get("key", "")), str(item.get("name", ""))))
        return normalized
    if isinstance(value, str):
        if value.startswith("data: ") or value.startswith("event: "):
            lines = []
            for line in value.splitlines():
                if line.startswith("data: "):
                    payload = line.removeprefix("data: ")
                    try:
                        payload = json.dumps(
                            normalize(json.loads(payload)), sort_keys=True, separators=(",", ":")
                        )
                    except json.JSONDecodeError:
                        payload = _normalize_string(payload)
                    lines.append(f"data: {payload}")
                else:
                    lines.append(_normalize_string(line))
            return "\n".join(lines)
        return _normalize_string(value)
    return value


def proof_document(run: dict[str, Any]) -> dict[str, Any]:
    proof = run["equivalence"]
    return normalize({"samples": proof["samples"], "jobs": proof["jobs"]})


def compare_runs(python_run: dict[str, Any], rust_run: dict[str, Any]) -> list[str]:
    python = proof_document(python_run)
    rust = proof_document(rust_run)
    if python == rust:
        return []
    python_text = json.dumps(python, indent=2, sort_keys=True).splitlines()
    rust_text = json.dumps(rust, indent=2, sort_keys=True).splitlines()
    differences = []
    for index, (left, right) in enumerate(zip(python_text, rust_text)):
        if left != right:
            differences.append(f"line {index + 1}: python={left!r}; rust={right!r}")
        if len(differences) == 20:
            break
    if len(python_text) != len(rust_text):
        differences.append(f"line counts differ: python={len(python_text)}, rust={len(rust_text)}")
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_run", type=Path)
    parser.add_argument("rust_run", type=Path)
    args = parser.parse_args(argv)
    python_run = json.loads(args.python_run.read_text())
    rust_run = json.loads(args.rust_run.read_text())
    differences = compare_runs(python_run, rust_run)
    if differences:
        print("equivalence mismatch", file=sys.stderr)
        print("\n".join(differences), file=sys.stderr)
        return 1
    print("equivalence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
