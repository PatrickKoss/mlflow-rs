"""Differential replay engine: variable binding, normalization, and diffing.

This module is the reusable core of the T12.4 compliance harness. It is
deliberately free of any server/process management so it can be unit-tested in
isolation. ``replay.py`` drives it against live Python and Rust servers.

Three mechanics, per the task design:

* **Variable binding** - state-dependent values (ids, tokens) are captured from
  earlier responses with ``{"bind": "run_id", "from": "$.run.info.run_id"}`` and
  substituted into later requests as ``${run_id}``. Each server runs the corpus
  in its own *session* with its own binding table, so Python's ids and Rust's ids
  never need to match byte-for-byte; the normalizer erases the id values before
  the pairwise compare.
* **Normalization** - recursive rewrite of a response body that replaces
  non-deterministic fields (timestamps, ids, tokens, artifact paths) with stable
  placeholders so two structurally equivalent responses compare equal.
* **Opacity-checked tokens** - ``next_page_token`` (and variants) are never
  compared byte-wise. The normalizer records only whether each server produced a
  token; a ``walk_pages`` case then follows the token on each server and asserts
  the *normalized pages* are equal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def json_path_get(obj: Any, path: str) -> Any:
    """Resolve a ``$.a.b[0].c`` path against a decoded-JSON object.

    Returns ``None`` when any segment is missing rather than raising, so binding
    directives degrade to an empty capture instead of blowing up the run.
    """
    if not path.startswith("$"):
        raise ValueError(f"json path must start with '$': {path!r}")
    cur = obj
    for key, idx in _PATH_TOKEN.findall(path[1:]):
        if cur is None:
            return None
        if idx != "":
            if not isinstance(cur, list) or int(idx) >= len(cur):
                return None
            cur = cur[int(idx)]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
    return cur


_VAR_RE = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


def substitute(value: Any, bindings: dict[str, Any]) -> Any:
    """Recursively replace ``${name}`` occurrences using ``bindings``.

    A string that is *exactly* ``${name}`` yields the raw bound value (preserving
    non-string types, e.g. an int id); a string with embedded vars is rendered as
    text. Missing bindings are left verbatim so a corpus typo surfaces as a diff
    rather than silently vanishing.
    """
    if isinstance(value, str):
        m = _VAR_RE.fullmatch(value)
        if m and m.group(1) in bindings:
            return bindings[m.group(1)]
        return _VAR_RE.sub(lambda mo: str(bindings.get(mo.group(1), mo.group(0))), value)
    if isinstance(value, list):
        return [substitute(v, bindings) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, bindings) for k, v in value.items()}
    return value


TIMESTAMP_FIELDS = frozenset({
    "creation_time",
    "creation_timestamp",
    "creation_timestamp_ms",
    "last_update_time",
    "last_update_timestamp",
    "last_updated_timestamp",
    "last_updated_timestamp_ms",
    "start_time",
    "start_time_unix_nano",
    "start_time_ms",
    "end_time",
    "end_time_unix_nano",
    "end_time_ms",
    "timestamp",
    "timestamp_ms",
    "request_time",
    "response_time",
    "execution_time_ms",
    "wall_time_ns",
    "created_time",
    "updated_time",
    "create_time",
    "update_time",
    "created_at",
    "last_updated_at",
})

ID_FIELDS = frozenset({
    "experiment_id",
    "run_id",
    "run_uuid",
    "trace_id",
    "request_id",
    "assessment_id",
    "model_id",
    "webhook_id",
    "role_id",
    "user_id",
    "id",
    "delivery_id",
    "eval_id",
    "secret_id",
    "model_definition_id",
    "mapping_id",
    "endpoint_id",
    "budget_policy_id",
    "guardrail_id",
    "dataset_id",
    "job_id",
})

TOKEN_FIELDS = frozenset({
    "next_page_token",
    "page_token",
    "token",
})

PATH_FIELDS = frozenset({
    "artifact_location",
    "artifact_uri",
    "storage_location",
    "source",
})

TS_SENTINEL = "<TS>"
TOKEN_PRESENT = "<TOKEN:present>"
TOKEN_ABSENT = "<TOKEN:absent>"

# Dict keys whose list values are compared order-insensitively (sorted
# canonically before diffing). MLflow never documents tag ordering — Python
# derives tag lists from dicts / un-ordered ORM relationships (e.g.
# ``experiment_tags={tag.key: ... for tag in sql_experiment.tags}``), while the
# Rust store orders by key, so byte-order comparison is meaningless. Scoped to
# ``tags`` only: ordering IS contractual elsewhere (metric history, search
# results).
SORTED_ARRAY_FIELDS = {"tags"}


@dataclass
class NormalizeOptions:
    """Per-case knobs layered on top of the global field sets."""

    drop_paths: list[str] = field(default_factory=list)
    extra_timestamp_fields: list[str] = field(default_factory=list)
    normalize_ids: bool = True
    normalize_paths: bool = True


class _IdCanonicalizer:
    """Maps opaque id values to stable ordinal tags within one response.

    ``exp-a1b2`` and ``exp-c3d4`` from the two servers both collapse to
    ``<ID:0>`` provided they occupy the same structural position, because each
    server's response is canonicalized independently in first-seen order.
    """

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def tag(self, value: Any) -> str:
        key = str(value)
        if key not in self._seen:
            self._seen[key] = f"<ID:{len(self._seen)}>"
        return self._seen[key]


def _path_marker(value: Any) -> str:
    if not isinstance(value, str):
        return "<PATH:nonstr>"
    tail = value.rstrip("/").rsplit("/", 1)[-1] if "/" in value else value
    return f"<PATH:*/{tail}>"


def normalize(body: Any, opts: NormalizeOptions) -> Any:
    """Return a normalized copy of ``body`` with non-deterministic fields erased.

    Timestamps -> ``<TS>``; ids -> stable ``<ID:n>`` tags; token fields removed
    from the compared body and replaced by a presence marker; path fields reduced
    to a basename shape. ``drop_paths`` deletes whole subtrees by ``$.`` path.
    """
    ts_fields = TIMESTAMP_FIELDS | set(opts.extra_timestamp_fields)
    ids = _IdCanonicalizer()

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k, v in node.items():
                if k in ts_fields:
                    out[k] = TS_SENTINEL
                elif k in TOKEN_FIELDS:
                    out[k] = TOKEN_PRESENT if v not in (None, "", []) else TOKEN_ABSENT
                elif opts.normalize_ids and k in ID_FIELDS and isinstance(v, (str, int)):
                    out[k] = ids.tag(v)
                elif opts.normalize_paths and k in PATH_FIELDS:
                    out[k] = _path_marker(v)
                else:
                    walked = _walk(v)
                    if k in SORTED_ARRAY_FIELDS and isinstance(walked, list):
                        walked = sorted(walked, key=lambda e: json.dumps(e, sort_keys=True))
                    out[k] = walked
            return out
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    normalized = _walk(body)
    for p in opts.drop_paths:
        _delete_path(normalized, p)
    return normalized


def _delete_path(obj: Any, path: str) -> None:
    if not path.startswith("$"):
        raise ValueError(f"drop path must start with '$': {path!r}")
    tokens = [(k or None, i if i != "" else None) for k, i in _PATH_TOKEN.findall(path[1:])]
    if not tokens:
        return
    cur = obj
    for key, idx in tokens[:-1]:
        if idx is not None:
            if not isinstance(cur, list) or int(idx) >= len(cur):
                return
            cur = cur[int(idx)]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return
            cur = cur[key]
    key, idx = tokens[-1]
    if idx is not None and isinstance(cur, list) and int(idx) < len(cur):
        del cur[int(idx)]
    elif isinstance(cur, dict) and key in cur:
        del cur[key]


@dataclass
class Diff:
    """A single normalized difference between the Python and Rust responses."""

    json_pointer: str
    python_value: Any
    rust_value: Any
    kind: str


def diff_normalized(py: Any, rust: Any, pointer: str = "") -> list[Diff]:
    """Recursively diff two normalized JSON structures into a list of ``Diff``."""
    diffs: list[Diff] = []
    if type(py) is not type(rust) and not (
        isinstance(py, (int, float))
        and isinstance(rust, (int, float))
        and not isinstance(py, bool)
        and not isinstance(rust, bool)
    ):
        diffs.append(Diff(pointer or "/", py, rust, "type"))
        return diffs
    if isinstance(py, dict):
        for k in py.keys() | rust.keys():
            child = f"{pointer}/{k}"
            if k not in rust:
                diffs.append(Diff(child, py[k], None, "missing_in_rust"))
            elif k not in py:
                diffs.append(Diff(child, None, rust[k], "missing_in_python"))
            else:
                diffs.extend(diff_normalized(py[k], rust[k], child))
    elif isinstance(py, list):
        if len(py) != len(rust):
            diffs.append(Diff(f"{pointer}/__len__", len(py), len(rust), "value"))
        for i, (a, b) in enumerate(zip(py, rust)):
            diffs.extend(diff_normalized(a, b, f"{pointer}/{i}"))
    elif py != rust:
        diffs.append(Diff(pointer or "/", py, rust, "value"))
    return diffs
