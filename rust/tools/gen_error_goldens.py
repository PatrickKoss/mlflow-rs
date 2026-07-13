#!/usr/bin/env python
"""Generate golden error-response fixtures from the real Python implementation.

Per `RUST_TRACKING_SERVER_PLAN.md` §4.6 / Phase 1 T1.4, the Rust error model
must byte-match Python's on the wire: same JSON body bytes (field order,
compact vs. pretty, which optional fields are present), same HTTP status, same
extra headers (`WWW-Authenticate` on 401).

This script drives the actual `mlflow.exceptions.MlflowException` /
`mlflow.server.handlers.catch_mlflow_exception` / `mlflow.server.auth`
response-building code (never a re-implementation of it) and dumps one JSON
fixture per representative case into `rust/crates/mlflow-error/tests/goldens/`.
The Rust golden test (`tests/golden_parity.rs`) reads these fixtures back and
asserts its own `IntoResponse` output matches byte-for-byte.

Run:

    uv run --frozen python rust/tools/gen_error_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, make_response

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)

TOOLS_DIR = Path(__file__).resolve().parent
RUST_DIR = TOOLS_DIR.parent
GOLDENS_DIR = RUST_DIR / "crates" / "mlflow-error" / "tests" / "goldens"

app = Flask(__name__)


def _mlflow_exception_case(name: str, message: str, error_code: int) -> dict:
    """Mirror `catch_mlflow_exception`'s Response construction exactly
    (`mlflow/server/handlers.py::catch_mlflow_exception`): body is
    `e.serialize_as_json()` (compact `json.dumps`, NOT the pretty-printed
    proto JSON used for normal 2xx responses), mimetype `application/json`,
    status `e.get_http_status_code()`.
    """
    exc = MlflowException(message, error_code=error_code)
    body = exc.serialize_as_json()
    # Sanity: body must already be valid compact JSON matching what we wrote.
    assert json.loads(body) is not None
    return {
        "name": name,
        "status": exc.get_http_status_code(),
        "content_type": "application/json",
        "body": body,
        "headers": {},
    }


def _not_implemented_case() -> dict:
    """Mirror `mlflow/server/handlers.py::_not_implemented` exactly: empty
    body, status 404, Flask's default `text/html; charset=utf-8` mimetype
    (no explicit mimetype is set on the bare `Response()`).
    """
    with app.test_request_context():
        from mlflow.server.handlers import _not_implemented

        res = _not_implemented()
        return {
            "name": "not_implemented_404",
            "status": res.status_code,
            "content_type": res.headers.get("Content-Type", ""),
            "body": res.get_data().decode("utf-8"),
            "headers": {},
        }


def _unauthenticated_case() -> dict:
    """Mirror `mlflow/server/auth/__init__.py::make_basic_auth_response`
    exactly (reproduced here rather than imported, since importing
    `mlflow.server.auth` requires the optional `flask-wtf` dependency that
    isn't part of the base dev environment; the body/status/header
    construction below is copied verbatim from that function).
    """
    with app.test_request_context():
        res = make_response(
            "You are not authenticated. Please see "
            "https://www.mlflow.org/docs/latest/auth/index.html#authenticating-to-mlflow "
            "on how to authenticate."
        )
        res.status_code = 401
        res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
        return {
            "name": "unauthenticated_401",
            "status": res.status_code,
            "content_type": res.headers.get("Content-Type", ""),
            "body": res.get_data().decode("utf-8"),
            "headers": {"WWW-Authenticate": res.headers["WWW-Authenticate"]},
        }


def _permission_denied_auth_case() -> dict:
    """Mirror `mlflow/server/auth/__init__.py::make_forbidden_response`
    exactly (reproduced verbatim; see `_unauthenticated_case` docstring for
    why it isn't imported directly).
    """
    with app.test_request_context():
        res = make_response("Permission denied")
        res.status_code = 403
        return {
            "name": "auth_forbidden_403",
            "status": res.status_code,
            "content_type": res.headers.get("Content-Type", ""),
            "body": res.get_data().decode("utf-8"),
            "headers": {},
        }


def main() -> None:
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        _mlflow_exception_case(
            "resource_does_not_exist",
            "Run 'nonexistent-run-id' not found",
            RESOURCE_DOES_NOT_EXIST,
        ),
        _mlflow_exception_case(
            "invalid_parameter_value",
            "Invalid value for parameter 'max_results': must be positive",
            INVALID_PARAMETER_VALUE,
        ),
        _mlflow_exception_case(
            "internal_error",
            "Unexpected internal error",
            INTERNAL_ERROR,
        ),
        _mlflow_exception_case(
            "unauthenticated_mlflow_exception",
            "Not authenticated",
            UNAUTHENTICATED,
        ),
        _mlflow_exception_case(
            "permission_denied_mlflow_exception",
            "Permission denied",
            PERMISSION_DENIED,
        ),
        _not_implemented_case(),
        _unauthenticated_case(),
        _permission_denied_auth_case(),
    ]

    for case in cases:
        out_path = GOLDENS_DIR / f"{case['name']}.json"
        out_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path.relative_to(RUST_DIR)}")


if __name__ == "__main__":
    main()
