"""T20.1 Assistant HTTP/SSE differential against an in-process provider.

The 27 comparisons cover all nine routes, FastAPI errors, session lifecycle,
and full SSE frames. No CLI process or live provider is used; T20.2 owns that
provider execution layer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import requests
import uvicorn
from fastapi import FastAPI

from mlflow.assistant.providers.base import AssistantProvider
from mlflow.assistant.types import Event, Message, TextBlock
from mlflow.server.assistant.api import assistant_router

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUST_BINARY = REPO_ROOT / "rust" / "target" / "debug" / "mlflow-server"
RUST_BINARY = Path(os.environ.get("MLFLOW_RUST_SERVER_BIN", DEFAULT_RUST_BINARY))
FIXTURE_DB = REPO_ROOT / "rust" / "crates" / "mlflow-server" / "tests" / "fixtures" / "tracking.db"
PREFIX = "/ajax-api/3.0/mlflow/assistant"
STUB_REPLY = (
    "This is a synthetic reply from the MLflow dev stub Claude CLI. The real "
    "Claude Code provider is replaced so the Assistant chat panel can be reviewed "
    "without credentials or LLM calls. No model was invoked to produce this message."
)


class ScriptedProvider(AssistantProvider):
    @property
    def name(self):
        return "claude_code"

    @property
    def display_name(self):
        return "Claude Code"

    @property
    def description(self):
        return "Differential fixture"

    def is_available(self):
        return True

    def check_connection(self, echo=None):
        return None

    def resolve_skills_path(self, base_directory):
        return base_directory / ".claude" / "skills"

    async def astream(
        self,
        prompt,
        tracking_uri,
        session_id=None,
        mlflow_session_id=None,
        cwd=None,
        context=None,
    ):
        session_id = session_id or "mlflow-dev-stub-pythonfixture"
        yield Event.from_message(Message(role="assistant", content=[TextBlock(text=STUB_REPLY)]))
        yield Event.from_stream_event({
            "type": "usage",
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 24,
                "total_tokens": 32,
                "total_cost_usd": 0.0,
            },
        })
        yield Event.from_result(STUB_REPLY, session_id)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def rust_server(tmp_path):
    assert RUST_BINARY.exists(), f"build first: cargo build -p mlflow-server ({RUST_BINARY})"
    database = tmp_path / "rust.db"
    shutil.copy(FIXTURE_DB, database)
    home = tmp_path / "rust-home"
    home.mkdir()
    sessions = tmp_path / "rust-tmp"
    sessions.mkdir()
    port = _free_port()
    env = {
        **os.environ,
        "HOME": str(home),
        "TMPDIR": str(sessions),
        "MLFLOW_ASSISTANT_DEV_STUB_PROVIDERS": "claude",
        "MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE": "true",
        "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
    }
    process = subprocess.Popen(
        [
            str(RUST_BINARY),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{database}",
        ],
        cwd=REPO_ROOT / "rust",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(base, process)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def python_server(tmp_path, stack):
    import mlflow.assistant.config as config_module
    import mlflow.server.assistant.session as session_module

    home = tmp_path / "python-home"
    home.mkdir()
    stack.enter_context(
        patch.object(config_module, "CONFIG_PATH", home / ".mlflow" / "assistant" / "config.json")
    )
    stack.enter_context(patch.object(session_module, "SESSION_DIR", tmp_path / "python-sessions"))
    stack.enter_context(
        patch("mlflow.server.assistant.api.list_providers", return_value=[ScriptedProvider()])
    )
    app = FastAPI()
    app.include_router(assistant_router)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(base)
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _wait_ready(base, process=None):
    for _ in range(100):
        if process is not None and process.poll() is not None:
            raise AssertionError(process.stderr.read())
        try:
            requests.get(f"{base}/health", timeout=0.1)
            return
        except requests.RequestException:
            time.sleep(0.05)
    raise AssertionError(f"server did not start: {base}")


def _normalize(content, session_id):
    if session_id:
        content = content.replace(session_id.encode(), b"<mlflow-session-id>")
    return re.sub(rb"mlflow-dev-stub-[A-Za-z0-9]+", b"<provider-session-id>", content)


def _compare(py_response, rs_response, py_session="", rs_session="", d18=False):
    assert py_response.status_code == rs_response.status_code
    py_body = py_response.content
    rs_body = rs_response.content
    if d18:
        py_value = py_response.json()
        py_value["stream_url"] = (
            py_value["stream_url"].replace("/assistant/stream/", "/assistant/sessions/") + "/stream"
        )
        py_body = json.dumps(py_value, separators=(",", ":")).encode()
    py_body = _normalize(py_body, py_session)
    rs_body = _normalize(rs_body, rs_session)
    assert py_body == rs_body, f"PY={py_body!r}\nRS={rs_body!r}"
    assert py_response.headers.get("content-type") == rs_response.headers.get("content-type")
    if py_response.headers.get("content-type", "").startswith("text/event-stream"):
        for name in ("cache-control", "connection", "x-accel-buffering"):
            assert py_response.headers.get(name) == rs_response.headers.get(name)


def test_python_rust_assistant_27_case_differential(tmp_path):
    with ExitStack() as stack, rust_server(tmp_path) as rust_base:
        with python_server(tmp_path, stack) as python_base:
            python = requests.Session()
            rust = requests.Session()
            stack.callback(python.close)
            stack.callback(rust.close)

            def pair(method, path, **kwargs):
                py = python.request(method, python_base + path, timeout=10, **kwargs)
                rs = rust.request(method, rust_base + path, timeout=10, **kwargs)
                return py, rs

            comparisons = 0

            def compare(method, path, **kwargs):
                nonlocal comparisons
                py, rs = pair(method, path, **kwargs)
                _compare(py, rs)
                comparisons += 1
                return py, rs

            compare("GET", f"{PREFIX}/config")  # 1
            compare("GET", f"{PREFIX}/providers/claude_code/health")  # 2
            compare("GET", f"{PREFIX}/providers/missing/health")  # 3
            compare("GET", f"{PREFIX}/providers/claude_code/models")  # 4
            compare("GET", f"{PREFIX}/providers/missing/models")  # 5
            compare("PUT", f"{PREFIX}/config", json={})  # 6
            compare("POST", f"{PREFIX}/skills/install", json={"type": "invalid"})  # 7
            compare("POST", f"{PREFIX}/message")  # 8
            compare("POST", f"{PREFIX}/message", json={})  # 9
            compare("POST", f"{PREFIX}/message", json={"message": 1})  # 10
            compare("POST", f"{PREFIX}/message", json={"message": "x", "context": []})  # 11

            py, rs = pair(
                "PUT",
                f"{PREFIX}/config",
                json={"providers": {"claude_code": {"selected": True}}},
            )
            _compare(py, rs)
            comparisons += 1  # 12
            compare("GET", f"{PREFIX}/config")  # 13
            compare("POST", f"{PREFIX}/skills/install", json={"type": "custom"})  # 14
            compare("POST", f"{PREFIX}/skills/install", json={"type": "project"})  # 15

            py, rs = pair(
                "POST",
                f"{PREFIX}/message",
                json={"message": "hello", "context": {"traceId": "tr-1"}},
            )
            py_session = py.json()["session_id"]
            rs_session = rs.json()["session_id"]
            _compare(py, rs, py_session, rs_session, d18=True)
            comparisons += 1  # 16

            py = python.get(python_base + f"{PREFIX}/sessions/{py_session}/stream", timeout=10)
            rs = rust.get(rust_base + f"{PREFIX}/sessions/{rs_session}/stream", timeout=10)
            _compare(py, rs, py_session, rs_session)
            comparisons += 1  # 17

            py = python.get(python_base + f"{PREFIX}/sessions/{py_session}/stream", timeout=10)
            rs = rust.get(rust_base + f"{PREFIX}/sessions/{rs_session}/stream", timeout=10)
            _compare(py, rs, py_session, rs_session)
            comparisons += 1  # 18
            compare(
                "POST",
                f"{PREFIX}/sessions/not-a-uuid/permission",
                json={"request_id": "tool-1", "decision": "allow"},
            )  # 19
            missing = "00000000-0000-0000-0000-000000000001"
            compare(
                "POST",
                f"{PREFIX}/sessions/{missing}/permission",
                json={"request_id": "tool-1", "decision": "allow"},
            )  # 20
            compare(
                "POST",
                f"{PREFIX}/sessions/{missing}/permission",
                json={"request_id": "tool-1", "decision": "invalid"},
            )  # 21
            compare(
                "PATCH",
                f"{PREFIX}/sessions/{missing}",
                json={"status": "invalid"},
            )  # 22
            compare("GET", f"{PREFIX}/sessions/{missing}/stream")  # 23

            py = python.post(
                python_base + f"{PREFIX}/sessions/{py_session}/permission",
                json={"request_id": "tool-1", "decision": "allow"},
                timeout=10,
            )
            rs = rust.post(
                rust_base + f"{PREFIX}/sessions/{rs_session}/permission",
                json={"request_id": "tool-1", "decision": "allow"},
                timeout=10,
            )
            _compare(py, rs, py_session, rs_session)
            comparisons += 1  # 24
            py = python.get(python_base + f"{PREFIX}/sessions/{py_session}/stream", timeout=10)
            rs = rust.get(rust_base + f"{PREFIX}/sessions/{rs_session}/stream", timeout=10)
            _compare(py, rs, py_session, rs_session)
            comparisons += 1  # 25
            py = python.patch(
                python_base + f"{PREFIX}/sessions/{py_session}",
                json={"status": "cancelled"},
                timeout=10,
            )
            rs = rust.patch(
                rust_base + f"{PREFIX}/sessions/{rs_session}",
                json={"status": "cancelled"},
                timeout=10,
            )
            _compare(py, rs, py_session, rs_session)
            comparisons += 1  # 26
            compare(
                "PUT",
                f"{PREFIX}/config",
                json={"projects": {"7": {"location": "/definitely/missing/path"}}},
            )  # 27

            assert comparisons == 27
