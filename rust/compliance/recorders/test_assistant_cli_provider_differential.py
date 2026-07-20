from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests
import uvicorn
from fastapi import FastAPI

from mlflow.assistant.providers import ClaudeCodeProvider, CodexProvider, MlflowGatewayProvider
from mlflow.assistant.providers.base import (
    CLINotInstalledError,
    NotAuthenticatedError,
    clear_config_cache,
)
from mlflow.assistant.types import Event
from mlflow.server.assistant.api import assistant_router

REPO_ROOT = Path(__file__).resolve().parents[3]
RUST_ROOT = REPO_ROOT / "rust"
RUST_RECORDER = RUST_ROOT / "target" / "debug" / "examples" / "assistant_provider_recorder"
DEFAULT_RUST_SERVER = RUST_ROOT / "target" / "debug" / "mlflow-server"
RUST_SERVER = Path(os.environ.get("MLFLOW_RUST_SERVER_BIN", DEFAULT_RUST_SERVER))
TRACKING_URI = "http://127.0.0.1:54321"
MLFLOW_SESSION_ID = "00000000-0000-0000-0000-000000000020"

CLAUDE_MODULE = importlib.import_module("mlflow.assistant.providers.claude_code")
CODEX_MODULE = importlib.import_module("mlflow.assistant.providers.codex")

_DEV_STUBS_SPEC = importlib.util.spec_from_file_location(
    "mlflow_assistant_differential_dev_stubs", REPO_ROOT / "dev" / "dev_stubs" / "__init__.py"
)
assert _DEV_STUBS_SPEC is not None
assert _DEV_STUBS_SPEC.loader is not None
dev_stubs = importlib.util.module_from_spec(_DEV_STUBS_SPEC)
sys.modules[_DEV_STUBS_SPEC.name] = dev_stubs
_DEV_STUBS_SPEC.loader.exec_module(dev_stubs)

SCRIPTED_CLI = r"""
from __future__ import annotations

import json
import os
import signal
import sys
import time


def emit(value):
    if isinstance(value, str):
        line = value
    else:
        line = json.dumps(value, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


args = sys.argv[1:]
is_codex = bool(args and args[0] == "exec")
stdin = sys.stdin.read() if is_codex else None
record = {
    "argv": args,
    "stdin": stdin,
    "cwd": os.getcwd(),
    "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
}
with open(os.environ["MLFLOW_SCRIPTED_CLI_LOG"], "a") as handle:
    handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

is_health = (is_codex and "--ephemeral" in args) or (
    not is_codex
    and "--output-format" in args
    and args[args.index("--output-format") + 1] == "json"
)
if is_health:
    sys.stderr.write(os.environ.get("MLFLOW_SCRIPTED_HEALTH_STDERR", ""))
    sys.stderr.flush()
    emit({"type": "result"})
    raise SystemExit(int(os.environ.get("MLFLOW_SCRIPTED_HEALTH_EXIT", "0")))

for entry in json.loads(os.environ.get("MLFLOW_SCRIPTED_TRANSCRIPT", "[]")):
    if isinstance(entry, dict) and "__sleep__" in entry:
        time.sleep(entry["__sleep__"])
    elif isinstance(entry, dict) and entry.get("__sigkill__"):
        os.kill(os.getpid(), signal.SIGKILL)
    else:
        emit(entry)
"""


@pytest.fixture(scope="module", autouse=True)
def build_rust_recorder():
    subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            "mlflow-server",
            "--bin",
            "mlflow-server",
            "--example",
            "assistant_provider_recorder",
        ],
        cwd=RUST_ROOT,
        check=True,
    )
    assert RUST_RECORDER.exists()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_http(url, process=None):
    for _ in range(100):
        if process is not None and process.poll() is not None:
            raise AssertionError(process.stderr.read())
        try:
            requests.get(url, timeout=0.1)
            return
        except requests.RequestException:
            time.sleep(0.05)
    raise AssertionError(f"server did not start: {url}")


@contextmanager
def _rust_http_server(tmp_path, environment, config):
    database = tmp_path / "route-rust.db"
    shutil.copy(REPO_ROOT / "rust/crates/mlflow-server/tests/fixtures/tracking.db", database)
    home = tmp_path / "route-rust-home"
    config_path = home / ".mlflow/assistant/config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"providers": {"claude_code": config}}))
    temp = tmp_path / "route-rust-tmp"
    temp.mkdir()
    port = _free_port()
    process = subprocess.Popen(
        [
            str(RUST_SERVER),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{database}",
        ],
        cwd=RUST_ROOT,
        env={
            **environment,
            "HOME": str(home),
            "TMPDIR": str(temp),
            "MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE": "true",
            "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_http(f"{base}/health", process)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def _python_http_server(tmp_path, environment, config):
    import mlflow.assistant.config as config_module
    import mlflow.server.assistant.session as session_module

    config_path = tmp_path / "route-python-config.json"
    config_path.write_text(json.dumps({"providers": {"claude_code": config}}))
    port = _free_port()
    app = FastAPI()
    app.include_router(assistant_router)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    with (
        patch.object(config_module, "CONFIG_PATH", config_path),
        patch.object(session_module, "SESSION_DIR", tmp_path / "route-python-sessions"),
        patch.dict(os.environ, environment, clear=True),
    ):
        clear_config_cache()
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        try:
            _wait_http(f"{base}/ajax-api/3.0/mlflow/assistant/config")
            yield base
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            clear_config_cache()


def _provider(provider: str):
    return ClaudeCodeProvider() if provider == "claude_code" else CodexProvider()


def _provider_module(provider: str):
    return CLAUDE_MODULE if provider == "claude_code" else CODEX_MODULE


def _config(
    provider: str, *, model: str = "default", permissions: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "model": model,
        "permissions": permissions
        or {
            "allow_edit_files": True,
            "allow_read_docs": True,
            "full_access": False,
        },
    }


def _stream_request(tmp_path: Path, *, session_id: str | None = None) -> dict[str, Any]:
    return {
        "prompt": "Explain café traces 😀",
        "tracking_uri": TRACKING_URI,
        "session_id": session_id,
        "cwd": str(tmp_path),
        "context": {"experimentId": "20", "selectedTraceIds": ["tr-1", "tr-2"]},
    }


@contextmanager
def _scripted_path(tmp_path: Path, provider: str):
    binary = "claude" if provider == "claude_code" else "codex"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / binary
    script.write_text(f"#!{sys.executable}\n{SCRIPTED_CLI}")
    script.chmod(0o755)
    yield bin_dir


def _write_python_config(tmp_path: Path, provider: str, config: dict[str, Any]) -> Path:
    path = tmp_path / "assistant-config.json"
    path.write_text(json.dumps({"providers": {provider: config}}))
    return path


async def _python_stream_async(
    provider_name: str,
    config: dict[str, Any],
    request: dict[str, Any],
    environment: dict[str, str],
    cancel_after_events: int | None,
) -> list[str]:
    provider = _provider(provider_name)
    provider_module = _provider_module(provider_name)
    config_path = _write_python_config(Path(request["cwd"]), provider_name, config)
    process_pid: list[int] = []

    def save_pid(_session_id, pid):
        process_pid.append(pid)

    frames = []
    clear_config_cache()
    try:
        with (
            patch("mlflow.assistant.config.CONFIG_PATH", config_path),
            patch.object(provider_module, "save_process_pid", save_pid),
            patch.object(provider_module, "clear_process_pid", lambda _session_id: None),
            patch.dict(os.environ, environment, clear=True),
        ):
            events = provider.astream(
                prompt=request["prompt"],
                tracking_uri=request["tracking_uri"],
                session_id=request.get("session_id"),
                mlflow_session_id=MLFLOW_SESSION_ID,
                cwd=Path(request["cwd"]),
                context=request.get("context"),
            )
            async for event in events:
                frames.append(event.to_sse_event())
                if cancel_after_events == len(frames):
                    assert process_pid
                    os.kill(process_pid[-1], signal.SIGTERM)
    finally:
        clear_config_cache()
    return frames


def _python_stream(
    provider: str,
    config: dict[str, Any],
    request: dict[str, Any],
    environment: dict[str, str],
    cancel_after_events: int | None = None,
) -> list[str]:
    return asyncio.run(
        _python_stream_async(provider, config, request, environment, cancel_after_events)
    )


def _rust_record(payload: dict[str, Any], environment: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        [str(RUST_RECORDER)],
        cwd=REPO_ROOT,
        env=environment,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _rust_stream(
    provider: str,
    config: dict[str, Any],
    request: dict[str, Any],
    environment: dict[str, str],
    cancel_after_events: int | None = None,
) -> list[str]:
    payload = {
        "action": "stream",
        "provider": provider,
        "config": config,
        "stream": request,
        "cancel_after_events": cancel_after_events,
    }
    return _rust_record(payload, environment)["frames"]


def _read_invocations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _assert_scripted_stream_parity(
    tmp_path: Path,
    provider: str,
    transcript: list[Any],
    *,
    config: dict[str, Any] | None = None,
    session_id: str | None = None,
    cancel_after_events: int | None = None,
) -> list[str]:
    config = config or _config(provider)
    request = _stream_request(tmp_path, session_id=session_id)
    log = tmp_path / "invocations.jsonl"
    with _scripted_path(tmp_path, provider) as bin_dir:
        environment = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "MLFLOW_SCRIPTED_CLI_LOG": str(log),
            "MLFLOW_SCRIPTED_TRANSCRIPT": json.dumps(transcript),
        }
        python_frames = _python_stream(provider, config, request, environment, cancel_after_events)
        rust_frames = _rust_stream(provider, config, request, environment, cancel_after_events)

    assert rust_frames == python_frames
    python_invocation, rust_invocation = _read_invocations(log)
    assert rust_invocation == python_invocation
    return rust_frames


def test_dev_claude_stub_frames_are_identical(tmp_path):
    stubs = dev_stubs.install_stubs(["claude"])
    try:
        environment = dict(os.environ)
        prepend = os.pathsep.join(str(path) for path in stubs.path_prepend)
        environment["PATH"] = f"{prepend}{os.pathsep}{environment.get('PATH', '')}"
        config = _config("claude_code")
        request = _stream_request(tmp_path, session_id="mlflow-dev-stub-session")
        python_frames = _python_stream("claude_code", config, request, environment)
        rust_frames = _rust_stream("claude_code", config, request, environment)
    finally:
        for path in stubs.cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)

    assert rust_frames == python_frames
    assert [frame.split("\n", 1)[0] for frame in rust_frames] == [
        "event: message",
        "event: stream_event",
        "event: done",
    ]


CLAUDE_TRANSCRIPT = [
    {"type": "system", "subtype": "init", "session_id": "claude-session"},
    {
        "type": "user",
        "message": {
            "content": [
                {"type": "text", "text": "Base directory for this skill: /fixture"},
                {"type": "text", "text": "must remain hidden"},
            ]
        },
    },
    {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "visible café 😀"},
                {"type": "thinking", "thinking": "reason", "signature": "fake-signature"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "mlflow --help"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "ok",
                    "is_error": False,
                },
            ]
        },
    },
    {"type": "stream_event", "event": {"type": "content_delta", "text": "chunk"}},
    {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}},
    {
        "type": "rate_limit_event",
        "rate_limit_info": {"status": "limited", "resetsAt": "soon"},
    },
    "plain non-json output",
    {
        "type": "result",
        "result": "complete",
        "session_id": "claude-session",
        "total_cost_usd": 0.1319,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 35,
            "cache_read_input_tokens": 100,
            "output_tokens": 5,
        },
    },
]


@pytest.mark.parametrize(
    "permissions",
    [
        {"allow_edit_files": False, "allow_read_docs": False, "full_access": False},
        {"allow_edit_files": True, "allow_read_docs": False, "full_access": False},
        {"allow_edit_files": False, "allow_read_docs": True, "full_access": False},
        {"allow_edit_files": True, "allow_read_docs": True, "full_access": False},
        {"allow_edit_files": True, "allow_read_docs": True, "full_access": True},
    ],
)
def test_claude_scripted_filter_usage_and_permission_frames(tmp_path, permissions):
    frames = _assert_scripted_stream_parity(
        tmp_path,
        "claude_code",
        CLAUDE_TRANSCRIPT,
        config=_config("claude_code", model="claude-fixture-model", permissions=permissions),
    )
    assert "must remain hidden" not in "".join(frames)
    assert [frame.split("\n", 1)[0] for frame in frames][-2:] == [
        "event: stream_event",
        "event: done",
    ]


CODEX_TRANSCRIPT = [
    "not json",
    {"type": "thread.started", "thread_id": "codex-thread"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "mcp_tool_call", "text": "hidden"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": ""}},
    {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "Codex café 😀"},
    },
    {
        "type": "turn.completed",
        "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 5},
    },
]


@pytest.mark.parametrize("full_access", [False, True])
def test_codex_scripted_filter_usage_and_permission_frames(tmp_path, full_access):
    frames = _assert_scripted_stream_parity(
        tmp_path,
        "codex",
        CODEX_TRANSCRIPT,
        config=_config(
            "codex",
            model="o4-mini",
            permissions={
                "allow_edit_files": not full_access,
                "allow_read_docs": not full_access,
                "full_access": full_access,
            },
        ),
    )
    assert "hidden" not in "".join(frames)
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: message",
        "event: stream_event",
        "event: done",
    ]


def test_codex_resume_argv_and_stdin_are_identical(tmp_path):
    _assert_scripted_stream_parity(
        tmp_path,
        "codex",
        CODEX_TRANSCRIPT,
        session_id="codex-existing-thread",
    )


@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_cancellation_mid_stream_frames_are_identical(tmp_path, provider):
    first = (
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "first"}]}}
        if provider == "claude_code"
        else {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}
    )
    frames = _assert_scripted_stream_parity(
        tmp_path,
        provider,
        [first, {"__sleep__": 60}],
        cancel_after_events=1,
    )
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: message",
        "event: error",
    ]
    assert "Process exited with code -15" in frames[-1]


@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_sigkill_frames_are_identical_and_end_interrupted(tmp_path, provider):
    first = (
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "first"}]}}
        if provider == "claude_code"
        else {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}
    )
    frames = _assert_scripted_stream_parity(
        tmp_path,
        provider,
        [first, {"__sigkill__": True}],
    )
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: message",
        "event: interrupted",
    ]


def _python_health(provider: str, environment: dict[str, str]) -> dict[str, Any]:
    try:
        with patch.dict(os.environ, environment, clear=True):
            _provider(provider).check_connection()
    except CLINotInstalledError as error:
        return {"status": 412, "body": {"detail": str(error)}}
    except NotAuthenticatedError as error:
        return {"status": 401, "body": {"detail": str(error)}}
    return {"status": 200, "body": {"status": "ok"}}


def _rust_health(provider: str, environment: dict[str, str]) -> dict[str, Any]:
    return _rust_record({"action": "health", "provider": provider}, environment)


@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_health_not_installed_412_body_is_identical(tmp_path, provider):
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    environment = {**os.environ, "PATH": str(empty_path)}
    assert _rust_health(provider, environment) == _python_health(provider, environment)


@pytest.mark.parametrize(
    ("provider", "stderr"),
    [("claude_code", "Login required"), ("codex", "OPENAI API key missing")],
)
def test_health_not_authenticated_401_body_and_argv_are_identical(tmp_path, provider, stderr):
    log = tmp_path / "health-invocations.jsonl"
    with _scripted_path(tmp_path, provider) as bin_dir:
        environment = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "MLFLOW_SCRIPTED_CLI_LOG": str(log),
            "MLFLOW_SCRIPTED_HEALTH_EXIT": "1",
            "MLFLOW_SCRIPTED_HEALTH_STDERR": stderr,
        }
        assert _rust_health(provider, environment) == _python_health(provider, environment)
    python_invocation, rust_invocation = _read_invocations(log)
    assert rust_invocation == python_invocation


@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_health_success_200_body_and_argv_are_identical(tmp_path, provider):
    log = tmp_path / "health-invocations.jsonl"
    with _scripted_path(tmp_path, provider) as bin_dir:
        environment = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "MLFLOW_SCRIPTED_CLI_LOG": str(log),
            "MLFLOW_SCRIPTED_HEALTH_EXIT": "0",
        }
        assert _rust_health(provider, environment) == _python_health(provider, environment)
    python_invocation, rust_invocation = _read_invocations(log)
    assert rust_invocation == python_invocation


def test_health_not_implemented_501_body_is_identical():
    try:
        MlflowGatewayProvider().check_connection()
    except NotImplementedError as error:
        detail = str(error)
    else:
        raise AssertionError("MLflow Gateway health probe unexpectedly succeeded")
    rust = _rust_record(
        {"action": "health_mapping", "health_error": "not_implemented", "detail": detail},
        dict(os.environ),
    )
    assert rust == {"status": 501, "body": {"detail": detail}}


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError(),
        ValueError(),
        RuntimeError("boom"),
        ValueError("bad value"),
    ],
)
def test_exception_fallback_frame_is_non_empty_and_identical(error):
    python = Event.from_exception(error).to_sse_event()
    rust = _rust_record(
        {
            "action": "exception_mapping",
            "detail": str(error),
            "exception_repr": repr(error),
        },
        dict(os.environ),
    )["frame"]
    assert rust == python
    assert json.loads(rust.split("data: ", 1)[1])["error"]


def test_full_stub_cli_chat_is_frame_identical_through_real_http_routes(tmp_path):
    config = _config("claude_code", model="claude-fixture-model")
    config["selected"] = True
    log = tmp_path / "route-invocations.jsonl"
    with _scripted_path(tmp_path, "claude_code") as bin_dir:
        environment = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "MLFLOW_SCRIPTED_CLI_LOG": str(log),
            "MLFLOW_SCRIPTED_TRANSCRIPT": json.dumps(CLAUDE_TRANSCRIPT),
        }

        def chat(base):
            response = requests.post(
                f"{base}/ajax-api/3.0/mlflow/assistant/message",
                json={"message": "Explain café traces 😀", "context": {"experimentId": "20"}},
                timeout=10,
            )
            response.raise_for_status()
            session_id = response.json()["session_id"]
            stream = requests.get(
                f"{base}/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream",
                timeout=10,
            )
            stream.raise_for_status()
            return stream.content

        with _python_http_server(tmp_path, environment, config) as python_base:
            python_frames = chat(python_base)
        with _rust_http_server(tmp_path, environment, config) as rust_base:
            rust_frames = chat(rust_base)

    assert rust_frames == python_frames
    assert [line for line in rust_frames.splitlines() if line.startswith(b"event: ")] == [
        b"event: message",
        b"event: stream_event",
        b"event: message",
        b"event: message",
        b"event: stream_event",
        b"event: done",
    ]
