from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from mlflow.assistant.providers.base import clear_config_cache
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
RUST_ROOT = REPO_ROOT / "rust"
RECORDER = RUST_ROOT / "target" / "debug" / "examples" / "assistant_openai_recorder"
SESSION_ID = "00000000-0000-0000-0000-000000000020"


class _Script:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []


@contextmanager
def scripted_server(turns):
    script = _Script(turns)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["content-length"])
            script.requests.append(json.loads(self.rfile.read(length)))
            chunks = script.turns.pop(0)
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            body += "data: [DONE]\n\n"
            encoded = body.encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", script
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture(scope="module", autouse=True)
def build_recorder():
    subprocess.run(
        ["cargo", "build", "-p", "mlflow-server", "--example", "assistant_openai_recorder"],
        cwd=RUST_ROOT,
        check=True,
    )


def _config(path, base_url, permissions=None):
    value = {
        "providers": {
            "oai_test": {
                "model": "fixture-model",
                "base_url": base_url,
                "permissions": permissions
                or {
                    "allow_edit_files": True,
                    "allow_read_docs": True,
                    "full_access": False,
                },
            }
        }
    }
    path.write_text(json.dumps(value))
    return value["providers"]["oai_test"]


def _provider():
    return OpenAICompatibleProvider(
        name="oai_test",
        display_name="OAI Test",
        description="fixture",
        connection_hint="fixture hint",
        list_models_fn=lambda *_args: ["fixture-model"],
    )


def _request(cwd, base_url, *, session_id=None, context=None, permissions=None):
    return {
        "base_url": base_url,
        "model": "fixture-model",
        "permissions": permissions
        or {
            "allow_edit_files": True,
            "allow_read_docs": True,
            "full_access": False,
        },
        "prompt": "perform fixture",
        "tracking_uri": "http://127.0.0.1:5000",
        "session_id": session_id,
        "mlflow_session_id": SESSION_ID,
        "cwd": str(cwd),
        "context": context or {},
    }


def _python_frames(tmp_path, base_url, request):
    config_path = tmp_path / "assistant-config.json"
    _config(config_path, base_url, request["permissions"])
    clear_config_cache()

    async def collect():
        events = _provider().astream(
            prompt=request["prompt"],
            tracking_uri=request["tracking_uri"],
            session_id=request.get("session_id"),
            mlflow_session_id=request["mlflow_session_id"],
            cwd=Path(request["cwd"]),
            context=request["context"],
        )
        return [event.to_sse_event() async for event in events]

    try:
        with patch("mlflow.assistant.config.CONFIG_PATH", config_path):
            return asyncio.run(collect())
    finally:
        clear_config_cache()


def _rust_frames(request):
    result = subprocess.run(
        [str(RECORDER)],
        cwd=REPO_ROOT,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return json.loads(result.stdout)["frames"]


def _delta(*, content=None, tool_calls=None):
    delta = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta, "index": 0}]}


def _compare(tmp_path, turns, request_factory):
    with scripted_server(turns) as (base_url, python_script):
        request = request_factory(base_url)
        python = _python_frames(tmp_path, base_url, request)
        python_requests = python_script.requests
    with scripted_server(turns) as (base_url, rust_script):
        request = request_factory(base_url)
        rust = _rust_frames(request)
        rust_requests = rust_script.requests
    assert rust == python
    assert rust_requests == python_requests
    return rust


def _done_session(frames):
    return json.loads(frames[-1].split("data: ", 1)[1])["session_id"]


def test_multi_tool_transcript_is_frame_and_request_identical(tmp_path):
    turns = [
        [
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "write-1",
                        "function": {
                            "name": "Write",
                            "arguments": '{"file_path":"note.txt","content":"alpha"}',
                        },
                    },
                    {
                        "index": 1,
                        "id": "read-1",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"note.txt"}',
                        },
                    },
                ]
            )
        ],
        [
            _delta(content="Done"),
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        ],
    ]
    frames = _compare(tmp_path, turns, lambda base: _request(tmp_path, base))
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: message",
        "event: message",
        "event: message",
        "event: message",
        "event: stream_event",
        "event: stream_event",
        "event: done",
    ]


def test_permission_pause_resume_transcript_is_identical(tmp_path):
    pause_turn = [
        [
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "bash-1",
                        "function": {"name": "Bash", "arguments": '{"command":"printf resumed"}'},
                    }
                ]
            )
        ]
    ]
    paused = _compare(tmp_path, pause_turn, lambda base: _request(tmp_path, base))
    assert paused[-2].startswith("event: permission_request")
    history = _done_session(paused)

    resumed = _compare(
        tmp_path,
        [[_delta(content="Resumed")]],
        lambda base: _request(
            tmp_path,
            base,
            session_id=history,
            context={"tool_decisions": {"bash-1": "allow"}},
        ),
    )
    assert [frame.split("\n", 1)[0] for frame in resumed] == [
        "event: message",
        "event: stream_event",
        "event: done",
    ]


def test_trim_boundary_transcript_is_identical(tmp_path):
    big = "x" * 180_000
    history = json.dumps([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"old-{big}"},
        {"role": "assistant", "content": f"old-answer-{big}"},
        {"role": "user", "content": f"new-{big}"},
    ])
    frames = _compare(
        tmp_path,
        [[_delta(content="final")]],
        lambda base: _request(tmp_path, base, session_id=history),
    )
    final = _done_session(frames)
    assert len(final.encode()) <= 500 * 1024
    assert not any(
        message.get("content", "").startswith("old-")
        for message in json.loads(final)
        if isinstance(message.get("content"), str)
    )
