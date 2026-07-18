"""T18.3 frame/byte differential against one hermetic provider server.

Run after building the Rust server:
  uv run --frozen pytest -q rust/compliance/recorders/test_gateway_runtime_differential.py

No live provider hostname or credential is reachable from this test. The only
egress target is a loopback ``ThreadingHTTPServer`` and every stored key is an
obvious fake.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import PropertyMock, patch

import requests
import uvicorn
from fastapi import FastAPI

from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.gateway.providers.gemini import GeminiProvider
from mlflow.server.fastapi_app import add_gateway_timing_middleware
from mlflow.server.gateway_api import gateway_router
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

REPO_ROOT = Path(__file__).resolve().parents[3]
RUST_BINARY = REPO_ROOT / "rust" / "target" / "debug" / "mlflow-server"
FIXED_TIME = 1_750_000_000
FAKE_PASSPHRASE = "obvious-fake-differential-passphrase"
PROVIDERS = ("openai", "azure", "anthropic", "gemini")


class MockProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def do_POST(self):
        size = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(size))
        assert self.headers.get("accept-encoding") == "gzip, deflate, identity"
        assert self.headers.get("x-mlflow-authorization") is None

        text = _find_text(body)
        # Keep provider duration above both implementations' integer-millisecond
        # threshold so timing-header presence is deterministic.
        time.sleep(0.005)
        if text == "error-429":
            self._json(429, {"error": {"message": "fixture provider limit"}})
            return
        if text == "error-500":
            self._json(500, {"error": {"message": "fixture provider failure"}})
            return

        streaming = "streamGenerateContent" in self.path or body.get("stream") is True
        if streaming:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("connection", "close")
            self.end_headers()
            chunks = _stream_chunks(self.path, text == "mid-stream-error")
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
            return

        if self.path.endswith("/messages"):
            value = {
                "id": "anthropic-fixture-id",
                "model": "fixture-model",
                "role": "assistant",
                "content": [{"type": "text", "text": "fixture answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        elif "generateContent" in self.path:
            value = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "fixture answer"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 5,
                },
            }
        else:
            value = {
                "id": "openai-fixture-id",
                "object": "chat.completion",
                "created": 7,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fixture answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        self._json(200, value)

    def _json(self, status: int, value: dict):
        data = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _find_text(body: dict) -> str:
    try:
        return body["messages"][0]["content"]
    except KeyError:
        return body.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")


def _stream_chunks(path: str, fail: bool) -> list[bytes]:
    if path.endswith("/messages"):
        chunks = [
            b": keep-alive\n\nevent: message_start\n",
            b'data: {"type":"message_start","message":{"id":"anthropic-stream-id",'
            b'"model":"fixture-model","usage":{"input_tokens":2}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"fixture "}}\n\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":3}}\n\n',
        ]
    elif "streamGenerateContent" in path:
        chunks = [
            b': keep-alive\n\ndata: {"candidates":[{"content":{"parts":[{"text":"fixture "}]}}]}\n',
            b'\ndata: {"candidates":[{"content":{"parts":[{"text":"answer"}]},'
            b'"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,'
            b'"candidatesTokenCount":3,"totalTokenCount":5}}\n\ndata: [DONE]\n\n',
        ]
    else:
        chunks = [
            b': keep-alive\n\ndata: {"id":"openai-stream-id","object":"chat.completion.chunk",'
            b'"created":7,"model":"fixture-model","choices":[{"index":0,"delta":'
            b'{"role":"assistant","content":"fixture "},"finish_reason":null}]}\n',
            b'\ndata: {"id":"openai-stream-id","object":"chat.completion.chunk",'
            b'"created":7,"model":"fixture-model","choices":[{"index":0,"delta":'
            b'{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,'
            b'"completion_tokens":3,"total_tokens":5}}\n\ndata: [DONE]\n\n',
        ]
    if fail:
        return [chunks[0], b"data: not-json\n\n"]
    return chunks


@contextmanager
def mock_provider_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _seed(store: SqlAlchemyStore, provider: str, base: str) -> None:
    auth_config = {
        "openai": {"api_base": f"{base}/v1"},
        "azure": {
            "api_type": "azure",
            "api_base": base,
            "api_version": "2025-01-01",
        },
        "anthropic": {"api_base": f"{base}/v1"},
        # Python's Gemini config has a fixed base URL and is patched below;
        # Rust consumes this stored loopback value directly.
        "gemini": {"api_base": f"{base}/v1beta/models"},
    }[provider]
    secret = store.create_gateway_secret(
        secret_name=f"obvious-fake-{provider}-differential-secret",
        secret_value={"api_key": f"obvious-fake-{provider}-differential-key"},
        provider=provider,
        auth_config=auth_config,
    )
    model = store.create_gateway_model_definition(
        name=f"{provider}-differential-definition",
        secret_id=secret.secret_id,
        provider=provider,
        model_name="fixture-model",
    )
    store.create_gateway_endpoint(
        name=f"{provider}-differential-endpoint",
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            )
        ],
        usage_tracking=False,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def rust_server(db_uri: str):
    assert RUST_BINARY.exists(), f"build first: cargo build -p mlflow-server ({RUST_BINARY})"
    port = _free_port()
    env = {
        **os.environ,
        "MLFLOW_CRYPTO_KEK_PASSPHRASE": FAKE_PASSPHRASE,
        "MLFLOW_GATEWAY_TEST_FIXED_TIME": str(FIXED_TIME),
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
            db_uri,
        ],
        cwd=REPO_ROOT / "rust",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise AssertionError(process.stderr.read())
            try:
                if requests.get(f"{base}/health", timeout=0.1).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.05)
        else:
            raise AssertionError("Rust gateway did not start")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def _python_server(store: SqlAlchemyStore, mock_base: str, stack: ExitStack):
    stack.enter_context(patch("mlflow.server.gateway_api._get_store", return_value=store))
    stack.enter_context(patch("mlflow.server.gateway_api.get_request_workspace", return_value=None))
    stack.enter_context(patch("mlflow.server.gateway_api.check_budget_limit"))
    stack.enter_context(patch("mlflow.server.gateway_api.load_guardrails", return_value=[]))
    stack.enter_context(
        patch.object(
            GeminiProvider,
            "base_url",
            new_callable=PropertyMock,
            return_value=f"{mock_base}/v1beta/models",
        )
    )
    stack.enter_context(
        patch("mlflow.gateway.providers.anthropic.time.time", return_value=FIXED_TIME)
    )
    stack.enter_context(patch("mlflow.gateway.providers.gemini.time.time", return_value=FIXED_TIME))
    app = FastAPI()
    app.include_router(gateway_router)
    add_gateway_timing_middleware(app)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                # A 404 proves the ASGI socket is accepting requests.
                requests.get(f"{base}/health", timeout=0.1)
                break
            except requests.RequestException:
                time.sleep(0.05)
        else:
            raise AssertionError("Python gateway did not start")
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _selected_headers(response) -> dict[str, str]:
    return {
        name: response.headers[name]
        for name in (
            "content-type",
            "x-mlflow-gateway-duration-ms",
            "x-mlflow-gateway-overhead-duration-ms",
        )
        if name in response.headers
    }


def test_python_rust_gateway_runtime_mock_differential(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", FAKE_PASSPHRASE)
    db_uri = f"sqlite:///{tmp_path / 'gateway.db'}"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    store = SqlAlchemyStore(db_uri, artifacts.as_uri())
    with mock_provider_server() as mock_base:
        for provider in PROVIDERS:
            _seed(store, provider, mock_base)

        with ExitStack() as stack, rust_server(db_uri) as rust_base:
            with _python_server(store, mock_base, stack) as python_base:
                python = requests.Session()
                rust = requests.Session()
                stack.callback(python.close)
                stack.callback(rust.close)

                for provider in PROVIDERS:
                    path = f"/gateway/{provider}-differential-endpoint/mlflow/invocations"
                    for content, streaming in (
                        ("hello", False),
                        ("hello", True),
                        ("error-429", False),
                        ("error-500", False),
                        ("mid-stream-error", True),
                    ):
                        body = {
                            "messages": [{"role": "user", "content": content}],
                            "stream": streaming,
                        }
                        py_response = python.post(f"{python_base}{path}", json=body, timeout=10)
                        rs_response = rust.post(f"{rust_base}{path}", json=body, timeout=10)
                        assert py_response.status_code == rs_response.status_code, (provider, body)
                        assert py_response.content == rs_response.content, (
                            f"{provider} {body}\nPY={py_response.content!r}\n"
                            f"RS={rs_response.content!r}"
                        )
                        py_headers = _selected_headers(py_response)
                        rs_headers = _selected_headers(rs_response)
                        assert py_headers.keys() == rs_headers.keys(), (provider, body)
                        assert py_headers["content-type"] == rs_headers["content-type"]
                        assert int(py_headers["x-mlflow-gateway-duration-ms"]) >= 0
                        assert int(rs_headers["x-mlflow-gateway-duration-ms"]) >= 0
                        if "x-mlflow-gateway-overhead-duration-ms" in py_headers:
                            assert int(py_headers["x-mlflow-gateway-overhead-duration-ms"]) >= 0
                            assert int(rs_headers["x-mlflow-gateway-overhead-duration-ms"]) >= 0

                for streaming in (False, True):
                    path = "/gateway/mlflow/v1/chat/completions"
                    body = {
                        "model": "openai-differential-endpoint",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": streaming,
                    }
                    py_response = python.post(f"{python_base}{path}", json=body, timeout=10)
                    rs_response = rust.post(f"{rust_base}{path}", json=body, timeout=10)
                    assert py_response.status_code == rs_response.status_code
                    assert py_response.content == rs_response.content
                    assert (
                        _selected_headers(py_response).keys()
                        == _selected_headers(rs_response).keys()
                    )

                endpoint_path = "/gateway/openai-differential-endpoint/mlflow/invocations"
                validation_cases = (
                    (endpoint_path, {}),
                    (endpoint_path, {"messages": []}),
                    (
                        endpoint_path,
                        {"messages": [{"role": "user", "content": "hello"}], "n": 0},
                    ),
                    (
                        endpoint_path,
                        {
                            "model": "forbidden-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    ),
                    (
                        "/gateway/mlflow/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "hello"}]},
                    ),
                )
                for path, body in validation_cases:
                    py_response = python.post(f"{python_base}{path}", json=body, timeout=10)
                    rs_response = rust.post(f"{rust_base}{path}", json=body, timeout=10)
                    assert py_response.status_code == rs_response.status_code, (path, body)
                    assert py_response.content == rs_response.content, (
                        f"{path} {body}\nPY={py_response.content!r}\nRS={rs_response.content!r}"
                    )
