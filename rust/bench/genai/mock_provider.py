"""Hermetic, seeded OpenAI/Anthropic-compatible benchmark provider."""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator


def canonical_request(body: bytes) -> bytes:
    """Canonicalize JSON so harmless client formatting does not alter a response."""
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _schema_value(schema: Any, digest: str, root: dict[str, Any], key: str = "") -> Any:
    """Build a small deterministic instance of an OpenAI response JSON schema."""
    if not isinstance(schema, dict):
        return f"bench-{digest[:12]}"
    if ref := schema.get("$ref"):
        target: Any = root
        for part in ref.removeprefix("#/").split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        return _schema_value(target, digest, root, key)
    if values := schema.get("enum"):
        if key == "severity" and "low" in values:
            return "low"
        return values[0]
    for variant_key in ("const", "default"):
        if variant_key in schema:
            return schema[variant_key]
    for union_key in ("anyOf", "oneOf"):
        if variants := schema.get(union_key):
            variant = next(
                (
                    item
                    for item in variants
                    if isinstance(item, dict) and item.get("type") != "null"
                ),
                variants[0],
            )
            return _schema_value(variant, digest, root, key)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), "string")
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {
            name: _schema_value(properties[name], digest, root, name)
            for name in required
            if name in properties
        }
    if kind == "array":
        return [_schema_value(schema.get("items", {}), digest, root, key)]
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.5
    if kind == "boolean":
        return True
    if key in {"category", "categories"}:
        return "quality"
    if key in {"result", "value"}:
        return "yes"
    return f"bench-{key or 'value'}-{digest[:12]}"


@dataclass
class ProviderObservation:
    route: str
    request_sha256: str
    response_sha256: str
    response_bytes: int


@dataclass
class ProviderState:
    seed: int
    route_latency_ms: dict[str, float] = field(default_factory=dict)
    frame_gap_ms: float = 0.0
    observations: list[ProviderObservation] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def digest(self, route: str, request: bytes) -> str:
        canonical = canonical_request(request)
        try:
            body = json.loads(canonical)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        assistant_turn = (
            re.search(r"assistant seed \d+ turn \d+", json.dumps(body, sort_keys=True))
            if isinstance(body, dict)
            else None
        )
        # OpenAI passthrough clients add different transport-only fields. The
        # benchmark's explicit ``user`` token identifies the logical request,
        # so response bytes remain stable while observations still hash the
        # full upstream request independently.
        if assistant_turn:
            canonical = assistant_turn.group(0).encode()
        elif isinstance(body, dict) and body.get("user"):
            canonical = _json_bytes({
                "max_tokens": body.get("max_tokens"),
                "route": route,
                "stream": body.get("stream", False),
                "user": body["user"],
            })
        material = f"{self.seed}:{route}:".encode() + canonical
        return hashlib.sha256(material).hexdigest()

    def observe(self, route: str, request: bytes, response: bytes) -> None:
        with self.lock:
            self.observations.append(
                ProviderObservation(
                    route=route,
                    request_sha256=hashlib.sha256(canonical_request(request)).hexdigest(),
                    response_sha256=hashlib.sha256(response).hexdigest(),
                    response_bytes=len(response),
                )
            )


class DeterministicProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        seed: int,
        route_latency_ms: dict[str, float] | None = None,
        frame_gap_ms: float = 0.0,
    ) -> None:
        self.state = ProviderState(seed, route_latency_ms or {}, frame_gap_ms)
        super().__init__(address, ProviderHandler)


class ProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def provider(self) -> DeterministicProvider:
        return self.server  # type: ignore[return-value]

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(200, {"data": [{"id": "genai-bench-model", "object": "model"}]})
        else:
            self._send_json(404, {"error": {"message": "mock provider route not found"}})

    def do_POST(self) -> None:
        size = int(self.headers.get("content-length", "0"))
        request = self.rfile.read(size)
        try:
            body = json.loads(request)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": {"message": "invalid JSON"}})
            return

        route = self._route()
        if route is None:
            self._send_json(404, {"error": {"message": "mock provider route not found"}})
            return
        latency = self.provider.state.route_latency_ms.get(route, 0.0)
        if latency > 0:
            time.sleep(latency / 1000)
        digest = self.provider.state.digest(route, request)
        if route == "embeddings":
            response = self._embedding_response(body, digest)
        elif route == "anthropic_messages":
            response = self._anthropic_response(body, digest)
        else:
            response = self._openai_response(body, digest)
        self.provider.state.observe(route, request, response)
        if body.get("stream"):
            self._send_stream(response)
        else:
            self._send_bytes(200, response, "application/json")

    def _route(self) -> str | None:
        path = self.path.split("?", 1)[0]
        if path.endswith("/embeddings"):
            return "embeddings"
        if path.endswith("/messages"):
            return "anthropic_messages"
        if path.endswith("/chat/completions"):
            return "chat_completions"
        return None

    def _openai_response(self, body: dict[str, Any], digest: str) -> bytes:
        model = str(body.get("model") or "genai-bench-model")
        if body.get("response_format"):
            response_format = body["response_format"]
            schema = response_format.get("json_schema", {}).get("schema", response_format)
            value = _schema_value(schema, digest, schema)
            text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            text = f"bench-{digest[:12]} {digest[12:20]}"
        prompt_tokens = 5 + int(digest[20:22], 16) % 19
        completion_tokens = 2 + int(digest[22:24], 16) % 7
        usage = {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if body.get("stream"):
            frame_count = self._openai_frame_count(body, digest)
            if frame_count is None:
                pieces = [text[:10], text[10:20], text[20:]]
            else:
                pieces = [
                    f"bench-{index:04d}-{digest[index % len(digest)]}"
                    for index in range(frame_count)
                ]
            frames = []
            for index, piece in enumerate(pieces):
                chunk: dict[str, Any] = {
                    "choices": [
                        {
                            "delta": {
                                "content": piece,
                                **({"role": "assistant"} if index == 0 else {}),
                            },
                            "finish_reason": "stop" if index == len(pieces) - 1 else None,
                            "index": 0,
                        }
                    ],
                    "created": int(digest[:8], 16),
                    "id": f"chatcmpl-{digest[:20]}",
                    "model": model,
                    "object": "chat.completion.chunk",
                }
                if index == len(pieces) - 1:
                    chunk["usage"] = usage
                frames.append(b"data: " + _json_bytes(chunk) + b"\n\n")
            frames.append(b"data: [DONE]\n\n")
            return b"".join(frames)
        return _json_bytes({
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": text, "role": "assistant"},
                }
            ],
            "created": int(digest[:8], 16),
            "id": f"chatcmpl-{digest[:20]}",
            "model": model,
            "object": "chat.completion",
            "usage": usage,
        })

    @staticmethod
    def _openai_frame_count(body: dict[str, Any], digest: str) -> int | None:
        """Return T23.4's deterministic small/large stream cardinality.

        Ordinary harness requests retain the original three-frame fixture. The
        benchmark selects a valid OpenAI ``max_tokens`` value, and cardinality
        then varies only with the seeded request digest.
        """
        max_tokens = body.get("max_tokens")
        if max_tokens == 32:
            return 9 + int(digest[24:26], 16) % 3
        if max_tokens == 512:
            return 112 + int(digest[24:26], 16) % 17
        return None

    def _embedding_response(self, body: dict[str, Any], digest: str) -> bytes:
        inputs = body.get("input", [])
        if not isinstance(inputs, list):
            inputs = [inputs]
        data = []
        for index, _value in enumerate(inputs):
            item_digest = hashlib.sha256(f"{digest}:{index}".encode()).digest()
            embedding = [round((byte - 127.5) / 127.5, 8) for byte in item_digest[:12]]
            data.append({"embedding": embedding, "index": index, "object": "embedding"})
        prompt_tokens = sum(max(1, len(str(value).split())) for value in inputs)
        return _json_bytes({
            "data": data,
            "model": str(body.get("model") or "genai-bench-embedding"),
            "object": "list",
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        })

    def _anthropic_response(self, body: dict[str, Any], digest: str) -> bytes:
        model = str(body.get("model") or "genai-bench-model")
        text = f"bench-{digest[:12]} {digest[12:20]}"
        input_tokens = 5 + int(digest[20:22], 16) % 19
        output_tokens = 2 + int(digest[22:24], 16) % 7
        if body.get("stream"):
            events = [
                (
                    "message_start",
                    {
                        "message": {
                            "content": [],
                            "id": f"msg-{digest[:20]}",
                            "model": model,
                            "role": "assistant",
                            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                        },
                        "type": "message_start",
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "delta": {"text": text, "type": "text_delta"},
                        "index": 0,
                        "type": "content_block_delta",
                    },
                ),
                (
                    "message_delta",
                    {
                        "delta": {"stop_reason": "end_turn"},
                        "type": "message_delta",
                        "usage": {"output_tokens": output_tokens},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            return b"".join(
                f"event: {event}\n".encode() + b"data: " + _json_bytes(value) + b"\n\n"
                for event, value in events
            )
        return _json_bytes({
            "content": [{"text": text, "type": "text"}],
            "id": f"msg-{digest[:20]}",
            "model": model,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        })

    def _send_json(self, status: int, value: Any) -> None:
        self._send_bytes(status, _json_bytes(value), "application/json")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_stream(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        frames = body.split(b"\n\n")
        for index, frame in enumerate(frames):
            if not frame:
                continue
            self.wfile.write(frame + b"\n\n")
            self.wfile.flush()
            if self.provider.state.frame_gap_ms > 0 and index + 1 < len(frames):
                time.sleep(self.provider.state.frame_gap_ms / 1000)
        self.close_connection = True


@contextlib.contextmanager
def provider_server(
    seed: int,
    route_latency_ms: dict[str, float] | None = None,
    frame_gap_ms: float = 0.0,
) -> Iterator[DeterministicProvider]:
    server = DeterministicProvider(("127.0.0.1", 0), seed, route_latency_ms, frame_gap_ms)
    thread = threading.Thread(target=server.serve_forever, name="genai-mock-provider", daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def deterministic_payloads(seed: int, count: int) -> list[dict[str, Any]]:
    """Small public fixture helper used by tests and future scenario cells."""
    rng = random.Random(seed)
    return [
        {
            "messages": [{"content": f"prompt-{index}-{rng.randrange(1_000_000)}", "role": "user"}],
            "model": "genai-bench-model",
            "stream": bool(index % 2),
        }
        for index in range(count)
    ]
