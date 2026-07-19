from __future__ import annotations

import json
import time

import requests

from rust.bench.genai.equivalence import compare_runs, normalize
from rust.bench.genai.mock_provider import deterministic_payloads, provider_server


def test_mock_provider_is_byte_stable_for_every_protocol_route() -> None:
    cases = [
        (
            "/v1/chat/completions",
            {"messages": [{"content": "hello", "role": "user"}], "model": "fixture"},
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [{"content": "hello", "role": "user"}],
                "model": "fixture",
                "stream": True,
            },
        ),
        ("/v1/embeddings", {"input": ["alpha", "beta"], "model": "fixture"}),
        (
            "/v1/messages",
            {"messages": [{"content": "hello", "role": "user"}], "model": "fixture"},
        ),
        (
            "/v1/messages",
            {
                "messages": [{"content": "hello", "role": "user"}],
                "model": "fixture",
                "stream": True,
            },
        ),
    ]
    runs = []
    for _ in range(2):
        with provider_server(2301) as server, requests.Session() as session:
            base = f"http://127.0.0.1:{server.server_port}"
            responses = []
            for path, body in cases:
                first = session.post(base + path, json=body, timeout=5)
                second = session.post(base + path, json=body, timeout=5)
                assert first.status_code == second.status_code == 200
                assert first.content == second.content
                responses.append(first.content)
            runs.append(responses)
    assert runs[0] == runs[1]


def test_mock_provider_seed_and_request_change_bytes() -> None:
    payload = {"messages": [{"content": "hello", "role": "user"}]}
    values = []
    for seed in (1, 2):
        with provider_server(seed) as server:
            base = f"http://127.0.0.1:{server.server_port}"
            values.append(
                requests.post(base + "/v1/chat/completions", json=payload, timeout=5).content
            )
    assert values[0] != values[1]


def test_mock_provider_fixed_route_latency() -> None:
    with provider_server(1, {"embeddings": 20}) as server:
        started = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{server.server_port}/v1/embeddings",
            json={"input": "hello"},
            timeout=5,
        )
        assert response.status_code == 200
        assert time.perf_counter() - started >= 0.018


def test_seeded_payloads_repeat_exactly() -> None:
    assert deterministic_payloads(42, 10) == deterministic_payloads(42, 10)
    assert deterministic_payloads(42, 10) != deterministic_payloads(43, 10)


def test_equivalence_normalizes_ids_times_and_sse_json() -> None:
    left = {
        "equivalence": {
            "jobs": [{"job_id": "left", "status": "SUCCEEDED", "duration_ms": 2}],
            "samples": [
                {
                    "response": {"run_id": "left", "timestamp_ms": 1},
                    "sse_frames": ['data: {"session_id":"00000000-0000-4000-8000-000000000001"}'],
                }
            ],
        }
    }
    right = {
        "equivalence": {
            "jobs": [{"job_id": "right", "status": "SUCCEEDED", "duration_ms": 9}],
            "samples": [
                {
                    "response": {"run_id": "right", "timestamp_ms": 8},
                    "sse_frames": ['data: {"session_id":"00000000-0000-4000-8000-000000000002"}'],
                }
            ],
        }
    }
    assert compare_runs(left, right) == []
    right["equivalence"]["jobs"][0]["status"] = "FAILED"
    assert compare_runs(left, right)
    assert normalize(json.loads('{"trace_ids":["tr-aaaaaaaaaaaaaaaa"]}')) == {"trace_ids": ["<id>"]}
