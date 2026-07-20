from __future__ import annotations

import json
import time

import requests

from rust.bench.genai.equivalence import compare_runs, normalize
from rust.bench.genai.mock_provider import deterministic_payloads, provider_server
from rust.bench.genai.t23_2 import (
    FAMILIES,
    Cell,
    cell_matrix,
    fixed_id,
    sample_sequences,
    seeded_stream,
)
from rust.bench.genai.t23_3 import (
    EVALUATE,
    ISSUES,
    JOB_KINDS,
    OPTIMIZE,
    SCORER,
    _ordered_direct_specs,
)
from rust.bench.genai.t23_3 import cell_matrix as job_cell_matrix
from rust.bench.genai.t23_4 import cell_matrix as streaming_cell_matrix


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
    assert normalize({"records": '[{"id":"left","optional":null}]'}) == {
        "records": '[{"id":"<id>"}]'
    }
    assert normalize("obvious-fake-model-208") == "<concurrent-model-state>"
    assert normalize("http://127.0.0.1:54321/v1") == "http://127.0.0.1:<port>/v1"


def _matrix_setup() -> dict:
    experiments = {
        family: {"read": f"read-{family}", "write": f"write-{family}"} for family in FAMILIES
    }
    return {
        "datasets": {"read_records": fixed_id("d-", "read"), "write": fixed_id("d-", "write")},
        "experiments": experiments,
        "gateway": {
            "read": {
                "endpoint_id": fixed_id("ep-", "read"),
                "model_definition_id": fixed_id("md-", "read"),
                "secret_id": fixed_id("s-", "read"),
            },
            "write": {
                "endpoint_id": fixed_id("ep-", "write"),
                "model_definition_id": fixed_id("md-", "write"),
                "secret_id": fixed_id("s-", "write"),
            },
            "read_guardrail_id": fixed_id("gr-", "read"),
            "scorer_id": fixed_id("sc-", "guardrail"),
            "scorer_version": 1,
        },
        "review_schema_ids": [fixed_id("ls-", index) for index in range(100)],
        "review_trace_ids": [fixed_id("tr-", index) for index in range(100)],
        "seed": 2320,
    }


def test_t23_2_fractional_matrix_covers_axes_and_canonical_counts() -> None:
    cells = cell_matrix(10_000, 1_000)
    assert len(cells) == 4
    assert {cell.payload_size for cell in cells} == {"small", "large"}
    assert {cell.concurrency for cell in cells} == {1, 16, 128}
    assert {cell.mix for cell in cells} == {"write-heavy", "read-heavy"}
    assert all(
        cell.requests == (10_000 if cell.payload_size == "small" else 1_000) for cell in cells
    )


def test_t23_2_stream_is_repeatable_and_has_exact_mix() -> None:
    setup = _matrix_setup()
    cell = Cell(0, "small", 1, "write-heavy", 100, 10_000)
    first = seeded_stream("issues", cell, setup, 2320)
    second = seeded_stream("issues", cell, setup, 2320)
    assert first == second
    assert sum("search" not in spec.endpoint for spec in first) == 90
    assert len(sample_sequences(first, 2320)) >= 16


def test_t23_2_dataset_large_batch_is_in_required_range() -> None:
    setup = _matrix_setup()
    cell = Cell(2, "large", 16, "write-heavy", 100, 1_000)
    specs = seeded_stream("datasets", cell, setup, 2320)
    upsert = next(spec for spec in specs if spec.endpoint == "dataset_records_upsert")
    body_bytes = len(json.dumps(upsert.json_body, separators=(",", ":")).encode())
    assert 256 * 1024 <= body_bytes <= 1024 * 1024


def test_t23_2_mutable_unique_names_do_not_collide_between_cells() -> None:
    setup = _matrix_setup()
    cells = cell_matrix(1_000, 100)
    for family, endpoint in (
        ("label_schemas", "label_schemas_update"),
        ("review_queues", "review_queues_update"),
    ):
        names_by_cell = [
            {
                spec.json_body["name"]
                for spec in seeded_stream(family, cell, setup, 2320)
                if spec.endpoint == endpoint
            }
            for cell in cells
        ]
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(names_by_cell)
            for right in names_by_cell[index + 1 :]
        )


def test_t23_3_fractional_matrix_covers_every_kind_and_shape() -> None:
    cells = job_cell_matrix(1_000, 10, 100, 20, 1_000)
    assert len(cells) == 12
    for kind in JOB_KINDS:
        kind_cells = [cell for cell in cells if kind in cell.kinds]
        assert {"high-fanout", "large-payload"} <= {cell.shape for cell in kind_cells}
        assert any(cell.shape == "burst" for cell in kind_cells)
        assert any(cell.shape == "steady-drip" for cell in kind_cells)
    assert all(
        cell.jobs_by_kind[kind] == 1_000
        for cell in cells
        if cell.shape == "high-fanout"
        for kind in cell.kinds
    )


def test_t23_3_steady_drip_interleaves_direct_kinds_by_schedule() -> None:
    cell = job_cell_matrix(1_000, 10, 100, 20, 1_000)[-1]
    specs = [
        (OPTIMIZE, 0, 2.4),
        (EVALUATE, 1, 3.2),
        (ISSUES, 0, 1.6),
        (SCORER, 0, 0.8),
        (EVALUATE, 0, 0.0),
    ]
    assert _ordered_direct_specs(cell, specs) == [
        (EVALUATE, 0, 0.0),
        (SCORER, 0, 0.8),
        (ISSUES, 0, 1.6),
        (OPTIMIZE, 0, 2.4),
        (EVALUATE, 1, 3.2),
    ]


def test_mock_provider_generates_nested_json_schema_instances() -> None:
    body = {
        "messages": [{"content": "cluster", "role": "user"}],
        "model": "fixture",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ClusterResponse",
                "schema": {
                    "type": "object",
                    "properties": {
                        "groups": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "indices": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["name", "indices"],
                            },
                        }
                    },
                    "required": ["groups"],
                },
            },
        },
    }
    with provider_server(2330) as server:
        response = requests.post(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            json=body,
            timeout=5,
        ).json()
    content = json.loads(response["choices"][0]["message"]["content"])
    assert content["groups"][0]["indices"] == [0]


def test_t23_4_fractional_matrix_covers_families_stream_concurrency_and_archival() -> None:
    cells = streaming_cell_matrix()
    assert {cell.family for cell in cells} == {
        "archival",
        "assistant",
        "gateway",
        "promptlab",
    }
    assert all(
        4 <= sum(cell.family == family for cell in cells) <= 6
        for family in {
            "archival",
            "assistant",
            "gateway",
            "promptlab",
        }
    )
    streaming = [cell for cell in cells if cell.kind.startswith(("stream", "assistant"))]
    assert {cell.concurrency for cell in streaming} >= {1, 16, 64}
    assert {cell.stream_variant for cell in cells if cell.family == "gateway"} >= {
        "small",
        "large",
    }
    assert sum(cell.count for cell in cells if cell.kind == "archive-pass") >= 1_000


def test_t23_4_mock_provider_has_seeded_small_and_large_frame_variants() -> None:
    counts = {}
    with provider_server(2340) as server:
        base = f"http://127.0.0.1:{server.server_port}"
        for variant, max_tokens in (("small", 32), ("large", 512)):
            body = {
                "max_tokens": max_tokens,
                "messages": [{"content": f"{variant} stream", "role": "user"}],
                "model": "fixture",
                "stream": True,
            }
            first = requests.post(base + "/v1/chat/completions", json=body, timeout=5).text
            second = requests.post(base + "/v1/chat/completions", json=body, timeout=5).text
            assert first == second
            counts[variant] = first.count("data: {")
    assert 9 <= counts["small"] <= 11
    assert 112 <= counts["large"] <= 128
