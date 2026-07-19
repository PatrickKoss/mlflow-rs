#!/usr/bin/env python3
"""Generate pinned-Python golden fixtures for the Rust trace archival codec."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.trace import Link as OTelLink
from opentelemetry.trace import Status, StatusCode

from mlflow.entities.span import Span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.otel_archival import spans_to_traces_data_pb
from mlflow.tracing.utils import build_otel_context

TRACE_ID = int("00112233445566778899aabbccddeeff", 16)
TRACE_ID_TEXT = "tr-00112233445566778899aabbccddeeff"
ROOT_ID = int("1020304050607080", 16)
CHILD_A_ID = int("1020304050607010", 16)
CHILD_B_ID = int("1020304050607020", 16)
LINK_TRACE_ID = int("ffeeddccbbaa99887766554433221100", 16)
LINK_SPAN_ID = int("8877665544332211", 16)


def _attributes(span_type: str):
    # Span's persisted representation stores each top-level attribute as a JSON string.
    values = {
        SpanAttributeKey.REQUEST_ID: TRACE_ID_TEXT,
        SpanAttributeKey.SPAN_TYPE: span_type,
        "text": "héllo",
        "integer": 42,
        "double": 1.25,
        "boolean": True,
        "array": ["first", 2, False],
        "object": {"z": 1, "a": "two"},
        "empty_array": [],
        "empty_object": {},
        "nothing": None,
    }
    return {key: json.dumps(value, ensure_ascii=False) for key, value in values.items()}


def _span(
    span_id: int,
    name: str,
    start: int,
    end: int,
    status: Status,
    *,
    parent_id: int | None = None,
    resource: Resource,
    with_details: bool = False,
) -> Span:
    readable = ReadableSpan(
        name=name,
        context=build_otel_context(TRACE_ID, span_id),
        parent=build_otel_context(TRACE_ID, parent_id) if parent_id else None,
        start_time=start,
        end_time=end,
        attributes=_attributes("CHAIN" if parent_id is None else "TOOL"),
        events=(
            [
                Event(
                    name="exception",
                    timestamp=start + 50,
                    attributes={
                        "message": "boom",
                        "attempt": 3,
                        "retryable": False,
                        "ratio": 0.5,
                        "labels": ["red", "blue"],
                    },
                )
            ]
            if with_details
            else []
        ),
        links=(
            [
                OTelLink(
                    context=build_otel_context(LINK_TRACE_ID, LINK_SPAN_ID),
                    attributes={"relationship": "causal", "weight": 7, "active": True},
                )
            ]
            if with_details
            else []
        ),
        status=status,
        resource=resource,
    )
    return Span(readable)


def build_spans() -> list[Span]:
    resource = Resource({
        "service.name": "archive-fixture",
        "service.version": "1.2.3",
        "replica": 2,
        "enabled": True,
        "ratios": (0.25, 0.75),
    })
    child_b = _span(
        CHILD_B_ID,
        "child-b",
        1_000,
        1_700,
        Status(StatusCode.UNSET),
        parent_id=ROOT_ID,
        resource=resource,
    )
    root = _span(
        ROOT_ID,
        "root",
        2_000,
        4_000,
        Status(StatusCode.ERROR, "root failed"),
        resource=resource,
        with_details=True,
    )
    child_a = _span(
        CHILD_A_ID,
        "child-a",
        1_000,
        1_500,
        Status(StatusCode.OK),
        parent_id=ROOT_ID,
        resource=resource,
    )
    # Deliberately noncanonical input: writer must emit root, child-a, child-b.
    return [child_b, root, child_a]


def _stored_span(span: Span) -> dict:
    content = json.dumps(span.to_dict())
    return {
        "trace_id": TRACE_ID_TEXT,
        "experiment_id": 7,
        "span_id": span.span_id,
        "parent_span_id": span.parent_id,
        "name": span.name,
        "span_type": span.span_type,
        "status": span.status.status_code.value,
        "start_time_unix_nano": span.start_time_ns,
        "end_time_unix_nano": span.end_time_ns,
        "duration_ns": span.end_time_ns - span.start_time_ns,
        "content": content,
        "dimension_attributes": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    spans = build_spans()
    (args.output / "python_resource_traces.pb").write_bytes(spans_to_traces_data_pb(spans))

    # SQL archival reconstructs Span entities from JSON, which intentionally has no resource.
    db_spans = [Span.from_dict(span.to_dict()) for span in spans]
    (args.output / "python_db_traces.pb").write_bytes(spans_to_traces_data_pb(db_spans))

    manifest = {
        "generator": {
            "command": (
                "uv run --frozen python rust/tools/gen_trace_archival_fixtures.py "
                "rust/crates/mlflow-server/tests/fixtures/trace_archival"
            ),
            "python": platform.python_version(),
            "protobuf": importlib.metadata.version("protobuf"),
            "opentelemetry-proto": importlib.metadata.version("opentelemetry-proto"),
            "opentelemetry-sdk": importlib.metadata.version("opentelemetry-sdk"),
        },
        "expected_order": ["root", "child-a", "child-b"],
        "resource_attributes": list(spans[0]._span.resource.attributes.items()),
        "stored_spans": [_stored_span(span) for span in spans],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
