#!/usr/bin/env python3
"""Verify that Python reads and byte-round-trips a Rust-written traces.pb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.tracing.otel.otel_archival import spans_to_traces_data_pb, traces_data_pb_to_spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rust_payload", type=Path)
    parser.add_argument("python_golden", type=Path)
    args = parser.parse_args()

    rust_bytes = args.rust_payload.read_bytes()
    python_bytes = args.python_golden.read_bytes()
    assert rust_bytes == python_bytes

    spans = traces_data_pb_to_spans(rust_bytes)
    assert [span.name for span in spans] == ["root", "child-a", "child-b"]
    root = spans[0]
    assert root.status.status_code.value == "ERROR"
    assert root.status.description == "root failed"
    assert root.attributes["object"] == {"z": 1, "a": "two"}
    assert root.events[0].attributes["labels"] == ["red", "blue"]
    assert root.links[0].trace_id == "tr-ffeeddccbbaa99887766554433221100"
    assert root.links[0].attributes == {
        "relationship": "causal",
        "weight": 7,
        "active": True,
    }
    assert root._span.resource.attributes == {}
    assert spans_to_traces_data_pb(spans) == rust_bytes

    print(  # noqa: T201 - machine-readable result consumed by the Rust test
        json.dumps(
            {
                "byte_equal_to_python": True,
                "python_read_rust": True,
                "python_round_trip_byte_equal": True,
                "span_count": len(spans),
                "span_order": [span.name for span in spans],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
