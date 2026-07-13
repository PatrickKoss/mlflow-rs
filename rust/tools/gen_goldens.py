#!/usr/bin/env python
"""Generate golden fixtures for the MLflow-compatible JSON codec (T1.3).

Run with:

    uv run --frozen python rust/tools/gen_goldens.py

For each representative message it writes a pair of files into
``rust/crates/mlflow-proto/tests/goldens/``:

* ``<name>.pb``   — the message serialized to protobuf wire bytes. The Rust test
                    decodes this and re-serializes with the Rust codec.
* ``<name>.json`` — ``message_to_json(msg)`` output, normalized so that
                    protobuf map fields have their keys sorted lexicographically.

Plus ``manifest.json`` mapping each golden name to its fully-qualified protobuf
type name (e.g. ``mlflow.Run``) so the Rust test knows which descriptor to use.

Why normalize map keys? Google's ``MessageToJson`` iterates protobuf map fields
in a per-process *randomized* hash order — the same map serializes with different
key orders across Python runs, so that output is not reproducible on either side.
The Rust codec emits map keys sorted lexicographically (the one intentional
deviation from raw Python output, documented in ``src/json.rs``); we sort the
Python goldens the same way so the byte-for-byte comparison is meaningful and
deterministic. Every non-map part of the output stays byte-identical to Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlflow.protos import model_registry_pb2, service_pb2, webhooks_pb2
from mlflow.utils.proto_json_utils import message_to_json

GOLDENS_DIR = (
    Path(__file__).resolve().parents[1]
    / "crates"
    / "mlflow-proto"
    / "tests"
    / "goldens"
)


@dataclass
class Golden:
    name: str
    type_name: str
    message: object


def _normalize(message, json_node):
    """Descriptor-aware normalization: sort keys of map fields only.

    ``json_node`` is the already-parsed JSON for ``message``. We walk the
    protobuf descriptor alongside it and, for each map field, replace its object
    with a key-sorted copy (recursing into message-valued maps and nested
    messages).
    """
    from google.protobuf.descriptor import FieldDescriptor

    if not hasattr(message, "DESCRIPTOR"):
        return json_node

    for field in message.DESCRIPTOR.fields:
        if field.name not in json_node:
            continue
        is_map = (
            field.type == FieldDescriptor.TYPE_MESSAGE
            and field.message_type.has_options
            and field.message_type.GetOptions().map_entry
        )
        if is_map:
            value_field = field.message_type.fields_by_name["value"]
            obj = json_node[field.name]
            proto_map = getattr(message, field.name)
            normalized = {}
            for key in sorted(obj.keys()):
                if value_field.type == FieldDescriptor.TYPE_MESSAGE:
                    # message-valued map: recurse. proto map keys may be int.
                    proto_key = _coerce_map_key(proto_map, key)
                    normalized[key] = _normalize(proto_map[proto_key], obj[key])
                else:
                    normalized[key] = obj[key]
            json_node[field.name] = normalized
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            sub = json_node[field.name]
            if field.is_repeated:
                proto_list = getattr(message, field.name)
                for i, item in enumerate(sub):
                    _normalize(proto_list[i], item)
            else:
                _normalize(getattr(message, field.name), sub)
    return json_node


def _coerce_map_key(proto_map, json_key: str):
    """Map JSON string keys back to the proto map's native key type."""
    for k in proto_map:
        if str(k) == json_key:
            return k
    return json_key


def _build_goldens() -> list[Golden]:
    goldens: list[Golden] = []

    # --- Run with metrics/params/tags/inputs ---------------------------------
    run = service_pb2.Run()
    run.info.run_id = "run-abc-123"
    run.info.run_uuid = "run-abc-123"
    run.info.experiment_id = "42"
    run.info.user_id = "alice"
    run.info.status = service_pb2.RunStatus.Value("FINISHED")
    run.info.start_time = 1700000000000
    run.info.end_time = 1700000005000
    run.info.artifact_uri = "s3://bucket/run-abc-123/artifacts"
    run.info.lifecycle_stage = "active"
    run.data.metrics.add(key="rmse", value=0.123, timestamp=1700000001000, step=1)
    run.data.metrics.add(
        key="big", value=1e20, timestamp=1700000002000, step=1000000000000
    )
    run.data.params.add(key="alpha", value="0.5")
    run.data.params.add(key="model_class", value="LogisticRegression")
    run.data.tags.add(key="mlflow.source.name", value="train.py")
    dataset = run.inputs.dataset_inputs.add()
    dataset.dataset.name = "training"
    dataset.dataset.digest = "abc123"
    dataset.dataset.source_type = "local"
    dataset.dataset.source = "/data/train.csv"
    dataset.tags.add(key="context", value="training")
    goldens.append(Golden("run", "mlflow.Run", run))

    # --- Run with an int64 value > 2^53 --------------------------------------
    big_run = service_pb2.Run()
    big_run.info.run_id = "big-int"
    big_run.info.start_time = 9007199254740993  # 2^53 + 1
    big_run.info.end_time = 9223372036854775807  # i64::MAX
    goldens.append(Golden("run_large_int64", "mlflow.Run", big_run))

    # --- Run with a proto2 field explicitly set to its default ---------------
    default_run = service_pb2.Run()
    default_run.info.run_id = "zero"
    default_run.info.start_time = 0  # explicitly set default -> MUST be emitted
    default_run.info.status = service_pb2.RunStatus.Value(
        "RUNNING"
    )  # non-default enum
    goldens.append(Golden("run_explicit_default", "mlflow.Run", default_run))

    # --- Unicode + special characters in string fields -----------------------
    uni = service_pb2.Run()
    uni.info.run_id = "unicode"
    uni.data.params.add(
        key="text",
        value='héllo 世界 \U0001f600 "quote" \\back / slash\n\ttab',
    )
    uni.data.params.add(key="ctrl", value="bell\x07 null\x00 esc\x1b")
    goldens.append(Golden("run_unicode", "mlflow.Run", uni))

    # --- Empty message -------------------------------------------------------
    goldens.append(Golden("run_empty", "mlflow.Run", service_pb2.Run()))

    # --- TraceInfoV3 with Timestamp/Duration/maps/enum/nested ----------------
    trace = service_pb2.TraceInfoV3()
    trace.trace_id = "tr-1"
    trace.client_request_id = "req-9"
    trace.trace_location.type = service_pb2.TraceLocation.TraceLocationType.Value(
        "MLFLOW_EXPERIMENT"
    )
    trace.trace_location.mlflow_experiment.experiment_id = "42"
    trace.request_preview = '{"q": "hi"}'
    trace.response_preview = '{"a": "yo"}'
    trace.request_time.FromMilliseconds(1700000000123)
    trace.execution_duration.FromMilliseconds(1500)
    trace.state = service_pb2.TraceInfoV3.State.Value("OK")
    # Multi-key maps to exercise deterministic sorted-key output.
    for k, v in [("zebra", "Z"), ("apple", "A"), ("mango", "M"), ("run_id", "r1")]:
        trace.trace_metadata[k] = v
    trace.tags["question_topic"] = "DBSQL"
    trace.tags["another_tag"] = "value"
    goldens.append(Golden("trace_info_v3", "mlflow.TraceInfoV3", trace))

    # --- TraceInfoV3 with empty repeated + empty maps (all omitted) ----------
    trace_min = service_pb2.TraceInfoV3()
    trace_min.trace_id = "tr-min"
    goldens.append(Golden("trace_info_v3_min", "mlflow.TraceInfoV3", trace_min))

    # --- SearchRuns request (HasField on run_view_type, defaults) ------------
    search = service_pb2.SearchRuns()
    search.experiment_ids.extend(["1", "2", "3"])
    search.filter = "metrics.rmse < 1"
    search.run_view_type = service_pb2.ViewType.Value("ACTIVE_ONLY")
    search.max_results = 500
    search.order_by.extend(["metrics.rmse ASC", "start_time DESC"])
    search.page_token = "dG9rZW4="
    goldens.append(Golden("search_runs_request", "mlflow.SearchRuns", search))

    # --- SearchRuns request with empty repeated lists ------------------------
    search_empty = service_pb2.SearchRuns()
    search_empty.max_results = 1000  # explicit default
    goldens.append(
        Golden("search_runs_request_empty", "mlflow.SearchRuns", search_empty)
    )

    # --- SearchRuns.Response with nested runs --------------------------------
    resp = service_pb2.SearchRuns.Response()
    r1 = resp.runs.add()
    r1.info.run_id = "r1"
    r1.info.start_time = 100
    r1.data.metrics.add(key="m", value=1.0, timestamp=1, step=0)
    r2 = resp.runs.add()
    r2.info.run_id = "r2"
    r2.info.start_time = 200
    resp.next_page_token = "next-123"
    goldens.append(
        Golden("search_runs_response", "mlflow.SearchRuns.Response", resp)
    )

    # --- RegisteredModel with latest_versions/tags/aliases -------------------
    rm = model_registry_pb2.RegisteredModel()
    rm.name = "my-model"
    rm.creation_timestamp = 1700000000000
    rm.last_updated_timestamp = 1700000009000
    rm.description = "A model"
    lv = rm.latest_versions.add()
    lv.name = "my-model"
    lv.version = "3"
    lv.creation_timestamp = 1700000000000
    lv.current_stage = "Production"
    lv.status = model_registry_pb2.ModelVersionStatus.Value("READY")
    rm.tags.add(key="team", value="ml")
    rm.tags.add(key="env", value="prod")
    alias = rm.aliases.add()
    alias.alias = "champion"
    alias.version = "3"
    goldens.append(
        Golden("registered_model", "mlflow.RegisteredModel", rm)
    )

    # --- ModelVersion full -------------------------------------------------
    mv = model_registry_pb2.ModelVersion()
    mv.name = "my-model"
    mv.version = "3"
    mv.creation_timestamp = 1700000000000
    mv.last_updated_timestamp = 1700000009000
    mv.user_id = "bob"
    mv.current_stage = "Production"
    mv.description = "v3"
    mv.source = "models:/my-model/2"
    mv.run_id = "run-abc-123"
    mv.status = model_registry_pb2.ModelVersionStatus.Value("READY")
    mv.tags.add(key="quality", value="high")
    mv.aliases.extend(["champion", "prod"])
    mv.model_id = "m-1"
    goldens.append(Golden("model_version", "mlflow.ModelVersion", mv))

    # --- Webhook with events + enums + int64 timestamps ----------------------
    wh = webhooks_pb2.Webhook()
    wh.webhook_id = "wh-1"
    wh.name = "notify"
    wh.description = "on model version create"
    wh.url = "https://example.com/hook"
    ev = wh.events.add()
    ev.entity = webhooks_pb2.WebhookEntity.Value("REGISTERED_MODEL")
    ev.action = webhooks_pb2.WebhookAction.Value("CREATED")
    wh.status = webhooks_pb2.WebhookStatus.Value("ACTIVE")
    wh.creation_timestamp = 1700000000000
    wh.last_updated_timestamp = 1700000009000
    goldens.append(Golden("webhook", "mlflow.Webhook", wh))

    return goldens


def main() -> None:
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    # Clean stale goldens so a rename/removal never leaves orphans behind.
    for existing in GOLDENS_DIR.glob("*"):
        existing.unlink()

    manifest: dict[str, str] = {}
    for g in _build_goldens():
        pb_bytes = g.message.SerializeToString(deterministic=True)
        (GOLDENS_DIR / f"{g.name}.pb").write_bytes(pb_bytes)

        parsed = json.loads(message_to_json(g.message))
        _normalize(g.message, parsed)
        # Re-serialize exactly like message_to_json's final json.dumps(indent=2)
        # but with our normalized (map-key-sorted) dict. sort_keys stays False so
        # regular field-number order (from MessageToJson) is preserved.
        golden_json = json.dumps(parsed, indent=2)
        (GOLDENS_DIR / f"{g.name}.json").write_text(golden_json, encoding="utf-8")

        manifest[g.name] = g.type_name

    manifest_path = GOLDENS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(manifest)} goldens to {GOLDENS_DIR}")


if __name__ == "__main__":
    main()
