#!/usr/bin/env python3

import argparse
import base64
import json
import os
import random
import threading
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import boto3

from mlflow.entities.span import Span
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey, SpansLocation, TraceMetadataKey, TraceTagKey
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class SeedMode:
    name: str
    experiment_name: str
    artifact_location: str
    object_prefix: str


_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed artifact-backed MLflow traces through proxy and direct S3 upload modes."
    )
    parser.add_argument("--tracking-uri", default="http://localhost")
    parser.add_argument("--s3-endpoint", default="http://localhost:9000")
    parser.add_argument("--bucket", default="mlflow")
    parser.add_argument("--access-key", default="minioadmin")
    parser.add_argument("--secret-key", default="minioadmin")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--total", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("both", "proxy", "direct"),
        default="both",
        help="Upload blobs through MLflow, directly to S3, or exercise both paths.",
    )
    parser.add_argument("--experiment-prefix", default="Rust MinIO Traces")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.s3_endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = args.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = args.secret_key
    os.environ["AWS_REGION"] = args.region
    os.environ["AWS_DEFAULT_REGION"] = args.region
    os.environ["AWS_S3_ADDRESSING_STYLE"] = "path"


def get_or_create_experiment(client: MlflowClient, mode: SeedMode) -> str:
    if experiment := client.get_experiment_by_name(mode.experiment_name):
        if experiment.artifact_location.rstrip("/") != mode.artifact_location.rstrip("/"):
            raise RuntimeError(
                f"Experiment {mode.experiment_name!r} already uses "
                f"{experiment.artifact_location!r}, expected {mode.artifact_location!r}"
            )
        return experiment.experiment_id
    return client.create_experiment(mode.experiment_name, mode.artifact_location)


def encoded_id(value: int, length: int) -> str:
    return base64.b64encode(value.to_bytes(length, byteorder="big")).decode()


def make_span(
    *,
    trace_id: str,
    otel_trace_id: int,
    span_id: int,
    parent_span_id: int | None,
    name: str,
    span_type: str,
    start_ns: int,
    duration_ns: int,
    attributes: dict[str, object],
) -> Span:
    serialized_attributes = {
        SpanAttributeKey.REQUEST_ID: json.dumps(trace_id),
        SpanAttributeKey.SPAN_TYPE: json.dumps(span_type),
        **{key: json.dumps(value) for key, value in attributes.items()},
    }
    return Span.from_dict({
        "trace_id": encoded_id(otel_trace_id, 16),
        "span_id": encoded_id(span_id, 8),
        "parent_span_id": encoded_id(parent_span_id, 8) if parent_span_id else "",
        "name": name,
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": start_ns + duration_ns,
        "events": [],
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": serialized_attributes,
        "links": [],
    })


def make_trace(experiment_id: str, mode_name: str, index: int) -> tuple[TraceInfo, TraceData]:
    rng = random.Random(index * 7919 + (0 if mode_name == "proxy" else 1))
    trace_id = f"tr-{uuid.uuid4().hex}"
    request = {
        "messages": [
            {"role": "system", "content": "Answer using the indexed MinIO knowledge base."},
            {"role": "user", "content": f"Summarize object-storage example {index}."},
        ]
    }
    response = {
        "answer": f"Example {index} was uploaded using the {mode_name} artifact path.",
        "citations": [f"s3://seed-documents/example-{index % 97}.txt"],
    }
    input_tokens = rng.randint(80, 900)
    output_tokens = rng.randint(25, 320)
    total_tokens = input_tokens + output_tokens
    duration_ms = rng.randint(80, 4200)
    request_time = int(time.time() * 1000) - rng.randint(0, 7 * 24 * 60 * 60 * 1000)
    start_ns = request_time * 1_000_000
    otel_trace_id = uuid.uuid4().int

    root = make_span(
        trace_id=trace_id,
        otel_trace_id=otel_trace_id,
        span_id=1,
        parent_span_id=None,
        name="minio_rag_agent",
        span_type="CHAIN",
        start_ns=start_ns,
        duration_ns=duration_ms * 1_000_000,
        attributes={
            SpanAttributeKey.INPUTS: request,
            SpanAttributeKey.OUTPUTS: response,
        },
    )
    llm = make_span(
        trace_id=trace_id,
        otel_trace_id=otel_trace_id,
        span_id=2,
        parent_span_id=1,
        name="generate_answer",
        span_type="LLM",
        start_ns=start_ns + 20_000_000,
        duration_ns=max(1, duration_ms - 40) * 1_000_000,
        attributes={
            SpanAttributeKey.INPUTS: request["messages"],
            SpanAttributeKey.OUTPUTS: response["answer"],
            SpanAttributeKey.MODEL: "fake-minio-model",
            SpanAttributeKey.MODEL_PROVIDER: "seed",
            SpanAttributeKey.CHAT_USAGE: {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        },
    )
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=request_time,
        state=TraceState.OK,
        request_preview=json.dumps(request),
        response_preview=json.dumps(response),
        execution_duration=duration_ms,
        trace_metadata={
            TraceMetadataKey.INPUTS: json.dumps(request),
            TraceMetadataKey.OUTPUTS: json.dumps(response),
            TraceMetadataKey.TOKEN_USAGE: json.dumps({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }),
            TraceMetadataKey.TRACE_SESSION: f"minio-session-{index % 250}",
            TraceMetadataKey.TRACE_USER: f"seed-user-{index % 20}",
        },
        tags={
            TraceTagKey.TRACE_NAME: f"minio_{mode_name}_{index:05d}",
            TraceTagKey.SPANS_LOCATION: SpansLocation.ARTIFACT_REPO.value,
            "seed.upload_mode": mode_name,
            "seed.dataset": "minio-ui-validation",
            "environment": rng.choice(("dev", "staging", "production")),
        },
    )
    return trace_info, TraceData(spans=[root, llm])


def tracing_client(tracking_uri: str) -> TracingClient:
    if not hasattr(_thread_local, "clients"):
        _thread_local.clients = {}
    if tracking_uri not in _thread_local.clients:
        _thread_local.clients[tracking_uri] = TracingClient(tracking_uri)
    return _thread_local.clients[tracking_uri]


def upload_one(
    tracking_uri: str, experiment_id: str, mode_name: str, index: int
) -> tuple[str, str]:
    client = tracing_client(tracking_uri)
    trace_info, trace_data = make_trace(experiment_id, mode_name, index)
    returned_info = client.start_trace(trace_info)
    client._upload_trace_data(returned_info, trace_data)
    return returned_info.trace_id, returned_info.tags["mlflow.artifactLocation"]


def count_objects(s3_client, bucket: str, prefix: str) -> int:
    paginator = s3_client.get_paginator("list_objects_v2")
    return sum(
        page.get("KeyCount", 0)
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    )


def verify_trace_view(tracking_uri: str, trace_id: str) -> int:
    query = urllib.parse.urlencode({"request_id": trace_id})
    url = f"{tracking_uri.rstrip('/')}/ajax-api/3.0/mlflow/get-trace-artifact?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        trace_data = json.load(response)
    return len(trace_data["spans"])


def seed_mode(
    args: argparse.Namespace, mode: SeedMode, experiment_id: str, count: int
) -> dict[str, object]:
    started = time.monotonic()
    samples = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            lambda index: upload_one(args.tracking_uri, experiment_id, mode.name, index),
            range(count),
        )
        for completed, result in enumerate(results, start=1):
            if len(samples) < 3:
                samples.append(result)
            if completed % 500 == 0 or completed == count:
                elapsed = time.monotonic() - started
                print(
                    f"{mode.name}: {completed}/{count} traces "
                    f"({completed / max(elapsed, 0.001):.1f} traces/s)",
                    flush=True,
                )
    return {
        "mode": mode.name,
        "experiment_id": experiment_id,
        "experiment_name": mode.experiment_name,
        "artifact_location": mode.artifact_location,
        "created": count,
        "seconds": round(time.monotonic() - started, 2),
        "sample_traces": [trace_id for trace_id, _ in samples],
        "sample_artifact_locations": [location for _, location in samples],
    }


def main() -> None:
    args = parse_args()
    if args.total < 1:
        raise ValueError("--total must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    configure_environment(args)

    modes = []
    if args.mode in ("both", "proxy"):
        modes.append(
            SeedMode(
                name="proxy",
                experiment_name=f"{args.experiment_prefix} - Tracking Server Proxy",
                artifact_location="mlflow-artifacts:/minio-proxy",
                object_prefix="minio-proxy",
            )
        )
    if args.mode in ("both", "direct"):
        modes.append(
            SeedMode(
                name="direct",
                experiment_name=f"{args.experiment_prefix} - Client Direct",
                artifact_location=f"s3://{args.bucket}/minio-direct",
                object_prefix="minio-direct",
            )
        )

    tracking_client = MlflowClient(tracking_uri=args.tracking_uri)
    counts = [args.total // len(modes)] * len(modes)
    for index in range(args.total % len(modes)):
        counts[index] += 1

    summaries = []
    for mode, count in zip(modes, counts, strict=True):
        experiment_id = get_or_create_experiment(tracking_client, mode)
        summaries.append(seed_mode(args, mode, experiment_id, count))

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        region_name=args.region,
    )
    for mode, summary in zip(modes, summaries, strict=True):
        summary["objects_in_prefix"] = count_objects(s3_client, args.bucket, mode.object_prefix)
        summary["verified_sample_span_count"] = verify_trace_view(
            args.tracking_uri, summary["sample_traces"][0]
        )
        summary["ui_url"] = (
            f"{args.tracking_uri.rstrip('/')}/#/experiments/{summary['experiment_id']}/traces"
        )

    print(json.dumps({"total_created": args.total, "results": summaries}, indent=2))


if __name__ == "__main__":
    main()
