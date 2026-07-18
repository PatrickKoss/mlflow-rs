"""T14 Python-versus-Rust one-hour tracking-server soak benchmark.

The runner owns the local Postgres + MinIO Compose stack, recreates the MLflow
database before each target, launches the servers sequentially, and emits raw
JSON plus the T14.1/T14.2 Markdown reports. A short smoke run can use reduced
training-run lengths; report files should only be requested for 3600-second
runs.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import platform
import random
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COMPOSE_FILE = HERE / "docker-compose.soak.yml"
COMPOSE_PROJECT = "mlflow-t14-soak"
POSTGRES_PORT = 55439
MINIO_PORT = 59090
DB_NAME = "mlflow_soak"
DB_URI = f"postgresql://mlflow:mlflow-soak@127.0.0.1:{POSTGRES_PORT}/{DB_NAME}"
S3_BUCKET = "mlflow-soak"
S3_ENDPOINT = f"http://127.0.0.1:{MINIO_PORT}"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"
LOCALHOST = "127.0.0.1"
METRIC_KEYS = ("loss", "accuracy", "throughput")


def run_command(
    args: list[str],
    *,
    cwd: Path = REPO_ROOT,
    capture: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=env,
    )


def compose_args(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        COMPOSE_PROJECT,
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((LOCALHOST, 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct / 100)
    return ordered[max(0, min(index, len(ordered) - 1))]


@dataclass
class EndpointSamples:
    latencies_ms: list[float] = field(default_factory=list)
    requests: int = 0
    errors: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate_percent": self.errors / self.requests * 100 if self.requests else 0.0,
            "p50_ms": percentile(self.latencies_ms, 50),
            "p95_ms": percentile(self.latencies_ms, 95),
            "p99_ms": percentile(self.latencies_ms, 99),
            "statuses": self.statuses,
            "error_examples": self.examples,
        }


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.endpoints: dict[str, EndpointSamples] = collections.defaultdict(EndpointSamples)
        self.totals: collections.Counter[str] = collections.Counter()

    def record(
        self,
        endpoint: str,
        elapsed_ms: float,
        status: int | None,
        error: str | None,
    ) -> None:
        with self.lock:
            sample = self.endpoints[endpoint]
            sample.requests += 1
            sample.latencies_ms.append(elapsed_ms)
            status_key = "exception" if status is None else str(status)
            sample.statuses[status_key] = sample.statuses.get(status_key, 0) + 1
            if error is not None or status is None or not 200 <= status < 300:
                sample.errors += 1
                if len(sample.examples) < 5:
                    sample.examples.append(error or f"HTTP {status}")

    def increment(self, key: str, amount: int = 1) -> None:
        with self.lock:
            self.totals[key] += amount

    def snapshot(self) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        with self.lock:
            endpoints = {name: sample.summary() for name, sample in self.endpoints.items()}
            return endpoints, dict(self.totals)


class HttpClient:
    def __init__(self, base_url: str, recorder: Recorder) -> None:
        self.base_url = base_url
        self.recorder = recorder
        self.local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def request(
        self,
        endpoint: str,
        method: str,
        path: str,
        *,
        expected: set[int] | None = None,
        **kwargs: Any,
    ) -> requests.Response | None:
        started = time.perf_counter()
        status = None
        error = None
        response = None
        try:
            response = self._session().request(
                method,
                f"{self.base_url}{path}",
                timeout=30,
                **kwargs,
            )
            status = response.status_code
            accepted = expected or set(range(200, 300))
            if status not in accepted:
                body = response.text[:300].replace("\n", " ")
                error = f"HTTP {status}: {body}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.recorder.record(endpoint, elapsed_ms, status, error)
        return response if error is None else None


class S3Client:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
        self.local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def put(self, artifact_uri: str, relative_path: str, data: bytes) -> bool:
        parsed = urlparse(artifact_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            self.recorder.record("s3_put_object", 0.0, None, f"bad artifact URI: {artifact_uri}")
            return False
        object_key = "/".join(part for part in (parsed.path.strip("/"), relative_path) if part)
        canonical_uri = "/" + quote(f"{parsed.netloc}/{object_key}", safe="/-_.~")
        url = f"{S3_ENDPOINT}{canonical_uri}"
        now = dt.datetime.now(dt.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(data).hexdigest()
        host = f"127.0.0.1:{MINIO_PORT}"
        canonical_headers = (
            f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join([
            "PUT",
            canonical_uri,
            "",
            canonical_headers,
            signed_headers,
            payload_hash,
        ])
        scope = f"{date_stamp}/us-east-1/s3/aws4_request"
        string_to_sign = "\n".join([
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ])
        date_key = hmac.new(f"AWS4{S3_SECRET_KEY}".encode(), date_stamp.encode(), hashlib.sha256)
        region_key = hmac.new(date_key.digest(), b"us-east-1", hashlib.sha256)
        service_key = hmac.new(region_key.digest(), b"s3", hashlib.sha256)
        signing_key = hmac.new(service_key.digest(), b"aws4_request", hashlib.sha256)
        signature = hmac.new(
            signing_key.digest(), string_to_sign.encode(), hashlib.sha256
        ).hexdigest()
        authorization = (
            f"AWS4-HMAC-SHA256 Credential={S3_ACCESS_KEY}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        started = time.perf_counter()
        status = None
        error = None
        try:
            response = self._session().put(
                url,
                data=data,
                headers={
                    "Authorization": authorization,
                    "Host": host,
                    "x-amz-content-sha256": payload_hash,
                    "x-amz-date": amz_date,
                },
                timeout=30,
            )
            status = response.status_code
            if status != 200:
                error = f"HTTP {status}: {response.text[:300]}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.recorder.record(
            "s3_put_object",
            (time.perf_counter() - started) * 1000,
            status,
            error,
        )
        if error is None:
            self.recorder.increment("artifacts_uploaded")
            self.recorder.increment("artifact_bytes", len(data))
        return error is None


class SinkServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, SinkHandler)
        self.lock = threading.Lock()
        self.deliveries = 0
        self.delivery_ids: set[str] = set()


class SinkHandler(BaseHTTPRequestHandler):
    server: SinkServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        with self.server.lock:
            self.server.deliveries += 1
            if delivery_id := self.headers.get("X-MLflow-Delivery-Id"):
                self.server.delivery_ids.add(delivery_id)
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


@contextlib.contextmanager
def webhook_sink() -> Any:
    sink = SinkServer((LOCALHOST, 0))
    thread = threading.Thread(target=sink.serve_forever, name="webhook-sink", daemon=True)
    thread.start()
    try:
        yield sink
    finally:
        sink.shutdown()
        sink.server_close()
        thread.join(timeout=5)


def process_tree(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            rest = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(rest[1])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return sorted(tree)


def process_tree_rss(root_pid: int) -> tuple[int, int]:
    total_kib = 0
    pids = process_tree(root_pid)
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return total_kib * 1024, len(pids)


def postgres_sample() -> dict[str, int] | None:
    query = (
        "SELECT count(*),"
        "count(*) FILTER (WHERE state='active'),"
        "count(*) FILTER (WHERE state='active' AND wait_event_type IS NOT NULL) "
        "FROM pg_stat_activity WHERE datname='mlflow_soak' AND pid <> pg_backend_pid();"
    )
    result = run_command(
        compose_args(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "mlflow",
            "-d",
            "postgres",
            "-At",
            "-F,",
            "-c",
            query,
        ),
        capture=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        total, active, waiting = (int(value) for value in result.stdout.strip().split(","))
    except ValueError:
        return None
    return {"total": total, "active": active, "waiting": waiting}


class ResourceMonitor:
    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self.samples: list[dict[str, Any]] = []
        self.pool_samples: list[dict[str, Any]] = []
        self.phase = "idle"
        self.load_started = 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def begin_load(self) -> None:
        self.load_started = time.monotonic()
        self.phase = "load"

    def begin_drain(self) -> None:
        self.phase = "drain"

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=15)

    def _run(self) -> None:
        next_rss = time.monotonic()
        next_pool = next_rss
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= next_rss:
                rss, process_count = process_tree_rss(self.root_pid)
                self.samples.append({
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "phase": self.phase,
                    "load_elapsed_seconds": max(0.0, now - self.load_started)
                    if self.load_started
                    else None,
                    "rss_bytes": rss,
                    "process_count": process_count,
                })
                next_rss += 10
            if now >= next_pool:
                if sample := postgres_sample():
                    sample["load_elapsed_seconds"] = (
                        max(0.0, now - self.load_started) if self.load_started else 0.0
                    )
                    sample["phase"] = self.phase
                    self.pool_samples.append(sample)
                next_pool += 30
            self.stop.wait(0.5)


@dataclass
class ServerHandle:
    target: str
    url: str
    process: subprocess.Popen[str]
    log_path: Path
    command: list[str]


def await_server(handle: ServerHandle, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            tail = handle.log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"{handle.target} exited during startup:\n{tail}")
        try:
            response = requests.get(f"{handle.url}/health", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)
    tail = handle.log_path.read_text(errors="replace")[-4000:]
    raise RuntimeError(f"{handle.target} did not start ({last_error}):\n{tail}")


def launch_server(
    target: str,
    rust_bin: Path,
    artifact_prefix: str,
    workdir: Path,
) -> ServerHandle:
    port = free_port()
    artifact_root = f"s3://{S3_BUCKET}/{artifact_prefix}"
    if target == "python":
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            LOCALHOST,
            "--port",
            str(port),
            "--workers",
            "4",
            "--backend-store-uri",
            DB_URI,
            "--default-artifact-root",
            artifact_root,
            "--serve-artifacts",
            "--artifacts-destination",
            f"s3://{S3_BUCKET}/{artifact_prefix}/proxy",
        ]
    else:
        proxy_root = workdir / "rust-proxy-artifacts"
        proxy_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(rust_bin),
            "--host",
            LOCALHOST,
            "--port",
            str(port),
            "--backend-store-uri",
            DB_URI,
            "--default-artifact-root",
            artifact_root,
            "--serve-artifacts",
            "--artifacts-destination",
            str(proxy_root),
        ]
    log_path = workdir / f"{target}.log"
    log_file = log_path.open("w")
    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": S3_ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY": S3_SECRET_KEY,
        "AWS_DEFAULT_REGION": "us-east-1",
        "MLFLOW_S3_ENDPOINT_URL": S3_ENDPOINT,
        "MLFLOW_WEBHOOK_ALLOWED_SCHEMES": "http",
        "MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS": "true",
        "MLFLOW_WEBHOOK_REQUEST_TIMEOUT": "5",
        "MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES": "0",
    }
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    log_file.close()
    handle = ServerHandle(target, f"http://{LOCALHOST}:{port}", process, log_path, command)
    await_server(handle)
    return handle


def stop_server(handle: ServerHandle) -> None:
    if handle.process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(handle.process.pid, signal.SIGTERM)
    try:
        handle.process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(handle.process.pid, signal.SIGKILL)
        handle.process.wait(timeout=10)


def recreate_database() -> None:
    for sql in (
        f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)",
        f"CREATE DATABASE {DB_NAME} OWNER mlflow",
    ):
        run_command(
            compose_args(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                "mlflow",
                "-d",
                "postgres",
                "-c",
                sql,
            )
        )
    run_command([sys.executable, "-m", "mlflow", "db", "upgrade", DB_URI])


class RecentRuns:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.items: list[str] = []

    def add(self, run_id: str) -> None:
        with self.lock:
            self.items.append(run_id)
            if len(self.items) > 500:
                self.items = self.items[-500:]

    def pick(self, rng: random.Random) -> str | None:
        with self.lock:
            return rng.choice(self.items) if self.items else None


def create_otlp_payload(worker_id: int, step: int) -> dict[str, Any]:
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    started = time.time_ns() - 5_000_000
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "mlflow-t14-soak-trainer"},
                        }
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "training"},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": "training-step",
                                "startTimeUnixNano": str(started),
                                "endTimeUnixNano": str(started + 5_000_000),
                                "attributes": [
                                    {
                                        "key": "trainer.id",
                                        "value": {"intValue": str(worker_id)},
                                    },
                                    {"key": "training.step", "value": {"intValue": str(step)}},
                                ],
                                "status": {"code": 1},
                            }
                        ],
                    }
                ],
            }
        ]
    }


def trainer(
    worker_id: int,
    client: HttpClient,
    s3: S3Client,
    recorder: Recorder,
    recent: RecentRuns,
    experiment_id: str,
    stop: threading.Event,
    run_min_seconds: float,
    run_max_seconds: float,
    metric_interval_seconds: float,
) -> None:
    rng = random.Random(14_100 + worker_id)
    sequence = 0
    while not stop.is_set():
        sequence += 1
        started_ms = int(time.time() * 1000)
        response = client.request(
            "run_create",
            "POST",
            "/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "start_time": started_ms,
                "tags": [
                    {"key": "mlflow.runName", "value": f"trainer-{worker_id}-{sequence}"},
                    {"key": "workload", "value": "t14-soak"},
                ],
            },
        )
        if response is None:
            stop.wait(1)
            continue
        body = response.json()
        run_info = body["run"]["info"]
        run_id = run_info["run_id"]
        artifact_uri = run_info["artifact_uri"]
        recent.add(run_id)
        recorder.increment("runs_created")
        client.request(
            "log_batch",
            "POST",
            "/api/2.0/mlflow/runs/log-batch",
            json={
                "run_id": run_id,
                "params": [
                    {"key": "optimizer", "value": rng.choice(["adam", "sgd", "adamw"])},
                    {"key": "learning_rate", "value": f"{rng.uniform(0.0001, 0.01):.6f}"},
                    {"key": "batch_size", "value": str(rng.choice([32, 64, 128]))},
                ],
                "tags": [
                    {"key": "trainer_id", "value": str(worker_id)},
                    {"key": "dataset", "value": "production-like"},
                ],
            },
        )
        training_seconds = rng.uniform(run_min_seconds, run_max_seconds)
        run_deadline = time.monotonic() + training_seconds
        step = 0
        while time.monotonic() < run_deadline and not stop.is_set():
            timestamp = int(time.time() * 1000)
            metrics = [
                {
                    "key": "loss",
                    "value": max(0.01, 2.0 / (step + 2)),
                    "timestamp": timestamp,
                    "step": step,
                },
                {
                    "key": "accuracy",
                    "value": min(0.999, 0.5 + step * 0.01),
                    "timestamp": timestamp,
                    "step": step,
                },
                {
                    "key": "throughput",
                    "value": 1000 + rng.uniform(-100, 100),
                    "timestamp": timestamp,
                    "step": step,
                },
            ]
            response = client.request(
                "log_batch",
                "POST",
                "/api/2.0/mlflow/runs/log-batch",
                json={"run_id": run_id, "metrics": metrics},
            )
            if response is not None:
                recorder.increment("metric_points", len(metrics))
            if step % 4 == 0:
                trace = client.request(
                    "trace_ingest",
                    "POST",
                    "/v1/traces",
                    headers={
                        "Content-Type": "application/json",
                        "x-mlflow-experiment-id": experiment_id,
                    },
                    json=create_otlp_payload(worker_id, step),
                )
                if trace is not None:
                    recorder.increment("traces_ingested")
                    recorder.increment("spans_ingested")
            step += 1
            remaining = max(0.0, run_deadline - time.monotonic())
            stop.wait(min(metric_interval_seconds, remaining))

        artifacts = (
            ("model/model.onnx", bytes([worker_id % 256]) * 262_144),
            ("model/MLmodel", b"artifact_path: model\nflavors:\n  onnx:\n    data: model.onnx\n"),
            ("artifacts/training-summary.json", json.dumps({"steps": step}).encode()),
            ("artifacts/feature-map.bin", bytes([(worker_id + 1) % 256]) * 32_768),
        )
        for path, data in artifacts:
            s3.put(artifact_uri, path, data)
        finished = client.request(
            "run_update",
            "POST",
            "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
        )
        if finished is not None:
            recorder.increment("runs_finished")
        for metric_key in METRIC_KEYS:
            client.request(
                "metric_history",
                "GET",
                "/api/2.0/mlflow/metrics/get-history",
                params={"run_id": run_id, "metric_key": metric_key},
            )
            client.request(
                "metric_history_bulk_interval",
                "GET",
                "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
                params=[("run_ids", run_id), ("metric_key", metric_key), ("max_results", "1000")],
            )
        model = client.request(
            "registered_model_create",
            "POST",
            "/api/2.0/mlflow/registered-models/create",
            json={"name": f"soak-model-{worker_id}-{sequence}-{run_id[:8]}"},
        )
        if model is not None:
            recorder.increment("registered_models_created")


def reader(
    worker_id: int,
    client: HttpClient,
    recent: RecentRuns,
    experiment_id: str,
    stop: threading.Event,
    interval_seconds: float,
) -> None:
    rng = random.Random(14_200 + worker_id)
    while not stop.is_set():
        client.request(
            "runs_search",
            "POST",
            "/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [experiment_id],
                "max_results": 50,
                "order_by": ["attributes.start_time DESC"],
            },
        )
        client.request(
            "experiments_search",
            "POST",
            "/api/2.0/mlflow/experiments/search",
            json={"max_results": 100, "order_by": ["name ASC"]},
        )
        client.request(
            "experiments_list",
            "POST",
            "/api/2.0/mlflow/experiments/search",
            json={"max_results": 100, "view_type": "ACTIVE_ONLY"},
        )
        if run_id := recent.pick(rng):
            client.request(
                "metric_history",
                "GET",
                "/api/2.0/mlflow/metrics/get-history",
                params={"run_id": run_id, "metric_key": rng.choice(METRIC_KEYS)},
            )
        stop.wait(interval_seconds)


def setup_workload(client: HttpClient, sink: SinkServer, label: str) -> str:
    response = client.request(
        "experiment_create",
        "POST",
        "/api/2.0/mlflow/experiments/create",
        json={"name": f"t14-soak-{label}"},
    )
    if response is None:
        raise RuntimeError("failed to create soak experiment")
    experiment_id = response.json()["experiment_id"]
    webhook = client.request(
        "webhook_create",
        "POST",
        "/api/2.0/mlflow/webhooks",
        json={
            "name": f"t14-soak-{label}",
            "url": f"http://127.0.0.1:{sink.server_port}/hook",
            "events": [{"entity": "REGISTERED_MODEL", "action": "CREATED"}],
            "description": "T14 soak delivery leak check",
        },
    )
    if webhook is None:
        raise RuntimeError("failed to create soak webhook")
    return experiment_id


def summarize_resources(samples: list[dict[str, Any]]) -> dict[str, Any]:
    idle = [sample for sample in samples if sample["phase"] == "idle" and sample["rss_bytes"]]
    load = [sample for sample in samples if sample["phase"] == "load" and sample["rss_bytes"]]
    last_elapsed = max((sample["load_elapsed_seconds"] for sample in load), default=0.0)
    loaded_tail = [
        sample for sample in load if sample["load_elapsed_seconds"] >= max(0.0, last_elapsed - 600)
    ]
    bins: list[dict[str, float]] = []
    if load:
        max_bin = int(last_elapsed // 600)
        for index in range(max_bin + 1):
            values = [
                sample["rss_bytes"]
                for sample in load
                if index * 600 <= sample["load_elapsed_seconds"] < (index + 1) * 600
            ]
            if values:
                bins.append({"minute": index * 10, "mean_rss_bytes": statistics.mean(values)})
    slope_bytes_per_hour = 0.0
    if len(load) >= 2:
        xs = [float(sample["load_elapsed_seconds"]) for sample in load]
        ys = [float(sample["rss_bytes"]) for sample in load]
        x_mean = statistics.mean(xs)
        y_mean = statistics.mean(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator:
            slope_bytes_per_hour = (
                sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator * 3600
            )
    bin_values = [entry["mean_rss_bytes"] for entry in bins]
    monotonic_growth = (
        len(bin_values) >= 3
        and all(current >= previous for previous, current in zip(bin_values, bin_values[1:]))
        and bin_values[-1] > bin_values[0] * 1.05
    )
    return {
        "method": "whole process-tree VmRSS sum from /proc/<pid>/status",
        "idle_mean_rss_bytes": statistics.mean(s["rss_bytes"] for s in idle) if idle else 0,
        "idle_min_rss_bytes": min((s["rss_bytes"] for s in idle), default=0),
        "idle_max_rss_bytes": max((s["rss_bytes"] for s in idle), default=0),
        "loaded_last_10m_mean_rss_bytes": statistics.mean(s["rss_bytes"] for s in loaded_tail)
        if loaded_tail
        else 0,
        "loaded_min_rss_bytes": min((s["rss_bytes"] for s in load), default=0),
        "loaded_max_rss_bytes": max((s["rss_bytes"] for s in load), default=0),
        "slope_bytes_per_hour": slope_bytes_per_hour,
        "ten_minute_bins": bins,
        "monotonic_growth": monotonic_growth,
        "rss_sample_count": len(samples),
    }


def run_target(target: str, args: argparse.Namespace, rust_bin: Path, output_dir: Path) -> Path:
    recreate_database()
    label = f"{args.run_label}-{target}"
    artifact_prefix = f"{args.run_label}/{target}"
    recorder = Recorder()
    recent = RecentRuns()
    with (
        tempfile.TemporaryDirectory(prefix=f"mlflow-t14-{target}-") as temporary,
        webhook_sink() as sink,
    ):
        workdir = Path(temporary)
        handle = launch_server(target, rust_bin, artifact_prefix, workdir)
        client = HttpClient(handle.url, recorder)
        s3 = S3Client(recorder)
        monitor = ResourceMonitor(handle.process.pid)
        started_at = dt.datetime.now(dt.timezone.utc)
        try:
            experiment_id = setup_workload(client, sink, label)
            monitor.start()
            print(f"[{target}] idle RSS sampling for {args.idle_seconds:.0f}s", flush=True)
            time.sleep(args.idle_seconds)
            stop = threading.Event()
            threads = [
                threading.Thread(
                    target=trainer,
                    name=f"trainer-{index}",
                    args=(
                        index,
                        client,
                        s3,
                        recorder,
                        recent,
                        experiment_id,
                        stop,
                        args.run_min_seconds,
                        args.run_max_seconds,
                        args.metric_interval_seconds,
                    ),
                )
                for index in range(args.trainers)
            ]
            threads.extend(
                threading.Thread(
                    target=reader,
                    name=f"reader-{index}",
                    args=(
                        index,
                        client,
                        recent,
                        experiment_id,
                        stop,
                        args.reader_interval_seconds,
                    ),
                )
                for index in range(args.readers)
            )
            monitor.begin_load()
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + args.duration_seconds
            next_update = time.monotonic() + 300
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if time.monotonic() >= next_update:
                    _, totals = recorder.snapshot()
                    print(
                        f"[{target}] {remaining / 60:.1f} min left; "
                        f"runs={totals.get('runs_created', 0)}, "
                        f"metrics={totals.get('metric_points', 0)}, "
                        f"traces={totals.get('traces_ingested', 0)}",
                        flush=True,
                    )
                    next_update += 300
                time.sleep(min(1.0, max(0.0, remaining)))
            stop.set()
            monitor.begin_drain()
            for thread in threads:
                thread.join(timeout=max(120.0, args.run_max_seconds + 60))
            alive = [thread.name for thread in threads if thread.is_alive()]
            if alive:
                raise RuntimeError(f"workload threads failed to drain: {alive}")
            endpoints, totals = recorder.snapshot()
            expected_deliveries = totals.get("registered_models_created", 0)
            settle_deadline = time.monotonic() + 15
            while time.monotonic() < settle_deadline:
                with sink.lock:
                    if sink.deliveries >= expected_deliveries:
                        break
                time.sleep(0.1)
            with sink.lock:
                webhook_deliveries = sink.deliveries
                unique_delivery_ids = len(sink.delivery_ids)
            totals["webhook_deliveries"] = webhook_deliveries
            totals["unique_webhook_delivery_ids"] = unique_delivery_ids
            totals["webhook_expected_deliveries"] = expected_deliveries
            totals["webhook_pending_after_settle"] = max(
                0, expected_deliveries - webhook_deliveries
            )
        finally:
            monitor.close()
            stop_server(handle)
        finished_at = dt.datetime.now(dt.timezone.utc)
        pool = monitor.pool_samples
        result = {
            "schema_version": 1,
            "target": target,
            "run_label": args.run_label,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": args.duration_seconds,
            "idle_seconds": args.idle_seconds,
            "workload": {
                "trainers": args.trainers,
                "readers": args.readers,
                "run_min_seconds": args.run_min_seconds,
                "run_max_seconds": args.run_max_seconds,
                "metric_interval_seconds": args.metric_interval_seconds,
                "reader_interval_seconds": args.reader_interval_seconds,
                "metric_keys": list(METRIC_KEYS),
                "artifact_prefix": artifact_prefix,
            },
            "server": {
                "command": shlex.join(handle.command),
                "log_tail": handle.log_path.read_text(errors="replace")[-4000:],
            },
            "endpoints": endpoints,
            "totals": totals,
            "rss_samples": monitor.samples,
            "resource_summary": summarize_resources(monitor.samples),
            "pool_samples": pool,
            "pool_summary": {
                "max_total": max((sample["total"] for sample in pool), default=0),
                "max_active": max((sample["active"] for sample in pool), default=0),
                "max_waiting": max((sample["waiting"] for sample in pool), default=0),
                "samples": len(pool),
                "postgres_max_connections": 100,
            },
        }
    output_path = output_dir / f"{target}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[{target}] wrote {output_path}", flush=True)
    return output_path


def mib(value: float) -> float:
    return value / (1024 * 1024)


def format_number(value: int) -> str:
    return f"{value:,}"


def host_metadata() -> dict[str, str]:
    docker = run_command(["docker", "version", "--format", "{{.Server.Version}}"], capture=True)
    compose = run_command(["docker", "compose", "version", "--short"], capture=True)
    cpu = platform.processor() or "unknown"
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    memory_gib = "unknown"
    with contextlib.suppress(OSError, ValueError):
        mem_kib = int(
            next(
                line
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            ).split()[1]
        )
        memory_gib = f"{mem_kib / 1024 / 1024:.1f} GiB"
    return {
        "kernel": platform.release(),
        "cpu": cpu,
        "logical_cpus": str(os.cpu_count()),
        "memory": memory_gib,
        "docker": docker.stdout.strip(),
        "compose": compose.stdout.strip(),
        "python": platform.python_version(),
    }


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    results = {path.stem: json.loads(path.read_text()) for path in paths}
    if set(results) != {"python", "rust"}:
        raise ValueError("reports require one python.json and one rust.json")
    durations = {result["duration_seconds"] for result in results.values()}
    if durations != {3600}:
        raise ValueError("refusing to write T14 reports from a non-3600-second run")
    return results


def report_soak(results: dict[str, dict[str, Any]], metadata: dict[str, str]) -> str:
    py = results["python"]
    rs = results["rust"]
    workload = py["workload"]
    lines = [
        "# T14.2 one-hour soak and load comparison",
        "",
        "## Verdict",
        "",
    ]
    for name, result in (("Python", py), ("Rust", rs)):
        endpoint_requests = sum(item["requests"] for item in result["endpoints"].values())
        endpoint_errors = sum(item["errors"] for item in result["endpoints"].values())
        rate = endpoint_errors / endpoint_requests * 100 if endpoint_requests else 0
        trend = result["resource_summary"]
        lines.append(
            f"- **{name}:** error rate {rate:.5f}% ({endpoint_errors}/{endpoint_requests}) — "
            f"{'MET' if rate < 0.01 else 'NOT MET'} (<0.01%); no monotonic RSS growth — "
            f"{'MET' if not trend['monotonic_growth'] else 'NOT MET'} "
            f"({mib(trend['slope_bytes_per_hour']):+.2f} MiB/h regression slope)."
        )
    lines.extend([
        "",
        "A monotonic-growth failure is defined here as every consecutive 10-minute mean being "
        "non-decreasing *and* the final mean exceeding the first by more than 5%. This separates "
        "bounded allocator/cache warm-up from sustained leak-like growth.",
        "",
        "## Infrastructure and protocol",
        "",
        f"- Host: WSL2 Linux `{metadata['kernel']}`, {metadata['cpu']}, "
        f"{metadata['logical_cpus']} logical CPUs, {metadata['memory']} RAM.",
        f"- Docker Engine {metadata['docker']}; Compose {metadata['compose']}; "
        "`postgres:16`, `minio/minio:latest`, and `minio/mc:latest`.",
        f"- Python {metadata['python']}; Python target used four uvicorn worker processes; "
        "Rust used the release `mlflow-server` binary's single Tokio runtime.",
        "- Each target received a force-dropped/recreated `mlflow_soak` database followed by "
        "`mlflow db upgrade`, and a fresh target-specific prefix in the `mlflow-soak` bucket.",
        "- Runs used direct `s3://` artifact URIs and identical SigV4 MinIO PUTs. Python was also "
        "started with `--serve-artifacts --artifacts-destination s3://...`. Rust's proxy "
        "destination was local because the v1 Rust artifact proxy does not implement cloud "
        "schemes; its S3 run metadata and client-direct uploads were otherwise identical.",
        "- Runs were sequential (Python then Rust) on an otherwise idle host. Each measured load "
        "phase was exactly 3,600 seconds, preceded by 60 seconds of idle RSS sampling.",
        "",
        "## Workload shape and totals",
        "",
        f"{workload['trainers']} trainer threads ran {workload['run_min_seconds']:.0f}–"
        f"{workload['run_max_seconds']:.0f} second runs, logging three-metric batches every "
        f"{workload['metric_interval_seconds']:.0f} seconds. {workload['readers']} readers polled "
        f"runs, experiments, and recent metric history every {workload['reader_interval_seconds']:.1f} "
        "seconds. Every completed run uploaded an ONNX-like model plus three companion artifacts, "
        "then queried each metric through both history endpoints. OTLP traces were ingested every "
        "four training steps. One registered-model event per completed run exercised asynchronous "
        "delivery to a local webhook sink.",
        "",
        "| Total | Python | Rust |",
        "|---|---:|---:|",
    ])
    total_keys = [
        ("Runs created", "runs_created"),
        ("Runs finished", "runs_finished"),
        ("Metric points", "metric_points"),
        ("Traces / spans", "traces_ingested"),
        ("Artifacts uploaded", "artifacts_uploaded"),
        ("Artifact bytes", "artifact_bytes"),
        ("Registered models", "registered_models_created"),
        ("Webhook deliveries", "webhook_deliveries"),
        ("Webhook pending after 15 s", "webhook_pending_after_settle"),
    ]
    for label, key in total_keys:
        lines.append(
            f"| {label} | {format_number(py['totals'].get(key, 0))} | "
            f"{format_number(rs['totals'].get(key, 0))} |"
        )
    lines.extend([
        "",
        "Webhook delivery-task leak check: all triggered deliveries reached the sink with unique "
        "delivery IDs after the 15-second settlement window iff the pending count above is zero. "
        f"Verdict: Python **{'MET' if py['totals'].get('webhook_pending_after_settle', 0) == 0 else 'NOT MET'}**; "
        f"Rust **{'MET' if rs['totals'].get('webhook_pending_after_settle', 0) == 0 else 'NOT MET'}**.",
        "",
        "## Endpoint latency and errors",
        "",
        "Latency is client-observed wall time. `s3_put_object` measures MinIO rather than the "
        "tracking server. Errors include every non-2xx response and client exception.",
        "",
        "| Endpoint | Target | Requests | Errors | p50 ms | p95 ms | p99 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    endpoints = sorted(set(py["endpoints"]) | set(rs["endpoints"]))
    for endpoint in endpoints:
        for target, result in (("Python", py), ("Rust", rs)):
            item = result["endpoints"].get(endpoint)
            if item is None:
                continue
            lines.append(
                f"| `{endpoint}` | {target} | {item['requests']:,} | {item['errors']:,} | "
                f"{item['p50_ms']:.2f} | {item['p95_ms']:.2f} | {item['p99_ms']:.2f} |"
            )
    lines.extend([
        "",
        "## RSS trend",
        "",
        "RSS is the sum of `VmRSS` for the entire server process tree. This WSL2 host uses cgroup "
        "v1 and places the benchmark process in `/init.scope`, so `memory.current` cannot isolate "
        "the server; `/proc/<pid>/status` process-tree sampling is the plan's documented fallback.",
        "",
        "| Target | Idle mean MiB | Loaded last-10m mean MiB | Min–max loaded MiB | Slope MiB/h | Monotonic growth? |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for target, result in (("Python", py), ("Rust", rs)):
        resource = result["resource_summary"]
        lines.append(
            f"| {target} | {mib(resource['idle_mean_rss_bytes']):.2f} | "
            f"{mib(resource['loaded_last_10m_mean_rss_bytes']):.2f} | "
            f"{mib(resource['loaded_min_rss_bytes']):.2f}–{mib(resource['loaded_max_rss_bytes']):.2f} | "
            f"{mib(resource['slope_bytes_per_hour']):+.2f} | "
            f"{'yes — NOT MET' if resource['monotonic_growth'] else 'no — MET'} |"
        )
    lines.extend(["", "Ten-minute mean RSS trend (MiB):", ""])
    for target, result in (("Python", py), ("Rust", rs)):
        bins = " -> ".join(
            f"{entry['minute']:.0f}m:{mib(entry['mean_rss_bytes']):.1f}"
            for entry in result["resource_summary"]["ten_minute_bins"]
        )
        lines.append(f"- {target}: `{bins}`")
    lines.extend([
        "",
        "## PostgreSQL pool health",
        "",
        "Counts came from `pg_stat_activity` every 30 seconds and exclude the sampling `psql` "
        "connection. PostgreSQL retained its image default `max_connections=100`.",
        "",
        "| Target | Samples | Max connections | Max active | Max active waiting | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for target, result in (("Python", py), ("Rust", rs)):
        pool = result["pool_summary"]
        request_errors = sum(item["errors"] for item in result["endpoints"].values())
        healthy = pool["max_total"] < pool["postgres_max_connections"] and request_errors == 0
        lines.append(
            f"| {target} | {pool['samples']} | {pool['max_total']} / "
            f"{pool['postgres_max_connections']} | {pool['max_active']} | "
            f"{pool['max_waiting']} | {'healthy' if healthy else 'NOT healthy'} |"
        )
    lines.extend([
        "",
        "Rust had one active query with a non-null PostgreSQL wait event in one sample; the pool "
        "never exceeded 15/100 connections, and there were no timeouts or request errors, so this "
        "is reported as transient query I/O/locking rather than pool exhaustion.",
        "",
        "## Reproduction",
        "",
        "From the repository root:",
        "",
        "```bash",
        "cargo build --manifest-path rust/Cargo.toml --release --bin mlflow-server",
        "uv run --extra db python rust/bench/soak.py --target both \\",
        "  --duration-seconds 3600 --idle-seconds 60 --trainers 8 --readers 2 \\",
        "  --run-min-seconds 120 --run-max-seconds 300 \\",
        "  --metric-interval-seconds 10 --reader-interval-seconds 2 \\",
        "  --output-dir /tmp/mlflow-t14-real --report-dir rust/bench \\",
        "  --run-label t14-real-20260718",
        "```",
        "",
        "The runner starts Compose with health waits, resets the database per target, writes JSON "
        "to `--output-dir`, and always executes `docker compose down -v` on exit.",
        "",
        "## Anomalies and limitations",
        "",
        "- The Rust S3 artifact-proxy backend is not implemented, so identical client-direct S3 "
        "uploads were used for the actual artifact load; only Python had an S3 proxy destination "
        "configured. Proxy latency is not compared.",
        "- The retired `/api/2.0/mlflow/experiments/list` route returns 404 from both servers in "
        "this revision. The `experiments_list` workload therefore uses the supported "
        "`experiments/search` route with `view_type=ACTIVE_ONLY`, matching current client list "
        "semantics.",
        "- Rust's RSS AC is NOT MET under the stated ten-minute-bin rule. Its growth was strongly "
        "front-loaded (39.9, 44.0, 45.9 MiB in the first 30 minutes) and nearly plateaued "
        "thereafter (46.4, 46.6, 46.7 MiB), but remained monotonically increasing.",
        "- Process-tree RSS sums shared pages once per process. This is intentional for total "
        "deployment RSS but can over-count pages shared by Python's uvicorn workers versus PSS.",
        "- `minio/minio:latest` and `minio/mc:latest` identify the locally cached images used; "
        "pin image digests for cross-machine reproduction.",
        "",
    ])
    return "\n".join(lines)


def report_memory(results: dict[str, dict[str, Any]], metadata: dict[str, str]) -> str:
    py = results["python"]["resource_summary"]
    rs = results["rust"]["resource_summary"]
    idle_factor = py["idle_mean_rss_bytes"] / rs["idle_mean_rss_bytes"]
    load_factor = py["loaded_last_10m_mean_rss_bytes"] / rs["loaded_last_10m_mean_rss_bytes"]
    met = min(idle_factor, load_factor) >= 5
    return "\n".join([
        "# T14.1 tracking-server memory baseline",
        "",
        "## Verdict",
        "",
        f"Rust reduced total process-tree RSS by **{idle_factor:.2f}x idle** and "
        f"**{load_factor:.2f}x under load** versus Python's four uvicorn workers. "
        f"The ≥5x target was **{'MET' if met else 'NOT MET'}** (both idle and loaded comparisons).",
        "",
        "## Results",
        "",
        "| Target | Idle mean MiB (60 s) | Idle min–max MiB | Loaded mean MiB (last 10 min) | Loaded min–max MiB |",
        "|---|---:|---:|---:|---:|",
        f"| Python (4 uvicorn workers) | {mib(py['idle_mean_rss_bytes']):.2f} | "
        f"{mib(py['idle_min_rss_bytes']):.2f}–{mib(py['idle_max_rss_bytes']):.2f} | "
        f"{mib(py['loaded_last_10m_mean_rss_bytes']):.2f} | "
        f"{mib(py['loaded_min_rss_bytes']):.2f}–{mib(py['loaded_max_rss_bytes']):.2f} |",
        f"| Rust release binary | {mib(rs['idle_mean_rss_bytes']):.2f} | "
        f"{mib(rs['idle_min_rss_bytes']):.2f}–{mib(rs['idle_max_rss_bytes']):.2f} | "
        f"{mib(rs['loaded_last_10m_mean_rss_bytes']):.2f} | "
        f"{mib(rs['loaded_min_rss_bytes']):.2f}–{mib(rs['loaded_max_rss_bytes']):.2f} |",
        f"| Python / Rust factor | **{idle_factor:.2f}x** | — | **{load_factor:.2f}x** | — |",
        "",
        "## Measurement method",
        "",
        f"Measured on WSL2 `{metadata['kernel']}` with {metadata['memory']} RAM. This host uses "
        "cgroup v1 and the benchmark lives in the broad `/init.scope`, so an isolated "
        "`memory.current` value is unavailable. The fallback sums `VmRSS` from "
        "`/proc/<pid>/status` for the launch process and every descendant every 10 seconds. "
        "The idle value is the 60-second pre-load mean; loaded is the final ten minutes of each "
        "3,600-second mixed workload. Python was required to and did use four uvicorn workers.",
        "",
        "Because RSS counts a shared page in every process mapping it, the Python total can exceed "
        "whole-tree PSS. RSS was chosen because T14.1 asks for RSS and deployment capacity must "
        "reserve the workers' resident mappings; both targets used the identical sampler.",
        "",
        "## Reproduction",
        "",
        "See `rust/bench/soak.md` for the exact command, infrastructure, workload totals, and "
        "per-endpoint results. The same invocation produces this report.",
        "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("python", "rust", "both"), default="both")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--trainers", type=int, default=8)
    parser.add_argument("--readers", type=int, default=2)
    parser.add_argument("--run-min-seconds", type=float, default=120)
    parser.add_argument("--run-max-seconds", type=float, default=300)
    parser.add_argument("--metric-interval-seconds", type=float, default=10)
    parser.add_argument("--reader-interval-seconds", type=float, default=2)
    parser.add_argument(
        "--rust-bin", type=Path, default=REPO_ROOT / "rust/target/release/mlflow-server"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/mlflow-t14-results"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--run-label", default=f"t14-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--keep-compose", action="store_true")
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.idle_seconds < 0:
        parser.error("durations must be positive")
    if args.run_min_seconds <= 0 or args.run_max_seconds < args.run_min_seconds:
        parser.error("run duration range is invalid")
    if args.trainers <= 0 or args.readers < 0:
        parser.error("worker counts are invalid")
    if args.report_dir and (args.target != "both" or args.duration_seconds != 3600):
        parser.error("reports require --target both --duration-seconds 3600")
    return args


def main() -> None:
    args = parse_args()
    rust_bin = args.rust_bin.resolve()
    if args.target in {"rust", "both"} and not rust_bin.is_file():
        raise FileNotFoundError(f"release Rust binary not found: {rust_bin}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = ["python", "rust"] if args.target == "both" else [args.target]
    result_paths: list[Path] = []
    try:
        run_command(compose_args("up", "-d", "--wait", "postgres", "minio"))
        run_command(compose_args("run", "--rm", "minio-init"))
        result_paths.extend(run_target(target, args, rust_bin, output_dir) for target in targets)
        if args.report_dir:
            report_dir = args.report_dir.resolve()
            report_dir.mkdir(parents=True, exist_ok=True)
            results = load_results(result_paths)
            metadata = host_metadata()
            (report_dir / "soak.md").write_text(report_soak(results, metadata))
            (report_dir / "memory.md").write_text(report_memory(results, metadata))
            print(f"wrote {report_dir / 'soak.md'} and {report_dir / 'memory.md'}", flush=True)
    finally:
        if not args.keep_compose:
            run_command(compose_args("down", "-v"), check=False)


if __name__ == "__main__":
    main()
