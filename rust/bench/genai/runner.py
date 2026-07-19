"""Run the Phase 23 GenAI smoke cell against Python and Rust."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from jsonschema import Draft202012Validator

from rust.bench.genai.equivalence import compare_runs
from rust.bench.genai.metrics import AsyncBenchClient, MetricsCollector, ResourceMonitor
from rust.bench.genai.mock_provider import canonical_request, provider_server

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUST_ROOT = REPO_ROOT / "rust"
COMPOSE_FILE = HERE / "docker-compose.yml"
SCHEMA_FILE = HERE / "raw-metrics.schema.json"
COMPOSE_PROJECT = "mlflow-t23-genai"
POSTGRES_PORT = 55440
MINIO_PORT = 59092
DB_NAME = "mlflow_genai_bench"
DB_URI = f"postgresql://mlflow:mlflow-genai-fake@127.0.0.1:{POSTGRES_PORT}/{DB_NAME}"
S3_BUCKET = "mlflow-genai-bench"
S3_ENDPOINT = f"http://127.0.0.1:{MINIO_PORT}"
FAKE_PASSPHRASE = "t23-obvious-fake-crypto-passphrase"
FAKE_API_KEY = "test-key-fake"
ASSISTANT_PREFIX = "/ajax-api/3.0/mlflow/assistant"
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED"}


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
    return ["docker", "compose", "-p", COMPOSE_PROJECT, "-f", str(COMPOSE_FILE), *args]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ServerHandle:
    target: str
    url: str
    process: subprocess.Popen[str]
    log_path: Path
    command: list[str]


def install_claude_stub() -> tuple[Path, Path]:
    spec = importlib.util.spec_from_file_location(
        "t23_dev_stubs", REPO_ROOT / "dev" / "dev_stubs" / "__init__.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load dev/dev_stubs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    installed = module.install_stubs(["claude"])
    return installed.path_prepend[0], installed.cleanup_paths[0]


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


def postgres_sample() -> dict[str, Any] | None:
    query = (
        "SELECT count(*),"
        "count(*) FILTER (WHERE state='active'),"
        "count(*) FILTER (WHERE state='active' AND wait_event_type IS NOT NULL) "
        f"FROM pg_stat_activity WHERE datname='{DB_NAME}' AND pid <> pg_backend_pid();"
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
    if result.returncode != 0:
        return None
    try:
        total, active, waiting = (int(value) for value in result.stdout.strip().split(","))
    except ValueError:
        return None
    return {"total": total, "active": active, "waiting": waiting, "source": "pg_stat_activity"}


def await_server(handle: ServerHandle, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            raise RuntimeError(
                f"{handle.target} exited during startup:\n"
                + handle.log_path.read_text(errors="replace")[-6000:]
            )
        try:
            if requests.get(f"{handle.url}/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise RuntimeError(
        f"{handle.target} did not start:\n" + handle.log_path.read_text(errors="replace")[-6000:]
    )


def launch_server(
    target: str,
    workdir: Path,
    provider_url: str,
    artifact_prefix: str,
    stub_path: Path,
) -> ServerHandle:
    port = free_port()
    home = workdir / "home"
    home.mkdir()
    assistant_config = home / ".mlflow" / "assistant" / "config.json"
    assistant_config.parent.mkdir(parents=True)
    assistant_config.write_text(
        json.dumps({
            "providers": {
                "claude_code": {
                    "model": "default",
                    "selected": True,
                    "permissions": {
                        "allow_edit_files": True,
                        "allow_read_docs": True,
                        "full_access": False,
                    },
                }
            }
        })
    )
    artifact_root = f"s3://{S3_BUCKET}/{artifact_prefix}"
    if target == "python":
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
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
            artifact_root + "/proxy",
        ]
    else:
        rust_binary = RUST_ROOT / "target" / "release" / "mlflow-server"
        proxy = workdir / "rust-artifacts"
        proxy.mkdir()
        command = [
            str(rust_binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            DB_URI,
            "--default-artifact-root",
            artifact_root,
            "--serve-artifacts",
            "--artifacts-destination",
            str(proxy),
        ]
    log_path = workdir / f"{target}.log"
    log = log_path.open("w")
    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": "minio-fake-access",
        "AWS_SECRET_ACCESS_KEY": "minio-fake-secret",
        "AWS_DEFAULT_REGION": "us-east-1",
        "HOME": str(home),
        "MLFLOW_CRYPTO_KEK_PASSPHRASE": FAKE_PASSPHRASE,
        "MLFLOW_GATEWAY_URI": f"http://127.0.0.1:{port}",
        "MLFLOW_GENAI_EVAL_MAX_RETRIES": "0",
        "MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT": "0",
        "MLFLOW_GENAI_WORKER_PATH": str(RUST_ROOT / "target" / "release" / "mlflow-genai-worker"),
        "MLFLOW_MODEL_CATALOG_URI": "",
        "MLFLOW_S3_ENDPOINT_URL": S3_ENDPOINT,
        "MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE": "true",
        "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "true",
        "PATH": f"{stub_path}{os.pathsep}{os.environ.get('PATH', '')}",
        "TMPDIR": str(workdir / "tmp"),
        "T23_MOCK_PROVIDER_URL": provider_url,
    }
    Path(env["TMPDIR"]).mkdir()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    log.close()
    handle = ServerHandle(target, f"http://127.0.0.1:{port}", process, log_path, command)
    try:
        await_server(handle)
    except BaseException:
        stop_server(handle)
        raise
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


def sync_request(
    session: requests.Session,
    base_url: str,
    method: str,
    path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = session.request(method, base_url + path, timeout=30, **kwargs)
    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"setup {method} {path}: HTTP {response.status_code}: {response.text[:500]}"
        )
    return response.json() if response.content else {}


def setup_target(base_url: str, provider_url: str, seed: int) -> dict[str, str]:
    session = requests.Session()
    try:
        experiment = sync_request(
            session,
            base_url,
            "POST",
            "/api/2.0/mlflow/experiments/create",
            json={"name": f"t23-smoke-{seed}"},
        )
        experiment_id = experiment["experiment_id"]
        dataset = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/datasets/create",
            json={
                "created_by": "t23-bench",
                "experiment_ids": [experiment_id],
                "name": f"t23-smoke-dataset-{seed}",
                "source_type": "HUMAN",
                "tags": json.dumps({"phase": "23", "seed": str(seed)}),
            },
        )
        secret = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "auth_config": {"api_base": f"{provider_url}/v1"},
                "created_by": "t23-bench",
                "provider": "openai",
                "secret_name": f"t23-fake-secret-{seed}",
                "secret_value": {"api_key": FAKE_API_KEY},
            },
        )["secret"]
        model = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "created_by": "t23-bench",
                "model_name": "genai-bench-model",
                "name": f"t23-fake-model-{seed}",
                "provider": "openai",
                "secret_id": secret["secret_id"],
            },
        )["model_definition"]
        endpoint_name = f"t23-smoke-endpoint-{seed}"
        sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "created_by": "t23-bench",
                "model_configs": [
                    {
                        "linkage_type": "PRIMARY",
                        "model_definition_id": model["model_definition_id"],
                        "weight": 1.0,
                    }
                ],
                "name": endpoint_name,
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "usage_tracking": False,
            },
        )
        trace_hex = hashlib.sha256(f"t23:{seed}".encode()).hexdigest()[:32]
        now_ns = 1_750_000_000_000_000_000
        otlp = {
            "resourceSpans": [
                {
                    "resource": {"attributes": []},
                    "scopeSpans": [
                        {
                            "scope": {"name": "t23-smoke"},
                            "spans": [
                                {
                                    "attributes": [
                                        {
                                            "key": "mlflow.spanInputs",
                                            "value": {
                                                "stringValue": json.dumps({
                                                    "question": "two plus two"
                                                })
                                            },
                                        },
                                        {
                                            "key": "mlflow.spanOutputs",
                                            "value": {"stringValue": json.dumps("two words")},
                                        },
                                        {
                                            "key": "mlflow.spanType",
                                            "value": {"stringValue": "CHAIN"},
                                        },
                                    ],
                                    "endTimeUnixNano": str(now_ns + 1_000_000),
                                    "name": "root",
                                    "spanId": trace_hex[:16],
                                    "startTimeUnixNano": str(now_ns),
                                    "status": {"code": 1},
                                    "traceId": trace_hex,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        response = session.post(
            base_url + "/v1/traces",
            headers={"content-type": "application/json", "x-mlflow-experiment-id": experiment_id},
            json=otlp,
            timeout=30,
        )
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"OTLP setup failed: {response.status_code}: {response.text[:500]}")
        return {
            "dataset_id": dataset["dataset"]["dataset_id"],
            "endpoint_name": endpoint_name,
            "experiment_id": experiment_id,
            "trace_id": f"tr-{trace_hex}",
        }
    finally:
        session.close()


async def run_seeded_searches(
    client: AsyncBenchClient,
    experiment_id: str,
    seed: int,
    count: int,
    concurrency: int,
) -> None:
    rng = random.Random(seed)
    offsets = []
    offset = 0.0
    for _ in range(count):
        offset += rng.choice((0.0, 0.0005, 0.001, 0.0015))
        offsets.append(offset)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(sequence: int, scheduled: float) -> None:
        await asyncio.sleep(scheduled)
        async with semaphore:
            await client.request(
                "datasets_search",
                "POST",
                "/api/3.0/mlflow/datasets/search",
                json={"experiment_ids": [experiment_id], "max_results": 10},
                sequence=sequence,
            )

    await asyncio.gather(*(one(index, scheduled) for index, scheduled in enumerate(offsets)))


async def poll_job(
    client: AsyncBenchClient,
    job_id: str,
    submitted: float,
    sequence: int,
) -> dict[str, Any]:
    first_running: float | None = None
    polls = 0
    while time.perf_counter() - submitted < 60:
        polls += 1
        status, body, _ = await client.request(
            "job_poll",
            "GET",
            f"/ajax-api/3.0/mlflow/jobs/{job_id}",
            sequence=sequence + polls,
        )
        if status != 200 or not isinstance(body, dict):
            await asyncio.sleep(0.05)
            continue
        state = str(body.get("status", "")).upper()
        now = time.perf_counter()
        if state == "RUNNING" and first_running is None:
            first_running = now
        if state in TERMINAL_STATES:
            terminal = now
            queue_end = first_running or terminal
            return {
                "execution_seconds": max(0.0, terminal - queue_end),
                "job_id": job_id,
                "polls": polls,
                "queue_wait_seconds": max(0.0, queue_end - submitted),
                "result": body.get("result"),
                "status": state,
                "status_details": body.get("status_details"),
                "submit_to_terminal_seconds": terminal - submitted,
            }
        await asyncio.sleep(0.25)
    raise RuntimeError(f"job {job_id} did not reach a terminal state")


def instructions_scorer(endpoint_name: str) -> str:
    value = json.loads(
        (
            RUST_ROOT
            / "crates"
            / "mlflow-genai"
            / "tests"
            / "fixtures"
            / "instructions_judge_scorer.json"
        ).read_text()
    )
    value["instructions_judge_pydantic_data"]["model"] = f"gateway:/{endpoint_name}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


async def execute_smoke_workload(
    base_url: str,
    setup: dict[str, str],
    seed: int,
    concurrency: int,
) -> tuple[MetricsCollector, list[dict[str, Any]]]:
    collector = MetricsCollector()
    client = AsyncBenchClient(base_url, concurrency, collector)
    jobs: list[dict[str, Any]] = []
    try:
        for _ in range(5):
            await client.request(
                "datasets_search",
                "POST",
                "/api/3.0/mlflow/datasets/search",
                measured=False,
                json={"experiment_ids": [setup["experiment_id"]], "max_results": 10},
            )
        await client.request(
            "gateway_chat",
            "POST",
            f"/gateway/{setup['endpoint_name']}/mlflow/invocations",
            measured=False,
            json={
                "messages": [{"content": "warmup", "role": "user"}],
                "stream": False,
            },
        )
        collector.started = time.perf_counter()
        await run_seeded_searches(client, setup["experiment_id"], seed, 90, concurrency)
        for index in range(6):
            await client.request(
                "gateway_chat_sse",
                "POST",
                f"/gateway/{setup['endpoint_name']}/mlflow/invocations",
                json={
                    "messages": [
                        {"content": f"seed-{seed}-gateway-stream-{index}", "role": "user"}
                    ],
                    "stream": True,
                },
                sequence=90 + index,
                sse=True,
            )
        submitted = time.perf_counter()
        status, submission, _ = await client.request(
            "scorer_job_submit",
            "POST",
            "/ajax-api/3.0/mlflow/scorer/invoke",
            json={
                "experiment_id": setup["experiment_id"],
                "log_assessments": False,
                "serialized_scorer": instructions_scorer(setup["endpoint_name"]),
                "trace_ids": [setup["trace_id"]],
            },
            sequence=96,
        )
        if status != 200 or not isinstance(submission, dict):
            raise RuntimeError(f"job submission failed: {submission}")
        submitted_jobs = submission.get("jobs", [])
        if not submitted_jobs:
            raise RuntimeError(f"job submission returned no jobs: {submission}")
        for index, job in enumerate(submitted_jobs):
            jobs.append(await poll_job(client, job["job_id"], submitted, 1000 + index * 100))
        status, message, _ = await client.request(
            "assistant_message",
            "POST",
            f"{ASSISTANT_PREFIX}/message",
            json={"context": {"phase": 23}, "message": f"assistant smoke seed {seed}"},
            sequence=97,
        )
        if status != 200 or not isinstance(message, dict):
            raise RuntimeError(f"assistant message failed: {message}")
        await client.request(
            "assistant_stream",
            "GET",
            f"{ASSISTANT_PREFIX}/sessions/{message['session_id']}/stream",
            sequence=98,
            sse=True,
        )
        collector.close()
        return collector, jobs
    finally:
        await client.close()


def build_equivalence_samples(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    candidates = [record for record in records if record["endpoint"] != "job_poll"]
    ranked = sorted(
        candidates,
        key=lambda record: hashlib.sha256(f"{seed}:{record['sequence']}".encode()).digest(),
    )
    selected = ranked[:16]
    for endpoint in sorted({record["endpoint"] for record in candidates}):
        record = next(item for item in candidates if item["endpoint"] == endpoint)
        if not any(item["endpoint"] == endpoint for item in selected):
            selected.append(record)
    selected.sort(key=lambda record: record["sequence"])
    return [
        {
            "endpoint": record["endpoint"],
            "method": record["method"],
            "path": record["path"],
            "response": record["response"],
            "sequence": record["sequence"],
            "sse_frames": record["sse"]["frames"] if record["sse"] else None,
            "status": record["status"],
        }
        for record in selected
    ]


def validate_raw_metrics(value: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_FILE.read_text())
    Draft202012Validator(schema).validate(value)


def write_raw_metrics(value: dict[str, Any], output: Path) -> None:
    validate_raw_metrics(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def route_latency(value: str) -> tuple[str, float]:
    try:
        route, milliseconds = value.split("=", 1)
        latency = float(milliseconds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected ROUTE=MILLISECONDS") from exc
    if route not in {"chat_completions", "embeddings", "anthropic_messages"} or latency < 0:
        raise argparse.ArgumentTypeError(
            "route must be chat_completions, embeddings, or anthropic_messages and latency >= 0"
        )
    return route, latency


def run_target(
    target: str,
    seed: int,
    concurrency: int,
    output: Path,
    provider_url: str,
    provider: Any,
    stub_path: Path,
) -> dict[str, Any]:
    recreate_database()
    with tempfile.TemporaryDirectory(prefix=f"mlflow-t23-{target}-") as temporary:
        workdir = Path(temporary)
        handle = launch_server(
            target,
            workdir,
            provider_url,
            f"seed-{seed}/{target}-{int(time.time_ns())}",
            stub_path,
        )
        monitor = ResourceMonitor(handle.process.pid, postgres_sample)
        provider_start = len(provider.state.observations)
        started_at = dt.datetime.now(dt.timezone.utc)
        monitor.start()
        try:
            time.sleep(1.05)
            setup = setup_target(handle.url, provider_url, seed)
            collector, jobs = asyncio.run(
                execute_smoke_workload(handle.url, setup, seed, concurrency)
            )
        finally:
            monitor.close()
            stop_server(handle)
        endpoints, overall = collector.summary()
        raw_records = collector.raw_records()
        observations = provider.state.observations[provider_start:]
        value = {
            "schema_version": "1.0.0",
            "run": {
                "concurrency": concurrency,
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "seed": seed,
                "started_at": started_at.isoformat(),
                "target": target,
                "warmup_requests": 6,
                "workload": "smoke",
            },
            "summary": {"endpoints": endpoints, "overall": overall},
            "requests": raw_records,
            "jobs": jobs,
            "resources": {
                "method": "1 s /proc whole-process-tree VmRSS and stat utime/stime sum",
                "samples": monitor.samples,
            },
            "db_pool": {
                "method": "pg_stat_activity (server-side pool occupancy proxy)",
                "samples": monitor.pool_samples,
            },
            "provider": {
                "observations": [observation.__dict__ for observation in observations],
                "route_latency_ms": provider.state.route_latency_ms,
                "seed": seed,
                "url": provider_url,
            },
            "equivalence": {
                "jobs": jobs,
                "sample_seed": seed,
                "samples": build_equivalence_samples(raw_records, seed),
            },
        }
        write_raw_metrics(value, output)
        return value


def assert_provider_stability(provider_url: str, seed: int) -> None:
    payloads = [
        {
            "messages": [{"content": "determinism", "role": "user"}],
            "model": "genai-bench-model",
            "stream": False,
        },
        {
            "messages": [{"content": "determinism", "role": "user"}],
            "model": "genai-bench-model",
            "stream": True,
        },
        {"input": ["alpha", "beta"], "model": "genai-bench-embedding"},
    ]
    paths = ["/v1/chat/completions", "/v1/chat/completions", "/v1/embeddings"]
    with requests.Session() as session:
        for path, payload in zip(paths, payloads):
            first = session.post(provider_url + path, json=payload, timeout=5).content
            second = session.post(provider_url + path, json=payload, timeout=5).content
            if first != second:
                raise AssertionError(f"mock provider response was not byte-stable for {path}")
            route = "embeddings" if path.endswith("embeddings") else "chat_completions"
            expected_hash = hashlib.sha256(
                f"{seed}:{route}:".encode() + canonical_request(json.dumps(payload).encode())
            ).hexdigest()
            if expected_hash[:8].encode() not in first and not path.endswith("embeddings"):
                raise AssertionError("mock provider response did not derive from seed/request hash")


def smoke(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    stub_path, cleanup_path = install_claude_stub()
    try:
        run_command(compose_args("up", "-d", "--wait", "postgres", "minio"))
        run_command(compose_args("run", "--rm", "minio-init"))
        run_command(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "mlflow-server",
                "-p",
                "mlflow-genai-worker",
            ],
            cwd=RUST_ROOT,
        )
        with provider_server(args.seed, dict(args.provider_latency)) as provider:
            provider_url = f"http://127.0.0.1:{provider.server_port}"
            assert_provider_stability(provider_url, args.seed)
            results = {}
            for target in args.targets:
                path = output_dir / f"smoke-{target}.json"
                print(f"[{target}] running T23.1 smoke cell", flush=True)
                results[target] = run_target(
                    target,
                    args.seed,
                    args.concurrency,
                    path,
                    provider_url,
                    provider,
                    stub_path,
                )
                errors = results[target]["summary"]["overall"]["errors"]
                if errors:
                    raise RuntimeError(f"{target} smoke recorded {errors} errors")
                failed_jobs = [
                    job for job in results[target]["jobs"] if job["status"] != "SUCCEEDED"
                ]
                if failed_jobs:
                    raise RuntimeError(f"{target} smoke jobs did not succeed: {failed_jobs}")
            if set(args.targets) == {"python", "rust"}:
                differences = compare_runs(results["python"], results["rust"])
                if differences:
                    raise RuntimeError("equivalence failed:\n" + "\n".join(differences))
                print("equivalence: PASS", flush=True)
            print(f"raw metrics: {output_dir}", flush=True)
            return 0
    finally:
        run_command(compose_args("down", "-v", "--remove-orphans"), check=False)
        shutil.rmtree(cleanup_path, ignore_errors=True)


def validate_command(args: argparse.Namespace) -> int:
    for path in args.files:
        validate_raw_metrics(json.loads(path.read_text()))
        print(f"valid: {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="run the ~100-request cell")
    smoke_parser.add_argument("--concurrency", type=int, default=8)
    smoke_parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    smoke_parser.add_argument("--seed", type=int, default=2301)
    smoke_parser.add_argument(
        "--provider-latency",
        action="append",
        default=[],
        metavar="ROUTE=MILLISECONDS",
        type=route_latency,
    )
    smoke_parser.add_argument(
        "--targets", nargs="+", choices=("python", "rust"), default=["python", "rust"]
    )
    smoke_parser.set_defaults(func=smoke)
    validate_parser = subparsers.add_parser("validate", help="validate raw result JSON")
    validate_parser.add_argument("files", nargs="+", type=Path)
    validate_parser.set_defaults(func=validate_command)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
