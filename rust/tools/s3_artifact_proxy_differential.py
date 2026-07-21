#!/usr/bin/env python3
"""Diff Python and Rust S3 artifact-proxy wire behavior against MinIO.

The script owns both server processes and always stops their process groups. It
expects an existing MinIO bucket and a built debug Rust server. Result bodies
are compared semantically; volatile upload IDs, SigV4 timestamps/signatures,
and destination prefixes are normalized while URL/query/header shapes remain
part of the comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "rust/crates/mlflow-server/tests/fixtures/tracking.db"
DEFAULT_RUST_BIN = ROOT / "rust/target/debug/mlflow-server"
DEFAULT_OUTPUT = ROOT / "rust/tools/results/t22_0_s3_differential.json"
SELECTED_HEADERS = ("content-type", "content-disposition")


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(base: str, method: str, path: str, body: bytes | None = None) -> dict:
    request = urllib.request.Request(base + path, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        response = error
    raw = response.read()
    return {
        "status": response.status,
        "reason": response.reason,
        "headers": {
            name: response.headers[name]
            for name in SELECTED_HEADERS
            if response.headers.get(name) is not None
        },
        "body": raw,
    }


def _wait(base: str, process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}; log:\n{log.read_text()}")
        try:
            if _request(base, "GET", "/health")["status"] == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server did not become healthy; log:\n{log.read_text()}")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _normalize_url(url: str, prefix: str) -> dict:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    volatile = {"X-Amz-Date", "X-Amz-Signature", "uploadId"}
    normalized_query = {key: "<volatile>" if key in volatile else value for key, value in query}
    path = parsed.path.replace(prefix, "/<destination>", 1)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": path,
        "query": dict(sorted(normalized_query.items())),
    }


def _normalize_response(response: dict, prefix: str, kind: str) -> dict:
    body = response["body"]
    if kind == "bytes":
        normalized_body: object = body.decode()
    elif not body:
        normalized_body = {} if kind == "json" else ""
    else:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"expected JSON for {kind}, got HTTP {response['status']}: {body[:500]!r}"
            ) from error
        if kind == "create":
            normalized_body = {
                "upload_id": "<upload-id>",
                "credentials": [
                    {
                        "part_number": item["part_number"],
                        "headers": item.get("headers", {}),
                        "url": _normalize_url(item["url"], prefix),
                    }
                    for item in parsed["credentials"]
                ],
            }
        else:
            normalized_body = parsed
    return {
        "status": response["status"],
        "reason": response["reason"],
        "headers": response["headers"],
        "body": normalized_body,
    }


def _sequence(base: str, prefix: str) -> dict:
    records: dict[str, dict] = {}

    response = _request(
        base,
        "PUT",
        "/api/2.0/mlflow-artifacts/artifacts/dir/plain.txt",
        b"artifact-proxy-differential",
    )
    records["put"] = _normalize_response(response, prefix, "json")
    response = _request(base, "GET", "/api/2.0/mlflow-artifacts/artifacts?path=dir")
    records["list"] = _normalize_response(response, prefix, "json")
    response = _request(base, "GET", "/api/2.0/mlflow-artifacts/artifacts/dir/plain.txt")
    records["get"] = _normalize_response(response, prefix, "bytes")
    response = _request(base, "GET", "/api/2.0/mlflow-artifacts/presigned/dir/plain.txt")
    records["presigned"] = _normalize_response(response, prefix, "json")
    presigned = json.loads(response["body"])
    records["presigned"]["body"]["url"] = _normalize_url(presigned["url"], prefix)
    direct = urllib.request.urlopen(presigned["url"], timeout=30)
    records["presigned_get"] = {
        "status": direct.status,
        "reason": direct.reason,
        "body": direct.read().decode(),
    }

    response = _request(
        base,
        "POST",
        "/api/2.0/mlflow-artifacts/mpu/create/differential",
        json.dumps({"path": "joined.bin", "num_parts": 2}).encode(),
    )
    records["mpu_create"] = _normalize_response(response, prefix, "create")
    created = json.loads(response["body"])
    parts = []
    payloads = (b"a" * (5 * 1024 * 1024), b"tail")
    for credential, payload in zip(created["credentials"], payloads):
        part = urllib.request.Request(credential["url"], data=payload, method="PUT")
        uploaded = urllib.request.urlopen(part, timeout=30)
        parts.append({
            "part_number": credential["part_number"],
            "etag": uploaded.headers["ETag"],
            "url": credential["url"],
        })
        uploaded.read()
    response = _request(
        base,
        "POST",
        "/api/2.0/mlflow-artifacts/mpu/complete/differential",
        json.dumps({
            "path": "joined.bin",
            "upload_id": created["upload_id"],
            "parts": parts,
        }).encode(),
    )
    records["mpu_complete"] = _normalize_response(response, prefix, "json")
    response = _request(base, "GET", "/api/2.0/mlflow-artifacts/artifacts/differential/joined.bin")
    records["mpu_get"] = {
        **_normalize_response(response, prefix, "bytes"),
        "body": {"length": len(response["body"]), "tail": response["body"][-4:].decode()},
    }

    response = _request(
        base,
        "POST",
        "/api/2.0/mlflow-artifacts/mpu/create/differential",
        json.dumps({"path": "aborted.bin", "num_parts": 1}).encode(),
    )
    aborted = json.loads(response["body"])
    response = _request(
        base,
        "POST",
        "/api/2.0/mlflow-artifacts/mpu/abort/differential",
        json.dumps({"path": "aborted.bin", "upload_id": aborted["upload_id"]}).encode(),
    )
    records["mpu_abort"] = _normalize_response(response, prefix, "json")
    response = _request(
        base,
        "DELETE",
        "/api/2.0/mlflow-artifacts/artifacts/dir/plain.txt",
    )
    records["delete"] = _normalize_response(response, prefix, "json")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MLFLOW_TEST_S3_ENDPOINT", "http://127.0.0.1:59090"),
    )
    parser.add_argument("--bucket", default=os.environ.get("MLFLOW_TEST_S3_BUCKET", "mlflow-soak"))
    parser.add_argument("--rust-bin", type=Path, default=DEFAULT_RUST_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-frozen", action="store_true")
    args = parser.parse_args()
    if not args.rust_bin.exists():
        raise SystemExit(f"Rust server binary is missing: {args.rust_bin}")

    run_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="mlflow-s3-differential-") as temporary:
        temp = Path(temporary)
        processes: list[subprocess.Popen] = []
        logs = []
        try:
            targets = {}
            for target in ("python", "rust"):
                port = _port()
                db = temp / f"{target}.db"
                shutil.copyfile(FIXTURE, db)
                log = temp / f"{target}.log"
                logs.append(log)
                prefix = f"/t22-0/differential/{run_id}/{target}"
                destination = f"s3://{args.bucket}{prefix}"
                if target == "python":
                    command = ["uv", "run"]
                    if not args.no_frozen:
                        command.append("--frozen")
                    command += ["--extra", "gateway"]
                    command += [
                        "mlflow",
                        "server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--backend-store-uri",
                        f"sqlite:///{db}",
                        "--serve-artifacts",
                        "--artifacts-destination",
                        destination,
                    ]
                else:
                    command = [
                        str(args.rust_bin),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--backend-store-uri",
                        f"sqlite:///{db}",
                        "--serve-artifacts",
                        "--artifacts-destination",
                        destination,
                    ]
                env = os.environ | {
                    "MLFLOW_S3_ENDPOINT_URL": args.endpoint,
                    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
                    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                    "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
                    "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                    "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
                }
                stream = log.open("wb")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                stream.close()
                processes.append(process)
                base = f"http://127.0.0.1:{port}"
                _wait(base, process, log)
                targets[target] = {"base": base, "prefix": prefix}

            results = {
                target: _sequence(values["base"], values["prefix"])
                for target, values in targets.items()
            }
            equal = results["python"] == results["rust"]
            evidence = {
                "task": "T22.0 S3 artifact proxy differential",
                "endpoint": args.endpoint,
                "bucket": args.bucket,
                "cases": len(results["python"]),
                "known_delta_allowlist": ["HTTP reason-phrase casing (none encountered)"],
                "non_allowlisted_diffs": 0 if equal else 1,
                "status": "PASS" if equal else "FAIL",
                "python": results["python"],
                "rust": results["rust"],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
            summary = {key: evidence[key] for key in ("status", "cases", "non_allowlisted_diffs")}
            print(json.dumps(summary))
            if not equal:
                raise SystemExit(1)
        except Exception:
            for log in logs:
                print(f"--- {log.name} ---\n{log.read_text(errors='replace')}")
            raise
        finally:
            for process in reversed(processes):
                _stop(process)


if __name__ == "__main__":
    main()
