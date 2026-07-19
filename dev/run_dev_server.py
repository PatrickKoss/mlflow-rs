"""Launch the MLflow dev backend and the React dev server for local development.

Cleans up child process groups on exit/SIGINT/SIGTERM so we don't leave zombies.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dev_stubs

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_DIR = REPO_ROOT / "mlflow" / "server" / "js"


def find_free_port(preferred: int, avoid: frozenset[int] = frozenset()) -> int:
    for port in range(preferred, preferred + 100):
        if port in avoid:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"No free port in [{preferred}, {preferred + 100})")


def cleanup(children: list[subprocess.Popen[bytes]], tmp_paths: list[Path]) -> None:
    for proc in children:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for proc in children:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for path in tmp_paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def on_signal(signum: int, _frame: object) -> None:
    sys.exit(128 + signum)


def start_backend(port: int) -> tuple[subprocess.Popen[bytes], list[Path]]:
    backend_args: list[str] = []
    tmp_paths: list[Path] = []
    server_type = os.environ.get("MLFLOW_SERVER_TYPE", "python").lower()
    if tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        backend_args += ["--backend-store-uri", tracking_uri, "--default-artifact-root", "mlruns"]
    elif backend_uri := os.environ.get("MLFLOW_BACKEND_STORE_URI"):
        backend_args += ["--backend-store-uri", backend_uri, "--default-artifact-root", "mlruns"]
    else:
        db_fd, db_path_str = tempfile.mkstemp(prefix="mlflow-dev-", suffix=".db")
        os.close(db_fd)
        db_path = Path(db_path_str)
        if server_type == "rust":
            fixture_db = (
                REPO_ROOT
                / "rust"
                / "crates"
                / "mlflow-server"
                / "tests"
                / "fixtures"
                / "tracking.db"
            )
            shutil.copyfile(
                fixture_db,
                db_path,
            )
        artifacts_path = Path(tempfile.mkdtemp(prefix="mlflow-dev-artifacts-"))
        tmp_paths += [db_path, artifacts_path]
        backend_args += [
            "--backend-store-uri",
            f"sqlite:///{db_path}",
            "--default-artifact-root",
            str(artifacts_path),
        ]
        print(f"Using tmp SQLite store: {db_path} (artifacts: {artifacts_path})")
    if registry_uri := os.environ.get("MLFLOW_REGISTRY_URI"):
        backend_args += ["--registry-store-uri", registry_uri]

    if server_type == "rust":
        rust_manifest = REPO_ROOT / "rust" / "Cargo.toml"
        cmd = [
            "cargo",
            "run",
            "--manifest-path",
            str(rust_manifest),
            "--bin",
            "mlflow-server",
            "--",
            *backend_args,
            "--port",
            str(port),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            *backend_args,
            "--dev",
            "--port",
            str(port),
        ]
    print(f"Running tracking server: {shlex.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
    wait_ready(f"http://localhost:{port}/health", "tracking server")
    return proc, tmp_paths


def start_frontend(backend_port: int, frontend_port: int) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        ["yarn", "start"],
        cwd=JS_DIR,
        env={
            **os.environ,
            "PORT": str(frontend_port),
            "MLFLOW_PROXY": f"http://localhost:{backend_port}",
            "MLFLOW_DEV_PROXY_MODE": "1",
            "BROWSER": "none",
        },
        start_new_session=True,
    )
    wait_ready(f"http://localhost:{frontend_port}/", "React dev server", timeout=180)
    return proc


def wait_ready(url: str, label: str, timeout: float = 60.0) -> None:
    print(f"Waiting for {label} to be ready...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    print(f"{label} is ready")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"Failed to launch {label} (gave up after {timeout:.0f}s)")


def smoke_assistant_backend(port: int) -> None:
    """Assert the credential-free Assistant HTTP/SSE surface used by the UI."""
    prefix = f"http://localhost:{port}/ajax-api/3.0/mlflow/assistant"

    def request(path: str, method: str = "GET", body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        return urllib.request.urlopen(
            urllib.request.Request(prefix + path, data=data, headers=headers, method=method),
            timeout=10,
        )

    with request("/providers/claude_code/health") as response:
        assert response.status == 200
        assert json.load(response) == {"status": "ok"}
    with request(
        "/config",
        method="PUT",
        body={"providers": {"claude_code": {"selected": True}}},
    ) as response:
        assert response.status == 200
        assert json.load(response)["providers"]["claude_code"]["selected"] is True
    with request("/message", method="POST", body={"message": "dev stub smoke"}) as response:
        assert response.status == 200
        sent = json.load(response)
    session_id = sent["session_id"]
    assert sent["stream_url"].endswith(f"/sessions/{session_id}/stream")
    with request(f"/sessions/{session_id}/stream") as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "text/event-stream"
        stream = response.read()
    for frame in (b"event: message\n", b"event: stream_event\n", b"event: done\n"):
        assert frame in stream
    assert b"synthetic reply from the MLflow dev stub Claude CLI" in stream
    print("Assistant Rust dev-stub smoke passed (health/config/message/SSE)")


def main() -> None:
    # Line-buffer prints so progress shows up live when stdout is redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stub-providers",
        default="",
        help=(
            "Comma-separated credential-free stubs to install before launch so "
            "provider-gated UI renders without real keys (for CI review / local dev). "
            f"Available: {', '.join(dev_stubs.AVAILABLE_STUBS)}."
        ),
    )
    parser.add_argument(
        "--assistant-smoke",
        action="store_true",
        help="Probe the backend Assistant dev-stub surface and exit without launching React.",
    )
    args = parser.parse_args()
    stub_names = [s.strip() for s in args.stub_providers.split(",") if s.strip()]

    if not args.assistant_smoke:
        subprocess.check_call(["yarn", "install"], cwd=JS_DIR)

    backend_port = find_free_port(5000)
    frontend_port = find_free_port(3000, avoid=frozenset({backend_port}))
    print(f"Backend:  http://localhost:{backend_port}")
    print(f"Frontend: http://localhost:{frontend_port} (with hot reload)")

    children: list[subprocess.Popen[bytes]] = []
    tmp_paths: list[Path] = []

    if args.assistant_smoke:
        original_home = Path.home()
        os.environ.setdefault("CARGO_HOME", str(original_home / ".cargo"))
        os.environ.setdefault("RUSTUP_HOME", str(original_home / ".rustup"))
        smoke_home = Path(tempfile.mkdtemp(prefix="mlflow-assistant-smoke-home-"))
        os.environ["HOME"] = str(smoke_home)
        tmp_paths.append(smoke_home)

    atexit.register(cleanup, children, tmp_paths)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    if stub_names:
        # Install before the backend launches so PATH changes propagate to it, and
        # register temp dirs with the same cleanup as the other children.
        stubs = dev_stubs.install_stubs(stub_names)
        dev_stubs.apply_to_environ(stubs)
        tmp_paths.extend(stubs.cleanup_paths)
        for message in stubs.messages:
            print(message)
        if os.environ.get("MLFLOW_SERVER_TYPE", "python").lower() == "rust":
            os.environ["MLFLOW_ASSISTANT_DEV_STUB_PROVIDERS"] = ",".join(stub_names)

    backend_proc, backend_tmp = start_backend(backend_port)
    children.append(backend_proc)
    tmp_paths.extend(backend_tmp)
    if args.assistant_smoke:
        smoke_assistant_backend(backend_port)
        return
    children.append(start_frontend(backend_port, frontend_port))

    # Block until any child exits; atexit reaps the rest.
    while all(proc.poll() is None for proc in children):
        time.sleep(1)
    exited = next(p for p in children if p.poll() is not None)
    print(f"Child process (pid {exited.pid}) exited with code {exited.returncode}")
    sys.exit(exited.returncode or 0)


if __name__ == "__main__":
    main()
