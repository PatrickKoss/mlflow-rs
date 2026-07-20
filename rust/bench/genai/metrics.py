"""Client-side latency/SSE metrics and Linux process-tree resource sampling."""

from __future__ import annotations

import asyncio
import collections
import datetime as dt
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100)
    return ordered[max(0, min(index, len(ordered) - 1))]


@dataclass
class SseMetrics:
    frame_count: int
    time_to_first_event_ms: float | None
    inter_frame_gap_ms: list[float]
    frame_elapsed_ms: list[float]
    frames: list[str]


@dataclass
class RequestRecord:
    sequence: int
    endpoint: str
    method: str
    path: str
    status: int | None
    latency_ms: float
    error: str | None
    response: Any
    request_bytes: int | None = None
    sse: SseMetrics | None = None


@dataclass
class MetricsCollector:
    records: list[RequestRecord] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None

    def add(self, record: RequestRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.finished = time.perf_counter()

    def summary(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        duration = max(1e-9, (self.finished or time.perf_counter()) - self.started)
        grouped: dict[str, list[RequestRecord]] = collections.defaultdict(list)
        for record in self.records:
            grouped[record.endpoint].append(record)
        endpoints = {}
        for name, records in sorted(grouped.items()):
            latencies = [record.latency_ms for record in records]
            errors = sum(record.error is not None for record in records)
            statuses = collections.Counter(
                "exception" if record.status is None else str(record.status) for record in records
            )
            ttfe = [
                record.sse.time_to_first_event_ms
                for record in records
                if record.sse and record.sse.time_to_first_event_ms is not None
            ]
            gaps = [
                gap for record in records if record.sse for gap in record.sse.inter_frame_gap_ms
            ]
            frames = sum(record.sse.frame_count for record in records if record.sse)
            endpoints[name] = {
                "requests": len(records),
                "errors": errors,
                "error_rate": errors / len(records),
                "rps": len(records) / duration,
                "latency_ms": {
                    "p50": percentile(latencies, 50),
                    "p95": percentile(latencies, 95),
                    "p99": percentile(latencies, 99),
                    "max": max(latencies),
                },
                "statuses": dict(sorted(statuses.items())),
                "sse": {
                    "streams": sum(record.sse is not None for record in records),
                    "frames": frames,
                    "frames_per_second": frames / duration,
                    "completion_errors": sum(
                        record.sse is not None and record.error is not None for record in records
                    ),
                    "completion_error_rate": (
                        sum(
                            record.sse is not None and record.error is not None
                            for record in records
                        )
                        / sum(record.sse is not None for record in records)
                        if any(record.sse is not None for record in records)
                        else 0.0
                    ),
                    "time_to_first_event_ms": {
                        "p50": percentile(ttfe, 50),
                        "p95": percentile(ttfe, 95),
                        "p99": percentile(ttfe, 99),
                        "max": max(ttfe) if ttfe else None,
                    },
                    "inter_frame_gap_ms": {
                        "p50": percentile(gaps, 50),
                        "p95": percentile(gaps, 95),
                        "p99": percentile(gaps, 99),
                        "max": max(gaps) if gaps else None,
                    },
                },
            }
        errors = sum(record.error is not None for record in self.records)
        latencies = [record.latency_ms for record in self.records]
        overall = {
            "duration_seconds": duration,
            "requests": len(self.records),
            "errors": errors,
            "error_rate": errors / len(self.records) if self.records else 0.0,
            "rps": len(self.records) / duration,
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": max(latencies) if latencies else None,
            },
        }
        return endpoints, overall

    def raw_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


class AsyncBenchClient:
    def __init__(
        self,
        base_url: str,
        concurrency: int,
        collector: MetricsCollector,
        *,
        timeout_seconds: float = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.collector = collector
        self.connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        )
        self._sequence = 0

    async def close(self) -> None:
        await self.session.close()

    async def request(
        self,
        endpoint: str,
        method: str,
        path: str,
        *,
        measured: bool = True,
        sse: bool = False,
        expected: set[int] | None = None,
        capture_response: bool = True,
        sequence: int | None = None,
        **kwargs: Any,
    ) -> tuple[int | None, Any, SseMetrics | None]:
        if sequence is None:
            sequence = self._sequence
            self._sequence += 1
        started = time.perf_counter()
        status: int | None = None
        error: str | None = None
        response_value: Any = None
        sse_metrics: SseMetrics | None = None
        response_bytes: bytes | None = None
        request_bytes = None
        if "json" in kwargs:
            request_bytes = len(
                json.dumps(kwargs["json"], sort_keys=True, separators=(",", ":")).encode()
            )
        elif isinstance(kwargs.get("data"), (bytes, str)):
            data = kwargs["data"]
            request_bytes = len(data.encode() if isinstance(data, str) else data)
        accepted = expected or set(range(200, 300))
        try:
            async with self.session.request(method, self.base_url + path, **kwargs) as response:
                status = response.status
                if sse:
                    response_value, sse_metrics = await self._read_sse(response, started)
                else:
                    raw = await response.read()
                    response_bytes = raw
                    try:
                        response_value = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        response_value = raw.decode(errors="replace")
                if status not in accepted:
                    error = f"HTTP {status}: {str(response_value)[:300]}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        if measured:
            recorded_response = response_value
            if not capture_response and response_bytes is not None:
                recorded_response = {
                    "body_bytes": len(response_bytes),
                    "body_sha256": hashlib.sha256(response_bytes).hexdigest(),
                }
            self.collector.add(
                RequestRecord(
                    sequence=sequence,
                    endpoint=endpoint,
                    method=method,
                    path=path,
                    status=status,
                    latency_ms=latency_ms,
                    error=error,
                    response=recorded_response,
                    request_bytes=request_bytes,
                    sse=sse_metrics,
                )
            )
        return status, response_value, sse_metrics

    async def _read_sse(
        self, response: aiohttp.ClientResponse, started: float
    ) -> tuple[list[str], SseMetrics]:
        buffer = b""
        frames: list[str] = []
        frame_times: list[float] = []
        async for chunk in response.content.iter_any():
            buffer += chunk.replace(b"\r\n", b"\n")
            while b"\n\n" in buffer:
                raw, buffer = buffer.split(b"\n\n", 1)
                if not raw or all(line.startswith(b":") for line in raw.splitlines()):
                    continue
                frames.append(raw.decode(errors="replace"))
                frame_times.append(time.perf_counter())
        if buffer.strip() and not all(line.startswith(b":") for line in buffer.splitlines()):
            frames.append(buffer.decode(errors="replace"))
            frame_times.append(time.perf_counter())
        ttfe = (frame_times[0] - started) * 1000 if frame_times else None
        gaps = [(right - left) * 1000 for left, right in zip(frame_times, frame_times[1:])]
        elapsed = [(frame_time - started) * 1000 for frame_time in frame_times]
        return frames, SseMetrics(len(frames), ttfe, gaps, elapsed, frames)


def process_tree(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            rest = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(rest[1])
        except (OSError, IndexError, ValueError):
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


def process_tree_resources(root_pid: int) -> dict[str, Any]:
    total_kib = 0
    thread_count = 0
    utime_ticks = 0
    stime_ticks = 0
    pids = process_tree(root_pid)
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                elif line.startswith("Threads:"):
                    thread_count += int(line.split()[1])
            rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            utime_ticks += int(rest[11])
            stime_ticks += int(rest[12])
        except (OSError, IndexError, ValueError):
            continue
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    return {
        "rss_bytes": total_kib * 1024,
        "process_count": len(pids),
        "thread_count": thread_count,
        "utime_seconds": utime_ticks / ticks_per_second,
        "stime_seconds": stime_ticks / ticks_per_second,
    }


class ResourceMonitor:
    def __init__(
        self,
        root_pid: int,
        pool_sampler: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.root_pid = root_pid
        self.pool_sampler = pool_sampler
        self.samples: list[dict[str, Any]] = []
        self.pool_samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="genai-resource-monitor", daemon=True)
        self.started = time.monotonic()

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        now = time.monotonic()
        elapsed = now - self.started
        if not self.samples or elapsed - self.samples[-1]["elapsed_seconds"] > 0.01:
            self.samples.append({
                "elapsed_seconds": elapsed,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                **process_tree_resources(self.root_pid),
            })
            if self.pool_sampler and (pool := self.pool_sampler()):
                self.pool_samples.append({"elapsed_seconds": elapsed, **pool})

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now >= next_sample:
                self.samples.append({
                    "elapsed_seconds": now - self.started,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    **process_tree_resources(self.root_pid),
                })
                if self.pool_sampler and (pool := self.pool_sampler()):
                    self.pool_samples.append({"elapsed_seconds": now - self.started, **pool})
                next_sample += 1
            self.stop_event.wait(max(0.01, min(0.1, next_sample - time.monotonic())))


def finite_number(value: float | None) -> float | None:
    return value if value is None or math.isfinite(value) else None
