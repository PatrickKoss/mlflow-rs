# Phase 23 GenAI benchmark harness

This directory is the reusable T23 performance harness. It does not assert the
full API contract; the compliance recorder owns that. It proves that both
targets executed an equivalent seeded workload, then emits latency, error,
streaming, job, RSS, CPU, and database-pool observations for later matrix
tasks.

Run the CI-sized cell (about 100 measured requests per target) with:

```bash
rust/bench/genai/smoke.sh
```

The wrapper enables MLflow's already-pinned `db` extra for the PostgreSQL
driver; it does not install an unpinned benchmark dependency.

The runner starts Postgres 16 and MinIO, creates a fresh database and unique S3
prefix for each target, builds the release Rust server and native worker, then
runs Python (four uvicorn workers plus Huey job runtime) and Rust sequentially.
All launched servers use their own process group and are terminated on success
or failure. Compose volumes and the staged fake `claude` shim are removed.

## Deterministic dependencies

`mock_provider.py` runs a loopback `ThreadingHTTPServer`. It implements OpenAI
chat completions (JSON and SSE), embeddings, model listing, and Anthropic
messages (JSON and SSE). The canonical request is compact, key-sorted JSON. A
route response key is `SHA256("<run seed>:<route>:" + canonical request)`;
IDs, text, embeddings, timestamps, and token counts all derive from that key.
No wall clock or global random state enters a response. Per-route fixed latency
can be supplied through `provider_server(...)` or repeated
`--provider-latency ROUTE=MILLISECONDS` options. The smoke command sends every fixture twice
and requires byte-identical response bodies before either target starts.

Gateway secrets point only to this loopback server and use the obvious fake key
`test-key-fake`. Assistant turns reuse `dev/dev_stubs`: the runner stages its
fake `claude` CLI on each target's PATH. The Phase 18 gateway recorder's
loopback provider shapes and the Phase 20 CLI-provider/session patterns are the
basis for these fixtures. There are no live LLM calls.

## Seeded client

The measured client uses one `aiohttp.ClientSession` and bounded reusable
connections. A local `random.Random(seed)` creates the complete request order
and cumulative think-time schedule before work begins. Warm-up requests use the
same connection pool but are not added to raw samples or summaries. SSE is
parsed incrementally and retains each frame plus time-to-first-event and every
inter-frame gap.

## Raw metrics schema

Every target writes one JSON document validated against
`raw-metrics.schema.json` (schema version `1.0.0`). Top-level fields are:

- `run`: target, cell, seed, concurrency, warm-up count, UTC bounds.
- `summary`: overall and per-endpoint count/error/RPS, p50/p95/p99/max latency,
  plus SSE time-to-first-event and inter-frame-gap percentiles.
- `requests`: every measured request, including status, latency, decoded body,
  error, and optional raw SSE frame/timing data.
- `jobs`: every submitted job's terminal state and submit-to-terminal time,
  split into observed queue wait and execution time. If a fast job skips the
  observable RUNNING poll, its conservative queue wait is submit-to-terminal
  and execution is zero.
- `resources`: one-second Linux `/proc` samples summing whole-process-tree
  `VmRSS` and cumulative `utime`/`stime` (server, Python workers/Huey, or Rust
  native worker subprocesses).
- `db_pool`: one-second `pg_stat_activity` occupancy samples. This is explicitly
  a server-side pool-occupancy proxy because neither server exposes a portable
  pool-stat API.
- `provider`: request/response hashes and byte counts observed by the local
  fake.
- `equivalence`: deterministic response sample and all job terminal states.

Validate artifacts independently with:

```bash
uv run --frozen python -m rust.bench.genai.runner validate \
  rust/bench/genai/results/smoke-python.json \
  rust/bench/genai/results/smoke-rust.json
uv run --frozen python -m rust.bench.genai.equivalence \
  rust/bench/genai/results/smoke-python.json \
  rust/bench/genai/results/smoke-rust.json
```
