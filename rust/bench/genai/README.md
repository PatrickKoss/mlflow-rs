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

## T23.2 CRUD + read-path matrix

Run the complete Tier-A matrix with:

```bash
uv run --frozen --extra db python -m rust.bench.genai.runner t23-2
```

The matrix is a four-cell fractional-factorial design repeated for every
family. It covers all three requested concurrency points without paying for the
full 24-combination Cartesian product:

| Cell | Payload | Clients | Mix | Measured requests |
| --- | --- | ---: | --- | ---: |
| `small-c1-wh` | small | 1 | 90% write / 10% read | 10,000 |
| `small-c128-rh` | small | 128 | 10% write / 90% read | 10,000 |
| `large-c16-wh` | large | 16 | 90% write / 10% read | 1,000 |
| `large-c128-rh` | large | 128 | 10% write / 90% read | 1,000 |

This gives a single-client write baseline, high-contention small reads,
mid-concurrency large writes, and high-concurrency large reads. The complete
stream, including inter-arrival times, is generated before a cell and is
identical for Python and Rust. Twenty warm-up requests use the same client pool
and are excluded from request and resource statistics.

Each target gets one fresh database and artifact prefix. Its seven family
corpora are bulk-seeded once with deterministic IDs and timestamps, then its 28
cells run serially. Python runs all cells before Rust; the targets are never
under load together. T23.2 sets the documented MLflow SQL pool knobs to 32
base + 8 overflow for both targets and PostgreSQL `max_connections=400`; this
prevents the Python prompt-search implementation's per-page pool reacquisition
from producing pool-timeout errors at 128 clients. Every family has at least
10,000 backing rows. Dataset reads cover both 10,000 datasets and 10,000
records. Scorer reads cover 100
names with 100 versions each. Prompt search scans 10,000 optimizer jobs, 10 of
which match the requested experiment so its unpaginated response stays bounded;
CRUD-only cancel/delete fixtures use a different job name. Gateway list reads
page through a 10,000-policy corpus.

“Large” follows each schema's realistic ceiling:

- datasets: eight 64 KiB record outputs, about 512 KiB per upsert;
- scorers: 64 KiB serialized scorer JSON;
- issues: 64 KiB description;
- label schemas: 250-character name, 1,000-character instruction, and ten
  64-character categorical options (about 2 KiB);
- review queues: ten 250-character users plus 100 schema or item references
  (about 6–8 KiB);
- prompt optimization: 5 KiB optimizer JSON on measured create requests (the
  largest realistic value below the 6,000-character run-parameter cap);
- gateway admin: 64 KiB obvious-fake secret material through the envelope
  encryption update path.

Prompt optimization keeps the real Python Huey / Rust native runtime enabled,
but only 1% of measured writes create and enqueue jobs; 9% cancel deterministic
pending fixtures and 90% delete deterministic finalized fixtures. This keeps
T23.2 focused on CRUD while still measuring create/enqueue. T23.3 owns sustained
job-engine saturation.

T23.2 writes one schema `1.1.0` JSON per target/cell under
`results/t23_2/`, plus `t23_2_summary.md`. Version 1.1 adds matrix axes,
request-body byte counts, overall latency percentiles, an equivalence verdict,
and the pre-cell load/pids process snapshot. The schema remains backward
compatible with checked-in T23.1 version `1.0.0` artifacts. Non-sampled
responses are retained as byte count + SHA-256 to keep large-payload artifacts
bounded; the deterministic equivalence sample retains full decoded responses.

For a short harness check, select a family and reduce volume into a temporary
directory; reduced counts are marked as trimmed in the raw metadata and
summary:

```bash
uv run --frozen --extra db python -m rust.bench.genai.runner t23-2 \
  --families datasets --small-requests 20 --large-requests 10 \
  --output-dir /tmp/t23-2-check
```

## T23.3 jobs + native-engine matrix

Run the complete async-job matrix with:

```bash
uv run --frozen --extra db --with litellm python -m rust.bench.genai.runner t23-3
```

The fractional matrix uses isolated high-fan-out and large-payload cells for
evaluation, scorer invoke, issue discovery, and prompt optimization. Online
trace and session scoring share paired cells because the public minute
scheduler dispatches both pools from the same registered-scorer experiment.
An all-kind burst adds cross-pool queue/fairness pressure, and a steady drip
keeps direct submissions below capacity while observing two real scheduler
ticks for each online kind. Canonical volumes are 1,000 jobs/kind for fan-out,
about 10 jobs/kind over 1,000 rows for large payloads, 100 jobs/kind for burst,
and 20 direct jobs/kind plus two scheduler jobs/kind for drip.

Issue discovery uses a separate 10,000-trace corpus so its ten large jobs each
receive a disjoint seeded 1,000-row partition. Rotating the same 1,000 traces
across all ten jobs caused overlapping issue-artifact writes and deterministic
timeouts during calibration; disjoint partitions preserve the canonical
ten-by-1,000 volume and measure payload cost instead of write contention.

Public invoke creation is capped at three concurrent requests on both targets,
leaving one of the four reference uvicorn workers available for the handlers'
loopback MLflow client calls. Mixed cells serialize the creation requests to
avoid the prompt dataset-lineage race. The measured burst is the resulting
worker-queue burst: its 100 jobs per kind remain far above every configured
job-pool concurrency.

Issue-invoke creation is serialized on both targets because multiple requests
can be accepted by one reference uvicorn worker and deadlock its loopback
trace reads. This affects only job creation; the 1,000-job queue is dispatched
at full issue-worker concurrency and remains the measured fan-out workload.

Online work is created only by public scorer registration and online-config
APIs plus the real periodic scheduler. Because the generic public jobs API has
no list operation, the harness uses a read-only jobs-table query to discover
scheduler-created IDs, then measures every state transition through the public
GET jobs API. It never inserts, updates, claims, or executes a job internally.

Schema `1.2.0` adds job-kind throughput/percentiles, burst fairness, per-sample
thread counts, and explicit leak-check inputs/verdicts. Every fan-out cell is a
1,000-completion leak observation; the verdict requires settled process/thread
counts and RSS to return to a bounded, non-monotonically-growing baseline. The
settled RSS allowance is 15% plus 64 MiB over the cold process-tree baseline
to accommodate lazily retained Python runtime caches; the final samples must
still be flat and process/thread counts bounded. Leak-applicable cells keep
sampling for at least five and up to 60 seconds after terminal completion so
runtime children must actually exit and the whole tree must reach that flat
tail; reaching the timeout without a flat tail fails the cell. The settled
thread-count spread allowance is the greater of two threads or 1% of the whole
tree, while the end count remains capped at 10% plus eight over baseline.
Polling uses a deterministic one-hour terminal deadline so the 1,000-job
prompt queue can drain through its two-worker pool; any job still nonterminal
at that deadline is recorded as an error.

For a reduced plumbing check:

```bash
uv run --frozen --extra db --with litellm python -m rust.bench.genai.runner t23-3 \
  --cells evaluation-high-fanout --fanout-jobs 2 --large-rows 2 \
  --targets rust --output-dir /tmp/t23-3-check
```
