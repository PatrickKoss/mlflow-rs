# Phase 23 GenAI evaluation: Python vs Rust MLflow server

## Executive summary

Across 122 raw result files covering every GenAI family in §12, the Rust
server preserved deterministic response, stream, job, and archive equivalence
with zero measured errors. It was faster in every selected headline cell and
used 117.8x less mean process-tree RSS and 9.9x less process-tree CPU in the
realistic mixed soak. The capstone soak completed 10,000 scheduled requests,
1,000 jobs, and a background 1,000-trace archive pass per target; neither
target showed monotonic RSS growth (Python -3299.19 MiB/h, Rust +10.23 MiB/h),
so both meet the leak criterion. This is a strong Phase 23 verdict for the Rust
port, with narrow but real exceptions: three few-large-job T23.3 cells were
slower because native worker subprocess startup dominated, and five T23.4
stream cells had a slower Rust TTFE tail percentile under socket-read frame
batching. Those regressions and the soak's positive Rust RSS slope are reported
below rather than averaged away.

## Method

The benchmark follows the [Phase 23 shared method](../../RUST_TRACKING_SERVER_PLAN.md#phase-23--genai-performance--resource-evaluation-python-vs-rust).
It ran on Linux 5.15.167.4 under WSL2, an Intel Core Ultra 9 285K with 24
logical CPUs and 46 GiB RAM, Docker 29.1.3, Compose 5.0.0, PostgreSQL 16, and
the locally cached MinIO images. The Python target was the production FastAPI
application with four uvicorn workers and its real Huey job runtime. The Rust
target was a release `mlflow-server` with real `mlflow-genai-worker`
subprocesses. Targets ran serially, Python then Rust, and each received a fresh
database, MinIO prefix, and local archive repository. They were never under
load at the same time.

The harness constructs the complete request order, payloads, and think-time
schedule from a fixed seed before either target starts. A loopback mock
provider derives IDs, text, embeddings, timestamps, token counts, and SSE
frames from the SHA-256 of the seed, route, and canonical request. Assistant
CLI traffic uses the repository's staged fake `claude` executable. External
dependencies are faked because provider latency, availability, billing, and
model nondeterminism are not properties of the tracking server; no live
provider call or real secret is involved.

"Equivalent" means normalized sampled HTTP responses match after removing
known nondeterministic IDs, UUIDs, timestamps, durations, ports, and fresh
storage prefixes; sampled SSE streams have the same complete ordered payload
sequence; all measured jobs reach the same terminal state with equivalent
sampled results; and the archived `traces.pb` proof bytes match. The soak's
archive SHA-256 was
`76fadae54b1c699b387dd99dd88792f6950fcda3077d55fb81bed4e347dd6cd4`
on both targets. All T23.2, T23.3, T23.4, and T23.5 equivalence checks passed.
Warm-ups are excluded from every result.

## Complete §12 coverage

The tables select one representative committed cell for every family. CRUD
rows use T23.2's 10,000-request, small-payload, 128-client read-heavy cell.
Jobs use T23.3's 1,000-job high-fan-out cells. Streaming, PromptLab, and
archival use canonical T23.4 cells. The T23.5 table later shows behavior when
these paths compete in one offered-load soak. Latency is client-observed wall
time. Throughput is saturation throughput in the matrix tables, but controlled
offered throughput in the soak.

### CRUD and administrative paths (T23.2)

| §12 family / representative cell | Python p50/p95/p99 ms | Rust p50/p95/p99 ms | Python/Rust RPS | Errors Py/Rust |
| --- | --- | --- | ---: | ---: |
| 12.1 Datasets / `small-c128-rh` | 1448.36 / 4959.93 / 5830.48 | 29.15 / 70.72 / 114.15 | 55.8 / 3220.2 | 0 / 0 |
| 12.4 Issues / `small-c128-rh` | 108.95 / 243.19 / 299.11 | 35.76 / 86.98 / 98.11 | 1005.0 / 3029.1 | 0 / 0 |
| 12.5 Label schemas / `small-c128-rh` | 98.01 / 308.50 / 379.15 | 31.74 / 39.30 / 44.56 | 966.1 / 3971.7 | 0 / 0 |
| 12.6 Review queues / `small-c128-rh` | 146.96 / 299.50 / 359.11 | 30.03 / 50.85 / 67.04 | 841.8 / 4037.7 | 0 / 0 |
| 12.7 Prompt optimization CRUD / `small-c128-rh` | 10619.19 / 15539.83 / 16314.97 | 450.93 / 1264.10 / 1442.21 | 12.0 / 240.2 | 0 / 0 |
| 12.8 Gateway CRUD / `small-c128-rh` | 90.31 / 797.67 / 8770.25 | 10.89 / 30.12 / 57.76 | 232.3 / 9040.0 | 0 / 0 |

T23.2 measured 154,000 requests per target across 28 paired cells. All cells
had zero errors and passed equivalence; Rust had no p50, p95, or throughput
regression in that matrix.

### Evaluation and scorer jobs (T23.3)

Job latency is submit-to-terminal wall time in seconds. Throughput is completed
jobs per minute.

| §12 family / representative cell | Python p50/p95/p99 s | Rust p50/p95/p99 s | Python/Rust jobs/min | Errors Py/Rust |
| --- | --- | --- | ---: | ---: |
| 12.2 Evaluation / `evaluation-high-fanout` | 355.521 / 628.572 / 650.280 | 0.214 / 0.217 / 0.219 | 85.2 / 14743.1 | 0 / 0 |
| 12.3 Scorers / `scorer-high-fanout` | 247.650 / 442.176 / 456.645 | 2.384 / 3.243 / 3.394 | 121.6 / 14078.6 | 0 / 0 |

The broader matrix also moved issue discovery from 78.9 to 4871.8 jobs/min
and prompt optimization from 20.1 to 857.6 jobs/min in their high-fan-out
cells. The few-large-job regressions are listed separately below.

### Gateway runtime, assistant, PromptLab, and archival (T23.4)

Streaming rows report TTFE and frames/s. PromptLab reports HTTP latency and
requests/s. Archival reports finalize-visibility cadence and traces/s.

| §12 family / representative cell | Metric | Python p50/p95/p99 ms | Rust p50/p95/p99 ms | Python/Rust throughput | Errors Py/Rust |
| --- | --- | --- | --- | ---: | ---: |
| 12.9 Gateway runtime / `chat-small-c16` | TTFE | 99.18 / 227.82 / 364.23 | 50.11 / 61.22 / 1040.80 | 986.4 / 1810.3 frames/s | 0 / 0 |
| 12.10 Assistant / `cli-c64` | TTFE | 118.16 / 355.57 / 386.73 | 71.84 / 116.75 / 153.26 | 870.3 / 2030.0 frames/s | 0 / 0 |
| 12.11 PromptLab / `small-c64` | HTTP | 930.01 / 1590.07 / 1926.06 | 41.10 / 68.37 / 107.13 | 67.5 / 1407.8 RPS | 0 / 0 |
| 12.12 Trace archival / `pass-small` | finalize | 7.72 / 13.69 / 15.16 | 3.50 / 6.11 / 6.63 | 127.8 / 282.6 traces/s | 0 / 0 |

The gateway headline deliberately retains its Rust-slower p99. Across T23.4,
Rust completed more frames/s in every streaming cell, and every stream,
PromptLab request, and archive operation completed without error.

## T23.5 mixed soak

The canonical seed 2350 schedule issued exactly 10,000 primary requests over
ten minutes at concurrency 64. Public job-state polls were measured control
traffic but excluded from the primary count and family latency. A one-pass
1,000-trace archive operation ran concurrently in the background.

| Traffic component | Requests | Mix |
| --- | ---: | ---: |
| Dataset upserts | 2,000 | 20.0% |
| Gateway chat, non-streaming | 2,500 | 25.0% |
| Gateway chat streams | 500 | 5.0% |
| Assistant requests | 500 | 5.0% |
| Label-schema reads | 1,750 | 17.5% |
| Review-queue reads | 1,750 | 17.5% |
| Evaluation submissions | 500 | 5.0% |
| Scorer submissions | 500 | 5.0% |

The 500 assistant requests form 250 complete session POST + stream GET pairs.
All 500 evaluation and 500 scorer jobs succeeded on each target.

| Soak family | Python p50/p95/p99 ms | Rust p50/p95/p99 ms | Python/Rust offered RPS | Errors Py/Rust |
| --- | --- | --- | ---: | ---: |
| Assistant | 23.89 / 58.48 / 80.64 | 21.63 / 47.13 / 57.32 | 0.785 / 0.794 | 0 / 0 |
| Datasets | 10.08 / 19.32 / 41.02 | 4.93 / 6.08 / 7.23 | 3.142 / 3.175 | 0 / 0 |
| Evaluation submit | 19.35 / 33.48 / 64.22 | 10.56 / 12.35 / 13.43 | 0.785 / 0.794 | 0 / 0 |
| Gateway | 39.45 / 86.95 / 140.08 | 22.56 / 30.84 / 35.74 | 4.713 / 4.762 | 0 / 0 |
| Label schemas | 6.78 / 15.18 / 44.87 | 1.92 / 2.54 / 2.90 | 2.749 / 2.778 | 0 / 0 |
| Review queues | 7.47 / 15.87 / 37.30 | 2.20 / 2.91 / 3.39 | 2.749 / 2.778 | 0 / 0 |
| Scorer submit | 7.17 / 14.01 / 29.30 | 3.38 / 4.36 / 5.52 | 0.785 / 0.794 | 0 / 0 |

Total wall time was 636.57 seconds for Python and 630.01 seconds for Rust,
giving 15.709 and 15.873 primary requests/s. These close numbers confirm that
both targets kept up with the identical offered schedule; they are not a
saturation claim. Aggregate completed-job rates were 99.14 and 100.23 jobs/min.
The background archive pass ran at 16.23 and 16.72 traces/s including watcher
activation; T23.4's isolated pass is the archival capacity comparison.

### RSS and CPU over time

RSS is sampled every second by summing `VmRSS` for the server and every current
descendant. The leak rule follows T14.2 at one-minute resolution: failure
requires every consecutive minute mean to be non-decreasing and the final mean
to exceed the first by more than 5%.

| Target | One-minute mean process-tree RSS MiB |
| --- | --- |
| Python | `6748.0 → 6904.8 → 7018.0 → 7083.7 → 7453.8 → 7353.6 → 7491.3 → 6984.3 → 6849.3 → 6885.8 → 5050.7` |
| Rust | `57.6 → 58.5 → 58.8 → 59.0 → 58.9 → 59.3 → 59.6 → 59.7 → 59.4 → 59.5 → 59.3` |

| Target | Mean RSS MiB | Peak RSS MiB | CPU-s | RSS slope MiB/h | Final/first | Monotonic growth? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Python | 6956.55 | 9533.19 | 424.63 | -3299.19 | -25.15% | no — PASS |
| Rust | 59.05 | 79.65 | 42.84 | +10.23 | +3.10% | no — PASS |

The Python last bin is the partial settlement tail after load ended; the slope
uses all sampled bins consistently for both targets. Rust's slope is positive,
but its minute means are not monotonic and final growth is below the 5%
threshold. The acceptance criterion is therefore met on both sides: there is
no monotonic RSS growth. PostgreSQL occupancy also remained bounded: Python
peaked at 55 total / 5 active / 5 active-waiting connections, Rust at 33 / 3 /
3, with no timeout or request error.

## Key ratios

- Dataset read-heavy p50/p95 moved from `1448.4/4959.9 → 29.2/70.7 ms`,
  while throughput moved from 55.8 to 3220.2 RPS.
- Prompt-optimization CRUD p50/p95 moved from
  `10619.2/15539.8 → 450.9/1264.1 ms`; gateway CRUD moved from
  `90.3/797.7 → 10.9/30.1 ms`.
- Evaluation high-fan-out job wall p50/p95 moved from
  `355.521/628.572 → 0.214/0.217 s`; scorer invoke moved from
  `247.650/442.176 → 2.384/3.243 s`.
- Gateway c16 TTFE p50/p95 moved from `99.2/227.8 → 50.1/61.2 ms`,
  although Rust's p99 regressed as disclosed below. Assistant c64 moved from
  `118.2/355.6 → 71.8/116.8 ms`.
- PromptLab c64 p50/p95 moved from `930.0/1590.1 → 41.1/68.4 ms`;
  archive finalize cadence moved from `7.7/13.7 → 3.5/6.1 ms` while
  throughput rose from 127.8 to 282.6 traces/s.
- In the soak, mean/peak RSS moved from
  `6956.6/9533.2 → 59.0/79.7 MiB`: 117.8x and 119.7x less for Rust.
  Process-tree CPU moved from `424.63 → 42.84 CPU-s`, a 9.9x reduction.

## Rust-slower cells and anomalies

Every measured Rust-slower percentile or throughput cell across the four
subtasks is listed here.

T23.3 had three few-large-job regressions:

| Cell | Python p50/p95/p99 s | Rust p50/p95/p99 s | Python/Rust jobs/min |
| --- | --- | --- | ---: |
| `prompt-optimization-large-payload` | 106.965 / 178.151 / 178.151 | 305.742 / 509.159 / 509.159 | 3.36 / 1.18 |
| `online-large-payload` trace scorer | 14.413 / 18.402 / 18.402 | 32.992 / 32.992 / 32.992 | 32.44 / 18.19 |
| `online-large-payload` session scorer | 14.507 / 18.494 / 18.494 | 32.992 / 32.993 / 32.993 | 32.44 / 18.19 |

Rust launches one native worker subprocess per job. That isolation performs
well when enough jobs amortize startup and parallelize, but subprocess startup
dominates these cells' very small number of large jobs. It is the mechanism
behind the lower throughput here, not a hidden error or equivalence failure.

T23.4 had five TTFE tail regressions:

| Cell / percentile | Python ms | Rust ms |
| --- | ---: | ---: |
| `assistant-openai-c16` p99 | 239.75 | 1064.23 |
| `gateway-chat-large-c16` p99 | 256.98 | 1039.82 |
| `gateway-chat-small-c16` p99 | 364.23 | 1040.80 |
| `gateway-chat-small-c64` p95 / p99 | 1016.35 / 1095.75 | 1080.42 / 3120.08 |
| `gateway-passthrough-large-c64` p95 / p99 | 557.04 / 704.95 | 1098.24 / 3631.14 |

Four of those shapes also produced a slower complete-stream HTTP p99:
`assistant-openai-c16` 353.73 → 1070.73 ms,
`gateway-chat-large-c16` 712.55 → 1186.07 ms,
`gateway-chat-small-c64` 1229.50 → 3142.74 ms, and
`gateway-passthrough-large-c64` 3710.10 → 3808.17 ms. At the harness's
unrounded timestamp resolution, Rust also had nominally higher same-read
inter-frame gaps in `assistant-cli-c1` (p50/p95/p99), `assistant-cli-c16`
(p95/p99), `assistant-cli-c64` (p50), and `gateway-chat-small-c1` (p50/p95).
All were between 0.001 and 0.0024 ms and round to 0.00 ms in the summary; they
are listed for completeness, not interpreted as provider-frame latency.

Parsed SSE frames received in one socket read share the client's timestamp.
That socket-read batching explains both 0 ms-looking inter-frame gaps and the
Rust TTFE tail shape under concurrency; it does not erase the regressions.
Rust still had higher frames/s and zero completion errors in all five cells.

The T23.5 soak had no Rust-slower family p50, p95, p99, throughput, or error
cell. Two isolated submit maxima were slower despite much lower Rust p99:
evaluation `233.69 → 339.61 ms` and scorer `53.57 → 104.26 ms`. Rust's
observed queue p50 was also higher because fast jobs often skipped the polled
RUNNING state, causing the harness to conservatively attribute the whole
submit-to-terminal interval to queue time. Finally, Rust's soak RSS slope was
positive at +10.23 MiB/h; its non-monotonic curve and +3.10% final growth make
the leak verdict PASS under the predeclared rule, not evidence of a negative
slope.

T23.2 had no Rust-slower p50, p95, p99, maximum-latency, or throughput cell
across its full matrix.

## Limitations

- The Rust artifact factory does not yet wire S3. T23.4 archival and the soak
  therefore used fresh local `file://` archive repositories on both targets.
  PromptLab used `mlflow-artifacts://localhost/`: Python proxied to its MinIO
  prefix and Rust used a fresh local proxy destination. This preserves server
  path comparison but is not an S3 artifact-factory benchmark.
- The AFTER Guidelines guardrail was loaded on the streamed gateway endpoint,
  but post-LLM guardrails are intentionally not executed on streams under the
  route contract. Budget accounting remained active.
- T23.2 Python results came from two serial fresh-target slices. Absolute RSS
  across that slice boundary has a retained-memory caveat; per-cell request,
  CPU, and resource series remain complete and no target loads overlapped.
- This host exposed no usable cgroup `pids.current`. Raw machine state records
  it as null and retains `/proc` process counts and target-tree PIDs instead.
- Process-tree RSS sums pages once per process, including shared uvicorn pages.
  It is deployment RSS rather than proportional-set-size accounting.
- Fast jobs can skip the polling interval's RUNNING observation, so queue vs
  execution splits are conservative. Submit-to-terminal wall time and terminal
  status are unaffected.
- The soak's archive throughput includes watcher activation and concurrent
  traffic. Use T23.4's isolated archival cells for capacity conclusions.
- WSL2 and a shared development host add filesystem and scheduling noise. The
  targets were serial and never overlapped, but ratios are more portable than
  absolute milliseconds. Pin the cached MinIO image digest for cross-machine
  reproduction.

## Reproduction and raw results

From the repository root, this one command starts the dependencies and runs
the canonical soak. The runner idempotently initializes MinIO, builds release
binaries, recreates storage per target, and executes `docker compose down -v`
on exit:

```bash
docker compose -p mlflow-t23-genai -f rust/bench/genai/docker-compose.yml up -d --wait postgres minio && \
uv run --frozen --extra db --extra extras --with litellm \
  python -m rust.bench.genai.runner t23-5
```

Re-run the component matrices without changing the command shape:

```bash
uv run --frozen --extra db python -m rust.bench.genai.runner t23-2
uv run --frozen --extra db --with litellm python -m rust.bench.genai.runner t23-3
uv run --frozen --extra db --extra extras --with litellm \
  python -m rust.bench.genai.runner t23-4
```

The [harness README](genai/README.md) documents reduced plumbing checks and
independent schema/equivalence validation. Committed raw inventory is 56 JSON
files for [T23.2](genai/results/t23_2/t23_2_summary.md), 24 for
[T23.3](genai/results/t23_3/t23_3_summary.md), 40 for
[T23.4](genai/results/t23_4/t23_4_summary.md), and the two canonical soak files
plus [T23.5 summary](genai/results/t23_5/t23_5_summary.md): 122 JSON artifacts
in total. T23.2–T23.4 numbers above are folded from those committed artifacts;
they were not re-run for this report.
