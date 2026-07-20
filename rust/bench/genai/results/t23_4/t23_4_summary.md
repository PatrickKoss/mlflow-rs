# T23.4 streaming + archival benchmark summary

This is raw material for T23.5, not the final Phase 23 report. Targets ran
serially on PostgreSQL 16 + MinIO with a fresh DB and artifact prefix per target.
Trace payloads used a fresh local file:// ARCHIVE_REPO per target because the Rust
artifact factory does not currently wire S3. Promptlab used the
`mlflow-artifacts://localhost/` proxy URI: Python proxied it to MinIO and Rust
used its fresh local proxy destination.
All upstream traffic used the loopback deterministic provider, fake Claude CLI,
or the assistant's OpenAI-compatible gateway stub; no live provider was reachable.
An AFTER Guidelines guardrail backed by the deterministic mock provider and a
global ALERT budget were attached/enabled on
the measured gateway endpoint. Per contract, post-LLM guardrails are loaded but
not executed on streams. Usage tracking remained enabled so budget accounting ran.

## Chosen matrix

The fractional design keeps 4-6 cells per family while covering 1/16/64 stream
concurrency, both ~10 and 100+ frame gateway variants, both assistant stub modes,
promptlab payload/concurrency pressure, two archive payload sizes, and both read APIs.
No volumes were trimmed; every cell ran at its canonical count.

| Family | Cell | Kind | C | Count | Canonical | Rationale |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gateway | `chat-small-c1` | stream-chat | 1 | 1,000 | 1,000 | single-stream baseline |
| gateway | `chat-small-c16` | stream-chat | 16 | 1,000 | 1,000 | ordinary multiplexing |
| gateway | `chat-small-c64` | stream-chat | 64 | 1,000 | 1,000 | high stream fan-out |
| gateway | `chat-large-c16` | stream-chat | 16 | 1,000 | 1,000 | 100+ frame stream cost |
| gateway | `passthrough-large-c64` | stream-passthrough | 64 | 1,000 | 1,000 | high-fanout 100+ frame passthrough |
| gateway | `nonstream-mixed-c16` | nonstream-mixed | 16 | 1,000 | 1,000 | chat, embeddings, passthrough baseline |
| assistant | `cli-c1` | assistant-stream | 1 | 1,000 | 1,000 | scripted CLI baseline |
| assistant | `cli-c16` | assistant-stream | 16 | 1,000 | 1,000 | CLI multiplexing |
| assistant | `cli-c64` | assistant-stream | 64 | 1,000 | 1,000 | CLI process fan-out |
| assistant | `openai-c16` | assistant-stream | 16 | 1,000 | 1,000 | OpenAI-compatible assistant path |
| promptlab | `small-c1` | promptlab | 1 | 1,000 | 1,000 | artifact writer baseline |
| promptlab | `small-c16` | promptlab | 16 | 1,000 | 1,000 | artifact writer multiplexing |
| promptlab | `small-c64` | promptlab | 64 | 1,000 | 1,000 | artifact writer saturation |
| promptlab | `large-c16` | promptlab | 16 | 1,000 | 1,000 | large prompt artifact pressure |
| archival | `pass-small` | archive-pass | 1 | 10,000 | 10,000 | 10k small-trace pass when untrimmed |
| archival | `pass-large` | archive-pass | 1 | 1,000 | 1,000 | 1k 64-KiB-trace pass |
| archival | `get-trace-c1` | archive-get-trace | 1 | 1,000 | 1,000 | archived getTrace baseline |
| archival | `get-trace-c16` | archive-get-trace | 16 | 1,000 | 1,000 | archived getTrace multiplexing |
| archival | `artifact-c64` | archive-artifact | 64 | 1,000 | 1,000 | archived artifact high concurrency |
| archival | `mixed-read-c16` | archive-mixed | 16 | 1,000 | 1,000 | balanced archived read APIs |

## Streaming and interactive cells

| Family/cell | N streams | Py TTFE p50/p95 ms | Rust TTFE p50/p95 ms | Py/Rust gap p95 ms | Py/Rust frames/s | Py/Rust completion errors | Py/Rust RSS MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `gateway/chat-small-c1` | 1,000 | 59.94/69.94 | 49.94/59.87 | 0.00/0.00 | 163.2/197.6 | 0/0 | 4456.6/46.4 | 57.43/6.34 | PASS |
| `gateway/chat-small-c16` | 1,000 | 99.18/227.82 | 50.11/61.22 | 16.13/0.00 | 986.4/1810.3 | 0/0 | 5150.0/62.8 | 43.72/6.44 | PASS |
| `gateway/chat-small-c64` | 1,000 | 277.55/1016.35 | 59.16/1080.42 | 40.32/1.11 | 1155.6/2010.6 | 0/0 | 5156.8/84.7 | 38.13/6.33 | PASS |
| `gateway/chat-large-c16` | 1,000 | 68.01/162.88 | 27.72/34.79 | 10.44/1.49 | 5614.9/9795.1 | 0/0 | 5182.6/90.4 | 60.06/8.84 | PASS |
| `gateway/passthrough-large-c64` | 1,000 | 170.21/557.04 | 35.84/1098.24 | 51.24/3.46 | 4114.5/7767.2 | 0/0 | 5228.2/91.9 | 126.97/7.86 | PASS |
| `assistant/cli-c1` | 1,000 | 50.01/60.02 | 49.38/59.39 | 0.00/0.00 | 29.1/58.3 | 0/0 | 5204.3/108.3 | 6.33/0.90 | PASS |
| `assistant/cli-c16` | 1,000 | 59.47/61.82 | 49.49/59.05 | 0.00/0.00 | 435.2/863.6 | 0/0 | 5331.7/307.0 | 4.27/0.78 | PASS |
| `assistant/cli-c64` | 1,000 | 118.16/355.57 | 71.84/116.75 | 9.80/0.04 | 870.3/2030.0 | 0/0 | 5382.4/997.2 | 4.69/1.15 | PASS |
| `assistant/openai-c16` | 1,000 | 75.93/179.40 | 49.60/59.49 | 21.62/0.02 | 463.5/887.9 | 0/0 | 5210.6/95.4 | 39.64/8.68 | PASS |

## Non-streaming gateway + promptlab

| Family/cell | N | Py p50/p95 ms | Rust p50/p95 ms | Py/Rust RPS | Py/Rust errors | Py/Rust RSS MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `gateway/nonstream-mixed-c16` | 1,000 | 159.74/329.61 | 23.47/29.43 | 92.2/399.1 | 0/0 | 5211.7/91.9 | 49.47/8.83 | PASS |
| `promptlab/small-c1` | 1,000 | 110.06/129.92 | 11.67/13.77 | 8.7/83.0 | 0/0 | 4298.7/33.9 | 61.07/2.86 | PASS |
| `promptlab/small-c16` | 1,000 | 239.94/289.68 | 16.06/19.33 | 64.8/932.1 | 0/0 | 4491.3/37.0 | 51.18/2.41 | PASS |
| `promptlab/small-c64` | 1,000 | 930.01/1590.07 | 41.10/68.37 | 67.5/1407.8 | 0/0 | 4512.4/48.2 | 52.80/1.81 | PASS |
| `promptlab/large-c16` | 1,000 | 219.57/280.05 | 16.26/19.17 | 75.1/915.8 | 0/0 | 4498.0/51.5 | 46.23/2.50 | PASS |

## Trace archival

### pass-small

| Target | Traces | traces/s | finalize visibility p50/p95 ms | RSS MiB | CPU-s | Eq |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Python | 10,000 | 127.8 | 7.72/13.69 | 5220.4 | 50.93 | PASS |
| Rust | 10,000 | 282.6 | 3.50/6.11 | 105.2 | 8.09 | PASS |

### pass-large

| Target | Traces | traces/s | finalize visibility p50/p95 ms | RSS MiB | CPU-s | Eq |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Python | 1,000 | 127.2 | 7.85/9.06 | 5228.6 | 5.50 | PASS |
| Rust | 1,000 | 267.6 | 3.63/4.11 | 113.8 | 0.90 | PASS |

- `get-trace-c1` (1,000 reads, c1): p50/p95 ms Python 49.97/59.97, Rust 1.21/1.42; RPS 19.6/798.1; errors 0/0; equivalence PASS.
- `get-trace-c16` (1,000 reads, c16): p50/p95 ms Python 59.89/80.15, Rust 1.99/2.68; RPS 247.8/7058.1; errors 0/0; equivalence PASS.
- `artifact-c64` (1,000 reads, c64): p50/p95 ms Python 99.76/228.93, Rust 9.54/19.35; RPS 596.9/4460.1; errors 0/0; equivalence PASS.
- `mixed-read-c16` (1,000 reads, c16): p50/p95 ms Python 59.80/79.77, Rust 2.54/3.53; RPS 252.6/5931.6; errors 0/0; equivalence PASS.

Archive `traces.pb` equivalence uses the T21 byte-parity payload itself: one
deterministic payload per pass is stored base64 + SHA-256 in both raw files.
Archived getTrace proof compares its complete ordered spans, excluding known
target-specific TraceInfo preview and artifact-location decoration.
SSE equivalence strips IDs/timing through the shared recorder normalizer and
compares the complete ordered frame payload sequence for 16 seeded streams/cell.
A cell counts only after both raw files are marked PASS.

Finalize latency is a 50 ms poll of consecutive ARCHIVE_REPO tag-commit visibility.
The pass is sequential, so each visibility gap includes the next trace's upload;
it is an operational finalize-cadence proxy, not isolated SQL COMMIT duration.

## Rust-slower cells and anomalies

- `gateway/chat-small-c64 TTFE p95`: Python 1016.35, Rust 1080.42.
- `gateway/passthrough-large-c64 TTFE p95`: Python 557.04, Rust 1098.24.
- Parsed SSE frames delivered in one socket read share a timestamp, so some
  client-observed inter-frame p95 values round to 0.00 ms despite the provider's
  fixed 1 ms write gap.
- RSS is whole process-tree RSS: Python includes four uvicorn workers plus its job
  runtime, while Rust includes its server and any native workers.

## Raw result inventory

- `archival-artifact-c64-python.json`
- `archival-artifact-c64-rust.json`
- `archival-get-trace-c1-python.json`
- `archival-get-trace-c1-rust.json`
- `archival-get-trace-c16-python.json`
- `archival-get-trace-c16-rust.json`
- `archival-mixed-read-c16-python.json`
- `archival-mixed-read-c16-rust.json`
- `archival-pass-large-python.json`
- `archival-pass-large-rust.json`
- `archival-pass-small-python.json`
- `archival-pass-small-rust.json`
- `assistant-cli-c1-python.json`
- `assistant-cli-c1-rust.json`
- `assistant-cli-c16-python.json`
- `assistant-cli-c16-rust.json`
- `assistant-cli-c64-python.json`
- `assistant-cli-c64-rust.json`
- `assistant-openai-c16-python.json`
- `assistant-openai-c16-rust.json`
- `gateway-chat-large-c16-python.json`
- `gateway-chat-large-c16-rust.json`
- `gateway-chat-small-c1-python.json`
- `gateway-chat-small-c1-rust.json`
- `gateway-chat-small-c16-python.json`
- `gateway-chat-small-c16-rust.json`
- `gateway-chat-small-c64-python.json`
- `gateway-chat-small-c64-rust.json`
- `gateway-nonstream-mixed-c16-python.json`
- `gateway-nonstream-mixed-c16-rust.json`
- `gateway-passthrough-large-c64-python.json`
- `gateway-passthrough-large-c64-rust.json`
- `promptlab-large-c16-python.json`
- `promptlab-large-c16-rust.json`
- `promptlab-small-c1-python.json`
- `promptlab-small-c1-rust.json`
- `promptlab-small-c16-python.json`
- `promptlab-small-c16-rust.json`
- `promptlab-small-c64-python.json`
- `promptlab-small-c64-rust.json`
