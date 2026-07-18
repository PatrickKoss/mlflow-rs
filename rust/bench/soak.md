# T14.2 one-hour soak and load comparison

## Verdict

- **Python:** error rate 0.00000% (0/19405) — MET (<0.01%); no monotonic RSS growth — MET (+14.41 MiB/h regression slope).
- **Rust:** error rate 0.00000% (0/21171) — MET (<0.01%); no monotonic RSS growth — NOT MET (+7.29 MiB/h regression slope).

A monotonic-growth failure is defined here as every consecutive 10-minute mean being non-decreasing *and* the final mean exceeding the first by more than 5%. This separates bounded allocator/cache warm-up from sustained leak-like growth.

## Infrastructure and protocol

- Host: WSL2 Linux `5.15.167.4-microsoft-standard-WSL2`, Intel(R) Core(TM) Ultra 9 285K, 24 logical CPUs, 46.7 GiB RAM.
- Docker Engine 29.1.3; Compose 5.0.0; `postgres:16`, `minio/minio:latest`, and `minio/mc:latest`.
- Python 3.10.18; Python target used four uvicorn worker processes; Rust used the release `mlflow-server` binary's single Tokio runtime.
- Each target received a force-dropped/recreated `mlflow_soak` database followed by `mlflow db upgrade`, and a fresh target-specific prefix in the `mlflow-soak` bucket.
- Runs used direct `s3://` artifact URIs and identical SigV4 MinIO PUTs. Python was also started with `--serve-artifacts --artifacts-destination s3://...`. Rust's proxy destination was local because the v1 Rust artifact proxy does not implement cloud schemes; its S3 run metadata and client-direct uploads were otherwise identical.
- Runs were sequential (Python then Rust) on an otherwise idle host. Each measured load phase was exactly 3,600 seconds, preceded by 60 seconds of idle RSS sampling.

## Workload shape and totals

8 trainer threads ran 120–300 second runs, logging three-metric batches every 10 seconds. 2 readers polled runs, experiments, and recent metric history every 2.0 seconds. Every completed run uploaded an ONNX-like model plus three companion artifacts, then queried each metric through both history endpoints. OTLP traces were ingested every four training steps. One registered-model event per completed run exercised asynchronous delivery to a local webhook sink.

| Total | Python | Rust |
|---|---:|---:|
| Runs created | 139 | 138 |
| Runs finished | 139 | 138 |
| Metric points | 9,609 | 9,600 |
| Traces / spans | 854 | 855 |
| Artifacts uploaded | 556 | 552 |
| Artifact bytes | 41,002,772 | 40,707,790 |
| Registered models | 139 | 138 |
| Webhook deliveries | 139 | 138 |
| Webhook pending after 15 s | 0 | 0 |

Webhook delivery-task leak check: all triggered deliveries reached the sink with unique delivery IDs after the 15-second settlement window iff the pending count above is zero. Verdict: Python **MET**; Rust **MET**.

## Endpoint latency and errors

Latency is client-observed wall time. `s3_put_object` measures MinIO rather than the tracking server. Errors include every non-2xx response and client exception.

| Endpoint | Target | Requests | Errors | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| `experiment_create` | Python | 1 | 0 | 151.49 | 151.49 | 151.49 |
| `experiment_create` | Rust | 1 | 0 | 12.91 | 12.91 | 12.91 |
| `experiments_list` | Python | 3,350 | 0 | 49.97 | 50.61 | 60.29 |
| `experiments_list` | Rust | 3,796 | 0 | 0.93 | 1.65 | 1.84 |
| `experiments_search` | Python | 3,350 | 0 | 51.14 | 59.20 | 63.65 |
| `experiments_search` | Rust | 3,796 | 0 | 1.06 | 1.69 | 2.02 |
| `log_batch` | Python | 3,342 | 0 | 9.45 | 49.98 | 60.05 |
| `log_batch` | Rust | 3,338 | 0 | 4.55 | 6.58 | 7.80 |
| `metric_history` | Python | 3,767 | 0 | 49.94 | 50.26 | 60.07 |
| `metric_history` | Rust | 4,208 | 0 | 0.99 | 1.27 | 1.56 |
| `metric_history_bulk_interval` | Python | 417 | 0 | 49.99 | 51.57 | 60.03 |
| `metric_history_bulk_interval` | Rust | 414 | 0 | 1.82 | 2.31 | 4.00 |
| `registered_model_create` | Python | 139 | 0 | 49.99 | 60.12 | 70.82 |
| `registered_model_create` | Rust | 138 | 0 | 3.50 | 4.88 | 7.04 |
| `run_create` | Python | 139 | 0 | 49.99 | 60.03 | 168.77 |
| `run_create` | Rust | 138 | 0 | 4.61 | 14.74 | 18.89 |
| `run_update` | Python | 139 | 0 | 6.63 | 7.87 | 9.17 |
| `run_update` | Rust | 138 | 0 | 2.89 | 4.06 | 8.01 |
| `runs_search` | Python | 3,350 | 0 | 28.54 | 89.07 | 95.30 |
| `runs_search` | Rust | 3,796 | 0 | 5.41 | 6.74 | 7.29 |
| `s3_put_object` | Python | 556 | 0 | 3.54 | 9.65 | 10.89 |
| `s3_put_object` | Rust | 552 | 0 | 3.67 | 10.73 | 13.81 |
| `trace_ingest` | Python | 854 | 0 | 11.61 | 13.98 | 22.85 |
| `trace_ingest` | Rust | 855 | 0 | 3.86 | 5.35 | 6.31 |
| `webhook_create` | Python | 1 | 0 | 68.65 | 68.65 | 68.65 |
| `webhook_create` | Rust | 1 | 0 | 4.39 | 4.39 | 4.39 |

## RSS trend

RSS is the sum of `VmRSS` for the entire server process tree. This WSL2 host uses cgroup v1 and places the benchmark process in `/init.scope`, so `memory.current` cannot isolate the server; `/proc/<pid>/status` process-tree sampling is the plan's documented fallback.

| Target | Idle mean MiB | Loaded last-10m mean MiB | Min–max loaded MiB | Slope MiB/h | Monotonic growth? |
|---|---:|---:|---:|---:|---|
| Python | 2976.21 | 3144.75 | 3109.39–3154.70 | +14.41 | no — MET |
| Rust | 27.95 | 46.70 | 36.33–46.70 | +7.29 | yes — NOT MET |

Ten-minute mean RSS trend (MiB):

- Python: `0m:3131.3 -> 10m:3139.0 -> 20m:3141.8 -> 30m:3142.8 -> 40m:3143.6 -> 50m:3144.7`
- Rust: `0m:39.9 -> 10m:44.0 -> 20m:45.9 -> 30m:46.4 -> 40m:46.6 -> 50m:46.7`

## PostgreSQL pool health

Counts came from `pg_stat_activity` every 30 seconds and exclude the sampling `psql` connection. PostgreSQL retained its image default `max_connections=100`.

| Target | Samples | Max connections | Max active | Max active waiting | Verdict |
|---|---:|---:|---:|---:|---|
| Python | 123 | 31 / 100 | 0 | 0 | healthy |
| Rust | 122 | 15 / 100 | 1 | 1 | healthy |

Rust had one active query with a non-null PostgreSQL wait event in one sample; the pool never exceeded 15/100 connections, and there were no timeouts or request errors, so this is reported as transient query I/O/locking rather than pool exhaustion.

## Reproduction

From the repository root:

```bash
cargo build --manifest-path rust/Cargo.toml --release --bin mlflow-server
uv run --extra db python rust/bench/soak.py --target both \
  --duration-seconds 3600 --idle-seconds 60 --trainers 8 --readers 2 \
  --run-min-seconds 120 --run-max-seconds 300 \
  --metric-interval-seconds 10 --reader-interval-seconds 2 \
  --output-dir /tmp/mlflow-t14-real --report-dir rust/bench \
  --run-label t14-real-20260718
```

The runner starts Compose with health waits, resets the database per target, writes JSON to `--output-dir`, and always executes `docker compose down -v` on exit.

## Anomalies and limitations

- The Rust S3 artifact-proxy backend is not implemented, so identical client-direct S3 uploads were used for the actual artifact load; only Python had an S3 proxy destination configured. Proxy latency is not compared.
- The retired `/api/2.0/mlflow/experiments/list` route returns 404 from both servers in this revision. The `experiments_list` workload therefore uses the supported `experiments/search` route with `view_type=ACTIVE_ONLY`, matching current client list semantics.
- Rust's RSS AC is NOT MET under the stated ten-minute-bin rule. Its growth was strongly front-loaded (39.9, 44.0, 45.9 MiB in the first 30 minutes) and nearly plateaued thereafter (46.4, 46.6, 46.7 MiB), but remained monotonically increasing.
- Process-tree RSS sums shared pages once per process. This is intentional for total deployment RSS but can over-count pages shared by Python's uvicorn workers versus PSS.
- `minio/minio:latest` and `minio/mc:latest` identify the locally cached images used; pin image digests for cross-machine reproduction.
