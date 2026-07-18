# MLflow Python vs Rust tracking-server benchmarks

This directory contains the T13.3 deterministic dataset generator and the six-scenario
benchmark runner. SQLite runs give Python and Rust byte-identical copies of one seed database,
boot one server at a time, perform unmeasured warmups, and then measure sequential requests with
no synthetic background load. OTLP uses binary protobuf.

Run commands from the repository root. Build the release binary first:

```bash
cargo build --manifest-path rust/Cargo.toml --release
```

## Laptop / WSL2 SQLite run

The following is a useful medium run. Adjust every entity count independently; `--seed` makes
the generated values and relationships deterministic.

```bash
uv run python rust/bench/seed.py --db /tmp/t133-bench.db \
    --runs 20000 --metrics-per-run 4 --history-points 100 \
    --traces 50000 --spans-per-trace 4 --model-versions 5000 \
    --experiments 40 --prompt-fraction 0.30 --seed 42 \
    --metadata /tmp/t133-seed.json

uv run python rust/bench/bench.py --db /tmp/t133-bench.db \
    --seed-metadata /tmp/t133-seed.json --iterations 30 \
    --warmup-iterations 3 --deep-pages 25 \
    --results rust/bench/RESULTS.md
```

Use a fresh database path. The generator migrates the database with MLflow's real Alembic chain,
then bulk-inserts runs, full metric histories and latest metrics, traces, spans and extracted span
attributes, registered models, model versions, and prompt tags. It does not remove or reset an
existing database.

Seed flags:

- `--runs`, `--experiments`: run and experiment cardinality.
- `--metrics-per-run`, `--history-points`: metric keys and history rows per key per run.
- `--traces`, `--spans-per-trace`: trace and span cardinality. Five indexed attributes are
  generated per span.
- `--model-versions`: versions distributed across approximately one registered model per ten
  versions.
- `--prompt-fraction`: fraction of registered models tagged as prompts, from 0 through 1.
- `--seed`: deterministic PRNG seed.
- `--metadata`: JSON containing the requested scale, actual row counts, and seeding times. Pass
  this file to the runner so `RESULTS.md` records the seed wall time.

Runner flags:

- `--iterations`: measured requests or OTLP batches per scenario (default 30).
- `--warmup-iterations`: unmeasured requests per scenario (default 3).
- `--deep-pages`: pages in the single deep-pagination walk; minimum 20 (default 25).
- `--otlp-spans-per-batch`: protobuf spans per measured POST (default 100).
- `-k SCENARIO`: run one named scenario; repeat the flag for more than one.
- `--rust-bin`: release binary path. The runner deliberately does not fall back to a debug build.
- `--results`: Markdown report destination.

The scenarios are `run_search_metric_filter`, `run_search_deep_pagination`,
`metric_history_bulk_interval`, `trace_search_span_filter`, `otlp_ingest_throughput`, and
`registry_search_prompt_antijoin`.

## Full-scale Postgres run

Use fresh databases on a dedicated machine. Postgres storage depends on its version, fill factor,
and index settings, so measure the result rather than claiming a flag set is exactly 100 GB. This
candidate scale is intentionally large; tune it on the target host until
`pg_database_size('mlflow_bench_seed')` is near the desired size.

```bash
createdb mlflow_bench_seed
export BENCH_SEED='postgresql://mlflow:mlflow@localhost/mlflow_bench_seed'
uv run python rust/bench/seed.py --db "$BENCH_SEED" \
    --runs 750000 --metrics-per-run 5 --history-points 100 \
    --traces 2000000 --spans-per-trace 5 --model-versions 100000 \
    --experiments 500 --seed 42 --metadata /tmp/t133-seed.json

psql "$BENCH_SEED" -Atc \
    "SELECT pg_size_pretty(pg_database_size('mlflow_bench_seed'));"

createdb mlflow_bench_python
createdb mlflow_bench_rust
pg_dump --format=custom "$BENCH_SEED" --file=/tmp/mlflow-bench.dump
pg_restore --dbname=mlflow_bench_python --jobs=8 /tmp/mlflow-bench.dump
pg_restore --dbname=mlflow_bench_rust --jobs=8 /tmp/mlflow-bench.dump
psql postgresql://mlflow:mlflow@localhost/mlflow_bench_python -c 'ANALYZE;'
psql postgresql://mlflow:mlflow@localhost/mlflow_bench_rust -c 'ANALYZE;'

cargo build --manifest-path rust/Cargo.toml --release
uv run python rust/bench/bench.py \
    --python-db-uri postgresql://mlflow:mlflow@localhost/mlflow_bench_python \
    --rust-db-uri postgresql://mlflow:mlflow@localhost/mlflow_bench_rust \
    --seed-metadata /tmp/t133-seed.json --iterations 200 --deep-pages 100 \
    --results rust/bench/RESULTS.md
```

Two restored databases are required because OTLP writes new spans. Pointing both servers at one
database would make the second measurement start from mutated state and could introduce lock
contention. Keep the servers on the same host and storage class, and run scenarios with no other
load.
