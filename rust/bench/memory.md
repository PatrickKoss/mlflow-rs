# T14.1 tracking-server memory baseline

## Verdict

Rust reduced total process-tree RSS by **106.50x idle** and **67.33x under load** versus Python's four uvicorn workers. The ≥5x target was **MET** (both idle and loaded comparisons).

## Results

| Target | Idle mean MiB (60 s) | Idle min–max MiB | Loaded mean MiB (last 10 min) | Loaded min–max MiB |
|---|---:|---:|---:|---:|
| Python (4 uvicorn workers) | 2976.21 | 2491.79–3073.10 | 3144.75 | 3109.39–3154.70 |
| Rust release binary | 27.95 | 27.95–27.95 | 46.70 | 36.33–46.70 |
| Python / Rust factor | **106.50x** | — | **67.33x** | — |

## Measurement method

Measured on WSL2 `5.15.167.4-microsoft-standard-WSL2` with 46.7 GiB RAM. This host uses cgroup v1 and the benchmark lives in the broad `/init.scope`, so an isolated `memory.current` value is unavailable. The fallback sums `VmRSS` from `/proc/<pid>/status` for the launch process and every descendant every 10 seconds. The idle value is the 60-second pre-load mean; loaded is the final ten minutes of each 3,600-second mixed workload. Python was required to and did use four uvicorn workers.

Because RSS counts a shared page in every process mapping it, the Python total can exceed whole-tree PSS. RSS was chosen because T14.1 asks for RSS and deployment capacity must reserve the workers' resident mappings; both targets used the identical sampler.

## Reproduction

See `rust/bench/soak.md` for the exact command, infrastructure, workload totals, and per-endpoint results. The same invocation produces this report.
