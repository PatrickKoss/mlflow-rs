# T23.3 jobs + native-engine benchmark summary

This is raw material for T23.5, not the final Phase 23 report. Python and Rust ran
serially on PostgreSQL 16 + MinIO with a fresh database and artifact prefix per
target. Python used four uvicorn workers and its real Huey subprocess runtime; Rust
used the release server and one native worker subprocess per claimed job. Every model
call went through the loopback deterministic provider; no live provider was reachable.
RSS, CPU, process, and thread samples cover the server's whole process tree at one-second
intervals, including Python job-runtime children and Rust native-worker subprocesses.
Online jobs were activated only through registered scorer + public online-config APIs
and the real minute scheduler; a read-only jobs-table query discovered their IDs so the
public GET jobs API could measure them. It did not create or mutate jobs.
Each online config was deactivated immediately after the first expected scheduler wave
was discovered. Public terminal polling started as soon as each job ID became available.
Leak-applicable cells sampled a five-to-60-second post-completion tail until the whole
process tree met the bounded flat-tail rule; reaching 60 seconds still failing was fatal.

## Chosen matrix

| Cell | Shape | Kinds | Jobs by kind | Rows/job | Rationale |
| --- | --- | --- | --- | ---: | --- |
| `evaluation-high-fanout` | high-fanout | invoke_genai_evaluate | invoke_genai_evaluate=1,000 | 1 | subprocess churn and leak pressure over a small corpus |
| `evaluation-large-payload` | large-payload | invoke_genai_evaluate | invoke_genai_evaluate=10 | 1,000 | about ten jobs processing a 1,000-row corpus |
| `scorer-high-fanout` | high-fanout | invoke_scorer | invoke_scorer=1,000 | 1 | subprocess churn and leak pressure over a small corpus |
| `scorer-large-payload` | large-payload | invoke_scorer | invoke_scorer=10 | 1,000 | about ten jobs processing a 1,000-row corpus |
| `issue-discovery-high-fanout` | high-fanout | invoke_issue_detection | invoke_issue_detection=1,000 | 1 | subprocess churn and leak pressure over a small corpus |
| `issue-discovery-large-payload` | large-payload | invoke_issue_detection | invoke_issue_detection=10 | 1,000 | about ten jobs processing a 1,000-row corpus |
| `prompt-optimization-high-fanout` | high-fanout | optimize_prompts | optimize_prompts=1,000 | 1 | subprocess churn and leak pressure over a small corpus |
| `prompt-optimization-large-payload` | large-payload | optimize_prompts | optimize_prompts=10 | 1,000 | about ten jobs processing a 1,000-row corpus |
| `online-high-fanout` | high-fanout | run_online_trace_scorer, run_online_session_scorer | run_online_trace_scorer=1,000, run_online_session_scorer=1,000 | 1 | subprocess churn and leak pressure over a small corpus |
| `online-large-payload` | large-payload | run_online_trace_scorer, run_online_session_scorer | run_online_trace_scorer=10, run_online_session_scorer=10 | 1,000 | about ten jobs processing a 1,000-row corpus |
| `mixed-burst` | burst | invoke_genai_evaluate, invoke_scorer, run_online_trace_scorer, run_online_session_scorer, invoke_issue_detection, optimize_prompts | invoke_genai_evaluate=100, invoke_scorer=100, run_online_trace_scorer=100, run_online_session_scorer=100, invoke_issue_detection=100, optimize_prompts=100 | 1 | all pools receive much more work than worker concurrency |
| `mixed-steady-drip` | steady-drip | invoke_genai_evaluate, invoke_scorer, run_online_trace_scorer, run_online_session_scorer, invoke_issue_detection, optimize_prompts | invoke_genai_evaluate=20, invoke_scorer=20, invoke_issue_detection=20, optimize_prompts=20, run_online_trace_scorer=2, run_online_session_scorer=2 | 1 | submission rate stays at or below the smallest pool capacity |

## invoke_genai_evaluate

| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `evaluation-high-fanout` | 1,000 | 85.2/14743.1 | 355.52/628.57/650.28/654.93 | 0.21/0.22/0.22/0.22 | 622.53/0.22 | 8.16/0.20 | 8723.4/97.6 | 2632.11/5.81 | 0/0 | PASS | PASS/PASS |
| `evaluation-large-payload` | 10 | 9.1/54.8 | 64.34/65.42/65.42/65.42 | 8.52/10.95/10.95/10.95 | 2.73/0.21 | 64.25/10.73 | 9815.3/773.0 | 249.09/4.18 | 0/0 | PASS | N/A/N/A |
| `mixed-burst` | 100 | 16.3/134.4 | 44.41/50.81/52.25/52.97 | 0.62/1.02/1.03/1.22 | 37.67/1.02 | 20.50/0.20 | 19814.7/432.5 | 570.53/8.02 | 0/0 | PASS | N/A/N/A |
| `mixed-steady-drip` | 20 | 9.8/10.8 | 5.38/6.42/7.49/7.49 | 0.81/1.02/1.22/1.22 | 1.13/1.02 | 5.99/0.20 | 7388.7/335.9 | 19.44/0.98 | 0/0 | PASS | N/A/N/A |

## invoke_scorer

| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `scorer-high-fanout` | 1,000 | 121.6/14078.6 | 247.65/442.18/456.65/460.22 | 2.38/3.24/3.39/3.42 | 438.00/3.22 | 5.46/0.21 | 8860.6/224.5 | 1766.57/7.86 | 0/0 | PASS | PASS/PASS |
| `scorer-large-payload` | 10 | 110.8/321.9 | 4.99/5.42/5.42/5.42 | 0.86/1.86/1.86/1.86 | 1.49/0.25 | 5.17/1.61 | 8914.4/274.0 | 1.03/0.49 | 0/0 | PASS | N/A/N/A |
| `mixed-burst` | 100 | 16.3/134.4 | 9.89/18.23/19.17/19.99 | 0.61/1.21/1.22/1.22 | 10.83/1.02 | 9.69/0.20 | 19814.7/432.5 | 570.53/8.02 | 0/0 | PASS | N/A/N/A |
| `mixed-steady-drip` | 20 | 9.8/10.8 | 4.33/4.58/6.80/6.80 | 0.61/1.01/1.21/1.21 | 0.63/1.01 | 4.15/0.00 | 7388.7/335.9 | 19.44/0.98 | 0/0 | PASS | N/A/N/A |

## run_online_trace_scorer

| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `online-high-fanout` | 1,000 | 60.5/1882.9 | 519.85/950.88/983.13/991.05 | 31.80/31.82/31.82/31.82 | 946.84/31.82 | 5.95/0.00 | 8119.1/369.7 | 3415.17/18.55 | 0/0 | PASS | PASS/PASS |
| `online-large-payload` | 10 | 32.4/18.2 | 14.41/18.40/18.40/18.40 | 32.99/32.99/32.99/32.99 | 14.51/32.99 | 4.96/0.00 | 8184.6/325.7 | 3.13/0.30 | 0/0 | PASS | N/A/N/A |
| `mixed-burst` | 100 | 16.3/134.4 | 95.13/164.61/169.30/169.63 | 44.41/44.62/44.62/44.62 | 157.60/44.62 | 9.40/0.20 | 19814.7/432.5 | 570.53/8.02 | 0/0 | PASS | N/A/N/A |
| `mixed-steady-drip` | 2 | 1.0/1.1 | 112.90/120.69/120.69/120.69 | 111.37/111.58/111.58/111.58 | 117.61/111.58 | 3.08/0.00 | 7388.7/335.9 | 19.44/0.98 | 0/0 | PASS | N/A/N/A |

## run_online_session_scorer

| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `online-high-fanout` | 1,000 | 60.5/1882.9 | 523.15/952.89/984.94/991.87 | 31.85/31.86/31.86/31.86 | 948.77/31.86 | 5.76/0.00 | 8119.1/369.7 | 3415.17/18.55 | 0/0 | PASS | PASS/PASS |
| `online-large-payload` | 10 | 32.4/18.2 | 14.51/18.49/18.49/18.49 | 32.99/32.99/32.99/32.99 | 14.45/32.99 | 4.96/0.00 | 8184.6/325.7 | 3.13/0.30 | 0/0 | PASS | N/A/N/A |
| `mixed-burst` | 100 | 16.3/134.4 | 96.91/162.71/168.57/168.59 | 44.42/44.63/44.63/44.63 | 155.52/44.63 | 9.40/0.20 | 19814.7/432.5 | 570.53/8.02 | 0/0 | PASS | N/A/N/A |
| `mixed-steady-drip` | 2 | 1.0/1.1 | 112.90/122.12/122.12/122.12 | 111.37/111.58/111.58/111.58 | 119.05/111.58 | 3.07/0.00 | 7388.7/335.9 | 19.44/0.98 | 0/0 | PASS | N/A/N/A |

## invoke_issue_detection

| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `issue-discovery-high-fanout` | 1,000 | 78.9/4871.8 | 333.06/521.10/533.20/535.55 | 0.21/0.42/0.62/0.62 | 514.37/0.41 | 8.48/0.20 | 9324.7/292.9 | 2824.88/8.68 | 0/0 | PASS | PASS/PASS |
| `issue-discovery-large-payload` | 10 | 20.5/98.8 | 25.68/28.68/28.68/28.68 | 5.85/6.07/6.07/6.07 | 0.78/0.22 | 28.15/5.85 | 9561.7/676.7 | 56.85/1.49 | 0/0 | PASS | N/A/N/A |
| `mixed-burst` | 100 | 16.3/134.4 | 53.82/64.15/67.30/67.88 | 0.24/1.02/1.02/1.03 | 52.31/0.82 | 24.07/0.21 | 19814.7/432.5 | 570.53/8.02 | 0/0 | PASS | N/A/N/A |
| `mixed-steady-drip` | 20 | 9.8/10.8 | 6.22/6.49/6.84/6.84 | 0.62/1.02/1.02/1.02 | 0.48/1.02 | 6.08/0.20 | 7388.7/335.9 | 19.44/0.98 | 0/0 | PASS | N/A/N/A |

## optimize_prompts

| Cell | N | Py/Rust jobs/min | Py wall p50/p95/p99/max s | Rust wall p50/p95/p99/max s | Py/Rust queue p95 s | Py/Rust exec p95 s | Py/Rust peak RSS MiB | Py/Rust CPU-s | Errors Py/Rust | Eq | Leak Py/Rust |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prompt-optimization-high-fanout` | 1,000 | 20.1/857.6 | 1529.10/2729.50/2822.07/2845.22 | 30.52/56.25/58.52/59.03 | 2724.35/56.15 | 7.14/0.21 | 5992.9/342.0 | 10652.61/27.14 | 0/0 | PASS | PASS/PASS |
| `prompt-optimization-large-payload` | 10 | 3.4/1.2 | 106.97/178.15/178.15/178.15 | 305.74/509.16/509.16/509.16 | 143.09/407.49 | 37.02/101.90 | 6046.0/336.5 | 443.42/3.37 | 0/0 | PASS | N/A/N/A |
| `mixed-burst` | 100 | 16.3/134.4 | 241.97/348.82/358.74/359.53 | 4.25/6.88/7.26/7.27 | 343.88/6.87 | 19.50/0.20 | 19814.7/432.5 | 570.53/8.02 | 0/0 | PASS | N/A/N/A |
| `mixed-steady-drip` | 20 | 9.8/10.8 | 5.25/5.51/6.87/6.87 | 0.82/1.22/1.22/1.22 | 0.44/1.02 | 5.17/0.20 | 7388.7/335.9 | 19.44/0.98 | 0/0 | PASS | N/A/N/A |

## Burst queueing and fairness

- python: max/min per-kind queue-p95 ratio 31.76; first-half completion shares {"invoke_genai_evaluate": 0.17333333333333334, "invoke_issue_detection": 0.15, "invoke_scorer": 0.24, "optimize_prompts": 0.04, "run_online_session_scorer": 0.2, "run_online_trace_scorer": 0.19666666666666666}.
  - `invoke_genai_evaluate` queue p95 37.67s; execution p95 20.50s.
  - `invoke_issue_detection` queue p95 52.31s; execution p95 24.07s.
  - `invoke_scorer` queue p95 10.83s; execution p95 9.69s.
  - `optimize_prompts` queue p95 343.88s; execution p95 19.50s.
  - `run_online_session_scorer` queue p95 155.52s; execution p95 9.40s.
  - `run_online_trace_scorer` queue p95 157.60s; execution p95 9.40s.
- rust: max/min per-kind queue-p95 ratio 54.25; first-half completion shares {"invoke_genai_evaluate": 0.3233333333333333, "invoke_issue_detection": 0.31333333333333335, "invoke_scorer": 0.23666666666666666, "optimize_prompts": 0.12666666666666668, "run_online_session_scorer": 0.0, "run_online_trace_scorer": 0.0}.
  - `invoke_genai_evaluate` queue p95 1.02s; execution p95 0.20s.
  - `invoke_issue_detection` queue p95 0.82s; execution p95 0.21s.
  - `invoke_scorer` queue p95 1.02s; execution p95 0.20s.
  - `optimize_prompts` queue p95 6.87s; execution p95 0.20s.
  - `run_online_session_scorer` queue p95 44.63s; execution p95 0.20s.
  - `run_online_trace_scorer` queue p95 44.62s; execution p95 0.20s.

## Leak checks

- `evaluation-high-fanout/python`: PASS; RSS 4301.2->4631.0 MiB (27.82 MiB/min), processes 15.0->15.0, threads 702.0->714.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 8).
- `evaluation-high-fanout/rust`: PASS; RSS 45.7->55.7 MiB (54.00 MiB/min), processes 1.0->1.0, threads 25.0->25.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 2).
- `scorer-high-fanout/python`: PASS; RSS 5021.1->5142.2 MiB (14.52 MiB/min), processes 15.0->15.0, threads 731.0->719.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 8).
- `scorer-high-fanout/rust`: PASS; RSS 190.7->170.4 MiB (-107.95 MiB/min), processes 1.0->1.0, threads 25.0->25.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 2).
- `issue-discovery-high-fanout/python`: PASS; RSS 5142.7->5147.1 MiB (0.35 MiB/min), processes 15.0->15.0, threads 709.0->723.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 8).
- `issue-discovery-high-fanout/rust`: PASS; RSS 233.5->204.6 MiB (-89.77 MiB/min), processes 1.0->1.0, threads 25.0->25.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 2).
- `prompt-optimization-high-fanout/python`: PASS; RSS 5183.0->5188.4 MiB (0.11 MiB/min), processes 15.0->15.0, threads 715.7->709.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 8).
- `prompt-optimization-high-fanout/rust`: PASS; RSS 342.0->289.4 MiB (-40.99 MiB/min), processes 1.0->1.0, threads 25.0->25.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 2).
- `online-high-fanout/python`: PASS; RSS 4342.2->4419.8 MiB (4.57 MiB/min), processes 15.0->15.0, threads 804.0->853.7; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/7 (thread allowance 9).
- `online-high-fanout/rust`: PASS; RSS 299.8->305.8 MiB (4.63 MiB/min), processes 1.0->1.0, threads 25.0->25.0; monotonic RSS/process/thread growth False/False/False; settled RSS/process/thread spread 0.0 MiB/0/0 (thread allowance 2).

## Rust-slower cells and anomalies

- `prompt-optimization-large-payload/optimize_prompts`: Rust p95 509.16s vs Python 178.15s.
- `online-large-payload/run_online_trace_scorer`: Rust p95 32.99s vs Python 18.40s.
- `online-large-payload/run_online_session_scorer`: Rust p95 32.99s vs Python 18.49s.

## Raw result inventory

- `evaluation-high-fanout-python.json`
- `evaluation-high-fanout-rust.json`
- `evaluation-large-payload-python.json`
- `evaluation-large-payload-rust.json`
- `issue-discovery-high-fanout-python.json`
- `issue-discovery-high-fanout-rust.json`
- `issue-discovery-large-payload-python.json`
- `issue-discovery-large-payload-rust.json`
- `mixed-burst-python.json`
- `mixed-burst-rust.json`
- `mixed-steady-drip-python.json`
- `mixed-steady-drip-rust.json`
- `online-high-fanout-python.json`
- `online-high-fanout-rust.json`
- `online-large-payload-python.json`
- `online-large-payload-rust.json`
- `prompt-optimization-high-fanout-python.json`
- `prompt-optimization-high-fanout-rust.json`
- `prompt-optimization-large-payload-python.json`
- `prompt-optimization-large-payload-rust.json`
- `scorer-high-fanout-python.json`
- `scorer-high-fanout-rust.json`
- `scorer-large-payload-python.json`
- `scorer-large-payload-rust.json`
