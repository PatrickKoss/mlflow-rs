# T23.5 mixed GenAI soak summary

Python and Rust ran serially against fresh PostgreSQL 16 databases, MinIO
prefixes, and local archive repositories. The same seed fixed request order,
payloads, and ten-minute schedule. Warm-up is excluded. Public terminal polls
are control observations outside the 10,000 scheduled primary requests.

## Traffic mix

| Family | Scheduled requests | Mix |
| --- | ---: | ---: |
| `assistant_requests` | 500 | 5.0% |
| `dataset_upserts` | 2,000 | 20.0% |
| `evaluation_jobs` | 500 | 5.0% |
| `gateway_chat` | 2,500 | 25.0% |
| `gateway_streams` | 500 | 5.0% |
| `labeling_reads` | 1,750 | 17.5% |
| `review_queue_reads` | 1,750 | 17.5% |
| `scorer_jobs` | 500 | 5.0% |

A 1,000-trace archive pass ran in the background. Assistant count includes
both session POSTs and streamed GETs (250 complete sessions). Gateway's 30%
share contains 500 streams. Jobs split evenly between evaluation and scorer
invocations; every terminal state is in the equivalence proof.

## Acceptance criteria

| Target | Requests | Errors | Error rate | Jobs succeeded | Archive traces | RSS slope MiB/h | Monotonic growth | Eq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Python | 10,000 | 0 | 0.00000% | 1,000/1,000 | 1,000 | -3299.19 | no — PASS | PASS |
| Rust | 10,000 | 0 | 0.00000% | 1,000/1,000 | 1,000 | +10.23 | no — PASS | PASS |

The RSS rule matches T14.2 at this soak scale: failure requires every
consecutive one-minute mean to be non-decreasing and the final mean to exceed
the first by more than 5%.

## Per-family HTTP results

| Family | Py p50/p95/p99 ms | Rust p50/p95/p99 ms | Py/Rust RPS | Py/Rust errors |
| --- | --- | --- | --- | --- |
| `assistant` | 23.89/58.48/80.64 | 21.63/47.13/57.32 | 0.79/0.79 | 0/0 |
| `datasets` | 10.08/19.32/41.02 | 4.93/6.08/7.23 | 3.14/3.17 | 0/0 |
| `evaluation_jobs` | 19.35/33.48/64.22 | 10.56/12.35/13.43 | 0.79/0.79 | 0/0 |
| `gateway` | 39.45/86.95/140.08 | 22.56/30.84/35.74 | 4.71/4.76 | 0/0 |
| `label_schemas` | 6.78/15.18/44.87 | 1.92/2.54/2.90 | 2.75/2.78 | 0/0 |
| `review_queues` | 7.47/15.87/37.30 | 2.20/2.91/3.39 | 2.75/2.78 | 0/0 |
| `scorer_jobs` | 7.17/14.01/29.30 | 3.38/4.36/5.52 | 0.79/0.79 | 0/0 |

## RSS + CPU over time

- Python one-minute mean RSS MiB: `6748.0 → 6904.8 → 7018.0 → 7083.7 → 7453.8 → 7353.6 → 7491.3 → 6984.3 → 6849.3 → 6885.8 → 5050.7`
- Python CPU: 424.63 process-tree CPU-s; RSS slope -3299.19 MiB/h; PASS.
- Rust one-minute mean RSS MiB: `57.6 → 58.5 → 58.8 → 59.0 → 58.9 → 59.3 → 59.6 → 59.7 → 59.4 → 59.5 → 59.3`
- Rust CPU: 42.84 process-tree CPU-s; RSS slope +10.23 MiB/h; PASS.

The Python final bin includes the partial settlement tail after load ended;
the regression uses every sampled bin on both targets. Rust's positive slope
is reported honestly, but the minute means are not monotonic and the final
mean is only 3.10% above the first, below the 5% failure threshold.

## Rust-slower cells and anomalies

- No soak family regressed at p50, p95, p99, or throughput.
- Two isolated submit maxima were slower on Rust despite lower Rust p99:
  evaluation 339.61 vs 233.69 ms and scorer 104.26 vs 53.57 ms.
- Rust's observed job queue p50 was higher because its fast jobs often skipped
  the polled RUNNING state; the harness conservatively attributes the entire
  submit-to-terminal interval to queue time in that case.
- Archive traces/s includes watcher activation and is a blended-soak cadence,
  not the isolated archival capacity result reported by T23.4.

## Raw result inventory

- `mixed-soak-python.json`
- `mixed-soak-rust.json`
