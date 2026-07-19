# T23.2 CRUD + read-path benchmark summary

This is raw material for T23.5, not the final Phase 23 report. Python and Rust
ran serially on PostgreSQL 16 + MinIO, with a fresh DB and artifact prefix per target.
Every read family had a deterministic 10,000-row corpus; warm-up requests are excluded.
Both targets used pool_size=32 + max_overflow=8; PostgreSQL max_connections was 400.
Final Python artifacts came from two serial target slices (datasets through review
queues, then prompt optimization through gateway); Rust used one all-family target.
No target loads overlapped, every slice used a fresh DB/prefix, and every per-cell
resource series is complete.
Compare absolute Python RSS across the slice boundary with that retained-memory caveat.
The host exposed no supported cgroup pids.current file, so that raw field is null;
the pre-cell /proc process count and complete target-tree PID list are still recorded.

## Chosen matrix

| Cell | Payload | Clients | Mix | Requests | Rationale |
| --- | --- | ---: | --- | ---: | --- |
| `small-c1-wh` | small | 1 | write-heavy | 10,000 | single-client write baseline |
| `small-c128-rh` | small | 128 | read-heavy | 10,000 | high-contention read path |
| `large-c16-wh` | large | 16 | write-heavy | 1,000 | mid-concurrency large-write pressure |
| `large-c128-rh` | large | 128 | read-heavy | 1,000 | high-concurrency large read path |

Large payload definitions:

- `datasets`: 8-record upsert with 64 KiB outputs per record (about 512 KiB JSON).
- `scorers`: 64 KiB serialized scorer JSON description.
- `issues`: 64 KiB issue description.
- `label_schemas`: maximum valid schema: 250-char name, 1000-char instruction, and ten 64-char options.
- `review_queues`: ten 250-char users plus 100 schema/item references (about 6-8 KiB JSON).
- `prompt_optimization`: 5 KiB optimizer_config_json (bounded by the 6,000-char run-param cap).
- `gateway_admin`: 64 KiB obvious-fake secret_value through AES-GCM envelope encryption.

## datasets

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 49.97/79.98/240.03/299.90 | 3.48/4.69/33.43/46.64 | 17.5/248.6 | 0/0 | 3126.2/3123.1 | 36.5/35.7 | 113.86/5.83 | PASS |
| `small-c128-rh` | 10,000 | 1448.36/4959.93/5830.48/179013.57 | 29.15/70.72/114.15/176.67 | 55.8/3220.2 | 0/0 | 4218.0/4095.3 | 60.8/53.8 | 792.29/3.91 | PASS |
| `large-c16-wh` | 1,000 | 101.05/292.85/373.10/412.88 | 16.45/35.37/70.59/171.35 | 122.9/900.9 | 0/0 | 4182.8/4129.6 | 136.3/106.2 | 23.65/2.18 | PASS |
| `large-c128-rh` | 1,000 | 1576.02/6207.57/7758.59/8584.15 | 40.69/99.15/114.30/120.36 | 50.9/2447.8 | 0/0 | 4357.5/4225.1 | 146.0/132.4 | 79.21/0.61 | PASS |

## scorers

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 50.00/80.19/90.19/160.11 | 2.54/4.91/7.12/15.75 | 18.4/362.1 | 0/0 | 4230.6/4188.7 | 146.0/145.0 | 79.17/4.93 | PASS |
| `small-c128-rh` | 10,000 | 368.95/1770.77/2029.98/2523.73 | 41.30/53.56/111.32/138.49 | 176.3/3000.4 | 0/0 | 4161.8/4025.3 | 146.2/144.5 | 259.94/7.98 | PASS |
| `large-c16-wh` | 1,000 | 21.99/110.00/269.13/360.33 | 4.79/8.08/10.07/13.17 | 423.8/3157.3 | 0/0 | 4038.6/4035.6 | 151.9/148.5 | 9.00/0.56 | PASS |
| `large-c128-rh` | 1,000 | 510.45/1884.96/2109.96/2370.38 | 42.52/56.98/68.16/76.48 | 170.7/2750.8 | 0/0 | 4042.7/4038.0 | 151.9/151.4 | 25.52/0.97 | PASS |

## issues

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 49.97/59.96/60.08/139.94 | 2.15/5.32/10.23/67.45 | 19.7/385.4 | 0/0 | 4039.5/4030.5 | 151.0/151.0 | 44.86/3.61 | PASS |
| `small-c128-rh` | 10,000 | 108.95/243.19/299.11/358.80 | 35.76/86.98/98.11/165.89 | 1005.0/3029.1 | 0/0 | 4033.9/4017.0 | 152.3/151.8 | 39.18/2.19 | PASS |
| `large-c16-wh` | 1,000 | 14.48/64.94/76.28/83.22 | 4.67/7.08/10.57/21.31 | 813.3/3114.8 | 0/0 | 4029.3/4025.5 | 152.3/152.3 | 4.75/0.47 | PASS |
| `large-c128-rh` | 1,000 | 121.09/254.04/298.60/364.88 | 38.23/91.12/102.28/124.79 | 925.4/2687.7 | 0/0 | 4043.3/4038.6 | 154.6/153.5 | 3.96/0.26 | PASS |

## label_schemas

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 49.97/59.93/60.05/149.98 | 2.28/3.74/4.08/6.49 | 19.7/413.0 | 0/0 | 4045.0/4034.6 | 154.6/154.6 | 44.94/4.03 | PASS |
| `small-c128-rh` | 10,000 | 98.01/308.50/379.15/461.74 | 31.74/39.30/44.56/54.49 | 966.1/3971.7 | 0/0 | 4042.9/4023.1 | 154.6/152.4 | 42.41/3.64 | PASS |
| `large-c16-wh` | 1,000 | 50.18/60.45/70.06/139.88 | 3.40/4.93/6.24/7.52 | 298.5/4529.8 | 0/0 | 4024.1/4023.5 | 151.8/151.8 | 3.47/0.33 | PASS |
| `large-c128-rh` | 1,000 | 113.58/221.65/232.35/298.29 | 32.01/40.07/46.56/53.04 | 952.4/3741.2 | 0/0 | 4023.7/4023.0 | 151.8/151.8 | 4.23/0.35 | PASS |

## review_queues

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 49.99/60.07/69.95/130.09 | 3.18/4.34/5.10/48.40 | 19.0/330.6 | 0/0 | 4023.7/4021.9 | 151.8/151.8 | 55.52/5.88 | PASS |
| `small-c128-rh` | 10,000 | 146.96/299.50/359.11/488.79 | 30.03/50.85/67.04/87.10 | 841.8/4037.7 | 0/0 | 4029.1/4019.8 | 151.8/142.3 | 49.24/2.92 | PASS |
| `large-c16-wh` | 1,000 | 60.08/1039.67/1209.50/1389.75 | 55.94/158.05/163.34/171.58 | 99.9/210.3 | 0/0 | 4025.7/4019.0 | 139.3/139.3 | 30.83/1.66 | PASS |
| `large-c128-rh` | 1,000 | 80.02/510.47/809.49/1787.62 | 26.95/74.81/261.24/317.62 | 492.6/1867.8 | 0/0 | 4022.0/4019.5 | 140.1/139.7 | 6.70/0.45 | PASS |

## prompt_optimization

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 49.97/610.20/729.98/850.03 | 2.24/17.53/18.87/67.49 | 9.1/254.4 | 0/0 | 3365.7/3155.1 | 201.4/183.8 | 224.54/19.35 | PASS |
| `small-c128-rh` | 10,000 | 10619.19/15539.83/16314.97/19234.05 | 450.93/1264.10/1442.21/5112.92 | 12.0/240.2 | 0/0 | 3493.7/3263.3 | 441.8/415.9 | 3674.14/932.04 | PASS |
| `large-c16-wh` | 1,000 | 69.97/1280.01/2279.98/2450.04 | 4.02/30.23/38.03/51.13 | 70.0/1817.6 | 0/0 | 3734.0/3447.3 | 424.5/422.7 | 32.84/2.88 | PASS |
| `large-c128-rh` | 1,000 | 7598.77/22749.95/29306.85/29956.97 | 449.50/1238.97/1359.10/4027.92 | 11.5/244.5 | 0/0 | 3507.0/3282.1 | 442.1/431.7 | 365.86/86.85 | PASS |

## gateway_admin

| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `small-c1-wh` | 10,000 | 50.01/1179.92/1249.97/1569.98 | 2.79/42.22/42.91/70.17 | 5.3/88.4 | 0/0 | 3276.4/3272.2 | 418.2/418.2 | 923.14/100.03 | PASS |
| `small-c128-rh` | 10,000 | 90.31/797.67/8770.25/35523.11 | 10.89/30.12/57.76/83.54 | 232.3/9040.0 | 0/0 | 3319.7/3305.9 | 418.2/398.3 | 177.66/11.73 | PASS |
| `large-c16-wh` | 1,000 | 259.99/540.41/730.10/1090.19 | 45.36/80.65/121.56/124.06 | 55.8/300.6 | 0/0 | 3355.7/3351.5 | 388.4/384.4 | 50.80/37.30 | PASS |
| `large-c128-rh` | 1,000 | 214.77/559.36/1357.67/1975.54 | 16.34/76.91/101.46/125.64 | 436.7/4044.4 | 0/0 | 3413.7/3387.5 | 383.3/383.3 | 8.99/4.16 | PASS |

## Rust-slower cells

No cell had Rust p95 above Python p95.

## Raw result inventory

- `datasets-large-c128-rh-python.json`
- `datasets-large-c128-rh-rust.json`
- `datasets-large-c16-wh-python.json`
- `datasets-large-c16-wh-rust.json`
- `datasets-small-c1-wh-python.json`
- `datasets-small-c1-wh-rust.json`
- `datasets-small-c128-rh-python.json`
- `datasets-small-c128-rh-rust.json`
- `gateway_admin-large-c128-rh-python.json`
- `gateway_admin-large-c128-rh-rust.json`
- `gateway_admin-large-c16-wh-python.json`
- `gateway_admin-large-c16-wh-rust.json`
- `gateway_admin-small-c1-wh-python.json`
- `gateway_admin-small-c1-wh-rust.json`
- `gateway_admin-small-c128-rh-python.json`
- `gateway_admin-small-c128-rh-rust.json`
- `issues-large-c128-rh-python.json`
- `issues-large-c128-rh-rust.json`
- `issues-large-c16-wh-python.json`
- `issues-large-c16-wh-rust.json`
- `issues-small-c1-wh-python.json`
- `issues-small-c1-wh-rust.json`
- `issues-small-c128-rh-python.json`
- `issues-small-c128-rh-rust.json`
- `label_schemas-large-c128-rh-python.json`
- `label_schemas-large-c128-rh-rust.json`
- `label_schemas-large-c16-wh-python.json`
- `label_schemas-large-c16-wh-rust.json`
- `label_schemas-small-c1-wh-python.json`
- `label_schemas-small-c1-wh-rust.json`
- `label_schemas-small-c128-rh-python.json`
- `label_schemas-small-c128-rh-rust.json`
- `prompt_optimization-large-c128-rh-python.json`
- `prompt_optimization-large-c128-rh-rust.json`
- `prompt_optimization-large-c16-wh-python.json`
- `prompt_optimization-large-c16-wh-rust.json`
- `prompt_optimization-small-c1-wh-python.json`
- `prompt_optimization-small-c1-wh-rust.json`
- `prompt_optimization-small-c128-rh-python.json`
- `prompt_optimization-small-c128-rh-rust.json`
- `review_queues-large-c128-rh-python.json`
- `review_queues-large-c128-rh-rust.json`
- `review_queues-large-c16-wh-python.json`
- `review_queues-large-c16-wh-rust.json`
- `review_queues-small-c1-wh-python.json`
- `review_queues-small-c1-wh-rust.json`
- `review_queues-small-c128-rh-python.json`
- `review_queues-small-c128-rh-rust.json`
- `scorers-large-c128-rh-python.json`
- `scorers-large-c128-rh-rust.json`
- `scorers-large-c16-wh-python.json`
- `scorers-large-c16-wh-rust.json`
- `scorers-small-c1-wh-python.json`
- `scorers-small-c1-wh-rust.json`
- `scorers-small-c128-rh-python.json`
- `scorers-small-c128-rh-rust.json`
