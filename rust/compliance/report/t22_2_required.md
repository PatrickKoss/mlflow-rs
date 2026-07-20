# T22.2 Python-suite and reachability conformance

Profile: `required`. Ledger reference: `c69051f534f4b0d171ed92d07c58a20f8c2a3461`.

Coverage status: `complete`. Required Rust repointable and client-only runs are present on both backends; repointable tests used a fresh server and database per test.

The required profile is the dependency-light HTTP/SDK core used on every Rust CI run. The full repointable ledger matrix runs nightly and on manual dispatch; tests classified `python_internal` remain inventory evidence but cannot cross an HTTP process boundary.

| Classification | Server | Backend | Isolation | Ledger tests | Passed | Failed | Errors | Skipped | Exit |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| server_reachable | rust | sqlite | fresh_server_and_database_per_test | 3 | 3 | 0 | 0 | 0 | 0 |
| client_only | rust | sqlite | shared_server | 4 | 11 | 0 | 0 | 0 | 0 |
| server_reachable | rust | postgres | fresh_server_and_database_per_test | 3 | 3 | 0 | 0 | 0 | 0 |
| client_only | rust | postgres | shared_server | 4 | 11 | 0 | 0 | 0 | 0 |

## Per-suite results

| Suite | Classification | Server | Backend | Ledger tests | Passed | Failed | Errors | Skipped |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | rust | sqlite | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_utils.py` | client_only | rust | sqlite | 2 | 4 | 0 | 0 | 0 |
| `tests/genai/simulators/test_utils.py` | client_only | rust | sqlite | 2 | 7 | 0 | 0 | 0 |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | rust | postgres | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_utils.py` | client_only | rust | postgres | 2 | 4 | 0 | 0 | 0 |
| `tests/genai/simulators/test_utils.py` | client_only | rust | postgres | 2 | 7 | 0 | 0 | 0 |

## Ledger invariants

- Server-reachable symbols/surfaces: 1546
- Client-only symbols: 375
- Dead symbols: 0
- Repointable server tests: 35
- Client-only SDK tests: 184
- Python-internal tests: 3433
- Unclassified paths: 0
- Server-reachable entries missing native owners: 0
