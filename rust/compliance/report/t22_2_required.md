# T22.2 Python-suite and reachability conformance

Profile: `required`. Ledger reference: `c69051f534f4b0d171ed92d07c58a20f8c2a3461`.

The required profile is the dependency-light HTTP/SDK core used on every Rust CI run. The full repointable ledger matrix runs nightly and on manual dispatch; tests classified `python_internal` remain inventory evidence but cannot cross an HTTP process boundary.

| Classification | Backend | Ledger tests | Passed | Failed | Errors | Skipped | Exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| server_reachable | sqlite | 3 | 3 | 0 | 0 | 0 | 0 |
| client_only | sqlite | 4 | 11 | 0 | 0 | 0 | 0 |
| server_reachable | postgres | 3 | 3 | 0 | 0 | 0 | 0 |
| client_only | postgres | 4 | 11 | 0 | 0 | 0 | 0 |

## Per-suite results

| Suite | Classification | Backend | Ledger tests | Passed | Failed | Errors | Skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | sqlite | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_utils.py` | client_only | sqlite | 2 | 4 | 0 | 0 | 0 |
| `tests/genai/simulators/test_utils.py` | client_only | sqlite | 2 | 7 | 0 | 0 | 0 |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | postgres | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_utils.py` | client_only | postgres | 2 | 4 | 0 | 0 | 0 |
| `tests/genai/simulators/test_utils.py` | client_only | postgres | 2 | 7 | 0 | 0 | 0 |

## Ledger invariants

- Server-reachable symbols/surfaces: 1546
- Client-only symbols: 346
- Dead symbols: 0
- Repointable server tests: 3
- Client-only SDK tests: 183
- Python-internal tests: 3376
- Unclassified paths: 0
- Server-reachable entries missing native owners: 0
