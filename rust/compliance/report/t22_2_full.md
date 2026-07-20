# T22.2 Python-suite and reachability conformance

Profile: `full`. Ledger reference: `c69051f534f4b0d171ed92d07c58a20f8c2a3461`.

Coverage status: `complete`. Python-HTTP baseline, Rust repointable, and client-only runs are present on both backends; repointable tests used a fresh server and database per test.

The required profile is the dependency-light HTTP/SDK core used on every Rust CI run. The full repointable ledger matrix runs nightly and on manual dispatch; tests classified `python_internal` remain inventory evidence but cannot cross an HTTP process boundary.

| Classification | Server | Backend | Isolation | Ledger tests | Passed | Failed | Errors | Skipped | Exit |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| server_reachable | python_http | sqlite | fresh_server_and_database_per_test | 35 | 41 | 0 | 0 | 0 | 0 |
| server_reachable | python_http | postgres | fresh_server_and_database_per_test | 35 | 41 | 0 | 0 | 0 | 0 |
| server_reachable | rust | sqlite | fresh_server_and_database_per_test | 35 | 41 | 0 | 0 | 0 | 0 |
| server_reachable | rust | postgres | fresh_server_and_database_per_test | 35 | 41 | 0 | 0 | 0 | 0 |
| client_only | rust | sqlite | shared_server | 183 | 292 | 0 | 0 | 1 | 0 |
| client_only | rust | postgres | shared_server | 183 | 292 | 0 | 0 | 1 | 0 |

## Per-suite results

| Suite | Classification | Server | Backend | Ledger tests | Passed | Failed | Errors | Skipped |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tests/genai/datasets/test_fluent.py` | server_reachable | python_http | sqlite | 32 | 38 | 0 | 0 | 0 |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | python_http | sqlite | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/datasets/test_fluent.py` | server_reachable | python_http | postgres | 32 | 38 | 0 | 0 | 0 |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | python_http | postgres | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/datasets/test_fluent.py` | server_reachable | rust | sqlite | 32 | 38 | 0 | 0 | 0 |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | rust | sqlite | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/datasets/test_fluent.py` | server_reachable | rust | postgres | 32 | 38 | 0 | 0 | 0 |
| `tests/genai/test_rust_http_conformance.py` | server_reachable | rust | postgres | 3 | 3 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_dspy_base.py` | client_only | rust | sqlite | 23 | 23 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_dspy_utils.py` | client_only | rust | sqlite | 33 | 85 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_gepa.py` | client_only | rust | sqlite | 5 | 5 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_simba.py` | client_only | rust | sqlite | 4 | 4 | 0 | 0 | 0 |
| `tests/genai/labeling/test_labeling.py` | client_only | rust | sqlite | 1 | 0 | 0 | 0 | 1 |
| `tests/genai/scorers/google_adk/test_google_adk.py` | client_only | rust | sqlite | 37 | 48 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_guardrails.py` | client_only | rust | sqlite | 10 | 19 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_registry.py` | client_only | rust | sqlite | 3 | 8 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_utils.py` | client_only | rust | sqlite | 4 | 6 | 0 | 0 | 0 |
| `tests/genai/simulators/test_distillation.py` | client_only | rust | sqlite | 11 | 16 | 0 | 0 | 0 |
| `tests/genai/simulators/test_simulator.py` | client_only | rust | sqlite | 46 | 65 | 0 | 0 | 0 |
| `tests/genai/simulators/test_utils.py` | client_only | rust | sqlite | 6 | 13 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_dspy_base.py` | client_only | rust | postgres | 23 | 23 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_dspy_utils.py` | client_only | rust | postgres | 33 | 85 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_gepa.py` | client_only | rust | postgres | 5 | 5 | 0 | 0 | 0 |
| `tests/genai/judges/optimizers/test_simba.py` | client_only | rust | postgres | 4 | 4 | 0 | 0 | 0 |
| `tests/genai/labeling/test_labeling.py` | client_only | rust | postgres | 1 | 0 | 0 | 0 | 1 |
| `tests/genai/scorers/google_adk/test_google_adk.py` | client_only | rust | postgres | 37 | 48 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_guardrails.py` | client_only | rust | postgres | 10 | 19 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_registry.py` | client_only | rust | postgres | 3 | 8 | 0 | 0 | 0 |
| `tests/genai/scorers/guardrails/test_utils.py` | client_only | rust | postgres | 4 | 6 | 0 | 0 | 0 |
| `tests/genai/simulators/test_distillation.py` | client_only | rust | postgres | 11 | 16 | 0 | 0 | 0 |
| `tests/genai/simulators/test_simulator.py` | client_only | rust | postgres | 46 | 65 | 0 | 0 | 0 |
| `tests/genai/simulators/test_utils.py` | client_only | rust | postgres | 6 | 13 | 0 | 0 | 0 |

## Ledger invariants

- Server-reachable symbols/surfaces: 1546
- Client-only symbols: 346
- Dead symbols: 0
- Repointable server tests: 35
- Client-only SDK tests: 183
- Python-internal tests: 3344
- Unclassified paths: 0
- Server-reachable entries missing native owners: 0
