# T22.1 Semantic Differential Corpus - Last Run

- Profile: **required**
- Cases run: **374**
- Non-allowlisted diffs: **0**
- Allowlisted diffs: **0**
- Live provider calls: **0**

## Per-section

| Section | Cases | Diffs |
|---|---:|---:|
| semantic_evaluation_harness | 26 | 0 |
| semantic_inline_judge_guardrails | 8 | 0 |
| semantic_issue_discovery | 7 | 0 |
| semantic_online_scoring | 1 | 0 |
| semantic_prompt_optimization | 2 | 0 |
| semantic_provider_manifest | 191 | 0 |
| semantic_scorer_execution | 139 | 0 |

## Commands

- **PASS** `scorer-python-rust-oracle`
- **PASS** `judge-python-rust-oracle`
- **PASS** `scorer-manifest-partition`
- **PASS** `third-party-python-golden-replay`
- **PASS** `provider-manifest-generation-check`
- **PASS** `provider-hermetic-matrix`
- **PASS** `evaluation-python-rust-oracle`
- **PASS** `issue-discovery-python-rust-oracle`
- **PASS** `online-scoring-shared-db-differential`
- **PASS** `optimizer-python-rust-oracle`
- **PASS** `guardrail-python-rust-matrix`

The required profile uses deterministic loopback fakes and committed Python-reference
goldens. The `oracle-refresh` profile regenerates the 112-entry third-party corpus
from the exact pinned packages; neither profile makes a live provider call.
