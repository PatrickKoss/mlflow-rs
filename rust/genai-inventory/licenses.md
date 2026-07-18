# Part II license and provenance audit

## BLOCKER

**Phoenix evaluator algorithms and prompt templates are licensed Elastic-2.0 and must not be
source-vendored into Apache-2.0 MLflow.** This blocks a direct T19.3 port of the six Phoenix
evaluators in `scorers.json`. Resolution requires an upstream relicense/permission grant or a
counsel-approved clean-room implementation derived from public behavior and recorded oracles, not
from Phoenix source. Until then, the Rust implementation must reject those serialized scorer
families explicitly rather than silently approximate them.

This is a source-vendoring compatibility inventory, not a legal opinion. Package versions are the
pinned reference set in the committed manifests. Wheel `METADATA` license fields were checked for
all third-party scorer/optimizer packages.

| Source to be ported | Pin | Material and provenance | License | Compatible |
| --- | --- | --- | --- | --- |
| MLflow built-in scorers and judges | reference SHA in `ledger.json` | `mlflow/genai/scorers/builtin_scorers.py`, `mlflow/genai/judges/`, repository `LICENSE.txt` | Apache-2.0 | yes |
| MLflow MetaPrompt optimizer | reference SHA in `ledger.json` | `mlflow/genai/optimize/optimizers/metaprompt_optimizer.py` and `mlflow/genai/optimize/optimizers/_metaprompt_utils.py` | Apache-2.0 | yes |
| LiteLLM provider transforms | `litellm==1.91.2` | [LiteLLM v1.91.2](https://github.com/BerriAI/litellm/tree/v1.91.2), wheel `litellm/llms/**/*.py` | MIT | yes |
| LiteLLM retry and error classification | `litellm==1.91.2` | wheel `litellm/router_utils/get_retry_from_policy.py` and `litellm/exceptions.py`; hashes in `providers.json` | MIT | yes |
| LiteLLM tokenizer mapping | `litellm==1.91.2` | wheel `litellm/utils.py` and `litellm/litellm_core_utils/get_llm_provider_logic.py`; hashes in `providers.json` | MIT | yes |
| LiteLLM model limits and prices | `litellm==1.91.2` | wheel `litellm/model_prices_and_context_window_backup.json`, SHA-256 pinned in `providers.json` | MIT | yes |
| GEPA algorithm and prompts | `gepa==0.0.27` | [GEPA v0.0.27](https://github.com/gepa-ai/gepa/tree/v0.0.27), wheel `gepa/` | MIT | yes |
| DSPy runtime used by `MemoryAugmentedJudge` | `dspy==3.2.1` | [DSPy 3.2.1](https://github.com/stanfordnlp/dspy/tree/3.2.1), wheel `dspy/` | MIT | yes |
| DeepEval metrics and prompts | `deepeval==4.0.7` | [DeepEval v4.0.7](https://github.com/confident-ai/deepeval/tree/v4.0.7), wheel `deepeval/metrics/` | Apache-2.0 | yes |
| Ragas metrics and prompts | `ragas==0.4.3` | [Ragas v0.4.3](https://github.com/explodinggradients/ragas/tree/v0.4.3), wheel `ragas/metrics/` | Apache-2.0 | yes |
| TruLens feedback algorithms and prompts | `trulens==2.8.1`, `trulens-providers-litellm==2.8.1` | [TruLens 2.8.1 source](https://github.com/truera/trulens/tree/trulens-2.8.1), wheel `trulens/feedback/` and LiteLLM provider package | MIT | yes |
| Phoenix evaluator algorithms and prompts | `arize-phoenix-evals==2.13.0` | [Phoenix evals package](https://pypi.org/project/arize-phoenix-evals/2.13.0/), wheel `phoenix/evals/` | Elastic-2.0 | **NO — BLOCKER** |

## Vendoring policy

- Vendor only the exact source assets required by the manifest. Preserve upstream copyright,
  license, and NOTICE material and record the pinned package/version plus asset hash beside the
  Rust port.
- The LiteLLM model table is the backup JSON shipped in the 1.91.2 wheel. Do not fetch the moving
  `main`-branch price map at build time, test time, or runtime. `providers.json` is the D16 snapshot
  and includes the source-asset hashes, all model rows, and presence flags.
- A compatible license does not make behavior self-defining. Phases T18–T19 must use the corpus
  oracles in `corpus-recorders.md` before translating transforms, prompts, and error behavior.
- The Phoenix block applies to source and prompt derivation. Black-box compatibility recordings may
  be used only under the project's approved clean-room process.
