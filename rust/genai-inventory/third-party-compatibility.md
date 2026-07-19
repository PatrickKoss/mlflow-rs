# T19.3 third-party scorer compatibility matrix

This is the implementation audit for all 112 third-party manifest rows. The
Rust execution artifact installs no Python runtime or third-party Python
distribution. Pinned packages are used only to generate the checked-in corpus.

| Family | Reference pin | License | Rows | Exact workflow | Deterministic | Pinned error | D23 rejected |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepEval | `deepeval==4.0.7` | Apache-2.0 | 44 | 29 | 2 | 13 | 0 |
| Ragas | `ragas==0.4.3` | Apache-2.0 | 37 | 24 | 10 | 3 | 0 |
| TruLens | `trulens==2.8.1`, `trulens-providers-litellm==2.8.1` | MIT | 25 | 3 | 0 | 22 | 0 |
| Phoenix | `arize-phoenix-evals==2.13.0` | Elastic-2.0 | 6 | 0 | 0 | 0 | 6 |

The optional Ragas oracle tools are pinned to `rapidfuzz==3.14.3`,
`sacrebleu==2.6.0`, `rouge-score==0.1.2`, and `datacompy==0.19.0`.

## Per-metric status

`Exact workflow ✓` means the complete ordered pinned chat/embedding sequence,
structured response schemas, parsed feedback, and malformed-first-response
behavior are corpus-backed. `Pinned error` means the reference MLflow wrapper
fails before a usable provider workflow exists; Rust returns that error instead
of fabricating a prompt.

| Family | Metric | Status | Positive calls |
| --- | --- | --- | ---: |
| DeepEval | AnswerRelevancy | Exact workflow ✓ | 3 |
| DeepEval | ArgumentCorrectness | Pinned error | 0 |
| DeepEval | Bias | Exact workflow ✓ | 3 |
| DeepEval | ContextualPrecision | Exact workflow ✓ | 2 |
| DeepEval | ContextualRecall | Exact workflow ✓ | 2 |
| DeepEval | ContextualRelevancy | Exact workflow ✓ | 2 |
| DeepEval | ConversationCompleteness | Exact workflow ✓ | 3 |
| DeepEval | ConversationalDAG | Pinned error | 0 |
| DeepEval | DAG | Pinned error | 0 |
| DeepEval | ExactMatch | Deterministic ✓ | 0 |
| DeepEval | Faithfulness | Exact workflow ✓ | 4 |
| DeepEval | GoalAccuracy | Exact workflow ✓ | 3 |
| DeepEval | Hallucination | Exact workflow ✓ | 2 |
| DeepEval | ImageCoherence | Pinned error | 0 |
| DeepEval | ImageEditing | Pinned error | 0 |
| DeepEval | ImageHelpfulness | Pinned error | 0 |
| DeepEval | ImageReference | Pinned error | 0 |
| DeepEval | JsonCorrectness | Pinned error | 0 |
| DeepEval | KnowledgeRetention | Exact workflow ✓ | 2 |
| DeepEval | MCPTaskCompletion | Pinned error | 0 |
| DeepEval | MCPUse | Pinned error | 0 |
| DeepEval | Misuse | Exact workflow ✓ | 3 |
| DeepEval | MultiTurnMCPUse | Pinned error | 0 |
| DeepEval | NonAdvice | Exact workflow ✓ | 3 |
| DeepEval | PIILeakage | Exact workflow ✓ | 3 |
| DeepEval | PatternMatch | Deterministic ✓ | 0 |
| DeepEval | PlanAdherence | Exact workflow ✓ | 3 |
| DeepEval | PlanQuality | Exact workflow ✓ | 3 |
| DeepEval | PromptAlignment | Exact workflow ✓ | 2 |
| DeepEval | RoleAdherence | Exact workflow ✓ | 2 |
| DeepEval | RoleViolation | Exact workflow ✓ | 3 |
| DeepEval | StepEfficiency | Exact workflow ✓ | 2 |
| DeepEval | Summarization | Exact workflow ✓ | 7 |
| DeepEval | TaskCompletion | Pinned error | 0 |
| DeepEval | TextToImage | Pinned error | 0 |
| DeepEval | ToolCorrectness | Exact workflow ✓ | 0 |
| DeepEval | ToolUse | Exact workflow ✓ | 3 |
| DeepEval | TopicAdherence | Exact workflow ✓ | 3 |
| DeepEval | Toxicity | Exact workflow ✓ | 3 |
| DeepEval | TurnContextualPrecision | Exact workflow ✓ | 1 |
| DeepEval | TurnContextualRecall | Exact workflow ✓ | 1 |
| DeepEval | TurnContextualRelevancy | Exact workflow ✓ | 1 |
| DeepEval | TurnFaithfulness | Exact workflow ✓ | 5 |
| DeepEval | TurnRelevancy | Exact workflow ✓ | 2 |
| Ragas | AgentGoalAccuracy | Pinned error | 0 |
| Ragas | AgentGoalAccuracyWithReference | Exact workflow ✓ | 2 |
| Ragas | AgentGoalAccuracyWithoutReference | Exact workflow ✓ | 2 |
| Ragas | AnswerAccuracy | Exact workflow ✓ | 2 |
| Ragas | AnswerCorrectness | Exact workflow ✓ | 3 |
| Ragas | AnswerRelevancy | Exact workflow ✓ | 5 |
| Ragas | BleuScore | Deterministic ✓ | 0 |
| Ragas | CHRFScore | Deterministic ✓ | 0 |
| Ragas | ContextEntityRecall | Exact workflow ✓ | 2 |
| Ragas | ContextPrecision | Exact workflow ✓ | 1 |
| Ragas | ContextPrecisionWithReference | Exact workflow ✓ | 1 |
| Ragas | ContextPrecisionWithoutReference | Exact workflow ✓ | 1 |
| Ragas | ContextRecall | Exact workflow ✓ | 1 |
| Ragas | ContextRelevance | Exact workflow ✓ | 2 |
| Ragas | ContextUtilization | Exact workflow ✓ | 1 |
| Ragas | DataCompyScore | Deterministic ✓ | 0 |
| Ragas | DomainSpecificRubrics | Exact workflow ✓ | 1 |
| Ragas | ExactMatch | Deterministic ✓ | 0 |
| Ragas | FactualCorrectness | Exact workflow ✓ | 4 |
| Ragas | Faithfulness | Exact workflow ✓ | 2 |
| Ragas | InstanceSpecificRubrics | Exact workflow ✓ | 1 |
| Ragas | MultiModalFaithfulness | Pinned error | 0 |
| Ragas | MultiModalRelevance | Pinned error | 0 |
| Ragas | NoiseSensitivity | Exact workflow ✓ | 5 |
| Ragas | NonLLMStringSimilarity | Deterministic ✓ | 0 |
| Ragas | QuotedSpansAlignment | Deterministic ✓ | 0 |
| Ragas | ResponseGroundedness | Exact workflow ✓ | 2 |
| Ragas | RougeScore | Deterministic ✓ | 0 |
| Ragas | RubricsScoreWithReference | Exact workflow ✓ | 1 |
| Ragas | RubricsScoreWithoutReference | Exact workflow ✓ | 1 |
| Ragas | SQLSemanticEquivalence | Exact workflow ✓ | 1 |
| Ragas | SemanticSimilarity | Exact workflow ✓ | 2 |
| Ragas | StringPresence | Deterministic ✓ | 0 |
| Ragas | SummaryScore | Exact workflow ✓ | 3 |
| Ragas | ToolCallAccuracy | Deterministic ✓ | 0 |
| Ragas | ToolCallF1 | Deterministic ✓ | 0 |
| Ragas | TopicAdherence | Exact workflow ✓ | 3 |
| TruLens | Coherence | Exact workflow ✓ | 1 |
| TruLens | Comprehensiveness | Pinned error | 0 |
| TruLens | Conciseness | Pinned error | 0 |
| TruLens | ContextRelevance | Exact workflow ✓ | 1 |
| TruLens | Controversiality | Pinned error | 0 |
| TruLens | Correctness | Pinned error | 0 |
| TruLens | Criminality | Pinned error | 0 |
| TruLens | ExecutionEfficiency | Pinned error | 0 |
| TruLens | Groundedness | Exact workflow ✓ | 2 |
| TruLens | Harmfulness | Pinned error | 0 |
| TruLens | Helpfulness | Pinned error | 0 |
| TruLens | Insensitivity | Pinned error | 0 |
| TruLens | LogicalConsistency | Pinned error | 0 |
| TruLens | Maliciousness | Pinned error | 0 |
| TruLens | Misogyny | Pinned error | 0 |
| TruLens | PlanAdherence | Pinned error | 0 |
| TruLens | PlanQuality | Pinned error | 0 |
| TruLens | QsRelevance | Pinned error | 0 |
| TruLens | Relevance | Pinned error | 0 |
| TruLens | Sentiment | Pinned error | 0 |
| TruLens | Stereotypes | Pinned error | 0 |
| TruLens | Summarization | Pinned error | 0 |
| TruLens | ToolCalling | Pinned error | 0 |
| TruLens | ToolQuality | Pinned error | 0 |
| TruLens | ToolSelection | Pinned error | 0 |
| Phoenix | Hallucination | Rejected D23 | 0 |
| Phoenix | QA | Rejected D23 | 0 |
| Phoenix | Relevance | Rejected D23 | 0 |
| Phoenix | SQL | Rejected D23 | 0 |
| Phoenix | Summarization | Rejected D23 | 0 |
| Phoenix | Toxicity | Rejected D23 | 0 |

## Pinned-error boundary

- DeepEval `ArgumentCorrectness` and `TaskCompletion` fail in 4.0.7 while
  rendering their pinned templates because required Jinja variables are not
  supplied. DAG rows require non-serializable graph objects. Multimodal and MCP
  rows cannot receive their required multimodal/MCP test-case types through the
  pinned MLflow mapper. `JsonCorrectness` requires a Pydantic schema class, but
  serialized scorer data contains a JSON object.
- Ragas `AgentGoalAccuracy` is not classified as multi-turn by the pinned MLflow
  registry and receives an incompatible sample. Both multimodal rows call the
  MLflow adapter's absent `_map_provider_params` method.
- The pinned TruLens registry maps arguments only for `Coherence`,
  `ContextRelevance`, and `Groundedness`. The other 22 names reach provider
  methods with no arguments and fail before a provider call.

These are observable pinned-wrapper failures, not limitations requiring a live
model. Treating them as errors is the only exact behavior available without
changing the audited Python integration contract.

## Provenance, corpus, and replay

DeepEval prompt templates/schemas/parsers and score algorithms were inspected
from `deepeval/metrics/` in the 4.0.7 wheel and tag. Ragas prompt classes,
multi-step pipelines, result types, and embedding shapes were inspected from
`ragas/metrics/` in 0.4.3. TruLens templates, structured-output parsing,
normalization, and retry behavior were inspected from `trulens/feedback/` in
2.8.1. MLflow mapping behavior is pinned to reference SHA
`2a36d19898fe7cdd5f596d4992be7494159efd15`.

`third_party_golden.json` contains all 112 manifest rows, 15 deterministic
cases, 94 non-deterministic workflow dispositions, 129 ordered positive calls,
and 65 calls across 55 malformed-response transcripts. The companion runtime
`pinned_workflows.json` contains the same workflow records. The generator and
`tools/third_party_oracle.py` regenerate both under the exact pins and require
byte-identical output. Their provider and embedding boundaries are patched and
the only credential is the obvious fake `sk-fake-t19-3-not-a-secret` value.

Phoenix source/prompt assets were not copied. All six Elastic-2.0 rows retain
the user-ratified D23 rejection with permissive MLflow alternatives.
