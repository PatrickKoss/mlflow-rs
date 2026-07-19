# T15.4 Python scorer oracles

These files are generated from the checkout's real `mlflow.genai` code:

```bash
uv run python rust/crates/mlflow-genai/tests/fixtures/generate_oracles.py
```

`builtin_response_length_scorer.json` is the exact dictionary emitted by
`ResponseLength.model_dump()`, which scorer CRUD persists with `json.dumps`.
The instructions-judge oracle starts a local OpenAI-compatible HTTP server and
invokes the real Python `InstructionsJudge` against it. This mocks the same
outbound HTTP boundary as the Rust engine, captures the request body in
`instructions_judge_request.json`, and replays a canned completion without a
real LLM or credential.

## T19.3 pinned third-party corpus

`third_party_golden.json` and the runtime `pinned_workflows.json` are generated
by `generate_third_party_oracles.py` from DeepEval 4.0.7, Ragas 0.4.3, and
TruLens 2.8.1. They freeze the 112-row compatibility partition, all ordered
per-metric chat/embedding requests, parsed feedback or pinned pre-call errors,
malformed-first-response behavior, deterministic cases, dynamic-name failures,
and the six D23 Phoenix rejections. Every model and embedding boundary is
patched and the generator requires the literal fake-key prefix `sk-fake-`; it
performs no live provider call.
