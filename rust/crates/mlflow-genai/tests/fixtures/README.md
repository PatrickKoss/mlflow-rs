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
