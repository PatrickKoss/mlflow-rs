# Part II corpus recorder definitions

These are recording contracts for T16–T20, not implementations. Recordings come from the pinned
Python reference in `ledger.json`; the Rust implementation is always a replay target and never a
golden-data producer.

## Shared harness contract

Follow the T12.4 harness structure in `rust/compliance/`:

- Reuse the recursive `${binding}` substitution, independent per-implementation binding tables,
  and JSON-path captures from `rust/compliance/engine.py::substitute` and `json_path_get`.
- Pass decoded payloads through `engine.normalize` with explicit `NormalizeOptions`. Reuse its
  `<TS>`, `<ID:n>`, `<TOKEN:present|absent>`, and `<PATH:*/basename>` conventions. Extend the field
  sets locally for GenAI fields; do not weaken the tracking corpus normalizer.
- Compare normalized structures with `engine.diff_normalized`. Preserve list order unless the
  Python API explicitly treats a field as a set. Never sort model messages, tool calls, assessment
  lists, SSE events, optimization candidates, or streamed deltas.
- As in `rust/compliance/replay.py`, seed a fresh SQLite store, use a temporary artifact root, bind
  IDs produced by earlier cases, and keep Python and Rust sessions isolated. Recorder output must
  contain no credentials, authorization headers, host-specific absolute paths, or user content not
  authored as a test fixture.
- Each fixture has `meta.json` with MLflow SHA/version, package pins, recorder schema version,
  invocation entry point, UTC recording time, seed, normalization options, and SHA-256 hashes of
  raw inputs. A fixture change is reviewed like an API behavior change.

Fixtures live below `rust/compliance/fixtures/genai/`. That directory is intentionally separate
from the existing request corpus: semantic calls are not all HTTP requests, while SSE needs a
framing-aware comparator. Recorder programs should be added under `rust/compliance/recorders/` by
the owning phase.

## Semantic execution recorder

### Entry points

Record all deserializable rows in `scorers.json` through the same Python entry point the job worker
uses, not by calling private metric helpers in isolation:

1. Deserialize the committed scorer payload with
   `mlflow.genai.scorers.base.Scorer.model_validate`, the same reconstruction path used for a
   registered scorer.
2. Invoke `mlflow.genai.scorers.base.Scorer.__call__` with the case's `inputs`, `outputs`,
   `expectations`, and optional `trace`.
3. For serialized judges, also invoke the reconstructed `Judge` directly through its inherited
   `Scorer.__call__` path so deserialization differences can be separated from judge semantics.
4. Record batch integration through `mlflow.genai.evaluation.harness.run` and the worker functions
   in `mlflow/genai/evaluation/job.py` and `mlflow/genai/scorers/job.py` for cancellation, partial
   failure, aggregation, and persistence behavior.
5. GEPA and MetaPrompt cases enter through
   `mlflow.genai.optimize.optimizers.GepaPromptOptimizer.optimize` and
   `MetaPromptOptimizer.optimize`, with the public job config from `mlflow/genai/optimize/job.py`.

Third-party scorers are recorded once per exact package pin. The Phoenix family remains subject to
the license block in `licenses.md`; its corpus may serve only an approved clean-room process.

### Capture envelope

Write one case per directory:

```text
rust/compliance/fixtures/genai/semantic/<family>/<name>/<case>/
  meta.json
  request.json
  model-transcript.json
  result.json
```

`request.json` contains the serialized scorer/judge/optimizer configuration and fully expanded
call arguments. `model-transcript.json` is an ordered list of model/embedding requests and pinned
responses. `result.json` contains exactly one of `return` or `error`; errors record Python exception
class, MLflow error code when present, normalized message, and the job terminal state.

No recorder case calls a live model. Install a transport at the lowest shared model boundary used
by the reference entry point (LiteLLM completion/embedding or the MLflow judge adapter), assert each
request against the next transcript entry, and return the stored response. A transcript mismatch is
a recorder failure. This pins prompt text, roles, tool schemas, provider transform shape, retry
classification, token accounting inputs, and multi-call order without storing secrets or spending
network resources.

### Semantic normalization

- Apply the T12.4 timestamp, ID, token, and path rules recursively to request, transcript, return,
  exception, trace, and persisted assessment objects. Add `span_id`, `session_id`, `job_id`,
  `scorer_id`, and `prompt_id` to the local ID set.
- Canonicalize mapping key order only at JSON serialization. Preserve strings byte-for-byte after
  replacing UUIDs, temporary directories, localhost ports, and timestamps with bindings/sentinels.
- Preserve numeric type and value. Encode non-JSON floats as the strings `<NaN>`, `<+Inf>`, and
  `<-Inf>`; do not round scores, costs, thresholds, or token counts.
- Fix `random`, NumPy, Faker, and algorithm seeds to zero when present. Freeze wall time. Record the
  seed and every model response in `meta.json`; a case that remains nondeterministic is invalid.
- Normalize exception paths and generated IDs, but retain exception class, MLflow error code,
  validation locations, retry count, and message punctuation.
- Include success, validation failure, provider failure, malformed serialized payload, empty input,
  trace/no-trace, tool-call, conversation, aggregation, cancellation, and budget-limit cases where
  the manifest family supports them.

## SSE stream recorder

### Entry points

Boot the Python application using the same process lifecycle as `rust/compliance/replay.py`. Replace
the provider at the registry/session boundary with an ordered async stub, then issue real HTTP
requests with streaming enabled:

- Gateway: every streaming branch in `mlflow.server.gateway_api`, including unified invocations,
  MLflow chat completions, OpenAI chat/responses, Anthropic messages, Gemini stream generation, and
  raw proxy streaming.
- Assistant: `mlflow.server.assistant.api.send_message`,
  `mlflow.server.assistant.api.stream_response`, permission resolution, cancellation, and provider
  failure. Stub the selected `AssistantProvider` and tool execution; do not spawn a real CLI or run
  shell tools.

The provider stub yields deliberately awkward fragments: multiple events in one yield, one event
split across yields, split UTF-8 code points, comments/heartbeats, empty data, multiline `data:`,
tool-call deltas, terminal usage, `[DONE]`, mid-stream exception, and client disconnect.

### Capture envelope and framing

```text
rust/compliance/fixtures/genai/sse/<gateway|assistant>/<route>/<case>/
  meta.json
  request.json
  upstream.json
  stream.json
```

`upstream.json` stores the ordered stub yields and exceptions. `stream.json` stores status, selected
response headers, SHA-256 of the raw response bytes, and parsed logical SSE events. Parse according
to the SSE line grammar: normalize CRLF/CR to LF, join repeated `data:` lines with `\n`, terminate an
event only on a blank line, preserve `event`, `id`, and `retry`, and preserve event order. Network/
ASGI chunk boundaries are diagnostic metadata, not comparison semantics.

For JSON-valued `data`, decode and run the T12.4 recursive normalizer, then serialize canonically.
For non-JSON data, preserve text exactly except approved binding substitution. Do not normalize
away `[DONE]`, event names, delta ordering, finish reasons, error types/statuses, tool-call indexes,
usage fields, or whether the final blank-line terminator is present.

### Replay assertions

The phase-owned replay test runs the same `request.json` and `upstream.json` independently against
Python and Rust. It asserts:

1. status and selected headers before consuming the body;
2. identical normalized logical event sequence;
3. identical termination class: clean EOF, explicit terminal event, provider error, cancellation,
   or client disconnect;
4. no extra event after terminal state and no lost buffered event at EOF; and
5. equivalent trace/usage side effects after the stream closes.

Raw-byte hashes are retained for diagnosing framing regressions but are not a parity gate because
server frameworks may choose different HTTP chunk boundaries. A framing difference that changes
the logical SSE event sequence is always a parity failure.
