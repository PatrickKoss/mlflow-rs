"""Generate the Phase 19.3 third-party compatibility corpus.

Run from the repository root with the exact reference environments:

    OPENAI_API_KEY=sk-fake-t19-3-not-a-secret uv run \
      --with 'deepeval==4.0.7' \
      --with 'ragas==0.4.3' \
      --with 'trulens==2.8.1' \
      --with 'trulens-providers-litellm==2.8.1' \
      --with 'rapidfuzz==3.14.3' \
      --with 'sacrebleu==2.6.0' \
      --with 'rouge-score==0.1.2' \
      --with 'datacompy==0.19.0' \
      python rust/crates/mlflow-genai/tests/fixtures/generate_third_party_oracles.py

No provider request leaves the process. Model adapter calls are patched at the
shared MLflow provider boundary and use the conspicuously fake credential above.
"""

import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

from mlflow.genai.scorers.deepeval import get_scorer as deepeval_scorer
from mlflow.genai.scorers.ragas import get_scorer as ragas_scorer
from mlflow.genai.scorers.trulens import get_scorer as trulens_scorer

ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "rust/genai-inventory/scorers.json"
OUTPUT = Path(
    os.environ.get(
        "MLFLOW_THIRD_PARTY_ORACLE_OUTPUT",
        Path(__file__).with_name("third_party_golden.json"),
    )
)
WORKFLOW_OUTPUT = Path(
    os.environ.get(
        "MLFLOW_THIRD_PARTY_WORKFLOW_OUTPUT",
        ROOT / "rust/crates/mlflow-genai/src/third_party/pinned_workflows.json",
    )
)
PINS = {
    "deepeval": ("4.0.7", "Apache-2.0"),
    "ragas": ("0.4.3", "Apache-2.0"),
    "trulens": ("2.8.1", "MIT"),
    "trulens-providers-litellm": ("2.8.1", "MIT"),
}
REFERENCE_TOOLS = {
    "datacompy": "0.19.0",
    "rapidfuzz": "3.14.3",
    "rouge-score": "0.1.2",
    "sacrebleu": "2.6.0",
}


def feedback(value):
    source_type = value.source.source_type
    return {
        "value": value.value,
        "rationale": value.rationale,
        "source_type": getattr(source_type, "value", source_type),
        "source_id": value.source.source_id,
        "metadata": value.metadata,
    }


def reference_trace(*, output, retrieval_context=None, tool_call=None):
    from mlflow.entities import Trace

    trace_id = "tr-00000000000000000000000000000000"

    def span(span_id, name, span_type, attributes, start):
        return {
            "trace_id": "AAAAAAAAAAAAAAAAAAAAAA==",
            "span_id": span_id,
            "parent_span_id": None,
            "name": name,
            "start_time_unix_nano": start,
            "end_time_unix_nano": start + 1,
            "events": [],
            "status": {"code": "STATUS_CODE_OK", "message": ""},
            "attributes": {
                "mlflow.traceRequestId": json.dumps(trace_id),
                "mlflow.spanType": json.dumps(span_type),
                **attributes,
            },
            "links": [],
        }

    spans = [
        span(
            "AAAAAAAAAAA=",
            "root",
            "CHAIN",
            {
                "mlflow.spanInputs": json.dumps({"question": "reference input"}),
                "mlflow.spanOutputs": json.dumps(output),
            },
            1,
        )
    ]
    if retrieval_context is not None:
        spans.append(
            span(
                "AQAAAAAAAAA=",
                "retrieve",
                "RETRIEVER",
                {"mlflow.spanOutputs": json.dumps([{"page_content": retrieval_context}])},
                2,
            )
        )
    if tool_call is not None:
        spans.append(
            span(
                "AgAAAAAAAAA=",
                tool_call["name"],
                "TOOL",
                {
                    "mlflow.spanInputs": json.dumps(tool_call["arguments"]),
                    "mlflow.spanOutputs": json.dumps({"temperature": 21}),
                },
                2,
            )
        )
    value = {
        "info": {
            "trace_id": trace_id,
            "trace_location": {
                "type": "MLFLOW_EXPERIMENT",
                "mlflow_experiment": {"experiment_id": "0"},
            },
            "request_time": "2026-01-01T00:00:00Z",
            "state": "OK",
            "trace_metadata": {
                "mlflow.traceInputs": json.dumps({"question": "reference input"}),
                "mlflow.traceOutputs": json.dumps(output),
            },
            "request_preview": "reference input",
            "response_preview": str(output),
            "execution_duration_ms": 1,
        },
        "data": {"spans": spans},
    }
    return Trace.from_json(json.dumps(value))


def deterministic_cases():
    cases = []
    for name, kwargs, outputs, expected in [
        ("ExactMatch", {}, " Paris ", "Paris"),
        ("ExactMatch", {"threshold": 0.25}, "Lyon", "Paris"),
        ("PatternMatch", {"pattern": r"answer: \d+"}, " answer: 42 ", None),
        ("PatternMatch", {"pattern": "YES", "ignore_case": True}, "yes", None),
    ]:
        scorer = deepeval_scorer(name, **kwargs)
        result = scorer(
            inputs="reference input",
            outputs=outputs,
            expectations={"expected_output": expected} if expected is not None else None,
        )
        cases.append({
            "family": "deepeval",
            "metric": name,
            "kwargs": kwargs,
            "model": scorer.model_dump()["third_party_scorer_data"]["model"],
            "inputs": "reference input",
            "outputs": outputs,
            "expectations": ({"expected_output": expected} if expected is not None else None),
            "trace": None,
            "feedback": feedback(result),
        })

    ragas_cases = [
        ("ExactMatch", {}, "Paris", "Paris"),
        ("ExactMatch", {}, " Paris ", "Paris"),
        ("StringPresence", {}, "The capital is Paris.", "Paris"),
        ("NonLLMStringSimilarity", {}, "kitten", "sitting"),
        ("BleuScore", {}, "The cat is on the mat.", "The cat is on the mat."),
        ("CHRFScore", {}, "Hello world!", "Hello world"),
        ("RougeScore", {"rouge_type": "rougeL"}, "a b d", "a b c"),
    ]
    for name, kwargs, outputs, expected in ragas_cases:
        scorer = ragas_scorer(name, **kwargs)
        result = scorer(
            inputs="reference input",
            outputs=outputs,
            expectations={"expected_output": expected},
        )
        cases.append({
            "family": "ragas",
            "metric": name,
            "kwargs": kwargs,
            "model": scorer.model_dump()["third_party_scorer_data"]["model"],
            "inputs": "reference input",
            "outputs": outputs,
            "expectations": {"expected_output": expected},
            "trace": None,
            "feedback": feedback(result),
        })

    csv_reference = "id,name\n1,Alice\n2,Bob"
    csv_response = "id,name\n1,Alice\n2,Bob\n3,Charlie"
    scorer = ragas_scorer("DataCompyScore")
    result = scorer(
        inputs="reference input",
        outputs=csv_response,
        expectations={"expected_output": csv_reference},
    )
    cases.append({
        "family": "ragas",
        "metric": "DataCompyScore",
        "kwargs": {},
        "model": scorer.model_dump()["third_party_scorer_data"]["model"],
        "inputs": "reference input",
        "outputs": csv_response,
        "expectations": {"expected_output": csv_reference},
        "trace": None,
        "feedback": feedback(result),
    })

    quoted_output = 'The source says "machine learning models improve accuracy".'
    trace = reference_trace(
        output=quoted_output,
        retrieval_context="Machine learning models improve accuracy by 15%.",
    )
    scorer = ragas_scorer("QuotedSpansAlignment")
    result = scorer(trace=trace)
    cases.append({
        "family": "ragas",
        "metric": "QuotedSpansAlignment",
        "kwargs": {},
        "model": scorer.model_dump()["third_party_scorer_data"]["model"],
        "inputs": None,
        "outputs": None,
        "expectations": None,
        "trace": json.loads(trace.to_json()),
        "feedback": feedback(result),
    })

    expected_tool_call = {"name": "get_weather", "arguments": {"location": "Paris"}}
    trace = reference_trace(output={"answer": "sunny"}, tool_call=expected_tool_call)
    for name in ["ToolCallAccuracy", "ToolCallF1"]:
        scorer = ragas_scorer(name)
        expectations = {"expected_tool_calls": [expected_tool_call]}
        result = scorer(trace=trace, expectations=expectations)
        cases.append({
            "family": "ragas",
            "metric": name,
            "kwargs": {},
            "model": scorer.model_dump()["third_party_scorer_data"]["model"],
            "inputs": None,
            "outputs": None,
            "expectations": expectations,
            "trace": json.loads(trace.to_json()),
            "feedback": feedback(result),
        })
    return cases


def adapter_transcripts():
    from pydantic import BaseModel

    from mlflow.genai.scorers.deepeval.models import MlflowDeepEvalLLM
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient
    from mlflow.genai.scorers.ragas.models import MlflowRagasLLM
    from mlflow.genai.scorers.trulens.models import _create_gateway_provider

    class Result(BaseModel):
        score: float
        reason: str

    transcripts = {}
    for family, adapter, call in [
        (
            "deepeval",
            MlflowDeepEvalLLM(ScorerLLMClient("openai:/fake-t19-3")),
            lambda adapter: adapter.generate("REFERENCE PROMPT", schema=Result),
        ),
        (
            "ragas",
            MlflowRagasLLM(ScorerLLMClient("openai:/fake-t19-3")),
            lambda adapter: adapter.generate("REFERENCE PROMPT", response_model=Result),
        ),
    ]:
        with patch(
            "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
            return_value='{"score": 0.75, "reason": "scripted"}',
        ) as provider:
            result = call(adapter)
        transcripts[family] = {
            "request": {
                "provider": provider.call_args.args[0],
                "model": provider.call_args.args[1],
                **provider.call_args.kwargs,
            },
            "parsed": result.model_dump(),
        }

    backend = ScorerLLMClient("openai:/fake-t19-3")
    provider = _create_gateway_provider(backend)
    messages = [
        {"role": "system", "content": "REFERENCE SYSTEM"},
        {"role": "user", "content": "REFERENCE USER"},
    ]
    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value='{"score": 2, "criteria": "scripted", "supporting_evidence": "fixture"}',
    ) as call:
        parsed = provider._create_chat_completion(messages=messages)
    transcripts["trulens"] = {
        "request": {
            "provider": call.call_args.args[0],
            "model": call.call_args.args[1],
            **call.call_args.kwargs,
        },
        "parsed": parsed,
    }
    return transcripts


def _schema_value(schema, root, field_name=""):
    """Build a deterministic valid value for a pinned structured-output schema."""
    if "$ref" in schema:
        resolved = root
        for part in schema["$ref"].split("/")[1:]:
            resolved = resolved[part]
        schema = resolved
    if schema.get("default") is not None:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if values := schema.get("enum"):
        return values[0]
    if variants := schema.get("anyOf"):
        variants = [variant for variant in variants if variant.get("type") != "null"]
        return _schema_value(variants[0], root, field_name) if variants else None
    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", properties)
        return {
            name: _schema_value(value, root, name)
            for name, value in properties.items()
            if name in required
        }
    if schema_type == "array":
        return [_schema_value(schema.get("items", {"type": "string"}), root, field_name)]
    if schema_type == "boolean":
        return True
    if schema_type == "integer":
        if field_name == "rating":
            return 2
        return 1
    if schema_type == "number":
        return 0.75
    if field_name in {"verdict", "classification"}:
        return "yes"
    if field_name in {"reason", "feedback", "explanation"}:
        return "scripted reason"
    return "scripted statement"


def _active_response_schema():
    for frame in inspect.stack():
        if frame.function in {"generate", "agenerate"}:
            if schema := (
                frame.frame.f_locals.get("schema") or frame.frame.f_locals.get("response_model")
            ):
                return schema
    return None


def _wire_request(args, call_kwargs):
    request = {
        "model": args[1],
        "messages": call_kwargs["messages"],
    }
    if parameters := call_kwargs.get("eval_parameters"):
        request.update(parameters)
    if response_format := call_kwargs.get("response_format"):
        request["response_format"] = response_format
    return request


def _assessment_error(value):
    error = getattr(value, "error", None)
    if error is None:
        return None
    return {
        "type": getattr(error, "error_code", type(error).__name__),
        "message": getattr(error, "error_message", str(error)),
    }


class _ScriptedEmbeddings:
    def __init__(self, calls):
        self.calls = calls

    async def aembed_text(self, text):
        self.calls.append({
            "kind": "embedding",
            "request": {"model": "text-embedding-3-small", "input": text},
            "response": {"embedding": [1.0, 0.0]},
        })
        return [1.0, 0.0]

    def embed_text(self, text):
        self.calls.append({
            "kind": "embedding",
            "request": {"model": "text-embedding-3-small", "input": text},
            "response": {"embedding": [1.0, 0.0]},
        })
        return [1.0, 0.0]

    async def aembed_texts(self, texts):
        self.calls.append({
            "kind": "embedding",
            "request": {"model": "text-embedding-3-small", "input": texts},
            "response": {"embeddings": [[1.0, 0.0] for _ in texts]},
        })
        return [[1.0, 0.0] for _ in texts]

    def embed_texts(self, texts):
        self.calls.append({
            "kind": "embedding",
            "request": {"model": "text-embedding-3-small", "input": texts},
            "response": {"embeddings": [[1.0, 0.0] for _ in texts]},
        })
        return [[1.0, 0.0] for _ in texts]


DEEPEVAL_KWARGS = {
    "JsonCorrectness": {"expected_schema": {"type": "object"}},
    "Misuse": {"domain": "general"},
    "NonAdvice": {"advice_types": ["medical"]},
    "PromptAlignment": {"prompt_instructions": ["be concise"]},
    "RoleViolation": {"role": "assistant"},
    "ToolUse": {"available_tools": ["lookup"]},
    "TopicAdherence": {"relevant_topics": ["science"]},
}

RAGAS_KWARGS = {
    "AnswerCorrectness": {"weights": [1.0, 0.0]},
    "DomainSpecificRubrics": {"rubrics": {"0": "bad", "1": "good"}},
    "RubricsScoreWithReference": {"rubrics": {"0": "bad", "1": "good"}},
    "RubricsScoreWithoutReference": {"rubrics": {"0": "bad", "1": "good"}},
}


def _reference_case():
    trace = reference_trace(
        output="reference output",
        retrieval_context="reference context",
        tool_call={"name": "lookup", "arguments": {"q": "x"}},
    )
    expectations = {
        "expected_output": "reference expected",
        "context": "reference context",
        "expected_tool_calls": [{"name": "lookup", "arguments": {"q": "x"}}],
        "reference_topics": ["science"],
        "rubrics": {"0": "bad", "1": "good"},
    }
    return trace, expectations


def _record_metric(family, metric, *, malformed=False):
    trace, expectations = _reference_case()
    calls = []

    def provider_call(*args, **call_kwargs):
        schema = _active_response_schema()
        response_schema = schema.model_json_schema() if schema is not None else None
        if malformed and not calls:
            response = "not-json"
        elif schema is not None:
            response = json.dumps(_schema_value(response_schema, response_schema))
        else:
            response = json.dumps({
                "score": 2,
                "criteria": "scripted",
                "supporting_evidence": "fixture",
            })
        calls.append({
            "kind": "chat",
            "request": _wire_request(args, call_kwargs),
            "response": response,
            "response_schema": response_schema,
        })
        return response

    kwargs = {}
    try:
        if family == "deepeval":
            kwargs = DEEPEVAL_KWARGS.get(metric, {})
            scorer = deepeval_scorer(metric, model="openai:/fake-t19-3", **kwargs)
        elif family == "ragas":
            kwargs = RAGAS_KWARGS.get(metric, {})
            if metric == "SemanticSimilarity":
                scorer = ragas_scorer(metric, **kwargs)
            else:
                scorer = ragas_scorer(metric, model="openai:/fake-t19-3", **kwargs)
            if getattr(scorer._metric, "embeddings", None):
                scorer._metric.embeddings = _ScriptedEmbeddings(calls)
        else:
            scorer = trulens_scorer(metric, model="openai:/fake-t19-3")
    except Exception as error:
        return {
            "family": family,
            "metric": metric,
            "status": "pinned-error",
            "kwargs": kwargs,
            "calls": calls,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        side_effect=provider_call,
    ):
        if family == "deepeval" and scorer.is_session_level_scorer:
            result = scorer(
                session=[trace],
                expectations={
                    "scenario": "test",
                    "chatbot_role": "assistant",
                    "expected_outcome": "help",
                    "context": ["reference context"],
                },
            )
        elif family == "ragas":
            result = scorer(
                inputs="reference input",
                outputs="reference output",
                expectations=expectations,
                trace=trace,
                session=[trace],
            )
        else:
            result = scorer(
                inputs="reference input",
                outputs="reference output",
                expectations=expectations,
                trace=trace,
            )

    record = {
        "family": family,
        "metric": metric,
        "status": (
            "exact-workflow" if calls or _assessment_error(result) is None else "pinned-error"
        ),
        "kwargs": kwargs,
        "calls": calls,
    }
    if error := _assessment_error(result):
        record["error"] = error
    else:
        record["feedback"] = feedback(result)
    return record


def workflow_transcripts(metrics):
    records = []
    for entry in metrics:
        family = entry["family"]
        if family == "phoenix" or entry["execution"] == "deterministic":
            continue
        record = _record_metric(family, entry["metric"])
        if record["status"] == "exact-workflow" and record["calls"]:
            record["malformed"] = _record_metric(family, entry["metric"], malformed=True)
        records.append(record)
    return records


def dynamic_errors():
    errors = {}
    for family, factory in [("deepeval", deepeval_scorer), ("ragas", ragas_scorer)]:
        try:
            factory("DefinitelyMissingMetric", model="openai:/fake-t19-3")
        except Exception as error:
            errors[family] = {"type": type(error).__name__, "message": str(error)}
    scorer = trulens_scorer(
        "DefinitelyMissingMetric",
        model="openai:/fake-t19-3",
    )
    result = scorer(inputs="reference input", outputs="reference output")
    errors["trulens"] = {
        "type": result.error.error_code,
        "message": result.error.error_message,
    }
    return errors


def main():
    assert os.environ.get("OPENAI_API_KEY", "").startswith("sk-fake-")
    versions = {}
    for package, (pin, license_name) in PINS.items():
        distribution = importlib.metadata.distribution(package)
        assert distribution.version == pin
        observed_license = distribution.metadata.get("License") or ""
        assert license_name.split("-")[0].lower() in observed_license.lower()
        versions[package] = {"version": pin, "license": license_name}
    reference_tools = {}
    for package, pin in REFERENCE_TOOLS.items():
        assert importlib.metadata.version(package) == pin
        reference_tools[package] = {
            "version": pin,
            "scope": "corpus-generation-only",
        }

    manifest = json.loads(MANIFEST.read_text())
    metrics = manifest["third_party_metrics"]
    assert len(metrics) == 112
    workflows = workflow_transcripts(metrics)
    workflow_trace, workflow_expectations = _reference_case()
    corpus = {
        "schema_version": 2,
        "generated_from": versions,
        "reference_tools": reference_tools,
        "fake_credential": "sk-fake-t19-3-not-a-secret",
        "live_provider_calls": 0,
        "manifest": metrics,
        "deterministic_cases": deterministic_cases(),
        "adapter_transcripts": adapter_transcripts(),
        "workflow_transcripts": workflows,
        "workflow_case": {
            "inputs": "reference input",
            "outputs": "reference output",
            "expectations": workflow_expectations,
            "trace": json.loads(workflow_trace.to_json()),
        },
        "dynamic_errors": dynamic_errors(),
        "phoenix": {
            "count": 6,
            "disposition": "rejected-elastic-2.0-d23",
            "metrics": [item["metric"] for item in metrics if item["family"] == "phoenix"],
        },
    }
    OUTPUT.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n")
    WORKFLOW_OUTPUT.write_text(json.dumps(workflows, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "workflow_output": str(WORKFLOW_OUTPUT),
                "manifest": len(metrics),
                "deterministic_cases": len(corpus["deterministic_cases"]),
                "adapter_transcripts": len(corpus["adapter_transcripts"]),
                "workflow_metrics": len(workflows),
                "workflow_calls": sum(len(record["calls"]) for record in workflows),
                "live_provider_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
