"""Differential SerializedScorer corpus for the Phase 19.1 Rust executor.

Run with:
  uv run --with dspy==3.2.1 python rust/tools/scorer_oracle.py
"""

import json
import subprocess
from pathlib import Path
from typing import Literal

from mlflow.genai import scorers
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers.memalign.optimizer import MemoryAugmentedJudge
from mlflow.genai.scorers.base import Scorer


ROOT = Path(__file__).resolve().parents[2]


def accepted_payloads():
    required = {
        "ConversationalGuidelines": {"guidelines": ["Be concise"]},
        "Guidelines": {"guidelines": ["Be concise"]},
        "RegexMatch": {"pattern": "^answer:"},
        "ResponseLength": {"max_length": 100},
    }
    for name in sorted(scorers.__all__):
        if name not in {
            "Completeness",
            "ConversationCompleteness",
            "ConversationalGuidelines",
            "ConversationalRoleAdherence",
            "ConversationalSafety",
            "ConversationalToolCallEfficiency",
            "Correctness",
            "Equivalence",
            "ExpectationsGuidelines",
            "Fluency",
            "Guidelines",
            "KnowledgeRetention",
            "PIIDetection",
            "RegexMatch",
            "RelevanceToQuery",
            "ResponseLength",
            "RetrievalGroundedness",
            "RetrievalRelevance",
            "RetrievalSufficiency",
            "Safety",
            "Summarization",
            "ToolCallCorrectness",
            "ToolCallEfficiency",
            "UserFrustration",
        }:
            continue
        if name == "KnowledgeRetention":
            yield {
                "name": "knowledge_retention",
                "is_session_level_scorer": True,
                "builtin_scorer_class": name,
                "builtin_scorer_pydantic_data": {},
            }
        else:
            yield getattr(scorers, name)(**required.get(name, {})).model_dump()

    judge = make_judge(
        name="scripted_judge",
        instructions="Evaluate {{ outputs }}.",
        model="openai:/fake-chat",
        feedback_value_type=Literal["yes", "no"],
    )
    yield judge.model_dump()
    yield MemoryAugmentedJudge(
        judge,
        retrieval_k=2,
        embedding_model="openai:/fake-embedding",
        embedding_dim=3,
        _defer_init=True,
    ).model_dump()


def rejected_payloads():
    yield None
    yield []
    yield {}
    yield {"name": "missing-kind"}
    yield {
        "name": "ambiguous",
        "builtin_scorer_class": "ResponseLength",
        "instructions_judge_pydantic_data": {},
    }
    yield {"name": "unknown", "builtin_scorer_class": "Nope"}
    yield {"name": "missing", "instructions_judge_pydantic_data": {}}
    yield {
        "name": "decorator",
        "call_source": "return True",
        "call_signature": "(outputs)",
        "original_func_name": "decorator",
    }
    yield {"name": "incomplete", "call_source": "return True"}
    yield {
        "name": "length",
        "builtin_scorer_class": "ResponseLength",
        "builtin_scorer_pydantic_data": {},
    }
    yield {
        "name": "regex",
        "builtin_scorer_class": "RegexMatch",
        "builtin_scorer_pydantic_data": {},
    }


def python_result(payload):
    try:
        Scorer.model_validate_json(json.dumps(payload))
    except Exception as error:
        return {
            "ok": False,
            "error": str(error),
            "error_class": error.error_code,
            "status": 400,
        }
    return {"ok": True}


def main():
    accepted = list(accepted_payloads())
    rejected = list(rejected_payloads())
    corpus = accepted + rejected
    process = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "mlflow-genai", "--example", "scorer_contract"],
        cwd=ROOT / "rust",
        input=json.dumps(corpus),
        text=True,
        capture_output=True,
        check=True,
    )
    rust = json.loads(process.stdout)
    for index, (payload, result) in enumerate(zip(corpus, rust, strict=True)):
        python = python_result(payload)
        assert result["ok"] == python["ok"], (index, payload, python, result)
        if python["ok"]:
            assert result["roundtrip"] == payload, (index, payload, result)
        else:
            assert result["error"] == python["error"], (index, payload, python, result)
            assert result["error_class"] == python["error_class"], (
                index,
                payload,
                python,
                result,
            )
            assert result["status"] == python["status"], (index, payload, python, result)
    print(
        json.dumps(
            {
                "accepted_natively": len(accepted),
                "rejected_correctly": len(rejected),
                "roundtrips": len(accepted),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
