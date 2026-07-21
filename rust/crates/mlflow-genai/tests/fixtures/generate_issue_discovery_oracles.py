"""Generate Phase 19.4 issue-discovery differential cases without provider calls."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlflow.genai.discovery.clustering import cluster_by_llm
from mlflow.genai.discovery.entities import _IdentifiedIssue
from mlflow.genai.discovery.pipeline import _dedup_issues
from mlflow.genai.discovery.utils import _compute_percentiles


def response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")]
    )


def build() -> dict:
    sampling = []
    for n, k in [(10, 4), (120, 5), (5000, 100)]:
        sampling.append({
            "population": n,
            "sample_size": k,
            "selected": random.Random(42).sample(range(n), k),
        })

    values = [0.1, 0.2, 0.4, 1.0]
    computed = _compute_percentiles(values, [50, 75, 90, 95, 99])
    latency = {
        "seconds": values,
        "p50": round(computed[0], 2),
        "p75": round(computed[1], 2),
        "p90": round(computed[2], 2),
        "p95": round(computed[3], 2),
        "p99": round(computed[4], 2),
        "count": len(values),
    }

    raw_clusters = {"groups": [{"name": "Issue: Shared", "indices": [0, 2, 99]}]}
    with patch(
        "mlflow.genai.discovery.clustering._call_llm",
        return_value=response(json.dumps(raw_clusters)),
    ):
        clusters = cluster_by_llm(
            ["a", "b", "c", "d"], 3, "openai:/fake-chat", categories=["correctness"]
        )

    issues = [
        _IdentifiedIssue(
            name="Issue: Wrong city",
            description="Wrong city returned",
            root_cause="lookup tool",
            example_indices=[0],
            severity="low",
            categories=["correctness"],
        ),
        _IdentifiedIssue(
            name="Issue: Timeout",
            description="Tool timed out",
            root_cause="weather tool",
            example_indices=[1],
            severity="medium",
            categories=["latency"],
        ),
        _IdentifiedIssue(
            name="Issue: Incorrect city",
            description="Incorrect city returned",
            root_cause="lookup tool",
            example_indices=[2],
            severity="high",
            categories=["execution", "correctness"],
        ),
    ]
    dedup_inputs = [issue.model_dump(mode="json") for issue in issues]
    raw_dedup = {
        "groups": [
            {
                "indices": [0, 2],
                "name": "Issue: City lookup incorrect",
                "description": "The lookup returns the wrong city.",
                "root_cause": "lookup tool mapping",
            }
        ]
    }
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=response(json.dumps(raw_dedup)),
    ):
        deduped = _dedup_issues(issues, model="openai:/fake-chat")

    return {
        "sampling": sampling,
        "latency": latency,
        "clustering": {"raw": raw_clusters, "label_count": 4, "max_issues": 3, "groups": clusters},
        "dedup": {
            "raw": raw_dedup,
            "inputs": dedup_inputs,
            "issues": [issue.model_dump(mode="json") for issue in deduped],
        },
        "e2e": {
            "result": {
                "summary": (
                    "Analyzed **1** traces. Found **1** issues:\n\n"
                    "### 1. Incorrect capital answer (severity: high)\n\n"
                    "The response names the wrong capital.\n\n"
                    "**Root causes:** agent response generation\n\n"
                    "**Categories:** correctness\n"
                ),
                "issues": 1,
                "total_traces_analyzed": 1,
                "total_cost_usd": 0.4,
            },
            "status_details": {"stage": "Generating summary..."},
            "sampling_count": 1,
            "affected_trace_ids": ["tr-1"],
            "assessment_count": 2,
            "artifact_count": 3,
            "model_calls": 5,
        },
        "live_provider_calls": 0,
    }


def main() -> None:
    output = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("issue_discovery_golden.json")
    )
    output.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
