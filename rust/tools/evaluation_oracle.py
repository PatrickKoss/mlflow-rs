"""Seeded Python/Rust differential for the Phase 19.2 evaluation core.

No provider is contacted. Python produces the rate-limit, scorer-result, and
aggregate-metric oracle; the Rust example consumes the identical seeded corpus.
"""

import json
import random
import subprocess
from pathlib import Path

from mlflow.entities import AssessmentSource, Feedback
from mlflow.genai.evaluation.entities import EvalItem, EvalResult
from mlflow.genai.evaluation.harness import _parse_rate_limit
from mlflow.genai.evaluation.utils import standardize_scorer_value
from mlflow.genai.scorers import ResponseLength
from mlflow.genai.scorers.aggregation import compute_aggregated_metrics

ROOT = Path(__file__).resolve().parents[2]


def normalized_feedback(feedback):
    return {
        "name": feedback.name,
        "value": feedback.value,
        "source_type": feedback.source.source_type,
        "source_id": feedback.source.source_id,
    }


def main():
    randomizer = random.Random(1902)
    rates = [None, "auto", "0", "2.5"]
    standard_values = [
        None,
        True,
        randomizer.randint(1, 100),
        "yes",
        [1, ["yes", "no"], False],
        {"dictionary-key": randomizer.random()},
    ]
    aggregate_values = [round(randomizer.random(), 8) for _ in range(7)] + ["yes", "no", True]
    corpus = {
        "rates": rates,
        "standard_values": standard_values,
        "aggregate_values": aggregate_values,
    }

    scorer = ResponseLength(
        name="quality",
        max_length=100,
        aggregations=["min", "max", "mean", "median", "variance", "p90"],
    )
    eval_item = EvalItem("seeded", {}, None, {})
    eval_results = [
        EvalResult(
            eval_item=eval_item,
            assessments=[
                Feedback(
                    name="quality",
                    value=value,
                    source=AssessmentSource("CODE", "quality"),
                )
            ],
        )
        for value in aggregate_values
    ]
    python = {
        "rates": [
            {"requests_per_second": parsed[0], "adaptive": parsed[1]}
            for parsed in map(_parse_rate_limit, rates)
        ],
        "standardized": [
            [
                normalized_feedback(feedback)
                for feedback in standardize_scorer_value("seeded", value)
            ]
            for value in standard_values
        ],
        "metrics": compute_aggregated_metrics(eval_results, [scorer]),
    }

    process = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "mlflow-genai", "--example", "evaluation_contract"],
        cwd=ROOT / "rust",
        input=json.dumps(corpus),
        text=True,
        capture_output=True,
        check=True,
    )
    rust = json.loads(process.stdout)
    assert rust["rates"] == python["rates"], (rust["rates"], python["rates"])
    assert rust["standardized"] == python["standardized"], (
        rust["standardized"],
        python["standardized"],
    )
    assert rust["metrics"].keys() == python["metrics"].keys()
    for name, expected in python["metrics"].items():
        assert abs(rust["metrics"][name] - expected) < 1e-12, (
            name,
            rust["metrics"][name],
            expected,
        )

    print(
        json.dumps(
            {
                "seed": 1902,
                "rate_cases": len(rates),
                "standardization_cases": len(standard_values),
                "aggregate_values": len(aggregate_values),
                "aggregate_metrics": len(python["metrics"]),
                "live_provider_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
