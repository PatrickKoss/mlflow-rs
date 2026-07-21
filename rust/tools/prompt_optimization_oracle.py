"""Pinned Python oracle for the native prompt-optimization state machine.

Run with:
  uv run --with 'gepa==0.0.27' python rust/tools/prompt_optimization_oracle.py
"""

import hashlib
import json

import gepa

from mlflow.genai.optimize.optimizers import MetaPromptOptimizer
from mlflow.genai.optimize.types import EvaluationResultRecord

EXPECTED = {
    "metaprompt_zero_sha256": "594ffc4bb7031773e33809b0ad4cf37b38593015b68a1059f344bd28f7418a6e",
    "metaprompt_few_sha256": "bc39355534afef48883c5fefdb337f7ff20f3aa73a47e7bec9555b0fa665bb54",
    "candidates": [
        {"prompt": "seed"},
        {"prompt": "candidate-2"},
        {"prompt": "candidate-4"},
    ],
    "scores": [0.5, 0.5, 0.5],
    "selected": [0, 0, 0, 0],
    "batches": [[4, 0, 2], [3, 1, 1], [1, 4, 3], [2, 0, 0]],
    "metric_calls": 39,
}


class _SilentLogger:
    def log(self, _message):
        pass


class _Events:
    def __init__(self):
        self.selected = []
        self.batches = []

    def on_candidate_selected(self, event):
        self.selected.append(event["candidate_idx"])

    def on_minibatch_sampled(self, event):
        self.batches.append(event["minibatch_ids"])


class _Adapter(gepa.GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        text = candidate["prompt"]
        version = int(text.removeprefix("candidate-")) if text.startswith("candidate-") else 0
        scores = [((item + version) % 5) / 4 for item in batch]
        trajectories = [{"id": item} for item in batch] if capture_traces else None
        return gepa.EvaluationBatch(
            outputs=[f"{version}:{item}" for item in batch],
            scores=scores,
            trajectories=trajectories,
            objective_scores=[{"quality": score} for score in scores],
        )

    def make_reflective_dataset(self, candidate, eval_batch, components):
        return {
            component: [
                {"id": trajectory["id"], "score": score}
                for trajectory, score in zip(
                    eval_batch.trajectories, eval_batch.scores, strict=True
                )
            ]
            for component in components
        }


class _ReflectionModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, _prompt):
        self.calls += 1
        return f"```candidate-{self.calls}```"


def _sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _metaprompt_result():
    optimizer = MetaPromptOptimizer(
        reflection_model="openai:/fake-model",
        guidelines="Prefer concise answers.",
    )
    prompts = {"qa": "Answer {{question}}."}
    variables = optimizer._extract_template_variables(prompts)
    zero = optimizer._build_zero_shot_meta_prompt(prompts, variables)
    records = [
        EvaluationResultRecord(
            inputs={"question": "2+2?"},
            outputs="5",
            expectations={"expected_response": "4"},
            score=0.0,
            trace=None,
            rationales={"accuracy": "Incorrect"},
            individual_scores={"accuracy": 0.0},
        )
    ]
    few = optimizer._build_few_shot_meta_prompt(prompts, variables, records)
    return {
        "metaprompt_zero_sha256": _sha256(zero),
        "metaprompt_few_sha256": _sha256(few),
    }


def main():
    events = _Events()
    result = gepa.optimize(
        seed_candidate={"prompt": "seed"},
        trainset=list(range(5)),
        adapter=_Adapter(),
        reflection_lm=_ReflectionModel(),
        max_metric_calls=35,
        seed=7,
        callbacks=[events],
        logger=_SilentLogger(),
        display_progress_bar=False,
    )
    actual = _metaprompt_result() | {
        "candidates": result.candidates,
        "scores": result.val_aggregate_scores,
        "selected": events.selected,
        "batches": events.batches,
        "metric_calls": result.total_metric_calls,
    }
    if actual != EXPECTED:
        raise AssertionError(
            "prompt optimization oracle drift:\n"
            + json.dumps({"expected": EXPECTED, "actual": actual}, indent=2, sort_keys=True)
        )
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
