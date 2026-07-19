"""Replay the pinned Phase 19.3 third-party corpus without live model calls."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "rust/crates/mlflow-genai/tests/fixtures/third_party_golden.json"
GENERATOR = ROOT / "rust/crates/mlflow-genai/tests/fixtures/generate_third_party_oracles.py"


def main():
    with tempfile.TemporaryDirectory(prefix="mlflow-t19-3-") as directory:
        generated = Path(directory) / "third_party_golden.json"
        env = {
            **os.environ,
            "OPENAI_API_KEY": "sk-fake-t19-3-not-a-secret",
            "MLFLOW_THIRD_PARTY_ORACLE_OUTPUT": str(generated),
        }
        subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "deepeval==4.0.7",
                "--with",
                "ragas==0.4.3",
                "--with",
                "trulens==2.8.1",
                "--with",
                "trulens-providers-litellm==2.8.1",
                "--with",
                "rapidfuzz==3.14.3",
                "--with",
                "sacrebleu==2.6.0",
                "--with",
                "rouge-score==0.1.2",
                "--with",
                "datacompy==0.19.0",
                "python",
                str(GENERATOR),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        assert generated.read_bytes() == FIXTURE.read_bytes(), "golden corpus drift"

    subprocess.run(
        ["cargo", "test", "--quiet", "-p", "mlflow-genai", "--test", "third_party"],
        cwd=ROOT / "rust",
        check=True,
    )
    corpus = json.loads(FIXTURE.read_text())
    families = {}
    for metric in corpus["manifest"]:
        families[metric["family"]] = families.get(metric["family"], 0) + 1
    print(  # noqa: T201 - command-line oracle reports its result
        json.dumps(
            {
                "manifest_coverage": len(corpus["manifest"]),
                "families": families,
                "deterministic_cases": len(corpus["deterministic_cases"]),
                "adapter_transcripts": len(corpus["adapter_transcripts"]),
                "dynamic_error_families": len(corpus["dynamic_errors"]),
                "corpus_diff": 0,
                "live_provider_calls": corpus["live_provider_calls"],
                "rust_suites": 5,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
