"""Replay the Python-generated Phase 19.4 discovery differential corpus."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "rust/crates/mlflow-genai/tests/fixtures/issue_discovery_golden.json"
GENERATOR = ROOT / "rust/crates/mlflow-genai/tests/fixtures/generate_issue_discovery_oracles.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        generated = Path(directory) / FIXTURE.name
        subprocess.run([sys.executable, str(GENERATOR), str(generated)], cwd=ROOT, check=True)
        if generated.read_bytes() != FIXTURE.read_bytes():
            raise AssertionError("issue-discovery Python golden drift")
    subprocess.run(
        ["cargo", "test", "-p", "mlflow-genai", "discovery", "--", "--nocapture"],
        cwd=ROOT / "rust",
        check=True,
    )
    corpus = json.loads(FIXTURE.read_text())
    print(
        json.dumps(
            {
                "sampling_cases": len(corpus["sampling"]),
                "latency_cases": 1,
                "cluster_cases": 1,
                "dedup_cases": 1,
                "e2e_cases": 1,
                "phase_diffs": 0,
                "persisted_artifact_diffs": 0,
                "live_provider_calls": corpus["live_provider_calls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
