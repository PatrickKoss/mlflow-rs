"""Python/Rust scripted judge request oracle for Phase 19.1.

This suite performs no provider calls. It verifies the committed request transcript and
trace-tool schemas directly against Python, then runs the Rust scripted feedback/tool/
embedding suites that consume only loopback mock responses.
"""

import json
import subprocess
from pathlib import Path
from typing import Literal

from mlflow.genai.judges import make_judge
from mlflow.genai.judges.adapters.gateway_adapter import _build_request
from mlflow.genai.judges.tools import list_judge_tools
from mlflow.types.llm import ChatMessage

ROOT = Path(__file__).resolve().parents[2]


def instructions_request():
    judge = make_judge(
        name="concise",
        instructions="Return yes when {{ inputs }} is answered by {{ outputs }}",
        model="openai:/fake-chat",
        feedback_value_type=Literal["yes", "no"],
        inference_params={"temperature": 0.0, "max_tokens": 50},
    )
    request = _build_request(
        [
            ChatMessage(role="system", content=judge._build_system_message(False)),
            ChatMessage(
                role="user",
                content=judge._build_user_message({"question": "capital?"}, "Paris", None, None),
            ),
        ],
        None,
        judge._create_response_format_model(),
        True,
        judge._inference_params,
    )
    request["model"] = "fake-chat"
    return request


def main():
    fixture = json.loads(
        (ROOT / "rust/crates/mlflow-genai/tests/fixtures/instructions_request.json").read_text()
    )
    assert instructions_request() == fixture

    python_tools = [tool.get_definition().to_dict() for tool in list_judge_tools()]
    rust_tools = json.loads((ROOT / "rust/crates/mlflow-genai/src/judge_tools.json").read_text())
    assert python_tools == rust_tools

    subprocess.run(
        ["cargo", "test", "--quiet", "-p", "mlflow-genai", "--test", "execution"],
        cwd=ROOT / "rust",
        check=True,
    )
    print(
        json.dumps(
            {
                "instructions_transcripts": 1,
                "trace_tool_schemas": len(python_tools),
                "rust_scripted_suites": 7,
                "live_provider_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
