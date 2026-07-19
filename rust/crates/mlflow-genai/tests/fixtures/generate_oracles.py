"""Regenerate T15.4 scorer payload, result, and mock-gateway request fixtures."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.scorers import ResponseLength

FIXTURE_DIR = Path(__file__).parent
INSTRUCTIONS = "Return 'yes' when {{ outputs }} is concise, otherwise return 'no'."
CANNED_RESULT = {"result": "yes", "rationale": "The response is concise."}


def write_json(name: str, value: object) -> None:
    (FIXTURE_DIR / name).write_text(json.dumps(value, indent=2) + "\n")


def feedback_projection(feedback, *, include_judge_fields: bool = False) -> dict:
    projected = {
        "name": feedback.name,
        "value": str(feedback.value),
        "rationale": feedback.rationale,
        "source": feedback.source.to_dictionary(),
    }
    if include_judge_fields:
        projected["metadata"] = feedback.metadata
    return projected


class MockGatewayHandler(BaseHTTPRequestHandler):
    request_body: dict | None = None

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["content-length"]))
        type(self).request_body = json.loads(body)
        completion = {
            "id": "mock-completion-1",
            "object": "chat.completion",
            "created": 0,
            "model": "mock-judge",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(CANNED_RESULT),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        }
        encoded = json.dumps(completion).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *args) -> None:
        pass


def main() -> None:
    builtin = ResponseLength(min_length=2, max_length=4, unit="words")
    write_json("builtin_response_length_scorer.json", builtin.model_dump())
    write_json(
        "builtin_response_length_expected.json",
        feedback_projection(builtin(outputs="native Rust worker works")),
    )

    os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        judge = InstructionsJudge(
            name="concise_answer",
            instructions=INSTRUCTIONS,
            model="openai:/mock-judge",
            feedback_value_type=str,
            base_url=gateway_url,
        )
        feedback = judge(outputs="Brief answer.")
    finally:
        server.shutdown()
        thread.join()

    write_json("instructions_judge_scorer.json", judge.model_dump())
    write_json(
        "instructions_judge_expected.json",
        feedback_projection(feedback, include_judge_fields=True),
    )
    write_json("instructions_judge_request.json", MockGatewayHandler.request_body)


if __name__ == "__main__":
    main()
