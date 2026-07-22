"""Import visible Codex conversation turns as MLflow traces.

The importer reads Codex JSONL session logs but deliberately excludes developer
instructions, hidden reasoning, tool calls, and tool outputs. Each user turn and
its visible assistant messages become one CHAT_MODEL trace in a shared session.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    prompt: str
    prompt_timestamp: str | None
    responses: list[str] = field(default_factory=list)


def _message_text(payload: dict[str, Any]) -> str:
    return "\n\n".join(
        part["text"]
        for part in payload.get("content", [])
        if part.get("type") in {"input_text", "output_text"} and part.get("text")
    )


def _is_injected_context(text: str) -> bool:
    return "# AGENTS.md instructions" in text or "<environment_context>" in text


def load_session(path: Path) -> tuple[dict[str, Any], list[Turn]]:
    metadata: dict[str, Any] = {}
    turns: list[Turn] = []
    current: Turn | None = None

    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("type") == "session_meta":
                metadata = event.get("payload", {})
                continue

            payload = event.get("payload", {})
            if event.get("type") != "response_item" or payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = _message_text(payload)
            if not text:
                continue

            if role == "user":
                if _is_injected_context(text):
                    continue
                if current is not None and current.responses:
                    turns.append(current)
                current = Turn(prompt=text, prompt_timestamp=event.get("timestamp"))
            elif current is not None:
                current.responses.append(text)

    if current is not None and current.responses:
        turns.append(current)
    return metadata, turns


def import_session(path: Path, tracking_uri: str, experiment_name: str) -> tuple[str, list[str]]:
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

    # MLflow reads tracing environment variables during import.
    import mlflow
    from mlflow.entities import SpanType

    metadata, turns = load_session(path)
    session_id = str(metadata.get("id") or metadata.get("session_id") or path.stem)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    trace_ids = []
    for index, turn in enumerate(turns, start=1):
        with mlflow.start_span(name=f"codex.turn.{index}", span_type=SpanType.CHAT_MODEL) as span:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                    "codex.source": "local-session-jsonl",
                }
            )
            span.set_inputs({"messages": [{"role": "user", "content": turn.prompt}]})
            span.set_outputs({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "\n\n".join(turn.responses),
                        }
                    }
                ]
            })
            span.set_attributes({
                "codex.session_id": session_id,
                "codex.turn_index": index,
                "codex.prompt_timestamp": turn.prompt_timestamp or "",
                "codex.cli_version": metadata.get("cli_version", ""),
                "codex.model_provider": metadata.get("model_provider", ""),
                "codex.cwd": metadata.get("cwd", ""),
                "codex.visible_response_count": len(turn.responses),
            })
            trace_ids.append(span.trace_id)

    return experiment.experiment_id, trace_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Codex rollout JSONL file")
    parser.add_argument("--tracking-uri", default="http://localhost")
    parser.add_argument("--experiment-name", default="Codex Live Sessions")
    args = parser.parse_args()

    experiment_id, trace_ids = import_session(
        args.session.resolve(), args.tracking_uri, args.experiment_name
    )
    print(json.dumps({"experiment_id": experiment_id, "trace_ids": trace_ids}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
