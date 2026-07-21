from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from mlflow.entities.assessment import Feedback
from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
from mlflow.gateway.guardrail_utils import run_post_llm_guardrails, run_pre_llm_guardrails
from mlflow.gateway.guardrails import GuardrailViolation, JudgeGuardrail
from mlflow.gateway.schemas import chat
from mlflow.types.chat import ChatCompletionRequest


class MockJudgeScorer:
    def __init__(self, base_url: str, outcome: str) -> None:
        self.url = f"{base_url}/judge/{outcome}"

    def __call__(self, **kwargs):
        request = urllib.request.Request(
            self.url,
            data=json.dumps(kwargs).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        return Feedback(value=payload["result"], rationale=payload["rationale"])


def response_payload(content: str = "fixture answer") -> chat.ResponsePayload:
    return chat.ResponsePayload(
        id="openai-fixture-id",
        object="chat.completion",
        created=7,
        model="target-model",
        choices=[
            chat.Choice(
                index=0,
                message=chat.ResponseMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=chat.ChatUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        provider="openai",
    )


async def matrix_cell(base_url: str, stage: str, action: str, outcome: str):
    guardrail = JudgeGuardrail(
        scorer=MockJudgeScorer(base_url, outcome),
        stage=GuardrailStage(stage),
        action=GuardrailAction(action),
        name=f"matrix-{stage.lower()}-{action.lower()}-{outcome}",
        action_llm_url=base_url,
        action_endpoint_name="sanitizer",
    )
    request = chat.RequestPayload(
        messages=[{"role": "user", "content": "unsafe input"}], stream=False
    ).model_dump()
    try:
        request = await run_pre_llm_guardrails(
            [guardrail], request, payload_schema=ChatCompletionRequest.model_json_schema()
        )
        response = await run_post_llm_guardrails([guardrail], request, response_payload())
        body = json.dumps(response.model_dump(), separators=(",", ":"))
        return {"status": 200, "body": body}
    except GuardrailViolation as error:
        return {
            "status": 400,
            "body": json.dumps({"detail": str(error)}, separators=(",", ":")),
        }


async def main() -> None:
    base_url = os.environ["MLFLOW_GUARDRAIL_MOCK_URL"]
    result = {}
    for stage in ("BEFORE", "AFTER"):
        for action in ("VALIDATION", "SANITIZATION"):
            for outcome in ("pass", "violation"):
                key = f"{stage}-{action}-{outcome}"
                result[key] = await matrix_cell(base_url, stage, action, outcome)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    asyncio.run(main())
