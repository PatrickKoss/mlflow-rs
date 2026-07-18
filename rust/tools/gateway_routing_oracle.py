"""Hermetic Python oracle for the T18.5 fallback/traffic-routing contract."""

import asyncio
import json

import numpy as np
from fastapi import HTTPException

from mlflow.entities.gateway_endpoint import FallbackStrategy
from mlflow.gateway.providers.base import FallbackProvider


class ScriptedProvider:
    def __init__(self, name, attempts, *, result=None, error=None, stream=()):
        self.name = name
        self.attempts = attempts
        self.result = result
        self.error = error
        self.stream = stream

    async def chat(self, _payload):
        self.attempts.append(self.name)
        if self.error:
            raise self.error
        return self.result

    async def chat_stream(self, _payload):
        self.attempts.append(self.name)
        for item in self.stream:
            if isinstance(item, Exception):
                raise item
            yield item


async def run_non_stream(script, max_attempts=None):
    attempts = []
    providers = [ScriptedProvider(name, attempts, **action) for name, action in script]
    provider = FallbackProvider(
        providers,
        strategy=FallbackStrategy.SEQUENTIAL,
        max_attempts=max_attempts,
    )
    try:
        result = await provider.chat({})
        return {"attempts": attempts, "result": result}
    except Exception as error:
        return {
            "attempts": attempts,
            "status": getattr(error, "status_code", None),
            "detail": str(error),
            "type": type(error).__name__,
        }


async def run_stream():
    attempts = []
    providers = [
        ScriptedProvider(
            "partial-stream",
            attempts,
            stream=("partial", ValueError("scripted stream failure")),
        ),
        ScriptedProvider("fallback-success", attempts, stream=("recovered",)),
    ]
    provider = FallbackProvider(
        providers,
        strategy=FallbackStrategy.SEQUENTIAL,
        max_attempts=2,
    )
    chunks = [chunk async for chunk in provider.chat_stream({})]
    return {"attempts": attempts, "chunks": chunks}


async def main():
    weights = np.array([int(weight * 100) for weight in (0.009, 0.691, 0.3)], dtype=np.float32)
    normalized = (weights / np.sum(weights)).tolist()
    output = {
        "weights": {"integer": weights.tolist(), "normalized": normalized},
        "first_500_then_success": await run_non_stream([
            ("fail-500", {"error": HTTPException(500, "scripted primary failure")}),
            ("fallback-success", {"result": {"target": "fallback-success"}}),
        ]),
        "all_fail": await run_non_stream([
            ("fail-500", {"error": HTTPException(500, "scripted primary failure")}),
            ("fail-429", {"error": HTTPException(429, "scripted final limit")}),
        ]),
        "generic_error_then_success": await run_non_stream([
            ("generic-error", {"error": ValueError("not classified as retryable")}),
            ("fallback-success", {"result": {"target": "fallback-success"}}),
        ]),
        "max_attempts": await run_non_stream(
            [
                ("fail-500", {"error": HTTPException(500, "scripted primary failure")}),
                (
                    "fail-500-second",
                    {"error": HTTPException(500, "scripted primary failure")},
                ),
                ("excluded-success", {"result": {"target": "excluded-success"}}),
            ],
            max_attempts=2,
        ),
        "partial_stream": await run_stream(),
    }
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
