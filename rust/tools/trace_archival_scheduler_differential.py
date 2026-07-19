"""Emit Python trace-archival scheduler decisions for Rust differential tests.

This is intentionally not a standalone fixture generator. The Rust scheduler
test supplies a JSON scenario through the required ``--content`` argument and
compares Python's seeded shuffle, interval gate, and shared-budget decisions.
"""

from __future__ import annotations

import argparse
import json
import random


def _gate_trace(polls: list[dict]) -> list[bool]:
    last_run = 0.0
    decisions = []
    for poll in polls:
        if not poll["configured"] or not poll["enabled"]:
            decisions.append(False)
            continue
        now = poll["monotonic_seconds"]
        if now - last_run < poll["interval_seconds"]:
            decisions.append(False)
            continue
        last_run = now
        decisions.append(True)
    return decisions


def _pass_trace(case: dict) -> dict:
    workspaces = list(case["workspaces"])
    random.Random(case["seed"]).shuffle(workspaces)
    remaining = case["max_traces_per_pass"]
    calls = []
    archived = []
    for workspace in workspaces:
        if remaining is not None and remaining <= 0:
            break
        calls.append({"workspace": workspace, "remaining_budget": remaining})
        scope = case["scopes"][workspace]
        if scope.get("error", False):
            continue
        selected = scope["candidates"]
        if remaining is not None:
            selected = selected[:remaining]
        archived.extend(
            {
                "workspace": workspace,
                "experiment_id": candidate["experiment_id"],
                "trace_id": candidate["trace_id"],
            }
            for candidate in selected
        )
        if remaining is not None:
            remaining = max(remaining - len(selected), 0)
    return {
        "workspace_order": workspaces,
        "calls": calls,
        "archived": archived,
        "remaining_budget": remaining,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True)
    args = parser.parse_args()
    content = json.loads(args.content)
    print(  # noqa: T201
        json.dumps(
            {
                "gate_traces": [_gate_trace(polls) for polls in content["gate_cases"]],
                "pass_traces": [_pass_trace(case) for case in content["pass_cases"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
