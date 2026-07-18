#!/usr/bin/env python3
"""Fixed-time Python budget oracle consumed by the Rust T18.6 tests."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.gateway.budget import check_budget_limit, fire_budget_exceeded_webhooks
from mlflow.gateway.budget_tracker.in_memory import InMemoryBudgetTracker
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig


NOW = datetime(2025, 6, 15, 10, 37, tzinfo=timezone.utc)


def policy(policy_id, amount, action):
    return GatewayBudgetPolicy(
        budget_policy_id=policy_id,
        budget_unit=BudgetUnit.USD,
        budget_amount=amount,
        duration=BudgetDuration(BudgetDurationUnit.DAYS, 1),
        target_scope=BudgetTargetScope.GLOBAL,
        budget_action=action,
        created_at=0,
        last_updated_at=0,
    )


def with_now(tracker, policies):
    with patch("mlflow.gateway.budget_tracker.in_memory.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        tracker.refresh_policies(policies)


def reject_case(spend):
    tracker = InMemoryBudgetTracker()
    reject = policy("bp-reject", 100.0, BudgetAction.REJECT)
    with_now(tracker, [reject])
    with patch("mlflow.gateway.budget_tracker.in_memory.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        tracker.record_cost(spend)
        exceeded, window = tracker.should_reject_request()
    if not exceeded:
        return {"reject": False, "detail": None}
    store = MagicMock()
    store.list_budget_policies.return_value = [reject]
    endpoint = GatewayEndpointConfig(
        endpoint_id="ep-test",
        endpoint_name="test-endpoint",
        experiment_id=None,
        usage_tracking=False,
        models=[],
    )
    with (
        patch("mlflow.gateway.budget.get_budget_tracker", return_value=tracker),
        patch("mlflow.gateway.budget_tracker.in_memory.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = NOW
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        try:
            check_budget_limit(store, endpoint)
        except Exception as exc:
            detail = exc.detail
    return {"reject": True, "detail": detail, "spend": window.cumulative_spend}


def alert_case():
    tracker = InMemoryBudgetTracker()
    alert = policy("bp-alert", 50.0, BudgetAction.ALERT)
    with_now(tracker, [alert])
    with patch("mlflow.gateway.budget_tracker.in_memory.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        tracker.record_cost(49.0)
        crossed = tracker.record_cost(1.0)
    with patch("mlflow.gateway.budget.deliver_webhook") as deliver:
        fire_budget_exceeded_webhooks(crossed, None, MagicMock())
        return deliver.call_args.kwargs["payload"]


def reset_case():
    tracker = InMemoryBudgetTracker()
    alert = policy("bp-reset", 100.0, BudgetAction.ALERT)
    with_now(tracker, [alert])
    with patch("mlflow.gateway.budget_tracker.in_memory.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        tracker.record_cost(150.0)
        mock_dt.now.return_value = NOW + timedelta(days=1)
        tracker.record_cost(10.0)
    window = tracker._get_window_info("bp-reset")
    return {"spend": window.cumulative_spend, "exceeded": window.exceeded}


print(
    json.dumps(
        {
            "under": reject_case(99.0),
            "boundary": reject_case(100.0),
            "over": reject_case(101.0),
            "alert": alert_case(),
            "reset": reset_case(),
        },
        sort_keys=True,
    )
)
