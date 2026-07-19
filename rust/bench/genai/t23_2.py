"""T23.2 Tier-A CRUD and read-path benchmark matrix."""

# Bulk seed construction is clearer as nested loops because several families
# append parent and child rows together.
# ruff: noqa: PERF401

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from mlflow.entities._job_status import JobStatus
from mlflow.store.tracking.dbmodels.models import (
    SqlEntityAssociation,
    SqlEvaluationDataset,
    SqlEvaluationDatasetRecord,
    SqlGatewayBudgetPolicy,
    SqlIssue,
    SqlJob,
    SqlLabelSchema,
    SqlReviewQueue,
    SqlScorer,
    SqlScorerVersion,
)
from rust.bench.genai.equivalence import compare_runs
from rust.bench.genai.metrics import (
    AsyncBenchClient,
    MetricsCollector,
    ResourceMonitor,
    process_tree,
)
from rust.bench.genai.mock_provider import provider_server
from rust.bench.genai.runner import (
    DB_URI,
    FAKE_API_KEY,
    HERE,
    RUST_ROOT,
    compose_args,
    install_claude_stub,
    launch_server,
    postgres_sample,
    recreate_database,
    run_command,
    stop_server,
    sync_request,
    validate_raw_metrics,
    write_raw_metrics,
)

CANONICAL_SMALL_REQUESTS = 10_000
CANONICAL_LARGE_REQUESTS = 1_000
CORPUS_ROWS = 10_000
WARMUP_REQUESTS = 20
CELL_TIMEOUT_SECONDS = 300
DB_POOL_CONFIG = {"max_overflow": 8, "pool_size": 32, "postgres_max_connections": 400}
SCHEMA_VERSION = "1.1.0"
FAMILIES = (
    "datasets",
    "scorers",
    "issues",
    "label_schemas",
    "review_queues",
    "prompt_optimization",
    "gateway_admin",
)
LARGE_DEFINITIONS = {
    "datasets": "8-record upsert with 64 KiB outputs per record (about 512 KiB JSON)",
    "scorers": "64 KiB serialized scorer JSON description",
    "issues": "64 KiB issue description",
    "label_schemas": (
        "maximum valid schema: 250-char name, 1000-char instruction, and ten 64-char options"
    ),
    "review_queues": ("ten 250-char users plus 100 schema/item references (about 6-8 KiB JSON)"),
    "prompt_optimization": "5 KiB optimizer_config_json (bounded by the 6,000-char run-param cap)",
    "gateway_admin": "64 KiB obvious-fake secret_value through AES-GCM envelope encryption",
}


@dataclass(frozen=True)
class Cell:
    index: int
    payload_size: str
    concurrency: int
    mix: str
    requests: int
    canonical_requests: int

    @property
    def slug(self) -> str:
        mix = "wh" if self.mix == "write-heavy" else "rh"
        return f"{self.payload_size}-c{self.concurrency}-{mix}"


@dataclass(frozen=True)
class RequestSpec:
    sequence: int
    endpoint: str
    method: str
    path: str
    json_body: Any | None
    scheduled_seconds: float


def cell_matrix(small_requests: int, large_requests: int) -> list[Cell]:
    # Fractional-factorial design: all three concurrency points occur in every
    # family while the two high-cost large cells keep the matrix bounded.
    return [
        Cell(0, "small", 1, "write-heavy", small_requests, CANONICAL_SMALL_REQUESTS),
        Cell(1, "small", 128, "read-heavy", small_requests, CANONICAL_SMALL_REQUESTS),
        Cell(2, "large", 16, "write-heavy", large_requests, CANONICAL_LARGE_REQUESTS),
        Cell(3, "large", 128, "read-heavy", large_requests, CANONICAL_LARGE_REQUESTS),
    ]


def fixed_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return (prefix + digest)[:36]


def pad(char: str, size: int) -> str:
    return (char * size)[:size]


def query(path: str, **params: object) -> str:
    values = [(key, value) for key, value in params.items() if value is not None]
    return f"{path}?{urlencode(values, doseq=True)}" if values else path


def setup_matrix_target(base_url: str, provider_url: str, seed: int) -> dict[str, Any]:
    setup: dict[str, Any] = {"experiments": {}}
    with requests.Session() as session:
        for family in FAMILIES:
            setup["experiments"][family] = {}
            for role in ("read", "write"):
                response = sync_request(
                    session,
                    base_url,
                    "POST",
                    "/api/2.0/mlflow/experiments/create",
                    json={"name": f"t23-2-{seed}-{family}-{role}"},
                )
                setup["experiments"][family][role] = response["experiment_id"]

        trace_ids = []
        spans = []
        now_ns = 1_750_000_000_000_000_000
        for index in range(100):
            trace_hex = hashlib.sha256(f"t23-2-review:{seed}:{index}".encode()).hexdigest()[:32]
            trace_ids.append(f"tr-{trace_hex}")
            spans.append({
                "attributes": [],
                "endTimeUnixNano": str(now_ns + index * 2_000_000 + 1_000_000),
                "name": f"review-trace-{index}",
                "spanId": trace_hex[:16],
                "startTimeUnixNano": str(now_ns + index * 2_000_000),
                "status": {"code": 1},
                "traceId": trace_hex,
            })
        otlp = {
            "resourceSpans": [
                {
                    "resource": {"attributes": []},
                    "scopeSpans": [{"scope": {"name": "t23-2-review"}, "spans": spans}],
                }
            ]
        }
        response = session.post(
            base_url + "/v1/traces",
            headers={
                "content-type": "application/json",
                "x-mlflow-experiment-id": setup["experiments"]["review_queues"]["write"],
            },
            json=otlp,
            timeout=30,
        )
        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                f"review trace setup failed: HTTP {response.status_code}: {response.text[:500]}"
            )
        setup["review_trace_ids"] = trace_ids

        dataset_ids = {}
        for role in ("read_records", "write"):
            experiment_role = "read" if role == "read_records" else "write"
            response = sync_request(
                session,
                base_url,
                "POST",
                "/api/3.0/mlflow/datasets/create",
                json={
                    "created_by": "t23-bench",
                    "experiment_ids": [setup["experiments"]["datasets"][experiment_role]],
                    "name": f"t23-2-{seed}-dataset-{role}",
                    "source_type": "HUMAN",
                    "tags": json.dumps({"role": role}),
                },
            )
            dataset_ids[role] = response["dataset"]["dataset_id"]
        setup["datasets"] = dataset_ids

        gateway = {}
        for role in ("read", "write"):
            secret = sync_request(
                session,
                base_url,
                "POST",
                "/api/3.0/mlflow/gateway/secrets/create",
                json={
                    "auth_config": {"api_base": f"{provider_url}/v1"},
                    "created_by": "t23-bench",
                    "provider": "openai",
                    "secret_name": f"t23-2-{seed}-gateway-{role}",
                    "secret_value": {"api_key": FAKE_API_KEY},
                },
            )["secret"]
            model = sync_request(
                session,
                base_url,
                "POST",
                "/api/3.0/mlflow/gateway/model-definitions/create",
                json={
                    "created_by": "t23-bench",
                    "model_name": "genai-bench-model",
                    "name": f"t23-2-{seed}-model-{role}",
                    "provider": "openai",
                    "secret_id": secret["secret_id"],
                },
            )["model_definition"]
            endpoint = sync_request(
                session,
                base_url,
                "POST",
                "/api/3.0/mlflow/gateway/endpoints/create",
                json={
                    "created_by": "t23-bench",
                    "model_configs": [
                        {
                            "linkage_type": "PRIMARY",
                            "model_definition_id": model["model_definition_id"],
                            "weight": 1.0,
                        }
                    ],
                    "name": f"t23-2-{seed}-endpoint-{role}",
                    "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                    "usage_tracking": False,
                },
            )["endpoint"]
            gateway[role] = {
                "endpoint_id": endpoint["endpoint_id"],
                "model_definition_id": model["model_definition_id"],
                "secret_id": secret["secret_id"],
            }
        scorer = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/scorers/register",
            json={
                "experiment_id": setup["experiments"]["gateway_admin"]["write"],
                "name": "t23-2-gateway-guardrail-scorer",
                "serialized_scorer": json.dumps({"name": "t23-2-gateway-guardrail-scorer"}),
            },
        )
        gateway["scorer_id"] = scorer["scorer_id"]
        gateway["scorer_version"] = scorer["version"]
        guardrail = sync_request(
            session,
            base_url,
            "POST",
            "/api/3.0/mlflow/gateway/guardrails/create",
            json={
                "action": "VALIDATION",
                "name": "t23-2-read-guardrail",
                "scorer_id": scorer["scorer_id"],
                "scorer_version": scorer["version"],
                "stage": "BEFORE",
            },
        )["guardrail"]
        gateway["read_guardrail_id"] = guardrail["guardrail_id"]
        setup["gateway"] = gateway
    return setup


def _bulk(session: Session, model: type[Any], rows: list[dict[str, Any]]) -> None:
    for offset in range(0, len(rows), 5_000):
        session.bulk_insert_mappings(model, rows[offset : offset + 5_000])


def seed_matrix_corpora(setup: dict[str, Any], seed: int) -> None:
    """Bulk-seed deterministic rows; setup is excluded from measurements."""
    now = 1_750_000_000_000
    future = 9_000_000_000_000
    experiments = setup["experiments"]
    engine = create_engine(DB_URI)
    with Session(engine) as session:
        dataset_rows = []
        associations = []
        read_dataset_exp = experiments["datasets"]["read"]
        for index in range(CORPUS_ROWS):
            dataset_id = fixed_id("d-", seed, "dataset-corpus", index)
            dataset_rows.append({
                "dataset_id": dataset_id,
                "workspace": "default",
                "name": f"corpus-dataset-{index:05d}",
                "schema": "{}",
                "profile": "{}",
                "digest": f"{index:064x}",
                "created_time": now + index,
                "last_update_time": now + index,
                "created_by": "t23-bench",
            })
            associations.append({
                "association_id": fixed_id("a-", seed, "dataset-association", index),
                "source_type": "evaluation_dataset",
                "source_id": dataset_id,
                "destination_type": "experiment",
                "destination_id": str(read_dataset_exp),
                "created_time": now,
            })
        for index in range(12_000):
            dataset_rows.append({
                "dataset_id": fixed_id("d-", seed, "dataset-delete", index),
                "workspace": "default",
                "name": f"delete-dataset-{index:05d}",
                "schema": None,
                "profile": None,
                "digest": None,
                "created_time": now - index - 1,
                "last_update_time": now - index - 1,
                "created_by": "t23-bench",
            })
        _bulk(session, SqlEvaluationDataset, dataset_rows)
        _bulk(session, SqlEntityAssociation, associations)

        record_rows = []
        for role, count in (("read", CORPUS_ROWS), ("write", 8_192)):
            dataset_id = setup["datasets"]["read_records" if role == "read" else "write"]
            for index in range(count):
                inputs = {"slot": index, "role": role}
                canonical = json.dumps(inputs, sort_keys=True)
                record_rows.append({
                    "dataset_record_id": fixed_id("dr-", seed, role, index),
                    "dataset_id": dataset_id,
                    "inputs": inputs,
                    "outputs": {"answer": f"seeded-{index}"},
                    "expectations": None,
                    "tags": {"corpus": role},
                    "source": None,
                    "source_id": None,
                    "source_type": None,
                    "created_time": now + index,
                    "last_update_time": now + index,
                    "created_by": "t23-bench",
                    "last_updated_by": "t23-bench",
                    "input_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                })
        _bulk(session, SqlEvaluationDatasetRecord, record_rows)

        scorer_rows = []
        version_rows = []
        scorer_exp = int(experiments["scorers"]["read"])
        for name_index in range(100):
            scorer_id = fixed_id("sc-", seed, "scorer-corpus", name_index)
            scorer_rows.append({
                "experiment_id": scorer_exp,
                "scorer_name": f"corpus-scorer-{name_index:03d}",
                "scorer_id": scorer_id,
            })
            for version in range(1, 101):
                version_rows.append({
                    "scorer_id": scorer_id,
                    "scorer_version": version,
                    "serialized_scorer": json.dumps({"name": f"corpus-{name_index}"}),
                    "creation_time": now + name_index * 100 + version,
                })
        delete_exp = int(experiments["scorers"]["write"])
        for index in range(12_000):
            scorer_id = fixed_id("sc-", seed, "scorer-delete", index)
            scorer_rows.append({
                "experiment_id": delete_exp,
                "scorer_name": f"delete-scorer-{index:05d}",
                "scorer_id": scorer_id,
            })
            version_rows.append({
                "scorer_id": scorer_id,
                "scorer_version": 1,
                "serialized_scorer": "{}",
                "creation_time": now,
            })
        _bulk(session, SqlScorer, scorer_rows)
        _bulk(session, SqlScorerVersion, version_rows)

        issue_rows = []
        for index in range(CORPUS_ROWS):
            issue_rows.append({
                "issue_id": fixed_id("iss-", seed, "issue-corpus", index),
                "experiment_id": int(experiments["issues"]["read"]),
                "name": f"corpus-issue-{index:05d}",
                "description": f"seeded issue {index}",
                "status": "pending",
                "severity": "medium",
                "root_causes": '["seeded"]',
                "source_run_id": None,
                "categories": '["quality"]',
                "created_timestamp": now + index,
                "last_updated_timestamp": now + index,
                "created_by": "t23-bench",
            })
        for index in range(1_024):
            issue_rows.append({
                "issue_id": fixed_id("iss-", seed, "issue-update", index),
                "experiment_id": int(experiments["issues"]["write"]),
                "name": f"update-issue-{index:04d}",
                "description": "update fixture",
                "status": "pending",
                "severity": "low",
                "root_causes": None,
                "source_run_id": None,
                "categories": None,
                "created_timestamp": now,
                "last_updated_timestamp": now,
                "created_by": "t23-bench",
            })
        _bulk(session, SqlIssue, issue_rows)

        label_rows = []
        for role, count in (("read", CORPUS_ROWS), ("update", 1_024), ("delete", 12_000)):
            experiment_id = int(experiments["label_schemas"]["read" if role == "read" else "write"])
            for index in range(count):
                label_rows.append({
                    "schema_id": fixed_id("ls-", seed, f"label-{role}", index),
                    "experiment_id": experiment_id,
                    "name": f"{role}-label-{index:05d}",
                    "type": "feedback",
                    "instruction": "seeded",
                    "enable_comment": False,
                    "input_type": "categorical",
                    "input_config": json.dumps({"options": ["yes", "no"], "multi_select": False}),
                    "created_by": "t23-bench",
                    "created_time": now + index,
                    "last_update_time": now + index,
                    "is_default": False,
                })
        review_schema_ids = []
        review_exp = int(experiments["review_queues"]["write"])
        for index in range(100):
            schema_id = fixed_id("ls-", seed, "review-schema", index)
            review_schema_ids.append(schema_id)
            label_rows.append({
                "schema_id": schema_id,
                "experiment_id": review_exp,
                "name": f"review-schema-{index:03d}",
                "type": "feedback",
                "instruction": "review fixture",
                "enable_comment": False,
                "input_type": "text",
                "input_config": "{}",
                "created_by": "t23-bench",
                "created_time": now,
                "last_update_time": now,
                "is_default": False,
            })
        _bulk(session, SqlLabelSchema, label_rows)
        setup["review_schema_ids"] = review_schema_ids

        queue_rows = []
        for role, count in (("read", CORPUS_ROWS), ("update", 1_024), ("delete", 12_000)):
            experiment_id = int(experiments["review_queues"]["read" if role == "read" else "write"])
            for index in range(count):
                name = f"{role}-queue-{index:05d}"
                queue_rows.append({
                    "queue_id": fixed_id("rq-", seed, f"queue-{role}", index),
                    "experiment_id": experiment_id,
                    "name": name,
                    "name_key": name.lower(),
                    "queue_type": "custom",
                    "created_by": "t23-bench",
                    "creation_time_ms": now + index,
                    "last_update_time_ms": now + index,
                })
        _bulk(session, SqlReviewQueue, queue_rows)

        jobs = []
        read_prompt_exp = experiments["prompt_optimization"]["read"]
        write_prompt_exp = experiments["prompt_optimization"]["write"]
        for role, count, status in (
            ("corpus", CORPUS_ROWS, JobStatus.CANCELED.to_int()),
            ("delete", 40_000, JobStatus.CANCELED.to_int()),
            ("cancel", 4_000, JobStatus.PENDING.to_int()),
        ):
            for index in range(count):
                # SearchPromptOptimizationJobs has no response pagination. Keep
                # ten rows in the requested experiment and the rest in the
                # same 10k-row table corpus so response materialization does
                # not monopolize Python's database pool.
                experiment_id = (
                    read_prompt_exp if role == "corpus" and index < 10 else write_prompt_exp
                )
                job_id = fixed_id("", seed, f"prompt-{role}", index)
                params = {
                    "run_id": fixed_id("", seed, f"prompt-run-{role}", index),
                    "experiment_id": str(experiment_id),
                    "prompt_uri": f"prompts:/t23-fake/{index + 1}",
                    "dataset_id": "",
                    "optimizer_type": "gepa",
                    "optimizer_config": {"max_metric_calls": 1},
                    "scorer_names": ["Correctness"],
                }
                jobs.append({
                    "id": job_id,
                    "creation_time": now + index,
                    "job_name": (
                        "optimize_prompts" if role == "corpus" else "t23_prompt_crud_fixture"
                    ),
                    "params": json.dumps(params),
                    "workspace": "default",
                    "timeout": None,
                    "status": status,
                    "result": None,
                    "retry_count": 0,
                    "last_update_time": now + index,
                    "status_details": None,
                })
        _bulk(session, SqlJob, jobs)

        budgets = []
        for index in range(CORPUS_ROWS):
            budgets.append({
                "budget_policy_id": fixed_id("bp-", seed, "budget-corpus", index),
                "budget_unit": "USD",
                "budget_amount": 100.0 + index,
                "duration_unit": "DAYS",
                "duration_value": 1,
                "target_scope": "WORKSPACE",
                "budget_action": "ALERT",
                "created_by": "t23-bench",
                "created_at": future + index,
                "last_updated_by": None,
                "last_updated_at": future + index,
                "workspace": "default",
            })
        for index in range(12_000):
            budgets.append({
                "budget_policy_id": fixed_id("bp-", seed, "budget-delete", index),
                "budget_unit": "USD",
                "budget_amount": 1.0,
                "duration_unit": "DAYS",
                "duration_value": 1,
                "target_scope": "WORKSPACE",
                "budget_action": "ALERT",
                "created_by": "t23-bench",
                "created_at": now - index,
                "last_updated_by": None,
                "last_updated_at": now - index,
                "workspace": "default",
            })
        _bulk(session, SqlGatewayBudgetPolicy, budgets)
        session.commit()
    engine.dispose()


def _body_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _dataset_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    base = "/api/3.0/mlflow/datasets"
    if index % 10 == 8:
        body = {
            "created_by": "t23-bench",
            "experiment_ids": [setup["experiments"]["datasets"]["write"]],
            "name": f"matrix-{cell.slug}-{sequence}",
            "source_type": "HUMAN",
            "tags": json.dumps({
                "payload": pad("s", 800 if cell.payload_size == "small" else 4_000)
            }),
        }
        return "datasets_create", "POST", f"{base}/create", body
    if index % 10 == 9:
        dataset_id = fixed_id(
            "d-", setup["seed"], "dataset-delete", cell.index * 3_000 + index // 10
        )
        return "datasets_delete", "DELETE", f"{base}/{dataset_id}", None
    batch = 1 if cell.payload_size == "small" else 8
    output_size = 700 if cell.payload_size == "small" else 65_536
    records = []
    for offset in range(batch):
        slot = (sequence * batch + offset) % 8_192
        records.append({
            "inputs": {"role": "write", "slot": slot},
            "outputs": {"answer": pad(chr(97 + offset % 26), output_size)},
            "tags": {"mlflow.user": "t23-bench"},
        })
    body = {"records": json.dumps(records, sort_keys=True), "updated_by": "t23-bench"}
    return "dataset_records_upsert", "POST", f"{base}/{setup['datasets']['write']}/records", body


def _dataset_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    if index % 2:
        return (
            "dataset_records_list",
            "GET",
            query(
                f"/api/3.0/mlflow/datasets/{setup['datasets']['read_records']}/records",
                max_results=10,
            ),
            None,
        )
    return (
        "datasets_search",
        "POST",
        "/api/3.0/mlflow/datasets/search",
        {"experiment_ids": [setup["experiments"]["datasets"]["read"]], "max_results": 10},
    )


def _scorer_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    if index % 10 == 9:
        slot = cell.index * 3_000 + index // 10
        return (
            "scorers_delete",
            "DELETE",
            "/api/3.0/mlflow/scorers/delete",
            {
                "experiment_id": setup["experiments"]["scorers"]["write"],
                "name": f"delete-scorer-{slot:05d}",
            },
        )
    description = pad("d", 700 if cell.payload_size == "small" else 65_536)
    serialized = json.dumps({"description": description, "name": f"matrix-{sequence}"})
    return (
        "scorers_register",
        "POST",
        "/api/3.0/mlflow/scorers/register",
        {
            "experiment_id": setup["experiments"]["scorers"]["write"],
            "name": f"matrix-{cell.slug}-{sequence}",
            "serialized_scorer": serialized,
        },
    )


def _scorer_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    experiment_id = setup["experiments"]["scorers"]["read"]
    if index % 2:
        return (
            "scorer_versions_list",
            "GET",
            query(
                "/api/3.0/mlflow/scorers/versions",
                experiment_id=experiment_id,
                name=f"corpus-scorer-{index % 100:03d}",
            ),
            None,
        )
    return (
        "scorers_list",
        "GET",
        query("/api/3.0/mlflow/scorers/list", experiment_id=experiment_id),
        None,
    )


def _issue_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    description = pad("i", 700 if cell.payload_size == "small" else 65_536)
    if index % 2:
        issue_id = fixed_id("iss-", setup["seed"], "issue-update", index % 1_024)
        return (
            "issues_update",
            "PATCH",
            f"/api/3.0/mlflow/issues/{issue_id}",
            {
                "description": description,
                "issue_id": issue_id,
                "name": f"updated-{sequence}",
                "severity": "high",
            },
        )
    return (
        "issues_create",
        "POST",
        "/api/3.0/mlflow/issues",
        {
            "categories": ["quality", "t23"],
            "created_by": "t23-bench",
            "description": description,
            "experiment_id": setup["experiments"]["issues"]["write"],
            "name": f"matrix-issue-{cell.slug}-{sequence}",
            "root_causes": ["seeded"],
            "severity": "medium",
        },
    )


def _issue_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    return (
        "issues_search",
        "POST",
        "/api/3.0/mlflow/issues/search",
        {
            "experiment_id": setup["experiments"]["issues"]["read"],
            "include_trace_count": index % 10 == 0,
            "max_results": 10,
        },
    )


def label_payload(cell: Cell, name: str) -> dict[str, Any]:
    if cell.payload_size == "large":
        options = [f"{index}-{pad(chr(97 + index), 60)}" for index in range(10)]
        return {
            "enable_comment": True,
            "input": {"categorical": {"multi_select": False, "options": options}},
            "instruction": pad("l", 1_000),
            "name": (name + "-" + pad("n", 250))[:250],
            "type": "FEEDBACK",
        }
    return {
        "enable_comment": True,
        "input": {"categorical": {"multi_select": False, "options": ["yes", "no"]}},
        "instruction": pad("l", 650),
        "name": name,
        "type": "FEEDBACK",
    }


def _label_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    operation = index % 3
    if operation == 0:
        body = label_payload(cell, f"matrix-label-{cell.slug}-{sequence}")
        body["experiment_id"] = setup["experiments"]["label_schemas"]["write"]
        return "label_schemas_create", "POST", "/api/3.0/mlflow/label-schemas/create", body
    if operation == 1:
        schema_id = fixed_id("ls-", setup["seed"], "label-update", index % 1_024)
        body = label_payload(cell, f"updated-label-{cell.slug}-{sequence}")
        body.pop("type")
        body["schema_id"] = schema_id
        return "label_schemas_update", "PATCH", "/api/3.0/mlflow/label-schemas/update", body
    slot = cell.index * 3_000 + index // 3
    return (
        "label_schemas_delete",
        "DELETE",
        "/api/3.0/mlflow/label-schemas/delete",
        {"schema_id": fixed_id("ls-", setup["seed"], "label-delete", slot)},
    )


def _label_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    return (
        "label_schemas_list",
        "GET",
        query(
            "/api/3.0/mlflow/label-schemas/list",
            experiment_id=setup["experiments"]["label_schemas"]["read"],
            max_results=10,
        ),
        None,
    )


def review_payload(setup: dict[str, Any], cell: Cell, name: str) -> dict[str, Any]:
    if cell.payload_size == "large":
        users = [f"user-{index}-{pad(chr(97 + index), 240)}"[:250] for index in range(10)]
        schema_ids = setup["review_schema_ids"]
    else:
        users = ["reviewer@example.com"]
        schema_ids = setup["review_schema_ids"][:1]
    return {"name": name[:250], "queue_type": "CUSTOM", "schema_ids": schema_ids, "users": users}


def _review_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    operation = index % 5
    base = "/api/3.0/mlflow/review-queues"
    if operation == 0:
        body = review_payload(setup, cell, f"matrix-queue-{cell.slug}-{sequence}")
        body["experiment_id"] = setup["experiments"]["review_queues"]["write"]
        return "review_queues_create", "POST", f"{base}/create", body
    queue_id = fixed_id("rq-", setup["seed"], "queue-update", index % 1_024)
    if operation == 1:
        payload = review_payload(setup, cell, f"updated-queue-{cell.slug}-{sequence}")
        return (
            "review_queues_update",
            "POST",
            f"{base}/update",
            {
                "name": payload["name"],
                "queue_id": queue_id,
                "update_users": True,
                "users": payload["users"],
            },
        )
    if operation in (2, 3):
        count = 1 if cell.payload_size == "small" else 100
        item_ids = setup["review_trace_ids"][:count]
        action = "add" if operation == 2 else "remove"
        return (
            f"review_queue_items_{action}",
            "POST",
            f"{base}/items/{action}",
            {"item_ids": item_ids, "item_type": "TRACE", "queue_id": queue_id},
        )
    slot = cell.index * 3_000 + index // 5
    return (
        "review_queues_delete",
        "POST",
        f"{base}/delete",
        {"queue_id": fixed_id("rq-", setup["seed"], "queue-delete", slot)},
    )


def _review_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    return (
        "review_queues_list",
        "GET",
        query(
            "/api/3.0/mlflow/review-queues/list",
            experiment_id=setup["experiments"]["review_queues"]["read"],
            max_results=10,
        ),
        None,
    )


def _prompt_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    base = "/api/3.0/mlflow/prompt-optimization/jobs"
    # Keep real job execution enabled but bound its influence in this CRUD task:
    # 1% of writes exercise create/enqueue, 9% cancel, and 90% finalized delete.
    if index % 100 == 0:
        config_size = 650 if cell.payload_size == "small" else 5_000
        config = json.dumps({"guidelines": pad("p", config_size), "max_metric_calls": 1})
        return (
            "prompt_optimization_create",
            "POST",
            base,
            {
                "config": {
                    "optimizer_config_json": config,
                    "optimizer_type": "OPTIMIZER_TYPE_GEPA",
                    "scorers": ["Correctness"],
                },
                "experiment_id": setup["experiments"]["prompt_optimization"]["write"],
                "source_prompt_uri": f"prompts:/obvious-fake-{cell.slug}/{sequence + 1}",
                "tags": [{"key": "phase", "value": "23"}],
            },
        )
    if index % 10 == 1:
        slot = cell.index * 1_000 + index // 10
        job_id = fixed_id("", setup["seed"], "prompt-cancel", slot)
        return "prompt_optimization_cancel", "POST", f"{base}/{job_id}/cancel", {}
    slot = cell.index * 10_000 + index
    job_id = fixed_id("", setup["seed"], "prompt-delete", slot)
    return "prompt_optimization_delete", "DELETE", f"{base}/{job_id}", None


def _prompt_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    return (
        "prompt_optimization_search",
        "POST",
        "/api/3.0/mlflow/prompt-optimization/jobs/search",
        {"experiment_id": setup["experiments"]["prompt_optimization"]["read"]},
    )


def _gateway_write(
    setup: dict[str, Any], cell: Cell, sequence: int, index: int
) -> tuple[str, str, str, Any]:
    gateway = setup["gateway"]
    if cell.payload_size == "large":
        return (
            "gateway_secrets_update_crypto",
            "POST",
            "/api/3.0/mlflow/gateway/secrets/update",
            {
                "secret_id": gateway["write"]["secret_id"],
                "secret_value": {"api_key": f"obvious-fake-{sequence}-" + pad("k", 65_536)},
                "updated_by": "t23-bench",
            },
        )
    operation = index % 8
    if operation == 0:
        return (
            "gateway_secrets_update_crypto",
            "POST",
            "/api/3.0/mlflow/gateway/secrets/update",
            {
                "secret_id": gateway["write"]["secret_id"],
                "secret_value": {"api_key": f"obvious-fake-{sequence}-" + pad("k", 750)},
                "updated_by": "t23-bench",
            },
        )
    if operation == 1:
        return (
            "gateway_endpoints_update",
            "POST",
            "/api/3.0/mlflow/gateway/endpoints/update",
            {
                "endpoint_id": gateway["write"]["endpoint_id"],
                "updated_by": "t23-bench",
                "usage_tracking": False,
            },
        )
    if operation == 2:
        return (
            "gateway_models_update",
            "POST",
            "/api/3.0/mlflow/gateway/model-definitions/update",
            {
                "model_definition_id": gateway["write"]["model_definition_id"],
                "model_name": f"obvious-fake-model-{sequence}",
                "updated_by": "t23-bench",
            },
        )
    if operation == 3:
        return (
            "gateway_secrets_create_crypto",
            "POST",
            "/api/3.0/mlflow/gateway/secrets/create",
            {
                "created_by": "t23-bench",
                "provider": "anthropic",
                "secret_name": f"matrix-secret-{cell.slug}-{sequence}",
                "secret_value": {"api_key": "obvious-fake-" + pad("s", 750)},
            },
        )
    if operation == 4:
        return (
            "gateway_models_create",
            "POST",
            "/api/3.0/mlflow/gateway/model-definitions/create",
            {
                "created_by": "t23-bench",
                "model_name": "obvious-fake-model",
                "name": f"matrix-model-{cell.slug}-{sequence}",
                "provider": "openai",
                "secret_id": gateway["write"]["secret_id"],
            },
        )
    if operation == 5:
        return (
            "gateway_endpoints_create",
            "POST",
            "/api/3.0/mlflow/gateway/endpoints/create",
            {
                "created_by": "t23-bench",
                "model_configs": [
                    {
                        "linkage_type": "PRIMARY",
                        "model_definition_id": gateway["write"]["model_definition_id"],
                        "weight": 1.0,
                    }
                ],
                "name": f"matrix-endpoint-{cell.slug}-{sequence}",
                "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
                "usage_tracking": False,
            },
        )
    if operation == 6:
        return (
            "gateway_guardrails_create",
            "POST",
            "/api/3.0/mlflow/gateway/guardrails/create",
            {
                "action": "VALIDATION",
                "name": f"matrix-guardrail-{cell.slug}-{sequence}",
                "scorer_id": gateway["scorer_id"],
                "scorer_version": gateway["scorer_version"],
                "stage": "BEFORE",
            },
        )
    slot = cell.index * 3_000 + index // 8
    return (
        "gateway_budgets_delete",
        "DELETE",
        "/api/3.0/mlflow/gateway/budgets/delete",
        {"budget_policy_id": fixed_id("bp-", setup["seed"], "budget-delete", slot)},
    )


def _gateway_read(setup: dict[str, Any], index: int) -> tuple[str, str, str, Any]:
    gateway = setup["gateway"]["read"]
    operation = index % 5
    if operation == 0:
        return (
            "gateway_budgets_list",
            "GET",
            query("/api/3.0/mlflow/gateway/budgets/list", max_results=10),
            None,
        )
    if operation == 1:
        return (
            "gateway_secrets_get",
            "GET",
            query("/api/3.0/mlflow/gateway/secrets/get", secret_id=gateway["secret_id"]),
            None,
        )
    if operation == 2:
        return (
            "gateway_endpoints_get",
            "GET",
            query("/api/3.0/mlflow/gateway/endpoints/get", endpoint_id=gateway["endpoint_id"]),
            None,
        )
    if operation == 3:
        return (
            "gateway_models_get",
            "GET",
            query(
                "/api/3.0/mlflow/gateway/model-definitions/get",
                model_definition_id=gateway["model_definition_id"],
            ),
            None,
        )
    return (
        "gateway_guardrails_get",
        "GET",
        query(
            "/api/3.0/mlflow/gateway/guardrails/get",
            guardrail_id=setup["gateway"]["read_guardrail_id"],
        ),
        None,
    )


WRITERS = {
    "datasets": _dataset_write,
    "scorers": _scorer_write,
    "issues": _issue_write,
    "label_schemas": _label_write,
    "review_queues": _review_write,
    "prompt_optimization": _prompt_write,
    "gateway_admin": _gateway_write,
}
READERS = {
    "datasets": _dataset_read,
    "scorers": _scorer_read,
    "issues": _issue_read,
    "label_schemas": _label_read,
    "review_queues": _review_read,
    "prompt_optimization": _prompt_read,
    "gateway_admin": _gateway_read,
}


def seeded_stream(family: str, cell: Cell, setup: dict[str, Any], seed: int) -> list[RequestSpec]:
    rng = random.Random(f"{seed}:{family}:{cell.slug}")
    write_count = cell.requests * (90 if cell.mix == "write-heavy" else 10) // 100
    flags = [True] * write_count + [False] * (cell.requests - write_count)
    rng.shuffle(flags)
    write_index = 0
    read_index = 0
    scheduled = 0.0
    result = []
    for sequence, is_write in enumerate(flags):
        scheduled += rng.choice((0.0, 0.0, 0.0001, 0.0002))
        if is_write:
            endpoint, method, path, body = WRITERS[family](setup, cell, sequence, write_index)
            write_index += 1
        else:
            endpoint, method, path, body = READERS[family](setup, read_index)
            read_index += 1
        result.append(RequestSpec(sequence, endpoint, method, path, body, scheduled))
    return result


def sample_sequences(specs: list[RequestSpec], seed: int) -> set[int]:
    ranked = sorted(
        specs,
        key=lambda spec: hashlib.sha256(f"{seed}:{spec.sequence}".encode()).digest(),
    )
    selected = {spec.sequence for spec in ranked[:16]}
    selected.update(
        next(spec.sequence for spec in specs if spec.endpoint == endpoint)
        for endpoint in {spec.endpoint for spec in specs}
    )
    return selected


async def execute_cell(
    base_url: str,
    specs: list[RequestSpec],
    concurrency: int,
    sample: set[int],
    monitor: ResourceMonitor,
) -> MetricsCollector:
    collector = MetricsCollector()
    # The 128-client Python cells can spend more than a minute queued behind
    # the four workers and their database pools. Keep that service latency in
    # the measurements instead of turning it into a client-side error.
    client = AsyncBenchClient(
        base_url,
        concurrency,
        collector,
        timeout_seconds=CELL_TIMEOUT_SECONDS,
    )
    queue_: asyncio.Queue[RequestSpec | None] = asyncio.Queue(maxsize=concurrency * 4)
    try:
        warmup = next(
            spec for spec in specs if spec.endpoint.endswith(("_search", "_list", "_get"))
        )
        for _ in range(WARMUP_REQUESTS):
            kwargs = {"json": warmup.json_body} if warmup.json_body is not None else {}
            await client.request(
                warmup.endpoint,
                warmup.method,
                warmup.path,
                measured=False,
                capture_response=False,
                **kwargs,
            )
        monitor.started = time.monotonic()
        monitor.start()
        collector.started = time.perf_counter()
        origin = time.perf_counter()

        async def producer() -> None:
            for spec in specs:
                delay = origin + spec.scheduled_seconds - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                await queue_.put(spec)
            for _ in range(concurrency):
                await queue_.put(None)

        async def worker() -> None:
            while (spec := await queue_.get()) is not None:
                kwargs = {"json": spec.json_body} if spec.json_body is not None else {}
                await client.request(
                    spec.endpoint,
                    spec.method,
                    spec.path,
                    capture_response=spec.sequence in sample,
                    sequence=spec.sequence,
                    **kwargs,
                )

        await asyncio.gather(producer(), *(worker() for _ in range(concurrency)))
        collector.close()
        return collector
    finally:
        await client.close()


def _read_pids_current() -> int | None:
    for path in (Path("/sys/fs/cgroup/pids.current"), Path("/sys/fs/cgroup/pids/pids.current")):
        try:
            return int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            continue
    return None


def machine_state(handle: Any | None) -> dict[str, Any]:
    target_pids = set(process_tree(handle.process.pid)) if handle else set()
    matches = []
    process_count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process_count += 1
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if (
            "uvicorn" in command and "mlflow.server.fastapi_app" in command
        ) or "/mlflow-server" in command:
            matches.append({"pid": int(entry.name), "command": command})
    stray = [item for item in matches if item["pid"] not in target_pids]
    if stray:
        raise RuntimeError(f"stray benchmark target processes: {stray}")
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "load_average": list(os.getloadavg()),
        "pids_current": _read_pids_current(),
        "process_count": process_count,
        "target_tree_pids": sorted(target_pids),
        "matching_processes": matches,
    }


def build_equivalence(records: list[dict[str, Any]], sample: set[int]) -> list[dict[str, Any]]:
    selected = sorted(
        (record for record in records if record["sequence"] in sample),
        key=lambda record: record["sequence"],
    )
    return [
        {
            "endpoint": record["endpoint"],
            "method": record["method"],
            "path": record["path"],
            "response": record["response"],
            "sequence": record["sequence"],
            "status": record["status"],
        }
        for record in selected
    ]


def run_cell(
    handle: Any,
    target: str,
    family: str,
    cell: Cell,
    setup: dict[str, Any],
    seed: int,
    output: Path,
    provider_url: str,
) -> dict[str, Any]:
    state = machine_state(handle)
    specs = seeded_stream(family, cell, setup, seed)
    sample = sample_sequences(specs, seed)
    monitor = ResourceMonitor(handle.process.pid, postgres_sample)
    started_at = dt.datetime.now(dt.timezone.utc)
    try:
        collector = asyncio.run(execute_cell(handle.url, specs, cell.concurrency, sample, monitor))
    finally:
        if monitor.thread.ident is not None:
            monitor.close()
    endpoints, overall = collector.summary()
    records = collector.raw_records()
    value = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "canonical_requests": cell.canonical_requests,
            "concurrency": cell.concurrency,
            "db_pool_config": DB_POOL_CONFIG,
            "family": family,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "large_definition": LARGE_DEFINITIONS[family],
            "measured_requests": cell.requests,
            "mix": cell.mix,
            "payload_size": cell.payload_size,
            "seed": seed,
            "started_at": started_at.isoformat(),
            "target": target,
            "warmup_requests": WARMUP_REQUESTS,
            "workload": f"t23_2/{family}/{cell.slug}",
        },
        "summary": {"endpoints": endpoints, "overall": overall},
        "requests": records,
        "jobs": [],
        "resources": {
            "method": "1 s /proc whole-process-tree VmRSS and stat utime/stime sum",
            "samples": monitor.samples,
        },
        "db_pool": {
            "method": "pg_stat_activity (server-side pool occupancy proxy)",
            "samples": monitor.pool_samples,
        },
        "provider": {
            "observations": [],
            "route_latency_ms": {},
            "seed": seed,
            "url": provider_url,
        },
        "machine_state": state,
        "equivalence": {
            "jobs": [],
            "sample_seed": seed,
            "samples": build_equivalence(records, sample),
            "verdict": "PENDING",
        },
    }
    write_raw_metrics(value, output)
    return value


def mark_verdict(path: Path, verdict: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    value["equivalence"]["verdict"] = verdict
    write_raw_metrics(value, path)
    return value


def resource_numbers(value: dict[str, Any]) -> tuple[float, float, float]:
    samples = value["resources"]["samples"]
    if not samples:
        return 0.0, 0.0, 0.0
    rss = [sample["rss_bytes"] / 1024 / 1024 for sample in samples]
    first = samples[0]["utime_seconds"] + samples[0]["stime_seconds"]
    last = samples[-1]["utime_seconds"] + samples[-1]["stime_seconds"]
    return max(rss), statistics.fmean(rss), max(0.0, last - first)


def summary_markdown(output_dir: Path, families: list[str], cells: list[Cell]) -> str:
    def format_latency(percentiles: dict[str, float]) -> str:
        return "/".join(f"{percentiles[key]:.2f}" for key in ("p50", "p95", "p99", "max"))

    lines = [
        "# T23.2 CRUD + read-path benchmark summary",
        "",
        "This is raw material for T23.5, not the final Phase 23 report. Python and Rust",
        "ran serially on PostgreSQL 16 + MinIO, with a fresh DB and artifact prefix per target.",
        "Every read family had a deterministic 10,000-row corpus; warm-up requests are excluded.",
        "Both targets used pool_size=32 + max_overflow=8; PostgreSQL max_connections was 400.",
        "Final Python artifacts came from two serial target slices (datasets through review",
        "queues, then prompt optimization through gateway); Rust used one all-family target.",
        "No target loads overlapped, every slice used a fresh DB/prefix, and every per-cell",
        "resource series is complete.",
        "Compare absolute Python RSS across the slice boundary with that retained-memory caveat.",
        "The host exposed no supported cgroup pids.current file, so that raw field is null;",
        "the pre-cell /proc process count and complete target-tree PID list are still recorded.",
        "",
        "## Chosen matrix",
        "",
        "| Cell | Payload | Clients | Mix | Requests | Rationale |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for cell in cells:
        rationale = (
            "single-client write baseline"
            if cell.index == 0
            else "high-contention read path"
            if cell.index == 1
            else "mid-concurrency large-write pressure"
            if cell.index == 2
            else "high-concurrency large read path"
        )
        trimmed = (
            ""
            if cell.requests == cell.canonical_requests
            else f" (trimmed from {cell.canonical_requests})"
        )
        lines.append(
            f"| `{cell.slug}` | {cell.payload_size} | {cell.concurrency} | {cell.mix} | "
            f"{cell.requests:,}{trimmed} | {rationale} |"
        )
    lines.extend(["", "Large payload definitions:", ""])
    for family in families:
        lines.append(f"- `{family}`: {LARGE_DEFINITIONS[family]}.")
    slower = []
    for family in families:
        lines.extend([
            "",
            f"## {family}",
            "",
            "| Cell | N | Py p50/p95/p99/max ms | Rust p50/p95/p99/max ms | "
            "Py/Rust RPS | Py/Rust errors | Py RSS peak/mean MiB | "
            "Rust RSS peak/mean MiB | Py/Rust CPU-s | Eq |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for cell in cells:
            values = {}
            for target in ("python", "rust"):
                path = output_dir / f"{family}-{cell.slug}-{target}.json"
                values[target] = json.loads(path.read_text())
            py = values["python"]
            rust = values["rust"]
            pyl = py["summary"]["overall"]["latency_ms"]
            rul = rust["summary"]["overall"]["latency_ms"]
            pyres = resource_numbers(py)
            rures = resource_numbers(rust)
            if rul["p95"] > pyl["p95"]:
                slower.append((family, cell.slug, pyl["p95"], rul["p95"]))
            lines.append(
                f"| `{cell.slug}` | {cell.requests:,} | {format_latency(pyl)} | "
                f"{format_latency(rul)} | "
                f"{py['summary']['overall']['rps']:.1f}/{rust['summary']['overall']['rps']:.1f} | "
                f"{py['summary']['overall']['errors']}/{rust['summary']['overall']['errors']} | "
                f"{pyres[0]:.1f}/{pyres[1]:.1f} | {rures[0]:.1f}/{rures[1]:.1f} | "
                f"{pyres[2]:.2f}/{rures[2]:.2f} | {rust['equivalence']['verdict']} |"
            )
    lines.extend(["", "## Rust-slower cells", ""])
    if slower:
        lines.append("Rust p95 exceeded Python p95 in:")
        lines.append("")
        for family, slug, python_p95, rust_p95 in slower:
            lines.append(
                f"- `{family}/{slug}`: Python {python_p95:.2f} ms, Rust {rust_p95:.2f} ms."
            )
    else:
        lines.append("No cell had Rust p95 above Python p95.")
    lines.extend(["", "## Raw result inventory", ""])
    for path in sorted(output_dir.glob("*.json")):
        lines.append(f"- `{path.name}`")
    return "\n".join(lines)


def matrix(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    families = list(args.families)
    cells = [
        cell
        for cell in cell_matrix(args.small_requests, args.large_requests)
        if not args.cells or cell.slug in args.cells
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    stub_path, cleanup_path = install_claude_stub()
    try:
        machine_state(None)
        run_command(compose_args("up", "-d", "--wait", "postgres", "minio"))
        run_command(compose_args("run", "--rm", "minio-init"))
        if not args.skip_build:
            run_command(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "mlflow-server",
                    "-p",
                    "mlflow-genai-worker",
                ],
                cwd=RUST_ROOT,
            )
        with provider_server(args.seed) as provider:
            provider_url = f"http://127.0.0.1:{provider.server_port}"
            for target in args.targets:
                recreate_database()
                with tempfile.TemporaryDirectory(prefix=f"mlflow-t23-2-{target}-") as temporary:
                    workdir = Path(temporary)
                    handle = launch_server(
                        target,
                        workdir,
                        provider_url,
                        f"t23-2/{target}-{time.time_ns()}",
                        stub_path,
                        {
                            "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW": str(
                                DB_POOL_CONFIG["max_overflow"]
                            ),
                            "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE": str(DB_POOL_CONFIG["pool_size"]),
                        },
                    )
                    try:
                        setup = setup_matrix_target(handle.url, provider_url, args.seed)
                        setup["seed"] = args.seed
                        print(f"[{target}] seeding deterministic corpora", flush=True)
                        seed_matrix_corpora(setup, args.seed)
                        for family in families:
                            for cell in cells:
                                output = output_dir / f"{family}-{cell.slug}-{target}.json"
                                print(
                                    f"[{target}] {family}/{cell.slug}: {cell.requests} requests",
                                    flush=True,
                                )
                                value = run_cell(
                                    handle,
                                    target,
                                    family,
                                    cell,
                                    setup,
                                    args.seed,
                                    output,
                                    provider_url,
                                )
                                if value["summary"]["overall"]["error_rate"] >= 0.0001:
                                    raise RuntimeError(
                                        f"{target} {family}/{cell.slug} error rate was "
                                        f"{value['summary']['overall']['error_rate']:.6%}"
                                    )
                                python_path = output_dir / f"{family}-{cell.slug}-python.json"
                                if target == "rust" and python_path.exists():
                                    python_value = json.loads(python_path.read_text())
                                    differences = compare_runs(python_value, value)
                                    verdict = "FAIL" if differences else "PASS"
                                    mark_verdict(python_path, verdict)
                                    mark_verdict(output, verdict)
                                    if differences:
                                        raise RuntimeError(
                                            f"equivalence failed for {family}/{cell.slug}:\n"
                                            + "\n".join(differences)
                                        )
                                print(
                                    f"[{target}] {family}/{cell.slug}: "
                                    f"{value['summary']['overall']['rps']:.1f} RPS, "
                                    f"{value['summary']['overall']['errors']} errors",
                                    flush=True,
                                )
                    finally:
                        stop_server(handle)
            if set(args.targets) == {"python", "rust"}:
                summary = summary_markdown(output_dir, families, cells)
                (output_dir / "t23_2_summary.md").write_text(summary + "\n")
                print(f"summary: {output_dir / 't23_2_summary.md'}", flush=True)
        for path in output_dir.glob("*.json"):
            validate_raw_metrics(json.loads(path.read_text()))
        return 0
    finally:
        run_command(compose_args("down", "-v", "--remove-orphans"), check=False)
        shutil.rmtree(cleanup_path, ignore_errors=True)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "t23_2")
    parser.add_argument("--seed", type=int, default=2320)
    parser.add_argument("--small-requests", type=int, default=CANONICAL_SMALL_REQUESTS)
    parser.add_argument("--large-requests", type=int, default=CANONICAL_LARGE_REQUESTS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--cells", nargs="+", default=[])
    parser.add_argument(
        "--targets", nargs="+", choices=("python", "rust"), default=["python", "rust"]
    )
    parser.set_defaults(func=matrix)
