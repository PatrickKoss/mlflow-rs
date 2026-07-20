"""Build the T15.5 reachability ledger and pinned compatibility manifests.

Run with the pinned LiteLLM reference available so the D16 manifest is derived
from the wheel rather than from a moving checkout::

    uv run --with 'litellm==1.91.2' --with 'dspy==3.2.1' \
        python rust/genai-inventory/build_inventory.py

The ordinary verification path does not run this script. ``validate_ledger.py``
instead checks the committed snapshot against the current source tree.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENAI_ROOT = ROOT / "mlflow" / "genai"

CLASSIFICATIONS = {"server_reachable", "client_only", "dead"}
TEST_CLASSIFICATIONS = {"server_reachable", "client_only", "python_internal"}
REFERENCE_MLFLOW_GIT_SHA = "c69051f534f4b0d171ed92d07c58a20f8c2a3461"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(path: str, pattern: str) -> int:
    for line_no, text in enumerate((ROOT / path).read_text().splitlines(), 1):
        if pattern in text:
            return line_no
    raise ValueError(f"{pattern!r} not found in {path}")


def _ref(path: str, pattern: str | None = None, line: int | None = None) -> str:
    if line is None:
        line = _line(path, pattern) if pattern else 1
    return f"{path}:{line}"


TEST_SCOPE_DIRS = (
    "tests/genai/",
    "tests/gateway/",
    "tests/assistant/",
    "tests/server/assistant/",
    "tests/server/jobs/",
)
TEST_KEYWORDS = (
    "archive_traces",
    "trace_archival",
    "promptlab",
    "gateway",
    "prompt_optimization",
    "review_queue",
    "label_schema",
    "evaluation_dataset",
    "online_scor",
    "scorer",
    "issue_detection",
)


def _walk_test_defs(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text())
    rel = path.relative_to(ROOT).as_posix()
    records: list[dict[str, Any]] = []

    def walk(body: list[ast.stmt], parents: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, [*parents, node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join([*parents, node.name])
                if node.name.startswith("test_"):
                    records.append({
                        "id": f"{rel}::{qualname}",
                        "path": rel,
                        "line": node.lineno,
                        "qualname": qualname,
                    })
                walk(node.body, [*parents, node.name])

    walk(tree.body, [])
    return records


def _test_inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        in_scope_dir = rel.startswith(TEST_SCOPE_DIRS)
        if not in_scope_dir and not any(keyword in rel.lower() for keyword in TEST_KEYWORDS):
            continue
        for record in _walk_test_defs(path):
            record["classification"] = _classify_test(record["id"])
            records.append(record)
    return records


def _classify_test(test_id: str) -> str:
    client_markers = (
        "/agent_server",
        "/agent_tester",
        "/git_versioning",
        "/labeling/",
        "/simulators/",
        "/scorers/google_adk/",
        "/scorers/guardrails/",
        "/judges/optimizers/test_dspy",
        "/judges/optimizers/test_gepa",
        "/judges/optimizers/test_simba",
    )
    if any(marker in test_id for marker in client_markers):
        return "client_only"
    if test_id.startswith("tests/genai/test_rust_http_conformance.py::"):
        return "server_reachable"
    # The remaining tests exercise Python handlers, stores, workers, or
    # monkeypatched implementation classes directly. Their source targets may
    # be server-reachable, but the test process cannot be repointed across an
    # HTTP boundary. T22.2 retains them in the inventory without presenting
    # them as Rust-server conformance.
    return "python_internal"


SERVER_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    ("mlflow/genai/discovery/", "T19.4", "mlflow-genai", "C", "issue discovery job"),
    ("mlflow/genai/evaluation/", "T19.2", "mlflow-genai", "C", "evaluation job"),
    (
        "mlflow/genai/label_schemas/label_schemas.py",
        "T16.3",
        "mlflow-store",
        "A",
        "label-schema entity conversion",
    ),
    (
        "mlflow/genai/label_schemas/validation.py",
        "T16.3",
        "mlflow-store",
        "A",
        "label-schema store validation",
    ),
    (
        "mlflow/genai/review_queues/review_queues.py",
        "T16.4",
        "mlflow-store",
        "A",
        "review-queue entities",
    ),
    (
        "mlflow/genai/review_queues/validation.py",
        "T16.4",
        "mlflow-store",
        "A",
        "review-queue store validation",
    ),
    ("mlflow/genai/optimize/", "T19.5", "mlflow-genai", "C", "prompt optimization job"),
    ("mlflow/genai/scorers/online/", "T19.2", "worker", "C", "online-scoring scheduler"),
    ("mlflow/genai/scorers/job.py", "T19.2", "worker", "C", "scorer jobs"),
    (
        "mlflow/genai/scorers/deepeval/",
        "T19.3",
        "mlflow-genai",
        "C",
        "server-allowlisted DeepEval scorer family",
    ),
    (
        "mlflow/genai/scorers/ragas/",
        "T19.3",
        "mlflow-genai",
        "C",
        "server-allowlisted Ragas scorer family",
    ),
    (
        "mlflow/genai/scorers/trulens/",
        "T19.3",
        "mlflow-genai",
        "C",
        "server-allowlisted TruLens scorer family",
    ),
    (
        "mlflow/genai/scorers/phoenix/",
        "T19.3",
        "mlflow-genai",
        "C",
        "server-allowlisted Phoenix scorer family",
    ),
    ("mlflow/genai/scorers/", "T19.1", "mlflow-genai", "C", "serialized scorer execution"),
    ("mlflow/genai/judges/", "T19.1", "mlflow-genai", "C", "serialized judge execution"),
    ("mlflow/genai/utils/", "T19.2", "mlflow-genai", "C", "evaluation/scorer utilities"),
)

SERVER_ANCHORS = {
    "mlflow/genai/discovery/": _ref("mlflow/server/handlers.py", "from mlflow.genai.discovery.job"),
    "mlflow/genai/evaluation/": _ref(
        "mlflow/server/handlers.py", "from mlflow.genai.evaluation.job"
    ),
    "mlflow/genai/label_schemas/": _ref(
        "mlflow/server/handlers.py", "from mlflow.genai.label_schemas.label_schemas"
    ),
    "mlflow/genai/review_queues/": _ref(
        "mlflow/server/handlers.py", "from mlflow.genai.review_queues import"
    ),
    "mlflow/genai/optimize/": _ref("mlflow/server/handlers.py", "from mlflow.genai.optimize.job"),
    "mlflow/genai/scorers/online/": _ref(
        "mlflow/server/jobs/utils.py",
        "from mlflow.genai.scorers.job import run_online_scoring_scheduler",
    ),
    "mlflow/genai/scorers/job.py": _ref(
        "mlflow/server/handlers.py", "from mlflow.genai.scorers.job import"
    ),
    "mlflow/genai/scorers/": _ref(
        "mlflow/server/handlers.py", "from mlflow.genai.scorers.base import Scorer"
    ),
    "mlflow/genai/judges/": _ref(
        "mlflow/store/tracking/sqlalchemy_store.py",
        "from mlflow.genai.judges.instructions_judge import",
    ),
    "mlflow/genai/utils/": _ref(
        "mlflow/genai/evaluation/harness.py", "from mlflow.genai.utils.trace_utils import"
    ),
}


def _server_rule(path: str) -> tuple[str, str, str, str] | None:
    for prefix, task, owner, tier, reason in SERVER_RULES:
        if path.startswith(prefix):
            return task, owner, tier, reason
    return None


def _anchor(path: str) -> str:
    for prefix, anchor in SERVER_ANCHORS.items():
        if path.startswith(prefix):
            return anchor
    return _ref(path)


def _classification(path: str, qualname: str | None, kind: str) -> dict[str, Any]:
    # Explicit Part II exclusions: these integrations construct SDK-side objects and
    # are not accepted by the OSS scorer deserializer allowlist.
    if path.startswith(("mlflow/genai/scorers/google_adk/", "mlflow/genai/scorers/guardrails/")):
        return _client(
            path,
            "SDK-only scorer family; absent from THIRD_PARTY_SCORER_ALLOWED_MODULES",
            _ref("mlflow/genai/scorers/scorer_utils.py", "THIRD_PARTY_SCORER_ALLOWED_MODULES"),
        )

    if path.startswith((
        "mlflow/genai/judges/optimizers/dspy.py",
        "mlflow/genai/judges/optimizers/dspy_utils.py",
        "mlflow/genai/judges/optimizers/gepa.py",
        "mlflow/genai/judges/optimizers/simba.py",
    )):
        return _client(path, "judge alignment API is invoked by the Python SDK, not a server job")

    if path == "mlflow/genai/judges/optimizers/memalign/optimizer.py" and qualname:
        if qualname == "MemAlignOptimizer" or qualname.startswith("MemAlignOptimizer."):
            return _client(
                path,
                "MemAlign alignment mutates a judge client-side; only the serialized judge "
                "executes",
            )

    if path == "mlflow/genai/optimize/__init__.py" and qualname == "optimize_prompt":
        return _client(path, "deprecated singular prompt-optimization SDK wrapper")

    # Prompt optimization's server job needs only these registry operations.
    if path == "mlflow/genai/prompts/__init__.py":
        if (
            kind == "module"
            or qualname
            in {
                "suppress_genai_migration_warning",
                "register_prompt",
                "load_prompt",
            }
            or (qualname and qualname.startswith("suppress_genai_migration_warning."))
        ):
            return _server(
                path, "T19.5", "mlflow-genai", "C", "prompt load/register in optimization"
            )
        return _client(
            path, "prompt-registry SDK operation not called by the server optimization job"
        )

    # The optimization job resolves one dataset, but dataset CRUD remains a Python client API.
    if path == "mlflow/genai/datasets/__init__.py":
        server_names = {
            "_databricks_profile_env",
            "_validate_databricks_params",
            "_validate_non_databricks_get_params",
            "_get_dataset_by_name",
            "get_dataset",
        }
        if (
            kind == "module"
            or qualname in server_names
            or any(qualname and qualname.startswith(f"{name}.") for name in server_names)
        ):
            return _server(path, "T19.5", "worker", "C", "prompt optimization dataset resolution")
        return _client(path, "evaluation-dataset CRUD SDK operation")
    if path == "mlflow/genai/datasets/evaluation_dataset.py":
        return _server(
            path, "T19.5", "mlflow-genai", "C", "dataset entity consumed by optimization"
        )

    if path in {
        "mlflow/genai/label_schemas/__init__.py",
        "mlflow/genai/review_queues/__init__.py",
    }:
        if kind == "module":
            task = "T16.3" if "label_schemas" in path else "T16.4"
            return _server(
                path, task, "mlflow-server", "A", "server imports exported enums/entities"
            )
        return _client(path, "public SDK wrapper around the corresponding HTTP CRUD surface")

    if path == "mlflow/genai/__init__.py":
        return _server(path, "T19.2", "mlflow-genai", "C", "parent package of native job imports")

    rule = _server_rule(path)
    if rule:
        result = _server(path, *rule)
        if kind != "module":
            result["ambiguity"] = (
                "The module has a proven server/job call root, but serialized and dynamic dispatch "
                "make per-symbol static reachability incomplete; classified server_reachable "
                "conservatively."
            )
        return result

    client_groups = {
        "mlflow/genai/agent_server/": "standalone user-agent serving process, not mlflow server",
        "mlflow/genai/agent_tester.py": "SDK-side agent test generator",
        "mlflow/genai/datasets/databricks": "Databricks dataset source SDK",
        "mlflow/genai/git_versioning/": "local Git/SDK context",
        "mlflow/genai/labeling/": "Databricks review-app SDK",
        "mlflow/genai/scheduled_scorers.py": "Databricks scheduled-scorer SDK model",
        "mlflow/genai/simulators/": "SDK-side conversation simulation",
        "mlflow/genai/utils/display_utils.py": "notebook/display-only output",
    }
    for prefix, reason in client_groups.items():
        if path.startswith(prefix):
            evidence = None
            if prefix == "mlflow/genai/labeling/":
                evidence = _ref("mlflow/genai/labeling/labeling.py", "databricks.agents.review_app")
            elif prefix == "mlflow/genai/scheduled_scorers.py":
                evidence = _ref(path, "databricks-agents")
            elif prefix == "mlflow/genai/agent_server/":
                evidence = _ref("mlflow/genai/agent_server/server.py", "def run(")
            return _client(path, reason, evidence)

    return _client(path, "public Python SDK helper with no MLflow server/job call root")


def _server(path: str, task: str, owner: str, tier: str, reason: str) -> dict[str, Any]:
    return {
        "classification": "server_reachable",
        "task": task,
        "native_owner_crate": owner,
        "tier": tier,
        "classification_reason": reason,
        "reachability_evidence": _anchor(path),
        "ambiguity": None,
    }


def _client(path: str, reason: str, evidence: str | None = None) -> dict[str, Any]:
    return {
        "classification": "client_only",
        "task": None,
        "native_owner_crate": None,
        "tier": None,
        "classification_reason": reason,
        "reachability_evidence": evidence or _ref(path),
        "ambiguity": None,
    }


def _definitions(path: Path) -> list[tuple[str, str | None, int]]:
    tree = ast.parse(path.read_text())
    result: list[tuple[str, str | None, int]] = [("module", None, 1)]

    def walk(body: list[ast.stmt], parents: list[str], parent_kind: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualname = ".".join([*parents, node.name])
                result.append(("class", qualname, node.lineno))
                walk(node.body, [*parents, node.name], "class")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join([*parents, node.name])
                if parent_kind == "class":
                    kind = "method"
                elif parents:
                    kind = "nested_function"
                else:
                    kind = "function"
                result.append((kind, qualname, node.lineno))
                walk(node.body, [*parents, node.name], "function")

    walk(tree.body, [], "module")
    return result


def _fixture_for(path: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    source = Path(path)
    stem = source.stem
    relative = source.relative_to("mlflow/genai")
    test_parent = relative.parent.as_posix().replace("evaluation", "evaluate")
    test_prefix = "tests/genai/" if test_parent == "." else f"tests/genai/{test_parent}/"
    aliases = {stem}
    parent_name = relative.parent.name.replace("evaluation", "evaluate")
    if stem == "__init__":
        aliases.add(parent_name)
    if stem in {"base", "entities", "harness", "utils", "validation"}:
        aliases.add(parent_name.removesuffix("s"))
    if stem == "builtin_scorers":
        aliases.update({"builtin_scorers", "serialization"})
    if stem == "job":
        aliases.update({parent_name, "job"})
    candidates = [
        test["id"]
        for test in tests
        if test["path"].startswith(test_prefix)
        and any(
            f"/test_{alias}.py" in test["path"] or f"test_{alias}_" in test["qualname"]
            for alias in aliases
        )
    ]
    if candidates:
        return {"strategy": "existing_python_tests", "tests": sorted(candidates)[:40]}
    return {
        "strategy": "corpus recorder needed",
        "recorder": "semantic" if "/gateway" not in path else "sse",
        "tests": [],
    }


def _genai_items(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in sorted(GENAI_ROOT.rglob("*.py")):
        path = source.relative_to(ROOT).as_posix()
        for kind, qualname, line in _definitions(source):
            classification = _classification(path, qualname, kind)
            symbol = qualname or "<module>"
            entry = {
                "id": f"genai:{path}:{line}:{symbol}",
                "scope": "mlflow/genai",
                "symbol_kind": kind,
                "module": path.removesuffix(".py").replace("/", "."),
                "qualname": qualname,
                "source": f"{path}:{line}",
                **classification,
            }
            entry["fixture_oracle"] = _fixture_for(path, tests)
            items.append(entry)
    return items


PROTO_GROUPS = {
    "datasets": {
        "rpcs": {
            "createDataset",
            "getDataset",
            "deleteDataset",
            "searchEvaluationDatasets",
            "setDatasetTags",
            "deleteDatasetTag",
            "upsertDatasetRecords",
            "getDatasetExperimentIds",
            "getDatasetRecords",
            "deleteDatasetRecords",
            "addDatasetToExperiments",
            "removeDatasetFromExperiments",
        },
        "task": "T16.1",
        "owner": "mlflow-store",
    },
    "scorers": {
        "rpcs": {
            "registerScorer",
            "listScorers",
            "listScorerVersions",
            "getScorer",
            "deleteScorer",
        },
        "task": "T16.2",
        "owner": "mlflow-store",
    },
    "issues": {
        "rpcs": {"createIssue", "updateIssue", "getIssue", "searchIssues"},
        "task": "T16.3",
        "owner": "mlflow-store",
    },
    "label_schemas": {
        "rpcs": {
            "createLabelSchema",
            "getLabelSchema",
            "getLabelSchemaByName",
            "listLabelSchemas",
            "updateLabelSchema",
            "deleteLabelSchema",
        },
        "task": "T16.3",
        "owner": "mlflow-store",
    },
    "review_queues": {
        "rpcs": {
            "createReviewQueue",
            "getOrCreateUserQueue",
            "getReviewQueue",
            "getReviewQueueByName",
            "listReviewQueues",
            "updateReviewQueue",
            "deleteReviewQueue",
            "addItemsToReviewQueue",
            "removeItemsFromReviewQueue",
            "listReviewQueueItems",
            "setReviewQueueItemStatus",
        },
        "task": "T16.4",
        "owner": "mlflow-store",
    },
    "prompt_optimization": {
        "rpcs": {
            "createPromptOptimizationJob",
            "getPromptOptimizationJob",
            "searchPromptOptimizationJobs",
            "cancelPromptOptimizationJob",
            "deletePromptOptimizationJob",
        },
        "task": "T16.6",
        "owner": "mlflow-server",
    },
    "gateway_crud": {
        "rpcs": {
            "createGatewaySecret",
            "getGatewaySecretInfo",
            "updateGatewaySecret",
            "deleteGatewaySecret",
            "listGatewaySecretInfos",
            "createGatewayEndpoint",
            "getGatewayEndpoint",
            "updateGatewayEndpoint",
            "deleteGatewayEndpoint",
            "listGatewayEndpoints",
            "createGatewayModelDefinition",
            "getGatewayModelDefinition",
            "listGatewayModelDefinitions",
            "updateGatewayModelDefinition",
            "deleteGatewayModelDefinition",
            "attachModelToEndpoint",
            "detachModelFromEndpoint",
            "createEndpointBinding",
            "deleteEndpointBinding",
            "listEndpointBindings",
            "setGatewayEndpointTag",
            "deleteGatewayEndpointTag",
            "createBudgetPolicy",
            "getBudgetPolicy",
            "updateBudgetPolicy",
            "deleteBudgetPolicy",
            "listBudgetPolicies",
            "listBudgetWindows",
            "createGatewayGuardrail",
            "getGatewayGuardrail",
            "deleteGatewayGuardrail",
            "listGatewayGuardrails",
            "addGuardrailToEndpoint",
            "removeGuardrailFromEndpoint",
            "listEndpointGuardrailConfigs",
            "updateEndpointGuardrailConfig",
        },
        "task": "T18.1",
        "owner": "mlflow-store",
    },
}


def _area_tests(area: str, tests: list[dict[str, Any]]) -> list[str]:
    tokens = {
        "datasets": ("dataset",),
        "scorers": ("scorer",),
        "issues": ("issue",),
        "label_schemas": ("label_schema",),
        "review_queues": ("review_queue",),
        "prompt_optimization": ("prompt_optimization", "optimize"),
        "gateway_crud": ("gateway",),
        "gateway_runtime": ("gateway", "provider"),
        "assistant": ("assistant",),
        "archival": ("archiv",),
        "jobs": ("job",),
        "promptlab": ("promptlab",),
    }.get(area, (area,))
    return sorted(
        test["id"] for test in tests if any(token in test["id"].lower() for token in tokens)
    )[:80]


def _surface_entry(
    *,
    name: str,
    area: str,
    task: str,
    owner: str,
    tier: str,
    evidence: str,
    tests: list[dict[str, Any]],
    method: str | None = None,
    paths: list[str] | None = None,
    kind: str = "http_route",
    ambiguity: str | None = None,
) -> dict[str, Any]:
    test_ids = _area_tests(area, tests)
    fixture = (
        {"strategy": "existing_python_tests", "tests": test_ids}
        if test_ids
        else {"strategy": "corpus recorder needed", "recorder": "sse", "tests": []}
    )
    return {
        "id": f"surface:{area}:{name}:{method or kind}",
        "scope": "part_ii_surface",
        "symbol_kind": kind,
        "module": None,
        "qualname": name,
        "source": evidence,
        "classification": "server_reachable",
        "task": task,
        "native_owner_crate": owner,
        "tier": tier,
        "classification_reason": f"Part II {area} server surface",
        "reachability_evidence": evidence,
        "ambiguity": ambiguity,
        "http_method": method,
        "paths": paths or [],
        "fixture_oracle": fixture,
    }


def _proto_surfaces(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from mlflow.protos import databricks_pb2
    from mlflow.protos.service_pb2 import MlflowService

    rpc_to_group = {
        rpc: (area, data) for area, data in PROTO_GROUPS.items() for rpc in data["rpcs"]
    }
    entries: list[dict[str, Any]] = []
    for method in MlflowService.DESCRIPTOR.methods:
        if method.name not in rpc_to_group:
            continue
        area, data = rpc_to_group[method.name]
        endpoints = method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        for index, endpoint in enumerate(endpoints):
            proto_path = endpoint.path
            evidence = _ref("mlflow/protos/service.proto", f'path: "{proto_path}"')
            version = endpoint.since.major
            paths = [f"/api/{version}.0{proto_path}", f"/ajax-api/{version}.0{proto_path}"]
            ambiguity = None
            if area in {"scorers", "gateway_crud"} and version == 3:
                ambiguity = (
                    "§12 labels this family /2.0, but get_service_endpoints uses since.major=3; "
                    "the generated descriptor registers /api/3.0 and /ajax-api/3.0."
                )
            entries.append(
                _surface_entry(
                    name=f"{method.name}[{index}]",
                    area=area,
                    task=data["task"],
                    owner=data["owner"],
                    tier="A",
                    evidence=evidence,
                    tests=tests,
                    method=endpoint.method,
                    paths=paths,
                    ambiguity=ambiguity,
                )
            )
    return entries


def _hand_surfaces(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (
            "evaluate_invoke",
            "jobs",
            "T17.4",
            "mlflow-server",
            "A",
            "POST",
            ["/ajax-api/3.0/mlflow/genai/evaluate/invoke"],
            "mlflow/server/handlers.py",
            "get_genai_evaluate_endpoints",
        ),
        (
            "scorer_invoke",
            "jobs",
            "T17.4",
            "mlflow-server",
            "A",
            "POST",
            ["/ajax-api/3.0/mlflow/scorer/invoke"],
            "mlflow/server/handlers.py",
            "get_gateway_endpoints",
        ),
        (
            "issue_invoke",
            "jobs",
            "T17.4",
            "mlflow-server",
            "A",
            "POST",
            ["/ajax-api/3.0/mlflow/issues/invoke"],
            "mlflow/server/handlers.py",
            "get_issues_detection_endpoints",
        ),
        (
            "job_get",
            "jobs",
            "T16.5",
            "mlflow-server",
            "A",
            "GET",
            ["/ajax-api/3.0/mlflow/jobs/{job_id}"],
            "mlflow/server/handlers.py",
            "def get_job_endpoints",
        ),
        (
            "job_cancel",
            "jobs",
            "T16.5",
            "mlflow-server",
            "A",
            "PATCH",
            ["/ajax-api/3.0/mlflow/jobs/cancel/{job_id}"],
            "mlflow/server/handlers.py",
            "def get_job_endpoints",
        ),
        (
            "online_configs_get",
            "scorers",
            "T16.2",
            "mlflow-server",
            "A",
            "GET",
            [
                "/api/3.0/mlflow/scorers/online-configs",
                "/ajax-api/3.0/mlflow/scorers/online-configs",
            ],
            "mlflow/server/handlers.py",
            "online-configs",
        ),
        (
            "online_config_put",
            "scorers",
            "T16.2",
            "mlflow-server",
            "A",
            "PUT",
            ["/api/3.0/mlflow/scorers/online-config", "/ajax-api/3.0/mlflow/scorers/online-config"],
            "mlflow/server/handlers.py",
            "online-config",
        ),
        (
            "supported_providers",
            "gateway_crud",
            "T18.2",
            "mlflow-server",
            "A",
            "GET",
            ["/ajax-api/3.0/mlflow/gateway/supported-providers"],
            "mlflow/server/handlers.py",
            "supported-providers",
        ),
        (
            "supported_models",
            "gateway_crud",
            "T18.2",
            "mlflow-server",
            "A",
            "GET",
            ["/ajax-api/3.0/mlflow/gateway/supported-models"],
            "mlflow/server/handlers.py",
            "supported-models",
        ),
        (
            "provider_config",
            "gateway_crud",
            "T18.2",
            "mlflow-server",
            "A",
            "GET",
            ["/ajax-api/3.0/mlflow/gateway/provider-config"],
            "mlflow/server/handlers.py",
            "provider-config",
        ),
        (
            "secrets_config",
            "gateway_crud",
            "T18.2",
            "mlflow-server",
            "A",
            "GET",
            ["/ajax-api/3.0/mlflow/gateway/secrets/config"],
            "mlflow/server/handlers.py",
            "secrets/config",
        ),
        (
            "gateway_proxy_get",
            "gateway_crud",
            "T18.2",
            "mlflow-server",
            "A",
            "GET",
            ["/ajax-api/2.0/mlflow/gateway-proxy"],
            "mlflow/server/__init__.py",
            "gateway-proxy",
        ),
        (
            "gateway_proxy_post",
            "gateway_crud",
            "T18.2",
            "mlflow-server",
            "A",
            "POST",
            ["/ajax-api/2.0/mlflow/gateway-proxy"],
            "mlflow/server/__init__.py",
            "gateway-proxy",
        ),
        (
            "unified_invocations",
            "gateway_runtime",
            "T18.3",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/{endpoint_name}/mlflow/invocations"],
            "mlflow/server/gateway_api.py",
            '"/{endpoint_name}/mlflow/invocations"',
        ),
        (
            "mlflow_chat_completions",
            "gateway_runtime",
            "T18.3",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/mlflow/v1/chat/completions"],
            "mlflow/server/gateway_api.py",
            '"/mlflow/v1/chat/completions"',
        ),
        (
            "openai_chat",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/openai/v1/chat/completions"],
            "mlflow/server/gateway_api.py",
            "async def openai_passthrough_chat",
        ),
        (
            "openai_embeddings",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/openai/v1/embeddings"],
            "mlflow/server/gateway_api.py",
            "async def openai_passthrough_embeddings",
        ),
        (
            "openai_responses",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/openai/v1/responses"],
            "mlflow/server/gateway_api.py",
            "async def openai_passthrough_responses",
        ),
        (
            "openai_responses_compact",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/openai/v1/responses/compact"],
            "mlflow/server/gateway_api.py",
            "async def openai_passthrough_responses_compact",
        ),
        (
            "anthropic_messages",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/anthropic/v1/messages"],
            "mlflow/server/gateway_api.py",
            "async def anthropic_passthrough_messages",
        ),
        (
            "gemini_generate",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/gemini/v1beta/models/{model}:generateContent"],
            "mlflow/server/gateway_api.py",
            "async def gemini_passthrough_generate_content",
        ),
        (
            "gemini_stream",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "POST",
            ["/gateway/gemini/v1beta/models/{model}:streamGenerateContent"],
            "mlflow/server/gateway_api.py",
            "async def gemini_passthrough_stream_generate_content",
        ),
        (
            "raw_proxy",
            "gateway_runtime",
            "T18.4",
            "mlflow-server",
            "B",
            "ANY",
            ["/gateway/proxy/{endpoint_name}/{path...}"],
            "mlflow/server/gateway_api.py",
            '"/proxy/{endpoint_name}/{path:path}"',
        ),
        (
            "assistant_message",
            "assistant",
            "T20.1",
            "mlflow-server",
            "B",
            "POST",
            ["/ajax-api/3.0/mlflow/assistant/message"],
            "mlflow/server/assistant/api.py",
            '@assistant_router.post("/message"',
        ),
        (
            "assistant_stream",
            "assistant",
            "T20.1",
            "mlflow-server",
            "B",
            "GET",
            ["/ajax-api/3.0/mlflow/assistant/sessions/{id}/stream"],
            "mlflow/server/assistant/api.py",
            '"/sessions/{session_id}/stream"',
        ),
        (
            "assistant_cancel",
            "assistant",
            "T20.1",
            "mlflow-server",
            "B",
            "PATCH",
            ["/ajax-api/3.0/mlflow/assistant/sessions/{id}"],
            "mlflow/server/assistant/api.py",
            '@assistant_router.patch("/sessions/{session_id}"',
        ),
        (
            "assistant_permission",
            "assistant",
            "T20.3",
            "mlflow-server",
            "B",
            "POST",
            ["/ajax-api/3.0/mlflow/assistant/sessions/{id}/permission"],
            "mlflow/server/assistant/api.py",
            '"/sessions/{session_id}/permission"',
        ),
        (
            "assistant_health",
            "assistant",
            "T20.2",
            "mlflow-server",
            "B",
            "GET",
            ["/ajax-api/3.0/mlflow/assistant/providers/{provider}/health"],
            "mlflow/server/assistant/api.py",
            '"/providers/{provider}/health"',
        ),
        (
            "assistant_config_get",
            "assistant",
            "T20.1",
            "mlflow-server",
            "B",
            "GET",
            ["/ajax-api/3.0/mlflow/assistant/config"],
            "mlflow/server/assistant/api.py",
            '@assistant_router.get("/config"',
        ),
        (
            "assistant_config_put",
            "assistant",
            "T20.1",
            "mlflow-server",
            "B",
            "PUT",
            ["/ajax-api/3.0/mlflow/assistant/config"],
            "mlflow/server/assistant/api.py",
            '@assistant_router.put("/config"',
        ),
        (
            "assistant_skills_install",
            "assistant",
            "T20.1",
            "mlflow-server",
            "B",
            "POST",
            ["/ajax-api/3.0/mlflow/assistant/skills/install"],
            "mlflow/server/assistant/api.py",
            '"/skills/install"',
        ),
        (
            "assistant_models",
            "assistant",
            "T20.2",
            "mlflow-server",
            "B",
            "GET",
            ["/ajax-api/3.0/mlflow/assistant/providers/{provider}/models"],
            "mlflow/server/assistant/api.py",
            '"/providers/{provider}/models"',
        ),
        (
            "promptlab_create_run",
            "promptlab",
            "T20.4",
            "mlflow-server",
            "A",
            "POST",
            ["/ajax-api/2.0/mlflow/runs/create-promptlab-run"],
            "mlflow/server/__init__.py",
            "create-promptlab-run",
        ),
    ]
    entries = []
    for name, area, task, owner, tier, method, paths, file, pattern in specs:
        entries.append(
            _surface_entry(
                name=name,
                area=area,
                task=task,
                owner=owner,
                tier=tier,
                evidence=_ref(file, pattern),
                tests=tests,
                method=method,
                paths=paths,
            )
        )

    components = [
        (
            "job_runner_state_machine",
            "jobs",
            "T17.1",
            "worker",
            "B",
            "mlflow/server/jobs/_job_runner.py",
            'if __name__ == "__main__"',
        ),
        (
            "native_worker_protocol",
            "jobs",
            "T17.2",
            "worker",
            "C",
            "mlflow/server/jobs/_job_subproc_entry.py",
            "def _main",
        ),
        (
            "online_scoring_scheduler",
            "jobs",
            "T17.3",
            "worker",
            "C",
            "mlflow/server/jobs/utils.py",
            "def online_scoring_scheduler",
        ),
        (
            "traffic_split_fallback",
            "gateway_runtime",
            "T18.5",
            "mlflow-server",
            "B",
            "mlflow/gateway/providers/base.py",
            "class FallbackProvider",
        ),
        (
            "budget_enforcement",
            "gateway_runtime",
            "T18.6",
            "mlflow-server",
            "B",
            "mlflow/gateway/budget.py",
            "def check_budget_limit",
        ),
        (
            "guardrail_execution",
            "gateway_runtime",
            "T18.7",
            "mlflow-genai",
            "C",
            "mlflow/gateway/guardrails.py",
            "class JudgeGuardrail",
        ),
        (
            "assistant_cli_providers",
            "assistant",
            "T20.2",
            "mlflow-server",
            "B",
            "mlflow/assistant/providers/claude_code.py",
            "class ClaudeCodeProvider",
        ),
        (
            "assistant_tool_loop",
            "assistant",
            "T20.3",
            "mlflow-server",
            "B",
            "mlflow/assistant/providers/tool_executor.py",
            "async def execute_tool",
        ),
        (
            "archival_config",
            "archival",
            "T21.1",
            "mlflow-server",
            "B",
            "mlflow/tracing/trace_archival_config.py",
            "class TraceArchivalServerConfig",
        ),
        (
            "archival_store",
            "archival",
            "T21.2",
            "mlflow-store",
            "B",
            "mlflow/store/tracking/sqlalchemy_store.py",
            "def archive_traces",
        ),
        (
            "archival_otlp",
            "archival",
            "T21.3",
            "mlflow-store",
            "B",
            "mlflow/tracing/otel/otel_archival.py",
            "def spans_to_traces_data_pb",
        ),
        (
            "archival_scheduler",
            "archival",
            "T21.4",
            "worker",
            "B",
            "mlflow/tracing/trace_archival_service.py",
            "def run_trace_archival_scheduler",
        ),
    ]
    for name, area, task, owner, tier, file, pattern in components:
        entries.append(
            _surface_entry(
                name=name,
                area=area,
                task=task,
                owner=owner,
                tier=tier,
                evidence=_ref(file, pattern),
                tests=tests,
                kind="runtime_component",
            )
        )
    return entries


def _parameter_schema(cls: type) -> list[dict[str, Any]]:
    parameters = []
    for name, parameter in inspect.signature(cls).parameters.items():
        default = parameter.default
        if isinstance(default, (set, frozenset)):
            if default:
                default_repr = "{" + ", ".join(repr(value) for value in sorted(default)) + "}"
            else:
                default_repr = "set()" if isinstance(default, set) else "frozenset()"
        else:
            default_repr = repr(default)
        parameters.append({
            "name": name,
            "kind": parameter.kind.name,
            "type": str(parameter.annotation),
            "required": default is inspect.Parameter.empty,
            "default": None if default is inspect.Parameter.empty else default_repr,
        })
    return parameters


UPSTREAM_METRICS = {
    "deepeval": [
        "AnswerRelevancy",
        "ArgumentCorrectness",
        "Bias",
        "ContextualPrecision",
        "ContextualRecall",
        "ContextualRelevancy",
        "ConversationCompleteness",
        "ConversationalDAG",
        "DAG",
        "ExactMatch",
        "Faithfulness",
        "GoalAccuracy",
        "Hallucination",
        "ImageCoherence",
        "ImageEditing",
        "ImageHelpfulness",
        "ImageReference",
        "JsonCorrectness",
        "KnowledgeRetention",
        "MCPTaskCompletion",
        "MCPUse",
        "Misuse",
        "MultiTurnMCPUse",
        "NonAdvice",
        "PIILeakage",
        "PatternMatch",
        "PlanAdherence",
        "PlanQuality",
        "PromptAlignment",
        "RoleAdherence",
        "RoleViolation",
        "StepEfficiency",
        "Summarization",
        "TaskCompletion",
        "TextToImage",
        "ToolCorrectness",
        "ToolUse",
        "TopicAdherence",
        "Toxicity",
        "TurnContextualPrecision",
        "TurnContextualRecall",
        "TurnContextualRelevancy",
        "TurnFaithfulness",
        "TurnRelevancy",
    ],
    "ragas": [
        "AgentGoalAccuracy",
        "AgentGoalAccuracyWithReference",
        "AgentGoalAccuracyWithoutReference",
        "AnswerAccuracy",
        "AnswerCorrectness",
        "AnswerRelevancy",
        "BleuScore",
        "CHRFScore",
        "ContextEntityRecall",
        "ContextPrecision",
        "ContextPrecisionWithReference",
        "ContextPrecisionWithoutReference",
        "ContextRecall",
        "ContextRelevance",
        "ContextUtilization",
        "DataCompyScore",
        "DomainSpecificRubrics",
        "ExactMatch",
        "FactualCorrectness",
        "Faithfulness",
        "InstanceSpecificRubrics",
        "MultiModalFaithfulness",
        "MultiModalRelevance",
        "NoiseSensitivity",
        "NonLLMStringSimilarity",
        "QuotedSpansAlignment",
        "ResponseGroundedness",
        "RougeScore",
        "RubricsScoreWithReference",
        "RubricsScoreWithoutReference",
        "SQLSemanticEquivalence",
        "SemanticSimilarity",
        "StringPresence",
        "SummaryScore",
        "ToolCallAccuracy",
        "ToolCallF1",
        "TopicAdherence",
    ],
    "trulens": [
        "Coherence",
        "Comprehensiveness",
        "Conciseness",
        "ContextRelevance",
        "Controversiality",
        "Correctness",
        "Criminality",
        "ExecutionEfficiency",
        "Groundedness",
        "Harmfulness",
        "Helpfulness",
        "Insensitivity",
        "LogicalConsistency",
        "Maliciousness",
        "Misogyny",
        "PlanAdherence",
        "PlanQuality",
        "QsRelevance",
        "Relevance",
        "Sentiment",
        "Stereotypes",
        "Summarization",
        "ToolCalling",
        "ToolQuality",
        "ToolSelection",
    ],
    "phoenix": ["Hallucination", "QA", "Relevance", "SQL", "Summarization", "Toxicity"],
}

PACKAGE_PINS = {
    "deepeval": "4.0.7",
    "ragas": "0.4.3",
    "trulens": "2.8.1",
    "trulens-providers-litellm": "2.8.1",
    "arize-phoenix-evals": "2.13.0",
    "litellm": "1.91.2",
    "gepa": "0.0.27",
    "dspy": "3.2.1",
}

LICENSE_AUDIT = [
    {
        "source": "MLflow native scorer/judge/MetaPrompt algorithms and prompt templates",
        "pin": "MLflow reference git SHA",
        "license": "Apache-2.0",
        "provenance": "LICENSE.txt and mlflow/genai/",
        "compatible": True,
    },
    {
        "source": "LiteLLM provider transforms, retry/tokenizer logic, and price table",
        "pin": "litellm==1.91.2",
        "license": "MIT",
        "provenance": "https://github.com/BerriAI/litellm/tree/v1.91.2",
        "compatible": True,
    },
    {
        "source": "GEPA optimization algorithm and prompts",
        "pin": "gepa==0.0.27",
        "license": "MIT",
        "provenance": "https://github.com/gepa-ai/gepa/tree/v0.0.27",
        "compatible": True,
    },
    {
        "source": "DSPy runtime used by MemoryAugmentedJudge",
        "pin": "dspy==3.2.1",
        "license": "MIT",
        "provenance": "https://github.com/stanfordnlp/dspy/tree/3.2.1",
        "compatible": True,
    },
    {
        "source": "DeepEval metric algorithms and prompts",
        "pin": "deepeval==4.0.7",
        "license": "Apache-2.0",
        "provenance": "https://github.com/confident-ai/deepeval/tree/v4.0.7",
        "compatible": True,
    },
    {
        "source": "Ragas metric algorithms and prompts",
        "pin": "ragas==0.4.3",
        "license": "Apache-2.0",
        "provenance": "https://github.com/explodinggradients/ragas/tree/v0.4.3",
        "compatible": True,
    },
    {
        "source": "TruLens feedback algorithms and prompts",
        "pin": "trulens==2.8.1; trulens-providers-litellm==2.8.1",
        "license": "MIT",
        "provenance": "https://github.com/truera/trulens/tree/trulens-2.8.1",
        "compatible": True,
    },
    {
        "source": "Phoenix evaluator algorithms and prompt templates",
        "pin": "arize-phoenix-evals==2.13.0",
        "license": "Elastic-2.0",
        "provenance": "https://github.com/Arize-ai/phoenix/tree/arize-phoenix-evals-v2.13.0",
        "compatible": False,
    },
]

CORPUS_RECORDERS = {
    "semantic": {
        "entry_points": [
            "mlflow.genai.scorers.base.Scorer.model_validate",
            "mlflow.genai.scorers.base.Scorer.__call__",
            "mlflow.genai.evaluation.harness.run",
        ],
        "fixture_root": "rust/compliance/fixtures/genai/semantic/",
        "normalization": "T12.4 recursive bindings plus volatile id/time/token/path replacement",
    },
    "sse": {
        "entry_points": [
            "mlflow.server.gateway_api gateway StreamingResponse routes",
            "mlflow.server.assistant.api.stream_response",
        ],
        "fixture_root": "rust/compliance/fixtures/genai/sse/",
        "normalization": (
            "preserve SSE event/data framing and order; normalize parsed JSON payloads"
        ),
    },
}


def _scorer_manifest(git_sha: str) -> dict[str, Any]:
    from mlflow.genai.scorers.builtin_scorers import _get_all_concrete_builtin_scorers

    deterministic = {"PIIDetection", "RegexMatch", "ResponseLength"}
    hybrid = {"ToolCallCorrectness"}
    builtins = []
    for cls in sorted(_get_all_concrete_builtin_scorers(), key=lambda item: item.__name__):
        mode = (
            "deterministic"
            if cls.__name__ in deterministic
            else "hybrid"
            if cls.__name__ in hybrid
            else "llm"
        )
        builtins.append({
            "name": cls.__name__,
            "serialized_name_default": inspect.signature(cls).parameters["name"].default,
            "pin": f"mlflow==3.14.1.dev0+git.{git_sha}",
            "params_schema": _parameter_schema(cls),
            "execution": mode,
            "source": f"mlflow/genai/scorers/builtin_scorers.py:{inspect.getsourcelines(cls)[1]}",
            "owner_task": "T19.1",
        })
    judges = [
        {
            "name": "InstructionsJudge",
            "pin": f"mlflow==3.14.1.dev0+git.{git_sha}",
            "params_schema": _parameter_schema(
                __import__(
                    "mlflow.genai.judges.instructions_judge", fromlist=["InstructionsJudge"]
                ).InstructionsJudge
            ),
            "execution": "llm",
            "source": _ref(
                "mlflow/genai/judges/instructions_judge/__init__.py", "class InstructionsJudge"
            ),
            "owner_task": "T19.1",
        },
        {
            "name": "MemoryAugmentedJudge",
            "pin": f"mlflow==3.14.1.dev0+git.{git_sha}; dspy=={PACKAGE_PINS['dspy']}",
            "params_schema": _parameter_schema(
                __import__(
                    "mlflow.genai.judges.optimizers.memalign.optimizer",
                    fromlist=["MemoryAugmentedJudge"],
                ).MemoryAugmentedJudge
            ),
            "execution": "llm+embeddings",
            "source": _ref(
                "mlflow/genai/judges/optimizers/memalign/optimizer.py",
                "class MemoryAugmentedJudge",
            ),
            "owner_task": "T19.1",
        },
    ]
    family_package = {
        "deepeval": "deepeval",
        "ragas": "ragas",
        "trulens": "trulens",
        "phoenix": "arize-phoenix-evals",
    }
    third_party = []
    for family, metrics in UPSTREAM_METRICS.items():
        package = family_package[family]
        for metric in metrics:
            execution = (
                "deterministic"
                if (
                    (family == "deepeval" and metric in {"ExactMatch", "PatternMatch"})
                    or (
                        family == "ragas"
                        and metric
                        in {
                            "BleuScore",
                            "CHRFScore",
                            "DataCompyScore",
                            "ExactMatch",
                            "NonLLMStringSimilarity",
                            "QuotedSpansAlignment",
                            "RougeScore",
                            "StringPresence",
                            "ToolCallAccuracy",
                            "ToolCallF1",
                        }
                    )
                )
                else "llm_or_embeddings"
            )
            third_party.append({
                "family": family,
                "metric": metric,
                "package": package,
                "pin": PACKAGE_PINS[package],
                "execution": execution,
                "owner_task": "T19.3",
                "fixture_oracle": "corpus recorder needed",
            })
    return {
        "schema_version": 1,
        "reference": {"mlflow_git_sha": git_sha, "package_pins": PACKAGE_PINS},
        "builtin_scorers": builtins,
        "serialized_judges": judges,
        "third_party_metrics": third_party,
        "rejected_payloads": [
            {
                "kind": "decorator_scorer",
                "condition": "call_source is not null on OSS",
                "source": _ref("mlflow/genai/scorers/base.py", "def _reconstruct_decorator_scorer"),
                "owner_task": "T19.1",
            }
        ],
    }


def _class_ref(value: Any) -> str | None:
    if value is None:
        return None
    cls = value if inspect.isclass(value) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _provider_manifest(git_sha: str) -> dict[str, Any]:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    import importlib.metadata

    import litellm
    from litellm.utils import ProviderConfigManager

    from mlflow.gateway.config import Provider

    version = importlib.metadata.version("litellm")
    if version != PACKAGE_PINS["litellm"]:
        raise RuntimeError(f"expected litellm {PACKAGE_PINS['litellm']}, found {version}")
    package_root = Path(inspect.getfile(litellm)).parent
    price_path = package_root / "model_prices_and_context_window_backup.json"
    raw_prices = json.loads(price_path.read_text())
    raw_prices.pop("sample_spec", None)

    price_fields = re.compile(r"(^|_)cost(_|$)|price")
    limit_fields = {"max_tokens", "max_input_tokens", "max_output_tokens"}
    models = []
    models_by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_key, info in sorted(raw_prices.items()):
        provider = str(info.get("litellm_provider") or "unknown")
        prices = {key: value for key, value in info.items() if price_fields.search(key)}
        limits = {key: info[key] for key in limit_fields if key in info}
        tokenizer = {
            key: value
            for key, value in info.items()
            if "token" in key and key not in limit_fields and "cost" not in key
        }
        entry = {
            "model_key": model_key,
            "provider": provider,
            "mode": info.get("mode"),
            "limits": limits,
            "prices": prices,
            "tokenizer_metadata": tokenizer,
        }
        models.append(entry)
        models_by_provider[provider].append(entry)

    provider_enums = {provider.value: provider for provider in litellm.provider_list}
    price_providers = set(models_by_provider)
    native_providers = Provider.values()
    all_providers = set(provider_enums) | price_providers | native_providers
    providers = []
    for provider in all_providers:
        provider_enum = provider_enums.get(provider)
        provider_models = models_by_provider.get(provider, [])
        representative = next(
            (m["model_key"].split("/", 1)[-1] for m in provider_models if m["mode"] == "chat"),
            "t15-reference-model",
        )
        if provider_enum is None:
            chat_config = embedding_config = None
        else:
            try:
                chat_config = ProviderConfigManager.get_provider_chat_config(
                    representative, provider_enum
                )
            except Exception:
                chat_config = None
            try:
                embedding_config = ProviderConfigManager.get_provider_embedding_config(
                    representative, provider_enum
                )
            except Exception:
                embedding_config = None
        providers.append({
            "name": provider,
            "mlflow_native_adapter": provider in native_providers,
            "litellm_fallback_reachable": provider in provider_enums
            or provider in price_providers
            or provider == "litellm",
            "source_membership": {
                "mlflow_gateway_registry": provider in native_providers,
                "litellm_provider_list": provider in provider_enums,
                "litellm_price_snapshot": provider in price_providers,
            },
            "chat_transform": _class_ref(chat_config),
            "chat_transform_present": chat_config is not None,
            "embedding_transform": _class_ref(embedding_config),
            "embedding_transform_present": embedding_config is not None,
            "limits_present": any(model["limits"] for model in provider_models),
            "price_present": any(model["prices"] for model in provider_models),
            "tokenizer_metadata_present": any(
                model["tokenizer_metadata"] for model in provider_models
            ),
            "model_entry_count": len(provider_models),
            "owner_task": "T18.4",
        })

    llm_files = sorted((package_root / "llms").rglob("*.py"))
    llm_digest = hashlib.sha256()
    for path in llm_files:
        llm_digest.update(path.relative_to(package_root).as_posix().encode())
        llm_digest.update(path.read_bytes())
    assets = []
    for relative in (
        "model_prices_and_context_window_backup.json",
        "utils.py",
        "exceptions.py",
        "litellm_core_utils/get_llm_provider_logic.py",
        "router_utils/get_retry_from_policy.py",
    ):
        path = package_root / relative
        assets.append({"path": relative, "sha256": _sha256(path)})
    assets.append({
        "path": "llms/**/*.py",
        "sha256": llm_digest.hexdigest(),
        "file_count": len(llm_files),
    })
    return {
        "schema_version": 1,
        "reference": {
            "mlflow_git_sha": git_sha,
            "litellm_pin": f"litellm=={version}",
            "price_source_policy": "wheel backup snapshot; never fetch LiteLLM main at runtime",
            "price_snapshot_sha256": _sha256(price_path),
            "source_url": f"https://github.com/BerriAI/litellm/tree/v{version}",
        },
        "providers": sorted(providers, key=lambda item: item["name"]),
        "models": models,
        "retry_classification": {
            "source_asset": "router_utils/get_retry_from_policy.py",
            "classes": [
                "AuthenticationError",
                "Timeout",
                "RateLimitError",
                "ContentPolicyViolationError",
                "BadRequestError",
            ],
        },
        "tokenizer_mapping": {
            "source_asset": "utils.py",
            "policy": "port _select_tokenizer/create_tokenizer behavior and model metadata exactly",
        },
        "source_assets": assets,
        "unpriced_or_transform_only_providers": sorted(
            provider["name"] for provider in providers if provider["model_entry_count"] == 0
        ),
    }


def _algorithms_manifest(git_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reference": {"mlflow_git_sha": git_sha, "package_pins": PACKAGE_PINS},
        "algorithms": [
            {
                "name": "GEPA",
                "pin": "gepa==0.0.27",
                "entry_points": [
                    "gepa.optimize",
                    "gepa.GEPAAdapter",
                    "mlflow.genai.optimize.optimizers.GepaPromptOptimizer.optimize",
                ],
                "accepted_server_config": [
                    "reflection_model",
                    "max_metric_calls",
                    "display_progress_bar",
                    "gepa_kwargs",
                ],
                "upstream_kwargs_entry_point": "gepa.optimize",
                "source": _ref(
                    "mlflow/genai/optimize/optimizers/gepa_optimizer.py",
                    "class GepaPromptOptimizer",
                ),
                "owner_task": "T19.5",
            },
            {
                "name": "MetaPrompt",
                "pin": f"mlflow==3.14.1.dev0+git.{git_sha}",
                "entry_points": ["mlflow.genai.optimize.optimizers.MetaPromptOptimizer.optimize"],
                "accepted_server_config": ["reflection_model", "lm_kwargs", "guidelines"],
                "source": _ref(
                    "mlflow/genai/optimize/optimizers/metaprompt_optimizer.py",
                    "class MetaPromptOptimizer",
                ),
                "owner_task": "T19.5",
            },
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _extract_fixture_oracles(items: list[dict[str, Any]]) -> dict[str, Any]:
    oracles: dict[str, Any] = {}
    for item in items:
        fixture = item["fixture_oracle"]
        if item["scope"] == "mlflow/genai":
            source_path = item["source"].rsplit(":", 1)[0]
            oracle_id = f"module:{source_path}"
        else:
            area = item["id"].split(":", 3)[1]
            oracle_id = f"surface:{area}"
        existing = oracles.setdefault(oracle_id, fixture)
        if existing != fixture:
            raise ValueError(f"conflicting fixture oracle definition for {oracle_id}")
        item["fixture_oracle"] = {
            "strategy": fixture["strategy"],
            "oracle_group": oracle_id,
            **({"recorder": fixture["recorder"]} if "recorder" in fixture else {}),
        }
    return oracles


def main() -> None:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    git_sha = REFERENCE_MLFLOW_GIT_SHA
    tests = _test_inventory()
    scorers = _scorer_manifest(git_sha)
    providers = _provider_manifest(git_sha)
    algorithms = _algorithms_manifest(git_sha)
    _write_json(HERE / "scorers.json", scorers)
    _write_json(HERE / "providers.json", providers)
    _write_json(HERE / "algorithms.json", algorithms)

    items = _genai_items(tests) + _proto_surfaces(tests) + _hand_surfaces(tests)
    fixture_oracles = _extract_fixture_oracles(items)
    manifest_counts = {
        "scorers": len(scorers["builtin_scorers"])
        + len(scorers["serialized_judges"])
        + len(scorers["third_party_metrics"]),
        "builtin_scorers": len(scorers["builtin_scorers"]),
        "serialized_judges": len(scorers["serialized_judges"]),
        "third_party_metrics": len(scorers["third_party_metrics"]),
        "providers": len(providers["providers"]),
        "provider_models": len(providers["models"]),
        "algorithms": len(algorithms["algorithms"]),
    }
    ledger = {
        "schema_version": 1,
        "reference": {
            "mlflow_version": "3.14.1.dev0",
            "git_sha": git_sha,
            "generated_by": "rust/genai-inventory/build_inventory.py",
            "classification_policy": (
                "Server reachability starts at OSS HTTP handlers, job dispatch, periodic jobs, "
                "gateway/assistant runtime, or archival. Ambiguity is classified server_reachable."
            ),
        },
        "items": items,
        "tests": tests,
        "fixture_oracles": fixture_oracles,
        "manifest_counts": manifest_counts,
        "manifest_sha256": {
            "scorers.json": _sha256(HERE / "scorers.json"),
            "providers.json": _sha256(HERE / "providers.json"),
            "algorithms.json": _sha256(HERE / "algorithms.json"),
        },
        "license_audit": LICENSE_AUDIT,
        "corpus_recorders": CORPUS_RECORDERS,
        "license_blockers": [
            {
                "name": "Phoenix evaluator algorithms and prompt templates",
                "license": "Elastic-2.0",
                "reason": "not compatible for source vendoring into Apache-2.0 MLflow",
                "resolution": (
                    "obtain permission/relicense or perform a counsel-approved clean-room "
                    "implementation"
                ),
            }
        ],
        "summary": {
            "classification_counts": {
                classification: sum(item["classification"] == classification for item in items)
                for classification in sorted(CLASSIFICATIONS)
            },
            "tier_counts": dict(Counter(item["tier"] for item in items if item["tier"])),
            "task_counts": dict(Counter(item["task"] for item in items if item["task"])),
            "phase_counts": {
                f"T{phase}": sum(
                    item["task"] is not None and item["task"].split(".", 1)[0] == f"T{phase}"
                    for item in items
                )
                for phase in range(16, 23)
            },
            "test_classification_counts": {
                classification: sum(test["classification"] == classification for test in tests)
                for classification in sorted(TEST_CLASSIFICATIONS)
            },
        },
    }
    _write_json(HERE / "ledger.json", ledger)


if __name__ == "__main__":
    main()
