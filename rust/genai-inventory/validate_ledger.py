"""Validate the committed T15.5 ledger and compatibility manifests."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENAI_ROOT = ROOT / "mlflow" / "genai"
CLASSIFICATIONS = {"server_reachable", "client_only", "dead"}
OWNERS = {"mlflow-genai", "mlflow-server", "mlflow-store", "worker"}
REQUIRED_SURFACE_AREAS = {
    "archival",
    "assistant",
    "datasets",
    "gateway_crud",
    "gateway_runtime",
    "issues",
    "jobs",
    "label_schemas",
    "prompt_optimization",
    "promptlab",
    "review_queues",
    "scorers",
}


def _load(name: str) -> dict[str, Any]:
    value = json.loads((HERE / name).read_text())
    if value.get("schema_version") != 1:
        raise AssertionError(f"{name}: unsupported schema version")
    return value


def _definitions(path: Path) -> list[tuple[str, str | None, int]]:
    result: list[tuple[str, str | None, int]] = [("module", None, 1)]
    tree = ast.parse(path.read_text())

    def walk(body: list[ast.stmt], parents: list[str], parent_kind: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                result.append(("class", ".".join([*parents, node.name]), node.lineno))
                walk(node.body, [*parents, node.name], "class")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if parent_kind == "class":
                    kind = "method"
                elif parents:
                    kind = "nested_function"
                else:
                    kind = "function"
                result.append((kind, ".".join([*parents, node.name]), node.lineno))
                walk(node.body, [*parents, node.name], "function")

    walk(tree.body, [], "module")
    return result


def _expected_genai_ids() -> set[str]:
    expected = set()
    for path in sorted(GENAI_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        expected.update(
            f"genai:{relative}:{line}:{qualname or '<module>'}"
            for _kind, qualname, line in _definitions(path)
        )
    return expected


def _validate_reference(reference: str) -> None:
    path, raw_line = reference.rsplit(":", 1)
    source = ROOT / path
    if not source.is_file():
        raise AssertionError(f"missing evidence file: {path}")
    line = int(raw_line)
    line_count = max(1, len(source.read_text().splitlines()))
    if not 1 <= line <= line_count:
        raise AssertionError(f"invalid evidence line: {reference}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_provider_names() -> set[str]:
    path = ROOT / "mlflow" / "gateway" / "config.py"
    tree = ast.parse(path.read_text())
    provider = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Provider"
    )
    names = set()
    for node in provider.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                names.add(node.value.value)
    return names


def main() -> None:
    ledger = _load("ledger.json")
    scorers = _load("scorers.json")
    providers = _load("providers.json")
    algorithms = _load("algorithms.json")
    items = ledger["items"]

    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate ledger item id")
    actual_genai_ids = {item["id"] for item in items if item["scope"] == "mlflow/genai"}
    if actual_genai_ids != _expected_genai_ids():
        missing = sorted(_expected_genai_ids() - actual_genai_ids)[:5]
        extra = sorted(actual_genai_ids - _expected_genai_ids())[:5]
        raise AssertionError(f"mlflow/genai coverage mismatch; missing={missing}, extra={extra}")

    oracle_groups = ledger["fixture_oracles"]
    test_ids = {test["id"] for test in ledger["tests"]}
    for item in items:
        classification = item.get("classification")
        if classification not in CLASSIFICATIONS:
            raise AssertionError(f"unclassified item: {item['id']}")
        _validate_reference(item["source"])
        _validate_reference(item["reachability_evidence"])
        fixture = item.get("fixture_oracle")
        if not fixture or fixture.get("oracle_group") not in oracle_groups:
            raise AssertionError(f"missing fixture/oracle: {item['id']}")
        if fixture.get("strategy") not in {"existing_python_tests", "corpus recorder needed"}:
            raise AssertionError(f"invalid fixture strategy: {item['id']}")
        if classification == "server_reachable":
            if not item.get("task") or not item["task"].startswith("T"):
                raise AssertionError(f"server item missing task: {item['id']}")
            if item.get("native_owner_crate") not in OWNERS:
                raise AssertionError(f"server item missing owner: {item['id']}")
            if item.get("tier") not in {"A", "B", "C"}:
                raise AssertionError(f"server item missing tier: {item['id']}")
        elif item.get("task") is not None or item.get("native_owner_crate") is not None:
            raise AssertionError(f"non-server item has native ownership: {item['id']}")
        if item["symbol_kind"] == "http_route":
            if not item.get("http_method") or not item.get("paths"):
                raise AssertionError(f"HTTP surface missing method/path: {item['id']}")

    for oracle_id, oracle in oracle_groups.items():
        strategy = oracle["strategy"]
        if strategy == "existing_python_tests":
            unknown = set(oracle["tests"]) - test_ids
            if unknown or not oracle["tests"]:
                raise AssertionError(f"invalid test oracle {oracle_id}: {sorted(unknown)[:3]}")
        elif strategy != "corpus recorder needed" or not oracle.get("recorder"):
            raise AssertionError(f"invalid corpus oracle: {oracle_id}")

    surfaces = [item for item in items if item["scope"] == "part_ii_surface"]
    areas = {item["id"].split(":", 3)[1] for item in surfaces}
    if areas != REQUIRED_SURFACE_AREAS:
        raise AssertionError(f"Part II surface area mismatch: {sorted(areas)}")
    if sum(item["task"] == "T18.1" for item in surfaces) != 36:
        raise AssertionError("gateway CRUD surface must contain all 36 proto routes")

    expected_counts = Counter(item["classification"] for item in items)
    expected_counts["dead"] += 0
    if ledger["summary"]["classification_counts"] != dict(sorted(expected_counts.items())):
        raise AssertionError("classification summary drift")
    expected_phases = {
        f"T{phase}": sum(
            item["task"] is not None and item["task"].split(".", 1)[0] == f"T{phase}"
            for item in items
        )
        for phase in range(16, 23)
    }
    if ledger["summary"]["phase_counts"] != expected_phases:
        raise AssertionError("phase summary drift")

    manifests = {
        "scorers.json": scorers,
        "providers.json": providers,
        "algorithms.json": algorithms,
    }
    for name in manifests:
        if ledger["manifest_sha256"][name] != _sha256(HERE / name):
            raise AssertionError(f"manifest checksum drift: {name}")

    scorer_rows = (
        scorers["builtin_scorers"] + scorers["serialized_judges"] + scorers["third_party_metrics"]
    )
    for scorer in scorer_rows:
        if (
            not {"name", "pin", "execution", "owner_task"} <= scorer.keys()
            and not {
                "metric",
                "pin",
                "execution",
                "owner_task",
            }
            <= scorer.keys()
        ):
            raise AssertionError(f"incomplete scorer row: {scorer}")
    for scorer in scorers["builtin_scorers"] + scorers["serialized_judges"]:
        if "params_schema" not in scorer:
            raise AssertionError(f"missing parameter schema: {scorer['name']}")

    provider_names = {provider["name"] for provider in providers["providers"]}
    model_providers = {model["provider"] for model in providers["models"]}
    if not _native_provider_names() <= provider_names or not model_providers <= provider_names:
        raise AssertionError("provider manifest does not include all native/price-map providers")
    provider_flags = {
        "chat_transform_present",
        "embedding_transform_present",
        "limits_present",
        "price_present",
        "tokenizer_metadata_present",
        "litellm_fallback_reachable",
        "mlflow_native_adapter",
    }
    for provider in providers["providers"]:
        if not provider_flags <= provider.keys():
            raise AssertionError(f"provider flags missing: {provider['name']}")
    for algorithm in algorithms["algorithms"]:
        if not algorithm.get("pin") or not algorithm.get("entry_points"):
            raise AssertionError(f"incomplete algorithm row: {algorithm.get('name')}")

    expected_manifest_counts = {
        "algorithms": len(algorithms["algorithms"]),
        "builtin_scorers": len(scorers["builtin_scorers"]),
        "provider_models": len(providers["models"]),
        "providers": len(providers["providers"]),
        "scorers": len(scorer_rows),
        "serialized_judges": len(scorers["serialized_judges"]),
        "third_party_metrics": len(scorers["third_party_metrics"]),
    }
    if ledger["manifest_counts"] != expected_manifest_counts:
        raise AssertionError("manifest count drift")


if __name__ == "__main__":
    main()
