#!/usr/bin/env python3
"""Generate the T18.4 native-provider verification manifest.

The source is the pinned LiteLLM 1.91.2 inventory produced by
``genai-inventory/build_inventory.py``.  This generator intentionally performs
no network access and never imports LiteLLM; upgrades first regenerate the
pinned inventory and then this derived runtime manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parents[1]
SOURCE = RUST_ROOT / "genai-inventory" / "providers.json"
DESTINATION = RUST_ROOT / "genai-inventory" / "provider_manifest.json"

EXPLICIT_ADAPTERS = {"openai", "azure", "anthropic", "gemini", "bedrock", "databricks"}
# Aliases the Rust runtime folds onto explicit adapters (normalize_provider in
# gateway_provider_matrix.rs); classify them the same so the manifest and
# adapter_for() never disagree.
EXPLICIT_ALIASES = {
    "amazon-bedrock": "bedrock",
    "databricks-model-serving": "databricks",
    "azure-openai": "azure",
}
DIFFERENTIAL_ADAPTERS = {"openai", "azure", "anthropic", "gemini"}
OPENAI_COMPATIBLE = {"groq", "deepseek", "xai", "openrouter", "ollama", "portkey"}
CHAT_NATIVE = (
    EXPLICIT_ADAPTERS
    | OPENAI_COMPATIBLE
    | {
        "ai21labs",
        "amazon-bedrock",
        "cohere",
        "databricks-model-serving",
        "huggingface-text-generation-inference",
        "litellm",
        "mistral",
        "mlflow-model-serving",
        "mosaicml",
        "palm",
        "togetherai",
        "vertex_ai",
    }
)
EMBEDDING_NATIVE = {
    "azure",
    "bedrock",
    "cohere",
    "databricks",
    "databricks-model-serving",
    "gemini",
    "groq",
    "huggingface-text-generation-inference",
    "litellm",
    "mistral",
    "mlflow-model-serving",
    "mosaicml",
    "ollama",
    "openai",
    "openrouter",
    "portkey",
    "togetherai",
    "vertex_ai",
}


def main() -> None:
    source = json.loads(SOURCE.read_text())
    modes: dict[str, set[str]] = {}
    for model in source["models"]:
        modes.setdefault(model["provider"], set()).add(model.get("mode"))

    providers = []
    for item in source["providers"]:
        name = item["name"]
        provider_modes = modes.get(name, set())
        chat = (
            item["chat_transform_present"]
            or "chat" in provider_modes
            or "completion" in provider_modes
            or "responses" in provider_modes
            or name in CHAT_NATIVE
        )
        embeddings = (
            item["embedding_transform_present"]
            or "embedding" in provider_modes
            or name in EMBEDDING_NATIVE
        )
        normalized = EXPLICIT_ALIASES.get(name, name)
        if normalized in EXPLICIT_ADAPTERS:
            adapter = "explicit_native"
            verification = (
                "differential_fixture" if normalized in DIFFERENTIAL_ADAPTERS else "hermetic_matrix"
            )
        elif name in OPENAI_COMPATIBLE:
            adapter = "openai_compatible"
            verification = "hermetic_matrix"
        else:
            adapter = "pinned_litellm_transform"
            verification = "hermetic_matrix"
        providers.append({
            "name": name,
            "adapter": adapter,
            "capabilities": {
                "chat": chat,
                "embeddings": embeddings,
                "stream": chat,
                "cost": item["price_present"],
            },
            "verification": verification,
            "unsupported": False,
            "chat_transform": item["chat_transform"],
            "embedding_transform": item["embedding_transform"],
            "model_entry_count": item["model_entry_count"],
        })

    manifest = {
        "schema_version": 1,
        "reference": source["reference"],
        "source_manifest": "providers.json",
        "coverage": {
            "providers": len(providers),
            "models": len(source["models"]),
            "unsupported": 0,
            "explicit_native": sum(p["adapter"] == "explicit_native" for p in providers),
            "openai_compatible": sum(p["adapter"] == "openai_compatible" for p in providers),
            "pinned_litellm_transform": sum(
                p["adapter"] == "pinned_litellm_transform" for p in providers
            ),
        },
        "retry_classification": source["retry_classification"],
        "tokenizer_mapping": source["tokenizer_mapping"],
        "providers": providers,
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        if not DESTINATION.exists() or DESTINATION.read_text() != rendered:
            raise SystemExit(
                "provider_manifest.json is stale; run rust/tools/build_provider_manifest.py"
            )
    else:
        DESTINATION.write_text(rendered)


if __name__ == "__main__":
    main()
