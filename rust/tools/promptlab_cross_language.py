#!/usr/bin/env python3
"""Load/predict differential for Rust- and Python-created promptlab models."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import requests
import yaml

import mlflow.pyfunc
from mlflow.deployments import set_deployments_target
from mlflow.entities import Param
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.promptlab_utils import _create_promptlab_run_impl

PROMPT_TEMPLATE = "Write about {{ thing }}."
PROMPT_PARAMETERS = [Param("thing", "books")]
MODEL_PARAMETERS = [Param("temperature", "0.1"), Param("max_tokens", "10")]
MODEL_OUTPUT_PARAMETERS = [Param("latency", "100")]
MODEL_ROUTE = "openai-endpoint"
PINNED_PYTHON_VERSION = "3.10.18"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rust_model", type=Path)
    parser.add_argument("rust_gateway")
    return parser.parse_args()


def python_twin(root: Path):
    store = _get_store(
        f"sqlite:///{root / 'tracking.db'}",
        artifact_uri=(root / "artifacts").as_uri(),
    )
    experiment_id = store.create_experiment("promptlab-python-twin")
    with (
        mock.patch("mlflow.pyfunc.PYTHON_VERSION", PINNED_PYTHON_VERSION),
        mock.patch("mlflow.utils.environment.PYTHON_VERSION", PINNED_PYTHON_VERSION),
    ):
        run = _create_promptlab_run_impl(
            store,
            experiment_id=experiment_id,
            run_name="promptlab-cross-language",
            tags=[],
            prompt_template=PROMPT_TEMPLATE,
            prompt_parameters=PROMPT_PARAMETERS,
            model_route=MODEL_ROUTE,
            model_parameters=MODEL_PARAMETERS,
            model_input="Write about books.",
            model_output_parameters=MODEL_OUTPUT_PARAMETERS,
            model_output="gateway:Write about books.",
            mlflow_version="ignored-by-python-writer",
            user_id="cross-language",
            start_time=123456,
        )
    return Path(run.info.artifact_uri.removeprefix("file://")) / "model", run


def compare_run_data(rust_gateway: str, rust_model: Path, python_run):
    rust_mlmodel = yaml.safe_load((rust_model / "MLmodel").read_bytes())
    response = requests.get(
        f"{rust_gateway}/api/2.0/mlflow/runs/get",
        params={"run_id": rust_mlmodel["run_id"]},
        timeout=30,
    )
    response.raise_for_status()
    rust_run = response.json()["run"]

    rust_params = {item["key"]: item["value"] for item in rust_run["data"]["params"]}
    python_params = dict(python_run.data.params)
    assert rust_params == python_params
    assert rust_run["data"].get("metrics", []) == list(python_run.data.metrics.values())

    rust_tags = {item["key"]: item["value"] for item in rust_run["data"]["tags"]}
    python_tags = dict(python_run.data.tags)
    for key in ("mlflow.loggedArtifacts", "mlflow.runSourceType"):
        assert rust_tags[key] == python_tags[key]

    def normalize_history(value):
        history = json.loads(value)
        for model in history:
            model["run_id"] = "<run-id>"
            model["model_uuid"] = "<model-uuid>"
            model["utc_time_created"] = "<utc-time-created>"
        return history

    assert normalize_history(rust_tags["mlflow.log-model.history"]) == normalize_history(
        python_tags["mlflow.log-model.history"]
    )
    assert rust_run["info"]["status"] == python_run.info.status
    assert rust_run["info"]["start_time"] == python_run.info.start_time


def compare_layout(rust_model: Path, python_model: Path):
    rust_files = sorted(path.name for path in rust_model.iterdir())
    python_files = sorted(path.name for path in python_model.iterdir())
    assert rust_files == python_files, (rust_files, python_files)

    byte_equal = []
    for name in rust_files:
        rust_bytes = (rust_model / name).read_bytes()
        python_bytes = (python_model / name).read_bytes()
        if name == "MLmodel":
            rust_text = rust_bytes.decode()
            python_text = python_bytes.decode()

            def normalize(text):
                return re.sub(
                    r"^(model_uuid|run_id|utc_time_created):.*$",
                    lambda match: f"{match.group(1)}: <normalized>",
                    text,
                    flags=re.MULTILINE,
                )

            assert normalize(rust_text) == normalize(python_text)
            rust_yaml = yaml.safe_load(rust_bytes)
            python_yaml = yaml.safe_load(python_bytes)
            for value in (rust_yaml, python_yaml):
                value["run_id"] = "<run-id>"
                value["model_uuid"] = "<model-uuid>"
                value["utc_time_created"] = "<utc-time-created>"
            assert rust_yaml == python_yaml, (rust_yaml, python_yaml)
        else:
            assert rust_bytes == python_bytes, name
            byte_equal.append(name)

    rust_eval = rust_model.parent / "eval_results_table.json"
    python_eval = python_model.parent / "eval_results_table.json"
    assert rust_eval.read_bytes() == python_eval.read_bytes()
    byte_equal.append("../eval_results_table.json")
    return rust_files, byte_equal


def predict_through_rust_gateway(model, gateway: str, frame: pd.DataFrame):
    class Endpoint:
        endpoint_type = "llm/v1/chat"

    def get_endpoint(_self, endpoint):
        assert endpoint == MODEL_ROUTE
        return Endpoint()

    def predict(_self, deployment_name=None, inputs=None, endpoint=None):
        assert deployment_name is None
        response = requests.post(
            f"{gateway}/gateway/{endpoint}/mlflow/invocations",
            json=inputs,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    with (
        mock.patch("mlflow.deployments.MlflowDeploymentClient.get_endpoint", get_endpoint),
        mock.patch("mlflow.deployments.MlflowDeploymentClient.predict", predict),
    ):
        return model.predict(frame)


def main():
    args = parse_args()
    set_deployments_target(args.rust_gateway)
    with tempfile.TemporaryDirectory() as directory:
        python_model, python_run = python_twin(Path(directory))
        compare_run_data(args.rust_gateway, args.rust_model, python_run)
        files, byte_equal = compare_layout(args.rust_model, python_model)
        rust_loaded = mlflow.pyfunc.load_model(args.rust_model)
        python_loaded = mlflow.pyfunc.load_model(python_model)
        frame = pd.DataFrame([{"thing": "books"}, {"thing": "coffee"}])
        rust_prediction = predict_through_rust_gateway(rust_loaded, args.rust_gateway, frame)
        python_prediction = predict_through_rust_gateway(python_loaded, args.rust_gateway, frame)
        expected = ["gateway:Write about books.", "gateway:Write about coffee."]
        assert rust_prediction == python_prediction == expected
        print(
            json.dumps(
                {
                    "files": files,
                    "byte_equal": byte_equal,
                    "mlmodel": "byte-equal after run/model UUID and UTC-line normalization",
                    "run_data": "params, metrics, and promptlab tags equal",
                    "rust_prediction": rust_prediction,
                    "python_prediction": python_prediction,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
