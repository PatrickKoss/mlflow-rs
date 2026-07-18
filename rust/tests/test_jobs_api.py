from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore

from tests.tracking.integration_test_utils import _init_server


@dataclass
class RustJobs:
    url: str
    store: SqlAlchemyJobStore


@pytest.fixture(scope="module")
def rust_jobs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("rust-jobs-api")
    uri = f"sqlite:///{root / 'shared.db'}"
    store = SqlAlchemyJobStore(uri)
    with _init_server(uri, str(root / "artifacts"), server_type="rust") as url:
        yield RustJobs(url, store)


def test_python_created_job_is_readable_via_legacy_rust_route(rust_jobs: RustJobs):
    job = rust_jobs.store.create_job("python_job", json.dumps({"x": 1}), timeout=2.5)
    response = requests.get(f"{rust_jobs.url}/ajax-api/3.0/mlflow/jobs/{job.job_id}", timeout=10)
    assert response.status_code == 200
    assert response.content == b'{"result":null,"status":"PENDING","status_details":null}\n'


def test_python_created_job_is_cancellable_via_rust(rust_jobs: RustJobs):
    job = rust_jobs.store.create_job("python_job", "{}")
    response = requests.patch(
        f"{rust_jobs.url}/ajax-api/3.0/mlflow/jobs/cancel/{job.job_id}", timeout=10
    )
    assert response.status_code == 200
    assert response.content == b'{"result":null,"status":"CANCELED"}\n'
    assert str(rust_jobs.store.get_job(job.job_id).status) == "CANCELED"


def test_fastapi_alias_preserves_complete_python_job_model(rust_jobs: RustJobs):
    job = rust_jobs.store.create_job("python_job", json.dumps({"x": 1}), timeout=2.5)
    response = requests.get(f"{rust_jobs.url}/ajax-api/3.0/jobs/{job.job_id}", timeout=10)
    response.raise_for_status()
    body = response.json()
    assert body == {
        "job_id": job.job_id,
        "creation_time": job.creation_time,
        "job_name": "python_job",
        "params": {"x": 1},
        "timeout": 2.5,
        "status": "PENDING",
        "result": None,
        "retry_count": 0,
        "last_update_time": job.last_update_time,
        "status_details": None,
    }


def test_missing_job_has_python_error_shape(rust_jobs: RustJobs):
    response = requests.get(f"{rust_jobs.url}/ajax-api/3.0/mlflow/jobs/missing", timeout=10)
    assert response.status_code == 404
    assert response.json()["error_code"] == "RESOURCE_DOES_NOT_EXIST"
    assert response.json()["message"] == "Job with ID missing not found"
