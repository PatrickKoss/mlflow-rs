import os
import shutil
from pathlib import Path

import pytest
import requests

from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore

from tests.tracking.integration_test_utils import _init_server

pytestmark = pytest.mark.skipif(
    os.environ.get("MLFLOW_SERVER_TYPE", "python").lower() != "rust",
    reason="The differential launches both implementations in the Rust parity job",
)

ADMIN = ("queue_admin", "queue-admin-password")
OWNER = ("queue_owner", "queue-owner-password")
READER = ("queue_reader", "queue-reader-password")


def _sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path}"


def _write_auth_config(path: Path, database_uri: str) -> Path:
    path.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = {database_uri}\n"
        f"admin_username = {ADMIN[0]}\n"
        f"admin_password = {ADMIN[1]}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
        "grant_default_workspace_access = false\n"
    )
    return path


def _request(url, method, path, auth, body=None):
    response = requests.request(method, f"{url}{path}", auth=auth, json=body)
    return response.status_code, response.content


def _permission_matrix(url, queue_id):
    base = "/api/3.0/mlflow/review-queues"
    return [
        # OWNER passes the create validator and reaches deterministic store
        # validation; READER is stopped at the authorization boundary.
        _request(
            url,
            "POST",
            f"{base}/create",
            OWNER,
            {"experiment_id": "1", "name": "default", "queue_type": "CUSTOM"},
        ),
        _request(
            url,
            "POST",
            f"{base}/create",
            READER,
            {"experiment_id": "1", "name": "default", "queue_type": "CUSTOM"},
        ),
        # Both users may read because READER is assigned to the queue.
        _request(url, "GET", f"{base}/get?queue_id={queue_id}", OWNER),
        _request(url, "GET", f"{base}/get?queue_id={queue_id}", READER),
        # A no-op update has a stable response: the EDIT owner passes, while a
        # READ member cannot mutate the queue.
        _request(url, "POST", f"{base}/update", OWNER, {"queue_id": queue_id}),
        _request(url, "POST", f"{base}/update", READER, {"queue_id": queue_id}),
        # The READ member cannot delete; the EDIT owner can delete CUSTOM.
        _request(url, "POST", f"{base}/delete", READER, {"queue_id": queue_id}),
        _request(url, "POST", f"{base}/delete", OWNER, {"queue_id": queue_id}),
    ]


def test_review_queue_crud_permissions_are_python_rust_byte_identical(tmp_path):
    baseline_tracking = tmp_path / "tracking-baseline.db"
    tracking = TrackingStore(_sqlite_uri(baseline_tracking), tmp_path.as_uri())
    experiment_id = tracking.create_experiment("review queue auth differential")
    assert experiment_id == "1"
    queue = tracking.create_review_queue(
        experiment_id,
        name="Two-user queue",
        queue_type="custom",
        created_by=OWNER[0],
        users=[READER[0]],
    )
    tracking.engine.dispose()

    baseline_auth = tmp_path / "auth-baseline.db"
    auth = AuthStore()
    auth.init_db(_sqlite_uri(baseline_auth))
    auth.create_user(*ADMIN, is_admin=True)
    auth.create_user(*OWNER)
    auth.create_user(*READER)
    auth.grant_user_permission(OWNER[0], "experiment", experiment_id, "EDIT")
    auth.grant_user_permission(READER[0], "experiment", experiment_id, "READ")
    auth.engine.dispose()

    tracking_paths = {
        implementation: tmp_path / f"tracking-{implementation}.db"
        for implementation in ("python", "rust")
    }
    auth_paths = {
        implementation: tmp_path / f"auth-{implementation}.db"
        for implementation in ("python", "rust")
    }
    configs = {}
    for implementation in ("python", "rust"):
        shutil.copyfile(baseline_tracking, tracking_paths[implementation])
        shutil.copyfile(baseline_auth, auth_paths[implementation])
        configs[implementation] = _write_auth_config(
            tmp_path / f"basic-auth-{implementation}.ini",
            _sqlite_uri(auth_paths[implementation]),
        )

    responses = {}
    for implementation, server_type in (("python", "fastapi"), ("rust", "rust")):
        extra_env = {
            "MLFLOW_AUTH_CONFIG_PATH": str(configs[implementation]),
            "MLFLOW_FLASK_SERVER_SECRET_KEY": "review-queue-differential-key",
            "_MLFLOW_SGI_NAME": "uvicorn",
        }
        if implementation == "rust":
            extra_env["MLFLOW_AUTH_DATABASE_URI"] = _sqlite_uri(auth_paths[implementation])
        with _init_server(
            backend_uri=_sqlite_uri(tracking_paths[implementation]),
            root_artifact_uri=(tmp_path / f"artifacts-{implementation}").as_uri(),
            extra_env=extra_env,
            app="mlflow.server.auth:create_app",
            server_type=server_type,
        ) as url:
            responses[implementation] = _permission_matrix(url, queue.queue_id)

    assert responses["rust"] == responses["python"]
    assert [status for status, _ in responses["rust"]] == [400, 403, 200, 200, 200, 403, 403, 200]
