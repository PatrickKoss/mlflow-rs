import configparser
import os
import re
from pathlib import Path
from typing import Literal

from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME
from mlflow.server.auth import auth_config
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str
from tests.tracking.integration_test_utils import _send_rest_tracking_post_request

PERMISSION = "READ"
NEW_PERMISSION = "EDIT"
ADMIN_USERNAME = auth_config.admin_username
ADMIN_PASSWORD = auth_config.admin_password

# T12.1: the same opt-in switch used by ``tests/tracking/test_rest_tracking.py``.
# DEFAULT is Python (Flask/FastAPI auth app); only ``MLFLOW_SERVER_TYPE=rust``
# routes the auth-server launch to the compiled Rust binary.
_RUN_AGAINST_RUST = os.environ.get("MLFLOW_SERVER_TYPE", "python").lower() == "rust"


def _auth_db_uri_from_config(config_path: Path) -> str:
    """Read ``database_uri`` out of an isolated ``basic_auth.ini``.

    The Rust server ignores the ini and reads the auth DB URI from the
    ``MLFLOW_AUTH_DATABASE_URI`` env var (see
    ``rust/crates/mlflow-server/src/main.rs::build_auth_store``), so tests must
    surface the ini's ``database_uri`` there.
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser["mlflow"]["database_uri"]


def _migrate_and_bootstrap_auth_db(auth_db_uri: str) -> None:
    """Migrate the auth DB and seed the admin user before launching Rust.

    The Rust server refuses an unmigrated auth DB and never bootstraps the admin
    account (both are Python-side responsibilities per §5.4). Mirror what
    ``mlflow.server.auth:create_app`` does on boot: ``SqlAlchemyStore.init_db``
    runs the auth Alembic chain, then ``create_admin_user`` seeds the admin. We
    reuse those helpers instead of reimplementing migrations.
    """
    from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

    store = SqlAlchemyStore()
    store.init_db(auth_db_uri)
    if not store.has_user(ADMIN_USERNAME):
        store.create_user(ADMIN_USERNAME, ADMIN_PASSWORD, is_admin=True)
    store.engine.dispose()


def _migrate_tracking_db(backend_uri: str) -> None:
    """Migrate the tracking backend store so the Rust server accepts it."""
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    store = SqlAlchemyStore(backend_uri, backend_uri)
    store.engine.dispose()


def resolve_auth_server_launch(
    backend_uri: str,
    extra_env: dict[str, str],
) -> tuple[Literal["flask", "fastapi", "rust"], dict[str, str]]:
    """Pick the auth-server launch flavor and env for ``_init_server`` (T12.1).

    Returns ``(server_type, extra_env)``. Without the switch this is a no-op that
    keeps the caller-supplied ``server_type`` decision (returned as ``"flask"``,
    the auth app's default). With ``MLFLOW_SERVER_TYPE=rust`` it migrates both the
    tracking and auth SQLite DBs, seeds the admin user, and injects
    ``MLFLOW_AUTH_DATABASE_URI`` so the Rust ``build_auth_store`` path binds to the
    isolated auth DB instead of the shipped ``basic_auth.db``.
    """
    if not _RUN_AGAINST_RUST:
        return "flask", extra_env

    config_path = extra_env.get("MLFLOW_AUTH_CONFIG_PATH")
    if not config_path:
        raise ValueError("Rust auth launch requires MLFLOW_AUTH_CONFIG_PATH in extra_env")
    auth_db_uri = _auth_db_uri_from_config(Path(config_path))

    _migrate_tracking_db(backend_uri)
    _migrate_and_bootstrap_auth_db(auth_db_uri)

    rust_env = {**extra_env, "MLFLOW_AUTH_DATABASE_URI": auth_db_uri}
    if read_uri := _read_replica_uri(extra_env):
        rust_env["MLFLOW_AUTH_READ_DATABASE_URI"] = read_uri
    return "rust", rust_env


def _read_replica_uri(extra_env: dict[str, str]) -> str | None:
    """Extract an optional read-replica URI from the isolated ini, if present."""
    config_path = extra_env.get("MLFLOW_AUTH_CONFIG_PATH")
    if not config_path:
        return None
    text = Path(config_path).read_text()
    match = re.search(r"^\s*read_database_uri\s*=\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def write_isolated_auth_config(tmp_path: Path) -> Path:
    """Write a basic_auth.ini under ``tmp_path`` whose ``database_uri`` points
    at an SQLite file inside the same dir, and return the config path.

    Tests that spawn the auth server via ``_init_server`` must point
    ``MLFLOW_AUTH_CONFIG_PATH`` at the returned file (through ``extra_env``) so
    the spawned auth server writes to the temp DB instead of the repo-root
    ``basic_auth.db`` shared with the dev server. Without this, integration
    tests pollute (and depending on the fixture, delete) the developer's local
    auth state.
    """
    config_path = tmp_path / "basic_auth.ini"
    db_path = tmp_path / "basic_auth.db"
    config_path.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = sqlite:///{db_path}\n"
        f"admin_username = {ADMIN_USERNAME}\n"
        f"admin_password = {ADMIN_PASSWORD}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
        "grant_default_workspace_access = false\n"
    )
    return config_path


def create_user(tracking_uri: str, username: str | None = None, password: str | None = None):
    username = random_str() if username is None else username
    password = random_str() if password is None else password
    response = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/2.0/mlflow/users/create",
        {
            "username": username,
            "password": password,
        },
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()
    return username, password


def grant_role_permission(
    tracking_uri: str,
    username: str,
    resource_type: str,
    resource_pattern: str,
    permission: str,
    workspace: str = DEFAULT_WORKSPACE_NAME,
    auth: tuple[str, str] | None = None,
) -> None:
    """
    Grant ``username`` ``permission`` on ``(resource_type, resource_pattern)`` via the
    role API. Creates a throwaway role in ``workspace`` with a single permission row,
    then assigns it to the user. Useful for auth-flow tests that previously called
    ``POST /mlflow/experiments/permissions/create`` (and equivalents) to set up state.
    """
    auth = auth or (ADMIN_USERNAME, ADMIN_PASSWORD)
    role_name = f"_test_{resource_type}_{random_str()}"
    create_resp = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/roles/create",
        {"name": role_name, "workspace": workspace},
        auth=auth,
    )
    create_resp.raise_for_status()
    role_id = create_resp.json()["role"]["id"]
    add_resp = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/roles/permissions/add",
        {
            "role_id": role_id,
            "resource_type": resource_type,
            "resource_pattern": resource_pattern,
            "permission": permission,
        },
        auth=auth,
    )
    add_resp.raise_for_status()
    assign_resp = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/roles/assign",
        {"username": username, "role_id": role_id},
        auth=auth,
    )
    assign_resp.raise_for_status()


class User:
    def __init__(self, username, password, monkeypatch):
        self.username = username
        self.password = password
        self.monkeypatch = monkeypatch

    def __enter__(self):
        self.monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, self.username)
        self.monkeypatch.setenv(MLFLOW_TRACKING_PASSWORD.name, self.password)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
        self.monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)
