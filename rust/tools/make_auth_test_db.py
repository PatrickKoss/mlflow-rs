"""Generate a real Alembic-migrated auth DB fixture for mlflow-auth tests.

Run from the repo root:

    uv run python rust/tools/make_auth_test_db.py

Produces (by default):
    rust/crates/mlflow-auth/tests/fixtures/basic_auth.db
    rust/crates/mlflow-auth/tests/fixtures/auth_fixture.json

The ``.db`` is a fully migrated MLflow **auth** database at the current Alembic
head (RUST_TRACKING_SERVER_PLAN.md §5.3, head ``f1a2b3c4d5e6``, version table
``alembic_version_auth``). It is created by pointing the genuine
``mlflow.server.auth.sqlalchemy_store.SqlAlchemyStore`` at a fresh SQLite file
(``init_db`` runs the whole auth migration chain), then creating a couple of
users with known passwords and attaching RBAC role/permission rows via raw SQL
(the RBAC mutation API is a later Rust task, so we seed the tables directly to
match the schema the Rust store reads).

``auth_fixture.json`` records everything the Rust cross-language test needs:

  * the users (username, known plaintext password, and the exact werkzeug hash
    string stored in the DB, one scrypt + one pbkdf2) so Rust can verify a
    Python-written hash;
  * a Rust-generated hash placeholder that a *separate* verification pass
    (``verify_auth_fixture.py``) checks with Python's ``check_password_hash``,
    proving the reverse direction;
  * the seeded role/permission rows so Rust can assert its grant reads;
  * the alembic head, asserted against the Rust-side constant.

This is the same fixture approach as the Fernet cross-language fixtures in
``rust/crates/mlflow-webhooks/tests/fixtures/``. Regenerate whenever the auth
Alembic head changes.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "rust" / "crates" / "mlflow-auth" / "tests" / "fixtures"
DEFAULT_DB = FIXTURE_DIR / "basic_auth.db"
DEFAULT_JSON = FIXTURE_DIR / "auth_fixture.json"

# Known users. Passwords are >12 chars to pass `_validate_password`.
SCRYPT_USER = ("alice_scrypt", "alice-password-123")
PBKDF2_USER = ("bob_pbkdf2", "bob-password-4567")
# A plaintext whose Rust-generated hash Python must accept (reverse direction).
RUST_ROUNDTRIP_PLAINTEXT = "rust-generated-secret-9"


def build_db(db_path: Path) -> dict:
    from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    uri = f"sqlite:///{db_path}"
    store = SqlAlchemyStore()
    # init_db runs migrate_if_needed(engine, "head"), creating the auth schema
    # (including the alembic_version_auth table) at head f1a2b3c4d5e6.
    store.init_db(uri)

    # Create the scrypt user through the genuine store (default werkzeug method
    # is scrypt), so its stored hash is exactly what Python writes at runtime.
    scrypt_user = store.create_user(SCRYPT_USER[0], SCRYPT_USER[1], is_admin=True)

    # For the pbkdf2 user we insert a pbkdf2-format hash directly so the fixture
    # exercises Rust's pbkdf2 verify path too (werkzeug's runtime default is
    # scrypt, but a DB migrated from an older werkzeug can hold pbkdf2 hashes).
    pbkdf2_hash = generate_password_hash(PBKDF2_USER[1], method="pbkdf2:sha256")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (PBKDF2_USER[0], pbkdf2_hash, 0),
        )
        conn.commit()

    # Read back the stored hashes and ids.
    with sqlite3.connect(db_path) as conn:
        rows = {
            username: (uid, pwhash, is_admin)
            for (uid, username, pwhash, is_admin) in conn.execute(
                "SELECT id, username, password_hash, is_admin FROM users"
            )
        }

    scrypt_id = rows[SCRYPT_USER[0]][0]
    pbkdf2_id = rows[PBKDF2_USER[0]][0]

    # Seed an RBAC role + permissions and assign it to the pbkdf2 user, so Rust
    # can read grants back. Columns match mlflow/server/auth/db/models.py.
    permissions = [
        ("experiment", "*", "EDIT"),
        ("registered_model", "my-model", "MANAGE"),
    ]
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO roles (name, workspace, description) VALUES (?, ?, ?)",
            ("editors", "default", "Can edit experiments"),
        )
        role_id = cur.lastrowid
        for resource_type, resource_pattern, permission in permissions:
            conn.execute(
                "INSERT INTO role_permissions "
                "(role_id, resource_type, resource_pattern, permission) VALUES (?, ?, ?, ?)",
                (role_id, resource_type, resource_pattern, permission),
            )
        conn.execute(
            "INSERT INTO user_role_assignments (user_id, role_id) VALUES (?, ?)",
            (pbkdf2_id, role_id),
        )
        conn.commit()

    # Sanity: both stored hashes verify with Python (the fixture must be sound).
    assert check_password_hash(rows[SCRYPT_USER[0]][1], SCRYPT_USER[1])
    assert check_password_hash(rows[PBKDF2_USER[0]][1], PBKDF2_USER[1])

    with sqlite3.connect(db_path) as conn:
        (head,) = conn.execute(
            "SELECT version_num FROM alembic_version_auth"
        ).fetchone()

    return {
        "alembic_head": head,
        "alembic_version_table": "alembic_version_auth",
        "users": [
            {
                "id": scrypt_id,
                "username": SCRYPT_USER[0],
                "password": SCRYPT_USER[1],
                "password_hash": rows[SCRYPT_USER[0]][1],
                "is_admin": bool(rows[SCRYPT_USER[0]][2]),
                "method": "scrypt",
            },
            {
                "id": pbkdf2_id,
                "username": PBKDF2_USER[0],
                "password": PBKDF2_USER[1],
                "password_hash": rows[PBKDF2_USER[0]][1],
                "is_admin": bool(rows[PBKDF2_USER[0]][2]),
                "method": "pbkdf2",
            },
        ],
        "roles": [
            {
                "id": role_id,
                "name": "editors",
                "workspace": "default",
                "description": "Can edit experiments",
                "assigned_user_id": pbkdf2_id,
                "permissions": [
                    {
                        "resource_type": rt,
                        "resource_pattern": rp,
                        "permission": perm,
                    }
                    for (rt, rp, perm) in permissions
                ],
            }
        ],
        # Reverse-direction check input: Rust generates a hash for this plaintext
        # and writes it into the fixture; verify_auth_fixture.py confirms Python
        # accepts it. Populated by the Rust test on first run; if absent, the
        # verify script skips (still sound because Rust also self-verifies).
        "rust_roundtrip_plaintext": RUST_ROUNDTRIP_PLAINTEXT,
        "rust_generated_hash": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    fixture = build_db(args.db)
    args.json.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote migrated auth SQLite fixture to {args.db} "
          f"(alembic head: {fixture['alembic_head']})")
    print(f"Wrote fixture metadata to {args.json}")


if __name__ == "__main__":
    main()
