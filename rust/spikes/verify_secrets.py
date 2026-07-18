"""Verify Rust-produced T15.3 envelopes with `mlflow.utils.crypto`.

Run from the repository root:

    uv run --frozen python rust/spikes/verify_secrets.py
"""

import base64
import json
import sys
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.utils.crypto import KEKManager, _create_aad, _decrypt_secret

DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "secrets_rust.json"


def decode(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def main() -> int:
    fixture_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIXTURE
    document = json.loads(fixture_path.read_text())
    managers: dict[tuple[str, int], KEKManager] = {}

    def manager(passphrase: str, version: int) -> KEKManager:
        return managers.setdefault(
            (passphrase, version), KEKManager(passphrase=passphrase, kek_version=version)
        )

    for case in document["cases"]:
        assert decode(case["aad_b64"]) == _create_aad(case["secret_id"], case["secret_name"])
        assert len(decode(case["wrapped_dek_b64"])) == 12 + 32 + 16
        plaintext = _decrypt_secret(
            decode(case["encrypted_value_b64"]),
            decode(case["wrapped_dek_b64"]),
            manager(case["passphrase"], case["kek_version"]),
            case["secret_id"],
            case["secret_name"],
        )
        assert plaintext == case["plaintext"], case["case_id"]

    for case in document["rotations"]:
        encrypted_value = decode(case["encrypted_value_b64"])
        old_wrapped_dek = decode(case["old_wrapped_dek_b64"])
        new_wrapped_dek = decode(case["new_wrapped_dek_b64"])
        old_manager = manager(case["old_passphrase"], case["old_kek_version"])
        new_manager = manager(case["new_passphrase"], case["new_kek_version"])
        assert (
            _decrypt_secret(
                encrypted_value,
                old_wrapped_dek,
                old_manager,
                case["secret_id"],
                case["secret_name"],
            )
            == case["plaintext"]
        )
        assert (
            _decrypt_secret(
                encrypted_value,
                new_wrapped_dek,
                new_manager,
                case["secret_id"],
                case["secret_name"],
            )
            == case["plaintext"]
        )
        try:
            _decrypt_secret(
                encrypted_value,
                new_wrapped_dek,
                old_manager,
                case["secret_id"],
                case["secret_name"],
            )
        except MlflowException:
            pass
        else:
            raise AssertionError(f"old KEK unexpectedly decrypted {case['case_id']}")

    print(  # noqa: T201
        f"Verified {len(document['cases'])} Rust envelopes and "
        f"{len(document['rotations'])} Rust rotations with Python"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
