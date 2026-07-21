"""Generate Python gateway-secret fixtures for the T15.3 Rust crypto spike.

Run from the repository root:

    uv run --frozen python rust/spikes/gen_secrets_fixtures.py
"""

import base64
import itertools
import json
from pathlib import Path

from mlflow.utils.crypto import (
    AES_256_KEY_LENGTH,
    GCM_NONCE_LENGTH,
    MLFLOW_KEK_SALT,
    PBKDF2_ITERATIONS,
    KEKManager,
    _create_aad,
    _encrypt_secret,
    _mask_secret_value,
    _mask_string_value,
    rotate_secret_encryption,
)

OUTPUT = Path(__file__).parent / "fixtures" / "secrets_python.json"

PLAINTEXTS = [
    "",
    "sk-test-1234567890abcdef",
    "🔐 Secret with emoji 密钥 and pässwörd",
    "long:" + "0123456789abcdef" * 256,
]
PASSPHRASES = [
    "alpha-test-passphrase-with-at-least-32-characters",
    "βeta-test-passphrase-with-unicode-and-sufficient-length",
]
AADS = [
    ("123e4567-e89b-12d3-a456-426614174000", "provider-api-key"),
    ("秘密-id-🔐", "名前|with-delimiter"),
]
KEK_VERSIONS = [1, 42]

MASK_INPUTS = [
    "",
    "abc",
    "1234567",
    "12345678",
    "sk-proj-1234567890abcdef",
    "ghp_1234567890abcdef",
    "🔐秘密abcde",
    "a" * 200,
    None,
    12345678,
    {"nested": "not-a-string"},
]
MASK_DICTIONARIES = [
    {},
    {"api_key": "sk-proj-1234567890abcdef"},
    {"username": "admin-user", "password": "secret123"},
    {"short": "abc", "empty": ""},
    {"config": {"host": "localhost", "port": 8080}},
    {"enabled": True, "count": 12345678, "missing": None},
    {"unicode": "🔐秘密abcde"},
    {"k" * 200: "test-longer-value-here"},
]


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def generate_cases(managers: dict[tuple[str, int], KEKManager]) -> list[dict]:
    cases = []
    combinations = itertools.product(PLAINTEXTS, PASSPHRASES, AADS, KEK_VERSIONS)
    for index, (plaintext, passphrase, (secret_id, secret_name), version) in enumerate(
        combinations
    ):
        encrypted = _encrypt_secret(
            plaintext,
            managers[(passphrase, version)],
            secret_id,
            secret_name,
        )
        cases.append({
            "case_id": f"python-{index:02d}",
            "plaintext": plaintext,
            "passphrase": passphrase,
            "kek_version": version,
            "secret_id": secret_id,
            "secret_name": secret_name,
            "aad_b64": b64(_create_aad(secret_id, secret_name)),
            "encrypted_value_b64": b64(encrypted.encrypted_value),
            "wrapped_dek_b64": b64(encrypted.wrapped_dek),
        })
    return cases


def generate_rotations(managers: dict[tuple[str, int], KEKManager]) -> list[dict]:
    old_passphrase, old_version = PASSPHRASES[0], KEK_VERSIONS[0]
    new_passphrase, new_version = PASSPHRASES[1], KEK_VERSIONS[1]
    rotations = []
    for index, plaintext in enumerate((PLAINTEXTS[1], PLAINTEXTS[2])):
        secret_id, secret_name = AADS[index]
        encrypted = _encrypt_secret(
            plaintext,
            managers[(old_passphrase, old_version)],
            secret_id,
            secret_name,
        )
        rotated = rotate_secret_encryption(
            encrypted.encrypted_value,
            encrypted.wrapped_dek,
            managers[(old_passphrase, old_version)],
            managers[(new_passphrase, new_version)],
        )
        assert rotated.encrypted_value == encrypted.encrypted_value
        assert rotated.wrapped_dek != encrypted.wrapped_dek
        rotations.append({
            "case_id": f"python-rotation-{index}",
            "plaintext": plaintext,
            "secret_id": secret_id,
            "secret_name": secret_name,
            "old_passphrase": old_passphrase,
            "old_kek_version": old_version,
            "new_passphrase": new_passphrase,
            "new_kek_version": new_version,
            "encrypted_value_b64": b64(encrypted.encrypted_value),
            "old_wrapped_dek_b64": b64(encrypted.wrapped_dek),
            "new_wrapped_dek_b64": b64(rotated.wrapped_dek),
        })
    return rotations


def main() -> None:
    managers = {
        (passphrase, version): KEKManager(passphrase=passphrase, kek_version=version)
        for passphrase, version in itertools.product(PASSPHRASES, KEK_VERSIONS)
    }
    document = {
        "format": {
            "cipher": "AES-256-GCM",
            "nonce_bytes": GCM_NONCE_LENGTH,
            "tag_bytes": 16,
            "dek_bytes": AES_256_KEY_LENGTH,
            "pbkdf2_hash": "HMAC-SHA256",
            "pbkdf2_iterations": PBKDF2_ITERATIONS,
            "kek_salt_b64": b64(MLFLOW_KEK_SALT),
            "kek_version_encoding": "unsigned-u32-big-endian appended to salt",
            "aad_encoding": "utf-8(secret_id + '|' + secret_name)",
        },
        "cases": generate_cases(managers),
        "rotations": generate_rotations(managers),
        "masking": {
            "strings": [
                {"input": value, "masked": _mask_string_value(value)} for value in MASK_INPUTS
            ],
            "dictionaries": [
                {"input": value, "masked": _mask_secret_value(value)} for value in MASK_DICTIONARIES
            ],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(document['cases'])} envelopes, {len(document['rotations'])} rotations, "
        f"and {len(MASK_INPUTS) + len(MASK_DICTIONARIES)} masking fixtures to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
