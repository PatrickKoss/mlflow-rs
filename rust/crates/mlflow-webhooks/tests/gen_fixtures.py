#!/usr/bin/env python3
"""Generate cross-language fixtures for the `mlflow-webhooks` crate tests.

Run from the repo root with the MLflow dev environment active:

    uv run python rust/crates/mlflow-webhooks/tests/gen_fixtures.py

or, if `cryptography` is importable directly:

    python3 rust/crates/mlflow-webhooks/tests/gen_fixtures.py

It writes two fixtures next to this script:

  * ``fernet.json`` — a FIXED Fernet key plus a Python-``cryptography``-generated
    token for a known plaintext (webhook secret). The Rust test decrypts the
    token and asserts it matches the plaintext (Python -> Rust direction). The
    reverse direction (Rust -> Python) is covered by a Rust encrypt round-trip
    plus this same fixed key, which Python can decrypt with
    ``Fernet(key).decrypt(token)``.

  * ``signature.json`` — the exact HMAC-SHA256 ``v1,<b64>`` signature Python's
    ``mlflow.webhooks.delivery._generate_hmac_signature`` produces over a known
    ``{delivery_id}.{timestamp}.{payload}`` triple, so the Rust signer is byte-
    matched against Python.

The key is intentionally FIXED (not ``Fernet.generate_key()``) so the fixture is
deterministic and re-running this script only rotates the token's random IV, not
the key — keeping the committed key stable for the Rust decrypt test.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

from cryptography.fernet import Fernet

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# A fixed, valid Fernet key: url-safe-base64 of 32 deterministic raw bytes
# (0x00..0x1f). Fixed on purpose so the committed key is stable across re-runs.
FERNET_KEY = base64.urlsafe_b64encode(bytes(range(32))).decode()
# Webhook secret plaintext, including non-ASCII to exercise UTF-8 round-tripping.
FERNET_PLAINTEXT = "s3cr3t-webhook-signing-key \U0001f511"

# Signature fixture inputs (Standard-Webhooks-style signed content).
SIG_SECRET = "my-webhook-secret"
SIG_DELIVERY_ID = "11111111-2222-3333-4444-555555555555"
SIG_TIMESTAMP = "1704067200"
SIG_PAYLOAD = '{"entity":"registered_model","action":"created","data":{"name":"m"}}'
SIGNATURE_VERSION = "v1"


def gen_fernet() -> dict:
    token = Fernet(FERNET_KEY.encode()).encrypt(FERNET_PLAINTEXT.encode())
    return {
        "key": FERNET_KEY,
        "plaintext": FERNET_PLAINTEXT,
        "token": token.decode(),
    }


def gen_signature() -> dict:
    signed_content = f"{SIG_DELIVERY_ID}.{SIG_TIMESTAMP}.{SIG_PAYLOAD}"
    digest = hmac.new(
        SIG_SECRET.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    signature = f"{SIGNATURE_VERSION},{base64.b64encode(digest).decode('utf-8')}"
    return {
        "secret": SIG_SECRET,
        "delivery_id": SIG_DELIVERY_ID,
        "timestamp": SIG_TIMESTAMP,
        "payload": SIG_PAYLOAD,
        "signature": signature,
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "fernet.json").write_text(
        json.dumps(gen_fernet(), indent=2, ensure_ascii=False) + "\n"
    )
    (FIXTURE_DIR / "signature.json").write_text(
        json.dumps(gen_signature(), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {FIXTURE_DIR / 'fernet.json'}")
    print(f"wrote {FIXTURE_DIR / 'signature.json'}")


if __name__ == "__main__":
    main()
