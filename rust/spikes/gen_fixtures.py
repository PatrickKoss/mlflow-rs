"""Generate cross-language crypto fixtures for the Rust tracking-server spike.

Run from the repo root:

    uv run --frozen python rust/spikes/gen_fixtures.py

Produces:
    rust/spikes/fixtures/werkzeug_hashes.json  (werkzeug password hashes)
    rust/spikes/fixtures/fernet.json           (a Fernet key + token)

The Rust tests read these fixtures and must be able to verify/decrypt them.
The reverse direction (Python verifying Rust output) is handled by verify.py.
"""

import json
from importlib.metadata import version
from pathlib import Path

from cryptography.fernet import Fernet
from werkzeug.security import generate_password_hash

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Passwords include ASCII, empty, whitespace, and unicode to stress the
# UTF-8 handling on both sides.
PASSWORDS = [
    "test",
    "correct horse battery staple",
    "",
    "pässwörd-ünïcöde-\U0001f510",  # unicode + emoji lock
    "a" * 128,
]


def gen_werkzeug() -> dict:
    entries = []
    for pw in PASSWORDS:
        entries.extend(
            {
                "password": pw,
                "method": method,
                "hash": generate_password_hash(pw, method=method),
            }
            for method in ("scrypt", "pbkdf2:sha256")
        )
    # Also capture whatever the library default is, so the Rust side proves it
    # handles the exact string a real auth DB would store.
    default_hash = generate_password_hash("test")
    return {
        "werkzeug_version": version("werkzeug"),
        "default_method_example": default_hash,
        "entries": entries,
    }


def gen_fernet() -> dict:
    key = Fernet.generate_key()
    plaintext = "s3cr3t-webhook-signing-key \U0001f511"
    token = Fernet(key).encrypt(plaintext.encode())
    return {
        "cryptography_version": version("cryptography"),
        "key": key.decode(),
        "plaintext": plaintext,
        "token": token.decode(),
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "werkzeug_hashes.json").write_text(
        json.dumps(gen_werkzeug(), indent=2, ensure_ascii=False) + "\n"
    )
    (FIXTURE_DIR / "fernet.json").write_text(
        json.dumps(gen_fernet(), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Wrote fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
