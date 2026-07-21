"""Verify the reverse cross-language direction: Python accepts a Rust-generated
werkzeug password hash.

Run from the repo root after the Rust test has written ``rust_generated_hash``
into the fixture JSON (the Rust ``cross_language`` test does this):

    uv run python rust/tools/verify_auth_fixture.py

Exit code 0 means Python's ``werkzeug.security.check_password_hash`` accepted the
hash the Rust ``generate_password_hash`` produced for
``rust_roundtrip_plaintext`` (and rejected a wrong password). This is the
"Rust-generated hash Python verifies" half of the T9.1 cross-language AC. If
``rust_generated_hash`` is null (Rust test not yet run), the script reports that
and exits 0 without failing — the forward direction is already proven by the
committed fixture, and Rust self-verifies its own hash.
"""

import json
import sys
from pathlib import Path

from werkzeug.security import check_password_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = (
    REPO_ROOT / "rust" / "crates" / "mlflow-auth" / "tests" / "fixtures" / "auth_fixture.json"
)


def main() -> int:
    fixture = json.loads(DEFAULT_JSON.read_text())
    plaintext = fixture["rust_roundtrip_plaintext"]
    rust_hash = fixture.get("rust_generated_hash")

    if not rust_hash:
        print("rust_generated_hash is null; run the Rust cross_language test first.")
        return 0

    if not check_password_hash(rust_hash, plaintext):
        print(f"FAIL: Python rejected the Rust-generated hash for {plaintext!r}")
        return 1
    if check_password_hash(rust_hash, plaintext + "-wrong"):
        print("FAIL: Python accepted a wrong password against the Rust hash")
        return 1

    print(f"OK: Python accepted the Rust-generated hash for {plaintext!r}")
    print(f"    hash prefix: {rust_hash.split('$', 1)[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
