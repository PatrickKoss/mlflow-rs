//! Cross-language auth-DB parity test (plan T9.1 AC: "Rust authenticates users
//! created by Python and vice versa on a shared `basic_auth.db`").
//!
//! ## Fixtures
//!
//! `tests/fixtures/basic_auth.db` is a genuine Alembic-migrated MLflow auth
//! database (head `f1a2b3c4d5e6`, version table `alembic_version_auth`) created
//! by `rust/tools/make_auth_test_db.py` via the real
//! `mlflow.server.auth.sqlalchemy_store.SqlAlchemyStore`. `auth_fixture.json`
//! records the users (known passwords + the exact werkzeug hashes stored in the
//! DB — one scrypt, one pbkdf2), the seeded RBAC role/permissions, and the
//! alembic head. Regenerate with:
//!
//! ```text
//! uv run --extra auth python rust/tools/make_auth_test_db.py
//! ```
//!
//! ## Directions covered
//!
//! * **Python -> Rust (verify):** Rust opens the Python-written DB, verifies the
//!   Alembic head, and `authenticate_user` succeeds for both the scrypt and the
//!   pbkdf2 user against the hashes Python stored — the "Rust authenticates
//!   users created by Python" half of the AC. It also reads the RBAC grants
//!   Python seeded.
//! * **Rust -> Python (generate):** Rust generates a hash for a known plaintext,
//!   self-verifies it, writes it into a copy of the fixture JSON, and (when
//!   `uv` is available) runs `rust/tools/verify_auth_fixture.py`, which asserts
//!   Python's `check_password_hash` accepts it — the "and vice versa" half. The
//!   direction is also proven offline by the `scrypt:32768:8:1` format + the
//!   RFC-7914 byte-exact vector in the unit tests, so the test still passes when
//!   `uv`/Python is unavailable (it reports skipped).

use std::path::{Path, PathBuf};
use std::process::Command;

use mlflow_auth::db::AuthDb;
use mlflow_auth::hash::{check_password_hash, generate_password_hash};
use mlflow_auth::{AuthStore, EXPECTED_AUTH_ALEMBIC_HEAD};
use mlflow_store::PoolConfig;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    alembic_head: String,
    alembic_version_table: String,
    users: Vec<FixtureUser>,
    roles: Vec<FixtureRole>,
    rust_roundtrip_plaintext: String,
}

#[derive(Deserialize)]
struct FixtureUser {
    id: i64,
    username: String,
    password: String,
    password_hash: String,
    is_admin: bool,
    method: String,
}

#[derive(Deserialize)]
struct FixtureRole {
    id: i64,
    name: String,
    workspace: String,
    description: Option<String>,
    assigned_user_id: i64,
    permissions: Vec<FixturePermission>,
}

#[derive(Deserialize)]
struct FixturePermission {
    resource_type: String,
    resource_pattern: String,
    permission: String,
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_fixture() -> Fixture {
    let json = std::fs::read_to_string(fixtures_dir().join("auth_fixture.json"))
        .expect("read auth_fixture.json");
    serde_json::from_str(&json).expect("parse auth_fixture.json")
}

/// Copy the fixture DB to a temp file so tests never mutate the committed
/// fixture (the auth store's writes would otherwise change it).
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_auth_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(fixtures_dir().join("basic_auth.db"), &path).expect("copy fixture db");
        TempDb { path }
    }

    fn uri(&self) -> String {
        format!("sqlite:///{}", self.path.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn open_store(db: &TempDb) -> AuthStore {
    let auth_db = AuthDb::connect_and_verify_with(&db.uri(), None, PoolConfig::default())
        .await
        .expect("connect + verify auth fixture");
    AuthStore::new(auth_db)
}

#[tokio::test]
async fn fixture_head_matches_rust_constant() {
    let fx = load_fixture();
    assert_eq!(fx.alembic_head, EXPECTED_AUTH_ALEMBIC_HEAD);
    assert_eq!(fx.alembic_version_table, "alembic_version_auth");
}

#[tokio::test]
async fn verify_schema_accepts_migrated_fixture() {
    let db = TempDb::new("verify_schema");
    // connect_and_verify would error on a stale/uninitialized head.
    let _ = open_store(&db).await;
}

#[tokio::test]
async fn rust_authenticates_python_created_users() {
    let fx = load_fixture();
    let db = TempDb::new("authn");
    let store = open_store(&db).await;

    // Forward direction: every user Python created authenticates in Rust
    // against the hash Python stored (both scrypt and pbkdf2 variants).
    let mut saw_scrypt = false;
    let mut saw_pbkdf2 = false;
    for u in &fx.users {
        assert!(
            store.authenticate_user(&u.username, &u.password).await,
            "auth should succeed for {} ({})",
            u.username,
            u.method
        );
        // Wrong password must fail.
        assert!(
            !store
                .authenticate_user(&u.username, &format!("{}-wrong", u.password))
                .await
        );
        // The stored hash is exactly what the DB holds; verify directly too.
        assert!(check_password_hash(&u.password_hash, &u.password));
        match u.method.as_str() {
            "scrypt" => {
                assert!(u.password_hash.starts_with("scrypt:32768:8:1$"));
                saw_scrypt = true;
            }
            "pbkdf2" => {
                assert!(u.password_hash.starts_with("pbkdf2:sha256:1000000$"));
                saw_pbkdf2 = true;
            }
            other => panic!("unexpected method {other}"),
        }

        // Entity fields round-trip.
        let got = store.get_user(&u.username).await.unwrap();
        assert_eq!(got.id, u.id);
        assert_eq!(got.is_admin, u.is_admin);
        assert_eq!(got.password_hash, u.password_hash);
    }
    assert!(
        saw_scrypt && saw_pbkdf2,
        "fixture must cover both hash methods"
    );

    // Missing user authenticates false (not an error).
    assert!(!store.authenticate_user("nobody", "whatever12345").await);
}

#[tokio::test]
async fn rust_reads_python_seeded_rbac_grants() {
    let fx = load_fixture();
    let db = TempDb::new("rbac");
    let store = open_store(&db).await;

    let expected_role = &fx.roles[0];
    let assignee = fx
        .users
        .iter()
        .find(|u| u.id == expected_role.assigned_user_id)
        .expect("assignee in fixture");

    let roles = store.get_user_roles(&assignee.username).await.unwrap();
    let role = roles
        .iter()
        .find(|r| r.name == expected_role.name)
        .expect("assigned role present");

    assert_eq!(role.id, expected_role.id);
    assert_eq!(role.workspace, expected_role.workspace);
    assert_eq!(role.description, expected_role.description);
    assert_eq!(role.permissions.len(), expected_role.permissions.len());
    for want in &expected_role.permissions {
        assert!(
            role.permissions
                .iter()
                .any(|p| p.resource_type == want.resource_type
                    && p.resource_pattern == want.resource_pattern
                    && p.permission == want.permission),
            "missing permission {want:?}",
        );
    }

    // The scrypt user has no role assignments.
    let scrypt_user = fx.users.iter().find(|u| u.method == "scrypt").unwrap();
    assert!(store
        .get_user_roles(&scrypt_user.username)
        .await
        .unwrap()
        .is_empty());
}

impl std::fmt::Debug for FixturePermission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}={}",
            self.resource_type, self.resource_pattern, self.permission
        )
    }
}

#[tokio::test]
async fn rust_generated_hash_round_trips_and_python_accepts_it() {
    let fx = load_fixture();
    let plaintext = &fx.rust_roundtrip_plaintext;

    // Rust generates a werkzeug-format hash and self-verifies it.
    let rust_hash = generate_password_hash(plaintext).expect("generate");
    assert!(rust_hash.starts_with("scrypt:32768:8:1$"));
    assert!(check_password_hash(&rust_hash, plaintext));
    assert!(!check_password_hash(
        &rust_hash,
        &format!("{plaintext}-wrong")
    ));

    // Also verify Rust can create a user and re-authenticate (write path).
    let db = TempDb::new("gen_write");
    let store = open_store(&db).await;
    let created = store
        .create_user("carol_rust", "carol-password-xyz", false)
        .await
        .expect("create_user");
    assert!(created.password_hash.starts_with("scrypt:32768:8:1$"));
    assert!(
        store
            .authenticate_user("carol_rust", "carol-password-xyz")
            .await
    );
    // Duplicate username is a RESOURCE_ALREADY_EXISTS.
    assert!(store
        .create_user("carol_rust", "carol-password-xyz", false)
        .await
        .is_err());

    // Reverse direction: write the Rust hash into a temp copy of the fixture
    // JSON and let Python's check_password_hash validate it. Skips cleanly if
    // uv/Python is unavailable (the offline byte-exact vectors already prove
    // format compatibility).
    run_python_reverse_check(&rust_hash, plaintext);
}

#[tokio::test]
async fn delete_user_removes_the_user() {
    // `AuthStore::delete_user` (T9.2 store addition): create a user, delete it,
    // and confirm it's gone and can't authenticate.
    let db = TempDb::new("delete_user");
    let store = open_store(&db).await;
    let user = store
        .create_user("dave_rust", "dave-password-xyz", false)
        .await
        .expect("create_user");
    let role = store
        .create_role("dave_viewer", "default", None)
        .await
        .expect("create_role");
    store
        .assign_role_to_user(user.id, role.id)
        .await
        .expect("assign_role_to_user");
    assert!(store.has_user("dave_rust").await.unwrap());

    store.delete_user("dave_rust").await.expect("delete_user");
    assert!(!store.has_user("dave_rust").await.unwrap());
    assert!(
        !store
            .authenticate_user("dave_rust", "dave-password-xyz")
            .await
    );
    // get_user now errors RESOURCE_DOES_NOT_EXIST.
    assert!(store.get_user("dave_rust").await.is_err());
    // Deleting a missing user errors (matches `_get_user`'s NoResultFound).
    assert!(store.delete_user("dave_rust").await.is_err());
}

/// Write the Rust-generated hash into a temp JSON alongside the plaintext and
/// run `verify_auth_fixture.py` against it via `uv`. A missing `uv` (offline
/// CI) is reported as skipped, not a failure.
fn run_python_reverse_check(rust_hash: &str, plaintext: &str) {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3) // crates/mlflow-auth -> crates -> rust -> repo root
        .expect("repo root")
        .to_path_buf();

    let tmp_json = std::env::temp_dir().join(format!(
        "mlflow_rust_auth_reverse_{}.json",
        std::process::id()
    ));
    let payload = serde_json::json!({
        "rust_roundtrip_plaintext": plaintext,
        "rust_generated_hash": rust_hash,
    });
    std::fs::write(&tmp_json, serde_json::to_vec_pretty(&payload).unwrap()).unwrap();

    // Inline the verification with a small Python one-liner so we don't have to
    // temporarily overwrite the committed fixture. Uses the same werkzeug the
    // fixture generator used.
    let script = format!(
        "import json,sys; \
         from werkzeug.security import check_password_hash; \
         d=json.load(open(r'{}')); \
         ok=check_password_hash(d['rust_generated_hash'], d['rust_roundtrip_plaintext']); \
         bad=check_password_hash(d['rust_generated_hash'], d['rust_roundtrip_plaintext']+'-wrong'); \
         sys.exit(0 if (ok and not bad) else 1)",
        tmp_json.display()
    );

    let output = Command::new("uv")
        .current_dir(&repo_root)
        .args(["run", "--extra", "auth", "python", "-c", &script])
        .output();

    let _ = std::fs::remove_file(&tmp_json);

    match output {
        Ok(out) if out.status.success() => {
            // Python accepted the Rust-generated hash: reverse direction proven.
        }
        Ok(out) => {
            // uv ran but the check failed -> a genuine parity break. But only
            // fail if werkzeug was actually importable; a missing extra prints
            // an ImportError to stderr, which we treat as skipped.
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("ModuleNotFoundError") || stderr.contains("No module named") {
                eprintln!("skipping Python reverse check: werkzeug unavailable ({stderr})");
            } else {
                panic!(
                    "Python rejected the Rust-generated hash (reverse-direction parity break): \
                     stderr={stderr}"
                );
            }
        }
        Err(e) => {
            eprintln!("skipping Python reverse check: uv not runnable ({e})");
        }
    }
}
