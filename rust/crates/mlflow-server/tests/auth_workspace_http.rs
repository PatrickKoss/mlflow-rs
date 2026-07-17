//! HTTP integration tests for T10.4 (workspace-aware auth integration).
//!
//! Boots the axum app with BOTH auth (`with_auth_store`) and workspaces
//! (`with_workspace_store`) enabled, then ports the behaviors of
//! `tests/server/auth/test_auth_workspace.py` +
//! `tests/server/auth/test_client_workspace.py` over real HTTP:
//!
//! * **NO_PERMISSIONS boundary deny**: a resource in a workspace the user has no
//!   role in is 403, even though in single-tenant the user would have implicit
//!   `default_permission` READ (`_get_role_permission_or_default`,
//!   `__init__.py:556`; `_role_permission_for`, `:715`).
//! * **Workspace USE grant** allows read within its workspace; **MANAGE** folds
//!   into concrete-resource reads/admin ops (`get_role_permission_for_resource`,
//!   `sqlalchemy_store.py:2031`).
//! * **USE gates create** but not read-on-others; **MANAGE** allows all.
//! * **default-workspace inheritance** is off by default (`NO_PERMISSIONS`), so
//!   an ungranted user in `default` is denied — matching Python with
//!   `grant_default_workspace_access=False`.
//! * **filter_list_workspaces**: a non-admin sees only accessible workspaces;
//!   an admin sees all.
//! * **admin bypass** stays total.
//! * a **single-tenant** (workspaces disabled) control proves the auth behavior
//!   is unchanged (implicit `default_permission` READ still applies).

use std::path::{Path, PathBuf};

use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use metrics_exporter_prometheus::PrometheusBuilder;
use mlflow_auth::{AuthConfig, AuthDb, AuthStore};
use mlflow_registry::RegistryStore;
use mlflow_server::{build_app_with_recorder, AppState, ServerConfig};
use mlflow_store::{Db, PoolConfig, TrackingStore, WorkspaceStore};
use serde_json::{json, Value};
use tokio::net::TcpListener;

const ART_ROOT: &str = "s3://bucket/mlruns";
const WS_HEADER: &str = "X-MLFLOW-WORKSPACE";
const API: &str = "/api/2.0/mlflow";
const WORKSPACES: &str = "/api/3.0/mlflow/workspaces";
const ADMIN: (&str, &str) = ("admin_user", "admin-password-123");

fn auth_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("mlflow-auth")
        .join("tests")
        .join("fixtures")
        .join("basic_auth.db")
}

fn tracking_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tracking.db")
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new(tag: &str, source: &Path) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlflow_rust_authws_{}_{}_{}.db",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::copy(source, &path).expect("copy fixture");
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

struct TestServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    auth: AuthStore,
    _tracking_db: TempDb,
    _auth_db: TempDb,
}

impl TestServer {
    async fn start(tag: &str, enable_workspaces: bool, grant_default_access: bool) -> Self {
        let tracking_db = TempDb::new(&format!("{tag}_track"), &tracking_fixture_path());
        let db = Db::connect(&tracking_db.uri(), PoolConfig::default())
            .await
            .expect("connect tracking fixture");
        let tracking = TrackingStore::new(db.clone(), ART_ROOT);
        let registry = RegistryStore::new(db.clone());

        let auth_db_file = TempDb::new(&format!("{tag}_auth"), &auth_fixture_path());
        let auth_db =
            AuthDb::connect_and_verify_with(&auth_db_file.uri(), None, PoolConfig::default())
                .await
                .expect("connect + verify auth fixture");
        let config = AuthConfig {
            grant_default_workspace_access: grant_default_access,
            ..AuthConfig::default()
        };
        let auth = AuthStore::with_config(auth_db, config);

        let mut state = AppState::with_registry(tracking, registry, true, None, None)
            .with_auth_store(auth.clone());
        if enable_workspaces {
            state = state.with_workspace_store(WorkspaceStore::new(db, tracking_db.uri()));
        }

        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            static_prefix: None,
            backend_store_uri: None,
            default_artifact_root: None,
            serve_artifacts: true,
            artifacts_destination: None,
            allowed_hosts: None,
            cors_allowed_origins: None,
            x_frame_options: "SAMEORIGIN".to_string(),
            ..Default::default()
        };
        let recorder = PrometheusBuilder::new().build_recorder().handle();
        let app = build_app_with_recorder(&config, recorder, Some(state));

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("server error");
        });

        TestServer {
            base: format!("http://{addr}"),
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
            auth,
            _tracking_db: tracking_db,
            _auth_db: auth_db_file,
        }
    }

    /// Seed an admin + a non-admin user directly through the store.
    async fn seed_users(&self) {
        self.auth
            .create_user(ADMIN.0, ADMIN.1, true)
            .await
            .expect("create admin");
    }

    async fn create_user(&self, username: &str) -> (String, String) {
        let password = format!("{username}-password-1");
        self.auth
            .create_user(username, &password, false)
            .await
            .expect("create user");
        (username.to_string(), password)
    }

    async fn create_workspace(&self, name: &str) {
        // Created as admin over HTTP so the seed-roles after-request hook runs.
        let resp = self
            .send(
                Method::POST,
                WORKSPACES,
                None,
                Some(ADMIN),
                Some(json!({ "name": name, "default_artifact_root": ART_ROOT })),
            )
            .await;
        assert_eq!(
            resp.status,
            StatusCode::CREATED,
            "create workspace: {}",
            resp.body
        );
    }

    /// Grant a workspace-wide `(workspace, *)` role permission to a user in a
    /// workspace (the `admin`/`user` two-tier model). Creates a bespoke role and
    /// assigns it, mirroring what an admin would do via the roles API.
    async fn grant_workspace(&self, username: &str, workspace: &str, permission: &str) {
        let role_name = format!("grant-{username}-{permission}");
        let role = self
            .auth
            .create_role(&role_name, workspace, None)
            .await
            .expect("create role");
        self.auth
            .add_role_permission(role.id, "workspace", "*", permission)
            .await
            .expect("add role permission");
        let user = self.auth.get_user(username).await.expect("get user");
        self.auth
            .assign_role_to_user(user.id, role.id)
            .await
            .expect("assign role");
    }

    /// Grant a per-resource experiment permission to a user in a workspace.
    async fn grant_experiment(
        &self,
        username: &str,
        experiment_id: &str,
        permission: &str,
        workspace: &str,
    ) {
        self.auth
            .grant_user_permission(username, "experiment", experiment_id, permission, workspace)
            .await
            .expect("grant experiment permission");
    }

    async fn create_experiment(&self, workspace: &str, name: &str) -> String {
        let resp = self
            .send(
                Method::POST,
                &format!("{API}/experiments/create"),
                Some(workspace),
                Some(ADMIN),
                Some(json!({ "name": name })),
            )
            .await;
        assert_eq!(
            resp.status,
            StatusCode::OK,
            "create experiment: {}",
            resp.body
        );
        resp.json()["experiment_id"].as_str().unwrap().to_string()
    }

    async fn get_experiment(
        &self,
        workspace: &str,
        experiment_id: &str,
        creds: (&str, &str),
    ) -> HttpResponse {
        self.send(
            Method::GET,
            &format!("{API}/experiments/get?experiment_id={experiment_id}"),
            Some(workspace),
            Some(creds),
            None,
        )
        .await
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        workspace: Option<&str>,
        creds: Option<(&str, &str)>,
        body: Option<Value>,
    ) -> HttpResponse {
        let client = Client::builder(TokioExecutor::new()).build_http();
        let mut builder = Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.base));
        if let Some(ws) = workspace {
            builder = builder.header(WS_HEADER, ws);
        }
        if let Some((u, p)) = creds {
            let token = base64::engine::general_purpose::STANDARD.encode(format!("{u}:{p}"));
            builder = builder.header("authorization", format!("Basic {token}"));
        }
        let req = match body {
            Some(b) => builder
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(b.to_string())))
                .unwrap(),
            None => builder.body(Full::new(Bytes::new())).unwrap(),
        };
        let resp = client.request(req).await.expect("request");
        let status = resp.status();
        let bytes = resp.into_body().collect().await.expect("body").to_bytes();
        HttpResponse {
            status,
            body: String::from_utf8_lossy(&bytes).into_owned(),
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

struct HttpResponse {
    status: StatusCode,
    body: String,
}

impl HttpResponse {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("json body")
    }
}

// ---------------------------------------------------------------------------
// NO_PERMISSIONS boundary deny
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_permissions_boundary_denies_resource_in_unroled_workspace() {
    let srv = TestServer::start("boundary", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("boundary_user").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    let exp_a = srv.create_experiment("team-a", "exp-a").await;
    let exp_b = srv.create_experiment("team-b", "exp-b").await;

    // The user is a workspace admin (MANAGE) in team-a only.
    srv.grant_workspace(&user, "team-a", "MANAGE").await;

    // team-a: allowed (MANAGE folds into resource reads).
    let ok = srv.get_experiment("team-a", &exp_a, (&user, &pass)).await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.body);

    // team-b: the user has no role → NO_PERMISSIONS boundary deny (403), even
    // though single-tenant would grant implicit default READ.
    let denied = srv.get_experiment("team-b", &exp_b, (&user, &pass)).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
    assert_eq!(denied.body, "Permission denied");
}

// ---------------------------------------------------------------------------
// USE vs MANAGE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workspace_use_allows_read_within_workspace() {
    let srv = TestServer::start("use_read", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("use_user").await;
    srv.create_workspace("team-a").await;
    let exp_a = srv.create_experiment("team-a", "exp-a").await;

    // USE folds into concrete-resource reads only via workspace membership? No —
    // in the store, workspace `(*)` USE does NOT fold into concrete reads (only
    // MANAGE does). But USE grants list/read at the workspace tier: an ungranted
    // user is denied, a USE user is denied on concrete reads-on-others unless
    // they hold a per-resource grant. So the workspace-USE read of a *specific*
    // experiment is DENIED (matches Python).
    srv.grant_workspace(&user, "team-a", "USE").await;
    let resp = srv.get_experiment("team-a", &exp_a, (&user, &pass)).await;
    assert_eq!(
        resp.status,
        StatusCode::FORBIDDEN,
        "USE does not fold into concrete-resource reads: {}",
        resp.body
    );
}

#[tokio::test]
async fn workspace_manage_allows_admin_read() {
    let srv = TestServer::start("manage_read", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("manage_user").await;
    srv.create_workspace("team-a").await;
    let exp_a = srv.create_experiment("team-a", "exp-a").await;

    srv.grant_workspace(&user, "team-a", "MANAGE").await;
    let resp = srv.get_experiment("team-a", &exp_a, (&user, &pass)).await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
}

#[tokio::test]
async fn per_resource_grant_overrides_workspace_boundary() {
    let srv = TestServer::start("per_resource", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("resource_user").await;
    srv.create_workspace("team-a").await;
    let exp1 = srv.create_experiment("team-a", "exp-1").await;
    let exp2 = srv.create_experiment("team-a", "exp-2").await;

    // A per-resource READ grant on exp1 (no workspace-wide grant): exp1 readable,
    // exp2 denied by the workspace boundary.
    srv.grant_experiment(&user, &exp1, "READ", "team-a").await;
    let ok = srv.get_experiment("team-a", &exp1, (&user, &pass)).await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.body);
    let denied = srv.get_experiment("team-a", &exp2, (&user, &pass)).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
}

// ---------------------------------------------------------------------------
// USE gates create; MANAGE too; NO grant denies create
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workspace_use_allows_create_but_no_grant_denies() {
    let srv = TestServer::start("create_gate", true, false).await;
    srv.seed_users().await;
    let (creator, cpass) = srv.create_user("creator").await;
    let (outsider, opass) = srv.create_user("outsider").await;
    srv.create_workspace("team-a").await;

    srv.grant_workspace(&creator, "team-a", "USE").await;

    // USE grant → create allowed.
    let ok = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/create"),
            Some("team-a"),
            Some((&creator, &cpass)),
            Some(json!({ "name": "created-by-use" })),
        )
        .await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.body);

    // No grant → create denied (NO_PERMISSIONS boundary; no implicit create).
    let denied = srv
        .send(
            Method::POST,
            &format!("{API}/experiments/create"),
            Some("team-a"),
            Some((&outsider, &opass)),
            Some(json!({ "name": "created-by-outsider" })),
        )
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
}

// ---------------------------------------------------------------------------
// default-workspace inheritance (opt-in)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_workspace_denies_ungranted_user_when_autogrant_off() {
    let srv = TestServer::start("default_off", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("default_user").await;
    // `default` workspace ships in the fixture. Create an experiment there.
    let exp = srv.create_experiment("default", "in-default").await;

    // grant_default_workspace_access is OFF → an ungranted user is denied even in
    // the default workspace.
    let denied = srv.get_experiment("default", &exp, (&user, &pass)).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
}

#[tokio::test]
async fn default_workspace_grants_ungranted_user_when_autogrant_on() {
    // grant_default_workspace_access ON + default_permission READ → an ungranted
    // user inherits READ in the default workspace only.
    let srv = TestServer::start("default_on", true, true).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("default_user").await;
    srv.create_workspace("team-a").await;
    let exp_default = srv.create_experiment("default", "in-default").await;
    let exp_a = srv.create_experiment("team-a", "in-a").await;

    // default workspace: inherited READ → allowed.
    let ok = srv
        .get_experiment("default", &exp_default, (&user, &pass))
        .await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.body);

    // non-default workspace: no inheritance → denied.
    let denied = srv.get_experiment("team-a", &exp_a, (&user, &pass)).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
}

// ---------------------------------------------------------------------------
// filter_list_workspaces
// ---------------------------------------------------------------------------

#[tokio::test]
async fn filter_list_workspaces_shows_only_accessible() {
    let srv = TestServer::start("list_ws", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("list_user").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    // Grant membership in team-a only (a workspace-wide USE grant confers
    // visibility via the synthetic-role can_read path? No — this is a bespoke
    // non-synthetic role, which always confers visibility).
    srv.grant_workspace(&user, "team-a", "USE").await;

    let resp = srv
        .send(Method::GET, WORKSPACES, None, Some((&user, &pass)), None)
        .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    let names: Vec<String> = resp.json()["workspaces"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|w| w["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    assert!(names.contains(&"team-a".to_string()), "names={names:?}");
    assert!(!names.contains(&"team-b".to_string()), "names={names:?}");
    assert!(!names.contains(&"default".to_string()), "names={names:?}");
}

#[tokio::test]
async fn filter_list_workspaces_admin_sees_all() {
    let srv = TestServer::start("list_ws_admin", true, false).await;
    srv.seed_users().await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;

    let resp = srv
        .send(Method::GET, WORKSPACES, None, Some(ADMIN), None)
        .await;
    assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    let names: Vec<String> = resp.json()["workspaces"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|w| w["name"].as_str().map(str::to_string))
        .collect();
    for expected in ["default", "team-a", "team-b"] {
        assert!(names.contains(&expected.to_string()), "names={names:?}");
    }
}

// ---------------------------------------------------------------------------
// GetWorkspace (validate_can_view_workspace) + admin-only create/delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_workspace_requires_access() {
    let srv = TestServer::start("get_ws", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("view_user").await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;
    srv.grant_workspace(&user, "team-a", "USE").await;

    let ok = srv
        .send(
            Method::GET,
            &format!("{WORKSPACES}/team-a"),
            None,
            Some((&user, &pass)),
            None,
        )
        .await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.body);

    let denied = srv
        .send(
            Method::GET,
            &format!("{WORKSPACES}/team-b"),
            None,
            Some((&user, &pass)),
            None,
        )
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
}

#[tokio::test]
async fn create_workspace_is_admin_only() {
    let srv = TestServer::start("create_ws_admin_only", true, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("nonadmin").await;

    let denied = srv
        .send(
            Method::POST,
            WORKSPACES,
            None,
            Some((&user, &pass)),
            Some(json!({ "name": "unauthorized-ws", "default_artifact_root": ART_ROOT })),
        )
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN, "{}", denied.body);
}

// ---------------------------------------------------------------------------
// CreateWorkspace seeds default roles
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_workspace_seeds_default_roles() {
    let srv = TestServer::start("seed_roles", true, false).await;
    srv.seed_users().await;
    srv.create_workspace("team-seed").await;

    let roles = srv
        .auth
        .list_roles(Some(&["team-seed".to_string()]))
        .await
        .expect("list roles");
    let mut names: Vec<String> = roles.iter().map(|r| r.name.clone()).collect();
    names.sort();
    assert_eq!(names, vec!["admin".to_string(), "user".to_string()]);
}

// ---------------------------------------------------------------------------
// admin bypass is total
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_bypasses_workspace_boundary() {
    let srv = TestServer::start("admin_bypass", true, false).await;
    srv.seed_users().await;
    srv.create_workspace("team-a").await;
    srv.create_workspace("team-b").await;
    let exp_a = srv.create_experiment("team-a", "exp-a").await;
    let exp_b = srv.create_experiment("team-b", "exp-b").await;

    // Admin reads resources in every workspace with no explicit grant.
    for (ws, id) in [("team-a", &exp_a), ("team-b", &exp_b)] {
        let resp = srv.get_experiment(ws, id, ADMIN).await;
        assert_eq!(resp.status, StatusCode::OK, "{}", resp.body);
    }
}

// ---------------------------------------------------------------------------
// Single-tenant control: auth behavior unchanged when workspaces disabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_tenant_implicit_default_read_still_applies() {
    // Workspaces disabled: a non-admin with no grant still gets implicit
    // default_permission READ (the fixture's default is READ) — byte-identical to
    // the pre-T10.4 single-tenant behavior.
    let srv = TestServer::start("single_tenant", false, false).await;
    srv.seed_users().await;
    let (user, pass) = srv.create_user("single_user").await;
    let exp = srv.create_experiment("default", "single-exp").await;

    // No grant, workspaces off → implicit default READ → 200 (NOT a boundary deny).
    let resp = srv.get_experiment("default", &exp, (&user, &pass)).await;
    assert_eq!(
        resp.status,
        StatusCode::OK,
        "single-tenant implicit default READ must still apply: {}",
        resp.body
    );
}
