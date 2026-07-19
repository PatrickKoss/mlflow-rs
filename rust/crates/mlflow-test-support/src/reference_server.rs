//! Lifecycle support for Python reference servers used by cross-server tests.
//!
//! Every server is put in a new session so normal teardown can signal the
//! entire `uv`/Python process group. On Linux, the direct child also receives
//! `SIGTERM` through `PR_SET_PDEATHSIG` if the test binary dies. That signal
//! applies only to the direct child: an intermediary such as `uv` can die
//! while leaving descendants alive, so every start first runs the conservative
//! stale-process reaper exposed by [`reap_stale_reference_servers`].
//!
//! New processes carry an owner PID and owner start time in their environment.
//! The reaper only removes a tagged, exact-signature uvicorn when that same-UID
//! owner no longer exists. For servers created before tagging was introduced,
//! it requires the exact reference-server command line and adoption by PID 1.
//! WSL uses a root-owned `/init` relay as its namespace reaper, so that exact
//! parent identity is treated as PID 1 too. A server with any ordinary live
//! parent—including `dev/run_dev_server.py`—is never considered legacy-stale.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Inherited tag identifying the test binary that owns a reference server.
pub const OWNER_PID_ENV: &str = "MLFLOW_TEST_SUPPORT_OWNER_PID";
/// Owner process start time from `/proc/<pid>/stat`, preventing PID-reuse mistakes.
pub const OWNER_START_TIME_ENV: &str = "MLFLOW_TEST_SUPPORT_OWNER_START_TIME";

const APP: &[u8] = b"mlflow.server.fastapi_app:app";
const TERM_GRACE: Duration = Duration::from_millis(750);
const KILL_GRACE: Duration = Duration::from_secs(2);

/// Child process isolated in a new session and process group.
pub struct ProcessGroupChild {
    child: Child,
    process_group: i32,
}

impl ProcessGroupChild {
    /// Spawn `command` as a session leader and arm direct-child parent-death safety.
    pub fn spawn(mut command: Command) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        let expected_parent = std::process::id() as libc::pid_t;
        // SAFETY: `pre_exec` runs after fork in the child. The closure performs
        // only async-signal-safe libc calls and constructs an error only on a
        // syscall failure, before any other thread can execute in the child.
        unsafe {
            command.pre_exec(move || {
                if libc::setsid() == -1 {
                    return Err(io::Error::last_os_error());
                }
                #[cfg(target_os = "linux")]
                {
                    // Parent-death signaling is an extra safety net, not a
                    // reason to make otherwise-valid process-group spawning
                    // fail on a kernel that rejects this Linux-specific prctl.
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                    // The parent can die between fork and prctl. Close that
                    // race instead of waiting for a later reaper invocation.
                    if libc::getppid() != expected_parent {
                        libc::raise(libc::SIGTERM);
                    }
                }
                Ok(())
            });
        }
        let child = command.spawn()?;
        let process_group = child.id() as i32;
        Ok(Self {
            child,
            process_group,
        })
    }

    /// Mutable access for readiness handshakes and diagnostics.
    pub fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    /// PID of the direct child and process-group leader.
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    fn terminate(&mut self) {
        signal_group(self.process_group, libc::SIGTERM);
        if !wait_until(TERM_GRACE, || {
            let _ = self.child.try_wait();
            !group_exists(self.process_group)
        }) {
            signal_group(self.process_group, libc::SIGKILL);
            let _ = wait_until(KILL_GRACE, || {
                let _ = self.child.try_wait();
                !group_exists(self.process_group)
            });
        }
        let _ = self.child.wait();
    }
}

impl Drop for ProcessGroupChild {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Shared Python reference server used by all cross-server tests.
pub struct ReferenceServer {
    process: ProcessGroupChild,
    port: u16,
}

impl ReferenceServer {
    /// Reap stale servers, choose a loopback port, and spawn the standard
    /// `uv run --frozen python -m uvicorn ...` reference-server command.
    pub fn spawn(repository: &Path, configure: impl FnOnce(&mut Command, u16)) -> io::Result<Self> {
        reap_stale_reference_servers()?;
        let port = free_port()?;
        let mut command = Command::new("uv");
        command
            .args([
                "run",
                "--frozen",
                "python",
                "-m",
                "uvicorn",
                "mlflow.server.fastapi_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--log-level",
                "error",
            ])
            .current_dir(repository)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure(&mut command, port);
        command.env(OWNER_PID_ENV, std::process::id().to_string());
        if let Some(start_time) = process_start_time(std::process::id() as i32) {
            command.env(OWNER_START_TIME_ENV, start_time.to_string());
        }
        Ok(Self {
            process: ProcessGroupChild::spawn(command)?,
            port,
        })
    }

    /// Selected loopback port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// `http://127.0.0.1:<port>` for client requests.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Direct intermediary/session-leader PID, useful in lifecycle diagnostics.
    pub fn child_id(&self) -> u32 {
        self.process.id()
    }
}

/// Summary returned by a stale-reference-server pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReapReport {
    /// Same-UID processes with the exact reference-server command line.
    pub matched: usize,
    /// Exact-match processes selected as stale.
    pub reaped: usize,
    /// Isolated tagged process groups selected as stale.
    pub groups: usize,
    /// Conservative untagged legacy processes selected as stale.
    pub legacy: usize,
}

/// Remove stale same-user Python reference servers from `/proc`.
///
/// This is best-effort with respect to processes racing with the scan: vanished
/// entries and `ESRCH` signals are successful outcomes. Failure to inspect a
/// candidate's environment is conservative and leaves it untouched.
#[cfg(target_os = "linux")]
pub fn reap_stale_reference_servers() -> io::Result<ReapReport> {
    let processes = scan_processes()?;
    let by_pid: HashMap<i32, &ProcessInfo> = processes.iter().map(|p| (p.pid, p)).collect();
    let current_uid = unsafe { libc::geteuid() };
    let mut report = ReapReport::default();
    let mut targets = Vec::new();
    let mut seen_groups = HashSet::new();

    for process in processes
        .iter()
        .filter(|p| p.uid == current_uid && is_reference_server(&p.cmdline))
    {
        report.matched += 1;
        match owner_tag(&process.environ) {
            OwnerTagState::Valid(owner) if !owner_is_alive(owner, process.uid, &by_pid) => {
                if process.process_group > 1
                    && process.session == process.process_group
                    && group_is_same_uid(process.process_group, current_uid, &processes)
                {
                    if seen_groups.insert(process.process_group) {
                        targets.push(ReapTarget::Group(process.process_group));
                        report.groups += 1;
                    }
                } else {
                    targets.push(ReapTarget::Process(process.pid));
                }
                report.reaped += 1;
            }
            OwnerTagState::Absent if legacy_is_orphaned(process, &by_pid) => {
                targets.push(ReapTarget::Process(process.pid));
                report.reaped += 1;
                report.legacy += 1;
            }
            OwnerTagState::Absent
            | OwnerTagState::Valid(_)
            | OwnerTagState::Malformed
            | OwnerTagState::Unreadable => {}
        }
    }

    terminate_targets(&targets);
    Ok(report)
}

/// `/proc` is Linux-specific; other Unix platforms still receive session and
/// process-group teardown but have no stale-server scan.
#[cfg(not(target_os = "linux"))]
pub fn reap_stale_reference_servers() -> io::Result<ReapReport> {
    Ok(ReapReport::default())
}

fn free_port() -> io::Result<u16> {
    Ok(std::net::TcpListener::bind(("127.0.0.1", 0))?
        .local_addr()?
        .port())
}

fn signal_group(process_group: i32, signal: i32) {
    // SAFETY: negative pid targets exactly the isolated process group.
    unsafe {
        libc::kill(-process_group, signal);
    }
}

fn group_exists(process_group: i32) -> bool {
    // SAFETY: signal zero performs existence/permission checking only.
    let result = unsafe { libc::kill(-process_group, 0) };
    result == 0 || io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

fn wait_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while !done() {
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(10));
    }
    true
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct ProcessInfo {
    pid: i32,
    ppid: i32,
    process_group: i32,
    session: i32,
    start_time: u64,
    uid: libc::uid_t,
    cmdline: Vec<Vec<u8>>,
    environ: Environment,
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
enum Environment {
    Read(Vec<u8>),
    Unreadable,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OwnerTag {
    pid: i32,
    start_time: Option<u64>,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerTagState {
    Absent,
    Valid(OwnerTag),
    Malformed,
    Unreadable,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy)]
enum ReapTarget {
    Group(i32),
    Process(i32),
}

#[cfg(target_os = "linux")]
fn scan_processes() -> io::Result<Vec<ProcessInfo>> {
    let mut processes = Vec::new();
    for entry in fs::read_dir("/proc")? {
        let Ok(entry) = entry else { continue };
        let Some(pid) = entry
            .file_name()
            .as_bytes()
            .iter()
            .all(u8::is_ascii_digit)
            .then(|| entry.file_name().to_string_lossy().parse::<i32>().ok())
            .flatten()
        else {
            continue;
        };
        if let Some(process) = read_process(pid) {
            processes.push(process);
        }
    }
    Ok(processes)
}

#[cfg(target_os = "linux")]
fn read_process(pid: i32) -> Option<ProcessInfo> {
    let root = PathBuf::from(format!("/proc/{pid}"));
    let stat = fs::read(root.join("stat")).ok()?;
    let fields = stat_fields(&stat)?;
    let status = fs::read(root.join("status")).ok()?;
    let uid = status_uid(&status)?;
    let cmdline = split_nul(&fs::read(root.join("cmdline")).ok()?);
    let environ = match fs::read(root.join("environ")) {
        Ok(bytes) => Environment::Read(bytes),
        Err(_) => Environment::Unreadable,
    };
    Some(ProcessInfo {
        pid,
        ppid: fields.ppid,
        process_group: fields.process_group,
        session: fields.session,
        start_time: fields.start_time,
        uid,
        cmdline,
        environ,
    })
}

#[cfg(target_os = "linux")]
struct StatFields {
    ppid: i32,
    process_group: i32,
    session: i32,
    start_time: u64,
}

#[cfg(target_os = "linux")]
fn stat_fields(stat: &[u8]) -> Option<StatFields> {
    let command_end = stat.windows(2).rposition(|window| window == b") ")?;
    let tail = std::str::from_utf8(&stat[command_end + 2..]).ok()?;
    let fields: Vec<&str> = tail.split_ascii_whitespace().collect();
    Some(StatFields {
        ppid: fields.get(1)?.parse().ok()?,
        process_group: fields.get(2)?.parse().ok()?,
        session: fields.get(3)?.parse().ok()?,
        start_time: fields.get(19)?.parse().ok()?,
    })
}

#[cfg(target_os = "linux")]
fn status_uid(status: &[u8]) -> Option<libc::uid_t> {
    std::str::from_utf8(status)
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("Uid:\t"))?
        .split_ascii_whitespace()
        .next()?
        .parse()
        .ok()
}

#[cfg(target_os = "linux")]
fn split_nul(bytes: &[u8]) -> Vec<Vec<u8>> {
    bytes
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(<[u8]>::to_vec)
        .collect()
}

#[cfg(target_os = "linux")]
fn is_reference_server(args: &[Vec<u8>]) -> bool {
    args.len() == 10
        && Path::new(OsStr::from_bytes(&args[0]))
            .file_name()
            .is_some_and(|name| name.as_bytes().starts_with(b"python"))
        && args[1] == b"-m"
        && args[2] == b"uvicorn"
        && args[3] == APP
        && args[4] == b"--host"
        && args[5] == b"127.0.0.1"
        && args[6] == b"--port"
        && !args[7].is_empty()
        && args[7].iter().all(u8::is_ascii_digit)
        && args[8] == b"--log-level"
        && args[9] == b"error"
}

#[cfg(target_os = "linux")]
fn owner_tag(environment: &Environment) -> OwnerTagState {
    let Environment::Read(environment) = environment else {
        return OwnerTagState::Unreadable;
    };
    parse_owner_tag(environment)
}

#[cfg(target_os = "linux")]
fn parse_owner_tag(environment: &[u8]) -> OwnerTagState {
    let pid_prefix = format!("{OWNER_PID_ENV}=");
    let start_prefix = format!("{OWNER_START_TIME_ENV}=");
    let mut pid_values = Vec::new();
    let mut start_values = Vec::new();
    for entry in environment.split(|byte| *byte == 0) {
        if let Some(value) = entry.strip_prefix(pid_prefix.as_bytes()) {
            pid_values.push(value);
        }
        if let Some(value) = entry.strip_prefix(start_prefix.as_bytes()) {
            start_values.push(value);
        }
    }
    if pid_values.is_empty() && start_values.is_empty() {
        return OwnerTagState::Absent;
    }
    if pid_values.len() != 1 || start_values.len() > 1 {
        return OwnerTagState::Malformed;
    }
    let Some(pid) = parse_decimal(pid_values[0]).and_then(|pid| i32::try_from(pid).ok()) else {
        return OwnerTagState::Malformed;
    };
    if pid <= 1 {
        return OwnerTagState::Malformed;
    }
    let start_time = if let Some(value) = start_values.first() {
        let Some(value) = parse_decimal(value) else {
            return OwnerTagState::Malformed;
        };
        Some(value)
    } else {
        None
    };
    OwnerTagState::Valid(OwnerTag { pid, start_time })
}

#[cfg(target_os = "linux")]
fn parse_decimal(value: &[u8]) -> Option<u64> {
    (!value.is_empty() && value.iter().all(u8::is_ascii_digit))
        .then(|| std::str::from_utf8(value).ok()?.parse().ok())
        .flatten()
}

#[cfg(target_os = "linux")]
fn owner_is_alive(
    owner: OwnerTag,
    expected_uid: libc::uid_t,
    processes: &HashMap<i32, &ProcessInfo>,
) -> bool {
    processes.get(&owner.pid).is_some_and(|process| {
        process.uid == expected_uid
            && owner
                .start_time
                .is_none_or(|start_time| start_time == process.start_time)
    })
}

#[cfg(target_os = "linux")]
fn legacy_is_orphaned(process: &ProcessInfo, processes: &HashMap<i32, &ProcessInfo>) -> bool {
    is_namespace_reaper(process.ppid, processes)
        || processes
            .get(&process.process_group)
            .is_some_and(|leader| is_namespace_reaper(leader.ppid, processes))
}

#[cfg(target_os = "linux")]
fn is_namespace_reaper(pid: i32, processes: &HashMap<i32, &ProcessInfo>) -> bool {
    if pid == 1 {
        return true;
    }
    processes.get(&pid).is_some_and(|parent| {
        parent.uid == 0 && parent.cmdline.len() == 1 && parent.cmdline[0].as_slice() == b"/init"
    })
}

#[cfg(target_os = "linux")]
fn group_is_same_uid(process_group: i32, uid: libc::uid_t, processes: &[ProcessInfo]) -> bool {
    processes
        .iter()
        .filter(|process| process.process_group == process_group)
        .all(|process| process.uid == uid)
}

#[cfg(target_os = "linux")]
fn terminate_targets(targets: &[ReapTarget]) {
    for target in targets {
        signal_target(*target, libc::SIGTERM);
    }
    if !wait_until(TERM_GRACE, || {
        targets.iter().all(|target| !target_exists(*target))
    }) {
        for target in targets {
            if target_exists(*target) {
                signal_target(*target, libc::SIGKILL);
            }
        }
        let _ = wait_until(KILL_GRACE, || {
            targets.iter().all(|target| !target_exists(*target))
        });
    }
}

#[cfg(target_os = "linux")]
fn signal_target(target: ReapTarget, signal: i32) {
    let pid = match target {
        ReapTarget::Group(process_group) => -process_group,
        ReapTarget::Process(pid) => pid,
    };
    // SAFETY: targets came from same-UID `/proc` entries and were revalidated
    // conservatively before selection. ESRCH races are expected.
    unsafe {
        libc::kill(pid, signal);
    }
}

#[cfg(target_os = "linux")]
fn target_exists(target: ReapTarget) -> bool {
    match target {
        ReapTarget::Group(process_group) => group_exists(process_group),
        ReapTarget::Process(pid) => Path::new(&format!("/proc/{pid}")).exists(),
    }
}

#[cfg(target_os = "linux")]
fn process_start_time(pid: i32) -> Option<u64> {
    let stat = fs::read(format!("/proc/{pid}/stat")).ok()?;
    Some(stat_fields(&stat)?.start_time)
}

#[cfg(not(target_os = "linux"))]
fn process_start_time(_pid: i32) -> Option<u64> {
    None
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    fn environment(value: &[u8]) -> Environment {
        Environment::Read(value.to_vec())
    }

    fn process(pid: i32, ppid: i32, uid: libc::uid_t, cmdline: Vec<Vec<u8>>) -> ProcessInfo {
        ProcessInfo {
            pid,
            ppid,
            process_group: pid,
            session: pid,
            start_time: 100,
            uid,
            cmdline,
            environ: environment(&[]),
        }
    }

    #[test]
    fn parses_owner_pid_and_optional_start_time() {
        assert_eq!(
            parse_owner_tag(b"A=1\0MLFLOW_TEST_SUPPORT_OWNER_PID=42\0"),
            OwnerTagState::Valid(OwnerTag {
                pid: 42,
                start_time: None,
            })
        );
        assert_eq!(
            parse_owner_tag(
                b"MLFLOW_TEST_SUPPORT_OWNER_PID=42\0MLFLOW_TEST_SUPPORT_OWNER_START_TIME=99\0"
            ),
            OwnerTagState::Valid(OwnerTag {
                pid: 42,
                start_time: Some(99),
            })
        );
        assert_eq!(parse_owner_tag(b"A=1\0"), OwnerTagState::Absent);
    }

    #[test]
    fn malformed_owner_tags_are_never_treated_as_legacy() {
        for environment in [
            b"MLFLOW_TEST_SUPPORT_OWNER_PID=nope\0".as_slice(),
            b"MLFLOW_TEST_SUPPORT_OWNER_PID=1\0".as_slice(),
            b"MLFLOW_TEST_SUPPORT_OWNER_START_TIME=99\0".as_slice(),
            b"MLFLOW_TEST_SUPPORT_OWNER_PID=42\0MLFLOW_TEST_SUPPORT_OWNER_PID=43\0".as_slice(),
        ] {
            assert_eq!(parse_owner_tag(environment), OwnerTagState::Malformed);
        }
    }

    #[test]
    fn owner_liveness_checks_uid_and_start_time() {
        let owner = process(42, 1, 1000, vec![b"test-binary".to_vec()]);
        let processes = HashMap::from([(42, &owner)]);
        assert!(owner_is_alive(
            OwnerTag {
                pid: 42,
                start_time: Some(100),
            },
            1000,
            &processes
        ));
        assert!(!owner_is_alive(
            OwnerTag {
                pid: 42,
                start_time: Some(101),
            },
            1000,
            &processes
        ));
        assert!(!owner_is_alive(
            OwnerTag {
                pid: 42,
                start_time: Some(100),
            },
            1001,
            &processes
        ));
    }

    #[test]
    fn legacy_orphan_requires_pid_one_or_exact_wsl_reaper() {
        let init = process(1, 0, 0, vec![b"/sbin/init".to_vec()]);
        let wsl = process(3005, 3004, 0, vec![b"/init".to_vec()]);
        let live_test = process(50, 1, 1000, vec![b"test-binary".to_vec()]);
        let direct_pid_one = process(60, 1, 1000, vec![]);
        let direct_wsl = process(61, 3005, 1000, vec![]);
        let live = process(62, 50, 1000, vec![]);
        let processes = HashMap::from([(1, &init), (3005, &wsl), (50, &live_test)]);
        assert!(legacy_is_orphaned(&direct_pid_one, &processes));
        assert!(legacy_is_orphaned(&direct_wsl, &processes));
        assert!(!legacy_is_orphaned(&live, &processes));
    }

    #[test]
    fn reference_signature_is_exact() {
        let args = split_nul(
            b"/venv/bin/python3\0-m\0uvicorn\0mlflow.server.fastapi_app:app\0--host\x00127.0.0.1\0--port\x004567\0--log-level\0error\0",
        );
        assert!(is_reference_server(&args));
        let mut dev = args.clone();
        dev.push(b"--reload".to_vec());
        assert!(!is_reference_server(&dev));
    }
}
