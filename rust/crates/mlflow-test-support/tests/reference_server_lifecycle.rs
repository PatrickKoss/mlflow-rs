#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlflow_test_support::reference_server::{reap_stale_reference_servers, ProcessGroupChild};

const MODE_ENV: &str = "MLFLOW_TEST_SUPPORT_HELPER_MODE";
const WAIT_TIMEOUT: Duration = Duration::from_secs(5);

struct OwnerTree {
    owner: Child,
    candidate: i32,
    grandchild: i32,
    intermediary: i32,
}

impl OwnerTree {
    fn spawn() -> Self {
        let mut owner = Command::new(env!("CARGO_BIN_EXE_reference-server-test-helper"))
            .env(MODE_ENV, "owner")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn owner helper");
        let stdout = owner.stdout.take().expect("owner stdout");
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .expect("read process-tree readiness");
        let ids: Vec<i32> = line
            .split_ascii_whitespace()
            .skip(1)
            .map(|value| value.parse().expect("numeric helper pid"))
            .collect();
        assert_eq!(ids.len(), 3, "unexpected helper output: {line:?}");
        Self {
            owner,
            candidate: ids[0],
            grandchild: ids[1],
            intermediary: ids[2],
        }
    }

    fn kill_owner(&mut self) {
        signal(self.owner.id() as i32, libc::SIGKILL);
        self.owner.wait().expect("reap owner helper");
        wait_gone(self.intermediary);
    }
}

impl Drop for OwnerTree {
    fn drop(&mut self) {
        let _ = self.owner.kill();
        let _ = self.owner.wait();
        let _ = reap_stale_reference_servers();
    }
}

#[test]
fn process_group_drop_terminates_the_entire_tree() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_reference-server-test-helper"));
    command
        .env(MODE_ENV, "tree")
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut group = ProcessGroupChild::spawn(command).expect("spawn isolated process tree");
    let stdout = group.child_mut().stdout.take().expect("tree stdout");
    let mut line = String::new();
    BufReader::new(stdout)
        .read_line(&mut line)
        .expect("read tree readiness");
    let ids: Vec<i32> = line
        .split_ascii_whitespace()
        .skip(1)
        .map(|value| value.parse().expect("numeric helper pid"))
        .collect();
    assert_eq!(ids.len(), 2, "unexpected helper output: {line:?}");
    drop(group);
    wait_gone(ids[0]);
    wait_gone(ids[1]);
}

#[test]
fn sigkill_owner_then_reaper_removes_intermediary_process_group() {
    let mut tree = OwnerTree::spawn();
    tree.kill_owner();
    assert!(process_exists(tree.candidate));
    assert!(process_exists(tree.grandchild));

    let report = reap_stale_reference_servers().expect("reap dead-owner server");
    assert!(report.reaped >= 1, "unexpected report: {report:?}");
    assert!(report.groups >= 1, "unexpected report: {report:?}");
    wait_gone(tree.candidate);
    wait_gone(tree.grandchild);
}

#[test]
fn concurrent_live_owners_are_preserved() {
    let mut dead_owner = OwnerTree::spawn();
    let mut live_owner = OwnerTree::spawn();
    dead_owner.kill_owner();

    reap_stale_reference_servers().expect("reap only dead owner");
    wait_gone(dead_owner.candidate);
    wait_gone(dead_owner.grandchild);
    assert!(process_exists(live_owner.owner.id() as i32));
    assert!(process_exists(live_owner.intermediary));
    assert!(process_exists(live_owner.candidate));
    assert!(process_exists(live_owner.grandchild));

    live_owner.kill_owner();
    reap_stale_reference_servers().expect("clean second owner");
    wait_gone(live_owner.candidate);
    wait_gone(live_owner.grandchild);
}

fn signal(pid: i32, signal: i32) {
    // SAFETY: test PIDs come directly from children created by this process.
    let result = unsafe { libc::kill(pid, signal) };
    assert_eq!(result, 0, "signal {signal} to {pid} failed");
}

fn process_exists(pid: i32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

fn wait_gone(pid: i32) {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    while process_exists(pid) {
        assert!(Instant::now() < deadline, "process {pid} did not exit");
        thread::sleep(Duration::from_millis(10));
    }
}
