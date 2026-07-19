//! Sandboxed tools used by OpenAI-compatible Assistant providers.
//!
//! File descriptors are opened relative to a canonical workspace descriptor
//! with Linux `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS|RESOLVE_NO_MAGICLINKS)`.
//! The descriptor-relative open is the enforcement point: the earlier
//! canonicalization is only for Python-compatible policy messages and cannot
//! be raced into following a replacement symlink.

use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::assistant_providers::PermissionsConfig;

const ALLOWED_BASH_COMMANDS: &[&str] = &["mlflow", "python", "python3"];
const BASH_TIMEOUT: Duration = Duration::from_secs(120);
const OUTPUT_CAP: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

pub fn static_permission_error(
    tool_name: &str,
    tool_input: &Value,
    permissions: &PermissionsConfig,
    cwd: Option<&Path>,
) -> Option<String> {
    if permissions.full_access {
        return None;
    }
    if tool_name == "Bash" {
        let command = tool_input
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if let Err(error) = validate_restricted_bash(command, cwd) {
            return Some(error);
        }
    }
    if matches!(tool_name, "Read" | "Write" | "Edit") && !permissions.allow_edit_files {
        return Some(format!("Permission denied: {tool_name} is not allowed"));
    }
    if matches!(tool_name, "Write" | "Edit") && cwd.is_none() {
        return Some(format!(
            "Permission denied: {tool_name} requires a configured project directory"
        ));
    }
    if matches!(tool_name, "Read" | "Write" | "Edit") {
        if let (Some(root), Some(raw_path)) = (cwd, file_path(tool_input)) {
            if resolve_workspace_relative(raw_path, root).is_err() {
                return Some(format!(
                    "Permission denied: path {raw_path} is outside the workspace {}",
                    root.display()
                ));
            }
        }
    }
    None
}

pub async fn execute_tool(
    tool_name: &str,
    tool_input: &Value,
    cwd: Option<&Path>,
    tracking_uri: Option<&str>,
    permissions: &PermissionsConfig,
) -> ToolResult {
    if let Some(error) = static_permission_error(tool_name, tool_input, permissions, cwd) {
        return ToolResult::error(error);
    }
    let result = match tool_name {
        "Bash" => execute_bash(tool_input, cwd, tracking_uri).await,
        "Read" if permissions.full_access || cwd.is_none() => {
            execute_file_unconfined(tool_input, cwd, FileOperation::Read)
        }
        "Write" if permissions.full_access => {
            execute_file_unconfined(tool_input, cwd, FileOperation::Write)
        }
        "Edit" if permissions.full_access => {
            execute_file_unconfined(tool_input, cwd, FileOperation::Edit)
        }
        "Read" => execute_file(tool_input, cwd, FileOperation::Read),
        "Write" => execute_file(tool_input, cwd, FileOperation::Write),
        "Edit" => execute_file(tool_input, cwd, FileOperation::Edit),
        _ => return ToolResult::error(format!("Unknown tool: {tool_name}")),
    };
    match result {
        Ok(result) => result,
        Err(error) => ToolResult::error(format!("Tool execution failed: {error}")),
    }
}

async fn execute_bash(
    input: &Value,
    cwd: Option<&Path>,
    tracking_uri: Option<&str>,
) -> std::io::Result<ToolResult> {
    let command = input.get("command").and_then(Value::as_str).unwrap_or("");
    if command.is_empty() {
        return Ok(ToolResult::error("No command provided"));
    }
    let mut child = Command::new("/bin/sh");
    child
        .arg("-c")
        .arg(command)
        .current_dir(cwd.unwrap_or_else(|| Path::new(".")))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(tracking_uri) = tracking_uri {
        child.env("MLFLOW_TRACKING_URI", tracking_uri);
    }
    let mut child = child.spawn()?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let stdout_task = tokio::spawn(read_capped(stdout));
    let stderr_task = tokio::spawn(read_capped(stderr));
    let status = match timeout(BASH_TIMEOUT, child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            let _ = child.kill().await;
            return Ok(ToolResult::error("Command timed out after 120 seconds"));
        }
    };
    let stdout = stdout_task.await.map_err(std::io::Error::other)??;
    let stderr = stderr_task.await.map_err(std::io::Error::other)??;
    let mut combined = stdout;
    combined.extend_from_slice(&stderr);
    combined.truncate(OUTPUT_CAP as usize);
    let output = String::from_utf8_lossy(&combined).trim().to_string();
    if status.success() {
        Ok(ToolResult::ok(if output.is_empty() {
            "(no output)".to_string()
        } else {
            output
        }))
    } else {
        Ok(ToolResult::error(if output.is_empty() {
            format!("Exit code: {}", status.code().unwrap_or(-1))
        } else {
            output
        }))
    }
}

async fn read_capped<R: tokio::io::AsyncRead + Unpin>(reader: R) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.take(OUTPUT_CAP).read_to_end(&mut bytes).await?;
    Ok(bytes)
}

fn validate_restricted_bash(command: &str, cwd: Option<&Path>) -> Result<(), String> {
    let argv =
        shlex::split(command).ok_or_else(|| "Permission denied: malformed command".to_string())?;
    if argv.is_empty() {
        return Err(allowed_commands_error());
    }
    if command.contains(['$', '`', '~', '\n', '\r', '<', '>']) {
        return Err("Permission denied: shell expansion or redirection is not allowed".to_string());
    }
    if command.contains([';', '&', '|']) {
        return Err("Permission denied: command chaining is not allowed".to_string());
    }
    if argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "|" | "||" | "&&" | ";" | "&"))
    {
        return Err("Permission denied: command chaining is not allowed".to_string());
    }
    if !ALLOWED_BASH_COMMANDS.contains(&argv[0].as_str()) || argv[0].contains('=') {
        return Err(allowed_commands_error());
    }
    if matches!(argv[0].as_str(), "python" | "python3") {
        if let Some(index) = argv.iter().position(|arg| arg == "-c") {
            let script = argv.get(index + 1).map(String::as_str).unwrap_or("");
            let dangerous = [
                "open(",
                "pathlib",
                "os.",
                "subprocess",
                "shutil",
                "socket",
                "__import__",
                "eval(",
                "exec(",
            ];
            if dangerous.iter().any(|needle| script.contains(needle)) {
                return Err(
                    "Permission denied: Python code may access resources outside the workspace"
                        .to_string(),
                );
            }
        }
    }
    if let Some(root) = cwd {
        for arg in argv.iter().skip(1).filter(|arg| !arg.starts_with('-')) {
            if arg.contains('\\') || arg.contains("file://") {
                return Err("Permission denied: ambiguous path syntax is not allowed".to_string());
            }
            if looks_like_path(arg) && resolve_workspace_relative(arg, root).is_err() {
                return Err(format!(
                    "Permission denied: path {arg} is outside the workspace {}",
                    root.display()
                ));
            }
        }
    }
    Ok(())
}

fn allowed_commands_error() -> String {
    "Permission denied: only mlflow, python, python3 commands are allowed".to_string()
}

fn looks_like_path(value: &str) -> bool {
    value.starts_with('/') || value.starts_with('.') || value.contains('/') || value.contains("\\")
}

fn file_path(input: &Value) -> Option<&str> {
    input
        .get("file_path")
        .or_else(|| input.get("path"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

#[derive(Clone, Copy)]
enum FileOperation {
    Read,
    Write,
    Edit,
}

fn execute_file(
    input: &Value,
    cwd: Option<&Path>,
    operation: FileOperation,
) -> std::io::Result<ToolResult> {
    let Some(raw_path) = file_path(input) else {
        return Ok(ToolResult::error("No file_path provided"));
    };
    let root = cwd.unwrap_or_else(|| Path::new("/"));
    let relative = resolve_workspace_relative(raw_path, root)?;
    let flags = match operation {
        FileOperation::Read | FileOperation::Edit => libc::O_RDONLY,
        FileOperation::Write => libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
    };
    if matches!(operation, FileOperation::Write) {
        secure_create_parents(root, relative.parent().unwrap_or_else(|| Path::new("")))?;
    }
    let mode = if flags & libc::O_CREAT != 0 { 0o666 } else { 0 };
    let mut file = secure_open(root, &relative, flags, mode)?;
    match operation {
        FileOperation::Read => {
            let mut content = String::new();
            file.read_to_string(&mut content)?;
            Ok(ToolResult::ok(content))
        }
        FileOperation::Write => {
            let content = input.get("content").and_then(Value::as_str).unwrap_or("");
            file.write_all(content.as_bytes())?;
            Ok(ToolResult::ok(format!(
                "Wrote {} bytes to {raw_path}",
                content.len()
            )))
        }
        FileOperation::Edit => {
            let mut content = String::new();
            file.read_to_string(&mut content)?;
            let old = input
                .get("old_string")
                .and_then(Value::as_str)
                .unwrap_or("");
            let new = input
                .get("new_string")
                .and_then(Value::as_str)
                .unwrap_or("");
            let Some(index) = content.find(old) else {
                return Ok(ToolResult::error(format!(
                    "old_string not found in {raw_path}"
                )));
            };
            content.replace_range(index..index + old.len(), new);
            drop(file);
            let mut file = secure_open(root, &relative, libc::O_WRONLY | libc::O_TRUNC, 0)?;
            file.write_all(content.as_bytes())?;
            Ok(ToolResult::ok(format!("Edited {raw_path}")))
        }
    }
}

fn execute_file_unconfined(
    input: &Value,
    cwd: Option<&Path>,
    operation: FileOperation,
) -> std::io::Result<ToolResult> {
    let Some(raw_path) = file_path(input) else {
        return Ok(ToolResult::error("No file_path provided"));
    };
    let expanded = expand_user(raw_path);
    let path = if expanded.is_absolute() {
        expanded
    } else {
        cwd.unwrap_or_else(|| Path::new(".")).join(expanded)
    };
    match operation {
        FileOperation::Read => Ok(ToolResult::ok(std::fs::read_to_string(path)?)),
        FileOperation::Write => {
            let content = input.get("content").and_then(Value::as_str).unwrap_or("");
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, content)?;
            Ok(ToolResult::ok(format!(
                "Wrote {} bytes to {raw_path}",
                content.len()
            )))
        }
        FileOperation::Edit => {
            let mut content = std::fs::read_to_string(&path)?;
            let old = input
                .get("old_string")
                .and_then(Value::as_str)
                .unwrap_or("");
            let new = input
                .get("new_string")
                .and_then(Value::as_str)
                .unwrap_or("");
            let Some(index) = content.find(old) else {
                return Ok(ToolResult::error(format!(
                    "old_string not found in {raw_path}"
                )));
            };
            content.replace_range(index..index + old.len(), new);
            std::fs::write(path, content)?;
            Ok(ToolResult::ok(format!("Edited {raw_path}")))
        }
    }
}

fn expand_user(raw: &str) -> PathBuf {
    if raw == "~" || raw.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(raw.trim_start_matches("~/"));
        }
    }
    PathBuf::from(raw)
}

fn resolve_workspace_relative(raw: &str, root: &Path) -> std::io::Result<PathBuf> {
    if raw.starts_with('~') && raw != "~" && !raw.starts_with("~/") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "unsupported user-home expansion",
        ));
    }
    let root = root.canonicalize()?;
    let expanded = expand_user(raw);
    let candidate = if expanded.is_absolute() {
        expanded
    } else {
        root.join(expanded)
    };
    let normalized = normalize_lexically(&candidate)?;
    let mut existing = normalized.as_path();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let name = existing.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "path escapes workspace",
            )
        })?;
        suffix.push(name.to_os_string());
        existing = existing.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "path escapes workspace",
            )
        })?;
    }
    let mut resolved = existing.canonicalize()?;
    for name in suffix.into_iter().rev() {
        resolved.push(name);
    }
    resolved
        .strip_prefix(&root)
        .map(Path::to_path_buf)
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "path escapes workspace",
            )
        })
}

fn normalize_lexically(path: &Path) -> std::io::Result<PathBuf> {
    let mut output = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                output.push(component)
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !output.pop() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "path escapes workspace",
                    ));
                }
            }
        }
    }
    Ok(output)
}

#[cfg(target_os = "linux")]
fn secure_open(root: &Path, relative: &Path, flags: i32, mode: u32) -> std::io::Result<File> {
    #[repr(C)]
    struct OpenHow {
        flags: u64,
        mode: u64,
        resolve: u64,
    }
    const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
    const RESOLVE_NO_SYMLINKS: u64 = 0x04;
    const RESOLVE_BENEATH: u64 = 0x08;
    let root = File::open(root)?;
    let relative =
        CString::new(relative.as_os_str().as_encoded_bytes()).map_err(std::io::Error::other)?;
    let how = OpenHow {
        flags: (flags | libc::O_CLOEXEC) as u64,
        mode: mode as u64,
        resolve: RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS,
    };
    // SAFETY: all pointers reference live values for the duration of the
    // syscall; the returned descriptor is owned exactly once by `File`.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            root.as_raw_fd(),
            relative.as_ptr(),
            &how,
            std::mem::size_of::<OpenHow>(),
        ) as i32
    };
    if fd < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: `fd` is a fresh successful openat2 result.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}

#[cfg(target_os = "linux")]
fn secure_create_parents(root: &Path, relative: &Path) -> std::io::Result<()> {
    let mut descriptors = vec![File::open(root)?];
    for component in relative.components() {
        let Component::Normal(name) = component else {
            continue;
        };
        let name = CString::new(name.as_encoded_bytes()).map_err(std::io::Error::other)?;
        let parent = descriptors.last().expect("root descriptor").as_raw_fd();
        // SAFETY: `parent` and `name` are live descriptors/strings. EEXIST is
        // expected; the following O_NOFOLLOW directory open validates it.
        let mkdir = unsafe { libc::mkdirat(parent, name.as_ptr(), 0o777) };
        if mkdir != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(error);
            }
        }
        // SAFETY: openat returns a new descriptor owned once below.
        let fd = unsafe {
            libc::openat(
                parent,
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `fd` is a fresh successful openat result.
        descriptors.push(unsafe { File::from_raw_fd(fd) });
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn secure_open(_root: &Path, _relative: &Path, _flags: i32, _mode: u32) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "secure assistant file tools require Linux openat2",
    ))
}

#[cfg(not(target_os = "linux"))]
fn secure_create_parents(_root: &Path, _relative: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "secure assistant file tools require Linux descriptor-relative opens",
    ))
}

pub fn tools_schema() -> Value {
    json!([
        {"type":"function","function":{"name":"Bash","description":"Execute a shell command to query or interact with MLflow. Use 'mlflow' CLI commands or Python one-liners with the MLflow SDK.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute."}},"required":["command"]}}},
        {"type":"function","function":{"name":"Read","description":"Read the contents of a file.","parameters":{"type":"object","properties":{"file_path":{"type":"string","description":"Absolute or relative path to the file."}},"required":["file_path"]}}},
        {"type":"function","function":{"name":"Write","description":"Write content to a file (creates or overwrites).","parameters":{"type":"object","properties":{"file_path":{"type":"string","description":"Absolute or relative path to the file."},"content":{"type":"string","description":"Content to write."}},"required":["file_path","content"]}}},
        {"type":"function","function":{"name":"Edit","description":"Replace the first occurrence of old_string with new_string in a file.","parameters":{"type":"object","properties":{"file_path":{"type":"string","description":"Absolute or relative path to the file."},"old_string":{"type":"string","description":"Exact string to find."},"new_string":{"type":"string","description":"String to replace it with."}},"required":["file_path","old_string","new_string"]}}}
    ])
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::os::unix::fs::symlink;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn descriptor_open_rejects_directory_relink_after_policy_resolution() {
        let fixture = TempDir::new().unwrap();
        let root = fixture.path().join("workspace");
        let outside = fixture.path().join("outside");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("target"), "sentinel").unwrap();
        std::fs::create_dir(root.join("slot")).unwrap();
        std::fs::write(root.join("slot/target"), "inside").unwrap();

        let resolved = resolve_workspace_relative("slot/target", &root).unwrap();
        std::fs::rename(root.join("slot"), root.join("old-slot")).unwrap();
        symlink(&outside, root.join("slot")).unwrap();

        assert!(secure_open(&root, &resolved, libc::O_WRONLY | libc::O_TRUNC, 0).is_err());
        assert_eq!(
            std::fs::read_to_string(outside.join("target")).unwrap(),
            "sentinel"
        );
    }

    #[test]
    fn descriptor_open_rejects_file_symlink_swap_after_policy_resolution() {
        let fixture = TempDir::new().unwrap();
        let root = fixture.path().join("workspace");
        std::fs::create_dir(&root).unwrap();
        let outside = fixture.path().join("outside");
        std::fs::write(&outside, "sentinel").unwrap();
        std::fs::write(root.join("target"), "inside").unwrap();

        let resolved = resolve_workspace_relative("target", &root).unwrap();
        std::fs::rename(root.join("target"), root.join("old-target")).unwrap();
        symlink(&outside, root.join("target")).unwrap();

        assert!(secure_open(&root, &resolved, libc::O_WRONLY | libc::O_TRUNC, 0).is_err());
        assert_eq!(std::fs::read_to_string(outside).unwrap(), "sentinel");
    }
}
