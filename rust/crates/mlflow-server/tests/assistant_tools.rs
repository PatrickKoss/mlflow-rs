use std::os::unix::fs::symlink;

use mlflow_server::assistant_providers::PermissionsConfig;
use mlflow_server::assistant_tools::{execute_tool, static_permission_error};
use serde_json::json;
use tempfile::TempDir;

fn restricted() -> PermissionsConfig {
    PermissionsConfig::default()
}

#[tokio::test]
async fn file_tools_support_safe_read_write_and_edit() {
    let sandbox = TempDir::new().unwrap();
    let root = sandbox.path();
    let write = execute_tool(
        "Write",
        &json!({"file_path":"nested/a.txt","content":"alpha"}),
        Some(root),
        None,
        &restricted(),
    )
    .await;
    assert_eq!(
        (write.content.as_str(), write.is_error),
        ("Wrote 5 bytes to nested/a.txt", false)
    );

    let edit = execute_tool(
        "Edit",
        &json!({"file_path":"nested/a.txt","old_string":"alpha","new_string":"beta"}),
        Some(root),
        None,
        &restricted(),
    )
    .await;
    assert_eq!(
        (edit.content.as_str(), edit.is_error),
        ("Edited nested/a.txt", false)
    );

    let read = execute_tool(
        "Read",
        &json!({"file_path":"nested/a.txt"}),
        Some(root),
        None,
        &restricted(),
    )
    .await;
    assert_eq!((read.content.as_str(), read.is_error), ("beta", false));
}

#[tokio::test]
async fn traversal_and_absolute_escape_matrix_is_all_negative() {
    let fixture = TempDir::new().unwrap();
    let root = fixture.path().join("workspace");
    std::fs::create_dir(&root).unwrap();
    let outside = fixture.path().join("outside.txt");
    std::fs::write(&outside, "sentinel").unwrap();
    let cases = [
        "../outside.txt".to_string(),
        "sub/../../outside.txt".to_string(),
        "./../outside.txt".to_string(),
        "sub/../../../outside.txt".to_string(),
        outside.to_string_lossy().into_owned(),
        format!("{}/../outside.txt", root.display()),
    ];
    for path in cases {
        for (tool, input) in [
            ("Read", json!({"file_path":path})),
            ("Write", json!({"file_path":path,"content":"escaped"})),
            (
                "Edit",
                json!({"file_path":path,"old_string":"sentinel","new_string":"escaped"}),
            ),
        ] {
            let result = execute_tool(tool, &input, Some(&root), None, &restricted()).await;
            assert!(result.is_error, "{tool} unexpectedly accepted {path}");
            assert!(result.content.starts_with("Permission denied: path"));
        }
    }
    assert_eq!(std::fs::read_to_string(outside).unwrap(), "sentinel");
}

#[tokio::test]
async fn precreated_symlink_escape_and_write_through_symlink_are_rejected() {
    let fixture = TempDir::new().unwrap();
    let root = fixture.path().join("workspace");
    let outside_dir = fixture.path().join("outside");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&outside_dir).unwrap();
    let outside = outside_dir.join("target.txt");
    std::fs::write(&outside, "sentinel").unwrap();
    symlink(&outside, root.join("file-link")).unwrap();
    symlink(&outside_dir, root.join("dir-link")).unwrap();

    for input in [
        json!({"file_path":"file-link","content":"escaped"}),
        json!({"file_path":"dir-link/target.txt","content":"escaped"}),
    ] {
        let result = execute_tool("Write", &input, Some(&root), None, &restricted()).await;
        assert!(result.is_error);
        assert!(result.content.starts_with("Permission denied: path"));
    }
    assert_eq!(std::fs::read_to_string(outside).unwrap(), "sentinel");
}

#[tokio::test]
async fn relink_and_rename_swap_states_are_rejected_without_touching_outside() {
    let fixture = TempDir::new().unwrap();
    let root = fixture.path().join("workspace");
    let outside_dir = fixture.path().join("outside");
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(&outside_dir).unwrap();
    let outside = outside_dir.join("target.txt");
    std::fs::write(&outside, "sentinel").unwrap();

    let swappable = root.join("swappable");
    std::fs::create_dir(&swappable).unwrap();
    std::fs::write(swappable.join("target.txt"), "inside").unwrap();
    std::fs::rename(&swappable, root.join("old-swappable")).unwrap();
    symlink(&outside_dir, &swappable).unwrap();
    let result = execute_tool(
        "Write",
        &json!({"file_path":"swappable/target.txt","content":"escaped"}),
        Some(&root),
        None,
        &restricted(),
    )
    .await;
    assert!(result.is_error);

    std::fs::remove_file(&swappable).unwrap();
    std::fs::rename(root.join("old-swappable"), &swappable).unwrap();
    std::fs::rename(swappable.join("target.txt"), swappable.join("inside.txt")).unwrap();
    symlink(&outside, swappable.join("target.txt")).unwrap();
    let result = execute_tool(
        "Edit",
        &json!({"file_path":"swappable/target.txt","old_string":"sentinel","new_string":"escaped"}),
        Some(&root),
        None,
        &restricted(),
    )
    .await;
    assert!(result.is_error);
    assert_eq!(std::fs::read_to_string(outside).unwrap(), "sentinel");
}

#[test]
fn tilde_and_bash_expansion_escape_matrix_is_all_negative() {
    let fixture = TempDir::new().unwrap();
    let root = fixture.path().join("workspace");
    std::fs::create_dir(&root).unwrap();
    let permissions = restricted();
    assert!(static_permission_error(
        "Write",
        &json!({"file_path":"~/.mlflow-escape","content":"x"}),
        &permissions,
        Some(&root),
    )
    .is_some());

    let commands = [
        "$(printf mlflow) --version",
        "`printf mlflow` --version",
        "$TOOL --version",
        "TOOL=mlflow $TOOL --version",
        "mlflow --version > ../outside.txt",
        "mlflow --version && python -c 'open(\"../outside\", \"w\")'",
        "python -c 'open(\"../outside\", \"w\").write(\"x\")'",
        "python -c 'import os; os.system(\"cat /etc/passwd\")'",
        "python ../outside.py",
        "python /tmp/outside.py",
    ];
    for command in commands {
        let denial = static_permission_error(
            "Bash",
            &json!({"command":command}),
            &permissions,
            Some(&root),
        );
        assert!(denial.is_some(), "unexpectedly accepted {command}");
    }
}

#[tokio::test]
async fn bash_output_is_capped() {
    let result = execute_tool(
        "Bash",
        &json!({"command":"python -c 'print(\"x\" * 2000000)'"}),
        None,
        None,
        &restricted(),
    )
    .await;
    assert!(result.content.len() <= 1024 * 1024);
}
