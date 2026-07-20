//! Permission levels, resource types, and their validation — a byte-faithful
//! port of `mlflow/server/auth/permissions.py`.
//!
//! The five permission levels (`READ < USE < EDIT < MANAGE`, plus
//! `NO_PERMISSIONS`) and the nine resource types (`experiment`,
//! `registered_model`, `prompt`, `scorer`, three gateway types, MCP servers, and the
//! special `workspace` slot) define the RBAC vocabulary. The validators here
//! reproduce Python's error wording verbatim so the HTTP surface returns
//! identical messages.

use mlflow_error::MlflowError;

/// A permission level and its capability flags (`permissions.py:7-14`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permission {
    pub name: &'static str,
    pub can_read: bool,
    pub can_use: bool,
    pub can_update: bool,
    pub can_delete: bool,
    pub can_manage: bool,
}

/// `READ` (`permissions.py:17`).
pub const READ: Permission = Permission {
    name: "READ",
    can_read: true,
    can_use: false,
    can_update: false,
    can_delete: false,
    can_manage: false,
};

/// `USE` (`permissions.py:26`).
pub const USE: Permission = Permission {
    name: "USE",
    can_read: true,
    can_use: true,
    can_update: false,
    can_delete: false,
    can_manage: false,
};

/// `EDIT` (`permissions.py:35`).
pub const EDIT: Permission = Permission {
    name: "EDIT",
    can_read: true,
    can_use: true,
    can_update: true,
    can_delete: false,
    can_manage: false,
};

/// `MANAGE` (`permissions.py:44`).
pub const MANAGE: Permission = Permission {
    name: "MANAGE",
    can_read: true,
    can_use: true,
    can_update: true,
    can_delete: true,
    can_manage: true,
};

/// `NO_PERMISSIONS` (`permissions.py:53`).
pub const NO_PERMISSIONS: Permission = Permission {
    name: "NO_PERMISSIONS",
    can_read: false,
    can_use: false,
    can_update: false,
    can_delete: false,
    can_manage: false,
};

/// `ALL_PERMISSIONS` (`permissions.py:62`), in insertion order so the
/// `Invalid permission ... Valid permissions are: (...)` message matches
/// Python's `tuple(ALL_PERMISSIONS)`.
pub const ALL_PERMISSIONS: [Permission; 5] = [READ, USE, EDIT, MANAGE, NO_PERMISSIONS];

/// Resource-type discriminants (`permissions.py:87-108`).
pub const RESOURCE_TYPE_EXPERIMENT: &str = "experiment";
pub const RESOURCE_TYPE_REGISTERED_MODEL: &str = "registered_model";
pub const RESOURCE_TYPE_PROMPT: &str = "prompt";
pub const RESOURCE_TYPE_SCORER: &str = "scorer";
pub const RESOURCE_TYPE_GATEWAY_SECRET: &str = "gateway_secret";
pub const RESOURCE_TYPE_GATEWAY_ENDPOINT: &str = "gateway_endpoint";
pub const RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION: &str = "gateway_model_definition";
pub const RESOURCE_TYPE_MCP_SERVER: &str = "mcp_server";
pub const RESOURCE_TYPE_WORKSPACE: &str = "workspace";

/// `VALID_RESOURCE_TYPES` (`permissions.py:110`).
pub const VALID_RESOURCE_TYPES: [&str; 9] = [
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_REGISTERED_MODEL,
    RESOURCE_TYPE_PROMPT,
    RESOURCE_TYPE_SCORER,
    RESOURCE_TYPE_GATEWAY_SECRET,
    RESOURCE_TYPE_GATEWAY_ENDPOINT,
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
    RESOURCE_TYPE_MCP_SERVER,
    RESOURCE_TYPE_WORKSPACE,
];

/// `get_permission` (`permissions.py:71`): look up a permission by name. The
/// name must be valid (callers validate first); panics on an unknown name to
/// mirror Python's `KeyError`, which never surfaces on a validated path.
pub fn get_permission(name: &str) -> &'static Permission {
    ALL_PERMISSIONS
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("unknown permission {name:?}"))
}

/// `PERMISSION_PRIORITY` (`permissions.py:75`). Unknown names sort to 0,
/// matching `PERMISSION_PRIORITY.get(a, 0)`.
pub fn permission_priority(name: &str) -> u8 {
    match name {
        "NO_PERMISSIONS" => 0,
        "READ" => 1,
        "USE" => 2,
        "EDIT" => 3,
        "MANAGE" => 4,
        _ => 0,
    }
}

/// `max_permission` (`permissions.py:182`): the higher-priority of two names
/// (ties keep `a`).
pub fn max_permission<'a>(a: &'a str, b: &'a str) -> &'a str {
    if permission_priority(a) >= permission_priority(b) {
        a
    } else {
        b
    }
}

/// `_validate_permission` (`permissions.py:135`).
pub fn validate_permission(permission: &str) -> Result<(), MlflowError> {
    if !ALL_PERMISSIONS.iter().any(|p| p.name == permission) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid permission '{permission}'. Valid permissions are: {}",
            py_tuple(ALL_PERMISSIONS.iter().map(|p| p.name)),
        )));
    }
    Ok(())
}

/// `_validate_resource_type` (`permissions.py:143`). The message lists the
/// types `sorted()`, matching `tuple(sorted(VALID_RESOURCE_TYPES))`.
pub fn validate_resource_type(resource_type: &str) -> Result<(), MlflowError> {
    if !VALID_RESOURCE_TYPES.contains(&resource_type) {
        let mut sorted = VALID_RESOURCE_TYPES;
        sorted.sort_unstable();
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid resource type '{resource_type}'. Valid resource types are: {}",
            py_tuple(sorted.iter().copied()),
        )));
    }
    Ok(())
}

/// Permissions grantable at workspace scope (`permissions.py:126`), sorted.
const WORKSPACE_GRANTABLE: [&str; 2] = ["MANAGE", "USE"];
/// Permissions grantable on a concrete resource (`permissions.py:132`), sorted.
const RESOURCE_GRANTABLE: [&str; 4] = ["EDIT", "MANAGE", "READ", "USE"];

/// `_validate_permission_for_resource_type` (`permissions.py:152`).
pub fn validate_permission_for_resource_type(
    permission: &str,
    resource_type: &str,
) -> Result<(), MlflowError> {
    validate_permission(permission)?;
    validate_resource_type(resource_type)?;
    if resource_type == RESOURCE_TYPE_WORKSPACE {
        if !WORKSPACE_GRANTABLE.contains(&permission) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid permission '{permission}' for resource_type='{RESOURCE_TYPE_WORKSPACE}'. \
                 Workspace-wide grants accept only: {}.",
                py_tuple(WORKSPACE_GRANTABLE.iter().copied()),
            )));
        }
        return Ok(());
    }
    if !RESOURCE_GRANTABLE.contains(&permission) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid permission '{permission}' for resource_type='{resource_type}'. \
             Resource-level grants accept only: {}.",
            py_tuple(RESOURCE_GRANTABLE.iter().copied()),
        )));
    }
    Ok(())
}

/// Render an iterator of strings as a Python tuple repr, e.g.
/// `('READ', 'USE', 'EDIT', 'MANAGE', 'NO_PERMISSIONS')` — including the
/// trailing comma Python emits for a single-element tuple. Used so validation
/// messages byte-match `f"... {tuple(...)}"`.
fn py_tuple<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let parts: Vec<String> = items.map(|s| format!("'{s}'")).collect();
    match parts.as_slice() {
        [only] => format!("({only},)"),
        _ => format!("({})", parts.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_permission_message_matches_python() {
        let err = validate_permission("BOGUS").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid permission 'BOGUS'. Valid permissions are: \
             ('READ', 'USE', 'EDIT', 'MANAGE', 'NO_PERMISSIONS')"
        );
    }

    #[test]
    fn invalid_resource_type_message_is_sorted() {
        let err = validate_resource_type("bogus").unwrap_err();
        assert_eq!(
            err.message,
            "Invalid resource type 'bogus'. Valid resource types are: \
             ('experiment', 'gateway_endpoint', 'gateway_model_definition', 'gateway_secret', \
             'mcp_server', 'prompt', 'registered_model', 'scorer', 'workspace')"
        );
    }

    #[test]
    fn workspace_rejects_read_and_edit() {
        for perm in ["READ", "EDIT", "NO_PERMISSIONS"] {
            let err = validate_permission_for_resource_type(perm, "workspace").unwrap_err();
            assert!(err
                .message
                .contains("Workspace-wide grants accept only: ('MANAGE', 'USE')"));
        }
        assert!(validate_permission_for_resource_type("USE", "workspace").is_ok());
        assert!(validate_permission_for_resource_type("MANAGE", "workspace").is_ok());
    }

    #[test]
    fn resource_rejects_no_permissions() {
        let err =
            validate_permission_for_resource_type("NO_PERMISSIONS", "experiment").unwrap_err();
        assert!(err
            .message
            .contains("Resource-level grants accept only: ('EDIT', 'MANAGE', 'READ', 'USE')"));
    }

    #[test]
    fn max_permission_picks_higher() {
        assert_eq!(max_permission("READ", "EDIT"), "EDIT");
        assert_eq!(max_permission("MANAGE", "READ"), "MANAGE");
        assert_eq!(max_permission("READ", "READ"), "READ");
    }
}
