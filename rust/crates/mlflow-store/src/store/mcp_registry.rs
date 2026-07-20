//! MCP server registry persistence over the Python-owned Alembic schema.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::net::{IpAddr, ToSocketAddrs};

use mlflow_error::MlflowError;
use mlflow_search::{Comparison, Value as SearchValue};
use serde_json::Value;
use url::Url;
use uuid::Uuid;

use super::dbutil::{RowLike, Val};
use super::experiments::{internal, now_millis};
use super::search::SEARCH_MAX_RESULTS_THRESHOLD;
use super::TrackingStore;

const MCP_MAX_RESULTS: i32 = 1000;
const MAX_SEMVER_LENGTH: usize = 128;
const MAX_SEMVER_CORE: u32 = 2_147_483_647;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpStatus {
    Draft,
    Active,
    Deprecated,
    Deleted,
}

impl McpStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Active => "active",
            Self::Deprecated => "deprecated",
            Self::Deleted => "deleted",
        }
    }

    pub fn parse(value: &str) -> Result<Self, MlflowError> {
        match value {
            "draft" => Ok(Self::Draft),
            "active" => Ok(Self::Active),
            "deprecated" => Ok(Self::Deprecated),
            "deleted" => Ok(Self::Deleted),
            _ => Err(MlflowError::invalid_parameter_value(format!(
                "Invalid status: '{value}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpTransportType {
    StreamableHttp,
    Sse,
}

impl McpTransportType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StreamableHttp => "streamable-http",
            Self::Sse => "sse",
        }
    }

    pub fn parse(value: &str) -> Result<Self, MlflowError> {
        match value {
            "streamable-http" => Ok(Self::StreamableHttp),
            "sse" => Ok(Self::Sse),
            _ => Err(MlflowError::invalid_parameter_value(format!(
                "Invalid transport_type: '{value}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpServerVersion {
    pub name: String,
    pub version: String,
    pub server_json: Value,
    pub display_name: Option<String>,
    pub workspace: String,
    pub status: McpStatus,
    pub tools: Option<Vec<Value>>,
    pub aliases: Vec<String>,
    pub tags: BTreeMap<String, String>,
    pub source: Option<String>,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    pub creation_timestamp: i64,
    pub last_updated_timestamp: i64,
    semver: SemVer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpAccessEndpoint {
    pub id: String,
    pub server_name: String,
    pub url: String,
    pub transport_type: McpTransportType,
    pub workspace: String,
    pub server_version: Option<String>,
    pub server_alias: Option<String>,
    pub resolved_version: Box<McpServerVersion>,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    pub creation_timestamp: i64,
    pub last_updated_timestamp: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpServer {
    pub name: String,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub icons: Option<Vec<Value>>,
    pub workspace: String,
    pub status: Option<McpStatus>,
    pub access_endpoints: Vec<McpAccessEndpoint>,
    pub latest_version: Option<String>,
    pub aliases: BTreeMap<String, String>,
    pub tags: BTreeMap<String, String>,
    pub created_by: Option<String>,
    pub last_updated_by: Option<String>,
    pub creation_timestamp: i64,
    pub last_updated_timestamp: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpServersPage {
    pub items: Vec<McpServer>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpServerVersionsPage {
    pub items: Vec<McpServerVersion>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McpAccessEndpointsPage {
    pub items: Vec<McpAccessEndpoint>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum McpPatch<T> {
    #[default]
    Unset,
    Set(T),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemVer {
    major: u32,
    minor: u32,
    patch: u32,
    prerelease: Vec<String>,
    prerelease_key: String,
}

impl TrackingStore {
    pub async fn create_mcp_server(
        &self,
        workspace: &str,
        name: &str,
        description: Option<&str>,
        icons: Option<Vec<Value>>,
        created_by: Option<&str>,
    ) -> Result<McpServer, MlflowError> {
        validate_name(name)?;
        validate_icons(icons.as_deref(), "icons")?;
        if self.mcp_server_exists(workspace, name).await? {
            return Err(MlflowError::resource_already_exists(format!(
                "MCP server with name '{name}' already exists"
            )));
        }
        let now = now_millis();
        let d = self.db().dialect();
        let mut placeholders = (1..=8)
            .map(|index| d.placeholder(index))
            .collect::<Vec<_>>();
        placeholders.insert(2, "NULL".to_string());
        self.db()
            .exec(
                &format!(
                    "INSERT INTO mcp_servers (workspace, name, display_name, description, icons, \
                     created_by, last_updated_by, created_at, last_updated_at) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(workspace.to_string()),
                    Val::Text(name.to_string()),
                    Val::OptText(description.map(str::to_string)),
                    Val::OptJson(icons.map(Value::Array)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Int(now),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_mcp_server(workspace, name).await
    }

    pub async fn get_mcp_server(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<McpServer, MlflowError> {
        let d = self.db().dialect();
        let row = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT workspace, name, display_name, description, icons, created_by, \
                     last_updated_by, created_at, last_updated_at FROM mcp_servers WHERE workspace \
                     = {} AND name = {}",
                    d.placeholder(1),
                    d.placeholder(2)
                ),
                &[Val::Text(workspace.into()), Val::Text(name.into())],
                map_server_base,
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!("MCP server '{name}' not found"))
            })?;
        self.enrich_server(row).await
    }

    pub async fn search_mcp_servers(
        &self,
        workspace: &str,
        filter_string: Option<&str>,
        max_results: i32,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<McpServersPage, MlflowError> {
        validate_max_results(max_results, MCP_MAX_RESULTS)?;
        let offset = page_offset(page_token)?;
        let d = self.db().dialect();
        let rows = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT workspace, name, display_name, description, icons, created_by, \
                     last_updated_by, created_at, last_updated_at FROM mcp_servers WHERE workspace \
                     = {}",
                    d.placeholder(1)
                ),
                &[Val::Text(workspace.into())],
                map_server_base,
            )
            .await
            .map_err(internal)?;
        let mut servers = Vec::with_capacity(rows.len());
        for row in rows {
            servers.push(self.enrich_server(row).await?);
        }
        if let Some(filter) = filter_string.filter(|value| !value.is_empty()) {
            let comparisons = mlflow_search::parse::mcp_server_filter(filter)
                .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
            servers.retain(|server| server_matches(server, &comparisons));
        }
        sort_servers(&mut servers, order_by)?;
        Ok(page(servers, offset, max_results, |items, token| {
            McpServersPage {
                items,
                next_page_token: token,
            }
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn update_mcp_server(
        &self,
        workspace: &str,
        name: &str,
        description: McpPatch<Option<String>>,
        display_name: McpPatch<Option<String>>,
        icons: McpPatch<Option<Vec<Value>>>,
        last_updated_by: Option<&str>,
    ) -> Result<McpServer, MlflowError> {
        self.require_mcp_server(workspace, name).await?;
        if let McpPatch::Set(value) = &icons {
            validate_icons(value.as_deref(), "icons")?;
        }
        let d = self.db().dialect();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let McpPatch::Set(value) = description {
            values.push(Val::OptText(value));
            assignments.push(format!("description = {}", d.placeholder(values.len())));
        }
        if let McpPatch::Set(value) = display_name {
            values.push(Val::OptText(value));
            assignments.push(format!("display_name = {}", d.placeholder(values.len())));
        }
        if let McpPatch::Set(value) = icons {
            values.push(Val::OptJson(value.map(Value::Array)));
            assignments.push(format!("icons = {}", d.placeholder(values.len())));
        }
        values.push(Val::OptText(last_updated_by.map(str::to_string)));
        assignments.push(format!("last_updated_by = {}", d.placeholder(values.len())));
        values.push(Val::Int(now_millis()));
        assignments.push(format!("last_updated_at = {}", d.placeholder(values.len())));
        values.push(Val::Text(workspace.into()));
        let workspace_ph = d.placeholder(values.len());
        values.push(Val::Text(name.into()));
        let name_ph = d.placeholder(values.len());
        self.db()
            .exec(
                &format!(
                    "UPDATE mcp_servers SET {} WHERE workspace = {workspace_ph} AND name = {name_ph}",
                    assignments.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        self.get_mcp_server(workspace, name).await
    }

    pub async fn delete_mcp_server(&self, workspace: &str, name: &str) -> Result<(), MlflowError> {
        self.require_mcp_server(workspace, name).await?;
        let mut versions = self.all_version_rows(workspace, Some(name), true).await?;
        versions.sort_by(|left, right| left.version.cmp(&right.version));
        if let Some(active) = versions
            .iter()
            .find(|version| version.status == McpStatus::Active)
        {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Cannot delete MCP server '{name}' while it still has an active version ('{}'). \
                 Delete or deactivate the active version first.",
                active.version
            )));
        }
        let d = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM mcp_servers WHERE workspace = {} AND name = {}",
                    d.placeholder(1),
                    d.placeholder(2)
                ),
                &[Val::Text(workspace.into()), Val::Text(name.into())],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_mcp_server_version(
        &self,
        workspace: &str,
        server_json: Value,
        display_name: Option<&str>,
        source: Option<&str>,
        status: McpStatus,
        tools: Option<Vec<Value>>,
        created_by: Option<&str>,
    ) -> Result<McpServerVersion, MlflowError> {
        let object = server_json
            .as_object()
            .ok_or_else(|| MlflowError::invalid_parameter_value("server_json must be an object"))?;
        let name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
            MlflowError::invalid_parameter_value(
                "server_json must contain 'name' and 'version' keys",
            )
        })?;
        let version = object
            .get("version")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                MlflowError::invalid_parameter_value(
                    "server_json must contain 'name' and 'version' keys",
                )
            })?;
        let name = name.to_string();
        let version = version.to_string();
        validate_name(&name)?;
        let semver = parse_semver(&version, "server_json.version")?;
        validate_icons(
            object
                .get("icons")
                .and_then(Value::as_array)
                .map(Vec::as_slice),
            "server_json.icons",
        )?;
        if !matches!(status, McpStatus::Draft | McpStatus::Active) {
            return Err(MlflowError::invalid_parameter_value(
                "Initial MCP server registration status must be 'draft' or 'active'.",
            ));
        }
        validate_tools(tools.as_deref())?;
        if !self.mcp_server_exists(workspace, &name).await? {
            self.create_mcp_server(workspace, &name, None, None, created_by)
                .await?;
        }
        if self.version_row_exists(workspace, &name, &version).await? {
            return Err(MlflowError::resource_already_exists(format!(
                "MCP server version '{name}' version '{version}' already exists"
            )));
        }
        let d = self.db().dialect();
        let now = now_millis();
        self.db()
            .exec(
                &format!(
                    "INSERT INTO mcp_server_versions (workspace, name, version, version_major, \
                     version_minor, version_patch, version_prerelease_sort_key, server_json, \
                     display_name, status, tools, source, created_by, last_updated_by, created_at, \
                     last_updated_at) VALUES ({})",
                    (1..=16)
                        .map(|index| d.placeholder(index))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                &[
                    Val::Text(workspace.into()),
                    Val::Text(name.clone()),
                    Val::Text(version.clone()),
                    Val::Int(i64::from(semver.major)),
                    Val::Int(i64::from(semver.minor)),
                    Val::Int(i64::from(semver.patch)),
                    Val::Text(semver.prerelease_key.clone()),
                    Val::OptJson(Some(server_json)),
                    Val::OptText(display_name.map(str::to_string)),
                    Val::Text(status.as_str().into()),
                    Val::OptJson(tools.map(Value::Array)),
                    Val::OptText(source.map(str::to_string)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Int(now),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_mcp_server_version(workspace, &name, &version)
            .await
    }

    pub async fn get_mcp_server_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<McpServerVersion, MlflowError> {
        self.get_version_row(workspace, name, version, false)
            .await?
            .ok_or_else(|| version_not_found(name, version))
    }

    pub async fn get_mcp_server_version_by_alias(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
    ) -> Result<McpServerVersion, MlflowError> {
        if alias == "latest" {
            return self.get_latest_mcp_server_version(workspace, name).await;
        }
        let aliases = self.aliases(workspace, name).await?;
        let version = aliases.get(alias).ok_or_else(|| {
            MlflowError::resource_does_not_exist(format!(
                "Alias '{alias}' not found for MCP server '{name}'"
            ))
        })?;
        self.get_mcp_server_version(workspace, name, version).await
    }

    pub async fn get_latest_mcp_server_version(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<McpServerVersion, MlflowError> {
        self.require_mcp_server(workspace, name).await?;
        let versions = self.all_version_rows(workspace, Some(name), false).await?;
        latest_version(versions).ok_or_else(|| {
            MlflowError::resource_does_not_exist(format!(
                "No resolved latest version found for MCP server '{name}'"
            ))
        })
    }

    pub async fn search_mcp_server_versions(
        &self,
        workspace: &str,
        name: &str,
        filter_string: Option<&str>,
        max_results: i32,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<McpServerVersionsPage, MlflowError> {
        validate_max_results(max_results, SEARCH_MAX_RESULTS_THRESHOLD as i32)?;
        let offset = page_offset(page_token)?;
        let mut versions = self.all_version_rows(workspace, Some(name), false).await?;
        if let Some(filter) = filter_string.filter(|value| !value.is_empty()) {
            let comparisons = mlflow_search::parse::mcp_version_filter(filter)
                .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
            for cmp in &comparisons {
                if cmp.key == "version" && matches!(cmp.comparator.as_str(), "LIKE" | "ILIKE") {
                    return Err(MlflowError::invalid_parameter_value(
                        "version only supports semantic comparators '=', '!=', '<', '<=', '>', and '>='",
                    ));
                }
            }
            versions.retain(|version| version_matches(version, &comparisons));
        }
        sort_versions(&mut versions, order_by)?;
        Ok(page(versions, offset, max_results, |items, token| {
            McpServerVersionsPage {
                items,
                next_page_token: token,
            }
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn update_mcp_server_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        display_name: McpPatch<Option<String>>,
        status: McpPatch<McpStatus>,
        tools: McpPatch<Option<Vec<Value>>>,
        last_updated_by: Option<&str>,
    ) -> Result<McpServerVersion, MlflowError> {
        let current = self
            .get_mcp_server_version(workspace, name, version)
            .await?;
        if let McpPatch::Set(new_status) = status {
            validate_status_transition(current.status, new_status)?;
        }
        if let McpPatch::Set(value) = &tools {
            validate_tools(value.as_deref())?;
        }
        let d = self.db().dialect();
        let mut values = Vec::new();
        let mut assignments = Vec::new();
        if let McpPatch::Set(value) = display_name {
            values.push(Val::OptText(value));
            assignments.push(format!("display_name = {}", d.placeholder(values.len())));
        }
        if let McpPatch::Set(value) = status {
            values.push(Val::Text(value.as_str().into()));
            assignments.push(format!("status = {}", d.placeholder(values.len())));
        }
        if let McpPatch::Set(value) = tools {
            values.push(Val::OptJson(value.map(Value::Array)));
            assignments.push(format!("tools = {}", d.placeholder(values.len())));
        }
        values.push(Val::OptText(last_updated_by.map(str::to_string)));
        assignments.push(format!("last_updated_by = {}", d.placeholder(values.len())));
        values.push(Val::Int(now_millis()));
        assignments.push(format!("last_updated_at = {}", d.placeholder(values.len())));
        values.push(Val::Text(workspace.into()));
        let workspace_ph = d.placeholder(values.len());
        values.push(Val::Text(name.into()));
        let name_ph = d.placeholder(values.len());
        values.push(Val::Text(version.into()));
        let version_ph = d.placeholder(values.len());
        self.db()
            .exec(
                &format!(
                    "UPDATE mcp_server_versions SET {} WHERE workspace = {workspace_ph} AND name \
                     = {name_ph} AND version = {version_ph}",
                    assignments.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        self.get_mcp_server_version(workspace, name, version).await
    }

    pub async fn delete_mcp_server_version(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<(), MlflowError> {
        let row = self
            .get_version_row(workspace, name, version, true)
            .await?
            .ok_or_else(|| version_not_found(name, version))?;
        validate_status_transition(row.status, McpStatus::Deleted)?;
        let d = self.db().dialect();
        let aliases = self.aliases(workspace, name).await?;
        let affected_aliases = aliases
            .iter()
            .filter(|(_, target)| target.as_str() == version)
            .map(|(alias, _)| alias.clone())
            .collect::<Vec<_>>();
        for alias in &affected_aliases {
            self.delete_alias_endpoints(workspace, name, alias).await?;
        }
        self.db()
            .exec(
                &format!(
                    "DELETE FROM mcp_server_aliases WHERE workspace = {} AND name = {} AND version = {}",
                    d.placeholder(1), d.placeholder(2), d.placeholder(3)
                ),
                &[
                    Val::Text(workspace.into()),
                    Val::Text(name.into()),
                    Val::Text(version.into()),
                ],
            )
            .await
            .map_err(internal)?;
        self.db()
            .exec(
                &format!(
                    "DELETE FROM mcp_access_endpoints WHERE workspace = {} AND server_name = {} \
                     AND server_version = {}",
                    d.placeholder(1),
                    d.placeholder(2),
                    d.placeholder(3)
                ),
                &[
                    Val::Text(workspace.into()),
                    Val::Text(name.into()),
                    Val::Text(version.into()),
                ],
            )
            .await
            .map_err(internal)?;
        self.db()
            .exec(
                &format!(
                    "UPDATE mcp_server_versions SET status = 'deleted', last_updated_at = {} WHERE \
                     workspace = {} AND name = {} AND version = {}",
                    d.placeholder(1), d.placeholder(2), d.placeholder(3), d.placeholder(4)
                ),
                &[
                    Val::Int(now_millis()),
                    Val::Text(workspace.into()),
                    Val::Text(name.into()),
                    Val::Text(version.into()),
                ],
            )
            .await
            .map_err(internal)?;
        if self
            .all_version_rows(workspace, Some(name), false)
            .await?
            .is_empty()
        {
            self.delete_alias_endpoints(workspace, name, "latest")
                .await?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_mcp_access_endpoint(
        &self,
        workspace: &str,
        server_name: &str,
        url: &str,
        transport_type: McpTransportType,
        server_version: Option<&str>,
        server_alias: Option<&str>,
        created_by: Option<&str>,
    ) -> Result<McpAccessEndpoint, MlflowError> {
        validate_exactly_one(server_version, server_alias)?;
        validate_endpoint_url(url)?;
        self.require_mcp_server(workspace, server_name).await?;
        self.resolve_endpoint_target(
            workspace,
            server_name,
            server_version,
            server_alias,
            "create",
        )
        .await?;
        let id = format!("ae-{}", Uuid::new_v4().simple());
        let now = now_millis();
        let d = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "INSERT INTO mcp_access_endpoints (id, workspace, server_name, server_version, \
                     server_alias, url, transport_type, created_by, last_updated_by, created_at, \
                     last_updated_at) VALUES ({})",
                    (1..=11)
                        .map(|index| d.placeholder(index))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                &[
                    Val::Text(id.clone()),
                    Val::Text(workspace.into()),
                    Val::Text(server_name.into()),
                    Val::OptText(server_version.map(str::to_string)),
                    Val::OptText(server_alias.map(str::to_string)),
                    Val::Text(url.into()),
                    Val::Text(transport_type.as_str().into()),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Int(now),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_mcp_access_endpoint(workspace, server_name, &id)
            .await
    }

    pub async fn get_mcp_access_endpoint(
        &self,
        workspace: &str,
        server_name: &str,
        endpoint_id: &str,
    ) -> Result<McpAccessEndpoint, MlflowError> {
        let base = self
            .get_endpoint_base(workspace, endpoint_id)
            .await?
            .ok_or_else(|| endpoint_not_found(endpoint_id))?;
        if base.server_name != server_name {
            return Err(MlflowError::resource_does_not_exist(format!(
                "MCPAccessEndpoint {endpoint_id} does not belong to server '{server_name}'"
            )));
        }
        self.enrich_endpoint(base)
            .await?
            .ok_or_else(|| endpoint_not_found(endpoint_id))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn search_mcp_access_endpoints(
        &self,
        workspace: &str,
        server_name: Option<&str>,
        server_version: Option<&str>,
        server_alias: Option<&str>,
        filter_string: Option<&str>,
        max_results: i32,
        order_by: &[String],
        page_token: Option<&str>,
    ) -> Result<McpAccessEndpointsPage, MlflowError> {
        validate_max_results(max_results, SEARCH_MAX_RESULTS_THRESHOLD as i32)?;
        let offset = page_offset(page_token)?;
        let mut endpoints = Vec::new();
        for base in self.all_endpoint_bases(workspace).await? {
            if server_name.is_some_and(|name| base.server_name != name)
                || server_version
                    .is_some_and(|version| base.server_version.as_deref() != Some(version))
                || server_alias.is_some_and(|alias| base.server_alias.as_deref() != Some(alias))
            {
                continue;
            }
            if let Some(endpoint) = self.enrich_endpoint(base).await? {
                endpoints.push(endpoint);
            }
        }
        if let Some(filter) = filter_string.filter(|value| !value.is_empty()) {
            let comparisons = mlflow_search::parse::mcp_endpoint_filter(filter)
                .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
            endpoints.retain(|endpoint| endpoint_matches(endpoint, &comparisons));
        }
        sort_endpoints(&mut endpoints, order_by)?;
        Ok(page(endpoints, offset, max_results, |items, token| {
            McpAccessEndpointsPage {
                items,
                next_page_token: token,
            }
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn update_mcp_access_endpoint(
        &self,
        workspace: &str,
        server_name: &str,
        endpoint_id: &str,
        server_version: McpPatch<Option<String>>,
        server_alias: McpPatch<Option<String>>,
        url: McpPatch<Option<String>>,
        transport_type: McpPatch<Option<McpTransportType>>,
        last_updated_by: Option<&str>,
    ) -> Result<McpAccessEndpoint, MlflowError> {
        let current = self
            .get_endpoint_base(workspace, endpoint_id)
            .await?
            .ok_or_else(|| endpoint_not_found(endpoint_id))?;
        if current.server_name != server_name {
            return Err(MlflowError::resource_does_not_exist(format!(
                "MCPAccessEndpoint {endpoint_id} does not belong to server '{server_name}'"
            )));
        }
        if matches!(
            (&server_version, &server_alias),
            (McpPatch::Set(Some(_)), McpPatch::Set(Some(_)))
        ) {
            return Err(MlflowError::invalid_parameter_value(
                "Cannot set both server_version and server_alias in a single update",
            ));
        }
        if let McpPatch::Set(Some(version)) = &server_version {
            self.resolve_endpoint_target(workspace, server_name, Some(version), None, "update")
                .await?;
        }
        if let McpPatch::Set(Some(alias)) = &server_alias {
            self.resolve_endpoint_target(workspace, server_name, None, Some(alias), "update")
                .await?;
        }
        if let McpPatch::Set(value) = &url {
            let value = value.as_deref().ok_or_else(|| {
                MlflowError::invalid_parameter_value("MCP access endpoint url cannot be None")
            })?;
            validate_endpoint_url(value)?;
        }
        let d = self.db().dialect();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let McpPatch::Set(Some(version)) = server_version {
            values.push(Val::Text(version));
            assignments.push(format!("server_version = {}", d.placeholder(values.len())));
            assignments.push("server_alias = NULL".to_string());
        }
        if let McpPatch::Set(Some(alias)) = server_alias {
            values.push(Val::Text(alias));
            assignments.push(format!("server_alias = {}", d.placeholder(values.len())));
            assignments.push("server_version = NULL".to_string());
        }
        if let McpPatch::Set(Some(value)) = url {
            values.push(Val::Text(value));
            assignments.push(format!("url = {}", d.placeholder(values.len())));
        }
        if let McpPatch::Set(Some(value)) = transport_type {
            values.push(Val::Text(value.as_str().into()));
            assignments.push(format!("transport_type = {}", d.placeholder(values.len())));
        }
        values.push(Val::OptText(last_updated_by.map(str::to_string)));
        assignments.push(format!("last_updated_by = {}", d.placeholder(values.len())));
        values.push(Val::Int(now_millis()));
        assignments.push(format!("last_updated_at = {}", d.placeholder(values.len())));
        values.push(Val::Text(endpoint_id.into()));
        let id_ph = d.placeholder(values.len());
        values.push(Val::Text(workspace.into()));
        let workspace_ph = d.placeholder(values.len());
        self.db()
            .exec(
                &format!(
                    "UPDATE mcp_access_endpoints SET {} WHERE id = {id_ph} AND workspace = {workspace_ph}",
                    assignments.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        self.get_mcp_access_endpoint(workspace, server_name, endpoint_id)
            .await
    }

    pub async fn delete_mcp_access_endpoint(
        &self,
        workspace: &str,
        server_name: &str,
        endpoint_id: &str,
    ) -> Result<(), MlflowError> {
        self.get_mcp_access_endpoint(workspace, server_name, endpoint_id)
            .await?;
        let d = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM mcp_access_endpoints WHERE id = {} AND workspace = {}",
                    d.placeholder(1),
                    d.placeholder(2)
                ),
                &[Val::Text(endpoint_id.into()), Val::Text(workspace.into())],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    pub async fn set_mcp_server_tag(
        &self,
        workspace: &str,
        name: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        self.require_mcp_server(workspace, name).await?;
        self.upsert_tag("mcp_server_tags", workspace, name, None, key, value)
            .await
    }

    pub async fn delete_mcp_server_tag(
        &self,
        workspace: &str,
        name: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        if !self
            .delete_mcp_tag_row("mcp_server_tags", workspace, name, None, key)
            .await?
        {
            return Err(MlflowError::resource_does_not_exist(format!(
                "Tag '{key}' not found on MCP server '{name}'"
            )));
        }
        Ok(())
    }

    pub async fn set_mcp_server_version_tag(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        self.get_mcp_server_version(workspace, name, version)
            .await?;
        self.upsert_tag(
            "mcp_server_version_tags",
            workspace,
            name,
            Some(version),
            key,
            value,
        )
        .await
    }

    pub async fn delete_mcp_server_version_tag(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        key: &str,
    ) -> Result<(), MlflowError> {
        self.get_mcp_server_version(workspace, name, version)
            .await?;
        if !self
            .delete_mcp_tag_row(
                "mcp_server_version_tags",
                workspace,
                name,
                Some(version),
                key,
            )
            .await?
        {
            return Err(MlflowError::resource_does_not_exist(format!(
                "Tag '{key}' not found on MCP server version '{name}' version '{version}'"
            )));
        }
        Ok(())
    }

    pub async fn set_mcp_server_alias(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
        version: &str,
    ) -> Result<(), MlflowError> {
        if alias == "latest" {
            return Err(MlflowError::invalid_parameter_value(
                "The alias name 'latest' is reserved for automatic resolution",
            ));
        }
        self.require_mcp_server(workspace, name).await?;
        let target = self
            .get_version_row(workspace, name, version, true)
            .await?
            .ok_or_else(|| version_not_found(name, version))?;
        if target.status == McpStatus::Deleted {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Cannot set alias '{alias}' to deleted MCP server version '{name}' version '{version}'"
            )));
        }
        let d = self.db().dialect();
        if self.aliases(workspace, name).await?.contains_key(alias) {
            self.db()
                .exec(
                    &format!(
                        "UPDATE mcp_server_aliases SET version = {} WHERE workspace = {} AND name = {} AND alias = {}",
                        d.placeholder(1), d.placeholder(2), d.placeholder(3), d.placeholder(4)
                    ),
                    &[
                        Val::Text(version.into()), Val::Text(workspace.into()),
                        Val::Text(name.into()), Val::Text(alias.into()),
                    ],
                )
                .await
                .map_err(internal)?;
        } else {
            self.db()
                .exec(
                    &format!(
                        "INSERT INTO mcp_server_aliases (workspace, name, alias, version) VALUES ({}, {}, {}, {})",
                        d.placeholder(1), d.placeholder(2), d.placeholder(3), d.placeholder(4)
                    ),
                    &[
                        Val::Text(workspace.into()), Val::Text(name.into()),
                        Val::Text(alias.into()), Val::Text(version.into()),
                    ],
                )
                .await
                .map_err(internal)?;
        }
        Ok(())
    }

    pub async fn delete_mcp_server_alias(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
    ) -> Result<(), MlflowError> {
        if !self.aliases(workspace, name).await?.contains_key(alias) {
            return Err(MlflowError::resource_does_not_exist(format!(
                "Alias '{alias}' not found on MCP server '{name}'"
            )));
        }
        self.delete_alias_endpoints(workspace, name, alias).await?;
        let d = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM mcp_server_aliases WHERE workspace = {} AND name = {} AND alias = {}",
                    d.placeholder(1), d.placeholder(2), d.placeholder(3)
                ),
                &[Val::Text(workspace.into()), Val::Text(name.into()), Val::Text(alias.into())],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    async fn enrich_server(&self, mut server: McpServer) -> Result<McpServer, MlflowError> {
        server.tags = self.server_tags(&server.workspace, &server.name).await?;
        server.aliases = self.aliases(&server.workspace, &server.name).await?;
        let versions = self
            .all_version_rows(&server.workspace, Some(&server.name), false)
            .await?;
        if let Some(latest) = latest_version(versions) {
            server.latest_version = Some(latest.version.clone());
            server.status = Some(latest.status);
            if server.description.is_none() {
                server.description = latest
                    .server_json
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
        }
        let mut endpoints = Vec::new();
        for endpoint in self.all_endpoint_bases(&server.workspace).await? {
            if endpoint.server_name == server.name {
                if let Some(endpoint) = self.enrich_endpoint(endpoint).await? {
                    endpoints.push(endpoint);
                }
            }
        }
        endpoints.sort_by(|left, right| left.id.cmp(&right.id));
        server.access_endpoints = endpoints;
        Ok(server)
    }

    async fn enrich_endpoint(
        &self,
        base: EndpointBase,
    ) -> Result<Option<McpAccessEndpoint>, MlflowError> {
        let target = match self
            .resolve_endpoint_target(
                &base.workspace,
                &base.server_name,
                base.server_version.as_deref(),
                base.server_alias.as_deref(),
                "resolve",
            )
            .await
        {
            Ok(target) => target,
            Err(error) if error.error_code == mlflow_error::ErrorCode::ResourceDoesNotExist => {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        Ok(Some(McpAccessEndpoint {
            id: base.id,
            server_name: base.server_name,
            url: base.url,
            transport_type: base.transport_type,
            workspace: base.workspace,
            server_version: base.server_version,
            server_alias: base.server_alias,
            resolved_version: Box::new(target),
            created_by: base.created_by,
            last_updated_by: base.last_updated_by,
            creation_timestamp: base.creation_timestamp,
            last_updated_timestamp: base.last_updated_timestamp,
        }))
    }

    async fn resolve_endpoint_target(
        &self,
        workspace: &str,
        name: &str,
        version: Option<&str>,
        alias: Option<&str>,
        action: &str,
    ) -> Result<McpServerVersion, MlflowError> {
        if let Some(version) = version {
            let row = self
                .get_version_row(workspace, name, version, true)
                .await?
                .ok_or_else(|| version_not_found(name, version))?;
            if row.status == McpStatus::Deleted {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Cannot {action} MCP access endpoint to deleted MCP server version '{name}' version '{version}'"
                )));
            }
            return Ok(row);
        }
        let alias = alias.ok_or_else(|| {
            MlflowError::invalid_parameter_value(
                "Exactly one of server_version or server_alias must be provided",
            )
        })?;
        self.get_mcp_server_version_by_alias(workspace, name, alias)
            .await
    }

    async fn all_version_rows(
        &self,
        workspace: &str,
        name: Option<&str>,
        include_deleted: bool,
    ) -> Result<Vec<McpServerVersion>, MlflowError> {
        let d = self.db().dialect();
        let mut values = vec![Val::Text(workspace.into())];
        let mut sql = format!(
            "SELECT workspace, name, version, version_major, version_minor, version_patch, \
             version_prerelease_sort_key, server_json, display_name, status, tools, source, \
             created_by, last_updated_by, created_at, last_updated_at FROM mcp_server_versions \
             WHERE workspace = {}",
            d.placeholder(1)
        );
        if let Some(name) = name {
            values.push(Val::Text(name.into()));
            sql.push_str(&format!(" AND name = {}", d.placeholder(values.len())));
        }
        if !include_deleted {
            sql.push_str(" AND status != 'deleted'");
        }
        let mut rows = self
            .db()
            .fetch_all(&sql, &values, map_version_base)
            .await
            .map_err(internal)?;
        for row in &mut rows {
            self.enrich_version(row).await?;
        }
        Ok(rows)
    }

    async fn get_version_row(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
        include_deleted: bool,
    ) -> Result<Option<McpServerVersion>, MlflowError> {
        Ok(self
            .all_version_rows(workspace, Some(name), include_deleted)
            .await?
            .into_iter()
            .find(|row| row.version == version))
    }

    async fn enrich_version(&self, version: &mut McpServerVersion) -> Result<(), MlflowError> {
        version.tags = self
            .version_tags(&version.workspace, &version.name, &version.version)
            .await?;
        version.aliases = self
            .aliases(&version.workspace, &version.name)
            .await?
            .into_iter()
            .filter_map(|(alias, target)| (target == version.version).then_some(alias))
            .collect();
        Ok(())
    }

    async fn mcp_server_exists(&self, workspace: &str, name: &str) -> Result<bool, MlflowError> {
        let d = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT name FROM mcp_servers WHERE workspace = {} AND name = {}",
                    d.placeholder(1),
                    d.placeholder(2)
                ),
                &[Val::Text(workspace.into()), Val::Text(name.into())],
                |row| row.get_string("name"),
            )
            .await
            .map(|value| value.is_some())
            .map_err(internal)
    }

    async fn require_mcp_server(&self, workspace: &str, name: &str) -> Result<(), MlflowError> {
        if self.mcp_server_exists(workspace, name).await? {
            Ok(())
        } else {
            Err(MlflowError::resource_does_not_exist(format!(
                "MCP server '{name}' not found"
            )))
        }
    }

    async fn version_row_exists(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<bool, MlflowError> {
        Ok(self
            .get_version_row(workspace, name, version, true)
            .await?
            .is_some())
    }

    async fn aliases(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<BTreeMap<String, String>, MlflowError> {
        self.string_map(
            "mcp_server_aliases",
            "alias",
            "version",
            workspace,
            name,
            None,
        )
        .await
    }

    async fn server_tags(
        &self,
        workspace: &str,
        name: &str,
    ) -> Result<BTreeMap<String, String>, MlflowError> {
        self.string_map("mcp_server_tags", "key", "value", workspace, name, None)
            .await
    }

    async fn version_tags(
        &self,
        workspace: &str,
        name: &str,
        version: &str,
    ) -> Result<BTreeMap<String, String>, MlflowError> {
        self.string_map(
            "mcp_server_version_tags",
            "key",
            "value",
            workspace,
            name,
            Some(version),
        )
        .await
    }

    async fn string_map(
        &self,
        table: &str,
        key_column: &str,
        value_column: &str,
        workspace: &str,
        name: &str,
        version: Option<&str>,
    ) -> Result<BTreeMap<String, String>, MlflowError> {
        let d = self.db().dialect();
        let key_column = d.quote_ident(key_column);
        let value_column = d.quote_ident(value_column);
        let mut values = vec![Val::Text(workspace.into()), Val::Text(name.into())];
        let mut sql = format!(
            "SELECT {key_column} AS map_key, {value_column} AS map_value FROM {table} WHERE \
             workspace = {} AND name = {}",
            d.placeholder(1),
            d.placeholder(2)
        );
        if let Some(version) = version {
            values.push(Val::Text(version.into()));
            sql.push_str(&format!(" AND version = {}", d.placeholder(3)));
        }
        sql.push_str(&format!(" ORDER BY {key_column}"));
        let rows = self
            .db()
            .fetch_all(&sql, &values, |row| {
                Ok((
                    row.get_string("map_key")?,
                    row.get_opt_string("map_value")?.unwrap_or_default(),
                ))
            })
            .await
            .map_err(internal)?;
        Ok(rows.into_iter().collect())
    }

    async fn all_endpoint_bases(&self, workspace: &str) -> Result<Vec<EndpointBase>, MlflowError> {
        let d = self.db().dialect();
        self.db()
            .fetch_all(
                &format!(
                    "SELECT id, workspace, server_name, server_version, server_alias, url, \
                     transport_type, created_by, last_updated_by, created_at, last_updated_at FROM \
                     mcp_access_endpoints WHERE workspace = {}",
                    d.placeholder(1)
                ),
                &[Val::Text(workspace.into())],
                map_endpoint_base,
            )
            .await
            .map_err(internal)
    }

    async fn get_endpoint_base(
        &self,
        workspace: &str,
        id: &str,
    ) -> Result<Option<EndpointBase>, MlflowError> {
        Ok(self
            .all_endpoint_bases(workspace)
            .await?
            .into_iter()
            .find(|endpoint| endpoint.id == id))
    }

    async fn upsert_tag(
        &self,
        table: &str,
        workspace: &str,
        name: &str,
        version: Option<&str>,
        key: &str,
        value: &str,
    ) -> Result<(), MlflowError> {
        let existing = self
            .string_map(table, "key", "value", workspace, name, version)
            .await?
            .contains_key(key);
        let d = self.db().dialect();
        let key_column = d.quote_ident("key");
        if existing {
            let mut values = vec![
                Val::Text(value.into()),
                Val::Text(workspace.into()),
                Val::Text(name.into()),
            ];
            let mut sql = format!(
                "UPDATE {table} SET value = {} WHERE workspace = {} AND name = {}",
                d.placeholder(1),
                d.placeholder(2),
                d.placeholder(3)
            );
            if let Some(version) = version {
                values.push(Val::Text(version.into()));
                sql.push_str(&format!(" AND version = {}", d.placeholder(values.len())));
            }
            values.push(Val::Text(key.into()));
            sql.push_str(&format!(
                " AND {key_column} = {}",
                d.placeholder(values.len())
            ));
            self.db().exec(&sql, &values).await.map_err(internal)?;
        } else {
            let (columns, values) = if let Some(version) = version {
                (
                    format!("workspace, name, version, {key_column}, value"),
                    vec![
                        Val::Text(workspace.into()),
                        Val::Text(name.into()),
                        Val::Text(version.into()),
                        Val::Text(key.into()),
                        Val::Text(value.into()),
                    ],
                )
            } else {
                (
                    format!("workspace, name, {key_column}, value"),
                    vec![
                        Val::Text(workspace.into()),
                        Val::Text(name.into()),
                        Val::Text(key.into()),
                        Val::Text(value.into()),
                    ],
                )
            };
            let placeholders = (1..=values.len())
                .map(|index| d.placeholder(index))
                .collect::<Vec<_>>()
                .join(", ");
            self.db()
                .exec(
                    &format!("INSERT INTO {table} ({columns}) VALUES ({placeholders})"),
                    &values,
                )
                .await
                .map_err(internal)?;
        }
        Ok(())
    }

    async fn delete_mcp_tag_row(
        &self,
        table: &str,
        workspace: &str,
        name: &str,
        version: Option<&str>,
        key: &str,
    ) -> Result<bool, MlflowError> {
        let d = self.db().dialect();
        let key_column = d.quote_ident("key");
        let mut values = vec![Val::Text(workspace.into()), Val::Text(name.into())];
        let mut sql = format!(
            "DELETE FROM {table} WHERE workspace = {} AND name = {}",
            d.placeholder(1),
            d.placeholder(2)
        );
        if let Some(version) = version {
            values.push(Val::Text(version.into()));
            sql.push_str(&format!(" AND version = {}", d.placeholder(values.len())));
        }
        values.push(Val::Text(key.into()));
        sql.push_str(&format!(
            " AND {key_column} = {}",
            d.placeholder(values.len())
        ));
        self.db()
            .exec(&sql, &values)
            .await
            .map(|affected| affected > 0)
            .map_err(internal)
    }

    async fn delete_alias_endpoints(
        &self,
        workspace: &str,
        name: &str,
        alias: &str,
    ) -> Result<(), MlflowError> {
        let d = self.db().dialect();
        self.db()
            .exec(
                &format!(
                    "DELETE FROM mcp_access_endpoints WHERE workspace = {} AND server_name = {} \
                     AND server_alias = {}",
                    d.placeholder(1),
                    d.placeholder(2),
                    d.placeholder(3)
                ),
                &[
                    Val::Text(workspace.into()),
                    Val::Text(name.into()),
                    Val::Text(alias.into()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }
}

fn map_server_base(row: &dyn RowLike) -> Result<McpServer, sqlx::Error> {
    Ok(McpServer {
        name: row.get_string("name")?,
        display_name: row.get_opt_string("display_name")?,
        description: row.get_opt_string("description")?,
        icons: json_array(row.get_opt_json("icons")?),
        workspace: row.get_string("workspace")?,
        status: None,
        access_endpoints: Vec::new(),
        latest_version: None,
        aliases: BTreeMap::new(),
        tags: BTreeMap::new(),
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        creation_timestamp: row.get_i64("created_at")?,
        last_updated_timestamp: row.get_i64("last_updated_at")?,
    })
}

fn map_version_base(row: &dyn RowLike) -> Result<McpServerVersion, sqlx::Error> {
    let major = row.get_int("version_major")? as u32;
    let minor = row.get_int("version_minor")? as u32;
    let patch = row.get_int("version_patch")? as u32;
    let prerelease_key = row.get_string("version_prerelease_sort_key")?;
    let status = McpStatus::parse(&row.get_string("status")?)
        .map_err(|error| sqlx::Error::Decode(error.message.into()))?;
    let server_json = row.get_opt_json("server_json")?.unwrap_or(Value::Null);
    let version = row.get_string("version")?;
    let parsed = parse_semver(&version, "version")
        .map_err(|error| sqlx::Error::Decode(error.message.into()))?;
    Ok(McpServerVersion {
        name: row.get_string("name")?,
        version,
        server_json,
        display_name: row.get_opt_string("display_name")?,
        workspace: row.get_string("workspace")?,
        status,
        tools: json_array(row.get_opt_json("tools")?),
        aliases: Vec::new(),
        tags: BTreeMap::new(),
        source: row.get_opt_string("source")?,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        creation_timestamp: row.get_i64("created_at")?,
        last_updated_timestamp: row.get_i64("last_updated_at")?,
        semver: SemVer {
            major,
            minor,
            patch,
            prerelease: parsed.prerelease,
            prerelease_key,
        },
    })
}

#[derive(Debug, Clone)]
struct EndpointBase {
    id: String,
    workspace: String,
    server_name: String,
    server_version: Option<String>,
    server_alias: Option<String>,
    url: String,
    transport_type: McpTransportType,
    created_by: Option<String>,
    last_updated_by: Option<String>,
    creation_timestamp: i64,
    last_updated_timestamp: i64,
}

fn map_endpoint_base(row: &dyn RowLike) -> Result<EndpointBase, sqlx::Error> {
    let transport_type = McpTransportType::parse(&row.get_string("transport_type")?)
        .map_err(|error| sqlx::Error::Decode(error.message.into()))?;
    Ok(EndpointBase {
        id: row.get_string("id")?,
        workspace: row.get_string("workspace")?,
        server_name: row.get_string("server_name")?,
        server_version: row.get_opt_string("server_version")?,
        server_alias: row.get_opt_string("server_alias")?,
        url: row.get_string("url")?,
        transport_type,
        created_by: row.get_opt_string("created_by")?,
        last_updated_by: row.get_opt_string("last_updated_by")?,
        creation_timestamp: row.get_i64("created_at")?,
        last_updated_timestamp: row.get_i64("last_updated_at")?,
    })
}

fn json_array(value: Option<Value>) -> Option<Vec<Value>> {
    value.and_then(|value| value.as_array().cloned())
}

fn validate_name(name: &str) -> Result<(), MlflowError> {
    if name.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "MCP server name must not be empty",
        ));
    }
    let Some((namespace, slug)) = name.split_once('/') else {
        return Err(invalid_name());
    };
    if namespace.is_empty()
        || slug.is_empty()
        || slug.contains('/')
        || ["aliases", "endpoints", "tags", "versions"].contains(&slug)
        || !valid_name_part(namespace, true)
        || !valid_name_part(slug, false)
    {
        return Err(invalid_name());
    }
    Ok(())
}

fn valid_name_part(value: &str, namespace: bool) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 2
        && bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.last().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.iter().all(|byte| {
            byte.is_ascii_alphanumeric()
                || *byte == b'.'
                || *byte == b'-'
                || (!namespace && *byte == b'_')
        })
}

fn invalid_name() -> MlflowError {
    MlflowError::invalid_parameter_value(
        "Invalid MCP server name. Expected '<reverse-dns namespace>/<server slug>' such as \
         'com.example/server-name'.",
    )
}

fn parse_semver(version: &str, param: &str) -> Result<SemVer, MlflowError> {
    if version.len() > MAX_SEMVER_LENGTH {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid semantic version for {param}: '{version}' (maximum length is \
             {MAX_SEMVER_LENGTH} characters)"
        )));
    }
    let without_build = match version.split_once('+') {
        Some((left, build)) if valid_identifiers(build, false) => left,
        Some(_) => return Err(invalid_semver(param, version)),
        None => version,
    };
    let (core, prerelease) = match without_build.split_once('-') {
        Some((core, pre)) if valid_identifiers(pre, true) => (core, pre),
        Some(_) => return Err(invalid_semver(param, version)),
        None => (without_build, ""),
    };
    let components = core.split('.').collect::<Vec<_>>();
    if components.len() != 3 {
        return Err(invalid_semver(param, version));
    }
    let mut parsed = Vec::new();
    for (label, value) in ["major", "minor", "patch"].into_iter().zip(components) {
        if value.is_empty()
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || (value.len() > 1 && value.starts_with('0'))
        {
            return Err(invalid_semver(param, version));
        }
        let number = value
            .parse::<u64>()
            .map_err(|_| invalid_semver(param, version))?;
        if number > u64::from(MAX_SEMVER_CORE) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid semantic version for {param}: '{version}' ({label} must be <= \
                 {MAX_SEMVER_CORE})"
            )));
        }
        parsed.push(number as u32);
    }
    let prerelease = if prerelease.is_empty() {
        Vec::new()
    } else {
        prerelease.split('.').map(str::to_string).collect()
    };
    let prerelease_key = encode_prerelease(&prerelease);
    Ok(SemVer {
        major: parsed[0],
        minor: parsed[1],
        patch: parsed[2],
        prerelease,
        prerelease_key,
    })
}

fn valid_identifiers(value: &str, prerelease: bool) -> bool {
    !value.is_empty()
        && value.split('.').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                && (!prerelease
                    || !part.bytes().all(|byte| byte.is_ascii_digit())
                    || part == "0"
                    || !part.starts_with('0'))
        })
}

fn invalid_semver(param: &str, version: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Invalid semantic version for {param}: '{version}'"
    ))
}

fn encode_prerelease(parts: &[String]) -> String {
    if parts.is_empty() {
        return "2".to_string();
    }
    parts
        .iter()
        .map(|part| {
            if part.bytes().all(|byte| byte.is_ascii_digit()) {
                format!("0{:03}{part}", part.len())
            } else {
                let encoded = part
                    .bytes()
                    .map(|byte| format!("{byte:03}"))
                    .collect::<String>();
                format!("1{encoded}000")
            }
        })
        .collect()
}

fn latest_version(mut versions: Vec<McpServerVersion>) -> Option<McpServerVersion> {
    versions.sort_by(latest_cmp);
    versions.into_iter().next()
}

fn latest_cmp(left: &McpServerVersion, right: &McpServerVersion) -> Ordering {
    let left_priority = if left.status == McpStatus::Active {
        0
    } else {
        1
    };
    let right_priority = if right.status == McpStatus::Active {
        0
    } else {
        1
    };
    left_priority
        .cmp(&right_priority)
        .then_with(|| semver_cmp(&right.semver, &left.semver))
        .then_with(|| right.creation_timestamp.cmp(&left.creation_timestamp))
        .then_with(|| right.version.cmp(&left.version))
}

fn semver_cmp(left: &SemVer, right: &SemVer) -> Ordering {
    left.major
        .cmp(&right.major)
        .then_with(|| left.minor.cmp(&right.minor))
        .then_with(|| left.patch.cmp(&right.patch))
        .then_with(|| left.prerelease_key.cmp(&right.prerelease_key))
}

fn validate_status_transition(current: McpStatus, new: McpStatus) -> Result<(), MlflowError> {
    let allowed: &[McpStatus] = match current {
        McpStatus::Draft => &[McpStatus::Active, McpStatus::Deleted],
        McpStatus::Active => &[McpStatus::Draft, McpStatus::Deprecated],
        McpStatus::Deprecated => &[McpStatus::Active, McpStatus::Deleted],
        McpStatus::Deleted => &[],
    };
    if allowed.contains(&new) {
        return Ok(());
    }
    let mut names = allowed
        .iter()
        .map(|status| status.as_str())
        .collect::<Vec<_>>();
    names.sort_unstable();
    Err(MlflowError::invalid_parameter_value(format!(
        "Invalid status transition from '{}' to '{}'. Allowed transitions: [{}]",
        current.as_str(),
        new.as_str(),
        names
            .into_iter()
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

fn validate_exactly_one(version: Option<&str>, alias: Option<&str>) -> Result<(), MlflowError> {
    if version.is_some() == alias.is_some() {
        Err(MlflowError::invalid_parameter_value(
            "Exactly one of server_version or server_alias must be provided",
        ))
    } else {
        Ok(())
    }
}

fn validate_endpoint_url(url: &str) -> Result<(), MlflowError> {
    if url.trim().is_empty() {
        Err(MlflowError::invalid_parameter_value(format!(
            "MCP access endpoint url cannot be empty or just whitespace: {url:?}"
        )))
    } else {
        Ok(())
    }
}

fn validate_icons(icons: Option<&[Value]>, field: &str) -> Result<(), MlflowError> {
    let Some(icons) = icons else { return Ok(()) };
    if icons.len() > 100 {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid {field}. It must contain at most 100 items."
        )));
    }
    for (index, icon) in icons.iter().enumerate() {
        let object = icon.as_object().ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid {field}[{index}]. Expected an object."
            ))
        })?;
        let src = object.get("src").and_then(Value::as_str).ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid {field}[{index}]. Missing required key 'src'."
            ))
        })?;
        validate_icon_url(src)?;
        if let Some(mime) = object.get("mimeType") {
            let mime = mime.as_str().ok_or_else(|| {
                MlflowError::invalid_parameter_value(format!(
                    "Invalid icon mimeType {mime:?}. Allowed values must use the 'image/*' media type."
                ))
            })?;
            let normalized = mime.trim().to_ascii_lowercase();
            if !normalized.starts_with("image/") || normalized == "image/" {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid icon mimeType {mime:?}. Allowed values must use the 'image/*' media type."
                )));
            }
        }
    }
    Ok(())
}

fn validate_icon_url(value: &str) -> Result<(), MlflowError> {
    if value.trim().is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Icon URL cannot be empty or just whitespace: {value:?}"
        )));
    }
    let parsed = Url::parse(value).map_err(|error| {
        MlflowError::invalid_parameter_value(format!("Invalid Icon URL {value:?}: {error:?}"))
    })?;
    let schemes = std::env::var("MLFLOW_ICON_URL_ALLOWED_SCHEMES")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .split(',')
                .map(|scheme| scheme.trim().to_ascii_lowercase())
                .filter(|scheme| !scheme.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["https".to_string()]);
    if !schemes.iter().any(|scheme| scheme == parsed.scheme()) {
        let allowed = schemes
            .iter()
            .map(|scheme| format!("'{scheme}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid Icon URL scheme: {:?}. Allowed schemes are: {allowed}.",
            parsed.scheme()
        )));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Icon URL must not include embedded credentials: {value:?}"
        )));
    }
    let hostname = parsed.host_str().ok_or_else(|| {
        MlflowError::invalid_parameter_value(format!("Icon URL must include a hostname: {value:?}"))
    })?;
    let allow_private = matches!(
        std::env::var("MLFLOW_ICON_URL_ALLOW_PRIVATE_IPS")
            .ok()
            .as_deref(),
        Some("true" | "True" | "TRUE" | "1")
    );
    if !allow_private {
        let addresses = (hostname, 0_u16).to_socket_addrs().map_err(|error| {
            MlflowError::invalid_parameter_value(format!(
                "Cannot resolve Icon URL hostname {hostname:?}: {error}"
            ))
        })?;
        for address in addresses {
            if !is_global_ip(address.ip()) {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Icon URL must not resolve to a non-public IP address. {hostname:?} resolves to {}.",
                    address.ip()
                )));
            }
        }
    }
    if let Ok(domains) = std::env::var("MLFLOW_ICON_URL_ALLOWED_DOMAINS") {
        let patterns = domains
            .split(',')
            .map(str::trim)
            .filter(|pattern| !pattern.is_empty())
            .collect::<Vec<_>>();
        if !patterns.is_empty()
            && !patterns.iter().any(|pattern| {
                glob_matches(
                    &hostname.to_ascii_lowercase(),
                    &pattern.to_ascii_lowercase(),
                )
            })
        {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Icon URL host {hostname:?} is not in the allowed domain list: {}.",
                patterns.join(", ")
            )));
        }
    }
    Ok(())
}

fn glob_matches(value: &str, pattern: &str) -> bool {
    let (value, pattern) = (value.as_bytes(), pattern.as_bytes());
    let mut matches = vec![false; value.len() + 1];
    matches[0] = true;
    for token in pattern {
        if *token == b'*' {
            for index in 1..=value.len() {
                matches[index] |= matches[index - 1];
            }
        } else {
            for index in (1..=value.len()).rev() {
                matches[index] =
                    matches[index - 1] && (*token == b'?' || *token == value[index - 1]);
            }
            matches[0] = false;
        }
    }
    matches[value.len()]
}

fn is_global_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(value) => {
            let octets = value.octets();
            !(value.is_loopback()
                || value.is_private()
                || value.is_link_local()
                || value.is_broadcast()
                || value.is_documentation()
                || value.is_unspecified()
                || value.is_multicast()
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                || (octets[0] == 198 && matches!(octets[1], 18 | 19))
                || octets[0] >= 240)
        }
        IpAddr::V6(value) => {
            let first = value.segments()[0];
            !(value.is_loopback()
                || value.is_unspecified()
                || value.is_multicast()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80)
                && value
                    .to_ipv4_mapped()
                    .is_none_or(|mapped| is_global_ip(IpAddr::V4(mapped)))
        }
    }
}

fn validate_tools(tools: Option<&[Value]>) -> Result<(), MlflowError> {
    let Some(tools) = tools else { return Ok(()) };
    if tools.len() > 1000 {
        return Err(MlflowError::invalid_parameter_value(
            "Invalid tools. It must contain at most 1000 items.",
        ));
    }
    for (index, tool) in tools.iter().enumerate() {
        let object = tool.as_object().ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid tools[{index}]. Expected an object."
            ))
        })?;
        if object.get("name").and_then(Value::as_str).is_none() {
            return Err(MlflowError::invalid_parameter_value(
                "Missing required key 'name' in MCPTool dictionary",
            ));
        }
        validate_icons(
            object
                .get("icons")
                .and_then(Value::as_array)
                .map(Vec::as_slice),
            &format!("tools[{index}].icons"),
        )?;
    }
    Ok(())
}

fn validate_max_results(value: i32, max: i32) -> Result<(), MlflowError> {
    if value < 1 {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value {value} for parameter 'max_results' supplied. It must be a positive integer"
        )));
    }
    if value > max {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid value for max_results. It must be at most {max}, but got {value}"
        )));
    }
    Ok(())
}

fn page_offset(token: Option<&str>) -> Result<i64, MlflowError> {
    mlflow_search::parse_start_offset_from_page_token(token)
        .map_err(|error| MlflowError::invalid_parameter_value(error.message))
}

fn page<T, P>(
    mut values: Vec<T>,
    offset: i64,
    max: i32,
    build: impl FnOnce(Vec<T>, Option<String>) -> P,
) -> P {
    let offset = usize::try_from(offset).unwrap_or(usize::MAX);
    let mut values = if offset < values.len() {
        values.split_off(offset)
    } else {
        Vec::new()
    };
    let has_next = values.len() > max as usize;
    values.truncate(max as usize);
    let token = has_next.then(|| mlflow_search::create_page_token(offset as i64 + i64::from(max)));
    build(values, token)
}

fn version_not_found(name: &str, version: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!(
        "MCP server version '{name}' version '{version}' not found"
    ))
}

fn endpoint_not_found(id: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("MCPAccessEndpoint {id} not found"))
}

fn server_matches(server: &McpServer, filters: &[Comparison]) -> bool {
    filters.iter().all(|filter| {
        if filter.entity_type == "tag" {
            return server
                .tags
                .get(&filter.key)
                .is_some_and(|value| compare_string(value, filter));
        }
        match filter.key.as_str() {
            "name" => compare_string(&server.name, filter),
            "display_name" => server
                .display_name
                .as_deref()
                .is_some_and(|value| compare_string(value, filter)),
            "status" => server
                .status
                .is_some_and(|value| compare_string(value.as_str(), filter)),
            "has_access_endpoints" => compare_string(
                if server.access_endpoints.is_empty() {
                    "false"
                } else {
                    "true"
                },
                filter,
            ),
            "created_at" => compare_i64(server.creation_timestamp, filter),
            "last_updated_at" => compare_i64(server.last_updated_timestamp, filter),
            _ => false,
        }
    })
}

fn version_matches(version: &McpServerVersion, filters: &[Comparison]) -> bool {
    filters.iter().all(|filter| match filter.key.as_str() {
        "name" => compare_string(&version.name, filter),
        "status" => compare_string(version.status.as_str(), filter),
        "created_at" => compare_i64(version.creation_timestamp, filter),
        "last_updated_at" => compare_i64(version.last_updated_timestamp, filter),
        "version" => match filter.comparator.as_str() {
            "=" | "!=" => compare_string(&version.version, filter),
            comparator => search_string(filter)
                .and_then(|target| parse_semver(target, "filter_string version").ok())
                .is_some_and(|target| {
                    compare_order(semver_cmp(&version.semver, &target), comparator)
                }),
        },
        _ => false,
    })
}

fn endpoint_matches(endpoint: &McpAccessEndpoint, filters: &[Comparison]) -> bool {
    filters.iter().all(|filter| match filter.key.as_str() {
        "status" => compare_string(endpoint.resolved_version.status.as_str(), filter),
        "server_name" => compare_string(&endpoint.server_name, filter),
        "transport_type" => compare_string(endpoint.transport_type.as_str(), filter),
        "created_at" => compare_i64(endpoint.creation_timestamp, filter),
        "last_updated_at" => compare_i64(endpoint.last_updated_timestamp, filter),
        _ => false,
    })
}

fn compare_string(actual: &str, filter: &Comparison) -> bool {
    match (&filter.comparator[..], &filter.value) {
        ("IN", SearchValue::List(values)) => values.iter().any(|value| value == actual),
        ("=", SearchValue::Str(value)) => actual == value,
        ("!=", SearchValue::Str(value)) => actual != value,
        (">", SearchValue::Str(value)) => actual > value.as_str(),
        (">=", SearchValue::Str(value)) => actual >= value.as_str(),
        ("<", SearchValue::Str(value)) => actual < value.as_str(),
        ("<=", SearchValue::Str(value)) => actual <= value.as_str(),
        ("LIKE", SearchValue::Str(value)) => sql_like(actual, value, false),
        ("ILIKE", SearchValue::Str(value)) => sql_like(actual, value, true),
        _ => false,
    }
}

fn compare_i64(actual: i64, filter: &Comparison) -> bool {
    let Some(value) = search_string(filter).and_then(|value| value.parse::<i64>().ok()) else {
        return false;
    };
    compare_order(actual.cmp(&value), &filter.comparator)
}

fn compare_order(ordering: Ordering, comparator: &str) -> bool {
    match comparator {
        "=" => ordering == Ordering::Equal,
        "!=" => ordering != Ordering::Equal,
        ">" => ordering == Ordering::Greater,
        ">=" => ordering != Ordering::Less,
        "<" => ordering == Ordering::Less,
        "<=" => ordering != Ordering::Greater,
        _ => false,
    }
}

fn search_string(filter: &Comparison) -> Option<&str> {
    match &filter.value {
        SearchValue::Str(value) => Some(value),
        _ => None,
    }
}

fn sql_like(actual: &str, pattern: &str, insensitive: bool) -> bool {
    let actual = if insensitive {
        actual.to_lowercase()
    } else {
        actual.to_string()
    };
    let pattern = if insensitive {
        pattern.to_lowercase()
    } else {
        pattern.to_string()
    };
    like_bytes(actual.as_bytes(), pattern.as_bytes())
}

fn like_bytes(actual: &[u8], pattern: &[u8]) -> bool {
    match pattern.split_first() {
        None => actual.is_empty(),
        Some((b'%', rest)) => {
            like_bytes(actual, rest)
                || actual
                    .split_first()
                    .is_some_and(|(_, tail)| like_bytes(tail, pattern))
        }
        Some((b'_', rest)) => actual
            .split_first()
            .is_some_and(|(_, tail)| like_bytes(tail, rest)),
        Some((head, rest)) => actual
            .split_first()
            .is_some_and(|(value, tail)| value == head && like_bytes(tail, rest)),
    }
}

fn parse_order_by(values: &[String], valid: &[&str]) -> Result<Vec<(String, bool)>, MlflowError> {
    let mut result = Vec::new();
    for raw in values {
        let pieces = raw.split_whitespace().collect::<Vec<_>>();
        if pieces.is_empty() || pieces.len() > 2 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid order_by clause '{raw}'. Could not be parsed."
            )));
        }
        let key = pieces[0].trim_matches('`').to_string();
        if !valid.contains(&key.as_str()) {
            let mut sorted = valid.to_vec();
            sorted.sort_unstable();
            let valid_keys = sorted
                .into_iter()
                .map(|value| format!("'{value}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid order_by key '{key}'. Valid keys: [{valid_keys}]"
            )));
        }
        if result.iter().any(|(observed, _)| observed == &key) {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Duplicate order_by field: '{key}'"
            )));
        }
        let ascending = match pieces.get(1).map(|value| value.to_ascii_lowercase()) {
            None => true,
            Some(value) if value == "asc" => true,
            Some(value) if value == "desc" => false,
            _ => {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid ordering key in order_by clause '{raw}'."
                )))
            }
        };
        result.push((key, ascending));
    }
    Ok(result)
}

fn sort_servers(values: &mut [McpServer], order_by: &[String]) -> Result<(), MlflowError> {
    let mut order = parse_order_by(order_by, &["name", "created_at", "last_updated_at"])?;
    if !order.iter().any(|(key, _)| key == "name") {
        order.push(("name".into(), true));
    }
    values.sort_by(|left, right| {
        chain_order(&order, |key| match key {
            "name" => left.name.cmp(&right.name),
            "created_at" => left.creation_timestamp.cmp(&right.creation_timestamp),
            "last_updated_at" => left
                .last_updated_timestamp
                .cmp(&right.last_updated_timestamp),
            _ => Ordering::Equal,
        })
    });
    Ok(())
}

fn sort_versions(values: &mut [McpServerVersion], order_by: &[String]) -> Result<(), MlflowError> {
    let mut order = parse_order_by(order_by, &["version", "created_at", "last_updated_at"])?;
    if !order.iter().any(|(key, _)| key == "created_at") {
        order.push(("created_at".into(), true));
    }
    if !order.iter().any(|(key, _)| key == "version") {
        order.push(("raw_version".into(), true));
    }
    values.sort_by(|left, right| {
        chain_order(&order, |key| match key {
            "version" => semver_cmp(&left.semver, &right.semver),
            "raw_version" => left.version.cmp(&right.version),
            "created_at" => left.creation_timestamp.cmp(&right.creation_timestamp),
            "last_updated_at" => left
                .last_updated_timestamp
                .cmp(&right.last_updated_timestamp),
            _ => Ordering::Equal,
        })
    });
    Ok(())
}

fn sort_endpoints(
    values: &mut [McpAccessEndpoint],
    order_by: &[String],
) -> Result<(), MlflowError> {
    let mut order = parse_order_by(
        order_by,
        &["id", "server_name", "created_at", "last_updated_at"],
    )?;
    if !order.iter().any(|(key, _)| key == "id") {
        order.push(("id".into(), true));
    }
    values.sort_by(|left, right| {
        chain_order(&order, |key| match key {
            "id" => left.id.cmp(&right.id),
            "server_name" => left.server_name.cmp(&right.server_name),
            "created_at" => left.creation_timestamp.cmp(&right.creation_timestamp),
            "last_updated_at" => left
                .last_updated_timestamp
                .cmp(&right.last_updated_timestamp),
            _ => Ordering::Equal,
        })
    });
    Ok(())
}

fn chain_order(order: &[(String, bool)], compare: impl Fn(&str) -> Ordering) -> Ordering {
    for (key, ascending) in order {
        let value = compare(key);
        if value != Ordering::Equal {
            return if *ascending { value } else { value.reverse() };
        }
    }
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_parser_and_ordering_cover_prerelease_rules() {
        let alpha2 = parse_semver("1.0.0-alpha.2", "version").unwrap();
        let alpha10 = parse_semver("1.0.0-alpha.10", "version").unwrap();
        let release = parse_semver("1.0.0", "version").unwrap();
        assert_eq!(semver_cmp(&alpha2, &alpha10), Ordering::Less);
        assert_eq!(semver_cmp(&alpha10, &release), Ordering::Less);
        assert!(parse_semver("01.0.0", "version").is_err());
        assert!(parse_semver("1.0.0-01", "version").is_err());
    }

    #[test]
    fn validates_names() {
        assert!(validate_name("com.example/server_name-1").is_ok());
        assert!(validate_name("server").is_err());
        assert!(validate_name("com.example/versions").is_err());
    }
}
