//! MCP server registry HTTP API (`mlflow/server/mcp_server_api.py`).

use std::collections::BTreeMap;

use axum::body::{Body, Bytes};
use axum::extract::{Extension, OriginalUri, Path, RawQuery, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::Router;
use mlflow_error::MlflowError;
use mlflow_store::{
    McpAccessEndpoint, McpPatch, McpServer, McpServerVersion, McpStatus, McpTransportType,
};
use serde_json::{json, Map, Value};

use crate::auth_middleware::AuthContext;
use crate::state::AppState;
use crate::workspace::Workspace;

pub const API_PREFIX: &str = "/api/3.0/mlflow/mcp-servers";
pub const AJAX_PREFIX: &str = "/ajax-api/3.0/mlflow/mcp-servers";

pub fn routes() -> Router<AppState> {
    Router::new()
        .merge(routes_for(API_PREFIX))
        .merge(routes_for(AJAX_PREFIX))
        .fallback(encoded_path_dispatch)
}

fn routes_for(prefix: &str) -> Router<AppState> {
    Router::new()
        .route(prefix, post(create_server).get(search_servers))
        .route(&format!("{prefix}/endpoints"), get(search_all_endpoints))
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/versions"),
            post(create_version).get(search_versions),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/versions/{{version}}"),
            get(get_version)
                .patch(update_version)
                .delete(delete_version),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/versions/{{version}}/tags"),
            post(set_version_tag),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/versions/{{version}}/tags/{{key}}"),
            delete(delete_version_tag),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/endpoints"),
            post(create_endpoint).get(search_server_endpoints),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/endpoints/{{endpoint_id}}"),
            get(get_endpoint)
                .patch(update_endpoint)
                .delete(delete_endpoint),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/tags"),
            post(set_server_tag),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/tags/{{key}}"),
            delete(delete_server_tag),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/aliases"),
            post(set_alias),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}/aliases/{{alias}}"),
            get(get_by_alias).delete(delete_alias),
        )
        .route(
            &format!("{prefix}/{{namespace}}/{{slug}}"),
            get(get_server).patch(update_server).delete(delete_server),
        )
}

type ServerPath = (String, String);
type VersionPath = (String, String, String);
type VersionTagPath = (String, String, String, String);
type EndpointPath = (String, String, String);
type ServerChildPath = (String, String, String);

async fn encoded_path_dispatch(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    RawQuery(query): RawQuery,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let Some(tail) = [API_PREFIX, AJAX_PREFIX]
        .iter()
        .find_map(|prefix| uri.path().strip_prefix(prefix))
    else {
        return Ok(StatusCode::NOT_FOUND.into_response());
    };
    let tail = crate::proto_http::percent_decode_path(tail.trim_matches('/'));
    let parts = tail.split('/').collect::<Vec<_>>();
    if parts.len() < 2 {
        return Ok(StatusCode::NOT_FOUND.into_response());
    }
    let server = (parts[0].to_string(), parts[1].to_string());
    let rest = &parts[2..];
    match (method.as_str(), rest) {
        ("GET", []) => get_server(State(state), workspace, Path(server)).await,
        ("PATCH", []) => update_server(State(state), workspace, auth, Path(server), body).await,
        ("DELETE", []) => delete_server(State(state), workspace, Path(server)).await,
        ("POST", ["versions"]) => {
            create_version(State(state), workspace, auth, Path(server), body).await
        }
        ("GET", ["versions"]) => {
            search_versions(State(state), workspace, Path(server), RawQuery(query)).await
        }
        ("GET", ["versions", version]) => {
            get_version(
                State(state),
                workspace,
                Path((server.0, server.1, (*version).into())),
            )
            .await
        }
        ("PATCH", ["versions", version]) => {
            update_version(
                State(state),
                workspace,
                auth,
                Path((server.0, server.1, (*version).into())),
                body,
            )
            .await
        }
        ("DELETE", ["versions", version]) => {
            delete_version(
                State(state),
                workspace,
                Path((server.0, server.1, (*version).into())),
            )
            .await
        }
        ("POST", ["versions", version, "tags"]) => {
            set_version_tag(
                State(state),
                workspace,
                Path((server.0, server.1, (*version).into())),
                body,
            )
            .await
        }
        ("DELETE", ["versions", version, "tags", key @ ..]) if !key.is_empty() => {
            delete_version_tag(
                State(state),
                workspace,
                Path((server.0, server.1, (*version).into(), key.join("/"))),
            )
            .await
        }
        ("POST", ["endpoints"]) => {
            create_endpoint(State(state), workspace, auth, Path(server), body).await
        }
        ("GET", ["endpoints"]) => {
            search_server_endpoints(State(state), workspace, Path(server), RawQuery(query)).await
        }
        ("GET", ["endpoints", endpoint]) => {
            get_endpoint(
                State(state),
                workspace,
                Path((server.0, server.1, (*endpoint).into())),
            )
            .await
        }
        ("PATCH", ["endpoints", endpoint]) => {
            update_endpoint(
                State(state),
                workspace,
                auth,
                Path((server.0, server.1, (*endpoint).into())),
                body,
            )
            .await
        }
        ("DELETE", ["endpoints", endpoint]) => {
            delete_endpoint(
                State(state),
                workspace,
                Path((server.0, server.1, (*endpoint).into())),
            )
            .await
        }
        ("POST", ["tags"]) => set_server_tag(State(state), workspace, Path(server), body).await,
        ("DELETE", ["tags", key @ ..]) if !key.is_empty() => {
            delete_server_tag(
                State(state),
                workspace,
                Path((server.0, server.1, key.join("/"))),
            )
            .await
        }
        ("POST", ["aliases"]) => set_alias(State(state), workspace, Path(server), body).await,
        ("GET", ["aliases", alias @ ..]) if !alias.is_empty() => {
            get_by_alias(
                State(state),
                workspace,
                Path((server.0, server.1, alias.join("/"))),
            )
            .await
        }
        ("DELETE", ["aliases", alias @ ..]) if !alias.is_empty() => {
            delete_alias(
                State(state),
                workspace,
                Path((server.0, server.1, alias.join("/"))),
            )
            .await
        }
        _ => Ok(StatusCode::NOT_FOUND.into_response()),
    }
}

pub async fn create_server(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    let name = required_string(object, "name")?;
    let description = optional_string(object, "description")?;
    let icons = normalize_icons(optional_array(object, "icons")?, "icons")?;
    let server = state
        .tracking_store()
        .create_mcp_server(
            workspace.name(),
            name,
            description,
            icons,
            username(auth.as_ref()),
        )
        .await?;
    json_response(server_json(server))
}

pub async fn search_servers(
    State(state): State<AppState>,
    workspace: Workspace,
    RawQuery(query): RawQuery,
) -> Result<Response, MlflowError> {
    let query = QueryArgs::parse(query.as_deref())?;
    let page = state
        .tracking_store()
        .search_mcp_servers(
            workspace.name(),
            query.one("filter_string"),
            query.max_results()?,
            &query.many("order_by"),
            query.one("page_token"),
        )
        .await?;
    json_response(json!({
        "mcp_servers": page.items.into_iter().map(server_json).collect::<Vec<_>>(),
        "next_page_token": page.next_page_token,
    }))
}

pub async fn get_server(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerPath>,
) -> Result<Response, MlflowError> {
    let name = server_name(&path.0, &path.1);
    json_response(server_json(
        state
            .tracking_store()
            .get_mcp_server(workspace.name(), &name)
            .await?,
    ))
}

pub async fn update_server(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    Path(path): Path<ServerPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    if object.contains_key("latest_version") {
        return Err(invalid_request(
            "Value error, latest_version is read-only; it is resolved automatically from \
             semantic-version ordering",
        ));
    }
    let name = server_name(&path.0, &path.1);
    let server = state
        .tracking_store()
        .update_mcp_server(
            workspace.name(),
            &name,
            patch_string(object, "description")?,
            patch_string(object, "display_name")?,
            patch_icons(object, "icons")?,
            username(auth.as_ref()),
        )
        .await?;
    json_response(server_json(server))
}

pub async fn delete_server(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerPath>,
) -> Result<Response, MlflowError> {
    state
        .tracking_store()
        .delete_mcp_server(workspace.name(), &server_name(&path.0, &path.1))
        .await?;
    empty_response()
}

pub async fn create_version(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    Path(path): Path<ServerPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    let mut server_json_value = object
        .get("server_json")
        .cloned()
        .ok_or_else(|| missing("server_json"))?;
    if !server_json_value.is_object() {
        return Err(invalid_request(
            "server_json: Input should be a valid dictionary",
        ));
    }
    normalize_server_json(&mut server_json_value)?;
    let name = server_name(&path.0, &path.1);
    let embedded_name = server_json_value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_request("server_json.name: Field required"))?;
    if embedded_name != name {
        return Err(MlflowError::invalid_parameter_value(format!(
            "server_json.name '{embedded_name}' does not match path parameter '{name}'"
        )));
    }
    let status = optional_string(object, "status")?.unwrap_or("draft");
    let status = McpStatus::parse(status)?;
    let version = state
        .tracking_store()
        .create_mcp_server_version(
            workspace.name(),
            server_json_value,
            optional_string(object, "display_name")?,
            optional_string(object, "source")?,
            status,
            normalize_tools(optional_array(object, "tools")?)?,
            username(auth.as_ref()),
        )
        .await?;
    json_response(version_json(version))
}

pub async fn get_version(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<VersionPath>,
) -> Result<Response, MlflowError> {
    let name = server_name(&path.0, &path.1);
    json_response(version_json(
        state
            .tracking_store()
            .get_mcp_server_version(workspace.name(), &name, &path.2)
            .await?,
    ))
}

pub async fn search_versions(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerPath>,
    RawQuery(query): RawQuery,
) -> Result<Response, MlflowError> {
    let query = QueryArgs::parse(query.as_deref())?;
    let page = state
        .tracking_store()
        .search_mcp_server_versions(
            workspace.name(),
            &server_name(&path.0, &path.1),
            query.one("filter_string"),
            query.max_results()?,
            &query.many("order_by"),
            query.one("page_token"),
        )
        .await?;
    json_response(json!({
        "mcp_server_versions": page.items.into_iter().map(version_json).collect::<Vec<_>>(),
        "next_page_token": page.next_page_token,
    }))
}

pub async fn update_version(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    Path(path): Path<VersionPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    let status = match object.get("status") {
        None => McpPatch::Unset,
        Some(Value::Null) => {
            return Err(MlflowError::invalid_parameter_value(
                "status cannot be null; omit the field to leave it unchanged",
            ))
        }
        Some(Value::String(value)) => McpPatch::Set(McpStatus::parse(value)?),
        Some(_) => return Err(invalid_request("status: Input should be a valid string")),
    };
    let name = server_name(&path.0, &path.1);
    let version = state
        .tracking_store()
        .update_mcp_server_version(
            workspace.name(),
            &name,
            &path.2,
            patch_string(object, "display_name")?,
            status,
            patch_tools(object, "tools")?,
            username(auth.as_ref()),
        )
        .await?;
    json_response(version_json(version))
}

pub async fn delete_version(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<VersionPath>,
) -> Result<Response, MlflowError> {
    state
        .tracking_store()
        .delete_mcp_server_version(workspace.name(), &server_name(&path.0, &path.1), &path.2)
        .await?;
    empty_response()
}

pub async fn set_version_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<VersionPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    state
        .tracking_store()
        .set_mcp_server_version_tag(
            workspace.name(),
            &server_name(&path.0, &path.1),
            &path.2,
            required_string(object, "key")?,
            required_string(object, "value")?,
        )
        .await?;
    empty_response()
}

pub async fn delete_version_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<VersionTagPath>,
) -> Result<Response, MlflowError> {
    state
        .tracking_store()
        .delete_mcp_server_version_tag(
            workspace.name(),
            &server_name(&path.0, &path.1),
            &path.2,
            &path.3,
        )
        .await?;
    empty_response()
}

pub async fn create_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    Path(path): Path<ServerPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    let endpoint = state
        .tracking_store()
        .create_mcp_access_endpoint(
            workspace.name(),
            &server_name(&path.0, &path.1),
            required_string(object, "url")?,
            McpTransportType::parse(
                optional_string(object, "transport_type")?.unwrap_or("streamable-http"),
            )?,
            optional_string(object, "server_version")?,
            optional_string(object, "server_alias")?,
            username(auth.as_ref()),
        )
        .await?;
    json_response(endpoint_json(endpoint, true))
}

pub async fn get_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<EndpointPath>,
) -> Result<Response, MlflowError> {
    json_response(endpoint_json(
        state
            .tracking_store()
            .get_mcp_access_endpoint(workspace.name(), &server_name(&path.0, &path.1), &path.2)
            .await?,
        true,
    ))
}

pub async fn update_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    auth: Option<Extension<AuthContext>>,
    Path(path): Path<EndpointPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    let transport = match object.get("transport_type") {
        None => McpPatch::Unset,
        Some(Value::Null) => McpPatch::Set(None),
        Some(Value::String(value)) => McpPatch::Set(Some(McpTransportType::parse(value)?)),
        Some(_) => {
            return Err(invalid_request(
                "transport_type: Input should be a valid string",
            ))
        }
    };
    let endpoint = state
        .tracking_store()
        .update_mcp_access_endpoint(
            workspace.name(),
            &server_name(&path.0, &path.1),
            &path.2,
            patch_string(object, "server_version")?,
            patch_string(object, "server_alias")?,
            patch_string(object, "url")?,
            transport,
            username(auth.as_ref()),
        )
        .await?;
    json_response(endpoint_json(endpoint, true))
}

pub async fn delete_endpoint(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<EndpointPath>,
) -> Result<Response, MlflowError> {
    state
        .tracking_store()
        .delete_mcp_access_endpoint(workspace.name(), &server_name(&path.0, &path.1), &path.2)
        .await?;
    empty_response()
}

pub async fn search_all_endpoints(
    State(state): State<AppState>,
    workspace: Workspace,
    RawQuery(query): RawQuery,
) -> Result<Response, MlflowError> {
    search_endpoints_impl(state, workspace, None, query.as_deref()).await
}

pub async fn search_server_endpoints(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerPath>,
    RawQuery(query): RawQuery,
) -> Result<Response, MlflowError> {
    let name = server_name(&path.0, &path.1);
    search_endpoints_impl(state, workspace, Some(&name), query.as_deref()).await
}

async fn search_endpoints_impl(
    state: AppState,
    workspace: Workspace,
    server_name: Option<&str>,
    query: Option<&str>,
) -> Result<Response, MlflowError> {
    let query = QueryArgs::parse(query)?;
    let page = state
        .tracking_store()
        .search_mcp_access_endpoints(
            workspace.name(),
            server_name,
            query.one("server_version"),
            query.one("server_alias"),
            query.one("filter_string"),
            query.max_results()?,
            &query.many("order_by"),
            query.one("page_token"),
        )
        .await?;
    json_response(json!({
        "mcp_access_endpoints": page.items.into_iter().map(|value| endpoint_json(value, true)).collect::<Vec<_>>(),
        "next_page_token": page.next_page_token,
    }))
}

pub async fn set_server_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    state
        .tracking_store()
        .set_mcp_server_tag(
            workspace.name(),
            &server_name(&path.0, &path.1),
            required_string(object, "key")?,
            required_string(object, "value")?,
        )
        .await?;
    empty_response()
}

pub async fn delete_server_tag(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerChildPath>,
) -> Result<Response, MlflowError> {
    state
        .tracking_store()
        .delete_mcp_server_tag(workspace.name(), &server_name(&path.0, &path.1), &path.2)
        .await?;
    empty_response()
}

pub async fn set_alias(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerPath>,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let body = parse_body(&body)?;
    let object = body_object(&body)?;
    state
        .tracking_store()
        .set_mcp_server_alias(
            workspace.name(),
            &server_name(&path.0, &path.1),
            required_string(object, "alias")?,
            required_string(object, "version")?,
        )
        .await?;
    empty_response()
}

pub async fn get_by_alias(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerChildPath>,
) -> Result<Response, MlflowError> {
    json_response(version_json(
        state
            .tracking_store()
            .get_mcp_server_version_by_alias(
                workspace.name(),
                &server_name(&path.0, &path.1),
                &path.2,
            )
            .await?,
    ))
}

pub async fn delete_alias(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path): Path<ServerChildPath>,
) -> Result<Response, MlflowError> {
    state
        .tracking_store()
        .delete_mcp_server_alias(workspace.name(), &server_name(&path.0, &path.1), &path.2)
        .await?;
    empty_response()
}

pub(crate) fn server_json(server: McpServer) -> Value {
    json!({
        "name": server.name,
        "display_name": server.display_name,
        "description": server.description,
        "icons": server.icons,
        "workspace": server.workspace,
        "status": server.status.map(McpStatus::as_str),
        "access_endpoints": server.access_endpoints.into_iter().map(|value| endpoint_json(value, false)).collect::<Vec<_>>(),
        "latest_version": server.latest_version,
        "aliases": server.aliases.into_iter().map(|(alias, version)| json!({"alias": alias, "version": version})).collect::<Vec<_>>(),
        "tags": server.tags,
        "created_by": server.created_by,
        "last_updated_by": server.last_updated_by,
        "creation_timestamp": server.creation_timestamp,
        "last_updated_timestamp": server.last_updated_timestamp,
    })
}

fn version_json(version: McpServerVersion) -> Value {
    json!({
        "name": version.name,
        "version": version.version,
        "server_json": version.server_json,
        "display_name": version.display_name,
        "workspace": version.workspace,
        "status": version.status.as_str(),
        "tools": version.tools.unwrap_or_default().into_iter().map(tool_json).collect::<Vec<_>>(),
        "aliases": version.aliases,
        "tags": version.tags,
        "source": version.source,
        "created_by": version.created_by,
        "last_updated_by": version.last_updated_by,
        "creation_timestamp": version.creation_timestamp,
        "last_updated_timestamp": version.last_updated_timestamp,
    })
}

fn tool_json(tool: Value) -> Value {
    let Some(mut object) = tool.as_object().cloned() else {
        return tool;
    };
    for field in [
        "title",
        "description",
        "inputSchema",
        "outputSchema",
        "annotations",
        "icons",
        "execution",
    ] {
        object.entry(field).or_insert(Value::Null);
    }
    Value::Object(object)
}

pub(crate) fn endpoint_json(endpoint: McpAccessEndpoint, include_tools: bool) -> Value {
    let tools = endpoint
        .resolved_version
        .tools
        .clone()
        .map(|tools| tools.into_iter().map(tool_json).collect());
    let mut value = Map::new();
    value.insert("id".into(), Value::String(endpoint.id));
    value.insert("server_name".into(), Value::String(endpoint.server_name));
    value.insert("url".into(), Value::String(endpoint.url));
    value.insert(
        "transport_type".into(),
        Value::String(endpoint.transport_type.as_str().into()),
    );
    value.insert("workspace".into(), Value::String(endpoint.workspace));
    if include_tools {
        value.insert(
            "tools".into(),
            tools.map(Value::Array).unwrap_or(Value::Null),
        );
    }
    value.insert(
        "server_version".into(),
        endpoint
            .server_version
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    value.insert(
        "server_alias".into(),
        endpoint
            .server_alias
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    value.insert(
        "resolved_version".into(),
        version_json(*endpoint.resolved_version),
    );
    value.insert(
        "created_by".into(),
        endpoint
            .created_by
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    value.insert(
        "last_updated_by".into(),
        endpoint
            .last_updated_by
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    value.insert(
        "creation_timestamp".into(),
        endpoint.creation_timestamp.into(),
    );
    value.insert(
        "last_updated_timestamp".into(),
        endpoint.last_updated_timestamp.into(),
    );
    Value::Object(value)
}

fn parse_body(body: &[u8]) -> Result<Value, MlflowError> {
    if body.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_slice(body)
        .map_err(|error| invalid_request(&format!("JSON decode error: {error}")))
}

fn body_object(value: &Value) -> Result<&Map<String, Value>, MlflowError> {
    value
        .as_object()
        .ok_or_else(|| invalid_request("Input should be a valid dictionary or object"))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, MlflowError> {
    match object.get(field) {
        None => Err(missing(field)),
        Some(Value::String(value)) => Ok(value),
        _ => Err(invalid_request(&format!(
            "{field}: Input should be a valid string"
        ))),
    }
}

fn optional_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a str>, MlflowError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        _ => Err(invalid_request(&format!(
            "{field}: Input should be a valid string"
        ))),
    }
}

fn optional_array(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<Value>>, MlflowError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(value)) => Ok(Some(value.clone())),
        _ => Err(invalid_request(&format!(
            "{field}: Input should be a valid list"
        ))),
    }
}

fn patch_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<McpPatch<Option<String>>, MlflowError> {
    match object.get(field) {
        None => Ok(McpPatch::Unset),
        Some(Value::Null) => Ok(McpPatch::Set(None)),
        Some(Value::String(value)) => Ok(McpPatch::Set(Some(value.clone()))),
        _ => Err(invalid_request(&format!(
            "{field}: Input should be a valid string"
        ))),
    }
}

fn patch_array(
    object: &Map<String, Value>,
    field: &str,
) -> Result<McpPatch<Option<Vec<Value>>>, MlflowError> {
    match object.get(field) {
        None => Ok(McpPatch::Unset),
        Some(Value::Null) => Ok(McpPatch::Set(None)),
        Some(Value::Array(value)) => Ok(McpPatch::Set(Some(value.clone()))),
        _ => Err(invalid_request(&format!(
            "{field}: Input should be a valid list"
        ))),
    }
}

fn patch_icons(
    object: &Map<String, Value>,
    field: &str,
) -> Result<McpPatch<Option<Vec<Value>>>, MlflowError> {
    match patch_array(object, field)? {
        McpPatch::Unset => Ok(McpPatch::Unset),
        McpPatch::Set(value) => Ok(McpPatch::Set(normalize_icons(value, field)?)),
    }
}

fn patch_tools(
    object: &Map<String, Value>,
    field: &str,
) -> Result<McpPatch<Option<Vec<Value>>>, MlflowError> {
    match patch_array(object, field)? {
        McpPatch::Unset => Ok(McpPatch::Unset),
        McpPatch::Set(value) => Ok(McpPatch::Set(normalize_tools(value)?)),
    }
}

fn normalize_server_json(value: &mut Value) -> Result<(), MlflowError> {
    let object = value.as_object_mut().expect("checked by caller");
    required_string(object, "name")?;
    required_string(object, "version")?;
    for field in ["title", "description", "websiteUrl"] {
        validate_optional_string(object, field, &format!("server_json.{field}"))?;
    }
    validate_optional_object(object, "_meta", "server_json._meta")?;
    if object.contains_key("icons") {
        let icons = optional_array_at(object, "icons", "server_json.icons")?;
        object.insert(
            "icons".into(),
            normalize_icons(icons, "server_json.icons")?
                .map(Value::Array)
                .unwrap_or(Value::Null),
        );
    }
    if let Some(packages) = optional_array_at(object, "packages", "server_json.packages")? {
        for (index, package) in packages.iter().enumerate() {
            let package = package.as_object().ok_or_else(|| {
                invalid_request(&format!(
                    "server_json.packages.{index}: Input should be a valid dictionary"
                ))
            })?;
            required_string_at(
                package,
                "registryType",
                &format!("server_json.packages.{index}.registryType"),
            )?;
            required_string_at(
                package,
                "identifier",
                &format!("server_json.packages.{index}.identifier"),
            )?;
            if !package.contains_key("transport") {
                return Err(missing(&format!("server_json.packages.{index}.transport")));
            }
            for field in ["registryBaseUrl", "version"] {
                validate_optional_string(
                    package,
                    field,
                    &format!("server_json.packages.{index}.{field}"),
                )?;
            }
            if let Some(variables) = optional_array_at(
                package,
                "environmentVariables",
                &format!("server_json.packages.{index}.environmentVariables"),
            )? {
                for (variable_index, variable) in variables.iter().enumerate() {
                    let path = format!(
                        "server_json.packages.{index}.environmentVariables.{variable_index}"
                    );
                    let variable = variable.as_object().ok_or_else(|| {
                        invalid_request(&format!("{path}: Input should be a valid dictionary"))
                    })?;
                    required_string_at(variable, "name", &format!("{path}.name"))?;
                    validate_optional_string(
                        variable,
                        "description",
                        &format!("{path}.description"),
                    )?;
                    for field in ["isRequired", "isSecret"] {
                        if let Some(value) = variable.get(field) {
                            if !value.is_null() && !value.is_boolean() {
                                return Err(invalid_request(&format!(
                                    "{path}.{field}: Input should be a valid boolean"
                                )));
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(remotes) = optional_array_at(object, "remotes", "server_json.remotes")? {
        for (index, remote) in remotes.iter().enumerate() {
            let path = format!("server_json.remotes.{index}");
            let remote = remote.as_object().ok_or_else(|| {
                invalid_request(&format!("{path}: Input should be a valid dictionary"))
            })?;
            for field in ["type", "url"] {
                validate_optional_string(remote, field, &format!("{path}.{field}"))?;
            }
        }
    }
    if let Some(repository) = object.get("repository").filter(|value| !value.is_null()) {
        let repository = repository.as_object().ok_or_else(|| {
            invalid_request("server_json.repository: Input should be a valid dictionary")
        })?;
        for field in ["url", "source"] {
            required_string_at(
                repository,
                field,
                &format!("server_json.repository.{field}"),
            )?;
        }
        for field in ["id", "subfolder"] {
            validate_optional_string(
                repository,
                field,
                &format!("server_json.repository.{field}"),
            )?;
        }
    }
    Ok(())
}

fn normalize_icons(
    icons: Option<Vec<Value>>,
    field: &str,
) -> Result<Option<Vec<Value>>, MlflowError> {
    let Some(mut icons) = icons else {
        return Ok(None);
    };
    for (index, icon) in icons.iter_mut().enumerate() {
        let path = format!("{field}.{index}");
        let object = icon.as_object_mut().ok_or_else(|| {
            invalid_request(&format!("{path}: Input should be a valid dictionary"))
        })?;
        required_string_at(object, "src", &format!("{path}.src"))?;
        if let Some(sizes) = object.get("sizes") {
            match sizes {
                Value::Null => {
                    object.remove("sizes");
                }
                Value::Array(values) if values.iter().all(Value::is_string) => {}
                _ => {
                    return Err(invalid_request(&format!(
                        "{path}.sizes: Input should be a valid list"
                    )))
                }
            }
        }
        if let Some(mime) = object.get("mimeType") {
            match mime {
                Value::Null => {
                    object.remove("mimeType");
                }
                Value::String(value) => {
                    object.insert(
                        "mimeType".into(),
                        Value::String(value.trim().to_ascii_lowercase()),
                    );
                }
                _ => {
                    return Err(invalid_request(&format!(
                        "{path}.mimeType: Input should be a valid string"
                    )))
                }
            }
        }
        if object.get("theme").is_some_and(Value::is_null) {
            object.remove("theme");
        } else {
            validate_optional_string(object, "theme", &format!("{path}.theme"))?;
        }
    }
    Ok(Some(icons))
}

fn normalize_tools(tools: Option<Vec<Value>>) -> Result<Option<Vec<Value>>, MlflowError> {
    let Some(tools) = tools else { return Ok(None) };
    let mut normalized = Vec::with_capacity(tools.len());
    for (index, tool) in tools.into_iter().enumerate() {
        let path = format!("tools.{index}");
        let object = tool.as_object().ok_or_else(|| {
            invalid_request(&format!("{path}: Input should be a valid dictionary"))
        })?;
        let mut result = Map::new();
        result.insert(
            "name".into(),
            Value::String(required_string_at(object, "name", &format!("{path}.name"))?.into()),
        );
        for field in ["title", "description"] {
            if let Some(value) = optional_string_at(object, field, &format!("{path}.{field}"))? {
                result.insert(field.into(), Value::String(value.into()));
            }
        }
        for field in ["inputSchema", "outputSchema", "annotations", "execution"] {
            if let Some(value) = object.get(field).filter(|value| !value.is_null()) {
                if !value.is_object() {
                    return Err(invalid_request(&format!(
                        "{path}.{field}: Input should be a valid dictionary"
                    )));
                }
                result.insert(field.into(), value.clone());
            }
        }
        if object.contains_key("icons") {
            if let Some(icons) = normalize_icons(
                optional_array_at(object, "icons", &format!("{path}.icons"))?,
                &format!("{path}.icons"),
            )? {
                result.insert("icons".into(), Value::Array(icons));
            }
        }
        normalized.push(Value::Object(result));
    }
    Ok(Some(normalized))
}

fn required_string_at<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<&'a str, MlflowError> {
    match object.get(field) {
        None => Err(missing(path)),
        Some(Value::String(value)) => Ok(value),
        _ => Err(invalid_request(&format!(
            "{path}: Input should be a valid string"
        ))),
    }
}

fn optional_string_at<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<Option<&'a str>, MlflowError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        _ => Err(invalid_request(&format!(
            "{path}: Input should be a valid string"
        ))),
    }
}

fn validate_optional_string(
    object: &Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<(), MlflowError> {
    optional_string_at(object, field, path).map(|_| ())
}

fn optional_array_at(
    object: &Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<Option<Vec<Value>>, MlflowError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(value)) => Ok(Some(value.clone())),
        _ => Err(invalid_request(&format!(
            "{path}: Input should be a valid list"
        ))),
    }
}

fn validate_optional_object(
    object: &Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<(), MlflowError> {
    match object.get(field) {
        None | Some(Value::Null | Value::Object(_)) => Ok(()),
        _ => Err(invalid_request(&format!(
            "{path}: Input should be a valid dictionary"
        ))),
    }
}

fn missing(field: &str) -> MlflowError {
    invalid_request(&format!("{field}: Field required"))
}

fn invalid_request(message: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!("Invalid request: {message}"))
}

fn server_name(namespace: &str, slug: &str) -> String {
    format!("{namespace}/{slug}")
}

fn username(auth: Option<&Extension<AuthContext>>) -> Option<&str> {
    auth.map(|Extension(value)| value.username.as_str())
}

fn empty_response() -> Result<Response, MlflowError> {
    json_response(json!({}))
}

fn json_response(value: Value) -> Result<Response, MlflowError> {
    let body = serde_json::to_vec(&value)
        .map_err(|error| MlflowError::internal_error(error.to_string()))?;
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Ok(response)
}

#[derive(Default)]
struct QueryArgs {
    values: BTreeMap<String, Vec<String>>,
}

impl QueryArgs {
    fn parse(raw: Option<&str>) -> Result<Self, MlflowError> {
        let mut values = BTreeMap::<String, Vec<String>>::new();
        for (key, value) in crate::proto_http::parse_query_pairs(raw.unwrap_or("")) {
            values.entry(key).or_default().push(value);
        }
        Ok(Self { values })
    }

    fn one(&self, key: &str) -> Option<&str> {
        self.values
            .get(key)
            .and_then(|values| values.first())
            .map(String::as_str)
    }

    fn many(&self, key: &str) -> Vec<String> {
        self.values.get(key).cloned().unwrap_or_default()
    }

    fn max_results(&self) -> Result<i32, MlflowError> {
        self.one("max_results")
            .map(|value| {
                value.parse::<i32>().map_err(|_| {
                    invalid_request(
                        "max_results: Input should be a valid integer, unable to parse string as an integer",
                    )
                })
            })
            .unwrap_or(Ok(100))
    }
}
