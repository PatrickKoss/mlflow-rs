use mlflow_store::{McpPatch, McpStatus, McpTransportType, TrackingStore};
use mlflow_test_support::TempDb;
use serde_json::json;

const WS: &str = "default";

async fn store(tag: &str) -> (TempDb, TrackingStore) {
    let temp = TempDb::new(tag).await;
    let store = TrackingStore::new(temp.connect().await, "s3://bucket/mlruns");
    (temp, store)
}

#[tokio::test]
async fn crud_semver_tags_aliases_endpoints_and_cleanup() {
    let (_temp, store) = store("mcp-crud").await;
    let name = "com.example/server";
    let created = store
        .create_mcp_server(WS, name, Some("parent"), None, Some("alice"))
        .await
        .unwrap();
    assert_eq!(created.created_by.as_deref(), Some("alice"));

    for (version, status) in [
        ("1.2.0-alpha", McpStatus::Active),
        ("1.2.0", McpStatus::Active),
        ("1.10.0", McpStatus::Draft),
    ] {
        store
            .create_mcp_server_version(
                WS,
                json!({
                    "name": name,
                    "version": version,
                    "future": {"explicit_null": null}
                }),
                None,
                None,
                status,
                Some(vec![json!({"name": "tool", "future": true})]),
                Some("alice"),
            )
            .await
            .unwrap();
    }
    let parent = store.get_mcp_server(WS, name).await.unwrap();
    assert_eq!(parent.latest_version.as_deref(), Some("1.2.0"));

    store
        .set_mcp_server_tag(WS, name, "env", "prod")
        .await
        .unwrap();
    store
        .set_mcp_server_version_tag(WS, name, "1.2.0", "tier", "stable")
        .await
        .unwrap();
    store
        .set_mcp_server_alias(WS, name, "prod", "1.2.0")
        .await
        .unwrap();
    let endpoint = store
        .create_mcp_access_endpoint(
            WS,
            name,
            "https://mcp.example.com",
            McpTransportType::StreamableHttp,
            None,
            Some("prod"),
            Some("alice"),
        )
        .await
        .unwrap();
    assert_eq!(endpoint.resolved_version.version, "1.2.0");
    assert_eq!(
        endpoint.resolved_version.server_json["future"]["explicit_null"],
        json!(null)
    );

    store
        .delete_mcp_server_alias(WS, name, "prod")
        .await
        .unwrap();
    assert!(store
        .get_mcp_access_endpoint(WS, name, &endpoint.id)
        .await
        .is_err());
}

#[tokio::test]
async fn workspace_scoping_pagination_and_status_transitions() {
    let (_temp, store) = store("mcp-workspace").await;
    for workspace in ["default", "other"] {
        store
            .create_mcp_server(workspace, "com.example/scoped", None, None, None)
            .await
            .unwrap();
    }
    assert_eq!(
        store
            .search_mcp_servers(WS, None, 1, &[], None)
            .await
            .unwrap()
            .items
            .len(),
        1
    );

    for index in 0..3 {
        store
            .create_mcp_server(WS, &format!("com.example/page-{index}"), None, None, None)
            .await
            .unwrap();
    }
    let page = store
        .search_mcp_servers(WS, None, 2, &[], None)
        .await
        .unwrap();
    assert_eq!(page.items.len(), 2);
    assert!(page.next_page_token.is_some());

    let name = "com.example/status";
    store
        .create_mcp_server_version(
            WS,
            json!({"name": name, "version": "1.0.0"}),
            None,
            None,
            McpStatus::Draft,
            None,
            None,
        )
        .await
        .unwrap();
    store
        .update_mcp_server_version(
            WS,
            name,
            "1.0.0",
            McpPatch::Unset,
            McpPatch::Set(McpStatus::Active),
            McpPatch::Unset,
            None,
        )
        .await
        .unwrap();
    assert!(store
        .delete_mcp_server_version(WS, name, "1.0.0")
        .await
        .is_err());
}
