//! Example event payloads for the `/test` endpoint, porting the `example()`
//! classmethods and `get_example_payload_for_event` in
//! `mlflow/webhooks/types.py`.
//!
//! Field insertion order matches the Python TypedDict `example()` bodies so the
//! serialized `data` object (and therefore the signed content) is byte-identical
//! — `serde_json::Value` is built with the `preserve_order` feature enabled
//! workspace-wide, so `Map` preserves insertion order.

use serde_json::{json, Map, Value};

use crate::entities::{WebhookAction, WebhookEntity, WebhookEvent};

/// `get_example_payload_for_event(event)`: the example `data` payload for the
/// `(entity, action)` pair, or `None` for an unknown combination (Python raises
/// `ValueError`; the caller maps that to a failed test result).
pub fn example_payload_for_event(event: WebhookEvent) -> Option<Value> {
    use WebhookAction::*;
    use WebhookEntity::*;
    let obj: Map<String, Value> = match (event.entity, event.action) {
        (RegisteredModel, Created) => obj([
            ("name", json!("example_model")),
            ("tags", json!({ "example_key": "example_value" })),
            ("description", json!("An example registered model")),
        ]),
        (ModelVersion, Created) => obj([
            ("name", json!("example_model")),
            ("version", json!("1")),
            ("source", json!("models:/123")),
            ("run_id", json!("abcd1234abcd5678")),
            ("tags", json!({ "example_key": "example_value" })),
            ("description", json!("An example model version")),
        ]),
        (ModelVersionTag, Set) => obj([
            ("name", json!("example_model")),
            ("version", json!("1")),
            ("key", json!("example_key")),
            ("value", json!("example_value")),
        ]),
        (ModelVersionTag, Deleted) => obj([
            ("name", json!("example_model")),
            ("version", json!("1")),
            ("key", json!("example_key")),
        ]),
        (ModelVersionAlias, Created) => obj([
            ("name", json!("example_model")),
            ("alias", json!("example_alias")),
            ("version", json!("1")),
        ]),
        (ModelVersionAlias, Deleted) => obj([
            ("name", json!("example_model")),
            ("alias", json!("example_alias")),
        ]),
        (Prompt, Created) => obj([
            ("name", json!("example_prompt")),
            ("tags", json!({ "example_key": "example_value" })),
            ("description", json!("An example prompt")),
        ]),
        (PromptVersion, Created) => obj([
            ("name", json!("example_prompt")),
            ("version", json!("1")),
            ("template", json!("Hello {{name}}!")),
            ("tags", json!({ "example_key": "example_value" })),
            ("description", json!("An example prompt version")),
        ]),
        (PromptTag, Set) => obj([
            ("name", json!("example_prompt")),
            ("key", json!("example_key")),
            ("value", json!("example_value")),
        ]),
        (PromptTag, Deleted) => obj([
            ("name", json!("example_prompt")),
            ("key", json!("example_key")),
        ]),
        (PromptVersionTag, Set) => obj([
            ("name", json!("example_prompt")),
            ("version", json!("1")),
            ("key", json!("example_key")),
            ("value", json!("example_value")),
        ]),
        (PromptVersionTag, Deleted) => obj([
            ("name", json!("example_prompt")),
            ("version", json!("1")),
            ("key", json!("example_key")),
        ]),
        (PromptAlias, Created) => obj([
            ("name", json!("example_prompt")),
            ("alias", json!("example_alias")),
            ("version", json!("1")),
        ]),
        (PromptAlias, Deleted) => obj([
            ("name", json!("example_prompt")),
            ("alias", json!("example_alias")),
        ]),
        (BudgetPolicy, Exceeded) => obj([
            ("budget_policy_id", json!("bp-abc123")),
            ("budget_unit", json!("USD")),
            ("budget_amount", json!(100.0)),
            ("current_spend", json!(105.50)),
            ("duration_unit", json!("MONTHS")),
            ("duration_value", json!(1)),
            ("target_scope", json!("WORKSPACE")),
            ("workspace", json!("default")),
            ("window_start", json!(1704067200000i64)),
        ]),
        _ => return None,
    };
    Some(Value::Object(obj))
}

fn obj<const N: usize>(entries: [(&str, Value); N]) -> Map<String, Value> {
    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_event_has_payload() {
        let p = example_payload_for_event(WebhookEvent::new(
            WebhookEntity::RegisteredModel,
            WebhookAction::Created,
        ))
        .unwrap();
        assert_eq!(p["name"], json!("example_model"));
    }

    #[test]
    fn field_order_preserved() {
        let p = example_payload_for_event(WebhookEvent::new(
            WebhookEntity::ModelVersion,
            WebhookAction::Created,
        ))
        .unwrap();
        let keys: Vec<&String> = p.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            vec!["name", "version", "source", "run_id", "tags", "description"]
        );
    }
}
