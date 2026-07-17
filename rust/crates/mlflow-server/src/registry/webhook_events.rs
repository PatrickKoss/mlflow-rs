//! Builds the `(WebhookEvent, data)` pairs the registry mutations fire (T8.4),
//! porting the `deliver_webhook(...)` calls in `mlflow/server/handlers.py`
//! (`_create_registered_model` L2636-2661, `_set_registered_model_tag` L2787,
//! `_delete_registered_model_tag` L2815, `_create_model_version` L2984-3017,
//! `_set_model_version_tag` L3193-3216, `_delete_model_version_tag`
//! L3239-3260, `_set_registered_model_alias` L3283-3304,
//! `_delete_registered_model_alias` L3322-3341).
//!
//! ## Payload shapes are byte-matched to Python
//!
//! Each `data` object is the `TypedDict` payload Python passes as
//! `WebhookPayload` (`mlflow/webhooks/types.py`). The dispatcher wraps it in
//! `{entity, action, timestamp, data}` and signs the whole thing, so the `data`
//! **field order and null-vs-omitted** must match Python exactly:
//!
//! * Field order follows the `TypedDict` constructor kwargs. `serde_json::Value`
//!   preserves insertion order (`preserve_order` is on workspace-wide), so the
//!   [`obj`] builder emits keys in the order listed.
//! * Every declared field is always present. Python constructs the `TypedDict`
//!   with all keys, so `None` values serialize as JSON `null` (they are **not**
//!   dropped). `Option<&str>` maps to `Value::Null` for `None`.
//! * `description`: `_create_registered_model` / prompt-created pass
//!   `request_message.description` verbatim (proto2 default `""`, so an absent
//!   description is the empty string, **not** null). `_create_model_version`
//!   passes `description or None` (empty → null). `run_id or None` likewise.
//!
//! ## Prompt classification is *instead of*, not in addition to
//!
//! Python fires **either** the model-entity event **or** the prompt-entity
//! event, never both. The classification differs per mutation:
//!
//! * create RM / create MV: `_is_prompt_request` — the request carries a tag
//!   whose key is `mlflow.prompt.is_prompt` (presence only, value ignored).
//! * tag/alias mutations: `_is_prompt(name)` — a fresh
//!   `get_registered_model(name)` whose stored `mlflow.prompt.is_prompt` tag
//!   lowercases to `"true"`. The lookup happens at trigger time (post-mutation),
//!   so a helper performs the same query with the same timing.
//!
//! Prompt payloads additionally strip the internal `mlflow.prompt.is_prompt`,
//! `_mlflow_prompt_type`, and (for prompt versions) `mlflow.prompt.text` tags
//! from the emitted `tags` dict; `mlflow.prompt.text` becomes the `template`
//! field.

use serde_json::{json, Map, Value};

use mlflow_webhooks::{WebhookAction, WebhookEntity, WebhookEvent};

/// `mlflow.prompt.is_prompt` (`mlflow/prompt/constants.py:4`).
pub const IS_PROMPT_TAG_KEY: &str = "mlflow.prompt.is_prompt";
/// `mlflow.prompt.text` (`mlflow/prompt/constants.py:6`).
const PROMPT_TEXT_TAG_KEY: &str = "mlflow.prompt.text";
/// `_mlflow_prompt_type` (`mlflow/prompt/constants.py:9`).
const PROMPT_TYPE_TAG_KEY: &str = "_mlflow_prompt_type";

/// A single request tag (`key`, `value`); `value` mirrors the proto default `""`
/// when absent.
pub struct Tag<'a> {
    pub key: &'a str,
    pub value: &'a str,
}

/// `_is_prompt_request(request_message)` (`handlers.py:3022`): the request tags
/// contain a `mlflow.prompt.is_prompt` key (presence only — the value is not
/// inspected here, unlike the stored-tag [`is_prompt_tag_true`] check).
pub fn request_has_is_prompt_tag(tags: &[Tag<'_>]) -> bool {
    tags.iter().any(|t| t.key == IS_PROMPT_TAG_KEY)
}

/// `RegisteredModel._is_prompt()` (`registered_model.py:97`): the stored
/// `mlflow.prompt.is_prompt` tag, defaulting to `"false"`, lowercases to
/// `"true"`. Applied to the tags returned by a fresh `get_registered_model`.
pub fn is_prompt_tag_true(value: Option<&str>) -> bool {
    value.unwrap_or("false").eq_ignore_ascii_case("true")
}

/// `_create_registered_model` webhook (`handlers.py:2636-2661`).
///
/// Prompt → `prompt/created` with the internal is-prompt/type tags stripped;
/// otherwise `registered_model/created` with all request tags.
pub fn registered_model_created(
    name: &str,
    tags: &[Tag<'_>],
    description: &str,
) -> (WebhookEvent, Value) {
    if request_has_is_prompt_tag(tags) {
        let filtered = tags_dict_excluding(tags, &[IS_PROMPT_TAG_KEY, PROMPT_TYPE_TAG_KEY]);
        (
            event(WebhookEntity::Prompt, WebhookAction::Created),
            Value::Object(obj([
                ("name", json!(name)),
                ("tags", Value::Object(filtered)),
                ("description", json!(description)),
            ])),
        )
    } else {
        (
            event(WebhookEntity::RegisteredModel, WebhookAction::Created),
            Value::Object(obj([
                ("name", json!(name)),
                ("tags", Value::Object(tags_dict(tags))),
                ("description", json!(description)),
            ])),
        )
    }
}

/// `_create_model_version` webhook (`handlers.py:2984-3017`).
///
/// Prompt → `prompt_version/created`: the `mlflow.prompt.text` tag is popped
/// into `template` (absent → `null`), and the text/is-prompt/type tags are
/// stripped from `tags`. Otherwise `model_version/created` carries `source`,
/// `run_id or None`, and all request tags. `description or None` in both.
pub fn model_version_created(
    name: &str,
    version: &str,
    source: &str,
    run_id: Option<&str>,
    tags: &[Tag<'_>],
    description: Option<&str>,
) -> (WebhookEvent, Value) {
    let description = description.filter(|s| !s.is_empty());
    if request_has_is_prompt_tag(tags) {
        let template = tags
            .iter()
            .find(|t| t.key == PROMPT_TEXT_TAG_KEY)
            .map(|t| t.value);
        let filtered = tags_dict_excluding(
            tags,
            &[PROMPT_TEXT_TAG_KEY, IS_PROMPT_TAG_KEY, PROMPT_TYPE_TAG_KEY],
        );
        (
            event(WebhookEntity::PromptVersion, WebhookAction::Created),
            Value::Object(obj([
                ("name", json!(name)),
                ("version", json!(version)),
                ("template", opt_str(template)),
                ("tags", Value::Object(filtered)),
                ("description", opt_str(description)),
            ])),
        )
    } else {
        (
            event(WebhookEntity::ModelVersion, WebhookAction::Created),
            Value::Object(obj([
                ("name", json!(name)),
                ("version", json!(version)),
                ("source", json!(source)),
                ("run_id", opt_str(run_id.filter(|s| !s.is_empty()))),
                ("tags", Value::Object(tags_dict(tags))),
                ("description", opt_str(description)),
            ])),
        )
    }
}

/// `_set_registered_model_tag` webhook (`handlers.py:2787`). Fires **only** when
/// the model is a prompt; a non-prompt RM tag set fires nothing.
pub fn registered_model_tag_set(
    is_prompt: bool,
    name: &str,
    key: &str,
    value: &str,
) -> Option<(WebhookEvent, Value)> {
    is_prompt.then(|| {
        (
            event(WebhookEntity::PromptTag, WebhookAction::Set),
            Value::Object(obj([
                ("name", json!(name)),
                ("key", json!(key)),
                ("value", json!(value)),
            ])),
        )
    })
}

/// `_delete_registered_model_tag` webhook (`handlers.py:2815`). Prompt-only.
pub fn registered_model_tag_deleted(
    is_prompt: bool,
    name: &str,
    key: &str,
) -> Option<(WebhookEvent, Value)> {
    is_prompt.then(|| {
        (
            event(WebhookEntity::PromptTag, WebhookAction::Deleted),
            Value::Object(obj([("name", json!(name)), ("key", json!(key))])),
        )
    })
}

/// `_set_model_version_tag` webhook (`handlers.py:3193-3216`). Prompt →
/// `prompt_version_tag/set`, else `model_version_tag/set`; identical shape.
pub fn model_version_tag_set(
    is_prompt: bool,
    name: &str,
    version: &str,
    key: &str,
    value: &str,
) -> (WebhookEvent, Value) {
    let entity = if is_prompt {
        WebhookEntity::PromptVersionTag
    } else {
        WebhookEntity::ModelVersionTag
    };
    (
        event(entity, WebhookAction::Set),
        Value::Object(obj([
            ("name", json!(name)),
            ("version", json!(version)),
            ("key", json!(key)),
            ("value", json!(value)),
        ])),
    )
}

/// `_delete_model_version_tag` webhook (`handlers.py:3239-3260`). Prompt →
/// `prompt_version_tag/deleted`, else `model_version_tag/deleted`.
pub fn model_version_tag_deleted(
    is_prompt: bool,
    name: &str,
    version: &str,
    key: &str,
) -> (WebhookEvent, Value) {
    let entity = if is_prompt {
        WebhookEntity::PromptVersionTag
    } else {
        WebhookEntity::ModelVersionTag
    };
    (
        event(entity, WebhookAction::Deleted),
        Value::Object(obj([
            ("name", json!(name)),
            ("version", json!(version)),
            ("key", json!(key)),
        ])),
    )
}

/// `_set_registered_model_alias` webhook (`handlers.py:3283-3304`). Prompt →
/// `prompt_alias/created`, else `model_version_alias/created`.
pub fn registered_model_alias_set(
    is_prompt: bool,
    name: &str,
    alias: &str,
    version: &str,
) -> (WebhookEvent, Value) {
    let entity = if is_prompt {
        WebhookEntity::PromptAlias
    } else {
        WebhookEntity::ModelVersionAlias
    };
    (
        event(entity, WebhookAction::Created),
        Value::Object(obj([
            ("name", json!(name)),
            ("alias", json!(alias)),
            ("version", json!(version)),
        ])),
    )
}

/// `_delete_registered_model_alias` webhook (`handlers.py:3322-3341`). Prompt →
/// `prompt_alias/deleted`, else `model_version_alias/deleted`.
pub fn registered_model_alias_deleted(
    is_prompt: bool,
    name: &str,
    alias: &str,
) -> (WebhookEvent, Value) {
    let entity = if is_prompt {
        WebhookEntity::PromptAlias
    } else {
        WebhookEntity::ModelVersionAlias
    };
    (
        event(entity, WebhookAction::Deleted),
        Value::Object(obj([("name", json!(name)), ("alias", json!(alias))])),
    )
}

fn event(entity: WebhookEntity, action: WebhookAction) -> WebhookEvent {
    WebhookEvent::new(entity, action)
}

/// `{t.key: t.value for t in tags}`.
fn tags_dict(tags: &[Tag<'_>]) -> Map<String, Value> {
    tags.iter()
        .map(|t| (t.key.to_string(), json!(t.value)))
        .collect()
}

/// `{t.key: t.value for t in tags if t.key not in excluded}`.
fn tags_dict_excluding(tags: &[Tag<'_>], excluded: &[&str]) -> Map<String, Value> {
    tags.iter()
        .filter(|t| !excluded.contains(&t.key))
        .map(|t| (t.key.to_string(), json!(t.value)))
        .collect()
}

/// `Option<&str>` → a JSON string or `null` (Python keeps the declared key with a
/// `None` value, so the key is always present).
fn opt_str(v: Option<&str>) -> Value {
    v.map_or(Value::Null, |s| json!(s))
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

    fn tag<'a>(key: &'a str, value: &'a str) -> Tag<'a> {
        Tag { key, value }
    }

    #[test]
    fn rm_created_non_prompt_keeps_all_tags_and_string_description() {
        let (ev, data) = registered_model_created("m", &[tag("a", "1")], "");
        assert_eq!(ev.entity, WebhookEntity::RegisteredModel);
        assert_eq!(ev.action, WebhookAction::Created);
        assert_eq!(data["name"], json!("m"));
        assert_eq!(data["tags"], json!({"a": "1"}));
        // description is the empty string, NOT null (proto2 default, no `or None`).
        assert_eq!(data["description"], json!(""));
        let keys: Vec<&String> = data.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["name", "tags", "description"]);
    }

    #[test]
    fn rm_created_prompt_strips_internal_tags() {
        let tags = [
            tag(IS_PROMPT_TAG_KEY, "true"),
            tag(PROMPT_TYPE_TAG_KEY, "text"),
            tag("user", "v"),
        ];
        let (ev, data) = registered_model_created("p", &tags, "d");
        assert_eq!(ev.entity, WebhookEntity::Prompt);
        assert_eq!(data["tags"], json!({"user": "v"}));
        assert_eq!(data["description"], json!("d"));
    }

    #[test]
    fn mv_created_non_prompt_nulls_empty_run_id_and_description() {
        let (ev, data) =
            model_version_created("m", "1", "s3://x", Some(""), &[tag("a", "1")], Some(""));
        assert_eq!(ev.entity, WebhookEntity::ModelVersion);
        assert_eq!(data["run_id"], Value::Null);
        assert_eq!(data["description"], Value::Null);
        assert_eq!(data["source"], json!("s3://x"));
        let keys: Vec<&String> = data.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            vec!["name", "version", "source", "run_id", "tags", "description"]
        );
    }

    #[test]
    fn mv_created_prompt_extracts_template_and_strips_tags() {
        let tags = [
            tag(IS_PROMPT_TAG_KEY, "true"),
            tag(PROMPT_TEXT_TAG_KEY, "Hello {{n}}!"),
            tag(PROMPT_TYPE_TAG_KEY, "text"),
            tag("user", "v"),
        ];
        let (ev, data) =
            model_version_created("p", "2", "prompt-template", None, &tags, Some("desc"));
        assert_eq!(ev.entity, WebhookEntity::PromptVersion);
        assert_eq!(data["template"], json!("Hello {{n}}!"));
        assert_eq!(data["tags"], json!({"user": "v"}));
        assert_eq!(data["description"], json!("desc"));
        let keys: Vec<&String> = data.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            vec!["name", "version", "template", "tags", "description"]
        );
    }

    #[test]
    fn mv_created_prompt_missing_template_is_null() {
        let tags = [tag(IS_PROMPT_TAG_KEY, "true")];
        let (_ev, data) = model_version_created("p", "1", "src", None, &tags, None);
        assert_eq!(data["template"], Value::Null);
    }

    #[test]
    fn rm_tag_set_only_fires_for_prompt() {
        assert!(registered_model_tag_set(false, "m", "k", "v").is_none());
        let (ev, data) = registered_model_tag_set(true, "p", "k", "v").unwrap();
        assert_eq!(ev.entity, WebhookEntity::PromptTag);
        assert_eq!(ev.action, WebhookAction::Set);
        assert_eq!(data, json!({"name": "p", "key": "k", "value": "v"}));
    }

    #[test]
    fn rm_tag_deleted_only_fires_for_prompt() {
        assert!(registered_model_tag_deleted(false, "m", "k").is_none());
        let (ev, _) = registered_model_tag_deleted(true, "p", "k").unwrap();
        assert_eq!(ev.entity, WebhookEntity::PromptTag);
        assert_eq!(ev.action, WebhookAction::Deleted);
    }

    #[test]
    fn mv_tag_and_alias_entities_switch_on_prompt() {
        assert_eq!(
            model_version_tag_set(false, "m", "1", "k", "v").0.entity,
            WebhookEntity::ModelVersionTag
        );
        assert_eq!(
            model_version_tag_set(true, "p", "1", "k", "v").0.entity,
            WebhookEntity::PromptVersionTag
        );
        assert_eq!(
            model_version_tag_deleted(true, "p", "1", "k").0.entity,
            WebhookEntity::PromptVersionTag
        );
        assert_eq!(
            registered_model_alias_set(false, "m", "a", "1").0.entity,
            WebhookEntity::ModelVersionAlias
        );
        assert_eq!(
            registered_model_alias_set(true, "p", "a", "1").0.entity,
            WebhookEntity::PromptAlias
        );
        assert_eq!(
            registered_model_alias_deleted(true, "p", "a").0.entity,
            WebhookEntity::PromptAlias
        );
    }

    #[test]
    fn is_prompt_tag_true_matches_python_semantics() {
        assert!(is_prompt_tag_true(Some("true")));
        assert!(is_prompt_tag_true(Some("True")));
        assert!(!is_prompt_tag_true(Some("false")));
        assert!(!is_prompt_tag_true(None));
    }
}
