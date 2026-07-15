//! Owned webhook entities returned by the store, mirroring
//! `mlflow/entities/webhook.py`.
//!
//! The entity carries the *plaintext* `secret` (the store decrypts it on read,
//! as `EncryptedString.process_result_value` does in Python), but the proto
//! serialization at the handler layer never emits it — Python's
//! `Webhook.to_proto` (`mlflow/entities/webhook.py:357`) has no `secret` field.
//!
//! Entity/action enums store the *lowercase* DB string (`registered_model`,
//! `created`, …), matching the `WebhookEntity`/`WebhookAction` `str` enums
//! whose values are the lowercase forms written to `webhook_events`.

use mlflow_error::MlflowError;

/// `WebhookStatus` (`mlflow/entities/webhook.py:16`). The DB stores the
/// uppercase name (`"ACTIVE"`/`"DISABLED"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookStatus {
    Active,
    Disabled,
}

impl WebhookStatus {
    /// The DB / proto-name string (`"ACTIVE"`/`"DISABLED"`).
    pub fn as_db_str(self) -> &'static str {
        match self {
            WebhookStatus::Active => "ACTIVE",
            WebhookStatus::Disabled => "DISABLED",
        }
    }

    /// Parse the DB string. Unknown values are an internal error (the column is
    /// constrained), but we surface a clear message rather than panic.
    pub fn from_db_str(s: &str) -> Result<Self, MlflowError> {
        match s {
            "ACTIVE" => Ok(WebhookStatus::Active),
            "DISABLED" => Ok(WebhookStatus::Disabled),
            other => Err(MlflowError::internal_error(format!(
                "Unknown webhook status in database: {other:?}"
            ))),
        }
    }

    /// The proto enum value (`WebhookStatus`: ACTIVE=1, DISABLED=2).
    pub fn to_proto_i32(self) -> i32 {
        match self {
            WebhookStatus::Active => 1,
            WebhookStatus::Disabled => 2,
        }
    }
}

/// `WebhookEntity` (`mlflow/entities/webhook.py:38`), holding the lowercase DB
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebhookEntity {
    RegisteredModel,
    ModelVersion,
    ModelVersionTag,
    ModelVersionAlias,
    Prompt,
    PromptVersion,
    PromptTag,
    PromptVersionTag,
    PromptAlias,
    BudgetPolicy,
}

impl WebhookEntity {
    /// The lowercase DB string (`"registered_model"`, …).
    pub fn as_db_str(self) -> &'static str {
        match self {
            WebhookEntity::RegisteredModel => "registered_model",
            WebhookEntity::ModelVersion => "model_version",
            WebhookEntity::ModelVersionTag => "model_version_tag",
            WebhookEntity::ModelVersionAlias => "model_version_alias",
            WebhookEntity::Prompt => "prompt",
            WebhookEntity::PromptVersion => "prompt_version",
            WebhookEntity::PromptTag => "prompt_tag",
            WebhookEntity::PromptVersionTag => "prompt_version_tag",
            WebhookEntity::PromptAlias => "prompt_alias",
            WebhookEntity::BudgetPolicy => "budget_policy",
        }
    }

    /// Parse the lowercase DB string; unknown → `None` (caller maps to the
    /// Python `ValueError`-shaped invalid-parameter error).
    pub fn from_db_str(s: &str) -> Option<Self> {
        Some(match s {
            "registered_model" => WebhookEntity::RegisteredModel,
            "model_version" => WebhookEntity::ModelVersion,
            "model_version_tag" => WebhookEntity::ModelVersionTag,
            "model_version_alias" => WebhookEntity::ModelVersionAlias,
            "prompt" => WebhookEntity::Prompt,
            "prompt_version" => WebhookEntity::PromptVersion,
            "prompt_tag" => WebhookEntity::PromptTag,
            "prompt_version_tag" => WebhookEntity::PromptVersionTag,
            "prompt_alias" => WebhookEntity::PromptAlias,
            "budget_policy" => WebhookEntity::BudgetPolicy,
            _ => return None,
        })
    }

    /// The proto enum value (`WebhookEntity`), the uppercase name's number.
    pub fn to_proto_i32(self) -> i32 {
        match self {
            WebhookEntity::RegisteredModel => 1,
            WebhookEntity::ModelVersion => 2,
            WebhookEntity::ModelVersionTag => 3,
            WebhookEntity::ModelVersionAlias => 4,
            WebhookEntity::Prompt => 5,
            WebhookEntity::PromptVersion => 6,
            WebhookEntity::PromptTag => 7,
            WebhookEntity::PromptVersionTag => 8,
            WebhookEntity::PromptAlias => 9,
            WebhookEntity::BudgetPolicy => 10,
        }
    }

    /// Map a proto enum value back to the entity. `ENTITY_UNSPECIFIED` (0) and
    /// any unknown value yield `None`.
    pub fn from_proto_i32(v: i32) -> Option<Self> {
        Some(match v {
            1 => WebhookEntity::RegisteredModel,
            2 => WebhookEntity::ModelVersion,
            3 => WebhookEntity::ModelVersionTag,
            4 => WebhookEntity::ModelVersionAlias,
            5 => WebhookEntity::Prompt,
            6 => WebhookEntity::PromptVersion,
            7 => WebhookEntity::PromptTag,
            8 => WebhookEntity::PromptVersionTag,
            9 => WebhookEntity::PromptAlias,
            10 => WebhookEntity::BudgetPolicy,
            _ => return None,
        })
    }
}

/// `WebhookAction` (`mlflow/entities/webhook.py:64`), holding the lowercase DB
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebhookAction {
    Created,
    Updated,
    Deleted,
    Set,
    Exceeded,
}

impl WebhookAction {
    /// The lowercase DB string (`"created"`, …).
    pub fn as_db_str(self) -> &'static str {
        match self {
            WebhookAction::Created => "created",
            WebhookAction::Updated => "updated",
            WebhookAction::Deleted => "deleted",
            WebhookAction::Set => "set",
            WebhookAction::Exceeded => "exceeded",
        }
    }

    /// Parse the lowercase DB string; unknown → `None`.
    pub fn from_db_str(s: &str) -> Option<Self> {
        Some(match s {
            "created" => WebhookAction::Created,
            "updated" => WebhookAction::Updated,
            "deleted" => WebhookAction::Deleted,
            "set" => WebhookAction::Set,
            "exceeded" => WebhookAction::Exceeded,
            _ => return None,
        })
    }

    /// The proto enum value (`WebhookAction`).
    pub fn to_proto_i32(self) -> i32 {
        match self {
            WebhookAction::Created => 1,
            WebhookAction::Updated => 2,
            WebhookAction::Deleted => 3,
            WebhookAction::Set => 4,
            WebhookAction::Exceeded => 5,
        }
    }

    /// Map a proto enum value back to the action. `ACTION_UNSPECIFIED` (0) and
    /// any unknown value yield `None`.
    pub fn from_proto_i32(v: i32) -> Option<Self> {
        Some(match v {
            1 => WebhookAction::Created,
            2 => WebhookAction::Updated,
            3 => WebhookAction::Deleted,
            4 => WebhookAction::Set,
            5 => WebhookAction::Exceeded,
            _ => return None,
        })
    }
}

/// A subscribed `(entity, action)` pair (`WebhookEvent`,
/// `mlflow/entities/webhook.py:148`). Validity of the pair is enforced by
/// [`crate::validation::validate_event_combination`] at the boundary (mirroring
/// the `WebhookEvent.__init__` check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WebhookEvent {
    pub entity: WebhookEntity,
    pub action: WebhookAction,
}

impl WebhookEvent {
    pub fn new(entity: WebhookEntity, action: WebhookAction) -> Self {
        Self { entity, action }
    }
}

/// The owned `Webhook` entity (`mlflow/entities/webhook.py:257`).
///
/// `secret` is the decrypted plaintext (or `None`); it is never serialized to
/// the proto response.
#[derive(Debug, Clone, PartialEq)]
pub struct Webhook {
    pub webhook_id: String,
    pub name: String,
    pub url: String,
    pub events: Vec<WebhookEvent>,
    pub description: Option<String>,
    pub status: WebhookStatus,
    pub secret: Option<String>,
    pub creation_timestamp: Option<i64>,
    pub last_updated_timestamp: Option<i64>,
    pub workspace: String,
}

/// `WebhookTestResult` (`mlflow/entities/webhook.py:385`).
#[derive(Debug, Clone, PartialEq)]
pub struct WebhookTestResult {
    pub success: bool,
    pub response_status: Option<i32>,
    pub response_body: Option<String>,
    pub error_message: Option<String>,
}
