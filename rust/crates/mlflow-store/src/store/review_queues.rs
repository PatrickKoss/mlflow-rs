//! Workspace-scoped review-queue CRUD and workflow semantics.

use std::collections::{HashMap, HashSet};

use mlflow_error::MlflowError;
use uuid::Uuid;

use super::dbutil::{RowLike, Tx, Val};
use super::experiments::{internal, is_unique_violation, now_millis, parse_experiment_id};
use super::search::{SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD};
use super::TrackingStore;
use crate::UpsertSpec;

const QUEUE_ID_PREFIX: &str = "rq-";
const QUEUE_NAME_MAX_LENGTH: usize = 250;
const USER_MAX_LENGTH: usize = 250;
const SCHEMA_ID_MAX_LENGTH: usize = 36;
const ITEM_ID_MAX_LENGTH: usize = 50;
const MAX_ASSIGNED_USERS: usize = 10;
const RESERVED_QUEUE_NAME: &str = "default";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewQueueType {
    User,
    Custom,
}

impl ReviewQueueType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewItemType {
    Trace,
}

impl ReviewItemType {
    pub fn as_str(self) -> &'static str {
        "trace"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewStatus {
    Pending,
    Complete,
    Declined,
}

impl ReviewStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Complete => "complete",
            Self::Declined => "declined",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewQueue {
    pub queue_id: String,
    pub experiment_id: String,
    pub name: String,
    pub queue_type: ReviewQueueType,
    pub created_by: Option<String>,
    pub creation_time_ms: i64,
    pub last_update_time_ms: i64,
    pub users: Vec<String>,
    /// Literal `review_queue_label_schemas` rows. This is empty for user queues.
    pub schema_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewQueueItem {
    pub queue_id: String,
    pub item_type: ReviewItemType,
    pub item_id: String,
    pub status: ReviewStatus,
    pub completed_by: Option<String>,
    pub completed_time_ms: Option<i64>,
    pub creation_time_ms: i64,
    pub last_update_time_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewQueuesPage {
    pub queues: Vec<ReviewQueue>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewQueueItemsPage {
    pub items: Vec<ReviewQueueItem>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ReviewQueueUpdate<'a> {
    pub users: Option<&'a [String]>,
    pub schema_ids: Option<&'a [String]>,
    pub name: Option<&'a str>,
    pub new_owner: Option<&'a str>,
}

struct ValidatedQueue {
    name: String,
    users: Vec<String>,
    schema_ids: Vec<String>,
}

impl TrackingStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn create_review_queue(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
        queue_type: ReviewQueueType,
        created_by: Option<&str>,
        users: &[String],
        schema_ids: &[String],
    ) -> Result<ReviewQueue, MlflowError> {
        let validated = validate_queue_for_create(name, queue_type, users, schema_ids)?;
        let parsed_experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, parsed_experiment_id)
            .await?;
        self.validate_review_queue_schema_ids(
            workspace,
            parsed_experiment_id,
            &validated.schema_ids,
        )
        .await?;

        if self
            .find_review_queue_by_name(workspace, parsed_experiment_id, &validated.name)
            .await?
            .is_some()
        {
            return Err(review_queue_exists(&validated.name));
        }

        let queue_id = format!("{QUEUE_ID_PREFIX}{}", Uuid::new_v4().simple());
        let now = now_millis();
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let insert = format!(
            "INSERT INTO review_queues (queue_id, experiment_id, name, queue_type, created_by, \
             creation_time_ms, last_update_time_ms, name_key) VALUES ({})",
            (1..=8)
                .map(|index| dialect.placeholder(index))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if let Err(error) = tx
            .exec(
                &insert,
                &[
                    Val::Text(queue_id.clone()),
                    Val::Int(parsed_experiment_id),
                    Val::Text(validated.name.clone()),
                    Val::Text(queue_type.as_str().to_string()),
                    Val::OptText(created_by.map(str::to_string)),
                    Val::Int(now),
                    Val::Int(now),
                    Val::Text(validated.name.to_lowercase()),
                ],
            )
            .await
        {
            return if is_unique_violation(&error) {
                Err(review_queue_exists(&validated.name))
            } else {
                Err(internal(error))
            };
        }
        insert_users(&mut tx, dialect, &queue_id, &validated.users).await?;
        insert_schema_ids(&mut tx, dialect, &queue_id, &validated.schema_ids).await?;
        tx.commit().await.map_err(internal)?;
        self.get_review_queue(workspace, &queue_id).await
    }

    pub async fn get_or_create_user_queue(
        &self,
        workspace: &str,
        experiment_id: &str,
        user: &str,
    ) -> Result<ReviewQueue, MlflowError> {
        let name = normalize_user(user);
        match self
            .create_review_queue(
                workspace,
                experiment_id,
                &name,
                ReviewQueueType::User,
                Some(&name),
                &[],
                &[],
            )
            .await
        {
            Ok(queue) => Ok(queue),
            Err(error) if error.error_code == mlflow_error::ErrorCode::ResourceAlreadyExists => {
                let existing = self
                    .get_review_queue_by_name(workspace, experiment_id, &name)
                    .await?;
                if existing.queue_type != ReviewQueueType::User {
                    return Err(MlflowError::resource_already_exists(format!(
                        "A non-user queue named '{name}' already exists; cannot get-or-create a \
                         user queue with that name."
                    )));
                }
                Ok(existing)
            }
            Err(error) => Err(error),
        }
    }

    pub async fn get_review_queue(
        &self,
        workspace: &str,
        queue_id: &str,
    ) -> Result<ReviewQueue, MlflowError> {
        let row = self
            .find_review_queue_by_id(workspace, queue_id)
            .await?
            .ok_or_else(|| review_queue_not_found(queue_id))?;
        self.hydrate_review_queues(vec![row])
            .await
            .map(|mut queues| {
                queues
                    .pop()
                    .expect("one base review queue row hydrates to one queue")
            })
    }

    pub async fn get_review_queue_by_name(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
    ) -> Result<ReviewQueue, MlflowError> {
        let parsed_experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, parsed_experiment_id)
            .await?;
        let row = self
            .find_review_queue_by_name(workspace, parsed_experiment_id, name)
            .await?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Review queue with name '{name}' not found for experiment '{experiment_id}'."
                ))
            })?;
        self.hydrate_review_queues(vec![row])
            .await
            .map(|mut queues| {
                queues
                    .pop()
                    .expect("one base review queue row hydrates to one queue")
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn list_review_queues(
        &self,
        workspace: &str,
        experiment_id: &str,
        user: Option<&str>,
        item_id: Option<&str>,
        max_results: Option<i32>,
        page_token: Option<&str>,
    ) -> Result<ReviewQueuesPage, MlflowError> {
        let max_results = validate_max_results(max_results)?;
        let offset = parse_page_token(page_token)?;
        let parsed_experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, parsed_experiment_id)
            .await?;
        let dialect = self.db().dialect();
        let mut values = vec![
            Val::Int(parsed_experiment_id),
            Val::Text(workspace.to_string()),
        ];
        let mut predicates = vec![
            format!("rq.experiment_id = {}", dialect.placeholder(1)),
            format!("e.workspace = {}", dialect.placeholder(2)),
        ];
        if let Some(user) = user {
            values.push(Val::Text(normalize_user(user)));
            predicates.push(format!(
                "rq.queue_id IN (SELECT queue_id FROM review_queue_users WHERE user_id = {})",
                dialect.placeholder(values.len())
            ));
        }
        if let Some(item_id) = item_id {
            values.push(Val::Text(item_id.to_string()));
            predicates.push(format!(
                "rq.queue_id IN (SELECT queue_id FROM review_queue_items WHERE item_id = {})",
                dialect.placeholder(values.len())
            ));
        }
        let mut rows = self
            .db()
            .fetch_all(
                &format!(
                    "{} WHERE {} ORDER BY rq.creation_time_ms DESC, rq.queue_id ASC LIMIT {} OFFSET {}",
                    review_queue_select(),
                    predicates.join(" AND "),
                    max_results + 1,
                    offset
                ),
                &values,
                map_review_queue_row,
            )
            .await
            .map_err(internal)?;
        let next_page_token = if rows.len() > max_results as usize {
            rows.truncate(max_results as usize);
            Some(mlflow_search::create_page_token(offset + max_results))
        } else {
            None
        };
        Ok(ReviewQueuesPage {
            queues: self.hydrate_review_queues(rows).await?,
            next_page_token,
        })
    }

    pub async fn update_review_queue(
        &self,
        workspace: &str,
        queue_id: &str,
        update: ReviewQueueUpdate<'_>,
    ) -> Result<ReviewQueue, MlflowError> {
        let existing = self.get_review_queue(workspace, queue_id).await?;
        if existing.queue_type == ReviewQueueType::User {
            return Err(MlflowError::invalid_parameter_value(
                "A user queue's name, assigned user, schemas, and owner are fixed and cannot be updated.",
            ));
        }
        if update.users.is_none()
            && update.schema_ids.is_none()
            && update.name.is_none()
            && update.new_owner.is_none()
        {
            return Ok(existing);
        }

        let users = update.users.map(normalize_users).transpose()?;
        let schema_ids = update.schema_ids.map(normalize_schema_ids).transpose()?;
        if let Some(schema_ids) = &schema_ids {
            self.validate_review_queue_schema_ids(
                workspace,
                parse_experiment_id(&existing.experiment_id)?,
                schema_ids,
            )
            .await?;
        }
        let name = update.name.map(validate_custom_queue_name).transpose()?;
        let new_owner = update.new_owner.map(validate_queue_owner).transpose()?;

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let queue_exists = tx
            .fetch_all(
                &format!(
                    "SELECT rq.queue_id FROM review_queues rq JOIN experiments e ON \
                     e.experiment_id = rq.experiment_id WHERE rq.queue_id = {} AND e.workspace = {}{}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    if dialect == crate::Dialect::Sqlite {
                        ""
                    } else {
                        " FOR UPDATE"
                    }
                ),
                &[
                    Val::Text(queue_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("queue_id"),
            )
            .await
            .map_err(internal)?;
        if queue_exists.is_empty() {
            return Err(review_queue_not_found(queue_id));
        }

        if let Some(schema_ids) = &schema_ids {
            let count = tx
                .fetch_all(
                    &format!(
                        "SELECT COUNT(*) AS item_count FROM review_queue_items WHERE queue_id = {}",
                        dialect.placeholder(1)
                    ),
                    &[Val::Text(queue_id.to_string())],
                    |row| row.get_i64("item_count"),
                )
                .await
                .map_err(internal)?
                .into_iter()
                .next()
                .unwrap_or(0);
            if count > 0 {
                return Err(MlflowError::invalid_parameter_value(
                    "A review queue's questions are locked once items are assigned to it. Remove the items before changing its questions.",
                ));
            }
            tx.exec(
                &format!(
                    "DELETE FROM review_queue_label_schemas WHERE queue_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(queue_id.to_string())],
            )
            .await
            .map_err(internal)?;
            insert_schema_ids(&mut tx, dialect, queue_id, schema_ids).await?;
        }
        if let Some(users) = &users {
            tx.exec(
                &format!(
                    "DELETE FROM review_queue_users WHERE queue_id = {}",
                    dialect.placeholder(1)
                ),
                &[Val::Text(queue_id.to_string())],
            )
            .await
            .map_err(internal)?;
            insert_users(&mut tx, dialect, queue_id, users).await?;
        }

        let mut assignments = Vec::new();
        let mut values = Vec::new();
        if let Some(name) = &name {
            values.push(Val::Text(name.clone()));
            assignments.push(format!("name = {}", dialect.placeholder(values.len())));
            values.push(Val::Text(name.to_lowercase()));
            assignments.push(format!("name_key = {}", dialect.placeholder(values.len())));
        }
        if let Some(owner) = &new_owner {
            values.push(Val::Text(owner.clone()));
            assignments.push(format!(
                "created_by = {}",
                dialect.placeholder(values.len())
            ));
        }
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_update_time_ms = {}",
            dialect.placeholder(values.len())
        ));
        values.push(Val::Text(queue_id.to_string()));
        let update_result = tx
            .exec(
                &format!(
                    "UPDATE review_queues SET {} WHERE queue_id = {}",
                    assignments.join(", "),
                    dialect.placeholder(values.len())
                ),
                &values,
            )
            .await;
        if let Err(error) = update_result {
            return if name.is_some() && is_unique_violation(&error) {
                Err(review_queue_exists(name.as_deref().unwrap_or_default()))
            } else {
                Err(internal(error))
            };
        }
        tx.commit().await.map_err(internal)?;
        self.get_review_queue(workspace, queue_id).await
    }

    pub async fn delete_review_queue(
        &self,
        workspace: &str,
        queue_id: &str,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let queue_scope = format!(
            "SELECT rq.queue_id FROM review_queues rq JOIN experiments e ON e.experiment_id = \
             rq.experiment_id WHERE rq.queue_id = {} AND e.workspace = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let values = [
            Val::Text(queue_id.to_string()),
            Val::Text(workspace.to_string()),
        ];
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        for table in [
            "review_queue_users",
            "review_queue_items",
            "review_queue_label_schemas",
        ] {
            tx.exec(
                &format!("DELETE FROM {table} WHERE queue_id IN ({queue_scope})"),
                &values,
            )
            .await
            .map_err(internal)?;
        }
        tx.exec(
            &format!("DELETE FROM review_queues WHERE queue_id IN ({queue_scope})"),
            &values,
        )
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)
    }

    pub async fn add_items_to_review_queue(
        &self,
        workspace: &str,
        queue_id: &str,
        item_ids: &[String],
        item_type: ReviewItemType,
    ) -> Result<Vec<ReviewQueueItem>, MlflowError> {
        let item_ids = normalize_item_ids(item_ids)?;
        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;
        let queue_exists = tx
            .fetch_all(
                &format!(
                    "SELECT rq.queue_id FROM review_queues rq JOIN experiments e ON \
                     e.experiment_id = rq.experiment_id WHERE rq.queue_id = {} AND e.workspace = {}{}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    if dialect == crate::Dialect::Sqlite {
                        ""
                    } else {
                        " FOR UPDATE"
                    }
                ),
                &[
                    Val::Text(queue_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| row.get_string("queue_id"),
            )
            .await
            .map_err(internal)?;
        if queue_exists.is_empty() {
            return Err(review_queue_not_found(queue_id));
        }

        let now = now_millis();
        let insert = dialect.upsert(&UpsertSpec {
            table: "review_queue_items",
            columns: &[
                "queue_id",
                "item_type",
                "item_id",
                "status",
                "completed_by",
                "completed_time_ms",
                "creation_time_ms",
                "last_update_time_ms",
            ],
            pk_columns: &["queue_id", "item_id"],
            update_columns: &[],
            json_columns: &[],
        });
        for item_id in &item_ids {
            tx.exec(
                &insert,
                &[
                    Val::Text(queue_id.to_string()),
                    Val::Text(item_type.as_str().to_string()),
                    Val::Text(item_id.clone()),
                    Val::Text(ReviewStatus::Pending.as_str().to_string()),
                    Val::OptText(None),
                    Val::OptInt(None),
                    Val::Int(now),
                    Val::Int(now),
                ],
            )
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)?;
        let rows = self.find_review_queue_items(queue_id, &item_ids).await?;
        let mut by_id = rows
            .into_iter()
            .map(|item| (item.item_id.clone(), item))
            .collect::<HashMap<_, _>>();
        Ok(item_ids
            .iter()
            .filter_map(|item_id| by_id.remove(item_id))
            .collect())
    }

    pub async fn remove_items_from_review_queue(
        &self,
        workspace: &str,
        queue_id: &str,
        item_ids: &[String],
    ) -> Result<(), MlflowError> {
        let item_ids = normalize_item_ids(item_ids)?;
        self.get_review_queue(workspace, queue_id).await?;
        let (placeholders, mut values) = in_values(self.db().dialect(), &item_ids, 2);
        values.insert(0, Val::Text(queue_id.to_string()));
        self.db()
            .exec(
                &format!(
                    "DELETE FROM review_queue_items WHERE queue_id = {} AND item_id IN ({})",
                    self.db().dialect().placeholder(1),
                    placeholders.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    pub async fn list_review_queue_items(
        &self,
        workspace: &str,
        queue_id: &str,
        status: Option<ReviewStatus>,
        max_results: Option<i32>,
        page_token: Option<&str>,
    ) -> Result<ReviewQueueItemsPage, MlflowError> {
        let max_results = validate_max_results(max_results)?;
        let offset = parse_page_token(page_token)?;
        self.get_review_queue(workspace, queue_id).await?;
        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(queue_id.to_string())];
        let mut predicate = format!("queue_id = {}", dialect.placeholder(1));
        if let Some(status) = status {
            values.push(Val::Text(status.as_str().to_string()));
            predicate.push_str(&format!(
                " AND status = {}",
                dialect.placeholder(values.len())
            ));
        }
        let mut items = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT queue_id, item_type, item_id, status, completed_by, \
                     completed_time_ms, creation_time_ms, last_update_time_ms FROM \
                     review_queue_items WHERE {predicate} ORDER BY creation_time_ms DESC, \
                     item_id ASC LIMIT {} OFFSET {}",
                    max_results + 1,
                    offset
                ),
                &values,
                map_review_queue_item,
            )
            .await
            .map_err(internal)?;
        let next_page_token = if items.len() > max_results as usize {
            items.truncate(max_results as usize);
            Some(mlflow_search::create_page_token(offset + max_results))
        } else {
            None
        };
        Ok(ReviewQueueItemsPage {
            items,
            next_page_token,
        })
    }

    pub async fn set_review_queue_item_status(
        &self,
        workspace: &str,
        queue_id: &str,
        item_id: &str,
        status: ReviewStatus,
        completed_by: Option<&str>,
    ) -> Result<ReviewQueueItem, MlflowError> {
        let item_id = normalize_item_id(item_id)?;
        let completed_by = completed_by.map(normalize_user);
        match status {
            ReviewStatus::Pending if completed_by.is_some() => {
                return Err(MlflowError::invalid_parameter_value(
                    "`completed_by` must not be set when reopening an item to `pending`.",
                ));
            }
            ReviewStatus::Complete | ReviewStatus::Declined
                if completed_by.as_deref().unwrap_or_default().is_empty() =>
            {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "`completed_by` is required when setting status to `{}`.",
                    status.as_str()
                )));
            }
            _ => {}
        }
        if let Some(completed_by) = &completed_by {
            let length = completed_by.chars().count();
            if length > USER_MAX_LENGTH {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "`completed_by` must be at most {USER_MAX_LENGTH} characters; got {length}."
                )));
            }
        }
        self.get_review_queue(workspace, queue_id).await?;
        let existing = self
            .find_review_queue_items(queue_id, std::slice::from_ref(&item_id))
            .await?
            .pop()
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Item '{item_id}' is not attached to review queue '{queue_id}'."
                ))
            })?;
        if existing.status == status {
            return Ok(existing);
        }

        let dialect = self.db().dialect();
        let now = now_millis();
        let (actor, completed_time) = if status == ReviewStatus::Pending {
            (None, None)
        } else {
            (completed_by, Some(now))
        };
        self.db()
            .exec(
                &format!(
                    "UPDATE review_queue_items SET status = {}, last_update_time_ms = {}, \
                     completed_by = {}, completed_time_ms = {} WHERE queue_id = {} AND item_id = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                    dialect.placeholder(4),
                    dialect.placeholder(5),
                    dialect.placeholder(6)
                ),
                &[
                    Val::Text(status.as_str().to_string()),
                    Val::Int(now),
                    Val::OptText(actor),
                    Val::OptInt(completed_time),
                    Val::Text(queue_id.to_string()),
                    Val::Text(item_id.clone()),
                ],
            )
            .await
            .map_err(internal)?;
        self.find_review_queue_items(queue_id, std::slice::from_ref(&item_id))
            .await?
            .pop()
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Item '{item_id}' is not attached to review queue '{queue_id}'."
                ))
            })
    }

    /// Resolve the queue's live questions without changing its literal wire entity.
    /// User queues inherit every current experiment schema; custom queues retain only
    /// their attached ids that still resolve after a soft-referenced schema is deleted.
    pub async fn resolve_review_queue_schema_ids(
        &self,
        workspace: &str,
        queue_id: &str,
    ) -> Result<Vec<String>, MlflowError> {
        let queue = self.get_review_queue(workspace, queue_id).await?;
        let schemas = self
            .list_label_schemas(
                workspace,
                &queue.experiment_id,
                SEARCH_MAX_RESULTS_THRESHOLD as i32,
                None,
            )
            .await?
            .schemas;
        if queue.queue_type == ReviewQueueType::User {
            return Ok(schemas.into_iter().map(|schema| schema.schema_id).collect());
        }
        let live = schemas
            .into_iter()
            .map(|schema| schema.schema_id)
            .collect::<HashSet<_>>();
        Ok(queue
            .schema_ids
            .into_iter()
            .filter(|schema_id| live.contains(schema_id))
            .collect())
    }

    async fn validate_review_queue_schema_ids(
        &self,
        workspace: &str,
        experiment_id: i64,
        schema_ids: &[String],
    ) -> Result<(), MlflowError> {
        if schema_ids.is_empty() {
            return Ok(());
        }
        let dialect = self.db().dialect();
        let (placeholders, mut values) = in_values(dialect, schema_ids, 3);
        values.insert(0, Val::Int(experiment_id));
        values.insert(1, Val::Text(workspace.to_string()));
        let found = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT ls.schema_id FROM label_schemas ls JOIN experiments e ON \
                     e.experiment_id = ls.experiment_id WHERE ls.experiment_id = {} AND \
                     e.workspace = {} AND ls.schema_id IN ({})",
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    placeholders.join(", ")
                ),
                &values,
                |row| row.get_string("schema_id"),
            )
            .await
            .map_err(internal)?
            .into_iter()
            .collect::<HashSet<_>>();
        let missing = schema_ids
            .iter()
            .filter(|schema_id| !found.contains(*schema_id))
            .cloned()
            .collect::<Vec<_>>();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(MlflowError::invalid_parameter_value(format!(
                "Label schema id(s) {} not found for experiment '{experiment_id}'.",
                python_string_list(&missing)
            )))
        }
    }

    async fn find_review_queue_by_id(
        &self,
        workspace: &str,
        queue_id: &str,
    ) -> Result<Option<ReviewQueue>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "{} WHERE rq.queue_id = {} AND e.workspace = {}",
                    review_queue_select(),
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(queue_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_review_queue_row,
            )
            .await
            .map_err(internal)
    }

    async fn find_review_queue_by_name(
        &self,
        workspace: &str,
        experiment_id: i64,
        name: &str,
    ) -> Result<Option<ReviewQueue>, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "{} WHERE rq.experiment_id = {} AND rq.name_key = {} AND e.workspace = {}",
                    review_queue_select(),
                    dialect.placeholder(1),
                    dialect.placeholder(2),
                    dialect.placeholder(3)
                ),
                &[
                    Val::Int(experiment_id),
                    Val::Text(name.to_lowercase()),
                    Val::Text(workspace.to_string()),
                ],
                map_review_queue_row,
            )
            .await
            .map_err(internal)
    }

    async fn hydrate_review_queues(
        &self,
        mut queues: Vec<ReviewQueue>,
    ) -> Result<Vec<ReviewQueue>, MlflowError> {
        if queues.is_empty() {
            return Ok(queues);
        }
        let queue_ids = queues
            .iter()
            .map(|queue| queue.queue_id.clone())
            .collect::<Vec<_>>();
        let dialect = self.db().dialect();
        let (placeholders, values) = in_values(dialect, &queue_ids, 1);
        let users = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT queue_id, user_id FROM review_queue_users WHERE queue_id IN ({}) \
                     ORDER BY user_id ASC",
                    placeholders.join(", ")
                ),
                &values,
                |row| Ok((row.get_string("queue_id")?, row.get_string("user_id")?)),
            )
            .await
            .map_err(internal)?;
        let schema_ids = self
            .db()
            .fetch_all(
                &format!(
                    "SELECT queue_id, schema_id FROM review_queue_label_schemas WHERE queue_id IN \
                     ({}) ORDER BY schema_id ASC",
                    placeholders.join(", ")
                ),
                &values,
                |row| Ok((row.get_string("queue_id")?, row.get_string("schema_id")?)),
            )
            .await
            .map_err(internal)?;
        let mut users_by_queue: HashMap<String, Vec<String>> = HashMap::new();
        for (queue_id, user) in users {
            users_by_queue.entry(queue_id).or_default().push(user);
        }
        let mut schemas_by_queue: HashMap<String, Vec<String>> = HashMap::new();
        for (queue_id, schema_id) in schema_ids {
            schemas_by_queue
                .entry(queue_id)
                .or_default()
                .push(schema_id);
        }
        for queue in &mut queues {
            queue.users = users_by_queue.remove(&queue.queue_id).unwrap_or_default();
            queue.schema_ids = schemas_by_queue.remove(&queue.queue_id).unwrap_or_default();
        }
        Ok(queues)
    }

    async fn find_review_queue_items(
        &self,
        queue_id: &str,
        item_ids: &[String],
    ) -> Result<Vec<ReviewQueueItem>, MlflowError> {
        if item_ids.is_empty() {
            return Ok(Vec::new());
        }
        let dialect = self.db().dialect();
        let (placeholders, mut values) = in_values(dialect, item_ids, 2);
        values.insert(0, Val::Text(queue_id.to_string()));
        self.db()
            .fetch_all(
                &format!(
                    "SELECT queue_id, item_type, item_id, status, completed_by, \
                     completed_time_ms, creation_time_ms, last_update_time_ms FROM \
                     review_queue_items WHERE queue_id = {} AND item_id IN ({})",
                    dialect.placeholder(1),
                    placeholders.join(", ")
                ),
                &values,
                map_review_queue_item,
            )
            .await
            .map_err(internal)
    }
}

fn review_queue_select() -> &'static str {
    "SELECT rq.queue_id, rq.experiment_id, rq.name, rq.queue_type, rq.created_by, \
     rq.creation_time_ms, rq.last_update_time_ms FROM review_queues rq JOIN experiments e ON \
     e.experiment_id = rq.experiment_id"
}

fn map_review_queue_row(row: &dyn RowLike) -> Result<ReviewQueue, sqlx::Error> {
    let queue_type = match row.get_string("queue_type")?.as_str() {
        "user" => ReviewQueueType::User,
        "custom" => ReviewQueueType::Custom,
        other => {
            return Err(sqlx::Error::Decode(
                format!("unknown queue type {other:?}").into(),
            ))
        }
    };
    Ok(ReviewQueue {
        queue_id: row.get_string("queue_id")?,
        experiment_id: row.get_int("experiment_id")?.to_string(),
        name: row.get_string("name")?,
        queue_type,
        created_by: row.get_opt_string("created_by")?,
        creation_time_ms: row.get_i64("creation_time_ms")?,
        last_update_time_ms: row.get_i64("last_update_time_ms")?,
        users: Vec::new(),
        schema_ids: Vec::new(),
    })
}

fn map_review_queue_item(row: &dyn RowLike) -> Result<ReviewQueueItem, sqlx::Error> {
    let item_type = match row.get_string("item_type")?.as_str() {
        "trace" => ReviewItemType::Trace,
        other => {
            return Err(sqlx::Error::Decode(
                format!("unknown item type {other:?}").into(),
            ))
        }
    };
    let status = match row.get_string("status")?.as_str() {
        "pending" => ReviewStatus::Pending,
        "complete" => ReviewStatus::Complete,
        "declined" => ReviewStatus::Declined,
        other => {
            return Err(sqlx::Error::Decode(
                format!("unknown review status {other:?}").into(),
            ))
        }
    };
    Ok(ReviewQueueItem {
        queue_id: row.get_string("queue_id")?,
        item_type,
        item_id: row.get_string("item_id")?,
        status,
        completed_by: row.get_opt_string("completed_by")?,
        completed_time_ms: row.get_opt_i64("completed_time_ms")?,
        creation_time_ms: row.get_i64("creation_time_ms")?,
        last_update_time_ms: row.get_i64("last_update_time_ms")?,
    })
}

async fn insert_users(
    tx: &mut Tx<'_>,
    dialect: crate::Dialect,
    queue_id: &str,
    users: &[String],
) -> Result<(), MlflowError> {
    for user in users {
        tx.exec(
            &format!(
                "INSERT INTO review_queue_users (queue_id, user_id) VALUES ({}, {})",
                dialect.placeholder(1),
                dialect.placeholder(2)
            ),
            &[Val::Text(queue_id.to_string()), Val::Text(user.clone())],
        )
        .await
        .map_err(internal)?;
    }
    Ok(())
}

async fn insert_schema_ids(
    tx: &mut Tx<'_>,
    dialect: crate::Dialect,
    queue_id: &str,
    schema_ids: &[String],
) -> Result<(), MlflowError> {
    for schema_id in schema_ids {
        tx.exec(
            &format!(
                "INSERT INTO review_queue_label_schemas (queue_id, schema_id) VALUES ({}, {})",
                dialect.placeholder(1),
                dialect.placeholder(2)
            ),
            &[
                Val::Text(queue_id.to_string()),
                Val::Text(schema_id.clone()),
            ],
        )
        .await
        .map_err(internal)?;
    }
    Ok(())
}

fn validate_queue_for_create(
    name: &str,
    queue_type: ReviewQueueType,
    users: &[String],
    schema_ids: &[String],
) -> Result<ValidatedQueue, MlflowError> {
    let mut users = normalize_users(users)?;
    let schema_ids = normalize_schema_ids(schema_ids)?;
    let name = match queue_type {
        ReviewQueueType::User => {
            let name = normalize_user(name);
            validate_non_empty(&name, "name", QUEUE_NAME_MAX_LENGTH)?;
            if users.is_empty() {
                users.push(name.clone());
            } else if users != [name.clone()] {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "A user queue must have exactly one assigned user equal to its name; got \
                     name='{name}' and users={}.",
                    python_string_list(&users)
                )));
            }
            if !schema_ids.is_empty() {
                return Err(MlflowError::invalid_parameter_value(
                    "A user queue cannot have explicitly-attached schemas; it resolves to all of the experiment's label schemas.",
                ));
            }
            name
        }
        ReviewQueueType::Custom => validate_custom_queue_name(name)?,
    };
    Ok(ValidatedQueue {
        name,
        users,
        schema_ids,
    })
}

fn validate_custom_queue_name(name: &str) -> Result<String, MlflowError> {
    let name = name.trim().to_string();
    validate_non_empty(&name, "name", QUEUE_NAME_MAX_LENGTH)?;
    if name.to_lowercase() == RESERVED_QUEUE_NAME {
        return Err(MlflowError::invalid_parameter_value(format!(
            "`{name}` is a reserved queue name and cannot be used for a custom queue."
        )));
    }
    Ok(name)
}

fn validate_queue_owner(owner: &str) -> Result<String, MlflowError> {
    let owner = owner.trim().to_string();
    validate_non_empty(&owner, "new_owner", USER_MAX_LENGTH)?;
    Ok(owner)
}

fn normalize_users(users: &[String]) -> Result<Vec<String>, MlflowError> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for user in users {
        let user = normalize_user(user);
        validate_non_empty(&user, "user", USER_MAX_LENGTH)?;
        if seen.insert(user.clone()) {
            normalized.push(user);
        }
    }
    if normalized.len() > MAX_ASSIGNED_USERS {
        return Err(MlflowError::invalid_parameter_value(format!(
            "A review queue can have at most {MAX_ASSIGNED_USERS} assigned users; got {}.",
            normalized.len()
        )));
    }
    Ok(normalized)
}

fn normalize_schema_ids(schema_ids: &[String]) -> Result<Vec<String>, MlflowError> {
    normalize_string_ids(schema_ids, "schema_id", SCHEMA_ID_MAX_LENGTH)
}

fn normalize_item_ids(item_ids: &[String]) -> Result<Vec<String>, MlflowError> {
    if item_ids.is_empty() {
        return Err(MlflowError::invalid_parameter_value(
            "`item_ids` must be a non-empty list; got [].",
        ));
    }
    normalize_string_ids(item_ids, "item_id", ITEM_ID_MAX_LENGTH)
}

fn normalize_string_ids(
    values: &[String],
    field: &str,
    max_length: usize,
) -> Result<Vec<String>, MlflowError> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for value in values {
        let value = value.trim().to_string();
        validate_non_empty(&value, field, max_length)?;
        if seen.insert(value.clone()) {
            normalized.push(value);
        }
    }
    Ok(normalized)
}

fn normalize_item_id(item_id: &str) -> Result<String, MlflowError> {
    let item_id = item_id.trim().to_string();
    validate_non_empty(&item_id, "item_id", ITEM_ID_MAX_LENGTH)?;
    Ok(item_id)
}

fn normalize_user(user: &str) -> String {
    user.trim().to_lowercase()
}

fn validate_non_empty(value: &str, field: &str, max_length: usize) -> Result<(), MlflowError> {
    if value.is_empty() {
        return Err(MlflowError::invalid_parameter_value(format!(
            "`{field}` must be a non-empty string; got ''."
        )));
    }
    let length = value.chars().count();
    if length > max_length {
        return Err(MlflowError::invalid_parameter_value(format!(
            "`{field}` must be at most {max_length} characters; got {length}."
        )));
    }
    Ok(())
}

fn validate_max_results(max_results: Option<i32>) -> Result<i64, MlflowError> {
    let max_results = i64::from(max_results.unwrap_or(SEARCH_MAX_RESULTS_DEFAULT as i32));
    if !(1..=SEARCH_MAX_RESULTS_THRESHOLD).contains(&max_results) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "max_results must be between 1 and {SEARCH_MAX_RESULTS_THRESHOLD}."
        )));
    }
    Ok(max_results)
}

fn parse_page_token(page_token: Option<&str>) -> Result<i64, MlflowError> {
    mlflow_search::parse_start_offset_from_page_token(page_token)
        .map_err(|error| MlflowError::invalid_parameter_value(error.message))
}

fn review_queue_not_found(queue_id: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("Review queue with id '{queue_id}' not found."))
}

fn review_queue_exists(name: &str) -> MlflowError {
    MlflowError::resource_already_exists(format!(
        "Review queue with name '{name}' already exists (names are case-insensitive)."
    ))
}

fn in_values(
    dialect: crate::Dialect,
    values: &[String],
    start_index: usize,
) -> (Vec<String>, Vec<Val>) {
    (
        (0..values.len())
            .map(|offset| dialect.placeholder(start_index + offset))
            .collect(),
        values.iter().cloned().map(Val::Text).collect(),
    )
}

fn python_string_list(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
