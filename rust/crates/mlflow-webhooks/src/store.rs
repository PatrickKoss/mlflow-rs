//! The [`WebhookStore`]: webhook CRUD + events lookup, mirroring the webhook
//! methods of `mlflow/store/model_registry/sqlalchemy_store.py:1601-1794`.
//!
//! ## Workspace scoping (plan §3.14/§3.17)
//!
//! The `webhooks` row carries a `workspace` column; `webhook_events` does not
//! (it hangs off `webhooks` by `webhook_id`). Every method takes an explicit
//! `workspace: &str` and filters `webhooks.workspace = ?`, so a lookup in the
//! wrong workspace yields the same `RESOURCE_DOES_NOT_EXIST` as a genuinely
//! missing webhook — matching `WorkspaceAwareSqlAlchemyStore`.
//!
//! ## Secret encryption
//!
//! The `secret` column is Fernet-encrypted on write and decrypted on read via
//! [`crate::crypto::SecretCipher`], reproducing the `EncryptedString`
//! `TypeDecorator`. The decrypted plaintext lives on the returned [`Webhook`]
//! entity but is never emitted to the proto response (the handler layer).
//!
//! ## Soft delete
//!
//! `delete_webhook` sets `deleted_timestamp` (and bumps `last_updated_timestamp`
//! to the same value); every read filters `deleted_timestamp IS NULL`.

use mlflow_error::MlflowError;
use mlflow_search::parse_start_offset_from_page_token;
use mlflow_store::Db;

use crate::crypto::SecretCipher;
use crate::dbutil::{DbExt, RowLike, Tx, Val};
use crate::entities::{Webhook, WebhookEvent, WebhookStatus};
use crate::schema::{WEBHOOKS, WEBHOOK_EVENTS};
use crate::validation::{
    validate_event_combination, validate_webhook_events, validate_webhook_name,
    validate_webhook_url,
};

/// A page of webhooks plus the next-page token (`PagedList`).
#[derive(Debug, Clone, PartialEq)]
pub struct WebhookPage {
    pub webhooks: Vec<Webhook>,
    pub next_page_token: Option<String>,
}

/// The webhook store: CRUD over `webhooks` + `webhook_events`.
///
/// Holds a [`Db`] pool and a [`SecretCipher`] built once at construction
/// (Python builds the `EncryptedString` cipher once per column, from
/// `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`).
#[derive(Debug, Clone)]
pub struct WebhookStore {
    db: Db,
    cipher: SecretCipher,
}

impl WebhookStore {
    /// Build a store over an already-connected/verified [`Db`], resolving the
    /// secret cipher from `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`.
    pub fn new(db: Db) -> Result<Self, MlflowError> {
        Ok(Self {
            db,
            cipher: SecretCipher::from_env()?,
        })
    }

    /// Build a store with an explicit cipher (used by tests / cross-language
    /// fixtures that pin a known key).
    pub fn with_cipher(db: Db, cipher: SecretCipher) -> Self {
        Self { db, cipher }
    }

    /// The underlying database pool.
    pub fn db(&self) -> &Db {
        &self.db
    }

    /// The secret cipher (never logs key material).
    pub fn cipher(&self) -> &SecretCipher {
        &self.cipher
    }

    /// `create_webhook` (`sqlalchemy_store.py:1602`). Validates name/url/events,
    /// generates a uuid4 id, inserts the webhook + its events in one
    /// transaction, and returns the created entity.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_webhook(
        &self,
        workspace: &str,
        name: &str,
        url: &str,
        events: &[WebhookEvent],
        description: Option<&str>,
        secret: Option<&str>,
        status: Option<WebhookStatus>,
    ) -> Result<Webhook, MlflowError> {
        validate_webhook_name(name)?;
        validate_webhook_url(url)?;
        validate_webhook_events(events)?;
        for e in events {
            validate_event_combination(e.entity, e.action)?;
        }

        let webhook_id = new_uuid();
        let now = now_millis();
        let status = status.unwrap_or(WebhookStatus::Active);
        let encrypted_secret = self.cipher.encrypt(secret);
        let dialect = self.db.dialect();
        let ph = |i| dialect.placeholder(i);

        let mut tx = self.db.begin_tx().await.map_err(internal)?;
        let insert_sql = format!(
            "INSERT INTO {WEBHOOKS} \
             (webhook_id, name, description, url, status, secret, \
              creation_timestamp, last_updated_timestamp, deleted_timestamp, workspace) \
             VALUES ({}, {}, {}, {}, {}, {}, {}, {}, NULL, {})",
            ph(1),
            ph(2),
            ph(3),
            ph(4),
            ph(5),
            ph(6),
            ph(7),
            ph(8),
            ph(9),
        );
        tx.exec(
            &insert_sql,
            &[
                Val::Text(webhook_id.clone()),
                Val::Text(name.to_string()),
                Val::OptText(description.map(str::to_string)),
                Val::Text(url.to_string()),
                Val::Text(status.as_db_str().to_string()),
                Val::OptText(encrypted_secret),
                Val::Int(now),
                Val::Int(now),
                Val::Text(workspace.to_string()),
            ],
        )
        .await
        .map_err(internal)?;

        insert_events(&mut tx, &dialect, &webhook_id, events).await?;
        tx.commit().await.map_err(internal)?;

        self.get_webhook(workspace, &webhook_id).await
    }

    /// `get_webhook` (`sqlalchemy_store.py:1642`). Errors
    /// `RESOURCE_DOES_NOT_EXIST` when absent or soft-deleted.
    pub async fn get_webhook(
        &self,
        workspace: &str,
        webhook_id: &str,
    ) -> Result<Webhook, MlflowError> {
        let row = self
            .fetch_webhook_row(workspace, webhook_id)
            .await?
            .ok_or_else(|| not_found(webhook_id))?;
        let events = self.fetch_events(webhook_id).await?;
        self.to_entity(row, events)
    }

    /// `list_webhooks` (`sqlalchemy_store.py:1647`). Offset pagination over
    /// non-deleted webhooks, ordered by `creation_timestamp DESC`.
    pub async fn list_webhooks(
        &self,
        workspace: &str,
        max_results: Option<i32>,
        page_token: Option<&str>,
    ) -> Result<WebhookPage, MlflowError> {
        let max_results = max_results.unwrap_or(100);
        if !(1..=1000).contains(&max_results) {
            return Err(MlflowError::invalid_parameter_value(
                "max_results must be between 1 and 1000.".to_string(),
            ));
        }
        let offset = parse_start_offset_from_page_token(page_token)
            .map_err(|e| MlflowError::invalid_parameter_value(e.to_string()))?;

        let rows = self
            .list_webhook_rows(workspace, None, i64::from(max_results), offset, page_token)
            .await?;
        self.paginate(rows, offset, max_results).await
    }

    /// `list_webhooks_by_event` (`sqlalchemy_store.py:1682`): non-deleted
    /// webhooks subscribed to a specific event. Used by the delivery engine
    /// (T8.3); included here so the store CRUD surface is complete.
    pub async fn list_webhooks_by_event(
        &self,
        workspace: &str,
        event: WebhookEvent,
        max_results: Option<i32>,
        page_token: Option<&str>,
    ) -> Result<WebhookPage, MlflowError> {
        let max_results = max_results.unwrap_or(100);
        if !(1..=1000).contains(&max_results) {
            return Err(MlflowError::invalid_parameter_value(
                "max_results must be between 1 and 1000.".to_string(),
            ));
        }
        let offset = parse_start_offset_from_page_token(page_token)
            .map_err(|e| MlflowError::invalid_parameter_value(e.to_string()))?;

        let rows = self
            .list_webhook_rows(
                workspace,
                Some(event),
                i64::from(max_results),
                offset,
                page_token,
            )
            .await?;
        self.paginate(rows, offset, max_results).await
    }

    /// `update_webhook` (`sqlalchemy_store.py:1722`). Partial update: any
    /// provided field is validated and applied; `events` fully replaces the
    /// event set. `last_updated_timestamp` is always bumped.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_webhook(
        &self,
        workspace: &str,
        webhook_id: &str,
        name: Option<&str>,
        description: Option<&str>,
        url: Option<&str>,
        events: Option<&[WebhookEvent]>,
        secret: Option<&str>,
        status: Option<WebhookStatus>,
    ) -> Result<Webhook, MlflowError> {
        // Existence check (workspace-scoped, non-deleted).
        self.fetch_webhook_row(workspace, webhook_id)
            .await?
            .ok_or_else(|| not_found(webhook_id))?;

        if let Some(n) = name {
            validate_webhook_name(n)?;
        }
        if let Some(u) = url {
            validate_webhook_url(u)?;
        }
        if let Some(evs) = events {
            validate_webhook_events(evs)?;
            for e in evs {
                validate_event_combination(e.entity, e.action)?;
            }
        }

        let now = now_millis();
        let dialect = self.db.dialect();
        let mut tx = self.db.begin_tx().await.map_err(internal)?;

        // Build the dynamic SET clause. Each provided field is one assignment;
        // `last_updated_timestamp` is always set.
        let mut sets: Vec<String> = Vec::new();
        let mut vals: Vec<Val> = Vec::new();
        let mut idx = 1usize;
        let mut push = |col: &str, val: Val, sets: &mut Vec<String>, vals: &mut Vec<Val>| {
            sets.push(format!("{col} = {}", dialect.placeholder(idx)));
            idx += 1;
            vals.push(val);
        };
        if let Some(n) = name {
            push("name", Val::Text(n.to_string()), &mut sets, &mut vals);
        }
        if let Some(d) = description {
            push(
                "description",
                Val::OptText(Some(d.to_string())),
                &mut sets,
                &mut vals,
            );
        }
        if let Some(u) = url {
            push("url", Val::Text(u.to_string()), &mut sets, &mut vals);
        }
        if let Some(s) = secret {
            let enc = self.cipher.encrypt(Some(s));
            push("secret", Val::OptText(enc), &mut sets, &mut vals);
        }
        if let Some(st) = status {
            push(
                "status",
                Val::Text(st.as_db_str().to_string()),
                &mut sets,
                &mut vals,
            );
        }
        push(
            "last_updated_timestamp",
            Val::Int(now),
            &mut sets,
            &mut vals,
        );

        let update_sql = format!(
            "UPDATE {WEBHOOKS} SET {} WHERE webhook_id = {} AND workspace = {} \
             AND deleted_timestamp IS NULL",
            sets.join(", "),
            dialect.placeholder(idx),
            dialect.placeholder(idx + 1),
        );
        vals.push(Val::Text(webhook_id.to_string()));
        vals.push(Val::Text(workspace.to_string()));
        tx.exec(&update_sql, &vals).await.map_err(internal)?;

        if let Some(evs) = events {
            let del_sql = format!(
                "DELETE FROM {WEBHOOK_EVENTS} WHERE webhook_id = {}",
                dialect.placeholder(1)
            );
            tx.exec(&del_sql, &[Val::Text(webhook_id.to_string())])
                .await
                .map_err(internal)?;
            insert_events(&mut tx, &dialect, webhook_id, evs).await?;
        }
        tx.commit().await.map_err(internal)?;

        self.get_webhook(workspace, webhook_id).await
    }

    /// `delete_webhook` (`sqlalchemy_store.py:1769`): soft delete by setting
    /// `deleted_timestamp` (and `last_updated_timestamp` to the same value).
    pub async fn delete_webhook(
        &self,
        workspace: &str,
        webhook_id: &str,
    ) -> Result<(), MlflowError> {
        self.fetch_webhook_row(workspace, webhook_id)
            .await?
            .ok_or_else(|| not_found(webhook_id))?;

        let now = now_millis();
        let dialect = self.db.dialect();
        let sql = format!(
            "UPDATE {WEBHOOKS} SET deleted_timestamp = {}, last_updated_timestamp = {} \
             WHERE webhook_id = {} AND workspace = {} AND deleted_timestamp IS NULL",
            dialect.placeholder(1),
            dialect.placeholder(2),
            dialect.placeholder(3),
            dialect.placeholder(4),
        );
        self.db
            .exec(
                &sql,
                &[
                    Val::Int(now),
                    Val::Int(now),
                    Val::Text(webhook_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
            )
            .await
            .map_err(internal)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    async fn paginate(
        &self,
        mut rows: Vec<WebhookRow>,
        offset: i64,
        max_results: i32,
    ) -> Result<WebhookPage, MlflowError> {
        let has_next = rows.len() as i64 > i64::from(max_results);
        let next_page_token = if has_next {
            rows.truncate(max_results as usize);
            Some(create_page_token(offset + i64::from(max_results)))
        } else {
            None
        };
        let mut webhooks = Vec::with_capacity(rows.len());
        for row in rows {
            let events = self.fetch_events(&row.webhook_id).await?;
            webhooks.push(self.to_entity(row, events)?);
        }
        Ok(WebhookPage {
            webhooks,
            next_page_token,
        })
    }

    async fn fetch_webhook_row(
        &self,
        workspace: &str,
        webhook_id: &str,
    ) -> Result<Option<WebhookRow>, MlflowError> {
        let dialect = self.db.dialect();
        let sql = format!(
            "SELECT webhook_id, name, description, url, status, secret, \
                    creation_timestamp, last_updated_timestamp, workspace \
             FROM {WEBHOOKS} \
             WHERE webhook_id = {} AND workspace = {} AND deleted_timestamp IS NULL",
            dialect.placeholder(1),
            dialect.placeholder(2),
        );
        self.db
            .fetch_optional(
                &sql,
                &[
                    Val::Text(webhook_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                map_webhook_row,
            )
            .await
            .map_err(internal)
    }

    async fn list_webhook_rows(
        &self,
        workspace: &str,
        event: Option<WebhookEvent>,
        max_results: i64,
        offset: i64,
        page_token: Option<&str>,
    ) -> Result<Vec<WebhookRow>, MlflowError> {
        let dialect = self.db.dialect();
        // `.limit(max_results + 1)` and `.offset(offset)` only when a page_token
        // was supplied (matching the Python `if page_token:` guard).
        let (join, event_filter, ws_ph, extra_vals) = match event {
            Some(ev) => {
                let join = format!(" JOIN {WEBHOOK_EVENTS} e ON e.webhook_id = w.webhook_id");
                let filter = format!(
                    " AND e.entity = {} AND e.action = {}",
                    dialect.placeholder(2),
                    dialect.placeholder(3),
                );
                (
                    join,
                    filter,
                    dialect.placeholder(1),
                    vec![
                        Val::Text(ev.entity.as_db_str().to_string()),
                        Val::Text(ev.action.as_db_str().to_string()),
                    ],
                )
            }
            None => (String::new(), String::new(), dialect.placeholder(1), vec![]),
        };

        let limit_ph = dialect.placeholder(2 + extra_vals.len());
        let mut sql = format!(
            "SELECT w.webhook_id, w.name, w.description, w.url, w.status, w.secret, \
                    w.creation_timestamp, w.last_updated_timestamp, w.workspace \
             FROM {WEBHOOKS} w{join} \
             WHERE w.workspace = {ws_ph} AND w.deleted_timestamp IS NULL{event_filter} \
             ORDER BY w.creation_timestamp DESC \
             LIMIT {limit_ph}"
        );
        let mut vals = vec![Val::Text(workspace.to_string())];
        vals.extend(extra_vals);
        vals.push(Val::Int(max_results + 1));
        if page_token.is_some() {
            let off_ph = dialect.placeholder(vals.len() + 1);
            sql.push_str(&format!(" OFFSET {off_ph}"));
            vals.push(Val::Int(offset));
        }
        self.db
            .fetch_all(&sql, &vals, map_webhook_row)
            .await
            .map_err(internal)
    }

    async fn fetch_events(&self, webhook_id: &str) -> Result<Vec<WebhookEvent>, MlflowError> {
        let dialect = self.db.dialect();
        let sql = format!(
            "SELECT entity, action FROM {WEBHOOK_EVENTS} WHERE webhook_id = {} \
             ORDER BY entity, action",
            dialect.placeholder(1)
        );
        let raw = self
            .db
            .fetch_all(
                &sql,
                &[Val::Text(webhook_id.to_string())],
                |row: &dyn RowLike| Ok((row.get_string("entity")?, row.get_string("action")?)),
            )
            .await
            .map_err(internal)?;
        let mut events = Vec::with_capacity(raw.len());
        for (entity, action) in raw {
            let entity = crate::entities::WebhookEntity::from_db_str(&entity).ok_or_else(|| {
                MlflowError::internal_error(format!(
                    "Unknown webhook entity in database: {entity:?}"
                ))
            })?;
            let action = crate::entities::WebhookAction::from_db_str(&action).ok_or_else(|| {
                MlflowError::internal_error(format!(
                    "Unknown webhook action in database: {action:?}"
                ))
            })?;
            events.push(WebhookEvent::new(entity, action));
        }
        Ok(events)
    }

    fn to_entity(
        &self,
        row: WebhookRow,
        events: Vec<WebhookEvent>,
    ) -> Result<Webhook, MlflowError> {
        let secret = self.cipher.decrypt(row.secret.as_deref())?;
        Ok(Webhook {
            webhook_id: row.webhook_id,
            name: row.name,
            url: row.url,
            events,
            description: row.description,
            status: WebhookStatus::from_db_str(&row.status)?,
            secret,
            creation_timestamp: row.creation_timestamp,
            last_updated_timestamp: row.last_updated_timestamp,
            workspace: row.workspace,
        })
    }
}

/// A raw `webhooks` row (secret still encrypted).
struct WebhookRow {
    webhook_id: String,
    name: String,
    description: Option<String>,
    url: String,
    status: String,
    secret: Option<String>,
    creation_timestamp: Option<i64>,
    last_updated_timestamp: Option<i64>,
    workspace: String,
}

fn map_webhook_row(row: &dyn RowLike) -> Result<WebhookRow, sqlx::Error> {
    Ok(WebhookRow {
        webhook_id: row.get_string("webhook_id")?,
        name: row.get_string("name")?,
        description: row.get_opt_string("description")?,
        url: row.get_string("url")?,
        status: row.get_string("status")?,
        secret: row.get_opt_string("secret")?,
        creation_timestamp: row.get_opt_i64("creation_timestamp")?,
        last_updated_timestamp: row.get_opt_i64("last_updated_timestamp")?,
        workspace: row.get_string("workspace")?,
    })
}

async fn insert_events(
    tx: &mut Tx<'_>,
    dialect: &mlflow_store::Dialect,
    webhook_id: &str,
    events: &[WebhookEvent],
) -> Result<(), MlflowError> {
    // Dedup by (entity, action) — the PK is (webhook_id, entity, action), so a
    // duplicate pair would violate it. Python builds distinct SqlWebhookEvent
    // rows; a client repeating a pair is not expected, but we keep the first.
    let mut seen: Vec<(
        crate::entities::WebhookEntity,
        crate::entities::WebhookAction,
    )> = Vec::new();
    let sql = format!(
        "INSERT INTO {WEBHOOK_EVENTS} (webhook_id, entity, action) VALUES ({}, {}, {})",
        dialect.placeholder(1),
        dialect.placeholder(2),
        dialect.placeholder(3),
    );
    for e in events {
        if seen.contains(&(e.entity, e.action)) {
            continue;
        }
        seen.push((e.entity, e.action));
        tx.exec(
            &sql,
            &[
                Val::Text(webhook_id.to_string()),
                Val::Text(e.entity.as_db_str().to_string()),
                Val::Text(e.action.as_db_str().to_string()),
            ],
        )
        .await
        .map_err(internal)?;
    }
    Ok(())
}

/// `get_current_time_millis()`.
fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// `str(uuid.uuid4())`.
fn new_uuid() -> String {
    // A minimal v4 UUID formatter without pulling the `uuid` crate: 16 random
    // bytes with the version/variant bits set, hyphenated. `getrandom` via
    // `fernet`'s RNG isn't exposed, so use the OS RNG through `std`.
    let mut bytes = [0u8; 16];
    fill_random(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
    let h = |b: u8| format!("{b:02x}");
    format!(
        "{}{}{}{}-{}{}-{}{}-{}{}-{}{}{}{}{}{}",
        h(bytes[0]),
        h(bytes[1]),
        h(bytes[2]),
        h(bytes[3]),
        h(bytes[4]),
        h(bytes[5]),
        h(bytes[6]),
        h(bytes[7]),
        h(bytes[8]),
        h(bytes[9]),
        h(bytes[10]),
        h(bytes[11]),
        h(bytes[12]),
        h(bytes[13]),
        h(bytes[14]),
        h(bytes[15]),
    )
}

/// Fill a buffer with OS randomness. Uses `/dev/urandom`-backed
/// `getrandom`-style entropy via a fresh Fernet key (32 bytes of CSPRNG
/// output), avoiding an extra dependency.
fn fill_random(buf: &mut [u8]) {
    use base64::Engine;
    // `Fernet::generate_key()` returns url-safe base64 of 32 CSPRNG bytes.
    let key = fernet::Fernet::generate_key();
    let raw = base64::engine::general_purpose::URL_SAFE
        .decode(key.as_bytes())
        .unwrap_or_else(|_| vec![0u8; 32]);
    for (i, b) in buf.iter_mut().enumerate() {
        *b = raw.get(i).copied().unwrap_or(0);
    }
}

/// `SearchUtils.create_page_token`: `base64(json.dumps({"offset": N}))`.
fn create_page_token(offset: i64) -> String {
    use base64::Engine;
    let json = format!("{{\"offset\": {offset}}}");
    base64::engine::general_purpose::STANDARD.encode(json.as_bytes())
}

fn internal(e: sqlx::Error) -> MlflowError {
    MlflowError::internal_error(format!("database error: {e}"))
}

/// `_get_webhook_by_id`'s not-found error
/// (`sqlalchemy_store.py:1783`): "Webhook with ID {webhook_id} not found."
fn not_found(webhook_id: &str) -> MlflowError {
    MlflowError::resource_does_not_exist(format!("Webhook with ID {webhook_id} not found."))
}
