//! Gateway budget tracking and enforcement (plan T18.6).
//!
//! This mirrors `mlflow/gateway/budget.py` and `budget_tracker/*`: fixed,
//! epoch-aligned windows; an in-process backend by default; and shared Redis
//! windows when `MLFLOW_GATEWAY_BUDGET_REDIS_URL` is configured.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, Datelike, TimeZone, Utc};
use mlflow_store::{BudgetPolicy, TrackingStore, SEARCH_MAX_RESULTS_DEFAULT};
use redis::aio::MultiplexedConnection;
use serde_json::{json, Value};
use tokio::sync::Mutex;

const REDIS_URL_ENV: &str = "MLFLOW_GATEWAY_BUDGET_REDIS_URL";
const REFRESH_INTERVAL_ENV: &str = "MLFLOW_GATEWAY_BUDGET_REFRESH_INTERVAL";
const DEFAULT_REFRESH_INTERVAL_SECONDS: u64 = 600;
const DEFAULT_WORKSPACE: &str = "default";
const KEY_PREFIX: &str = "mlflow:budget:";

const ENSURE_WINDOW_LUA: &str = r#"
local wkey = KEYS[1]
local new_start = ARGV[1]
local new_end = ARGV[2]
local ttl = tonumber(ARGV[3])

local current_start = redis.call('HGET', wkey, 'window_start')
if current_start == new_start then
    local spend = redis.call('HGET', wkey, 'cumulative_spend') or '0.0'
    local exceeded = redis.call('HGET', wkey, 'exceeded') or '0'
    return {0, spend, exceeded}
end

redis.call('HSET', wkey,
    'window_start', new_start,
    'window_end', new_end,
    'cumulative_spend', '0.0',
    'exceeded', '0')

if ttl and ttl > 0 then
    redis.call('EXPIRE', wkey, ttl)
end

return {1, '0.0', '0'}
"#;

const RECORD_COST_LUA: &str = r#"
local wkey = KEYS[1]
local cost = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

local new_spend = redis.call('HINCRBYFLOAT', wkey, 'cumulative_spend', cost)
new_spend = tonumber(new_spend)

if new_spend >= limit then
    local was_exceeded = redis.call('HGET', wkey, 'exceeded')
    if was_exceeded ~= '1' then
        redis.call('HSET', wkey, 'exceeded', '1')
        return {tostring(new_spend), 1}
    end
end

return {tostring(new_spend), 0}
"#;

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyWindow {
    pub policy: BudgetPolicy,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub cumulative_spend: f64,
    pub exceeded: bool,
}

#[derive(Clone)]
pub enum BudgetTracker {
    InMemory(Arc<InMemoryBudgetTracker>),
    Redis(Arc<RedisBudgetTracker>),
}

impl BudgetTracker {
    pub fn from_env() -> Self {
        match std::env::var(REDIS_URL_ENV)
            .ok()
            .filter(|url| !url.is_empty())
        {
            Some(url) => Self::Redis(Arc::new(RedisBudgetTracker::new(url))),
            None => Self::InMemory(Arc::new(InMemoryBudgetTracker::default())),
        }
    }

    pub async fn needs_refresh(&self) -> bool {
        let interval = refresh_interval();
        match self {
            Self::InMemory(tracker) => tracker.needs_refresh(interval).await,
            Self::Redis(tracker) => tracker.needs_refresh(interval).await,
        }
    }

    pub async fn refresh_policies(
        &self,
        policies: Vec<BudgetPolicy>,
        now: DateTime<Utc>,
    ) -> Result<Vec<PolicyWindow>, BudgetError> {
        match self {
            Self::InMemory(tracker) => tracker.refresh_policies(policies, now).await,
            Self::Redis(tracker) => tracker.refresh_policies(policies, now).await,
        }
    }

    pub async fn backfill_spend(
        &self,
        spend_by_policy: &HashMap<String, f64>,
    ) -> Result<(), BudgetError> {
        match self {
            Self::InMemory(tracker) => {
                tracker.backfill_spend(spend_by_policy).await;
                Ok(())
            }
            Self::Redis(tracker) => tracker.backfill_spend(spend_by_policy).await,
        }
    }

    pub async fn record_cost(
        &self,
        cost_usd: f64,
        workspace: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<Vec<PolicyWindow>, BudgetError> {
        match self {
            Self::InMemory(tracker) => Ok(tracker.record_cost(cost_usd, workspace, now).await),
            Self::Redis(tracker) => tracker.record_cost(cost_usd, workspace, now).await,
        }
    }

    pub async fn should_reject_request(
        &self,
        workspace: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<Option<PolicyWindow>, BudgetError> {
        match self {
            Self::InMemory(tracker) => Ok(tracker.should_reject_request(workspace, now).await),
            Self::Redis(tracker) => tracker.should_reject_request(workspace, now).await,
        }
    }

    #[cfg(test)]
    async fn get_all_windows(&self) -> Result<Vec<PolicyWindow>, BudgetError> {
        match self {
            Self::InMemory(tracker) => Ok(tracker.state.lock().await.windows.clone()),
            Self::Redis(tracker) => tracker.get_all_windows().await,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct BudgetError(String);

impl From<redis::RedisError> for BudgetError {
    fn from(error: redis::RedisError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Default)]
pub struct InMemoryBudgetTracker {
    state: Mutex<InMemoryState>,
}

#[derive(Default)]
struct InMemoryState {
    windows: Vec<PolicyWindow>,
    last_refresh: Option<Instant>,
}

impl InMemoryBudgetTracker {
    async fn needs_refresh(&self, interval: Duration) -> bool {
        self.state
            .lock()
            .await
            .last_refresh
            .is_none_or(|last| last.elapsed() >= interval)
    }

    async fn refresh_policies(
        &self,
        policies: Vec<BudgetPolicy>,
        now: DateTime<Utc>,
    ) -> Result<Vec<PolicyWindow>, BudgetError> {
        let mut state = self.state.lock().await;
        let mut old = std::mem::take(&mut state.windows)
            .into_iter()
            .map(|window| (window.policy.budget_policy_id.clone(), window))
            .collect::<HashMap<_, _>>();
        let mut windows = Vec::with_capacity(policies.len());
        for policy in policies {
            let (window_start, window_end) = window_bounds(&policy, now)?;
            if let Some(mut existing) = old.remove(&policy.budget_policy_id) {
                if existing.window_start == window_start {
                    existing.policy = policy;
                    existing.window_end = window_end;
                    windows.push(existing);
                    continue;
                }
            }
            windows.push(PolicyWindow {
                policy,
                window_start,
                window_end,
                cumulative_spend: 0.0,
                exceeded: false,
            });
        }
        state.windows = windows;
        state.last_refresh = Some(Instant::now());
        Ok(state.windows.clone())
    }

    async fn record_cost(
        &self,
        cost_usd: f64,
        workspace: Option<&str>,
        now: DateTime<Utc>,
    ) -> Vec<PolicyWindow> {
        let mut state = self.state.lock().await;
        let mut newly_exceeded = Vec::new();
        for window in &mut state.windows {
            if now >= window.window_end {
                if let Ok((start, end)) = window_bounds(&window.policy, now) {
                    window.window_start = start;
                    window.window_end = end;
                    window.cumulative_spend = 0.0;
                    window.exceeded = false;
                }
            }
            if !policy_applies(&window.policy, workspace) {
                continue;
            }
            window.cumulative_spend += cost_usd;
            if !window.exceeded && window.cumulative_spend >= window.policy.budget_amount {
                window.exceeded = true;
                newly_exceeded.push(window.clone());
            }
        }
        newly_exceeded
    }

    async fn should_reject_request(
        &self,
        workspace: Option<&str>,
        now: DateTime<Utc>,
    ) -> Option<PolicyWindow> {
        self.state.lock().await.windows.iter().find_map(|window| {
            (now < window.window_end
                && policy_applies(&window.policy, workspace)
                && window.policy.budget_action == "REJECT"
                && window.cumulative_spend >= window.policy.budget_amount)
                .then(|| window.clone())
        })
    }

    async fn backfill_spend(&self, spend_by_policy: &HashMap<String, f64>) {
        let mut state = self.state.lock().await;
        for window in &mut state.windows {
            if let Some(spend) = spend_by_policy.get(&window.policy.budget_policy_id) {
                window.cumulative_spend = window.cumulative_spend.max(*spend);
                window.exceeded = window.cumulative_spend >= window.policy.budget_amount;
            }
        }
    }
}

pub struct RedisBudgetTracker {
    url: String,
    connection: Mutex<Option<MultiplexedConnection>>,
    policy_cache: RwLock<HashMap<String, BudgetPolicy>>,
    last_refresh: Mutex<Option<Instant>>,
}

impl RedisBudgetTracker {
    fn new(url: String) -> Self {
        Self {
            url,
            connection: Mutex::new(None),
            policy_cache: RwLock::new(HashMap::new()),
            last_refresh: Mutex::new(None),
        }
    }

    async fn connection(&self) -> Result<MultiplexedConnection, BudgetError> {
        let mut slot = self.connection.lock().await;
        if let Some(connection) = slot.as_ref() {
            return Ok(connection.clone());
        }
        let client = redis::Client::open(self.url.as_str())?;
        let connection = client.get_multiplexed_async_connection().await?;
        *slot = Some(connection.clone());
        Ok(connection)
    }

    async fn needs_refresh(&self, interval: Duration) -> bool {
        self.last_refresh
            .lock()
            .await
            .is_none_or(|last| last.elapsed() >= interval)
    }

    async fn ensure_window(
        &self,
        connection: &mut MultiplexedConnection,
        policy: &BudgetPolicy,
        now: DateTime<Utc>,
    ) -> Result<(PolicyWindow, bool), BudgetError> {
        let (window_start, window_end) = window_bounds(policy, now)?;
        let ttl = (window_end - now).num_seconds() + 3600;
        let result: (i64, String, String) = redis::Script::new(ENSURE_WINDOW_LUA)
            .key(window_key(&policy.budget_policy_id))
            .arg(window_start.to_rfc3339())
            .arg(window_end.to_rfc3339())
            .arg(ttl)
            .invoke_async(connection)
            .await?;
        Ok((
            PolicyWindow {
                policy: policy.clone(),
                window_start,
                window_end,
                cumulative_spend: result.1.parse().unwrap_or(0.0),
                exceeded: result.2 == "1",
            },
            result.0 != 0,
        ))
    }

    async fn refresh_policies(
        &self,
        policies: Vec<BudgetPolicy>,
        now: DateTime<Utc>,
    ) -> Result<Vec<PolicyWindow>, BudgetError> {
        let mut connection = self.connection().await?;
        let new_ids = policies
            .iter()
            .map(|policy| policy.budget_policy_id.clone())
            .collect::<HashSet<_>>();
        let existing_ids: HashSet<String> = redis::cmd("SMEMBERS")
            .arg(policy_set_key())
            .query_async(&mut connection)
            .await?;
        let stale_ids = existing_ids.difference(&new_ids).collect::<Vec<_>>();
        if !stale_ids.is_empty() {
            let mut pipe = redis::pipe();
            for id in stale_ids {
                pipe.del(window_key(id))
                    .ignore()
                    .del(policy_key(id))
                    .ignore()
                    .srem(policy_set_key(), id)
                    .ignore();
            }
            pipe.query_async::<()>(&mut connection).await?;
        }

        if !policies.is_empty() {
            let mut pipe = redis::pipe();
            for policy in &policies {
                pipe.hset(
                    policy_key(&policy.budget_policy_id),
                    "data",
                    serialize_policy(policy),
                )
                .ignore();
            }
            pipe.query_async::<()>(&mut connection).await?;
            redis::cmd("SADD")
                .arg(policy_set_key())
                .arg(new_ids.iter().collect::<Vec<_>>())
                .exec_async(&mut connection)
                .await?;
        }
        *self.policy_cache.write().expect("policy cache poisoned") = policies
            .iter()
            .map(|policy| (policy.budget_policy_id.clone(), policy.clone()))
            .collect();

        let mut fresh = Vec::new();
        for policy in &policies {
            let (window, created) = self.ensure_window(&mut connection, policy, now).await?;
            if created {
                fresh.push(window);
            }
        }
        *self.last_refresh.lock().await = Some(Instant::now());
        Ok(fresh)
    }

    async fn load_policy(
        &self,
        connection: &mut MultiplexedConnection,
        policy_id: &str,
    ) -> Result<Option<BudgetPolicy>, BudgetError> {
        if let Some(policy) = self
            .policy_cache
            .read()
            .expect("policy cache poisoned")
            .get(policy_id)
            .cloned()
        {
            return Ok(Some(policy));
        }
        let data: Option<String> = redis::cmd("HGET")
            .arg(policy_key(policy_id))
            .arg("data")
            .query_async(connection)
            .await?;
        data.map(|data| deserialize_policy(&data)).transpose()
    }

    async fn record_cost(
        &self,
        cost_usd: f64,
        workspace: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<Vec<PolicyWindow>, BudgetError> {
        let mut connection = self.connection().await?;
        let policy_ids: Vec<String> = redis::cmd("SMEMBERS")
            .arg(policy_set_key())
            .query_async(&mut connection)
            .await?;
        let mut newly_exceeded = Vec::new();
        for policy_id in policy_ids {
            let Some(policy) = self.load_policy(&mut connection, &policy_id).await? else {
                continue;
            };
            if !policy_applies(&policy, workspace) {
                continue;
            }
            let (mut window, _) = self.ensure_window(&mut connection, &policy, now).await?;
            let result: (String, i64) = redis::Script::new(RECORD_COST_LUA)
                .key(window_key(&policy_id))
                .arg(cost_usd.to_string())
                .arg(policy.budget_amount.to_string())
                .invoke_async(&mut connection)
                .await?;
            if result.1 != 0 {
                window.cumulative_spend = result.0.parse().unwrap_or(0.0);
                window.exceeded = true;
                newly_exceeded.push(window);
            }
        }
        Ok(newly_exceeded)
    }

    async fn should_reject_request(
        &self,
        workspace: Option<&str>,
        now: DateTime<Utc>,
    ) -> Result<Option<PolicyWindow>, BudgetError> {
        let mut connection = self.connection().await?;
        let policy_ids: Vec<String> = redis::cmd("SMEMBERS")
            .arg(policy_set_key())
            .query_async(&mut connection)
            .await?;
        for policy_id in policy_ids {
            let Some(policy) = self.load_policy(&mut connection, &policy_id).await? else {
                continue;
            };
            if !policy_applies(&policy, workspace) || policy.budget_action != "REJECT" {
                continue;
            }
            let stored: HashMap<String, String> = redis::cmd("HGETALL")
                .arg(window_key(&policy_id))
                .query_async(&mut connection)
                .await?;
            if let Some(window) = build_stored_window(policy, &stored) {
                if now < window.window_end && window.cumulative_spend >= window.policy.budget_amount
                {
                    return Ok(Some(window));
                }
            }
        }
        Ok(None)
    }

    async fn backfill_spend(
        &self,
        spend_by_policy: &HashMap<String, f64>,
    ) -> Result<(), BudgetError> {
        let mut connection = self.connection().await?;
        let mut pipe = redis::pipe();
        for (policy_id, spend) in spend_by_policy {
            let exists: bool = redis::cmd("EXISTS")
                .arg(window_key(policy_id))
                .query_async(&mut connection)
                .await?;
            if !exists {
                continue;
            }
            let Some(policy) = self.load_policy(&mut connection, policy_id).await? else {
                continue;
            };
            pipe.hset(window_key(policy_id), "cumulative_spend", spend.to_string())
                .ignore()
                .hset(
                    window_key(policy_id),
                    "exceeded",
                    if *spend >= policy.budget_amount {
                        "1"
                    } else {
                        "0"
                    },
                )
                .ignore();
        }
        pipe.query_async::<()>(&mut connection).await?;
        Ok(())
    }

    #[cfg(test)]
    async fn get_window_info(
        &self,
        connection: &mut MultiplexedConnection,
        policy_id: &str,
    ) -> Result<Option<PolicyWindow>, BudgetError> {
        let stored: HashMap<String, String> = redis::cmd("HGETALL")
            .arg(window_key(policy_id))
            .query_async(connection)
            .await?;
        let Some(policy) = self.load_policy(connection, policy_id).await? else {
            return Ok(None);
        };
        Ok(build_stored_window(policy, &stored))
    }

    #[cfg(test)]
    async fn get_all_windows(&self) -> Result<Vec<PolicyWindow>, BudgetError> {
        let mut connection = self.connection().await?;
        let policy_ids: Vec<String> = redis::cmd("SMEMBERS")
            .arg(policy_set_key())
            .query_async(&mut connection)
            .await?;
        let mut windows = Vec::new();
        for policy_id in policy_ids {
            if let Some(window) = self.get_window_info(&mut connection, &policy_id).await? {
                windows.push(window);
            }
        }
        Ok(windows)
    }
}

/// Refresh policies and backfill each returned window from trace history.
/// Refresh failures are deliberately swallowed by callers, matching Python's
/// fail-open `maybe_refresh_budget_policies` wrapper.
pub async fn refresh_from_store(
    tracker: &BudgetTracker,
    store: &TrackingStore,
    workspace: &str,
    now: DateTime<Utc>,
) -> Result<(), BudgetError> {
    if !tracker.needs_refresh().await {
        return Ok(());
    }
    let policies = store
        .list_budget_policies(workspace, SEARCH_MAX_RESULTS_DEFAULT, None)
        .await
        .map_err(|error| BudgetError(error.to_string()))?
        .policies;
    let windows = tracker.refresh_policies(policies, now).await?;
    let mut spend = HashMap::new();
    for window in windows {
        let spend_workspace =
            (window.policy.target_scope == "WORKSPACE").then_some(window.policy.workspace.as_str());
        if let Ok(value) = store
            .sum_gateway_trace_cost(
                window.window_start.timestamp_millis(),
                window.window_end.timestamp_millis(),
                spend_workspace,
            )
            .await
        {
            if value > 0.0 {
                spend.insert(window.policy.budget_policy_id.clone(), value);
            }
        }
    }
    tracker.backfill_spend(&spend).await
}

pub fn reject_message(window: &PolicyWindow) -> String {
    let mut unit = window.policy.duration_unit.to_ascii_lowercase();
    if window.policy.duration_value == 1 {
        unit = unit.trim_end_matches('s').to_string();
    }
    format!(
        "Budget limit exceeded. Limit: ${:.2} USD per {} {}. Budget resets at {}. Request rejected.",
        window.policy.budget_amount,
        window.policy.duration_value,
        unit,
        window.window_end.format("%Y-%m-%dT%H:%M:%SZ")
    )
}

pub fn exceeded_payload(window: &PolicyWindow, workspace: Option<&str>) -> Value {
    json!({
        "budget_policy_id": window.policy.budget_policy_id,
        "budget_unit": window.policy.budget_unit,
        "budget_amount": window.policy.budget_amount,
        "current_spend": window.cumulative_spend,
        "duration_unit": window.policy.duration_unit,
        "duration_value": window.policy.duration_value,
        "target_scope": window.policy.target_scope,
        "workspace": workspace.unwrap_or(&window.policy.workspace),
        "window_start": window.window_start.timestamp_millis(),
    })
}

fn refresh_interval() -> Duration {
    Duration::from_secs(
        std::env::var(REFRESH_INTERVAL_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_REFRESH_INTERVAL_SECONDS),
    )
}

fn policy_applies(policy: &BudgetPolicy, workspace: Option<&str>) -> bool {
    policy.target_scope == "GLOBAL" || policy.workspace == workspace.unwrap_or(DEFAULT_WORKSPACE)
}

pub(crate) fn window_bounds(
    policy: &BudgetPolicy,
    now: DateTime<Utc>,
) -> Result<(DateTime<Utc>, DateTime<Utc>), BudgetError> {
    let value = i64::from(policy.duration_value);
    if value <= 0 {
        return Err(BudgetError(format!(
            "duration.value must be positive, got {value}"
        )));
    }
    let epoch = Utc.timestamp_opt(0, 0).single().expect("Unix epoch");
    let start = match policy.duration_unit.as_str() {
        "MINUTES" => {
            let elapsed = now.timestamp().div_euclid(60);
            epoch + chrono::Duration::minutes(elapsed.div_euclid(value) * value)
        }
        "HOURS" => {
            let elapsed = now.timestamp().div_euclid(3600);
            epoch + chrono::Duration::hours(elapsed.div_euclid(value) * value)
        }
        "DAYS" => {
            let elapsed = (now - epoch).num_days();
            epoch + chrono::Duration::days(elapsed.div_euclid(value) * value)
        }
        "WEEKS" => {
            let sunday = epoch - chrono::Duration::days(4);
            let elapsed = (now - sunday).num_days();
            sunday + chrono::Duration::days(elapsed.div_euclid(7 * value) * 7 * value)
        }
        "MONTHS" => {
            let total = i64::from(now.year() - 1970) * 12 + i64::from(now.month0());
            let month = total.div_euclid(value) * value;
            Utc.with_ymd_and_hms(
                1970 + i32::try_from(month.div_euclid(12)).unwrap_or_default(),
                u32::try_from(month.rem_euclid(12) + 1).unwrap_or(1),
                1,
                0,
                0,
                0,
            )
            .single()
            .ok_or_else(|| BudgetError("Invalid monthly budget window".to_string()))?
        }
        unit => return Err(BudgetError(format!("Unknown duration type: {unit}"))),
    };
    let end = match policy.duration_unit.as_str() {
        "MINUTES" => start + chrono::Duration::minutes(value),
        "HOURS" => start + chrono::Duration::hours(value),
        "DAYS" => start + chrono::Duration::days(value),
        "WEEKS" => start + chrono::Duration::weeks(value),
        "MONTHS" => {
            let total = i64::from(start.year()) * 12 + i64::from(start.month0()) + value;
            Utc.with_ymd_and_hms(
                i32::try_from(total.div_euclid(12)).unwrap_or_default(),
                u32::try_from(total.rem_euclid(12) + 1).unwrap_or(1),
                1,
                0,
                0,
                0,
            )
            .single()
            .ok_or_else(|| BudgetError("Invalid monthly budget window".to_string()))?
        }
        _ => unreachable!(),
    };
    Ok((start, end))
}

fn window_key(policy_id: &str) -> String {
    format!("{KEY_PREFIX}window:{policy_id}")
}

fn policy_key(policy_id: &str) -> String {
    format!("{KEY_PREFIX}policy:{policy_id}")
}

fn policy_set_key() -> &'static str {
    "mlflow:budget:policies"
}

fn serialize_policy(policy: &BudgetPolicy) -> String {
    json!({
        "budget_policy_id": policy.budget_policy_id,
        "budget_unit": policy.budget_unit,
        "budget_amount": policy.budget_amount,
        "duration_unit": policy.duration_unit,
        "duration_value": policy.duration_value,
        "target_scope": policy.target_scope,
        "budget_action": policy.budget_action,
        "workspace": policy.workspace,
        "created_at": policy.created_at,
        "last_updated_at": policy.last_updated_at,
    })
    .to_string()
}

fn deserialize_policy(data: &str) -> Result<BudgetPolicy, BudgetError> {
    let value: Value =
        serde_json::from_str(data).map_err(|error| BudgetError(error.to_string()))?;
    let string = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    Ok(BudgetPolicy {
        budget_policy_id: string("budget_policy_id"),
        budget_unit: string("budget_unit"),
        budget_amount: value
            .get("budget_amount")
            .and_then(Value::as_f64)
            .unwrap_or_default(),
        duration_unit: string("duration_unit"),
        duration_value: value
            .get("duration_value")
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or_default(),
        target_scope: string("target_scope"),
        budget_action: string("budget_action"),
        created_by: None,
        created_at: value
            .get("created_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        last_updated_by: None,
        last_updated_at: value
            .get("last_updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        workspace: string("workspace"),
    })
}

fn build_stored_window(
    policy: BudgetPolicy,
    stored: &HashMap<String, String>,
) -> Option<PolicyWindow> {
    if stored.is_empty() {
        return None;
    }
    Some(PolicyWindow {
        policy,
        window_start: DateTime::parse_from_rfc3339(stored.get("window_start")?)
            .ok()?
            .with_timezone(&Utc),
        window_end: DateTime::parse_from_rfc3339(stored.get("window_end")?)
            .ok()?
            .with_timezone(&Utc),
        cumulative_spend: stored
            .get("cumulative_spend")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.0),
        exceeded: stored.get("exceeded").is_some_and(|value| value == "1"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command, Stdio};

    fn policy(id: &str, amount: f64, action: &str) -> BudgetPolicy {
        BudgetPolicy {
            budget_policy_id: id.to_string(),
            budget_unit: "USD".to_string(),
            budget_amount: amount,
            duration_unit: "DAYS".to_string(),
            duration_value: 1,
            target_scope: "GLOBAL".to_string(),
            budget_action: action.to_string(),
            created_by: None,
            created_at: 0,
            last_updated_by: None,
            last_updated_at: 0,
            workspace: DEFAULT_WORKSPACE.to_string(),
        }
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 6, 15, 10, 37, 0)
            .single()
            .unwrap()
    }

    #[tokio::test]
    async fn in_memory_boundary_alert_and_reject_semantics() {
        let tracker = BudgetTracker::InMemory(Arc::new(InMemoryBudgetTracker::default()));
        tracker
            .refresh_policies(
                vec![
                    policy("alert", 100.0, "ALERT"),
                    policy("reject", 100.0, "REJECT"),
                ],
                now(),
            )
            .await
            .unwrap();
        let crossed = tracker.record_cost(100.0, None, now()).await.unwrap();
        assert_eq!(crossed.len(), 2);
        assert_eq!(
            tracker
                .should_reject_request(None, now())
                .await
                .unwrap()
                .unwrap()
                .policy
                .budget_policy_id,
            "reject"
        );
        assert!(tracker
            .record_cost(1.0, None, now())
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn in_memory_refresh_backfill_and_window_reset() {
        let tracker = BudgetTracker::InMemory(Arc::new(InMemoryBudgetTracker::default()));
        tracker
            .refresh_policies(vec![policy("one", 50.0, "ALERT")], now())
            .await
            .unwrap();
        tracker.record_cost(40.0, None, now()).await.unwrap();
        tracker
            .backfill_spend(&HashMap::from([("one".to_string(), 10.0)]))
            .await
            .unwrap();
        assert_eq!(
            tracker.get_all_windows().await.unwrap()[0].cumulative_spend,
            40.0
        );

        let next_day = now() + chrono::Duration::days(1);
        tracker.record_cost(5.0, None, next_day).await.unwrap();
        let window = tracker.get_all_windows().await.unwrap().remove(0);
        assert_eq!(window.cumulative_spend, 5.0);
        assert!(!window.exceeded);
    }

    #[test]
    fn window_alignment_and_message_match_python() {
        let mut p = policy("one", 500.0, "REJECT");
        p.duration_unit = "MONTHS".to_string();
        p.duration_value = 3;
        let (start, end) = window_bounds(&p, now()).unwrap();
        assert_eq!(start, Utc.with_ymd_and_hms(2025, 4, 1, 0, 0, 0).unwrap());
        assert_eq!(end, Utc.with_ymd_and_hms(2025, 7, 1, 0, 0, 0).unwrap());
        let window = PolicyWindow {
            policy: p,
            window_start: start,
            window_end: end,
            cumulative_spend: 600.0,
            exceeded: true,
        };
        assert_eq!(
            reject_message(&window),
            "Budget limit exceeded. Limit: $500.00 USD per 3 months. Budget resets at 2025-07-01T00:00:00Z. Request rejected."
        );
    }

    #[test]
    fn alert_payload_field_order_and_values_match_python() {
        let p = policy("bp-test", 50.0, "ALERT");
        let (start, end) = window_bounds(&p, now()).unwrap();
        let value = exceeded_payload(
            &PolicyWindow {
                policy: p,
                window_start: start,
                window_end: end,
                cumulative_spend: 60.0,
                exceeded: true,
            },
            None,
        );
        assert_eq!(
            value.to_string(),
            format!(
                "{{\"budget_policy_id\":\"bp-test\",\"budget_unit\":\"USD\",\"budget_amount\":50.0,\"current_spend\":60.0,\"duration_unit\":\"DAYS\",\"duration_value\":1,\"target_scope\":\"GLOBAL\",\"workspace\":\"default\",\"window_start\":{}}}",
                start.timestamp_millis()
            )
        );
    }

    struct RedisServerGuard {
        child: Child,
        _directory: tempfile::TempDir,
        url: String,
    }

    impl RedisServerGuard {
        async fn start_if_available() -> Option<Self> {
            if Command::new("redis-server")
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_err()
            {
                return None;
            }
            let listener = std::net::TcpListener::bind("127.0.0.1:0").ok()?;
            let port = listener.local_addr().ok()?.port();
            drop(listener);
            let directory = tempfile::tempdir().ok()?;
            let child = Command::new("redis-server")
                .args([
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    &port.to_string(),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                    "--dir",
                    directory.path().to_str()?,
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let url = format!("redis://127.0.0.1:{port}/0");
            for _ in 0..100 {
                if tokio::net::TcpStream::connect(("127.0.0.1", port))
                    .await
                    .is_ok()
                {
                    return Some(Self {
                        child,
                        _directory: directory,
                        url,
                    });
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            None
        }
    }

    impl Drop for RedisServerGuard {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    #[tokio::test]
    async fn redis_backend_boundary_rollover_and_shared_state() {
        let Some(server) = RedisServerGuard::start_if_available().await else {
            eprintln!("redis-server unavailable; skipping service-dependent Redis tracker test");
            return;
        };
        let first = BudgetTracker::Redis(Arc::new(RedisBudgetTracker::new(server.url.clone())));
        let second = BudgetTracker::Redis(Arc::new(RedisBudgetTracker::new(server.url.clone())));
        first
            .refresh_policies(vec![policy("redis", 100.0, "REJECT")], now())
            .await
            .unwrap();
        assert_eq!(
            first.record_cost(100.0, None, now()).await.unwrap().len(),
            1
        );
        assert!(second
            .should_reject_request(None, now())
            .await
            .unwrap()
            .is_some());
        first
            .record_cost(5.0, None, now() + chrono::Duration::days(1))
            .await
            .unwrap();
        let window = second.get_all_windows().await.unwrap().remove(0);
        assert_eq!(window.cumulative_spend, 5.0);
        assert!(!window.exceeded);
    }

    #[tokio::test]
    async fn seeded_fixture_matches_python_reject_alert_and_reset_oracle() {
        let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("repository root");
        let output = Command::new("uv")
            .args(["run", "--frozen", "python", "rust/tools/budget_oracle.py"])
            .current_dir(repository)
            .output()
            .expect("run Python budget oracle");
        assert!(
            output.status.success(),
            "Python oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let python: Value = serde_json::from_slice(&output.stdout).unwrap();

        let mut rust = serde_json::Map::new();
        for (name, spend) in [("under", 99.0), ("boundary", 100.0), ("over", 101.0)] {
            let tracker = BudgetTracker::InMemory(Arc::new(InMemoryBudgetTracker::default()));
            tracker
                .refresh_policies(vec![policy("bp-reject", 100.0, "REJECT")], now())
                .await
                .unwrap();
            tracker.record_cost(spend, None, now()).await.unwrap();
            let rejected = tracker.should_reject_request(None, now()).await.unwrap();
            rust.insert(
                name.to_string(),
                match rejected {
                    Some(window) => json!({
                        "reject": true,
                        "detail": reject_message(&window),
                        "spend": window.cumulative_spend,
                    }),
                    None => json!({"reject": false, "detail": Value::Null}),
                },
            );
        }

        let alert_tracker = BudgetTracker::InMemory(Arc::new(InMemoryBudgetTracker::default()));
        alert_tracker
            .refresh_policies(vec![policy("bp-alert", 50.0, "ALERT")], now())
            .await
            .unwrap();
        alert_tracker.record_cost(49.0, None, now()).await.unwrap();
        let crossed = alert_tracker.record_cost(1.0, None, now()).await.unwrap();
        rust.insert("alert".to_string(), exceeded_payload(&crossed[0], None));

        let reset_tracker = BudgetTracker::InMemory(Arc::new(InMemoryBudgetTracker::default()));
        reset_tracker
            .refresh_policies(vec![policy("bp-reset", 100.0, "ALERT")], now())
            .await
            .unwrap();
        reset_tracker.record_cost(150.0, None, now()).await.unwrap();
        reset_tracker
            .record_cost(10.0, None, now() + chrono::Duration::days(1))
            .await
            .unwrap();
        let reset = reset_tracker.get_all_windows().await.unwrap().remove(0);
        rust.insert(
            "reset".to_string(),
            json!({"spend": reset.cumulative_spend, "exceeded": reset.exceeded}),
        );

        assert_eq!(Value::Object(rust), python);
    }
}
