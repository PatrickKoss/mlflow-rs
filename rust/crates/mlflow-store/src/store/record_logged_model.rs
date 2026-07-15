//! `record_logged_model` — the legacy `runs/log-model` write path, mirroring
//! `SqlAlchemyStore.record_logged_model` (`sqlalchemy_store.py:2117`).
//!
//! This is the pre-LoggedModels ("model history") API: it appends the model's
//! *tags dict* to the run's `mlflow.log-model.history` tag, whose value is a JSON
//! array of model dicts.
//!
//! ## Byte-for-byte tag value (why key order matters)
//!
//! Python builds the appended dict via `Model.from_dict(model).get_tags_dict()`
//! (`mlflow/models/model.py:709`), which keeps only `run_id`, `artifact_path`,
//! `utc_time_created`, `model_uuid` (in that `Model.__dict__` order) and then
//! appends a rebuilt `flavors` map whose per-flavor config drops the nested
//! `config` key. It re-serializes with `json.dumps(...)` (default separators
//! `", "` / `": "`, `ensure_ascii=True`). We reproduce the key order using
//! `serde_json`'s `preserve_order` feature and emit compact JSON with the same
//! separators; `to_string()` on a `serde_json::Value` uses `","`/`":"` (no
//! spaces), so we serialize with an explicit `PrettyFormatter`-free compact
//! writer that inserts Python's spacing. See [`dumps_python`].
//!
//! The `model_uuid` key is always present in the appended dict (Python's
//! `from_dict` inserts `model_uuid = None` when absent), so we emit it as JSON
//! `null` when the input omits it.

use mlflow_error::MlflowError;
use serde_json::{Map, Value};

use super::dbutil::{Tx, Val};
use super::experiments::internal;
use super::params_tags::upsert_tag;
use super::runs::check_run_active;
use super::validation;
use super::TrackingStore;

/// The legacy model-history tag key (`MLFLOW_LOGGED_MODELS`,
/// `mlflow/utils/mlflow_tags.py:23`).
pub(crate) const MLFLOW_LOGGED_MODELS: &str = "mlflow.log-model.history";

impl TrackingStore {
    /// `record_logged_model`: append `model_json`'s tags dict to the run's
    /// `mlflow.log-model.history` tag. Workspace-scoped; requires ACTIVE run.
    ///
    /// `model_json` is the already-parsed MLmodel dict (the handler validates it
    /// is valid JSON and carries the mandatory fields before calling here).
    pub async fn record_logged_model(
        &self,
        workspace: &str,
        run_id: &str,
        model_json: &Value,
    ) -> Result<(), MlflowError> {
        let row = self.resolve_run_row(workspace, run_id).await?;
        check_run_active(&row)?;

        let model_dict = get_tags_dict(model_json);

        let dialect = self.db().dialect();
        let mut tx = self.db().begin_tx().await.map_err(internal)?;

        let existing = self
            .existing_tag_value(&mut tx, run_id, MLFLOW_LOGGED_MODELS)
            .await?;
        let mut history: Vec<Value> = match existing.as_deref() {
            Some(v) => match serde_json::from_str::<Vec<Value>>(v) {
                Ok(arr) => arr,
                // Python would raise on malformed JSON here; a non-array existing
                // value is not something Python writes, so start fresh only for a
                // genuinely absent tag (handled above). Surface a parse failure as
                // an internal error rather than silently dropping history.
                Err(e) => {
                    return Err(MlflowError::new(
                        format!("Malformed '{MLFLOW_LOGGED_MODELS}' tag value: {e}"),
                        mlflow_error::ErrorCode::InternalError,
                    ));
                }
            },
            None => Vec::new(),
        };
        history.push(model_dict);

        let value = dumps_python(&Value::Array(history));
        validation::validate_tag(MLFLOW_LOGGED_MODELS, &value, None)?;
        upsert_tag(&mut tx, dialect, run_id, MLFLOW_LOGGED_MODELS, &value).await?;

        tx.commit().await.map_err(internal)
    }

    /// Read a single tag value for a run inside a transaction (or `None`).
    async fn existing_tag_value(
        &self,
        tx: &mut Tx<'_>,
        run_id: &str,
        key: &str,
    ) -> Result<Option<String>, MlflowError> {
        let dialect = self.db().dialect();
        let sql = format!(
            "SELECT value FROM tags WHERE run_uuid = {} AND \"key\" = {}",
            dialect.placeholder(1),
            dialect.placeholder(2)
        );
        let rows = tx
            .fetch_all(
                &sql,
                &[Val::Text(run_id.to_string()), Val::Text(key.to_string())],
                |r| r.get_string("value"),
            )
            .await
            .map_err(internal)?;
        Ok(rows.into_iter().next())
    }
}

/// `Model.get_tags_dict` (`mlflow/models/model.py:709`): keep `run_id`,
/// `artifact_path`, `utc_time_created`, `model_uuid` (in that order — Python's
/// `Model.__dict__` order via `to_dict`), then append a rebuilt `flavors` map
/// whose per-flavor config drops the nested `config` key.
fn get_tags_dict(model: &Value) -> Value {
    let mut out = Map::new();
    for key in ["run_id", "artifact_path", "utc_time_created", "model_uuid"] {
        // `model_uuid` is always inserted by `Model.from_dict` (as `None` when
        // absent); the others are guaranteed present by the handler's
        // mandatory-field check.
        let value = model.get(key).cloned().unwrap_or(Value::Null);
        out.insert(key.to_string(), value);
    }

    let flavors = match model.get("flavors") {
        Some(Value::Object(flavors)) => {
            let mut rebuilt = Map::new();
            for (flavor, config) in flavors {
                let cleaned = match config {
                    Value::Object(cfg) => {
                        let mut m = Map::new();
                        for (k, v) in cfg {
                            if k != "config" {
                                m.insert(k.clone(), v.clone());
                            }
                        }
                        Value::Object(m)
                    }
                    other => other.clone(),
                };
                rebuilt.insert(flavor.clone(), cleaned);
            }
            Value::Object(rebuilt)
        }
        _ => Value::Object(Map::new()),
    };
    out.insert("flavors".to_string(), flavors);

    Value::Object(out)
}

/// Serialize a `serde_json::Value` the way Python's `json.dumps` does by
/// default: `", "` between items and `": "` between key and value,
/// `ensure_ascii=True` (non-ASCII escaped). `serde_json`'s own `to_string` uses
/// no spaces, so we render with a custom compact-but-spaced formatter.
fn dumps_python(value: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, value);
    out
}

fn write_json(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_json_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_string(out, k);
                out.push_str(": ");
                write_json(out, v);
            }
            out.push('}');
        }
    }
}

/// Write a JSON string with `ensure_ascii=True` escaping (matching Python's
/// `json.dumps`): control chars and non-ASCII become `\uXXXX` (astral chars as
/// surrogate pairs), with the standard short escapes for `"`, `\`, and the C0
/// whitespace set.
fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_ascii() => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn get_tags_dict_orders_keys_and_strips_flavor_config() {
        let model = json!({
            "artifact_path": "model",
            "run_id": "abc",
            "utc_time_created": "2020-01-01",
            "model_uuid": "uuid1",
            "flavors": {
                "python_function": {"loader_module": "m", "config": {"drop": true}},
                "sklearn": {"pickled_model": "model.pkl"}
            },
            "extra": "dropped"
        });
        let dict = get_tags_dict(&model);
        let s = dumps_python(&dict);
        assert_eq!(
            s,
            "{\"run_id\": \"abc\", \"artifact_path\": \"model\", \
             \"utc_time_created\": \"2020-01-01\", \"model_uuid\": \"uuid1\", \
             \"flavors\": {\"python_function\": {\"loader_module\": \"m\"}, \
             \"sklearn\": {\"pickled_model\": \"model.pkl\"}}}"
        );
    }

    #[test]
    fn missing_model_uuid_becomes_null() {
        let model = json!({
            "artifact_path": "model",
            "run_id": "abc",
            "utc_time_created": "2020-01-01",
            "flavors": {}
        });
        let dict = get_tags_dict(&model);
        assert_eq!(dict.get("model_uuid"), Some(&Value::Null));
    }

    #[test]
    fn non_ascii_is_escaped() {
        let v = Value::String("café🚀".to_string());
        assert_eq!(dumps_python(&v), "\"caf\\u00e9\\ud83d\\ude80\"");
    }
}
