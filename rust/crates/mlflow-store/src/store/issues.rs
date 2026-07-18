//! Issue CRUD and search, including Python-compatible assessment trace counts.

use mlflow_error::MlflowError;
use mlflow_search::Value as SearchValue;
use serde_json::Value;
use uuid::Uuid;

use crate::dialect::Dialect;

use super::dbutil::{RowLike, Val};
use super::evaluation_datasets::python_json_dumps;
use super::experiments::{internal, now_millis, parse_experiment_id};
use super::search::SEARCH_MAX_RESULTS_THRESHOLD;
use super::TrackingStore;

const ISSUE_ID_PREFIX: &str = "iss-";
const DEFAULT_MAX_RESULTS: i32 = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    pub issue_id: String,
    pub experiment_id: String,
    pub name: String,
    pub description: String,
    pub status: String,
    pub severity: Option<String>,
    pub root_causes: Vec<String>,
    pub source_run_id: Option<String>,
    pub created_timestamp: i64,
    pub last_updated_timestamp: i64,
    pub created_by: Option<String>,
    pub categories: Vec<String>,
    pub trace_count: Option<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct IssueUpdate<'a> {
    pub name: Option<&'a str>,
    pub description: Option<&'a str>,
    pub status: Option<&'a str>,
    pub severity: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssuesPage {
    pub issues: Vec<Issue>,
    pub next_page_token: Option<String>,
}

impl TrackingStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn create_issue(
        &self,
        workspace: &str,
        experiment_id: &str,
        name: &str,
        description: &str,
        status: &str,
        severity: Option<&str>,
        root_causes: &[String],
        source_run_id: Option<&str>,
        categories: &[String],
        created_by: Option<&str>,
    ) -> Result<Issue, MlflowError> {
        let experiment_id = parse_experiment_id(experiment_id)?;
        self.require_active_experiment_row(workspace, experiment_id)
            .await?;
        let issue_id = format!("{ISSUE_ID_PREFIX}{}", Uuid::new_v4().simple());
        let now = now_millis();
        let root_causes_json = (!root_causes.is_empty()).then(|| {
            python_json_dumps(
                &Value::Array(root_causes.iter().cloned().map(Value::String).collect()),
                false,
            )
        });
        let categories_json = (!categories.is_empty()).then(|| {
            python_json_dumps(
                &Value::Array(categories.iter().cloned().map(Value::String).collect()),
                false,
            )
        });
        let dialect = self.db().dialect();
        let placeholders = (1..=12)
            .map(|index| dialect.placeholder(index))
            .collect::<Vec<_>>();
        self.db()
            .exec(
                &format!(
                    "INSERT INTO issues (issue_id, experiment_id, name, description, status, \
                     severity, root_causes, source_run_id, categories, created_timestamp, \
                     last_updated_timestamp, created_by) VALUES ({})",
                    placeholders.join(", ")
                ),
                &[
                    Val::Text(issue_id.clone()),
                    Val::Int(experiment_id),
                    Val::Text(name.to_string()),
                    Val::Text(description.to_string()),
                    Val::Text(status.to_string()),
                    Val::OptText(severity.map(str::to_string)),
                    Val::OptText(root_causes_json),
                    Val::OptText(source_run_id.map(str::to_string)),
                    Val::OptText(categories_json),
                    Val::Int(now),
                    Val::Int(now),
                    Val::OptText(created_by.map(str::to_string)),
                ],
            )
            .await
            .map_err(internal)?;
        self.get_issue(workspace, &issue_id).await
    }

    pub async fn get_issue(&self, workspace: &str, issue_id: &str) -> Result<Issue, MlflowError> {
        let dialect = self.db().dialect();
        self.db()
            .fetch_optional(
                &format!(
                    "SELECT i.issue_id, i.experiment_id, i.name, i.description, i.status, \
                     i.severity, i.root_causes, i.source_run_id, i.categories, \
                     i.created_timestamp, i.last_updated_timestamp, i.created_by \
                     FROM issues i JOIN experiments e ON e.experiment_id = i.experiment_id \
                     WHERE i.issue_id = {} AND e.workspace = {}",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[
                    Val::Text(issue_id.to_string()),
                    Val::Text(workspace.to_string()),
                ],
                |row| map_issue(row, false),
            )
            .await
            .map_err(internal)?
            .ok_or_else(|| {
                MlflowError::resource_does_not_exist(format!(
                    "Issue with ID '{issue_id}' not found"
                ))
            })
    }

    pub async fn update_issue(
        &self,
        workspace: &str,
        issue_id: &str,
        update: IssueUpdate<'_>,
    ) -> Result<Issue, MlflowError> {
        self.get_issue(workspace, issue_id).await?;
        let dialect = self.db().dialect();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        for (column, value) in [
            ("name", update.name),
            ("description", update.description),
            ("status", update.status),
            ("severity", update.severity),
        ] {
            if let Some(value) = value {
                values.push(Val::Text(value.to_string()));
                assignments.push(format!("{column} = {}", dialect.placeholder(values.len())));
            }
        }
        values.push(Val::Int(now_millis()));
        assignments.push(format!(
            "last_updated_timestamp = {}",
            dialect.placeholder(values.len())
        ));
        values.push(Val::Text(issue_id.to_string()));
        let issue_placeholder = dialect.placeholder(values.len());
        values.push(Val::Text(workspace.to_string()));
        let workspace_placeholder = dialect.placeholder(values.len());
        self.db()
            .exec(
                &format!(
                    "UPDATE issues SET {} WHERE issue_id = {issue_placeholder} AND \
                     experiment_id IN (SELECT experiment_id FROM experiments WHERE workspace = \
                     {workspace_placeholder})",
                    assignments.join(", ")
                ),
                &values,
            )
            .await
            .map_err(internal)?;
        self.get_issue(workspace, issue_id).await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn search_issues(
        &self,
        workspace: &str,
        experiment_id: Option<&str>,
        filter_string: Option<&str>,
        max_results: Option<i32>,
        page_token: Option<&str>,
        include_trace_count: bool,
    ) -> Result<IssuesPage, MlflowError> {
        let max_results = max_results.unwrap_or(DEFAULT_MAX_RESULTS);
        if max_results < 1 {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {max_results} for parameter 'max_results' supplied. It must be a \
                 positive integer"
            )));
        }
        if i64::from(max_results) > SEARCH_MAX_RESULTS_THRESHOLD {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Invalid value {max_results} for parameter 'max_results' supplied. It must be at \
                 most {SEARCH_MAX_RESULTS_THRESHOLD}"
            )));
        }
        let offset = mlflow_search::parse_start_offset_from_page_token(page_token)
            .map_err(|error| MlflowError::invalid_parameter_value(error.message))?;
        let filters = filter_string
            .filter(|value| !value.is_empty())
            .map(mlflow_search::parse::issues_filter)
            .transpose()
            .map_err(|error| MlflowError::invalid_parameter_value(error.message))?
            .unwrap_or_default();

        let dialect = self.db().dialect();
        let mut values = vec![Val::Text(workspace.to_string())];
        let mut select = String::from(
            "SELECT i.issue_id, i.experiment_id, i.name, i.description, i.status, i.severity, \
             i.root_causes, i.source_run_id, i.categories, i.created_timestamp, \
             i.last_updated_timestamp, i.created_by",
        );
        if include_trace_count {
            let trace_count = if dialect == Dialect::MySql {
                "CAST(COUNT(DISTINCT a.trace_id) AS SIGNED)"
            } else {
                "COUNT(DISTINCT a.trace_id)"
            };
            select.push_str(&format!(", {trace_count} AS trace_count"));
        }
        select.push_str(" FROM issues i JOIN experiments e ON e.experiment_id = i.experiment_id");
        if include_trace_count {
            // Python intentionally does not filter `valid` assessments here.
            select.push_str(
                " LEFT OUTER JOIN assessments a ON a.name = i.issue_id AND \
                 a.assessment_type = 'issue'",
            );
        }
        select.push_str(&format!(" WHERE e.workspace = {}", dialect.placeholder(1)));
        if let Some(experiment_id) = experiment_id.filter(|value| !value.is_empty()) {
            values.push(Val::Int(parse_experiment_id(experiment_id)?));
            select.push_str(&format!(
                " AND i.experiment_id = {}",
                dialect.placeholder(values.len())
            ));
        }
        for filter in filters {
            let SearchValue::Str(value) = filter.value else {
                unreachable!("issue filters only contain strings")
            };
            values.push(Val::Text(value));
            let column = match filter.key.as_str() {
                "status" => "i.status",
                "source_run_id" => "i.source_run_id",
                _ => unreachable!("parser validates issue filter keys"),
            };
            select.push_str(&format!(
                " AND {column} {} {}",
                filter.comparator,
                dialect.placeholder(values.len())
            ));
        }
        if include_trace_count {
            select.push_str(
                " GROUP BY i.issue_id, i.experiment_id, i.name, i.description, i.status, \
                 i.severity, i.root_causes, i.source_run_id, i.categories, \
                 i.created_timestamp, i.last_updated_timestamp, i.created_by",
            );
        }
        select.push_str(
            " ORDER BY CASE i.severity WHEN 'not_an_issue' THEN 0 WHEN 'low' THEN 1 \
             WHEN 'medium' THEN 2 WHEN 'high' THEN 3 ELSE -1 END DESC",
        );
        if include_trace_count {
            select.push_str(", trace_count DESC");
        }
        select.push_str(&format!(
            ", i.created_timestamp DESC, i.issue_id DESC LIMIT {} OFFSET {}",
            i64::from(max_results) + 1,
            offset
        ));

        let mut issues = self
            .db()
            .fetch_all(&select, &values, |row| map_issue(row, include_trace_count))
            .await
            .map_err(internal)?;
        let next_page_token = if issues.len() > max_results as usize {
            issues.truncate(max_results as usize);
            Some(mlflow_search::create_page_token(
                offset + i64::from(max_results),
            ))
        } else {
            None
        };
        Ok(IssuesPage {
            issues,
            next_page_token,
        })
    }

    pub(crate) async fn require_active_experiment_row(
        &self,
        workspace: &str,
        experiment_id: i64,
    ) -> Result<(), MlflowError> {
        let dialect = self.db().dialect();
        let found = self
            .db()
            .fetch_optional(
                &format!(
                    "SELECT experiment_id FROM experiments WHERE experiment_id = {} AND \
                     workspace = {} AND lifecycle_stage = 'active'",
                    dialect.placeholder(1),
                    dialect.placeholder(2)
                ),
                &[Val::Int(experiment_id), Val::Text(workspace.to_string())],
                |row| row.get_int("experiment_id"),
            )
            .await
            .map_err(internal)?;
        if found.is_none() {
            return Err(MlflowError::resource_does_not_exist(format!(
                "No Experiment with id={experiment_id} exists"
            )));
        }
        Ok(())
    }
}

fn map_issue(row: &dyn RowLike, include_trace_count: bool) -> Result<Issue, sqlx::Error> {
    Ok(Issue {
        issue_id: row.get_string("issue_id")?,
        experiment_id: row.get_int("experiment_id")?.to_string(),
        name: row.get_string("name")?,
        description: row.get_string("description")?,
        status: row.get_string("status")?,
        severity: row.get_opt_string("severity")?,
        root_causes: parse_string_list(row.get_opt_string("root_causes")?)?,
        source_run_id: row.get_opt_string("source_run_id")?,
        categories: parse_string_list(row.get_opt_string("categories")?)?,
        created_timestamp: row.get_i64("created_timestamp")?,
        last_updated_timestamp: row.get_i64("last_updated_timestamp")?,
        created_by: row.get_opt_string("created_by")?,
        trace_count: if include_trace_count {
            Some(i32::try_from(row.get_i64("trace_count")?).unwrap_or(i32::MAX))
        } else {
            None
        },
    })
}

fn parse_string_list(value: Option<String>) -> Result<Vec<String>, sqlx::Error> {
    match value {
        None => Ok(Vec::new()),
        Some(value) => {
            serde_json::from_str(&value).map_err(|error| sqlx::Error::Decode(Box::new(error)))
        }
    }
}
