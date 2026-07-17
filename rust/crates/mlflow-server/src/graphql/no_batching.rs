//! Query-safety / no-batching guard — a faithful port of
//! `mlflow/server/graphql/graphql_no_batching.py`.
//!
//! Python's `scan_query` walks every operation's selection set breadth-wise,
//! enforcing (in this order, per level):
//!
//! * **max depth** `_MAX_DEPTH = 10` — raised the moment a selection set at
//!   depth > 10 is popped (`"Query exceeds maximum depth of 10"`);
//! * **max total selections** `_MAX_SELECTIONS = 1000` — counting every
//!   `FieldNode` visited across the whole document
//!   (`"Query exceeds maximum total selections of 1000"`);
//! * **root fields** — the number of depth-1 field selections, checked against
//!   `MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS` (default 10);
//! * **max aliases** — the maximum number of *aliased* field selections within a
//!   single selection set, checked against `MLFLOW_SERVER_GRAPHQL_MAX_ALIASES`
//!   (default 10).
//!
//! The depth/selections limits raise a `GraphQLError` mid-scan (returned as the
//! sole error, `data: null`); the root-fields/aliases limits are checked after
//! the full scan and produce the env-var-named message. All four messages are
//! reproduced verbatim.
//!
//! ## Scan-order fidelity
//!
//! Python pushes child selection sets onto a stack and pops them (LIFO), and the
//! depth/selections checks fire on whichever offending node is reached first
//! under that traversal. For a *conforming* query the traversal order doesn't
//! matter (no error), and for the tests that deliberately exceed a single limit
//! only that limit can fire — so matching Python's exact pop order is
//! unobservable. We use the same explicit-stack LIFO walk regardless, so the
//! counting is identical.

use graphql_parser::query::{Definition, Document, Selection, SelectionSet};

/// `MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS` default (`environment_variables.py`).
const MAX_ROOT_FIELDS: usize = 10;
/// `MLFLOW_SERVER_GRAPHQL_MAX_ALIASES` default.
const MAX_ALIASES: usize = 10;
/// `_MAX_DEPTH`.
const MAX_DEPTH: usize = 10;
/// `_MAX_SELECTIONS`.
const MAX_SELECTIONS: usize = 1000;

/// The outcome of [`check_query_safety`]: `Ok(())` when the query is safe, or
/// `Err(message)` carrying the exact Python error message to surface as the sole
/// GraphQL error.
pub type SafetyResult = Result<(), String>;

/// `check_query_safety(ast_node)` — returns the first violated limit's message,
/// or `Ok(())`.
pub fn check_query_safety(doc: &Document<'_, String>) -> SafetyResult {
    let info = scan_query(doc)?;

    if info.root_fields > MAX_ROOT_FIELDS {
        return Err(limit_message(
            "root fields",
            "MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS",
            MAX_ROOT_FIELDS,
            info.root_fields,
        ));
    }
    if info.max_aliases > MAX_ALIASES {
        return Err(limit_message(
            "aliases",
            "MLFLOW_SERVER_GRAPHQL_MAX_ALIASES",
            MAX_ALIASES,
            info.max_aliases,
        ));
    }
    Ok(())
}

struct QueryInfo {
    root_fields: usize,
    max_aliases: usize,
}

/// `scan_query(ast_node)`: count root fields + the max aliases-per-level, raising
/// the depth/total-selections errors mid-walk (`Err(message)`).
fn scan_query(doc: &Document<'_, String>) -> Result<QueryInfo, String> {
    let mut root_fields = 0usize;
    let mut max_aliases = 0usize;
    let mut total_selections = 0usize;

    for def in &doc.definitions {
        let Some(selection_set) = definition_selection_set(def) else {
            continue;
        };
        // LIFO stack of (selection_set, depth), depth starting at 1 (Python).
        let mut stack: Vec<(&SelectionSet<'_, String>, usize)> = vec![(selection_set, 1)];
        while let Some((set, depth)) = stack.pop() {
            if depth > MAX_DEPTH {
                return Err(format!("Query exceeds maximum depth of {MAX_DEPTH}"));
            }
            let mut current_aliases = 0usize;
            for selection in &set.items {
                // Python only counts `FieldNode`s (fragment spreads / inline
                // fragments are ignored by `scan_query`).
                let Selection::Field(field) = selection else {
                    continue;
                };
                if depth == 1 {
                    root_fields += 1;
                }
                if field.alias.is_some() {
                    current_aliases += 1;
                }
                if !field.selection_set.items.is_empty() {
                    stack.push((&field.selection_set, depth + 1));
                }
                total_selections += 1;
                if total_selections > MAX_SELECTIONS {
                    return Err(format!(
                        "Query exceeds maximum total selections of {MAX_SELECTIONS}"
                    ));
                }
            }
            max_aliases = max_aliases.max(current_aliases);
        }
    }

    Ok(QueryInfo {
        root_fields,
        max_aliases,
    })
}

/// The top-level selection set for a definition (operations only; fragment
/// *definitions* also carry one, matching Python's
/// `getattr(definition, "selection_set", None)`).
fn definition_selection_set<'r, 'a>(
    def: &'r Definition<'a, String>,
) -> Option<&'r SelectionSet<'a, String>> {
    use graphql_parser::query::OperationDefinition;
    match def {
        Definition::Operation(op) => Some(match op {
            OperationDefinition::SelectionSet(s) => s,
            OperationDefinition::Query(q) => &q.selection_set,
            OperationDefinition::Mutation(m) => &m.selection_set,
            OperationDefinition::Subscription(s) => &s.selection_set,
        }),
        Definition::Fragment(f) => Some(&f.selection_set),
    }
}

/// The root-fields / aliases over-limit message:
/// `"GraphQL queries should have at most {limit} {kind}, got {value} {kind}. To
/// increase the limit, set the {env_var} environment variable."`.
fn limit_message(kind: &str, env_var: &str, limit: usize, value: usize) -> String {
    format!(
        "GraphQL queries should have at most {limit} {kind}, got {value} {kind}. \
         To increase the limit, set the {env_var} environment variable."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphql_parser::query::parse_query;

    fn check(q: &str) -> SafetyResult {
        let doc = parse_query::<String>(q).expect("valid query");
        check_query_safety(&doc)
    }

    #[test]
    fn conforming_query_is_safe() {
        assert!(check("query Q { mlflowGetRun(input: {runId: \"r\"}) { run { info { runId } } } }")
            .is_ok());
    }

    #[test]
    fn too_many_root_fields_reports_limit() {
        let fields = (0..MAX_ROOT_FIELDS + 2)
            .map(|i| format!("k_{i}: test(inputString: \"a\") {{ output }}"))
            .collect::<Vec<_>>()
            .join(" ");
        let err = check(&format!("query Q {{ {fields} }}")).unwrap_err();
        assert!(
            err.starts_with(&format!("GraphQL queries should have at most {MAX_ROOT_FIELDS}")),
            "{err}"
        );
        assert!(err.contains("root fields"), "{err}");
    }

    #[test]
    fn too_many_aliases_reports_limit() {
        let aliases = (0..MAX_ALIASES + 2)
            .map(|i| format!("e_{i}: experiment {{ name }}"))
            .collect::<Vec<_>>()
            .join(" ");
        let q = format!("query Q {{ mlflowGetExperiment(input: {{experimentId: \"1\"}}) {{ {aliases} }} }}");
        let err = check(&q).unwrap_err();
        assert!(
            err.contains(&format!("at most {MAX_ALIASES} aliases")),
            "{err}"
        );
    }

    #[test]
    fn too_deep_reports_depth() {
        let mut inner = String::from("name");
        for _ in 0..12 {
            inner = format!("name {{ {inner} }}");
        }
        let q = format!(
            "query Q {{ mlflowGetExperiment(input: {{experimentId: \"1\"}}) {{ experiment {{ {inner} }} }} }}"
        );
        assert_eq!(
            check(&q).unwrap_err(),
            format!("Query exceeds maximum depth of {MAX_DEPTH}")
        );
    }

    #[test]
    fn too_many_selections_reports_limit() {
        let selections = (0..1002)
            .map(|i| format!("field_{i} {{ name }}"))
            .collect::<Vec<_>>()
            .join(" ");
        let q = format!(
            "query Q {{ mlflowGetExperiment(input: {{experimentId: \"1\"}}) {{ experiment {{ {selections} }} }} }}"
        );
        assert_eq!(
            check(&q).unwrap_err(),
            format!("Query exceeds maximum total selections of {MAX_SELECTIONS}")
        );
    }
}
