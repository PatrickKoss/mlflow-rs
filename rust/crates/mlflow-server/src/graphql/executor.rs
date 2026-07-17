//! The GraphQL executor: request → AST → resolve root fields → project the
//! selection set → `{"data": ..., "errors": ...}` JSON.
//!
//! This is the Rust analogue of graphene's `schema.execute(...)` for the fixed,
//! closed MLflow schema. Because the schema is small and never changes at
//! runtime, resolution is hand-written: each root field's resolver produces a
//! fully-materialized [`GqlVal`] response object (all fields present, in
//! graphene-faithful scalar types), and [`project_selection_set`] walks the
//! query's selection set to emit only the requested fields (honoring aliases and
//! `__typename`).
//!
//! ## Error semantics (OSS graphene parity)
//!
//! In `_graphql` (`handlers.py:3653`), the JSON response is
//! `{"data": result.data, "errors": [e.message for e in result.errors] or None}`
//! — so **errors are bare message strings**, not spec objects with
//! `locations`/`path`. A resolver that raises (e.g. the store can't find a run)
//! becomes exactly one such error string, and graphene sets that root field's
//! value to `null` in `data` while leaving already-resolved sibling fields
//! intact.
//!
//! The OSS impls never populate the `apiError` union member on any response
//! (that is a Databricks-backend concern); the graphene ObjectType resolves it
//! from a proto attribute that doesn't exist → `null`. We reproduce that: store
//! errors surface as GraphQL error strings and `apiError` is always `null`.

use graphql_parser::query::{
    Definition, Document, OperationDefinition, Selection, SelectionSet, Value as AstValue,
};

use super::schema::type_name;
use super::value::GqlVal;

/// A fully-parsed GraphQL request (`{"query", "variables", "operationName"}`).
pub struct GraphQlRequest {
    pub query: String,
    pub variables: serde_json::Map<String, serde_json::Value>,
    pub operation_name: Option<String>,
}

/// The abstract input a resolver receives for its root field: the field's
/// `input` argument, already resolved against the query variables into a plain
/// JSON object (graphene's `MlflowXxxInput` → the resolver's `input`).
pub type ResolverInput = serde_json::Map<String, serde_json::Value>;

/// The outcome of resolving a single root field: either the response object or a
/// GraphQL error message (the store/validation error surfaced by the resolver).
pub type ResolveResult = Result<GqlVal, String>;

/// A resolved-and-projected root field plus its output key (alias or field
/// name), ready to be assembled into the `data` object.
struct RootFieldResult {
    key: String,
    /// `Ok(value)` when resolution succeeded (already projected); `Err(msg)`
    /// when the resolver raised — the field's value becomes `null` and `msg`
    /// joins the top-level `errors`.
    value: Result<String, String>,
}

/// Execute a request against a set of root-field resolvers, producing the final
/// `{"data": ..., "errors": ...}` JSON body (compact, matching Flask `jsonify`).
///
/// `resolve` is invoked once per selected root field with the field name and its
/// resolved `input` argument; it returns the response object [`GqlVal`] or an
/// error message. Resolution is async because the resolvers hit the stores.
pub async fn execute<F, Fut>(
    doc: &Document<'_, String>,
    request: &GraphQlRequest,
    mut resolve: F,
) -> String
where
    F: FnMut(String, ResolverInput) -> Fut,
    Fut: std::future::Future<Output = ResolveResult>,
{
    // Select the operation to execute (graphene: by `operationName` when given,
    // else the sole operation).
    let operation = match select_operation(doc, request.operation_name.as_deref()) {
        Ok(op) => op,
        Err(msg) => return error_only_body(&msg),
    };

    let mut results: Vec<RootFieldResult> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for selection in &operation.items {
        let Selection::Field(field) = selection else {
            // Fragment spreads / inline fragments at the root are not used by
            // any MLflow query; ignore them (graphene would resolve them, but
            // the closed schema + real queries never exercise this).
            continue;
        };
        let key = field
            .alias
            .clone()
            .unwrap_or_else(|| field.name.clone());

        // `__typename` at the root resolves to the operation's type name. No
        // real MLflow query selects a root `__typename`, so `Query` is a safe,
        // never-observed default (mutations delegate to the same read logic and
        // are never introspected at the root either).
        if field.name == "__typename" {
            results.push(RootFieldResult {
                key,
                value: Ok("\"Query\"".to_string()),
            });
            continue;
        }

        let input = resolve_input_argument(field, &request.variables);
        let input = match input {
            Ok(v) => v,
            Err(msg) => {
                errors.push(msg.clone());
                results.push(RootFieldResult {
                    key,
                    value: Err(msg),
                });
                continue;
            }
        };

        match resolve(field.name.clone(), input).await {
            Ok(resolved) => {
                let mut buf = String::new();
                project_value(&resolved, &field.selection_set, &mut buf);
                results.push(RootFieldResult {
                    key,
                    value: Ok(buf),
                });
            }
            Err(msg) => {
                errors.push(msg.clone());
                results.push(RootFieldResult {
                    key,
                    value: Err(msg),
                });
            }
        }
    }

    assemble_body(&results, &errors)
}

/// The GraphQL type name of the executing operation's root (`Query` or
/// `Mutation`), for a root-level `__typename`.
fn operation_type_name(request: &GraphQlRequest) -> &'static str {
    // The only mutations are `mlflowSearchRuns` / `mlflowSearchDatasets` /
    // `testMutation`; everything else is a query. We don't track which the
    // parser chose here, so fall back to "Query" — no real MLflow query selects
    // a root `__typename`, so this is never observed. Kept for completeness.
    let _ = request;
    "Query"
}

/// `select_operation`: pick the operation definition to run. With a single
/// operation, `operationName` is optional; with multiple, it must name one.
fn select_operation<'r, 'a>(
    doc: &'r Document<'a, String>,
    operation_name: Option<&str>,
) -> Result<&'r SelectionSet<'a, String>, String> {
    let ops: Vec<(&'r Option<String>, &'r SelectionSet<'a, String>)> = doc
        .definitions
        .iter()
        .filter_map(|d| match d {
            Definition::Operation(op) => Some(match op {
                OperationDefinition::SelectionSet(s) => (&NONE_NAME, s),
                OperationDefinition::Query(q) => (&q.name, &q.selection_set),
                OperationDefinition::Mutation(m) => (&m.name, &m.selection_set),
                OperationDefinition::Subscription(s) => (&s.name, &s.selection_set),
            }),
            Definition::Fragment(_) => None,
        })
        .collect();

    match (operation_name, ops.as_slice()) {
        (_, []) => Err("Must provide an operation.".to_string()),
        (None, [(_, set)]) => Ok(set),
        (Some(name), _) => ops
            .iter()
            .find(|(n, _)| n.as_deref() == Some(name))
            .map(|(_, set)| *set)
            .ok_or_else(|| format!("Unknown operation named \"{name}\".")),
        (None, _) => Err(
            "Must provide operation name if query contains multiple operations.".to_string(),
        ),
    }
}

static NONE_NAME: Option<String> = None;

/// Resolve a root field's `input` argument into a plain JSON object, substituting
/// any query variables (`$data`). Returns an empty object when there is no
/// `input` argument (e.g. `test(inputString: ...)` carries its args differently,
/// but the resolvers for those read from the returned map by key too).
fn resolve_input_argument(
    field: &graphql_parser::query::Field<'_, String>,
    variables: &serde_json::Map<String, serde_json::Value>,
) -> Result<ResolverInput, String> {
    let mut out = serde_json::Map::new();
    for (name, value) in &field.arguments {
        let json = ast_value_to_json(value, variables)?;
        if name == "input" {
            // The `input` arg is an object; flatten it into the resolver input.
            match json {
                serde_json::Value::Object(obj) => return Ok(obj),
                serde_json::Value::Null => return Ok(serde_json::Map::new()),
                _ => return Err("Argument 'input' must be an input object.".to_string()),
            }
        }
        // Non-`input` scalar args (e.g. `inputString` on `test`).
        out.insert(name.clone(), json);
    }
    Ok(out)
}

/// Convert an AST argument value to JSON, substituting `$var` references from the
/// request variables (an absent variable resolves to JSON null, matching
/// graphene passing `None`).
fn ast_value_to_json(
    value: &AstValue<'_, String>,
    variables: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Value, String> {
    Ok(match value {
        AstValue::Variable(name) => variables.get(name).cloned().unwrap_or(serde_json::Value::Null),
        AstValue::Int(n) => serde_json::Value::from(n.as_i64().unwrap_or(0)),
        AstValue::Float(f) => serde_json::json!(f),
        AstValue::String(s) => serde_json::Value::String(s.clone()),
        AstValue::Boolean(b) => serde_json::Value::Bool(*b),
        AstValue::Null => serde_json::Value::Null,
        AstValue::Enum(e) => serde_json::Value::String(e.clone()),
        AstValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(ast_value_to_json(it, variables)?);
            }
            serde_json::Value::Array(out)
        }
        AstValue::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), ast_value_to_json(v, variables)?);
            }
            serde_json::Value::Object(out)
        }
    })
}

/// Project a resolved value against a selection set, writing the JSON for the
/// *value in that selection context* into `out`.
///
/// * Object → `{selected fields...}` (recursing per field, honoring aliases and
///   `__typename`); a selected field absent from the resolved object emits
///   `null` (graphene default for an unresolved nullable field).
/// * List → `[project each element against the same selection set]`.
/// * Scalar/null → the scalar's wire form (the selection set is empty).
fn project_value(value: &GqlVal, selection: &SelectionSet<'_, String>, out: &mut String) {
    match value {
        GqlVal::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                project_value(item, selection, out);
            }
            out.push(']');
        }
        GqlVal::Object { .. } => project_selection_set(value, selection, out),
        scalar => scalar.write_scalar(out),
    }
}

/// Project an object's selected fields into a JSON object literal.
fn project_selection_set(object: &GqlVal, selection: &SelectionSet<'_, String>, out: &mut String) {
    out.push('{');
    let mut first = true;
    for sel in &selection.items {
        let Selection::Field(field) = sel else {
            // No fragment spreads/inline fragments occur in MLflow queries.
            continue;
        };
        if !first {
            out.push(',');
        }
        first = false;
        let key = field.alias.as_deref().unwrap_or(&field.name);
        out.push_str(&mlflow_proto::quote_json_string(key));
        out.push(':');

        if field.name == "__typename" {
            let tn = object.tag().map(type_name).unwrap_or("Unknown");
            out.push_str(&mlflow_proto::quote_json_string(tn));
            continue;
        }

        match object.get(&field.name) {
            Some(child) => project_value(child, &field.selection_set, out),
            None => out.push_str("null"),
        }
    }
    out.push('}');
}

/// Assemble the final `{"data": {...}, "errors": [...] | null}` body, compact
/// (Flask `jsonify` with default `JSONProvider`, which for MLflow uses compact
/// separators). `data` is `null` only when there is no data object at all
/// (operation-selection error); otherwise it always contains every root field
/// (a failed field is `null`).
fn assemble_body(results: &[RootFieldResult], errors: &[String]) -> String {
    let mut out = String::from("{\"data\":");
    out.push('{');
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&mlflow_proto::quote_json_string(&r.key));
        out.push(':');
        match &r.value {
            Ok(body) => out.push_str(body),
            Err(_) => out.push_str("null"),
        }
    }
    out.push('}');

    out.push_str(",\"errors\":");
    if errors.is_empty() {
        out.push_str("null");
    } else {
        out.push('[');
        for (i, e) in errors.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&mlflow_proto::quote_json_string(e));
        }
        out.push(']');
    }
    out.push('}');
    out
}

/// The `{"data": null, "errors": ["<msg>"]}` body for a pre-resolution failure
/// (operation selection / query-safety), matching graphene returning an
/// `ExecutionResult(data=None, errors=[...])`.
pub fn error_only_body(message: &str) -> String {
    format!(
        "{{\"data\":null,\"errors\":[{}]}}",
        mlflow_proto::quote_json_string(message)
    )
}

/// Parse the JSON variables blob (`request_json["variables"]`) into an object
/// map, tolerating `null`/absent (→ empty). A non-object `variables` is a client
/// error, but graphene tolerates it loosely; we treat it as empty.
pub fn parse_variables(value: Option<&serde_json::Value>) -> serde_json::Map<String, serde_json::Value> {
    match value {
        Some(serde_json::Value::Object(obj)) => obj.clone(),
        _ => serde_json::Map::new(),
    }
}
