//! Shared parser helpers ported from `SearchUtils` base-class methods.
//!
//! These mirror the class-level helpers that every domain reuses:
//! `_is_quoted`, `_trim_ends`, `_trim_backticks`, `_strip_quotes`, the
//! statement-splitting / invalid-clause detection, and the `_join_in_comparison_tokens`
//! IN/IS grouping that runs on top of sqlparse's grouping.

use crate::error::{Result, SearchError};
use crate::group::{group_children, GroupClass, Node, Statement};
use crate::token::TokenKind;

/// `_is_quoted(value, pattern)`: len>=2 and starts+ends with the 1-char pattern.
pub fn is_quoted(value: &str, pat: char) -> bool {
    let n = value.chars().count();
    n >= 2 && value.starts_with(pat) && value.ends_with(pat)
}

/// `_trim_ends`: drop the first and last character.
pub fn trim_ends(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars[1..chars.len() - 1].iter().collect()
}

/// `_trim_backticks`: remove surrounding backticks if present.
pub fn trim_backticks(s: &str) -> String {
    if is_quoted(s, '`') {
        trim_ends(s)
    } else {
        s.to_string()
    }
}

/// `_strip_quotes(value, expect_quoted_value)`.
pub fn strip_quotes(value: &str, expect_quoted_value: bool) -> Result<String> {
    if is_quoted(value, '\'') || is_quoted(value, '"') {
        Ok(trim_ends(value))
    } else if expect_quoted_value {
        Err(SearchError::invalid_parameter_value(format!(
            "Parameter value is either not quoted or unidentified quote types used for string \
             value {value}. Use either single or double quotes."
        )))
    } else {
        Ok(value.to_string())
    }
}

/// A node in a comparison after grouping + IN/IS post-joining. This mirrors the
/// heterogeneous token stream MLflow iterates: either a sqlparse node, or a
/// synthetic keyword token that `_join_in_comparison_tokens` fabricates for
/// `IS NULL` / `IS NOT NULL` / `NOT IN`.
#[derive(Debug, Clone)]
pub enum CmpToken {
    /// A real grouped node.
    Node(Node),
    /// A synthesized keyword token (value = e.g. "IS NULL", "NOT IN").
    Keyword(String),
}

impl CmpToken {
    pub fn value(&self) -> String {
        match self {
            CmpToken::Node(n) => n.value(),
            CmpToken::Keyword(s) => s.clone(),
        }
    }
    pub fn is_identifier(&self) -> bool {
        matches!(self, CmpToken::Node(n) if n.class() == Some(GroupClass::Identifier))
    }
    pub fn is_parenthesis(&self) -> bool {
        matches!(self, CmpToken::Node(n) if n.class() == Some(GroupClass::Parenthesis))
    }
    pub fn ttype(&self) -> Option<TokenKind> {
        match self {
            CmpToken::Node(n) => n.ttype(),
            CmpToken::Keyword(_) => Some(TokenKind::Keyword),
        }
    }
}

/// A valid comparison clause: its stripped, non-whitespace tokens.
pub type Clause = Vec<CmpToken>;

/// True if a top-level node is an `AND` keyword.
fn is_and(node: &Node) -> bool {
    matches!(node, Node::Leaf(t) if t.kind == TokenKind::Keyword && t.value.eq_ignore_ascii_case("and"))
}

/// A node that matches a builtin keyword name (for the trace `timestamp` path).
fn is_builtin_named(node: &Node, names: &[&str]) -> bool {
    matches!(node, Node::Leaf(t) if t.kind == TokenKind::NameBuiltin
        && names.iter().any(|n| t.value.eq_ignore_ascii_case(n)))
}

fn is_keyword_value(node: &Node, value: &str) -> bool {
    matches!(node, Node::Leaf(t) if t.kind == TokenKind::Keyword && t.value.eq_ignore_ascii_case(value))
}

/// Split a statement into clauses, applying `_join_in_comparison_tokens`
/// (IN / NOT IN / IS NULL / IS NOT NULL joining) and the invalid-token filter.
///
/// `search_traces` toggles the trace-specific `timestamp`/`timestamp_ms`
/// numeric-comparison path in `_join_in_comparison_tokens`.
pub fn process_statement(
    stmt: &Statement,
    search_traces: bool,
    quote_invalids: bool,
) -> Result<Vec<Clause>> {
    let joined = join_in_comparison_tokens(&stmt.tokens, search_traces)?;

    // Filter invalid statement tokens: not a Comparison, not whitespace, not AND.
    let mut clauses: Vec<Clause> = Vec::new();
    let mut invalids: Vec<String> = Vec::new();
    for item in &joined {
        match item {
            JoinItem::Comparison(children) => clauses.push(children.clone()),
            JoinItem::Node(node) => {
                if node.is_whitespace() || is_and(node) {
                    continue;
                }
                invalids.push(node.value());
            }
            JoinItem::Keyword(_) => {
                // A leftover synthetic keyword can't appear at statement level.
            }
        }
    }

    if !invalids.is_empty() {
        // Runs/traces/logged-models quote each invalid clause (`f"'{token}'"`);
        // experiments/registered-models/model-versions use `str(token)`.
        let joined = invalids
            .iter()
            .map(|s| {
                if quote_invalids {
                    format!("'{s}'")
                } else {
                    s.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(SearchError::invalid_parameter_value(format!(
            "Invalid clause(s) in filter string: {joined}"
        )));
    }
    Ok(clauses)
}

/// An item produced by `_join_in_comparison_tokens`: an already-grouped
/// sqlparse Comparison (its stripped children), a raw node, or a synthetic
/// keyword (unused at top level but kept for fidelity).
#[derive(Debug, Clone)]
enum JoinItem {
    Comparison(Vec<CmpToken>),
    Node(Node),
    #[allow(dead_code)]
    Keyword(String),
}

/// Stripped non-whitespace children of a grouped Comparison node, as `CmpToken`s.
fn comparison_children(node: &Node) -> Vec<CmpToken> {
    group_children(node)
        .iter()
        .filter(|c| !c.is_whitespace())
        .cloned()
        .map(CmpToken::Node)
        .collect()
}

/// Port of `_join_in_comparison_tokens` (sqlparse >= 0.4.4 branch).
///
/// Operates on the non-whitespace top-level tokens, matching IN / NOT IN /
/// IS NULL / IS NOT NULL sequences (and, for traces, `timestamp <op> <int>`)
/// into synthetic Comparison items.
fn join_in_comparison_tokens(tokens: &[Node], search_traces: bool) -> Result<Vec<JoinItem>> {
    let nz: Vec<&Node> = tokens.iter().filter(|t| !t.is_whitespace()).collect();
    let mut out: Vec<JoinItem> = Vec::new();
    let num = nz.len();
    let mut index = 0;

    while index < num {
        let first = nz[index];

        // Not enough tokens left to form an IN/NOT-IN comparison → passthrough.
        if num - index < 3 {
            for t in &nz[index..] {
                out.push(promote(t));
            }
            break;
        }

        if search_traces && is_builtin_named(first, &["timestamp", "timestamp_ms"]) {
            // Python consumes second+third here via next(iterator).
            let second = nz[index + 1];
            let third = nz[index + 2];
            let second_ok = matches!(second, Node::Leaf(t)
                if t.kind == TokenKind::OperatorComparison
                    && VALID_NUMERIC_ATTRIBUTE_COMPARATORS.contains(&t.value.as_str()));
            let third_ok = matches!(third.ttype(), Some(TokenKind::Integer));
            if second_ok && third_ok {
                out.push(JoinItem::Comparison(vec![
                    CmpToken::Node(first.clone()),
                    CmpToken::Node(second.clone()),
                    CmpToken::Node(third.clone()),
                ]));
                index += 3;
                continue;
            }
            // else: Python `joined.extend([first, second, third])` then FALLS
            // THROUGH (no `continue`) to the `not isinstance(first, Identifier)`
            // check below. `first` (a Name.Builtin) is never an Identifier, so
            // it appends `first` a second time and continues. Net: emit
            // [first, second, third, first] and advance past all three.
            out.push(promote(first));
            out.push(promote(second));
            out.push(promote(third));
            out.push(promote(first));
            index += 3;
            continue;
        }

        if first.class() != Some(GroupClass::Identifier) {
            out.push(promote(first));
            index += 1;
            continue;
        }

        let second = nz[index + 1];
        let third = nz[index + 2];

        // IN
        if is_keyword_value(second, "IN") && third.class() == Some(GroupClass::Parenthesis) {
            out.push(JoinItem::Comparison(vec![
                CmpToken::Node(first.clone()),
                CmpToken::Keyword("IN".to_string()),
                CmpToken::Node(third.clone()),
            ]));
            index += 3;
            continue;
        }

        // IS NULL
        if is_keyword_value(second, "IS") && is_keyword_value(third, "NULL") {
            out.push(JoinItem::Comparison(vec![
                CmpToken::Node(first.clone()),
                CmpToken::Keyword("IS NULL".to_string()),
            ]));
            index += 3;
            continue;
        }

        // IS NOT NULL: third is a Keyword whose upper == "NOT NULL"
        if is_keyword_value(second, "IS")
            && matches!(third, Node::Leaf(t)
                if t.kind == TokenKind::Keyword && t.value.to_uppercase() == "NOT NULL")
        {
            out.push(JoinItem::Comparison(vec![
                CmpToken::Node(first.clone()),
                CmpToken::Keyword("IS NOT NULL".to_string()),
            ]));
            index += 3;
            continue;
        }

        if index + 3 >= num {
            out.push(promote(first));
            out.push(promote(second));
            out.push(promote(third));
            break;
        }
        let fourth = nz[index + 3];

        // NOT IN
        if is_keyword_value(second, "NOT")
            && is_keyword_value(third, "IN")
            && fourth.class() == Some(GroupClass::Parenthesis)
        {
            out.push(JoinItem::Comparison(vec![
                CmpToken::Node(first.clone()),
                CmpToken::Keyword("NOT IN".to_string()),
                CmpToken::Node(fourth.clone()),
            ]));
            index += 4;
            continue;
        }

        out.push(promote(first));
        out.push(promote(second));
        out.push(promote(third));
        out.push(promote(fourth));
        index += 4;
    }

    Ok(out)
}

/// Turn a top-level node into a JoinItem, unwrapping already-grouped
/// Comparison nodes into their stripped children.
fn promote(node: &Node) -> JoinItem {
    if node.class() == Some(GroupClass::Comparison) {
        JoinItem::Comparison(comparison_children(node))
    } else {
        JoinItem::Node(node.clone())
    }
}

/// Numeric comparators (`VALID_METRIC_COMPARATORS` / `VALID_NUMERIC_ATTRIBUTE_COMPARATORS`).
pub const VALID_NUMERIC_ATTRIBUTE_COMPARATORS: &[&str] = &[">", ">=", "!=", "=", "<", "<="];
