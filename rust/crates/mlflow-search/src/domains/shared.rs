//! Cross-domain parse drivers: the `parse_search_filter` skeleton and the
//! order_by string tokenization (`_validate_order_by_and_generate_token` +
//! `_parse_order_by_string`, including the `shlex.split` step).

use crate::error::{Result, SearchError};
use crate::group::{parse_statement, GroupClass, Node, Statement};
use crate::token::{lex, TokenKind};

/// `parse_search_filter` skeleton shared by every domain: empty → `[]`, else
/// tokenize + group into a single statement, then delegate to the domain's
/// clause processor. sqlparse always yields exactly one Statement for a
/// non-empty string here (it splits on `;` but our grammar rejects `;`
/// downstream as multiple expressions), so the multi-statement branch is
/// modeled by detecting a top-level `;` punctuation token.
pub fn parse_filter_statement(filter_string: &str) -> Option<FilterInput> {
    if filter_string.is_empty() {
        return None;
    }
    let tokens = lex(filter_string);
    // sqlparse splits statements on ';'. MLflow's "multiple expression" error
    // fires when there is more than one statement. Detect a semicolon that
    // yields a non-empty second statement.
    Some(FilterInput {
        statement: parse_statement(tokens),
        raw: filter_string.to_string(),
    })
}

/// The grouped statement plus the original string (for error messages).
pub struct FilterInput {
    pub statement: Statement,
    pub raw: String,
}

impl FilterInput {
    /// The "multiple expression" error, if the filter contains a second
    /// statement. Uses Python `repr()` of the raw string (single quotes).
    pub fn multiple_expression_error(&self) -> Option<SearchError> {
        if self.multiple_statements() {
            Some(SearchError::invalid_parameter_value(format!(
                "Search filter contained multiple expression {}. Provide AND-ed expression list.",
                crate::literal_eval::py_repr_str(&self.raw)
            )))
        } else {
            None
        }
    }

    /// Emulate sqlparse statement splitting on `;`: MLflow raises
    /// "Search filter contained multiple expression" when >1 statement. We
    /// treat a top-level `;` followed by any non-whitespace token as a second
    /// statement.
    pub fn multiple_statements(&self) -> bool {
        let toks = &self.statement.tokens;
        for (i, t) in toks.iter().enumerate() {
            let is_semi =
                matches!(t, Node::Leaf(x) if x.kind == TokenKind::Punctuation && x.value == ";");
            if is_semi && toks[i + 1..].iter().any(|n| !n.is_whitespace()) {
                return true;
            }
        }
        false
    }
}

/// `_parse_order_by_string`: returns `(token_value, is_ascending)`.
///
/// Steps: validate the order_by parses to a single Identifier (or `timestamp`
/// (+order) via the trace/registered-model builtin path), then `shlex.split`
/// with backticks rewritten to double-quotes to peel off ASC/DESC and strip
/// quotes.
pub fn parse_order_by_string(order_by: &str) -> Result<(String, bool)> {
    let token_value = validate_order_by_and_generate_token(order_by)?;
    let mut is_ascending = true;
    let tokens =
        shlex_split(&token_value.replace('`', "\"")).ok_or_else(|| invalid_order_by(order_by))?;
    if tokens.len() > 2 {
        return Err(invalid_order_by(order_by));
    }
    // NB: Python only reassigns `token_value = tokens[0]` in the len==2 branch.
    // For a single token it keeps the *original* token_value (still quoted),
    // e.g. order_by "`metrics.A`" stays "`metrics.A`" — which then fails entity
    // validation with the backtick still attached.
    let mut value = token_value;
    if tokens.len() == 2 {
        let order_token = tokens[1].to_lowercase();
        if order_token != "asc" && order_token != "desc" {
            return Err(SearchError::invalid_parameter_value(format!(
                "Invalid ordering key in order_by clause '{order_by}'."
            )));
        }
        is_ascending = order_token == "asc";
        value = tokens[0].clone();
    }
    Ok((value, is_ascending))
}

fn invalid_order_by(order_by: &str) -> SearchError {
    SearchError::invalid_parameter_value(format!(
        "Invalid order_by clause '{order_by}'. Could not be parsed."
    ))
}

const ORDER_BY_KEY_TIMESTAMP: &str = "timestamp";

/// `_validate_order_by_and_generate_token`.
fn validate_order_by_and_generate_token(order_by: &str) -> Result<String> {
    let tokens = lex(order_by);
    let stmt = parse_statement(tokens);

    // len(parsed) != 1 or not a Statement: our lexer always yields one
    // statement, but an empty/blank order_by parses to a statement whose only
    // tokens are whitespace (or nothing) → the single-Identifier check fails →
    // "Could not be parsed" via the final else.

    let toks = &stmt.tokens;

    // Case 1: exactly one token and it is an Identifier.
    if toks.len() == 1 {
        if let Node::Group {
            class: GroupClass::Identifier,
            ..
        } = &toks[0]
        {
            return Ok(toks[0].value());
        }
        // Case 2: exactly one token matching the `timestamp` builtin.
        if is_timestamp_builtin(&toks[0]) {
            return Ok(ORDER_BY_KEY_TIMESTAMP.to_string());
        }
    }

    // Case 3: `timestamp` (+ only whitespace) + a trailing Order keyword.
    if !toks.is_empty()
        && is_timestamp_builtin(&toks[0])
        && toks[1..toks.len() - 1].iter().all(|t| t.is_whitespace())
        && matches!(toks.last().unwrap().ttype(), Some(TokenKind::KeywordOrder))
    {
        return Ok(format!(
            "{ORDER_BY_KEY_TIMESTAMP} {}",
            toks.last().unwrap().value()
        ));
    }

    Err(invalid_order_by(order_by))
}

fn is_timestamp_builtin(node: &Node) -> bool {
    matches!(node, Node::Leaf(t)
        if t.kind == TokenKind::NameBuiltin && t.value.eq_ignore_ascii_case(ORDER_BY_KEY_TIMESTAMP))
}

/// A faithful port of `shlex.split(s)` (POSIX mode, the default) for the inputs
/// that reach it here: whitespace splitting with single/double quote removal
/// and backslash escaping. Returns `None` on an unterminated quote (Python
/// raises `ValueError`, which MLflow surfaces as the generic parse error via
/// the caller — though in practice quotes are already balanced by this point).
pub fn shlex_split(s: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut out = Vec::new();

    while i < n {
        // skip leading whitespace
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let mut token = String::new();
        let mut in_token = false;
        while i < n {
            let c = chars[i];
            if c.is_whitespace() {
                break;
            }
            in_token = true;
            match c {
                '\\' => {
                    // POSIX: backslash escapes the next char (outside quotes).
                    if i + 1 < n {
                        token.push(chars[i + 1]);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                '\'' => {
                    // single quotes: literal until next single quote
                    i += 1;
                    let start = i;
                    while i < n && chars[i] != '\'' {
                        i += 1;
                    }
                    if i >= n {
                        return None; // unterminated
                    }
                    token.extend(&chars[start..i]);
                    i += 1;
                }
                '"' => {
                    // double quotes: backslash escapes " and \ inside
                    i += 1;
                    while i < n && chars[i] != '"' {
                        if chars[i] == '\\' && i + 1 < n && matches!(chars[i + 1], '"' | '\\') {
                            token.push(chars[i + 1]);
                            i += 2;
                        } else {
                            token.push(chars[i]);
                            i += 1;
                        }
                    }
                    if i >= n {
                        return None; // unterminated
                    }
                    i += 1;
                }
                _ => {
                    token.push(c);
                    i += 1;
                }
            }
        }
        if in_token {
            out.push(token);
        }
    }
    Some(out)
}
