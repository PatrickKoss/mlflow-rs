//! A focused port of `ast.literal_eval` for the parenthesised IN-list values.
//!
//! MLflow parses `IN (...)` list values with
//! `ast.literal_eval(token.value)` (`SearchUtils._parse_list_from_sql_token`),
//! then validates the result with `_check_valid_identifier_list` and filters
//! run_ids with `str.islower()`. The observable behaviors we must reproduce:
//!
//! - `('a', 'b')` → tuple of two strings.
//! - `('a')` → a single string (not a tuple); MLflow then wraps it as `('a',)`.
//! - `()` → empty tuple → "expected a non-empty list ... empty list" error.
//! - `('a', 5)` → tuple `('a', 5)`; the non-string element triggers the
//!   "got different type in list: ('a', 5)" error, whose text embeds Python's
//!   `repr()` of the tuple.
//! - `('a' 'b')` → implicit string concatenation → `'ab'`.
//! - A malformed list raises `SyntaxError` → "ill-formed list" error.
//!
//! Only string and numeric literals (and nested tuples, which MLflow never
//! produces here) are supported — enough for the grammar. Anything else yields
//! the ill-formed-list error, matching a Python `ValueError`/`SyntaxError`.

use crate::error::{Result, SearchError};

/// A Python literal value produced by [`literal_eval`].
#[derive(Debug, Clone, PartialEq)]
pub enum PyLit {
    Str(String),
    Int(i64),
    Float(f64),
    Tuple(Vec<PyLit>),
}

impl PyLit {
    /// Python `repr()` of the value (used verbatim in an MLflow error message).
    pub fn repr(&self) -> String {
        match self {
            PyLit::Str(s) => py_repr_str(s),
            PyLit::Int(i) => i.to_string(),
            PyLit::Float(f) => py_repr_float(*f),
            PyLit::Tuple(items) => {
                if items.len() == 1 {
                    format!("({},)", items[0].repr())
                } else {
                    let inner: Vec<String> = items.iter().map(PyLit::repr).collect();
                    format!("({})", inner.join(", "))
                }
            }
        }
    }
}

/// Mirrors `ast.literal_eval(text)` for the supported literal subset. Returns
/// the ill-formed-list `SearchError` on any parse failure (Python raises
/// `SyntaxError`/`ValueError`, both mapped by MLflow to the same message).
pub fn literal_eval(text: &str) -> Result<PyLit> {
    let chars: Vec<char> = text.chars().collect();
    let mut p = Parser { chars, pos: 0 };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(ill_formed());
    }
    Ok(value)
}

fn ill_formed() -> SearchError {
    SearchError::invalid_parameter_value(
        "While parsing a list in the query, expected a non-empty list of string values, \
         but got ill-formed list.",
    )
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// Parse a single literal: a (possibly parenthesised) value. A parenthesis
    /// forms a tuple when it contains a comma; otherwise it is grouping.
    fn parse_value(&mut self) -> Result<PyLit> {
        match self.peek() {
            Some('(') => self.parse_paren(),
            Some('\'') | Some('"') => Ok(PyLit::Str(self.parse_string_concat()?)),
            Some(c) if c == '-' || c == '+' || c.is_ascii_digit() || c == '.' => {
                self.parse_number()
            }
            _ => Err(ill_formed()),
        }
    }

    fn parse_paren(&mut self) -> Result<PyLit> {
        // consume '('
        self.pos += 1;
        self.skip_ws();
        let mut items: Vec<PyLit> = Vec::new();
        let mut saw_comma = false;

        if self.peek() == Some(')') {
            self.pos += 1;
            return Ok(PyLit::Tuple(Vec::new())); // ()
        }

        loop {
            self.skip_ws();
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    saw_comma = true;
                    self.pos += 1;
                    self.skip_ws();
                    if self.peek() == Some(')') {
                        // trailing comma
                        self.pos += 1;
                        break;
                    }
                }
                Some(')') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ill_formed()),
            }
        }

        if saw_comma {
            Ok(PyLit::Tuple(items))
        } else {
            // Single grouped value: `('a')` == `'a'`.
            Ok(items.into_iter().next().unwrap())
        }
    }

    /// Parse one or more adjacent string literals (Python implicit concat).
    fn parse_string_concat(&mut self) -> Result<String> {
        let mut out = String::new();
        let first = self.parse_string()?;
        out.push_str(&first);
        loop {
            let save = self.pos;
            self.skip_ws();
            if matches!(self.peek(), Some('\'') | Some('"')) {
                let next = self.parse_string()?;
                out.push_str(&next);
            } else {
                self.pos = save;
                break;
            }
        }
        Ok(out)
    }

    fn parse_string(&mut self) -> Result<String> {
        let quote = self.peek().ok_or_else(ill_formed)?;
        if quote != '\'' && quote != '"' {
            return Err(ill_formed());
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '\\' {
                self.pos += 1;
                let e = self.peek().ok_or_else(ill_formed)?;
                out.push(unescape(e));
                self.pos += 1;
                continue;
            }
            if c == quote {
                self.pos += 1;
                return Ok(out);
            }
            out.push(c);
            self.pos += 1;
        }
        Err(ill_formed())
    }

    fn parse_number(&mut self) -> Result<PyLit> {
        let start = self.pos;
        if matches!(self.peek(), Some('-') | Some('+')) {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == '.' || c == 'e' || c == 'E' {
                is_float = true;
                self.pos += 1;
            } else if (c == '+' || c == '-')
                && matches!(self.chars.get(self.pos - 1), Some('e') | Some('E'))
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            text.parse::<f64>()
                .map(PyLit::Float)
                .map_err(|_| ill_formed())
        } else {
            text.parse::<i64>()
                .map(PyLit::Int)
                .map_err(|_| ill_formed())
        }
    }
}

fn unescape(c: char) -> char {
    match c {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        '0' => '\0',
        other => other,
    }
}

/// Python `repr()` of a string: prefers single quotes, switches to double
/// quotes if the string contains a single quote but no double quote.
pub fn py_repr_str(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let (quote, escape_quote) = if has_single && !has_double {
        ('"', '"')
    } else {
        ('\'', '\'')
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == escape_quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_repr_float(f: f64) -> String {
    if f == f.trunc() && f.is_finite() {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}
