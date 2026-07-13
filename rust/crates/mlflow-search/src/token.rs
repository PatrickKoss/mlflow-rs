//! sqlparse-compatible tokenizer.
//!
//! MLflow parses filter/order_by strings with the `sqlparse` library, so this
//! crate must reproduce sqlparse's **observable** lexing: the same token
//! boundaries and token kinds, because MLflow's parser branches on
//! `token.ttype` and on `isinstance(token, Identifier/Parenthesis)` groupings.
//!
//! [`lex`] ports the relevant rows of `sqlparse.keywords.SQL_REGEX`
//! (`sqlparse` 0.5.x) in priority order. Only the productions reachable from
//! the MLflow grammar are implemented; anything sqlparse would lex as a
//! comment/CTE/JOIN/etc. is out of the grammar and would surface downstream as
//! an "Invalid clause(s)" error either way. The keyword table is embedded from
//! sqlparse verbatim (see `keywords_generated.rs`).

use crate::keywords_generated::KEYWORDS;

/// The kind a bare word maps to via the sqlparse keyword table.
///
/// `KeywordOrder` never appears in the embedded table (ASC/DESC are matched by
/// a dedicated regex, not keyword lookup) but is kept so the mapping mirrors
/// sqlparse's full taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum KeywordKind {
    Keyword,
    KeywordOrder,
    NameBuiltin,
    /// sqlparse has exactly one keyword typed `Operator` (`DIV`); harmless here.
    Operator,
    Name,
}

/// sqlparse token type, restricted to the kinds the MLflow grammar observes.
///
/// A couple of variants (`Literal`) are reachable by sqlparse but not by the
/// MLflow grammar; they are retained for taxonomic fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TokenKind {
    /// `Token.Name`
    Name,
    /// `Token.Name.Builtin` (e.g. `timestamp`)
    NameBuiltin,
    /// `Token.Keyword` (IS, IN, NOT, AND, OR, NULL, TRUE, ...) and the
    /// composite `NOT NULL`.
    Keyword,
    /// `Token.Keyword.Order` (ASC / DESC, with optional NULLS FIRST/LAST)
    KeywordOrder,
    /// `Token.Operator.Comparison` (`=`, `!=`, `<`, `>`, `<=`, `>=`, `<>`,
    /// `==`, `~`, and word-operators LIKE/ILIKE/RLIKE/REGEXP with optional NOT)
    OperatorComparison,
    /// `Token.Operator` (`-`, `%`, `+`, JSON ops, ...)
    Operator,
    /// `Token.Punctuation` (`.`, `,`, `(`, `)`, `;`, `[`, `]`, `:`)
    Punctuation,
    /// `Token.Literal.String.Single` (`'...'`)
    StringSingle,
    /// `Token.Literal.String.Symbol` (`"..."`)
    StringSymbol,
    /// `Token.Literal.Number.Integer`
    Integer,
    /// `Token.Literal.Number.Float`
    Float,
    /// `Token.Literal.Number.Hexadecimal`
    Hexadecimal,
    /// `Token.Literal` (dollar-quoted) — reachable but grammar-invalid.
    Literal,
    /// `Token.Text.Whitespace` (any run collapses to a single token here; the
    /// grammar only ever checks `is_whitespace`, never the exact run).
    Whitespace,
    /// `Token.Error` — e.g. a lone unmatched quote.
    Error,
    /// `Token.Wildcard` (`*`)
    Wildcard,
}

impl TokenKind {
    pub fn is_whitespace(self) -> bool {
        matches!(self, TokenKind::Whitespace)
    }
}

/// A flat lexer token: kind + exact source slice (`token.value`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub value: String,
}

impl Token {
    fn new(kind: TokenKind, value: &str) -> Self {
        Self {
            kind,
            value: value.to_string(),
        }
    }
}

fn keyword_kind(word: &str) -> KeywordKind {
    let upper = word.to_uppercase();
    match KEYWORDS.binary_search_by(|(k, _)| (*k).cmp(upper.as_str())) {
        Ok(idx) => KEYWORDS[idx].1,
        Err(_) => KeywordKind::Name,
    }
}

fn is_word_char(c: char) -> bool {
    // Python `\w` under `re.UNICODE` plus sqlparse's `[$#\w]` extension.
    c == '$' || c == '#' || c == '_' || c.is_alphanumeric()
}

/// sqlparse's `A-ZÀ-Ü` name-start class (with `re.IGNORECASE` and Unicode):
/// effectively any Unicode letter, plus `_`.
fn is_name_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

/// Tokenize `input` exactly as `sqlparse.parse(input)[0].flatten()` would, for
/// the productions reachable from the MLflow grammar.
pub fn lex(input: &str) -> Vec<Token> {
    let chars: Vec<char> = input.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut out = Vec::new();

    while i < n {
        let c = chars[i];

        // Whitespace run (`\s+?` is non-greedy in sqlparse but adjacent
        // whitespace tokens are indistinguishable to the grammar; we still emit
        // one token per contiguous run, matching `is_whitespace` filtering).
        if c.is_whitespace() {
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            out.push(Token::new(TokenKind::Whitespace, &slice(&chars, start, i)));
            continue;
        }

        // `::` -> Punctuation, `:=` -> Assignment (treated as Punctuation-ish;
        // grammar-invalid regardless). Handle `:` fallthrough below.
        if c == ':' && i + 1 < n && chars[i + 1] == ':' {
            out.push(Token::new(TokenKind::Punctuation, "::"));
            i += 2;
            continue;
        }

        // Wildcard `*`
        if c == '*' {
            out.push(Token::new(TokenKind::Wildcard, "*"));
            i += 1;
            continue;
        }

        // Backtick / acute-accent quoted name: `(``|[^`])*` and ´(´´|[^´])*´
        if c == '`' || c == '\u{b4}' {
            if let Some(end) = match_quoted_name(&chars, i, c) {
                out.push(Token::new(TokenKind::Name, &slice(&chars, i, end)));
                i = end;
                continue;
            }
            // Unterminated: sqlparse emits Error for the lone quote char.
            out.push(Token::new(TokenKind::Error, &slice(&chars, i, i + 1)));
            i += 1;
            continue;
        }

        // Word starting with a name-start char. sqlparse applies several
        // regexes before the generic `\w[$#\w]*` keyword lookup, in this order:
        //   (a) `(CASE|IN|VALUES|USING|FROM|AS)\b` → Keyword (highest priority)
        //   (b) `[A-ZÀ-Ü]\w*(?=\s*\.)` → Name (word before optional-ws + dot)
        //   (c) `(?<=\.)[A-ZÀ-Ü]\w*` → Name (word immediately after a dot)
        //   (d) multi-word keyword forms / word-operators (NOT NULL, ASC/DESC,
        //       LIKE, ...)
        //   (e) generic word → keyword-table lookup.
        // Rules (b)/(c) are why `tags.version` keeps `version` as a Name even
        // though VERSION is a keyword.
        if is_name_start(c) {
            // (a) the six always-keyword words.
            if let Some(end) = match_forced_keyword(&chars, i) {
                out.push(Token::new(TokenKind::Keyword, &slice(&chars, i, end)));
                i = end;
                continue;
            }
            // (c) preceded by a dot → Name. sqlparse's `(?<=\.)` lookbehind is
            // on the raw text; the previous emitted token is a `.` punctuation.
            let after_dot =
                matches!(out.last(), Some(t) if t.kind == TokenKind::Punctuation && t.value == ".");
            if after_dot {
                let end = word_end(&chars, i);
                out.push(Token::new(TokenKind::Name, &slice(&chars, i, end)));
                i = end;
                continue;
            }
            // (b) followed by optional-ws + dot → Name.
            if word_before_dot(&chars, i) {
                let end = word_end(&chars, i);
                out.push(Token::new(TokenKind::Name, &slice(&chars, i, end)));
                i = end;
                continue;
            }
            // (d) multi-word keyword forms / word-operators.
            if let Some((kind, end)) = match_word_forms(&chars, i) {
                out.push(Token::new(kind, &slice(&chars, i, end)));
                i = end;
                continue;
            }
        }

        // Numbers. sqlparse order: hex, exponent-float, dot-float, integer.
        if let Some((kind, end)) = match_number(&chars, i) {
            out.push(Token::new(kind, &slice(&chars, i, end)));
            i = end;
            continue;
        }

        // Single-quoted string: '(''|\\'|[^'])*'
        if c == '\'' {
            if let Some(end) = match_single_quoted(&chars, i) {
                out.push(Token::new(TokenKind::StringSingle, &slice(&chars, i, end)));
                i = end;
                continue;
            }
            out.push(Token::new(TokenKind::Error, "'"));
            i += 1;
            continue;
        }

        // Double-quoted symbol: "(""|\\"|[^"])*"
        if c == '"' {
            if let Some(end) = match_double_quoted(&chars, i) {
                out.push(Token::new(TokenKind::StringSymbol, &slice(&chars, i, end)));
                i = end;
                continue;
            }
            out.push(Token::new(TokenKind::Error, "\""));
            i += 1;
            continue;
        }

        // Generic word (`\w[$#\w]*`) that did not name-start (e.g. leading
        // digit like `1_000`, or `$foo`). sqlparse still classifies via keyword
        // lookup; a leading digit that failed the number regexes ends up here.
        if is_word_char(c) {
            let start = i;
            i += 1;
            while i < n && is_word_char(chars[i]) {
                i += 1;
            }
            let word = slice(&chars, start, i);
            out.push(Token::new(word_kind(&word), &word));
            continue;
        }

        // Punctuation: [;:()\[\],\.]
        if matches!(c, ';' | ':' | '(' | ')' | '[' | ']' | ',' | '.') {
            out.push(Token::new(TokenKind::Punctuation, &c.to_string()));
            i += 1;
            continue;
        }

        // Comparison operators: [<>=~!]+
        if matches!(c, '<' | '>' | '=' | '~' | '!') {
            let start = i;
            while i < n && matches!(chars[i], '<' | '>' | '=' | '~' | '!') {
                i += 1;
            }
            out.push(Token::new(
                TokenKind::OperatorComparison,
                &slice(&chars, start, i),
            ));
            continue;
        }

        // Other operators: [+/@#%^&|^-]+
        if matches!(c, '+' | '/' | '@' | '#' | '%' | '^' | '&' | '|' | '-') {
            let start = i;
            while i < n
                && matches!(
                    chars[i],
                    '+' | '/' | '@' | '#' | '%' | '^' | '&' | '|' | '-'
                )
            {
                i += 1;
            }
            out.push(Token::new(TokenKind::Operator, &slice(&chars, start, i)));
            continue;
        }

        // Anything else sqlparse leaves as Error (it never matches a rule).
        out.push(Token::new(TokenKind::Error, &c.to_string()));
        i += 1;
    }

    out
}

fn slice(chars: &[char], start: usize, end: usize) -> String {
    chars[start..end].iter().collect()
}

/// A `\w` character (Python `re.UNICODE`): letter, digit, or `_`. Excludes the
/// `$#` that sqlparse's *generic* word rule additionally allows.
fn is_re_word(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// End index of a `[A-ZÀ-Ü]\w*` word (the name-start char is at `start`).
fn word_end(chars: &[char], start: usize) -> usize {
    let mut j = start + 1;
    while j < chars.len() && is_re_word(chars[j]) {
        j += 1;
    }
    j
}

/// `(CASE|IN|VALUES|USING|FROM|AS)\b`: returns the word end if `start` begins
/// one of the six always-keyword words.
fn match_forced_keyword(chars: &[char], start: usize) -> Option<usize> {
    for kw in ["VALUES", "USING", "CASE", "FROM", "IN", "AS"] {
        if word_ci(chars, start, kw) && word_boundary_after(chars, start + kw.len()) {
            return Some(start + kw.len());
        }
    }
    None
}

/// `[A-ZÀ-Ü]\w*(?=\s*\.)`: true if the word at `start` is followed by optional
/// whitespace and then a `.`.
fn word_before_dot(chars: &[char], start: usize) -> bool {
    let end = word_end(chars, start);
    let mut j = end;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    j < chars.len() && chars[j] == '.'
}

/// Map a plain `\w[$#\w]*` word to a token kind via the keyword table.
fn word_kind(word: &str) -> TokenKind {
    match keyword_kind(word) {
        KeywordKind::Keyword => TokenKind::Keyword,
        KeywordKind::KeywordOrder => TokenKind::KeywordOrder,
        KeywordKind::NameBuiltin => TokenKind::NameBuiltin,
        KeywordKind::Operator => TokenKind::Operator,
        KeywordKind::Name => TokenKind::Name,
    }
}

/// Match backtick/acute-quoted name starting at `start` with quote char `q`.
/// Returns the exclusive end index, or None if unterminated.
fn match_quoted_name(chars: &[char], start: usize, q: char) -> Option<usize> {
    let n = chars.len();
    let mut i = start + 1;
    while i < n {
        if chars[i] == q {
            // Doubled quote (`` inside) is an escaped quote.
            if i + 1 < n && chars[i + 1] == q {
                i += 2;
                continue;
            }
            return Some(i + 1);
        }
        i += 1;
    }
    None
}

fn match_single_quoted(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut i = start + 1;
    while i < n {
        match chars[i] {
            '\\' if i + 1 < n => {
                // \\' or \\" backslash-escape
                i += 2;
            }
            '\'' => {
                if i + 1 < n && chars[i + 1] == '\'' {
                    i += 2; // '' escaped quote
                    continue;
                }
                return Some(i + 1);
            }
            _ => i += 1,
        }
    }
    None
}

fn match_double_quoted(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut i = start + 1;
    while i < n {
        match chars[i] {
            '\\' if i + 1 < n => i += 2,
            '"' => {
                if i + 1 < n && chars[i + 1] == '"' {
                    i += 2;
                    continue;
                }
                return Some(i + 1);
            }
            _ => i += 1,
        }
    }
    None
}

/// Handle the sqlparse regexes that begin with a name-start character and take
/// priority over the generic word rule: multi-word keyword forms and the
/// word-operators. Returns `(kind, end)` when one matches, else None to fall
/// back to number/generic-word handling.
fn match_word_forms(chars: &[char], start: usize) -> Option<(TokenKind, usize)> {
    // `NOT NULL\b` and `NOT <ws> LIKE/ILIKE/RLIKE/REGEXP` take precedence over
    // lexing `NOT` as a bare keyword.
    if word_ci(chars, start, "NOT") && word_boundary_after(chars, start + 3) {
        // NOT NULL (allowing an internal whitespace run, `\s+`)
        if let Some(after_ws) = skip_ws(chars, start + 3) {
            if after_ws > start + 3 && word_ci(chars, after_ws, "NULL") {
                let end = after_ws + 4;
                if word_boundary_after(chars, end) {
                    // sqlparse keeps the original text (incl. internal spaces).
                    return Some((TokenKind::Keyword, end));
                }
            }
            // NOT LIKE / NOT ILIKE / NOT RLIKE / NOT REGEXP
            for op in ["ILIKE", "RLIKE", "LIKE", "REGEXP"] {
                if after_ws > start + 3 && word_ci(chars, after_ws, op) {
                    let end = after_ws + op.len();
                    if word_boundary_after(chars, end) {
                        return Some((TokenKind::OperatorComparison, end));
                    }
                }
            }
        }
    }

    // ASC / DESC with optional NULLS FIRST/LAST.
    for kw in ["ASC", "DESC"] {
        if word_ci(chars, start, kw) && word_boundary_after(chars, start + kw.len()) {
            let mut end = start + kw.len();
            if let Some(after_ws) = skip_ws(chars, end) {
                if after_ws > end && word_ci(chars, after_ws, "NULLS") {
                    if let Some(after_ws2) = skip_ws(chars, after_ws + 5) {
                        for pos in ["FIRST", "LAST"] {
                            if after_ws2 > after_ws + 5
                                && word_ci(chars, after_ws2, pos)
                                && word_boundary_after(chars, after_ws2 + pos.len())
                            {
                                end = after_ws2 + pos.len();
                            }
                        }
                    }
                }
            }
            return Some((TokenKind::KeywordOrder, end));
        }
    }

    // NULLS FIRST/LAST as a standalone order keyword.
    if word_ci(chars, start, "NULLS") && word_boundary_after(chars, start + 5) {
        if let Some(after_ws) = skip_ws(chars, start + 5) {
            for pos in ["FIRST", "LAST"] {
                if after_ws > start + 5
                    && word_ci(chars, after_ws, pos)
                    && word_boundary_after(chars, after_ws + pos.len())
                {
                    return Some((TokenKind::KeywordOrder, after_ws + pos.len()));
                }
            }
        }
    }

    // Word-operators without NOT: LIKE / ILIKE / RLIKE / REGEXP(+BINARY).
    for op in ["ILIKE", "RLIKE", "LIKE"] {
        if word_ci(chars, start, op) && word_boundary_after(chars, start + op.len()) {
            return Some((TokenKind::OperatorComparison, start + op.len()));
        }
    }
    if word_ci(chars, start, "REGEXP") && word_boundary_after(chars, start + 6) {
        // optional `\s+BINARY`
        let mut end = start + 6;
        if let Some(after_ws) = skip_ws(chars, end) {
            if after_ws > end && word_ci(chars, after_ws, "BINARY") {
                let b_end = after_ws + 6;
                if word_boundary_after(chars, b_end) {
                    end = b_end;
                }
            }
        }
        return Some((TokenKind::OperatorComparison, end));
    }

    None
}

/// True if `chars[start..]` begins with `word` (ASCII case-insensitive).
fn word_ci(chars: &[char], start: usize, word: &str) -> bool {
    let wch: Vec<char> = word.chars().collect();
    if start + wch.len() > chars.len() {
        return false;
    }
    for (k, wc) in wch.iter().enumerate() {
        if !chars[start + k].eq_ignore_ascii_case(wc) {
            return false;
        }
    }
    true
}

/// `\b`: the char at `pos` (if any) must not be a word char.
fn word_boundary_after(chars: &[char], pos: usize) -> bool {
    pos >= chars.len() || !is_word_char(chars[pos])
}

/// Skip a run of whitespace; returns the index after it (== start if none).
fn skip_ws(chars: &[char], start: usize) -> Option<usize> {
    let mut i = start;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    Some(i)
}

/// Match a number literal, mirroring sqlparse's four number regexes in order.
fn match_number(chars: &[char], start: usize) -> Option<(TokenKind, usize)> {
    let n = chars.len();
    let c = chars[start];
    let neg = c == '-';
    let first = if neg { start + 1 } else { start };
    if first >= n {
        return None;
    }

    // Hexadecimal: -?0x[\dA-F]+  (case-insensitive on A-F)
    if chars[first] == '0' && first + 1 < n && (chars[first + 1] == 'x' || chars[first + 1] == 'X')
    {
        let mut j = first + 2;
        while j < n
            && (chars[j].is_ascii_digit() || matches!(chars[j].to_ascii_uppercase(), 'A'..='F'))
        {
            j += 1;
        }
        if j > first + 2 {
            return Some((TokenKind::Hexadecimal, j));
        }
    }

    // Exponent float: -?\d+(\.\d+)?E-?\d+
    if let Some(end) = match_exponent_float(chars, first) {
        return Some((TokenKind::Float, end));
    }

    // The dot/int regexes have negative lookbehind/ahead `(?![_A-ZÀ-Ü])`
    // asserting the char before `-`/digit and after the number is not a word
    // char. `start-1` boundary is guaranteed by the caller (we only reach here
    // after emitting the previous token). Check the trailing boundary.

    // Dot float: -?(\d+(\.\d*)|\.\d+)
    if let Some(end) = match_dot_float(chars, first) {
        if !word_follows(chars, end) {
            return Some((TokenKind::Float, end));
        }
    }

    // Integer: -?\d+
    if chars[first].is_ascii_digit() {
        let mut j = first;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        if !word_follows(chars, j) {
            return Some((TokenKind::Integer, j));
        }
    }

    None
}

/// `(?![_A-ZÀ-Ü])` trailing lookahead: fails if a word char follows.
fn word_follows(chars: &[char], pos: usize) -> bool {
    pos < chars.len() && (chars[pos] == '_' || chars[pos].is_alphabetic())
}

fn match_exponent_float(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = start;
    let d0 = j;
    while j < n && chars[j].is_ascii_digit() {
        j += 1;
    }
    if j == d0 {
        return None; // needs \d+ before optional fraction
    }
    if j < n && chars[j] == '.' {
        let dot = j;
        j += 1;
        let f0 = j;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        if j == f0 {
            j = dot; // no fraction digits; back up (fraction is required here)
        }
    }
    if j < n && (chars[j] == 'e' || chars[j] == 'E') {
        let mut k = j + 1;
        if k < n && chars[k] == '-' {
            k += 1;
        }
        let e0 = k;
        while k < n && chars[k].is_ascii_digit() {
            k += 1;
        }
        if k > e0 {
            return Some(k);
        }
    }
    None
}

fn match_dot_float(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    // \d+\.\d*
    if chars[start].is_ascii_digit() {
        let mut j = start;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        if j < n && chars[j] == '.' {
            j += 1;
            while j < n && chars[j].is_ascii_digit() {
                j += 1;
            }
            return Some(j);
        }
        return None;
    }
    // \.\d+
    if chars[start] == '.' && start + 1 < n && chars[start + 1].is_ascii_digit() {
        let mut j = start + 1;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        return Some(j);
    }
    None
}
