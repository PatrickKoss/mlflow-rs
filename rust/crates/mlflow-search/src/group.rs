//! sqlparse-compatible token grouping.
//!
//! Ports the subset of `sqlparse.engine.grouping.group()` that the MLflow
//! filter/order_by grammar observes. MLflow's parser inspects the grouped tree
//! (`isinstance(t, Comparison/Identifier/Parenthesis)`, `token.ttype`,
//! `token.value`), so we must reproduce the *same* nesting sqlparse produces.
//!
//! The grouping passes and their order match `grouping.group()` (only the
//! passes reachable from this grammar are implemented):
//!
//! 1. `group_parenthesis` — bracket matching `( ... )`
//! 2. `group_period` — `<name> . <name>` → Identifier
//! 3. `group_identifier` — bare `Name`/`String.Symbol` → Identifier
//! 4. `group_order` — `<Identifier|Number> <ASC|DESC>` → Identifier
//! 5. `group_comparison` — `<x> <cmp-op> <y>` → Comparison
//! 6. `group_aliased` — `<Identifier|Number> <Identifier>` → Identifier
//!    (this is what absorbs a trailing token, e.g. `foo bar`)
//! 7. `group_identifier_list` — comma-separated → IdentifierList
//!
//! Each pass recurses into already-formed subgroups exactly like sqlparse's
//! `@recurse` / `_group(recurse=True)`.

use crate::token::{Token, TokenKind};

/// A node in the grouped parse tree: either a flat token or a group with a
/// class (mirroring `sqlparse.sql.*` classes) and children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Leaf(Token),
    Group {
        class: GroupClass,
        children: Vec<Node>,
    },
}

/// The `sqlparse.sql` group classes this grammar cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupClass {
    Identifier,
    Comparison,
    Parenthesis,
    IdentifierList,
}

impl Node {
    fn leaf(kind: TokenKind, value: &str) -> Node {
        Node::Leaf(Token {
            kind,
            value: value.to_string(),
        })
    }

    pub fn is_whitespace(&self) -> bool {
        matches!(self, Node::Leaf(t) if t.kind.is_whitespace())
    }

    pub fn class(&self) -> Option<GroupClass> {
        match self {
            Node::Group { class, .. } => Some(*class),
            Node::Leaf(_) => None,
        }
    }

    /// `token.ttype` for a leaf; `None` for a group (matching sqlparse where
    /// grouped tokens have `ttype == None`).
    pub fn ttype(&self) -> Option<TokenKind> {
        match self {
            Node::Leaf(t) => Some(t.kind),
            Node::Group { .. } => None,
        }
    }

    /// Reconstructs `token.value` (the original source slice), like sqlparse's
    /// `str(token)` / `token.value` on a group (concatenation of children).
    pub fn value(&self) -> String {
        match self {
            Node::Leaf(t) => t.value.clone(),
            Node::Group { children, .. } => children.iter().map(Node::value).collect(),
        }
    }

    fn children(&self) -> &[Node] {
        match self {
            Node::Group { children, .. } => children,
            Node::Leaf(_) => &[],
        }
    }
}

/// A statement is the top-level list of nodes (sqlparse's `Statement.tokens`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statement {
    pub tokens: Vec<Node>,
}

/// Group a flat token stream into a [`Statement`], applying the sqlparse passes
/// in `grouping.group()` order (only the passes reachable from this grammar).
pub fn parse_statement(tokens: Vec<Token>) -> Statement {
    let mut nodes: Vec<Node> = tokens.into_iter().map(Node::Leaf).collect();
    group_parenthesis(&mut nodes);
    recurse_then(&mut nodes, &group_period);
    recurse_then(&mut nodes, &group_identifier);
    recurse_then(&mut nodes, &group_order);
    recurse_then(&mut nodes, &group_comparison);
    recurse_then(&mut nodes, &group_aliased);
    recurse_then(&mut nodes, &group_identifier_list);
    Statement { tokens: nodes }
}

/// Recurse into every subgroup, then apply `f` to this level — mirroring
/// sqlparse's `@recurse` / `_group(recurse=True)` (children first, then self).
fn recurse_then<F: Fn(&mut Vec<Node>)>(nodes: &mut Vec<Node>, f: &F) {
    for node in nodes.iter_mut() {
        if let Node::Group { children, .. } = node {
            recurse_then(children, f);
        }
    }
    f(nodes);
}

fn group_parenthesis(nodes: &mut Vec<Node>) {
    // Bracket matching: turn `(` ... `)` runs into Parenthesis groups, honoring
    // nesting. Mirrors `_group_matching(tlist, Parenthesis)`.
    let mut result: Vec<Node> = Vec::new();
    let mut stack: Vec<Vec<Node>> = Vec::new();

    for node in nodes.drain(..) {
        match &node {
            Node::Leaf(t) if t.kind == TokenKind::Punctuation && t.value == "(" => {
                stack.push(Vec::new());
                // push the '(' as the first child of the new open group
                stack.last_mut().unwrap().push(node);
            }
            Node::Leaf(t) if t.kind == TokenKind::Punctuation && t.value == ")" => {
                if let Some(mut children) = stack.pop() {
                    children.push(node);
                    let grp = Node::Group {
                        class: GroupClass::Parenthesis,
                        children,
                    };
                    match stack.last_mut() {
                        Some(parent) => parent.push(grp),
                        None => result.push(grp),
                    }
                } else {
                    // Unmatched ')': sqlparse leaves it ungrouped.
                    result.push(node);
                }
            }
            _ => match stack.last_mut() {
                Some(top) => top.push(node),
                None => result.push(node),
            },
        }
    }

    // Any unclosed '(' groups: sqlparse leaves their contents ungrouped, flat.
    // Flush the stack from outermost to innermost, in original order.
    if !stack.is_empty() {
        let mut leftover: Vec<Node> = Vec::new();
        for frame in stack.drain(..) {
            leftover.extend(frame);
        }
        result.extend(leftover);
    }

    *nodes = result;
}

/// `group_period`: `<prev> . <next?>` → Identifier. `valid_prev` is a Name /
/// String.Symbol / Identifier. Unlike the generic matcher, sqlparse's
/// `group_period` uses `valid_next = True` (always groups when prev is valid)
/// and a custom `post`: it includes the next token only if it is a valid
/// identifier-part (Name/Symbol/Wildcard/String.Single/Function/SquareBrackets);
/// otherwise it groups just `[prev ..= dot]` (so `params.` → Identifier
/// `params.`).
fn group_period(nodes: &mut Vec<Node>) {
    let valid_prev = |n: &Node| {
        matches!(n.class(), Some(GroupClass::Identifier))
            || matches!(
                n.ttype(),
                Some(TokenKind::Name) | Some(TokenKind::StringSymbol)
            )
    };
    let valid_next = |n: Option<&Node>| match n {
        None => false,
        Some(node) => matches!(
            node.ttype(),
            Some(TokenKind::Name)
                | Some(TokenKind::StringSymbol)
                | Some(TokenKind::Wildcard)
                | Some(TokenKind::StringSingle)
        ),
    };
    let is_dot =
        |n: &Node| matches!(n, Node::Leaf(t) if t.kind == TokenKind::Punctuation && t.value == ".");

    let mut i = 0;
    let mut prev_idx: Option<usize> = None;
    while i < nodes.len() {
        if nodes[i].is_whitespace() {
            i += 1;
            continue;
        }
        if is_dot(&nodes[i]) && prev_idx.map(|p| valid_prev(&nodes[p])).unwrap_or(false) {
            let from = prev_idx.unwrap();
            let nidx = next_non_ws(nodes, i);
            let include_next = valid_next(nidx.map(|n| &nodes[n]));
            let to = if include_next { nidx.unwrap() } else { i };
            let drained: Vec<Node> = nodes.drain(from..=to).collect();
            let children = flatten_extend(GroupClass::Identifier, drained);
            nodes.insert(
                from,
                Node::Group {
                    class: GroupClass::Identifier,
                    children,
                },
            );
            prev_idx = Some(from);
            i = from + 1;
            continue;
        }
        prev_idx = Some(i);
        i += 1;
    }
}

/// `group_identifier`: wrap each bare Name / String.Symbol leaf into Identifier.
// Takes `&mut Vec` (not `&mut [_]`) for a uniform signature with the other
// resizing passes driven by `recurse_then`.
#[allow(clippy::ptr_arg)]
fn group_identifier(nodes: &mut Vec<Node>) {
    for node in nodes.iter_mut() {
        let wrap = matches!(
            node.ttype(),
            Some(TokenKind::Name) | Some(TokenKind::StringSymbol)
        );
        if wrap {
            let leaf = std::mem::replace(node, Node::leaf(TokenKind::Name, ""));
            *node = Node::Group {
                class: GroupClass::Identifier,
                children: vec![leaf],
            };
        }
    }
}

/// `group_order`: `<Identifier|Number> <ws?> <ASC|DESC>` → Identifier.
fn group_order(nodes: &mut Vec<Node>) {
    let mut i = 0;
    while i < nodes.len() {
        let is_order = matches!(nodes[i].ttype(), Some(TokenKind::KeywordOrder));
        if is_order {
            // find previous non-whitespace
            if let Some(pidx) = prev_non_ws(nodes, i) {
                let prev_ok = matches!(nodes[pidx].class(), Some(GroupClass::Identifier))
                    || matches!(
                        nodes[pidx].ttype(),
                        Some(TokenKind::Integer) | Some(TokenKind::Float)
                    );
                if prev_ok {
                    let drained: Vec<Node> = nodes.drain(pidx..=i).collect();
                    nodes.insert(
                        pidx,
                        Node::Group {
                            class: GroupClass::Identifier,
                            children: drained,
                        },
                    );
                    i = pidx + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// `group_comparison`: `<x> <cmp-op> <y>` → Comparison.
fn group_comparison(nodes: &mut Vec<Node>) {
    let valid = |n: Option<&Node>| match n {
        None => false,
        Some(node) => {
            if matches!(
                node.class(),
                Some(GroupClass::Parenthesis) | Some(GroupClass::Identifier)
            ) {
                return true;
            }
            // sqlparse T_NUMERICAL + T_STRING + T_NAME (note: NOT Hexadecimal,
            // NOT Name.Builtin — so bare `timestamp <op> n` never groups).
            match node.ttype() {
                Some(
                    TokenKind::Integer
                    | TokenKind::Float
                    | TokenKind::StringSingle
                    | TokenKind::StringSymbol
                    | TokenKind::Name,
                ) => true,
                // `token.is_keyword and token.normalized == 'NULL'`
                Some(TokenKind::Keyword) => {
                    matches!(node, Node::Leaf(t) if t.value.eq_ignore_ascii_case("null"))
                }
                _ => false,
            }
        }
    };
    let is_match = |n: &Node| matches!(n.ttype(), Some(TokenKind::OperatorComparison));
    generic_group(
        nodes,
        GroupClass::Comparison,
        &is_match,
        &|p| valid(Some(p)),
        &valid,
        false,
    );
}

/// `group_aliased`: `<Identifier|Number> <ws?> <Identifier>` → Identifier
/// (extend). This absorbs a trailing identifier (e.g. `foo bar`,
/// `metrics.foo > 1 extra`).
fn group_aliased(nodes: &mut Vec<Node>) {
    let mut i = 0;
    while i < nodes.len() {
        let is_target = matches!(
            nodes[i].class(),
            Some(GroupClass::Parenthesis)
                | Some(GroupClass::Identifier)
                | Some(GroupClass::Comparison)
        ) || matches!(
            nodes[i].ttype(),
            Some(TokenKind::Integer) | Some(TokenKind::Float) | Some(TokenKind::Hexadecimal)
        );
        if is_target {
            if let Some(nidx) = next_non_ws(nodes, i) {
                if matches!(nodes[nidx].class(), Some(GroupClass::Identifier)) {
                    // group [i ..= nidx] into an Identifier (extend=True).
                    let drained: Vec<Node> = nodes.drain(i..=nidx).collect();
                    let children = flatten_extend(GroupClass::Identifier, drained);
                    nodes.insert(
                        i,
                        Node::Group {
                            class: GroupClass::Identifier,
                            children,
                        },
                    );
                    // stay at i to allow further chaining (sqlparse advances)
                    i += 1;
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// `group_identifier_list`: comma-joined items → IdentifierList.
fn group_identifier_list(nodes: &mut Vec<Node>) {
    let valid = |n: Option<&Node>| match n {
        None => false,
        Some(node) => {
            if matches!(
                node.class(),
                Some(GroupClass::Identifier)
                    | Some(GroupClass::Comparison)
                    | Some(GroupClass::IdentifierList)
            ) {
                return true;
            }
            // T_NUMERICAL + T_STRING + T_NAME + (Keyword, Comment, Wildcard).
            matches!(
                node.ttype(),
                Some(
                    TokenKind::Integer
                        | TokenKind::Float
                        | TokenKind::StringSingle
                        | TokenKind::StringSymbol
                        | TokenKind::Name
                        | TokenKind::Keyword
                        | TokenKind::Wildcard
                )
            )
        }
    };
    let is_match =
        |n: &Node| matches!(n, Node::Leaf(t) if t.kind == TokenKind::Punctuation && t.value == ",");
    generic_group(
        nodes,
        GroupClass::IdentifierList,
        &is_match,
        &|p| valid(Some(p)),
        &valid,
        true,
    );
}

/// Generic middle-token grouping, mirroring sqlparse's `_group(...)`:
/// when `is_match(token)` and `valid_prev(prev)` and `valid_next(next)`, group
/// `[prev_idx ..= next_idx]` into `class`. `extend` merges a same-class prev.
fn generic_group(
    nodes: &mut Vec<Node>,
    class: GroupClass,
    is_match: &dyn Fn(&Node) -> bool,
    valid_prev: &dyn Fn(&Node) -> bool,
    valid_next: &dyn Fn(Option<&Node>) -> bool,
    extend: bool,
) {
    let mut i = 0;
    // Track the previous non-whitespace node index as we scan.
    let mut prev_idx: Option<usize> = None;
    while i < nodes.len() {
        if nodes[i].is_whitespace() {
            i += 1;
            continue;
        }
        if is_match(&nodes[i]) {
            let nidx = next_non_ws(nodes, i);
            let next_ref = nidx.map(|n| &nodes[n]);
            let prev_ok = prev_idx.map(|p| valid_prev(&nodes[p])).unwrap_or(false);
            if prev_ok && valid_next(next_ref) {
                let from = prev_idx.unwrap();
                let to = nidx.unwrap();
                let drained: Vec<Node> = nodes.drain(from..=to).collect();
                let children = if extend {
                    flatten_extend(class, drained)
                } else {
                    drained
                };
                nodes.insert(from, Node::Group { class, children });
                prev_idx = Some(from);
                i = from + 1;
                continue;
            }
        }
        prev_idx = Some(i);
        i += 1;
    }
}

/// For `extend=True` grouping: if the leading node is already the same class,
/// splice its children in (sqlparse's `group_tokens(extend=True)`).
fn flatten_extend(class: GroupClass, drained: Vec<Node>) -> Vec<Node> {
    let mut it = drained.into_iter();
    let mut children = Vec::new();
    if let Some(first) = it.next() {
        match first {
            Node::Group {
                class: c,
                children: inner,
            } if c == class => children.extend(inner),
            other => children.push(other),
        }
    }
    children.extend(it);
    children
}

fn prev_non_ws(nodes: &[Node], idx: usize) -> Option<usize> {
    (0..idx).rev().find(|&j| !nodes[j].is_whitespace())
}

fn next_non_ws(nodes: &[Node], idx: usize) -> Option<usize> {
    ((idx + 1)..nodes.len()).find(|&j| !nodes[j].is_whitespace())
}

/// Expose children of a group for the parser.
pub fn group_children(node: &Node) -> &[Node] {
    node.children()
}
