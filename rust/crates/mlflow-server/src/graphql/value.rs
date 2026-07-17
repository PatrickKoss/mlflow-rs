//! `GqlVal` — the resolved-value tree the resolvers build and the executor
//! projects the query selection set onto.
//!
//! A plain `serde_json::Value` can't reproduce graphene's on-the-wire scalar
//! formatting: the `LongString` custom scalar serializes 64-bit ints as JSON
//! *strings* (`str(long)`, to dodge JS's 53-bit integer truncation), and `Float`
//! values go through Python's `json.dumps` float `repr` (so `5.0` stays `5.0`,
//! and NaN would be `NaN` — though the `MlflowMetricExtension.value` resolver
//! maps NaN to `None` upstream, matching `graphql_schema_extensions.py`). This
//! tagged value type carries that distinction and renders the final projected
//! result byte-for-byte like graphene's json output (compact separators — the
//! Flask view `jsonify`s the `{"data", "errors"}` dict).

use mlflow_proto::python_float_repr;

/// A resolved GraphQL value with graphene-faithful scalar semantics.
#[derive(Debug, Clone)]
pub enum GqlVal {
    Null,
    Bool(bool),
    /// A `String`/enum scalar (rendered as a quoted JSON string).
    Str(String),
    /// A `LongString` scalar: a 64-bit integer rendered as a quoted string
    /// (`str(long)`), matching `graphql_custom_scalars.LongString.serialize`.
    Long(i64),
    /// A `Float` scalar, rendered via Python's float `repr`
    /// ([`python_float_repr`]) so integer-valued floats keep their `.0`.
    Float(f64),
    /// An `Int` scalar (rendered as a bare integer).
    Int(i64),
    List(Vec<GqlVal>),
    /// An object. `tag` is an internal type tag used to resolve `__typename`
    /// (see [`crate::graphql::schema::type_name`]); `fields` are `(gql_field_name,
    /// value)` pairs in a stable order.
    Object {
        tag: &'static str,
        fields: Vec<(&'static str, GqlVal)>,
    },
}

impl GqlVal {
    /// Build an object value from a type tag and its fields.
    pub fn object(tag: &'static str, fields: Vec<(&'static str, GqlVal)>) -> Self {
        GqlVal::Object { tag, fields }
    }

    /// Look up a field on an object by its GraphQL name. Returns `None` for
    /// non-objects or absent fields (the caller decides how to handle it —
    /// `__typename` and missing-field cases are handled in the executor).
    pub fn get(&self, name: &str) -> Option<&GqlVal> {
        match self {
            GqlVal::Object { fields, .. } => {
                fields.iter().find(|(k, _)| *k == name).map(|(_, v)| v)
            }
            _ => None,
        }
    }

    /// Mutable field lookup on an object (used by the T9.6 auth gate to
    /// post-filter the `modelVersions` list in place). `None` for non-objects
    /// or absent fields.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut GqlVal> {
        match self {
            GqlVal::Object { fields, .. } => {
                fields.iter_mut().find(|(k, _)| *k == name).map(|(_, v)| v)
            }
            _ => None,
        }
    }

    /// The internal type tag for an object (for `__typename`), or `None`.
    pub fn tag(&self) -> Option<&'static str> {
        match self {
            GqlVal::Object { tag, .. } => Some(tag),
            _ => None,
        }
    }

    /// Serialize a leaf scalar / null to its JSON wire form (compact). Objects
    /// and lists are handled by the executor's projection, not here.
    pub fn write_scalar(&self, out: &mut String) {
        match self {
            GqlVal::Null => out.push_str("null"),
            GqlVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            GqlVal::Str(s) => out.push_str(&mlflow_proto::quote_json_string(s)),
            GqlVal::Long(n) => {
                // `LongString.serialize` → `str(long)`, then json-quoted.
                out.push('"');
                out.push_str(&n.to_string());
                out.push('"');
            }
            GqlVal::Float(f) => out.push_str(&python_float_repr(*f)),
            GqlVal::Int(n) => out.push_str(&n.to_string()),
            GqlVal::List(_) | GqlVal::Object { .. } => {
                // Never reached: the executor recurses into lists/objects itself.
                debug_assert!(false, "write_scalar called on a composite value");
            }
        }
    }
}

/// Convenience constructors used pervasively by the resolvers.
impl GqlVal {
    /// A nullable `String` from an `Option<String>` (`None` → GraphQL null).
    pub fn opt_str(v: Option<String>) -> Self {
        v.map(GqlVal::Str).unwrap_or(GqlVal::Null)
    }

    /// A non-null `String`.
    pub fn str(v: impl Into<String>) -> Self {
        GqlVal::Str(v.into())
    }
}
