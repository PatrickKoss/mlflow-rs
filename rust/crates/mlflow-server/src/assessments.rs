//! Assessment endpoints (plan T4.4, §3.9): create, get, update, delete under
//! `/mlflow/traces/{trace_id}/assessments...`.
//!
//! Each handler mirrors its Python counterpart in `mlflow/server/handlers.py`
//! (`_create_assessment`, `_get_assessment`, `_update_assessment`,
//! `_delete_assessment`), translating the wire proto (`mlflow.assessments.*`,
//! `mlflow.{Create,Get,Update,Delete}Assessment`) to/from the
//! [`mlflow_store`] entity types the T2.12 store layer already implements
//! (create/get/update/delete, override/supersede, feedback/expectation JSON
//! encoding).
//!
//! ## Path parameters
//!
//! All four routes carry `{trace_id}` (three also `{assessment_id}`) as REST
//! path segments, using the same [`crate::proto_http::parse_request_with_path_params`]
//! mechanism as [`crate::logged_models`]. `createAssessment` is the odd one
//! out: its proto path is `/mlflow/traces/{assessment.trace_id}/assessments` —
//! a *nested* field, not a top-level one — so instead of merging into the
//! request JSON (which only supports top-level overlays), this mirrors
//! Python's handler literally: parse `assessment` from the body, then
//! unconditionally overwrite `assessment.trace_id` with the URL segment
//! (`assessment.trace_id = trace_id`, `handlers.py:4351`), regardless of what
//! the body contained.
//!
//! ## `google.protobuf.Value` wire encoding
//!
//! `Feedback.value`/`Expectation.value` are `google.protobuf.Value` — MLflow's
//! JSON codec ([`mlflow_proto::json`]) collapses these to their bare JSON
//! scalar/object/array form (T4.4 added this; see that module's
//! `well_known_to_json`), matching Python's `MessageToDict`/`ParseDict`
//! round-trip byte-for-byte (including the surprising `4` -> `4.0` widening,
//! since `NumberValue` is always a double). The store already speaks this
//! same "one JSON string" representation (`AssessmentValue::{Expectation,
//! Feedback}.value_json`), so the handlers here only need to convert between
//! `google.protobuf.Value` and `serde_json::Value` — a message <-> JSON-text
//! transcode, no manual oneof walking.
//!
//! ## FieldMask -> store update translation (the T2.12 completion note's
//! deferred item)
//!
//! `updateAssessment`'s `update_mask` allows paths `assessment_name`,
//! `expectation`, `feedback`, `rationale`, `metadata`, `valid`
//! (`handlers.py:4391-4403`). Five of the six map onto
//! [`mlflow_store::AssessmentUpdate`] fields one-to-one. `valid` does not: **the
//! store's `update_assessment` has no `valid` parameter in Python either**
//! (`sqlalchemy_store.py:4522-4531` — `valid` is only ever flipped by the
//! override/delete machinery, never by `update_assessment`). Python's handler
//! still unconditionally builds `kwargs["valid"] = assessment_proto.valid` for
//! that path (`handlers.py:4402-4403`) and passes it to
//! `_get_tracking_store().update_assessment(trace_id=..., assessment_id=...,
//! **kwargs)` — an uncaught `TypeError: update_assessment() got an unexpected
//! keyword argument 'valid'` that `catch_mlflow_exception` does NOT catch
//! (`handlers.py:1166-1183` only catches `MlflowException`), surfacing as a
//! bare Flask 500. This is the same class of "genuine Python bug reproduced
//! faithfully" as T2.3's search-utils deviations (see the plan's T2.3
//! completion note) — [`update_assessment`] reproduces the *observable*
//! behavior (an internal-error 500) via [`MlflowError::internal_error`]
//! rather than inventing store-level `valid` semantics Python doesn't have.
//! Flagged for the Phase 12 differential allowlist (Python's body is Flask's
//! HTML error page, not a JSON MLflow error — a wire-format nuance not worth
//! byte-matching for what is, on the real backend, always a server bug).

use std::collections::BTreeMap;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::MlflowError;
use mlflow_proto::mlflow as pb;
use mlflow_proto::mlflow::assessments as apb;
use mlflow_store::{
    Assessment, AssessmentError, AssessmentSource, AssessmentUpdate, AssessmentValue,
    FeedbackUpdate, NewAssessment,
};

use crate::proto_http::{parse_request, parse_request_with_path_params, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

/// Metadata key Python's `Assessment.__post_init__` reads to populate
/// `run_id` when the client didn't set it directly (`AssessmentMetadataKey.
/// SOURCE_RUN_ID`, `mlflow/tracing/constant.py:160`). The proto has no
/// `run_id` field at all — this is the only way it ever reaches the store.
const SOURCE_RUN_ID_METADATA_KEY: &str = "mlflow.assessment.sourceRunId";

/// `_create_assessment` (`handlers.py:4338`), path: `POST
/// /mlflow/traces/{assessment.trace_id}/assessments`.
pub async fn create_assessment(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<std::collections::HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateAssessment = parse_request(&parts, &body, "mlflow.CreateAssessment")?;
    let assessment_proto = req.assessment.ok_or_else(|| missing_param("assessment"))?;

    // Python: `assessment.trace_id = trace_id` unconditionally overwrites
    // whatever the body carried, using the URL segment (see module docs).
    let trace_id = path_params.get("trace_id").cloned().unwrap_or_default();

    let new_assessment = new_assessment_from_proto(assessment_proto, trace_id)?;

    let created = state
        .tracking_store()
        .create_assessment(workspace.name(), new_assessment)
        .await?;

    let resp = pb::create_assessment::Response {
        assessment: Some(to_proto_assessment(created)),
    };
    proto_response(&resp, "mlflow.CreateAssessment.Response")
}

/// `_get_assessment` (`handlers.py:4360`), path: `GET
/// /mlflow/traces/{trace_id}/assessments/{assessment_id}`. Python takes both
/// segments purely as URL path params (no body schema).
pub async fn get_assessment(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<std::collections::HashMap<String, String>>,
) -> Result<Response, MlflowError> {
    let trace_id = path_params.get("trace_id").cloned().unwrap_or_default();
    let assessment_id = path_params
        .get("assessment_id")
        .cloned()
        .unwrap_or_default();

    let assessment = state
        .tracking_store()
        .get_assessment(workspace.name(), &trace_id, &assessment_id)
        .await?;

    let resp = pb::get_assessment_request::Response {
        assessment: Some(to_proto_assessment(assessment)),
    };
    proto_response(&resp, "mlflow.GetAssessmentRequest.Response")
}

/// `_update_assessment` (`handlers.py:4373`), path: `PATCH
/// /mlflow/traces/{trace_id}/assessments/{assessment_id}`.
pub async fn update_assessment(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<std::collections::HashMap<String, String>>,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateAssessment = parse_request_with_path_params(
        &parts,
        &body,
        "mlflow.UpdateAssessment",
        &path_param_pairs(&path_params, &["trace_id", "assessment_id"]),
    )?;
    let assessment_proto = req.assessment.ok_or_else(|| missing_param("assessment"))?;
    let update_mask = req
        .update_mask
        .ok_or_else(|| missing_param("update_mask"))?;

    let trace_id = path_params.get("trace_id").cloned().unwrap_or_default();
    let assessment_id = path_params
        .get("assessment_id")
        .cloned()
        .unwrap_or_default();

    let mut update = AssessmentUpdate::default();
    for path in &update_mask.paths {
        match path.as_str() {
            "assessment_name" => {
                update.name = Some(assessment_proto.assessment_name.clone().unwrap_or_default());
            }
            "expectation" => {
                update.expectation_value_json = Some(expectation_value_json(&assessment_proto)?);
            }
            "feedback" => {
                update.feedback = Some(feedback_update(&assessment_proto)?);
            }
            "rationale" => {
                update.rationale = Some(assessment_proto.rationale.clone().unwrap_or_default());
            }
            "metadata" => {
                update.metadata = Some(
                    assessment_proto
                        .metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );
            }
            // Python's store-level `update_assessment` has no `valid`
            // parameter; the handler nonetheless builds `kwargs["valid"]` for
            // this path and passes it through, which raises an uncaught
            // `TypeError` (not an `MlflowException`) that surfaces as a bare
            // Flask 500. See the module docs for why we reproduce that
            // observable behavior (an internal error) rather than adding
            // store-level `valid` semantics Python doesn't have.
            "valid" => {
                return Err(MlflowError::internal_error(
                    "update_assessment() got an unexpected keyword argument 'valid'",
                ));
            }
            // Unknown paths are silently ignored, matching Python's `for path
            // in update_mask.paths: if path == ...: ... ` chain, which has no
            // `else` branch.
            _ => {}
        }
    }

    let updated = state
        .tracking_store()
        .update_assessment(workspace.name(), &trace_id, &assessment_id, update)
        .await?;

    let resp = pb::update_assessment::Response {
        assessment: Some(to_proto_assessment(updated)),
    };
    proto_response(&resp, "mlflow.UpdateAssessment.Response")
}

/// `_delete_assessment` (`handlers.py:4415`), path: `DELETE
/// /mlflow/traces/{trace_id}/assessments/{assessment_id}`. Python takes both
/// segments purely as URL path params (no body schema).
pub async fn delete_assessment(
    State(state): State<AppState>,
    workspace: Workspace,
    Path(path_params): Path<std::collections::HashMap<String, String>>,
) -> Result<Response, MlflowError> {
    let trace_id = path_params.get("trace_id").cloned().unwrap_or_default();
    let assessment_id = path_params
        .get("assessment_id")
        .cloned()
        .unwrap_or_default();

    state
        .tracking_store()
        .delete_assessment(workspace.name(), &trace_id, &assessment_id)
        .await?;

    proto_response(
        &pb::delete_assessment::Response {},
        "mlflow.DeleteAssessment.Response",
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn missing_param(param: &str) -> MlflowError {
    MlflowError::invalid_parameter_value(format!(
        "Missing value for required parameter '{param}'. \
         See the API docs for more information about request parameters."
    ))
}

/// Build the `path_params` overlay slice for
/// [`parse_request_with_path_params`] from the axum-captured path segments
/// (see [`crate::logged_models::path_param_pairs`] for the general pattern;
/// duplicated locally to keep modules independent).
fn path_param_pairs(
    path_params: &std::collections::HashMap<String, String>,
    names: &[&'static str],
) -> Vec<(&'static str, String)> {
    names
        .iter()
        .filter_map(|name| path_params.get(*name).map(|v| (*name, v.clone())))
        .collect()
}

/// `Assessment.from_proto` + `assessment.trace_id = trace_id`
/// (`handlers.py:4350-4351`), producing the store's [`NewAssessment`] input.
///
/// Mirrors `Assessment.__post_init__`'s "exactly one of expectation/feedback/
/// issue" dispatch (`assessment.py:83-88`, surfaced via `Assessment.from_proto`'s
/// `WhichOneof` dispatch, `assessment.py:157-168`) and its `run_id`-from-metadata
/// extraction (`assessment.py:114-120`).
fn new_assessment_from_proto(
    proto: apb::Assessment,
    trace_id: String,
) -> Result<NewAssessment, MlflowError> {
    let name = proto.assessment_name.clone().unwrap_or_default();
    let source = source_from_proto(proto.source.as_ref())?;
    let span_id = proto.span_id.clone().filter(|s| !s.is_empty());
    let rationale = proto.rationale.clone().filter(|s| !s.is_empty());
    let metadata = metadata_from_proto(&proto.metadata);
    let overrides = proto.overrides.clone().filter(|s| !s.is_empty());

    let (create_time_ms, last_update_time_ms) = (
        proto.create_time.as_ref().map(timestamp_to_millis),
        proto.last_update_time.as_ref().map(timestamp_to_millis),
    );

    let run_id = metadata
        .as_ref()
        .and_then(|m| m.get(SOURCE_RUN_ID_METADATA_KEY))
        .cloned();

    let value = assessment_value_from_proto(&proto)?;

    Ok(NewAssessment {
        trace_id,
        name,
        value,
        source,
        run_id,
        span_id,
        rationale,
        metadata,
        create_time_ms,
        last_update_time_ms,
        assessment_id: proto.assessment_id.clone().filter(|s| !s.is_empty()),
        overrides,
    })
}

/// `Assessment.from_proto`'s `WhichOneof("value")` dispatch (`assessment.py:157-168`):
/// exactly one of `expectation`/`feedback`/`issue` must be set, else
/// `"Unknown assessment type: {WhichOneof result}"` (`None` when unset).
fn assessment_value_from_proto(proto: &apb::Assessment) -> Result<AssessmentValue, MlflowError> {
    match &proto.value {
        Some(apb::assessment::Value::Expectation(e)) => Ok(AssessmentValue::Expectation {
            value_json: expectation_proto_to_value_json(e)?,
        }),
        Some(apb::assessment::Value::Feedback(f)) => {
            let value_json = value_message_to_json_string(f.value.as_ref())?;
            let error = f.error.as_ref().map(|e| AssessmentError {
                error_code: e.error_code.clone().unwrap_or_default(),
                error_message: e.error_message.clone().filter(|s| !s.is_empty()),
                stack_trace: e.stack_trace.clone().filter(|s| !s.is_empty()),
            });
            Ok(AssessmentValue::Feedback { value_json, error })
        }
        Some(apb::assessment::Value::Issue(i)) => Ok(AssessmentValue::Issue {
            issue_name: i.issue_name.clone().unwrap_or_default(),
        }),
        None => Err(MlflowError::invalid_parameter_value(
            "Unknown assessment type: None",
        )),
    }
}

/// `ExpectationValue.from_proto`/`to_proto` (`assessment.py:629-645`): a plain
/// `value` (`google.protobuf.Value`) or a `serialized_value` (opaque JSON
/// string under a `serialization_format`, only `"JSON_FORMAT"` supported).
/// Returns the store's `value_json` string either way.
fn expectation_proto_to_value_json(e: &apb::Expectation) -> Result<String, MlflowError> {
    if let Some(serialized) = &e.serialized_value {
        let format = serialized.serialization_format.as_deref().unwrap_or("");
        if format != "JSON_FORMAT" {
            return Err(MlflowError::invalid_parameter_value(format!(
                "Unknown serialization format: {format}. Only JSON_FORMAT is supported."
            )));
        }
        // Validate it's well-formed JSON (Python's `json.loads` would raise
        // otherwise) but store the original text (round-trips byte-for-byte).
        let text = serialized.value.clone().unwrap_or_default();
        serde_json::from_str::<serde_json::Value>(&text).map_err(|e| {
            MlflowError::invalid_parameter_value(format!(
                "Failed to parse serialized expectation value as JSON: {e}"
            ))
        })?;
        return Ok(text);
    }
    value_message_to_json_string(e.value.as_ref())
}

/// `FeedbackValue.from_proto(proto.feedback)` (`assessment.py:4396-4397`,
/// `692-697`) — note this is `proto.feedback`, a *direct field access* on the
/// whole `Assessment` message, not gated on `WhichOneof("value") ==
/// "feedback"`. Protobuf auto-vivifies an unset submessage field to its
/// all-defaults instance on access, so Python builds this from a
/// default-valued `Feedback{}` (`value=None`, no error) whenever the client's
/// payload didn't actually set the `feedback` oneof variant (e.g. it set
/// `expectation` instead) — it does **not** raise here. The real type check
/// happens later, store-side, against the *existing* assessment's type
/// (`sqlalchemy_store.py:4569-4572`, mirrored by the Rust store's
/// `update_assessment`). Rust's `oneof` can only hold one variant at a time
/// (no auto-vivified sibling to read), so this falls back to an
/// all-defaults `Feedback` in that case — the same value Python's
/// auto-vivification would have produced.
fn feedback_update(proto: &apb::Assessment) -> Result<FeedbackUpdate, MlflowError> {
    static EMPTY: apb::Feedback = apb::Feedback {
        value: None,
        error: None,
    };
    let f = match &proto.value {
        Some(apb::assessment::Value::Feedback(f)) => f,
        _ => &EMPTY,
    };
    let value_json = value_message_to_json_string(f.value.as_ref())?;
    let error = f.error.as_ref().map(|e| AssessmentError {
        error_code: e.error_code.clone().unwrap_or_default(),
        error_message: e.error_message.clone().filter(|s| !s.is_empty()),
        stack_trace: e.stack_trace.clone().filter(|s| !s.is_empty()),
    });
    Ok(FeedbackUpdate { value_json, error })
}

/// `ExpectationValue.from_proto(proto.expectation)` for the `update_mask`'s
/// `expectation` path (`assessment.py:4394-4395`) — same auto-vivification
/// caveat as [`feedback_update`]: Python reads `proto.expectation` directly,
/// regardless of which oneof variant the payload actually set.
fn expectation_value_json(proto: &apb::Assessment) -> Result<String, MlflowError> {
    static EMPTY: apb::Expectation = apb::Expectation {
        value: None,
        serialized_value: None,
    };
    let e = match &proto.value {
        Some(apb::assessment::Value::Expectation(e)) => e,
        _ => &EMPTY,
    };
    expectation_proto_to_value_json(e)
}

/// Transcode a `google.protobuf.Value` to the store's `value_json` string
/// representation (`json.dumps(MessageToDict(proto.value))`-equivalent — see
/// module docs). An absent `value` field transcodes the same as an
/// all-default `Value` message: JSON `null` (`MessageToJson(Value()) ==
/// "null"`, matching `ParseDict(None, Value())`'s round-trip for a
/// feedback/expectation value of `None`).
fn value_message_to_json_string(value: Option<&prost_types::Value>) -> Result<String, MlflowError> {
    let v = value.cloned().unwrap_or_default();
    let json = mlflow_proto::to_mlflow_json(&v, "google.protobuf.Value").map_err(|e| {
        MlflowError::internal_error(format!("Failed to encode assessment value: {e}"))
    })?;
    Ok(json)
}

/// The reverse of [`value_message_to_json_string`]: parse the store's
/// `value_json` string back into a `google.protobuf.Value` (`ParseDict(...,
/// Value())`-equivalent).
fn value_json_to_message(value_json: &str) -> prost_types::Value {
    mlflow_proto::from_mlflow_json::<prost_types::Value>(value_json, "google.protobuf.Value")
        .unwrap_or_default()
}

/// `AssessmentSource.from_proto`/`to_proto` (`assessment_source.py:87-99`).
/// `source_type` is required by the proto (`validate_required`) but that
/// extension isn't enforced by JSON parsing (`parse_dict` has no notion of
/// it) — an absent `source` (or absent `source_type`) parses as
/// `SOURCE_TYPE_UNSPECIFIED`, matching a default-constructed proto message,
/// same as Python.
fn source_from_proto(
    proto: Option<&apb::AssessmentSource>,
) -> Result<AssessmentSource, MlflowError> {
    let source_type_num = proto.and_then(|s| s.source_type).unwrap_or(0);
    let source_type = apb::assessment_source::SourceType::try_from(source_type_num)
        .map(|t| t.as_str_name().to_string())
        .map_err(|_| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid assessment source type: {source_type_num}"
            ))
        })?;
    let source_id = proto
        .and_then(|s| s.source_id.clone())
        .filter(|s| !s.is_empty());
    Ok(AssessmentSource {
        source_type,
        source_id,
    })
}

/// `dict(proto.metadata) if proto.metadata else None` (`assessment.py:307`,
/// `445`, `562`): an empty metadata map is `None`, not `Some({})`.
fn metadata_from_proto(
    metadata: &std::collections::HashMap<String, String>,
) -> Option<BTreeMap<String, String>> {
    if metadata.is_empty() {
        return None;
    }
    Some(
        metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )
}

/// `proto.create_time.ToMilliseconds()` / `FromMilliseconds` — floor-division
/// seconds + `mod`-scaled nanos, matching protobuf's `Timestamp` millis
/// conversion (verified against `google.protobuf.Timestamp.FromMilliseconds`/
/// `ToMilliseconds` in Python).
fn timestamp_to_millis(ts: &prost_types::Timestamp) -> i64 {
    ts.seconds * 1000 + i64::from(ts.nanos) / 1_000_000
}

fn millis_to_timestamp(ms: i64) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: ms.div_euclid(1000),
        nanos: (ms.rem_euclid(1000) * 1_000_000) as i32,
    }
}

/// `Assessment.to_proto` (`assessment.py:122-155`): the store entity ->
/// response proto direction.
fn to_proto_assessment(a: Assessment) -> apb::Assessment {
    let value = match a.value {
        AssessmentValue::Expectation { value_json } => {
            apb::assessment::Value::Expectation(apb::Expectation {
                value: Some(value_json_to_message(&value_json)),
                serialized_value: None,
            })
        }
        AssessmentValue::Feedback { value_json, error } => {
            apb::assessment::Value::Feedback(apb::Feedback {
                value: Some(value_json_to_message(&value_json)),
                error: error.map(|e| apb::AssessmentError {
                    error_code: Some(e.error_code),
                    error_message: e.error_message,
                    stack_trace: e.stack_trace,
                }),
            })
        }
        AssessmentValue::Issue { issue_name } => {
            apb::assessment::Value::Issue(apb::IssueReference {
                issue_name: Some(issue_name),
            })
        }
    };

    let source_type = apb::assessment_source::SourceType::from_str_name(&a.source.source_type)
        .unwrap_or(apb::assessment_source::SourceType::Unspecified);

    apb::Assessment {
        assessment_id: Some(a.assessment_id),
        assessment_name: Some(a.name),
        trace_id: Some(a.trace_id),
        span_id: a.span_id,
        source: Some(apb::AssessmentSource {
            source_type: Some(source_type as i32),
            source_id: a.source.source_id,
        }),
        create_time: Some(millis_to_timestamp(a.create_time_ms)),
        last_update_time: Some(millis_to_timestamp(a.last_update_time_ms)),
        rationale: a.rationale,
        metadata: a.metadata.unwrap_or_default().into_iter().collect(),
        overrides: a.overrides,
        valid: Some(a.valid),
        value: Some(value),
        // `error` is `[Deprecated, use the `error` field in `feedback`
        // instead]` (`assessments.proto:134-136`); Python's `Assessment.to_proto`
        // never sets it either (`assessment.py:122-155`).
        ..Default::default()
    }
}
