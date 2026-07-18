//! Label-schema RPC handlers (T16.3).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::request::Parts;
use axum::response::Response;
use mlflow_error::{ErrorCode, MlflowError};
use mlflow_proto::mlflow::label_schemas::{self as pb, label_schema_input};
use mlflow_store::{LabelSchema, LabelSchemaInput, LabelSchemaType, LabelSchemaUpdate};

use crate::proto_http::{parse_request, proto_response};
use crate::state::AppState;
use crate::workspace::Workspace;

pub async fn create_label_schema(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::CreateLabelSchema =
        parse_request(&parts, &body, "mlflow.label_schemas.CreateLabelSchema")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let schema_type = schema_type(req.r#type)?;
    let input = input_from_proto(req.input.as_ref())?;
    let schema = state
        .tracking_store()
        .create_label_schema(
            workspace.name(),
            experiment_id,
            name,
            schema_type,
            &input,
            req.instruction.as_deref(),
            req.enable_comment.unwrap_or(false),
        )
        .await?;
    proto_response(
        &pb::create_label_schema::Response {
            label_schema: Some(to_proto(schema)),
        },
        "mlflow.label_schemas.CreateLabelSchema.Response",
    )
}

pub async fn get_label_schema(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetLabelSchema =
        parse_request(&parts, &body, "mlflow.label_schemas.GetLabelSchema")?;
    let schema_id = required(req.schema_id.as_deref(), "schema_id")?;
    let schema = state
        .tracking_store()
        .get_label_schema(workspace.name(), schema_id)
        .await?;
    proto_response(
        &pb::get_label_schema::Response {
            label_schema: Some(to_proto(schema)),
        },
        "mlflow.label_schemas.GetLabelSchema.Response",
    )
}

pub async fn get_label_schema_by_name(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::GetLabelSchemaByName =
        parse_request(&parts, &body, "mlflow.label_schemas.GetLabelSchemaByName")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let name = required(req.name.as_deref(), "name")?;
    let schema = state
        .tracking_store()
        .get_label_schema_by_name(workspace.name(), experiment_id, name)
        .await?;
    proto_response(
        &pb::get_label_schema_by_name::Response {
            label_schema: Some(to_proto(schema)),
        },
        "mlflow.label_schemas.GetLabelSchemaByName.Response",
    )
}

pub async fn list_label_schemas(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::ListLabelSchemas =
        parse_request(&parts, &body, "mlflow.label_schemas.ListLabelSchemas")?;
    let experiment_id = required(req.experiment_id.as_deref(), "experiment_id")?;
    let page = state
        .tracking_store()
        .list_label_schemas(
            workspace.name(),
            experiment_id,
            req.max_results.unwrap_or(100),
            req.page_token.as_deref().filter(|value| !value.is_empty()),
        )
        .await?;
    proto_response(
        &pb::list_label_schemas::Response {
            label_schemas: page.schemas.into_iter().map(to_proto).collect(),
            next_page_token: Some(page.next_page_token.unwrap_or_default()),
        },
        "mlflow.label_schemas.ListLabelSchemas.Response",
    )
}

pub async fn update_label_schema(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::UpdateLabelSchema =
        parse_request(&parts, &body, "mlflow.label_schemas.UpdateLabelSchema")?;
    let schema_id = required(req.schema_id.as_deref(), "schema_id")?;
    let input = req
        .input
        .as_ref()
        .map(|input| input_from_proto(Some(input)))
        .transpose()?;
    let schema = state
        .tracking_store()
        .update_label_schema(
            workspace.name(),
            schema_id,
            LabelSchemaUpdate {
                name: req.name.as_deref(),
                instruction: req.instruction.as_deref(),
                enable_comment: req.enable_comment,
                input: input.as_ref(),
            },
        )
        .await?;
    proto_response(
        &pb::update_label_schema::Response {
            label_schema: Some(to_proto(schema)),
        },
        "mlflow.label_schemas.UpdateLabelSchema.Response",
    )
}

pub async fn delete_label_schema(
    State(state): State<AppState>,
    workspace: Workspace,
    parts: Parts,
    body: Bytes,
) -> Result<Response, MlflowError> {
    let req: pb::DeleteLabelSchema =
        parse_request(&parts, &body, "mlflow.label_schemas.DeleteLabelSchema")?;
    let schema_id = required(req.schema_id.as_deref(), "schema_id")?;
    state
        .tracking_store()
        .delete_label_schema(workspace.name(), schema_id)
        .await?;
    proto_response(
        &pb::delete_label_schema::Response {},
        "mlflow.label_schemas.DeleteLabelSchema.Response",
    )
}

fn schema_type(value: Option<i32>) -> Result<LabelSchemaType, MlflowError> {
    match value.and_then(|value| pb::LabelSchemaType::try_from(value).ok()) {
        Some(pb::LabelSchemaType::Feedback) => Ok(LabelSchemaType::Feedback),
        Some(pb::LabelSchemaType::Expectation) => Ok(LabelSchemaType::Expectation),
        _ => Err(MlflowError::invalid_parameter_value(format!(
            "Label schema `type` must be one of FEEDBACK or EXPECTATION; got proto enum value {}.",
            value.unwrap_or_default()
        ))),
    }
}

fn input_from_proto(input: Option<&pb::LabelSchemaInput>) -> Result<LabelSchemaInput, MlflowError> {
    match input.and_then(|input| input.input.as_ref()) {
        Some(label_schema_input::Input::PassFail(input)) => Ok(LabelSchemaInput::PassFail {
            positive_label: input.positive_label.clone().unwrap_or_default(),
            negative_label: input.negative_label.clone().unwrap_or_default(),
        }),
        Some(label_schema_input::Input::Categorical(input)) => Ok(LabelSchemaInput::Categorical {
            options: input.options.clone(),
            multi_select: input.multi_select.unwrap_or(false),
        }),
        Some(label_schema_input::Input::Numeric(input)) => Ok(LabelSchemaInput::Numeric {
            min_value: input.min_value,
            max_value: input.max_value,
        }),
        Some(label_schema_input::Input::Text(input)) => Ok(LabelSchemaInput::Text {
            max_length: input.max_length,
        }),
        None => Err(MlflowError::invalid_parameter_value(
            "Label schema `input` must have exactly one of `pass_fail`, `categorical`, `numeric`, \
             or `text` set; got an empty oneof.",
        )),
    }
}

fn to_proto(schema: LabelSchema) -> pb::LabelSchema {
    let schema_type = match schema.schema_type {
        LabelSchemaType::Feedback => pb::LabelSchemaType::Feedback,
        LabelSchemaType::Expectation => pb::LabelSchemaType::Expectation,
    };
    let input = match schema.input {
        LabelSchemaInput::PassFail {
            positive_label,
            negative_label,
        } => label_schema_input::Input::PassFail(pb::InputPassFail {
            positive_label: Some(positive_label),
            negative_label: Some(negative_label),
        }),
        LabelSchemaInput::Categorical {
            options,
            multi_select,
        } => label_schema_input::Input::Categorical(pb::InputCategorical {
            options,
            multi_select: Some(multi_select),
        }),
        LabelSchemaInput::Numeric {
            min_value,
            max_value,
        } => label_schema_input::Input::Numeric(pb::InputNumeric {
            min_value,
            max_value,
        }),
        LabelSchemaInput::Text { max_length } => {
            label_schema_input::Input::Text(pb::InputText { max_length })
        }
    };
    pb::LabelSchema {
        schema_id: Some(schema.schema_id),
        experiment_id: Some(schema.experiment_id),
        name: Some(schema.name),
        r#type: Some(schema_type as i32),
        instruction: schema.instruction,
        enable_comment: Some(schema.enable_comment),
        input: Some(pb::LabelSchemaInput { input: Some(input) }),
        created_by: schema.created_by,
        created_at: Some(schema.created_at),
        last_updated_at: Some(schema.updated_at),
        is_default: Some(schema.is_default),
    }
}

fn required<'a>(value: Option<&'a str>, param: &str) -> Result<&'a str, MlflowError> {
    value
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing_required(param))
}

fn missing_required(param: &str) -> MlflowError {
    MlflowError::new(
        format!(
            "Missing value for required parameter '{param}'. See the API docs for more \
             information about request parameters."
        ),
        ErrorCode::InvalidParameterValue,
    )
}
