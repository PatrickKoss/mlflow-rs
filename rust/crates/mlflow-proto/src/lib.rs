//! `mlflow-proto`: generated protobuf types and the generated HTTP route table.
//!
//! Per `RUST_TRACKING_SERVER_PLAN.md` (§2.3, §4, Phase 1 T1.2/T1.3), this crate
//! is the single source of truth for message types compiled from
//! `service.proto`, `model_registry.proto`, `webhooks.proto`,
//! `assessments.proto`, `databricks.proto`, `mlflow_artifacts.proto`, and the
//! OTLP protos (via `prost`/`protox`, no system `protoc` dependency).
//!
//! It also exposes [`ROUTE_TABLE`], a build-time-generated list of every
//! proto-backed HTTP endpoint (decoded from the `databricks.rpc` `MethodOptions`
//! extension), plus [`RouteSpec::expand`] which turns a raw proto route into the
//! concrete `/api/...` + `/ajax-api/...` Flask paths, mirroring
//! `mlflow/server/handlers.py::_get_paths`.

/// Generated protobuf message types, grouped by proto package.
//
// The `include!`d modules are prost-generated and carry proto comments verbatim
// as rustdoc. A few of those comments have irregular list indentation (from
// `service.proto` etc.) that trips `clippy::doc_overindented_list_items`. We keep
// the proto docs (they are useful) and scope-allow the lint on the generated
// modules only.
#[allow(clippy::doc_overindented_list_items)]
pub mod mlflow {
    include!(concat!(env!("OUT_DIR"), "/mlflow.rs"));

    pub mod artifacts {
        include!(concat!(env!("OUT_DIR"), "/mlflow.artifacts.rs"));
    }
    pub mod assessments {
        include!(concat!(env!("OUT_DIR"), "/mlflow.assessments.rs"));
    }
    pub mod datasets {
        include!(concat!(env!("OUT_DIR"), "/mlflow.datasets.rs"));
    }
    pub mod issues {
        include!(concat!(env!("OUT_DIR"), "/mlflow.issues.rs"));
    }
    pub mod label_schemas {
        include!(concat!(env!("OUT_DIR"), "/mlflow.label_schemas.rs"));
    }
    pub mod review_queues {
        include!(concat!(env!("OUT_DIR"), "/mlflow.review_queues.rs"));
    }
}

/// OpenTelemetry protobuf types (OTLP), used by the `/v1/traces` handler.
#[allow(clippy::doc_overindented_list_items)]
pub mod opentelemetry {
    pub mod proto {
        pub mod common {
            pub mod v1 {
                include!(concat!(
                    env!("OUT_DIR"),
                    "/opentelemetry.proto.common.v1.rs"
                ));
            }
        }
        pub mod resource {
            pub mod v1 {
                include!(concat!(
                    env!("OUT_DIR"),
                    "/opentelemetry.proto.resource.v1.rs"
                ));
            }
        }
        pub mod trace {
            pub mod v1 {
                include!(concat!(env!("OUT_DIR"), "/opentelemetry.proto.trace.v1.rs"));
            }
        }
        pub mod collector {
            pub mod trace {
                pub mod v1 {
                    include!(concat!(
                        env!("OUT_DIR"),
                        "/opentelemetry.proto.collector.trace.v1.rs"
                    ));
                }
            }
        }
    }
}

/// `scalapb` options types (referenced by the MLflow protos' imports).
#[allow(clippy::doc_overindented_list_items)]
pub mod scalapb {
    include!(concat!(env!("OUT_DIR"), "/scalapb.rs"));
}

mod json;
pub use json::{
    dynamic_from_mlflow_json, dynamic_from_query_pairs, from_mlflow_json, from_query_pairs,
    to_mlflow_json, JsonCodecError,
};

mod routes;
pub use routes::{ExpandedRoute, RouteSpec, ROUTE_TABLE};
