//! Build script for `mlflow-proto` (T1.2).
//!
//! Two outputs, both written to `OUT_DIR` (nothing is checked in):
//!
//! 1. **prost types** for every MLflow service proto + the OTLP protos needed
//!    by the `/v1/traces` handler, compiled with `protox` (pure Rust, no system
//!    `protoc`).
//! 2. **A generated route table** (`routes_generated.rs`) built by decoding the
//!    custom `databricks.rpc` `MethodOptions` extension (field 51310, defined in
//!    `databricks.proto`) off the `FileDescriptorSet` via `prost-reflect`. Each
//!    RPC carries one or more `HttpEndpoint`s (method / path / `since` version);
//!    we emit them RAW (proto-level path + version) so `lib.rs` can expand them
//!    into the concrete `/api/...` + `/ajax-api/...` Flask paths exactly like
//!    `mlflow/server/handlers.py::_get_paths` does.

use std::path::{Path, PathBuf};

use prost::Message;
use prost_reflect::{DescriptorPool, Value};
use prost_types::FileDescriptorSet;
use protox::Compiler;

/// Full name of the `databricks.rpc` extension on
/// `google.protobuf.MethodOptions` (declared in `mlflow/protos/databricks.proto`,
/// package `mlflow`, field number 51310).
const DATABRICKS_RPC_EXTENSION: &str = "mlflow.rpc";

/// Service protos we compile and scan for `databricks.rpc` route options.
/// Order mirrors `get_endpoints()` in `handlers.py` so a byte-diff of the two
/// route dumps is easy to reason about.
const ROUTE_SERVICE_PROTOS: &[&str] = &[
    "service.proto",
    "model_registry.proto",
    "mlflow_artifacts.proto",
    "webhooks.proto",
];

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = find_repo_root(&manifest_dir);
    let protos_dir = repo_root.join("mlflow").join("protos");
    let vendor_dir = manifest_dir.join("vendor");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Files we compile. Keep the Part II entity protos explicit even though
    // `service.proto` imports them: they are public `mlflow-proto` inputs in
    // their own right, and listing them makes Cargo rebuild when any one changes.
    // Imports such as jobs, scalapb, and the OTLP trace proto are still resolved
    // transitively; the registry / artifacts / webhooks protos come explicitly.
    let proto_files: Vec<PathBuf> = [
        "service.proto",
        "model_registry.proto",
        "webhooks.proto",
        "assessments.proto",
        "databricks.proto",
        "mlflow_artifacts.proto",
        "datasets.proto",
        "issues.proto",
        "label_schemas.proto",
        "review_queues.proto",
        "prompt_optimization.proto",
    ]
    .iter()
    .map(|f| protos_dir.join(f))
    .chain(std::iter::once(
        // OTLP trace-service proto is vendored (not shipped in the wheel).
        vendor_dir.join("opentelemetry/proto/collector/trace/v1/trace_service.proto"),
    ))
    .collect();

    // Include paths: repo `mlflow/protos` (so `scalapb/scalapb.proto`,
    // `opentelemetry/proto/...`, etc. resolve) plus our vendor dir for the
    // OTLP collector proto.
    let includes = [protos_dir.clone(), vendor_dir.clone()];

    // Re-run when any input proto changes.
    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    // 1. Compile with protox (pure Rust, no system `protoc`).
    //
    // NB: we drive the `Compiler` directly rather than `protox::compile`. The
    // convenience wrapper returns a `prost_types::FileDescriptorSet` built from
    // the typed `FileDescriptorProto`s, which SILENTLY DROPS custom extension
    // options (the `databricks.rpc` route metadata lives in the extension range
    // of `MethodOptions`, which prost-types cannot hold). `Compiler` keeps a
    // `prost-reflect` `DescriptorPool` that preserves those extensions, and its
    // `encode_file_descriptor_set()` re-encodes them into the wire bytes.
    let compiler = {
        let mut c = Compiler::new(includes).expect("failed to create protox compiler");
        c.include_source_info(true).include_imports(true);
        c.open_files(&proto_files)
            .expect("failed to open protos with protox");
        c
    };

    // 2. Generate prost Rust types. Decode from the extension-preserving bytes
    //    so prost-build sees a complete descriptor set (no protoc run).
    let fds_bytes = compiler.encode_file_descriptor_set();
    let file_descriptor_set = FileDescriptorSet::decode(fds_bytes.as_slice())
        .expect("failed to decode FileDescriptorSet");
    prost_build::compile_fds(file_descriptor_set)
        .expect("prost-build failed to generate Rust types");

    // 2b. Persist the extension-preserving FileDescriptorSet bytes so the crate
    //     can rebuild a runtime `prost-reflect` `DescriptorPool` (used by the
    //     MLflow-compatible JSON codec in `src/json.rs`, T1.3). We embed these
    //     bytes via `include_bytes!` rather than re-running protox at runtime.
    std::fs::write(out_dir.join("file_descriptor_set.bin"), &fds_bytes)
        .expect("failed to write file_descriptor_set.bin");

    // 3. Decode the databricks.rpc route options and emit routes_generated.rs.
    let pool = compiler.descriptor_pool();
    let routes = extract_routes(&pool);
    write_route_table(&out_dir, &routes);
}

/// A single raw endpoint as declared by a `databricks.rpc` `HttpEndpoint`.
struct RawRoute {
    service: String,
    method: String,
    http_method: String,
    path: String,
    since_major: i32,
    since_minor: i32,
}

fn extract_routes(pool: &DescriptorPool) -> Vec<RawRoute> {
    // Extension descriptor for databricks.rpc on MethodOptions.
    let rpc_ext = pool
        .get_extension_by_name(DATABRICKS_RPC_EXTENSION)
        .expect("databricks.rpc extension (mlflow.rpc / 51310) not found in descriptor pool");

    let mut routes = Vec::new();
    for service in pool.services() {
        // Only scan the services that back HTTP routes.
        if !ROUTE_SERVICE_PROTOS.contains(&service.parent_file().name()) {
            continue;
        }
        let service_name = service.name().to_string();
        for method in service.methods() {
            let options = method.options();
            let ext_value = options.get_extension(&rpc_ext);
            let rpc_msg = match ext_value.as_ref() {
                Value::Message(m) => m,
                _ => continue,
            };
            // `DatabricksRpcOptions.endpoints` is field 1 (repeated HttpEndpoint).
            let endpoints = match rpc_msg.get_field_by_name("endpoints") {
                Some(v) => v,
                None => continue,
            };
            let list = match endpoints.as_ref() {
                Value::List(l) => l,
                _ => continue,
            };
            for ep in list {
                let ep_msg = match ep {
                    Value::Message(m) => m,
                    _ => continue,
                };
                let http_method = field_string(ep_msg, "method").unwrap_or_else(|| "POST".into());
                let path = field_string(ep_msg, "path").unwrap_or_default();
                let (since_major, since_minor) = field_since(ep_msg);
                routes.push(RawRoute {
                    service: service_name.clone(),
                    method: method.name().to_string(),
                    http_method,
                    path,
                    since_major,
                    since_minor,
                });
            }
        }
    }
    routes
}

fn field_string(msg: &prost_reflect::DynamicMessage, name: &str) -> Option<String> {
    match msg.get_field_by_name(name)?.as_ref() {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn field_since(ep_msg: &prost_reflect::DynamicMessage) -> (i32, i32) {
    // `HttpEndpoint.since` is an ApiVersion { major, minor }. `since.major`
    // drives the URL version component in Python; default to 2 when unset,
    // matching proto2 default-numeric semantics used by the endpoints (all
    // MLflow endpoints declare an explicit `since`, but be defensive).
    let Some(since) = ep_msg.get_field_by_name("since") else {
        return (2, 0);
    };
    let Value::Message(since_msg) = since.as_ref() else {
        return (2, 0);
    };
    let major = field_i32(since_msg, "major").unwrap_or(2);
    let minor = field_i32(since_msg, "minor").unwrap_or(0);
    (major, minor)
}

fn field_i32(msg: &prost_reflect::DynamicMessage, name: &str) -> Option<i32> {
    match msg.get_field_by_name(name)?.as_ref() {
        Value::I32(v) => Some(*v),
        Value::I64(v) => Some(*v as i32),
        _ => None,
    }
}

fn write_route_table(out_dir: &Path, routes: &[RawRoute]) {
    let mut src = String::new();
    src.push_str("// @generated by mlflow-proto/build.rs — do not edit.\n");
    src.push_str("// Raw proto-level route table decoded from the databricks.rpc\n");
    src.push_str("// MethodOptions extension. Expand via RouteSpec::expand().\n\n");
    src.push_str("pub static ROUTE_TABLE: &[RouteSpec] = &[\n");
    for r in routes {
        src.push_str(&format!(
            "    RouteSpec {{ service: {:?}, method: {:?}, http_method: {:?}, path: {:?}, since_major: {}, since_minor: {} }},\n",
            r.service, r.method, r.http_method, r.path, r.since_major, r.since_minor
        ));
    }
    src.push_str("];\n");

    std::fs::write(out_dir.join("routes_generated.rs"), src)
        .expect("failed to write routes_generated.rs");
}

/// Walk upward from the crate manifest dir to find the repo root (the dir that
/// contains `mlflow/protos`). Robust to the crate being built from any cwd.
fn find_repo_root(manifest_dir: &Path) -> PathBuf {
    let mut dir = manifest_dir;
    loop {
        if dir
            .join("mlflow")
            .join("protos")
            .join("service.proto")
            .exists()
        {
            return dir.to_path_buf();
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => panic!(
                "could not locate repo root (mlflow/protos/service.proto) walking up from {}",
                manifest_dir.display()
            ),
        }
    }
}
