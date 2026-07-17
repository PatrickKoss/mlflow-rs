//! Real-socket integration test: binds an ephemeral port, serves the app
//! with `axum::serve` + graceful shutdown (exactly as `main.rs` does),
//! issues a plain hyper HTTP/1.1 request to `/health` over TCP, then
//! triggers and verifies graceful shutdown.

use std::time::Duration;

use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper::Request;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use mlflow_server::{build_app, ServerConfig};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[tokio::test]
async fn serves_health_over_real_socket_and_shuts_down_gracefully() {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_prefix: None,
        backend_store_uri: None,
        default_artifact_root: None,
        serve_artifacts: true,
        artifacts_destination: None,
        allowed_hosts: None,
        cors_allowed_origins: None,
        x_frame_options: "SAMEORIGIN".to_string(),
    };
    let app = build_app(&config);

    let listener = TcpListener::bind((config.host.as_str(), config.port))
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("server error");
    });

    // Give the accept loop a moment to start; poll instead of a fixed sleep
    // to keep this fast and non-flaky.
    let client = Client::builder(TokioExecutor::new()).build_http();
    let url: hyper::Uri = format!("http://{addr}/health").parse().unwrap();

    let mut last_err = None;
    let mut response = None;
    for _ in 0..50 {
        match client
            .request(
                Request::builder()
                    .uri(url.clone())
                    .body(Empty::<Bytes>::new())
                    .unwrap(),
            )
            .await
        {
            Ok(res) => {
                response = Some(res);
                break;
            }
            Err(err) => {
                last_err = Some(err);
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }

    let response = response
        .unwrap_or_else(|| panic!("failed to connect to server after retries: {last_err:?}"));

    assert_eq!(response.status(), hyper::StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"OK");

    // Trigger graceful shutdown and make sure the server task actually
    // stops within a reasonable timeout.
    shutdown_tx.send(()).expect("send shutdown signal");
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("server did not shut down in time")
        .expect("server task panicked");
}
