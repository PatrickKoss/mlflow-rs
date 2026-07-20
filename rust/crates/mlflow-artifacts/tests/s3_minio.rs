//! MinIO-gated S3 repository integration coverage.
//!
//! Run with (the bucket must already exist):
//! `MLFLOW_TEST_S3_ENDPOINT=http://127.0.0.1:59090 MLFLOW_TEST_S3_BUCKET=mlflow-soak \
//!  MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:59090 AWS_ACCESS_KEY_ID=minioadmin \
//!  AWS_SECRET_ACCESS_KEY=minioadmin AWS_REGION=us-east-1 \
//!  cargo test -p mlflow-artifacts --features aws --test s3_minio`

use bytes::Bytes;
use futures::StreamExt;
use mlflow_artifacts::repo::MultipartUploadPart;
use mlflow_error::MlflowError;

fn body(chunks: Vec<Bytes>) -> futures::stream::BoxStream<'static, Result<Bytes, MlflowError>> {
    futures::stream::iter(chunks.into_iter().map(Ok)).boxed()
}

async fn download(repo: &dyn mlflow_artifacts::ArtifactRepo, path: &str) -> Vec<u8> {
    let mut stream = repo.get(path).await.unwrap().stream;
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    bytes
}

fn minio_uri(test: &str) -> Option<String> {
    std::env::var("MLFLOW_TEST_S3_ENDPOINT").ok()?;
    let bucket =
        std::env::var("MLFLOW_TEST_S3_BUCKET").unwrap_or_else(|_| "mlflow-soak".to_string());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    Some(format!(
        "s3://{bucket}/t22-0/{test}-{}-{nonce}",
        std::process::id()
    ))
}

#[tokio::test]
async fn put_get_list_delete_roundtrip() {
    let Some(uri) = minio_uri("roundtrip") else {
        eprintln!("skipped: MLFLOW_TEST_S3_ENDPOINT is unset");
        return;
    };
    let repo = mlflow_artifacts::factory::repo_from_uri(&uri).unwrap();
    repo.put(
        "dir/payload.bin",
        body(vec![
            Bytes::from_static(b"part-a"),
            Bytes::from_static(b"part-b"),
        ]),
    )
    .await
    .unwrap();
    assert_eq!(
        download(repo.as_ref(), "dir/payload.bin").await,
        b"part-apart-b"
    );
    let root = repo.list(None).await.unwrap();
    assert_eq!(root.len(), 1);
    assert_eq!(root[0].path, "dir");
    assert!(root[0].is_dir);
    let nested = repo.list(Some("dir")).await.unwrap();
    assert_eq!(nested.len(), 1);
    assert_eq!(nested[0].path, "dir/payload.bin");
    assert_eq!(nested[0].file_size, Some(12));
    let presigned = repo
        .get_download_presigned_url("dir/payload.bin", 300)
        .await
        .unwrap();
    assert_eq!(presigned.file_size, Some(12));
    assert!(presigned.headers.is_empty());
    let direct = reqwest::get(&presigned.url).await.unwrap();
    assert!(direct.status().is_success());
    assert_eq!(direct.bytes().await.unwrap().as_ref(), b"part-apart-b");
    repo.delete("dir").await.unwrap();
    assert!(repo.list(None).await.unwrap().is_empty());
}

#[tokio::test]
async fn multipart_complete_and_abort() {
    let Some(uri) = minio_uri("multipart") else {
        eprintln!("skipped: MLFLOW_TEST_S3_ENDPOINT is unset");
        return;
    };
    let repo = mlflow_artifacts::factory::repo_from_uri(&uri).unwrap();
    let created = repo
        .create_multipart_upload("complete.bin", 2)
        .await
        .unwrap();
    assert_eq!(created.credentials.len(), 2);
    let client = reqwest::Client::new();
    let payloads = [vec![b'a'; 5 * 1024 * 1024], b"tail".to_vec()];
    let mut parts = Vec::new();
    for (credential, payload) in created.credentials.iter().zip(payloads.iter()) {
        let response = client
            .put(&credential.url)
            .body(payload.clone())
            .send()
            .await
            .unwrap();
        assert!(
            response.status().is_success(),
            "{}",
            response.text().await.unwrap()
        );
        let etag = response
            .headers()
            .get(reqwest::header::ETAG)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        parts.push(MultipartUploadPart {
            part_number: credential.part_number,
            etag,
            url: credential.url.clone(),
        });
    }
    repo.complete_multipart_upload("complete.bin", &created.upload_id, &parts)
        .await
        .unwrap();
    let completed = download(repo.as_ref(), "complete.bin").await;
    assert_eq!(completed.len(), 5 * 1024 * 1024 + 4);
    assert!(completed[..5 * 1024 * 1024]
        .iter()
        .all(|byte| *byte == b'a'));
    assert_eq!(&completed[5 * 1024 * 1024..], b"tail");

    let aborted = repo
        .create_multipart_upload("aborted.bin", 1)
        .await
        .unwrap();
    let response = client
        .put(&aborted.credentials[0].url)
        .body("discard me")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    repo.abort_multipart_upload("aborted.bin", &aborted.upload_id)
        .await
        .unwrap();
    assert!(repo.get("aborted.bin").await.is_err());

    repo.delete("complete.bin").await.unwrap();
}
