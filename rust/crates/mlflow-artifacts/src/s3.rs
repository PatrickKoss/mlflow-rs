//! S3-backed artifact repository and Python-compatible S3 configuration.
//!
//! `AmazonS3Builder::from_env` supplies the standard AWS credential chain and
//! `AWS_REGION` / `AWS_DEFAULT_REGION`. MLflow-specific settings are layered on
//! top: `MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_S3_IGNORE_TLS`, and
//! `MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE`. Python/botocore's `auto` addressing is
//! mapped to path style for a custom endpoint (MinIO and similar services), and
//! virtual-hosted style otherwise. Explicit `path` and `virtual` values always
//! win. Plain HTTP is enabled only when the custom endpoint itself uses HTTP.
//!
//! The generic object-store API handles ordinary artifact operations. S3's
//! externally uploaded multipart protocol is implemented here with the
//! credential provider exposed by `object_store` and a small SigV4 signer: the
//! create/complete/abort REST calls are header-signed and each UploadPart URL is
//! query-presigned for the same one-hour lifetime used by boto3.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use hmac::{Hmac, Mac};
use mlflow_error::{ErrorCode, MlflowError};
use object_store::aws::{AmazonS3, AmazonS3Builder, AwsCredential};
use object_store::path::Path as ObjPath;
use object_store::{ClientConfigKey, ObjectStoreExt};
use reqwest::{Method, StatusCode, Url};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::repo::{
    store_error, ArtifactDownload, ArtifactFileInfo, ArtifactRepo, CreateMultipartUploadResult,
    MultipartUploadCredential, MultipartUploadPart, ObjectStoreRepo, PresignedDownloadResult,
};

const PRESIGNED_PART_TTL: Duration = Duration::from_secs(3600);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressingStyle {
    Path,
    Virtual,
}

/// The MLflow-specific subset layered over `AmazonS3Builder::from_env`.
/// Kept separate so its mapping can be tested without mutating process-global
/// environment variables or making network requests.
#[derive(Debug, Clone, PartialEq, Eq)]
struct S3RuntimeConfig {
    bucket: String,
    root: ObjPath,
    endpoint: Option<String>,
    region: String,
    addressing_style: AddressingStyle,
    ignore_tls: bool,
}

impl S3RuntimeConfig {
    fn from_env(uri: &str) -> Result<Self, MlflowError> {
        Self::from_lookup(uri, |name| std::env::var(name).ok())
    }

    fn from_lookup(
        uri: &str,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, MlflowError> {
        let (bucket, root) = parse_s3_uri(uri)?;
        let endpoint = lookup("MLFLOW_S3_ENDPOINT_URL").filter(|value| !value.is_empty());
        if let Some(value) = &endpoint {
            let parsed = Url::parse(value).map_err(|e| {
                MlflowError::invalid_parameter_value(format!(
                    "Invalid MLFLOW_S3_ENDPOINT_URL '{value}': {e}"
                ))
            })?;
            if !matches!(parsed.scheme(), "http" | "https") {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "Invalid MLFLOW_S3_ENDPOINT_URL scheme '{}': expected http or https",
                    parsed.scheme()
                )));
            }
        }

        let style = lookup("MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "auto".to_string());
        let addressing_style = match style.to_ascii_lowercase().as_str() {
            "path" => AddressingStyle::Path,
            "virtual" => AddressingStyle::Virtual,
            "auto" if endpoint.is_some() => AddressingStyle::Path,
            "auto" => AddressingStyle::Virtual,
            _ => {
                return Err(MlflowError::invalid_parameter_value(format!(
                    "MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE must be one of 'auto', 'path', or \
                     'virtual', but got '{style}'"
                )));
            }
        };
        let ignore_tls = parse_python_bool("MLFLOW_S3_IGNORE_TLS", lookup("MLFLOW_S3_IGNORE_TLS"))?;
        let region = lookup("AWS_REGION")
            .or_else(|| lookup("AWS_DEFAULT_REGION"))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "us-east-1".to_string());

        Ok(Self {
            bucket,
            root,
            endpoint,
            region,
            addressing_style,
            ignore_tls,
        })
    }

    fn virtual_endpoint(&self) -> Result<Option<String>, MlflowError> {
        let Some(endpoint) = &self.endpoint else {
            return Ok(None);
        };
        if self.addressing_style == AddressingStyle::Path {
            return Ok(Some(endpoint.clone()));
        }
        let mut url = Url::parse(endpoint).map_err(|e| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid MLFLOW_S3_ENDPOINT_URL '{endpoint}': {e}"
            ))
        })?;
        let host = url.host_str().ok_or_else(|| {
            MlflowError::invalid_parameter_value(format!(
                "Invalid MLFLOW_S3_ENDPOINT_URL '{endpoint}': endpoint has no host"
            ))
        })?;
        url.set_host(Some(&format!("{}.{}", self.bucket, host)))
            .map_err(|_| {
                MlflowError::invalid_parameter_value(format!(
                    "Invalid bucket '{}' for virtual-hosted S3 addressing",
                    self.bucket
                ))
            })?;
        Ok(Some(url.to_string().trim_end_matches('/').to_string()))
    }

    fn object_store(&self) -> Result<AmazonS3, MlflowError> {
        let mut builder = AmazonS3Builder::from_env()
            .with_bucket_name(&self.bucket)
            .with_region(&self.region)
            .with_virtual_hosted_style_request(self.addressing_style == AddressingStyle::Virtual);
        if let Some(endpoint) = self.virtual_endpoint()? {
            builder = builder
                .with_endpoint(endpoint)
                .with_allow_http(self.endpoint.as_deref().is_some_and(is_http));
        }
        if self.ignore_tls {
            builder = builder.with_config(
                object_store::aws::AmazonS3ConfigKey::Client(
                    ClientConfigKey::AllowInvalidCertificates,
                ),
                "true",
            );
        }
        builder.build().map_err(store_error)
    }

    fn object_url(&self, path: &ObjPath) -> Result<Url, MlflowError> {
        let endpoint = if let Some(endpoint) = self.virtual_endpoint()? {
            if self.addressing_style == AddressingStyle::Path {
                format!("{}/{}", endpoint.trim_end_matches('/'), self.bucket)
            } else {
                endpoint
            }
        } else if self.addressing_style == AddressingStyle::Virtual {
            format!("https://{}.s3.{}.amazonaws.com", self.bucket, self.region)
        } else {
            format!("https://s3.{}.amazonaws.com/{}", self.region, self.bucket)
        };
        let endpoint = endpoint.trim_end_matches('/');
        let encoded_path = aws_uri_encode(path.as_ref(), false);
        let url = if encoded_path.is_empty() {
            format!("{endpoint}/")
        } else {
            format!("{endpoint}/{encoded_path}")
        };
        Url::parse(&url).map_err(|e| {
            MlflowError::internal_error(format!("Failed to construct S3 request URL: {e}"))
        })
    }
}

fn is_http(endpoint: &str) -> bool {
    endpoint
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
}

fn parse_python_bool(name: &str, value: Option<String>) -> Result<bool, MlflowError> {
    let Some(value) = value else {
        return Ok(false);
    };
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(MlflowError::invalid_parameter_value(format!(
            "{name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), but got {value}"
        ))),
    }
}

fn parse_s3_uri(uri: &str) -> Result<(String, ObjPath), MlflowError> {
    let rest = uri
        .strip_prefix("s3://")
        .ok_or_else(|| MlflowError::invalid_parameter_value(format!("Not an S3 URI: {uri}")))?;
    let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    if bucket.is_empty() || bucket.contains(['?', '#']) {
        return Err(MlflowError::invalid_parameter_value(format!(
            "Invalid S3 artifact URI '{uri}': bucket name is required"
        )));
    }
    let root = ObjPath::parse(prefix).map_err(|e| {
        MlflowError::invalid_parameter_value(format!(
            "Invalid S3 artifact URI path in '{uri}': {e}"
        ))
    })?;
    Ok((bucket.to_string(), root))
}

/// S3 repository with generic object operations plus externally uploaded MPU.
pub struct S3ArtifactRepo {
    inner: ObjectStoreRepo,
    s3: AmazonS3,
    config: S3RuntimeConfig,
    http: reqwest::Client,
}

impl S3ArtifactRepo {
    pub fn from_uri(uri: &str) -> Result<Self, MlflowError> {
        let config = S3RuntimeConfig::from_env(uri)?;
        let s3 = config.object_store()?;
        let http = reqwest::Client::builder()
            .danger_accept_invalid_certs(config.ignore_tls)
            .build()
            .map_err(|e| {
                MlflowError::internal_error(format!("Failed to build S3 HTTP client: {e}"))
            })?;
        let inner = ObjectStoreRepo::new(Arc::new(s3.clone()), config.root.clone());
        Ok(Self {
            inner,
            s3,
            config,
            http,
        })
    }

    fn full_path(&self, rel: &str) -> ObjPath {
        self.inner.full_path(rel)
    }

    async fn credential(&self) -> Result<Arc<AwsCredential>, MlflowError> {
        self.s3
            .credentials()
            .get_credential()
            .await
            .map_err(store_error)
    }

    async fn signed_request(
        &self,
        method: Method,
        path: &ObjPath,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Result<reqwest::Response, MlflowError> {
        let mut url = self.config.object_url(path)?;
        set_query(&mut url, query);
        let credential = self.credential().await?;
        let now = Utc::now();
        let headers =
            authorization_headers(&method, &url, &body, &credential, &self.config.region, now);
        let response = self
            .http
            .request(method, url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|e| MlflowError::internal_error(format!("S3 request failed: {e}")))?;
        if response.status().is_success() {
            return Ok(response);
        }
        Err(s3_response_error(response).await)
    }
}

#[async_trait::async_trait]
impl ArtifactRepo for S3ArtifactRepo {
    async fn get(&self, path: &str) -> Result<ArtifactDownload, MlflowError> {
        self.inner.get(path).await
    }

    async fn put(
        &self,
        path: &str,
        body: BoxStream<'static, Result<Bytes, MlflowError>>,
    ) -> Result<(), MlflowError> {
        self.inner.put(path, body).await
    }

    async fn list(&self, path: Option<&str>) -> Result<Vec<ArtifactFileInfo>, MlflowError> {
        self.inner.list(path).await
    }

    async fn delete(&self, path: &str) -> Result<(), MlflowError> {
        self.inner.delete(path).await
    }

    async fn create_multipart_upload(
        &self,
        path: &str,
        num_parts: i64,
    ) -> Result<CreateMultipartUploadResult, MlflowError> {
        let full = self.full_path(path);
        let response = self
            .signed_request(Method::POST, &full, &[("uploads", "")], Vec::new())
            .await?;
        let body = response.bytes().await.map_err(|e| {
            MlflowError::internal_error(format!("Failed to read S3 multipart response: {e}"))
        })?;
        let result: InitiateMultipartUploadResult = quick_xml::de::from_reader(body.as_ref())
            .map_err(|e| {
                MlflowError::internal_error(format!(
                    "Invalid S3 CreateMultipartUpload response: {e}"
                ))
            })?;

        let credential = self.credential().await?;
        let mut credentials = Vec::new();
        for part_number in 1..=num_parts {
            let url = presigned_url(
                Method::PUT,
                self.config.object_url(&full)?,
                &[
                    ("partNumber", part_number.to_string()),
                    ("uploadId", result.upload_id.clone()),
                ],
                &credential,
                &self.config.region,
                Utc::now(),
                PRESIGNED_PART_TTL,
            );
            credentials.push(MultipartUploadCredential {
                url,
                part_number,
                headers: Vec::new(),
            });
        }
        Ok(CreateMultipartUploadResult {
            upload_id: result.upload_id,
            credentials,
        })
    }

    async fn complete_multipart_upload(
        &self,
        path: &str,
        upload_id: &str,
        parts: &[MultipartUploadPart],
    ) -> Result<(), MlflowError> {
        let mut body = String::from("<CompleteMultipartUpload>");
        for part in parts {
            body.push_str("<Part><PartNumber>");
            body.push_str(&part.part_number.to_string());
            body.push_str("</PartNumber><ETag>");
            body.push_str(&xml_escape(&part.etag));
            body.push_str("</ETag></Part>");
        }
        body.push_str("</CompleteMultipartUpload>");
        self.signed_request(
            Method::POST,
            &self.full_path(path),
            &[("uploadId", upload_id)],
            body.into_bytes(),
        )
        .await?;
        Ok(())
    }

    async fn abort_multipart_upload(&self, path: &str, upload_id: &str) -> Result<(), MlflowError> {
        self.signed_request(
            Method::DELETE,
            &self.full_path(path),
            &[("uploadId", upload_id)],
            Vec::new(),
        )
        .await?;
        Ok(())
    }

    async fn get_download_presigned_url(
        &self,
        path: &str,
        expiration_seconds: u64,
    ) -> Result<PresignedDownloadResult, MlflowError> {
        let full = self.full_path(path);
        let meta = match self.s3.head(&full).await {
            Ok(meta) => meta,
            Err(object_store::Error::NotFound { .. }) => {
                return Err(MlflowError::resource_does_not_exist(
                    "The specified key does not exist.",
                ));
            }
            Err(error) => return Err(store_error(error)),
        };
        let credential = self.credential().await?;
        let url = presigned_url(
            Method::GET,
            self.config.object_url(&full)?,
            &[] as &[(String, String)],
            &credential,
            &self.config.region,
            Utc::now(),
            Duration::from_secs(expiration_seconds),
        );
        Ok(PresignedDownloadResult {
            url,
            headers: Vec::new(),
            file_size: Some(meta.size as i64),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct InitiateMultipartUploadResult {
    upload_id: String,
}

async fn s3_response_error(response: reqwest::Response) -> MlflowError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let code = xml_tag(&body, "Code");
    let message = xml_tag(&body, "Message").unwrap_or_else(|| body.trim().to_string());
    match (status, code.as_deref()) {
        (StatusCode::NOT_FOUND, _) | (_, Some("NoSuchBucket" | "NoSuchKey")) => {
            MlflowError::resource_does_not_exist(message)
        }
        (StatusCode::FORBIDDEN, Some("InvalidAccessKeyId" | "SignatureDoesNotMatch")) => {
            MlflowError::new(message, ErrorCode::Unauthenticated)
        }
        (StatusCode::FORBIDDEN, _) => MlflowError::permission_denied(message),
        _ => {
            MlflowError::internal_error(format!("S3 request failed with HTTP {status}: {message}"))
        }
    }
}

fn xml_tag(body: &str, name: &str) -> Option<String> {
    let start_tag = format!("<{name}>");
    let end_tag = format!("</{name}>");
    let start = body.find(&start_tag)? + start_tag.len();
    let end = body[start..].find(&end_tag)? + start;
    Some(body[start..end].to_string())
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn set_query(url: &mut Url, query: &[(&str, &str)]) {
    let query = canonical_query(
        query
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string())),
    );
    url.set_query(Some(&query));
}

fn authorization_headers(
    method: &Method,
    url: &Url,
    body: &[u8],
    credential: &AwsCredential,
    region: &str,
    now: DateTime<Utc>,
) -> reqwest::header::HeaderMap {
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();
    let payload_hash = hex_sha256(body);
    let host = url_host(url);
    let mut canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
    let mut signed_headers = String::from("host;x-amz-content-sha256;x-amz-date");
    if let Some(token) = &credential.token {
        canonical_headers.push_str(&format!("x-amz-security-token:{}\n", token.trim()));
        signed_headers.push_str(";x-amz-security-token");
    }
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method.as_str(),
        url.path(),
        url.query().unwrap_or_default(),
        canonical_headers,
        signed_headers,
        payload_hash
    );
    let scope = format!("{date}/{region}/s3/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex_sha256(canonical_request.as_bytes())
    );
    let signature = sigv4_signature(&credential.secret_key, &date, region, &string_to_sign);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        credential.key_id
    );

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("host", host.parse().expect("URL host is a valid header"));
    headers.insert(
        "x-amz-content-sha256",
        payload_hash.parse().expect("SHA-256 is a valid header"),
    );
    headers.insert(
        "x-amz-date",
        amz_date.parse().expect("AWS date is a valid header"),
    );
    headers.insert(
        reqwest::header::AUTHORIZATION,
        authorization
            .parse()
            .expect("SigV4 authorization is a valid header"),
    );
    if let Some(token) = &credential.token {
        headers.insert(
            "x-amz-security-token",
            token.parse().expect("AWS session token is a valid header"),
        );
    }
    headers
}

fn presigned_url(
    method: Method,
    mut url: Url,
    operation_query: &[(impl AsRef<str>, String)],
    credential: &AwsCredential,
    region: &str,
    now: DateTime<Utc>,
    expires: Duration,
) -> String {
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();
    let scope = format!("{date}/{region}/s3/aws4_request");
    let mut params = operation_query
        .iter()
        .map(|(key, value)| (key.as_ref().to_string(), value.clone()))
        .collect::<Vec<_>>();
    params.extend([
        (
            "X-Amz-Algorithm".to_string(),
            "AWS4-HMAC-SHA256".to_string(),
        ),
        (
            "X-Amz-Credential".to_string(),
            format!("{}/{scope}", credential.key_id),
        ),
        ("X-Amz-Date".to_string(), amz_date.clone()),
        ("X-Amz-Expires".to_string(), expires.as_secs().to_string()),
        ("X-Amz-SignedHeaders".to_string(), "host".to_string()),
    ]);
    if let Some(token) = &credential.token {
        params.push(("X-Amz-Security-Token".to_string(), token.clone()));
    }
    let query = canonical_query(params);
    let host = url_host(&url);
    let canonical_request = format!(
        "{}\n{}\n{}\nhost:{}\n\nhost\nUNSIGNED-PAYLOAD",
        method.as_str(),
        url.path(),
        query,
        host
    );
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex_sha256(canonical_request.as_bytes())
    );
    let signature = sigv4_signature(&credential.secret_key, &date, region, &string_to_sign);
    url.set_query(Some(&format!("{query}&X-Amz-Signature={signature}")));
    url.to_string()
}

fn canonical_query(params: impl IntoIterator<Item = (String, String)>) -> String {
    let mut encoded = params
        .into_iter()
        .map(|(key, value)| (aws_uri_encode(&key, true), aws_uri_encode(&value, true)))
        .collect::<Vec<_>>();
    encoded.sort();
    encoded
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn aws_uri_encode(value: &str, encode_slash: bool) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        if byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b'~')
            || (!encode_slash && byte == b'/')
        {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
    }
    encoded
}

fn url_host(url: &Url) -> String {
    url.as_str()
        .strip_prefix(&format!("{}://", url.scheme()))
        .and_then(|rest| rest.split('/').next())
        .expect("absolute S3 URLs always have an authority")
        .to_string()
}

fn hex_sha256(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn hmac_sha256(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(value);
    mac.finalize().into_bytes().to_vec()
}

fn sigv4_signature(secret: &str, date: &str, region: &str, string_to_sign: &str) -> String {
    let date_key = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let region_key = hmac_sha256(&date_key, region.as_bytes());
    let service_key = hmac_sha256(&region_key, b"s3");
    let signing_key = hmac_sha256(&service_key, b"aws4_request");
    let signature = hmac_sha256(&signing_key, string_to_sign.as_bytes());
    signature.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn config(uri: &str, values: &[(&str, &str)]) -> Result<S3RuntimeConfig, MlflowError> {
        let values = values.iter().copied().collect::<BTreeMap<_, _>>();
        S3RuntimeConfig::from_lookup(uri, |name| values.get(name).map(ToString::to_string))
    }

    #[test]
    fn parses_bucket_and_prefix() {
        let config = config("s3://bucket/some/deep/prefix", &[]).unwrap();
        assert_eq!(config.bucket, "bucket");
        assert_eq!(config.root.as_ref(), "some/deep/prefix");
        assert_eq!(config.addressing_style, AddressingStyle::Virtual);
        assert_eq!(config.region, "us-east-1");
    }

    #[test]
    fn rejects_missing_bucket_and_unsafe_prefix() {
        assert!(config("s3:///prefix", &[]).is_err());
        assert!(config("s3://bucket/a/../b", &[]).is_err());
    }

    #[test]
    fn custom_endpoint_defaults_to_path_style_and_allows_http() {
        let mapped = config(
            "s3://bucket/prefix",
            &[("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")],
        )
        .unwrap();
        assert_eq!(mapped.addressing_style, AddressingStyle::Path);
        assert!(is_http(mapped.endpoint.as_deref().unwrap()));
        assert_eq!(
            mapped
                .object_url(&ObjPath::from("prefix/a b"))
                .unwrap()
                .as_str(),
            "http://127.0.0.1:9000/bucket/prefix/a%20b"
        );
    }

    #[test]
    fn explicit_addressing_styles_override_auto() {
        let path = config(
            "s3://bucket",
            &[("MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE", "path")],
        )
        .unwrap();
        assert_eq!(path.addressing_style, AddressingStyle::Path);
        let virtual_custom = config(
            "s3://bucket",
            &[
                (
                    "MLFLOW_S3_ENDPOINT_URL",
                    "https://objects.example.test/base",
                ),
                ("MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE", "virtual"),
            ],
        )
        .unwrap();
        assert_eq!(virtual_custom.addressing_style, AddressingStyle::Virtual);
        assert_eq!(
            virtual_custom
                .object_url(&ObjPath::from("key"))
                .unwrap()
                .as_str(),
            "https://bucket.objects.example.test/base/key"
        );
    }

    #[test]
    fn maps_region_and_tls_flag() {
        let mapped = config(
            "s3://bucket",
            &[
                ("AWS_DEFAULT_REGION", "eu-west-2"),
                ("AWS_REGION", "eu-central-1"),
                ("MLFLOW_S3_IGNORE_TLS", "TrUe"),
            ],
        )
        .unwrap();
        assert_eq!(mapped.region, "eu-central-1");
        assert!(mapped.ignore_tls);
        assert!(config("s3://bucket", &[("MLFLOW_S3_IGNORE_TLS", "yes")]).is_err());
    }

    #[test]
    fn rejects_invalid_endpoint_and_addressing_style() {
        assert!(config(
            "s3://bucket",
            &[("MLFLOW_S3_ENDPOINT_URL", "ftp://example.test")]
        )
        .is_err());
        assert!(config(
            "s3://bucket",
            &[("MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE", "dns")]
        )
        .is_err());
    }

    #[test]
    fn sigv4_query_encoding_and_order_are_aws_compatible() {
        let query = canonical_query([
            ("uploadId".to_string(), "a+b/c=".to_string()),
            ("partNumber".to_string(), "2".to_string()),
            ("X-Amz-Date".to_string(), "20260720T000000Z".to_string()),
        ]);
        assert_eq!(
            query,
            "X-Amz-Date=20260720T000000Z&partNumber=2&uploadId=a%2Bb%2Fc%3D"
        );
    }
}
