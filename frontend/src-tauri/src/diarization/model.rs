//! Speaker-embedding model location, verification and download.
//!
//! Model: WeSpeaker ResNet34 (VoxCeleb), ONNX export from the sherpa-onnx model release.
//! Trained on English-dominant data; quality on Polish speech must be verified on real recordings.

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

pub const MODEL_FILE: &str = "wespeaker_en_voxceleb_resnet34.onnx";
pub const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx";
pub const MODEL_SHA256: &str = "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94";
pub const MODEL_SIZE_BYTES: u64 = 26_534_365;

#[derive(thiserror::Error, Debug)]
pub enum ModelDownloadError {
    #[error("download failed: {0}")]
    Http(String),
    #[error("file error: {0}")]
    Io(#[from] std::io::Error),
    #[error("downloaded file is corrupted (checksum mismatch)")]
    ChecksumMismatch,
}

/// `<models_dir>/speaker/<MODEL_FILE>`
pub fn model_path(models_dir: &Path) -> PathBuf {
    models_dir.join("speaker").join(MODEL_FILE)
}

/// Cheap readiness check (file exists with the expected size). The checksum is verified at download time.
pub fn is_model_ready(models_dir: &Path) -> bool {
    std::fs::metadata(model_path(models_dir)).map(|m| m.is_file() && m.len() == MODEL_SIZE_BYTES).unwrap_or(false)
}

/// Downloads `url` to the model path via a `.part` file, verifying SHA-256 before the final rename.
/// `on_progress(downloaded, total)` is called as data arrives (`total` is 0 when unknown).
pub async fn download_model(
    models_dir: &Path,
    url: &str,
    expected_sha256: &str,
    on_progress: impl Fn(u64, u64),
) -> Result<PathBuf, ModelDownloadError> {
    let target = model_path(models_dir);
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let part = target.with_extension("onnx.part");

    let response = reqwest::get(url).await.map_err(|e| ModelDownloadError::Http(e.to_string()))?;
    if !response.status().is_success() {
        return Err(ModelDownloadError::Http(format!("HTTP {}", response.status())));
    }
    let total = response.content_length().unwrap_or(0);

    let mut file = tokio::fs::File::create(&part).await?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    let result: Result<(), ModelDownloadError> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ModelDownloadError::Http(e.to_string()))?;
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            on_progress(downloaded, total);
        }
        file.flush().await?;
        Ok(())
    }
    .await;
    drop(file);

    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e);
    }
    let actual: String = hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect();
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(ModelDownloadError::ChecksumMismatch);
    }
    tokio::fs::rename(&part, &target).await?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    /// Serves `body` once on a local port and returns the URL.
    async fn serve(body: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { break };
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = sock.read(&mut buf).await;
                    let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(&body).await;
                });
            }
        });
        format!("http://{addr}/m.onnx")
    }

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(data).iter().map(|b| format!("{:02x}", b)).collect()
    }

    #[tokio::test]
    async fn downloads_verifies_and_renames() {
        let body: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let url = serve(body.clone()).await;
        let dir = tempfile::tempdir().unwrap();
        let last = AtomicU64::new(0);
        let path = download_model(dir.path(), &url, &sha256_hex(&body), |d, _| last.store(d, Ordering::SeqCst)).await.unwrap();
        assert_eq!(path, model_path(dir.path()));
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(last.load(Ordering::SeqCst), body.len() as u64);
        assert!(!path.with_extension("onnx.part").exists());
    }

    #[tokio::test]
    async fn checksum_mismatch_leaves_no_files() {
        let url = serve(vec![1, 2, 3, 4]).await;
        let dir = tempfile::tempdir().unwrap();
        let err = download_model(dir.path(), &url, MODEL_SHA256, |_, _| {}).await.unwrap_err();
        assert!(matches!(err, ModelDownloadError::ChecksumMismatch));
        assert!(!model_path(dir.path()).exists());
        assert!(!model_path(dir.path()).with_extension("onnx.part").exists());
    }

    #[tokio::test]
    async fn http_error_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let err = download_model(dir.path(), "http://127.0.0.1:1/none", MODEL_SHA256, |_, _| {}).await.unwrap_err();
        assert!(matches!(err, ModelDownloadError::Http(_)));
    }

    #[test]
    fn readiness_requires_exact_size() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_model_ready(dir.path()));
        std::fs::create_dir_all(model_path(dir.path()).parent().unwrap()).unwrap();
        std::fs::write(model_path(dir.path()), vec![0u8; 10]).unwrap();
        assert!(!is_model_ready(dir.path()));
        let f = std::fs::File::create(model_path(dir.path())).unwrap();
        f.set_len(MODEL_SIZE_BYTES).unwrap();
        assert!(is_model_ready(dir.path()));
    }
}
