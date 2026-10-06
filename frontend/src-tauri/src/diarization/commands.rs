//! Tauri commands + runtime factory for live speaker diarization.

use super::config::{load_settings, save_settings, DiarizationSettings};
use super::identifier::{IdentifierConfig, SpeakerIdentifier};
use super::model::{self, MODEL_SIZE_BYTES};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{command, AppHandle, Emitter, Manager, Runtime};

static DOWNLOADING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Serialize)]
pub struct DiarizationStatus {
    pub enabled: bool,
    pub max_speakers: usize,
    pub model_ready: bool,
    pub model_size_bytes: u64,
    pub downloading: bool,
}

fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    app.path().app_data_dir().map(|d| d.join("models")).map_err(|e| e.to_string())
}

fn status<R: Runtime>(app: &AppHandle<R>) -> Result<DiarizationStatus, String> {
    let settings = load_settings(app);
    Ok(DiarizationStatus {
        enabled: settings.enabled,
        max_speakers: settings.max_speakers,
        model_ready: model::is_model_ready(&models_dir(app)?),
        model_size_bytes: MODEL_SIZE_BYTES,
        downloading: DOWNLOADING.load(Ordering::SeqCst),
    })
}

#[command]
pub async fn diarization_get_status<R: Runtime>(app: AppHandle<R>) -> Result<DiarizationStatus, String> {
    status(&app)
}

#[command]
pub async fn diarization_set_enabled<R: Runtime>(
    app: AppHandle<R>,
    enabled: bool,
    max_speakers: Option<usize>,
) -> Result<DiarizationStatus, String> {
    let mut settings: DiarizationSettings = load_settings(&app);
    settings.enabled = enabled;
    if let Some(m) = max_speakers {
        settings.max_speakers = m;
    }
    save_settings(&app, &settings)?;
    status(&app)
}

#[command]
pub async fn diarization_download_model<R: Runtime>(app: AppHandle<R>) -> Result<DiarizationStatus, String> {
    if DOWNLOADING.swap(true, Ordering::SeqCst) {
        return Err("Model download already in progress".into());
    }
    let dir = models_dir(&app);
    let result = match dir {
        Ok(dir) => {
            let app_progress = app.clone();
            model::download_model(&dir, model::MODEL_URL, model::MODEL_SHA256, move |done, total| {
                let _ = app_progress.emit(
                    "diarization-model-download-progress",
                    serde_json::json!({ "downloaded": done, "total": if total > 0 { total } else { MODEL_SIZE_BYTES } }),
                );
            })
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
        }
        Err(e) => Err(e),
    };
    DOWNLOADING.store(false, Ordering::SeqCst);
    result?;
    status(&app)
}

/// Builds the live identifier for a recording, or `None` when diarization is disabled,
/// the model is missing, or loading failed (transcription then proceeds without speakers).
pub fn create_identifier<R: Runtime>(app: &AppHandle<R>) -> Option<Arc<Mutex<SpeakerIdentifier>>> {
    let settings = load_settings(app);
    if !settings.enabled {
        return None;
    }
    let dir = models_dir(app).ok()?;
    if !model::is_model_ready(&dir) {
        log::warn!("Diarization enabled but model is not downloaded; continuing without speaker labels");
        return None;
    }
    let mut cfg = IdentifierConfig::default();
    cfg.clusterer.max_speakers = settings.max_speakers;
    match SpeakerIdentifier::new(&model::model_path(&dir), cfg, 2) {
        Ok(id) => {
            log::info!("Live speaker diarization enabled (max {} speakers)", settings.max_speakers);
            Some(Arc::new(Mutex::new(id)))
        }
        Err(e) => {
            log::error!("Failed to load diarization model: {}", e);
            None
        }
    }
}
