//! Persistent diarization settings (tauri-plugin-store, `diarization_settings.json`).

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Runtime};
use tauri_plugin_store::StoreExt;

const STORE_FILE: &str = "diarization_settings.json";
const KEY_ENABLED: &str = "enabled";
const KEY_MAX_SPEAKERS: &str = "max_speakers";

pub const DEFAULT_MAX_SPEAKERS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSettings {
    pub enabled: bool,
    pub max_speakers: usize,
}

impl Default for DiarizationSettings {
    fn default() -> Self {
        Self { enabled: false, max_speakers: DEFAULT_MAX_SPEAKERS }
    }
}

pub fn load_settings<R: Runtime>(app: &AppHandle<R>) -> DiarizationSettings {
    let mut settings = DiarizationSettings::default();
    if let Ok(store) = app.store(STORE_FILE) {
        if let Some(v) = store.get(KEY_ENABLED).and_then(|v| v.as_bool()) {
            settings.enabled = v;
        }
        if let Some(v) = store.get(KEY_MAX_SPEAKERS).and_then(|v| v.as_u64()) {
            settings.max_speakers = (v as usize).clamp(2, 20);
        }
    }
    settings
}

pub fn save_settings<R: Runtime>(app: &AppHandle<R>, settings: &DiarizationSettings) -> Result<(), String> {
    let store = app.store(STORE_FILE).map_err(|e| e.to_string())?;
    store.set(KEY_ENABLED, serde_json::json!(settings.enabled));
    store.set(KEY_MAX_SPEAKERS, serde_json::json!(settings.max_speakers.clamp(2, 20)));
    store.save().map_err(|e| e.to_string())
}
