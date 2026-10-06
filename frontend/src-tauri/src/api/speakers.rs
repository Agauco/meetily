//! Tauri commands for voice-based speakers (speaker diarization, phase 1: data layer).
//!
//! All commands are thin wrappers over `SpeakersRepository`. Mutations emit a
//! `speaker-updated` event so open views can refresh.

use log::{error as log_error, info as log_info};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime};

use crate::database::repositories::speaker::{
    SegmentSpeaker, Speaker, SpeakerSummary, SpeakersRepository,
};
use crate::state::AppState;

#[derive(Clone, Serialize)]
struct SpeakerUpdated {
    meeting_id: String,
    /// "rename" | "assign" | "merge" | "create" | "undo"
    kind: &'static str,
}

fn notify<R: Runtime>(app: &AppHandle<R>, meeting_id: &str, kind: &'static str) {
    let _ = app.emit(
        "speaker-updated",
        SpeakerUpdated {
            meeting_id: meeting_id.to_string(),
            kind,
        },
    );
}

fn to_err(context: &str, e: impl std::fmt::Display) -> String {
    log_error!("{}: {}", context, e);
    format!("{}: {}", context, e)
}

#[tauri::command]
pub async fn api_list_speakers(
    meeting_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<SpeakerSummary>, String> {
    SpeakersRepository::list_speakers(state.db_manager.pool(), &meeting_id)
        .await
        .map_err(|e| to_err("Failed to list speakers", e))
}

#[tauri::command]
pub async fn api_list_segment_speakers(
    meeting_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<SegmentSpeaker>, String> {
    SpeakersRepository::list_segment_speakers(state.db_manager.pool(), &meeting_id)
        .await
        .map_err(|e| to_err("Failed to list segment speakers", e))
}

#[tauri::command]
pub async fn api_create_speaker<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    name: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<Speaker, String> {
    let speaker = SpeakersRepository::create_speaker(
        state.db_manager.pool(),
        &meeting_id,
        name.as_deref(),
        None,
    )
    .await
    .map_err(|e| to_err("Failed to create speaker", e))?;
    notify(&app, &meeting_id, "create");
    Ok(speaker)
}

#[tauri::command]
pub async fn api_rename_speaker<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    speaker_id: String,
    name: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    SpeakersRepository::rename_speaker(state.db_manager.pool(), &speaker_id, name.as_deref())
        .await
        .map_err(|e| to_err("Failed to rename speaker", e))?;
    log_info!("Renamed speaker {} in meeting {}", speaker_id, meeting_id);
    notify(&app, &meeting_id, "rename");
    Ok(())
}

/// Manually assigns a speaker to one or more segments (`speaker_id = null` clears the override).
/// Returns the created edit ids, usable with `api_undo_speaker_edit`.
#[tauri::command]
pub async fn api_assign_segment_speaker<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    transcript_ids: Vec<String>,
    speaker_id: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let edit_ids = SpeakersRepository::assign_segment_speaker(
        state.db_manager.pool(),
        &meeting_id,
        &transcript_ids,
        speaker_id.as_deref(),
    )
    .await
    .map_err(|e| to_err("Failed to assign speaker", e))?;
    if !edit_ids.is_empty() {
        notify(&app, &meeting_id, "assign");
    }
    Ok(edit_ids)
}

/// Merges `from_speaker_id` into `into_speaker_id`; returns the surviving speaker id.
#[tauri::command]
pub async fn api_merge_speakers<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    from_speaker_id: String,
    into_speaker_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let survivor = SpeakersRepository::merge_speakers(
        state.db_manager.pool(),
        &from_speaker_id,
        &into_speaker_id,
    )
    .await
    .map_err(|e| to_err("Failed to merge speakers", e))?;
    notify(&app, &meeting_id, "merge");
    Ok(survivor)
}

#[tauri::command]
pub async fn api_undo_speaker_edit<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    edit_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    SpeakersRepository::undo_edit(state.db_manager.pool(), &edit_id)
        .await
        .map_err(|e| to_err("Failed to undo speaker edit", e))?;
    notify(&app, &meeting_id, "undo");
    Ok(())
}
