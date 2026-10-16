//! Offline (re)diarization of a saved meeting: decodes the recording, analyzes the voice of every
//! transcript segment, clusters the whole meeting at once and rewrites only the *automatic* speakers.
//!
//! Runs automatically after a recording is saved (when diarization is enabled) and on demand via
//! `api_rediarize_meeting` (e.g. after an import or with a known number of participants).
//! Manual corrections and speaker names are never overwritten.

use super::config::load_settings;
use super::embedder::SpeakerEmbedder;
use super::identifier::{analyze_segment, IdentifierConfig};
use super::model;
use super::offline::{cluster_segments, map_clusters_to_speakers, OfflineConfig, OfflineLabel, VoiceSegment};
use crate::audio::decoder::decode_audio_file;
use crate::database::repositories::speaker::{
    DiarizationSegmentRow, OfflineAssignment, OfflineTarget, SpeakersRepository,
};
use crate::state::AppState;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, Runtime};

const SAMPLE_RATE: usize = 16_000;
/// Frame used for trimming silence inside a segment (25 ms).
const TRIM_FRAME: usize = 400;

fn running() -> &'static Mutex<HashSet<String>> {
    static RUNNING: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    RUNNING.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Marks a meeting as being processed for the lifetime of the guard.
struct RunGuard(String);

impl RunGuard {
    fn acquire(meeting_id: &str) -> Option<Self> {
        let mut set = running().lock().unwrap_or_else(|e| e.into_inner());
        set.insert(meeting_id.to_string()).then(|| RunGuard(meeting_id.to_string()))
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        running().lock().unwrap_or_else(|e| e.into_inner()).remove(&self.0);
    }
}

pub fn is_running(meeting_id: &str) -> bool {
    running().lock().unwrap_or_else(|e| e.into_inner()).contains(meeting_id)
}

#[derive(Debug, Clone, Serialize)]
pub struct RediarizeSummary {
    pub meeting_id: String,
    pub segments_total: usize,
    pub segments_analyzed: usize,
    pub speaker_count: usize,
    pub changed_transcript_ids: Vec<String>,
    pub created_speakers: usize,
    pub removed_speakers: usize,
}

#[derive(Clone, Serialize)]
struct Progress<'a> {
    meeting_id: &'a str,
    /// "decoding" | "analyzing" | "saving" | "done" | "failed"
    stage: &'a str,
    /// 0..100
    progress: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn emit_progress<R: Runtime>(app: &AppHandle<R>, meeting_id: &str, stage: &str, progress: u32, error: Option<String>) {
    let _ = app.emit("speaker-diarization-progress", Progress { meeting_id, stage, progress, error });
}

/// Drops near-silent frames so the voice vector describes the speaker, not the room noise.
/// Returns the input unchanged when too little would remain.
pub(crate) fn trim_silence(samples: &[f32]) -> Vec<f32> {
    if samples.len() < TRIM_FRAME * 4 {
        return samples.to_vec();
    }
    let frames: Vec<&[f32]> = samples.chunks(TRIM_FRAME).collect();
    let rms: Vec<f32> = frames
        .iter()
        .map(|f| (f.iter().map(|x| x * x).sum::<f32>() / f.len() as f32).sqrt())
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p90 = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)];
    let threshold = (p90 * 0.15).max(1e-3);
    let kept: Vec<f32> = frames
        .iter()
        .zip(&rms)
        .filter(|(_, r)| **r >= threshold)
        .flat_map(|(f, _)| f.iter().copied())
        .collect();
    if kept.len() < SAMPLE_RATE / 2 {
        samples.to_vec()
    } else {
        kept
    }
}

/// Sample range of a segment in a 16 kHz signal of `total` samples.
pub(crate) fn segment_range(row: &DiarizationSegmentRow, total: usize) -> Option<(usize, usize)> {
    let start = row.audio_start_time?;
    let end = row
        .audio_end_time
        .or_else(|| row.duration.map(|d| start + d))
        .filter(|e| *e > start)?;
    let s = ((start.max(0.0)) * SAMPLE_RATE as f64) as usize;
    let e = ((end * SAMPLE_RATE as f64) as usize).min(total);
    (e > s).then_some((s, e))
}

/// Finds the meeting's recording, waiting up to `wait` for it to appear (it is finalized right
/// after the recording stops).
async fn find_audio(folder: &Path, wait: Duration) -> Result<PathBuf, String> {
    let deadline = std::time::Instant::now() + wait;
    loop {
        match crate::audio::retranscription::find_audio_file(folder) {
            Ok(p) => return Ok(p),
            Err(e) if std::time::Instant::now() >= deadline => return Err(e.to_string()),
            Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
        }
    }
}

fn analysis_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get() / 2).unwrap_or(1).clamp(1, 4)
}

/// Re-diarizes a saved meeting. `wait_for_audio` gives a just-stopped recording time to be written.
pub async fn rediarize_meeting<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    expected_speakers: Option<usize>,
    wait_for_audio: Duration,
) -> Result<RediarizeSummary, String> {
    let Some(_guard) = RunGuard::acquire(meeting_id) else {
        return Err("Rozpoznawanie mówców dla tego spotkania już trwa".into());
    };
    let result = run(app, meeting_id, expected_speakers, wait_for_audio).await;
    match &result {
        Ok(summary) => {
            emit_progress(app, meeting_id, "done", 100, None);
            let _ = app.emit("speakers-relabeled", summary);
            let _ = app.emit(
                "speaker-updated",
                serde_json::json!({ "meeting_id": meeting_id, "kind": "rediarize" }),
            );
        }
        Err(e) => {
            log::warn!("Offline diarization of {} failed: {}", meeting_id, e);
            emit_progress(app, meeting_id, "failed", 100, Some(e.clone()));
        }
    }
    result
}

async fn run<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    expected_speakers: Option<usize>,
    wait_for_audio: Duration,
) -> Result<RediarizeSummary, String> {
    let settings = load_settings(app);
    let models_dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
    if !model::is_model_ready(&models_dir) {
        return Err("Model rozpoznawania głosu nie jest pobrany (Ustawienia → Rozpoznawanie mówców)".into());
    }
    let pool = app.state::<AppState>().db_manager.pool().clone();

    let folder: Option<String> = sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id = ?")
        .bind(meeting_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Nie znaleziono spotkania {}", meeting_id))?;
    let folder = folder.filter(|f| !f.is_empty()).ok_or("Spotkanie nie ma zapisanego nagrania")?;
    let audio_path = find_audio(Path::new(&folder), wait_for_audio).await?;

    let rows = SpeakersRepository::list_segments_for_diarization(&pool, meeting_id)
        .await
        .map_err(|e| e.to_string())?;
    if rows.is_empty() {
        return Err("Spotkanie nie ma transkrypcji".into());
    }

    emit_progress(app, meeting_id, "decoding", 0, None);
    log::info!("Offline diarization of {}: {} segments, audio {}", meeting_id, rows.len(), audio_path.display());

    // Decode + analyze on a blocking thread (CPU heavy).
    let model_path = model::model_path(&models_dir);
    let rows_for_task = rows.clone();
    let app_for_task = app.clone();
    let meeting_for_task = meeting_id.to_string();
    let segments: Vec<VoiceSegment> = tokio::task::spawn_blocking(move || -> Result<Vec<VoiceSegment>, String> {
        let decoded = decode_audio_file(&audio_path).map_err(|e| format!("Nie udało się odczytać nagrania: {}", e))?;
        let audio = decoded.to_whisper_format();
        drop(decoded);
        let mut embedder = SpeakerEmbedder::load(&model_path, analysis_threads()).map_err(|e| e.to_string())?;
        let id_cfg = IdentifierConfig::default();
        let total = rows_for_task.len();
        let mut last_reported = 0u32;
        let mut out = Vec::with_capacity(total);
        for (i, row) in rows_for_task.iter().enumerate() {
            let seg = match segment_range(row, audio.len()) {
                Some((s, e)) => {
                    let voiced = trim_silence(&audio[s..e]);
                    analyze_segment(&mut embedder, &id_cfg, &voiced)
                }
                None => VoiceSegment::default(),
            };
            out.push(seg);
            let pct = (5 + (i + 1) * 85 / total) as u32;
            if pct >= last_reported + 5 {
                last_reported = pct;
                emit_progress(&app_for_task, &meeting_for_task, "analyzing", pct, None);
            }
        }
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())??;

    let analyzed = segments.iter().filter(|s| s.embedding.is_some() || s.is_room).count();
    let cfg = OfflineConfig {
        max_speakers: settings.max_speakers,
        expected_speakers: expected_speakers.filter(|k| *k > 0),
        ..OfflineConfig::default()
    };
    let result = cluster_segments(&segments, &cfg);

    emit_progress(app, meeting_id, "saving", 92, None);
    let current = SpeakersRepository::displayed_person_speakers(&pool, meeting_id, &rows)
        .await
        .map_err(|e| e.to_string())?;
    let mapping = map_clusters_to_speakers(&result, &segments, &current, &cfg);

    let mut existing_centroids: HashMap<String, Vec<f32>> = HashMap::new();
    let mut new_centroids: HashMap<usize, Vec<f32>> = HashMap::new();
    for (cluster, target) in mapping.iter().enumerate() {
        match target {
            Some(id) => existing_centroids.insert(id.clone(), result.centroids[cluster].clone()),
            None => new_centroids.insert(cluster, result.centroids[cluster].clone()),
        };
    }
    let assignments: Vec<OfflineAssignment> = rows
        .iter()
        .zip(&result.labels)
        .filter_map(|(row, label)| {
            let (target, confidence) = match *label {
                OfflineLabel::Person { cluster, confidence } => (
                    match &mapping[cluster] {
                        Some(id) => OfflineTarget::Existing(id.clone()),
                        None => OfflineTarget::New(cluster),
                    },
                    confidence,
                ),
                OfflineLabel::Room { confidence } => (OfflineTarget::Room, confidence),
                OfflineLabel::Unknown => return None,
            };
            Some(OfflineAssignment { transcript_id: row.id.clone(), target, confidence: confidence as f64 })
        })
        .collect();

    let applied = SpeakersRepository::apply_offline_assignments(&pool, meeting_id, &assignments, &existing_centroids, &new_centroids)
        .await
        .map_err(|e| e.to_string())?;

    log::info!(
        "Offline diarization of {}: {} speakers, {}/{} segments analyzed, {} changed, {} new, {} removed",
        meeting_id,
        result.cluster_count(),
        analyzed,
        rows.len(),
        applied.changed_transcript_ids.len(),
        applied.created_speakers,
        applied.removed_speakers
    );

    Ok(RediarizeSummary {
        meeting_id: meeting_id.to_string(),
        segments_total: rows.len(),
        segments_analyzed: analyzed,
        speaker_count: result.cluster_count(),
        changed_transcript_ids: applied.changed_transcript_ids,
        created_speakers: applied.created_speakers,
        removed_speakers: applied.removed_speakers,
    })
}

/// Starts the offline pass in the background after a recording was saved, when diarization is on
/// and the voice model is available. Never fails the save.
pub fn spawn_after_save<R: Runtime>(app: &AppHandle<R>, meeting_id: String) {
    let settings = load_settings(app);
    if !settings.enabled {
        return;
    }
    let ready = app
        .path()
        .app_data_dir()
        .map(|d| model::is_model_ready(&d.join("models")))
        .unwrap_or(false);
    if !ready {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let _ = rediarize_meeting(&app, &meeting_id, None, Duration::from_secs(60)).await;
    });
}

#[tauri::command]
pub async fn api_rediarize_meeting<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    expected_speakers: Option<usize>,
) -> Result<RediarizeSummary, String> {
    rediarize_meeting(&app, &meeting_id, expected_speakers, Duration::from_secs(0)).await
}

#[tauri::command]
pub async fn api_is_rediarizing(meeting_id: String) -> bool {
    is_running(&meeting_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(start: Option<f64>, end: Option<f64>, duration: Option<f64>) -> DiarizationSegmentRow {
        DiarizationSegmentRow {
            id: "t".into(),
            audio_start_time: start,
            audio_end_time: end,
            duration,
            auto_speaker_id: None,
            manual_speaker_id: None,
        }
    }

    #[test]
    fn segment_range_uses_end_or_duration_and_clamps() {
        assert_eq!(segment_range(&row(Some(1.0), Some(2.5), None), 100_000), Some((16_000, 40_000)));
        assert_eq!(segment_range(&row(Some(1.0), None, Some(0.5)), 100_000), Some((16_000, 24_000)));
        assert_eq!(segment_range(&row(Some(5.0), Some(9.0), None), 100_000), Some((80_000, 100_000)));
        assert_eq!(segment_range(&row(Some(7.0), Some(9.0), None), 100_000), None); // starts after the audio
        assert_eq!(segment_range(&row(None, Some(2.0), None), 100_000), None);
        assert_eq!(segment_range(&row(Some(2.0), Some(1.0), None), 100_000), None);
    }

    #[test]
    fn trim_silence_drops_quiet_frames_but_keeps_short_or_flat_input() {
        let mut s = vec![0.0f32; 16_000];
        s.extend((0..16_000).map(|i| ((i as f32) * 0.05).sin() * 0.5));
        s.extend(vec![0.0f32; 16_000]);
        let t = trim_silence(&s);
        assert!(t.len() >= 15_600 && t.len() <= 16_400, "kept {}", t.len());
        let quiet = vec![0.0f32; 20_000];
        assert_eq!(trim_silence(&quiet).len(), 20_000);
        let short = vec![0.3f32; 1000];
        assert_eq!(trim_silence(&short).len(), 1000);
    }

    #[test]
    fn guard_prevents_parallel_runs() {
        let g = RunGuard::acquire("m-guard").unwrap();
        assert!(RunGuard::acquire("m-guard").is_none());
        assert!(is_running("m-guard"));
        drop(g);
        assert!(!is_running("m-guard"));
    }

    /// Simple energy-based speech segmentation, for the real-recording check below.
    fn energy_segments(audio: &[f32]) -> Vec<(usize, usize)> {
        let frame = 480; // 30 ms
        let rms: Vec<f32> = audio.chunks(frame).map(|f| (f.iter().map(|x| x * x).sum::<f32>() / f.len() as f32).sqrt()).collect();
        let mut sorted = rms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let noise = sorted[sorted.len() / 5];
        let thr = (noise * 3.0).max(1e-3);
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        let mut silent = 0;
        for (i, r) in rms.iter().enumerate() {
            if *r >= thr {
                if start.is_none() { start = Some(i); }
                silent = 0;
            } else if let Some(s) = start {
                silent += 1;
                let len = i - s;
                if silent > 17 || len > 500 { // 0.5 s pause or 15 s max
                    let end = i + 1 - silent.min(i + 1 - s);
                    if end - s >= 20 { out.push((s * frame, end * frame)); }
                    start = None;
                    silent = 0;
                }
            }
        }
        out
    }

    /// Runs the offline pass and the live clusterer on a real recording and prints a comparison.
    /// `DZ_MODEL=... DZ_AUDIO=meeting.wav [DZ_EXPECTED=n] cargo test --lib real_recording -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_recording_offline_vs_live() {
        use super::super::identifier::SpeakerIdentifier;
        let (Ok(model), Ok(audio_path)) = (std::env::var("DZ_MODEL"), std::env::var("DZ_AUDIO")) else { return };
        let limit: f64 = std::env::var("DZ_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(f64::MAX);
        let decoded = decode_audio_file(Path::new(&audio_path)).unwrap();
        let mut audio = decoded.to_whisper_format();
        audio.truncate(((limit * SAMPLE_RATE as f64) as usize).min(audio.len()));
        let segs = energy_segments(&audio);
        let speech: f32 = segs.iter().map(|(s, e)| (e - s) as f32 / SAMPLE_RATE as f32).sum();
        println!("audio {:.0}s, {} segments, {:.0}s speech", audio.len() as f32 / SAMPLE_RATE as f32, segs.len(), speech);

        let t0 = std::time::Instant::now();
        let mut live = SpeakerIdentifier::new(Path::new(&model), IdentifierConfig::default(), 2).unwrap();
        let live_labels: Vec<String> = segs.iter().map(|(s, e)| match live.identify(&audio[*s..*e]) {
            Some(t) => match t.kind { super::super::identifier::SpeakerKind::Person(i) => format!("L{}", i + 1), _ => "Sala".into() },
            None => "-".into(),
        }).collect();
        println!("live: {} speakers in {:.1}s", live.speaker_count(), t0.elapsed().as_secs_f32());

        let t1 = std::time::Instant::now();
        let mut embedder = SpeakerEmbedder::load(Path::new(&model), 2).unwrap();
        let cfgi = IdentifierConfig::default();
        let voice: Vec<VoiceSegment> = segs.iter().map(|(s, e)| analyze_segment(&mut embedder, &cfgi, &trim_silence(&audio[*s..*e]))).collect();
        let analysis = t1.elapsed().as_secs_f32();
        let mut cfg = OfflineConfig { expected_speakers: std::env::var("DZ_EXPECTED").ok().and_then(|v| v.parse().ok()), ..Default::default() };
        if let Some(m) = std::env::var("DZ_MERGE").ok().and_then(|v| v.parse().ok()) { cfg.merge_threshold = m; }
        let t2 = std::time::Instant::now();
        let r = cluster_segments(&voice, &cfg);
        println!("offline: {} speakers; analysis {:.1}s, clustering {:.2}s", r.cluster_count(), analysis, t2.elapsed().as_secs_f32());

        // speaking time per offline cluster and agreement with live
        let mut per: HashMap<String, f32> = HashMap::new();
        let mut joint: HashMap<(String, String), f32> = HashMap::new();
        for (i, (s, e)) in segs.iter().enumerate() {
            let sec = (e - s) as f32 / SAMPLE_RATE as f32;
            let o = match r.labels[i] { OfflineLabel::Person { cluster, .. } => format!("O{}", cluster + 1), OfflineLabel::Room { .. } => "Sala".into(), OfflineLabel::Unknown => "-".into() };
            *per.entry(o.clone()).or_default() += sec;
            *joint.entry((o, live_labels[i].clone())).or_default() += sec;
        }
        let mut per: Vec<_> = per.into_iter().collect();
        per.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (k, v) in &per { println!("  {k:>5}: {v:7.1}s"); }
        let mut joint: Vec<_> = joint.into_iter().collect();
        joint.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("offline x live (top):");
        for ((o, l), v) in joint.iter().take(25) { println!("  {o:>5} {l:>5} {v:7.1}s"); }
        if std::env::var("DZ_TIMELINE").is_ok() {
            for (i, (s, e)) in segs.iter().enumerate() {
                println!("{:7.2} {:7.2} {:?} live={}", *s as f32 / 16000.0, *e as f32 / 16000.0, r.labels[i], live_labels[i]);
            }
        }
    }
}
