//! Repository for voice-based speakers (speaker diarization).
//!
//! Model: every transcript segment has an automatic speaker (`auto_speaker_id`,
//! written by the diarization engine) and an optional manual override
//! (`manual_speaker_id`, written only by the user). The displayed speaker is
//! `COALESCE(manual_speaker_id, auto_speaker_id)`, resolved through
//! `speakers.merged_into`. Automatic processing never touches manual overrides.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, SqlitePool};
use std::collections::HashMap;
use uuid::Uuid;

/// Colors assigned to new speakers in order (cycled).
const SPEAKER_COLORS: [&str; 10] = [
    "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316",
    "#6366F1", "#84CC16",
];

/// Label and color of the reserved "room" speaker (overlapping / unidentifiable speech).
pub const ROOM_LABEL: &str = "Sala";
pub const ROOM_COLOR: &str = "#6B7280";
pub const KIND_PERSON: &str = "person";
pub const KIND_ROOM: &str = "room";

/// Maximum depth followed when resolving `merged_into` chains (guards against corrupt data).
const MAX_MERGE_DEPTH: usize = 32;

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Speaker {
    pub id: String,
    pub meeting_id: String,
    pub label: String,
    pub name: Option<String>,
    pub color: Option<String>,
    pub created_at: String,
    pub merged_into: Option<String>,
    /// "person" or "room"
    pub kind: String,
}

/// Speaker with per-meeting statistics (active speakers only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSummary {
    pub id: String,
    pub label: String,
    pub name: Option<String>,
    /// `name` when set, otherwise `label`.
    pub display_name: String,
    pub color: Option<String>,
    /// "person" or "room"
    pub kind: String,
    pub utterance_count: i64,
    pub total_duration: f64,
}

/// Displayed speaker of one transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentSpeaker {
    pub transcript_id: String,
    pub speaker_id: Option<String>,
    pub display_name: Option<String>,
    pub color: Option<String>,
    pub is_manual: bool,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct SpeakerEdit {
    pub id: String,
    pub meeting_id: String,
    pub transcript_id: String,
    pub old_speaker: Option<String>,
    pub new_speaker: Option<String>,
    pub edited_at: String,
    pub undone: bool,
}

/// Timing and speakers of one segment, as needed by the offline diarization pass.
#[derive(Debug, Clone, FromRow)]
pub struct DiarizationSegmentRow {
    pub id: String,
    pub audio_start_time: Option<f64>,
    pub audio_end_time: Option<f64>,
    pub duration: Option<f64>,
    pub auto_speaker_id: Option<String>,
    pub manual_speaker_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum SpeakerError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("speaker not found: {0}")]
    SpeakerNotFound(String),
    #[error("speaker {0} belongs to a different meeting")]
    WrongMeeting(String),
    #[error("cannot merge a speaker into itself")]
    SelfMerge,
    #[error("edit not found or already undone: {0}")]
    EditNotUndoable(String),
    #[error("segment {0} was changed after this edit; cannot undo")]
    UndoConflict(String),
    #[error("the room speaker cannot be renamed or merged")]
    RoomSpeaker,
}

/// Voice-based result for one saved transcript segment, as produced live by the diarization engine.
#[derive(Debug, Clone, Default)]
pub struct AutoSpeakerInput {
    pub transcript_id: String,
    /// Meeting-local person index (0-based); `None` when the segment is unassigned or room.
    pub speaker_index: Option<usize>,
    /// True for overlapping / unidentifiable speech ("Sala").
    pub is_room: bool,
    pub confidence: Option<f64>,
    /// Name the user gave the automatic speaker during the recording (applies to the whole meeting).
    pub speaker_name: Option<String>,
    /// Live manual correction of this segment (person index, possibly one created by the user), if any.
    pub manual_index: Option<usize>,
    /// Live manual correction to the room speaker.
    pub manual_is_room: bool,
    pub manual_name: Option<String>,
}

async fn ensure_room_speaker(conn: &mut sqlx::SqliteConnection, meeting_id: &str) -> Result<String, sqlx::Error> {
    if let Some(id) = sqlx::query_scalar::<_, String>("SELECT id FROM speakers WHERE meeting_id = ? AND kind = 'room' LIMIT 1")
        .bind(meeting_id)
        .fetch_optional(&mut *conn)
        .await?
    {
        return Ok(id);
    }
    let id = Uuid::new_v4().to_string();
    sqlx::query("INSERT INTO speakers (id, meeting_id, label, color, created_at, kind) VALUES (?, ?, ?, ?, ?, 'room')")
        .bind(&id)
        .bind(meeting_id)
        .bind(ROOM_LABEL)
        .bind(ROOM_COLOR)
        .bind(Utc::now().to_rfc3339())
        .execute(&mut *conn)
        .await?;
    Ok(id)
}

/// Creates the speakers referenced by `inputs` (one per distinct person index, plus the room speaker when
/// needed) and stores them as the *automatic* speaker of the given transcript segments.
/// Meant to run inside the transaction that saves a recorded meeting. Manual overrides are never touched.
pub async fn persist_auto_speakers(
    conn: &mut sqlx::SqliteConnection,
    meeting_id: &str,
    inputs: &[AutoSpeakerInput],
) -> Result<(), sqlx::Error> {
    let mut indices: Vec<usize> = inputs.iter().filter(|i| !i.is_room).filter_map(|i| i.speaker_index).collect();
    indices.extend(inputs.iter().filter(|i| !i.manual_is_room).filter_map(|i| i.manual_index));
    indices.sort_unstable();
    indices.dedup();

    // First non-empty name seen for an index wins.
    let mut names: HashMap<usize, String> = HashMap::new();
    for i in inputs {
        let pairs = [(i.speaker_index.filter(|_| !i.is_room), &i.speaker_name), (i.manual_index.filter(|_| !i.manual_is_room), &i.manual_name)];
        for (idx, name) in pairs {
            if let (Some(idx), Some(name)) = (idx, name) {
                let name = name.trim();
                if !name.is_empty() {
                    names.entry(idx).or_insert_with(|| name.to_string());
                }
            }
        }
    }

    let mut ids: HashMap<usize, String> = HashMap::new();
    for index in indices {
        let id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO speakers (id, meeting_id, label, name, color, created_at, kind) VALUES (?, ?, ?, ?, ?, ?, 'person')")
            .bind(&id)
            .bind(meeting_id)
            .bind(format!("Mówca {}", index + 1))
            .bind(names.get(&index).cloned())
            .bind(SPEAKER_COLORS[index % SPEAKER_COLORS.len()])
            .bind(Utc::now().to_rfc3339())
            .execute(&mut *conn)
            .await?;
        ids.insert(index, id);
    }
    let room_id = if inputs.iter().any(|i| i.is_room || i.manual_is_room) {
        Some(ensure_room_speaker(&mut *conn, meeting_id).await?)
    } else {
        None
    };

    for input in inputs {
        let speaker_id = if input.is_room {
            room_id.clone()
        } else {
            input.speaker_index.and_then(|i| ids.get(&i).cloned())
        };
        if speaker_id.is_some() {
            sqlx::query("UPDATE transcripts SET auto_speaker_id = ?, speaker_confidence = ? WHERE id = ? AND meeting_id = ?")
                .bind(speaker_id)
                .bind(input.confidence)
                .bind(&input.transcript_id)
                .bind(meeting_id)
                .execute(&mut *conn)
                .await?;
        }
        let manual_id = if input.manual_is_room {
            room_id.clone()
        } else {
            input.manual_index.and_then(|i| ids.get(&i).cloned())
        };
        if manual_id.is_some() {
            sqlx::query("UPDATE transcripts SET manual_speaker_id = ? WHERE id = ? AND meeting_id = ?")
                .bind(manual_id)
                .bind(&input.transcript_id)
                .bind(meeting_id)
                .execute(&mut *conn)
                .await?;
        }
    }
    Ok(())
}

/// Next free "Mówca N" number in a meeting. Uses the highest existing number (not the count), because
/// live numbering can skip indices and offline cleanup can remove speakers.
async fn next_person_number(conn: &mut sqlx::SqliteConnection, meeting_id: &str) -> Result<usize, sqlx::Error> {
    let labels: Vec<String> = sqlx::query_scalar("SELECT label FROM speakers WHERE meeting_id = ? AND kind = 'person'")
        .bind(meeting_id)
        .fetch_all(&mut *conn)
        .await?;
    let max = labels
        .iter()
        .filter_map(|l| l.strip_prefix("Mówca ").and_then(|n| n.trim().parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    Ok(max.max(labels.len()) + 1)
}

fn centroid_bytes(centroid: &[f32]) -> Vec<u8> {
    centroid.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Target of one segment in an offline (re)diarization result.
#[derive(Debug, Clone, PartialEq)]
pub enum OfflineTarget {
    /// An existing speaker of the meeting keeps this voice (name and color preserved).
    Existing(String),
    /// A new speaker, identified by the offline cluster index (several segments share one).
    New(usize),
    /// Overlapping / unidentifiable speech ("Sala").
    Room,
}

#[derive(Debug, Clone)]
pub struct OfflineAssignment {
    pub transcript_id: String,
    pub target: OfflineTarget,
    pub confidence: f64,
}

/// Result of writing an offline diarization.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OfflineApplyResult {
    /// Segments whose automatic speaker changed.
    pub changed_transcript_ids: Vec<String>,
    /// Speakers created by this pass.
    pub created_speakers: usize,
    /// Unnamed speakers left without any segment and removed.
    pub removed_speakers: usize,
}

pub struct SpeakersRepository;

/// Follows `merged_into` links to the active (root) speaker.
fn resolve_root(links: &HashMap<String, Option<String>>, id: &str) -> String {
    let mut current = id.to_string();
    for _ in 0..MAX_MERGE_DEPTH {
        match links.get(&current) {
            Some(Some(next)) if next != &current => current = next.clone(),
            _ => break,
        }
    }
    current
}

impl SpeakersRepository {
    async fn merge_links(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<HashMap<String, Option<String>>, sqlx::Error> {
        let rows: Vec<(String, Option<String>)> =
            sqlx::query_as("SELECT id, merged_into FROM speakers WHERE meeting_id = ?")
                .bind(meeting_id)
                .fetch_all(pool)
                .await?;
        Ok(rows.into_iter().collect())
    }

    async fn get_speaker(pool: &SqlitePool, speaker_id: &str) -> Result<Speaker, SpeakerError> {
        sqlx::query_as::<_, Speaker>(
            "SELECT id, meeting_id, label, name, color, created_at, merged_into, kind FROM speakers WHERE id = ?",
        )
        .bind(speaker_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| SpeakerError::SpeakerNotFound(speaker_id.to_string()))
    }

    /// Creates a speaker with an automatic label ("Mówca N") and a color.
    pub async fn create_speaker(
        pool: &SqlitePool,
        meeting_id: &str,
        name: Option<&str>,
        centroid: Option<&[f32]>,
    ) -> Result<Speaker, SpeakerError> {
        let n = {
            let mut conn = pool.acquire().await?;
            next_person_number(&mut conn, meeting_id).await?
        };
        let speaker = Speaker {
            id: Uuid::new_v4().to_string(),
            meeting_id: meeting_id.to_string(),
            label: format!("Mówca {}", n),
            name: name.map(str::trim).filter(|s| !s.is_empty()).map(String::from),
            color: Some(SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.len()].to_string()),
            created_at: Utc::now().to_rfc3339(),
            merged_into: None,
            kind: KIND_PERSON.to_string(),
        };
        let centroid_bytes: Option<Vec<u8>> = centroid.map(centroid_bytes);
        sqlx::query(
            "INSERT INTO speakers (id, meeting_id, label, name, color, centroid, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&speaker.id)
        .bind(&speaker.meeting_id)
        .bind(&speaker.label)
        .bind(&speaker.name)
        .bind(&speaker.color)
        .bind(centroid_bytes)
        .bind(&speaker.created_at)
        .execute(pool)
        .await?;
        Ok(speaker)
    }

    /// Active (not merged) speakers of a meeting with utterance count and speaking time,
    /// counted by the *displayed* speaker of each segment.
    pub async fn list_speakers(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<Vec<SpeakerSummary>, SpeakerError> {
        let speakers: Vec<Speaker> = sqlx::query_as(
            "SELECT id, meeting_id, label, name, color, created_at, merged_into, kind FROM speakers WHERE meeting_id = ? ORDER BY created_at, id",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await?;
        let links: HashMap<String, Option<String>> = speakers
            .iter()
            .map(|s| (s.id.clone(), s.merged_into.clone()))
            .collect();

        let rows: Vec<(Option<String>, Option<String>, Option<f64>)> = sqlx::query_as(
            "SELECT manual_speaker_id, auto_speaker_id, duration FROM transcripts WHERE meeting_id = ?",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await?;

        let mut stats: HashMap<String, (i64, f64)> = HashMap::new();
        for (manual, auto, duration) in rows {
            if let Some(shown) = manual.or(auto) {
                let root = resolve_root(&links, &shown);
                let e = stats.entry(root).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += duration.unwrap_or(0.0);
            }
        }

        Ok(speakers
            .into_iter()
            .filter(|s| s.merged_into.is_none())
            .map(|s| {
                let (count, total) = stats.get(&s.id).copied().unwrap_or((0, 0.0));
                SpeakerSummary {
                    display_name: s.name.clone().unwrap_or_else(|| s.label.clone()),
                    id: s.id,
                    label: s.label,
                    name: s.name,
                    color: s.color,
                    kind: s.kind,
                    utterance_count: count,
                    total_duration: total,
                }
            })
            .collect())
    }

    /// Sets (or clears, with `None`/empty) the user-provided name. One update covers
    /// every segment of that speaker, past and future.
    pub async fn rename_speaker(
        pool: &SqlitePool,
        speaker_id: &str,
        name: Option<&str>,
    ) -> Result<(), SpeakerError> {
        let name = name.map(str::trim).filter(|s| !s.is_empty());
        if Self::get_speaker(pool, speaker_id).await?.kind == KIND_ROOM {
            return Err(SpeakerError::RoomSpeaker);
        }
        let res = sqlx::query("UPDATE speakers SET name = ? WHERE id = ?")
            .bind(name)
            .bind(speaker_id)
            .execute(pool)
            .await?;
        if res.rows_affected() == 0 {
            return Err(SpeakerError::SpeakerNotFound(speaker_id.to_string()));
        }
        Ok(())
    }

    /// Merges `from` into `into` (both resolved to their active roots).
    /// Returns the id of the surviving speaker.
    pub async fn merge_speakers(
        pool: &SqlitePool,
        from_id: &str,
        into_id: &str,
    ) -> Result<String, SpeakerError> {
        let from = Self::get_speaker(pool, from_id).await?;
        let into = Self::get_speaker(pool, into_id).await?;
        if from.meeting_id != into.meeting_id {
            return Err(SpeakerError::WrongMeeting(into_id.to_string()));
        }
        if from.kind == KIND_ROOM || into.kind == KIND_ROOM {
            return Err(SpeakerError::RoomSpeaker);
        }
        let links = Self::merge_links(pool, &from.meeting_id).await?;
        let from_root = resolve_root(&links, from_id);
        let into_root = resolve_root(&links, into_id);
        if from_root == into_root {
            return Err(SpeakerError::SelfMerge);
        }

        let mut tx = pool.begin().await?;
        // Re-point everything that already pointed at `from_root` to keep chains short.
        sqlx::query("UPDATE speakers SET merged_into = ? WHERE merged_into = ?")
            .bind(&into_root)
            .bind(&from_root)
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE speakers SET merged_into = ? WHERE id = ?")
            .bind(&into_root)
            .bind(&from_root)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(into_root)
    }

    /// Manually assigns `speaker_id` (or clears the override with `None`) for the given
    /// segments of a meeting. Returns the ids of the `speaker_edits` rows created, one per
    /// segment actually changed (usable with `undo_edit`).
    pub async fn assign_segment_speaker(
        pool: &SqlitePool,
        meeting_id: &str,
        transcript_ids: &[String],
        speaker_id: Option<&str>,
    ) -> Result<Vec<String>, SpeakerError> {
        let target: Option<String> = match speaker_id {
            Some(id) => {
                let sp = Self::get_speaker(pool, id).await?;
                if sp.meeting_id != meeting_id {
                    return Err(SpeakerError::WrongMeeting(id.to_string()));
                }
                let links = Self::merge_links(pool, meeting_id).await?;
                Some(resolve_root(&links, id))
            }
            None => None,
        };

        let mut tx = pool.begin().await?;
        let now = Utc::now().to_rfc3339();
        let mut edit_ids: Vec<String> = Vec::new();
        for tid in transcript_ids {
            let row: Option<(Option<String>,)> = sqlx::query_as(
                "SELECT manual_speaker_id FROM transcripts WHERE id = ? AND meeting_id = ?",
            )
            .bind(tid)
            .bind(meeting_id)
            .fetch_optional(&mut *tx)
            .await?;
            let Some((old,)) = row else { continue };
            if old == target {
                continue;
            }
            sqlx::query("UPDATE transcripts SET manual_speaker_id = ? WHERE id = ?")
                .bind(&target)
                .bind(tid)
                .execute(&mut *tx)
                .await?;
            let edit_id = Uuid::new_v4().to_string();
            sqlx::query(
                "INSERT INTO speaker_edits (id, meeting_id, transcript_id, old_speaker, new_speaker, edited_at) VALUES (?, ?, ?, ?, ?, ?)",
            )
            .bind(&edit_id)
            .bind(meeting_id)
            .bind(tid)
            .bind(&old)
            .bind(&target)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
            edit_ids.push(edit_id);
        }
        tx.commit().await?;
        Ok(edit_ids)
    }

    /// Writes the automatic (diarization) result. Never touches `manual_speaker_id`.
    pub async fn set_auto_speaker(
        pool: &SqlitePool,
        transcript_id: &str,
        speaker_id: Option<&str>,
        confidence: Option<f64>,
    ) -> Result<(), SpeakerError> {
        sqlx::query("UPDATE transcripts SET auto_speaker_id = ?, speaker_confidence = ? WHERE id = ?")
            .bind(speaker_id)
            .bind(confidence)
            .bind(transcript_id)
            .execute(pool)
            .await?;
        Ok(())
    }

    /// Displayed speaker of every segment of a meeting (for the transcript view).
    pub async fn list_segment_speakers(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<Vec<SegmentSpeaker>, SpeakerError> {
        let speakers: Vec<Speaker> = sqlx::query_as(
            "SELECT id, meeting_id, label, name, color, created_at, merged_into, kind FROM speakers WHERE meeting_id = ?",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await?;
        let links: HashMap<String, Option<String>> = speakers
            .iter()
            .map(|s| (s.id.clone(), s.merged_into.clone()))
            .collect();
        let by_id: HashMap<&str, &Speaker> = speakers.iter().map(|s| (s.id.as_str(), s)).collect();

        let rows: Vec<(String, Option<String>, Option<String>, Option<f64>)> = sqlx::query_as(
            "SELECT id, manual_speaker_id, auto_speaker_id, speaker_confidence FROM transcripts WHERE meeting_id = ?",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|(tid, manual, auto, conf)| {
                let is_manual = manual.is_some();
                let root = manual.or(auto).map(|id| resolve_root(&links, &id));
                let sp = root.as_deref().and_then(|id| by_id.get(id));
                SegmentSpeaker {
                    transcript_id: tid,
                    display_name: sp.map(|s| s.name.clone().unwrap_or_else(|| s.label.clone())),
                    color: sp.and_then(|s| s.color.clone()),
                    speaker_id: root,
                    is_manual,
                    // confidence refers to the automatic decision only
                    confidence: if is_manual { None } else { conf },
                }
            })
            .collect())
    }

    /// Writes the result of an offline (re)diarization in one transaction.
    ///
    /// Only `auto_speaker_id` / `speaker_confidence` are written — manual overrides are never touched.
    /// Segments not listed keep their current automatic speaker. `centroids` (per offline cluster) are
    /// stored on new speakers and refreshed on existing ones. Unnamed person speakers that end up with
    /// no segment at all (and are not referenced by merges or the edit history) are removed.
    pub async fn apply_offline_assignments(
        pool: &SqlitePool,
        meeting_id: &str,
        assignments: &[OfflineAssignment],
        existing_centroids: &HashMap<String, Vec<f32>>,
        new_centroids: &HashMap<usize, Vec<f32>>,
    ) -> Result<OfflineApplyResult, SpeakerError> {
        let mut tx = pool.begin().await?;
        let mut result = OfflineApplyResult::default();

        // Existing targets must be active speakers of this meeting.
        let known: HashMap<String, (String, Option<String>)> =
            sqlx::query_as::<_, (String, String, Option<String>)>("SELECT id, kind, merged_into FROM speakers WHERE meeting_id = ?")
                .bind(meeting_id)
                .fetch_all(&mut *tx)
                .await?
                .into_iter()
                .map(|(id, kind, merged)| (id, (kind, merged)))
                .collect();
        for a in assignments {
            if let OfflineTarget::Existing(id) = &a.target {
                match known.get(id) {
                    None => return Err(SpeakerError::WrongMeeting(id.clone())),
                    Some((kind, _)) if kind == KIND_ROOM => return Err(SpeakerError::RoomSpeaker),
                    _ => {}
                }
            }
        }

        // Create new speakers (in cluster order, so numbering follows first appearance).
        let mut new_clusters: Vec<usize> = assignments
            .iter()
            .filter_map(|a| if let OfflineTarget::New(c) = a.target { Some(c) } else { None })
            .collect();
        new_clusters.sort_unstable();
        new_clusters.dedup();
        let mut new_ids: HashMap<usize, String> = HashMap::new();
        for cluster in new_clusters {
            let n = next_person_number(&mut *tx, meeting_id).await?;
            let id = Uuid::new_v4().to_string();
            sqlx::query("INSERT INTO speakers (id, meeting_id, label, color, centroid, created_at, kind) VALUES (?, ?, ?, ?, ?, ?, 'person')")
                .bind(&id)
                .bind(meeting_id)
                .bind(format!("Mówca {}", n))
                .bind(SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.len()])
                .bind(new_centroids.get(&cluster).map(|c| centroid_bytes(c)))
                .bind(Utc::now().to_rfc3339())
                .execute(&mut *tx)
                .await?;
            new_ids.insert(cluster, id);
            result.created_speakers += 1;
        }
        for (id, centroid) in existing_centroids {
            sqlx::query("UPDATE speakers SET centroid = ? WHERE id = ? AND meeting_id = ?")
                .bind(centroid_bytes(centroid))
                .bind(id)
                .bind(meeting_id)
                .execute(&mut *tx)
                .await?;
        }
        let room_id = if assignments.iter().any(|a| a.target == OfflineTarget::Room) {
            Some(ensure_room_speaker(&mut *tx, meeting_id).await?)
        } else {
            None
        };

        let current: HashMap<String, Option<String>> =
            sqlx::query_as::<_, (String, Option<String>)>("SELECT id, auto_speaker_id FROM transcripts WHERE meeting_id = ?")
                .bind(meeting_id)
                .fetch_all(&mut *tx)
                .await?
                .into_iter()
                .collect();
        for a in assignments {
            let Some(old) = current.get(&a.transcript_id) else { continue };
            let new = match &a.target {
                OfflineTarget::Existing(id) => Some(id.clone()),
                OfflineTarget::New(c) => new_ids.get(c).cloned(),
                OfflineTarget::Room => room_id.clone(),
            };
            if new.is_none() {
                continue;
            }
            sqlx::query("UPDATE transcripts SET auto_speaker_id = ?, speaker_confidence = ? WHERE id = ? AND meeting_id = ?")
                .bind(&new)
                .bind(a.confidence)
                .bind(&a.transcript_id)
                .bind(meeting_id)
                .execute(&mut *tx)
                .await?;
            // "changed" = the speaker the automatic decision points to (after merges) is different
            let resolve = |id: &Option<String>| -> Option<String> {
                let mut cur = id.clone()?;
                for _ in 0..MAX_MERGE_DEPTH {
                    match known.get(&cur).and_then(|(_, m)| m.clone()) {
                        Some(next) if next != cur => cur = next,
                        _ => break,
                    }
                }
                Some(cur)
            };
            if resolve(old) != resolve(&new) {
                result.changed_transcript_ids.push(a.transcript_id.clone());
            }
        }

        let removed = sqlx::query(
            "DELETE FROM speakers WHERE meeting_id = ?1 AND kind = 'person' AND name IS NULL AND merged_into IS NULL
               AND id NOT IN (SELECT auto_speaker_id FROM transcripts WHERE meeting_id = ?1 AND auto_speaker_id IS NOT NULL)
               AND id NOT IN (SELECT manual_speaker_id FROM transcripts WHERE meeting_id = ?1 AND manual_speaker_id IS NOT NULL)
               AND id NOT IN (SELECT merged_into FROM speakers WHERE meeting_id = ?1 AND merged_into IS NOT NULL)
               AND id NOT IN (SELECT old_speaker FROM speaker_edits WHERE meeting_id = ?1 AND old_speaker IS NOT NULL)
               AND id NOT IN (SELECT new_speaker FROM speaker_edits WHERE meeting_id = ?1 AND new_speaker IS NOT NULL)",
        )
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
        result.removed_speakers = removed.rows_affected() as usize;

        tx.commit().await?;
        Ok(result)
    }

    /// Segment timing and current speakers of a meeting, ordered by time (input for the offline pass).
    pub async fn list_segments_for_diarization(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<Vec<DiarizationSegmentRow>, SpeakerError> {
        Ok(sqlx::query_as::<_, DiarizationSegmentRow>(
            "SELECT id, audio_start_time, audio_end_time, duration, auto_speaker_id, manual_speaker_id
             FROM transcripts WHERE meeting_id = ? ORDER BY audio_start_time ASC, rowid ASC",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await?)
    }

    /// Displayed *person* speaker (manual override, else automatic; resolved through merges) of each
    /// given row; `None` for unassigned and room segments.
    pub async fn displayed_person_speakers(
        pool: &SqlitePool,
        meeting_id: &str,
        rows: &[DiarizationSegmentRow],
    ) -> Result<Vec<Option<String>>, SpeakerError> {
        let speakers: Vec<(String, Option<String>, String)> =
            sqlx::query_as("SELECT id, merged_into, kind FROM speakers WHERE meeting_id = ?")
                .bind(meeting_id)
                .fetch_all(pool)
                .await?;
        let links: HashMap<String, Option<String>> = speakers.iter().map(|(id, m, _)| (id.clone(), m.clone())).collect();
        let kinds: HashMap<String, String> = speakers.into_iter().map(|(id, _, k)| (id, k)).collect();
        Ok(rows
            .iter()
            .map(|r| {
                let root = r.manual_speaker_id.clone().or_else(|| r.auto_speaker_id.clone()).map(|id| resolve_root(&links, &id))?;
                (kinds.get(&root).map(String::as_str) == Some(KIND_PERSON)).then_some(root)
            })
            .collect())
    }

    /// Returns the meeting's reserved "Sala" speaker, creating it when missing.
    pub async fn get_or_create_room_speaker(pool: &SqlitePool, meeting_id: &str) -> Result<Speaker, SpeakerError> {
        let mut conn = pool.acquire().await?;
        let id = ensure_room_speaker(&mut conn, meeting_id).await?;
        drop(conn);
        Self::get_speaker(pool, &id).await
    }

    /// Undoes one manual edit, unless the segment has been changed again since.
    pub async fn undo_edit(pool: &SqlitePool, edit_id: &str) -> Result<(), SpeakerError> {
        let mut tx = pool.begin().await?;
        let edit: SpeakerEdit = sqlx::query_as(
            "SELECT id, meeting_id, transcript_id, old_speaker, new_speaker, edited_at, undone FROM speaker_edits WHERE id = ? AND undone = 0",
        )
        .bind(edit_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| SpeakerError::EditNotUndoable(edit_id.to_string()))?;

        let current: Option<(Option<String>,)> =
            sqlx::query_as("SELECT manual_speaker_id FROM transcripts WHERE id = ?")
                .bind(&edit.transcript_id)
                .fetch_optional(&mut *tx)
                .await?;
        match current {
            Some((cur,)) if cur == edit.new_speaker => {}
            _ => return Err(SpeakerError::UndoConflict(edit.transcript_id)),
        }
        sqlx::query("UPDATE transcripts SET manual_speaker_id = ? WHERE id = ?")
            .bind(&edit.old_speaker)
            .bind(&edit.transcript_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE speaker_edits SET undone = 1 WHERE id = ?")
            .bind(edit_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use sqlx::SqlitePool;

    async fn setup() -> SqlitePool {
        let pool = SqlitePoolOptions::new().max_connections(1).connect("sqlite::memory:").await.unwrap();
        sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        sqlx::query("INSERT INTO meetings (id,title,created_at,updated_at) VALUES ('m1','t','x','x'),('m2','t','x','x')").execute(&pool).await.unwrap();
        for (i, d) in [(1, 2.0), (2, 3.0), (3, 1.0), (4, 4.0)] {
            sqlx::query("INSERT INTO transcripts (id,meeting_id,transcript,timestamp,duration) VALUES (?, 'm1', 'txt', 'ts', ?)")
                .bind(format!("t{i}")).bind(d).execute(&pool).await.unwrap();
        }
        pool
    }

    #[tokio::test]
    async fn manual_override_wins_and_survives_auto_rerun() {
        let pool = setup().await;
        let a = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        let b = SpeakersRepository::create_speaker(&pool, "m1", None, Some(&[0.1, 0.2])).await.unwrap();
        assert_eq!(a.label, "Mówca 1");
        assert_eq!(b.label, "Mówca 2");
        for t in ["t1", "t2", "t3", "t4"] {
            SpeakersRepository::set_auto_speaker(&pool, t, Some(&a.id), Some(0.9)).await.unwrap();
        }
        let n = SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t2".into(), "t3".into()], Some(&b.id)).await.unwrap();
        assert_eq!(n.len(), 2);
        // re-running diarization changes only auto
        for t in ["t1", "t2", "t3", "t4"] {
            SpeakersRepository::set_auto_speaker(&pool, t, Some(&a.id), Some(0.5)).await.unwrap();
        }
        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        let get = |id: &str| segs.iter().find(|s| s.transcript_id == id).unwrap().clone();
        assert_eq!(get("t1").speaker_id.as_deref(), Some(a.id.as_str()));
        assert_eq!(get("t2").speaker_id.as_deref(), Some(b.id.as_str()));
        assert!(get("t2").is_manual && get("t2").confidence.is_none());
        assert_eq!(get("t1").confidence, Some(0.5));
        let list = SpeakersRepository::list_speakers(&pool, "m1").await.unwrap();
        assert_eq!(list[0].utterance_count, 2);
        assert_eq!(list[0].total_duration, 6.0);
        assert_eq!(list[1].utterance_count, 2);
        assert_eq!(list[1].total_duration, 4.0);
    }

    #[tokio::test]
    async fn rename_applies_everywhere() {
        let pool = setup().await;
        let a = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        SpeakersRepository::set_auto_speaker(&pool, "t1", Some(&a.id), None).await.unwrap();
        SpeakersRepository::rename_speaker(&pool, &a.id, Some("  Anna ")).await.unwrap();
        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        assert_eq!(segs.iter().find(|s| s.transcript_id == "t1").unwrap().display_name.as_deref(), Some("Anna"));
        SpeakersRepository::rename_speaker(&pool, &a.id, Some("   ")).await.unwrap();
        let list = SpeakersRepository::list_speakers(&pool, "m1").await.unwrap();
        assert_eq!(list[0].display_name, "Mówca 1");
        assert!(SpeakersRepository::rename_speaker(&pool, "nope", Some("x")).await.is_err());
    }

    #[tokio::test]
    async fn merge_and_chains() {
        let pool = setup().await;
        let a = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        let b = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        let c = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        SpeakersRepository::set_auto_speaker(&pool, "t1", Some(&a.id), None).await.unwrap();
        SpeakersRepository::set_auto_speaker(&pool, "t2", Some(&b.id), None).await.unwrap();
        SpeakersRepository::set_auto_speaker(&pool, "t3", Some(&c.id), None).await.unwrap();
        assert!(matches!(SpeakersRepository::merge_speakers(&pool, &a.id, &a.id).await, Err(SpeakerError::SelfMerge)));
        SpeakersRepository::merge_speakers(&pool, &a.id, &b.id).await.unwrap(); // a -> b
        let survivor = SpeakersRepository::merge_speakers(&pool, &b.id, &c.id).await.unwrap(); // b -> c (a repointed)
        assert_eq!(survivor, c.id);
        assert!(matches!(SpeakersRepository::merge_speakers(&pool, &a.id, &c.id).await, Err(SpeakerError::SelfMerge)));
        let list = SpeakersRepository::list_speakers(&pool, "m1").await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].utterance_count, 3);
        // assigning to a merged speaker resolves to the root
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t4".into()], Some(&a.id)).await.unwrap();
        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        assert_eq!(segs.iter().find(|s| s.transcript_id == "t4").unwrap().speaker_id.as_deref(), Some(c.id.as_str()));
    }

    #[tokio::test]
    async fn cross_meeting_rejected() {
        let pool = setup().await;
        let a = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        let x = SpeakersRepository::create_speaker(&pool, "m2", None, None).await.unwrap();
        assert!(matches!(SpeakersRepository::merge_speakers(&pool, &a.id, &x.id).await, Err(SpeakerError::WrongMeeting(_))));
        assert!(matches!(SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&x.id)).await, Err(SpeakerError::WrongMeeting(_))));
    }

    #[tokio::test]
    async fn undo_and_clear() {
        let pool = setup().await;
        let a = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        let b = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&a.id)).await.unwrap();
        // no-op assignment records nothing
        assert!(SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&a.id)).await.unwrap().is_empty());
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&b.id)).await.unwrap();
        let edits: Vec<(String, Option<String>)> = sqlx::query_as("SELECT id, new_speaker FROM speaker_edits ORDER BY edited_at, rowid").fetch_all(&pool).await.unwrap();
        assert_eq!(edits.len(), 2);
        // first edit can't be undone (segment changed since)
        assert!(matches!(SpeakersRepository::undo_edit(&pool, &edits[0].0).await, Err(SpeakerError::UndoConflict(_))));
        SpeakersRepository::undo_edit(&pool, &edits[1].0).await.unwrap();
        SpeakersRepository::undo_edit(&pool, &edits[0].0).await.unwrap();
        assert!(SpeakersRepository::undo_edit(&pool, &edits[0].0).await.is_err());
        // clear override
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&a.id)).await.unwrap();
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], None).await.unwrap();
        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        assert!(segs.iter().find(|s| s.transcript_id == "t1").unwrap().speaker_id.is_none());
    }

    #[tokio::test]
    async fn deleting_meeting_cascades() {
        let pool = setup().await;
        SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        sqlx::query("DELETE FROM meetings WHERE id='m1'").execute(&pool).await.unwrap();
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM speakers").fetch_one(&pool).await.unwrap();
        assert_eq!(n, 0);
    }
    #[tokio::test]
    async fn persists_live_speakers_and_room() {
        let pool = setup().await;
        let inputs = vec![
            AutoSpeakerInput { transcript_id: "t1".into(), speaker_index: Some(0), is_room: false, confidence: Some(0.9), ..Default::default() },
            AutoSpeakerInput { transcript_id: "t2".into(), speaker_index: Some(2), is_room: false, confidence: Some(0.8), ..Default::default() },
            AutoSpeakerInput { transcript_id: "t3".into(), speaker_index: None, is_room: true, confidence: Some(0.5), ..Default::default() },
            AutoSpeakerInput { transcript_id: "t4".into(), speaker_index: None, is_room: false, confidence: None, ..Default::default() }, // unassigned
        ];
        let mut conn = pool.acquire().await.unwrap();
        persist_auto_speakers(&mut conn, "m1", &inputs).await.unwrap();
        drop(conn);

        let list = SpeakersRepository::list_speakers(&pool, "m1").await.unwrap();
        let names: Vec<(&str, &str)> = list.iter().map(|s| (s.display_name.as_str(), s.kind.as_str())).collect();
        assert!(names.contains(&("Mówca 1", "person")) && names.contains(&("Mówca 3", "person")) && names.contains(&("Sala", "room")));
        assert_eq!(list.len(), 3); // no "Mówca 2": index 1 was never used

        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        let name_of = |id: &str| segs.iter().find(|s| s.transcript_id == id).unwrap().display_name.clone();
        assert_eq!(name_of("t1").as_deref(), Some("Mówca 1"));
        assert_eq!(name_of("t2").as_deref(), Some("Mówca 3"));
        assert_eq!(name_of("t3").as_deref(), Some("Sala"));
        assert_eq!(name_of("t4"), None);
        assert_eq!(segs.iter().find(|s| s.transcript_id == "t1").unwrap().confidence, Some(0.9));
        // persisting twice for the same indices in another meeting is independent
        let mut conn = pool.acquire().await.unwrap();
        persist_auto_speakers(&mut conn, "m2", &inputs[..1]).await.unwrap();
    }

    #[tokio::test]
    async fn live_names_and_manual_overrides_are_persisted() {
        let pool = setup().await;
        let inputs = vec![
            AutoSpeakerInput { transcript_id: "t1".into(), speaker_index: Some(0), speaker_name: Some("Anna".into()), ..Default::default() },
            AutoSpeakerInput { transcript_id: "t2".into(), speaker_index: Some(0), speaker_name: Some("Anna".into()), manual_index: Some(1), manual_name: Some("Piotr".into()), ..Default::default() },
            AutoSpeakerInput { transcript_id: "t3".into(), speaker_index: Some(0), speaker_name: Some("Anna".into()), manual_is_room: true, ..Default::default() },
        ];
        let mut conn = pool.acquire().await.unwrap();
        persist_auto_speakers(&mut conn, "m1", &inputs).await.unwrap();
        drop(conn);
        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        let get = |id: &str| segs.iter().find(|s| s.transcript_id == id).unwrap().clone();
        assert_eq!(get("t1").display_name.as_deref(), Some("Anna"));
        assert!(!get("t1").is_manual);
        assert_eq!(get("t2").display_name.as_deref(), Some("Piotr"));
        assert!(get("t2").is_manual);
        assert_eq!(get("t3").display_name.as_deref(), Some("Sala"));
    }

    #[tokio::test]
    async fn room_speaker_is_unique_and_protected() {
        let pool = setup().await;
        let r1 = SpeakersRepository::get_or_create_room_speaker(&pool, "m1").await.unwrap();
        let r2 = SpeakersRepository::get_or_create_room_speaker(&pool, "m1").await.unwrap();
        assert_eq!(r1.id, r2.id);
        assert_eq!(r1.label, "Sala");
        // person numbering ignores the room speaker
        let p = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        assert_eq!(p.label, "Mówca 1");
        assert!(matches!(SpeakersRepository::rename_speaker(&pool, &r1.id, Some("X")).await, Err(SpeakerError::RoomSpeaker)));
        assert!(matches!(SpeakersRepository::merge_speakers(&pool, &p.id, &r1.id).await, Err(SpeakerError::RoomSpeaker)));
        // but a segment can be manually assigned to the room
        let n = SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&r1.id)).await.unwrap();
        assert_eq!(n.len(), 1);
    }

    #[tokio::test]
    async fn create_speaker_numbering_skips_used_labels() {
        let pool = setup().await;
        let inputs = vec![
            AutoSpeakerInput { transcript_id: "t1".into(), speaker_index: Some(0), ..Default::default() },
            AutoSpeakerInput { transcript_id: "t2".into(), speaker_index: Some(2), ..Default::default() },
        ];
        let mut conn = pool.acquire().await.unwrap();
        persist_auto_speakers(&mut conn, "m1", &inputs).await.unwrap(); // "Mówca 1", "Mówca 3"
        drop(conn);
        let s = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        assert_eq!(s.label, "Mówca 4");
    }

    #[tokio::test]
    async fn offline_pass_rewrites_only_automatic_speakers() {
        let pool = setup().await;
        let anna = SpeakersRepository::create_speaker(&pool, "m1", Some("Anna"), None).await.unwrap(); // Mówca 1
        let stray = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap(); // Mówca 2, unnamed
        let piotr = SpeakersRepository::create_speaker(&pool, "m1", Some("Piotr"), None).await.unwrap(); // Mówca 3
        for t in ["t1", "t2", "t3"] {
            SpeakersRepository::set_auto_speaker(&pool, t, Some(&anna.id), Some(0.9)).await.unwrap();
        }
        SpeakersRepository::set_auto_speaker(&pool, "t4", Some(&stray.id), Some(0.6)).await.unwrap();
        // user corrected t3 to Piotr
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t3".into()], Some(&piotr.id)).await.unwrap();

        let assignments = vec![
            OfflineAssignment { transcript_id: "t1".into(), target: OfflineTarget::Existing(anna.id.clone()), confidence: 0.95 },
            OfflineAssignment { transcript_id: "t2".into(), target: OfflineTarget::New(1), confidence: 0.8 },
            OfflineAssignment { transcript_id: "t3".into(), target: OfflineTarget::Existing(anna.id.clone()), confidence: 0.7 },
            OfflineAssignment { transcript_id: "t4".into(), target: OfflineTarget::Room, confidence: 0.5 },
            OfflineAssignment { transcript_id: "unknown".into(), target: OfflineTarget::Room, confidence: 0.5 },
        ];
        let mut existing = HashMap::new();
        existing.insert(anna.id.clone(), vec![1.0f32, 0.0]);
        let mut new = HashMap::new();
        new.insert(1usize, vec![0.0f32, 1.0]);
        let res = SpeakersRepository::apply_offline_assignments(&pool, "m1", &assignments, &existing, &new).await.unwrap();
        assert_eq!(res.created_speakers, 1);
        assert_eq!(res.removed_speakers, 1, "unnamed speaker with no segments is removed");
        let mut changed = res.changed_transcript_ids.clone();
        changed.sort();
        assert_eq!(changed, vec!["t2".to_string(), "t4".to_string()]);

        let segs = SpeakersRepository::list_segment_speakers(&pool, "m1").await.unwrap();
        let get = |id: &str| segs.iter().find(|s| s.transcript_id == id).unwrap().clone();
        assert_eq!(get("t1").display_name.as_deref(), Some("Anna"));
        assert_eq!(get("t1").confidence, Some(0.95));
        assert_eq!(get("t2").display_name.as_deref(), Some("Mówca 4"), "new speaker numbered after the highest label");
        assert_eq!(get("t3").display_name.as_deref(), Some("Piotr"), "manual correction survives");
        assert!(get("t3").is_manual);
        assert_eq!(get("t4").display_name.as_deref(), Some("Sala"));
        let auto_t3: Option<String> = sqlx::query_scalar("SELECT auto_speaker_id FROM transcripts WHERE id='t3'").fetch_one(&pool).await.unwrap();
        assert_eq!(auto_t3.as_deref(), Some(anna.id.as_str()));

        // named speaker without automatic segments (Piotr) is kept
        let list = SpeakersRepository::list_speakers(&pool, "m1").await.unwrap();
        assert!(list.iter().any(|s| s.id == piotr.id));
        assert!(!list.iter().any(|s| s.id == stray.id));
        let centroid: Option<Vec<u8>> = sqlx::query_scalar("SELECT centroid FROM speakers WHERE id = ?").bind(&anna.id).fetch_one(&pool).await.unwrap();
        assert_eq!(centroid.map(|c| c.len()), Some(8));

        // running again with the same result changes nothing
        let rows = SpeakersRepository::list_segments_for_diarization(&pool, "m1").await.unwrap();
        let shown = SpeakersRepository::displayed_person_speakers(&pool, "m1", &rows).await.unwrap();
        assert_eq!(rows.len(), 4);
        let t4 = rows.iter().position(|r| r.id == "t4").unwrap();
        assert_eq!(shown[t4], None, "room is not a person");
        let t3 = rows.iter().position(|r| r.id == "t3").unwrap();
        assert_eq!(shown[t3].as_deref(), Some(piotr.id.as_str()));
    }

    #[tokio::test]
    async fn offline_pass_rejects_foreign_or_room_targets_and_keeps_referenced_speakers() {
        let pool = setup().await;
        let other = SpeakersRepository::create_speaker(&pool, "m2", None, None).await.unwrap();
        let a = vec![OfflineAssignment { transcript_id: "t1".into(), target: OfflineTarget::Existing(other.id.clone()), confidence: 1.0 }];
        assert!(matches!(
            SpeakersRepository::apply_offline_assignments(&pool, "m1", &a, &HashMap::new(), &HashMap::new()).await,
            Err(SpeakerError::WrongMeeting(_))
        ));
        let room = SpeakersRepository::get_or_create_room_speaker(&pool, "m1").await.unwrap();
        let a = vec![OfflineAssignment { transcript_id: "t1".into(), target: OfflineTarget::Existing(room.id.clone()), confidence: 1.0 }];
        assert!(matches!(
            SpeakersRepository::apply_offline_assignments(&pool, "m1", &a, &HashMap::new(), &HashMap::new()).await,
            Err(SpeakerError::RoomSpeaker)
        ));
        // an unnamed speaker referenced by the edit history must not be deleted (undo would break)
        let x = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        let y = SpeakersRepository::create_speaker(&pool, "m1", None, None).await.unwrap();
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&x.id)).await.unwrap();
        SpeakersRepository::assign_segment_speaker(&pool, "m1", &["t1".into()], Some(&y.id)).await.unwrap();
        let res = SpeakersRepository::apply_offline_assignments(&pool, "m1", &[], &HashMap::new(), &HashMap::new()).await.unwrap();
        assert_eq!(res.removed_speakers, 0);
        let edits: Vec<String> = sqlx::query_scalar("SELECT id FROM speaker_edits ORDER BY edited_at DESC, rowid DESC").fetch_all(&pool).await.unwrap();
        SpeakersRepository::undo_edit(&pool, &edits[0]).await.unwrap();
    }
}
