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
            "SELECT id, meeting_id, label, name, color, created_at, merged_into FROM speakers WHERE id = ?",
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
        let existing: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM speakers WHERE meeting_id = ?")
            .bind(meeting_id)
            .fetch_one(pool)
            .await?;
        let n = existing as usize + 1;
        let speaker = Speaker {
            id: Uuid::new_v4().to_string(),
            meeting_id: meeting_id.to_string(),
            label: format!("Mówca {}", n),
            name: name.map(str::trim).filter(|s| !s.is_empty()).map(String::from),
            color: Some(SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.len()].to_string()),
            created_at: Utc::now().to_rfc3339(),
            merged_into: None,
        };
        let centroid_bytes: Option<Vec<u8>> =
            centroid.map(|c| c.iter().flat_map(|v| v.to_le_bytes()).collect());
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
            "SELECT id, meeting_id, label, name, color, created_at, merged_into FROM speakers WHERE meeting_id = ? ORDER BY created_at, id",
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
            "SELECT id, meeting_id, label, name, color, created_at, merged_into FROM speakers WHERE meeting_id = ?",
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
}
