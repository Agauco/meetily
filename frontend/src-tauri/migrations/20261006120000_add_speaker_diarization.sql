-- Migration: speaker diarization (voice-based speaker identification)
-- `transcripts.speaker` keeps its meaning (audio source: 'mic' / 'system').
-- Voice-based speakers live in their own table and columns.

CREATE TABLE IF NOT EXISTS speakers (
    id          TEXT PRIMARY KEY,
    meeting_id  TEXT NOT NULL,
    label       TEXT NOT NULL,              -- automatic label, e.g. "Mówca 1"
    name        TEXT,                       -- user-provided name, NULL = not named
    color       TEXT,                       -- UI color
    centroid    BLOB,                       -- voice embedding (little-endian f32 values), optional
    created_at  TEXT NOT NULL,
    merged_into TEXT,                       -- set when merged into another speaker
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
    FOREIGN KEY (merged_into) REFERENCES speakers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_speakers_meeting ON speakers(meeting_id);

ALTER TABLE transcripts ADD COLUMN auto_speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL;
ALTER TABLE transcripts ADD COLUMN manual_speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL;
ALTER TABLE transcripts ADD COLUMN speaker_confidence REAL;

CREATE INDEX IF NOT EXISTS idx_transcripts_auto_speaker ON transcripts(auto_speaker_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_manual_speaker ON transcripts(manual_speaker_id);

-- History of manual assignment changes (supports undo)
CREATE TABLE IF NOT EXISTS speaker_edits (
    id            TEXT PRIMARY KEY,
    meeting_id    TEXT NOT NULL,
    transcript_id TEXT NOT NULL,
    old_speaker   TEXT,                     -- previous manual_speaker_id (NULL = no override)
    new_speaker   TEXT,                     -- new manual_speaker_id (NULL = override cleared)
    edited_at     TEXT NOT NULL,
    undone        INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_speaker_edits_meeting ON speaker_edits(meeting_id, edited_at);
