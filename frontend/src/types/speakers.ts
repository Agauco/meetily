// Voice-based speakers (speaker diarization). Mirrors src-tauri/src/database/repositories/speaker.rs

export interface SpeakerSummary {
  id: string;
  label: string; // automatic label, e.g. "Mówca 1"
  name: string | null; // user-provided name
  display_name: string; // name ?? label
  color: string | null;
  utterance_count: number;
  total_duration: number;
}

export interface SegmentSpeaker {
  transcript_id: string;
  speaker_id: string | null; // displayed speaker (manual override, else automatic), merges resolved
  display_name: string | null;
  color: string | null;
  is_manual: boolean;
  confidence: number | null; // automatic decision only
}

/** Speaker info attached to a transcript segment for display. */
export interface SegmentSpeakerInfo {
  speakerId: string;
  name: string;
  color: string | null;
  isManual: boolean;
  confidence?: number;
}
