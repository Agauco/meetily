// Voice-based speakers (speaker diarization). Mirrors src-tauri/src/database/repositories/speaker.rs

export interface SpeakerSummary {
  id: string;
  label: string; // automatic label, e.g. "Mówca 1"
  name: string | null; // user-provided name
  display_name: string; // name ?? label
  color: string | null;
  kind: 'person' | 'room'; // the room ("Sala") speaker cannot be renamed or merged
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

/** Result of an offline (re)diarization pass. Mirrors `RediarizeSummary` in src-tauri/src/diarization/rediarize.rs */
export interface RediarizeSummary {
  meeting_id: string;
  segments_total: number;
  segments_analyzed: number;
  speaker_count: number;
  changed_transcript_ids: string[];
  created_speakers: number;
  removed_speakers: number;
}

/** `speaker-diarization-progress` event payload. */
export interface DiarizationProgress {
  meeting_id: string;
  stage: 'decoding' | 'analyzing' | 'saving' | 'done' | 'failed';
  progress: number; // 0..100
  error?: string;
}
