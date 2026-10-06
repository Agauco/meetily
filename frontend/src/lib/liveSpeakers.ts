import type { Transcript } from '@/types';
import type { SegmentSpeakerInfo, SpeakerSummary } from '@/types/speakers';

// Live (pre-save) speaker state. It lives directly on the transcript objects so that it flows into
// `api_save_transcript` unchanged: automatic result (speaker_index / speaker_is_room), the name the user
// gave the speaker (speaker_name), and manual per-segment corrections (manual_speaker_*).

export const ROOM_ID = 'room';
export const ROOM_LABEL = 'Sala';
export const ROOM_COLOR = '#6B7280';
export const SPEAKER_COLORS = [
  '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#84CC16',
];

export const personId = (index: number) => `p${index}`;
const parsePersonId = (id: string): number | null => (/^p\d+$/.test(id) ? Number(id.slice(1)) : null);
const colorOf = (index: number) => SPEAKER_COLORS[index % SPEAKER_COLORS.length];

/** Effective speaker id of a segment: manual correction wins over the automatic result. */
export function effectiveSpeakerId(t: Transcript): string | undefined {
  if (t.manual_speaker_is_room) return ROOM_ID;
  if (t.manual_speaker_index != null) return personId(t.manual_speaker_index);
  if (t.speaker_is_room) return ROOM_ID;
  if (t.speaker_index != null) return personId(t.speaker_index);
  return undefined;
}

/** Name the user gave to a person index (looked up across all segments), if any. */
export function nameOfIndex(transcripts: Transcript[], index: number): string | undefined {
  for (const t of transcripts) {
    if (t.speaker_index === index && t.speaker_name) return t.speaker_name;
    if (t.manual_speaker_index === index && t.manual_speaker_name) return t.manual_speaker_name;
  }
  return undefined;
}

export function buildLiveSpeakers(transcripts: Transcript[]): {
  speakers: SpeakerSummary[];
  segmentSpeakers: Record<string, SegmentSpeakerInfo>;
} {
  const stats = new Map<string, { count: number; duration: number }>();
  const segmentSpeakers: Record<string, SegmentSpeakerInfo> = {};
  const names = new Map<number, string | undefined>();
  const nameFor = (index: number) => {
    if (!names.has(index)) names.set(index, nameOfIndex(transcripts, index));
    return names.get(index);
  };
  for (const t of transcripts) {
    const id = effectiveSpeakerId(t);
    if (!id) continue;
    const s = stats.get(id) ?? { count: 0, duration: 0 };
    s.count += 1;
    s.duration += t.duration ?? 0;
    stats.set(id, s);
    const index = parsePersonId(id);
    const isManual = t.manual_speaker_is_room || t.manual_speaker_index != null;
    segmentSpeakers[t.id] = {
      speakerId: id,
      name: index === null ? ROOM_LABEL : nameFor(index) ?? `Mówca ${index + 1}`,
      color: index === null ? ROOM_COLOR : colorOf(index),
      isManual: !!isManual,
      confidence: t.speaker_confidence ?? undefined,
    };
  }
  const speakers: SpeakerSummary[] = [...stats.entries()]
    .map(([id, s]) => {
      const index = parsePersonId(id);
      const label = index === null ? ROOM_LABEL : `Mówca ${index + 1}`;
      const name = index === null ? null : nameFor(index) ?? null;
      return {
        id,
        label,
        name,
        display_name: name ?? label,
        color: index === null ? ROOM_COLOR : colorOf(index),
        kind: (index === null ? 'room' : 'person') as 'room' | 'person',
        utterance_count: s.count,
        total_duration: s.duration,
        _order: index ?? 1e9,
      };
    })
    .sort((a, b) => a._order - b._order)
    .map(({ _order, ...rest }) => rest);
  return { speakers, segmentSpeakers };
}

/** Sets (or clears, with null/empty) the name of a person speaker on every segment that references it. */
export function renameLiveSpeaker(transcripts: Transcript[], speakerId: string, name: string | null): Transcript[] {
  const index = parsePersonId(speakerId);
  if (index === null) return transcripts;
  const clean = name?.trim() || undefined;
  return transcripts.map((t) => {
    let next = t;
    if (t.speaker_index === index) next = { ...next, speaker_name: clean };
    if (t.manual_speaker_index === index) next = { ...next, manual_speaker_name: clean };
    return next;
  });
}

/** Manually assigns segments to a speaker; `null` clears their manual correction. */
export function assignLiveSegments(transcripts: Transcript[], ids: string[], speakerId: string | null): Transcript[] {
  const set = new Set(ids);
  const index = speakerId ? parsePersonId(speakerId) : null;
  const name = index !== null ? nameOfIndex(transcripts, index) : undefined;
  return transcripts.map((t) => {
    if (!set.has(t.id)) return t;
    if (speakerId === null) {
      return { ...t, manual_speaker_index: undefined, manual_speaker_is_room: false, manual_speaker_name: undefined };
    }
    if (speakerId === ROOM_ID) {
      return { ...t, manual_speaker_index: undefined, manual_speaker_is_room: true, manual_speaker_name: undefined };
    }
    if (index === null) return t;
    return { ...t, manual_speaker_index: index, manual_speaker_is_room: false, manual_speaker_name: name };
  });
}

export function nextSpeakerIndex(transcripts: Transcript[]): number {
  let max = -1;
  for (const t of transcripts) {
    if (t.speaker_index != null) max = Math.max(max, t.speaker_index);
    if (t.manual_speaker_index != null) max = Math.max(max, t.manual_speaker_index);
  }
  return max + 1;
}

/** Creates a new person speaker (optionally named) and assigns the segments to it. */
export function createAndAssignLive(transcripts: Transcript[], ids: string[], name?: string): Transcript[] {
  const index = nextSpeakerIndex(transcripts);
  const set = new Set(ids);
  const clean = name?.trim() || undefined;
  return transcripts.map((t) =>
    set.has(t.id)
      ? { ...t, manual_speaker_index: index, manual_speaker_is_room: false, manual_speaker_name: clean }
      : t
  );
}

/** Moves every segment currently displayed as `fromId` to `intoId` (as manual corrections). */
export function mergeLiveSpeakers(transcripts: Transcript[], fromId: string, intoId: string): Transcript[] {
  const ids = transcripts.filter((t) => effectiveSpeakerId(t) === fromId).map((t) => t.id);
  return assignLiveSegments(transcripts, ids, intoId);
}

/** Newly arrived segments inherit the name the user already gave to their (automatic) speaker. */
export function inheritSpeakerNames(existing: Transcript[], incoming: Transcript[]): Transcript[] {
  return incoming.map((t) => {
    if (t.speaker_index == null || t.speaker_name) return t;
    const name = nameOfIndex(existing, t.speaker_index);
    return name ? { ...t, speaker_name: name } : t;
  });
}
