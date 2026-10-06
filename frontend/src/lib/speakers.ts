import type { SegmentSpeaker, SegmentSpeakerInfo } from '@/types/speakers';

/** Builds the per-segment display info from the backend rows (segments without a speaker are omitted). */
export function buildSegmentSpeakerMap(rows: SegmentSpeaker[]): Record<string, SegmentSpeakerInfo> {
  const map: Record<string, SegmentSpeakerInfo> = {};
  for (const row of rows) {
    if (!row.speaker_id || !row.display_name) continue;
    map[row.transcript_id] = {
      speakerId: row.speaker_id,
      name: row.display_name,
      color: row.color,
      isManual: row.is_manual,
      confidence: row.confidence ?? undefined,
    };
  }
  return map;
}

/**
 * For each segment, how many segments *directly following* it belong to the same speaker
 * (a contiguous "block"). Segments without a speaker have 0.
 */
export function computeFollowingCounts(speakerIds: Array<string | undefined>): number[] {
  const counts = new Array<number>(speakerIds.length).fill(0);
  for (let i = speakerIds.length - 2; i >= 0; i--) {
    const current = speakerIds[i];
    if (current && speakerIds[i + 1] === current) {
      counts[i] = counts[i + 1] + 1;
    }
  }
  return counts;
}

/** Ids of segment `index` plus the next `count` segments (clamped to the list). */
export function blockIds(ids: string[], index: number, count: number): string[] {
  if (index < 0 || index >= ids.length) return [];
  return ids.slice(index, Math.min(ids.length, index + 1 + Math.max(0, count)));
}
