import { describe, expect, test } from 'bun:test';
import { blockIds, buildSegmentSpeakerMap, computeFollowingCounts } from '../../src/lib/speakers';

describe('speakers helpers', () => {
  test('buildSegmentSpeakerMap skips segments without a speaker and maps confidence', () => {
    const map = buildSegmentSpeakerMap([
      { transcript_id: 't1', speaker_id: 's1', display_name: 'Anna', color: '#fff', is_manual: false, confidence: 0.8 },
      { transcript_id: 't2', speaker_id: null, display_name: null, color: null, is_manual: false, confidence: null },
      { transcript_id: 't3', speaker_id: 's2', display_name: 'Mówca 2', color: null, is_manual: true, confidence: null },
    ]);
    expect(Object.keys(map)).toEqual(['t1', 't3']);
    expect(map.t1).toEqual({ speakerId: 's1', name: 'Anna', color: '#fff', isManual: false, confidence: 0.8 });
    expect(map.t3.isManual).toBe(true);
    expect(map.t3.confidence).toBeUndefined();
  });

  test('computeFollowingCounts counts the contiguous block after each segment', () => {
    expect(computeFollowingCounts(['a', 'a', 'a', 'b', 'b', undefined, 'a'])).toEqual([2, 1, 0, 1, 0, 0, 0]);
    expect(computeFollowingCounts([])).toEqual([]);
    expect(computeFollowingCounts(['a'])).toEqual([0]);
    expect(computeFollowingCounts([undefined, undefined])).toEqual([0, 0]);
  });

  test('blockIds returns the segment and the following ones, clamped', () => {
    const ids = ['t1', 't2', 't3', 't4'];
    expect(blockIds(ids, 1, 2)).toEqual(['t2', 't3', 't4']);
    expect(blockIds(ids, 3, 5)).toEqual(['t4']);
    expect(blockIds(ids, 0, 0)).toEqual(['t1']);
    expect(blockIds(ids, 9, 1)).toEqual([]);
  });
});
