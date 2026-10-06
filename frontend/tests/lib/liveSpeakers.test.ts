import { describe, expect, test } from 'bun:test';
import type { Transcript } from '../../src/types';
import {
  assignLiveSegments, buildLiveSpeakers, createAndAssignLive, effectiveSpeakerId,
  inheritSpeakerNames, mergeLiveSpeakers, renameLiveSpeaker,
} from '../../src/lib/liveSpeakers';

const t = (id: string, extra: Partial<Transcript> = {}): Transcript => ({ id, text: id, timestamp: '', duration: 2, ...extra });
const base = () => [
  t('a', { speaker_index: 0 }), t('b', { speaker_index: 1 }), t('c', { speaker_is_room: true }), t('d', { speaker_index: 0 }), t('e'),
];

describe('live speakers', () => {
  test('effective speaker prefers manual correction', () => {
    expect(effectiveSpeakerId(t('x', { speaker_index: 0, manual_speaker_index: 2 }))).toBe('p2');
    expect(effectiveSpeakerId(t('x', { speaker_index: 0, manual_speaker_is_room: true }))).toBe('room');
    expect(effectiveSpeakerId(t('x'))).toBeUndefined();
  });

  test('build lists persons then room with counts', () => {
    const { speakers, segmentSpeakers } = buildLiveSpeakers(base());
    expect(speakers.map((s) => [s.id, s.display_name, s.kind, s.utterance_count])).toEqual([
      ['p0', 'Mówca 1', 'person', 2], ['p1', 'Mówca 2', 'person', 1], ['room', 'Sala', 'room', 1],
    ]);
    expect(segmentSpeakers.e).toBeUndefined();
  });

  test('rename applies to all segments and is inherited by new ones', () => {
    const renamed = renameLiveSpeaker(base(), 'p0', ' Anna ');
    expect(buildLiveSpeakers(renamed).segmentSpeakers.d.name).toBe('Anna');
    const incoming = inheritSpeakerNames(renamed, [t('f', { speaker_index: 0 })]);
    expect(incoming[0].speaker_name).toBe('Anna');
    expect(renameLiveSpeaker(renamed, 'room', 'x')).toBe(renamed); // room cannot be renamed
  });

  test('manual assign, clear, create and merge', () => {
    let list = assignLiveSegments(renameLiveSpeaker(base(), 'p1', 'Piotr'), ['a'], 'p1');
    expect(buildLiveSpeakers(list).segmentSpeakers.a).toMatchObject({ name: 'Piotr', isManual: true });
    list = assignLiveSegments(list, ['a'], 'room');
    expect(effectiveSpeakerId(list[0])).toBe('room');
    list = assignLiveSegments(list, ['a'], null);
    expect(effectiveSpeakerId(list[0])).toBe('p0');
    list = createAndAssignLive(list, ['b', 'c'], 'Ewa');
    expect(list[1].manual_speaker_index).toBe(2);
    expect(buildLiveSpeakers(list).segmentSpeakers.c.name).toBe('Ewa');
    list = mergeLiveSpeakers(list, 'p2', 'p0');
    expect(effectiveSpeakerId(list[1])).toBe('p0');
  });
});
