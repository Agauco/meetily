import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';

const originalCore = { ...await import('@tauri-apps/api/core') };
const originalEvent = { ...await import('@tauri-apps/api/event') };

const calls: Array<{ command: string; args: Record<string, unknown> }> = [];
const speakerRows = [{ id: 's1', label: 'Mówca 1', name: null, display_name: 'Mówca 1', color: '#3B82F6', utterance_count: 1, total_duration: 2 }];
const segmentRows = [
  { transcript_id: 't1', speaker_id: 's1', display_name: 'Mówca 1', color: '#3B82F6', is_manual: false, confidence: 0.9 },
  { transcript_id: 't2', speaker_id: null, display_name: null, color: null, is_manual: false, confidence: null },
];
const invoke = mock(async (command: string, args: Record<string, unknown>) => {
  calls.push({ command, args });
  if (command === 'api_list_speakers') return speakerRows;
  if (command === 'api_list_segment_speakers') return segmentRows;
  if (command === 'api_create_speaker') return { id: 'new-speaker' };
  return null;
});
let eventHandler: ((event: { payload: { meeting_id: string } }) => void) | undefined;
const unlisten = mock(() => {});
const listen = mock(async (_name: string, handler: typeof eventHandler) => {
  eventHandler = handler;
  return unlisten;
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
mock.module('@tauri-apps/api/event', () => ({ ...originalEvent, listen }));
const { useMeetingSpeakers } = await import('../../src/hooks/useMeetingSpeakers');
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('@tauri-apps/api/event', () => originalEvent);
});

let state: ReturnType<typeof useMeetingSpeakers>;
let renderer: ReactTestRenderer | undefined;
function View({ meetingId, enabled = true }: { meetingId: string | null; enabled?: boolean }) {
  state = useMeetingSpeakers(meetingId, enabled);
  return <output>{state.speakers.length}</output>;
}
async function show(meetingId: string | null, enabled = true) {
  await act(async () => {
    if (renderer) renderer.update(<View meetingId={meetingId} enabled={enabled} />);
    else renderer = create(<View meetingId={meetingId} enabled={enabled} />);
  });
}
const commands = () => calls.map(c => c.command);

beforeEach(() => {
  calls.length = 0;
  eventHandler = undefined;
  invoke.mockClear();
  listen.mockClear();
  unlisten.mockClear();
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
  mock.module('@tauri-apps/api/event', () => ({ ...originalEvent, listen }));
});
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
});

describe('useMeetingSpeakers', () => {
  test('loads speakers and maps only segments that have a speaker', async () => {
    await show('m1');
    expect(commands().sort()).toEqual(['api_list_segment_speakers', 'api_list_speakers']);
    expect(calls[0].args).toEqual({ meetingId: 'm1' });
    expect(state.speakers).toHaveLength(1);
    expect(Object.keys(state.segmentSpeakers)).toEqual(['t1']);
    expect(state.segmentSpeakers.t1.name).toBe('Mówca 1');
  });

  test('stays inert without a meeting id or when disabled', async () => {
    await show(null);
    await show('m1', false);
    expect(calls).toHaveLength(0);
    expect(listen).not.toHaveBeenCalled();
  });

  test('assign sends the camelCase Tauri arguments and refreshes', async () => {
    await show('m1');
    calls.length = 0;
    await act(async () => state.actions.assign(['t1', 't2'], null));
    expect(calls[0]).toEqual({ command: 'api_assign_segment_speaker', args: { meetingId: 'm1', transcriptIds: ['t1', 't2'], speakerId: null } });
    expect(commands()).toContain('api_list_speakers');
    calls.length = 0;
    await act(async () => state.actions.assign([], 's1'));
    expect(calls).toHaveLength(0);
  });

  test('createAndAssign creates a speaker first and assigns it', async () => {
    await show('m1');
    calls.length = 0;
    await act(async () => state.actions.createAndAssign(['t2'], 'Anna'));
    expect(calls[0]).toEqual({ command: 'api_create_speaker', args: { meetingId: 'm1', name: 'Anna' } });
    expect(calls[1]).toEqual({ command: 'api_assign_segment_speaker', args: { meetingId: 'm1', transcriptIds: ['t2'], speakerId: 'new-speaker' } });
  });

  test('rename and merge call their commands', async () => {
    await show('m1');
    calls.length = 0;
    await act(async () => state.actions.rename('s1', 'Anna'));
    await act(async () => state.actions.merge('s2', 's1'));
    expect(calls.find(c => c.command === 'api_rename_speaker')?.args).toEqual({ meetingId: 'm1', speakerId: 's1', name: 'Anna' });
    expect(calls.find(c => c.command === 'api_merge_speakers')?.args).toEqual({ meetingId: 'm1', fromSpeakerId: 's2', intoSpeakerId: 's1' });
  });

  test('refetches on speaker-updated for this meeting only and unsubscribes on unmount', async () => {
    await show('m1');
    calls.length = 0;
    await act(async () => eventHandler!({ payload: { meeting_id: 'other' } }));
    expect(calls).toHaveLength(0);
    await act(async () => eventHandler!({ payload: { meeting_id: 'm1' } }));
    expect(commands()).toContain('api_list_speakers');
    await act(async () => renderer!.unmount());
    renderer = undefined;
    expect(unlisten).toHaveBeenCalledTimes(1);
  });
});
