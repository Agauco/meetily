import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { SegmentSpeaker, SegmentSpeakerInfo, SpeakerSummary } from '@/types/speakers';
import { buildSegmentSpeakerMap } from '@/lib/speakers';

export interface SpeakerActions {
  rename: (speakerId: string, name: string | null) => Promise<void>;
  /** Assigns a speaker to segments; `null` clears the manual override. */
  assign: (transcriptIds: string[], speakerId: string | null) => Promise<void>;
  /** Creates a new speaker (optionally named) and assigns it to the segments. */
  createAndAssign: (transcriptIds: string[], name?: string) => Promise<void>;
  merge: (fromSpeakerId: string, intoSpeakerId: string) => Promise<void>;
}

export interface UseMeetingSpeakersReturn {
  speakers: SpeakerSummary[];
  segmentSpeakers: Record<string, SegmentSpeakerInfo>;
  actions: SpeakerActions;
  refresh: () => Promise<void>;
}

/**
 * Loads speakers and per-segment speaker info of a saved meeting and keeps them in sync
 * with the backend (`speaker-updated` event). Inert when `meetingId` is missing or disabled.
 */
export function useMeetingSpeakers(meetingId?: string | null, enabled: boolean = true): UseMeetingSpeakersReturn {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [segmentRows, setSegmentRows] = useState<SegmentSpeaker[]>([]);
  const activeMeetingRef = useRef<string | null | undefined>(meetingId);
  activeMeetingRef.current = meetingId;

  const refresh = useCallback(async () => {
    if (!meetingId || !enabled) {
      setSpeakers([]);
      setSegmentRows([]);
      return;
    }
    try {
      const [speakerList, rows] = await Promise.all([
        invoke<SpeakerSummary[]>('api_list_speakers', { meetingId }),
        invoke<SegmentSpeaker[]>('api_list_segment_speakers', { meetingId }),
      ]);
      // Ignore a stale response if the meeting changed meanwhile.
      if (activeMeetingRef.current !== meetingId) return;
      setSpeakers(speakerList);
      setSegmentRows(rows);
    } catch (error) {
      console.error('Failed to load speakers:', error);
    }
  }, [meetingId, enabled]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!meetingId || !enabled) return;
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    listen<{ meeting_id: string }>('speaker-updated', (event) => {
      if (event.payload?.meeting_id === meetingId) void refresh();
    })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch((error) => console.error('Failed to listen for speaker updates:', error));
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [meetingId, enabled, refresh]);

  const actions = useMemo<SpeakerActions>(
    () => ({
      rename: async (speakerId, name) => {
        if (!meetingId) return;
        await invoke('api_rename_speaker', { meetingId, speakerId, name });
        await refresh();
      },
      assign: async (transcriptIds, speakerId) => {
        if (!meetingId || transcriptIds.length === 0) return;
        await invoke('api_assign_segment_speaker', { meetingId, transcriptIds, speakerId });
        await refresh();
      },
      createAndAssign: async (transcriptIds, name) => {
        if (!meetingId || transcriptIds.length === 0) return;
        const speaker = await invoke<{ id: string }>('api_create_speaker', { meetingId, name: name ?? null });
        await invoke('api_assign_segment_speaker', { meetingId, transcriptIds, speakerId: speaker.id });
        await refresh();
      },
      merge: async (fromSpeakerId, intoSpeakerId) => {
        if (!meetingId) return;
        await invoke('api_merge_speakers', { meetingId, fromSpeakerId, intoSpeakerId });
        await refresh();
      },
    }),
    [meetingId, refresh]
  );

  const segmentSpeakers = useMemo(() => buildSegmentSpeakerMap(segmentRows), [segmentRows]);

  return { speakers, segmentSpeakers, actions, refresh };
}
