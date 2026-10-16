import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { DiarizationProgress, RediarizeSummary } from '@/types/speakers';

export interface UseRediarizeReturn {
  /** True while an offline pass runs for this meeting (also the automatic one after recording). */
  running: boolean;
  /** 0..100 while running */
  progress: number;
  error: string | null;
  lastSummary: RediarizeSummary | null;
  /** Re-runs speaker recognition for the whole meeting; `expectedSpeakers` fixes the number of speakers. */
  run: (expectedSpeakers?: number | null) => Promise<RediarizeSummary | null>;
}

/** Tracks and triggers the offline speaker-recognition pass of a saved meeting. */
export function useRediarize(meetingId?: string | null): UseRediarizeReturn {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [lastSummary, setLastSummary] = useState<RediarizeSummary | null>(null);

  useEffect(() => {
    setRunning(false);
    setProgress(0);
    setError(null);
    setLastSummary(null);
    if (!meetingId) return;
    let cancelled = false;
    let unlisten: (() => void) | undefined;
    invoke<boolean>('api_is_rediarizing', { meetingId })
      .then((r) => {
        if (!cancelled && r) setRunning(true);
      })
      .catch(() => {});
    listen<DiarizationProgress>('speaker-diarization-progress', (event) => {
      const p = event.payload;
      if (!p || p.meeting_id !== meetingId) return;
      if (p.stage === 'done') {
        setRunning(false);
        setProgress(100);
      } else if (p.stage === 'failed') {
        setRunning(false);
        setError(p.error ?? 'Nie udało się rozpoznać mówców');
      } else {
        setRunning(true);
        setError(null);
        setProgress(p.progress);
      }
    })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch((e) => console.error('Failed to listen for diarization progress:', e));
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [meetingId]);

  const run = useCallback(
    async (expectedSpeakers?: number | null) => {
      if (!meetingId) return null;
      setRunning(true);
      setProgress(0);
      setError(null);
      try {
        const summary = await invoke<RediarizeSummary>('api_rediarize_meeting', {
          meetingId,
          expectedSpeakers: expectedSpeakers && expectedSpeakers > 0 ? expectedSpeakers : null,
        });
        setLastSummary(summary);
        return summary;
      } catch (e) {
        setError(String(e));
        return null;
      } finally {
        setRunning(false);
      }
    },
    [meetingId]
  );

  return { running, progress, error, lastSummary, run };
}
