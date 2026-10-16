"use client";

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Progress } from '@/components/ui/progress';
import { useRediarize } from '@/hooks/useRediarize';

interface RediarizeButtonProps {
  meetingId: string;
}

/**
 * Re-runs voice-based speaker recognition for the whole meeting (offline pass).
 * Names and manual corrections are kept. Also shows progress of the automatic pass after recording.
 * Hidden when the voice model is not downloaded.
 */
export function RediarizeButton({ meetingId }: RediarizeButtonProps) {
  const { running, progress, error, lastSummary, run } = useRediarize(meetingId);
  const [modelReady, setModelReady] = useState(false);
  const [open, setOpen] = useState(false);
  const [expected, setExpected] = useState('');

  useEffect(() => {
    invoke<{ model_ready: boolean }>('diarization_get_status')
      .then((s) => setModelReady(s.model_ready))
      .catch(() => setModelReady(false));
  }, [meetingId]);

  if (!modelReady) return null;

  const start = async () => {
    const n = parseInt(expected, 10);
    await run(Number.isFinite(n) && n > 0 ? Math.min(n, 20) : null);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          size="sm"
          variant="outline"
          className="px-2 @[22rem]:px-3"
          title="Rozpoznaj mówców ponownie"
          aria-label="Rozpoznaj mówców ponownie"
        >
          {running ? <Loader2 className="animate-spin @[22rem]:mr-2" size={18} /> : <Users className="@[22rem]:mr-2" size={18} />}
          <span className="hidden @[22rem]:inline">{running ? `Mówcy ${progress}%` : 'Mówcy'}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-3 p-3 text-sm">
        <div className="font-medium">Rozpoznaj mówców ponownie</div>
        <p className="text-xs text-gray-600">
          Analizuje głosy w całym nagraniu i poprawia automatyczne etykiety. Nadane nazwy i ręczne korekty zostają bez zmian.
        </p>
        {running ? (
          <div className="space-y-1">
            <Progress value={progress} />
            <div className="text-xs text-gray-600">Trwa analiza… {progress}%</div>
          </div>
        ) : (
          <>
            <label className="block space-y-1">
              <span className="text-xs text-gray-700">Liczba mówców (opcjonalnie)</span>
              <Input
                type="number"
                min={1}
                max={20}
                placeholder="automatycznie"
                value={expected}
                onChange={(e) => setExpected(e.target.value)}
                className="h-8"
              />
            </label>
            <Button size="sm" className="w-full" onClick={start}>
              Uruchom
            </Button>
          </>
        )}
        {error && <div className="text-xs text-red-600" role="alert">{error}</div>}
        {!running && !error && lastSummary && (
          <div className="text-xs text-gray-600" role="status">
            Gotowe. Rozpoznani mówcy: {lastSummary.speaker_count}; zmienione wypowiedzi:{' '}
            {lastSummary.changed_transcript_ids.length} z {lastSummary.segments_total}.
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

