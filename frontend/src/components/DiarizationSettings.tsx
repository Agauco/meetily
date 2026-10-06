"use client"

import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Users } from 'lucide-react';
import { Switch } from './ui/switch';
import { Button } from './ui/button';
import { Progress } from './ui/progress';

interface DiarizationStatus {
  enabled: boolean;
  max_speakers: number;
  model_ready: boolean;
  model_size_bytes: number;
  downloading: boolean;
}

/** Settings for recognizing speakers by voice: on/off, speaker limit and the voice model download. */
export function DiarizationSettings() {
  const [status, setStatus] = useState<DiarizationStatus | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setStatus(await invoke<DiarizationStatus>('diarization_get_status'));
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void load();
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    listen<{ downloaded: number; total: number }>('diarization-model-download-progress', (event) => {
      const { downloaded, total } = event.payload;
      if (total > 0) setProgress(Math.min(100, Math.round((downloaded / total) * 100)));
    }).then((fn) => (cancelled ? fn() : (unlisten = fn)));
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [load]);

  const setEnabled = async (enabled: boolean) => {
    setError(null);
    try {
      setStatus(await invoke<DiarizationStatus>('diarization_set_enabled', { enabled, maxSpeakers: status?.max_speakers }));
    } catch (e) {
      setError(String(e));
    }
  };

  const setMaxSpeakers = async (value: number) => {
    if (!status) return;
    try {
      setStatus(await invoke<DiarizationStatus>('diarization_set_enabled', { enabled: status.enabled, maxSpeakers: value }));
    } catch (e) {
      setError(String(e));
    }
  };

  const download = async () => {
    setError(null);
    setProgress(0);
    setStatus((s) => (s ? { ...s, downloading: true } : s));
    try {
      setStatus(await invoke<DiarizationStatus>('diarization_download_model'));
    } catch (e) {
      setError(String(e));
      await load();
    } finally {
      setProgress(null);
    }
  };

  if (!status) return null;
  const sizeMb = Math.round(status.model_size_bytes / 1_000_000);

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Users className="h-5 w-5 text-gray-600" />
            <h3 className="text-lg font-semibold text-gray-900">Rozpoznawanie mówców</h3>
          </div>
          <p className="text-sm text-gray-600">
            Rozpoznaje różne osoby po głosie podczas nagrywania, także gdy mówią do jednego mikrofonu. Fragmenty,
            w których mówi kilka osób naraz, są oznaczane jako „Sala”. Przetwarzanie odbywa się lokalnie.
          </p>
        </div>
        <Switch checked={status.enabled} onCheckedChange={setEnabled} disabled={!status.model_ready && !status.enabled} />
      </div>

      {!status.model_ready ? (
        <div className="space-y-2">
          <p className="text-sm text-gray-600">Wymagany model głosu (ok. {sizeMb} MB) nie jest jeszcze pobrany.</p>
          <Button size="sm" onClick={download} disabled={status.downloading}>
            {status.downloading ? 'Pobieranie…' : 'Pobierz model'}
          </Button>
          {progress !== null && <Progress value={progress} className="h-2" />}
        </div>
      ) : (
        <div className="flex items-center gap-3 text-sm text-gray-700">
          <label htmlFor="max-speakers">Maksymalna liczba mówców:</label>
          <input
            id="max-speakers"
            type="number"
            min={2}
            max={20}
            defaultValue={status.max_speakers}
            onBlur={(e) => setMaxSpeakers(Math.min(20, Math.max(2, Number(e.target.value) || status.max_speakers)))}
            className="w-20 rounded border border-gray-300 px-2 py-1"
          />
          <span className="text-xs text-gray-500">Model pobrany. Zmiany obowiązują od następnego nagrania.</span>
        </div>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
