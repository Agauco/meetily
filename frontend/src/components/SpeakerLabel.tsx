'use client';

import { memo, useState } from 'react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { SegmentSpeakerInfo, SpeakerSummary } from '@/types/speakers';
import type { SpeakerActions } from '@/hooks/useMeetingSpeakers';

export interface SpeakerLabelProps {
    segmentId: string;
    speaker?: SegmentSpeakerInfo;
    speakers: SpeakerSummary[];
    /** Number of directly following segments of the same speaker (a contiguous block). */
    followingCount: number;
    actions: SpeakerActions;
    /** Resolves the segment ids an action applies to (this segment, optionally its following block). */
    resolveIds: (segmentId: string, includeFollowing: boolean) => string[];
    /** Render only a colored dot (used for continuation segments of the same speaker). */
    compact?: boolean;
}

const FALLBACK_COLOR = '#9CA3AF';

function Dot({ color }: { color: string | null | undefined }) {
    return <span className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ backgroundColor: color ?? FALLBACK_COLOR }} />;
}

/**
 * Speaker label of a transcript segment. Click opens a popover to: rename the speaker
 * (applies to the whole meeting), reassign just this segment (optionally its following
 * block), create a new speaker, restore the automatic assignment, or merge two speakers.
 */
export const SpeakerLabel = memo(function SpeakerLabel({
    segmentId,
    speaker,
    speakers,
    followingCount,
    actions,
    resolveIds,
    compact = false,
}: SpeakerLabelProps) {
    const [open, setOpen] = useState(false);
    const [renameValue, setRenameValue] = useState('');
    const [newName, setNewName] = useState('');
    const [includeFollowing, setIncludeFollowing] = useState(false);
    const [mergeInto, setMergeInto] = useState('');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const run = async (task: () => Promise<void>, closeAfter = true) => {
        setBusy(true);
        setError(null);
        try {
            await task();
            if (closeAfter) setOpen(false);
        } catch (e) {
            setError(typeof e === 'string' ? e : 'Nie udało się zapisać zmiany');
        } finally {
            setBusy(false);
        }
    };

    const onOpenChange = (next: boolean) => {
        setOpen(next);
        if (next) {
            setRenameValue(speaker && speaker.name !== speakers.find((s) => s.id === speaker.speakerId)?.label ? speaker.name : '');
            setNewName('');
            setIncludeFollowing(false);
            setMergeInto('');
            setError(null);
        }
    };

    const ids = () => resolveIds(segmentId, includeFollowing && followingCount > 0);
    const currentSummary = speakers.find((s) => s.id === speaker?.speakerId);
    const isRoom = currentSummary?.kind === 'room';
    const hasRoom = speakers.some((s) => s.kind === 'room');
    const otherSpeakers = speakers.filter((s) => s.id !== speaker?.speakerId && s.kind !== 'room');

    return (
        <Popover open={open} onOpenChange={onOpenChange}>
            <PopoverTrigger asChild>
                {speaker && compact ? (
                    <button
                        type="button"
                        className="inline-flex h-4 w-4 items-center justify-center rounded-full hover:bg-gray-100"
                        title={`${speaker.name} — zmień mówcę`}
                        aria-label={`Mówca: ${speaker.name}. Zmień mówcę`}
                    >
                        <Dot color={speaker.color} />
                    </button>
                ) : speaker ? (
                    <button
                        type="button"
                        className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 bg-white px-2 py-0.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
                        title="Zmień mówcę"
                    >
                        <Dot color={speaker.color} />
                        <span>{speaker.name}</span>
                        {speaker.isManual && <span className="text-[10px] font-normal text-gray-400">ręcznie</span>}
                        {!speaker.isManual && speaker.confidence !== undefined && speaker.confidence < 0.6 && (
                            <span className="text-[10px] font-normal text-amber-600" title="Niska pewność rozpoznania">?</span>
                        )}
                    </button>
                ) : (
                    <button
                        type="button"
                        className="inline-flex items-center rounded-full border border-dashed border-gray-300 px-2 py-0.5 text-xs text-gray-400 opacity-0 transition-opacity hover:text-gray-600 group-hover:opacity-100 focus:opacity-100"
                        title="Przypisz mówcę"
                    >
                        + mówca
                    </button>
                )}
            </PopoverTrigger>
            <PopoverContent align="start" className="w-72 space-y-3 p-3 text-sm">
                {speaker && !isRoom && (
                    <div>
                        <label className="mb-1 block text-xs font-medium text-gray-500">
                            Nazwa mówcy (dla całego spotkania)
                        </label>
                        <div className="flex gap-1">
                            <input
                                value={renameValue}
                                onChange={(e) => setRenameValue(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter') void run(() => actions.rename(speaker.speakerId, renameValue.trim() || null));
                                }}
                                placeholder="np. Anna"
                                className="min-w-0 flex-1 rounded border border-gray-300 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                            />
                            <button
                                type="button"
                                disabled={busy}
                                onClick={() => run(() => actions.rename(speaker.speakerId, renameValue.trim() || null))}
                                className="rounded bg-blue-600 px-2 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                            >
                                Zapisz
                            </button>
                        </div>
                    </div>
                )}

                <div>
                    <div className="mb-1 text-xs font-medium text-gray-500">Przypisz ten fragment do:</div>
                    <ul className="max-h-40 space-y-0.5 overflow-y-auto">
                        {speakers.map((s) => (
                            <li key={s.id}>
                                <button
                                    type="button"
                                    disabled={busy || s.id === speaker?.speakerId}
                                    onClick={() => run(() => actions.assign(ids(), s.id))}
                                    className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-gray-100 disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent"
                                >
                                    <Dot color={s.color} />
                                    <span className="flex-1 truncate">{s.display_name}</span>
                                    {s.id === speaker?.speakerId && <span className="text-xs text-gray-400">obecny</span>}
                                </button>
                            </li>
                        ))}
                        {!hasRoom && (
                            <li>
                                <button
                                    type="button"
                                    disabled={busy}
                                    onClick={() => run(() => actions.assignRoom(ids()))}
                                    className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-gray-100"
                                >
                                    <Dot color="#6B7280" />
                                    <span className="flex-1 truncate">Sala (kilka osób naraz)</span>
                                </button>
                            </li>
                        )}
                        {speakers.length === 0 && <li className="px-2 py-1 text-xs text-gray-400">Brak mówców — dodaj pierwszego poniżej.</li>}
                    </ul>
                    <div className="mt-1 flex gap-1">
                        <input
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') void run(() => actions.createAndAssign(ids(), newName.trim() || undefined));
                            }}
                            placeholder="Nowy mówca (nazwa opcjonalna)"
                            className="min-w-0 flex-1 rounded border border-gray-300 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                        />
                        <button
                            type="button"
                            disabled={busy}
                            onClick={() => run(() => actions.createAndAssign(ids(), newName.trim() || undefined))}
                            className="rounded border border-gray-300 px-2 py-1 text-xs font-medium hover:bg-gray-50 disabled:opacity-50"
                        >
                            Dodaj
                        </button>
                    </div>

                    {followingCount > 0 && (
                        <label className="mt-2 flex items-center gap-2 text-xs text-gray-600">
                            <input
                                type="checkbox"
                                checked={includeFollowing}
                                onChange={(e) => setIncludeFollowing(e.target.checked)}
                            />
                            Także następne {followingCount} {followingCount === 1 ? 'wypowiedź' : 'wypowiedzi'} tego bloku
                        </label>
                    )}

                    {speaker?.isManual && (
                        <button
                            type="button"
                            disabled={busy}
                            onClick={() => run(() => actions.assign(ids(), null))}
                            className="mt-2 text-xs text-blue-600 hover:underline disabled:opacity-50"
                        >
                            Przywróć automatyczne przypisanie
                        </button>
                    )}
                </div>

                {speaker && !isRoom && otherSpeakers.length > 0 && (
                    <div className="border-t border-gray-100 pt-2">
                        <div className="mb-1 text-xs font-medium text-gray-500">Scal „{speaker.name}” z innym mówcą</div>
                        <div className="flex gap-1">
                            <select
                                value={mergeInto}
                                onChange={(e) => setMergeInto(e.target.value)}
                                className="min-w-0 flex-1 rounded border border-gray-300 px-1 py-1 text-sm"
                            >
                                <option value="">Wybierz…</option>
                                {otherSpeakers.map((s) => (
                                    <option key={s.id} value={s.id}>
                                        {s.display_name}
                                    </option>
                                ))}
                            </select>
                            <button
                                type="button"
                                disabled={busy || !mergeInto}
                                onClick={() => run(() => actions.merge(speaker.speakerId, mergeInto))}
                                className="rounded border border-gray-300 px-2 py-1 text-xs font-medium hover:bg-gray-50 disabled:opacity-50"
                            >
                                Scal
                            </button>
                        </div>
                    </div>
                )}

                {error && <p className="text-xs text-red-600">{error}</p>}
            </PopoverContent>
        </Popover>
    );
});
