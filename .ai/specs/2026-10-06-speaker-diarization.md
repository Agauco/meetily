# Rozpoznawanie mówców po głosie (speaker diarization) z ręczną korektą

| | |
|---|---|
| Status | Draft (spec, design-only) |
| Data | 2026-10-06 |
| Slug | `speaker-diarization` |
| Zakres | Tauri/Rust (`frontend/src-tauri`) + Next.js (`frontend/src`) |
| Język spotkań | polski |

## 1. Problem i cel

Meetily transkrybuje spotkania, ale nie wie, **kto** mówi. Pole `transcripts.speaker` istnieje (migracja `20251110000001_add_speaker_field.sql`), jednak oznacza **źródło dźwięku** (`mic` / `system`) i nie jest nigdzie zapisywane (wszystkie `INSERT INTO transcripts` je pomijają). Gdy kilka osób mówi do jednego mikrofonu, transkrypt jest jednym strumieniem bez podziału na rozmówców, a podsumowania LLM nie mogą przypisywać decyzji i zadań do osób.

**Cel:** rozpoznawać różne osoby **po głosie** (nie po źródle dźwięku), na żywo i po zakończeniu nagrania, pozwolić użytkownikowi nadawać im nazwy oraz ręcznie poprawiać błędne przypisania dla pojedynczych fragmentów.

**Poza zakresem (v1):** rozpoznawanie mówców między różnymi spotkaniami (trwałe profile głosowe), rozdzielanie nakładającej się mowy na osobne ścieżki audio, chmurowe API diaryzacji.

## 2. Wymagania

### Funkcjonalne

- F1. Każdy segment transkrypcji ma przypisanego mówcę rozpoznanego po głosie (etykieta „Mówca 1”, „Mówca 2”, …).
- F2. Etykiety pojawiają się **w trakcie nagrywania**, z opóźnieniem rzędu kilku sekund.
- F3. Użytkownik może nadać nazwę rozpoznanemu mówcy; nazwa obowiązuje wstecz i naprzód w całym spotkaniu (jedna aktualizacja rekordu mówcy).
- F4. Użytkownik może **ręcznie zmienić mówcę dla pojedynczego segmentu**, dla zakresu segmentów oraz podzielić segment na dwa i przypisać części osobno.
- F5. Ręczne korekty mają pierwszeństwo przed automatem i **nigdy nie są nadpisywane** przez ponowną diaryzację.
- F6. Po zakończeniu nagrania uruchamia się dokładniejszy przebieg offline, który poprawia etykiety automatyczne, zachowując nazwy i korekty ręczne.
- F7. Użytkownik może scalić dwóch mówców (automat rozbił jedną osobę na dwie) oraz utworzyć nowego.
- F8. Podsumowania i action items używają wyświetlanej nazwy mówcy; po zmianie przypisań dostępna jest akcja „wygeneruj podsumowanie ponownie”.
- F9. Funkcję można wyłączyć w ustawieniach; wyłączona nie zmienia dotychczasowego zachowania i nie ładuje modeli.

### Niefunkcjonalne

- N1. Całość lokalnie (zgodnie z założeniem prywatności produktu); żadne audio ani embeddingi nie opuszczają urządzenia.
- N2. Diaryzacja na żywo nie może opóźniać transkrypcji ani gubić chunków (worker jest serialny i gwarantuje brak utraty chunków; diaryzacja działa równolegle, nie w ścieżce krytycznej).
- N3. Narzut CPU na żywo: cel poniżej 15% jednego rdzenia na typowym laptopie.
- N4. Brak regresji: nagrania sprzed funkcji i importy działają bez zmian (`speaker_*` = NULL).
- N5. Modele pobierane na żądanie (jak modele Whisper/Parakeet), z postępem i obsługą błędów.

## 3. Stan obecny (ustalenia z kodu)

| Obszar | Fakt | Wpływ |
|---|---|---|
| `transcripts.speaker` | kolumna istnieje, znaczenie `mic`/`system`, nigdy nie zapisywana | nie nadpisujemy semantyki; dodajemy osobne kolumny |
| `TranscriptUpdate` (`audio/transcription/worker.rs`) | ma `source`, `audio_start_time`, `audio_end_time` | czas segmentu wystarcza do dopasowania mówcy |
| `AudioChunk` (`audio/recording_state.rs`) | ma `data`, `sample_rate`, `timestamp`, `device_type` | embedding liczymy z `data` po VAD |
| Zależności | `ort` 2.0.0-rc.10, `ndarray`, `rubato`, `silero_rs` | ONNX Runtime i resampling już są |
| Wstawianie transkryptów | `database/repositories/transcript.rs`, `audio/import.rs`, `audio/retranscription.rs` | trzy miejsca do zmiany |
| Parakeet | `parakeet_provider.rs` ignoruje podpowiedź języka; v3 jest wielojęzyczny | do weryfikacji dla polskiego (patrz §10) |
| Whisper | język z `get_language_preference_internal()` | polski ustawiamy jawnie |
| Frontend | `types/index.ts` `Transcript` bez pola mówcy; widoki `VirtualizedTranscriptView`, `TranscriptPanel` | rozszerzenie typu i UI |

## 4. Architektura

```
AudioChunk (po VAD) ──► Transkrypcja (bez zmian) ─► TranscriptUpdate{ start,end,text }
        │                                                    │
        └──► DiarizationEngine (osobny task, kolejka) ───────┤
              embedding → klasteryzacja online                │
              wynik: (start,end) → auto_speaker_id            ▼
                                                  merge po nakładaniu się czasów
                                                              │
                                  zapis: auto_speaker_id  ◄───┘   manual_speaker_id (UI)
                                                              │
        Stop nagrania ─► OfflineDiarizer (cały plik) ─► aktualizuje TYLKO auto_speaker_id
```

### 4.1 Moduł `diarization` (nowy, `src-tauri/src/diarization/`)

- `embedder.rs` — ładuje model embeddingów głosu (ONNX przez `ort`), przyjmuje PCM 16 kHz mono, zwraca wektor L2-znormalizowany. Kandydaci: WeSpeaker ResNet34 / CAM++ (do wyboru po benchmarku na polskiej mowie, §10).
- `online_clusterer.rs` — klasteryzacja online: dla nowego embeddingu liczy podobieństwo kosinusowe do centroidów mówców; powyżej progu `T_match` przypisuje i aktualizuje centroid (średnia ruchoma), poniżej `T_new` tworzy nowego mówcę, w strefie pośredniej przypisuje najbliższego z niższą pewnością. Stabilne `speaker_id` w ramach spotkania. Maksymalna liczba mówców konfigurowalna (domyślnie 10); opcjonalnie „znana liczba uczestników” zawęża wynik.
- `offline.rs` — pełna diaryzacja po nagraniu (segmentacja + embeddingi + klasteryzacja aglomeracyjna). Preferowana implementacja: **sherpa-onnx** (gotowy pipeline pyannote-segmentation + embedder); alternatywa: własny pipeline na `ort`. Wybór po spike’u (§10).
- `assign.rs` — dopasowanie wyników do segmentów transkrypcji po największym nakładaniu się czasowym; segment, w którym zmienia się mówca, jest sygnalizowany jako kandydat do podziału.

### 4.2 Zasady jakości

- Segmenty krótsze niż ~1,0 s nie tworzą nowych mówców; dostają najbliższego znanego z flagą niskiej pewności.
- Embedding liczony jest z mowy po VAD (bez ciszy); dla długich segmentów — z okien 1,5–3 s i uśredniany.
- Dwa źródła (`mic`, `system`) są diaryzowane **niezależnie**, ale w jednej przestrzeni identyfikatorów mówców w spotkaniu, żeby ta sama osoba po obu stronach nie dostała dwóch etykiet, o ile profile są zbliżone.
- Pewność przypisania (`speaker_confidence`) jest zapisywana i może być użyta w UI do wyróżnienia niepewnych fragmentów.

## 5. Model danych

Migracja `YYYYMMDDHHMMSS_add_speaker_diarization.sql` (nowe pliki; istniejących migracji nie edytujemy).

```sql
CREATE TABLE speakers (
  id           TEXT PRIMARY KEY,           -- uuid
  meeting_id   TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
  label        TEXT NOT NULL,              -- "Mówca 1" (nadawane automatycznie)
  name         TEXT,                       -- nazwa nadana przez użytkownika, NULL = brak
  color        TEXT,                       -- kolor etykiety w UI
  centroid     BLOB,                       -- embedding (f32 x N), opcjonalnie
  created_at   TEXT NOT NULL,
  merged_into  TEXT REFERENCES speakers(id) -- po scaleniu; NULL = aktywny
);
CREATE INDEX idx_speakers_meeting ON speakers(meeting_id);

ALTER TABLE transcripts ADD COLUMN auto_speaker_id    TEXT REFERENCES speakers(id);
ALTER TABLE transcripts ADD COLUMN manual_speaker_id  TEXT REFERENCES speakers(id);
ALTER TABLE transcripts ADD COLUMN speaker_confidence REAL;

CREATE TABLE speaker_edits (               -- historia ręcznych zmian (cofanie)
  id            TEXT PRIMARY KEY,
  meeting_id    TEXT NOT NULL,
  transcript_id TEXT NOT NULL,
  old_speaker   TEXT,
  new_speaker   TEXT,
  edited_at     TEXT NOT NULL
);
```

**Wyświetlany mówca** = `COALESCE(manual_speaker_id, auto_speaker_id)`, rozwiązywany przez `merged_into`. Nazwa wyświetlana = `speakers.name` albo `speakers.label`. Istniejące `transcripts.speaker` (`mic`/`system`) pozostaje bez zmian.

Podział segmentu tworzy dwa wiersze `transcripts` z rozdzieloną treścią i czasem (`audio_start_time`/`audio_end_time`/`duration`); tekst dzielony jest po granicy słów najbliższej punktowi podziału, a oryginał jest zachowany w `speaker_edits` do cofnięcia.

## 6. Kontrakt zdarzeń i komend Tauri

Rozszerzenie `TranscriptUpdate` (Rust) i `Transcript` (`frontend/src/types/index.ts`) o:

```
speaker_id?: string        // wyświetlany mówca (po rozwiązaniu manual/auto)
speaker_label?: string     // "Mówca 2" lub nadana nazwa
speaker_confidence?: number
```

Nowe zdarzenia: `speaker-updated` (zmiana nazwy/koloru/scalenie), `speakers-relabeled` (po przebiegu offline; zawiera listę zmienionych `transcript_id`, żeby UI odświeżył tylko je).

Nowe komendy:

| Komenda | Opis |
|---|---|
| `list_speakers(meeting_id)` | mówcy spotkania z liczbą wypowiedzi i łącznym czasem |
| `rename_speaker(speaker_id, name)` | F3 |
| `assign_segment_speaker(transcript_ids[], speaker_id \| new)` | F4, ustawia `manual_speaker_id` |
| `clear_segment_speaker_override(transcript_ids[])` | wraca do `auto_speaker_id` |
| `split_segment(transcript_id, at_seconds, left_speaker, right_speaker)` | F4 |
| `merge_speakers(from_id, into_id)` | F7, ustawia `merged_into` |
| `undo_speaker_edit(edit_id)` | cofnięcie z `speaker_edits` |
| `rediarize_meeting(meeting_id, expected_speakers?)` | F6 ręcznie, np. po imporcie |

## 7. Interfejs użytkownika

- **Transkrypt (live i po spotkaniu):** przy każdej wypowiedzi kolorowa etykieta mówcy; grupowanie kolejnych wypowiedzi tego samego mówcy; ikona niepewności przy niskim `speaker_confidence`.
- **Nazwanie:** kliknięcie etykiety otwiera popover: zmiana nazwy (dotyczy całego spotkania), lista innych mówców do przepisania segmentu, „Nowy mówca”, „Scal z…”.
- **Ręczna korekta segmentu:** w tym samym popoverze wybór mówcy tylko dla tego segmentu (domyślnie), opcje „zastosuj do następnych N wypowiedzi tego samego bloku”, zaznaczenie zakresu (shift+klik) do zbiorczej zmiany.
- **Podział segmentu:** akcja „Podziel tutaj” na wybranym słowie; po podziale oba fragmenty można przypisać osobno.
- **Wyróżnienie ręcznych korekt:** drobny znacznik „ręcznie”, możliwość wyczyszczenia nadpisania.
- **Ustawienia:** przełącznik diaryzacji, maksymalna liczba mówców lub „znana liczba uczestników”, pobieranie modeli z paskiem postępu.
- **Podsumowanie:** po zmianie przypisań banner „Przypisania mówców zmieniły się — wygeneruj podsumowanie ponownie”.
- Dostępność: etykiety nie opierają się wyłącznie na kolorze (nazwa tekstowa zawsze widoczna).

## 8. Podsumowania (LLM)

Transkrypt przekazywany do `summary/` jest formatowany jako `Nazwa mówcy: tekst`. Gdy mówcy nie mają nazw, używane są etykiety „Mówca N”. Prompt dostaje instrukcję przypisywania decyzji i zadań do mówców tylko wtedy, gdy nazwa została nadana; w innym wypadku opis ogólny.

## 9. Fazy implementacji

| Faza | Zakres | Kryterium ukończenia |
|---|---|---|
| 0 | Spike: benchmark modeli embeddingów i sherpa-onnx na polskich nagraniach (§10) | decyzja o modelu, progach `T_match`/`T_new`, raport DER/pomyłek |
| 1 | Migracja + modele + repozytoria + komendy `list/rename/assign/merge` (bez algorytmu) | testy jednostkowe repozytoriów; stare dane bez regresji |
| 2 | Rozszerzenie `TranscriptUpdate`/`Transcript`, UI etykiet, nazywanie, ręczna korekta pojedynczego segmentu | działa na danych testowych bez algorytmu |
| 3 | `embedder` + `online_clusterer` + zadanie diaryzacji na żywo | etykiety live w nagraniu 2–4 osób, brak utraty chunków |
| 4 | Przebieg offline po zatrzymaniu, scalanie z korektami ręcznymi | korekty ręczne zachowane po `rediarize_meeting` |
| 5 | Podział segmentu, zakres, cofanie (`speaker_edits`) | testy UI + testy komend |
| 6 | Integracja z podsumowaniami, regeneracja, dokumentacja (`docs/`) | podsumowanie zawiera nazwy mówców |
| 7 | Ustawienia, pobieranie modeli, wydajność (N3), Windows/macOS | build na obu platformach |

Każda faza = osobny, niezależnie wdrażalny PR; fazy 1–2 nie zmieniają zachowania aplikacji przy wyłączonej funkcji.

## 10. Rozstrzygnięte założenia (autonomiczne domyślne) i pytania otwarte

Poniższe wartości przyjęto, żeby spec był kompletny; każdą można zmienić.

| # | Kwestia | Przyjęte | Do potwierdzenia |
|---|---|---|---|
| A1 | Biblioteka offline | sherpa-onnx | Spike w fazie 0: wsparcie Windows/macOS, rozmiar binarki, licencja |
| A2 | Model embeddingów | WeSpeaker/CAM++ ONNX | Benchmark na polskiej mowie (modele trenowane głównie na en/zh) |
| A3 | Maks. liczba mówców | 10, konfigurowalna | — |
| A4 | Profile między spotkaniami | poza zakresem v1 | Czy chcesz to w v2? |
| A5 | Silnik transkrypcji dla polskiego | Whisper (`medium`+), język `pl` ustawiony jawnie | Czy Parakeet v3 w repo obsługuje polski i czy podpowiedź języka da się przekazać (obecnie ignorowana) |
| A6 | Nakładająca się mowa | segment dostaje dominującego mówcę, flaga niskiej pewności | Akceptowalne w v1? |
| A7 | Hosting modeli | jak istniejące modele (pobieranie na żądanie) | Źródło i mirror modeli diaryzacji |
| A8 | Dwa źródła audio (`mic`/`system`) | wspólna przestrzeń mówców, niezależna diaryzacja strumieni | Czy `system` ma być diaryzowany domyślnie |

## 11. Ryzyka

- **Jakość na polskiej mowie** — brak własnych pomiarów; spike w fazie 0 jest warunkiem dalszych prac.
- **Krótkie wypowiedzi i podobne głosy** — typowe pomyłki; łagodzimy flagą pewności i łatwą korektą ręczną.
- **Rozmiar i wydajność** — dodatkowe modele ONNX; pobieranie na żądanie, wyłączalne ustawienie.
- **Spójność przy korekcie** — ręczne nadpisania muszą przetrwać ponowną diaryzację; pokryte testem (faza 4).
- **Różnice platform** — ONNX Runtime ładowany dynamicznie na części platform (`load-dynamic`); weryfikacja na Windows i macOS.

## 12. Testy i weryfikacja

- Jednostkowe: klasteryzator online (stabilność ID, progi), dopasowanie po nakładaniu czasu, rozwiązywanie `COALESCE(manual, auto)` i `merged_into`.
- Repozytoria: migracja na kopii bazy ze starymi danymi; brak regresji importu i retranskrypcji.
- Integracyjne: nagranie testowe z 3 mówcami (PL) — liczba wykrytych mówców, odsetek błędnie przypisanych sekund; `rediarize_meeting` nie rusza `manual_speaker_id`.
- UI: nazwanie mówcy, korekta pojedynczego segmentu, zakres, podział, cofnięcie.
- Wydajność: pomiar narzutu CPU na żywo (N3) i brak wzrostu opóźnienia emisji transkryptu.
