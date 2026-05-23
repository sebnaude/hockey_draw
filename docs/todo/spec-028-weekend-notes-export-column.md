<!-- status: building -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->
<!-- owner: session=opus-aa5b71e-20260523T091653Z claimed=2026-05-23T09:16:53Z -->
<!-- reviewed: adversarial Sonnet review 2026-05-23 (re-review — dependency audit) — fixes applied inline -->

# spec-028 — Per-weekend notes column in the published season xlsx

**Spec source:** convenor request (this session) — "the season draw export should also export the notes for each weekend."

## Why

The published season draw (`draws/{year}/season/2026 Season Draw V{X.Y}.xlsx`) is what the convenor hands to clubs. Today it shows games + byes per week but carries **no context** for *why* a weekend looks the way it does: that a weekend is a State Masters weekend with no morning games at Broadmeadow, that a club requested a specific arrangement, that a marquee fixture was forced onto a date (Norths v Wests 80th Anniversary), or that a club is preferred home that weekend (NRL-Knights-at-Maitland). All of this lives in config (`BLOCKED_GAMES`, `FORCED_GAMES`, `PREFERRED_WEEKENDS`) or in the convenor's head, and the convenor currently re-explains it by hand every time they publish. Surfacing it directly in the published sheet, per weekend, removes that manual step and makes the published draw self-explaining.

The reusable, maintained export is `DrawStorage.export_schedule_xlsx()` (`analytics/storage.py:416`). Future season exports inherit from it; the one-off `scripts/export_v4_*.py` are historical snapshots and are **out of scope** (see Out of scope).

## Design

### Note categories and their sources

Five categories, two source mechanisms. The contract is deliberately **opt-in** so routine/mechanical config (e.g. the dozens of "NIHC Friday blocked" / "Maitland Friday - only Gosford allowed" entries) never leaks into the published sheet as noise.

| Category | Source mechanism | How a note is produced |
|----------|------------------|------------------------|
| `Field` | hand-authored `data/{year}/notes.json` | Convenor writes free text (e.g. "Masters SC at Broadmeadow — no AM games"). `FIELD_UNAVAILABILITIES` is **not** auto-derived — it carries only `datetime`s with code-comments, no machine-readable reason. |
| `Request` | hand-authored `data/{year}/notes.json` | Convenor-tracked club requests, keyed by date. This is the "requests tracking" the convenor asked for. |
| `Preferred` | auto-derived from `data['preferred_weekends']` (`PREFERRED_WEEKENDS`, spec-006) | Each entry with an opt-in `note` field → "Preferred: …". |
| `Blocked` | auto-derived from `data['blocked_games']` (`BLOCKED_GAMES`) | Each entry with an opt-in `note` field → "Blocked: …". |
| `Forced` | auto-derived from `data['forced_games']` (`FORCED_GAMES`) | Each entry with an opt-in `note` field → "Forced: …". |

**Opt-in rule for auto-derived sources.** A `BLOCKED_GAMES` / `FORCED_GAMES` / `PREFERRED_WEEKENDS` entry is surfaced **only if it carries a truthy `'note'` key**:
- `'note': "<text>"` → use that text verbatim.
- `'note': True` → fall back to the entry's `description` (or `reason` if no `description`).
- no `'note'` key (the default for the ~60 existing mechanical entries) → **never surfaced**.

This means landing spec-028 changes the published sheet **only** once a convenor opts an entry in or writes `notes.json` — zero behavioural surprise on first run.

**Date requirement for auto-derived sources.** Only entries with a concrete, resolvable date produce a note:
- `BLOCKED_GAMES`/`FORCED_GAMES`: surfaced only if the entry has a `'date'` (ISO string). Entries with only `'day'` (solver picks the date, e.g. "Maitland vs Souths — exactly 1 NIHC Friday") cannot be placed on a weekend and are skipped even if they carry `'note'`. Note: some entries have BOTH `'date'` and `'day'` (e.g. Gosford unavailable entries in season_2026.py) — these ARE resolvable via the `'date'` key.
- `PREFERRED_WEEKENDS`: uses `'date'` (all current entries use the singular form). A `'dates'` plural list is also supported for future entries.

### `data/{year}/notes.json` schema

New, hand-authored, optional file. Absent file → no hand-authored notes (auto-derived still run). Keyed by ISO date string → list of note objects:

```json
{
  "2026-05-15": [
    {"category": "Field",   "text": "Masters SC at Broadmeadow — no games before 1pm"},
    {"category": "Request", "text": "Gosford requested 8pm Friday start"}
  ],
  "2026-06-12": [
    {"category": "Preferred", "text": "Norths home — 80th Anniversary weekend"}
  ]
}
```

- `category`: one of `Field` | `Request` | `Preferred` | `Blocked` | `Forced` (free string; unknown values are passed through but should be one of these). Optional — defaults to `Note`.
- `text`: required, the displayed note.
- A shorthand `"2026-05-15": ["plain string", ...]` (list of bare strings) is also accepted; bare strings get category `Note`.

### `build_weekend_notes()` — new builder (`analytics/notes.py`)

```python
def build_weekend_notes(
    draw: "DrawStorage",
    data: dict,
    notes_path: Optional[str] = None,
) -> Dict[int, List[str]]:
    """Return {week_number: [formatted note lines]} for the export Notes column.

    Sources, merged then deduped per week (see spec-028):
      - hand-authored notes_path JSON (default data/{year}/notes.json)
      - opt-in 'note' entries in data['blocked_games'] / ['forced_games'] / ['preferred_weekends']
    Each output line is "Category: text". Order within a week is stable:
    Field, Request, Preferred, Blocked, Forced; ties broken by insertion order.
    """
```

Responsibilities (all date→week logic lives here so the renderer stays dumb):
1. Build a `date(ISO str) → week(int)` map from `draw.games` (every `StoredGame` contributes `g.date → g.week`; both `.date` and `.week` are confirmed attributes of `StoredGame` at `analytics/storage.py:38-39`).
2. For dates appearing in a source but **not** in any scheduled game (a blocked date with zero games), resolve the week by 7-day bucketing from the season start date. (review fix — Critical: `data['start_date']` does NOT exist in the data dict returned by `build_season_data()` / `load_season_data()`. The `build_season_data()` return dict at `utils.py:4073–4106` does not include `start_date`. The value lives in `SEASON_CONFIG` at `config/season_2026.py:1144` as a `datetime` object but is not threaded into the data dict. The builder MUST retrieve it via `from config import load_season_config; start_date = load_season_config(data['year'])['start_date']` — do NOT use `data['start_date']`.) If it falls outside the season window, the note is **dropped** (and logged), never crashes.
3. Collect notes per week, format as `"{category}: {text}"`, **dedup** by exact formatted string (kills the 10×"Masters SC weekend - NIHC PHL only" duplication), preserving category order then insertion order.
4. `notes_path` defaults to `data/{data['year']}/notes.json` (using `data['year']` which IS in the returned data dict at `utils.py:4074`); a missing file is fine (treated as `{}`).

### Renderer change (`analytics/storage.py:export_schedule_xlsx`)

Add an optional param and a Notes column. The renderer does **not** load config or notes.json itself — the caller passes the prebuilt dict (keeps `DrawStorage` config-free, consistent with its current design).

- New signature: `export_schedule_xlsx(self, filename, weeks=None, sheet_title=None, weekend_notes: Optional[Dict[int, List[str]]] = None)`.
- Column layout (current `HEADERS` are cols A–K = 1–11, `NUM_COLS = 11`): leave **L (12) and M (13) blank as spacers**, put **Notes in N (14)**. Per the convenor: ≥2 blank columns between the last filled column (Venue, K) and Notes.
- Header row: write `"Notes"` in N on each week's column-header row. The existing loop `for ci, h in enumerate(HEADERS, 1)` at `storage.py:508-512` writes columns A–K only — do NOT modify `HEADERS` (that would change `NUM_COLS` and break the week-title fill range). Instead, write N separately AFTER the loop: `ws.cell(row=row, column=14, value="Notes")` with `header_font`/`header_fill`/`thin_border`. L and M stay empty (no header, no fill) on all rows. (review note — Low: the week title row fills `range(1, NUM_COLS + 1)` = A–K only; column N on the title row should also get `week_bg` fill so it visually belongs to the week band. Add `ws.cell(row=title_row, column=14).fill = week_bg` when writing the title row — N does not get `week_font`, just the background.)
- Note placement: **stacked from the week's first game row downward**, one note per row in column N. The week's first game data row is the first row written in the `for date … for field … for g in field_games` block (`storage.py:547-555`). Capture that row index before writing the first game (e.g. `if first_game_row is None: first_game_row = row` at top of the inner game loop), then after all games are written, write `weekend_notes.get(week, [])[i]` into `(first_game_row + i, 14)` for each note `i`. If a week has more notes than game rows, continue writing notes on subsequent rows — notes never overwrite game cells (they're in column N; games occupy A–K) and never collide with byes (byes are column A only).
- Notes cells: `Font(italic=False)`, wrap off, `thin_border` to match the grid; week background fill (`week_bg`) so the note rows visually belong to the week band.
- Column widths: add `'L': 3, 'M': 3, 'N': 60` to `COL_WIDTHS` (`storage.py:476-479`). (review note — Low: use `NUM_COLS + 3` = 14 as the Notes column index constant rather than hardcoding 14 so the column tracks future changes to `HEADERS`.)

### Caller wiring

No maintained script currently calls `export_schedule_xlsx` in the final-form worktree — the `draws/` directory is empty and no publish automation exists yet in this branch. (review fix — High: the plan said "identify the current caller" but verification shows there is NO automated publish caller in the final-form worktree. The only callers are: (a) CLAUDE.md Quick Commands (ad-hoc one-liners), (b) `tests/test_analytics_coverage.py` (test fixture). The maintained publish path does not yet exist in final-form.) DoD #7 is therefore satisfied by: adding a documented `build_weekend_notes` → `export_schedule_xlsx` snippet to **CLAUDE.md Quick Commands** (the reference for how the convenor publishes manually). No script changes are needed unless the team decides to add one.

## Definition of Done

1. `analytics/notes.py` exists with `build_weekend_notes(draw, data, notes_path=None) -> Dict[int, List[str]]` implementing the source contract above (hand-authored JSON + opt-in `note` on blocked/forced/preferred), date→week resolution including off-draw dates, dedup, and stable category ordering.
2. A missing `data/{year}/notes.json` returns auto-derived-only notes without error; a malformed JSON raises a clear `ValueError` naming the file (not a bare `JSONDecodeError`).
3. Auto-derived sources surface an entry **iff** it has a truthy `'note'` key AND (for blocked/forced) a concrete `'date'`. An entry without `'note'` produces no note. `'note': True` falls back to `description`/`reason`.
4. `export_schedule_xlsx` accepts `weekend_notes` and, when supplied, writes a `"Notes"` header in column **N (14)** on each week's header row (written separately after the HEADERS loop, not by modifying HEADERS), leaves columns **L (12)** and **M (13)** empty, applies `week_bg` fill to N on the week title row, and stacks each week's notes one-per-row from the week's first game row. Column widths L=3, M=3, N=60 applied.
5. Calling `export_schedule_xlsx` with `weekend_notes=None` (or omitted) produces a sheet byte-for-cell-identical to today in columns A–K, with no Notes header and nothing in L–N — i.e. the feature is strictly additive and opt-in.
6. A sample `data/2026/notes.json` is committed with at least one real entry per hand-authored category (`Field`, `Request`), and at least one `BLOCKED_GAMES`, one `FORCED_GAMES`, and one `PREFERRED_WEEKENDS` entry in `config/season_2026.py` is given an opt-in `'note'` so the exported sheet demonstrably shows all five categories.
7. CLAUDE.md Quick Commands documents the publish snippet: `from analytics.notes import build_weekend_notes; notes = build_weekend_notes(draw, data); draw.export_schedule_xlsx(path, weekend_notes=notes)`. (No automated publish script required — publishing in final-form is currently manual; no `draws/2026/season/` directory exists yet in this worktree.)
8. Docs updated per the doc registry below.
9. Tests pass (see per-unit test outlines); coverage ≥85% on new/changed code in `analytics/notes.py` and the changed branch of `export_schedule_xlsx`.

## Implementation units

### Unit A — notes builder + sample data + tests

- **Files touched:**
  - `analytics/notes.py` (NEW) — `build_weekend_notes()` + private helpers (`_load_notes_json`, `_date_to_week_map`, `_resolve_week`, `_derive_from_config`).
  - `data/2026/notes.json` (NEW) — sample hand-authored notes (DoD #6).
  - `config/season_2026.py` — add an opt-in `'note'` to one existing entry each in `BLOCKED_GAMES`, `FORCED_GAMES`, `PREFERRED_WEEKENDS` (DoD #6). No structural/behavioural change — `'note'` is inert to the solver. (review fix — Medium: the scope-matcher at `utils.py:650` iterates only `_SCOPE_FIELDS = {'grade', 'day', 'day_slot', 'time', 'week', 'date', 'round_no', 'field_name', 'field_location'}`. A `'note'` key is not in `_SCOPE_FIELDS` and is silently skipped; no validation rejects unknown keys anywhere in `utils.py` or the constraints layer. Adding `'note'` to a config entry is **confirmed safe and inert** — no ignore-list change is needed.)
  - `tests/test_weekend_notes.py` (NEW).
- **Change summary:** Pure derivation. No solver/constraint contact. Reads `data['blocked_games']`, `data['forced_games']`, `data['preferred_weekends']`, and `notes.json`; uses a `DrawStorage` only for the date→week map.
- **Depends on:** none.
- **No-mock test outline (Given/When/Then, hand-computed oracles):**
  - (review fix — High: `draws/2026/current.json` does NOT exist in the final-form worktree — the `draws/` directory is empty. All tests MUST build `DrawStorage` objects synthetically (programmatically via `StoredGame` instances or `DrawStorage.from_X_solution`), never by loading a file. Replace the first test below accordingly.)
  - *Given* a synthetic `DrawStorage` built with one `StoredGame` on date `'2026-05-17'` set to `week=9`, and a `notes.json` temp file containing `{"2026-05-17": [{"category": "Field", "text": "Test note"}]}`, and `data = load_season_data(2026)`, *when* `build_weekend_notes(draw, data, notes_path=<tmp>)`, *then* `result[9]` contains `"Field: Test note"`. (Oracle: the synthetic game maps date `2026-05-17` to week 9 explicitly; assert `result[9] == ["Field: Test note"]`.)
  - *Given* a synthetic `data` dict with `'blocked_games'` containing one entry `{'date': '2026-05-17', 'description': 'Masters SC weekend', 'note': True}` plus nine sibling entries on the same date with the same description but **no** `'note'` key, and a synthetic `DrawStorage` with a game on `'2026-05-17'` at `week=9`, *when* built, *then* `result[9]` contains exactly **one** `"Blocked: Masters SC weekend"` (dedup + opt-in proven). (Oracle: `2026-05-17` is 56 days after season start `2026-03-22`; raw bucket = 56//7+1 = week 9; `len(result[9]) == 1`.)
  - *Given* a `FORCED_GAMES`-style entry with only `'day': 'Friday'` and no `'date'` but `'note': True`, *when* built, *then* it appears in **no** week (un-dateable). (Oracle: assert the formatted string is absent from all values in the result dict.)
  - *Given* a synthetic `data` dict with one `'preferred_weekends'` entry `{'date': '2026-06-12', 'note': True, 'description': 'Norths home — 80th Anniversary', 'mode': 'avoid'}` and a synthetic `DrawStorage` with a game on `'2026-06-12'` at `week=12`, *then* `result[12]` contains `"Preferred: Norths home — 80th Anniversary"`. (Oracle: `2026-06-12` is 82 days after `2026-03-22`; raw bucket = 82//7+1 = week 12; game-based resolution yields the same value; assert `"Preferred: Norths home — 80th Anniversary"` in `result[12]`.)
  - *Given* `notes_path` pointing at malformed JSON, *when* built, *then* `ValueError` mentioning the filename is raised.
  - *Given* a note dated outside the season window (`2026-01-01`), *then* it is dropped (present in no week) and does not raise.

### Unit B — Notes column rendering in `export_schedule_xlsx`

- **Files touched:**
  - `analytics/storage.py` — `export_schedule_xlsx` signature + Notes column rendering + `COL_WIDTHS`.
  - `tests/test_export_schedule_xlsx_notes.py` (NEW).
- **Change summary:** Add `weekend_notes` param; render header in N (after the HEADERS loop, not by modifying HEADERS), spacers L/M blank everywhere, title row N gets `week_bg`, stack notes from first game row; widths. Strictly additive when param is `None`.
- **Depends on:** Unit A — consumes A's `Dict[int, List[str]]` output shape. Develop against the agreed shape; integrate after A lands. Different file from A (no merge conflict).
- **No-mock test outline (Given/When/Then, hand-computed oracles):**
  - *Given* a synthetic `DrawStorage` with 2 weeks, 3 games all in week 1 (one field, three consecutive game rows), and `weekend_notes={1: ["Field: X", "Request: Y", "Preferred: Z"]}`, *when* `export_schedule_xlsx(tmp.xlsx, weekend_notes=...)`, *then* reopening with `openpyxl`: row 1 = week title (col 1 = "Week 1"); row 2 = column headers (N2 = "Notes" with `header_font`/`header_fill`/`thin_border`); row 3 = field sub-header; rows 4–6 = the 3 game rows (cols A–K). Notes column N: N4 = "Field: X", N5 = "Request: Y", N6 = "Preferred: Z". L and M on all rows are `None`. (Oracle: title=row1, headers=row2, field-header=row3, game1=row4 → first_game_row=4; notes stack from row4 in col N; assert N4=="Field: X", N5=="Request: Y", N6=="Preferred: Z", L4 is None, M4 is None, N2=="Notes".)
  - *Given* the same draw with `weekend_notes=None`, *when* exported, *then* every cell in columns L, M, N is `None` and columns A–K match a baseline export (assert A–K equal between the two files for all rows). (Oracle: additive-only guarantee, DoD #5.)
  - *Given* a week with **more** notes than game rows (e.g. 1 game, 4 notes), *then* all 4 notes appear in consecutive N cells (N at first-game-row through first-game-row+3) and bye-row column A still holds team names (N on those rows may hold overflow notes or be empty but column A is unchanged). (Oracle: with 1 game: title=row1, headers=row2, field-header=row3, game=row4; notes → N4, N5, N6, N7; blank separator at row8; BYES label at row9 if any; assert all 4 N cells have the note strings and ws.cell(row=9, column=1).value != None.)

## Doc registry

- `CLAUDE.md` (repo root) — **Export Functions → `DrawStorage.export_schedule_xlsx()`**: document the new `weekend_notes` param and the column layout (A–K games, L/M spacers, N Notes). **Quick Commands**: add the build-and-export snippet (`build_weekend_notes` → `export_schedule_xlsx(..., weekend_notes=...)`). **File Structure**: register `data/{year}/notes.json` and `analytics/notes.py`.
- `docs/system/SYSTEM_OVERVIEW.md` (final-form, confirmed present at `docs/system/SYSTEM_OVERVIEW.md`) — one line under the DrawStorage / export description noting weekend notes and the `notes.json` source.
- `docs/todo/README.md` — no change (status header convention already covered); spec auto-listed by being a `spec-*.md`.
- `data/2026/notes.json` — self-documenting via the committed sample (DoD #6); a leading `_comment` key may describe the schema.

## Out of scope

- **Retrofitting `scripts/export_v4_*.py`** — historical snapshot scripts; convenor chose "library function only". They are frozen artifacts, not maintained code.
- **Making `notes.json` feed the solver** — notes are presentation-only. They never become constraints, never affect variable generation, never influence `PREFERRED_WEEKENDS`' actual soft-constraint behaviour (spec-006 owns that). Adding a `'note'` field to a config entry is inert to the solver (confirmed: `utils.py:_SCOPE_FIELDS` does not include `'note'`; the scope-matcher silently ignores it).
- **Auto-deriving notes from `FIELD_UNAVAILABILITIES`** — excluded by design: those entries are bare `datetime`s with only code-comments, no machine-readable reason; the human "why" (Masters, etc.) is authored in `notes.json` under category `Field`.
- **A dedicated "Notes" worksheet or per-game Notes column** — convenor chose the stacked single-column-per-weekend layout.
- **Per-club / per-grade report notes** (`analytics/reports.py`) — this spec covers only the season schedule xlsx. Not real deferred work; genuinely not requested.

## Dependencies

- **Other plans:** `depends_on: none`. spec-028 touches `analytics/notes.py` (new), `analytics/storage.py`, `config/season_2026.py` (additive `note` keys), `data/2026/notes.json` (new), tests, `CLAUDE.md`, `docs/system/SYSTEM_OVERVIEW.md`. None of these are touched by the active specs 021/022/023 (which live in `constraints/*`, `run.py`, `main_staged.py`), so spec-028 is **independently parallelisable** with all of them. It does **rely on** spec-006 (done) having landed `PREFERRED_WEEKENDS` + `data['preferred_weekends']` — confirmed present at `config/season_2026.py:1049` and `utils.py:4098`. (review fix — Medium: the previous review noted a concurrent-conflict risk with `spec-024` — but spec-024 (field-spread) is already **done** per the dependency tree and is no longer a concurrent concern. The active concurrent risk is **spec-025** (LOCKED_PAIRINGS config), which is `ready` now and rewrites `config/season_2026.py:693-938` (the FORCED_GAMES locked-week entries). spec-028 adds additive `'note'` keys to FORCED_GAMES entries in the same file. If both land concurrently there will be a merge conflict on `config/season_2026.py`. The implementer of whichever lands second must rebase the `'note'` additions onto the post-spec-025 FORCED_GAMES layout. Functionally independent — `depends_on: none` is still correct.)

- **Within this plan:** Unit B depends_on Unit A (consumes its return shape). Unit A is independently completable and testable first.

## Risks & blast radius

- **Column-index drift:** if `HEADERS`/`NUM_COLS` change later, the hardcoded N=14 / L,M spacers could misalign. Mitigation: derive the Notes column as `NUM_COLS + 3` rather than literal 14 so it tracks `HEADERS`.
- **Off-draw date→week resolution:** blocked dates with zero scheduled games rely on 7-day bucketing from the season start; an off-by-one in bucketing would file a note under the wrong week. The hand-computed test oracles pin this down. Bucketing formula (verified): `week = (date - start_date).days // 7 + 1` gives week 9 for `2026-05-17` and week 12 for `2026-06-12` from season start `2026-03-22`.
- **`'note'` key in config entries is safe:** confirmed inert to the solver. `utils.py:_SCOPE_FIELDS` (line 525) is `{'grade', 'day', 'day_slot', 'time', 'week', 'date', 'round_no', 'field_name', 'field_location'}`; `'note'` is not in this set and is never evaluated by the scope-matcher or any constraint code. No ignore-list addition is needed.
- **Season start retrieval:** `data['start_date']` does NOT exist; use `load_season_config(data['year'])['start_date']` which returns a `datetime` object. Convert to `date` for arithmetic: `start_date.date()` if needed for `date - datetime` type safety.

## Open Questions

None.
