<!-- status: done -->
<!-- completed: 2026-05-24 — commit ccc3d07; Mode B verified (9/9 DoD, 90 tests, 93% notes.py coverage, all 6 club_days callsites patched). -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->
<!-- owner: session=opus-clubday-20260524 claimed=2026-05-24 -->

# spec-029 — Club-day weekends in the published-draw Notes column

**Spec source:** convenor request (this session) — "make sure the notes-export spec includes exporting which weekends a club day falls on (a note on the correct weekend saying what the club day is), and add a comment to the CLUB_DAYS dict that becomes the note text."

## Why

spec-028 (done, merged `c077c28`) added a per-weekend **Notes** column to the published season xlsx, fed by five categories: `Field`, `Request`, `Preferred` (from `PREFERRED_WEEKENDS`), `Blocked` (from `BLOCKED_GAMES`), `Forced` (from `FORCED_GAMES`). **Club days are not one of those sources** — yet a club day (Crusaders Jun 14, University Jul 26 for 2026) is exactly the kind of "why does this weekend look special" context the convenor wants surfaced on the published sheet. The only `club_days` consumers that already tolerate dict values are solver constraint atoms routed through `normalize_club_day` (`constraints/atoms/_club_day_shared.py` → the `ClubDay` atom group; `constraints/unified.py` → passes through `normalize_club_day` at line 221). Six other direct-access callsites must be updated as part of this spec (see Risks &amp; blast radius). Nothing carries the club-day fact through to the exported notes. (review fix — H2: corrected false "only consumers" claim) The convenor re-explains "this is Crusaders' club day" by hand every publish. Surfacing it as a `Club Day:` note on the correct weekend removes that manual step, consistent with spec-028's design.

`CLUB_DAYS` entries are currently bare `datetime`s with a Python `#` comment (`config/season_2026.py:179`). A `#` comment is not machine-readable at export time, so the note text must live as a data field. `normalize_club_day` (`utils.py:117`) already accepts a dict form `{'date': ..., 'opponent': ...}` and ignores any extra keys, so adding a `'note'` key is confirmed inert to the solver and the constraint atoms.

## Design

### New note category: `Club Day`

A sixth category joins spec-028's five, **opt-in** on the same principle: a club day is surfaced as a note **iff its `CLUB_DAYS` value carries a truthy `'note'`**. This keeps seasons whose `CLUB_DAYS` are still bare `datetime`s (2025, 2027, template) silent — strictly additive, zero behavioural surprise — exactly mirroring spec-028's opt-in contract for the auto-derived sources.

| Category | Source mechanism | How a note is produced |
|----------|------------------|------------------------|
| `Club Day` | auto-derived from `data['club_days']` (`CLUB_DAYS`) | Each entry whose value is a dict with a truthy `'note'` → `"Club Day: <text>"`. `'note': True` falls back to `"<ClubName> Club Day"`. A bare-`datetime`/`str` value, or a dict without `'note'`, is **never surfaced**. |

### `CLUB_DAYS` value grammar (extended, back-compat-free per forward-only)

`normalize_club_day` already supports three input shapes; the dict shape gains an optional `'note'`:

```python
CLUB_DAYS = {
    'Crusaders':  {'date': datetime(2026, 6, 14), 'note': 'Crusaders Club Day'},
    'University': {'date': datetime(2026, 7, 26), 'note': 'University Club Day'},
}
```

- `'date'`: required (datetime or ISO string) — already consumed by `normalize_club_day`.
- `'opponent'`: optional — already consumed by `normalize_club_day` (unchanged).
- `'note'`: optional, NEW — read **only** by the notes builder; ignored by `normalize_club_day` and every constraint atom that already calls `normalize_club_day`. `str` → used verbatim as the note text; `True` → falls back to `"<ClubName> Club Day"`.

This is a **format extension, not a break**: the bare-`datetime` form stays valid everywhere (2025/2027/template untouched), and the dict form already round-trips through `normalize_club_day`. Callers that already call `normalize_club_day(value)` before using the date are unaffected. **However, six callsites access `club_days` values directly without going through `normalize_club_day` and must be updated** (see Risks & blast radius and Files touched). (review fix — C1: corrected false "inert everywhere" claim)

### Builder change — `analytics/notes.py`

1. Extend `_CATEGORY_ORDER` to place `Club Day` in the stable per-week ordering. New ordering: `Field=0, Request=1, Club Day=2, Preferred=3, Blocked=4, Forced=5`; unknown → `6` (`_DEFAULT_CATEGORY_ORDER` bumped 5→6).
2. Add `_derive_from_club_days(club_days, date_to_week, start_date, end_date) -> List[tuple]` mirroring the shape/return contract of `_derive_from_config`:
   - Iterate `club_days.items()` (a `{club_name: value}` dict).
   - Resolve the note text **before** touching the date so a no-`note` entry is cheaply skipped: only `dict` values with a truthy `'note'` proceed; everything else (`datetime`, `str`, or dict without `'note'`) is skipped (opt-in). `'note': True` → text `f"{club_name} Club Day"`; `'note': <str>` → that string.
   - Resolve the date via `normalize_club_day(value)[0]` (reuses the canonical parser; the `[0]` is the date, `opponent` is unused here). Convert to an ISO string with a single helper `_club_day_iso(date_val)` that handles `datetime`/`date`/`str` (`datetime`/`date` → `.strftime('%Y-%m-%d')` via the existing `date.date()` pattern used in `_club_day_shared.py:date.date().strftime(...)`; `str` → used as-is). If the value can't yield a date, log INFO and skip (dark-path logged, never silent).
   - `week = _resolve_week(iso, date_to_week, start_date, end_date, source=f"Club Day:{club_name}")`; on `None`, skip (already INFO-logged inside `_resolve_week`).
   - Append `(_category_sort("Club Day"), "Club Day", text, week)`.
3. Wire it into `build_weekend_notes`: add `raw.extend(_derive_from_club_days(data.get("club_days", {}), date_to_week, start_date, end_date))` alongside the existing `_derive_from_config`/`_derive_from_preferred` calls. Note `club_days` defaults to `{}` (dict, not list) — guard accordingly.
4. Update `build_weekend_notes`'s docstring (the "Sources" list and the `data` arg's "Must contain" list) to include `club_days`.

The renderer (`export_schedule_xlsx`) is **untouched**: it consumes the `Dict[int, List[str]]` opaquely and renders any `"Category: text"` line; a `"Club Day: …"` line stacks in column N like any other. Verified against `analytics/storage.py` Notes rendering (spec-028 Unit B) — no per-category logic exists there.

## Definition of Done

1. `analytics/notes.py` has `_derive_from_club_days(club_days, date_to_week, start_date, end_date)` returning the `(sort_key, category, text, week)` tuple shape, wired into `build_weekend_notes` via `data.get("club_days", {})`.
2. `_CATEGORY_ORDER` contains `"Club Day": 2` with `Preferred/Blocked/Forced` renumbered to `3/4/5` and `_DEFAULT_CATEGORY_ORDER == 6`; the per-week output orders a `Club Day` line after `Request` and before `Preferred`.
3. Opt-in proven: a `CLUB_DAYS` entry whose value is a bare `datetime`, a bare ISO `str`, or a dict **without** `'note'` produces **no** note; a dict with `'note': <str>` produces `"Club Day: <str>"`; a dict with `'note': True` produces `"Club Day: <ClubName> Club Day"`.
4. Date resolution: a club day on a date present in the draw resolves to that game's week; a club day on a date with no scheduled game resolves via the 7-day bucket from `start_date`; an out-of-season club-day date is dropped (no crash) and INFO-logged.
5. `config/season_2026.py` `CLUB_DAYS` converted to the dict-with-`'note'` form for both entries (Crusaders, University), preserving each date and any opponent semantics. Loading `load_season_data(2026)` still succeeds and the `ClubDay` atom group still sees both club days (constraint behaviour unchanged). All six direct-access callsites in `utils.py`, `constraints/soft.py`, `constraints/unified.py`, `analytics/tester.py`, and `analytics/preseason_report.py` updated to call `normalize_club_day(value)` before extracting the date — confirmed by running `load_season_data(2026)` (which invokes `_check_club_days_availability`) without fatals. (review fix — C2: original DoD#5 falsely claimed `_check_club_days_availability` wouldn't error; it crashes on dict values because it does `hasattr(club_date, 'strftime')` which is False for a dict, falling back to `str({'date': ..., 'note': ...})` which never matches a timeslot date, firing a fatal)
6. End-to-end: building `build_weekend_notes(draw, load_season_data(2026))` against a `DrawStorage` that has games on `2026-06-14` and `2026-07-26` yields a `"Club Day: Crusaders Club Day"` line in week-of-Jun-14 and a `"Club Day: University Club Day"` line in week-of-Jul-26.
7. Strictly additive: with every `CLUB_DAYS` entry lacking `'note'` (e.g. the 2025 config), `build_weekend_notes` returns byte-identical output to pre-spec-029 for the non-club-day categories (no spurious `Club Day` lines, no reordering of existing categories — verified by the renumber being order-preserving for Field/Request and pushing Preferred/Blocked/Forced down uniformly).
8. Tests in `tests/test_weekend_notes.py` cover DoD 3, 4, 6, 7 as no-mock Given/When/Then with hand-computed week oracles; new/changed code in `analytics/notes.py` ≥85% covered. Type-check clean; changed-file lint clean; AST sweep clean (no dead code / unlogged dark paths).
9. Docs updated per the doc registry below.

## Implementation units

Single unit. The work is one cohesive, demonstrable feature (~40 LOC of derivation + a two-entry config edit + tests + docs); the config note and the derivation logic are interdependent — neither is independently *demonstrable* (the end-to-end DoD #6 needs both), and they share no awkward file boundary worth parallelising. Splitting a change this small into two worktrees costs more coordination than it saves. Graded S2 for the new code path + new public-ish category, executed as one unit on one worktree.

### Unit A — Club-day notes source + config note text + docs

- **Files touched:**
  - `analytics/notes.py` — `_CATEGORY_ORDER` (+`Club Day`, renumber, bump default), new `_derive_from_club_days()` + `_club_day_iso()` helper, wire into `build_weekend_notes`, docstring update.
  - `config/season_2026.py` — `CLUB_DAYS`: convert both entries to `{'date': datetime(...), 'note': '<text>'}`. No other config touched.
  - `utils.py` — `_check_club_days_availability` (line ~1836) and `_check_club_days_vs_blocked` (line ~1931): both iterate `club_days.items()` and call `club_date.strftime(...)` / `hasattr(club_date, 'strftime')` directly. Replace with `date_val, _ = normalize_club_day(club_date); date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)` in each loop body. (review fix — C1/C2)
  - `constraints/soft.py` — `ClubDayConstraintSoft.apply` (line ~143): `desired_date = club_days[club_name]` then `desired_date.strftime(...)` — crashes on dict. Replace with `desired_date, _ = normalize_club_day(club_days[club_name])`. Import `normalize_club_day` from `utils` (already used in `_club_day_shared`). (review fix — C1)
  - `constraints/unified.py` — `_apply_club_day_inter_week` (line ~716): `desired_date.date().strftime(...)` — crashes on dict. Replace with `date_val, _ = normalize_club_day(desired_date); date_str = date_val.date().strftime('%Y-%m-%d') if hasattr(date_val, 'date') else str(date_val)`. (review fix — C1)
  - `analytics/tester.py` — `check_club_day_constraint` (line ~1936): uses `hasattr(desired_date, 'strftime')` fallback to `str(...)` — produces garbage string for dict values, silently matching no games. Replace with `normalize_club_day(desired_date)` call before the date extraction. (review fix — C1)
  - `analytics/preseason_report.py` — `get_special_requests` (line ~162) and `get_calendar_events` (line ~280): both guard with `isinstance(date, datetime)` which silently skips dict values. Replace with `normalize_club_day(date)` and check `isinstance(date_val, datetime)` on the unwrapped value. (review fix — C1)
  - `tests/test_weekend_notes.py` — add club-day scenarios (extend the existing spec-028 test module; no existing test modified, only new test functions appended — avoids touching other units' assertions). Update `_minimal_data()` to accept a `club_days=None` kwarg defaulting to `{}` and include it in the returned dict. (review fix — M2)
  - `CLAUDE.md` — Export Functions / notes section: add `Club Day` to the category list and document the `CLUB_DAYS` `'note'` field.
  - `docs/system/SYSTEM_OVERVIEW.md` — one line: weekend notes now include club days from `CLUB_DAYS['…']['note']`.
- **Change summary:** Additive derivation + config field + docs. Also requires compat patches to six direct-access callsites in `utils.py`, `constraints/soft.py`, `constraints/unified.py`, `analytics/tester.py`, and `analytics/preseason_report.py` that treat `club_days` values as raw datetimes — these must call `normalize_club_day()` first. All callers already in scope of this unit (no new worktree needed — they are touched iff CLUB_DAYS format changes). (review fix — H1)
- **Depends on:** none.
- **Executor model:** Opus (small but touches the shared `_CATEGORY_ORDER` renumber, which must stay order-preserving — subtle enough to not hand to Sonnet on the line).
- **No-mock test outline (Given/When/Then, hand-computed oracles):**

  **Test infrastructure note:** extend `_minimal_data()` in `test_weekend_notes.py` to accept a `club_days=None` kwarg (default `{}`) and include `'club_days': club_days or {}` in the returned dict. All new tests use this extended helper. (review fix — M2: existing helper lacked `club_days` key)

  - *Given* a synthetic `data` dict with `club_days = {'Crusaders': {'date': datetime(2026,6,14), 'note': 'Crusaders Club Day'}}` and a synthetic `DrawStorage` with one `StoredGame` on `'2026-06-14'` at `week=13`, and `start_date=2026-03-22`, *when* `build_weekend_notes(draw, data)`, *then* `result[13]` contains `"Club Day: Crusaders Club Day"`. (Oracle: game maps `2026-06-14`→week 13 directly; assert membership.)
  - *Given* `club_days = {'University': {'date': datetime(2026,7,26)}}` (dict, **no** `'note'`) and a draw with a game on that date, *then* the formatted string `"Club Day: …"` is absent from **every** week (opt-in proven). (Oracle: assert no value in the result dict contains a `Club Day:` prefix.)
  - *Given* `club_days = {'Crusaders': datetime(2026,6,14)}` (bare datetime) **and** `club_days = {'Norths': {'date': datetime(2026,7,26), 'note': True}}` in a second case, *then* the bare-datetime entry yields no note, and the `'note': True` entry yields `"Club Day: Norths Club Day"`. (Oracle: fallback text == `f"{club} Club Day"`.)
  - *Given* a club day on `'2026-06-14'` with **no** scheduled game on that date (empty draw — draw has game on a different date) and `start_date=2026-03-22`, *then* it resolves to week 13 via 7-day bucketing. (Oracle: `(2026-06-14 − 2026-03-22).days = 84`; `84//7 + 1 = 13`.)
  - *Given* a club day dated `'2026-01-01'` (before season), *then* it is dropped (in no week) and does not raise. (Oracle: `(2026-01-01 − 2026-03-22).days = -80`; `-80//7 + 1 = -11`; week -11 < 1 → dropped.)
  - *Given* the real `load_season_data(2026)` and a `DrawStorage` with games on `'2026-06-14'` (`week=13`, `game_id='G00001'`) and `'2026-07-26'` (`week=19`, `game_id='G00002'`), *then* `result[13]` ∋ `"Club Day: Crusaders Club Day"` and `result[19]` ∋ `"Club Day: University Club Day"`. (Oracle: `(Jun 14 − Mar 22).days = 84 → 84//7+1 = 13`; `(Jul 26 − Mar 22).days = 126 → 126//7+1 = 19`; game-based resolution matches bucket. **Unique game_ids required** — `_make_game` defaults to `"G00001"` for all; pass distinct ids explicitly. Week numbers on the synthetic StoredGames drive the draw-map lookup, so assert against whatever week the synthetic games declare.) (review fix — M1: added unique game_id reminder)
  - *Given* a week that already has a `Request` note and a `Preferred` note plus a `Club Day` note, *then* the output order is `Request` line, then `Club Day` line, then `Preferred` line. (Oracle: `_CATEGORY_ORDER`: Request=1 < Club Day=2 < Preferred=3.)

## Doc registry

- `CLAUDE.md` (repo root) — the weekend-notes / `export_schedule_xlsx` description: add `Club Day` to the category list and document `CLUB_DAYS[...]['note']` as the note source. One line under the club-days mention if one exists.
- `docs/system/SYSTEM_OVERVIEW.md` — append `Club Day` to the notes-source enumeration added by spec-028.
- `docs/todo/00-dependency-tree.md` — add spec-029 node (depends_on: none; ready-to-start) while live; on completion mark done and return the folder to "no live specs".
- `data/2026/notes.json` — **no change required**: club-day notes are auto-derived from `CLUB_DAYS`, not hand-authored. (Stated explicitly so a reviewer doesn't expect a sample entry.)

## Out of scope

- **Auto-deriving club days from anything other than `CLUB_DAYS`** — the dict is the single source of truth; no inference from team names or fixtures.
- **Changing `export_schedule_xlsx`** — the renderer is category-agnostic; spec-028 owns it and it needs no change for a new category. Confirmed by inspection, not assumption.
- **Back-filling `'note'` into 2025/2027/template `CLUB_DAYS`** — those seasons stay bare-`datetime` (opt-in keeps them silent); no behavioural reason to touch them.
- **Per-club / per-grade report notes** (`analytics/reports.py`) — spec-028's same exclusion; not requested.

## Dependencies

- **Other plans:** `depends_on: none`. spec-028 (the notes infrastructure this extends) is `done` and merged; this spec consumes its `build_weekend_notes`/`_derive_*` contract and `export_schedule_xlsx` Notes column as-is. No other live spec exists (dependency tree shows all done), so there is zero concurrent-file-conflict risk.
- **Within this plan:** single unit; no internal dependencies.

## Risks & blast radius

- **`_CATEGORY_ORDER` renumber:** pushing `Preferred/Blocked/Forced` from `2/3/4` to `3/4/5` is order-preserving (relative order of all existing categories is unchanged; only the absolute ints shift and a new value slots between Request and Preferred). The dedup/sort in `build_weekend_notes` keys only on the sort int, so existing weeks' line ordering is unaffected except where a `Club Day` line is newly interleaved. DoD #7 + the ordering test pin this.
- **`club_days` is a dict, not a list:** unlike `blocked_games`/`forced_games`/`preferred_weekends` (lists), `data['club_days']` is a `{club: value}` dict. The new helper must iterate `.items()` and default to `{}` — a `.get("club_days", [])` (list default) would still iterate but yield wrong shapes; the plan specifies `{}`.
- **`'note'` key inertness — PARTIAL, NOT TOTAL:** `normalize_club_day` reads only `value['date']`/`value.get('opponent')`; `_club_day_shared.parse_club_day_entries` calls `normalize_club_day` and never inspects other keys; `constraints/unified.py:_apply_club_day_intra_day` (line ~980) only iterates club names and never touches the value — these are all safe. **But** six other callsites access the value directly as a datetime and do NOT call `normalize_club_day` first: `utils._check_club_days_availability` (~1836), `utils._check_club_days_vs_blocked` (~1931), `constraints/soft.py:ClubDayConstraintSoft.apply` (~143), `constraints/unified.py:_apply_club_day_inter_week` (~716), `analytics/tester.py:check_club_day_constraint` (~1936), `analytics/preseason_report.py:get_special_requests` (~162) and `get_calendar_events` (~280). Converting CLUB_DAYS to dict form without patching these will: crash `soft.py` and `unified.py:716` with `AttributeError` (no `strftime`/`date` on dict); corrupt `tester.py` silently (garbage `date_str` mismatches every game); fire a fatal in `_check_club_days_availability` (garbage string never matches timeslot dates); silently omit club days from the preseason report. All must be patched as part of this unit. (review fix — C1)
- **Date typing:** `CLUB_DAYS` dates are `datetime` objects; `_resolve_week` expects an ISO string. The `_club_day_iso` helper normalises before resolution — an un-normalised `datetime` passed to `_resolve_week`'s `date.fromisoformat` path would raise; the helper + its test guard this. (review note — Low: `normalize_club_day` returns the raw value it received, which is already a `datetime` for the common case — so the `.date()` sub-call in `_club_day_iso` strips the time component unnecessarily but harmlessly; the `strftime` call works on both `datetime` and `date` objects. Keep the defensive handling; it's correct.)

## Open Questions

None.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Autonomous: run end-to-end without waiting for the user, except where this hits `blocked`. -->
1. Status must be `ready` (carries a `reviewed:` stamp from /adversarial Mode A). If `review_pending`/`under_review`, let review finish — do not implement. If `blocked`, STOP: the Open Questions need the user.
2. Stamp `building`, claim `owner`. You are the orchestrator (Opus).
3. Single unit (Unit A): own worktree+branch; implement on Opus (subtle `_CATEGORY_ORDER` renumber). Run the S2 gates: type-check, changed-file lint, AST dead-code/dark-path sweep, no-mock Given/When/Then tests (≥85% on changed `analytics/notes.py` code), docs updated.
4. After implementing, launch `/adversarial` Mode B to verify the diff against this plan's DoD. Route fixes, re-verify. NEVER merge unverified.
5. Merge → push origin → post-merge verify (build `load_season_data(2026)` + a synthetic-draw `build_weekend_notes` showing the two club-day lines) → remove worktree. Tick the unit's checkbox.
6. Stamp the plan `done`, move it to `docs/todo/done/` per repo convention, and update `docs/todo/00-dependency-tree.md` back to "no live specs".
