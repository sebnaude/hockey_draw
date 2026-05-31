<!-- status: ready -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->
<!-- owner: session=unclaimed -->
<!-- reviewed: adversarial Sonnet review 2026-05-30 — fixes applied inline -->

# spec-043 — Pre-draw venue-capacity feasibility precheck (UI-callable)

> Branches off **final-form**. Authored as research+plan only; **no implementation** until the
> user gives an explicit go-ahead (per `basic`).

## Why

Before a multi-hour/day solve, the convenor needs to know whether the season's *timeslot supply*
can physically hold the season's *game demand* at each venue — and, specifically, whether a
Broadmeadow (NIHC) Sunday needs all 8 timeslots or could drop the worst slot (19:00). Today there
is no clean way to ask this: the capacity logic that exists (`utils._check_scheduling_feasibility`,
Phase 20) runs only *inside* `validate_game_config(data)` on the fully-**built** data dict (teams
loaded from CSV, `timeslots` already generated), prints free-text, and accumulates `warnings`/
`fatals` lists. It is not callable from a UI editing raw *fields and teams*, and it does **not**
compute the two diagnostics the convenor actually wants: (a) an **away-venue per-day floor**
(available slots on a playing day ≥ number of home teams there), and (b) a **Broadmeadow
per-Sunday slot-demand** with a **"can the 19:00 slot be dropped?"** recommendation that accounts
for removed weekends and partial-day blocks. The cost of not having this: the convenor either
over-provisions slots (worst-timeslot games nobody wants) or discovers infeasibility only after a
wasted long solve.

This spec adds a **standalone, config-primitives-in / structured-dict-out** capacity analyser, a
`precheck` CLI wrapper, and an auto-run hook before `generate`. It deliberately **does not** modify
the existing Phase-20 check (user decision: leave it in place; some overlap is accepted).

## What already exists (gap analysis — much of this is partly built)

The repo already has substantial pre-solve feasibility machinery. The new module must **reuse**, not
duplicate (forward-only):

| Existing | Where | Covers | Does **not** cover (the gap) |
|---|---|---|---|
| `validate_game_config(data)` + ~20 `_check_*` phases | `utils.py:3316` (called from `generate_X` at `utils.py:3545`) | Orchestrates all pre-solve config checks; emits `warnings`/`fatals`; runs on the built `data` | Not a pure/UI function; prints; no structured return |
| `_check_scheduling_feasibility` (Phase 20) | `utils.py:2631` | Friday slot counts; **Gosford/Maitland Sunday season capacity vs `(rounds+1)//2` home demand**; total-Friday-PHL ≤ total-PHL; unique weeks/grade; per-grade season slot capacity | **No per-day away floor**; **no Broadmeadow per-Sunday slot-need / drop-slot recommendation**; needs built `timeslots`; left untouched per user |
| `_check_team_capacity` | `utils.py:1628` | Per-team playable-date count after grade/home/block filters | Team-level, not venue-day capacity |
| `PreSeasonReport.get_slot_capacity_analysis(total_matchups)` | `analytics/preseason_report.py:434` | **Coarse aggregate**: `min_slots_per_weekend = ceil(total_matchups / weekends)`; sums `len(times)*num_fields` across **all** venues; `capacity_ratio` | Not Broadmeadow-isolated; does **not** subtract away-venue or Friday-PHL games; **no per-field slots-needed**; **no drop-19:00 decision with headroom**; **no away per-day floor**; **no all-grade BLOCKED_GAMES partial-day removal**; takes built `data`+`config`, not primitives |
| `PreSeasonReport.calculate_available_weekends()` | `analytics/preseason_report.py:59` | Counts Sundays start→end minus blocked weekends | Subtracts **only** NIHC `field_unavailabilities['…']['weekends']` (Saturday-marker match) — **not** `whole_days` / `part_days`. The new module reuses `generate_timeslots` (which *does* handle whole_days/part_days), **not** this method, so the limitation does not propagate. |
| `scripts/verify_draw_feasibility.py`, `scripts/analyze_phl_feasibility.py` | `scripts/` | Ad-hoc feasibility scripts | Not the convenor-facing capacity/drop-slot question |

**Net genuinely-new work:** (1) a **pure primitives-in function** (UI contract); (2) **Broadmeadow-
isolated per-Sunday slot-demand** with the **drop-the-19:00-slot recommendation + headroom**;
(3) the **away-venue per-day floor** (slots ≥ home teams); (4) **all-grade partial-day BLOCKED_GAMES**
capacity removal. Everything else is reuse (`generate_timeslots`, the `(rounds+1)//2` convention,
the venue/field slot-counting pattern from `get_slot_capacity_analysis`).

## Scope decisions (locked with the user — see Open Questions = 0)

1. **Broadmeadow game count** = `total_season_games_all_grades − Maitland_home_games −
   Gosford_home_games` (everything not hosted at an away venue is at Broadmeadow), spread over the
   count of playable Broadmeadow Sundays. *(review fix — M4: this identity is correct **only** when
   the away venues are exactly the non-central `home_field_map` values. Unit A must read the distinct
   `home_field_map` values, treat each as an away venue, and `away_games_excluded = Σ over teams of
   away clubs`. If a `home_field_map` value other than the two known venues appears, emit a `warnings`
   line — the identity still holds as "total − Σ all away-club home games", but flag the new venue so
   the convenor notices.)*
2. **Capacity model = flat day-capacity** (`fields × times`) for the primary number, **plus** a
   *secondary* note comparing the PHL+2nd subset against **EF+WF-only** capacity (PHL/2nd cannot use
   South Field). No full per-grade time-window bin-packing in v1.
3. **Surfaces (all three):** standalone module + pure function; `run.py precheck` CLI; auto-run
   before `generate`.
4. **Drop recommendation:** recommend dropping the **last** Sunday slot (19:00 at NIHC) only if
   peak demand still fits in the remaining slots with a **configurable headroom buffer**
   (`headroom`, default 1 spare slot).
5. **Away-venue check = per-day floor:** on each playing day at an away venue, available game-slots
   that day ≥ number of home teams whose club maps to that venue.
6. **Input contract = core (primitives) + thin wrapper.** Core takes the raw config pieces (so a UI
   editing fields/teams passes them directly). An `analyze_capacity_for_year(year)` wrapper loads a
   season's data and unpacks to call the core (for the CLI / auto-run).
7. **Partial days:** capacity counting subtracts **both** `FIELD_UNAVAILABILITIES`
   (weekend/whole-day/part-day, already handled by `generate_timeslots`) **and** all-grade
   `BLOCKED_GAMES` time-of-day removals. *(review fix — M3: confirmed against the real config. The
   two qualifying entries at the tail of `season_2026.py` BLOCKED_GAMES are May 24
   `{date:'2026-05-24', time:['16:00','17:30','19:00'], field_location:'Newcastle International
   Hockey Centre'}` and Aug 16 `{date:'2026-08-16', time:['08:30','10:00','11:30'],
   field_location:'Newcastle International Hockey Centre'}` — both have **no** `teams`/`club`/`grade`/
   `grades` keys, so they qualify as all-grade. The `time` value is a **list** of `'HH:MM'` strings,
   not a scalar — Unit A's matcher must accept scalar **or** list for every key it checks.)* A block
   reduces capacity only when it has none of `teams`/`club`/`grade`/`grades` and matches the slot on
   every key it *does* specify among `field_location`, `date`, `time`. Grade/club/team-scoped blocks
   (e.g. the `{club:'Gosford', time:[...], field_location:...}` entry immediately following in the
   list) do **not** reduce raw capacity and are ignored here.
8. **Relationship to Phase 20:** new module is **separate**; Phase 20 is left untouched. The new
   module reuses `generate_timeslots()` and the `(rounds+1)//2` home-game convention so numbers are
   consistent, but does not call into or refactor `_check_scheduling_feasibility`.

## Definition of Done

A unit is done only when its criteria are observable and verified (`/adversarial` Mode B).

1. New module `analytics/capacity_precheck.py` exposes a pure function
   `analyze_capacity(teams, fields, day_time_map, start_date, end_date, field_unavailabilities,
   num_rounds, home_field_map, *, blocked_games=None, headroom=1,
   central_venue='Newcastle International Hockey Centre') -> dict` that performs **no** file I/O,
   **no** solver calls, and does not mutate its inputs.
2. The returned dict has exactly these top-level keys: `central_venue`, `away_venues` (dict keyed by
   venue), `broadmeadow` (dict), `fatals` (list[str]), `warnings` (list[str]), `ok` (bool ==
   `len(fatals)==0`).
3. `away_venues[v]` contains: `home_clubs` (list), `home_team_count` (int), `min_slots_any_day`
   (int), `days_below_floor` (list of `{date, slots, needed}`), `ok` (bool). A day where available
   slots `<` home_team_count appends a `fatals` entry naming the venue, date, slot count, and team
   count.
4. `broadmeadow` contains at least: `total_games_all_grades`, `away_games_excluded`,
   `broadmeadow_games`, `friday_phl_games_excluded`, `broadmeadow_sunday_games`, `playable_sundays`,
   `fields`, `times_per_sunday`, `capacity_per_sunday_full` (`fields * times_per_sunday`),
   `capacity_per_sunday_drop_one` (`fields * (times_per_sunday - 1)`), `games_per_sunday` (=
   `ceil(broadmeadow_sunday_games / playable_sundays)`), `slots_needed_per_sunday` (=
   `ceil(games_per_sunday / fields)`), `headroom`, `can_drop_last_slot` (bool), `recommendation`
   (str), and `phl_2nd_efwf` (dict: `phl_2nd_games`, `efwf_capacity_per_sunday`, `phl_2nd_per_sunday`,
   `ok`).
5. **`can_drop_last_slot` is True iff** `games_per_sunday + headroom <= fields * (times_per_sunday
   - 1)`. The `recommendation` string states drop-vs-keep and the headroom used.
6. Capacity counting subtracts all-grade `BLOCKED_GAMES` time removals (proven by a synthetic test
   where a date+time+field_location block reduces a Sunday's slot count by `fields`), and ignores
   grade/club/team-scoped blocks (proven by a second synthetic block that must **not** change
   capacity).
7. `analyze_capacity` reuses `utils.generate_timeslots` for calendar expansion (no re-implemented
   weekend/whole-day/part-day logic).
8. A thin wrapper `analyze_capacity_for_year(year, *, headroom=1) -> dict` obtains the season's
   `teams`, `fields`, and `num_rounds` from `config.load_season_data(year)` (these require built
   `Team` objects — `load_season_config` returns only the raw dict and does **not** build teams) and
   the raw `day_time_map` / `start_date` / `end_date` / `field_unavailabilities` / `home_field_map` /
   `blocked_games` / `constraint_defaults` / `forced_games` from the same `data` dict (or
   `load_season_config`), then calls the core. The wrapper does the config/CSV loading the core
   forbids. *(review fix — H1/M2: original text said "loads `load_season_config(year)` + team list";
   `load_season_config` (`config/__init__.py:28`) returns the raw config with **no** Team objects —
   only `load_season_data` (`config/__init__.py:50`) builds `data['teams']`. Using the raw config
   would force re-implementing CSV parsing, which is forbidden forward-only.)*
9. `run.py precheck --year YYYY [--json PATH] [--headroom N]` prints a human-readable report
   (fatals, warnings, the Broadmeadow per-Sunday verdict, away-venue floors) and, with `--json`,
   writes the result dict; exit code is non-zero iff `fatals` is non-empty.
10. `generate` runs the precheck before the solve starts (in **both** solver entry points — see
    Unit C) and prints its fatals/warnings; fatals print a clear banner and abort the solve unless
    `--force` is passed.
11. No-mock Given/When/Then tests with hand-computed oracles cover: away-venue floor pass + fail;
    Broadmeadow drop=safe, drop=unsafe, and exactly-at-headroom boundary; all-grade block reduces
    capacity; scoped block does not; PHL/2nd EF+WF secondary note. Plus one integration test that
    runs `analyze_capacity_for_year(2026)` and asserts the dict shape + sane invariants (no magic
    constants that real-config edits would break).
12. `type-check`/compile clean on all touched files; changed-file lint clean; tests green; ≥85%
    coverage on new code in `analytics/capacity_precheck.py`.
13. Docs updated (see Doc registry).

## Implementation units

### Unit A — core analyser module + tests  *(depends_on: none; executor: Opus — subtle math/edge cases)*

**Files touched**
- `analytics/capacity_precheck.py` (new) — `analyze_capacity()` core + small private helpers
  (`_count_sunday_slots`, `_apply_allgrade_blocks`, `_away_venue_floors`, `_broadmeadow_demand`).
  Logging via the project's standard logger (reuse the `logging_setup`-style verbose/simple pair if
  `analytics/` already exposes one; otherwise the existing logger used by `analytics/`); **no `print`
  in the core** — dark-path skips (e.g. a dropped slot, an unknown away venue) log at INFO.
- `tests/test_capacity_precheck.py` (new) — synthetic G/W/T scenarios with hand-computed oracles.

**Change summary**
1. Derive `teams_by_grade`, `teams_by_club`, and `venue → home clubs` from `home_field_map`. The
   away venues are the **distinct values** of `home_field_map`; the central venue is the
   `central_venue` arg (NIHC). Warn if a `home_field_map` value is neither known venue (review fix M4).
2. Expand the calendar by calling `utils.generate_timeslots(start_date, end_date, day_time_map,
   fields, field_unavailabilities)` → list of slot dicts (each has `date, day, time, week, day_slot,
   field, round_no`).
3. Subtract all-grade `BLOCKED_GAMES` time removals: drop a slot when a block entry has **no**
   `teams`/`club`/`grade`/`grades` and matches the slot on every key it *does* specify among
   `field_location` (vs `slot.field.location`), `date` (vs `slot.date`), `time` (vs `slot.time`) —
   each block value may be a scalar or a list (membership test). Log each dropped slot at INFO.
4. **Away-venue floor:** for each away venue, group surviving slots by date; `min_slots_any_day` =
   min over playing dates of slot count; any date with `slots < home_team_count` →
   `days_below_floor` entry + a `fatals` line.
5. **Broadmeadow demand:**
   - `total_games[g] = num_rounds[g] * len(teams_in_grade[g]) // 2` (matches
     `_check_scheduling_feasibility:2685`); `total_games_all_grades = Σ_g total_games[g]`.
   - `away_games_excluded = Σ_{team whose club is an away club} (num_rounds[team.grade] + 1) // 2`
     (its home games, hosted at its own away venue — matches the `(rounds+1)//2` convention at
     `utils.py:2743/2761`). `broadmeadow_games = total_games_all_grades − away_games_excluded`.
   - `friday_phl_games_excluded` = **count of central-venue PHL Friday games actually planned**:
     count `FORCED_GAMES` entries with `day=='Friday'`, `field_location==central_venue`,
     `grade=='PHL'`, and `constraint in (None,'equal')`; **fall back** to
     `constraint_defaults['max_friday_broadmeadow']` only when no such forced games exist.
     `broadmeadow_sunday_games = max(0, broadmeadow_games − friday_phl_games_excluded)`. *(review fix
     — H2: original priority was inverted (max first). Using `max_friday_broadmeadow` (=3) when the
     true forced count is lower **over-subtracts** Sunday demand → a falsely optimistic "can drop
     19:00". Forced-count-first makes any residual error a **safe under-count** of Sunday demand.
     Note: NIHC has **no** Friday entry in `DAY_TIME_MAP` (review fix M5) — Friday NIHC PHL slots come
     from `PHL_GAME_TIMES`, so `generate_timeslots` produces only Sunday NIHC slots and this term is a
     pure demand-side adjustment.)*
   - `playable_sundays` = distinct Sunday dates among surviving central-venue slots.
   - `fields` = distinct central-venue field names (3 for NIHC: EF/WF/SF); `times_per_sunday` =
     `len(day_time_map[central_venue]['Sunday'])` (8 for NIHC; `DAY_TIME_MAP[central_venue]` has only
     a `'Sunday'` key — review fix M5).
   - `games_per_sunday = ceil(broadmeadow_sunday_games / playable_sundays)`;
     `slots_needed_per_sunday = ceil(games_per_sunday / fields)`.
   - `can_drop_last_slot` per DoD #5; build `recommendation`.
   - **Secondary PHL/2nd note:** `phl_2nd_games = (total_games[PHL] − PHL away-venue home games) +
     (total_games[2nd] − 2nd away-venue home games)`, where away-venue home games for a PHL/2nd team
     use the same `(rounds+1)//2` convention (so the EF+WF subset excludes PHL/2nd games hosted at
     Maitland/Gosford). `efwf_capacity_per_sunday = (fields − SF_field_count) * times_per_sunday`
     (`fields − 1` for NIHC → `2 * 8 = 16`); `phl_2nd_per_sunday = ceil(phl_2nd_games /
     playable_sundays)`; warn if `phl_2nd_per_sunday > efwf_capacity_per_sunday`. *(review fix — H3:
     original used the raw `total_games[PHL]+total_games[2nd]`, which double-counts PHL Friday/Sunday
     games at Gosford (8 Friday + Sundays) and Maitland (2 Friday + Sundays) that never touch EF/WF,
     making the EF+WF check far too pessimistic.)*
6. Return the dict per DoD #2–4; `ok = not fatals`.

**No-mock test outline (hand-computed oracles)**
- *Away floor — fail:* synthetic away venue with 1 field × 2 Sunday times (2 slots/day) and 3 home
  teams → expect `min_slots_any_day=2`, a `fatals` entry, `ok=False`. **Oracle:** 2 < 3.
- *Away floor — pass:* same venue, 2 home teams → no fatal, `away_venues[v].ok=True`. **Oracle:** 2 ≥ 2.
- *Broadmeadow drop=safe:* construct teams/rounds so `broadmeadow_sunday_games=126`,
  `playable_sundays=21`, `fields=3`, `times=8`, `headroom=1` → `games_per_sunday=ceil(126/21)=6`,
  `slots_needed=ceil(6/3)=2`, `6+1=7 ≤ 3*(8-1)=21` → `can_drop_last_slot=True`. **Oracle hand-computed.**
- *Broadmeadow drop=unsafe:* set `broadmeadow_sunday_games=441`, `playable_sundays=21` →
  `games_per_sunday=ceil(441/21)=21`; `21+1=22 > 21` → `False`. *(review fix — M6: added the explicit
  game count (441/21=21) so the executor doesn't have to reverse-engineer it.)*
- *Headroom boundary:* `broadmeadow_sunday_games=420`, `playable_sundays=21` →
  `games_per_sunday=20`; `headroom=1` → `20+1=21 ≤ 21` → `True`; same data with `headroom=2` →
  `22 > 21` → `False`. Proves the buffer is honoured exactly. *(review fix — M6: added the 420/21=20
  construction.)*
- *All-grade block reduces capacity:* add a `{date,time,field_location}` block on one central Sunday
  time → that Sunday's slot count drops by `fields` (=3 for NIHC); assert reflected in that Sunday's
  surviving-slot count.
- *Scoped block ignored:* add a `{club:'Gosford',time,field_location}` block → capacity unchanged.
- *PHL/2nd EF+WF note:* set PHL+2nd demand so `phl_2nd_per_sunday > efwf_capacity_per_sunday`
  (for NIHC `efwf_capacity_per_sunday = (fields-1)*times_per_sunday = 2*8 = 16`) → warning present.
  *(review fix — M7: oracle now references `fields_excl_SF * times_per_sunday`, not a hardcoded
  `2*times`.)*

### Unit B — `precheck` CLI subcommand + year wrapper + tests  *(depends_on: Unit A; executor: Sonnet — mechanical wiring)*

**Files touched**
- `analytics/capacity_precheck.py` — add `analyze_capacity_for_year(year, *, headroom=1)` wrapper:
  call `config.load_season_data(year)` for `teams`/`fields`/`num_rounds`, pull
  `day_time_map`/`start_date`/`end_date`/`field_unavailabilities`/`home_field_map`/`blocked_games`/
  `constraint_defaults`/`forced_games` from that same `data` dict, call the core. *(review fix — H1:
  must use `load_season_data`, not `load_season_config`, for Team objects.)*
- `run.py` — register the `precheck` subparser (mirror the `preseason`/`validate`/`diagnose`
  registration pattern; the subparsers block is ~L60–205) + add the dispatch branch (`elif
  args.command == 'precheck': run_precheck(args)` alongside the others ~L217–237) + `run_precheck(args)`
  handler (human report; `--json` write; `sys.exit(1)` on fatals, mirroring `run_preseason` at
  `run.py:1336`).
- `tests/test_capacity_precheck_cli.py` (new) — drive `run_precheck` for 2026; assert exit code,
  stdout sections, and `--json` file contents shape.

**Change summary** — Thin wrapper + CLI only; all logic lives in Unit A. Human report formats the
Broadmeadow verdict ("Need N slots/Sunday; 19:00 slot can be dropped (headroom M)" or "KEEP all 8")
and lists away-venue floors and any fatals/warnings.

**No-mock test outline** — *Given* the real 2026 config, *When* `run_precheck(year=2026)` runs,
*Then* exit code is 0/1 consistent with `fatals`, stdout contains the Broadmeadow verdict line and
an away-venue section, and `--json` round-trips to a dict with the DoD keys. Oracle: assert
structural invariants, not magic numbers.

### Unit C — auto-run before `generate` + docs  *(depends_on: Unit A; executor: Sonnet)*

**Files touched**
- `main_staged.py` — add the precheck call at the start of **both** solver entry points used by
  `run_generate`: `main_staged()` and `main_simple()` (~`main_staged.py:1413`). Both call
  `data = load_data(year)` near their top; insert the precheck immediately after that load and
  **before** variable generation / solve, using the already-loaded `data` (no second load). Print
  fatals/warnings; fatals print a prominent banner and **abort** unless `--force` is passed.
  *(review fix — H5/M1: original listed only `main_staged.py` generically and speculated about an
  existing hook. Verified: there is **no** pre-solve validation hook in `main_staged` —
  `validate_game_config` is invoked from inside `generate_X()` (`utils.py:3545`), not from the entry
  points — so Unit C adds a **new** call site in **both** `main_staged()` and `main_simple()`.)*
- `run.py` — add `--force` to the `generate` subparser (the gen subparser is ~L60–159; `--force` is
  **not** currently present — review fix C2) and thread it through `run_generate` → `main_staged()` /
  `main_simple()`.
- `tests/test_capacity_precheck_generate_gate.py` (new) — a synthetic infeasible config makes the
  generate entry abort before solving (no solver invoked); `--force` bypasses.
- Docs per registry.

**Change summary** — Wire the precheck as a pre-solve gate in both entry points; no solver behaviour
changes beyond the gate. Uses the already-loaded `data` dict (no double `build_season_data`).

**No-mock test outline** — *Given* a synthetic config whose away-venue floor fails, *When* the
generate entry point runs without `--force`, *Then* it raises/exits before any model is built (assert
no solver object created); *When* run with `--force`, *Then* it proceeds past the gate. Oracle:
the synthetic config has 1 field × 1 Sunday time at an away venue with 2 home teams → floor 1 < 2.

## Doc registry

- `CLAUDE.md` — add `precheck` to the "Quick Commands" + the Skills/CLI surface; add a row to the
  Key Functions table for `analyze_capacity` / `analyze_capacity_for_year`; note the new pre-solve
  gate and `--force` under the Solver Execution rules.
- `docs/ai/AI_OPERATIONS_MANUAL.md` — document the capacity precheck (inputs, output dict, the
  Broadmeadow drop-slot logic, away-venue floor).
- `docs/PERENNIAL_RULES.md` — cross-reference: the "is the 19:00/worst slot needed?" question is now
  answerable via `precheck` (ties to the existing 7pm-is-worst review rule).
- `docs/todo/00-dependency-tree.md` — add the spec-043 node (done at author time).

## Out of scope

- **No change to `utils._check_scheduling_feasibility` (Phase 20).** User decision: leave it. Overlap
  between the new module and Phase 20 is accepted, not refactored. (A deliberate boundary; if
  convergence is wanted later it is a *separate* filed spec, not implied here.)
- **No rework of `PreSeasonReport.get_slot_capacity_analysis`.** It stays as the coarse aggregate
  view it is. Folding the new diagnostics *into* the preseason report's output is a possible later
  enhancement — if wanted, a **separate filed spec**, not implied here.
- **No full per-grade time-window bin-packing.** v1 is flat capacity + EF/WF secondary note only.
- **No solver/objective changes.** This is pre-solve analysis only.
- **No new UI.** This spec delivers the *callable contract* a UI would consume; the UI itself is a
  separate system not in this repo.
- **Friday-night capacity correctness** beyond excluding Friday PHL from the Sunday count — the
  existing Phase-20 Friday checks remain the authority on Friday feasibility.

## Dependencies

- `depends_on: none` at the spec level (branches off current final-form).
- Internal: Unit A blocks Units B and C. **Units B and C both edit `run.py`** (B adds the `precheck`
  subparser/dispatch/handler; C adds `--force` to the `generate` subparser) → **serialise B before
  C** to avoid a collision. Recommended order: A → B → C.

## Risks & blast radius

- **`run.py` shared by B and C** — both edit it. Mitigation: serialise B→C; C confined to the
  generate subparser + the `--force` threading.
- **Auto-run gate could block a previously-working `generate`** if the precheck is over-strict.
  Mitigation: `--force` escape; fatals only on genuine capacity shortfalls (slots < demand), not on
  soft tightness (those are warnings).
- **No double data-load:** `validate_game_config` is called from inside `generate_X()`
  (`utils.py:3545`), **not** from `main_staged.py`; there is no existing pre-solve hook. Unit C
  inserts its precheck after the existing `load_data(year)` in each entry point and reuses that
  `data` — zero extra loads. *(review fix — H4: risk was stated vaguely; true call path verified.)*
- **Number drift vs Phase 20** — two capacity calculators can disagree and confuse the convenor.
  Mitigation: reuse `generate_timeslots` and the `(rounds+1)//2` convention; document that Phase 20
  remains the hard gate and `precheck` is the planning/advisory view.

## Open Questions

None — all design decisions were resolved with the user (see Scope decisions). `open_questions: 0`.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). -->
0. **Do NOT start without an explicit user instruction to implement this plan.** `ready` means
   "authorised when asked", not "build now". If you arrived here off authoring/review with no user
   go-ahead, STOP and ask.
1. Status must be `ready` (carries a `reviewed:` stamp). If `review_pending`/`under_review`, let
   review finish. If `blocked`, STOP — Open Questions need the user (there are none now).
2. Only after the user says to implement: stamp `building`, claim `owner`. Orchestrator = Opus.
3. Per unit: own worktree+branch off **final-form**; delegate B/C to Sonnet, keep A on Opus (subtle
   math); run the S2 gates (type-check, changed-file lint, AST sweep, no-mock tests ≥85% on new
   code, docs).
4. After each unit, launch `/adversarial` Mode B against the diff + this DoD. Route fixes, re-verify.
   Never merge unverified.
5. Merge → push origin → post-merge verify (`run.py precheck --year 2026` actually runs and prints
   the Broadmeadow verdict) → remove worktree. Tick the unit box.
6. When all units pass: stamp `done`; update `docs/todo/00-dependency-tree.md`.

## Units checklist
- [ ] Unit A — core `analyze_capacity` + tests
- [ ] Unit B — `precheck` CLI + year wrapper + tests
- [ ] Unit C — auto-run gate before `generate` + `--force` + docs
