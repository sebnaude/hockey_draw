<!-- status: ready -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-024 (LOCKED_PAIRINGS config + generate_X enforcement — regen writes pins into it). Builds on the already-shipped locked-weeks architecture (run.py:71-79, main_staged.py:1037-1059). The regen soft-constraint group (spec-026) is selected by this mode but is NOT a blocker: without spec-026 the regen still runs, just with the normal hard constraints (which is the "may be infeasible" case spec-026 fixes). Shares run.py + main_staged.py + analytics/storage.py with spec-021/022/023 — rebase before merge. -->
<!-- owner: session=none claimed=none -->

# spec-025 — Unified regeneration mode: freeze everything outside a scope, re-solve the rest

## Why

The convenor needs to modify a published draw without re-rolling the whole season. Two
real situations, one mechanism:

- **(a) Roster change.** A team drops from 5th to 6th grade mid-pre-season. The 5th and 6th
  grade fixtures must be regenerated; PHL/2nd/3rd/4th keep their exact pairings and weekends.
- **(b) Mid-season re-time.** Weeks 1-9 have been played; weeks 10-22 need re-timing (a venue
  became available, a clash emerged) but the *pairings and weekends* for those future weeks
  should stay put — only the time-on-the-day may change.

Both are the same operation: **freeze everything except a chosen regeneration scope, then
re-solve.** The scope is expressed along two axes — **grades** and **weeks** — and "frozen"
means one of two strengths:

- **Hard-locked** (already-played past weeks): exact game keys pinned to 1, nothing moves. This
  already exists (`--locked` / `--lock-weeks`, `main_staged.py:1037-1056`).
- **Pinned** (future, kept-but-retimeable): each pairing pinned to its **date** with time/slot/
  field free, via `LOCKED_PAIRINGS` (spec-024).

The regeneration scope itself (e.g. grades 5th+6th, or weeks 10-22) is left **fully free**: its
variables are generated fresh and the solver re-decides pairings, weekends, and times.

Without this, the convenor's only options are hand-editing `current.json` game-by-game (slow,
error-prone) or a full re-solve (throws away every other grade's settled schedule). The cost of
not building it is exactly the pain that motivated the `final-form` rework: changes to a
published draw are first-class (GOALS §1), and right now they aren't supported beyond a single
swap.

### Research findings (verified against final-form)

- **Locked-weeks is the hard-freeze half, already complete.** `run.py:71-79` parses `--locked`
  (source draw/checkpoint/pkl) and `--lock-weeks 1,2,3`; `run.py:245-297` (`_load_locked_keys`)
  loads keys; `main_staged.py:1037-1056` does `model.Add(X[key]==1)` for matched keys and
  `model.Add(var==0)` for every other var in a locked week. `data['locked_weeks']` and
  `data['locked_keys_set']` carry the state; every atom skips locked weeks via `key[6] in
  locked_weeks`. spec-025 reuses this verbatim for the played-weeks freeze.
- **Pinning is the soft-freeze half (spec-024).** `LOCKED_PAIRINGS` pins a pairing to its date,
  freeing time/slot/field. spec-025 *generates* these pins from the source draw.
- **Extraction primitives already exist on `DrawStorage`.** `to_key()`
  (`analytics/storage.py:49-55`) yields the 11-tuple; `get_locked_games(weeks)`
  (`:209-218`), `get_remaining_games(weeks)` (`:220-229`), and `lock_and_split(weeks)`
  (`:231-266`) already partition a draw by week. spec-025 adds a grade-and-week aware extractor
  that turns a frozen `StoredGame` into a `{teams, grade, date}` LOCKED_PAIRINGS entry (drop
  time/slot/field — that's the whole point).
- **Grade filtering in generation is keyed on `key[2]`.** `generate_X` filters PHL/2nd/lower
  grades by grade name (`utils.py:3636-3668`). Freeing a grade for regen means: do NOT pin its
  pairings (don't emit LOCKED_PAIRINGS for it) — its vars are generated and constrained
  normally. Freezing a grade means: emit a pin for every one of its games.
- **No existing `data['mode']` branch.** Regen is wired as data the existing solver consumes
  (locked_weeks + locked_pairings + a group selection), not a new solver path — keeping it
  inside the one-solver-run model (GOALS §1).
- **Game-count math reacts to roster changes automatically.** `max_games_per_grade`
  (`utils.py`) recomputes per-grade game counts from the (updated) team lists, and
  `EqualGamesAndBalanceMatchUps` enforces them — so after a 5th→6th move, regenerating those two
  grades with the new team lists yields the correct new counts. Frozen grades keep their counts
  because their pairings are pinned. (This is *why* regen scopes by grade cleanly: the changed
  grades are exactly the ones whose counts changed.)

## Definition of Done

1. **Extractor** `extract_locked_pairings(draw, *, freeze_grades, freeze_weeks,
   exclude_weeks=frozenset())` in `analytics/storage.py` (or a new
   `analytics/regen.py`): returns a `LOCKED_PAIRINGS` list (one `{teams:[t1,t2], grade, date,
   description}` per frozen game), covering every game whose grade ∈ `freeze_grades` AND week ∈
   `freeze_weeks`, EXCLUDING games in `exclude_weeks` (the hard-locked played weeks, which are
   pinned exactly by the locked-weeks path and must not be double-pinned). `team1`/`team2`
   preserve the alphabetical key order; `time`/`day_slot`/`field` are intentionally dropped.
   A bye/placeholder game (no real opponent) produces no pin.
2. **CLI** on `run.py generate`:
   - `--regen-from SOURCE` — the source draw JSON (e.g. `draws/2026/current.json`) to freeze
     from. Required for regen mode; its presence *is* what enables regen.
   - `--regen-grades G [G ...]` — grades to REGENERATE (free). All other grades are frozen
     (pinned). Omitted → no grade is frozen-by-grade (all grades free along the grade axis).
   - `--regen-weeks SPEC` — weeks to REGENERATE (free), as a range/list (`10-22`, `10,12,14`).
     All other future weeks are frozen (pinned). Omitted → all weeks free along the week axis.
   - Interaction with the existing `--lock-weeks` (hard-locked played weeks): a week that is
     `--lock-weeks` (played) is hard-locked and never pinned; a future week not in
     `--regen-weeks` is pinned; a week in `--regen-weeks` is free. Spell out the precedence in
     `--help` and validate that `--regen-weeks` and `--lock-weeks` do not overlap (FATAL if they
     do — a week can't be both replayed and re-solved).
   - **A game is FREE iff its grade ∈ regen-grades OR its week ∈ regen-weeks** (union — either
     axis frees it). Everything else is frozen. Document this union semantics explicitly; it is
     the one non-obvious rule. (Rationale: regenerating 5th grade means *all weeks* of 5th grade
     move; regenerating weeks 10-22 means *all grades* in those weeks move; doing both frees the
     union. Convenor confirmed "one unified mode scoped by grade and/or week.")
3. **Orchestration** (`run.py` `run_generate` → `main_staged`/`main_simple`): when
   `--regen-from` is given,
   - load the source draw;
   - hard-lock `--lock-weeks` via the existing path (unchanged);
   - compute the frozen set = all games NOT free (per DoD 2) and NOT in a hard-locked week;
   - call `extract_locked_pairings` over the frozen set and feed the result as
     `data['locked_pairings']` (in addition to any hand-authored `LOCKED_PAIRINGS` from config —
     concatenate; spec-024 enforces both);
   - select the **regen constraint group** (spec-026) when available; if spec-026 has not landed,
     fall back to the normal full constraint selection and print a clear warning that the solve
     may be infeasible because frozen-but-retimed games can violate hard adjacency/co-location/
     spacing (this is the documented gap spec-026 closes).
4. **Free-scope variable generation is correct:** for a freed grade, `generate_X` produces its
   full variable set with the (possibly updated) team list and no pins; for a frozen grade,
   every game is pinned to its date. A test asserts: after a 5th→6th roster move, a regen with
   `--regen-grades 5th 6th` generates vars for the new 5th/6th team lists and pins every
   PHL/2nd/3rd/4th game to its original date.
5. **Output is a MINOR version bump, not MAJOR.** Per `feedback_versioning`, a regen that keeps
   most of a published draw is a modification, not a fresh solve — `save_solver_output` /
   `DrawVersionManager` records it as a MINOR bump (vN.M → vN.M+1) with a regen diff
   (which grades/weeks were freed, pin count, games changed vs source). Confirm this is the
   intended versioning with the convenor in the plan's Open Questions if ambiguous — RESOLVED:
   regen is a modification → MINOR (per `feedback_versioning` and CLAUDE.md "Solver runs bump
   MAJOR; hand edits bump MINOR"; a scoped regen is closer to a hand edit than a fresh solve).
6. **Metadata:** draw metadata records a `regen` block: `source_draw`, `regen_grades`,
   `regen_weeks`, `frozen_pin_count`, `hard_locked_weeks`, and a `games_changed` count
   (games whose time/slot/field differ from the source). Written by `save_solver_output`.
7. **A regen leaves frozen pairings on their dates:** an end-to-end test takes a real 2026 draw,
   regens `--regen-grades 6th`, and asserts every non-6th game in the output has the same
   `(team1, team2, grade, date)` as the source (time/field may differ), while 6th-grade pairings
   are re-decided.
8. Full suite green; `run.py generate --regen-from … --regen-grades 6th --year 2026 --simple`
   runs to a feasible solution on the 2026 config (post-merge verification per /basic step 9).

## Implementation units

> `run.py`, `main_staged.py`, `analytics/storage.py` are shared across units; sequence on one
> worktree, commit per unit.

### Unit A — Extractor
- Files: `analytics/storage.py` (or new `analytics/regen.py`) — `extract_locked_pairings`.
- Depends on: spec-024 Unit A (the `LOCKED_PAIRINGS` grammar must exist to target).
- Test (`tests/test_regen_extract.py`, GWT, no mocks, hand oracle): GIVEN a 3-game `DrawStorage`
  (2 in grade 4th, 1 in grade 6th, known dates), WHEN
  `extract_locked_pairings(draw, freeze_grades={'4th'}, freeze_weeks=all)` THEN exactly the two
  4th-grade pins are returned with their dates and no time/field keys (hand-list them). GIVEN
  `exclude_weeks={1}` and one 4th game in week 1, THEN that game yields no pin.

### Unit B — CLI flags + scope resolution + validation
- Files: `run.py` (`--regen-from`, `--regen-grades`, `--regen-weeks`; the union free-scope
  resolver; the `--regen-weeks`∩`--lock-weeks` overlap FATAL).
- Depends on Unit A.
- Test (`tests/test_regen_cli.py`): the free/frozen partition matches the union rule on a
  synthetic draw (hand-computed sets for grade-only, week-only, and both); overlapping
  `--regen-weeks`/`--lock-weeks` → FATAL.

### Unit C — Orchestration wiring + group selection + metadata + versioning
- Files: `run.py::run_generate`, `main_staged.py` / `main_simple` (feed `data['locked_pairings']`,
  select regen group when present, concatenate config + extracted pins), `analytics/versioning.py`
  / `analytics/storage.py` (MINOR bump + `regen` metadata block + `games_changed` diff).
- Depends on Units A+B.
- Test (`tests/test_regen_end_to_end.py`, no mocks): the DoD-7 frozen-pairings-keep-their-dates
  assertion on a real 2026 draw; metadata `regen` block populated; version bumped MINOR.

## Doc registry

- `CLAUDE.md` — new "Regeneration" subsection under the hand-edit/Quick-Commands area: the
  `--regen-from`/`--regen-grades`/`--regen-weeks` flags, the union free-scope rule, the
  hard-lock-vs-pin distinction, and that frozen grades keep date/pairing but may re-time.
- `docs/operator-human/RULES.md` (or the operator runbook) — a convenor-facing "how to
  regenerate part of a draw" recipe (roster change → `--regen-grades`; future re-time →
  `--regen-weeks`).
- `docs/operator-ai/AI_OPERATIONS_MANUAL.md` — the regen orchestration flow (load → hard-lock →
  extract pins → select group → solve → MINOR-version).
- `docs/system/STAGES.md` — note that regen selects the `regen` group (cross-ref spec-026).
- `docs/todo/GOALS.md` — add the spec-025 row + summary; reflect regen as a shipped success
  criterion (GOALS §4 item 5 "apply a convenor edit … all in one script" extends to scoped
  regen).
- `docs/todo/00-dependency-tree.md` — register spec-025 (depends spec-024; unblocks spec-026's
  consumer side).

## Risks & blast radius

- **Frozen-but-retimed games can make the model infeasible under the normal hard constraints**
  (e.g. PHL/2nd adjacency, club-day co-location, spacing assume the solver controls the time;
  pinning the date but freeing the time can still leave no feasible time given the other frozen
  games). This is the central risk and is **the entire reason spec-026 exists** — it softens
  those constraints for regen. spec-025 ships usable without spec-026 only when the regen scope
  is small enough to stay feasible; the warning in DoD 3 surfaces this honestly.
- **Double-pinning** a played week (both hard-locked and extracted as a pin) would create
  `X[key]==1` and a redundant `sum==1` — harmless but wasteful. Prevented by `exclude_weeks` in
  the extractor (DoD 1) and tested (Unit A).
- **Stale source keys** (a source draw with `round_no`/`day_slot` that no longer match the
  current calendar) — the pin only uses `(teams, grade, date)`, so ancillary-field staleness is
  irrelevant to pinning (unlike hard-lock, which has `--repair-locked` for exactly this). Note
  this as a deliberate robustness property.

## Out of scope

- **The regen soft-constraint group itself** — spec-026. spec-025 *selects* it but does not
  define the soft atoms or the core-hard set.
- **A GUI / interactive scope picker** — CLI flags only.
- **Regenerating a sub-set of a single grade** (e.g. only one club's 6th-grade games) — the
  scope axes are grade and week; finer slicing is not requested. If needed later, file a new
  spec.
- **Auto-detecting which grades changed from a config diff** — the convenor names them via
  `--regen-grades`; auto-detection is a separate convenience, not this spec.
- **`scripts/test_scenario.py` integration** (BEFORE/AFTER violation diff for a regen) — the
  existing harness already diffs arbitrary draws; wiring a regen-specific preview is a follow-on
  only if the convenor asks. The metadata `games_changed` diff (DoD 6) covers the audit need
  here.
