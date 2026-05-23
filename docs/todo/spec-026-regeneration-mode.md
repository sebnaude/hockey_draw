<!-- status: ready -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-025 (LOCKED_PAIRINGS config + generate_X enforcement — regen writes pins into it). Builds on the already-shipped locked-weeks architecture (run.py:71-79, main_staged.py:1037-1059). The regen soft-constraint group (spec-027) is selected by this mode but is NOT a blocker: without spec-027 the regen still runs, just with the normal hard constraints (which is the "may be infeasible" case spec-027 fixes). Shares run.py + main_staged.py + analytics/storage.py with spec-021/022/023 — rebase before merge. -->
<!-- owner: session=none claimed=none -->
<!-- reviewed: adversarial Sonnet review 2026-05-23 — fixes applied inline -->

# spec-026 — Unified regeneration mode: freeze everything outside a scope, re-solve the rest

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
  already exists (`--locked` / `--lock-weeks`, `main_staged.py:1037-1059`).
- **Pinned** (future, kept-but-retimeable): each pairing pinned to its **date** with time/slot/
  field free, via `LOCKED_PAIRINGS` (spec-025).

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
  loads keys; `main_staged.py:1037-1059` does `model.Add(X[key]==1)` for matched keys (lines
  1037-1046), `model.Add(var==0)` for every other var in a locked week (lines 1049-1056), and
  stores `data['locked_keys_set']` at line 1059. `data['locked_weeks']` and
  `data['locked_keys_set']` carry the state; every atom skips locked weeks via `key[6] in
  locked_weeks`. spec-026 reuses this verbatim for the played-weeks freeze.
  (review fix — corrected body cite from :1037-1056 to :1037-1059; added per-sub-block
  breakdown to match the actual three-clause structure at lines 1037/1049/1059.)
- **Pinning is the soft-freeze half (spec-025).** `LOCKED_PAIRINGS` pins a pairing to its date,
  freeing time/slot/field. spec-026 *generates* these pins from the source draw. The config key
  is `data['locked_pairings']` (injected by `build_season_data`, per spec-025 DoD 2); the
  extractor must write into this same key.
  (review fix — added explicit config key name to avoid any ambiguity at the injection site.)
- **Extraction primitives already exist on `DrawStorage`.** `to_key()`
  (`analytics/storage.py:49-55`) yields the 11-tuple; `get_locked_games(weeks)`
  (`:209-218`), `get_remaining_games(weeks)` (`:220-229`), and `lock_and_split(weeks)`
  (`:231-266`) already partition a draw by week. spec-026 adds a grade-and-week aware extractor
  that turns a frozen `StoredGame` into a `{teams, grade, date}` LOCKED_PAIRINGS entry (drop
  time/slot/field — that's the whole point).
- **Grade filtering in generation is keyed on `key[2]`.** `generate_X` filters PHL/2nd/lower
  grades by grade name string (`utils.py:3636-3668`). Freeing a grade for regen means: do NOT
  pin its pairings (don't emit LOCKED_PAIRINGS for it) — its vars are generated and constrained
  normally. Freezing a grade means: emit a pin for every one of its games.
- **No existing `data['mode']` branch.** Confirmed by source scan: no `data['mode']` key exists
  in production code (only in spec docs). Regen is wired as data the existing solver consumes
  (locked_weeks + locked_pairings + a group selection), not a new solver path — keeping it
  inside the one-solver-run model (GOALS §1).
- **Group selection has no current infrastructure hook.** The `--groups` CLI flag and
  `resolve_groups()` mechanism belong to spec-023 (not yet landed). "Select the regen group" in
  DoD 3 means: once spec-023 ships, pass `--groups regen` (or call `resolve_groups(['regen'])`
  programmatically); until then, emit the documented warning and fall through to the default
  constraint set. spec-026 does NOT add a `--groups` flag itself — that is spec-023's scope.
  (review fix — clarified the fallback path is the only available path at ship time; calling
  `--groups regen` is conditional on spec-023 landing first.)
- **Game-count math reacts to roster changes automatically.** `max_games_per_grade`
  (`utils.py:355`) recomputes per-grade game counts from the (updated) team lists, and
  `EqualGamesAndBalanceMatchUps` enforces them — so after a 5th→6th move, regenerating those two
  grades with the new team lists yields the correct new counts. Frozen grades keep their counts
  because their pairings are pinned. (This is *why* regen scopes by grade cleanly: the changed
  grades are exactly the ones whose counts changed.)
  (review fix — added confirmed line number for `max_games_per_grade`.)
- **`main_simple` is a function inside `main_staged.py`** (line 1239), not a separate file.
  Unit C file references using `main_staged.py / main_simple` are correct (same file).
  (review fix — confirmed to prevent a builder from creating a phantom `main_simple.py`.)

## Definition of Done

1. **Extractor** `extract_locked_pairings(draw, *, freeze_grades, freeze_weeks,
   exclude_weeks=frozenset())` in `analytics/storage.py` (or a new
   `analytics/regen.py`): returns a `LOCKED_PAIRINGS` list (one `{teams:[t1,t2], grade, date,
   description}` per frozen game), covering every game whose grade ∈ `freeze_grades` AND week ∈
   `freeze_weeks`, EXCLUDING games in `exclude_weeks` (the hard-locked played weeks, which are
   pinned exactly by the locked-weeks path and must not be double-pinned). `team1`/`team2`
   preserve the alphabetical key order; `time`/`day_slot`/`field` are intentionally dropped.
   A bye/placeholder game (no real opponent) produces no pin. `freeze_grades=frozenset()` (empty)
   means no grade is frozen by grade; `freeze_weeks=frozenset()` (empty) means no week is frozen
   by week; caller must pass the full set of all draw weeks if they want all weeks frozen.
   (review fix — explicit empty-set semantics added to prevent misinterpretation of "omit".)
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
     concatenate; spec-025 enforces both);
   - if spec-027 has landed (i.e. `resolve_groups(['regen'])` succeeds without ImportError/
     KeyError), select the **regen constraint group** by calling `resolve_groups(['regen'])`
     and passing the result to `main_simple`/`main_staged` via the `groups` kwarg (spec-023
     mechanism). If spec-027 has not landed (or spec-023 groups are not available), fall back
     to the normal full constraint selection and print a clear warning: "WARNING: spec-027 regen
     group not available — using full hard constraints; regen may be infeasible if frozen-but-
     retimed games violate adjacency/spacing/co-location rules."
   (review fix — made the spec-023 dependency explicit; `resolve_groups(['regen'])` is the
   call site; fallback wording tightened so the builder knows exactly what to print.)
4. **Free-scope variable generation is correct:** for a freed grade, `generate_X` produces its
   full variable set with the (possibly updated) team list and no pins; for a frozen grade,
   every game is pinned to its date. A test asserts: after a 5th→6th roster move, a regen with
   `--regen-grades 5th 6th` generates vars for the new 5th/6th team lists and pins every
   PHL/2nd/3rd/4th game to its original date.
5. **Output versioning:** a regen solve calls `save_solver_output(…, is_major=False)` (the
   `is_major` kwarg already exists in `analytics/versioning.py:save_solver_output` with
   default `True`). This triggers the `save_modified_draw` path, bumping MINOR (vN.M →
   vN.M+1). The regen diff (grades/weeks freed, pin count, games changed vs source) is recorded
   in the `regen` metadata block (DoD 6) so the CHANGELOG entry is self-contained.
   **Convention note:** `CLAUDE.md` and `feedback_versioning` say "Solver runs bump MAJOR; hand
   edits bump MINOR." A regen IS a solver run — but it is a scoped modification of a published
   draw, not a fresh season-wide solve. The implementer MUST confirm with the convenor before
   shipping whether MINOR is the right call here (the spec's current position is MINOR). If the
   convenor says MAJOR, change `is_major=True` and remove the `save_modified_draw` path. This
   is an open implementation decision, not blocked, but must be resolved before Unit C merges.
   (review fix — surfaced the MAJOR/MINOR convention tension explicitly; made the `is_major`
   kwarg cite concrete; added a required pre-merge confirmation step.)
6. **Metadata:** draw metadata records a `regen` block: `source_draw`, `regen_grades`,
   `regen_weeks`, `frozen_pin_count`, `hard_locked_weeks`, and a `games_changed` count
   (games whose time/slot/field differ from the source). Written by `save_solver_output`.
7. **A regen leaves frozen pairings on their dates:** an end-to-end test takes a real 2026 draw,
   regens `--regen-grades 6th`, and asserts every non-6th game in the output has the same
   `(team1, team2, grade, date)` as the source (time/field may differ — pins constrain date
   only, not time or field), while 6th-grade pairings are re-decided (at least one 6th game
   should differ from source in time or pairing to confirm the solver ran freely).
   (review fix — added the "at least one 6th game differs" positive assertion so the test is
   not vacuously green if regen silently replicates the source draw.)
8. Full suite green; `run.py generate --regen-from … --regen-grades 6th --year 2026 --simple`
   runs to a feasible solution on the 2026 config (post-merge verification per /basic step 9).

## Implementation units

> `run.py`, `main_staged.py` (contains `main_simple` at line 1239 — same file),
> `analytics/storage.py` are shared across units; sequence on one worktree, commit per unit.

### Unit A — Extractor
- Files: `analytics/storage.py` (or new `analytics/regen.py`) — `extract_locked_pairings`.
- Depends on: spec-025 Unit A (the `LOCKED_PAIRINGS` grammar must exist to target).
- Test (`tests/test_regen_extract.py`, GWT, no mocks, hand oracle): GIVEN a 3-game `DrawStorage`
  (2 in grade `'4th'`, week 2, 1 in grade `'6th'`, week 2, all on date `'2026-04-05'`), WHEN
  `extract_locked_pairings(draw, freeze_grades={'4th'}, freeze_weeks={2}, exclude_weeks=frozenset())`
  THEN exactly 2 pins are returned; each pin has keys `teams`, `grade`, `date`, `description`
  and NO keys `time`, `day_slot`, `field_name`, `field_location`; both pins have
  `grade='4th'` and `date='2026-04-05'` (hand-listed). GIVEN `exclude_weeks={2}`, WHEN the same
  call is made THEN 0 pins are returned (the week-2 game is excluded). GIVEN a draw containing
  a game with `team2=''` (bye/placeholder), THEN that game produces no pin.
  (review fix — replaced `freeze_weeks=all` (invalid Python) with a concrete, hand-computed
  frozenset; added explicit key-whitelist assertion; added the bye test case required by DoD 1.)

### Unit B — CLI flags + scope resolution + validation
- Files: `run.py` (`--regen-from`, `--regen-grades`, `--regen-weeks`; the union free-scope
  resolver; the `--regen-weeks`∩`--lock-weeks` overlap FATAL).
- Depends on Unit A.
- Test (`tests/test_regen_cli.py`): the free/frozen partition matches the union rule on a
  synthetic draw (hand-computed sets for grade-only, week-only, and both); overlapping
  `--regen-weeks`/`--lock-weeks` → FATAL. Hand-compute the expected free/frozen sets:
  draw has grades `{PHL, 6th}` × weeks `{1,2,3}`. Case 1 (`--regen-grades 6th`): free =
  all 6th games (weeks 1-3), frozen = all PHL games (weeks 1-3). Case 2 (`--regen-weeks 3`):
  free = all week-3 games (both grades), frozen = weeks 1-2 of both grades. Case 3 (both):
  free = 6th all weeks + PHL week 3, frozen = PHL weeks 1-2. Assert each case matches exactly.
  (review fix — added hand-computed oracle so the test is not implementation-defined.)

### Unit C — Orchestration wiring + group selection + metadata + versioning
- Files: `run.py::run_generate` (new `--regen-from`/`--regen-grades`/`--regen-weeks` wiring;
  `data['locked_pairings']` population; spec-023 group selection guard),
  `main_staged.py` (both `main_staged` and `main_simple` at line 1239 — feed
  `data['locked_pairings']` as concatenation of config + extracted pins),
  `analytics/versioning.py` (`save_solver_output` called with `is_major=False`; `regen`
  metadata block written by `_build_draw_metadata`) /
  `analytics/storage.py` (`games_changed` diff helper if not already present).
- Depends on Units A+B.
- MUST resolve the MAJOR/MINOR convention with the convenor before merge (see DoD 5).
- Test (`tests/test_regen_end_to_end.py`, no mocks): the DoD-7 frozen-pairings-keep-their-dates
  assertion on a real 2026 draw; metadata `regen` block populated with all 6 required keys
  (`source_draw`, `regen_grades`, `regen_weeks`, `frozen_pin_count`, `hard_locked_weeks`,
  `games_changed`); version bumped MINOR (confirm `vN.M` → `vN.{M+1}`).
  (review fix — made the 6 required metadata keys explicit in the test assertion; added
  MAJOR/MINOR resolution gate; listed all three affected files explicitly.)

## Doc registry

- `CLAUDE.md` — new "Regeneration" subsection under the hand-edit/Quick-Commands area: the
  `--regen-from`/`--regen-grades`/`--regen-weeks` flags, the union free-scope rule, the
  hard-lock-vs-pin distinction, and that frozen grades keep date/pairing but may re-time.
- `docs/operator-human/RULES.md` (or the operator runbook) — a convenor-facing "how to
  regenerate part of a draw" recipe (roster change → `--regen-grades`; future re-time →
  `--regen-weeks`).
- `docs/operator-ai/AI_OPERATIONS_MANUAL.md` — the regen orchestration flow (load → hard-lock →
  extract pins → select group → solve → MINOR-version).
- `docs/system/STAGES.md` — note that regen selects the `regen` group (cross-ref spec-027).
- `docs/todo/GOALS.md` — add the spec-026 row + summary; reflect regen as a shipped success
  criterion (GOALS §4 item 5 "apply a convenor edit … all in one script" extends to scoped
  regen).
- `docs/todo/00-dependency-tree.md` — register spec-026 (depends spec-025; unblocks spec-027's
  consumer side).

## Risks & blast radius

- **Frozen-but-retimed games can make the model infeasible under the normal hard constraints**
  (e.g. PHL/2nd adjacency, club-day co-location, spacing assume the solver controls the time;
  pinning the date but freeing the time can still leave no feasible time given the other frozen
  games). This is the central risk and is **the entire reason spec-027 exists** — it softens
  those constraints for regen. spec-026 ships usable without spec-027 only when the regen scope
  is small enough to stay feasible; the warning in DoD 3 surfaces this honestly.
- **Group selection requires spec-023.** Until spec-023 lands, `resolve_groups(['regen'])` does
  not exist, and the fallback (full hard constraints + warning) is the only available path. This
  means DoD 8 ("runs to a feasible solution") depends on the regen scope being small enough to
  satisfy the full hard constraint set without spec-027's softening. The DoD 8 test (`--regen-
  grades 6th`) is expected to be feasible even under full hard constraints because 6th grade
  does not participate in PHL/2nd adjacency rules.
  (review fix — added this risk explicitly; it was implicit but not called out.)
- **Double-pinning** a played week (both hard-locked and extracted as a pin) would create
  `X[key]==1` and a redundant `sum==1` — harmless but wasteful. Prevented by `exclude_weeks` in
  the extractor (DoD 1) and tested (Unit A).
- **Stale source keys** (a source draw with `round_no`/`day_slot` that no longer match the
  current calendar) — the pin only uses `(teams, grade, date)`, so ancillary-field staleness is
  irrelevant to pinning (unlike hard-lock, which has `--repair-locked` for exactly this). Note
  this as a deliberate robustness property.
- **`data['locked_pairings']` concatenation order.** When both hand-authored config pins and
  auto-extracted regen pins exist, both are fed to spec-025's `generate_X` pass. If a pair
  appears in BOTH sources (e.g. a hand-authored pin for a game that is also in the frozen set),
  both `sum==1` constraints apply to the same scope — this is harmless (idempotent if the
  scopes are identical) but the implementer should document the dedup behaviour or explicitly
  dedup by `(teams, grade, date)` before passing. Flag if dedup is missing.
  (review fix — added this new risk; the spec was silent about it.)

## Out of scope

- **The regen soft-constraint group itself** — spec-027. spec-026 *selects* it but does not
  define the soft atoms or the core-hard set.
- **The `--groups` CLI flag and `resolve_groups()` infrastructure** — spec-023. spec-026 calls
  the spec-023 API (guarded by availability) but does not implement it.
  (review fix — made the spec-023 dependency explicit in Out of scope to prevent scope creep.)
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
