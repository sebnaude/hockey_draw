<!-- status: ready -->
<!-- reviewed: adversarial Sonnet review 2026-05-28 — fixes applied inline -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->

# spec-037 — `AwayClubHomeWeekendsCount` redesign: derived two-sided range, no forced-Friday awareness

**Spec source:** convenor + spec-035 handoff investigation (this session). Surfaced as one of the
two structurally-broken atoms in `docs/todo/spec-035-e2e-infeasibility-handoff.md` (§6). Supersedes
the WIP `spec035-flense` branch (commit `b6b7186`).

## Why

`AwayClubHomeWeekendsCount` is the first of two atoms whose presence makes the `core` constraint set
infeasible during CP-SAT's initial constraint copy on the forced-free `season_test` config (handoff
§3, §5, §6). Current behaviour pins three hard equalities per away-based club:

1. `sum(friday_home_indicators) == phl_forced_friday_count(data, club)`
2. `sum(sunday_home_indicators) == max(PHL_required − forced_fridays, max_other_grade)`
3. `sum(all_home_indicators)    == max(PHL_required, max_other_grade)`

Three problems, all confirmed by code reading + the forced-free probe:

- **(1) is redundant.** `FORCED_GAMES` already enforces "exactly N PHL Fridays at venue X" via
  partial-key `{count: N, constraint: 'equal'}` entries (verified in spec-035 §8). Two solver
  mechanisms encoding the same fact is a forward-only smell — one of them is wrong.
- **(2)+(3) over-constrain.** On forced-free `season_test` they reduce to `sundays_home == 20` and
  `total_home == 20` for both Maitland and Gosford. Combined with `AwayClubPerOpponentAndAggregateHomeBalance`,
  the model goes infeasible (handoff §6 — flense alone made the atom feasible standalone but the
  pair stayed infeasible: removing forced-Friday code didn't fix it, so the issue isn't there).
- **The atom reads from `FORCED_GAMES` config.** Per convenor instruction
  (`memory/feedback_forcing_belongs_in_config.md`): forcing belongs in config, not baked into
  constraint atoms. The atom should be derivable purely from the schedule structure.

The redesign: replace the three hard equalities with **one two-sided range bound on Sunday-home
weekends**, derived from per-grade home-game counts (`num_rounds[g] // 2`). The bound's floor is
the maximum home-game demand across non-PHL grades (those games MUST be on Sunday — no Friday
alternative). The bound's ceiling is the maximum home-game demand across ALL grades incl. PHL (no
team plays more than its 50/50 share). Forced Fridays then fit between the bounds automatically:
every PHL home game pulled to Friday by `FORCED_GAMES` reduces the PHL Sunday demand, which can
only LOWER the effective Sunday total — and the floor protects the non-PHL grades from being
squeezed. No atom-internal FORCED-aware math required.

### Worked example — Maitland on `season_test` (forced-free)

`num_rounds`: PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18. Maitland fields PHL/3rd/4th/5th/6th (no 2nd).

| Grade | Home games (`num_rounds // 2`) | Must be Sunday? |
|---|---|---|
| PHL | 10 | No (PHL has a Friday option) |
| 3rd | 9 | Yes |
| 4th | 9 | Yes |
| 5th | 8 | Yes |
| 6th | 9 | Yes |

- `min_sundays_home = max(9, 9, 8, 9) = 9` (max across non-PHL grades — the floor)
- `max_sundays_home = max(10, 9, 9, 8, 9) = 10` (max across all grades incl. PHL — the ceiling)

The atom enforces only: `9 ≤ sum(sunday_home_indicators) ≤ 10`. With 0 forced Fridays, PHL's 10
home games land on 10 Sundays → total = 10. With 3 forced Fridays, PHL contributes 7 home Sundays
+ 3 home Fridays → total Sundays = max(9, 7) = 9. Both inside `[9, 10]`. ✓

### Worked example — another-grade-dominant (sanity)

If a hypothetical setup gives PHL=18, 3rd=20 (3rd plays MORE than PHL):

- `min = max(10, ...) = 10` (3rd drives)
- `max = max(9, 10, ...) = 10` (3rd still drives — > PHL's 9)

`10 ≤ sundays_home ≤ 10` → exact equality, falls out naturally. No special-case code.

## Definition of Done

1. **`AwayClubHomeWeekendsCount.apply` (file `constraints/atoms/away_club_home_weekends_count.py`)**
   enforces only `min_sundays_home(data, club) ≤ sum(sunday_home_indicators_for_club) ≤ max_sundays_home(data, club)`.
   No Friday-indicator sum constraint, no "all-home indicator" sum constraint, no call into
   `phl_forced_friday_count` / `away_club_required_sundays` / `away_club_total_weekends`.
2. **New helpers in `constraints/atoms/_phl_forced_friday_helper.py`:**
   - `away_club_min_sundays_home(data, club) -> int` returning `max(num_rounds[g] // 2 for g in non-PHL grades the club fields)`, `0` if the club fields no non-PHL grades.
   - `away_club_max_sundays_home(data, club) -> int` returning `max(num_rounds[g] // 2 for g in ALL grades the club fields)`, `0` if the club has no teams.
   - Both functions use `_grade_required_games(data, g)` (already exists in the helper module) for the per-grade game count, divided by 2 with floor for floor and ceil for ceiling — but since `num_rounds[g]` is always even in production configs (paired home/away), integer division is exact. Where it might not be (odd `num_rounds`), document the rounding choice: floor for the lower bound, ceil for the upper.
   - Gosford case: `away_club_min_sundays_home('Gosford') = 0` (no non-PHL grades), `away_club_max_sundays_home('Gosford') = 10`. The atom must handle `min=0` without emitting a vacuous `sum >= 0` constraint — the `Add(sum >= min)` is simply skipped when `min == 0` (it's always satisfied). The upper bound `sum <= max` is still added. (review note — Low: Gosford forces at least 8 PHL Fridays via FORCED_GAMES, meaning at most 2 PHL home Sundays; the lower bound of 0 is correct and non-constraining here; the upper of 10 correctly allows any split.)
3. **Forward-only deletion** of orphaned helpers from `_phl_forced_friday_helper.py`:
   `phl_forced_friday_count`, `away_club_required_sundays`, `away_club_total_weekends`,
   `_entry_targets_club_phl_friday`, `_iter_candidate_friday_phl_keys`
   are all REMOVED. The module's `__all__` is updated to `['phl_forced_friday_meetings']` plus
   the two new bounds functions.
   `phl_forced_friday_meetings` STAYS (consumed by `_club_vs_club_stacked_shared.py`, which is
   used by `ClubVsClubStackedWeekends` — spec-038 will redesign that atom but still needs the
   per-pair Friday count for PHL).
   `_friday_phl_forced_entries` STAYS — it is a private helper called by `phl_forced_friday_meetings`
   at line 327 of `_phl_forced_friday_helper.py`, so it is NOT orphaned once
   `phl_forced_friday_count` is removed. Likewise `_matched_var_keys_for_entry`, `_entry_count`,
   `_iter_candidate_friday_phl_keys_for_pair`, and `_entry_targets_pair_phl_friday` all stay
   (they are consumed by `phl_forced_friday_meetings`).
   (review fix — C1: plan incorrectly listed `_friday_phl_forced_entries` for deletion; it is
   called by `phl_forced_friday_meetings` (verified: `_phl_forced_friday_helper.py:327`), which
   STAYS — deleting it would break the stacked-weekends atom and spec-038.)
4. **`AwayClubHomeWeekendsCountRegenSoft`** (the regen-soft twin at
   `constraints/atoms/away_club_home_weekends_count_regen_soft.py`) gets the same redesign: a
   single deviation penalty `max(0, min_sundays - sum) + max(0, sum - max_sundays)` against the
   Sunday-home indicator total. No Friday penalty, no total-weekend penalty. The penalty weight
   key (`regen_away_club_home_weekends_count`, default 90000) is reused.
5. **Helper module docstring** updated: the per-club forced-Friday concept is gone; only the
   per-pair (`phl_forced_friday_meetings`) variant remains. Replace the long "Clarification
   (added 2026-05-18)" section with a short note explaining the new model.
6. **Tests rewritten** with hand-computed oracles, no mocks:
   - `tests/atoms/test_away_club_home_weekends_count.py` — six scenarios:
     - GIVEN Maitland fields PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18, WHEN computing bounds, THEN min=9, max=10.
     - GIVEN Gosford fields only PHL=20, WHEN computing bounds, THEN min=0 (no non-PHL grades), max=10.
     - GIVEN a club fields a 3rd=22 grade (dominant), PHL=18, WHEN computing bounds, THEN min=11, max=11.
     - GIVEN a CP-SAT model with the atom applied to a Maitland-shaped fixture, WHEN the solver finds any feasible assignment, THEN 9 ≤ count(sunday home indicators true) ≤ 10.
     - GIVEN the same fixture with 3 PHL Friday games forced via `FORCED_GAMES` (`{grade: 'PHL', day: 'Friday', field_location: 'Maitland Park', count: 3, constraint: 'equal'}`), WHEN the solver runs, THEN the model is feasible AND the Sunday-home indicator count is still in [9, 10] (specifically: 9, since PHL contributes 7 home Sundays and non-PHL drives the floor).
     - GIVEN an all-PHL-forced-to-Friday extreme (10 PHL home games all on Friday), WHEN the solver runs, THEN the model is feasible AND Sunday-home count = max(9, 0) = 9.
   - `tests/atoms/test_away_club_home_weekends_count_regen_soft.py` — three scenarios using a
     ONE-club fixture (n_penalties = 1 Sunday deviation var → normalized_weight = 90000/1 = 90000):
     - GIVEN a draw with Sunday-home count = 10 (inside bounds), WHEN applying the regen-soft atom, THEN penalty = 0.
     - GIVEN a draw with Sunday-home count = 7 (below floor of 9), WHEN applying, THEN deviation = 2, penalty contribution = 2 × normalized_weight = 2 × 90000 = 180000.
     - GIVEN a draw with Sunday-home count = 12 (above ceiling of 10), WHEN applying, THEN deviation = 2, penalty contribution = 2 × 90000 = 180000.
     (review fix — M3: penalty formula depends on n_penalties via normalization in `_build_normalized_penalty`; oracle is correct only when the fixture has exactly 1 club → 1 deviation var → normalized_weight equals the raw weight.)
   - `tests/atoms/test_phl_forced_friday_helper.py` — drop all tests of removed functions; KEEP the `phl_forced_friday_meetings` tests; ADD tests for the two new bounds functions matching the oracle table above.
7. **Bisect-harness probes pass** after the redesign lands. From `draw-final-form` worktree:
   ```
   .venv\Scripts\python.exe scripts\bisect_core_feasibility.py --max-time 120 --workers 10 \
       --exclude <every-core-atom-except-AwayClubHomeWeekendsCount-and-fundamentals>
   ```
   Note: `bisect_core_feasibility.py` ALWAYS excludes `ClubGameSpread` in addition to any
   `--exclude` arguments (hardcoded in `probe()`, `scripts/bisect_core_feasibility.py:66`).
   The harness uses `--groups core` so "fundamentals" = core group minus `ClubGameSpread` minus
   whatever is listed in `--exclude`. The `--exclude` list must contain every other hard core
   atom (e.g. `NoDoubleBookingTeams NoDoubleBookingFields EqualGamesAndBalanceMatchUps
   EqualMatchUpSpacing AwayClubPerOpponentAndAggregateHomeBalance ClubVsClubStackedWeekends
   ClubVsClubStackedCoLocation ...`) to isolate `AwayClubHomeWeekendsCount` alone.
   (review fix — M2: harness silently adds ClubGameSpread; implementer needs to know to pass only
   the remaining core atoms to --exclude, not ClubGameSpread again.)
   - `AwayClubHomeWeekendsCount` ALONE on fundamentals → ✅ `REACHED_SEARCH` (was: ❌ INFEASIBLE pre-flense; ✅ FEASIBLE post-flense, expected to remain ✅ here).
   - `AwayClubHomeWeekendsCount + AwayClubPerOpponentAndAggregateHomeBalance` on fundamentals → ✅ `REACHED_SEARCH` (was: ❌ INFEASIBLE post-flense — the pair that broke under the flense. THIS is the canonical acceptance probe).
8. **Forward-only cleanup of the WIP flense.** The `spec035-flense` local branch and its worktree
   (`C:/Users/c3205/Documents/Code/python/draw-s035-flense`) are torn down — the redesign
   supersedes them. (Remote was never pushed, per the handoff §2; "never delete remote branches"
   rule trivially holds.)
9. **Doc updates:**
   - `docs/system/COUNT_ADJUSTERS.md` — remove the `phl_forced_friday_count` / `away_club_required_sundays` / `away_club_total_weekends` sections; add a short "AwayClub home-Sunday derived range" section describing the new floor/ceiling formula with the worked example above.
   - `docs/system/CONSTRAINT_INVENTORY.md` — update the `AwayClubHomeWeekendsCount` row's description to reflect the new behaviour (one Sunday range bound, no Friday/total constraints).
   - `docs/todo/00-dependency-tree.md` — add spec-037 as a live entry; note it as ready-to-start (no deps).
10. **Registry untouched** — `AwayClubHomeWeekendsCount` keeps its `canonical_name`, `solver_class_names`, `severity_level=1`, `groups={...}`. Only internal semantics change. Registry count unchanged (49).
11. **Atom raises feasibility-violating ValueError** with a clear message if `min_sundays_home > max_sundays_home` (which should be mathematically impossible if `num_rounds` is consistent — but a config sanity error or a future grade reshape could trip it). Replaces the three current `ValueError`s with one.

## Implementation units

### Unit A — Atom redesign + helper cleanup + regen-soft parity + tests + docs (single coherent unit)

- **Files touched:**
  - `constraints/atoms/away_club_home_weekends_count.py` — apply method rewritten; imports trimmed.
  - `constraints/atoms/away_club_home_weekends_count_regen_soft.py` — apply method rewritten in parallel.
  - `constraints/atoms/_phl_forced_friday_helper.py` — remove 3 public functions (`phl_forced_friday_count`, `away_club_required_sundays`, `away_club_total_weekends`) + 2 private helpers (`_entry_targets_club_phl_friday`, `_iter_candidate_friday_phl_keys`); keep `_friday_phl_forced_entries` (used by `phl_forced_friday_meetings`); add 2 new public functions; update `__all__` and module docstring. (review fix — C1: `_friday_phl_forced_entries` is not orphaned.)
  - `tests/atoms/test_away_club_home_weekends_count.py` — full rewrite per DoD #6.
  - `tests/atoms/test_away_club_home_weekends_count_regen_soft.py` — full rewrite per DoD #6.
  - `tests/atoms/test_phl_forced_friday_helper.py` — drop dead-function tests; add new-function tests; keep `phl_forced_friday_meetings` tests as-is.
  - `docs/system/COUNT_ADJUSTERS.md` — section rewrite per DoD #9.
  - `docs/system/CONSTRAINT_INVENTORY.md` — row update per DoD #9.
  - `docs/todo/00-dependency-tree.md` — live entry update.
- **Why one unit, not split:** The atom rewrite, regen-soft twin, helper-module signature change, and tests all touch a tightly-coupled API surface — splitting into "atom" / "regen-soft" / "tests" sub-units would cause merge contention on the helper module and create intermediate states where the regen group is internally inconsistent. The whole thing is ~3 files of code + ~3 files of tests; cohesive scope of one S2 unit.
- **Suggested executor model:** Opus. The change is mechanical in shape but the test oracles need careful hand-computation against `season_test`, and the forced-Friday-extreme test case has subtle solver-behaviour expectations (must observe that `FORCED_GAMES` produces the Friday count without the atom enforcing it). Erring to Opus on the line, per `basic`.
- **Dependencies within plan:** none — single unit.
- **No-mock test outline:** Tests construct real `data` dicts via `load_season_data('test')` then mutate `num_rounds` / `teams` / `FORCED_GAMES` to produce each scenario. CP-SAT model assertions use the real `cp_model.CpModel()` and `AwayClubHomeWeekendsCount().apply(...)`. Regen-soft tests use the real `SoftConstraintBucket` flow per the existing test pattern. Hand oracles per the scenario tables in DoD #6.
- **Acceptance:** all the DoD criteria + `scripts/bisect_core_feasibility.py` shows the canonical pair (AwayClub + AwayClubBalance) reaching search per DoD #7.

### Unit B — Worktree teardown (DoD #8)

- **Files touched:** none in the codebase.
- **Actions:** In the parent repo: (1) `git worktree remove C:/Users/c3205/Documents/Code/python/draw-s035-flense` first (removes the checked-out worktree at that path), then (2) `git branch -D spec035-flense` (removes the local branch — the worktree must be removed first or `branch -D` will be rejected by git). `spec035-flense` has no remote tracking branch (confirmed: not in `git branch -r`), so "never delete remote branches" rule is satisfied trivially. Verify both gone after.
- **Suggested executor:** orchestrator (S0 housekeeping — no code change).
- **Dependencies:** runs LAST (after Unit A is merged and the post-merge verification has confirmed nothing references `spec035-flense`).

## Doc registry

- `docs/system/COUNT_ADJUSTERS.md` — drop dead-function sections; add derived-range section.
- `docs/system/CONSTRAINT_INVENTORY.md` — update `AwayClubHomeWeekendsCount` row.
- `docs/todo/00-dependency-tree.md` — add spec-037 entry, update parallelism list.

## Out of scope

- **`ClubVsClubStackedWeekends` granularity rework** — filed separately as **spec-038**. Spec-035
  §7 surfaced both atoms together; they are independent code paths (different atoms, different
  helper functions in the same module, no shared call site after this spec lands). spec-038
  proceeds in parallel.
- **`AwayClubPerOpponentAndAggregateHomeBalance`** — untouched. The redesigned bounds in
  `AwayClubHomeWeekendsCount` are constructed to be jointly satisfiable with this atom's
  per-opponent + aggregate inequalities (the joint feasibility check is the canonical acceptance
  probe, DoD #7).
- **`phl_forced_friday_meetings`** — STAYS as-is. spec-038 still uses it for the PHL Sunday-budget
  subtraction (PHL pairs are 1×1 so the per-club concept doesn't apply, but per-pair does).
- **`spec-035` ULTIMATE e2e run** — the actual core+symmetry-readout run resumes only after BOTH
  spec-037 and spec-038 land. Not in scope here; tracked via spec-035's own status.
- **Changing `num_rounds` semantics or the three-tier override** — the redesign reads
  `num_rounds[g]` exactly as the helper already did; no schema change.
- **Generalising to a third away-based club** — atoms iterate `home_field_map`, so any future
  away club picks the new derived bounds automatically; no further code needed.

## Dependencies

- `depends_on: none`. spec-037 is fully independent — it touches code spec-035 was investigating,
  but spec-035's ULTIMATE run is itself blocked on this work + spec-038, so it depends on us,
  not the other way around. The `spec035-flense` WIP is superseded (deleted in Unit B), not
  depended on.
- spec-038 runs in parallel; no shared files between the two specs (verified at plan time:
  spec-037 touches `away_club_home_weekends_count*.py` + the per-club helpers in
  `_phl_forced_friday_helper.py`; spec-038 touches `club_vs_club_stacked_*.py` +
  `_club_vs_club_stacked_shared.py` + the per-pair helper. Only overlap is
  `_phl_forced_friday_helper.py`, where spec-037 removes per-club functions and spec-038 leaves
  the per-pair function unchanged — additive, no merge contention).

## Risks & blast radius

- **Bounds may still be too tight or too loose when combined with `AwayClubPerOpponentAndAggregateHomeBalance`**
  in production-shaped configs (where `num_rounds` and team layouts differ from `season_test`).
  Mitigation: the bisect probe in DoD #7 is the canonical acceptance test; if it passes on
  `season_test`, the joint feasibility is established for at least one realistic shape. Production
  shape gets validated by spec-035's eventual e2e run, not by this spec.
- **Orphan-deletion confidence.** The public functions `phl_forced_friday_count`,
  `away_club_required_sundays`, `away_club_total_weekends` are removed. Their only callers
  outside the helper module are: the hard atom, the regen-soft twin, and the test files — all
  of which are rewritten in Unit A. The private helpers `_entry_targets_club_phl_friday` and
  `_iter_candidate_friday_phl_keys` are also removed (their only caller is `phl_forced_friday_count`).
  `_friday_phl_forced_entries` STAYS — it is also called by `phl_forced_friday_meetings`
  (verified grep at plan time: `_phl_forced_friday_helper.py:327`). Implementation must re-run
  the grep immediately before deletion to confirm no concurrent consumer was added by other agents.
  (review fix — C1)
- **Regen group consistency.** The regen-soft twin must be rewritten *atomically* with the hard
  atom (same Unit A) — if the hard atom ships first with new bounds and the soft twin still
  computes the old three-way penalty, the regen group's deviation-direction signals would be
  internally inconsistent. Single-unit scope guarantees atomicity.
- **Test oracle drift.** `season_test` and the `num_rounds` config can shift under us. Tests in
  Unit A use a frozen fixture (constructed inline in the test) for the bounds-arithmetic
  scenarios, and the `load_season_data('test')` scenarios assert against the *current* config
  values at test runtime — if the config changes, the tests recompute the oracle the same way.

## Open Questions

0 — the design is fully specified by the worked examples and the convenor's go-ahead this session.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->

0. **Do NOT start without an explicit user instruction to implement this plan.** A `ready` status
   means "authorised to be built when asked", not "build now". If you arrived here straight off
   authoring/review with no user go-ahead, STOP and ask.
1. Status must be `ready` (carries a `reviewed:` stamp). If `review_pending`/`under_review`, let
   review finish. If `blocked`, STOP: the Open Questions need the user.
2. Only after the user says to implement: stamp `building`, claim `owner`. You are the
   orchestrator (Opus).
3. **Unit A** runs first on its own worktree+branch off `final-form`:
   - `git worktree add -b spec037-unitA C:/Users/c3205/Documents/Code/python/draw-spec037-A final-form` from the parent repo, or use `EnterWorktree`.
   - Delegate to an Opus subagent with this plan as the brief.
   - Run gates: type-check + `pytest tests/atoms/test_away_club_home_weekends_count.py tests/atoms/test_away_club_home_weekends_count_regen_soft.py tests/atoms/test_phl_forced_friday_helper.py -v` (all pass with hand-computed oracles); AST sweep for dead code in the touched files; the bisect-harness probe in DoD #7.
   - Launch `/adversarial` Mode B on the diff. Route fixes, re-verify.
   - Push branch → merge to `final-form` → push `final-form` → post-merge verification (re-run the bisect probe on the merged source branch).
   - Tear down the worktree.
4. **Unit B** runs LAST: in the parent repo, `git worktree remove C:/Users/c3205/Documents/Code/python/draw-s035-flense && git branch -D spec035-flense`. Verify both gone.
5. When all units pass: stamp the plan `done`, archive to `docs/todo/done/spec-037-away-club-home-weekends-derived-range.md`, update `docs/todo/00-dependency-tree.md`.
