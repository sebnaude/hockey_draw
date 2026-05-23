<!-- status: done -->
<!-- owner: session=goal-final-form claimed=2026-05-22 -->
<!-- reviewed: adversarial Sonnet review 2026-05-22 — fixes applied inline (see "(review fix — …)" annotations) -->
<!-- depends_on: none (additive — new test only; no shared-file edits) -->

# spec-019 — Lock in `ClubVsClubStackedWeekends` per-pair Friday-awareness with a regression test

## Why

The convenor asked whether `ClubVsClubStackedWeekends` is Friday-aware: if PHL plays Tigers 3×
and 2nd plays Tigers 3× but **one PHL-vs-Tigers meeting is FORCED onto a Friday**, only **2**
weekends should require PHL/2nd alignment (the 3rd has only 2nd playing).

Research confirms this is **already implemented correctly and per-pair**:
- `constraints/atoms/club_vs_club_stacked_weekends.py` computes the Sunday budget via
  `pair_grade_sunday_meetings(data, pair, grade)` and pins `sum_w play[g,w] == budget`.
  The `apply()` signature is `apply(self, model, X, data, registry)` — the `registry`
  (`HelperVarRegistry`) argument is required; tests must supply it (see existing
  `TestStackedAtomSmoke` in `tests/atoms/test_club_vs_club_stacked_alignment.py` for the pattern).
  (review fix — H2: added registry parameter note to prevent implementer mistake)
- `constraints/atoms/_club_vs_club_stacked_shared.py::pair_grade_sunday_meetings` returns,
  for PHL only, `max(0, total_meetings(pair) - phl_forced_friday_meetings(club_a, club_b))`.
- `constraints/atoms/_phl_forced_friday_helper.py::phl_forced_friday_meetings(data, club_a, club_b)`
  counts FORCED PHL Friday games **for that specific club pair**, with a greedy partition that
  avoids double-counting umbrella vs per-pair FORCED scopes.

So no behaviour change is needed. The risk is a future refactor silently breaking the per-pair
reduction (e.g. reverting to an aggregate Friday count). This spec adds a **dedicated
regression test** that pins the exact scenario the convenor described, so the guarantee is
enforced by CI rather than by reading the code.

## Definition of Done

1. A new test (in `tests/atoms/test_cvc_stacked_friday_aware.py` — new file; do NOT extend
   `tests/atoms/test_club_vs_club_stacked_alignment.py` which already contains a correct
   `TestScenarioTwoForcedFridayBudget` class covering the exact DoD-1 solver scenario with
   PHL=4 total / 2 FORCED Fridays / resulting budget=2; extending that file with duplicate
   scenario coverage adds noise — the new file should contain ONLY the gap tests in DoD 2)
   encodes the convenor's scenario with synthetic data (no mocks), GWT structure, hand-computed
   oracle. (review fix — M3: clarified scope split between existing file and new file)
   - **Given:** a 2-club-pair fixture (one club's PHL + 2nd both meet the other club 3× each
     across the season) and ONE FORCED entry pinning a single PHL meeting of that pair onto a
     Friday at the away venue. This FORCED entry MUST be **pair-specific**
     (`teams=[A PHL, B PHL]`), NOT an umbrella `club:`/venue entry —
     `phl_forced_friday_meetings(A,B)` via `_entry_targets_pair_phl_friday` only counts entries
     that name BOTH clubs (returns `False` for umbrella scopes per module docstring); an umbrella
     entry returns 0 from the helper and the budget subtraction would be 0, making the test pass
     trivially without exercising the per-pair reduction.
   - **When:** `ClubVsClubStackedWeekends().apply(model, X, data, registry)` runs (with a
     `HelperVarRegistry(model)` as the `registry` argument — see `TestStackedAtomSmoke` for the
     pattern). (review fix — H2: explicit registry argument)
   - **Then:** `pair_grade_sunday_meetings(data, PAIR, 'PHL')` == **2** (hand oracle: 3 total −
     1 forced Friday), `pair_grade_sunday_meetings(data, PAIR, '2nd')` == **3**, the model is
     FEASIBLE, and a solve places PHL on exactly 2 Sunday weeks for the pair. NOTE: the plan
     originally said "the pinned `sum(play_PHL) == 2` constraint is present" — this is
     untestable via public API (constraints are internal to CP-SAT). Assert via helper return
     value + solve outcome, NOT by introspecting the model's constraint list. (review fix — M2:
     corrected un-introspectable claim)

2. A direct unit test on `pair_grade_sunday_meetings` and `phl_forced_friday_meetings`
   asserting the **A-shared per-pair isolation** (the gap not covered by existing tests):
   a FORCED Friday entry for pair (A, B) must NOT reduce the budget of pair (A, C) where club A
   is shared. Concretely:
   - Fixture has 3 clubs: Maitland, Norths, Wests (all with PHL teams).
   - FORCED entry: 1 Maitland-vs-Norths PHL Friday (`teams=['Maitland PHL', 'Norths PHL']`).
   - `phl_forced_friday_meetings(data, 'Maitland', 'Norths')` == 1.
   - `phl_forced_friday_meetings(data, 'Maitland', 'Wests')` == 0.  ← the gap assertion
   - `pair_grade_sunday_meetings(data, ('Maitland', 'Norths'), 'PHL')` == total − 1.
   - `pair_grade_sunday_meetings(data, ('Maitland', 'Wests'), 'PHL')` == total (unchanged).
   NOTE: `tests/atoms/test_phl_forced_friday_helper.py::TestPhlForcedFridayMeetings::
   test_given_other_pair_entry_returns_zero` only checks (Norths-vs-Wests) doesn't affect
   (Maitland-vs-Norths) — DISJOINT clubs, not A-SHARED. The A-shared cross-pair isolation
   is the specific regression gap this spec must close. (review fix — H1: identified precise
   gap; provided hand-computed oracle for A-shared scenario)

3. Both tests pass on the current code with **no production change** (this is verify-only). If
   either fails, that is a real bug — STOP and surface it (the spec then expands to a fix).

4. Full suite green.

## Implementation units

### Unit A — Regression tests for per-pair Friday-aware stacking

- Files: `tests/atoms/test_cvc_stacked_friday_aware.py` (new). No production files. Do NOT
  edit `tests/atoms/test_club_vs_club_stacked_alignment.py` — it already has the end-to-end
  solver scenario. (review fix — M3: single target file, not "or extend")
- Import the fixture helper via `from tests.atoms.club_vs_club_stacked_fixture import
  build_stacked_fixture` — the existing fixture already supports multi-club scenarios
  through the `grade_counts` parameter; check whether it also supports 3-club setups or
  whether a minimal hand-built fixture is cleaner for DoD 2.
- Test per DoD 1 (solver-level: PHL Sunday count in solution == 2 when 1 Friday FORCED) and
  DoD 2 (helper-level: FORCED for (A,B) does not reduce budget for (A,C)).
- Hand-computed oracles are in DoD 1 and DoD 2 above — use them verbatim in the test
  docstrings; don't re-derive.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — add a one-line note to the
  `ClubVsClubStackedWeekends` row confirming per-pair FORCED-Friday budget reduction is
  regression-tested (point to `tests/atoms/test_cvc_stacked_friday_aware.py`).
  NOTE: The GOALS.md spec-019 row already exists (status: ready); do NOT re-add it.
  (review fix — L1: remove redundant GOALS.md action that is already done)

## Out of scope

- Any change to the stacking algorithm itself (it's correct).
- Non-PHL Friday handling (only PHL plays Friday; other grades' budgets are full by design).
- Maitland/away-venue Friday counts — see spec-015 (Gosford) and the Maitland-Friday note.
- DoD 1 end-to-end solver coverage (already in `TestScenarioTwoForcedFridayBudget` in
  `test_club_vs_club_stacked_alignment.py` — that class is the "existing coverage" baseline
  this spec's DoD 3 "both tests pass" check should confirm still passes).
