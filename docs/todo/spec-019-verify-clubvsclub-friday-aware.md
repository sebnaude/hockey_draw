<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (additive — new test only; no shared-file edits) -->

# spec-019 — Lock in `ClubVsClubStackedWeekends` per-pair Friday-awareness with a regression test

## Why

The convenor asked whether `ClubVsClubStackedWeekends` is Friday-aware: if PHL plays Tigers 3×
and 2nd plays Tigers 3× but **one PHL-vs-Tigers meeting is FORCED onto a Friday**, only **2**
weekends should require PHL/2nd alignment (the 3rd has only 2nd playing).

Research confirms this is **already implemented correctly and per-pair**:
- `constraints/atoms/club_vs_club_stacked_weekends.py` computes the Sunday budget via
  `pair_grade_sunday_meetings(data, pair, grade)` and pins `sum_w play[g,w] == budget`.
- `_club_vs_club_stacked_shared.py::pair_grade_sunday_meetings` returns, for PHL only,
  `max(0, total_meetings(pair) - phl_forced_friday_meetings(club_a, club_b))`.
- `_phl_forced_friday_helper.py::phl_forced_friday_meetings(data, club_a, club_b)` counts
  FORCED PHL Friday games **for that specific club pair**, with a greedy partition that avoids
  double-counting umbrella vs per-pair FORCED scopes.

So no behaviour change is needed. The risk is a future refactor silently breaking the per-pair
reduction (e.g. reverting to an aggregate Friday count). This spec adds a **dedicated
regression test** that pins the exact scenario the convenor described, so the guarantee is
enforced by CI rather than by reading the code.

## Definition of Done

1. A new test (in `tests/atoms/test_club_vs_club_stacked_alignment.py` or a new
   `tests/atoms/test_cvc_stacked_friday_aware.py`) encodes the convenor's scenario with
   synthetic data (no mocks), GWT structure, hand-computed oracle:
   - **Given:** a 2-club-pair fixture (one club's PHL + 2nd both meet the other club 3× each
     across the season) and ONE FORCED entry pinning a single PHL meeting of that pair onto a
     Friday at the away venue.
   - **When:** `ClubVsClubStackedWeekends().apply(...)` runs.
   - **Then:** the PHL Sunday budget for that pair == **2** (hand oracle: 3 total − 1 forced
     Friday), the 2nd-grade budget == **3**, and the pinned `sum(play_PHL) == 2` constraint is
     present; the model is FEASIBLE and a solve places PHL/2nd aligned on exactly 2 Sundays.
2. A direct unit test on `pair_grade_sunday_meetings` and `phl_forced_friday_meetings`
   asserting the per-pair (not aggregate) reduction: a FORCED Friday entry for pair (A,B) must
   NOT reduce the budget of an unrelated pair (A,C).
3. Both tests pass on the current code with **no production change** (this is verify-only). If
   either fails, that is a real bug — STOP and surface it (the spec then expands to a fix).
4. Full suite green.

## Implementation units

### Unit A — Regression tests for per-pair Friday-aware stacking
- Files: `tests/atoms/test_cvc_stacked_friday_aware.py` (new) or extend
  `tests/atoms/test_club_vs_club_stacked_alignment.py`. No production files.
- Test: per DoD 1 + 2. Hand-compute budgets and assert against `pair_grade_sunday_meetings`
  return values and the emitted CP-SAT constraint.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — add a one-line note to the
  `ClubVsClubStackedWeekends` row confirming per-pair FORCED-Friday budget reduction is
  regression-tested (point to the test).
- `docs/todo/GOALS.md` — add spec-019 row (status: verify-only).

## Out of scope

- Any change to the stacking algorithm itself (it's correct).
- Non-PHL Friday handling (only PHL plays Friday; other grades' budgets are full by design).
- Maitland/away-venue Friday counts — see spec-015 (Gosford) and the Maitland-Friday note.
