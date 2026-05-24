# spec-027 regen-soft GWT test — no mocks, real CP-SAT, hand-computed oracles.
"""Tests for the spec-027 SOFT atom `ClubVsClubStackedCoLocationRegenSoft`.

The atom is the soft analogue of the HARD `ClubVsClubStackedCoLocation`: on a
Sunday where a club-pair stacks (>= 2 grades), instead of FORBIDDING multi-field
/ gapped layouts it charges a penalty equal to the violation amount:

  - co-field penalty  = max(0, distinct_fields_used - 1)   (1 unit / extra field)
  - contiguity penalty = number of internal slot gaps        (1 unit / gap)

Both scenarios PIN an explicit layout (so X is fully determined for the pair),
solve while MINIMISING the penalty bucket sum, assert FEASIBLE, and compare the
realised penalty to a hand-computed oracle.

Pair under test: (Maitland, Norths), grades {PHL: 2, 2nd: 2} on a 4-slot fixture.
"""
from __future__ import annotations

import os
import sys

from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms.clubvsclub_stacked_colocation_regen_soft import (
    ClubVsClubStackedCoLocationRegenSoft,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_vs_club_stacked_fixture import (
    build_model_X,
    build_stacked_fixture,
    solve_with_timeout,
)


PAIR = ('Maitland', 'Norths')


def _fixture():
    """Two grades (PHL=2, 2nd=2), num_weeks=3, 4 slots/field — enough slots for
    an internal gap (need >= 3 distinct slots), >= 2 grades so the pair stacks."""
    return build_stacked_fixture(
        grade_counts={'PHL': 2, '2nd': 2}, num_weeks=3, slots_per_field=4,
    )


def _pair_sunday_keys(data, X):
    """All Sunday (Maitland, Norths) cross-club var keys in X, by (grade, week)."""
    team_club = {t.name: t.club.name for t in data['teams']}
    out = []
    for key in X:
        if len(key) < 11 or key[3] != 'Sunday':
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2 or {c1, c2} != set(PAIR):
            continue
        out.append(key)
    return out


def _pin_layout(model, data, X, placements):
    """Pin the pair's Sunday games to exactly `placements` and zero everything
    else for the pair on Sundays.

    `placements` is a dict {(grade, week): (field_name, day_slot)}. The var
    whose key matches each placement is forced to 1; every other Sunday var for
    the pair is forced to 0. This fully determines the pair's Sunday layout so
    the penalty is a deterministic function of `placements`.
    """
    pins = set()
    for key in _pair_sunday_keys(data, X):
        grade, week, day_slot, field_name = key[2], key[6], key[4], key[9]
        want = placements.get((grade, week))
        if want is not None and want == (field_name, day_slot):
            model.Add(X[key] == 1)
            pins.add((grade, week))
        else:
            model.Add(X[key] == 0)
    # Sanity: every requested placement was matched by a real candidate var.
    assert pins == set(placements.keys()), (
        f'Unmatched placements: requested {set(placements.keys())}, pinned {pins}'
    )


def _bucket_total(data, solver):
    bucket = data['penalties']['regen_clubvsclub_stacked_colocation']
    return sum(solver.Value(v) for v in bucket['penalties'])


# ---------------------------------------------------------------------------
# Scenario 1 — VIOLATION: different fields AND a non-contiguous slot gap.
# ---------------------------------------------------------------------------


class TestScenarioOneViolation:
    """Given the pair's two stacked-grade games on the SAME Sunday (week 1) but
    on DIFFERENT fields and with an internal slot gap, When the soft atom runs
    and the penalty is minimised, Then total penalty == 2 and the model is
    FEASIBLE.

    Hand layout (week 1, NIHC fields EF/WF, slots 1..4):

        slot 1   slot 2   slot 3   slot 4
    EF  PHL      -        -        -
    WF  -        -        2nd      -

      Fields used by the pair this Sunday: {EF, WF} -> 2 distinct.
        co-field penalty = max(0, 2 - 1) = 1.
      Slot-used indicators over slots [1,2,3,4]: {1:1, 2:0, 3:1, 4:0}.
        Internal triples (mid = 2 and mid = 3):
          mid=2: prev(1)=1, next(3)=1, mid(2)=0  -> GAP  (penalty 1)
          mid=3: prev(2)=0                         -> no gap
        contiguity penalty = 1 gap.
      TOTAL = co-field(1) + gaps(1) = 2.
    """

    def test_total_penalty_is_two_and_feasible(self):
        data = _fixture()
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)

        _pin_layout(model, data, X, {
            ('PHL', 1): ('EF', 1),
            ('2nd', 1): ('WF', 3),
        })

        n = ClubVsClubStackedCoLocationRegenSoft().apply(model, X, data, reg)
        assert n >= 1, 'Atom should have emitted at least one penalty term'

        bucket = data['penalties']['regen_clubvsclub_stacked_colocation']
        model.Minimize(sum(bucket['penalties']))
        status, solver = solve_with_timeout(model)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Soft atom must keep the model feasible for any X; got {status}'
        )
        assert _bucket_total(data, solver) == 2, (
            f'Expected total penalty 2 (1 extra field + 1 gap), '
            f'got {_bucket_total(data, solver)}'
        )


# ---------------------------------------------------------------------------
# Scenario 2 — CLEAN: same field, contiguous slots.
# ---------------------------------------------------------------------------


class TestScenarioTwoClean:
    """Given the pair's two stacked-grade games on the SAME Sunday (week 1), the
    SAME field, and CONTIGUOUS slots, When the soft atom runs and the penalty is
    minimised, Then total penalty == 0 and the model is FEASIBLE.

    Hand layout (week 1):

        slot 1   slot 2   slot 3   slot 4
    EF  PHL      2nd      -        -

      Fields used: {EF} -> 1 distinct -> co-field penalty = max(0, 1-1) = 0.
      Slot-used: {1:1, 2:1, 3:0, 4:0}.
        mid=2: prev(1)=1, next(3)=0 -> no gap
        mid=3: prev(2)=1, next(4)=0 -> no gap
        contiguity penalty = 0.
      TOTAL = 0.
    """

    def test_total_penalty_is_zero_and_feasible(self):
        data = _fixture()
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)

        _pin_layout(model, data, X, {
            ('PHL', 1): ('EF', 1),
            ('2nd', 1): ('EF', 2),
        })

        ClubVsClubStackedCoLocationRegenSoft().apply(model, X, data, reg)

        bucket = data['penalties']['regen_clubvsclub_stacked_colocation']
        model.Minimize(sum(bucket['penalties']))
        status, solver = solve_with_timeout(model)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Clean layout must be feasible; got {status}'
        )
        assert _bucket_total(data, solver) == 0, (
            f'Expected total penalty 0 for same-field contiguous layout, '
            f'got {_bucket_total(data, solver)}'
        )
