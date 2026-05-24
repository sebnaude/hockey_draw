# spec-027: GWT pass — no mocks, hand-computed oracles, Given/When/Then.
"""Tests for the spec-027 regen-soft atom `ClubVsClubStackedWeekendsRegenSoft`.

The atom is the SOFT analogue of the hard `ClubVsClubStackedWeekends`: instead of
hard-pinning each grade's Sunday play count to its budget and hard-forcing the
nested-superset implication chain, it emits PENALTY vars:

  * one IntVar `dev = |sum_w play[g, w] - budget|` per (pair, grade) — units =
    how many weekends the grade's Sunday count is off-budget;
  * one BoolVar `v = lo_play AND NOT hi_play` per (consecutive grade pair, week)
    — 1 unit per MISSING stacked coincidence (lower-count grade plays a Sunday
    where the higher-count grade does not).

Because nothing is hard-forbidden, the model is FEASIBLE for ANY X. We PIN the
play pattern via `model.Add(X[key] == 0/1)` so the penalty values are fully
determined, then read them back and compare to a hand oracle.
"""
from __future__ import annotations

import os
import sys

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms.clubvsclub_stacked_weekends_regen_soft import (
    REGEN_CLUBVSCLUB_STACKED_WEEKENDS_DEFAULT_WEIGHT,
    ClubVsClubStackedWeekendsRegenSoft,
)
from constraints.atoms._club_vs_club_stacked_shared import (
    pair_grade_sunday_meetings,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_vs_club_stacked_fixture import (
    build_model_X,
    build_stacked_fixture,
    solve_with_timeout,
)


PAIR = ('Maitland', 'Norths')
BUCKET_NAME = 'regen_clubvsclub_stacked_weekends'


def _sunday_key_for(X, grade, week, *, field='EF'):
    """Return one Sunday X-key for the (Maitland, Norths) pair in `grade`/`week`
    on `field`. The fixture has exactly one matchup per grade for the pair, so
    there's one game; we pick the (week, slot=1, field) candidate."""
    for key in X:
        if len(key) < 11 or key[3] != 'Sunday':
            continue
        if key[2] != grade or key[6] != week:
            continue
        if key[9] != field or key[4] != 1:
            continue
        return key
    raise AssertionError(
        f'No Sunday key for grade={grade} week={week} field={field}'
    )


def _pin_play_pattern(model, X, pattern):
    """Force ON exactly one Sunday game (slot 1, EF) per (grade, play-week).

    `pattern` is `{grade: set(weeks_the_grade_plays)}`. Combined with
    `_force_off_complement`, this makes play[grade, week] deterministic: 1 iff
    week in pattern[grade], else 0.
    """
    for grade, weeks in pattern.items():
        for w in weeks:
            model.Add(X[_sunday_key_for(X, grade, w)] == 1)


def _force_off_complement(model, X, data, pattern):
    """Force OFF every pair Sunday var not explicitly turned ON by the pattern."""
    teams = data['teams']
    team_club = {t.name: t.club.name for t in teams}
    on_keys = set()
    for grade, weeks in pattern.items():
        for w in weeks:
            on_keys.add(_sunday_key_for(X, grade, w))
    for key, var in X.items():
        if len(key) < 11 or key[3] != 'Sunday':
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2 or {c1, c2} != set(PAIR):
            continue
        if key in on_keys:
            continue
        model.Add(var == 0)


def _total_penalty(data, solver):
    """Sum the bucket's penalty var VALUES (raw units, not weighted)."""
    bucket = data['penalties'][BUCKET_NAME]
    return sum(solver.Value(v) for v in bucket['penalties'])


# ---------------------------------------------------------------------------
# Scenario 1 — violation: missing coincidence + budget deviation
# ---------------------------------------------------------------------------


class TestScenarioOneViolation:
    """Given a 2-grade pair fixture (PHL budget=2, 2nd budget=1; T=2 so each is
    a single matchup), When we PIN a play pattern that (a) leaves PHL one short
    of budget and (b) has 2nd play a Sunday where PHL does NOT, Then the atom's
    total penalty equals the exact hand-computed unit count and the model is
    FEASIBLE.

    Fixture budgets (no FORCED Fridays, T=2 so meetings == num_rounds):
        PHL = 2, 2nd = 1.   Sorted desc: PHL(2), 2nd(1) → 1 consecutive pair
        (hi=PHL, lo=2nd).

    PINNED play pattern over weeks {1,2,3,4}:
        PHL plays weeks {1}        → PHL Sunday count = 1
        2nd plays weeks {2}        → 2nd Sunday count = 1

    Hand oracle:
      Budget deviations (one dev IntVar per grade):
        PHL: |1 - 2| = 1
        2nd: |1 - 1| = 0
        budget penalty subtotal = 1
      Missing coincidences (v = lo_play AND NOT hi_play, for lo=2nd, hi=PHL,
      per week):
        week 1: lo(2nd)=0, hi(PHL)=1 → 0
        week 2: lo(2nd)=1, hi(PHL)=0 → 1   ← 2nd plays but PHL doesn't
        week 3: lo=0, hi=0 → 0
        week 4: lo=0, hi=0 → 0
        coincidence penalty subtotal = 1
      TOTAL penalty = 1 + 1 = 2.
    """

    PATTERN = {'PHL': {1}, '2nd': {2}}
    EXPECTED_TOTAL = 2

    def _data(self):
        return build_stacked_fixture(
            grade_counts={'PHL': 2, '2nd': 1},
            num_weeks=4,
            slots_per_field=4,
        )

    def test_budgets_are_two_and_one(self):
        """Sanity: the fixture really produces PHL=2, 2nd=1 Sunday budgets."""
        data = self._data()
        assert pair_grade_sunday_meetings(data, PAIR, 'PHL') == 2
        assert pair_grade_sunday_meetings(data, PAIR, '2nd') == 1

    def test_feasible_and_total_penalty_is_two(self):
        data = self._data()
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)

        _pin_play_pattern(model, X, self.PATTERN)
        _force_off_complement(model, X, data, self.PATTERN)

        n = ClubVsClubStackedWeekendsRegenSoft().apply(model, X, data, reg)
        # 2 budget dev vars (PHL, 2nd) + 1 grade-pair × 4 weeks = 4 coincidence
        # vars → 6 penalty constraints reported.
        assert n == 6

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Soft atom must keep the model feasible for any X; got {status}'
        )
        assert _total_penalty(data, solver) == self.EXPECTED_TOTAL


# ---------------------------------------------------------------------------
# Scenario 2 — clean: budgets met, coincidences hold
# ---------------------------------------------------------------------------


class TestScenarioTwoClean:
    """Given the same PHL=2, 2nd=1 pair fixture, When we PIN a properly stacked
    pattern (PHL plays exactly its 2 Sundays, 2nd plays a subset of those so
    every 2nd-Sunday coincides with a PHL-Sunday), Then total penalty == 0 and
    the model is FEASIBLE.

    PINNED play pattern over weeks {1,2,3,4}:
        PHL plays weeks {1, 2}     → PHL Sunday count = 2 == budget
        2nd plays weeks {1}        → 2nd Sunday count = 1 == budget, and
                                     week 1 also has PHL (coincidence holds)

    Hand oracle:
      Budget deviations: PHL |2-2|=0, 2nd |1-1|=0 → 0.
      Missing coincidences (lo=2nd, hi=PHL):
        week 1: lo=1, hi=1 → 0
        week 2: lo=0, hi=1 → 0
        weeks 3,4: lo=0, hi=0 → 0
      TOTAL penalty = 0.
    """

    PATTERN = {'PHL': {1, 2}, '2nd': {1}}

    def _data(self):
        return build_stacked_fixture(
            grade_counts={'PHL': 2, '2nd': 1},
            num_weeks=4,
            slots_per_field=4,
        )

    def test_feasible_and_zero_penalty(self):
        data = self._data()
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)

        _pin_play_pattern(model, X, self.PATTERN)
        _force_off_complement(model, X, data, self.PATTERN)

        n = ClubVsClubStackedWeekendsRegenSoft().apply(model, X, data, reg)
        assert n == 6

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert _total_penalty(data, solver) == 0


# ---------------------------------------------------------------------------
# Atom-mechanics coverage
# ---------------------------------------------------------------------------


class TestAtomMechanics:
    def test_zero_weight_returns_zero_and_no_bucket(self):
        """Given penalty weight 0, When applied, Then the atom no-ops (returns 0,
        creates no penalties)."""
        data = build_stacked_fixture(
            grade_counts={'PHL': 2, '2nd': 1}, num_weeks=4, slots_per_field=4,
        )
        data['penalty_weights'] = {'regen_clubvsclub_stacked_weekends': 0}
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)
        n = ClubVsClubStackedWeekendsRegenSoft().apply(model, X, data, reg)
        assert n == 0
        assert BUCKET_NAME not in data.get('penalties', {})

    def test_bucket_weight_uses_default(self):
        """Given no configured weight, When applied, Then the bucket records the
        module-default weight (80000)."""
        data = build_stacked_fixture(
            grade_counts={'PHL': 2, '2nd': 1}, num_weeks=4, slots_per_field=4,
        )
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)
        ClubVsClubStackedWeekendsRegenSoft().apply(model, X, data, reg)
        assert (
            data['penalties'][BUCKET_NAME]['weight']
            == REGEN_CLUBVSCLUB_STACKED_WEEKENDS_DEFAULT_WEIGHT
            == 80000
        )

    def test_overbudget_does_not_raise(self):
        """The hard atom RAISES ValueError when a budget exceeds available weeks.
        The soft atom must NOT — it stays feasible (the deviation just gets
        penalised). Given PHL budget 10 with only 3 weeks, When applied + solved,
        Then no raise and FEASIBLE."""
        data = build_stacked_fixture(
            grade_counts={'PHL': 10}, num_weeks=3, slots_per_field=4,
        )
        model, X = build_model_X(data)
        reg = HelperVarRegistry(model)
        # Must not raise.
        ClubVsClubStackedWeekendsRegenSoft().apply(model, X, data, reg)
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
