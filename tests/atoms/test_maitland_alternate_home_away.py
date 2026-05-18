"""Tests for the MaitlandAlternateHomeAway atom (spec-012, Part B).

The atom adds a soft penalty per consecutive Maitland playable-week pair
where both weekends are the same type (HH or AA). HH is already
hard-forbidden by NonDefaultHomeGrouping in the production stage list, so in
the production solver the AA branch does the work — but the atom codes both
symmetrically for robustness.

All tests use real CP-SAT models with hand-computed oracles. No mocks.

Fixture: one pair (Maitland 1st, Norths 1st), one game per week across 4
weeks. Each week has two timeslot choices — Broadmeadow (away for Maitland)
and Maitland Park (home for Maitland). This gives us 2 X-vars per week, 8
total, and the solver can choose the H/A pattern by activating exactly one
var per week.

Scenarios:

1. `TestPenaltyForAARun` — given a forced AA pattern (week 1 + 2 both away),
   the atom contributes a penalty of 1 (`both_away[w1,w2] == 1`).

2. `TestSolverPrefersAlternation` — when the solver is free to pick (with a
   matching number of home/away slots), minimising the bucket drives it to
   an alternating pattern.

3. `TestByeWeekCarriesNoPenalty` — when one of the consecutive weeks has no
   game (no Maitland var that week), neither both_home nor both_away can
   fire → that pair contributes 0.

4. `TestNoMaitlandInConfig` — when home_field_map has no Maitland entry, the
   atom returns 0 (graceful no-op).

5. `TestWeightZeroDisables` — `penalty_weights['maitland_alternate_home_away']
   = 0` disables the atom entirely (returns 0, no bucket).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms.maitland_alternate_home_away import (
    MaitlandAlternateHomeAway,
)
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


BROADMEADOW = 'Newcastle International Hockey Centre'
MAITLAND_PARK = 'Maitland Park'


def _registry(model) -> HelperVarRegistry:
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


def _build_fixture(
    *,
    weeks: List[int],
    include_home_each_week: bool = True,
    include_away_each_week: bool = True,
    maitland_in_home_map: bool = True,
) -> Dict:
    """Build a single-pair Maitland-vs-Norths fixture across `weeks`.

    Each listed week gets up to two timeslots: one at Broadmeadow (away for
    Maitland) and one at Maitland Park (home for Maitland). Toggles let
    individual tests force-include or omit a venue.
    """
    bm_ef = PlayingField(location=BROADMEADOW, name='EF')
    mp_main = PlayingField(location=MAITLAND_PARK, name='Main')

    clubs = [
        Club(name='Maitland', home_field=MAITLAND_PARK),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland 1st', club=clubs[0], grade='PHL'),
        Team(name='Norths 1st', club=clubs[1], grade='PHL'),
    ]
    grades = [Grade(name='PHL', teams=[t.name for t in teams])]

    timeslots = []
    for week in weeks:
        # Synthetic date — each week one Sunday a week apart.
        date_str = f'2026-03-{15 + week * 7:02d}'  # week=1 -> 22, week=2 -> 29, etc.
        if include_away_each_week:
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time='11:30',
                week=week, day_slot=1, field=bm_ef, round_no=week,
            ))
        if include_home_each_week:
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time='13:30',
                week=week, day_slot=2, field=mp_main, round_no=week,
            ))

    home_field_map = {'Maitland': MAITLAND_PARK} if maitland_in_home_map else {}

    return {
        'games': [('Maitland 1st', 'Norths 1st', 'PHL')],
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': [bm_ef, mp_main],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'PHL': len(weeks), 'max': len(weeks)},
        'constraint_slack': {},
        'penalty_weights': {'maitland_alternate_home_away': 100},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': home_field_map,
        'constraint_defaults': {},
        'preferred_weekends': [],
    }


def _build_model_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        t1_c, t2_c = sorted((t1, t2))
        for ts in data['timeslots']:
            key = (
                t1_c, t2_c, grade,
                ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no,
                ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_w{ts.week}_{ts.field.location[:2]}'
            )
    return model, X


def _away_vars(X, week):
    return [v for k, v in X.items() if k[6] == week and k[10] == BROADMEADOW]


def _home_vars(X, week):
    return [v for k, v in X.items() if k[6] == week and k[10] == MAITLAND_PARK]


# ---------------------------------------------------------------------------
# Scenario 1: forced AA pattern produces a penalty of exactly 1
# ---------------------------------------------------------------------------

class TestPenaltyForAARun:
    """Given two consecutive weeks both forced away,
    When the atom applies,
    Then the both_away[w0,w1] BoolVar = 1 and both_home = 0.

    Hand oracle:
      Fixture: 2 weeks, each with 1 home + 1 away timeslot for the single
      Maitland pair. Force the away vars ON and home vars OFF in both weeks.
      home_ind[w] = 0 → both_home = 0.
      away_ind[w] = 1 (any var=1 and home var=0) → both_away = 1.
      bucket has 2 BoolVars per pair (both_home, both_away) × 1 pair = 2 vars,
      with values (0, 1). Sum of bucket = 1.
    """

    def test_aa_pair_contributes_one(self):
        data = _build_fixture(weeks=[1, 2])
        model, X = _build_model_X(data)

        # Force away both weeks (pin away var to 1, home var to 0)
        for w in (1, 2):
            for v in _away_vars(X, w):
                model.Add(v == 1)
            for v in _home_vars(X, w):
                model.Add(v == 0)

        n = MaitlandAlternateHomeAway().apply(model, X, data, _registry(model))
        # 1 consecutive pair × 2 BoolVars (both_home, both_away) = 2.
        assert n == 2, f'expected 2 penalty vars, got {n}'

        bucket = data['penalties']['maitland_alternate_home_away']
        assert len(bucket['penalties']) == 2

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # Sum of bucket = 1 (only both_away contributes).
        total = sum(solver.Value(p) for p in bucket['penalties'])
        assert total == 1, f'expected sum == 1 (only both_away), got {total}'


# ---------------------------------------------------------------------------
# Scenario 2: solver minimises bucket → picks alternating pattern
# ---------------------------------------------------------------------------

class TestSolverPrefersAlternation:
    """Given 4 weeks with exactly 2 home + 2 away required,
    When minimising the bucket,
    Then the solver picks an alternating (H A H A or A H A H) pattern.

    Hand oracle:
      Fixture: 4 weeks, each with 1 home + 1 away slot. Require exactly one
      game per week (sum vars per week == 1) — 4 games total. Also require
      total_home == 2 and total_away == 2 (so the choice is purely about
      ordering, not balance).

      With matched counts, the minimum-AA pattern is either H A H A or A H A H
      — both have zero (HH or AA) pairs. Other patterns:
        - H H A A → 1 HH + 1 AA = 2 same-type pairs
        - H A A H → 1 AA
        - A H H A → 1 HH
        - A A H H → 1 AA + 1 HH = 2
      Optimal bucket-sum = 0. The solver must land on H A H A or A H A H.

      Each "same-type pair" contributes 1 penalty BoolVar = 1 to the bucket.
    """

    def test_solver_picks_alternating_pattern(self):
        data = _build_fixture(weeks=[1, 2, 3, 4])
        model, X = _build_model_X(data)

        # Exactly one game per week.
        for w in (1, 2, 3, 4):
            wk_vars = [v for k, v in X.items() if k[6] == w]
            assert len(wk_vars) == 2
            model.Add(sum(wk_vars) == 1)

        # Force 2 home + 2 away across the season.
        all_home = [v for k, v in X.items() if k[10] == MAITLAND_PARK]
        all_away = [v for k, v in X.items() if k[10] == BROADMEADOW]
        model.Add(sum(all_home) == 2)
        model.Add(sum(all_away) == 2)

        n = MaitlandAlternateHomeAway().apply(model, X, data, _registry(model))
        # 3 consecutive pairs × 2 = 6 BoolVars.
        assert n == 6, f'expected 6 penalty vars (3 pairs × 2), got {n}'

        bucket = data['penalties']['maitland_alternate_home_away']
        model.Minimize(sum(bucket['penalties']))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        assert status == cp_model.OPTIMAL

        # Hand oracle: optimal bucket sum = 0 (alternating pattern available).
        assert solver.ObjectiveValue() == 0

        # Recover the pattern: per week, did we pick home or away?
        pattern = []
        for w in (1, 2, 3, 4):
            h = sum(solver.Value(v) for v in _home_vars(X, w))
            a = sum(solver.Value(v) for v in _away_vars(X, w))
            assert h + a == 1
            pattern.append('H' if h == 1 else 'A')

        # Verify it's one of the two alternating patterns.
        assert pattern in (['H', 'A', 'H', 'A'], ['A', 'H', 'A', 'H']), (
            f'expected H A H A or A H A H, got {pattern}'
        )


# ---------------------------------------------------------------------------
# Scenario 3: bye-week pair contributes 0
# ---------------------------------------------------------------------------

class TestByeWeekCarriesNoPenalty:
    """Given week 2 has NO Maitland var (bye), the (w1, w2) pair must be
    excluded from the alternation check.

    Implementation detail: the atom only iterates weeks that have at least one
    Maitland var. A bye week is absent from `weeks`, so the pair list jumps
    from week 1 directly to week 3 — and the (w1, w3) pair is the only one
    that contributes a penalty.

    Hand oracle:
      Fixture: weeks 1, 3 only (week 2 has zero timeslots, modelling a bye
      for the entire competition — equivalent for Maitland purposes).
      → `weeks` after collection = [1, 3]. 1 consecutive pair × 2 BoolVars = 2.

      Force away in week 1, home in week 3: both_home = 0, both_away = 0
      (since week 3's home_ind = 1, week 1's home_ind = 0 → 0+1-1 = 0).
      Bucket sum = 0.

      Force away in BOTH weeks 1 and 3: both_away = 1 (sum=1).
    """

    def test_bye_excludes_pair(self):
        # Skip week 2 — fixture only has weeks 1 and 3.
        data = _build_fixture(weeks=[1, 3])
        model, X = _build_model_X(data)

        n = MaitlandAlternateHomeAway().apply(model, X, data, _registry(model))
        # 2 weeks → 1 consecutive pair → 2 BoolVars.
        assert n == 2, f'expected 2 penalty vars (1 pair × 2), got {n}'

    def test_aa_across_bye_still_counted_as_consecutive_pair(self):
        # The atom treats playable rounds as the unit of "consecutive" — so a
        # bye week between two away weekends still produces an AA pair penalty.
        # This matches the convenor's view that "no-play weeks don't count as
        # gaps" (CLAUDE.md §"Weeks vs Rounds").
        data = _build_fixture(weeks=[1, 3])
        model, X = _build_model_X(data)
        for w in (1, 3):
            for v in _away_vars(X, w):
                model.Add(v == 1)
            for v in _home_vars(X, w):
                model.Add(v == 0)

        MaitlandAlternateHomeAway().apply(model, X, data, _registry(model))
        bucket = data['penalties']['maitland_alternate_home_away']

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        # Both away → both_away = 1, both_home = 0. Sum = 1.
        total = sum(solver.Value(p) for p in bucket['penalties'])
        assert total == 1


# ---------------------------------------------------------------------------
# Scenario 4: no Maitland in home_field_map → graceful no-op
# ---------------------------------------------------------------------------

class TestNoMaitlandInConfig:
    def test_returns_zero_without_home_field_entry(self):
        data = _build_fixture(weeks=[1, 2], maitland_in_home_map=False)
        model, X = _build_model_X(data)
        n = MaitlandAlternateHomeAway().apply(model, X, data, _registry(model))
        assert n == 0
        # Bucket not created.
        assert 'maitland_alternate_home_away' not in data.get('penalties', {})


# ---------------------------------------------------------------------------
# Scenario 5: weight 0 disables the atom
# ---------------------------------------------------------------------------

class TestWeightZeroDisables:
    def test_weight_zero_returns_zero(self):
        data = _build_fixture(weeks=[1, 2])
        data['penalty_weights']['maitland_alternate_home_away'] = 0
        model, X = _build_model_X(data)
        n = MaitlandAlternateHomeAway().apply(model, X, data, _registry(model))
        assert n == 0
        assert 'maitland_alternate_home_away' not in data.get('penalties', {})
