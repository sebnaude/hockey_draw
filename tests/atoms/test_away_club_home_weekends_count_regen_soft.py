"""Tests for AwayClubHomeWeekendsCountRegenSoft atom (spec-027, redesigned in spec-037).

Real CP-SAT models, no mocks. GWT style with hand-computed oracles.

The SOFT atom emits, per away-based club, a SINGLE absolute-deviation IntVar
penalty against the Sunday-home indicator total:

    dev = max(0, min_sundays - sum, sum - max_sundays)

(Under ``min <= max`` the under-floor and over-ceiling terms are mutually
exclusive, so ``dev == max(0, min - sum) + max(0, sum - max)``.)

The minimising objective drives ``dev`` to ``|deviation|``. We force the
solver to land the Sunday-home indicator count at known actuals, then assert
the per-club deviation equals the hand-computed magnitude.

Per spec-037 DoD #6: a ONE-club fixture means
``n_penalties = 1 Sunday deviation var`` -> ``normalized_weight = 90000 / 1
= 90000``. We assert the RAW dev value (the penalty bucket holds the IntVar),
not the post-normalisation contribution to the objective — normalisation is
done in ``main_staged._build_normalized_penalty`` at solve-orchestration time,
not by the atom itself.

Three scenarios per DoD #6 (bounds [9, 10] under PHL=20 + 3rd=18):
  - Sunday-home count = 10 (inside bounds): dev = 0.
  - Sunday-home count = 7 (below floor 9): dev = 2; contribution = 2 * 90000 = 180000.
  - Sunday-home count = 12 (above ceiling 10): dev = 2; contribution = 2 * 90000 = 180000.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms._phl_forced_friday_helper import (
    away_club_max_sundays_home,
    away_club_min_sundays_home,
)
from constraints.atoms.away_club_home_weekends_count_regen_soft import (
    REGEN_AWAY_CLUB_HOME_WEEKENDS_COUNT_DEFAULT_WEIGHT,
    AwayClubHomeWeekendsCountRegenSoft,
)
from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Fixture builder — ONE away club (Maitland), PHL + 3rd to fix bounds [9, 10].
# Each of `num_weeks` weeks has ONE Maitland Park Sunday slot per grade (so
# the atom's indicator OR'ing is a no-op — at most one home-Sunday-var per
# (week, grade), at most two per week across both grades). That keeps the
# indicator union under our direct control: pin which weeks have games to
# pin the union count.
# ---------------------------------------------------------------------------


def _build_one_club_fixture(*, num_weeks: int) -> Dict:
    bm_ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Main')

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland PHL', club=clubs[0], grade='PHL'),
        Team(name='Norths PHL', club=clubs[1], grade='PHL'),
        Team(name='Maitland 3rd', club=clubs[0], grade='3rd'),
        Team(name='Norths 3rd', club=clubs[1], grade='3rd'),
    ]
    grade_objs = [
        Grade(name='PHL', teams=['Maitland PHL', 'Norths PHL']),
        Grade(name='3rd', teams=['Maitland 3rd', 'Norths 3rd']),
    ]
    # Bounds:
    #   non-PHL = ['3rd']; floor = 18 // 2 = 9
    #   all = ['PHL', '3rd']; ceiling = max((20+1)//2, 18//2) = max(10, 9) = 10
    num_rounds = {'PHL': 20, '3rd': 18, 'max': 20}

    games: List[Tuple[str, str, str]] = [
        ('Maitland PHL', 'Norths PHL', 'PHL'),
        ('Maitland 3rd', 'Norths 3rd', '3rd'),
    ]

    # One MP Sunday slot per (week, grade) — slot 1 for PHL, slot 2 for 3rd —
    # and one BM Sunday slot per (week, grade) for away placement.
    timeslots: List[Timeslot] = []
    base_date_n = 22
    for week in range(1, num_weeks + 1):
        sun_d = f'2026-03-{base_date_n + 7 * (week - 1):02d}'
        for slot, time in enumerate(['11:30', '13:30'], 1):
            timeslots.append(Timeslot(
                date=sun_d, day='Sunday', time=time, week=week,
                day_slot=slot, field=bm_ef, round_no=week,
            ))
            timeslots.append(Timeslot(
                date=sun_d, day='Sunday', time=time, week=week,
                day_slot=slot, field=mp, round_no=week,
            ))

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grade_objs,
        'clubs': clubs,
        'fields': [bm_ef, mp],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': num_rounds,
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
    }


def _build_X(model: cp_model.CpModel, data: Dict) -> Dict:
    X = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{ts.week}_{ts.day_slot}_{ts.field.name}'
            )
    return X


def _pin_mp_sunday_weeks(model, X: Dict, occupied_weeks: set, num_weeks: int) -> None:
    """Force the Sunday-home INDICATOR union (distinct weeks with >=1 MP
    Sunday game) to equal exactly ``occupied_weeks``.

    For each week:
      - If week IN occupied_weeks: force sum(MP-Sunday vars for the week) >= 1.
      - If week NOT IN occupied_weeks: force sum(MP-Sunday vars) == 0.

    Sets no other constraints (pair sums, slot caps) — those are irrelevant
    to the soft penalty calculation and would over-constrain the fixture.
    """
    from collections import defaultdict
    mp_sun_by_week: Dict[int, List] = defaultdict(list)
    for key, var in X.items():
        if key[3] == 'Sunday' and key[10] == MAITLAND:
            mp_sun_by_week[key[6]].append(var)
    for week in range(1, num_weeks + 1):
        vars_list = mp_sun_by_week.get(week, [])
        if not vars_list:
            continue
        if week in occupied_weeks:
            model.Add(sum(vars_list) >= 1)
        else:
            model.Add(sum(vars_list) == 0)


def _solve(model: cp_model.CpModel, *, time_s: float = 10.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_s
    return solver, solver.Solve(model)


def _total_dev(solver: cp_model.CpSolver, bucket: Dict) -> int:
    return sum(solver.Value(p) for p in bucket['penalties'])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAwayClubHomeWeekendsCountRegenSoft:

    def test_scenario_a_inside_bounds_then_zero_penalty(self):
        """GIVEN ONE-club fixture (Maitland; bounds [9, 10]),
              + Sunday-home indicator union pinned to 10 weeks,
        WHEN regen-soft atom applied and dev minimised,
        THEN dev == 0 (10 is inside [9, 10]).

        Hand-computed:
          min=9, max=10 (bounds helpers on PHL=20, 3rd=18).
          Pinned actual = 10.
          dev = max(0, 9 - 10, 10 - 10) = max(0, -1, 0) = 0.
        n_penalties = 1 (one club). Default raw weight = 90000.
        """
        num_weeks = 22
        data = _build_one_club_fixture(num_weeks=num_weeks)
        assert away_club_min_sundays_home(data, 'Maitland') == 9
        assert away_club_max_sundays_home(data, 'Maitland') == 10

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _pin_mp_sunday_weeks(model, X, set(range(1, 11)), num_weeks)

        registry = HelperVarRegistry(model)
        atom = AwayClubHomeWeekendsCountRegenSoft()
        created = atom.apply(model, X, data, registry)
        assert created == 1, f'Expected 1 penalty IntVar (one club), got {created}'

        bucket = data['penalties']['regen_away_club_home_weekends_count']
        assert bucket['weight'] == REGEN_AWAY_CLUB_HOME_WEEKENDS_COUNT_DEFAULT_WEIGHT
        model.Minimize(sum(bucket['penalties']))

        solver, status = _solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )
        assert _total_dev(solver, bucket) == 0

    def test_scenario_b_below_floor_then_dev_equals_two(self):
        """GIVEN ONE-club fixture (Maitland; bounds [9, 10]),
              + Sunday-home indicator union pinned to 7 weeks (below floor),
        WHEN regen-soft atom applied and dev minimised,
        THEN dev == 2 (|7 - 9| = 2);
              normalised penalty contribution = 2 * (90000 / 1) = 180000.

        Hand-computed:
          dev = max(0, 9 - 7, 7 - 10) = max(0, 2, -3) = 2.
          n_penalties = 1, raw weight = 90000 -> normalized weight per unit = 90000.
          Total normalized contribution = 2 * 90000 = 180000.
        """
        num_weeks = 22
        data = _build_one_club_fixture(num_weeks=num_weeks)
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _pin_mp_sunday_weeks(model, X, set(range(1, 8)), num_weeks)

        registry = HelperVarRegistry(model)
        atom = AwayClubHomeWeekendsCountRegenSoft()
        created = atom.apply(model, X, data, registry)
        assert created == 1

        bucket = data['penalties']['regen_away_club_home_weekends_count']
        model.Minimize(sum(bucket['penalties']))

        solver, status = _solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )
        dev_total = _total_dev(solver, bucket)
        assert dev_total == 2, f'Expected dev=2, got {dev_total}'

        # Hand-computed normalised contribution (mirrors what
        # main_staged._build_normalized_penalty does):
        normalised_weight = max(1, bucket['weight'] // len(bucket['penalties']))
        assert normalised_weight == 90000, (
            f'Expected normalised weight 90000, got {normalised_weight}'
        )
        contribution = dev_total * normalised_weight
        assert contribution == 180000, (
            f'Expected normalised contribution 180000, got {contribution}'
        )

    def test_scenario_c_above_ceiling_then_dev_equals_two(self):
        """GIVEN ONE-club fixture (Maitland; bounds [9, 10]),
              + Sunday-home indicator union pinned to 12 weeks (above ceiling),
        WHEN regen-soft atom applied and dev minimised,
        THEN dev == 2 (|12 - 10| = 2);
              normalised contribution = 2 * 90000 = 180000.

        Hand-computed:
          dev = max(0, 9 - 12, 12 - 10) = max(0, -3, 2) = 2.
        """
        num_weeks = 22
        data = _build_one_club_fixture(num_weeks=num_weeks)
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _pin_mp_sunday_weeks(model, X, set(range(1, 13)), num_weeks)

        registry = HelperVarRegistry(model)
        atom = AwayClubHomeWeekendsCountRegenSoft()
        created = atom.apply(model, X, data, registry)
        assert created == 1

        bucket = data['penalties']['regen_away_club_home_weekends_count']
        model.Minimize(sum(bucket['penalties']))

        solver, status = _solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )
        dev_total = _total_dev(solver, bucket)
        assert dev_total == 2, f'Expected dev=2, got {dev_total}'

        normalised_weight = max(1, bucket['weight'] // len(bucket['penalties']))
        assert normalised_weight == 90000
        contribution = dev_total * normalised_weight
        assert contribution == 180000

    def test_given_weight_zero_then_no_op(self):
        """weight == 0 -> atom returns 0 and creates no penalty bucket."""
        data = _build_one_club_fixture(num_weeks=4)
        data['penalty_weights'] = {'regen_away_club_home_weekends_count': 0}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        registry = HelperVarRegistry(model)
        count = AwayClubHomeWeekendsCountRegenSoft().apply(model, X, data, registry)
        assert count == 0
        assert 'regen_away_club_home_weekends_count' not in data.get('penalties', {})
