"""Tests for AwayClubHomeWeekendsCountRegenSoft atom (spec-027 regen-soft).

Real CP-SAT models, no mocks. GWT style with hand-computed oracles.

The SOFT atom emits, per away-based club, three absolute-deviation penalty
IntVars (friday / sunday / total home-weekend count vs target). The minimising
objective drives each dev to ``|actual_sum - target|``. We force the solver to
land the home-weekend counts at known actuals, then assert the summed penalty
equals the hand-computed total deviation.

Fixture is copied from ``test_away_club_home_weekends_count.py``: a single
Maitland-vs-Norths PHL pair across N weeks, with Friday + Sunday slots at both
Broadmeadow (away for Maitland) and Maitland Park (home for Maitland).
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
    away_club_required_sundays,
    away_club_total_weekends,
    phl_forced_friday_count,
)
from constraints.atoms.away_club_home_weekends_count_regen_soft import (
    AwayClubHomeWeekendsCountRegenSoft,
)
from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Fixture builder (mirrors the hard-atom test).
# ---------------------------------------------------------------------------


def _build_fixture(*, num_weeks: int, phl_required: int) -> Dict:
    bm_ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Main')

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland PHL', club=clubs[0], grade='PHL'),
        Team(name='Norths PHL', club=clubs[1], grade='PHL'),
    ]
    grade_objs = [Grade(name='PHL', teams=[t.name for t in teams])]
    num_rounds = {'PHL': phl_required, 'max': phl_required}

    games: List[Tuple[str, str, str]] = []
    for grade in grade_objs:
        for i, t1 in enumerate(grade.teams):
            for t2 in grade.teams[i + 1:]:
                games.append((t1, t2, grade.name))

    timeslots: List[Timeslot] = []
    base_date_n = 22  # 2026-03-22 = week 1 Sunday
    for week in range(1, num_weeks + 1):
        sun_d = f'2026-03-{base_date_n + 7 * (week - 1):02d}'
        fri_d = f'2026-03-{base_date_n + 7 * (week - 1) - 2:02d}'
        timeslots.append(Timeslot(date=sun_d, day='Sunday', time='11:30',
                                  week=week, day_slot=1, field=bm_ef, round_no=week))
        timeslots.append(Timeslot(date=sun_d, day='Sunday', time='11:30',
                                  week=week, day_slot=1, field=mp, round_no=week))
        timeslots.append(Timeslot(date=fri_d, day='Friday', time='19:00',
                                  week=week, day_slot=1, field=bm_ef, round_no=week))
        timeslots.append(Timeslot(date=fri_d, day='Friday', time='19:00',
                                  week=week, day_slot=1, field=mp, round_no=week))

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
            key = (t1, t2, grade, ts.day, ts.day_slot, ts.time,
                   ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{ts.week}_{ts.day}_{ts.field.name}')
    return X


def _add_basic_hard_constraints(model: cp_model.CpModel, X: Dict, data: Dict) -> None:
    """Each pair plays exactly num_rounds[grade]; at most one game per
    (week, day, slot, field, location)."""
    from collections import defaultdict
    pair_vars = defaultdict(list)
    field_slot_vars = defaultdict(list)
    for key, var in X.items():
        t1, t2, grade = key[0], key[1], key[2]
        pair_vars[(t1, t2, grade)].append(var)
        field_slot_vars[(key[6], key[3], key[4], key[9], key[10])].append(var)
    num_rounds = data['num_rounds']
    for (t1, t2, grade), vars_list in pair_vars.items():
        model.Add(sum(vars_list) == num_rounds[grade])
    for vars_list in field_slot_vars.values():
        model.Add(sum(vars_list) <= 1)


def _mp_vars(X: Dict, day: str) -> List:
    """Maitland Park vars for a given day (one slot per week)."""
    return [v for k, v in X.items() if k[10] == MAITLAND and k[3] == day]


def _total_penalty(solver, bucket) -> int:
    return sum(solver.Value(p) for p in bucket['penalties'])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAwayClubHomeWeekendsCountRegenSoft:

    def test_given_counts_off_target_then_penalty_equals_deviation(self):
        """Scenario 1 (violation): force home-weekend counts off-target by D.

        Fixture: Maitland-vs-Norths PHL, 5 games required, 10 weeks, no FORCED
        Fridays.

        Targets (via the shared helper, same as the hard atom):
          - phl_forced_friday_count(Maitland)   = 0   -> friday_target  = 0
          - away_club_required_sundays(Maitland)= 5   -> sunday_target  = 5
          - away_club_total_weekends(Maitland)  = 5   -> total_target   = 5

        Force the actuals OFF target:
          - 0 Maitland-Park Friday games  -> friday actual = 0  (dev |0-0| = 0)
          - 6 Maitland-Park Sunday games  -> sunday actual = 6  (dev |6-5| = 1)
          Since each week has exactly one MP Sunday slot, 6 MP-Sunday games
          land in 6 DISTINCT weeks, and there are 0 MP Friday games, so the
          any-day total = 6 distinct weeks (dev |6-5| = 1).

        Hand oracle: D = 0 (fri) + 1 (sun) + 1 (total) = 2.
        """
        # The helper targets are computed from phl_required = 5
        # (fri0 / sun5 / total5). To force the ACTUAL counts off target we
        # decouple from the pair-sum: we do NOT add the pair == 5 constraint,
        # only the field/slot single-booking rule, then pin 6 MP-Sunday games.
        data = _build_fixture(num_weeks=10, phl_required=5)

        # Confirm helper targets (the atom uses these exactly).
        assert phl_forced_friday_count(data, 'Maitland') == 0
        assert away_club_required_sundays(data, 'Maitland') == 5
        assert away_club_total_weekends(data, 'Maitland') == 5

        model = cp_model.CpModel()
        X = _build_X(model, data)

        # Field/slot single-booking (no double game in same slot/field/week).
        from collections import defaultdict
        field_slot_vars = defaultdict(list)
        for key, var in X.items():
            field_slot_vars[(key[6], key[3], key[4], key[9], key[10])].append(var)
        for vars_list in field_slot_vars.values():
            model.Add(sum(vars_list) <= 1)

        # Force EXACTLY 6 Maitland-Park Sunday games (each in a distinct week)
        # and 0 Maitland-Park Friday games. This pins the actual home-weekend
        # counts: friday=0, sunday=6, total=6.
        mp_sun = _mp_vars(X, 'Sunday')
        mp_fri = _mp_vars(X, 'Friday')
        model.Add(sum(mp_sun) == 6)
        model.Add(sum(mp_fri) == 0)

        registry = HelperVarRegistry(model)
        atom = AwayClubHomeWeekendsCountRegenSoft()
        created = atom.apply(model, X, data, registry)
        assert created == 3, f'Expected 3 penalty vars (fri/sun/total), got {created}'

        bucket = data['penalties']['regen_away_club_home_weekends_count']
        # Minimise total penalty so each dev settles to |actual - target|.
        model.Minimize(sum(bucket['penalties']))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        status = solver.Solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )

        total = _total_penalty(solver, bucket)
        # Hand oracle: |0-0| + |6-5| + |6-5| = 0 + 1 + 1 = 2.
        assert total == 2, f'Expected total penalty 2 (D), got {total}'

    def test_given_counts_on_target_then_zero_penalty(self):
        """Scenario 2 (clean): force counts exactly on target -> penalty 0.

        Fixture: Maitland-vs-Norths PHL, 5 games required, 10 weeks, no FORCED
        Fridays. Targets: friday=0, sunday=5, total=5.

        Force exactly 5 Maitland-Park Sunday games (5 distinct weeks) and 0 MP
        Friday games -> actual friday=0, sunday=5, total=5, all ON target.

        Hand oracle: D = |0-0| + |5-5| + |5-5| = 0.
        """
        data = _build_fixture(num_weeks=10, phl_required=5)
        assert phl_forced_friday_count(data, 'Maitland') == 0
        assert away_club_required_sundays(data, 'Maitland') == 5
        assert away_club_total_weekends(data, 'Maitland') == 5

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)

        # Pin exactly 5 MP Sundays, 0 MP Fridays -> all counts on target.
        mp_sun = _mp_vars(X, 'Sunday')
        mp_fri = _mp_vars(X, 'Friday')
        model.Add(sum(mp_sun) == 5)
        model.Add(sum(mp_fri) == 0)

        registry = HelperVarRegistry(model)
        atom = AwayClubHomeWeekendsCountRegenSoft()
        created = atom.apply(model, X, data, registry)
        assert created == 3

        bucket = data['penalties']['regen_away_club_home_weekends_count']
        model.Minimize(sum(bucket['penalties']))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        status = solver.Solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )

        total = _total_penalty(solver, bucket)
        assert total == 0, f'Expected zero penalty (on target), got {total}'

    def test_given_weight_zero_then_no_op(self):
        """weight == 0 -> atom returns 0 and creates no penalty bucket."""
        data = _build_fixture(num_weeks=4, phl_required=3)
        data['penalty_weights'] = {'regen_away_club_home_weekends_count': 0}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        registry = HelperVarRegistry(model)
        count = AwayClubHomeWeekendsCountRegenSoft().apply(model, X, data, registry)
        assert count == 0
        assert 'regen_away_club_home_weekends_count' not in data.get('penalties', {})
