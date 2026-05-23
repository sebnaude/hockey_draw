"""Tests for AwayClubHomeWeekendsCount atom (spec-004).

Real CP-SAT models, no mocks. Each test:
  - constructs a tiny scenario (games, timeslots, FORCED_GAMES);
  - hand-computes the expected (friday, sunday, total) home-weekend counts;
  - applies the atom AND a minimum-set of supporting hard constraints so the
    solver has a feasible search space;
  - asserts the solver finds a feasible model AND that the home-weekend counts
    match the hand-computed oracles.
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

from constraints.atoms._phl_forced_friday_helper import (
    away_club_required_sundays,
    away_club_total_weekends,
    phl_forced_friday_count,
)
from constraints.atoms.away_club_home_weekends_count import (
    AwayClubHomeWeekendsCount,
)
from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Fixture builder — single Maitland-vs-Norths PHL pair, multi-week, with
# Friday + Sunday slots at both BM (away) and Maitland Park (home).
# ---------------------------------------------------------------------------


def _build_fixture(
    *,
    num_weeks: int,
    phl_required: int,
    other_grade_required: int = 0,
    forced_games: List[Dict] = None,
) -> Dict:
    """Build a fixture with `num_weeks` playable weeks.

    Each week has:
      - one Sunday slot at Maitland Park (home for Maitland).
      - one Sunday slot at Broadmeadow (away for Maitland).
      - one Friday slot at Maitland Park (home for Maitland).
      - one Friday slot at Broadmeadow (away for Maitland).
    """
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
    num_rounds = {'PHL': phl_required, 'max': max(phl_required, other_grade_required)}

    if other_grade_required > 0:
        teams.extend([
            Team(name='Maitland 3rd', club=clubs[0], grade='3rd'),
            Team(name='Norths 3rd', club=clubs[1], grade='3rd'),
        ])
        grade_objs.append(Grade(name='3rd', teams=['Maitland 3rd', 'Norths 3rd']))
        num_rounds['3rd'] = other_grade_required

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
        # Sunday: BM and Maitland Park, slot 1.
        timeslots.append(Timeslot(
            date=sun_d, day='Sunday', time='11:30', week=week,
            day_slot=1, field=bm_ef, round_no=week,
        ))
        timeslots.append(Timeslot(
            date=sun_d, day='Sunday', time='11:30', week=week,
            day_slot=1, field=mp, round_no=week,
        ))
        # Friday: BM and Maitland Park, slot 1.
        timeslots.append(Timeslot(
            date=fri_d, day='Friday', time='19:00', week=week,
            day_slot=1, field=bm_ef, round_no=week,
        ))
        timeslots.append(Timeslot(
            date=fri_d, day='Friday', time='19:00', week=week,
            day_slot=1, field=mp, round_no=week,
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
        'forced_games': forced_games or [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
    }


def _build_X(model: cp_model.CpModel, data: Dict) -> Dict:
    """Build X vars for every (game, timeslot)."""
    X = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{ts.week}_{ts.day}_{ts.field.name}'
            )
    return X


def _solve_and_count(
    model: cp_model.CpModel, X: Dict, data: Dict, club: str,
) -> Tuple[int, int, int]:
    """Solve and return (friday_home_weekends, sunday_home_weekends, total_home_weekends).

    Counts WEEKS (not games) — multiple home games in the same week count once.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
        f'Expected feasible solve, got {solver.status_name(status)}'
    )

    home_venue = data['home_field_map'][club]
    team_to_club = {t.name: t.club.name for t in data['teams']}
    fri_weeks = set()
    sun_weeks = set()
    all_weeks = set()
    for key, var in X.items():
        if solver.Value(var) != 1:
            continue
        if key[10] != home_venue:
            continue
        t1, t2 = key[0], key[1]
        if team_to_club.get(t1) != club and team_to_club.get(t2) != club:
            continue
        week = key[6]
        day = key[3]
        if day == 'Friday':
            fri_weeks.add(week)
        elif day == 'Sunday':
            sun_weeks.add(week)
        all_weeks.add(week)
    return len(fri_weeks), len(sun_weeks), len(all_weeks)


def _add_basic_hard_constraints(model: cp_model.CpModel, X: Dict, data: Dict) -> None:
    """Per-pair: each pair plays exactly num_rounds[grade] games across the season.
    Per-(field, week, slot): at most one game.
    """
    # Pair sums
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


# ---------------------------------------------------------------------------
# Scenario 1: spec DoD criterion — 2 FORCED Fridays + 8 PHL + 6 other grade.
# Expected home weekends == 8 (2 Friday + 6 Sunday, distinct weeks).
# ---------------------------------------------------------------------------


class TestAwayClubHomeWeekendsCount:

    def test_given_two_forced_fridays_phl_eight_other_six_then_total_eight(self):
        """Spec-004 DoD scenario.

        Given: Maitland PHL plays 8 games, 3rd grade plays 6 games, 2 FORCED
              PHL Fridays at Maitland Park.
        When: solving with AwayClubHomeWeekendsCount only.
        Then: helper computes:
              - phl_forced_friday_count == 2
              - away_club_required_sundays == max(8-2, 6) == 6
              - away_club_total_weekends == max(8, 6) == 8
              And the solver finds: 2 Friday home + 6 Sunday home in DISTINCT
              weeks (total = 8).
        """
        data = _build_fixture(
            num_weeks=10, phl_required=8, other_grade_required=6,
            forced_games=[
                {'grade': 'PHL', 'day': 'Friday',
                 'field_location': MAITLAND,
                 'count': 2, 'constraint': 'equal',
                 'description': 'Maitland Park Fridays = 2'},
            ],
        )
        # Verify helper outputs.
        assert phl_forced_friday_count(data, 'Maitland') == 2
        assert away_club_required_sundays(data, 'Maitland') == 6
        assert away_club_total_weekends(data, 'Maitland') == 8

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)
        # Apply the FORCED entry as a `sum == 2` over matching vars (so the
        # atom's `friday_target` aligns with the solver's count).
        from collections import defaultdict
        forced_vars = []
        for key, var in X.items():
            if (key[3] == 'Friday' and key[2] == 'PHL'
                    and key[10] == MAITLAND):
                forced_vars.append(var)
        model.Add(sum(forced_vars) == 2)

        registry = HelperVarRegistry(model)
        atom = AwayClubHomeWeekendsCount()
        atom.apply(model, X, data, registry)

        fri, sun, total = _solve_and_count(model, X, data, 'Maitland')
        # Hand-computed: 2, 6, 8.
        assert fri == 2, f'Expected 2 Friday home weeks, got {fri}'
        assert sun == 6, f'Expected 6 Sunday home weeks, got {sun}'
        assert total == 8, f'Expected 8 total home weeks, got {total}'

    def test_given_no_forced_fridays_then_all_sundays(self):
        """Given no FORCED Fridays, Then friday==0, sunday==total==phl_required.

        Maitland plays 5 PHL games, no other grade. helper returns:
          forced_fri=0, sundays=5, total=5.
        Solver places all 5 Maitland home games on Sundays at Maitland Park.
        """
        data = _build_fixture(num_weeks=8, phl_required=5)
        assert phl_forced_friday_count(data, 'Maitland') == 0
        assert away_club_required_sundays(data, 'Maitland') == 5
        assert away_club_total_weekends(data, 'Maitland') == 5

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)

        registry = HelperVarRegistry(model)
        AwayClubHomeWeekendsCount().apply(model, X, data, registry)

        fri, sun, total = _solve_and_count(model, X, data, 'Maitland')
        assert (fri, sun, total) == (0, 5, 5), (
            f'Expected (0, 5, 5); got ({fri}, {sun}, {total})'
        )

    def test_given_phl_only_three_fridays_then_two_sundays(self):
        """PHL-only club; 3 FORCED Fridays out of 5 required → 2 Sundays.

        Hand-computed: PHL=5, forced=3, no other grade.
          forced_fri=3, sundays = max(5-3, 0) = 2, total = max(5, 0) = 5.
        """
        data = _build_fixture(
            num_weeks=10, phl_required=5,
            forced_games=[
                {'grade': 'PHL', 'day': 'Friday',
                 'field_location': MAITLAND,
                 'count': 3, 'constraint': 'equal',
                 'description': 'Maitland Park Fridays = 3'},
            ],
        )
        assert phl_forced_friday_count(data, 'Maitland') == 3
        assert away_club_required_sundays(data, 'Maitland') == 2
        assert away_club_total_weekends(data, 'Maitland') == 5

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)
        # Apply FORCED count constraint.
        forced_vars = [
            v for k, v in X.items()
            if k[3] == 'Friday' and k[2] == 'PHL' and k[10] == MAITLAND
        ]
        model.Add(sum(forced_vars) == 3)

        registry = HelperVarRegistry(model)
        AwayClubHomeWeekendsCount().apply(model, X, data, registry)

        fri, sun, total = _solve_and_count(model, X, data, 'Maitland')
        assert (fri, sun, total) == (3, 2, 5), (
            f'Expected (3, 2, 5); got ({fri}, {sun}, {total})'
        )

    def test_given_phl18_other20_two_forced_fridays_then_total_twenty(self):
        """Spec-004 Clarification edge case: PHL=18, 3rd=20, 2 FORCED Fridays.

        Hand-computed:
          phl_forced_friday_count = 2
          away_club_required_sundays = max(18 - 2, 20) = 20  (driven by 3rd)
          away_club_total_weekends   = max(18, 20) = 20
        Solver places 20 home weekends (all with Sunday); 2 of those weeks
        ALSO have a Friday home game (PHL forced Friday overlaps with a 3rd
        grade Sunday in the same week, so total still = 20 distinct weeks).

        We need >= 20 playable weeks for this scenario to be feasible.
        """
        data = _build_fixture(
            num_weeks=22, phl_required=18, other_grade_required=20,
            forced_games=[
                {'grade': 'PHL', 'day': 'Friday',
                 'field_location': MAITLAND,
                 'count': 2, 'constraint': 'equal',
                 'description': 'Maitland Park Fridays = 2'},
            ],
        )
        assert phl_forced_friday_count(data, 'Maitland') == 2
        assert away_club_required_sundays(data, 'Maitland') == 20
        assert away_club_total_weekends(data, 'Maitland') == 20

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)
        # FORCED count for Friday Maitland PHL = 2.
        forced_vars = [
            v for k, v in X.items()
            if k[3] == 'Friday' and k[2] == 'PHL' and k[10] == MAITLAND
        ]
        model.Add(sum(forced_vars) == 2)

        registry = HelperVarRegistry(model)
        AwayClubHomeWeekendsCount().apply(model, X, data, registry)

        fri, sun, total = _solve_and_count(model, X, data, 'Maitland')
        # Hand-computed: Friday==2, Sunday==20, total==20 (Fridays absorbed).
        assert fri == 2, f'Expected 2 Friday home weeks, got {fri}'
        assert sun == 20, f'Expected 20 Sunday home weeks, got {sun}'
        assert total == 20, f'Expected 20 total home weeks, got {total}'

    def test_given_no_away_clubs_then_zero_constraints(self):
        """Empty home_field_map → atom is a no-op."""
        data = _build_fixture(num_weeks=4, phl_required=3)
        data['home_field_map'] = {}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        registry = HelperVarRegistry(model)
        count = AwayClubHomeWeekendsCount().apply(model, X, data, registry)
        assert count == 0
