"""Shared fixtures for atom tests.

Builds a small PHL+2nd grade fixture covering Broadmeadow + Maitland Park +
Central Coast Hockey Park venues, multiple weeks, Friday + Sunday slots — enough
to exercise every PHL atom. No mocks; real CP-SAT models throughout.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from constraints.atoms.base import BROADMEADOW, GOSFORD, MAITLAND
from models import Club, Grade, PlayingField, Team, Timeslot


def _build_phl_fixture() -> Dict:
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    mp = PlayingField(location=MAITLAND, name='Maitland Main Field')
    cc = PlayingField(location=GOSFORD, name='Wyong Main Field')
    fields = [ef, wf, mp, cc]

    clubs = [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Gosford', home_field=GOSFORD),
    ]

    teams = []
    for club in clubs:
        teams.append(Team(name=f'{club.name} PHL', club=club, grade='PHL'))
        teams.append(Team(name=f'{club.name} 2nd', club=club, grade='2nd'))

    grades = [
        Grade(name='PHL', teams=[t.name for t in teams if t.grade == 'PHL']),
        Grade(name='2nd', teams=[t.name for t in teams if t.grade == '2nd']),
    ]

    games: List[Tuple[str, str, str]] = []
    for grade in grades:
        for t1, t2 in combinations(grade.teams, 2):
            games.append((t1, t2, grade.name))

    timeslots: List[Timeslot] = []
    base_dates = [
        (1, '2026-03-22'),
        (2, '2026-03-29'),
        (3, '2026-04-05'),
        (4, '2026-04-12'),
        (5, '2026-04-19'),
    ]
    for round_no, date_str in base_dates:
        for field in (ef, wf):
            for slot, time in enumerate(['11:30', '13:00', '14:30', '16:00'], 1):
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time=time, week=round_no,
                    day_slot=slot, field=field, round_no=round_no,
                ))
        timeslots.append(Timeslot(
            date=date_str, day='Sunday', time='12:00', week=round_no,
            day_slot=1, field=cc, round_no=round_no,
        ))
        timeslots.append(Timeslot(
            date=date_str, day='Sunday', time='13:30', week=round_no,
            day_slot=2, field=cc, round_no=round_no,
        ))
        for slot, time in enumerate(['12:00', '13:30', '15:00', '16:30'], 1):
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time=time, week=round_no,
                day_slot=slot, field=mp, round_no=round_no,
            ))

    friday_dates = {
        1: '2026-03-20',
        2: '2026-03-27',
        3: '2026-04-03',
        4: '2026-04-10',
        5: '2026-04-17',
    }
    for round_no, fri_date in friday_dates.items():
        timeslots.append(Timeslot(
            date=fri_date, day='Friday', time='19:00', week=round_no,
            day_slot=1, field=ef, round_no=round_no,
        ))
        timeslots.append(Timeslot(
            date=fri_date, day='Friday', time='19:00', week=round_no,
            day_slot=1, field=mp, round_no=round_no,
        ))
        timeslots.append(Timeslot(
            date=fri_date, day='Friday', time='20:00', week=round_no,
            day_slot=1, field=cc, round_no=round_no,
        ))

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'PHL': 5, '2nd': 5, 'max': 5},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND, 'Gosford': GOSFORD},
        'constraint_defaults': {
            'max_friday_broadmeadow': 3,
            'gosford_friday_games': 2,
            'maitland_friday_games': 1,
            'gosford_friday_rounds': [2, 4],
        },
    }


@pytest.fixture
def phl_data():
    """Fresh PHL fixture per test (don't share — atoms mutate `data['penalties']`)."""
    return _build_phl_fixture()


def build_model_X(data: Dict, *, allow_phl: bool = True, allow_2nd: bool = True,
                  filter_keys=None) -> Tuple[cp_model.CpModel, Dict]:
    """Build a CP-SAT model with PHL/2nd vars matching `data`.

    Mirrors the production filtering: PHL keys at any (slot, location) only
    where 11-tuple matches a (game, timeslot). 2nd grade restricted to
    Sundays at Broadmeadow + Maitland (no Gosford, no Friday).

    `filter_keys` is an optional callable `key -> bool` for tests that need
    to drop specific vars (e.g. limiting Friday slots to Gosford-vs-Maitland).
    """
    model = cp_model.CpModel()
    X = {}
    for (t1, t2, grade) in data['games']:
        if grade == 'PHL' and not allow_phl:
            continue
        if grade == '2nd' and not allow_2nd:
            continue
        for t in data['timeslots']:
            if not t.day:
                continue
            location = t.field.location
            if grade == '2nd':
                if t.day == 'Friday':
                    continue
                if location == GOSFORD:
                    continue
            key = (t1, t2, grade, t.day, t.day_slot, t.time,
                   t.week, t.date, t.round_no, t.field.name, location)
            if filter_keys is not None and not filter_keys(key):
                continue
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}_{t.day}')
    return model, X


def solve_with_timeout(model: cp_model.CpModel, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    return status, solver
