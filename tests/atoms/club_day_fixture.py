"""Small fixture exercising the 5 ClubDay atoms.

One host club ("Tigers") has 2 teams in 3rd grade so derbies are well-defined,
plus 1 team in 4th. An opponent club ("Wests") has matching teams. A neutral
club ("Norths") provides extra opponents on the club day. All games happen at
Broadmeadow on a single fixed date with multiple fields and slots so the
SameField + ContiguousSlots atoms have something to bite on.

Team names use the production convention `<slug> <grade>` so
`name.rsplit(' ', 1)[1]` returns the grade.
"""
from __future__ import annotations

from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from constraints.atoms.base import BROADMEADOW
from models import Club, Grade, PlayingField, Team, Timeslot


CLUB_DAY_DATE = datetime(2026, 6, 14)
CLUB_DAY_DATE_STR = '2026-06-14'
OTHER_DATE_STR = '2026-06-21'


def build_club_day_fixture(*, opponent: str | None = None) -> Dict:
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    sf = PlayingField(location=BROADMEADOW, name='SF')
    fields = [ef, wf, sf]

    clubs = [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
    ]

    teams = [
        Team(name='Tigers-1 3rd', club=clubs[0], grade='3rd'),
        Team(name='Tigers-2 3rd', club=clubs[0], grade='3rd'),
        Team(name='Tigers 4th', club=clubs[0], grade='4th'),
        Team(name='Wests-1 3rd', club=clubs[1], grade='3rd'),
        Team(name='Wests-2 3rd', club=clubs[1], grade='3rd'),
        Team(name='Wests 4th', club=clubs[1], grade='4th'),
        Team(name='Norths 3rd', club=clubs[2], grade='3rd'),
        Team(name='Norths 4th', club=clubs[2], grade='4th'),
    ]

    grades = [
        Grade(name='3rd', teams=[t.name for t in teams if t.grade == '3rd']),
        Grade(name='4th', teams=[t.name for t in teams if t.grade == '4th']),
    ]

    games: List[Tuple[str, str, str]] = []
    for grade in grades:
        for t1, t2 in combinations(grade.teams, 2):
            t1, t2 = sorted((t1, t2))
            games.append((t1, t2, grade.name))

    timeslots: List[Timeslot] = []
    times = ['11:30', '13:00', '14:30', '16:00']
    for date_str, week_no in [(CLUB_DAY_DATE_STR, 10), (OTHER_DATE_STR, 11)]:
        for field in fields:
            for slot, time in enumerate(times, 1):
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time=time, week=week_no,
                    day_slot=slot, field=field, round_no=week_no,
                ))

    if opponent is None:
        club_days = {'Tigers': CLUB_DAY_DATE}
    else:
        club_days = {'Tigers': {'date': CLUB_DAY_DATE, 'opponent': opponent}}

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'3rd': 2, '4th': 2, 'max': 2},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': club_days,
        'preference_no_play': {},
        'home_field_map': {},
        'constraint_defaults': {},
    }


def build_model_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    """Build CP-SAT model with one var for every (game, timeslot) pair."""
    model = cp_model.CpModel()
    X = {}
    for (t1, t2, grade) in data['games']:
        for t in data['timeslots']:
            if not t.day:
                continue
            key = (t1, t2, grade, t.day, t.day_slot, t.time,
                   t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}'
            )
    return model, X


def solve_with_timeout(model: cp_model.CpModel, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    return status, solver
