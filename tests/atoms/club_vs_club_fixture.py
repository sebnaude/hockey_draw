"""Small fixture exercising the 4 ClubVsClubAlignment atoms.

Two clubs with mixed grades. Per-team games:
  3rd (4 teams, R=4): 4 // (4-1) = 1
  4th (2 teams, R=4): 4 // (2-1) = 4
  PHL (2 teams, R=4): 4
  2nd (2 teams, R=4): 4

So the lower-grade alignment block triggers on the (3rd, 4th) pair and the
PHL/2nd back-to-back block triggers on the (PHL, 2nd) pair. R=4 also gives
the round walk enough room to exercise field-limit and deficit penalties.

Multiple Sunday slots on EF + WF give both atom families the field /
day-slot variety they need.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from constraints.atoms.base import BROADMEADOW
from models import Club, Grade, PlayingField, Team, Timeslot


def build_cvc_fixture() -> Dict:
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    fields = [ef, wf]

    clubs = [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
    ]

    teams = [
        Team(name='Tigers-1 3rd', club=clubs[0], grade='3rd'),
        Team(name='Tigers-2 3rd', club=clubs[0], grade='3rd'),
        Team(name='Wests-1 3rd', club=clubs[1], grade='3rd'),
        Team(name='Wests-2 3rd', club=clubs[1], grade='3rd'),
        Team(name='Tigers 4th', club=clubs[0], grade='4th'),
        Team(name='Wests 4th', club=clubs[1], grade='4th'),
        Team(name='Tigers PHL', club=clubs[0], grade='PHL'),
        Team(name='Wests PHL', club=clubs[1], grade='PHL'),
        Team(name='Tigers 2nd', club=clubs[0], grade='2nd'),
        Team(name='Wests 2nd', club=clubs[1], grade='2nd'),
    ]

    grades = [
        Grade(name='PHL', teams=[t.name for t in teams if t.grade == 'PHL']),
        Grade(name='2nd', teams=[t.name for t in teams if t.grade == '2nd']),
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
    sunday_dates = [
        (1, '2026-03-22'), (2, '2026-03-29'),
        (3, '2026-04-05'), (4, '2026-04-12'),
    ]
    for week, date_str in sunday_dates:
        for field in fields:
            for slot, time in enumerate(times, 1):
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time=time, week=week,
                    day_slot=slot, field=field, round_no=week,
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
        'num_rounds': {'PHL': 4, '2nd': 4, '3rd': 4, '4th': 4, 'max': 4},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {},
        'constraint_defaults': {},
        'penalties': {},
    }


def build_model_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
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
