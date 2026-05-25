"""spec-033 Unit D — ClubGameSpread hard field-concentration cap (<=2 fields +slack).

ClubGameSpread is now TWO interlocked structures:
  1. contiguity (<=1 hole/field for >=4-game fields, push->0 holes) — spec-021/024
  2. field concentration (HARD <=club_game_spread_max_fields fields per
     (club,week,day) +slack, soft push->1 field via off_primary) — THIS unit

These tests drive the real ``UnifiedConstraintEngine`` (no mocks/patches), with a
hand-computed oracle in every comment. ``data['constraint_defaults']`` is left
empty so the engine reads its default ``club_game_spread_max_fields == 2``.

Field-used channelling under test (the subtle part):
  field_used[f] is a BoolVar, and for every game var v on field f we add
  ``model.Add(field_used[f] >= v)``. Scheduling a game on f (v=1) forces
  field_used[f]=1. The cap ``sum(field_used) <= max_fields + slack`` then bounds
  how many distinct fields can carry a game -> the cap bites.
"""
from __future__ import annotations

from typing import List, Tuple

from ortools.sat.python import cp_model

from constraints.atoms.base import BROADMEADOW
from constraints.unified import UnifiedConstraintEngine
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars

GRADES = ['PHL', '2nd', '3rd', '4th', '5th', '6th']
SLOTS = list(range(1, 7))


def _fixture_nfields(field_names, n_club_teams, constraint_slack=None):
    """Club 'C' fields `n_club_teams` teams (distinct grades), each vs a 1-team
    opponent club. The given NIHC fields each offer 6 slots, week 1, so a club's
    games can be split across up to len(field_names) fields."""
    fields = [PlayingField(location=BROADMEADOW, name=fn) for fn in field_names]
    grades = GRADES[:n_club_teams]
    c = Club(name='C', home_field=BROADMEADOW)
    teams = [Team(name=f'C {g}', club=c, grade=g) for g in grades]
    opp_clubs = [Club(name=f'O{i}', home_field=BROADMEADOW) for i in range(n_club_teams)]
    teams += [Team(name=f'O{i} {g}', club=opp_clubs[i], grade=g)
              for i, g in enumerate(grades)]
    grade_objs = [Grade(name=g, teams=[f'C {g}', f'O{i} {g}'])
                  for i, g in enumerate(grades)]
    games: List[Tuple[str, str, str]] = [
        (f'C {g}', f'O{i} {g}', g) for i, g in enumerate(grades)
    ]
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time=f'{8 + s}:00',
                 week=1, day_slot=s, field=fld, round_no=1)
        for fld in fields for s in SLOTS
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': grade_objs, 'clubs': [c] + opp_clubs, 'fields': fields,
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {g: 1 for g in grades},
        'constraint_slack': constraint_slack or {},
        'penalty_weights': {}, 'penalties': {}, 'forced_games': [],
        'blocked_games': [], 'team_conflicts': [], 'phl_preferences': {},
        'club_days': {}, 'preference_no_play': {}, 'home_field_map': {},
        'constraint_defaults': {},
    }
    return model, X, data


def _engine(model, X, data):
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    return eng


def _pin_field(model, X, t1, t2, field_name, slot):
    for k, v in X.items():
        if {k[0], k[1]} == {t1, t2}:
            model.Add(v == (1 if (k[4] == slot and k[9] == field_name) else 0))


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    return solver.Solve(model)


class TestClubGameSpreadFieldCap:
    def test_three_fields_infeasible_at_slack0(self):
        # 3 games, one on EF, one on WF, one on SF -> 3 distinct fields.
        # cap = max_fields(2) + slack(0) = 2. 3 > 2 -> INFEASIBLE.
        model, X, data = _fixture_nfields(['EF', 'WF', 'SF'], 3)
        _engine(model, X, data)._club_game_spread_hard()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'WF', 1)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'SF', 1)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_two_fields_feasible_with_off_primary_penalty(self):
        # 3 games: EF holds 2 ({1,2}), WF holds 1 ({1}). 2 distinct fields == cap.
        # FEASIBLE. off_primary = total(3) - max_field_count(2) = 1; holes = 0
        # (EF {1,2} contiguous). ClubGameSpread penalty bucket sums to exactly 1.
        model, X, data = _fixture_nfields(['EF', 'WF', 'SF'], 3)
        eng = _engine(model, X, data)
        eng._club_game_spread_hard()
        eng._club_game_spread_soft()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'WF', 1)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        penalties = data['penalties']['ClubGameSpread']['penalties']
        total = sum(solver.Value(p) for p in penalties)
        assert total == 1, f"off_primary oracle = 1, got {total}"

    def test_one_field_zero_off_primary(self):
        # 3 games all on EF, contiguous {1,2,3}. 1 distinct field <= cap 2.
        # off_primary = total(3) - max_field_count(3) = 0; holes = 0 -> sum 0.
        model, X, data = _fixture_nfields(['EF', 'WF', 'SF'], 3)
        eng = _engine(model, X, data)
        eng._club_game_spread_hard()
        eng._club_game_spread_soft()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'EF', 3)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        penalties = data['penalties']['ClubGameSpread']['penalties']
        total = sum(solver.Value(p) for p in penalties)
        assert total == 0, f"single-field penalty oracle = 0, got {total}"

    def test_three_fields_feasible_at_slack1(self):
        # Same 3-distinct-field layout as the infeasible case, but slack 1.
        # cap = max_fields(2) + slack(1) = 3. 3 <= 3 -> FEASIBLE.
        model, X, data = _fixture_nfields(
            ['EF', 'WF', 'SF'], 3, constraint_slack={'ClubGameSpread': 1})
        _engine(model, X, data)._club_game_spread_hard()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'WF', 1)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'SF', 1)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_field_used_channel_makes_cap_bite(self):
        # Direct evidence the channel forces field_used up: pin all 3 games onto
        # 3 distinct fields at slack 0 -> the only way to satisfy the cap would be
        # to set a field_used=0 for an occupied field, which Add(fu >= v) forbids
        # -> INFEASIBLE. (If the channel were Add(fu <= sum(v)) only, the solver
        # could leave fu=0 with a game scheduled and dodge the cap.)
        model, X, data = _fixture_nfields(['EF', 'WF', 'SF'], 3)
        _engine(model, X, data)._club_game_spread_hard()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 3)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'WF', 4)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'SF', 5)
        assert _solve(model) == cp_model.INFEASIBLE
