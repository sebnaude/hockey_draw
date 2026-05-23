"""GWT tests for ClubNoConcurrentSlot (spec-021), DoD 6b hand oracles.

No mocks: real cp_model, real registry, hand-built X (11-tuples) + real Team/Club
models. The cap is capacity-aware: cap = max(1, ceil(club_team_count / slots)).
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import ClubNoConcurrentSlot
from constraints.helper_vars import HelperVarRegistry
from models import Club, Team

LOC = 'Central Coast Hockey Park'  # 2 timeslots in 2026 config
DATE = '2026-03-22'


def _key(t1, t2, grade, slot):
    return (t1, t2, grade, 'Sunday', slot, '12:00', 1, DATE, 1, 'Wyong Main Field', LOC)


def _data(home_teams, slots=2):
    """Club 'G' fields `home_teams` (a list of grades). Each opponent is its own
    1-team club so only G's cap binds."""
    g = Club(name='G', home_field=LOC)
    teams = [Team(name=f'G {gr}', club=g, grade=gr) for gr in home_teams]
    for i, gr in enumerate(home_teams):
        oc = Club(name=f'O{i}', home_field=LOC)
        teams.append(Team(name=f'O{i} {gr}', club=oc, grade=gr))
    return {
        'teams': teams,
        'locked_weeks': set(),
        'no_field_slots': {LOC: slots},
    }


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    return solver.Solve(model)


def _apply(model, X, data):
    return ClubNoConcurrentSlot().apply(model, X, data, HelperVarRegistry(model))


class TestClubNoConcurrentSlot:
    def test_two_teams_two_slot_venue_cap_one(self):
        # Hand oracle: club with 2 games at a 2-time venue -> cap ceil(2/2)=1.
        # Both games in one slot -> sum 2 > 1 -> INFEASIBLE.
        data = _data(['PHL', '2nd'])
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        _apply(model, X, data)
        # Force both G games into slot 1.
        model.Add(X[_key('G PHL', 'O0 PHL', 'PHL', 1)] == 1)
        model.Add(X[_key('G 2nd', 'O1 2nd', '2nd', 1)] == 1)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_three_teams_two_slot_venue_cap_two(self):
        # Hand oracle: club with 3 games at a 2-time venue -> cap ceil(3/2)=2.
        # Two in one slot is FEASIBLE; three in one slot is INFEASIBLE.
        grades = ['PHL', '2nd', '3rd']
        data = _data(grades)

        # FEASIBLE: 2 in slot 1, 1 in slot 2.
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(grades) for s in (1, 2)}
        _apply(model, X, data)
        model.Add(X[_key('G PHL', 'O0 PHL', 'PHL', 1)] == 1)
        model.Add(X[_key('G 2nd', 'O1 2nd', '2nd', 1)] == 1)
        model.Add(X[_key('G 3rd', 'O2 3rd', '3rd', 2)] == 1)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # INFEASIBLE: all three in slot 1 (3 > cap 2).
        model2 = cp_model.CpModel()
        X2 = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model2.NewBoolVar(f'g{i}s{s}')
              for i, gr in enumerate(grades) for s in (1, 2)}
        _apply(model2, X2, data)
        for i, gr in enumerate(grades):
            model2.Add(X2[_key(f'G {gr}', f'O{i} {gr}', gr, 1)] == 1)
        assert _solve(model2) == cp_model.INFEASIBLE

    def test_locked_week_skipped(self):
        data = _data(['PHL', '2nd'])
        data['locked_weeks'] = {1}
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        assert _apply(model, X, data) == 0


class TestComputeNoFieldSlots:
    def test_derives_distinct_time_count_per_location(self):
        # Hand oracle: max distinct times across each location's days.
        from config.defaults import compute_no_field_slots
        dtm = {
            'NIHC': {'Sunday': ['8:30', '10:00', '11:30'], 'Friday': ['19:00']},
            'Maitland': {'Sunday': ['9:00', '10:30']},
            'Empty': {},
        }
        out = compute_no_field_slots(dtm)
        assert out == {'NIHC': 3, 'Maitland': 2, 'Empty': 0}

    def test_real_2026_config(self):
        from config import load_season_data
        nfs = load_season_data(2026)['no_field_slots']
        assert nfs['Newcastle International Hockey Centre'] == 8
        assert nfs['Maitland Park'] == 6
        assert nfs['Central Coast Hockey Park'] == 2
