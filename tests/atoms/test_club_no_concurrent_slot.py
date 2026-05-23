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


def _pin(model, X, assignment):
    """Fully pin every var: each (grade -> chosen slot) plays in that slot only."""
    for key, v in X.items():
        grade = key[2]
        chosen = assignment.get(grade)
        model.Add(v == (1 if key[4] == chosen else 0))


class TestClubNoConcurrentSlot:
    def test_two_teams_two_slot_venue_cap_one(self):
        # Hand oracle: 2 games at a 2-time venue -> n_loc=2, S=2 -> cap ceil(2/2)=1.
        # Both games in slot 1 -> 2 > 1 -> INFEASIBLE. (n_loc is pinned to 2.)
        data = _data(['PHL', '2nd'])
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        _apply(model, X, data)
        _pin(model, X, {'PHL': 1, '2nd': 1})  # both in slot 1
        assert _solve(model) == cp_model.INFEASIBLE

    def test_three_teams_two_slot_venue_cap_two(self):
        # Hand oracle: 3 games at a 2-time venue -> n_loc=3, S=2 -> cap ceil(3/2)=2.
        grades = ['PHL', '2nd', '3rd']
        data = _data(grades)

        # FEASIBLE: 2 in slot 1, 1 in slot 2 (n_loc=3 pinned).
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(grades) for s in (1, 2)}
        _apply(model, X, data)
        _pin(model, X, {'PHL': 1, '2nd': 1, '3rd': 2})
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # INFEASIBLE: all three in slot 1 (n_loc=3, cap 2, slot1=3 > 2).
        model2 = cp_model.CpModel()
        X2 = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model2.NewBoolVar(f'g{i}s{s}')
              for i, gr in enumerate(grades) for s in (1, 2)}
        _apply(model2, X2, data)
        _pin(model2, X2, {'PHL': 1, '2nd': 1, '3rd': 1})
        assert _solve(model2) == cp_model.INFEASIBLE

    def test_more_teams_than_games_at_venue_still_cap_one(self):
        # SF-1 regression: a club with MANY teams but only 2 games actually at the
        # 2-slot venue must still get cap 1 (n_loc=2, not team_count). Both games in
        # one slot -> INFEASIBLE. (Old n_teams-based cap would have been ceil(5/2)=3.)
        data = _data(['PHL', '2nd', '3rd', '4th', '5th'])  # 5 club teams
        model = cp_model.CpModel()
        # Only PHL and 2nd have candidate games at the venue (2 games), in 2 slots.
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        _apply(model, X, data)
        _pin(model, X, {'PHL': 1, '2nd': 1})  # both at the venue in slot 1
        assert _solve(model) == cp_model.INFEASIBLE

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
