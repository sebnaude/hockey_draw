"""spec-021 — ClubGameSpread games-derived hole cap + DoD 9 (no IntVars) + DoD 12 guard.

ClubGameSpread is an engine method, so we drive it through the real
`UnifiedConstraintEngine` (no mocks). Hand oracle:
  holes = (max_used - min_used + 1) - num_distinct_used_slots
  gap_cap = max(0, min(1, n_games - 3))   (<=3 games -> 0 holes; >=4 -> 1 hole)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from config.defaults import DEFAULT_STAGES
from constraints.atoms.base import BROADMEADOW
from constraints.stages import ALL_ENGINE_KEYS, ENGINE_HARD_KEYS, atom_to_engine_key
from constraints.unified import UnifiedConstraintEngine
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars

GRADES = ['PHL', '2nd', '3rd', '4th']
SLOTS = list(range(1, 7))  # 6 NIHC slots in one week


def _fixture(n_club_teams: int):
    """Club 'C' fields `n_club_teams` teams (distinct grades); each plays one
    opponent from its own 1-team club. One NIHC field, 6 slots, week 1."""
    field = PlayingField(location=BROADMEADOW, name='EF')
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
                 week=1, day_slot=s, field=field, round_no=1)
        for s in SLOTS
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': grade_objs, 'clubs': [c] + opp_clubs, 'fields': [field],
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {g: 1 for g in grades}, 'constraint_slack': {},
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


def _pin(model, X, t1, t2, slot):
    """Force matchup (t1,t2) to play in `slot` (and nowhere else)."""
    for k, v in X.items():
        if {k[0], k[1]} == {t1, t2}:
            model.Add(v == (1 if k[4] == slot else 0))


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    return solver.Solve(model)


class TestClubSpreadHoleCap:
    def test_three_games_no_hole_feasible(self):
        # n=3 -> gap_cap 0. Used {1,2,3} -> 0 holes -> FEASIBLE.
        model, X, data = _fixture(3)
        _engine(model, X, data)._club_game_spread_hard()
        _pin(model, X, 'C PHL', 'O0 PHL', 1)
        _pin(model, X, 'C 2nd', 'O1 2nd', 2)
        _pin(model, X, 'C 3rd', 'O2 3rd', 3)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_three_games_one_hole_infeasible(self):
        # n=3 -> gap_cap 0. Used {1,2,4} -> hole at 3 -> INFEASIBLE.
        model, X, data = _fixture(3)
        _engine(model, X, data)._club_game_spread_hard()
        _pin(model, X, 'C PHL', 'O0 PHL', 1)
        _pin(model, X, 'C 2nd', 'O1 2nd', 2)
        _pin(model, X, 'C 3rd', 'O2 3rd', 4)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_four_games_one_hole_feasible(self):
        # n=4 -> gap_cap 1. Used {1,2,4,5} -> 1 hole (slot 3) -> FEASIBLE.
        model, X, data = _fixture(4)
        _engine(model, X, data)._club_game_spread_hard()
        _pin(model, X, 'C PHL', 'O0 PHL', 1)
        _pin(model, X, 'C 2nd', 'O1 2nd', 2)
        _pin(model, X, 'C 3rd', 'O2 3rd', 4)
        _pin(model, X, 'C 4th', 'O3 4th', 5)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_four_games_two_holes_infeasible(self):
        # n=4 -> gap_cap 1. Used {1,2,5,6} -> 2 holes (slots 3,4) -> INFEASIBLE.
        model, X, data = _fixture(4)
        _engine(model, X, data)._club_game_spread_hard()
        _pin(model, X, 'C PHL', 'O0 PHL', 1)
        _pin(model, X, 'C 2nd', 'O1 2nd', 2)
        _pin(model, X, 'C 3rd', 'O2 3rd', 5)
        _pin(model, X, 'C 4th', 'O3 4th', 6)
        assert _solve(model) == cp_model.INFEASIBLE


class TestClubSpreadDropsIntVars:
    def test_hard_method_adds_only_boolean_vars(self):
        """DoD 9: the rewrite drops every range/min/max/spread/overlap IntVar —
        the club-spread method now adds ONLY Boolean variables."""
        model, X, data = _fixture(4)
        engine = _engine(model, X, data)
        before = len(model.Proto().variables)
        engine._club_game_spread_hard()
        new_vars = list(model.Proto().variables)[before:]
        non_bool = [v for v in new_vars if list(v.domain) != [0, 1]]
        assert non_bool == [], f"expected only BoolVars, found non-boolean: {len(non_bool)}"


class TestNoHardKeyStrandedInSoftOnly:
    def test_no_hard_engine_key_only_in_soft_only(self):
        """DoD 12 (systemic guard): every hard engine key reachable in
        DEFAULT_STAGES must appear in at least one non-soft_only stage. Key-level
        (not atom-level) so legitimately-soft atoms sharing a hard key with hard
        siblings don't false-positive."""
        soft_flags: Dict[str, List[bool]] = defaultdict(list)
        for stage in DEFAULT_STAGES:
            soft_only = bool(stage.get('soft_only'))
            for atom in stage.get('atoms', []):
                ek = atom_to_engine_key(atom)
                if ek in ENGINE_HARD_KEYS:
                    soft_flags[ek].append(soft_only)
        stranded = sorted(k for k, flags in soft_flags.items() if flags and all(flags))
        assert stranded == [], f"hard engine keys reachable only from soft_only: {stranded}"
