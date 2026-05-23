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
from constraints.stages import (ALL_ENGINE_KEYS, ENGINE_HARD_KEYS,
                                 atom_to_engine_key, validate_solver_stages)
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


def _fixture_2field(n_club_teams: int):
    """Like _fixture but offers TWO NIHC fields (EF, WF), 6 slots each, so a
    club's games can be split across fields. Used for spec-024 per-field tests."""
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
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
        for field in (ef, wf) for s in SLOTS
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': grade_objs, 'clubs': [c] + opp_clubs, 'fields': [ef, wf],
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {g: 1 for g in grades}, 'constraint_slack': {},
        'penalty_weights': {}, 'penalties': {}, 'forced_games': [],
        'blocked_games': [], 'team_conflicts': [], 'phl_preferences': {},
        'club_days': {}, 'preference_no_play': {}, 'home_field_map': {},
        'constraint_defaults': {},
    }
    return model, X, data


def _pin_field(model, X, t1, t2, field_name, slot):
    """Force matchup (t1,t2) onto (field_name, slot) exactly (0 elsewhere)."""
    for k, v in X.items():
        if {k[0], k[1]} == {t1, t2}:
            model.Add(v == (1 if (k[4] == slot and k[9] == field_name) else 0))


class TestClubSpreadPerField:
    """spec-024 DoD 6/7: contiguity is PER FIELD; cross-field gaps are not holes;
    off-primary-field games are penalised softly."""

    def test_split_across_two_fields_each_contiguous_feasible(self):
        # 2 games EF {1,2}, 2 games WF {5,6}. Per field each block is contiguous
        # -> 0 holes on EF, 0 holes on WF -> FEASIBLE. (Day-scoped rule would see
        # slots {1,2,5,6}, n=4, gap_cap 1, 2 holes -> INFEASIBLE: the regression.)
        model, X, data = _fixture_2field(4)
        eng = _engine(model, X, data)
        eng._club_game_spread_hard()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'WF', 5)
        _pin_field(model, X, 'C 4th', 'O3 4th', 'WF', 6)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_per_field_hole_still_infeasible(self):
        # 3 games on EF in slots {1,2,4} -> hole at 3 on EF, n_field=3 gap_cap 0
        # -> INFEASIBLE even though a 4th game sits on WF.
        model, X, data = _fixture_2field(4)
        eng = _engine(model, X, data)
        eng._club_game_spread_hard()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'EF', 4)
        _pin_field(model, X, 'C 4th', 'O3 4th', 'WF', 1)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_off_primary_penalty_counts_non_primary_games(self):
        # 2 games EF {1,2} + 2 games WF {5,6}. Primary field holds 2 -> off_primary
        # = total(4) - max_field_count(2) = 2. Holes are 0. So the ClubGameSpread
        # penalty bucket sums to exactly 2.
        model, X, data = _fixture_2field(4)
        eng = _engine(model, X, data)
        eng._club_game_spread_hard()
        eng._club_game_spread_soft()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'WF', 5)
        _pin_field(model, X, 'C 4th', 'O3 4th', 'WF', 6)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        penalties = data['penalties']['ClubGameSpread']['penalties']
        total = sum(solver.Value(p) for p in penalties)
        assert total == 2, f"off_primary oracle = 2, got {total}"

    def test_single_field_optimum_zero_penalty(self):
        # All 4 games on EF, contiguous {1,2,3,4}. off_primary = 4 - 4 = 0,
        # holes = 0 -> ClubGameSpread penalty sums to 0.
        model, X, data = _fixture_2field(4)
        eng = _engine(model, X, data)
        eng._club_game_spread_hard()
        eng._club_game_spread_soft()
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'EF', 3)
        _pin_field(model, X, 'C 4th', 'O3 4th', 'EF', 4)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        penalties = data['penalties']['ClubGameSpread']['penalties']
        total = sum(solver.Value(p) for p in penalties)
        assert total == 0, f"single-field penalty oracle = 0, got {total}"


class TestClubGameSpreadWiringSpec024:
    """spec-024 DoD 8 (production wiring): ClubGameSpread runs in a non-soft_only
    stage (its hard part actually applies), the two club-balance constraints are
    gone from every stage, and the real engine emits per-field HARD constraints
    plus a soft bucket carrying BOTH per-field hole BoolVars and off_primary
    IntVars. (The per-field/off-primary numeric oracles are in TestClubSpreadPerField
    above. Full-config `generate_X(2026)` is deliberately NOT built here: doing it
    twice in one process hits the memory ceiling noted in CLAUDE.md and segfaults a
    later real-config test — so wiring is proven via DEFAULT_STAGES + a real-engine
    2-field fixture rather than a second full-season build.)"""

    def test_clubgamespread_in_non_soft_only_stage(self):
        hard = [s for s in DEFAULT_STAGES
                if 'ClubGameSpread' in s['atoms'] and not s.get('soft_only')]
        assert len(hard) >= 1, "ClubGameSpread must sit in a non-soft_only stage"
        assert validate_solver_stages(DEFAULT_STAGES) == []

    def test_deleted_constraints_absent_from_all_stages(self):
        for stage in DEFAULT_STAGES:
            assert 'MaximiseClubsPerTimeslotBroadmeadow' not in stage['atoms']
            assert 'MinimiseClubsOnAFieldBroadmeadow' not in stage['atoms']

    def test_engine_emits_hard_and_both_soft_terms(self):
        model, X, data = _fixture_2field(4)
        eng = _engine(model, X, data)
        hard = eng._club_game_spread_hard()
        eng._club_game_spread_soft()
        assert hard > 0, "no per-field hard ClubGameSpread constraints emitted"
        assert all(len(k) == 4 for k in eng._cgs_keys), \
            "spec-024: _cgs_keys must be (club, week, day, field) 4-tuples"
        penalties = data['penalties']['ClubGameSpread']['penalties']
        domains = [list(p.Proto().domain) for p in penalties]
        assert any(d == [0, 1] for d in domains), "expected per-field hole BoolVars"
        assert any(d != [0, 1] for d in domains), "expected off_primary IntVar(s)"
        assert 'MaximiseClubsPerTimeslotBroadmeadow' not in data['penalties']
        assert 'MinimiseClubsOnAFieldBroadmeadow' not in data['penalties']


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
