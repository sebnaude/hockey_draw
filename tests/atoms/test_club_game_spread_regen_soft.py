"""spec-027 — ClubGameSpreadRegenSoft regen-soft atom.

The atom is the SOFT analogue of the ClubGameSpread per-field-hole + off-primary
pressure: it emits penalty/indicator vars equal to the violation amount and
NEVER forbids any assignment (the model stays feasible for any X).

Driven through the real class (no mocks), GWT style, with hand-computed oracles.

Penalty bucket: ``regen_club_game_spread``. One unit per:
  - a per-field residual interior HOLE (used-before AND used-after AND empty), and
  - an OFF-PRIMARY game = total_games_that_day - max_games_on_a_single_field.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from constraints.atoms.base import BROADMEADOW
from constraints.atoms.club_game_spread_regen_soft import (
    REGEN_CLUB_GAME_SPREAD_DEFAULT_WEIGHT, ClubGameSpreadRegenSoft)
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars

GRADES = ['PHL', '2nd', '3rd', '4th']
SLOTS = list(range(1, 7))  # 6 NIHC slots per field


def _fixture_2field(n_club_teams: int):
    """Club 'C' fields `n_club_teams` teams across two NIHC fields (EF, WF),
    6 slots each, week 1. Every team plays one opponent from its own 1-team
    club, so each matchup is a distinct grade and can land on any (field, slot).
    """
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


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    return status, solver


def _bucket_total(data, solver):
    pens = data['penalties']['regen_club_game_spread']['penalties']
    return sum(solver.Value(p) for p in pens)


class TestClubGameSpreadRegenSoftViolation:
    def test_off_primary_and_hole_penalty(self):
        # GIVEN club C plays 4 games on one day, split:
        #   EF used slots {1, 2, 4}  (3 games on EF)
        #   WF used slot  {1}        (1 game  on WF)
        # WHEN the regen-soft atom defines its penalties and we solve.
        #
        # HAND ORACLE
        #   Per-field HOLES:
        #     EF sorted-used slots = [1, 2, 4]; the offered EF slots are 1..6, so
        #     the slot_used channel covers every slot. Interior empty slot 3 has a
        #     used slot before (1,2) and after (4) -> 1 hole on EF.
        #     WF has a single used slot -> < 2 used, no interior hole.
        #     => holes = 1.
        #   OFF-PRIMARY (club, week, day):
        #     total = 4 ; field counts EF=3, WF=1 ; max_field_count = 3.
        #     off_primary = total - max = 4 - 3 = 1.
        #   TOTAL penalty = holes(1) + off_primary(1) = 2.
        model, X, data = _fixture_2field(4)
        n = ClubGameSpreadRegenSoft().apply(
            model, X, data, HelperVarRegistry(model))
        assert n > 0
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'EF', 4)
        _pin_field(model, X, 'C 4th', 'O3 4th', 'WF', 1)

        status, solver = _solve(model)
        # Regen-soft: model is FEASIBLE for this (violating) assignment.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert _bucket_total(data, solver) == 2

    def test_weight_recorded_in_bucket(self):
        model, X, data = _fixture_2field(4)
        ClubGameSpreadRegenSoft().apply(model, X, data, HelperVarRegistry(model))
        bucket = data['penalties']['regen_club_game_spread']
        assert bucket['weight'] == REGEN_CLUB_GAME_SPREAD_DEFAULT_WEIGHT == 20000


class TestClubGameSpreadRegenSoftClean:
    def test_single_field_contiguous_zero_penalty(self):
        # GIVEN all 4 of club C's games that day on EF, contiguous slots {1,2,3,4}.
        # HAND ORACLE
        #   EF used slots [1,2,3,4]: no interior empty slot -> 0 holes.
        #   off_primary: single field -> structurally 0 (skipped).
        #   TOTAL penalty = 0.
        model, X, data = _fixture_2field(4)
        ClubGameSpreadRegenSoft().apply(model, X, data, HelperVarRegistry(model))
        _pin_field(model, X, 'C PHL', 'O0 PHL', 'EF', 1)
        _pin_field(model, X, 'C 2nd', 'O1 2nd', 'EF', 2)
        _pin_field(model, X, 'C 3rd', 'O2 3rd', 'EF', 3)
        _pin_field(model, X, 'C 4th', 'O3 4th', 'EF', 4)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert _bucket_total(data, solver) == 0


class TestClubGameSpreadRegenSoftDisabled:
    def test_zero_weight_short_circuits(self):
        model, X, data = _fixture_2field(4)
        data['penalty_weights'] = {'regen_club_game_spread': 0}
        n = ClubGameSpreadRegenSoft().apply(
            model, X, data, HelperVarRegistry(model))
        assert n == 0
        assert 'regen_club_game_spread' not in data.get('penalties', {})
