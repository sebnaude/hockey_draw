"""Tests for the `TeamPairNoConcurrency` atom (spec-007).

Real CP-SAT models, no mocks. Scenarios:

1. CONFIG EMPTY: no entries declared. Atom is a no-op.
2. PAIR AVOIDABLE: a pair (TeamA, TeamB) where the scheduler has room to
   place them in different slots. The objective drives them apart.
3. PAIR INFEASIBLE-TO-AVOID: a pair pinned into the same slot by other
   constraints. Model still SAT (atom is soft); penalty is paid.
4. WEIGHT MULTIPLIER: an entry with weight=10 produces a larger penalty
   contribution than weight=1, demonstrated by comparing total objective
   penalty between two minimal models.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from constraints.atoms import TeamPairNoConcurrency
from constraints.atoms.base import BROADMEADOW
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


def _registry(model):
    r = HelperVarRegistry(model)
    return r


def _fixture() -> Dict:
    """Two clubs, two PHL games, two slots. The two team pairs can be split
    across slots OR forced together depending on constraints applied."""
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')

    clubs = [
        Club(name='Alpha', home_field=BROADMEADOW),
        Club(name='Bravo', home_field=BROADMEADOW),
        Club(name='Charlie', home_field=BROADMEADOW),
        Club(name='Delta', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Alpha PHL',   club=clubs[0], grade='PHL'),
        Team(name='Bravo PHL',   club=clubs[1], grade='PHL'),
        Team(name='Charlie PHL', club=clubs[2], grade='PHL'),
        Team(name='Delta PHL',   club=clubs[3], grade='PHL'),
    ]
    grades = [Grade(name='PHL', teams=[t.name for t in teams])]
    games = [
        ('Alpha PHL', 'Bravo PHL', 'PHL'),
        ('Charlie PHL', 'Delta PHL', 'PHL'),
    ]
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                 day_slot=1, field=ef, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                 day_slot=1, field=wf, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='13:00', week=1,
                 day_slot=2, field=ef, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='13:00', week=1,
                 day_slot=2, field=wf, round_no=1),
    ]
    return {
        'games': games, 'teams': teams, 'clubs': clubs, 'grades': grades,
        'timeslots': timeslots, 'fields': [ef, wf],
        'locked_weeks': set(),
        'num_rounds': {'PHL': 1, 'max': 1},
        'penalties': {},
        'constraint_defaults': {},
        'penalty_weights': {},
    }


def _build_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (t1, t2, grade, ts.day, ts.day_slot, ts.time,
                   ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_w{ts.week}_s{ts.day_slot}_{ts.field.name}'
            )
    return model, X


def _solve(model, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


# ----------------------------------------------------------------------
# Scenario 1: NO ENTRIES — atom is a no-op
# ----------------------------------------------------------------------


class TestNoEntries:
    def test_empty_config_means_zero_constraints_added(self):
        """Given: TEAM_PAIR_NO_CONCURRENCY = []
        When: atom applied
        Then: zero constraints added; model trivially feasible."""
        data = _fixture()
        data['constraint_defaults']['TEAM_PAIR_NO_CONCURRENCY'] = []
        model, X = _build_X(data)
        n = TeamPairNoConcurrency().apply(model, X, data, _registry(model))
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 2: AVOIDABLE — solver drives the pair apart when objective rewards it
# ----------------------------------------------------------------------


class TestAvoidableConflict:
    def test_pair_avoided_when_solver_minimises_penalty(self):
        """Given: TEAM_PAIR_NO_CONCURRENCY=[('Alpha PHL','Charlie PHL')].
        Each Alpha-Bravo and Charlie-Delta game must occur exactly once
        across the four timeslot variables.
        When: atom applied + objective Minimises the penalty bucket sum.
        Then: optimal objective is 0 -- Alpha-Bravo and Charlie-Delta land
              in DIFFERENT (week, day_slot) buckets, so the pair never
              co-occurs.

        Hand calc: there are two distinct day_slots (1, 2). Placing
        Alpha-Bravo at slot 1 and Charlie-Delta at slot 2 (or vice versa)
        keeps both pair team-sums at 1 in their own slot and 0 in the
        other. raw = max(0, 1+0-1) = 0 in both slots. Total penalty = 0."""
        data = _fixture()
        data['constraint_defaults']['TEAM_PAIR_NO_CONCURRENCY'] = [
            ('Alpha PHL', 'Charlie PHL'),
        ]
        model, X = _build_X(data)
        # Exactly one of the four Alpha-Bravo vars must be 1 (game played
        # exactly once), and same for Charlie-Delta.
        ab_vars = [v for k, v in X.items()
                   if {k[0], k[1]} == {'Alpha PHL', 'Bravo PHL'}]
        cd_vars = [v for k, v in X.items()
                   if {k[0], k[1]} == {'Charlie PHL', 'Delta PHL'}]
        model.Add(sum(ab_vars) == 1)
        model.Add(sum(cd_vars) == 1)
        TeamPairNoConcurrency().apply(model, X, data, _registry(model))
        # Manual objective: minimise the penalty bucket.
        pens = data['penalties']['TeamPairNoConcurrency']['penalties']
        model.Minimize(sum(pens))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.ObjectiveValue() == 0.0


# ----------------------------------------------------------------------
# Scenario 3: UNAVOIDABLE — soft, so model is STILL SAT
# ----------------------------------------------------------------------


class TestUnavoidableConflict:
    def test_pair_pinned_together_remains_sat(self):
        """Given: external constraints pin Alpha-Bravo to slot 1 EF AND
        Charlie-Delta to slot 1 WF. The pair (Alpha PHL, Charlie PHL)
        therefore MUST co-occur in week=1 day_slot=1.
        When: atom applied.
        Then: model is FEASIBLE with non-zero penalty (atom is soft).

        Hand calc: in slot (1, 1), vars_Alpha = {Alpha-Bravo @ slot 1 EF} -> sum=1;
        vars_Charlie = {Charlie-Delta @ slot 1 WF} -> sum=1. raw =
        max(0, 1+1-1) = 1. Solver returns FEASIBLE; objective value (penalty)
        = 1 if we minimise."""
        data = _fixture()
        data['constraint_defaults']['TEAM_PAIR_NO_CONCURRENCY'] = [
            ('Alpha PHL', 'Charlie PHL'),
        ]
        model, X = _build_X(data)
        ab_pin = next(k for k in X
                      if {k[0], k[1]} == {'Alpha PHL', 'Bravo PHL'}
                      and k[4] == 1 and k[9] == 'EF')
        cd_pin = next(k for k in X
                      if {k[0], k[1]} == {'Charlie PHL', 'Delta PHL'}
                      and k[4] == 1 and k[9] == 'WF')
        model.Add(X[ab_pin] == 1)
        model.Add(X[cd_pin] == 1)
        # Exactly one game per pair.
        for tpair in [('Alpha PHL', 'Bravo PHL'), ('Charlie PHL', 'Delta PHL')]:
            vars_ = [v for k, v in X.items()
                     if {k[0], k[1]} == set(tpair)]
            model.Add(sum(vars_) == 1)
        TeamPairNoConcurrency().apply(model, X, data, _registry(model))
        pens = data['penalties']['TeamPairNoConcurrency']['penalties']
        model.Minimize(sum(pens))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.ObjectiveValue() == 1.0


# ----------------------------------------------------------------------
# Scenario 4: WEIGHT MULTIPLIER
# ----------------------------------------------------------------------


class TestWeightMultiplier:
    def test_weight_multiplier_scales_penalty(self):
        """Given: identical infeasible-to-avoid scenarios but with weight 1
        vs weight 10 on the same pair.
        Then: minimised objective is 1 in the weight=1 case and 10 in the
              weight=10 case.

        Hand calc: the raw penalty is 1 (forced co-occurrence). The scaled
        penalty IntVar equals multiplier * raw, so scaled=1 vs scaled=10.
        Minimised total objective matches exactly."""
        results = {}
        for multiplier in (1, 10):
            data = _fixture()
            data['constraint_defaults']['TEAM_PAIR_NO_CONCURRENCY'] = [
                ('Alpha PHL', 'Charlie PHL', multiplier),
            ]
            model, X = _build_X(data)
            ab_pin = next(k for k in X
                          if {k[0], k[1]} == {'Alpha PHL', 'Bravo PHL'}
                          and k[4] == 1 and k[9] == 'EF')
            cd_pin = next(k for k in X
                          if {k[0], k[1]} == {'Charlie PHL', 'Delta PHL'}
                          and k[4] == 1 and k[9] == 'WF')
            model.Add(X[ab_pin] == 1)
            model.Add(X[cd_pin] == 1)
            for tpair in [('Alpha PHL', 'Bravo PHL'), ('Charlie PHL', 'Delta PHL')]:
                vars_ = [v for k, v in X.items()
                         if {k[0], k[1]} == set(tpair)]
                model.Add(sum(vars_) == 1)
            TeamPairNoConcurrency().apply(model, X, data, _registry(model))
            pens = data['penalties']['TeamPairNoConcurrency']['penalties']
            model.Minimize(sum(pens))
            status, solver = _solve(model)
            assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            results[multiplier] = solver.ObjectiveValue()

        assert results[1] == 1.0
        assert results[10] == 10.0
