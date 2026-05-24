# spec-027: GWT, no-mock, hand-computed-oracle tests for
# PHLAnd2ndAdjacencyRegenSoft (SOFT regen analogue of PHLAnd2ndAdjacency).
"""Tests for the `PHLAnd2ndAdjacencyRegenSoft` atom (spec-027).

SOFT rule: per (club, week, day) where the club fields BOTH a PHL and a 2nd
game, emit a penalty BoolVar = 1 exactly when the adjacency rule is BROKEN
(neither same-venue-back-to-back nor cross-venue->=150-min holds). The model
stays FEASIBLE for any X; the objective subtracts the penalties.

Real CP-SAT models on the small `phl_data` fixture (tests/atoms/conftest.py).
No mocks. Each scenario forces the two games of interest to 1, sums the
penalty vars, and checks feasibility + total penalty with the oracle in the
comments.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.phl_2nd_adjacency_regen_soft import (
    PHLAnd2ndAdjacencyRegenSoft,
)
from constraints.atoms.base import BROADMEADOW
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.conftest import build_model_X, solve_with_timeout


def _registry(model):
    return HelperVarRegistry(model)


def _key(X, *, grade, teams, week, day_slot, field_name, location, day='Sunday'):
    """Locate exactly one X key for the exact matchup `teams` (a 2-set).

    Pinning a *full* matchup (both teams) — rather than "any game involving
    team T" — keeps the violation confined to a single (club, week, day)
    bucket. Each game registers against BOTH clubs it involves; choosing
    distinct opponents for the PHL and 2nd pinned games ensures only the
    Tigers bucket ever holds BOTH a pinned PHL and a pinned 2nd game.
    """
    teams = set(teams)
    return next(
        k for k in X
        if k[2] == grade and {k[0], k[1]} == teams and k[3] == day
        and k[6] == week and k[4] == day_slot and k[9] == field_name
        and k[10] == location
    )


def _penalty_sum(data):
    return data['penalties']['regen_phl_2nd_adjacency']['penalties']


def _only_tigers(key):
    """Keep only Tigers PHL / Tigers 2nd vars so the model is tiny and the
    minimisation is trivial — every penalty pair in the bucket then involves
    the pinned games only."""
    return key[0] in ('Tigers PHL', 'Tigers 2nd') or key[1] in (
        'Tigers PHL', 'Tigers 2nd'
    )


class TestPHLAnd2ndAdjacencyRegenSoft:

    def test_same_venue_non_adjacent_penalty_one(self, phl_data):
        """Scenario 1 (violation).

        Given Tigers PHL @ NIHC EF day_slot 1 (11:30) and Tigers 2nd @ NIHC EF
        day_slot 3 (14:30), both forced to 1.

        Oracle: SAME venue (both Broadmeadow), so rule (a) applies: needs same
        field AND adjacent day_slots. Same field (EF) but |1 - 3| = 2 != 1 ->
        NOT adjacent -> breaks (a). Rule (b) (cross-venue >=150 min) does not
        apply because the venues are identical. Neither (a) nor (b) holds ->
        BROKEN -> penalty == 1.

        Then FEASIBLE (soft) and total penalty == 1.
        """
        model, X = build_model_X(phl_data, filter_keys=_only_tigers)
        phl = _key(X, grade='PHL', teams={'Tigers PHL', 'Norths PHL'}, week=1,
                   day_slot=1, field_name='EF', location=BROADMEADOW)
        sec = _key(X, grade='2nd', teams={'Tigers 2nd', 'Wests 2nd'}, week=1,
                   day_slot=3, field_name='EF', location=BROADMEADOW)
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)

        n = PHLAnd2ndAdjacencyRegenSoft().apply(
            model, X, phl_data, _registry(model)
        )
        assert n > 0  # at least the violating pair emits a penalty var

        penalties = _penalty_sum(phl_data)
        total = model.NewIntVar(0, len(penalties), 'total_penalty')
        model.Add(total == sum(penalties))
        # Minimise so free (non-pinned) penalty vars are driven to 0; the only
        # penalty forced to 1 is the pinned Tigers violation -> total == 1.
        model.Minimize(total)

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(total) == 1

    def test_same_venue_adjacent_penalty_zero(self, phl_data):
        """Scenario 2 (clean).

        Given Tigers PHL @ NIHC EF day_slot 1 (11:30) and Tigers 2nd @ NIHC EF
        day_slot 2 (13:00), both forced to 1.

        Oracle: SAME venue (Broadmeadow), same field (EF), |1 - 2| = 1 ->
        adjacent -> rule (a) holds -> COMPLIANT -> penalty == 0.

        Then FEASIBLE and total penalty == 0.
        """
        model, X = build_model_X(phl_data, filter_keys=_only_tigers)
        phl = _key(X, grade='PHL', teams={'Tigers PHL', 'Norths PHL'}, week=1,
                   day_slot=1, field_name='EF', location=BROADMEADOW)
        sec = _key(X, grade='2nd', teams={'Tigers 2nd', 'Wests 2nd'}, week=1,
                   day_slot=2, field_name='EF', location=BROADMEADOW)
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)

        PHLAnd2ndAdjacencyRegenSoft().apply(
            model, X, phl_data, _registry(model)
        )

        penalties = _penalty_sum(phl_data)
        total = model.NewIntVar(0, max(1, len(penalties)), 'total_penalty')
        model.Add(total == sum(penalties))
        # Minimise so the solver is free to drive non-pinned penalty vars to 0.
        model.Minimize(total)

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(total) == 0

    def test_adds_no_hard_forbidding(self, phl_data):
        """Even when a violating pair is present, the atom must NOT make the
        model infeasible (it is purely soft). Force the Scenario-1 violation
        and assert the model still solves."""
        model, X = build_model_X(phl_data, filter_keys=_only_tigers)
        phl = _key(X, grade='PHL', teams={'Tigers PHL', 'Norths PHL'}, week=1,
                   day_slot=1, field_name='EF', location=BROADMEADOW)
        sec = _key(X, grade='2nd', teams={'Tigers 2nd', 'Wests 2nd'}, week=1,
                   day_slot=3, field_name='EF', location=BROADMEADOW)
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)
        PHLAnd2ndAdjacencyRegenSoft().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
