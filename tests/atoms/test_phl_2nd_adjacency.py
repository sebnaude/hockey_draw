# spec-014: GWT, no-mock, hand-computed-oracle tests for PHLAnd2ndAdjacency.
"""Tests for the `PHLAnd2ndAdjacency` atom (spec-014).

Rule: per (club, week, day) where the club fields BOTH a PHL and a 2nd game:
- same venue  -> same field AND adjacent day_slots (back-to-back);
- different venue -> start times >= phl_2nd_cross_venue_min_minutes (180) apart.

Real CP-SAT models on the small `phl_data` fixture (tests/atoms/conftest.py).
No mocks. Each scenario forces the two games of interest to 1 and checks
feasibility, with the rule's outcome hand-computed in the comments.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import PHLAnd2ndAdjacency
from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.conftest import build_model_X, solve_with_timeout


def _registry(model):
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


def _key(X, *, grade, team, week, day_slot, field_name, location, day='Sunday'):
    """Locate exactly one X key matching the criteria (team in either slot)."""
    return next(
        k for k in X
        if k[2] == grade and team in (k[0], k[1]) and k[3] == day
        and k[6] == week and k[4] == day_slot and k[9] == field_name
        and k[10] == location
    )


class TestPHLAnd2ndAdjacency:

    def test_same_venue_non_adjacent_infeasible(self, phl_data):
        """Given Tigers PHL @ NIHC EF slot 1 and Tigers 2nd @ NIHC EF slot 3
        (same venue, same field, |1-3|=2 not adjacent). Then INFEASIBLE."""
        model, X = build_model_X(phl_data)
        phl = _key(X, grade='PHL', team='Tigers PHL', week=1, day_slot=1,
                   field_name='EF', location=BROADMEADOW)
        sec = _key(X, grade='2nd', team='Tigers 2nd', week=1, day_slot=3,
                   field_name='EF', location=BROADMEADOW)
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)
        n = PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        assert n > 0  # the forbidden pair (at least) is added
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE

    def test_same_venue_same_field_adjacent_feasible(self, phl_data):
        """Given Tigers PHL @ NIHC EF slot 1 and Tigers 2nd @ NIHC EF slot 2
        (same field, adjacent slots = back-to-back). Then FEASIBLE."""
        model, X = build_model_X(phl_data)
        phl = _key(X, grade='PHL', team='Tigers PHL', week=1, day_slot=1,
                   field_name='EF', location=BROADMEADOW)
        sec = _key(X, grade='2nd', team='Tigers 2nd', week=1, day_slot=2,
                   field_name='EF', location=BROADMEADOW)
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)
        PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_cross_venue_under_180_infeasible(self, phl_data):
        """Given Tigers PHL @ NIHC EF slot 3 (14:30) and Tigers 2nd @ Maitland
        slot 1 (12:00): different venues, |14:30-12:00| = 150 min < 180.
        Then INFEASIBLE."""
        model, X = build_model_X(phl_data)
        phl = _key(X, grade='PHL', team='Tigers PHL', week=1, day_slot=3,
                   field_name='EF', location=BROADMEADOW)  # 14:30
        sec = _key(X, grade='2nd', team='Tigers 2nd', week=1, day_slot=1,
                   field_name='Maitland Main Field', location=MAITLAND)  # 12:00
        # Oracle: 14:30 = 870 min, 12:00 = 720 min, |870-720| = 150 < 180.
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)
        n = PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        assert n > 0
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE

    def test_cross_venue_at_least_180_feasible(self, phl_data):
        """Given Tigers PHL @ NIHC EF slot 1 (11:30) and Tigers 2nd @ Maitland
        slot 3 (15:00): different venues, |15:00-11:30| = 210 min >= 180.
        Then FEASIBLE."""
        model, X = build_model_X(phl_data)
        phl = _key(X, grade='PHL', team='Tigers PHL', week=1, day_slot=1,
                   field_name='EF', location=BROADMEADOW)  # 11:30
        sec = _key(X, grade='2nd', team='Tigers 2nd', week=1, day_slot=3,
                   field_name='Maitland Main Field', location=MAITLAND)  # 15:00
        # Oracle: 11:30 = 690 min, 15:00 = 900 min, |900-690| = 210 >= 180.
        model.Add(X[phl] == 1)
        model.Add(X[sec] == 1)
        PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_club_fields_only_phl_adds_nothing(self, phl_data):
        """Given a model with PHL vars only (no 2nd grade). Then the atom adds
        zero constraints (no club fields both grades)."""
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        assert n == 0

    def test_adds_no_decision_variables(self, phl_data):
        """The atom is a pure constraint atom — it must add ZERO new decision
        variables (spec-014 'Chosen encoding': forbid-over-infeasible-pairs)."""
        model, X = build_model_X(phl_data)
        before = len(model.Proto().variables)
        PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        after = len(model.Proto().variables)
        assert after == before
