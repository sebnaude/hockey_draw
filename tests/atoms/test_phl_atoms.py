"""Solo-clean + solo-violation tests for the 8 PHLAndSecondGradeTimes atoms.

Each atom gets:
- Solo-clean: applied on a clean fixture, model is feasible.
- Solo-violation: with a programmatic violation construction, atom forces the
  expected constraint.

Tests use real CP-SAT models on the small `phl_data` fixture in conftest.py.
No mocks.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import (
    GosfordFridayRoundsForced,
    PHLAnd2ndConcurrencyAtBroadmeadow,
    PHLConcurrencyAtBroadmeadow,
    PHLRoundOnePlay,
    PreferredDates,
)
from constraints.atoms.base import BROADMEADOW, GOSFORD, MAITLAND
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.conftest import build_model_X, solve_with_timeout


def _registry(model):
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


# ----------------------------------------------------------------------
# PHLConcurrencyAtBroadmeadow
# ----------------------------------------------------------------------


class TestPHLConcurrencyAtBroadmeadow:
    def test_solo_clean_feasible(self, phl_data):
        model, X = build_model_X(phl_data)
        n = PHLConcurrencyAtBroadmeadow().apply(model, X, phl_data, _registry(model))
        assert n > 0
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_blocks_two_phl_in_same_slot(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        keys_at_slot = [
            k for k in X
            if k[2] == 'PHL' and k[3] == 'Sunday' and k[10] == BROADMEADOW
            and k[6] == 1 and k[4] == 1
        ]
        assert len(keys_at_slot) >= 2, 'fixture must produce two PHL candidates at slot 1'
        k1, k2 = keys_at_slot[0], keys_at_slot[1]
        model.Add(X[k1] == 1)
        model.Add(X[k2] == 1)
        PHLConcurrencyAtBroadmeadow().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE

    def test_does_not_constrain_outside_broadmeadow(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        PHLConcurrencyAtBroadmeadow().apply(model, X, phl_data, _registry(model))
        gosford_keys = [k for k in X if k[10] == GOSFORD and k[6] == 1 and k[3] == 'Sunday']
        if len(gosford_keys) >= 2:
            model.Add(X[gosford_keys[0]] == 1)
            model.Add(X[gosford_keys[1]] == 1)
            status, _ = solve_with_timeout(model)
            assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# PHLAnd2ndConcurrencyAtBroadmeadow
# ----------------------------------------------------------------------


class TestPHLAnd2ndConcurrencyAtBroadmeadow:
    def test_solo_clean_feasible(self, phl_data):
        model, X = build_model_X(phl_data)
        PHLAnd2ndConcurrencyAtBroadmeadow().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_same_club_phl_and_2nd_at_broadmeadow_slot(self, phl_data):
        model, X = build_model_X(phl_data)
        phl_key = next(
            k for k in X
            if k[2] == 'PHL' and k[10] == BROADMEADOW and k[3] == 'Sunday'
            and ('Tigers PHL' in (k[0], k[1])) and k[6] == 1 and k[4] == 1
        )
        second_key = next(
            k for k in X
            if k[2] == '2nd' and k[10] == BROADMEADOW and k[3] == 'Sunday'
            and ('Tigers 2nd' in (k[0], k[1]))
            and k[6] == 1 and k[4] == 1
        )
        model.Add(X[phl_key] == 1)
        model.Add(X[second_key] == 1)
        PHLAnd2ndConcurrencyAtBroadmeadow().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE


# ----------------------------------------------------------------------
# GosfordFridayRoundsForced
# ----------------------------------------------------------------------


class TestGosfordFridayRoundsForced:
    def test_solo_clean_feasible(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        GosfordFridayRoundsForced().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_rules_force_at_least_one_in_each_required_round(self, phl_data):
        phl_data['constraint_defaults']['gosford_friday_rounds'] = [2]
        model, X = build_model_X(phl_data, allow_2nd=False)
        for k in [k for k in X if k[2] == 'PHL' and k[3] == 'Friday'
                  and k[10] == GOSFORD and k[8] == 2]:
            model.Add(X[k] == 0)
        GosfordFridayRoundsForced().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE


# ----------------------------------------------------------------------
# PHLRoundOnePlay
# ----------------------------------------------------------------------


class TestPHLRoundOnePlay:
    def test_solo_clean_feasible(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        PHLRoundOnePlay().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_team_must_play_in_round_one(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        for k in [k for k in X if k[2] == 'PHL' and k[8] == 1
                  and ('Tigers PHL' in (k[0], k[1]))]:
            model.Add(X[k] == 0)
        PHLRoundOnePlay().apply(model, X, phl_data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE


# ----------------------------------------------------------------------
# PreferredDates
# ----------------------------------------------------------------------


class TestPreferredDates:
    def test_no_preferences_is_no_op(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PreferredDates().apply(model, X, phl_data, _registry(model))
        assert n == 0

    def test_records_penalty_for_preferred_dates(self, phl_data):
        from datetime import datetime as _dt
        phl_data['phl_preferences'] = {
            'preferred_dates': [_dt(2026, 3, 22), _dt(2026, 3, 29)],
        }
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PreferredDates().apply(model, X, phl_data, _registry(model))
        assert n >= 1
        pen_block = phl_data['penalties']['phl_preferences']
        assert pen_block['weight'] == 10000
        assert len(pen_block['penalties']) >= 1

    def test_invalid_preference_key_raises(self, phl_data):
        phl_data['phl_preferences'] = {'unsupported_key': []}
        model, X = build_model_X(phl_data, allow_2nd=False)
        try:
            PreferredDates().apply(model, X, phl_data, _registry(model))
        except ValueError as e:
            assert 'unsupported_key' in str(e)
        else:
            raise AssertionError('expected ValueError on unknown phl_preferences key')
