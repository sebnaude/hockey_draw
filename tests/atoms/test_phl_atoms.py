# spec-013: GWT pass — tests confirmed to meet /basic Given/When/Then + hand-computed-oracle bar.
"""Solo-clean + solo-violation tests for the PHLAndSecondGradeTimes atoms.

spec-010: PHLRoundOnePlay was removed from production stages and the atom
file + registry entry were later DELETED; no tests remain for it.

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
    PHLAnd2ndConcurrencyAtBroadmeadow,
    PHLConcurrencyAtBroadmeadow,
)
from constraints.atoms.base import BROADMEADOW, GOSFORD, MAITLAND
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.conftest import build_model_X, solve_with_timeout


def _registry(model):
    r = HelperVarRegistry(model)
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
# spec-015: GosfordFridayRoundsForced was DELETED. Its per-round `sum == 1`
# rule is now expressed via generic FORCED_GAMES count entries; that generic
# capability is tested in tests/test_forced_games_count_rules.py.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# spec-010: PHLRoundOnePlay was removed from production stages, and the atom
# file + registry entry were later DELETED. "Every PHL team plays round 1" is
# now expressed via FORCED_GAMES when the convenor wants it. No atom tests
# remain for it.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# spec-020: `PreferredDates` deleted. Its narrow `|sum − 1|`-on-a-date
# behaviour is now a PREFERRED_GAMES entry handled by the generic
# `PreferredGames` atom. See tests/atoms/test_preferred_games.py
# (incl. the equivalence test proving the migrated entry yields the same
# `|sum − 1|` penalty PreferredDates produced).
# ----------------------------------------------------------------------
