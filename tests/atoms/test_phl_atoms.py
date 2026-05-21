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
# PreferredDates
# ----------------------------------------------------------------------


class TestPreferredDates:
    def test_no_preferences_is_no_op(self, phl_data):
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PreferredDates().apply(model, X, phl_data, _registry(model))
        assert n == 0

    def test_records_penalty_for_preferred_dates(self, phl_data):
        """Scenario: PreferredDates records exactly one penalty entry per preferred date that has PHL candidate vars."""
        # Given: the conftest `_build_phl_fixture` produces 5 PHL teams
        # (Tigers, Wests, Norths, Maitland, Gosford) → C(5,2)=10 PHL pairings.
        # For each pairing, build_model_X(allow_2nd=False) creates Sunday vars
        # on dates 2026-03-22 (round 1), 2026-03-29 (round 2), and three later
        # dates. Two preferred dates are pinned: 2026-03-22 and 2026-03-29.
        # Both dates have many PHL vars (10 pairs × 8 Sunday slot/field combos
        # at NIHC + 2 Gosford slots + 4 Maitland slots subject to home filter,
        # but home filter only restricts Maitland Park / Gosford venues — NIHC
        # is open so each pairing has 8 NIHC Sunday vars per date).
        from datetime import datetime as _dt
        phl_data['phl_preferences'] = {
            'preferred_dates': [_dt(2026, 3, 22), _dt(2026, 3, 29)],
        }
        model, X = build_model_X(phl_data, allow_2nd=False)

        # And: count actual PHL vars on each preferred date — confirms BOTH
        # dates have at least one PHL candidate (the atom only emits a penalty
        # for dates that do).
        phl_vars_on_d1 = sum(
            1 for k in X
            if k[2] == 'PHL' and k[7] == '2026-03-22'
        )
        phl_vars_on_d2 = sum(
            1 for k in X
            if k[2] == 'PHL' and k[7] == '2026-03-29'
        )
        assert phl_vars_on_d1 > 0 and phl_vars_on_d2 > 0, (
            'fixture must have PHL vars on both preferred dates'
        )

        # When: the atom is applied.
        n = PreferredDates().apply(model, X, phl_data, _registry(model))

        # Then: the atom returns 2 — one penalty IntVar per preferred date.
        # Oracle: the atom (constraints/atoms/preferred_dates.py) loops over
        # `per_date.items()`, creating one `pen = |sum(vars_list) - 1|` per
        # date with vars. Two preferred dates × both have vars → n == 2.
        # Locked weeks are empty in the conftest fixture so no date is skipped.
        assert n == 2

        # And: the penalty bucket carries default weight 10000 (atom default
        # when `penalty_weights['phl_preferences']` is unset in the fixture).
        pen_block = phl_data['penalties']['phl_preferences']
        assert pen_block['weight'] == 10000

        # And: bucket length matches n exactly — one IntVar per preferred date.
        # Hand-computed: 2 preferred dates × 1 entry per date = 2 entries.
        assert len(pen_block['penalties']) == 2

    def test_invalid_preference_key_raises(self, phl_data):
        phl_data['phl_preferences'] = {'unsupported_key': []}
        model, X = build_model_X(phl_data, allow_2nd=False)
        try:
            PreferredDates().apply(model, X, phl_data, _registry(model))
        except ValueError as e:
            assert 'unsupported_key' in str(e)
        else:
            raise AssertionError('expected ValueError on unknown phl_preferences key')
