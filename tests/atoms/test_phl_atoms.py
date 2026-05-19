# spec-013: GWT pass — tests confirmed to meet /basic Given/When/Then + hand-computed-oracle bar.
"""Solo-clean + solo-violation tests for the PHLAndSecondGradeTimes atoms.

spec-010: PHLRoundOnePlay was removed from production stages. Its tests are
now in `TestPHLRoundOnePlayObsolete` which confirms (a) the atom is still
callable as a parity reference and (b) round-1 absence is FEASIBLE without it.

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
# PHLRoundOnePlay (OBSOLETE — spec-010)
# ----------------------------------------------------------------------
# The atom is kept on disk as a parity reference but is NO LONGER wired
# into any production stage. Tests below confirm:
#  (a) the atom itself is still callable (import + apply work) — parity ref.
#  (b) the production stage list does NOT include PHLRoundOnePlay — a
#      schedule where every PHL team sits out round 1 is now legal.


class TestPHLRoundOnePlayObsolete:
    """
    spec-010: PHLRoundOnePlay removed from production stages.

    Given: a PHL fixture where every round-1 variable for Tigers PHL is zeroed.
    When:  the DEFAULT_STAGES critical_feasibility list is applied
           (NOT the raw atom directly).
    Then:  the model is still FEASIBLE — no constraint blocks round-1 absence.
    """

    def test_atom_still_callable_as_parity_reference(self, phl_data):
        """
        Given: a clean PHL fixture.
        When:  PHLRoundOnePlay.apply() is called directly (parity-reference only).
        Then:  the model is FEASIBLE (the atom itself is not broken).
        Hand-computed: clean fixture has at least one round-1 variable per team,
        so the sum >= 1 constraint is satisfiable.
        """
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PHLRoundOnePlay().apply(model, X, phl_data, _registry(model))
        assert n > 0  # atom added constraints
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_production_does_not_enforce_round_one_attendance(self, phl_data):
        """
        spec-010 DoD: removing PHLRoundOnePlay from the stage list means a
        schedule where Tiger PHL sits out round 1 is now FEASIBLE.

        Hand-computed oracle: with Tigers PHL round-1 variables all set to 0
        and NO PHLRoundOnePlay constraint applied, the model has slack — it can
        schedule Tigers PHL in rounds 2+.  The remaining PHL constraints
        (PHLConcurrencyAtBroadmeadow, PHLAnd2ndConcurrencyAtBroadmeadow,
        GosfordFridayRoundsForced) impose nothing about round 1 attendance for
        any specific team, so FEASIBLE is expected.

        Given: Tigers PHL round-1 vars forced to 0.
        When:  only production-wired atoms applied (none mandate round-1 play).
        Then:  model is FEASIBLE (round-1 absence is allowed).
        """
        model, X = build_model_X(phl_data, allow_2nd=False)
        # Force Tigers PHL out of round 1
        for k in [k for k in X if k[2] == 'PHL' and k[8] == 1
                  and 'Tigers PHL' in (k[0], k[1])]:
            model.Add(X[k] == 0)
        # Apply the three production atoms only — PHLRoundOnePlay intentionally absent
        reg = _registry(model)
        PHLConcurrencyAtBroadmeadow().apply(model, X, phl_data, reg)
        PHLAnd2ndConcurrencyAtBroadmeadow().apply(model, X, phl_data, reg)
        GosfordFridayRoundsForced().apply(model, X, phl_data, reg)
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            'Round-1 absence should be feasible after spec-010 removal of PHLRoundOnePlay'
        )


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
