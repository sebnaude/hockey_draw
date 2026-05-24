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
    PHLAnd2ndAdjacency,
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

    # spec-030 (DoD 6): locked-week skip. PHLConcurrencyAtBroadmeadow iterates
    # via `iter_phl_keys`, which (default `include_locked=False`) drops
    # locked-week vars before they reach the atom — so the atom already skips
    # locked weeks (no inline guard is added; one would be dead code, since the
    # locked-week keys never reach the grouping loop). These two tests pin that
    # behaviour as a regression guard.
    #
    # Fixture: two DISTINCT PHL games sharing (week=3, Sunday, slot 1,
    # Broadmeadow). The grouping key is (week, day, day_slot, location), so both
    # land in one group of size 2.
    @staticmethod
    def _two_phl_same_broadmeadow_slot(model, week):
        k1 = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '11:30',
              week, '2026-04-05', week, 'EF', BROADMEADOW)
        k2 = ('Gosford PHL', 'Norths PHL', 'PHL', 'Sunday', 1, '11:30',
              week, '2026-04-05', week, 'EF', BROADMEADOW)
        return {k1: model.NewBoolVar('p1'), k2: model.NewBoolVar('p2')}

    def test_skips_locked_week(self):
        # Given two same-(week,day,slot) Broadmeadow PHL games on a LOCKED week,
        model = cp_model.CpModel()
        X = self._two_phl_same_broadmeadow_slot(model, week=3)
        # When apply runs with week 3 locked,
        n = PHLConcurrencyAtBroadmeadow().apply(
            model, X, {'locked_weeks': {3}}, _registry(model))
        # Then 0 constraints are added (hand oracle: iter_phl_keys yields nothing
        # on a locked week, so `groups` is empty).
        assert n == 0
        assert len(model.Proto().constraints) == 0

    def test_applies_on_non_locked_week(self):
        # Given the same fixture on a NON-locked week,
        model = cp_model.CpModel()
        X = self._two_phl_same_broadmeadow_slot(model, week=3)
        # When apply runs with no locked weeks,
        n = PHLConcurrencyAtBroadmeadow().apply(
            model, X, {'locked_weeks': set()}, _registry(model))
        # Then exactly 1 `sum <= 1` constraint is added (one group of size 2).
        assert n == 1
        assert len(model.Proto().constraints) == 1


# ----------------------------------------------------------------------
# spec-030 (DoD 7): PHLAnd2ndAdjacency 2.5h cross-venue boundary. The deleted
# PHLAnd2ndConcurrencyAtBroadmeadow's same-slot rule is subsumed here (same
# venue is only allowed on the same field in adjacent slots), so we test the
# surviving atom's cross-venue minute threshold (now 150 = 2.5 h, via the
# `phl_2nd_cross_venue_min_minutes` fallback).
# ----------------------------------------------------------------------


class TestPHLAnd2ndAdjacencyCrossVenueGap:
    @staticmethod
    def _cross_venue_pair(model, second_time):
        """One club (Tigers) fields PHL 11:00 at Broadmeadow and 2nd at
        `second_time` at Maitland. Opponents differ (Wests PHL, Norths 2nd) so
        only the Tigers bucket has BOTH grades -> at most one evaluated pair."""
        phl = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '11:00',
               1, '2026-03-22', 1, 'EF', BROADMEADOW)
        second = ('Norths 2nd', 'Tigers 2nd', '2nd', 'Sunday', 2, second_time,
                  1, '2026-03-22', 1, 'Maitland Main Field', MAITLAND)
        return {phl: model.NewBoolVar('phl'), second: model.NewBoolVar('2nd')}

    def test_exactly_150min_is_allowed(self, phl_data):
        # Given PHL 11:00 (660) and 2nd 13:30 (810) on DIFFERENT venues,
        # |660 - 810| = 150 >= 150 -> allowed.
        model = cp_model.CpModel()
        X = self._cross_venue_pair(model, '13:30')
        n = PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        assert n == 0
        assert len(model.Proto().constraints) == 0

    def test_140min_is_forbidden(self, phl_data):
        # Given PHL 11:00 (660) and 2nd 13:20 (800) on DIFFERENT venues,
        # |660 - 800| = 140 < 150 -> forbidden (one `p + q <= 1`).
        model = cp_model.CpModel()
        X = self._cross_venue_pair(model, '13:20')
        n = PHLAnd2ndAdjacency().apply(model, X, phl_data, _registry(model))
        assert n == 1
        assert len(model.Proto().constraints) == 1


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
