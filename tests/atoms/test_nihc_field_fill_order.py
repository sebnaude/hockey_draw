"""Tests for the `NIHCFillWFBeforeEF` and `NIHCFillEFBeforeSF` atoms (spec-003).

Real CP-SAT models, no mocks. Each test computes its expected outcome by hand
in comments. Scenarios:

1. ONLY-EF-NO-WF: a (date, day_slot) with an EF game scheduled but no WF
   game; both WF and EF are real options. `NIHCFillWFBeforeEF` must make
   the model infeasible.
2. ONLY-SF-NO-EF: a (date, day_slot) with an SF game scheduled but no EF
   game; both SF and EF are real options. `NIHCFillEFBeforeSF` must make
   the model infeasible.
3. ALL-THREE-USED: WF, EF, and SF all carry games at the same slot. Both
   implications hold. Model must remain feasible.
4. ONLY-WF-USED: only WF carries a game at the slot. Both implications are
   vacuously satisfied. Model must remain feasible.
5. EDGE-CASE-SF-NOT-A-VALID-SLOT: a (date, day_slot) where SF does not
   exist as a real timeslot for the day (no SF games anywhere in the
   draw on that date). Forcing an EF game and zero SF games must remain
   feasible because the atom must skip this bucket -- it cannot assert
   "EF must be used" via SF's helper if SF isn't even an option, and
   crucially must not infer "SF must be used" either.
   This test pairs with a separate guarantee for `NIHCFillEFBeforeSF`:
   when SF doesn't exist as a slot, no constraint is added.
6. WF-NOT-A-VALID-SLOT: a (date, day_slot) where WF does not exist (no
   WF vars in X at all). An EF game must remain feasible because the
   atom must skip the bucket rather than assert an impossible
   implication.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

from constraints.atoms import NIHCFillWFBeforeEF, NIHCFillEFBeforeSF
from constraints.atoms.base import BROADMEADOW
from constraints.helper_vars import HelperVarRegistry


def _registry(model):
    r = HelperVarRegistry(model)
    # Freeze with empty maps -- atoms use pool-style get_or_create_bool
    # which is valid post-freeze.
    r.freeze({}, {})
    return r


def _build_X(
    model: cp_model.CpModel, vars_spec: List[Tuple],
) -> Dict[Tuple, Tuple]:
    """Build X from a list of tuples:
    (t1, t2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
    Returns dict {key: BoolVar}.
    """
    X: Dict = {}
    for spec in vars_spec:
        label = f'X_{spec[0]}_{spec[1]}_w{spec[6]}_s{spec[4]}_{spec[9]}'
        X[spec] = model.NewBoolVar(label)
    return X


def _data(locked_weeks=None):
    return {
        'locked_weeks': set(locked_weeks or []),
    }


def _solve(model, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


# ----------------------------------------------------------------------
# Scenario 1: ONLY EF, NO WF -- WF/EF implication must fire
# ----------------------------------------------------------------------


class TestWFBeforeEF:
    def test_ef_used_wf_not_makes_model_infeasible(self):
        """Given: a single (date='2026-03-22', day_slot=2) bucket at NIHC
        with two vars present -- one EF game and one WF game. Force
        EF==1 and WF==0.
        When: `NIHCFillWFBeforeEF` is applied.
        Then: model is INFEASIBLE.

        Hand calc: the atom collects two buckets for the (date, slot,
        field). Both WF and EF are present so the implication
        `EF_used <= WF_used` is added. Since EF_used = 1 and WF_used = 0,
        1 <= 0 is false -- INFEASIBLE."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 1)
        model.Add(X[wf_key] == 0)
        n = NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        assert n == 1
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE

    def test_both_ef_and_wf_used_is_feasible(self):
        """Given: same bucket as above but BOTH EF and WF forced ==1.
        When: atom applied.
        Then: FEASIBLE. EF_used=1 <= WF_used=1 holds."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 1)
        model.Add(X[wf_key] == 1)
        NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_only_wf_used_is_feasible(self):
        """Given: only WF carries a game; EF var exists but is 0.
        When: atom applied.
        Then: FEASIBLE. EF_used=0 <= WF_used=1 holds."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 0)
        model.Add(X[wf_key] == 1)
        NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_skips_when_wf_has_no_variables_at_slot(self):
        """Given: a (date, day_slot) with an EF game only -- WF is not
        present in X at all (a real-world case: field unavailability).
        When: atom applied.
        Then: zero constraints added AND model FEASIBLE with EF==1.

        Hand calc: the slot_field_index sees WF not in `present`, so the
        bucket is skipped. No constraint added. EF==1 is fine."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-05-15', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [ef_key])
        model.Add(X[ef_key] == 1)
        n = NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 2: ONLY SF, NO EF -- EF/SF implication must fire
# ----------------------------------------------------------------------


class TestEFBeforeSF:
    def test_sf_used_ef_not_makes_model_infeasible(self):
        """Given: a single (date, slot) with SF==1 and EF==0 (both
        present as vars), WF also present (==1 to satisfy the other
        implication, though we only apply EF/SF here).
        When: `NIHCFillEFBeforeSF` is applied.
        Then: INFEASIBLE -- SF_used=1, EF_used=0, 1<=0 fails."""
        model = cp_model.CpModel()
        sf_key = ('A', 'B', '6th', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'SF', BROADMEADOW)
        ef_key = ('C', 'D', '6th', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [sf_key, ef_key])
        model.Add(X[sf_key] == 1)
        model.Add(X[ef_key] == 0)
        n = NIHCFillEFBeforeSF().apply(model, X, _data(), _registry(model))
        assert n == 1
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE

    def test_both_sf_and_ef_used_is_feasible(self):
        model = cp_model.CpModel()
        sf_key = ('A', 'B', '6th', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'SF', BROADMEADOW)
        ef_key = ('C', 'D', '6th', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [sf_key, ef_key])
        model.Add(X[sf_key] == 1)
        model.Add(X[ef_key] == 1)
        NIHCFillEFBeforeSF().apply(model, X, _data(), _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 3: ALL THREE FIELDS USED -- both implications hold
# ----------------------------------------------------------------------


class TestAllThreeFieldsUsed:
    def test_all_three_used_is_feasible(self):
        """Given: WF, EF, SF all carry a game at the same (date, slot).
        When: BOTH atoms applied.
        Then: FEASIBLE. SF_used=1 <= EF_used=1 <= WF_used=1.

        Hand calc: atoms add `ef <= wf` and `sf <= ef`. With all three
        forced ==1, both inequalities are 1<=1. FEASIBLE."""
        model = cp_model.CpModel()
        wf_key = ('A', 'B', '3rd', 'Sunday', 3, '14:30', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        ef_key = ('C', 'D', '3rd', 'Sunday', 3, '14:30', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        sf_key = ('E', 'F', '3rd', 'Sunday', 3, '14:30', 1,
                  '2026-03-22', 1, 'SF', BROADMEADOW)
        X = _build_X(model, [wf_key, ef_key, sf_key])
        model.Add(X[wf_key] == 1)
        model.Add(X[ef_key] == 1)
        model.Add(X[sf_key] == 1)
        reg = _registry(model)
        NIHCFillWFBeforeEF().apply(model, X, _data(), reg)
        NIHCFillEFBeforeSF().apply(model, X, _data(), reg)
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 4: ONLY WF USED -- both implications vacuously hold
# ----------------------------------------------------------------------


class TestOnlyWFUsed:
    def test_only_wf_used_is_feasible(self):
        """Given: all three vars exist; only WF==1, EF==0, SF==0.
        When: BOTH atoms applied.
        Then: FEASIBLE. EF=0<=WF=1 and SF=0<=EF=0 both hold."""
        model = cp_model.CpModel()
        wf_key = ('A', 'B', '3rd', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        ef_key = ('C', 'D', '3rd', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        sf_key = ('E', 'F', '3rd', 'Sunday', 1, '11:30', 1,
                  '2026-03-22', 1, 'SF', BROADMEADOW)
        X = _build_X(model, [wf_key, ef_key, sf_key])
        model.Add(X[wf_key] == 1)
        model.Add(X[ef_key] == 0)
        model.Add(X[sf_key] == 0)
        reg = _registry(model)
        NIHCFillWFBeforeEF().apply(model, X, _data(), reg)
        NIHCFillEFBeforeSF().apply(model, X, _data(), reg)
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 5: EDGE CASE -- SF not a valid slot. Atom must skip.
# ----------------------------------------------------------------------


class TestSFNotAValidSlot:
    def test_atom_skips_when_sf_has_no_vars_at_slot(self):
        """Given: a (date, slot) bucket with only WF and EF vars (no SF
        at all -- representing a date/day where SF wasn't an option).
        Force EF==1, WF==1.
        When: `NIHCFillEFBeforeSF` applied.
        Then: zero constraints added AND model FEASIBLE.

        Hand calc: slot_field_index has {'WF', 'EF'} -- SF not present.
        Atom's check `if 'SF' not in present` triggers `continue`. No
        constraint added. Crucially: the atom must NOT demand SF be
        used (which would be impossible)."""
        model = cp_model.CpModel()
        wf_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        ef_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [wf_key, ef_key])
        model.Add(X[wf_key] == 1)
        model.Add(X[ef_key] == 1)
        n = NIHCFillEFBeforeSF().apply(model, X, _data(), _registry(model))
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 6: Non-NIHC variables are ignored
# ----------------------------------------------------------------------


class TestNonNIHCIgnored:
    def test_maitland_park_vars_are_ignored(self):
        """Given: a game at Maitland Park's 'Maitland Main Field' with the
        same (date, day_slot) coordinates as a NIHC EF game. Force the
        EF game ==1 and don't add WF.
        When: `NIHCFillWFBeforeEF` applied.
        Then: zero constraints added; model FEASIBLE.

        Hand calc: only field_location == BROADMEADOW vars feed the
        buckets. The Maitland var is filtered out before reaching the
        slot_field_index. The NIHC bucket has only EF, so 'WF' not in
        present and the bucket is skipped (per the WF-not-available
        rule). No constraint added."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        maitland_key = ('E', 'F', '3rd', 'Sunday', 2, '13:00', 1,
                        '2026-03-22', 1, 'Maitland Main Field',
                        'Maitland Park')
        X = _build_X(model, [ef_key, maitland_key])
        model.Add(X[ef_key] == 1)
        model.Add(X[maitland_key] == 1)
        n = NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 7: Locked weeks are skipped
# ----------------------------------------------------------------------


class TestLockedWeeksSkipped:
    def test_locked_week_vars_dont_drive_constraints(self):
        """Given: an EF==1, WF==0 setup that would be infeasible, but
        the variables live in a locked week (week 1) and data carries
        `locked_weeks={1}`.
        When: atom applied.
        Then: zero constraints added; FEASIBLE.

        Hand calc: `_collect_nihc_field_vars` skips key[6] in
        locked_weeks. Buckets end empty. No implication added."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 1)
        model.Add(X[wf_key] == 0)
        n = NIHCFillWFBeforeEF().apply(
            model, X, _data(locked_weeks={1}), _registry(model)
        )
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 8: Dummy keys (len < 11) are ignored
# ----------------------------------------------------------------------


class TestDummyKeysIgnored:
    def test_dummy_short_key_is_not_treated_as_a_game(self):
        """Given: a dummy var (4-tuple key) alongside an EF==1 var.
        When: atom applied.
        Then: behaviour identical to "EF only" -- dummy contributes nothing.

        Hand calc: `_collect_nihc_field_vars` short-circuits on
        `len(key) < 11`. The dummy is invisible. Only the EF bucket is
        seen; without WF, the bucket is skipped."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1,
                  '2026-03-22', 1, 'EF', BROADMEADOW)
        dummy_key = ('A', 'B', '3rd', 0)  # 4-tuple, len < 11
        X = _build_X(model, [ef_key])
        X[dummy_key] = model.NewBoolVar('dummy')
        model.Add(X[ef_key] == 1)
        model.Add(X[dummy_key] == 1)
        n = NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
