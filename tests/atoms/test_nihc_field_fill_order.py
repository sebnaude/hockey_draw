"""Tests for the `NIHCFillWFBeforeEF` and `NIHCFillEFBeforeSF` atoms.

spec-016: these are now SOFT symmetry-breakers. Per (date, day_slot) at NIHC
where both fields are real options, the atom adds a penalty BoolVar that is 1
exactly when the higher-priority field is empty while the lower one is used
(EF without WF; SF without EF). They NEVER add a hard implication, so a FORCED
out-of-order placement stays feasible.

Real CP-SAT models, no mocks. Each test computes its expected outcome by hand.
The penalty var is pinned exactly (three linear constraints), so its value is
determined even without an objective term.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from constraints.atoms import NIHCFillWFBeforeEF, NIHCFillEFBeforeSF
from constraints.atoms.base import BROADMEADOW
from constraints.helper_vars import HelperVarRegistry


def _registry(model):
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


def _build_X(model: cp_model.CpModel, vars_spec: List[Tuple]) -> Dict[Tuple, object]:
    X: Dict = {}
    for spec in vars_spec:
        label = f'X_{spec[0]}_{spec[1]}_w{spec[6]}_s{spec[4]}_{spec[9]}'
        X[spec] = model.NewBoolVar(label)
    return X


def _data(locked_weeks=None, weight=None):
    d = {'locked_weeks': set(locked_weeks or [])}
    if weight is not None:
        d['penalty_weights'] = {'nihc_fill_order': weight}
    return d


def _solve(model, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


def _penalties(data):
    return data.get('penalties', {}).get('nihc_fill_order', {}).get('penalties', [])


# ----------------------------------------------------------------------
# WF-before-EF: soft penalty, never infeasible
# ----------------------------------------------------------------------


class TestWFBeforeEFSoft:
    def test_ef_used_wf_not_is_feasible_with_one_penalty(self):
        """Given: (date, slot) with EF==1 and WF==0 (both real options).
        When: NIHCFillWFBeforeEF applied.
        Then: FEASIBLE (no hard rule) AND the penalty bucket has exactly one
        term whose solved value is 1.

        Hand calc: viol = EF_used AND NOT WF_used = 1 AND NOT 0 = 1. Pinned by
        viol>=1-0, viol<=1, viol<=1-0 → viol=1."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 1)
        model.Add(X[wf_key] == 0)
        data = _data()
        n = NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        assert n == 1
        pens = _penalties(data)
        assert len(pens) == 1
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(pens[0]) == 1

    def test_forced_ef_with_empty_wf_stays_feasible(self):
        """spec-016 core: a FORCED EF game with WF empty must be FEASIBLE
        (the old hard implication would have made this INFEASIBLE)."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 1)   # forced onto EF
        model.Add(X[wf_key] == 0)   # WF deliberately empty this slot
        NIHCFillWFBeforeEF().apply(model, X, _data(), _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_canonical_fill_has_zero_penalty(self):
        """Given: both EF and WF used (canonical fill). Then penalty == 0.

        Hand calc: viol = 1 AND NOT 1 = 0. Pinned by viol<=1-WF_used=0."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 1)
        model.Add(X[wf_key] == 1)
        data = _data()
        NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(solver.Value(p) for p in _penalties(data)) == 0

    def test_only_wf_used_zero_penalty(self):
        """Given: WF==1, EF==0. Then penalty == 0 (viol = 0 AND ... = 0)."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        model.Add(X[ef_key] == 0)
        model.Add(X[wf_key] == 1)
        data = _data()
        NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(solver.Value(p) for p in _penalties(data)) == 0

    def test_skips_when_wf_not_a_slot(self):
        """WF absent from X at the slot → no penalty term (n==0)."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-05-15', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [ef_key])
        model.Add(X[ef_key] == 1)
        data = _data()
        n = NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        assert n == 0
        assert _penalties(data) == []

    def test_weight_zero_is_no_op(self):
        """weight 0 → atom adds nothing and creates no penalty bucket."""
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        data = _data(weight=0)
        n = NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        assert n == 0
        assert 'nihc_fill_order' not in data.get('penalties', {})


# ----------------------------------------------------------------------
# EF-before-SF: soft penalty
# ----------------------------------------------------------------------


class TestEFBeforeSFSoft:
    def test_sf_used_ef_not_is_feasible_with_one_penalty(self):
        """Given: SF==1, EF==0 (both real options). Then FEASIBLE and exactly
        one penalty term of value 1 (viol = SF_used AND NOT EF_used = 1)."""
        model = cp_model.CpModel()
        sf_key = ('A', 'B', '6th', 'Sunday', 1, '11:30', 1, '2026-03-22', 1, 'SF', BROADMEADOW)
        ef_key = ('C', 'D', '6th', 'Sunday', 1, '11:30', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [sf_key, ef_key])
        model.Add(X[sf_key] == 1)
        model.Add(X[ef_key] == 0)
        data = _data()
        n = NIHCFillEFBeforeSF().apply(model, X, data, _registry(model))
        assert n == 1
        pens = _penalties(data)
        assert len(pens) == 1
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(pens[0]) == 1

    def test_both_sf_and_ef_used_zero_penalty(self):
        model = cp_model.CpModel()
        sf_key = ('A', 'B', '6th', 'Sunday', 1, '11:30', 1, '2026-03-22', 1, 'SF', BROADMEADOW)
        ef_key = ('C', 'D', '6th', 'Sunday', 1, '11:30', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [sf_key, ef_key])
        model.Add(X[sf_key] == 1)
        model.Add(X[ef_key] == 1)
        data = _data()
        NIHCFillEFBeforeSF().apply(model, X, data, _registry(model))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(solver.Value(p) for p in _penalties(data)) == 0


# ----------------------------------------------------------------------
# Both atoms share the nihc_fill_order bucket; canonical fill = zero penalty
# ----------------------------------------------------------------------


class TestSharedBucketCanonicalFill:
    def test_all_three_used_zero_total_penalty(self):
        """WF+EF+SF all used at one slot. Both atoms write to the shared
        'nihc_fill_order' bucket; with canonical fill every term is 0.

        Hand calc: WF/EF viol = EF&!WF = 1&0 = 0; EF/SF viol = SF&!EF = 1&0 = 0.
        Bucket has 2 terms, sum 0."""
        model = cp_model.CpModel()
        wf_key = ('A', 'B', '3rd', 'Sunday', 3, '14:30', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        ef_key = ('C', 'D', '3rd', 'Sunday', 3, '14:30', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        sf_key = ('E', 'F', '3rd', 'Sunday', 3, '14:30', 1, '2026-03-22', 1, 'SF', BROADMEADOW)
        X = _build_X(model, [wf_key, ef_key, sf_key])
        model.Add(X[wf_key] == 1)
        model.Add(X[ef_key] == 1)
        model.Add(X[sf_key] == 1)
        data = _data()
        reg = _registry(model)
        NIHCFillWFBeforeEF().apply(model, X, data, reg)
        NIHCFillEFBeforeSF().apply(model, X, data, reg)
        pens = _penalties(data)
        assert len(pens) == 2  # one per atom, shared bucket
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(solver.Value(p) for p in pens) == 0


# ----------------------------------------------------------------------
# Locked weeks / dummy keys / non-NIHC are ignored (unchanged from hard atoms)
# ----------------------------------------------------------------------


class TestExclusions:
    def test_locked_week_vars_add_no_penalty(self):
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        wf_key = ('C', 'D', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'WF', BROADMEADOW)
        X = _build_X(model, [ef_key, wf_key])
        data = _data(locked_weeks={1})
        n = NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        assert n == 0
        assert _penalties(data) == []

    def test_dummy_key_ignored(self):
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        X = _build_X(model, [ef_key])
        X[('A', 'B', '3rd', 0)] = model.NewBoolVar('dummy')
        data = _data()
        n = NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        assert n == 0

    def test_non_nihc_vars_ignored(self):
        model = cp_model.CpModel()
        ef_key = ('A', 'B', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1, 'EF', BROADMEADOW)
        maitland_key = ('E', 'F', '3rd', 'Sunday', 2, '13:00', 1, '2026-03-22', 1,
                        'Maitland Main Field', 'Maitland Park')
        X = _build_X(model, [ef_key, maitland_key])
        data = _data()
        n = NIHCFillWFBeforeEF().apply(model, X, data, _registry(model))
        assert n == 0
