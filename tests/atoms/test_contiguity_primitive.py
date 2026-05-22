"""Given/When/Then unit tests for the shared `_contiguity` primitive.

No mocks: real `cp_model.CpModel` instances and a real `HelperVarRegistry`.
Each scenario states the hand-computed oracle (which slots may be used, which
constraint count is added, which solver outcome results) and asserts the real
model matches it.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms._contiguity import (
    enforce_monotone_fill,
    enforce_no_gaps,
    slot_used_indicators,
)
from constraints.helper_vars import HelperVarRegistry


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    return solver.Solve(model)


# ----------------------------------------------------------------------
# slot_used_indicators
# ----------------------------------------------------------------------


class TestSlotUsedIndicators:
    def test_channels_max_and_dedupes(self):
        # Given: slot 1 has one var, slot 2 has two vars, slot 3 has none.
        model = cp_model.CpModel()
        reg = HelperVarRegistry(model)
        a = model.NewBoolVar('a')
        b = model.NewBoolVar('b')
        c = model.NewBoolVar('c')
        vars_by_slot = {1: [a], 2: [b, c], 3: []}

        # When: building indicators twice with the same kind+prefix.
        inds1 = slot_used_indicators(reg, vars_by_slot, 'venue_slot_used', 'NIHC', 7)
        inds2 = slot_used_indicators(reg, vars_by_slot, 'venue_slot_used', 'NIHC', 7)

        # Then: same BoolVar object returned per (kind, prefix, slot) — shared, not duplicated.
        assert inds1[1] is inds2[1]
        assert inds1[2] is inds2[2]
        assert set(inds1) == {1, 2, 3}

        # And: slot 3 (empty vars) is pinned to 0; slot 1 indicator == a.
        # Hand oracle: force a=1 -> ind[1] must be 1; force b=c=0 -> ind[2] must be 0.
        model.Add(a == 1)
        model.Add(b == 0)
        model.Add(c == 0)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(inds1[1]) == 1
        assert solver.Value(inds1[2]) == 0
        assert solver.Value(inds1[3]) == 0


# ----------------------------------------------------------------------
# enforce_no_gaps  (FLOATING: contiguous block, may start anywhere)
# ----------------------------------------------------------------------


class TestEnforceNoGaps:
    def _three_slot_model(self):
        model = cp_model.CpModel()
        slot_inds = {s: model.NewBoolVar(f's{s}') for s in (1, 2, 3)}
        return model, slot_inds

    def test_constraint_count_is_len_minus_two(self):
        # Hand oracle: 3 sorted slots -> exactly 1 middle slot -> 1 constraint.
        model, slot_inds = self._three_slot_model()
        n = enforce_no_gaps(model, slot_inds)
        assert n == 1

    def test_hole_in_middle_infeasible(self):
        # Given used {1,3}, empty {2}: an interior hole. When no-gap enforced. Then INFEASIBLE.
        model, slot_inds = self._three_slot_model()
        enforce_no_gaps(model, slot_inds)
        model.Add(slot_inds[1] == 1)
        model.Add(slot_inds[2] == 0)
        model.Add(slot_inds[3] == 1)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_contiguous_block_feasible(self):
        # Given used {1,2}, empty {3}: contiguous. Then FEASIBLE.
        model, slot_inds = self._three_slot_model()
        enforce_no_gaps(model, slot_inds)
        model.Add(slot_inds[1] == 1)
        model.Add(slot_inds[2] == 1)
        model.Add(slot_inds[3] == 0)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_floating_start_feasible(self):
        # Given used {2,3}, empty {1}: floating (starts at slot 2). no-gap allows it. FEASIBLE.
        model, slot_inds = self._three_slot_model()
        enforce_no_gaps(model, slot_inds)
        model.Add(slot_inds[1] == 0)
        model.Add(slot_inds[2] == 1)
        model.Add(slot_inds[3] == 1)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# enforce_monotone_fill  (ANCHORED: pack into earliest slots)
# ----------------------------------------------------------------------


class TestEnforceMonotoneFill:
    def _slot_model(self, slots):
        model = cp_model.CpModel()
        slot_inds = {s: model.NewBoolVar(f's{s}') for s in slots}
        return model, slot_inds

    def test_constraint_count_is_len_minus_one(self):
        # Hand oracle: 3 sorted slots -> 2 consecutive pairs -> 2 implications.
        model, slot_inds = self._slot_model((1, 2, 3))
        n = enforce_monotone_fill(model, slot_inds)
        assert n == 2

    def test_late_slot_without_early_infeasible(self):
        # Given slot 3 used while slot 1 empty: not anchored to earliest. INFEASIBLE.
        model, slot_inds = self._slot_model((1, 2, 3))
        enforce_monotone_fill(model, slot_inds)
        model.Add(slot_inds[3] == 1)
        model.Add(slot_inds[1] == 0)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_earliest_pack_feasible(self):
        # Given used {1,2}, empty {3}: packed into earliest slots. FEASIBLE.
        model, slot_inds = self._slot_model((1, 2, 3))
        enforce_monotone_fill(model, slot_inds)
        model.Add(slot_inds[1] == 1)
        model.Add(slot_inds[2] == 1)
        model.Add(slot_inds[3] == 0)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_floating_block_infeasible_under_monotone(self):
        # Given used {2,3}, empty {1}: a floating block enforce_no_gaps would allow,
        # but monotone-fill forbids (slot 2 used implies slot 1 used). INFEASIBLE.
        model, slot_inds = self._slot_model((1, 2, 3))
        enforce_monotone_fill(model, slot_inds)
        model.Add(slot_inds[1] == 0)
        model.Add(slot_inds[2] == 1)
        model.Add(slot_inds[3] == 1)
        assert _solve(model) == cp_model.INFEASIBLE
