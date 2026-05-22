"""Given/When/Then tests for the VenueEarliestSlotFill atom (spec-021).

No mocks: a real `cp_model.CpModel`, a real `HelperVarRegistry`, and a hand-built
X of decision BoolVars keyed exactly like production (11-tuples). Each scenario
states the hand-computed oracle and asserts the real solver outcome matches.

Decision-var key layout (see CLAUDE.md):
  (team1, team2, grade, day, day_slot, time, week, date, round_no,
   field_name, field_location)
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import VenueEarliestSlotFill
from constraints.helper_vars import HelperVarRegistry

LOC = 'Newcastle International Hockey Centre'
DATE = '2026-03-22'


def _key(slot, field='EF', t1='A', t2='B', grade='PHL', week=1):
    return (t1, t2, grade, 'Sunday', slot, '08:30', week, DATE, 1, field, LOC)


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    return solver.Solve(model)


def _apply(model, X):
    reg = HelperVarRegistry(model)
    return VenueEarliestSlotFill().apply(model, X, {'locked_weeks': set()}, reg)


class TestVenueEarliestSlotFill:
    def test_constraint_count_three_slots(self):
        # Given one candidate var in each of slots 1,2,3 at one venue.
        # Hand oracle: monotone fill over 3 sorted slots -> 2 implications.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        assert _apply(model, X) == 2

    def test_late_slot_without_earliest_infeasible(self):
        # Given a game forced into slot 3 while slot 1 is empty. Not earliest-packed.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        _apply(model, X)
        model.Add(X[_key(3)] == 1)
        model.Add(X[_key(1)] == 0)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_earliest_pack_feasible(self):
        # Given games in slots 1,2 and slot 3 empty: packed into earliest. FEASIBLE.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        _apply(model, X)
        model.Add(X[_key(1)] == 1)
        model.Add(X[_key(2)] == 1)
        model.Add(X[_key(3)] == 0)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_interior_gap_infeasible(self):
        # Given games in slots 1 and 3 but not 2: an interior hole. INFEASIBLE.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        _apply(model, X)
        model.Add(X[_key(1)] == 1)
        model.Add(X[_key(2)] == 0)
        model.Add(X[_key(3)] == 1)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_combined_across_fields(self):
        # Given slot 1 has two field candidates (EF, WF) and slots 2,3 one each.
        # The slot_used indicator is the OR across fields. Hand oracle: if EITHER
        # slot-1 field var is on, slot 1 counts as used. Force slot 3 used, both
        # slot-1 fields off -> slot 1 unused while slot 3 used -> INFEASIBLE.
        model = cp_model.CpModel()
        X = {
            _key(1, field='EF'): model.NewBoolVar('s1ef'),
            _key(1, field='WF', t1='C', t2='D'): model.NewBoolVar('s1wf'),
            _key(2): model.NewBoolVar('s2'),
            _key(3): model.NewBoolVar('s3'),
        }
        _apply(model, X)
        model.Add(X[_key(3)] == 1)
        model.Add(X[_key(1, field='EF')] == 0)
        model.Add(X[_key(1, field='WF', t1='C', t2='D')] == 0)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_combined_field_satisfies_when_one_field_on(self):
        # Same fixture: slot 3 used, but ONE slot-1 field on and slot 2 on.
        # slot 1 used (OR) and slot 2 used -> earliest-packed {1,2,3}. FEASIBLE.
        model = cp_model.CpModel()
        X = {
            _key(1, field='EF'): model.NewBoolVar('s1ef'),
            _key(1, field='WF', t1='C', t2='D'): model.NewBoolVar('s1wf'),
            _key(2): model.NewBoolVar('s2'),
            _key(3): model.NewBoolVar('s3'),
        }
        _apply(model, X)
        model.Add(X[_key(1, field='EF')] == 1)
        model.Add(X[_key(2)] == 1)
        model.Add(X[_key(3)] == 1)
        assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_locked_week_skipped(self):
        # Given a locked week: the atom must not constrain it (key[6] in locked_weeks).
        # Hand oracle: with week 1 locked, zero constraints added.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        reg = HelperVarRegistry(model)
        n = VenueEarliestSlotFill().apply(model, X, {'locked_weeks': {1}}, reg)
        assert n == 0
