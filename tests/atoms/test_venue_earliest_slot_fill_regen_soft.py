"""Given/When/Then tests for VenueEarliestSlotFillRegenSoft (spec-027).

No mocks: real cp_model.CpModel, real HelperVarRegistry, hand-built X of
decision BoolVars keyed exactly like production (11-tuples). Each scenario
states the hand-computed oracle and asserts the real solver outcome matches.

Decision-var key layout (see CLAUDE.md):
  (team1, team2, grade, day, day_slot, time, week, date, round_no,
   field_name, field_location)
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.venue_earliest_slot_fill_regen_soft import (
    VenueEarliestSlotFillRegenSoft,
    REGEN_VENUE_EARLIEST_SLOT_FILL_DEFAULT_WEIGHT,
)
from constraints.helper_vars import HelperVarRegistry

LOC = 'Newcastle International Hockey Centre'
DATE = '2026-03-22'
WEEK = 1


def _key(slot, field='EF', t1='A', t2='B', grade='PHL', week=WEEK):
    """Build an 11-tuple decision-var key for a given day_slot."""
    return (t1, t2, grade, 'Sunday', slot, '08:30', week, DATE, 1, field, LOC)


def _data(penalty_weights=None, locked_weeks=None):
    return {
        'locked_weeks': locked_weeks or set(),
        'penalty_weights': penalty_weights or {},
    }


def _apply(model, X, data=None):
    """Apply atom; return (n_terms, registry, data) for inspection."""
    if data is None:
        data = _data()
    registry = HelperVarRegistry(model)
    n = VenueEarliestSlotFillRegenSoft().apply(model, X, data, registry)
    return n, registry, data


def _solve_penalty(model, X, penalty_vars):
    """Minimise total penalty, solve, return (status, total_penalty_value)."""
    total = model.NewIntVar(0, len(penalty_vars), 'total_penalty')
    model.Add(total == sum(penalty_vars))
    model.Minimize(total)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)
    penalty_value = solver.Value(total) if status in (
        cp_model.OPTIMAL, cp_model.FEASIBLE
    ) else None
    return status, penalty_value, solver


class TestVenueEarliestSlotFillRegenSoftViolation:
    """Scenario 1 — gap forces non-zero penalty; model stays FEASIBLE.

    Setup:
      Slots 1, 2, 3 at one venue/date.
      Force: slot 1 empty (no game), slot 2 used (game), slot 3 used (game).

    Slot occupancy:
      used[1] = 0  (forced empty)
      used[2] = 1  (forced)
      used[3] = 1  (forced)

    Consecutive pairs evaluated by the atom (sorted order):
      (1 → 2): used[2]=1, used[1]=0 → v = 1  (gap: slot 2 used, slot 1 empty)
      (2 → 3): used[3]=1, used[2]=1 → v = 0  (no gap)

    Hand oracle: total penalty = 1.
    """

    def test_feasible_with_gap(self):
        # Given: three slots at one venue/date, slot 2 + slot 3 used, slot 1 empty.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}

        # When: atom is applied (no hard constraints — model must stay feasible).
        n, registry, data = _apply(model, X)

        # Force the gapped assignment.
        model.Add(X[_key(1)] == 0)   # slot 1 empty
        model.Add(X[_key(2)] == 1)   # slot 2 used
        model.Add(X[_key(3)] == 1)   # slot 3 used

        # Then: FEASIBLE (soft, not hard).
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            "Model must remain FEASIBLE even with a slot gap (soft penalty only)"
        )

    def test_penalty_equals_one_for_single_gap(self):
        # Given: three slots at one venue/date, slot 2 + slot 3 used, slot 1 empty.
        # Hand oracle: exactly 1 penalty term fires (pair 1→2).
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}

        n, registry, data = _apply(model, X)
        penalty_vars = data['penalties']['regen_venue_earliest_slot_fill']['penalties']

        # Force the gapped assignment.
        model.Add(X[_key(1)] == 0)
        model.Add(X[_key(2)] == 1)
        model.Add(X[_key(3)] == 1)

        # When: minimise penalty (or simply solve — the forced assignment gives oracle).
        status, total_penalty, solver = _solve_penalty(model, X, penalty_vars)

        # Then: FEASIBLE and total penalty == 1.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Unexpected solver status {status}"
        )
        assert total_penalty == 1, (
            f"Expected penalty=1 (one gap: slot2 used, slot1 empty), got {total_penalty}"
        )

    def test_atom_returns_two_penalty_terms_for_three_slots(self):
        # Given: 3 slots → 2 consecutive pairs → atom returns 2 penalty BoolVars.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        n, registry, data = _apply(model, X)
        # Then: 2 penalty terms registered (one per consecutive pair).
        assert n == 2, f"Expected 2 penalty terms, got {n}"
        penalty_vars = data['penalties']['regen_venue_earliest_slot_fill']['penalties']
        assert len(penalty_vars) == 2


class TestVenueEarliestSlotFillRegenSoftClean:
    """Scenario 2 — no gap, penalty == 0; model FEASIBLE.

    Setup:
      Slots 1, 2, 3 at one venue/date.
      Force: slot 1 used, slot 2 used, slot 3 empty.

    Slot occupancy:
      used[1] = 1  (forced)
      used[2] = 1  (forced)
      used[3] = 0  (forced empty)

    Consecutive pairs:
      (1 → 2): used[2]=1, used[1]=1 → v = 0  (no gap)
      (2 → 3): used[3]=0, used[2]=1 → v = 0  (slot 3 empty — no game to penalise)

    Hand oracle: total penalty = 0.
    """

    def test_no_penalty_when_packed_from_first(self):
        # Given: slots 1 and 2 used, slot 3 empty (packed from earliest).
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}

        n, registry, data = _apply(model, X)
        penalty_vars = data['penalties']['regen_venue_earliest_slot_fill']['penalties']

        model.Add(X[_key(1)] == 1)   # slot 1 used
        model.Add(X[_key(2)] == 1)   # slot 2 used
        model.Add(X[_key(3)] == 0)   # slot 3 empty

        status, total_penalty, solver = _solve_penalty(model, X, penalty_vars)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Unexpected solver status {status}"
        )
        assert total_penalty == 0, (
            f"Expected penalty=0 (no gap: slots packed from 1), got {total_penalty}"
        )


class TestVenueEarliestSlotFillRegenSoftEdgeCases:
    """Edge-case coverage."""

    def test_weight_zero_returns_zero_immediately(self):
        # Given: penalty weight set to 0.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        data = _data(penalty_weights={'regen_venue_earliest_slot_fill': 0})
        n, registry, returned_data = _apply(model, X, data)
        # When: weight == 0, atom bails out early.
        assert n == 0, "Expected 0 terms when weight=0"
        # Penalty bucket should not be created.
        assert 'regen_venue_earliest_slot_fill' not in returned_data.get('penalties', {}), (
            "Penalty bucket must not be created when weight=0"
        )

    def test_locked_week_skipped(self):
        # Given: week 1 is locked — atom must skip those keys.
        model = cp_model.CpModel()
        X = {_key(s, week=1): model.NewBoolVar(f'v{s}') for s in (1, 2, 3)}
        data = _data(locked_weeks={1})
        n, registry, returned_data = _apply(model, X, data)
        assert n == 0, "Expected 0 terms for locked week"

    def test_dummy_key_skipped(self):
        # Given: a 4-tuple dummy key (len < 11) — must be ignored.
        model = cp_model.CpModel()
        dummy_key = ('A', 'B', 'PHL', 0)   # len=4 < 11
        real_key_1 = _key(1)
        real_key_2 = _key(2)
        X = {
            dummy_key: model.NewBoolVar('dummy'),
            real_key_1: model.NewBoolVar('real1'),
            real_key_2: model.NewBoolVar('real2'),
        }
        n, registry, data = _apply(model, X)
        # Two real slots → 1 pair → 1 term; dummy key ignored.
        assert n == 1, f"Expected 1 term (dummy key skipped), got {n}"

    def test_single_slot_at_venue_no_terms(self):
        # Given: only one day_slot at a venue — no consecutive pair possible.
        model = cp_model.CpModel()
        X = {_key(1): model.NewBoolVar('v1')}
        n, registry, data = _apply(model, X)
        assert n == 0, "Expected 0 terms for venue with only 1 slot"

    def test_default_weight_stored_in_bucket(self):
        # Given: no explicit penalty_weights entry.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2)}
        n, registry, data = _apply(model, X)
        bucket = data['penalties']['regen_venue_earliest_slot_fill']
        assert bucket['weight'] == REGEN_VENUE_EARLIEST_SLOT_FILL_DEFAULT_WEIGHT

    def test_two_gaps_produce_penalty_two(self):
        # Given: 4 slots; force slot 1 empty, slots 2+3+4 used.
        # Slot occupancy: used[1]=0, used[2]=1, used[3]=1, used[4]=1
        # Pairs: (1→2): gap=1; (2→3): gap=0; (3→4): gap=0 → total=1.
        # BUT also check: if slot 1 empty AND slot 3 empty with slots 2 and 4 used:
        # used[1]=0, used[2]=1, used[3]=0, used[4]=1
        # (1→2): used[2]=1, used[1]=0 → v=1 (gap)
        # (2→3): used[3]=0, used[2]=1 → v=0 (slot 3 not used, no penalty)
        # (3→4): used[4]=1, used[3]=0 → v=1 (gap)
        # Total = 2.
        model = cp_model.CpModel()
        X = {_key(s): model.NewBoolVar(f'v{s}') for s in (1, 2, 3, 4)}
        n, registry, data = _apply(model, X)
        penalty_vars = data['penalties']['regen_venue_earliest_slot_fill']['penalties']
        assert n == 3, "Expected 3 penalty terms for 4 slots (3 consecutive pairs)"

        model.Add(X[_key(1)] == 0)   # slot 1 empty
        model.Add(X[_key(2)] == 1)   # slot 2 used
        model.Add(X[_key(3)] == 0)   # slot 3 empty
        model.Add(X[_key(4)] == 1)   # slot 4 used

        status, total_penalty, solver = _solve_penalty(model, X, penalty_vars)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert total_penalty == 2, (
            f"Expected 2 penalties (gaps at 1→2 and 3→4), got {total_penalty}"
        )

    def test_combined_fields_share_slot_used_indicator(self):
        # Given: slot 1 has two field candidates (EF, WF); slot 2 has one.
        # Either field being on makes slot 1 "used" (OR indicator).
        # Force: slot-1-EF off, slot-1-WF on, slot-2 on → packed, penalty=0.
        model = cp_model.CpModel()
        X = {
            _key(1, field='EF'): model.NewBoolVar('s1ef'),
            _key(1, field='WF', t1='C', t2='D'): model.NewBoolVar('s1wf'),
            _key(2): model.NewBoolVar('s2'),
        }
        n, registry, data = _apply(model, X)
        penalty_vars = data['penalties']['regen_venue_earliest_slot_fill']['penalties']

        model.Add(X[_key(1, field='EF')] == 0)
        model.Add(X[_key(1, field='WF', t1='C', t2='D')] == 1)  # WF on → slot 1 used
        model.Add(X[_key(2)] == 1)

        status, total_penalty, solver = _solve_penalty(model, X, penalty_vars)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert total_penalty == 0, (
            "slot 1 used (via WF), slot 2 used → no gap → penalty must be 0"
        )
