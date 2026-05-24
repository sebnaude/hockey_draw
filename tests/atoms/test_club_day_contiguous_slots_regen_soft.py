"""Tests for ClubDayContiguousSlotsRegenSoft (spec-027).

Given / When / Then structure, no mocks, real CP-SAT solver, hand-computed
oracles.

Fixture summary
---------------
Tigers club has games on CLUB_DAY_DATE_STR ('2026-06-14').
Available slots on that date (from club_day_fixture): 1, 2, 3, 4.
Each slot has EF, WF, SF fields at BROADMEADOW.

Slot-occupancy reasoning for the gap scenarios:

  Scenario 1 — violation (1 internal gap):
    Force exactly one Tigers game into slot 1 (EF) and exactly one into
    slot 3 (EF), using two DIFFERENT team pairs so both are satisfiable.
    Block ALL Tigers vars in slot 2 on the date.
    Slot occupancy after solve: slot 1 = USED, slot 2 = EMPTY, slot 3 = USED.
    The sorted triple is (1, 2, 3):  prev=1, mid=2, next=3.
      prev_used=1, next_used=1, mid_used=0  →  gap at slot 2  →  v=1.
    Only one middle slot (slot 2) lies between used slots 1 and 3.
    The triple (2, 3, 4) has mid=3 used → no gap there.
    Hand oracle: total penalty == 1.

  Scenario 2 — clean (no gap):
    Force one Tigers game into slot 1 and one into slot 2 (different pairs).
    Slot occupancy: slot 1 = USED, slot 2 = USED.
    With only two occupied slots the atom needs len(slot_vars) >= 3 distinct
    slots to emit any penalty vars.  In practice, with exactly two slots used
    (1 and 2) the atom still sees all 4 possible slots (1-4) but the used
    indicators for slots 3 and 4 will be 0.  Triples examined:
      (1, 2, 3): mid_used=1 → gap = 0 (mid is used, no hole).
      (2, 3, 4): prev_used=1, mid_used=0, next_used=0 →
                 v >= 1+0-1-0 = 0, v<=1, v<=0, v<=1 → v==0, no gap.
    Hand oracle: total penalty == 0.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.club_day_contiguous_slots_regen_soft import (
    ClubDayContiguousSlotsRegenSoft,
    REGEN_CLUB_DAY_CONTIGUOUS_SLOTS_DEFAULT_WEIGHT,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_day_fixture import (
    CLUB_DAY_DATE_STR,
    build_club_day_fixture,
    build_model_X,
    solve_with_timeout,
)


def _registry(model):
    return HelperVarRegistry(model)


def _total_penalty(solver, bucket) -> int:
    """Sum the solved values of every penalty BoolVar in the bucket."""
    return sum(solver.Value(v) for v in bucket['penalties'])


# ---------------------------------------------------------------------------
# Scenario 1: slot 1 used, slot 2 EMPTY, slot 3 used → 1 internal gap
# ---------------------------------------------------------------------------


class TestScenario1Violation:
    """
    Given: Tigers games forced into slot 1 (EF) and slot 3 (EF);
           all Tigers vars on slot 2 forced off.
    When:  ClubDayContiguousSlotsRegenSoft is applied and the model is solved.
    Then:  status is FEASIBLE (model never becomes infeasible) and the total
           penalty value equals 1 (exactly one internal gap at slot 2).

    Hand oracle:
      Slot indicators on CLUB_DAY_DATE_STR for Tigers:
        slot 1: USED (forced game in EF slot 1)
        slot 2: EMPTY (all vars blocked)
        slot 3: USED (forced game in EF slot 3)
        slot 4: EMPTY (no forced games there)
      Triples (sorted slots 1,2,3,4):
        i=1 → (prev=1, mid=2, next=3):
          prev_used=1, next_used=1, mid_used=0 → gap v=1  ✓
        i=2 → (prev=2, mid=3, next=4):
          prev_used=0, next_used=0, mid_used=1 → gap v=0
      Total penalty = 1.
    """

    def test_feasible_and_penalty_equals_one(self):
        # GIVEN
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        # Two distinct Tigers team pairs for slot 1 and slot 3 (must differ).
        ef_slot1 = [
            k for k in X
            if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and k[4] == 1 and 'Tigers' in (k[0] + k[1])
        ]
        ef_slot3 = [
            k for k in X
            if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and k[4] == 3 and 'Tigers' in (k[0] + k[1])
            # Must be a different team pair from the slot-1 game.
            and (k[0], k[1]) != (ef_slot1[0][0], ef_slot1[0][1])
        ]
        assert ef_slot1, 'need at least one EF slot-1 Tigers var'
        assert ef_slot3, 'need at least one EF slot-3 Tigers var (different pair)'

        # Force one game in slot 1 and one in slot 3.
        model.Add(X[ef_slot1[0]] == 1)
        model.Add(X[ef_slot3[0]] == 1)

        # Block every Tigers var on slot 2 of the club day (any field).
        for k, v in X.items():
            if k[7] == CLUB_DAY_DATE_STR and k[4] == 2 and 'Tigers' in (k[0] + k[1]):
                model.Add(v == 0)

        # WHEN
        atom = ClubDayContiguousSlotsRegenSoft()
        n = atom.apply(model, X, data, _registry(model))
        assert n >= 1, 'atom should emit at least one penalty var for 4 slots'

        status, solver = solve_with_timeout(model)

        # THEN: model is feasible despite the gap.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'expected FEASIBLE, got {status}'
        )

        bucket = data['penalties']['regen_club_day_contiguous_slots']
        penalty = _total_penalty(solver, bucket)
        assert penalty == 1, (
            f'expected total penalty == 1 (one internal gap at slot 2), got {penalty}'
        )


# ---------------------------------------------------------------------------
# Scenario 2: slot 1 used, slot 2 used → no internal gap
# ---------------------------------------------------------------------------


class TestScenario2Clean:
    """
    Given: Tigers games forced into slot 1 (EF) and slot 2 (EF) using
           different team pairs; no other slot-blocking applied.
    When:  ClubDayContiguousSlotsRegenSoft is applied and the model is solved.
    Then:  status is FEASIBLE and total penalty == 0 (no gaps).

    Hand oracle:
      Slots 1 and 2 are both used; slots 3 and 4 may or may not be used by
      the solver (they are unconstrained).  The atom creates penalty vars for
      every middle slot in the sorted 4-slot sequence.
      Worst-case analysis (slots 3 and 4 unused):
        i=1 → (1, 2, 3): mid_used=1 → gap=0 (used middle).
        i=2 → (2, 3, 4): prev_used=1, mid_used=0, next_used=0
                         v >= 1+0-1-0=0 ; v<=0 (next_used=0) → v=0.
      Any additional slot usage can only leave mid_used=1 for triples,
      which also sets v=0.  Total penalty = 0 in all valid solutions.
    """

    def test_feasible_and_penalty_zero(self):
        # GIVEN
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        ef_slot1 = [
            k for k in X
            if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and k[4] == 1 and 'Tigers' in (k[0] + k[1])
        ]
        ef_slot2 = [
            k for k in X
            if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and k[4] == 2 and 'Tigers' in (k[0] + k[1])
            and (k[0], k[1]) != (ef_slot1[0][0], ef_slot1[0][1])
        ]
        assert ef_slot1, 'need at least one EF slot-1 Tigers var'
        assert ef_slot2, 'need at least one EF slot-2 Tigers var (different pair)'

        # Force contiguous games in slots 1 and 2.
        model.Add(X[ef_slot1[0]] == 1)
        model.Add(X[ef_slot2[0]] == 1)

        # WHEN
        atom = ClubDayContiguousSlotsRegenSoft()
        n = atom.apply(model, X, data, _registry(model))
        assert n >= 1, 'atom should emit penalty vars for the 4-slot sequence'

        status, solver = solve_with_timeout(model)

        # THEN: model is feasible.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'expected FEASIBLE, got {status}'
        )

        bucket = data['penalties']['regen_club_day_contiguous_slots']
        penalty = _total_penalty(solver, bucket)
        assert penalty == 0, (
            f'expected total penalty == 0 (no internal gaps), got {penalty}'
        )


# ---------------------------------------------------------------------------
# Ancillary: weight=0 disables the atom
# ---------------------------------------------------------------------------


class TestWeightZeroSkips:
    """When penalty_weights['regen_club_day_contiguous_slots'] == 0, apply()
    returns 0 immediately and leaves data['penalties'] untouched."""

    def test_zero_weight_returns_zero(self):
        data = build_club_day_fixture()
        data['penalty_weights']['regen_club_day_contiguous_slots'] = 0
        model, X = build_model_X(data)
        n = ClubDayContiguousSlotsRegenSoft().apply(model, X, data, _registry(model))
        assert n == 0
        assert 'regen_club_day_contiguous_slots' not in data.get('penalties', {})


# ---------------------------------------------------------------------------
# Ancillary: default weight is written into the bucket
# ---------------------------------------------------------------------------


class TestDefaultWeight:
    """When penalty_weights is empty, the default weight constant is stored in
    the bucket."""

    def test_default_weight_in_bucket(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        ClubDayContiguousSlotsRegenSoft().apply(model, X, data, _registry(model))
        bucket = data.get('penalties', {}).get('regen_club_day_contiguous_slots')
        assert bucket is not None
        assert bucket['weight'] == REGEN_CLUB_DAY_CONTIGUOUS_SLOTS_DEFAULT_WEIGHT
