"""Tests for EqualMatchUpSpacingRegenSoft atom (spec-027).

Real CP-SAT models, no mocks. Two GWT scenarios exercise the penalty
semantics: one with a violation (hand-oracle = 1), one without (oracle = 0).

Fixture recap
-------------
``phl_data`` has 5 PHL teams → T=5, so space = T-1 = 4.
``num_rounds['max']`` = R = 5.

Sliding windows: r_start ∈ {1, 2}   (range(1, R - space + 2) = range(1, 3))
  Window A: rounds 1..4  (r_start=1, r_end=4)
  Window B: rounds 2..5  (r_start=2, r_end=5)

Scenario 1 — violation (oracle = 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GIVEN the fixture data and a specific PHL pair (t1, t2) such that t1 < t2
  alphabetically (matching variable key ordering).
WHEN one decision var for that pair is forced to 1 in round 1 AND one var is
  forced to 1 in round 2.
THEN the model is FEASIBLE/OPTIMAL and the total penalty is exactly 1.

Hand arithmetic:
  Window A [1..4]: contains both the forced round-1 var and the forced
    round-2 var. The solver minimises pen_A >= sum(window_A_vars) - 1.
    sum(window_A_vars) >= 2 (the two forced vars), so pen_A >= 1.
    Solver sets pen_A = 1.
  Window B [2..5]: contains the forced round-2 var but NOT the forced
    round-1 var (round 1 is outside [2..5]). sum(window_B_vars) >= 1,
    so pen_B >= 0. Solver sets pen_B = 0.
  Total penalty = pen_A + pen_B = 1 + 0 = 1.

Scenario 2 — clean (oracle = 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GIVEN the same fixture and the same PHL pair.
WHEN one var is forced to 1 in round 1 AND one var is forced to 1 in round 5.
THEN the model is FEASIBLE/OPTIMAL and the total penalty is exactly 0.

Hand arithmetic:
  Window A [1..4]: contains round-1 var but NOT round-5 (5 > 4). sum >= 1,
    pen_A >= 0. Solver sets pen_A = 0.
  Window B [2..5]: contains round-5 var but NOT round-1 (1 < 2). sum >= 1,
    pen_B >= 0. Solver sets pen_B = 0.
  Total penalty = 0.
"""
from __future__ import annotations

import pytest
from ortools.sat.python import cp_model

from constraints.atoms.equal_matchup_spacing_regen_soft import (
    EqualMatchUpSpacingRegenSoft,
    REGEN_EQUAL_MATCHUP_SPACING_DEFAULT_WEIGHT,
)
from constraints.helper_vars import HelperVarRegistry

# Import shared conftest helpers (conftest.py is picked up automatically,
# but we import the factory functions directly for use in helper methods).
from tests.atoms.conftest import build_model_X, phl_data  # noqa: F401


# ---------------------------------------------------------------------------
# Helper: pick one var from X for a given pair at a specific round.
# ---------------------------------------------------------------------------

def _pick_var(X: dict, t1: str, t2: str, grade: str, round_no: int):
    """Return the first decision var matching (t1, t2, grade, round_no).

    Raises if none found — a test-setup failure, not an atom bug.
    """
    for key, var in X.items():
        if (key[0] == t1 and key[1] == t2 and key[2] == grade
                and key[8] == round_no):
            return key, var
    raise ValueError(
        f'No var found for ({t1!r}, {t2!r}, {grade!r}) at round {round_no}. '
        f'Check that the pair exists and the round is in the fixture.'
    )


# ---------------------------------------------------------------------------
# Scenario 1 — violation: same pair forced into rounds 1 and 2
# ---------------------------------------------------------------------------

class TestViolation:
    """GIVEN a PHL pair forced to play in rounds 1 AND 2 (both inside
    Window A [1..4]), WHEN EqualMatchUpSpacingRegenSoft is applied and the
    model is solved, THEN the model is FEASIBLE and the total penalty is 1.
    """

    def test_penalty_is_one_for_back_to_back_meetings(self, phl_data):
        # ----------------------------------------------------------------
        # Given — build model and X from the shared fixture.
        # T=5 teams, R=5 rounds, space=4.  Two windows: [1..4] and [2..5].
        # ----------------------------------------------------------------
        data = phl_data
        # Activate the atom with a non-zero weight (default applies).
        # Empty penalty_weights → the atom uses REGEN_EQUAL_MATCHUP_SPACING_DEFAULT_WEIGHT.
        data['penalty_weights'] = {}

        model, X = build_model_X(data)
        registry = HelperVarRegistry(model)

        # Pick the first PHL pair that actually exists in X (pair ordering
        # follows combinations() over the grade.teams list, NOT alphabetical).
        phl_pairs = sorted({(k[0], k[1]) for k in X if k[2] == 'PHL'})
        t1, t2 = phl_pairs[0]

        # Force one var for the pair in round 1.
        key_r1, var_r1 = _pick_var(X, t1, t2, 'PHL', 1)
        model.Add(var_r1 == 1)

        # Force one var for the pair in round 2.
        key_r2, var_r2 = _pick_var(X, t1, t2, 'PHL', 2)
        model.Add(var_r2 == 1)

        # ----------------------------------------------------------------
        # When — apply the atom.
        # ----------------------------------------------------------------
        n_penalties = EqualMatchUpSpacingRegenSoft().apply(model, X, data, registry)
        assert n_penalties > 0, 'Atom must create at least one penalty var'

        # Add an objective so the solver minimises total penalty.
        penalties = data['penalties']['regen_equal_matchup_spacing']['penalties']
        model.Minimize(sum(penalties))

        # ----------------------------------------------------------------
        # Then — model is feasible and total penalty equals 1.
        #
        # Hand oracle:
        #   Window A [1..4]:
        #     sum(window_A) >= 2  (var_r1 + var_r2 both forced to 1)
        #     pen_A >= 2 - 1 = 1  → solver sets pen_A = 1
        #   Window B [2..5]:
        #     round 1 is outside [2..5]; only var_r2 is forced to 1 here.
        #     sum(window_B) >= 1  → pen_B >= 0 → solver sets pen_B = 0
        #   Total = pen_A + pen_B = 1 + 0 = 1
        # ----------------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Model must be FEASIBLE/OPTIMAL (soft atom never forbids anything); '
            f'got {solver.StatusName(status)}'
        )
        total_penalty = sum(solver.Value(p) for p in penalties)
        assert total_penalty == 1, (
            f'Hand oracle: pen_A=1 (both rounds in window [1..4]) + pen_B=0 '
            f'(only round 2 in window [2..5]) = 1; got {total_penalty}'
        )


# ---------------------------------------------------------------------------
# Scenario 2 — clean: pair forced into rounds 1 and 5 (no window contains both)
# ---------------------------------------------------------------------------

class TestClean:
    """GIVEN a PHL pair forced to play in rounds 1 AND 5 (different windows),
    WHEN EqualMatchUpSpacingRegenSoft is applied and the model is solved,
    THEN the model is FEASIBLE and the total penalty is 0.
    """

    def test_penalty_is_zero_for_well_spaced_meetings(self, phl_data):
        # ----------------------------------------------------------------
        # Given — same fixture, pair spread across rounds 1 and 5.
        # ----------------------------------------------------------------
        data = phl_data
        data['penalty_weights'] = {}

        model, X = build_model_X(data)
        registry = HelperVarRegistry(model)

        phl_pairs = sorted({(k[0], k[1]) for k in X if k[2] == 'PHL'})
        t1, t2 = phl_pairs[0]

        # Force one var for the pair in round 1.
        _key_r1, var_r1 = _pick_var(X, t1, t2, 'PHL', 1)
        model.Add(var_r1 == 1)

        # Force one var for the pair in round 5.
        _key_r5, var_r5 = _pick_var(X, t1, t2, 'PHL', 5)
        model.Add(var_r5 == 1)

        # ----------------------------------------------------------------
        # When — apply the atom.
        # ----------------------------------------------------------------
        EqualMatchUpSpacingRegenSoft().apply(model, X, data, registry)

        penalties = data['penalties']['regen_equal_matchup_spacing']['penalties']
        model.Minimize(sum(penalties) if penalties else model.NewConstant(0))

        # ----------------------------------------------------------------
        # Then — model is feasible and total penalty is 0.
        #
        # Hand oracle:
        #   space = T-1 = 4.  Windows: [1..4] and [2..5].
        #   Window A [1..4]:
        #     var_r1 (round 1) is inside; var_r5 (round 5) is NOT (5 > 4).
        #     sum(window_A) >= 1 → pen_A >= 0 → solver sets pen_A = 0
        #   Window B [2..5]:
        #     var_r5 (round 5) is inside; var_r1 (round 1) is NOT (1 < 2).
        #     sum(window_B) >= 1 → pen_B >= 0 → solver sets pen_B = 0
        #   Total = 0 + 0 = 0
        # ----------------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Model must be FEASIBLE/OPTIMAL; got {solver.StatusName(status)}'
        )
        total_penalty = sum(solver.Value(p) for p in penalties) if penalties else 0
        assert total_penalty == 0, (
            f'Hand oracle: no window contains both rounds 1 and 5 '
            f'(window A=[1..4], window B=[2..5]); total penalty must be 0, '
            f'got {total_penalty}'
        )
