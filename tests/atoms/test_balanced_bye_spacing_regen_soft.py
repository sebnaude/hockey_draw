"""Tests for the ``BalancedByeSpacingRegenSoft`` atom.

Real CP-SAT models, no mocks. Each scenario builds a tiny grade fixture
end-to-end using the same ``_make_grade_fixture`` / ``_build_X`` / pin
helpers as the sibling hard-atom test
(``tests/atoms/test_balanced_bye_spacing.py``).

Scenarios
---------

1. **Violation detected — penalty counted correctly.**
   Given 6 teams, R=10 playable rounds, games_per_team=7 → byes=3/team.

   Hand oracle:
     ideal_bye_gap(10, 3) = 10//3 - 1 = 3 - 1 = 2   ← S
     (10//3 == 3 because Python int-divides 10÷3=3.)

   We force one target team to bye at rounds {1, 2, 3} (three byes, all
   within S=2 of each other) and play in rounds {4..10} (7 rounds).

   Too-close bye pairs (gap <= S=2):
     (1,2) gap=1 ≤ 2  ✓ violation
     (1,3) gap=2 ≤ 2  ✓ violation
     (2,3) gap=1 ≤ 2  ✓ violation
   → 3 penalty terms for this team.

   The atom creates ONE penalty BoolVar per (r1, r2) pair per team. With
   the byes forced to {1,2,3}, both B_r1 and B_r2 are 1 for each of those
   three pairs, so every penalty BoolVar v is forced to 1 by the three
   encoding constraints (v >= 1+1-1=1 → v=1). Total penalty == 3.

   The model stays FEASIBLE (no hard clause forbids anything).
   We Maximize(sum of penalty vars) to confirm the solver achieves 3.

2. **Clean — byes spaced > S apart, penalty == 0.**
   Same fixture (R=10, byes=3, S=2). Force one team to bye at
   {1, 4, 8} — all gaps ≥ 3 > S=2. No pair has gap ≤ 2.
   After solving: total penalty == 0.

3. **Slack clamps S to 0 — atom returns 0, no penalty vars.**
   Same fixture. Set ``constraint_slack['BalancedByeSpacing'] = 2``
   (== ideal_bye_gap(10,3)==2). S = 2 - 2 = 0 → atom returns 0 terms,
   bucket stays empty (or doesn't exist).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import pytest
from ortools.sat.python import cp_model

from constraints.atoms.balanced_bye_spacing_regen_soft import (
    BalancedByeSpacingRegenSoft,
    REGEN_BALANCED_BYE_SPACING_DEFAULT_WEIGHT,
)
from constraints.atoms._spacing import ideal_bye_gap
from constraints.helper_vars import HelperVarRegistry
from constraints.atoms.base import BROADMEADOW
from models import Club, Grade, PlayingField, Team, Timeslot

# Re-export these helpers from the hard-atom test module so we share exactly
# the same fixture-building logic without duplicating it.
from tests.atoms.test_balanced_bye_spacing import (
    _make_grade_fixture,
    _build_X,
    _enforce_one_game_per_team_per_round,
    _pin_team_to_rounds,
)


# ---------------------------------------------------------------------------
# Shared tiny helpers
# ---------------------------------------------------------------------------


def _registry(model):
    return HelperVarRegistry(model)


def _solve(model, seconds: float = 10.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


# ---------------------------------------------------------------------------
# Scenario 1 — violation: penalty BoolVars fire when byes are too close
# ---------------------------------------------------------------------------


class TestViolationCounted:
    """6-team grade, R=10, games_per_team=7 → byes_per_team=3.

    Hand oracle:
      ideal_bye_gap(10, 3) = 10//3 - 1 = 3 - 1 = 2   S = 2

    Target team byes forced at rounds {1, 2, 3} (all gaps <= S=2):
      (1,2) gap=1 ≤ 2 → violation
      (1,3) gap=2 ≤ 2 → violation
      (2,3) gap=1 ≤ 2 → violation
    → 3 penalty BoolVars, each forced to 1 by encoding constraints.

    Other teams are unconstrained, so the solver can satisfy any leftover
    scheduling. Model is FEASIBLE and maximising the penalty sum yields 3.
    """

    GRADE = '3rd'
    R = 10
    GPT = 7           # games per team → 3 byes per team
    BYE_ROUNDS = (1, 2, 3)

    def _setup(self):
        data = _make_grade_fixture(
            grade_name=self.GRADE,
            num_teams=6,
            num_rounds=self.R,
            games_per_team=self.GPT,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        return data, model, X

    def test_feasible_and_penalty_count(self):
        """Forcing byes at {1,2,3} keeps the model FEASIBLE (soft, not hard)
        and the solver achieves total_penalty == 3 when maximising it.

        Hand oracle:
          S = ideal_bye_gap(10, 3) = 10//3 - 1 = 2
          Pairs with gap <= 2: (1,2), (1,3), (2,3) → 3 violations for target team.
          (Other teams are unconstrained; their byes might add more penalty vars
          but we verify only the lower bound by checking the penalty bucket size
          and that the target team's 3 violations all fire.)
        """
        # Verify formula used in comment above.
        assert ideal_bye_gap(self.R, self.R - self.GPT) == 2, (
            f"ideal_bye_gap({self.R}, {self.R - self.GPT}) should be 2"
        )

        data, model, X = self._setup()
        target = data['teams'][0].name  # 'C0 3rd'
        play_rounds = [r for r in range(1, self.R + 1) if r not in self.BYE_ROUNDS]
        # Force target team to play in exactly play_rounds → bye in BYE_ROUNDS.
        _pin_team_to_rounds(model, X, target, self.GRADE, play_rounds)

        atom = BalancedByeSpacingRegenSoft()
        n_terms = atom.apply(model, X, data, _registry(model))

        # Penalty bucket must exist and hold the right weight.
        bucket = data.get('penalties', {}).get('regen_balanced_bye_spacing', {})
        assert bucket, "Penalty bucket must be created"
        assert bucket['weight'] == REGEN_BALANCED_BYE_SPACING_DEFAULT_WEIGHT
        assert len(bucket['penalties']) == n_terms

        # EXACT per-team oracle (Mode B review fix): isolate the TARGET team's
        # penalty vars by their label `bye_soft_pen_{team}_{grade}_r{r1}_r{r2}`.
        # The atom emits one var per round-pair within S (a var is 1 only when
        # BOTH rounds are byes). The target byes {1,2,3} with S=2, so EXACTLY the
        # three pairs (1,2),(1,3),(2,3) have both-bye → resolve to 1; every other
        # target var has at least one non-bye round → resolves to 0. So the
        # target team's penalty vars SUM to exactly 3, regardless of how the rest
        # of the schedule (other teams) is solved — a precise per-team oracle.
        label_prefix = f'bye_soft_pen_{target}_{self.GRADE}_'
        target_pen_vars = [
            v for v in bucket['penalties'] if v.Name().startswith(label_prefix)
        ]
        assert target_pen_vars, "Target team must have bye penalty vars"

        # Model stays FEASIBLE (soft atom) and the target's bye-pair vars sum to 3.
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Model must stay FEASIBLE (soft atom); got "
            f"{cp_model.CpSolver().status_name(status)}"
        )
        target_total = sum(solver.Value(v) for v in target_pen_vars)
        assert target_total == 3, (
            f"Target team's byes at {{1,2,3}} → exactly 3 too-close bye pairs "
            f"fire (==3); got {target_total}"
        )

    def test_no_hard_infeasibility(self):
        """Even with byes jammed into {1,2,3} the model must NOT become
        INFEASIBLE — that's the defining property of a soft atom.
        """
        data, model, X = self._setup()
        target = data['teams'][0].name
        play_rounds = [r for r in range(1, self.R + 1) if r not in self.BYE_ROUNDS]
        _pin_team_to_rounds(model, X, target, self.GRADE, play_rounds)

        atom = BalancedByeSpacingRegenSoft()
        atom.apply(model, X, data, _registry(model))

        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Soft atom must never make a model INFEASIBLE; got "
            f"{cp_model.CpSolver().status_name(status)}"
        )


# ---------------------------------------------------------------------------
# Scenario 2 — clean: well-spaced byes yield penalty == 0
# ---------------------------------------------------------------------------


class TestCleanNoPenalty:
    """6-team grade, R=10, games_per_team=7 → byes_per_team=3, S=2.

    Byes forced at {1, 4, 8}:
      (1,4) gap=3 > 2  ✓ OK
      (1,8) gap=7 > 2  ✓ OK
      (4,8) gap=4 > 2  ✓ OK
    → zero penalty terms for this team.
    After minimising (or just solving), total penalty == 0.
    """

    GRADE = '3rd'
    R = 10
    GPT = 7
    BYE_ROUNDS = (1, 4, 8)

    def test_no_penalty_when_byes_spaced(self):
        data = _make_grade_fixture(
            grade_name=self.GRADE,
            num_teams=6,
            num_rounds=self.R,
            games_per_team=self.GPT,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)

        target = data['teams'][0].name
        play_rounds = [r for r in range(1, self.R + 1) if r not in self.BYE_ROUNDS]
        _pin_team_to_rounds(model, X, target, self.GRADE, play_rounds)

        atom = BalancedByeSpacingRegenSoft()
        atom.apply(model, X, data, _registry(model))

        bucket = data.get('penalties', {}).get('regen_balanced_bye_spacing', {})
        if not bucket or not bucket.get('penalties'):
            # Atom emitted no penalty terms at all — that's fine too if no
            # team has close byes.  Model is trivially FEASIBLE.
            status, _ = _solve(model)
            assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            return

        # Minimise penalty sum; the target team contributes 0 (byes far apart).
        # Other unconstrained teams may have penalty vars but those are free
        # to be 0 too.  The minimum achievable penalty is 0.
        model.Minimize(sum(bucket['penalties']))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Model must be FEASIBLE; got "
            f"{cp_model.CpSolver().status_name(status)}"
        )
        total_penalty = solver.ObjectiveValue()
        assert total_penalty == 0, (
            f"Well-spaced byes at {{1,4,8}} must yield zero total penalty; "
            f"got {total_penalty}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — slack clamps S to 0, atom returns 0 terms
# ---------------------------------------------------------------------------


class TestSlackDisables:
    """With slack == ideal_bye_gap, S clamps to 0 and the atom is a no-op.

    Hand: R=10, byes=3, ideal_bye_gap(10,3)=2. Slack=2 → S = 2-2 = 0 →
    atom returns 0, penalty bucket stays absent.
    """

    GRADE = '3rd'
    R = 10
    GPT = 7

    def test_slack_clamps_S_to_zero(self):
        data = _make_grade_fixture(
            grade_name=self.GRADE,
            num_teams=6,
            num_rounds=self.R,
            games_per_team=self.GPT,
        )
        # Slack == S_base → S = 0 → atom disabled.
        data['constraint_slack'] = {'BalancedByeSpacing': 2}
        model = cp_model.CpModel()
        X = _build_X(model, data)

        n_terms = BalancedByeSpacingRegenSoft().apply(
            model, X, data, _registry(model)
        )
        assert n_terms == 0, (
            f"Slack=2 must clamp S to 0 and disable the atom; got {n_terms} terms"
        )
        # Bucket must not have been populated.
        bucket = data.get('penalties', {}).get('regen_balanced_bye_spacing', None)
        assert bucket is None or not bucket.get('penalties'), (
            "No penalty vars should exist when atom is disabled"
        )

    def test_weight_zero_disables_atom(self):
        """Setting penalty_weights['regen_balanced_bye_spacing']=0 short-circuits
        the atom before any bucket or vars are created.
        """
        data = _make_grade_fixture(
            grade_name=self.GRADE,
            num_teams=6,
            num_rounds=self.R,
            games_per_team=self.GPT,
        )
        data['penalty_weights'] = {'regen_balanced_bye_spacing': 0}
        model = cp_model.CpModel()
        X = _build_X(model, data)

        n_terms = BalancedByeSpacingRegenSoft().apply(
            model, X, data, _registry(model)
        )
        assert n_terms == 0, (
            f"weight=0 must disable atom immediately; got {n_terms} terms"
        )
        assert 'regen_balanced_bye_spacing' not in data.get('penalties', {}), (
            "Bucket must not be created when weight==0"
        )
