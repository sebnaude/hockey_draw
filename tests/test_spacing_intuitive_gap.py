"""spec-008 Part A — intuitive `gap` semantics for matchup spacing.

Real CP-SAT models, no mocks. Each scenario hand-computes the expected
threshold and the legacy threshold so the parity invariant
("physical schedule unchanged at default slack") is visible in the test
body itself.

Semantics under spec-008:

    S = effective_spacing(T, base_slack, config_slack)
    HARD: forbid every pair of repeat-meeting rounds (r1, r2) with
          gap = r2 - r1 <= S.

For T=10 at default slack:
    legacy_min_gap(10) = max(1, 9 - max(1, 9 - 6)) = 6  (legacy forbade gap < 6)
    ideal_gap(10)      = legacy_min_gap(10) - 1     = 5  (new forbids gap <= 5)
    => forbidden set is the same {1..5}; physical schedule unchanged.

For an input of the convenor-facing S=2 the new rule must forbid:
    gap=1 (r2 = r1+1) AND gap=2 (r2 = r1+2) — the spec example.
The old rule with min_gap=2 only forbade gap=1.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

from constraints.atoms._spacing import (
    _legacy_min_gap,
    effective_spacing,
    ideal_gap,
)


# ----------------------------------------------------------------------
# Helpers — minimal grade fixtures so we can call effective_spacing
# directly and exercise the gap math against a small CP-SAT model.
# ----------------------------------------------------------------------


def _round_robin_pair_vars(
    model: cp_model.CpModel, num_rounds: int
) -> Dict[int, cp_model.IntVar]:
    """Return {round_no: BoolVar} for a single pair across `num_rounds`."""
    return {
        r: model.NewBoolVar(f'pair_r{r}')
        for r in range(1, num_rounds + 1)
    }


def _apply_pairwise_spacing(
    model: cp_model.CpModel,
    pair_vars: Dict[int, cp_model.IntVar],
    S: int,
) -> int:
    """Encode the spec-008 hard rule: forbid (r1, r2) when r2-r1 <= S.

    Returns the number of pairwise constraints emitted. Mirrors the
    branch inside `unified.py::_matchup_spacing_hard` so the test can
    confirm the threshold without needing the whole engine fixture.
    """
    if S <= 0:
        return 0
    rounds = sorted(pair_vars.keys())
    n = 0
    for i, r1 in enumerate(rounds):
        for r2 in rounds[i + 1:]:
            gap = r2 - r1
            if gap > S:
                break
            model.Add(pair_vars[r1] + pair_vars[r2] <= 1)
            n += 1
    return n


def _solve(model: cp_model.CpModel):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    return solver.Solve(model), solver


# ----------------------------------------------------------------------
# Scenario A1: ideal_gap is exactly legacy_min_gap - 1 across grade sizes
# ----------------------------------------------------------------------


class TestIdealGapMatchesLegacyMinus1:
    """The math invariant that guarantees "physical schedule unchanged"."""

    @pytest.mark.parametrize('T', [3, 4, 5, 6, 7, 8, 9, 10, 12])
    def test_ideal_gap_is_legacy_min_gap_minus_one(self, T):
        """Hand: legacy_min_gap and ideal_gap must differ by exactly 1.

        This is the entire off-by-one fix: the new hard rule `gap <= S`
        coincides exactly with the old `gap < legacy_min_gap` when
        ``S = legacy_min_gap - 1``.
        """
        legacy = _legacy_min_gap(T)
        new = ideal_gap(T)
        assert new == legacy - 1, (
            f"T={T}: ideal_gap({T})={new} should be legacy_min_gap-1={legacy-1}"
        )

    def test_ideal_gap_is_zero_for_small_grades(self):
        """T<3 has no meaningful spacing."""
        assert ideal_gap(0) == 0
        assert ideal_gap(1) == 0
        assert ideal_gap(2) == 0
        assert _legacy_min_gap(0) == 0
        assert _legacy_min_gap(2) == 0


# ----------------------------------------------------------------------
# Scenario A2: S=2 forbids gap=1 AND gap=2 (the spec example)
# ----------------------------------------------------------------------


class TestSpacingTwoForbidsGapOneAndTwo:
    """Spec-008 quote: 'S=2 should forbid (r1, r2=r1+1) and (r1, r2=r1+2)'.

    The old code with min_gap=2 only forbade gap=1.
    """

    def test_s_two_forbids_gap_one(self):
        """Hand: with S=2, forcing pair at rounds 1 and 2 is INFEASIBLE.

        Old rule (`gap < 2`) would also forbid this — but that's the easy
        case. The interesting case is gap=2 (below).
        """
        model = cp_model.CpModel()
        pair = _round_robin_pair_vars(model, num_rounds=6)
        _apply_pairwise_spacing(model, pair, S=2)
        model.Add(pair[1] == 1)
        model.Add(pair[2] == 1)
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            f"S=2, gap=1 must be INFEASIBLE, got {cp_model.CpSolver().status_name(status)}"
        )

    def test_s_two_forbids_gap_two(self):
        """Hand: with S=2, forcing pair at rounds 1 and 3 must be INFEASIBLE.

        This is the off-by-one fix in action — gap=2 satisfies the OLD
        rule `gap >= min_gap=2` but VIOLATES the new rule `gap > S=2`.
        Convenor reads S=2 as "two rounds of breathing room between
        meetings"; rounds 1 and 3 have only one round (round 2) in
        between, so it's correctly forbidden under the new math.
        """
        model = cp_model.CpModel()
        pair = _round_robin_pair_vars(model, num_rounds=6)
        _apply_pairwise_spacing(model, pair, S=2)
        model.Add(pair[1] == 1)
        model.Add(pair[3] == 1)
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            f"S=2, gap=2 must be INFEASIBLE under new semantics; "
            f"got {cp_model.CpSolver().status_name(status)}"
        )

    def test_s_two_allows_gap_three(self):
        """Hand: with S=2, rounds 1 and 4 (gap=3) are FEASIBLE.

        Two rounds (2 and 3) between meetings — convenor's expectation
        is satisfied.
        """
        model = cp_model.CpModel()
        pair = _round_robin_pair_vars(model, num_rounds=6)
        _apply_pairwise_spacing(model, pair, S=2)
        model.Add(pair[1] == 1)
        model.Add(pair[4] == 1)
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"S=2, gap=3 must be FEASIBLE; "
            f"got {cp_model.CpSolver().status_name(status)}"
        )


# ----------------------------------------------------------------------
# Scenario A3: physical-schedule-unchanged invariant for T=10 grade
# ----------------------------------------------------------------------


class TestPhysicalScheduleUnchangedAtDefaultSlack:
    """At default slack, every gap the legacy code accepted is still accepted,
    and every gap the legacy code rejected is still rejected.

    For T=10:
      legacy_min_gap = 6  =>  legacy forbids gaps {1, 2, 3, 4, 5}
      ideal_gap      = 5  =>  new forbids gaps {1, 2, 3, 4, 5}
    Identical.
    """

    @pytest.mark.parametrize('gap', [1, 2, 3, 4, 5])
    def test_forbidden_gaps_unchanged(self, gap):
        """Hand: every gap in {1..5} must be INFEASIBLE under new semantics."""
        T = 10
        S = effective_spacing(T)  # default slack
        assert S == 5, f"sanity: ideal_gap(10)=5, got {S}"
        model = cp_model.CpModel()
        pair = _round_robin_pair_vars(model, num_rounds=15)
        _apply_pairwise_spacing(model, pair, S=S)
        model.Add(pair[1] == 1)
        model.Add(pair[1 + gap] == 1)
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            f"T=10, gap={gap}: legacy forbade, new must forbid; got "
            f"{cp_model.CpSolver().status_name(status)}"
        )

    @pytest.mark.parametrize('gap', [6, 7, 8, 9])
    def test_allowed_gaps_unchanged(self, gap):
        """Hand: every gap in {6..9} was FEASIBLE under legacy, must remain so."""
        T = 10
        S = effective_spacing(T)
        model = cp_model.CpModel()
        pair = _round_robin_pair_vars(model, num_rounds=20)
        _apply_pairwise_spacing(model, pair, S=S)
        model.Add(pair[1] == 1)
        model.Add(pair[1 + gap] == 1)
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"T=10, gap={gap}: legacy allowed, new must allow; got "
            f"{cp_model.CpSolver().status_name(status)}"
        )


# ----------------------------------------------------------------------
# Scenario A4: slack semantics — K loosens S by K
# ----------------------------------------------------------------------


class TestSlackLoosensByExactlyK:
    """DoD #4 + #6: slack `K` means convenor accepts gap shrinking by up to
    `K` rounds from ideal. The CLI semantics are unchanged from the
    convenor's perspective.
    """

    def test_slack_zero_matches_ideal(self):
        """Hand: ideal_gap(8) = legacy_min_gap(8) - 1.

        legacy: ideal_distance=7, hardcoded_slack=max(1, 7 - 4)=3,
        legacy_min_gap=max(1, 7-3)=4. ideal_gap=3.
        """
        assert effective_spacing(8) == 3

    def test_slack_one_loosens_by_one(self):
        """Hand: S(T=8, K=1) = ideal_gap(8) - 1 = 2."""
        assert effective_spacing(8, config_slack=1) == 2

    def test_slack_two_loosens_by_two(self):
        assert effective_spacing(8, config_slack=2) == 1

    def test_slack_enough_to_disable(self):
        """Hand: at slack=3 for T=8, S clamps to 0 (no spacing enforced)."""
        assert effective_spacing(8, config_slack=3) == 0

    def test_base_slack_and_config_slack_stack(self):
        """Hand: base + config both subtract from ideal_gap."""
        assert effective_spacing(10, base_slack=2, config_slack=1) == 5 - 3

    def test_negative_net_slack_tightens(self):
        """Forced-rounds adjuster surfaces as negative slack — S grows above
        ideal_gap. Hand: ideal_gap(10)=5; net_slack=-2 ⇒ S=7."""
        assert effective_spacing(10, base_slack=-2) == 7
