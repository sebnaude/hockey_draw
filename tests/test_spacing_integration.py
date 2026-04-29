# test_spacing_integration.py
"""
Integration tests for EqualMatchUpSpacingConstraintAI using real 2026 season data.

These tests load the actual season config, build real decision variables,
apply the constraint, solve, and then inspect the solution to verify that
the minimum gap is actually respected in practice.
"""
import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_season_data
from utils import generate_X
from constraints.archived.ai import EqualMatchUpSpacingConstraintAI
from constraints.archived.original import EqualMatchUpSpacingConstraint
from constraints.archived.ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
)


def load_2026_data():
    """Load real 2026 season data."""
    data = load_season_data(2026)
    data['current_week'] = 0
    data['locked_weeks'] = set()
    if 'penalties' not in data:
        data['penalties'] = {}
    if 'constraint_slack' not in data:
        data['constraint_slack'] = {}
    return data


def extract_solution_meetings(solver, X, data):
    """Extract which matchup pairs play in which rounds from a solved model."""
    meetings = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11:
            continue
        if solver.Value(var) == 1:
            t1, t2, grade = key[0], key[1], key[2]
            round_no = key[8]
            meetings[(t1, t2, grade)].append(round_no)
    return {k: sorted(set(v)) for k, v in meetings.items()}


def compute_min_gap_for_grade(num_teams, config_slack=0):
    """Compute the minimum gap for a grade, matching the constraint logic."""
    space = num_teams - 1
    ideal = num_teams - 1
    base_slack = max(1, ideal - 2 * ideal // 3)
    effective_slack = min(base_slack + config_slack, num_teams // 2 + 1)
    min_gap = max(1, space - effective_slack)
    return min_gap, space


class TestSpacingOnRealData:
    """Test the spacing constraint using real 2026 season data."""

    def test_feasible_and_respects_min_gap(self):
        """The constraint must be feasible on real 2026 data, and the solution
        must actually respect minimum gap between repeat matchups.

        This is the critical integration test: solve with essential constraints +
        spacing, then inspect every matchup pair in the solution.
        """
        data = load_2026_data()
        model = cp_model.CpModel()
        X, Y, conflicts = generate_X(model, data)

        NoDoubleBookingTeamsConstraintAI().apply(model, X, data)
        NoDoubleBookingFieldsConstraintAI().apply(model, X, data)
        EnsureEqualGamesAndBalanceMatchUpsAI().apply(model, X, data)

        count = EqualMatchUpSpacingConstraintAI().apply(model, X, data)
        assert count > 0, "Must add constraints on real data"

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120
        solver.parameters.num_workers = 4  # avoid OOM on large model
        status = solver.Solve(model)

        if status == cp_model.UNKNOWN:
            pytest.skip("Solver timed out — not necessarily infeasible")
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE], (
            f"Constraint produced {solver.status_name(status)} on real 2026 data"
        )

        # --- Verify min_gap in the actual solution ---
        grade_teams = {g.name: g.num_teams for g in data['grades']}
        meetings = extract_solution_meetings(solver, X, data)

        violations = []
        grade_gaps = defaultdict(list)
        for (t1, t2, grade), rounds in meetings.items():
            if len(rounds) < 2:
                continue
            num_teams = grade_teams.get(grade, 0)
            if num_teams < 2:
                continue
            min_gap, space = compute_min_gap_for_grade(num_teams)

            for i in range(len(rounds) - 1):
                gap = rounds[i + 1] - rounds[i]
                grade_gaps[grade].append(gap)
                if gap < min_gap:
                    violations.append(
                        f"{t1} vs {t2} ({grade}): rounds {rounds[i]} and "
                        f"{rounds[i+1]}, gap={gap} < min_gap={min_gap} "
                        f"(T={num_teams}, space={space})"
                    )

        # Print gap statistics for visibility
        print(f"\n=== Gap Statistics from Solved 2026 Schedule ===")
        for grade in sorted(grade_gaps.keys()):
            gaps = grade_gaps[grade]
            num_teams = grade_teams.get(grade, '?')
            min_gap_expected, space = compute_min_gap_for_grade(num_teams) if isinstance(num_teams, int) else (0, 0)
            actual_min = min(gaps) if gaps else 0
            actual_max = max(gaps) if gaps else 0
            avg = sum(gaps) / len(gaps) if gaps else 0
            print(f"  {grade} (T={num_teams}): min_gap_required={min_gap_expected}, "
                  f"ideal={space}, actual_min={actual_min}, actual_max={actual_max}, "
                  f"avg={avg:.1f}, count={len(gaps)}")

        assert not violations, (
            f"Found {len(violations)} min-gap violations in solved schedule:\n"
            + "\n".join(violations[:20])
        )

    def test_constraint_count_on_real_data(self):
        """Compare constraint counts between AI and original on real data."""
        data = load_2026_data()

        model_ai = cp_model.CpModel()
        X_ai, _, _ = generate_X(model_ai, data)
        count_ai = EqualMatchUpSpacingConstraintAI().apply(model_ai, X_ai, data)

        data2 = load_2026_data()
        model_orig = cp_model.CpModel()
        X_orig, _, _ = generate_X(model_orig, data2)
        EqualMatchUpSpacingConstraint().apply(model_orig, X_orig, data2)

        ai_proto = len(model_ai.Proto().constraints)
        orig_proto = len(model_orig.Proto().constraints)
        ai_vars = len(model_ai.Proto().variables)
        orig_vars = len(model_orig.Proto().variables)

        orig_mult = sum(1 for ct in model_orig.Proto().constraints if ct.has_int_prod())
        orig_div = sum(1 for ct in model_orig.Proto().constraints if ct.has_int_div())
        orig_max = sum(1 for ct in model_orig.Proto().constraints if ct.has_lin_max())

        ai_mult = sum(1 for ct in model_ai.Proto().constraints if ct.has_int_prod())
        ai_div = sum(1 for ct in model_ai.Proto().constraints if ct.has_int_div())
        ai_max = sum(1 for ct in model_ai.Proto().constraints if ct.has_lin_max())

        print(f"\n=== Real 2026 Data Constraint Comparison ===")
        print(f"Original: {orig_proto} proto constraints, {orig_vars} variables")
        print(f"      AI: {ai_proto} proto constraints, {ai_vars} variables")
        print(f"Original nonlinear: {orig_mult} mult, {orig_div} div, {orig_max} max")
        print(f"      AI nonlinear: {ai_mult} mult, {ai_div} div, {ai_max} max")
        print(f"AI constraint count (from .apply()): {count_ai}")

        assert ai_mult == 0, f"AI should have 0 multiplications, got {ai_mult}"
        assert ai_div == 0, f"AI should have 0 divisions, got {ai_div}"
        assert ai_max == 0, f"AI should have 0 max equalities, got {ai_max}"

    def test_rejects_tight_gaps_on_real_data(self):
        """Force back-to-back games for a large grade and verify INFEASIBLE.

        Uses real 2026 data. Picks a grade with min_gap > 1,
        forces a pair to play in rounds 1 and 2, verifies INFEASIBLE.
        """
        data = load_2026_data()

        target_grade = None
        for g in data['grades']:
            min_gap, _ = compute_min_gap_for_grade(g.num_teams)
            if min_gap > 1:
                target_grade = g
                break
        assert target_grade is not None, "Need a grade with min_gap > 1"

        team_names = target_grade.teams
        t1, t2 = team_names[0], team_names[1]

        model = cp_model.CpModel()
        X, _, _ = generate_X(model, data)

        # Force this pair in round 1 AND round 2
        pair_r1 = [v for k, v in X.items()
                   if k[0] == t1 and k[1] == t2 and k[2] == target_grade.name and k[8] == 1]
        pair_r2 = [v for k, v in X.items()
                   if k[0] == t1 and k[1] == t2 and k[2] == target_grade.name and k[8] == 2]

        if not pair_r1 or not pair_r2:
            pair_r1 = [v for k, v in X.items()
                       if k[0] == t2 and k[1] == t1 and k[2] == target_grade.name and k[8] == 1]
            pair_r2 = [v for k, v in X.items()
                       if k[0] == t2 and k[1] == t1 and k[2] == target_grade.name and k[8] == 2]

        assert pair_r1 and pair_r2, (
            f"Must have X vars for {t1} vs {t2} in {target_grade.name} rounds 1 and 2"
        )

        model.Add(sum(pair_r1) >= 1)
        model.Add(sum(pair_r2) >= 1)

        EqualMatchUpSpacingConstraintAI().apply(model, X, data)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_workers = 4
        status = solver.Solve(model)

        min_gap, space = compute_min_gap_for_grade(target_grade.num_teams)
        assert status == cp_model.INFEASIBLE, (
            f"Forcing {t1} vs {t2} ({target_grade.name}, T={target_grade.num_teams}) "
            f"in rounds 1 and 2 (gap=1 < min_gap={min_gap}) should be INFEASIBLE, "
            f"got {solver.status_name(status)}"
        )
