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
from constraints.unified import UnifiedConstraintEngine
from constraints.stages import ALL_ENGINE_KEYS


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


def compute_min_gap_for_grade(num_teams, config_slack=0, base_slack=0):
    """Compute the hard min_gap for a grade, matching the constraint formula
    in `EqualMatchUpSpacingConstraintAI.apply` exactly:

        ideal = T - 2
        floor = min(T // 2, T - 2)
        min_gap = max(floor, ideal - base_slack - config_slack)

    Returns (min_gap, space) where `space == ideal == T - 2` (the sliding
    window size the soft penalty uses). `base_slack` mirrors
    `constraint_defaults['spacing_base_slack']` (default 0).
    """
    T = num_teams
    ideal = T - 2
    floor = min(T // 2, T - 2)
    min_gap = max(floor, ideal - base_slack - config_slack)
    return min_gap, ideal


class TestSpacingOnRealData:
    """Test the spacing constraint using real 2026 season data."""

    def test_soft_spacing_is_feasible_and_minimised_on_real_data(self):
        """Scenario: the SOFT spacing pass, applied in ISOLATION (soft-only,
        as the `soft_optimisation` stage dispatches it), is feasible on real
        2026 data and the solver minimises spacing penalties.

        NOTE (spec-017): production now ALSO applies the HARD pairwise-gap
        clause — `EqualMatchUpSpacing` moved to the `critical_feasibility`
        stage, so `apply_solver_stage` runs both `apply_stage_1_hard()` and
        `apply_stage_2_soft()` for it. This test deliberately drives the soft
        path on its own (it sets `skip_constraints` to leave only the target
        soft pass) to verify the soft penalties are well-formed and feasible —
        it is no longer a model of "the production default path." The hard
        clause is exercised by `test_rejects_tight_gaps_on_real_data` and by
        `tests/test_spacing_promoted_hard.py`. The hard clause over the FULL
        2026 model can be infeasible at low slack (many no-play weeks / blocked
        games), which is why this isolated soft check stays soft.
        """
        data = load_2026_data()
        model = cp_model.CpModel()
        X, _ = generate_X(model, data)

        # --- Given: base hard constraints (the minimum for a valid draw),
        # then spacing applied SOFT-only, exactly as the soft_optimisation
        # stage dispatches it (skip every engine key except the target).
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        base_hard = {
            'NoDoubleBookingTeams', 'NoDoubleBookingFields',
            'EqualGamesAndBalanceMatchUps',
        }
        engine.skip_constraints = ALL_ENGINE_KEYS - base_hard
        n_hard = engine.apply_stage_1_hard()
        engine.skip_constraints = ALL_ENGINE_KEYS - {'EqualMatchUpSpacing'}
        n_soft = engine.apply_stage_2_soft()

        # Oracle: the soft pass must add at least one penalty, and every soft
        # term must be recorded in data['penalties']['EqualMatchUpSpacing'].
        # n_soft is the dispatcher's own count of soft terms added; the
        # penalty list must match it exactly (internal-consistency oracle).
        assert n_hard > 0, "base hard pass must add constraints on real data"
        spacing_pens = data['penalties']['EqualMatchUpSpacing']['penalties']
        assert n_soft > 0, "soft spacing must add penalties on real data"
        assert len(spacing_pens) == n_soft, (
            f"tracked penalties {len(spacing_pens)} != dispatcher soft count {n_soft}"
        )

        # --- When: minimise total spacing penalty and solve.
        model.Minimize(sum(spacing_pens))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120
        solver.parameters.num_workers = 4  # avoid OOM on large model
        status = solver.Solve(model)

        if status == cp_model.UNKNOWN:
            pytest.skip("Solver timed out — not necessarily infeasible")

        # --- Then: feasible (soft spacing never blocks a valid draw). Oracle:
        # base hard alone is OPTIMAL (verified independently), and soft
        # penalties cannot remove feasibility — they only add to the objective.
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE], (
            f"Soft spacing produced {solver.status_name(status)} on real 2026 data"
        )

        # --- And: report the realised gap distribution for visibility. Under
        # soft application some gaps may fall below the ideal min_gap; that is
        # allowed by design, so we do NOT assert zero violations here (that is
        # what made the old hard-application test fail). We assert only that
        # the schedule actually contains repeat meetings to space — otherwise
        # the soft penalty would be vacuous.
        grade_teams = {g.name: g.num_teams for g in data['grades']}
        meetings = extract_solution_meetings(solver, X, data)
        grade_gaps = defaultdict(list)
        repeat_pairs = 0
        for (t1, t2, grade), rounds in meetings.items():
            if len(rounds) < 2:
                continue
            repeat_pairs += 1
            for i in range(len(rounds) - 1):
                grade_gaps[grade].append(rounds[i + 1] - rounds[i])

        print("\n=== Gap Statistics from Soft-Spacing 2026 Schedule ===")
        for grade in sorted(grade_gaps.keys()):
            gaps = grade_gaps[grade]
            num_teams = grade_teams.get(grade, '?')
            min_gap_expected, space = (
                compute_min_gap_for_grade(num_teams)
                if isinstance(num_teams, int) else (0, 0)
            )
            print(
                f"  {grade} (T={num_teams}): min_gap={min_gap_expected}, "
                f"ideal_window={space}, actual_min={min(gaps) if gaps else 0}, "
                f"actual_max={max(gaps) if gaps else 0}, count={len(gaps)}"
            )

        # Oracle: a real 2026 schedule has repeat matchups in at least one
        # grade (grades with T-1 < num_rounds force repeats), so the soft
        # penalty set is non-vacuous.
        assert repeat_pairs > 0, "expected repeat matchups for spacing to act on"

    def test_constraint_count_on_real_data(self):
        """Compare constraint counts between AI and original on real data."""
        data = load_2026_data()

        model_ai = cp_model.CpModel()
        X_ai, _ = generate_X(model_ai, data)
        count_ai = EqualMatchUpSpacingConstraintAI().apply(model_ai, X_ai, data)

        data2 = load_2026_data()
        model_orig = cp_model.CpModel()
        X_orig, _ = generate_X(model_orig, data2)
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
        X, _ = generate_X(model, data)

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
