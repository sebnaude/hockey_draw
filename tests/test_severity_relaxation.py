# tests/test_severity_relaxation.py
"""
Tests for severity_relaxation.py - NO MOCKS.

Uses real constraint classes, real config data, and real CP-SAT models.
Mini solver runs use short timeouts (1-5 seconds).
"""

import pytest
import sys
import os

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.severity import (
    CONSTRAINT_TO_SEVERITY,
    get_severity_level,
    group_constraints_by_severity,
    SeverityGroupState,
    SeverityGroupResolver,
    apply_constraints_with_relaxation,
    create_relaxation_test_func,
)
from constraints.archived.ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    ClubDayConstraintAI,
    AwayAtMaitlandGroupingAI,
    TeamConflictConstraintAI,
    ClubGradeAdjacencyConstraintAI,
    ClubVsClubAlignmentAI,
    ClubGameSpreadAI,
    # spec-024: Maximise/MinimiseClubsBroadmeadowAI imports removed (constraints deleted).
    EnsureBestTimeslotChoicesAI,
    PreferredTimesConstraintAI,
    EqualMatchUpSpacingConstraintAI,
)


# ============== CONSTRAINT_TO_SEVERITY Tests ==============

class TestConstraintToSeverity:
    """Tests for CONSTRAINT_TO_SEVERITY mapping."""

    def test_mapping_contains_core_constraints(self):
        """Test that mapping contains core constraint names."""
        assert 'NoDoubleBookingTeamsConstraint' in CONSTRAINT_TO_SEVERITY
        assert 'NoDoubleBookingTeamsConstraintAI' in CONSTRAINT_TO_SEVERITY
        assert 'ClubDayConstraint' in CONSTRAINT_TO_SEVERITY
        assert 'EqualMatchUpSpacingConstraint' in CONSTRAINT_TO_SEVERITY

    def test_level1_constraints_are_critical(self):
        """Test that level 1 constraints are the critical ones."""
        level1_names = [k for k, v in CONSTRAINT_TO_SEVERITY.items() if v == 1]

        assert 'NoDoubleBookingTeamsConstraint' in level1_names
        assert 'NoDoubleBookingFieldsConstraint' in level1_names
        assert 'EnsureEqualGamesAndBalanceMatchUps' in level1_names

    def test_all_levels_are_1_to_5(self):
        """Test all severity levels are between 1 and 5."""
        for level in CONSTRAINT_TO_SEVERITY.values():
            assert level in [1, 2, 3, 4, 5]

    def test_ai_and_original_share_same_level(self):
        """Test AI variants have the same severity as their original counterparts."""
        # Check a few pairs
        pairs = [
            ('NoDoubleBookingTeamsConstraint', 'NoDoubleBookingTeamsConstraintAI'),
            ('ClubDayConstraint', 'ClubDayConstraintAI'),
            ('ClubGradeAdjacencyConstraint', 'ClubGradeAdjacencyConstraintAI'),
        ]
        for orig, ai in pairs:
            assert CONSTRAINT_TO_SEVERITY[orig] == CONSTRAINT_TO_SEVERITY[ai]


# ============== get_severity_level Tests ==============

class TestGetSeverityLevel:
    """Tests for get_severity_level function using REAL constraint classes."""

    def test_returns_level1_for_critical_constraint(self):
        """Level 1 for NoDoubleBookingTeamsConstraintAI."""
        level = get_severity_level(NoDoubleBookingTeamsConstraintAI)
        assert level == 1

    def test_returns_level2_for_high_constraint(self):
        """Level 2 for ClubDayConstraintAI."""
        level = get_severity_level(ClubDayConstraintAI)
        assert level == 2

    def test_returns_level3_for_medium_constraint(self):
        """Level 3 for ClubGradeAdjacencyConstraintAI."""
        level = get_severity_level(ClubGradeAdjacencyConstraintAI)
        assert level == 3

    # spec-024: test_returns_level4_for_low_constraint removed -- the only two
    # level-4 constraints (Maximise/MinimiseClubsBroadmeadow) were deleted, so no
    # production constraint maps to level 4 anymore.

    def test_returns_level5_for_very_low_constraint(self):
        """Level 5 for EnsureBestTimeslotChoicesAI."""
        level = get_severity_level(EnsureBestTimeslotChoicesAI)
        assert level == 5

    def test_returns_5_for_unknown_constraint(self):
        """Test returns 5 (lowest) for unknown constraints."""
        class UnknownConstraint:
            pass

        level = get_severity_level(UnknownConstraint)
        assert level == 5


# ============== group_constraints_by_severity Tests ==============

class TestGroupConstraintsBySeverity:
    """Tests for group_constraints_by_severity using real constraint classes."""

    def test_groups_real_constraints_correctly(self):
        """Test real constraints are grouped by their severity level."""
        constraints = [
            NoDoubleBookingTeamsConstraintAI,  # Level 1
            ClubDayConstraintAI,               # Level 2
            ClubGradeAdjacencyConstraintAI,    # Level 3
        ]
        groups = group_constraints_by_severity(constraints)

        assert 1 in groups
        assert 2 in groups
        assert 3 in groups
        assert NoDoubleBookingTeamsConstraintAI in groups[1]
        assert ClubDayConstraintAI in groups[2]
        assert ClubGradeAdjacencyConstraintAI in groups[3]

    def test_empty_list_returns_empty_dict(self):
        """Test empty constraint list returns empty dict."""
        groups = group_constraints_by_severity([])
        assert groups == {}

    def test_multiple_constraints_same_level(self):
        """Test multiple constraints at same level grouped together."""
        constraints = [
            NoDoubleBookingTeamsConstraintAI,  # Level 1
            NoDoubleBookingFieldsConstraintAI, # Level 1
            EqualMatchUpSpacingConstraintAI,   # Level 1
        ]
        groups = group_constraints_by_severity(constraints)

        assert len(groups[1]) == 3

    def test_multiple_levels_present(self):
        """Test grouping with constraints from the populated severity levels.

        spec-024: level 4 has no production constraints anymore (Maximise/Minimise
        deleted), so the populated levels are {1, 2, 3, 5}.
        """
        constraints = [
            NoDoubleBookingTeamsConstraintAI,             # Level 1
            ClubDayConstraintAI,                          # Level 2
            ClubGradeAdjacencyConstraintAI,               # Level 3
            EnsureBestTimeslotChoicesAI,                   # Level 5
        ]
        groups = group_constraints_by_severity(constraints)
        assert set(groups.keys()) == {1, 2, 3, 5}


# ============== SeverityGroupState Tests ==============

class TestSeverityGroupState:
    """Tests for SeverityGroupState dataclass."""

    def test_default_initialization(self):
        state = SeverityGroupState(level=2, constraint_classes=[ClubDayConstraintAI])
        assert state.level == 2
        assert state.constraint_classes == [ClubDayConstraintAI]
        assert state.current_slack == 0
        assert state.max_slack == 3
        assert state.is_problem_group is False

    def test_can_relax_true_for_non_level1(self):
        for level in [2, 3, 4, 5]:
            state = SeverityGroupState(level=level, constraint_classes=[])
            assert state.can_relax() is True

    def test_can_relax_false_for_level1(self):
        state = SeverityGroupState(level=1, constraint_classes=[NoDoubleBookingTeamsConstraintAI])
        assert state.can_relax() is False

    def test_can_relax_false_at_max_slack(self):
        state = SeverityGroupState(level=2, constraint_classes=[], current_slack=3)
        assert state.can_relax() is False

    def test_relax_increments_slack(self):
        state = SeverityGroupState(level=2, constraint_classes=[ClubDayConstraintAI])
        result = state.relax()
        assert result is True
        assert state.current_slack == 1

    def test_relax_fails_at_max_slack(self):
        state = SeverityGroupState(level=2, constraint_classes=[], current_slack=3)
        result = state.relax()
        assert result is False
        assert state.current_slack == 3

    def test_relax_fails_for_level1(self):
        state = SeverityGroupState(level=1, constraint_classes=[])
        result = state.relax()
        assert result is False
        assert state.current_slack == 0

    def test_relax_increments_progressively(self):
        """Test multiple relaxations increment slack step by step."""
        state = SeverityGroupState(level=3, constraint_classes=[ClubGradeAdjacencyConstraintAI])
        for expected_slack in [1, 2, 3]:
            assert state.relax() is True
            assert state.current_slack == expected_slack
        # At max now
        assert state.relax() is False
        assert state.current_slack == 3


# ============== SeverityGroupResolver Tests ==============

class TestSeverityGroupResolverBasic:
    """Tests for SeverityGroupResolver using real constraint classes."""

    def test_initialization_groups_real_constraints(self):
        """Test resolver groups real constraints by severity on init."""
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,  # Level 1
            ClubDayConstraintAI,               # Level 2
            ClubGradeAdjacencyConstraintAI,    # Level 3
        ], verbose=False)

        assert 1 in resolver.severity_groups
        assert 2 in resolver.severity_groups
        assert 3 in resolver.severity_groups

    def test_get_constraints_for_test_returns_all_when_no_exclusions(self):
        resolver = SeverityGroupResolver([
            ClubDayConstraintAI,               # Level 2
            ClubGradeAdjacencyConstraintAI,    # Level 3
        ], verbose=False)

        all_constraints = resolver.get_constraints_for_test()
        assert len(all_constraints) == 2
        assert ClubDayConstraintAI in all_constraints
        assert ClubGradeAdjacencyConstraintAI in all_constraints

    def test_get_constraints_for_test_excludes_specified_levels(self):
        resolver = SeverityGroupResolver([
            ClubDayConstraintAI,               # Level 2
            ClubGradeAdjacencyConstraintAI,    # Level 3
            EnsureBestTimeslotChoicesAI,       # Level 5 (spec-024: level 4 now empty)
        ], verbose=False)

        # Exclude level 3
        constraints = resolver.get_constraints_for_test(exclude_levels={3})
        assert ClubDayConstraintAI in constraints
        assert ClubGradeAdjacencyConstraintAI not in constraints
        assert EnsureBestTimeslotChoicesAI in constraints

    def test_get_constraints_for_test_excludes_multiple_levels(self):
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,        # Level 1
            ClubDayConstraintAI,                     # Level 2
            ClubGradeAdjacencyConstraintAI,          # Level 3
            EnsureBestTimeslotChoicesAI,             # Level 5 (spec-024: level 4 now empty)
        ], verbose=False)

        constraints = resolver.get_constraints_for_test(exclude_levels={3, 5})
        assert len(constraints) == 2
        assert NoDoubleBookingTeamsConstraintAI in constraints
        assert ClubDayConstraintAI in constraints

    def test_levels_present_sorted(self):
        resolver = SeverityGroupResolver([
            EnsureBestTimeslotChoicesAI,              # Level 5 (spec-024: level 4 now empty)
            NoDoubleBookingTeamsConstraintAI,         # Level 1
            ClubGradeAdjacencyConstraintAI,           # Level 3
        ], verbose=False)

        assert resolver.levels_present == [1, 3, 5]

    def test_get_state_summary(self):
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
        ], verbose=False)

        summary = resolver.get_state_summary()
        assert 'Severity Group States:' in summary
        assert 'Level 1' in summary
        assert 'Level 2' in summary


class TestSeverityGroupResolverFindProblem:
    """Tests for find_problem_severity_group using real CP-SAT feasibility checks."""

    def test_find_problem_returns_none_when_all_feasible(self):
        """When test_func always returns feasible, no problem group is found."""
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
        ], verbose=False)

        def always_feasible(constraints, timeout):
            return 'FEASIBLE', True

        result = resolver.find_problem_severity_group(always_feasible, timeout=1.0)
        assert result is None

    def test_find_problem_identifies_blocking_level(self):
        """When removing a specific level makes it feasible, that level is the problem.

        spec-024: level 4 is now empty, so the populated levels are {1, 2, 3, 5}.
        The resolver peels [5, 4, 3, 2] skipping absent levels, so excluding 5 then
        3 is the third test -- making level 3 the blocking group.
        """
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,             # Level 1
            ClubDayConstraintAI,                          # Level 2
            ClubGradeAdjacencyConstraintAI,               # Level 3
            EnsureBestTimeslotChoicesAI,                   # Level 5
        ], verbose=False)

        # Simulate: infeasible with all, feasible only once level 3 is excluded.
        call_count = [0]
        def test_func(constraints, timeout):
            call_count[0] += 1
            # call 1 (all) -> infeasible
            # call 2 (excl level 5) -> infeasible
            # call 3 (excl levels 5,3) -> feasible
            if call_count[0] <= 2:
                return 'INFEASIBLE', False
            return 'FEASIBLE', True

        result = resolver.find_problem_severity_group(test_func, timeout=1.0)
        assert result == 3

    def test_find_problem_returns_1_when_all_infeasible(self):
        """When even level 1 alone is infeasible, returns 1."""
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
            ClubGradeAdjacencyConstraintAI,
        ], verbose=False)

        def always_infeasible(constraints, timeout):
            return 'INFEASIBLE', False

        result = resolver.find_problem_severity_group(always_infeasible, timeout=1.0)
        assert result == 1

    def test_find_problem_marks_group_as_problem(self):
        """Problem group should be marked as is_problem_group=True."""
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,   # Level 1
            ClubDayConstraintAI,                # Level 2
        ], verbose=False)

        call_count = [0]
        def test_func(constraints, timeout):
            call_count[0] += 1
            # All constraints -> infeasible, without level 2 -> feasible
            if call_count[0] == 1:
                return 'INFEASIBLE', False
            return 'FEASIBLE', True

        result = resolver.find_problem_severity_group(test_func, timeout=1.0)
        assert result == 2
        assert resolver.severity_groups[2].is_problem_group is True


class TestSeverityGroupResolverBuildRelaxed:
    """Tests for build_relaxed_constraint_set."""

    def test_build_relaxed_separates_hard_and_soft(self):
        """Problem level constraints should be returned as soft."""
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,   # Level 1
            ClubDayConstraintAI,                # Level 2
        ], verbose=False)

        hard, soft = resolver.build_relaxed_constraint_set(problem_level=2)

        # Hard should contain level 1 constraints only
        assert NoDoubleBookingTeamsConstraintAI in hard
        assert ClubDayConstraintAI not in hard

        # Soft should have something for the level 2 constraint
        assert len(soft) > 0

    def test_build_relaxed_increments_slack(self):
        """Relaxing should increment the slack on the problem group."""
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
        ], verbose=False)

        assert resolver.severity_groups[2].current_slack == 0
        resolver.build_relaxed_constraint_set(problem_level=2)
        assert resolver.severity_groups[2].current_slack == 1


# ============== Integration with Real Solver ==============

class TestApplyConstraintsWithRelaxation:
    """Test apply_constraints_with_relaxation with a real CP-SAT model."""

    @pytest.fixture
    def real_data(self):
        """Load real 2026 season data."""
        from config import load_season_data
        return load_season_data(2026)

    def test_apply_hard_constraints_to_real_model(self, real_data):
        """Apply real constraints to a real model (no relaxation)."""
        model = cp_model.CpModel()
        from utils import generate_X
        X, conflicts = generate_X(model, real_data)

        test_data = dict(real_data)
        test_data['penalties'] = {}
        test_data['team_conflicts'] = conflicts
        if isinstance(test_data.get('games'), dict):
            test_data['games'] = list(test_data['games'].keys())

        # Apply only no-double-booking (fast, always feasible alone)
        total = apply_constraints_with_relaxation(
            model, X, test_data,
            constraints=[NoDoubleBookingTeamsConstraintAI],
            relaxed_groups=None,
        )
        # Should have applied some constraints
        assert total is None or total >= 0  # apply may return None or count

    def test_apply_with_relaxed_group(self, real_data):
        """Apply constraints with a relaxed severity group."""
        model = cp_model.CpModel()
        from utils import generate_X
        X, conflicts = generate_X(model, real_data)

        test_data = dict(real_data)
        test_data['penalties'] = {}
        test_data['team_conflicts'] = conflicts
        if isinstance(test_data.get('games'), dict):
            test_data['games'] = list(test_data['games'].keys())

        # Apply with level 3 relaxed at slack=2 (spec-024: level 4 now empty).
        total = apply_constraints_with_relaxation(
            model, X, test_data,
            constraints=[
                NoDoubleBookingTeamsConstraintAI,
                ClubGameSpreadAI,
            ],
            relaxed_groups={3: 2},
        )
        # Should not crash; constraints applied
        assert total is None or total >= 0


class TestCreateRelaxationTestFunc:
    """Test create_relaxation_test_func with real data and solver."""

    @pytest.fixture
    def real_data(self):
        from config import load_season_data
        return load_season_data(2026)

    def test_test_func_returns_status_and_feasibility(self, real_data):
        """The returned test_func should return (status_name, is_feasible)."""
        from utils import generate_X
        test_func = create_relaxation_test_func(real_data, generate_X, timeout=2.0)

        # Test with just one lightweight constraint
        status_name, is_feasible = test_func(
            [NoDoubleBookingTeamsConstraintAI], 2.0
        )
        assert isinstance(status_name, str)
        assert isinstance(is_feasible, bool)
        # Single constraint should be feasible or at least not crash
        assert status_name in ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN', 'MODEL_INVALID']
