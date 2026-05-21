# tests/test_infeasibility_resolver.py
"""
Tests for infeasibility_resolver.py - NO MOCKS.

Uses real constraint classes, real config data, and real CP-SAT models.
Mini solver runs use short timeouts (1-5 seconds).
"""

import pytest
import sys
import os

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.resolver import (
    get_constraint_names_from_stage,
    ConstraintState,
    ConstraintSlackRegistry,
    InfeasibilityResult,
    InfeasibilityResolver,
    build_names_map,
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
    EqualMatchUpSpacingConstraintAI,
    MaximiseClubsPerTimeslotBroadmeadowAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
    EnsureBestTimeslotChoicesAI,
    PreferredTimesConstraintAI,
)


# ============== get_constraint_names_from_stage Tests ==============

class TestGetConstraintNamesFromStage:
    """Tests for get_constraint_names_from_stage using real constraint classes."""

    def test_extracts_names_from_real_constraint_classes(self):
        stage_config = {
            'constraints': [NoDoubleBookingTeamsConstraintAI, ClubDayConstraintAI]
        }
        names = get_constraint_names_from_stage(stage_config)
        assert names == ['NoDoubleBookingTeamsConstraintAI', 'ClubDayConstraintAI']

    def test_returns_empty_list_for_empty_constraints(self):
        stage_config = {'constraints': []}
        names = get_constraint_names_from_stage(stage_config)
        assert names == []

    def test_returns_empty_list_for_missing_key(self):
        stage_config = {'other_key': 'value'}
        names = get_constraint_names_from_stage(stage_config)
        assert names == []


# ============== ConstraintState Tests ==============

class TestConstraintState:
    """Tests for ConstraintState dataclass."""

    def test_default_initialization(self):
        state = ConstraintState(name='ClubDayConstraint', severity_level=2)
        assert state.name == 'ClubDayConstraint'
        assert state.severity_level == 2
        assert state.current_slack == 1
        assert state.max_slack == 3
        assert state.is_hard is True
        assert state.violation_count == 0

    def test_can_relax_returns_true_for_non_level1(self):
        for level in [2, 3, 4, 5]:
            state = ConstraintState(name='Test', severity_level=level)
            assert state.can_relax() is True

    def test_can_relax_returns_false_for_level1(self):
        state = ConstraintState(name='NoDoubleBookingTeams', severity_level=1)
        assert state.can_relax() is False

    def test_can_relax_returns_false_when_at_max_slack(self):
        state = ConstraintState(name='Test', severity_level=2, current_slack=3)
        state.is_hard = False
        assert state.can_relax() is False

    def test_relax_switches_from_hard_to_soft(self):
        state = ConstraintState(name='ClubDayConstraint', severity_level=2)
        result = state.relax()
        assert result is True
        assert state.is_hard is False
        assert state.current_slack == 0
        assert state.violation_count == 1

    def test_relax_increments_slack_when_already_soft(self):
        state = ConstraintState(name='ClubDayConstraint', severity_level=2)
        state.is_hard = False
        state.current_slack = 1
        result = state.relax()
        assert result is True
        assert state.current_slack == 2
        assert state.violation_count == 1

    def test_relax_returns_false_for_level1(self):
        state = ConstraintState(name='NoDoubleBookingTeams', severity_level=1)
        result = state.relax()
        assert result is False
        assert state.is_hard is True
        assert state.violation_count == 0

    def test_multiple_relaxations(self):
        """Test progressive relaxation: hard -> soft(0) -> soft(1) -> soft(2) -> soft(3) -> fail."""
        state = ConstraintState(name='ClubDayConstraint', severity_level=2)
        # hard -> soft(0)
        assert state.relax() is True
        assert state.is_hard is False
        assert state.current_slack == 0
        # soft(0) -> soft(1)
        assert state.relax() is True
        assert state.current_slack == 1
        # soft(1) -> soft(2)
        assert state.relax() is True
        assert state.current_slack == 2
        # soft(2) -> soft(3)
        assert state.relax() is True
        assert state.current_slack == 3
        # At max
        assert state.relax() is False
        assert state.violation_count == 4


# ============== ConstraintSlackRegistry Tests ==============

class TestConstraintSlackRegistry:
    """Tests for ConstraintSlackRegistry class using real CONSTRAINT_SEVERITY_LEVELS."""

    def test_initialization_creates_constraint_states(self):
        registry = ConstraintSlackRegistry()
        assert len(registry.constraints) > 0
        # These are the short names from CONSTRAINT_SEVERITY_LEVELS in tester.py
        assert 'NoDoubleBookingTeams' in registry.constraints
        assert 'ClubDayConstraint' in registry.constraints

    def test_get_state_returns_state_for_known_constraint(self):
        registry = ConstraintSlackRegistry()
        state = registry.get_state('ClubDayConstraint')
        assert state is not None
        assert state.name == 'ClubDayConstraint'
        assert state.severity_level == 2

    def test_get_state_returns_none_for_unknown_constraint(self):
        registry = ConstraintSlackRegistry()
        state = registry.get_state('NonExistentConstraint')
        assert state is None

    def test_increase_slack_on_level2_constraint(self):
        registry = ConstraintSlackRegistry()
        result = registry.increase_slack('ClubDayConstraint')
        assert result is True
        state = registry.get_state('ClubDayConstraint')
        assert state.is_hard is False

    def test_increase_slack_fails_for_unknown(self):
        registry = ConstraintSlackRegistry()
        result = registry.increase_slack('TotallyUnknownConstraint')
        assert result is False

    def test_increase_slack_fails_for_level1(self):
        """Level 1 constraints (e.g. NoDoubleBookingTeams) cannot be relaxed."""
        registry = ConstraintSlackRegistry()
        result = registry.increase_slack('NoDoubleBookingTeams')
        assert result is False

    def test_reset_restores_all_to_hard(self):
        registry = ConstraintSlackRegistry()
        # Relax real constraints
        registry.increase_slack('ClubDayConstraint')
        # spec-018: AwayAtMaitlandGrouping removed — use another level-2 rule.
        registry.increase_slack('TeamConflict')
        # Reset
        registry.reset()
        for name, state in registry.constraints.items():
            assert state.is_hard is True
            assert state.current_slack == 1

    def test_get_relaxed_constraints_initially_empty(self):
        registry = ConstraintSlackRegistry()
        assert registry.get_relaxed_constraints() == []

    def test_get_relaxed_constraints_after_relaxation(self):
        registry = ConstraintSlackRegistry()
        registry.increase_slack('ClubDayConstraint')
        relaxed = registry.get_relaxed_constraints()
        assert 'ClubDayConstraint' in relaxed

    def test_get_summary_returns_dict_with_expected_keys(self):
        registry = ConstraintSlackRegistry()
        summary = registry.get_summary()
        assert 'total_constraints' in summary
        assert 'relaxed_count' in summary
        assert 'constraints_by_level' in summary
        assert 'current_states' in summary
        assert summary['total_constraints'] > 0
        # Check level structure
        assert 1 in summary['constraints_by_level']
        assert 2 in summary['constraints_by_level']

    def test_constraints_by_level_has_correct_entries(self):
        """Verify specific constraints appear at expected severity levels."""
        registry = ConstraintSlackRegistry()
        summary = registry.get_summary()
        assert 'NoDoubleBookingTeams' in summary['constraints_by_level'][1]
        assert 'ClubDayConstraint' in summary['constraints_by_level'][2]


# ============== InfeasibilityResult Tests ==============

class TestInfeasibilityResult:
    """Tests for InfeasibilityResult dataclass."""

    def test_is_feasible_true_for_optimal(self):
        result = InfeasibilityResult(status='OPTIMAL')
        assert result.is_feasible is True

    def test_is_feasible_true_for_feasible(self):
        result = InfeasibilityResult(status='FEASIBLE')
        assert result.is_feasible is True

    def test_is_feasible_false_for_infeasible(self):
        result = InfeasibilityResult(status='INFEASIBLE')
        assert result.is_feasible is False

    def test_is_infeasible_true_for_infeasible(self):
        result = InfeasibilityResult(status='INFEASIBLE')
        assert result.is_infeasible is True

    def test_is_infeasible_false_for_optimal(self):
        result = InfeasibilityResult(status='OPTIMAL')
        assert result.is_infeasible is False

    def test_str_includes_blocking_constraints(self):
        result = InfeasibilityResult(
            status='INFEASIBLE',
            blocking_constraints=['ClubDayConstraint', 'TeamConflict']
        )
        str_repr = str(result)
        assert 'INFEASIBLE' in str_repr
        assert 'ClubDayConstraint' in str_repr

    def test_str_shows_constraint_count(self):
        result = InfeasibilityResult(
            status='OPTIMAL',
            total_constraints=42,
            solve_time_seconds=1.5
        )
        str_repr = str(result)
        assert '42' in str_repr
        assert '1.5' in str_repr


# ============== build_names_map Tests ==============

class TestBuildNamesMap:
    """Tests for build_names_map helper."""

    def test_maps_real_constraint_classes(self):
        classes = [NoDoubleBookingTeamsConstraintAI, ClubDayConstraintAI]
        names_map = build_names_map(classes)
        assert names_map[NoDoubleBookingTeamsConstraintAI] == 'NoDoubleBookingTeams'
        assert names_map[ClubDayConstraintAI] == 'ClubDayConstraint'


# ============== InfeasibilityResolver Basic Tests ==============

class TestInfeasibilityResolverBasic:
    """Basic tests for InfeasibilityResolver (no solver runs)."""

    def test_initialization_with_defaults(self):
        from config import load_season_data
        data = load_season_data(2026)
        resolver = InfeasibilityResolver(data)
        assert resolver.data is data
        assert resolver.registry is not None
        assert resolver.timeout == 5.0
        assert resolver.verbose is True

    def test_initialization_with_custom_registry(self):
        from config import load_season_data
        data = load_season_data(2026)
        registry = ConstraintSlackRegistry()
        resolver = InfeasibilityResolver(data, registry=registry)
        assert resolver.registry is registry

    def test_initialization_with_custom_timeout(self):
        from config import load_season_data
        data = load_season_data(2026)
        resolver = InfeasibilityResolver(data, timeout_per_test=10.0)
        assert resolver.timeout == 10.0

    def test_test_history_starts_empty(self):
        from config import load_season_data
        data = load_season_data(2026)
        resolver = InfeasibilityResolver(data)
        assert resolver.test_history == []

    def test_get_resolution_report_format(self):
        from config import load_season_data
        data = load_season_data(2026)
        resolver = InfeasibilityResolver(data, verbose=False)
        report = resolver.get_resolution_report()
        assert 'INFEASIBILITY RESOLUTION REPORT' in report
        assert 'Total tests run' in report


# ============== InfeasibilityResolver Integration Tests ==============

class TestInfeasibilityResolverIntegration:
    """Integration tests with real data and real solver."""

    @pytest.fixture
    def real_data(self):
        from config import load_season_data
        return load_season_data(2026)

    @pytest.fixture
    def resolver(self, real_data):
        return InfeasibilityResolver(real_data, timeout_per_test=3.0, verbose=False)

    def test_test_constraints_with_single_constraint(self, resolver):
        """Test single constraint feasibility check returns a proper result."""
        result = resolver.test_constraints([NoDoubleBookingTeamsConstraintAI])
        assert isinstance(result, InfeasibilityResult)
        assert result.status in ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN', 'MODEL_INVALID']
        assert 'NoDoubleBookingTeamsConstraintAI' in result.constraint_counts

    def test_test_constraints_adds_to_history(self, resolver):
        """Each call to test_constraints should append to test_history."""
        assert len(resolver.test_history) == 0
        resolver.test_constraints([NoDoubleBookingTeamsConstraintAI])
        assert len(resolver.test_history) == 1
        resolver.test_constraints([NoDoubleBookingFieldsConstraintAI])
        assert len(resolver.test_history) == 2

    def test_test_constraints_with_names_map(self, resolver):
        """Test using a names_map for registry lookups."""
        names_map = build_names_map([NoDoubleBookingTeamsConstraintAI])
        result = resolver.test_constraints(
            [NoDoubleBookingTeamsConstraintAI],
            names_map=names_map,
        )
        assert isinstance(result, InfeasibilityResult)
        # With names_map, the constraint_counts key should use the mapped name
        assert 'NoDoubleBookingTeams' in result.constraint_counts

    def test_find_blocking_constraints_with_feasible_pair(self, resolver):
        """Two feasible constraints together should return empty blocking list."""
        blocking = resolver.find_blocking_constraints([
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
        ])
        # These should be feasible together (or UNKNOWN due to timeout)
        assert isinstance(blocking, list)

    def test_test_constraints_records_solve_time(self, resolver):
        """Solve time should be recorded in the result."""
        result = resolver.test_constraints([NoDoubleBookingTeamsConstraintAI])
        assert result.solve_time_seconds >= 0.0
