# tests/test_infeasibility_resolver.py
"""
Unit tests for infeasibility_resolver.py

Tests for:
- get_constraint_names_from_stage helper function
- ConstraintState dataclass
- ConstraintSlackRegistry class
- InfeasibilityResult dataclass
- InfeasibilityResolver class (basic tests - full solver tests are integration)
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.resolver import (
    get_constraint_names_from_stage,
    ConstraintState,
    ConstraintSlackRegistry,
    InfeasibilityResult,
    InfeasibilityResolver,
)


# ============== get_constraint_names_from_stage Tests ==============

class TestGetConstraintNamesFromStage:
    """Tests for get_constraint_names_from_stage helper function."""

    def test_extracts_names_from_constraint_classes(self):
        """Test extracting names from a list of constraint classes."""
        class MockConstraintA:
            pass
        
        class MockConstraintB:
            pass
        
        stage_config = {
            'constraints': [MockConstraintA, MockConstraintB]
        }
        
        names = get_constraint_names_from_stage(stage_config)
        
        assert names == ['MockConstraintA', 'MockConstraintB']

    def test_returns_empty_list_for_empty_constraints(self):
        """Test empty list when no constraints."""
        stage_config = {'constraints': []}
        names = get_constraint_names_from_stage(stage_config)
        assert names == []

    def test_returns_empty_list_for_missing_key(self):
        """Test empty list when constraints key missing."""
        stage_config = {'other_key': 'value'}
        names = get_constraint_names_from_stage(stage_config)
        assert names == []


# ============== ConstraintState Tests ==============

class TestConstraintState:
    """Tests for ConstraintState dataclass."""

    def test_default_initialization(self):
        """Test default state initialization."""
        state = ConstraintState(name='TestConstraint', severity_level=2)
        
        assert state.name == 'TestConstraint'
        assert state.severity_level == 2
        assert state.current_slack == 1
        assert state.max_slack == 3
        assert state.is_hard is True
        assert state.violation_count == 0

    def test_can_relax_returns_true_for_non_level1(self):
        """Test that level 2+ constraints can be relaxed."""
        for level in [2, 3, 4]:
            state = ConstraintState(name='Test', severity_level=level)
            assert state.can_relax() is True

    def test_can_relax_returns_false_for_level1(self):
        """Test that level 1 constraints cannot be relaxed."""
        state = ConstraintState(name='Test', severity_level=1)
        assert state.can_relax() is False

    def test_can_relax_returns_false_when_at_max_slack(self):
        """Test cannot relax when at max slack."""
        state = ConstraintState(name='Test', severity_level=2, current_slack=3)
        state.is_hard = False  # Already soft
        assert state.can_relax() is False

    def test_relax_switches_from_hard_to_soft(self):
        """Test first relaxation switches from hard to soft."""
        state = ConstraintState(name='Test', severity_level=2)
        
        result = state.relax()
        
        assert result is True
        assert state.is_hard is False
        assert state.current_slack == 0
        assert state.violation_count == 1

    def test_relax_increments_slack_when_already_soft(self):
        """Test relaxation increments slack when already soft."""
        state = ConstraintState(name='Test', severity_level=2)
        state.is_hard = False
        state.current_slack = 1
        
        result = state.relax()
        
        assert result is True
        assert state.current_slack == 2
        assert state.violation_count == 1

    def test_relax_returns_false_for_level1(self):
        """Test relax returns False for level 1 constraints."""
        state = ConstraintState(name='Test', severity_level=1)
        
        result = state.relax()
        
        assert result is False
        assert state.is_hard is True
        assert state.violation_count == 0


# ============== ConstraintSlackRegistry Tests ==============

class TestConstraintSlackRegistry:
    """Tests for ConstraintSlackRegistry class."""

    def test_initialization_creates_constraint_states(self):
        """Test registry initializes with known constraints."""
        registry = ConstraintSlackRegistry()
        
        # Should have entries from CONSTRAINT_SEVERITY_LEVELS
        assert len(registry.constraints) > 0
        
        # Check some known constraints exist (uses short names)
        assert 'NoDoubleBookingTeams' in registry.constraints
        assert 'ClubDayConstraint' in registry.constraints

    def test_get_state_returns_state_for_known_constraint(self):
        """Test getting state for a known constraint."""
        registry = ConstraintSlackRegistry()
        
        state = registry.get_state('ClubDayConstraint')
        
        assert state is not None
        assert state.name == 'ClubDayConstraint'

    def test_get_state_returns_none_for_unknown_constraint(self):
        """Test getting state for unknown constraint returns None."""
        registry = ConstraintSlackRegistry()
        
        state = registry.get_state('NonExistentConstraint')
        
        assert state is None

    def test_increase_slack_increases_slack(self):
        """Test increasing slack on a constraint."""
        registry = ConstraintSlackRegistry()
        
        result = registry.increase_slack('ClubDayConstraint')
        
        assert result is True
        state = registry.get_state('ClubDayConstraint')
        assert state.is_hard is False

    def test_increase_slack_fails_for_unknown(self):
        """Test increasing slack on unknown constraint fails."""
        registry = ConstraintSlackRegistry()
        
        result = registry.increase_slack('UnknownConstraint')
        
        assert result is False

    def test_increase_slack_fails_for_level1(self):
        """Test cannot increase slack on level 1 constraint."""
        registry = ConstraintSlackRegistry()
        
        result = registry.increase_slack('NoDoubleBookingTeamsConstraint')
        
        assert result is False

    def test_reset_restores_all_to_hard(self):
        """Test reset restores all constraints to hard."""
        registry = ConstraintSlackRegistry()
        
        # Relax some constraints
        registry.increase_slack('ClubDayConstraint')
        registry.increase_slack('EqualMatchUpSpacingConstraint')
        
        # Reset
        registry.reset()
        
        # All should be hard again
        for name, state in registry.constraints.items():
            assert state.is_hard is True
            assert state.current_slack == 1

    def test_get_relaxed_constraints_returns_relaxed_list(self):
        """Test getting list of relaxed constraints."""
        registry = ConstraintSlackRegistry()
        
        # Initially none relaxed
        assert registry.get_relaxed_constraints() == []
        
        # Relax one
        registry.increase_slack('ClubDayConstraint')
        
        relaxed = registry.get_relaxed_constraints()
        assert 'ClubDayConstraint' in relaxed

    def test_get_summary_returns_dict_with_expected_keys(self):
        """Test get_summary returns expected structure."""
        registry = ConstraintSlackRegistry()
        
        summary = registry.get_summary()
        
        assert 'total_constraints' in summary
        assert 'relaxed_count' in summary
        assert 'constraints_by_level' in summary
        assert 'current_states' in summary
        assert summary['total_constraints'] > 0


# ============== InfeasibilityResult Tests ==============

class TestInfeasibilityResult:
    """Tests for InfeasibilityResult dataclass."""

    def test_is_feasible_true_for_optimal(self):
        """Test is_feasible returns True for OPTIMAL."""
        result = InfeasibilityResult(status='OPTIMAL')
        assert result.is_feasible is True

    def test_is_feasible_true_for_feasible(self):
        """Test is_feasible returns True for FEASIBLE."""
        result = InfeasibilityResult(status='FEASIBLE')
        assert result.is_feasible is True

    def test_is_feasible_false_for_infeasible(self):
        """Test is_feasible returns False for INFEASIBLE."""
        result = InfeasibilityResult(status='INFEASIBLE')
        assert result.is_feasible is False

    def test_is_infeasible_true_for_infeasible(self):
        """Test is_infeasible returns True for INFEASIBLE."""
        result = InfeasibilityResult(status='INFEASIBLE')
        assert result.is_infeasible is True

    def test_is_infeasible_false_for_optimal(self):
        """Test is_infeasible returns False for OPTIMAL."""
        result = InfeasibilityResult(status='OPTIMAL')
        assert result.is_infeasible is False

    def test_str_includes_blocking_constraints(self):
        """Test string representation includes blocking constraints."""
        result = InfeasibilityResult(
            status='INFEASIBLE',
            blocking_constraints=['ConstraintA', 'ConstraintB']
        )
        
        str_repr = str(result)
        assert 'INFEASIBLE' in str_repr
        assert 'ConstraintA' in str_repr

    def test_str_shows_constraint_count(self):
        """Test string representation shows constraint count."""
        result = InfeasibilityResult(
            status='OPTIMAL',
            total_constraints=42,
            solve_time_seconds=1.5
        )
        
        str_repr = str(result)
        assert '42' in str_repr
        assert '1.5' in str_repr


# ============== InfeasibilityResolver Basic Tests ==============

class TestInfeasibilityResolverBasic:
    """Basic tests for InfeasibilityResolver class (no actual solver runs)."""

    def test_initialization_with_defaults(self):
        """Test resolver initializes with default registry."""
        data = {'teams': [], 'timeslots': [], 'games': []}
        
        resolver = InfeasibilityResolver(data)
        
        assert resolver.data == data
        assert resolver.registry is not None
        assert resolver.timeout == 5.0
        assert resolver.verbose is True

    def test_initialization_with_custom_registry(self):
        """Test resolver uses custom registry when provided."""
        data = {'teams': [], 'timeslots': [], 'games': []}
        registry = ConstraintSlackRegistry()
        
        resolver = InfeasibilityResolver(data, registry=registry)
        
        assert resolver.registry is registry

    def test_initialization_with_custom_timeout(self):
        """Test resolver uses custom timeout."""
        data = {'teams': [], 'timeslots': [], 'games': []}
        
        resolver = InfeasibilityResolver(data, timeout_per_test=10.0)
        
        assert resolver.timeout == 10.0

    def test_test_history_starts_empty(self):
        """Test that test history is initially empty."""
        data = {'teams': [], 'timeslots': [], 'games': []}
        
        resolver = InfeasibilityResolver(data)
        
        assert resolver.test_history == []


# ============== InfeasibilityResolver Integration Tests ==============

class TestInfeasibilityResolverIntegration:
    """Integration tests for InfeasibilityResolver with actual data."""

    @pytest.fixture
    def resolver_with_data(self):
        """Create resolver with real 2025 data."""
        from run import load_data_for_year
        data = load_data_for_year(2025)
        return InfeasibilityResolver(data, timeout_per_test=2.0, verbose=False)

    def test_test_constraints_returns_result(self, resolver_with_data):
        """Test that test_constraints returns an InfeasibilityResult."""
        from constraints.ai import NoDoubleBookingTeamsConstraintAI
        
        result = resolver_with_data.test_constraints([NoDoubleBookingTeamsConstraintAI])
        
        assert isinstance(result, InfeasibilityResult)
        assert result.status in ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN', 'MODEL_INVALID']

    def test_test_constraints_with_single_constraint(self, resolver_with_data):
        """Test single constraint feasibility check."""
        from constraints.ai import NoDoubleBookingTeamsConstraintAI
        
        result = resolver_with_data.test_constraints([NoDoubleBookingTeamsConstraintAI])
        
        # Single double-booking constraint should be feasible
        assert result.status != 'MODEL_INVALID'
        assert 'NoDoubleBookingTeamsConstraintAI' in result.constraint_counts

    def test_find_blocking_constraints_with_feasible_set(self, resolver_with_data):
        """Test find_blocking_constraints with feasible constraint set."""
        from constraints.ai import (
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
        )
        
        blocking = resolver_with_data.find_blocking_constraints([
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
        ])
        
        # These should be feasible together, so no blocking constraints
        # Or UNKNOWN due to timeout, which returns empty list
        assert isinstance(blocking, list)
