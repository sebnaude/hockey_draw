# tests/test_severity_relaxation.py
"""
Unit tests for severity_relaxation.py

Tests for:
- CONSTRAINT_TO_SEVERITY mapping
- get_severity_level function
- group_constraints_by_severity function
- SeverityGroupState dataclass
- SeverityGroupResolver class (basic tests)
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.severity import (
    CONSTRAINT_TO_SEVERITY,
    get_severity_level,
    group_constraints_by_severity,
    SeverityGroupState,
    SeverityGroupResolver,
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

    def test_all_levels_are_1_to_4(self):
        """Test all severity levels are between 1 and 4."""
        for level in CONSTRAINT_TO_SEVERITY.values():
            assert level in [1, 2, 3, 4]


# ============== get_severity_level Tests ==============

class TestGetSeverityLevel:
    """Tests for get_severity_level function."""

    def test_returns_correct_level_for_known_constraint(self):
        """Test returns correct level for a known constraint."""
        class NoDoubleBookingTeamsConstraint:
            pass
        
        level = get_severity_level(NoDoubleBookingTeamsConstraint)
        assert level == 1

    def test_returns_correct_level_for_ai_constraint(self):
        """Test returns correct level for AI constraint variant."""
        class ClubDayConstraintAI:
            pass
        
        level = get_severity_level(ClubDayConstraintAI)
        assert level == 2

    def test_returns_4_for_unknown_constraint(self):
        """Test returns 4 (lowest) for unknown constraints."""
        class UnknownConstraint:
            pass
        
        level = get_severity_level(UnknownConstraint)
        assert level == 4


# ============== group_constraints_by_severity Tests ==============

class TestGroupConstraintsBySeverity:
    """Tests for group_constraints_by_severity function."""

    def test_groups_constraints_correctly(self):
        """Test constraints are grouped by severity level."""
        class Level1Constraint:
            pass
        
        class Level2Constraint:
            pass
        
        # Mock the get_severity_level to return specific values
        with patch('constraints.severity.get_severity_level') as mock_get:
            mock_get.side_effect = lambda c: 1 if 'Level1' in c.__name__ else 2
            
            groups = group_constraints_by_severity([Level1Constraint, Level2Constraint])
            
            assert 1 in groups
            assert 2 in groups
            assert Level1Constraint in groups[1]
            assert Level2Constraint in groups[2]

    def test_empty_list_returns_empty_dict(self):
        """Test empty constraint list returns empty dict."""
        groups = group_constraints_by_severity([])
        assert groups == {}

    def test_multiple_constraints_same_level(self):
        """Test multiple constraints at same level grouped together."""
        class Constraint1:
            pass
        
        class Constraint2:
            pass
        
        with patch('constraints.severity.get_severity_level', return_value=3):
            groups = group_constraints_by_severity([Constraint1, Constraint2])
            
            assert len(groups[3]) == 2


# ============== SeverityGroupState Tests ==============

class TestSeverityGroupState:
    """Tests for SeverityGroupState dataclass."""

    def test_default_initialization(self):
        """Test default state initialization."""
        state = SeverityGroupState(level=2, constraint_classes=[])
        
        assert state.level == 2
        assert state.constraint_classes == []
        assert state.current_slack == 0
        assert state.max_slack == 3
        assert state.is_problem_group is False

    def test_can_relax_true_for_non_level1(self):
        """Test can_relax returns True for level 2+ with slack room."""
        for level in [2, 3, 4]:
            state = SeverityGroupState(level=level, constraint_classes=[])
            assert state.can_relax() is True

    def test_can_relax_false_for_level1(self):
        """Test can_relax returns False for level 1."""
        state = SeverityGroupState(level=1, constraint_classes=[])
        assert state.can_relax() is False

    def test_can_relax_false_at_max_slack(self):
        """Test can_relax returns False when at max slack."""
        state = SeverityGroupState(level=2, constraint_classes=[], current_slack=3)
        assert state.can_relax() is False

    def test_relax_increments_slack(self):
        """Test relax increments slack by 1."""
        state = SeverityGroupState(level=2, constraint_classes=[])
        
        result = state.relax()
        
        assert result is True
        assert state.current_slack == 1

    def test_relax_fails_at_max_slack(self):
        """Test relax returns False at max slack."""
        state = SeverityGroupState(level=2, constraint_classes=[], current_slack=3)
        
        result = state.relax()
        
        assert result is False
        assert state.current_slack == 3

    def test_relax_fails_for_level1(self):
        """Test relax returns False for level 1."""
        state = SeverityGroupState(level=1, constraint_classes=[])
        
        result = state.relax()
        
        assert result is False
        assert state.current_slack == 0


# ============== SeverityGroupResolver Basic Tests ==============

class TestSeverityGroupResolverBasic:
    """Basic tests for SeverityGroupResolver class."""

    def test_initialization(self):
        """Test resolver initializes correctly."""
        class MockConstraint:
            pass
        
        with patch('constraints.severity.get_severity_level', return_value=2):
            resolver = SeverityGroupResolver([MockConstraint], verbose=False)
            
            assert resolver.all_constraints == [MockConstraint]
            assert resolver.verbose is False

    def test_initialization_groups_constraints(self):
        """Test resolver groups constraints by severity on init."""
        class Level1:
            pass
        
        class Level2:
            pass
        
        def mock_level(c):
            return 1 if 'Level1' in c.__name__ else 2
        
        with patch('constraints.severity.get_severity_level', side_effect=mock_level):
            resolver = SeverityGroupResolver([Level1, Level2], verbose=False)
            
            assert 1 in resolver.severity_groups
            assert 2 in resolver.severity_groups

    def test_get_constraints_for_test_returns_all_when_no_exclusions(self):
        """Test getting all constraints when no exclusions specified."""
        class Level2Constraint:
            pass
        
        class Level3Constraint:
            pass
        
        def mock_level(c):
            return 2 if 'Level2' in c.__name__ else 3
        
        with patch('constraints.severity.get_severity_level', side_effect=mock_level):
            resolver = SeverityGroupResolver(
                [Level2Constraint, Level3Constraint], 
                verbose=False
            )
            
            all_constraints = resolver.get_constraints_for_test()
            assert len(all_constraints) == 2
            assert Level2Constraint in all_constraints
            assert Level3Constraint in all_constraints

    def test_get_constraints_for_test_excludes_specified_levels(self):
        """Test excluding constraints by severity level."""
        class Level2Constraint:
            pass
        
        class Level3Constraint:
            pass
        
        def mock_level(c):
            return 2 if 'Level2' in c.__name__ else 3
        
        with patch('constraints.severity.get_severity_level', side_effect=mock_level):
            resolver = SeverityGroupResolver(
                [Level2Constraint, Level3Constraint], 
                verbose=False
            )
            
            # Exclude level 3
            constraints = resolver.get_constraints_for_test(exclude_levels={3})
            assert Level2Constraint in constraints
            assert Level3Constraint not in constraints


# ============== Integration Tests with Real Constraints ==============

class TestSeverityGroupResolverIntegration:
    """Integration tests using real constraint classes."""

    def test_with_real_constraint_classes(self):
        """Test grouping with actual constraint classes from codebase."""
        from constraints.ai import (
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
            EqualMatchUpSpacingConstraintAI,
        )
        
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
            EqualMatchUpSpacingConstraintAI,
        ], verbose=False)
        
        # Check grouping - level 1 should have NoDoubleBooking
        assert 1 in resolver.severity_groups
        level1_classes = resolver.severity_groups[1].constraint_classes
        assert NoDoubleBookingTeamsConstraintAI in level1_classes
        
        # Level 2 should have ClubDay
        assert 2 in resolver.severity_groups
        level2_classes = resolver.severity_groups[2].constraint_classes
        assert ClubDayConstraintAI in level2_classes
        
        # Level 3 should have EqualMatchUpSpacing
        assert 3 in resolver.severity_groups
        level3_classes = resolver.severity_groups[3].constraint_classes
        assert EqualMatchUpSpacingConstraintAI in level3_classes

    def test_severity_group_state_can_relax(self):
        """Test checking if a severity group can be relaxed."""
        from constraints.ai import (
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
        )
        
        resolver = SeverityGroupResolver([
            NoDoubleBookingTeamsConstraintAI,
            ClubDayConstraintAI,
        ], verbose=False)
        
        # Level 1 cannot be relaxed
        assert resolver.severity_groups[1].can_relax() is False
        
        # Level 2 can be relaxed
        assert resolver.severity_groups[2].can_relax() is True
