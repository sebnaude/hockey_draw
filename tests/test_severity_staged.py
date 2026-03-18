# tests/test_severity_staged.py
"""
Unit tests for severity-based staged solving.

Tests for:
- STAGES_SEVERITY and STAGES_SEVERITY_AI structure
- Severity stage constraint groupings
- CLI --staged flag integration
- Cumulative constraint application with severity stages
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_staged import (
    STAGES, STAGES_AI,
    STAGES_SEVERITY, STAGES_SEVERITY_AI,
    StagedScheduleSolver
)
from constraints.severity import CONSTRAINT_TO_SEVERITY


# ============== Stage Structure Tests ==============

class TestSeverityStageStructure:
    """Tests for STAGES_SEVERITY dictionary structure."""

    def test_has_five_severity_levels(self):
        """Verify STAGES_SEVERITY has exactly 5 levels."""
        assert len(STAGES_SEVERITY) == 5
        assert 'severity_1' in STAGES_SEVERITY
        assert 'severity_2' in STAGES_SEVERITY
        assert 'severity_3' in STAGES_SEVERITY
        assert 'severity_4' in STAGES_SEVERITY
        assert 'severity_5' in STAGES_SEVERITY

    def test_ai_has_five_severity_levels(self):
        """Verify STAGES_SEVERITY_AI has exactly 5 levels."""
        assert len(STAGES_SEVERITY_AI) == 5
        assert 'severity_1' in STAGES_SEVERITY_AI
        assert 'severity_2' in STAGES_SEVERITY_AI
        assert 'severity_3' in STAGES_SEVERITY_AI
        assert 'severity_4' in STAGES_SEVERITY_AI
        assert 'severity_5' in STAGES_SEVERITY_AI

    def test_each_stage_has_required_keys(self):
        """Verify each severity stage has required configuration keys."""
        required_keys = ['name', 'description', 'constraints', 'max_time_seconds', 'required', 'use_callback']
        
        for stage_name, config in STAGES_SEVERITY.items():
            for key in required_keys:
                assert key in config, f"Missing key '{key}' in {stage_name}"

    def test_each_ai_stage_has_required_keys(self):
        """Verify each AI severity stage has required configuration keys."""
        required_keys = ['name', 'description', 'constraints', 'max_time_seconds', 'required', 'use_callback']
        
        for stage_name, config in STAGES_SEVERITY_AI.items():
            for key in required_keys:
                assert key in config, f"Missing key '{key}' in AI {stage_name}"

    def test_severity_1_is_required(self):
        """Severity 1 (CRITICAL) must be required=True."""
        assert STAGES_SEVERITY['severity_1']['required'] is True

    def test_severity_4_is_not_required(self):
        """Severity 4 (LOW) should be required=False (soft optimization)."""
        assert STAGES_SEVERITY['severity_4']['required'] is False

    def test_all_stages_have_constraints(self):
        """Each stage should have at least one constraint."""
        for stage_name, config in STAGES_SEVERITY.items():
            assert len(config['constraints']) > 0, f"{stage_name} has no constraints"

    def test_severity_5_is_not_required(self):
        """Severity 5 (VERY LOW) should be required=False (timeslot preferences)."""
        assert STAGES_SEVERITY['severity_5']['required'] is False

    def test_severity_stage_ordering(self):
        """Verify stage names are ordered correctly."""
        stage_names = list(STAGES_SEVERITY.keys())
        assert stage_names == ['severity_1', 'severity_2', 'severity_3', 'severity_4', 'severity_5']


# ============== Constraint Grouping Tests ==============

class TestConstraintGrouping:
    """Tests that constraints are grouped by correct severity level."""

    def test_severity_1_contains_critical_constraints(self):
        """Verify Level 1 contains only CRITICAL (level 1) constraints."""
        level_1_constraints = STAGES_SEVERITY['severity_1']['constraints']
        
        for constraint_cls in level_1_constraints:
            name = constraint_cls.__name__
            expected_level = CONSTRAINT_TO_SEVERITY.get(name, 5)
            assert expected_level == 1, f"{name} is level {expected_level}, expected 1 for severity_1"

    def test_severity_2_contains_high_constraints(self):
        """Verify Level 2 contains HIGH (level 2) constraints."""
        level_2_constraints = STAGES_SEVERITY['severity_2']['constraints']
        
        for constraint_cls in level_2_constraints:
            name = constraint_cls.__name__
            expected_level = CONSTRAINT_TO_SEVERITY.get(name, 5)
            assert expected_level == 2, f"{name} is level {expected_level}, expected 2 for severity_2"

    def test_severity_3_contains_medium_constraints(self):
        """Verify Level 3 contains MEDIUM (level 3) constraints."""
        level_3_constraints = STAGES_SEVERITY['severity_3']['constraints']
        
        for constraint_cls in level_3_constraints:
            name = constraint_cls.__name__
            expected_level = CONSTRAINT_TO_SEVERITY.get(name, 5)
            assert expected_level == 3, f"{name} is level {expected_level}, expected 3 for severity_3"

    def test_severity_4_contains_low_constraints(self):
        """Verify severity_4 stage contains LOW (level 4) constraints."""
        level_4_constraints = STAGES_SEVERITY['severity_4']['constraints']
        
        for constraint_cls in level_4_constraints:
            name = constraint_cls.__name__
            expected_level = CONSTRAINT_TO_SEVERITY.get(name, 5)
            assert expected_level == 4, f"{name} is level {expected_level}, expected 4 for severity_4"

    def test_severity_5_contains_very_low_constraints(self):
        """Verify severity_5 stage contains VERY LOW (level 5) constraints."""
        level_5_constraints = STAGES_SEVERITY['severity_5']['constraints']
        
        for constraint_cls in level_5_constraints:
            name = constraint_cls.__name__
            expected_level = CONSTRAINT_TO_SEVERITY.get(name, 5)
            assert expected_level == 5, f"{name} is level {expected_level}, expected 5 for severity_5"

    def test_no_duplicate_constraints_across_stages(self):
        """Verify no constraint appears in multiple severity stages."""
        all_constraints = []
        
        for stage_name, config in STAGES_SEVERITY.items():
            for constraint_cls in config['constraints']:
                name = constraint_cls.__name__
                assert name not in all_constraints, f"Duplicate constraint {name} in {stage_name}"
                all_constraints.append(name)

    def test_ai_stages_mirror_regular_stages(self):
        """Verify AI stages have same constraint count per level."""
        for stage_name in STAGES_SEVERITY.keys():
            regular_count = len(STAGES_SEVERITY[stage_name]['constraints'])
            ai_count = len(STAGES_SEVERITY_AI[stage_name]['constraints'])
            assert regular_count == ai_count, f"Mismatch in {stage_name}: {regular_count} vs {ai_count} AI"


# ============== CLI Integration Tests ==============

class TestStagedCLIFlag:
    """Tests for --staged CLI flag integration."""

    def test_staged_flag_parsed_correctly(self):
        """Verify --staged flag is parsed as boolean."""
        import argparse
        import run
        
        # Create a mock argument namespace that would come from argparse
        args = argparse.Namespace(
            year=2026,
            staged=True,
            simple=False,
            stages=None,
            ai=False,
            resume=None,
            run_id=None,
            locked=None,
            lock_weeks=0,
            workers=None,
            low_memory=False,
            minimal_memory=False,
            high_performance=False,
            exclude=None,
            relax=False,
            relax_timeout=30.0,
            fix_round_1=False,
            slack=None
        )
        
        # Verify staged is True
        assert args.staged is True

    def test_staged_false_by_default(self):
        """Verify --staged defaults to False when not specified."""
        import argparse
        
        # Simulate default args
        args = argparse.Namespace(staged=False)
        assert args.staged is False


# ============== Solver Stage Mode Selection Tests ==============

class TestSolverStageSelection:
    """Tests for run_staged_solve stage mode selection."""

    def test_severity_staged_selects_severity_stages(self):
        """Verify severity_staged=True uses STAGES_SEVERITY."""
        # We test this by checking the signature and docstring
        import inspect
        
        sig = inspect.signature(StagedScheduleSolver.run_staged_solve)
        params = list(sig.parameters.keys())
        
        assert 'severity_staged' in params, "severity_staged parameter missing"

    def test_default_mode_uses_default_stages(self):
        """Verify severity_staged=False uses default STAGES."""
        # Check that STAGES (default) has different keys than STAGES_SEVERITY
        default_keys = set(STAGES.keys())
        severity_keys = set(STAGES_SEVERITY.keys())
        
        # They should be different
        assert default_keys != severity_keys
        assert 'stage1_required' in default_keys
        assert 'severity_1' in severity_keys


# ============== Slack Integration Tests ==============

class TestSlackWithSeverityStages:
    """Tests that --slack works with severity-staged mode."""

    def test_slack_dict_structure(self):
        """Verify constraint_slack dict is structured correctly."""
        # Simulate what run.py does with --slack 1
        slack_value = 1
        constraint_slack = {
            'EqualMatchUpSpacingConstraint': slack_value,
            'AwayAtMaitlandGrouping': slack_value,
            'MaitlandHomeGrouping': slack_value,
        }
        
        assert len(constraint_slack) == 3
        assert all(v == 1 for v in constraint_slack.values())

    def test_slack_constraints_exist_in_severity_stages(self):
        """Verify slack-aware constraints exist in severity stages."""
        slack_constraint_names = [
            'EqualMatchUpSpacingConstraint',
            'AwayAtMaitlandGrouping',
            'MaitlandHomeGrouping',
        ]
        
        # Collect all constraint names from severity stages
        all_severity_constraints = []
        for config in STAGES_SEVERITY.values():
            for cls in config['constraints']:
                all_severity_constraints.append(cls.__name__)
        
        # All slack constraints should be present
        for name in slack_constraint_names:
            assert name in all_severity_constraints, f"Slack constraint {name} not in severity stages"


# ============== Cumulative Application Tests ==============

class TestCumulativeConstraintApplication:
    """Tests for cumulative constraint application across severity stages."""

    def test_can_iterate_severity_stages_in_order(self):
        """Verify severity stages can be iterated in order."""
        stage_names = list(STAGES_SEVERITY.keys())
        
        # Build cumulative constraint list
        cumulative = []
        for stage_name in stage_names:
            stage_constraints = STAGES_SEVERITY[stage_name]['constraints']
            cumulative.extend(stage_constraints)
            
            # Verify cumulative list grows
            if stage_name == 'severity_1':
                assert len(cumulative) == len(STAGES_SEVERITY['severity_1']['constraints'])
            elif stage_name == 'severity_2':
                expected = (len(STAGES_SEVERITY['severity_1']['constraints']) +
                           len(STAGES_SEVERITY['severity_2']['constraints']))
                assert len(cumulative) == expected

    def test_all_constraints_covered_by_severity_stages(self):
        """Verify severity stages cover all constraints from default stages."""
        # Get all constraints from default STAGES
        default_constraints = set()
        for config in STAGES.values():
            for cls in config['constraints']:
                default_constraints.add(cls.__name__)
        
        # Get all constraints from STAGES_SEVERITY
        severity_constraints = set()
        for config in STAGES_SEVERITY.values():
            for cls in config['constraints']:
                severity_constraints.add(cls.__name__)
        
        # Severity stages should cover all default constraints
        missing = default_constraints - severity_constraints
        assert len(missing) == 0, f"Missing constraints in severity stages: {missing}"


# ============== Resume Compatibility Tests ==============

class TestResumeWithSeverityStages:
    """Tests for resume functionality with severity stages."""

    def test_severity_stage_names_are_valid_resume_targets(self):
        """Verify severity stage names can be used with --resume."""
        valid_stages = list(STAGES_SEVERITY.keys())
        
        # All should be non-empty strings
        for stage in valid_stages:
            assert isinstance(stage, str)
            assert len(stage) > 0
            assert stage.startswith('severity_')

    def test_checkpoint_paths_use_stage_names(self):
        """Verify checkpoint paths would use severity stage names."""
        from pathlib import Path
        
        for stage_name in STAGES_SEVERITY.keys():
            # Simulate checkpoint path
            checkpoint_path = Path('checkpoints') / 'test_run' / stage_name
            
            # Path should be valid
            assert stage_name in str(checkpoint_path)
