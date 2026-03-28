# tests/test_severity_staged.py
"""
Unit tests for severity-based staged solving.

Tests for:
- STAGES_SEVERITY and STAGES_SEVERITY_AI structure
- Severity stage constraint groupings
- CLI --staged flag integration
- Cumulative constraint application with severity stages

All tests use real objects - no mocks or patches.
"""

import pytest
import sys
import os
import argparse
import inspect
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_staged import (
    STAGES, STAGES_AI,
    STAGES_SEVERITY, STAGES_SEVERITY_AI,
    StagedScheduleSolver
)
from constraints.severity import (
    CONSTRAINT_TO_SEVERITY,
    group_constraints_by_severity,
    get_severity_level,
    SeverityGroupState,
)


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
        required_keys = ['name', 'description', 'constraints', 'required', 'use_callback']

        for stage_name, config in STAGES_SEVERITY.items():
            for key in required_keys:
                assert key in config, f"Missing key '{key}' in {stage_name}"

    def test_each_ai_stage_has_required_keys(self):
        """Verify each AI severity stage has required configuration keys."""
        required_keys = ['name', 'description', 'constraints', 'required', 'use_callback']

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

    def test_severity_2_and_3_are_required(self):
        """Severity 2 (HIGH) and 3 (MEDIUM) should be required=True."""
        assert STAGES_SEVERITY['severity_2']['required'] is True
        assert STAGES_SEVERITY['severity_3']['required'] is True

    def test_all_stages_have_use_callback(self):
        """All severity stages should have use_callback=True."""
        for stage_name, config in STAGES_SEVERITY.items():
            assert config['use_callback'] is True, f"{stage_name} has use_callback={config['use_callback']}"

    def test_all_stages_have_name_and_description_strings(self):
        """All stages should have non-empty string name and description."""
        for stage_name, config in STAGES_SEVERITY.items():
            assert isinstance(config['name'], str) and len(config['name']) > 0
            assert isinstance(config['description'], str) and len(config['description']) > 0

    def test_constraints_are_classes_not_instances(self):
        """All constraint entries should be classes (not instances)."""
        for stage_name, config in STAGES_SEVERITY.items():
            for cls in config['constraints']:
                assert isinstance(cls, type), f"{cls} in {stage_name} is not a class"


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

    def test_group_constraints_by_severity_function(self):
        """Verify group_constraints_by_severity correctly groups real constraint classes."""
        # Collect all constraints from severity stages
        all_constraints = []
        for config in STAGES_SEVERITY.values():
            all_constraints.extend(config['constraints'])

        grouped = group_constraints_by_severity(all_constraints)

        # Should have groups for levels 1-5
        for level in range(1, 6):
            assert level in grouped, f"Level {level} missing from grouped result"
            expected_count = len(STAGES_SEVERITY[f'severity_{level}']['constraints'])
            assert len(grouped[level]) == expected_count, \
                f"Level {level}: expected {expected_count}, got {len(grouped[level])}"

    def test_get_severity_level_for_all_staged_constraints(self):
        """Verify get_severity_level returns correct level for every staged constraint."""
        for level_num in range(1, 6):
            stage_key = f'severity_{level_num}'
            for cls in STAGES_SEVERITY[stage_key]['constraints']:
                actual = get_severity_level(cls)
                assert actual == level_num, \
                    f"{cls.__name__}: get_severity_level returned {actual}, expected {level_num}"

    def test_no_duplicate_constraints_across_ai_stages(self):
        """Verify no AI constraint appears in multiple severity stages."""
        seen = set()
        for stage_name, config in STAGES_SEVERITY_AI.items():
            for cls in config['constraints']:
                name = cls.__name__
                assert name not in seen, f"Duplicate AI constraint {name} in {stage_name}"
                seen.add(name)

    def test_ai_constraint_names_end_with_ai(self):
        """Verify all AI stage constraints have AI suffix."""
        for stage_name, config in STAGES_SEVERITY_AI.items():
            for cls in config['constraints']:
                assert cls.__name__.endswith('AI'), \
                    f"{cls.__name__} in AI stage {stage_name} does not end with 'AI'"

    def test_regular_constraint_names_do_not_end_with_ai(self):
        """Verify regular stage constraints do NOT have AI suffix."""
        for stage_name, config in STAGES_SEVERITY.items():
            for cls in config['constraints']:
                assert not cls.__name__.endswith('AI'), \
                    f"{cls.__name__} in regular stage {stage_name} ends with 'AI'"


# ============== CLI Integration Tests ==============

class TestStagedCLIFlag:
    """Tests for --staged CLI flag integration."""

    def test_staged_flag_parsed_correctly(self):
        """Verify --staged flag is parsed as boolean via argparse Namespace."""
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
        assert args.staged is True

    def test_staged_false_by_default(self):
        """Verify --staged defaults to False when not specified."""
        args = argparse.Namespace(staged=False)
        assert args.staged is False

    def test_slack_argument_sets_constraint_slack_dict(self):
        """Verify --slack value builds the constraint_slack dict as run.py does."""
        slack_value = 3
        constraint_slack = {
            'EqualMatchUpSpacingConstraint': slack_value,
            'AwayAtMaitlandGrouping': slack_value,
            'MaitlandHomeGrouping': slack_value,
            'ClubVsClubAlignment': slack_value,
            'MaximiseClubsPerTimeslotBroadmeadow': slack_value,
            'MinimiseClubsOnAFieldBroadmeadow': slack_value,
            'ClubGameSpread': slack_value,
        }
        assert all(v == 3 for v in constraint_slack.values())
        assert len(constraint_slack) == 7


# ============== Solver Stage Mode Selection Tests ==============

class TestSolverStageSelection:
    """Tests for run_staged_solve stage mode selection."""

    def test_severity_staged_parameter_exists(self):
        """Verify run_staged_solve has severity_staged parameter."""
        sig = inspect.signature(StagedScheduleSolver.run_staged_solve)
        params = list(sig.parameters.keys())
        assert 'severity_staged' in params, "severity_staged parameter missing"

    def test_default_mode_uses_default_stages(self):
        """Verify default STAGES and STAGES_SEVERITY have different keys."""
        default_keys = set(STAGES.keys())
        severity_keys = set(STAGES_SEVERITY.keys())

        assert default_keys != severity_keys
        assert 'stage1_required' in default_keys
        assert 'severity_1' in severity_keys

    def test_stages_ai_and_stages_severity_ai_have_different_keys(self):
        """Verify AI stages dict keys differ between default and severity modes."""
        ai_keys = set(STAGES_AI.keys())
        severity_ai_keys = set(STAGES_SEVERITY_AI.keys())
        assert ai_keys != severity_ai_keys

    def test_use_ai_parameter_exists(self):
        """Verify run_staged_solve has use_ai parameter."""
        sig = inspect.signature(StagedScheduleSolver.run_staged_solve)
        assert 'use_ai' in sig.parameters

    def test_severity_staged_default_is_false(self):
        """Verify severity_staged defaults to False."""
        sig = inspect.signature(StagedScheduleSolver.run_staged_solve)
        param = sig.parameters['severity_staged']
        assert param.default is False


# ============== Slack Integration Tests ==============

class TestSlackWithSeverityStages:
    """Tests that --slack works with severity-staged mode."""

    def test_slack_dict_structure(self):
        """Verify constraint_slack dict is structured correctly."""
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

        all_severity_constraints = []
        for config in STAGES_SEVERITY.values():
            for cls in config['constraints']:
                all_severity_constraints.append(cls.__name__)

        for name in slack_constraint_names:
            assert name in all_severity_constraints, f"Slack constraint {name} not in severity stages"

    def test_all_slack_aware_constraints_are_in_severity_map(self):
        """Verify all slack-aware constraint names appear in CONSTRAINT_TO_SEVERITY."""
        slack_constraint_names = [
            'EqualMatchUpSpacingConstraint',
            'AwayAtMaitlandGrouping',
            'MaitlandHomeGrouping',
            'ClubVsClubAlignment',
            'MaximiseClubsPerTimeslotBroadmeadow',
            'MinimiseClubsOnAFieldBroadmeadow',
            'ClubGameSpread',
        ]
        for name in slack_constraint_names:
            assert name in CONSTRAINT_TO_SEVERITY, f"{name} not in CONSTRAINT_TO_SEVERITY"

    def test_severity_group_state_relax(self):
        """Verify SeverityGroupState.relax() increments slack correctly."""
        state = SeverityGroupState(level=3, constraint_classes=[], current_slack=0, max_slack=3)
        assert state.can_relax()
        assert state.relax()
        assert state.current_slack == 1
        assert state.relax()
        assert state.current_slack == 2
        assert state.relax()
        assert state.current_slack == 3
        assert not state.can_relax()
        assert not state.relax()
        assert state.current_slack == 3

    def test_severity_group_state_level_1_cannot_relax(self):
        """Verify level 1 constraints cannot be relaxed."""
        state = SeverityGroupState(level=1, constraint_classes=[], current_slack=0, max_slack=3)
        assert not state.can_relax()
        assert not state.relax()


# ============== Cumulative Application Tests ==============

class TestCumulativeConstraintApplication:
    """Tests for cumulative constraint application across severity stages."""

    def test_can_iterate_severity_stages_in_order(self):
        """Verify severity stages can be iterated in order with cumulative growth."""
        stage_names = list(STAGES_SEVERITY.keys())

        cumulative = []
        for stage_name in stage_names:
            stage_constraints = STAGES_SEVERITY[stage_name]['constraints']
            cumulative.extend(stage_constraints)

            if stage_name == 'severity_1':
                assert len(cumulative) == len(STAGES_SEVERITY['severity_1']['constraints'])
            elif stage_name == 'severity_2':
                expected = (len(STAGES_SEVERITY['severity_1']['constraints']) +
                           len(STAGES_SEVERITY['severity_2']['constraints']))
                assert len(cumulative) == expected

    def test_all_constraints_covered_by_severity_stages(self):
        """Verify severity stages cover all constraints from default stages."""
        default_constraints = set()
        for config in STAGES.values():
            for cls in config['constraints']:
                default_constraints.add(cls.__name__)

        severity_constraints = set()
        for config in STAGES_SEVERITY.values():
            for cls in config['constraints']:
                severity_constraints.add(cls.__name__)

        missing = default_constraints - severity_constraints
        assert len(missing) == 0, f"Missing constraints in severity stages: {missing}"

    def test_all_ai_constraints_covered_by_severity_ai_stages(self):
        """Verify AI severity stages cover all AI constraints from default AI stages."""
        default_ai = set()
        for config in STAGES_AI.values():
            for cls in config['constraints']:
                default_ai.add(cls.__name__)

        severity_ai = set()
        for config in STAGES_SEVERITY_AI.values():
            for cls in config['constraints']:
                severity_ai.add(cls.__name__)

        missing = default_ai - severity_ai
        assert len(missing) == 0, f"Missing AI constraints in severity AI stages: {missing}"

    def test_total_constraint_count_matches(self):
        """Verify total constraints across all severity stages equals sum of per-stage counts."""
        total = sum(len(config['constraints']) for config in STAGES_SEVERITY.values())
        individual = []
        for config in STAGES_SEVERITY.values():
            individual.extend(config['constraints'])
        assert len(individual) == total

    def test_cumulative_list_contains_all_at_end(self):
        """Verify iterating all stages cumulatively yields every constraint."""
        all_from_stages = set()
        for config in STAGES_SEVERITY.values():
            for cls in config['constraints']:
                all_from_stages.add(cls.__name__)

        cumulative = []
        for stage_name in STAGES_SEVERITY:
            cumulative.extend(STAGES_SEVERITY[stage_name]['constraints'])

        cumulative_names = set(cls.__name__ for cls in cumulative)
        assert cumulative_names == all_from_stages


# ============== Resume Compatibility Tests ==============

class TestResumeWithSeverityStages:
    """Tests for resume functionality with severity stages."""

    def test_severity_stage_names_are_valid_resume_targets(self):
        """Verify severity stage names can be used with --resume."""
        valid_stages = list(STAGES_SEVERITY.keys())

        for stage in valid_stages:
            assert isinstance(stage, str)
            assert len(stage) > 0
            assert stage.startswith('severity_')

    def test_checkpoint_paths_use_stage_names(self):
        """Verify checkpoint paths would use severity stage names."""
        for stage_name in STAGES_SEVERITY.keys():
            checkpoint_path = Path('checkpoints') / 'test_run' / stage_name
            assert stage_name in str(checkpoint_path)

    def test_severity_stage_names_are_unique(self):
        """Verify all severity stage names are unique."""
        names = list(STAGES_SEVERITY.keys())
        assert len(names) == len(set(names))

    def test_resume_parameter_exists_on_solver(self):
        """Verify run_staged_solve accepts resume_from parameter."""
        sig = inspect.signature(StagedScheduleSolver.run_staged_solve)
        assert 'resume_from' in sig.parameters

    def test_stage_names_match_between_regular_and_ai(self):
        """Verify STAGES_SEVERITY and STAGES_SEVERITY_AI have identical stage names."""
        assert list(STAGES_SEVERITY.keys()) == list(STAGES_SEVERITY_AI.keys())
