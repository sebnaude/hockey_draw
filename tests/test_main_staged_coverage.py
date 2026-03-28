"""
Tests for main_staged.py - structural validation, STAGES consistency,
constraint registration, checkpoint management, and configuration.

No mocks or patches. Uses real objects and config data.
"""
import os
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest
from ortools.sat.python import cp_model

from main_staged import (
    STAGES, STAGES_AI, STAGES_SEVERITY, STAGES_SEVERITY_AI, STAGES_UNIFIED,
    IntermediateSolutionCallback, CheckpointManager,
    _build_normalized_penalty, load_data,
)
from constraints.severity import CONSTRAINT_TO_SEVERITY


# =============================================================================
# STAGES / STAGES_AI Parity
# =============================================================================

class TestStagesAndStagesAIParity:
    """Verify STAGES and STAGES_AI are structural mirrors."""

    def test_same_keys(self):
        assert set(STAGES.keys()) == set(STAGES_AI.keys())

    def test_same_constraint_counts_per_stage(self):
        for key in STAGES:
            orig_count = len(STAGES[key]['constraints'])
            ai_count = len(STAGES_AI[key]['constraints'])
            assert orig_count == ai_count, (
                f"Stage {key}: STAGES has {orig_count} constraints, "
                f"STAGES_AI has {ai_count}"
            )

    def test_every_original_has_ai_mirror(self):
        for key in STAGES:
            orig_names = [c.__name__ for c in STAGES[key]['constraints']]
            ai_names = [c.__name__ for c in STAGES_AI[key]['constraints']]
            for orig in orig_names:
                expected_ai = orig + 'AI'
                matches = [a for a in ai_names if a == expected_ai or a == orig.replace('Constraint', 'ConstraintAI')]
                assert len(matches) >= 1, (
                    f"Stage {key}: no AI mirror for {orig}. AI names: {ai_names}"
                )

    def test_all_ai_names_end_with_ai(self):
        for key in STAGES_AI:
            for cls in STAGES_AI[key]['constraints']:
                assert cls.__name__.endswith('AI'), (
                    f"Stage {key}: {cls.__name__} doesn't end with 'AI'"
                )

    def test_all_stages_have_required_keys(self):
        required = {'name', 'description', 'constraints', 'required', 'use_callback'}
        for name, stage in STAGES.items():
            missing = required - set(stage.keys())
            assert not missing, f"STAGES[{name}] missing keys: {missing}"
        for name, stage in STAGES_AI.items():
            missing = required - set(stage.keys())
            assert not missing, f"STAGES_AI[{name}] missing keys: {missing}"


# =============================================================================
# STAGES_SEVERITY Coverage
# =============================================================================

class TestSeverityCoverage:
    """Verify severity stages cover all constraints from default STAGES."""

    def test_severity_covers_all_default_constraints(self):
        default_names = set()
        for stage in STAGES.values():
            for cls in stage['constraints']:
                default_names.add(cls.__name__)

        severity_names = set()
        for stage in STAGES_SEVERITY.values():
            for cls in stage['constraints']:
                severity_names.add(cls.__name__)

        missing = default_names - severity_names
        assert not missing, f"Constraints in STAGES but not STAGES_SEVERITY: {missing}"

    def test_severity_ai_covers_all_ai_constraints(self):
        ai_names = set()
        for stage in STAGES_AI.values():
            for cls in stage['constraints']:
                ai_names.add(cls.__name__)

        severity_ai_names = set()
        for stage in STAGES_SEVERITY_AI.values():
            for cls in stage['constraints']:
                severity_ai_names.add(cls.__name__)

        missing = ai_names - severity_ai_names
        assert not missing, f"Constraints in STAGES_AI but not STAGES_SEVERITY_AI: {missing}"

    def test_severity_ai_mirrors_severity(self):
        assert set(STAGES_SEVERITY.keys()) == set(STAGES_SEVERITY_AI.keys())
        for key in STAGES_SEVERITY:
            orig_count = len(STAGES_SEVERITY[key]['constraints'])
            ai_count = len(STAGES_SEVERITY_AI[key]['constraints'])
            assert orig_count == ai_count

    def test_no_duplicate_constraints_across_severity_levels(self):
        seen = {}
        for level_name, stage in STAGES_SEVERITY.items():
            for cls in stage['constraints']:
                name = cls.__name__
                assert name not in seen, (
                    f"{name} appears in both {seen[name]} and {level_name}"
                )
                seen[name] = level_name

    def test_five_severity_levels(self):
        assert len(STAGES_SEVERITY) == 5
        expected = {'severity_1', 'severity_2', 'severity_3', 'severity_4', 'severity_5'}
        assert set(STAGES_SEVERITY.keys()) == expected


# =============================================================================
# STAGES_UNIFIED
# =============================================================================

class TestStagesUnified:
    def test_has_three_phases(self):
        assert len(STAGES_UNIFIED) == 3

    def test_phase_keys(self):
        expected = {'phase_a_hard', 'phase_b_soft', 'phase_c_intraday'}
        assert set(STAGES_UNIFIED.keys()) == expected

    def test_each_has_required_keys(self):
        for name, stage in STAGES_UNIFIED.items():
            assert 'name' in stage
            assert 'description' in stage
            assert 'phase' in stage
            assert 'required' in stage

    def test_phase_labels(self):
        assert STAGES_UNIFIED['phase_a_hard']['phase'] == 'a'
        assert STAGES_UNIFIED['phase_b_soft']['phase'] == 'b'
        assert STAGES_UNIFIED['phase_c_intraday']['phase'] == 'c'

    def test_phase_a_is_required(self):
        assert STAGES_UNIFIED['phase_a_hard']['required'] is True

    def test_phase_c_is_not_required(self):
        assert STAGES_UNIFIED['phase_c_intraday']['required'] is False


# =============================================================================
# CONSTRAINT_TO_SEVERITY Consistency
# =============================================================================

class TestConstraintToSeverityConsistency:

    def test_all_stages_constraints_in_severity_map(self):
        all_names = set()
        for stage in STAGES.values():
            for cls in stage['constraints']:
                all_names.add(cls.__name__)
        for stage in STAGES_AI.values():
            for cls in stage['constraints']:
                all_names.add(cls.__name__)

        for name in all_names:
            assert name in CONSTRAINT_TO_SEVERITY, (
                f"{name} is in STAGES but not in CONSTRAINT_TO_SEVERITY"
            )

    def test_severity_map_entries_exist_in_some_stages(self):
        all_stages_names = set()
        for stages_dict in [STAGES, STAGES_AI, STAGES_SEVERITY, STAGES_SEVERITY_AI]:
            for stage in stages_dict.values():
                if 'constraints' in stage:
                    for cls in stage['constraints']:
                        all_stages_names.add(cls.__name__)

        for name in CONSTRAINT_TO_SEVERITY:
            assert name in all_stages_names, (
                f"{name} is in CONSTRAINT_TO_SEVERITY but not in any STAGES dict"
            )

    def test_severity_levels_are_valid(self):
        for name, level in CONSTRAINT_TO_SEVERITY.items():
            assert 1 <= level <= 5, f"{name} has invalid severity level {level}"

    def test_severity_levels_match_stage_placement(self):
        for level_name, stage in STAGES_SEVERITY.items():
            expected_level = int(level_name.split('_')[1])
            for cls in stage['constraints']:
                actual = CONSTRAINT_TO_SEVERITY.get(cls.__name__)
                assert actual == expected_level, (
                    f"{cls.__name__} in {level_name} but severity map says level {actual}"
                )


# =============================================================================
# _build_normalized_penalty
# =============================================================================

class TestBuildNormalizedPenalty:
    """_build_normalized_penalty returns list of (coefficient, var) pairs."""

    def test_empty_input(self):
        result = _build_normalized_penalty({})
        assert result == []

    def test_single_group_returns_pairs(self):
        penalties = {
            'TestConstraint': {'weight': 100, 'penalties': ['p1', 'p2', 'p3']}
        }
        result = _build_normalized_penalty(penalties)
        assert isinstance(result, list)
        assert len(result) == 3
        # Each item is (coefficient, var)
        for coeff, var in result:
            assert isinstance(coeff, (int, float))

    def test_empty_penalty_list_returns_empty(self):
        penalties = {'TestConstraint': {'weight': 100, 'penalties': []}}
        result = _build_normalized_penalty(penalties)
        assert result == []

    def test_multiple_groups(self):
        penalties = {
            'A': {'weight': 100, 'penalties': ['p1', 'p2']},
            'B': {'weight': 200, 'penalties': ['p1']},
        }
        result = _build_normalized_penalty(penalties)
        assert len(result) == 3  # 2 + 1

    def test_weight_normalization(self):
        """Higher weight should produce higher coefficients."""
        penalties = {
            'Low': {'weight': 10, 'penalties': ['p1']},
            'High': {'weight': 1000, 'penalties': ['p2']},
        }
        result = _build_normalized_penalty(penalties)
        coeffs = {var: coeff for coeff, var in result}
        assert coeffs['p2'] > coeffs['p1']


# =============================================================================
# CheckpointManager
# =============================================================================

class TestCheckpointManager:

    def test_init_creates_directory(self, tmp_path):
        cp_dir = str(tmp_path / 'checkpoints')
        mgr = CheckpointManager(cp_dir)
        assert mgr.checkpoint_dir.exists()

    def test_get_run_dir_auto(self, tmp_path):
        cp_dir = str(tmp_path / 'checkpoints')
        mgr = CheckpointManager(cp_dir)
        run_dir = mgr.get_run_dir()
        assert run_dir.exists()
        assert 'run_' in run_dir.name

    def test_get_run_dir_explicit(self, tmp_path):
        cp_dir = str(tmp_path / 'checkpoints')
        mgr = CheckpointManager(cp_dir)
        run_dir = mgr.get_run_dir('my_run')
        assert 'my_run' in str(run_dir)

    def test_load_nonexistent_stage(self, tmp_path):
        cp_dir = str(tmp_path / 'checkpoints')
        mgr = CheckpointManager(cp_dir)
        run_dir = mgr.get_run_dir('test_run')
        result = mgr.load_stage(run_dir, 'nonexistent')
        assert result is None


# =============================================================================
# IntermediateSolutionCallback
# =============================================================================

class TestIntermediateSolutionCallback:

    def test_instantiation(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / 'cp'))
        run_dir = mgr.get_run_dir('test')
        cb = IntermediateSolutionCallback(
            X={}, checkpoint_manager=mgr, run_dir=run_dir,
            stage_name='test', data={}
        )
        assert cb is not None

    def test_custom_save_interval(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / 'cp'))
        run_dir = mgr.get_run_dir('test')
        cb = IntermediateSolutionCallback(
            X={}, checkpoint_manager=mgr, run_dir=run_dir,
            stage_name='test', data={}, save_interval=5
        )
        assert cb.save_interval == 5 or hasattr(cb, '_save_interval')


# =============================================================================
# Constraint Exclusion Logic
# =============================================================================

class TestConstraintExclusionLogic:

    def test_excluding_original_removes_both(self):
        exclude = ['FooConstraint']
        exclude_set = set()
        for name in exclude:
            exclude_set.add(name)
            if not name.endswith('AI'):
                exclude_set.add(name + 'AI')
            else:
                exclude_set.add(name[:-2])

        assert 'FooConstraint' in exclude_set
        assert 'FooConstraintAI' in exclude_set

    def test_excluding_ai_removes_both(self):
        exclude = ['FooConstraintAI']
        exclude_set = set()
        for name in exclude:
            exclude_set.add(name)
            if not name.endswith('AI'):
                exclude_set.add(name + 'AI')
            else:
                exclude_set.add(name[:-2])

        assert 'FooConstraint' in exclude_set
        assert 'FooConstraintAI' in exclude_set

    def test_empty_exclusion_keeps_all(self):
        constraints = ['A', 'B', 'C']
        exclude_set = set()
        filtered = [c for c in constraints if c not in exclude_set]
        assert len(filtered) == 3


# =============================================================================
# load_data
# =============================================================================

class TestLoadData:

    def test_returns_dict(self):
        data = load_data(2026)
        assert isinstance(data, dict)

    def test_has_teams(self):
        data = load_data(2026)
        assert 'teams' in data and len(data['teams']) > 0

    def test_has_grades(self):
        data = load_data(2026)
        assert 'grades' in data and len(data['grades']) > 0

    def test_has_timeslots(self):
        data = load_data(2026)
        assert 'timeslots' in data and len(data['timeslots']) > 0

    def test_has_clubs(self):
        data = load_data(2026)
        assert 'clubs' in data and len(data['clubs']) > 0

    def test_has_fields(self):
        data = load_data(2026)
        assert 'fields' in data and len(data['fields']) > 0

    def test_has_year(self):
        data = load_data(2026)
        assert data['year'] == 2026

    def test_has_num_rounds(self):
        data = load_data(2026)
        assert 'num_rounds' in data and isinstance(data['num_rounds'], dict)

    def test_invalid_year_raises(self):
        with pytest.raises(Exception):
            load_data(9999)


# =============================================================================
# Constraint Slack Structure
# =============================================================================

class TestConstraintSlackStructure:
    SLACK_CONSTRAINTS = [
        'EqualMatchUpSpacingConstraint',
        'AwayAtMaitlandGrouping',
        'MaitlandHomeGrouping',
        'ClubVsClubAlignment',
        'MaximiseClubsPerTimeslotBroadmeadow',
        'MinimiseClubsOnAFieldBroadmeadow',
        'ClubGameSpread',
    ]

    def test_all_slack_constraints_in_stages(self):
        all_names = set()
        for stage in STAGES.values():
            for cls in stage['constraints']:
                all_names.add(cls.__name__)
        for name in self.SLACK_CONSTRAINTS:
            assert name in all_names, f"Slack constraint {name} not in STAGES"

    def test_all_slack_constraints_in_severity_map(self):
        for name in self.SLACK_CONSTRAINTS:
            assert name in CONSTRAINT_TO_SEVERITY


# =============================================================================
# Constraint Count Sanity
# =============================================================================

class TestConstraintCountSanity:

    def test_total_constraint_count(self):
        total = sum(len(s['constraints']) for s in STAGES.values())
        assert total == 19, f"Expected 19 constraints, got {total}"

    def test_ai_count_matches(self):
        total_orig = sum(len(s['constraints']) for s in STAGES.values())
        total_ai = sum(len(s['constraints']) for s in STAGES_AI.values())
        assert total_orig == total_ai

    def test_severity_count_matches(self):
        total_orig = sum(len(s['constraints']) for s in STAGES.values())
        total_sev = sum(len(s['constraints']) for s in STAGES_SEVERITY.values())
        assert total_orig == total_sev

    def test_all_constraints_have_apply_method(self):
        for stages_dict in [STAGES, STAGES_AI]:
            for stage in stages_dict.values():
                for cls in stage['constraints']:
                    assert hasattr(cls, 'apply'), f"{cls.__name__} has no apply()"
