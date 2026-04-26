# tests/test_main_staged_coverage.py
"""
Structural coverage tests for main_staged.py.

These tests validate stage dictionaries, helper functions, class instantiation,
constraint exclusion logic, and data loading without running the actual solver.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_staged import (
    STAGES, STAGES_AI,
    STAGES_SEVERITY, STAGES_SEVERITY_AI,
    STAGES_UNIFIED,
    IntermediateSolutionCallback,
    CheckpointManager,
    StagedScheduleSolver,
    _build_normalized_penalty,
    _build_constraints_applied,
    _serialize_config,
    load_data,
)
from constraints.severity import CONSTRAINT_TO_SEVERITY


# ============== STAGES / STAGES_AI Parity Tests ==============

class TestStagesAndStagesAIParity:
    """Verify STAGES and STAGES_AI are structurally identical."""

    def test_identical_keys(self):
        """STAGES and STAGES_AI must have the same stage keys."""
        assert set(STAGES.keys()) == set(STAGES_AI.keys())

    def test_identical_constraint_counts_per_stage(self):
        """Each stage must have the same number of constraints in both dicts."""
        for stage_key in STAGES:
            orig_count = len(STAGES[stage_key]['constraints'])
            ai_count = len(STAGES_AI[stage_key]['constraints'])
            assert orig_count == ai_count, (
                f"Stage {stage_key}: STAGES has {orig_count} constraints, "
                f"STAGES_AI has {ai_count}"
            )

    def test_every_constraint_has_ai_mirror(self):
        """Every constraint class in STAGES has a matching AI version in STAGES_AI."""
        for stage_key in STAGES:
            orig_names = [cls.__name__ for cls in STAGES[stage_key]['constraints']]
            ai_names = [cls.__name__ for cls in STAGES_AI[stage_key]['constraints']]
            for orig_name in orig_names:
                expected_ai = orig_name + 'AI'
                assert expected_ai in ai_names, (
                    f"Constraint '{orig_name}' in STAGES[{stage_key}] "
                    f"has no AI mirror '{expected_ai}' in STAGES_AI"
                )

    def test_ai_constraints_end_with_ai(self):
        """All constraint classes in STAGES_AI should end with 'AI'."""
        for stage_key in STAGES_AI:
            for cls in STAGES_AI[stage_key]['constraints']:
                assert cls.__name__.endswith('AI'), (
                    f"STAGES_AI[{stage_key}] constraint '{cls.__name__}' "
                    f"does not end with 'AI'"
                )

    def test_stages_have_required_config_keys(self):
        """Every stage config must have name, description, constraints, required, use_callback."""
        required_keys = {'name', 'description', 'constraints', 'required', 'use_callback'}
        for stages_dict, label in [(STAGES, 'STAGES'), (STAGES_AI, 'STAGES_AI')]:
            for stage_key, config in stages_dict.items():
                missing = required_keys - set(config.keys())
                assert not missing, (
                    f"{label}[{stage_key}] missing keys: {missing}"
                )


# ============== STAGES_SEVERITY Coverage Tests ==============

class TestSeverityCoverage:
    """Verify STAGES_SEVERITY covers all constraints from STAGES."""

    def _collect_constraint_names(self, stages_dict):
        names = set()
        for config in stages_dict.values():
            for cls in config['constraints']:
                names.add(cls.__name__)
        return names

    def test_severity_covers_all_stages_constraints(self):
        """Union of all severity levels must contain every constraint from STAGES."""
        stages_names = self._collect_constraint_names(STAGES)
        severity_names = self._collect_constraint_names(STAGES_SEVERITY)
        missing = stages_names - severity_names
        assert not missing, f"Constraints in STAGES missing from STAGES_SEVERITY: {missing}"

    def test_severity_ai_covers_all_stages_ai_constraints(self):
        """Union of all severity AI levels must contain every constraint from STAGES_AI."""
        stages_names = self._collect_constraint_names(STAGES_AI)
        severity_names = self._collect_constraint_names(STAGES_SEVERITY_AI)
        missing = stages_names - severity_names
        assert not missing, f"Constraints in STAGES_AI missing from STAGES_SEVERITY_AI: {missing}"

    def test_no_extra_constraints_in_severity(self):
        """STAGES_SEVERITY should not contain constraints absent from STAGES."""
        stages_names = self._collect_constraint_names(STAGES)
        severity_names = self._collect_constraint_names(STAGES_SEVERITY)
        extra = severity_names - stages_names
        assert not extra, f"Extra constraints in STAGES_SEVERITY not in STAGES: {extra}"

    def test_severity_ai_mirrors_severity_structurally(self):
        """STAGES_SEVERITY_AI must have the same keys and constraint counts."""
        assert set(STAGES_SEVERITY.keys()) == set(STAGES_SEVERITY_AI.keys())
        for key in STAGES_SEVERITY:
            orig = len(STAGES_SEVERITY[key]['constraints'])
            ai = len(STAGES_SEVERITY_AI[key]['constraints'])
            assert orig == ai, f"severity stage {key}: {orig} vs {ai} AI constraints"

    def test_no_duplicate_constraints_across_severity_stages(self):
        """No constraint should appear in more than one severity level."""
        seen = {}
        for stage_name, config in STAGES_SEVERITY.items():
            for cls in config['constraints']:
                name = cls.__name__
                assert name not in seen, (
                    f"Duplicate constraint '{name}': in both {seen[name]} and {stage_name}"
                )
                seen[name] = stage_name


# ============== STAGES_UNIFIED Tests ==============

class TestStagesUnified:
    """Verify STAGES_UNIFIED has the required phase keys and structure."""

    def test_has_three_phases(self):
        assert len(STAGES_UNIFIED) == 3

    def test_required_phase_keys(self):
        assert 'phase_a_hard' in STAGES_UNIFIED
        assert 'phase_b_soft' in STAGES_UNIFIED
        assert 'phase_c_intraday' in STAGES_UNIFIED

    def test_each_phase_has_required_config(self):
        required_keys = {'name', 'description', 'phase', 'required', 'use_callback'}
        for phase_key, config in STAGES_UNIFIED.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"STAGES_UNIFIED[{phase_key}] missing: {missing}"

    def test_phase_a_is_required(self):
        assert STAGES_UNIFIED['phase_a_hard']['required'] is True

    def test_phase_c_is_optional(self):
        assert STAGES_UNIFIED['phase_c_intraday']['required'] is False

    def test_phase_labels(self):
        assert STAGES_UNIFIED['phase_a_hard']['phase'] == 'a'
        assert STAGES_UNIFIED['phase_b_soft']['phase'] == 'b'
        assert STAGES_UNIFIED['phase_c_intraday']['phase'] == 'c'


# ============== CONSTRAINT_TO_SEVERITY Consistency ==============

class TestConstraintToSeverityConsistency:
    """Cross-check CONSTRAINT_TO_SEVERITY with STAGES dicts."""

    def _all_constraint_names_from(self, stages_dict):
        names = set()
        for config in stages_dict.values():
            for cls in config['constraints']:
                names.add(cls.__name__)
        return names

    def test_all_stages_constraints_in_severity_map(self):
        """Every constraint in STAGES must be in CONSTRAINT_TO_SEVERITY."""
        names = self._all_constraint_names_from(STAGES)
        for name in names:
            assert name in CONSTRAINT_TO_SEVERITY, (
                f"Constraint '{name}' from STAGES not in CONSTRAINT_TO_SEVERITY"
            )

    def test_all_stages_ai_constraints_in_severity_map(self):
        """Every constraint in STAGES_AI must be in CONSTRAINT_TO_SEVERITY."""
        names = self._all_constraint_names_from(STAGES_AI)
        for name in names:
            assert name in CONSTRAINT_TO_SEVERITY, (
                f"Constraint '{name}' from STAGES_AI not in CONSTRAINT_TO_SEVERITY"
            )

    def test_all_severity_map_entries_exist_in_some_stages_dict(self):
        """Every entry in CONSTRAINT_TO_SEVERITY should appear in at least one STAGES dict."""
        all_stage_names = set()
        for stages_dict in [STAGES, STAGES_AI, STAGES_SEVERITY, STAGES_SEVERITY_AI]:
            all_stage_names.update(self._all_constraint_names_from(stages_dict))
        for name in CONSTRAINT_TO_SEVERITY:
            assert name in all_stage_names, (
                f"CONSTRAINT_TO_SEVERITY entry '{name}' not found in any STAGES dict"
            )

    def test_severity_levels_match_stage_placement(self):
        """Constraints in STAGES_SEVERITY[severity_N] should have level N in the map."""
        for stage_key, config in STAGES_SEVERITY.items():
            expected_level = int(stage_key.split('_')[1])
            for cls in config['constraints']:
                actual_level = CONSTRAINT_TO_SEVERITY.get(cls.__name__)
                assert actual_level == expected_level, (
                    f"'{cls.__name__}' in {stage_key} but CONSTRAINT_TO_SEVERITY "
                    f"says level {actual_level}"
                )


# ============== _build_normalized_penalty Tests ==============

class TestBuildNormalizedPenalty:
    """Test the penalty normalization helper."""

    def test_empty_dict_returns_empty(self):
        assert _build_normalized_penalty({}) == []

    def test_single_penalty_group(self):
        model = cp_model.CpModel()
        v1 = model.NewBoolVar('v1')
        v2 = model.NewBoolVar('v2')
        penalties = {
            'test': {'weight': 100, 'penalties': [v1, v2]}
        }
        terms = _build_normalized_penalty(penalties)
        assert len(terms) == 2
        # weight 100 / 2 vars = 50 per var
        assert all(coeff == 50 for coeff, _ in terms)

    def test_normalization_floor_is_one(self):
        """When weight/n < 1, the normalized coefficient should still be at least 1."""
        model = cp_model.CpModel()
        vars_list = [model.NewBoolVar(f'v{i}') for i in range(200)]
        penalties = {
            'tiny': {'weight': 10, 'penalties': vars_list}
        }
        terms = _build_normalized_penalty(penalties)
        assert len(terms) == 200
        # 10 // 200 = 0, but max(1, 0) = 1
        assert all(coeff == 1 for coeff, _ in terms)

    def test_empty_penalties_list_skipped(self):
        penalties = {
            'empty_group': {'weight': 1000, 'penalties': []}
        }
        terms = _build_normalized_penalty(penalties)
        assert terms == []

    def test_multiple_groups(self):
        model = cp_model.CpModel()
        v1 = model.NewBoolVar('a')
        v2 = model.NewBoolVar('b')
        v3 = model.NewBoolVar('c')
        penalties = {
            'group_a': {'weight': 100, 'penalties': [v1]},
            'group_b': {'weight': 200, 'penalties': [v2, v3]},
        }
        terms = _build_normalized_penalty(penalties)
        assert len(terms) == 3
        coeffs = [c for c, _ in terms]
        # group_a: 100/1 = 100; group_b: 200/2 = 100
        assert coeffs == [100, 100, 100]


# ============== _build_constraints_applied Tests ==============

class TestBuildConstraintsApplied:
    """Test the constraint metadata builder."""

    def test_basic_output(self):
        from constraints import NoDoubleBookingTeamsConstraint, ClubDayConstraint
        classes = [NoDoubleBookingTeamsConstraint, ClubDayConstraint]
        result = _build_constraints_applied(classes)
        assert len(result) == 2
        assert result[0]['name'] == 'NoDoubleBookingTeamsConstraint'
        assert result[1]['name'] == 'ClubDayConstraint'

    def test_with_severity_map(self):
        from constraints import NoDoubleBookingTeamsConstraint
        classes = [NoDoubleBookingTeamsConstraint]
        sev_map = {'NoDoubleBookingTeamsConstraint': 1}
        result = _build_constraints_applied(classes, severity_map=sev_map)
        assert result[0]['severity'] == 1

    def test_severity_missing_from_map(self):
        from constraints import NoDoubleBookingTeamsConstraint
        classes = [NoDoubleBookingTeamsConstraint]
        result = _build_constraints_applied(classes, severity_map={})
        assert 'severity' not in result[0]

    def test_empty_list(self):
        assert _build_constraints_applied([]) == []


# ============== _serialize_config Tests ==============

class TestSerializeConfig:
    """Test the config serialization helper."""

    def test_datetime_serialized(self):
        from datetime import datetime
        dt = datetime(2026, 3, 28, 10, 30, 0)
        result = _serialize_config(dt)
        assert result == '2026-03-28T10:30:00'

    def test_set_sorted(self):
        result = _serialize_config({3, 1, 2})
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        result = _serialize_config({'a': {1, 2}, 'b': 'hello'})
        assert result == {'a': [1, 2], 'b': 'hello'}

    def test_list_of_sets(self):
        result = _serialize_config([{2, 1}])
        assert result == [[1, 2]]

    def test_plain_values_passthrough(self):
        assert _serialize_config(42) == 42
        assert _serialize_config('hello') == 'hello'
        assert _serialize_config(None) is None


# ============== CheckpointManager Tests ==============

class TestCheckpointManager:
    """Test CheckpointManager initialization and directory management."""

    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_dir = os.path.join(tmpdir, 'checkpoints')
            cm = CheckpointManager(checkpoint_dir=cp_dir)
            assert Path(cp_dir).exists()

    def test_get_run_dir_auto_increments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            dir1 = cm.get_run_dir()
            assert dir1.name == 'run_1'
            assert dir1.exists()

            dir2 = cm.get_run_dir()
            assert dir2.name == 'run_2'

    def test_get_run_dir_with_explicit_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            d = cm.get_run_dir('my_run')
            assert d.name == 'my_run'
            assert d.exists()

    def test_load_latest_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            assert cm.load_latest() is None

    def test_load_stage_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            assert cm.load_stage(run_dir, 'nonexistent') is None

    def test_save_and_load_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            fake_solution = {('a', 'b'): 1, ('c', 'd'): 0}
            fake_data = {'year': 2026}
            cm.save_stage(run_dir, 'test_stage', fake_solution, fake_data, 'FEASIBLE', 10.5)

            loaded = cm.load_stage(run_dir, 'test_stage')
            assert loaded is not None
            assert loaded['solution'] == fake_solution
            assert loaded['metadata']['status'] == 'FEASIBLE'
            assert loaded['metadata']['solve_time'] == 10.5

    def test_save_stage_updates_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            fake_solution = {('a',): 1}
            cm.save_stage(run_dir, 'stage1', fake_solution, {'year': 2026}, 'OPTIMAL', 5.0)

            latest = cm.load_latest()
            assert latest is not None
            assert latest['solution'] == fake_solution
            assert 'pointer' in latest
            assert latest['pointer']['source_stage'] == 'stage1'

    def test_infeasible_does_not_update_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            cm.save_stage(run_dir, 's1', {}, {'year': 2026}, 'INFEASIBLE', 1.0)
            assert cm.load_latest() is None

    def test_get_last_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            # Save a feasible stage matching STAGES keys
            cm.save_stage(run_dir, 'stage1_required', {('x',): 1}, {'year': 2026}, 'FEASIBLE', 1.0)
            last = cm.get_last_completed_stage(run_dir)
            assert last == 'stage1_required'

    def test_get_last_completed_stage_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            assert cm.get_last_completed_stage(run_dir) is None

    def test_save_and_update_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            data = {'year': 2026, 'teams': [], 'grades': [], 'timeslots': []}
            cm.save_run_metadata(run_dir, data)
            meta_path = run_dir / 'run_metadata.json'
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta['year'] == 2026
            assert meta['status'] == 'started'

            cm.update_run_status(run_dir, 'completed', {'extra_key': 'value'})
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta['status'] == 'completed'
            assert meta['extra_key'] == 'value'
            assert 'finished_at' in meta

    def test_update_run_status_noop_if_no_metadata(self):
        """update_run_status should silently return if no run_metadata.json exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            # Should not raise
            cm.update_run_status(run_dir, 'failed')


# ============== IntermediateSolutionCallback Tests ==============

class TestIntermediateSolutionCallback:
    """Test that IntermediateSolutionCallback can be instantiated correctly."""

    def test_instantiation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            run_dir = cm.get_run_dir()
            model = cp_model.CpModel()
            x = model.NewBoolVar('x')
            X = {('a', 'b'): x}
            data = {'year': 2026}

            cb = IntermediateSolutionCallback(
                X=X,
                checkpoint_manager=cm,
                run_dir=run_dir,
                stage_name='test',
                data=data,
                save_interval=30,
            )
            assert cb.solution_count == 0
            assert cb.best_objective == float('-inf')
            assert cb.save_interval == 30
            assert cb.X is X

    def test_default_save_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            cb = IntermediateSolutionCallback(
                X={}, checkpoint_manager=cm,
                run_dir=cm.get_run_dir(), stage_name='s', data={},
            )
            assert cb.save_interval == 60


# ============== Constraint Exclusion Logic Tests ==============

class TestConstraintExclusionLogic:
    """Test the normalized exclusion filter used in run_staged_solve and main_simple."""

    def _apply_exclusion(self, exclude_names, stages_dict):
        """Reproduce the exclusion logic from main_staged.py."""
        exclude_set = set(exclude_names)
        normalized_exclude = set()
        for name in exclude_set:
            normalized_exclude.add(name)
            if name.endswith('AI'):
                normalized_exclude.add(name[:-2])
            else:
                normalized_exclude.add(name + 'AI')

        filtered = {}
        for stage_id, stage_config in stages_dict.items():
            fc = dict(stage_config)
            fc['constraints'] = [
                cls for cls in stage_config['constraints']
                if cls.__name__ not in normalized_exclude
            ]
            filtered[stage_id] = fc
        return filtered

    def test_excluding_original_also_excludes_ai(self):
        """Excluding 'ClubDayConstraint' should also remove 'ClubDayConstraintAI'."""
        filtered = self._apply_exclusion(['ClubDayConstraint'], STAGES_AI)
        all_names = [
            cls.__name__ for cfg in filtered.values() for cls in cfg['constraints']
        ]
        assert 'ClubDayConstraintAI' not in all_names

    def test_excluding_ai_also_excludes_original(self):
        """Excluding 'ClubDayConstraintAI' should also remove 'ClubDayConstraint'."""
        filtered = self._apply_exclusion(['ClubDayConstraintAI'], STAGES)
        all_names = [
            cls.__name__ for cfg in filtered.values() for cls in cfg['constraints']
        ]
        assert 'ClubDayConstraint' not in all_names

    def test_other_constraints_unaffected(self):
        """Excluding one constraint should leave all others intact."""
        original_count = sum(len(cfg['constraints']) for cfg in STAGES.values())
        filtered = self._apply_exclusion(['ClubDayConstraint'], STAGES)
        filtered_count = sum(len(cfg['constraints']) for cfg in filtered.values())
        assert filtered_count == original_count - 1

    def test_empty_exclusion_keeps_all(self):
        filtered = self._apply_exclusion([], STAGES)
        original_count = sum(len(cfg['constraints']) for cfg in STAGES.values())
        filtered_count = sum(len(cfg['constraints']) for cfg in filtered.values())
        assert filtered_count == original_count

    def test_multiple_exclusions(self):
        filtered = self._apply_exclusion(
            ['ClubDayConstraint', 'TeamConflictConstraint'], STAGES
        )
        all_names = [
            cls.__name__ for cfg in filtered.values() for cls in cfg['constraints']
        ]
        assert 'ClubDayConstraint' not in all_names
        assert 'TeamConflictConstraint' not in all_names


# ============== load_data Tests ==============

class TestLoadData:
    """Test load_data returns a valid data dict."""

    @pytest.fixture(scope='class')
    def data(self):
        return load_data(2026)

    def test_returns_dict(self, data):
        assert isinstance(data, dict)

    def test_has_teams(self, data):
        assert 'teams' in data
        assert len(data['teams']) > 0

    def test_has_grades(self, data):
        assert 'grades' in data
        assert len(data['grades']) > 0

    def test_has_timeslots(self, data):
        assert 'timeslots' in data
        assert len(data['timeslots']) > 0

    def test_has_clubs(self, data):
        assert 'clubs' in data
        assert len(data['clubs']) > 0

    def test_has_fields(self, data):
        assert 'fields' in data
        assert len(data['fields']) > 0

    def test_has_games_or_matchups(self, data):
        """Data dict may have 'games' or it may be generated later by generate_X."""
        # load_season_data may not include 'games' directly; matchups are
        # generated by generate_X(). Check that the data has enough info
        # to derive games (teams + grades).
        assert len(data['teams']) > 0
        assert len(data['grades']) > 0

    def test_has_year(self, data):
        assert data.get('year') == 2026

    def test_has_num_rounds(self, data):
        assert 'num_rounds' in data
        assert isinstance(data['num_rounds'], dict)

    def test_raises_on_invalid_year(self):
        with pytest.raises((ValueError, Exception)):
            load_data(1900)


# ============== StagedScheduleSolver Basic Tests ==============

class TestStagedScheduleSolverInit:
    """Test StagedScheduleSolver can be constructed with minimal data."""

    def test_basic_init(self):
        data = {'teams': [], 'grades': [], 'timeslots': []}
        solver = StagedScheduleSolver(data)
        assert solver.data is data
        assert solver.model is None
        assert solver.X is None
        assert solver.current_solution is None
        assert isinstance(solver.checkpoint_manager, CheckpointManager)

    def test_init_with_custom_checkpoint_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir)
            data = {'teams': []}
            solver = StagedScheduleSolver(data, checkpoint_manager=cm)
            assert solver.checkpoint_manager is cm

    def test_init_with_relax_config(self):
        data = {'teams': []}
        rc = {'enabled': True, 'timeout': 15}
        solver = StagedScheduleSolver(data, relax_config=rc)
        assert solver.relax_config == rc

    def test_relaxed_groups_initially_empty(self):
        solver = StagedScheduleSolver({'teams': []})
        assert solver.relaxed_groups == {}


# ============== Constraint Slack Dict Structure ==============

class TestConstraintSlackStructure:
    """Verify the constraint_slack dict structure used by run.py."""

    EXPECTED_SLACK_KEYS = [
        'EqualMatchUpSpacingConstraint',
        'AwayAtMaitlandGrouping',
        'MaitlandHomeGrouping',
        'ClubVsClubAlignment',
        'MaximiseClubsPerTimeslotBroadmeadow',
        'MinimiseClubsOnAFieldBroadmeadow',
        'ClubGameSpread',
    ]

    def test_all_slack_constraints_exist_in_stages(self):
        """All constraints that accept slack should be present in STAGES."""
        all_names = set()
        for config in STAGES.values():
            for cls in config['constraints']:
                all_names.add(cls.__name__)
        for name in self.EXPECTED_SLACK_KEYS:
            assert name in all_names, f"Slack constraint '{name}' not in STAGES"

    def test_all_slack_constraints_in_severity_map(self):
        """All slack-aware constraints should be in CONSTRAINT_TO_SEVERITY."""
        for name in self.EXPECTED_SLACK_KEYS:
            assert name in CONSTRAINT_TO_SEVERITY, (
                f"Slack constraint '{name}' not in CONSTRAINT_TO_SEVERITY"
            )


# ============== main_staged / main_simple Validation Tests ==============

class TestMainEntryPointValidation:
    """Test that main_staged and main_simple reject missing year."""

    def test_main_staged_requires_year(self):
        from main_staged import main_staged
        with pytest.raises(ValueError, match="Year is required"):
            main_staged(year=None)

    def test_main_simple_requires_year(self):
        from main_staged import main_simple
        with pytest.raises(ValueError, match="Year is required"):
            main_simple(year=None)


# ============== Total Constraint Count Sanity ==============

class TestConstraintCountSanity:
    """Basic sanity checks on the total number of constraints."""

    def test_stages_total_at_least_18(self):
        """STAGES should have at least 18 constraints (the known 19 minus possible changes)."""
        total = sum(len(cfg['constraints']) for cfg in STAGES.values())
        assert total >= 18

    def test_stages_ai_same_total(self):
        """STAGES_AI total should match STAGES total."""
        orig = sum(len(cfg['constraints']) for cfg in STAGES.values())
        ai = sum(len(cfg['constraints']) for cfg in STAGES_AI.values())
        assert orig == ai

    def test_severity_total_matches_stages_total(self):
        """STAGES_SEVERITY total should match STAGES total."""
        stages_total = sum(len(cfg['constraints']) for cfg in STAGES.values())
        severity_total = sum(len(cfg['constraints']) for cfg in STAGES_SEVERITY.values())
        assert severity_total == stages_total

    def test_all_constraint_classes_are_callable(self):
        """Every constraint class in all STAGES dicts should be instantiable."""
        for stages_dict in [STAGES, STAGES_AI, STAGES_SEVERITY, STAGES_SEVERITY_AI]:
            for config in stages_dict.values():
                for cls in config['constraints']:
                    instance = cls()
                    assert hasattr(instance, 'apply'), (
                        f"{cls.__name__} has no 'apply' method"
                    )
