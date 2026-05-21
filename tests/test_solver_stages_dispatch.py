"""Phase 7b: tests for SOLVER_STAGES dispatch + CLI flag plumbing.

Covers:
- `apply_solver_stage` dispatches into the engine's hard/soft methods.
- `_resolve_solver_stages` honours --stages-config / --stage-only / --skip-stage.
- `_validate_stages` rejects invalid `data['solver_stages']`.
- `--list-stages` calls `list_stages` and exits without solving.

No mocks: builds a tiny PHL fixture and a real CP-SAT model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------
# apply_solver_stage end-to-end against atom fixture
# ----------------------------------------------------------------------


class TestApplySolverStage:
    def _build(self):
        from constraints.unified import UnifiedConstraintEngine
        from tests.atoms.conftest import _build_phl_fixture, build_model_X
        data = _build_phl_fixture()
        model, X = build_model_X(data)
        engine = UnifiedConstraintEngine(model, X, data, skip_constraints=set())
        engine.build_groupings()
        return model, X, data, engine

    def test_engine_atoms_apply_to_model(self):
        from constraints.stages import apply_solver_stage
        model, X, data, engine = self._build()
        prior = len(model.Proto().constraints)

        stage = {
            'name': 'fence',
            'atoms': ['NoDoubleBookingTeams', 'NoDoubleBookingFields'],
        }
        added, atoms = apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )
        assert added > 0
        assert set(atoms) == {'NoDoubleBookingTeams', 'NoDoubleBookingFields'}
        assert len(model.Proto().constraints) > prior

    def test_skips_already_applied_atoms(self):
        from constraints.stages import apply_solver_stage
        model, X, data, engine = self._build()
        applied_atoms = {'NoDoubleBookingTeams'}
        applied_engine_keys = {'NoDoubleBookingTeams'}

        stage = {
            'name': 's',
            'atoms': ['NoDoubleBookingTeams', 'NoDoubleBookingFields'],
        }
        added, atoms = apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys=applied_engine_keys,
            applied_atoms=applied_atoms,
        )
        # NoDoubleBookingTeams was already applied — only Fields runs.
        assert atoms == ['NoDoubleBookingFields']
        assert added > 0

    def test_returns_zero_when_all_atoms_already_applied(self):
        from constraints.stages import apply_solver_stage
        model, X, data, engine = self._build()
        stage = {'name': 's', 'atoms': ['NoDoubleBookingTeams']}
        added, atoms = apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys={'NoDoubleBookingTeams'},
            applied_atoms={'NoDoubleBookingTeams'},
        )
        assert added == 0
        assert atoms == []


# ----------------------------------------------------------------------
# Engine-key mapping
# ----------------------------------------------------------------------


class TestAtomToEngineKey:
    def test_phl_atom_maps_to_combined_name(self):
        from constraints.stages import atom_to_engine_key
        assert atom_to_engine_key('PHLConcurrencyAtBroadmeadow') == 'PHLAndSecondGradeTimes'
        # spec-015: GosfordFridayRoundsForced deleted.

    def test_club_day_atom_maps_to_combined_name(self):
        from constraints.stages import atom_to_engine_key
        assert atom_to_engine_key('ClubDayParticipation') == 'ClubDay'

    def test_phase6_alias_maps_to_legacy_name(self):
        from constraints.stages import atom_to_engine_key
        assert atom_to_engine_key('NonDefaultHomeGrouping') == 'MaitlandHomeGrouping'
        assert atom_to_engine_key('AwayAtNonDefaultGrouping') == 'AwayAtMaitlandGrouping'
        assert atom_to_engine_key('PreferredTimes') == 'PreferredTimesConstraint'

    def test_self_canonical(self):
        from constraints.stages import atom_to_engine_key
        assert atom_to_engine_key('NoDoubleBookingTeams') == 'NoDoubleBookingTeams'

    def test_non_engine_atom_returns_none(self):
        from constraints.stages import atom_to_engine_key
        # MaximiseClubsPerTimeslotBroadmeadow isn't in the unified engine's hard/soft.
        # It's still a valid registry entry but not engine-handled.
        # (This test asserts current behavior — if the engine adds it, update.)
        assert atom_to_engine_key('MaximiseClubsPerTimeslotBroadmeadow') is None


# ----------------------------------------------------------------------
# _resolve_solver_stages CLI helper
# ----------------------------------------------------------------------


class TestResolveSolverStages:
    def test_default_stages_when_no_flags(self, tmp_path):
        from run import _resolve_solver_stages
        args = type('A', (), {
            'stages_config': None, 'stage_only': None, 'skip_stage': [],
        })()
        stages = _resolve_solver_stages(args, {})
        names = [s['name'] for s in stages]
        assert 'critical_feasibility' in names

    def test_stage_only_filters_to_one(self):
        from run import _resolve_solver_stages
        args = type('A', (), {
            'stages_config': None, 'stage_only': 'critical_feasibility', 'skip_stage': [],
        })()
        stages = _resolve_solver_stages(args, {})
        assert len(stages) == 1
        assert stages[0]['name'] == 'critical_feasibility'

    def test_skip_stage_removes(self):
        from run import _resolve_solver_stages
        args = type('A', (), {
            'stages_config': None, 'stage_only': None,
            'skip_stage': ['soft_optimisation'],
        })()
        stages = _resolve_solver_stages(args, {})
        names = [s['name'] for s in stages]
        assert 'soft_optimisation' not in names

    def test_stages_config_file(self, tmp_path):
        from run import _resolve_solver_stages
        path = tmp_path / 'custom.json'
        path.write_text(json.dumps([
            {'name': 'only', 'atoms': ['NoDoubleBookingTeams']},
        ]))
        args = type('A', (), {
            'stages_config': str(path), 'stage_only': None, 'skip_stage': [],
        })()
        stages = _resolve_solver_stages(args, {})
        assert len(stages) == 1
        assert stages[0]['name'] == 'only'

    def test_invalid_stage_name_exits(self):
        from run import _resolve_solver_stages
        args = type('A', (), {
            'stages_config': None, 'stage_only': 'no_such_stage', 'skip_stage': [],
        })()
        with pytest.raises(SystemExit):
            _resolve_solver_stages(args, {})

    def test_unknown_atom_in_stages_config_exits(self, tmp_path):
        from run import _resolve_solver_stages
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps([
            {'name': 's', 'atoms': ['NotAnAtom']},
        ]))
        args = type('A', (), {
            'stages_config': str(path), 'stage_only': None, 'skip_stage': [],
        })()
        with pytest.raises(SystemExit):
            _resolve_solver_stages(args, {})


# ----------------------------------------------------------------------
# _validate_stages config validation phase
# ----------------------------------------------------------------------


class TestValidateStagesPhase:
    def test_no_stages_no_op(self):
        from utils import _validate_stages
        warnings, fatals = [], []
        _validate_stages({}, warnings, fatals)
        assert warnings == [] and fatals == []

    def test_invalid_stages_emits_fatal(self):
        from utils import _validate_stages
        warnings, fatals = [], []
        _validate_stages(
            {'solver_stages': [{'name': 's', 'atoms': ['NotARealAtom']}]},
            warnings, fatals,
        )
        assert any('NotARealAtom' in f for f in fatals)

    def test_valid_stages_no_fatal(self):
        from utils import _validate_stages
        warnings, fatals = [], []
        _validate_stages(
            {'solver_stages': [{'name': 's', 'atoms': ['NoDoubleBookingTeams']}]},
            warnings, fatals,
        )
        assert fatals == []
