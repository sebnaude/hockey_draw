"""Phase 7b: SOLVER_STAGES config + validation tests."""
from __future__ import annotations

import pytest

from constraints.stages import (
    load_solver_stages, validate_solver_stages, list_stages,
)


class TestLoadSolverStages:
    def test_default_stages_load_when_unspecified(self):
        stages = load_solver_stages({})
        assert isinstance(stages, list)
        assert len(stages) >= 4
        names = [s['name'] for s in stages]
        assert 'critical_feasibility' in names
        assert 'soft_optimisation' in names

    def test_season_override_replaces_default(self):
        custom = [
            {'name': 'only_one', 'atoms': ['NoDoubleBookingTeams']},
        ]
        stages = load_solver_stages({'solver_stages': custom})
        assert len(stages) == 1
        assert stages[0]['name'] == 'only_one'

    def test_load_returns_independent_copies(self):
        stages = load_solver_stages({})
        stages[0]['atoms'].append('mutation_test')
        stages2 = load_solver_stages({})
        assert 'mutation_test' not in stages2[0]['atoms']


class TestValidateSolverStages:
    def test_default_stages_pass_validation(self):
        stages = load_solver_stages({})
        errors = validate_solver_stages(stages)
        assert errors == [], errors

    def test_unknown_atom_fails(self):
        stages = [{'name': 's', 'atoms': ['NotARealAtom']}]
        errors = validate_solver_stages(stages)
        assert any('NotARealAtom' in e for e in errors)

    def test_duplicate_atom_across_stages_fails(self):
        stages = [
            {'name': 'a', 'atoms': ['NoDoubleBookingTeams']},
            {'name': 'b', 'atoms': ['NoDoubleBookingTeams']},
        ]
        errors = validate_solver_stages(stages)
        assert any('appears in stages' in e for e in errors)

    def test_duplicate_stage_name_fails(self):
        stages = [
            {'name': 'x', 'atoms': ['NoDoubleBookingTeams']},
            {'name': 'x', 'atoms': ['NoDoubleBookingFields']},
        ]
        errors = validate_solver_stages(stages)
        assert any('duplicate stage name' in e for e in errors)

    def test_empty_atoms_fails(self):
        stages = [{'name': 's', 'atoms': []}]
        errors = validate_solver_stages(stages)
        assert any('non-empty list' in e for e in errors)

    def test_unknown_key_fails(self):
        stages = [{'name': 's', 'atoms': ['NoDoubleBookingTeams'], 'foo': 1}]
        errors = validate_solver_stages(stages)
        assert any('unknown keys' in e for e in errors)


class TestListStages:
    def test_list_stages_includes_names_and_atoms(self):
        stages = load_solver_stages({})
        rendered = list_stages(stages)
        assert 'critical_feasibility' in rendered
        assert 'NoDoubleBookingTeams' in rendered
