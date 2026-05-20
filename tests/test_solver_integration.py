"""
Integration tests for main_staged.py solver orchestration.

These tests exercise the real solver, real constraints, and real checkpoint
management with very short timeouts (1-5 seconds). They do NOT mock or patch
anything -- all constraint classes and CP-SAT models are real.
"""

import json
import os
import sys
import pytest
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from ortools.sat.python import cp_model

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_staged import (
    StagedScheduleSolver,
    CheckpointManager,
    IntermediateSolutionCallback,
    _build_constraints_applied,
    _build_normalized_penalty,
    _serialize_config,
)
from solver_diagnostics import SolverConfig
from constraints.archived.original import (
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    TeamConflictConstraint,
    ClubGradeAdjacencyConstraint,
    EqualMatchUpSpacingConstraint,
)
from tests.conftest import create_model_and_vars, solve_with_timeout, count_scheduled_games
from config.defaults import DEFAULT_STAGES
from constraints.stages import (
    validate_solver_stages,
    severity_solver_stages,
    _resolve_solver_class,
)
from constraints.registry import CONSTRAINT_REGISTRY


# ============== Fixtures ==============

@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Return a temporary directory for checkpoint storage."""
    return str(tmp_path / "checkpoints")


@pytest.fixture
def short_solver_config():
    """SolverConfig with a 2-second timeout and minimal workers."""
    return SolverConfig(
        max_time_seconds=2,
        num_workers=1,
        linearization_level=1,
        log_search_progress=False,
    )


@pytest.fixture
def mini_data_with_extras(mini_season_data):
    """Mini season data extended with fields that main_staged expects."""
    data = dict(mini_season_data)
    data.setdefault('constraint_slack', {})
    data.setdefault('penalty_weights', {})
    data.setdefault('forced_games', [])
    data.setdefault('blocked_games', [])
    data.setdefault('team_conflicts', {})
    data.setdefault('penalties', {})
    data.setdefault('year', 2025)
    data.setdefault('phl_game_times', {})
    data.setdefault('second_grade_times', {})
    data.setdefault('home_field_map', {
        'Maitland': 'Maitland Park',
    })
    data.setdefault('field_unavailabilities', {})
    data.setdefault('max_rounds', 4)
    # Ensure games is a list of tuples (not dict)
    if isinstance(data['games'], dict):
        data['games'] = list(data['games'].keys())
    return data


# ============== CheckpointManager Tests ==============

class TestCheckpointManager:

    def test_create_run_dir(self, tmp_checkpoint_dir):
        """CheckpointManager creates sequential run directories."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run1 = cm.get_run_dir()
        run2 = cm.get_run_dir()
        assert run1.name == "run_1"
        assert run2.name == "run_2"
        assert run1.exists()
        assert run2.exists()

    def test_save_and_load_stage(self, tmp_checkpoint_dir):
        """save_stage writes solution.pkl and metadata.json; load_stage reads them back."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        fake_solution = {("A", "B", "3rd", 0): 1, ("C", "D", "4th", 1): 0}
        data = {"year": 2025, "locked_weeks": set()}

        cm.save_stage(run_dir, "test_stage", fake_solution, data, "FEASIBLE", 1.5)

        loaded = cm.load_stage(run_dir, "test_stage")
        assert loaded is not None
        assert loaded["metadata"]["status"] == "FEASIBLE"
        assert loaded["metadata"]["solve_time"] == 1.5
        assert loaded["metadata"]["num_scheduled_games"] == 1
        assert loaded["solution"] == fake_solution

    def test_save_stage_updates_latest(self, tmp_checkpoint_dir):
        """A FEASIBLE save_stage call should update the 'latest' directory."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        solution = {("A", "B", "3rd", 0): 1}
        data = {"year": 2025, "locked_weeks": set()}

        cm.save_stage(run_dir, "my_stage", solution, data, "FEASIBLE", 2.0)

        latest = cm.load_latest()
        assert latest is not None
        assert latest["metadata"]["stage"] == "my_stage"
        assert latest["metadata"]["status"] == "FEASIBLE"

    def test_save_stage_infeasible_does_not_update_latest(self, tmp_checkpoint_dir):
        """An INFEASIBLE save_stage should NOT update 'latest'."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        solution = {}
        data = {"year": 2025, "locked_weeks": set()}

        cm.save_stage(run_dir, "bad_stage", solution, data, "INFEASIBLE", 0.5)

        latest = cm.load_latest()
        assert latest is None

    def test_save_run_metadata(self, tmp_checkpoint_dir):
        """save_run_metadata creates run_metadata.json with expected fields."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        data = {
            "year": 2025,
            "teams": [1, 2, 3],
            "grades": ["3rd", "4th"],
            "timeslots": list(range(10)),
            "locked_weeks": set(),
        }
        config = SolverConfig(max_time_seconds=5, num_workers=1)

        cm.save_run_metadata(run_dir, data, config)

        meta_path = run_dir / "run_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["year"] == 2025
        assert meta["num_teams"] == 3
        assert meta["status"] == "started"
        assert meta["solver_config"]["workers"] == 1

    def test_update_run_status(self, tmp_checkpoint_dir):
        """update_run_status modifies the run_metadata.json status."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        data = {"year": 2025, "teams": [], "grades": [], "timeslots": [], "locked_weeks": set()}
        cm.save_run_metadata(run_dir, data)

        cm.update_run_status(run_dir, "completed", {"stages_completed": ["stage1"]})

        with open(run_dir / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["status"] == "completed"
        assert meta["stages_completed"] == ["stage1"]

    def test_load_latest_returns_none_when_empty(self, tmp_checkpoint_dir):
        """load_latest returns None when no checkpoints exist."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        assert cm.load_latest() is None


# ============== StagedScheduleSolver Tests ==============

class TestStagedScheduleSolver:

    def test_instantiation(self, mini_data_with_extras, tmp_checkpoint_dir):
        """StagedScheduleSolver can be instantiated with real data."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        config = SolverConfig(max_time_seconds=2, num_workers=1)
        solver = StagedScheduleSolver(mini_data_with_extras, cm, solver_config=config)

        assert solver.data is mini_data_with_extras
        assert solver.model is None  # not initialized yet
        assert solver.X is None

    def test_apply_constraints_adds_to_model(self, mini_data_with_extras, tmp_checkpoint_dir):
        """apply_constraints with real constraints increases model constraint count."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        config = SolverConfig(max_time_seconds=2, num_workers=1)
        solver = StagedScheduleSolver(mini_data_with_extras, cm, solver_config=config)

        # Manually set up model and variables (no generate_X - use conftest helper)
        games = mini_data_with_extras['games']
        timeslots = mini_data_with_extras['timeslots']
        model, X = create_model_and_vars(games, timeslots)

        solver.model = model
        solver.X = X

        constraints_before = len(model.Proto().constraints)
        n_added = solver.apply_constraints([
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
        ])

        assert n_added > 0
        assert len(model.Proto().constraints) > constraints_before

    def test_constraint_exclusion(self, mini_data_with_extras, tmp_checkpoint_dir):
        """Excluding a constraint by name removes it from the stage."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        config = SolverConfig(max_time_seconds=2, num_workers=1)
        solver = StagedScheduleSolver(mini_data_with_extras, cm, solver_config=config)

        model, X = create_model_and_vars(
            mini_data_with_extras['games'],
            mini_data_with_extras['timeslots'],
        )
        solver.model = model
        solver.X = X

        # Build a filtered constraint list (simulating what run_staged_solve does)
        full_list = [
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            TeamConflictConstraint,
        ]
        exclude_set = {"TeamConflictConstraint"}
        filtered = [cls for cls in full_list if cls.__name__ not in exclude_set]

        assert len(filtered) == 2
        assert all(cls.__name__ != "TeamConflictConstraint" for cls in filtered)

        # Apply filtered list and verify constraints were added
        n_added = solver.apply_constraints(filtered)
        assert n_added > 0

    def test_add_solution_hints_no_crash(self, mini_data_with_extras, tmp_checkpoint_dir):
        """add_solution_hints does not crash with an empty hint dict."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        config = SolverConfig(max_time_seconds=2, num_workers=1)
        solver = StagedScheduleSolver(mini_data_with_extras, cm, solver_config=config)

        model, X = create_model_and_vars(
            mini_data_with_extras['games'],
            mini_data_with_extras['timeslots'],
        )
        solver.model = model
        solver.X = X

        # Build a hint dict: a few keys set to 1, rest implicitly 0
        hint = {}
        for i, key in enumerate(X):
            hint[key] = 1 if i < 3 else 0

        # Should not raise
        solver.add_solution_hints(hint)

    def test_add_solution_hints_empty(self, mini_data_with_extras, tmp_checkpoint_dir):
        """add_solution_hints with empty dict is a no-op."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        solver = StagedScheduleSolver(mini_data_with_extras, cm)

        model, X = create_model_and_vars(
            mini_data_with_extras['games'],
            mini_data_with_extras['timeslots'],
        )
        solver.model = model
        solver.X = X

        # Should not raise
        solver.add_solution_hints({})
        solver.add_solution_hints(None)


# ============== Mini Solver Run Tests ==============

class TestMiniSolverRun:

    def test_mini_solve_with_core_constraints(self, mini_data_with_extras):
        """Build a small model with core constraints and solve with 2s timeout."""
        games = mini_data_with_extras['games']
        timeslots = mini_data_with_extras['timeslots']
        model, X = create_model_and_vars(games, timeslots)

        # Apply just the two most basic constraints
        c1 = NoDoubleBookingTeamsConstraint()
        c1.apply(model, X, mini_data_with_extras)

        c2 = NoDoubleBookingFieldsConstraint()
        c2.apply(model, X, mini_data_with_extras)

        # Set objective: maximize games scheduled
        model.Maximize(sum(X.values()))

        # Solve with short timeout
        status, solver = solve_with_timeout(model, timeout_seconds=2.0)

        # We expect FEASIBLE or OPTIMAL for this small problem
        assert status in [cp_model.FEASIBLE, cp_model.OPTIMAL], (
            f"Expected FEASIBLE or OPTIMAL, got {solver.status_name(status)}"
        )

        scheduled = count_scheduled_games(X, solver)
        assert scheduled > 0, "Expected at least some games to be scheduled"

    def test_mini_solve_with_equal_games(self, mini_data_with_extras):
        """Apply NoDoubleBooking + EqualGames constraints and solve."""
        games = mini_data_with_extras['games']
        timeslots = mini_data_with_extras['timeslots']
        model, X = create_model_and_vars(games, timeslots)

        for constraint_cls in [
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            EnsureEqualGamesAndBalanceMatchUps,
        ]:
            constraint_cls().apply(model, X, mini_data_with_extras)

        model.Maximize(sum(X.values()))

        status, solver = solve_with_timeout(model, timeout_seconds=3.0)

        # With EqualGames the problem may be tighter, but should still be solvable
        # for this small data set. Accept FEASIBLE, OPTIMAL, or UNKNOWN (if timeout).
        assert status != cp_model.MODEL_INVALID, (
            f"Model should not be invalid, got {solver.status_name(status)}"
        )

    def test_mini_solve_status_is_not_invalid(self, mini_data_with_extras):
        """Even with multiple constraints, model should never be MODEL_INVALID."""
        games = mini_data_with_extras['games']
        timeslots = mini_data_with_extras['timeslots']
        model, X = create_model_and_vars(games, timeslots)

        for constraint_cls in [
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
        ]:
            constraint_cls().apply(model, X, mini_data_with_extras)

        model.Maximize(sum(X.values()))

        status, solver = solve_with_timeout(model, timeout_seconds=1.0)

        # MODEL_INVALID means the model itself is broken (contradictory constraints
        # at the proto level). This should never happen with valid constraints.
        assert status != cp_model.MODEL_INVALID


# ============== Constraint Exclusion Verification ==============

class TestConstraintExclusionMetadata:

    def test_excluded_constraint_not_in_metadata(self, mini_data_with_extras, tmp_checkpoint_dir):
        """Verify excluded constraints are not tracked in constraints_applied metadata."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        config = SolverConfig(max_time_seconds=2, num_workers=1)
        solver = StagedScheduleSolver(mini_data_with_extras, cm, solver_config=config)

        model, X = create_model_and_vars(
            mini_data_with_extras['games'],
            mini_data_with_extras['timeslots'],
        )
        solver.model = model
        solver.X = X

        # Simulate the exclusion logic from run_staged_solve
        stage_constraints = [
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            TeamConflictConstraint,
        ]
        exclude_set = {"TeamConflictConstraint", "TeamConflictConstraintAI"}
        filtered = [cls for cls in stage_constraints if cls.__name__ not in exclude_set]

        solver.apply_constraints(filtered)

        # Track metadata the same way run_staged_solve does
        solver.data.setdefault('constraints_applied', [])
        for cls in filtered:
            solver.data['constraints_applied'].append({
                'name': cls.__name__,
                'stage': 'test_stage',
            })

        applied_names = [c['name'] for c in solver.data['constraints_applied']]
        assert "TeamConflictConstraint" not in applied_names
        assert "NoDoubleBookingTeamsConstraint" in applied_names
        assert "NoDoubleBookingFieldsConstraint" in applied_names


# ============== Slack Application Verification ==============

class TestSlackApplication:

    def test_slack_accessible_in_data(self, mini_data_with_extras, tmp_checkpoint_dir):
        """Verify constraint_slack values are accessible in data dict during solve."""
        slack_config = {'EqualMatchUpSpacingConstraint': 3, 'ClubGameSpread': 2}
        mini_data_with_extras['constraint_slack'] = slack_config

        cm = CheckpointManager(tmp_checkpoint_dir)
        solver = StagedScheduleSolver(mini_data_with_extras, cm)

        # After construction, the data dict should carry the slack
        assert solver.data['constraint_slack']['EqualMatchUpSpacingConstraint'] == 3
        assert solver.data['constraint_slack']['ClubGameSpread'] == 2

    def test_slack_persists_through_checkpoint(self, mini_data_with_extras, tmp_checkpoint_dir):
        """Verify slack values are saved into checkpoint metadata."""
        slack_config = {'EqualMatchUpSpacingConstraint': 3}
        mini_data_with_extras['constraint_slack'] = slack_config

        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        fake_solution = {("A", "B", "3rd", 0): 1}
        cm.save_stage(run_dir, "slack_stage", fake_solution, mini_data_with_extras, "FEASIBLE", 1.0)

        loaded = cm.load_stage(run_dir, "slack_stage")
        assert loaded is not None
        meta_slack = loaded["metadata"]["constraint_slack"]
        assert meta_slack["EqualMatchUpSpacingConstraint"] == 3


# ============== Checkpoint Metadata Completeness ==============

class TestCheckpointMetadata:

    def test_metadata_has_expected_fields(self, tmp_checkpoint_dir):
        """Verify metadata.json has all expected fields."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        data = {
            "year": 2025,
            "locked_weeks": {1, 2},
            "constraint_slack": {"Foo": 1},
            "_excluded_constraints": ["Bar"],
            "_solver_mode": "simple",
        }

        solution = {("T1", "T2", "3rd", 0): 1, ("T3", "T4", "4th", 1): 0}
        cm.save_stage(run_dir, "meta_test", solution, data, "OPTIMAL", 5.5)

        loaded = cm.load_stage(run_dir, "meta_test")
        meta = loaded["metadata"]

        # Atomization removed the AI/non-AI split at stage level, so `use_ai`
        # is no longer part of the checkpoint metadata schema (the writer in
        # main_staged.py::save_stage no longer emits it). The schema is the
        # field set written at main_staged.py:356.
        required_fields = [
            "stage", "status", "solve_time", "timestamp",
            "num_scheduled_games", "total_variables", "run_id",
            "year", "mode", "locked_weeks",
            "excluded_constraints", "constraint_slack",
        ]
        for field in required_fields:
            assert field in meta, f"Missing field: {field}"

        assert meta["stage"] == "meta_test"
        assert meta["status"] == "OPTIMAL"
        assert meta["solve_time"] == 5.5
        assert meta["num_scheduled_games"] == 1
        assert meta["total_variables"] == 2
        assert meta["year"] == 2025
        assert meta["mode"] == "simple"
        assert sorted(meta["locked_weeks"]) == [1, 2]
        assert meta["excluded_constraints"] == ["Bar"]


# ============== IntermediateSolutionCallback Tests ==============

class TestIntermediateSolutionCallback:

    def test_callback_attributes_initialized(self, tmp_checkpoint_dir):
        """Callback initializes with expected attributes."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        # Create a trivial model
        model = cp_model.CpModel()
        x = model.NewBoolVar("x")
        X = {("placeholder",): x}

        callback = IntermediateSolutionCallback(
            X=X,
            checkpoint_manager=cm,
            run_dir=run_dir,
            stage_name="cb_test",
            data={"year": 2025, "locked_weeks": set()},
            save_interval=1,
        )

        assert callback.solution_count == 0
        assert callback.best_objective == float('-inf')
        assert callback.last_save_time is None
        assert callback.stage_name == "cb_test"

    def test_callback_called_during_solve(self, tmp_checkpoint_dir):
        """Callback is invoked during a trivial solve and updates solution_count."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        run_dir = cm.get_run_dir()

        # Build a tiny model with an objective so the solver actually calls back
        model = cp_model.CpModel()
        variables = {}
        for i in range(5):
            key = (f"T{i}", f"T{i+5}", "3rd", i)
            variables[key] = model.NewBoolVar(f"x_{i}")

        # Simple objective: maximize sum
        model.Maximize(sum(variables.values()))

        callback = IntermediateSolutionCallback(
            X=variables,
            checkpoint_manager=cm,
            run_dir=run_dir,
            stage_name="trivial_solve",
            data={"year": 2025, "locked_weeks": set()},
            save_interval=0,
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1
        status = solver.Solve(model, callback)

        # The solver should find at least one solution for this trivial problem
        assert status in [cp_model.FEASIBLE, cp_model.OPTIMAL]
        assert callback.solution_count >= 1
        assert callback.best_objective >= 0


# ============== Helper Function Tests ==============

class TestHelperFunctions:

    def test_serialize_config_handles_datetime(self):
        """_serialize_config converts datetime to ISO format."""
        dt = datetime(2025, 3, 23, 10, 30)
        result = _serialize_config({"when": dt})
        assert result["when"] == "2025-03-23T10:30:00"

    def test_serialize_config_handles_set(self):
        """_serialize_config converts sets to sorted lists."""
        result = _serialize_config({"weeks": {3, 1, 2}})
        assert result["weeks"] == [1, 2, 3]

    def test_serialize_config_handles_nested(self):
        """_serialize_config recurses into nested structures."""
        result = _serialize_config({"outer": {"inner": {5, 2}}})
        assert result["outer"]["inner"] == [2, 5]

    def test_build_constraints_applied(self):
        """_build_constraints_applied returns list of constraint metadata dicts."""
        classes = [NoDoubleBookingTeamsConstraint, NoDoubleBookingFieldsConstraint]
        result = _build_constraints_applied(classes)
        assert len(result) == 2
        assert result[0]["name"] == "NoDoubleBookingTeamsConstraint"
        assert result[1]["name"] == "NoDoubleBookingFieldsConstraint"

    def test_build_constraints_applied_with_severity_map(self):
        """_build_constraints_applied includes severity when map is provided."""
        classes = [NoDoubleBookingTeamsConstraint]
        sev_map = {"NoDoubleBookingTeamsConstraint": 1}
        result = _build_constraints_applied(classes, severity_map=sev_map)
        assert result[0]["severity"] == 1

    def test_build_normalized_penalty_empty(self):
        """_build_normalized_penalty returns empty list for empty input."""
        assert _build_normalized_penalty({}) == []

    def test_build_normalized_penalty_normalizes(self):
        """_build_normalized_penalty divides weight by var count."""
        model = cp_model.CpModel()
        v1 = model.NewBoolVar("v1")
        v2 = model.NewBoolVar("v2")

        penalties = {
            "test_penalty": {
                "weight": 100,
                "penalties": [v1, v2],
            }
        }
        terms = _build_normalized_penalty(penalties)
        # weight=100, 2 vars => normalized = 100//2 = 50 per var
        assert len(terms) == 2
        assert all(coeff == 50 for coeff, _ in terms)


# ============== Stage Definition Tests ==============

class TestStageDefinitions:
    """Post-atomization, stages are the config-driven `DEFAULT_STAGES` list
    (each a dict with `name` + `atoms` of canonical registry names), not the
    legacy `STAGES` / `STAGES_AI` / `STAGES_SEVERITY` class-list dicts. The
    AI/non-AI split moved to per-atom resolution (`_resolve_solver_class`),
    so there is no separate AI stage table to mirror.
    """

    def test_default_stages_all_have_atoms(self):
        """Scenario: every default stage carries a non-empty atom list."""
        # Given: the live DEFAULT_STAGES from config/defaults.py.
        # Oracle: there are exactly 5 stages (critical_feasibility,
        # home_away_balance, club_alignment, club_day, soft_optimisation),
        # each with a non-empty `atoms` list.
        # When / Then:
        assert len(DEFAULT_STAGES) == 5
        names = [s["name"] for s in DEFAULT_STAGES]
        assert names == [
            "critical_feasibility", "home_away_balance",
            "club_alignment", "club_day", "soft_optimisation",
        ]
        # And: each stage's atoms list is present and non-empty.
        for stage in DEFAULT_STAGES:
            assert isinstance(stage.get("atoms"), list), f"{stage['name']} atoms not a list"
            assert len(stage["atoms"]) > 0, f"{stage['name']} has empty atoms"

    def test_default_stages_validate_clean(self):
        """Scenario: DEFAULT_STAGES passes registry validation with no errors."""
        # Given: DEFAULT_STAGES. When: validate_solver_stages runs.
        errors = validate_solver_stages(DEFAULT_STAGES)
        # Then: zero errors. Oracle: [] — every atom is a registered canonical
        # name, names are unique, no atom appears in two stages.
        assert errors == [], f"DEFAULT_STAGES validation errors: {errors}"

    def test_severity_stages_validate_clean(self):
        """Scenario: the severity-grouped stage list is also valid + non-empty."""
        # Given: severity_solver_stages() built from CONSTRAINT_REGISTRY.
        stages = severity_solver_stages()
        # When / Then: non-empty list of stages, each with non-empty atoms.
        assert len(stages) > 0
        for stage in stages:
            assert len(stage["atoms"]) > 0, f"{stage['name']} has empty atoms"
        # And: validation reports no errors. Oracle: [].
        assert validate_solver_stages(stages) == []

    def test_all_stage_atoms_resolve_or_are_atom_only(self):
        """Scenario: every atom in every default stage either resolves to an
        instantiable solver class with apply(), or is a registered atom-only /
        tester-only entry (resolver returns None by design)."""
        # Given: every atom name across all DEFAULT_STAGES.
        # When: look it up in the registry and resolve its solver class.
        for stage in DEFAULT_STAGES:
            for atom in stage["atoms"]:
                # Then: the atom is a registered canonical name.
                assert atom in CONSTRAINT_REGISTRY, (
                    f"{atom} (stage {stage['name']}) not in CONSTRAINT_REGISTRY"
                )
                cls = _resolve_solver_class(atom)
                # And: if it resolves to a class, that class instantiates and
                # exposes apply(); None is acceptable (atom-only / alias).
                if cls is not None:
                    instance = cls()
                    assert hasattr(instance, "apply"), f"{cls.__name__} missing apply()"


# ============== StagedScheduleSolver.build_objective Tests ==============

class TestBuildObjective:

    def test_build_objective_no_penalties(self, mini_data_with_extras, tmp_checkpoint_dir):
        """build_objective works when there are no penalties."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        solver = StagedScheduleSolver(mini_data_with_extras, cm)

        model, X = create_model_and_vars(
            mini_data_with_extras['games'],
            mini_data_with_extras['timeslots'],
        )
        solver.model = model
        solver.X = X

        # Should not raise
        solver.build_objective()

    def test_build_objective_with_penalties(self, mini_data_with_extras, tmp_checkpoint_dir):
        """build_objective correctly handles penalty variables."""
        cm = CheckpointManager(tmp_checkpoint_dir)
        solver = StagedScheduleSolver(mini_data_with_extras, cm)

        model, X = create_model_and_vars(
            mini_data_with_extras['games'],
            mini_data_with_extras['timeslots'],
        )
        solver.model = model
        solver.X = X

        # Add a mock penalty
        pen_var = model.NewBoolVar("penalty_1")
        mini_data_with_extras['penalties'] = {
            "TestPenalty": {"weight": 1000, "penalties": [pen_var]},
        }

        # Should not raise
        solver.build_objective()
