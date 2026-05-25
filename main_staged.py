# main_staged.py
"""
Staged solving main entry point for the scheduling system.

This version solves the scheduling problem in stages:
1. Stage 1: Required constraints (no double-booking, equal games)
2. Stage 2: Strong structural constraints (adjacency, time separation)
3. Stage 3: Medium venue/scheduling constraints
4. Stage 4: Soft preference constraints

Each stage:
- Loads the previous solution as hints
- Adds new constraints
- Solves with time limits
- Saves checkpoint for resumption

Benefits:
- Allows resumption from any stage
- Provides intermediate solutions
- Isolates infeasibility to specific stages
- More predictable solving times
"""

import os
import sys
import pickle
import json
import logging
import traceback
from datetime import datetime
from typing import Optional
from collections import defaultdict
from pathlib import Path

from ortools.sat.python import cp_model
from utils import generate_X
from solver_diagnostics import (
    setup_logging, ResourceMonitor, SolverConfig, get_recommended_config,
    log_system_info, log_model_info, log_solve_result, PSUTIL_AVAILABLE
)
from constraints.stages import (
    apply_solver_stage,
    load_solver_stages,
    list_stages,
    severity_solver_stages,
)


# ============== Metadata Helpers ==============

def _serialize_config(obj):
    """Make config objects JSON-serializable (datetime, time, etc.)."""
    if isinstance(obj, dict):
        return {k: _serialize_config(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_config(v) for v in obj]
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, 'strftime'):
        return obj.strftime('%H:%M')
    elif isinstance(obj, set):
        return sorted(obj)
    return obj


def _build_constraints_applied(constraint_classes, severity_map=None):
    """Build a serializable list of constraint metadata from constraint classes."""
    result = []
    for cls in constraint_classes:
        entry = {'name': cls.__name__}
        if severity_map and cls.__name__ in severity_map:
            entry['severity'] = severity_map[cls.__name__]
        result.append(entry)
    return result


def _build_normalized_penalty(penalties_dict: dict) -> list:
    """
    Build a normalized penalty expression from the penalties dictionary.

    Each soft constraint registers penalties as:
        penalties_dict[name] = {'weight': W, 'penalties': [var1, var2, ...]}

    Raw objective would be: sum(W_i * sum(vars_i)) — but a constraint with
    4000 vars at weight 50K dominates one with 24 vars at weight 1M.

    Normalization: divide each group's configured weight by its var count,
    so the weight becomes a per-violation cost. The configured weights then
    act as relative priorities independent of how many vars each constraint
    creates.

    Returns a list of (coefficient, var) pairs for the objective.
    """
    terms = []
    for name, info in penalties_dict.items():
        pen_vars = info.get('penalties', [])
        if not pen_vars:
            continue
        raw_weight = info['weight']
        n = len(pen_vars)
        # Normalized weight = configured weight / var count
        # This means: if you set weight=100_000, each violation costs 100_000/N
        # and a full-violation scenario (all N vars = max) costs ~100_000 total.
        normalized = max(1, raw_weight // n)
        for var in pen_vars:
            terms.append((normalized, var))
    return terms


def _apply_objective_lower_bound(model, objective_expr, data):
    """
    Apply objective lower bound from config to prune bad search space.

    If 'objective_lower_bound' is set in data (from SEASON_CONFIG), adds a
    constraint that the objective must be >= that value. This tells CP-SAT
    to prune any branch that can't reach the bound, speeding up the search.

    Args:
        model: CpModel instance
        objective_expr: The objective expression (before Maximize)
        data: Data dict containing config

    Returns:
        The objective expression (possibly wrapped in an IntVar with lower bound)
    """
    lower_bound = data.get('objective_lower_bound')
    if lower_bound is not None:
        model.Add(objective_expr >= lower_bound)
        print(f"  Objective lower bound: {lower_bound:,} (pruning worse solutions)")
    return objective_expr


# ============== Solution Callback for Intermediate Saves ==============

class IntermediateSolutionCallback(cp_model.CpSolverSolutionCallback):
    """Callback to save intermediate solutions during solving."""
    
    def __init__(self, X: dict, checkpoint_manager, run_dir: Path, stage_name: str, 
                 data: dict, save_interval: int = 60):
        """
        Args:
            X: Decision variables dictionary
            checkpoint_manager: CheckpointManager instance
            run_dir: Directory for checkpoints
            stage_name: Current stage name
            data: Data dictionary
            save_interval: Minimum seconds between saves (default 60)
        """
        super().__init__()
        self.X = X
        self.checkpoint_manager = checkpoint_manager
        self.run_dir = run_dir
        self.stage_name = stage_name
        self.data = data
        self.save_interval = save_interval
        
        self.solution_count = 0
        self.best_objective = float('-inf')
        self.last_save_time = None
        self.start_time = datetime.now()
    
    def on_solution_callback(self):
        """Called each time a new solution is found."""
        self.solution_count += 1
        current_objective = self.ObjectiveValue()
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        print(f"  [Callback] Solution #{self.solution_count} at {elapsed:.1f}s, objective: {current_objective:.0f}")
        
        # Save every improving solution (callback is only called on new solutions)
        if current_objective > self.best_objective or self.solution_count == 1:
            self.best_objective = current_objective
            self.last_save_time = current_time
            
            # Extract solution
            solution = {key: self.Value(var) for key, var in self.X.items()}
            games_scheduled = sum(1 for v in solution.values() if v == 1)
            
            print(f"  [Callback] Saving intermediate solution (games: {games_scheduled})")
            
            # Save to checkpoint with intermediate marker
            self.checkpoint_manager.save_stage(
                self.run_dir, 
                f"{self.stage_name}_intermediate_{self.solution_count}",
                solution, 
                self.data, 
                'FEASIBLE', 
                elapsed
            )


# ============== Stage Definitions ==============
#
# Phase 7c: legacy STAGES / STAGES_AI / STAGES_UNIFIED / STAGES_SEVERITY[_AI]
# dicts have been removed. Stage configuration now flows through
# `config/defaults.py::DEFAULT_STAGES` (or per-season overrides) and the
# `constraints/stages.py::apply_solver_stage` dispatcher. Severity-staged
# solving uses `severity_solver_stages()` to build the per-level stage list
# from the registry.


# ============== Checkpoint Management ==============

class CheckpointManager:
    """Manages saving and loading of solver state.
    
    Maintains a 'latest' directory that always points to the most recent
    successful checkpoint, making it easy to find the current solver state.
    
    Structure:
        checkpoints/
        ├── latest/                # Copy of most recent successful stage
        │   ├── solution.pkl
        │   ├── metadata.json
        │   └── penalties.json
        ├── run_1/
        │   ├── stage1_required/
        │   └── stage2_soft/
        └── run_N/
    """
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_run_dir(self, run_id: str = None) -> Path:
        """Get or create a run directory."""
        if run_id is None:
            # Create new run - find highest existing run number and increment
            existing_nums = []
            for d in self.checkpoint_dir.iterdir():
                if d.is_dir() and d.name.startswith('run_'):
                    try:
                        existing_nums.append(int(d.name.split('_', 1)[1]))
                    except ValueError:
                        pass
            run_num = max(existing_nums, default=0) + 1
            run_id = f'run_{run_num}'
        
        run_dir = self.checkpoint_dir / run_id
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def _update_latest(self, stage_dir: Path):
        """Update the 'latest' directory to mirror the most recent checkpoint."""
        import shutil
        latest_dir = self.checkpoint_dir / 'latest'
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(stage_dir, latest_dir)
        
        # Also save a pointer metadata so we know which run/stage this came from
        pointer = {
            'source_run': stage_dir.parent.name,
            'source_stage': stage_dir.name,
            'source_path': str(stage_dir),
            'updated_at': datetime.now().isoformat(),
        }
        with open(latest_dir / 'pointer.json', 'w') as f:
            json.dump(pointer, f, indent=2)
    
    def save_run_metadata(self, run_dir: Path, data: dict, solver_config: SolverConfig = None):
        """Save run-level metadata at run start, before any solving.

        This captures the intent and inputs so that every run directory
        is self-documenting even if the solve never completes.
        """
        import platform

        meta = {
            'run_id': run_dir.name,
            'started_at': datetime.now().isoformat(),
            'status': 'started',

            # User intent
            'description': data.get('_user_description', ''),
            'year': data.get('year'),
            'mode': data.get('_solver_mode', 'unknown'),

            # Inputs
            'locked_weeks': sorted(data.get('locked_weeks', set())),
            'locked_source': data.get('_provenance', {}).get('locked_source', ''),
            'locked_game_count': data.get('_provenance', {}).get('locked_game_count', 0),
            'hint_source': data.get('_provenance', {}).get('hint_source', ''),
            'excluded_constraints': data.get('_excluded_constraints', []),
            'constraint_slack': _serialize_config(data.get('constraint_slack', {})),
            'penalty_weights': data.get('penalty_weights', {}),
            'forced_games': _serialize_config(data.get('forced_games', [])),
            'blocked_games': _serialize_config(data.get('blocked_games', [])),
            'field_unavailabilities': _serialize_config(data.get('field_unavailabilities', {})),

            # Data shape
            'num_teams': len(data.get('teams', [])),
            'num_grades': len(data.get('grades', [])),
            'num_timeslots': len(data.get('timeslots', [])),

            # Solver config
            'solver_config': {
                'workers': solver_config.num_workers if solver_config else None,
                'linearization_level': solver_config.linearization_level if solver_config else None,
                'max_memory_mb': getattr(solver_config, 'max_memory_mb', None) if solver_config else None,
            },

            # Environment
            'environment': {
                'python_version': sys.version.split()[0],
                'platform': platform.platform(),
                'cpu_count': os.cpu_count(),
            },
        }

        try:
            import psutil
            mem = psutil.virtual_memory()
            meta['environment']['total_memory_gb'] = round(mem.total / (1024**3), 1)
            meta['environment']['available_memory_gb'] = round(mem.available / (1024**3), 1)
        except ImportError:
            pass

        try:
            from ortools.sat.python import cp_model
            meta['environment']['ortools_version'] = getattr(cp_model, '__version__', 'unknown')
        except ImportError:
            pass

        with open(run_dir / 'run_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    def update_run_status(self, run_dir: Path, status: str, extra: dict = None):
        """Update the run-level metadata with final status and optional extra fields."""
        meta_path = run_dir / 'run_metadata.json'
        if not meta_path.exists():
            return
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['status'] = status
        meta['finished_at'] = datetime.now().isoformat()
        if extra:
            meta.update(extra)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def save_stage(self, run_dir: Path, stage_name: str, X_solution: dict, data: dict,
                   status: str, solve_time: float):
        """Save stage results to checkpoint and update latest."""
        stage_dir = run_dir / stage_name
        stage_dir.mkdir(exist_ok=True)
        
        # Save solution
        with open(stage_dir / 'solution.pkl', 'wb') as f:
            pickle.dump(X_solution, f)
        
        # Save metadata
        num_games = sum(1 for v in X_solution.values() if v == 1)
        metadata = {
            'stage': stage_name,
            'status': status,
            'solve_time': solve_time,
            'timestamp': datetime.now().isoformat(),
            'num_scheduled_games': num_games,
            'total_variables': len(X_solution),
            'run_id': run_dir.name,
            'year': data.get('year'),
            'description': data.get('_user_description', ''),
            'mode': data.get('_solver_mode', 'unknown'),
            'locked_weeks': sorted(data.get('locked_weeks', set())),
            'excluded_constraints': data.get('_excluded_constraints', []),
            'constraint_slack': _serialize_config(data.get('constraint_slack', {})),
            'penalty_weights': data.get('penalty_weights', {}),
            'forced_games': data.get('forced_games', []),
            'blocked_games': data.get('blocked_games', []),
            'field_unavailabilities': _serialize_config(data.get('field_unavailabilities', {})),
        }

        # Include constraints applied if tracked in data
        if 'constraints_applied' in data:
            metadata['constraints_applied'] = data['constraints_applied']

        with open(stage_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save penalties summary if available
        if 'penalties' in data:
            penalties_summary = {
                name: len(info.get('penalties', []))
                for name, info in data['penalties'].items()
            }
            with open(stage_dir / 'penalties.json', 'w') as f:
                json.dump(penalties_summary, f, indent=2)
        
        print(f"  Checkpoint saved to {stage_dir}")
        
        # Update latest pointer if this was a successful solve
        if status in ['OPTIMAL', 'FEASIBLE']:
            self._update_latest(stage_dir)
            print(f"  Updated checkpoints/latest -> {run_dir.name}/{stage_name}")
    
    def load_latest(self) -> Optional[dict]:
        """Load the latest successful checkpoint."""
        latest_dir = self.checkpoint_dir / 'latest'
        if not latest_dir.exists() or not (latest_dir / 'solution.pkl').exists():
            return None
        
        with open(latest_dir / 'solution.pkl', 'rb') as f:
            solution = pickle.load(f)
        
        with open(latest_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        result = {'solution': solution, 'metadata': metadata}
        
        pointer_path = latest_dir / 'pointer.json'
        if pointer_path.exists():
            with open(pointer_path, 'r') as f:
                result['pointer'] = json.load(f)
        
        return result
    
    def load_stage(self, run_dir: Path, stage_name: str) -> dict:
        """Load stage results from checkpoint."""
        stage_dir = run_dir / stage_name
        
        if not (stage_dir / 'solution.pkl').exists():
            return None
        
        with open(stage_dir / 'solution.pkl', 'rb') as f:
            solution = pickle.load(f)
        
        with open(stage_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return {
            'solution': solution,
            'metadata': metadata,
        }
    
    def get_last_completed_stage(self, run_dir: Path, stage_names: list = None) -> str:
        """Find the last successfully completed stage.

        Phase 7c: takes the candidate stage names as an argument (callers
        usually pass `[s['name'] for s in data['solver_stages']]`).
        """
        if stage_names is None:
            return None
        for stage_name in reversed(list(stage_names)):
            stage_dir = run_dir / stage_name
            if (stage_dir / 'solution.pkl').exists():
                with open(stage_dir / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                if metadata.get('status') in ['OPTIMAL', 'FEASIBLE']:
                    return stage_name
        return None


# ============== Staged Solver ==============

class StagedScheduleSolver:
    """Solves the scheduling problem in stages."""
    
    def __init__(self, data: dict, checkpoint_manager: CheckpointManager = None,
                 solver_config: SolverConfig = None, logger: logging.Logger = None,
                 relax_config: dict = None):
        self.data = data
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.model = None
        self.X = None
        self.current_solution = None
        
        # Diagnostics
        self.logger = logger or logging.getLogger("solver")
        self.solver_config = solver_config or get_recommended_config()
        self.resource_monitor = ResourceMonitor(logger=self.logger) if PSUTIL_AVAILABLE else None
        
        # Relaxation config
        self.relax_config = relax_config or {}
        self.relaxed_groups = {}  # Dict[int, int] mapping severity level -> slack level
        
        self.logger.info(f"StagedScheduleSolver initialized")
        self.logger.info(f"Solver config: workers={self.solver_config.num_workers}, "
                        f"linearization={self.solver_config.linearization_level}")
        if self.relax_config.get('enabled'):
            self.logger.info(f"Relaxation enabled: timeout={self.relax_config.get('timeout', 30)}s")
    
    def initialize_model(self, unavailability_path: str):
        """Initialize the CP model with decision variables."""
        self.logger.info("Initializing model and decision variables...")
        print("Initializing model and decision variables...")
        
        self.model = cp_model.CpModel()
        
        # Generate decision variables
        self.X, conflicts = generate_X(
            self.model, self.data
        )

        self.data['team_conflicts'] = conflicts
        self.data['games'] = list(self.data['games'].keys()) if isinstance(self.data['games'], dict) else self.data['games']

        print(f"  Created {len(self.X)} decision variables")
        
        return self.X
    
    def add_solution_hints(self, solution: dict):
        """Add previous solution as hints to guide the solver."""
        if not solution:
            return
        
        hints_added = 0
        for key, value in solution.items():
            if key in self.X and value == 1:
                self.model.AddHint(self.X[key], 1)
                hints_added += 1
        
        print(f"  Added {hints_added} solution hints from previous stage")
    
    def apply_constraints(self, constraint_classes: list) -> int:
        """Apply a list of constraints to the model, respecting relaxation settings."""
        total_constraints = 0
        prior_count = len(self.model.Proto().constraints)
        
        for constraint_class in constraint_classes:
            # Check if this constraint's severity group is relaxed
            if self.relaxed_groups and self.relax_config.get('enabled'):
                from constraints.severity import get_severity_level
                level = get_severity_level(constraint_class)
                
                if level in self.relaxed_groups:
                    # Use soft version with specified slack
                    slack = self.relaxed_groups[level]
                    from constraints.soft import get_soft_constraint
                    soft_instance = get_soft_constraint(constraint_class.__name__, slack)
                    
                    if soft_instance:
                        soft_instance.apply(self.model, self.X, self.data)
                        current_count = len(self.model.Proto().constraints)
                        added = current_count - prior_count
                        print(f"    {constraint_class.__name__} (SOFT slack={slack}): {added} constraints")
                        total_constraints += added
                        prior_count = current_count
                        continue
            
            # Use hard constraint
            constraint = constraint_class()
            constraint.apply(self.model, self.X, self.data)
            
            current_count = len(self.model.Proto().constraints)
            added = current_count - prior_count
            print(f"    {constraint_class.__name__}: {added} constraints")
            total_constraints += added
            prior_count = current_count
        
        return total_constraints
    
    def find_and_relax_problem_group(self, constraint_classes: list) -> Optional[int]:
        """
        Find which severity group causes infeasibility and relax it.
        
        This performs quick feasibility tests by dropping severity groups
        (4, then 3, then 2) to identify the problem group. Once found,
        sets self.relaxed_groups to use soft constraints for that group.
        
        Returns:
            Problem severity level, or None if all feasible
        """
        from constraints.severity import (
            SeverityGroupResolver, 
            create_relaxation_test_func,
            group_constraints_by_severity
        )
        
        timeout = self.relax_config.get('timeout', 30.0)
        
        print("\n" + "="*60)
        print("SEVERITY-BASED RELAXATION")
        print("="*60)
        self.logger.info("Starting severity-based relaxation analysis...")
        
        # Create resolver
        resolver = SeverityGroupResolver(constraint_classes, verbose=True)
        
        # Create test function
        test_func = create_relaxation_test_func(
            self.data, 
            generate_X,
            timeout=timeout
        )
        
        # Find problem group
        problem_level = resolver.find_problem_severity_group(test_func, timeout)
        
        if problem_level is None:
            print("\n[OK] All constraints are feasible!")
            self.logger.info("All constraints feasible, no relaxation needed")
            return None
        
        if problem_level == 1:
            print("\n[FATAL] Level 1 constraints are infeasible - cannot relax")
            self.logger.error("Level 1 constraints infeasible - check data/config")
            return 1
        
        # Relax the problem group
        print(f"\n[RELAX] Setting slack=1 for severity level {problem_level}")
        self.relaxed_groups[problem_level] = 1
        self.logger.info(f"Relaxed severity group {problem_level} with slack=1")
        
        print(resolver.get_state_summary())
        
        return problem_level
    
    def build_objective(self):
        """Build the optimization objective with normalized penalty weights."""
        penalties_dict = self.data.get('penalties', {})

        # Log penalty summary
        for name, info in penalties_dict.items():
            n = len(info.get('penalties', []))
            w = info['weight']
            nw = max(1, w // n) if n > 0 else 0
            print(f"  Penalty: {name}: {n} vars, weight {w:,} -> normalized {nw:,}/var")

        # Build normalized penalty terms
        penalty_terms = _build_normalized_penalty(penalties_dict)
        total_penalty = sum(coeff * var for coeff, var in penalty_terms)

        # Objective: maximize games scheduled, minimize penalties
        objective_expr = sum(self.X.values()) - total_penalty
        _apply_objective_lower_bound(self.model, objective_expr, self.data)
        self.model.Maximize(objective_expr)

    def solve_stage(self, stage_config: dict, run_dir: Path = None, stage_name: str = None) -> tuple:
        """Solve a single stage with comprehensive logging and resource monitoring."""
        # Use stage-specific time if set, otherwise fall back to config, then default 2 days
        max_time = stage_config.get('max_time_seconds',
                                    self.data.get('max_time_per_stage', 172800))
        self.logger.info(f"Starting solve for stage: {stage_name}")
        self.logger.info(f"  Max time: {max_time}s ({max_time/3600:.1f}h)")

        # Log model info before solve
        log_model_info(self.model, self.X, self.logger)

        # Log resource state before solve
        if self.resource_monitor:
            self.logger.info("Pre-solve resource snapshot:")
            self.resource_monitor.log_snapshot(prefix="PRE-SOLVE")

        solver = cp_model.CpSolver()

        # Apply solver configuration (includes num_workers, memory settings)
        stage_specific_config = SolverConfig(
            max_time_seconds=max_time,
            num_workers=self.solver_config.num_workers,
            linearization_level=self.solver_config.linearization_level,
            cp_model_probing_level=self.solver_config.cp_model_probing_level,
            log_search_progress=True
        )
        stage_specific_config.apply_to_solver(solver)
        
        self.logger.info(f"Solver configured: workers={stage_specific_config.num_workers}, "
                        f"linearization={stage_specific_config.linearization_level}")
        
        start_time = datetime.now()
        
        # Start resource monitoring during solve
        if self.resource_monitor:
            self.resource_monitor.start_monitoring(interval=30.0)
        
        try:
            # Use callback for stages with soft constraints to save intermediate solutions
            if stage_config.get('use_callback', False) and run_dir and stage_name:
                self.logger.info("Using solution callback for intermediate saves...")
                print("  Using solution callback for intermediate saves...")
                callback = IntermediateSolutionCallback(
                    X=self.X,
                    checkpoint_manager=self.checkpoint_manager,
                    run_dir=run_dir,
                    stage_name=stage_name,
                    data=self.data,
                    save_interval=60  # Save at most every 60 seconds
                )
                status = solver.Solve(self.model, callback)
                self.logger.info(f"Callback found {callback.solution_count} solutions")
                print(f"  Callback found {callback.solution_count} solutions")
            else:
                self.logger.info("Calling solver.Solve() without callback...")
                status = solver.Solve(self.model)
                self.logger.info("solver.Solve() returned")
        except Exception as e:
            self.logger.error(f"Exception during solve: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Stop resource monitoring
            if self.resource_monitor:
                snapshots = self.resource_monitor.stop_monitoring()
                peak_memory = self.resource_monitor.get_peak_memory()
                if peak_memory:
                    self.logger.info(f"Peak process memory during solve: {peak_memory:.0f}MB")
        
        solve_time = (datetime.now() - start_time).total_seconds()
        
        status_name = solver.status_name(status)
        self.logger.info(f"Solve completed: status={status_name}, time={solve_time:.1f}s")
        print(f"  Status: {status_name}")
        print(f"  Solve time: {solve_time:.1f}s")
        
        # Log post-solve resources
        if self.resource_monitor:
            self.logger.info("Post-solve resource snapshot:")
            self.resource_monitor.log_snapshot(prefix="POST-SOLVE")
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self.logger.info("Extracting solution from solver...")
            solution = {key: solver.Value(var) for key, var in self.X.items()}
            objective = solver.ObjectiveValue()
            games_scheduled = sum(1 for v in solution.values() if v == 1)
            
            log_solve_result(status_name, objective, solve_time, games_scheduled, self.logger)
            print(f"  Objective: {objective}")
            print(f"  Games scheduled: {games_scheduled}")
            return status_name, solution, solve_time
        else:
            self.logger.warning(f"Stage {stage_name} did not find solution: {status_name}")
            return status_name, {}, solve_time
    
    def run_solver_stages_solve(self, run_id: str = None,
                                stages_override: list = None) -> dict:
        """Run solving driven by `data['solver_stages']` (Phase 7b).

        Uses `UnifiedConstraintEngine` for atomized clusters; falls back to
        the legacy solver class for non-engine atoms (e.g. Maximise/Minimise
        Broadmeadow constraints) via the registry.

        Each stage applies its atoms, builds the objective, and solves with
        the previous solution carried as hints. Solution is saved as a
        checkpoint per stage.
        """
        from constraints.unified import UnifiedConstraintEngine

        run_dir = self.checkpoint_manager.get_run_dir(run_id)
        self.logger.info(f"Run directory: {run_dir}")
        print(f"Run directory: {run_dir}")
        self.checkpoint_manager.save_run_metadata(run_dir, self.data, self.solver_config)

        stages = stages_override if stages_override is not None else self.data.get('solver_stages')
        if not stages:
            stages = load_solver_stages({})
        self.data['solver_stages'] = stages
        self.logger.info(f"Stages to run ({len(stages)}): {[s['name'] for s in stages]}")
        print(f"Mode: SOLVER_STAGES (config-driven, {len(stages)} stages)")

        engine = UnifiedConstraintEngine(self.model, self.X, self.data, skip_constraints=set())
        engine.build_groupings()

        if 'constraints_applied' not in self.data:
            self.data['constraints_applied'] = []

        applied_engine_keys: set = set()
        applied_atoms: set = set()

        for stage in stages:
            stage_name = stage['name']
            self.logger.info("=" * 60)
            self.logger.info(f"STARTING STAGE: {stage_name}")
            self.logger.info(f"  Atoms: {stage.get('atoms', [])}")
            self.logger.info("=" * 60)
            print(f"\n{'='*60}")
            print(f"STAGE: {stage_name}")
            print(f"Description: {stage.get('description', '')}")
            print(f"Atoms: {stage.get('atoms', [])}")
            print(f"{'='*60}")

            if self.current_solution:
                self.add_solution_hints(self.current_solution)

            print("Applying atoms via UnifiedConstraintEngine + registry fallbacks...")
            added, atoms_this_stage = apply_solver_stage(
                stage,
                model=self.model,
                X=self.X,
                data=self.data,
                engine=engine,
                applied_engine_keys=applied_engine_keys,
                applied_atoms=applied_atoms,
            )
            for atom in atoms_this_stage:
                self.data['constraints_applied'].append({
                    'name': atom,
                    'stage': stage_name,
                })
            print(f"  Atoms applied this stage: {atoms_this_stage}")
            print(f"  Constraints added this stage: {added}")
            print(f"  Model total constraints: {len(self.model.Proto().constraints)}")

            self.build_objective()

            stage_config = {
                'use_callback': True,
                'max_time_seconds': stage.get(
                    'time_limit_seconds',
                    self.data.get('max_time_per_stage', 172800),
                ),
            }
            print("Solving...")
            try:
                status, solution, solve_time = self.solve_stage(stage_config, run_dir, stage_name)
            except Exception as e:
                self.logger.critical(f"CRITICAL ERROR in stage {stage_name}: {e}")
                self.logger.critical(traceback.format_exc())
                print(f"CRITICAL ERROR: {e}")
                raise

            self.checkpoint_manager.save_stage(
                run_dir, stage_name, solution, self.data, status, solve_time
            )
            if status in ['OPTIMAL', 'FEASIBLE']:
                self.current_solution = solution
            elif stage.get('requires_complete_solution', True):
                self.logger.error(f"Required stage {stage_name} failed with status {status}")
                print(f"ERROR: Required stage {stage_name} failed with status {status}")
                self.checkpoint_manager.update_run_status(run_dir, 'failed', {'failed_stage': stage_name})
                return None
            else:
                print(f"WARNING: Stage {stage_name} did not find solution, using previous")

        final_games = sum(1 for v in self.current_solution.values() if v == 1) if self.current_solution else 0
        self.checkpoint_manager.update_run_status(run_dir, 'completed' if self.current_solution else 'failed', {
            'stages_completed': [s['name'] for s in stages],
            'num_scheduled_games': final_games,
        })
        return self.current_solution

    def run_staged_solve(self, run_id: str = None, resume_from: str = None,
                         stages_to_run: list = None, severity_staged: bool = False,
                         exclude_constraints: list = None,
                         keep_constraints: list = None) -> dict:
        """Run the staged solving process.

        Phase 7c: this method is now a thin shim over
        :meth:`run_solver_stages_solve`. The legacy ``STAGES`` /
        ``STAGES_AI`` dicts are gone; severity-staged solves build their
        stage list from the registry via
        :func:`constraints.stages.severity_solver_stages`. Other modes go
        through :data:`config.defaults.DEFAULT_STAGES` (or per-season
        ``solver_stages`` overrides).

        ``exclude_constraints`` becomes a name-based filter on the
        resolved stages' atom lists. ``keep_constraints`` (spec-023) is the
        complementary inclusion filter: when set, each stage's atoms are
        restricted to that set so a --groups selection applies identically here
        as in the simple path. When None, no inclusion filtering is done (legacy
        full selection).
        """
        if severity_staged:
            stages = severity_solver_stages()
        else:
            stages = self.data.get('solver_stages') or load_solver_stages({})
        if stages_to_run:
            wanted = set(stages_to_run)
            stages = [s for s in stages if s['name'] in wanted]

        # spec-023: restrict every stage's atoms to the resolved --groups set.
        # Stages that end up empty are removed.
        if keep_constraints is not None:
            keep_set = set(keep_constraints)
            filtered = []
            for s in stages:
                kept = [a for a in s.get('atoms', []) if a in keep_set]
                if kept:
                    filtered.append({**s, 'atoms': kept})
            stages = filtered

        # Drop excluded atoms from every stage. Stages that end up empty
        # are removed.
        exclude_set = set(exclude_constraints or [])
        if exclude_set:
            filtered = []
            for s in stages:
                kept = [a for a in s.get('atoms', []) if a not in exclude_set]
                if kept:
                    filtered.append({**s, 'atoms': kept})
            stages = filtered
            self.logger.info(f"Excluding constraints: {', '.join(exclude_set)}")
            print(f"  Excluding constraints: {', '.join(exclude_set)}")

        return self.run_solver_stages_solve(run_id=run_id, stages_override=stages)


# ============== Data Loading ==============

def load_data(year: int) -> dict:
    """
    Load all required data for scheduling for a specific year.
    
    This is the main entry point for loading season data in the staged solver.
    It uses the config system to load year-specific settings.
    
    Args:
        year: The season year (e.g., 2025, 2026). Required.
        
    Returns:
        Complete data dict ready for solver with teams, grades, timeslots, etc.
        
    Raises:
        ValueError: If no configuration exists for the specified year.
    """
    from config import load_season_data
    return load_season_data(year)


def _merge_regen_pins(data: dict, regen_locked_pairings: list) -> None:
    """Concatenate auto-extracted regen pins into data['locked_pairings'] (spec-026 DoD #3).

    Called AFTER load_data() inside main_staged/main_simple (the INJECTION ORDER
    NOTE: run_generate has no `data`, so the merge must happen here).  The
    extracted regen pins are appended to any hand-authored LOCKED_PAIRINGS from
    config so spec-025's generate_X pass enforces BOTH.

    Dedup: a hand-authored config pin that duplicates an extracted pin is
    collapsed so the same pairing is not double-pinned (Risks bullet —
    "data['locked_pairings'] concatenation order").  Two pins are the same iff
    they share ``(tuple(teams), grade, date)``.  Existing config pins win
    (kept in their original order); duplicate regen pins are dropped.  Pins
    that lack a ``teams`` key (scope-only hand-authored entries) are never
    deduped against and are passed through untouched.

    Mutates ``data['locked_pairings']`` in place.
    """
    if not regen_locked_pairings:
        return

    existing = data.get('locked_pairings', []) or []

    def _dedup_key(pin):
        # Only team-shaped pins participate in dedup; scope-only entries return
        # None and are always kept (no false collapse).
        if 'teams' not in pin:
            return None
        return (tuple(pin['teams']), pin.get('grade'), pin.get('date'))

    seen = set()
    merged = []
    for pin in existing:
        merged.append(pin)
        k = _dedup_key(pin)
        if k is not None:
            seen.add(k)
    dropped = 0
    for pin in regen_locked_pairings:
        k = _dedup_key(pin)
        if k is not None and k in seen:
            dropped += 1
            continue
        merged.append(pin)
        if k is not None:
            seen.add(k)

    data['locked_pairings'] = merged
    msg = (f"  Regen: merged {len(regen_locked_pairings)} extracted pin(s) with "
           f"{len(existing)} config pin(s) -> {len(merged)} total "
           f"({dropped} duplicate(s) collapsed)")
    print(msg)


# ============== Main Entry Points ==============

def main_staged(run_id: str = None, resume_from: str = None, locked_keys: set = None,
                locked_weeks: set = None, solver_config: SolverConfig = None, year: int = None,
                stages_to_run: list = None, relax_config: dict = None,
                fix_round_1: bool = False, constraint_slack: dict = None,
                severity_staged: bool = False, hint_solution: dict = None,
                exclude_constraints: list = None,
                description: str = '', provenance: dict = None,
                solver_stages: list = None,
                constraint_names: list = None, groups_selected: list = None,
                regen_locked_pairings: list = None,
                regen_info: dict = None):
    """Main entry point for staged solving (Phase 7c-clean).

    Args:
        run_id: Identifier for this run (auto-generated if None).
        resume_from: Stage name to resume from.
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        locked_weeks: Optional set of week numbers that are locked.
        solver_config: Optional solver configuration for resource management.
        year: The season year (e.g., 2025, 2026). Required.
        stages_to_run: Restrict to specific stage names. Default = all stages.
        relax_config: Optional dict with 'enabled' and 'timeout' for severity-based relaxation.
        fix_round_1: If True, apply Round 1 symmetry breaking to reduce search space.
        constraint_slack: Optional dict mapping constraint names to slack values.
        severity_staged: If True, dispatch via `severity_solver_stages()` instead of DEFAULT_STAGES.
        hint_solution: Optional dict of variable hints from a prior solution.
        exclude_constraints: Optional list of canonical atom names to exclude.
        description: User-provided description for metadata.
        provenance: Dict with locked_source, hint_source paths etc.
        solver_stages: Optional override for the stage list (from `--stages-config` etc.).
        constraint_names: spec-023 — resolved deduped union of selected --groups
            (canonical names), minus --exclude. When set, each stage's atoms are
            restricted to this set (and emptied stages dropped) so --groups
            selects an identical constraint set across the no-flag single solve,
            --staged, and --severity. When None, the full stage selection runs
            unchanged.
        groups_selected: spec-023 — the list of group names the operator
            requested (for metadata; defaults to ['default']).
        regen_locked_pairings: spec-026 regen pins (one per frozen game) to merge
            into data['locked_pairings'] after load_data(). None for a normal run.
        regen_info: spec-026 regen metadata block (source_draw, regen_grades, etc.)
            recorded in draw metadata. None for a normal run.

    Raises:
        ValueError: If year is not provided or no configuration exists for the year.
    """
    if year is None:
        raise ValueError("Year is required. Use --year YYYY to specify the season.")
    # Set up logging
    logger = setup_logging(run_id=run_id)
    logger.info("=" * 60)
    logger.info("HOCKEY DRAW SCHEDULER - STAGED SOLVING")
    logger.info("=" * 60)

    print("="*60)
    print("HOCKEY DRAW SCHEDULER - STAGED SOLVING")
    print("="*60)
    
    # Log system info
    log_system_info(logger)
    
    # Get solver config
    if solver_config is None:
        solver_config = get_recommended_config()
    logger.info(f"Using solver config: workers={solver_config.num_workers}, "
               f"linearization={solver_config.linearization_level}")
    
    # Load data
    logger.info(f"Loading data for year {year}...")
    print(f"\nLoading data for year {year}...")
    data = load_data(year)

    # spec-026: merge auto-extracted regen pins into data['locked_pairings']
    # (INJECTION ORDER NOTE — data is only available here, not in run_generate).
    _merge_regen_pins(data, regen_locked_pairings)
    if regen_info:
        data['_regen_info'] = regen_info

    # Set locked weeks in data for constraints to use
    if locked_weeks:
        data['locked_weeks'] = set(locked_weeks)

    # Apply constraint slack overrides
    if constraint_slack:
        data['constraint_slack'] = {**data.get('constraint_slack', {}), **constraint_slack}
        logger.info(f"  Constraint slack overrides: {constraint_slack}")
        print(f"  Constraint slack overrides: {constraint_slack}")

    logger.info(f"  Teams: {len(data['teams'])}")
    logger.info(f"  Grades: {len(data['grades'])}")
    logger.info(f"  Timeslots: {len(data['timeslots'])}")
    print(f"  {len(data['teams'])} teams")
    print(f"  {len(data['grades'])} grades")
    print(f"  {len(data['timeslots'])} timeslots")

    # Store solver provenance for metadata
    data['_solver_mode'] = 'severity' if severity_staged else 'staged'
    data['_solver_workers'] = solver_config.num_workers if solver_config else None
    data['_relax_enabled'] = bool(relax_config and relax_config.get('enabled'))
    data['_excluded_constraints'] = list(exclude_constraints or [])
    # spec-023: record the resolved group selection + applied constraint set.
    data['_groups_selected'] = list(groups_selected or ['default'])
    if constraint_names is not None:
        data['_constraint_names'] = list(constraint_names)
    if description:
        data['_user_description'] = description
    if provenance:
        data['_provenance'] = provenance

    # Phase 7b: caller-supplied solver_stages override (from --stages-config etc.).
    if solver_stages is not None:
        data['solver_stages'] = solver_stages

    # Initialize solver with config
    checkpoint_manager = CheckpointManager()
    solver = StagedScheduleSolver(
        data, 
        checkpoint_manager, 
        solver_config=solver_config, 
        logger=logger,
        relax_config=relax_config
    )
    
    # Initialize model
    unavailability_path = os.path.join('data', str(year), 'noplay')
    solver.initialize_model(unavailability_path)
    
    # Apply Round 1 symmetry breaking if requested
    if fix_round_1:
        logger.info("Applying Round 1 symmetry breaking constraints...")
        print("\nApplying Round 1 symmetry breaking...")
        from constraints.symmetry import FixRound1SymmetryBreaking
        symmetry_constraint = FixRound1SymmetryBreaking()
        num_constraints = symmetry_constraint.apply(solver.model, solver.X, data)
        logger.info(f"  Added {num_constraints} symmetry breaking constraints")
    
    # Handle locked games
    if locked_keys:
        locked_keys_set = set(locked_keys) if not isinstance(locked_keys, set) else locked_keys

        # Pre-validate locked keys against current timeslot data
        from utils import validate_locked_keys_or_exit, repair_locked_keys
        repair_mode = solver_config.repair_locked if solver_config and hasattr(solver_config, 'repair_locked') else False
        if repair_mode:
            repaired, repair_log = repair_locked_keys(list(locked_keys_set), data['timeslots'])
            if repair_log:
                repaired_count = sum(1 for r in repair_log if r.get('repaired'))
                failed_count = sum(1 for r in repair_log if not r.get('repaired'))
                logger.info(f"  Repaired {repaired_count} locked key(s), {failed_count} unrepairable")
                print(f"  Repaired {repaired_count} locked key(s)")
                for r in repair_log[:5]:
                    if r.get('repaired'):
                        diffs = ', '.join(f"{k}: {v['draw']}->{v['timeslot']}"
                                         for k, v in r['field_diffs'].items())
                        print(f"    Fixed: {r['original'][0]} vs {r['original'][1]} ({r['original'][2]}): {diffs}")
                if len(repair_log) > 5:
                    print(f"    ... and {len(repair_log) - 5} more")
                locked_keys_set = set(repaired)
        else:
            validate_locked_keys_or_exit(list(locked_keys_set), data,
                                         source_label=solver_config.locked_source if solver_config and hasattr(solver_config, 'locked_source') else 'locked draw')

        logger.info(f"Locking {len(locked_keys_set)} games from locked weeks...")
        print(f"  Locking {len(locked_keys_set)} games from locked weeks...")
        matched = 0
        for key in locked_keys_set:
            if key in solver.X:
                solver.model.Add(solver.X[key] == 1)
                matched += 1
        if matched < len(locked_keys_set):
            missed = len(locked_keys_set) - matched
            logger.warning(f"  WARNING: {missed}/{len(locked_keys_set)} locked keys not found in model variables!")
            print(f"  WARNING: {missed}/{len(locked_keys_set)} locked keys not found in model variables!")
        else:
            logger.info(f"  All {matched} locked keys matched model variables")

        # Zero out all other variables in locked weeks
        if locked_weeks:
            zeroed = 0
            for key, var in solver.X.items():
                if len(key) >= 7 and key[6] in locked_weeks and key not in locked_keys_set:
                    solver.model.Add(var == 0)
                    zeroed += 1
            logger.info(f"  Zeroed {zeroed} non-locked variables in locked weeks")
            print(f"  Zeroed {zeroed} non-locked variables in locked weeks")

        # Store locked keys in data so constraints can access them
        data['locked_keys_set'] = locked_keys_set

    # Apply solution hints if provided
    if hint_solution:
        hints_added = 0
        for key, value in hint_solution.items():
            if key in solver.X:
                solver.model.AddHint(solver.X[key], value)
                if value == 1:
                    hints_added += 1
        logger.info(f"Added {hints_added} solution hints from prior solution")
        print(f"  Added {hints_added} solution hints (solver will use as starting point)")

    # Run staged solve. Default path is SOLVER_STAGES (Phase 7b);
    # severity_staged builds its stage list from the registry via
    # `severity_solver_stages()`.
    try:
        if severity_staged:
            solution = solver.run_staged_solve(
                run_id=run_id, resume_from=resume_from,
                stages_to_run=stages_to_run, severity_staged=True,
                exclude_constraints=exclude_constraints,
                keep_constraints=constraint_names,
            )
        else:
            stages_override = data.get('solver_stages')
            if not stages_override:
                stages_override = load_solver_stages({})
                data['solver_stages'] = stages_override
            if stages_to_run:
                names = set(stages_to_run)
                stages_override = [s for s in stages_override if s['name'] in names]
            # spec-023: restrict each stage's atoms to the resolved --groups set
            # (drop emptied stages) so --groups selects identically here and in
            # the no-flag single solve. None => full selection unchanged.
            if constraint_names is not None:
                keep_set = set(constraint_names)
                filtered = []
                for s in stages_override:
                    kept = [a for a in s.get('atoms', []) if a in keep_set]
                    if kept:
                        filtered.append({**s, 'atoms': kept})
                stages_override = filtered
            solution = solver.run_solver_stages_solve(
                run_id=run_id, stages_override=stages_override,
            )
    except Exception as e:
        logger.critical(f"FATAL ERROR during solve: {e}")
        logger.critical(traceback.format_exc())
        print(f"\nFATAL ERROR: {e}")
        print("Check log files for details.")
        raise
    
    if solution:
        logger.info("Solve completed successfully, exporting results...")
        # Convert and export using unified versioning system
        print("\n" + "="*60)
        print("EXPORTING RESULTS")
        print("="*60)
        
        from analytics.versioning import DrawVersionManager
        
        version_manager = DrawVersionManager('draws', year=year)
        
        # Build description
        stages_desc = ', '.join(stages_to_run) if stages_to_run else 'all stages'
        auto_desc = f"Season {year} draw - {stages_desc}"
        description = f"{data['_user_description']} | {auto_desc}" if data.get('_user_description') else auto_desc
        
        version = version_manager.save_solver_output(
            solution, data,
            description=description,
            mode="staged",
            is_major=True
        )
        
        logger.info(f"Saved as {version.version_string}")
        logger.info("Main staged solve completed successfully")
        return solution, data
    else:
        print("\nNo valid solution found.")
        return None, data


def _main_simple_unified(model, X, data, solver_config, resource_monitor,
                         checkpoint_manager, run_dir, logger,
                         constraint_names: list = None):
    """Run a SINGLE full solve applying the entire selected constraint set.

    spec-036 (Unit A): this is now the no-flag default path. Instead of the old
    ``engine.apply_phase_a/b/c()`` (which silently omitted ~15 production atoms),
    we resolve the DEFAULT_STAGES list exactly as the staged dispatcher does and
    loop ``apply_solver_stage`` over EVERY stage in apply-only mode (no per-stage
    solve). This means the applied constraint set is IDENTICAL to the staged path
    for the same ``--groups`` selection. After all stages are applied we build the
    objective ONCE and run a SINGLE ``solver.Solve``.

    ``constraint_names`` (the resolved deduped union of selected --groups, minus
    --exclude) restricts each stage's atom list to the selection — exactly the
    ``keep_constraints`` filter used by ``run_staged_solve``. When ``None`` (no
    --groups passed), no inclusion filtering is done and the full DEFAULT_STAGES
    set is applied (a no-op filter, since DEFAULT_STAGES ⊆ the default group).
    --exclude is already subtracted from ``constraint_names`` upstream in run.py,
    so this single filter honours both --groups and --exclude.
    """
    from constraints.unified import UnifiedConstraintEngine

    # Resolve the stage list the same way the staged path does (DEFAULT_STAGES,
    # or a per-season `solver_stages` override).
    stages = data.get('solver_stages') or load_solver_stages({})

    # spec-036/spec-023: restrict every stage's atoms to the resolved --groups
    # set (= --groups minus --exclude, computed upstream). Stages that end up
    # empty are removed. When constraint_names is None, no filtering is done.
    if constraint_names is not None:
        keep_set = set(constraint_names)
        filtered = []
        for s in stages:
            kept = [a for a in s.get('atoms', []) if a in keep_set]
            if kept:
                filtered.append({**s, 'atoms': kept})
        stages = filtered
        logger.info(f"[spec-036] single-solve --groups selection: "
                    f"{len(keep_set)} atom(s) kept across {len(stages)} stage(s)")
        print(f"\n[spec-036] --groups selection: applying {len(keep_set)} atom(s) "
              f"across {len(stages)} stage(s)")

    # Build the engine with NO skip — filtering now happens on the stage atom
    # lists (apply_solver_stage dispatches engine vs non-engine internally).
    engine = UnifiedConstraintEngine(model, X, data, skip_constraints=set())
    engine.build_groupings()

    data['constraints_applied'] = []
    applied_engine_keys: set = set()
    applied_atoms: set = set()

    print("\nApplying full constraint set (single solve, all stages applied at once)...")
    logger.info(f"[spec-036] single-solve: applying {len(stages)} stage(s): "
                f"{[s['name'] for s in stages]}")
    for stage in stages:
        stage_name = stage['name']
        added, atoms_this_stage = apply_solver_stage(
            stage,
            model=model,
            X=X,
            data=data,
            engine=engine,
            applied_engine_keys=applied_engine_keys,
            applied_atoms=applied_atoms,
        )
        for atom in atoms_this_stage:
            data['constraints_applied'].append({
                'name': atom,
                'stage': stage_name,
            })
        print(f"  Stage '{stage_name}': atoms {atoms_this_stage} "
              f"({added} constraints)")
        logger.info(f"[spec-036] stage '{stage_name}' applied atoms "
                    f"{atoms_this_stage} ({added} constraints)")

    print(f"  Total atoms applied: {len(applied_atoms)} "
          f"({len(model.Proto().constraints)} model constraints)")

    # Build objective with normalized penalties
    penalties_dict = data.get('penalties', {})
    penalty_terms = _build_normalized_penalty(penalties_dict)
    total_penalty = sum(coeff * var for coeff, var in penalty_terms)
    objective_expr = sum(X.values()) - total_penalty
    _apply_objective_lower_bound(model, objective_expr, data)
    model.Maximize(objective_expr)

    # Penalties summary
    pen_summary = [(k, len(v.get('penalties', []))) for k, v in penalties_dict.items()]
    if pen_summary:
        print(f"  Penalties: {pen_summary}")

    # Solve
    print("\nSolving...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 259200

    if solver_config:
        solver_config.apply_to_solver(solver)
        print(f"  Solver configured: workers={solver_config.num_workers}")

    log_model_info(model, X, logger)

    if resource_monitor:
        print("\n[*] Pre-solve resource snapshot:")
        resource_monitor.log_snapshot(prefix="PRE-SOLVE")
        resource_monitor.start_monitoring(interval=30.0)
        logger.info("Started resource monitoring (interval: 30.0s)")

    stage_name = "unified_solve"
    callback = IntermediateSolutionCallback(
        X=X, checkpoint_manager=checkpoint_manager,
        run_dir=run_dir, stage_name=stage_name,
        data=data, save_interval=60
    )
    logger.info("Using solution callback for intermediate saves...")
    print("  Using solution callback for intermediate saves...")

    start_time = datetime.now()
    try:
        status = solver.Solve(model, callback)
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during solve: {e}")
        raise
    finally:
        if resource_monitor:
            resource_monitor.stop_monitoring()

    solve_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Callback found {callback.solution_count} intermediate solutions")
    print(f"  Callback found {callback.solution_count} intermediate solutions")

    if resource_monitor:
        resource_monitor.log_snapshot(prefix="POST-SOLVE")

    status_name = solver.status_name(status)
    print(f"\nStatus: {status_name}")
    print(f"Solve time: {solve_time:.1f}s")

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = {key: solver.Value(var) for key, var in X.items()}
        objective = solver.ObjectiveValue()
        games_scheduled = sum(1 for v in solution.values() if v == 1)

        log_solve_result(status_name, objective, solve_time, games_scheduled, logger)
        print(f"  Objective: {objective}")
        print(f"  Games scheduled: {games_scheduled}")

        checkpoint_manager.save_stage(run_dir, stage_name, solution, data, status_name, solve_time)

        from analytics.versioning import DrawVersionManager
        year = data.get('year', data.get('_year'))
        version_manager = DrawVersionManager('draws', year=year)
        version = version_manager.save_solver_output(
            solution, data, description=f"Season {year} draw - unified mode",
            mode="unified", is_major=True
        )
        logger.info(f"Saved as {version.version_string}")
        return solution, data
    else:
        logger.warning(f"Unified solve did not find solution: {status_name}")
        print("No valid solution found.")
        return None, data


def main_simple(locked_keys=None, locked_weeks=None, solver_config=None,
                exclude_constraints=None, year: int = None,
                relax_config: dict = None, fix_round_1: bool = False,
                constraint_slack: dict = None, hint_solution: dict = None,
                use_unified: bool = True, run_id: str = None,
                description: str = '', provenance: dict = None,
                constraint_names: list = None, groups_selected: list = None,
                regen_locked_pairings: list = None,
                regen_info: dict = None):
    """Single-solve main entry point (spec-036 no-flag DEFAULT).

    spec-036 (Unit A): this is the no-mode-flag default path. It routes through
    ``_main_simple_unified`` which now resolves the full DEFAULT_STAGES list and
    loops ``apply_solver_stage`` over EVERY stage (apply-only, no per-stage solve),
    then builds the objective once and runs a SINGLE ``solver.Solve``. The applied
    constraint set is therefore IDENTICAL to the staged path's for the same
    ``--groups`` selection (previously the engine-phase path silently omitted ~15
    production atoms). ``use_unified`` is retained as a no-op param for call-site
    back-compat (the CLI flag itself was removed in Unit A).

    spec-023: ``constraint_names`` is the resolved deduped union of selected
    --groups (canonical names), minus --exclude. It is propagated to
    ``_main_simple_unified`` where each resolved stage's atom list is filtered to
    this set — making --groups select identically in single-solve as in the
    staged path. When None, the full DEFAULT_STAGES set is applied unchanged.
    ``groups_selected`` is recorded in metadata.

    Args (spec-026 regen, None for a normal run):
        regen_locked_pairings: regen pins (one per frozen game) merged into
            data['locked_pairings'] after load_data().
        regen_info: regen metadata block recorded in draw metadata.

    Raises:
        ValueError: If year is not provided or no configuration exists for the year.
    """
    if year is None:
        raise ValueError("Year is required. Use --year YYYY to specify the season.")
    # Set up logging for single solve mode
    logger = setup_logging(run_id=run_id or "simple")

    logger.info("=" * 60)
    logger.info("HOCKEY DRAW SCHEDULER - SINGLE SOLVE")
    logger.info("=" * 60)
    print("="*60)
    print("HOCKEY DRAW SCHEDULER - SINGLE SOLVE")
    print("="*60)
    
    # Log system info
    log_system_info(logger)
    
    # Initialize resource monitoring
    resource_monitor = ResourceMonitor(logger=logger) if PSUTIL_AVAILABLE else None
    if resource_monitor:
        print("\n[*] Resource monitoring enabled")
        resource_monitor.log_snapshot(prefix="STARTUP")
    else:
        print("\n[!] Resource monitoring unavailable (install psutil)")
    
    # Initialize checkpoint manager for intermediate saves
    checkpoint_manager = CheckpointManager()
    run_dir = checkpoint_manager.get_run_dir()
    logger.info(f"Run directory: {run_dir}")
    print(f"Run directory: {run_dir}")
    
    # Load data
    print(f"\nLoading data for year {year}...")
    data = load_data(year)

    # spec-026: merge auto-extracted regen pins into data['locked_pairings']
    # (INJECTION ORDER NOTE — data is only available here, not in run_generate).
    _merge_regen_pins(data, regen_locked_pairings)
    if regen_info:
        data['_regen_info'] = regen_info

    # Set locked weeks in data for constraints to use
    if locked_weeks:
        data['locked_weeks'] = set(locked_weeks)

    # Apply constraint slack overrides
    if constraint_slack:
        data['constraint_slack'] = {**data.get('constraint_slack', {}), **constraint_slack}
        print(f"  Constraint slack overrides: {constraint_slack}")

    # Store solver provenance for metadata
    data['_solver_mode'] = 'unified'
    data['_solver_workers'] = solver_config.num_workers if solver_config else None
    data['_relax_enabled'] = bool(relax_config and relax_config.get('enabled'))
    data['_excluded_constraints'] = list(exclude_constraints or [])
    # spec-023: record the resolved group selection + applied constraint set.
    data['_groups_selected'] = list(groups_selected or ['default'])
    if constraint_names is not None:
        data['_constraint_names'] = list(constraint_names)
    if description:
        data['_user_description'] = description
    if provenance:
        data['_provenance'] = provenance

    # Save run-level metadata at start (before solving)
    checkpoint_manager.save_run_metadata(run_dir, data, solver_config)

    # Create model
    model = cp_model.CpModel()
    
    # Initialize variables
    X, conflicts = generate_X(model, data)

    # Handle locked games
    if locked_keys:
        locked_keys_set = set(locked_keys) if not isinstance(locked_keys, set) else locked_keys

        # Pre-validate locked keys against current timeslot data
        from utils import validate_locked_keys_or_exit, repair_locked_keys
        repair_mode = solver_config.repair_locked if solver_config and hasattr(solver_config, 'repair_locked') else False
        if repair_mode:
            repaired, repair_log = repair_locked_keys(list(locked_keys_set), data['timeslots'])
            if repair_log:
                repaired_count = sum(1 for r in repair_log if r.get('repaired'))
                print(f"  Repaired {repaired_count} locked key(s)")
                for r in repair_log[:5]:
                    if r.get('repaired'):
                        diffs = ', '.join(f"{k}: {v['draw']}->{v['timeslot']}"
                                         for k, v in r['field_diffs'].items())
                        print(f"    Fixed: {r['original'][0]} vs {r['original'][1]} ({r['original'][2]}): {diffs}")
                if len(repair_log) > 5:
                    print(f"    ... and {len(repair_log) - 5} more")
                locked_keys_set = set(repaired)
        else:
            validate_locked_keys_or_exit(list(locked_keys_set), data,
                                         source_label=solver_config.locked_source if solver_config and hasattr(solver_config, 'locked_source') else 'locked draw')

        print(f"  Locking {len(locked_keys_set)} games from locked weeks...")
        matched = 0
        for key in locked_keys_set:
            if key in X:
                model.Add(X[key] == 1)
                matched += 1
        if matched < len(locked_keys_set):
            missed = len(locked_keys_set) - matched
            print(f"  WARNING: {missed}/{len(locked_keys_set)} locked keys not found in model variables!")
        else:
            print(f"  All {matched} locked keys matched model variables")

        # Zero out all other variables in locked weeks
        if locked_weeks:
            zeroed = 0
            for key, var in X.items():
                if len(key) >= 7 and key[6] in locked_weeks and key not in locked_keys_set:
                    model.Add(var == 0)
                    zeroed += 1
            print(f"  Zeroed {zeroed} non-locked variables in locked weeks")

    # Store locked keys in data so constraints can access them
    if locked_keys:
        data['locked_keys_set'] = locked_keys_set

    data['team_conflicts'] = conflicts
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    
    print(f"  {len(X)} decision variables")

    # Apply solution hints if provided
    if hint_solution:
        hints_added = 0
        for key, value in hint_solution.items():
            if key in X:
                model.AddHint(X[key], value)
                if value == 1:
                    hints_added += 1
        print(f"  Added {hints_added} solution hints (solver will use as starting point)")

    # Apply Round 1 symmetry breaking if requested
    if fix_round_1:
        print("\nApplying Round 1 symmetry breaking...")
        from constraints.symmetry import FixRound1SymmetryBreaking
        symmetry_constraint = FixRound1SymmetryBreaking()
        num_constraints = symmetry_constraint.apply(model, X, data)
        print(f"  Added {num_constraints} symmetry breaking constraints")

    # spec-036: the single-solve path now applies the FULL DEFAULT_STAGES set in
    # one shot (loop apply_solver_stage over every stage, build objective once,
    # solve once) — identical applied set to the staged path. See
    # _main_simple_unified for the stage-resolution + filter + apply loop.
    return _main_simple_unified(model, X, data, solver_config, resource_monitor,
                                 checkpoint_manager, run_dir, logger,
                                 constraint_names=constraint_names)


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("HOCKEY DRAW SCHEDULER")
    print("="*60)
    print("\nUsage: python run.py generate --year YYYY [options]")
    print("\nDirect invocation of main_staged.py is deprecated.")
    print("Please use run.py as the entry point instead.")
    print("\nExamples:")
    print("  python run.py generate --year 2025")
    print("  python run.py generate --year 2026")
    print("  python run.py generate --year 2025 --resume run_13 stage1_required")
    print("  python run.py test draws/schedule.json --year 2025")
    print("  python run.py --help")
    sys.exit(1)
