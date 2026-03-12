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
from datetime import datetime, timedelta, time as tm
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from pathlib import Path

from ortools.sat.python import cp_model
import pandas as pd

from models import PlayingField, Club, Team, Grade, Timeslot
from utils import (
    convert_X_to_roster, export_roster_to_excel, get_teams_from_club,
    generate_timeslots, max_games_per_grade, generate_X
)
from solver_diagnostics import (
    setup_logging, ResourceMonitor, SolverConfig, get_recommended_config,
    log_system_info, log_model_info, log_solve_result, PSUTIL_AVAILABLE
)
from constraints import (
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    PHLAndSecondGradeAdjacency,
    PHLAndSecondGradeTimes,
    FiftyFiftyHomeandAway,
    TeamConflictConstraint,
    ClubGradeAdjacencyConstraint,
    MaxMaitlandHomeWeekends,
    EnsureBestTimeslotChoices,
    ClubDayConstraint,
    EqualMatchUpSpacingConstraint,
    ClubVsClubAlignment,
    MaitlandHomeGrouping,
    AwayAtMaitlandGrouping,
    MaximiseClubsPerTimeslotBroadmeadow,
    MinimiseClubsOnAFieldBroadmeadow,
    PreferredTimesConstraint,
)
from constraints.ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    PHLAndSecondGradeAdjacencyAI,
    PHLAndSecondGradeTimesAI,
    FiftyFiftyHomeandAwayAI,
    TeamConflictConstraintAI,
    ClubGradeAdjacencyConstraintAI,
    MaxMaitlandHomeWeekendsAI,
    EnsureBestTimeslotChoicesAI,
    ClubDayConstraintAI,
    EqualMatchUpSpacingConstraintAI,
    ClubVsClubAlignmentAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    MaximiseClubsPerTimeslotBroadmeadowAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
    PreferredTimesConstraintAI,
)


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

STAGES = {
    'stage1_required': {
        'name': 'Required Constraints',
        'description': 'Core scheduling rules that must be satisfied',
        'constraints': [
            # Core double-booking prevention
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            # Game balance
            EnsureEqualGamesAndBalanceMatchUps,
            # Grade adjacency and timing
            PHLAndSecondGradeAdjacency,
            PHLAndSecondGradeTimes,
            # Home/Away balance
            FiftyFiftyHomeandAway,
            # Team conflicts
            TeamConflictConstraint,
            # Venue constraints
            MaxMaitlandHomeWeekends,
            # Club day events
            ClubDayConstraint,
            # Spacing
            EqualMatchUpSpacingConstraint,
            # Grade adjacency for clubs
            ClubGradeAdjacencyConstraint,
            # Club alignment
            ClubVsClubAlignment,
            # Maitland grouping (has hard element: no back-to-back)
            MaitlandHomeGrouping,
            # Away at Maitland (hard limit of 3 away clubs)
            AwayAtMaitlandGrouping,
        ],
        'max_time_seconds': 259200,  # 72 hours (3 days)
        'required': True,
        'use_callback': True,
    },
    'stage2_soft': {
        'name': 'Soft Preferences and Timeslot Optimization',
        'description': 'Quality optimizations with penalties',
        'constraints': [
            # Timeslot optimization
            EnsureBestTimeslotChoices,
            # Club diversity at Broadmeadow
            MaximiseClubsPerTimeslotBroadmeadow,
            # Field continuity at Broadmeadow
            MinimiseClubsOnAFieldBroadmeadow,
            # Preferred times / no-play constraints
            PreferredTimesConstraint,
        ],
        'max_time_seconds': 259200,  # 72 hours (3 days)
        'required': False,
        'use_callback': True,  # Save intermediate solutions
    },
}

# AI-enhanced constraint stages (mirrors STAGES but uses AI implementations)
STAGES_AI = {
    'stage1_required': {
        'name': 'Required Constraints (AI)',
        'description': 'Core scheduling rules - AI implementations',
        'constraints': [
            # Core double-booking prevention
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
            # Game balance
            EnsureEqualGamesAndBalanceMatchUpsAI,
            # Grade adjacency and timing
            PHLAndSecondGradeAdjacencyAI,
            PHLAndSecondGradeTimesAI,
            # Home/Away balance
            FiftyFiftyHomeandAwayAI,
            # Team conflicts
            TeamConflictConstraintAI,
            # Venue constraints
            MaxMaitlandHomeWeekendsAI,
            # Club day events
            ClubDayConstraintAI,
            # Spacing
            EqualMatchUpSpacingConstraintAI,
            # Grade adjacency for clubs
            ClubGradeAdjacencyConstraintAI,
            # Club alignment
            ClubVsClubAlignmentAI,
            # Maitland grouping (has hard element: no back-to-back)
            MaitlandHomeGroupingAI,
            # Away at Maitland (hard limit of 3 away clubs)
            AwayAtMaitlandGroupingAI,
        ],
        'max_time_seconds': 259200,  # 72 hours (3 days)
        'required': True,
        'use_callback': True,
    },
    'stage2_soft': {
        'name': 'Soft Preferences and Timeslot Optimization (AI)',
        'description': 'Quality optimizations - AI implementations',
        'constraints': [
            # Timeslot optimization
            EnsureBestTimeslotChoicesAI,
            # Club diversity at Broadmeadow
            MaximiseClubsPerTimeslotBroadmeadowAI,
            # Field continuity at Broadmeadow
            MinimiseClubsOnAFieldBroadmeadowAI,
            # Preferred times / no-play constraints
            PreferredTimesConstraintAI,
        ],
        'max_time_seconds': 259200,
        'required': False,
        'use_callback': True,
    },
}


# ============== Severity-Based Stages ==============
# 
# These stages organize constraints by severity level (1-4).
# Each stage adds constraints cumulatively:
#   - severity_1: Level 1 only (CRITICAL)
#   - severity_2: Level 1 + Level 2 (add HIGH)  
#   - severity_3: Level 1 + Level 2 + Level 3 (add MEDIUM)
#   - severity_4: All levels (add LOW)
#
# Between stages, the previous solution is used as a HINT.

STAGES_SEVERITY = {
    'severity_1': {
        'name': 'Level 1 - CRITICAL',
        'description': 'Core constraints that must never be broken',
        'constraints': [
            # Double-booking prevention
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            # Game balance
            EnsureEqualGamesAndBalanceMatchUps,
            # Grade adjacency and timing
            PHLAndSecondGradeAdjacency,
            PHLAndSecondGradeTimes,
            # Home/Away balance
            FiftyFiftyHomeandAway,
            # Venue constraints
            MaxMaitlandHomeWeekends,
            # Maitland grouping (hard element: no back-to-back)
            MaitlandHomeGrouping,
        ],
        'max_time_seconds': 259200,
        'required': True,
        'use_callback': True,
    },
    'severity_2': {
        'name': 'Level 2 - HIGH',
        'description': 'Structural, club-specific constraints',
        'constraints': [
            # Club day events
            ClubDayConstraint,
            # Away at Maitland (hard limit)
            AwayAtMaitlandGrouping,
            # Team conflicts
            TeamConflictConstraint,
        ],
        'max_time_seconds': 259200,
        'required': True,
        'use_callback': True,
    },
    'severity_3': {
        'name': 'Level 3 - MEDIUM',
        'description': 'Spacing and alignment constraints',
        'constraints': [
            # Spacing
            EqualMatchUpSpacingConstraint,
            # Grade adjacency for clubs
            ClubGradeAdjacencyConstraint,
            # Club alignment
            ClubVsClubAlignment,
        ],
        'max_time_seconds': 259200,
        'required': True,
        'use_callback': True,
    },
    'severity_4': {
        'name': 'Level 4 - LOW',
        'description': 'Soft optimization constraints',
        'constraints': [
            # Timeslot optimization
            EnsureBestTimeslotChoices,
            # Club diversity at Broadmeadow
            MaximiseClubsPerTimeslotBroadmeadow,
            # Field continuity at Broadmeadow
            MinimiseClubsOnAFieldBroadmeadow,
            # Preferred times / no-play constraints
            PreferredTimesConstraint,
        ],
        'max_time_seconds': 259200,
        'required': False,
        'use_callback': True,
    },
}

STAGES_SEVERITY_AI = {
    'severity_1': {
        'name': 'Level 1 - CRITICAL (AI)',
        'description': 'Core constraints - AI implementations',
        'constraints': [
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
            EnsureEqualGamesAndBalanceMatchUpsAI,
            PHLAndSecondGradeAdjacencyAI,
            PHLAndSecondGradeTimesAI,
            FiftyFiftyHomeandAwayAI,
            MaxMaitlandHomeWeekendsAI,
            MaitlandHomeGroupingAI,
        ],
        'max_time_seconds': 259200,
        'required': True,
        'use_callback': True,
    },
    'severity_2': {
        'name': 'Level 2 - HIGH (AI)',
        'description': 'Structural constraints - AI implementations',
        'constraints': [
            ClubDayConstraintAI,
            AwayAtMaitlandGroupingAI,
            TeamConflictConstraintAI,
        ],
        'max_time_seconds': 259200,
        'required': True,
        'use_callback': True,
    },
    'severity_3': {
        'name': 'Level 3 - MEDIUM (AI)',
        'description': 'Spacing constraints - AI implementations',
        'constraints': [
            EqualMatchUpSpacingConstraintAI,
            ClubGradeAdjacencyConstraintAI,
            ClubVsClubAlignmentAI,
        ],
        'max_time_seconds': 259200,
        'required': True,
        'use_callback': True,
    },
    'severity_4': {
        'name': 'Level 4 - LOW (AI)',
        'description': 'Soft optimization - AI implementations',
        'constraints': [
            EnsureBestTimeslotChoicesAI,
            MaximiseClubsPerTimeslotBroadmeadowAI,
            MinimiseClubsOnAFieldBroadmeadowAI,
            PreferredTimesConstraintAI,
        ],
        'max_time_seconds': 259200,
        'required': False,
        'use_callback': True,
    },
}


# ============== Checkpoint Management ==============

class CheckpointManager:
    """Manages saving and loading of solver state."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_run_dir(self, run_id: str = None) -> Path:
        """Get or create a run directory."""
        if run_id is None:
            # Create new run
            existing = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
            run_num = len(existing) + 1
            run_id = f'run_{run_num}'
        
        run_dir = self.checkpoint_dir / run_id
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def save_stage(self, run_dir: Path, stage_name: str, X_solution: dict, data: dict, 
                   status: str, solve_time: float):
        """Save stage results to checkpoint."""
        stage_dir = run_dir / stage_name
        stage_dir.mkdir(exist_ok=True)
        
        # Save solution
        with open(stage_dir / 'solution.pkl', 'wb') as f:
            pickle.dump(X_solution, f)
        
        # Save metadata
        metadata = {
            'stage': stage_name,
            'status': status,
            'solve_time': solve_time,
            'timestamp': datetime.now().isoformat(),
            'num_scheduled_games': sum(1 for v in X_solution.values() if v == 1),
        }
        
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
    
    def get_last_completed_stage(self, run_dir: Path) -> str:
        """Find the last successfully completed stage."""
        for stage_name in reversed(list(STAGES.keys())):
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
        self.Y = None
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
        self.X, self.Y, conflicts, unavailable_games = generate_X(
            self.model, self.data
        )
        
        # Merge X and Y
        self.X.update(self.Y)
        
        self.data['unavailable_games'] = unavailable_games
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
        """Build the optimization objective."""
        penalties_dict = self.data.get('penalties', {})
        
        print(f"  Penalties: {[(name, len(info.get('penalties', []))) for name, info in penalties_dict.items()]}")
        
        total_penalty = sum(
            info['weight'] * sum(info['penalties'])
            for info in penalties_dict.values() if 'penalties' in info
        )
        
        # Objective: maximize games scheduled, minimize dummy variables, minimize penalties
        self.model.Maximize(
            sum(self.X.values()) - sum(self.Y.values() if self.Y else []) - total_penalty
        )
    
    def solve_stage(self, stage_config: dict, run_dir: Path = None, stage_name: str = None) -> tuple:
        """Solve a single stage with comprehensive logging and resource monitoring."""
        self.logger.info(f"Starting solve for stage: {stage_name}")
        self.logger.info(f"  Max time: {stage_config['max_time_seconds']}s ({stage_config['max_time_seconds']/3600:.1f}h)")
        
        # Log model info before solve
        log_model_info(self.model, self.X, self.logger)
        
        # Log resource state before solve
        if self.resource_monitor:
            self.logger.info("Pre-solve resource snapshot:")
            self.resource_monitor.log_snapshot(prefix="PRE-SOLVE")
        
        solver = cp_model.CpSolver()
        
        # Apply solver configuration (includes num_workers, memory settings)
        stage_specific_config = SolverConfig(
            max_time_seconds=stage_config['max_time_seconds'],
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
    
    def run_staged_solve(self, run_id: str = None, resume_from: str = None, 
                         stages_to_run: list = None, use_ai: bool = False,
                         severity_staged: bool = False) -> dict:
        """
        Run the staged solving process.
        
        IMPORTANT: OR-Tools CP-SAT does not allow "locking in" a partial solution
        and adding more constraints. Instead, we use a HINT-based approach:
        
        1. When resuming from a checkpoint, we load the prior solution as a HINT
        2. We build a NEW model with ALL constraints (prior stages + current stage)
        3. The solver uses the hint to guide its search toward the prior solution
        4. The solver is free to deviate if the combined constraints require it
        
        This is the correct way to do staged solving with OR-Tools.
        
        Args:
            run_id: Identifier for this run (for checkpointing)
            resume_from: Stage name to resume from (solution used as hint)
            stages_to_run: List of stage names to run (default: all)
            use_ai: Whether to use AI-enhanced constraint implementations
            severity_staged: If True, use severity-based stages (4 levels) instead of default stages
        
        Returns:
            Final solution dictionary
        """
        run_dir = self.checkpoint_manager.get_run_dir(run_id)
        self.logger.info(f"Run directory: {run_dir}")
        print(f"Run directory: {run_dir}")
        
        # Get the stage definitions to use based on mode
        if severity_staged:
            stage_defs = STAGES_SEVERITY_AI if use_ai else STAGES_SEVERITY
            self.logger.info("Using SEVERITY-BASED stages (4 levels)")
            print("Mode: SEVERITY-BASED stages (Level 1 → 2 → 3 → 4)")
        else:
            stage_defs = STAGES_AI if use_ai else STAGES
            self.logger.info("Using DEFAULT stages (required + soft)")
        
        all_stage_names = list(stage_defs.keys())
        stages_to_run = stages_to_run or all_stage_names
        self.logger.info(f"Stages to run: {stages_to_run}")
        
        # Track which stages have been applied (for cumulative constraint application)
        applied_stages = set()
        
        # Handle resumption - load prior solution as HINT
        if resume_from:
            self.logger.info(f"Attempting to resume from stage: {resume_from}")
            loaded = self.checkpoint_manager.load_stage(run_dir, resume_from)
            if loaded:
                self.current_solution = loaded['solution']
                games_in_hint = sum(1 for v in loaded['solution'].values() if v == 1)
                self.logger.info(f"Loaded checkpoint from {resume_from}: {games_in_hint} games as HINT")
                print(f"Resuming from {resume_from} (loaded {games_in_hint} games as hint)")
                
                # Mark that we have a hint but need to apply ALL constraints
                # Find the resume stage index to determine which stages to run
                try:
                    resume_idx = all_stage_names.index(resume_from)
                    # We run from the NEXT stage after resume_from
                    stages_to_run = all_stage_names[resume_idx + 1:]
                    self.logger.info(f"Will run remaining stages: {stages_to_run}")
                    print(f"Will run remaining stages: {stages_to_run}")
                    
                    # BUT we need to apply constraints from ALL prior stages (including resume_from)
                    # This happens in the stage loop below
                except ValueError:
                    self.logger.warning(f"Stage {resume_from} not found in {all_stage_names}")
            else:
                self.logger.warning(f"Could not load checkpoint for {resume_from}")
        
        # Process each stage
        for stage_name in stages_to_run:
            stage_config = stage_defs[stage_name]
            
            self.logger.info("=" * 60)
            self.logger.info(f"STARTING STAGE: {stage_name}")
            self.logger.info(f"  Name: {stage_config['name']}")
            self.logger.info(f"  Description: {stage_config['description']}")
            self.logger.info(f"  Max time: {stage_config['max_time_seconds']}s")
            self.logger.info(f"  Required: {stage_config.get('required', False)}")
            self.logger.info("=" * 60)
            
            print(f"\n{'='*60}")
            print(f"STAGE: {stage_config['name']}")
            print(f"Description: {stage_config['description']}")
            print(f"{'='*60}")
            
            # Add hints from previous solution BEFORE applying any constraints
            # This guides the solver toward a known feasible solution
            if self.current_solution:
                self.add_solution_hints(self.current_solution)
            
            # Determine which constraints to apply for this stage
            # We need ALL constraints from prior stages + current stage (cumulative)
            constraints_to_apply = []
            
            # Find stages up to and including current one that haven't been applied yet
            for s_name in all_stage_names:
                if s_name not in applied_stages:
                    s_config = stage_defs[s_name]
                    constraints_to_apply.extend(s_config['constraints'])
                    applied_stages.add(s_name)
                    
                    # Stop when we reach the current stage
                    if s_name == stage_name:
                        break
            
            self.logger.info(f"  Constraints to apply: {len(constraints_to_apply)} from stages: {applied_stages}")
            print(f"  Applying {len(constraints_to_apply)} constraints from stages: {list(applied_stages)}")
            
            # If relaxation is enabled, check feasibility first and relax if needed
            if self.relax_config.get('enabled'):
                problem_level = self.find_and_relax_problem_group(constraints_to_apply)
                if problem_level == 1:
                    # Level 1 infeasibility - cannot proceed
                    self.logger.error("Level 1 constraints infeasible - aborting")
                    print("\n[FATAL] Level 1 constraints are infeasible. Check data/config.")
                    return None
            
            # Apply constraints (uses soft versions for relaxed groups)
            self.logger.info("Applying constraints...")
            print("Applying constraints:")
            constraints_added = self.apply_constraints(constraints_to_apply)
            self.logger.info(f"  Constraints added this stage: {constraints_added}")
            self.logger.info(f"  Model total constraints: {len(self.model.Proto().constraints)}")
            print(f"  Total: {constraints_added} constraints added")
            print(f"  Model total: {len(self.model.Proto().constraints)} constraints")
            
            # Build objective
            self.build_objective()
            
            # Solve with full exception handling
            print("Solving...")
            try:
                status, solution, solve_time = self.solve_stage(stage_config, run_dir, stage_name)
            except Exception as e:
                self.logger.critical(f"CRITICAL ERROR in stage {stage_name}: {e}")
                self.logger.critical(traceback.format_exc())
                print(f"CRITICAL ERROR: {e}")
                raise
            
            # Save checkpoint
            self.logger.info(f"Saving checkpoint for {stage_name}...")
            self.checkpoint_manager.save_stage(
                run_dir, stage_name, solution, self.data, status, solve_time
            )
            self.logger.info(f"Checkpoint saved successfully")
            
            # Handle result
            if status in ['OPTIMAL', 'FEASIBLE']:
                self.current_solution = solution
                self.logger.info(f"Stage {stage_name} completed successfully")
            else:
                if stage_config.get('required', False):
                    self.logger.error(f"Required stage {stage_name} failed with status {status}")
                    print(f"ERROR: Required stage {stage_name} failed with status {status}")
                    return None
                else:
                    print(f"WARNING: Stage {stage_name} did not find solution, using previous")
        
        return self.current_solution


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


# ============== Main Entry Points ==============

def main_staged(run_id: str = None, resume_from: str = None, locked_keys: set = None,
                solver_config: SolverConfig = None, year: int = None,
                stages_to_run: list = None, relax_config: dict = None,
                fix_round_1: bool = False, constraint_slack: dict = None,
                use_ai: bool = False, severity_staged: bool = False):
    """
    Main entry point for staged solving.
    
    Args:
        run_id: Identifier for this run (auto-generated if None)
        resume_from: Stage name to resume from (solution is used as HINT)
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        solver_config: Optional solver configuration for resource management.
        year: The season year (e.g., 2025, 2026). Required.
        stages_to_run: List of stage names to run (default: all).
            Default mode: stage1_required, stage2_soft
            Severity mode (--staged): severity_1, severity_2, severity_3, severity_4
        relax_config: Optional dict with 'enabled' and 'timeout' for severity-based relaxation.
        fix_round_1: If True, apply Round 1 symmetry breaking to reduce search space.
        constraint_slack: Optional dict mapping constraint names to slack values.
        use_ai: If True, use AI-enhanced constraint implementations.
        severity_staged: If True, use severity-based staging (4 levels by severity).
        
    Note:
        When resuming, the prior solution is used as a HINT to guide the solver,
        but ALL constraints (from prior stages + current stage) are applied.
        The solver may deviate from the hint if required by the combined constraints.
        
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
        logger.info(f"Locking {len(locked_keys)} games from prior weeks...")
        print(f"  Locking {len(locked_keys)} games from prior weeks...")
        for key in locked_keys:
            if key in solver.X:
                solver.model.Add(solver.X[key] == 1)
    
    # Run staged solve
    try:
        solution = solver.run_staged_solve(
            run_id=run_id, 
            resume_from=resume_from, 
            stages_to_run=stages_to_run,
            use_ai=use_ai,
            severity_staged=severity_staged
        )
    except Exception as e:
        logger.critical(f"FATAL ERROR during solve: {e}")
        logger.critical(traceback.format_exc())
        print(f"\nFATAL ERROR: {e}")
        print("Check log files for details.")
        raise
    
    if solution:
        logger.info("Solve completed successfully, exporting results...")
        # Convert and export
        print("\n" + "="*60)
        print("EXPORTING RESULTS")
        print("="*60)
        
        roster = convert_X_to_roster(solution, data)
        
        output_dir = Path('draws')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'schedule_{timestamp}.xlsx'
        
        export_roster_to_excel(roster, data, filename=str(output_path))
        logger.info(f"Schedule exported to {output_path}")
        print(f"Schedule exported to {output_path}")
        
        # ============== NEW: Save pliable format and generate analytics ==============
        from draw_analytics import DrawStorage, DrawAnalytics
        from draw_tester import DrawTester
        
        # Save in pliable JSON format
        draw = DrawStorage.from_X_solution(solution, description=f"Season Draw {timestamp}")
        json_path = output_dir / f'draw_{timestamp}.json'
        draw.save(str(json_path))
        logger.info(f"Draw saved to {json_path}")
        print(f"Draw saved to {json_path}")
        
        # Generate comprehensive analytics
        analytics = DrawAnalytics(draw, data)
        analytics_path = output_dir / f'analytics_{timestamp}.xlsx'
        analytics.export_analytics_to_excel(str(analytics_path))
        logger.info(f"Analytics exported to {analytics_path}")
        print(f"Analytics exported to {analytics_path}")
        
        # Run violation check
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()
        logger.info(f"Violation Check: {report.summary()}")
        print(f"\nViolation Check: {report.summary()}")
        
        if report.has_violations:
            report_path = output_dir / f'violations_{timestamp}.txt'
            with open(report_path, 'w') as f:
                f.write(report.full_report())
            logger.warning(f"Violations found! Report saved to {report_path}")
            print(f"Violation report saved to {report_path}")
        
        logger.info("Main staged solve completed successfully")
        return solution, data
    else:
        print("\nNo valid solution found.")
        return None, data


def main_simple(locked_keys=None, solver_config=None, exclude_constraints=None, use_ai=False, year: int = None, relax_config: dict = None, fix_round_1: bool = False, constraint_slack: dict = None):
    """
    Simple (non-staged) main entry point.
    Uses all constraints in a single solve.
    
    Args:
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        solver_config: Optional SolverConfig for solver parameters.
        exclude_constraints: Optional list of constraint class names to exclude.
        use_ai: If True, use AI-enhanced constraint implementations.
        year: The season year (e.g., 2025, 2026). Required.
        relax_config: Optional dict with 'enabled' and 'timeout' for severity-based relaxation.
        fix_round_1: If True, apply Round 1 symmetry breaking to reduce search space.
        constraint_slack: Optional dict mapping constraint names to slack values.
        
    Raises:
        ValueError: If year is not provided or no configuration exists for the year.
    """
    if year is None:
        raise ValueError("Year is required. Use --year YYYY to specify the season.")
    mode_label = "AI-ENHANCED" if use_ai else "ORIGINAL"
    print("="*60)
    print(f"HOCKEY DRAW SCHEDULER - SINGLE SOLVE ({mode_label})")
    print("="*60)
    
    # Initialize resource monitoring
    resource_monitor = ResourceMonitor() if PSUTIL_AVAILABLE else None
    if resource_monitor:
        print("\n📊 Resource monitoring enabled")
        resource_monitor.log_snapshot(prefix="STARTUP")
    else:
        print("\n⚠️  Resource monitoring unavailable (install psutil)")
    
    # Load data
    print(f"\nLoading data for year {year}...")
    data = load_data(year)
    
    # Apply constraint slack overrides
    if constraint_slack:
        data['constraint_slack'] = {**data.get('constraint_slack', {}), **constraint_slack}
        print(f"  Constraint slack overrides: {constraint_slack}")
    
    # Create model
    model = cp_model.CpModel()
    
    # Initialize variables
    X, Y, conflicts, unavailable_games = generate_X(model, data)
    X.update(Y)
    
    # Handle locked games
    if locked_keys:
        print(f"  Locking {len(locked_keys)} games from prior weeks...")
        for key in locked_keys:
            if key in X:
                model.Add(X[key] == 1)
    
    data['unavailable_games'] = unavailable_games
    data['team_conflicts'] = conflicts
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    
    print(f"  {len(X)} decision variables")
    
    # Apply Round 1 symmetry breaking if requested
    if fix_round_1:
        print("\nApplying Round 1 symmetry breaking...")
        from constraints.symmetry import FixRound1SymmetryBreaking
        symmetry_constraint = FixRound1SymmetryBreaking()
        num_constraints = symmetry_constraint.apply(model, X, data)
        print(f"  Added {num_constraints} symmetry breaking constraints")
    
    # Get constraint classes (not instances yet)
    stages = STAGES_AI if use_ai else STAGES
    
    exclude_set = set(exclude_constraints or [])
    # Normalize exclusion: accept names with or without 'AI' suffix
    # e.g., '--exclude FooConstraint' also excludes 'FooConstraintAI' and vice versa
    normalized_exclude = set()
    for name in exclude_set:
        normalized_exclude.add(name)
        if name.endswith('AI'):
            normalized_exclude.add(name[:-2])  # Strip 'AI'
        else:
            normalized_exclude.add(name + 'AI')  # Add 'AI'
    
    if exclude_set:
        print(f"  Excluding constraints: {', '.join(exclude_set)}")
    
    all_constraint_classes = [
        cls for stage in stages.values() 
        for cls in stage['constraints']
        if cls.__name__ not in normalized_exclude
    ]
    
    # Relaxation: find problem severity group if enabled
    relaxed_groups = {}  # Dict[int, int] mapping severity level -> slack level
    
    if relax_config and relax_config.get('enabled'):
        from constraints.severity import (
            SeverityGroupResolver,
            create_relaxation_test_func,
            get_severity_level
        )
        
        timeout = relax_config.get('timeout', 30.0)
        
        print("\n" + "="*60)
        print("SEVERITY-BASED RELAXATION")
        print("="*60)
        
        resolver = SeverityGroupResolver(all_constraint_classes, verbose=True)
        test_func = create_relaxation_test_func(data, generate_X, timeout=timeout)
        
        problem_level = resolver.find_problem_severity_group(test_func, timeout)
        
        if problem_level == 1:
            print("\n[FATAL] Level 1 constraints are infeasible. Check data/config.")
            return None, data
        elif problem_level is not None:
            print(f"\n[RELAX] Setting slack=1 for severity level {problem_level}")
            relaxed_groups[problem_level] = 1
            print(resolver.get_state_summary())
    
    # Apply all constraints (using soft versions for relaxed groups)
    print("\nApplying constraints...")
    
    for cls in all_constraint_classes:
        # Check if this constraint's severity group is relaxed
        if relaxed_groups:
            from constraints.severity import get_severity_level
            level = get_severity_level(cls)
            
            if level in relaxed_groups:
                # Use soft version with specified slack
                slack = relaxed_groups[level]
                from constraints.soft import get_soft_constraint
                soft_instance = get_soft_constraint(cls.__name__, slack)
                
                if soft_instance:
                    soft_instance.apply(model, X, data)
                    print(f"  {cls.__name__} (SOFT slack={slack})")
                    continue
        
        # Use hard constraint
        constraint = cls()
        constraint.apply(model, X, data)
        print(f"  {cls.__name__}")
    
    print(f"  Total: {len(model.Proto().constraints)} constraints")
    
    # Build objective
    penalties_dict = data.get('penalties', {})
    total_penalty = sum(
        info['weight'] * sum(info['penalties'])
        for info in penalties_dict.values() if 'penalties' in info
    )
    
    model.Maximize(sum(X.values()) - sum(Y.values()) - total_penalty)
    
    # Solve
    print("\nSolving...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 28800  # 8 hours
    
    # Apply solver configuration if provided
    if solver_config:
        solver_config.apply_to_solver(solver)
        print(f"  Solver configured: workers={solver_config.num_workers}")
    
    # Log pre-solve resource state
    if resource_monitor:
        print("\n📊 Pre-solve resource snapshot:")
        resource_monitor.log_snapshot(prefix="PRE-SOLVE")
    
    status = solver.Solve(model)
    
    # Log post-solve resource state
    if resource_monitor:
        print("\n📊 Post-solve resource snapshot:")
        resource_monitor.log_snapshot(prefix="POST-SOLVE")
    
    print(f"\nStatus: {solver.status_name(status)}")
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = {key: solver.Value(var) for key, var in X.items()}
        
        roster = convert_X_to_roster(solution, data)
        
        output_path = os.path.join('draws', 'schedule.xlsx')
        export_roster_to_excel(roster, data, filename=output_path)
        print(f"Schedule exported to {output_path}")
        
        return solution, data
    else:
        print("No valid solution found.")
        return None, data


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
    print("  python run.py generate --year 2026 --simple")
    print("  python run.py generate --year 2025 --resume run_13 stage1_required")
    print("  python run.py test draws/schedule.json --year 2025")
    print("  python run.py --help")
    sys.exit(1)
