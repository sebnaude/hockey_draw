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
from constraints_ai import (
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
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            EnsureEqualGamesAndBalanceMatchUps,
            FiftyFiftyHomeandAway,
        ],
        'max_time_seconds': 7200,  # 2 hours
        'required': True,
        'use_callback': False,
    },
    'stage2_strong': {
        'name': 'Strong Structural Constraints',
        'description': 'Important practical constraints for schedule quality',
        'constraints': [
            PHLAndSecondGradeAdjacency,
            ClubGradeAdjacencyConstraint,
            MaxMaitlandHomeWeekends,
        ],
        'max_time_seconds': 14400,  # 4 hours
        'required': True,
        'use_callback': True,  # Save intermediate solutions
    },
    'stage3_medium': {
        'name': 'Venue and Scheduling Optimization',
        'description': 'Venue limits and scheduling efficiency',
        'constraints': [
            ClubDayConstraint,
            EqualMatchUpSpacingConstraint,
        ],
        'max_time_seconds': 28800,  # 8 hours
        'required': False,
        'use_callback': True,  # Save intermediate solutions
    },
    'stage4_soft': {
        'name': 'Soft Preferences',
        'description': 'Quality optimizations with penalties',
        'constraints': [
            # Hybrid constraints (have both hard and soft elements)
            PHLAndSecondGradeTimes,
            MaitlandHomeGrouping,
            AwayAtMaitlandGrouping,
            # Pure soft constraints
            MaximiseClubsPerTimeslotBroadmeadow,
            MinimiseClubsOnAFieldBroadmeadow,
            ClubVsClubAlignment,
            PreferredTimesConstraint,
            EnsureBestTimeslotChoices,
            TeamConflictConstraint,
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
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
            EnsureEqualGamesAndBalanceMatchUpsAI,
            FiftyFiftyHomeandAwayAI,
        ],
        'max_time_seconds': 7200,
        'required': True,
        'use_callback': False,
    },
    'stage2_strong': {
        'name': 'Strong Structural Constraints (AI)',
        'description': 'Important practical constraints - AI implementations',
        'constraints': [
            PHLAndSecondGradeAdjacencyAI,
            ClubGradeAdjacencyConstraintAI,
            MaxMaitlandHomeWeekendsAI,
        ],
        'max_time_seconds': 14400,
        'required': True,
        'use_callback': True,
    },
    'stage3_medium': {
        'name': 'Venue and Scheduling Optimization (AI)',
        'description': 'Venue limits and scheduling efficiency - AI implementations',
        'constraints': [
            ClubDayConstraintAI,
            EqualMatchUpSpacingConstraintAI,
        ],
        'max_time_seconds': 28800,
        'required': False,
        'use_callback': True,
    },
    'stage4_soft': {
        'name': 'Soft Preferences (AI)',
        'description': 'Quality optimizations - AI implementations',
        'constraints': [
            PHLAndSecondGradeTimesAI,
            MaitlandHomeGroupingAI,
            AwayAtMaitlandGroupingAI,
            MaximiseClubsPerTimeslotBroadmeadowAI,
            MinimiseClubsOnAFieldBroadmeadowAI,
            ClubVsClubAlignmentAI,
            PreferredTimesConstraintAI,
            EnsureBestTimeslotChoicesAI,
            TeamConflictConstraintAI,
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
                 solver_config: SolverConfig = None, logger: logging.Logger = None):
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
        
        self.logger.info(f"StagedScheduleSolver initialized")
        self.logger.info(f"Solver config: workers={self.solver_config.num_workers}, "
                        f"linearization={self.solver_config.linearization_level}")
    
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
        """Apply a list of constraints to the model."""
        total_constraints = 0
        prior_count = len(self.model.Proto().constraints)
        
        for constraint_class in constraint_classes:
            constraint = constraint_class()
            constraint.apply(self.model, self.X, self.data)
            
            current_count = len(self.model.Proto().constraints)
            added = current_count - prior_count
            print(f"    {constraint_class.__name__}: {added} constraints")
            total_constraints += added
            prior_count = current_count
        
        return total_constraints
    
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
                         stages_to_run: list = None) -> dict:
        """
        Run the staged solving process.
        
        Args:
            run_id: Identifier for this run (for checkpointing)
            resume_from: Stage name to resume from
            stages_to_run: List of stage names to run (default: all)
        
        Returns:
            Final solution dictionary
        """
        run_dir = self.checkpoint_manager.get_run_dir(run_id)
        self.logger.info(f"Run directory: {run_dir}")
        print(f"Run directory: {run_dir}")
        
        stages_to_run = stages_to_run or list(STAGES.keys())
        self.logger.info(f"Stages to run: {stages_to_run}")
        
        # Handle resumption
        if resume_from:
            self.logger.info(f"Attempting to resume from stage: {resume_from}")
            last_stage = resume_from
            loaded = self.checkpoint_manager.load_stage(run_dir, resume_from)
            if loaded:
                self.current_solution = loaded['solution']
                self.logger.info(f"Loaded checkpoint from {resume_from}: "
                               f"{sum(1 for v in loaded['solution'].values() if v == 1)} games")
                print(f"Resuming from {resume_from}")
                
                # Skip stages before resume point
                skip_idx = stages_to_run.index(resume_from) + 1
                stages_to_run = stages_to_run[skip_idx:]
                self.logger.info(f"Remaining stages after resume: {stages_to_run}")
            else:
                self.logger.warning(f"Could not load checkpoint for {resume_from}")
        
        # Process each stage
        for stage_name in stages_to_run:
            stage_config = STAGES[stage_name]
            
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
            
            # Add hints from previous solution
            if self.current_solution:
                self.add_solution_hints(self.current_solution)
            
            # Apply constraints
            self.logger.info("Applying constraints...")
            print("Applying constraints:")
            constraints_added = self.apply_constraints(stage_config['constraints'])
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
                solver_config: SolverConfig = None, year: int = None):
    """
    Main entry point for staged solving.
    
    Args:
        run_id: Identifier for this run (auto-generated if None)
        resume_from: Stage name to resume from
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        solver_config: Optional solver configuration for resource management.
        year: The season year (e.g., 2025, 2026). Required.
        
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
    logger.info(f"  Teams: {len(data['teams'])}")
    logger.info(f"  Grades: {len(data['grades'])}")
    logger.info(f"  Timeslots: {len(data['timeslots'])}")
    print(f"  {len(data['teams'])} teams")
    print(f"  {len(data['grades'])} grades")
    print(f"  {len(data['timeslots'])} timeslots")
    
    # Initialize solver with config
    checkpoint_manager = CheckpointManager()
    solver = StagedScheduleSolver(data, checkpoint_manager, solver_config=solver_config, logger=logger)
    
    # Initialize model
    unavailability_path = os.path.join('data', str(year), 'noplay')
    solver.initialize_model(unavailability_path)
    
    # Handle locked games
    if locked_keys:
        logger.info(f"Locking {len(locked_keys)} games from prior weeks...")
        print(f"  Locking {len(locked_keys)} games from prior weeks...")
        for key in locked_keys:
            if key in solver.X:
                solver.model.Add(solver.X[key] == 1)
    
    # Run staged solve
    try:
        solution = solver.run_staged_solve(run_id=run_id, resume_from=resume_from)
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


def main_simple(locked_keys=None, solver_config=None, exclude_constraints=None, use_ai=False, year: int = None):
    """
    Simple (non-staged) main entry point.
    Uses all constraints in a single solve.
    
    Args:
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        solver_config: Optional SolverConfig for solver parameters.
        exclude_constraints: Optional list of constraint class names to exclude.
        use_ai: If True, use AI-enhanced constraint implementations.
        year: The season year (e.g., 2025, 2026). Required.
        
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
    
    # Apply all constraints
    print("\nApplying constraints...")
    
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
    
    all_constraints = [
        cls() for stage in stages.values() 
        for cls in stage['constraints']
        if cls.__name__ not in normalized_exclude
    ]
    
    for constraint in all_constraints:
        constraint.apply(model, X, data)
        print(f"  {constraint.__class__.__name__}")
    
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
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--simple':
            main_simple()
        elif sys.argv[1] == '--resume':
            run_id = sys.argv[2] if len(sys.argv) > 2 else None
            resume_from = sys.argv[3] if len(sys.argv) > 3 else None
            main_staged(run_id=run_id, resume_from=resume_from)
        else:
            main_staged(run_id=sys.argv[1])
    else:
        main_staged()
