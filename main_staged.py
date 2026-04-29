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
    ClubGameSpread,
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
    ClubGameSpreadAI,
)
from constraints.stages import (
    apply_solver_stage,
    load_solver_stages,
    list_stages,
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
            # Club game closeness
            ClubGameSpread,
        ],
        'required': False,
        'use_callback': True,
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
            # Club game closeness
            ClubGameSpreadAI,
        ],
        'required': False,
        'use_callback': True,
    },
}

# Unified constraint engine stages (3-phase: hard → soft → intra-day)
STAGES_UNIFIED = {
    'phase_a_hard': {
        'name': 'Hard Inter-week Constraints (Unified)',
        'description': 'Feasibility: scheduling decisions, no penalties',
        'phase': 'a',
        'required': True,
        'use_callback': True,
    },
    'phase_b_soft': {
        'name': 'Soft Inter-week Penalties (Unified)',
        'description': 'Quality: inter-week penalties on top of Phase A',
        'phase': 'b',
        'required': True,
        'use_callback': True,
    },
    'phase_c_intraday': {
        'name': 'Intra-day Optimization (Unified)',
        'description': 'Field/slot assignment within each day',
        'phase': 'c',
        'required': False,
        'use_callback': True,
    },
}

# Severity-based stages: group constraints by severity level (1-5)
# These allow progressive solving from CRITICAL → VERY LOW priority constraints.
STAGES_SEVERITY = {
    'severity_1': {
        'name': 'Critical Constraints',
        'description': 'Double-booking, equal games, home/away balance, PHL adjacency, matchup spacing',
        'constraints': [
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            EnsureEqualGamesAndBalanceMatchUps,
            EqualMatchUpSpacingConstraint,
            FiftyFiftyHomeandAway,
            MaitlandHomeGrouping,
            MaxMaitlandHomeWeekends,
            PHLAndSecondGradeAdjacency,
            PHLAndSecondGradeTimes,
        ],
        'required': True,
        'use_callback': True,
    },
    'severity_2': {
        'name': 'High Priority Constraints',
        'description': 'Club days, team conflicts, Maitland away grouping',
        'constraints': [
            AwayAtMaitlandGrouping,
            ClubDayConstraint,
            TeamConflictConstraint,
        ],
        'required': True,
        'use_callback': True,
    },
    'severity_3': {
        'name': 'Medium Priority Constraints',
        'description': 'Grade adjacency, club vs club alignment, club game spread',
        'constraints': [
            ClubGradeAdjacencyConstraint,
            ClubVsClubAlignment,
            ClubGameSpread,
        ],
        'required': True,
        'use_callback': True,
    },
    'severity_4': {
        'name': 'Low Priority Constraints',
        'description': 'Club density at Broadmeadow',
        'constraints': [
            MaximiseClubsPerTimeslotBroadmeadow,
            MinimiseClubsOnAFieldBroadmeadow,
        ],
        'required': False,
        'use_callback': True,
    },
    'severity_5': {
        'name': 'Very Low Priority Constraints',
        'description': 'Timeslot choices, preferred times',
        'constraints': [
            EnsureBestTimeslotChoices,
            PreferredTimesConstraint,
        ],
        'required': False,
        'use_callback': True,
    },
}

STAGES_SEVERITY_AI = {
    'severity_1': {
        'name': 'Critical Constraints (AI)',
        'description': 'Double-booking, equal games, home/away balance, PHL adjacency, matchup spacing - AI',
        'constraints': [
            NoDoubleBookingTeamsConstraintAI,
            NoDoubleBookingFieldsConstraintAI,
            EnsureEqualGamesAndBalanceMatchUpsAI,
            EqualMatchUpSpacingConstraintAI,
            FiftyFiftyHomeandAwayAI,
            MaitlandHomeGroupingAI,
            MaxMaitlandHomeWeekendsAI,
            PHLAndSecondGradeAdjacencyAI,
            PHLAndSecondGradeTimesAI,
        ],
        'required': True,
        'use_callback': True,
    },
    'severity_2': {
        'name': 'High Priority Constraints (AI)',
        'description': 'Club days, team conflicts, Maitland away grouping - AI',
        'constraints': [
            AwayAtMaitlandGroupingAI,
            ClubDayConstraintAI,
            TeamConflictConstraintAI,
        ],
        'required': True,
        'use_callback': True,
    },
    'severity_3': {
        'name': 'Medium Priority Constraints (AI)',
        'description': 'Grade adjacency, club vs club alignment, club game spread - AI',
        'constraints': [
            ClubGradeAdjacencyConstraintAI,
            ClubVsClubAlignmentAI,
            ClubGameSpreadAI,
        ],
        'required': True,
        'use_callback': True,
    },
    'severity_4': {
        'name': 'Low Priority Constraints (AI)',
        'description': 'Club density at Broadmeadow - AI',
        'constraints': [
            MaximiseClubsPerTimeslotBroadmeadowAI,
            MinimiseClubsOnAFieldBroadmeadowAI,
        ],
        'required': False,
        'use_callback': True,
    },
    'severity_5': {
        'name': 'Very Low Priority Constraints (AI)',
        'description': 'Timeslot choices, preferred times - AI',
        'constraints': [
            EnsureBestTimeslotChoicesAI,
            PreferredTimesConstraintAI,
        ],
        'required': False,
        'use_callback': True,
    },
}

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
            'use_ai': data.get('_use_ai', False),

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
            'use_ai': data.get('_use_ai', False),
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
                         use_ai: bool = False, exclude_constraints: list = None) -> dict:
        """
        Run the staged solving process.

        Args:
            run_id: Identifier for this run (for checkpointing)
            resume_from: Stage name to resume from
            stages_to_run: List of stage names to run (default: all)
            severity_staged: If True, use severity-based staging.
            use_ai: If True, use AI-enhanced constraint implementations.
            exclude_constraints: Optional list of constraint class names to exclude.

        Returns:
            Final solution dictionary
        """
        run_dir = self.checkpoint_manager.get_run_dir(run_id)
        self.logger.info(f"Run directory: {run_dir}")
        print(f"Run directory: {run_dir}")

        # Save run-level metadata at start (before solving)
        self.checkpoint_manager.save_run_metadata(run_dir, self.data, self.solver_config)

        # Select stage dictionary based on mode and AI flag
        if severity_staged:
            active_stages = STAGES_SEVERITY_AI if use_ai else STAGES_SEVERITY
            mode_label = f"SEVERITY-BASED stages ({'AI' if use_ai else 'original'})"
        else:
            active_stages = STAGES_AI if use_ai else STAGES
            mode_label = f"DEFAULT stages ({'AI' if use_ai else 'original'})"

        # Build exclusion set (normalize AI/non-AI suffixes)
        exclude_set = set(exclude_constraints or [])
        normalized_exclude = set()
        for name in exclude_set:
            normalized_exclude.add(name)
            if name.endswith('AI'):
                normalized_exclude.add(name[:-2])
            else:
                normalized_exclude.add(name + 'AI')

        # Filter excluded constraints from each stage
        if normalized_exclude:
            filtered_stages = {}
            for stage_id, stage_config in active_stages.items():
                filtered_config = dict(stage_config)
                filtered_config['constraints'] = [
                    cls for cls in stage_config['constraints']
                    if cls.__name__ not in normalized_exclude
                ]
                filtered_stages[stage_id] = filtered_config
            active_stages = filtered_stages
            self.logger.info(f"Excluding constraints: {', '.join(exclude_set)}")
            print(f"  Excluding constraints: {', '.join(exclude_set)}")
        print(f"Mode: {mode_label}")
        self.logger.info(f"Using {mode_label}")
        stages_to_run = stages_to_run or list(active_stages.keys())
        self.logger.info(f"Stages to run: {stages_to_run} (severity_staged={severity_staged})")
        
        # Handle resumption
        if resume_from:
            self.logger.info(f"Attempting to resume from stage: {resume_from}")
            last_stage = resume_from
            loaded = self.checkpoint_manager.load_stage(run_dir, resume_from)
            if loaded:
                self.current_solution = loaded['solution']
                games_in_hint = sum(1 for v in loaded['solution'].values() if v == 1)
                self.logger.info(f"Loaded checkpoint from {resume_from}: {games_in_hint} games")
                print(f"Resuming from {resume_from} (loaded {games_in_hint} games as hint)")
                
                # Skip stages before resume point
                skip_idx = stages_to_run.index(resume_from) + 1
                stages_to_run = stages_to_run[skip_idx:]
                print(f"Will run remaining stages: {stages_to_run}")
                self.logger.info(f"Remaining stages after resume: {stages_to_run}")
            else:
                self.logger.warning(f"Stage {resume_from} not found or checkpoint not loadable")
                self.logger.warning(f"Could not load checkpoint for {resume_from}")
        
        # Process each stage
        for stage_name in stages_to_run:
            stage_config = active_stages[stage_name]
            
            self.logger.info("=" * 60)
            self.logger.info(f"STARTING STAGE: {stage_name}")
            self.logger.info(f"  Name: {stage_config['name']}")
            self.logger.info(f"  Description: {stage_config['description']}")
            max_time = stage_config.get('max_time_seconds', self.data.get('max_time_per_stage', 172800))
            self.logger.info(f"  Max time: {max_time}s ({max_time/3600:.1f}h)")
            self.logger.info(f"  Required: {stage_config.get('required', False)}")
            self.logger.info("=" * 60)
            
            print(f"\n{'='*60}")
            print(f"STAGE: {stage_config['name']}")
            print(f"Description: {stage_config['description']}")
            print(f"{'='*60}")
            
            # Add hints from previous solution
            if self.current_solution:
                self.add_solution_hints(self.current_solution)
            
            # If relaxation is enabled, check feasibility first and relax if needed
            if self.relax_config.get('enabled'):
                problem_level = self.find_and_relax_problem_group(stage_config['constraints'])
                if problem_level == 1:
                    # Level 1 infeasibility - cannot proceed
                    self.logger.error("Level 1 constraints infeasible - aborting")
                    print("\n[FATAL] Level 1 constraints are infeasible. Check data/config.")
                    return None
            
            # Apply constraints (uses soft versions for relaxed groups)
            self.logger.info("Applying constraints...")
            print("Applying constraints:")
            constraints_added = self.apply_constraints(stage_config['constraints'])
            constraint_names = [cls.__name__ for cls in stage_config['constraints']]

            # Track cumulative constraints applied for metadata
            if 'constraints_applied' not in self.data:
                self.data['constraints_applied'] = []
            for cls in stage_config['constraints']:
                self.data['constraints_applied'].append({
                    'name': cls.__name__,
                    'stage': stage_name,
                })
            self.logger.info(f"  Constraints added this stage: {constraints_added}")
            self.logger.info(f"  Model total constraints: {len(self.model.Proto().constraints)}")
            print(f"  Applying {len(stage_config['constraints'])} constraints from stage: {constraint_names}")
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

        # Update run metadata with final status
        final_status = 'completed' if self.current_solution else 'failed'
        final_games = sum(1 for v in self.current_solution.values() if v == 1) if self.current_solution else 0
        self.checkpoint_manager.update_run_status(run_dir, final_status, {
            'stages_completed': stages_to_run,
            'num_scheduled_games': final_games,
        })

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
                locked_weeks: set = None, solver_config: SolverConfig = None, year: int = None,
                stages_to_run: list = None, relax_config: dict = None,
                fix_round_1: bool = False, constraint_slack: dict = None,
                severity_staged: bool = False, hint_solution: dict = None,
                use_ai: bool = False, exclude_constraints: list = None,
                description: str = '', provenance: dict = None,
                solver_stages: list = None):
    """
    Main entry point for staged solving.

    Args:
        run_id: Identifier for this run (auto-generated if None)
        resume_from: Stage name to resume from
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        locked_weeks: Optional set of week numbers that are locked.
        solver_config: Optional solver configuration for resource management.
        year: The season year (e.g., 2025, 2026). Required.
        stages_to_run: List of stage names to run (default: all). Available: stage1_required, stage2_soft.
            With severity_staged: severity_1, severity_2, severity_3, severity_4.
        relax_config: Optional dict with 'enabled' and 'timeout' for severity-based relaxation.
        fix_round_1: If True, apply Round 1 symmetry breaking to reduce search space.
        constraint_slack: Optional dict mapping constraint names to slack values.
        severity_staged: If True, use severity-based staging (5 levels by severity).
        hint_solution: Optional dict of variable hints from a prior solution.
        use_ai: If True, use AI-enhanced constraint implementations.
        exclude_constraints: Optional list of constraint class names to exclude.
        description: User-provided description for metadata.
        provenance: Dict with locked_source, hint_source paths etc.

    Raises:
        ValueError: If year is not provided or no configuration exists for the year.
    """
    if year is None:
        raise ValueError("Year is required. Use --year YYYY to specify the season.")
    # Set up logging
    logger = setup_logging(run_id=run_id)
    mode_label = "AI-ENHANCED" if use_ai else "ORIGINAL"
    logger.info("=" * 60)
    logger.info(f"HOCKEY DRAW SCHEDULER - STAGED SOLVING ({mode_label})")
    logger.info("=" * 60)

    print("="*60)
    print(f"HOCKEY DRAW SCHEDULER - STAGED SOLVING ({mode_label})")
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
    data['_use_ai'] = use_ai
    data['_solver_workers'] = solver_config.num_workers if solver_config else None
    data['_relax_enabled'] = bool(relax_config and relax_config.get('enabled'))
    data['_excluded_constraints'] = list(exclude_constraints or [])
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

    # Run staged solve. Default path is SOLVER_STAGES (Phase 7b); the legacy
    # severity_staged flow keeps using the hardcoded STAGES_SEVERITY map.
    try:
        if severity_staged:
            solution = solver.run_staged_solve(
                run_id=run_id, resume_from=resume_from,
                stages_to_run=stages_to_run, severity_staged=True,
                use_ai=use_ai, exclude_constraints=exclude_constraints,
            )
        else:
            stages_override = data.get('solver_stages')
            if not stages_override:
                stages_override = load_solver_stages({})
                data['solver_stages'] = stages_override
            if stages_to_run:
                names = set(stages_to_run)
                stages_override = [s for s in stages_override if s['name'] in names]
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
                         checkpoint_manager, run_dir, logger):
    """Run unified 3-phase constraint engine in simple mode."""
    from constraints.unified import UnifiedConstraintEngine

    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()

    print("\nApplying unified constraints (Phase A + B + C)...")
    a = engine.apply_phase_a()
    print(f"  Phase A (hard inter-week): {a} constraints")
    b = engine.apply_phase_b()
    print(f"  Phase B (soft inter-week): {b} constraints")
    c = engine.apply_phase_c()
    print(f"  Phase C (intra-day): {c} constraints")
    print(f"  Total: {a + b + c} engine constraints ({len(model.Proto().constraints)} model constraints)")

    data['constraints_applied'] = [
        {'name': 'UnifiedConstraintEngine', 'stage': 'unified_all'}
    ]

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


def main_simple(locked_keys=None, locked_weeks=None, solver_config=None, exclude_constraints=None, use_ai=False, year: int = None, relax_config: dict = None, fix_round_1: bool = False, constraint_slack: dict = None, hint_solution: dict = None, use_unified: bool = False, run_id: str = None, description: str = '', provenance: dict = None):
    """
    Simple (non-staged) main entry point.
    Uses all constraints in a single solve.

    Args:
        locked_keys: Optional set of game keys that are locked (pre-scheduled).
        locked_weeks: Optional set of week numbers that are locked.
        solver_config: Optional SolverConfig for solver parameters.
        exclude_constraints: Optional list of constraint class names to exclude.
        use_ai: If True, use AI-enhanced constraint implementations.
        year: The season year (e.g., 2025, 2026). Required.
        relax_config: Optional dict with 'enabled' and 'timeout' for severity-based relaxation.
        fix_round_1: If True, apply Round 1 symmetry breaking to reduce search space.
        constraint_slack: Optional dict mapping constraint names to slack values.
        hint_solution: Optional dict of variable hints from a prior solution.
        use_unified: If True, use unified constraint engine.
        run_id: Identifier for this run (default: "simple").

    Raises:
        ValueError: If year is not provided or no configuration exists for the year.
    """
    if year is None:
        raise ValueError("Year is required. Use --year YYYY to specify the season.")
    # Set up logging for single solve mode
    logger = setup_logging(run_id=run_id or "simple")
    
    mode_label = "UNIFIED" if use_unified else ("AI-ENHANCED" if use_ai else "ORIGINAL")
    logger.info("=" * 60)
    logger.info(f"HOCKEY DRAW SCHEDULER - SINGLE SOLVE ({mode_label})")
    logger.info("=" * 60)
    print("="*60)
    print(f"HOCKEY DRAW SCHEDULER - SINGLE SOLVE ({mode_label})")
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
    
    # Set locked weeks in data for constraints to use
    if locked_weeks:
        data['locked_weeks'] = set(locked_weeks)
    
    # Apply constraint slack overrides
    if constraint_slack:
        data['constraint_slack'] = {**data.get('constraint_slack', {}), **constraint_slack}
        print(f"  Constraint slack overrides: {constraint_slack}")

    # Store solver provenance for metadata
    data['_solver_mode'] = 'unified' if use_unified else 'simple'
    data['_use_ai'] = use_ai
    data['_solver_workers'] = solver_config.num_workers if solver_config else None
    data['_relax_enabled'] = bool(relax_config and relax_config.get('enabled'))
    data['_excluded_constraints'] = list(exclude_constraints or [])
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

    # === UNIFIED ENGINE PATH ===
    if use_unified:
        return _main_simple_unified(model, X, data, solver_config, resource_monitor,
                                     checkpoint_manager, run_dir, logger)

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

    # Track constraints for metadata
    data['constraints_applied'] = _build_constraints_applied(all_constraint_classes)

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
    
    # Build objective with normalized penalties
    penalties_dict = data.get('penalties', {})

    print(f"\nPenalty weight normalization:")
    for name, info in penalties_dict.items():
        n = len(info.get('penalties', []))
        w = info['weight']
        nw = max(1, w // n) if n > 0 else 0
        print(f"  {name}: {n} vars, weight {w:,} -> normalized {nw:,}/var")

    penalty_terms = _build_normalized_penalty(penalties_dict)
    total_penalty = sum(coeff * var for coeff, var in penalty_terms)

    objective_expr = sum(X.values()) - total_penalty
    _apply_objective_lower_bound(model, objective_expr, data)
    model.Maximize(objective_expr)

    # Solve
    print("\nSolving...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 259200  # 72 hours (matches staged mode)
    
    # Apply solver configuration if provided
    if solver_config:
        solver_config.apply_to_solver(solver)
        print(f"  Solver configured: workers={solver_config.num_workers}")
    
    # Log model info before solve
    log_model_info(model, X, logger)
    
    # Log pre-solve resource state
    if resource_monitor:
        print("\n[*] Pre-solve resource snapshot:")
        resource_monitor.log_snapshot(prefix="PRE-SOLVE")
    
    # Start background resource monitoring during solve
    if resource_monitor:
        resource_monitor.start_monitoring(interval=30.0)
        logger.info("Started resource monitoring (interval: 30.0s)")
    
    # Create intermediate solution callback for checkpoint saves
    stage_name = "simple_solve"
    callback = IntermediateSolutionCallback(
        X=X,
        checkpoint_manager=checkpoint_manager,
        run_dir=run_dir,
        stage_name=stage_name,
        data=data,
        save_interval=60
    )
    logger.info("Using solution callback for intermediate saves...")
    print("  Using solution callback for intermediate saves...")
    
    start_time = datetime.now()
    
    try:
        status = solver.Solve(model, callback)
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during solve: {e}")
        logger.critical(traceback.format_exc())
        print(f"\nCRITICAL ERROR: {e}")
        raise
    finally:
        # Stop resource monitoring
        if resource_monitor:
            snapshots = resource_monitor.stop_monitoring()
            peak_memory = resource_monitor.get_peak_memory()
            if peak_memory:
                logger.info(f"Peak process memory during solve: {peak_memory:.0f}MB")
                print(f"\n[*] Peak process memory: {peak_memory:.0f}MB")
    
    solve_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Callback found {callback.solution_count} intermediate solutions")
    print(f"  Callback found {callback.solution_count} intermediate solutions")
    
    # Log post-solve resource state
    if resource_monitor:
        print("\n[*] Post-solve resource snapshot:")
        resource_monitor.log_snapshot(prefix="POST-SOLVE")
    
    status_name = solver.status_name(status)
    print(f"\nStatus: {status_name}")
    print(f"Solve time: {solve_time:.1f}s")
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = {key: solver.Value(var) for key, var in X.items()}
        objective = solver.ObjectiveValue()
        games_scheduled = sum(1 for v in solution.values() if v == 1)
        
        # Log solve result
        log_solve_result(status_name, objective, solve_time, games_scheduled, logger)
        print(f"  Objective: {objective}")
        print(f"  Games scheduled: {games_scheduled}")
        
        # Save final checkpoint
        logger.info(f"Saving final checkpoint for {stage_name}...")
        checkpoint_manager.save_stage(
            run_dir, stage_name, solution, data, status_name, solve_time
        )
        logger.info("Final checkpoint saved successfully")
        
        # Use unified versioning system
        print("\n" + "="*60)
        print("EXPORTING RESULTS")
        print("="*60)
        
        from analytics.versioning import DrawVersionManager
        
        version_manager = DrawVersionManager('draws', year=year)
        
        excluded_desc = f", excluded: {', '.join(exclude_set)}" if exclude_set else ""
        auto_desc = f"Season {year} draw - simple mode{excluded_desc}"
        description = f"{data['_user_description']} | {auto_desc}" if data.get('_user_description') else auto_desc
        
        version = version_manager.save_solver_output(
            solution, data,
            description=description,
            mode="simple",
            is_major=True
        )
        
        logger.info(f"Saved as {version.version_string}")
        logger.info("Simple solve completed successfully")
        checkpoint_manager.update_run_status(run_dir, 'completed', {
            'num_scheduled_games': games_scheduled,
            'objective': objective,
            'solve_time': solve_time,
            'status': status_name,
        })
        return solution, data
    else:
        logger.warning(f"Simple solve did not find solution: {status_name}")
        print("No valid solution found.")
        checkpoint_manager.update_run_status(run_dir, 'failed', {
            'solve_time': solve_time,
            'status': status_name,
        })
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
