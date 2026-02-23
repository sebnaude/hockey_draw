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
import pickle
import json
from datetime import datetime, timedelta, time as tm
from collections import defaultdict
from pathlib import Path

from ortools.sat.python import cp_model
import pandas as pd

from models import PlayingField, Club, Team, Grade, Timeslot
from utils import convert_X_to_roster, export_roster_to_excel, get_teams_from_club
from generate_x import generate_X
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


# ============== Stage Definitions ==============

STAGES = {
    'stage1_required': {
        'name': 'Required Constraints',
        'description': 'Core scheduling rules that must be satisfied',
        'constraints': [
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            EnsureEqualGamesAndBalanceMatchUps,
        ],
        'max_time_seconds': 3600,  # 1 hour
        'required': True,
    },
    'stage2_strong': {
        'name': 'Strong Structural Constraints',
        'description': 'Important practical constraints for schedule quality',
        'constraints': [
            TeamConflictConstraint,
            PHLAndSecondGradeAdjacency,
            PHLAndSecondGradeTimes,
            FiftyFiftyHomeandAway,
            ClubGradeAdjacencyConstraint,
        ],
        'max_time_seconds': 7200,  # 2 hours
        'required': True,
    },
    'stage3_medium': {
        'name': 'Venue and Scheduling Optimization',
        'description': 'Venue limits and scheduling efficiency',
        'constraints': [
            MaxMaitlandHomeWeekends,
            EnsureBestTimeslotChoices,
            ClubDayConstraint,
            EqualMatchUpSpacingConstraint,
        ],
        'max_time_seconds': 7200,  # 2 hours
        'required': False,
    },
    'stage4_soft': {
        'name': 'Soft Preferences',
        'description': 'Quality optimizations with penalties',
        'constraints': [
            MaitlandHomeGrouping,
            AwayAtMaitlandGrouping,
            MaximiseClubsPerTimeslotBroadmeadow,
            MinimiseClubsOnAFieldBroadmeadow,
            ClubVsClubAlignment,
            PreferredTimesConstraint,
        ],
        'max_time_seconds': 14400,  # 4 hours
        'required': False,
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
    
    def __init__(self, data: dict, checkpoint_manager: CheckpointManager = None):
        self.data = data
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.model = None
        self.X = None
        self.Y = None
        self.current_solution = None
    
    def initialize_model(self, unavailability_path: str):
        """Initialize the CP model with decision variables."""
        print("Initializing model and decision variables...")
        
        self.model = cp_model.CpModel()
        
        # Import generate_X from the notebook translation for full functionality
        from main_notebook_translation import generate_X as generate_X_full
        
        self.X, self.Y, conflicts, unavailable_games = generate_X_full(
            unavailability_path, self.model, self.data
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
    
    def solve_stage(self, stage_config: dict) -> tuple:
        """Solve a single stage."""
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = stage_config['max_time_seconds']
        
        start_time = datetime.now()
        status = solver.Solve(self.model)
        solve_time = (datetime.now() - start_time).total_seconds()
        
        status_name = solver.StatusName()
        print(f"  Status: {status_name}")
        print(f"  Solve time: {solve_time:.1f}s")
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = {key: solver.Value(var) for key, var in self.X.items()}
            objective = solver.ObjectiveValue()
            print(f"  Objective: {objective}")
            print(f"  Games scheduled: {sum(1 for v in solution.values() if v == 1)}")
            return status_name, solution, solve_time
        else:
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
        print(f"Run directory: {run_dir}")
        
        stages_to_run = stages_to_run or list(STAGES.keys())
        
        # Handle resumption
        if resume_from:
            last_stage = resume_from
            loaded = self.checkpoint_manager.load_stage(run_dir, resume_from)
            if loaded:
                self.current_solution = loaded['solution']
                print(f"Resuming from {resume_from}")
                
                # Skip stages before resume point
                skip_idx = stages_to_run.index(resume_from) + 1
                stages_to_run = stages_to_run[skip_idx:]
        
        # Process each stage
        for stage_name in stages_to_run:
            stage_config = STAGES[stage_name]
            
            print(f"\n{'='*60}")
            print(f"STAGE: {stage_config['name']}")
            print(f"Description: {stage_config['description']}")
            print(f"{'='*60}")
            
            # Add hints from previous solution
            if self.current_solution:
                self.add_solution_hints(self.current_solution)
            
            # Apply constraints
            print("Applying constraints:")
            constraints_added = self.apply_constraints(stage_config['constraints'])
            print(f"  Total: {constraints_added} constraints added")
            print(f"  Model total: {len(self.model.Proto().constraints)} constraints")
            
            # Build objective
            self.build_objective()
            
            # Solve
            print("Solving...")
            status, solution, solve_time = self.solve_stage(stage_config)
            
            # Save checkpoint
            self.checkpoint_manager.save_stage(
                run_dir, stage_name, solution, self.data, status, solve_time
            )
            
            # Handle result
            if status in ['OPTIMAL', 'FEASIBLE']:
                self.current_solution = solution
            else:
                if stage_config.get('required', False):
                    print(f"ERROR: Required stage {stage_name} failed with status {status}")
                    return None
                else:
                    print(f"WARNING: Stage {stage_name} did not find solution, using previous")
        
        return self.current_solution


# ============== Data Loading ==============

def load_data() -> dict:
    """Load all required data for scheduling."""
    from main_notebook_translation import generate_timeslots, max_games_per_grade
    
    # Define fields
    FIELDS = [
        PlayingField(location='Newcastle International Hockey Centre', name='SF'),
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
        PlayingField(location='Central Coast Hockey Park', name='Wyong Main Field'),
    ]
    
    # Load teams and clubs from CSV files
    TEAMS_DATA = os.path.join('data', '2025', 'teams')
    CLUBS = []
    TEAMS = []
    
    for file in os.listdir(TEAMS_DATA):
        if not file.endswith('.csv'):
            continue
        
        df = pd.read_csv(os.path.join(TEAMS_DATA, file))
        club_name = df['Club'].iloc[0].strip()
        
        home_field = (
            'Maitland Park' if club_name == 'Maitland' else
            'Central Coast Hockey Park' if club_name == 'Gosford' else
            'Newcastle International Hockey Centre'
        )
        
        club = Club(name=club_name, home_field=home_field)
        CLUBS.append(club)
        
        teams = [
            Team(
                name=f"{row['Team Name'].strip()} {row['Grade'].strip()}", 
                club=club, 
                grade=row['Grade'].strip()
            ) 
            for _, row in df.iterrows()
        ]
        TEAMS.extend(teams)
    
    # Create grades
    teams_by_grade = defaultdict(list)
    for team in TEAMS:
        teams_by_grade[team.grade].append(team.name)
    
    GRADES = [Grade(name=grade, teams=teams) for grade, teams in sorted(teams_by_grade.items())]
    
    # Update club team counts
    teams_by_club = defaultdict(list)
    for team in TEAMS:
        teams_by_club[team.club.name].append(team.name)
    
    for club in CLUBS:
        club.num_teams = len(teams_by_club.get(club.name, []))
    
    for grade in GRADES:
        grade.num_teams = len(grade.teams)
    
    # Time configuration
    day_time_map = {
        'Newcastle International Hockey Centre': {
            'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
        },
        'Maitland Park': {
            'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
        }
    }
    
    phl_game_times = {
        'Newcastle International Hockey Centre': {
            'Friday': [tm(19, 0)], 
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        'Central Coast Hockey Park': {
            'Friday': [tm(20, 0)], 
            'Sunday': [tm(15, 0)]
        },
        'Maitland Park': {
            'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
        }
    }
    
    # Field unavailabilities
    field_unavailabilities = {
        'Maitland Park': {
            'weekends': [
                datetime(2025, 4, 19), datetime(2025, 4, 12), datetime(2025, 5, 10),
                datetime(2025, 5, 24), datetime(2025, 6, 28), datetime(2025, 5, 3), datetime(2025, 6, 7)
            ],
            'whole_days': [datetime(2025, 4, 25)],
            'part_days': [],
        },
        'Newcastle International Hockey Centre': {
            'weekends': [datetime(2025, 4, 19), datetime(2025, 5, 3), datetime(2025, 6, 7)],
            'whole_days': [datetime(2025, 4, 25), datetime(2025, 5, 31)],
            'part_days': [
                datetime(2025, 6, 1, 8, 30), 
                datetime(2025, 6, 1, 10, 0), 
                datetime(2025, 6, 1, 11, 30)
            ],
        },
        'Central Coast Hockey Park': {
            'weekends': [datetime(2025, 4, 19), datetime(2025, 4, 5), datetime(2025, 5, 3), datetime(2025, 6, 7)],
            'whole_days': [datetime(2025, 4, 25)],
            'part_days': [],
        },
    }
    
    # Generate timeslots
    start = datetime(2025, 3, 21)
    end = datetime(2025, 9, 2)
    
    merged_dict = defaultdict(lambda: defaultdict(list))
    for d in (phl_game_times, day_time_map):
        for field, days in d.items():
            for key, times in days.items():
                merged_dict[field][key].extend(times)
    
    for field in merged_dict:
        for key in merged_dict[field]:
            merged_dict[field][key] = list(dict.fromkeys(merged_dict[field][key]))
            merged_dict[field][key].sort()
    
    timeslots = generate_timeslots(start, end, merged_dict, FIELDS, field_unavailabilities)
    TIMESLOTS = [
        Timeslot(
            date=t['date'], day=t['day'], time=t['time'], week=t['week'],
            day_slot=t['day_slot'], field=t['field'], round_no=t['round_no']
        ) 
        for t in timeslots
    ]
    
    # Calculate rounds per grade
    max_rounds = 21
    num_rounds = max_games_per_grade(GRADES, max_rounds)
    num_rounds['max'] = max_rounds
    
    for grade, rounds in num_rounds.items():
        grade_obj = next((g for g in GRADES if g.name == grade), None)
        if grade_obj:
            grade_obj.set_games(rounds)
    
    # Max day slots per field
    max_day_slot_per_field = {
        field.location: max(t.day_slot for t in TIMESLOTS if t.field.location == field.location)
        for field in FIELDS
    }
    
    # Club days
    club_days = {
        'Crusaders': datetime(2025, 6, 22),
        'Wests': datetime(2025, 7, 13),
        'University': datetime(2025, 7, 27),
        'Tigers': datetime(2025, 7, 6),
        'Port Stephens': datetime(2025, 7, 20)
    }
    
    # No-play preferences
    preference_no_play = {
        'Maitland': [
            {'date': '2025-07-20', 'field_location': 'Newcastle International Hockey Centre'},
            {'date': '2025-08-24', 'field_location': 'Newcastle International Hockey Centre'}
        ],
        'Norths': [
            {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '11:30'},
            {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '13:00'},
            {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '14:30'},
            {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '16:00'}
        ]
    }
    
    phl_preferences = {'preferred_dates': []}
    
    return {
        'teams': TEAMS,
        'grades': GRADES,
        'fields': FIELDS,
        'clubs': CLUBS,
        'timeslots': TIMESLOTS,
        'num_rounds': num_rounds,
        'current_week': 0,
        'penalties': {},
        'day_time_map': day_time_map,
        'phl_game_times': phl_game_times,
        'phl_preferences': phl_preferences,
        'max_day_slot_per_field': max_day_slot_per_field,
        'field_unavailabilities': field_unavailabilities,
        'club_days': club_days,
        'preference_no_play': preference_no_play,
        'num_dummy_timeslots': 3,
    }


# ============== Main Entry Points ==============

def main_staged(run_id: str = None, resume_from: str = None):
    """
    Main entry point for staged solving.
    
    Args:
        run_id: Identifier for this run (auto-generated if None)
        resume_from: Stage name to resume from
    """
    print("="*60)
    print("HOCKEY DRAW SCHEDULER - STAGED SOLVING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = load_data()
    print(f"  {len(data['teams'])} teams")
    print(f"  {len(data['grades'])} grades")
    print(f"  {len(data['timeslots'])} timeslots")
    
    # Initialize solver
    checkpoint_manager = CheckpointManager()
    solver = StagedScheduleSolver(data, checkpoint_manager)
    
    # Initialize model
    unavailability_path = os.path.join('data', '2025', 'noplay')
    solver.initialize_model(unavailability_path)
    
    # Run staged solve
    solution = solver.run_staged_solve(run_id=run_id, resume_from=resume_from)
    
    if solution:
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
        print(f"Schedule exported to {output_path}")
        
        # ============== NEW: Save pliable format and generate analytics ==============
        from draw_analytics import DrawStorage, DrawAnalytics
        from draw_tester import DrawTester
        
        # Save in pliable JSON format
        draw = DrawStorage.from_X_solution(solution, description=f"Season Draw {timestamp}")
        json_path = output_dir / f'draw_{timestamp}.json'
        draw.save(str(json_path))
        print(f"Draw saved to {json_path}")
        
        # Generate comprehensive analytics
        analytics = DrawAnalytics(draw, data)
        analytics_path = output_dir / f'analytics_{timestamp}.xlsx'
        analytics.export_analytics_to_excel(str(analytics_path))
        print(f"Analytics exported to {analytics_path}")
        
        # Run violation check
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()
        print(f"\nViolation Check: {report.summary()}")
        
        if report.has_violations:
            report_path = output_dir / f'violations_{timestamp}.txt'
            with open(report_path, 'w') as f:
                f.write(report.full_report())
            print(f"Violation report saved to {report_path}")
        
        return solution, data
    else:
        print("\nNo valid solution found.")
        return None, data


def main_simple():
    """
    Simple (non-staged) main entry point for backwards compatibility.
    Uses all constraints in a single solve.
    """
    print("="*60)
    print("HOCKEY DRAW SCHEDULER - SINGLE SOLVE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = load_data()
    
    # Create model
    model = cp_model.CpModel()
    
    # Initialize variables
    unavailability_path = os.path.join('data', '2025', 'noplay')
    from main_notebook_translation import generate_X as generate_X_full
    
    X, Y, conflicts, unavailable_games = generate_X_full(unavailability_path, model, data)
    X.update(Y)
    
    data['unavailable_games'] = unavailable_games
    data['team_conflicts'] = conflicts
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    
    print(f"  {len(X)} decision variables")
    
    # Apply all constraints
    print("\nApplying constraints...")
    
    all_constraints = [
        cls() for stage in STAGES.values() 
        for cls in stage['constraints']
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
    
    status = solver.Solve(model)
    
    print(f"\nStatus: {solver.StatusName()}")
    
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
