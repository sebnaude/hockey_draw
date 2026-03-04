"""Quick test for ALL AI constraints."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ortools.sat.python import cp_model
from main_staged import load_data
from utils import generate_X
from constraints_ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    FiftyFiftyHomeandAwayAI,
    PHLAndSecondGradeAdjacencyAI,
    ClubGradeAdjacencyConstraintAI,
    MaxMaitlandHomeWeekendsAI,
    ClubDayConstraintAI,
    EqualMatchUpSpacingConstraintAI,
    PHLAndSecondGradeTimesAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    MaximiseClubsPerTimeslotBroadmeadowAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
    ClubVsClubAlignmentAI,
    PreferredTimesConstraintAI,
    EnsureBestTimeslotChoicesAI,
    TeamConflictConstraintAI,
)

ALL_CONSTRAINTS = [
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    FiftyFiftyHomeandAwayAI,
    PHLAndSecondGradeAdjacencyAI,
    ClubGradeAdjacencyConstraintAI,
    MaxMaitlandHomeWeekendsAI,
    ClubDayConstraintAI,
    EqualMatchUpSpacingConstraintAI,
    PHLAndSecondGradeTimesAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    MaximiseClubsPerTimeslotBroadmeadowAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
    ClubVsClubAlignmentAI,
    PreferredTimesConstraintAI,
    EnsureBestTimeslotChoicesAI,
    TeamConflictConstraintAI,
]

def main():
    print("Loading data...", flush=True)
    data = load_data()
    print(f"  Teams: {len(data['teams'])}, Timeslots: {len(data['timeslots'])}", flush=True)
    
    # Fresh model
    model = cp_model.CpModel()
    
    # Generate X
    X, Y, conflicts, unavailable_games = generate_X(model, data)
    
    # Prepare data
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    data['unavailable_games'] = unavailable_games
    data['team_conflicts'] = conflicts
    
    print(f"\nApplying all {len(ALL_CONSTRAINTS)} constraints...", flush=True)
    total = 0
    for i, cls in enumerate(ALL_CONSTRAINTS, 1):
        c = cls()
        count = c.apply(model, X, data)
        total += count
        print(f"  {i:2}. {cls.__name__}: +{count} ({total} total)", flush=True)
    
    print("\nSolving (5 second timeout)...", flush=True)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_workers = 2
    
    status = solver.Solve(model)
    status_name = solver.status_name(status)
    
    if status == cp_model.INFEASIBLE:
        print(f"\n[FAIL] INFEASIBLE - Model is broken!", flush=True)
    elif status == cp_model.MODEL_INVALID:
        print(f"\n[FAIL] MODEL_INVALID - Check constraints!", flush=True)
    else:
        print(f"\n[OK] {status_name} - All AI constraints valid!", flush=True)
        print(f"     Total constraints: {total}", flush=True)


if __name__ == "__main__":
    main()
