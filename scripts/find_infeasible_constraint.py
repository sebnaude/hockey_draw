"""Binary search to find which AI constraint combination causes INFEASIBLE."""
import os
import sys
# Add parent directory to path for imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ortools.sat.python import cp_model
from main_staged import load_data
from utils import generate_X
from constraints.ai import (
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

# Order matches STAGES_AI
CONSTRAINT_CLASSES = [
    # Stage 1 - Required
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    FiftyFiftyHomeandAwayAI,
    # Stage 2 - Strong
    PHLAndSecondGradeAdjacencyAI,
    ClubGradeAdjacencyConstraintAI,
    MaxMaitlandHomeWeekendsAI,
    # Stage 3 - Medium
    ClubDayConstraintAI,
    EqualMatchUpSpacingConstraintAI,
    # Stage 4 - Soft
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

def test_constraints(constraint_list, data_template):
    """Test if a list of constraints is feasible."""
    # Fresh model each time
    model = cp_model.CpModel()
    
    # Fresh data copy
    data = dict(data_template)
    
    # Generate X
    X, Y, conflicts, unavailable_games = generate_X(model, data)
    
    # Prepare data
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    data['unavailable_games'] = unavailable_games
    data['team_conflicts'] = conflicts
    
    # Apply constraints
    total_constraints = 0
    for cls in constraint_list:
        constraint = cls()
        count = constraint.apply(model, X, data)
        total_constraints += count
    
    # Quick solve to check feasibility (just presolve)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2  # Very short - INFEASIBLE detected in presolve
    solver.parameters.num_workers = 1
    
    status = solver.Solve(model)
    return status, solver.status_name(status), total_constraints


def main():
    import sys
    print("Loading data...")
    sys.stdout.flush()
    data = load_data()
    print(f"  Teams: {len(data['teams'])}, Timeslots: {len(data['timeslots'])}")
    sys.stdout.flush()
    
    print("\n" + "="*60)
    print("INCREMENTAL CONSTRAINT TESTING")
    print("="*60)
    sys.stdout.flush()
    
    for i in range(1, len(CONSTRAINT_CLASSES) + 1):
        constraints_to_test = CONSTRAINT_CLASSES[:i]
        names = [c.__name__ for c in constraints_to_test]
        
        print(f"\n[{i}/{len(CONSTRAINT_CLASSES)}] Testing: +{names[-1]}")
        sys.stdout.flush()
        
        status, status_name, count = test_constraints(constraints_to_test, data)
        
        if status == cp_model.INFEASIBLE:
            print(f"  [FAIL] INFEASIBLE at {names[-1]} ({count} total constraints)")
            print(f"\n  CULPRIT FOUND: {names[-1]}")
            print(f"  Previous working set: {names[:-1]}")
            sys.stdout.flush()
            return
        elif status == cp_model.MODEL_INVALID:
            print(f"  [WARN] MODEL_INVALID at {names[-1]}")
            sys.stdout.flush()
            return
        else:
            print(f"  [OK] {status_name} ({count} constraints)")
            sys.stdout.flush()
    
    print("\n" + "="*60)
    print("All constraints passed! No INFEASIBLE found.")
    print("="*60)


if __name__ == "__main__":
    main()
