"""
Binary search for infeasible constraint with locked week 1.

Usage:
  python scripts/diagnose_constraints.py --set A   # First half
  python scripts/diagnose_constraints.py --set B   # Second half
  python scripts/diagnose_constraints.py --only 0,1,2  # Specific constraints
"""
import sys
import argparse
sys.path.insert(0, '.')

from ortools.sat.python import cp_model
from main_staged import load_data
from utils import generate_X
from analytics.storage import DrawStorage

# Severity 1 constraints in order
from constraints.original import (
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    PHLAndSecondGradeAdjacency,
    PHLAndSecondGradeTimes,
    FiftyFiftyHomeandAway,
    MaxMaitlandHomeWeekends,
    MaitlandHomeGrouping,
)

ALL_CONSTRAINTS = [
    ('NoDoubleBookingTeams', NoDoubleBookingTeamsConstraint),
    ('NoDoubleBookingFields', NoDoubleBookingFieldsConstraint),
    ('EnsureEqualGamesAndBalanceMatchUps', EnsureEqualGamesAndBalanceMatchUps),
    ('PHLAndSecondGradeAdjacency', PHLAndSecondGradeAdjacency),
    ('PHLAndSecondGradeTimes', PHLAndSecondGradeTimes),
    ('FiftyFiftyHomeandAway', FiftyFiftyHomeandAway),
    ('MaxMaitlandHomeWeekends', MaxMaitlandHomeWeekends),
    ('MaitlandHomeGrouping', MaitlandHomeGrouping),
]

def test_constraints(constraint_indices: list, lock_week1: bool = True, timeout: int = 30, slack: int = 0):
    """Test feasibility with specific constraints."""
    print(f"\n{'='*60}")
    print(f"Testing constraints: {constraint_indices}")
    print(f"Lock week 1: {lock_week1}, Slack: {slack}")
    print(f"{'='*60}")
    
    # Load data
    data = load_data(2026)
    
    # Set locked_weeks if locking week 1
    if lock_week1:
        data['locked_weeks'] = {1}
        print("Set data['locked_weeks'] = {1}")
    
    # Set slack if specified
    if slack > 0:
        data['constraint_slack'] = {
            'EqualMatchUpSpacingConstraint': slack,
            'AwayAtMaitlandGrouping': slack,
            'MaitlandHomeGrouping': slack,
            'ClubVsClubAlignment': slack,
            'MaximiseClubsPerTimeslotBroadmeadow': slack,
            'MinimiseClubsOnAFieldBroadmeadow': slack,
        }
        print(f"Set constraint_slack for all applicable constraints: {slack}")
    
    # Build model
    model = cp_model.CpModel()
    result = generate_X(model, data)
    X = result[0]
    print(f"Generated {len(X)} variables")
    
    # Lock week 1 games
    locked_count = 0
    if lock_week1:
        draw = DrawStorage.load('draws/2026/draw_v4.0.json')
        for game in draw.games:
            if game.week == 1:
                key = game.to_key()
                if key in X:
                    model.Add(X[key] == 1)
                    locked_count += 1
        print(f"Locked {locked_count} week 1 games")
    
    # Apply selected constraints
    print(f"\nApplying constraints:")
    for i in constraint_indices:
        name, cls = ALL_CONSTRAINTS[i]
        print(f"  [{i}] {name}")
        constraint = cls()
        constraint.apply(model, X, data)
    
    # Solve with short timeout
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = 4
    solver.parameters.log_search_progress = False
    
    print(f"\nSolving (timeout={timeout}s)...")
    status = solver.Solve(model)
    status_name = solver.status_name(status)
    
    print(f"\nResult: {status_name}")
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("✅ FEASIBLE - these constraints are OK")
    elif status == cp_model.INFEASIBLE:
        print("❌ INFEASIBLE - problem is in these constraints")
    else:
        print(f"⚠️ {status_name} - may need more time")
    
    return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', choices=['A', 'B', 'all'], help='A=first half, B=second half, all=all')
    parser.add_argument('--only', help='Comma-separated constraint indices (0-7)')
    parser.add_argument('--exclude', help='Comma-separated constraint indices to exclude')
    parser.add_argument('--no-lock', action='store_true', help='Do not lock week 1')
    parser.add_argument('--timeout', type=int, default=30, help='Solver timeout in seconds')
    parser.add_argument('--slack', type=int, default=0, help='Slack value for constraints')
    args = parser.parse_args()
    
    print("Available constraints:")
    for i, (name, _) in enumerate(ALL_CONSTRAINTS):
        print(f"  [{i}] {name}")
    
    # Determine which constraints to test
    if args.only:
        indices = [int(x) for x in args.only.split(',')]
    elif args.set == 'A':
        indices = [0, 1, 2, 3]  # First half
    elif args.set == 'B':
        indices = [4, 5, 6, 7]  # Second half
    elif args.set == 'all':
        indices = list(range(8))
    else:
        # Default: test incrementally
        print("\nNo set specified. Running incremental test...")
        for i in range(len(ALL_CONSTRAINTS)):
            status = test_constraints([i], lock_week1=not args.no_lock, timeout=args.timeout, slack=args.slack)
            if status == cp_model.INFEASIBLE:
                print(f"\n🎯 FOUND: Constraint {i} ({ALL_CONSTRAINTS[i][0]}) alone is infeasible!")
                return
        print("\nAll individual constraints are feasible. Testing combinations...")
        return
    
    if args.exclude:
        exclude = [int(x) for x in args.exclude.split(',')]
        indices = [i for i in indices if i not in exclude]
    
    test_constraints(indices, lock_week1=not args.no_lock, timeout=args.timeout, slack=args.slack)

if __name__ == '__main__':
    main()
