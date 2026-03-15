#!/usr/bin/env python
"""
Verify draw feasibility with the solver's constraint propagation.

This script answers the question: "Is this draw feasible under all constraints?"

Unlike `run.py test` which does post-hoc statistical violation checking, this script:
1. Loads the draw as locked games
2. Builds a fresh model with ALL constraints
3. Runs the solver's presolve to verify constraint propagation
4. Reports whether the combined constraints + locked games are satisfiable

Usage:
    python scripts/verify_draw_feasibility.py draws/draw.json --year 2026
    python scripts/verify_draw_feasibility.py draws/draw.json --year 2026 --lock-weeks 5
    python scripts/verify_draw_feasibility.py draws/draw.json --year 2026 --timeout 30
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ortools.sat.python import cp_model
from analytics.storage import DrawStorage
from config import load_season_data
from utils import generate_X
from main_staged import STAGES, STAGES_SEVERITY


def verify_draw_feasibility(
    draw_path: str,
    year: int,
    lock_weeks: int = None,
    timeout: float = 30.0,
    use_severity_stages: bool = False,
    verbose: bool = True
) -> tuple:
    """
    Verify if a draw is feasible under all constraints.
    
    Args:
        draw_path: Path to draw JSON file
        year: Season year
        lock_weeks: Only lock games up to this week (default: all weeks)
        timeout: Solver timeout in seconds (default: 30)
        use_severity_stages: Use severity-based stages instead of default
        verbose: Print detailed output
        
    Returns:
        Tuple of (is_feasible, status_name, message)
    """
    if verbose:
        print("="*60)
        print("DRAW FEASIBILITY VERIFICATION")
        print("="*60)
    
    # Load draw
    if verbose:
        print(f"\n[1/5] Loading draw from {draw_path}...")
    
    full_draw = DrawStorage.load(draw_path)
    
    if lock_weeks is not None:
        _, locked_keys = DrawStorage.load_and_lock(draw_path, lock_weeks)
        if verbose:
            print(f"  Locking {len(locked_keys)} games from weeks 1-{lock_weeks}")
    else:
        locked_keys = [game.to_key() for game in full_draw.games]
        if verbose:
            print(f"  Locking ALL {len(locked_keys)} games")
    
    # Load season data
    if verbose:
        print(f"\n[2/5] Loading season data for {year}...")
    
    data = load_season_data(year)
    
    if verbose:
        print(f"  Teams: {len(data['teams'])}")
        print(f"  Grades: {len(data['grades'])}")
        print(f"  Timeslots: {len(data['timeslots'])}")
    
    # Build model
    if verbose:
        print(f"\n[3/5] Building model and generating variables...")
    
    model = cp_model.CpModel()
    X, Y, conflicts, unavailable_games = generate_X(model, data)
    X.update(Y)
    
    data['unavailable_games'] = unavailable_games
    data['team_conflicts'] = conflicts
    data['games'] = list(data['games'].keys()) if isinstance(data['games'], dict) else data['games']
    
    if verbose:
        print(f"  Created {len(X)} decision variables")
    
    # Lock games
    if verbose:
        print(f"\n[4/5] Locking {len(locked_keys)} games...")
    
    locked_count = 0
    not_found = []
    
    for key in locked_keys:
        if key in X:
            model.Add(X[key] == 1)
            locked_count += 1
        else:
            not_found.append(key)
    
    if verbose:
        print(f"  Locked: {locked_count}")
        if not_found:
            print(f"  NOT FOUND in model: {len(not_found)}")
            if len(not_found) <= 5:
                for k in not_found:
                    print(f"    - {k[0]} vs {k[1]} ({k[2]}) @ {k[9]} {k[3]} {k[5]}")
            else:
                for k in not_found[:3]:
                    print(f"    - {k[0]} vs {k[1]} ({k[2]}) @ {k[9]} {k[3]} {k[5]}")
                print(f"    ... and {len(not_found) - 3} more")
    
    # If any locked keys weren't found, report but continue
    if not_found:
        print(f"\n⚠️ WARNING: {len(not_found)} locked games have no matching variable!")
        print("   This could mean the game is impossible (wrong time/venue for grade).")
    
    # Apply ALL constraints
    if verbose:
        print(f"\n[5/5] Applying constraints...")
    
    stage_defs = STAGES_SEVERITY if use_severity_stages else STAGES
    
    all_constraint_classes = []
    for stage_config in stage_defs.values():
        all_constraint_classes.extend(stage_config['constraints'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_constraints = []
    for c in all_constraint_classes:
        if c not in seen:
            seen.add(c)
            unique_constraints.append(c)
    
    if verbose:
        print(f"  Applying {len(unique_constraints)} unique constraints...")
    
    for constraint_class in unique_constraints:
        constraint = constraint_class()
        constraint.apply(model, X, data)
    
    if verbose:
        print(f"  Total model constraints: {len(model.Proto().constraints)}")
    
    # Solve with short timeout (we just want feasibility check)
    if verbose:
        print(f"\n[*] Running solver (timeout: {timeout}s)...")
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = 1
    solver.parameters.log_search_progress = False
    
    status = solver.Solve(model)
    status_name = solver.status_name(status)
    
    # Interpret result
    if status == cp_model.INFEASIBLE:
        is_feasible = False
        message = "INFEASIBLE: The locked games violate constraints and cannot form a valid schedule."
    elif status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        is_feasible = True
        message = "FEASIBLE: The locked games satisfy all constraints."
    elif status == cp_model.UNKNOWN:
        # Solver didn't prove infeasibility in time limit - treat as probably feasible
        is_feasible = None  # Unknown
        message = "UNKNOWN: Solver timed out. Increase timeout to verify (but probably feasible)."
    elif status == cp_model.MODEL_INVALID:
        is_feasible = False
        message = "MODEL_INVALID: There's an error in the constraint model."
    else:
        is_feasible = None
        message = f"Unexpected status: {status_name}"
    
    if verbose:
        print(f"\n{'='*60}")
        print("RESULT")
        print("="*60)
        
        if is_feasible is True:
            print("✅ FEASIBLE")
        elif is_feasible is False:
            print("❌ INFEASIBLE")
        else:
            print("⚠️ UNKNOWN (timeout)")
        
        print(f"\nStatus: {status_name}")
        print(f"Message: {message}")
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            scheduled = sum(1 for k, v in X.items() if solver.Value(v) == 1)
            print(f"Total games in solution: {scheduled}")
    
    return is_feasible, status_name, message


def main():
    parser = argparse.ArgumentParser(
        description='Verify draw feasibility with solver constraint propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Verify entire draw is feasible
    python scripts/verify_draw_feasibility.py draws/draw.json --year 2026
    
    # Verify only locked weeks (useful for partial draws)
    python scripts/verify_draw_feasibility.py draws/draw.json --year 2026 --lock-weeks 5
    
    # Increase timeout for complex draws
    python scripts/verify_draw_feasibility.py draws/draw.json --year 2026 --timeout 60
        """
    )
    
    parser.add_argument('draw_file', help='Path to draw JSON file')
    parser.add_argument('--year', type=int, required=True, help='Season year')
    parser.add_argument('--lock-weeks', type=int, default=None,
                        help='Only verify games up to this week (default: all)')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Solver timeout in seconds (default: 30)')
    parser.add_argument('--severity', action='store_true',
                        help='Use severity-based stages instead of default')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    is_feasible, status, message = verify_draw_feasibility(
        args.draw_file,
        args.year,
        lock_weeks=args.lock_weeks,
        timeout=args.timeout,
        use_severity_stages=args.severity,
        verbose=not args.quiet
    )
    
    # Exit code: 0 if feasible, 1 if infeasible, 2 if unknown
    if is_feasible is True:
        sys.exit(0)
    elif is_feasible is False:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
