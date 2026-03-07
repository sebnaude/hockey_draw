# infeasibility_resolver.py
"""
Infeasibility Resolver for Hockey Draw Scheduler.

This module provides infrastructure to:
1. Test ALL constraints together at once
2. When infeasible, identify which constraint(s) are the culprit using:
   - Constraint removal testing (remove each one and check if now feasible)
   - This finds constraints that are NECESSARY for infeasibility
3. Programmatically increase slack on the identified constraint
4. Retry until feasible

The solver stages (stage1, stage2) remain unchanged. This is a diagnostic/resolution
tool that works alongside the solver.

Usage:
    from infeasibility_resolver import InfeasibilityResolver, ConstraintSlackRegistry
    
    # Create resolver
    registry = ConstraintSlackRegistry()
    resolver = InfeasibilityResolver(data, registry)
    
    # Test all constraints together
    result = resolver.test_all_constraints(constraint_classes)
    
    if result.status == 'INFEASIBLE':
        # Find which constraint(s) are blocking
        culprits = resolver.find_blocking_constraints(constraint_classes)
        
        # Increase slack on the culprit and retry
        for culprit in culprits:
            registry.increase_slack(culprit)
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Type, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ortools.sat.python import cp_model

# Local imports
from utils import generate_X
from analytics.tester import CONSTRAINT_SEVERITY_LEVELS, SEVERITY_LEVEL_LABELS


# ============== Constraint Slack Registry ==============

@dataclass
class ConstraintState:
    """Tracks the current slack state of a constraint."""
    name: str
    severity_level: int
    current_slack: int = 1  # 0=tight, 1=normal, 2=relaxed, 3+=very relaxed
    max_slack: int = 3
    is_hard: bool = True  # True = using hard version, False = using soft version
    violation_count: int = 0  # Track how many times this caused issues
    
    def can_relax(self) -> bool:
        """Check if this constraint can be relaxed further."""
        # Level 1 constraints are never relaxed
        if self.severity_level == 1:
            return False
        return self.current_slack < self.max_slack
    
    def relax(self) -> bool:
        """
        Relax this constraint by one level.
        Returns True if relaxation was possible.
        """
        if not self.can_relax():
            return False
        
        if self.is_hard:
            # First relaxation: switch from hard to soft
            self.is_hard = False
            self.current_slack = 0  # Start with tight soft
        else:
            # Subsequent relaxations: increase slack
            self.current_slack += 1
        
        self.violation_count += 1
        return True


class ConstraintSlackRegistry:
    """
    Registry that tracks and manages slack levels for all constraints.
    """
    
    def __init__(self):
        """Initialize registry with all known constraints."""
        self.constraints: Dict[str, ConstraintState] = {}
        self._initialize_constraints()
    
    def _initialize_constraints(self):
        """Initialize constraint states from severity levels."""
        for name, level in CONSTRAINT_SEVERITY_LEVELS.items():
            self.constraints[name] = ConstraintState(
                name=name,
                severity_level=level,
                is_hard=True,
                current_slack=1,
            )
    
    def get_state(self, name: str) -> Optional[ConstraintState]:
        """Get the current state of a constraint."""
        return self.constraints.get(name)
    
    def increase_slack(self, name: str) -> bool:
        """Increase slack for a constraint. Returns True if successful."""
        state = self.constraints.get(name)
        if state is None:
            print(f"[WARNING] Unknown constraint: {name}")
            return False
        
        if state.relax():
            mode = "SOFT" if not state.is_hard else "HARD"
            print(f"[SLACK] {name}: relaxed to slack={state.current_slack} ({mode})")
            return True
        else:
            print(f"[SLACK] {name}: cannot relax further (level={state.severity_level})")
            return False
    
    def reset(self):
        """Reset all constraints to their default (hard) state."""
        for state in self.constraints.values():
            state.is_hard = True
            state.current_slack = 1
    
    def get_relaxed_constraints(self) -> List[str]:
        """Get list of constraints that have been relaxed from hard."""
        return [name for name, state in self.constraints.items() 
                if not state.is_hard or state.current_slack > 1]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current registry state."""
        return {
            'total_constraints': len(self.constraints),
            'relaxed_count': len(self.get_relaxed_constraints()),
            'constraints_by_level': {
                level: [name for name, s in self.constraints.items() 
                        if s.severity_level == level]
                for level in range(1, 5)
            },
            'current_states': {
                name: {
                    'level': s.severity_level,
                    'is_hard': s.is_hard,
                    'slack': s.current_slack,
                    'violations': s.violation_count,
                }
                for name, s in self.constraints.items()
            }
        }
    
    def get_constraint_instance(self, name: str, constraint_cls):
        """Get appropriate constraint instance based on current state."""
        state = self.constraints.get(name)
        if state is None or state.is_hard:
            return constraint_cls
        else:
            return self._get_soft_constraint(name, state.current_slack)
    
    def _get_soft_constraint(self, name: str, slack_level: int):
        """Get a soft constraint instance with specified slack."""
        try:
            from constraints_soft import (
                ClubDayConstraintSoft,
                AwayAtMaitlandGroupingSoft,
                TeamConflictConstraintSoft,
                EqualMatchUpSpacingConstraintSoft,
                ClubGradeAdjacencyConstraintSoft,
                ClubVsClubAlignmentSoft,
                EnsureBestTimeslotChoicesSoft,
                MaximiseClubsPerTimeslotBroadmeadowSoft,
                MinimiseClubsOnAFieldBroadmeadowSoft,
                PreferredTimesConstraintSoft,
            )
            
            mapping = {
                'ClubDayConstraint': ClubDayConstraintSoft,
                'AwayAtMaitlandGrouping': AwayAtMaitlandGroupingSoft,
                'TeamConflict': TeamConflictConstraintSoft,
                'EqualMatchUpSpacing': EqualMatchUpSpacingConstraintSoft,
                'ClubGradeAdjacency': ClubGradeAdjacencyConstraintSoft,
                'ClubVsClubAlignment': ClubVsClubAlignmentSoft,
                'EnsureBestTimeslotChoices': EnsureBestTimeslotChoicesSoft,
                'MaximiseClubsPerTimeslotBroadmeadow': MaximiseClubsPerTimeslotBroadmeadowSoft,
                'MinimiseClubsOnAFieldBroadmeadow': MinimiseClubsOnAFieldBroadmeadowSoft,
                'PreferredTimesConstraint': PreferredTimesConstraintSoft,
            }
            
            soft_cls = mapping.get(name)
            if soft_cls:
                return soft_cls(slack_level=slack_level)
        except ImportError:
            pass
        return None


# ============== Infeasibility Result ==============

@dataclass
class InfeasibilityResult:
    """Result of a feasibility test."""
    status: str  # 'FEASIBLE', 'OPTIMAL', 'INFEASIBLE', 'UNKNOWN', 'MODEL_INVALID'
    solve_time_seconds: float = 0.0
    constraint_counts: Dict[str, int] = field(default_factory=dict)
    total_constraints: int = 0
    blocking_constraints: List[str] = field(default_factory=list)
    
    @property
    def is_feasible(self) -> bool:
        return self.status in ('FEASIBLE', 'OPTIMAL')
    
    @property 
    def is_infeasible(self) -> bool:
        return self.status == 'INFEASIBLE'
    
    def __str__(self) -> str:
        if self.is_infeasible and self.blocking_constraints:
            return f"INFEASIBLE: blocked by {self.blocking_constraints}"
        return f"{self.status} ({self.total_constraints} constraints, {self.solve_time_seconds:.1f}s)"


# ============== Infeasibility Resolver ==============

class InfeasibilityResolver:
    """
    Identifies blocking constraints and manages iterative relaxation.
    
    Strategy:
    1. Test ALL constraints together
    2. If infeasible, identify culprit by removal testing
    3. Relax the identified constraint(s) and retry
    """
    
    def __init__(self, data: dict, registry: Optional[ConstraintSlackRegistry] = None,
                 timeout_per_test: float = 5.0, verbose: bool = True):
        self.data = data
        self.registry = registry or ConstraintSlackRegistry()
        self.timeout = timeout_per_test
        self.verbose = verbose
        self.test_history: List[InfeasibilityResult] = []
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
            sys.stdout.flush()
    
    def _create_model_and_vars(self) -> Tuple[cp_model.CpModel, dict, dict]:
        """Create a fresh model with decision variables."""
        model = cp_model.CpModel()
        test_data = dict(self.data)
        test_data['penalties'] = {}
        
        X, Y, conflicts, unavailable_games = generate_X(model, test_data)
        
        if isinstance(test_data.get('games'), dict):
            test_data['games'] = list(test_data['games'].keys())
        test_data['unavailable_games'] = unavailable_games
        test_data['team_conflicts'] = conflicts
        
        return model, X, test_data
    
    def test_constraints(self, constraint_classes: List[Type], 
                         names_map: Dict[Type, str] = None) -> InfeasibilityResult:
        """Test if a list of constraint classes produces a feasible model."""
        model, X, test_data = self._create_model_and_vars()
        
        constraint_counts = {}
        start_time = datetime.now()
        
        for cls in constraint_classes:
            name = names_map.get(cls, cls.__name__) if names_map else cls.__name__
            constraint = self.registry.get_constraint_instance(name, cls)
            
            if isinstance(constraint, type):
                instance = constraint()
            else:
                instance = constraint
            
            if instance is None:
                continue
            
            try:
                count = instance.apply(model, X, test_data)
                constraint_counts[name] = count if count else 0
            except Exception as e:
                self._log(f"  [ERROR] {name}: {e}")
                constraint_counts[name] = -1
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout
        solver.parameters.num_workers = 1
        
        status = solver.Solve(model)
        solve_time = (datetime.now() - start_time).total_seconds()
        
        result = InfeasibilityResult(
            status=solver.status_name(status),
            solve_time_seconds=solve_time,
            constraint_counts=constraint_counts,
            total_constraints=sum(c for c in constraint_counts.values() if c > 0),
        )
        
        self.test_history.append(result)
        return result
    
    def find_blocking_constraints(self, constraint_classes: List[Type],
                                   names_map: Dict[Type, str] = None) -> List[str]:
        """
        Find which constraint(s) cause infeasibility by removal testing.
        
        Tests ALL constraints together first. If infeasible, removes each
        constraint one at a time to find which removal(s) make it feasible.
        """
        self._log("\n" + "="*60)
        self._log("INFEASIBILITY ANALYSIS: Testing all constraints together")
        self._log("="*60)
        
        # Test all constraints together
        self._log(f"\n[1] Testing ALL {len(constraint_classes)} constraints together...")
        result = self.test_constraints(constraint_classes, names_map)
        
        self._log(f"    Status: {result.status}")
        self._log(f"    Total constraints: {result.total_constraints}")
        self._log(f"    Solve time: {result.solve_time_seconds:.1f}s")
        
        if not result.is_infeasible:
            self._log(f"\n[OK] Model is {result.status} - no blocking constraints")
            return []
        
        # Model is infeasible - find which constraint(s) are responsible
        self._log(f"\n[2] Model is INFEASIBLE. Testing constraint removal...")
        
        blocking = []
        
        for i, cls in enumerate(constraint_classes):
            name = names_map.get(cls, cls.__name__) if names_map else cls.__name__
            
            # Skip Level 1 constraints - they can't be relaxed anyway
            state = self.registry.get_state(name)
            level = state.severity_level if state else CONSTRAINT_SEVERITY_LEVELS.get(name, 4)
            
            if level == 1:
                self._log(f"    [{i+1}/{len(constraint_classes)}] {name}: SKIP (Level 1)")
                continue
            
            # Test without this constraint
            remaining = [c for c in constraint_classes if c != cls]
            self._log(f"    [{i+1}/{len(constraint_classes)}] Without {name}...", )
            
            test_result = self.test_constraints(remaining, names_map)
            
            if test_result.is_feasible or test_result.status == 'UNKNOWN':
                self._log(f" → {test_result.status} ✓ (blocking)")
                blocking.append(name)
            else:
                self._log(f" → still {test_result.status}")
        
        if blocking:
            self._log(f"\n[FOUND] Blocking constraints: {blocking}")
        else:
            self._log(f"\n[WARNING] No single constraint removal fixes infeasibility.")
        
        return blocking
    
    def resolve_iteratively(self, constraint_classes: List[Type],
                            names_map: Dict[Type, str] = None,
                            max_iterations: int = 10) -> Tuple[bool, List[str]]:
        """Iteratively find and relax blocking constraints until feasible."""
        self._log("\n" + "="*60)
        self._log("ITERATIVE RELAXATION")
        self._log("="*60)
        
        relaxed = []
        
        for iteration in range(1, max_iterations + 1):
            self._log(f"\n--- Iteration {iteration}/{max_iterations} ---")
            
            blocking = self.find_blocking_constraints(constraint_classes, names_map)
            
            if not blocking:
                result = self.test_constraints(constraint_classes, names_map)
                if result.is_feasible or result.status == 'UNKNOWN':
                    self._log(f"\n[SUCCESS] Model is {result.status} after {len(relaxed)} relaxations")
                    return True, relaxed
                else:
                    self._log(f"\n[FAILED] No relaxable constraints found but still {result.status}")
                    return False, relaxed
            
            # Sort by severity (prefer relaxing lower severity first: level 4 before 3)
            blocking_with_levels = [
                (name, self.registry.get_state(name).severity_level if self.registry.get_state(name) else 4)
                for name in blocking
            ]
            blocking_with_levels.sort(key=lambda x: -x[1])
            
            to_relax = blocking_with_levels[0][0]
            
            if self.registry.increase_slack(to_relax):
                relaxed.append(to_relax)
            else:
                self._log(f"\n[FAILED] Cannot relax '{to_relax}'")
                return False, relaxed
        
        self._log(f"\n[FAILED] Max iterations ({max_iterations}) reached")
        return False, relaxed
    
    def get_resolution_report(self) -> str:
        """Generate a report of the resolution process."""
        lines = [
            "=" * 60,
            "INFEASIBILITY RESOLUTION REPORT",
            "=" * 60,
            "",
            f"Total tests run: {len(self.test_history)}",
            f"Relaxed constraints: {self.registry.get_relaxed_constraints()}",
        ]
        
        summary = self.registry.get_summary()
        lines.append("")
        lines.append("Constraint States (modified only):")
        
        for name, state_info in summary['current_states'].items():
            if state_info['violations'] > 0 or not state_info['is_hard']:
                mode = "HARD" if state_info['is_hard'] else f"SOFT(slack={state_info['slack']})"
                lines.append(f"  {name}: L{state_info['level']}, {mode}, violations={state_info['violations']}")
        
        return "\n".join(lines)


# ============== Helper Functions ==============

CLASS_TO_REGISTRY = {
    'NoDoubleBookingTeamsConstraint': 'NoDoubleBookingTeams',
    'NoDoubleBookingTeamsConstraintAI': 'NoDoubleBookingTeams',
    'NoDoubleBookingFieldsConstraint': 'NoDoubleBookingFields',
    'NoDoubleBookingFieldsConstraintAI': 'NoDoubleBookingFields',
    'EnsureEqualGamesAndBalanceMatchUps': 'EqualGames',
    'EnsureEqualGamesAndBalanceMatchUpsAI': 'EqualGames',
    'PHLAndSecondGradeAdjacency': 'PHLAndSecondGradeAdjacency',
    'PHLAndSecondGradeAdjacencyAI': 'PHLAndSecondGradeAdjacency',
    'PHLAndSecondGradeTimes': 'PHLAndSecondGradeTimes',
    'PHLAndSecondGradeTimesAI': 'PHLAndSecondGradeTimes',
    'FiftyFiftyHomeandAway': 'FiftyFiftyHomeAway',
    'FiftyFiftyHomeandAwayAI': 'FiftyFiftyHomeAway',
    'MaxMaitlandHomeWeekends': 'MaxMaitlandHomeWeekends',
    'MaxMaitlandHomeWeekendsAI': 'MaxMaitlandHomeWeekends',
    'MaitlandHomeGrouping': 'MaitlandHomeGrouping',
    'MaitlandHomeGroupingAI': 'MaitlandHomeGrouping',
    'ClubDayConstraint': 'ClubDayConstraint',
    'ClubDayConstraintAI': 'ClubDayConstraint',
    'AwayAtMaitlandGrouping': 'AwayAtMaitlandGrouping',
    'AwayAtMaitlandGroupingAI': 'AwayAtMaitlandGrouping',
    'TeamConflictConstraint': 'TeamConflict',
    'TeamConflictConstraintAI': 'TeamConflict',
    'EqualMatchUpSpacingConstraint': 'EqualMatchUpSpacing',
    'EqualMatchUpSpacingConstraintAI': 'EqualMatchUpSpacing',
    'ClubGradeAdjacencyConstraint': 'ClubGradeAdjacency',
    'ClubGradeAdjacencyConstraintAI': 'ClubGradeAdjacency',
    'ClubVsClubAlignment': 'ClubVsClubAlignment',
    'ClubVsClubAlignmentAI': 'ClubVsClubAlignment',
    'EnsureBestTimeslotChoices': 'EnsureBestTimeslotChoices',
    'EnsureBestTimeslotChoicesAI': 'EnsureBestTimeslotChoices',
    'MaximiseClubsPerTimeslotBroadmeadow': 'MaximiseClubsPerTimeslotBroadmeadow',
    'MaximiseClubsPerTimeslotBroadmeadowAI': 'MaximiseClubsPerTimeslotBroadmeadow',
    'MinimiseClubsOnAFieldBroadmeadow': 'MinimiseClubsOnAFieldBroadmeadow',
    'MinimiseClubsOnAFieldBroadmeadowAI': 'MinimiseClubsOnAFieldBroadmeadow',
    'PreferredTimesConstraint': 'PreferredTimesConstraint',
    'PreferredTimesConstraintAI': 'PreferredTimesConstraint',
}


def build_names_map(constraint_classes: List[Type]) -> Dict[Type, str]:
    """Build a mapping from constraint classes to registry names."""
    return {cls: CLASS_TO_REGISTRY.get(cls.__name__, cls.__name__) 
            for cls in constraint_classes}


def get_stage_constraints(stage_name: str, use_ai: bool = False) -> List[Type]:
    """Get constraint classes for a specific stage."""
    from main_staged import STAGES, STAGES_AI
    stages = STAGES_AI if use_ai else STAGES
    
    if stage_name not in stages:
        raise ValueError(f"Unknown stage: {stage_name}. Available: {list(stages.keys())}")
    
    return stages[stage_name]['constraints']


def get_all_constraints(use_ai: bool = False) -> List[Type]:
    """Get all constraint classes from all stages."""
    from main_staged import STAGES, STAGES_AI
    stages = STAGES_AI if use_ai else STAGES
    
    all_constraints = []
    for stage_info in stages.values():
        all_constraints.extend(stage_info['constraints'])
    
    return all_constraints


# ============== CLI Entry Point ==============

def main():
    """Command-line interface for infeasibility resolution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and resolve constraint infeasibility")
    parser.add_argument('--year', type=int, required=True, help='Season year')
    parser.add_argument('--stage', type=str, default=None,
                        help='Stage to analyze (e.g., stage1_required). Default: all stages')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Timeout per test in seconds (default: 5)')
    parser.add_argument('--resolve', action='store_true',
                        help='Attempt iterative relaxation')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Max resolution iterations (default: 10)')
    parser.add_argument('--ai', action='store_true',
                        help='Use AI constraint implementations')
    
    args = parser.parse_args()
    
    from main_staged import load_data
    
    print(f"Loading {args.year} season data...")
    data = load_data(args.year)
    
    if args.stage:
        constraint_classes = get_stage_constraints(args.stage, args.ai)
        print(f"\nAnalyzing stage: {args.stage}")
    else:
        constraint_classes = get_all_constraints(args.ai)
        print(f"\nAnalyzing ALL constraints from all stages")
    
    print(f"Total constraints: {len(constraint_classes)}")
    
    names_map = build_names_map(constraint_classes)
    registry = ConstraintSlackRegistry()
    resolver = InfeasibilityResolver(data, registry, timeout_per_test=args.timeout)
    
    if args.resolve:
        success, relaxed = resolver.resolve_iteratively(
            constraint_classes, names_map, max_iterations=args.max_iterations
        )
        print("\n" + resolver.get_resolution_report())
        return 0 if success else 1
    else:
        blocking = resolver.find_blocking_constraints(constraint_classes, names_map)
        if blocking:
            print(f"\nBlocking constraints: {blocking}")
            print("Use --resolve to attempt automatic relaxation")
            return 1
        else:
            print("\nAll constraints are feasible together!")
            return 0


if __name__ == "__main__":
    sys.exit(main())
