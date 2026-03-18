# severity_relaxation.py
"""
Severity-based Constraint Relaxation for Hockey Draw Scheduler.

This module provides infrastructure to:
1. Test feasibility by dropping constraints by severity group (4 → 3 → 2)
2. Identify which severity group causes infeasibility
3. Relax all slack variables in that group and retry
4. Always solve with ALL constraints together (no locked-in partial solutions)

Usage:
    # Add --relax flag to any generate command
    python run.py generate --year 2025 --relax
    python run.py generate --year 2025 --stages stage1_required --relax
    python run.py generate --year 2025 --simple --relax

The resolver:
1. Tries solving with all constraints
2. If INFEASIBLE, drops severity group 4 constraints and retries
3. If still INFEASIBLE, drops severity group 3 and retries
4. If still INFEASIBLE, drops severity group 2 and retries
5. Once feasible, the last-dropped group is the problem group
6. Relaxes all constraints in that group (slack +1) and solves with ALL constraints
"""

import sys
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from ortools.sat.python import cp_model


# ============== Severity Level Definitions ==============

# Mapping constraint classes to severity levels
# Level 1: CRITICAL - never relaxed
# Level 2: HIGH - structural constraints
# Level 3: MEDIUM - spacing/alignment
# Level 4: LOW - club density/optimization
# Level 5: VERY LOW - timeslot preferences

CONSTRAINT_TO_SEVERITY = {
    # Level 1 - CRITICAL (never dropped/relaxed)
    'NoDoubleBookingTeamsConstraint': 1,
    'NoDoubleBookingTeamsConstraintAI': 1,
    'NoDoubleBookingFieldsConstraint': 1,
    'NoDoubleBookingFieldsConstraintAI': 1,
    'EnsureEqualGamesAndBalanceMatchUps': 1,
    'EnsureEqualGamesAndBalanceMatchUpsAI': 1,
    'PHLAndSecondGradeAdjacency': 1,
    'PHLAndSecondGradeAdjacencyAI': 1,
    'PHLAndSecondGradeTimes': 1,
    'PHLAndSecondGradeTimesAI': 1,
    'FiftyFiftyHomeandAway': 1,
    'FiftyFiftyHomeandAwayAI': 1,
    'MaxMaitlandHomeWeekends': 1,
    'MaxMaitlandHomeWeekendsAI': 1,
    'MaitlandHomeGrouping': 1,
    'MaitlandHomeGroupingAI': 1,
    
    # Level 2 - HIGH (structural, club-specific)
    'ClubDayConstraint': 2,
    'ClubDayConstraintAI': 2,
    'AwayAtMaitlandGrouping': 2,
    'AwayAtMaitlandGroupingAI': 2,
    'TeamConflictConstraint': 2,
    'TeamConflictConstraintAI': 2,
    'EqualMatchUpSpacingConstraint': 2,
    'EqualMatchUpSpacingConstraintAI': 2,
    
    # Level 3 - MEDIUM (spacing, alignment)
    'ClubGradeAdjacencyConstraint': 3,
    'ClubGradeAdjacencyConstraintAI': 3,
    'ClubVsClubAlignment': 3,
    'ClubVsClubAlignmentAI': 3,
    
    # Level 4 - LOW (club density/optimization)
    'ClubGameSpread': 4,
    'ClubGameSpreadAI': 4,
    'MaximiseClubsPerTimeslotBroadmeadow': 4,
    'MaximiseClubsPerTimeslotBroadmeadowAI': 4,
    'MinimiseClubsOnAFieldBroadmeadow': 4,
    'MinimiseClubsOnAFieldBroadmeadowAI': 4,
    
    # Level 5 - VERY LOW (timeslot preferences)
    'EnsureBestTimeslotChoices': 5,
    'EnsureBestTimeslotChoicesAI': 5,
    'PreferredTimesConstraint': 5,
    'PreferredTimesConstraintAI': 5,
}


def get_severity_level(constraint_cls) -> int:
    """Get severity level for a constraint class."""
    name = constraint_cls.__name__
    return CONSTRAINT_TO_SEVERITY.get(name, 5)  # Default to lowest severity


def group_constraints_by_severity(constraints: list) -> Dict[int, list]:
    """
    Group a list of constraint classes by their severity level.
    
    Returns:
        Dict mapping severity level (1-5) to list of constraint classes
    """
    groups = defaultdict(list)
    for constraint_cls in constraints:
        level = get_severity_level(constraint_cls)
        groups[level].append(constraint_cls)
    return dict(groups)


# ============== Slack State Tracking ==============

@dataclass
class SeverityGroupState:
    """Tracks relaxation state for a severity group."""
    level: int
    constraint_classes: List[Any]
    current_slack: int = 0  # 0 = original, 1+ = relaxed
    max_slack: int = 3
    is_problem_group: bool = False
    
    def can_relax(self) -> bool:
        """Check if this group can be relaxed further."""
        return self.level > 1 and self.current_slack < self.max_slack
    
    def relax(self) -> bool:
        """Increase slack by 1. Returns True if successful."""
        if not self.can_relax():
            return False
        self.current_slack += 1
        return True


# ============== Severity Group Resolver ==============

class SeverityGroupResolver:
    """
    Resolves infeasibility by testing and relaxing severity groups.
    
    Key principle: Never lock in partial solutions. Always solve with
    all applicable constraints together.
    """
    
    def __init__(self, constraint_classes: list, verbose: bool = True):
        """
        Initialize resolver with constraint classes to use.
        
        Args:
            constraint_classes: List of constraint classes from the stage(s) being run
            verbose: Print progress information
        """
        self.all_constraints = constraint_classes
        self.verbose = verbose
        
        # Group constraints by severity
        self.severity_groups: Dict[int, SeverityGroupState] = {}
        grouped = group_constraints_by_severity(constraint_classes)
        
        for level, classes in grouped.items():
            self.severity_groups[level] = SeverityGroupState(
                level=level,
                constraint_classes=classes,
            )
        
        # Track which levels are present
        self.levels_present = sorted(self.severity_groups.keys())
        self._log(f"Constraints grouped by severity: {[f'L{lvl}:{len(self.severity_groups[lvl].constraint_classes)}' for lvl in self.levels_present]}")
    
    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[RELAX] {message}")
            sys.stdout.flush()
    
    def get_constraints_for_test(self, exclude_levels: Set[int] = None) -> list:
        """
        Get constraint classes to test, excluding specified severity levels.
        
        Args:
            exclude_levels: Set of severity levels to exclude (e.g., {5, 4})
            
        Returns:
            List of constraint classes to apply
        """
        exclude_levels = exclude_levels or set()
        result = []
        for level, group in self.severity_groups.items():
            if level not in exclude_levels:
                result.extend(group.constraint_classes)
        return result
    
    def get_soft_constraints_for_group(self, level: int, slack_level: int) -> list:
        """
        Get soft constraint instances for a severity group.
        
        Args:
            level: Severity level (2, 3, or 4)
            slack_level: Slack level to use (0=tight, 1=normal, 2=relaxed, etc.)
            
        Returns:
            List of soft constraint INSTANCES (not classes)
        """
        from constraints.soft import get_soft_constraint
        
        if level not in self.severity_groups:
            return []
        
        group = self.severity_groups[level]
        soft_instances = []
        
        for constraint_cls in group.constraint_classes:
            soft = get_soft_constraint(constraint_cls.__name__, slack_level)
            if soft:
                soft_instances.append(soft)
            else:
                # No soft version available, use hard version
                self._log(f"  No soft version for {constraint_cls.__name__}, using hard")
                soft_instances.append(constraint_cls)
        
        return soft_instances
    
    def find_problem_severity_group(self, 
                                     test_func,
                                     timeout: float = 10.0) -> Optional[int]:
        """
        Find which severity group causes infeasibility by progressive exclusion.
        
        Process:
        1. Test with all constraints → if feasible, no problem
        2. If infeasible, exclude level 4 and test
        3. If still infeasible, exclude level 3 and test
        4. If still infeasible, exclude level 2 and test
        5. The last-excluded level that makes it feasible is the problem group
        
        Args:
            test_func: Function(constraint_classes) -> (status, is_feasible)
                       Should create model, apply constraints, and return status
            timeout: Timeout per test in seconds
            
        Returns:
            Severity level that causes infeasibility, or None if all feasible
        """
        self._log("\n" + "="*60)
        self._log("FINDING PROBLEM SEVERITY GROUP")
        self._log("="*60)
        
        # Test with all constraints first
        all_constraints = self.get_constraints_for_test()
        self._log(f"\n[TEST] All constraints ({len(all_constraints)} total)")
        status, is_feasible = test_func(all_constraints, timeout)
        
        if is_feasible:
            self._log(f"  Result: {status} - ALL CONSTRAINTS FEASIBLE")
            return None
        
        self._log(f"  Result: {status} - INFEASIBLE, starting exclusion tests...")
        
        # Progressive exclusion: try excluding each level from lowest to highest
        # (Level 5 first, then 4, then 3, then 2; Level 1 is never excluded)
        excluded = set()
        problem_level = None
        
        for level in [5, 4, 3, 2]:
            if level not in self.severity_groups:
                self._log(f"\n[SKIP] Level {level} - no constraints at this level")
                continue
            
            excluded.add(level)
            test_constraints = self.get_constraints_for_test(exclude_levels=excluded)
            
            self._log(f"\n[TEST] Excluding levels {excluded} ({len(test_constraints)} constraints remain)")
            status, is_feasible = test_func(test_constraints, timeout)
            
            if is_feasible:
                # Found it! The last-added exclusion level is the problem
                problem_level = level
                self._log(f"  Result: {status} - FEASIBLE without level {level}")
                self._log(f"\n  >>> PROBLEM GROUP FOUND: Severity Level {level} <<<")
                break
            else:
                self._log(f"  Result: {status} - still INFEASIBLE")
        
        if problem_level is None and not is_feasible:
            # Even with levels 5,4,3,2 excluded, still infeasible
            # This means Level 1 constraints are mutually infeasible
            self._log("\n  >>> FATAL: Level 1 constraints alone are INFEASIBLE <<<")
            self._log("  Check your data configuration for conflicts.")
            return 1  # Return 1 to indicate Level 1 is the problem (can't be relaxed)
        
        if problem_level:
            self.severity_groups[problem_level].is_problem_group = True
        
        return problem_level
    
    def build_relaxed_constraint_set(self, problem_level: int) -> Tuple[list, list]:
        """
        Build a constraint set with the problem group relaxed.
        
        Returns:
            Tuple of (hard_constraint_classes, soft_constraint_instances)
            - Hard constraints: All levels except the problem level
            - Soft constraints: Problem level with increased slack
        """
        group = self.severity_groups.get(problem_level)
        if not group:
            return self.all_constraints, []
        
        # Increase slack for problem group
        old_slack = group.current_slack
        if group.relax():
            self._log(f"Relaxed level {problem_level}: slack {old_slack} → {group.current_slack}")
        else:
            self._log(f"Cannot relax level {problem_level} further (at max slack)")
        
        # Build constraint sets
        hard_constraints = []
        for level, grp in self.severity_groups.items():
            if level != problem_level:
                hard_constraints.extend(grp.constraint_classes)
        
        soft_constraints = self.get_soft_constraints_for_group(
            problem_level, 
            group.current_slack
        )
        
        self._log(f"Constraint set: {len(hard_constraints)} hard + {len(soft_constraints)} soft (relaxed)")
        
        return hard_constraints, soft_constraints
    
    def get_state_summary(self) -> str:
        """Get a summary of the current relaxation state."""
        lines = ["Severity Group States:"]
        for level in sorted(self.severity_groups.keys()):
            group = self.severity_groups[level]
            status = "PROBLEM" if group.is_problem_group else "OK"
            lines.append(f"  Level {level}: {len(group.constraint_classes)} constraints, "
                        f"slack={group.current_slack}, {status}")
        return "\n".join(lines)


# ============== Integration with Solver ==============

def create_relaxation_test_func(data: dict, generate_X_func, timeout: float = 10.0):
    """
    Create a test function for the resolver.
    
    Args:
        data: Season data dict
        generate_X_func: Function to generate decision variables
        timeout: Solve timeout in seconds
        
    Returns:
        Function(constraint_classes, timeout) -> (status_name, is_feasible)
    """
    def test_constraints(constraint_classes: list, test_timeout: float) -> Tuple[str, bool]:
        """Test if a set of constraints is feasible."""
        # Create fresh model
        model = cp_model.CpModel()
        
        # Fresh data copy
        test_data = dict(data)
        test_data['penalties'] = {}
        
        # Generate X
        X, Y, conflicts, unavailable_games = generate_X_func(model, test_data)
        
        # Prepare data
        if isinstance(test_data.get('games'), dict):
            test_data['games'] = list(test_data['games'].keys())
        test_data['unavailable_games'] = unavailable_games
        test_data['team_conflicts'] = conflicts
        
        # Apply constraints
        for constraint in constraint_classes:
            # Handle both classes and instances
            if isinstance(constraint, type):
                instance = constraint()
            else:
                instance = constraint
            
            try:
                instance.apply(model, X, test_data)
            except Exception as e:
                print(f"  [ERROR] {constraint}: {e}")
        
        # Quick solve for feasibility
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = test_timeout
        solver.parameters.num_workers = 2  # Minimal workers for speed
        
        status = solver.Solve(model)
        status_name = solver.status_name(status)
        
        is_feasible = status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        return status_name, is_feasible
    
    return test_constraints


def apply_constraints_with_relaxation(model, X, data, constraints: list, 
                                       relaxed_groups: Dict[int, int] = None):
    """
    Apply constraints to model, using soft versions for relaxed groups.
    
    Args:
        model: CP-SAT model
        X: Decision variables
        data: Season data dict
        constraints: List of constraint classes
        relaxed_groups: Dict mapping severity level -> slack level for relaxed groups
        
    Returns:
        Total constraints applied
    """
    from constraints.soft import get_soft_constraint
    
    relaxed_groups = relaxed_groups or {}
    total = 0
    
    for constraint_cls in constraints:
        level = get_severity_level(constraint_cls)
        
        if level in relaxed_groups:
            # Use soft version with specified slack
            slack = relaxed_groups[level]
            soft = get_soft_constraint(constraint_cls.__name__, slack)
            if soft:
                count = soft.apply(model, X, data)
                total += count if count else 0
                continue
        
        # Use hard version
        instance = constraint_cls()
        count = instance.apply(model, X, data)
        total += count if count else 0
    
    return total
