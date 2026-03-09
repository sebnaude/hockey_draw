# constraints/symmetry.py
"""
Symmetry breaking constraints for the scheduling system.

These constraints reduce the search space by eliminating equivalent solutions
that differ only in their ordering (e.g., which games appear in which round).
"""

from ortools.sat.python import cp_model
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class Constraint(ABC):
    """Abstract base class for all scheduling constraints."""
    
    @abstractmethod
    def apply(self, model: cp_model.CpModel, X: dict, data: dict):
        """Apply constraint to the OR-Tools model."""
        pass


class FixRound1SymmetryBreaking(Constraint):
    """
    Fix Round 1 pairings using the circle method to break symmetry.
    
    This constraint eliminates equivalent schedules that differ only in
    which games appear in which round. By fixing Round 1 pairings, we
    cut the search space dramatically (from ~21! equivalent orderings
    per grade to just 1).
    
    For each grade:
    - Computes circle method Round 1 pairings
    - For each pairing (team1, team2), constrains that exactly one game
      between them occurs in round_no=1 timeslots
    
    Byes (odd team counts) are handled by excluding ghost team pairings.
    """
    
    def apply(self, model: cp_model.CpModel, X: dict, data: dict) -> int:
        """
        Apply Round 1 symmetry breaking constraints.
        
        Args:
            model: CP-SAT model
            X: Decision variables dict
            data: Problem data including teams, timeslots, grades
            
        Returns:
            Number of constraints added
        """
        from utils import circle_method_round_1_pairings
        
        teams = data['teams']
        timeslots = data['timeslots']
        games = data['games']
        
        # Build teams by grade
        teams_by_grade: Dict[str, List[str]] = defaultdict(list)
        for team in teams:
            teams_by_grade[team.grade].append(team.name)
        
        # Get circle method pairings for Round 1
        round_1_pairings = circle_method_round_1_pairings(dict(teams_by_grade))
        
        # Find all round_no=1 timeslots
        round_1_timeslots = [t for t in timeslots if t.round_no == 1]
        
        if not round_1_timeslots:
            print("  WARNING: No Round 1 timeslots found - symmetry breaking skipped")
            return 0
        
        constraints_added = 0
        
        # For each grade, constrain Round 1 pairings
        for grade, pairings in round_1_pairings.items():
            if not pairings:
                continue
                
            print(f"  Grade {grade}: {len(pairings)} Round 1 pairings")
            
            for team1, team2 in pairings:
                # Find all X variables for this pairing in Round 1
                # Key: (t1, t2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
                round_1_vars = []
                
                for t in round_1_timeslots:
                    # Try both orderings since we don't know which is in X
                    key1 = (team1, team2, grade, t.day, t.day_slot, t.time, 
                           t.week, t.date, t.round_no, t.field.name, t.field.location)
                    key2 = (team2, team1, grade, t.day, t.day_slot, t.time,
                           t.week, t.date, t.round_no, t.field.name, t.field.location)
                    
                    if key1 in X:
                        round_1_vars.append(X[key1])
                    elif key2 in X:
                        round_1_vars.append(X[key2])
                
                if round_1_vars:
                    # Exactly one of these Round 1 timeslots must be used for this pairing
                    model.Add(sum(round_1_vars) == 1)
                    constraints_added += 1
                else:
                    # No valid Round 1 timeslots for this pairing - this shouldn't
                    # happen unless there's a constraint conflict
                    print(f"    WARNING: No Round 1 variables for {team1} vs {team2}")
        
        print(f"  Total: {constraints_added} symmetry breaking constraints added")
        return constraints_added


class FixRound1SymmetryBreakingAI(FixRound1SymmetryBreaking):
    """AI-enhanced alias for FixRound1SymmetryBreaking (identical implementation)."""
    pass
