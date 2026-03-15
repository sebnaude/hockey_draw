# constraints_soft.py
"""
Soft/Relaxed constraint classes for scheduling system.

These constraints are designed as fallback versions of the hard constraints
when the solver finds the problem infeasible. They introduce configurable
slack variables and penalties to allow the solver to find a solution while
minimizing constraint violations.

Severity Levels (from tester.py):
- Level 1: CRITICAL - Never softened (NoDoubleBooking, EqualGames, etc.)
- Level 2: HIGH - ClubDayConstraint, AwayAtMaitlandGrouping, TeamConflictConstraint
- Level 3: MEDIUM - EqualMatchUpSpacing, ClubGradeAdjacency, ClubVsClubAlignment
- Level 4: LOW - EnsureBestTimeslotChoices, MaximiseClubsPerTimeslot, etc.

Usage:
    # Instead of:
    constraint = ClubDayConstraint()
    
    # Use the soft version with configurable slack:
    constraint = ClubDayConstraintSoft(slack_level=1)  # 0=tight, 1=normal, 2=relaxed
    
    # Or with explicit parameters:
    constraint = TeamConflictConstraintSoft(
        max_violations_per_pair=1,  # Allow up to 1 conflict per pair
        violation_penalty_weight=50000
    )
"""
from ortools.sat.python import cp_model
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

from utils import (
    get_club, get_duplicated_graded_teams, get_teams_from_club, 
    get_club_from_clubname, get_nearest_week_by_date
)
from constraints.original import _normalize_preference_no_play


# ============== Base Classes ==============

class SoftConstraint(ABC):
    """
    Abstract base class for soft constraints.
    
    Soft constraints use penalty variables instead of hard bounds where possible,
    allowing the solver to find feasible solutions when hard constraints fail.
    """
    
    # Default penalty weights by severity level
    DEFAULT_WEIGHTS = {
        2: 100000,   # Level 2 - HIGH
        3: 50000,    # Level 3 - MEDIUM  
        4: 10000,    # Level 4 - LOW
    }
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None):
        """
        Initialize soft constraint.
        
        Args:
            slack_level: 0=tight (minimal slack), 1=normal, 2=relaxed (max slack)
            penalty_weight: Override default penalty weight for violations
        """
        self.slack_level = slack_level
        self._penalty_weight = penalty_weight
        self.severity_level = 4  # Default, override in subclasses
    
    @property
    def penalty_weight(self) -> int:
        """Get the penalty weight for this constraint."""
        if self._penalty_weight is not None:
            return self._penalty_weight
        return self.DEFAULT_WEIGHTS.get(self.severity_level, 10000)
    
    def get_slack_multiplier(self) -> float:
        """Get multiplier based on slack level."""
        multipliers = {0: 0.5, 1: 1.0, 2: 2.0}
        return multipliers.get(self.slack_level, 1.0)
    
    def _init_penalties(self, data: dict, name: str):
        """Initialize penalty tracking in data dict."""
        if 'penalties' not in data:
            data['penalties'] = {}
        if name not in data['penalties']:
            data['penalties'][name] = {
                'weight': self.penalty_weight,
                'penalties': []
            }
    
    @abstractmethod
    def apply(self, model: cp_model.CpModel, X: dict, data: dict):
        """Apply constraint to the OR-Tools model."""
        pass


# ============== Level 2 Soft Constraints ==============

class ClubDayConstraintSoft(SoftConstraint):
    """
    Soft version of ClubDayConstraint.
    
    Original hard constraints:
    - Every club team must play on club day
    - Intra-club matchups for teams in same grade
    - All games at same field
    - Contiguous timeslots
    
    Soft version:
    - Penalizes teams not playing (but allows it)
    - Penalizes missing intra-club matchups
    - Penalizes using multiple fields
    - Penalizes gaps in timeslots
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 allow_missed_teams: bool = True,
                 allow_multiple_fields: bool = False,
                 allow_gaps: bool = True):
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 2
        self.allow_missed_teams = allow_missed_teams
        self.allow_multiple_fields = allow_multiple_fields
        self.allow_gaps = allow_gaps
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'ClubDayConstraintSoft')
        
        club_days = data['club_days']
        teams = data['teams']
        clubs = data['clubs']
        locked_weeks = data.get('locked_weeks', set())
        
        allowed_keys = ['team1', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        
        for club_name in club_days:
            if club_name.lower() not in [c.name.lower() for c in clubs]:
                raise ValueError(f'Invalid team name {club_name} in ClubDay Dictionary')
            
            desired_date = club_days[club_name]
            closest_week = get_nearest_week_by_date(desired_date.strftime("%Y-%m-%d"), data['timeslots'])
            
            if closest_week in locked_weeks:
                continue
            
            club = get_club_from_clubname(club_name, data['clubs'])
            club_teams = get_teams_from_club(club_name, teams)
            
            # Locate all games for the club on the desired date
            club_games = [key for key in X if len(key) > 5 
                          and key[allowed_keys.index('date')] == desired_date.date().strftime('%Y-%m-%d')
                          and (key[allowed_keys.index('team1')] in club_teams 
                               or key[allowed_keys.index('team2')] in club_teams)]
            
            if not club_games:
                continue  # No games possible - skip instead of error
            
            teams_by_grade = {}
            for team in club_teams:
                grade = team.rsplit(' ', 1)[1]
                teams_by_grade.setdefault(grade, []).append(team)
            
            # SOFT: Every club team should play (penalize if not)
            if self.allow_missed_teams:
                for team in club_teams:
                    team_plays = model.NewBoolVar(f'team_plays_{club_name}_{team}')
                    team_games = [X[game_key] for game_key in club_games
                                  if team in [game_key[allowed_keys.index('team1')], 
                                            game_key[allowed_keys.index('team2')]]]
                    if team_games:
                        model.AddMaxEquality(team_plays, team_games)
                        
                        # Penalty if team doesn't play
                        penalty = model.NewIntVar(0, 1, f'penalty_team_missed_{club_name}_{team}')
                        model.Add(penalty == 1 - team_plays)
                        data['penalties']['ClubDayConstraintSoft']['penalties'].append(penalty)
            else:
                # Keep hard constraint
                for team in club_teams:
                    model.Add(sum(X[game_key] for game_key in club_games
                                  if team in [game_key[allowed_keys.index('team1')], 
                                            game_key[allowed_keys.index('team2')]]) >= 1)
            
            # SOFT: Intra-club matchups (penalize missing matchups)
            for grade, teams_in_grade in teams_by_grade.items():
                if len(teams_in_grade) > 1:
                    intra_club_pairs = list(combinations(teams_in_grade, 2))
                    intra_club_games = [key for key in club_games if
                                        (key[allowed_keys.index('team1')], key[allowed_keys.index('team2')]) in intra_club_pairs or
                                        (key[allowed_keys.index('team2')], key[allowed_keys.index('team1')]) in intra_club_pairs]
                    
                    no_potential_pairs = len(teams_in_grade) // 2
                    game_vars = []
                    for pair in intra_club_pairs:
                        team1, team2 = pair
                        pair2 = (team2, team1)
                        game_vars.extend([X[game_key] for game_key in intra_club_games
                                          if ((game_key[allowed_keys.index('team1')], game_key[allowed_keys.index('team2')]) == pair 
                                              or (game_key[allowed_keys.index('team1')], game_key[allowed_keys.index('team2')]) == pair2)])
                    
                    # Soft version: penalize shortfall
                    actual_pairs = model.NewIntVar(0, len(game_vars), f'actual_pairs_{club_name}_{grade}')
                    model.Add(actual_pairs == sum(game_vars))
                    
                    shortfall = model.NewIntVar(0, no_potential_pairs, f'pair_shortfall_{club_name}_{grade}')
                    model.Add(shortfall >= no_potential_pairs - actual_pairs)
                    data['penalties']['ClubDayConstraintSoft']['penalties'].append(shortfall)
            
            # SOFT/HARD: All games at same field
            field_usage_vars = defaultdict(list)
            for game_key in club_games:
                field_name = game_key[allowed_keys.index('field_name')]
                field_usage_vars[field_name].append(X[game_key])
            
            field_indicator_vars = []
            for field_name, games in field_usage_vars.items():
                field_var = model.NewBoolVar(f'field_used_{club_name}_{field_name}')
                model.AddMaxEquality(field_var, games)
                field_indicator_vars.append(field_var)
            
            if self.allow_multiple_fields:
                # Soft: penalize multiple fields
                num_fields = model.NewIntVar(0, len(field_indicator_vars), f'num_fields_{club_name}')
                model.Add(num_fields == sum(field_indicator_vars))
                
                extra_fields = model.NewIntVar(0, len(field_indicator_vars), f'extra_fields_{club_name}')
                model.Add(extra_fields >= num_fields - 1)
                data['penalties']['ClubDayConstraintSoft']['penalties'].append(extra_fields)
            else:
                # Hard: exactly one field
                model.Add(sum(field_indicator_vars) == 1)
            
            # SOFT: Contiguous timeslots
            timeslot_groups = defaultdict(list)
            for game in club_games:
                timeslot_groups[game[allowed_keys.index('day_slot')]].append(X[game])
            
            timeslot_indicators = {}
            for day_slot, game_vars in timeslot_groups.items():
                timeslot_indicator = model.NewBoolVar(f'timeslot_indicator_{club_name}_{day_slot}')
                if len(game_vars) > 1:
                    model.AddMaxEquality(timeslot_indicator, game_vars)
                elif len(game_vars) == 1:
                    model.Add(timeslot_indicator == game_vars[0])
                timeslot_indicators[day_slot] = timeslot_indicator
            
            sorted_slots = sorted(timeslot_indicators.keys())
            
            if self.allow_gaps:
                # Soft: penalize gaps
                for i in range(1, len(sorted_slots) - 1):
                    prior_slot = timeslot_indicators[sorted_slots[i - 1]]
                    relevant_slot = timeslot_indicators[sorted_slots[i]]
                    following_slot = timeslot_indicators[sorted_slots[i + 1]]
                    
                    gap_penalty = model.NewBoolVar(f'gap_{club_name}_{sorted_slots[i]}')
                    # Gap exists if prior and following are used but current is not
                    model.Add(gap_penalty >= prior_slot + following_slot - relevant_slot - 1)
                    data['penalties']['ClubDayConstraintSoft']['penalties'].append(gap_penalty)
            else:
                # Hard: no gaps
                for i in range(1, len(sorted_slots) - 1):
                    prior_slot = timeslot_indicators[sorted_slots[i - 1]]
                    relevant_slot = timeslot_indicators[sorted_slots[i]]
                    following_slot = timeslot_indicators[sorted_slots[i + 1]]
                    model.Add(prior_slot + following_slot <= 1).OnlyEnforceIf(relevant_slot.Not())


class AwayAtMaitlandGroupingSoft(SoftConstraint):
    """
    Soft version of AwayAtMaitlandGrouping.
    
    Original: Hard limit of 3 away clubs per weekend at Maitland.
    Soft version: Configurable limit with penalty for exceeding.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 soft_limit: int = 3, hard_limit: Optional[int] = None):
        """
        Args:
            soft_limit: Number of away clubs before penalties apply (default 3)
            hard_limit: Absolute maximum (None = no hard limit, uses only penalties)
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 2
        self.soft_limit = soft_limit
        
        # Adjust limits based on slack level
        slack_adj = int(self.get_slack_multiplier())
        self.soft_limit = soft_limit + slack_adj - 1
        
        if hard_limit is not None:
            self.hard_limit = hard_limit + slack_adj
        else:
            self.hard_limit = None
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'AwayAtMaitlandGroupingSoft')
        
        away_clubs_per_week = defaultdict(lambda: defaultdict(list))
        locked_weeks = data.get('locked_weeks', set())
        
        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                if "Maitland Park" in t.field.location:
                    key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                    
                    if key in X and t.day:
                        away_club = get_club(t1, data['teams']) if "Maitland" in t2 else get_club(t2, data['teams'])
                        away_clubs_per_week[t.week][away_club].append(X[key])
        
        for week, club_games in away_clubs_per_week.items():
            if week in locked_weeks:
                continue
            
            club_scheduled_vars = {}
            for club, game_vars in club_games.items():
                club_scheduled_var = model.NewBoolVar(f'club_{club}_week_{week}_scheduled_soft')
                model.AddMaxEquality(club_scheduled_var, game_vars)
                club_scheduled_vars[club] = club_scheduled_var
            
            num_clubs_var = model.NewIntVar(0, len(club_scheduled_vars), f'num_away_clubs_week{week}_soft')
            model.Add(num_clubs_var == sum(club_scheduled_vars.values()))
            
            # Hard limit if specified
            if self.hard_limit is not None:
                model.Add(num_clubs_var <= self.hard_limit)
            
            # Soft penalty for exceeding soft_limit
            excess = model.NewIntVar(0, len(club_scheduled_vars), f'excess_clubs_week{week}_soft')
            model.Add(excess >= num_clubs_var - self.soft_limit)
            data['penalties']['AwayAtMaitlandGroupingSoft']['penalties'].append(excess)


class TeamConflictConstraintSoft(SoftConstraint):
    """
    Soft version of TeamConflictConstraint.
    
    Original: Teams specified as conflicting cannot play at the same time (hard).
    Soft version: Penalizes conflicts but allows them if necessary.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 max_violations_per_pair: Optional[int] = None):
        """
        Args:
            max_violations_per_pair: Hard limit on violations per pair (None = soft only)
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 2
        
        # Adjust based on slack level
        if max_violations_per_pair is not None:
            self.max_violations_per_pair = max_violations_per_pair + int(self.get_slack_multiplier())
        else:
            self.max_violations_per_pair = None
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'TeamConflictConstraintSoft')
        
        conflicts = data['team_conflicts']
        locked_weeks = data.get('locked_weeks', set())
        timeslots = data['timeslots']
        games = data['games']
        
        for team_pairing in conflicts:
            team1 = team_pairing[0]
            team2 = team_pairing[1]
            
            pair_violations = []
            
            # Group by (week, day_slot) - same time regardless of field
            time_slots_vars = defaultdict(list)
            
            for t in timeslots:
                if t.week in locked_weeks:
                    continue
                if not t.day:
                    continue
                
                for (t1, t2, grade) in games:
                    if t1 in [team1, team2] or t2 in [team1, team2]:
                        key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                        if key in X:
                            time_slots_vars[(t.week, t.day_slot)].append(X[key])
            
            # Soft constraint: penalize conflicts
            for (week, day_slot), game_vars in time_slots_vars.items():
                if len(game_vars) > 1:
                    conflict_count = model.NewIntVar(0, len(game_vars), f'conflict_{team1}_{team2}_w{week}_s{day_slot}')
                    model.Add(conflict_count == sum(game_vars))
                    
                    # Penalty for more than 1 game (conflict)
                    violation = model.NewIntVar(0, len(game_vars) - 1, f'violation_{team1}_{team2}_w{week}_s{day_slot}')
                    model.Add(violation >= conflict_count - 1)
                    data['penalties']['TeamConflictConstraintSoft']['penalties'].append(violation)
                    pair_violations.append(violation)
            
            # Optional hard limit on total violations per pair
            if self.max_violations_per_pair is not None and pair_violations:
                total_violations = model.NewIntVar(0, len(pair_violations) * 10, f'total_violations_{team1}_{team2}')
                model.Add(total_violations == sum(pair_violations))
                model.Add(total_violations <= self.max_violations_per_pair)


# ============== Level 3 Soft Constraints ==============

class EqualMatchUpSpacingConstraintSoft(SoftConstraint):
    """
    Soft version of EqualMatchUpSpacingConstraint.
    
    Original: SLACK = 1 (hard-coded), enforces bounds on round spacing.
    Soft version: Configurable SLACK, penalties for exceeding bounds.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 base_slack: int = 1, enforce_bounds: bool = False):
        """
        Args:
            base_slack: Base slack value (will be multiplied by slack_level)
            enforce_bounds: If True, keeps hard bounds but with more slack
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 3
        self.base_slack = base_slack
        self.enforce_bounds = enforce_bounds
        
        # Calculate actual slack based on slack level
        self.SLACK = int(base_slack * (1 + self.slack_level))
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'EqualMatchUpSpacingConstraintSoft')
        
        games = data['games']
        timeslots = data['timeslots']
        max_rounds = data['num_rounds']['max']
        
        grade_spacing_vars = defaultdict(lambda: defaultdict(int))
        
        for grade in data['grades']:
            # Ideal spacing = T - 1 (see all other opponents before rematch)
            space = grade.num_teams - 1
            
            min_grade_spacing_var = model.NewIntVar(0, space + self.SLACK, f'grade_spacing_{grade.name}_soft')
            model.Add(min_grade_spacing_var == max(0, space - self.SLACK))
            
            max_grade_spacing_var = model.NewIntVar(0, space + self.SLACK * 2, f'max_grade_spacing_{grade.name}_soft')
            model.Add(max_grade_spacing_var == space + self.SLACK)
            
            grade_spacing_vars[grade.name]['min'] = min_grade_spacing_var
            grade_spacing_vars[grade.name]['max'] = max_grade_spacing_var
        
        meetings = defaultdict(lambda: defaultdict(list))
        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day:
                    meetings[(t1, t2, grade)][t.round_no].append(X[key])
        
        for (team_pair, rounds) in meetings.items():
            indicator_list = []
            week_no_list = []
            
            sorted_rounds = dict(sorted(rounds.items(), key=lambda x: x[0]))
            for round_no, game_vars in sorted_rounds.items():
                if len(game_vars) >= 1:
                    indicator_var = model.NewBoolVar(f'meeting_indicator_{team_pair[0]}_{team_pair[1]}_round{round_no}_soft')
                    model.AddMaxEquality(indicator_var, game_vars)
                    indicator_list.append(indicator_var)
                    
                    week_var = model.NewIntVar(0, max_rounds, f'week_var_{team_pair[0]}_{team_pair[1]}_round{round_no}_soft')
                    model.Add(week_var == round_no).OnlyEnforceIf(indicator_var)
                    model.Add(week_var == 0).OnlyEnforceIf(indicator_var.Not())
                    week_no_list.append(week_var)
            
            if not indicator_list:
                continue
            
            grade = team_pair[2]
            
            no_meetings = model.NewIntVar(0, len(indicator_list), f'no_meetings_{team_pair[0]}_{team_pair[1]}_soft')
            model.Add(no_meetings == sum(indicator_list))
            
            meets_twice = model.NewBoolVar(f"meets2_{team_pair[0]}_{team_pair[1]}_soft")
            model.Add(no_meetings >= 2).OnlyEnforceIf(meets_twice)
            model.Add(no_meetings < 2).OnlyEnforceIf(meets_twice.Not())
            
            round_sum = model.NewIntVar(0, max_rounds * len(indicator_list), f'round_sum_{team_pair[0]}_{team_pair[1]}_soft')
            model.Add(round_sum == sum(week_no_list))
            
            max_round_meet = model.NewIntVar(0, max_rounds, f'max_round_meet_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddMaxEquality(max_round_meet, week_no_list)
            
            multi_max = model.NewIntVar(0, max_rounds * len(indicator_list), f'max_week_sum_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddMultiplicationEquality(multi_max, [max_round_meet, no_meetings])
            
            # Calculate bounds
            upper_bound_space_coef = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_mult_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddMultiplicationEquality(upper_bound_space_coef, [no_meetings - 1, no_meetings])
            
            upper_bound_space_coef_2 = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_mult2_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddDivisionEquality(upper_bound_space_coef_2, upper_bound_space_coef, 2)
            
            upper_bound_subtraction = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_sub_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddMultiplicationEquality(upper_bound_subtraction, [grade_spacing_vars[grade]['min'], upper_bound_space_coef_2])
            
            lower_bound_space_coef = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_mult_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddMultiplicationEquality(lower_bound_space_coef, [no_meetings, no_meetings - 1])
            
            lower_bound_coef_2 = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_mult2_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddDivisionEquality(lower_bound_coef_2, lower_bound_space_coef, 2)
            
            lower_bound_subtraction = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_sub_{team_pair[0]}_{team_pair[1]}_soft')
            model.AddMultiplicationEquality(lower_bound_subtraction, [grade_spacing_vars[grade]['max'], lower_bound_coef_2])
            
            upper_bound = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_{team_pair[0]}_{team_pair[1]}_soft')
            model.Add(upper_bound == multi_max - upper_bound_subtraction)
            
            lower_bound = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_{team_pair[0]}_{team_pair[1]}_soft')
            model.Add(lower_bound == multi_max - lower_bound_subtraction)
            
            if self.enforce_bounds:
                # Hard constraint for minimum gap only (no maximum gap)
                model.Add(round_sum <= upper_bound).OnlyEnforceIf(meets_twice)
            else:
                # Soft: penalize bound violations
                upper_violation = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_violation_{team_pair[0]}_{team_pair[1]}_soft')
                model.Add(upper_violation >= round_sum - upper_bound).OnlyEnforceIf(meets_twice)
                model.Add(upper_violation == 0).OnlyEnforceIf(meets_twice.Not())
                
                lower_violation = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_violation_{team_pair[0]}_{team_pair[1]}_soft')
                model.Add(lower_violation >= lower_bound - round_sum).OnlyEnforceIf(meets_twice)
                model.Add(lower_violation == 0).OnlyEnforceIf(meets_twice.Not())
                
                data['penalties']['EqualMatchUpSpacingConstraintSoft']['penalties'].append(upper_violation)
                data['penalties']['EqualMatchUpSpacingConstraintSoft']['penalties'].append(lower_violation)


class ClubGradeAdjacencyConstraintSoft(SoftConstraint):
    """
    Soft version of ClubGradeAdjacencyConstraint.
    
    Original: Adjacent grades from same club cannot play at same time (hard).
    Soft version: Penalizes adjacent grade conflicts.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 max_conflicts_per_club: Optional[int] = None):
        """
        Args:
            max_conflicts_per_club: Hard limit on conflicts per club (None = soft only)
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 3
        self.max_conflicts_per_club = max_conflicts_per_club
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'ClubGradeAdjacencyConstraintSoft')
        
        games = data['games']
        timeslots = data['timeslots']
        teams = data['teams']
        CLUBS = data['clubs']
        GRADES = data['grades']
        
        grade_order = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
        adj_pairs = [(grade_order[i], grade_order[i+1]) for i in range(len(grade_order)-1)]
        
        def club_of(team_name):
            for t in teams:
                if t.name == team_name:
                    return t.club.name
            raise ValueError(f"Unknown team {team_name}")
        
        club_dup_grades = defaultdict(lambda: defaultdict(list))
        for club in [c.name for c in CLUBS]:
            for grade in [g.name for g in GRADES]:
                dup_teams = get_duplicated_graded_teams(club, grade, teams)
                club_dup_grades[club][grade].extend(dup_teams)
        
        club_slot_games = defaultdict(lambda: [])
        club_dup_games = defaultdict(lambda: [])
        
        for (t1, t2, grade) in games:
            t1_club = club_of(t1)
            t2_club = club_of(t2)
            
            for ts in timeslots:
                slot_id = (ts.week, ts.day_slot)
                key = (t1, t2, grade, ts.day, ts.day_slot, ts.time, ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
                if key in X:
                    var = X[key]
                    if t1_club != t2_club:
                        club_slot_games[(t1_club, slot_id, grade)].append(var)
                        club_slot_games[(t2_club, slot_id, grade)].append(var)
                        if t1 in club_dup_grades[t1_club][grade]:
                            club_dup_games[(t1_club, slot_id, grade)].append(var)
                        elif t2 in club_dup_grades[t2_club][grade]:
                            club_dup_games[(t2_club, slot_id, grade)].append(var)
                    else:
                        club_slot_games[(t1_club, slot_id, grade)].append(var)
        
        # Same club, same grade teams - keep hard (this is usually critical)
        for (club, slot_id, grade), vars_ in club_dup_games.items():
            if not vars_:
                continue
            model.Add(sum(vars_) <= 1)
        
        # Track conflicts per club for optional hard limit
        club_conflicts = defaultdict(list)
        
        # Adjacent grades - soft constraint with penalty
        for club in [c.name for c in CLUBS]:
            slot_ids = {slot_id for (c_name, slot_id, g) in club_slot_games if c_name == club}
            
            for slot_id in slot_ids:
                for g1, g2 in adj_pairs:
                    vars_g1 = club_slot_games.get((club, slot_id, g1), [])
                    vars_g2 = club_slot_games.get((club, slot_id, g2), [])
                    if not vars_g1 or not vars_g2:
                        continue
                    
                    # Soft: penalize conflicts
                    conflict = model.NewIntVar(0, len(vars_g1) + len(vars_g2), f'adj_conflict_{club}_{slot_id}_{g1}_{g2}_soft')
                    model.Add(conflict >= sum(vars_g1) + sum(vars_g2) - 1)
                    data['penalties']['ClubGradeAdjacencyConstraintSoft']['penalties'].append(conflict)
                    club_conflicts[club].append(conflict)
        
        # Optional hard limit on total conflicts per club
        if self.max_conflicts_per_club is not None:
            for club, conflicts in club_conflicts.items():
                if conflicts:
                    total = model.NewIntVar(0, len(conflicts) * 10, f'total_conflicts_{club}_soft')
                    model.Add(total == sum(conflicts))
                    model.Add(total <= self.max_conflicts_per_club)


class ClubVsClubAlignmentSoft(SoftConstraint):
    """
    Soft version of ClubVsClubAlignment.
    
    Original: Teams in a club should play only one other club on a weekend (hard).
    Soft version: Penalizes multiple opponent clubs per weekend.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 enforce_same_field: bool = False):
        """
        Args:
            enforce_same_field: If True, keeps same-field requirement hard
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 3
        self.enforce_same_field = enforce_same_field
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'ClubVsClubAlignmentSoft')
        
        num_rounds = data['num_rounds']
        per_team_games = {
            grade.name: (num_rounds['max'] // (grade.num_teams - 1)) if grade.num_teams % 2 == 0 
            else (num_rounds['max'] // grade.num_teams) 
            for grade in data['grades']
        }
        
        ordered_games = dict(sorted(per_team_games.items(), key=lambda item: item[1]))
        grades_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        fields_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for t1, t2, grade in data['games']:
            if grade in ['PHL', '2nd']:
                continue
            
            for t in data['timeslots']:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day:
                    playing_clubs = tuple(sorted((get_club(t1, data['teams']), get_club(t2, data['teams']))))
                    grades_dict[grade][playing_clubs][t.round_no].append(X[key])
                    if t.day == 'Sunday':
                        fields_dict[playing_clubs][t.round_no][t.field.name].append(X[key])
        
        used_grades = []
        ini_num = 0
        
        for grade, num_games in ordered_games.items():
            original_grade = grades_dict[grade]
            used_grades.append(grade)
            
            if num_games <= ini_num:
                continue
            ini_num = num_games
            
            for grade2, club_dict in grades_dict.items():
                if grade2 in used_grades:
                    continue
                
                for clubs, rounds in original_grade.items():
                    if clubs in club_dict:
                        coincide_vars = []
                        
                        for round_no, game_vars in rounds.items():
                            if round_no in club_dict[clubs]:
                                game_indicator = model.NewBoolVar(f"game_played_{clubs}_{round_no}_soft")
                                model.AddMaxEquality(game_indicator, game_vars)
                                
                                second_game_vars = club_dict[clubs][round_no]
                                second_indicator = model.NewBoolVar(f"second_played_{clubs}_{round_no}_soft")
                                model.AddMaxEquality(second_indicator, second_game_vars)
                                
                                coincide = model.NewBoolVar(f"coincide_{clubs}_{round_no}_soft")
                                model.Add(coincide <= game_indicator)
                                model.Add(coincide <= second_indicator)
                                model.Add(coincide >= game_indicator + second_indicator - 1)
                                
                                coincide_vars.append(coincide)
                                
                                # Field alignment
                                field_usage_vars = defaultdict(list)
                                for field, fgame_vars in fields_dict[clubs][round_no].items():
                                    field_indicator = model.NewBoolVar(f"field_games_{clubs}_{round_no}_{field}_soft")
                                    model.AddMaxEquality(field_indicator, fgame_vars)
                                    field_usage_vars[field].append(field_indicator)
                                
                                if self.enforce_same_field:
                                    # Hard: same field
                                    model.Add(sum([v for var in field_usage_vars.values() for v in var]) == 1).OnlyEnforceIf(coincide)
                                else:
                                    # Soft: penalize multiple fields
                                    num_fields = model.NewIntVar(0, len(field_usage_vars), f'num_fields_{clubs}_{round_no}_soft')
                                    model.Add(num_fields == sum([v for var in field_usage_vars.values() for v in var]))
                                    
                                    extra_fields = model.NewIntVar(0, len(field_usage_vars), f'extra_fields_{clubs}_{round_no}_soft')
                                    model.Add(extra_fields >= num_fields - 1).OnlyEnforceIf(coincide)
                                    model.Add(extra_fields == 0).OnlyEnforceIf(coincide.Not())
                                    data['penalties']['ClubVsClubAlignmentSoft']['penalties'].append(extra_fields)
                        
                        # Soft: penalize non-alignment instead of hard requirement
                        if coincide_vars:
                            actual_coincides = model.NewIntVar(0, len(coincide_vars), f'actual_coincides_{clubs}_soft')
                            model.Add(actual_coincides == sum(coincide_vars))
                            
                            shortfall = model.NewIntVar(0, num_games, f'coincide_shortfall_{clubs}_soft')
                            model.Add(shortfall >= num_games - actual_coincides)
                            data['penalties']['ClubVsClubAlignmentSoft']['penalties'].append(shortfall)


# ============== Level 4 Soft Constraints ==============

class EnsureBestTimeslotChoicesSoft(SoftConstraint):
    """
    Soft version of EnsureBestTimeslotChoices.
    
    Original: No gaps between used timeslots, efficient game organization (hard).
    Soft version: Penalizes gaps and inefficient slot usage.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 allow_gaps: bool = True, allow_overflow: bool = True):
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 4
        self.allow_gaps = allow_gaps
        self.allow_overflow = allow_overflow
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'EnsureBestTimeslotChoicesSoft')
        
        games = data['games']
        timeslots = data['timeslots']
        fields = data['fields']
        locked_weeks = data.get('locked_weeks', set())
        
        timeslots_weekly = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        games_per_location = defaultdict(lambda: defaultdict(list))
        
        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day:
                    timeslots_weekly[(t.week, t.day)][t.field.location][t.day_slot].append(X[key])
                    games_per_location[(t.week, t.day)][t.field.location].append(X[key])
        
        timeslots_indicators = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        timeslot_numbers = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        
        for (week, day), locations in timeslots_weekly.items():
            if week in locked_weeks:
                continue
            
            for location, day_slots in locations.items():
                for day_slot, game_vars in day_slots.items():
                    if len(game_vars) > 1:
                        timeslot_indicator = model.NewBoolVar(f'timeslots_indicator_{week}_{location}_soft')
                        timeslots_indicators[(week, day)][location][day_slot] = timeslot_indicator
                        model.AddMaxEquality(timeslot_indicator, game_vars)
                        
                        timeslot_number = model.NewIntVar(0, len(day_slots), f'timeslot_number_{week}_{location}_soft')
                        timeslot_numbers[(week, day)][location][day_slot] = timeslot_number
                        model.Add(timeslot_number == int(day_slot))
        
        # Gap constraint - soft version
        for (week, day), locations in timeslots_indicators.items():
            for location, day_slots in locations.items():
                sorted_slots = sorted(day_slots.keys())
                for i in range(1, len(sorted_slots) - 1):
                    if sorted_slots[i-1] not in day_slots or sorted_slots[i+1] not in day_slots:
                        continue
                    
                    prior_slot = day_slots[sorted_slots[i - 1]]
                    relevant_slot = day_slots[sorted_slots[i]]
                    following_slot = day_slots[sorted_slots[i + 1]]
                    
                    if self.allow_gaps:
                        # Soft: penalize gaps
                        gap = model.NewBoolVar(f'gap_{week}_{location}_{sorted_slots[i]}_soft')
                        model.Add(gap >= prior_slot + following_slot - relevant_slot - 1)
                        data['penalties']['EnsureBestTimeslotChoicesSoft']['penalties'].append(gap)
                    else:
                        # Hard: no gaps
                        model.Add(prior_slot + following_slot <= 1).OnlyEnforceIf(relevant_slot.Not())
        
        # Timeslot efficiency
        for (week, day), locations in games_per_location.items():
            if week in locked_weeks:
                continue
            
            for location in locations:
                fields_at_location = [field for field in fields if field.location == location]
                num_fields = len(fields_at_location)
                if num_fields == 0:
                    continue
                
                no_location_games = model.NewIntVar(0, len(games), f'no_location_games_{week}_{location}_soft')
                model.Add(no_location_games == sum(locations[location]))
                
                quotient = model.NewIntVar(0, len(timeslots), f'quotient_{week}_{location}_soft')
                model.AddDivisionEquality(quotient, no_location_games, num_fields)
                
                no_timeslots = model.NewIntVar(0, len(timeslots), f'no_timeslots_{week}_{location}_soft')
                model.Add(no_timeslots == quotient + 1)
                
                number_vars = timeslot_numbers[(week, day)][location]
                for day_slot, number_var in number_vars.items():
                    indicator_var = timeslots_indicators[(week, day)][location].get(day_slot)
                    if indicator_var is None:
                        continue
                    
                    if location == 'Newcastle International Hockey Centre':
                        equivalence_indicator = model.NewIntVar(0, 200, f'equivalence_indicator_{week}_{location}_{day_slot}_soft')
                        model.Add(equivalence_indicator >= 6)
                        model.Add(equivalence_indicator >= no_timeslots)
                        
                        no_timeslots_indic = model.NewBoolVar(f'no_timeslots_indicator_{week}_{location}_{day_slot}_soft')
                        model.Add(no_timeslots <= 6).OnlyEnforceIf(no_timeslots_indic)
                        model.Add(no_timeslots > 6).OnlyEnforceIf(no_timeslots_indic.Not())
                        
                        model.Add(equivalence_indicator <= 6).OnlyEnforceIf(no_timeslots_indic)
                        model.Add(equivalence_indicator <= no_timeslots).OnlyEnforceIf(no_timeslots_indic.Not())
                        
                        if self.allow_overflow:
                            overflow = model.NewIntVar(0, 100, f'slot_overflow_{week}_{location}_{day_slot}_soft')
                            model.Add(overflow >= number_var - equivalence_indicator).OnlyEnforceIf(indicator_var)
                            model.Add(overflow == 0).OnlyEnforceIf(indicator_var.Not())
                            data['penalties']['EnsureBestTimeslotChoicesSoft']['penalties'].append(overflow)
                        else:
                            model.Add(number_var <= equivalence_indicator).OnlyEnforceIf(indicator_var)
                    else:
                        if self.allow_overflow:
                            overflow = model.NewIntVar(0, 100, f'slot_overflow_{week}_{location}_{day_slot}_soft')
                            model.Add(overflow >= number_var - no_timeslots).OnlyEnforceIf(indicator_var)
                            model.Add(overflow == 0).OnlyEnforceIf(indicator_var.Not())
                            data['penalties']['EnsureBestTimeslotChoicesSoft']['penalties'].append(overflow)
                        else:
                            model.Add(number_var <= no_timeslots).OnlyEnforceIf(indicator_var)


class MaximiseClubsPerTimeslotBroadmeadowSoft(SoftConstraint):
    """
    Soft version of MaximiseClubsPerTimeslotBroadmeadow.
    
    Original: Hard minimum clubs per timeslot.
    Soft version: Penalizes low club diversity instead of hard minimum.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 min_clubs_reduction: int = 0):
        """
        Args:
            min_clubs_reduction: Reduce minimum club requirement by this amount
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 4
        # Reduce minimum based on slack level
        self.min_clubs_reduction = min_clubs_reduction + self.slack_level
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'MaximiseClubsPerTimeslotBroadmeadowSoft')
        
        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())
        
        game_dict = defaultdict(lambda: defaultdict(list))
        
        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                
                if key in X and t.field.location == 'Newcastle International Hockey Centre' and t.day in ['Saturday', 'Sunday']:
                    club1 = get_club(t1, data['teams'])
                    club2 = get_club(t2, data['teams'])
                    
                    game_dict[(t.week, t.day, t.day_slot)][club1].append(X[key])
                    game_dict[(t.week, t.day, t.day_slot)][club2].append(X[key])
        
        for (week, day, timeslot), club_games in game_dict.items():
            if week in locked_weeks:
                continue
            
            club_presence_vars = {}
            for club, game_vars in club_games.items():
                club_var = model.NewBoolVar(f'club_{club}_week{week}_day_slot{timeslot}_soft')
                model.Add(sum(game_vars) >= 1).OnlyEnforceIf(club_var)
                model.Add(sum(game_vars) == 0).OnlyEnforceIf(club_var.Not())
                club_presence_vars[club] = club_var
            
            total_teams_playing = model.NewIntVar(0, len([var for v in club_games.values() for var in v]), f'total_games_week{week}_day_slot{timeslot}_soft')
            model.Add(total_teams_playing == sum([var for v in club_games.values() for var in v]))
            
            # Calculate soft minimum (reduced from original)
            soft_minimum = model.NewIntVar(0, len(club_presence_vars), f'soft_minimum_week{week}_day_slot{timeslot}_soft')
            soft_min_start = model.NewIntVar(0, len(club_presence_vars), f'soft_min_start_week{week}_day_slot{timeslot}_soft')
            model.AddDivisionEquality(soft_min_start, total_teams_playing, 2)
            
            # Reduce minimum by slack
            model.Add(soft_minimum == soft_min_start - self.min_clubs_reduction)
            
            num_clubs_var = model.NewIntVar(0, len(club_presence_vars), f'num_clubs_week{week}_day_slot{timeslot}_soft')
            model.Add(num_clubs_var == sum(club_presence_vars.values()))
            
            timeslot_used_indicator = model.NewBoolVar(f'timeslot_used_week{week}_day_slot{timeslot}_soft')
            model.Add(total_teams_playing >= 1).OnlyEnforceIf(timeslot_used_indicator)
            model.Add(total_teams_playing == 0).OnlyEnforceIf(timeslot_used_indicator.Not())
            
            # Soft: penalize shortfall from minimum instead of hard constraint
            shortfall = model.NewIntVar(0, len(club_presence_vars), f'min_shortfall_week{week}_day_slot{timeslot}_soft')
            model.Add(shortfall >= soft_minimum - num_clubs_var).OnlyEnforceIf(timeslot_used_indicator)
            model.Add(shortfall == 0).OnlyEnforceIf(timeslot_used_indicator.Not())
            data['penalties']['MaximiseClubsPerTimeslotBroadmeadowSoft']['penalties'].append(shortfall)
            
            # Penalty for low diversity (same as original)
            penalty_var = model.NewIntVar(0, len([var for v in club_games.values() for var in v]), f'penalty_week{week}_day_slot{timeslot}_soft')
            model.Add(penalty_var >= total_teams_playing - num_clubs_var).OnlyEnforceIf(timeslot_used_indicator)
            model.Add(penalty_var == 0).OnlyEnforceIf(timeslot_used_indicator.Not())
            data['penalties']['MaximiseClubsPerTimeslotBroadmeadowSoft']['penalties'].append(penalty_var)


class MinimiseClubsOnAFieldBroadmeadowSoft(SoftConstraint):
    """
    Soft version of MinimiseClubsOnAFieldBroadmeadow.
    
    Original: Hard limit of 5 clubs per field per day.
    Soft version: Configurable limit with overflow penalty.
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 soft_limit: int = 5, hard_limit: Optional[int] = None):
        """
        Args:
            soft_limit: Number of clubs before penalties apply
            hard_limit: Absolute maximum (None = no hard limit)
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 4
        
        # Adjust limits based on slack level
        slack_adj = self.slack_level
        self.soft_limit = soft_limit + slack_adj
        self.hard_limit = hard_limit + slack_adj if hard_limit else None
    
    def apply(self, model, X, data):
        self._init_penalties(data, 'MinimiseClubsOnAFieldBroadmeadowSoft')
        
        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())
        
        game_dict = defaultdict(lambda: defaultdict(list))
        
        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                
                if key in X and t.field.location == 'Newcastle International Hockey Centre' and t.day in ['Saturday', 'Sunday']:
                    club1 = get_club(t1, data['teams'])
                    club2 = get_club(t2, data['teams'])
                    
                    game_dict[(t.week, t.date, t.field.name)][club1].append(X[key])
                    game_dict[(t.week, t.date, t.field.name)][club2].append(X[key])
        
        for (week, day, field_name), club_games in game_dict.items():
            if week in locked_weeks:
                continue
            
            club_presence_vars = {}
            for club, game_vars in club_games.items():
                club_var = model.NewBoolVar(f'club_{club}_week{week}_day{day}_field{field_name}_soft')
                model.AddBoolOr([v for v in game_vars]).OnlyEnforceIf(club_var)
                model.AddBoolAnd([v.Not() for v in game_vars]).OnlyEnforceIf(club_var.Not())
                club_presence_vars[club] = club_var
            
            num_clubs_var = model.NewIntVar(0, len(games), f'num_clubs_week{week}_day{day}_field{field_name}_soft')
            model.Add(num_clubs_var == sum(club_presence_vars.values()))
            
            # Hard limit if specified
            if self.hard_limit is not None:
                model.Add(num_clubs_var <= self.hard_limit)
            
            # Soft: penalize exceeding soft_limit
            overflow = model.NewIntVar(0, len(games), f'overflow_week{week}_day{day}_field{field_name}_soft')
            model.Add(overflow >= num_clubs_var - self.soft_limit)
            data['penalties']['MinimiseClubsOnAFieldBroadmeadowSoft']['penalties'].append(overflow)
            
            # Penalty for deviation from ideal (2 clubs)
            penalty_var = model.NewIntVar(0, len(games), f'penalty_week{week}_day{day}_field{field_name}_soft')
            model.AddAbsEquality(penalty_var, num_clubs_var - 2)
            data['penalties']['MinimiseClubsOnAFieldBroadmeadowSoft']['penalties'].append(penalty_var)


class PreferredTimesConstraintSoft(SoftConstraint):
    """
    Soft version of PreferredTimesConstraint.
    
    Supports two PREFERENCE_NO_PLAY formats:
    - 2025 format: {'ClubName': [{'date': '...', ...}]}
    - 2026 format: {'EntryName': {'club': '...', 'dates': [...], 'grade': '...'}}
    """
    
    def __init__(self, slack_level: int = 1, penalty_weight: Optional[int] = None,
                 weight_multiplier: float = 1.0):
        """
        Args:
            weight_multiplier: Multiply base penalty weight (lower = more relaxed)
        """
        super().__init__(slack_level, penalty_weight)
        self.severity_level = 4
        
        # Reduce weight based on slack level
        self.weight_multiplier = weight_multiplier / (1 + self.slack_level * 0.5)
    
    def apply(self, model, X, data):
        adjusted_weight = int(self.penalty_weight * self.weight_multiplier)
        
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['PreferredTimesConstraintSoft'] = {
            'weight': adjusted_weight,
            'penalties': []
        }
        
        teams = data['teams']
        clubs = data['clubs']
        noplay = data.get('preference_no_play', {})
        locked_weeks = data.get('locked_weeks', set())
        
        if not noplay:
            return  # No preferences to apply
        
        # Keys used to match game tuples to restriction dicts
        allowed_keys = ['team_name', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        allowed_keys2 = ['team1', 'team_name', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        
        # Normalize both formats to consistent structure
        try:
            normalized = _normalize_preference_no_play(noplay, teams, clubs)
        except ValueError as e:
            print(f"Warning: {e} - skipping PreferredTimesConstraintSoft")
            return
        
        # Enforce no-play times with penalties
        for entry_key, club_name, club_teams, constraint in normalized:
            if 'date' not in constraint:
                continue
            
            if get_nearest_week_by_date(constraint['date'], data['timeslots']) in locked_weeks:
                continue

            for i, game_key in enumerate(X):
                # Check if any club team is in this game
                if game_key[0] not in club_teams and game_key[1] not in club_teams:
                    continue
                    
                # Try matching with both key orderings
                game_dict = dict(zip(allowed_keys, game_key))
                game_dict2 = dict(zip(allowed_keys2, game_key))
                
                matches = all(game_dict.get(k) == v for k, v in constraint.items())
                matches2 = all(game_dict2.get(k) == v for k, v in constraint.items())
                
                if matches or matches2:
                    penalty_var = model.NewIntVar(0, 1, f"penalty_{entry_key}_{i}_soft")
                    model.Add(penalty_var == X[game_key])
                    data['penalties']['PreferredTimesConstraintSoft']['penalties'].append(penalty_var)


# ============== Constraint Factory ==============

def get_soft_constraint(constraint_name: str, slack_level: int = 1, **kwargs):
    """
    Factory function to get the soft version of a constraint.
    
    Args:
        constraint_name: Name of the original constraint class (with or without AI suffix)
        slack_level: 0=tight, 1=normal, 2=relaxed, 3+=very relaxed
        **kwargs: Additional arguments for the specific constraint
    
    Returns:
        Instance of the soft constraint class, or None if no soft version available
    """
    SOFT_CONSTRAINTS = {
        # Level 2 - HIGH
        'ClubDayConstraint': ClubDayConstraintSoft,
        'ClubDayConstraintAI': ClubDayConstraintSoft,
        'AwayAtMaitlandGrouping': AwayAtMaitlandGroupingSoft,
        'AwayAtMaitlandGroupingAI': AwayAtMaitlandGroupingSoft,
        'TeamConflictConstraint': TeamConflictConstraintSoft,
        'TeamConflictConstraintAI': TeamConflictConstraintSoft,
        
        # Level 3 - MEDIUM
        'EqualMatchUpSpacingConstraint': EqualMatchUpSpacingConstraintSoft,
        'EqualMatchUpSpacingConstraintAI': EqualMatchUpSpacingConstraintSoft,
        'ClubGradeAdjacencyConstraint': ClubGradeAdjacencyConstraintSoft,
        'ClubGradeAdjacencyConstraintAI': ClubGradeAdjacencyConstraintSoft,
        'ClubVsClubAlignment': ClubVsClubAlignmentSoft,
        'ClubVsClubAlignmentAI': ClubVsClubAlignmentSoft,
        
        # Level 4 - LOW
        'EnsureBestTimeslotChoices': EnsureBestTimeslotChoicesSoft,
        'EnsureBestTimeslotChoicesAI': EnsureBestTimeslotChoicesSoft,
        'MaximiseClubsPerTimeslotBroadmeadow': MaximiseClubsPerTimeslotBroadmeadowSoft,
        'MaximiseClubsPerTimeslotBroadmeadowAI': MaximiseClubsPerTimeslotBroadmeadowSoft,
        'MinimiseClubsOnAFieldBroadmeadow': MinimiseClubsOnAFieldBroadmeadowSoft,
        'MinimiseClubsOnAFieldBroadmeadowAI': MinimiseClubsOnAFieldBroadmeadowSoft,
        'PreferredTimesConstraint': PreferredTimesConstraintSoft,
        'PreferredTimesConstraintAI': PreferredTimesConstraintSoft,
    }
    
    if constraint_name not in SOFT_CONSTRAINTS:
        # No soft version available (Level 1 constraints, etc.)
        return None
    
    return SOFT_CONSTRAINTS[constraint_name](slack_level=slack_level, **kwargs)


# ============== Soft Stage Definitions ==============

def get_soft_stage_constraints(slack_level: int = 1):
    """
    Get soft constraint instances for staged solving.
    
    Args:
        slack_level: 0=tight, 1=normal, 2=relaxed
    
    Returns:
        Dict with soft constraint lists by stage
    """
    return {
        'stage1_soft': {
            'name': 'Soft Stage 1 - Level 2 Constraints',
            'description': 'Relaxed versions of HIGH severity constraints',
            'constraints': [
                ClubDayConstraintSoft(slack_level=slack_level),
                AwayAtMaitlandGroupingSoft(slack_level=slack_level),
                TeamConflictConstraintSoft(slack_level=slack_level),
            ],
        },
        'stage2_soft': {
            'name': 'Soft Stage 2 - Level 3 Constraints',
            'description': 'Relaxed versions of MEDIUM severity constraints',
            'constraints': [
                EqualMatchUpSpacingConstraintSoft(slack_level=slack_level),
                ClubGradeAdjacencyConstraintSoft(slack_level=slack_level),
                ClubVsClubAlignmentSoft(slack_level=slack_level),
            ],
        },
        'stage3_soft': {
            'name': 'Soft Stage 3 - Level 4 Constraints',
            'description': 'Relaxed versions of LOW severity constraints',
            'constraints': [
                EnsureBestTimeslotChoicesSoft(slack_level=slack_level),
                MaximiseClubsPerTimeslotBroadmeadowSoft(slack_level=slack_level),
                MinimiseClubsOnAFieldBroadmeadowSoft(slack_level=slack_level),
                PreferredTimesConstraintSoft(slack_level=slack_level),
            ],
        },
    }
