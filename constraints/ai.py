# constraints_ai.py
"""
AI-Enhanced Constraint classes for scheduling system.

These constraints achieve the same outcomes as the original constraints but are
written with:
1. Cleaner, more maintainable code structure
2. Fewer edge cases through better abstractions
3. Optimized constraint formulations that may solve faster
4. Better variable reuse to reduce model size
5. More descriptive naming and documentation

Each constraint class mirrors its counterpart in constraints.py with the suffix _AI.
"""

from ortools.sat.python import cp_model
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from dataclasses import dataclass

# Import utility functions
from utils import (
    get_club, get_duplicated_graded_teams, get_teams_from_club, 
    get_club_from_clubname, get_nearest_week_by_date
)


# ============== Base Classes ==============

class ConstraintAI(ABC):
    """Enhanced abstract base class for all scheduling constraints."""
    
    # Constraint classification for staged solving
    PRIORITY = "required"  # "required", "strong", "medium", "soft"
    
    @abstractmethod
    def apply(self, model: cp_model.CpModel, X: dict, data: dict) -> int:
        """
        Apply constraint to the OR-Tools model.
        
        Returns:
            Number of constraints added to the model.
        """
        pass
    
    @staticmethod
    def _get_game_key_parts(key: tuple) -> dict:
        """Extract named parts from a game key tuple."""
        if len(key) >= 11:
            return {
                'team1': key[0],
                'team2': key[1],
                'grade': key[2],
                'day': key[3],
                'day_slot': key[4],
                'time': key[5],
                'week': key[6],
                'date': key[7],
                'round_no': key[8],
                'field_name': key[9],
                'field_location': key[10],
            }
        return {}
    
    @staticmethod
    def _filter_vars_by_week(X: dict, locked_weeks: set) -> dict:
        """Filter decision variables to exclude locked weeks."""
        if not locked_weeks:
            return {k: v for k, v in X.items() if len(k) >= 7 and k[3]}
        return {k: v for k, v in X.items() if len(k) >= 7 and k[6] not in locked_weeks and k[3]}
    
    @staticmethod
    def _group_vars_by(X: dict, key_func) -> dict:
        """Group decision variables by a key function."""
        groups = defaultdict(list)
        for k, v in X.items():
            group_key = key_func(k)
            if group_key is not None:
                groups[group_key].append(v)
        return dict(groups)


@dataclass
class PenaltyConfig:
    """Configuration for soft constraint penalties."""
    name: str
    weight: int
    penalties: List


# ============== Core Scheduling Constraints ==============

class NoDoubleBookingTeamsConstraintAI(ConstraintAI):
    """
    Ensure no team is scheduled for more than one game per week.
    
    Enhanced version uses set-based grouping for cleaner logic.
    """
    PRIORITY = "required"
    
    def apply(self, model, X, data) -> int:
        locked_weeks = data.get('locked_weeks', set())
        X_filtered = self._filter_vars_by_week(X, locked_weeks)
        
        # Group by (week, team) - each team appears in both t1 and t2 positions
        week_team_vars = defaultdict(list)
        
        for key, var in X_filtered.items():
            parts = self._get_game_key_parts(key)
            week = parts['week']
            week_team_vars[(week, parts['team1'])].append(var)
            week_team_vars[(week, parts['team2'])].append(var)
        
        # Add constraints
        constraints_added = 0
        for (week, team), vars_list in week_team_vars.items():
            if len(vars_list) > 1:  # Only add constraint if multiple options exist
                model.Add(sum(vars_list) <= 1)
                constraints_added += 1
        
        return constraints_added


class NoDoubleBookingFieldsConstraintAI(ConstraintAI):
    """
    Ensure no field hosts more than one game per timeslot.
    
    Enhanced version uses tuple-based slot identification.
    """
    PRIORITY = "required"
    
    def apply(self, model, X, data) -> int:
        locked_weeks = data.get('locked_weeks', set())
        X_filtered = self._filter_vars_by_week(X, locked_weeks)
        
        # Group by (week, day_slot, field_name)
        slot_vars = self._group_vars_by(
            X_filtered,
            lambda k: (k[6], k[4], k[9]) if len(k) >= 10 else None
        )
        
        constraints_added = 0
        for slot, vars_list in slot_vars.items():
            if len(vars_list) > 1:
                model.Add(sum(vars_list) <= 1)
                constraints_added += 1
        
        return constraints_added


class EnsureEqualGamesAndBalanceMatchUpsAI(ConstraintAI):
    """
    Ensure balanced games and matchups.
    
    Enhanced version:
    - Cleaner base/extra calculation
    - Unified dummy slot handling
    - Better bounds computation
    """
    PRIORITY = "required"
    
    def apply(self, model, X, data) -> int:
        games = data['games']
        timeslots = data['timeslots']
        num_rounds = data['num_rounds']
        num_dummy = data.get('num_dummy_timeslots', 0)
        teams = data['teams']
        
        constraints_added = 0
        
        # Collect vars per team and per pair, grouped by grade
        team_vars = defaultdict(lambda: defaultdict(list))
        pair_vars = defaultdict(lambda: defaultdict(list))
        
        for (t1, t2, grade) in games:
            # Real timeslots
            for t in timeslots:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, 
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    var = X[key]
                    team_vars[grade][t1].append(var)
                    team_vars[grade][t2].append(var)
                    pair_vars[grade][tuple(sorted((t1, t2)))].append(var)
            
            # Dummy slots
            for i in range(num_dummy):
                dummy_key = (t1, t2, grade, i)
                if dummy_key in X:
                    var = X[dummy_key]
                    team_vars[grade][t1].append(var)
                    team_vars[grade][t2].append(var)
                    pair_vars[grade][tuple(sorted((t1, t2)))].append(var)
        
        # Apply constraints per grade
        for grade, teams_in_grade in team_vars.items():
            T = len(teams_in_grade)  # Number of teams in grade
            R = num_rounds.get(grade, 0)  # Target rounds per team
            
            if T < 2 or R == 0:
                continue
            
            # 1. Each team plays exactly R games
            for team, vars_list in teams_in_grade.items():
                model.Add(sum(vars_list) == R)
                constraints_added += 1
            
            # 2. Calculate balanced matchup bounds
            # base = how many times each pair meets minimally
            base = R // (T - 1) if T % 2 == 0 else R // T
            
            # 3. Each pair meets between base and base+1 times
            for pair, vars_list in pair_vars[grade].items():
                model.Add(sum(vars_list) >= base)
                model.Add(sum(vars_list) <= base + 1)
                constraints_added += 2
        
        return constraints_added


# ============== PHL and 2nd Grade Constraints ==============

class PHLAndSecondGradeAdjacencyAI(ConstraintAI):
    """
    Ensure PHL and 2nds from same club don't play in adjacent slots at different locations.
    
    Enhanced version:
    - Cleaner time window calculation
    - Precomputed club-team mappings
    - Reduced nested loops
    """
    PRIORITY = "strong"
    
    ADJACENCY_MINUTES = 120  # 2 hours
    
    def apply(self, model, X, data) -> int:
        games = data['games']
        timeslots = data['timeslots']
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        # Precompute club to 2nd grade teams mapping
        club_2nd_teams = defaultdict(list)
        for team in teams:
            if team.grade == '2nd':
                club_2nd_teams[team.club.name].append(team.name)
        
        # Group PHL games by (club, week, day) with their time/location
        phl_games_info = defaultdict(lambda: defaultdict(list))
        second_games_info = defaultdict(lambda: defaultdict(list))
        
        for t in timeslots:
            if t.week in locked_weeks or not t.day:
                continue
                
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, 
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key not in X:
                    continue
                
                if grade == 'PHL':
                    for team in [t1, t2]:
                        club = get_club(team, teams)
                        if club in club_2nd_teams:
                            phl_games_info[(club, t.week, t.day)][(t.time, t.field.location)].append(X[key])
                
                elif grade == '2nd':
                    for team in [t1, t2]:
                        club = get_club(team, teams)
                        second_games_info[(club, t.week, t.day)][(t.time, t.field.location, team)].append(X[key])
        
        # Build adjacency constraints
        for club_week_day, phl_slots in phl_games_info.items():
            club, week, day = club_week_day
            second_slots = second_games_info[club_week_day]
            
            for (phl_time, phl_loc), phl_vars in phl_slots.items():
                if not phl_time:
                    continue
                    
                phl_time_dt = datetime.strptime(phl_time, '%H:%M')
                min_time = (phl_time_dt - timedelta(minutes=self.ADJACENCY_MINUTES)).time()
                max_time = (phl_time_dt + timedelta(minutes=self.ADJACENCY_MINUTES)).time()
                
                for (sec_time, sec_loc, sec_team), sec_vars in second_slots.items():
                    if not sec_time:
                        continue
                    
                    sec_time_t = datetime.strptime(sec_time, '%H:%M').time()
                    
                    # Case 1: Adjacent time, different location = conflict
                    if min_time <= sec_time_t <= max_time and sec_loc != phl_loc:
                        model.Add(sum(phl_vars) + sum(sec_vars) <= 1)
                        constraints_added += 1
                    # Case 2: Same location, non-adjacent time = conflict
                    elif (sec_time_t >= max_time or sec_time_t <= min_time) and sec_loc == phl_loc:
                        model.Add(sum(phl_vars) + sum(sec_vars) <= 1)
                        constraints_added += 1
        
        return constraints_added


class PHLAndSecondGradeTimesAI(ConstraintAI):
    """
    PHL timing rules:
    - No concurrent PHL games at Broadmeadow
    - No concurrent 2nd grade and PHL from same club at Broadmeadow
    - Max 3 Friday night games at Broadmeadow
    - Exactly 8 Friday night games at Gosford (AGM decision)
    - Soft penalty for preferred dates
    """
    PRIORITY = "strong"
    
    BROADMEADOW = 'Newcastle International Hockey Centre'
    GOSFORD = 'Central Coast Hockey Park'
    MAX_FRIDAY_GAMES = 3
    GOSFORD_FRIDAY_GAMES = 8  # AGM confirmed
    
    def apply(self, model, X, data) -> int:
        # Initialize penalties
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['phl_preferences'] = {'weight': 10000, 'penalties': []}
        
        games = data['games']
        timeslots = data['timeslots']
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        phl_preferences = data.get('phl_preferences', {})
        
        constraints_added = 0
        
        # Build mappings
        phl_slot_vars = defaultdict(list)  # (week, day, day_slot, location) -> vars
        club_phl_vars = defaultdict(list)  # (week, day, day_slot, location, club) -> vars
        club_2nd_vars = defaultdict(list)  # Same structure
        friday_broadmeadow_vars = []
        friday_gosford_vars = []
        preferred_date_vars = defaultdict(list)
        
        preferred_dates = set(
            d.date().strftime('%Y-%m-%d') 
            for d in phl_preferences.get('preferred_dates', [])
        )
        
        for t in timeslots:
            if t.week in locked_weeks or not t.day:
                continue
            
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key not in X:
                    continue
                
                var = X[key]
                slot_id = (t.week, t.day, t.day_slot, t.field.location)
                
                if grade == 'PHL':
                    phl_slot_vars[slot_id].append(var)
                    
                    # Track club involvement
                    for team in [t1, t2]:
                        club = get_club(team, teams)
                        club_phl_vars[(*slot_id, club)].append(var)
                    
                    # Friday at Broadmeadow
                    if t.day == 'Friday' and t.field.location == self.BROADMEADOW:
                        friday_broadmeadow_vars.append(var)
                    
                    # Friday at Gosford (Central Coast Hockey Park)
                    if t.day == 'Friday' and t.field.location == self.GOSFORD:
                        friday_gosford_vars.append(var)
                    
                    # Preferred dates
                    if t.date in preferred_dates:
                        preferred_date_vars[t.date].append(var)
                
                elif grade == '2nd':
                    for team in [t1, t2]:
                        club = get_club(team, teams)
                        club_2nd_vars[(*slot_id, club)].append(var)
        
        # Constraint 1: No concurrent PHL at Broadmeadow
        for slot_id, vars_list in phl_slot_vars.items():
            if slot_id[3] == self.BROADMEADOW and len(vars_list) > 1:
                model.Add(sum(vars_list) <= 1)
                constraints_added += 1
        
        # Constraint 2: No concurrent 2nd grade and PHL from same club at Broadmeadow
        for club_slot, phl_vars in club_phl_vars.items():
            week, day, day_slot, location, club = club_slot
            if location != self.BROADMEADOW:
                continue
            
            second_vars = club_2nd_vars.get(club_slot, [])
            if phl_vars and second_vars:
                model.Add(sum(phl_vars) + sum(second_vars) <= 1)
                constraints_added += 1
        
        # Constraint 3: Max Friday night games at Broadmeadow
        if friday_broadmeadow_vars:
            model.Add(sum(friday_broadmeadow_vars) <= self.MAX_FRIDAY_GAMES)
            constraints_added += 1
        
        # Constraint 4: Exactly 8 Friday night games at Gosford (AGM decision)
        if friday_gosford_vars:
            model.Add(sum(friday_gosford_vars) == self.GOSFORD_FRIDAY_GAMES)
            constraints_added += 1
        
        # Soft constraint: Preferred dates (penalty for not having exactly 1 game)
        for date, vars_list in preferred_date_vars.items():
            if vars_list:
                week_no = get_nearest_week_by_date(date, timeslots)
                if week_no not in locked_weeks:
                    penalty = model.NewIntVar(0, len(vars_list), f"pref_date_penalty_{date}")
                    model.AddAbsEquality(penalty, sum(vars_list) - 1)
                    data['penalties']['phl_preferences']['penalties'].append(penalty)
        
        return constraints_added


# ============== Home/Away Balance Constraints ==============

class FiftyFiftyHomeandAwayAI(ConstraintAI):
    """
    Push toward 50% home and 50% away for away-venue teams.
    
    Enhanced version:
    - Unified handling for both Maitland and Gosford
    - Cleaner home/away computation
    """
    PRIORITY = "strong"
    
    AWAY_VENUES = {
        'Maitland': 'Maitland Park',
        'Gosford': 'Central Coast Hockey Park',
    }
    
    def apply(self, model, X, data) -> int:
        timeslots = data['timeslots']
        games = data['games']
        
        constraints_added = 0
        
        # Track home/away games per (team, opponent)
        home_away_vars = defaultdict(lambda: {'home': [], 'away': []})
        
        for t in timeslots:
            if not t.day:
                continue
            
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key not in X:
                    continue
                
                var = X[key]
                location = t.field.location
                
                for venue_name, venue_location in self.AWAY_VENUES.items():
                    # Find which team (if any) is from this venue
                    for team, other in [(t1, t2), (t2, t1)]:
                        if venue_name in team and venue_name not in other:
                            pair_key = (team, other)
                            if location == venue_location:
                                home_away_vars[pair_key]['home'].append(var)
                            else:
                                home_away_vars[pair_key]['away'].append(var)
        
        # Add balance constraints
        for pair_key, ha_vars in home_away_vars.items():
            home_vars = ha_vars['home']
            away_vars = ha_vars['away']
            
            if not home_vars or not away_vars:
                continue
            
            home_count = model.NewIntVar(0, len(home_vars), f'home_{pair_key[0]}_{pair_key[1]}')
            total_count = model.NewIntVar(0, len(home_vars) + len(away_vars), f'total_{pair_key[0]}_{pair_key[1]}')
            
            model.Add(home_count == sum(home_vars))
            model.Add(total_count == sum(home_vars) + sum(away_vars))
            
            # Balance: home * 2 should be close to total
            model.Add(home_count * 2 >= total_count - 1)
            model.Add(home_count * 2 <= total_count + 1)
            constraints_added += 2
        
        return constraints_added


# ============== Team Conflict Constraint ==============

class TeamConflictConstraintAI(ConstraintAI):
    """
    Prevent user-specified team pairs from playing at the same timeslot.
    
    Enhanced version:
    - Precomputed slot groupings
    - Single pass per conflict pair
    """
    PRIORITY = "strong"
    
    def apply(self, model, X, data) -> int:
        conflicts = data.get('team_conflicts', [])
        if not conflicts:
            return 0
        
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        # Group vars by (week, day_slot)
        slot_team_vars = defaultdict(lambda: defaultdict(list))
        
        for key, var in X.items():
            if len(key) < 11 or not key[3] or key[6] in locked_weeks:
                continue
            
            slot = (key[6], key[4])  # (week, day_slot)
            slot_team_vars[slot][key[0]].append(var)  # team1
            slot_team_vars[slot][key[1]].append(var)  # team2
        
        # Add constraints for each conflict pair
        for team1, team2 in conflicts:
            for slot, team_vars in slot_team_vars.items():
                vars_t1 = team_vars.get(team1, [])
                vars_t2 = team_vars.get(team2, [])
                
                if vars_t1 and vars_t2:
                    model.Add(sum(vars_t1) + sum(vars_t2) <= 1)
                    constraints_added += 1
        
        return constraints_added


# ============== Venue Constraints ==============

class MaxMaitlandHomeWeekendsAI(ConstraintAI):
    """
    Limit number of playable weekends at away venues.
    
    Enhanced version:
    - Generalized for any away venue
    - Cleaner max games calculation
    """
    PRIORITY = "medium"
    
    def apply(self, model, X, data) -> int:
        timeslots = data['timeslots']
        games = data['games']
        clubs = data['clubs']
        teams = data['teams']
        grades = data['grades']
        
        constraints_added = 0
        
        # Calculate max games per home field
        home_fields = {club.name: club.home_field for club in clubs}
        grade_games = {g.name: g.num_games for g in grades}
        
        max_games_per_field = defaultdict(int)
        for club_name, home_field in home_fields.items():
            club_teams = get_teams_from_club(club_name, teams)
            for team_name in club_teams:
                team = next((t for t in teams if t.name == team_name), None)
                if team:
                    max_games_per_field[home_field] = max(
                        max_games_per_field[home_field],
                        grade_games.get(team.grade, 0)
                    )
        
        # Group games by (week, location) for non-Broadmeadow venues
        week_location_vars = defaultdict(list)
        
        for t in timeslots:
            if not t.day or t.field.location == 'Newcastle International Hockey Centre':
                continue
            
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    week_location_vars[(t.week, t.field.location)].append(X[key])
        
        # Create weekend indicators and limit total weekends per location
        location_indicators = defaultdict(list)
        
        for (week, location), vars_list in week_location_vars.items():
            indicator = model.NewBoolVar(f"weekend_{week}_{location}")
            model.AddMaxEquality(indicator, vars_list)
            location_indicators[location].append(indicator)
        
        # Limit weekends per location
        for location, indicators in location_indicators.items():
            max_weekends = max_games_per_field.get(location, 10) // 2 + 1
            model.Add(sum(indicators) <= max_weekends)
            constraints_added += 1
        
        return constraints_added


# ============== Timeslot Optimization ==============

class EnsureBestTimeslotChoicesAI(ConstraintAI):
    """
    Ensure optimal timeslot usage:
    - No gaps between used timeslots
    - Contiguous scheduling from earliest slots
    
    Enhanced version:
    - Cleaner indicator variable creation
    - Simplified gap detection logic
    """
    PRIORITY = "medium"
    
    BROADMEADOW = 'Newcastle International Hockey Centre'
    BROADMEADOW_MAX_SLOTS = 6
    
    def apply(self, model, X, data) -> int:
        timeslots = data['timeslots']
        games = data['games']
        fields = data['fields']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        # Group vars by (week, day, location, day_slot) and also by (week, day, location) for total games
        slot_vars = defaultdict(list)
        games_per_location = defaultdict(lambda: defaultdict(list))
        
        for t in timeslots:
            if t.week in locked_weeks or not t.day:
                continue
            
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    slot_key = (t.week, t.day, t.field.location, t.day_slot)
                    slot_vars[slot_key].append(X[key])
                    games_per_location[(t.week, t.day)][t.field.location].append(X[key])
        
        # Create slot indicators and slot number variables
        location_day_slots = defaultdict(dict)
        slot_number_vars = defaultdict(dict)
        
        for slot_key, vars_list in slot_vars.items():
            week, day, location, day_slot = slot_key
            
            if len(vars_list) > 1:
                indicator = model.NewBoolVar(f'ai_slot_used_{week}_{day}_{location}_{day_slot}')
                model.AddMaxEquality(indicator, vars_list)
                location_day_slots[(week, day, location)][day_slot] = indicator
                
                # Track slot number for bounding
                num_var = model.NewIntVar(0, len(slot_vars), f'ai_slot_num_{week}_{location}_{day_slot}')
                model.Add(num_var == int(day_slot))
                slot_number_vars[(week, day, location)][day_slot] = num_var
        
        # Part 1: No gaps constraint
        for (week, day, location), day_slots in location_day_slots.items():
            sorted_slots = sorted(day_slots.keys())
            
            for i in range(1, len(sorted_slots) - 1):
                prev_slot = sorted_slots[i - 1]
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]
                
                if prev_slot in day_slots and curr_slot in day_slots and next_slot in day_slots:
                    model.Add(
                        day_slots[prev_slot] + day_slots[next_slot] <= 1
                    ).OnlyEnforceIf(day_slots[curr_slot].Not())
                    constraints_added += 1
        
        # Part 2: Slot-number bounding — push games into earliest timeslots
        for (week, day), locations in games_per_location.items():
            for location, location_vars in locations.items():
                # Count fields at this location
                fields_at_location = [f for f in fields if f.location == location]
                num_fields = len(fields_at_location)
                if num_fields == 0:
                    continue
                
                # Calculate needed timeslots: ceil(games / fields) = floor(games / fields) + 1
                no_location_games = model.NewIntVar(0, len(games), f'ai_loc_games_{week}_{location}')
                model.Add(no_location_games == sum(location_vars))
                
                quotient = model.NewIntVar(0, len(timeslots), f'ai_quot_{week}_{location}')
                model.AddDivisionEquality(quotient, no_location_games, num_fields)
                
                no_timeslots = model.NewIntVar(0, len(timeslots), f'ai_nslots_{week}_{location}')
                model.Add(no_timeslots == quotient + 1)
                
                # Apply slot number bounds
                number_vars = slot_number_vars.get((week, day, location), {})
                for day_slot, number_var in number_vars.items():
                    indicator_var = location_day_slots[(week, day, location)][day_slot]
                    
                    if location == self.BROADMEADOW:
                        # Broadmeadow: cap at BROADMEADOW_MAX_SLOTS
                        equiv = model.NewIntVar(0, 200, f'ai_equiv_{week}_{location}_{day_slot}')
                        model.Add(equiv >= self.BROADMEADOW_MAX_SLOTS)
                        model.Add(equiv >= no_timeslots)
                        
                        cap_indicator = model.NewBoolVar(f'ai_cap_ind_{week}_{location}_{day_slot}')
                        model.Add(no_timeslots <= self.BROADMEADOW_MAX_SLOTS).OnlyEnforceIf(cap_indicator)
                        model.Add(no_timeslots > self.BROADMEADOW_MAX_SLOTS).OnlyEnforceIf(cap_indicator.Not())
                        
                        model.Add(equiv <= self.BROADMEADOW_MAX_SLOTS).OnlyEnforceIf(cap_indicator)
                        model.Add(equiv <= no_timeslots).OnlyEnforceIf(cap_indicator.Not())
                        
                        model.Add(number_var <= equiv).OnlyEnforceIf(indicator_var)
                    else:
                        # Other locations: slot number must be within needed timeslots
                        model.Add(number_var <= no_timeslots).OnlyEnforceIf(indicator_var)
                    
                    constraints_added += 1
        
        return constraints_added


# ============== Club Day Constraint ==============

class ClubDayConstraintAI(ConstraintAI):
    """
    Ensure club days meet requirements:
    - Every club team plays
    - Intra-club matchups when possible
    - All games on same field
    - Contiguous timeslots
    
    Enhanced version:
    - Cleaner sub-constraint organization
    - Better intra-club pair detection
    """
    PRIORITY = "medium"
    
    def apply(self, model, X, data) -> int:
        club_days = data.get('club_days', {})
        if not club_days:
            return 0
        
        teams = data['teams']
        clubs = data['clubs']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        for club_name, desired_date in club_days.items():
            # Validate club exists
            club = get_club_from_clubname(club_name, clubs)
            club_teams = get_teams_from_club(club_name, teams)
            date_str = desired_date.date().strftime('%Y-%m-%d')
            
            closest_week = get_nearest_week_by_date(date_str, timeslots)
            if closest_week in locked_weeks:
                continue
            
            # Find all games for this club on this date
            club_game_vars = {}
            for key, var in X.items():
                if len(key) < 11 or key[7] != date_str:
                    continue
                
                t1, t2 = key[0], key[1]
                if t1 in club_teams or t2 in club_teams:
                    club_game_vars[key] = var
            
            if not club_game_vars:
                continue
            
            # Constraint 1: Every club team must play
            for team in club_teams:
                team_vars = [v for k, v in club_game_vars.items() if team in (k[0], k[1])]
                if team_vars:
                    model.Add(sum(team_vars) >= 1)
                    constraints_added += 1
            
            # Constraint 2: Intra-club matchups for same-grade teams
            teams_by_grade = defaultdict(list)
            for team in club_teams:
                grade = team.rsplit(' ', 1)[1]
                teams_by_grade[grade].append(team)
            
            for grade, grade_teams in teams_by_grade.items():
                if len(grade_teams) > 1:
                    intra_pairs = list(combinations(grade_teams, 2))
                    intra_vars = []
                    
                    for key, var in club_game_vars.items():
                        pair = (key[0], key[1])
                        if pair in intra_pairs or (pair[1], pair[0]) in intra_pairs:
                            intra_vars.append(var)
                    
                    if intra_vars:
                        expected_pairs = len(grade_teams) // 2
                        model.Add(sum(intra_vars) >= expected_pairs)
                        constraints_added += 1
            
            # Constraint 3: All games on same field
            field_vars = defaultdict(list)
            for key, var in club_game_vars.items():
                field_vars[key[9]].append(var)
            
            if len(field_vars) > 1:
                field_indicators = []
                for field_name, vars_list in field_vars.items():
                    indicator = model.NewBoolVar(f'field_{club_name}_{field_name}')
                    model.AddMaxEquality(indicator, vars_list)
                    field_indicators.append(indicator)
                
                model.Add(sum(field_indicators) == 1)
                constraints_added += 1
            
            # Constraint 4: Contiguous timeslots
            slot_vars = defaultdict(list)
            for key, var in club_game_vars.items():
                slot_vars[key[4]].append(var)  # day_slot
            
            slot_indicators = {}
            for day_slot, vars_list in slot_vars.items():
                indicator = model.NewBoolVar(f'slot_{club_name}_{day_slot}')
                model.AddMaxEquality(indicator, vars_list)
                slot_indicators[day_slot] = indicator
            
            sorted_slots = sorted(slot_indicators.keys())
            for i in range(1, len(sorted_slots) - 1):
                prev_slot = sorted_slots[i - 1]
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]
                
                model.Add(
                    slot_indicators[prev_slot] + slot_indicators[next_slot] <= 1
                ).OnlyEnforceIf(slot_indicators[curr_slot].Not())
                constraints_added += 1
        
        return constraints_added


# ============== Matchup Spacing ==============

class EqualMatchUpSpacingConstraintAI(ConstraintAI):
    """
    Spread matchups evenly across rounds.
    
    Aligned with original EqualMatchUpSpacingConstraint:
    - Ideal spacing = T - 1 (see all opponents before rematch)
    - base_slack = max(1, ideal - 2*ideal//3) ≈ one-third of ideal
    - Only minimum gap enforced as HARD constraint (no maximum gap)
    - |diff| from ideal penalized as SOFT
    """
    PRIORITY = "medium"
    
    PENALTY_WEIGHT = 5000
    
    def apply(self, model, X, data) -> int:
        games = data['games']
        timeslots = data['timeslots']
        R = data['num_rounds']['max']
        grades = {g.name: g.num_teams for g in data['grades']}
        
        # Get additional slack from config (--slack flag)
        config_slack = data.get('constraint_slack', {}).get('EqualMatchUpSpacingConstraint', 0)
        
        constraints_added = 0
        
        # Initialize penalty tracking
        if 'penalties' not in data:
            data['penalties'] = {}
        if 'EqualMatchUpSpacing' not in data.get('penalties', {}):
            data['penalties']['EqualMatchUpSpacing'] = {'weight': self.PENALTY_WEIGHT, 'penalties': []}

        # Derive base slack so that min gap = 2*(T-1)//3
        # space = T - 1 (ideal: see all other opponents before rematch)
        # base_slack = (T-1) - 2*(T-1)//3 (approx one-third of T-1)
        def get_base_slack(num_teams):
            ideal = num_teams - 1
            return max(1, ideal - 2 * ideal // 3)
        
        slack_per_grade = {
            name: min(get_base_slack(T) + config_slack, T // 2 + 1)
            for name, T in grades.items()
        }
        
        # Ideal spacing = T - 1 (see all other opponents before rematch)
        space_per_grade = {
            name: T - 1
            for name, T in grades.items()
        }

        # Gather game-vars by (t1, t2, grade, round_no)
        meetings = defaultdict(lambda: defaultdict(list))
        for t in timeslots:
            if not t.day:
                continue
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    meetings[(t1, t2, grade)][t.round_no].append(X[key])

        # For each matchup pair, build spacing constraints
        for (t1, t2, grade), round_map in meetings.items():
            base_slack = slack_per_grade[grade]
            space = space_per_grade[grade]
            
            # Build round indicators and week vars
            indic = []
            weekvs = []
            for r in range(1, R + 1):
                vars_at_r = round_map.get(r, [])
                # indicator: do they meet in round r?
                b = model.NewBoolVar(f"ai_eqsp_ind_{t1}_{t2}_{grade}_r{r}")
                if vars_at_r:
                    model.AddMaxEquality(b, vars_at_r)
                else:
                    model.Add(b == 0)
                indic.append(b)
                constraints_added += 1

                # week var: =r if meeting, else 0
                wv = model.NewIntVar(0, R, f"ai_eqsp_wv_{t1}_{t2}_{grade}_r{r}")
                model.Add(wv == r).OnlyEnforceIf(b)
                model.Add(wv == 0).OnlyEnforceIf(b.Not())
                weekvs.append(wv)
                constraints_added += 2

            # K = total number of meetings
            K = model.NewIntVar(0, R, f"ai_eqsp_K_{t1}_{t2}_{grade}")
            model.Add(K == sum(indic))
            constraints_added += 1

            # Only enforce when K >= 2
            meets2 = model.NewBoolVar(f"ai_eqsp_m2_{t1}_{t2}_{grade}")
            model.Add(K >= 2).OnlyEnforceIf(meets2)
            model.Add(K < 2).OnlyEnforceIf(meets2.Not())
            constraints_added += 2

            # round_sum = sum of weeks where they meet
            round_sum = model.NewIntVar(0, R * R, f"ai_eqsp_rs_{t1}_{t2}_{grade}")
            model.Add(round_sum == sum(weekvs))
            constraints_added += 1

            # max_r = latest week they meet
            max_r = model.NewIntVar(0, R, f"ai_eqsp_mr_{t1}_{t2}_{grade}")
            model.AddMaxEquality(max_r, weekvs)
            constraints_added += 1

            # Compute K*(K-1)/2 for spacing formula
            km1 = model.NewIntVar(-R, R, f"ai_eqsp_km1_{t1}_{t2}_{grade}")
            prod = model.NewIntVar(-R * R, R * R, f"ai_eqsp_prd_{t1}_{t2}_{grade}")
            half = model.NewIntVar(0, R * (R - 1) // 2, f"ai_eqsp_half_{t1}_{t2}_{grade}")
            model.Add(km1 == K - 1)
            model.AddMultiplicationEquality(prod, [K, km1]).OnlyEnforceIf(meets2)
            model.AddDivisionEquality(half, prod, 2).OnlyEnforceIf(meets2)
            constraints_added += 3

            # ideal = K * max_r - space * half
            # This is the expected sum if meetings were evenly spaced backward from max_r
            kmax = model.NewIntVar(0, R * R, f"ai_eqsp_kmax_{t1}_{t2}_{grade}")
            space_half = model.NewIntVar(-R * R, R * R, f"ai_eqsp_sphalf_{t1}_{t2}_{grade}")
            ideal = model.NewIntVar(-R * R, R * R, f"ai_eqsp_ideal_{t1}_{t2}_{grade}")
            model.AddMultiplicationEquality(kmax, [K, max_r]).OnlyEnforceIf(meets2)
            model.AddMultiplicationEquality(space_half, [space, half]).OnlyEnforceIf(meets2)
            model.Add(ideal == kmax - space_half).OnlyEnforceIf(meets2)
            constraints_added += 3

            # Compute allowed slack = base_slack * half (scales with number of meetings)
            max_slack = model.NewIntVar(0, R * (R - 1) // 2 * (base_slack + 5), f"ai_eqsp_maxslk_{t1}_{t2}_{grade}")
            model.Add(max_slack == half * base_slack).OnlyEnforceIf(meets2)
            constraints_added += 1

            # diff = round_sum - ideal
            # Positive diff means games are bunched (gap < ideal) → enforce min gap
            # Negative diff means games are spread (gap > ideal) → allow freely
            diff = model.NewIntVar(-R * R, R * R, f"ai_eqsp_diff_{t1}_{t2}_{grade}")
            model.Add(diff == round_sum - ideal).OnlyEnforceIf(meets2)
            constraints_added += 1

            # HARD constraint: diff <= max_slack (enforces minimum gap only, no maximum gap)
            model.Add(diff <= max_slack).OnlyEnforceIf(meets2)
            constraints_added += 1

            # SOFT penalty: minimize |diff| to prefer spacing close to ideal
            abs_diff = model.NewIntVar(0, R * R, f"ai_eqsp_absdiff_{t1}_{t2}_{grade}")
            model.AddAbsEquality(abs_diff, diff)
            data['penalties']['EqualMatchUpSpacing']['penalties'].append(abs_diff)
        
        return constraints_added


# ============== Club Grade Adjacency ==============

class ClubGradeAdjacencyConstraintAI(ConstraintAI):
    """
    Prevent adjacent grades from same club playing simultaneously.
    
    Enhanced version:
    - Precomputed adjacency pairs
    - Cleaner club-grade mapping
    
    EDGE CASES:
    - slot_id is (week, day_slot) - ignores field, so same time on different fields still conflicts
    - Both teams in a game may be from the same club (intra-club match)
    - Both teams contribute to their respective club's slot
    """
    PRIORITY = "strong"
    
    GRADE_ORDER = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
    
    def apply(self, model, X, data) -> int:
        teams = data['teams']
        clubs = data['clubs']
        grades = data['grades']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        # Initialize penalty tracking for soft adjacent-grade constraint
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubGradeAdjacencyConstraint'] = {'weight': 50000, 'penalties': []}
        
        # Adjacent grade pairs
        adj_pairs = [
            (self.GRADE_ORDER[i], self.GRADE_ORDER[i + 1])
            for i in range(len(self.GRADE_ORDER) - 1)
        ]
        
        # Build team-to-club mapping
        team_club = {t.name: t.club.name for t in teams}
        
        # Group vars by (slot, club, grade) where slot = (week, day_slot)
        # NOTE: slot ignores field - same time on different fields still conflicts
        slot_club_grade_vars = defaultdict(list)
        
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            
            # Skip locked weeks
            if key[6] in locked_weeks:
                continue
            
            t1, t2, grade = key[0], key[1], key[2]
            slot = (key[6], key[4])  # week, day_slot - NO field!
            
            t1_club = team_club.get(t1)
            t2_club = team_club.get(t2)
            
            # When clubs differ, add var to both clubs' buckets
            # When clubs same (intra-club), add var only ONCE to avoid double-counting
            if t1_club and t2_club:
                if t1_club != t2_club:
                    # Different clubs: add to both
                    slot_club_grade_vars[(slot, t1_club, grade)].append(var)
                    slot_club_grade_vars[(slot, t2_club, grade)].append(var)
                else:
                    # Same club (intra-club match): add only once
                    slot_club_grade_vars[(slot, t1_club, grade)].append(var)
        
        # Constraint 2 (SOFT): Adjacent grades - penalize overlaps but allow them
        # penalty = max(0, sum(g1) + sum(g2) - 1) for each slot
        adj_idx = 0
        for club in [c.name for c in clubs]:
            # Get all slots where this club has games
            club_slots = {k[0] for k in slot_club_grade_vars if k[1] == club}
            
            for slot in club_slots:
                for g1, g2 in adj_pairs:
                    vars_g1 = slot_club_grade_vars.get((slot, club, g1), [])
                    vars_g2 = slot_club_grade_vars.get((slot, club, g2), [])
                    
                    if vars_g1 and vars_g2:
                        max_possible = len(vars_g1) + len(vars_g2)
                        combined = model.NewIntVar(0, max_possible, f'adj_ai_combined_{adj_idx}')
                        model.Add(combined == sum(vars_g1) + sum(vars_g2))
                        
                        penalty = model.NewIntVar(0, max_possible, f'adj_ai_penalty_{adj_idx}')
                        model.AddMaxEquality(penalty, [combined - 1, model.NewConstant(0)])
                        data['penalties']['ClubGradeAdjacencyConstraint']['penalties'].append(penalty)
                        adj_idx += 1
                        constraints_added += 1
        
        return constraints_added


# ============== Club vs Club Alignment ==============

class ClubVsClubAlignmentAI(ConstraintAI):
    """
    Align games between clubs across grades.
    When club pair games coincide on a round, Sunday games must be on the same field.
    
    Enhanced version:
    - Cleaner grade ordering
    - Better coincidence detection
    - Sunday field alignment (ported from original)
    """
    PRIORITY = "soft"
    
    def apply(self, model, X, data) -> int:
        grades = data['grades']
        teams = data['teams']
        num_rounds = data['num_rounds']
        
        constraints_added = 0
        
        # Calculate games per team per grade
        per_team_games = {
            g.name: (num_rounds['max'] // (g.num_teams - 1)) if g.num_teams > 1 and g.num_teams % 2 == 0 
                    else (num_rounds['max'] // g.num_teams) if g.num_teams > 0 
                    else 0
            for g in grades
        }
        
        # Sort grades by number of games (ascending)
        ordered_grades = sorted(per_team_games.items(), key=lambda x: x[1])
        
        # Group games by (grade, club_pair, round) and track Sunday field usage
        grade_club_round_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        fields_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            
            t1, t2, grade = key[0], key[1], key[2]
            round_no = key[8]
            day = key[3]
            field_name = key[9]
            
            if grade in ['PHL', '2nd']:  # Skip top grades
                continue
            
            club1 = get_club(t1, teams)
            club2 = get_club(t2, teams)
            club_pair = tuple(sorted((club1, club2)))
            
            grade_club_round_vars[grade][club_pair][round_no].append(var)
            
            # Track Sunday field usage for field alignment
            if day == 'Sunday':
                fields_dict[club_pair][round_no][field_name].append(var)
        
        # Build alignment constraints
        processed_grades = []
        prev_num = 0
        
        for grade, num_games in ordered_grades:
            if grade in ['PHL', '2nd']:
                continue
            
            processed_grades.append(grade)
            
            if num_games <= prev_num:
                continue
            prev_num = num_games
            
            # For each other grade with more games
            for other_grade in [g for g, n in ordered_grades if g not in processed_grades]:
                for club_pair, rounds in grade_club_round_vars[grade].items():
                    other_rounds = grade_club_round_vars[other_grade].get(club_pair, {})
                    
                    coincide_vars = []
                    for round_no, vars_list in rounds.items():
                        if round_no not in other_rounds:
                            continue
                        
                        # Create indicators for both grades having a game in this round
                        ind1 = model.NewBoolVar(f"ai_g1_{grade}_{club_pair}_{round_no}")
                        model.AddMaxEquality(ind1, vars_list)
                        
                        ind2 = model.NewBoolVar(f"ai_g2_{other_grade}_{club_pair}_{round_no}")
                        model.AddMaxEquality(ind2, other_rounds[round_no])
                        
                        # Coincidence indicator
                        coincide = model.NewBoolVar(f"ai_coin_{club_pair}_{round_no}")
                        model.Add(coincide <= ind1)
                        model.Add(coincide <= ind2)
                        model.Add(coincide >= ind1 + ind2 - 1)
                        
                        coincide_vars.append(coincide)
                        
                        # Sunday field alignment: when coinciding, all Sunday games on same field
                        field_round_vars = fields_dict[club_pair].get(round_no, {})
                        if field_round_vars:
                            field_indicators = []
                            for field_name, game_vars in field_round_vars.items():
                                fi = model.NewBoolVar(f"ai_fld_{club_pair}_{round_no}_{field_name}")
                                model.AddMaxEquality(fi, game_vars)
                                field_indicators.append(fi)
                            
                            # When coinciding, exactly 1 field should be used
                            if field_indicators:
                                model.Add(sum(field_indicators) == 1).OnlyEnforceIf(coincide)
                                constraints_added += 1
                    
                    if coincide_vars:
                        # Get slack from config (--slack flag)
                        config_slack = data.get('constraint_slack', {}).get('ClubVsClubAlignment', 0)
                        min_required = max(0, num_games - config_slack)
                        model.Add(sum(coincide_vars) >= min_required)
                        constraints_added += 1
        
        return constraints_added


# ============== Soft Constraints with Penalties ==============

class MaitlandHomeGroupingAI(ConstraintAI):
    """
    Encourage Maitland games to be grouped as all home or all away per week.
    Hard constraint: No back-to-back home weekends.
    """
    PRIORITY = "soft"
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MaitlandHomeGrouping'] = {'weight': 1000000, 'penalties': []}
        
        # Allow slack override from config (--slack flag)
        slack = data.get('constraint_slack', {}).get('MaitlandHomeGrouping', 0)
        back_to_back_limit = 1 + slack
        
        timeslots = data['timeslots']
        games = data['games']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        # Group Maitland games by week
        week_home_vars = defaultdict(list)
        week_all_vars = defaultdict(list)
        
        for t in timeslots:
            if t.week in locked_weeks or not t.day:
                continue
            
            for (t1, t2, grade) in games:
                if 'Maitland' not in t1 and 'Maitland' not in t2:
                    continue
                
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key not in X:
                    continue
                
                var = X[key]
                week_all_vars[t.week].append(var)
                
                if t.field.location == 'Maitland Park':
                    week_home_vars[t.week].append(var)
        
        # Create home week indicators and add penalties
        home_indicators = {}
        
        for week in sorted(week_all_vars.keys()):
            all_vars = week_all_vars[week]
            home_vars = week_home_vars.get(week, [])
            
            if not all_vars:
                continue
            
            max_games = len(all_vars)
            
            # Home games count
            home_count = model.NewIntVar(0, max_games, f'mait_home_{week}')
            if home_vars:
                model.Add(home_count == sum(home_vars))
            else:
                model.Add(home_count == 0)
            
            # Away games count
            away_count = model.NewIntVar(0, max_games, f'mait_away_{week}')
            model.Add(away_count == sum(all_vars) - home_count)
            
            # Penalty: min(home, away) - encourages all home or all away
            penalty = model.NewIntVar(0, max_games, f'mait_penalty_{week}')
            model.AddMinEquality(penalty, [home_count, away_count])
            data['penalties']['MaitlandHomeGrouping']['penalties'].append(penalty)
            
            # Home week indicator
            home_ind = model.NewBoolVar(f'mait_home_ind_{week}')
            if home_vars:
                model.AddMaxEquality(home_ind, home_vars)
            else:
                model.Add(home_ind == 0)
            home_indicators[week] = home_ind
        
        # Hard constraint: No back-to-back home weekends
        sorted_weeks = sorted(home_indicators.keys())
        for i in range(1, len(sorted_weeks)):
            prev_week = sorted_weeks[i - 1]
            curr_week = sorted_weeks[i]
            
            model.Add(home_indicators[prev_week] + home_indicators[curr_week] <= back_to_back_limit)
            constraints_added += 1
        
        return constraints_added


class AwayAtMaitlandGroupingAI(ConstraintAI):
    """
    Limit away clubs visiting Maitland per weekend.
    Hard limit: Max 3 away clubs per weekend.
    Soft penalty: Encourage fewer clubs.
    """
    PRIORITY = "soft"
    
    HARD_LIMIT = 3
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['AwayAtMaitlandGrouping'] = {'weight': 100000, 'penalties': []}
        
        # Allow slack override from config (--slack flag)
        slack = data.get('constraint_slack', {}).get('AwayAtMaitlandGrouping', 0)
        hard_limit = self.HARD_LIMIT + slack
        
        teams = data['teams']
        timeslots = data['timeslots']
        games = data['games']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        # Track away clubs per week at Maitland
        week_away_club_vars = defaultdict(lambda: defaultdict(list))
        
        for t in timeslots:
            if t.week in locked_weeks or not t.day:
                continue
            
            if t.field.location != 'Maitland Park':
                continue
            
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key not in X:
                    continue
                
                # Find away club
                for team in [t1, t2]:
                    if 'Maitland' not in team:
                        club = get_club(team, teams)
                        week_away_club_vars[t.week][club].append(X[key])
        
        # Add constraints per week
        for week, club_vars in week_away_club_vars.items():
            # Create presence indicators for each club
            club_indicators = []
            
            for club, vars_list in club_vars.items():
                indicator = model.NewBoolVar(f'away_{club}_{week}')
                model.AddMaxEquality(indicator, vars_list)
                club_indicators.append(indicator)
            
            if not club_indicators:
                continue
            
            # Count clubs
            num_clubs = model.NewIntVar(0, len(club_indicators), f'num_away_{week}')
            model.Add(num_clubs == sum(club_indicators))
            
            # Hard limit
            model.Add(num_clubs <= hard_limit)
            constraints_added += 1
            
            # Soft penalty
            more_than_one = model.NewBoolVar(f'gt1_{week}')
            model.Add(num_clubs > 1).OnlyEnforceIf(more_than_one)
            model.Add(num_clubs <= 1).OnlyEnforceIf(more_than_one.Not())
            
            penalty = model.NewIntVar(0, len(club_indicators), f'away_penalty_{week}')
            model.Add(penalty == num_clubs - 1).OnlyEnforceIf(more_than_one)
            model.Add(penalty == 0).OnlyEnforceIf(more_than_one.Not())
            
            data['penalties']['AwayAtMaitlandGrouping']['penalties'].append(penalty)
        
        return constraints_added


class MaximiseClubsPerTimeslotBroadmeadowAI(ConstraintAI):
    """
    Maximize club diversity within timeslots at Broadmeadow.
    Hard minimum: at least total_games/2 clubs must be present.
    Penalty: total_teams - num_clubs
    """
    PRIORITY = "soft"
    
    HARD_LIMIT = 0  # Added to dynamic hard minimum calculation
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MaximiseClubsPerTimeslotBroadmeadow'] = {'weight': 5000, 'penalties': []}
        
        # Allow slack override from config (--slack flag)
        # Negative because slack DECREASES the minimum requirement
        slack = data.get('constraint_slack', {}).get('MaximiseClubsPerTimeslotBroadmeadow', 0)
        hard_limit_offset = self.HARD_LIMIT - slack
        
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        
        # Group by (week, day_slot) at Broadmeadow on weekends
        slot_club_vars = defaultdict(lambda: defaultdict(list))
        
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            
            if key[6] in locked_weeks:
                continue
            
            if key[10] != 'Newcastle International Hockey Centre':
                continue
            
            if key[3] not in ['Saturday', 'Sunday']:
                continue
            
            slot = (key[6], key[4])  # week, day_slot
            
            club1 = get_club(key[0], teams)
            club2 = get_club(key[1], teams)
            
            slot_club_vars[slot][club1].append(var)
            slot_club_vars[slot][club2].append(var)
        
        constraints_added = 0
        
        for slot, club_vars in slot_club_vars.items():
            # Club presence indicators
            presence_vars = []
            all_game_vars = []
            
            for club, vars_list in club_vars.items():
                indicator = model.NewBoolVar(f'ai_club_{club}_{slot}')
                model.Add(sum(vars_list) >= 1).OnlyEnforceIf(indicator)
                model.Add(sum(vars_list) == 0).OnlyEnforceIf(indicator.Not())
                presence_vars.append(indicator)
                all_game_vars.extend(vars_list)
            
            if not presence_vars:
                continue
            
            # Count clubs and total games
            num_clubs = model.NewIntVar(0, len(presence_vars), f'ai_nclubs_{slot}')
            model.Add(num_clubs == sum(presence_vars))
            
            total_teams = model.NewIntVar(0, len(all_game_vars), f'ai_nteams_{slot}')
            model.Add(total_teams == sum(all_game_vars))
            
            # Slot used indicator
            slot_used = model.NewBoolVar(f'ai_slot_used_{slot}')
            model.Add(total_teams >= 1).OnlyEnforceIf(slot_used)
            model.Add(total_teams == 0).OnlyEnforceIf(slot_used.Not())
            
            # Dynamic hard minimum: at least total_games/2 + HARD_LIMIT clubs
            hard_min_base = model.NewIntVar(0, len(presence_vars), f'ai_hmin_base_{slot}')
            model.AddDivisionEquality(hard_min_base, total_teams, 2)
            
            raw_minimum = model.NewIntVar(-10, len(presence_vars), f'ai_hmin_raw_{slot}')
            model.Add(raw_minimum == hard_min_base + hard_limit_offset)
            hard_minimum = model.NewIntVar(0, len(presence_vars), f'ai_hmin_{slot}')
            model.AddMaxEquality(hard_minimum, [raw_minimum, model.NewConstant(0)])
            
            model.Add(num_clubs >= hard_minimum)
            
            # Soft penalty: total_teams - num_clubs
            penalty = model.NewIntVar(0, len(all_game_vars), f'ai_div_penalty_{slot}')
            model.Add(penalty >= total_teams - num_clubs).OnlyEnforceIf(slot_used)
            model.Add(penalty == 0).OnlyEnforceIf(slot_used.Not())
            
            data['penalties']['MaximiseClubsPerTimeslotBroadmeadow']['penalties'].append(penalty)
            constraints_added += 1
        
        return constraints_added


class MinimiseClubsOnAFieldBroadmeadowAI(ConstraintAI):
    """
    Minimize clubs on each field per day at Broadmeadow.
    Hard limit: Max 5 clubs per field per day.
    Soft penalty: |num_clubs - 2|
    """
    PRIORITY = "soft"
    
    HARD_LIMIT = 5
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MinimiseClubsOnAFieldBroadmeadow'] = {'weight': 5000, 'penalties': []}
        
        # Allow slack override from config (--slack flag)
        slack = data.get('constraint_slack', {}).get('MinimiseClubsOnAFieldBroadmeadow', 0)
        hard_limit = self.HARD_LIMIT + slack
        
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        
        # Group by (week, date, field)
        field_club_vars = defaultdict(lambda: defaultdict(list))
        
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            
            if key[6] in locked_weeks:
                continue
            
            if key[10] != 'Newcastle International Hockey Centre':
                continue
            
            if key[3] not in ['Saturday', 'Sunday']:
                continue
            
            field_key = (key[6], key[7], key[9])  # week, date, field
            
            club1 = get_club(key[0], teams)
            club2 = get_club(key[1], teams)
            
            field_club_vars[field_key][club1].append(var)
            field_club_vars[field_key][club2].append(var)
        
        constraints_added = 0
        
        for field_key, club_vars in field_club_vars.items():
            # Club presence indicators
            presence_vars = []
            
            for club, vars_list in club_vars.items():
                indicator = model.NewBoolVar(f'field_club_{club}_{field_key}')
                model.AddBoolOr(vars_list).OnlyEnforceIf(indicator)
                model.AddBoolAnd([v.Not() for v in vars_list]).OnlyEnforceIf(indicator.Not())
                presence_vars.append(indicator)
            
            if not presence_vars:
                continue
            
            # Count clubs
            num_clubs = model.NewIntVar(0, len(presence_vars), f'field_nclubs_{field_key}')
            model.Add(num_clubs == sum(presence_vars))
            
            # Hard limit
            model.Add(num_clubs <= hard_limit)
            constraints_added += 1
            
            # Penalty
            penalty = model.NewIntVar(0, len(presence_vars), f'field_penalty_{field_key}')
            model.AddAbsEquality(penalty, num_clubs - 2)
            
            data['penalties']['MinimiseClubsOnAFieldBroadmeadow']['penalties'].append(penalty)
        
        return constraints_added


class PreferredTimesConstraintAI(ConstraintAI):
    """
    Apply penalties for scheduling at non-preferred times.
    """
    PRIORITY = "soft"
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['PreferredTimesConstraint'] = {'weight': 10000000, 'penalties': []}
        
        teams = data['teams']
        clubs = data['clubs']
        noplay = data.get('preference_no_play', {})
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())
        
        constraints_added = 0
        
        for club_name, restrictions in noplay.items():
            club_teams = get_teams_from_club(club_name, teams)
            
            for idx, constraint in enumerate(restrictions):
                date = constraint.get('date')
                if not date:
                    continue
                
                week = get_nearest_week_by_date(date, timeslots)
                if week in locked_weeks:
                    continue
                
                # Find matching games
                for key, var in X.items():
                    if len(key) < 11:
                        continue
                    
                    t1, t2 = key[0], key[1]
                    if t1 not in club_teams and t2 not in club_teams:
                        continue
                    
                    # Check all constraint conditions
                    matches = True
                    
                    if 'date' in constraint and key[7] != constraint['date']:
                        matches = False
                    if 'field_location' in constraint and key[10] != constraint['field_location']:
                        matches = False
                    if 'time' in constraint and key[5] != constraint['time']:
                        matches = False
                    if 'team_name' in constraint:
                        if constraint['team_name'] not in (t1, t2):
                            matches = False
                    
                    if matches:
                        penalty = model.NewIntVar(0, 1, f'noplay_{club_name}_{idx}_{hash(key)}')
                        model.Add(penalty == var)
                        data['penalties']['PreferredTimesConstraint']['penalties'].append(penalty)
                        constraints_added += 1
        
        return constraints_added


# ============== Club Game Spread (Closeness) ==============

class ClubGameSpreadAI(ConstraintAI):
    """
    Minimize gaps between a club's games on a given day.
    
    For each (club, week, day):
    1. Count distinct timeslots the club occupies (two teams at the same
       day_slot counts as ONE timeslot used, not two)
    2. Find min and max day_slot used
    3. gaps = (max_slot - min_slot + 1) - timeslots_used
       This gives the number of UNUSED timeslots between their first and last game.
    
    Hard constraint:
        gaps <= HARD_LIMIT + slack (default HARD_LIMIT = 2)
    
    Soft constraint:
        Penalize gaps — ideal is 0 (all games in consecutive slots).
    
    EDGE CASES:
    - Clubs with <= 1 timeslot used have 0 gaps (skipped)
    - Two teams from the same club playing at the same day_slot = 1 timeslot
    - Friday PHL games are a different day — gaps computed per (week, day)
    """
    PRIORITY = "soft"
    
    HARD_LIMIT = 2
    PENALTY_WEIGHT = 5000
    
    def apply(self, model, X, data) -> int:
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        config_slack = data.get('constraint_slack', {}).get('ClubGameSpread', 0)
        
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubGameSpread'] = {
            'weight': self.PENALTY_WEIGHT, 'penalties': []
        }
        
        constraints_added = 0
        hard_limit = self.HARD_LIMIT + config_slack
        
        # Build team-to-club mapping
        team_club = {t.name: t.club.name for t in teams}
        
        # Group X vars by (club, week, day, day_slot) -> list of vars
        # We group to the slot level so we can build one indicator per slot
        club_week_day_slot_vars = defaultdict(list)
        
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            week = key[6]
            if week in locked_weeks:
                continue
            
            day = key[3]
            day_slot = key[4]
            t1, t2 = key[0], key[1]
            
            t1_club = team_club.get(t1)
            t2_club = team_club.get(t2)
            
            if t1_club:
                club_week_day_slot_vars[(t1_club, week, day, day_slot)].append(var)
            if t2_club and t2_club != t1_club:
                club_week_day_slot_vars[(t2_club, week, day, day_slot)].append(var)
        
        # Regroup: (club, week, day) -> sorted list of unique day_slots with their vars
        club_week_day_groups = defaultdict(dict)
        for (club, week, day, day_slot), vars_list in club_week_day_slot_vars.items():
            club_week_day_groups[(club, week, day)][day_slot] = vars_list
        
        for (club, week, day), slots_dict in club_week_day_groups.items():
            unique_slots = sorted(slots_dict.keys())
            
            if len(unique_slots) <= 1:
                continue
            
            min_slot = unique_slots[0]
            max_slot = unique_slots[-1]
            
            # is_active[s] = 1 iff club has at least one game at slot s
            is_active = {}
            for s in unique_slots:
                indicator = model.NewBoolVar(f'cgs_active_{club}_w{week}_{day}_s{s}')
                model.AddMaxEquality(indicator, slots_dict[s])
                is_active[s] = indicator
            
            # num_timeslots_used = sum of active indicators
            num_used = model.NewIntVar(0, len(unique_slots),
                                       f'cgs_used_{club}_w{week}_{day}')
            model.Add(num_used == sum(is_active[s] for s in unique_slots))
            constraints_added += 1
            
            # min_active and max_active day_slots
            min_active = model.NewIntVar(min_slot, max_slot,
                                          f'cgs_min_{club}_w{week}_{day}')
            max_active = model.NewIntVar(min_slot, max_slot,
                                          f'cgs_max_{club}_w{week}_{day}')
            
            for s in unique_slots:
                model.Add(min_active <= s).OnlyEnforceIf(is_active[s])
                model.Add(max_active >= s).OnlyEnforceIf(is_active[s])
                constraints_added += 2
            
            # range_size = max_active - min_active + 1
            range_size = model.NewIntVar(1, max_slot - min_slot + 1,
                                          f'cgs_range_{club}_w{week}_{day}')
            model.Add(range_size == max_active - min_active + 1)
            constraints_added += 1
            
            # gaps = range_size - num_timeslots_used
            max_gaps = max_slot - min_slot  # theoretical maximum
            gaps = model.NewIntVar(0, max_gaps,
                                    f'cgs_gaps_{club}_w{week}_{day}')
            model.Add(gaps == range_size - num_used)
            constraints_added += 1
            
            # Only enforce when club uses >= 2 timeslots
            has_multiple = model.NewBoolVar(f'cgs_multi_{club}_w{week}_{day}')
            model.Add(num_used >= 2).OnlyEnforceIf(has_multiple)
            model.Add(num_used <= 1).OnlyEnforceIf(has_multiple.Not())
            constraints_added += 2
            
            # HARD: gaps <= hard_limit (only when multiple timeslots used)
            model.Add(gaps <= hard_limit).OnlyEnforceIf(has_multiple)
            constraints_added += 1
            
            # SOFT: penalize gaps
            penalty = model.NewIntVar(0, max_gaps,
                                       f'cgs_pen_{club}_w{week}_{day}')
            model.Add(penalty >= gaps).OnlyEnforceIf(has_multiple)
            model.Add(penalty == 0).OnlyEnforceIf(has_multiple.Not())
            data['penalties']['ClubGameSpread']['penalties'].append(penalty)
            constraints_added += 2
        
        return constraints_added


# ============== Constraint Collection ==============

def get_all_ai_constraints() -> List[ConstraintAI]:
    """Return all AI-enhanced constraints."""
    return [
        NoDoubleBookingTeamsConstraintAI(),
        NoDoubleBookingFieldsConstraintAI(),
        EnsureEqualGamesAndBalanceMatchUpsAI(),
        PHLAndSecondGradeAdjacencyAI(),
        PHLAndSecondGradeTimesAI(),
        FiftyFiftyHomeandAwayAI(),
        TeamConflictConstraintAI(),
        MaxMaitlandHomeWeekendsAI(),
        EnsureBestTimeslotChoicesAI(),
        ClubDayConstraintAI(),
        EqualMatchUpSpacingConstraintAI(),
        ClubGradeAdjacencyConstraintAI(),
        ClubVsClubAlignmentAI(),
        MaitlandHomeGroupingAI(),
        AwayAtMaitlandGroupingAI(),
        MaximiseClubsPerTimeslotBroadmeadowAI(),
        MinimiseClubsOnAFieldBroadmeadowAI(),
        PreferredTimesConstraintAI(),
        ClubGameSpreadAI(),
    ]


def get_constraints_by_priority(priority: str = None) -> List[ConstraintAI]:
    """
    Get constraints filtered by priority.
    
    Args:
        priority: One of "required", "strong", "medium", "soft", or None for all.
    """
    all_constraints = get_all_ai_constraints()
    
    if priority is None:
        return all_constraints
    
    return [c for c in all_constraints if c.PRIORITY == priority]


def get_staged_constraints() -> Dict[str, List[ConstraintAI]]:
    """Get constraints organized by stage for staged solving."""
    return {
        'stage1_required': get_constraints_by_priority('required'),
        'stage2_strong': get_constraints_by_priority('strong'),
        'stage3_medium': get_constraints_by_priority('medium'),
        'stage4_soft': get_constraints_by_priority('soft'),
    }
