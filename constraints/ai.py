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
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from dataclasses import dataclass

# Import utility functions
from utils import (
    get_club, get_duplicated_graded_teams, get_teams_from_club,
    get_club_from_clubname, get_nearest_week_by_date, normalize_club_day
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
            lambda k: (k[6], k[3], k[4], k[9]) if len(k) >= 10 else None
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
    
    ADJACENCY_MINUTES = 180  # 3 hours (matches original)
    
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
    MAITLAND = 'Maitland Park'

    def apply(self, model, X, data) -> int:
        # Initialize penalties
        if 'penalties' not in data:
            data['penalties'] = {}
        weights = data.get('penalty_weights', {})
        data['penalties']['phl_preferences'] = {'weight': weights.get('phl_preferences', 10000), 'penalties': []}

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
        friday_maitland_vars = []
        friday_gosford_round_vars = defaultdict(list)  # round_no -> vars
        phl_round1_vars = defaultdict(list)  # team -> vars
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
                        friday_gosford_round_vars[t.round_no].append(var)

                    # Friday at Maitland
                    if t.day == 'Friday' and t.field.location == self.MAITLAND:
                        friday_maitland_vars.append(var)

                    # PHL round 1 participation
                    if t.round_no == 1:
                        phl_round1_vars[t1].append(var)
                        phl_round1_vars[t2].append(var)

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
        
        # Read configurable limits
        defaults = data.get('constraint_defaults', {})
        max_friday_broadmeadow = defaults.get('max_friday_broadmeadow', 3)
        gosford_friday_games = defaults.get('gosford_friday_games', 8)
        maitland_friday_games = defaults.get('maitland_friday_games', 2)

        # Constraint 3: Max Friday night games at Broadmeadow
        if friday_broadmeadow_vars:
            model.Add(sum(friday_broadmeadow_vars) <= max_friday_broadmeadow)
            constraints_added += 1

        # Constraint 4: Exactly N Friday night games at Gosford
        if friday_gosford_vars:
            model.Add(sum(friday_gosford_vars) == gosford_friday_games)
            constraints_added += 1

        # Constraint 4b: Exactly N Friday night games at Maitland (Gosford vs Maitland only)
        if friday_maitland_vars:
            model.Add(sum(friday_maitland_vars) == maitland_friday_games)
            constraints_added += 1

        # Constraint 5: Gosford Friday games in specific rounds
        for round_no, round_vars in friday_gosford_round_vars.items():
            if round_no in [2, 4, 5, 9, 10]:
                model.Add(sum(round_vars) == 1)
                constraints_added += 1

        # Constraint 6: Every PHL team must play in round 1
        phl_teams = [team.name for team in teams if team.grade == 'PHL']
        for phl_team in phl_teams:
            if phl_team in phl_round1_vars:
                model.Add(sum(phl_round1_vars[phl_team]) >= 1)
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
    Ensure games stack from earliest timeslots with no gaps, per location.

    Rule: at a given location, you cannot use slot N on ANY field until slot N-1
    is filled on ALL fields at that location. Each location (Broadmeadow, Maitland,
    Gosford) is treated independently.

    This naturally enforces:
    - Per-field contiguity (no gaps on any single field)
    - Cross-field stacking (all fields fill before moving to next slot)
    - Games pushed to earliest slots (can't skip to late slots)

    Additionally, 7pm (19:00) games incur a soft penalty as the worst timeslot.
    """
    PRIORITY = "medium"
    WORST_TIME = '19:00'  # 7pm is the worst timeslot

    def apply(self, model, X, data) -> int:
        timeslots = data['timeslots']
        games = data['games']
        locked_weeks = data.get('locked_weeks', set())

        constraints_added = 0

        # Group vars by (week, day, location, field_name, day_slot)
        field_slot_vars = defaultdict(list)

        for t in timeslots:
            if t.week in locked_weeks or not t.day:
                continue
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    fs_key = (t.week, t.day, t.field.location, t.field.name, t.day_slot)
                    field_slot_vars[fs_key].append(X[key])

        # Build per-field-slot indicators:
        # (week, day, location) -> {field_name: {day_slot: indicator_var}}
        loc_fields = defaultdict(lambda: defaultdict(dict))

        for fs_key, vars_list in field_slot_vars.items():
            week, day, location, field_name, day_slot = fs_key
            if len(vars_list) == 1:
                indicator = vars_list[0]
            else:
                indicator = model.NewBoolVar(f'ai_fs_{week}_{field_name}_{day_slot}')
                model.AddMaxEquality(indicator, vars_list)
            loc_fields[(week, day, location)][field_name][day_slot] = indicator

        # Stacking constraint: for each consecutive pair of available slots,
        # if ANY field uses the later slot, ALL fields must use the earlier slot.
        # When f == f2 this gives per-field contiguity (no gaps).
        # When f != f2 this gives cross-field stacking (fill row before next).
        for (week, day, location), fields_dict in loc_fields.items():
            field_names = list(fields_dict.keys())

            # Collect all slots that have variables on any field
            all_slots = set()
            for field_slots in fields_dict.values():
                all_slots.update(field_slots.keys())
            sorted_slots = sorted(all_slots)

            if len(sorted_slots) < 2:
                continue

            for i in range(len(sorted_slots) - 1):
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]

                for f in field_names:
                    curr_ind = fields_dict[f].get(curr_slot)
                    if curr_ind is None:
                        continue  # field has no vars at this slot

                    for f2 in field_names:
                        next_ind = fields_dict[f2].get(next_slot)
                        if next_ind is None:
                            continue  # other field has no vars at next slot

                        # If f2 uses next_slot, then f must use curr_slot
                        model.AddImplication(next_ind, curr_ind)
                        constraints_added += 1

        # Soft penalty: 7pm (19:00) is the worst timeslot
        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        penalty_key = 'EnsureBestTimeslotChoices_7pm'
        data['penalties'][penalty_key] = {
            'weight': weights.get(penalty_key, 100_000),
            'penalties': []
        }

        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            if key[6] in locked_weeks:
                continue
            if key[5] == self.WORST_TIME:
                pv = model.NewIntVar(0, 1, f'ai_7pm_penalty_{key[6]}_{key[0]}_{key[1]}')
                model.Add(pv == var)
                data['penalties'][penalty_key]['penalties'].append(pv)
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

        for club_name, raw_value in club_days.items():
            desired_date, opponent = normalize_club_day(raw_value)

            # Validate club exists
            club = get_club_from_clubname(club_name, clubs)
            club_teams = get_teams_from_club(club_name, teams)
            date_str = desired_date.date().strftime('%Y-%m-%d')

            # Validate opponent club exists
            if opponent is not None:
                if opponent.lower() not in [c.name.lower() for c in clubs]:
                    raise ValueError(f'Invalid opponent club name {opponent} in ClubDay Dictionary for {club_name}')

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

            # Constraint 2: Matchup logic (derby or opponent)
            teams_by_grade = defaultdict(list)
            for team in club_teams:
                grade = team.rsplit(' ', 1)[1]
                teams_by_grade[grade].append(team)

            opp_by_grade = defaultdict(list)
            if opponent is not None:
                opp_teams = get_teams_from_club(opponent, teams)
                for team in opp_teams:
                    grade = team.rsplit(' ', 1)[1]
                    opp_by_grade[grade].append(team)

            for grade, host_grade_teams in teams_by_grade.items():
                if opponent is not None and grade in opp_by_grade:
                    # Opponent has teams in this grade: force cross-club matchups
                    opp_grade_teams = opp_by_grade[grade]
                    cross_vars = []
                    for key, var in club_game_vars.items():
                        if key[2] != grade:
                            continue
                        if ((key[0] in host_grade_teams and key[1] in opp_grade_teams)
                                or (key[0] in opp_grade_teams and key[1] in host_grade_teams)):
                            cross_vars.append(var)

                    if cross_vars:
                        required = min(len(host_grade_teams), len(opp_grade_teams))
                        model.Add(sum(cross_vars) >= required)
                        constraints_added += 1
                elif len(host_grade_teams) > 1:
                    # No opponent or opponent has no teams in this grade: derby (intra-club)
                    intra_pairs = list(combinations(host_grade_teams, 2))
                    intra_vars = []

                    for key, var in club_game_vars.items():
                        pair = (key[0], key[1])
                        if pair in intra_pairs or (pair[1], pair[0]) in intra_pairs:
                            intra_vars.append(var)

                    if intra_vars:
                        expected_pairs = len(host_grade_teams) // 2
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
    Spread matchups evenly across rounds using pairwise forbidden gaps
    and sliding window density penalties.

    Equivalent to original EqualMatchUpSpacingConstraint but uses zero
    nonlinear operations (no multiplication, division, max, or abs equality).

    HARD constraint (pairwise forbidden gaps):
        For each matchup pair, for each pair of rounds (r1, r2) where
        0 < r2 - r1 < min_gap: the pair cannot play in both rounds.
        This is mathematically equivalent to enforcing a minimum gap between
        any two meetings, for all values of K (number of meetings).

    SOFT penalty (sliding window density):
        For each matchup pair, slide a window of size `space` across all
        rounds. In each window, penalize having more than 1 meeting.
        This naturally pushes gaps toward >= ideal spacing using only
        O(R - space) IntVars per pair (no BoolVars needed).

    Parameters (same as original):
        - Ideal spacing = T - 2 (play all other opponents before rematch)
        - Floor = T // 2 + 1 (minimum gap can never go below this)
        - spacing_base_slack: configurable in season config (default 0)
        - --slack N: added on top of base_slack to further loosen
        - min_gap = max(T // 2 + 1, ideal - base_slack - config_slack)
    """
    PRIORITY = "medium"

    def apply(self, model, X, data) -> int:
        games = data['games']
        timeslots = data['timeslots']
        R = data['num_rounds']['max']
        grades = {g.name: g.num_teams for g in data['grades']}

        config_slack = data.get('constraint_slack', {}).get('EqualMatchUpSpacingConstraint', 0)
        defaults = data.get('constraint_defaults', {})
        base_slack = defaults.get('spacing_base_slack', 0)
        weights = data.get('penalty_weights', {})

        constraints_added = 0

        if 'penalties' not in data:
            data['penalties'] = {}
        if 'EqualMatchUpSpacing' not in data.get('penalties', {}):
            data['penalties']['EqualMatchUpSpacing'] = {'weight': weights.get('EqualMatchUpSpacing', 5000), 'penalties': []}

        min_gap_per_grade = {}
        space_per_grade = {}
        for name, T in grades.items():
            ideal = T - 2
            floor = min(T // 2, T - 2)
            min_gap_per_grade[name] = max(floor, ideal - base_slack - config_slack)
            space_per_grade[name] = ideal

        # Gather game-vars by (t1, t2, grade, round_no)
        meetings = defaultdict(lambda: defaultdict(list))
        for t in timeslots:
            if not t.day:
                continue
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    meetings[(t1, t2, grade)][t.round_no].append(X[key])

        for (t1, t2, grade), round_map in meetings.items():
            min_gap = min_gap_per_grade[grade]
            space = space_per_grade[grade]

            # Collect rounds that actually have variables for this pair
            active_rounds = sorted(r for r in round_map if round_map[r])

            # --- HARD constraint: pairwise forbidden gaps ---
            # For any two rounds closer than min_gap, the pair cannot play in both.
            for i, r1 in enumerate(active_rounds):
                vars_r1 = round_map[r1]
                for r2 in active_rounds[i + 1:]:
                    gap = r2 - r1
                    if gap >= min_gap:
                        break  # sorted, so all subsequent gaps are larger
                    model.Add(sum(vars_r1) + sum(round_map[r2]) <= 1)
                    constraints_added += 1

            # --- SOFT penalty: sliding window density ---
            # Slide a window of `space` consecutive rounds across the season.
            # In each window, penalize having more than 1 meeting.
            # This pushes the solver toward ideal spacing (gap >= space)
            # using only O(R - space) IntVars per pair.
            if space >= R:
                continue  # window covers entire season, nothing to penalize

            for r_start in range(1, R - space + 2):
                r_end = r_start + space - 1
                # Collect all X vars for this pair within [r_start, r_end]
                window_vars = []
                for r in range(r_start, r_end + 1):
                    if r in round_map:
                        window_vars.extend(round_map[r])

                if len(window_vars) < 2:
                    continue  # can't have > 1 meeting with < 2 vars

                pen = model.NewIntVar(
                    0, len(window_vars),
                    f"ai_eqsp_wpen_{t1}_{t2}_{grade}_w{r_start}")
                model.Add(pen >= sum(window_vars) - 1)
                data['penalties']['EqualMatchUpSpacing']['penalties'].append(pen)
                constraints_added += 1

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
        weights = data.get('penalty_weights', {})
        data['penalties']['ClubGradeAdjacencyConstraint'] = {'weight': weights.get('ClubGradeAdjacencyConstraint', 50000), 'penalties': []}
        
        # Adjacent grade pairs
        adj_pairs = [
            (self.GRADE_ORDER[i], self.GRADE_ORDER[i + 1])
            for i in range(len(self.GRADE_ORDER) - 1)
        ]
        
        # Build team-to-club mapping
        team_club = {t.name: t.club.name for t in teams}

        # Precompute duplicate-grade teams per club
        club_dup_grades = defaultdict(lambda: defaultdict(list))
        for club in [c.name for c in clubs]:
            for grade in [g.name for g in grades]:
                dup_teams = get_duplicated_graded_teams(club, grade, teams)
                if dup_teams:
                    club_dup_grades[club][grade].extend(dup_teams)

        # Group vars by (slot, club, grade) where slot = (week, day_slot)
        # NOTE: slot ignores field - same time on different fields still conflicts
        slot_club_grade_vars = defaultdict(list)
        # Track duplicate-grade team games for hard constraint
        club_dup_games = defaultdict(list)

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
                    slot_club_grade_vars[(slot, t1_club, grade)].append(var)
                    slot_club_grade_vars[(slot, t2_club, grade)].append(var)
                    # Track games involving duplicate-grade teams
                    if t1 in club_dup_grades[t1_club].get(grade, []):
                        club_dup_games[(t1_club, slot, grade)].append(var)
                    if t2 in club_dup_grades[t2_club].get(grade, []):
                        club_dup_games[(t2_club, slot, grade)].append(var)
                else:
                    # Same club (intra-club match): add only once
                    slot_club_grade_vars[(slot, t1_club, grade)].append(var)

        # Constraint 1 (HARD): Duplicate teams in same grade can't play simultaneously
        for (club, slot, grade), vars_ in club_dup_games.items():
            if vars_:
                model.Add(sum(vars_) <= 1)
                constraints_added += 1

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
        weights = data.get('penalty_weights', {})
        COINCIDE_PENALTY_WEIGHT = weights.get('ClubVsClubAlignment', 100000)
        FIELD_PENALTY_WEIGHT = weights.get('ClubVsClubAlignmentField', 50000)

        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubVsClubAlignment'] = {'weight': COINCIDE_PENALTY_WEIGHT, 'penalties': []}
        data['penalties']['ClubVsClubAlignmentField'] = {'weight': FIELD_PENALTY_WEIGHT, 'penalties': []}

        grades = data['grades']
        teams = data['teams']
        num_rounds = data['num_rounds']

        constraints_added = 0
        field_penalty_idx = 0

        # Calculate games per team per grade
        per_team_games = {
            g.name: (num_rounds['max'] // (g.num_teams - 1)) if g.num_teams > 1 and g.num_teams % 2 == 0 
                    else (num_rounds['max'] // g.num_teams) if g.num_teams > 0 
                    else 0
            for g in grades
        }
        
        # Sort grades by number of games (ascending)
        ordered_grades = sorted(per_team_games.items(), key=lambda x: x[1])
        
        locked_weeks = data.get('locked_weeks', set())

        # Group games by (grade, club_pair, round) and track Sunday field usage
        # Skip locked weeks — this is an optimization constraint and locked rounds
        # may not satisfy it (they were potentially solved without it).
        grade_club_round_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        fields_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            if key[6] in locked_weeks:
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
                        
                        # Sunday field alignment: when coinciding, limit fields used
                        field_round_vars = fields_dict[club_pair].get(round_no, {})
                        if field_round_vars:
                            field_indicators = []
                            for field_name, game_vars in field_round_vars.items():
                                fi = model.NewBoolVar(f"ai_fld_{club_pair}_{round_no}_{field_name}")
                                model.AddMaxEquality(fi, game_vars)
                                field_indicators.append(fi)

                            if field_indicators:
                                # HARD: max 2 fields when coinciding
                                num_fields_used = model.NewIntVar(0, len(field_indicators),
                                    f"ai_nflds_{club_pair}_{round_no}_{field_penalty_idx}")
                                model.Add(num_fields_used == sum(field_indicators))
                                model.Add(num_fields_used <= 2).OnlyEnforceIf(coincide)
                                constraints_added += 1

                                # SOFT: penalize using 2 fields (prefer 1)
                                field_excess = model.NewIntVar(0, len(field_indicators),
                                    f"ai_fexcess_{club_pair}_{round_no}_{field_penalty_idx}")
                                model.Add(field_excess >= num_fields_used - 1).OnlyEnforceIf(coincide)
                                model.Add(field_excess == 0).OnlyEnforceIf(coincide.Not())
                                data['penalties']['ClubVsClubAlignmentField']['penalties'].append(field_excess)
                                field_penalty_idx += 1

                    if coincide_vars:
                        # Get slack from config (--slack flag)
                        config_slack = data.get('constraint_slack', {}).get('ClubVsClubAlignment', 0)
                        min_required = max(0, num_games - config_slack)

                        # HARD: at least min_required coincidences
                        model.Add(sum(coincide_vars) >= min_required)
                        constraints_added += 1

                        # SOFT: penalize each miss below num_games target
                        actual_coincidences = model.NewIntVar(0, len(coincide_vars),
                            f"ai_actual_coin_{club_pair}_{grade}_{other_grade}")
                        model.Add(actual_coincidences == sum(coincide_vars))
                        coincide_deficit = model.NewIntVar(0, num_games,
                            f"ai_coin_deficit_{club_pair}_{grade}_{other_grade}")
                        model.Add(coincide_deficit >= num_games - actual_coincidences)
                        data['penalties']['ClubVsClubAlignment']['penalties'].append(coincide_deficit)
        
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
        weights = data.get('penalty_weights', {})
        data['penalties']['MaitlandHomeGrouping'] = {'weight': weights.get('MaitlandHomeGrouping', 1000000), 'penalties': []}

        defaults = data.get('constraint_defaults', {})
        base_max = defaults.get('maitland_max_consecutive_home', 1)
        slack = data.get('constraint_slack', {}).get('MaitlandHomeGrouping', 0)
        back_to_back_limit = base_max + slack

        timeslots = data['timeslots']
        games = data['games']
        locked_weeks = data.get('locked_weeks', set())

        constraints_added = 0

        # Group Maitland games by week — include ALL weeks (locked and unlocked)
        # so the back-to-back hard constraint sees locked weeks' home status.
        week_home_vars = defaultdict(list)
        week_all_vars = defaultdict(list)

        for t in timeslots:
            if not t.day:
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

        # Create home week indicators for ALL weeks (locked + unlocked)
        # but only add penalty variables for UNLOCKED weeks.
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

            # Penalty and away count only for unlocked weeks (can't optimise locked weeks)
            if week not in locked_weeks:
                away_count = model.NewIntVar(0, max_games, f'mait_away_{week}')
                model.Add(away_count == sum(all_vars) - home_count)

                penalty = model.NewIntVar(0, max_games, f'mait_penalty_{week}')
                model.AddMinEquality(penalty, [home_count, away_count])
                data['penalties']['MaitlandHomeGrouping']['penalties'].append(penalty)

            # Home week indicator — needed for ALL weeks (back-to-back constraint)
            # For locked weeks the variables are already fixed (0 or 1),
            # so the indicator is deterministic.
            home_ind = model.NewBoolVar(f'mait_home_ind_{week}')
            if home_vars:
                model.AddMaxEquality(home_ind, home_vars)
            else:
                model.Add(home_ind == 0)
            home_indicators[week] = home_ind

        # Hard constraint: Max N consecutive home weekends (sliding window)
        # slack=0: max 1 consecutive (no back-to-back)
        # slack=1: max 2 consecutive, slack=2: max 3, etc.
        max_consecutive = back_to_back_limit  # already 1 + slack
        window_size = max_consecutive + 1
        sorted_weeks = sorted(home_indicators.keys())
        for i in range(len(sorted_weeks) - window_size + 1):
            window_indicators = [home_indicators[sorted_weeks[j]] for j in range(i, i + window_size)]
            model.Add(sum(window_indicators) <= max_consecutive)
            constraints_added += 1

        return constraints_added


class AwayAtMaitlandGroupingAI(ConstraintAI):
    """
    Limit away clubs visiting Maitland per weekend.
    Hard limit: Max 3 away clubs per weekend.
    Soft penalty: Encourage fewer clubs.
    """
    PRIORITY = "soft"
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        weights = data.get('penalty_weights', {})
        data['penalties']['AwayAtMaitlandGrouping'] = {'weight': weights.get('AwayAtMaitlandGrouping', 100000), 'penalties': []}

        defaults = data.get('constraint_defaults', {})
        base_limit = defaults.get('away_maitland_max_clubs', 3)
        slack = data.get('constraint_slack', {}).get('AwayAtMaitlandGrouping', 0)
        hard_limit = base_limit + slack

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
        weights = data.get('penalty_weights', {})
        data['penalties']['MaximiseClubsPerTimeslotBroadmeadow'] = {'weight': weights.get('MaximiseClubsPerTimeslotBroadmeadow', 5000), 'penalties': []}
        
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
    
    def apply(self, model, X, data) -> int:
        if 'penalties' not in data:
            data['penalties'] = {}
        weights = data.get('penalty_weights', {})
        data['penalties']['MinimiseClubsOnAFieldBroadmeadow'] = {'weight': weights.get('MinimiseClubsOnAFieldBroadmeadow', 5000), 'penalties': []}

        defaults = data.get('constraint_defaults', {})
        base_limit = defaults.get('max_clubs_per_field', 5)
        slack = data.get('constraint_slack', {}).get('MinimiseClubsOnAFieldBroadmeadow', 0)
        hard_limit = base_limit + slack
        
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
    Supports both 2025 format ({club: [restrictions]}) and
    2026 format ({entry_key: {club, dates, grade}}).
    """
    PRIORITY = "soft"

    def apply(self, model, X, data) -> int:
        from constraints.original import _normalize_preference_no_play

        if 'penalties' not in data:
            data['penalties'] = {}
        weights = data.get('penalty_weights', {})
        data['penalties']['PreferredTimesConstraint'] = {'weight': weights.get('PreferredTimesConstraint', 10000000), 'penalties': []}

        teams = data['teams']
        clubs = data['clubs']
        noplay = data.get('preference_no_play', {})
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())

        if not noplay:
            return 0

        constraints_added = 0

        # Keys for matching game tuples to restriction dicts
        allowed_keys = ['team_name', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        allowed_keys2 = ['team1', 'team_name', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']

        # Normalize both formats to consistent structure
        normalized = _normalize_preference_no_play(noplay, teams, clubs)

        for entry_key, club_name, club_teams, constraint in normalized:
            if 'date' not in constraint:
                continue

            if get_nearest_week_by_date(constraint['date'], timeslots) in locked_weeks:
                continue

            for i, game_key in enumerate(X):
                if len(game_key) < 11:
                    continue

                # Check if any club team is in this game
                if game_key[0] not in club_teams and game_key[1] not in club_teams:
                    continue

                # Try matching with both key orderings
                game_dict = dict(zip(allowed_keys, game_key))
                game_dict2 = dict(zip(allowed_keys2, game_key))

                matches = all(game_dict.get(k) == v for k, v in constraint.items())
                matches2 = all(game_dict2.get(k) == v for k, v in constraint.items())

                if matches or matches2:
                    penalty_var = model.NewIntVar(0, 1, f"ai_penalty_{entry_key}_{i}")
                    model.Add(penalty_var == X[game_key])
                    data['penalties']['PreferredTimesConstraint']['penalties'].append(penalty_var)
                    constraints_added += 1

        return constraints_added


# ============== Club Game Spread (Closeness) ==============

class ClubGameSpreadAI(ConstraintAI):
    """
    Minimize gaps between a club's games on a given day, and limit double-ups.

    For each (club, week, day):
    1. Count total games the club has scheduled (num_games)
    2. Find min and max day_slot used
    3. gap = (max_slot - min_slot + 1) - num_games
       Positive gap = unused slots in the range (games are spread out).
       Negative gap = double-ups (more games than slots in the range).
       Zero = games perfectly fill consecutive slots.

    Hard constraints (only when club has >= 2 games on that day):
        UPPER: gap <= max_gap + slack          (limits spread, default max_gap=2)
        LOWER: gap >= -(max_overlap + slack)   (limits double-ups, default max_overlap=0)

    With max_overlap=0 and slack=0:
        gap >= 0 means range >= num_games, so no double-ups allowed.
        A 4-team club needs at least 4 distinct slots.

    Config params (in constraint_defaults):
        club_game_spread_max_gap:     upper bound base (default 2)
        club_game_spread_max_overlap: lower bound base (default 0, no double-ups)

    Soft constraint (time):
        Penalize |gap| — ideal is 0 (all games in consecutive slots).
        Both spread (positive gap) and double-ups (negative gap) are penalized.

    Field concentration (per club, week, day):
        field_spread = num_games - max_games_on_any_single_field
        Hard: field_spread <= max(0, num_games // 2 - 1) + slack
        Soft: penalize field_spread (ideal = 0, all games on one field).

    EDGE CASES:
    - Clubs with <= 1 game on a day are skipped
    - Two teams from the same club at the same day_slot counts as 2 games
      (not 1), making gap more negative — penalized by lower bound
    - Friday PHL games are a different day — gaps computed per (week, day)
    - Single-slot case: if all possible games are at one slot, range=1,
      gap=1-num_games; lower bound still enforced
    """
    PRIORITY = "soft"

    def apply(self, model, X, data) -> int:
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        defaults = data.get('constraint_defaults', {})
        max_gap_base = defaults.get('club_game_spread_max_gap', 2)
        max_overlap_base = defaults.get('club_game_spread_max_overlap', 0)
        config_slack = data.get('constraint_slack', {}).get('ClubGameSpread', 0)
        weights = data.get('penalty_weights', {})

        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubGameSpread'] = {
            'weight': weights.get('ClubGameSpread', 5000), 'penalties': []
        }
        data['penalties']['ClubFieldConcentration'] = {
            'weight': weights.get('ClubFieldConcentration', 5000), 'penalties': []
        }

        constraints_added = 0
        hard_upper = max_gap_base + config_slack
        hard_lower = -(max_overlap_base + config_slack)

        # Build team-to-club mapping
        team_club = {t.name: t.club.name for t in teams}

        # Group X vars by (club, week, day, day_slot) -> list of vars
        # Also group by (club, week, day, field) for field concentration
        club_week_day_slot_vars = defaultdict(list)
        club_week_day_field_vars = defaultdict(list)

        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            week = key[6]
            if week in locked_weeks:
                continue

            day = key[3]
            day_slot = key[4]
            field_key = (key[9], key[10])  # (field_name, field_location)
            t1, t2 = key[0], key[1]

            t1_club = team_club.get(t1)
            t2_club = team_club.get(t2)

            if t1_club:
                club_week_day_slot_vars[(t1_club, week, day, day_slot)].append(var)
                club_week_day_field_vars[(t1_club, week, day, field_key)].append(var)
            if t2_club and t2_club != t1_club:
                club_week_day_slot_vars[(t2_club, week, day, day_slot)].append(var)
                club_week_day_field_vars[(t2_club, week, day, field_key)].append(var)

        # Regroup: (club, week, day) -> {day_slot: [vars]}
        club_week_day_groups = defaultdict(dict)
        for (club, week, day, day_slot), vars_list in club_week_day_slot_vars.items():
            club_week_day_groups[(club, week, day)][day_slot] = vars_list

        # Regroup: (club, week, day) -> {field: [vars]}
        club_week_day_field_groups = defaultdict(dict)
        for (club, week, day, field_key), vars_list in club_week_day_field_vars.items():
            club_week_day_field_groups[(club, week, day)][field_key] = vars_list

        for (club, week, day), slots_dict in club_week_day_groups.items():
            unique_slots = sorted(slots_dict.keys())

            if len(unique_slots) <= 1:
                # Single slot: range=1, gap=1-num_games.
                # Still enforce lower bound to prevent excessive double-ups.
                all_vars = slots_dict[unique_slots[0]]
                if len(all_vars) < 2:
                    continue

                num_games = model.NewIntVar(0, len(all_vars),
                                            f'cgs_ng_{club}_w{week}_{day}')
                model.Add(num_games == sum(all_vars))
                constraints_added += 1

                has_multiple = model.NewBoolVar(f'cgs_multi_{club}_w{week}_{day}')
                model.Add(num_games >= 2).OnlyEnforceIf(has_multiple)
                model.Add(num_games <= 1).OnlyEnforceIf(has_multiple.Not())
                constraints_added += 2

                # gap = 1 - num_games >= hard_lower => num_games <= 1 - hard_lower
                max_allowed = 1 - hard_lower
                model.Add(num_games <= max_allowed).OnlyEnforceIf(has_multiple)
                constraints_added += 1
                continue

            min_slot = unique_slots[0]
            max_slot = unique_slots[-1]

            # Collect ALL vars for the club on this (week, day)
            all_vars_for_day = []
            for s in unique_slots:
                all_vars_for_day.extend(slots_dict[s])

            # is_active[s] = 1 iff club has at least one game at slot s
            is_active = {}
            for s in unique_slots:
                indicator = model.NewBoolVar(f'cgs_active_{club}_w{week}_{day}_s{s}')
                model.AddMaxEquality(indicator, slots_dict[s])
                is_active[s] = indicator

            # num_games = total games for this club on this day
            num_games = model.NewIntVar(0, len(all_vars_for_day),
                                        f'cgs_ng_{club}_w{week}_{day}')
            model.Add(num_games == sum(all_vars_for_day))
            constraints_added += 1

            # min_active and max_active day_slots (exact, not just bounds)
            # Use AddMinEquality/AddMaxEquality with sentinel values for inactive slots
            min_active = model.NewIntVar(min_slot, max_slot,
                                          f'cgs_min_{club}_w{week}_{day}')
            max_active = model.NewIntVar(min_slot, max_slot,
                                          f'cgs_max_{club}_w{week}_{day}')

            # For min: inactive slots get sentinel max_slot+1 (won't be chosen as min)
            # For max: inactive slots get sentinel min_slot-1 (won't be chosen as max)
            min_candidates = []
            max_candidates = []
            for s in unique_slots:
                # min candidate: s if active, else max_slot (high sentinel)
                mc = model.NewIntVar(min_slot, max_slot,
                                      f'cgs_minc_{club}_w{week}_{day}_s{s}')
                model.Add(mc == s).OnlyEnforceIf(is_active[s])
                model.Add(mc == max_slot).OnlyEnforceIf(is_active[s].Not())
                min_candidates.append(mc)

                # max candidate: s if active, else min_slot (low sentinel)
                xc = model.NewIntVar(min_slot, max_slot,
                                      f'cgs_maxc_{club}_w{week}_{day}_s{s}')
                model.Add(xc == s).OnlyEnforceIf(is_active[s])
                model.Add(xc == min_slot).OnlyEnforceIf(is_active[s].Not())
                max_candidates.append(xc)
                constraints_added += 4

            model.AddMinEquality(min_active, min_candidates)
            model.AddMaxEquality(max_active, max_candidates)
            constraints_added += 2

            # range_size = max_active - min_active + 1
            range_size = model.NewIntVar(1, max_slot - min_slot + 1,
                                          f'cgs_range_{club}_w{week}_{day}')
            model.Add(range_size == max_active - min_active + 1)
            constraints_added += 1

            # gap = range_size - num_games (can be negative = double-ups)
            max_gap_possible = max_slot - min_slot
            min_gap_possible = 1 - len(all_vars_for_day)
            gap = model.NewIntVar(min_gap_possible, max_gap_possible,
                                   f'cgs_gap_{club}_w{week}_{day}')
            model.Add(gap == range_size - num_games)
            constraints_added += 1

            # Only enforce when club has >= 2 games
            has_multiple = model.NewBoolVar(f'cgs_multi_{club}_w{week}_{day}')
            model.Add(num_games >= 2).OnlyEnforceIf(has_multiple)
            model.Add(num_games <= 1).OnlyEnforceIf(has_multiple.Not())
            constraints_added += 2

            # HARD UPPER: gap <= hard_upper (limits spread)
            model.Add(gap <= hard_upper).OnlyEnforceIf(has_multiple)
            constraints_added += 1

            # HARD LOWER: gap >= hard_lower (limits double-ups)
            model.Add(gap >= hard_lower).OnlyEnforceIf(has_multiple)
            constraints_added += 1

            # SOFT: penalize |gap| — both spread (positive) and double-ups (negative)
            # push toward gap=0 (perfect consecutive fill)
            max_abs = max(max_gap_possible, -min_gap_possible)
            penalty = model.NewIntVar(0, max_abs,
                                       f'cgs_pen_{club}_w{week}_{day}')
            model.Add(penalty >= gap).OnlyEnforceIf(has_multiple)
            model.Add(penalty >= -gap).OnlyEnforceIf(has_multiple)
            model.Add(penalty == 0).OnlyEnforceIf(has_multiple.Not())
            data['penalties']['ClubGameSpread']['penalties'].append(penalty)
            constraints_added += 3

            # --- Field concentration ---
            # field_spread = num_games - max_games_on_any_single_field
            # Ideal = 0 (all games on one field). Hard cap scales with club size.
            field_dict = club_week_day_field_groups.get((club, week, day), {})
            if field_dict:
                # games_on_field[f] = sum of vars for this club on field f this day
                field_game_counts = []
                n_all = len(all_vars_for_day)
                for f_key, f_vars in field_dict.items():
                    fcount = model.NewIntVar(0, len(f_vars),
                                              f'cgs_fc_{club}_w{week}_{day}_{f_key[0]}')
                    model.Add(fcount == sum(f_vars))
                    field_game_counts.append(fcount)
                    constraints_added += 1

                # max_field_games = max games on any single field
                max_field = model.NewIntVar(0, n_all,
                                             f'cgs_maxf_{club}_w{week}_{day}')
                model.AddMaxEquality(max_field, field_game_counts)
                constraints_added += 1

                # field_spread = num_games - max_field_games
                field_spread = model.NewIntVar(0, n_all,
                                                f'cgs_fspread_{club}_w{week}_{day}')
                model.Add(field_spread == num_games - max_field)
                constraints_added += 1

                # Hard cap: field_spread <= max(0, num_games // 2 - 1) + slack
                # Since num_games is a variable, linearize:
                # field_spread <= num_games // 2 - 1 + slack
                # => 2 * field_spread <= num_games - 2 + 2*slack
                # field_spread >= 0 by domain, so small clubs (num_games<2)
                # are handled by has_multiple guard.
                model.Add(2 * field_spread <= num_games - 2 + 2 * config_slack).OnlyEnforceIf(has_multiple)
                constraints_added += 1

                # SOFT: penalize field_spread
                f_penalty = model.NewIntVar(0, n_all,
                                             f'cgs_fpen_{club}_w{week}_{day}')
                model.Add(f_penalty >= field_spread).OnlyEnforceIf(has_multiple)
                model.Add(f_penalty == 0).OnlyEnforceIf(has_multiple.Not())
                data['penalties']['ClubFieldConcentration']['penalties'].append(f_penalty)
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
