# constraints.py
"""
Constraint classes for scheduling system.
"""
from ortools.sat.python import cp_model
from abc import ABC, abstractmethod
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

from utils import (
    get_club, get_duplicated_graded_teams, get_teams_from_club, get_club_from_clubname, get_nearest_week_by_date
)


class Constraint(ABC):
    """Abstract base class for all scheduling constraints."""
    
    @abstractmethod
    def apply(self, model: cp_model.CpModel, X: dict, data: dict):
        """Apply constraint to the OR-Tools model."""
        pass

class NoDoubleBookingTeamsConstraint(Constraint):
    """Ensure no team is scheduled for more than one game per week."""
    
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())
        weekly_games = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    weekly_games[(t.week, t1)].append(X[key])
                    weekly_games[(t.week, t2)].append(X[key])

        for (week, team), game_vars in weekly_games.items():
            if week in locked_weeks:
                continue
            model.Add(sum(game_vars) <= 1)

class NoDoubleBookingFieldsConstraint(Constraint):
    """Ensure no field is scheduled for more than one game per time slot."""
    
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())
        field_usage = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    field_usage[(t.day, t.day_slot, t.week, t.field.name)].append(X[key])

        for slot, game_vars in field_usage.items():
            if slot[2] in locked_weeks:
                continue
            model.Add(sum(game_vars) <= 1)


class EnsureEqualGamesAndBalanceMatchUps(Constraint):
    """Ensure each team plays exactly num_rounds[grade] games,
       and that pair‐matchups are balanced: every pair meets
       either `base` or `base+1` times, never more."""

    def apply(self, model, X, data):
        games = data['games']                    # list of (t1, t2, grade)
        timeslots = data['timeslots']
        num_rounds = data['num_rounds']          # dict: grade -> max games per team
        num_dummy = data.get('num_dummy_timeslots', 0)
        teams = data['teams']                    # list of Team objects

        # collect per‐team and per‐pair vars
        team_games  = defaultdict(lambda: defaultdict(list))
        pair_games  = defaultdict(lambda: defaultdict(list))

        # bucket real + dummy slots 
        for (t1, t2, grade) in games:
            key_base = (t1, t2, grade)
            # real timeslots
            for t in timeslots:
                key = (*key_base,
                       t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no,
                       t.field.name, t.field.location)
                if key in X:
                    v = X[key]
                    team_games[grade][t1].append(v)
                    team_games[grade][t2].append(v)
                    pair_games[grade][tuple(sorted((t1, t2)))].append(v)
            # dummy slots
            for i in range(num_dummy):
                dummy_key = (*key_base, i)
                if dummy_key in X:
                    v = X[dummy_key]
                    team_games[grade][t1].append(v)
                    team_games[grade][t2].append(v)
                    pair_games[grade][tuple(sorted((t1, t2)))].append(v)

        # now enforce per‐team total and per‐pair bounds
        for grade, per_team in team_games.items():
            T = sum(1 for tm in teams if tm.grade == grade)
            R = num_rounds[grade]

            # 1) exactly R games per team
            for tm, vars_ in per_team.items():
                model.Add(sum(vars_) == R)

            # 2) compute base pair‐meets and extras
            if T % 2 == 0:
                base = R // (T - 1)
                capacity_used = base * (T - 1)
            else:
                base = R // T
                capacity_used = base * T

            extras = R - capacity_used

            # 3) bounds for each distinct pair
            for pair, vars_ in pair_games[grade].items():
                # each pair meets at least `base`
                model.Add(sum(vars_) >= base)
                # at most base+1, so extras are spread 1‐per‐pair
                model.Add(sum(vars_) <= base + 1)
                
class PHLAndSecondGradeAdjacency(Constraint):
    '''Ensure that PHL and 2nds do not play in adjacent day_slots at different locations.''' 

    def apply(self, model, X, data):


        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())

        phl_games = defaultdict(lambda: defaultdict(list))

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)

                if grade == 'PHL' and key in X and t.day: # Ensure dummy timeslots not counted
                    club = get_club(t1, data['teams'])
                    dup_2nds = get_duplicated_graded_teams(club, '2nd', data['teams'])
                    for team in dup_2nds:
                        phl_games[(club, team, t.week, t.day)][(t.time, t.field.location)].append(X[key])


                    club = get_club(t2, data['teams'])
                    dup_2nds = get_duplicated_graded_teams(club, '2nd', data['teams'])

                    for team in dup_2nds:
                        phl_games[(club, team, t.week, t.day)][(t.time, t.field.location)].append(X[key])

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)

                if grade == '2nd' and key in X and t.day: # Ensure dummy timeslots not counted
                    club = get_club(t1, data['teams'])
                    for identifier in phl_games[(club, t1, t.week, t.day)]:
                        if identifier[0] != '':
                            base_time = datetime.strptime(identifier[0], '%H:%M') 
                        else:
                            continue

                        min_time = (base_time - timedelta(minutes=180)).time()
                        max_time = (base_time + timedelta(minutes=180)).time()

                        if isinstance(t.time, str):
                            t_time = datetime.strptime(t.time, '%H:%M').time()
                        else:
                            t_time = t.time 

                        if min_time <= t_time <= max_time and identifier[1] != t.field.location:
                            phl_games[(club, t1, t.week, t.day)][identifier].append(X[key])         
                        elif (t_time >= max_time or t_time <=min_time)  and identifier[1] == t.field.location:
                            phl_games[(club, t1, t.week, t.day)][identifier].append(X[key])

                    club = get_club(t2, data['teams'])
                    for identifier in phl_games[(club, t2, t.week, t.day)]:
                        if identifier[0] != '':
                            base_time = datetime.strptime(identifier[0], '%H:%M') 
                        else:
                            continue

                        min_time = (base_time - timedelta(minutes=180)).time()
                        max_time = (base_time + timedelta(minutes=180)).time()

                        if isinstance(t.time, str):
                            t_time = datetime.strptime(t.time, '%H:%M').time()
                        else:
                            t_time = t.time  

                        if min_time <= t_time <= max_time and identifier[1] != t.field.location:
                            phl_games[(club, t2, t.week, t.day)][identifier].append(X[key])  
                        elif (t_time >= max_time or t_time <=min_time)  and identifier[1] == t.field.location:
                            phl_games[(club, t2, t.week, t.day)][identifier].append(X[key])

        for date, day_slots in phl_games.items():
            if date[2] in locked_weeks:
                continue
            for day_slot, game_vars in day_slots.items():
                    model.Add(sum(game_vars) <= 1)    
               
# Has Soft Element
class PHLAndSecondGradeTimes(Constraint):        
    ''' PHL games should not be played at the same time. 2nds games cannot be played at the same time as PHL games within the same club. Ensure max 3 Friday night games at Broadmeadow and enforce preferred dates. 
        So we do not need to enforce this for other locations as there is only one field there.
    '''      
    def apply(self, model, X, data):


        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['phl_preferences'] = {'weight': weights.get('phl_preferences', 10000), 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']

        locked_weeks = data.get('locked_weeks', set())

        phl_preferences = data['phl_preferences']
        allowed_keys = {"preferred_dates"}
        invalid_keys = set(phl_preferences.keys()) - allowed_keys 

        if invalid_keys:
            raise ValueError(f"Invalid keys found: {invalid_keys}, currently do not support any keys other than {allowed_keys}")
   
        team_games = defaultdict(lambda: defaultdict(list))
        club_games = defaultdict(lambda: defaultdict(list))
        friday_games = defaultdict(list)
        friday_games_gosford = defaultdict(list)
        friday_games_maitland = []
        preferred_dates = defaultdict(list)
        phl_round1_games = defaultdict(list)  # Track PHL team games in round 1

        # HACK: Count locked PHL Friday games from locked_keys_set so
        # Friday total constraints are adjusted for already-decided games.
        locked_gosford_fridays = 0
        locked_maitland_fridays = 0
        locked_broadmeadow_fridays = 0
        if locked_weeks:
            for key in data.get('locked_keys_set', set()):
                if len(key) >= 11 and key[2] == 'PHL' and key[3] == 'Friday':
                    loc = key[10]
                    if 'Central Coast' in loc:
                        locked_gosford_fridays += 1
                    elif 'Maitland' in loc:
                        locked_maitland_fridays += 1
                    elif 'Newcastle' in loc:
                        locked_broadmeadow_fridays += 1

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)

                if grade in ['PHL', '2nd'] and key in X and t.day: # Ensure dummy timeslots not counted
                    # HACK: Skip locked weeks for Friday totals and round 1 tracking
                    in_locked = locked_weeks and t.week in locked_weeks

                    if grade == 'PHL':
                        # Stop PHL games across clubs from being played at the same time
                        team_games[(t.week, t.day)][(t.day_slot, t.field.location)].append(X[key])

                        # Stop PHL and 2nd grade from being played at the same time, within the same club
                        club = get_club(t1, data['teams'])
                        dup_2nds = get_duplicated_graded_teams(club, '2nd', data['teams'])
                        for extra_team in dup_2nds:
                            club_games[(t.week, t.day, t.field.location)][(t.day_slot, club, extra_team)].append(X[key])

                        club = get_club(t2, data['teams'])
                        dup_2nds = get_duplicated_graded_teams(club, '2nd', data['teams'])
                        for extra_team in dup_2nds:
                            club_games[(t.week, t.day, t.field.location)][(t.day_slot, club, extra_team)].append(X[key])

                        if not in_locked:
                            # Max 3 Friday night games at Newcastle International Hockey Centre
                            if t.day == 'Friday' and t.field.location == 'Newcastle International Hockey Centre':
                                friday_games['Friday'].append(X[key])

                            # Enforce Gosford Home game number on a Friday Night
                            if t.day == 'Friday' and t.field.location == 'Central Coast Hockey Park':
                                friday_games_gosford[t.round_no].append(X[key])

                            # Friday at Maitland (Gosford vs Maitland only)
                            if t.day == 'Friday' and t.field.location == 'Maitland Park':
                                friday_games_maitland.append(X[key])

                            # Track PHL teams playing in round 1
                            if t.round_no == 1:
                                phl_round1_games[t1].append(X[key])
                                phl_round1_games[t2].append(X[key])

                        # Enforce preferred dates (regardless of locked)
                        if t.date in [d.date().strftime('%Y-%m-%d') for d in phl_preferences['preferred_dates']]:
                            preferred_dates[t.date].append(X[key])                     

                    else:
                        club = get_club(t1, data['teams'])
                        club_games[(t.week, t.day, t.field.location)][(t.day_slot, club, t1)].append(X[key])

                        club = get_club(t2, data['teams'])
                        club_games[(t.week, t.day, t.field.location)][(t.day_slot, club, t2)].append(X[key])

        for date, day_slots in team_games.items():
            if date[0] in locked_weeks:
                continue

            for day_slot, game_vars in day_slots.items():
                if day_slot[1] == 'Newcastle International Hockey Centre': # Stop concurrent PHL games only at Broadmeadow
                    model.Add(sum(game_vars) <= 1)

        for date, day_slots in club_games.items():
            if date[0] in locked_weeks:
                continue

            if date[2] == 'Newcastle International Hockey Centre': # Stop concurrent 2nd grade and PHL only at Broadmeadow
                for day_slot, game_vars in day_slots.items():
                        model.Add(sum(game_vars) <= 1)

        defaults = data.get('constraint_defaults', {})
        max_friday_broadmeadow = defaults.get('max_friday_broadmeadow', 3)
        gosford_friday_games = defaults.get('gosford_friday_games', 8)
        maitland_friday_games = defaults.get('maitland_friday_games', 2)

        # HACK: Adjust Friday targets to account for locked-week games already decided
        adjusted_broadmeadow = max(0, max_friday_broadmeadow - locked_broadmeadow_fridays)
        adjusted_gosford = max(0, gosford_friday_games - locked_gosford_fridays)
        adjusted_maitland = max(0, maitland_friday_games - locked_maitland_fridays)

        # Friday totals — adjusted for locked weeks
        gosford_vars = [v for vs in friday_games_gosford.values() for v in vs]
        print(f"  [PHLAndSecondGradeTimes] Locked Friday counts: Gosford={locked_gosford_fridays}, Maitland={locked_maitland_fridays}, Broadmeadow={locked_broadmeadow_fridays}")
        print(f"  [PHLAndSecondGradeTimes] Adjusted targets: Gosford=={adjusted_gosford}, Maitland=={adjusted_maitland}, Broadmeadow<={adjusted_broadmeadow}")
        print(f"  [PHLAndSecondGradeTimes] Gosford Friday vars: {len(gosford_vars)} across {len(friday_games_gosford)} rounds: {sorted(friday_games_gosford.keys())}")
        print(f"  [PHLAndSecondGradeTimes] Maitland Friday vars: {len(friday_games_maitland)}")
        print(f"  [PHLAndSecondGradeTimes] Broadmeadow Friday vars: {sum(len(v) for v in friday_games.values())}")
        # Testing: Broadmeadow and Maitland enabled, Gosford disabled
        for day, game_vars in friday_games.items():
            model.Add(sum(game_vars) <= adjusted_broadmeadow)
        if friday_games_maitland:
            model.Add(sum(friday_games_maitland) == adjusted_maitland)
        # if gosford_vars:
        #     model.Add(sum(gosford_vars) == adjusted_gosford)

        # Round 1 enforcement — skip when locked (round 1 games already decided)
        if not locked_weeks:
            phl_teams = [team.name for team in data['teams'] if team.grade == 'PHL']
            for phl_team in phl_teams:
                if phl_team in phl_round1_games and phl_round1_games[phl_team]:
                    model.Add(sum(phl_round1_games[phl_team]) >= 1)

        for date, game_vars in preferred_dates.items():
            week_no = get_nearest_week_by_date(date, data['timeslots'])
            if week_no in locked_weeks:
                continue

            penalty_var = model.NewIntVar(0, len(game_vars), f"preferred_date_penalty_{date}")
            model.AddAbsEquality(penalty_var, sum(game_vars) - 1)
            data['penalties']['phl_preferences']['penalties'].append(penalty_var)

class FiftyFiftyHomeandAway(Constraint):
    ''' Push toward 50% of games vs each team played at home and away for each away clubs team. '''
        
    def apply(self, model, X, data):


        games = data['games']
        timeslots = data['timeslots']

        home_games = defaultdict(list)
        away_games = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if ('Maitland' in t1 or 'Maitland' in t2) and key in X and t.day: # Ensure dummy timeslots not counted
                    relevant_team = t1 if 'Maitland' in t1 else t2
                    other_team = t2 if relevant_team == t1 else t1
                    if 'Maitland' in other_team:
                        continue
                    elif t.field.location == 'Maitland Park':
                        home_games[(relevant_team, other_team)].append(X[key])
                    else:
                        away_games[(relevant_team, other_team)].append(X[key])

                if ('Gosford' in t1 or 'Gosford' in t2) and key in X and t.day: # Ensure dummy timeslots not counted
                    relevant_team = t1 if 'Gosford' in t1 else t2
                    other_team = t2 if relevant_team == t1 else t1
                    if 'Gosford' in other_team:
                        continue
                    elif t.field.location == 'Central Coast Hockey Park':
                        home_games[(relevant_team, other_team)].append(X[key])
                    else:
                        away_games[(relevant_team, other_team)].append(X[key])
                        
        for team, home_vars in home_games.items():
            away_vars = away_games[team]
            
            if not home_vars or not away_vars: 
                print(f'Team {team} has no games when calculating 50/50 home and away. Home games {home_vars}, Away games {away_vars}')  
                continue
            
            home_games_count = model.NewIntVar(0, len(home_vars), f'home_games_count_{team}')
            model.Add(home_games_count == sum(home_vars))
            
            total_games_count = model.NewIntVar(0, len(home_vars) + len(away_vars), f'total_games_count_{team}')
            model.Add(total_games_count == home_games_count + sum(away_vars))

            # Ensure aim_games is as close to 50% as possible
            model.Add(home_games_count * 2 >= total_games_count - 1)  # Ensure aim_games is at least half floor
            model.Add(home_games_count * 2 <= total_games_count + 1) 

class TeamConflictConstraint(Constraint):
    ''' Ensure that when clubs specify two teams cannot play at the same time, they do not.'''

    def apply(self, model, X, data):
        conflicts = data['team_conflicts']
        locked_weeks = data.get('locked_weeks', set())
        timeslots = data['timeslots']
        games = data['games']
        
        for team_pairing in conflicts:
            team1 = team_pairing[0]
            team2 = team_pairing[1]
            
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
            
            # Apply constraint: at most one game involving conflicting teams per time slot
            for (week, day_slot), game_vars in time_slots_vars.items():
                model.Add(sum(game_vars) <= 1)

class MaxMaitlandHomeWeekends(Constraint):
    ''' Set maximum number of playable Maitland Weekends. '''
        
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']

        home_fields = {club.name: club.home_field for club in data['clubs']}
        weeks = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                        if t.field.location != 'Newcastle International Hockey Centre':
                            weeks[(t.week, t.field.location)].append(X[key])
                            
        indicator_dict = defaultdict(list)

        team_to_grade = {team.name: team.grade for team in data['teams']}
        grade_to_num_games = {grade.name: grade.num_games for grade in data['grades']}
        home_fields = {club.name: club.home_field for club in data['clubs']}

        # Note this is not the max at home field but the max games IN THE GRADE per home field
        max_games_at_home_field = {}

        for club_name, home_field in home_fields.items():
            team_names = get_teams_from_club(club_name, data['teams'])
            
            grade_games = {}
            for team_name in team_names:
                grade = team_to_grade[team_name]
                grade_name = grade
                grade_games[grade_name] = grade_to_num_games[grade_name]
            
            max_games = max(grade_games.values(), default=0)
            max_games_at_home_field[home_field] = max_games

        for week, game_vars in weeks.items():
            indicator_var = model.NewBoolVar(f"maitland_weekend_games_week{week}")
            indicator_dict[week[1]].append(indicator_var)
            model.AddMaxEquality(indicator_var, game_vars)  

        for location, indicator_var in indicator_dict.items():
            num_games = max_games_at_home_field[location]
            model.Add(sum(indicator_var) <= sum([num_games // 2, 1]))  

class EnsureBestTimeslotChoices(Constraint):
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

    WORST_TIME = '19:00'

    def apply(self, model, X, data):
        timeslots = data['timeslots']
        games = data['games']
        locked_weeks = data.get('locked_weeks', set())

        # Group vars by (week, day, location, field_name, day_slot)
        field_slot_vars = defaultdict(list)

        for t in timeslots:
            if not t.day:
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
            if week in locked_weeks:
                continue
            if len(vars_list) == 1:
                indicator = vars_list[0]
            else:
                indicator = model.NewBoolVar(f'fs_used_{week}_{field_name}_{day_slot}')
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
                pv = model.NewIntVar(0, 1, f'7pm_penalty_{key[6]}_{key[0]}_{key[1]}')
                model.Add(pv == var)
                data['penalties'][penalty_key]['penalties'].append(pv)

class ClubDayConstraint(Constraint):
    """ This consstraint deals with ensuring club days occur correctly. """
    
    def apply(self, model, X, data):

        club_days = data['club_days']
        teams = data['teams']
        clubs = data['clubs']
        locked_weeks = data.get('locked_weeks', set())

        allowed_keys = ['team1', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'round_no', 'field_name', 'field_location']
        for club_name in club_days:   
            if club_name.lower() not in [c.name.lower() for c in clubs]:
                raise ValueError(f'Invalid team name {club_name} in ClubDay Dictionary')   

            desired_date = club_days[club_name]
            closest_week = get_nearest_week_by_date(desired_date.strftime("%Y-%m-%d"), data['timeslots'])

            if closest_week in locked_weeks:
                print(f"Skipping club day constraint for {club_name} as it is in a locked week.")
                continue

            club = get_club_from_clubname(club_name, data['clubs'])
            club_teams = get_teams_from_club(club_name, teams)
            home_field = club.home_field
            
            # Locate all games for the club on the desired date
            club_games = [key for key in X if len(key) > 5 and key[allowed_keys.index('date')] == desired_date.date().strftime('%Y-%m-%d')  
                          and (key[allowed_keys.index('team1')] in club_teams 
                               or key[allowed_keys.index('team2')] in club_teams)]
            
            if not club_games:
                raise ValueError(f"No games found for club {club_name} on {desired_date.date()}")
            
            teams_by_grade = {}
            for team in club_teams:
                grade = team.rsplit(' ', 1)[1]
                teams_by_grade.setdefault(grade, []).append(team) 
                
           # Constraint: Every club team must play
            for team in club_teams:
                model.Add(sum(X[game_key] for game_key in club_games
                              if team in [game_key[allowed_keys.index('team1')], game_key[allowed_keys.index('team2')]]) >= 1) 

            # Constraint: Intra-club matchups for teams in the same grade
            for grade, teams_in_grade in teams_by_grade.items():
                if len(teams_in_grade) > 1:
                    # Create constraints to ensure intra-club matchups
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
                                            if ((game_key[allowed_keys.index('team1')], game_key[allowed_keys.index('team2')]) == pair or (game_key[allowed_keys.index('team1')], game_key[allowed_keys.index('team2')]) == pair2)])
                        
                    model.Add(sum(game_vars) >= no_potential_pairs)
                    
            # Constraint: Ensure that all games are played at the same field
            field_usage_vars = defaultdict(list)
            for game_key in club_games:
                field_name = game_key[allowed_keys.index('field_name')]
                field_usage_vars[field_name].append(X[game_key])
            
            # Create indicator if field is used
            field_indicator_vars = []
            for field_name, games in field_usage_vars.items():
                field_var = model.NewBoolVar(f'field_used_{club_name}_{field_name}')    
                model.AddMaxEquality(field_var, games)
                field_indicator_vars.append(field_var)

            model.Add(sum(field_indicator_vars) == 1)

            # Constraint: Ensure contiguity in game slots
            timeslot_groups = defaultdict(list)
            for game in club_games:
                timeslot_groups[game[allowed_keys.index('day_slot')]].append(X[game])

            slot_indicators = {}
            for day_slot, game_vars in timeslot_groups.items():
                if len(game_vars) > 1:
                    indicator = model.NewBoolVar(f'timeslot_indicator_{club_name}_{day_slot}')
                    slot_indicators[day_slot] = indicator
                    model.AddMaxEquality(indicator, game_vars)

            sorted_slots = sorted(slot_indicators.keys())
            for i in range(1, len(sorted_slots) - 1):
                prev_slot = sorted_slots[i - 1]
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]

                model.Add(slot_indicators[prev_slot] + slot_indicators[next_slot] <= 1).OnlyEnforceIf(slot_indicators[curr_slot].Not())    


class EqualMatchUpSpacingConstraint(Constraint):
    """
    Spread matchups evenly across rounds using pairwise forbidden gaps
    and sliding window density penalties.

    Uses zero nonlinear operations (no multiplication, division, max, or abs equality).
    Original algebraic version archived in constraints/archived_equalspacing_original.py.

    HARD constraint (pairwise forbidden gaps):
        For each matchup pair, for each pair of rounds (r1, r2) where
        0 < r2 - r1 < min_gap: the pair cannot play in both rounds.

    SOFT penalty (sliding window density):
        For each matchup pair, slide a window of size `space` across all
        rounds. In each window, penalize having more than 1 meeting.

    Parameters:
        - Ideal spacing = T - 2 (play all other opponents before rematch)
        - Floor = T // 2 + 1 (minimum gap can never go below this)
        - spacing_base_slack: configurable in season config (default 0)
        - --slack N: added on top of base_slack to further loosen
        - min_gap = max(T // 2 + 1, ideal - base_slack - config_slack)
    """

    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        R = data['num_rounds']['max']
        grades = {g.name: g.num_teams for g in data['grades']}

        config_slack = data.get('constraint_slack', {}).get('EqualMatchUpSpacingConstraint', 0)
        defaults = data.get('constraint_defaults', {})
        base_slack = defaults.get('spacing_base_slack', 0)

        weights = data.get('penalty_weights', {})
        PENALTY_WEIGHT = weights.get('EqualMatchUpSpacing', 5000)

        if 'penalties' not in data:
            data['penalties'] = {}
        if 'EqualMatchUpSpacing' not in data.get('penalties', {}):
            data['penalties']['EqualMatchUpSpacing'] = {'weight': PENALTY_WEIGHT, 'penalties': []}

        min_gap_per_grade = {}
        space_per_grade = {}
        for name, T in grades.items():
            ideal = T - 2
            floor = min(T // 2, T - 2)
            min_gap_per_grade[name] = max(floor, ideal - base_slack - config_slack)
            space_per_grade[name] = ideal

        # Gather game-vars by (t1, t2, grade, round_no)
        locked_weeks = data.get('locked_weeks', set())

        # HACK: When locked weeks are active, only apply to PHL and 2nd grade.
        # 3rd-6th grade matchups are forced and spacing constraints conflict
        # with forced game sum==1 constraints. Without locked weeks, apply to all.
        spacing_grades = {'PHL', '2nd'} if locked_weeks else set(grades.keys())
        meetings = defaultdict(lambda: defaultdict(list))
        for t in timeslots:
            if not t.day:
                continue
            if locked_weeks and t.week in locked_weeks:
                continue
            for (t1, t2, grade) in games:
                if grade not in spacing_grades:
                    continue
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    meetings[(t1, t2, grade)][t.round_no].append(X[key])

        for (t1, t2, grade), round_map in meetings.items():
            min_gap = min_gap_per_grade[grade]
            space = space_per_grade[grade]

            # Collect rounds that actually have variables for this pair
            active_rounds = sorted(r for r in round_map if round_map[r])

            # --- HARD constraint: pairwise forbidden gaps ---
            for i, r1 in enumerate(active_rounds):
                vars_r1 = round_map[r1]
                for r2 in active_rounds[i + 1:]:
                    gap = r2 - r1
                    if gap >= min_gap:
                        break
                    model.Add(sum(vars_r1) + sum(round_map[r2]) <= 1)

            # --- SOFT penalty: sliding window density ---
            if space >= R:
                continue

            for r_start in range(1, R - space + 2):
                r_end = r_start + space - 1
                window_vars = []
                for r in range(r_start, r_end + 1):
                    if r in round_map:
                        window_vars.extend(round_map[r])

                if len(window_vars) < 2:
                    continue

                pen = model.NewIntVar(
                    0, len(window_vars),
                    f"eqsp_wpen_{t1}_{t2}_{grade}_w{r_start}")
                model.Add(pen >= sum(window_vars) - 1)
                data['penalties']['EqualMatchUpSpacing']['penalties'].append(pen)


class ClubGradeAdjacencyConstraint(Constraint):
    """Ensure that a club's teams in adjacent grades do not play at the same time.
    
    Two constraints:
    1. Duplicate teams (HARD): Clubs with multiple teams in same grade can't play simultaneously
    2. Adjacent grades (SOFT): Penalizes games in adjacent grades at same timeslot
       - Allows overlaps but solver will minimize them
       - Helps families watch multiple grades for their club
    """

    def apply(self, model, X, data):
        games = data['games']          # list of (team1, team2, grade)
        timeslots = data['timeslots']  # list of Timeslot objects
        teams = data['teams']          # list of Team objects
        CLUBS = data['clubs']          # list of Club objects
        GRADES = data['grades']          # list of Grade objects
        locked_weeks = data.get('locked_weeks', set())  # Skip locked weeks (default: none)

        # Initialize penalty tracking for soft adjacent-grade constraint
        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubGradeAdjacencyConstraint'] = {'weight': weights.get('ClubGradeAdjacencyConstraint', 50000), 'penalties': []}

        # Define the grade‐ordering so we know which grades are "adjacent"
        grade_order = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
        # Precompute adjacent pairs: [("PHL","2nd"), ("2nd","3rd"), ...]
        adj_pairs = [
            (grade_order[i], grade_order[i+1])
            for i in range(len(grade_order)-1)
        ]

        # For quick lookup: which club each team belongs to
        def club_of(team_name):
            for t in teams:
                if t.name == team_name:
                    return t.club.name
            raise ValueError(f"Unknown team {team_name}")

        # Build a structure: club_slot_games[(club, slot_id, grade)] = [BoolVars...]
        # where slot_id is (week, day_slot)
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
                # Skip locked weeks - no point constraining already-fixed games
                if ts.week in locked_weeks:
                    continue
                # slot_id represents "same time" - same week and day_slot, regardless of field
                slot_id = (ts.week, ts.day_slot)
                key = (t1, t2, grade,
                       ts.day, ts.day_slot, ts.time,
                       ts.week, ts.date, ts.round_no,
                       ts.field.name, ts.field.location)
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

        # Constraint 1 (HARD): Duplicate teams in same grade can't play simultaneously
        for (club, slot_id, grade), vars_ in club_dup_games.items():
            if not vars_:
                continue
            model.Add(sum(vars_) <= 1)

        # Constraint 2 (SOFT): Adjacent grades - penalize overlaps but allow them
        # penalty = max(0, sum(g1) + sum(g2) - 1) for each slot
        adj_idx = 0
        for club in [c.name for c in CLUBS]:
            slot_ids = {
                slot_id
                for (c_name, slot_id, g) in club_slot_games
                if c_name == club
            }
            for slot_id in slot_ids:
                for g1, g2 in adj_pairs:
                    vars_g1 = club_slot_games.get((club, slot_id, g1), [])
                    vars_g2 = club_slot_games.get((club, slot_id, g2), [])
                    if not vars_g1 or not vars_g2:
                        continue
                    
                    # Count combined games in adjacent grades at this slot
                    max_possible = len(vars_g1) + len(vars_g2)
                    combined = model.NewIntVar(0, max_possible, f'adj_combined_{adj_idx}')
                    model.Add(combined == sum(vars_g1) + sum(vars_g2))
                    
                    # Soft penalty: penalize when combined > 1 (both grades playing same slot)
                    # penalty = max(0, combined - 1)
                    penalty = model.NewIntVar(0, max_possible, f'adj_penalty_{adj_idx}')
                    model.AddMaxEquality(penalty, [combined - 1, model.NewConstant(0)])
                    data['penalties']['ClubGradeAdjacencyConstraint']['penalties'].append(penalty)
                    adj_idx += 1


                
class ClubVsClubAlignment(Constraint):
    """ Ensures PHL and 2nd grade games for the same club pair coincide on Sundays,
    scheduled back-to-back on the same field.

    Hard constraints:
    - When grades coincide (same club-pair same Sunday round), must be on the same
      field AND in adjacent timeslots (back-to-back)
    - Minimum coincidences = num_games - slack

    Soft constraints (penalties):
    - Prefer maximum coincidences (penalize each miss below num_games)
    """
    def apply(self, model, X, data):
        # Initialize penalty tracking
        weights = data.get('penalty_weights', {})
        COINCIDE_PENALTY_WEIGHT = weights.get('ClubVsClubAlignment', 100000)

        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubVsClubAlignment'] = {'weight': COINCIDE_PENALTY_WEIGHT, 'penalties': []}

        # Get slack from config (--slack flag + base slack from CONSTRAINT_DEFAULTS)
        base_slack = data.get('constraint_defaults', {}).get('club_vs_club_alignment_base_slack', 0)
        config_slack = data.get('constraint_slack', {}).get('ClubVsClubAlignment', 0) + base_slack

        num_rounds = data['num_rounds']
        per_team_games = {grade.name: (num_rounds['max'] // (grade.num_teams - 1) )  if grade.num_teams % 2 == 0 else (num_rounds['max'] //grade.num_teams) for grade in data['grades']}

        ordered_games = dict(sorted(per_team_games.items(), key=lambda item: item[1]))
        locked_weeks = data.get('locked_weeks', set())

        # Track Sunday games with field and day_slot metadata
        # grade -> club_pair -> round_no -> list of (var, field_name, day_slot)
        sunday_games = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for t1, t2, grade in data['games']:
            if grade not in ['PHL', '2nd']:
                continue

            for t in data['timeslots']:
                if t.week in locked_weeks:
                    continue
                if t.day != 'Sunday':
                    continue
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    playing_clubs = tuple(sorted((get_club(t1, data['teams']), get_club(t2, data['teams']))))
                    sunday_games[grade][playing_clubs][t.round_no].append((X[key], t.field.name, t.day_slot))

        used_grades = []
        ini_num = 0
        btb_idx = 0
        for grade, num_games in ordered_games.items():
            if grade not in ['PHL', '2nd']:
                continue

            original_grade = sunday_games[grade]
            used_grades.append(grade)
            if num_games <= ini_num:
                continue
            ini_num = num_games

            for grade2 in sunday_games:
                if grade2 in used_grades:
                    continue
                club_dict = sunday_games[grade2]

                for clubs, rounds in original_grade.items():
                    if clubs not in club_dict:
                        continue

                    coincide_vars = []
                    for round_no, game_info_list in rounds.items():
                        if round_no not in club_dict[clubs]:
                            continue

                        other_game_info = club_dict[clubs][round_no]

                        # Create grade indicators from just the vars
                        vars1 = [gi[0] for gi in game_info_list]
                        vars2 = [gi[0] for gi in other_game_info]

                        game_indicator = model.NewBoolVar(f"game_played_{clubs}_{round_no}")
                        model.AddMaxEquality(game_indicator, vars1)

                        second_indicator = model.NewBoolVar(f"second_played_{clubs}_{round_no}")
                        model.AddMaxEquality(second_indicator, vars2)

                        coincide = model.NewBoolVar(f"coincide_{clubs}_{round_no}")
                        model.Add(coincide <= game_indicator)
                        model.Add(coincide <= second_indicator)
                        model.Add(coincide >= game_indicator + second_indicator - 1)

                        coincide_vars.append(coincide)

                        # HARD: back-to-back on the same field when coinciding
                        # Find all valid pairs: same field AND adjacent day_slot
                        btb_pairs = []
                        for var1, field1, slot1 in game_info_list:
                            for var2, field2, slot2 in other_game_info:
                                if field1 == field2 and abs(slot1 - slot2) == 1:
                                    pair_ind = model.NewBoolVar(
                                        f"btb_{clubs}_{round_no}_{field1}_{slot1}_{slot2}_{btb_idx}")
                                    model.Add(pair_ind <= var1)
                                    model.Add(pair_ind <= var2)
                                    model.Add(pair_ind >= var1 + var2 - 1)
                                    btb_pairs.append(pair_ind)
                                    btb_idx += 1

                        if btb_pairs:
                            # When coinciding, at least one back-to-back same-field pair
                            model.Add(sum(btb_pairs) >= 1).OnlyEnforceIf(coincide)
                        else:
                            # No valid back-to-back pairs exist — cannot coincide
                            model.Add(coincide == 0)

                    if coincide_vars:
                        # Calculate required minimum with slack
                        min_required = max(0, num_games - config_slack)

                        # HARD: At least min_required coincidences
                        model.Add(sum(coincide_vars) >= min_required)

                        # SOFT: Penalize each miss below num_games target
                        actual_coincidences = model.NewIntVar(0, len(coincide_vars), f"actual_coincide_{clubs}_{grade}_{grade2}")
                        model.Add(actual_coincidences == sum(coincide_vars))

                        coincide_deficit = model.NewIntVar(0, num_games, f"coincide_deficit_{clubs}_{grade}_{grade2}")
                        model.Add(coincide_deficit >= num_games - actual_coincidences)
                        data['penalties']['ClubVsClubAlignment']['penalties'].append(coincide_deficit)

# SOFT CONSTRAINTS

# Has hard element, forces no back to back Maitland home games
class MaitlandHomeGrouping(Constraint):
    '''Encourage all of Maitland's games to be at home or away as a group each week.'''

    def apply(self, model, X, data):


        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MaitlandHomeGrouping'] = {'weight': weights.get('MaitlandHomeGrouping', 1000000), 'penalties': []}

        maitland_games_per_week = {}
        maitland_home_games_per_week = {}
        locked_weeks = data.get('locked_weeks', set())

        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                if "Maitland" in t1 or "Maitland" in t2:
                    key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                    if key in X and t.day: # Ensure dummy timeslots not counted
                        week = t.week
                        if week not in maitland_games_per_week:
                            maitland_games_per_week[week] = []
                            maitland_home_games_per_week[week] = []

                        game_var = X[key] 
                        maitland_games_per_week[week].append(game_var)

                        if t.field.location == "Maitland Park":
                            maitland_home_games_per_week[week].append(game_var)

       
        maitland_home_games_per_week = dict(sorted(maitland_home_games_per_week.items()))
        maitland_games_per_week = dict(sorted(maitland_games_per_week.items()))

        home_week_indicators = []

        for week in maitland_games_per_week:
            if not maitland_games_per_week[week]:  
                continue
            if week in locked_weeks:
                continue
            max_games_per_week = len(data['timeslots'])

            home_games_var = model.NewIntVar(0, max_games_per_week, f'home_games_week_{week}')
            model.Add(home_games_var == sum(maitland_home_games_per_week[week]))

            away_games_var = model.NewIntVar(0, max_games_per_week, f'away_games_week_{week}')
            model.Add(away_games_var == sum(maitland_games_per_week[week]) - home_games_var)

            imbalance_penalty = model.NewIntVar(0, max_games_per_week, f'imbalance_penalty_week_{week}')
            model.AddMinEquality(imbalance_penalty, [home_games_var, away_games_var])

            data['penalties']['MaitlandHomeGrouping']['penalties'].append(imbalance_penalty)

            # Back to back no home enforcement
            week_indicator = model.NewBoolVar(f'week_indicator_{week}')
            if maitland_home_games_per_week[week]:
                model.AddMaxEquality(week_indicator, maitland_home_games_per_week[week])
            else:
                model.Add(week_indicator == 0)

            home_week_indicators.append(week_indicator)

        defaults = data.get('constraint_defaults', {})
        base_max = defaults.get('maitland_max_consecutive_home', 1)
        slack = data.get('constraint_slack', {}).get('MaitlandHomeGrouping', 0)
        max_consecutive = base_max + slack

        # Sliding window: in any window of (max_consecutive + 1) consecutive entries,
        # at most max_consecutive can be home weeks
        window_size = max_consecutive + 1
        for i in range(len(home_week_indicators) - window_size + 1):
            window = home_week_indicators[i:i + window_size]
            model.Add(sum(window) <= max_consecutive)

# Has Hard Element
class AwayAtMaitlandGrouping(Constraint):
    '''Encourage all of a club's games against Maitland to be played at Maitland when they are the away team.'''

    def apply(self, model, X, data):

        defaults = data.get('constraint_defaults', {})
        base_limit = defaults.get('away_maitland_max_clubs', 3)
        slack = data.get('constraint_slack', {}).get('AwayAtMaitlandGrouping', 0)
        HARD_LIMIT = base_limit + slack
        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['AwayAtMaitlandGrouping'] = {'weight': weights.get('AwayAtMaitlandGrouping', 100000), 'penalties': []}

        away_clubs_per_week = defaultdict(lambda: defaultdict(list)) 
        locked_weeks = data.get('locked_weeks', set())

        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                if "Maitland Park" in t.field.location:
                    key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)

                    if key in X and t.day: # Ensure dummy timeslots not counted 
                        away_club = get_club(t1, data['teams']) if "Maitland" in t2 else get_club(t2, data['teams'])  
                        away_clubs_per_week[t.week][away_club].append(X[key]) 

        for week, club_games in away_clubs_per_week.items():
            if week in locked_weeks:
                continue

            club_scheduled_vars = {} 

            for club, game_vars in club_games.items():
                club_scheduled_var = model.NewBoolVar(f'club_{club}_week_{week}_scheduled')
                model.AddMaxEquality(club_scheduled_var, game_vars)
                club_scheduled_vars[club] = club_scheduled_var

            # Note that the above dictionary tracks ONLY AWAY CLUBS 
            num_clubs_var = model.NewIntVar(0, len(club_scheduled_vars), f'num_away_clubs_week{week}')
            model.Add(num_clubs_var == sum(club_scheduled_vars.values())) 

            # Add hard limit of 3 away clubs
            model.Add(num_clubs_var <= HARD_LIMIT)   

            num_clubs_gt_1 = model.NewBoolVar(f'num_clubs_gt_1_week{week}')

            model.Add(num_clubs_var > 1).OnlyEnforceIf(num_clubs_gt_1)

            model.Add(num_clubs_var <= 1).OnlyEnforceIf(num_clubs_gt_1.Not())

            penalty_var = model.NewIntVar(0, len(club_scheduled_vars), f'penalty_week{week}_away_club_mismatch')
            
            model.Add(penalty_var == num_clubs_var - 1).OnlyEnforceIf(num_clubs_gt_1)
            model.Add(penalty_var == 0).OnlyEnforceIf(num_clubs_gt_1.Not())

            data['penalties']['AwayAtMaitlandGrouping']['penalties'].append(penalty_var)
 
class MaximiseClubsPerTimeslotBroadmeadow(Constraint):
    """Maximises the number of clubs per timeslot at Broadmeadow, ensuring diversity in clubs playing within the same timeslot.
    
    Hard element: minimum clubs per timeslot = floor(total_teams/2) - slack (floor at 0)
    Soft element: penalizes low diversity (prefer each game to be different clubs)
    """

    def apply(self, model, X, data):
        # Base hard limit offset = 0. With slack, we DECREASE the minimum by slack.
        # Since minimum = floor(games/2) + HARD_LIMIT, we use negative offset.
        # With --slack 1: HARD_LIMIT = -1, so minimum = floor(games/2) - 1
        slack = data.get('constraint_slack', {}).get('MaximiseClubsPerTimeslotBroadmeadow', 0)
        HARD_LIMIT = -slack  # Negative because we're reducing the minimum

        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MaximiseClubsPerTimeslotBroadmeadow'] = {'weight': weights.get('MaximiseClubsPerTimeslotBroadmeadow', 5000), 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())

        game_dict = defaultdict(lambda: defaultdict(list))  # { (week, day, timeslot): {club: [game_vars]} }

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                
                if key in X and t.field.location == 'Newcastle International Hockey Centre' and t.day in ['Saturday', 'Sunday']: 
                    club1 = get_club(t1, data['teams'])
                    club2 = get_club(t2, data['teams'])

                    game_dict[(t.week, t.day, t.day_slot)][club1].append(X[key])
                    game_dict[(t.week, t.day, t.day_slot)][club2].append(X[key]) # If same club then two equal keys in the one dict

        for (week, day, timeslot), club_games in game_dict.items():

            if week in locked_weeks:
                continue

            club_presence_vars = {}
            for club, game_vars in club_games.items():
                club_var = model.NewBoolVar(f'club_{club}_week{week}_day_slot{timeslot}')
                model.Add(sum(game_vars) >= 1).OnlyEnforceIf(club_var)  # Club present if they play
                model.Add(sum(game_vars) == 0).OnlyEnforceIf(club_var.Not())  # Otherwise, club absent
                club_presence_vars[club] = club_var

            total_teams_playing = model.NewIntVar(0, len([var for v in club_games.values() for var in v]), f'total_games_week{week}_day_slot{timeslot}')
            model.Add(total_teams_playing == sum([var for v in club_games.values() for var in v])) # Includes consideration for club v club

            hard_minimum_indicator = model.NewIntVar(0, len(club_presence_vars), f'hard_minimum_indicator_week{week}_day_slot{timeslot}')
            hard_min_start = model.NewIntVar(0, len(club_presence_vars), f'hard_min_start_week{week}_day_slot{timeslot}')
            model.AddDivisionEquality(hard_min_start, total_teams_playing, 2)
            # hard_minimum_indicator = floor(games/2) + HARD_LIMIT
            # With HARD_LIMIT = -slack, this reduces the minimum requirement
            # Use max(0, ...) to ensure we don't go negative
            raw_minimum = model.NewIntVar(-10, len(club_presence_vars), f'raw_minimum_week{week}_day_slot{timeslot}')
            model.Add(raw_minimum == hard_min_start + HARD_LIMIT)
            model.AddMaxEquality(hard_minimum_indicator, [raw_minimum, model.NewConstant(0)])  # Floor at 0

            # Compute number of clubs playing in the timeslot
            num_clubs_var = model.NewIntVar(0, len([var for v in club_games.values() for var in v]), f'num_clubs_week{week}_day_slot{timeslot}')
            model.Add(num_clubs_var == sum(club_presence_vars.values()))

            timeslot_used_indicator = model.NewBoolVar(f'timeslot_used_week{week}_day_slot{timeslot}')
            model.Add(total_teams_playing >= 1).OnlyEnforceIf(timeslot_used_indicator)  # Timeslot used if more than 1 game
            model.Add(total_teams_playing == 0).OnlyEnforceIf(timeslot_used_indicator.Not())  # Otherwise, timeslot not used

            # Define deviation penalty from ideal clubs
            penalty_var = model.NewIntVar(0, len([var for v in club_games.values() for var in v]), f'penalty_week{week}_day_slot{timeslot}')

            model.Add(penalty_var >=  total_teams_playing - num_clubs_var).OnlyEnforceIf(timeslot_used_indicator)
            model.Add(penalty_var == 0).OnlyEnforceIf(timeslot_used_indicator.Not())

            # Add Hard Minimum
            model.Add(num_clubs_var >= hard_minimum_indicator)

            # Store penalties for minimization
            data['penalties']['MaximiseClubsPerTimeslotBroadmeadow']['penalties'].append(penalty_var)

class MinimiseClubsOnAFieldBroadmeadow(Constraint):
    """ Minimises the number of clubs playing on a field on any particular day, this way clubs get continuity of games.
    
    Hard element: max clubs per field per day = 5 + slack
    Soft element: penalizes deviation from ideal of 2 clubs per field
    """

    def apply(self, model, X, data):
        defaults = data.get('constraint_defaults', {})
        base_limit = defaults.get('max_clubs_per_field', 5)
        slack = data.get('constraint_slack', {}).get('MinimiseClubsOnAFieldBroadmeadow', 0)
        HARD_LIMIT = base_limit + slack

        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MinimiseClubsOnAFieldBroadmeadow'] = {'weight': weights.get('MinimiseClubsOnAFieldBroadmeadow', 5000), 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']
        locked_weeks = data.get('locked_weeks', set())

        game_dict = defaultdict(lambda: defaultdict(list))  # { (week, day, timeslot): {club: [game_vars]} }

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                
                if key in X and t.field.location == 'Newcastle International Hockey Centre' and t.day in ['Saturday', 'Sunday']: # Ensure dummy timeslots not counted
                    club1 = get_club(t1, data['teams'])
                    club2 = get_club(t2, data['teams'])

                    game_dict[(t.week, t.date, t.field.name)][club1].append(X[key])
                    game_dict[(t.week, t.date, t.field.name)][club2].append(X[key])

        for (week, day, field_name), club_games in game_dict.items():
            if week in locked_weeks:
                continue

            club_presence_vars = {}
            for club, game_vars in club_games.items():
                club_var = model.NewBoolVar(f'club_{club}_week{week}_day{day}_field{field_name}')
                model.AddBoolOr([v for v in game_vars]).OnlyEnforceIf(club_var)
                model.AddBoolAnd([v.Not() for v in game_vars]).OnlyEnforceIf(club_var.Not())

                club_presence_vars[club] = club_var

            # Compute number of clubs playing in the field_name
            num_clubs_var = model.NewIntVar(0, len(games), f'num_clubs_week{week}_day{day}_field{field_name}')
            model.Add(num_clubs_var == sum(club_presence_vars.values()))

            # Add hard limit of 5
            model.Add(num_clubs_var <= HARD_LIMIT)

            # Define deviation penalty from ideal clubs
            penalty_var = model.NewIntVar(0, len(games), f'penalty_week{week}_day{day}_field{field_name}')
            model.AddAbsEquality(penalty_var, num_clubs_var - 2)


            # Store penalties for minimization
            data['penalties']['MinimiseClubsOnAFieldBroadmeadow']['penalties'].append(penalty_var)


def _normalize_preference_no_play(noplay: dict, teams: list, clubs: list) -> list:
    """
    Normalize PREFERENCE_NO_PLAY to a consistent format.
    
    Supports two input formats:
    
    2025 format (legacy):
        {'Maitland': [{'date': '2025-07-20', 'field_location': '...'}]}
        - Key = club name
        - Value = list of restriction dicts
    
    2026 format (structured):
        {'Crusaders_6th_Masters': {
            'club': 'Crusaders',
            'grade': '6th',           # Optional: single grade
            'grades': ['PHL', '2nd'], # Optional: multiple grades  
            'dates': [datetime(...)],
            'reason': '...',          # Optional: for documentation
        }}
        - Key = descriptive identifier
        - Value = structured dict with club, dates, optional grade filter
    
    Returns:
        List of (club_name, team_names, restriction_dicts) tuples
    """
    from datetime import datetime
    
    normalized = []
    club_names_lower = [c.name.lower() for c in clubs]
    
    for key, value in noplay.items():
        # Detect format: new format has 'club' key, old format is a list
        if isinstance(value, dict) and 'club' in value:
            # 2026 structured format
            club_name = value['club']
            if club_name.lower() not in club_names_lower:
                raise ValueError(f"Invalid club '{club_name}' in PREFERENCE_NO_PLAY entry '{key}'")
            
            # Get dates - can be 'dates' list or single 'date'
            dates = value.get('dates', [])
            if 'date' in value:
                dates = [value['date']]
            
            # Filter teams by grade if specified
            club_teams = get_teams_from_club(club_name, teams)
            if 'grade' in value:
                grade = value['grade']
                club_teams = [t for t in club_teams if grade.lower() in t.lower()]
            elif 'grades' in value:
                grades = [g.lower() for g in value['grades']]
                club_teams = [t for t in club_teams if any(g in t.lower() for g in grades)]
            
            # Convert dates to restriction dicts
            for date in dates:
                if isinstance(date, datetime):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                normalized.append((key, club_name, club_teams, {'date': date_str}))
                
        elif isinstance(value, list):
            # 2025 legacy format - key is the club name
            club_name = key
            if club_name.lower() not in club_names_lower:
                raise ValueError(f"Invalid club name '{club_name}' in PREFERENCE_NO_PLAY")
            
            club_teams = get_teams_from_club(club_name, teams)
            for restriction in value:
                normalized.append((key, club_name, club_teams, restriction))
        else:
            raise ValueError(f"Invalid PREFERENCE_NO_PLAY format for key '{key}': expected dict with 'club' key or list")
    
    return normalized


# Soft or User set constraints
class PreferredTimesConstraint(Constraint):
    """Ensure teams play at preferred times.
    
    Supports two PREFERENCE_NO_PLAY formats:
    - 2025 format: {'ClubName': [{'date': '...', ...}]}
    - 2026 format: {'EntryName': {'club': '...', 'dates': [...], 'grade': '...'}}
    """
    
    def apply(self, model, X, data):

        weights = data.get('penalty_weights', {})
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['PreferredTimesConstraint'] = {'weight': weights.get('PreferredTimesConstraint', 10000000), 'penalties': []}

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
        normalized = _normalize_preference_no_play(noplay, teams, clubs)
        
        # Enforce no-play times with penalties
        for entry_key, club_name, club_teams, constraint in normalized:
            if 'date' not in constraint:
                print(f"Warning: Skipping constraint '{entry_key}' - no date specified")
                continue
            
            if get_nearest_week_by_date(constraint['date'], data['timeslots']) in locked_weeks:
                print(f"Skipping noplay constraint for {entry_key} as it is in a locked week.")
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
                    penalty_var = model.NewIntVar(0, 1, f"penalty_{entry_key}_{i}")
                    model.Add(penalty_var == X[game_key])
                    data['penalties']['PreferredTimesConstraint']['penalties'].append(penalty_var)


class ClubGameSpread(Constraint):
    """Minimize gaps between a club's games on a given day, and limit double-ups.

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

    Config params (in constraint_defaults):
        club_game_spread_max_gap:     upper bound base (default 2)
        club_game_spread_max_overlap: lower bound base (default 0, no double-ups)

    Soft constraint:
        Penalize |gap| — ideal is 0 (all games in consecutive slots).
        Both spread (positive gap) and double-ups (negative gap) are penalized.

    Intra-club matches (e.g. Maitland A vs Maitland B) count as 1 game
    for that club — one timeslot is consumed for both teams.
    """

    def apply(self, model, X, data):
        teams = data['teams']
        locked_weeks = data.get('locked_weeks', set())
        defaults = data.get('constraint_defaults', {})
        max_gap_base = defaults.get('club_game_spread_max_gap', 2)
        config_slack = data.get('constraint_slack', {}).get('ClubGameSpread', 0)
        weights = data.get('penalty_weights', {})

        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubGameSpread'] = {
            'weight': weights.get('ClubGameSpread', 5000), 'penalties': []
        }

        max_overlap_base = defaults.get('club_game_spread_max_overlap', 0)

        hard_upper = max_gap_base + config_slack
        hard_lower = -(max_overlap_base + config_slack)

        # Build team-to-club mapping
        team_club = {t.name: t.club.name for t in teams}

        # Group X vars by (club, week, day, day_slot) -> list of vars
        club_week_day_slot_vars = defaultdict(list)

        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            # Only Broadmeadow — away venues have 1 field so gap/overlap is irrelevant
            if key[10] != 'Newcastle International Hockey Centre':
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

        # Regroup: (club, week, day) -> {day_slot: [vars]}
        club_week_day_groups = defaultdict(dict)
        for (club, week, day, day_slot), vars_list in club_week_day_slot_vars.items():
            club_week_day_groups[(club, week, day)][day_slot] = vars_list

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

                has_multiple = model.NewBoolVar(f'cgs_multi_{club}_w{week}_{day}')
                model.Add(num_games >= 2).OnlyEnforceIf(has_multiple)
                model.Add(num_games <= 1).OnlyEnforceIf(has_multiple.Not())

                # gap = 1 - num_games >= hard_lower => num_games <= 1 - hard_lower
                max_allowed = 1 - hard_lower
                model.Add(num_games <= max_allowed).OnlyEnforceIf(has_multiple)
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

            # min_active and max_active day_slots (exact via sentinel values)
            min_active = model.NewIntVar(min_slot, max_slot,
                                          f'cgs_min_{club}_w{week}_{day}')
            max_active = model.NewIntVar(min_slot, max_slot,
                                          f'cgs_max_{club}_w{week}_{day}')

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

            model.AddMinEquality(min_active, min_candidates)
            model.AddMaxEquality(max_active, max_candidates)

            # range_size = max_active - min_active + 1
            # Lower bound accommodates all-inactive case (sentinels: min=max_slot, max=min_slot)
            range_size = model.NewIntVar(min_slot - max_slot + 1, max_slot - min_slot + 1,
                                          f'cgs_range_{club}_w{week}_{day}')
            model.Add(range_size == max_active - min_active + 1)

            # gap = range_size - num_games (can be negative = double-ups)
            max_gap_possible = max_slot - min_slot
            min_gap_possible = min(1 - len(all_vars_for_day), min_slot - max_slot + 1)
            gap = model.NewIntVar(min_gap_possible, max_gap_possible,
                                   f'cgs_gap_{club}_w{week}_{day}')
            model.Add(gap == range_size - num_games)

            # Only enforce when club has >= 2 games
            has_multiple = model.NewBoolVar(f'cgs_multi_{club}_w{week}_{day}')
            model.Add(num_games >= 2).OnlyEnforceIf(has_multiple)
            model.Add(num_games <= 1).OnlyEnforceIf(has_multiple.Not())

            # HARD UPPER: gap <= hard_upper (limits spread)
            model.Add(gap <= hard_upper).OnlyEnforceIf(has_multiple)

            # HARD LOWER: gap >= hard_lower (limits double-ups)
            model.Add(gap >= hard_lower).OnlyEnforceIf(has_multiple)

            # SOFT: penalize |gap| — both spread and double-ups
            max_abs = max(max_gap_possible, -min_gap_possible)
            penalty = model.NewIntVar(0, max_abs,
                                       f'cgs_pen_{club}_w{week}_{day}')
            model.Add(penalty >= gap).OnlyEnforceIf(has_multiple)
            model.Add(penalty >= -gap).OnlyEnforceIf(has_multiple)
            model.Add(penalty == 0).OnlyEnforceIf(has_multiple.Not())
            data['penalties']['ClubGameSpread']['penalties'].append(penalty)

