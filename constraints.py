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
        current_week = data['current_week']
        weekly_games = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    weekly_games[(t.week, t1)].append(X[key])
                    weekly_games[(t.week, t2)].append(X[key])

        for (week, team), game_vars in weekly_games.items():
            if week <= current_week:
                continue
            model.Add(sum(game_vars) <= 1)

class NoDoubleBookingFieldsConstraint(Constraint):
    """Ensure no field is scheduled for more than one game per time slot."""
    
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        current_week = data['current_week']
        field_usage = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    field_usage[(t.day, t.day_slot, t.week, t.field.name)].append(X[key])

        for slot, game_vars in field_usage.items():
            if slot[2] <= current_week:
                continue
            model.Add(sum(game_vars) <= 1)

'''
class EnsureEqualGamesAndBalanceMatchUps(Constraint):
    """Ensure each team plays the most it can."""
    
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        num_rounds = data['num_rounds']
        num_dummy_timeslots = data['num_dummy_timeslots']

        team_games = defaultdict(lambda: defaultdict(list))
        grade_games = defaultdict(lambda: defaultdict(list))

        for (t1, t2, grade) in games:
            for t in timeslots:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X: # Do not filter dummys here
                    team_games[grade][t1].append(X[key])
                    team_games[grade][t2].append(X[key])
                    grade_games[grade][tuple(sorted((t1, t2)))].append(X[key])
            for i in range(num_dummy_timeslots):
                dummy_key = (t1, t2, grade,i)
                if dummy_key in X:
                    team_games[grade][t1].append(X[dummy_key])
                    team_games[grade][t2].append(X[dummy_key])
                    grade_games[grade][tuple(sorted((t1, t2)))].append(X[dummy_key])

        # Ensure that each team plays the correct number of games
        for grade, teams in team_games.items():
            num_teams = len([team for team in data['teams'] if team.grade == grade])
            total_games = (num_rounds[grade] // (num_teams - 1) ) * (num_teams - 1) if num_teams % 2 == 0 else (num_rounds[grade] //num_teams) * (num_teams - 1)

            print(f'Aim total games for grade {grade} is {total_games}.')

            for team, game_vars in teams.items():
                model.Add(sum(game_vars) == total_games)  

        # Ensure that each team plays each other team exactly the same amount of times
        for grade, teams in grade_games.items():
            num_teams = len([team for team in data['teams'] if team.grade == grade])
            per_team_games = (num_rounds[grade] // (num_teams - 1) )  if num_teams % 2 == 0 else (num_rounds[grade] //num_teams) 
            
            for team, game_vars in teams.items():
                model.Add(sum(game_vars) == per_team_games) 
'''

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
        current_week = data['current_week']

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

                        min_time = (base_time - timedelta(minutes=120)).time()
                        max_time = (base_time + timedelta(minutes=120)).time()

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

                        min_time = (base_time - timedelta(minutes=120)).time()
                        max_time = (base_time + timedelta(minutes=120)).time()

                        if isinstance(t.time, str):
                            t_time = datetime.strptime(t.time, '%H:%M').time()
                        else:
                            t_time = t.time  

                        if min_time <= t_time <= max_time and identifier[1] != t.field.location:
                            phl_games[(club, t2, t.week, t.day)][identifier].append(X[key])  
                        elif (t_time >= max_time or t_time <=min_time)  and identifier[1] == t.field.location:
                            phl_games[(club, t1, t.week, t.day)][identifier].append(X[key])

        for date, day_slots in phl_games.items():
            if date[2] <= current_week:
                continue
            for day_slot, game_vars in day_slots.items():
                    model.Add(sum(game_vars) <= 1)    
               
# Has Soft Element
class PHLAndSecondGradeTimes(Constraint):        
    ''' PHL games should not be played at the same time. 2nds games cannot be played at the same time as PHL games within the same club. Ensure max 3 Friday night games at Broadmeadow and enforce preferred dates. 
        So we do not need to enforce this for other locations as there is only one field there.
    '''      
    def apply(self, model, X, data):


        if 'penalties' not in data:
            data['penalties'] = {'phl_preferences': {'weight': 10000, 'penalties': []}}
        else:
            data['penalties']['phl_preferences'] = {'weight': 10000, 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']

        current_week = data['current_week']

        phl_preferences = data['phl_preferences']
        allowed_keys = {"preferred_dates"}
        invalid_keys = set(phl_preferences.keys()) - allowed_keys 

        if invalid_keys:
            raise ValueError(f"Invalid keys found: {invalid_keys}, currently do not support any keys other than {allowed_keys}")
   
        team_games = defaultdict(lambda: defaultdict(list))
        club_games = defaultdict(lambda: defaultdict(list))
        friday_games = defaultdict(list)
        preferred_dates = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)

                if grade in ['PHL', '2nd'] and key in X and t.day: # Ensure dummy timeslots not counted
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

                        # Max 3 Friday night games at Newcastle International Hockey Centre
                        if t.day == 'Friday' and t.field.location == 'Newcastle International Hockey Centre':
                            friday_games['Friday'].append(X[key])
                        
                        # Enforce preferred dates
                        if t.date in [d.date().strftime('%Y-%m-%d') for d in phl_preferences['preferred_dates']]:
                            preferred_dates[t.date].append(X[key])

                    else:
                        club = get_club(t1, data['teams'])
                        club_games[(t.week, t.day, t.field.location)][(t.day_slot, club, t1)].append(X[key])

                        club = get_club(t2, data['teams'])
                        club_games[(t.week, t.day, t.field.location)][(t.day_slot, club, t2)].append(X[key])

        for date, day_slots in team_games.items():
            if date[0] <= current_week: 
                continue

            for day_slot, game_vars in day_slots.items():
                if day_slot[1] == 'Newcastle International Hockey Centre': # Stop concurrent PHL games only at Broadmeadow
                    model.Add(sum(game_vars) <= 1)    
     
        for date, day_slots in club_games.items():
            if date[0] <= current_week: 
                continue

            for day_slot, game_vars in day_slots.items():
                if day_slot[2] == 'Newcastle International Hockey Centre': # Stop concurrent 2nd grade and PHL only at Broadmeadow 
                    model.Add(sum(game_vars) <= 1)

        for day, game_vars in friday_games.items(): # Ensure that only 3 Broadmedow games on a Friday are held
            model.Add(sum(game_vars) <= 3)

        for date, game_vars in preferred_dates.items():
            week_no = get_nearest_week_by_date(date, data['timeslots'])
            if week_no <= current_week:
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
        current_week = data['current_week']
        timeslots = data['timeslots']
        games = data['games']
        
        for team_pairing in conflicts:
            team1 = team_pairing[0]
            team2 = team_pairing[1]
            
            # Group by (week, day_slot) - same time regardless of field
            time_slots_vars = defaultdict(list)
            
            for t in timeslots:
                if t.week <= current_week:
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
    """ Ensures best choices of timeslots. """

    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        fields = data['fields']
        current_week = data['current_week']

        timeslots_weekly = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        games_per_location = defaultdict(lambda: defaultdict(list))

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    timeslots_weekly[(t.week, t.day)][t.field.location][t.day_slot].append(X[key])
                    games_per_location[(t.week, t.day)][t.field.location].append(X[key])
                    

        timeslots_indicators = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        timeslot_numbers = defaultdict(lambda: defaultdict(lambda: defaultdict()))

        for (week, day), locations in timeslots_weekly.items():

            if week <= current_week:
                continue
            
            for location, day_slots in locations.items():
                for day_slot, game_vars in day_slots.items():
                    if len(game_vars) > 1:
                        timeslot_indicator = model.NewBoolVar(f'timeslots_indicator_{week}_{location}')
                        timeslots_indicators[(week, day)][location][day_slot] = timeslot_indicator
                        model.AddMaxEquality(timeslot_indicator, game_vars)

                        timeslot_number = model.NewIntVar(0, len(day_slots), f'timeslot_number_{week}_{location}')
                        timeslot_numbers[(week, day)][location][day_slot] = timeslot_number
                        model.Add(timeslot_number == int(day_slot))

        # Stop unused timeslots between two used ones.
        for (week, day), locations in timeslots_indicators.items():
            for location, day_slots in locations.items():
                for i in range(2, len(day_slots) - 1):
                    prior_slot = day_slots[i - 1]
                    relevant_slot = day_slots[i]
                    following_slot = day_slots[i + 1]

                    model.Add(prior_slot + following_slot <= 1).OnlyEnforceIf(relevant_slot.Not())
                    
        for (week, day), locations in games_per_location.items():
            if week <= current_week:
                continue

            # no_weekly_games = model.NewIntVar(0, len(locations), f'no_weekly_games_{week}')
            # model.Add(no_weekly_games == sum([var for vars in locations.values() for var in vars]))

            for location in locations:
                fields_at_location = [field for field in fields if field.location == location]
                num_fields = len(fields_at_location)
                assert num_fields > 0, f"No fields found for location {location}"

                no_location_games = model.NewIntVar(0, len(games), f'no_location_games_{week}_{location}')
                model.Add(no_location_games == sum(locations[location]))

                quotient = model.NewIntVar(0, len(timeslots), f'quotient_{week}_{location}')
                model.AddDivisionEquality(quotient, no_location_games, num_fields)

                no_timeslots = model.NewIntVar(0, len(timeslots), f'no_timeslots_{week}_{location}')
                model.Add(no_timeslots == quotient + 1)

                number_vars = timeslot_numbers[(week, day)][location]
                for day_slot, number_var in number_vars.items():
                    indicator_var = timeslots_indicators[(week, day)][location][day_slot]

                    if location == 'Newcastle International Hockey Centre':
                        equivalence_indicator = model.NewIntVar(0, 200, f'equivalence_indicator_{week}_{location}_{day_slot}')

                        model.Add(equivalence_indicator >= 6)
                        model.Add(equivalence_indicator >= no_timeslots)

                        no_timeslots_indic  = model.NewBoolVar(f'no_timeslots_indicator_{week}_{location}_{day_slot}') 
                        model.Add(no_timeslots <= 6).OnlyEnforceIf(no_timeslots_indic)
                        model.Add(no_timeslots > 6).OnlyEnforceIf(no_timeslots_indic.Not())

                        model.Add(equivalence_indicator <= 6).OnlyEnforceIf(no_timeslots_indic)
                        model.Add(equivalence_indicator <= no_timeslots).OnlyEnforceIf(no_timeslots_indic.Not())

                        model.Add(number_var <= equivalence_indicator).OnlyEnforceIf(indicator_var)

                    else:
                        model.Add(number_var <= no_timeslots).OnlyEnforceIf(indicator_var)

class ClubDayConstraint(Constraint):
    """ This consstraint deals with ensuring club days occur correctly. """
    
    def apply(self, model, X, data):

        club_days = data['club_days']
        teams = data['teams']
        clubs = data['clubs']
        current_week = data['current_week']

        allowed_keys = ['team1', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        for club_name in club_days:   
            if club_name.lower() not in [c.name.lower() for c in clubs]:
                raise ValueError(f'Invalid team name {club_name} in ClubDay Dictionary')   

            desired_date = club_days[club_name]
            closest_week = get_nearest_week_by_date(desired_date.strftime("%Y-%m-%d"), data['timeslots'])

            if closest_week <= current_week:
                print(f"Skipping club day constraint for {club_name} as it is in the past.")
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

            timeslot_indicators = defaultdict(list)
            for day_slot, game_vars in timeslot_groups.items():
                if len(game_vars) > 1:
                    timeslot_indicator = model.NewBoolVar(f'timeslot_indicator_{club_name}_{day_slot}')
                    timeslot_indicators[day_slot] = timeslot_indicator
                    model.AddMaxEquality(timeslot_indicator, game_vars)

            for i in range(2, len(timeslot_indicators) - 1):
                prior_slot = timeslot_indicators[i - 1]
                relevant_slot = timeslot_indicators[i]
                following_slot = timeslot_indicators[i + 1]

                model.Add(prior_slot + following_slot <= 1).OnlyEnforceIf(relevant_slot.Not())    


class EqualMatchUpSpacingConstraint(Constraint):
    ''' Spread out matchups evenly across rounds. '''
        
    def apply(self, model, X, data):

        SLACK = 1

        games = data['games']
        timeslots = data['timeslots']

        home_games = defaultdict(list)
        away_games = defaultdict(list)

        max_rounds = data['num_rounds']['max']

        grade_spacing_vars = defaultdict(lambda: defaultdict(int))

        for grade in data['grades']:
            space =  max_rounds // grade.num_teams

            min_grade_spacing_var = model.NewIntVar(0, space + SLACK, f'grade_spacing_{grade.name}')
            model.Add(min_grade_spacing_var == space - SLACK)

            max_grade_spacing_var = model.NewIntVar(0, space + SLACK, f'max_grade_spacing_{grade.name}')
            model.Add(max_grade_spacing_var == space + SLACK)

            grade_spacing_vars[grade.name]['min'] = min_grade_spacing_var
            grade_spacing_vars[grade.name]['max'] = max_grade_spacing_var

        meetings = defaultdict(lambda: defaultdict(list))  # (team1, team2) -> list of BoolVars
        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)    
                if key in X and t.day:  # Ensure dummy timeslots not counted
                    meetings[(t1, t2, grade)][t.round_no].append(X[key])

        for (team_pair, rounds) in meetings.items():
            
            indicator_list = []
            week_no_list = []

            sorted_rounds = dict(sorted(rounds.items(), key=lambda x: x[0]))  # Sort rounds by round number
            for round_no, game_vars in sorted_rounds.items():
                if len(game_vars) > 1:
                    indicator_var = model.NewBoolVar(f'meeting_indicator_{team_pair[0]}_{team_pair[1]}_round{round_no}')
                    model.AddMaxEquality(indicator_var, game_vars)
                    indicator_list.append(indicator_var)

                    week_var = model.NewIntVar(0, max_rounds, f'week_var_{team_pair[0]}_{team_pair[1]}_round{round_no}')
                    model.Add(week_var == round_no).OnlyEnforceIf(indicator_var)
                    model.Add(week_var == 0).OnlyEnforceIf(indicator_var.Not())  # If not played, week is 0
                    week_no_list.append(week_var)

            grade = team_pair[2]

            no_meetings = model.NewIntVar(0, len(indicator_list), f'no_meetings_{team_pair[0]}_{team_pair[1]}')
            model.Add(no_meetings == sum(indicator_list))

            meets_twice = model.NewBoolVar(f"meets2_{t1}_{t2}")
            model.Add(no_meetings >= 2).OnlyEnforceIf(meets_twice)
            model.Add(no_meetings <  2).OnlyEnforceIf(meets_twice.Not())

            round_sum = model.NewIntVar(0, max_rounds * len(indicator_list), f'round_sum_{team_pair[0]}_{team_pair[1]}')
            model.Add(round_sum == sum(week_no_list))

            max_round_meet = model.NewIntVar(0, max_rounds, f'max_round_meet_{team_pair[0]}_{team_pair[1]}')
            model.AddMaxEquality(max_round_meet, week_no_list)

            multi_max = model.NewIntVar(0, max_rounds * len(indicator_list), f'max_week_sum_{team_pair[0]}_{team_pair[1]}')
            model.AddMultiplicationEquality(multi_max, [max_round_meet, no_meetings])

            upper_bound_space_coef = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_multiplication_{team_pair[0]}_{team_pair[1]}')
            model.AddMultiplicationEquality(upper_bound_space_coef, [no_meetings - 1, no_meetings])

            upper_bound_space_coef_2 = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_multiplication_2_{team_pair[0]}_{team_pair[1]}')
            model.AddDivisionEquality(upper_bound_space_coef_2, upper_bound_space_coef, 2)

            upper_bound_subtraction = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_{team_pair[0]}_{team_pair[1]}')    
            model.AddMultiplicationEquality(upper_bound_subtraction, [grade_spacing_vars[grade]['min'], upper_bound_space_coef_2])

            lower_bound_space_coef = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_multiplication_{team_pair[0]}_{team_pair[1]}')
            model.AddMultiplicationEquality(lower_bound_space_coef, [no_meetings, no_meetings - 1])

            lower_bound_coef_2 = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_multiplication_2_{team_pair[0]}_{team_pair[1]}')
            model.AddDivisionEquality(lower_bound_coef_2, lower_bound_space_coef, 2)

            lower_bound_subtraction = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_{team_pair[0]}_{team_pair[1]}')
            model.AddMultiplicationEquality(lower_bound_subtraction, [grade_spacing_vars[grade]['max'], lower_bound_coef_2])

            upper_bound = model.NewIntVar(0, max_rounds * len(indicator_list), f'upper_bound_{team_pair[0]}_{team_pair[1]}')
            model.Add(upper_bound == multi_max - upper_bound_subtraction)

            lower_bound = model.NewIntVar(0, max_rounds * len(indicator_list), f'lower_bound_{team_pair[0]}_{team_pair[1]}')
            model.Add(lower_bound == multi_max - lower_bound_subtraction)

            model.Add(round_sum <= upper_bound).OnlyEnforceIf(meets_twice)
            model.Add(round_sum >= lower_bound).OnlyEnforceIf(meets_twice)

'''
class EqualMatchUpSpacingConstraint(Constraint):
    """Spread out matchups evenly across rounds."""

    def apply(self, model, X, data):
        SLACK      = 1
        games      = data['games']
        timeslots  = data['timeslots']
        R          = data['num_rounds']['max']
        grades     = {g.name: g.num_teams for g in data['grades']}

        # 1) Prepare per‐grade “space = floor(R / T)”
        space_per_grade = {
            name: R // T
            for name, T in grades.items()
        }

        # 2) Gather game-vars by (t1,t2,grade,round_no)
        meetings = defaultdict(lambda: defaultdict(list))
        for t in timeslots:
            if not t.day: continue
            for (t1, t2, grade) in games:
                key = (t1, t2, grade,
                       t.day, t.day_slot, t.time,
                       t.week, t.date, t.round_no,
                       t.field.name, t.field.location)
                if key in X:
                    meetings[(t1,t2,grade)][t.round_no].append(X[key])

        # 3) For each pair+grade, build spacing
        for (t1, t2, grade), round_map in meetings.items():
            # Build exactly R indicators & week_vars
            indic  = []
            weekvs = []
            for r in range(1, R+1):
                vars_at_r = round_map.get(r, [])
                # indicator: do they meet in r?
                b = model.NewBoolVar(f"ind_{t1}_{t2}_{grade}_r{r}")
                if vars_at_r:
                    model.AddMaxEquality(b, vars_at_r)
                else:
                    model.Add(b == 0)
                indic.append(b)

                # week var: =r if b else 0
                wv = model.NewIntVar(0, R, f"wv_{t1}_{t2}_{grade}_r{r}")
                model.Add(wv == r).OnlyEnforceIf(b)
                model.Add(wv == 0).OnlyEnforceIf(b.Not())
                weekvs.append(wv)

            # K = total meetings
            K = model.NewIntVar(0, R, f"K_{t1}_{t2}_{grade}")
            model.Add(K == sum(indic))

            # Only enforce when K>=2
            meets2 = model.NewBoolVar(f"m2_{t1}_{t2}_{grade}")
            model.Add(K >= 2).OnlyEnforceIf(meets2)
            model.Add(K <  2).OnlyEnforceIf(meets2.Not())

            # round_sum = sum of weeks
            round_sum = model.NewIntVar(0, R*R, f"rs_{t1}_{t2}_{grade}")
            model.Add(round_sum == sum(weekvs))

            # max_r = max week
            max_r = model.NewIntVar(0, R, f"mr_{t1}_{t2}_{grade}")
            model.AddMaxEquality(max_r, weekvs)

            # fK = K*(K-1)/2
            nm1  = model.NewIntVar(-R, R, f"nm1_{t1}_{t2}_{grade}")
            prod = model.NewIntVar(-R*R, R*R, f"prd_{t1}_{t2}_{grade}")
            half = model.NewIntVar(0, R* (R-1)//2, f"hf_{t1}_{t2}_{grade}")
            model.Add(nm1 == K - 1)
            model.AddMultiplicationEquality(prod, [K, nm1]).OnlyEnforceIf(meets2)
            model.AddDivisionEquality(half, prod, model.NewConstant(2)).OnlyEnforceIf(meets2)

            # ideal = K*max_r - space*half
            space = space_per_grade[grade]
            t1v   = model.NewIntVar(0, R*R, f"t1_{t1}_{t2}_{grade}")
            t2v   = model.NewIntVar(-R*R, R*R, f"t2_{t1}_{t2}_{grade}")
            ideal = model.NewIntVar(-R*R, R*R, f"idl_{t1}_{t2}_{grade}")
            model.AddMultiplicationEquality(t1v, [K, max_r]).OnlyEnforceIf(meets2)
            model.AddMultiplicationEquality(t2v, [model.NewConstant(space), half]).OnlyEnforceIf(meets2)
            model.Add(ideal == t1v - t2v).OnlyEnforceIf(meets2)

            # slack = SLACK * half
            slack = model.NewIntVar(0, R*(R-1)//2 * SLACK, f"slk_{t1}_{t2}_{grade}")
            model.Add(slack == half * SLACK).OnlyEnforceIf(meets2)

            # |round_sum - ideal| <= slack
            diff = model.NewIntVar(-R*R, R*R, f"diff_{t1}_{t2}_{grade}")
            model.Add(diff == round_sum - ideal).OnlyEnforceIf(meets2)
            model.AddAbsEquality(slack, diff).OnlyEnforceIf(meets2)
'''
            
class ClubGradeAdjacencyConstraint(Constraint):
    """Ensure that a club's teams in adjacent grades do not play at the same time."""

    def apply(self, model, X, data):
        games = data['games']          # list of (team1, team2, grade)
        timeslots = data['timeslots']  # list of Timeslot objects
        teams = data['teams']          # list of Team objects
        CLUBS = data['clubs']          # list of Club objects
        GRADES = data['grades']          # list of Grade objects


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
        # where slot_id is (week, day_slot, field_name)
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
            # we assume t1 and t2 are from same club here if we care—
            # but we only index by the club of t1 (and t2) when they are same
            for ts in timeslots:
                # slot_id represents "same time" - same week and day_slot, regardless of field
                slot_id = (ts.week, ts.day_slot)
                key = (t1, t2, grade,
                       ts.day, ts.day_slot, ts.time,
                       ts.week, ts.date, ts.round_no,
                       ts.field.name, ts.field.location)
                if key in X:
                    # if either t1 or t2 plays at this slot, record it
                    var = X[key]

                    if t1_club != t2_club:
                        club_slot_games[(t1_club, slot_id, grade)].append(var)
                        club_slot_games[(t2_club, slot_id, grade)].append(var)
                        if t1 in club_dup_grades[t1_club][grade]: # If there are multiple teams in the same club in the same grade, add to this list only if they aren't playing each other.
                            club_dup_games[(t1_club, slot_id, grade)].append(var)
                        elif t2 in club_dup_grades[t2_club][grade]:
                            club_dup_games[(t2_club, slot_id, grade)].append(var)
                    else:
                        club_slot_games[(t1_club, slot_id, grade)].append(var)

        for (club, slot_id, grade), vars_ in club_dup_games.items():
            if not vars_:
                continue
            # If there are multiple teams in the same club in the same grade, ensure they do not play at the same time
            model.Add(sum(vars_) <= 1)

        # Now, for each club, each timeslot, and each adjacent‐grade pair,
        # forbid both grades having a game in the same slot:
        for club in [c.name for c in CLUBS]:
            # collect all slot_ids for which this club has any game
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
                    # sum of any game in g1 plus any in g2 ≤ 1
                    model.Add(sum(vars_g1) + sum(vars_g2) <= 1)


                
class ClubVsClubAlignment(Constraint):
    """ This is designed to ensure that each team in a club should play only one club on a weekend. So if 2nd grade plays Tigers twice, and minimally so does every other grade, there will be two weekends where that club only plays Tigers"""
    def apply(self, model, X, data):
        # Get relevant clubs
        num_rounds = data['num_rounds']
        per_team_games = {grade.name: (num_rounds['max'] // (grade.num_teams - 1) )  if grade.num_teams % 2 == 0 else (num_rounds['max'] //grade.num_teams) for grade in data['grades']}


        ordered_games = dict(sorted(per_team_games.items(), key=lambda item: item[1]))
        # print(f"Reverse order games: {ordered_games}")
        grades_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        fields_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for t1, t2, grade in data['games']:
            if grade in ['PHL', '2nd']:
                continue

            for t in data['timeslots']:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
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
                                game_indicator = model.NewBoolVar(f"game_played_{clubs}_{round_no}")
                                model.AddMaxEquality(game_indicator, game_vars)
                                
                                second_game_vars = club_dict[clubs][round_no]
                                second_indicator = model.NewBoolVar(f"second_played_{clubs}_{round_no}")
                                model.AddMaxEquality(second_indicator, second_game_vars)
                                
                                coincide = model.NewBoolVar(f"coincide_{clubs}_{round_no}")
                                model.Add(coincide <= game_indicator)
                                model.Add(coincide <= second_indicator)
                                model.Add(coincide >= game_indicator + second_indicator - 1)
                                
                                coincide_vars.append(coincide)

                                field_usage_vars = defaultdict(list)

                                for field, game_vars in fields_dict[clubs][round_no].items():
                                    field_indicator = model.NewBoolVar(f"field_games_{clubs}_{round_no}_{field}")
                                    model.AddMaxEquality(field_indicator, game_vars)
                                    
                                    field_usage_vars[field].append(field_indicator)
                                    
                                # Put them on the same field
                                model.Add(sum([v for var in field_usage_vars.values() for v in var]) == 1).OnlyEnforceIf(coincide)           


                        if coincide_vars:
                            model.Add(sum(coincide_vars) == num_games)

# SOFT CONSTRAINTS

# Has hard element, forces no back to back Maitland home games
class MaitlandHomeGrouping(Constraint):
    '''Encourage all of Maitland's games to be at home or away as a group each week.'''

    def apply(self, model, X, data):


        if 'penalties' not in data:
            data['penalties'] = {'MaitlandHomeGrouping': {'weight': 1000000, 'penalties': []}}
        else:
            data['penalties']['MaitlandHomeGrouping'] = {'weight': 1000000, 'penalties': []}

        maitland_games_per_week = {}
        maitland_home_games_per_week = {}
        current_week = data['current_week']

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
            if week <= current_week:
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

        for i in range(1, len(home_week_indicators)):
            prior_week = home_week_indicators[i - 1]
            current_week = home_week_indicators[i]

            # If the current week has a home game, then the previous week must not have a home game
            model.Add(prior_week + current_week <= 1)

# Has Hard Element
class AwayAtMaitlandGrouping(Constraint):
    '''Encourage all of a club's games against Maitland to be played at Maitland when they are the away team.'''

    def apply(self, model, X, data):

        HARD_LIMIT = 3
        if 'penalties' not in data:
            data['penalties'] = {'AwayAtMaitlandGrouping': {'weight': 100000, 'penalties': []}}
        else:
            data['penalties']['AwayAtMaitlandGrouping'] = {'weight': 100000, 'penalties': []}

        away_clubs_per_week = defaultdict(lambda: defaultdict(list)) 
        current_week = data['current_week']

        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                if "Maitland Park" in t.field.location:
                    key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)

                    if key in X and t.day: # Ensure dummy timeslots not counted 
                        away_club = get_club(t1, data['teams']) if "Maitland" in t2 else get_club(t2, data['teams'])  
                        away_clubs_per_week[t.week][away_club].append(X[key]) 

        for week, club_games in away_clubs_per_week.items():
            if week <= current_week:
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
    """Maximises the number of clubs per timeslot at Broadmeadow, ensuring diversity in clubs playing within the same timeslot."""

    def apply(self, model, X, data):
        HARD_LIMIT = 0

        if 'penalties' not in data:
            data['penalties'] = {'MaximiseClubsPerTimeslotBroadmeadow': {'weight': 5000, 'penalties': []}}
        else:
            data['penalties']['MaximiseClubsPerTimeslotBroadmeadow'] = {'weight': 5000, 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']
        current_week = data['current_week']

        game_dict = defaultdict(lambda: defaultdict(list))  # { (week, day, timeslot): {club: [game_vars]} }

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                
                if key in X and t.field.location == 'Newcastle International Hockey Centre' and t.day in ['Satuday', 'Sunday']: 
                    club1 = get_club(t1, data['teams'])
                    club2 = get_club(t2, data['teams'])

                    game_dict[(t.week, t.day, t.day_slot)][club1].append(X[key])
                    game_dict[(t.week, t.day, t.day_slot)][club2].append(X[key]) # If same club then two equal keys in the one dict

        for (week, day, timeslot), club_games in game_dict.items():

            if week <= current_week:
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
            model.Add(hard_minimum_indicator == hard_min_start + HARD_LIMIT)  # Ensure at least 3 clubs in the timeslot

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
    """ Minimises the number of clubs playing on a field on any particular day, this way clubs get continuity of games. """

    def apply(self, model, X, data):

        HARD_LIMIT = 5

        if 'penalties' not in data:
            data['penalties'] = {'MinimiseClubsOnAFieldBroadmeadow': {'weight': 5000, 'penalties': []}}
        else:
            data['penalties']['MinimiseClubsOnAFieldBroadmeadow'] = {'weight': 5000, 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']
        current_week = data['current_week']

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
            if week <= current_week:
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


# Soft or User set constraints
class PreferredTimesConstraint(Constraint):
    """Ensure teams play at preferred times."""
    
    def apply(self, model, X, data):

        if 'penalties' not in data:
            data['penalties'] = {'PreferredTimesConstraint': {'weight': 10000000, 'penalties': []}}
        else:
            data['penalties']['PreferredTimesConstraint'] = {'weight': 10000000, 'penalties': []}

        teams = data['teams']
        clubs = data['clubs']
        noplay = data['preference_no_play']
        current_week = data['current_week']

        allowed_keys = ['team_name', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        
        allowed_keys2 = ['team1', 'team_name', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']

        # Enforce no-play times with penalties
        for club_name, restrictions in noplay.items():
            club_teams = get_teams_from_club(club_name, teams)
            if club_name.lower() not in [c.name.lower() for c in clubs]:
                raise ValueError(f'Invalid team name {club_name} in PreferredTimeConstraint')

            
            for index, constraint in enumerate(restrictions):
                if not all(key in allowed_keys for key in constraint.keys()):
                    raise ValueError(f"Invalid key in noplay constraint for {club_name}: {constraint.keys()}")
                if 'date' not in constraint:
                    raise ValueError(f"Missing date in noplay constraint for {club_name}: {constraint}")
                
                if get_nearest_week_by_date(constraint['date'], data['timeslots']) <= current_week:
                    print(f"Skipping noplay constraint for {club_name} as it is in the past.")
                    continue

                for i, game_key in enumerate(X):
                    game_dict = dict(zip(allowed_keys, game_key))
                    if all(game_dict.get(k) == v for k, v in constraint.items()) and (game_key[0] in club_teams or game_key[1] in club_teams):
                        penalty_var = model.NewIntVar(0, 1, f"penalty_{club_name}_{index}")
                        model.Add(penalty_var == X[game_key])
                        data['penalties']['PreferredTimesConstraint']['penalties'].append(penalty_var)

                    game_dict = dict(zip(allowed_keys2, game_key))
                    if all(game_dict.get(k) == v for k, v in constraint.items()) and (game_key[0] in club_teams or game_key[1] in club_teams):
                        penalty_var = model.NewIntVar(0, 1, f"penalty_{club_name}_{index}_{i}")
                        model.Add(penalty_var == X[game_key])
                        data['penalties']['PreferredTimesConstraint']['penalties'].append(penalty_var)

