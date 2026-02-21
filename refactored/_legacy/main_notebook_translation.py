#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pydantic import BaseModel, Field,  field_validator
from typing import List, Dict, Tuple
from datetime import time as tm
from datetime import timedelta, datetime
from ortools.sat.python import cp_model
import pandas as pd
import os
from collections import defaultdict, Counter
import re
import json
from abc import ABC, abstractmethod
import math
import numpy as np
from itertools import chain
import csv
from itertools import combinations
from ortools.sat.python import cp_model
import pickle
import os


class PlayingField(BaseModel):
    name: str = Field(..., description="Name of the field")
    location: str = Field(..., description="Location of the field")

    def __getattr__(self, attr):
        if attr == "field_location":
            return self.location
        elif attr == "field_name":
            return self.name
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

class Grade(BaseModel):
    name: str = Field(..., description="Grade name")
    teams: List[str] = Field(..., description="List of team names in this grade")
    num_teams: int = Field(0, description="Number of teams in this grade")
    num_games: int = Field(0, description="Number of games in this grade")


    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "num_teams", len(self.teams))

    def __lt__(self, other):
        expected_order = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]  

        if not isinstance(other, Grade):
            return NotImplemented
        try:
            self_index = expected_order.index(self.name)
            other_index = expected_order.index(other.name)
        except ValueError:
            raise ValueError(f"Unknown grade in comparison: {self.name} or {other.name}")

        return self_index > other_index    

    def set_games(self, num_rounds):
        self.num_games = (num_rounds // (self.num_teams - 1) ) * (self.num_teams - 1) if self.num_teams % 2 == 0 else (num_rounds //self.num_teams) * (self.num_teams - 1)

class Timeslot(BaseModel):
    date: str = Field(..., description="Date of the game (e.g., '2025-03-04')")
    day: str = Field(..., description="Day of the game (e.g., 'Saturday', 'Sunday')")
    time: str = Field(..., description="Time of the game (e.g., 14:00 for 2 PM)")
    week: int = Field(..., description="The week number for the season")
    day_slot: int = Field(..., description="The game slot for the day (e.g., 1 for first game of the day)")
    field: PlayingField = Field(..., description="Field where the game is played")
    round_no: int = Field(..., description="Round number for the season")

class Club(BaseModel):
    name: str = Field(..., description="Club name")
    home_field: str = Field(..., description="Home field")
    preferred_times: List[Timeslot] = Field(default=[], description="Preferred play times for the club")
    num_teams: int = Field(0, description="Number of teams in this club")

class Team(BaseModel):
    name: str = Field(..., description="Name of the team")
    club: Club = Field(..., description="Club the team belongs to")
    grade: str = Field(..., description="Grade the team belongs to")
    preferred_times: List[Timeslot] = Field(default=[], description="Times the team prefers to play")
    unavailable_times: List[Timeslot] = Field(default=[], description="Times the team cannot play")
    constraints: List[str] = Field(default=[], description="Special scheduling constraints for the team")

class ClubDay(BaseModel):
    date: str = Field(..., description="Date of the game (e.g., '2025-03-04')")
    day: str = Field(..., description="Day of the game (e.g., 'Saturday', 'Sunday')")
    week: int = Field(..., description="The week number for the season")
    field: PlayingField = Field(..., description="Field where the game is played")


class Game(BaseModel):
    team1: str = Field(..., description="First team playing")
    team2: str = Field(..., description="Second team playing")
    timeslot: Timeslot = Field(..., description="Scheduled time for the game")
    field: PlayingField = Field(..., description="Field where the game is played")
    grade: Grade = Field(..., description="Grade the game belongs to")

class WeeklyDraw(BaseModel):
    week: int = Field(..., description="Week number in the season")
    round_no: int = Field(..., description="Round number for the season")
    games: List[Game] = Field(..., description="Games scheduled for this week")
    bye_teams: List[str] = Field(default=[], description="Teams with a bye this week")

class Roster(BaseModel):
    weeks: List[WeeklyDraw] = Field(..., description="Complete schedule for the season")

    def save(self, path: str) -> None:
        """Save the roster to a JSON file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.dict(), f, indent=4)

FIELDS = [PlayingField(location='Newcastle International Hockey Centre', name='SF'),
          PlayingField(location='Newcastle International Hockey Centre', name='EF'),
          PlayingField(location='Newcastle International Hockey Centre', name='WF'),
          PlayingField(location='Maitland Park', name='Maitland Main Field',),
          PlayingField(location='Central Coast Hockey Park', name='Wyong Main Field'),]

TEAMS_DATA = r'data\2025\teams'
CLUBS = []
TEAMS = []

for file in os.listdir(TEAMS_DATA):
    df = pd.read_csv(os.path.join(TEAMS_DATA, file))
    club = Club(name=df['Club'].iloc[0].strip(), home_field="Maitland Park" if df['Club'].iloc[0].strip() == "Maitland" else 
                      'Central Coast Hockey Park' if df['Club'].iloc[0].strip() == 'Gosford' else "Newcastle International Hockey Centre")
    CLUBS.append(club)
    teams = [Team(name=f"{row['Team Name'].strip()} {row['Grade'].strip()}", club=club, grade=row['Grade'].strip(), home_field=club.home_field) for index, row in df.iterrows()]
    TEAMS.extend(teams)

teams_by_grade = defaultdict(list)
for team in TEAMS:
    teams_by_grade[team.grade].append(team.name)
GRADES = [Grade(name=f'{grade}', teams=teams) for grade, teams in sorted(teams_by_grade.items())]

teams_by_club = defaultdict(list)
for team in TEAMS:
    teams_by_club[team.club.name].append(team.name)

for club, teams in teams_by_club.items():
    club_obj = next((c for c in CLUBS if c.name == club), None)
    club_obj.num_teams = len(teams)

for grade in GRADES:
    print(grade.name, grade.teams, grade.num_teams)

for club in CLUBS:  
    print(club.name, club.num_teams)


# In[3]:




# Aim total games for grade PHL is 20.
# Aim total games for grade 4th is 16.
# Aim total games for grade 5th is 18.
# Aim total games for grade 6th is 14.
# Aim total games for grade 3rd is 16.
# Aim total games for grade 2nd is 16.

# In[2]:


# import csv
# from itertools import combinations

# # Find 6th grade teams
# six_grade_teams = next((grade.teams for grade in GRADES if grade.name == "6th"), None)

# # Generate all possible matchups (pairs)
# team_pairs = set(combinations(six_grade_teams, 2))

# # Export to CSV
# csv_filename = "sixth_grade_matchups.csv"
# with open(csv_filename, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Team 1", "Team 2"])  # CSV Header
#     writer.writerows(team_pairs)

# print(f"CSV file '{csv_filename}' created successfully!")


# In[3]:


def get_club(team_name, teams):
    for team in teams:
        if team_name == team.name:
            return team.club.name
    raise ValueError(f"Team {team_name} not found in teams when calling get_club") 

def get_club_object(team_name, teams):
    for team in teams:
        if team_name == team.name:
            return team.club
    raise ValueError(f"Team {team_name} not found in teams when calling get_club_object") 

def get_teams_from_club(club_name, teams):
    return [team.name for team in teams if team.club.name == club_name]

def get_club_from_clubname(club, CLUBS):
    for c in CLUBS:
        if c.name == club:
            return c
    raise ValueError(f"Club {club} not found in CLUBS when calling get_club_from_clubname")

def get_duplicated_graded_teams(club, grade, teams):
    dup_teams = []
    for team in teams:
        if team.club.name == club and team.grade == grade:
            dup_teams.append(team.name)
    return dup_teams

def split_number_suffix(text):
    match = re.match(r"(\d+)([a-zA-Z]+)", text)  
    if match:
        return match.group(1), match.group(2)  

    return text, ""  

def add_ordinal_suffix(number):
    if 11 <= number % 100 <= 13: 
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")

    return f"{number}{suffix}"

def get_round_number_for_week(week, timeslots):
    for t in timeslots:
        if t.week == week:
            return t.round_no
    raise ValueError(f"Week {week} not found in timeslots when calling get_round_number_for_week")

def get_nearest_week_by_date(target_date_str, timeslots, date_format="%Y-%m-%d"):
    target_date = datetime.strptime(target_date_str, date_format).date()

    def parse_date(t):
        return datetime.strptime(t.date, date_format).date()

    closest = min(timeslots, key=lambda t: abs((parse_date(t) - target_date).days))
    return closest.week

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

        # bucket real + dummy slots exactly as before
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
        fields = data['fields']
        current_week = data['current_week']
        for team_pairing in conflicts:
            team1 = team_pairing[0]
            team2 = team_pairing[1]
            for t in data['timeslots']:
                if t.week <= current_week:
                    continue
                game_vars = []

                for (t1, t2, grade) in data['games']:
                    if t1 in [team1, team2] or t2 in [team1, team2]:
                        for field in fields:
                            key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, field.name, field.location)
                            if key in X and t.day: # Ensure dummy timeslots not counted
                                game_vars.append(X[key])
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
                slot_id = (ts.week, ts.day_slot, ts.field.name)
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
            model.Add(num_clubs_var <= 3)   

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
            model.Add(hard_minimum_indicator == hard_min_start)

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
            model.Add(num_clubs_var <= 5)

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


####### OLD CONSTRAINTS ########
## Hard
"""
# class NoBackToBackByes(Constraint):
#     ''' Ensure there are no back to back byes'''

#     def apply(self, model, X, data):
#         games = data['games']
#         timeslots = data['timeslots']
#         num_rounds = data['num_rounds']

#         team_games = defaultdict(lambda: defaultdict(list))
#         print(f'NoBackTOBackByes skipped for 6th grade')
#         for t in timeslots:
#             for (t1, t2, grade) in games:

#                 if grade == '6th':
#                     continue

#                 key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
#                 if key in X and t.day: # Ensure dummy timeslots not counted
#                     if t.round_no + 1 <= num_rounds['max']:
#                         team_games[t1][(t.round_no, t.round_no + 1)].append(X[key])
#                         team_games[t2][(t.round_no, t.round_no + 1)].append(X[key])

#                     if t.round_no > 1:
#                         team_games[t1][(t.round_no - 1, t.round_no)].append(X[key])
#                         team_games[t2][(t.round_no - 1, t.round_no)].append(X[key])

#         for team, time_slot in team_games.items():
#             for slot, game_vars in time_slot.items():
#                 model.Add(sum(game_vars) >= 1)
"""

## Soft
class NoBackToBackByes(Constraint):
    '''Ensure there are no back-to-back byes (soft constraint).'''

    def apply(self, model, X, data):

        if 'penalties' not in data:
            data['penalties'] = {'NoBackToBackByes': {'weight': 100000, 'penalties': []}}
        else:
            data['penalties']['NoBackToBackByes'] = {'weight': 100000, 'penalties': []}

        games = data['games']
        timeslots = data['timeslots']
        num_rounds = data['num_rounds']

        team_games = defaultdict(lambda: defaultdict(list))

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    if t.week + 1 <= num_rounds['max']:
                        team_games[t1][(t.week, t.week + 1)].append(X[key])
                        team_games[t2][(t.week, t.week + 1)].append(X[key])

                    if t.week > 1:
                        team_games[t1][(t.week - 1, t.week)].append(X[key])
                        team_games[t2][(t.week - 1, t.week)].append(X[key])

        # Apply soft penalty for back-to-back byes
        for team, time_slot in team_games.items():
            for slot, game_vars in time_slot.items():
                penalty_var = model.NewIntVar(0, 1, f"bye_penalty_{team}_{slot}")
                model.Add(sum(game_vars) >= 1 - penalty_var)
                data['penalties']['NoBackToBackByes']['penalties'].append(penalty_var)

class PHL2ndAlignment(Constraint):

    def apply(self, model, X, data):
        # Get relevant clubs
        num_rounds = data['num_rounds']
        per_team_games = {grade.name: (num_rounds[grade.name] // (grade.num_teams - 1) )  if grade.num_teams % 2 == 0 else (num_rounds[grade.name] //grade.num_teams) for grade in data['grades']}

        aim_games = min(per_team_games['PHL'], per_team_games['2nd']) 

        current_week = data['current_week']
        current_round = get_round_number_for_week(current_week, data['timeslots'])

        seconds_dict = defaultdict(lambda: defaultdict(list))
        phl_dict = defaultdict(lambda: defaultdict(list))
        fields_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for t1, t2, grade in data['games']:
            if grade not in ['PHL', '2nd']:
                continue

            for t in data['timeslots']:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    if grade == 'PHL':
                        playing_clubs = tuple(sorted((get_club(t1, data['teams']), get_club(t2, data['teams']))))
                        phl_dict[playing_clubs][t.round_no].append(X[key])  
                        fields_dict[playing_clubs][t.round_no][t.field.name].append(X[key])
                    else:
                        playing_clubs = tuple(sorted((get_club(t1, data['teams']), get_club(t2, data['teams']))))
                        seconds_dict[playing_clubs][t.round_no].append(X[key])               
                        fields_dict[playing_clubs][t.round_no][t.field.name].append(X[key])

        for clubs, phl_rounds in phl_dict.items():
            if clubs in seconds_dict:
                coincide_vars = []  
                for round_no, phl_game_vars in phl_rounds.items():
                    if round_no in seconds_dict[clubs]:
                        phl_indicator = model.NewBoolVar(f"phl_played_{clubs}_{round_no}")
                        model.AddMaxEquality(phl_indicator, phl_game_vars)

                        second_game_vars = seconds_dict[clubs][round_no]
                        second_indicator = model.NewBoolVar(f"second_played_{clubs}_{round_no}")
                        model.AddMaxEquality(second_indicator, second_game_vars)

                        coincide = model.NewBoolVar(f"coincide_{clubs}_{round_no}")
                        model.Add(coincide <= phl_indicator)
                        model.Add(coincide <= second_indicator)
                        model.Add(coincide >= phl_indicator + second_indicator - 1)

                        coincide_vars.append(coincide)

                        if round_no <= current_round:
                            continue

                        field_usage_vars = defaultdict()

                        for field, game_vars in fields_dict[clubs][round_no].items():
                            field_indicator = model.NewBoolVar(f"field_games_{clubs}_{round_no}_{field}")
                            model.AddMaxEquality(field_indicator, game_vars)
                            field_usage_vars[field] = field_indicator

                        model.Add(sum(field_usage_vars.values()) == 1).OnlyEnforceIf(coincide)              

                if coincide_vars:
                    model.Add(sum(coincide_vars) >= aim_games)

## Soft
"""
# class FiftyFiftyHomeandAway(Constraint):
#     ''' Push toward 50% home and away games for Maitland'''

    # def apply(self, model, X, data):
    #     if 'penalties' not in data:
    #         data['penalties'] = {'FiftyFiftyHomeandAway': {'weight': 10000000, 'penalties': []}}
    #     else:
    #         data['penalties']['FiftyFiftyHomeandAway'] = {'weight': 10000000, 'penalties': []}

    #     games = data['games']
    #     timeslots = data['timeslots']

    #     home_games = defaultdict(list)
    #     away_games = defaultdict(list)

    #     # Step 1: Collect home and away games for each team
    #     for t in timeslots:
    #         for (t1, t2, grade) in games:
    #             key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
    #             if ('Maitland' in t1 or 'Maitland' in t2) and key in X and t.day: # Ensure dummy timeslots not counted
    #                 relevant_team = t1 if 'Maitland' in t1 else t2
    #                 other_team = t2 if relevant_team == t1 else t1
    #                 if 'Maitland' in other_team:
    #                     continue
    #                 elif t.field.location == 'Maitland Park':
    #                     home_games[(relevant_team, other_team)].append(X[key])
    #                 else:
    #                     away_games[(relevant_team, other_team)].append(X[key])

    #             if ('Gosford' in t1 or 'Gosford' in t2) and key in X:
    #                 relevant_team = t1 if 'Gosford' in t1 else t2
    #                 other_team = t2 if relevant_team == t1 else t1
    #                 if 'Gosford' in other_team:
    #                     continue
    #                 elif t.field.location == 'Central Coast Hockey Park':
    #                     home_games[(relevant_team, other_team)].append(X[key])
    #                 else:
    #                     away_games[(relevant_team, other_team)].append(X[key])

#         # Step 2: Apply 50/50 balancing constraint
#         for team, home_vars in home_games.items():
#             away_vars = away_games[team]

#             if not home_vars or not away_vars:  # Skip teams with no games
#                 continue  

#             home_games_count = sum(home_vars)
#             total_games_count = home_games_count + sum(away_vars)

#             # Ensure aim_games is as close to 50% as possible
#             aim_games = model.NewIntVar(0, len(home_vars) + len(away_vars), f'aim_games_{team}')
#             model.Add(aim_games * 2 >= total_games_count)  # Ensure aim_games is at least half
#             model.Add(aim_games * 2 <= total_games_count + 1)  # Allow rounding up

#             # Define penalty variable as an absolute deviation
#             penalty_var = model.NewIntVar(0, len(home_vars) + len(away_vars), f'deviation_{team}')
#             model.AddAbsEquality(penalty_var, home_games_count - aim_games)

#             # Store penalty correctly
#             data['penalties']['FiftyFiftyHomeandAway']['penalties'].append(penalty_var)
"""

class EnsureUniqueTeamsEvery3Weeks(Constraint):
    ''' Push toward 50% of games vs each team played at home and away for each away clubs team. '''

    def apply(self, model, X, data):

        games = data['games']
        timeslots = data['timeslots']
        max_rounds = data['num_rounds']['max']
        game_dict = defaultdict(lambda: defaultdict(list))


        # Step 1: Collect home and away games for each team
        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X and t.day: # Ensure dummy timeslots not counted
                    # if t.round_no > 1 and t.round_no <= max_rounds - 1:
                    #     game_dict[(t1, t2)][t.round_no - 1, t.round_no, t.round_no +1].append(X[key])
                    # if t.round_no > 2:
                    #     game_dict[(t1, t2)][t.round_no - 2, t.round_no -1 , t.round_no].append(X[key])
                    # if t.round_no <= max_rounds - 2:
                    #     game_dict[(t1, t2)][t.round_no, t.round_no + 1, t.round_no + 2].append(X[key])
                    if t.round_no > 1 and t.round_no <= max_rounds - 1:
                        game_dict[(t1, t2)][ t.round_no, t.round_no +1].append(X[key])
                    if t.round_no > 1:
                        game_dict[(t1, t2)][ t.round_no -1 , t.round_no].append(X[key])


        for team, round_group in game_dict.items():
            for round_group, game_vars in round_group.items():
                model.Add(sum(game_vars) <= 1) # For any particular pairing of teams, for any particular 3 week group, have them play once.

class MaitlandGradeOrder(Constraint):
    '''Encourage games in Maitland to be scheduled in grade order (7th to PHL).'''

    def apply(self, model, X, data):
        if 'penalties' not in data:
            data['penalties'] = {'MaitlandGradeOrder': {'weight': 10000, 'penalties': []}}
        else:
            data['penalties']['MaitlandGradeOrder'] = {'weight': 10000, 'penalties': []}

        expected_order = ["7th", "6th", "5th", "4th", "3rd", "2nd", "PHL"]
        grade_to_rank = {grade: rank for rank, grade in enumerate(expected_order)}
        current_week = data['current_week']

        maitland_games = defaultdict(lambda: defaultdict(list))  # {week: {day: [(day_slot, grade, X[key])]}}

        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                if t.field.location == "Maitland Park":
                    key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                    if key in X and t.day: # Ensure dummy timeslots not counted
                        maitland_games[t.week][t.day].append((t.day_slot, grade, X[key]))

        for week, days in maitland_games.items():
            if week <= current_week:
                continue
            for day, games in days.items():
                games.sort(key=lambda game: game[0])  # Sort by day_slot

                grade_ranks = [grade_to_rank[grade] for _, grade, _ in games if grade in grade_to_rank]

                penalty_var = model.NewIntVar(0, sum(range(len(grade_ranks))), f'grade_order_penalty_week{week}_day{day}')
                model.Add(penalty_var == sum(grade_ranks[i] * games[i][2] for i in range(len(grade_ranks))))

                data['penalties']['MaitlandGradeOrder']['penalties'].append(penalty_var)


"""

# Soft Club Day only
class ClubDayConstraint(Constraint):
    ''' This consstraint deals with ensuring club days occur correctly. '''

    def apply(self, model, X, data):

        if 'penalties' not in data:
            data['penalties'] = {'ClubDayConstraint': {'weight': 1000000000, 'penalties': []}}
        else:
            data['penalties']['ClubDayConstraint'] = {'weight': 1000000000, 'penalties': []}

        club_days = data['club_days']
        teams = data['teams']
        clubs = data['clubs']

        allowed_keys = ['team1', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']

        for club_name in club_days:   
            if club_name.lower() not in [c.name.lower() for c in clubs]:
                raise ValueError(f'Invalid team name {club_name} in ClubDay Dictionary')   

            desired_date = club_days[club_name]
            club = get_club_from_clubname(club_name, data['clubs'])
            club_teams = get_teams_from_club(club_name, teams)
            home_field = club.home_field

            # Locate all games for the club on the desired date
            club_games = [key for key in X if key[allowed_keys.index('date')] == desired_date.date().strftime('%Y-%m-%d')  
                          and (key[allowed_keys.index('team1')] in club_teams 
                               or key[allowed_keys.index('team2')] in club_teams)]

            if not club_games:
                raise ValueError(f"No games found for club {club_name} on {desired_date.date()}")

            # Track which fields are used 
            field_usage_vars = {}
            for game_key in club_games:
                field_name = game_key[allowed_keys.index('field_name')]
                if field_name not in field_usage_vars:
                    field_usage_vars[field_name] = model.NewBoolVar(f'field_used_{club_name}_{field_name}')

                model.Add(X[game_key] <= field_usage_vars[field_name])

            # Count the number of distinct fields used 
            num_fields_used = model.NewIntVar(0, len(field_usage_vars), f'num_fields_used_{club_name}')
            model.Add(num_fields_used == sum(field_usage_vars.values()))

            penalty_var1 = model.NewIntVar(0, len(field_usage_vars), f'club_field_penalty_{club_name}')
            model.Add(penalty_var1 == num_fields_used - 1)
            data['penalties']['ClubDayConstraint']['penalties'].append(penalty_var1)

            # Track which teams are scheduled 
            team_scheduled_vars = {}
            for team in club_teams:
                team_scheduled_vars[team] = model.NewBoolVar(f'team_scheduled_{club_name}_{team}')

            for game_key in club_games:
                team1, team2 = game_key[allowed_keys.index('team1')], game_key[allowed_keys.index('team2')]
                if team1 in team_scheduled_vars:
                    model.Add(X[game_key] <= team_scheduled_vars[team1])
                if team2 in team_scheduled_vars:
                    model.Add(X[game_key] <= team_scheduled_vars[team2])

            # Count the number of scheduled club teams 
            num_scheduled_teams = model.NewIntVar(0, len(club_teams), f'num_scheduled_teams_{club_name}')
            model.Add(num_scheduled_teams == sum(team_scheduled_vars.values()))

            penalty_var2 = model.NewIntVar(0, len(club_teams), f'club_teams_penalty_{club_name}')
            model.Add(penalty_var2 == len(club_teams) - num_scheduled_teams)
            data['penalties']['ClubDayConstraint']['penalties'].append(penalty_var2)

# class PHL2ndByeAlignment(Constraint):
#     '''
    For each week, if Gosford is scheduled to play a team from a given club in a PHL game,
    then force that club's 2nd-grade team(s) to get a bye.
    If the club has only one 2nd-grade team, it should not play at all.
    If the club has multiple 2nd-grade teams, at most one may be scheduled.
    This only applies when the overall number of 2nd-grade teams is odd.
    '''

    def apply(self, model, X, data):
        # Check if 2nd grade has an odd number of teams 
        grade_2nd = next((g for g in data['grades'] if g.name == "2nd"), None)
        if not grade_2nd:
            raise ValueError("2nd grade missing from data['grades']")
        if len(grade_2nd.teams) % 2 == 0:
            print("Even number of 2nd grade teams; no forced bye alignment needed.")
            return

        current_week = data['current_week']

        weeks = {t.week for t in data["timeslots"]}

        for week in weeks:

            if week <= current_week:
                continue

            for club in data["clubs"]:
                indicator = model.NewBoolVar(f"is_gosford_vs_{club.name}_week{week}")

                relevant_games = []
                for t in data["timeslots"]:
                    if t.week != week:
                        continue
                    for (t1, t2, grade) in data["games"]:
                        key = (
                            t1, t2, grade,
                            t.day, t.day_slot, t.time, t.week,
                            t.date, t.round_no, t.field.name, t.field.location
                        )
                        if grade != "PHL" or key not in X or not t.day: # Skip dummy variables
                            continue

                        if "Gosford" in t1 or "Gosford" in t2:
                            opponent = t2 if "Gosford" in t1 else t1
                            opp_club = get_club_object(opponent, data["teams"])
                            if opp_club.name == club.name:
                                relevant_games.append(X[key])

                M = len(relevant_games) if relevant_games else 1
                if relevant_games:
                    model.Add(sum(relevant_games) <= M * indicator)
                    model.Add(sum(relevant_games) >= indicator)
                else:
                    model.Add(indicator == 0)

                club_2nd_teams = get_duplicated_graded_teams(club.name, "2nd", data["teams"])
                if not club_2nd_teams:
                    continue 


                scheduled_vars = []
                for t in data["timeslots"]:
                    if t.week != week:
                        continue
                    for (t1, t2, grade) in data["games"]:
                        key = (
                            t1, t2, grade,
                            t.day, t.day_slot, t.time, t.week,
                            t.date, t.round_no, t.field.name, t.field.location
                        )
                        if grade != "2nd" or key not in X or not t.day: # Skip dummy variables
                            continue
                        if any(team in {t1, t2} for team in club_2nd_teams):
                            scheduled_vars.append(X[key])

                if club_2nd_teams and scheduled_vars:
                    if len(club_2nd_teams) == 1:
                        model.Add(sum(scheduled_vars) <= (1 - indicator) * M)
                    else:
                        model.Add(sum(scheduled_vars) <= 1 + (1 - indicator) * M)

class PHL2ndByeAlignment(Constraint):
    '''
    For each week, if Gosford is scheduled to play a team from a given club in a PHL game,
    then force that club's 2nd-grade team(s) to get a bye.
    If the club has only one 2nd-grade team, it should not play at all.
    If the club has multiple 2nd-grade teams, at most one may be scheduled.
    This only applies when the overall number of 2nd-grade teams is odd.
    '''

    def apply(self, model, X, data):
        # Check if 2nd grade has an odd number of teams 
        grade_2nd = next((g for g in data['grades'] if g.name == "2nd"), None)
        if not grade_2nd:
            raise ValueError("2nd grade missing from data['grades']")
        if len(grade_2nd.teams) % 2 == 0:
            print("Even number of 2nd grade teams; no forced bye alignment needed.")
            return

        current_week = data['current_week']
        weeks = {t.week for t in data["timeslots"]}
        club_indicators = {}

        for week in weeks:

            if week <= current_week:
                continue

            for club in data["clubs"]:

                relevant_games_phl = defaultdict(list)
                relevant_games_seconds = defaultdict(list)
                dup_seconds = get_duplicated_graded_teams(club.name, "2nd", data["teams"])

                for t in data["timeslots"]:
                    if t.week != week:
                        continue
                    for (t1, t2, grade) in data["games"]:
                        key = (
                            t1, t2, grade,
                            t.day, t.day_slot, t.time, t.week,
                            t.date, t.round_no, t.field.name, t.field.location
                        )
                        if grade == "PHL" and key in X and  t.day: # Skip dummy variables
                            if "Gosford" in t1 or "Gosford" in t2:
                                opponent = t2 if "Gosford" in t1 else t1
                                opp_club = get_club_object(opponent, data["teams"])
                                if opp_club.name == club.name:
                                    relevant_games_phl[t.week].append(X[key])

                        elif grade == "2nd" and key in X and t.day:
                            if any(team in {t1, t2} for team in dup_seconds):
                                relevant_games_seconds[t.week].append(X[key])

                indicators_seconds = defaultdict(list)
                indicators_phl = defaultdict(list)

                for week, game_vars in relevant_games_phl.items():
                    phl_indicator = model.NewBoolVar(f"is_gosford_vs_{club.name}_week{week}")
                    seconds_indicator = model.NewBoolVar(f"2nd_grade_played_{club.name}_week{week}")

                    seconds_vars = relevant_games_seconds[week]

                    if game_vars:
                        model.AddMaxEquality(phl_indicator, game_vars)
                        model.AddMaxEquality(seconds_indicator, seconds_vars)

                        model.Add(seconds_indicator <= len(dup_seconds) - 1).OnlyEnforceIf(phl_indicator)

"""

class MinimizeLateGames(Constraint):
    '''Encourage games at all fields to be scheduled back-to-back and earlier in the day.'''

    def apply(self, model, X, data):
        if 'penalties' not in data:
            data['penalties'] = {'MinimizeLateGames': {'weight': 10000, 'penalties': []}}
        else:
            data['penalties']['MinimizeLateGames'] = {'weight': 10000, 'penalties': []}

        games_by_field = defaultdict(lambda: defaultdict(list))  # {week: {field: [(day_slot, game_var)]}}

        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    games_by_field[(t.week, t.day)][t.field.name].append((t.day_slot, X[key]))

        for week, fields in games_by_field.items():
            for field, games in fields.items():
                # Sort games by `day_slot`
                games.sort(key=lambda g: g[0])  
                game_vars = [var for _, var in games]
                day_slots = [slot for slot, _ in games]

                num_games_var = model.NewIntVar(0, len(game_vars), f'num_games_week{week}_field{field}')
                model.Add(num_games_var == sum(game_vars))  # Count number of scheduled games

                max_games = len(game_vars)  # Upper bound

                actual_sum_var = model.NewIntVar(0, max_games, f'actual_sum_week{week}_field{field}')
                model.Add(actual_sum_var == sum(slot * var for slot, var in games))

                # Define expected sum based on the formula S(n) = n(n+1)/2
                temp_var = model.NewIntVar(0, max_games * (max_games + 1), f'temp_mult_week{week}_field{field}')
                model.AddMultiplicationEquality(temp_var, num_games_var, num_games_var + 1)

                expected_sum_var = model.NewIntVar(0, (max_games * (max_games + 1)) // 2, f'expected_sum_week{week}_field{field}')
                model.AddDivisionEquality(expected_sum_var, temp_var, 2)  # expected_sum = (n(n+1)) / 2

                penalty_var = model.NewIntVar(0, sum(day_slots), f'late_game_penalty_week{week}_field{field}')
                model.Add(penalty_var == actual_sum_var - expected_sum_var) 

                data['penalties']['MinimizeLateGames']['penalties'].append(penalty_var)

class MaximiseSundayGames(Constraint):
    '''Encourage games to be scheduled on Sundays by applying a negative penalty.'''

    def apply(self, model, X, data):
        if 'penalties' not in data:
            data['penalties'] = {'RewardSundayGames': {'weight': -8000, 'penalties': []}}
        else:
            data['penalties']['RewardSundayGames'] = {'weight': -8000, 'penalties': []}

        sunday_games_by_field = defaultdict(lambda: defaultdict(list))  # {week: {field: [game_var]}}

        for t in data['timeslots']:
            if t.day == "Sunday":  # Only consider Sunday games
                for (t1, t2, grade) in data['games']:
                    key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                    if key in X:
                        sunday_games_by_field[(t.week, t.day)][t.field.name].append(X[key])

        for week, fields in sunday_games_by_field.items():
            for field, game_vars in fields.items():
                num_sunday_games_var = model.NewIntVar(0, len(game_vars), f'sunday_games_week{week}_field{field}')
                model.Add(num_sunday_games_var == sum(game_vars))  # Count number of Sunday games

                reward_var = model.NewIntVar(0, len(game_vars), f'sunday_reward_week{week}_field{field}')
                model.Add(reward_var == num_sunday_games_var)  # More Sunday games, higher reward

                data['penalties']['RewardSundayGames']['penalties'].append(reward_var)



# In[4]:


def generate_timeslots(start_date, end_date, day_time_map, fields, field_unavailabilities):
    """Generate weekly timeslots between two dates, considering field unavailability."""
    timeslots = []
    current_date = start_date
    week_number = 1
    round_no = 0
    day_slot = 1
    c_day = None
    round_indic = True
    draw_start = False

    # Check if all fields in field_unavailabilities exist in fields
    field_name_check = list(field_unavailabilities.keys())
    known_fields = [field.location for field in fields]
    for field_name in field_name_check:
        if field_name not in known_fields:
            raise ValueError(f"Field {field_name} in field_unavailabilities does not exist in fields!") 
    # Check that fields are correct in day time map
    for field_name in day_time_map.keys():
        if field_name not in known_fields:
            raise ValueError(f"Field {field_name} in day_time_map does not exist in fields!")

    while current_date <= end_date:

        day_name = current_date.strftime('%A')
        if day_name in [key for field in day_time_map for key in day_time_map[field].keys()]:
            draw_start = True
            if c_day != day_name:
                day_slot = 1
                c_day = day_name

            # Check if the whole weekend is out for Broadmeadow to set the rounds
            if any(current_date.date() in [(w - timedelta(days=1)).date(), w.date(), (w + timedelta(days=1)).date()] 
                for w in field_unavailabilities.get('Newcastle International Hockey Centre', {}).get('weekends', [])):
                pass
            elif round_indic:
                round_no += 1
                round_indic = False


            for field in fields:
                field_name = field.location
                if current_date == datetime(2025, 4, 6) and field_name == 'Central Coast Hockey Park':
                    print(1)
                # Check if the whole weekend is unavailable 
                if any(current_date.date() in [(w - timedelta(days=1)).date(), w.date(), (w + timedelta(days=1)).date()] 
                    for w in field_unavailabilities.get(field_name, {}).get('weekends', [])):
                    continue

                day_slot = 1

                for t in day_time_map[field_name][day_name]:

                    # Check if the whole day is unavailable 
                    if current_date.date() in [d.date() for d in field_unavailabilities.get(field_name, {}).get('whole_days', [])]:
                        continue

                    # Check if a partial day is unavailable 
                    if any(current_date.date() == pd.date() and t == pd.time() 
                        for pd in field_unavailabilities.get(field_name, {}).get('part_days', [])):
                        continue



                    timeslots.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'day': day_name,
                        'time': t.strftime('%H:%M'),
                        'week': week_number,
                        'day_slot': day_slot,
                        'field': field,
                        'round_no': round_no
                    })
                    day_slot += 1

        if current_date.strftime('%A') == 'Monday' and draw_start:
            week_number += 1
            day_slot = 1
            round_indic = True

        current_date += timedelta(days=1)

    print(f'Number of timeslots generated is {len(timeslots)}.')
    print(f'Total rounds found are {round_no}.')

    cur_week = 0
    count = 0
    for slot in timeslots:
        if slot['week'] != cur_week:
            count += 1
            cur_week = slot['week']

    print(f'Number of weekends with games on is {count}')
    return timeslots

def generate_X(folder_path, model, data):
    """
    Optimized version of game filtering to improve performance while tracking unavailable games.
    """
    # Create games dictionary
    games = {
        (t1.name, t2.name, t1.grade): (t1, t2, t1.grade)
        for i, t1 in enumerate(data['teams'])
        for t2 in data['teams'][i + 1:] if t1.grade == t2.grade
    }
    data['games'] = games
    print(f"Generated {len(games)} games.")  

    # Generate X variables
    timeslots = data['timeslots']
    teams = data['teams']
    phl_game_times = data['phl_game_times']
    day_time_map = data['day_time_map']
    num_dummy_timeslots = data['num_dummy_timeslots']

    X = {
        (t1.name, t2.name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location):
        model.NewBoolVar(f'X_{t1}_{t2}_{t.day}_{t.time}_{t.week}_{t.field.name}_{t.field.location}')
        for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items()
        for t in timeslots
        if grade_name not in  ['PHL', '2nd']
        if t.day # Stops dummy timeslot being used
        if t.field.location in {t1.club.home_field, t2.club.home_field}
        if ('Maitland' not in t1.name and 'Maitland' not in t2.name)
        if t.day in day_time_map[t.field.location] and datetime.strptime(t.time, "%H:%M").time() in day_time_map[t.field.location][t.day]
    }

    print(f'Decision variables {len(X)}')

    maitland_X = {
        (t1.name, t2.name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location):
        model.NewBoolVar(f'X_{t1}_{t2}_{t.day}_{t.time}_{t.week}_{t.field.name}_{t.field.location}')
        for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items()
        for t in timeslots
        if t.day # Stops dummy timeslot being used
        if t.field.location in {t1.club.home_field, t2.club.home_field} # Means that clubs that have multiple teams in the same grade will only play grudge matches at home
        if grade_name not in  ['PHL', '2nd'] 
        if ('Maitland' in t1.name or 'Maitland' in t2.name)
        if t.day in day_time_map[t.field.location] and datetime.strptime(t.time, "%H:%M").time() in day_time_map[t.field.location][t.day]

    }
    print(f'Maitland decision variables {len(maitland_X)}')

    sec_X = {
        (t1.name, t2.name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location):
        model.NewBoolVar(f'X_{t1}_{t2}_{t.day}_{t.time}_{t.week}_{t.field.name}_{t.field.location}')
        for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items()
        for t in timeslots
        if t.day # Stops dummy timeslot being used
        if grade_name == '2nd'
        if t.field.location in {t1.club.home_field, t2.club.home_field}
        if t.day in day_time_map[t.field.location] and datetime.strptime(t.time, "%H:%M").time() in day_time_map[t.field.location][t.day]
    # Conditions
        if t.field.name != 'SF' # Rule out south field
        if t.day != 'Friday'
        if not datetime.strptime(t.time, "%H:%M").time() < tm(11,30) # Allow 2nd to play at only special times
        if not datetime.strptime(t.time, "%H:%M").time() > tm(17,30) # Allow 2nd to play at only special times
        }
    print(f'2nds decision variables {len(sec_X)}')

    phl_X = {
        (t1.name, t2.name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location):
        model.NewBoolVar(f'X_{t1}_{t2}_{t.day}_{t.time}_{t.week}_{t.field.name}_{t.field.location}')
        for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items()
        for t in TIMESLOTS
        if t.day # Stops dummy timeslot being used
        if grade_name == 'PHL'
        if t.day in phl_game_times[t.field.location] and datetime.strptime(t.time, "%H:%M").time() in phl_game_times[t.field.location][t.day] # Allow phl to play at only special times
        if t.field.location in {t1.club.home_field, t2.club.home_field}
    # Conditions
        if t.field.name != 'SF' # Rule out south field
        if not (t.field.location == 'Maitland Park' and t.day == 'Friday') # No friday night games at maitland
        if (t.day != 'Friday' or (t.day == 'Friday' and 
            t1.name in ['Gosford PHL', 'Maitland PHL', 'Tigers PHL', 'Wests PHL', 'Souths PHL'] and   # Set it so that only the above teams play at Wyong on a Friday
            t2.name in ['Gosford PHL', 'Maitland PHL', 'Tigers PHL', 'Wests PHL', 'Souths PHL'] and
            t.field.location == 'Central Coast Hockey Park') or (t.day != 'Friday' or (t.day == 'Friday' and 
            t1.name in ['Maitland PHL', 'Tigers PHL', 'Wests PHL', 'Souths PHL'] and  # Only clubs with Juniors versions are to play on a friday night at Broadmeadow
            t2.name in ['Maitland PHL', 'Tigers PHL', 'Wests PHL', 'Souths PHL'] and
            t.field.location == 'Newcastle International Hockey Centre')) ) 


        if not datetime.strptime(t.date, '%Y-%m-%d') in [datetime(2025, 5, 9), datetime(2025, 5, 23), datetime(2025, 6, 6), datetime(2025, 6, 27), datetime(2025, 7, 4)] # These are dates the Juniors convenor said no to
        if not (t.field.location == 'Central Coast Hockey Park' and t.day == 'Friday' and datetime.strptime(t.date, '%Y-%m-%d') < datetime(2025, 5, 1))  # Do not schedule Friday games at Wyong before May 1st

        }
    print(f'PHL decision variables {len(phl_X)}')

    X.update(maitland_X)
    X.update(phl_X)
    X.update(sec_X)

    filtered_games = set(X.keys())  
    unavailable_games = set()
    conflicting_matchups = []

    club_names = [c.name.lower() for c in data['clubs']]

    for file in os.listdir(folder_path):
        if not file.endswith("_noplay.xlsx"):
            continue

        club_name = file.split("_noplay.xlsx")[0]
        club_teams = get_teams_from_club(club_name, teams)

        file_path = os.path.join(folder_path, file)

        if club_name.lower() not in club_names:
            raise ValueError(f"Club {club_name} in {file} does not exist in clubs!")

        if '~$' in club_name: # Filter out any temporary files
            continue

        # Read only required columns and convert dates in bulk
        club_noplay = pd.read_excel(file_path, sheet_name="club_noplay", usecols=["whole_weekend", "whole_day", "timeslot"])
        teams_noplay = pd.read_excel(file_path, sheet_name="teams_noplay", usecols=["team", "whole_weekend", "whole_day", "timeslot"])
        team_conflicts = pd.read_excel(file_path, sheet_name="team_conflicts", usecols=["team1", "team2"])

        for df in [club_noplay, teams_noplay]:
            df.fillna(np.nan, inplace=True)
            df["whole_weekend"] = pd.to_datetime(df["whole_weekend"], format="%d/%m/%Y", errors="coerce")
            df["whole_day"] = pd.to_datetime(df["whole_day"], format="%d/%m/%Y", errors="coerce")
            df["timeslot"] = pd.to_datetime(df["timeslot"], format="%d/%m/%Y %H:%M", errors="coerce")

        # Remove club-wide restricted times 
        for _, row in club_noplay.iterrows():
            if pd.notna(row["whole_weekend"]):
                week_num = row["whole_weekend"].isocalendar()[1]
                to_remove = {k for k in filtered_games if datetime.strptime(k[7], "%Y-%m-%d").isocalendar()[1] == week_num and (k[0] in club_teams or k[1] in club_teams)}
                unavailable_games.update(to_remove)
                filtered_games -= to_remove
            if pd.notna(row["whole_day"]):
                day_date = row["whole_day"].date()
                to_remove = {k for k in filtered_games if datetime.strptime(k[7], "%Y-%m-%d").date() == day_date and (k[0] in club_teams or k[1] in club_teams)}
                unavailable_games.update(to_remove)
                filtered_games -= to_remove
            if pd.notna(row["timeslot"]):
                date_time = row["timeslot"]
                to_remove = {k for k in filtered_games if datetime.strptime(k[7], "%Y-%m-%d").date() == date_time.date() and k[5] == date_time.strftime("%H:%M") and (k[0] in club_teams or k[1] in club_teams)}
                unavailable_games.update(to_remove)
                filtered_games -= to_remove

        # Remove team-specific restrictions 
        valid_teams = {k[0].lower() for k in filtered_games} | {k[1].lower() for k in filtered_games}

        for _, row in teams_noplay.iterrows():
            team = row["team"]
            if team.lower() not in valid_teams:
                raise ValueError(f"Team {team} in {file} does not exist in games!")

            if pd.notna(row["whole_weekend"]):
                week_num = row["whole_weekend"].isocalendar()[1]
                to_remove = {k for k in filtered_games if team in k[:2] and datetime.strptime(k[7], "%Y-%m-%d").isocalendar()[1] == week_num}
                unavailable_games.update(to_remove)
                filtered_games -= to_remove
            if pd.notna(row["whole_day"]):
                day_date = row["whole_day"].date()
                to_remove = {k for k in filtered_games if team in k[:2] and datetime.strptime(k[7], "%Y-%m-%d").date() == day_date}
                unavailable_games.update(to_remove)
                filtered_games -= to_remove
            if pd.notna(row["timeslot"]):
                date_time = row["timeslot"]
                to_remove = {k for k in filtered_games if team in k[:2] and datetime.strptime(k[7], "%Y-%m-%d").date() == date_time.date() and k[5] == date_time.strftime("%H:%M")}
                unavailable_games.update(to_remove)
                filtered_games -= to_remove

        # Record conflicting team matchups
        conflicting_matchups.extend(list(team_conflicts.itertuples(index=False, name=None)))

        # Validate that all teams in conflicting_matchups exist in valid_teams
        for team1, team2 in conflicting_matchups:
            if team1.lower() not in valid_teams or team2.lower() not in valid_teams:
                raise ValueError(f"Conflicting teams {team1} and {team2} are not in the valid teams list!")

    final_games = {k: X[k] for k in filtered_games}

    dummy_X = {
        (t1.name, t2.name, grade_name, i):
        model.NewBoolVar(f'X_dummy_{t1}_{t2}_{i}')
        for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items()
        for i in range(num_dummy_timeslots)
        }


    return final_games, dummy_X, conflicting_matchups, unavailable_games

class SaveStateCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, interval=1):
        super().__init__()
        self.variables = variables                
        self.interval = interval
        self.counter = 0
        self.best_solution = None
        self.best_obj_value = float('inf')

    def on_solution_callback(self):
        obj = self.ObjectiveValue()
        if obj < self.best_obj_value:
            self.counter += 1
            self.best_obj_value = obj
            self.best_solution = {v.Name(): self.Value(v) for v in self.variables}

            if self.counter == 1:
                self.save_checkpoint()
                print(f"Model is feasible, initial solution saved.")

            elif self.counter % self.interval == 0:
                self.save_checkpoint()
                print(f"Callback hit at solution {self.counter}.")


    def save_checkpoint(self):
        base_dir = "checkpoints"
        os.makedirs(base_dir, exist_ok=True)

        existing_runs = [d for d in os.listdir(base_dir) if re.match(r'run_\d+', d)]
        run_numbers = [int(re.search(r'\d+', run).group()) for run in existing_runs]
        next_run_number = max(run_numbers, default=0) + 1

        run_dir = os.path.join(base_dir, f"run_{next_run_number}")
        os.makedirs(run_dir, exist_ok=True)

        checkpoint_path = os.path.join(run_dir, "checkpoint.pkl")
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump((self.best_solution, self.best_obj_value), f)
            print(f"Checkpoint saved in {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path="checkpoint.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.best_solution, self.best_obj_value = pickle.load(f)
            print("Checkpoint loaded.")
            return True
        return False

def create_schedule(game_vars, data, constraints, model):

    X, Y = game_vars
    X.update(Y)
    print(f"Generated {len(X)} decision variables.")  

    prior_val = 0
    for constraint in constraints:
        constraint.apply(model, X, data)
        print(constraint.__class__.__name__, f" {len(model.Proto().constraints) - prior_val} constraints added.")
        prior_val = len(model.Proto().constraints) 

    penalties_dict = data.get('penalties', {})

    print(f'Number of penalties per constraint: {[(name, len(info.get("penalties", []))) for name, info in penalties_dict.items()]}')

    total_penalty = sum(
        info['weight'] * sum(info['penalties'])  # Multiply weight by sum of violations
        for info in penalties_dict.values() if 'penalties' in info
    )

    model.Maximize(
        sum(X.values()) - sum(Y.values()) - total_penalty  
    )
    print(f"Total constraints in model: {len(model.Proto().constraints)}")

    solver = cp_model.CpSolver()

    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_probing_level = 0  # Avoid aggressive pruning to analyze infeasibility

    solver.parameters.max_time_in_seconds = 28800 # this last one was close, maybe double it and be ok

    all_vars = list(X.values())
    callback = SaveStateCallback(all_vars, interval=1)

    checkpoint_path = data.get('checkpoint_path', None)
    if checkpoint_path is not None:
        if not callback.load_checkpoint(path = checkpoint_path):
            print("No checkpoint found, starting fresh.")

    # Solve—note callback is positional
    status = solver.Solve(model, callback)

    status = solver.Solve(model, solution_callback=callback)
    # Print the best solution found
    if callback.best_solution:
        print("Best solution found:", callback.best_solution)
        print("Objective value:", callback.best_obj_value)

    print(f"Solver status: {solver.StatusName()}") 
    X_outcome = X

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        X_solution = {key: solver.Value(var) for key, var in X.items()}
        for constraint_name, info in data.get('penalties', {}).items():
            if 'penalties' in info:
                unresolved_penalties = [p for p in info['penalties'] if solver.Value(p) > 0]
                data['penalties'][constraint_name]['unresolved'] = unresolved_penalties
        return X_outcome, X_solution, data
    else:
        print("No feasible schedule found.")

        # # Investigate constraints that might be failing
        # infeasible_constraints = []

        # for constraint in constraints:
        #     print(f"Checking constraint: {constraint.__class__.__name__}")

        #     try:
        #         if hasattr(constraint, "apply") and callable(getattr(constraint, "apply", None)):
        #             test_model = cp_model.CpModel()
        #             test_X = {k: test_model.NewBoolVar(str(k)) for k in X.keys()}
        #             constraint.apply(test_model, test_X, data)

        #             test_solver = cp_model.CpSolver()
        #             test_status = test_solver.Solve(test_model)
        #             test_solver.parameters.max_time_in_seconds = 600
        #             if test_status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        #                 infeasible_constraints.append(constraint.__class__.__name__)
        #                 print(f"❌ Constraint {constraint.__class__.__name__} is likely causing infeasibility.")
        #             else:
        #                 print(f"✅ Constraint {constraint.__class__.__name__} appears feasible.")
        #         else:
        #             print(f"Skipping {constraint.__class__.__name__} (no apply method).")
        #     except Exception as e:
        #         print(f"Error while testing {constraint.__class__.__name__}: {e}")

        # if infeasible_constraints:
        #     print(f"\n🚨 The following constraints are likely causing infeasibility:\n{infeasible_constraints}\n")
        # else:
        #     print("\n⚠️ No single constraint was found to be infeasible, check interactions between them.")

        return X_outcome, {}, data

def max_games_per_grade(
    grades: List[Grade],
    max_rounds: int
) -> Dict[str, int]:
    """
    Given a list of Grade objects (each with num_teams) and the maximum
    number of rounds, returns a dict mapping grade.name → max games per team.

    In each round you can have floor(T/2) matches, so over R rounds:
      total_matches ≤ R * floor(T/2)
    and since each team plays g games, total_matches = g * T / 2 must be integer.
    """
    games_per_grade: Dict[str, int] = {}

    for grade in grades:
        T = grade.num_teams
        if T < 2:
            games_per_grade[grade.name] = 0
            continue

        # maximum matches across all rounds
        max_matches = max_rounds * (T // 2)

        # g0 = floor( 2 * max_matches / T )
        g0 = (2 * max_matches) // T
        # can't exceed one game per round
        g0 = min(g0, max_rounds)

        # ensure g0*T is even → if T odd, force g0 even
        if T % 2 == 1 and (g0 % 2) == 1:
            g0 -= 1

        games_per_grade[grade.name] = g0

    return games_per_grade


day_time_map = {'Newcastle International Hockey Centre':
                    { 
                       # 'Saturday':[tm(12, 30), tm(14, 00), tm(15, 30), tm(17, 00)],
                     'Sunday':[tm(8, 30), tm(10, 0), tm(11,30), tm(13, 00), tm(14, 30), tm(16, 00), tm(17, 30), tm(19, 00)]
                     },
                'Maitland Park':
                    {
                       # 'Saturday':[tm(12, 30), tm(14, 00), tm(15, 30), tm(17, 00)], # Needed because Port and maitland want to play on a Saturday
                     'Sunday':[tm(9, 00), tm(10, 30), tm(12, 00), tm(13, 30), tm(15, 00), tm(16,30)]
                     }
}

assert len(day_time_map['Maitland Park']['Sunday']) >= get_club_from_clubname('Maitland', CLUBS).num_teams, "Not enough timeslots for Maitland teams!"

phl_game_times = {'Newcastle International Hockey Centre':
                    {'Friday':[tm(19, 00)],
                    'Sunday':[tm(11,30), tm(13, 00), tm(14, 30), tm(16, 00)]
                    },    
                    'Central Coast Hockey Park':
                        {'Friday':[tm(20, 00)],
                         'Sunday': [tm(15, 00)]
                         },  
                    'Maitland Park':
                    {
                     'Sunday':[ tm(12, 00), tm(13, 30), tm(15, 00), tm(16, 30)]
                     }
                    } 

field_unavailabilities = {'Maitland Park': 
                            {'weekends':[datetime(2025, 4, 19), datetime(2025, 4, 12), datetime(2025, 5, 10), datetime(2025, 5, 24), datetime(2025, 6, 28), datetime(2025,5,3) , datetime(2025,6,7) ], # # Enter the saturday of the weekend to be safe and definitely catch Fridays as being out too #
                            'whole_days':[datetime(2025, 4, 25),],
                            'part_days':[],
                            },
                        'Newcastle International Hockey Centre': 
                            {'weekends':[datetime(2025, 4, 19), datetime(2025,5,3), datetime(2025,6,7)],
                            'whole_days':[datetime(2025, 4, 25), datetime(2025, 5, 31)],
                            'part_days':[datetime(2025, 6,1, 8, 30), datetime(2025, 6, 1, 10,0), datetime(2025,6,1,11,30), ],
                            },
                        'Central Coast Hockey Park': 
                            {'weekends':[ datetime(2025, 4, 19), datetime(2025, 4, 5),datetime(2025,5,3), datetime(2025,6,7),],#
                            'whole_days':[datetime(2025, 4, 25), ],
                            'part_days':[],
                            },
                        }

"""
For field unavailabilities, enter the SATURDAY to rule out the whole weekend.
"""

start = datetime(2025, 3, 21) # Set start date a few days before weekend
end = datetime(2025, 9, 2)  # Set end date a days after weekend

weekend_count = sum(1 for i in range((end - start).days + 1)
                    if (start + timedelta(days=i)).weekday() in {5, 6}) / 2 # 5 = Saturday, 6 = Sunday

max_rounds = 21
num_rounds = max_games_per_grade(GRADES, max_rounds)
num_rounds['max'] = max_rounds

for grade, rounds in num_rounds.items():
    grade = next((t for t in GRADES if t.name == grade), None)
    if grade is not None:
        grade.set_games(rounds)

print(f"Number of weekends: {weekend_count}")

club_days = {'Crusaders': datetime(2025, 6, 22),
             'Wests': datetime(2025, 7, 13),
             'University': datetime(2025, 7, 27),
             'Tigers': datetime(2025, 7, 6),
             'Port Stephens': datetime(2025, 7, 20),
             }

"""
On club days, pick one field and schedule all games there, prioritise morning and grade order
"""

preference_no_play = {'Maitland':[{'date': '2025-07-20', 'field_location':'Newcastle International Hockey Centre'},
                                  {'date': '2025-08-24', 'field_location':'Newcastle International Hockey Centre' },
                                    ],
                    'Norths':[{'team_name': 'Norths PHL', 'date': '2025-03-23', 'time':'11:30'},
                              {'team_name': 'Norths PHL',  'date': '2025-03-23', 'time':'13:00'},
                              {'team_name': 'Norths PHL',  'date': '2025-03-23', 'time':'14:30'},
                              {'team_name': 'Norths PHL',  'date': '2025-03-23', 'time':'16:00'},
                              ],

                      }

phl_preferences = {'preferred_dates' :[]}
"""
On preference_no_play, use same keys as either Field or Timeslot attributes, special exemption for field_location and field_name, but if specifying a team, must enter the key team_name
"""
merged_dict = defaultdict(lambda: defaultdict(list))


# Iterate over both dictionaries
for d in (phl_game_times, day_time_map):
    for field, days in d.items():
        for key, times in days.items():
            if field in merged_dict and key in merged_dict[field]:
                merged_dict[field][key].extend(times)
            else:
                merged_dict[field][key] = list(times)

for field in merged_dict:
    for key in merged_dict[field]:
        merged_dict[field][key] = list(dict.fromkeys(merged_dict[field][key]))
        merged_dict[field][key].sort()

timeslots = generate_timeslots(start, end, merged_dict, FIELDS, field_unavailabilities)

TIMESLOTS = [Timeslot(date=t['date'], day=t['day'], time=t['time'], week=t['week'], day_slot=t['day_slot'], field=t['field'], round_no=t['round_no']) for t in timeslots]

max_day_slot_per_field = {field.location: max(t.day_slot for t in TIMESLOTS if t.field.location == field.location) for field in FIELDS}

UNAVAILABILITY_PATH = r'data\2025\noplay'

print(num_rounds)


# In[5]:


# Instantiate data

data = {'teams': TEAMS, 'grades': GRADES, 'fields': FIELDS, 'timeslots': TIMESLOTS, 'clubs': CLUBS, 'num_rounds': num_rounds, 'day_time_map': day_time_map, 'phl_game_times': phl_game_times, 'phl_preferences': phl_preferences, 'max_day_slot_per_field': max_day_slot_per_field, 'field_unavailabilities': field_unavailabilities, 'club_days': club_days}
model = cp_model.CpModel()
data['num_dummy_timeslots'] = 3
X, Y, conflicts, unavailable_games = generate_X(UNAVAILABILITY_PATH, model, data)
data['unavailable_games'] = unavailable_games
data['preference_no_play'] = preference_no_play
data['team_conflicts'] = conflicts
future_games = []
# future_games = [{'team1': 'Tigers PHL', "team2": 'Wests PHL', "grade": "PHL", "day": "Friday", "date": "2025-06-20", "field_location": "Newcastle International Hockey Centre"},
#                 {'team1': 'Souths PHL', "team2": 'Maitland PHL', "grade": "PHL", "day": "Friday", "date": "2025-05-30", "field_location": "Newcastle International Hockey Centre"},
#                 {'team1': 'Tigers PHL', "grade": "PHL", "day": "Friday", "date": "2025-08-01", "field_location": "Newcastle International Hockey Centre"},
#                 ]


# future_games.extend([{'team1': i, "date": "2025-07-27", "field_location": "Newcastle International Hockey Centre"} for i in get_teams_from_club('University', TEAMS)])
# future_games.extend([{'team1': i, "date": "2025-06-22", "field_location": "Newcastle International Hockey Centre"} for i in get_teams_from_club('Crusaders', TEAMS)])
# future_games.extend([{'team1': i, "date": "2025-07-13", "field_location": "Newcastle International Hockey Centre"} for i in get_teams_from_club('Wests', TEAMS)])
# future_games.extend([{'team1': i, "date": "2025-07-06", "field_location": "Newcastle International Hockey Centre"} for i in get_teams_from_club('Tigers', TEAMS)])
# future_games.extend([{'team1': i, "date": "2025-07-20", "field_location": "Newcastle International Hockey Centre"} for i in get_teams_from_club('Port Stephens', TEAMS)])

# future_games.extend([{'team1': i, "date": "2025-06-15", "field_location": "Maitland Park"} for i in get_teams_from_club('Port Stephens', TEAMS)])



# In[6]:


def enforce_schedule_from_xlsx(model, X, xlsx_filename, max_week, data):
    """
    Reads an Excel file (with multiple sheets for each week) and updates the constraint model 
    by enforcing the correct game variables up to and including the given max_week.

    Args:
        model (cp_model.CpModel): The constraint model.
        X (dict): Decision variables.
        xlsx_filename (str): Path to the Excel file.
        max_week (int): The last week to enforce scheduling for.

    Returns:
        None (modifies the model directly).
    """
    scheduled_games = set()  

    with pd.ExcelFile(xlsx_filename) as xlsx_data:

        for week_num in range(1, max_week + 1):
            sheet_name = f"Week {week_num}"
            if sheet_name not in xlsx_data.sheet_names:
                print(f"Warning: Sheet '{sheet_name}' not found in the Excel file. Skipping...")
                continue  

            df = pd.read_excel(xlsx_data, sheet_name)

            required_columns = {"Team 1", "Team 2", "Grade", "Field Name", "Date", "Day", "Time"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Sheet '{sheet_name}' is missing one or more required columns: {required_columns}")

            for _, row in df.iterrows():
                if (pd.isna(row["Team 1"])  and pd.isna(row["Team 2"])) or str(row["Team 1"]).strip().lower() == "byes and no games":
                    break
                team1 = str(row.get('Team 1', '')).strip() if not pd.isna(row.get('Team 1')) else ''
                team2 = str(row.get('Team 2', '')).strip() if not pd.isna(row.get('Team 2')) else ''
                grade = str(row.get('Grade', '')).strip() if not pd.isna(row.get('Grade')) else ''
                field_name = str(row.get('Field Name', '')).strip() if not pd.isna(row.get('Field Name')) else ''
                field_location = str(row.get('Field Location', '')).strip() if not pd.isna(row.get('Field Location')) else ''
                date = str(row.get('Date', '')).strip() if not pd.isna(row.get('Date')) else ''
                day = str(row.get('Day', '')).strip() if not pd.isna(row.get('Day')) else ''
                time = str(row.get('Time', '')).strip() if not pd.isna(row.get('Time')) else ''
                round_no = int(row.get('Round', '')) if not pd.isna(row.get('Round')) else 0
                day_slot = int(row.get('Day Slot', 0)) if not pd.isna(row.get('Day Slot')) else 0

                try:
                    game_date = datetime.strptime(date, "%Y-%m-%d").date().strftime('%Y-%m-%d') if date else ''
                    game_time = datetime.strptime(time, "%H:%M").time().strftime('%H:%M') if time else ''
                except ValueError:
                    print(f"Skipping invalid date/time in row: {date} {time}")
                    game_date, game_time = '', ''

                ################################################

                    # Special import conditions to make draw work right now

                ################################################

                matching_keys = [
                    key for key in X
                    if len(key) > 5 and # Filter out dummy variables
                    (
                        ((team1 == key[0] and team2 == key[1]) or 
                        (team2 == key[0] and team1 == key[1]))  
                        and (grade == key[2] if grade else True)  
                        and (day == key[3] if day else True)  
                        and (game_time == key[5] if game_time else True) 
                        and (game_date == key[7] if game_date else True) 
                        and (field_name == key[9] if field_name else True)  
                        and (week_num == key[6] if week_num is not None else True)  
                    )
                ]

                if len(matching_keys) == 0:
                    if team1 not in [t.name for t in data['teams']] or team2 not in [t.name for t in data['teams']]:
                        print(f"Warning: Ignoring game {team1} vs {team2} in {sheet_name} because one or both teams are not in the dataset.")
                        continue
                    if team1.split(' ')[-1] != grade:
                        raise ValueError(f"Error: Team {team1} does not match the expected grade {grade}.")

                    print(f"Warning: Invalid field or time detected for {team1} vs {team2} in {sheet_name}. Creating dummy variable.")

                    dummy_key = (team1, team2, grade, day, day_slot, game_time, week_num, game_date, round_no, field_name, field_location)
                    field_obj = next((f for f in data['fields'] if (f.name == field_name and f.location == field_location) or (field_name == '' and f.location == field_location)), None)
                    if field_obj is None:
                        raise ValueError(f"Error: Field {field_name} not found in the dataset.")
                    data['timeslots'].append(Timeslot(date=game_date, day=day, time=game_time, week=week_num, day_slot=day_slot, field=field_obj, round_no=round_no))  # Add dummy timeslot to data
                    print(dummy_key)
                    X[dummy_key] = model.NewBoolVar(f'X_{team1}_{team2}_dummy')
                    model.Add(X[dummy_key] == 1)
                    scheduled_games.add(dummy_key)
                    continue

                if len(matching_keys) != 1:
                    print(grade, team1, team2, day, game_time, game_date, field_name, week_num)
                    print(len(scheduled_games))
                    print(matching_keys)
                    raise ValueError(f"Error: Expected one match for game {team1} vs {team2} in {sheet_name}, found {len(matching_keys)}")

                game_key = matching_keys[0]
                model.Add(X[game_key] == 1)

                scheduled_games.add(game_key)

        keys_to_delete = [key for key in X if len(key) > 5 and int(key[6]) <= max_week and key not in scheduled_games]

        for key in keys_to_delete:
            del X[key]

        assert len(scheduled_games) == len(set(scheduled_games)), "Error: Duplicate games detected in the schedule."

    print(f"Successfully enforced {len(scheduled_games)} scheduled games from {xlsx_filename} up to Week {max_week}.")

def enforce_future_schedule_from_xlsx(model, X, xlsx_filename, future_weeks):
    """
    Enforces games from selected future weeks in an Excel schedule.
    If a game has multiple valid assignments, it ensures at least one is scheduled.
    If no matching game is found, it raises an error.

    Args:
        model (cp_model.CpModel): The constraint model.
        X (dict): Decision variables mapping game attributes to model variables.
        xlsx_filename (str): Path to the Excel file.
        future_weeks (list of int): List of week numbers to enforce.

    Returns:
        None (modifies the model directly).
    """
    scheduled_games = set() 
    with pd.ExcelFile(xlsx_filename) as xlsx_data:
        for week_num in future_weeks:
            sheet_name = f"Week {week_num}"
            if sheet_name not in xlsx_data.sheet_names:
                print(f"Warning: Sheet '{sheet_name}' not found in the Excel file. Skipping...")
                continue  

            df = pd.read_excel(xlsx_data, sheet_name)

            required_columns = {"Team 1", "Team 2", "Grade", "Field Name", "Date", "Day", "Time"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Sheet '{sheet_name}' is missing one or more required columns: {required_columns}")

            for _, row in df.iterrows():
                if (pd.isna(row["Team 1"])  and pd.isna(row["Team 2"])) or str(row["Team 1"]).strip().lower() == "byes and no games":
                    break  
                # Read and clean data with safety checks
                # team1 = str(row.get('Team 1', '')).strip().rsplit(' ', 1)[0].strip() if not pd.isna(row.get('Team 1')) else ''
                # team2 = str(row.get('Team 2', '')).strip().rsplit(' ', 1)[0].strip() if not pd.isna(row.get('Team 2')) else ''
                team1 = str(row.get('Team 1', '')).strip() if not pd.isna(row.get('Team 1')) else ''
                team2 = str(row.get('Team 2', '')).strip() if not pd.isna(row.get('Team 2')) else ''
                grade = str(row.get('Grade', '')).strip() if not pd.isna(row.get('Grade')) else ''
                field_name = str(row.get('Field Name', '')).strip() if not pd.isna(row.get('Field Name')) else ''


                date = str(row.get('Date', '')).strip() if not pd.isna(row.get('Date')) else ''
                day = str(row.get('Day', '')).strip() if not pd.isna(row.get('Day')) else ''
                time = str(row.get('Time', '')).strip() if not pd.isna(row.get('Time')) else ''

                try:
                    game_date = datetime.strptime(date, "%Y-%m-%d").date().strftime('%Y-%m-%d') if date else ''
                    game_time = datetime.strptime(time, "%H:%M").time().strftime('%H:%M') if time else ''
                except ValueError:
                    print(f"Skipping invalid date/time in {sheet_name}: {date} {time}")
                    continue

                matching_keys = [
                    key for key in X
                    if len(key) > 5 and  # Filter out dummy variables
                    (
                        # Check if teams match in either order
                        # (not team1 or not team2 or 
                        #  ((team1 == key[0].rsplit(' ', 1)[0].strip() and team2 == key[1].rsplit(' ', 1)[0].strip()) or
                        #   (team2 == key[0].rsplit(' ', 1)[0].strip() and team1 == key[1].rsplit(' ', 1)[0].strip())))
                        (not team1 or not team2 or 
                         ((team1 == key[0] and team2 == key[1]) or
                          (team2 == key[0] and team1 == key[1])))
                        and (not grade or key[2] == grade)
                        and (not day or key[3] == day)
                        and (not game_time or key[5] == game_time)
                        and (not game_date or key[7] == game_date)
                        and (week_num is None or key[6] == week_num)  
                        and (not field_name or key[9] == field_name)  
                    )
                ]

                if not matching_keys:
                    print(grade, team1, team2, day, game_time, game_date, field_name, week_num)
                    print(len(scheduled_games))
                    print(matching_keys)
                    raise ValueError(f"Error: No matching game found for {team1} vs {team2} in Week {week_num}")

                if len(matching_keys) > 1:
                    print(f"Warning: Multiple matches found for {team1} vs {team2} in Week {week_num}. Enforcing at least one.")
                    model.Add(sum(X[key] for key in matching_keys) >= 1)
                else:
                    model.Add(X[matching_keys[0]] == 1)

                scheduled_games.add(matching_keys[0])

    print(f"Successfully enforced games for future weeks: {future_weeks}")

def enforce_future_games(model, X, future_games, allow_multiple_matches=False):
    """
    Enforce scheduling of specific games provided in a list of dictionaries, updating the constraint model.
    Unlike enforce_schedule_from_xlsx, this function does not remove any other games.

    Args:
        model (cp_model.CpModel): The constraint model.
        X (dict): Decision variables mapping game attributes to model variables.
        future_games (list of dict): List of game dictionaries specifying games to enforce.
        allow_multiple_matches (bool, optional): If True, allows multiple matches and ensures at least one is set to 1. Defaults to False.

    Returns:
        None (modifies the model directly).
    """
    for game in future_games:
        team1 = game.get('team1', '').strip()
        team2 = game.get('team2', '').strip()
        grade = game.get('grade', '').strip()
        field_name = game.get('field_name', None)
        field_location = game.get('field_location', None)
        round_no = game.get('round_no', None)
        day_slot = game.get('day_slot', None)
        game_date = game.get('date', '').strip()
        day = game.get('day', '').strip()
        game_time = game.get('time', '').strip()
        week_num = game.get('week', None)
        if week_num is not None:
            week_num = int(week_num)

        matching_keys = [
            key for key in X
            if len(key) > 5 and  # Filter out dummy variables
            (
                (team1 and team2 and (
                    (team1 == key[0] and team2 == key[1]) or
                    (team2 == key[0] and team1 == key[1])
                ))
                or (team1 and not team2 and (team1 == key[0] or team1 == key[1]))
                or (team2 and not team1 and (team2 == key[0] or team2 == key[1]))
            )
            and (not grade or key[2] == grade)  
            and (not day or key[3] == day) 
            and (not game_time or key[5] == game_time)  
            and (not game_date or key[7] == game_date)  
            and (week_num is None or key[6] == week_num)
            and (field_name is None or key[9] == field_name) 
            and (field_location is None or key[10] == field_location)  
            and (day_slot is None or key[4] == day_slot)  
            and (round_no is None or key[8] == round_no) 
        ]

        if not matching_keys:
            print(f"Warning: No matching game found for {team1 or team2} on {game_date or 'any date'} at {game_time or 'any time'} ({field_name or 'any field'}).")
            continue

        if allow_multiple_matches and len(matching_keys) > 1:
            print(f"Warning: Multiple matches found for {team1 or team2} on {game_date or 'any date'} at {game_time or 'any time'}. Ensuring at least one is scheduled.")
            print(matching_keys)
            model.Add(sum(X[key] for key in matching_keys) >= 1)
        elif len(matching_keys) == 1:
            model.Add(X[matching_keys[0]] == 1)
        else:
            raise ValueError(f"Error: Multiple matches found for {team1 or team2} but allow_multiple_matches is False.")

    print(f"Successfully enforced {len(future_games)} future games.")

def enforce_no_maitland_home(model, X, weeks):
    for key in X:
        if len(key) > 5 and key[10] == 'Maitland Park' and int(key[6]) in weeks:
            model.Add(X[key] == 0)

current_week = 0
data['current_week'] = current_week

# enforce_future_games(model, X, future_games, allow_multiple_matches=True)
# enforce_schedule_from_xlsx(model, X, r"draws/V6/schedule Wk1-5.xlsx", current_week, data)

# # enforce_future_schedule_from_xlsx(model, X_final, r"draws\V3\schedule V2.xlsx", [ 9, 10, 11, 13, 14, 16, 17, 18, 19])
# # enforce_no_maitland_home(model, X_final, [ 8, 10, 14, 17, 20, 24 ]) # 1, 


# In[ ]:


constraints =  [EnsureEqualGamesAndBalanceMatchUps(),  NoDoubleBookingTeamsConstraint(), NoDoubleBookingFieldsConstraint(),
                FiftyFiftyHomeandAway(), MaxMaitlandHomeWeekends(), ClubDayConstraint(), 
                 PHLAndSecondGradeTimes(),  PHLAndSecondGradeAdjacency() , 
                 TeamConflictConstraint(), AwayAtMaitlandGrouping(),
                 ClubVsClubAlignment(), ClubGradeAdjacencyConstraint(), MaitlandHomeGrouping(), EqualMatchUpSpacingConstraint(),
                 EnsureBestTimeslotChoices(), ]#MinimiseClubsOnAFieldBroadmeadow(),  MaximiseClubsPerTimeslotBroadmeadow(),]#, PreferredTimesConstraint()]

X_outcome, X_solution, data = create_schedule(game_vars=(X, Y), data=data, constraints=constraints, model=model)


# In[ ]:


def get_field_by_name(name, FIELDS):
    for field in FIELDS:
        if field.name == name:
            return field  
    raise ValueError(f"Field {name} not found in field list.") 

def get_grade_by_name(name, GRADE):
    for grade in GRADE:
        if grade.name == name:
            return grade  
    raise ValueError(f"Grade {name} not found in grade list.")

def convert_X_to_roster(X: Dict, data: Dict) -> Roster:
    weekly_games: Dict[int, List[Game]] = {}
    all_teams: Set[str] = {team.name for team in data['teams']}  

    for key, var in X.items():
        if len(key) > 5 and var > 0:
            (team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, location) = key
            field_class = get_field_by_name(field_name, data['fields'])
            grade_class = get_grade_by_name(grade, data['grades'])
            game = Game(
                team1=team1,
                team2=team2,
                timeslot=Timeslot(date=date, day=day, time=time, week=week, field=field_class, day_slot=day_slot, round_no=round_no),
                field=field_class,
                grade=grade_class
            )
            weekly_games.setdefault((week, round_no), []).append(game)
        elif var > 0:
            (team1, team2, grade, i) = key
            grade_class = get_grade_by_name(grade, data['grades'])

            game = Game(
                team1=team1,
                team2=team2,
                timeslot=Timeslot(date='', day='', time='', week=0, field=PlayingField(location='',name=''), day_slot=0, round_no=0),
                field=PlayingField(location='',name=''),
                grade=grade_class
            )

    weekly_draws = []
    for (week, round_no), games in sorted(weekly_games.items()):
        teams_played = {game.team1 for game in games} | {game.team2 for game in games}
        bye_teams = list(all_teams - teams_played)  
        weekly_draws.append(WeeklyDraw(week=week, games=games, bye_teams=bye_teams, round_no=round_no))

    return Roster(weeks=weekly_draws)

def export_roster_to_excel(roster: Roster, data:Dict, filename="schedule.xlsx"):
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        workbook = writer.book  
        teams = data['teams']
        for weekly_draw in roster.weeks:
            week = weekly_draw.week

            # Extract games data
            game_data = [
                [game.timeslot.round_no, game.team1, game.team2, game.grade.name, game.field.name, game.field.location, game.timeslot.date, game.timeslot.day, game.timeslot.time, game.timeslot.day_slot]
                 if get_club_object(game.team1, teams).home_field == game.field.location else 
                 [game.timeslot.round_no, game.team2, game.team1, game.grade.name, game.field.name, game.field.location, game.timeslot.date, game.timeslot.day, game.timeslot.time, game.timeslot.day_slot]
                 for game in weekly_draw.games
            ]
            df_games = pd.DataFrame(game_data, columns=["Round", "Team 1", "Team 2", "Grade", "Field Name", "Field Location", "Date", "Day", "Time", "Day Slot"])
            df_games.to_excel(writer, sheet_name=f"Week {week}", index=False)

            bye_data = []

            for team in weekly_draw.bye_teams:
                grade = team.rsplit(' ', 1)[1].strip()
                bye_data.append([grade, team])

            df_byes = pd.DataFrame(bye_data, columns=["Grade", "Teams with Bye or No Game"])

            sheet = writer.sheets[f"Week {week}"]  
            last_row = len(df_games) + 3 

            if not df_byes.empty:
                sheet.write(last_row, 0, "Byes and No Games", workbook.add_format({"bold": True}))
                df_byes.to_excel(writer, sheet_name=f"Week {week}", startrow=last_row + 1, index=False, header=True)

def check_back_to_back_byes(roster: Roster) -> List[str]:
    """
    Ensures that no team has back-to-back byes in the season schedule.
    Returns a list of teams that have consecutive byes.
    """
    previous_week = []
    current_week = []
    violations = {}
    curr_round = 1
    for weekly_draw in roster.weeks:
        if len(weekly_draw.games) == 0:
            continue
        if weekly_draw.bye_teams:
            if weekly_draw.round_no == 1:
                previous_week = weekly_draw.bye_teams
                continue
            else:
                if weekly_draw.round_no != curr_round:
                    current_week = weekly_draw.bye_teams
                    if any(team in previous_week for team in current_week):
                        violations[weekly_draw.week] = [team for team in current_week if team in previous_week]
                    previous_week = current_week
                    curr_round = weekly_draw.round_no

    return violations

def check_maitland_complex_limitations(roster: Roster, data: Dict) -> List[str]:
    """
    Check to see that when maitland plays at home, they all play at home
    Check to see that when maitland at home, the grades are played in order
    Check that when maitland plays at home, that all of one club comes up for the day
    """
    teams = data['teams']
    clubs = data['clubs']
    violations = {}
    weekly_proportions = []
    away_game_clubs_list = []
    maitland_team_order_list = []

    for weekly_draw in roster.weeks:

        away_game_clubs = defaultdict(list) 
        maitland_team_order = defaultdict(lambda: defaultdict(list)) 
        team_order_helper = defaultdict(lambda: defaultdict(list))
        maitland_location = defaultdict(list) 

        for game in weekly_draw.games:
            if game.field.location == 'Maitland Park':
                maitland_location['Home'].append(game)

                club = get_club(game.team1, teams)
                if club != 'Maitland':
                    away_game_clubs[club].append(game)
                else:
                    club = get_club(game.team2, teams)
                    away_game_clubs[club].append(game)

                if 'Maitland' in game.team1:
                    maitland_team_order[game.timeslot.day][game.timeslot.day_slot].append(game.grade)
                    team_order_helper[game.timeslot.day][game.timeslot.day_slot].append(game.grade.name)
                elif 'Maitland' in  game.team2:
                    maitland_team_order[game.timeslot.day][game.timeslot.day_slot].append(game.grade)
                    team_order_helper[game.timeslot.day][game.timeslot.day_slot].append(game.grade.name)

            elif 'Maitland' in game.team1 or 'Maitland' in  game.team2:
                maitland_location['Away'].append(game)


        # Home proportion requirement
        weekly_proportion = 0
        home_var = 0
        away_var = 0
        for location, games in maitland_location.items():
            if location == 'Home':
                home_var = len(games)
            elif location == 'Away':
                away_var = len(games)
        weekly_proportion = home_var / (home_var + away_var) if (home_var + away_var) > 0 else np.nan

        weekly_proportions.append((weekly_draw.week, weekly_proportion))

        # Away game club requirement
        count = []
        for club, games in away_game_clubs.items():
            count.append((club, len(games)))

        away_game_clubs_list.append((weekly_draw.week, count))

        # Grade progression requirement
        team_order = []
        for day in maitland_team_order.keys():
            for i, day_slot in enumerate(sorted(maitland_team_order[day].keys())):
                if i == 0:
                    prev_grade = maitland_team_order[day][day_slot]
                else:
                    if maitland_team_order[day][day_slot] < prev_grade:
                        team_order.append((day_slot, prev_grade[0].name, maitland_team_order[day][day_slot][0].name))
                    prev_grade = maitland_team_order[day][day_slot]
        maitland_team_order_list.append((weekly_draw.week, team_order))


    violations['Weekly Proportions'] = weekly_proportions   
    violations['Away Game Clubs'] = away_game_clubs_list
    violations['Maitland Team Order'] = maitland_team_order_list
    return violations

def check_home_proportion(roster: Roster, data: dict):
    """
    Check that Maitland plays 50% of games at home
    """

    maitland_teams = defaultdict(lambda: defaultdict(list))

    violations = {}
    for weekly_draw in roster.weeks:
        for game in weekly_draw.games:

            if 'Maitland' in game.team1:
                if 'Maitland Park' in game.field.location:
                    maitland_teams[game.team1]['Home'].append(game)
                else:
                    maitland_teams[game.team1]['Away'].append(game)

            elif 'Maitland' in game.team2:
                if 'Maitland Park' in game.field.location:
                    maitland_teams[game.team2]['Home'].append(game)
                else:
                    maitland_teams[game.team2]['Away'].append(game)

            if 'Gosford' in game.team1:
                if 'Central Coast Hockey Park' in game.field.location:
                    maitland_teams[game.team1]['Home'].append(game)
                else:
                    maitland_teams[game.team1]['Away'].append(game)

            elif 'Gosford' in game.team2:
                if 'Central Coast Hockey Park' in game.field.location:
                    maitland_teams[game.team2]['Home'].append(game)
                else:
                    maitland_teams[game.team2]['Away'].append(game)

    for key in maitland_teams.keys():
        home_games = len(maitland_teams[key]['Home'])
        total_games = home_games + len(maitland_teams[key]['Away'])

        violations[key] = home_games / total_games 

    return violations

def check_no_draw_gaps(roster: Roster, data: Dict) -> List[str]:
    """
    Check that there are no gaps in the draw (games should be consecutive for each field)
    """
    violations = {}

    for weekly_draw in roster.weeks:
        weekly_games = defaultdict(lambda: defaultdict(list))
        for game in weekly_draw.games:
            field_name = game.timeslot.field.name
            weekly_games[field_name][game.timeslot.day].append(game)

        for field_name, days in weekly_games.items():
            for day, games  in days.items():
                sorted_games = sorted(games, key=lambda g: g.timeslot.day_slot)
                day_slots = [game.timeslot.day_slot for game in sorted_games]

            expected_day_slots = list(range(1, len(day_slots) + 1))
            if day_slots != expected_day_slots:
                if weekly_draw.week not in violations:
                    violations[weekly_draw.week] = []
                violations[weekly_draw.week].append((field_name, day_slots))

    return violations

def check_roster_for_requested_unavailability(roster, unavailable_games, field_unavailabilities):
    violations = []

    for week in roster.weeks:
        for game in week.games:
            key = (game.team1, game.team2, game.grade.name, game.timeslot.day, game.timeslot.day_slot, game.timeslot.time, game.timeslot.week, game.timeslot.date, game.field.name, game.field.location)
            if key in unavailable_games:
                violations.append(f"Game {game} was scheduled but should have been unavailable.")

            if game.field.location in field_unavailabilities:
                field_info = field_unavailabilities[game.field.location]
                game_date = datetime.strptime(game.timeslot.date, "%Y-%m-%d").date()  
                game_iso_week = game_date.isocalendar()[1]  
                for weekend in field_info.get('weekends', []):
                    if game_iso_week == weekend.isocalendar()[1] :
                        violations.append(f"Game {game} was scheduled on an unavailable weekend.")

                # Check for whole-day unavailability
                if game_date in [d.date() for d in field_info.get('whole_days', [])]:
                    violations.append(f"Game {game} was scheduled on an unavailable field date.")

                # Check for partial-day unavailability
                for part_day in field_info.get('part_days', []):
                    if game_date == part_day.date() and game.timeslot.time == part_day.time():
                        violations.append(f"Game {game} was scheduled during an unavailable time slot.")

    return violations

def check_preference_violations(roster, data):
    """Check if any games in the roster violate the preference_no_play constraints."""
    violations = []
    preference_no_play = data['preference_no_play']

    for club, restrictions in preference_no_play.items():
        for restriction in restrictions:
            club_teams = get_teams_from_club(club, data['teams'])
            specific_team = restriction.get('team_name')

            if specific_team:
                assert specific_team in club_teams, f"Team {specific_team} not found in club {club}."
                for week in roster.weeks:
                    for game in week.games:
                        if specific_team in [game.team1, game.team2]:
                            if all(getattr(game.timeslot, key, None) == value or getattr(game.field, key, None) == value 
                                    for key, value in restriction.items()):
                                violations.append(f"Game {game} violates preference_no_play for club {club}.")
            else:
                for week in roster.weeks:
                    for game in week.games:
                        if game.team1 in club_teams or game.team2 in club_teams:

                            if all(getattr(game.timeslot, key, None) == value or getattr(game.field, key, None) == value 
                                    for key, value in restriction.items()):
                                violations.append(f"Game {game} violates preference_no_play for club {club}.")

        # Check for team conflict violations
        for (team1, team2) in data['team_conflicts']:
            for week in roster.weeks:
                scheduled_games = []
                for game in week.games:
                    if game.team1 in [team1, team2] or game.team2 in [team1, team2]:
                        game_time = (game.timeslot.date, game.timeslot.time, game.field.location)
                        if game_time in scheduled_games:
                            violations.append(f"Teams {team1} and {team2} were scheduled to play at the same time: {game}.")
                        scheduled_games.append(game_time)

    return violations

roster = convert_X_to_roster(X_solution, data)
export_roster_to_excel(roster, data)


# In[ ]:


[X for X in X_solution if X[-3] == 5 and X[-1] == 'Maitland Park']


# In[ ]:


def analyze_roster(roster: Roster, data: dict):
    field_usage = defaultdict(set)  # {date_time: field}
    matchup_matrices = {}
    game_counts_per_week = defaultdict(lambda: defaultdict(set))
    violations = []

    # Iterate through weekly draws
    for weekly_draw in roster.weeks:
        week_num = weekly_draw.week
        for game in weekly_draw.games:
            grade_name = game.grade.name
            team1, team2 = game.team1, game.team2
            timeslot_key = (game.timeslot.date, game.timeslot.time, game.field.name)

            # ✅ 1. Field Conflict Check                                                                                                NoDoubleBookingFieldsConstraint()
            game_identifier = (game.team1, game.team2, game.grade.name)
            if timeslot_key in field_usage:
                violations.append(f"Field Conflict: {game.field} has multiple games at {game.timeslot.date} {game.timeslot.time}.")
            field_usage[timeslot_key].add(game_identifier)

            # ✅ 2. Matchup Matrix
            if grade_name not in matchup_matrices:
                teams = game.grade.teams
                matchup_matrices[grade_name] = pd.DataFrame(0, index=teams, columns=teams)

            matchup_matrices[grade_name].loc[team1, team2] += 1
            matchup_matrices[grade_name].loc[team2, team1] += 1
            game_counts_per_week[grade_name][team1].add(week_num)
            game_counts_per_week[grade_name][team2].add(week_num)

# ✅ 3. Balanced Participation Check                                                                                                        EnsureEqualGamesAndBalanceConstraint()
    for grade, matrix in matchup_matrices.items():
        teams = list(matrix.index)
        games_played = {team: sum(matrix.loc[team]) for team in teams}
        matchups = matrix.to_dict()

        total_weeks = set(range(1, len(roster.weeks) + 1))
        byes = {team: len(total_weeks) - len(game_counts_per_week[grade][team]) for team in teams}

        if len(set([len(game_counts_per_week[grade][team]) for team in teams])) > 1:
            violations.append(f"Different measure, unequal games in {grade}: {games_played} \n {matrix}")
        if len(set(games_played.values())) > 1:
            violations.append(f"Unequal games in {grade}: {games_played} \n {matrix}")
        if len(set(byes.values())) > 1:
            violations.append(f"Unequal byes in {grade}: {byes} \n {matrix}")
        for team, opponents in matchups.items():
            del opponents[team]
            if len(set(opponents.values())) > 1:
                violations.append(f"Unequal matchups in {grade}: {team} has {opponents} \n {matrix}")

    # ✅ 4. PHL Time Restrictions                                                                                                               
    for weekly_draw in roster.weeks:
        phl_dict = defaultdict(list)
        for game in weekly_draw.games:
            if game.grade.name == 'PHL' and game.field.location == 'Newcastle International Hockey Centre':
                phl_dict[(game.timeslot.date, game.timeslot.day_slot)].append(game)
                if (int(game.timeslot.time[:2]) < 11 or int(game.timeslot.time[:2]) >= 16):
                    violations.append(f"PHL game outside allowed hours: {game.timeslot.date} at {game.timeslot.time} on {game.field.location}.")
        for games in phl_dict.values():
            if len(games) > 1:
                violations.append(f"PHL games on same timeslot: {', '.join(game.team1 for game in games)} on {games[0].timeslot.date}.")
        # PHL and Second grade in same club time restriction
    teams = data['teams']
    for weekly_draw in roster.weeks:
        phl_dict = defaultdict(lambda: defaultdict(list))
        for game in weekly_draw.games:
            if game.grade.name in ['PHL', '2nd']:
                phl_dict[(game.timeslot.date, game.timeslot.time)][get_club(game.team1, teams)].append(game)
                phl_dict[(game.timeslot.date, game.timeslot.time)][get_club(game.team2, teams)].append(game)
        for date, clubs in phl_dict.items():
            for club, games in clubs.items():
                if len(games) > 1 and games[0] != games[1] and len(set(game.grade.name for game in games)) > 1:
                    teams_ = {game.team1 for game in games if club in game.team1} | {game.team2 for game in games if club in game.team2}
                    violations.append(f"PHL and 2nd grade timing violation: {', '.join(teams_)} on {games[0].timeslot.date}.")

    # ✅ 5. Club Scheduling Rule for 2nds and PHL
    phl_games_by_week_field = defaultdict(lambda: defaultdict(list))  # {week: {field: [day_slots]}}
    second_grade_games_by_week_field = defaultdict(lambda: defaultdict(list))

    for weekly_draw in roster.weeks:
        for game in weekly_draw.games:
            if game.grade.name == 'PHL':
                phl_games_by_week_field[weekly_draw.week,game.timeslot.day, get_club(game.team1, teams)][game.field.location].append(game.timeslot.time)
                phl_games_by_week_field[weekly_draw.week, game.timeslot.day, get_club(game.team2, teams)][game.field.location].append(game.timeslot.time)

            elif game.grade.name == '2nd':
                second_grade_games_by_week_field[weekly_draw.week, game.timeslot.day, get_club(game.team1, teams)][game.field.location].append(game.timeslot.time)
                second_grade_games_by_week_field[weekly_draw.week, game.timeslot.day, get_club(game.team2, teams)][game.field.location].append(game.timeslot.time)


    for week, fields in phl_games_by_week_field.items():            
        for field, phl_slots in fields.items():
            second_slots = second_grade_games_by_week_field[week]
            for second_field, second_slot in second_slots.items():

                if second_field != field and any((datetime.strptime(slot, '%H:%M') - timedelta(minutes=120)).time().strftime('%H:%M') in second_slots or (datetime.strptime(slot, '%H:%M') + timedelta(minutes=120)).time().strftime('%H:%M') in second_slots for slot in phl_slots):
                    violations.append(f"PHL/2nd scheduling violation in week {week} with 2nds on field {second_field} at time {second_slot} and PHL on {field} at time {phl_slots}.")

    # ✅ 6. Game Count Per Grade
    grade_counts = defaultdict(lambda: defaultdict(int))
    for weekly_draw in roster.weeks:
        for game in weekly_draw.games:
            grade_counts[game.grade.name][game.grade.num_teams] += 1 

    # ✅ 7. Soft Constraint Violations
    if 'penalties' in data:
        for constraint_name, constraint in data['penalties'].items():
            for penalty_var in constraint['unresolved']:
                violations.append(f"{constraint_name} violation: {penalty_var}")
    # 8. Back to Back Byes
    potential_violations = check_back_to_back_byes(roster)
    for week, teams in potential_violations.items():
        violations.append(f"Teams {', '.join(teams)} have consecutive byes with second bye in week {week}.")

    # 9. Check Maitland home/away proportions
    potential_violations = check_home_proportion(roster, data)
    for team, proportion in potential_violations.items():
        violations.append(f"Team {team} has {proportion} of games at home.")

    # 10. Check complex Maitland requirements
    potential_violations = check_maitland_complex_limitations(roster, data)
    for key, value in potential_violations.items():
        violations.append(f"{key}: {value}")

    # 11. Check no draw gaps
    potential_violations = check_no_draw_gaps(roster, data)
    for week, fields in potential_violations.items():
        violations.append(f"Draw gap in week {week} at field {fields[0]}: {fields[0]}")

    # 12. Check for requested unavailability
    potential_violations = check_roster_for_requested_unavailability(roster, data['unavailable_games'], data['field_unavailabilities'])
    for viol in potential_violations:
        violations.append(viol)

    # 13. Check for preference violations
    potential_violations = check_preference_violations(roster, data)
    for viol in potential_violations:
        violations.append(viol)

    # ✅ Print Results
    print("Assessed Conditions:")
    if violations:
        for violation in violations:
            print(violation)
    else:
        print("No violations found.")

    print("\nNumber of Games Per Grade:")
    for grade, num_teams in grade_counts.items():
        print(f"{grade}: {2 * list(num_teams.values())[0]/list(num_teams.keys())[0]}") # Multiply by 2 because there are 2 teams per game count

analyze_roster(roster, data)


# In[ ]:


def train_schedule(model, X, max_time=900):
    """
    Continues solving the scheduling model with the enforced schedule, 
    improving the solution without adding new constraints.

    Args:
        model (cp_model.CpModel): The existing model with enforced constraints.
        X (dict): The decision variables.
        max_time (int): Maximum time in seconds to run the solver.

    Returns:
        dict: Updated solution if feasible.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time  

    print("Re-solving the model to further optimize the schedule...")

    status = solver.Solve(model)

    print(f"Solver status: {solver.StatusName()}")

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        X_solution = {key: solver.Value(var) for key, var in X.items()}
        print(f"Trained schedule with {sum(X_solution.values())} scheduled games.")
        return X_solution
    else:
        print("No feasible schedule found after training.")
        return {}
# X_final_sol = train_schedule(model, X, max_time=900)


# In[ ]:


import pickle
with open(r"draws\V1\X.pkl", "rb") as f:
    loaded_X = pickle.load(f)

with open(r"draws\V1\data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
roster = convert_X_to_roster(loaded_X, data)
analyze_roster(roster, data)



# In[ ]:


import pandas as pd
from typing import Dict
from pydantic import BaseModel
from pathlib import Path

def load_roster_from_excel(filename: str, data: Dict) -> Roster:
    all_teams: Set[str] = {team.name for team in data['teams']} 
    roster_weeks = []


    with pd.ExcelFile(filename) as xls:
        for sheet_name in xls.sheet_names:
            try:
                week = int(sheet_name.replace("Week ", "").strip()) 
                df = pd.read_excel(xls, sheet_name=sheet_name)

                games = []
                teams_played = set()

                for _, row in df.iterrows():
                    if (pd.isna(row["Team 1"])  and pd.isna(row["Team 2"])) or str(row["Team 1"]).strip().lower() == "byes and no games":
                        break 

                    field_class = get_field_by_name(str(row["Field Name"]).strip(), data['fields'])
                    timeslot = Timeslot(
                        day=row["Day"].strip(), time=row["Time"].strip(), week=week, 
                        field=field_class, day_slot=row['Day Slot'], date=row["Date"].strip(), round_no=row['Round']
                    )
                    game = Game(
                        team1=row["Team 1"].strip(), team2=row["Team 2"].strip(), timeslot=timeslot,
                        field=field_class, grade=get_grade_by_name(row["Grade"].strip(), data['grades'])
                    )

                    games.append(game)
                    teams_played.update([row["Team 1"].strip(), row["Team 2"].strip()])
            except Exception as e:
                print(e)
                print(row)

            bye_teams = list(all_teams - teams_played)

            weekly_draw = WeeklyDraw(week=week, games=games, bye_teams=bye_teams, round_no=1)
            roster_weeks.append(weekly_draw)

    return Roster(weeks=roster_weeks)

roster_File = r'draws\V6\schedule Wk1-5.xlsx'

roster = load_roster_from_excel(roster_File, data)
# analyze_roster(roster, data)


# In[ ]:


def export_roster_to_excel_revformat(roster: Roster, filename="schedule_revformat.csv", week_no = None):
    grade_map = {'PHL': 'HCPHL', '2nd': '2ND GRADE', '3rd': '3RD GRADE', '4th': '4TH GRADE', '5th': '5TH GRADE', '6th': '6TH GRADE'}
    field_map = {'EF':'EF', 'WF':'WF', 'SF':'SF', 'Maitland Main Field':'', 'Wyong Main Field':''}

    all_rows = []

    for weekly_draw in roster.weeks:
        if week_no and weekly_draw.week > week_no:
            continue
        for game in weekly_draw.games:
            team1_name = 'Newcastle Hockey Association ' + game.team1.rsplit(" ", 1)[0]  # Remove grade
            team2_name = 'Newcastle Hockey Association ' + game.team2.rsplit(" ", 1)[0]  # Remove grade
            all_rows.append([
                datetime.strptime(game.timeslot.date, '%Y-%m-%d').strftime("%d/%m/%Y"),  # Australian format date
                game.timeslot.time,
                field_map.get(game.field.name, game.field.name),
                game.field.location,
                game.timeslot.round_no, 
                grade_map.get(game.grade.name, game.grade.name),
                team1_name,
                team2_name
            ])

        for team in weekly_draw.bye_teams:
            team_name = 'Newcastle Hockey Association ' + team.rsplit(" ", 1)[0]  
            grade = team.rsplit(" ", 1)[1].strip()  # Extract grade
            all_rows.append([
                datetime.strptime(game.timeslot.date, '%Y-%m-%d').strftime("%d/%m/%Y"),  # Australian format date
                '',
                '',
                '',
                game.timeslot.round_no, 
                grade_map.get(grade, grade),
                team_name,
                'BYE'
            ])

    df = pd.DataFrame(
        all_rows, 
        columns=[ "DATE", "TIME", "FIELD", "VENUE", "ROUND", "GRADE", "TEAM 1", "TEAM 2"]
    )
    df["IS_BYE"] = df["TEAM 2"] == "BYE"
    df.sort_values(
        by=["ROUND", "IS_BYE", "GRADE"],
        inplace=True
    )
    df.drop(columns="IS_BYE", inplace=True)
    df.to_csv(filename, index=False)
    print(f"Schedule successfully exported to {filename}")


export_roster_to_excel_revformat(roster, week_no=6)


# In[ ]:


def parse_full_draw_old_format(filename: str) -> Roster:
    df = pd.read_excel(filename, sheet_name="FULL DRAW", skiprows=19)

    weeks = []
    current_round = None
    games = []
    teams_played = set()
    teams_with_bye = []

    for idx, row in df.iterrows():
        round_val = row.get('ROUND')
        team1 = str(row.get('TEAM 1')).strip() if pd.notna(row.get('TEAM 1')) else ''
        team2 = str(row.get('TEAM 2')).strip() if pd.notna(row.get('TEAM 2')) else ''

        # Skip completely blank or empty rows
        if not round_val and not team1 and not team2:
            continue

        # Detect round change and push current week to list
        if pd.notna(round_val):
            try:
                round_val = int(round_val)
            except ValueError:
                continue  # Skip rows with non-integer ROUND values

            if current_round is None:
                current_round = round_val
            elif round_val != current_round:
                weeks.append(WeeklyDraw(
                    round_number=current_round,
                    games=games,
                    teams_with_bye=teams_with_bye
                ))
                # Reset for the new round
                games = []
                teams_played = set()
                teams_with_bye = []
                current_round = round_val

        # Skip if we don’t have a valid round number
        if current_round is None:
            continue

        # Handle BYE matches
        if 'BYE' in team1.upper():
            if team2 and team2 not in teams_played:
                teams_with_bye.append(team2)
                teams_played.add(team2)
            continue
        elif 'BYE' in team2.upper():
            if team1 and team1 not in teams_played:
                teams_with_bye.append(team1)
                teams_played.add(team1)
            continue

        # If either team name is missing, skip
        if not team1 or not team2:
            continue

        # Skip if already played
        if team1 in teams_played or team2 in teams_played:
            continue

        game = Game(
            team1=team1,
            team2=team2,
            field=str(row.get('FIELD')).strip(),
            grade=str(row.get('GRADE')).strip(),
            time=str(row.get('TIME')).strip(),
            day=str(row.get('DAY')).strip(),
            date=str(row.get('DATE')).strip()
        )
        games.append(game)
        teams_played.update([team1, team2])

    # Append final round if there are remaining games
    if games or teams_with_bye:
        weeks.append(WeeklyDraw(
            round_number=current_round,
            games=games,
            teams_with_bye=teams_with_bye
        ))

    return Roster(weeks=weeks)

new_roster = parse_full_draw_old_format(r'draws\V5\NMHA and HCPHL 2025 DRAW draft - V5.xlsx')
analyze_roster(new_roster, data)


# In[ ]:




