from abc import ABC, abstractmethod
from ortools.sat.python import cp_model
from collections import defaultdict
import math
from pydantic import BaseModel, Field,  field_validator
from typing import List, Dict, Tuple
from datetime import time as tm
from datetime import timedelta, datetime
from ortools.sat.python import cp_model
import pandas as pd
import os
from collections import defaultdict
import re
import json

class PlayingField(BaseModel):
    name: str = Field(..., description="Name of the field")
    location: str = Field(..., description="Location of the field")

class Grade(BaseModel):
    name: str = Field(..., description="Grade name")
    teams: List[str] = Field(..., description="List of team names in this grade")
    num_teams: int = Field(0, description="Number of teams in this grade")


    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "num_teams", len(self.teams))

class ClubDay(BaseModel):
    date: str = Field(..., description="Date of the game (e.g., '2025-03-04')")
    day: str = Field(..., description="Day of the game (e.g., 'Saturday', 'Sunday')")
    week: int = Field(..., description="The week number for the season")
    field: PlayingField = Field(..., description="Field where the game is played")

class Timeslot(BaseModel):
    date: str = Field(..., description="Date of the game (e.g., '2025-03-04')")
    day: str = Field(..., description="Day of the game (e.g., 'Saturday', 'Sunday')")
    time: str = Field(..., description="Time of the game (e.g., 14:00 for 2 PM)")
    week: int = Field(..., description="The week number for the season")
    day_slot: int = Field(..., description="The game slot for the day (e.g., 1 for first game of the day)")
    field: PlayingField = Field(..., description="Field where the game is played")

class Club(BaseModel):
    name: str = Field(..., description="Club name")
    home_field: str = Field(..., description="Home field")
    preferred_times: List[Timeslot] = Field(default=[], description="Preferred play times for the club")
    club_day: List[ClubDay] = Field(default=[], description="The club day") 

class Team(BaseModel):
    name: str = Field(..., description="Name of the team")
    club: Club = Field(..., description="Club the team belongs to")
    grade: str = Field(..., description="Grade the team belongs to")
    preferred_times: List[Timeslot] = Field(default=[], description="Times the team prefers to play")
    unavailable_times: List[Timeslot] = Field(default=[], description="Times the team cannot play")
    constraints: List[str] = Field(default=[], description="Special scheduling constraints for the team")

class Game(BaseModel):
    team1: str = Field(..., description="First team playing")
    team2: str = Field(..., description="Second team playing")
    timeslot: Timeslot = Field(..., description="Scheduled time for the game")
    field: PlayingField = Field(..., description="Field where the game is played")
    grade: Grade = Field(..., description="Grade the game belongs to")

class WeeklyDraw(BaseModel):
    week: int = Field(..., description="Week number in the season")
    games: List[Game] = Field(..., description="Games scheduled for this week")
    
class Roster(BaseModel):
    weeks: List[WeeklyDraw] = Field(..., description="Complete schedule for the season")


def generate_timeslots(start_date, end_date, days, times, excluded_weekends, fields):
    """Generate weekly timeslots between two dates."""
    timeslots = []
    current_date = start_date
    week_number = 1
    day_slot = 1
    c_day = 'a day'
    while current_date <= end_date:
        if current_date.strftime('%A') in days:

            if c_day != current_date.strftime('%A'):
                day_slot = 1
                c_day = current_date.strftime('%A')
                
            for t in times:
                for f in fields:
                    timeslots.append({'date': current_date.strftime('%Y-%m-%d'), 'day': current_date.strftime('%A'), 'time': t.strftime("%H:%M:%S"), 'week': week_number, 'day_slot': day_slot, 'field': f})
                day_slot += 1

        if current_date.strftime('%A') == 'Monday':
            week_number += 1
            day_slot = 1
        current_date += timedelta(days=1)
    return timeslots

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

def get_club(team_name, teams):
    for team in teams:
        if team_name == team.name:
            return team.club.name
    return None  


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
        
        weekly_games = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)
                if key in X:
                    weekly_games[(t.week, t1)].append(X[key])
                    weekly_games[(t.week, t2)].append(X[key])

        for (week, team), game_vars in weekly_games.items():
            model.Add(sum(game_vars) <= 1)

class NoDoubleBookingFieldsConstraint(Constraint):
    """Ensure no field is scheduled for more than one game per time slot."""
    
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        
        field_usage = defaultdict(list)

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)
                if key in X:
                    field_usage[(t.day, t.day_slot, t.week, t.field.name)].append(X[key])

        for slot, game_vars in field_usage.items():
            model.Add(sum(game_vars) <= 1)

class EnsureEqualGamesAndBalanceMatchUps(Constraint):
    """Ensure each team plays the most it can."""
    
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        num_rounds = data['num_rounds']

        team_games = defaultdict(lambda: defaultdict(list))
        grade_games = defaultdict(lambda: defaultdict(list))

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)
                if key in X:
                    team_games[grade][t1].append(X[key])
                    team_games[grade][t2].append(X[key])
                    grade_games[grade][tuple(sorted((t1, t2)))].append(X[key])

        for grade, teams in team_games.items():
            num_teams = len([team for team in data['teams'] if team.grade == grade])
            total_games = (num_rounds // (num_teams - 1) ) * (num_teams - 1)

            if len(teams.keys()) % 2 != 0:
                total_games_max = num_rounds - num_rounds // num_teams
                while total_games > total_games_max:
                    total_games = total_games - (num_teams - 1)


            for team, game_vars in teams.items():
                model.Add(sum(game_vars) == total_games)  

        for grade, teams in grade_games.items():

            num_teams = len([team for team in data['teams'] if team.grade == grade])
            total_games = (num_rounds // (num_teams - 1) ) * (num_teams - 1)

            if len(teams.keys()) % 2 != 0:
                total_games_max = num_rounds - num_rounds // num_teams
                while total_games > total_games_max:
                    total_games = total_games - (num_teams - 1)

            team_v_team_games = total_games // (num_teams - 1)

            for team, game_vars in teams.items():
                model.Add(sum(game_vars) == team_v_team_games) 

class PHLAndSecondGradeAdjacency(Constraint):
    '''Ensure that PHL and 2nds do not play in adjacent day_slots at different locations.''' 
    def apply(self, model, X, data):
        teams = data['teams']
        interested_grades = {'PHL', '2nd'}

        # Store penalties in data
        if 'penalties' not in data:
            data['penalties'] = []

        for t in data['timeslots']:
            for (t1, t2, grade) in data['games']:
                if grade not in interested_grades:
                    continue
                
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)
                if key not in X:
                    continue
                
                club = get_club(t1, teams)
                game_var = X[key]
                
                # Check adjacent time slots
                for adj_t in data['timeslots']:
                    if adj_t.day == t.day and abs(adj_t.day_slot - t.day_slot) == 1:
                        adj_key = (t1, t2, grade, adj_t.day, adj_t.day_slot, adj_t.time, adj_t.week, adj_t.date, adj_t.field.name, adj_t.field.location)
                        if adj_key in X:
                            adj_game_var = X[adj_key]
                            
                            # If fields are different, create a soft penalty
                            if t.field.location != adj_t.field.location:
                                penalty_var = model.NewBoolVar(f'penalty_{key}_{adj_key}')
                                model.Add(penalty_var >= game_var + adj_game_var - 1)
                                
                                # Store penalty in data
                                data['penalties'].append(penalty_var)


                        
class PHLAndSecondGradeTimes(Constraint):        
    ''' PHL games must be played between 11 and 4 and should not be played at the same time.'''      
    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        num_rounds = data['num_rounds']

        team_games = defaultdict(lambda: defaultdict(list))
        club_games = defaultdict(lambda: defaultdict(list))

        for t in timeslots:
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)

                # Stop PHL from playing out of bounds time
                if grade == 'PHL' and key in X:
                    if t.time < '11:00:00' or t.time > '15:00:00':
                        model.Add(X[key] == 0)
                    else:
                        # Stop PHL games across clubs from being played at the same time
                        team_games[(t.week, t.day)][(t.day_slot, t.field.location)].append(X[key])

                # # Stop PHL and 2nd grade from being played at the same time, within the same club
                # if grade in ['PHL', '2nd'] and key in X:
                #     club = get_club(t1, data['teams'])
                #     club_games[(t.week, t.day)][(t.day_slot, club)].append(X[key])

                #     club = get_club(t2, data['teams'])
                #     club_games[(t.week, t.day)][(t.day_slot, club)].append(X[key])

        for date, day_slots in team_games.items():
            for day_slot, game_vars in day_slots.items():
                if day_slot[1] == 'Broadmedow':
                    model.Add(sum(game_vars) <= 1)    
        
        for date, day_slots in club_games.items():
            for day_slot, game_vars in day_slots.items():
                model.Add(sum(game_vars) <= 1)

# Soft or User set constaints
class UnavailableTimesConstraint(Constraint):
    """Ensure teams do not play at times they cannot play."""
    
    def apply(self, model, X, data):
        teams = data['teams']
        
        for team in teams:
            for t in team.unavailable_times:
                for (t1, t2, grade) in data['games']:
                    if t1 == team.name or t2 == team.name:
                        if (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location) in X:
                            model.Add(X[(t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)] == 0)

class PreferredTimesConstraint(Constraint):
    """Ensure teams play at preferred times."""
    
    def apply(self, model, X, data):
        teams = data['teams']
        
        for team in teams:
            for t in team.preferred_times:
                for (t1, t2, grade) in data['games']:
                    if t1 == team.name or t2 == team.name:
                        if (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location) in X:
                            model.Add(X[(t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)] == 1)


class ClubPreferredTimesConstraint(Constraint):
    """Ensure clubs play at preferred times."""
    
    def apply(self, model, X, data):
        clubs = data['clubs']

        for club in clubs:
            for t in club.preferred_times:
                for (t1, t2, grade) in data['games']:
                    if t1 in [team.name for team in club.teams] or t2 in [team.name for team in club.teams]:
                        if (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location) in X:
                            model.Add(X[(t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)] == 1)
                            
class ClubDayConstraint(Constraint):
    """Ensure club days are played on the same field and back to back."""
    
    def apply(self, model, X, data):
        clubs = data['clubs']

        for club in clubs:
            for club_day in club.club_day:
                    for (t1, t2, grade) in data['games']:
                        if t1 in [team.name for team in club.teams] or t2 in [team.name for team in club.teams]:
                            for t in data['timeslots']:
                                if t.week == club_day.week:
                                    if (t1, t2,  grade, t.day, t.time, t.week, t.date, t.field.name, t.field.location) in X:
                                        if t.day == club_day.day: 
                                            model.Add(X[(t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)] == 1)
                                        else:
                                            model.Add(X[(t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location)] == 0)




class FiftyFiftyHomeFieldSplit(Constraint):
    pass


def create_schedule(data, constraints, save_path="draws/schedule.json"):
    model = cp_model.CpModel()

    # Generate game pairs
    games = {(t1.name, t2.name, t1.grade): (t1, t2, t1.grade) 
             for t1 in data['teams'] for t2 in data['teams'] 
             if t1.name < t2.name and t1.grade == t2.grade}
    print(f"Generated {len(games)} games.")  
    data['games'] = games

    # Generate variables
    X = {
        (t1.name, t2.name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.field.name, t.field.location): 
        model.NewBoolVar(f'X_{t1}_{t2}_{t.day}_{t.time}_{t.week}_{t.field.name}_{t.field.location}')
        for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items() 
        for t in data['timeslots']
        if t.field.location in {t1.club.home_field, t2.club.home_field}
    }

    print(f"Generated {len(X)} decision variables.")  
    
    # Apply constraints dynamically
    for constraint in constraints:
        constraint.apply(model, X, data)

    # Define weight for penalty
    penalty_weight = 10  
    penalties = data.get('penalties', [])

    # Maximize scheduled games while penalizing adjacent games at different locations
    model.Maximize(
        sum(X[key] for key in X) - penalty_weight * sum(penalties)
    )

    print(f"Total constraints in model: {len(model.Proto().constraints)}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2400  
    status = solver.Solve(model)

    print(f"Solver status: {solver.StatusName()}") 

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # Extract the solution (only scheduled games)
        schedule = {str(key): solver.Value(var) for key, var in X.items() if solver.Value(var) == 1}

        # Save to JSON
        with open(save_path, "w") as f:
            json.dump(schedule, f, indent=4)

        print(f"Schedule saved to {save_path}")
        return schedule
    else:
        print("No feasible schedule found.")
        return {}

FIELDS = [PlayingField(location='Broadmedow', name='South Field'),
          PlayingField(location='Broadmedow', name='East Field'),
          PlayingField(location='Broadmedow', name='West Field'),
          PlayingField(location='Maitland', name='Maitland Main Field',),
          PlayingField(location='Wyong', name='Wyong Main Field'),]

TEAMS_DATA = r'C:\Users\c3205\Documents\Code\Jupyter Notebooks\draw\data\2025\teams'
CLUBS = []
TEAMS = []
for file in os.listdir(TEAMS_DATA):
    df = pd.read_csv(os.path.join(TEAMS_DATA, file))
    club = Club(name=df['Club'].iloc[0], home_field="Maitland" if df['Club'].iloc[0] == "Maitland" else 
                      'Wyong' if df['Club'].iloc[0] == 'Gosford' else "Broadmedow")
    CLUBS.append(club)
    teams = [Team(name=f"{row['Team Name']} {row['Grade']}", club=club, grade=row['Grade'], home_field=club.home_field) for index, row in df.iterrows()]
    TEAMS.extend(teams)

teams_by_grade = defaultdict(list)
for team in TEAMS:
    teams_by_grade[team.grade].append(team.name)
GRADES = [Grade(name=f'{grade}', teams=teams) for grade, teams in sorted(teams_by_grade.items())]

for grade in GRADES:
    print(grade.name, grade.teams)

# Generate timeslots
num_rounds = 24
times = [tm(8, 30), tm(10, 0), tm(11,30), tm(13, 00), tm(14, 30), tm(16, 00), tm(17, 30), tm(19, 00)]
days = ['Saturday', 'Sunday']
start = datetime(2025, 3, 31)
end = datetime(2025, 9, 16)
excluded_weekends = []
timeslots = generate_timeslots(start, end, days, times, excluded_weekends, FIELDS)
TIMESLOTS = [Timeslot(date=t['date'], day=t['day'], time=t['time'], week=t['week'], day_slot=t['day_slot'], field=t['field']) for t in timeslots]
# Instantiate data
data = {'teams': TEAMS, 'grades': GRADES, 'fields': FIELDS, 'timeslots': TIMESLOTS, 'num_rounds': num_rounds, 'clubs': CLUBS}
constraints = [NoDoubleBookingTeamsConstraint(), NoDoubleBookingFieldsConstraint(),  EnsureEqualGamesAndBalanceMatchUps(), PHLAndSecondGradeTimes(), PHLAndSecondGradeAdjacency() ]
X = create_schedule(data=data, constraints=constraints)
