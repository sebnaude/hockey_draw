# conftest.py
"""
Shared pytest fixtures and test data generators for constraint testing.

This module provides:
1. Basic fixtures for simple unit tests (3 clubs, 5 teams, 2 grades)
2. Mini season fixtures for integration testing (4 clubs, 8 teams, 2 grades, 4 weeks)
3. Full test season fixtures for comprehensive testing (6 clubs, 12+ teams, 3+ grades, 8 weeks)
4. Helper functions for creating models and decision variables
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import List, Dict, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot


# ============== Field Fixtures ==============

@pytest.fixture
def broadmeadow_fields():
    """Create Broadmeadow (Newcastle International Hockey Centre) fields."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
    ]


@pytest.fixture
def maitland_field():
    """Create Maitland Park field."""
    return PlayingField(location='Maitland Park', name='Maitland Main Field')


@pytest.fixture
def all_fields(broadmeadow_fields, maitland_field):
    """All available fields."""
    return broadmeadow_fields + [maitland_field]


# ============== Club Fixtures ==============

@pytest.fixture
def broadmeadow_clubs():
    """Clubs that play home games at Broadmeadow."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
        Club(name='University', home_field='Newcastle International Hockey Centre'),
    ]


@pytest.fixture
def maitland_club():
    """Maitland club that plays home games at Maitland Park."""
    return Club(name='Maitland', home_field='Maitland Park')


@pytest.fixture
def all_clubs(broadmeadow_clubs, maitland_club):
    """All clubs in the competition."""
    return broadmeadow_clubs + [maitland_club]


# ============== Basic Fixtures (for simple unit tests) ==============

@pytest.fixture
def basic_fields():
    """Create basic test fields (3 fields)."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
    ]


@pytest.fixture
def basic_clubs():
    """Create basic test clubs (3 clubs)."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def basic_teams(basic_clubs):
    """Create basic test teams (5 teams across 2 grades)."""
    tigers, wests, maitland = basic_clubs
    return [
        Team(name='Tigers 3rd', club=tigers, grade='3rd'),
        Team(name='Wests 3rd', club=wests, grade='3rd'),
        Team(name='Maitland 3rd', club=maitland, grade='3rd'),
        Team(name='Tigers 4th', club=tigers, grade='4th'),
        Team(name='Wests 4th', club=wests, grade='4th'),
    ]


@pytest.fixture
def basic_grades():
    """Create basic test grades."""
    return [
        Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd', 'Maitland 3rd']),
        Grade(name='4th', teams=['Tigers 4th', 'Wests 4th']),
    ]


@pytest.fixture
def basic_timeslots(basic_fields):
    """Create basic test timeslots (5 slots across 2 weeks)."""
    ef = basic_fields[0]
    wf = basic_fields[1]
    return [
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='11:30', week=1, day_slot=2, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=ef, round_no=2),
        Timeslot(date='2025-03-30', day='Sunday', time='11:30', week=2, day_slot=2, field=ef, round_no=2),
    ]


# ============== Mini Season Fixtures (for integration tests) ==============

@pytest.fixture
def mini_season_clubs():
    """4 clubs for mini season testing."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def mini_season_teams(mini_season_clubs):
    """8 teams (4 clubs x 2 grades) for mini season testing."""
    tigers, wests, norths, maitland = mini_season_clubs
    return [
        # 3rd Grade
        Team(name='Tigers 3rd', club=tigers, grade='3rd'),
        Team(name='Wests 3rd', club=wests, grade='3rd'),
        Team(name='Norths 3rd', club=norths, grade='3rd'),
        Team(name='Maitland 3rd', club=maitland, grade='3rd'),
        # 4th Grade
        Team(name='Tigers 4th', club=tigers, grade='4th'),
        Team(name='Wests 4th', club=wests, grade='4th'),
        Team(name='Norths 4th', club=norths, grade='4th'),
        Team(name='Maitland 4th', club=maitland, grade='4th'),
    ]


@pytest.fixture
def mini_season_grades():
    """2 grades for mini season testing."""
    return [
        Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd', 'Norths 3rd', 'Maitland 3rd']),
        Grade(name='4th', teams=['Tigers 4th', 'Wests 4th', 'Norths 4th', 'Maitland 4th']),
    ]


@pytest.fixture
def mini_season_fields():
    """3 fields for mini season testing."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
    ]


@pytest.fixture
def mini_season_timeslots(mini_season_fields):
    """Create 4 weeks of timeslots for mini season (16 total slots)."""
    ef, wf, maitland = mini_season_fields
    timeslots = []
    
    base_date = datetime(2025, 3, 23)  # Start on a Sunday
    
    for week in range(1, 5):  # 4 weeks
        week_date = base_date + timedelta(weeks=week-1)
        date_str = week_date.strftime('%Y-%m-%d')
        
        # 4 slots per week at Broadmeadow (2 per field)
        for field in [ef, wf]:
            for slot, time in enumerate(['10:00', '11:30'], 1):
                timeslots.append(Timeslot(
                    date=date_str,
                    day='Sunday',
                    time=time,
                    week=week,
                    day_slot=slot,
                    field=field,
                    round_no=week
                ))
    
    return timeslots


@pytest.fixture
def mini_season_games(mini_season_teams):
    """Generate all possible games for mini season (round-robin per grade)."""
    games = []
    
    # Group teams by grade
    teams_by_grade = defaultdict(list)
    for team in mini_season_teams:
        teams_by_grade[team.grade].append(team.name)
    
    # Generate round-robin matchups for each grade
    for grade, team_names in teams_by_grade.items():
        for t1, t2 in combinations(team_names, 2):
            games.append((t1, t2, grade))
    
    return games


@pytest.fixture
def mini_season_data(mini_season_games, mini_season_timeslots, mini_season_teams, 
                     mini_season_grades, mini_season_clubs, mini_season_fields):
    """Complete mini season data dictionary for constraint testing."""
    return {
        'games': mini_season_games,
        'timeslots': mini_season_timeslots,
        'teams': mini_season_teams,
        'grades': mini_season_grades,
        'clubs': mini_season_clubs,
        'fields': mini_season_fields,
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'3rd': 4, '4th': 4},
        'num_dummy_timeslots': 0,
    }


# ============== Full Season Fixtures (for comprehensive tests) ==============

@pytest.fixture
def full_season_clubs():
    """6 clubs for full season testing."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
        Club(name='University', home_field='Newcastle International Hockey Centre'),
        Club(name='Souths', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def full_season_teams(full_season_clubs):
    """18 teams (6 clubs x 3 grades) for full season testing."""
    tigers, wests, norths, uni, souths, maitland = full_season_clubs
    teams = []
    
    for club in [tigers, wests, norths, uni, souths, maitland]:
        for grade in ['PHL', '2nd', '3rd']:
            teams.append(Team(name=f'{club.name} {grade}', club=club, grade=grade))
    
    return teams


@pytest.fixture
def full_season_grades(full_season_teams):
    """3 grades for full season testing."""
    teams_by_grade = defaultdict(list)
    for team in full_season_teams:
        teams_by_grade[team.grade].append(team.name)
    
    return [Grade(name=grade, teams=team_names) for grade, team_names in teams_by_grade.items()]


@pytest.fixture
def full_season_fields():
    """3 fields for full season testing."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
    ]


@pytest.fixture
def full_season_timeslots(full_season_fields):
    """Create 8 weeks of timeslots for full season."""
    ef, wf, maitland = full_season_fields
    timeslots = []
    
    base_date = datetime(2025, 3, 23)
    
    for week in range(1, 9):  # 8 weeks
        week_date = base_date + timedelta(weeks=week-1)
        date_str = week_date.strftime('%Y-%m-%d')
        
        # Broadmeadow slots
        for field in [ef, wf]:
            for slot, time in enumerate(['10:00', '11:30', '13:00', '14:30'], 1):
                timeslots.append(Timeslot(
                    date=date_str,
                    day='Sunday',
                    time=time,
                    week=week,
                    day_slot=slot,
                    field=field,
                    round_no=week
                ))
        
        # Maitland slots (fewer)
        for slot, time in enumerate(['10:00', '11:30'], 1):
            timeslots.append(Timeslot(
                date=date_str,
                day='Sunday',
                time=time,
                week=week,
                day_slot=slot,
                field=maitland,
                round_no=week
            ))
    
    return timeslots


# ============== Helper Functions ==============

def create_model_and_vars(games: List[Tuple], timeslots: List[Timeslot], 
                          current_week: int = 0) -> Tuple[cp_model.CpModel, Dict]:
    """
    Helper to create an OR-Tools model and decision variables.
    
    Args:
        games: List of (team1, team2, grade) tuples
        timeslots: List of Timeslot objects
        current_week: Week number to start from (0 = include all)
    
    Returns:
        Tuple of (CpModel, X dict of decision variables)
    """
    model = cp_model.CpModel()
    X = {}
    
    for (t1, t2, grade) in games:
        for t in timeslots:
            if not t.day:
                continue
            key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}')
    
    return model, X


def create_model_with_dummies(games: List[Tuple], timeslots: List[Timeslot], 
                               num_dummy: int = 5) -> Tuple[cp_model.CpModel, Dict]:
    """
    Create model with dummy timeslots for handling byes.
    
    Args:
        games: List of (team1, team2, grade) tuples
        timeslots: List of Timeslot objects
        num_dummy: Number of dummy slots to create per game
    
    Returns:
        Tuple of (CpModel, X dict of decision variables)
    """
    model, X = create_model_and_vars(games, timeslots)
    
    # Add dummy variables
    for (t1, t2, grade) in games:
        for i in range(num_dummy):
            dummy_key = (t1, t2, grade, i)
            X[dummy_key] = model.NewBoolVar(f'X_dummy_{t1}_{t2}_{i}')
    
    return model, X


def generate_round_robin_games(teams: List[Team]) -> List[Tuple[str, str, str]]:
    """
    Generate round-robin matchups for all teams grouped by grade.
    
    Args:
        teams: List of Team objects
    
    Returns:
        List of (team1, team2, grade) tuples
    """
    games = []
    teams_by_grade = defaultdict(list)
    
    for team in teams:
        teams_by_grade[team.grade].append(team.name)
    
    for grade, team_names in teams_by_grade.items():
        for t1, t2 in combinations(team_names, 2):
            games.append((t1, t2, grade))
    
    return games


def solve_with_timeout(model: cp_model.CpModel, timeout_seconds: float = 5.0) -> Tuple[int, cp_model.CpSolver]:
    """
    Solve a model with a timeout.
    
    Args:
        model: The CP model to solve
        timeout_seconds: Maximum time to spend solving
    
    Returns:
        Tuple of (status, solver)
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    status = solver.Solve(model)
    return status, solver


def count_scheduled_games(X: Dict, solver: cp_model.CpSolver) -> int:
    """Count how many games are scheduled in a solution."""
    return sum(1 for k, v in X.items() if len(k) >= 11 and solver.Value(v) == 1)


def get_team_games_in_week(X: Dict, solver: cp_model.CpSolver, team: str, week: int) -> int:
    """Count how many games a team has scheduled in a specific week."""
    count = 0
    for k, v in X.items():
        if len(k) >= 11 and k[6] == week and team in (k[0], k[1]) and solver.Value(v) == 1:
            count += 1
    return count
