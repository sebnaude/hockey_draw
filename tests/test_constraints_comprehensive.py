# test_constraints_comprehensive.py
"""
Comprehensive tests for all constraint classes.

This module provides tests for constraints that were not covered in the original
test_constraints.py file. It uses the mini season fixtures from conftest.py to
run constraints in a controlled, small environment.
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot
from constraints import (
    PHLAndSecondGradeAdjacency,
    PHLAndSecondGradeTimes,
    MaxMaitlandHomeWeekends,
    EnsureBestTimeslotChoices,
    ClubDayConstraint,
    EqualMatchUpSpacingConstraint,
    ClubVsClubAlignment,
    MaximiseClubsPerTimeslotBroadmeadow,
    PreferredTimesConstraint,
)


# Helper functions (also defined in conftest.py but importing for standalone use)
def create_model_and_vars(games, timeslots, current_week=0):
    """Helper to create model and decision variables."""
    model = cp_model.CpModel()
    X = {}
    
    for (t1, t2, grade) in games:
        for t in timeslots:
            if not t.day:
                continue
            key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}')
    
    return model, X


def generate_round_robin_games(teams):
    """Generate round-robin matchups for all teams grouped by grade."""
    games = []
    teams_by_grade = defaultdict(list)
    
    for team in teams:
        teams_by_grade[team.grade].append(team.name)
    
    for grade, team_names in teams_by_grade.items():
        for t1, t2 in combinations(team_names, 2):
            games.append((t1, t2, grade))
    
    return games


# ============== PHL and 2nd Grade Fixtures ==============

@pytest.fixture
def phl_2nd_clubs():
    """Clubs with PHL and 2nd grade teams."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def phl_2nd_teams(phl_2nd_clubs):
    """Teams in PHL and 2nd grades for testing adjacency."""
    tigers, wests, maitland = phl_2nd_clubs
    return [
        # PHL
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Wests PHL', club=wests, grade='PHL'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
        # 2nd
        Team(name='Tigers 2nd', club=tigers, grade='2nd'),
        Team(name='Wests 2nd', club=wests, grade='2nd'),
        Team(name='Maitland 2nd', club=maitland, grade='2nd'),
    ]


@pytest.fixture
def phl_2nd_fields():
    """Fields for PHL/2nd testing."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
    ]


@pytest.fixture
def phl_2nd_timeslots(phl_2nd_fields):
    """Timeslots for PHL/2nd testing - multiple slots at different locations."""
    ef, wf, maitland = phl_2nd_fields
    return [
        # Week 1 - Multiple slots
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='11:30', week=1, day_slot=2, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='11:30', week=1, day_slot=2, field=wf, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=maitland, round_no=1),
        # Week 2
        Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=ef, round_no=2),
        Timeslot(date='2025-03-30', day='Sunday', time='11:30', week=2, day_slot=2, field=ef, round_no=2),
    ]


# ============== PHLAndSecondGradeAdjacency Tests ==============

class TestPHLAndSecondGradeAdjacency:
    """Tests for PHLAndSecondGradeAdjacency constraint."""

    def test_allows_non_adjacent_timeslots(self, phl_2nd_teams, phl_2nd_timeslots, phl_2nd_clubs, phl_2nd_fields):
        """Test that PHL and 2nd can play if not in adjacent timeslots at different locations."""
        games = [
            ('Tigers PHL', 'Wests PHL', 'PHL'),
            ('Tigers 2nd', 'Wests 2nd', '2nd'),
        ]
        
        model, X = create_model_and_vars(games, phl_2nd_timeslots)
        
        data = {
            'games': games,
            'timeslots': phl_2nd_timeslots,
            'teams': phl_2nd_teams,
            'clubs': phl_2nd_clubs,
            'fields': phl_2nd_fields,
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = PHLAndSecondGradeAdjacency()
        constraint.apply(model, X, data)
        
        # Force games to happen
        all_games = list(X.values())
        model.Add(sum(all_games) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be feasible - games can be scheduled in non-adjacent slots
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== PHLAndSecondGradeTimes Tests ==============

class TestPHLAndSecondGradeTimes:
    """Tests for PHLAndSecondGradeTimes constraint."""

    def test_max_phl_games_at_broadmeadow(self, phl_2nd_teams, phl_2nd_fields, phl_2nd_clubs):
        """Test that max 3 PHL Friday night games at Broadmeadow is enforced."""
        # Create Friday night timeslots
        ef = phl_2nd_fields[0]
        friday_slots = [
            Timeslot(date='2025-03-21', day='Friday', time='19:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-21', day='Friday', time='20:30', week=1, day_slot=2, field=ef, round_no=1),
            Timeslot(date='2025-03-28', day='Friday', time='19:00', week=2, day_slot=1, field=ef, round_no=2),
            Timeslot(date='2025-03-28', day='Friday', time='20:30', week=2, day_slot=2, field=ef, round_no=2),
            Timeslot(date='2025-04-04', day='Friday', time='19:00', week=3, day_slot=1, field=ef, round_no=3),
        ]
        
        games = [
            ('Tigers PHL', 'Wests PHL', 'PHL'),
            ('Tigers PHL', 'Maitland PHL', 'PHL'),
            ('Wests PHL', 'Maitland PHL', 'PHL'),
        ]
        
        model, X = create_model_and_vars(games, friday_slots)
        
        data = {
            'games': games,
            'timeslots': friday_slots,
            'teams': phl_2nd_teams,
            'clubs': phl_2nd_clubs,
            'fields': phl_2nd_fields,
            'current_week': 0, 'locked_weeks': set(),
            'phl_preferences': {'preferred_dates': []},
        }
        
        constraint = PHLAndSecondGradeTimes()
        constraint.apply(model, X, data)
        
        # Force all 3 games
        all_games = list(X.values())
        model.Add(sum(all_games) >= 3)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Count Friday games - should be at most 3
        friday_game_count = sum(1 for k, v in X.items() if solver.Value(v) == 1)
        assert friday_game_count <= 3

    def test_gosford_friday_games_exactly_8(self):
        """Test that exactly 8 PHL games are scheduled at Gosford on Friday (AGM decision)."""
        # Create Gosford field
        gosford_field = PlayingField(location='Central Coast Hockey Park', name='Wyong Main')
        
        # Create clubs and teams
        gosford_club = Club(name='Gosford', home_field='Central Coast Hockey Park')
        tigers_club = Club(name='Tigers', home_field='Newcastle International Hockey Centre')
        wests_club = Club(name='Wests', home_field='Newcastle International Hockey Centre')
        maitland_club = Club(name='Maitland', home_field='Maitland Park')
        
        clubs = [gosford_club, tigers_club, wests_club, maitland_club]
        
        teams = [
            Team(name='Gosford PHL', club=gosford_club, grade='PHL'),
            Team(name='Tigers PHL', club=tigers_club, grade='PHL'),
            Team(name='Wests PHL', club=wests_club, grade='PHL'),
            Team(name='Maitland PHL', club=maitland_club, grade='PHL'),
        ]
        
        # Generate games (Gosford vs all others = 3 home games per round * 2 rounds = 6)
        # But we need exactly 8, so include full round robin
        games = [
            ('Gosford PHL', 'Tigers PHL', 'PHL'),
            ('Gosford PHL', 'Wests PHL', 'PHL'),
            ('Gosford PHL', 'Maitland PHL', 'PHL'),
            ('Tigers PHL', 'Wests PHL', 'PHL'),
            ('Tigers PHL', 'Maitland PHL', 'PHL'),
            ('Wests PHL', 'Maitland PHL', 'PHL'),
        ]
        
        # Create 10 Friday night slots at Gosford (need more than 8 to test constraint)
        friday_slots = []
        for week in range(1, 11):
            date = datetime(2025, 3, 21) + timedelta(weeks=week-1)
            friday_slots.append(Timeslot(
                date=date.strftime('%Y-%m-%d'),
                day='Friday',
                time='20:00',
                week=week,
                day_slot=1,
                field=gosford_field,
                round_no=week
            ))
        
        model, X = create_model_and_vars(games, friday_slots)
        
        data = {
            'games': games,
            'timeslots': friday_slots,
            'teams': teams,
            'clubs': clubs,
            'fields': [gosford_field],
            'current_week': 0, 'locked_weeks': set(),
            'phl_preferences': {'preferred_dates': []},
        }
        
        constraint = PHLAndSecondGradeTimes()
        constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be feasible
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE], f"Solver status: {solver.status_name(status)}"
        
        # Count Friday games at Gosford - should be exactly 8
        gosford_friday_count = sum(
            1 for k, v in X.items() 
            if solver.Value(v) == 1 and k[10] == 'Central Coast Hockey Park'
        )
        assert gosford_friday_count == 8, f"Expected 8 Gosford Friday games, got {gosford_friday_count}"


# ============== MaxMaitlandHomeWeekends Tests ==============

class TestMaxMaitlandHomeWeekends:
    """Tests for MaxMaitlandHomeWeekends constraint."""

    def test_limits_maitland_home_weekends(self):
        """Test that Maitland home weekends are limited appropriately."""
        maitland_club = Club(name='Maitland', home_field='Maitland Park')
        tigers_club = Club(name='Tigers', home_field='Newcastle International Hockey Centre')
        
        teams = [
            Team(name='Maitland 3rd', club=maitland_club, grade='3rd'),
            Team(name='Tigers 3rd', club=tigers_club, grade='3rd'),
        ]
        
        grades = [Grade(name='3rd', teams=['Maitland 3rd', 'Tigers 3rd'])]
        for g in grades:
            g.set_games(4)
        
        fields = [
            PlayingField(location='Maitland Park', name='Maitland Main Field'),
            PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        ]
        
        # Create 4 weeks of timeslots at Maitland
        timeslots = []
        for week in range(1, 5):
            date = datetime(2025, 3, 23) + timedelta(weeks=week-1)
            timeslots.append(Timeslot(
                date=date.strftime('%Y-%m-%d'),
                day='Sunday',
                time='10:00',
                week=week,
                day_slot=1,
                field=fields[0],  # Maitland
                round_no=week
            ))
            timeslots.append(Timeslot(
                date=date.strftime('%Y-%m-%d'),
                day='Sunday',
                time='10:00',
                week=week,
                day_slot=1,
                field=fields[1],  # Broadmeadow
                round_no=week
            ))
        
        games = [('Maitland 3rd', 'Tigers 3rd', '3rd')]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'grades': grades,
            'clubs': [maitland_club, tigers_club],
            'fields': fields,
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = MaxMaitlandHomeWeekends()
        constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== EqualMatchUpSpacingConstraint Tests ==============

class TestEqualMatchUpSpacingConstraint:
    """Tests for EqualMatchUpSpacingConstraint constraint."""

    def test_constraint_applies_without_error(self):
        """Test that the constraint can be applied without error."""
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Wests', home_field='Newcastle International Hockey Centre'),
            Club(name='Norths', home_field='Newcastle International Hockey Centre'),
        ]
        
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
            Team(name='Norths 3rd', club=clubs[2], grade='3rd'),
        ]
        
        grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd', 'Norths 3rd'])]
        
        fields = [
            PlayingField(location='Newcastle International Hockey Centre', name='EF'),
            PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        ]
        
        # Create 6 weeks of timeslots with multiple slots per week
        timeslots = []
        for week in range(1, 7):
            date = datetime(2025, 3, 23) + timedelta(weeks=week-1)
            for field in fields:
                for slot, time in enumerate(['10:00', '11:30'], 1):
                    timeslots.append(Timeslot(
                        date=date.strftime('%Y-%m-%d'),
                        day='Sunday',
                        time=time,
                        week=week,
                        day_slot=slot,
                        field=field,
                        round_no=week
                    ))
        
        games = generate_round_robin_games(teams)
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'grades': grades,
            'clubs': clubs,
            'fields': fields,
            'current_week': 0, 'locked_weeks': set(),
            'num_rounds': {'3rd': 6, 'max': 6},
        }
        
        # The constraint should apply without raising an exception
        constraint = EqualMatchUpSpacingConstraint()
        try:
            constraint.apply(model, X, data)
            applied_successfully = True
        except Exception as e:
            applied_successfully = False
            print(f"Constraint application failed: {e}")
        
        assert applied_successfully, "EqualMatchUpSpacingConstraint should apply without error"


# ============== MaximiseClubsPerTimeslotBroadmeadow Tests ==============

class TestMaximiseClubsPerTimeslotBroadmeadow:
    """Tests for MaximiseClubsPerTimeslotBroadmeadow constraint."""

    def test_encourages_club_diversity_in_timeslot(self):
        """Test that different clubs are encouraged to play in the same timeslot."""
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Wests', home_field='Newcastle International Hockey Centre'),
            Club(name='Norths', home_field='Newcastle International Hockey Centre'),
            Club(name='Souths', home_field='Newcastle International Hockey Centre'),
        ]
        
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
            Team(name='Norths 3rd', club=clubs[2], grade='3rd'),
            Team(name='Souths 3rd', club=clubs[3], grade='3rd'),
        ]
        
        fields = [
            PlayingField(location='Newcastle International Hockey Centre', name='EF'),
            PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        ]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=fields[0], round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=fields[1], round_no=1),
        ]
        
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Norths 3rd', 'Souths 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'clubs': clubs,
            'fields': fields,
            'current_week': 0, 'locked_weeks': set(),
            'penalties': {},
        }
        
        constraint = MaximiseClubsPerTimeslotBroadmeadow()
        constraint.apply(model, X, data)
        
        # Force both games to happen
        all_games = list(X.values())
        model.Add(sum(all_games) == 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== PreferredTimesConstraint Tests ==============

class TestPreferredTimesConstraint:
    """Tests for PreferredTimesConstraint constraint."""

    def test_penalizes_games_at_non_preferred_times(self):
        """Test that games at non-preferred times are penalized."""
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        ]
        
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
        ]
        
        fields = [PlayingField(location='Newcastle International Hockey Centre', name='EF')]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=fields[0], round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=fields[0], round_no=2),
        ]
        
        games = [('Tigers 3rd', 'Wests 3rd', '3rd')]
        
        model, X = create_model_and_vars(games, timeslots)
        
        # Tigers wants to avoid playing on 2025-03-23
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'clubs': clubs,
            'fields': fields,
            'current_week': 0, 'locked_weeks': set(),
            'preference_no_play': {
                'Tigers': [{'date': '2025-03-23'}],
            },
            'penalties': {},
        }
        
        constraint = PreferredTimesConstraint()
        constraint.apply(model, X, data)
        
        # Force game to happen
        all_games = list(X.values())
        model.Add(sum(all_games) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Check that penalties were created
        assert 'PreferredTimesConstraint' in data['penalties']


# ============== Integration Tests with Mini Season ==============

class TestMiniSeasonIntegration:
    """Integration tests using mini season fixtures."""

    def test_all_core_constraints_with_mini_season(self, mini_season_data):
        """Test that all core constraints work together on mini season."""
        from constraints import (
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
            EnsureEqualGamesAndBalanceMatchUps,
        )
        
        games = mini_season_data['games']
        timeslots = mini_season_data['timeslots']
        
        model, X = create_model_and_vars(games, timeslots)
        
        # Apply core constraints
        NoDoubleBookingTeamsConstraint().apply(model, X, mini_season_data)
        NoDoubleBookingFieldsConstraint().apply(model, X, mini_season_data)
        
        # Add dummy variables for EnsureEqualGamesAndBalanceMatchUps
        num_dummy = 5
        for (t1, t2, grade) in games:
            for i in range(num_dummy):
                dummy_key = (t1, t2, grade, i)
                X[dummy_key] = model.NewBoolVar(f'X_dummy_{t1}_{t2}_{i}')
        
        mini_season_data['num_dummy_timeslots'] = num_dummy
        EnsureEqualGamesAndBalanceMatchUps().apply(model, X, mini_season_data)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    def test_constraint_counts_games_correctly(self, mini_season_data):
        """Test that constraint system correctly counts scheduled games."""
        from constraints import NoDoubleBookingTeamsConstraint
        
        games = mini_season_data['games']
        timeslots = mini_season_data['timeslots']
        
        model, X = create_model_and_vars(games, timeslots)
        
        NoDoubleBookingTeamsConstraint().apply(model, X, mini_season_data)
        
        # Force at least one game per week
        for week in range(1, 5):
            week_games = [v for k, v in X.items() if len(k) >= 7 and k[6] == week]
            if week_games:
                model.Add(sum(week_games) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Count games per week - should respect no double booking
        for week in range(1, 5):
            teams_this_week = defaultdict(int)
            for k, v in X.items():
                if len(k) >= 7 and k[6] == week and solver.Value(v) == 1:
                    teams_this_week[k[0]] += 1
                    teams_this_week[k[1]] += 1
            
            # Each team should play at most once per week
            for team, count in teams_this_week.items():
                assert count <= 1, f"Team {team} plays {count} times in week {week}"
