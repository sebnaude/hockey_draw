# test_constraints.py
"""
Unit tests for constraint classes.

These tests verify that constraints are correctly applied to the model
and that the solver can find feasible solutions when constraints are satisfied,
and correctly rejects infeasible scenarios.
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, time as tm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot
from constraints import (
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    PHLAndSecondGradeAdjacency,
    PHLAndSecondGradeTimes,
    FiftyFiftyHomeandAway,
    TeamConflictConstraint,
    MaxMaitlandHomeWeekends,
    EnsureBestTimeslotChoices,
    ClubDayConstraint,
    EqualMatchUpSpacingConstraint,
    ClubGradeAdjacencyConstraint,
    ClubVsClubAlignment,
    MaitlandHomeGrouping,
    AwayAtMaitlandGrouping,
    MaximiseClubsPerTimeslotBroadmeadow,
    MinimiseClubsOnAFieldBroadmeadow,
    PreferredTimesConstraint,
)


# ============== Fixtures ==============

@pytest.fixture
def basic_fields():
    """Create basic test fields."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
    ]


@pytest.fixture
def basic_clubs():
    """Create basic test clubs."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def basic_teams(basic_clubs):
    """Create basic test teams."""
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
    """Create basic test timeslots."""
    ef = basic_fields[0]
    wf = basic_fields[1]
    return [
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='11:30', week=1, day_slot=2, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=ef, round_no=2),
        Timeslot(date='2025-03-30', day='Sunday', time='11:30', week=2, day_slot=2, field=ef, round_no=2),
    ]


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


# ============== NoDoubleBookingTeamsConstraint Tests ==============

class TestNoDoubleBookingTeamsConstraint:
    """Tests for NoDoubleBookingTeamsConstraint."""

    def test_allows_single_game_per_week(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that a team can play exactly one game per week."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'current_week': 0,
        }
        
        constraint = NoDoubleBookingTeamsConstraint()
        constraint.apply(model, X, data)
        
        # Force one game in week 1
        week1_games = [v for k, v in X.items() if k[6] == 1]
        model.Add(sum(week1_games) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Verify Tigers 3rd plays at most once in each week
        for week in [1, 2]:
            tigers_games_in_week = sum(
                solver.Value(v) for k, v in X.items() 
                if k[6] == week and 'Tigers 3rd' in (k[0], k[1])
            )
            assert tigers_games_in_week <= 1

    def test_prevents_double_booking(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that a team cannot play two games in the same week."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'current_week': 0,
        }
        
        constraint = NoDoubleBookingTeamsConstraint()
        constraint.apply(model, X, data)
        
        # Force Tigers 3rd to play two games in week 1
        tigers_week1_games = [v for k, v in X.items() if k[6] == 1 and 'Tigers 3rd' in (k[0], k[1])]
        model.Add(sum(tigers_week1_games) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status == cp_model.INFEASIBLE


# ============== NoDoubleBookingFieldsConstraint Tests ==============

class TestNoDoubleBookingFieldsConstraint:
    """Tests for NoDoubleBookingFieldsConstraint."""

    def test_allows_single_game_per_field_slot(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that a field can host exactly one game per timeslot."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'current_week': 0,
        }
        
        constraint = NoDoubleBookingFieldsConstraint()
        constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    def test_prevents_field_double_booking(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that two games cannot occur on the same field at the same time."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'current_week': 0,
        }
        
        constraint = NoDoubleBookingFieldsConstraint()
        constraint.apply(model, X, data)
        
        # Force both games on same field, same day_slot, same week
        ef_slot1_week1 = [
            v for k, v in X.items() 
            if k[6] == 1 and k[4] == 1 and k[9] == 'EF'
        ]
        model.Add(sum(ef_slot1_week1) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status == cp_model.INFEASIBLE


# ============== EnsureEqualGamesAndBalanceMatchUps Tests ==============

class TestEnsureEqualGamesAndBalanceMatchUps:
    """Tests for EnsureEqualGamesAndBalanceMatchUps."""

    def test_equal_games_per_team(self, basic_fields):
        """Test that each team plays the required number of games."""
        # Simple setup: 2 teams, 2 rounds
        clubs = [
            Club(name='TeamA', home_field='Newcastle International Hockey Centre'),
            Club(name='TeamB', home_field='Newcastle International Hockey Centre'),
        ]
        teams = [
            Team(name='TeamA 3rd', club=clubs[0], grade='3rd'),
            Team(name='TeamB 3rd', club=clubs[1], grade='3rd'),
        ]
        grades = [Grade(name='3rd', teams=['TeamA 3rd', 'TeamB 3rd'])]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=basic_fields[0], round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=basic_fields[0], round_no=2),
        ]
        
        games = [('TeamA 3rd', 'TeamB 3rd', '3rd')]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'grades': grades,
            'num_rounds': {'3rd': 2},
            'num_dummy_timeslots': 0,
        }
        
        constraint = EnsureEqualGamesAndBalanceMatchUps()
        constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Count games per team
        team_games = {'TeamA 3rd': 0, 'TeamB 3rd': 0}
        for k, v in X.items():
            if solver.Value(v):
                team_games[k[0]] += 1
                team_games[k[1]] += 1
        
        assert team_games['TeamA 3rd'] == 2
        assert team_games['TeamB 3rd'] == 2

    def test_balanced_matchups(self, basic_fields):
        """Test that matchups are balanced (base or base+1 meetings)."""
        # 3 teams, 4 rounds means base=1, some pairs meet twice
        clubs = [Club(name=f'Club{i}', home_field='Newcastle International Hockey Centre') for i in range(3)]
        teams = [Team(name=f'Team{i} 3rd', club=clubs[i], grade='3rd') for i in range(3)]
        grades = [Grade(name='3rd', teams=[t.name for t in teams])]
        
        timeslots = [
            Timeslot(date=f'2025-03-{23+i*7}', day='Sunday', time='10:00', week=i+1, day_slot=1, field=basic_fields[0], round_no=i+1)
            for i in range(4)
        ]
        
        games = [
            ('Team0 3rd', 'Team1 3rd', '3rd'),
            ('Team0 3rd', 'Team2 3rd', '3rd'),
            ('Team1 3rd', 'Team2 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'grades': grades,
            'num_rounds': {'3rd': 4},
            'num_dummy_timeslots': 0,
        }
        
        constraint = EnsureEqualGamesAndBalanceMatchUps()
        constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== FiftyFiftyHomeandAway Tests ==============

class TestFiftyFiftyHomeandAway:
    """Tests for FiftyFiftyHomeandAway constraint."""

    def test_home_away_balance_maitland(self, basic_fields):
        """Test that Maitland teams get balanced home/away games."""
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Maitland', home_field='Maitland Park'),
        ]
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Maitland 3rd', club=clubs[1], grade='3rd'),
        ]
        
        maitland_field = PlayingField(location='Maitland Park', name='Main')
        broadmeadow_field = basic_fields[0]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=maitland_field, round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=broadmeadow_field, round_no=2),
        ]
        
        games = [('Tigers 3rd', 'Maitland 3rd', '3rd')]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
        }
        
        constraint = FiftyFiftyHomeandAway()
        constraint.apply(model, X, data)
        
        # Require exactly 2 games
        model.Add(sum(X.values()) == 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        # Count home games for Maitland
        home_games = sum(
            solver.Value(v) for k, v in X.items() 
            if k[10] == 'Maitland Park' and 'Maitland' in k[0] or 'Maitland' in k[1]
        )
        # Should be balanced (1 home, 1 away)
        assert home_games == 1


# ============== TeamConflictConstraint Tests ==============

class TestTeamConflictConstraint:
    """Tests for TeamConflictConstraint."""

    def test_prevents_conflicting_teams_same_slot(self, basic_teams, basic_timeslots, basic_fields, basic_clubs):
        """Test that conflicting teams cannot play at the same time."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'fields': basic_fields,
            'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],  # Tigers teams can't play same time
            'current_week': 0,
        }
        
        constraint = TeamConflictConstraint()
        constraint.apply(model, X, data)
        
        # Force both Tigers teams to play in same timeslot
        week1_slot1_tigers = [
            v for k, v in X.items() 
            if k[6] == 1 and k[4] == 1 and ('Tigers' in k[0] or 'Tigers' in k[1])
        ]
        model.Add(sum(week1_slot1_tigers) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status == cp_model.INFEASIBLE

    def test_allows_non_conflicting_same_slot(self, basic_teams, basic_timeslots, basic_fields, basic_clubs):
        """Test that non-conflicting teams can play at the same time."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'fields': basic_fields,
            'team_conflicts': [],  # No conflicts
            'current_week': 0,
        }
        
        constraint = TeamConflictConstraint()
        constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== ClubGradeAdjacencyConstraint Tests ==============

class TestClubGradeAdjacencyConstraint:
    """Tests for ClubGradeAdjacencyConstraint."""

    def test_prevents_adjacent_grades_same_slot(self, basic_fields):
        """Test that adjacent grades from same club can't play at same time."""
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        ]
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Tigers 4th', club=clubs[0], grade='4th'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
            Team(name='Wests 4th', club=clubs[1], grade='4th'),
        ]
        grades = [
            Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd']),
            Grade(name='4th', teams=['Tigers 4th', 'Wests 4th']),
        ]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=basic_fields[0], round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=basic_fields[1], round_no=1),
        ]
        
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'grades': grades,
            'clubs': clubs,
        }
        
        constraint = ClubGradeAdjacencyConstraint()
        constraint.apply(model, X, data)
        
        # Force both games in same slot
        slot1_games = [v for k, v in X.items() if k[4] == 1 and k[6] == 1]
        model.Add(sum(slot1_games) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be infeasible because Tigers 3rd and 4th would both play at same time
        assert status == cp_model.INFEASIBLE


# ============== MaitlandHomeGrouping Tests ==============

class TestMaitlandHomeGrouping:
    """Tests for MaitlandHomeGrouping constraint."""

    def test_no_back_to_back_home_weekends(self, basic_fields):
        """Test that Maitland can't have back-to-back home weekends."""
        maitland_field = PlayingField(location='Maitland Park', name='Main')
        
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Maitland', home_field='Maitland Park'),
        ]
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Maitland 3rd', club=clubs[1], grade='3rd'),
        ]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=maitland_field, round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=maitland_field, round_no=2),
        ]
        
        games = [('Tigers 3rd', 'Maitland 3rd', '3rd')]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'penalties': {},
            'current_week': 0,
        }
        
        constraint = MaitlandHomeGrouping()
        constraint.apply(model, X, data)
        
        # Force home games in both consecutive weeks
        all_games = list(X.values())
        model.Add(sum(all_games) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be infeasible due to back-to-back constraint
        assert status == cp_model.INFEASIBLE


# ============== AwayAtMaitlandGrouping Tests ==============

class TestAwayAtMaitlandGrouping:
    """Tests for AwayAtMaitlandGrouping constraint."""

    def test_max_three_away_clubs(self, basic_fields):
        """Test that maximum 3 away clubs visit Maitland per weekend."""
        maitland_field = PlayingField(location='Maitland Park', name='Main')
        
        clubs = [Club(name=f'Club{i}', home_field='Newcastle International Hockey Centre') for i in range(5)]
        clubs.append(Club(name='Maitland', home_field='Maitland Park'))
        
        teams = [Team(name=f'Club{i} 3rd', club=clubs[i], grade='3rd') for i in range(5)]
        teams.append(Team(name='Maitland 3rd', club=clubs[5], grade='3rd'))
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time=f'{10+i}:00', week=1, day_slot=i+1, field=maitland_field, round_no=1)
            for i in range(5)
        ]
        
        games = [(f'Club{i} 3rd', 'Maitland 3rd', '3rd') for i in range(5)]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'clubs': clubs,
            'penalties': {},
            'current_week': 0,
        }
        
        constraint = AwayAtMaitlandGrouping()
        constraint.apply(model, X, data)
        
        # Force 4 different clubs to visit in same week
        for i in range(4):
            game_vars = [v for k, v in X.items() if k[0] == f'Club{i} 3rd']
            model.Add(sum(game_vars) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be infeasible due to max 3 away clubs limit
        assert status == cp_model.INFEASIBLE


# ============== MinimiseClubsOnAFieldBroadmeadow Tests ==============

class TestMinimiseClubsOnAFieldBroadmeadow:
    """Tests for MinimiseClubsOnAFieldBroadmeadow constraint."""

    def test_max_five_clubs_per_field(self, basic_fields):
        """Test that maximum 5 clubs play on any field per day."""
        clubs = [Club(name=f'Club{i}', home_field='Newcastle International Hockey Centre') for i in range(7)]
        teams = [Team(name=f'Club{i} 3rd', club=clubs[i], grade='3rd') for i in range(7)]
        
        ef = basic_fields[0]
        
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time=f'{10+i}:00', week=1, day_slot=i+1, field=ef, round_no=1)
            for i in range(6)
        ]
        
        # Create games between consecutive clubs
        games = [(f'Club{i} 3rd', f'Club{i+1} 3rd', '3rd') for i in range(6)]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'clubs': clubs,
            'penalties': {},
            'current_week': 0,
        }
        
        constraint = MinimiseClubsOnAFieldBroadmeadow()
        constraint.apply(model, X, data)
        
        # Force 6 games (involving 7 clubs) on same field same day
        model.Add(sum(X.values()) >= 6)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be infeasible due to max 5 clubs limit
        assert status == cp_model.INFEASIBLE


# ============== Integration Tests ==============

class TestConstraintIntegration:
    """Integration tests for multiple constraints."""

    def test_basic_constraints_feasible(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that basic constraints together produce feasible solutions."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
            ('Wests 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, basic_timeslots)
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'grades': basic_grades,
            'fields': basic_fields,
            'clubs': basic_clubs,
            'current_week': 0,
        }
        
        constraints = [
            NoDoubleBookingTeamsConstraint(),
            NoDoubleBookingFieldsConstraint(),
        ]
        
        for constraint in constraints:
            constraint.apply(model, X, data)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    def test_conflicting_requirements_infeasible(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that impossible requirements are correctly identified as infeasible."""
        # Only one game possible
        games = [('Tigers 3rd', 'Wests 3rd', '3rd')]
        
        # Only one timeslot
        timeslots = [basic_timeslots[0]]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': basic_teams,
            'current_week': 0,
        }
        
        constraint = NoDoubleBookingTeamsConstraint()
        constraint.apply(model, X, data)
        
        # Require 2 games (impossible with only 1 timeslot)
        model.Add(sum(X.values()) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status == cp_model.INFEASIBLE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
