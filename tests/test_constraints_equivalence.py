# test_constraints_equivalence.py
"""
Parametrized tests that run the same test cases against both original and AI constraints.

This ensures both implementations handle the same edge cases correctly.
Documented edge cases per constraint are noted in comments.
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot

# Import both original and AI versions
from constraints import (
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    FiftyFiftyHomeandAway,
    TeamConflictConstraint,
    ClubGradeAdjacencyConstraint,
    MaitlandHomeGrouping,
    AwayAtMaitlandGrouping,
    MinimiseClubsOnAFieldBroadmeadow,
)

from constraints.ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    FiftyFiftyHomeandAwayAI,
    TeamConflictConstraintAI,
    ClubGradeAdjacencyConstraintAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
)


# ============== Fixtures ==============

@pytest.fixture
def sample_fields():
    """Create test fields at multiple locations."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main'),
        PlayingField(location='Central Coast Hockey Park', name='Gosford Main'),
    ]


@pytest.fixture
def sample_clubs():
    """Create test clubs with different home fields."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
        Club(name='Gosford', home_field='Central Coast Hockey Park'),
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
    ]


@pytest.fixture
def sample_teams(sample_clubs):
    """Create test teams across multiple grades."""
    tigers, wests, maitland, gosford, norths = sample_clubs
    return [
        # PHL teams
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Wests PHL', club=wests, grade='PHL'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
        # 2nd grade teams
        Team(name='Tigers 2nd', club=tigers, grade='2nd'),
        Team(name='Wests 2nd', club=wests, grade='2nd'),
        Team(name='Maitland 2nd', club=maitland, grade='2nd'),
        # 3rd grade teams
        Team(name='Tigers 3rd', club=tigers, grade='3rd'),
        Team(name='Wests 3rd', club=wests, grade='3rd'),
        Team(name='Maitland 3rd', club=maitland, grade='3rd'),
        Team(name='Gosford 3rd', club=gosford, grade='3rd'),
        Team(name='Norths 3rd', club=norths, grade='3rd'),
        # 4th grade teams
        Team(name='Tigers 4th', club=tigers, grade='4th'),
        Team(name='Wests 4th', club=wests, grade='4th'),
    ]


@pytest.fixture
def sample_grades():
    """Create test grades."""
    return [
        Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL', 'Maitland PHL']),
        Grade(name='2nd', teams=['Tigers 2nd', 'Wests 2nd', 'Maitland 2nd']),
        Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd', 'Maitland 3rd', 'Gosford 3rd', 'Norths 3rd']),
        Grade(name='4th', teams=['Tigers 4th', 'Wests 4th']),
    ]


@pytest.fixture
def sample_timeslots(sample_fields):
    """Create comprehensive test timeslots spanning multiple weeks."""
    ef, wf, maitland, gosford = sample_fields
    timeslots = []
    
    # Week 1 - multiple slots at different fields
    for slot in range(1, 6):
        timeslots.append(Timeslot(
            date='2025-03-23', day='Sunday', time=f'{9+slot}:00', 
            week=1, day_slot=slot, field=ef, round_no=1
        ))
        timeslots.append(Timeslot(
            date='2025-03-23', day='Sunday', time=f'{9+slot}:00', 
            week=1, day_slot=slot, field=wf, round_no=1
        ))
    
    # Week 1 - Maitland slots
    for slot in range(1, 3):
        timeslots.append(Timeslot(
            date='2025-03-23', day='Sunday', time=f'{9+slot}:00', 
            week=1, day_slot=slot, field=maitland, round_no=1
        ))
    
    # Week 2
    for slot in range(1, 6):
        timeslots.append(Timeslot(
            date='2025-03-30', day='Sunday', time=f'{9+slot}:00', 
            week=2, day_slot=slot, field=ef, round_no=2
        ))
        timeslots.append(Timeslot(
            date='2025-03-30', day='Sunday', time=f'{9+slot}:00', 
            week=2, day_slot=slot, field=wf, round_no=2
        ))
    
    # Week 2 - Maitland slots
    for slot in range(1, 3):
        timeslots.append(Timeslot(
            date='2025-03-30', day='Sunday', time=f'{9+slot}:00', 
            week=2, day_slot=slot, field=maitland, round_no=2
        ))
    
    return timeslots


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


def solve_and_get_status(model):
    """Solve model and return status."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    return solver.Solve(model)


# ============== NoDoubleBookingTeams Tests ==============
# EDGE CASES:
# 1. Dummy timeslots (no day) should be excluded
# 2. current_week filtering - past weeks should be ignored
# 3. Team appears in both t1 and t2 positions

class TestNoDoubleBookingTeamsEquivalence:
    """Tests that both implementations prevent double booking equally."""

    @pytest.mark.parametrize("constraint_class", [
        NoDoubleBookingTeamsConstraint,
        NoDoubleBookingTeamsConstraintAI,
    ])
    def test_allows_single_game_per_week(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades):
        """Test that a team can play exactly one game per week."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force exactly one game in week 1
        week1_games = [v for k, v in X.items() if k[6] == 1]
        model.Add(sum(week1_games) == 1)
        
        status = solve_and_get_status(model)
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    @pytest.mark.parametrize("constraint_class", [
        NoDoubleBookingTeamsConstraint,
        NoDoubleBookingTeamsConstraintAI,
    ])
    def test_prevents_double_booking(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades):
        """Test that a team cannot play two games in the same week."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force Tigers 3rd to play two games in week 1
        tigers_week1_games = [v for k, v in X.items() if k[6] == 1 and 'Tigers 3rd' in (k[0], k[1])]
        model.Add(sum(tigers_week1_games) >= 2)
        
        status = solve_and_get_status(model)
        assert status == cp_model.INFEASIBLE

    @pytest.mark.parametrize("constraint_class", [
        NoDoubleBookingTeamsConstraint,
        NoDoubleBookingTeamsConstraintAI,
    ])
    def test_current_week_filtering(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades):
        """Test that locked weeks are not constrained."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        # Set locked_weeks={1}, so week 1 should not be constrained
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'current_week': 1,
            'locked_weeks': {1},
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force Tigers 3rd to play two games in week 1 (should be allowed - it's past)
        tigers_week1_games = [v for k, v in X.items() if k[6] == 1 and 'Tigers 3rd' in (k[0], k[1])]
        if tigers_week1_games:
            model.Add(sum(tigers_week1_games) >= 2)
            
            status = solve_and_get_status(model)
            # Should be feasible since week 1 is not constrained
            assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== NoDoubleBookingFields Tests ==============
# EDGE CASES:
# 1. Same timeslot but different fields should be allowed
# 2. Dummy timeslots should be excluded
# 3. Same field, different day_slot should be allowed

class TestNoDoubleBookingFieldsEquivalence:
    """Tests that both implementations prevent field double booking equally."""

    @pytest.mark.parametrize("constraint_class", [
        NoDoubleBookingFieldsConstraint,
        NoDoubleBookingFieldsConstraintAI,
    ])
    def test_allows_same_time_different_field(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades):
        """Test that two games can occur at the same time on different fields."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force games on EF and WF at same time
        ef_slot1_week1 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1 and k[9] == 'EF']
        wf_slot1_week1 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1 and k[9] == 'WF']
        
        if ef_slot1_week1 and wf_slot1_week1:
            model.Add(sum(ef_slot1_week1) >= 1)
            model.Add(sum(wf_slot1_week1) >= 1)
            
            status = solve_and_get_status(model)
            assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    @pytest.mark.parametrize("constraint_class", [
        NoDoubleBookingFieldsConstraint,
        NoDoubleBookingFieldsConstraintAI,
    ])
    def test_prevents_field_double_booking(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades):
        """Test that two games cannot occur on the same field at the same time."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force both games on same field, same day_slot, same week
        ef_slot1_week1 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1 and k[9] == 'EF']
        model.Add(sum(ef_slot1_week1) >= 2)
        
        status = solve_and_get_status(model)
        assert status == cp_model.INFEASIBLE


# ============== TeamConflictConstraint Tests ==============
# EDGE CASES:
# 1. Groups by (week, day_slot) - same time regardless of field
# 2. current_week filtering
# 3. Both teams in conflict must be involved in games

class TestTeamConflictEquivalence:
    """Tests for team conflict constraint equivalence."""

    @pytest.mark.parametrize("constraint_class", [
        TeamConflictConstraint,
        TeamConflictConstraintAI,
    ])
    def test_prevents_conflicting_teams_same_slot(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades, sample_fields):
        """Test that conflicting teams cannot play at the same timeslot."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Maitland 3rd', '3rd'),  # Tigers 4th involved
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        # Tigers 3rd and Tigers 4th cannot play at same time
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force both games at week 1, slot 1 (different fields allowed, but same time)
        slot1_week1_game1 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1 and k[0] == 'Tigers 3rd']
        slot1_week1_game2 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1 and k[0] == 'Tigers 4th']
        
        if slot1_week1_game1 and slot1_week1_game2:
            model.Add(sum(slot1_week1_game1) >= 1)
            model.Add(sum(slot1_week1_game2) >= 1)
            
            status = solve_and_get_status(model)
            assert status == cp_model.INFEASIBLE

    @pytest.mark.parametrize("constraint_class", [
        TeamConflictConstraint,
        TeamConflictConstraintAI,
    ])
    def test_allows_non_conflicting_same_slot(self, constraint_class, sample_teams, sample_timeslots, sample_clubs, sample_grades, sample_fields):
        """Test that non-conflicting teams can play at the same timeslot."""
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Maitland 3rd', 'Norths 3rd', '3rd'),
        ]
        
        model, X = create_model_and_vars(games, sample_timeslots)
        
        # Tigers 3rd and Tigers 4th conflict, but Maitland 3rd is not involved
        data = {
            'games': games,
            'timeslots': sample_timeslots,
            'teams': sample_teams,
            'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force both games at week 1, slot 1
        slot1_week1 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1]
        model.Add(sum(slot1_week1) >= 2)
        
        status = solve_and_get_status(model)
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== FiftyFiftyHomeandAway Tests ==============
# EDGE CASES:
# 1. Both Maitland and Gosford handling
# 2. Intra-club games (Maitland vs Maitland) should be skipped
# 3. Teams with no games should not cause errors

class TestFiftyFiftyHomeAwayEquivalence:
    """Tests for home/away balance constraint equivalence."""

    @pytest.mark.parametrize("constraint_class", [
        FiftyFiftyHomeandAway,
        FiftyFiftyHomeandAwayAI,
    ])
    def test_home_away_balance_maitland(self, constraint_class, sample_fields):
        """Test that Maitland games are balanced between home and away."""
        maitland_field = sample_fields[2]  # Maitland Park
        ef = sample_fields[0]  # Broadmeadow
        
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Maitland', home_field='Maitland Park'),
        ]
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Maitland 3rd', club=clubs[1], grade='3rd'),
        ]
        
        # Create timeslots - 2 at home (Maitland), 2 away (EF)
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=maitland_field, round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=maitland_field, round_no=2),
            Timeslot(date='2025-04-06', day='Sunday', time='10:00', week=3, day_slot=1, field=ef, round_no=3),
            Timeslot(date='2025-04-13', day='Sunday', time='10:00', week=4, day_slot=1, field=ef, round_no=4),
        ]
        
        games = [('Tigers 3rd', 'Maitland 3rd', '3rd')]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force all 4 games to be played
        model.Add(sum(X.values()) == 4)
        
        status = solve_and_get_status(model)
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


# ============== ClubGradeAdjacencyConstraint Tests ==============
# EDGE CASES:
# 1. Adjacent grades from same club cannot play at same time
# 2. Non-adjacent grades are allowed
# 3. Same club in same grade (multiple teams) should be handled
# 4. Slot_id is (week, day_slot) - ignores field

class TestClubGradeAdjacencyEquivalence:
    """Tests for club grade adjacency constraint equivalence."""

    @pytest.mark.parametrize("constraint_class", [
        ClubGradeAdjacencyConstraint,
        ClubGradeAdjacencyConstraintAI,
    ])
    def test_prevents_adjacent_grades_same_slot(self, constraint_class, sample_fields):
        """Test that adjacent grades from same club are penalized when playing at same time.
        
        ClubGradeAdjacencyConstraint is now SOFT - it allows overlaps but penalizes them.
        """
        ef = sample_fields[0]
        wf = sample_fields[1]
        
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
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='11:30', week=1, day_slot=2, field=ef, round_no=1),
        ]
        
        # Tigers 3rd vs Wests 3rd (3rd grade)
        # Tigers 4th vs Wests 4th (4th grade) - adjacent to 3rd
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 4th', 'Wests 4th', '4th'),
        ]
        
        model, X = create_model_and_vars(games, timeslots)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'clubs': clubs,
            'grades': grades,
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force both games in slot 1 (same time, different fields allowed structurally)
        slot1_week1 = [v for k, v in X.items() if k[6] == 1 and k[4] == 1]
        model.Add(sum(slot1_week1) >= 2)
        
        status = solve_and_get_status(model)
        # Constraint is now SOFT - should be OPTIMAL with penalties applied
        assert status == cp_model.OPTIMAL
        # Verify penalty was tracked
        assert 'ClubGradeAdjacencyConstraint' in data['penalties']
        assert len(data['penalties']['ClubGradeAdjacencyConstraint']['penalties']) > 0


# ============== MaitlandHomeGrouping Tests ==============
# EDGE CASES:
# 1. No back-to-back home weekends (hard constraint)
# 2. Penalty for mixed home/away within a week (soft constraint)
# 3. current_week filtering

class TestMaitlandHomeGroupingEquivalence:
    """Tests for Maitland home grouping constraint equivalence."""

    @pytest.mark.parametrize("constraint_class", [
        MaitlandHomeGrouping,
        MaitlandHomeGroupingAI,
    ])
    def test_no_back_to_back_home_weekends(self, constraint_class, sample_fields):
        """Test that Maitland can't have back-to-back home weekends."""
        maitland_field = sample_fields[2]
        
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
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force home games in both consecutive weeks
        all_games = list(X.values())
        model.Add(sum(all_games) >= 2)
        
        status = solve_and_get_status(model)
        assert status == cp_model.INFEASIBLE


# ============== AwayAtMaitlandGrouping Tests ==============
# EDGE CASES:
# 1. Hard limit of 3 away clubs per weekend
# 2. Penalty for more than 1 away club (soft)
# 3. Only tracks AWAY clubs visiting Maitland

class TestAwayAtMaitlandGroupingEquivalence:
    """Tests for away at Maitland grouping constraint equivalence."""

    @pytest.mark.parametrize("constraint_class", [
        AwayAtMaitlandGrouping,
        AwayAtMaitlandGroupingAI,
    ])
    def test_max_three_away_clubs(self, constraint_class, sample_fields):
        """Test that maximum 3 away clubs visit Maitland per weekend."""
        maitland_field = sample_fields[2]
        
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
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        # Force 4 different clubs to visit in same week
        for i in range(4):
            game_vars = [v for k, v in X.items() if k[0] == f'Club{i} 3rd']
            model.Add(sum(game_vars) >= 1)
        
        status = solve_and_get_status(model)
        assert status == cp_model.INFEASIBLE


# ============== MinimiseClubsOnAFieldBroadmeadow Tests ==============
# EDGE CASES:
# 1. Hard limit of 5 clubs per field per day
# 2. Only applies to Broadmeadow (Newcastle International Hockey Centre)
# 3. Only Saturday/Sunday games
# 4. Groups by (week, date, field_name)

class TestMinimiseClubsOnAFieldEquivalence:
    """Tests for minimise clubs on a field constraint equivalence."""

    @pytest.mark.parametrize("constraint_class", [
        MinimiseClubsOnAFieldBroadmeadow,
        MinimiseClubsOnAFieldBroadmeadowAI,
    ])
    def test_max_five_clubs_per_field(self, constraint_class, sample_fields):
        """Test that maximum 5 clubs can use a field per day."""
        ef = sample_fields[0]  # EF at Broadmeadow
        
        # Create 7 clubs
        clubs = [Club(name=f'Club{i}', home_field='Newcastle International Hockey Centre') for i in range(7)]
        teams = [Team(name=f'Club{i} 3rd', club=clubs[i], grade='3rd') for i in range(7)]
        
        # Create enough timeslots for 6 games (12 teams = 6 clubs per side)
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time=f'{10+i}:00', week=1, day_slot=i+1, field=ef, round_no=1)
            for i in range(6)
        ]
        
        # 6 unique games between 7 clubs (ensuring 6+ clubs play on same field/day)
        games = [
            ('Club0 3rd', 'Club1 3rd', '3rd'),
            ('Club2 3rd', 'Club3 3rd', '3rd'),
            ('Club4 3rd', 'Club5 3rd', '3rd'),
            ('Club6 3rd', 'Club0 3rd', '3rd'),  # Reuse Club0
            ('Club1 3rd', 'Club2 3rd', '3rd'),  # Reuse Club1, Club2
            ('Club3 3rd', 'Club4 3rd', '3rd'),  # Reuse Club3, Club4
        ]
        
        model, X = create_model_and_vars(games, timeslots)
        
        # Force each unique game to be scheduled exactly once
        for g_idx, (t1, t2, grade) in enumerate(games):
            game_vars = [v for k, v in X.items() if k[0] == t1 and k[1] == t2 and k[2] == grade]
            model.Add(sum(game_vars) == 1)
        
        data = {
            'games': games,
            'timeslots': timeslots,
            'teams': teams,
            'penalties': {},
            'current_week': 0, 'locked_weeks': set(),
        }
        
        constraint = constraint_class()
        constraint.apply(model, X, data)
        
        status = solve_and_get_status(model)
        # Should be infeasible - 7 clubs would be on same field same day
        assert status == cp_model.INFEASIBLE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
