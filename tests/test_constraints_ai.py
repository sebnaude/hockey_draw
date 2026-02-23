# test_constraints_ai.py
"""
Unit tests for AI-enhanced constraint classes.

These tests verify that the AI-enhanced constraints produce the same
outcomes as the original constraints but may solve faster.
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot
from constraints_ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    FiftyFiftyHomeandAwayAI,
    TeamConflictConstraintAI,
    ClubGradeAdjacencyConstraintAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    get_all_ai_constraints,
    get_constraints_by_priority,
    get_staged_constraints,
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


# ============== NoDoubleBookingTeamsConstraintAI Tests ==============

class TestNoDoubleBookingTeamsConstraintAI:
    """Tests for NoDoubleBookingTeamsConstraintAI."""

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
        
        constraint = NoDoubleBookingTeamsConstraintAI()
        constraints_added = constraint.apply(model, X, data)
        
        assert constraints_added > 0
        
        # Force one game in week 1
        week1_games = [v for k, v in X.items() if k[6] == 1]
        model.Add(sum(week1_games) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

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
        
        constraint = NoDoubleBookingTeamsConstraintAI()
        constraint.apply(model, X, data)
        
        # Force Tigers 3rd to play two games in week 1
        tigers_week1_games = [v for k, v in X.items() if k[6] == 1 and 'Tigers 3rd' in (k[0], k[1])]
        model.Add(sum(tigers_week1_games) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        assert status == cp_model.INFEASIBLE


# ============== NoDoubleBookingFieldsConstraintAI Tests ==============

class TestNoDoubleBookingFieldsConstraintAI:
    """Tests for NoDoubleBookingFieldsConstraintAI."""

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
        
        constraint = NoDoubleBookingFieldsConstraintAI()
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


# ============== Constraint Priority Tests ==============

class TestConstraintPriorities:
    """Tests for constraint organization by priority."""

    def test_get_all_constraints(self):
        """Test that all constraints are returned."""
        constraints = get_all_ai_constraints()
        assert len(constraints) >= 15  # We have at least 15 constraint types

    def test_get_required_constraints(self):
        """Test filtering for required constraints."""
        required = get_constraints_by_priority('required')
        assert len(required) >= 3
        
        for c in required:
            assert c.PRIORITY == 'required'

    def test_get_staged_constraints(self):
        """Test getting constraints organized by stage."""
        stages = get_staged_constraints()
        
        assert 'stage1_required' in stages
        assert 'stage2_strong' in stages
        assert 'stage3_medium' in stages
        assert 'stage4_soft' in stages
        
        # Verify no overlap
        all_constraints = []
        for stage_constraints in stages.values():
            for c in stage_constraints:
                assert type(c).__name__ not in [type(x).__name__ for x in all_constraints]
                all_constraints.append(c)


# ============== MaitlandHomeGroupingAI Tests ==============

class TestMaitlandHomeGroupingAI:
    """Tests for MaitlandHomeGroupingAI constraint."""

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
        
        constraint = MaitlandHomeGroupingAI()
        constraint.apply(model, X, data)
        
        # Force home games in both consecutive weeks
        all_games = list(X.values())
        model.Add(sum(all_games) >= 2)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be infeasible due to back-to-back constraint
        assert status == cp_model.INFEASIBLE


# ============== AwayAtMaitlandGroupingAI Tests ==============

class TestAwayAtMaitlandGroupingAI:
    """Tests for AwayAtMaitlandGroupingAI constraint."""

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
        
        constraint = AwayAtMaitlandGroupingAI()
        constraint.apply(model, X, data)
        
        # Force 4 different clubs to visit in same week
        for i in range(4):
            game_vars = [v for k, v in X.items() if k[0] == f'Club{i} 3rd']
            model.Add(sum(game_vars) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Should be infeasible due to max 3 away clubs limit
        assert status == cp_model.INFEASIBLE


# ============== Equivalence Tests ==============

class TestAIConstraintEquivalence:
    """Tests that AI constraints produce equivalent results to original."""

    def test_double_booking_equivalence(self, basic_teams, basic_timeslots, basic_grades, basic_fields, basic_clubs):
        """Test that AI version produces same feasibility as original."""
        from constraints import NoDoubleBookingTeamsConstraint
        
        games = [
            ('Tigers 3rd', 'Wests 3rd', '3rd'),
            ('Tigers 3rd', 'Maitland 3rd', '3rd'),
        ]
        
        data = {
            'games': games,
            'timeslots': basic_timeslots,
            'teams': basic_teams,
            'current_week': 0,
        }
        
        # Test with original constraint
        model1, X1 = create_model_and_vars(games, basic_timeslots)
        original = NoDoubleBookingTeamsConstraint()
        original.apply(model1, X1, data)
        
        model1.Add(sum(X1.values()) >= 2)  # Force 2 games
        
        solver1 = cp_model.CpSolver()
        status1 = solver1.Solve(model1)
        
        # Test with AI constraint
        model2, X2 = create_model_and_vars(games, basic_timeslots)
        ai_version = NoDoubleBookingTeamsConstraintAI()
        ai_version.apply(model2, X2, data)
        
        model2.Add(sum(X2.values()) >= 2)  # Force 2 games
        
        solver2 = cp_model.CpSolver()
        status2 = solver2.Solve(model2)
        
        # Both should give same feasibility result
        assert (status1 in [cp_model.OPTIMAL, cp_model.FEASIBLE]) == \
               (status2 in [cp_model.OPTIMAL, cp_model.FEASIBLE])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
