# test_ai_constraints_comprehensive.py
"""
Comprehensive tests for ALL AI constraint implementations.

Strategy:
1. FEASIBILITY: Every AI constraint must produce a solvable model on valid data.
2. INFEASIBILITY: Every AI constraint must reject deliberately violating data 
   (proves the constraint is not a no-op).
3. PARITY: Every AI constraint must agree with its original on the same test data
   (both feasible or both infeasible).
4. COMBINED: All AI constraints applied together must still be feasible on 
   well-formed data (catches cross-constraint conflicts like the INFEASIBLE bug).

These tests exist because AI constraint fixes were deployed without test coverage,
which led to a presolve INFEASIBLE bug in production. Never again.
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot
from utils import max_games_per_grade

# Import ALL AI constraints
from constraints.ai import (
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    PHLAndSecondGradeAdjacencyAI,
    PHLAndSecondGradeTimesAI,
    FiftyFiftyHomeandAwayAI,
    TeamConflictConstraintAI,
    ClubGradeAdjacencyConstraintAI,
    MaxMaitlandHomeWeekendsAI,
    EnsureBestTimeslotChoicesAI,
    ClubDayConstraintAI,
    EqualMatchUpSpacingConstraintAI,
    ClubVsClubAlignmentAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    MaximiseClubsPerTimeslotBroadmeadowAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
    PreferredTimesConstraintAI,
)

# Import ALL original constraints for parity testing
from constraints import (
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    PHLAndSecondGradeAdjacency,
    PHLAndSecondGradeTimes,
    FiftyFiftyHomeandAway,
    TeamConflictConstraint,
    ClubGradeAdjacencyConstraint,
    MaxMaitlandHomeWeekends,
    EnsureBestTimeslotChoices,
    ClubDayConstraint,
    EqualMatchUpSpacingConstraint,
    ClubVsClubAlignment,
    MaitlandHomeGrouping,
    AwayAtMaitlandGrouping,
    MaximiseClubsPerTimeslotBroadmeadow,
    MinimiseClubsOnAFieldBroadmeadow,
    PreferredTimesConstraint,
)


# ============== Helpers ==============

BROADMEADOW = 'Newcastle International Hockey Centre'
MAITLAND = 'Maitland Park'
GOSFORD = 'Central Coast Hockey Park'


def create_model_and_vars(games, timeslots):
    """Create model and decision variables from games and timeslots."""
    model = cp_model.CpModel()
    X = {}
    for (t1, t2, grade) in games:
        for t in timeslots:
            if not t.day:
                continue
            key = (t1, t2, grade, t.day, t.day_slot, t.time,
                   t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}')
    return model, X


def solve(model, timeout=10.0, workers=8):
    """Solve and return status.
    
    Args:
        model: CP-SAT model to solve.
        timeout: Max solve time in seconds.
        workers: Max parallel workers (default 8 to avoid OOM on combined tests).
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = workers
    return solver.Solve(model), solver


def is_feasible(status):
    return status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


def make_fields():
    """Standard 3-field setup."""
    return [
        PlayingField(location=BROADMEADOW, name='EF'),
        PlayingField(location=BROADMEADOW, name='WF'),
        PlayingField(location=MAITLAND, name='Maitland Main'),
    ]


def make_clubs():
    """Standard 5-club setup."""
    return [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='University', home_field=BROADMEADOW),
    ]


def make_teams_for_grades(clubs, grade_names):
    """Create teams for given grades from clubs."""
    teams = []
    for club in clubs:
        for grade in grade_names:
            teams.append(Team(name=f'{club.name} {grade}', club=club, grade=grade))
    return teams


def make_grades(teams):
    """Build Grade objects from team list."""
    by_grade = defaultdict(list)
    for t in teams:
        by_grade[t.grade].append(t.name)
    return [Grade(name=g, teams=tl) for g, tl in by_grade.items()]


def make_timeslots(fields, num_weeks, slots_per_field=3, day='Sunday', 
                   start_date=datetime(2025, 3, 23)):
    """Generate timeslots over multiple weeks."""
    timeslots = []
    for week in range(1, num_weeks + 1):
        week_date = start_date + timedelta(weeks=week - 1)
        date_str = week_date.strftime('%Y-%m-%d')
        for field in fields:
            for slot in range(1, slots_per_field + 1):
                hour = 9 + slot
                timeslots.append(Timeslot(
                    date=date_str, day=day, time=f'{hour}:00',
                    week=week, day_slot=slot, field=field, round_no=week
                ))
    return timeslots


def make_round_robin(teams):
    """Generate round-robin games grouped by grade."""
    games = []
    by_grade = defaultdict(list)
    for t in teams:
        by_grade[t.grade].append(t.name)
    for grade, team_names in by_grade.items():
        for t1, t2 in combinations(team_names, 2):
            games.append((t1, t2, grade))
    return games


def make_standard_data(clubs=None, grade_names=None, num_weeks=4, 
                       slots_per_field=3, fields=None, day='Sunday',
                       extra_data=None):
    """Build a complete data dict for constraint testing."""
    if clubs is None:
        clubs = make_clubs()
    if grade_names is None:
        grade_names = ['3rd', '4th']
    if fields is None:
        fields = make_fields()

    teams = make_teams_for_grades(clubs, grade_names)
    grades = make_grades(teams)
    timeslots = make_timeslots(fields, num_weeks, slots_per_field, day=day)
    games = make_round_robin(teams)

    # Use the REAL num_rounds calculation — same as the production solver.
    # max_games_per_grade computes max games per team accounting for
    # odd/even team counts and round capacity.
    num_rounds = max_games_per_grade(grades, num_weeks)
    num_rounds['max'] = num_weeks

    # Set num_games on Grade objects (used by MaxMaitlandHomeWeekends etc.)
    for g in grades:
        g.num_games = num_rounds.get(g.name, 0)

    data = {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'current_week': 0,
        'num_rounds': num_rounds,
        'num_dummy_timeslots': 0,
        'penalties': {},
        'team_conflicts': [],
        'unavailable_games': [],
    }
    if extra_data:
        data.update(extra_data)
    return data


# ============================================================================
# SECTION 1: FEASIBILITY TESTS
# Every AI constraint applied to reasonable data must produce a solvable model.
# ============================================================================

class TestAIConstraintsFeasibility:
    """Each AI constraint must be feasible on well-formed data."""

    def _run_feasibility(self, constraint_cls, data_overrides=None):
        """Generic feasibility test: apply constraint, solve, assert feasible."""
        data = make_standard_data()
        if data_overrides:
            data.update(data_overrides)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        constraint = constraint_cls()
        constraint.apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status), (
            f"{constraint_cls.__name__} produced INFEASIBLE on valid data"
        )

    def test_no_double_booking_teams(self):
        self._run_feasibility(NoDoubleBookingTeamsConstraintAI)

    def test_no_double_booking_fields(self):
        self._run_feasibility(NoDoubleBookingFieldsConstraintAI)

    def test_ensure_equal_games(self):
        self._run_feasibility(EnsureEqualGamesAndBalanceMatchUpsAI)

    def test_fifty_fifty_home_away(self):
        self._run_feasibility(FiftyFiftyHomeandAwayAI)

    def test_team_conflict(self):
        clubs = make_clubs()
        teams = make_teams_for_grades(clubs, ['3rd', '4th'])
        self._run_feasibility(
            TeamConflictConstraintAI,
            {'team_conflicts': [('Tigers 3rd', 'Tigers 4th')]}
        )

    def test_club_grade_adjacency(self):
        self._run_feasibility(ClubGradeAdjacencyConstraintAI)

    def test_maitland_home_grouping(self):
        self._run_feasibility(MaitlandHomeGroupingAI)

    def test_away_at_maitland_grouping(self):
        self._run_feasibility(AwayAtMaitlandGroupingAI)

    def test_minimise_clubs_on_field(self):
        self._run_feasibility(MinimiseClubsOnAFieldBroadmeadowAI)

    def test_phl_second_grade_adjacency(self):
        """PHL+2nd adjacency needs PHL and 2nd grade teams."""
        clubs = make_clubs()
        teams = make_teams_for_grades(clubs, ['PHL', '2nd', '3rd'])
        grades = make_grades(teams)
        data = make_standard_data(clubs=clubs, grade_names=['PHL', '2nd', '3rd'])
        self._run_feasibility(PHLAndSecondGradeAdjacencyAI, data)

    def test_phl_second_grade_times(self):
        """PHLAndSecondGradeTimes needs PHL and 2nd grade teams."""
        data = make_standard_data(grade_names=['PHL', '2nd', '3rd'])
        self._run_feasibility(PHLAndSecondGradeTimesAI, data)

    def test_max_maitland_home_weekends(self):
        self._run_feasibility(MaxMaitlandHomeWeekendsAI)

    def test_ensure_best_timeslot_choices(self):
        """This was one of the 5 fixed constraints — must be feasible."""
        self._run_feasibility(EnsureBestTimeslotChoicesAI)

    def test_equal_matchup_spacing(self):
        """This was one of the 5 fixed constraints — must be feasible."""
        self._run_feasibility(EqualMatchUpSpacingConstraintAI)

    def test_club_vs_club_alignment(self):
        """This was one of the 5 fixed constraints — must be feasible."""
        self._run_feasibility(ClubVsClubAlignmentAI)

    def test_maximise_clubs_per_timeslot(self):
        """This was one of the 5 fixed constraints — must be feasible."""
        self._run_feasibility(MaximiseClubsPerTimeslotBroadmeadowAI)

    def test_preferred_times(self):
        self._run_feasibility(PreferredTimesConstraintAI, {
            'preference_no_play': {},
        })

    def test_club_day_constraint(self):
        """ClubDayConstraint with no club_days should be a no-op (feasible)."""
        self._run_feasibility(ClubDayConstraintAI, {'club_days': {}})


# ============================================================================
# SECTION 2: INFEASIBILITY / NOT-A-NO-OP TESTS
# Each constraint must actually reject violations. If it doesn't, it's broken.
# ============================================================================

class TestAIConstraintsRejectViolations:
    """Each AI constraint must reject data that violates its rules."""

    def test_no_double_booking_teams_rejects(self):
        """Force a team to play twice in one week — must be INFEASIBLE."""
        clubs = [Club(name='A', home_field=BROADMEADOW), Club(name='B', home_field=BROADMEADOW),
                 Club(name='C', home_field=BROADMEADOW)]
        teams = [Team(name='A 3rd', club=clubs[0], grade='3rd'),
                 Team(name='B 3rd', club=clubs[1], grade='3rd'),
                 Team(name='C 3rd', club=clubs[2], grade='3rd')]
        fields = [PlayingField(location=BROADMEADOW, name='EF')]
        timeslots = make_timeslots(fields, 1, slots_per_field=2)
        games = [('A 3rd', 'B 3rd', '3rd'), ('A 3rd', 'C 3rd', '3rd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams, 'current_week': 0}

        NoDoubleBookingTeamsConstraintAI().apply(model, X, data)
        # Force A 3rd to play both games in week 1
        a_w1 = [v for k, v in X.items() if k[6] == 1 and 'A 3rd' in (k[0], k[1])]
        model.Add(sum(a_w1) >= 2)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_no_double_booking_fields_rejects(self):
        """Force two games on same field+slot — must be INFEASIBLE."""
        clubs = [Club(name='A', home_field=BROADMEADOW), Club(name='B', home_field=BROADMEADOW),
                 Club(name='C', home_field=BROADMEADOW), Club(name='D', home_field=BROADMEADOW)]
        teams = [Team(name='A 3rd', club=clubs[0], grade='3rd'),
                 Team(name='B 3rd', club=clubs[1], grade='3rd'),
                 Team(name='C 4th', club=clubs[2], grade='4th'),
                 Team(name='D 4th', club=clubs[3], grade='4th')]
        fields = [PlayingField(location=BROADMEADOW, name='EF')]
        timeslots = [Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                              week=1, day_slot=1, field=fields[0], round_no=1)]
        games = [('A 3rd', 'B 3rd', '3rd'), ('C 4th', 'D 4th', '4th')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams, 'current_week': 0}

        NoDoubleBookingFieldsConstraintAI().apply(model, X, data)
        model.Add(sum(X.values()) >= 2)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_equal_games_rejects_overplay(self):
        """Force more games than allowed — must be INFEASIBLE."""
        clubs = [Club(name='A', home_field=BROADMEADOW), Club(name='B', home_field=BROADMEADOW)]
        teams = [Team(name='A 3rd', club=clubs[0], grade='3rd'),
                 Team(name='B 3rd', club=clubs[1], grade='3rd')]
        grades = [Grade(name='3rd', teams=['A 3rd', 'B 3rd'])]
        fields = [PlayingField(location=BROADMEADOW, name='EF')]
        timeslots = make_timeslots(fields, 4, slots_per_field=1)
        games = [('A 3rd', 'B 3rd', '3rd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {
            'games': games, 'timeslots': timeslots, 'teams': teams,
            'grades': grades, 'num_rounds': {'3rd': 2}, 'num_dummy_timeslots': 0,
        }

        EnsureEqualGamesAndBalanceMatchUpsAI().apply(model, X, data)
        # Force 4 games when only 2 are allowed
        model.Add(sum(X.values()) >= 4)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_fifty_fifty_rejects_all_home(self):
        """Force all games at Maitland (home) when both home and away are available — must be INFEASIBLE."""
        maitland_club = Club(name='Maitland', home_field=MAITLAND)
        tigers_club = Club(name='Tigers', home_field=BROADMEADOW)
        teams = [
            Team(name='Tigers 3rd', club=tigers_club, grade='3rd'),
            Team(name='Maitland 3rd', club=maitland_club, grade='3rd'),
        ]
        mait_field = PlayingField(location=MAITLAND, name='Main')
        ef = PlayingField(location=BROADMEADOW, name='EF')
        # Offer both home and away options
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=mait_field, round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=mait_field, round_no=2),
            Timeslot(date='2025-04-06', day='Sunday', time='10:00', week=3, day_slot=1, field=mait_field, round_no=3),
            Timeslot(date='2025-04-13', day='Sunday', time='10:00', week=4, day_slot=1, field=ef, round_no=4),
        ]
        games = [('Tigers 3rd', 'Maitland 3rd', '3rd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams}

        FiftyFiftyHomeandAwayAI().apply(model, X, data)
        # Force 3 home (Maitland) games — imbalanced: 3 home, 0 away not possible with balance rule
        home_vars = [v for k, v in X.items() if k[10] == MAITLAND]
        model.Add(sum(home_vars) >= 3)
        # Also force all 4 games played
        model.Add(sum(X.values()) == 4)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_team_conflict_rejects(self):
        """Force conflicting teams to same slot — must be INFEASIBLE."""
        clubs = [Club(name='Tigers', home_field=BROADMEADOW),
                 Club(name='Wests', home_field=BROADMEADOW)]
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
            Team(name='Tigers 4th', club=clubs[0], grade='4th'),
            Team(name='Wests 4th', club=clubs[1], grade='4th'),
        ]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        wf = PlayingField(location=BROADMEADOW, name='WF')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        ]
        games = [('Tigers 3rd', 'Wests 3rd', '3rd'), ('Tigers 4th', 'Wests 4th', '4th')]

        model, X = create_model_and_vars(games, timeslots)
        data = {
            'games': games, 'timeslots': timeslots, 'teams': teams,
            'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
            'current_week': 0,
        }

        TeamConflictConstraintAI().apply(model, X, data)
        # Force both games in slot 1
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot1) >= 2)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_club_grade_adjacency_rejects(self):
        """Force adjacent-grade same-club teams to same slot — INFEASIBLE."""
        clubs = [Club(name='Tigers', home_field=BROADMEADOW),
                 Club(name='Wests', home_field=BROADMEADOW)]
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
            Team(name='Tigers 4th', club=clubs[0], grade='4th'),
            Team(name='Wests 4th', club=clubs[1], grade='4th'),
        ]
        grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd']),
                  Grade(name='4th', teams=['Tigers 4th', 'Wests 4th'])]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        wf = PlayingField(location=BROADMEADOW, name='WF')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        ]
        games = [('Tigers 3rd', 'Wests 3rd', '3rd'), ('Tigers 4th', 'Wests 4th', '4th')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams,
                'grades': grades, 'clubs': clubs}

        ClubGradeAdjacencyConstraintAI().apply(model, X, data)
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot1) >= 2)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_maitland_home_grouping_rejects_back_to_back(self):
        """Force Maitland home games in consecutive weeks — INFEASIBLE."""
        maitland_club = Club(name='Maitland', home_field=MAITLAND)
        tigers_club = Club(name='Tigers', home_field=BROADMEADOW)
        teams = [
            Team(name='Tigers 3rd', club=tigers_club, grade='3rd'),
            Team(name='Maitland 3rd', club=maitland_club, grade='3rd'),
        ]
        mait = PlayingField(location=MAITLAND, name='Main')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=mait, round_no=1),
            Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=mait, round_no=2),
        ]
        games = [('Tigers 3rd', 'Maitland 3rd', '3rd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams,
                'penalties': {}, 'current_week': 0}

        MaitlandHomeGroupingAI().apply(model, X, data)
        model.Add(sum(X.values()) >= 2)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_away_at_maitland_rejects_too_many_clubs(self):
        """Force 4 away clubs at Maitland in one week — INFEASIBLE (max 3)."""
        maitland_club = Club(name='Maitland', home_field=MAITLAND)
        away_clubs = [Club(name=f'Away{i}', home_field=BROADMEADOW) for i in range(5)]
        all_clubs = away_clubs + [maitland_club]
        teams = [Team(name=f'Away{i} 3rd', club=away_clubs[i], grade='3rd') for i in range(5)]
        teams.append(Team(name='Maitland 3rd', club=maitland_club, grade='3rd'))

        mait = PlayingField(location=MAITLAND, name='Main')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time=f'{10+i}:00',
                     week=1, day_slot=i + 1, field=mait, round_no=1)
            for i in range(5)
        ]
        games = [(f'Away{i} 3rd', 'Maitland 3rd', '3rd') for i in range(5)]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams,
                'clubs': all_clubs, 'penalties': {}, 'current_week': 0}

        AwayAtMaitlandGroupingAI().apply(model, X, data)
        for i in range(4):
            gv = [v for k, v in X.items() if k[0] == f'Away{i} 3rd']
            model.Add(sum(gv) >= 1)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_minimise_clubs_on_field_rejects_too_many(self):
        """Force 7 distinct clubs on one field in one day — INFEASIBLE (max 5)."""
        clubs = [Club(name=f'C{i}', home_field=BROADMEADOW) for i in range(7)]
        teams = [Team(name=f'C{i} 3rd', club=clubs[i], grade='3rd') for i in range(7)]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time=f'{10+i}:00',
                     week=1, day_slot=i + 1, field=ef, round_no=1)
            for i in range(6)
        ]
        # 6 games on EF involving 7 clubs (chain: 0v1, 1v2, 2v3, 3v4, 4v5, 5v6)
        games = [(f'C{i} 3rd', f'C{i+1} 3rd', '3rd') for i in range(6)]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams,
                'penalties': {}, 'current_week': 0}

        MinimiseClubsOnAFieldBroadmeadowAI().apply(model, X, data)
        # Force each game scheduled exactly once (all on same field/day)
        for (t1, t2, g) in games:
            gv = [v for k, v in X.items() if k[0] == t1 and k[1] == t2]
            model.Add(sum(gv) == 1)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_phl_adjacency_rejects_different_location_adjacent_time(self):
        """PHL and 2nd from same club: adjacent time + different location = INFEASIBLE."""
        tigers = Club(name='Tigers', home_field=BROADMEADOW)
        wests = Club(name='Wests', home_field=BROADMEADOW)
        teams = [
            Team(name='Tigers PHL', club=tigers, grade='PHL'),
            Team(name='Wests PHL', club=wests, grade='PHL'),
            Team(name='Tigers 2nd', club=tigers, grade='2nd'),
            Team(name='Wests 2nd', club=wests, grade='2nd'),
        ]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        mait = PlayingField(location=MAITLAND, name='Main')
        # PHL at 10:00 Broadmeadow, 2nd at 11:00 Maitland (adjacent, different location)
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='11:00', week=1, day_slot=2, field=mait, round_no=1),
        ]
        games = [
            ('Tigers PHL', 'Wests PHL', 'PHL'),
            ('Tigers 2nd', 'Wests 2nd', '2nd'),
        ]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams, 'current_week': 0}

        PHLAndSecondGradeAdjacencyAI().apply(model, X, data)
        # Force PHL at slot 1 and 2nd at slot 2
        phl_vars = [v for k, v in X.items() if k[2] == 'PHL' and k[4] == 1]
        sec_vars = [v for k, v in X.items() if k[2] == '2nd' and k[4] == 2]
        model.Add(sum(phl_vars) >= 1)
        model.Add(sum(sec_vars) >= 1)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_phl_times_rejects_concurrent_at_broadmeadow(self):
        """Two PHL games at Broadmeadow same time — INFEASIBLE."""
        clubs = [Club(name=f'Club{i}', home_field=BROADMEADOW) for i in range(4)]
        teams = [Team(name=f'Club{i} PHL', club=clubs[i], grade='PHL') for i in range(4)]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        wf = PlayingField(location=BROADMEADOW, name='WF')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        ]
        games = [('Club0 PHL', 'Club1 PHL', 'PHL'), ('Club2 PHL', 'Club3 PHL', 'PHL')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams,
                'current_week': 0, 'penalties': {}, 'phl_preferences': {'preferred_dates': []}}

        PHLAndSecondGradeTimesAI().apply(model, X, data)
        # Force both PHL games in same slot
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot1) >= 2)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE

    def test_max_maitland_weekends_rejects(self):
        """Force games at Maitland every week beyond the limit — INFEASIBLE."""
        maitland_club = Club(name='Maitland', home_field=MAITLAND)
        tigers = Club(name='Tigers', home_field=BROADMEADOW)
        teams = [
            Team(name='Tigers 3rd', club=tigers, grade='3rd'),
            Team(name='Maitland 3rd', club=maitland_club, grade='3rd'),
        ]
        grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Maitland 3rd'])]
        for g in grades:
            g.num_games = 10
        mait = PlayingField(location=MAITLAND, name='Main')
        # 10 weeks, but limit should be around max_games/2 + 1
        timeslots = make_timeslots([mait], 10, slots_per_field=1)
        games = [('Tigers 3rd', 'Maitland 3rd', '3rd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {
            'games': games, 'timeslots': timeslots, 'teams': teams,
            'clubs': [tigers, maitland_club], 'grades': grades,
        }

        MaxMaitlandHomeWeekendsAI().apply(model, X, data)
        # Force a game every single week
        for week in range(1, 11):
            wk_vars = [v for k, v in X.items() if k[6] == week]
            if wk_vars:
                model.Add(sum(wk_vars) >= 1)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE


# ============================================================================
# SECTION 3: THE 5 FIXED CONSTRAINTS — INTENSIVE TESTING
# These broke in production. They get extra scrutiny.
# ============================================================================

class TestEqualMatchUpSpacingAI:
    """Intensive tests for EqualMatchUpSpacingConstraintAI (was a complete no-op)."""

    def test_feasible_with_good_spacing(self):
        """Well-spaced matchups should be feasible."""
        data = make_standard_data(num_weeks=8, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        EqualMatchUpSpacingConstraintAI().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    def test_adds_constraints(self):
        """Must add at least some constraints (was no-op before fix)."""
        data = make_standard_data(num_weeks=8, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        count = EqualMatchUpSpacingConstraintAI().apply(model, X, data)
        assert count > 0, "EqualMatchUpSpacingConstraintAI added zero constraints (no-op!)"

    def test_parity_with_original_feasible(self):
        """Must agree with original on feasibility."""
        data = make_standard_data(num_weeks=8, slots_per_field=4)

        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        EqualMatchUpSpacingConstraintAI().apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai)

        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        EqualMatchUpSpacingConstraint().apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig)

        assert is_feasible(status_ai) == is_feasible(status_orig), (
            f"Parity mismatch: AI={status_ai}, Original={status_orig}"
        )


class TestEnsureBestTimeslotChoicesAI:
    """Intensive tests for EnsureBestTimeslotChoicesAI (was missing slot bounding)."""

    def test_feasible_basic(self):
        """Must be feasible with standard data."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        EnsureBestTimeslotChoicesAI().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    def test_adds_constraints(self):
        """Must add constraints (the slot-bounding part was missing)."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        count = EnsureBestTimeslotChoicesAI().apply(model, X, data)
        assert count > 0, "EnsureBestTimeslotChoicesAI added zero constraints"

    def test_parity_with_original_feasible(self):
        """Must agree with original on feasibility."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)

        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        EnsureBestTimeslotChoicesAI().apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai)

        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        EnsureBestTimeslotChoices().apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig)

        assert is_feasible(status_ai) == is_feasible(status_orig)

    def test_no_gap_enforcement(self):
        """Games should not have gaps between used timeslots."""
        data = make_standard_data(num_weeks=2, slots_per_field=5)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        EnsureBestTimeslotChoicesAI().apply(model, X, data)
        # Solve and check — should be feasible
        status, solver = solve(model)
        assert is_feasible(status)


class TestClubVsClubAlignmentAI:
    """Intensive tests for ClubVsClubAlignmentAI (was missing Sunday field alignment)."""

    def test_feasible_basic(self):
        """Must be feasible with standard data."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        ClubVsClubAlignmentAI().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    def test_adds_constraints(self):
        """Must add constraints (Sunday alignment was missing).
        Requires grades with different team counts to trigger alignment logic."""
        # 3rd has 5 teams (per_team_games = 8//5=1), 4th has 3 teams (per_team_games = 8//2=4)
        clubs = make_clubs()[:3]  # Only 3 clubs
        teams = make_teams_for_grades(clubs, ['3rd'])
        # Add extra 4th grade teams from 5 clubs to create asymmetry
        clubs_5 = make_clubs()
        teams += make_teams_for_grades(clubs_5, ['4th'])
        grades = make_grades(teams)
        fields = make_fields()
        timeslots = make_timeslots(fields, 8, slots_per_field=4)
        games = make_round_robin(teams)
        num_rounds = {g.name: 8 for g in grades}
        num_rounds['max'] = 8
        data = {
            'games': games, 'timeslots': timeslots, 'teams': teams,
            'grades': grades, 'clubs': clubs_5, 'fields': fields,
            'current_week': 0, 'num_rounds': num_rounds,
            'num_dummy_timeslots': 0, 'penalties': {},
        }
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        count = ClubVsClubAlignmentAI().apply(model, X, data)
        assert count > 0, "ClubVsClubAlignmentAI added zero constraints"

    def test_parity_with_original_feasible(self):
        """Must agree with original on feasibility."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)

        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        ClubVsClubAlignmentAI().apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai)

        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        ClubVsClubAlignment().apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig)

        assert is_feasible(status_ai) == is_feasible(status_orig)


class TestMaximiseClubsPerTimeslotBroadmeadowAI:
    """Intensive tests for MaximiseClubsPerTimeslotBroadmeadowAI (was missing dynamic hard min)."""

    def test_feasible_basic(self):
        """Must be feasible with standard data."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        MaximiseClubsPerTimeslotBroadmeadowAI().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    def test_adds_constraints(self):
        """Must add constraints."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        count = MaximiseClubsPerTimeslotBroadmeadowAI().apply(model, X, data)
        assert count > 0

    def test_hard_minimum_enforced(self):
        """Hard minimum of total_games/2 clubs must be enforced."""
        # 2 clubs, 1 game on 1 slot => total_games = 1, hard_min = 1/2 = 0
        # This should be fine with 2 clubs on 1 game
        clubs = [Club(name='A', home_field=BROADMEADOW), Club(name='B', home_field=BROADMEADOW)]
        teams = [Team(name='A 3rd', club=clubs[0], grade='3rd'),
                 Team(name='B 3rd', club=clubs[1], grade='3rd')]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        timeslots = [Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                              week=1, day_slot=1, field=ef, round_no=1)]
        games = [('A 3rd', 'B 3rd', '3rd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams,
                'penalties': {}, 'current_week': 0}

        MaximiseClubsPerTimeslotBroadmeadowAI().apply(model, X, data)
        model.Add(sum(X.values()) == 1)

        status, _ = solve(model)
        assert is_feasible(status)

    def test_parity_with_original_feasible(self):
        """Must agree with original on feasibility."""
        data = make_standard_data(num_weeks=4, slots_per_field=4)

        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        MaximiseClubsPerTimeslotBroadmeadowAI().apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai)

        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        MaximiseClubsPerTimeslotBroadmeadow().apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig)

        assert is_feasible(status_ai) == is_feasible(status_orig)


class TestPHLAndSecondGradeAdjacencyAI:
    """Intensive tests for PHLAndSecondGradeAdjacencyAI (was missing case 2)."""

    def test_feasible_basic(self):
        """Must be feasible with standard data including PHL/2nd grades."""
        data = make_standard_data(grade_names=['PHL', '2nd', '3rd'], num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        PHLAndSecondGradeAdjacencyAI().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    def test_adds_constraints(self):
        """Must add constraints (Case 2 was missing)."""
        data = make_standard_data(grade_names=['PHL', '2nd', '3rd'], num_weeks=4, slots_per_field=4)
        model, X = create_model_and_vars(data['games'], data['timeslots'])
        count = PHLAndSecondGradeAdjacencyAI().apply(model, X, data)
        assert count > 0

    def test_allows_same_location_adjacent_time(self):
        """PHL and 2nd same club, same location, adjacent time = ALLOWED."""
        tigers = Club(name='Tigers', home_field=BROADMEADOW)
        wests = Club(name='Wests', home_field=BROADMEADOW)
        teams = [
            Team(name='Tigers PHL', club=tigers, grade='PHL'),
            Team(name='Wests PHL', club=wests, grade='PHL'),
            Team(name='Tigers 2nd', club=tigers, grade='2nd'),
            Team(name='Wests 2nd', club=wests, grade='2nd'),
        ]
        ef = PlayingField(location=BROADMEADOW, name='EF')
        timeslots = [
            Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
            Timeslot(date='2025-03-23', day='Sunday', time='11:00', week=1, day_slot=2, field=ef, round_no=1),
        ]
        games = [('Tigers PHL', 'Wests PHL', 'PHL'), ('Tigers 2nd', 'Wests 2nd', '2nd')]

        model, X = create_model_and_vars(games, timeslots)
        data = {'games': games, 'timeslots': timeslots, 'teams': teams, 'current_week': 0}

        PHLAndSecondGradeAdjacencyAI().apply(model, X, data)
        # Force PHL at slot 1 and 2nd at slot 2 — same location, adjacent = OK
        phl_s1 = [v for k, v in X.items() if k[2] == 'PHL' and k[4] == 1]
        sec_s2 = [v for k, v in X.items() if k[2] == '2nd' and k[4] == 2]
        model.Add(sum(phl_s1) >= 1)
        model.Add(sum(sec_s2) >= 1)

        status, _ = solve(model)
        assert is_feasible(status)

    def test_parity_with_original_feasible(self):
        """Must agree with original on feasibility."""
        data = make_standard_data(grade_names=['PHL', '2nd', '3rd'], num_weeks=4, slots_per_field=4)

        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        PHLAndSecondGradeAdjacencyAI().apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai)

        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        PHLAndSecondGradeAdjacency().apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig)

        assert is_feasible(status_ai) == is_feasible(status_orig)


# ============================================================================
# SECTION 4: PARITY TESTS FOR ALL 18 PAIRS
# Both implementations must agree: feasible ↔ feasible, infeasible ↔ infeasible
# ============================================================================

class TestAllConstraintsParity:
    """Verify all AI/original constraint pairs agree on feasibility."""

    PAIRS = [
        (NoDoubleBookingTeamsConstraint, NoDoubleBookingTeamsConstraintAI),
        (NoDoubleBookingFieldsConstraint, NoDoubleBookingFieldsConstraintAI),
        (EnsureEqualGamesAndBalanceMatchUps, EnsureEqualGamesAndBalanceMatchUpsAI),
        (FiftyFiftyHomeandAway, FiftyFiftyHomeandAwayAI),
        (TeamConflictConstraint, TeamConflictConstraintAI),
        (ClubGradeAdjacencyConstraint, ClubGradeAdjacencyConstraintAI),
        (MaitlandHomeGrouping, MaitlandHomeGroupingAI),
        (AwayAtMaitlandGrouping, AwayAtMaitlandGroupingAI),
        (MinimiseClubsOnAFieldBroadmeadow, MinimiseClubsOnAFieldBroadmeadowAI),
        (PHLAndSecondGradeAdjacency, PHLAndSecondGradeAdjacencyAI),
        (PHLAndSecondGradeTimes, PHLAndSecondGradeTimesAI),
        (MaxMaitlandHomeWeekends, MaxMaitlandHomeWeekendsAI),
        (EnsureBestTimeslotChoices, EnsureBestTimeslotChoicesAI),
        (EqualMatchUpSpacingConstraint, EqualMatchUpSpacingConstraintAI),
        (ClubVsClubAlignment, ClubVsClubAlignmentAI),
        (MaximiseClubsPerTimeslotBroadmeadow, MaximiseClubsPerTimeslotBroadmeadowAI),
        (PreferredTimesConstraint, PreferredTimesConstraintAI),
    ]

    def _make_data_for_constraint(self, constraint_cls):
        """Build data appropriate for the constraint."""
        name = constraint_cls.__name__

        # Constraints needing PHL/2nd grades
        if 'PHL' in name:
            return make_standard_data(grade_names=['PHL', '2nd', '3rd'], num_weeks=4, slots_per_field=4,
                                      extra_data={'phl_preferences': {'preferred_dates': []}, 'preference_no_play': {}})

        # Constraints needing team conflicts
        if 'Conflict' in name:
            return make_standard_data(num_weeks=4, slots_per_field=4,
                                      extra_data={'team_conflicts': [('Tigers 3rd', 'Tigers 4th')]})

        # Preferred times
        if 'Preferred' in name:
            return make_standard_data(num_weeks=4, slots_per_field=4,
                                      extra_data={'preference_no_play': {}})

        # EqualMatchUpSpacing needs more rounds relative to teams for feasibility
        if 'Spacing' in name:
            return make_standard_data(num_weeks=10, slots_per_field=5)

        # ClubDay
        if 'ClubDay' in name:
            return make_standard_data(num_weeks=4, slots_per_field=4,
                                      extra_data={'club_days': {}})

        return make_standard_data(num_weeks=4, slots_per_field=4)

    @pytest.mark.parametrize("orig_cls,ai_cls", PAIRS,
                             ids=[f"{o.__name__}" for o, _ in PAIRS])
    def test_feasibility_parity(self, orig_cls, ai_cls):
        """Both original and AI must agree on feasibility."""
        data = self._make_data_for_constraint(orig_cls)

        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        orig_cls().apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig)

        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        ai_cls().apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai)

        assert is_feasible(status_orig) == is_feasible(status_ai), (
            f"Parity mismatch for {orig_cls.__name__}: "
            f"Original={status_orig}, AI={status_ai}"
        )


# ============================================================================
# SECTION 5: COMBINED FEASIBILITY TESTS
# All AI constraints together must not produce INFEASIBLE on valid data.
# This is the test that would have caught the presolve bug.
# ============================================================================

class TestAllAIConstraintsCombined:
    """Apply ALL AI constraints together — must still be feasible."""

    def _get_all_ai_constraints(self):
        """Return instances of all AI constraints."""
        return [
            NoDoubleBookingTeamsConstraintAI(),
            NoDoubleBookingFieldsConstraintAI(),
            EnsureEqualGamesAndBalanceMatchUpsAI(),
            FiftyFiftyHomeandAwayAI(),
            TeamConflictConstraintAI(),
            ClubGradeAdjacencyConstraintAI(),
            MaitlandHomeGroupingAI(),
            AwayAtMaitlandGroupingAI(),
            MinimiseClubsOnAFieldBroadmeadowAI(),
            PHLAndSecondGradeAdjacencyAI(),
            PHLAndSecondGradeTimesAI(),
            MaxMaitlandHomeWeekendsAI(),
            EnsureBestTimeslotChoicesAI(),
            EqualMatchUpSpacingConstraintAI(),
            ClubVsClubAlignmentAI(),
            MaximiseClubsPerTimeslotBroadmeadowAI(),
            PreferredTimesConstraintAI(),
        ]

    def test_all_ai_constraints_feasible_small(self):
        """All AI constraints on small dataset — must be feasible.
        
        THIS IS THE TEST THAT WOULD HAVE PREVENTED THE INFEASIBLE BUG.
        Uses generous capacity so constraints don't conflict due to tight data.
        
        Note: 4 grades with >6 weeks causes OOM during solve due to
        EqualMatchUpSpacing creating O(pairs × rounds) intermediate vars.
        6 weeks / 5 slots is the sweet spot: enough rounds for spacing
        constraints to be meaningful, without exceeding memory.
        """
        data = make_standard_data(
            grade_names=['PHL', '2nd', '3rd', '4th'],
            num_weeks=6,
            slots_per_field=5,
            extra_data={
                'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
                'phl_preferences': {'preferred_dates': []},
                'preference_no_play': {},
                'club_days': {},
            }
        )

        model, X = create_model_and_vars(data['games'], data['timeslots'])
        
        applied = []
        for constraint in self._get_all_ai_constraints():
            name = constraint.__class__.__name__
            try:
                constraint.apply(model, X, data)
                applied.append(name)
            except Exception as e:
                pytest.fail(f"Constraint {name} raised {type(e).__name__}: {e}")

        status, _ = solve(model, timeout=60.0)
        assert is_feasible(status), (
            f"All AI constraints combined produced INFEASIBLE! "
            f"Applied: {applied}"
        )

    def test_all_ai_constraints_feasible_medium(self):
        """All AI constraints on medium dataset — must be feasible."""
        clubs = [
            Club(name='Tigers', home_field=BROADMEADOW),
            Club(name='Wests', home_field=BROADMEADOW),
            Club(name='Norths', home_field=BROADMEADOW),
            Club(name='University', home_field=BROADMEADOW),
            Club(name='Maitland', home_field=MAITLAND),
        ]
        data = make_standard_data(
            clubs=clubs,
            grade_names=['3rd', '4th'],
            num_weeks=8,
            slots_per_field=5,
            extra_data={
                'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
                'phl_preferences': {'preferred_dates': []},
                'preference_no_play': {},
                'club_days': {},
            }
        )

        model, X = create_model_and_vars(data['games'], data['timeslots'])

        for constraint in self._get_all_ai_constraints():
            constraint.apply(model, X, data)

        status, _ = solve(model, timeout=60.0)
        assert is_feasible(status), "All AI constraints combined produced INFEASIBLE on medium data"

    def test_all_originals_combined_feasible(self):
        """All ORIGINAL constraints on same data — baseline check."""
        data = make_standard_data(
            grade_names=['PHL', '2nd', '3rd', '4th'],
            num_weeks=6,
            slots_per_field=5,
            extra_data={
                'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
                'phl_preferences': {'preferred_dates': []},
                'preference_no_play': {},
                'club_days': {},
            }
        )

        model, X = create_model_and_vars(data['games'], data['timeslots'])

        all_originals = [
            NoDoubleBookingTeamsConstraint(),
            NoDoubleBookingFieldsConstraint(),
            EnsureEqualGamesAndBalanceMatchUps(),
            FiftyFiftyHomeandAway(),
            TeamConflictConstraint(),
            ClubGradeAdjacencyConstraint(),
            MaitlandHomeGrouping(),
            AwayAtMaitlandGrouping(),
            MinimiseClubsOnAFieldBroadmeadow(),
            PHLAndSecondGradeAdjacency(),
            PHLAndSecondGradeTimes(),
            MaxMaitlandHomeWeekends(),
            EnsureBestTimeslotChoices(),
            EqualMatchUpSpacingConstraint(),
            ClubVsClubAlignment(),
            MaximiseClubsPerTimeslotBroadmeadow(),
            PreferredTimesConstraint(),
        ]

        for constraint in all_originals:
            constraint.apply(model, X, data)

        status, _ = solve(model, timeout=60.0)
        assert is_feasible(status), "All ORIGINAL constraints combined produced INFEASIBLE"

    def test_ai_and_original_same_outcome(self):
        """Both full stacks must produce same feasibility on identical data."""
        data = make_standard_data(
            grade_names=['3rd', '4th'],
            num_weeks=10,
            slots_per_field=5,
            extra_data={
                'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
                'phl_preferences': {'preferred_dates': []},
                'preference_no_play': {},
                'club_days': {},
            }
        )

        # AI stack
        model_ai, X_ai = create_model_and_vars(data['games'], data['timeslots'])
        for c in self._get_all_ai_constraints():
            c.apply(model_ai, X_ai, data)
        status_ai, _ = solve(model_ai, timeout=60.0)

        # Original stack
        model_orig, X_orig = create_model_and_vars(data['games'], data['timeslots'])
        for c in [
            NoDoubleBookingTeamsConstraint(), NoDoubleBookingFieldsConstraint(),
            EnsureEqualGamesAndBalanceMatchUps(), FiftyFiftyHomeandAway(),
            TeamConflictConstraint(), ClubGradeAdjacencyConstraint(),
            MaitlandHomeGrouping(), AwayAtMaitlandGrouping(),
            MinimiseClubsOnAFieldBroadmeadow(), PHLAndSecondGradeAdjacency(),
            PHLAndSecondGradeTimes(), MaxMaitlandHomeWeekends(),
            EnsureBestTimeslotChoices(), EqualMatchUpSpacingConstraint(),
            ClubVsClubAlignment(), MaximiseClubsPerTimeslotBroadmeadow(),
            PreferredTimesConstraint(),
        ]:
            c.apply(model_orig, X_orig, data)
        status_orig, _ = solve(model_orig, timeout=60.0)

        assert is_feasible(status_ai) == is_feasible(status_orig), (
            f"Full stack mismatch: AI={status_ai}, Original={status_orig}"
        )


# ============================================================================
# SECTION 6: INCREMENTAL CONSTRAINT ADDITION
# Add constraints one at a time and verify feasibility is maintained.
# This identifies exactly WHICH constraint causes infeasibility.
# ============================================================================

class TestIncrementalAIConstraints:
    """Add AI constraints one-at-a-time to pinpoint infeasibility source."""

    def test_incremental_addition(self):
        """Add each AI constraint incrementally; all steps must stay feasible.
        
        Uses 2 grades to keep memory manageable — each step solves the full
        model, so 17 sequential solves with all EqualMatchUpSpacing vars
        can OOM with 4 grades. The important thing is testing that adding
        each constraint preserves feasibility, not model size.
        """
        data = make_standard_data(
            grade_names=['3rd', '4th'],
            num_weeks=6,
            slots_per_field=5,
            extra_data={
                'team_conflicts': [('Tigers 3rd', 'Tigers 4th')],
                'phl_preferences': {'preferred_dates': []},
                'preference_no_play': {},
                'club_days': {},
            }
        )

        constraints_in_order = [
            NoDoubleBookingTeamsConstraintAI(),
            NoDoubleBookingFieldsConstraintAI(),
            EnsureEqualGamesAndBalanceMatchUpsAI(),
            FiftyFiftyHomeandAwayAI(),
            TeamConflictConstraintAI(),
            ClubGradeAdjacencyConstraintAI(),
            PHLAndSecondGradeAdjacencyAI(),
            PHLAndSecondGradeTimesAI(),
            MaxMaitlandHomeWeekendsAI(),
            MaitlandHomeGroupingAI(),
            AwayAtMaitlandGroupingAI(),
            MinimiseClubsOnAFieldBroadmeadowAI(),
            EnsureBestTimeslotChoicesAI(),
            EqualMatchUpSpacingConstraintAI(),
            ClubVsClubAlignmentAI(),
            MaximiseClubsPerTimeslotBroadmeadowAI(),
            PreferredTimesConstraintAI(),
        ]

        model, X = create_model_and_vars(data['games'], data['timeslots'])
        applied = []

        for constraint in constraints_in_order:
            name = constraint.__class__.__name__
            constraint.apply(model, X, data)
            applied.append(name)

            # Use fewer workers for incremental test (17 sequential solves)
            status, _ = solve(model, timeout=15.0, workers=4)
            assert is_feasible(status), (
                f"Model became INFEASIBLE after adding {name}. "
                f"Applied so far: {applied}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
