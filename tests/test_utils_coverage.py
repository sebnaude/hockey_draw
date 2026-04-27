# test_utils_coverage.py
"""
Coverage-focused tests for utils.py targeting uncovered lines/branches.

Covers: convert_X_to_roster edge cases, generate_X filtering logic,
circle_method_round_1_pairings, forced/blocked game rules, max_games_per_grade
edge cases, and generate_timeslots edge cases.

No mocks/patches - uses real objects and real config data.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot, Game, WeeklyDraw, Roster
from utils import (
    convert_X_to_roster,
    generate_X,
    generate_games,
    generate_timeslots,
    max_games_per_grade,
    circle_method_round_1_pairings,
    _build_forced_game_rules,
    _check_forced_game_status,
    _get_matching_forced_scopes,
    _build_blocked_game_rules,
    _is_blocked_by_no_play,
    build_season_data,
)


# ============== Fixtures ==============

@pytest.fixture
def nihc_field():
    return PlayingField(name='EF', location='Newcastle International Hockey Centre')

@pytest.fixture
def wf_field():
    return PlayingField(name='WF', location='Newcastle International Hockey Centre')

@pytest.fixture
def maitland_field():
    return PlayingField(name='MF', location='Maitland Park')

@pytest.fixture
def gosford_field():
    return PlayingField(name='GF', location='Central Coast Hockey Park')

@pytest.fixture
def clubs():
    return [
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
        Club(name='Gosford', home_field='Central Coast Hockey Park'),
    ]

@pytest.fixture
def teams(clubs):
    norths, tigers, maitland, gosford = clubs
    return [
        Team(name='Norths PHL', club=norths, grade='PHL'),
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
        Team(name='Gosford PHL', club=gosford, grade='PHL'),
        Team(name='Norths 2nd', club=norths, grade='2nd'),
        Team(name='Tigers 2nd', club=tigers, grade='2nd'),
        Team(name='Maitland 2nd', club=maitland, grade='2nd'),
        Team(name='Norths 3rd', club=norths, grade='3rd'),
        Team(name='Tigers 3rd', club=tigers, grade='3rd'),
        Team(name='Maitland 3rd', club=maitland, grade='3rd'),
    ]

@pytest.fixture
def grades():
    return [
        Grade(name='PHL', teams=['Norths PHL', 'Tigers PHL', 'Maitland PHL', 'Gosford PHL']),
        Grade(name='2nd', teams=['Norths 2nd', 'Tigers 2nd', 'Maitland 2nd']),
        Grade(name='3rd', teams=['Norths 3rd', 'Tigers 3rd', 'Maitland 3rd']),
    ]

@pytest.fixture
def fields(nihc_field, wf_field, maitland_field, gosford_field):
    return [nihc_field, wf_field, maitland_field, gosford_field]

@pytest.fixture
def sunday_timeslots(nihc_field, wf_field, maitland_field, gosford_field):
    """A minimal set of timeslots for testing generate_X."""
    slots = []
    for week in [1, 2]:
        for field in [nihc_field, wf_field, maitland_field, gosford_field]:
            slots.append(Timeslot(
                date=f'2026-03-{20 + week}', day='Sunday', time='11:30',
                week=week, day_slot=1, field=field, round_no=week,
            ))
            slots.append(Timeslot(
                date=f'2026-03-{20 + week}', day='Sunday', time='13:00',
                week=week, day_slot=2, field=field, round_no=week,
            ))
    return slots

@pytest.fixture
def friday_timeslots(nihc_field, gosford_field):
    """Friday timeslots (PHL only)."""
    return [
        Timeslot(date='2026-03-20', day='Friday', time='19:00',
                 week=1, day_slot=1, field=nihc_field, round_no=1),
        Timeslot(date='2026-03-20', day='Friday', time='20:00',
                 week=1, day_slot=2, field=gosford_field, round_no=1),
    ]

@pytest.fixture
def all_timeslots(sunday_timeslots, friday_timeslots):
    return sunday_timeslots + friday_timeslots

@pytest.fixture
def base_data(teams, grades, fields, all_timeslots):
    """Minimal data dict for generate_X."""
    return {
        'teams': teams,
        'grades': grades,
        'fields': fields,
        'timeslots': all_timeslots,
        'phl_game_times': {},
        'second_grade_times': {},
        'home_field_map': {},
        'forced_games': [],
        'blocked_games': [],
    }


# ============== convert_X_to_roster ==============

class TestConvertXToRosterCoverage:
    """Cover lines 91-93 (solution_value branch), 96-98 (short keys/unpacking)."""

    def test_skips_false_vars(self, teams, fields, grades, nihc_field):
        """Line 93-94: elif not var: continue -- skips vars with falsy value."""
        data = {'teams': teams, 'fields': fields, 'grades': grades}
        key = ('Norths PHL', 'Tigers PHL', 'PHL', 'Sunday', 1, '11:30',
               1, '2026-03-21', 1, 'EF', 'Newcastle International Hockey Centre')
        X = {key: 0}  # falsy value -> should skip
        roster = convert_X_to_roster(X, data)
        assert len(roster.weeks) == 0

    def test_skips_short_keys(self, teams, fields, grades):
        """Line 96-97: len(key) < 11 -> skip malformed keys."""
        data = {'teams': teams, 'fields': fields, 'grades': grades}
        short_key = ('Norths PHL', 'Tigers PHL', 'PHL', 0)
        X = {short_key: 1}
        roster = convert_X_to_roster(X, data)
        assert len(roster.weeks) == 0

    def test_converts_true_var(self, teams, fields, grades, nihc_field):
        """Line 98: full key unpacking when var is truthy."""
        data = {'teams': teams, 'fields': fields, 'grades': grades}
        key = ('Norths PHL', 'Tigers PHL', 'PHL', 'Sunday', 1, '11:30',
               1, '2026-03-21', 1, 'EF', 'Newcastle International Hockey Centre')
        X = {key: 1}
        roster = convert_X_to_roster(X, data)
        assert len(roster.weeks) == 1
        assert len(roster.weeks[0].games) == 1
        game = roster.weeks[0].games[0]
        assert game.team1 == 'Norths PHL'
        assert game.team2 == 'Tigers PHL'

    def test_multiple_weeks_with_byes(self, teams, fields, grades, nihc_field):
        """Cover bye_teams calculation and multi-week grouping."""
        data = {'teams': teams, 'fields': fields, 'grades': grades}
        key1 = ('Norths PHL', 'Tigers PHL', 'PHL', 'Sunday', 1, '11:30',
                1, '2026-03-21', 1, 'EF', 'Newcastle International Hockey Centre')
        key2 = ('Maitland PHL', 'Gosford PHL', 'PHL', 'Sunday', 2, '13:00',
                2, '2026-03-28', 2, 'EF', 'Newcastle International Hockey Centre')
        X = {key1: 1, key2: 1}
        roster = convert_X_to_roster(X, data)
        assert len(roster.weeks) == 2
        # Week 1: only Norths/Tigers play, others are byes
        w1 = roster.weeks[0]
        assert len(w1.bye_teams) > 0
        assert 'Maitland PHL' in w1.bye_teams

    def test_mixed_true_false_vars(self, teams, fields, grades):
        """Mix of truthy and falsy vars."""
        data = {'teams': teams, 'fields': fields, 'grades': grades}
        key_true = ('Norths PHL', 'Tigers PHL', 'PHL', 'Sunday', 1, '11:30',
                    1, '2026-03-21', 1, 'EF', 'Newcastle International Hockey Centre')
        key_false = ('Maitland PHL', 'Gosford PHL', 'PHL', 'Sunday', 2, '13:00',
                     1, '2026-03-21', 1, 'EF', 'Newcastle International Hockey Centre')
        X = {key_true: 1, key_false: 0}
        roster = convert_X_to_roster(X, data)
        assert len(roster.weeks) == 1
        assert len(roster.weeks[0].games) == 1


# ============== circle_method_round_1_pairings ==============

class TestCircleMethodPairings:
    """Cover lines 348-384 (entire function)."""

    def test_even_teams(self):
        teams = {'PHL': ['A', 'B', 'C', 'D']}
        result = circle_method_round_1_pairings(teams)
        assert 'PHL' in result
        pairings = result['PHL']
        assert len(pairings) == 2
        # Each pairing is a sorted tuple
        for pair in pairings:
            assert pair[0] < pair[1]

    def test_odd_teams(self):
        """Odd teams use ghost/bye at position 0, one team gets a bye."""
        teams = {'3rd': ['A', 'B', 'C']}
        result = circle_method_round_1_pairings(teams)
        pairings = result['3rd']
        # 3 teams, 1 ghost: 2 positions paired, but ghost pairing is excluded
        assert len(pairings) == 1

    def test_single_team(self):
        """Less than 2 teams returns empty list."""
        teams = {'5th': ['Solo']}
        result = circle_method_round_1_pairings(teams)
        assert result['5th'] == []

    def test_two_teams(self):
        teams = {'6th': ['X', 'Y']}
        result = circle_method_round_1_pairings(teams)
        assert result['6th'] == [('X', 'Y')]

    def test_multiple_grades(self):
        teams = {
            'PHL': ['A', 'B', 'C', 'D'],
            '2nd': ['E', 'F', 'G'],
        }
        result = circle_method_round_1_pairings(teams)
        assert 'PHL' in result
        assert '2nd' in result

    def test_empty_dict(self):
        result = circle_method_round_1_pairings({})
        assert result == {}

    def test_five_teams_odd(self):
        """5 teams -> ghost + 5 = 6 positions, 3 pairings but 1 involves ghost."""
        teams = {'PHL': ['A', 'B', 'C', 'D', 'E']}
        result = circle_method_round_1_pairings(teams)
        pairings = result['PHL']
        assert len(pairings) == 2  # 3 pairings minus 1 ghost pairing


# ============== generate_X ==============

class TestGenerateX:
    """Cover lines 433-517 (main body), 599, 604-606, 629, 640-650, 684-686,
    699-700, 756-862, 866-870, 887-905, 970."""

    def test_basic_variable_creation(self, base_data):
        """Basic test: generate_X creates variables."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        # Without phl_game_times, PHL vars will be skipped (no valid slots)
        # So set empty phl_game_times = all PHL vars skipped
        X, conflicts = generate_X(model, base_data)
        # Should create some variables (at least for 3rd grade on Sunday)
        assert len(X) > 0

    def test_phl_filtering_nested_format(self, base_data, nihc_field):
        """Lines 802-808: nested format PHL_GAME_TIMES filters PHL variables."""
        from ortools.sat.python import cp_model
        from datetime import time as tm
        model = cp_model.CpModel()
        base_data['phl_game_times'] = {
            'Newcastle International Hockey Centre': {
                'EF': {
                    'Sunday': [tm(11, 30)],
                }
            }
        }
        X, _ = generate_X(model, base_data)
        # PHL vars should only exist for EF Sunday 11:30 at NIHC
        phl_keys = [k for k in X if len(k) >= 11 and k[2] == 'PHL']
        for k in phl_keys:
            assert k[5] == '11:30'  # time
            assert k[9] == 'EF'  # field_name
            assert k[10] == 'Newcastle International Hockey Centre'

    def test_phl_filtering_simple_format(self, base_data):
        """Lines 794-800: simple (2025) format PHL_GAME_TIMES."""
        from ortools.sat.python import cp_model
        from datetime import time as tm
        model = cp_model.CpModel()
        base_data['phl_game_times'] = {
            'Newcastle International Hockey Centre': {
                'Sunday': [tm(11, 30)],
            }
        }
        X, _ = generate_X(model, base_data)
        phl_keys = [k for k in X if len(k) >= 11 and k[2] == 'PHL']
        for k in phl_keys:
            assert k[5] == '11:30'
            assert k[10] == 'Newcastle International Hockey Centre'

    def test_second_grade_filtering(self, base_data):
        """Lines 810-819, 910-914: second_grade_times filters 2nd grade vars."""
        from ortools.sat.python import cp_model
        from datetime import time as tm
        model = cp_model.CpModel()
        base_data['second_grade_times'] = {
            'Newcastle International Hockey Centre': {
                'WF': {
                    'Sunday': [tm(13, 0)],
                }
            }
        }
        X, _ = generate_X(model, base_data)
        second_keys = [k for k in X if len(k) >= 11 and k[2] == '2nd']
        for k in second_keys:
            assert k[5] == '13:00'
            assert k[9] == 'WF'

    def test_lower_grades_excluded_from_gosford(self, base_data):
        """Lines 858-861, 920-922: 3rd grade cannot play at Gosford."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        X, _ = generate_X(model, base_data)
        third_keys = [k for k in X if len(k) >= 11 and k[2] == '3rd']
        for k in third_keys:
            assert k[10] != 'Central Coast Hockey Park'

    def test_lower_grades_excluded_from_friday(self, base_data):
        """Lines 861, 923-925: 3rd grade cannot play on Friday."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        X, _ = generate_X(model, base_data)
        third_keys = [k for k in X if len(k) >= 11 and k[2] == '3rd']
        for k in third_keys:
            assert k[3] != 'Friday'

    def test_home_field_map_filtering(self, base_data):
        """Lines 866-870, 931-937: games at Maitland Park must involve Maitland."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        base_data['home_field_map'] = {
            'Maitland': 'Maitland Park',
            'Gosford': 'Central Coast Hockey Park',
        }
        X, _ = generate_X(model, base_data)
        for k in X:
            if len(k) < 11:
                continue
            venue = k[10]
            t1, t2 = k[0], k[1]
            if venue == 'Maitland Park':
                assert 'Maitland' in t1 or 'Maitland' in t2, \
                    f"Game at Maitland Park without Maitland team: {t1} vs {t2}"
            if venue == 'Central Coast Hockey Park':
                assert 'Gosford' in t1 or 'Gosford' in t2, \
                    f"Game at Gosford without Gosford team: {t1} vs {t2}"

    def test_forced_games(self, base_data):
        """Lines 943-951: forced game rules track matching vars."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        base_data['forced_games'] = [
            {
                'teams': ['Norths PHL', 'Tigers PHL'],
                'grade': 'PHL',
                'week': 1,
                'description': 'Test forced game',
            }
        ]
        # Need PHL game times so PHL vars exist
        from datetime import time as tm
        base_data['phl_game_times'] = {
            'Newcastle International Hockey Centre': {
                'EF': {'Sunday': [tm(11, 30)]},
            }
        }
        X, _ = generate_X(model, base_data)
        # The forced game vars should be in X
        forced_keys = [k for k in X if len(k) >= 11 and k[0] == 'Norths PHL'
                       and k[1] == 'Tigers PHL' and k[6] == 1]
        assert len(forced_keys) > 0

    def test_blocked_games(self, base_data):
        """Lines 954-956: blocked game rules eliminate vars."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        base_data['blocked_games'] = [
            {
                'teams': ['Norths 3rd'],
                'date': '2026-03-21',
                'description': 'Block Norths 3rd on week 1',
            }
        ]
        X, _ = generate_X(model, base_data)
        # No Norths 3rd game on date 2026-03-21
        blocked = [k for k in X if len(k) >= 11 and
                   ('Norths 3rd' in (k[0], k[1])) and k[7] == '2026-03-21']
        assert len(blocked) == 0

    def test_skip_no_day_timeslots(self, base_data, nihc_field):
        """Line 890-891: timeslots with no day are skipped."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        # Add a timeslot with empty day
        empty_day_ts = Timeslot(date='2026-03-21', day='', time='11:30',
                            week=1, day_slot=1, field=nihc_field, round_no=1)
        base_data['timeslots'].append(empty_day_ts)
        X, _ = generate_X(model, base_data)
        # Should still work without error; empty-day ts skipped
        assert isinstance(X, dict)


# ============== _build_forced_game_rules ==============

class TestBuildForcedGameRules:
    """Cover lines 433-540."""

    def test_empty_forced_games(self, teams):
        result, _, _ = _build_forced_game_rules([], teams)
        assert result == {}

    def test_none_forced_games(self, teams):
        result, _, _ = _build_forced_game_rules(None, teams)
        assert result == {}

    def test_pair_forced_game(self, teams):
        """Two-team forced game creates pair matcher."""
        entries = [{'teams': ['Norths PHL', 'Tigers PHL'], 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        assert len(result) == 1
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert len(matchers) == 1
        assert matchers[0][0] == 'pair'

    def test_single_team_forced_game(self, teams):
        """Single-team forced game creates 'any' matcher."""
        entries = [{'teams': ['Norths PHL'], 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert matchers[0][0] == 'any'
        assert matchers[0][1] == 'Norths PHL'

    def test_club_name_resolution(self, teams):
        """Club name resolves to full team name with grade."""
        entries = [{'teams': ['Norths'], 'grade': 'PHL', 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert any(m[1] == 'Norths PHL' for m in matchers)

    def test_club_name_without_grade(self, teams):
        """Club name without grade resolves to all club teams."""
        entries = [{'teams': ['Norths'], 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        # Norths has PHL, 2nd, 3rd teams
        team_names = {m[1] for m in matchers}
        assert 'Norths PHL' in team_names
        assert 'Norths 2nd' in team_names
        assert 'Norths 3rd' in team_names

    def test_team1_team2_keys(self, teams):
        """Lines 515-530: team1/team2 keys instead of 'teams'."""
        entries = [{'team1': 'Norths PHL', 'team2': 'Tigers PHL', 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert matchers[0][0] == 'pair'

    def test_team1_only(self, teams):
        """Line 525-527: only team1 specified."""
        entries = [{'team1': 'Norths PHL', 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert matchers[0][0] == 'any'
        assert matchers[0][1] == 'Norths PHL'

    def test_team2_only(self, teams):
        """Line 528-530: only team2 specified."""
        entries = [{'team2': 'Tigers PHL', 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert matchers[0][0] == 'any'
        assert matchers[0][1] == 'Tigers PHL'

    def test_list_scope_value(self, teams):
        """Lines 494-496: scope field with list value (converted to tuple)."""
        entries = [{'teams': ['Norths PHL'], 'week': [1, 2]}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        # The week should be stored as tuple in scope
        for idx, val in scope_key:
            if idx == 6:  # week index
                assert val == (1, 2)

    def test_grade_list_resolution(self, teams):
        """Lines 473-477: grade as list resolves for each grade."""
        entries = [{'teams': ['Norths'], 'grade': ['PHL', '2nd'], 'week': 1}]
        result, _, _ = _build_forced_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        names = {m[1] for m in matchers}
        assert 'Norths PHL' in names
        assert 'Norths 2nd' in names


# ============== _check_forced_game_status ==============

class TestCheckForcedGameStatus:
    """Cover lines 543-590."""

    def test_no_rules_returns_normal(self):
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        status, scope = _check_forced_game_status(key, {})
        assert status == 'normal'
        assert scope is None

    def test_matching_pair_returns_force(self):
        scope_key = frozenset([(6, 1)])  # week=1
        rules = {scope_key: [('pair', 'A', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        status, scope = _check_forced_game_status(key, rules)
        assert status == 'force'
        assert scope == scope_key

    def test_matching_any_returns_force(self):
        scope_key = frozenset([(6, 1)])  # week=1
        rules = {scope_key: [('any', 'A')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        status, scope = _check_forced_game_status(key, rules)
        assert status == 'force'

    def test_any_matches_team2(self):
        scope_key = frozenset([(6, 1)])
        rules = {scope_key: [('any', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        status, _ = _check_forced_game_status(key, rules)
        assert status == 'force'

    def test_scope_mismatch_returns_normal(self):
        scope_key = frozenset([(6, 2)])  # week=2
        rules = {scope_key: [('pair', 'A', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        status, _ = _check_forced_game_status(key, rules)
        assert status == 'normal'

    def test_team_mismatch_in_scope_returns_normal(self):
        """In scope but teams don't match -> normal (not eliminated)."""
        scope_key = frozenset([(6, 1)])  # week=1
        rules = {scope_key: [('pair', 'C', 'D')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        status, _ = _check_forced_game_status(key, rules)
        assert status == 'normal'

    def test_list_scope_match(self):
        """Lines 564-567: tuple val match (any-of)."""
        scope_key = frozenset([(6, (1, 2))])  # week in [1, 2]
        rules = {scope_key: [('pair', 'A', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 2, '2026-03-28', 2, 'EF', 'NIHC')
        status, _ = _check_forced_game_status(key, rules)
        assert status == 'force'

    def test_list_scope_no_match(self):
        scope_key = frozenset([(6, (1, 2))])  # week in [1, 2]
        rules = {scope_key: [('pair', 'A', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 3, '2026-04-04', 3, 'EF', 'NIHC')
        status, _ = _check_forced_game_status(key, rules)
        assert status == 'normal'


# ============== _get_matching_forced_scopes (multi-scope match) ==============

class TestGetMatchingForcedScopes:
    """A variable that matches multiple forced scopes must be registered
    against EVERY matching scope, not just the first one. Otherwise the
    sum constraints operate on disjoint buckets and the solver loses
    flexibility (e.g. overlap between a date-scope `('all',)` and a
    team-scope `('pair', X, Y)` would falsely require two distinct games)."""

    def test_no_match_returns_empty(self):
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _get_matching_forced_scopes(key, {}) == []

    def test_single_scope_match(self):
        scope = frozenset([(6, 1)])  # week=1
        rules = {scope: [('pair', 'A', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _get_matching_forced_scopes(key, rules) == [scope]

    def test_overlapping_date_and_team_scopes_both_match(self):
        """Regression: Norths-Gosford-Apr-17-CCHP-Friday must register
        against BOTH the date-scope (Apr 17 at CCHP, all-matcher) and the
        team-scope (Norths-Gosford on any Friday at CCHP, pair-matcher)."""
        date_scope = frozenset([
            (2, 'PHL'),                              # grade
            (3, 'Friday'),                           # day
            (7, '2026-04-17'),                       # date
            (10, 'Central Coast Hockey Park'),       # field_location
        ])
        team_scope = frozenset([
            (2, 'PHL'),
            (3, 'Friday'),
            (10, 'Central Coast Hockey Park'),
            ('_entry_idx', 99),                      # team-specific entries get an entry-idx
        ])
        rules = {
            date_scope: [('all',)],
            team_scope: [('pair', 'Gosford PHL', 'Norths PHL')],
        }
        key = ('Gosford PHL', 'Norths PHL', 'PHL', 'Friday', 1, '20:00',
               4, '2026-04-17', 4, 'Wyong Main Field', 'Central Coast Hockey Park')
        matches = _get_matching_forced_scopes(key, rules)
        assert set(matches) == {date_scope, team_scope}, (
            f"Expected variable to match both scopes, got {matches}"
        )

    def test_team_scope_only_when_date_differs(self):
        """If date doesn't match the date-scope, only the team-scope matches."""
        date_scope = frozenset([
            (2, 'PHL'), (3, 'Friday'),
            (7, '2026-04-17'), (10, 'Central Coast Hockey Park'),
        ])
        team_scope = frozenset([
            (2, 'PHL'), (3, 'Friday'),
            (10, 'Central Coast Hockey Park'),
            ('_entry_idx', 99),
        ])
        rules = {
            date_scope: [('all',)],
            team_scope: [('pair', 'Gosford PHL', 'Norths PHL')],
        }
        # Norths-Gosford on Jun 26 (a different Friday)
        key = ('Gosford PHL', 'Norths PHL', 'PHL', 'Friday', 1, '20:00',
               14, '2026-06-26', 14, 'Wyong Main Field', 'Central Coast Hockey Park')
        assert _get_matching_forced_scopes(key, rules) == [team_scope]

    def test_date_scope_only_when_pair_differs(self):
        """If teams don't match the team-scope, only the date-scope matches."""
        date_scope = frozenset([
            (2, 'PHL'), (3, 'Friday'),
            (7, '2026-04-17'), (10, 'Central Coast Hockey Park'),
        ])
        team_scope = frozenset([
            (2, 'PHL'), (3, 'Friday'),
            (10, 'Central Coast Hockey Park'),
            ('_entry_idx', 99),
        ])
        rules = {
            date_scope: [('all',)],
            team_scope: [('pair', 'Gosford PHL', 'Norths PHL')],
        }
        # Tigers-Gosford on Apr 17 (date matches, team-pair doesn't)
        key = ('Gosford PHL', 'Tigers PHL', 'PHL', 'Friday', 1, '20:00',
               4, '2026-04-17', 4, 'Wyong Main Field', 'Central Coast Hockey Park')
        assert _get_matching_forced_scopes(key, rules) == [date_scope]


# ============== _build_blocked_game_rules ==============

class TestBuildBlockedGameRules:
    """Cover lines 598-683."""

    def test_empty_returns_empty(self, teams):
        assert _build_blocked_game_rules([], teams) == {}

    def test_none_returns_empty(self, teams):
        assert _build_blocked_game_rules(None, teams) == {}

    def test_club_key_blocked(self, teams):
        """Lines 674-677: 'club' key in blocked entry."""
        entries = [{'club': 'Norths', 'date': '2026-03-21'}]
        result = _build_blocked_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        names = {m[1] for m in matchers}
        assert 'Norths PHL' in names
        assert 'Norths 2nd' in names

    def test_grades_list_blocked(self, teams):
        """Lines 642-643: 'grades' list instead of 'grade'."""
        entries = [{'club': 'Norths', 'grades': ['PHL', '2nd'], 'date': '2026-03-21'}]
        result = _build_blocked_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        names = {m[1] for m in matchers}
        assert 'Norths PHL' in names
        assert 'Norths 2nd' in names

    def test_two_team_pair_blocked(self, teams):
        entries = [{'teams': ['Norths PHL', 'Tigers PHL'], 'date': '2026-03-21'}]
        result = _build_blocked_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert matchers[0][0] == 'pair'

    def test_single_team_blocked(self, teams):
        entries = [{'teams': ['Norths PHL'], 'date': '2026-03-21'}]
        result = _build_blocked_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        matchers = result[scope_key]
        assert matchers[0][0] == 'any'

    def test_list_scope_value(self, teams):
        """Scope with list val (e.g. date list)."""
        entries = [{'teams': ['Norths PHL'], 'week': [1, 2]}]
        result = _build_blocked_game_rules(entries, teams)
        scope_key = list(result.keys())[0]
        for idx, val in scope_key:
            if idx == 6:
                assert val == (1, 2)

    def test_reason_field_in_description(self, teams):
        """Line 679: 'reason' used as fallback for description."""
        entries = [{'teams': ['Norths PHL'], 'date': '2026-03-21', 'reason': 'Holiday'}]
        # Should not raise -- reason is used in print output
        result = _build_blocked_game_rules(entries, teams)
        assert len(result) == 1


# ============== _is_blocked_by_no_play ==============

class TestIsBlockedByNoPlay:
    """Cover lines 686-729."""

    def test_empty_rules(self):
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, {}) is False

    def test_pair_match_blocked(self):
        scope_key = frozenset([(7, '2026-03-21')])
        rules = {scope_key: [('pair', 'A', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, rules) is True

    def test_any_match_team1_blocked(self):
        scope_key = frozenset([(7, '2026-03-21')])
        rules = {scope_key: [('any', 'A')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, rules) is True

    def test_any_match_team2_blocked(self):
        scope_key = frozenset([(7, '2026-03-21')])
        rules = {scope_key: [('any', 'B')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, rules) is True

    def test_scope_mismatch_not_blocked(self):
        scope_key = frozenset([(7, '2026-04-01')])
        rules = {scope_key: [('any', 'A')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 1, '2026-03-21', 1, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, rules) is False

    def test_tuple_scope_match(self):
        """Tuple (any-of) scope matching."""
        scope_key = frozenset([(6, (1, 2))])
        rules = {scope_key: [('any', 'A')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 2, '2026-03-28', 2, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, rules) is True

    def test_tuple_scope_no_match(self):
        scope_key = frozenset([(6, (1, 2))])
        rules = {scope_key: [('any', 'A')]}
        key = ('A', 'B', 'PHL', 'Sunday', 1, '11:30', 3, '2026-04-04', 3, 'EF', 'NIHC')
        assert _is_blocked_by_no_play(key, rules) is False


# ============== max_games_per_grade ==============

class TestMaxGamesPerGradeCoverage:
    """Cover lines 302-304, 348-384."""

    def test_none_defaults(self):
        """Lines 302-304: None params default to empty dicts."""
        grades = [Grade(name='PHL', teams=['A', 'B', 'C', 'D'])]
        result = max_games_per_grade(grades, 20,
                                     max_weekends_per_grade=None,
                                     grade_rounds_override=None,
                                     grade_scheduling_method=None)
        assert 'PHL' in result
        assert result['PHL'] > 0

    def test_single_team_grade(self):
        """Line 307-309: grade with < 2 teams gets 0 games."""
        grades = [Grade(name='6th', teams=['Solo'])]
        result = max_games_per_grade(grades, 20)
        assert result['6th'] == 0

    def test_exact_override(self):
        """Lines 312-314: grade_rounds_override bypasses calculation."""
        grades = [Grade(name='2nd', teams=['A', 'B', 'C', 'D'])]
        result = max_games_per_grade(grades, 20, grade_rounds_override={'2nd': 18})
        assert result['2nd'] == 18

    def test_max_weekends_override(self):
        """Lines 317: max_weekends_per_grade overrides max_rounds for a grade."""
        grades = [Grade(name='PHL', teams=['A', 'B', 'C', 'D', 'E', 'F'])]
        result = max_games_per_grade(grades, 20, max_weekends_per_grade={'PHL': 22})
        assert result['PHL'] == 20  # 22 weekends, 6 teams -> 20 (multiple of 5)

    def test_odd_team_count_forces_even(self):
        """Lines 328-329: odd T forces g0 even."""
        grades = [Grade(name='3rd', teams=['A', 'B', 'C', 'D', 'E'])]
        result = max_games_per_grade(grades, 20)
        assert result['3rd'] % 2 == 0

    def test_method_2_maximize(self):
        """Lines 331-338: method 2 skips round-robin rounding."""
        grades = [Grade(name='PHL', teams=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])]
        result_m1 = max_games_per_grade(grades, 20, grade_scheduling_method={'PHL': 1})
        result_m2 = max_games_per_grade(grades, 20, grade_scheduling_method={'PHL': 2})
        # Method 2 should give >= method 1 (no rounding down)
        assert result_m2['PHL'] >= result_m1['PHL']


# ============== generate_games ==============

class TestGenerateGamesCoverage:
    """Cover line 416-421."""

    def test_same_grade_only(self, teams):
        games = generate_games(teams)
        for key in games:
            t1, t2, grade = key
            # Both teams should be in same grade
            t1_grade = next(t.grade for t in teams if t.name == t1)
            t2_grade = next(t.grade for t in teams if t.name == t2)
            assert t1_grade == t2_grade == grade

    def test_key_ordering(self, teams):
        games = generate_games(teams)
        for key in games:
            t1, t2, _ = key
            assert t1 < t2  # Alphabetically ordered


# ============== generate_timeslots edge cases ==============

class TestGenerateTimeslotsCoverage:
    """Cover lines 195, 303-304."""

    def test_field_unavailability_unknown_field_raises(self):
        """Line 193-194: unknown field in field_unavailabilities raises ValueError."""
        from datetime import datetime
        fields = [PlayingField(name='EF', location='NIHC')]
        with pytest.raises(ValueError, match="does not exist"):
            generate_timeslots(
                start_date=datetime(2026, 3, 20),
                end_date=datetime(2026, 3, 22),
                day_time_map={'NIHC': {'Sunday': []}},
                fields=fields,
                field_unavailabilities={'Unknown Venue': {'weekends': []}},
            )

    def test_day_time_map_unknown_field_raises(self):
        """Line 197-198: unknown field in day_time_map raises ValueError."""
        from datetime import datetime
        fields = [PlayingField(name='EF', location='NIHC')]
        with pytest.raises(ValueError, match="does not exist"):
            generate_timeslots(
                start_date=datetime(2026, 3, 20),
                end_date=datetime(2026, 3, 22),
                day_time_map={'Bad Venue': {'Sunday': []}},
                fields=fields,
                field_unavailabilities={},
            )


# ============== build_season_data (integration test with real config) ==============

class TestBuildSeasonDataIntegration:
    """Cover lines 1001-1002, 1030+ using real 2026 config."""

    def test_loads_real_2026_config(self):
        """Integration: load real config and validate structure."""
        from config import load_season_data
        data = load_season_data(2026)
        assert 'teams' in data
        assert 'grades' in data
        assert 'timeslots' in data
        assert 'fields' in data
        assert len(data['teams']) > 30
        assert len(data['grades']) >= 5

    def test_real_config_num_rounds(self):
        """Verify num_rounds calculated correctly for real config."""
        from config import load_season_data
        data = load_season_data(2026)
        num_rounds = data['num_rounds']
        assert num_rounds['PHL'] == 20
        assert num_rounds['2nd'] == 18

    def test_real_config_timeslots_generated(self):
        """Verify timeslots include Fridays and Sundays."""
        from config import load_season_data
        data = load_season_data(2026)
        days = {t.day for t in data['timeslots']}
        assert 'Sunday' in days
        assert 'Friday' in days

    def test_real_generate_x(self):
        """Integration: generate_X with real 2026 config data."""
        from config import load_season_data
        from ortools.sat.python import cp_model
        data = load_season_data(2026)
        model = cp_model.CpModel()
        X, conflicts = generate_X(model, data)
        assert len(X) > 10000  # Should be many variables
        # PHL should not be at South Field
        phl_keys = [k for k in X if len(k) >= 11 and k[2] == 'PHL']
        for k in phl_keys:
            assert k[9] != 'SF', "PHL should not have South Field variables"

    def test_real_home_field_filtering(self):
        """Integration: verify home_field_map filtering with real config."""
        from config import load_season_data
        from ortools.sat.python import cp_model
        data = load_season_data(2026)
        model = cp_model.CpModel()
        X, _ = generate_X(model, data)
        # Games at Maitland Park must involve a Maitland team
        team_to_club = {t.name: t.club.name for t in data['teams']}
        for k in X:
            if len(k) < 11:
                continue
            if k[10] == 'Maitland Park':
                t1_club = team_to_club[k[0]]
                t2_club = team_to_club[k[1]]
                assert t1_club == 'Maitland' or t2_club == 'Maitland'


# ============== generate_X with games pre-populated ==============

class TestGenerateXWithPrePopulatedGames:
    """Cover lines 822-829: games already in data dict."""

    def test_games_already_in_data(self, base_data):
        """Line 825-826: games already populated as dict."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        games = generate_games(base_data['teams'])
        base_data['games'] = games
        X, _ = generate_X(model, base_data)
        assert len(X) > 0

    def test_games_as_list_in_data(self, base_data):
        """Lines 827-829: games as list format converted to dict."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        games = generate_games(base_data['teams'])
        base_data['games'] = list(games.keys())
        X, _ = generate_X(model, base_data)
        assert len(X) > 0
