# test_utils.py
"""
Unit tests for utility functions in utils.py.

These tests verify the helper functions used across the scheduling system.
"""

import pytest
import sys
import os
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot, Game, WeeklyDraw, Roster
from utils import (
    get_club,
    get_club_object,
    get_teams_from_club,
    get_club_from_clubname,
    get_duplicated_graded_teams,
    split_number_suffix,
    add_ordinal_suffix,
    get_round_number_for_week,
    get_nearest_week_by_date,
    get_field_by_name,
    get_grade_by_name,
    convert_X_to_roster,
    export_roster_to_excel,
    _build_blocked_game_rules,
    _is_blocked_by_no_play,
)


# ============== Fixtures ==============

@pytest.fixture
def sample_clubs():
    """Create sample clubs for testing."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def sample_teams(sample_clubs):
    """Create sample teams for testing."""
    tigers, wests, maitland = sample_clubs
    return [
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Tigers 2nd', club=tigers, grade='2nd'),
        Team(name='Tigers 3rd', club=tigers, grade='3rd'),
        Team(name='Tigers 3rd B', club=tigers, grade='3rd'),  # Duplicate grade team
        Team(name='Wests PHL', club=wests, grade='PHL'),
        Team(name='Wests 2nd', club=wests, grade='2nd'),
        Team(name='Wests 3rd', club=wests, grade='3rd'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
        Team(name='Maitland 2nd', club=maitland, grade='2nd'),
    ]


@pytest.fixture
def sample_fields():
    """Create sample playing fields."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Main'),
    ]


@pytest.fixture
def sample_grades():
    """Create sample grades."""
    return [
        Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL', 'Maitland PHL']),
        Grade(name='2nd', teams=['Tigers 2nd', 'Wests 2nd', 'Maitland 2nd']),
        Grade(name='3rd', teams=['Tigers 3rd', 'Tigers 3rd B', 'Wests 3rd']),
    ]


@pytest.fixture
def sample_timeslots(sample_fields):
    """Create sample timeslots across multiple weeks."""
    ef = sample_fields[0]
    wf = sample_fields[1]
    return [
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='11:30', week=1, day_slot=2, field=ef, round_no=1),
        Timeslot(date='2025-03-23', day='Sunday', time='10:00', week=1, day_slot=1, field=wf, round_no=1),
        Timeslot(date='2025-03-30', day='Sunday', time='10:00', week=2, day_slot=1, field=ef, round_no=2),
        Timeslot(date='2025-04-06', day='Sunday', time='10:00', week=3, day_slot=1, field=ef, round_no=3),
    ]


# ============== get_club Tests ==============

class TestGetClub:
    """Tests for get_club function."""

    def test_returns_club_name(self, sample_teams):
        """Test that get_club returns the correct club name."""
        assert get_club('Tigers PHL', sample_teams) == 'Tigers'
        assert get_club('Wests 3rd', sample_teams) == 'Wests'
        assert get_club('Maitland 2nd', sample_teams) == 'Maitland'

    def test_raises_error_for_unknown_team(self, sample_teams):
        """Test that get_club raises ValueError for unknown team."""
        with pytest.raises(ValueError, match="not found"):
            get_club('Unknown Team', sample_teams)


# ============== get_club_object Tests ==============

class TestGetClubObject:
    """Tests for get_club_object function."""

    def test_returns_club_object(self, sample_teams, sample_clubs):
        """Test that get_club_object returns the actual Club object."""
        club = get_club_object('Tigers PHL', sample_teams)
        assert isinstance(club, Club)
        assert club.name == 'Tigers'
        assert club.home_field == 'Newcastle International Hockey Centre'

    def test_raises_error_for_unknown_team(self, sample_teams):
        """Test that get_club_object raises ValueError for unknown team."""
        with pytest.raises(ValueError, match="not found"):
            get_club_object('Unknown Team', sample_teams)


# ============== get_teams_from_club Tests ==============

class TestGetTeamsFromClub:
    """Tests for get_teams_from_club function."""

    def test_returns_all_teams_from_club(self, sample_teams):
        """Test that get_teams_from_club returns all teams in a club."""
        tigers_teams = get_teams_from_club('Tigers', sample_teams)
        assert len(tigers_teams) == 4  # PHL, 2nd, 3rd, 3rd B
        assert 'Tigers PHL' in tigers_teams
        assert 'Tigers 2nd' in tigers_teams
        assert 'Tigers 3rd' in tigers_teams
        assert 'Tigers 3rd B' in tigers_teams

    def test_returns_empty_for_nonexistent_club(self, sample_teams):
        """Test that get_teams_from_club returns empty list for unknown club."""
        teams = get_teams_from_club('Unknown Club', sample_teams)
        assert teams == []


# ============== get_club_from_clubname Tests ==============

class TestGetClubFromClubname:
    """Tests for get_club_from_clubname function."""

    def test_returns_club_object(self, sample_clubs):
        """Test that get_club_from_clubname returns Club object."""
        club = get_club_from_clubname('Tigers', sample_clubs)
        assert isinstance(club, Club)
        assert club.name == 'Tigers'

    def test_raises_error_for_unknown_club(self, sample_clubs):
        """Test that get_club_from_clubname raises ValueError for unknown club."""
        with pytest.raises(ValueError, match="not found"):
            get_club_from_clubname('Unknown Club', sample_clubs)


# ============== get_duplicated_graded_teams Tests ==============

class TestGetDuplicatedGradedTeams:
    """Tests for get_duplicated_graded_teams function."""

    def test_returns_teams_in_same_grade(self, sample_teams):
        """Test that get_duplicated_graded_teams returns teams in same grade."""
        dup_teams = get_duplicated_graded_teams('Tigers', '3rd', sample_teams)
        assert len(dup_teams) == 2
        assert 'Tigers 3rd' in dup_teams
        assert 'Tigers 3rd B' in dup_teams

    def test_returns_single_team_when_no_duplicates(self, sample_teams):
        """Test single team when no duplicates exist."""
        teams = get_duplicated_graded_teams('Tigers', 'PHL', sample_teams)
        assert len(teams) == 1
        assert 'Tigers PHL' in teams

    def test_returns_empty_for_no_teams(self, sample_teams):
        """Test empty list when no teams match."""
        teams = get_duplicated_graded_teams('Unknown', '3rd', sample_teams)
        assert teams == []


# ============== split_number_suffix Tests ==============

class TestSplitNumberSuffix:
    """Tests for split_number_suffix function."""

    def test_splits_number_and_suffix(self):
        """Test splitting '3rd' into ('3', 'rd')."""
        number, suffix = split_number_suffix('3rd')
        assert number == '3'
        assert suffix == 'rd'

    def test_handles_two_digit_number(self):
        """Test splitting '11th' into ('11', 'th')."""
        number, suffix = split_number_suffix('11th')
        assert number == '11'
        assert suffix == 'th'

    def test_returns_original_for_no_match(self):
        """Test that non-matching text returns unchanged."""
        result, suffix = split_number_suffix('PHL')
        assert result == 'PHL'
        assert suffix == ''


# ============== add_ordinal_suffix Tests ==============

class TestAddOrdinalSuffix:
    """Tests for add_ordinal_suffix function."""

    def test_first(self):
        assert add_ordinal_suffix(1) == '1st'

    def test_second(self):
        assert add_ordinal_suffix(2) == '2nd'

    def test_third(self):
        assert add_ordinal_suffix(3) == '3rd'

    def test_fourth(self):
        assert add_ordinal_suffix(4) == '4th'

    def test_eleventh(self):
        """Test special case for 11th."""
        assert add_ordinal_suffix(11) == '11th'

    def test_twelfth(self):
        """Test special case for 12th."""
        assert add_ordinal_suffix(12) == '12th'

    def test_thirteenth(self):
        """Test special case for 13th."""
        assert add_ordinal_suffix(13) == '13th'

    def test_twenty_first(self):
        assert add_ordinal_suffix(21) == '21st'


# ============== get_round_number_for_week Tests ==============

class TestGetRoundNumberForWeek:
    """Tests for get_round_number_for_week function."""

    def test_returns_correct_round(self, sample_timeslots):
        """Test that correct round number is returned for week."""
        assert get_round_number_for_week(1, sample_timeslots) == 1
        assert get_round_number_for_week(2, sample_timeslots) == 2
        assert get_round_number_for_week(3, sample_timeslots) == 3

    def test_raises_error_for_missing_week(self, sample_timeslots):
        """Test ValueError for non-existent week."""
        with pytest.raises(ValueError, match="not found"):
            get_round_number_for_week(99, sample_timeslots)


# ============== get_nearest_week_by_date Tests ==============

class TestGetNearestWeekByDate:
    """Tests for get_nearest_week_by_date function."""

    def test_returns_exact_week(self, sample_timeslots):
        """Test exact date match returns correct week."""
        assert get_nearest_week_by_date('2025-03-23', sample_timeslots) == 1
        assert get_nearest_week_by_date('2025-03-30', sample_timeslots) == 2

    def test_returns_nearest_week(self, sample_timeslots):
        """Test that nearest week is returned for in-between dates."""
        # Date between week 1 (March 23) and week 2 (March 30)
        # March 25 is closer to March 23
        assert get_nearest_week_by_date('2025-03-25', sample_timeslots) == 1
        # March 28 is closer to March 30
        assert get_nearest_week_by_date('2025-03-28', sample_timeslots) == 2


# ============== get_field_by_name Tests ==============

class TestGetFieldByName:
    """Tests for get_field_by_name function."""

    def test_returns_correct_field(self, sample_fields):
        """Test that correct field is returned."""
        field = get_field_by_name('EF', sample_fields)
        assert field.name == 'EF'
        assert field.location == 'Newcastle International Hockey Centre'

    def test_raises_error_for_unknown_field(self, sample_fields):
        """Test ValueError for non-existent field."""
        with pytest.raises(ValueError, match="not found"):
            get_field_by_name('Unknown Field', sample_fields)


# ============== get_grade_by_name Tests ==============

class TestGetGradeByName:
    """Tests for get_grade_by_name function."""

    def test_returns_correct_grade(self, sample_grades):
        """Test that correct grade is returned."""
        grade = get_grade_by_name('PHL', sample_grades)
        assert grade.name == 'PHL'
        assert 'Tigers PHL' in grade.teams

    def test_raises_error_for_unknown_grade(self, sample_grades):
        """Test ValueError for non-existent grade."""
        with pytest.raises(ValueError, match="not found"):
            get_grade_by_name('Unknown Grade', sample_grades)


# ============== convert_X_to_roster Tests ==============

class TestConvertXToRoster:
    """Tests for convert_X_to_roster function."""

    def test_converts_solved_games_to_roster(self, sample_teams, sample_fields, sample_grades, sample_timeslots):
        """Test conversion of X dictionary to Roster object."""
        # Create a mock X dictionary with some scheduled games
        X = {}
        ef = sample_fields[0]
        
        # Game: Tigers 3rd vs Wests 3rd in week 1
        key1 = ('Tigers 3rd', 'Wests 3rd', '3rd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        X[key1] = True  # Scheduled
        
        # Game: Tigers PHL vs Wests PHL in week 2
        key2 = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 2, '2025-03-30', 2, 'EF', 'Newcastle International Hockey Centre')
        X[key2] = True  # Scheduled
        
        # Game that is NOT scheduled
        key3 = ('Tigers 2nd', 'Wests 2nd', '2nd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        X[key3] = False  # Not scheduled
        
        data = {
            'teams': sample_teams,
            'fields': sample_fields,
            'grades': sample_grades,
        }
        
        roster = convert_X_to_roster(X, data)
        
        assert isinstance(roster, Roster)
        assert len(roster.weeks) == 2  # Week 1 and Week 2
        
        # Check week 1 has one game
        week1 = [w for w in roster.weeks if w.week == 1][0]
        assert len(week1.games) == 1
        assert week1.games[0].team1 == 'Tigers 3rd'
        assert week1.games[0].team2 == 'Wests 3rd'
        
        # Check week 2 has one game
        week2 = [w for w in roster.weeks if w.week == 2][0]
        assert len(week2.games) == 1

    def test_handles_empty_x(self, sample_teams, sample_fields, sample_grades):
        """Test that empty X produces empty roster."""
        X = {}
        data = {
            'teams': sample_teams,
            'fields': sample_fields,
            'grades': sample_grades,
        }
        
        roster = convert_X_to_roster(X, data)
        assert isinstance(roster, Roster)
        assert len(roster.weeks) == 0


# Check if xlsxwriter is available
try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False


# ============== export_roster_to_excel Tests ==============

class TestExportRosterToExcel:
    """Tests for export_roster_to_excel function."""

    @pytest.mark.skipif(not HAS_XLSXWRITER, reason="xlsxwriter not installed")
    def test_exports_to_excel(self, sample_teams, sample_fields, sample_grades):
        """Test that roster is exported to Excel file."""
        # Create a simple roster
        ef = sample_fields[0]
        phl_grade = sample_grades[0]
        
        timeslot = Timeslot(
            date='2025-03-23',
            day='Sunday',
            time='10:00',
            week=1,
            day_slot=1,
            field=ef,
            round_no=1
        )
        
        game = Game(
            team1='Tigers PHL',
            team2='Wests PHL',
            timeslot=timeslot,
            field=ef,
            grade=phl_grade
        )
        
        weekly_draw = WeeklyDraw(
            week=1,
            round_no=1,
            games=[game],
            bye_teams=['Maitland PHL']
        )
        
        roster = Roster(weeks=[weekly_draw])
        
        data = {
            'teams': sample_teams,
        }
        
        # Export to a temp file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            filename = tmp.name
        
        try:
            export_roster_to_excel(roster, data, filename)
            
            # Verify file was created
            assert os.path.exists(filename)
            
            # Verify content using pandas
            import pandas as pd
            df = pd.read_excel(filename, sheet_name='Week 1')
            assert len(df) >= 1
            assert 'Tigers PHL' in df['TEAM 1'].values or 'Tigers PHL' in df['TEAM 2'].values
        finally:
            # Cleanup
            if os.path.exists(filename):
                os.remove(filename)


# ============== generate_timeslots Tests ==============

class TestGenerateTimeslots:
    """Tests for generate_timeslots function."""

    def test_generates_timeslots_for_date_range(self):
        """Test that timeslots are generated for the date range."""
        from utils import generate_timeslots
        from datetime import datetime, time as tm
        
        fields = [
            PlayingField(location='NIHC', name='EF'),
        ]
        
        day_time_map = {
            'NIHC': {
                'Sunday': [tm(10, 0), tm(11, 30)],
            }
        }
        
        start_date = datetime(2025, 3, 23)  # Sunday
        end_date = datetime(2025, 3, 30)    # Sunday
        
        timeslots = generate_timeslots(
            start_date, end_date, day_time_map, fields, {}
        )
        
        assert len(timeslots) > 0
        # Should have 2 Sundays * 2 times = 4 timeslots
        assert len(timeslots) == 4

    def test_respects_field_unavailabilities(self):
        """Test that unavailable dates/times are excluded."""
        from utils import generate_timeslots
        from datetime import datetime, time as tm
        
        fields = [
            PlayingField(location='NIHC', name='EF'),
        ]
        
        day_time_map = {
            'NIHC': {
                'Sunday': [tm(10, 0), tm(11, 30)],
            }
        }
        
        # Make first Sunday unavailable
        field_unavailabilities = {
            'NIHC': {
                'whole_days': [datetime(2025, 3, 23)]
            }
        }
        
        start_date = datetime(2025, 3, 23)  # Sunday
        end_date = datetime(2025, 3, 30)    # Sunday
        
        timeslots = generate_timeslots(
            start_date, end_date, day_time_map, fields, field_unavailabilities
        )
        
        # Only the second Sunday should have timeslots
        assert all(t['date'] == '2025-03-30' for t in timeslots)

    def test_validates_field_names(self):
        """Test that invalid field names raise errors."""
        from utils import generate_timeslots
        from datetime import datetime, time as tm
        
        fields = [
            PlayingField(location='NIHC', name='EF'),
        ]
        
        # Use a field name that doesn't exist
        day_time_map = {
            'NonexistentField': {
                'Sunday': [tm(10, 0)],
            }
        }
        
        with pytest.raises(ValueError, match="does not exist"):
            generate_timeslots(
                datetime(2025, 3, 23), datetime(2025, 3, 30),
                day_time_map, fields, {}
            )


# ============== max_games_per_grade Tests ==============

class TestMaxGamesPerGrade:
    """Tests for max_games_per_grade function."""

    def test_calculates_max_games(self):
        """Test max games calculation for various team counts."""
        from utils import max_games_per_grade
        
        grades = [
            Grade(name='PHL', teams=['A', 'B', 'C', 'D']),  # 4 teams
            Grade(name='2nd', teams=['A', 'B', 'C', 'D', 'E']),  # 5 teams
        ]
        
        result = max_games_per_grade(grades, max_rounds=15)
        
        assert 'PHL' in result
        assert '2nd' in result
        assert result['PHL'] > 0
        assert result['2nd'] > 0

    def test_handles_single_team_grade(self):
        """Test that single-team grade returns 0 games."""
        from utils import max_games_per_grade
        
        grades = [
            Grade(name='6th', teams=['A']),  # Only 1 team
        ]
        
        result = max_games_per_grade(grades, max_rounds=15)
        
        assert result['6th'] == 0

    def test_handles_empty_grade(self):
        """Test handling of empty grades (no teams)."""
        from utils import max_games_per_grade
        
        # Create grade with empty teams list
        grades = [Grade(name='Empty', teams=[])]
        
        # This should be handled gracefully
        result = max_games_per_grade(grades, max_rounds=15)
        assert result['Empty'] == 0


# ============== generate_games Tests ==============

class TestGenerateGames:
    """Tests for generate_games function."""

    def test_generates_all_matchups(self, sample_clubs):
        """Test that all possible matchups are generated."""
        from utils import generate_games
        
        tigers, wests, maitland = sample_clubs
        teams = [
            Team(name='A PHL', club=tigers, grade='PHL'),
            Team(name='B PHL', club=wests, grade='PHL'),
            Team(name='C PHL', club=maitland, grade='PHL'),
        ]
        
        games = generate_games(teams)
        
        # With 3 teams, should have 3 matchups: A-B, A-C, B-C
        assert len(games) == 3

    def test_only_same_grade_matchups(self, sample_clubs):
        """Test that only same-grade teams are matched."""
        from utils import generate_games
        
        tigers, wests, _ = sample_clubs
        teams = [
            Team(name='Tigers PHL', club=tigers, grade='PHL'),
            Team(name='Tigers 2nd', club=tigers, grade='2nd'),
            Team(name='Wests PHL', club=wests, grade='PHL'),
        ]
        
        games = generate_games(teams)
        
        # Only PHL teams can play each other
        assert len(games) == 1
        key = list(games.keys())[0]
        assert key[2] == 'PHL'

    def test_game_key_format(self, sample_clubs):
        """Test that game keys have correct format."""
        from utils import generate_games
        
        tigers, wests, _ = sample_clubs
        teams = [
            Team(name='Tigers PHL', club=tigers, grade='PHL'),
            Team(name='Wests PHL', club=wests, grade='PHL'),
        ]
        
        games = generate_games(teams)
        
        key = list(games.keys())[0]
        assert len(key) == 3  # (team1, team2, grade)
        assert key[0] == 'Tigers PHL'
        assert key[1] == 'Wests PHL'
        assert key[2] == 'PHL'


# ============== build_season_data Tests ==============

class TestBuildSeasonData:
    """Tests for build_season_data function."""

    def test_builds_data_from_config(self):
        """Test building season data from a config dict."""
        from utils import build_season_data
        from config import load_season_config
        
        config = load_season_config(2025)
        data = build_season_data(config)
        
        assert data is not None
        assert 'teams' in data
        assert 'grades' in data
        assert 'fields' in data
        assert 'clubs' in data
        assert 'timeslots' in data

    def test_creates_team_objects(self):
        """Test that Team objects are properly created."""
        from utils import build_season_data
        from config import load_season_config
        
        config = load_season_config(2025)
        data = build_season_data(config)
        
        for team in data['teams']:
            assert isinstance(team, Team)
            assert hasattr(team, 'name')
            assert hasattr(team, 'club')
            assert hasattr(team, 'grade')

    def test_creates_grade_objects(self):
        """Test that Grade objects are properly created."""
        from utils import build_season_data
        from config import load_season_config
        
        config = load_season_config(2025)
        data = build_season_data(config)
        
        for grade in data['grades']:
            assert isinstance(grade, Grade)
            assert grade.num_teams > 0

    def test_calculates_num_rounds(self):
        """Test that num_rounds is calculated correctly."""
        from utils import build_season_data
        from config import load_season_config
        
        config = load_season_config(2025)
        data = build_season_data(config)
        
        assert 'num_rounds' in data
        assert 'max' in data['num_rounds']
        
        for grade in data['grades']:
            assert grade.name in data['num_rounds']

    def test_raises_error_for_missing_path(self):
        """Test that missing teams_data_path raises error."""
        from utils import build_season_data
        
        config = {
            'year': 2099,
            'teams_data_path': '/nonexistent/path/to/teams',
            'start_date': datetime(2099, 3, 1),
            'end_date': datetime(2099, 8, 1),
            'max_rounds': 15,
            'fields': [{'location': 'Test', 'name': 'TF'}],
            'day_time_map': {},
            'phl_game_times': {},
            'field_unavailabilities': {},
        }
        
        with pytest.raises(FileNotFoundError):
            build_season_data(config)

    def test_preserves_year(self):
        """Test that year is preserved in output data."""
        from utils import build_season_data
        from config import load_season_config
        
        config = load_season_config(2025)
        data = build_season_data(config)
        
        assert data['year'] == 2025

    def test_includes_preference_data(self):
        """Test that preference data is included."""
        from utils import build_season_data
        from config import load_season_config
        
        config = load_season_config(2025)
        data = build_season_data(config)
        
        assert 'day_time_map' in data
        assert 'phl_game_times' in data
        assert 'field_unavailabilities' in data


# ============== Blocked Games (No-Play Variable Removal) Tests ==============

class TestBuildBlockedGameRules:
    """Tests for _build_blocked_game_rules function."""

    def test_empty_list_returns_empty(self, sample_teams):
        result = _build_blocked_game_rules([], sample_teams)
        assert result == {}

    def test_club_name_resolves_to_all_club_teams(self, sample_teams):
        """Club-only entry should resolve to all teams from that club."""
        blocked = [{
            'club': 'Tigers',
            'date': '2025-03-23',
            'description': 'Test block',
        }]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        assert len(rules) == 1
        scope_key = list(rules.keys())[0]
        matchers = rules[scope_key]
        matched_names = {m[1] for m in matchers}
        assert 'Tigers PHL' in matched_names
        assert 'Tigers 2nd' in matched_names
        assert 'Tigers 3rd' in matched_names

    def test_club_with_grade_resolves_to_specific_team(self, sample_teams):
        """Club + grade should resolve to only the specific graded team."""
        blocked = [{
            'club': 'Tigers',
            'grade': 'PHL',
            'date': '2025-03-23',
            'description': 'Test block PHL only',
        }]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        scope_key = list(rules.keys())[0]
        matchers = rules[scope_key]
        matched_names = {m[1] for m in matchers}
        assert matched_names == {'Tigers PHL'}

    def test_club_with_multiple_grades(self, sample_teams):
        """Club + grades list should resolve both grades."""
        blocked = [{
            'club': 'Tigers',
            'grades': ['PHL', '2nd'],
            'date': '2025-03-23',
            'description': 'Test multi-grade block',
        }]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        scope_key = list(rules.keys())[0]
        matchers = rules[scope_key]
        matched_names = {m[1] for m in matchers}
        assert matched_names == {'Tigers PHL', 'Tigers 2nd'}


class TestIsBlockedByNoPlay:
    """Tests for _is_blocked_by_no_play function."""

    def test_matching_team_and_scope_is_blocked(self, sample_teams):
        """Variable matching both scope and team should be blocked."""
        blocked = [{
            'club': 'Tigers',
            'grade': 'PHL',
            'date': '2025-03-23',
        }]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        # Tigers PHL vs Wests PHL on 2025-03-23 — should be blocked
        key = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key, rules) is True

    def test_matching_team_as_team2_is_blocked(self, sample_teams):
        """Variable with blocked team as team2 should also be blocked."""
        blocked = [{'club': 'Tigers', 'grade': 'PHL', 'date': '2025-03-23'}]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        # Wests PHL vs Tigers PHL (Tigers is team2)
        key = ('Maitland PHL', 'Tigers PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key, rules) is True

    def test_different_date_not_blocked(self, sample_teams):
        """Same team, different date should NOT be blocked."""
        blocked = [{'club': 'Tigers', 'grade': 'PHL', 'date': '2025-03-23'}]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        key = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 2, '2025-03-30', 2, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key, rules) is False

    def test_different_team_not_blocked(self, sample_teams):
        """Different team on same date should NOT be blocked."""
        blocked = [{'club': 'Tigers', 'grade': 'PHL', 'date': '2025-03-23'}]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        # Wests vs Maitland (no Tigers) on blocked date
        key = ('Maitland PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key, rules) is False

    def test_different_grade_not_blocked(self, sample_teams):
        """Same club different grade should NOT be blocked when grade is specified."""
        blocked = [{'club': 'Tigers', 'grade': 'PHL', 'date': '2025-03-23'}]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        key = ('Tigers 2nd', 'Wests 2nd', '2nd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key, rules) is False

    def test_no_grade_blocks_all_club_grades(self, sample_teams):
        """Club-only block (no grade) should block all grades."""
        blocked = [{'club': 'Tigers', 'date': '2025-03-23'}]
        rules = _build_blocked_game_rules(blocked, sample_teams)
        key_phl = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        key_2nd = ('Tigers 2nd', 'Wests 2nd', '2nd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key_phl, rules) is True
        assert _is_blocked_by_no_play(key_2nd, rules) is True

    def test_empty_rules_blocks_nothing(self):
        """No blocked game rules should block nothing."""
        key = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre')
        assert _is_blocked_by_no_play(key, {}) is False
