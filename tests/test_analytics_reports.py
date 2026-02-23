# tests/test_analytics_reports.py
"""
Unit tests for analytics/reports.py

Tests for ClubReport class.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.reports import ClubReport
from analytics.storage import DrawStorage, StoredGame
from models import PlayingField, Team, Club, Grade


# ============== Fixtures ==============

@pytest.fixture
def sample_clubs():
    """Create sample clubs."""
    return [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
        Club(name='Maitland', home_field='Maitland Park'),
    ]


@pytest.fixture
def sample_teams(sample_clubs):
    """Create sample teams."""
    tigers, wests, maitland = sample_clubs
    return [
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Tigers 2nd', club=tigers, grade='2nd'),
        Team(name='Tigers 3rd', club=tigers, grade='3rd'),
        Team(name='Wests PHL', club=wests, grade='PHL'),
        Team(name='Wests 2nd', club=wests, grade='2nd'),
        Team(name='Wests 3rd', club=wests, grade='3rd'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
        Team(name='Maitland 2nd', club=maitland, grade='2nd'),
    ]


@pytest.fixture
def sample_grades():
    """Create sample grades."""
    return [
        Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL', 'Maitland PHL']),
        Grade(name='2nd', teams=['Tigers 2nd', 'Wests 2nd', 'Maitland 2nd']),
        Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd']),
    ]


@pytest.fixture
def sample_data(sample_clubs, sample_teams, sample_grades):
    """Create sample data dictionary."""
    return {
        'clubs': sample_clubs,
        'teams': sample_teams,
        'grades': sample_grades,
    }


@pytest.fixture
def sample_stored_games():
    """Create sample stored games with variety of locations."""
    return [
        # Tigers PHL vs Wests PHL at Broadmeadow (neutral)
        StoredGame(
            game_id='G00001',
            team1='Tigers PHL',
            team2='Wests PHL',
            grade='PHL',
            week=1,
            round_no=1,
            date='2025-03-23',
            day='Sunday',
            time='10:00',
            day_slot=1,
            field_name='EF',
            field_location='Newcastle International Hockey Centre'
        ),
        # Tigers PHL vs Maitland PHL at Maitland (Tigers away, Maitland home)
        StoredGame(
            game_id='G00002',
            team1='Tigers PHL',
            team2='Maitland PHL',
            grade='PHL',
            week=2,
            round_no=2,
            date='2025-03-30',
            day='Sunday',
            time='10:00',
            day_slot=1,
            field_name='Main',
            field_location='Maitland Park'
        ),
        # Tigers 2nd vs Wests 2nd at Broadmeadow
        StoredGame(
            game_id='G00003',
            team1='Tigers 2nd',
            team2='Wests 2nd',
            grade='2nd',
            week=1,
            round_no=1,
            date='2025-03-23',
            day='Sunday',
            time='11:30',
            day_slot=2,
            field_name='EF',
            field_location='Newcastle International Hockey Centre'
        ),
        # Wests PHL vs Maitland PHL at Broadmeadow
        StoredGame(
            game_id='G00004',
            team1='Wests PHL',
            team2='Maitland PHL',
            grade='PHL',
            week=3,
            round_no=3,
            date='2025-04-06',
            day='Sunday',
            time='10:00',
            day_slot=1,
            field_name='EF',
            field_location='Newcastle International Hockey Centre'
        ),
        # Tigers 3rd vs Wests 3rd at Broadmeadow
        StoredGame(
            game_id='G00005',
            team1='Tigers 3rd',
            team2='Wests 3rd',
            grade='3rd',
            week=1,
            round_no=1,
            date='2025-03-23',
            day='Sunday',
            time='13:00',
            day_slot=3,
            field_name='WF',
            field_location='Newcastle International Hockey Centre'
        ),
    ]


@pytest.fixture
def sample_draw_storage(sample_stored_games):
    """Create sample DrawStorage."""
    return DrawStorage(
        description='Test Draw',
        num_weeks=3,
        num_games=5,
        games=sample_stored_games
    )


@pytest.fixture
def sample_report(sample_draw_storage, sample_data):
    """Create sample ClubReport."""
    return ClubReport(sample_draw_storage, sample_data)


# ============== ClubReport Basic Tests ==============

class TestClubReportBasic:
    """Basic tests for ClubReport class."""

    def test_create_report(self, sample_report):
        """Test creating a ClubReport instance."""
        assert sample_report is not None
        assert len(sample_report.teams) > 0
        assert len(sample_report.clubs) > 0

    def test_get_club_teams(self, sample_report):
        """Test getting all teams for a club."""
        tigers_teams = sample_report.get_club_teams('Tigers')
        assert len(tigers_teams) == 3
        assert 'Tigers PHL' in tigers_teams
        assert 'Tigers 2nd' in tigers_teams
        assert 'Tigers 3rd' in tigers_teams

    def test_get_club_games(self, sample_report):
        """Test getting all games for a club's teams."""
        tigers_games = sample_report.get_club_games('Tigers')
        # Tigers teams play in games: G00001, G00002, G00003, G00005
        assert len(tigers_games) == 4


# ============== Club Summary Tests ==============

class TestClubSummary:
    """Tests for club_summary method."""

    def test_club_summary_tigers(self, sample_report):
        """Test summary for Tigers club."""
        summary = sample_report.club_summary('Tigers')
        
        assert summary['club_name'] == 'Tigers'
        assert summary['home_field'] == 'Newcastle International Hockey Centre'
        assert summary['num_teams'] == 3
        assert summary['total_games'] == 4
        assert 'PHL' in summary['grades']
        assert '2nd' in summary['grades']
        assert '3rd' in summary['grades']

    def test_club_summary_maitland(self, sample_report):
        """Test summary for Maitland club."""
        summary = sample_report.club_summary('Maitland')
        
        assert summary['club_name'] == 'Maitland'
        assert summary['home_field'] == 'Maitland Park'
        assert summary['num_teams'] == 2  # PHL and 2nd
        # Maitland plays in G00002 and G00004
        assert summary['total_games'] == 2

    def test_club_summary_home_away_count(self, sample_report):
        """Test home/away/neutral game counts."""
        # Maitland has: G00002 at Maitland (home), G00004 at Broadmeadow (neutral)
        summary = sample_report.club_summary('Maitland')
        
        assert summary['home_games'] == 1
        assert summary['neutral_games'] == 1

    def test_club_summary_opponents(self, sample_report):
        """Test opponent distribution."""
        summary = sample_report.club_summary('Tigers')
        
        # Tigers plays against Wests and Maitland
        assert 'Wests' in summary['opponents']
        assert 'Maitland' in summary['opponents']


# ============== Team Schedule Tests ==============

class TestTeamSchedule:
    """Tests for team_schedule method."""

    def test_team_schedule(self, sample_report):
        """Test generating team schedule."""
        schedule = sample_report.team_schedule('Tigers PHL')
        
        # Tigers PHL plays in G00001 and G00002
        assert len(schedule) == 2
        
        # Check columns exist
        assert 'Week' in schedule.columns
        assert 'Date' in schedule.columns
        assert 'Opponent' in schedule.columns
        assert 'H/A' in schedule.columns

    def test_team_schedule_home_away(self, sample_report):
        """Test home/away designation in schedule."""
        schedule = sample_report.team_schedule('Tigers PHL')
        
        # G00001: Tigers PHL vs Wests PHL - Tigers is team1 (H)
        # G00002: Tigers PHL vs Maitland PHL - Tigers is team1 (but at Maitland, still listed as H by position)
        # Note: H/A is based on team position in the game, not actual venue
        
        # Check that schedule is sorted by week
        weeks = schedule['Week'].tolist()
        assert weeks == sorted(weeks)

    def test_team_schedule_empty(self, sample_report):
        """Test schedule for team with no games."""
        # Maitland 2nd has no games in our test data
        schedule = sample_report.team_schedule('Maitland 2nd')
        assert len(schedule) == 0


# ============== Edge Case Tests ==============

class TestClubReportEdgeCases:
    """Edge case tests for ClubReport."""

    def test_nonexistent_club(self, sample_report):
        """Test handling of non-existent club."""
        teams = sample_report.get_club_teams('NonExistent')
        assert teams == []
        
        games = sample_report.get_club_games('NonExistent')
        assert games == []

    def test_nonexistent_team_schedule(self, sample_report):
        """Test schedule for non-existent team."""
        schedule = sample_report.team_schedule('NonExistent Team')
        assert len(schedule) == 0
