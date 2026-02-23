# tests/test_analytics_storage.py
"""
Unit tests for analytics/storage.py

Tests for DrawStorage and StoredGame classes.
"""

import pytest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.storage import DrawStorage, StoredGame
from models import PlayingField, Team, Club, Grade, Timeslot, Game, WeeklyDraw, Roster


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
        Team(name='Wests PHL', club=wests, grade='PHL'),
        Team(name='Wests 2nd', club=wests, grade='2nd'),
        Team(name='Maitland PHL', club=maitland, grade='PHL'),
    ]


@pytest.fixture
def sample_fields():
    """Create sample playing fields."""
    return [
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
    ]


@pytest.fixture
def sample_grades():
    """Create sample grades."""
    return [
        Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL', 'Maitland PHL']),
        Grade(name='2nd', teams=['Tigers 2nd', 'Wests 2nd']),
    ]


@pytest.fixture
def sample_stored_games():
    """Create sample stored games for testing."""
    return [
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
        StoredGame(
            game_id='G00002',
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
        StoredGame(
            game_id='G00003',
            team1='Tigers PHL',
            team2='Maitland PHL',
            grade='PHL',
            week=2,
            round_no=2,
            date='2025-03-30',
            day='Sunday',
            time='10:00',
            day_slot=1,
            field_name='WF',
            field_location='Newcastle International Hockey Centre'
        ),
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
    ]


@pytest.fixture
def sample_draw_storage(sample_stored_games):
    """Create a sample DrawStorage instance."""
    return DrawStorage(
        description='Test Draw',
        num_weeks=3,
        num_games=4,
        games=sample_stored_games
    )


# ============== StoredGame Tests ==============

class TestStoredGame:
    """Tests for StoredGame model."""

    def test_create_stored_game(self):
        """Test creating a StoredGame instance."""
        game = StoredGame(
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
        )
        assert game.game_id == 'G00001'
        assert game.team1 == 'Tigers PHL'
        assert game.team2 == 'Wests PHL'

    def test_teams_property(self):
        """Test teams property returns tuple of both teams."""
        game = StoredGame(
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
        )
        assert game.teams == ('Tigers PHL', 'Wests PHL')

    def test_to_key(self):
        """Test conversion to X-dict key format."""
        game = StoredGame(
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
        )
        key = game.to_key()
        assert key == (
            'Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1,
            '10:00', 1, '2025-03-23', 1,
            'EF', 'Newcastle International Hockey Centre'
        )
        assert len(key) == 11


# ============== DrawStorage Basic Tests ==============

class TestDrawStorageBasic:
    """Basic tests for DrawStorage model."""

    def test_create_empty_draw_storage(self):
        """Test creating an empty DrawStorage."""
        draw = DrawStorage()
        assert draw.num_games == 0
        assert draw.num_weeks == 0
        assert len(draw.games) == 0

    def test_create_with_games(self, sample_draw_storage):
        """Test creating DrawStorage with games."""
        assert sample_draw_storage.num_games == 4
        assert sample_draw_storage.num_weeks == 3
        assert len(sample_draw_storage.games) == 4

    def test_get_game_by_id(self, sample_draw_storage):
        """Test retrieving game by ID."""
        game = sample_draw_storage.get_game_by_id('G00001')
        assert game is not None
        assert game.team1 == 'Tigers PHL'
        
        # Test non-existent ID
        game = sample_draw_storage.get_game_by_id('G99999')
        assert game is None

    def test_get_games_by_team(self, sample_draw_storage):
        """Test retrieving games for a specific team."""
        tigers_phl_games = sample_draw_storage.get_games_by_team('Tigers PHL')
        assert len(tigers_phl_games) == 2  # G00001 and G00003
        
        wests_phl_games = sample_draw_storage.get_games_by_team('Wests PHL')
        assert len(wests_phl_games) == 2  # G00001 and G00004

    def test_get_games_by_week(self, sample_draw_storage):
        """Test retrieving games for a specific week."""
        week1_games = sample_draw_storage.get_games_by_week(1)
        assert len(week1_games) == 2  # G00001 and G00002
        
        week2_games = sample_draw_storage.get_games_by_week(2)
        assert len(week2_games) == 1  # G00003

    def test_get_games_by_grade(self, sample_draw_storage):
        """Test retrieving games for a specific grade."""
        phl_games = sample_draw_storage.get_games_by_grade('PHL')
        assert len(phl_games) == 3  # G00001, G00003, G00004
        
        second_games = sample_draw_storage.get_games_by_grade('2nd')
        assert len(second_games) == 1  # G00002


# ============== DrawStorage Filtering Tests ==============

class TestDrawStorageFiltering:
    """Tests for DrawStorage filter methods."""

    def test_filter_games_by_team(self, sample_draw_storage):
        """Test filtering games by team."""
        games = sample_draw_storage.filter_games(team='Tigers PHL')
        assert len(games) == 2

    def test_filter_games_by_grade(self, sample_draw_storage):
        """Test filtering games by grade."""
        games = sample_draw_storage.filter_games(grade='PHL')
        assert len(games) == 3

    def test_filter_games_by_week(self, sample_draw_storage):
        """Test filtering games by week."""
        games = sample_draw_storage.filter_games(week=1)
        assert len(games) == 2

    def test_filter_games_multiple_criteria(self, sample_draw_storage):
        """Test filtering games by multiple criteria."""
        games = sample_draw_storage.filter_games(grade='PHL', week=1)
        assert len(games) == 1
        assert games[0].game_id == 'G00001'


# ============== DrawStorage Lock and Split Tests ==============

class TestDrawStorageLockAndSplit:
    """Tests for lock_and_split functionality."""

    def test_get_locked_games(self, sample_draw_storage):
        """Test getting locked games up to a week."""
        locked = sample_draw_storage.get_locked_games(lock_weeks_up_to=1)
        assert len(locked) == 2
        assert all(g.week == 1 for g in locked)

    def test_get_remaining_games(self, sample_draw_storage):
        """Test getting remaining games after lock point."""
        remaining = sample_draw_storage.get_remaining_games(lock_weeks_up_to=1)
        assert len(remaining) == 2
        assert all(g.week > 1 for g in remaining)

    def test_lock_and_split(self, sample_draw_storage):
        """Test splitting draw into locked and unlocked portions."""
        locked_draw, remaining_draw = sample_draw_storage.lock_and_split(lock_weeks_up_to=2)
        
        # Locked should have weeks 1 and 2
        assert locked_draw.num_games == 3
        assert all(g.week <= 2 for g in locked_draw.games)
        
        # Remaining should have week 3 only
        assert remaining_draw.num_games == 1
        assert all(g.week > 2 for g in remaining_draw.games)


# ============== DrawStorage Serialization Tests ==============

class TestDrawStorageSerialization:
    """Tests for save/load functionality."""

    def test_save_and_load(self, sample_draw_storage):
        """Test saving and loading DrawStorage to/from JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            temp_path = f.name
        
        try:
            # Save
            sample_draw_storage.save(temp_path)
            
            # Load
            loaded = DrawStorage.load(temp_path)
            
            # Verify
            assert loaded.num_games == sample_draw_storage.num_games
            assert loaded.num_weeks == sample_draw_storage.num_weeks
            assert len(loaded.games) == len(sample_draw_storage.games)
            
            # Check first game
            assert loaded.games[0].game_id == sample_draw_storage.games[0].game_id
            assert loaded.games[0].team1 == sample_draw_storage.games[0].team1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_to_x_dict(self, sample_draw_storage):
        """Test converting DrawStorage back to X dict format."""
        x_dict = sample_draw_storage.to_X_dict()
        
        assert len(x_dict) == 4
        
        # Check first game's key is in the dict
        first_key = sample_draw_storage.games[0].to_key()
        assert first_key in x_dict
        assert x_dict[first_key] == 1


# ============== DrawStorage From X Solution Tests ==============

class TestDrawStorageFromXSolution:
    """Tests for creating DrawStorage from X solution."""

    def test_from_x_solution_dict(self):
        """Test creating DrawStorage from X dict (scheduled games as True)."""
        X = {
            ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): True,
            ('Tigers 2nd', 'Wests 2nd', '2nd', 'Sunday', 2, '11:30', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): True,
            ('Tigers PHL', 'Maitland PHL', 'PHL', 'Sunday', 1, '10:00', 2, '2025-03-30', 2, 'WF', 'Newcastle International Hockey Centre'): False,  # Not scheduled
        }
        
        draw = DrawStorage.from_X_solution(X, description='Test Draw')
        
        assert draw.num_games == 2
        assert len(draw.games) == 2

    def test_from_x_solution_handles_empty(self):
        """Test creating DrawStorage from empty X."""
        X = {}
        draw = DrawStorage.from_X_solution(X)
        
        assert draw.num_games == 0
        assert len(draw.games) == 0


# ============== DrawStorage From Roster Tests ==============

class TestDrawStorageFromRoster:
    """Tests for creating DrawStorage from Roster object."""

    def test_from_roster(self, sample_fields, sample_grades):
        """Test creating DrawStorage from Roster."""
        ef = sample_fields[0]
        phl_grade = sample_grades[0]
        
        timeslot1 = Timeslot(
            date='2025-03-23',
            day='Sunday',
            time='10:00',
            week=1,
            day_slot=1,
            field=ef,
            round_no=1
        )
        
        game1 = Game(
            team1='Tigers PHL',
            team2='Wests PHL',
            timeslot=timeslot1,
            field=ef,
            grade=phl_grade
        )
        
        weekly_draw = WeeklyDraw(
            week=1,
            round_no=1,
            games=[game1],
            bye_teams=['Maitland PHL']
        )
        
        roster = Roster(weeks=[weekly_draw])
        
        draw = DrawStorage.from_roster(roster, description='From Roster Test')
        
        assert draw.num_games == 1
        assert draw.num_weeks == 1
        assert draw.games[0].team1 == 'Tigers PHL'
        assert draw.games[0].team2 == 'Wests PHL'
        assert draw.description == 'From Roster Test'
