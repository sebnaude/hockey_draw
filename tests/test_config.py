# tests/test_config.py
"""
Unit tests for config module.

Tests for:
- load_season_config: Load raw config dict for a season
- load_season_data: Load complete solver-ready data
- Dynamic module loading
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_season_config, load_season_data
from models import PlayingField, Team, Club, Grade, Timeslot


# ============== load_season_config Tests ==============

class TestLoadSeasonConfig:
    """Tests for load_season_config function."""

    def test_loads_2025_config(self):
        """Test loading 2025 season configuration."""
        config = load_season_config(2025)
        
        assert config is not None
        assert config['year'] == 2025
        assert 'start_date' in config
        assert 'end_date' in config
        assert 'fields' in config
        assert 'day_time_map' in config
        assert 'teams_data_path' in config

    def test_loads_2026_config(self):
        """Test loading 2026 season configuration."""
        config = load_season_config(2026)
        
        assert config is not None
        assert config['year'] == 2026
        assert 'start_date' in config
        assert 'end_date' in config

    def test_raises_error_for_nonexistent_year(self):
        """Test that loading a non-existent year raises ValueError."""
        with pytest.raises(ValueError, match="No configuration found"):
            load_season_config(1999)

    def test_config_has_required_fields(self):
        """Test that config contains all required fields."""
        config = load_season_config(2025)
        
        required_fields = [
            'year',
            'start_date',
            'end_date',
            'max_rounds',
            'teams_data_path',
            'fields',
            'day_time_map',
            'phl_game_times',
            'field_unavailabilities',
        ]
        
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

    def test_config_fields_is_list(self):
        """Test that fields is a list of field definitions."""
        config = load_season_config(2025)
        
        assert isinstance(config['fields'], list)
        assert len(config['fields']) > 0
        
        # Each field should have name and location
        for field in config['fields']:
            assert 'name' in field
            assert 'location' in field


# ============== load_season_data Tests ==============

class TestLoadSeasonData:
    """Tests for load_season_data function."""

    def test_loads_2025_data(self):
        """Test loading complete 2025 season data."""
        data = load_season_data(2025)
        
        assert data is not None
        assert data['year'] == 2025
        assert 'teams' in data
        assert 'grades' in data
        assert 'fields' in data
        assert 'clubs' in data
        assert 'timeslots' in data

    def test_loads_2026_data(self):
        """Test loading complete 2026 season data."""
        data = load_season_data(2026)
        
        assert data is not None
        assert data['year'] == 2026

    def test_raises_error_for_nonexistent_year(self):
        """Test that loading data for non-existent year raises ValueError."""
        with pytest.raises(ValueError, match="No configuration found"):
            load_season_data(1999)

    def test_data_has_team_objects(self):
        """Test that data contains Team model objects."""
        data = load_season_data(2025)
        
        assert len(data['teams']) > 0
        for team in data['teams']:
            assert isinstance(team, Team)
            assert hasattr(team, 'name')
            assert hasattr(team, 'club')
            assert hasattr(team, 'grade')

    def test_data_has_grade_objects(self):
        """Test that data contains Grade model objects."""
        data = load_season_data(2025)
        
        assert len(data['grades']) > 0
        for grade in data['grades']:
            assert isinstance(grade, Grade)
            assert hasattr(grade, 'name')
            assert hasattr(grade, 'teams')
            assert hasattr(grade, 'num_teams')

    def test_data_has_field_objects(self):
        """Test that data contains PlayingField model objects."""
        data = load_season_data(2025)
        
        assert len(data['fields']) > 0
        for field in data['fields']:
            assert isinstance(field, PlayingField)
            assert hasattr(field, 'name')
            assert hasattr(field, 'location')

    def test_data_has_club_objects(self):
        """Test that data contains Club model objects."""
        data = load_season_data(2025)
        
        assert len(data['clubs']) > 0
        for club in data['clubs']:
            assert isinstance(club, Club)
            assert hasattr(club, 'name')
            assert hasattr(club, 'home_field')

    def test_data_has_timeslot_objects(self):
        """Test that data contains Timeslot model objects."""
        data = load_season_data(2025)
        
        assert len(data['timeslots']) > 0
        for ts in data['timeslots']:
            assert isinstance(ts, Timeslot)
            assert hasattr(ts, 'date')
            assert hasattr(ts, 'day')
            assert hasattr(ts, 'time')
            assert hasattr(ts, 'week')

    def test_data_has_num_rounds(self):
        """Test that data contains calculated num_rounds."""
        data = load_season_data(2025)
        
        assert 'num_rounds' in data
        assert isinstance(data['num_rounds'], dict)
        assert 'max' in data['num_rounds']
        
        # Each grade should have a rounds entry
        for grade in data['grades']:
            assert grade.name in data['num_rounds']

    def test_team_clubs_are_linked(self):
        """Test that team club references are valid Club objects."""
        data = load_season_data(2025)
        
        club_names = {club.name for club in data['clubs']}
        
        for team in data['teams']:
            assert isinstance(team.club, Club)
            assert team.club.name in club_names

    def test_grades_contain_valid_team_names(self):
        """Test that grade team lists match actual team names."""
        data = load_season_data(2025)
        
        all_team_names = {team.name for team in data['teams']}
        
        for grade in data['grades']:
            for team_name in grade.teams:
                assert team_name in all_team_names, f"Grade {grade.name} references unknown team: {team_name}"

    def test_different_years_have_different_team_counts(self):
        """Test that different years can have different team configurations."""
        data_2025 = load_season_data(2025)
        data_2026 = load_season_data(2026)
        
        # Both should have teams, but counts may differ
        assert len(data_2025['teams']) > 0
        assert len(data_2026['teams']) > 0
        
        # Years should be correctly set
        assert data_2025['year'] == 2025
        assert data_2026['year'] == 2026

    def test_data_includes_preference_config(self):
        """Test that data includes preference configuration."""
        data = load_season_data(2025)
        
        # Should have preference-related fields
        assert 'day_time_map' in data
        assert 'phl_game_times' in data
        assert 'field_unavailabilities' in data

    def test_timeslots_are_in_chronological_order(self):
        """Test that timeslots are roughly ordered by date and time."""
        data = load_season_data(2025)
        
        # Get unique weeks
        weeks = sorted({ts.week for ts in data['timeslots']})
        assert weeks == list(range(1, len(weeks) + 1))

    def test_timeslots_have_valid_days(self):
        """Test that timeslots only have valid day names."""
        data = load_season_data(2025)
        
        valid_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                      'Friday', 'Saturday', 'Sunday'}
        
        for ts in data['timeslots']:
            assert ts.day in valid_days, f"Invalid day: {ts.day}"

    def test_timeslots_have_valid_fields(self):
        """Test that timeslot fields reference valid PlayingField objects."""
        data = load_season_data(2025)
        
        field_names = {f.name for f in data['fields']}
        
        for ts in data['timeslots']:
            assert isinstance(ts.field, PlayingField)
            assert ts.field.name in field_names


# ============== Integration Tests ==============

class TestConfigIntegration:
    """Integration tests for config module."""

    def test_config_and_data_consistency(self):
        """Test that config and data are consistent."""
        config = load_season_config(2025)
        data = load_season_data(2025)
        
        # Year should match
        assert config['year'] == data['year']
        
        # Number of fields should match
        assert len(config['fields']) == len(data['fields'])

    def test_multiple_loads_are_consistent(self):
        """Test that multiple loads return equivalent data."""
        data1 = load_season_data(2025)
        data2 = load_season_data(2025)
        
        # Should have same team count
        assert len(data1['teams']) == len(data2['teams'])
        
        # Same team names
        names1 = {t.name for t in data1['teams']}
        names2 = {t.name for t in data2['teams']}
        assert names1 == names2
