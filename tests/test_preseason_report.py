# tests/test_preseason_report.py
"""
Unit tests for analytics/preseason_report.py

Tests for:
- PreSeasonReport class
- Report generation methods
- Slot capacity analysis
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.preseason_report import PreSeasonReport


# ============== Fixtures ==============

@pytest.fixture
def sample_config():
    """Create sample season config."""
    return {
        'year': 2025,
        'start_date': datetime(2025, 3, 23),
        'last_round_date': datetime(2025, 8, 31),
        'end_date': datetime(2025, 9, 20),
        'max_rounds': 20,
        'club_days': {},
        'preference_no_play': {},
        'phl_preferences': {},
        'field_unavailabilities': {},
        'phl_game_times': {},
        'second_grade_times': {},
    }


@pytest.fixture
def sample_data():
    """Create sample data dictionary."""
    from models import Club, Team, Grade, PlayingField
    
    clubs = [
        Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
        Club(name='Wests', home_field='Newcastle International Hockey Centre'),
    ]
    
    teams = [
        Team(name='Tigers PHL', club=clubs[0], grade='PHL'),
        Team(name='Tigers 2nd', club=clubs[0], grade='2nd'),
        Team(name='Wests PHL', club=clubs[1], grade='PHL'),
        Team(name='Wests 2nd', club=clubs[1], grade='2nd'),
    ]
    
    grades = [
        Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL']),
        Grade(name='2nd', teams=['Tigers 2nd', 'Wests 2nd']),
    ]
    
    fields = [
        PlayingField(name='EF', location='Newcastle International Hockey Centre'),
        PlayingField(name='WF', location='Newcastle International Hockey Centre'),
    ]
    
    return {
        'year': 2025,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'timeslots': [],
    }


# ============== PreSeasonReport Initialization Tests ==============

class TestPreSeasonReportInit:
    """Tests for PreSeasonReport initialization."""

    def test_initialization_with_data_and_config(self, sample_data, sample_config):
        """Test report initializes with data and config."""
        report = PreSeasonReport(sample_data, sample_config)
        
        assert report.year == 2025
        assert len(report.teams) == 4
        assert len(report.grades) == 2

    def test_initialization_extracts_config_values(self, sample_data, sample_config):
        """Test report extracts key config values."""
        report = PreSeasonReport(sample_data, sample_config)
        
        assert report.start_date == datetime(2025, 3, 23)
        assert report.max_rounds == 20


# ============== Report Generation Tests ==============

class TestReportGeneration:
    """Tests for report generation methods."""

    def test_generate_returns_string(self, sample_data, sample_config):
        """Test generate_text_report() returns a string."""
        report = PreSeasonReport(sample_data, sample_config)
        
        output = report.generate_text_report()
        
        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_includes_year(self, sample_data, sample_config):
        """Test generated report includes the year."""
        report = PreSeasonReport(sample_data, sample_config)
        
        output = report.generate_text_report()
        
        assert '2025' in output


# ============== Integration Tests with Real Data ==============

class TestPreSeasonReportIntegration:
    """Integration tests using real season data."""

    def test_with_2025_data(self):
        """Test report generation with real 2025 data."""
        from run import load_data_for_year
        from config.season_2025 import SEASON_CONFIG
        
        data = load_data_for_year(2025)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        output = report.generate_text_report()
        
        assert '2025' in output
        assert len(output) > 100

    def test_with_2026_data(self):
        """Test report generation with real 2026 data."""
        from run import load_data_for_year
        from config.season_2026 import SEASON_CONFIG
        
        data = load_data_for_year(2026)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        output = report.generate_text_report()
        
        assert '2026' in output
        assert len(output) > 100

    def test_report_includes_team_info(self):
        """Test report includes team information."""
        from run import load_data_for_year
        from config.season_2025 import SEASON_CONFIG
        
        data = load_data_for_year(2025)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        output = report.generate_text_report()
        
        # Should include team/grade info
        assert 'PHL' in output or 'Grade' in output

    def test_report_includes_venue_info(self):
        """Test report includes venue information."""
        from run import load_data_for_year
        from config.season_2025 import SEASON_CONFIG
        
        data = load_data_for_year(2025)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        output = report.generate_text_report()
        
        # Should include venue info
        assert 'Newcastle' in output or 'Maitland' in output or 'Venue' in output


# ============== Slot Capacity Analysis Tests ==============

class TestSlotCapacityAnalysis:
    """Tests for slot capacity analysis functionality."""

    def test_slot_capacity_analysis_exists(self):
        """Test that slot capacity analysis method exists."""
        from run import load_data_for_year
        from config.season_2026 import SEASON_CONFIG
        
        data = load_data_for_year(2026)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        # Should have the method
        assert hasattr(report, 'get_slot_capacity_analysis')

    def test_slot_capacity_returns_dict(self):
        """Test slot capacity analysis returns a dictionary."""
        from run import load_data_for_year
        from config.season_2026 import SEASON_CONFIG
        
        data = load_data_for_year(2026)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        # Need total_matchups parameter - use a sample value
        analysis = report.get_slot_capacity_analysis(total_matchups=100)
        
        assert isinstance(analysis, dict)

    def test_slot_capacity_in_report(self):
        """Test slot capacity analysis appears in generated report."""
        from run import load_data_for_year
        from config.season_2026 import SEASON_CONFIG
        
        data = load_data_for_year(2026)
        report = PreSeasonReport(data, SEASON_CONFIG)
        
        output = report.generate_text_report()
        
        # Should include slot capacity info
        assert 'SLOT CAPACITY' in output or 'timeslot' in output.lower() or 'capacity' in output.lower()
