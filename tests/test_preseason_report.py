# tests/test_preseason_report.py
"""
Unit tests for analytics/preseason_report.py - no mocks, real objects only.

Tests for:
- PreSeasonReport class
- Report generation methods
- Slot capacity analysis
- Integration with real season data
"""

import pytest
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.preseason_report import PreSeasonReport, run_preseason_check


# ============== Fixtures ==============

@pytest.fixture
def sample_data_and_config():
    """Create sample data and config using real model objects."""
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

    data = {
        'year': 2025,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'timeslots': [],
        'num_rounds': {'PHL': 10, '2nd': 10, 'max': 20},
    }

    config = {
        'year': 2025,
        'start_date': datetime(2025, 3, 23),
        'end_date': datetime(2025, 8, 31),
        'max_rounds': 20,
        'club_days': {},
        'preference_no_play': {},
        'phl_preferences': {},
        'field_unavailabilities': {},
        'phl_game_times': {},
        'second_grade_times': {},
        'day_time_map': {},
    }

    return data, config


@pytest.fixture
def real_2026_report():
    """Create a PreSeasonReport using real 2026 season data."""
    from config.season_2026 import SEASON_CONFIG, get_season_data
    data = get_season_data()
    return PreSeasonReport(data, SEASON_CONFIG)


# ============== PreSeasonReport Initialization Tests ==============

class TestPreSeasonReportInit:
    """Tests for PreSeasonReport initialization."""

    def test_initialization_with_data_and_config(self, sample_data_and_config):
        """Test report initializes with data and config."""
        data, config = sample_data_and_config
        report = PreSeasonReport(data, config)

        assert report.year == 2025
        assert len(report.teams) == 4
        assert len(report.grades) == 2

    def test_initialization_extracts_config_values(self, sample_data_and_config):
        """Test report extracts key config values."""
        data, config = sample_data_and_config
        report = PreSeasonReport(data, config)

        assert report.start_date == datetime(2025, 3, 23)
        assert report.max_rounds == 20

    def test_initialization_with_real_2026_data(self, real_2026_report):
        """Test report initializes correctly with real 2026 data."""
        report = real_2026_report

        assert report.year == 2026
        assert len(report.teams) > 0
        assert len(report.grades) > 0
        assert len(report.clubs) > 0


# ============== Report Generation Tests ==============

class TestReportGeneration:
    """Tests for report generation methods."""

    def test_generate_returns_string(self, sample_data_and_config):
        """Test generate_text_report() returns a string."""
        data, config = sample_data_and_config
        report = PreSeasonReport(data, config)

        output = report.generate_text_report()

        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_includes_year(self, sample_data_and_config):
        """Test generated report includes the year."""
        data, config = sample_data_and_config
        report = PreSeasonReport(data, config)

        output = report.generate_text_report()

        assert '2025' in output

    def test_generate_2026_report(self, real_2026_report):
        """Test generating a full 2026 report."""
        output = real_2026_report.generate_text_report()

        assert '2026' in output
        assert len(output) > 100
        assert 'PRE-SEASON CONFIGURATION REPORT' in output
        assert 'END OF REPORT' in output


# ============== Report Content Tests ==============

class TestReportContent:
    """Tests for specific report content with real data."""

    def test_report_includes_team_info(self, real_2026_report):
        """Test report includes team information."""
        output = real_2026_report.generate_text_report()

        assert 'PHL' in output
        assert 'TEAMS BY GRADE' in output

    def test_report_includes_venue_info(self, real_2026_report):
        """Test report includes venue information."""
        output = real_2026_report.generate_text_report()

        assert 'Newcastle' in output or 'Maitland' in output or 'Venue' in output

    def test_teams_by_grade(self, real_2026_report):
        """Test get_teams_by_grade returns correct structure."""
        teams_by_grade = real_2026_report.get_teams_by_grade()

        assert isinstance(teams_by_grade, dict)
        assert 'PHL' in teams_by_grade
        assert len(teams_by_grade['PHL']) > 0
        # Teams should be sorted
        assert teams_by_grade['PHL'] == sorted(teams_by_grade['PHL'])

    def test_special_requests(self, real_2026_report):
        """Test get_special_requests returns correct structure."""
        special = real_2026_report.get_special_requests()

        assert isinstance(special, dict)
        assert 'club_days' in special
        assert 'no_play_dates' in special
        assert 'team_conflicts' in special
        assert 'friday_night_allocations' in special
        assert 'special_games' in special

    def test_special_events(self, real_2026_report):
        """Test get_special_events returns a list."""
        events = real_2026_report.get_special_events()

        assert isinstance(events, list)
        # Should have at least some events for 2026
        if events:
            assert 'name' in events[0]
            assert 'date' in events[0]
            assert 'type' in events[0]

    def test_venue_times(self, real_2026_report):
        """Test get_venue_times returns correct structure."""
        venue_times = real_2026_report.get_venue_times()

        assert isinstance(venue_times, dict)
        assert len(venue_times) > 0
        # Each venue should have standard/phl/second categories
        for venue, categories in venue_times.items():
            assert 'standard' in categories
            assert 'phl' in categories
            assert 'second' in categories

    def test_field_unavailabilities(self, real_2026_report):
        """Test get_field_unavailabilities_summary returns correct structure."""
        unavail = real_2026_report.get_field_unavailabilities_summary()

        assert isinstance(unavail, dict)

    def test_validate_rounds(self, real_2026_report):
        """Test validate_rounds returns correct structure."""
        validation = real_2026_report.validate_rounds()

        assert isinstance(validation, dict)
        assert 'available_weekends' in validation
        assert 'configured_max_rounds' in validation
        assert 'valid' in validation
        assert isinstance(validation['valid'], bool)

    def test_calculate_available_weekends(self, real_2026_report):
        """Test calculate_available_weekends returns sensible values."""
        count, weekends = real_2026_report.calculate_available_weekends()

        assert count > 0
        assert len(weekends) == count
        # All should be Sundays
        for w in weekends:
            assert w.strftime('%A') == 'Sunday'


# ============== Slot Capacity Analysis Tests ==============

class TestSlotCapacityAnalysis:
    """Tests for slot capacity analysis functionality."""

    def test_slot_capacity_analysis_exists(self, real_2026_report):
        """Test that slot capacity analysis method exists."""
        assert hasattr(real_2026_report, 'get_slot_capacity_analysis')

    def test_slot_capacity_returns_dict(self, real_2026_report):
        """Test slot capacity analysis returns a dictionary."""
        analysis = real_2026_report.get_slot_capacity_analysis(total_matchups=100)

        assert isinstance(analysis, dict)
        assert 'total_matchups' in analysis
        assert 'weekends' in analysis
        assert 'capacity_ratio' in analysis
        assert analysis['total_matchups'] == 100

    def test_slot_capacity_in_report(self, real_2026_report):
        """Test slot capacity analysis appears in generated report."""
        output = real_2026_report.generate_text_report()

        assert 'SLOT CAPACITY' in output or 'capacity' in output.lower()

    def test_slot_capacity_values_sensible(self, real_2026_report):
        """Test slot capacity values are reasonable."""
        analysis = real_2026_report.get_slot_capacity_analysis(total_matchups=200)

        assert analysis['weekends'] > 0
        assert analysis['min_slots_per_weekend'] > 0
        assert analysis['total_slots_per_weekend'] >= 0
        assert analysis['capacity_ratio'] >= 0


# ============== Save/Output Tests ==============

class TestReportSave:
    """Tests for saving reports to files."""

    def test_save_report_to_file(self, real_2026_report, tmp_path):
        """Test saving report to a file."""
        output_file = str(tmp_path / "preseason_test.txt")
        real_2026_report.save(output_file)

        assert os.path.exists(output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '2026' in content
        assert len(content) > 100


# ============== Integration Tests ==============

class TestPreSeasonReportIntegration:
    """Integration tests using real season data."""

    def test_run_preseason_check_2026(self, tmp_path, capsys):
        """Test run_preseason_check with real 2026 data."""
        output_file = str(tmp_path / "preseason_2026.txt")
        result = run_preseason_check(2026, output=output_file)

        assert isinstance(result, bool)
        assert os.path.exists(output_file)

    def test_run_preseason_check_invalid_year(self):
        """Test run_preseason_check with invalid year."""
        result = run_preseason_check(9999)
        assert result is False

    def test_full_2026_report_no_errors(self):
        """Test that a full 2026 report generates without errors."""
        from config.season_2026 import SEASON_CONFIG, get_season_data
        data = get_season_data()
        report = PreSeasonReport(data, SEASON_CONFIG)

        # All these should work without exceptions
        output = report.generate_text_report()
        teams_by_grade = report.get_teams_by_grade()
        special = report.get_special_requests()
        events = report.get_special_events()
        venue_times = report.get_venue_times()
        unavail = report.get_field_unavailabilities_summary()
        validation = report.validate_rounds()

        assert len(output) > 500
        assert len(teams_by_grade) >= 1
        assert isinstance(special, dict)
        assert isinstance(events, list)
        assert isinstance(venue_times, dict)
        assert isinstance(unavail, dict)
        assert isinstance(validation, dict)
