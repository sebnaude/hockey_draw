# tests/test_analytics_tester.py
"""
Unit tests for analytics/tester.py

Tests for DrawTester, Violation, and ViolationReport classes.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.tester import DrawTester, Violation, ViolationReport
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
    """Create sample stored games."""
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
            field_name='EF',
            field_location='Newcastle International Hockey Centre'
        ),
        StoredGame(
            game_id='G00004',
            team1='Tigers 3rd',
            team2='Wests 3rd',
            grade='3rd',
            week=2,
            round_no=2,
            date='2025-03-30',
            day='Sunday',
            time='11:30',
            day_slot=2,
            field_name='EF',
            field_location='Newcastle International Hockey Centre'
        ),
    ]


@pytest.fixture
def sample_draw_storage(sample_stored_games):
    """Create sample DrawStorage."""
    return DrawStorage(
        description='Test Draw',
        num_weeks=2,
        num_games=4,
        games=sample_stored_games
    )


@pytest.fixture
def sample_tester(sample_draw_storage, sample_data):
    """Create sample DrawTester."""
    return DrawTester(sample_draw_storage, sample_data)


# ============== Violation Tests ==============

class TestViolation:
    """Tests for Violation dataclass."""

    def test_create_violation(self):
        """Test creating a Violation instance."""
        v = Violation(
            constraint='NoDoubleBooking',
            severity='CRITICAL',
            message='Team Tigers PHL is double booked in week 1'
        )
        assert v.constraint == 'NoDoubleBooking'
        assert v.severity == 'CRITICAL'
        assert v.message == 'Team Tigers PHL is double booked in week 1'

    def test_violation_with_games(self):
        """Test creating a Violation with affected games."""
        v = Violation(
            constraint='NoDoubleBooking',
            severity='CRITICAL',
            message='Team is double booked',
            affected_games=['G00001', 'G00002'],
            week=1
        )
        assert len(v.affected_games) == 2
        assert v.week == 1

    def test_violation_str(self):
        """Test Violation string representation."""
        v = Violation(
            constraint='NoDoubleBooking',
            severity='CRITICAL',
            message='Team is double booked',
            affected_games=['G00001', 'G00002']
        )
        s = str(v)
        assert 'CRITICAL' in s
        assert 'NoDoubleBooking' in s
        assert 'G00001' in s


# ============== ViolationReport Tests ==============

class TestViolationReport:
    """Tests for ViolationReport dataclass."""

    def test_create_empty_report(self):
        """Test creating an empty ViolationReport."""
        report = ViolationReport(
            draw_description='Test Draw',
            total_games=10
        )
        assert not report.has_violations
        assert report.critical_count == 0
        assert report.high_count == 0

    def test_report_with_violations(self):
        """Test creating a report with violations."""
        violations = [
            Violation(constraint='A', severity='CRITICAL', message='Error 1'),
            Violation(constraint='B', severity='CRITICAL', message='Error 2'),
            Violation(constraint='C', severity='HIGH', message='Error 3'),
            Violation(constraint='D', severity='MEDIUM', message='Error 4'),
            Violation(constraint='E', severity='LOW', message='Error 5'),
        ]
        report = ViolationReport(
            draw_description='Test Draw',
            total_games=10,
            violations=violations
        )
        
        assert report.has_violations
        assert report.critical_count == 2
        assert report.high_count == 1
        assert report.medium_count == 1
        assert report.low_count == 1

    def test_summary_no_violations(self):
        """Test summary when no violations."""
        report = ViolationReport(
            draw_description='Test Draw',
            total_games=10
        )
        summary = report.summary()
        assert 'PASS' in summary
        assert 'No violations' in summary

    def test_summary_with_violations(self):
        """Test summary when violations exist."""
        violations = [
            Violation(constraint='A', severity='LOW', message='Error 1'),
        ]
        report = ViolationReport(
            draw_description='Test Draw',
            total_games=10,
            violations=violations
        )
        summary = report.summary()
        assert 'FAIL' in summary
        assert 'LOW' in summary or '1 violations' in summary  # Check violation exists

    def test_full_report(self):
        """Test full report generation."""
        violations = [
            Violation(constraint='NoDoubleBooking', severity='CRITICAL', message='Error 1'),
        ]
        report = ViolationReport(
            draw_description='Test Draw',
            total_games=10,
            violations=violations
        )
        full = report.full_report()
        assert 'CONSTRAINT VIOLATION REPORT' in full
        assert 'Test Draw' in full
        assert 'NoDoubleBooking' in full

    def test_by_constraint(self):
        """Test grouping violations by constraint."""
        violations = [
            Violation(constraint='A', severity='CRITICAL', message='Error 1'),
            Violation(constraint='A', severity='HIGH', message='Error 2'),
            Violation(constraint='B', severity='LOW', message='Error 3'),
        ]
        report = ViolationReport(
            draw_description='Test Draw',
            total_games=10,
            violations=violations
        )
        grouped = report.by_constraint()
        
        assert 'A' in grouped
        assert len(grouped['A']) == 2
        assert 'B' in grouped
        assert len(grouped['B']) == 1


# ============== DrawTester Tests ==============

class TestDrawTester:
    """Tests for DrawTester class."""

    def test_create_tester(self, sample_tester, sample_draw_storage):
        """Test creating a DrawTester instance."""
        assert sample_tester.draw.num_games == sample_draw_storage.num_games
        assert len(sample_tester.modifications) == 0

    def test_reset(self, sample_tester):
        """Test resetting tester to original state."""
        # Make a modification
        sample_tester.modifications.append("test modification")
        
        # Reset
        sample_tester.reset()
        
        # Check modifications cleared
        assert len(sample_tester.modifications) == 0

    def test_move_game(self, sample_tester):
        """Test moving a game to a new week."""
        # Move game G00001 from week 1 to week 3
        success = sample_tester.move_game('G00001', new_week=3)
        
        assert success
        
        # Verify the game was moved
        game = sample_tester.draw.get_game_by_id('G00001')
        assert game.week == 3
        
        # Verify modification was logged
        assert len(sample_tester.modifications) > 0

    def test_move_game_multiple_attributes(self, sample_tester):
        """Test moving a game with multiple attribute changes."""
        success = sample_tester.move_game(
            'G00001',
            new_week=3,
            new_day='Saturday',
            new_time='14:00',
            new_day_slot=3
        )
        
        assert success
        
        game = sample_tester.draw.get_game_by_id('G00001')
        assert game.week == 3
        assert game.day == 'Saturday'
        assert game.time == '14:00'
        assert game.day_slot == 3

    def test_move_game_nonexistent(self, sample_tester):
        """Test moving a non-existent game returns False."""
        success = sample_tester.move_game('G99999', new_week=3)
        assert not success

    def test_swap_games(self, sample_tester):
        """Test swapping two games' timeslots."""
        # Get original values
        game1_original = sample_tester.draw.get_game_by_id('G00001')
        game2_original = sample_tester.draw.get_game_by_id('G00002')
        
        week1 = game1_original.week
        week2 = game2_original.week
        slot1 = game1_original.day_slot
        slot2 = game2_original.day_slot
        
        # Swap
        success = sample_tester.swap_games('G00001', 'G00002')
        assert success
        
        # Verify swap
        game1_after = sample_tester.draw.get_game_by_id('G00001')
        game2_after = sample_tester.draw.get_game_by_id('G00002')
        
        # Note: both were in week 1 originally, so week doesn't change
        assert game1_after.day_slot == slot2
        assert game2_after.day_slot == slot1

    def test_swap_games_nonexistent(self, sample_tester):
        """Test swapping with non-existent game returns False."""
        success = sample_tester.swap_games('G00001', 'G99999')
        assert not success

    def test_find_game(self, sample_tester):
        """Test finding games by criteria."""
        # Find games for Tigers PHL
        games = sample_tester.find_game(team='Tigers PHL')
        assert len(games) == 2  # G00001 and G00003
        
        # Find games in week 1
        games = sample_tester.find_game(week=1)
        assert len(games) == 2  # G00001 and G00002
        
        # Find games by grade
        games = sample_tester.find_game(grade='PHL')
        assert len(games) == 2  # G00001 and G00003

    def test_find_game_team_vs_opponent(self, sample_tester):
        """Test finding specific matchup."""
        # Find Tigers PHL vs Wests PHL
        games = sample_tester.find_game(team='Tigers PHL', opponent='Wests PHL')
        assert len(games) == 1
        assert games[0].game_id == 'G00001'


# ============== DrawTester From Methods Tests ==============

class TestDrawTesterFromMethods:
    """Tests for DrawTester factory methods."""

    def test_from_file(self, sample_draw_storage, sample_data):
        """Test creating tester from file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            temp_path = f.name
        
        try:
            # Save draw to file
            sample_draw_storage.save(temp_path)
            
            # Create tester from file
            tester = DrawTester.from_file(temp_path, sample_data)
            
            assert tester.draw.num_games == sample_draw_storage.num_games
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_from_x_solution(self, sample_data):
        """Test creating tester from X solution."""
        X = {
            ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): True,
        }
        
        tester = DrawTester.from_X_solution(X, sample_data, description='X Test')
        
        assert tester.draw.num_games == 1
        assert tester.draw.description == 'X Test'
