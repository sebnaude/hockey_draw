# tests/test_models.py
"""
Unit tests for models.py

Tests for all Pydantic model classes:
- PlayingField
- Grade  
- Timeslot
- Club
- Team
- ClubDay
- Game
- WeeklyDraw
- Roster
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    PlayingField, Grade, Timeslot, Club, Team, 
    ClubDay, Game, WeeklyDraw, Roster
)


# ============== PlayingField Tests ==============

class TestPlayingField:
    """Tests for PlayingField model."""

    def test_create_playing_field(self):
        """Test creating a PlayingField with valid data."""
        field = PlayingField(
            name='EF',
            location='Newcastle International Hockey Centre'
        )
        
        assert field.name == 'EF'
        assert field.location == 'Newcastle International Hockey Centre'

    def test_field_location_alias(self):
        """Test that field_location alias works."""
        field = PlayingField(
            name='EF',
            location='Newcastle International Hockey Centre'
        )
        
        assert field.field_location == 'Newcastle International Hockey Centre'

    def test_field_name_alias(self):
        """Test that field_name alias works."""
        field = PlayingField(
            name='EF',
            location='Newcastle International Hockey Centre'
        )
        
        assert field.field_name == 'EF'

    def test_invalid_attribute_raises_error(self):
        """Test that accessing invalid attribute raises AttributeError."""
        field = PlayingField(
            name='EF',
            location='Newcastle International Hockey Centre'
        )
        
        with pytest.raises(AttributeError):
            _ = field.nonexistent_attribute

    def test_playing_field_equality(self):
        """Test that two identical fields are equal."""
        field1 = PlayingField(name='EF', location='NIHC')
        field2 = PlayingField(name='EF', location='NIHC')
        
        assert field1 == field2

    def test_playing_field_dict_conversion(self):
        """Test converting PlayingField to dict."""
        field = PlayingField(name='EF', location='NIHC')
        
        d = field.model_dump()
        assert d['name'] == 'EF'
        assert d['location'] == 'NIHC'


# ============== Grade Tests ==============

class TestGrade:
    """Tests for Grade model."""

    def test_create_grade(self):
        """Test creating a Grade with valid data."""
        grade = Grade(
            name='PHL',
            teams=['Tigers PHL', 'Wests PHL', 'Maitland PHL']
        )
        
        assert grade.name == 'PHL'
        assert len(grade.teams) == 3
        assert grade.num_teams == 3

    def test_num_teams_auto_calculated(self):
        """Test that num_teams is automatically calculated from teams list."""
        grade = Grade(name='2nd', teams=['A', 'B', 'C', 'D', 'E'])
        
        assert grade.num_teams == 5

    def test_grade_comparison_phl_vs_2nd(self):
        """Test grade comparison - PHL is higher than 2nd."""
        phl = Grade(name='PHL', teams=['A'])
        second = Grade(name='2nd', teams=['B'])
        
        # PHL has lower index, so PHL < 2nd returns False
        # (smaller index = higher grade)
        assert not phl < second

    def test_grade_comparison_5th_vs_3rd(self):
        """Test grade comparison - 5th is lower than 3rd."""
        third = Grade(name='3rd', teams=['A'])
        fifth = Grade(name='5th', teams=['B'])
        
        # 5th has higher index, so 5th < 3rd returns True
        assert fifth < third

    def test_grade_comparison_same_grade(self):
        """Test comparing same grade returns False for less than."""
        grade1 = Grade(name='4th', teams=['A'])
        grade2 = Grade(name='4th', teams=['B'])
        
        assert not grade1 < grade2
        assert not grade2 < grade1

    def test_set_games_even_teams(self):
        """Test set_games calculation for even number of teams."""
        grade = Grade(name='PHL', teams=['A', 'B', 'C', 'D'])
        grade.set_games(15)
        
        # With 4 teams, num_teams-1 = 3
        # (15 // 3) * 3 = 15
        assert grade.num_games == 15

    def test_set_games_odd_teams(self):
        """Test set_games calculation for odd number of teams."""
        grade = Grade(name='5th', teams=['A', 'B', 'C', 'D', 'E'])
        grade.set_games(15)
        
        # With 5 teams, formula is different
        # (15 // 5) * 4 = 12
        assert grade.num_games == 12

    def test_grade_invalid_comparison(self):
        """Test that comparing Grade with non-Grade raises error."""
        grade = Grade(name='PHL', teams=['A'])
        
        result = grade < "not a grade"
        assert result == NotImplemented

    def test_grade_unknown_name_comparison(self):
        """Test comparing grades with unknown names."""
        grade1 = Grade(name='Unknown', teams=['A'])
        grade2 = Grade(name='PHL', teams=['B'])
        
        with pytest.raises(ValueError, match="Unknown grade"):
            _ = grade1 < grade2


# ============== Club Tests ==============

class TestClub:
    """Tests for Club model."""

    def test_create_club(self):
        """Test creating a Club with valid data."""
        club = Club(
            name='Tigers',
            home_field='Newcastle International Hockey Centre'
        )
        
        assert club.name == 'Tigers'
        assert club.home_field == 'Newcastle International Hockey Centre'
        assert club.num_teams == 0
        assert club.preferred_times == []

    def test_club_with_num_teams(self):
        """Test creating a Club with num_teams set."""
        club = Club(
            name='Tigers',
            home_field='NIHC',
            num_teams=6
        )
        
        assert club.num_teams == 6

    def test_club_with_preferred_times(self):
        """Test creating a Club with preferred times."""
        field = PlayingField(name='EF', location='NIHC')
        timeslot = Timeslot(
            date='2025-03-23',
            day='Sunday',
            time='10:00',
            week=1,
            day_slot=1,
            field=field,
            round_no=1
        )
        
        club = Club(
            name='Tigers',
            home_field='NIHC',
            preferred_times=[timeslot]
        )
        
        assert len(club.preferred_times) == 1


# ============== Team Tests ==============

class TestTeam:
    """Tests for Team model."""

    def test_create_team(self):
        """Test creating a Team with valid data."""
        club = Club(name='Tigers', home_field='NIHC')
        team = Team(
            name='Tigers PHL',
            club=club,
            grade='PHL'
        )
        
        assert team.name == 'Tigers PHL'
        assert team.club.name == 'Tigers'
        assert team.grade == 'PHL'
        assert team.preferred_times == []
        assert team.unavailable_times == []
        assert team.constraints == []

    def test_team_with_constraints(self):
        """Test creating a Team with constraints."""
        club = Club(name='Tigers', home_field='NIHC')
        team = Team(
            name='Tigers PHL',
            club=club,
            grade='PHL',
            constraints=['no_friday_night', 'prefer_morning']
        )
        
        assert len(team.constraints) == 2
        assert 'no_friday_night' in team.constraints


# ============== Timeslot Tests ==============

class TestTimeslot:
    """Tests for Timeslot model."""

    def test_create_timeslot(self):
        """Test creating a Timeslot with valid data."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(
            date='2025-03-23',
            day='Sunday',
            time='10:00',
            week=1,
            day_slot=1,
            field=field,
            round_no=1
        )
        
        assert ts.date == '2025-03-23'
        assert ts.day == 'Sunday'
        assert ts.time == '10:00'
        assert ts.week == 1
        assert ts.day_slot == 1
        assert ts.round_no == 1
        assert ts.field.name == 'EF'

    def test_timeslot_different_days(self):
        """Test creating timeslots for different days."""
        field = PlayingField(name='EF', location='NIHC')
        
        sat = Timeslot(date='2025-03-22', day='Saturday', time='14:00',
                       week=1, day_slot=1, field=field, round_no=1)
        sun = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                       week=1, day_slot=1, field=field, round_no=1)
        
        assert sat.day == 'Saturday'
        assert sun.day == 'Sunday'


# ============== ClubDay Tests ==============

class TestClubDay:
    """Tests for ClubDay model."""

    def test_create_club_day(self):
        """Test creating a ClubDay with valid data."""
        field = PlayingField(name='EF', location='NIHC')
        club_day = ClubDay(
            date='2025-04-06',
            day='Sunday',
            week=3,
            field=field
        )
        
        assert club_day.date == '2025-04-06'
        assert club_day.day == 'Sunday'
        assert club_day.week == 3
        assert club_day.field.name == 'EF'


# ============== Game Tests ==============

class TestGame:
    """Tests for Game model."""

    def test_create_game(self):
        """Test creating a Game with valid data."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL'])
        
        game = Game(
            team1='Tigers PHL',
            team2='Wests PHL',
            timeslot=ts,
            field=field,
            grade=grade
        )
        
        assert game.team1 == 'Tigers PHL'
        assert game.team2 == 'Wests PHL'
        assert game.timeslot.week == 1
        assert game.field.name == 'EF'
        assert game.grade.name == 'PHL'

    def test_game_dict_conversion(self):
        """Test converting Game to dict."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL'])
        
        game = Game(
            team1='Tigers PHL',
            team2='Wests PHL',
            timeslot=ts,
            field=field,
            grade=grade
        )
        
        d = game.model_dump()
        assert d['team1'] == 'Tigers PHL'
        assert d['team2'] == 'Wests PHL'


# ============== WeeklyDraw Tests ==============

class TestWeeklyDraw:
    """Tests for WeeklyDraw model."""

    def test_create_weekly_draw(self):
        """Test creating a WeeklyDraw with valid data."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['A', 'B', 'C'])
        
        game = Game(team1='A', team2='B', timeslot=ts, field=field, grade=grade)
        
        weekly = WeeklyDraw(
            week=1,
            round_no=1,
            games=[game],
            bye_teams=['C']
        )
        
        assert weekly.week == 1
        assert weekly.round_no == 1
        assert len(weekly.games) == 1
        assert 'C' in weekly.bye_teams

    def test_weekly_draw_empty_byes(self):
        """Test WeeklyDraw with no bye teams."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['A', 'B'])
        
        game = Game(team1='A', team2='B', timeslot=ts, field=field, grade=grade)
        
        weekly = WeeklyDraw(week=1, round_no=1, games=[game])
        
        assert weekly.bye_teams == []

    def test_weekly_draw_multiple_games(self):
        """Test WeeklyDraw with multiple games."""
        field = PlayingField(name='EF', location='NIHC')
        ts1 = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                       week=1, day_slot=1, field=field, round_no=1)
        ts2 = Timeslot(date='2025-03-23', day='Sunday', time='11:30',
                       week=1, day_slot=2, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['A', 'B', 'C', 'D'])
        
        game1 = Game(team1='A', team2='B', timeslot=ts1, field=field, grade=grade)
        game2 = Game(team1='C', team2='D', timeslot=ts2, field=field, grade=grade)
        
        weekly = WeeklyDraw(week=1, round_no=1, games=[game1, game2])
        
        assert len(weekly.games) == 2


# ============== Roster Tests ==============

class TestRoster:
    """Tests for Roster model."""

    def test_create_roster(self):
        """Test creating a Roster with valid data."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['A', 'B'])
        
        game = Game(team1='A', team2='B', timeslot=ts, field=field, grade=grade)
        weekly = WeeklyDraw(week=1, round_no=1, games=[game])
        
        roster = Roster(weeks=[weekly])
        
        assert len(roster.weeks) == 1
        assert roster.weeks[0].week == 1

    def test_roster_multiple_weeks(self):
        """Test Roster with multiple weeks."""
        field = PlayingField(name='EF', location='NIHC')
        grade = Grade(name='PHL', teams=['A', 'B'])
        
        weeks_list = []
        for week in range(1, 4):
            ts = Timeslot(date=f'2025-03-{20+week}', day='Sunday', time='10:00',
                          week=week, day_slot=1, field=field, round_no=week)
            game = Game(team1='A', team2='B', timeslot=ts, field=field, grade=grade)
            weekly = WeeklyDraw(week=week, round_no=week, games=[game])
            weeks_list.append(weekly)
        
        roster = Roster(weeks=weeks_list)
        
        assert len(roster.weeks) == 3

    def test_roster_save(self):
        """Test saving Roster to JSON file."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['A', 'B'])
        
        game = Game(team1='A', team2='B', timeslot=ts, field=field, grade=grade)
        weekly = WeeklyDraw(week=1, round_no=1, games=[game])
        roster = Roster(weeks=[weekly])
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filename = tmp.name
        
        try:
            roster.save(filename)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            assert 'weeks' in data
            assert len(data['weeks']) == 1
            assert data['weeks'][0]['week'] == 1
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_roster_dict_conversion(self):
        """Test converting Roster to dict."""
        field = PlayingField(name='EF', location='NIHC')
        ts = Timeslot(date='2025-03-23', day='Sunday', time='10:00',
                      week=1, day_slot=1, field=field, round_no=1)
        grade = Grade(name='PHL', teams=['A', 'B'])
        
        game = Game(team1='A', team2='B', timeslot=ts, field=field, grade=grade)
        weekly = WeeklyDraw(week=1, round_no=1, games=[game])
        roster = Roster(weeks=[weekly])
        
        d = roster.model_dump()
        
        assert 'weeks' in d
        assert len(d['weeks']) == 1
        assert d['weeks'][0]['games'][0]['team1'] == 'A'


# ============== Model Validation Tests ==============

class TestModelValidation:
    """Tests for model validation."""

    def test_playing_field_requires_name(self):
        """Test that PlayingField requires name field."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            PlayingField(location='NIHC')

    def test_playing_field_requires_location(self):
        """Test that PlayingField requires location field."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            PlayingField(name='EF')

    def test_grade_requires_name(self):
        """Test that Grade requires name field."""
        with pytest.raises(Exception):
            Grade(teams=['A', 'B'])

    def test_grade_requires_teams(self):
        """Test that Grade requires teams field."""
        with pytest.raises(Exception):
            Grade(name='PHL')

    def test_team_requires_club(self):
        """Test that Team requires club field."""
        with pytest.raises(Exception):
            Team(name='Tigers PHL', grade='PHL')

    def test_timeslot_requires_all_fields(self):
        """Test that Timeslot requires all mandatory fields."""
        field = PlayingField(name='EF', location='NIHC')
        
        # Missing time should fail
        with pytest.raises(Exception):
            Timeslot(date='2025-03-23', day='Sunday', week=1, 
                     day_slot=1, field=field, round_no=1)
