# test_draw_outcomes.py
"""
Outcome tests for validating completed draws.

These tests examine a completed draw (X_solution) and verify that no constraint
violations exist. If the optimizer worked correctly, all tests should pass.
These tests can also be used to validate manually created or imported draws.
"""

import pytest
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot


# ============== Helper Functions ==============

def extract_scheduled_games(X_solution: Dict) -> List[Tuple]:
    """Extract only the scheduled games (value=1) from solution."""
    return [key for key, value in X_solution.items() if value == 1 and len(key) >= 11]


def get_team_names_from_game(game_key: Tuple) -> Tuple[str, str]:
    """Extract team names from a game key."""
    return game_key[0], game_key[1]


def get_club_from_team(team_name: str, teams: List[Team]) -> str:
    """Get club name for a team."""
    for team in teams:
        if team.name == team_name:
            return team.club.name
    raise ValueError(f"Team {team_name} not found")


# ============== Outcome Validation Classes ==============

class DrawOutcomeValidator:
    """Base class for validating draw outcomes."""
    
    def __init__(self, X_solution: Dict, data: Dict):
        self.X_solution = X_solution
        self.data = data
        self.scheduled_games = extract_scheduled_games(X_solution)
        self.violations = []
    
    def validate(self) -> List[str]:
        """Run validation and return list of violations."""
        raise NotImplementedError
    
    def is_valid(self) -> bool:
        """Return True if no violations found."""
        return len(self.validate()) == 0


class NoDoubleBookingTeamsValidator(DrawOutcomeValidator):
    """Validate that no team plays more than one game per week."""
    
    def validate(self) -> List[str]:
        self.violations = []
        
        # Group games by (week, team)
        team_games_per_week = defaultdict(list)
        
        for game in self.scheduled_games:
            t1, t2 = get_team_names_from_game(game)
            week = game[6]
            team_games_per_week[(week, t1)].append(game)
            team_games_per_week[(week, t2)].append(game)
        
        # Check for violations
        for (week, team), games in team_games_per_week.items():
            if len(games) > 1:
                self.violations.append(
                    f"Team '{team}' plays {len(games)} games in week {week}: "
                    f"{[f'{g[0]} vs {g[1]}' for g in games]}"
                )
        
        return self.violations


class NoDoubleBookingFieldsValidator(DrawOutcomeValidator):
    """Validate that no field hosts more than one game per timeslot."""
    
    def validate(self) -> List[str]:
        self.violations = []
        
        # Group games by (week, day_slot, field_name)
        field_games_per_slot = defaultdict(list)
        
        for game in self.scheduled_games:
            week = game[6]
            day_slot = game[4]
            field_name = game[9]
            field_games_per_slot[(week, day_slot, field_name)].append(game)
        
        # Check for violations
        for (week, day_slot, field), games in field_games_per_slot.items():
            if len(games) > 1:
                self.violations.append(
                    f"Field '{field}' has {len(games)} games in week {week}, slot {day_slot}: "
                    f"{[f'{g[0]} vs {g[1]}' for g in games]}"
                )
        
        return self.violations


class EqualGamesValidator(DrawOutcomeValidator):
    """Validate that each team plays the required number of games."""
    
    def validate(self) -> List[str]:
        self.violations = []
        
        # Count games per team per grade
        team_game_counts = defaultdict(int)
        
        for game in self.scheduled_games:
            t1, t2 = get_team_names_from_game(game)
            team_game_counts[t1] += 1
            team_game_counts[t2] += 1
        
        # Check against expected
        teams = self.data.get('teams', [])
        num_rounds = self.data.get('num_rounds', {})
        
        for team in teams:
            expected = num_rounds.get(team.grade, 0)
            actual = team_game_counts.get(team.name, 0)
            
            if expected > 0 and actual != expected:
                self.violations.append(
                    f"Team '{team.name}' has {actual} games but expected {expected}"
                )
        
        return self.violations


class BalancedMatchupsValidator(DrawOutcomeValidator):
    """Validate that pair matchups are balanced (base or base+1 times)."""
    
    def validate(self) -> List[str]:
        self.violations = []
        
        # Count matchups per pair per grade
        pair_matchups = defaultdict(int)
        
        for game in self.scheduled_games:
            t1, t2 = get_team_names_from_game(game)
            grade = game[2]
            pair = tuple(sorted([t1, t2]))
            pair_matchups[(grade, pair)] += 1
        
        # Calculate expected bounds per grade
        teams = self.data.get('teams', [])
        num_rounds = self.data.get('num_rounds', {})
        grades = self.data.get('grades', [])
        
        for grade in grades:
            T = grade.num_teams
            R = num_rounds.get(grade.name, 0)
            
            if T < 2 or R == 0:
                continue
            
            if T % 2 == 0:
                base = R // (T - 1)
            else:
                base = R // T
            
            # Check each pair in this grade
            for (g, pair), count in pair_matchups.items():
                if g != grade.name:
                    continue
                
                if count < base:
                    self.violations.append(
                        f"Pair {pair} in grade {g} meets {count} times, expected at least {base}"
                    )
                elif count > base + 1:
                    self.violations.append(
                        f"Pair {pair} in grade {g} meets {count} times, expected at most {base + 1}"
                    )
        
        return self.violations


class FiftyFiftyHomeAwayValidator(DrawOutcomeValidator):
    """Validate that away teams (Maitland, Gosford) have balanced home/away games."""
    
    def validate(self) -> List[str]:
        self.violations = []
        teams = self.data.get('teams', [])
        
        # Track home/away for Maitland and Gosford teams
        for away_location in ['Maitland', 'Gosford']:
            location_field = 'Maitland Park' if away_location == 'Maitland' else 'Central Coast Hockey Park'
            
            team_home_games = defaultdict(lambda: {'home': 0, 'away': 0})
            
            for game in self.scheduled_games:
                t1, t2 = get_team_names_from_game(game)
                field_location = game[10]
                
                for team in [t1, t2]:
                    if away_location in team:
                        other = t2 if team == t1 else t1
                        if away_location in other:
                            continue  # Skip intra-location matchups
                        
                        if field_location == location_field:
                            team_home_games[team]['home'] += 1
                        else:
                            team_home_games[team]['away'] += 1
            
            # Check balance
            for team, counts in team_home_games.items():
                home = counts['home']
                away = counts['away']
                total = home + away
                
                if total > 0:
                    # Check: home*2 >= total-1 and home*2 <= total+1
                    if home * 2 < total - 1:
                        self.violations.append(
                            f"Team '{team}' has too few home games: {home}/{total}"
                        )
                    if home * 2 > total + 1:
                        self.violations.append(
                            f"Team '{team}' has too many home games: {home}/{total}"
                        )
        
        return self.violations


class TeamConflictValidator(DrawOutcomeValidator):
    """Validate that conflicting teams don't play at the same time."""
    
    def validate(self) -> List[str]:
        self.violations = []
        conflicts = self.data.get('team_conflicts', [])
        
        if not conflicts:
            return self.violations
        
        # Group games by (week, day_slot)
        games_per_slot = defaultdict(list)
        
        for game in self.scheduled_games:
            week = game[6]
            day_slot = game[4]
            games_per_slot[(week, day_slot)].append(game)
        
        # Check each timeslot for conflicts
        for (week, day_slot), games in games_per_slot.items():
            teams_playing = set()
            for game in games:
                t1, t2 = get_team_names_from_game(game)
                teams_playing.add(t1)
                teams_playing.add(t2)
            
            for team1, team2 in conflicts:
                if team1 in teams_playing and team2 in teams_playing:
                    self.violations.append(
                        f"Conflicting teams '{team1}' and '{team2}' both play in week {week}, slot {day_slot}"
                    )
        
        return self.violations


class ClubGradeAdjacencyValidator(DrawOutcomeValidator):
    """Validate that adjacent grades from same club don't play simultaneously."""
    
    GRADE_ORDER = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
    
    def validate(self) -> List[str]:
        self.violations = []
        teams = self.data.get('teams', [])
        
        # Build team-to-club mapping
        team_club = {t.name: t.club.name for t in teams}
        
        # Group games by (week, day_slot, field_name)
        games_per_slot = defaultdict(list)
        
        for game in self.scheduled_games:
            week = game[6]
            day_slot = game[4]
            field_name = game[9]
            games_per_slot[(week, day_slot, field_name)].append(game)
        
        # Adjacent grade pairs
        adj_pairs = [(self.GRADE_ORDER[i], self.GRADE_ORDER[i+1]) 
                     for i in range(len(self.GRADE_ORDER)-1)]
        
        # Check each slot
        for (week, day_slot, field), games in games_per_slot.items():
            # Track which clubs have which grades playing
            club_grades = defaultdict(set)
            
            for game in games:
                t1, t2 = get_team_names_from_game(game)
                grade = game[2]
                
                if t1 in team_club:
                    club_grades[team_club[t1]].add(grade)
                if t2 in team_club:
                    club_grades[team_club[t2]].add(grade)
            
            # Check for adjacent grades in same club
            for club, grades in club_grades.items():
                for g1, g2 in adj_pairs:
                    if g1 in grades and g2 in grades:
                        self.violations.append(
                            f"Club '{club}' has adjacent grades {g1} and {g2} playing "
                            f"in week {week}, slot {day_slot}"
                        )
        
        return self.violations


class MaitlandBackToBackValidator(DrawOutcomeValidator):
    """Validate no back-to-back Maitland home weekends."""
    
    def validate(self) -> List[str]:
        self.violations = []
        
        # Find weeks with Maitland home games
        home_weeks = set()
        
        for game in self.scheduled_games:
            t1, t2 = get_team_names_from_game(game)
            if 'Maitland' in t1 or 'Maitland' in t2:
                if game[10] == 'Maitland Park':
                    home_weeks.add(game[6])
        
        # Check for consecutive weeks
        sorted_weeks = sorted(home_weeks)
        for i in range(1, len(sorted_weeks)):
            if sorted_weeks[i] == sorted_weeks[i-1] + 1:
                self.violations.append(
                    f"Back-to-back Maitland home weekends: week {sorted_weeks[i-1]} and {sorted_weeks[i]}"
                )
        
        return self.violations


class AwayClubLimitValidator(DrawOutcomeValidator):
    """Validate maximum 3 away clubs visit Maitland per weekend."""
    
    def validate(self) -> List[str]:
        self.violations = []
        teams = self.data.get('teams', [])
        
        # Track away clubs per week at Maitland
        away_clubs_per_week = defaultdict(set)
        
        for game in self.scheduled_games:
            if game[10] == 'Maitland Park':
                t1, t2 = get_team_names_from_game(game)
                
                for team_name in [t1, t2]:
                    if 'Maitland' not in team_name:
                        club = get_club_from_team(team_name, teams)
                        away_clubs_per_week[game[6]].add(club)
        
        for week, clubs in away_clubs_per_week.items():
            if len(clubs) > 3:
                self.violations.append(
                    f"Week {week} has {len(clubs)} away clubs at Maitland (max 3): {clubs}"
                )
        
        return self.violations


class FieldClubLimitValidator(DrawOutcomeValidator):
    """Validate maximum 5 clubs per field per day at Broadmeadow."""
    
    def validate(self) -> List[str]:
        self.violations = []
        teams = self.data.get('teams', [])
        
        # Track clubs per (week, date, field)
        clubs_per_field_day = defaultdict(set)
        
        for game in self.scheduled_games:
            if game[10] == 'Newcastle International Hockey Centre':
                t1, t2 = get_team_names_from_game(game)
                week = game[6]
                date = game[7]
                field = game[9]
                
                club1 = get_club_from_team(t1, teams)
                club2 = get_club_from_team(t2, teams)
                
                clubs_per_field_day[(week, date, field)].add(club1)
                clubs_per_field_day[(week, date, field)].add(club2)
        
        for (week, date, field), clubs in clubs_per_field_day.items():
            if len(clubs) > 5:
                self.violations.append(
                    f"Week {week}, {date}, field {field} has {len(clubs)} clubs (max 5): {clubs}"
                )
        
        return self.violations


class MatchupSpacingValidator(DrawOutcomeValidator):
    """Validate that matchups are reasonably spaced across rounds."""
    
    SLACK = 1
    
    def validate(self) -> List[str]:
        self.violations = []
        grades = self.data.get('grades', [])
        num_rounds = self.data.get('num_rounds', {})
        
        # Track rounds where each pair meets
        pair_rounds = defaultdict(list)
        
        for game in self.scheduled_games:
            t1, t2 = get_team_names_from_game(game)
            grade = game[2]
            round_no = game[8]
            
            pair = tuple(sorted([t1, t2]))
            pair_rounds[(grade, pair)].append(round_no)
        
        # Check spacing for pairs that meet twice or more
        for (grade, pair), rounds in pair_rounds.items():
            if len(rounds) < 2:
                continue
            
            rounds_sorted = sorted(rounds)
            
            grade_obj = next((g for g in grades if g.name == grade), None)
            if not grade_obj:
                continue
            
            T = grade_obj.num_teams
            R = num_rounds.get(grade, 21)
            expected_spacing = R // T
            
            for i in range(1, len(rounds_sorted)):
                spacing = rounds_sorted[i] - rounds_sorted[i-1]
                min_spacing = max(1, expected_spacing - self.SLACK)
                max_spacing = expected_spacing + self.SLACK
                
                if spacing < min_spacing:
                    self.violations.append(
                        f"Pair {pair} has spacing {spacing} between rounds "
                        f"{rounds_sorted[i-1]} and {rounds_sorted[i]} (min: {min_spacing})"
                    )
        
        return self.violations


class PHLTimingValidator(DrawOutcomeValidator):
    """Validate PHL-specific timing rules."""
    
    GOSFORD_FRIDAY_TARGET = 8  # AGM decision
    
    def validate(self) -> List[str]:
        self.violations = []
        
        # Track PHL games per (week, day, day_slot) at Broadmeadow
        phl_per_slot = defaultdict(list)
        friday_broadmeadow_count = 0
        friday_gosford_count = 0
        
        for game in self.scheduled_games:
            if game[2] == 'PHL':
                week = game[6]
                day = game[3]
                day_slot = game[4]
                location = game[10]
                
                if location == 'Newcastle International Hockey Centre':
                    phl_per_slot[(week, day, day_slot)].append(game)
                    
                    if day == 'Friday':
                        friday_broadmeadow_count += 1
                
                # Track Gosford Friday games
                if location == 'Central Coast Hockey Park' and day == 'Friday':
                    friday_gosford_count += 1
        
        # Check no concurrent PHL at Broadmeadow
        for (week, day, day_slot), games in phl_per_slot.items():
            if len(games) > 1:
                self.violations.append(
                    f"Multiple PHL games at Broadmeadow in week {week}, {day}, slot {day_slot}: "
                    f"{[f'{g[0]} vs {g[1]}' for g in games]}"
                )
        
        # Check max Friday night games at Broadmeadow
        if friday_broadmeadow_count > 3:
            self.violations.append(
                f"Too many Friday PHL games at Broadmeadow: {friday_broadmeadow_count} (max 3)"
            )
        
        # Check exactly 8 Friday night games at Gosford (AGM decision)
        if friday_gosford_count != self.GOSFORD_FRIDAY_TARGET:
            self.violations.append(
                f"Gosford Friday PHL games: {friday_gosford_count} (expected exactly {self.GOSFORD_FRIDAY_TARGET} - AGM decision)"
            )
        
        return self.violations


# ============== Comprehensive Validator ==============

class ComprehensiveDrawValidator:
    """Run all validators and collect all violations."""
    
    VALIDATORS = [
        NoDoubleBookingTeamsValidator,
        NoDoubleBookingFieldsValidator,
        EqualGamesValidator,
        BalancedMatchupsValidator,
        FiftyFiftyHomeAwayValidator,
        TeamConflictValidator,
        ClubGradeAdjacencyValidator,
        MaitlandBackToBackValidator,
        AwayClubLimitValidator,
        FieldClubLimitValidator,
        MatchupSpacingValidator,
        PHLTimingValidator,
    ]
    
    def __init__(self, X_solution: Dict, data: Dict):
        self.X_solution = X_solution
        self.data = data
        self.results = {}
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Run all validators and return results."""
        for validator_class in self.VALIDATORS:
            validator = validator_class(self.X_solution, self.data)
            violations = validator.validate()
            self.results[validator_class.__name__] = violations
        
        return self.results
    
    def is_valid(self) -> bool:
        """Return True if no violations found in any validator."""
        if not self.results:
            self.validate_all()
        return all(len(v) == 0 for v in self.results.values())
    
    def summary(self) -> str:
        """Return a summary of validation results."""
        if not self.results:
            self.validate_all()
        
        lines = ["Draw Validation Summary", "=" * 50]
        total_violations = 0
        
        for validator_name, violations in self.results.items():
            status = "✓ PASS" if len(violations) == 0 else f"✗ FAIL ({len(violations)} violations)"
            lines.append(f"{validator_name}: {status}")
            total_violations += len(violations)
            
            if violations:
                for v in violations[:5]:  # Show first 5 violations
                    lines.append(f"  - {v}")
                if len(violations) > 5:
                    lines.append(f"  ... and {len(violations) - 5} more")
        
        lines.append("=" * 50)
        lines.append(f"Total: {total_violations} violations")
        
        return "\n".join(lines)


# ============== Test Cases ==============

class TestOutcomeValidators:
    """Test the outcome validators with known good and bad draws."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        clubs = [
            Club(name='Tigers', home_field='Newcastle International Hockey Centre'),
            Club(name='Wests', home_field='Newcastle International Hockey Centre'),
            Club(name='Maitland', home_field='Maitland Park'),
        ]
        
        teams = [
            Team(name='Tigers 3rd', club=clubs[0], grade='3rd'),
            Team(name='Wests 3rd', club=clubs[1], grade='3rd'),
            Team(name='Maitland 3rd', club=clubs[2], grade='3rd'),
        ]
        
        grades = [Grade(name='3rd', teams=[t.name for t in teams])]
        
        return {
            'teams': teams,
            'grades': grades,
            'clubs': clubs,
            'num_rounds': {'3rd': 4},
            'team_conflicts': [],
        }

    def test_valid_draw_passes(self, sample_data):
        """Test that a valid draw passes all validators."""
        # Create a valid draw where each team plays each other once in different weeks
        X_solution = {
            ('Tigers 3rd', 'Wests 3rd', '3rd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): 1,
            ('Tigers 3rd', 'Maitland 3rd', '3rd', 'Sunday', 1, '10:00', 2, '2025-03-30', 2, 'EF', 'Newcastle International Hockey Centre'): 1,
            ('Wests 3rd', 'Maitland 3rd', '3rd', 'Sunday', 1, '10:00', 3, '2025-04-06', 3, 'EF', 'Newcastle International Hockey Centre'): 1,
        }
        
        validator = NoDoubleBookingTeamsValidator(X_solution, sample_data)
        assert validator.is_valid()
        
        validator = NoDoubleBookingFieldsValidator(X_solution, sample_data)
        assert validator.is_valid()

    def test_double_booked_team_fails(self, sample_data):
        """Test that a draw with double-booked team fails."""
        # Tigers 3rd plays twice in week 1
        X_solution = {
            ('Tigers 3rd', 'Wests 3rd', '3rd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): 1,
            ('Tigers 3rd', 'Maitland 3rd', '3rd', 'Sunday', 2, '11:30', 1, '2025-03-23', 1, 'WF', 'Newcastle International Hockey Centre'): 1,
        }
        
        validator = NoDoubleBookingTeamsValidator(X_solution, sample_data)
        violations = validator.validate()
        
        assert len(violations) > 0
        assert 'Tigers 3rd' in violations[0]

    def test_double_booked_field_fails(self, sample_data):
        """Test that a draw with double-booked field fails."""
        # Two games on EF at same time
        X_solution = {
            ('Tigers 3rd', 'Wests 3rd', '3rd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): 1,
            ('Tigers 3rd', 'Maitland 3rd', '3rd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): 1,
        }
        
        validator = NoDoubleBookingFieldsValidator(X_solution, sample_data)
        violations = validator.validate()
        
        assert len(violations) > 0

    def test_comprehensive_validator(self, sample_data):
        """Test the comprehensive validator."""
        X_solution = {
            ('Tigers 3rd', 'Wests 3rd', '3rd', 'Sunday', 1, '10:00', 1, '2025-03-23', 1, 'EF', 'Newcastle International Hockey Centre'): 1,
        }
        
        validator = ComprehensiveDrawValidator(X_solution, sample_data)
        results = validator.validate_all()
        
        assert isinstance(results, dict)
        assert 'NoDoubleBookingTeamsValidator' in results
        
        summary = validator.summary()
        assert 'Draw Validation Summary' in summary


# ============== Utility Functions for External Use ==============

def validate_draw(X_solution: Dict, data: Dict) -> ComprehensiveDrawValidator:
    """
    Validate a completed draw against all constraint rules.
    
    Args:
        X_solution: Dictionary of game keys to 0/1 values
        data: Data dictionary with teams, grades, etc.
    
    Returns:
        ComprehensiveDrawValidator with results
    """
    validator = ComprehensiveDrawValidator(X_solution, data)
    validator.validate_all()
    return validator


def print_validation_report(X_solution: Dict, data: Dict):
    """Print a validation report for a draw."""
    validator = validate_draw(X_solution, data)
    print(validator.summary())
    return validator.is_valid()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
