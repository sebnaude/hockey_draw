# analytics/tester.py
"""
Draw Testing and Modification Tool.

This module provides utilities for:
1. Loading and modifying draws
2. Running constraint violation tests
3. Generating violation reports
4. What-if analysis for game moves

Usage:
    # Load a draw
    tester = DrawTester.from_file("draw.json", data)
    
    # Move a game and test
    tester.move_game("G00123", new_week=5, new_day_slot=2, new_time="14:00")
    report = tester.run_violation_check()
    print(report)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.storage import DrawStorage, StoredGame
from models import Team, Club, Grade


@dataclass
class Violation:
    """Represents a single constraint violation."""
    constraint: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    message: str
    affected_games: List[str] = field(default_factory=list)
    week: Optional[int] = None
    
    def __str__(self) -> str:
        games_str = f" [{', '.join(self.affected_games)}]" if self.affected_games else ""
        return f"[{self.severity}] {self.constraint}: {self.message}{games_str}"


@dataclass
class ViolationReport:
    """Complete violation report for a draw."""
    draw_description: str
    total_games: int
    violations: List[Violation] = field(default_factory=list)
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'CRITICAL')
    
    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'HIGH')
    
    @property
    def medium_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'MEDIUM')
    
    @property
    def low_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'LOW')
    
    def summary(self) -> str:
        """Return a concise summary."""
        if not self.has_violations:
            return f"✅ PASS: No violations found in {self.total_games} games."
        
        return (
            f"❌ FAIL: {len(self.violations)} violations found\n"
            f"   CRITICAL: {self.critical_count}\n"
            f"   HIGH: {self.high_count}\n"
            f"   MEDIUM: {self.medium_count}\n"
            f"   LOW: {self.low_count}"
        )
    
    def full_report(self) -> str:
        """Return a detailed report."""
        lines = [
            "=" * 60,
            "CONSTRAINT VIOLATION REPORT",
            "=" * 60,
            f"Draw: {self.draw_description}",
            f"Total Games: {self.total_games}",
            f"Total Violations: {len(self.violations)}",
            "-" * 60
        ]
        
        if not self.violations:
            lines.append("✅ ALL CONSTRAINTS SATISFIED")
        else:
            # Group by severity
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                sev_violations = [v for v in self.violations if v.severity == severity]
                if sev_violations:
                    lines.append(f"\n{severity} ({len(sev_violations)}):")
                    lines.append("-" * 40)
                    for v in sev_violations:
                        lines.append(f"  • {v.constraint}")
                        lines.append(f"    {v.message}")
                        if v.affected_games:
                            lines.append(f"    Games: {', '.join(v.affected_games[:5])}")
                            if len(v.affected_games) > 5:
                                lines.append(f"    ... and {len(v.affected_games) - 5} more")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def by_constraint(self) -> Dict[str, List[Violation]]:
        """Group violations by constraint name."""
        result = defaultdict(list)
        for v in self.violations:
            result[v.constraint].append(v)
        return dict(result)


class DrawTester:
    """Main class for testing and modifying draws."""
    
    GRADE_ORDER = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]
    
    def __init__(self, draw: DrawStorage, data: Dict):
        """
        Initialize tester with a draw and data.
        
        Args:
            draw: DrawStorage object (will be copied for modification)
            data: Data dict containing teams, grades, clubs, etc.
        """
        # Deep copy to avoid modifying original
        self.original_draw = draw
        self.draw = DrawStorage(**draw.model_dump())
        self.data = data
        self.teams: List[Team] = data.get('teams', [])
        self.grades: List[Grade] = data.get('grades', [])
        self.clubs: List[Club] = data.get('clubs', [])
        
        # Build lookups
        self._team_to_club = {t.name: t.club.name for t in self.teams}
        self._team_to_grade = {t.name: t.grade for t in self.teams}
        
        # Modification log
        self.modifications: List[str] = []
    
    @classmethod
    def from_file(cls, path: str, data: Dict) -> "DrawTester":
        """Create tester from a saved draw file."""
        draw = DrawStorage.load(path)
        return cls(draw, data)
    
    @classmethod
    def from_X_solution(cls, X_solution: Dict, data: Dict, description: str = "") -> "DrawTester":
        """Create tester from X solution dict."""
        draw = DrawStorage.from_X_solution(X_solution, description)
        return cls(draw, data)
    
    def reset(self) -> None:
        """Reset to original draw, discarding modifications."""
        self.draw = DrawStorage(**self.original_draw.model_dump())
        self.modifications = []
    
    # ============== Game Modification Methods ==============
    
    def move_game(
        self,
        game_id: str,
        new_week: Optional[int] = None,
        new_day: Optional[str] = None,
        new_day_slot: Optional[int] = None,
        new_time: Optional[str] = None,
        new_date: Optional[str] = None,
        new_field_name: Optional[str] = None,
        new_field_location: Optional[str] = None
    ) -> bool:
        """
        Move a game to a new timeslot.
        
        Args:
            game_id: The game ID to move
            new_week: Optional new week number
            new_day: Optional new day name
            new_day_slot: Optional new day slot
            new_time: Optional new time
            new_date: Optional new date
            new_field_name: Optional new field name
            new_field_location: Optional new field location
        
        Returns:
            True if game was found and modified, False otherwise
        """
        for i, game in enumerate(self.draw.games):
            if game.game_id == game_id:
                # Create modified game
                game_dict = game.model_dump()
                
                if new_week is not None:
                    game_dict['week'] = new_week
                if new_day is not None:
                    game_dict['day'] = new_day
                if new_day_slot is not None:
                    game_dict['day_slot'] = new_day_slot
                if new_time is not None:
                    game_dict['time'] = new_time
                if new_date is not None:
                    game_dict['date'] = new_date
                if new_field_name is not None:
                    game_dict['field_name'] = new_field_name
                if new_field_location is not None:
                    game_dict['field_location'] = new_field_location
                
                self.draw.games[i] = StoredGame(**game_dict)
                
                # Log modification
                mod_desc = f"Moved game {game_id}"
                changes = []
                if new_week is not None:
                    changes.append(f"week={new_week}")
                if new_day is not None:
                    changes.append(f"day={new_day}")
                if new_day_slot is not None:
                    changes.append(f"slot={new_day_slot}")
                if new_time is not None:
                    changes.append(f"time={new_time}")
                if new_field_name is not None:
                    changes.append(f"field={new_field_name}")
                
                self.modifications.append(f"{mod_desc}: {', '.join(changes)}")
                return True
        
        return False
    
    def swap_games(self, game_id_1: str, game_id_2: str) -> bool:
        """
        Swap the timeslots of two games.
        
        Returns:
            True if both games were found and swapped, False otherwise
        """
        game1 = None
        game2 = None
        idx1 = idx2 = -1
        
        for i, game in enumerate(self.draw.games):
            if game.game_id == game_id_1:
                game1 = game
                idx1 = i
            elif game.game_id == game_id_2:
                game2 = game
                idx2 = i
        
        if game1 is None or game2 is None:
            return False
        
        # Swap timeslot attributes
        swap_attrs = ['week', 'round_no', 'date', 'day', 'time', 'day_slot', 'field_name', 'field_location']
        
        g1_dict = game1.model_dump()
        g2_dict = game2.model_dump()
        
        for attr in swap_attrs:
            g1_dict[attr], g2_dict[attr] = g2_dict[attr], g1_dict[attr]
        
        self.draw.games[idx1] = StoredGame(**g1_dict)
        self.draw.games[idx2] = StoredGame(**g2_dict)
        
        self.modifications.append(f"Swapped games {game_id_1} and {game_id_2}")
        return True
    
    def find_game(
        self,
        team: Optional[str] = None,
        opponent: Optional[str] = None,
        week: Optional[int] = None,
        grade: Optional[str] = None
    ) -> List[StoredGame]:
        """Find games matching criteria."""
        results = []
        for game in self.draw.games:
            if team and team not in (game.team1, game.team2):
                continue
            if opponent and opponent not in (game.team1, game.team2):
                continue
            if team and opponent and not (
                (game.team1 == team and game.team2 == opponent) or
                (game.team2 == team and game.team1 == opponent)
            ):
                continue
            if week is not None and game.week != week:
                continue
            if grade and game.grade != grade:
                continue
            results.append(game)
        return results
    
    def find_available_slots(self, week: int, field_location: Optional[str] = None) -> List[Dict]:
        """
        Find available timeslots in a given week.
        
        Args:
            week: Week number to check
            field_location: Optional filter by location
            
        Returns:
            List of dicts with available slot information from timeslots config.
        """
        # Get all possible slots from data if available
        timeslots = self.data.get('timeslots', [])
        
        # Get slots used in this week
        used_slot_keys = set()
        for game in self.draw.games:
            if game.week == week:
                key = (game.day_slot, game.field_name, game.field_location)
                used_slot_keys.add(key)
        
        available = []
        for t in timeslots:
            if t.week != week:
                continue
            if field_location and t.field.location != field_location:
                continue
            
            key = (t.day_slot, t.field.name, t.field.location)
            if key not in used_slot_keys:
                available.append({
                    'week': t.week,
                    'day': t.day,
                    'day_slot': t.day_slot,
                    'time': t.time,
                    'date': t.date,
                    'round_no': t.round_no,
                    'field_name': t.field.name,
                    'field_location': t.field.location
                })
        
        return available
    
    def move_game_to_available_slot(
        self,
        game_id: str,
        target_week: int,
        target_day_slot: Optional[int] = None,
        target_field_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Move a game to an available slot.
        
        If day_slot or field_name not specified, picks first available.
        
        Args:
            game_id: ID of game to move
            target_week: Week to move the game to
            target_day_slot: Optional specific slot (auto-selects if None)
            target_field_name: Optional specific field (auto-selects if None)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Find the game
        game = None
        for g in self.draw.games:
            if g.game_id == game_id:
                game = g
                break
        
        if not game:
            return False, f"Game {game_id} not found"
        
        # Get available slots in target week
        available = self.find_available_slots(target_week)
        
        if not available:
            return False, f"No available slots in week {target_week}"
        
        # Filter by preferences if specified
        if target_day_slot is not None:
            available = [s for s in available if s['day_slot'] == target_day_slot]
        if target_field_name is not None:
            available = [s for s in available if s['field_name'] == target_field_name]
        
        if not available:
            return False, f"No matching available slots in week {target_week}"
        
        # Pick first available slot
        slot = available[0]
        
        # Move the game
        success = self.move_game(
            game_id,
            new_week=slot['week'],
            new_day=slot['day'],
            new_day_slot=slot['day_slot'],
            new_time=slot['time'],
            new_date=slot['date'],
            new_field_name=slot['field_name'],
            new_field_location=slot['field_location']
        )
        
        if success:
            msg = f"Moved {game_id} to week {target_week}, slot {slot['day_slot']}, {slot['field_name']}"
            return True, msg
        else:
            return False, "Failed to move game"
    
    def list_unused_slots(self, week: Optional[int] = None) -> List[Dict]:
        """
        List all unused slots, optionally filtered by week.
        
        Args:
            week: Optional week filter
            
        Returns:
            List of unused slot dicts
        """
        if week is not None:
            return self.find_available_slots(week)
        
        # Get all weeks
        weeks = sorted(set(g.week for g in self.draw.games))
        all_unused = []
        for w in weeks:
            all_unused.extend(self.find_available_slots(w))
        return all_unused
    
    def print_unused_slots(self, week: Optional[int] = None) -> None:
        """Print unused slots to console."""
        from collections import defaultdict
        
        if week is not None:
            unused = self.find_available_slots(week)
        else:
            weeks = sorted(set(t.week for t in self.data.get('timeslots', [])))
            unused = []
            for w in weeks:
                unused.extend(self.find_available_slots(w))
        
        if not unused:
            print("No unused slots found" + (f" in week {week}" if week else ""))
            return
        
        print(f"\n{'='*60}")
        print(f"UNUSED SLOTS" + (f" - Week {week}" if week else " - All Weeks"))
        print(f"{'='*60}")
        
        by_week = defaultdict(list)
        for slot in unused:
            by_week[slot['week']].append(slot)
        
        for wk in sorted(by_week.keys()):
            print(f"\nWeek {wk}:")
            for slot in sorted(by_week[wk], key=lambda x: (x['day_slot'], x['field_name'])):
                print(f"  Slot {slot['day_slot']}: {slot['field_name']} @ {slot['field_location']} "
                      f"({slot['day']} {slot['time']})")
        
        print(f"\nTotal unused: {len(unused)} slots")
    
    # ============== Constraint Checking ==============
    
    def run_violation_check(self) -> ViolationReport:
        """Run all constraint checks and return violation report."""
        violations = []
        
        # Run all checks
        violations.extend(self._check_no_double_booking_teams())
        violations.extend(self._check_no_double_booking_fields())
        violations.extend(self._check_equal_games())
        violations.extend(self._check_balanced_matchups())
        violations.extend(self._check_fifty_fifty_home_away())
        violations.extend(self._check_maitland_back_to_back())
        violations.extend(self._check_maitland_away_clubs_limit())
        violations.extend(self._check_club_grade_adjacency())
        violations.extend(self._check_phl_second_grade_adjacency())
        violations.extend(self._check_phl_second_grade_times())
        
        return ViolationReport(
            draw_description=self.draw.description or "Modified Draw",
            total_games=len(self.draw.games),
            violations=violations
        )
    
    def _check_no_double_booking_teams(self) -> List[Violation]:
        """Check that no team plays more than once per week."""
        violations = []
        team_games_per_week = defaultdict(list)
        
        for game in self.draw.games:
            team_games_per_week[(game.week, game.team1)].append(game.game_id)
            team_games_per_week[(game.week, game.team2)].append(game.game_id)
        
        for (week, team), games in team_games_per_week.items():
            if len(games) > 1:
                violations.append(Violation(
                    constraint="NoDoubleBookingTeams",
                    severity="CRITICAL",
                    message=f"Team '{team}' plays {len(games)} games in week {week}",
                    affected_games=games,
                    week=week
                ))
        
        return violations
    
    def _check_no_double_booking_fields(self) -> List[Violation]:
        """Check that no field hosts more than one game per slot."""
        violations = []
        field_games = defaultdict(list)
        
        for game in self.draw.games:
            key = (game.week, game.day_slot, game.field_name)
            field_games[key].append(game.game_id)
        
        for (week, slot, field), games in field_games.items():
            if len(games) > 1:
                violations.append(Violation(
                    constraint="NoDoubleBookingFields",
                    severity="CRITICAL",
                    message=f"Field '{field}' has {len(games)} games in week {week}, slot {slot}",
                    affected_games=games,
                    week=week
                ))
        
        return violations
    
    def _check_equal_games(self) -> List[Violation]:
        """Check each team plays the expected number of games."""
        violations = []
        team_counts = defaultdict(int)
        
        for game in self.draw.games:
            team_counts[game.team1] += 1
            team_counts[game.team2] += 1
        
        num_rounds = self.data.get('num_rounds', {})
        
        for team in self.teams:
            expected = num_rounds.get(team.grade, 0)
            actual = team_counts.get(team.name, 0)
            
            if expected > 0 and actual != expected:
                violations.append(Violation(
                    constraint="EqualGames",
                    severity="HIGH",
                    message=f"Team '{team.name}' has {actual} games (expected {expected})"
                ))
        
        return violations
    
    def _check_balanced_matchups(self) -> List[Violation]:
        """Check pair matchups are balanced (base or base+1)."""
        violations = []
        pair_counts = defaultdict(int)
        
        for game in self.draw.games:
            pair = tuple(sorted([game.team1, game.team2]))
            pair_counts[(game.grade, pair)] += 1
        
        num_rounds = self.data.get('num_rounds', {})
        
        for grade in self.grades:
            T = grade.num_teams
            R = num_rounds.get(grade.name, 0)
            
            if T < 2 or R == 0:
                continue
            
            base = R // (T - 1) if T % 2 == 0 else R // T
            
            for (g, pair), count in pair_counts.items():
                if g != grade.name:
                    continue
                
                if count < base:
                    violations.append(Violation(
                        constraint="BalancedMatchups",
                        severity="MEDIUM",
                        message=f"Pair {pair} in {g} meets {count} times (min {base})"
                    ))
                elif count > base + 1:
                    violations.append(Violation(
                        constraint="BalancedMatchups",
                        severity="MEDIUM",
                        message=f"Pair {pair} in {g} meets {count} times (max {base+1})"
                    ))
        
        return violations
    
    def _check_fifty_fifty_home_away(self) -> List[Violation]:
        """Check Maitland/Gosford teams have balanced home/away."""
        violations = []
        
        for away_prefix, home_field in [('Maitland', 'Maitland Park'), ('Gosford', 'Central Coast Hockey Park')]:
            team_balance = defaultdict(lambda: {'home': 0, 'away': 0})
            
            for game in self.draw.games:
                for team in [game.team1, game.team2]:
                    if away_prefix in team:
                        other = game.team2 if team == game.team1 else game.team1
                        if away_prefix in other:
                            continue
                        
                        if game.field_location == home_field:
                            team_balance[team]['home'] += 1
                        else:
                            team_balance[team]['away'] += 1
            
            for team, counts in team_balance.items():
                home = counts['home']
                away = counts['away']
                total = home + away
                
                if total > 0 and abs(home - away) > 1:
                    violations.append(Violation(
                        constraint="FiftyFiftyHomeAway",
                        severity="HIGH",
                        message=f"Team '{team}' has {home} home, {away} away (imbalanced)"
                    ))
        
        return violations
    
    def _check_maitland_back_to_back(self) -> List[Violation]:
        """Check no back-to-back Maitland home weekends."""
        violations = []
        home_weeks = set()
        
        for game in self.draw.games:
            if ('Maitland' in game.team1 or 'Maitland' in game.team2):
                if game.field_location == 'Maitland Park':
                    home_weeks.add(game.week)
        
        sorted_weeks = sorted(home_weeks)
        for i in range(1, len(sorted_weeks)):
            if sorted_weeks[i] == sorted_weeks[i-1] + 1:
                violations.append(Violation(
                    constraint="MaxMaitlandHomeWeekends",
                    severity="HIGH",
                    message=f"Back-to-back Maitland home: weeks {sorted_weeks[i-1]} and {sorted_weeks[i]}",
                    week=sorted_weeks[i]
                ))
        
        return violations
    
    def _check_maitland_away_clubs_limit(self) -> List[Violation]:
        """Check max 3 away clubs at Maitland per weekend."""
        violations = []
        away_clubs_per_week = defaultdict(set)
        
        for game in self.draw.games:
            if game.field_location == 'Maitland Park':
                for team in [game.team1, game.team2]:
                    if 'Maitland' not in team:
                        club = self._team_to_club.get(team, 'Unknown')
                        away_clubs_per_week[game.week].add(club)
        
        for week, clubs in away_clubs_per_week.items():
            if len(clubs) > 3:
                violations.append(Violation(
                    constraint="AwayAtMaitlandGrouping",
                    severity="MEDIUM",
                    message=f"Week {week}: {len(clubs)} away clubs at Maitland (max 3): {clubs}",
                    week=week
                ))
        
        return violations
    
    def _check_club_grade_adjacency(self) -> List[Violation]:
        """Check adjacent grades from same club don't play simultaneously."""
        violations = []
        
        adj_pairs = [(self.GRADE_ORDER[i], self.GRADE_ORDER[i+1]) 
                     for i in range(len(self.GRADE_ORDER)-1)]
        
        games_per_slot = defaultdict(list)
        for game in self.draw.games:
            games_per_slot[(game.week, game.day_slot, game.field_name)].append(game)
        
        for (week, slot, field), games in games_per_slot.items():
            club_grades = defaultdict(set)
            
            for game in games:
                for team in [game.team1, game.team2]:
                    club = self._team_to_club.get(team)
                    if club:
                        club_grades[club].add(game.grade)
            
            for club, grades in club_grades.items():
                for g1, g2 in adj_pairs:
                    if g1 in grades and g2 in grades:
                        violations.append(Violation(
                            constraint="ClubGradeAdjacency",
                            severity="MEDIUM",
                            message=f"Club '{club}' has adjacent grades {g1}/{g2} at week {week}, slot {slot}"
                        ))
        
        return violations
    
    def _check_phl_second_grade_adjacency(self) -> List[Violation]:
        """Check PHL and 2nd grade from same club play adjacent slots."""
        violations = []
        
        # Group games by (week, club, grade)
        club_games = defaultdict(list)
        for game in self.draw.games:
            for team in [game.team1, game.team2]:
                club = self._team_to_club.get(team)
                if club:
                    club_games[(game.week, club, game.grade)].append(game)
        
        # Find weeks where club has both PHL and 2nd
        weeks_clubs = defaultdict(dict)
        for (week, club, grade), games in club_games.items():
            if grade in ['PHL', '2nd']:
                weeks_clubs[(week, club)][grade] = games
        
        for (week, club), grade_games in weeks_clubs.items():
            if 'PHL' in grade_games and '2nd' in grade_games:
                phl_slots = {g.day_slot for g in grade_games['PHL']}
                second_slots = {g.day_slot for g in grade_games['2nd']}
                
                adjacent = any(
                    abs(p - s) == 1 
                    for p in phl_slots 
                    for s in second_slots
                )
                
                if not adjacent:
                    violations.append(Violation(
                        constraint="PHLAndSecondGradeAdjacency",
                        severity="LOW",
                        message=f"Club '{club}' week {week}: PHL (slots {phl_slots}) and 2nd (slots {second_slots}) not adjacent",
                        week=week
                    ))
        
        return violations
    
    def _check_phl_second_grade_times(self) -> List[Violation]:
        """Check PHL plays before 2nd grade for same club."""
        violations = []
        
        club_games = defaultdict(list)
        for game in self.draw.games:
            for team in [game.team1, game.team2]:
                club = self._team_to_club.get(team)
                if club:
                    club_games[(game.week, club, game.grade)].append(game)
        
        weeks_clubs = defaultdict(dict)
        for (week, club, grade), games in club_games.items():
            if grade in ['PHL', '2nd']:
                weeks_clubs[(week, club)][grade] = games
        
        for (week, club), grade_games in weeks_clubs.items():
            if 'PHL' in grade_games and '2nd' in grade_games:
                phl_min_slot = min(g.day_slot for g in grade_games['PHL'])
                second_max_slot = max(g.day_slot for g in grade_games['2nd'])
                
                if phl_min_slot > second_max_slot:
                    violations.append(Violation(
                        constraint="PHLAndSecondGradeTimes",
                        severity="LOW",
                        message=f"Club '{club}' week {week}: PHL plays after 2nd grade",
                        week=week
                    ))
        
        return violations
    
    # ============== Reporting ==============
    
    def print_modifications(self) -> None:
        """Print all modifications made to the draw."""
        if not self.modifications:
            print("No modifications made.")
            return
        
        print(f"\n{'='*40}")
        print(f"MODIFICATIONS ({len(self.modifications)})")
        print('='*40)
        for i, mod in enumerate(self.modifications, 1):
            print(f"  {i}. {mod}")
    
    def save_modified_draw(self, path: str) -> None:
        """Save the modified draw to a file."""
        self.draw.description = f"{self.draw.description} (Modified: {len(self.modifications)} changes)"
        self.draw.save(path)


# ============== Convenience Functions ==============

def test_draw(draw_path: str, data: Dict) -> ViolationReport:
    """Load a draw and run violation checks."""
    tester = DrawTester.from_file(draw_path, data)
    return tester.run_violation_check()


def test_solution(X_solution: Dict, data: Dict) -> ViolationReport:
    """Test an X solution dict for violations."""
    tester = DrawTester.from_X_solution(X_solution, data)
    return tester.run_violation_check()


def what_if_move_game(
    draw_path: str,
    data: Dict,
    game_id: str,
    **new_slot_kwargs
) -> ViolationReport:
    """
    Test what happens if a game is moved.
    
    Args:
        draw_path: Path to draw JSON file
        data: Data dictionary
        game_id: ID of game to move
        **new_slot_kwargs: New slot parameters (new_week, new_day_slot, etc.)
    
    Returns:
        ViolationReport showing any violations caused by the move
    """
    tester = DrawTester.from_file(draw_path, data)
    
    success = tester.move_game(game_id, **new_slot_kwargs)
    if not success:
        raise ValueError(f"Game {game_id} not found in draw")
    
    return tester.run_violation_check()


def compare_draws(
    draw1_path: str,
    draw2_path: str,
    data: Dict
) -> Tuple[ViolationReport, ViolationReport]:
    """Compare violation reports for two draws."""
    report1 = test_draw(draw1_path, data)
    report2 = test_draw(draw2_path, data)
    return report1, report2
