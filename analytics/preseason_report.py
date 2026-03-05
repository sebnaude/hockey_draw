# analytics/preseason_report.py
"""
Pre-Season Configuration Report.

This module generates a comprehensive report of the season configuration BEFORE
the solver runs. It validates configuration data and shows:
- Teams per grade (with full names)
- Calculated rounds vs configured rounds (with validation)
- Special requests (club days, no-play dates, team conflicts)
- Special event days
- Available field times per venue

Usage:
    python run.py preseason --year 2026
    python run.py preseason --year 2026 --output preseason_2026.txt
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class PreSeasonReport:
    """
    Generate a comprehensive pre-season configuration report.
    
    This report is generated from the season config and data files BEFORE
    the solver runs. It helps validate that all configuration is correct.
    """
    
    def __init__(self, data: Dict, config: Dict):
        """
        Initialize report generator.
        
        Args:
            data: Data dictionary from build_season_data()
            config: SEASON_CONFIG dictionary from season file
        """
        self.data = data
        self.config = config
        self.teams = data.get('teams', [])
        self.grades = data.get('grades', [])
        self.fields = data.get('fields', [])
        self.clubs = data.get('clubs', [])
        self.timeslots = data.get('timeslots', [])
        
        # Config values
        self.year = config.get('year', data.get('year'))
        self.start_date = config.get('start_date')
        self.last_round_date = config.get('last_round_date')  # End of regular rounds
        self.end_date = config.get('end_date')  # Grand final date
        self.max_rounds = config.get('max_rounds')
        
    def calculate_available_weekends(self) -> Tuple[int, List[datetime]]:
        """
        Calculate the number of available playing weekends.
        
        Returns:
            Tuple of (count, list_of_weekend_dates)
        """
        # Use last_round_date if available, otherwise fall back to end_date
        round_end_date = self.last_round_date or self.end_date
        if not self.start_date or not round_end_date:
            return 0, []
        
        field_unavailabilities = self.config.get('field_unavailabilities', {})
        
        # Get all blocked weekend dates (using NIHC as reference for rounds)
        blocked_weekends = set()
        nihc_unavail = field_unavailabilities.get('Newcastle International Hockey Centre', {})
        for weekend in nihc_unavail.get('weekends', []):
            blocked_weekends.add(weekend.date())
        
        # Count Sundays between start and last round date
        current = self.start_date
        weekends = []
        while current <= round_end_date:
            if current.strftime('%A') == 'Sunday':
                # Check if this weekend is blocked
                # Weekend is blocked if any day Fri-Sun matches a blocked weekend
                weekend_dates = [
                    (current - timedelta(days=2)).date(),  # Friday
                    (current - timedelta(days=1)).date(),  # Saturday
                    current.date()  # Sunday
                ]
                is_blocked = any(
                    d in blocked_weekends or 
                    (current - timedelta(days=1)).date() in blocked_weekends or 
                    (current + timedelta(days=1)).date() in blocked_weekends
                    for d in weekend_dates
                )
                # Actually check the weekend marker date (usually Saturday)
                weekend_marker = (current - timedelta(days=1)).date()  # Saturday
                is_blocked = weekend_marker in blocked_weekends
                
                if not is_blocked:
                    weekends.append(current)
            current += timedelta(days=1)
        
        return len(weekends), weekends
    
    def validate_rounds(self) -> Dict[str, Any]:
        """
        Validate that calculated rounds match configured max_rounds.
        
        Returns:
            Dict with validation results including any warnings
        """
        available_count, weekends = self.calculate_available_weekends()
        
        # Get blocked weekends for the report
        field_unavailabilities = self.config.get('field_unavailabilities', {})
        nihc_unavail = field_unavailabilities.get('Newcastle International Hockey Centre', {})
        blocked_weekends = nihc_unavail.get('weekends', [])
        
        configured_rounds = self.max_rounds
        
        # Check if they match
        matches = available_count >= configured_rounds
        
        return {
            'available_weekends': available_count,
            'configured_max_rounds': configured_rounds,
            'blocked_weekends': len(blocked_weekends),
            'blocked_dates': [d.strftime('%Y-%m-%d') for d in blocked_weekends],
            'valid': matches,
            'warning': None if matches else f"WARNING: Only {available_count} weekends available but {configured_rounds} rounds configured!"
        }
    
    def get_teams_by_grade(self) -> Dict[str, List[str]]:
        """
        Get teams organized by grade with full team names.
        
        Returns:
            Dict mapping grade name to list of team names
        """
        result = {}
        for grade in self.grades:
            # Sort teams by name for consistent display
            result[grade.name] = sorted(grade.teams)
        return result
    
    def get_special_requests(self) -> Dict[str, Any]:
        """
        Get all special requests organized by type.
        
        Returns:
            Dict with categories: club_days, no_play_dates, team_conflicts, etc.
        """
        result = {
            'club_days': [],
            'no_play_dates': [],
            'team_conflicts': [],
            'friday_night_allocations': {},
            'special_games': [],
        }
        
        # Club Days
        club_days = self.config.get('club_days', {})
        for club, date in club_days.items():
            if isinstance(date, datetime):
                result['club_days'].append({
                    'club': club,
                    'date': date.strftime('%Y-%m-%d'),
                    'description': f"All teams back-to-back on same field"
                })
        
        # No-Play Preferences (soft constraints)
        preference_no_play = self.config.get('preference_no_play', {})
        for key, pref in preference_no_play.items():
            dates = pref.get('dates', [])
            result['no_play_dates'].append({
                'key': key,
                'club': pref.get('club', 'Unknown'),
                'grade': pref.get('grade', pref.get('grades', 'All')),
                'dates': [d.strftime('%Y-%m-%d') for d in dates] if dates else [],
                'reason': pref.get('reason', ''),
                'type': 'soft'
            })
        
        # Field Unavailabilities (hard constraints) - these affect all teams
        field_unavailabilities = self.config.get('field_unavailabilities', {})
        for field, unavail in field_unavailabilities.items():
            for weekend in unavail.get('weekends', []):
                result['no_play_dates'].append({
                    'key': f'{field}_weekend_{weekend.strftime("%Y%m%d")}',
                    'club': 'ALL',
                    'grade': 'ALL',
                    'dates': [weekend.strftime('%Y-%m-%d')],
                    'reason': f'Field unavailable: {field}',
                    'type': 'hard'
                })
            for day in unavail.get('whole_days', []):
                result['no_play_dates'].append({
                    'key': f'{field}_day_{day.strftime("%Y%m%d")}',
                    'club': 'ALL',
                    'grade': 'ALL',
                    'dates': [day.strftime('%Y-%m-%d')],
                    'reason': f'Field unavailable: {field}',
                    'type': 'hard'
                })
        
        # Friday Night Configuration
        friday_config = self.config.get('friday_night_config', {})
        if friday_config:
            result['friday_night_allocations'] = {
                'total_matches': friday_config.get('gosford_friday_count', 0),
                'clubs': friday_config.get('friday_clubs', {}),
                'dates': [d.strftime('%Y-%m-%d') for d in friday_config.get('friday_dates', [])],
            }
        
        # Special Games
        special_games = self.config.get('special_games', {})
        for key, game in special_games.items():
            result['special_games'].append({
                'key': key,
                'teams': game.get('teams', []),
                'grades': game.get('grades', []),
                'date': game.get('date').strftime('%Y-%m-%d') if game.get('date') else 'TBC',
                'month': game.get('month'),
            })
        
        return result
    
    def get_special_events(self) -> List[Dict[str, Any]]:
        """
        Get list of special event days.
        
        Returns:
            List of special events with dates and descriptions
        """
        events = []
        
        # Club Days
        club_days = self.config.get('club_days', {})
        for club, date in club_days.items():
            if isinstance(date, datetime):
                events.append({
                    'name': f'{club} Club Day',
                    'date': date.strftime('%Y-%m-%d'),
                    'type': 'club_day'
                })
        
        # Friday Night special dates
        friday_config = self.config.get('friday_night_config', {})
        for date in friday_config.get('friday_dates', []):
            # Check if it's a special one (e.g., Norths 80th)
            events.append({
                'name': f'Friday Night PHL at Gosford',
                'date': date.strftime('%Y-%m-%d'),
                'type': 'friday_night'
            })
        
        # Special Games
        special_games = self.config.get('special_games', {})
        for key, game in special_games.items():
            if game.get('date'):
                events.append({
                    'name': key.replace('_', ' ').title(),
                    'date': game['date'].strftime('%Y-%m-%d'),
                    'type': 'special_game'
                })
        
        # ANZAC Sunday (if playing)
        if self.config.get('play_anzac_sunday', False):
            events.append({
                'name': 'ANZAC Sunday (playing)',
                'date': '2026-04-26',  # Day after ANZAC Day
                'type': 'special_day'
            })
        
        return sorted(events, key=lambda x: x['date'])
    
    def get_venue_times(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get available game times per venue (default slots before exclusions).
        
        Returns:
            Dict mapping venue -> day -> list of times
        """
        day_time_map = self.config.get('day_time_map', {})
        phl_game_times = self.config.get('phl_game_times', {})
        
        result = {}
        
        # Standard game times
        for venue, days in day_time_map.items():
            if venue not in result:
                result[venue] = {'standard': {}, 'phl': {}}
            for day, times in days.items():
                result[venue]['standard'][day] = [t.strftime('%H:%M') for t in times]
        
        # PHL-specific times
        for venue, days in phl_game_times.items():
            if venue not in result:
                result[venue] = {'standard': {}, 'phl': {}}
            for day, times in days.items():
                result[venue]['phl'][day] = [t.strftime('%H:%M') for t in times]
        
        return result
    
    def get_field_unavailabilities_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of field unavailabilities.
        
        Returns:
            Dict mapping venue -> list of unavailable date descriptions
        """
        result = {}
        field_unavailabilities = self.config.get('field_unavailabilities', {})
        
        for field, unavail in field_unavailabilities.items():
            dates = []
            for weekend in unavail.get('weekends', []):
                dates.append(f"Weekend of {weekend.strftime('%Y-%m-%d')} (blocked)")
            for day in unavail.get('whole_days', []):
                dates.append(f"{day.strftime('%Y-%m-%d')} (whole day)")
            for part in unavail.get('part_days', []):
                dates.append(f"{part.strftime('%Y-%m-%d %H:%M')} (part day)")
            result[field] = dates
        
        return result
    
    def generate_text_report(self) -> str:
        """
        Generate full text report.
        
        Returns:
            Formatted text report string
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"PRE-SEASON CONFIGURATION REPORT - {self.year}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Season Dates
        lines.append("-" * 40)
        lines.append("SEASON DATES")
        lines.append("-" * 40)
        if self.start_date:
            lines.append(f"First Round:  {self.start_date.strftime('%A, %d %B %Y')}")
        if self.last_round_date:
            lines.append(f"Last Round:   {self.last_round_date.strftime('%A, %d %B %Y')}")
        if self.end_date:
            lines.append(f"Grand Final:  {self.end_date.strftime('%A, %d %B %Y')}")
        lines.append("")
        
        # Rounds Validation
        lines.append("-" * 40)
        lines.append("ROUNDS CALCULATION & VALIDATION")
        lines.append("-" * 40)
        validation = self.validate_rounds()
        lines.append(f"Configured Max Rounds: {validation['configured_max_rounds']}")
        lines.append(f"Available Weekends:    {validation['available_weekends']}")
        lines.append(f"Blocked Weekends:      {validation['blocked_weekends']}")
        if validation['blocked_dates']:
            lines.append(f"  Blocked dates: {', '.join(validation['blocked_dates'])}")
        if validation['valid']:
            lines.append("✓ VALID: Sufficient weekends available for configured rounds")
        else:
            lines.append(f"✗ {validation['warning']}")
        lines.append("")
        
        # Teams by Grade
        lines.append("-" * 40)
        lines.append("TEAMS BY GRADE")
        lines.append("-" * 40)
        teams_by_grade = self.get_teams_by_grade()
        total_teams = 0
        for grade_name, team_names in teams_by_grade.items():
            total_teams += len(team_names)
            lines.append(f"\n{grade_name} Grade ({len(team_names)} teams):")
            for i, team in enumerate(team_names, 1):
                lines.append(f"  {i:2}. {team}")
        lines.append(f"\nTOTAL TEAMS: {total_teams}")
        lines.append("")
        
        # Venue Times
        lines.append("-" * 40)
        lines.append("AVAILABLE GAME TIMES BY VENUE")
        lines.append("-" * 40)
        venue_times = self.get_venue_times()
        for venue, categories in venue_times.items():
            lines.append(f"\n{venue}:")
            if categories.get('standard'):
                lines.append("  Standard times:")
                for day, times in categories['standard'].items():
                    lines.append(f"    {day}: {', '.join(times)}")
            if categories.get('phl'):
                lines.append("  PHL-specific times:")
                for day, times in categories['phl'].items():
                    lines.append(f"    {day}: {', '.join(times)}")
        lines.append("")
        
        # Field Unavailabilities
        lines.append("-" * 40)
        lines.append("FIELD UNAVAILABILITIES (HARD CONSTRAINTS)")
        lines.append("-" * 40)
        unavail_summary = self.get_field_unavailabilities_summary()
        for field, dates in unavail_summary.items():
            lines.append(f"\n{field}:")
            if dates:
                for d in dates:
                    lines.append(f"  - {d}")
            else:
                lines.append("  (none)")
        lines.append("")
        
        # Special Requests
        lines.append("-" * 40)
        lines.append("SPECIAL REQUESTS")
        lines.append("-" * 40)
        special = self.get_special_requests()
        
        # Club Days
        lines.append("\nCLUB DAYS:")
        if special['club_days']:
            for cd in special['club_days']:
                lines.append(f"  - {cd['club']}: {cd['date']} - {cd['description']}")
        else:
            lines.append("  (none configured)")
        
        # No-Play Dates (Soft)
        lines.append("\nNO-PLAY PREFERENCES (soft constraints):")
        soft_no_play = [np for np in special['no_play_dates'] if np['type'] == 'soft']
        if soft_no_play:
            for np in soft_no_play:
                grade_str = np['grade'] if isinstance(np['grade'], str) else '/'.join(np['grade'])
                lines.append(f"  - {np['club']} ({grade_str}): {', '.join(np['dates'])}")
                lines.append(f"    Reason: {np['reason']}")
        else:
            lines.append("  (none configured)")
        
        # Friday Night Allocations
        lines.append("\nFRIDAY NIGHT PHL ALLOCATIONS:")
        friday = special.get('friday_night_allocations', {})
        if friday:
            lines.append(f"  Total matches requested: {friday.get('total_matches', 0)}")
            lines.append(f"  Confirmed dates: {', '.join(friday.get('dates', []))}")
            lines.append("  Club allocations:")
            for club, count in friday.get('clubs', {}).items():
                lines.append(f"    - {club}: {count} match(es)")
        else:
            lines.append("  (none configured)")
        
        # Special Games
        lines.append("\nSPECIAL GAMES:")
        if special['special_games']:
            for sg in special['special_games']:
                teams_str = ' vs '.join(sg['teams']) if sg['teams'] else 'TBD'
                grades_str = '/'.join(sg['grades']) if sg['grades'] else 'TBD'
                lines.append(f"  - {sg['key']}: {teams_str} ({grades_str}) - {sg['date']}")
        else:
            lines.append("  (none configured)")
        
        # Team Conflicts
        lines.append("\nTEAM CONFLICTS (cannot play at same time):")
        if special['team_conflicts']:
            for tc in special['team_conflicts']:
                lines.append(f"  - {tc}")
        else:
            lines.append("  (none configured)")
        lines.append("")
        
        # Special Event Days
        lines.append("-" * 40)
        lines.append("SPECIAL EVENT CALENDAR")
        lines.append("-" * 40)
        events = self.get_special_events()
        if events:
            for event in events:
                lines.append(f"  {event['date']}: {event['name']} [{event['type']}]")
        else:
            lines.append("  (no special events)")
        lines.append("")
        
        # PHL Configuration Summary
        lines.append("-" * 40)
        lines.append("PHL CONFIGURATION")
        lines.append("-" * 40)
        phl_prefs = self.config.get('phl_preferences', {})
        lines.append(f"  PHL/2nd back-to-back: {phl_prefs.get('phl_2nd_back_to_back', False)}")
        lines.append(f"  Gosford 2nd grade bye: {phl_prefs.get('gosford_2nd_grade_bye', False)}")
        
        # SC Weekend PHL Slots
        phl_sc_slots = self.config.get('phl_sc_weekend_slots', {})
        if phl_sc_slots:
            lines.append("\n  STATE CHAMPIONSHIP WEEKEND PHL SLOTS:")
            lines.append("  (PHL games allowed during otherwise blocked SC weekends)")
            for sc_date, fields_slots in sorted(phl_sc_slots.items()):
                lines.append(f"    {sc_date.strftime('%A %d %b %Y')}:")
                for field_name, times in fields_slots.items():
                    times_str = ', '.join(t.strftime('%H:%M') for t in times)
                    lines.append(f"      {field_name}: {times_str}")
        lines.append("")
        
        # Home Field Mappings
        lines.append("-" * 40)
        lines.append("HOME FIELD MAPPINGS")
        lines.append("-" * 40)
        home_field_map = self.config.get('home_field_map', {})
        default_field = 'Newcastle International Hockey Centre'
        lines.append(f"  Default home field: {default_field}")
        if home_field_map:
            lines.append("  Custom mappings:")
            for club, field in home_field_map.items():
                lines.append(f"    - {club}: {field}")
        lines.append("")
        
        # Summary
        lines.append("=" * 80)
        lines.append("CONFIGURATION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total Teams: {total_teams}")
        lines.append(f"Total Grades: {len(self.grades)}")
        lines.append(f"Total Clubs: {len(self.clubs)}")
        lines.append(f"Total Venues: {len(self.fields)}")
        lines.append(f"Total Timeslots Generated: {len(self.timeslots)}")
        lines.append(f"Max Rounds: {self.max_rounds}")
        lines.append("")
        
        # Pre-start validation
        lines.append("-" * 40)
        lines.append("PRE-START VALIDATION")
        lines.append("-" * 40)
        errors = []
        warnings = []
        
        # Check rounds
        if not validation['valid']:
            errors.append(validation['warning'])
        
        # Check minimum teams per grade
        for grade in self.grades:
            if grade.num_teams < 2:
                errors.append(f"Grade {grade.name} has only {grade.num_teams} team(s) - need at least 2")
        
        # Check odd team grades (will have byes)
        for grade in self.grades:
            if grade.num_teams % 2 == 1:
                warnings.append(f"Grade {grade.name} has {grade.num_teams} teams (odd) - will have byes each round")
        
        # Check timeslots generated
        if len(self.timeslots) == 0:
            errors.append("No timeslots generated! Check date range and day_time_map configuration")
        
        if errors:
            lines.append("ERRORS (must fix before running solver):")
            for err in errors:
                lines.append(f"  ✗ {err}")
        else:
            lines.append("✓ No critical errors found")
        
        if warnings:
            lines.append("\nWARNINGS (may need attention):")
            for warn in warnings:
                lines.append(f"  ⚠ {warn}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save(self, filepath: str) -> None:
        """Save report to file."""
        report = self.generate_text_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Pre-season report saved to: {filepath}")


def run_preseason_check(year: int, output: Optional[str] = None) -> bool:
    """
    Run pre-season validation and generate report.
    
    Args:
        year: Season year
        output: Optional output file path
        
    Returns:
        True if validation passes, False if there are errors
    """
    # Import season config
    try:
        if year == 2025:
            from config.season_2025 import SEASON_CONFIG, get_season_data
        elif year == 2026:
            from config.season_2026 import SEASON_CONFIG, get_season_data
        else:
            print(f"Unknown season year: {year}")
            return False
    except ImportError as e:
        print(f"Error loading season config for {year}: {e}")
        return False
    
    # Build data
    data = get_season_data()
    
    # Generate report
    report = PreSeasonReport(data, SEASON_CONFIG)
    text = report.generate_text_report()
    
    # Print to console
    print(text)
    
    # Save if output specified
    if output:
        report.save(output)
    
    # Return validation status
    validation = report.validate_rounds()
    return validation['valid']
