# utils.py
"""
Utility functions for scheduling system.
"""
from typing import List, Dict, Any, Tuple, Set
from models import Team, Club, Grade, Game, WeeklyDraw, Roster, Timeslot, PlayingField
import re
import os
import sys
from datetime import datetime
import pandas as pd
from collections import defaultdict

def get_club(team_name: str, teams: List[Team]) -> str:
    for team in teams:
        if team_name == team.name:
            return team.club.name
    raise ValueError(f"Team {team_name} not found in teams when calling get_club")

def get_club_object(team_name: str, teams: List[Team]) -> Club:
    for team in teams:
        if team_name == team.name:
            return team.club
    raise ValueError(f"Team {team_name} not found in teams when calling get_club_object")

def get_teams_from_club(club_name: str, teams: List[Team]) -> List[str]:
    return [team.name for team in teams if team.club.name == club_name]

def get_club_from_clubname(club: str, clubs: List[Club]) -> Club:
    for c in clubs:
        if c.name == club:
            return c
    raise ValueError(f"Club {club} not found in CLUBS when calling get_club_from_clubname")

def get_duplicated_graded_teams(club: str, grade: str, teams: List[Team]) -> List[str]:
    dup_teams = []
    for team in teams:
        if team.club.name == club and team.grade == grade:
            dup_teams.append(team.name)
    return dup_teams

def split_number_suffix(text: str):
    match = re.match(r"(\d+)([a-zA-Z]+)", text)
    if match:
        return match.group(1), match.group(2)
    return text, ""

def add_ordinal_suffix(number: int) -> str:
    if 11 <= number % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"

def get_round_number_for_week(week: int, timeslots: List[Any]) -> int:
    for t in timeslots:
        if t.week == week:
            return t.round_no
    raise ValueError(f"Week {week} not found in timeslots when calling get_round_number_for_week")

def get_nearest_week_by_date(target_date_str: str, timeslots: List[Any], date_format: str = "%Y-%m-%d") -> int:
    target_date = datetime.strptime(target_date_str, date_format).date()
    def parse_date(t):
        return datetime.strptime(t.date, date_format).date()
    closest = min(timeslots, key=lambda t: abs((parse_date(t) - target_date).days))
    return closest.week

def get_field_by_name(name: str, fields: List[PlayingField]) -> PlayingField:
    for field in fields:
        if field.name == name:
            return field
    raise ValueError(f"Field {name} not found in field list.")

def get_grade_by_name(name: str, grades: List[Grade]) -> Grade:
    for grade in grades:
        if grade.name == name:
            return grade
    raise ValueError(f"Grade {name} not found in grade list.")

def convert_X_to_roster(X: Dict[Any, Any], data: Dict) -> Roster:
    """
    Convert the solution X (dict of scheduled games) into a Roster object.
    """
    weekly_games: Dict[int, List[Game]] = defaultdict(list)
    all_teams: Set[str] = {team.name for team in data['teams']}
    fields = data['fields']
    grades = data['grades']

    for key, var in X.items():
        # Only include scheduled games (var is True/1 or solved to 1)
        if hasattr(var, 'solution_value'):
            if var.solution_value() < 0.5:
                continue
        elif not var:
            continue
        # Key: (t1, t2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
        if len(key) < 11:
            continue
        t1, t2, grade_name, day, day_slot, time, week, date, round_no, field_name, field_location = key[:11]
        field = get_field_by_name(field_name, fields)
        grade = get_grade_by_name(grade_name, grades)
        timeslot = Timeslot(
            date=date,
            day=day,
            time=time,
            week=int(week),
            day_slot=int(day_slot),
            field=field,
            round_no=int(round_no)
        )
        game = Game(
            team1=t1,
            team2=t2,
            timeslot=timeslot,
            field=field,
            grade=grade
        )
        weekly_games[int(week)].append(game)

    weeks = []
    for week, games in sorted(weekly_games.items()):
        teams_played = set()
        for game in games:
            teams_played.update([game.team1, game.team2])
        bye_teams = list(all_teams - teams_played)
        round_no = games[0].timeslot.round_no if games else 1
        weeks.append(WeeklyDraw(
            week=week,
            round_no=round_no,
            games=games,
            bye_teams=bye_teams
        ))
    return Roster(weeks=weeks)

def export_roster_to_excel(roster: Roster, data: Dict, filename: str = "schedule.xlsx") -> None:
    """
    Export the Roster to an Excel file, one sheet per week, with game details.
    """
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        teams = data['teams']
        for weekly_draw in roster.weeks:
            week = weekly_draw.week
            # Extract games data
            game_data = [
                [game.timeslot.round_no, game.team1, game.team2, game.grade.name, game.field.name, game.field.location, game.timeslot.date, game.timeslot.day, game.timeslot.time, game.timeslot.day_slot]
                for game in weekly_draw.games
            ]
            df = pd.DataFrame(game_data, columns=[
                "ROUND", "TEAM 1", "TEAM 2", "GRADE", "FIELD", "LOCATION", "DATE", "DAY", "TIME", "DAY_SLOT"
            ])
            # Add byes
            if weekly_draw.bye_teams:
                for bye_team in weekly_draw.bye_teams:
                    df = pd.concat([
                        df,
                        pd.DataFrame([[None, bye_team, "BYE", None, None, None, None, None, None, None]], columns=df.columns)
                    ], ignore_index=True)
            df.to_excel(writer, sheet_name=f"Week {week}", index=False)
    print(f"Schedule successfully exported to {filename}")


# ============== Timeslot and Game Generation Functions ==============
# (Ported from legacy/main_notebook_translation.py)

def generate_timeslots(start_date, end_date, day_time_map, fields, field_unavailabilities):
    """
    Generate weekly timeslots between two dates, considering field unavailability.
    
    Args:
        start_date: datetime - Start date of the season
        end_date: datetime - End date of the season
        day_time_map: dict - Mapping of field location -> day -> list of times
        fields: list - List of PlayingField objects
        field_unavailabilities: dict - Mapping of field location -> unavailability info
        
    Returns:
        list of dicts with timeslot information
    """
    from datetime import timedelta
    
    timeslots = []
    current_date = start_date
    week_number = 1
    round_no = 0
    day_slot = 1
    c_day = None
    round_indic = True
    draw_start = False

    # Check if all fields in field_unavailabilities exist in fields
    field_name_check = list(field_unavailabilities.keys())
    known_fields = [field.location for field in fields]
    for field_name in field_name_check:
        if field_name not in known_fields:
            raise ValueError(f"Field {field_name} in field_unavailabilities does not exist in fields!") 
    # Check that fields are correct in day time map
    for field_name in day_time_map.keys():
        if field_name not in known_fields:
            raise ValueError(f"Field {field_name} in day_time_map does not exist in fields!")

    while current_date <= end_date:
        day_name = current_date.strftime('%A')
        if day_name in [key for field in day_time_map for key in day_time_map[field].keys()]:
            draw_start = True
            if c_day != day_name:
                day_slot = 1
                c_day = day_name

            # Check if the whole weekend is out for Broadmeadow to set the rounds
            if any(current_date.date() in [(w - timedelta(days=1)).date(), w.date(), (w + timedelta(days=1)).date()] 
                for w in field_unavailabilities.get('Newcastle International Hockey Centre', {}).get('weekends', [])):
                pass
            elif round_indic:
                round_no += 1
                round_indic = False

            for field in fields:
                field_name = field.location
                
                # Check if the whole weekend is unavailable 
                if any(current_date.date() in [(w - timedelta(days=1)).date(), w.date(), (w + timedelta(days=1)).date()] 
                    for w in field_unavailabilities.get(field_name, {}).get('weekends', [])):
                    continue

                day_slot = 1

                if field_name not in day_time_map or day_name not in day_time_map[field_name]:
                    continue

                for t in day_time_map[field_name][day_name]:
                    # Check if the whole day is unavailable 
                    if current_date.date() in [d.date() for d in field_unavailabilities.get(field_name, {}).get('whole_days', [])]:
                        continue

                    # Check if a partial day is unavailable 
                    if any(current_date.date() == pd.date() and t == pd.time() 
                        for pd in field_unavailabilities.get(field_name, {}).get('part_days', [])):
                        continue

                    timeslots.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'day': day_name,
                        'time': t.strftime('%H:%M'),
                        'week': week_number,
                        'day_slot': day_slot,
                        'field': field,
                        'round_no': round_no
                    })
                    day_slot += 1

        if current_date.strftime('%A') == 'Monday' and draw_start:
            week_number += 1
            day_slot = 1
            round_indic = True

        current_date += timedelta(days=1)

    return timeslots


def max_games_per_grade(
    grades: List,
    max_rounds: int,
    max_weekends_per_grade: Dict[str, int] = None,
    grade_rounds_override: Dict[str, int] = None,
    grade_scheduling_method: Dict[str, int] = None
) -> Dict[str, int]:
    """
    Given a list of Grade objects (each with num_teams), the default maximum
    number of rounds, and optional per-grade configurations, returns a dict
    mapping grade.name → max games per team.

    Two scheduling methods:
      Method 1 (default): Balanced round-robin. Games = largest multiple of
        (T-1) that fits, so each team plays every opponent the same number
        of times. E.g. 6 teams, 22 weekends → 20 games (4× each opponent).
      Method 2: Maximize games. Fits as many games as possible, allowing
        base/base+1 matchup frequency. E.g. 8 teams, 20 weekends → 20 games.

    In each round you can have floor(T/2) matches, so over R rounds:
      total_matches ≤ R * floor(T/2)
    and since each team plays g games, total_matches = g * T / 2 must be integer.

    Args:
        grades: List of Grade objects with num_teams attribute
        max_rounds: Default maximum number of rounds in the season
        max_weekends_per_grade: Optional dict of grade name → max weekends for that grade
                               (overrides max_rounds for specific grades)
        grade_rounds_override: Optional dict of grade name → exact rounds to play
                              (completely overrides the calculation)
        grade_scheduling_method: Optional dict of grade name → 1 or 2 (default: 1)

    Returns:
        Dict mapping grade name to max games per team
    """
    games_per_grade: Dict[str, int] = {}

    # Ensure defaults
    if max_weekends_per_grade is None:
        max_weekends_per_grade = {}
    if grade_rounds_override is None:
        grade_rounds_override = {}
    if grade_scheduling_method is None:
        grade_scheduling_method = {}

    for grade in grades:
        T = grade.num_teams
        if T < 2:
            games_per_grade[grade.name] = 0
            continue

        # Check for exact override first
        if grade.name in grade_rounds_override:
            games_per_grade[grade.name] = grade_rounds_override[grade.name]
            continue

        # Get max weekends for this grade (or use default)
        grade_max_rounds = max_weekends_per_grade.get(grade.name, max_rounds)

        # maximum matches across all rounds
        max_matches = grade_max_rounds * (T // 2)

        # g0 = floor( 2 * max_matches / T )
        g0 = (2 * max_matches) // T
        # can't exceed one game per round
        g0 = min(g0, grade_max_rounds)

        # ensure g0*T is even → if T odd, force g0 even
        if T % 2 == 1 and (g0 % 2) == 1:
            g0 -= 1

        # Apply scheduling method (default: method 1 = balanced round-robin)
        method = grade_scheduling_method.get(grade.name, 1)
        if method == 1:
            # Round down to nearest multiple of (T-1) so every team
            # plays every opponent the same number of times
            opponents = T - 1
            if g0 % opponents != 0:
                g0 = (g0 // opponents) * opponents

        games_per_grade[grade.name] = g0

    return games_per_grade


def circle_method_round_1_pairings(teams_by_grade: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate Round 1 pairings using the circle method for each grade.
    
    The circle method produces a canonical round-robin schedule. By fixing
    Round 1 pairings, we eliminate symmetry from equivalent schedules that
    differ only in which games appear in which round.
    
    For odd team counts, a "ghost" team is used at position 0. Pairings with
    the ghost represent byes and are excluded from the output.
    
    Args:
        teams_by_grade: Dict mapping grade name to list of team names
        
    Returns:
        Dict mapping grade name to list of (team1, team2) pairs for Round 1
        
    Example:
        >>> teams = {'PHL': ['A', 'B', 'C', 'D']}
        >>> circle_method_round_1_pairings(teams)
        {'PHL': [('A', 'B'), ('D', 'C')]}
    """
    result = {}
    
    for grade, team_names in teams_by_grade.items():
        n = len(team_names)
        if n < 2:
            result[grade] = []
            continue
        
        # For circle method, we need even number of positions
        # Position 0 is fixed, positions 1 to n-1 rotate
        # For odd teams, position 0 is a "ghost" (bye)
        if n % 2 == 1:
            # Add ghost at position 0, teams at positions 1 to n
            positions = [None] + list(team_names)  # None = ghost/bye
        else:
            # First team fixed at position 0, rest rotate
            positions = list(team_names)
        
        num_positions = len(positions)
        
        # Round 1 pairings: position i plays position (n-1-i) for i < n/2
        # In circle method, position 0 plays position n-1, 1 plays n-2, etc.
        pairings = []
        for i in range(num_positions // 2):
            j = num_positions - 1 - i
            team_a = positions[i]
            team_b = positions[j]
            
            # Skip if either is ghost (bye)
            if team_a is not None and team_b is not None:
                # Use consistent ordering (sorted) to match generate_games
                pair = tuple(sorted((team_a, team_b)))
                pairings.append(pair)
        
        result[grade] = pairings
    
    return result


def generate_games(teams: List[Team]) -> Dict[Tuple[str, str, str], Tuple]:
    """
    Generate all possible games between teams of the same grade.
    
    Args:
        teams: List of Team objects
        
    Returns:
        Dict mapping (team1_name, team2_name, grade) -> (team1, team2, grade)
    """
    games = {
        (min(t1.name, t2.name), max(t1.name, t2.name), t1.grade): (t1, t2, t1.grade)
        for i, t1 in enumerate(teams)
        for t2 in teams[i + 1:] if t1.grade == t2.grade
    }
    return games


# ============== Forced Games (Partial Key Matching) ==============
# Key indices for the 11-tuple: (team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
_KEY_INDEX = {
    'team1': 0, 'team2': 1, 'grade': 2, 'day': 3, 'day_slot': 4,
    'time': 5, 'week': 6, 'date': 7, 'round_no': 8, 'field_name': 9, 'field_location': 10,
}
_SCOPE_FIELDS = {'grade', 'day', 'day_slot', 'time', 'week', 'date', 'round_no', 'field_name', 'field_location'}


def _build_forced_game_rules(forced_games: list, teams: list) -> tuple:
    """
    Build lookup structure from FORCED_GAMES config for fast variable filtering.

    Groups entries by their scope (non-team fields). For each scope, collects
    all allowed team matchers. A variable matching the scope must match at least
    one team matcher to survive.

    Each entry supports a 'constraint' field controlling the equality type:
        'equal'    (default) — sum(vars) == 1
        'lesse'              — sum(vars) <= 1
        'greater'            — sum(vars) > 1  (i.e. >= 2)
        'greatere'           — sum(vars) >= 1
        'less'               — sum(vars) < 1  (i.e. == 0, rarely useful)

    Team names in config can be club names (e.g., 'Maitland') — they are auto-resolved
    to full team names (e.g., 'Maitland PHL') using the grade field and teams list.

    Returns:
        Tuple of (scope_groups dict, constraint_types dict).
        scope_groups: scope_key (frozenset) -> list of team matchers.
        constraint_types: scope_key (frozenset) -> constraint type string.
        Each team matcher is ('pair', team1, team2) or ('any', team_name).
    """
    if not forced_games:
        return {}, {}

    # Build lookup: (club_name, grade) -> [full team names]
    team_lookup = defaultdict(list)
    team_names_set = set()
    for t in teams:
        team_names_set.add(t.name)
        team_lookup[(t.club.name, t.grade)].append(t.name)
    
    def resolve_team_name(name, grade=None):
        """Resolve a club name or team name to full team name(s)."""
        # If already a full team name, use it
        if name in team_names_set:
            return [name]
        # If grade specified, try {club} {grade}
        if grade and not isinstance(grade, (list, tuple)):
            full = f"{name} {grade}"
            if full in team_names_set:
                return [full]
            # Look up by club + grade
            matches = team_lookup.get((name, grade), [])
            if matches:
                return matches
        # If grade is a list, resolve for each grade
        if grade and isinstance(grade, (list, tuple)):
            results = []
            for g in grade:
                results.extend(resolve_team_name(name, g))
            return results
        # No grade — find all teams from this club
        results = [t.name for t in teams if t.club.name == name]
        return results if results else [name]
    
    # scope_key -> list of team matchers
    scope_groups = defaultdict(list)
    # scope_key -> constraint type ('equal', 'lesse', 'greater', 'greatere', 'less')
    constraint_types = {}

    for entry in forced_games:
        grade = entry.get('grade')
        
        # Build scope from non-team fields
        scope = []
        for field in _SCOPE_FIELDS:
            if field in entry:
                val = entry[field]
                idx = _KEY_INDEX[field]
                if isinstance(val, list):
                    scope.append((idx, tuple(val)))
                else:
                    scope.append((idx, val))
        
        scope_key = frozenset(scope)

        # Store constraint type (default: 'equal' for sum == 1)
        constraint_types[scope_key] = entry.get('constraint', 'equal')

        # Build team matcher from 'teams' key or team1/team2
        # If no teams specified, use ('all',) matcher to match any team pair in scope
        raw_teams = entry.get('teams', [])
        if raw_teams:
            if len(raw_teams) == 2:
                resolved_t1 = resolve_team_name(raw_teams[0], grade)
                resolved_t2 = resolve_team_name(raw_teams[1], grade)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        scope_groups[scope_key].append(('pair', pair[0], pair[1]))
            elif len(raw_teams) == 1:
                resolved = resolve_team_name(raw_teams[0], grade)
                for rt in resolved:
                    scope_groups[scope_key].append(('any', rt))
        elif 'team1' in entry or 'team2' in entry:
            t1_raw = entry.get('team1')
            t2_raw = entry.get('team2')
            if t1_raw and t2_raw:
                resolved_t1 = resolve_team_name(t1_raw, grade)
                resolved_t2 = resolve_team_name(t2_raw, grade)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        scope_groups[scope_key].append(('pair', pair[0], pair[1]))
            elif t1_raw:
                for rt in resolve_team_name(t1_raw, grade):
                    scope_groups[scope_key].append(('any', rt))
            elif t2_raw:
                for rt in resolve_team_name(t2_raw, grade):
                    scope_groups[scope_key].append(('any', rt))
        else:
            # No teams specified — force any game matching the scope
            scope_groups[scope_key].append(('all',))
        
        desc = entry.get('description', f"scope={dict(scope)}")
        matchers = scope_groups[scope_key]
        matcher_details = ', '.join(
            "any vs any" if m[0] == 'all'
            else f"{m[1]} vs {m[2]}" if m[0] == 'pair'
            else f"{m[1]} vs any"
            for m in matchers
        )
        ctype = constraint_types[scope_key]
        ctype_desc = f" [{ctype}]" if ctype != 'equal' else ''
        print(f"  Forced game rule: {desc} -> {len(matchers)} team matcher(s): [{matcher_details}]{ctype_desc}")

    return dict(scope_groups), constraint_types


def _check_forced_game_status(key: tuple, forced_rules: dict):
    """
    Check a variable key against forced game rules.

    FORCED_GAMES works by finding all variables that match the partial key
    (scope + team matcher) and adding sum == 1 to ensure exactly one outcome
    occurs. Variables that DON'T match are left alone — they are NOT eliminated.

    Returns:
        ('force', scope_key) — variable matches scope AND teams → track it for sum == 1
        ('normal', None)     — variable doesn't match any forced rule → create normally

    Args:
        key: 11-tuple (team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
        forced_rules: Output of _build_forced_game_rules()
    """
    for scope_key, team_matchers in forced_rules.items():
        # Check if variable matches this scope
        in_scope = True
        for idx, val in scope_key:
            key_val = key[idx]
            if isinstance(val, tuple):
                # List match (any-of)
                if key_val not in val:
                    in_scope = False
                    break
            else:
                if key_val != val:
                    in_scope = False
                    break

        if not in_scope:
            continue

        # Variable is in scope — check if it matches ANY team matcher
        t1, t2 = key[0], key[1]
        sorted_pair = tuple(sorted([t1, t2]))
        for matcher in team_matchers:
            if matcher[0] == 'all':
                return ('force', scope_key)  # No team filter — any game in scope
            elif matcher[0] == 'pair':
                if sorted_pair[0] == matcher[1] and sorted_pair[1] == matcher[2]:
                    return ('force', scope_key)  # Matches — force this game
            elif matcher[0] == 'any':
                if t1 == matcher[1] or t2 == matcher[1]:
                    return ('force', scope_key)  # Matches — force this game

        # In scope but doesn't match teams — leave it alone, it's a different game

    return ('normal', None)  # Not a forced game — create normally


# ============== Blocked Games (No-Play Variable Removal) ==============
# Sister mechanism to FORCED_GAMES. Same config format but different action:
# FORCED_GAMES: variables matching scope AND matching teams → sum == 1 (ensure game happens)
# BLOCKED_GAMES: variables matching scope AND matching teams → eliminated (prevent game)

def _build_blocked_game_rules(blocked_games: list, teams: list) -> dict:
    """
    Build lookup structure from BLOCKED_GAMES config for fast variable filtering.
    
    Same format as FORCED_GAMES entries. For each scope, collects team matchers.
    A variable matching both the scope AND a team matcher is eliminated.
    
    Returns:
        Dict mapping scope_key (frozenset) -> list of team matchers.
        Each team matcher is ('pair', team1, team2) or ('any', team_name).
    """
    if not blocked_games:
        return {}
    
    # Build lookup: (club_name, grade) -> [full team names]
    team_lookup = defaultdict(list)
    team_names_set = set()
    for t in teams:
        team_names_set.add(t.name)
        team_lookup[(t.club.name, t.grade)].append(t.name)
    
    def resolve_team_name(name, grade=None):
        """Resolve a club name or team name to full team name(s)."""
        if name in team_names_set:
            return [name]
        if grade and not isinstance(grade, (list, tuple)):
            full = f"{name} {grade}"
            if full in team_names_set:
                return [full]
            matches = team_lookup.get((name, grade), [])
            if matches:
                return matches
        if grade and isinstance(grade, (list, tuple)):
            results = []
            for g in grade:
                results.extend(resolve_team_name(name, g))
            return results
        results = [t.name for t in teams if t.club.name == name]
        return results if results else [name]
    
    scope_groups = defaultdict(list)
    
    for entry in blocked_games:
        grade = entry.get('grade')
        grades = entry.get('grades', [])
        effective_grade = grades if grades else grade
        
        # Build scope from non-team fields
        scope = []
        for field in _SCOPE_FIELDS:
            if field in entry:
                val = entry[field]
                idx = _KEY_INDEX[field]
                if isinstance(val, list):
                    scope.append((idx, tuple(val)))
                else:
                    scope.append((idx, val))
        
        scope_key = frozenset(scope)
        
        # Build team matchers from 'teams' or 'club' key
        raw_teams = entry.get('teams', [])
        club = entry.get('club')
        
        if raw_teams:
            if len(raw_teams) == 2:
                resolved_t1 = resolve_team_name(raw_teams[0], effective_grade)
                resolved_t2 = resolve_team_name(raw_teams[1], effective_grade)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        scope_groups[scope_key].append(('pair', pair[0], pair[1]))
            elif len(raw_teams) == 1:
                resolved = resolve_team_name(raw_teams[0], effective_grade)
                for rt in resolved:
                    scope_groups[scope_key].append(('any', rt))
        elif club:
            resolved = resolve_team_name(club, effective_grade)
            for rt in resolved:
                scope_groups[scope_key].append(('any', rt))
        else:
            # No teams or club specified — block ALL variables matching this scope
            # Ensure the scope key exists (with empty matcher list = block all)
            if scope_key not in scope_groups:
                scope_groups[scope_key] = []

        desc = entry.get('description', entry.get('reason', f"scope={dict(scope)}"))
        matchers = scope_groups[scope_key]
        matcher_desc = f"{len(matchers)} team matcher(s)" if matchers else "ALL teams (no filter)"
        print(f"  Blocked game rule: {desc} -> {matcher_desc}")
    
    return dict(scope_groups)


def _is_blocked_by_no_play(key: tuple, blocked_rules: dict) -> bool:
    """
    Check if a variable key is blocked by no-play rules.
    
    A variable is blocked if it matches a rule's SCOPE AND matches
    ANY of the team matchers for that scope.
    (Inverse of FORCED_GAMES logic.)
    
    Args:
        key: 11-tuple variable key
        blocked_rules: Output of _build_blocked_game_rules()
    
    Returns:
        True if the variable should be eliminated.
    """
    for scope_key, team_matchers in blocked_rules.items():
        # Check if variable matches this scope
        in_scope = True
        for idx, val in scope_key:
            key_val = key[idx]
            if isinstance(val, tuple):
                if key_val not in val:
                    in_scope = False
                    break
            else:
                if key_val != val:
                    in_scope = False
                    break
        
        if not in_scope:
            continue

        # No team matchers = block ALL variables in scope
        if not team_matchers:
            return True

        # Variable is in scope — check if it matches ANY team matcher
        t1, t2 = key[0], key[1]
        sorted_pair = tuple(sorted([t1, t2]))
        for matcher in team_matchers:
            if matcher[0] == 'pair':
                if sorted_pair[0] == matcher[1] and sorted_pair[1] == matcher[2]:
                    return True  # Matches scope AND team — BLOCKED
            elif matcher[0] == 'any':
                if t1 == matcher[1] or t2 == matcher[1]:
                    return True  # Matches scope AND team — BLOCKED

    return False  # No match — not blocked


def _diagnose_missing_forced_game(entry: dict, forced_rules: dict, scope_key, data: dict) -> list:
    """
    Diagnose why a forced game rule matched zero decision variables.

    Checks each filter layer that could eliminate variables and returns
    a list of human-readable reason strings.
    """
    reasons = []
    timeslots = data['timeslots']
    teams = data['teams']
    team_names = {t.name for t in teams}
    team_to_club = {t.name: t.club.name for t in teams}

    # Extract scope fields for readable output
    idx_to_name = {v: k for k, v in _KEY_INDEX.items()}
    scope_dict = {idx_to_name.get(idx, str(idx)): val for idx, val in scope_key}

    grade = scope_dict.get('grade')
    date = scope_dict.get('date')
    day = scope_dict.get('day')
    field_location = scope_dict.get('field_location')

    # 1. Check if specified date has any timeslots at all
    if date:
        date_timeslots = [t for t in timeslots if t.date == date]
        if not date_timeslots:
            all_dates = sorted(set(t.date for t in timeslots if t.date))
            reasons.append(f"Date '{date}' has no timeslots in the season. "
                          f"Season dates range from {all_dates[0]} to {all_dates[-1]}.")
            return reasons

        # Check if the day matches any timeslots on that date
        if day:
            day_match = [t for t in date_timeslots if t.day == day]
            if not day_match:
                actual_days = set(t.day for t in date_timeslots)
                reasons.append(f"Date '{date}' has timeslots on {actual_days}, not '{day}'.")
                return reasons

    # 2. Check if specified venue has timeslots on that date/day
    if field_location:
        venue_slots = [t for t in timeslots if t.field.location == field_location]
        if not venue_slots:
            reasons.append(f"Venue '{field_location}' has no timeslots in the season.")
            return reasons
        if date:
            venue_date = [t for t in venue_slots if t.date == date]
            if not venue_date:
                reasons.append(f"Venue '{field_location}' has no timeslots on date '{date}'. "
                              f"Check FIELD_UNAVAILABILITIES.")

    # 3. Check team resolution
    raw_teams = entry.get('teams', [])
    if raw_teams:
        team_lookup = defaultdict(list)
        for t in teams:
            team_lookup[(t.club.name, t.grade)].append(t.name)

        for team_name in raw_teams:
            full_name = f"{team_name} {grade}" if grade else team_name
            if full_name not in team_names and team_name not in team_names:
                club_teams = [t.name for t in teams if t.club.name == team_name]
                if not club_teams:
                    reasons.append(f"Team/club '{team_name}' not found in any grade.")
                elif grade:
                    grade_teams = team_lookup.get((team_name, grade), [])
                    if not grade_teams:
                        reasons.append(f"Club '{team_name}' has no team in grade '{grade}'. "
                                      f"Their grades: {sorted(set(t.grade for t in teams if t.club.name == team_name))}")

    # 4. Check PHL game times filter
    if grade == 'PHL':
        phl_game_times = data.get('phl_game_times', {})
        if phl_game_times and field_location:
            venue_times = phl_game_times.get(field_location, {})
            if not venue_times:
                reasons.append(f"PHL_GAME_TIMES has no entries for venue '{field_location}'.")
            elif day:
                # Check nested format: venue -> field -> day -> times
                has_day = False
                for field_or_day, val in venue_times.items():
                    if isinstance(val, dict):
                        # Nested: field -> day -> times
                        if day in val:
                            has_day = True
                    elif field_or_day == day:
                        # Simple: day -> times
                        has_day = True
                if not has_day:
                    reasons.append(f"PHL_GAME_TIMES has no '{day}' slots at '{field_location}'.")

    # 5. Check 2nd grade time filter
    if grade == '2nd':
        second_grade_times = data.get('second_grade_times', {})
        if second_grade_times and field_location:
            venue_times = second_grade_times.get(field_location, {})
            if not venue_times:
                reasons.append(f"SECOND_GRADE_TIMES has no entries for venue '{field_location}'.")

    # 6. Check lower grades excluded from PHL-only venues/days
    if grade and grade not in ('PHL', '2nd'):
        if field_location == 'Central Coast Hockey Park':
            reasons.append(f"Grade '{grade}' cannot play at Gosford (PHL-only venue).")
        if day == 'Friday':
            reasons.append(f"Grade '{grade}' cannot play on Fridays (PHL-only day).")

    # 7. Check home venue filter
    home_field_map = data.get('home_field_map', {})
    if field_location and field_location in home_field_map.values():
        home_club = None
        for club_name, venue in home_field_map.items():
            if venue == field_location:
                home_club = club_name
                break
        if home_club and raw_teams:
            team_clubs = set()
            for t_name in raw_teams:
                if t_name in team_to_club.values():
                    team_clubs.add(t_name)
                else:
                    club = team_to_club.get(f"{t_name} {grade}", team_to_club.get(t_name))
                    if club:
                        team_clubs.add(club)
            if home_club not in team_clubs:
                reasons.append(f"Venue '{field_location}' requires home club '{home_club}', "
                              f"but forced teams are from clubs: {team_clubs}.")

    # 8. Check BLOCKED_GAMES conflict
    blocked_games = data.get('blocked_games', [])
    for blocked in blocked_games:
        # Simple check: does a blocked game overlap scope?
        overlap = True
        for field in _SCOPE_FIELDS:
            if field in entry and field in blocked:
                if entry[field] != blocked[field]:
                    overlap = False
                    break
        if overlap and 'date' in entry and 'date' in blocked:
            blocked_teams = blocked.get('teams', [])
            forced_teams = entry.get('teams', [])
            if blocked.get('club'):
                # Club-level block — check if any forced team is from that club
                for ft in forced_teams:
                    club = team_to_club.get(f"{ft} {grade}", '')
                    if club == blocked['club'] or ft == blocked['club']:
                        reasons.append(f"BLOCKED_GAMES blocks club '{blocked['club']}' on "
                                      f"date '{blocked['date']}': {blocked.get('description', '')}")
            elif blocked_teams:
                common = set(forced_teams) & set(blocked_teams)
                if common:
                    reasons.append(f"BLOCKED_GAMES blocks team(s) {common} on "
                                  f"date '{blocked['date']}': {blocked.get('description', '')}")

    if not reasons:
        reasons.append("No specific cause identified. Check that the combination of scope fields "
                      "(grade, date, day, venue, field) produces valid timeslots after all filters.")

    return reasons


def generate_X(model, data: dict) -> Tuple[Dict, Dict, Dict]:
    """
    Generate decision variables for all possible games and timeslots.
    
    IMPORTANT: PHL, 2nd grade, and lower grades are filtered to valid timeslots.
    This dramatically reduces the variable count.
    
    PHL restrictions enforced here (NOT as constraints):
    - PHL cannot play on South Field (SF) at NIHC - only EF and WF
    - PHL can only play at times defined in phl_game_times
    - Gosford has limited slots (1 per week effectively)
    
    2nd grade restrictions enforced here (NOT as constraints):
    - 2nd grade cannot play on South Field (SF) at NIHC - only EF and WF
    - 2nd grade CANNOT play at Gosford (PHL-only venue)
    - 2nd grade plays at PHL times PLUS one slot before/after (where available)
    
    Lower grades (3rd-6th) restrictions enforced here (NOT as constraints):
    - Cannot play at Gosford (Central Coast Hockey Park) - PHL-only venue
    - Cannot play on Fridays - PHL-only timeslot
    
    NOTE: Cannot create NEW timeslots - only existing DAY_TIME_MAP slots are valid.
    
    Args:
        model: CP-SAT model
        data: Data dictionary containing teams, timeslots, etc.
        
    Returns:
        Tuple of (X, Y, conflicts)
        - X: Dict of decision variables for real timeslots
        - Y: Dict of decision variables for dummy timeslots (currently empty)
        - conflicts: Dict of team conflicts (currently empty, populated elsewhere)
    """
    teams = data['teams']
    timeslots = data['timeslots']
    
    # Get PHL and 2nd grade game times for filtering
    # Supports TWO structures:
    #   - 2025 (simple): { venue: { day: [times] } }  -> any field at venue is valid
    #   - 2026 (nested): { venue: { field: { day: [times] } } }  -> specific fields only
    phl_game_times = data.get('phl_game_times', {})
    second_grade_times = data.get('second_grade_times', {})
    
    # Detect structure format by checking first venue's first value
    # If it's a dict with day keys (Monday, Tuesday, etc.), it's the simple format
    # If it's a dict with field names as keys, it's the nested format
    is_simple_format = False
    for venue, venue_data in phl_game_times.items():
        if isinstance(venue_data, dict):
            first_key = next(iter(venue_data.keys()), None)
            # Day names in simple format, field names in nested format
            if first_key in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
                is_simple_format = True
        break
    
    # Build a set of valid slots for PHL
    # For simple format: (venue, day, time) - field is not restricted
    # For nested format: (venue, field, day, time) - specific field required
    phl_valid_slots = set()
    phl_valid_venue_day_time = set()  # For simple format fallback
    
    if is_simple_format:
        # 2025 format: { venue: { day: [times] } }
        for venue, days in phl_game_times.items():
            for day, times in days.items():
                for t in times:
                    time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)
                    phl_valid_venue_day_time.add((venue, day, time_str))
    else:
        # 2026 format: { venue: { field: { day: [times] } } }
        for venue, fields in phl_game_times.items():
            for field, days in fields.items():
                for day, times in days.items():
                    for t in times:
                        time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)
                        phl_valid_slots.add((venue, field, day, time_str))
    
    # Build a set of valid slots for 2nd grade (2026+ nested format only)
    # If no second_grade_times defined, 2nd grade uses all slots (backward compat)
    second_valid_slots = set()
    if second_grade_times:
        for venue, fields in second_grade_times.items():
            for field, days in fields.items():
                for day, times in days.items():
                    for t in times:
                        time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)
                        second_valid_slots.add((venue, field, day, time_str))
    
    # Generate all games if not already in data
    if 'games' not in data or not data['games']:
        games = generate_games(teams)
        data['games'] = games
    else:
        games = data['games']
        if isinstance(games, list):
            # Convert list to dict format if needed
            games = {g: g for g in games}
    
    X = {}
    Y = {}
    conflicts = {}
    
    # Handle both dict and list formats for games
    game_items = games.items() if isinstance(games, dict) else [(g, g) for g in games]
    
    phl_vars_created = 0
    phl_vars_skipped = 0
    second_vars_created = 0
    second_vars_skipped = 0
    other_vars_created = 0
    other_vars_skipped = 0  # Track skipped vars for lower grades
    forced_vars_skipped = 0  # Track vars eliminated by FORCED_GAMES
    forced_vars_forced = 0   # Track vars forced by FORCED_GAMES
    blocked_vars_skipped = 0  # Track vars eliminated by BLOCKED_GAMES
    
    # Build forced games lookup from config
    forced_game_rules, forced_constraint_types = _build_forced_game_rules(data.get('forced_games', []), teams)
    
    # Build blocked games (no-play) lookup from config
    blocked_game_rules = _build_blocked_game_rules(data.get('blocked_games', []), teams)
    
    # Collect forced variables per scope for adding sum==1 constraints
    forced_scope_vars = defaultdict(list)  # scope_key -> [(key, var), ...]
    
    # Build set of PHL-only venues (Gosford) and days (Friday)
    # These should be excluded from non-PHL grades
    phl_only_venues = {'Central Coast Hockey Park'}  # Gosford is PHL-only
    phl_only_days = {'Friday'}  # Friday nights are PHL-only

    # Build venue-to-home-club lookup for home-field restriction
    # Games at away venues (Maitland Park, Central Coast) must involve the home club.
    # Broadmeadow (NIHC) is the default/neutral venue — no restriction.
    home_field_map = data.get('home_field_map', {})
    venue_to_home_club = {}  # venue -> club name
    for club_name, venue in home_field_map.items():
        venue_to_home_club[venue] = club_name
    # Build team-to-club lookup for fast checking
    team_to_club = {t.name: t.club.name for t in teams}
    home_venue_skipped = 0
    
    for game_key, game_val in game_items:
        if isinstance(game_key, tuple) and len(game_key) >= 3:
            t1_name, t2_name, grade_name = game_key[0], game_key[1], game_key[2]
        else:
            continue
        
        is_phl = (grade_name == 'PHL')
        is_second = (grade_name == '2nd')
            
        # Create dummy variables (short 4-tuple keys: t1, t2, grade, index)
        num_dummy = data.get('num_dummy_timeslots', 0)
        for i in range(num_dummy):
            dummy_key = (t1_name, t2_name, grade_name, i)
            Y[dummy_key] = model.NewBoolVar(f"y_{t1_name}_{t2_name}_{grade_name}_dummy_{i}")

        for t in timeslots:
            if not t.day:
                continue  # skip dummy timeslots
            
            # For PHL games, only create variables for valid timeslots
            if is_phl:
                # Check using appropriate format
                if is_simple_format:
                    # 2025 format: check (venue, day, time) only - any field OK
                    slot_key = (t.field.location, t.day, t.time)
                    if slot_key not in phl_valid_venue_day_time:
                        phl_vars_skipped += 1
                        continue
                else:
                    # 2026 format: check (venue, field, day, time) - specific field required
                    slot_key = (t.field.location, t.field.name, t.day, t.time)
                    if slot_key not in phl_valid_slots:
                        phl_vars_skipped += 1
                        continue
                phl_vars_created += 1
            # For 2nd grade games, filter if second_grade_times is defined
            elif is_second and second_valid_slots:
                slot_key = (t.field.location, t.field.name, t.day, t.time)
                if slot_key not in second_valid_slots:
                    second_vars_skipped += 1
                    continue
                second_vars_created += 1
            else:
                # Lower grades (3rd-6th): Exclude PHL-only venues and days
                # - Gosford (Central Coast) is PHL-only
                # - Friday nights are PHL-only
                if t.field.location in phl_only_venues:
                    other_vars_skipped += 1
                    continue
                if t.day in phl_only_days:
                    other_vars_skipped += 1
                    continue
                other_vars_created += 1

            # Home-field restriction: games at away venues must involve the home club
            # e.g. games at Maitland Park must have a Maitland team,
            #      games at Central Coast must have a Gosford team
            home_club = venue_to_home_club.get(t.field.location)
            if home_club is not None:
                t1_club = team_to_club.get(t1_name, '')
                t2_club = team_to_club.get(t2_name, '')
                if t1_club != home_club and t2_club != home_club:
                    home_venue_skipped += 1
                    continue

            key = (t1_name, t2_name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
            
            # Check forced game rules - track matching vars for sum == 1 constraint.
            # Non-matching vars are left alone (not eliminated).
            if forced_game_rules:
                status, scope_key = _check_forced_game_status(key, forced_game_rules)
                if status == 'force':
                    var = model.NewBoolVar(f'X_{t1_name}_{t2_name}_{t.day}_{t.time}_{t.week}_{t.field.name}')
                    X[key] = var
                    forced_scope_vars[scope_key].append(var)
                    forced_vars_forced += 1
                    # Skip blocked games check — forced game takes priority
                    continue
            
            # Check blocked game rules - eliminate vars for teams that cannot play on these dates
            if blocked_game_rules and _is_blocked_by_no_play(key, blocked_game_rules):
                blocked_vars_skipped += 1
                continue
            
            X[key] = model.NewBoolVar(f'X_{t1_name}_{t2_name}_{t.day}_{t.time}_{t.week}_{t.field.name}')
    
    # Pre-check: verify all forced game rules matched at least one variable.
    # If any forced game has zero matching vars, diagnose why and exit.
    idx_to_name = {v: k for k, v in _KEY_INDEX.items()}
    missing_forced = []
    for scope_key in forced_game_rules:
        if scope_key not in forced_scope_vars or len(forced_scope_vars[scope_key]) == 0:
            missing_forced.append(scope_key)

    if missing_forced:
        # Build scope_key -> original entry mapping for diagnostics
        forced_games = data.get('forced_games', [])
        scope_to_entry = {}
        for entry in forced_games:
            scope = []
            for field in _SCOPE_FIELDS:
                if field in entry:
                    val = entry[field]
                    idx = _KEY_INDEX[field]
                    if isinstance(val, list):
                        scope.append((idx, tuple(val)))
                    else:
                        scope.append((idx, val))
            sk = frozenset(scope)
            scope_to_entry[sk] = entry

        print(f"\n{'='*70}")
        print(f"FATAL: {len(missing_forced)} forced game(s) have NO playable variables!")
        print(f"The solver cannot satisfy these forced games. Fix config before running.")
        print(f"{'='*70}")
        for sk in missing_forced:
            scope_desc = ', '.join(f"{idx_to_name.get(idx, idx)}={val}" for idx, val in sorted(sk))
            entry = scope_to_entry.get(sk, {})
            desc = entry.get('description', scope_desc)
            print(f"\n  FORCED GAME: {desc}")
            print(f"    Scope: {scope_desc}")
            teams_info = entry.get('teams', [])
            if teams_info:
                print(f"    Teams: {teams_info}")
            reasons = _diagnose_missing_forced_game(entry, forced_game_rules, sk, data)
            print(f"    Diagnosis:")
            for reason in reasons:
                print(f"      - {reason}")
        print(f"\n{'='*70}")
        print("Fix the FORCED_GAMES config, BLOCKED_GAMES, PHL_GAME_TIMES, "
              "FIELD_UNAVAILABILITIES, or season dates to resolve.")
        print(f"{'='*70}\n")
        sys.exit(1)

    for scope_key, vars_list in forced_scope_vars.items():
        ctype = forced_constraint_types.get(scope_key, 'equal')
        if ctype == 'equal':
            model.Add(sum(vars_list) == 1)
        elif ctype == 'lesse':
            model.Add(sum(vars_list) <= 1)
        elif ctype == 'greatere':
            model.Add(sum(vars_list) >= 1)
        elif ctype == 'greater':
            model.Add(sum(vars_list) > 1)
        elif ctype == 'less':
            model.Add(sum(vars_list) < 1)
        else:
            print(f"  WARNING: Unknown constraint type '{ctype}' for forced game, defaulting to equal")
            model.Add(sum(vars_list) == 1)
    
    print(f"Created {len(X)} decision variables")
    print(f"  - PHL: {phl_vars_created} created, {phl_vars_skipped} skipped (invalid venue/field/day/time)")
    if second_valid_slots:
        print(f"  - 2nd: {second_vars_created} created, {second_vars_skipped} skipped (invalid venue/field/day/time)")
    print(f"  - Other grades: {other_vars_created} created, {other_vars_skipped} skipped (PHL-only venues/days)")
    if home_venue_skipped:
        print(f"  - Home venue filter: {home_venue_skipped} vars eliminated (game at away venue without home club)")
    if forced_vars_forced:
        print(f"  - Forced games: {forced_vars_forced} vars forced across {len(forced_scope_vars)} forced game scopes")
        for scope_key, vars_list in forced_scope_vars.items():
            scope_desc = ', '.join(f"{idx_to_name.get(idx, idx)}={val}" for idx, val in sorted(scope_key))
            ctype = forced_constraint_types.get(scope_key, 'equal')
            op = {'equal': '==', 'lesse': '<=', 'greatere': '>=', 'greater': '>', 'less': '<'}.get(ctype, '==')
            print(f"    scope({scope_desc}): {len(vars_list)} decision vars -> sum {op} 1")
    if blocked_vars_skipped:
        print(f"  - Blocked games: {blocked_vars_skipped} vars eliminated by {len(data.get('blocked_games', []))} no-play rules")
    return X, Y, conflicts


# ============== Season Data Builder ==============

def build_season_data(config: dict) -> dict:
    """
    Build complete data dictionary from a season configuration.
    
    This is the main entry point for loading season data. It takes a SEASON_CONFIG
    dict from a season file (e.g., config/season_2025.py) and builds the complete
    data dictionary needed by the solver.
    
    Args:
        config: SEASON_CONFIG dict from a season file containing:
            - year: int
            - start_date: datetime
            - end_date: datetime  
            - max_rounds: int
            - teams_data_path: str
            - fields: list of field dicts
            - day_time_map: dict
            - phl_game_times: dict
            - field_unavailabilities: dict
            - club_days: dict
            - preference_no_play: dict
            - phl_preferences: dict
            - home_field_map: dict
            - grade_order: list
            
    Returns:
        Complete data dict ready for solver with:
            - teams: List[Team]
            - grades: List[Grade]
            - fields: List[PlayingField]
            - clubs: List[Club]
            - timeslots: List[Timeslot]
            - num_rounds: dict
            - And all other config values
    """
    # Extract config values
    year = config['year']
    teams_data_path = config['teams_data_path']
    start_date = config['start_date']
    end_date = config['end_date']
    max_rounds = config['max_rounds']
    home_field_map = config.get('home_field_map', {})
    grade_order = config.get('grade_order', ['PHL', '2nd', '3rd', '4th', '5th', '6th'])
    
    # Build PlayingField objects from config dicts
    FIELDS = [
        PlayingField(location=f['location'], name=f['name'])
        for f in config['fields']
    ]
    
    # Load teams and clubs from CSV files
    CLUBS = []
    TEAMS = []
    
    if not os.path.exists(teams_data_path):
        raise FileNotFoundError(f"Teams data path not found: {teams_data_path}")
    
    for file in os.listdir(teams_data_path):
        if not file.endswith('.csv'):
            continue
        
        df = pd.read_csv(os.path.join(teams_data_path, file))
        club_name = df['Club'].iloc[0].strip()
        
        # Determine home field from config or default
        home_field = home_field_map.get(club_name, 'Newcastle International Hockey Centre')
        
        club = Club(name=club_name, home_field=home_field)
        CLUBS.append(club)
        
        teams = [
            Team(
                name=f"{row['Team Name'].strip()} {row['Grade'].strip()}", 
                club=club, 
                grade=row['Grade'].strip()
            ) 
            for _, row in df.iterrows()
        ]
        TEAMS.extend(teams)
    
    # Create grades (sorted by grade_order)
    teams_by_grade = defaultdict(list)
    for team in TEAMS:
        teams_by_grade[team.grade].append(team.name)
    
    # Sort grades according to grade_order
    def grade_sort_key(item):
        grade_name = item[0]
        try:
            return grade_order.index(grade_name)
        except ValueError:
            return len(grade_order)  # Unknown grades go last
    
    GRADES = [
        Grade(name=grade, teams=teams) 
        for grade, teams in sorted(teams_by_grade.items(), key=grade_sort_key)
    ]
    
    # Update club team counts
    teams_by_club = defaultdict(list)
    for team in TEAMS:
        teams_by_club[team.club.name].append(team.name)
    
    for club in CLUBS:
        club.num_teams = len(teams_by_club.get(club.name, []))
    
    for grade in GRADES:
        grade.num_teams = len(grade.teams)
    
    # Get time configurations from config
    day_time_map = config['day_time_map']
    phl_game_times = config['phl_game_times']
    field_unavailabilities = config['field_unavailabilities']
    
    # Detect if phl_game_times uses nested format (2026+) or simple format (2025)
    # Nested: {venue: {field: {day: [times]}}}
    # Simple: {venue: {day: [times]}}
    def is_nested_format(pgt):
        for venue, venue_data in pgt.items():
            if isinstance(venue_data, dict):
                first_key = next(iter(venue_data.keys()), None)
                # Day names in simple format, field names in nested format
                if first_key not in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
                    return True
        return False
    
    # Flatten nested phl_game_times to simple format for timeslot merge
    # We merge to {venue: {day: [times]}} for generate_timeslots
    def flatten_phl_times(pgt):
        if not is_nested_format(pgt):
            return pgt
        flat = defaultdict(lambda: defaultdict(list))
        for venue, fields in pgt.items():
            for field, days in fields.items():
                for day, times in days.items():
                    flat[venue][day].extend(times)
        # Dedupe and sort
        for venue in flat:
            for day in flat[venue]:
                flat[venue][day] = sorted(list(set(flat[venue][day])))
        return dict(flat)
    
    phl_times_flat = flatten_phl_times(phl_game_times)
    
    # Merge day_time_map and phl_game_times for timeslot generation
    merged_dict = defaultdict(lambda: defaultdict(list))
    for d in (phl_times_flat, day_time_map):
        for field, days in d.items():
            for key, times in days.items():
                merged_dict[field][key].extend(times)
    
    for field in merged_dict:
        for key in merged_dict[field]:
            merged_dict[field][key] = list(dict.fromkeys(merged_dict[field][key]))
            merged_dict[field][key].sort()
    
    # Generate timeslots
    timeslots = generate_timeslots(start_date, end_date, merged_dict, FIELDS, field_unavailabilities)
    TIMESLOTS = [
        Timeslot(
            date=t['date'], day=t['day'], time=t['time'], week=t['week'],
            day_slot=t['day_slot'], field=t['field'], round_no=t['round_no']
        )
        for t in timeslots
    ]
    
    # Get grade-specific round configuration
    max_weekends_per_grade = config.get('max_weekends_per_grade', {})
    grade_rounds_override = config.get('grade_rounds_override', {})
    grade_scheduling_method = config.get('grade_scheduling_method', {})

    # Calculate rounds per grade (with overrides)
    num_rounds = max_games_per_grade(
        GRADES,
        max_rounds,
        max_weekends_per_grade=max_weekends_per_grade,
        grade_rounds_override=grade_rounds_override,
        grade_scheduling_method=grade_scheduling_method
    )
    num_rounds['max'] = max_rounds
    # Also store per-grade max weekends for reference
    num_rounds['max_weekends_per_grade'] = max_weekends_per_grade
    num_rounds['grade_rounds_override'] = grade_rounds_override
    num_rounds['grade_scheduling_method'] = grade_scheduling_method
    
    for grade, rounds in num_rounds.items():
        grade_obj = next((g for g in GRADES if g.name == grade), None)
        if grade_obj:
            grade_obj.set_games(rounds)
    
    # Max day slots per field
    max_day_slot_per_field = {
        field.location: max((t.day_slot for t in TIMESLOTS if t.field.location == field.location), default=0)
        for field in FIELDS
    }
    
    # Get preferences from config
    club_days = config.get('club_days', {})
    preference_no_play = config.get('preference_no_play', {})
    phl_preferences = config.get('phl_preferences', {'preferred_dates': []})
    num_dummy_timeslots = config.get('num_dummy_timeslots', 3)
    
    return {
        'year': year,
        'teams': TEAMS,
        'grades': GRADES,
        'fields': FIELDS,
        'clubs': CLUBS,
        'timeslots': TIMESLOTS,
        'num_rounds': num_rounds,
        'current_week': 0,
        'locked_weeks': set(),
        'penalties': {},
        'day_time_map': day_time_map,
        'phl_game_times': phl_game_times,
        'second_grade_times': config.get('second_grade_times', {}),
        'phl_preferences': phl_preferences,
        'max_day_slot_per_field': max_day_slot_per_field,
        'field_unavailabilities': field_unavailabilities,
        'club_days': club_days,
        'preference_no_play': preference_no_play,
        'num_dummy_timeslots': num_dummy_timeslots,
        'grade_order': grade_order,
        # Include any extra config values that constraints might need
        'home_field_map': home_field_map,
        'special_games': config.get('special_games', {}),
        'forced_games': config.get('forced_games', []),
        'blocked_games': config.get('blocked_games', []),
        'penalty_weights': config.get('penalty_weights', {}),
        'constraint_defaults': config.get('constraint_defaults', {}),
    }

