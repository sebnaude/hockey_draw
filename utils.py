# utils.py
"""
Utility functions for scheduling system.
"""
from typing import List, Dict, Any, Tuple, Set
from models import Team, Club, Grade, Game, WeeklyDraw, Roster, Timeslot, PlayingField
import re
import os
import sys
from datetime import datetime, date, timedelta
import pandas as pd
from collections import defaultdict
import math

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


def normalize_preference_no_play(noplay: dict, teams: list, clubs: list) -> list:
    """Normalize PREFERENCE_NO_PLAY to a list of (key, club_name, team_names, restriction) tuples.

    Supports:
      - 2025 legacy format: ``{'ClubName': [{'date': '...', ...}, ...]}``
      - 2026 structured format: ``{'EntryKey': {'club': 'ClubName',
        'dates': [datetime, ...], 'grade': '...' | 'grades': [...]}}``
      - spec-012 time-only / day-only / field-only filters:
        ``{'EntryKey': {'club': 'ClubName', 'time': '08:30'}}``. Any combination
        of ``time``, ``day``, ``day_slot``, ``week``, ``field_name``,
        ``field_location`` may be added alongside (or instead of) ``date`` /
        ``dates``. When NO date is supplied, a single normalized restriction is
        emitted carrying the non-date filters — the consumer applies it across
        every week.

    Used by `constraints/soft.py`, `constraints/unified.py`, and
    `constraints.archived.ai`'s `PreferredTimesConstraintAI`. Lives here in
    `utils.py` so all three can pull the helper without crossing the
    `constraints/archived/` lockdown line.
    """
    from datetime import datetime as _dt

    # Keys that travel into the restriction dict alongside (or instead of)
    # 'date'. The consumer (unified.py::_preferred_times, soft.py::
    # PreferredTimesConstraintSoft) matches them against X-key columns via
    # `dict(zip(allowed_keys, game_key))`.
    _FILTER_KEYS = ('time', 'day', 'day_slot', 'week',
                    'field_name', 'field_location')

    normalized = []
    club_names_lower = [c.name.lower() for c in clubs]

    for key, value in noplay.items():
        if isinstance(value, dict) and 'club' in value:
            club_name = value['club']
            if club_name.lower() not in club_names_lower:
                raise ValueError(f"Invalid club '{club_name}' in PREFERENCE_NO_PLAY entry '{key}'")
            dates = value.get('dates', [])
            if 'date' in value:
                dates = [value['date']]
            club_teams = get_teams_from_club(club_name, teams)
            if 'grade' in value:
                grade = value['grade']
                club_teams = [t for t in club_teams if grade.lower() in t.lower()]
            elif 'grades' in value:
                grades = [g.lower() for g in value['grades']]
                club_teams = [t for t in club_teams if any(g in t.lower() for g in grades)]

            # spec-012: collect non-date filters that should travel with each
            # emitted restriction.
            extra_filters = {k: value[k] for k in _FILTER_KEYS if k in value}

            if dates:
                for date in dates:
                    if isinstance(date, _dt):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)
                    restriction = {'date': date_str, **extra_filters}
                    normalized.append((key, club_name, club_teams, restriction))
            elif extra_filters:
                # spec-012: no dates but at least one time/day/field filter.
                # Emit a single restriction without a 'date' key — the
                # consumer applies it across every week.
                normalized.append((key, club_name, club_teams, dict(extra_filters)))
            # else: entry has neither dates nor filters → silently no-op.
        elif isinstance(value, list):
            club_name = key
            if club_name.lower() not in club_names_lower:
                raise ValueError(f"Invalid club name '{club_name}' in PREFERENCE_NO_PLAY")
            club_teams = get_teams_from_club(club_name, teams)
            for restriction in value:
                normalized.append((key, club_name, club_teams, restriction))
        else:
            raise ValueError(
                f"Invalid PREFERENCE_NO_PLAY format for key '{key}': expected dict with 'club' key or list"
            )

    return normalized

def get_club_from_clubname(club: str, clubs: List[Club]) -> Club:
    for c in clubs:
        if c.name == club:
            return c
    raise ValueError(f"Club {club} not found in CLUBS when calling get_club_from_clubname")

def normalize_club_day(value):
    """Normalize CLUB_DAYS entry to (date, opponent) tuple.

    Supports these formats:
        datetime(2026, 6, 22)                                   -> (datetime, None)
        '2026-06-22'                                            -> (str, None)
        {'date': datetime(2026, 7, 13), 'opponent': 'Souths'}  -> (datetime, 'Souths')
    """
    if isinstance(value, dict):
        return value['date'], value.get('opponent')
    return value, None

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
_VALID_CONSTRAINT_TYPES = {'equal', 'lesse', 'less', 'greater', 'greatere'}


def _build_team_lookups(teams):
    """Build lookup structures for team name resolution.

    Returns:
        Tuple of (team_names_set, team_lookup).
        team_names_set: set of all full team names (e.g. 'Norths PHL').
        team_lookup: dict mapping (club_name, grade) -> [full team names].
    """
    team_names_set = set()
    team_lookup = defaultdict(list)
    for t in teams:
        team_names_set.add(t.name)
        team_lookup[(t.club.name, t.grade)].append(t.name)
    return team_names_set, team_lookup


def _resolve_team_name(name, grade, team_names_set, team_lookup, all_teams):
    """Resolve a club name or team name to full team name(s).

    Args:
        name: Club name (e.g. 'Norths') or full team name (e.g. 'Norths PHL').
        grade: Grade string, list of grades, or None.
        team_names_set: Set of all known full team names.
        team_lookup: Dict mapping (club_name, grade) -> [full team names].
        all_teams: List of all Team objects.

    Returns:
        List of resolved full team names. Returns [name] unchanged if no resolution found.
    """
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
            results.extend(_resolve_team_name(name, g, team_names_set, team_lookup, all_teams))
        return results
    # No grade — find all teams from this club
    results = [t.name for t in all_teams if t.club.name == name]
    return results if results else [name]


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
        Tuple of (scope_groups dict, constraint_types dict, constraint_counts dict).
        scope_groups: scope_key (frozenset) -> list of team matchers.
        constraint_types: scope_key (frozenset) -> constraint type string.
        constraint_counts: scope_key (frozenset) -> count override (default 1).
        Each team matcher is ('pair', team1, team2) or ('any', team_name).
    """
    if not forced_games:
        return {}, {}, {}

    team_names_set, team_lookup = _build_team_lookups(teams)

    # scope_key -> list of team matchers
    scope_groups = defaultdict(list)
    # scope_key -> constraint type ('equal', 'lesse', 'greater', 'greatere', 'less')
    constraint_types = {}
    # scope_key -> count override for constraint threshold (default 1)
    constraint_counts = {}

    for entry_idx, entry in enumerate(forced_games):
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

        # Bug A fix: inject grades (plural) into scope if grade (singular) not present
        if grades and 'grade' not in entry:
            scope.append((_KEY_INDEX['grade'], tuple(grades)))

        # When an entry specifies a team pair, give it a unique scope key so it
        # doesn't merge with other entries on the same date/grade. Otherwise
        # multiple pair-specific entries (e.g. 5 locked matchups on the same date)
        # would share one sum==1 constraint instead of each getting their own.
        # Entries without teams (e.g. "any game at Gosford on this date") can
        # merge safely since they use the ('all',) matcher.
        raw_teams = entry.get('teams', [])
        has_team1_team2 = 'team1' in entry or 'team2' in entry
        has_club = 'club' in entry
        if raw_teams or has_team1_team2 or has_club:
            scope.append(('_entry_idx', entry_idx))

        scope_key = frozenset(scope)

        # Bug B fix: detect constraint type collisions on same scope_key
        new_ctype = entry.get('constraint', 'equal')
        if scope_key in constraint_types and constraint_types[scope_key] != new_ctype:
            print(f"  WARNING: Constraint type collision for scope: "
                  f"'{constraint_types[scope_key]}' vs '{new_ctype}'. Using '{new_ctype}'.")
        constraint_types[scope_key] = new_ctype
        if 'count' in entry:
            constraint_counts[scope_key] = entry['count']

        # Build team matcher from 'teams' key or team1/team2
        # If no teams specified, use ('all',) matcher to match any team pair in scope
        raw_teams = entry.get('teams', [])
        if raw_teams:
            if len(raw_teams) == 2:
                resolved_t1 = _resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, teams)
                resolved_t2 = _resolve_team_name(raw_teams[1], effective_grade, team_names_set, team_lookup, teams)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        scope_groups[scope_key].append(('pair', pair[0], pair[1]))
            elif len(raw_teams) == 1:
                resolved = _resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, teams)
                for rt in resolved:
                    scope_groups[scope_key].append(('any', rt))
        elif 'team1' in entry or 'team2' in entry:
            t1_raw = entry.get('team1')
            t2_raw = entry.get('team2')
            if t1_raw and t2_raw:
                resolved_t1 = _resolve_team_name(t1_raw, effective_grade, team_names_set, team_lookup, teams)
                resolved_t2 = _resolve_team_name(t2_raw, effective_grade, team_names_set, team_lookup, teams)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        scope_groups[scope_key].append(('pair', pair[0], pair[1]))
            elif t1_raw:
                for rt in _resolve_team_name(t1_raw, effective_grade, team_names_set, team_lookup, teams):
                    scope_groups[scope_key].append(('any', rt))
            elif t2_raw:
                for rt in _resolve_team_name(t2_raw, effective_grade, team_names_set, team_lookup, teams):
                    scope_groups[scope_key].append(('any', rt))
        elif has_club:
            # Club filter: any game involving a team from this club (resolved
            # via _resolve_team_name, which expands club name → all teams of
            # that club at the effective grade). Mirrors BLOCKED_GAMES.
            club_val = entry['club']
            resolved = _resolve_team_name(club_val, effective_grade, team_names_set, team_lookup, teams)
            for rt in resolved:
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

    return dict(scope_groups), constraint_types, constraint_counts


def _get_matching_forced_scopes(key: tuple, forced_rules: dict) -> list:
    """
    Return every scope_key whose scope + team matcher matches this variable.

    A variable can satisfy multiple forced-game scopes simultaneously — e.g.
    "1 game on Apr 17 at CCHP" (date scope, all-matcher) and "Norths-Gosford
    on some Friday at CCHP" (team scope) are both satisfied by the same
    Norths-Gosford-Apr-17-CCHP variable. The variable must be registered
    against every matching scope so each scope's sum constraint sees it as a
    valid candidate; otherwise the buckets become artificially disjoint and
    the solver loses flexibility.

    Args:
        key: 11-tuple (team1, team2, grade, day, day_slot, time, week, date,
             round_no, field_name, field_location)
        forced_rules: Output of _build_forced_game_rules()

    Returns:
        List of scope_keys this variable matches. Empty list if none match.
    """
    matches = []
    t1, t2 = key[0], key[1]
    sorted_pair = tuple(sorted([t1, t2]))

    for scope_key, team_matchers in forced_rules.items():
        # Check if variable matches this scope
        in_scope = True
        for idx, val in scope_key:
            if not isinstance(idx, int):
                continue  # skip non-index entries (e.g. _entry_idx)
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
        for matcher in team_matchers:
            if matcher[0] == 'all':
                matches.append(scope_key)
                break
            elif matcher[0] == 'pair':
                if sorted_pair[0] == matcher[1] and sorted_pair[1] == matcher[2]:
                    matches.append(scope_key)
                    break
            elif matcher[0] == 'any':
                if t1 == matcher[1] or t2 == matcher[1]:
                    matches.append(scope_key)
                    break

    return matches


def _check_forced_game_status(key: tuple, forced_rules: dict):
    """
    Back-compat wrapper around _get_matching_forced_scopes.

    Returns:
        ('force', scope_key) — variable matches at least one forced rule;
                               scope_key is the FIRST match (iteration order).
        ('normal', None)     — variable doesn't match any forced rule.

    Note: production code (generate_X) uses _get_matching_forced_scopes
    directly so a variable can be registered against ALL matching scopes.
    This wrapper is retained for callers/tests that only need a single
    yes/no answer.
    """
    matches = _get_matching_forced_scopes(key, forced_rules)
    if matches:
        return ('force', matches[0])
    return ('normal', None)


# ============== Blocked Games (No-Play Variable Removal) ==============
# Sister mechanism to FORCED_GAMES. Same config format but different action:
# FORCED_GAMES: variables matching scope AND matching teams → sum == 1 (ensure game happens)
# BLOCKED_GAMES: variables matching scope AND matching teams → eliminated (prevent game)

def _build_blocked_game_rules_with_perennial(blocked_games: list, teams: list) -> tuple:
    """
    Build BLOCKED rules and tag which scope_keys came from perennial entries.

    Returns:
        (rules_dict, perennial_scope_keys)
        rules_dict: scope_key (frozenset) -> list of team matchers (same shape
                    as `_build_blocked_game_rules`).
        perennial_scope_keys: set of scope_keys whose source entry carried
                    `'perennial': True`. These scopes are eligible to be
                    overridden by a matching FORCED_GAMES scope in `generate_X`
                    (see spec-001).

    Note: a single scope_key may be shared by multiple BLOCKED entries
    (matchers concatenated). If ANY of those entries is marked perennial, the
    scope_key is treated as perennial — the perennial-source entry's
    permission-to-be-overridden is preserved.
    """
    if not blocked_games:
        return {}, set()

    team_names_set, team_lookup = _build_team_lookups(teams)

    scope_groups = defaultdict(list)
    perennial_scope_keys: set = set()

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

        # Bug A fix: inject grades (plural) into scope if grade (singular) not present
        if grades and 'grade' not in entry:
            scope.append((_KEY_INDEX['grade'], tuple(grades)))

        scope_key = frozenset(scope)

        if entry.get('perennial') is True:
            perennial_scope_keys.add(scope_key)

        # Build team matchers from 'teams' or 'club' key
        raw_teams = entry.get('teams', [])
        club = entry.get('club')

        if raw_teams:
            if len(raw_teams) == 2:
                resolved_t1 = _resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, teams)
                resolved_t2 = _resolve_team_name(raw_teams[1], effective_grade, team_names_set, team_lookup, teams)
                for rt1 in resolved_t1:
                    for rt2 in resolved_t2:
                        pair = tuple(sorted([rt1, rt2]))
                        scope_groups[scope_key].append(('pair', pair[0], pair[1]))
            elif len(raw_teams) == 1:
                resolved = _resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, teams)
                for rt in resolved:
                    scope_groups[scope_key].append(('any', rt))
        elif club:
            resolved = _resolve_team_name(club, effective_grade, team_names_set, team_lookup, teams)
            for rt in resolved:
                scope_groups[scope_key].append(('any', rt))
        else:
            # Bug D fix: No teams/club = block ALL in scope. Always overwrite.
            # Block-all (empty list) is a superset of any team-specific matchers,
            # so it must always win regardless of entry order.
            scope_groups[scope_key] = []

        desc = entry.get('description', entry.get('reason', f"scope={dict(scope)}"))
        matchers = scope_groups[scope_key]
        matcher_desc = f"{len(matchers)} team matcher(s)" if matchers else "ALL teams (no filter)"
        perennial_tag = ' [perennial]' if entry.get('perennial') is True else ''
        print(f"  Blocked game rule: {desc} -> {matcher_desc}{perennial_tag}")

    return dict(scope_groups), perennial_scope_keys


def _build_blocked_game_rules(blocked_games: list, teams: list) -> dict:
    """
    Build lookup structure from BLOCKED_GAMES config for fast variable filtering.

    Same format as FORCED_GAMES entries. For each scope, collects team matchers.
    A variable matching both the scope AND a team matcher is eliminated.

    Returns:
        Dict mapping scope_key (frozenset) -> list of team matchers.
        Each team matcher is ('pair', team1, team2) or ('any', team_name).

    This is a back-compat shim around `_build_blocked_game_rules_with_perennial`
    that discards the perennial scope-key set. Callers that need
    perennial-aware behaviour (`generate_X`) should use the underlying helper.
    """
    rules, _perennial = _build_blocked_game_rules_with_perennial(blocked_games, teams)
    return rules


def _matching_blocked_scope_keys(key: tuple, blocked_rules: dict) -> list:
    """
    Return every BLOCKED scope_key whose scope + team matcher matches this var.

    Caller-side helper for perennial-vs-FORCED resolution: `generate_X` needs
    to know not just *whether* a variable is blocked but *which* scope_keys
    block it, so it can check whether ALL matching scopes are perennial (and
    thus overridable) versus any being non-perennial (hard block).

    Args:
        key: 11-tuple variable key
        blocked_rules: Output of `_build_blocked_game_rules`

    Returns:
        List of scope_keys this variable is blocked by. Empty list means the
        variable is not blocked.
    """
    matches = []
    t1, t2 = key[0], key[1]
    sorted_pair = tuple(sorted([t1, t2]))

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
            matches.append(scope_key)
            continue

        # Variable is in scope — check if it matches ANY team matcher
        for matcher in team_matchers:
            if matcher[0] == 'pair':
                if sorted_pair[0] == matcher[1] and sorted_pair[1] == matcher[2]:
                    matches.append(scope_key)
                    break
            elif matcher[0] == 'any':
                if t1 == matcher[1] or t2 == matcher[1]:
                    matches.append(scope_key)
                    break

    return matches


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
    return bool(_matching_blocked_scope_keys(key, blocked_rules))


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


# ============== Pre-Solver Config Validation ==============

def _validate_entry_fields(entries, label, valid_dates, valid_locations, valid_field_names,
                           valid_grades, valid_days, team_names_set, team_lookup, all_teams,
                           warnings, fatals):
    """Validate scope field values and date formats in FORCED_GAMES / BLOCKED_GAMES entries."""
    is_forced = (label == 'FORCED_GAMES')

    for i, entry in enumerate(entries):
        desc = entry.get('description', entry.get('reason', f'entry #{i+1}'))

        # Date format check: must be string, not datetime
        date_val = entry.get('date')
        if date_val is not None and not isinstance(date_val, str):
            warnings.append(f"{label} '{desc}': 'date' is {type(date_val).__name__}, not string. "
                           f"Auto-converting to '{date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)}'.")
            if hasattr(date_val, 'strftime'):
                entry['date'] = date_val.strftime('%Y-%m-%d')
            date_val = entry.get('date')

        # Validate date exists in season timeslots
        if date_val and date_val not in valid_dates:
            msg = f"{label} '{desc}': date '{date_val}' has no timeslots in the season."
            if is_forced:
                fatals.append(msg)
            else:
                warnings.append(msg)

        # Validate field_location
        loc = entry.get('field_location')
        if loc and loc not in valid_locations:
            msg = f"{label} '{desc}': venue '{loc}' not found. Known venues: {sorted(valid_locations)}"
            if is_forced:
                fatals.append(msg)
            else:
                warnings.append(msg)

        # Validate field_name
        fname = entry.get('field_name')
        if fname and fname not in valid_field_names:
            msg = f"{label} '{desc}': field_name '{fname}' not found. Known fields: {sorted(valid_field_names)}"
            if is_forced:
                fatals.append(msg)
            else:
                warnings.append(msg)

        # Validate grade(s)
        grade = entry.get('grade')
        if grade and grade not in valid_grades:
            msg = f"{label} '{desc}': grade '{grade}' not found. Known grades: {sorted(valid_grades)}"
            if is_forced:
                fatals.append(msg)
            else:
                warnings.append(msg)
        for g in entry.get('grades', []):
            if g not in valid_grades:
                msg = f"{label} '{desc}': grade '{g}' (in grades list) not found. Known grades: {sorted(valid_grades)}"
                if is_forced:
                    fatals.append(msg)
                else:
                    warnings.append(msg)

        # Validate day
        day = entry.get('day')
        if day and day not in valid_days:
            msg = f"{label} '{desc}': day '{day}' not found. Known days: {sorted(valid_days)}"
            if is_forced:
                fatals.append(msg)
            else:
                warnings.append(msg)

        # Validate constraint type (forced games only)
        if is_forced:
            ctype = entry.get('constraint', 'equal')
            if ctype not in _VALID_CONSTRAINT_TYPES:
                fatals.append(f"{label} '{desc}': invalid constraint type '{ctype}'. "
                             f"Must be one of: {sorted(_VALID_CONSTRAINT_TYPES)}")

        # Validate team names resolve
        grade_for_resolve = entry.get('grades', []) or entry.get('grade')
        for team_name in entry.get('teams', []):
            resolved = _resolve_team_name(team_name, grade_for_resolve, team_names_set, team_lookup, all_teams)
            if resolved == [team_name] and team_name not in team_names_set:
                msg = f"{label} '{desc}': team/club '{team_name}' could not be resolved to any known team."
                if is_forced:
                    fatals.append(msg)
                else:
                    warnings.append(msg)

        club = entry.get('club')
        if club:
            resolved = _resolve_team_name(club, grade_for_resolve, team_names_set, team_lookup, all_teams)
            if resolved == [club] and club not in team_names_set:
                msg = f"{label} '{desc}': club '{club}' could not be resolved to any known team."
                if is_forced:
                    fatals.append(msg)
                else:
                    warnings.append(msg)


def _check_forced_constraint_collisions(forced_games, fatals):
    """Check for two forced game entries with same scope but different constraint types (Bug B).

    Entries with team filters (`teams`, `team1`/`team2`, `club`) are NOT
    collision candidates with no-team-filter entries on the same scope dict —
    they constrain different variable sets and `_build_forced_game_rules`
    already gives each its own scope_key via `_entry_idx`. Only flag
    collisions among entries that share scope-dict AND share the same
    team-filter shape (both no-filter, or both targeting the same team set).
    """
    scope_entries = {}  # scope_key -> (constraint_type, description)

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
        grades = entry.get('grades', [])
        if grades and 'grade' not in entry:
            scope.append((_KEY_INDEX['grade'], tuple(grades)))

        # Skip entries with team filters — they get distinct scope_keys at
        # rule-build time and never share a count constraint with a no-filter
        # entry on the same scope dict.
        has_team_filter = ('teams' in entry or 'team1' in entry
                           or 'team2' in entry or 'club' in entry)
        if has_team_filter:
            continue

        scope_key = frozenset(scope)
        ctype = entry.get('constraint', 'equal')
        desc = entry.get('description', f"scope={dict(scope)}")

        if scope_key in scope_entries:
            prev_ctype, prev_desc = scope_entries[scope_key]
            if prev_ctype != ctype:
                fatals.append(f"Constraint type collision for same forced game scope: "
                             f"'{prev_desc}' uses '{prev_ctype}', but '{desc}' uses '{ctype}'.")
        else:
            scope_entries[scope_key] = (ctype, desc)


def _check_forced_team_conflicts(forced_games, teams, warnings, fatals):
    """Check for same team forced into multiple games on the same date (Bug E)."""
    team_names_set, team_lookup = _build_team_lookups(teams)

    # team_name -> date -> [(entry_desc, constraint_type)]
    team_date_forces = defaultdict(lambda: defaultdict(list))

    for entry in forced_games:
        date_val = entry.get('date')
        if not date_val:
            continue

        grade = entry.get('grade')
        grades = entry.get('grades', [])
        effective_grade = grades if grades else grade
        ctype = entry.get('constraint', 'equal')
        desc = entry.get('description', str(entry))

        # Resolve all teams involved in this forced game
        involved_teams = set()
        raw_teams = entry.get('teams', [])
        if raw_teams:
            for t in raw_teams:
                for rt in _resolve_team_name(t, effective_grade, team_names_set, team_lookup, teams):
                    involved_teams.add(rt)
        elif entry.get('club'):
            for rt in _resolve_team_name(entry['club'], effective_grade, team_names_set, team_lookup, teams):
                involved_teams.add(rt)

        for team in involved_teams:
            team_date_forces[team][date_val].append((desc, ctype))

    for team, date_entries in team_date_forces.items():
        for date_val, entries in date_entries.items():
            if len(entries) < 2:
                continue
            # Check if all are 'equal' (hard conflict) vs softer types
            all_equal = all(ct == 'equal' for _, ct in entries)
            descs = [d for d, _ in entries]
            if all_equal:
                fatals.append(f"Team '{team}' is forced into {len(entries)} games on {date_val} "
                             f"(all constraint='equal'): {descs}")
            else:
                warnings.append(f"Team '{team}' has {len(entries)} forced game entries on {date_val} "
                               f"(mixed constraint types): {descs}")


def _check_forced_venue_team_compat(forced_games, home_field_map, teams, warnings, fatals):
    """Check that forced games at home venues have compatible teams."""
    if not home_field_map:
        return

    team_names_set, team_lookup = _build_team_lookups(teams)
    team_to_club = {t.name: t.club.name for t in teams}

    # Invert home_field_map: venue -> club
    venue_to_home_club = {}
    for club_name, venue in home_field_map.items():
        venue_to_home_club[venue] = club_name

    for entry in forced_games:
        loc = entry.get('field_location')
        if not loc or loc not in venue_to_home_club:
            continue

        home_club = venue_to_home_club[loc]
        desc = entry.get('description', str(entry))

        # If no teams specified, any game at this venue will involve home club
        # (enforced by home_field_map filter in generate_X), so it's fine
        raw_teams = entry.get('teams', [])
        club = entry.get('club')
        if not raw_teams and not club:
            continue

        # Resolve teams and check if any belongs to the home club
        grade = entry.get('grade')
        grades = entry.get('grades', [])
        effective_grade = grades if grades else grade

        resolved_teams = set()
        if raw_teams:
            for t in raw_teams:
                for rt in _resolve_team_name(t, effective_grade, team_names_set, team_lookup, teams):
                    resolved_teams.add(rt)
        if club:
            for rt in _resolve_team_name(club, effective_grade, team_names_set, team_lookup, teams):
                resolved_teams.add(rt)

        team_clubs = {team_to_club.get(t, '') for t in resolved_teams}
        if home_club not in team_clubs:
            fatals.append(f"Forced game '{desc}' at venue '{loc}' requires home club '{home_club}', "
                         f"but teams are from clubs: {sorted(team_clubs - {''})}.")


def _forced_entry_scope_dict(entry):
    """Build a dict of scope-field → value (or tuple of values) from a forced
    entry. Lists are turned into tuples for set membership comparison.
    'grade' alone (singular) is normalised to a 1-tuple if 'grades' isn't given."""
    scope = {}
    for field in _SCOPE_FIELDS:
        if field in entry:
            v = entry[field]
            scope[field] = tuple(v) if isinstance(v, list) else (v,)
    grades = entry.get('grades', [])
    if grades and 'grade' not in entry:
        scope['grade'] = tuple(grades)
    return scope


def _scope_dict_subset(narrow, broad):
    """True if `narrow`'s var set is restricted to a subset of `broad`'s
    by scope-dict alone — i.e. for every key broad pins, narrow pins it
    to a value covered by broad's allowed set, AND narrow has at least
    every key broad has (narrow can pin extra fields, that's still narrower).
    """
    for k, vals in broad.items():
        if k not in narrow:
            return False
        if not set(narrow[k]).issubset(set(vals)):
            return False
    return True


def _forced_entry_pair_set(entry, all_teams, team_names_set, team_lookup):
    """Set of sorted (team1, team2) pairs the entry's team predicate matches,
    restricted to teams of the entry's effective grade(s).

    Returns None when grade scope is missing — pair-set computation requires
    knowing which grade's team list to enumerate.
    """
    grade = entry.get('grade')
    grades = entry.get('grades', [])
    effective_grades = grades if grades else ([grade] if grade else None)
    if not effective_grades:
        return None

    grade_teams = [t.name for t in all_teams if t.grade in effective_grades]
    raw_teams = entry.get('teams', [])
    has_t1_t2 = 'team1' in entry or 'team2' in entry
    club = entry.get('club')

    if not raw_teams and not has_t1_t2 and not club:
        # 'all' matcher — every pair within the grade(s) matches
        pairs = set()
        for i, a in enumerate(grade_teams):
            for b in grade_teams[i+1:]:
                pairs.add(tuple(sorted((a, b))))
        return pairs

    effective_grade = grades if grades else grade
    pairs = set()

    if raw_teams and len(raw_teams) == 2:
        r1 = _resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, all_teams)
        r2 = _resolve_team_name(raw_teams[1], effective_grade, team_names_set, team_lookup, all_teams)
        for a in r1:
            for b in r2:
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
        return pairs
    if raw_teams and len(raw_teams) == 1:
        anchors = set(_resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, all_teams))
        for a in anchors:
            for b in grade_teams:
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
        return pairs
    if has_t1_t2:
        t1 = entry.get('team1')
        t2 = entry.get('team2')
        if t1 and t2:
            r1 = _resolve_team_name(t1, effective_grade, team_names_set, team_lookup, all_teams)
            r2 = _resolve_team_name(t2, effective_grade, team_names_set, team_lookup, all_teams)
            for a in r1:
                for b in r2:
                    if a != b:
                        pairs.add(tuple(sorted((a, b))))
        else:
            anchor_raw = t1 or t2
            anchors = set(_resolve_team_name(anchor_raw, effective_grade, team_names_set, team_lookup, all_teams))
            for a in anchors:
                for b in grade_teams:
                    if a != b:
                        pairs.add(tuple(sorted((a, b))))
        return pairs
    if club:
        anchors = set(_resolve_team_name(club, effective_grade, team_names_set, team_lookup, all_teams))
        for a in anchors:
            for b in grade_teams:
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
        return pairs
    return set()


def _check_forced_scope_subset_consistency(forced_games, teams, warnings, fatals):
    """Phase 21: detect FORCED entries whose var sets are in a subset relation
    AND whose count constraints are mutually unsatisfiable.

    If entry B's var set is a subset of entry A's var set, then
    `sum(B_vars) <= sum(A_vars)`. With `equal` constraints on both, that
    forces `count_B <= count_A`. Anything else is infeasible.

    For non-`equal` constraint types and partial overlap (neither subset nor
    disjoint), this check warns rather than fails — the solver detects those.
    """
    if not forced_games or len(forced_games) < 2:
        return
    team_names_set, team_lookup = _build_team_lookups(teams)

    enriched = []
    for entry in forced_games:
        scope_dict = _forced_entry_scope_dict(entry)
        pair_set = _forced_entry_pair_set(entry, teams, team_names_set, team_lookup)
        ctype = entry.get('constraint', 'equal')
        count = entry.get('count', 1)
        desc = entry.get('description', str(entry))
        enriched.append((entry, scope_dict, pair_set, ctype, count, desc))

    for i, (a_entry, a_scope, a_pairs, a_ctype, a_count, a_desc) in enumerate(enriched):
        for j, (b_entry, b_scope, b_pairs, b_ctype, b_count, b_desc) in enumerate(enriched):
            if i == j:
                continue
            # Skip if pair sets unknown (no grade scope) — too risky to compare.
            if a_pairs is None or b_pairs is None:
                continue
            # Is B narrower than A? (B's vars ⊆ A's vars)
            if not _scope_dict_subset(b_scope, a_scope):
                continue
            if not b_pairs.issubset(a_pairs):
                continue
            # B is narrower than A. Check count compatibility for the both-equal case.
            if a_ctype == 'equal' and b_ctype == 'equal':
                if b_count > a_count:
                    fatals.append(
                        f"Forced game count overlap is infeasible: '{b_desc}' "
                        f"(equal {b_count}) is a subset of '{a_desc}' (equal {a_count}); "
                        f"narrower scope cannot demand more games than the broader one."
                    )
            elif a_ctype == 'equal' and b_ctype == 'greatere':
                if b_count > a_count:
                    fatals.append(
                        f"Forced game count overlap is infeasible: '{b_desc}' "
                        f"(>= {b_count}) is a subset of '{a_desc}' (equal {a_count}); "
                        f"narrower scope cannot demand more games than the broader one."
                    )
            elif a_ctype == 'lesse' and b_ctype == 'equal':
                if b_count > a_count:
                    fatals.append(
                        f"Forced game count overlap is infeasible: '{b_desc}' "
                        f"(equal {b_count}) is a subset of '{a_desc}' (<= {a_count}); "
                        f"narrower scope cannot exceed the broader cap."
                    )
            elif a_ctype == 'lesse' and b_ctype == 'greatere':
                if b_count > a_count:
                    fatals.append(
                        f"Forced game count overlap is infeasible: '{b_desc}' "
                        f"(>= {b_count}) is a subset of '{a_desc}' (<= {a_count}); "
                        f"narrower floor exceeds broader cap."
                    )


def _check_team_capacity(data, warnings, fatals):
    """Check that each team has enough playable dates after all filters and blocks."""
    teams = data['teams']
    timeslots = data['timeslots']
    num_rounds = data.get('num_rounds', {})
    phl_game_times = data.get('phl_game_times', {})
    second_grade_times = data.get('second_grade_times', {})
    home_field_map = data.get('home_field_map', {})
    blocked_games = data.get('blocked_games', [])

    team_names_set, team_lookup = _build_team_lookups(teams)
    team_to_club = {t.name: t.club.name for t in teams}

    # Build venue -> home_club lookup
    venue_to_home_club = {}
    for club_name, venue in home_field_map.items():
        venue_to_home_club[venue] = club_name

    # Build PHL valid slots
    phl_valid_slots = set()
    phl_valid_venue_day_time = set()
    is_simple_format = False
    for venue, venue_data in phl_game_times.items():
        if isinstance(venue_data, dict):
            first_key = next(iter(venue_data.keys()), None)
            if first_key in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
                is_simple_format = True
        break

    if is_simple_format:
        for venue, days in phl_game_times.items():
            for day, times in days.items():
                for t in times:
                    time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)
                    phl_valid_venue_day_time.add((venue, day, time_str))
    else:
        for venue, fields in phl_game_times.items():
            for field, days in fields.items():
                for day, times in days.items():
                    for t in times:
                        time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)
                        phl_valid_slots.add((venue, field, day, time_str))

    # Build 2nd grade valid slots
    second_valid_slots = set()
    if second_grade_times:
        for venue, fields in second_grade_times.items():
            for field, days in fields.items():
                for day, times in days.items():
                    for t in times:
                        time_str = t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)
                        second_valid_slots.add((venue, field, day, time_str))

    phl_only_venues = {'Central Coast Hockey Park'}
    phl_only_days = {'Friday'}

    # Build blocked rules
    blocked_rules = _build_blocked_game_rules(blocked_games, teams) if blocked_games else {}

    # For each team, determine playable dates
    # A date is playable if the team can appear in at least one valid timeslot on that date
    # (considering grade filters, home venue, and blocked games)
    teams_by_grade = defaultdict(list)
    for t in teams:
        teams_by_grade[t.grade].append(t)

    for grade, grade_teams in teams_by_grade.items():
        required = num_rounds.get(grade)
        if not required:
            continue

        is_phl = (grade == 'PHL')
        is_second = (grade == '2nd')

        for team in grade_teams:
            team_name = team.name
            team_club = team_to_club.get(team_name, '')
            playable_dates = set()

            for ts in timeslots:
                if not ts.day:
                    continue

                # Grade-specific time filters (only apply when configured)
                if is_phl and (phl_valid_slots or phl_valid_venue_day_time):
                    if is_simple_format:
                        if (ts.field.location, ts.day, ts.time) not in phl_valid_venue_day_time:
                            continue
                    else:
                        if (ts.field.location, ts.field.name, ts.day, ts.time) not in phl_valid_slots:
                            continue
                elif is_second and second_valid_slots:
                    if (ts.field.location, ts.field.name, ts.day, ts.time) not in second_valid_slots:
                        continue
                else:
                    if ts.field.location in phl_only_venues:
                        continue
                    if ts.day in phl_only_days:
                        continue

                # Home venue filter: team must be from home club for away venues
                home_club = venue_to_home_club.get(ts.field.location)
                if home_club is not None and team_club != home_club:
                    # Team can still play here if their OPPONENT is from the home club,
                    # but we can't know that at capacity-check time. So we check: is this
                    # team FROM the home club? If not, can they play against someone who is?
                    # For capacity purposes, if ANY team from the home club exists in this grade,
                    # this team can play there (as an away team).
                    home_club_teams_in_grade = [t for t in grade_teams if team_to_club.get(t.name) == home_club]
                    if not home_club_teams_in_grade:
                        continue

                # Check blocked games: build a representative key to test
                # We need to check if this team is blocked on this date.
                # Use a simplified check: iterate blocked rules and check scope + team match.
                if blocked_rules:
                    blocked_on_date = True
                    # Check if ALL possible opponents are blocked on this timeslot
                    # For capacity, a date is playable if at least one opponent is available
                    opponents = [t for t in grade_teams if t.name != team_name]
                    any_opponent_ok = False
                    for opp in opponents:
                        pair = tuple(sorted([team_name, opp.name]))
                        test_key = (pair[0], pair[1], grade, ts.day, ts.day_slot, ts.time,
                                   ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
                        if not _is_blocked_by_no_play(test_key, blocked_rules):
                            any_opponent_ok = True
                            break
                    if not any_opponent_ok:
                        continue

                playable_dates.add(ts.date)

            if len(playable_dates) < required:
                fatals.append(f"Team '{team_name}' has only {len(playable_dates)} playable dates "
                             f"but needs {required} rounds. Blocked games or filters are too restrictive.")


def _check_friday_night_feasibility(data, warnings, fatals):
    """Phase 6: Check that Friday night PHL game count targets are achievable
    given the number of Friday timeslots at each venue."""
    timeslots = data.get('timeslots', [])
    defaults = data.get('constraint_defaults', {})

    if not defaults:
        return

    # Count Friday timeslots per venue (unique week+field combos = distinct game slots)
    friday_slots_by_venue = defaultdict(set)
    for ts in timeslots:
        if not ts.day or ts.day != 'Friday':
            continue
        # Each (week, field_name) pair is a distinct game slot on a Friday at that venue
        friday_slots_by_venue[ts.field.location].add((ts.week, ts.field.name, ts.day_slot))

    venue_checks = [
        ('Central Coast Hockey Park', 'gosford_friday_games', 'exact'),
        ('Maitland Park', 'maitland_friday_games', 'exact'),
        ('Newcastle International Hockey Centre', 'max_friday_broadmeadow', 'max'),
    ]

    for venue, config_key, constraint_type in venue_checks:
        required = defaults.get(config_key)
        if required is None:
            continue

        available = len(friday_slots_by_venue.get(venue, set()))

        if constraint_type == 'exact':
            if available < required:
                fatals.append(
                    f"Friday night config '{config_key}' requires exactly {required} PHL game(s) "
                    f"at '{venue}', but only {available} Friday timeslot(s) exist there."
                )
        elif constraint_type == 'max':
            if required > 0 and available == 0:
                warnings.append(
                    f"Friday night config '{config_key}' allows up to {required} PHL game(s) "
                    f"at '{venue}', but no Friday timeslots exist there."
                )
            elif available > 0 and available < required:
                warnings.append(
                    f"Friday night config '{config_key}' allows up to {required} PHL game(s) "
                    f"at '{venue}', but only {available} Friday timeslot(s) exist there. "
                    f"The max constraint will still be satisfiable but fewer slots than expected."
                )


def _check_club_days_availability(data, warnings, fatals):
    """Phase 7: Check that CLUB_DAYS dates have sufficient timeslots for each club's games."""
    club_days = data.get('club_days', {})
    if not club_days:
        return

    timeslots = data.get('timeslots', [])
    teams = data.get('teams', [])

    # Build timeslot index: date_str -> field_key -> count of slots
    slots_by_date_field = defaultdict(lambda: defaultdict(int))
    all_dates = set()
    for ts in timeslots:
        if not ts.day:
            continue
        date_str = ts.date if isinstance(ts.date, str) else ts.date
        all_dates.add(date_str)
        field_key = (ts.field.location, ts.field.name)
        slots_by_date_field[date_str][field_key] += 1

    for club_name, club_date in club_days.items():
        # Convert datetime to string format matching timeslot dates
        if hasattr(club_date, 'strftime'):
            date_str = club_date.strftime('%Y-%m-%d')
        else:
            date_str = str(club_date)

        # Check if date has timeslots
        if date_str not in all_dates:
            fatals.append(
                f"CLUB_DAYS: Club '{club_name}' has club day on {date_str}, "
                f"but no timeslots exist on that date (no-play week or outside season)."
            )
            continue

        # Count teams for this club
        club_teams = [t for t in teams if t.club.name == club_name]
        if not club_teams:
            warnings.append(
                f"CLUB_DAYS: Club '{club_name}' has club day on {date_str}, "
                f"but no teams found for this club."
            )
            continue

        # Club day needs ceil(num_teams / 2) games on ONE field
        games_needed = math.ceil(len(club_teams) / 2)

        # Find max slots at any single field on that date
        field_slots = slots_by_date_field.get(date_str, {})
        max_slots_at_one_field = max(field_slots.values()) if field_slots else 0

        if max_slots_at_one_field < games_needed:
            fatals.append(
                f"CLUB_DAYS: Club '{club_name}' has {len(club_teams)} teams needing "
                f"{games_needed} contiguous game(s) on one field on {date_str}, "
                f"but the most slots at any single field is {max_slots_at_one_field}."
            )


def _check_rounds_vs_weekends(data, warnings, fatals):
    """Phase 8: Check that GRADE_ROUNDS_OVERRIDE does not exceed MAX_WEEKENDS_PER_GRADE."""
    overrides = data.get('grade_rounds_override', {})
    max_weekends = data.get('max_weekends_per_grade', {})

    if not overrides:
        return

    for grade, required_rounds in overrides.items():
        if required_rounds is None or required_rounds <= 0:
            warnings.append(
                f"GRADE_ROUNDS_OVERRIDE: Grade '{grade}' has invalid round count: {required_rounds}."
            )
            continue

        available = max_weekends.get(grade)
        if available is None:
            # Fall back to max_rounds from season config
            available = data.get('max_rounds', 20)

        if required_rounds > available:
            fatals.append(
                f"GRADE_ROUNDS_OVERRIDE: Grade '{grade}' requires {required_rounds} rounds, "
                f"but only {available} weekends are available. Teams cannot play more games "
                f"than available weekends (NoDoubleBookingTeams prevents >1 game/team/week)."
            )


def _check_club_days_vs_blocked(data, warnings, fatals):
    """Phase 9: Check that CLUB_DAYS teams are not fully blocked on their club day date."""
    club_days = data.get('club_days', {})
    blocked_games = data.get('blocked_games', [])

    if not club_days or not blocked_games:
        return

    teams = data.get('teams', [])
    timeslots = data.get('timeslots', [])

    blocked_rules = _build_blocked_game_rules(blocked_games, teams)
    if not blocked_rules:
        return

    # Group teams by grade for opponent lookup
    teams_by_grade = defaultdict(list)
    for t in teams:
        teams_by_grade[t.grade].append(t)

    # Get a representative timeslot for each date (for building test keys)
    timeslots_by_date = defaultdict(list)
    for ts in timeslots:
        if not ts.day:
            continue
        date_str = ts.date if isinstance(ts.date, str) else ts.date
        timeslots_by_date[date_str].append(ts)

    for club_name, club_date in club_days.items():
        if hasattr(club_date, 'strftime'):
            date_str = club_date.strftime('%Y-%m-%d')
        else:
            date_str = str(club_date)

        date_timeslots = timeslots_by_date.get(date_str, [])
        if not date_timeslots:
            continue  # Already caught by _check_club_days_availability

        club_teams = [t for t in teams if t.club.name == club_name]

        for team in club_teams:
            grade = team.grade
            grade_teams = teams_by_grade.get(grade, [])
            opponents = [t for t in grade_teams if t.name != team.name]

            if not opponents:
                continue

            # Check if this team can play ANY opponent on ANY timeslot on this date
            any_game_possible = False
            for ts in date_timeslots:
                for opp in opponents:
                    pair = tuple(sorted([team.name, opp.name]))
                    test_key = (pair[0], pair[1], grade, ts.day, ts.day_slot, ts.time,
                               ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
                    if not _is_blocked_by_no_play(test_key, blocked_rules):
                        any_game_possible = True
                        break
                if any_game_possible:
                    break

            if not any_game_possible:
                fatals.append(
                    f"CLUB_DAYS conflict: Team '{team.name}' must play on club day {date_str} "
                    f"(ClubDayConstraint requires ALL {club_name} teams to play), but all "
                    f"possible opponents are blocked by BLOCKED_GAMES on that date."
                )


def _check_matchup_balance_feasibility(data, warnings, fatals):
    """Phase 10: Check that round count is compatible with the scheduling method for each grade.

    Only checks grades that have an explicit GRADE_ROUNDS_OVERRIDE, since the computed
    num_rounds from the formula is already validated to be feasible by construction.
    """
    overrides = data.get('grade_rounds_override', {})
    num_rounds = data.get('num_rounds', {})
    methods = data.get('grade_scheduling_method', {})
    teams = data.get('teams', [])

    if not overrides:
        return

    # Count teams per grade
    teams_per_grade = defaultdict(int)
    for t in teams:
        teams_per_grade[t.grade] += 1

    for grade in overrides:
        required = num_rounds.get(grade)
        if required is None or required <= 0:
            continue

        team_count = teams_per_grade.get(grade, 0)
        if team_count < 2:
            continue

        method = methods.get(grade, 1)  # Default to method 1 (balanced)

        if method == 1:
            # Balanced round-robin: every team plays every opponent the same number of times.
            # Required rounds must be a multiple of (T-1).
            opponents = team_count - 1
            if required % opponents != 0:
                fatals.append(
                    f"Matchup balance: Grade '{grade}' uses method 1 (balanced round-robin) "
                    f"with {team_count} teams ({opponents} opponents each). "
                    f"Required rounds ({required}) is not a multiple of {opponents}. "
                    f"EnsureEqualGamesAndBalanceMatchUps requires every pair to meet equally, "
                    f"which is impossible with {required} rounds."
                )
        elif method == 2:
            # Maximize games: total game slots = required * team_count / 2 must be an integer
            total_slots = required * team_count
            if total_slots % 2 != 0:
                fatals.append(
                    f"Matchup balance: Grade '{grade}' uses method 2 (maximize games) "
                    f"with {team_count} teams. Required rounds ({required}) * teams ({team_count}) "
                    f"= {total_slots} total team-slots, which is odd. Each game uses 2 team-slots, "
                    f"so total must be even."
                )


def _check_objective_lower_bound(data, warnings, fatals):
    """Phase 15: Check that objective_lower_bound is not set too high.

    The objective is roughly total_scheduled_games - soft_penalties.
    If the bound exceeds the maximum possible games, the solver will
    be silently infeasible.
    """
    bound = data.get('objective_lower_bound')
    if bound is None or not isinstance(bound, (int, float)):
        return

    num_rounds = data.get('num_rounds', {})
    teams = data.get('teams', [])
    if not num_rounds or not teams:
        return

    # Count teams per grade
    teams_per_grade = defaultdict(int)
    for t in teams:
        teams_per_grade[t.grade] += 1

    # Estimate total games: each game involves 2 teams, so
    # total_games = sum(rounds * team_count / 2) for each grade
    estimated_total = 0
    for grade, rounds in num_rounds.items():
        if not isinstance(rounds, (int, float)) or rounds <= 0:
            continue
        team_count = teams_per_grade.get(grade, 0)
        if team_count >= 2:
            estimated_total += int(rounds) * team_count // 2

    if estimated_total > 0 and bound > estimated_total:
        fatals.append(
            f"objective_lower_bound ({bound}) exceeds maximum possible games ({estimated_total}). "
            f"The solver cannot satisfy this bound. Remove or lower objective_lower_bound in config."
        )
    elif bound > 0:
        warnings.append(
            f"objective_lower_bound is set to {bound}. This is positive, which is suspicious "
            f"since soft penalties reduce the objective below total games ({estimated_total}). "
            f"Ensure this bound accounts for expected penalty deductions."
        )


def _check_duplicate_entries(data, warnings, fatals):
    """Phase 16: Check for exact duplicate entries in FORCED_GAMES or BLOCKED_GAMES.

    Duplicates waste config space and usually indicate copy-paste errors.
    Compares entries after removing 'description' and 'reason' keys (labels only).
    """
    for list_name in ('forced_games', 'blocked_games'):
        entries = data.get(list_name, [])
        if not entries:
            continue

        seen = set()
        label = list_name.upper()
        for i, entry in enumerate(entries):
            # Build a comparable key excluding description/reason labels
            comparable_items = []
            for k, v in sorted(entry.items()):
                if k in ('description', 'reason'):
                    continue
                # Convert unhashable types (lists) to tuples
                if isinstance(v, list):
                    v = tuple(v)
                elif isinstance(v, dict):
                    v = tuple(sorted(v.items()))
                comparable_items.append((k, v))

            key = tuple(comparable_items)
            if key in seen:
                desc = entry.get('description') or entry.get('reason') or f'entry #{i+1}'
                warnings.append(
                    f"Duplicate entry in {label}: '{desc}'. "
                    f"Remove the duplicate to keep config clean."
                )
            else:
                seen.add(key)


def _check_anzac_consistency(data, warnings, fatals):
    """Phase 17: Check play_anzac_sunday flag consistency with timeslots.

    The play_anzac_sunday flag is informational only — actual ANZAC handling
    is done via FIELD_UNAVAILABILITIES. This check warns if the flag disagrees
    with what the timeslots show.
    """
    play_anzac = data.get('play_anzac_sunday')
    if play_anzac is None:
        return

    year = data.get('year')
    if year is None:
        return

    # ANZAC Day is always April 25. Find the Sunday on or after April 25.
    anzac_day = date(year, 4, 25)
    days_until_sunday = (6 - anzac_day.weekday()) % 7  # Sunday = 6
    anzac_sunday = anzac_day + timedelta(days=days_until_sunday)
    anzac_sunday_str = anzac_sunday.strftime('%Y-%m-%d')

    # Check if any timeslot falls on ANZAC Sunday
    timeslots = data.get('timeslots', [])
    anzac_in_timeslots = any(t.date == anzac_sunday_str for t in timeslots if t.day)

    if not play_anzac and anzac_in_timeslots:
        warnings.append(
            f"play_anzac_sunday is False but ANZAC Sunday ({anzac_sunday_str}) has timeslots. "
            f"Add {anzac_sunday_str} to FIELD_UNAVAILABILITIES to actually block it."
        )
    elif play_anzac and not anzac_in_timeslots:
        warnings.append(
            f"play_anzac_sunday is True but ANZAC Sunday ({anzac_sunday_str}) has no timeslots. "
            f"Check FIELD_UNAVAILABILITIES — the date may be blocked unintentionally."
        )


def _check_forced_field_double_booking(data, warnings, fatals):
    """Phase 11: Check if two forced game entries with constraint='equal' target the same field+timeslot.

    Two forced entries that both require exactly 1 game at the same (date, field_name,
    field_location, time) slot would conflict with NoDoubleBookingFields, which only
    allows 1 game per field+timeslot. This is guaranteed infeasibility.

    Only checks entries where ALL timeslot-identifying fields are fully specified.
    """
    forced_games = data.get('forced_games', [])
    if not forced_games:
        return

    # Collect fully-specified slot keys for 'equal' constraint entries
    slot_entries = defaultdict(list)
    for entry in forced_games:
        ctype = entry.get('constraint', 'equal')
        if ctype != 'equal':
            continue

        date_val = entry.get('date')
        time_val = entry.get('time')
        field_name = entry.get('field_name')
        field_location = entry.get('field_location')

        # Only check when the timeslot is fully determined
        if date_val is None or time_val is None or field_name is None or field_location is None:
            continue

        slot_key = (date_val, field_name, field_location, time_val)
        desc = entry.get('description', str(entry))
        slot_entries[slot_key].append(desc)

    for slot_key, descs in slot_entries.items():
        if len(descs) >= 2:
            date_val, field_name, field_location, time_val = slot_key
            fatals.append(
                f"Forced game field double-booking: {len(descs)} entries with constraint='equal' "
                f"target the same slot (date={date_val}, field={field_name}, "
                f"venue={field_location}, time={time_val}). "
                f"NoDoubleBookingFields allows only 1 game per field+timeslot. "
                f"Entries: {descs}"
            )


def _check_phl_2nd_adjacency_feasibility(data, warnings, fatals):
    """Phase 12: Check PHL/2nd grade adjacency feasibility.

    The PHLAndSecondGradeAdjacency constraint requires that for each club with both
    PHL and 2nd grade teams, their games must be within 180 minutes at the same location.
    If PHL_GAME_TIMES and SECOND_GRADE_TIMES have no overlapping venue where times are
    within 180 minutes, this constraint is unsatisfiable.
    """
    phl_game_times = data.get('phl_game_times', {})
    second_grade_times = data.get('second_grade_times', {})

    # If second_grade_times is empty/not configured, skip (2nd grade uses all slots)
    if not second_grade_times:
        return
    if not phl_game_times:
        return

    teams = data.get('teams', [])
    if not teams:
        return

    # Find clubs with both PHL and 2nd grade teams
    clubs_by_grade = defaultdict(set)
    for t in teams:
        clubs_by_grade[t.grade].add(t.club.name)

    dual_clubs = clubs_by_grade.get('PHL', set()) & clubs_by_grade.get('2nd', set())
    if not dual_clubs:
        return

    def _parse_time_to_minutes(time_val):
        """Parse a time string like '11:30' or time object to minutes since midnight."""
        if hasattr(time_val, 'strftime'):
            time_str = time_val.strftime('%H:%M')
        else:
            time_str = str(time_val)
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])

    # Build venue+day -> set of times (in minutes) for each grade
    # PHL times by (venue, day)
    phl_venue_day_times = defaultdict(set)
    for venue, fields in phl_game_times.items():
        if isinstance(fields, dict):
            first_key = next(iter(fields.keys()), None)
            if first_key in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
                # Simple format: venue -> day -> times
                for day, times in fields.items():
                    for t in times:
                        phl_venue_day_times[(venue, day)].add(_parse_time_to_minutes(t))
                continue
        # Nested format: venue -> field -> day -> times
        for field, days in fields.items():
            if not isinstance(days, dict):
                continue
            for day, times in days.items():
                for t in times:
                    phl_venue_day_times[(venue, day)].add(_parse_time_to_minutes(t))

    # 2nd grade times by (venue, day)
    second_venue_day_times = defaultdict(set)
    for venue, fields in second_grade_times.items():
        for field, days in fields.items():
            if not isinstance(days, dict):
                continue
            for day, times in days.items():
                for t in times:
                    second_venue_day_times[(venue, day)].add(_parse_time_to_minutes(t))

    # Check if any venue+day has PHL and 2nd grade times within 180 minutes
    has_compatible = False
    for venue_day, phl_times in phl_venue_day_times.items():
        second_times = second_venue_day_times.get(venue_day)
        if not second_times:
            continue
        for pt in phl_times:
            for st in second_times:
                if abs(pt - st) <= 180:
                    has_compatible = True
                    break
            if has_compatible:
                break
        if has_compatible:
            break

    if not has_compatible:
        phl_venues = sorted(set(v for v, _ in phl_venue_day_times.keys()))
        second_venues = sorted(set(v for v, _ in second_venue_day_times.keys()))
        warnings.append(
            f"PHLAndSecondGradeAdjacency may be infeasible: no venue+day combination has "
            f"PHL and 2nd grade times within 180 minutes. "
            f"Clubs affected: {sorted(dual_clubs)}. "
            f"PHL venues: {phl_venues}. 2nd grade venues: {second_venues}."
        )


def _check_perennial_blocked_inclusion(data, warnings, fatals):
    """Phase 13: Check that perennial blocked games (rounds 1-2 at away venues) are included.

    config/defaults.py defines PERENNIAL_BLOCKED_GAMES - standing rules like 'no games at
    Maitland Park or Central Coast in rounds 1-2'. Season configs should include these.
    Warn if any expected combinations are missing.
    """
    blocked_games = data.get('blocked_games', [])

    expected_venues = {'Maitland Park', 'Central Coast Hockey Park'}
    expected_rounds = {1, 2}

    # Check which (round, venue) combinations are covered by blocked_games
    covered = set()
    for entry in blocked_games:
        loc = entry.get('field_location')
        if loc not in expected_venues:
            continue
        round_no = entry.get('round_no')
        if round_no is not None:
            if isinstance(round_no, (list, tuple)):
                for r in round_no:
                    if r in expected_rounds:
                        covered.add((r, loc))
            elif round_no in expected_rounds:
                covered.add((round_no, loc))

    expected = {(r, v) for r in expected_rounds for v in expected_venues}
    missing = expected - covered
    if missing:
        missing_desc = ', '.join(f"round {r} at {v}" for r, v in sorted(missing))
        warnings.append(
            f"Perennial blocked games may be missing: {missing_desc}. "
            f"Rounds 1-2 should block games at Maitland Park and Central Coast Hockey Park. "
            f"Check that PERENNIAL_BLOCKED_GAMES from config/defaults.py is included in BLOCKED_GAMES."
        )


def _check_home_field_map_venues(data, warnings, fatals):
    """Phase 14: Check that home_field_map venues exist in the field definitions.

    If home_field_map maps a club to a venue that doesn't exist in the timeslot field
    definitions, that club will never have home games, likely breaking FiftyFiftyHomeandAway.
    """
    home_field_map = data.get('home_field_map', {})
    if not home_field_map:
        return

    timeslots = data.get('timeslots', [])
    if not timeslots:
        return

    known_venues = {t.field.location for t in timeslots if t.day}
    for club_name, venue in home_field_map.items():
        if venue not in known_venues:
            fatals.append(
                f"home_field_map maps club '{club_name}' to venue '{venue}', "
                f"but this venue does not exist in any timeslot. "
                f"Known venues: {sorted(known_venues)}. "
                f"This club will never have home games, breaking FiftyFiftyHomeandAway."
            )


def validate_draw_keys(keys, timeslots, label='draw'):
    """
    Validate game keys (11-tuples) against current timeslot data.

    Each key is an 11-tuple: (team1, team2, grade, day, day_slot, time, week, date,
    round_no, field_name, field_location). The timeslot portion (indices 3-10) must
    exactly match a generated timeslot for the key to be valid as a solver variable.

    For mismatched keys, diagnoses WHICH fields differ and suggests the correct values.
    This catches stale round_no, shifted day_slot, changed game times, etc.

    Args:
        keys: List/set of 11-tuple game keys (from draw JSON, pickle, etc.)
        timeslots: List of Timeslot objects from the current season data.
        label: Label for diagnostic messages (e.g. 'locked games', 'draw v10.6').

    Returns:
        Tuple of (valid_keys, issues).
        valid_keys: List of keys that match a timeslot exactly.
        issues: List of dicts with diagnostic info for each mismatched key:
            {'key': tuple, 'reason': str, 'field_diffs': dict, 'suggested_key': tuple or None}
    """
    # Build lookup: full timeslot signature -> timeslot fields
    # Key = (day, day_slot, time, week, date, round_no, field_name, field_location)
    ts_signatures = set()
    # Also build partial lookups for diagnosis
    # (date, field_name, field_location, time) -> list of (day, day_slot, week, round_no)
    ts_by_date_field_time = defaultdict(list)
    # (date, field_name, field_location) -> list of (day, day_slot, time, week, round_no)
    ts_by_date_field = defaultdict(list)
    # (date) -> list of full ts tuples
    ts_by_date = defaultdict(list)

    for ts in timeslots:
        if not ts.day:
            continue
        sig = (ts.day, ts.day_slot, ts.time, ts.week, ts.date, ts.round_no,
               ts.field.name, ts.field.location)
        ts_signatures.add(sig)
        ts_by_date_field_time[(ts.date, ts.field.name, ts.field.location, ts.time)].append(
            {'day': ts.day, 'day_slot': ts.day_slot, 'week': ts.week, 'round_no': ts.round_no}
        )
        ts_by_date_field[(ts.date, ts.field.name, ts.field.location)].append(
            {'day': ts.day, 'day_slot': ts.day_slot, 'time': ts.time,
             'week': ts.week, 'round_no': ts.round_no}
        )
        ts_by_date[ts.date].append(sig)

    valid_keys = []
    issues = []

    # Field names at each index in the key tuple (indices 3-10)
    ts_field_names = ['day', 'day_slot', 'time', 'week', 'date', 'round_no',
                      'field_name', 'field_location']

    for key in keys:
        if len(key) < 11:
            issues.append({
                'key': key,
                'reason': f'Short key (len={len(key)}), expected 11-tuple',
                'field_diffs': {},
                'suggested_key': None,
            })
            continue

        t1, t2, grade = key[0], key[1], key[2]
        key_sig = key[3:]  # (day, day_slot, time, week, date, round_no, field_name, field_location)

        if key_sig in ts_signatures:
            valid_keys.append(key)
            continue

        # Key doesn't match — diagnose which fields differ
        date_val = key[7]       # date
        field_name = key[9]     # field_name
        field_loc = key[10]     # field_location
        time_val = key[5]       # time

        field_diffs = {}
        suggested_key = None

        # Try matching on (date, field_name, field_location, time) — most specific
        dft_key = (date_val, field_name, field_loc, time_val)
        matches = ts_by_date_field_time.get(dft_key, [])
        if matches:
            # Found timeslot with same date/field/time — check which ancillary fields differ
            match = matches[0]  # Should be exactly one
            for field, key_idx in [('day', 3), ('day_slot', 4), ('week', 6), ('round_no', 8)]:
                if key[key_idx] != match[field]:
                    field_diffs[field] = {'draw': key[key_idx], 'timeslot': match[field]}
            # Build suggested corrected key
            suggested_key = (t1, t2, grade, match['day'], match['day_slot'], time_val,
                             match['week'], date_val, match['round_no'], field_name, field_loc)
        else:
            # Try matching on (date, field_name, field_location) — maybe time changed
            df_key = (date_val, field_name, field_loc)
            df_matches = ts_by_date_field.get(df_key, [])
            if df_matches:
                # Find closest time match
                best = None
                best_diff = float('inf')
                for m in df_matches:
                    # Compare times as minutes
                    try:
                        key_mins = int(time_val.split(':')[0]) * 60 + int(time_val.split(':')[1])
                        ts_mins = int(m['time'].split(':')[0]) * 60 + int(m['time'].split(':')[1])
                        diff = abs(key_mins - ts_mins)
                    except (ValueError, IndexError):
                        diff = float('inf')
                    if diff < best_diff:
                        best_diff = diff
                        best = m
                if best:
                    field_diffs['time'] = {'draw': time_val, 'timeslot': best['time']}
                    for field, key_idx in [('day', 3), ('day_slot', 4), ('week', 6), ('round_no', 8)]:
                        if key[key_idx] != best[field]:
                            field_diffs[field] = {'draw': key[key_idx], 'timeslot': best[field]}
                    suggested_key = (t1, t2, grade, best['day'], best['day_slot'], best['time'],
                                     best['week'], date_val, best['round_no'], field_name, field_loc)
            else:
                # No timeslots at all on that date/field — check if date exists
                date_matches = ts_by_date.get(date_val, [])
                if not date_matches:
                    field_diffs['date'] = {'draw': date_val, 'timeslot': '(no timeslots on this date)'}
                else:
                    # Date exists but not at this field/location
                    available_fields = set()
                    for sig in date_matches:
                        available_fields.add((sig[6], sig[7]))  # (field_name, field_location)
                    field_diffs['field'] = {
                        'draw': f'{field_name} @ {field_loc}',
                        'timeslot': f'Available: {sorted(available_fields)}'
                    }

        # Build reason string
        if field_diffs:
            diff_strs = []
            for fname, vals in field_diffs.items():
                diff_strs.append(f"{fname}: draw={vals['draw']} vs timeslot={vals['timeslot']}")
            reason = f"Key mismatch for {t1} vs {t2} ({grade}) on {date_val}: {'; '.join(diff_strs)}"
        else:
            reason = (f"No matching timeslot for {t1} vs {t2} ({grade}) on {date_val} "
                      f"at {field_name} ({field_loc}) {time_val}")

        issues.append({
            'key': key,
            'reason': reason,
            'field_diffs': field_diffs,
            'suggested_key': suggested_key,
        })

    return valid_keys, issues


def validate_locked_keys_or_exit(locked_keys, data, source_label='draw'):
    """
    Validate locked game keys against current timeslot data. Exit on mismatches.

    Called pre-solve to ensure all locked keys will match solver variables.
    Prints detailed diagnostics for any mismatches including which fields
    differ and what the correct values should be.

    Args:
        locked_keys: List of 11-tuple keys to validate.
        data: Season data dict (must contain 'timeslots').
        source_label: Label for messages (e.g. 'draws/2026/current.json').
    """
    if not locked_keys:
        return

    timeslots = data.get('timeslots', [])
    if not timeslots:
        return

    valid_keys, issues = validate_draw_keys(locked_keys, timeslots, label=source_label)

    if not issues:
        print(f"  Locked key validation: all {len(valid_keys)} keys match current timeslots (OK)")
        return

    print(f"\n{'='*70}")
    print(f"LOCKED KEY VALIDATION FAILED")
    print(f"{'='*70}")
    print(f"  Source: {source_label}")
    print(f"  Valid keys: {len(valid_keys)}/{len(locked_keys)}")
    print(f"  Mismatched keys: {len(issues)}")

    # Group issues by type of mismatch
    by_field = defaultdict(list)
    for issue in issues:
        fields = list(issue['field_diffs'].keys())
        key_label = ', '.join(fields) if fields else 'unknown'
        by_field[key_label].append(issue)

    for field_label, field_issues in sorted(by_field.items()):
        print(f"\n  --- {field_label} mismatch ({len(field_issues)} games) ---")
        for issue in field_issues[:5]:  # Show first 5 per category
            print(f"    {issue['reason']}")
            if issue['suggested_key']:
                sk = issue['suggested_key']
                print(f"    -> Suggested fix: round_no={sk[8]}, day_slot={sk[4]}, "
                      f"week={sk[6]}, time={sk[5]}")
        if len(field_issues) > 5:
            print(f"    ... and {len(field_issues) - 5} more")

    print(f"\n{'='*70}")
    print(f"The draw file has {len(issues)} game(s) whose keys don't match current")
    print(f"timeslot data. These games would silently fail to lock, causing infeasibility.")
    print(f"")
    print(f"Common causes:")
    if any('round_no' in i['field_diffs'] for i in issues):
        print(f"  - round_no: draw was created with different FIELD_UNAVAILABILITIES")
        print(f"    (changing blocked weekends shifts round numbers)")
    if any('day_slot' in i['field_diffs'] for i in issues):
        print(f"  - day_slot: timeslot generation order changed (fields/times reordered)")
    if any('time' in i['field_diffs'] for i in issues):
        print(f"  - time: DAY_TIME_MAP game times changed since the draw was created")
    if any('date' in i['field_diffs'] for i in issues):
        print(f"  - date: season dates or FIELD_UNAVAILABILITIES changed")
    print(f"")
    print(f"Fix: re-export the draw with current timeslot data, or use")
    print(f"  --repair-locked to auto-fix mismatched keys (matches on date+field+time).")
    print(f"{'='*70}\n")
    sys.exit(1)


def repair_locked_keys(locked_keys, timeslots):
    """
    Attempt to repair mismatched locked keys by finding the closest matching timeslot.

    For each mismatched key, finds a timeslot matching on (date, field_name,
    field_location, time) and substitutes the correct ancillary fields (round_no,
    day_slot, week, day). If a match on time also fails, tries closest time at
    the same (date, field_name, field_location).

    Args:
        locked_keys: List of 11-tuple keys to repair.
        timeslots: List of Timeslot objects.

    Returns:
        Tuple of (repaired_keys, repair_log).
        repaired_keys: List of corrected 11-tuple keys.
        repair_log: List of dicts describing each repair made.
    """
    _, issues = validate_draw_keys(locked_keys, timeslots)
    if not issues:
        return list(locked_keys), []

    issue_keys = {tuple(i['key']) for i in issues}
    repaired_keys = []
    repair_log = []

    for key in locked_keys:
        if tuple(key) not in issue_keys:
            repaired_keys.append(key)
            continue

        # Find the issue for this key
        issue = next(i for i in issues if tuple(i['key']) == tuple(key))
        if issue['suggested_key']:
            repaired_keys.append(issue['suggested_key'])
            repair_log.append({
                'original': key,
                'repaired': issue['suggested_key'],
                'field_diffs': issue['field_diffs'],
            })
        else:
            # Can't repair — keep original (will fail at lock time)
            repaired_keys.append(key)
            repair_log.append({
                'original': key,
                'repaired': None,
                'reason': issue['reason'],
            })

    return repaired_keys, repair_log


def _check_scheduling_feasibility(data, warnings, fatals):
    """Phase 20: Comprehensive scheduling feasibility audit.

    Verifies that the season has enough timeslots at each venue to satisfy all
    hard numerical constraints. Accounts for locked weeks, blocked games, forced
    games, field unavailabilities, and grade-specific time restrictions.

    Checks:
    1. Friday night slot counts vs required PHL games at each venue
    2. Gosford Sunday PHL capacity vs home game demand
    3. Maitland Sunday capacity vs all-grade home game demand
    4. Total Friday PHL demand doesn't exceed total PHL games
    5. Locked week Friday capacity reduction
    6. Sufficient unique playing weeks per grade
    7. Per-grade slot capacity at available venues
    """
    timeslots = data.get('timeslots', [])
    teams = data.get('teams', [])
    num_rounds = data.get('num_rounds', {})
    defaults = data.get('constraint_defaults', {})
    home_field_map = data.get('home_field_map', {})
    locked_weeks = data.get('locked_weeks', set())
    blocked_games = data.get('blocked_games', [])

    if not timeslots or not teams:
        return

    team_to_club = {t.name: t.club.name for t in teams}
    teams_by_grade = defaultdict(list)
    for t in teams:
        teams_by_grade[t.grade].append(t)

    venue_to_home_club = {}
    for club_name, venue in home_field_map.items():
        venue_to_home_club[venue] = club_name

    # Count timeslots by venue, day, and locked status
    friday_slots = defaultdict(set)
    friday_slots_unlocked = defaultdict(set)
    sunday_slots = defaultdict(int)

    for ts in timeslots:
        if not ts.day:
            continue
        venue = ts.field.location
        is_locked = ts.week in locked_weeks

        if ts.day == 'Friday':
            friday_slots[venue].add((ts.week, ts.field.name, ts.day_slot))
            if not is_locked:
                friday_slots_unlocked[venue].add((ts.week, ts.field.name, ts.day_slot))
        elif ts.day == 'Sunday':
            sunday_slots[venue] += 1

    # Calculate total games per grade
    total_games = {}
    for grade, grade_teams in teams_by_grade.items():
        rounds = num_rounds.get(grade, 0)
        if rounds and len(grade_teams) >= 2:
            total_games[grade] = rounds * len(grade_teams) // 2

    gosford_venue = 'Central Coast Hockey Park'
    maitland_venue = 'Maitland Park'
    nihc_venue = 'Newcastle International Hockey Centre'

    gosford_required = defaults.get('gosford_friday_games')
    maitland_required = defaults.get('maitland_friday_games')
    broadmeadow_max = defaults.get('max_friday_broadmeadow')

    # ---- Check 1: Friday night slot counts vs required games ----
    if gosford_required is not None:
        available = len(friday_slots.get(gosford_venue, set()))
        if available < gosford_required:
            fatals.append(
                f"Gosford Friday nights: need exactly {gosford_required} PHL games but only "
                f"{available} Friday timeslot(s) exist at {gosford_venue}."
            )
        # Count blocked Gosford Fridays (date-specific)
        blocked_gosford_fridays = sum(
            1 for bg in blocked_games
            if bg.get('day') == 'Friday'
            and (bg.get('field_location') == gosford_venue or bg.get('club') == 'Gosford')
            and bg.get('date')
        )
        forced_gosford_fridays = sum(
            1 for fg in data.get('forced_games', [])
            if fg.get('day') == 'Friday'
            and fg.get('field_location') == gosford_venue
            and fg.get('constraint', 'equal') == 'equal'
        )
        effective_available = available - blocked_gosford_fridays
        if effective_available < gosford_required and available >= gosford_required:
            warnings.append(
                f"Gosford Friday nights: {available} total slots, but "
                f"{blocked_gosford_fridays} blocked by BLOCKED_GAMES. "
                f"Effective: {effective_available}, required: {gosford_required}. "
                f"{forced_gosford_fridays} forced game(s) override blocks."
            )

    if maitland_required is not None:
        available = len(friday_slots.get(maitland_venue, set()))
        if available < maitland_required:
            fatals.append(
                f"Maitland Friday nights: need exactly {maitland_required} PHL games but only "
                f"{available} Friday timeslot(s) exist at {maitland_venue}."
            )

    # ---- Check 2: Gosford Sunday PHL home game capacity ----
    gosford_phl = [t for t in teams if t.club.name == 'Gosford' and t.grade == 'PHL']
    if gosford_phl and num_rounds.get('PHL', 0) > 0:
        phl_rounds = num_rounds['PHL']
        # Home games needed: ceil(phl_rounds / 2) for 50/50 balance
        home_games_needed = (phl_rounds + 1) // 2
        gosford_friday_home = gosford_required or 0
        sunday_home_needed = max(0, home_games_needed - gosford_friday_home)

        gosford_sunday_weeks = len({ts.week for ts in timeslots
                                     if ts.day == 'Sunday' and ts.field.location == gosford_venue})
        if sunday_home_needed > 0 and gosford_sunday_weeks < sunday_home_needed:
            fatals.append(
                f"Gosford PHL Sunday capacity: need ~{sunday_home_needed} Sunday home games "
                f"(after {gosford_friday_home} Friday home games from target), but only "
                f"{gosford_sunday_weeks} Sunday week(s) available at {gosford_venue}."
            )

    # ---- Check 3: Maitland Sunday capacity vs all-grade home game demand ----
    maitland_teams = [t for t in teams if t.club.name == 'Maitland']
    if maitland_teams and maitland_venue in venue_to_home_club:
        maitland_sunday_total = sunday_slots.get(maitland_venue, 0)
        # Total home games needed across all Maitland teams (~half their rounds)
        total_maitland_home = sum((num_rounds.get(mt.grade, 0) + 1) // 2
                                  for mt in maitland_teams)

        maitland_sunday_weeks = len({ts.week for ts in timeslots
                                      if ts.day == 'Sunday' and ts.field.location == maitland_venue})

        if maitland_sunday_total < total_maitland_home:
            fatals.append(
                f"Maitland Park Sunday capacity: {len(maitland_teams)} teams need "
                f"~{total_maitland_home} home games, but only "
                f"{maitland_sunday_total} Sunday slots across {maitland_sunday_weeks} weeks."
            )

        # Flag if Maitland Sunday weeks are very tight relative to home-game demand.
        if maitland_sunday_weeks > 0 and total_maitland_home > maitland_sunday_weeks * 6:
            warnings.append(
                f"Maitland Park very dense: {total_maitland_home} home games across "
                f"{maitland_sunday_weeks} weeks = {total_maitland_home / maitland_sunday_weeks:.1f} "
                f"games/week average (6 slots/week max)."
            )

    # ---- Check 4: Total Friday PHL demand doesn't exceed total PHL games ----
    total_friday_required = (gosford_required or 0) + (maitland_required or 0) + (broadmeadow_max or 0)
    total_phl_games = total_games.get('PHL', 0)
    if total_friday_required > total_phl_games:
        fatals.append(
            f"Friday PHL game count exceeds total PHL games: "
            f"Gosford ({gosford_required or 0}) + Maitland ({maitland_required or 0}) + "
            f"Broadmeadow ({broadmeadow_max or 0}) = {total_friday_required}, "
            f"but only {total_phl_games} total PHL games."
        )

    # ---- Check 5: Locked weeks — remaining Friday capacity ----
    if locked_weeks and gosford_required:
        unlocked_gosford = len(friday_slots_unlocked.get(gosford_venue, set()))
        if unlocked_gosford < gosford_required:
            warnings.append(
                f"Gosford Fridays after locking weeks {sorted(locked_weeks)}: "
                f"{unlocked_gosford} unlocked slot(s), target {gosford_required}. "
                f"The constraint adjusts for locked games — verify locked weeks "
                f"contain enough Gosford Friday games."
            )

    # ---- Check 6: Sufficient unique playing weeks per grade ----
    all_sunday_weeks = {ts.week for ts in timeslots if ts.day == 'Sunday'}
    all_friday_weeks = {ts.week for ts in timeslots if ts.day == 'Friday'}

    for grade in teams_by_grade:
        rounds = num_rounds.get(grade, 0)
        if not rounds:
            continue
        if grade == 'PHL':
            available_weeks = len(all_sunday_weeks | all_friday_weeks)
        else:
            available_weeks = len(all_sunday_weeks)

        if available_weeks < rounds:
            fatals.append(
                f"Grade '{grade}' needs {rounds} rounds but only {available_weeks} "
                f"unique playing week(s) exist."
            )

    # ---- Check 7: Per-grade slot capacity ----
    for grade in ['3rd', '4th', '5th', '6th']:
        grade_total = total_games.get(grade, 0)
        if not grade_total:
            continue

        has_maitland = any(t.club.name == 'Maitland' for t in teams_by_grade.get(grade, []))

        grade_nihc = sum(1 for ts in timeslots
                         if ts.day == 'Sunday' and ts.field.location == nihc_venue)
        grade_mait = sum(1 for ts in timeslots
                         if ts.day == 'Sunday' and ts.field.location == maitland_venue) if has_maitland else 0

        grade_available = grade_nihc + grade_mait
        if grade_available < grade_total:
            fatals.append(
                f"Grade '{grade}' needs {grade_total} game slots but only "
                f"{grade_available} Sunday slots (NIHC: {grade_nihc}, "
                f"Maitland: {grade_mait})."
            )

    # ---- Summary line ----
    total_all_games = sum(total_games.values())
    total_all_slots = sum(1 for ts in timeslots if ts.day)
    utilization = total_all_games / total_all_slots * 100 if total_all_slots else 0
    print(f"  Scheduling feasibility: {total_all_games} games, {total_all_slots} timeslots "
          f"({utilization:.0f}% utilization)")


def _check_forced_game_feasibility(data, warnings, fatals):
    """Phase 18: Simulate variable filtering for each forced game to detect infeasibility.

    For each FORCED_GAMES entry (constraint='equal' or 'greatere'), simulates what
    generate_X() would do: applies PHL_GAME_TIMES, SECOND_GRADE_TIMES, lower-grade
    exclusions, home_field_map, and BLOCKED_GAMES filters to determine whether at
    least one decision variable would survive.

    This catches forced-vs-blocked clashes, forced games on dates with no valid
    timeslots for the grade, and forced games at venues incompatible with the
    specified teams — all BEFORE the solver starts.
    """
    forced_games = data.get('forced_games', [])
    if not forced_games:
        return

    teams = data['teams']
    timeslots = data['timeslots']
    blocked_games = data.get('blocked_games', [])
    phl_game_times = data.get('phl_game_times', {})
    second_grade_times = data.get('second_grade_times', {})
    home_field_map = data.get('home_field_map', {})

    team_names_set, team_lookup = _build_team_lookups(teams)
    team_to_club = {t.name: t.club.name for t in teams}
    teams_by_grade = defaultdict(list)
    for t in teams:
        teams_by_grade[t.grade].append(t)

    # Build venue -> home_club lookup
    venue_to_home_club = {}
    for club_name, venue in home_field_map.items():
        venue_to_home_club[venue] = club_name

    # Build PHL valid slots (same logic as generate_X)
    phl_valid_slots = set()
    phl_valid_venue_day_time = set()
    is_simple_format = False
    for venue, venue_data in phl_game_times.items():
        if isinstance(venue_data, dict):
            first_key = next(iter(venue_data.keys()), None)
            if first_key in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'):
                is_simple_format = True
        break

    if is_simple_format:
        for venue, days in phl_game_times.items():
            for day, times in days.items():
                for t_val in times:
                    time_str = t_val.strftime('%H:%M') if hasattr(t_val, 'strftime') else str(t_val)
                    phl_valid_venue_day_time.add((venue, day, time_str))
    else:
        for venue, fields in phl_game_times.items():
            for field, days in fields.items():
                for day, times in days.items():
                    for t_val in times:
                        time_str = t_val.strftime('%H:%M') if hasattr(t_val, 'strftime') else str(t_val)
                        phl_valid_slots.add((venue, field, day, time_str))

    # Build 2nd grade valid slots
    second_valid_slots = set()
    if second_grade_times:
        for venue, fields in second_grade_times.items():
            for field, days in fields.items():
                for day, times in days.items():
                    for t_val in times:
                        time_str = t_val.strftime('%H:%M') if hasattr(t_val, 'strftime') else str(t_val)
                        second_valid_slots.add((venue, field, day, time_str))

    phl_only_venues = {'Central Coast Hockey Park'}
    phl_only_days = {'Friday'}

    # Build blocked rules. Track perennial scope_keys so the feasibility
    # simulation here matches `generate_X`: vars matched only by perennial
    # BLOCKED scopes are kept when a FORCED scope also matches (spec-001).
    if blocked_games:
        blocked_rules, perennial_blocked_scopes = _build_blocked_game_rules_with_perennial(
            blocked_games, teams,
        )
    else:
        blocked_rules, perennial_blocked_scopes = {}, set()

    for entry in forced_games:
        ctype = entry.get('constraint', 'equal')
        # Only check entries that REQUIRE at least one variable (equal or greatere)
        if ctype not in ('equal', 'greatere'):
            continue

        desc = entry.get('description', entry.get('reason', str(entry)))
        grade = entry.get('grade')
        grades_list = entry.get('grades', [])
        effective_grades = grades_list if grades_list else ([grade] if grade else [])

        # Resolve teams from the forced game entry
        raw_teams = entry.get('teams', [])
        club = entry.get('club')

        # For each effective grade, resolve the team pairs that could satisfy this forced game
        for eff_grade in (effective_grades or [None]):
            is_phl = (eff_grade == 'PHL')
            is_second = (eff_grade == '2nd')

            # Resolve which team pairs could match
            candidate_pairs = []
            if raw_teams:
                if len(raw_teams) == 2:
                    resolved_t1 = _resolve_team_name(raw_teams[0], eff_grade, team_names_set, team_lookup, teams)
                    resolved_t2 = _resolve_team_name(raw_teams[1], eff_grade, team_names_set, team_lookup, teams)
                    for rt1 in resolved_t1:
                        for rt2 in resolved_t2:
                            if rt1 != rt2:
                                candidate_pairs.append(tuple(sorted([rt1, rt2])))
                elif len(raw_teams) == 1:
                    resolved = _resolve_team_name(raw_teams[0], eff_grade, team_names_set, team_lookup, teams)
                    grade_teams = teams_by_grade.get(eff_grade, []) if eff_grade else teams
                    for rt in resolved:
                        for opp in grade_teams:
                            if opp.name != rt:
                                candidate_pairs.append(tuple(sorted([rt, opp.name])))
            elif club:
                resolved = _resolve_team_name(club, eff_grade, team_names_set, team_lookup, teams)
                grade_teams = teams_by_grade.get(eff_grade, []) if eff_grade else teams
                for rt in resolved:
                    for opp in grade_teams:
                        if opp.name != rt:
                            candidate_pairs.append(tuple(sorted([rt, opp.name])))
            else:
                # No teams specified — any pair in this grade
                grade_teams = teams_by_grade.get(eff_grade, []) if eff_grade else teams
                for i, t1 in enumerate(grade_teams):
                    for t2 in grade_teams[i+1:]:
                        candidate_pairs.append(tuple(sorted([t1.name, t2.name])))

            if not candidate_pairs:
                # No teams resolved — already caught by Phase 1 team validation
                continue

            # Build scope fields for timeslot matching
            scope_date = entry.get('date')
            scope_day = entry.get('day')
            scope_field_name = entry.get('field_name')
            scope_field_location = entry.get('field_location')
            scope_time = entry.get('time')
            scope_week = entry.get('week')
            scope_round_no = entry.get('round_no')
            scope_day_slot = entry.get('day_slot')

            # Find timeslots matching the scope
            surviving_vars = 0
            elimination_reasons = defaultdict(int)
            # Track vars removed by BLOCKED_GAMES (blocked takes priority over forced)
            blocked_removed_vars = 0

            for ts in timeslots:
                if not ts.day:
                    continue

                # Match scope fields against timeslot
                if scope_date and ts.date != scope_date:
                    continue
                if scope_day and ts.day != scope_day:
                    continue
                if scope_field_name and ts.field.name != scope_field_name:
                    continue
                if scope_field_location and ts.field.location != scope_field_location:
                    continue
                if scope_time and ts.time != scope_time:
                    continue
                if scope_week and ts.week != scope_week:
                    continue
                if scope_round_no and ts.round_no != scope_round_no:
                    continue
                if scope_day_slot and ts.day_slot != scope_day_slot:
                    continue

                # Apply grade-specific time filters (same logic as generate_X)
                if is_phl and (phl_valid_slots or phl_valid_venue_day_time):
                    if is_simple_format:
                        if (ts.field.location, ts.day, ts.time) not in phl_valid_venue_day_time:
                            elimination_reasons['PHL_GAME_TIMES filter'] += 1
                            continue
                    else:
                        if (ts.field.location, ts.field.name, ts.day, ts.time) not in phl_valid_slots:
                            elimination_reasons['PHL_GAME_TIMES filter'] += 1
                            continue
                elif is_second and second_valid_slots:
                    if (ts.field.location, ts.field.name, ts.day, ts.time) not in second_valid_slots:
                        elimination_reasons['SECOND_GRADE_TIMES filter'] += 1
                        continue
                elif eff_grade and eff_grade not in ('PHL', '2nd'):
                    if ts.field.location in phl_only_venues:
                        elimination_reasons['PHL-only venue (Gosford)'] += 1
                        continue
                    if ts.day in phl_only_days:
                        elimination_reasons['PHL-only day (Friday)'] += 1
                        continue

                # Check each candidate pair against home_field_map and blocked games
                for pair in candidate_pairs:
                    t1_name, t2_name = pair

                    # Home-field restriction
                    home_club = venue_to_home_club.get(ts.field.location)
                    if home_club is not None:
                        t1_club = team_to_club.get(t1_name, '')
                        t2_club = team_to_club.get(t2_name, '')
                        if t1_club != home_club and t2_club != home_club:
                            elimination_reasons['home_field_map filter'] += 1
                            continue

                    # Blocked games take priority over forced games. A var that
                    # matches both forced and blocked is eliminated (not created).
                    test_grade = eff_grade or entry.get('grade', '')
                    test_key = (t1_name, t2_name, test_grade, ts.day, ts.day_slot,
                                ts.time, ts.week, ts.date, ts.round_no,
                                ts.field.name, ts.field.location)

                    if blocked_rules:
                        match_scopes = _matching_blocked_scope_keys(test_key, blocked_rules)
                        if match_scopes:
                            # spec-001: if every matching BLOCKED scope is
                            # perennial, this FORCED entry rescues the var.
                            # Only non-perennial blocks (or a mix) eliminate.
                            all_perennial = all(
                                sk in perennial_blocked_scopes for sk in match_scopes
                            )
                            if not all_perennial:
                                blocked_removed_vars += 1
                                elimination_reasons['BLOCKED_GAMES filter'] += 1
                                continue
                            # else: FORCED overrides perennial — var survives

                    surviving_vars += 1

            if surviving_vars == 0:
                grade_label = eff_grade or 'all grades'
                # Build detailed diagnosis
                diag_parts = [f"Forced game '{desc}' (grade={grade_label}) has ZERO playable variables."]
                if elimination_reasons:
                    diag_parts.append("Filters that eliminated candidate variables:")
                    for reason, count in sorted(elimination_reasons.items(), key=lambda x: -x[1]):
                        diag_parts.append(f"  {reason}: {count} eliminated")
                else:
                    diag_parts.append("No timeslots matched the scope fields at all.")
                    if scope_date:
                        # Check if date exists in timeslots
                        date_exists = any(ts.date == scope_date for ts in timeslots if ts.day)
                        if not date_exists:
                            diag_parts.append(f"  Date '{scope_date}' has no timeslots (FIELD_UNAVAILABILITIES or outside season).")
                        elif scope_day:
                            days_on_date = {ts.day for ts in timeslots if ts.day and ts.date == scope_date}
                            diag_parts.append(f"  Date '{scope_date}' has timeslots on: {sorted(days_on_date)}, but forced day is '{scope_day}'.")
                    if scope_field_location:
                        loc_exists = any(ts.field.location == scope_field_location and ts.date == scope_date
                                        for ts in timeslots if ts.day) if scope_date else \
                                    any(ts.field.location == scope_field_location for ts in timeslots if ts.day)
                        if not loc_exists:
                            diag_parts.append(f"  Venue '{scope_field_location}' has no timeslots"
                                            + (f" on date '{scope_date}'." if scope_date else " in the season."))

                fatals.append(' '.join(diag_parts))

            elif blocked_removed_vars > 0 and surviving_vars > 0:
                # Some vars were removed by blocked games but forced game still has
                # enough remaining vars. Flag as informational warning.
                grade_label = eff_grade or 'all grades'
                warnings.append(
                    f"Forced game '{desc}' (grade={grade_label}) partially overlaps with "
                    f"BLOCKED_GAMES: {blocked_removed_vars} variable(s) removed by blocked rules, "
                    f"{surviving_vars} remaining. Blocked games take priority — verify the "
                    f"forced game still has enough candidate slots."
                )


def _check_forced_blocked_scope_overlap(data, warnings, fatals):
    """Phase 19: Detect direct scope overlaps between FORCED_GAMES and BLOCKED_GAMES.

    Finds cases where a blocked game entry's scope is a superset of (or equal to) a
    forced game entry's scope, AND the blocked teams include the forced teams.
    These are direct configuration contradictions that guarantee infeasibility.

    This is a fast config-level check (no timeslot simulation needed).
    """
    forced_games = data.get('forced_games', [])
    blocked_games = data.get('blocked_games', [])
    if not forced_games or not blocked_games:
        return

    teams = data['teams']
    team_names_set, team_lookup = _build_team_lookups(teams)
    team_to_club = {t.name: t.club.name for t in teams}
    teams_by_grade = defaultdict(list)
    for t in teams:
        teams_by_grade[t.grade].append(t)

    # Build date -> round_no lookup for cross-referencing
    timeslots = data.get('timeslots', [])
    date_to_rounds = defaultdict(set)
    round_to_dates = defaultdict(set)
    for ts in timeslots:
        if ts.day and ts.date and ts.round_no:
            date_to_rounds[ts.date].add(ts.round_no)
            round_to_dates[ts.round_no].add(ts.date)

    for f_entry in forced_games:
        f_ctype = f_entry.get('constraint', 'equal')
        if f_ctype not in ('equal', 'greatere'):
            continue

        f_desc = f_entry.get('description', str(f_entry))
        f_grade = f_entry.get('grade')
        f_grades = f_entry.get('grades', [])

        # Resolve forced teams
        f_raw_teams = f_entry.get('teams', [])
        f_club = f_entry.get('club')
        f_effective_grade = f_grades if f_grades else f_grade

        f_resolved_teams = set()
        if f_raw_teams:
            for t in f_raw_teams:
                for rt in _resolve_team_name(t, f_effective_grade, team_names_set, team_lookup, teams):
                    f_resolved_teams.add(rt)
        elif f_club:
            for rt in _resolve_team_name(f_club, f_effective_grade, team_names_set, team_lookup, teams):
                f_resolved_teams.add(rt)

        for b_entry in blocked_games:
            b_desc = b_entry.get('description', b_entry.get('reason', str(b_entry)))
            b_grade = b_entry.get('grade')
            b_grades = b_entry.get('grades', [])

            # Check scope overlap: for each scope field present in BOTH entries,
            # their values must overlap. Also cross-reference date vs round_no.
            scope_overlaps = True
            for field in _SCOPE_FIELDS:
                if field not in b_entry or field not in f_entry:
                    continue  # One side doesn't constrain this field — not a mismatch
                b_val = b_entry[field]
                f_val = f_entry[field]
                # Check if values overlap
                if isinstance(b_val, list) and isinstance(f_val, list):
                    if not set(b_val) & set(f_val):
                        scope_overlaps = False
                        break
                elif isinstance(b_val, list):
                    if f_val not in b_val:
                        scope_overlaps = False
                        break
                elif isinstance(f_val, list):
                    if b_val not in f_val:
                        scope_overlaps = False
                        break
                else:
                    if b_val != f_val:
                        scope_overlaps = False
                        break

            if not scope_overlaps:
                continue

            # Cross-reference date vs round_no when one has date and the other has round_no
            # (but not both). E.g., forced has date='2026-08-16' and blocked has round_no=1.
            if 'date' in f_entry and 'round_no' in b_entry and 'date' not in b_entry and 'round_no' not in f_entry:
                f_date = f_entry['date']
                b_round = b_entry['round_no']
                f_rounds = date_to_rounds.get(f_date, set())
                b_rounds = {b_round} if not isinstance(b_round, list) else set(b_round)
                if f_rounds and not (f_rounds & b_rounds):
                    continue  # Date and round_no don't overlap
            elif 'round_no' in f_entry and 'date' in b_entry and 'round_no' not in b_entry and 'date' not in f_entry:
                f_round = f_entry['round_no']
                b_date = b_entry['date']
                f_rounds = {f_round} if not isinstance(f_round, list) else set(f_round)
                b_rounds = date_to_rounds.get(b_date, set())
                if b_rounds and not (f_rounds & b_rounds):
                    continue  # Round_no and date don't overlap

            # Check grade overlap
            f_grade_set = set(f_grades) if f_grades else ({f_grade} if f_grade else set())
            b_grade_set = set(b_grades) if b_grades else ({b_grade} if b_grade else set())
            if f_grade_set and b_grade_set and not (f_grade_set & b_grade_set):
                continue

            # Scope overlaps — now check if blocked teams cover forced teams
            b_raw_teams = b_entry.get('teams', [])
            b_club = b_entry.get('club')
            b_effective_grade = b_grades if b_grades else b_grade

            if not b_raw_teams and not b_club:
                # Blocked entry blocks ALL teams in scope — direct clash if scope overlaps.
                # Only flag if both entries share at least 2 concrete scope fields
                # (to avoid noisy warnings from broad entries like day=Friday + field_location).
                # Cross-referenced date/round_no counts as a shared field since we already
                # confirmed they refer to the same playing round.
                shared_scope = sum(1 for f in _SCOPE_FIELDS if f in f_entry and f in b_entry)
                has_date_round_xref = (
                    ('date' in f_entry and 'round_no' in b_entry and 'date' not in b_entry) or
                    ('round_no' in f_entry and 'date' in b_entry and 'round_no' not in b_entry)
                )
                if has_date_round_xref:
                    shared_scope += 1
                if shared_scope >= 2:
                    warnings.append(
                        f"Forced/blocked scope overlap: forced '{f_desc}' overlaps with "
                        f"blocked '{b_desc}' which blocks ALL teams in that scope. "
                        f"Blocked games take priority — variables in the overlap will be "
                        f"removed from the forced game's candidate pool. This may leave "
                        f"the forced game with zero playable variables."
                    )
                continue

            # Resolve blocked teams
            b_resolved_teams = set()
            if b_raw_teams:
                for t in b_raw_teams:
                    for rt in _resolve_team_name(t, b_effective_grade, team_names_set, team_lookup, teams):
                        b_resolved_teams.add(rt)
            elif b_club:
                for rt in _resolve_team_name(b_club, b_effective_grade, team_names_set, team_lookup, teams):
                    b_resolved_teams.add(rt)

            if not f_resolved_teams or not b_resolved_teams:
                continue

            # Check if blocked teams include any forced teams.
            # Only warn if both entries share at least 2 scope fields or the forced entry
            # has a specific date (to avoid noisy warnings from broad entries).
            shared_scope = sum(1 for f in _SCOPE_FIELDS if f in f_entry and f in b_entry)
            if shared_scope < 2 and 'date' not in f_entry:
                continue

            overlap = f_resolved_teams & b_resolved_teams
            if overlap:
                if len(f_raw_teams) == 2:
                    warnings.append(
                        f"Forced/blocked team clash: forced '{f_desc}' requires teams "
                        f"{sorted(f_resolved_teams)}, but blocked '{b_desc}' blocks "
                        f"{sorted(overlap)} on the same scope. This may reduce "
                        f"available variables for the forced game."
                    )
                else:
                    warnings.append(
                        f"Forced/blocked team clash: forced '{f_desc}' involves "
                        f"{sorted(f_resolved_teams)}, but blocked '{b_desc}' blocks "
                        f"{sorted(overlap)} on the same scope."
                    )


def _validate_stages(data, warnings, fatals):
    """Phase 22 (Phase 7b): validate `data['solver_stages']` if set.

    No-op when the key is absent — `load_solver_stages` will populate it
    later from `DEFAULT_STAGES`.
    """
    stages = data.get('solver_stages')
    if stages is None:
        return
    from constraints.stages import validate_solver_stages
    errors = validate_solver_stages(stages)
    for err in errors:
        fatals.append(f"solver_stages: {err}")


def validate_game_config(data: dict) -> None:
    """
    Pre-validate FORCED_GAMES, BLOCKED_GAMES, and related config before
    variable generation. Catches config errors that would cause infeasibility
    or silent misbehavior.

    Called from generate_X() before variable creation.

    FATAL errors -> sys.exit(1)
    WARNINGS -> print and continue
    """
    teams = data['teams']
    timeslots = data['timeslots']
    forced_games = data.get('forced_games', [])
    blocked_games = data.get('blocked_games', [])

    # Filter out forced/blocked entries whose dates fall entirely in locked weeks.
    # Locked weeks are already solved — forced/blocked rules don't apply to them.
    # We store filtered lists back into data so sub-functions also see the filtered versions.
    locked_weeks = data.get('locked_weeks', set())
    if locked_weeks:
        date_to_week = {}
        for t in timeslots:
            if t.date and t.week:
                date_to_week[t.date] = t.week
        def _not_in_locked_week(entry):
            d = entry.get('date')
            if not d:
                return True  # no date → keep (applies across all weeks)
            if isinstance(d, list):
                return any(date_to_week.get(str(dd), 0) not in locked_weeks for dd in d)
            return date_to_week.get(str(d), 0) not in locked_weeks
        forced_before = len(forced_games)
        blocked_before = len(blocked_games)
        forced_games = [e for e in forced_games if _not_in_locked_week(e)]
        blocked_games = [e for e in blocked_games if _not_in_locked_week(e)]
        forced_skipped = forced_before - len(forced_games)
        blocked_skipped = blocked_before - len(blocked_games)
        if forced_skipped or blocked_skipped:
            print(f"  Locked weeks {sorted(locked_weeks)}: skipped {forced_skipped} forced + "
                  f"{blocked_skipped} blocked game entries from validation")
        # Store filtered lists so sub-functions (phases 5+) also use them
        data['forced_games'] = forced_games
        data['blocked_games'] = blocked_games

    # Check if ANY validation is needed (forced/blocked games OR other config mechanisms)
    club_days = data.get('club_days', {})
    constraint_defaults = data.get('constraint_defaults', {})
    grade_rounds_override = data.get('grade_rounds_override', {})
    home_field_map = data.get('home_field_map', {})
    phl_game_times = data.get('phl_game_times', {})
    has_config_to_validate = (forced_games or blocked_games or club_days
                              or constraint_defaults or grade_rounds_override
                              or home_field_map or phl_game_times)

    if not has_config_to_validate:
        return

    print(f"\n{'='*70}")
    print("CONFIG VALIDATION")
    print(f"{'='*70}")

    team_names_set, team_lookup = _build_team_lookups(teams)

    # Collect valid values from season data
    valid_dates = {t.date for t in timeslots if t.day}
    valid_locations = {t.field.location for t in timeslots if t.day}
    valid_field_names = {t.field.name for t in timeslots if t.day}
    valid_grades = {t.grade for t in teams}
    valid_days = {t.day for t in timeslots if t.day}

    warnings = []
    fatals = []

    # Phase 1: Validate individual entry field values (Bug C + date format + constraint type)
    _validate_entry_fields(forced_games, 'FORCED_GAMES', valid_dates, valid_locations,
                          valid_field_names, valid_grades, valid_days,
                          team_names_set, team_lookup, teams, warnings, fatals)
    _validate_entry_fields(blocked_games, 'BLOCKED_GAMES', valid_dates, valid_locations,
                          valid_field_names, valid_grades, valid_days,
                          team_names_set, team_lookup, teams, warnings, fatals)

    # Phase 2: Check forced constraint type collisions (Bug B)
    _check_forced_constraint_collisions(forced_games, fatals)

    # Phase 3: Check forced game team conflicts (Bug E)
    _check_forced_team_conflicts(forced_games, teams, warnings, fatals)

    # Phase 4: Check forced game venue-team compatibility
    home_field_map = data.get('home_field_map', {})
    _check_forced_venue_team_compat(forced_games, home_field_map, teams, warnings, fatals)

    # Phase 5: Team capacity check
    _check_team_capacity(data, warnings, fatals)

    # Phase 6: Friday night game count feasibility
    _check_friday_night_feasibility(data, warnings, fatals)

    # Phase 7: Club days date availability
    _check_club_days_availability(data, warnings, fatals)

    # Phase 8: Grade rounds override vs max weekends
    _check_rounds_vs_weekends(data, warnings, fatals)

    # Phase 9: Club days vs blocked games conflict
    _check_club_days_vs_blocked(data, warnings, fatals)

    # Phase 10: Matchup balance feasibility
    _check_matchup_balance_feasibility(data, warnings, fatals)

    # Phase 15: Objective lower bound sanity check
    _check_objective_lower_bound(data, warnings, fatals)

    # Phase 16: Duplicate config entries
    _check_duplicate_entries(data, warnings, fatals)

    # Phase 17: ANZAC Sunday consistency
    _check_anzac_consistency(data, warnings, fatals)

    # Phase 11: Forced game field/timeslot double-booking
    _check_forced_field_double_booking(data, warnings, fatals)

    # Phase 12: PHL/2nd grade adjacency feasibility
    _check_phl_2nd_adjacency_feasibility(data, warnings, fatals)

    # Phase 13: Perennial blocked games inclusion
    _check_perennial_blocked_inclusion(data, warnings, fatals)

    # Phase 14: home_field_map venue existence
    _check_home_field_map_venues(data, warnings, fatals)

    # Phase 18: Forced game feasibility (simulate variable filtering)
    _check_forced_game_feasibility(data, warnings, fatals)

    # Phase 19: Direct forced/blocked scope overlap detection
    _check_forced_blocked_scope_overlap(data, warnings, fatals)

    # Phase 20: Comprehensive scheduling feasibility audit
    _check_scheduling_feasibility(data, warnings, fatals)

    # Phase 21: Forced scope subset consistency
    _check_forced_scope_subset_consistency(forced_games, teams, warnings, fatals)

    # Phase 22: SOLVER_STAGES validation (Phase 7b)
    _validate_stages(data, warnings, fatals)

    # Report
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")

    if fatals:
        for f in fatals:
            print(f"  FATAL: {f}")
        print(f"\n{'='*70}")
        print(f"{len(fatals)} fatal error(s) found. Fix config before running.")
        print(f"{'='*70}\n")
        sys.exit(1)

    print(f"  Config validation passed ({len(warnings)} warning(s))")
    print(f"{'='*70}\n")




def generate_X(model, data: dict) -> Tuple[Dict, Dict]:
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
        Tuple of (X, conflicts)
        - X: Dict of decision variables for real timeslots
        - conflicts: Dict of team conflicts
    """
    teams = data['teams']
    timeslots = data['timeslots']

    # Pre-validate config before creating variables
    validate_game_config(data)

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
    
    # Locked weeks — forced/blocked rules don't apply to these (already solved)
    locked_weeks = data.get('locked_weeks', set())

    # Build forced games lookup from config
    forced_game_rules, forced_constraint_types, forced_constraint_counts = _build_forced_game_rules(data.get('forced_games', []), teams)

    # Build blocked games (no-play) lookup from config.
    # `perennial_blocked_scopes` tracks scope_keys whose source entry was marked
    # `'perennial': True` (e.g. the rounds-1-2 Broadmeadow-only rule from
    # `config/defaults.py::PERENNIAL_BLOCKED_GAMES`). A variable matched by a
    # perennial scope is overridable by any matching FORCED_GAMES scope —
    # FORCED entries are deliberate convenor exceptions to perennial defaults.
    # See spec-001 (docs/todo/done/spec-001-r1r2-broadmeadow-forced-exempt.md).
    blocked_game_rules, perennial_blocked_scopes = _build_blocked_game_rules_with_perennial(
        data.get('blocked_games', []), teams,
    )
    perennial_exempt_count = 0  # vars kept because a FORCED scope beat a perennial block

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
            
        for t in timeslots:
            if not t.day:
                continue
            
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

            # Skip forced/blocked game rules for locked weeks — those games are
            # already solved and will be locked by the downstream locking logic.
            in_locked_week = locked_weeks and t.week in locked_weeks

            # Check blocked game rules. Season-specific BLOCKED scopes always
            # eliminate the variable (no override). Perennial BLOCKED scopes —
            # those marked `'perennial': True` in their source entry (e.g.
            # PERENNIAL_BLOCKED_GAMES from config/defaults.py) — are *defaults*
            # the convenor can override with a matching FORCED_GAMES entry. So
            # a variable matched by a perennial scope is kept iff it also
            # matches a FORCED scope. If it's blocked by ANY non-perennial
            # scope as well, the non-perennial block wins. (spec-001)
            matched_block_scopes = (
                _matching_blocked_scope_keys(key, blocked_game_rules)
                if blocked_game_rules and not in_locked_week else []
            )
            if matched_block_scopes:
                forced_matches_block = (
                    _get_matching_forced_scopes(key, forced_game_rules)
                    if forced_game_rules else []
                )
                all_perennial = all(
                    sk in perennial_blocked_scopes for sk in matched_block_scopes
                )
                if forced_matches_block and all_perennial:
                    # FORCED overrides perennial-only block — keep the variable
                    # and register it against every matching FORCED scope so
                    # each scope's sum constraint sees it as a candidate.
                    var = model.NewBoolVar(
                        f'X_{t1_name}_{t2_name}_{t.day}_{t.time}_{t.week}_{t.field.name}'
                    )
                    X[key] = var
                    for scope_key in forced_matches_block:
                        forced_scope_vars[scope_key].append(var)
                    forced_vars_forced += 1
                    perennial_exempt_count += 1
                    continue
                # Either no FORCED match or at least one non-perennial block — eliminate.
                blocked_vars_skipped += 1
                continue

            # Check forced game rules - track matching vars for sum == N constraint.
            # A variable may match multiple forced scopes (e.g. a date-scope and an
            # overlapping team-scope) and must be registered against ALL matching
            # scopes so each scope's sum constraint sees it as a candidate.
            # Non-matching vars are left alone (not eliminated).
            if forced_game_rules and not in_locked_week:
                matching_scopes = _get_matching_forced_scopes(key, forced_game_rules)
                if matching_scopes:
                    var = model.NewBoolVar(f'X_{t1_name}_{t2_name}_{t.day}_{t.time}_{t.week}_{t.field.name}')
                    X[key] = var
                    for scope_key in matching_scopes:
                        forced_scope_vars[scope_key].append(var)
                    forced_vars_forced += 1
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
        for entry_idx, entry in enumerate(forced_games):
            scope = []
            for field in _SCOPE_FIELDS:
                if field in entry:
                    val = entry[field]
                    idx = _KEY_INDEX[field]
                    if isinstance(val, list):
                        scope.append((idx, tuple(val)))
                    else:
                        scope.append((idx, val))
            grades = entry.get('grades', [])
            if grades and 'grade' not in entry:
                scope.append((_KEY_INDEX['grade'], tuple(grades)))
            raw_teams = entry.get('teams', [])
            has_team1_team2 = 'team1' in entry or 'team2' in entry
            has_club = 'club' in entry
            if raw_teams or has_team1_team2 or has_club:
                scope.append(('_entry_idx', entry_idx))
            sk = frozenset(scope)
            scope_to_entry[sk] = entry

        print(f"\n{'='*70}")
        print(f"FATAL: {len(missing_forced)} forced game(s) have NO playable variables!")
        print(f"The solver cannot satisfy these forced games. Fix config before running.")
        print(f"{'='*70}")
        for sk in missing_forced:
            scope_desc = ', '.join(f"{idx_to_name.get(idx, idx)}={val}" for idx, val in sk if isinstance(idx, int))
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
        count = forced_constraint_counts.get(scope_key, 1)
        if ctype == 'equal':
            model.Add(sum(vars_list) == count)
        elif ctype == 'lesse':
            model.Add(sum(vars_list) <= count)
        elif ctype == 'greatere':
            model.Add(sum(vars_list) >= count)
        elif ctype == 'greater':
            model.Add(sum(vars_list) > count)
        elif ctype == 'less':
            model.Add(sum(vars_list) < count)
        else:
            print(f"  WARNING: Unknown constraint type '{ctype}' for forced game, defaulting to equal")
            model.Add(sum(vars_list) == count)
    
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
            scope_desc = ', '.join(f"{idx_to_name.get(idx, idx)}={val}" for idx, val in scope_key if isinstance(idx, int))
            ctype = forced_constraint_types.get(scope_key, 'equal')
            count = forced_constraint_counts.get(scope_key, 1)
            op = {'equal': '==', 'lesse': '<=', 'greatere': '>=', 'greater': '>', 'less': '<'}.get(ctype, '==')
            print(f"    scope({scope_desc}): {len(vars_list)} decision vars -> sum {op} {count}")
    if blocked_vars_skipped:
        print(f"  - Blocked games: {blocked_vars_skipped} vars eliminated by {len(data.get('blocked_games', []))} no-play rules")
    if perennial_exempt_count:
        print(f"  - Perennial-BLOCKED exemptions: {perennial_exempt_count} vars kept "
              f"because a FORCED_GAMES scope overrides a perennial BLOCKED scope (spec-001)")
    return X, conflicts


# ============== Season Data Builder ==============

def _merge_constraint_defaults(season_overrides: dict) -> dict:
    """Merge season-specific overrides over the perennial defaults from config.defaults.

    Season configs may set only the keys they want to change; everything else
    inherits from `config.defaults.CONSTRAINT_DEFAULTS`. This guarantees every
    constraint can rely on the full set of keys without a per-season copy.
    """
    from config.defaults import CONSTRAINT_DEFAULTS as _DEFAULTS
    merged = dict(_DEFAULTS)
    if season_overrides:
        merged.update(season_overrides)
    return merged


def _merge_away_venue_rules(season_overrides: dict) -> dict:
    """Merge season-specific AWAY_VENUE_RULES over the perennial defaults.

    Season configs override only what they want changed; the rest inherits
    from `config.defaults.AWAY_VENUE_RULES`. The merge is shallow per club,
    so a season may override e.g. `Maitland.max_consecutive_home` without
    repeating the whole club's rule block.
    """
    from config.defaults import AWAY_VENUE_RULES as _DEFAULTS
    merged: dict = {}
    for club, rules in _DEFAULTS.items():
        merged[club] = dict(rules)
    for club, rules in (season_overrides or {}).items():
        merged.setdefault(club, {}).update(rules)
    return merged


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
    
    return {
        'year': year,
        'teams': TEAMS,
        'grades': GRADES,
        'fields': FIELDS,
        'clubs': CLUBS,
        'timeslots': TIMESLOTS,
        'num_rounds': num_rounds,
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
        'grade_order': grade_order,
        # Include any extra config values that constraints might need
        'home_field_map': home_field_map,
        'special_games': config.get('special_games', {}),
        # spec-006: preferred / avoided away-ground weekends (e.g. NRL clashes).
        'preferred_weekends': config.get('preferred_weekends', []),
        'forced_games': config.get('forced_games', []),
        'blocked_games': config.get('blocked_games', []),
        'penalty_weights': config.get('penalty_weights', {}),
        'constraint_defaults': _merge_constraint_defaults(config.get('constraint_defaults', {})),
        'away_venue_rules': _merge_away_venue_rules(config.get('away_venue_rules', {})),
    }

