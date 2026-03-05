# utils.py
"""
Utility functions for scheduling system.
"""
from typing import List, Dict, Any, Tuple
from models import Team, Club, Grade, Game, WeeklyDraw, Roster, Timeslot, PlayingField
import re
import os
from datetime import datetime, time as tm
import pandas as pd
from collections import defaultdict
from typing import Set

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


def max_games_per_grade(grades: List, max_rounds: int) -> Dict[str, int]:
    """
    Given a list of Grade objects (each with num_teams) and the maximum
    number of rounds, returns a dict mapping grade.name → max games per team.

    In each round you can have floor(T/2) matches, so over R rounds:
      total_matches ≤ R * floor(T/2)
    and since each team plays g games, total_matches = g * T / 2 must be integer.
    
    Args:
        grades: List of Grade objects with num_teams attribute
        max_rounds: Maximum number of rounds in the season
        
    Returns:
        Dict mapping grade name to max games per team
    """
    games_per_grade: Dict[str, int] = {}

    for grade in grades:
        T = grade.num_teams
        if T < 2:
            games_per_grade[grade.name] = 0
            continue

        # maximum matches across all rounds
        max_matches = max_rounds * (T // 2)

        # g0 = floor( 2 * max_matches / T )
        g0 = (2 * max_matches) // T
        # can't exceed one game per round
        g0 = min(g0, max_rounds)

        # ensure g0*T is even → if T odd, force g0 even
        if T % 2 == 1 and (g0 % 2) == 1:
            g0 -= 1

        games_per_grade[grade.name] = g0

    return games_per_grade


def generate_games(teams: List[Team]) -> Dict[Tuple[str, str, str], Tuple]:
    """
    Generate all possible games between teams of the same grade.
    
    Args:
        teams: List of Team objects
        
    Returns:
        Dict mapping (team1_name, team2_name, grade) -> (team1, team2, grade)
    """
    games = {
        (t1.name, t2.name, t1.grade): (t1, t2, t1.grade)
        for i, t1 in enumerate(teams)
        for t2 in teams[i + 1:] if t1.grade == t2.grade
    }
    return games


def generate_X(model, data: dict) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Generate decision variables for all possible games and timeslots.
    
    Args:
        model: CP-SAT model
        data: Data dictionary containing teams, timeslots, etc.
        
    Returns:
        Tuple of (X, Y, conflicts, unavailable_games)
        - X: Dict of decision variables for real timeslots
        - Y: Dict of decision variables for dummy timeslots (currently empty)
        - conflicts: Dict of team conflicts (currently empty, populated elsewhere)
        - unavailable_games: Dict of unavailable games (currently empty)
    """
    teams = data['teams']
    timeslots = data['timeslots']
    
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
    unavailable_games = {}
    
    # Handle both dict and list formats for games
    game_items = games.items() if isinstance(games, dict) else [(g, g) for g in games]
    
    for game_key, game_val in game_items:
        if isinstance(game_key, tuple) and len(game_key) >= 3:
            t1_name, t2_name, grade_name = game_key[0], game_key[1], game_key[2]
        else:
            continue
            
        for t in timeslots:
            if not t.day:
                continue  # skip dummy timeslots
            key = (t1_name, t2_name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1_name}_{t2_name}_{t.day}_{t.time}_{t.week}_{t.field.name}')
    
    print(f"Created {len(X)} decision variables")
    return X, Y, conflicts, unavailable_games


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
    
    # Merge day_time_map and phl_game_times for timeslot generation
    merged_dict = defaultdict(lambda: defaultdict(list))
    for d in (phl_game_times, day_time_map):
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
    
    # Calculate rounds per grade
    num_rounds = max_games_per_grade(GRADES, max_rounds)
    num_rounds['max'] = max_rounds
    
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
        'penalties': {},
        'day_time_map': day_time_map,
        'phl_game_times': phl_game_times,
        'phl_preferences': phl_preferences,
        'max_day_slot_per_field': max_day_slot_per_field,
        'field_unavailabilities': field_unavailabilities,
        'club_days': club_days,
        'preference_no_play': preference_no_play,
        'num_dummy_timeslots': num_dummy_timeslots,
        'grade_order': grade_order,
        # Include any extra config values that constraints might need
        'home_field_map': home_field_map,
        'friday_night_config': config.get('friday_night_config', {}),
        'special_games': config.get('special_games', {}),
    }

