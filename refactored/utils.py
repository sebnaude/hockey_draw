# utils.py
"""
Utility functions for scheduling system.
"""
from typing import List, Dict, Any, Tuple
from models import Team, Club, Grade, Game, WeeklyDraw, Roster, Timeslot, PlayingField
import re
from datetime import datetime
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
                        pd.DataFrame([[None, bye_team, "BYE", None, None, None, None, None, None, None, None]], columns=df.columns)
                    ], ignore_index=True)
            df.to_excel(writer, sheet_name=f"Week {week}", index=False)
    print(f"Schedule successfully exported to {filename}")
