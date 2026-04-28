"""Shared helpers for ClubDay atoms — keep parsing/lookup logic in one place."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from utils import normalize_club_day, get_nearest_week_by_date


def parse_club_day_entries(data: Dict) -> List[Tuple[str, str, Optional[str]]]:
    """Yield `(club_name, date_str, opponent_or_None)` for each non-locked club day.

    Mirrors `original.py:ClubDayConstraint`: skips entries whose nearest-week
    falls inside `data['locked_weeks']`, and validates club / opponent names.
    """
    club_days = data.get('club_days') or {}
    if not club_days:
        return []

    clubs = data['clubs']
    timeslots = data.get('timeslots', [])
    locked_weeks = data.get('locked_weeks', set())
    club_names_lower = {c.name.lower(): c.name for c in clubs}

    out: List[Tuple[str, str, Optional[str]]] = []
    for club_name, value in club_days.items():
        if club_name.lower() not in club_names_lower:
            raise ValueError(f'Invalid club name {club_name!r} in CLUB_DAYS')
        date, opponent = normalize_club_day(value)
        if opponent is not None and opponent.lower() not in club_names_lower:
            raise ValueError(
                f'Invalid opponent club {opponent!r} in CLUB_DAYS for {club_name}'
            )
        date_str = date.date().strftime('%Y-%m-%d')
        closest_week = get_nearest_week_by_date(date_str, timeslots)
        if closest_week in locked_weeks:
            continue
        out.append((club_name, date_str, opponent))
    return out


def club_day_game_keys(
    X: Dict, club_team_names: Iterable[str], date_str: str,
) -> List[tuple]:
    """Return all real-game X keys involving a team from the club on the given date."""
    teams = set(club_team_names)
    keys = []
    for key in X:
        if len(key) < 11 or not key[3]:
            continue
        if key[7] != date_str:
            continue
        if key[0] in teams or key[1] in teams:
            keys.append(key)
    return keys


def teams_by_grade(team_names: Iterable[str]) -> Dict[str, List[str]]:
    """Group team names by grade (parsed from the trailing word of the team name)."""
    out: Dict[str, List[str]] = {}
    for name in team_names:
        grade = name.rsplit(' ', 1)[1]
        out.setdefault(grade, []).append(name)
    return out
