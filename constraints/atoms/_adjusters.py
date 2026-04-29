"""FORCED/BLOCKED count adjusters (Phase 4) for constraints not yet atomized.

Each adjuster is a free function:

    adjuster(data, forced_games, blocked_games) -> dict | None

Engine calls every registered adjuster once during
`UnifiedConstraintEngine.build_groupings()` and stashes the result under
`data['count_adjustments'][canonical_name]`. The legacy `_matchup_spacing_*`,
`_maitland_grouping_*`, `_away_maitland_*` methods on `UnifiedConstraintEngine`
read their entry by canonical name during apply.

Adjusters for atoms that already exist (e.g. `ClubVsClubCoincidence`) live next
to their atom file. This module collects the ones whose atoms are still inside
`unified.py`'s legacy methods.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from constraints.registry import CONSTRAINT_REGISTRY


# ----------------------------------------------------------------------
# Helpers shared across adjusters
# ----------------------------------------------------------------------


def _resolve_teams_in_entry(
    entry: Dict, teams: List
) -> List[Tuple[str, str]]:
    """Return list of (team1, team2) pairs an entry's 'teams'/'team1'/'team2'
    fields resolve to (alphabetical). Returns [] when the entry has no team
    matchers (i.e. an "any game in scope" entry)."""
    from utils import _build_team_lookups, _resolve_team_name

    grade = entry.get('grade')
    grades = entry.get('grades')
    effective_grade = grades if grades else grade

    team_names_set, team_lookup = _build_team_lookups(teams)
    raw_teams = entry.get('teams') or []

    if len(raw_teams) == 2:
        rt1 = _resolve_team_name(raw_teams[0], effective_grade, team_names_set, team_lookup, teams)
        rt2 = _resolve_team_name(raw_teams[1], effective_grade, team_names_set, team_lookup, teams)
        return [tuple(sorted([a, b])) for a in rt1 for b in rt2 if a != b]
    if 'team1' in entry and 'team2' in entry:
        rt1 = _resolve_team_name(entry['team1'], effective_grade, team_names_set, team_lookup, teams)
        rt2 = _resolve_team_name(entry['team2'], effective_grade, team_names_set, team_lookup, teams)
        return [tuple(sorted([a, b])) for a in rt1 for b in rt2 if a != b]
    return []


def _team_club(team_name: str, teams: List) -> Optional[str]:
    for t in teams:
        if t.name == team_name:
            return t.club.name
    return None


def _entry_weeks(entry: Dict, data: Dict) -> Set[int]:
    """Resolve which week numbers an entry pins. Supports 'week', 'round_no',
    'date'. Returns empty set if the entry isn't pinned to any specific week."""
    weeks: Set[int] = set()
    if 'week' in entry:
        w = entry['week']
        if isinstance(w, (list, tuple)):
            weeks.update(int(x) for x in w)
        else:
            weeks.add(int(w))
    if 'round_no' in entry:
        # round_no maps roughly to week; for adjusters we just track it.
        r = entry['round_no']
        if isinstance(r, (list, tuple)):
            weeks.update(int(x) for x in r)
        else:
            weeks.add(int(r))
    if 'date' in entry and not weeks:
        # Only resolve date->week if no explicit week/round_no
        d = entry['date']
        date_str = d if isinstance(d, str) else d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else None
        if date_str:
            for ts in data.get('timeslots', []) or []:
                ts_date = getattr(ts, 'date', None)
                if ts_date is None:
                    continue
                ts_str = ts_date.strftime('%Y-%m-%d') if hasattr(ts_date, 'strftime') else str(ts_date)
                if ts_str == date_str:
                    weeks.add(getattr(ts, 'week_no', getattr(ts, 'week', 0)))
                    break
    return weeks


# ----------------------------------------------------------------------
# #2 — EqualMatchUpSpacing adjuster
# ----------------------------------------------------------------------


def equal_matchup_spacing_adjuster(
    data: Dict, forced_games: List, blocked_games: List
) -> Optional[Dict[Tuple[str, str, str], Set[int]]]:
    """For each FORCED entry that pins (t1, t2, grade) into specific weeks/rounds,
    accumulate the rounds. Atom uses this to clamp the spacing window down.

    Returns: { (team1_alpha_first, team2, grade): set of forced rounds }.
    """
    teams = data.get('teams', []) or []
    if not forced_games or not teams:
        return None

    forced_rounds_per_pair: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    for entry in forced_games:
        weeks = _entry_weeks(entry, data)
        if not weeks:
            continue
        grade = entry.get('grade')
        if not grade or isinstance(grade, (list, tuple)):
            continue
        pairs = _resolve_teams_in_entry(entry, teams)
        for t1, t2 in pairs:
            forced_rounds_per_pair[(t1, t2, grade)].update(weeks)

    if not forced_rounds_per_pair:
        return None
    return dict(forced_rounds_per_pair)


# ----------------------------------------------------------------------
# #3 — MaitlandHomeGrouping adjuster (will rename to NonDefaultHomeGrouping)
# ----------------------------------------------------------------------


def maitland_home_grouping_adjuster(
    data: Dict, forced_games: List, blocked_games: List
) -> Optional[Dict[str, Set[int]]]:
    """For each FORCED entry that pins a game at a non-default home venue with
    a team of that home club, accumulate the week as a definite home weekend.

    Returns: { club_name: set of forced home weeks }.
    """
    teams = data.get('teams', []) or []
    home_field_map = data.get('home_field_map', {}) or {}
    if not forced_games or not teams or not home_field_map:
        return None

    venue_to_club = {v: c for c, v in home_field_map.items()}

    forced_home: Dict[str, Set[int]] = defaultdict(set)
    for entry in forced_games:
        venue = entry.get('field_location')
        if venue not in venue_to_club:
            continue
        home_club = venue_to_club[venue]
        weeks = _entry_weeks(entry, data)
        if not weeks:
            continue
        pairs = _resolve_teams_in_entry(entry, teams)
        if not pairs:
            # 'all' entry — any team plays at this venue this week. We can't
            # know which club is home so skip.
            continue
        for t1, t2 in pairs:
            c1 = _team_club(t1, teams)
            c2 = _team_club(t2, teams)
            if home_club in (c1, c2):
                forced_home[home_club].update(weeks)

    if not forced_home:
        return None
    return {c: set(ws) for c, ws in forced_home.items()}


# ----------------------------------------------------------------------
# #4 — AwayAtMaitlandGrouping adjuster (will rename to AwayAtNonDefaultGrouping)
# ----------------------------------------------------------------------


def away_at_maitland_grouping_adjuster(
    data: Dict, forced_games: List, blocked_games: List
) -> Optional[Dict[Tuple[int, str], Set[str]]]:
    """For each FORCED entry that pins a game at a non-default home venue,
    record the away team's club. The atom uses len(set) per (week, venue) as a
    floor for the away-clubs-per-week count.

    Returns: { (week, venue): set of away club names }.
    """
    teams = data.get('teams', []) or []
    home_field_map = data.get('home_field_map', {}) or {}
    if not forced_games or not teams or not home_field_map:
        return None

    venue_to_club = {v: c for c, v in home_field_map.items()}

    forced_away: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
    for entry in forced_games:
        venue = entry.get('field_location')
        if venue not in venue_to_club:
            continue
        home_club = venue_to_club[venue]
        weeks = _entry_weeks(entry, data)
        if not weeks:
            continue
        pairs = _resolve_teams_in_entry(entry, teams)
        if not pairs:
            continue
        for t1, t2 in pairs:
            c1 = _team_club(t1, teams)
            c2 = _team_club(t2, teams)
            if c1 == home_club and c2 and c2 != home_club:
                away = c2
            elif c2 == home_club and c1 and c1 != home_club:
                away = c1
            else:
                continue
            for w in weeks:
                forced_away[(w, venue)].add(away)

    if not forced_away:
        return None
    return {k: set(v) for k, v in forced_away.items()}


# ----------------------------------------------------------------------
# Wire adjusters into the registry on import.
# ----------------------------------------------------------------------


CONSTRAINT_REGISTRY['EqualMatchUpSpacing'].forced_blocked_adjuster = (
    equal_matchup_spacing_adjuster
)
CONSTRAINT_REGISTRY['MaitlandHomeGrouping'].forced_blocked_adjuster = (
    maitland_home_grouping_adjuster
)
CONSTRAINT_REGISTRY['AwayAtMaitlandGrouping'].forced_blocked_adjuster = (
    away_at_maitland_grouping_adjuster
)
