"""FORCED/BLOCKED count adjusters (Phase 4) for constraints not yet atomized.

Each adjuster is a free function:

    adjuster(data, forced_games, blocked_games) -> dict | None

Engine calls every registered adjuster once during
`UnifiedConstraintEngine.build_groupings()` and stashes the result under
`data['count_adjustments'][canonical_name]`. The legacy `_matchup_spacing_*`
methods on `UnifiedConstraintEngine` read their entry by canonical name during
apply. (spec-018 removed the `_maitland_grouping_*` / `_away_maitland_*`
adjusters along with the venue-sequencing rules they fed.)

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


# spec-018: `maitland_home_grouping_adjuster` (NonDefaultHomeGrouping) and
# `away_at_maitland_grouping_adjuster` (AwayAtNonDefaultGrouping) deleted —
# the venue-sequencing rules they fed were removed.


# ----------------------------------------------------------------------
# Wire adjusters into the registry on import.
# ----------------------------------------------------------------------


CONSTRAINT_REGISTRY['EqualMatchUpSpacing'].forced_blocked_adjuster = (
    equal_matchup_spacing_adjuster
)
