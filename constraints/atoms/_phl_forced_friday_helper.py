"""Shared helper for FORCED-aware PHL-Friday / Sunday math (spec-004 + spec-005).

This module computes the per-away-club totals required by:
  * `AwayClubHomeWeekendsCount` (spec-004) — needs total Sundays, total weekends,
    and the FORCED-Friday count.
  * `ClubVsClubStackedPHLSundayBudget` (spec-005) — needs forced-Friday meeting
    counts per club-pair, layered on top of `phl_forced_friday_count`.

## The convenor's distinction (see spec-004 "Clarification (added 2026-05-18)")

For each away-based club (a club whose home venue is NOT Broadmeadow):

  total_weekends_required(club)
      = max(phl_games_required, max_other_grade_games)
        — the ORIGINAL maximum across grades, ignoring FORCED Fridays.
          Represents total home-ground appearances (Friday + Sunday combined).

  away_club_required_sundays(club)
      = max(phl_games_required - phl_forced_friday_count(club),
            max_other_grade_games)
        — the CALCULATED number of Sundays required AFTER subtracting
          FORCED PHL Fridays. Strictly <= total_weekends_required.

  phl_forced_friday_count(club)
      = number of PHL Friday games this club WILL play, computed from FORCED_GAMES
        scopes WITHOUT double-counting (one variable matching multiple FORCED
        scopes counts as ONE Friday game).

If another grade requires *more* games than PHL has (e.g. PHL plays 18,
3rd plays 20), then `total_sundays_required = 20` (driven by 3rd grade) and
`total_weekends_required = 20` — the FORCED PHL Fridays are absorbed into the
same 20 weekend slots. The two-helper API lets the atom pick the right
invariant.

## Why count VARIABLES not sum of FORCED entry counts

A FORCED entry "Maitland Fridays count==2 equal" + a FORCED entry
"Maitland-vs-Tigers Friday count==1" describe TWO scopes but ONE pair of Friday
games (the per-pair entry is one of the umbrella's two). Summing the counts
gives 3; counting distinct X-key candidates matched, partitioned by
subset-disjointness, gives the correct 2.

Algorithm:
  1. For each FORCED entry that pins/caps PHL Friday games involving the club,
     build its scope rules + compute its "matched candidate var set" — the
     concrete (game, timeslot) X-key candidates that fall in scope.
  2. Greedy partition: sort scopes by matched-set size descending. For each
     scope, if its matched-set is NOT entirely covered by already-claimed scopes,
     contribute its `count` and claim its vars. Otherwise contribute 0.

This correctly handles:
  - nested scopes (umbrella + per-pair) → umbrella's count, per-pair contributes 0.
  - disjoint scopes (Maitland Park venue + NIHC venue) → sum of counts.
  - lesse/equal alike — we treat the scope's `count` as the maximum-games
    target the convenor expressed, which is what the solver enforces.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from utils import (
    _build_forced_game_rules,
    _get_matching_forced_scopes,
)


# Candidate-key tuple shape (subset of the 11-tuple X-key).
# We use the full 11-tuple to match how `_get_matching_forced_scopes` indexes
# the scope frozenset against `key[idx]`.
_CandidateKey = Tuple


def _away_club_home_venue(data: Dict, club: str) -> Optional[str]:
    """Return the club's away/home venue from `home_field_map`, or None."""
    home_field_map: Dict[str, str] = data.get('home_field_map', {}) or {}
    return home_field_map.get(club)


def _grade_required_games(data: Dict, grade: str) -> int:
    """Return the per-team games required for `grade` from `data['num_rounds']`.

    Tests and production both populate `data['num_rounds']` as a `{grade: N}`
    dict (with a `'max'` aggregate). Falls back to the Grade.num_games attribute
    when the key is missing.
    """
    num_rounds = data.get('num_rounds') or {}
    if grade in num_rounds:
        return int(num_rounds[grade])
    # Fallback: scan the Grade objects.
    for g in data.get('grades', []):
        if g.name == grade:
            return int(g.num_games)
    return 0


def _grades_played_by_club(data: Dict, club: str) -> List[str]:
    """List of grade names the club has at least one team in."""
    grades_seen: Set[str] = set()
    for team in data.get('teams', []):
        if team.club.name == club:
            grades_seen.add(team.grade)
    return sorted(grades_seen)


def _club_team_names(data: Dict, club: str, grade: Optional[str] = None) -> Set[str]:
    """All team names belonging to `club`, optionally filtered by `grade`."""
    return {
        team.name
        for team in data.get('teams', [])
        if team.club.name == club and (grade is None or team.grade == grade)
    }


def _iter_candidate_friday_phl_keys(data: Dict, club: str):
    """Yield candidate 11-tuple X-keys for PHL Friday games involving `club`.

    Mirrors the key shape produced by `generate_X`, so they slot directly into
    `_get_matching_forced_scopes(key, scope_rules)`. The yielded keys are the
    full universe of (game, timeslot) combinations the solver could pick from —
    not the post-elimination X dict. That's deliberate: the helper must reason
    about what FORCED would pick, regardless of whether `generate_X` has already
    eliminated other Friday vars.
    """
    club_phl_teams = _club_team_names(data, club, grade='PHL')
    if not club_phl_teams:
        return

    for (t1, t2, grade) in data.get('games', []):
        if grade != 'PHL':
            continue
        if t1 not in club_phl_teams and t2 not in club_phl_teams:
            continue
        for ts in data.get('timeslots', []):
            if not ts.day or ts.day != 'Friday':
                continue
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location,
            )
            yield key


def _friday_phl_forced_entries(forced_games: List[Dict]) -> List[Dict]:
    """Return only the FORCED entries that target PHL Friday games."""
    out = []
    for entry in forced_games or []:
        if entry.get('day') != 'Friday':
            continue
        # `grade` may be missing if the entry uses `grades` list — accept
        # PHL via either field.
        grade = entry.get('grade')
        grades = entry.get('grades') or []
        if grade != 'PHL' and 'PHL' not in grades:
            continue
        out.append(entry)
    return out


def _entry_targets_club_phl_friday(entry: Dict, club: str, data: Dict) -> bool:
    """Cheap pre-filter: does the FORCED entry plausibly involve `club`'s PHL teams?

    Used to skip per-pair entries that explicitly name different clubs (e.g.
    `Norths vs Wests` doesn't involve Maitland). Returns True conservatively
    when the entry has no team filter (umbrella scope) or names the club.
    """
    club_phl_teams = _club_team_names(data, club, grade='PHL')
    if not club_phl_teams:
        return False

    raw_teams = entry.get('teams') or []
    team1 = entry.get('team1')
    team2 = entry.get('team2')
    club_filter = entry.get('club')

    if not (raw_teams or team1 or team2 or club_filter):
        # Umbrella scope — applies to ALL Friday PHL games. Filter by venue
        # at the candidate-iteration stage.
        return True

    # Resolve any string name (could be a team name OR a club name) and check
    # whether it refers to this club.
    def _names_to_resolve():
        for x in raw_teams:
            yield x
        for x in (team1, team2, club_filter):
            if x is not None:
                yield x

    for name in _names_to_resolve():
        # Direct club name match.
        if name == club:
            return True
        # Or a team belonging to the club (full team name, e.g. 'Maitland PHL').
        if name in club_phl_teams:
            return True
    return False


def _matched_var_keys_for_entry(
    entry: Dict, candidate_keys: List[_CandidateKey], data: Dict
) -> Set[_CandidateKey]:
    """Compute the set of candidate X-keys that this FORCED entry's scope matches.

    Wraps the standard `_build_forced_game_rules` on a single-entry list so we
    re-use the production scope-matching exactly. Returns a `set` of keys.
    """
    teams_obj_list = data.get('teams', [])
    scope_rules, _ctypes, _ccounts = _build_forced_game_rules([entry], teams_obj_list)
    matched: Set[_CandidateKey] = set()
    if not scope_rules:
        return matched
    for key in candidate_keys:
        if _get_matching_forced_scopes(key, scope_rules):
            matched.add(key)
    return matched


def _entry_count(entry: Dict) -> int:
    """Return the FORCED entry's target count (default 1 for `equal`)."""
    return int(entry.get('count', 1))


def phl_forced_friday_count(data: Dict, club: str) -> int:
    """Count of PHL Friday games this club WILL play, FORCED-aware.

    Handles multi-scope FORCED entries (umbrella + per-pair on the same Friday
    bucket) WITHOUT double-counting. See module docstring for the algorithm.

    Returns 0 if the club has no PHL teams or no FORCED Friday entries.
    """
    club_phl_teams = _club_team_names(data, club, grade='PHL')
    if not club_phl_teams:
        return 0

    forced_games = data.get('forced_games') or []
    relevant_entries = [
        e for e in _friday_phl_forced_entries(forced_games)
        if _entry_targets_club_phl_friday(e, club, data)
    ]
    if not relevant_entries:
        return 0

    candidate_keys = list(_iter_candidate_friday_phl_keys(data, club))
    if not candidate_keys:
        return 0

    # Build (matched-set, count) per entry.
    scoped: List[Tuple[Set[_CandidateKey], int]] = []
    for entry in relevant_entries:
        matched = _matched_var_keys_for_entry(entry, candidate_keys, data)
        if not matched:
            continue
        scoped.append((matched, _entry_count(entry)))

    if not scoped:
        return 0

    # Greedy partition: walk scopes largest-matched-set first; claim vars; any
    # subsequent scope whose matched-set is fully covered by claimed vars
    # contributes 0 (it's a subset of an already-counted scope).
    scoped.sort(key=lambda sc: len(sc[0]), reverse=True)
    claimed: Set[_CandidateKey] = set()
    total = 0
    for matched, count in scoped:
        unclaimed = matched - claimed
        if not unclaimed:
            # Fully covered by an earlier (broader) scope — its count already
            # accounts for this entry's contribution.
            continue
        total += count
        claimed |= matched
    return total


def away_club_total_weekends(data: Dict, club: str) -> int:
    """The ORIGINAL (unadjusted) max games across grades, ignoring FORCED Fridays.

    = max(phl_games_required, max_other_grade_games)

    This is the total number of home-ground appearances the club should make
    across the season (Friday + Sunday combined). Returns 0 if the club has no
    teams.
    """
    grades = _grades_played_by_club(data, club)
    if not grades:
        return 0
    return max(_grade_required_games(data, g) for g in grades)


def away_club_required_sundays(data: Dict, club: str) -> int:
    """The CALCULATED number of Sundays the club needs at its home ground.

    = max(phl_games_required - forced_phl_fridays, max_other_grade_games)

    Strictly <= away_club_total_weekends. Returns 0 if the club has no teams.
    """
    grades = _grades_played_by_club(data, club)
    if not grades:
        return 0

    phl_required = _grade_required_games(data, 'PHL') if 'PHL' in grades else 0
    other_grade_max = max(
        (_grade_required_games(data, g) for g in grades if g != 'PHL'),
        default=0,
    )
    forced_fridays = phl_forced_friday_count(data, club)
    phl_sundays_required = max(0, phl_required - forced_fridays)
    return max(phl_sundays_required, other_grade_max)


__all__ = [
    'phl_forced_friday_count',
    'away_club_required_sundays',
    'away_club_total_weekends',
]
