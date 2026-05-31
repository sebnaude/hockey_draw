"""Shared helpers for away-club home-weekend math and per-pair PHL-Friday math.

This module now exposes only TWO concerns:

  1. Per-PAIR FORCED-aware PHL Friday meeting counts via
     ``phl_forced_friday_meetings(data, club_a, club_b)`` — consumed by
     ``ClubVsClubStackedWeekends`` (spec-005) for the per-pair PHL Sunday
     budget subtraction.

  2. The derived Sunday-home-weekend range for an away-based club via
     ``away_club_min_sundays_home(data, club)`` and
     ``away_club_max_sundays_home(data, club)`` — consumed by
     ``AwayClubHomeWeekendsCount`` (spec-004, redesigned in spec-037).

## The Sunday-home derived range (spec-037)

For each away-based club (home venue != Broadmeadow), the redesigned
``AwayClubHomeWeekendsCount`` atom enforces a single two-sided range on the
number of Sunday weeks the club has a home game at its venue:

    min_sundays_home(club) <= sum(sunday_home_indicators) <= max_sundays_home(club)

where:

  min_sundays_home(club) = max(num_rounds[g] // 2 for g in non-PHL grades the club fields)
                         = 0 if the club fields no non-PHL grade.

  max_sundays_home(club) = max(upper(g) for g in ALL grades the club fields)
                         = 0 if the club has no teams,
    where upper(PHL) = (num_rounds[PHL] + 1) // 2 (ceil — PHL can lend Sundays
    to forced Fridays, so its upper bound rounds up for odd num_rounds) and
    upper(g) = num_rounds[g] // 2 for non-PHL grades (floor — those grades
    have no Friday alternative, so their ceiling equals their exact home count).

The floor is the strict number of Sundays demanded by non-PHL grades (those
have no Friday alternative — every home game MUST land on a Sunday). The
ceiling is the maximum across ALL grades incl. PHL (no team plays more than
its 50/50 home share, and PHL can lend its Sundays to forced Fridays without
pushing the total above its raw home count).

Forced PHL Fridays then fit between the bounds automatically: each PHL home
game pulled to Friday by ``FORCED_GAMES`` reduces the PHL Sunday demand
without affecting the non-PHL floor. No FORCED-aware math inside the atom.

For odd ``num_rounds[g]`` (not seen in production, where pairs imply even
``num_rounds``), we use floor (`x // 2`) for the lower bound and ceil
(`(x + 1) // 2`) for the upper bound — a defensive rounding choice so a future
odd grade still yields a feasible range.

## Per-pair PHL Friday meetings (spec-005, unchanged)

``phl_forced_friday_meetings(data, club_a, club_b)`` returns the number of PHL
Friday games FORCED to be played between two SPECIFIC clubs, using a greedy
matched-set partition so umbrella + per-pair scopes never double-count.

Per-pair semantics: an umbrella entry like ``{club: Maitland, day: Friday,
count: 2}`` does NOT guarantee any Maitland-vs-Norths Friday (the 2 forced
games could all be vs Tigers). Only entries whose scope names BOTH clubs
contribute. This deliberately UNDER-counts vs over-counting, so spec-005's
PHL Sunday budget `total_pair_meetings - phl_forced_friday_meetings` stays
a LOWER bound on pair Sundays (more is fine; fewer would break alignment).

## Per-club umbrella forced Fridays (spec-044)

``club_umbrella_forced_friday_meetings(data, club)`` returns a SOUND LOWER
BOUND on the number of Friday games a club is forced to play at its home
away-venue via venue-wide ("umbrella") FORCED_GAMES entries that name no
specific pair. Unlike ``phl_forced_friday_meetings`` (which deliberately
ignores umbrella scopes for per-pair soundness), this helper credits the
umbrella scopes — but only at the CLUB level. It is subtracted from the
per-pair Sunday meeting LOWER bound only (never the ceiling). Contract:

* **Max-per-away-venue** — the count is the MAX ``_entry_count`` over the
  club's away-venue umbrella entries (NOT the sum), so subset scopes (e.g. the
  seven date-specific CCHP ``count==1`` entries) are dominated by the
  ``count==8`` CCHP umbrella rather than double-counted.
* **Home-club attribution** — each away-venue umbrella is attributed to its
  home club via ``home_field_map`` (oriented club -> away venue:
  Gosford -> Central Coast Hockey Park, Maitland -> Maitland Park).
* **equal-only** — only ``constraint in (None, 'equal')`` entries count toward
  the floor; ``lesse``/``greatere`` etc. do not guarantee a hard floor.
* **NIHC / Broadmeadow skipped** — central-venue umbrellas name no single home
  club, so they are not attributed to any club. A central club (whose home
  venue does not appear in ``home_field_map``) therefore returns 0. This keeps
  the result a sound lower bound.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from utils import (
    _build_forced_game_rules,
    _get_matching_forced_scopes,
)


# Candidate-key tuple shape (subset of the 11-tuple X-key).
_CandidateKey = Tuple


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


def _iter_candidate_friday_phl_keys_for_pair(
    data: Dict, club_a: str, club_b: str,
):
    """Yield candidate 11-tuple X-keys for PHL Friday games WHERE both teams
    belong to {club_a, club_b}.

    Used by `phl_forced_friday_meetings(data, club_a, club_b)` (spec-005) to
    compute the FORCED-Friday meeting count between a specific pair of clubs
    without double-counting umbrella + per-pair scopes.
    """
    a_phl_teams = _club_team_names(data, club_a, grade='PHL')
    b_phl_teams = _club_team_names(data, club_b, grade='PHL')
    if not a_phl_teams or not b_phl_teams:
        return

    for (t1, t2, grade) in data.get('games', []):
        if grade != 'PHL':
            continue
        cross = (
            (t1 in a_phl_teams and t2 in b_phl_teams)
            or (t1 in b_phl_teams and t2 in a_phl_teams)
        )
        if not cross:
            continue
        for ts in data.get('timeslots', []):
            if not ts.day or ts.day != 'Friday':
                continue
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location,
            )
            yield key


def _entry_targets_pair_phl_friday(
    entry: Dict, club_a: str, club_b: str, data: Dict
) -> bool:
    """True iff the FORCED entry specifically pins PHL Friday games between
    BOTH `club_a` and `club_b`.

    Per-pair semantics (spec-005): an umbrella scope like `{club: Maitland,
    day: Friday, count: 2}` does NOT guarantee any Maitland-vs-Norths Friday
    — those 2 could all be Maitland-vs-Tigers. So per-pair helpers credit
    only entries whose scope **names both** clubs (`teams=[A, B]` or
    `team1=A, team2=B`, accepting team names or club names for either side).

    Returns False for umbrella scopes — the per-pair helper deliberately
    UNDER-counts rather than over-counting "what's pinned between the pair."
    """
    a_phl_teams = _club_team_names(data, club_a, grade='PHL')
    b_phl_teams = _club_team_names(data, club_b, grade='PHL')
    if not a_phl_teams or not b_phl_teams:
        return False

    def _names_belong_to_a_b():
        raw_teams = entry.get('teams') or []
        team1 = entry.get('team1')
        team2 = entry.get('team2')
        # Build the two-sided team list.
        sides = []
        if len(raw_teams) >= 2:
            sides = [raw_teams[0], raw_teams[1]]
        elif team1 is not None and team2 is not None:
            sides = [team1, team2]
        if len(sides) != 2:
            return False
        # Each side must resolve to either club_a or club_b's PHL team(s).
        def _side_club(name: str) -> str | None:
            if name == club_a or name in a_phl_teams:
                return club_a
            if name == club_b or name in b_phl_teams:
                return club_b
            return None
        c1 = _side_club(sides[0])
        c2 = _side_club(sides[1])
        if c1 is None or c2 is None:
            return False
        return {c1, c2} == {club_a, club_b}

    return _names_belong_to_a_b()


def phl_forced_friday_meetings(data: Dict, club_a: str, club_b: str) -> int:
    """Number of PHL Friday games FORCED to be played between `club_a` and
    `club_b`, FORCED-aware (no double-counting across umbrella + per-pair
    scopes).

    spec-005 helper: used by `ClubVsClubStackedAlignment` to compute the PHL
    Sunday budget per club pair. `total_phl_meetings(A,B) -
    phl_forced_friday_meetings(A,B)` is the number of PHL Sunday weekends
    available for stacking.

    Returns 0 if either club has no PHL teams, if the pair has no FORCED
    Friday entries that resolve to them, or if no candidate Friday slot
    exists for the pair.
    """
    if club_a == club_b:
        return 0
    a_phl_teams = _club_team_names(data, club_a, grade='PHL')
    b_phl_teams = _club_team_names(data, club_b, grade='PHL')
    if not a_phl_teams or not b_phl_teams:
        return 0

    forced_games = data.get('forced_games') or []
    relevant_entries = [
        e for e in _friday_phl_forced_entries(forced_games)
        if _entry_targets_pair_phl_friday(e, club_a, club_b, data)
    ]
    if not relevant_entries:
        return 0

    candidate_keys = list(_iter_candidate_friday_phl_keys_for_pair(
        data, club_a, club_b,
    ))
    if not candidate_keys:
        return 0

    scoped: List[Tuple[Set[_CandidateKey], int]] = []
    for entry in relevant_entries:
        matched = _matched_var_keys_for_entry(entry, candidate_keys, data)
        if not matched:
            continue
        scoped.append((matched, _entry_count(entry)))

    if not scoped:
        return 0

    # Greedy partition: broadest scope wins; subset scopes contribute 0.
    scoped.sort(key=lambda sc: len(sc[0]), reverse=True)
    claimed: Set[_CandidateKey] = set()
    total = 0
    for matched, count in scoped:
        unclaimed = matched - claimed
        if not unclaimed:
            continue
        total += count
        claimed |= matched
    return total


def away_club_min_sundays_home(data: Dict, club: str) -> int:
    """Floor of the derived Sunday-home-weekend range for an away-based club.

    = max(num_rounds[g] // 2 for g in non-PHL grades the club fields), or
    0 if the club fields no non-PHL grade (PHL has a Friday alternative, so
    nothing forces a Sunday-home floor when PHL is the only grade).

    For odd `num_rounds[g]` (defensive — never seen in production), floor
    rounding `x // 2` is used.
    """
    grades = _grades_played_by_club(data, club)
    non_phl_grades = [g for g in grades if g != 'PHL']
    if not non_phl_grades:
        return 0
    return max(_grade_required_games(data, g) // 2 for g in non_phl_grades)


def away_club_max_sundays_home(data: Dict, club: str) -> int:
    """Ceiling of the derived Sunday-home-weekend range for an away-based club.

    = max(num_rounds[g] // 2 for g in ALL grades the club fields), with PHL
    using ceiling rounding `(x + 1) // 2` for odd `num_rounds['PHL']` to make
    sure the range stays feasible if a future config sets an odd PHL count.
    Returns 0 if the club has no teams.

    Non-PHL grades use floor (`x // 2`) — they have no Friday alternative, so
    the ceiling can't legitimately exceed their actual home-game count.
    """
    grades = _grades_played_by_club(data, club)
    if not grades:
        return 0

    def _upper(g: str) -> int:
        n = _grade_required_games(data, g)
        if g == 'PHL':
            # Ceil for PHL — PHL can lend Sundays to forced Fridays, and on an
            # odd `num_rounds['PHL']` we want the bound to be reachable.
            return (n + 1) // 2
        return n // 2

    return max(_upper(g) for g in grades)


def _entry_is_pair(entry: Dict) -> bool:
    """True if ``entry`` names a specific pair (two sides), else umbrella.

    An entry is a "pair" entry iff it names two specific teams/clubs: either a
    ``teams`` list/tuple of length >= 2, or BOTH ``team1`` and ``team2``.
    Anything else (no team naming, or only one side named) is an umbrella scope.
    """
    if not isinstance(entry, dict):
        return False
    teams = entry.get('teams')
    if isinstance(teams, (list, tuple)) and len(teams) >= 2:
        return True
    if entry.get('team1') and entry.get('team2'):
        return True
    return False


def _entry_field_locations(entry: Dict) -> Set[str]:
    """Return the set of field_location spellings on an entry (scalar or list)."""
    if not isinstance(entry, dict):
        return set()
    loc = entry.get('field_location')
    if loc is None:
        return set()
    if isinstance(loc, (list, tuple, set)):
        return {str(x) for x in loc if x is not None}
    return {str(loc)}


def club_umbrella_forced_friday_meetings(data: Dict, club: str) -> int:
    """Sound lower bound on a club's umbrella-forced PHL Friday games.

    Counts the MAX forced ``count`` over PHL-Friday ``FORCED_GAMES`` umbrella
    (non-pair) entries with ``constraint in (None, 'equal')`` that map to the
    club's home away-venue (via ``home_field_map``, oriented club -> away
    venue). Central-venue (NIHC / Broadmeadow) umbrellas name no single home
    club and are skipped, so a central club returns 0. Max-per-away-venue
    de-duplicates subset scopes (e.g. seven date-specific CCHP ``count==1``
    entries are dominated by the ``count==8`` CCHP umbrella). Returns 0 if the
    club has no away-venue umbrella entries.

    See module docstring "Per-club umbrella forced Fridays (spec-044)" for the
    full contract. This term is subtracted from the per-pair Sunday meeting
    LOWER bound only (spec-044) — never the ceiling.
    """
    if not club:
        return 0
    home_field_map = (data or {}).get('home_field_map') or {}
    # home_field_map is oriented club -> away venue. Clubs whose home venue is
    # the central/NIHC venue do not appear here -> no away-venue umbrella to
    # attribute -> 0 (keeps the count a sound lower bound).
    away_venue = home_field_map.get(club)
    if not away_venue:
        return 0
    away_venue = str(away_venue)
    forced_games = data.get('forced_games') or []
    best = 0
    for entry in _friday_phl_forced_entries(forced_games):
        if _entry_is_pair(entry):
            continue
        constraint = entry.get('constraint')
        if constraint not in (None, 'equal'):
            continue
        if away_venue not in _entry_field_locations(entry):
            continue
        best = max(best, _entry_count(entry))
    return best


__all__ = [
    'phl_forced_friday_meetings',
    'club_umbrella_forced_friday_meetings',
    'away_club_min_sundays_home',
    'away_club_max_sundays_home',
]
