"""Shared helpers for the ClubVsClubAlignment atoms.

The atoms split the legacy `ClubVsClubAlignment` constraint into 3 +1 atoms:
- `ClubVsClubCoincidence` (3rd-6th grade pairs): builds the per-round
  coincidence indicator and the HARD `sum >= num_games - slack` requirement.
- `ClubVsClubFieldLimit` (3rd-6th): HARD ≤ 2 fields when coinciding +
  SOFT field-excess penalty.
- `ClubVsClubDeficitPenalty` (3rd-6th): SOFT penalty per missing coincidence.
- `PHLAnd2ndBackToBackSameField` (PHL/2nd Sunday): coincidence + back-to-back
  same-field requirement + deficit penalty.

The first three atoms share a `coincide` BoolVar per `(grade, other_grade,
club_pair, round_no)`. The Coincidence atom creates and registers it via the
helper-var pool; the other two atoms read it back. The pool key is
`('cvc_coincide', grade, other_grade, club_pair, round_no)`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


COINCIDE_KEY_PREFIX = 'cvc_coincide'
PHL_BTB_COINCIDE_PREFIX = 'cvc_phl_btb_coincide'


def per_team_games(data: Dict) -> Dict[str, int]:
    """Calculate per-team game count per grade.

    Mirrors `original.py:ClubVsClubAlignment` line 999:
        even teams: max_rounds // (num_teams - 1)
        odd teams:  max_rounds // num_teams
    """
    num_rounds = data.get('num_rounds', {})
    R = num_rounds.get('max', 0)
    out: Dict[str, int] = {}
    for grade in data['grades']:
        T = grade.num_teams
        if T <= 1:
            out[grade.name] = 0
        elif T % 2 == 0:
            out[grade.name] = R // (T - 1)
        else:
            out[grade.name] = R // T
    return out


def lower_grade_pairs_to_compare(
    data: Dict,
) -> List[Tuple[str, str, int]]:
    """Yield (grade, other_grade, num_games_for_grade) where grade has strictly
    fewer per-team games than every previously-yielded grade — i.e. mirror the
    `ini_num`/`used_grades` walk in `original.py:ClubVsClubAlignment`. Excludes
    PHL/2nd (those go through the back-to-back atom)."""
    games = per_team_games(data)
    ordered = sorted(games.items(), key=lambda x: x[1])

    pairs: List[Tuple[str, str, int]] = []
    used: List[str] = []
    ini_num = 0
    for grade, num in ordered:
        if grade in ('PHL', '2nd'):
            continue
        used.append(grade)
        if num <= ini_num:
            continue
        ini_num = num
        for other_grade, _ in ordered:
            if other_grade in used:
                continue
            if other_grade in ('PHL', '2nd'):
                continue
            pairs.append((grade, other_grade, num))
    return pairs


def phl_2nd_grade_pairs_to_compare(data: Dict) -> List[Tuple[str, str, int]]:
    """Same `ini_num` walk but restricted to PHL/2nd."""
    games = per_team_games(data)
    ordered = sorted(games.items(), key=lambda x: x[1])

    pairs: List[Tuple[str, str, int]] = []
    used: List[str] = []
    ini_num = 0
    for grade, num in ordered:
        if grade not in ('PHL', '2nd'):
            continue
        used.append(grade)
        if num <= ini_num:
            continue
        ini_num = num
        for other_grade, _ in ordered:
            if other_grade in used:
                continue
            if other_grade not in ('PHL', '2nd'):
                continue
            pairs.append((grade, other_grade, num))
    return pairs


def collect_grade_pair_round_vars(
    X: Dict, data: Dict, grade: str, *, sunday_only: bool = False,
) -> Dict[Tuple[str, str], Dict[int, List]]:
    """Build `(club_pair) -> {round_no: [vars]}` for a grade.

    Skips locked weeks and dummy keys. `sunday_only=True` filters out non-Sunday
    games (used for PHL/2nd back-to-back which is Sunday-only)."""
    locked_weeks = set(data.get('locked_weeks', set()))
    team_club: Dict[str, str] = {t.name: t.club.name for t in data['teams']}

    out: Dict[Tuple[str, str], Dict[int, List]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, var in X.items():
        if len(key) < 11 or not key[3]:
            continue
        if key[2] != grade:
            continue
        if key[6] in locked_weeks:
            continue
        if sunday_only and key[3] != 'Sunday':
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2:
            continue
        club_pair = tuple(sorted((c1, c2)))
        out[club_pair][key[8]].append(var)
    return out


def collect_sunday_clubpair_field_round(
    X: Dict, data: Dict, grade: str,
) -> Dict[Tuple[str, str], Dict[int, Dict[str, List]]]:
    """`(club_pair) -> {round_no: {field_name: [vars]}}` for Sunday games of a grade."""
    locked_weeks = set(data.get('locked_weeks', set()))
    team_club: Dict[str, str] = {t.name: t.club.name for t in data['teams']}

    out: Dict[Tuple[str, str], Dict[int, Dict[str, List]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for key, var in X.items():
        if len(key) < 11 or not key[3]:
            continue
        if key[2] != grade:
            continue
        if key[3] != 'Sunday':
            continue
        if key[6] in locked_weeks:
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2:
            continue
        club_pair = tuple(sorted((c1, c2)))
        out[club_pair][key[8]][key[9]].append(var)
    return out


def collect_phl_2nd_sunday_field_slot(
    X: Dict, data: Dict, grade: str,
) -> Dict[Tuple[str, str], Dict[int, List[Tuple]]]:
    """For PHL/2nd Sunday games: `(club_pair) -> {round_no: [(var, field_name, day_slot)]}`."""
    locked_weeks = set(data.get('locked_weeks', set()))
    team_club: Dict[str, str] = {t.name: t.club.name for t in data['teams']}

    out: Dict[Tuple[str, str], Dict[int, List[Tuple]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, var in X.items():
        if len(key) < 11 or not key[3]:
            continue
        if key[2] != grade:
            continue
        if key[3] != 'Sunday':
            continue
        if key[6] in locked_weeks:
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2:
            continue
        club_pair = tuple(sorted((c1, c2)))
        out[club_pair][key[8]].append((var, key[9], key[4]))
    return out
