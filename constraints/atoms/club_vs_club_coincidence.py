"""3rd-6th club-pair meetings should coincide on the same round across grades.

For each pair of grades (X, Y) where Y has strictly more per-team games than X
(see `per_team_games` walk in `_club_vs_club_shared.py`), and each club_pair
seen in both grades, this atom:

1. Builds per-round indicators (`g1`, `g2`) using the helper-var pool.
2. Channels a `coincide` BoolVar = `g1 AND g2` per round.
3. Registers the `coincide` var in the helper-var pool keyed by
   `('cvc_coincide', grade, other_grade, club_pair, round_no)` so that
   `ClubVsClubFieldLimit` and `ClubVsClubDeficitPenalty` can read it back.
4. Adds the HARD requirement `sum(coincide_vars) >= num_games - slack`.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_shared import (
    COINCIDE_KEY_PREFIX,
    collect_grade_pair_round_vars,
    lower_grade_pairs_to_compare,
    per_team_games,
)
from constraints.registry import CONSTRAINT_REGISTRY


class ClubVsClubCoincidence(Atom):
    canonical_name = 'ClubVsClubCoincidence'
    atom_group = 'ClubVsClubAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        slack = data.get('constraint_slack', {}).get('ClubVsClubAlignment', 0)
        games = per_team_games(data)

        # Phase 4 adjuster output: per-grade per-pair expected meetings,
        # reduced for FORCED off-Sunday and BLOCKED on-Sunday entries.
        adjustment = (
            data.get('count_adjustments', {}).get('ClubVsClubCoincidence') or {}
        )

        per_grade_vars = {}
        for grade, _other, _num in lower_grade_pairs_to_compare(data):
            if grade not in per_grade_vars:
                per_grade_vars[grade] = collect_grade_pair_round_vars(
                    X, data, grade,
                )

        for grade, other_grade, num_games in lower_grade_pairs_to_compare(data):
            if other_grade not in per_grade_vars:
                per_grade_vars[other_grade] = collect_grade_pair_round_vars(
                    X, data, other_grade,
                )
            grade_pairs = per_grade_vars[grade]
            other_pairs = per_grade_vars[other_grade]
            grade_adj = adjustment.get(grade, {}) if isinstance(adjustment, dict) else {}

            for club_pair, rounds in grade_pairs.items():
                if club_pair not in other_pairs:
                    continue
                other_rounds = other_pairs[club_pair]
                coincide_vars = []
                for round_no, vars_list in rounds.items():
                    if round_no not in other_rounds:
                        continue
                    ind1 = registry.get_or_create_bool(
                        ('align_g1', grade, club_pair, round_no), vars_list,
                        f'g1_{grade}_{club_pair}_{round_no}',
                    )
                    ind2 = registry.get_or_create_bool(
                        ('align_g2', other_grade, club_pair, round_no),
                        other_rounds[round_no],
                        f'g2_{other_grade}_{club_pair}_{round_no}',
                    )
                    coincide = model.NewBoolVar(
                        f'coin_{grade}_{other_grade}_{club_pair}_{round_no}'
                    )
                    model.Add(coincide <= ind1)
                    model.Add(coincide <= ind2)
                    model.Add(coincide >= ind1 + ind2 - 1)
                    registry.register(
                        (COINCIDE_KEY_PREFIX, grade, other_grade, club_pair, round_no),
                        coincide,
                    )
                    coincide_vars.append(coincide)

                if coincide_vars:
                    expected = grade_adj.get(club_pair, num_games)
                    min_required = max(0, expected - slack)
                    model.Add(sum(coincide_vars) >= min_required)
                    n += 1
        return n


# ----------------------------------------------------------------------
# Phase 4 #5 — FORCED/BLOCKED count adjuster for ClubVsClubCoincidence.
# ----------------------------------------------------------------------


def club_vs_club_coincidence_adjuster(
    data: Dict, forced_games: List, blocked_games: List
) -> Optional[Dict[str, Dict[Tuple[str, str], int]]]:
    """User's worked example: ClubVsClubCoincidence counts how many rounds a
    club-pair meets on Sunday. If FORCED forces N games of a club-pair onto a
    non-Sunday day, those games never appear in the Sunday alignment block, so
    the expected coincidence count drops by N. Same for BLOCKED entries that
    eliminate Sunday vars.

    Returns: { grade: { (club_a, club_b): expected_meetings } } where expected_
    meetings = total - forced_off_sunday - blocked_on_sunday, clamped to 0.
    Only pairs with non-zero adjustment are included.
    """
    teams = data.get('teams', []) or []
    if not teams:
        return None

    from constraints.atoms._adjusters import _resolve_teams_in_entry

    team_club: Dict[str, str] = {t.name: t.club.name for t in teams}

    # Initialise expected counts from per_team_games (matches atom semantics)
    per_grade = per_team_games(data)
    grades = data.get('grades', []) or []
    grade_names = {g.name for g in grades}

    # We only adjust pairs we actually see in forced/blocked entries. The atom
    # falls back to per_team_games when the adjustment dict has no entry for
    # a pair, so we don't need to enumerate every pair.
    adjustments: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(dict)

    def _classify(entry: Dict) -> str:
        """Classify an entry as 'sunday' / 'non_sunday' / 'unknown'."""
        day = entry.get('day')
        if isinstance(day, (list, tuple)):
            days = set(day)
            if days == {'Sunday'}:
                return 'sunday'
            if 'Sunday' not in days:
                return 'non_sunday'
            return 'unknown'
        if day == 'Sunday':
            return 'sunday'
        if day:  # any non-Sunday day
            return 'non_sunday'
        return 'unknown'

    # Tally forced games that pin a club-pair off Sunday
    deltas: Dict[Tuple[str, Tuple[str, str]], int] = defaultdict(int)
    for entry in forced_games or []:
        if _classify(entry) != 'non_sunday':
            continue
        grade = entry.get('grade')
        if not grade or grade not in grade_names:
            continue
        pairs = _resolve_teams_in_entry(entry, teams)
        if not pairs:
            continue
        # Use entry 'count' as the number of forced occurrences (default 1).
        count = int(entry.get('count', 1))
        seen_clubpairs = set()
        for t1, t2 in pairs:
            c1, c2 = team_club.get(t1), team_club.get(t2)
            if not c1 or not c2 or c1 == c2:
                continue
            club_pair = tuple(sorted([c1, c2]))
            if club_pair in seen_clubpairs:
                continue
            seen_clubpairs.add(club_pair)
            deltas[(grade, club_pair)] += count

    for entry in blocked_games or []:
        # BLOCKED entries removing Sunday vars also reduce expected coincidences.
        day = entry.get('day')
        if day != 'Sunday' and not (isinstance(day, (list, tuple)) and 'Sunday' in day):
            continue
        grade = entry.get('grade')
        if not grade or grade not in grade_names:
            continue
        pairs = _resolve_teams_in_entry(entry, teams)
        if not pairs:
            continue
        count = int(entry.get('count', 1))
        seen_clubpairs = set()
        for t1, t2 in pairs:
            c1, c2 = team_club.get(t1), team_club.get(t2)
            if not c1 or not c2 or c1 == c2:
                continue
            club_pair = tuple(sorted([c1, c2]))
            if club_pair in seen_clubpairs:
                continue
            seen_clubpairs.add(club_pair)
            deltas[(grade, club_pair)] += count

    if not deltas:
        return None

    for (grade, club_pair), delta in deltas.items():
        baseline = per_grade.get(grade, 0)
        adjustments[grade][club_pair] = max(0, baseline - delta)

    return {g: dict(p) for g, p in adjustments.items()}


CONSTRAINT_REGISTRY['ClubVsClubCoincidence'].forced_blocked_adjuster = (
    club_vs_club_coincidence_adjuster
)
