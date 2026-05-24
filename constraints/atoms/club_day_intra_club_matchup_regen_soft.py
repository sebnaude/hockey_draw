"""SOFT analogue of ``ClubDayIntraClubMatchup`` (spec-027 regen-soft).

The HARD atom (``constraints/atoms/club_day_intra_club_matchup.py``) enforces
that, on a club's club-day, same-grade duplicate teams within the club must
matchup (derbies) — at least ``len(host_grade_teams) // 2`` intra-club games of
that grade are forced ONTO the club-day, but only when there is no opponent
covering that grade.

This SOFT analogue emits NO hard clause. Instead, for every grade the hard atom
would govern (host has >1 team in the grade, and either no opponent or the
opponent has no teams in the grade), it adds ONE penalty unit per intra-club
matchup of that grade scheduled OUTSIDE the club-day (i.e. on any date other
than ``date_str``). The model stays feasible for ANY X: the penalty simply
measures, and the objective discourages, derbies that happen off the club-day.

One penalty unit == one intra-club matchup game scheduled on a non-club-day
date. Because each intra-club matchup var is itself a 0/1, the penalty for a var
is just the var (it is 1 exactly when that off-day derby is scheduled), so the
bucket gets the raw decision vars appended — no extra linearization needed.

Skip/no-op conditions mirror the hard atom and the shared helpers:
- weight == 0 → return 0 (atom disabled).
- dummy keys (len < 11) and no-day keys are skipped.
- locked-week club-day entries are skipped via ``parse_club_day_entries``.
- grades covered by the opponent, or with <=1 host team, are skipped.
- locked-week vars (``key[6] in locked_weeks``) are skipped.
"""
from itertools import combinations

from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    parse_club_day_entries, teams_by_grade,
)

# Default penalty weight when `PENALTY_WEIGHTS['regen_club_day_intra_club_matchup']`
# is unset. Large — a regen-soft analogue of a CRITICAL/HIGH hard rule.
REGEN_CLUB_DAY_INTRA_CLUB_MATCHUP_DEFAULT_WEIGHT = 35000


class ClubDayIntraClubMatchupRegenSoft(Atom):
    """SOFT: penalise intra-club derbies scheduled OFF the club-day."""

    canonical_name = 'ClubDayIntraClubMatchupRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_club_day_intra_club_matchup',
            REGEN_CLUB_DAY_INTRA_CLUB_MATCHUP_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        locked_weeks = set(data.get('locked_weeks', set()))

        teams = data['teams']
        club_team_lookup = {}
        for team in teams:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_club_day_intra_club_matchup',
            {'weight': weight, 'penalties': []},
        )

        n = 0
        for club_name, date_str, opponent in parse_club_day_entries(data):
            host_team_names = club_team_lookup.get(club_name, [])
            host_by_grade = teams_by_grade(host_team_names)
            opp_by_grade = (
                teams_by_grade(club_team_lookup.get(opponent, []))
                if opponent else {}
            )

            for grade, host_grade_teams in host_by_grade.items():
                # Mirror the hard atom's guards: only grades it would govern.
                if opponent is not None and grade in opp_by_grade:
                    continue
                if len(host_grade_teams) <= 1:
                    continue

                intra_pairs = list(combinations(host_grade_teams, 2))
                pair_set = (
                    set(intra_pairs) | {(b, a) for (a, b) in intra_pairs}
                )

                for key, var in X.items():
                    if len(key) < 11 or not key[3]:
                        continue
                    if locked_weeks and key[6] in locked_weeks:
                        continue
                    if key[2] != grade:
                        continue
                    if (key[0], key[1]) not in pair_set:
                        continue
                    # SOFT: penalise this intra-club matchup only if it is
                    # scheduled OFF the club-day date.
                    if key[7] == date_str:
                        continue
                    bucket['penalties'].append(var)
                    n += 1
        return n
