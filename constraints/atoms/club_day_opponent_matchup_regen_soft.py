"""SOFT analogue of ``ClubDayOpponentMatchup`` (spec-027 regen-soft).

The HARD atom (``constraints/atoms/club_day_opponent_matchup.py``) enforces
that, when a club's club-day names an opponent, the cross-club host-vs-opponent
matchups for each shared grade are scheduled ONTO the club-day:
``sum(cross_vars on club-day) >= min(host_grade_count, opp_grade_count)``.

This SOFT analogue emits NO hard clause. Instead, for every grade the hard atom
would govern (the opponent has at least one team in that grade), it adds ONE
penalty unit per cross-club (host-vs-opponent) matchup of that grade scheduled
OUTSIDE the club-day (i.e. on any date other than ``date_str``). The model stays
feasible for ANY X: the penalty simply measures, and the objective discourages,
opponent matchups that happen off the club-day.

One penalty unit == one cross-club host-vs-opponent matchup game scheduled on a
non-club-day date. Because each such matchup var is itself a 0/1, the penalty
for a var is just the var (it is 1 exactly when that off-day matchup is
scheduled), so the bucket gets the raw decision vars appended — no extra
linearization needed.

Skip/no-op conditions mirror the hard atom and the shared helpers:
- weight == 0 → return 0 (atom disabled).
- entries with no opponent are skipped.
- dummy keys (len < 11) and no-day keys are skipped.
- locked-week club-day entries are skipped via ``parse_club_day_entries``.
- grades the opponent has no teams in are skipped.
- locked-week vars (``key[6] in locked_weeks``) are skipped.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    parse_club_day_entries, teams_by_grade,
)

# Default penalty weight when `PENALTY_WEIGHTS['regen_club_day_opponent_matchup']`
# is unset. Large — a regen-soft analogue of a CRITICAL/HIGH hard rule.
REGEN_CLUB_DAY_OPPONENT_MATCHUP_DEFAULT_WEIGHT = 35000


class ClubDayOpponentMatchupRegenSoft(Atom):
    """SOFT: penalise cross-club opponent matchups scheduled OFF the club-day."""

    canonical_name = 'ClubDayOpponentMatchupRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_club_day_opponent_matchup',
            REGEN_CLUB_DAY_OPPONENT_MATCHUP_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        locked_weeks = set(data.get('locked_weeks', set()))

        teams = data['teams']
        club_team_lookup = {}
        for team in teams:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_club_day_opponent_matchup',
            {'weight': weight, 'penalties': []},
        )

        n = 0
        for club_name, date_str, opponent in parse_club_day_entries(data):
            if opponent is None:
                continue
            host_team_names = club_team_lookup.get(club_name, [])
            opp_team_names = club_team_lookup.get(opponent, [])
            host_by_grade = teams_by_grade(host_team_names)
            opp_by_grade = teams_by_grade(opp_team_names)

            for grade, host_grade_teams in host_by_grade.items():
                # Mirror the hard atom's guard: only grades the opponent shares.
                if grade not in opp_by_grade:
                    continue
                host_set = set(host_grade_teams)
                opp_set = set(opp_by_grade[grade])

                for key, var in X.items():
                    if len(key) < 11 or not key[3]:
                        continue
                    if locked_weeks and key[6] in locked_weeks:
                        continue
                    if key[2] != grade:
                        continue
                    # Cross-club host-vs-opponent matchup (either orientation).
                    is_cross = (
                        (key[0] in host_set and key[1] in opp_set)
                        or (key[0] in opp_set and key[1] in host_set)
                    )
                    if not is_cross:
                        continue
                    # SOFT: penalise this opponent matchup only if it is
                    # scheduled OFF the club-day date.
                    if key[7] == date_str:
                        continue
                    bucket['penalties'].append(var)
                    n += 1
        return n
