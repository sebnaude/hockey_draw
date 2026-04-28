"""On club day, same-grade duplicate teams within the club must matchup (derbies).

Applies only when there is no opponent for that grade — i.e. opponent is None,
or opponent has no teams in this grade. Mirrors `original.py:ClubDayConstraint`
elif-branch (lines 701–716).
"""
from itertools import combinations

from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries, teams_by_grade,
)


class ClubDayIntraClubMatchup(Atom):
    canonical_name = 'ClubDayIntraClubMatchup'
    atom_group = 'ClubDay'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        teams = data['teams']
        club_team_lookup = {}
        for team in teams:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        for club_name, date_str, opponent in parse_club_day_entries(data):
            host_team_names = club_team_lookup.get(club_name, [])
            host_by_grade = teams_by_grade(host_team_names)
            opp_by_grade = (
                teams_by_grade(club_team_lookup.get(opponent, []))
                if opponent else {}
            )
            game_keys = club_day_game_keys(X, host_team_names, date_str)
            if not game_keys:
                continue

            for grade, host_grade_teams in host_by_grade.items():
                # Only run this branch when ClubDayOpponentMatchup wouldn't apply.
                if opponent is not None and grade in opp_by_grade:
                    continue
                if len(host_grade_teams) <= 1:
                    continue

                intra_pairs = list(combinations(host_grade_teams, 2))
                pair_set = set(intra_pairs) | {(b, a) for (a, b) in intra_pairs}
                intra_vars = [
                    X[key] for key in game_keys
                    if key[2] == grade and (key[0], key[1]) in pair_set
                ]
                if intra_vars:
                    expected = len(host_grade_teams) // 2
                    model.Add(sum(intra_vars) >= expected)
                    n += 1
        return n
