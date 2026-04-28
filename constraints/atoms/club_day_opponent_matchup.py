"""On club day, when an opponent is set, force cross-club matchups for each grade.

Mirrors `original.py:ClubDayConstraint` if-branch (lines 690–700). Applies per
grade where the opponent has at least one team — forces
`sum(cross_vars) >= min(host_grade_count, opp_grade_count)`. Decision #4 locks
this atom to the original.py opponent semantics; ai.py's date-only form was a
regression and is dropped.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries, teams_by_grade,
)


class ClubDayOpponentMatchup(Atom):
    canonical_name = 'ClubDayOpponentMatchup'
    atom_group = 'ClubDay'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        teams = data['teams']
        club_team_lookup = {}
        for team in teams:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        for club_name, date_str, opponent in parse_club_day_entries(data):
            if opponent is None:
                continue
            host_team_names = club_team_lookup.get(club_name, [])
            opp_team_names = club_team_lookup.get(opponent, [])
            host_by_grade = teams_by_grade(host_team_names)
            opp_by_grade = teams_by_grade(opp_team_names)
            game_keys = club_day_game_keys(X, host_team_names, date_str)
            if not game_keys:
                continue

            for grade, host_grade_teams in host_by_grade.items():
                if grade not in opp_by_grade:
                    continue
                opp_grade_teams = opp_by_grade[grade]
                host_set = set(host_grade_teams)
                opp_set = set(opp_grade_teams)
                cross_vars = [
                    X[key] for key in game_keys
                    if key[2] == grade
                    and (
                        (key[0] in host_set and key[1] in opp_set)
                        or (key[0] in opp_set and key[1] in host_set)
                    )
                ]
                if cross_vars:
                    required = min(len(host_grade_teams), len(opp_grade_teams))
                    model.Add(sum(cross_vars) >= required)
                    n += 1
        return n
