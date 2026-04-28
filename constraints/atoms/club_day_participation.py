"""Every team in a club plays on the club's club-day date."""
from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries,
)


class ClubDayParticipation(Atom):
    canonical_name = 'ClubDayParticipation'
    atom_group = 'ClubDay'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        club_team_lookup = {}
        for team in data['teams']:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        for club_name, date_str, _opponent in parse_club_day_entries(data):
            club_team_names = club_team_lookup.get(club_name, [])
            game_keys = club_day_game_keys(X, club_team_names, date_str)
            if not game_keys:
                raise ValueError(
                    f'No games found for club {club_name} on {date_str}'
                )
            for team in club_team_names:
                team_vars = [
                    X[key] for key in game_keys
                    if team in (key[0], key[1])
                ]
                if team_vars:
                    model.Add(sum(team_vars) >= 1)
                    n += 1
        return n
