"""All club-day games for a club share a single field name.

Builds one indicator per field used by the club's games on its club-day date,
forces `sum(indicators) == 1`. Indicators are channeled via the helper-var
registry (`club_day_field_used`) so other atoms can reuse them.
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries,
)


class ClubDaySameField(Atom):
    canonical_name = 'ClubDaySameField'
    atom_group = 'ClubDay'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        club_team_lookup = {}
        for team in data['teams']:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        for club_name, date_str, _opponent in parse_club_day_entries(data):
            host_team_names = club_team_lookup.get(club_name, [])
            game_keys = club_day_game_keys(X, host_team_names, date_str)
            if not game_keys:
                continue

            field_vars = defaultdict(list)
            for key in game_keys:
                field_vars[key[9]].append(X[key])
            if len(field_vars) <= 1:
                continue

            indicators = []
            for field_name, vars_list in field_vars.items():
                ind = registry.get_or_create_bool(
                    ('club_day_field_used', club_name, field_name),
                    vars_list,
                    f'cd_field_{club_name}_{field_name}',
                )
                indicators.append(ind)
            model.Add(sum(indicators) == 1)
            n += 1
        return n
