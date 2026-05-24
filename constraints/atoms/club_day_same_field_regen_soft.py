"""Soft analogue of ClubDaySameField — penalise field splits instead of forbidding them.

For each club-day matchup, the HARD atom (`ClubDaySameField`) enforces that ALL
of the club's games on its club-day date land on exactly one field (via
``sum(field_used_indicators) == 1``).  This SOFT atom keeps the solver feasible
for ANY assignment but charges a penalty equal to the number of *extra* fields
used beyond one:

    penalty = max(0, number_of_distinct_fields_used - 1)

Concretely: build one ``field_used`` indicator per field that has at least one
candidate variable for the matchup (reusing the ``club_day_field_used`` pool so
the hard atom can share them when both are active), then add an IntVar ``pen``
and post:

    pen >= sum(field_used_indicators) - 1   (≥ 0 always, = 0 iff ≤1 field used)

The penalty per matchup is 0 when all games are on one field and k-1 when spread
across k fields.  One unit of penalty == one extra field beyond the first.
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys,
    parse_club_day_entries,
)

# Module-level default for the penalty weight.
REGEN_CLUB_DAY_SAME_FIELD_DEFAULT_WEIGHT = 30_000


class ClubDaySameFieldRegenSoft(Atom):
    """Soft analogue of ClubDaySameField.

    Emits a penalty = (number of distinct fields used by the club-day matchup)
    minus 1, so penalty == 0 iff all games are on one field.  The model is
    always feasible.
    """

    canonical_name = 'ClubDaySameFieldRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_club_day_same_field', REGEN_CLUB_DAY_SAME_FIELD_DEFAULT_WEIGHT
        )
        if weight == 0:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_club_day_same_field', {'weight': weight, 'penalties': []}
        )

        club_team_lookup = {}
        for team in data['teams']:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        n = 0
        for club_name, date_str, _opponent in parse_club_day_entries(data):
            host_team_names = club_team_lookup.get(club_name, [])
            game_keys = club_day_game_keys(X, host_team_names, date_str)
            if not game_keys:
                continue

            # Group candidate vars by field_name (key index 9).
            field_vars = defaultdict(list)
            for key in game_keys:
                field_vars[key[9]].append(X[key])

            # If only one field has vars there is nothing to split — skip as
            # the hard atom does.
            if len(field_vars) <= 1:
                continue

            # Build (or reuse) a field_used indicator per field via the pool.
            field_used_indicators = []
            max_fields = len(field_vars)
            for field_name, vars_list in field_vars.items():
                ind = registry.get_or_create_bool(
                    ('club_day_field_used', club_name, field_name),
                    vars_list,
                    f'cd_field_{club_name}_{field_name}',
                )
                field_used_indicators.append(ind)

            # Penalty variable: pen ≥ sum(indicators) - 1, lb=0, ub=max_fields-1.
            pen = model.NewIntVar(0, max_fields - 1,
                                  f'cd_sf_pen_{club_name}_{date_str}')
            model.Add(pen >= sum(field_used_indicators) - 1)
            bucket['penalties'].append(pen)
            n += 1

        return n
