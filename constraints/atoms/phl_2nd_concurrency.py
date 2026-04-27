"""PHL and 2nd-grade games of the same club cannot share a Broadmeadow slot."""
from collections import defaultdict

from constraints.atoms.base import (
    Atom, BROADMEADOW, get_team_club_map, iter_grade_keys, iter_phl_keys,
)


class PHLAnd2ndConcurrencyAtBroadmeadow(Atom):
    canonical_name = 'PHLAnd2ndConcurrencyAtBroadmeadow'
    atom_group = 'PHLAndSecondGradeTimes'

    def apply(self, model, X, data, registry) -> int:
        team_club = get_team_club_map(data)
        clubs_with_2nd = {
            t.club.name for t in data['teams'] if t.grade == '2nd'
        }

        phl_per_slot_club = defaultdict(list)
        second_per_slot_club = defaultdict(list)

        for key, var in iter_phl_keys(X, data):
            location = key[10]
            if location != BROADMEADOW:
                continue
            week, day, day_slot = key[6], key[3], key[4]
            for team in (key[0], key[1]):
                club = team_club.get(team)
                if club and club in clubs_with_2nd:
                    phl_per_slot_club[(week, day, day_slot, location, club)].append(var)

        for key, var in iter_grade_keys(X, data, '2nd'):
            location = key[10]
            if location != BROADMEADOW:
                continue
            week, day, day_slot = key[6], key[3], key[4]
            for team in (key[0], key[1]):
                club = team_club.get(team)
                if club:
                    second_per_slot_club[(week, day, day_slot, location, club)].append(var)

        n = 0
        for slot_club, phl_vars in phl_per_slot_club.items():
            second_vars = second_per_slot_club.get(slot_club, [])
            if phl_vars and second_vars:
                model.Add(sum(phl_vars) + sum(second_vars) <= 1)
                n += 1
        return n
