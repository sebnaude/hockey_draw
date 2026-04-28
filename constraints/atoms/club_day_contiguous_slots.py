"""Club-day games occupy contiguous timeslots — no gaps of empty middle slots.

For each `(club, day_slot)` an indicator is channeled via the registry
(`club_day_slot_used`). For each middle slot in sorted order, when that slot's
indicator is 0 the prior + following indicators must sum ≤ 1 (so empty slots
can't be flanked by used slots on both sides).
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries,
)


class ClubDayContiguousSlots(Atom):
    canonical_name = 'ClubDayContiguousSlots'
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

            slot_vars = defaultdict(list)
            for key in game_keys:
                slot_vars[key[4]].append(X[key])
            if len(slot_vars) < 3:
                continue

            slot_inds = {}
            for ds, vars_list in slot_vars.items():
                slot_inds[ds] = registry.get_or_create_bool(
                    ('club_day_slot_used', club_name, ds),
                    vars_list,
                    f'cd_slot_{club_name}_{ds}',
                )

            sorted_slots = sorted(slot_inds.keys())
            for i in range(1, len(sorted_slots) - 1):
                ps, cs, ns = (
                    sorted_slots[i - 1], sorted_slots[i], sorted_slots[i + 1],
                )
                model.Add(
                    slot_inds[ps] + slot_inds[ns] <= 1
                ).OnlyEnforceIf(slot_inds[cs].Not())
                n += 1
        return n
