"""Club-day games occupy contiguous timeslots — no gaps of empty middle slots.

For each `(club, day_slot)` an indicator is channeled via the registry
(`club_day_slot_used`). For each middle slot in sorted order, when that slot's
indicator is 0 the prior + following indicators must sum ≤ 1 (so empty slots
can't be flanked by used slots on both sides).
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._contiguity import enforce_no_gaps, slot_used_indicators
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

            # spec-021: shared floating no-gap primitive. The pool key
            # ('club_day_slot_used', club_name, slot) is preserved exactly, so
            # the channeled indicators are behaviour-identical to before.
            slot_inds = slot_used_indicators(
                registry, slot_vars, 'club_day_slot_used', club_name)
            n += enforce_no_gaps(model, slot_inds)
        return n
