"""A club's games don't pile into one timeslot — capacity-aware (spec-021).

Extracted from the old `ClubGameSpread` lower no-double-up bound (which is
concurrency, not contiguity). For each `(club, week, day_slot, location)` the
club's games across all grades/teams in that one timeslot are capped so a
parent isn't pulled to two of their club's games at once.

The cap is **capacity-aware**: a flat "≤1 per slot" is wrong when a club has
more games at a venue than the venue has timeslots (then double-ups are forced).
The cap is `ceil(n_loc / S)` where `n_loc` = the club's games scheduled at that
location that day and `S = no_field_slots[location]` (the venue's distinct-time
count, derived by `config.defaults.compute_no_field_slots` and surfaced as
`data['no_field_slots']`). e.g. with 3 games at a 2-time venue → cap `ceil(3/2)=2`;
with 2 games → cap 1.

`n_loc` is itself a decision (which games land at the venue), so rather than a
fixed cap we encode the equivalent linear inequality per slot — no IntVar:

    slot_count <= ceil(n_loc / S)   ⟺   S * slot_count <= n_loc + S - 1

where `slot_count` = the club's games in that one slot and `n_loc` = the club's
games at that (location, day) across all slots. When `n_loc <= S` this forces
`slot_count <= 1` (no double-ups); each extra game over `S` raises the per-slot
ceiling by one only when genuinely unavoidable.

Complements `SameGradeSameClubNoConcurrency` (which covers same-grade clashes)
by covering the cross-grade club case. HARD, severity 2.
"""
from collections import defaultdict

from constraints.atoms.base import Atom, get_team_club_map


class ClubNoConcurrentSlot(Atom):
    canonical_name = 'ClubNoConcurrentSlot'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        locked_weeks = set(data.get('locked_weeks', set()))
        no_field_slots = data.get('no_field_slots', {})
        team_club = get_team_club_map(data)

        # (club, week, day, location) -> {day_slot: [vars]}
        groups = defaultdict(lambda: defaultdict(list))
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            week = key[6]
            if week in locked_weeks:
                continue
            day, day_slot, location = key[3], key[4], key[10]
            clubs = set()
            for team_name in (key[0], key[1]):
                club = team_club.get(team_name)
                if club is not None:
                    clubs.add(club)
            for club in clubs:
                groups[(club, week, day, location)][day_slot].append(var)

        n = 0
        for (club, week, day, location), slot_map in groups.items():
            all_vars = [v for vs in slot_map.values() for v in vs]
            if len(all_vars) < 2:
                continue  # can't have two concurrent games anyway
            # S = venue timeslot capacity; fall back to the number of distinct
            # slots seen (so cap is 1 when n_loc <= available slots).
            S = no_field_slots.get(location) or len(slot_map) or 1
            n_loc = sum(all_vars)  # the club's games scheduled at this venue/day
            for day_slot, slot_vars in slot_map.items():
                if len(slot_vars) < 2:
                    continue  # this slot can hold at most one game regardless
                # slot_count <= ceil(n_loc / S)  ⟺  S*slot_count <= n_loc + S - 1
                model.Add(S * sum(slot_vars) <= n_loc + S - 1)
                n += 1
        return n
