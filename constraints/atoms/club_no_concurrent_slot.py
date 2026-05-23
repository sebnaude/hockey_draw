"""A club's games don't pile into one timeslot — capacity-aware (spec-021).

Extracted from the old `ClubGameSpread` lower no-double-up bound (which is
concurrency, not contiguity). For each `(club, week, day_slot, location)` the
club's games across all grades/teams in that one timeslot are capped so a
parent isn't pulled to two of their club's games at once.

The cap is **capacity-aware**: a flat "≤1 per slot" is wrong when a club has
more games at a venue than the venue has timeslots (then double-ups are forced).
So `cap = max(1, ceil(club_team_count / no_field_slots[location]))`, where
`no_field_slots[location]` is the derived distinct-time count for the venue
(`config.defaults.compute_no_field_slots`, surfaced as `data['no_field_slots']`).
e.g. a club with 3 teams at a 2-time venue → cap `ceil(3/2) = 2`.

Complements `SameGradeSameClubNoConcurrency` (which covers same-grade clashes)
by covering the cross-grade club case. HARD, severity 2.
"""
from collections import defaultdict
from math import ceil

from constraints.atoms.base import Atom, get_team_club_map


class ClubNoConcurrentSlot(Atom):
    canonical_name = 'ClubNoConcurrentSlot'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        locked_weeks = set(data.get('locked_weeks', set()))
        no_field_slots = data.get('no_field_slots', {})
        team_club = get_team_club_map(data)

        club_team_count = defaultdict(set)
        for team in data['teams']:
            club_team_count[team.club.name].add(team.name)

        # (club, week, day_slot, location) -> [vars]
        groups = defaultdict(list)
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            week = key[6]
            if week in locked_weeks:
                continue
            day_slot, location = key[4], key[10]
            clubs = set()
            for team_name in (key[0], key[1]):
                club = team_club.get(team_name)
                if club is not None:
                    clubs.add(club)
            for club in clubs:
                groups[(club, week, day_slot, location)].append(var)

        n = 0
        for (club, week, day_slot, location), vars_list in groups.items():
            if len(vars_list) < 2:
                continue  # at most one game can be selected anyway
            slots = no_field_slots.get(location, 0)
            n_teams = len(club_team_count.get(club, ()))
            cap = max(1, ceil(n_teams / slots)) if slots > 0 else 1
            if cap >= len(vars_list):
                continue  # cap can never bind for this group
            model.Add(sum(vars_list) <= cap)
            n += 1
        return n
