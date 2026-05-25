"""A club's games stagger across slots rather than overlap — soft + slack (spec-033 Unit E).

Originally (spec-021) extracted from the old `ClubGameSpread` lower no-double-up
bound (which is concurrency, not contiguity). For each
`(club, week, day, location, day_slot)` the club's games across all
grades/teams in that one timeslot are constrained so a parent isn't pulled to
two of their club's games at once. The aggregation across a location's fields
means this also covers cross-field (EF/WF/SF) overlap at NIHC: two club games
in the same time slot at NIHC, even on different fields, count as one overlap.

spec-033 Unit E — soft + slack stance (convenor direction):
The old capacity-aware `ceil(n_loc / S)` cap *permitted* forced double-ups when
a club had more games at a venue than the venue had distinct timeslots. The
convenor wants the opposite: stagger a club's games across slots wherever
possible rather than overlapping. So:

  - HARD ceiling of **1 overlap per slot (+slack)**:
        slack = data['constraint_slack'].get('ClubNoConcurrentSlot', 0)
        model.Add(sum(slot_vars) <= 1 + slack)
    With slack 0 a club may schedule at most one game per (location, slot) —
    no overlap at all. slack is the release valve when a venue genuinely has
    fewer distinct slots than the club's games (then the solve is infeasible at
    slack 0; raise slack — surfaced to the convenor, the first real solve is
    spec-035).

  - SOFT penalty pushing overlaps → 0: for each slot, an `over` IntVar with
        over >= sum(slot_vars) - 1   (and over >= 0)
    counts games beyond the first in that slot. Appended to
    `data['penalties']['ClubNoConcurrentSlot']`. Weight read via the direct-dict
    pattern atoms use (NOT the engine-only `_get_penalty_weight`).

The `no_field_slots` / `compute_no_field_slots` capacity no longer gates the
cap — the ceiling is a flat `1 + slack`, not capacity-aware.

Complements `SameGradeSameClubNoConcurrency` (which covers same-grade clashes)
by covering the cross-grade club case. Non-engine atom (dispatched via the
stages.py legacy-class fallback / staged path). The no-flag single solve
dispatches it via that same staged path (it is part of the full constraint
set) — same as `BalancedByeSpacing`).
"""
from collections import defaultdict

from constraints.atoms.base import Atom, get_team_club_map


# Default soft penalty weight for ClubNoConcurrentSlot. High — a physical
# clash (a club's parents pulled to two concurrent games). Set in config
# PENALTY_WEIGHTS; this is the fallback when the config key is absent.
CLUB_NO_CONCURRENT_SLOT_DEFAULT_WEIGHT = 200_000


class ClubNoConcurrentSlot(Atom):
    canonical_name = 'ClubNoConcurrentSlot'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        locked_weeks = set(data.get('locked_weeks', set()))
        team_club = get_team_club_map(data)

        slack_dict = data.get('constraint_slack', {}) or {}
        slack = int(slack_dict.get('ClubNoConcurrentSlot', 0) or 0)

        # Soft penalty bucket. Weight read via the direct-dict pattern atoms use.
        # weight==0 disables the soft term (the hard cap is unaffected).
        soft_weight = data.get('penalty_weights', {}).get(
            'ClubNoConcurrentSlot', CLUB_NO_CONCURRENT_SLOT_DEFAULT_WEIGHT
        )
        soft_bucket = None
        if soft_weight:
            soft_bucket = data.setdefault('penalties', {}).setdefault(
                'ClubNoConcurrentSlot',
                {'weight': soft_weight, 'penalties': []},
            )

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
            for day_slot, slot_vars in slot_map.items():
                if len(slot_vars) < 2:
                    continue  # this slot can hold at most one game regardless
                # HARD: at most 1 + slack games for this club in this
                # (location, slot) — aggregated across the location's fields, so
                # this covers cross-field (EF/WF/SF) overlap at NIHC.
                model.Add(sum(slot_vars) <= 1 + slack)
                n += 1
                # SOFT: push overlaps -> 0. `over` counts games beyond the
                # first in this slot. over >= sum - 1 and over >= 0.
                if soft_bucket is not None:
                    over = model.NewIntVar(
                        0, len(slot_vars),
                        f'cncs_over_{club}_{week}_{day}_{location}_{day_slot}'
                    )
                    model.Add(over >= sum(slot_vars) - 1)
                    soft_bucket['penalties'].append(over)
        return n
