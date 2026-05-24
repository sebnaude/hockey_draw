"""Venue earliest-slot fill — SOFT (regen) analogue of VenueEarliestSlotFill.

spec-027. Mirrors the HARD atom `VenueEarliestSlotFill` (spec-021) but instead
of forbidding gaps via `AddImplication`, it EMITS A PENALTY for each gap — a
(date, location, slot-pair) where a later slot is used while the immediately-
earlier slot at that venue/date is empty.

The model remains feasible for ANY assignment of X; a non-zero penalty simply
makes the objective worse so the solver is nudged toward earliest-packed layouts.

**Penalty semantics**: 1 unit = one (date, location, slot-pair) gap where a game
is scheduled in slot ``s`` while slot ``s-1`` (the immediately-earlier slot at
the same venue/date) is completely empty. Each such hole is tracked by a BoolVar
``v`` pinned to 1 exactly when ``used[s] == 1`` and ``used[s-1] == 0``.

Slot-used indicators are channeled via the registry pool under kind
``'venue_slot_used'`` — the same kind as the hard atom — so if both atoms are
loaded, they share the indicator BoolVars.

Weight key: ``penalty_weights['regen_venue_earliest_slot_fill']``
Default weight: 10000 (large: a missing-earliest-slot gap is a serious layout
flaw, though not hard-infeasible).
"""
from __future__ import annotations

from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._contiguity import slot_used_indicators

# Default penalty weight. High so the solver strongly disfavours gaps.
REGEN_VENUE_EARLIEST_SLOT_FILL_DEFAULT_WEIGHT = 10_000


class VenueEarliestSlotFillRegenSoft(Atom):
    """SOFT venue earliest-slot fill: penalise each slot-gap rather than forbid.

    spec-027. Severity 5 (VERY LOW) — never a hard constraint, always feasible.
    The penalty count equals the number of (date, location, slot-pair) holes.
    """

    canonical_name = 'VenueEarliestSlotFillRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_venue_earliest_slot_fill',
            REGEN_VENUE_EARLIEST_SLOT_FILL_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_venue_earliest_slot_fill',
            {'weight': weight, 'penalties': []},
        )

        locked_weeks = set(data.get('locked_weeks', set()))

        # (week, date, location) -> {day_slot: [vars]}, combined across fields.
        venue_slots: dict = defaultdict(lambda: defaultdict(list))
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            week = key[6]
            if week in locked_weeks:
                continue
            day_slot, date, location = key[4], key[7], key[10]
            venue_slots[(week, date, location)][day_slot].append(var)

        n = 0
        for (week, date, location), vars_by_slot in venue_slots.items():
            # < 2 distinct day_slots → no consecutive pair to check.
            if len(vars_by_slot) < 2:
                continue

            # Build (or reuse from the hard atom) slot-used indicators.
            slot_inds = slot_used_indicators(
                registry, vars_by_slot, 'venue_slot_used', week, date, location,
            )

            sorted_slots = sorted(slot_inds.keys())
            for i in range(1, len(sorted_slots)):
                s_prev = sorted_slots[i - 1]
                s_curr = sorted_slots[i]
                used_prev = slot_inds[s_prev]
                used_curr = slot_inds[s_curr]

                # Penalty var: 1 iff slot s_curr is used AND slot s_prev is NOT.
                # Three linear constraints pin v exactly for 0/1 indicator vars:
                #   v >= used_curr - used_prev  (if curr=1, prev=0 -> v>=1)
                #   v <= used_curr              (if curr=0 -> v=0)
                #   v <= 1 - used_prev          (if prev=1 -> v=0)
                label = f'regen_vef_{date}_{location[:4]}_{s_prev}_{s_curr}'
                v = model.NewBoolVar(label)
                model.Add(v >= used_curr - used_prev)
                model.Add(v <= used_curr)
                model.Add(v <= 1 - used_prev)

                bucket['penalties'].append(v)
                n += 1

        return n
