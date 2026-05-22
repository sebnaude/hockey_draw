"""Venue earliest-slot fill — games at a venue pack into the earliest timeslots.

spec-021. Replaces the hard behaviour of the old `EnsureBestTimeslotChoices`
engine method (no-gap + slot-number bounding + cross-field stacking) with a
single anchored monotone-fill chain over **combined-field** slot indicators.

For each `(week, date, location)` an indicator is channeled per `day_slot`
(OR across every field at that location in that slot, via the pool registry).
`enforce_monotone_fill` then makes "use slot s ⇒ use slot s-1" for consecutive
slots — so the occupied slots are exactly `{first .. k}`: no interior gaps AND
anchored to the earliest available slot. Packing into the earliest slots is
what structurally avoids the 7 pm slot, so no separate 7 pm penalty is needed.

This is a HARD structural rule (severity 2). It deliberately drops the old
`AddDivisionEquality`/`nts`/`BROADMEADOW_MAX_SLOTS` slot-cap IntVars and the
per-field stacking detail — the venue-level monotone fill subsumes their
schedule-shape effect at a fraction of the variable cost.
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._contiguity import (
    enforce_monotone_fill,
    slot_used_indicators,
)


class VenueEarliestSlotFill(Atom):
    canonical_name = 'VenueEarliestSlotFill'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        locked_weeks = set(data.get('locked_weeks', set()))

        # (week, date, location) -> {day_slot: [vars]}, combined across fields.
        venue_slots = defaultdict(lambda: defaultdict(list))
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
            # < 2 occupied-able slots → nothing to order.
            if len(vars_by_slot) < 2:
                continue
            slot_inds = slot_used_indicators(
                registry, vars_by_slot, 'venue_slot_used', week, date, location,
            )
            n += enforce_monotone_fill(model, slot_inds)
        return n
