"""Shared contiguity building blocks for timeslot-fill atoms.

Three constraints want "don't leave holes between used timeslots" but at
different scopes and with different start semantics:

- **anchored** (venue earliest-fill): used slots must pack into the *earliest*
  available slots — using slot ``s`` requires every earlier slot be used too.
- **floating** (club contiguity): the block of used slots must be contiguous
  (no interior hole) but may start anywhere.

Both reduce to an implication/coincidence chain over per-slot ``slot_used``
indicator BoolVars — no ``AddDivisionEquality`` / range / min-max IntVars. This
module factors out the indicator construction (via the pool-style
``HelperVarRegistry`` API, matching ``_club_day_shared``) and the two distinct
chain semantics so the venue, club-spread, and club-day atoms share one cheap
encoding without merging into a single atom (see GOALS §2 — "extract a helper,
don't merge").
"""
from __future__ import annotations

from typing import Dict, List


def slot_used_indicators(registry, vars_by_slot: Dict[int, List], kind: str,
                         *key_prefix) -> Dict[int, object]:
    """One channeled ``slot_used`` BoolVar per slot in ``vars_by_slot``.

    ``vars_by_slot`` maps a slot index (``key[4]``/``day_slot``) to the list of
    decision vars that would occupy that slot. Each indicator is ``1`` iff any
    of that slot's vars is selected (``AddMaxEquality`` channeling, via
    ``registry.get_or_create_bool``). The cache key is ``(kind, *key_prefix,
    slot)`` so two callers asking for the same slot indicator share one var.

    Returns ``{slot: BoolVar}`` for every slot present in ``vars_by_slot``.
    """
    slot_inds: Dict[int, object] = {}
    for slot, vars_list in vars_by_slot.items():
        slot_inds[slot] = registry.get_or_create_bool(
            (kind, *key_prefix, slot),
            vars_list,
            f'{kind}_{"_".join(str(p) for p in key_prefix)}_{slot}',
        )
    return slot_inds


def enforce_no_gaps(model, slot_inds: Dict[int, object]) -> int:
    """FLOATING strict no-hole: the used block is contiguous, starts anywhere.

    For each consecutive triple ``(prev, mid, next)`` over the sorted slots,
    when ``mid`` is unused the flanking ``prev`` and ``next`` cannot both be
    used (``prev + next <= 1`` enforced only if ``mid`` is 0). A used slot on
    both sides of an empty middle slot is the definition of an interior hole.

    Returns the number of constraints added.
    """
    sorted_slots = sorted(slot_inds.keys())
    n = 0
    for i in range(1, len(sorted_slots) - 1):
        ps, cs, ns = sorted_slots[i - 1], sorted_slots[i], sorted_slots[i + 1]
        model.Add(
            slot_inds[ps] + slot_inds[ns] <= 1
        ).OnlyEnforceIf(slot_inds[cs].Not())
        n += 1
    return n


def enforce_monotone_fill(model, slot_inds: Dict[int, object]) -> int:
    """ANCHORED no-hole + earliest-start: used slots pack into the earliest.

    For each consecutive pair ``(s_prev, s)`` over the sorted slots, using ``s``
    implies ``s_prev`` is used (``AddImplication(slot_inds[s], slot_inds[s_prev])``).
    Transitively this forces all earlier slots used before any later one, so the
    occupied slots are exactly ``{first .. k}`` — no holes AND anchored to the
    earliest available slot. Strictly stronger than ``enforce_no_gaps``.

    Returns the number of constraints added.
    """
    sorted_slots = sorted(slot_inds.keys())
    n = 0
    for i in range(1, len(sorted_slots)):
        s_prev, s = sorted_slots[i - 1], sorted_slots[i]
        model.AddImplication(slot_inds[s], slot_inds[s_prev])
        n += 1
    return n
