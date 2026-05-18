"""NIHC field-fill ordering: WF must be used before EF (spec-003).

At Newcastle International Hockey Centre (Broadmeadow), the operator
preference is to fill West Field before East Field within any single
(date, day_slot) bucket. This atom enforces the implication
``EF_used -> WF_used`` per (date, day_slot) where BOTH fields exist as
real timeslot options.

The atom is the generalisation of the legacy "last game of the day on
West Field" perennial rule -- WF priority over EF applies to every slot
of the day, not only the final one.

Edge cases:
- (date, day_slot) buckets where WF has no real timeslot are skipped.
  Asserting ``EF_used -> WF_used`` when WF isn't even available would
  be infeasible by construction.
- (date, day_slot) buckets where EF has no real timeslot are also
  skipped (nothing to constrain -- the LHS is structurally 0).
- Dummy keys (len < 11) and locked weeks are skipped via the standard
  uniformity rules in the unified engine, mirrored here.

Helper variables are declared via the ``nihc_field_used`` kind so the
sibling atom ``NIHCFillEFBeforeSF`` shares the same EF indicator.
"""
from collections import defaultdict
from typing import Dict, Set, Tuple

from constraints.atoms.base import Atom, BROADMEADOW


def _collect_nihc_field_vars(
    X: Dict, data: Dict
) -> Dict[Tuple[str, int, str], list]:
    """Return {(date, day_slot, field_name): [decision vars]} restricted
    to NIHC and to real (non-dummy, non-locked) variables.
    """
    locked_weeks = set(data.get('locked_weeks', set()))
    buckets: Dict[Tuple[str, int, str], list] = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11:
            continue
        if not key[3]:  # dummy / no-day
            continue
        if key[10] != BROADMEADOW:
            continue
        if locked_weeks and key[6] in locked_weeks:
            continue
        date = key[7]
        day_slot = key[4]
        field_name = key[9]
        buckets[(date, day_slot, field_name)].append(var)
    return buckets


def _slot_field_index(
    buckets: Dict[Tuple[str, int, str], list]
) -> Dict[Tuple[str, int], Set[str]]:
    """Reverse the bucket index to {(date, day_slot): set(field_names)}.

    A field appears in the set iff at least one real variable exists for
    that field at the (date, day_slot). That is the cheap-and-correct way
    to detect "is this field a valid slot here?" without having to walk
    `data['timeslots']` separately.
    """
    out: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
    for (date, day_slot, field_name), vars_list in buckets.items():
        if vars_list:
            out[(date, day_slot)].add(field_name)
    return out


def _declare_field_used_helper(registry, date: str, day_slot: int,
                               field_name: str, vars_list):
    """Idempotent declaration + immediate build via the pool API.

    The atoms run inside the `apply()` step where the registry is
    typically already frozen (the engine freezes after `declare_helpers`).
    We therefore use the pool-style `get_or_create_bool` cache which is
    valid post-freeze. Cache key is namespaced by `nihc_field_used` so it
    cannot collide with other kinds.
    """
    return registry.get_or_create_bool(
        ('nihc_field_used', date, day_slot, field_name),
        vars_list,
        f'nihc_used_{date}_s{day_slot}_{field_name}',
    )


class NIHCFillWFBeforeEF(Atom):
    """Per (date, day_slot) at NIHC: EF_used -> WF_used.

    Severity 1 (CRITICAL): operational rule that must hold for a
    publishable draw.
    """

    canonical_name = 'NIHCFillWFBeforeEF'
    atom_group = 'NIHCFieldFillOrder'

    def apply(self, model, X, data, registry) -> int:
        buckets = _collect_nihc_field_vars(X, data)
        if not buckets:
            return 0
        slot_fields = _slot_field_index(buckets)

        n = 0
        for (date, day_slot), present in slot_fields.items():
            if 'EF' not in present or 'WF' not in present:
                # Either EF isn't a valid slot here (nothing to imply) or
                # WF isn't a valid slot here (cannot demand it).
                continue
            wf_vars = buckets[(date, day_slot, 'WF')]
            ef_vars = buckets[(date, day_slot, 'EF')]
            wf_used = _declare_field_used_helper(
                registry, date, day_slot, 'WF', wf_vars
            )
            ef_used = _declare_field_used_helper(
                registry, date, day_slot, 'EF', ef_vars
            )
            # EF_used implies WF_used: ef_used <= wf_used.
            model.Add(ef_used <= wf_used)
            n += 1
        return n
