"""NIHC field-fill ordering: EF must be used before SF (spec-003).

Sibling of `NIHCFillWFBeforeEF`. Enforces the implication
``SF_used -> EF_used`` per (date, day_slot) at NIHC where BOTH fields
exist as real timeslot options.

Together with the WF-before-EF atom, the two implications transitively
imply ``SF_used -> WF_used`` -- no third atom is required.

Helper variables are declared via the ``nihc_field_used`` kind so the
EF indicator is shared with the sibling atom (no duplicate channeling).
"""
from constraints.atoms.base import Atom
from constraints.atoms.nihc_fill_wf_before_ef import (
    _collect_nihc_field_vars,
    _declare_field_used_helper,
    _slot_field_index,
)


class NIHCFillEFBeforeSF(Atom):
    """Per (date, day_slot) at NIHC: SF_used -> EF_used.

    Severity 1 (CRITICAL): companion to `NIHCFillWFBeforeEF`. Same
    skip-when-not-a-valid-slot rule -- if EF or SF isn't a real option at
    that (date, day_slot), no constraint is added.
    """

    canonical_name = 'NIHCFillEFBeforeSF'
    atom_group = 'NIHCFieldFillOrder'

    def apply(self, model, X, data, registry) -> int:
        buckets = _collect_nihc_field_vars(X, data)
        if not buckets:
            return 0
        slot_fields = _slot_field_index(buckets)

        n = 0
        for (date, day_slot), present in slot_fields.items():
            if 'SF' not in present or 'EF' not in present:
                continue
            sf_vars = buckets[(date, day_slot, 'SF')]
            ef_vars = buckets[(date, day_slot, 'EF')]
            sf_used = _declare_field_used_helper(
                registry, date, day_slot, 'SF', sf_vars
            )
            ef_used = _declare_field_used_helper(
                registry, date, day_slot, 'EF', ef_vars
            )
            # SF_used implies EF_used: sf_used <= ef_used.
            model.Add(sf_used <= ef_used)
            n += 1
        return n
