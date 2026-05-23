"""NIHC field-fill ordering: prefer EF before SF (spec-003 / spec-016).

Sibling of `NIHCFillWFBeforeEF`. **spec-016: SOFT symmetry-breaker.** Per
(date, day_slot) at NIHC where BOTH EF and SF are real options, adds a penalty
term that is 1 exactly when SF is used while EF is empty (``SF_used AND NOT
EF_used``), into the shared ``nihc_fill_order`` penalty bucket.

Together with the WF-before-EF atom, the two soft penalties make WF→EF→SF the
canonical (cheapest) fill order; SF-before-WF/EF emerges for free since SF is
lowest priority. No hard implication is added.

Helper variables are registered under the ``nihc_field_used`` kind so the
EF indicator is shared with the sibling atom (no duplicate channeling).
"""
from constraints.atoms.base import Atom
from constraints.atoms.nihc_fill_wf_before_ef import (
    _add_order_penalty,
    _collect_nihc_field_vars,
    _declare_field_used_helper,
    _fill_order_bucket,
    _fill_order_weight,
    _slot_field_index,
)


class NIHCFillEFBeforeSF(Atom):
    """Per (date, day_slot) at NIHC: SOFT-prefer EF used before SF.

    spec-016: severity 5 (VERY LOW) symmetry-breaker — companion to
    `NIHCFillWFBeforeEF`. Same skip-when-not-a-valid-slot rule -- if EF or SF
    isn't a real option at that (date, day_slot), no penalty is added.
    """

    canonical_name = 'NIHCFillEFBeforeSF'
    atom_group = 'NIHCFieldFillOrder'

    def apply(self, model, X, data, registry) -> int:
        weight = _fill_order_weight(data)
        if weight == 0:
            return 0
        buckets = _collect_nihc_field_vars(X, data)
        if not buckets:
            return 0
        slot_fields = _slot_field_index(buckets)
        bucket = _fill_order_bucket(data, weight)

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
            # SOFT: penalise SF_used AND NOT EF_used (out-of-order fill).
            _add_order_penalty(
                model, bucket, ef_used, sf_used,
                f'nihc_ef_sf_viol_{date}_s{day_slot}',
            )
            n += 1
        return n
