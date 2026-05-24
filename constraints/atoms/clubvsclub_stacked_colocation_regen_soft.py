"""Club-vs-club stacked co-location SOFT analogue (spec-027 regen-soft atom).

Soft mirror of ``ClubVsClubStackedCoLocation``. The HARD atom enforces that on
every Sunday where a club-pair has >= 2 grades playing, ALL those games land on
the SAME field with CONTIGUOUS day_slots (no internal gap). This SOFT atom keeps
the model feasible for ANY assignment of X and instead emits penalties equal to
the violation amount:

  - Co-field penalty (per pair, Sunday-week): ``max(0, distinct_fields_used - 1)``.
    0 when all the pair's games that Sunday share one field; ``k - 1`` when they
    spread across ``k`` fields. One unit == one extra field beyond the first.

  - Contiguity penalty (per pair, Sunday-week): one penalty BoolVar per internal
    slot gap, mirroring the HARD atom's exact gap definition. For each consecutive
    triple ``(prev, mid, next)`` over the sorted slot-used indicators, a gap at
    ``mid`` is ``prev_used == 1 AND next_used == 1 AND mid_used == 0``. One unit ==
    one internal gap.

CRITICAL: in regen mode the HARD stacked atoms are NOT applied, so this atom
builds EVERY field-used / slot-used indicator directly from X via its OWN pool
cache keys (``regen_cvc_stack_field_used`` / ``regen_cvc_stack_slot_used``). It
does NOT reuse the hard atoms' ``cvc_stack_*`` registry helpers, and it does NOT
look up ``cvc_stack_play`` (so no ``stack_active`` gate). The field/slot-used
indicators self-zero when few games are placed, so no gate is needed — the
penalty is exactly the structural violation amount.

Pair iteration and skip/no-op conditions mirror the hard atom:
  - Only pairs with >= 2 grades carrying a non-zero Sunday budget (else nothing
    can stack -> no-op).
  - Per (pair, week): need >= 2 of the pair's Sunday vars before any co-field
    penalty (one game cannot split fields by itself).
  - Contiguity needs >= 3 distinct slots before any internal middle slot exists.
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_stacked_shared import (
    collect_pair_week_sunday_vars,
    enumerate_club_pairs,
    pair_grade_sunday_meetings,
    per_pair_grade_meeting_counts,
    sorted_grades_by_desc_count,
)

# Module-level default for the penalty weight (core-feasibility-adjacent rule).
REGEN_CLUBVSCLUB_STACKED_COLOCATION_DEFAULT_WEIGHT = 70000


class ClubVsClubStackedCoLocationRegenSoft(Atom):
    """SOFT analogue of ClubVsClubStackedCoLocation.

    Emits, per (pair, Sunday-week) where the pair stacks (>= 2 grades),
    a co-field penalty (extra fields beyond one) plus one gap penalty per
    internal slot gap. The model is always feasible for any X.
    """

    canonical_name = 'ClubVsClubStackedCoLocationRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_clubvsclub_stacked_colocation',
            REGEN_CLUBVSCLUB_STACKED_COLOCATION_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        # Sunday weeks present in X (skip dummy / no-day; locked weeks are
        # already dropped by collect_pair_week_sunday_vars below).
        locked_weeks = set(data.get('locked_weeks', set()) or set())
        all_weeks = sorted({
            key[6]
            for key in X
            if len(key) >= 11 and key[3] == 'Sunday'
            and key[6] not in locked_weeks
        })
        if not all_weeks:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_clubvsclub_stacked_colocation',
            {'weight': weight, 'penalties': []},
        )

        n = 0
        for pair in enumerate_club_pairs(data):
            # Mirror the hard atom: only pairs with >= 2 grades having a
            # non-zero Sunday budget can stack. Otherwise no-op.
            grade_meetings_total = per_pair_grade_meeting_counts(data, pair)
            sunday_budget = {}
            for grade in grade_meetings_total:
                sb = pair_grade_sunday_meetings(data, pair, grade)
                if sb > 0:
                    sunday_budget[grade] = sb

            sorted_grades = sorted_grades_by_desc_count(sunday_budget)
            if len(sorted_grades) < 2:
                continue

            week_to_var_keys = collect_pair_week_sunday_vars(X, data, pair)

            for w in all_weeks:
                var_keys = week_to_var_keys.get(w, [])
                if len(var_keys) < 2:
                    # 0 or 1 game this week for the pair -> cannot split
                    # fields or open a gap. Skip (mirrors the hard atom).
                    continue

                # --- Co-field penalty: max(0, distinct fields used - 1). ---
                field_to_vars = defaultdict(list)
                for var, key in var_keys:
                    field_to_vars[key[9]].append(var)

                if len(field_to_vars) > 1:
                    field_inds = []
                    for fname, vars_at_field in field_to_vars.items():
                        fi = registry.get_or_create_bool(
                            ('regen_cvc_stack_field_used', pair, w, fname),
                            vars_at_field,
                            f'regen_cvc_fld_{pair[0]}_{pair[1]}_{w}_{fname}',
                        )
                        field_inds.append(fi)

                    max_extra = len(field_inds) - 1
                    pen = model.NewIntVar(
                        0, max_extra,
                        f'regen_cvc_cofield_pen_{pair[0]}_{pair[1]}_{w}',
                    )
                    # pen >= (#fields used) - 1, lb 0 => pen == max(0, used-1).
                    model.Add(pen >= sum(field_inds) - 1)
                    bucket['penalties'].append(pen)
                    n += 1

                # --- Contiguity penalty: one per internal slot gap. ---
                slot_to_vars = defaultdict(list)
                for var, key in var_keys:
                    slot_to_vars[key[4]].append(var)

                if len(slot_to_vars) >= 3:
                    slot_inds = {}
                    for ds, vars_at_slot in slot_to_vars.items():
                        slot_inds[ds] = registry.get_or_create_bool(
                            ('regen_cvc_stack_slot_used', pair, w, ds),
                            vars_at_slot,
                            f'regen_cvc_slot_{pair[0]}_{pair[1]}_{w}_{ds}',
                        )

                    sorted_slots = sorted(slot_inds.keys())
                    for i in range(1, len(sorted_slots) - 1):
                        ps = sorted_slots[i - 1]
                        cs = sorted_slots[i]
                        ns = sorted_slots[i + 1]
                        prev_used = slot_inds[ps]
                        mid_used = slot_inds[cs]
                        next_used = slot_inds[ns]

                        # v == 1 iff prev_used=1 AND next_used=1 AND mid_used=0.
                        v = model.NewBoolVar(
                            f'regen_cvc_gap_{pair[0]}_{pair[1]}_{w}_s{cs}'
                        )
                        model.Add(v >= prev_used + next_used - 1 - mid_used)
                        model.Add(v <= prev_used)
                        model.Add(v <= next_used)
                        model.Add(v <= 1 - mid_used)
                        bucket['penalties'].append(v)
                        n += 1

        return n


__all__ = ['ClubVsClubStackedCoLocationRegenSoft']
