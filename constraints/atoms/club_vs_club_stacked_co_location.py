"""spec-005 atom 2/2: co-location on stacked weekends.

On every Sunday where (club_pair, week) has ≥ 2 grades playing for the same
pair, all those games must be on the SAME field with contiguous day_slots
(no internal gaps). Generalises the legacy PHL/2nd back-to-back rule to
every stacked grade — the convenor's intent is "the bigger group brings
along the smaller grades on one field, back-to-back."

## Stack-active gating

The `ClubVsClubStackedWeekends` atom enforces a strict implication chain in
descending-count order. So `stack_active[pair, w] = play[g_1, w]` where
`g_1` is the **second-ranked** grade by Sunday count. When `g_1` plays for
the pair this week, by the implication chain `g_0` (highest count) also
plays — so ≥ 2 grades are active. Conversely if `g_1` doesn't play, only
`g_0` (or none) plays — no stacking required.

If a pair has only ONE grade with non-zero Sunday budget the atom is a
no-op for that pair (nothing to co-locate).

## Co-location constraints

For each (pair, week) where there are ≥ 2 grades active:

1. **Same field:** sum over distinct field names f of `field_used[pair, w, f] <= 1`
   when `stack_active == 1`. Build `field_used` via `AddMaxEquality` over
   the pair's Sunday vars at field f on week w.

2. **Contiguous slots:** for each (pair, week), build `slot_used[pair, w, s]`
   per day_slot. For each middle slot, enforce "empty middle => prior + next
   <= 1" when `stack_active == 1` (the standard club-day pattern).

Both constraints are conditional on `stack_active` via `OnlyEnforceIf`.

## Helper-var keys

- `(STACK_FIELD_USED_PREFIX, club_pair, week, field_name)` — per-field indicator.
- `(STACK_SLOT_USED_PREFIX, club_pair, week, day_slot)` — per-slot indicator.
- `(STACK_ACTIVE_PREFIX, club_pair, week)` — stack-active gate (alias of
  `play[g_1, w]`, registered for symmetry / introspection).

Parallel to the existing `club_day_field_used` / `club_day_slot_used`
helper kinds (whose keys are `(club, field)` / `(club, slot)` — different
shapes, different semantics, no collision). Spec-005's keys carry the week
explicitly because co-location is per-week, not per-season.
"""
from __future__ import annotations

from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_stacked_shared import (
    STACK_ACTIVE_PREFIX,
    STACK_FIELD_USED_PREFIX,
    STACK_PLAY_PREFIX,
    STACK_SLOT_USED_PREFIX,
    collect_pair_week_sunday_vars,
    enumerate_club_pairs,
    pair_grade_sunday_meetings,
    per_pair_grade_meeting_counts,
    sorted_grades_by_desc_count,
)


class ClubVsClubStackedCoLocation(Atom):
    """Same-field + contiguous-slot constraint, gated by stack-active per
    (club_pair, week). Must run AFTER `ClubVsClubStackedWeekends` so the
    `play` indicators are in the registry."""

    canonical_name = 'ClubVsClubStackedCoLocation'
    atom_group = 'ClubVsClubStackedAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        all_weeks = sorted({
            key[6]
            for key in X
            if len(key) >= 11 and key[3] == 'Sunday'
        })
        if not all_weeks:
            return 0

        for pair in enumerate_club_pairs(data):
            # 1. Determine the second-ranked grade (by Sunday budget).
            grade_meetings_total = per_pair_grade_meeting_counts(data, pair)
            sunday_budget = {}
            for grade in grade_meetings_total:
                sb = pair_grade_sunday_meetings(data, pair, grade)
                if sb > 0:
                    sunday_budget[grade] = sb

            sorted_grades = sorted_grades_by_desc_count(sunday_budget)
            if len(sorted_grades) < 2:
                # Only one grade plays Sundays for this pair — no stacking
                # possible, nothing to co-locate. The PHL atom previously
                # handled this case via the back-to-back rule when 2nd
                # grade played the same round; we preserve that semantics
                # since 2-grade pairs still trigger here.
                continue

            second_grade, _ = sorted_grades[1]

            # 2. Gather per-week vars for this pair (any grade, Sunday only).
            week_to_var_keys = collect_pair_week_sunday_vars(X, data, pair)

            for w in all_weeks:
                var_keys = week_to_var_keys.get(w, [])
                if len(var_keys) < 2:
                    # 0 or 1 game total this week for this pair => no
                    # co-location to enforce. (1 game cannot violate
                    # same-field or contiguity by itself.)
                    continue

                # Gate: stack_active = play[second_grade, w]. Registered by
                # ClubVsClubStackedWeekends; here we just look it up. If
                # the stacking atom hasn't run yet, that's a programming
                # error (this atom must run AFTER).
                play_g1 = registry.get(
                    (STACK_PLAY_PREFIX, pair, second_grade, w)
                )
                if play_g1 is None:
                    raise RuntimeError(
                        f'ClubVsClubStackedCoLocation: missing play indicator '
                        f'for pair={pair} grade={second_grade} week={w}. '
                        f'Did ClubVsClubStackedWeekends run before this atom?'
                    )
                stack_active = play_g1
                # Record for introspection / parity with co-location key prefix.
                registry.register(
                    (STACK_ACTIVE_PREFIX, pair, w), stack_active
                )

                # 3. Same-field constraint.
                field_to_vars = defaultdict(list)
                for var, key in var_keys:
                    field_to_vars[key[9]].append(var)

                field_inds = []
                for fname, vars_at_field in field_to_vars.items():
                    fi = registry.get_or_create_bool(
                        (STACK_FIELD_USED_PREFIX, pair, w, fname),
                        vars_at_field,
                        f'cvc_stack_fld_{pair[0]}_{pair[1]}_{w}_{fname}',
                    )
                    field_inds.append(fi)

                if len(field_inds) > 1:
                    # When stack_active == 1, at most one field may be used.
                    # (sum == 1 would force EXACTLY one used; we use <= 1
                    # because some grades may have no game this week — the
                    # play-count constraint upstream handles "must play".)
                    model.Add(sum(field_inds) <= 1).OnlyEnforceIf(stack_active)
                    n += 1

                # 4. Contiguous-slots constraint.
                slot_to_vars = defaultdict(list)
                for var, key in var_keys:
                    slot_to_vars[key[4]].append(var)

                if len(slot_to_vars) >= 3:
                    slot_inds = {}
                    for ds, vars_at_slot in slot_to_vars.items():
                        slot_inds[ds] = registry.get_or_create_bool(
                            (STACK_SLOT_USED_PREFIX, pair, w, ds),
                            vars_at_slot,
                            f'cvc_stack_slot_{pair[0]}_{pair[1]}_{w}_{ds}',
                        )

                    sorted_slots = sorted(slot_inds.keys())
                    for i in range(1, len(sorted_slots) - 1):
                        ps = sorted_slots[i - 1]
                        cs = sorted_slots[i]
                        ns = sorted_slots[i + 1]
                        # Empty middle slot + stack active => prior+next <= 1.
                        # Encoded as: when stack_active=1 AND mid=0,
                        # prior+next <= 1. Combine via reification — use a
                        # gate var equivalent to `stack_active AND mid.Not()`.
                        gate = model.NewBoolVar(
                            f'cvc_stack_gap_gate_{pair[0]}_{pair[1]}_{w}_{cs}'
                        )
                        model.AddBoolAnd(
                            [stack_active, slot_inds[cs].Not()]
                        ).OnlyEnforceIf(gate)
                        model.AddBoolOr(
                            [stack_active.Not(), slot_inds[cs]]
                        ).OnlyEnforceIf(gate.Not())
                        model.Add(
                            slot_inds[ps] + slot_inds[ns] <= 1
                        ).OnlyEnforceIf(gate)
                        n += 1

        return n


__all__ = ['ClubVsClubStackedCoLocation']
