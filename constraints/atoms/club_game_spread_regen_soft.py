"""ClubGameSpread regen-soft atom (spec-027).

A REGEN-SOFT atom is the SOFT analogue of a HARD constraint: instead of a hard
clause forbidding a pattern, it emits penalty/indicator variables equal to the
violation amount, leaving the model feasible for ANY assignment of ``X``.

This atom mirrors, as a standalone, the soft pressure that legacy
``UnifiedConstraintEngine._club_game_spread_soft`` appends to the
``ClubGameSpread`` penalty bucket (spec-024). It feeds its own bucket,
``regen_club_game_spread``, with TWO kinds of penalty term:

1. **Per-field residual interior holes.** Group a club's games by
   ``(club, week, day, field)``. Over the sorted ``day_slot`` values used on that
   field, an interior position ``i`` is a *hole* iff a used slot exists strictly
   before it AND a used slot exists strictly after it AND the slot itself is
   empty. Each such hole indicator is appended as one penalty unit. (The legacy
   HARD method also caps holes per field — this atom does NOT; it only penalises,
   so the model stays feasible for any ``X``.)

2. **Off-primary-field games.** Group a club's games by ``(club, week, day)``.
   ``off_primary = total_games_that_day - max_games_on_a_single_field`` — the
   number of the club's games NOT on its most-used field that day, ``0`` iff all
   that day's games sit on one field. One ``off_primary`` IntVar per
   ``(club, week, day)`` (only when the club uses >= 2 fields that day), appended
   as a penalty term. One unit of penalty per off-primary game.

Both terms share the single ``regen_club_game_spread`` bucket. The hole
indicators and the off_primary IntVars are STRUCTURAL channeling definitions
(``AddMaxEquality`` / ``AddBoolAnd`` / ``AddBoolOr`` / an ``== `` linear
definition); none of them FORBID any pattern, so the model is feasible for any
``X``.
"""
from collections import defaultdict
from typing import Dict

from constraints.atoms._contiguity import slot_used_indicators
from constraints.atoms.base import Atom, get_team_club_map

# Default penalty weight when `penalty_weights['regen_club_game_spread']` unset.
REGEN_CLUB_GAME_SPREAD_DEFAULT_WEIGHT = 20000


class ClubGameSpreadRegenSoft(Atom):
    """Soft analogue of ClubGameSpread: penalise per-field holes + off-primary
    games without forbidding any assignment (regen-soft, spec-027)."""

    canonical_name = 'ClubGameSpreadRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_club_game_spread', REGEN_CLUB_GAME_SPREAD_DEFAULT_WEIGHT
        )
        if weight == 0:
            return 0

        locked_weeks = set(data.get('locked_weeks', set()))
        club_map = get_team_club_map(data)

        # Group X by (club, week, day, field) -> {day_slot: [vars]}, mirroring
        # UnifiedConstraintEngine.by_club_week_day_field_slot.
        by_field_slot: Dict[tuple, Dict[int, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key, var in X.items():
            if len(key) < 11:
                continue
            day = key[3]
            if not day:  # dummy / no-day
                continue
            week = key[6]
            if locked_weeks and week in locked_weeks:
                continue
            t1, t2 = key[0], key[1]
            day_slot = key[4]
            field_name = key[9]
            t1_club = club_map.get(t1)
            t2_club = club_map.get(t2)
            if t1_club:
                by_field_slot[(t1_club, week, day, field_name)][day_slot].append(var)
            if t2_club and t2_club != t1_club:
                by_field_slot[(t2_club, week, day, field_name)][day_slot].append(var)

        if not by_field_slot:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_club_game_spread', {'weight': weight, 'penalties': []}
        )

        n = 0

        # (1) Per-field residual interior holes. Build the slot_used channel and
        # the per-slot hole indicator exactly as the legacy hard method, but only
        # append the hole BoolVars as penalty terms (no hole cap).
        for (club, week, day, field), slots_dict in by_field_slot.items():
            sorted_slots = sorted(slots_dict.keys())
            if len(sorted_slots) < 2:
                continue

            slot_inds = slot_used_indicators(
                registry, slots_dict, 'regen_cgs_slot_used',
                club, week, day, field)

            m = len(sorted_slots)
            # "a used slot exists before / after each position".
            pref = [slot_inds[sorted_slots[0]]] + [None] * (m - 1)
            for i in range(1, m):
                p = model.NewBoolVar(
                    f'regen_cgs_pref_{club}_w{week}_{day}_{field}_{i}')
                model.AddMaxEquality(p, [pref[i - 1], slot_inds[sorted_slots[i]]])
                pref[i] = p
            suf = [None] * (m - 1) + [slot_inds[sorted_slots[m - 1]]]
            for i in range(m - 2, -1, -1):
                s = model.NewBoolVar(
                    f'regen_cgs_suf_{club}_w{week}_{day}_{field}_{i}')
                model.AddMaxEquality(s, [suf[i + 1], slot_inds[sorted_slots[i]]])
                suf[i] = s

            # hole[i] = (used before) AND (used after) AND (this slot empty).
            for i in range(1, m - 1):
                used_before, used_after = pref[i - 1], suf[i + 1]
                cur = slot_inds[sorted_slots[i]]
                h = model.NewBoolVar(
                    f'regen_cgs_hole_{club}_w{week}_{day}_{field}_s{sorted_slots[i]}')
                model.AddBoolAnd(
                    [used_before, used_after, cur.Not()]).OnlyEnforceIf(h)
                model.AddBoolOr(
                    [used_before.Not(), used_after.Not(), cur]).OnlyEnforceIf(h.Not())
                bucket['penalties'].append(h)
                n += 1

        # (2) Off-primary-field penalty per (club, week, day).
        club_day_fields = defaultdict(dict)  # (club,week,day) -> {field: [vars]}
        for (club, week, day, field), slots_dict in by_field_slot.items():
            flat = [v for vs in slots_dict.values() for v in vs]
            if flat:
                club_day_fields[(club, week, day)][field] = flat

        for (club, week, day), fields in club_day_fields.items():
            if len(fields) < 2:
                continue  # single field -> off_primary is structurally 0.
            all_vars = [v for fl in fields.values() for v in fl]
            total = len(all_vars)  # upper bound on games that day
            field_counts = []
            for field, fl in fields.items():
                c = model.NewIntVar(
                    0, len(fl),
                    f'regen_cgs_fcount_{club}_w{week}_{day}_{field}')
                model.Add(c == sum(fl))
                field_counts.append(c)
            max_count = model.NewIntVar(
                0, total, f'regen_cgs_fmax_{club}_w{week}_{day}')
            model.AddMaxEquality(max_count, field_counts)
            off_primary = model.NewIntVar(
                0, total, f'regen_cgs_offprimary_{club}_w{week}_{day}')
            model.Add(off_primary == sum(all_vars) - max_count)
            bucket['penalties'].append(off_primary)
            n += 1

        return n
