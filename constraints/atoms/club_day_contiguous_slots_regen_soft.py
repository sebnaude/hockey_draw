"""Club-day contiguous-slots SOFT analogue (spec-027 regen-soft atom).

Mirrors ClubDayContiguousSlots but replaces every hard "no internal gap"
implication with a PENALTY BoolVar that is 1 exactly when an internal gap
exists.  The model stays feasible for any assignment of X; the solver just
pays a configurable weight per gap.

Gap definition (matches the hard atom in ``_contiguity.enforce_no_gaps``):
  For each consecutive triple of slot indicators (prev, mid, next) in sorted
  order, a gap at *mid* is defined as:

      prev_used == 1  AND  next_used == 1  AND  mid_used == 0

  Linearised with three constraints so ``v == 1`` iff the gap pattern holds:
      v >= prev_used + next_used - 1 - mid_used
      v <= prev_used
      v <= next_used
      v <= 1 - mid_used

  (The fourth clause ``v >= 0`` is implicit for BoolVars.)

  One penalty var per (club, middle-slot-triple) per club-day entry.  The
  total raw-penalty sum (pre-weighting) equals the number of internal gaps.
"""
from collections import defaultdict

from constraints.atoms.base import Atom
from constraints.atoms._contiguity import slot_used_indicators
from constraints.atoms._club_day_shared import (
    club_day_game_keys, parse_club_day_entries,
)

# Default penalty per internal gap when the key is absent from penalty_weights.
REGEN_CLUB_DAY_CONTIGUOUS_SLOTS_DEFAULT_WEIGHT = 25_000


class ClubDayContiguousSlotsRegenSoft(Atom):
    """SOFT version of ClubDayContiguousSlots.

    Emits one penalty BoolVar per internal gap in a club's club-day slot
    usage.  No hard Add() that could make the model infeasible.
    """

    canonical_name = 'ClubDayContiguousSlotsRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_club_day_contiguous_slots',
            REGEN_CLUB_DAY_CONTIGUOUS_SLOTS_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_club_day_contiguous_slots',
            {'weight': weight, 'penalties': []},
        )

        club_team_lookup = {}
        for team in data['teams']:
            club_team_lookup.setdefault(team.club.name, []).append(team.name)

        n = 0
        for club_name, date_str, _opponent in parse_club_day_entries(data):
            host_team_names = club_team_lookup.get(club_name, [])
            game_keys = club_day_game_keys(X, host_team_names, date_str)
            if not game_keys:
                continue

            slot_vars = defaultdict(list)
            for key in game_keys:
                slot_vars[key[4]].append(X[key])
            if len(slot_vars) < 3:
                # Need at least 3 distinct slots for any middle slot to exist.
                continue

            # Build channeled slot-used indicators (shared pool, same cache key
            # as the hard atom so both can coexist without duplicating vars).
            slot_inds = slot_used_indicators(
                registry, slot_vars, 'club_day_slot_used', club_name,
            )

            sorted_slots = sorted(slot_inds.keys())
            for i in range(1, len(sorted_slots) - 1):
                ps, cs, ns = sorted_slots[i - 1], sorted_slots[i], sorted_slots[i + 1]
                prev_used = slot_inds[ps]
                mid_used = slot_inds[cs]
                next_used = slot_inds[ns]

                # v == 1  iff  prev_used=1 AND next_used=1 AND mid_used=0.
                v = model.NewBoolVar(
                    f'regen_cd_gap_{club_name}_{date_str}_s{cs}'
                )
                # Lower bound: v must be 1 when the gap pattern holds.
                model.Add(v >= prev_used + next_used - 1 - mid_used)
                # Upper bounds: v can only be 1 when each condition holds.
                model.Add(v <= prev_used)
                model.Add(v <= next_used)
                model.Add(v <= 1 - mid_used)

                bucket['penalties'].append(v)
                n += 1

        return n
