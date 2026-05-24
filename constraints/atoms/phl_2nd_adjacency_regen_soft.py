"""PHL/2nd grade same-club adjacency — SOFT regen analogue (spec-027).

This is the SOFT analogue of the HARD atom
``constraints/atoms/phl_2nd_adjacency.py`` (``PHLAnd2ndAdjacency``).

The hard atom enforces, per ``(club, week, day)`` where a club fields BOTH a
PHL and a 2nd-grade game:

- **Same venue**  -> the two games must be on the **same field** in **adjacent
  day_slots** (back-to-back, no gap);
- **Different venue** -> the two **start times** must differ by at least
  ``phl_2nd_cross_venue_min_minutes`` (default 180 = 3 h).

Any other arrangement is forbidden by the hard atom via ``p + q <= 1`` for
each violating cross-grade pair.

This SOFT analogue NEVER forbids any X assignment. Instead, for each violating
cross-grade pair ``(p, q)`` — exactly the pairs the hard atom would forbid — it
emits a penalty BoolVar that is 1 exactly when BOTH games are scheduled
(``p AND q``). The objective subtracts these penalties, so the model stays
feasible for ANY X.

Penalty semantics: one penalty unit per (club, week, day) violating weekend.
Because ``NoDoubleBookingTeams`` pins each team to <= 1 game per day, at most
one PHL var and one 2nd var in a bucket can be 1 simultaneously, so at most one
violating pair fires per bucket — i.e. one unit per genuinely violating
weekend, mirroring the hard atom's per-pair structure.

The "broken" condition is computed with the SAME same-venue / cross-venue /
time-window logic as the hard sibling, so the soft penalty fires on exactly the
arrangements the hard atom would reject.

This atom builds every indicator it needs directly from X — it does NOT rely on
any helper var produced by the hard sibling (in regen the hard sibling is not
applied).
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from constraints.atoms.base import Atom, get_team_club_map

# Default penalty weight when `PENALTY_WEIGHTS['regen_phl_2nd_adjacency']`
# is unset. Large — this mirrors a severity-1 CRITICAL hard rule, so a
# violation must dominate ordinary soft preferences.
REGEN_PHL_2ND_ADJACENCY_DEFAULT_WEIGHT = 100000


def _time_to_minutes(time_str: str) -> int:
    """Minutes since midnight for an ``HH:MM`` string."""
    hh, mm = time_str.split(':')
    return int(hh) * 60 + int(mm)


class PHLAnd2ndAdjacencyRegenSoft(Atom):
    """SOFT regen analogue of ``PHLAnd2ndAdjacency``. Emits a penalty var
    (= ``p AND q``) for each violating same-club PHL/2nd cross-grade pair;
    never forbids any assignment.
    """

    canonical_name = 'PHLAnd2ndAdjacencyRegenSoft'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_phl_2nd_adjacency', REGEN_PHL_2ND_ADJACENCY_DEFAULT_WEIGHT
        )
        if weight == 0:
            return 0

        team_club = get_team_club_map(data)
        locked_weeks = set(data.get('locked_weeks', set()))
        cross_venue_min = data.get('constraint_defaults', {}).get(
            'phl_2nd_cross_venue_min_minutes', 180
        )

        # (club, week, day) -> {'PHL': [(field, slot, loc, minutes, var)],
        #                       '2nd': [...]}
        buckets: Dict[Tuple[str, int, str], Dict[str, List]] = defaultdict(
            lambda: {'PHL': [], '2nd': []}
        )

        for key, var in X.items():
            if len(key) < 11:
                continue  # dummy slot
            if not key[3]:
                continue  # no day
            t1, t2, grade, day, day_slot, time, week, _date, _round, fname, floc = key
            if grade not in ('PHL', '2nd'):
                continue
            if locked_weeks and week in locked_weeks:
                continue
            if not time:
                continue
            minutes = _time_to_minutes(time)
            entry = (fname, day_slot, floc, minutes, var)
            for team in (t1, t2):
                club = team_club.get(team)
                if club is None:
                    continue
                buckets[(club, week, day)][grade].append(entry)

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_phl_2nd_adjacency', {'weight': weight, 'penalties': []}
        )

        n = 0
        for (club, week, day), by_grade in buckets.items():
            phl_entries = by_grade['PHL']
            second_entries = by_grade['2nd']
            if not phl_entries or not second_entries:
                continue  # club fields only one grade that day -> nothing
            for (p_field, p_slot, p_loc, p_min, p_var) in phl_entries:
                for (q_field, q_slot, q_loc, q_min, q_var) in second_entries:
                    if p_loc == q_loc:
                        allowed = (p_field == q_field
                                   and abs(p_slot - q_slot) == 1)
                    else:
                        allowed = abs(p_min - q_min) >= cross_venue_min
                    if allowed:
                        continue
                    # SOFT: penalty v = (p_var AND q_var). Fires (=1) exactly
                    # when both games of a violating pair are scheduled.
                    # Standard 0/1 "A AND B" linearization:
                    #   v >= p + q - 1 ; v <= p ; v <= q.
                    v = model.NewBoolVar(
                        f'regen_phl2nd_viol_{club}_w{week}_{day}_{n}'
                    )
                    model.Add(v >= p_var + q_var - 1)
                    model.Add(v <= p_var)
                    model.Add(v <= q_var)
                    bucket['penalties'].append(v)
                    n += 1
        return n
