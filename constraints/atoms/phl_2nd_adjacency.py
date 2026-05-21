"""PHL/2nd grade same-club adjacency (HARD, spec-014).

Within a club, that club's PHL team and 2nd-grade team should be watchable
together when they play on the same day. Concretely, per ``(club, week, day)``
where the club fields BOTH a PHL game and a 2nd-grade game:

- **Same venue** -> the two games must be on the **same field** in **adjacent
  day_slots** (back-to-back, no gap).
- **Different venue** -> the two **start times** must differ by at least
  ``phl_2nd_cross_venue_min_minutes`` (default 180 = 3 h). Rationale
  (start-to-start, since that is all the schedule controls): game 1 length +
  warm-down + travel between grounds + warm-up before game 2.

If the club fields only one (or neither) of the two grades that day, the atom
adds nothing for that club/day.

This replaces the legacy ``UnifiedConstraintEngine._phl_adjacency_hard``, which
*forbade* two bad patterns inside a symmetric +/-180-min window but never
*forced* adjacency. The two rules here are genuinely distinct: the same-venue
rule is about slot adjacency (no minute threshold); the cross-venue rule is a
real start-time gap in minutes (different fields run different clocks, so
comparing day_slot indices across venues would be wrong).

Encoding (see spec-014 "Chosen encoding"): zero new decision variables. For
each candidate (PHL var, 2nd var) pair in a bucket, emit ``p + q <= 1`` for
exactly the pairs that violate the rule. Each club fields one PHL team and one
2nd team, and ``NoDoubleBookingTeams`` pins each to <= 1 game per day, so the
bucket is small; forbidding the infeasible pairs forces the relationship
whenever both grades play.

Severity 1 (CRITICAL): a publishable-draw operational rule.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from constraints.atoms.base import Atom, get_team_club_map


def _time_to_minutes(time_str: str) -> int:
    """Minutes since midnight for an ``HH:MM`` string."""
    hh, mm = time_str.split(':')
    return int(hh) * 60 + int(mm)


class PHLAnd2ndAdjacency(Atom):
    """Per (club, week, day): same-club PHL/2nd back-to-back same venue, or
    >= cross-venue-minimum start-time gap across venues. Severity 1.
    """

    canonical_name = 'PHLAnd2ndAdjacency'
    atom_group = None

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
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
            t1, t2, grade, day, day_slot, time, week, _date, _round, fname, floc = key
            if grade not in ('PHL', '2nd'):
                continue
            if not day:
                continue
            if locked_weeks and week in locked_weeks:
                continue
            if not time:
                continue
            minutes = _time_to_minutes(time)
            entry = (fname, day_slot, floc, minutes, var)
            # A game involves two teams (hence up to two clubs); both clubs
            # field that grade that (week, day). Register against each.
            for team in (t1, t2):
                club = team_club.get(team)
                if club is None:
                    continue
                buckets[(club, week, day)][grade].append(entry)

        n = 0
        for (_club, _week, _day), by_grade in buckets.items():
            phl_entries = by_grade['PHL']
            second_entries = by_grade['2nd']
            if not phl_entries or not second_entries:
                continue  # club fields only one grade that day -> nothing (DoD 2c)
            # PHL and 2nd entries are disjoint sets of vars (grade is part of
            # the key), so every (p, q) is a genuine cross-grade pair.
            for (p_field, p_slot, p_loc, p_min, p_var) in phl_entries:
                for (q_field, q_slot, q_loc, q_min, q_var) in second_entries:
                    if p_loc == q_loc:
                        allowed = (p_field == q_field
                                   and abs(p_slot - q_slot) == 1)
                    else:
                        allowed = abs(p_min - q_min) >= cross_venue_min
                    if not allowed:
                        model.Add(p_var + q_var <= 1)
                        n += 1
        return n
