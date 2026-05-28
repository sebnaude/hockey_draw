"""AwayClubHomeWeekendsCountRegenSoft atom (spec-027 regen-soft; redesigned in spec-037).

SOFT analogue of the ``AwayClubHomeWeekendsCount`` hard atom.

The hard atom enforces, for each away-based club (home venue != Broadmeadow):

    min_sundays_home(data, club) <= sum(sunday_home_indicators) <= max_sundays_home(data, club)

This SOFT analogue keeps the SAME indicator construction and the SAME derived
bounds, but replaces the hard bounds with a single deviation IntVar:

    dev = max(0, min_sundays - sum, sum - max_sundays)

so the penalty mirrors ``max(0, min - sum) + max(0, sum - max)`` exactly
(under-floor and over-ceiling are mutually exclusive when ``min <= max``, so
the per-direction maxes sum to the per-direction max-of-three). The minimising
objective drives ``dev`` to 0 when the Sunday-home count is inside the range,
or to the deviation magnitude when it isn't.

The only ``model.Add`` calls here are STRUCTURAL: the OR-indicator definitions
(``AddMaxEquality``) and the three deviation-bounding constraints (none of
which forbid any X assignment). The model stays feasible for ANY X.

**Config:**

    penalty_weights['regen_away_club_home_weekends_count']  (default: 90000)

Set to 0 to disable the atom entirely (returns 0 immediately).

The configured weight is normalised by ``_build_normalized_penalty`` (in
``main_staged.py``) by dividing the weight by the number of deviation IntVars
the atom registers (one per away-based club).

**Skip conditions (mirror the hard atom):**

- Empty ``home_field_map`` -> no-op (return 0).
- Dummy keys: ``len(key) < 11``.
- No-day keys: ``not key[3]``.
- Locked weeks: contribute to neither indicators nor target; bounds reduced
  by the locked Sunday-home contribution.
- A club with no Sunday-home indicators emits no penalty (no var to penalise
  against). A soft atom must never raise / make the model infeasible.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from constraints.atoms.base import Atom
from constraints.atoms._phl_forced_friday_helper import (
    away_club_max_sundays_home,
    away_club_min_sundays_home,
)

# Default penalty weight. CRITICAL-tier intent (the hard analogue pins a count
# range), so a high weight — above the spacing/alignment soft weights — so
# off-target home-weekend counts are strongly discouraged.
REGEN_AWAY_CLUB_HOME_WEEKENDS_COUNT_DEFAULT_WEIGHT = 90000


class AwayClubHomeWeekendsCountRegenSoft(Atom):
    """SOFT per-away-club Sunday-home range penalty (spec-027, spec-037).

    Mirrors ``AwayClubHomeWeekendsCount`` (spec-004 redesigned) as a single
    deviation IntVar per club:

        dev = max(0, min_sundays - sum, sum - max_sundays)

    Adds no hard forbidding ``model.Add`` — only structural indicator
    definitions and deviation-bounding constraints — so the model is feasible
    for any X.
    """

    canonical_name = 'AwayClubHomeWeekendsCountRegenSoft'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_away_club_home_weekends_count',
            REGEN_AWAY_CLUB_HOME_WEEKENDS_COUNT_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        home_field_map: Dict[str, str] = data.get('home_field_map', {}) or {}
        if not home_field_map:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_away_club_home_weekends_count',
            {'weight': weight, 'penalties': []},
        )

        locked_weeks: Set[int] = set(data.get('locked_weeks', set()) or set())
        team_to_club = {team.name: team.club.name for team in data.get('teams', [])}

        n = 0

        for club, home_venue in home_field_map.items():
            if not home_venue:
                continue

            # Collect candidate Sunday X vars per week at the club's venue.
            sunday_vars_by_week: Dict[int, List] = defaultdict(list)
            locked_sunday_weeks: Set[int] = set()

            for key, var in X.items():
                if len(key) < 11:
                    continue  # dummy
                if not key[3]:
                    continue  # no day
                if key[3] != 'Sunday':
                    continue
                if key[10] != home_venue:
                    continue
                t1, t2 = key[0], key[1]
                club1 = team_to_club.get(t1)
                club2 = team_to_club.get(t2)
                if club != club1 and club != club2:
                    continue
                week = key[6]
                if week in locked_weeks:
                    locked_sunday_weeks.add(week)
                    continue
                sunday_vars_by_week[week].append(var)

            min_sundays = away_club_min_sundays_home(data, club)
            max_sundays = away_club_max_sundays_home(data, club)
            # Soft atom: never raise on inverted bounds — penalise harmlessly.
            if min_sundays > max_sundays:
                # Make the deviation always >= (min - max) so the optimiser
                # still gets a signal that the config is inconsistent.
                min_sundays, max_sundays = max_sundays, min_sundays

            locked_count = len(locked_sunday_weeks)
            effective_min = max(0, min_sundays - locked_count)
            effective_max = max(0, max_sundays - locked_count)

            sunday_indicators = _build_week_indicators(
                model, sunday_vars_by_week, label=f'regen_{club}_sun_home',
            )

            if not sunday_indicators:
                # No vars to penalise against — soft atom skips silently.
                continue

            _add_dev_penalty(
                model, bucket, sunday_indicators,
                lower=effective_min, upper=effective_max,
                label=f'regen_{club}_sun_dev',
            )
            n += 1

        return n


def _add_dev_penalty(
    model, bucket, indicators: List, *, lower: int, upper: int, label: str,
):
    """Append a single deviation IntVar = max(0, lower - sum, sum - upper).

    Defined by three bounding constraints (none forbids any X assignment):
        dev >= 0
        dev >= lower - sum
        dev >= sum - upper

    Minimising drives ``dev`` to ``max(0, lower - sum, sum - upper)`` exactly.
    Because under-floor and over-ceiling are mutually exclusive when
    ``lower <= upper``, ``dev == max(0, lower-sum) + max(0, sum-upper)``
    automatically.
    """
    big = max(len(indicators), lower, upper, 1)
    dev = model.NewIntVar(0, big, label)
    total = sum(indicators)
    model.Add(dev >= lower - total)
    model.Add(dev >= total - upper)
    bucket['penalties'].append(dev)
    return dev


def _build_week_indicators(
    model, vars_by_week: Dict[int, List], *, label: str,
) -> List:
    """Return one OR-indicator per non-empty week (mirrors the hard atom)."""
    indicators = []
    for week in sorted(vars_by_week.keys()):
        week_vars = vars_by_week[week]
        if not week_vars:
            continue
        if len(week_vars) == 1:
            indicators.append(week_vars[0])
            continue
        indicator = model.NewBoolVar(f'regen_{label}_w{week}')
        model.AddMaxEquality(indicator, week_vars)
        indicators.append(indicator)
    return indicators
