"""AwayClubHomeWeekendsCountRegenSoft atom (spec-027 regen-soft).

SOFT analogue of the ``AwayClubHomeWeekendsCount`` hard constraint (spec-004).

The hard atom pins, for each away-based club (home venue != Broadmeadow), three
home-weekend counts to computed targets:

  1. sum(friday_home_indicators) == phl_forced_friday_count(data, club)
  2. sum(sunday_home_indicators) == away_club_required_sundays(data, club)
  3. sum(all_home_indicators)    == away_club_total_weekends(data, club)

This SOFT analogue keeps the SAME sum constructions and the SAME target
computations, but replaces each hard ``model.Add(sum == target)`` with an
absolute-deviation penalty IntVar:

    dev = NewIntVar(0, BIG, ...)
    model.Add(dev >= actual_sum - target)
    model.Add(dev >= target - actual_sum)

Because the objective minimises ``dev``, it settles to ``|actual_sum - target|``.
One penalty unit corresponds to one weekend the actual count is off-target.

The only ``model.Add`` calls here are STRUCTURAL: the OR-indicator definitions
(``AddMaxEquality``) and the two deviation-bounding constraints (which define
``dev`` without forbidding any X assignment). The model stays feasible for ANY X.

**Config:**

    penalty_weights['regen_away_club_home_weekends_count']  (default: 90000)

Set to 0 to disable the atom entirely (returns 0 immediately).

**Skip conditions (mirror the hard atom):**

- Empty ``home_field_map`` -> no-op (return 0).
- Dummy keys: ``len(key) < 11``.
- No-day keys: ``not key[3]``.
- Locked weeks: contribute to neither indicators nor target; targets reduced
  by locked home-weekend counts (same as the hard atom).
- A (friday/sunday/all) bucket with no indicators emits no penalty; unlike the
  hard atom we do NOT raise when ``target > 0`` and no candidate vars exist —
  a soft atom must never raise / make the model infeasible. The off-target gap
  simply cannot be expressed (there are no vars to penalise against), so the
  bucket is skipped.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from constraints.atoms.base import Atom
from constraints.atoms._phl_forced_friday_helper import (
    away_club_required_sundays,
    away_club_total_weekends,
    phl_forced_friday_count,
)

# Default penalty weight. CRITICAL-tier intent (the hard analogue is a hard
# count pin), so a high weight — above the spacing/alignment soft weights — so
# off-target home-weekend counts are strongly discouraged.
REGEN_AWAY_CLUB_HOME_WEEKENDS_COUNT_DEFAULT_WEIGHT = 90000


class AwayClubHomeWeekendsCountRegenSoft(Atom):
    """SOFT per-away-club home-weekend count penalty (spec-027).

    Mirrors ``AwayClubHomeWeekendsCount`` (spec-004) as an absolute-deviation
    penalty. Adds no hard forbidding ``model.Add`` — only structural indicator
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

            # Collect candidate X vars per (week, day) at the club's venue.
            friday_vars_by_week: Dict[int, List] = defaultdict(list)
            sunday_vars_by_week: Dict[int, List] = defaultdict(list)
            locked_friday_weeks: Set[int] = set()
            locked_sunday_weeks: Set[int] = set()

            for key, var in X.items():
                if len(key) < 11:
                    continue  # dummy
                if not key[3]:
                    continue  # no day
                if key[10] != home_venue:
                    continue  # different venue
                t1, t2 = key[0], key[1]
                club1 = team_to_club.get(t1)
                club2 = team_to_club.get(t2)
                if club != club1 and club != club2:
                    continue
                week = key[6]
                day = key[3]
                if week in locked_weeks:
                    if day == 'Friday':
                        locked_friday_weeks.add(week)
                    elif day == 'Sunday':
                        locked_sunday_weeks.add(week)
                    continue
                if day == 'Friday':
                    friday_vars_by_week[week].append(var)
                elif day == 'Sunday':
                    sunday_vars_by_week[week].append(var)

            # Compute the three targets via the shared helper (same as hard).
            forced_fridays = phl_forced_friday_count(data, club)
            sundays_required = away_club_required_sundays(data, club)
            total_weekends = away_club_total_weekends(data, club)

            # Reduce targets by locked contribution.
            friday_target = max(0, forced_fridays - len(locked_friday_weeks))
            sunday_target = max(0, sundays_required - len(locked_sunday_weeks))
            total_locked_weeks = locked_friday_weeks | locked_sunday_weeks
            total_target = max(0, total_weekends - len(total_locked_weeks))

            # Build OR-indicators per (week, day) and per (week, any-day).
            friday_indicators = _build_week_indicators(
                model, friday_vars_by_week, label=f'{club}_fri_home',
            )
            sunday_indicators = _build_week_indicators(
                model, sunday_vars_by_week, label=f'{club}_sun_home',
            )

            all_weeks = set(friday_vars_by_week.keys()) | set(sunday_vars_by_week.keys())
            all_indicators: List = []
            for week in sorted(all_weeks):
                fri_vars = friday_vars_by_week.get(week, [])
                sun_vars = sunday_vars_by_week.get(week, [])
                week_vars = list(fri_vars) + list(sun_vars)
                if not week_vars:
                    continue
                indicator = model.NewBoolVar(f'regen_{club}_any_home_w{week}')
                model.AddMaxEquality(indicator, week_vars)
                all_indicators.append(indicator)

            # SOFT: one absolute-deviation penalty per count (only when there
            # are indicators to express the deviation against).
            if friday_indicators:
                _add_dev_penalty(
                    model, bucket, friday_indicators, friday_target,
                    f'regen_{club}_fri_dev',
                )
                n += 1
            if sunday_indicators:
                _add_dev_penalty(
                    model, bucket, sunday_indicators, sunday_target,
                    f'regen_{club}_sun_dev',
                )
                n += 1
            if all_indicators:
                _add_dev_penalty(
                    model, bucket, all_indicators, total_target,
                    f'regen_{club}_all_dev',
                )
                n += 1

        return n


def _add_dev_penalty(model, bucket, indicators: List, target: int, label: str):
    """Append an absolute-deviation penalty IntVar = |sum(indicators) - target|.

    Defined by two bounding constraints (neither forbids any X assignment):
        dev >= sum(indicators) - target
        dev >= target - sum(indicators)
    The minimising objective drives dev to exactly |sum - target|.
    """
    # Upper bound: deviation can never exceed max(num_indicators, target).
    big = max(len(indicators), target)
    dev = model.NewIntVar(0, big, label)
    total = sum(indicators)
    model.Add(dev >= total - target)
    model.Add(dev >= target - total)
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
