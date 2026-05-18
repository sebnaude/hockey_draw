"""AwayClubHomeWeekendsCount — pin away-club home weekend totals (spec-004).

For each club whose home venue is NOT Broadmeadow (Maitland, Gosford, future
expansions), force three counts:

  1. sum(friday_home_indicators) == phl_forced_friday_count(data, club)
       — each FORCED PHL Friday consumes a home Friday slot at the club's
         venue, so the number of weeks with a Friday home game equals the
         FORCED-Friday count exactly.

  2. sum(sunday_home_indicators) == away_club_required_sundays(data, club)
       — the FORCED-aware Sunday count: max(PHL - FORCED_Fridays, max_other).

  3. sum(all_home_indicators)    == away_club_total_weekends(data, club)
       — total distinct weeks with a home game (Friday OR Sunday).
         Equal to max(PHL_required, max_other_grade_games), the unadjusted max.
         In the equality case (PHL >= max_other) this implies Friday/Sunday
         home weekends are in DISJOINT weeks; in the other-grade-dominant case
         (PHL < max_other), FORCED Fridays are absorbed into the same total.

This atom replaces the home-weekend logic that was historically buried inside
`FiftyFiftyHomeandAway` + the legacy `MaxMaitlandHomeWeekends` heuristic. Both
of those are now obsolete for production (see CONSTRAINT_INVENTORY.md §5 and
the `FiftyFiftyHomeandAway` row).

## Indicator construction

For each (week, day) of (Friday, Sunday), collect the X-vars where the
field_location equals the club's home venue AND the variable involves the club.
The indicator is `OR(vars)` — represented in CP-SAT via NewBoolVar + AddMaxEquality.

When a club has no candidate vars for a (week, day) bucket, the indicator is
fixed to 0 (omitted from the sum).

## Locked-week handling

Locked weeks contribute neither to the sum target nor to the indicators — the
target counts are reduced by the number of locked weeks that ALREADY have a
home game for the club (in the locked solution). This mirrors the pattern used
by other atoms; see `_count_locked_home_weekends`.
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


class AwayClubHomeWeekendsCount(Atom):
    """Pin Friday / Sunday / total home weekend counts for each away-based club."""

    canonical_name = 'AwayClubHomeWeekendsCount'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        home_field_map: Dict[str, str] = data.get('home_field_map', {}) or {}
        if not home_field_map:
            return 0

        locked_weeks: Set[int] = set(data.get('locked_weeks', set()) or set())
        team_to_club = {team.name: team.club.name for team in data.get('teams', [])}

        constraints_added = 0

        for club, home_venue in home_field_map.items():
            # Skip clubs whose default home IS Broadmeadow (no "away" rule
            # applies). The map convention: only non-default-home clubs appear.
            # We rely on the season config to omit Broadmeadow-default clubs.
            if not home_venue:
                continue

            # Collect candidate X vars per (week, day) at the club's venue.
            # Only vars involving the club itself contribute.
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

            # Compute the three targets via the shared helper.
            forced_fridays = phl_forced_friday_count(data, club)
            sundays_required = away_club_required_sundays(data, club)
            total_weekends = away_club_total_weekends(data, club)

            # Reduce targets by locked contribution so the sum constraint
            # applies only to the unlocked horizon.
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

            # All-weekend indicators: OR(friday[w], sunday[w]) per week.
            all_weeks = set(friday_vars_by_week.keys()) | set(sunday_vars_by_week.keys())
            all_indicators: List = []
            for week in sorted(all_weeks):
                fri_vars = friday_vars_by_week.get(week, [])
                sun_vars = sunday_vars_by_week.get(week, [])
                week_vars = list(fri_vars) + list(sun_vars)
                if not week_vars:
                    continue
                indicator = model.NewBoolVar(f'{club}_any_home_w{week}')
                model.AddMaxEquality(indicator, week_vars)
                all_indicators.append(indicator)

            # Add the three sum constraints — only if there are indicators to
            # constrain (otherwise the targets must be 0).
            if friday_indicators:
                model.Add(sum(friday_indicators) == friday_target)
                constraints_added += 1
            elif friday_target > 0:
                # No Friday vars exist but target > 0 — infeasible by
                # construction; surface immediately. (Fixture sanity check.)
                raise ValueError(
                    f'AwayClubHomeWeekendsCount: club {club!r} requires '
                    f'{friday_target} home Fridays but no candidate Friday '
                    f'variables exist at {home_venue!r}.'
                )

            if sunday_indicators:
                model.Add(sum(sunday_indicators) == sunday_target)
                constraints_added += 1
            elif sunday_target > 0:
                raise ValueError(
                    f'AwayClubHomeWeekendsCount: club {club!r} requires '
                    f'{sunday_target} home Sundays but no candidate Sunday '
                    f'variables exist at {home_venue!r}.'
                )

            if all_indicators:
                model.Add(sum(all_indicators) == total_target)
                constraints_added += 1
            elif total_target > 0:
                raise ValueError(
                    f'AwayClubHomeWeekendsCount: club {club!r} requires '
                    f'{total_target} home weekends but no candidate home '
                    f'variables exist at {home_venue!r}.'
                )

        return constraints_added


def _build_week_indicators(
    model, vars_by_week: Dict[int, List], *, label: str,
) -> List:
    """Return one OR-indicator per non-empty week."""
    indicators = []
    for week in sorted(vars_by_week.keys()):
        week_vars = vars_by_week[week]
        if not week_vars:
            continue
        if len(week_vars) == 1:
            indicators.append(week_vars[0])
            continue
        indicator = model.NewBoolVar(f'{label}_w{week}')
        model.AddMaxEquality(indicator, week_vars)
        indicators.append(indicator)
    return indicators
