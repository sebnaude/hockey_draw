"""AwayClubHomeWeekendsCount — bound the Sunday-home weekend count for each
away-based club to a derived two-sided range (spec-004, redesigned in spec-037).

For each club whose home venue is NOT Broadmeadow (Maitland, Gosford, future
expansions), enforce:

    min_sundays_home(data, club) <= sum(sunday_home_indicators) <= max_sundays_home(data, club)

where the bounds come from the per-grade home-game demands the club fields
(see ``constraints/atoms/_phl_forced_friday_helper.py``):

  * floor = max home-game count across NON-PHL grades the club fields (those
    grades have no Friday alternative; every home game MUST land on a Sunday).
    Zero when the club fields no non-PHL grade.
  * ceiling = max home-game count across ALL grades incl. PHL. PHL can lend
    Sundays to forced Fridays without pushing the total above its raw home
    count.

Forced Fridays are NOT subtracted by this atom — ``FORCED_GAMES`` config
already enforces the Friday count via partial-key entries (e.g. ``{count: N,
constraint: 'equal', day: 'Friday', field_location: ...}``). Two solver
mechanisms encoding the same fact were a forward-only smell; this atom now
covers Sunday range only and lets PHL-Friday counts fall out of the FORCED
config.

## Indicator construction

For each week, OR together the X-vars at the club's home venue on Sunday that
involve the club. The indicator is `OR(vars)` via NewBoolVar +
AddMaxEquality. If a week has no candidate Sunday-home vars, no indicator is
created and the week contributes 0 to the sum.

## Locked-week handling

Locked weeks contribute neither to the sum nor to the indicators — the bounds
are reduced by the number of locked weeks that ALREADY have a Sunday-home
game for the club (in the locked solution), mirroring the pattern used by
other atoms.

## Feasibility-violating raise

If ``min_sundays_home > max_sundays_home`` for a club, the atom raises
``ValueError`` — this is mathematically impossible when ``num_rounds`` is
consistent, but a config sanity error could trip it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from constraints.atoms.base import Atom
from constraints.atoms._phl_forced_friday_helper import (
    away_club_max_sundays_home,
    away_club_min_sundays_home,
)


class AwayClubHomeWeekendsCount(Atom):
    """Bound the Sunday-home-weekend count for each away-based club."""

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
            if not home_venue:
                continue

            # Collect candidate Sunday X vars per week at the club's venue.
            # Only vars involving the club itself contribute.
            sunday_vars_by_week: Dict[int, List] = defaultdict(list)
            locked_sunday_weeks: Set[int] = set()

            for key, var in X.items():
                if len(key) < 11:
                    continue  # dummy
                if not key[3]:
                    continue  # no day
                if key[3] != 'Sunday':
                    continue  # only Sunday matters for the derived range
                if key[10] != home_venue:
                    continue  # different venue
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

            # Derived range bounds from per-grade home-game demand.
            min_sundays = away_club_min_sundays_home(data, club)
            max_sundays = away_club_max_sundays_home(data, club)

            if min_sundays > max_sundays:
                raise ValueError(
                    f'AwayClubHomeWeekendsCount: club {club!r} has '
                    f'min_sundays_home={min_sundays} > '
                    f'max_sundays_home={max_sundays} — config inconsistency.'
                )

            # Reduce bounds by locked Sunday-home contribution so the sum
            # applies only to the unlocked horizon. Floor at 0.
            locked_count = len(locked_sunday_weeks)
            effective_min = max(0, min_sundays - locked_count)
            effective_max = max(0, max_sundays - locked_count)

            # Build one OR-indicator per non-empty week.
            sunday_indicators = _build_week_indicators(
                model, sunday_vars_by_week, label=f'{club}_sun_home',
            )

            # Lower bound: only emit when strictly positive (vacuous when 0).
            if effective_min > 0:
                if not sunday_indicators:
                    # No Sunday vars exist but floor > 0 — infeasible by
                    # construction; surface immediately (fixture sanity).
                    raise ValueError(
                        f'AwayClubHomeWeekendsCount: club {club!r} requires '
                        f'>= {effective_min} Sunday home weekends but no '
                        f'candidate Sunday variables exist at {home_venue!r}.'
                    )
                model.Add(sum(sunday_indicators) >= effective_min)
                constraints_added += 1

            # Upper bound: always emit when there are indicators (even when
            # effective_max == 0 — that pins them all to 0, the correct semantic).
            if sunday_indicators:
                model.Add(sum(sunday_indicators) <= effective_max)
                constraints_added += 1

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
