"""Atomic constraints — one idea per class.

Each atom subclasses `Atom` (constraints/atoms/base.py) and:
- `declare_helpers(registry, data)` declares helper-vars it needs (default: none).
- `apply(model, X, data, registry)` adds CP-SAT constraints, returns count added.

Atoms are instantiated and dispatched by `UnifiedConstraintEngine`. Their
`canonical_name` matches a `ConstraintInfo` entry in `constraints/registry.py`
(with `atom_group` set to the legacy combined-constraint they were split from).
"""
from constraints.atoms.base import Atom
from constraints.atoms.phl_concurrency import PHLConcurrencyAtBroadmeadow
from constraints.atoms.phl_2nd_concurrency import PHLAnd2ndConcurrencyAtBroadmeadow
from constraints.atoms.gosford_friday_rounds import GosfordFridayRoundsForced
from constraints.atoms.phl_round_one_play import PHLRoundOnePlay
from constraints.atoms.preferred_dates import PreferredDates
from constraints.atoms.club_day_participation import ClubDayParticipation
from constraints.atoms.club_day_intra_club_matchup import ClubDayIntraClubMatchup
from constraints.atoms.club_day_opponent_matchup import ClubDayOpponentMatchup
from constraints.atoms.club_day_same_field import ClubDaySameField
from constraints.atoms.club_day_contiguous_slots import ClubDayContiguousSlots


PHL_TIMES_ATOMS = [
    PHLConcurrencyAtBroadmeadow,
    PHLAnd2ndConcurrencyAtBroadmeadow,
    GosfordFridayRoundsForced,
    PHLRoundOnePlay,
    PreferredDates,
]


CLUB_DAY_ATOMS = [
    ClubDayParticipation,
    ClubDayIntraClubMatchup,
    ClubDayOpponentMatchup,
    ClubDaySameField,
    ClubDayContiguousSlots,
]


__all__ = [
    'Atom',
    'PHLConcurrencyAtBroadmeadow',
    'PHLAnd2ndConcurrencyAtBroadmeadow',
    'GosfordFridayRoundsForced',
    'PHLRoundOnePlay',
    'PreferredDates',
    'PHL_TIMES_ATOMS',
    'ClubDayParticipation',
    'ClubDayIntraClubMatchup',
    'ClubDayOpponentMatchup',
    'ClubDaySameField',
    'ClubDayContiguousSlots',
    'CLUB_DAY_ATOMS',
]
