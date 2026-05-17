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
from constraints.atoms.club_vs_club_coincidence import ClubVsClubCoincidence
from constraints.atoms.club_vs_club_field_limit import ClubVsClubFieldLimit
from constraints.atoms.club_vs_club_deficit_penalty import ClubVsClubDeficitPenalty
from constraints.atoms.phl_2nd_back_to_back import PHLAnd2ndBackToBackSameField
from constraints.atoms.soft_lex_matchup_ordering import SoftLexMatchupOrdering
from constraints.atoms.same_grade_same_club_no_concurrency import (
    SameGradeSameClubNoConcurrency,
)
from constraints.atoms.team_pair_no_concurrency import TeamPairNoConcurrency

# Side-effect import: registers FORCED/BLOCKED count adjusters for the
# constraints whose atoms haven't been split out yet (Phase 4).
from constraints.atoms import _adjusters  # noqa: F401


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


CLUB_VS_CLUB_ATOMS = [
    ClubVsClubCoincidence,
    ClubVsClubFieldLimit,
    ClubVsClubDeficitPenalty,
    PHLAnd2ndBackToBackSameField,
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
    'ClubVsClubCoincidence',
    'ClubVsClubFieldLimit',
    'ClubVsClubDeficitPenalty',
    'PHLAnd2ndBackToBackSameField',
    'CLUB_VS_CLUB_ATOMS',
    'SoftLexMatchupOrdering',
    'SameGradeSameClubNoConcurrency',
    'TeamPairNoConcurrency',
]
