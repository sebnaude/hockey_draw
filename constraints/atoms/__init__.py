"""Atomic constraints — one idea per class.

Each atom subclasses `Atom` (constraints/atoms/base.py) and implements:
- `apply(model, X, data, registry)` — adds CP-SAT constraints, returns count
  added. Shared helper variables are created/looked-up via the pool-style
  `registry` API (`get_or_create_bool`, `get_or_create_presence`, `register`).

Atoms are instantiated and dispatched by `UnifiedConstraintEngine`. Their
`canonical_name` matches a `ConstraintInfo` entry in `constraints/registry.py`
(with `atom_group` set to the legacy combined-constraint they were split from).
"""
from constraints.atoms.base import Atom
from constraints.atoms.phl_concurrency import PHLConcurrencyAtBroadmeadow
from constraints.atoms.phl_2nd_concurrency import PHLAnd2ndConcurrencyAtBroadmeadow
# spec-015: GosfordFridayRoundsForced deleted — its per-round sum==1 rule is
# expressed generically via FORCED_GAMES count entries (scope + count +
# constraint type). See docs/system/FORCED_GAMES_AS_COUNT_RULES.md.
from constraints.atoms.club_day_participation import ClubDayParticipation
from constraints.atoms.club_day_intra_club_matchup import ClubDayIntraClubMatchup
from constraints.atoms.club_day_opponent_matchup import ClubDayOpponentMatchup
from constraints.atoms.club_day_same_field import ClubDaySameField
from constraints.atoms.club_day_contiguous_slots import ClubDayContiguousSlots
from constraints.atoms.venue_earliest_slot_fill import VenueEarliestSlotFill
from constraints.atoms.club_no_concurrent_slot import ClubNoConcurrentSlot
# spec-005: the legacy Phase-3c ClubVsClubAlignment atoms (Coincidence,
# FieldLimit, DeficitPenalty, PHLAnd2ndBackToBackSameField) were deleted —
# fully superseded by the `ClubVsClubStackedAlignment` cluster below.
from constraints.atoms.club_vs_club_stacked_weekends import (
    ClubVsClubStackedWeekends,
)
from constraints.atoms.club_vs_club_stacked_co_location import (
    ClubVsClubStackedCoLocation,
)
from constraints.atoms.soft_lex_matchup_ordering import SoftLexMatchupOrdering
from constraints.atoms.same_grade_same_club_no_concurrency import (
    SameGradeSameClubNoConcurrency,
)
from constraints.atoms.team_pair_no_concurrency import TeamPairNoConcurrency
from constraints.atoms.nihc_fill_wf_before_ef import NIHCFillWFBeforeEF
from constraints.atoms.nihc_fill_ef_before_sf import NIHCFillEFBeforeSF
# spec-014: PHL/2nd same-club adjacency atom (replaces the legacy
# `_phl_adjacency_hard` engine method).
from constraints.atoms.phl_2nd_adjacency import PHLAnd2ndAdjacency
from constraints.atoms.preferred_weekends_away_ground import (
    PreferredWeekendsAwayGround,
)
# spec-020: generic soft analogue of the whole FORCED_GAMES grammar
# (penalty-on-deviation). Replaces the narrow PHL-only PreferredDates.
from constraints.atoms.preferred_games import PreferredGames
from constraints.atoms.away_club_home_weekends_count import (
    AwayClubHomeWeekendsCount,
)
from constraints.atoms.away_club_home_balance import (
    AwayClubPerOpponentAndAggregateHomeBalance,
)
# spec-008 Part B: byes-as-first-class spacing atom.
from constraints.atoms.balanced_bye_spacing import BalancedByeSpacing

# Side-effect import: registers FORCED/BLOCKED count adjusters for the
# constraints whose atoms haven't been split out yet (Phase 4).
from constraints.atoms import _adjusters  # noqa: F401


# spec-010: PHLRoundOnePlay removed from the dispatched-atom list; spec-(final)
# deleted the atom file entirely (convenor uses FORCED_GAMES for round-1 play).
PHL_TIMES_ATOMS = [
    PHLConcurrencyAtBroadmeadow,
    PHLAnd2ndConcurrencyAtBroadmeadow,
]


CLUB_DAY_ATOMS = [
    ClubDayParticipation,
    ClubDayIntraClubMatchup,
    ClubDayOpponentMatchup,
    ClubDaySameField,
    ClubDayContiguousSlots,
]


# spec-005 replacement cluster. ORDER MATTERS: `ClubVsClubStackedWeekends`
# must run first so the co-location atom can read the `play` indicators
# from the helper-var registry.
CLUB_VS_CLUB_STACKED_ATOMS = [
    ClubVsClubStackedWeekends,
    ClubVsClubStackedCoLocation,
]


NIHC_FIELD_FILL_ORDER_ATOMS = [
    NIHCFillWFBeforeEF,
    NIHCFillEFBeforeSF,
]


__all__ = [
    'Atom',
    'PHLConcurrencyAtBroadmeadow',
    'PHLAnd2ndConcurrencyAtBroadmeadow',
    'PHL_TIMES_ATOMS',
    'ClubDayParticipation',
    'ClubDayIntraClubMatchup',
    'ClubDayOpponentMatchup',
    'ClubDaySameField',
    'ClubDayContiguousSlots',
    'VenueEarliestSlotFill',
    'ClubNoConcurrentSlot',
    'CLUB_DAY_ATOMS',
    'ClubVsClubStackedWeekends',
    'ClubVsClubStackedCoLocation',
    'CLUB_VS_CLUB_STACKED_ATOMS',
    'SoftLexMatchupOrdering',
    'SameGradeSameClubNoConcurrency',
    'TeamPairNoConcurrency',
    'NIHCFillWFBeforeEF',
    'NIHCFillEFBeforeSF',
    'NIHC_FIELD_FILL_ORDER_ATOMS',
    'PHLAnd2ndAdjacency',
    'PreferredWeekendsAwayGround',
    'PreferredGames',
    'AwayClubHomeWeekendsCount',
    'AwayClubPerOpponentAndAggregateHomeBalance',
    'BalancedByeSpacing',
]
