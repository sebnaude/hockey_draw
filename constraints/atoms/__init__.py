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
# spec-030: PHLAnd2ndConcurrencyAtBroadmeadow deleted — its same-club PHL/2nd
# same-Broadmeadow-slot rule is a strict subset of PHLAnd2ndAdjacency's
# same-venue branch (which only allows same-field adjacent slots), so the
# same-slot case was already forbidden. Redundant in a fresh build.
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

# spec-027: regeneration soft-analogue atoms. Each is a SOFT analogue of a
# production HARD constraint, emitting a penalty into data['penalties'] instead
# of a hard clause. Selected ONLY by the `regen` group (scoped regeneration,
# spec-026); a fresh season build never applies them.
from constraints.atoms.phl_2nd_adjacency_regen_soft import PHLAnd2ndAdjacencyRegenSoft
from constraints.atoms.away_club_home_weekends_count_regen_soft import (
    AwayClubHomeWeekendsCountRegenSoft,
)
from constraints.atoms.clubvsclub_stacked_weekends_regen_soft import (
    ClubVsClubStackedWeekendsRegenSoft,
)
from constraints.atoms.clubvsclub_stacked_colocation_regen_soft import (
    ClubVsClubStackedCoLocationRegenSoft,
)
from constraints.atoms.equal_matchup_spacing_regen_soft import (
    EqualMatchUpSpacingRegenSoft,
)
from constraints.atoms.balanced_bye_spacing_regen_soft import (
    BalancedByeSpacingRegenSoft,
)
from constraints.atoms.club_day_participation_regen_soft import (
    ClubDayParticipationRegenSoft,
)
from constraints.atoms.club_day_intra_club_matchup_regen_soft import (
    ClubDayIntraClubMatchupRegenSoft,
)
from constraints.atoms.club_day_opponent_matchup_regen_soft import (
    ClubDayOpponentMatchupRegenSoft,
)
from constraints.atoms.club_day_same_field_regen_soft import (
    ClubDaySameFieldRegenSoft,
)
from constraints.atoms.club_day_contiguous_slots_regen_soft import (
    ClubDayContiguousSlotsRegenSoft,
)
from constraints.atoms.club_game_spread_regen_soft import ClubGameSpreadRegenSoft
from constraints.atoms.venue_earliest_slot_fill_regen_soft import (
    VenueEarliestSlotFillRegenSoft,
)

# Side-effect import: registers FORCED/BLOCKED count adjusters for the
# constraints whose atoms haven't been split out yet (Phase 4).
from constraints.atoms import _adjusters  # noqa: F401


# spec-010: PHLRoundOnePlay removed from the dispatched-atom list; spec-(final)
# deleted the atom file entirely (convenor uses FORCED_GAMES for round-1 play).
PHL_TIMES_ATOMS = [
    PHLConcurrencyAtBroadmeadow,
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


# spec-027: the regeneration soft-analogue cluster (selected only by `regen`).
REGEN_SOFT_ATOMS = [
    PHLAnd2ndAdjacencyRegenSoft,
    AwayClubHomeWeekendsCountRegenSoft,
    ClubVsClubStackedWeekendsRegenSoft,
    ClubVsClubStackedCoLocationRegenSoft,
    EqualMatchUpSpacingRegenSoft,
    BalancedByeSpacingRegenSoft,
    ClubDayParticipationRegenSoft,
    ClubDayIntraClubMatchupRegenSoft,
    ClubDayOpponentMatchupRegenSoft,
    ClubDaySameFieldRegenSoft,
    ClubDayContiguousSlotsRegenSoft,
    ClubGameSpreadRegenSoft,
    VenueEarliestSlotFillRegenSoft,
]


__all__ = [
    'Atom',
    'PHLConcurrencyAtBroadmeadow',
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
    # spec-027 regen soft-analogue atoms
    'PHLAnd2ndAdjacencyRegenSoft',
    'AwayClubHomeWeekendsCountRegenSoft',
    'ClubVsClubStackedWeekendsRegenSoft',
    'ClubVsClubStackedCoLocationRegenSoft',
    'EqualMatchUpSpacingRegenSoft',
    'BalancedByeSpacingRegenSoft',
    'ClubDayParticipationRegenSoft',
    'ClubDayIntraClubMatchupRegenSoft',
    'ClubDayOpponentMatchupRegenSoft',
    'ClubDaySameFieldRegenSoft',
    'ClubDayContiguousSlotsRegenSoft',
    'ClubGameSpreadRegenSoft',
    'VenueEarliestSlotFillRegenSoft',
    'REGEN_SOFT_ATOMS',
]
