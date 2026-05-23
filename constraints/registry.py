"""
Constraint Registry -- single source of truth for constraint naming across all subsystems.

Maps between solver class names, tester check methods, unified engine skip names,
severity levels, and slack keys.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set
import re


@dataclass
class ConstraintInfo:
    """Metadata for a single logical constraint."""
    canonical_name: str
    solver_class_names: List[str]  # All variants (original, AI)
    tester_check_methods: List[str]  # Can be >1 (EqualGames + BalancedMatchups)
    tester_violation_names: List[str]  # Violation.constraint values emitted by tester
    severity_level: int
    slack_key: Optional[str] = None
    has_soft_component: bool = False
    tester_only: bool = False  # True if no solver equivalent (diagnostic check)
    # spec-023: named groups this constraint belongs to. A "group" is a named,
    # possibly-overlapping set of WHOLE constraints; a solve applies the deduped
    # union of selected groups in canonical (registry insertion) order. Overlap
    # is allowed and expected (e.g. {core, critical_feasibility}). Empty for
    # tester-only / obsolete legacy entries that are not selected by any group.
    groups: FrozenSet[str] = frozenset()
    # Phase 2 additions for atomization
    atom_group: Optional[str] = None
    """Name of the legacy combined constraint this atom was split from
    (e.g. 'PHLAndSecondGradeTimes'). None for single-idea constraints."""
    required_helpers: List[str] = field(default_factory=list)
    """Helper-var kinds the atom declares (must exist in the helper-var catalog)."""
    forced_blocked_adjuster: Optional[Callable[[Dict, List, List], Dict]] = None
    """Optional callable that returns count-adjustment metadata.

    Signature: adjuster(constraint_data, forced_games, blocked_games) -> dict.
    Engine runs every adjuster after FORCED/BLOCKED parsing, before constraints
    apply(). Output is stored at data['count_adjustments'][canonical_name]."""


# Master registry -- keyed by canonical name
CONSTRAINT_REGISTRY: Dict[str, ConstraintInfo] = {
    'NoDoubleBookingTeams': ConstraintInfo(
        canonical_name='NoDoubleBookingTeams',
        solver_class_names=['NoDoubleBookingTeamsConstraint', 'NoDoubleBookingTeamsConstraintAI'],
        tester_check_methods=['_check_no_double_booking_teams'],
        tester_violation_names=['NoDoubleBookingTeams'],
        severity_level=1,
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    'NoDoubleBookingFields': ConstraintInfo(
        canonical_name='NoDoubleBookingFields',
        solver_class_names=['NoDoubleBookingFieldsConstraint', 'NoDoubleBookingFieldsConstraintAI'],
        tester_check_methods=['_check_no_double_booking_fields'],
        tester_violation_names=['NoDoubleBookingFields'],
        severity_level=1,
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    'EqualGamesAndBalanceMatchUps': ConstraintInfo(
        canonical_name='EqualGamesAndBalanceMatchUps',
        solver_class_names=['EnsureEqualGamesAndBalanceMatchUps', 'EnsureEqualGamesAndBalanceMatchUpsAI'],
        tester_check_methods=['_check_equal_games', '_check_balanced_matchups'],
        tester_violation_names=['EqualGames', 'BalancedMatchups'],
        severity_level=1,
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    # OBSOLETE (spec-004): superseded by `AwayClubHomeWeekendsCount` +
    # `AwayClubPerOpponentAndAggregateHomeBalance`. The legacy combined class
    # remains importable in `constraints/archived/` for parity reference, but
    # `FiftyFiftyHomeandAway` is NOT wired into any production stage.
    # Registry entry kept so legacy solver-class-name lookups still resolve
    # and the tester can keep emitting violations under the same name for the
    # per-pair balance case.
    'FiftyFiftyHomeandAway': ConstraintInfo(
        canonical_name='FiftyFiftyHomeandAway',
        solver_class_names=['FiftyFiftyHomeandAway', 'FiftyFiftyHomeandAwayAI'],
        tester_check_methods=['_check_fifty_fifty_home_away'],
        tester_violation_names=['FiftyFiftyHomeAway'],
        severity_level=1,
    ),
    # spec-004: FORCED-Friday aware home-weekend count atom. For each
    # away-based club (non-Broadmeadow home venue), pins three sums:
    # friday-home weekends, sunday-home weekends, and total-home weekends.
    # Replaces the home-weekend logic historically in `FiftyFiftyHomeandAway`.
    # spec-018: the venue-sequencing rules were deleted along with the
    # `_check_maitland_back_to_back` tester method this entry used to reuse.
    # Per-club home-weekend counts are verified by this atom's own no-mock
    # CP-SAT unit test (tests/atoms/test_away_club_home_weekends_count.py); the
    # registry maps it to the live per-pair/aggregate balance check, which is
    # the closest standing tester check covering away-club home/away semantics.
    'AwayClubHomeWeekendsCount': ConstraintInfo(
        canonical_name='AwayClubHomeWeekendsCount',
        solver_class_names=['AwayClubHomeWeekendsCount'],
        tester_check_methods=['_check_fifty_fifty_home_away'],
        tester_violation_names=['FiftyFiftyHomeAway'],
        severity_level=1,
        groups=frozenset({'core', 'home_away_balance'}),
    ),
    # spec-004: per-opponent + aggregate home/away balance atom. Replaces the
    # per-pair and aggregate blocks of the obsolete `FiftyFiftyHomeandAway`.
    'AwayClubPerOpponentAndAggregateHomeBalance': ConstraintInfo(
        canonical_name='AwayClubPerOpponentAndAggregateHomeBalance',
        solver_class_names=['AwayClubPerOpponentAndAggregateHomeBalance'],
        tester_check_methods=['_check_fifty_fifty_home_away'],
        tester_violation_names=['FiftyFiftyHomeAway'],
        severity_level=1,
        groups=frozenset({'core', 'home_away_balance', 'core_hard'}),
    ),
    # spec-018: `NonDefaultHomeGrouping` (alias `MaitlandHomeGrouping`;
    # solver classes `MaitlandHomeGrouping*` / `MaxMaitlandHomeWeekends*`)
    # DELETED — the convenor no longer wants the solver enforcing the
    # *sequence* of an away-based club's home/away weekends (back-to-back home
    # weekends and long away runs are both fine). Per-club home-weekend counts
    # are still enforced by the spec-004 `AwayClubHomeWeekendsCount` atom.
    # spec-014: rewrite of the legacy `PHLAndSecondGradeAdjacency`. The old
    # engine method `_phl_adjacency_hard` only *forbade* two bad +/-180-min
    # patterns; the new atom *forces* same-club PHL/2nd back-to-back at one
    # venue, or a >= 180-min start-time gap across venues. Dispatched via the
    # non-engine fallback (own name only in `solver_class_names` so dispatch
    # resolves correctly under both `use_ai` modes — see spec-014 design note).
    'PHLAnd2ndAdjacency': ConstraintInfo(
        canonical_name='PHLAnd2ndAdjacency',
        solver_class_names=['PHLAnd2ndAdjacency'],
        tester_check_methods=['_check_phl_2nd_adjacency'],
        tester_violation_names=['PHLAnd2ndAdjacency'],
        severity_level=1,
        groups=frozenset({'core', 'critical_feasibility'}),
    ),
    'PHLAndSecondGradeTimes': ConstraintInfo(
        canonical_name='PHLAndSecondGradeTimes',
        solver_class_names=['PHLAndSecondGradeTimes', 'PHLAndSecondGradeTimesAI'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
    ),
    # --- Atoms split from PHLAndSecondGradeTimes (Phase 3) ---
    'PHLConcurrencyAtBroadmeadow': ConstraintInfo(
        canonical_name='PHLConcurrencyAtBroadmeadow',
        solver_class_names=['PHLConcurrencyAtBroadmeadow'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
        atom_group='PHLAndSecondGradeTimes',
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    'PHLAnd2ndConcurrencyAtBroadmeadow': ConstraintInfo(
        canonical_name='PHLAnd2ndConcurrencyAtBroadmeadow',
        solver_class_names=['PHLAnd2ndConcurrencyAtBroadmeadow'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
        atom_group='PHLAndSecondGradeTimes',
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    # spec-015: GosfordFridayRoundsForced entry removed. The per-round
    # `sum == 1` rule is expressed generically via FORCED_GAMES count entries
    # (scope + count + constraint type) in the season config — see
    # docs/system/FORCED_GAMES_AS_COUNT_RULES.md. The bespoke atom was a second
    # source of truth for the same capability.
    # OBSOLETE (spec-010): "every PHL team plays round 1" removed; the atom
    # file and registry entry were subsequently DELETED. The convenor uses
    # FORCED_GAMES entries to express deliberate round-1 placement when needed.
    # spec-020: `PreferredDates` (narrow PHL-only soft `|sum − 1|` on a date)
    # DELETED — its behaviour is now a `PREFERRED_GAMES` config entry handled by
    # the generic `PreferredGames` soft atom below.
    'EqualMatchUpSpacing': ConstraintInfo(
        canonical_name='EqualMatchUpSpacing',
        solver_class_names=['EqualMatchUpSpacingConstraint', 'EqualMatchUpSpacingConstraintAI'],
        tester_check_methods=['_check_equal_matchup_spacing'],
        tester_violation_names=['EqualMatchUpSpacing'],
        severity_level=1,
        slack_key='EqualMatchUpSpacingConstraint',
        groups=frozenset({'core', 'critical_feasibility'}),
    ),
    # spec-008 Part B: byes are first-class. A team's bye rounds are spread
    # across the season using the same `_spacing.ideal_bye_gap` math as
    # repeat matchups. HARD severity 2; separate slack key
    # `BalancedByeSpacing` so the convenor can loosen one without the other.
    'BalancedByeSpacing': ConstraintInfo(
        canonical_name='BalancedByeSpacing',
        solver_class_names=['BalancedByeSpacing'],
        tester_check_methods=['_check_balanced_bye_spacing'],
        tester_violation_names=['BalancedByeSpacing'],
        severity_level=2,
        slack_key='BalancedByeSpacing',
        groups=frozenset({'core', 'critical_feasibility'}),
    ),
    'ClubDay': ConstraintInfo(
        canonical_name='ClubDay',
        solver_class_names=['ClubDayConstraint', 'ClubDayConstraintAI'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
    ),
    # --- Atoms split from ClubDay (Phase 3b) ---
    'ClubDayParticipation': ConstraintInfo(
        canonical_name='ClubDayParticipation',
        solver_class_names=['ClubDayParticipation'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        groups=frozenset({'core', 'club_day'}),
    ),
    'ClubDayIntraClubMatchup': ConstraintInfo(
        canonical_name='ClubDayIntraClubMatchup',
        solver_class_names=['ClubDayIntraClubMatchup'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        groups=frozenset({'core', 'club_day'}),
    ),
    'ClubDayOpponentMatchup': ConstraintInfo(
        canonical_name='ClubDayOpponentMatchup',
        solver_class_names=['ClubDayOpponentMatchup'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        groups=frozenset({'core', 'club_day'}),
    ),
    'ClubDaySameField': ConstraintInfo(
        canonical_name='ClubDaySameField',
        solver_class_names=['ClubDaySameField'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        required_helpers=['club_day_field_used'],
        groups=frozenset({'core', 'club_day'}),
    ),
    'ClubDayContiguousSlots': ConstraintInfo(
        canonical_name='ClubDayContiguousSlots',
        solver_class_names=['ClubDayContiguousSlots'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        required_helpers=['club_day_slot_used'],
        groups=frozenset({'core', 'club_day'}),
    ),
    # spec-018: `AwayAtNonDefaultGrouping` (alias `AwayAtMaitlandGrouping`;
    # solver classes `AwayAtMaitlandGrouping*`) DELETED — the convenor no
    # longer caps how many distinct away clubs may visit a non-default venue
    # in one weekend.
    'TeamConflict': ConstraintInfo(
        canonical_name='TeamConflict',
        solver_class_names=['TeamConflictConstraint', 'TeamConflictConstraintAI'],
        tester_check_methods=['_check_team_conflict'],
        tester_violation_names=['TeamConflict'],
        severity_level=2,
        # spec-027: a named team pair shares players — two games in one week is a
        # genuine impossibility, so it stays HARD in regen. Tagged core_hard ONLY
        # (no `core`): TeamConflict is data-driven and was never in DEFAULT_STAGES,
        # so it is deliberately absent from the fresh-build `default` group; adding
        # only `core_hard` pulls it into the `regen` group without changing any
        # fresh-season behaviour.
        groups=frozenset({'core_hard'}),
    ),
    # OBSOLETE (spec-007): the legacy `ClubGradeAdjacencyConstraint` did two
    # things — (1) a hard same-grade-same-club no-concurrency rule and (2) a
    # soft adjacent-grade penalty. The soft adjacent-grade portion has been
    # REMOVED ENTIRELY; the hard portion is now `SameGradeSameClubNoConcurrency`.
    # This entry remains so legacy solver-class-name lookups still resolve to a
    # canonical name (the archived class file is still importable for parity
    # tests), but `ClubGradeAdjacency` is NOT wired into any production stage.
    # The tester still emits violations under the `ClubGradeAdjacency` name for
    # the same-grade-same-club case only.
    'ClubGradeAdjacency': ConstraintInfo(
        canonical_name='ClubGradeAdjacency',
        solver_class_names=['ClubGradeAdjacencyConstraint', 'ClubGradeAdjacencyConstraintAI'],
        tester_check_methods=['_check_club_grade_adjacency'],
        tester_violation_names=['ClubGradeAdjacency'],
        severity_level=3,
    ),
    # spec-007: hard atom carrying the genuinely-fundamental portion of the
    # obsolete ClubGradeAdjacency cluster. Same-grade, same-club teams must
    # not play simultaneously.
    'SameGradeSameClubNoConcurrency': ConstraintInfo(
        canonical_name='SameGradeSameClubNoConcurrency',
        solver_class_names=['SameGradeSameClubNoConcurrency'],
        tester_check_methods=['_check_club_grade_adjacency'],
        tester_violation_names=['ClubGradeAdjacency'],
        severity_level=1,
        atom_group='ClubGradeAdjacency',
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    # spec-007: soft atom for convenor-supplied team pairs that should avoid
    # concurrent (week, day_slot) appearances (siblings in non-adjacent grades,
    # specific coach conflicts, etc.). Reads `TEAM_PAIR_NO_CONCURRENCY`.
    'TeamPairNoConcurrency': ConstraintInfo(
        canonical_name='TeamPairNoConcurrency',
        solver_class_names=['TeamPairNoConcurrency'],
        tester_check_methods=['_check_team_pair_no_concurrency'],
        tester_violation_names=['TeamPairNoConcurrency'],
        severity_level=3,
        has_soft_component=True,
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    # spec-003: NIHC field-fill order. Two atoms per (date, day_slot) at
    # Broadmeadow forcing WF -> EF -> SF fill priority. Replaces the legacy
    # perennial "last game of day on WF" review-only rule with a hard
    # constraint that holds for every slot.
    # spec-016: re-leveled from hard severity 1 to SOFT severity 5. The two
    # atoms now add a `nihc_fill_order` penalty term (out-of-order fill)
    # instead of a hard implication — a symmetry-breaker, not a feasibility
    # rule. has_soft_component=True; they live in `soft_optimisation`.
    'NIHCFillWFBeforeEF': ConstraintInfo(
        canonical_name='NIHCFillWFBeforeEF',
        solver_class_names=['NIHCFillWFBeforeEF'],
        tester_check_methods=['_check_nihc_fill_wf_before_ef'],
        tester_violation_names=['NIHCFillWFBeforeEF'],
        severity_level=5,
        has_soft_component=True,
        atom_group='NIHCFieldFillOrder',
        required_helpers=['nihc_field_used'],
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    'NIHCFillEFBeforeSF': ConstraintInfo(
        canonical_name='NIHCFillEFBeforeSF',
        solver_class_names=['NIHCFillEFBeforeSF'],
        tester_check_methods=['_check_nihc_fill_ef_before_sf'],
        tester_violation_names=['NIHCFillEFBeforeSF'],
        severity_level=5,
        has_soft_component=True,
        atom_group='NIHCFieldFillOrder',
        required_helpers=['nihc_field_used'],
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    'ClubVsClubAlignment': ConstraintInfo(
        canonical_name='ClubVsClubAlignment',
        solver_class_names=['ClubVsClubAlignment', 'ClubVsClubAlignmentAI'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
    ),
    # The four Phase-3c atoms split from ClubVsClubAlignment (Coincidence,
    # FieldLimit, DeficitPenalty, PHLAnd2ndBackToBackSameField) were DELETED
    # (spec-005) — fully superseded by the `ClubVsClubStackedAlignment`
    # cluster below. The `ClubVsClubAlignment` entry above remains as the
    # slack-key holder + legacy parity-reference engine path.
    # spec-005: replacement cluster `ClubVsClubStackedAlignment`. Two atoms
    # cooperate via `STACK_PLAY_PREFIX` helper-var keys:
    #   - `ClubVsClubStackedWeekends` (HARD): pins per-(pair, grade) Sunday
    #     meetings and enforces a strict nested-superset implication chain
    #     across weeks (higher-count grades play whenever lower-count
    #     grades play for the same pair on the same Sunday).
    #   - `ClubVsClubStackedCoLocation` (HARD): on every Sunday where ≥ 2
    #     grades play for a pair, all those games must be on the same field
    #     with contiguous day_slots (no internal gaps).
    # PHL Sunday budget is FORCED-Friday-aware via the per-pair helper
    # `phl_forced_friday_meetings(data, a, b)` added to
    # `constraints/atoms/_phl_forced_friday_helper.py`.
    'ClubVsClubStackedWeekends': ConstraintInfo(
        canonical_name='ClubVsClubStackedWeekends',
        solver_class_names=['ClubVsClubStackedWeekends'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
        atom_group='ClubVsClubStackedAlignment',
        required_helpers=['cvc_stack_play'],
        groups=frozenset({'core', 'club_alignment'}),
    ),
    'ClubVsClubStackedCoLocation': ConstraintInfo(
        canonical_name='ClubVsClubStackedCoLocation',
        solver_class_names=['ClubVsClubStackedCoLocation'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
        atom_group='ClubVsClubStackedAlignment',
        required_helpers=[
            'cvc_stack_play', 'cvc_stack_field_used', 'cvc_stack_slot_used',
            'cvc_stack_active',
        ],
        groups=frozenset({'core', 'club_alignment'}),
    ),
    'ClubGameSpread': ConstraintInfo(
        canonical_name='ClubGameSpread',
        solver_class_names=['ClubGameSpread', 'ClubGameSpreadAI'],
        tester_check_methods=['_check_club_game_spread'],
        tester_violation_names=['ClubGameSpread'],
        severity_level=3,
        slack_key='ClubGameSpread',
        # spec-023: §1 lists this as {core} minimum; reconciled against
        # DEFAULT_STAGES — ClubGameSpread lives in the (non-soft_only) club_day
        # stage, so it carries the club_day legacy-stage group tag too.
        groups=frozenset({'core', 'club_day'}),
    ),
    # spec-021: extracted from ClubGameSpread's lower no-double-up bound (which
    # is concurrency, not contiguity). HARD, capacity-aware via no_field_slots.
    'ClubNoConcurrentSlot': ConstraintInfo(
        canonical_name='ClubNoConcurrentSlot',
        solver_class_names=['ClubNoConcurrentSlot'],
        tester_check_methods=['_check_club_no_concurrent_slot'],
        tester_violation_names=['ClubNoConcurrentSlot'],
        severity_level=2,
        # spec-023: §1 lists this as {core} minimum; reconciled against
        # DEFAULT_STAGES — ClubNoConcurrentSlot is wired into critical_feasibility
        # (NOT club_day, which the spec's illustrative example guessed), so it
        # carries the critical_feasibility legacy-stage group tag.
        # spec-027: physical slot-concurrency — stays HARD in regen (core_hard).
        groups=frozenset({'core', 'critical_feasibility', 'core_hard'}),
    ),
    'ClubFieldConcentration': ConstraintInfo(
        canonical_name='ClubFieldConcentration',
        solver_class_names=[],  # Tester-only diagnostic
        tester_check_methods=['_check_club_field_concentration'],
        tester_violation_names=['ClubFieldConcentration'],
        severity_level=3,
        tester_only=True,
    ),
    # spec-024: `MaximiseClubsPerTimeslotBroadmeadow` and
    # `MinimiseClubsOnAFieldBroadmeadow` deleted. Their "spread the clubs around"
    # intent is now expressed club-side by the field-aware `ClubGameSpread`
    # (per-field contiguity + off-primary-field soft penalty).
    # spec-021: replaces the old soft_only `EnsureBestTimeslotChoices` engine
    # rule with a HARD anchored monotone-fill atom (severity 2). Games at a
    # venue pack into the earliest timeslots — no gaps + earliest start, which
    # structurally avoids the 7 pm slot (no 7 pm penalty re-added). Non-engine
    # atom (dispatched via the stages.py legacy-class fallback).
    'VenueEarliestSlotFill': ConstraintInfo(
        canonical_name='VenueEarliestSlotFill',
        solver_class_names=['VenueEarliestSlotFill'],
        tester_check_methods=['_check_venue_earliest_slot_fill'],
        tester_violation_names=['VenueEarliestSlotFill'],
        severity_level=2,
        required_helpers=['venue_slot_used'],
        # spec-023: §1 lists this as {core} minimum; reconciled against
        # DEFAULT_STAGES — VenueEarliestSlotFill is wired into critical_feasibility,
        # so it carries the critical_feasibility legacy-stage group tag.
        groups=frozenset({'core', 'critical_feasibility'}),
    ),
    'PreferredTimes': ConstraintInfo(
        canonical_name='PreferredTimes',
        solver_class_names=['PreferredTimesConstraint', 'PreferredTimesConstraintAI'],
        tester_check_methods=['_check_preferred_times'],
        tester_violation_names=['PreferredTimesConstraint'],
        severity_level=5,
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    'SoftLexMatchupOrdering': ConstraintInfo(
        canonical_name='SoftLexMatchupOrdering',
        solver_class_names=['SoftLexMatchupOrdering'],
        tester_check_methods=[],
        tester_violation_names=[],
        severity_level=5,
        has_soft_component=True,
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    # spec-006: soft penalty for preferred / avoided weekends at away grounds
    # (e.g. NRL-Knights home games at Maitland Park). Pure soft: never blocks
    # feasibility. Reads `data['preferred_weekends']` set by the season config.
    'PreferredWeekendsAwayGround': ConstraintInfo(
        canonical_name='PreferredWeekendsAwayGround',
        solver_class_names=['PreferredWeekendsAwayGround'],
        tester_check_methods=[],
        tester_violation_names=[],
        severity_level=5,
        has_soft_component=True,
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    # spec-020: generic soft analogue of the whole FORCED_GAMES grammar.
    # Penalty-on-deviation from a per-scope target (equal/lesse/less/greater/
    # greatere + count). Pure soft: never blocks feasibility. Reads
    # `data['preferred_games']`. Replaces the deleted PHL-only `PreferredDates`.
    'PreferredGames': ConstraintInfo(
        canonical_name='PreferredGames',
        solver_class_names=['PreferredGames'],
        tester_check_methods=['_check_preferred_games'],
        tester_violation_names=['PreferredGames'],
        severity_level=5,
        has_soft_component=True,
        groups=frozenset({'soft', 'soft_optimisation'}),
    ),
    # spec-018: `MaitlandAlternateHomeAway` (spec-012 soft) DELETED — its whole
    # purpose was to push an H-A-H-A weekend pattern, the sequencing the
    # convenor is discarding. Removed alongside `NonDefaultHomeGrouping`.
    'ForcedGames': ConstraintInfo(
        canonical_name='ForcedGames',
        solver_class_names=[],  # Enforced by generate_X variable elimination, not a Constraint class
        tester_check_methods=['_check_forced_games'],
        tester_violation_names=['ForcedGames'],
        severity_level=1,
        tester_only=True,
        # spec-027: a freeze pin. Stays HARD in regen (core_hard). Enforced at
        # generate_X time (no solver class), so apply_constraint_set skips it;
        # the tag is for `regen`-group membership + documentation completeness.
        groups=frozenset({'core_hard'}),
    ),
    'BlockedGames': ConstraintInfo(
        canonical_name='BlockedGames',
        solver_class_names=[],  # Enforced by generate_X variable elimination, not a Constraint class
        tester_check_methods=['_check_blocked_games'],
        tester_violation_names=['BlockedGames'],
        severity_level=1,
        tester_only=True,
        # spec-027: a freeze pin (variable elimination). Stays HARD in regen.
        groups=frozenset({'core_hard'}),
    ),
    # spec-025: LOCKED_PAIRINGS — mechanical date-pins (pairing + date, time/slot/
    # field free). Enforced by generate_X via a per-pin `sum == 1` over candidate
    # vars on the pin's date (not a Constraint class). The tester re-verifies each
    # pin's pairing is present on its date in the finished draw.
    'LockedPairings': ConstraintInfo(
        canonical_name='LockedPairings',
        solver_class_names=[],  # Enforced by generate_X sum==1, not a Constraint class
        tester_check_methods=['_check_locked_pairings'],
        tester_violation_names=['LockedPairings'],
        severity_level=1,
        tester_only=True,
        # spec-027: THE freeze itself in regen — soft pins would defeat the
        # purpose. Stays HARD (core_hard). Enforced at generate_X time.
        groups=frozenset({'core_hard'}),
    ),
}


# Build reverse lookup: solver class name -> canonical name
_SOLVER_NAME_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _info in CONSTRAINT_REGISTRY.items():
    for _solver_name in _info.solver_class_names:
        _SOLVER_NAME_TO_CANONICAL[_solver_name] = _canonical

# Build reverse lookup: tester method -> canonical name
_METHOD_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _info in CONSTRAINT_REGISTRY.items():
    for _method in _info.tester_check_methods:
        # Only map if not already mapped (first registry entry wins for shared methods)
        if _method not in _METHOD_TO_CANONICAL:
            _METHOD_TO_CANONICAL[_method] = _canonical


# Suffix patterns to strip for normalization
_SUFFIX_PATTERN = re.compile(r'(ConstraintAI|Constraint|AI)$')


def normalize_constraint_name(name: str) -> Optional[str]:
    """
    Strip Constraint/AI/ConstraintAI suffixes to get canonical name.

    Returns the canonical name if found in the registry, otherwise the
    stripped name (which may or may not be in the registry).
    """
    # Direct match first
    if name in CONSTRAINT_REGISTRY:
        return name

    # Try solver name lookup
    canonical = _SOLVER_NAME_TO_CANONICAL.get(name)
    if canonical:
        return canonical

    # Strip suffixes and try again
    stripped = _SUFFIX_PATTERN.sub('', name)
    if stripped in CONSTRAINT_REGISTRY:
        return stripped

    # Try prefixed forms (e.g., 'Ensure' prefix)
    for canon_name in CONSTRAINT_REGISTRY:
        if stripped.endswith(canon_name) or canon_name.endswith(stripped):
            return canon_name

    return stripped if stripped else None


def get_canonical_for_solver_name(solver_name: str) -> Optional[str]:
    """Map solver class name (e.g., 'NoDoubleBookingTeamsConstraintAI') to canonical name."""
    return _SOLVER_NAME_TO_CANONICAL.get(solver_name)


def get_checks_for_applied_constraints(applied_names: List[str]) -> Set[str]:
    """
    Given solver constraint names from metadata, return set of tester method names to run.

    Args:
        applied_names: List of solver class names (e.g., from metadata['constraints_applied'])

    Returns:
        Set of tester method names (e.g., {'_check_no_double_booking_teams', ...})
    """
    methods = set()
    for name in applied_names:
        canonical = get_canonical_for_solver_name(name) or normalize_constraint_name(name)
        if canonical and canonical in CONSTRAINT_REGISTRY:
            for method in CONSTRAINT_REGISTRY[canonical].tester_check_methods:
                methods.add(method)
    return methods


def get_all_tester_methods() -> List[str]:
    """Return all tester _check_* method names from the registry."""
    methods = []
    seen = set()
    for info in CONSTRAINT_REGISTRY.values():
        for method in info.tester_check_methods:
            if method not in seen:
                methods.append(method)
                seen.add(method)
    return methods


def get_slack_key(canonical_name: str) -> Optional[str]:
    """Get the slack dict key for a constraint, or None if not slack-aware."""
    info = CONSTRAINT_REGISTRY.get(canonical_name)
    if info:
        return info.slack_key
    return None


def get_all_canonical_names() -> List[str]:
    """Return all canonical constraint names."""
    return list(CONSTRAINT_REGISTRY.keys())


def get_tester_only_constraints() -> List[str]:
    """Return canonical names of tester-only diagnostics (no solver equivalent)."""
    return [name for name, info in CONSTRAINT_REGISTRY.items() if info.tester_only]


# ----------------------------------------------------------------------
# Helper-var catalog (Phase 2). Atoms in `required_helpers` must reference
# kinds that exist here so a typo fails at registry-validation time.
# ----------------------------------------------------------------------

HELPER_VAR_CATALOG: Set[str] = {
    'is_slot_used',           # (week, day, location, day_slot)
    'is_field_used',          # (week, day, location, field, day_slot)
    'weekend_used',           # (week, location)
    'team_plays_in_week',     # (team, week)
    'pair_plays_in_week',     # (team1, team2, week, grade)
    'pair_plays_on_date',     # (team1, team2, date, grade)
    'club_plays_on_date_at_field',  # (club, date, field)
    'club_grade_in_slot',     # (club, grade, week, day, day_slot)
    'home_game',              # (team, week, opponent)
    'is_phl_friday',          # (week,)
    'club_day_field_used',    # (club, field_name)
    'club_day_slot_used',     # (club, day_slot)
    'phl_2nd_btb_pair',       # (clubs, round_no, field, slot1, slot2) — back-to-back same-field indicator
    'nihc_field_used',        # (date, day_slot, field_name) — spec-003 NIHC fill-order shared indicator
    # spec-005 ClubVsClubStackedAlignment helpers. Parallel to the
    # `club_day_*` family but keyed per (club_pair, week) rather than
    # (club, ...) — different semantics, no collision.
    'cvc_stack_play',         # (pair, grade, week) — does this grade play this Sunday for the pair?
    'cvc_stack_field_used',   # (pair, week, field_name) — pair using this field this Sunday
    'cvc_stack_slot_used',    # (pair, week, day_slot) — pair using this slot this Sunday
    'cvc_stack_active',       # (pair, week) — stacking active this Sunday (≥ 2 grades coincide)
    # spec-021 shared contiguity primitive (constraints/atoms/_contiguity.py).
    # Anchored venue fill vs floating club spread use distinct kinds so their
    # slot_used indicators never collide.
    'venue_slot_used',        # (week, date, location, day_slot) — VenueEarliestSlotFill
    'club_spread_slot_used',  # spec-024: (club, week, day, field, day_slot) — per-field ClubGameSpread contiguity
}


def get_atoms_in_group(atom_group: str) -> List[str]:
    """Return canonical names whose atom_group matches the given group."""
    return [name for name, info in CONSTRAINT_REGISTRY.items() if info.atom_group == atom_group]


def get_adjuster(canonical_name: str) -> Optional[Callable]:
    """Return the FORCED/BLOCKED count adjuster for a constraint, or None."""
    info = CONSTRAINT_REGISTRY.get(canonical_name)
    if info is None:
        return None
    return info.forced_blocked_adjuster


def run_count_adjusters(data: Dict) -> Dict[str, Dict]:
    """Run every registered FORCED/BLOCKED count adjuster and stash results.

    For every `ConstraintInfo` with `forced_blocked_adjuster` set, calls
    `adjuster(data, forced_games, blocked_games)` and stores the result under
    `data['count_adjustments'][canonical_name]`.

    Returns the populated `data['count_adjustments']` dict (also mutated in
    place). Atoms read their entry by canonical name during `apply()`.

    Adjusters that raise are caught and re-raised with a wrapping context so
    a buggy adjuster doesn't crash the whole pipeline silently.
    """
    forced = data.get('forced_games', []) or []
    blocked = data.get('blocked_games', []) or []
    adjustments = data.setdefault('count_adjustments', {})

    for canonical_name, info in CONSTRAINT_REGISTRY.items():
        if info.forced_blocked_adjuster is None:
            continue
        try:
            result = info.forced_blocked_adjuster(data, forced, blocked)
        except Exception as e:
            raise RuntimeError(
                f"forced_blocked_adjuster for {canonical_name!r} raised: {e}"
            ) from e
        if result is not None:
            adjustments[canonical_name] = result
    return adjustments


def validate_required_helpers() -> List[str]:
    """Return list of (constraint_name, helper_kind) pairs whose helper kind is unknown."""
    bad = []
    for name, info in CONSTRAINT_REGISTRY.items():
        for kind in info.required_helpers:
            if kind not in HELPER_VAR_CATALOG:
                bad.append(f"{name} -> {kind}")
    return bad


# ======================================================================
# spec-023: Constraint groups — composable, deduped, flag-selected.
#
# A "group" is a named, possibly-overlapping set of WHOLE constraints. A solve
# applies the deduped UNION of the selected groups in one canonical order
# (= CONSTRAINT_REGISTRY dict insertion order; see resolve_groups). Selecting
# the same constraint via two groups applies it exactly once.
#
# Group membership has TWO sources, and only the first is hand-maintained:
#   1. Explicit tags — `ConstraintInfo.groups` (e.g. {'core', 'critical_feasibility'}).
#      This is the single hand-maintained membership store.
#   2. Derived predicates — `DERIVED_GROUPS` below. Membership is COMPUTED over
#      ConstraintInfo, never stored, so there is no second source of truth.
# ======================================================================

# Derived group predicates. A name maps to a predicate over ConstraintInfo;
# resolve_group computes membership on demand. `severity_1`..`severity_5` select
# by severity level.
#
# `default`/`all`/`production` select the normal FRESH-SEASON-BUILD set: every
# constraint carrying `core` or `soft`. (spec-027 changed this from the old
# `bool(info.groups)` — behaviour-identical for every constraint that existed
# before spec-027, since each carried `core` or `soft` — so that two new kinds
# of group tag are EXCLUDED from fresh builds: (a) the `core_hard`-only freeze
# pins + TeamConflict, which were never in DEFAULT_STAGES, and (b) the spec-027
# `regen_soft` atoms, which must NEVER apply in a fresh build — only in regen.)
#
# `regen` (spec-027) is the regeneration constraint set: the kept-hard physical
# atoms (`core_hard`), the soft-analogue atoms (`regen_soft`), and the always-soft
# atoms (`soft`). It deliberately does NOT select the normal hard atoms of the
# softened constraints — they carry only `{core, ...}`, never `core_hard`/
# `regen_soft`/`soft`, so this predicate excludes them automatically (this is how
# regen "softens" e.g. PHLAnd2ndAdjacency, the ClubDay atoms, ClubVsClubStacked*,
# BalancedByeSpacing, VenueEarliestSlotFill, and the engine EqualMatchUpSpacing /
# ClubGameSpread keys — by selecting their RegenSoft analogue instead of them).
def _is_fresh_build(info: 'ConstraintInfo') -> bool:
    return bool(info.groups & {'core', 'soft'})


DERIVED_GROUPS: Dict[str, Callable[[ConstraintInfo], bool]] = {
    **{
        f'severity_{n}': (lambda info, n=n: info.severity_level == n)
        for n in range(1, 6)
    },
    'default': _is_fresh_build,
    'all': _is_fresh_build,
    'production': _is_fresh_build,
    'regen': (lambda info: bool(info.groups & {'core_hard', 'regen_soft', 'soft'})),
}


def _explicit_group_names() -> Set[str]:
    """Every explicit group tag that appears on at least one ConstraintInfo."""
    names: Set[str] = set()
    for info in CONSTRAINT_REGISTRY.values():
        names |= set(info.groups)
    return names


def list_group_names() -> List[str]:
    """Return all known group names: explicit tags ∪ derived names.

    Explicit tags are listed first (sorted), then derived names (in their
    DERIVED_GROUPS declaration order), so the output is stable and readable.
    """
    explicit = sorted(_explicit_group_names())
    derived = [n for n in DERIVED_GROUPS if n not in explicit]
    return explicit + derived


def resolve_group(name: str) -> Set[str]:
    """Resolve a single group name to its set of canonical constraint names.

    A name may be an explicit tag (membership read from `ConstraintInfo.groups`)
    or a derived group (membership computed via its `DERIVED_GROUPS` predicate).
    An explicit tag and a derived predicate of the same name would both apply;
    in practice the namespaces are disjoint. Unknown names resolve to the empty
    set (callers that want strictness should check `list_group_names()` first).
    """
    members: Set[str] = set()
    # Explicit-tag membership.
    for canonical, info in CONSTRAINT_REGISTRY.items():
        if name in info.groups:
            members.add(canonical)
    # Derived-predicate membership.
    predicate = DERIVED_GROUPS.get(name)
    if predicate is not None:
        for canonical, info in CONSTRAINT_REGISTRY.items():
            if predicate(info):
                members.add(canonical)
    return members


# Canonical order = CONSTRAINT_REGISTRY dict insertion order (Open decision A:
# no per-entry `order` int; registry order suffices and already places helper-var
# producers before consumers). Index lookup for ordering and the dep validator.
_REGISTRY_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(CONSTRAINT_REGISTRY)
}


def canonical_index(canonical_name: str) -> int:
    """Registry insertion index of a canonical name (its position in apply order)."""
    return _REGISTRY_INDEX[canonical_name]


def resolve_groups(names: Iterable[str]) -> List[str]:
    """Deduped UNION of the membership of every named group, in canonical order.

    This is the heart of the dedup-union guarantee: a constraint reachable via
    two requested groups appears EXACTLY ONCE. The returned list is sorted by
    canonical (registry insertion) order, which is the single global apply order
    and already satisfies producer-before-consumer for shared helper vars
    (validated by `validate_group_order`).
    """
    union: Set[str] = set()
    for name in names:
        union |= resolve_group(name)
    return sorted(union, key=lambda n: _REGISTRY_INDEX[n])


# Known helper-var producer → consumer relationships. The producer registers the
# helper-var kind(s) the consumer reads, so the producer's apply() must run first.
# With registry-insertion order as the apply order, the producer's registry index
# must be strictly lower than the consumer's. `validate_group_order` asserts this.
#
# Each entry: (producer_canonical, consumer_canonical, [shared_helper_kinds]).
HELPER_VAR_PRODUCER_CONSUMER: List[tuple] = [
    # spec-005: Weekends registers `cvc_stack_play`; CoLocation reads it (plus
    # its own kinds). Weekends must run first. (registry: 25 < 26.)
    ('ClubVsClubStackedWeekends', 'ClubVsClubStackedCoLocation', ['cvc_stack_play']),
]


def validate_group_order() -> List[str]:
    """Assert every helper-var producer precedes its consumer in canonical order.

    Returns a list of human-readable violations ([] when OK), matching the style
    of `validate_solver_stages` / `validate_required_helpers`. A future reorder
    of CONSTRAINT_REGISTRY that places a consumer before its producer trips this.

    Two layers of checking:
      1. The hand-curated `HELPER_VAR_PRODUCER_CONSUMER` pairs.
      2. Any pair surfaced by `required_helpers` metadata: for each shared helper
         kind declared by a consumer, the producer of that kind (the lowest-index
         constraint declaring it) must precede the consumer.
    """
    violations: List[str] = []

    def _idx(name: str) -> Optional[int]:
        return _REGISTRY_INDEX.get(name)

    # Layer 1: explicit producer/consumer pairs.
    for producer, consumer, kinds in HELPER_VAR_PRODUCER_CONSUMER:
        pi, ci = _idx(producer), _idx(consumer)
        if pi is None:
            violations.append(f"unknown producer {producer!r}")
            continue
        if ci is None:
            violations.append(f"unknown consumer {consumer!r}")
            continue
        if not pi < ci:
            violations.append(
                f"{producer} (idx {pi}) must precede {consumer} (idx {ci}) "
                f"for helper kinds {kinds}"
            )

    # Layer 2: derive producer for each helper kind = the first (lowest-index)
    # constraint declaring it; every later declarer is a consumer that must
    # follow it. This catches dep pairs surfaced purely by required_helpers
    # metadata even if not hand-listed above.
    first_declarer: Dict[str, str] = {}
    for canonical, info in CONSTRAINT_REGISTRY.items():
        for kind in info.required_helpers:
            if kind not in first_declarer:
                first_declarer[kind] = canonical
    for canonical, info in CONSTRAINT_REGISTRY.items():
        for kind in info.required_helpers:
            producer = first_declarer[kind]
            if producer == canonical:
                continue
            pi, ci = _REGISTRY_INDEX[producer], _REGISTRY_INDEX[canonical]
            if not pi < ci:
                violations.append(
                    f"{producer} (idx {pi}) declares helper {kind!r} read by "
                    f"{canonical} (idx {ci}) but does not precede it"
                )
    return violations
