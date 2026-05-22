"""
Constraint Registry -- single source of truth for constraint naming across all subsystems.

Maps between solver class names, tester check methods, unified engine skip names,
severity levels, and slack keys.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
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
    ),
    'NoDoubleBookingFields': ConstraintInfo(
        canonical_name='NoDoubleBookingFields',
        solver_class_names=['NoDoubleBookingFieldsConstraint', 'NoDoubleBookingFieldsConstraintAI'],
        tester_check_methods=['_check_no_double_booking_fields'],
        tester_violation_names=['NoDoubleBookingFields'],
        severity_level=1,
    ),
    'EqualGamesAndBalanceMatchUps': ConstraintInfo(
        canonical_name='EqualGamesAndBalanceMatchUps',
        solver_class_names=['EnsureEqualGamesAndBalanceMatchUps', 'EnsureEqualGamesAndBalanceMatchUpsAI'],
        tester_check_methods=['_check_equal_games', '_check_balanced_matchups'],
        tester_violation_names=['EqualGames', 'BalancedMatchups'],
        severity_level=1,
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
    ),
    # spec-004: per-opponent + aggregate home/away balance atom. Replaces the
    # per-pair and aggregate blocks of the obsolete `FiftyFiftyHomeandAway`.
    'AwayClubPerOpponentAndAggregateHomeBalance': ConstraintInfo(
        canonical_name='AwayClubPerOpponentAndAggregateHomeBalance',
        solver_class_names=['AwayClubPerOpponentAndAggregateHomeBalance'],
        tester_check_methods=['_check_fifty_fifty_home_away'],
        tester_violation_names=['FiftyFiftyHomeAway'],
        severity_level=1,
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
    ),
    'PHLAnd2ndConcurrencyAtBroadmeadow': ConstraintInfo(
        canonical_name='PHLAnd2ndConcurrencyAtBroadmeadow',
        solver_class_names=['PHLAnd2ndConcurrencyAtBroadmeadow'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
        atom_group='PHLAndSecondGradeTimes',
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
    ),
    'ClubDayIntraClubMatchup': ConstraintInfo(
        canonical_name='ClubDayIntraClubMatchup',
        solver_class_names=['ClubDayIntraClubMatchup'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
    ),
    'ClubDayOpponentMatchup': ConstraintInfo(
        canonical_name='ClubDayOpponentMatchup',
        solver_class_names=['ClubDayOpponentMatchup'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
    ),
    'ClubDaySameField': ConstraintInfo(
        canonical_name='ClubDaySameField',
        solver_class_names=['ClubDaySameField'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        required_helpers=['club_day_field_used'],
    ),
    'ClubDayContiguousSlots': ConstraintInfo(
        canonical_name='ClubDayContiguousSlots',
        solver_class_names=['ClubDayContiguousSlots'],
        tester_check_methods=['_check_club_day'],
        tester_violation_names=['ClubDayConstraint'],
        severity_level=2,
        atom_group='ClubDay',
        required_helpers=['club_day_slot_used'],
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
    ),
    'ClubGameSpread': ConstraintInfo(
        canonical_name='ClubGameSpread',
        solver_class_names=['ClubGameSpread', 'ClubGameSpreadAI'],
        tester_check_methods=['_check_club_game_spread'],
        tester_violation_names=['ClubGameSpread'],
        severity_level=3,
        slack_key='ClubGameSpread',
    ),
    'ClubFieldConcentration': ConstraintInfo(
        canonical_name='ClubFieldConcentration',
        solver_class_names=[],  # Tester-only diagnostic
        tester_check_methods=['_check_club_field_concentration'],
        tester_violation_names=['ClubFieldConcentration'],
        severity_level=3,
        tester_only=True,
    ),
    'MaximiseClubsPerTimeslotBroadmeadow': ConstraintInfo(
        canonical_name='MaximiseClubsPerTimeslotBroadmeadow',
        solver_class_names=['MaximiseClubsPerTimeslotBroadmeadow', 'MaximiseClubsPerTimeslotBroadmeadowAI'],
        tester_check_methods=['_check_maximise_clubs_per_timeslot_broadmeadow'],
        tester_violation_names=['MaximiseClubsPerTimeslotBroadmeadow'],
        severity_level=4,
        slack_key='MaximiseClubsPerTimeslotBroadmeadow',
    ),
    'MinimiseClubsOnAFieldBroadmeadow': ConstraintInfo(
        canonical_name='MinimiseClubsOnAFieldBroadmeadow',
        solver_class_names=['MinimiseClubsOnAFieldBroadmeadow', 'MinimiseClubsOnAFieldBroadmeadowAI'],
        tester_check_methods=['_check_minimise_clubs_on_a_field_broadmeadow'],
        tester_violation_names=['MinimiseClubsOnAFieldBroadmeadow'],
        severity_level=4,
        slack_key='MinimiseClubsOnAFieldBroadmeadow',
    ),
    'EnsureBestTimeslotChoices': ConstraintInfo(
        canonical_name='EnsureBestTimeslotChoices',
        solver_class_names=['EnsureBestTimeslotChoices', 'EnsureBestTimeslotChoicesAI'],
        tester_check_methods=['_check_ensure_best_timeslot_choices'],
        tester_violation_names=['EnsureBestTimeslotChoices'],
        severity_level=5,
    ),
    'PreferredTimes': ConstraintInfo(
        canonical_name='PreferredTimes',
        solver_class_names=['PreferredTimesConstraint', 'PreferredTimesConstraintAI'],
        tester_check_methods=['_check_preferred_times'],
        tester_violation_names=['PreferredTimesConstraint'],
        severity_level=5,
    ),
    'SoftLexMatchupOrdering': ConstraintInfo(
        canonical_name='SoftLexMatchupOrdering',
        solver_class_names=['SoftLexMatchupOrdering'],
        tester_check_methods=[],
        tester_violation_names=[],
        severity_level=5,
        has_soft_component=True,
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
    ),
    'BlockedGames': ConstraintInfo(
        canonical_name='BlockedGames',
        solver_class_names=[],  # Enforced by generate_X variable elimination, not a Constraint class
        tester_check_methods=['_check_blocked_games'],
        tester_violation_names=['BlockedGames'],
        severity_level=1,
        tester_only=True,
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
