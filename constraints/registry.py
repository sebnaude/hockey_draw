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
    # Replaces the home-weekend logic historically in `FiftyFiftyHomeandAway`
    # + `MaxMaitlandHomeWeekends`. Reuses the existing `_check_maitland_back_
    # _to_back` tester method which observes per-club home-weekend totals.
    'AwayClubHomeWeekendsCount': ConstraintInfo(
        canonical_name='AwayClubHomeWeekendsCount',
        solver_class_names=['AwayClubHomeWeekendsCount'],
        tester_check_methods=['_check_maitland_back_to_back'],
        tester_violation_names=['MaxMaitlandHomeWeekends'],
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
    # Phase 6: `NonDefaultHomeGrouping` is the canonical name. It owns the
    # solver class names (incl. legacy `MaitlandHomeGrouping*` classes which
    # live in `constraints/archived/`) so `_SOLVER_NAME_TO_CANONICAL` resolves
    # any legacy name to the generic canonical. The `MaitlandHomeGrouping`
    # entry below is a back-compat alias kept so older configs / data dicts
    # / tests that look it up by name continue to work.
    'NonDefaultHomeGrouping': ConstraintInfo(
        canonical_name='NonDefaultHomeGrouping',
        solver_class_names=[
            'MaitlandHomeGrouping', 'MaitlandHomeGroupingAI',
            'MaxMaitlandHomeWeekends', 'MaxMaitlandHomeWeekendsAI',
        ],
        tester_check_methods=['_check_maitland_back_to_back'],
        tester_violation_names=['MaxMaitlandHomeWeekends'],
        severity_level=1,
        slack_key='MaitlandHomeGrouping',
        has_soft_component=True,
    ),
    'MaitlandHomeGrouping': ConstraintInfo(
        canonical_name='MaitlandHomeGrouping',
        # Deprecated alias for `NonDefaultHomeGrouping`. Empty class names
        # so `_SOLVER_NAME_TO_CANONICAL` routes through the canonical entry.
        # Same tester/slack/severity metadata so by-name lookups still work.
        solver_class_names=[],
        tester_check_methods=['_check_maitland_back_to_back'],
        tester_violation_names=['MaxMaitlandHomeWeekends'],
        severity_level=1,
        slack_key='MaitlandHomeGrouping',
        has_soft_component=True,
    ),
    'PHLAndSecondGradeAdjacency': ConstraintInfo(
        canonical_name='PHLAndSecondGradeAdjacency',
        solver_class_names=['PHLAndSecondGradeAdjacency', 'PHLAndSecondGradeAdjacencyAI'],
        tester_check_methods=['_check_phl_second_grade_adjacency'],
        tester_violation_names=['PHLAndSecondGradeAdjacency'],
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
    'GosfordFridayRoundsForced': ConstraintInfo(
        canonical_name='GosfordFridayRoundsForced',
        solver_class_names=['GosfordFridayRoundsForced'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
        atom_group='PHLAndSecondGradeTimes',
    ),
    'PHLRoundOnePlay': ConstraintInfo(
        canonical_name='PHLRoundOnePlay',
        solver_class_names=['PHLRoundOnePlay'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
        atom_group='PHLAndSecondGradeTimes',
    ),
    'PreferredDates': ConstraintInfo(
        canonical_name='PreferredDates',
        solver_class_names=['PreferredDates'],
        tester_check_methods=['_check_phl_second_grade_times'],
        tester_violation_names=['PHLAndSecondGradeTimes'],
        severity_level=1,
        has_soft_component=True,
        atom_group='PHLAndSecondGradeTimes',
    ),
    'EqualMatchUpSpacing': ConstraintInfo(
        canonical_name='EqualMatchUpSpacing',
        solver_class_names=['EqualMatchUpSpacingConstraint', 'EqualMatchUpSpacingConstraintAI'],
        tester_check_methods=['_check_equal_matchup_spacing'],
        tester_violation_names=['EqualMatchUpSpacing'],
        severity_level=1,
        slack_key='EqualMatchUpSpacingConstraint',
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
    # Phase 6: `AwayAtNonDefaultGrouping` is the canonical name. Owns the
    # solver class names; `AwayAtMaitlandGrouping` is a back-compat alias.
    'AwayAtNonDefaultGrouping': ConstraintInfo(
        canonical_name='AwayAtNonDefaultGrouping',
        solver_class_names=['AwayAtMaitlandGrouping', 'AwayAtMaitlandGroupingAI'],
        tester_check_methods=['_check_maitland_away_clubs_limit'],
        tester_violation_names=['AwayAtMaitlandGrouping'],
        severity_level=2,
        slack_key='AwayAtMaitlandGrouping',
    ),
    'AwayAtMaitlandGrouping': ConstraintInfo(
        canonical_name='AwayAtMaitlandGrouping',
        # Deprecated alias for `AwayAtNonDefaultGrouping`.
        solver_class_names=[],
        tester_check_methods=['_check_maitland_away_clubs_limit'],
        tester_violation_names=['AwayAtMaitlandGrouping'],
        severity_level=2,
        slack_key='AwayAtMaitlandGrouping',
    ),
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
    'NIHCFillWFBeforeEF': ConstraintInfo(
        canonical_name='NIHCFillWFBeforeEF',
        solver_class_names=['NIHCFillWFBeforeEF'],
        tester_check_methods=['_check_nihc_fill_wf_before_ef'],
        tester_violation_names=['NIHCFillWFBeforeEF'],
        severity_level=1,
        atom_group='NIHCFieldFillOrder',
        required_helpers=['nihc_field_used'],
    ),
    'NIHCFillEFBeforeSF': ConstraintInfo(
        canonical_name='NIHCFillEFBeforeSF',
        solver_class_names=['NIHCFillEFBeforeSF'],
        tester_check_methods=['_check_nihc_fill_ef_before_sf'],
        tester_violation_names=['NIHCFillEFBeforeSF'],
        severity_level=1,
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
    # --- Atoms split from ClubVsClubAlignment (Phase 3c) ---
    'ClubVsClubCoincidence': ConstraintInfo(
        canonical_name='ClubVsClubCoincidence',
        solver_class_names=['ClubVsClubCoincidence'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
        atom_group='ClubVsClubAlignment',
    ),
    'ClubVsClubFieldLimit': ConstraintInfo(
        canonical_name='ClubVsClubFieldLimit',
        solver_class_names=['ClubVsClubFieldLimit'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
        has_soft_component=True,
        atom_group='ClubVsClubAlignment',
    ),
    'ClubVsClubDeficitPenalty': ConstraintInfo(
        canonical_name='ClubVsClubDeficitPenalty',
        solver_class_names=['ClubVsClubDeficitPenalty'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
        has_soft_component=True,
        atom_group='ClubVsClubAlignment',
    ),
    'PHLAnd2ndBackToBackSameField': ConstraintInfo(
        canonical_name='PHLAnd2ndBackToBackSameField',
        solver_class_names=['PHLAnd2ndBackToBackSameField'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
        has_soft_component=True,
        atom_group='ClubVsClubAlignment',
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
