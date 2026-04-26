"""
Constraint Registry -- single source of truth for constraint naming across all subsystems.

Maps between solver class names, tester check methods, unified engine skip names,
severity levels, and slack keys.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
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
    'FiftyFiftyHomeandAway': ConstraintInfo(
        canonical_name='FiftyFiftyHomeandAway',
        solver_class_names=['FiftyFiftyHomeandAway', 'FiftyFiftyHomeandAwayAI'],
        tester_check_methods=['_check_fifty_fifty_home_away'],
        tester_violation_names=['FiftyFiftyHomeAway'],
        severity_level=1,
    ),
    'MaitlandHomeGrouping': ConstraintInfo(
        canonical_name='MaitlandHomeGrouping',
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
    'AwayAtMaitlandGrouping': ConstraintInfo(
        canonical_name='AwayAtMaitlandGrouping',
        solver_class_names=['AwayAtMaitlandGrouping', 'AwayAtMaitlandGroupingAI'],
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
    'ClubGradeAdjacency': ConstraintInfo(
        canonical_name='ClubGradeAdjacency',
        solver_class_names=['ClubGradeAdjacencyConstraint', 'ClubGradeAdjacencyConstraintAI'],
        tester_check_methods=['_check_club_grade_adjacency'],
        tester_violation_names=['ClubGradeAdjacency'],
        severity_level=3,
    ),
    'ClubVsClubAlignment': ConstraintInfo(
        canonical_name='ClubVsClubAlignment',
        solver_class_names=['ClubVsClubAlignment', 'ClubVsClubAlignmentAI'],
        tester_check_methods=['_check_club_vs_club_alignment'],
        tester_violation_names=['ClubVsClubAlignment'],
        severity_level=3,
        slack_key='ClubVsClubAlignment',
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
