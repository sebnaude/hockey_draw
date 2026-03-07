# constraints/__init__.py
"""
Constraint classes for the Hockey Draw Scheduler.

This package provides backward-compatible imports for all constraint-related code.
Files were reorganized from root into this package for cleaner structure.

Modules:
- original: Original constraint implementations (from constraints.py)
- ai: AI-enhanced constraint implementations (from constraints_ai.py)  
- soft: Soft/relaxable constraint versions (from constraints_soft.py)
- severity: Severity levels and relaxation (from severity_relaxation.py)
- resolver: Infeasibility resolution (from infeasibility_resolver.py)
"""

# Original constraints (backward compatible with: from constraints import ...)
from constraints.original import (
    Constraint,
    NoDoubleBookingTeamsConstraint,
    NoDoubleBookingFieldsConstraint,
    EnsureEqualGamesAndBalanceMatchUps,
    PHLAndSecondGradeAdjacency,
    PHLAndSecondGradeTimes,
    FiftyFiftyHomeandAway,
    TeamConflictConstraint,
    MaxMaitlandHomeWeekends,
    EnsureBestTimeslotChoices,
    ClubDayConstraint,
    EqualMatchUpSpacingConstraint,
    ClubGradeAdjacencyConstraint,
    ClubVsClubAlignment,
    MaitlandHomeGrouping,
    AwayAtMaitlandGrouping,
    MaximiseClubsPerTimeslotBroadmeadow,
    MinimiseClubsOnAFieldBroadmeadow,
    PreferredTimesConstraint,
)

# AI constraints (backward compatible with: from constraints_ai import ...)
from constraints.ai import (
    ConstraintAI,
    PenaltyConfig,
    NoDoubleBookingTeamsConstraintAI,
    NoDoubleBookingFieldsConstraintAI,
    EnsureEqualGamesAndBalanceMatchUpsAI,
    PHLAndSecondGradeAdjacencyAI,
    PHLAndSecondGradeTimesAI,
    FiftyFiftyHomeandAwayAI,
    TeamConflictConstraintAI,
    MaxMaitlandHomeWeekendsAI,
    EnsureBestTimeslotChoicesAI,
    ClubDayConstraintAI,
    EqualMatchUpSpacingConstraintAI,
    ClubGradeAdjacencyConstraintAI,
    ClubVsClubAlignmentAI,
    MaitlandHomeGroupingAI,
    AwayAtMaitlandGroupingAI,
    MaximiseClubsPerTimeslotBroadmeadowAI,
    MinimiseClubsOnAFieldBroadmeadowAI,
    PreferredTimesConstraintAI,
)

# Soft constraints
from constraints.soft import (
    SoftConstraint,
    ClubDayConstraintSoft,
    AwayAtMaitlandGroupingSoft,
    TeamConflictConstraintSoft,
    EqualMatchUpSpacingConstraintSoft,
    ClubGradeAdjacencyConstraintSoft,
    ClubVsClubAlignmentSoft,
    EnsureBestTimeslotChoicesSoft,
    MaximiseClubsPerTimeslotBroadmeadowSoft,
    MinimiseClubsOnAFieldBroadmeadowSoft,
    PreferredTimesConstraintSoft,
    get_soft_constraint,
    get_soft_stage_constraints,
)

# Severity/relaxation
from constraints.severity import (
    get_severity_level,
    group_constraints_by_severity,
    SeverityGroupState,
    SeverityGroupResolver,
    create_relaxation_test_func,
    apply_constraints_with_relaxation,
)

# Infeasibility resolver
from constraints.resolver import (
    ConstraintState,
    ConstraintSlackRegistry,
    InfeasibilityResult,
    InfeasibilityResolver,
    get_constraint_names_from_stage,
    build_names_map,
    get_stage_constraints,
    get_all_constraints,
)

__all__ = [
    # Original
    'Constraint',
    'NoDoubleBookingTeamsConstraint',
    'NoDoubleBookingFieldsConstraint',
    'EnsureEqualGamesAndBalanceMatchUps',
    'PHLAndSecondGradeAdjacency',
    'PHLAndSecondGradeTimes',
    'FiftyFiftyHomeandAway',
    'TeamConflictConstraint',
    'MaxMaitlandHomeWeekends',
    'EnsureBestTimeslotChoices',
    'ClubDayConstraint',
    'EqualMatchUpSpacingConstraint',
    'ClubGradeAdjacencyConstraint',
    'ClubVsClubAlignment',
    'MaitlandHomeGrouping',
    'AwayAtMaitlandGrouping',
    'MaximiseClubsPerTimeslotBroadmeadow',
    'MinimiseClubsOnAFieldBroadmeadow',
    'PreferredTimesConstraint',
    # AI
    'ConstraintAI',
    'PenaltyConfig',
    'NoDoubleBookingTeamsConstraintAI',
    'NoDoubleBookingFieldsConstraintAI',
    'EnsureEqualGamesAndBalanceMatchUpsAI',
    'PHLAndSecondGradeAdjacencyAI',
    'PHLAndSecondGradeTimesAI',
    'FiftyFiftyHomeandAwayAI',
    'TeamConflictConstraintAI',
    'MaxMaitlandHomeWeekendsAI',
    'EnsureBestTimeslotChoicesAI',
    'ClubDayConstraintAI',
    'EqualMatchUpSpacingConstraintAI',
    'ClubGradeAdjacencyConstraintAI',
    'ClubVsClubAlignmentAI',
    'MaitlandHomeGroupingAI',
    'AwayAtMaitlandGroupingAI',
    'MaximiseClubsPerTimeslotBroadmeadowAI',
    'MinimiseClubsOnAFieldBroadmeadowAI',
    'PreferredTimesConstraintAI',
    # Soft
    'SoftConstraint',
    'ClubDayConstraintSoft',
    'AwayAtMaitlandGroupingSoft',
    'TeamConflictConstraintSoft',
    'EqualMatchUpSpacingConstraintSoft',
    'ClubGradeAdjacencyConstraintSoft',
    'ClubVsClubAlignmentSoft',
    'EnsureBestTimeslotChoicesSoft',
    'MaximiseClubsPerTimeslotBroadmeadowSoft',
    'MinimiseClubsOnAFieldBroadmeadowSoft',
    'PreferredTimesConstraintSoft',
    'get_soft_constraint',
    'get_soft_stage_constraints',
    # Severity
    'get_severity_level',
    'group_constraints_by_severity',
    'SeverityGroupState',
    'SeverityGroupResolver',
    'create_relaxation_test_func',
    'apply_constraints_with_relaxation',
    # Resolver
    'ConstraintState',
    'ConstraintSlackRegistry',
    'InfeasibilityResult',
    'InfeasibilityResolver',
    'get_constraint_names_from_stage',
    'build_names_map',
    'get_stage_constraints',
    'get_all_constraints',
]
