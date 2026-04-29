# constraints/__init__.py
"""Constraint package — atom-driven (Phase 7c).

Public surface:
- `constraints.atoms` — atomized constraint implementations.
- `constraints.registry` — `CONSTRAINT_REGISTRY` + canonical-name lookup.
- `constraints.helper_vars` — `HelperVarRegistry` / `SharedVariablePool`.
- `constraints.unified` — `UnifiedConstraintEngine`.
- `constraints.stages` — SOLVER_STAGES dispatch + validation.
- `constraints.severity`, `constraints.soft`, `constraints.resolver`,
  `constraints.symmetry` — supporting subsystems.

The pre-atomization combined-constraint classes used to be re-exported
here from `constraints.original` / `constraints.ai`. Phase 7c moved both
modules into `constraints/archived/` and removed the re-export. New code
must go through atoms + registry. Tests that need the legacy classes
import from `constraints.archived.original` / `constraints.archived.ai`
directly — those paths are exempt from the lockdown test.
"""

# Severity / relaxation (no legacy-class deps; safe to keep).
from constraints.severity import (
    get_severity_level,
    group_constraints_by_severity,
    SeverityGroupState,
    SeverityGroupResolver,
    create_relaxation_test_func,
    apply_constraints_with_relaxation,
)

# Infeasibility resolver (no legacy-class deps; safe to keep).
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
