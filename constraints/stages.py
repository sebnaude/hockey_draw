"""Phase 7b: SOLVER_STAGES config + dispatch helpers.

Provides the stage-config validation/loading layer plus the dispatcher that
`main_staged.py` and `main_simple` use to apply atoms via the
`UnifiedConstraintEngine`. Atoms in the registry that don't go through the
engine (e.g. `MaximiseClubsPerTimeslotBroadmeadow`) are instantiated as
solver classes via the registry's `solver_class_names`.

Usage:
    from constraints.stages import (
        load_solver_stages, validate_solver_stages, list_stages,
        apply_solver_stage,
    )

    stages = load_solver_stages(config)  # config is a season config dict
    errors = validate_solver_stages(stages)
    assert not errors, errors
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from constraints.registry import CONSTRAINT_REGISTRY


REQUIRED_KEYS = {'name', 'atoms'}
OPTIONAL_KEYS = {
    'description', 'time_limit_seconds', 'use_prior_solution_as_hint',
    'soft_only', 'requires_complete_solution',
}
ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


# ----------------------------------------------------------------------
# Engine-key mapping. Each atom resolves to either:
#   - an engine skip-key (handled inside UnifiedConstraintEngine), or
#   - None (handled outside the engine, via legacy solver class).
#
# Engine skip-keys come from the literal strings used in
# `apply_stage_1_hard()` / `apply_stage_2_soft()` in `constraints/unified.py`.
# Phase-6 generic aliases (`NonDefaultHomeGrouping`, `AwayAtNonDefaultGrouping`)
# share their keys with the legacy Maitland-named entries.
# ----------------------------------------------------------------------

ENGINE_HARD_KEYS: Set[str] = {
    'NoDoubleBookingTeams', 'NoDoubleBookingFields', 'EqualGamesAndBalanceMatchUps',
    'FiftyFiftyHomeandAway', 'TeamConflict', 'MaxMaitlandHomeWeekends',
    'PHLAndSecondGradeAdjacency', 'PHLAndSecondGradeTimes',
    'EqualMatchUpSpacing', 'ClubGradeAdjacency', 'ClubVsClubAlignment',
    'MaitlandHomeGrouping', 'AwayAtMaitlandGrouping',
    'ClubDay', 'ClubGameSpread', 'EnsureBestTimeslotChoices',
}

ENGINE_SOFT_KEYS: Set[str] = {
    'EqualMatchUpSpacing', 'ClubGradeAdjacency', 'ClubVsClubAlignment',
    'MaitlandHomeGrouping', 'AwayAtMaitlandGrouping', 'PHLAndSecondGradeTimes',
    'PreferredTimesConstraint', 'EnsureBestTimeslotChoices', 'ClubGameSpread',
}

ALL_ENGINE_KEYS: Set[str] = ENGINE_HARD_KEYS | ENGINE_SOFT_KEYS

# Phase-6 alias canonical names share an engine key with their legacy entry.
_ENGINE_KEY_ALIASES: Dict[str, str] = {
    'NonDefaultHomeGrouping': 'MaitlandHomeGrouping',
    'AwayAtNonDefaultGrouping': 'AwayAtMaitlandGrouping',
    'PreferredTimes': 'PreferredTimesConstraint',
}


def atom_to_engine_key(atom_name: str) -> Optional[str]:
    """Return the engine skip-key for `atom_name`, or None if not engine-handled.

    Atoms with `atom_group` set return the legacy combined name (e.g.
    `PHLConcurrencyAtBroadmeadow` → `PHLAndSecondGradeTimes`). Atoms whose
    canonical name is itself an engine key return that name. Phase-6 aliases
    map via `_ENGINE_KEY_ALIASES`. Anything else returns None — those atoms
    are handled by legacy solver classes outside the engine.
    """
    if atom_name in _ENGINE_KEY_ALIASES:
        return _ENGINE_KEY_ALIASES[atom_name]
    info = CONSTRAINT_REGISTRY.get(atom_name)
    if info is None:
        return None
    if info.atom_group and info.atom_group in ALL_ENGINE_KEYS:
        return info.atom_group
    if atom_name in ALL_ENGINE_KEYS:
        return atom_name
    return None


def collect_engine_keys(atoms: List[str]) -> Tuple[Set[str], List[str]]:
    """Split a list of canonical atom names into (engine_keys, non_engine_atoms).

    `engine_keys` is the set of engine skip-keys covered by the input atoms.
    `non_engine_atoms` is the list of canonical names that fell through —
    they need legacy-class instantiation.
    """
    engine_keys: Set[str] = set()
    non_engine: List[str] = []
    for atom in atoms:
        key = atom_to_engine_key(atom)
        if key is None:
            non_engine.append(atom)
        else:
            engine_keys.add(key)
    return engine_keys, non_engine


def load_solver_stages(season_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the configured stage list, falling back to perennial DEFAULT_STAGES."""
    stages = season_config.get('solver_stages')
    if stages is None:
        from config.defaults import DEFAULT_STAGES
        stages = DEFAULT_STAGES
    # Return a deep-ish copy so callers can mutate without surprising side effects.
    return [dict(stage, atoms=list(stage.get('atoms', []))) for stage in stages]


def validate_solver_stages(stages: List[Dict[str, Any]]) -> List[str]:
    """Validate a stage list against the registry. Returns a list of error
    strings (empty list = valid).

    Rules:
    - Every stage has `name` + `atoms` (non-empty).
    - Stage names are unique.
    - Every atom in any stage is a registered canonical name.
    - No atom appears in more than one stage.
    - Optional keys are well-typed.
    """
    errors: List[str] = []
    seen_names: Set[str] = set()
    seen_atoms: Dict[str, str] = {}  # atom_name -> stage_name where first found

    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f'stage[{i}] is not a dict: {stage!r}')
            continue
        unknown_keys = set(stage) - ALL_KEYS
        if unknown_keys:
            errors.append(
                f'stage[{i}] has unknown keys: {sorted(unknown_keys)}'
            )

        name = stage.get('name')
        if not name or not isinstance(name, str):
            errors.append(f'stage[{i}] missing or non-string name')
            continue
        if name in seen_names:
            errors.append(f'duplicate stage name {name!r}')
        seen_names.add(name)

        atoms = stage.get('atoms')
        if not isinstance(atoms, list) or not atoms:
            errors.append(f'stage {name!r}: atoms must be a non-empty list')
            continue

        for atom in atoms:
            if atom not in CONSTRAINT_REGISTRY:
                errors.append(
                    f'stage {name!r}: atom {atom!r} not in CONSTRAINT_REGISTRY'
                )
            elif atom in seen_atoms and seen_atoms[atom] != name:
                errors.append(
                    f'atom {atom!r} appears in stages '
                    f'{seen_atoms[atom]!r} and {name!r}'
                )
            else:
                seen_atoms[atom] = name

    return errors


def severity_solver_stages() -> List[Dict[str, Any]]:
    """Return a SOLVER_STAGES list grouped by severity level (1=critical → 5=very low).

    Built from `CONSTRAINT_REGISTRY` so it stays in sync as atoms come and
    go. Tester-only entries are skipped, and atomized clusters surface their
    atoms (not the legacy combined name) so the dispatcher routes through
    the engine path. The result is suitable to pass to
    `StagedScheduleSolver.run_solver_stages_solve(stages_override=...)`
    when the user opts into severity-based staging via `--staged`.
    """
    from collections import defaultdict
    # Names of atom_groups whose atoms cover them (skip the combined name so
    # the dispatcher uses atoms instead of legacy classes).
    atomized_groups = {
        info.atom_group for info in CONSTRAINT_REGISTRY.values()
        if info.atom_group
    }
    by_severity: Dict[int, List[str]] = defaultdict(list)
    for name, info in CONSTRAINT_REGISTRY.items():
        if info.tester_only:
            continue
        if name in atomized_groups:
            # Skip the legacy combined name; its atoms appear separately.
            continue
        by_severity[info.severity_level].append(name)

    stages: List[Dict[str, Any]] = []
    for level in sorted(by_severity):
        stages.append({
            'name': f'severity_{level}',
            'description': f'Severity level {level} constraints',
            'atoms': sorted(by_severity[level]),
        })
    return stages


def list_stages(stages: List[Dict[str, Any]]) -> str:
    """Return a human-readable string of the configured stages."""
    lines = []
    for stage in stages:
        atoms = stage.get('atoms', [])
        soft = ' [soft-only]' if stage.get('soft_only') else ''
        desc = stage.get('description', '')
        lines.append(f"- {stage['name']}{soft}: {desc}")
        for atom in atoms:
            lines.append(f"    {atom}")
    return '\n'.join(lines)


# ----------------------------------------------------------------------
# Stage dispatch — applies a single stage's atoms onto the model. Used by
# `main_staged.py` (`run_solver_stages`) and `main_simple()`.
#
# State threaded across stages:
#   - `applied_engine_keys: Set[str]` — engine skip-keys already added to the
#     model. Prevents duplicate constraints when the same engine cluster has
#     atoms split across multiple stages.
#   - `applied_atoms: Set[str]` — canonical atom names already applied
#     (including non-engine ones). Used for reporting.
# ----------------------------------------------------------------------


def apply_solver_stage(
    stage: Dict[str, Any],
    *,
    model,
    X: Dict,
    data: Dict,
    engine,
    applied_engine_keys: Set[str],
    applied_atoms: Set[str],
) -> Tuple[int, List[str]]:
    """Apply one stage's atoms to the model.

    Returns `(constraints_added, atoms_applied_this_stage)`. Mutates
    `applied_engine_keys` and `applied_atoms` in place.
    """
    atoms = stage.get('atoms', [])
    new_atoms = [a for a in atoms if a not in applied_atoms]
    if not new_atoms:
        return 0, []

    engine_keys, non_engine = collect_engine_keys(new_atoms)
    new_engine_keys = engine_keys - applied_engine_keys

    constraints_added = 0
    soft_only = bool(stage.get('soft_only'))

    if new_engine_keys:
        # Set engine skip = everything except this stage's new engine keys.
        engine.skip_constraints = ALL_ENGINE_KEYS - new_engine_keys
        if not soft_only:
            constraints_added += engine.apply_stage_1_hard()
        constraints_added += engine.apply_stage_2_soft()
        applied_engine_keys.update(new_engine_keys)

    # Legacy-class fallback for atoms not handled by the engine.
    for atom in non_engine:
        cls = _resolve_solver_class(atom, use_ai=data.get('_use_ai', False))
        if cls is None:
            continue
        constraint = cls()
        prior = len(model.Proto().constraints)
        constraint.apply(model, X, data)
        constraints_added += len(model.Proto().constraints) - prior

    applied_atoms.update(new_atoms)
    return constraints_added, new_atoms


def _resolve_solver_class(canonical_name: str, *, use_ai: bool = False):
    """Look up a solver class by canonical name. Prefer AI variant if requested.

    Returns None for atoms with no solver-class equivalent (atom-only,
    tester-only, or aliases that share another entry's classes).
    """
    info = CONSTRAINT_REGISTRY.get(canonical_name)
    if info is None or not info.solver_class_names:
        return None
    candidates = list(info.solver_class_names)
    if use_ai:
        candidates.sort(key=lambda n: 0 if n.endswith('AI') else 1)
    else:
        candidates.sort(key=lambda n: 0 if not n.endswith('AI') else 1)
    name = candidates[0]
    return _import_solver_class(name)


def _import_solver_class(class_name: str):
    """Import a solver class by name from `constraints.original` or `constraints.ai`.

    After Phase 7c, the source modules become `constraints.archived.*` but
    the registry keeps the same `solver_class_names`. This helper survives
    the move — it just changes which module path it tries.
    """
    # Try AI module first if name ends with AI; else try original first.
    modules = ('constraints.ai', 'constraints.original')
    if not class_name.endswith('AI'):
        modules = ('constraints.original', 'constraints.ai')
    for mod_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[class_name])
        except ImportError:
            continue
        if hasattr(mod, class_name):
            return getattr(mod, class_name)
    return None
