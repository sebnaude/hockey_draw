"""Phase 7b: SOLVER_STAGES config + dispatch helpers.

Provides the stage-config validation/loading layer plus the dispatcher that
`main_staged.py` and `main_simple` use to apply atoms via the
`UnifiedConstraintEngine`. Atoms in the registry that don't go through the
engine (e.g. `PreferredTimes`) are instantiated as
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
# spec-023: `soft_only` removed — a constraint is applied WHOLE (hard+soft
# together), never peeled into a soft-only pass. There is no longer any
# stage-level switch that suppresses the hard half of a constraint.
OPTIONAL_KEYS = {
    'description', 'time_limit_seconds', 'use_prior_solution_as_hint',
    'requires_complete_solution',
}
ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


# ----------------------------------------------------------------------
# Engine-key mapping. Each atom resolves to either:
#   - an engine skip-key (handled inside UnifiedConstraintEngine), or
#   - None (handled outside the engine, via legacy solver class).
#
# Engine skip-keys come from the literal strings used in
# `apply_stage_1_hard()` / `apply_stage_2_soft()` in `constraints/unified.py`.
# ----------------------------------------------------------------------

ENGINE_HARD_KEYS: Set[str] = {
    'NoDoubleBookingTeams', 'NoDoubleBookingFields', 'EqualGamesAndBalanceMatchUps',
    'FiftyFiftyHomeandAway', 'TeamConflict',
    # spec-014: PHL/2nd adjacency is no longer an engine key — it's the
    # `PHLAnd2ndAdjacency` atom dispatched via the non-engine fallback.
    'PHLAndSecondGradeTimes',
    'EqualMatchUpSpacing', 'ClubVsClubAlignment',
    # spec-018: `MaxMaitlandHomeWeekends` / `MaitlandHomeGrouping` /
    # `AwayAtMaitlandGrouping` engine keys deleted (venue-sequencing rules
    # removed). Per-club home-weekend counts are the spec-004
    # `AwayClubHomeWeekendsCount` atom (dispatched via the non-engine fallback).
    'ClubDay', 'ClubGameSpread',
    # spec-021: `EnsureBestTimeslotChoices` engine key removed — replaced by the
    # non-engine `VenueEarliestSlotFill` atom (dispatched via the fallback).
    # spec-007: `ClubGradeAdjacency` removed from the engine. Hard portion is
    # now the `SameGradeSameClubNoConcurrency` atom dispatched via the
    # non-engine legacy-class fallback; soft adjacent-grade rule was removed.
}

ENGINE_SOFT_KEYS: Set[str] = {
    'EqualMatchUpSpacing', 'ClubVsClubAlignment',
    # spec-018: `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` soft keys
    # deleted alongside their hard keys.
    'PHLAndSecondGradeTimes',
    'PreferredTimesConstraint', 'ClubGameSpread',
    # spec-021: `EnsureBestTimeslotChoices` soft key removed (BestTimeslotWF
    # penalty deleted; WF order owned by NIHCFillWFBeforeEF).
    # spec-007: `ClubGradeAdjacency` soft penalty removed entirely.
}

ALL_ENGINE_KEYS: Set[str] = ENGINE_HARD_KEYS | ENGINE_SOFT_KEYS

# Alias canonical names that share an engine key with another entry.
# spec-018: `NonDefaultHomeGrouping` / `AwayAtNonDefaultGrouping` aliases
# removed (the rules they pointed at were deleted).
_ENGINE_KEY_ALIASES: Dict[str, str] = {
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
    """Validate a stage/group list against the registry. Returns a list of error
    strings (empty list = valid).

    spec-023 (DoD 6): the no-atom-in-two-stages (no-overlap) rule is REMOVED —
    a constraint may legally appear in more than one group/stage, because a
    solve applies the deduped UNION of selected groups. Validation instead
    checks:
    - Every stage has `name` + `atoms` (non-empty); stage names are unique.
    - Every member of any stage is a registered canonical name OR a resolvable
      group name (so a stage may reference a group by name).
    - The canonical-order helper-var producer/consumer check passes
      (`validate_group_order` from Unit A).
    - Optional keys are well-typed.
    """
    from constraints.registry import (
        list_group_names as _list_group_names,
        validate_group_order as _validate_group_order,
    )

    errors: List[str] = []
    seen_names: Set[str] = set()
    group_names = set(_list_group_names())

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

        # Each member must be a registered canonical name or a resolvable
        # group name. Overlap across stages is now legal — no dedup check.
        for atom in atoms:
            if atom in CONSTRAINT_REGISTRY:
                continue
            if atom in group_names:
                continue
            errors.append(
                f'stage {name!r}: atom {atom!r} is neither a registered '
                f'canonical name nor a resolvable group name'
            )

    # Canonical-order helper-dep check (DoD 3 / Unit A). A registry reorder that
    # places a helper-var consumer before its producer trips this.
    errors.extend(_validate_group_order())

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
        desc = stage.get('description', '')
        lines.append(f"- {stage['name']}: {desc}")
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


def apply_constraint_set(
    canonical_names: List[str],
    *,
    model,
    X: Dict,
    data: Dict,
    engine,
    applied_engine_keys: Set[str],
    applied_atoms: Set[str],
) -> Tuple[int, List[str]]:
    """Apply a resolved/ordered/deduped list of WHOLE constraints to the model.

    spec-023: this is the single dispatch entry point. Each constraint is
    applied whole — there is no soft-only pass:
      - engine keys run `apply_stage_1_hard()` AND `apply_stage_2_soft()`;
      - non-engine atoms run their full `apply(model, X, data, registry)`.

    `canonical_names` should already be deduped and in canonical (registry
    insertion) order — see `constraints.registry.resolve_groups`. This function
    additionally skips any name already in `applied_atoms` (and any engine key
    already in `applied_engine_keys`), so calling it across several stages never
    double-applies a constraint.

    Returns `(constraints_added, names_applied_this_call)`. Mutates
    `applied_engine_keys` and `applied_atoms` in place.
    """
    new_names = [a for a in canonical_names if a not in applied_atoms]
    if not new_names:
        return 0, []

    engine_keys, non_engine = collect_engine_keys(new_names)
    new_engine_keys = engine_keys - applied_engine_keys

    constraints_added = 0

    if new_engine_keys:
        # Set engine skip = everything except this call's new engine keys.
        # spec-023: ALWAYS run hard AND soft — engine keys are applied whole.
        engine.skip_constraints = ALL_ENGINE_KEYS - new_engine_keys
        constraints_added += engine.apply_stage_1_hard()
        constraints_added += engine.apply_stage_2_soft()
        applied_engine_keys.update(new_engine_keys)

    # Legacy-class fallback for atoms not handled by the engine.
    #
    # For Atom subclasses we MUST share a single helper-var registry across
    # every atom in the same call: the spec-005
    # `ClubVsClubStackedCoLocation` reads helper vars registered by
    # `ClubVsClubStackedWeekends`, so an ephemeral-per-atom registry would
    # silently break the cross-atom lookup. Prefer the engine's registry
    # (used when we mix engine + atom dispatch); fall back to a single
    # ephemeral registry built once per call.
    from constraints.atoms.base import Atom as _AtomBase
    stage_registry = (
        getattr(engine, 'helper_registry', None)
        or getattr(engine, 'registry', None)
    )
    if stage_registry is None:
        stage_registry = _ephemeral_registry(model)

    for atom in non_engine:
        cls = _resolve_solver_class(atom, use_ai=data.get('_use_ai', False))
        if cls is None:
            continue
        constraint = cls()
        prior = len(model.Proto().constraints)
        # New atoms (subclasses of `constraints.atoms.base.Atom`) take a
        # helper-var registry as a fourth arg. Legacy classes take three.
        if isinstance(constraint, _AtomBase):
            constraint.apply(model, X, data, stage_registry)
        else:
            constraint.apply(model, X, data)
        constraints_added += len(model.Proto().constraints) - prior

    applied_atoms.update(new_names)
    return constraints_added, new_names


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
    """Apply one stage's atoms to the model (thin wrapper over
    `apply_constraint_set`).

    spec-023: a stage is just a named list of WHOLE constraints. This resolves
    the stage's `atoms` and hands them to `apply_constraint_set`, which always
    runs hard+soft for engine keys (the `soft_only` switch is gone). Returns
    `(constraints_added, atoms_applied_this_stage)`; mutates the two
    `applied_*` sets in place.
    """
    return apply_constraint_set(
        list(stage.get('atoms', [])),
        model=model, X=X, data=data, engine=engine,
        applied_engine_keys=applied_engine_keys,
        applied_atoms=applied_atoms,
    )


def _ephemeral_registry(model):
    """Build a fresh `HelperVarRegistry` for atoms dispatched outside the engine.

    Atoms create shared helpers lazily via the pool-style API inside `apply()`,
    so a fresh registry per call is sufficient — no shared state is needed. This
    avoids requiring the engine to expose a registry attribute when the atom
    dispatches purely outside it.
    """
    from constraints.helper_vars import HelperVarRegistry
    return HelperVarRegistry(model)


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
    """Import a solver class by name from `constraints.original`, `.ai`, or `.atoms`.

    Phase 7c moved legacy classes under `constraints/archived/`. For atoms
    born directly under `constraints/atoms/` (e.g. spec-007's
    `SameGradeSameClubNoConcurrency` and `TeamPairNoConcurrency`) the lookup
    also tries the atoms package — those classes are subclasses of
    `constraints.atoms.base.Atom` and the caller dispatches them with the
    atom-shaped four-arg `apply` signature.
    """
    # Try AI module first if name ends with AI; else try original first.
    modules = ('constraints.ai', 'constraints.original', 'constraints.atoms')
    if not class_name.endswith('AI'):
        modules = ('constraints.original', 'constraints.ai', 'constraints.atoms')
    for mod_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[class_name])
        except ImportError:
            continue
        if hasattr(mod, class_name):
            return getattr(mod, class_name)
    return None
