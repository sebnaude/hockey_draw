"""Phase 7b: SOLVER_STAGES config helpers.

This module provides the stage-config validation/loading layer. It does NOT
yet rewire `main_staged.py` to dispatch via these stages — the legacy
hardcoded `STAGES` and `STAGES_AI` dicts in `main_staged.py` stay live until
Phase 7c retires them. The pieces here are wired so a future commit can flip
the dispatch over by reading `data['solver_stages']` instead of the legacy
dicts.

Usage:
    from constraints.stages import load_solver_stages, validate_solver_stages

    stages = load_solver_stages(config)  # config is a season config dict
    errors = validate_solver_stages(stages)
    assert not errors, errors
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from constraints.registry import CONSTRAINT_REGISTRY


REQUIRED_KEYS = {'name', 'atoms'}
OPTIONAL_KEYS = {
    'description', 'time_limit_seconds', 'use_prior_solution_as_hint',
    'soft_only', 'requires_complete_solution',
}
ALL_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


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
