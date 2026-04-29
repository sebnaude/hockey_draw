# `constraints/archived/` — pre-atomization constraint code

These modules are the original, pre-Phase-7 implementations of the combined
constraint classes (e.g. `PHLAndSecondGradeTimes`, `ClubDayConstraint`,
`ClubVsClubAlignment`). They have been replaced by the atomized
implementations under `constraints/atoms/` and the
`UnifiedConstraintEngine` dispatcher.

## Why keep them?

- **Parity tests** (`tests/test_constraints_equivalence.py`,
  `tests/atoms/test_*_atoms_parity.py`) compare new atoms against these
  legacy implementations to confirm semantic equivalence before atomized
  code becomes the sole source of truth.
- **Reference reading** for anyone trying to understand a constraint's
  pre-atomization shape — sometimes the legacy single-method version is
  the easiest way to get a mental model.

## Lockdown rule

Production code (anything outside `constraints/archived/` and `tests/`)
**must not** import from this package. The lockdown test
`tests/test_no_legacy_imports.py` scans the repo and fails if a prod
module references `constraints.archived.*`. New constraint work must go
through:

- `constraints/atoms/` for atom implementations.
- `constraints/registry.py` for canonical-name lookup and FORCED/BLOCKED
  count adjusters.
- `constraints/stages.py` for SOLVER_STAGES dispatch and engine
  integration.
