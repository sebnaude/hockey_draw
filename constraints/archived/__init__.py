"""Pre-atomization constraint implementations — historical reference only.

These modules contain the original combined constraint classes (e.g.
`PHLAndSecondGradeTimes`, `ClubDayConstraint`) that the atomized solver
replaced. They are kept here so old draws and analytics tooling can
continue to reference them, and so the parity tests in
`tests/test_constraints_equivalence.py` have something to compare the
atoms against.

**Do NOT import from this package in production code.** New constraints go
through:
  - `constraints/atoms/` — atom implementations.
  - `constraints/registry.py` — canonical-name lookup + count adjusters.
  - `constraints/stages.py` — SOLVER_STAGES dispatch.

The lockdown test `tests/test_no_legacy_imports.py` enforces this.
"""

__all__: list = []
