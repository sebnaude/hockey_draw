"""Parity test: PHL atoms vs the legacy `_phl_times_hard()` method.

Asserts the atom dispatch adds the same number of constraints, and yields the
same feasibility behavior, on a small synthesised fixture exercised with the
real `UnifiedConstraintEngine`.

Note on `MaitlandFridayCount`: legacy `_phl_times_hard()` in unified.py does NOT
enforce a Maitland Friday count, but legacy `original.py:PHLAndSecondGradeTimes`
does. Atomization restores the original behavior (the per-inventory split lists
`MaitlandFridayCount` as one of the 8 atoms). Parity tests below compare counts
modulo this single restored constraint and flag it explicitly.
"""
from __future__ import annotations

from itertools import combinations

from ortools.sat.python import cp_model

from constraints.unified import UnifiedConstraintEngine
from constraints.atoms.base import BROADMEADOW

from tests.atoms.conftest import build_model_X, solve_with_timeout


def _engine(data):
    model, X = build_model_X(data)
    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()
    return engine


class TestPHLAtomsParityWithLegacy:
    def test_atoms_and_legacy_add_same_constraint_count(self, phl_data):
        """Atom dispatch adds the same constraint count as the legacy method,
        plus one for `MaitlandFridayCount` (intentional behavior restore — see
        module docstring)."""
        engine_atoms = _engine(phl_data)
        atom_count = engine_atoms._phl_times_atoms_hard()

        from tests.atoms.conftest import _build_phl_fixture
        legacy_data = _build_phl_fixture()
        engine_legacy = _engine(legacy_data)
        legacy_count = engine_legacy._phl_times_hard()

        # Maitland Friday count is the single restored constraint.
        maitland_data = _build_phl_fixture()
        from constraints.atoms import MaitlandFridayCount
        from constraints.helper_vars import HelperVarRegistry
        m, X = build_model_X(maitland_data)
        r = HelperVarRegistry(m); r.freeze({}, {})
        maitland_added = MaitlandFridayCount().apply(m, X, maitland_data, r)

        assert atom_count == legacy_count + maitland_added, (
            f"atom={atom_count} legacy={legacy_count} maitland_added={maitland_added}"
        )

    def test_atoms_and_legacy_yield_same_feasibility(self, phl_data):
        from tests.atoms.conftest import _build_phl_fixture

        engine_a = _engine(phl_data)
        engine_a._phl_times_atoms_hard()
        status_a, _ = solve_with_timeout(engine_a.model, seconds=5.0)

        legacy_data = _build_phl_fixture()
        engine_b = _engine(legacy_data)
        engine_b._phl_times_hard()
        status_b, _ = solve_with_timeout(engine_b.model, seconds=5.0)

        assert status_a == status_b, (
            f"atoms status {status_a} != legacy status {status_b}"
        )

    def test_atoms_match_legacy_under_locked_weeks(self, phl_data):
        """Locked weeks: both atoms and legacy drop locked-week vars; atom count
        stays one above legacy due to the restored MaitlandFridayCount."""
        from tests.atoms.conftest import _build_phl_fixture

        phl_data['locked_weeks'] = {1}
        engine_a = _engine(phl_data)
        atom_count = engine_a._phl_times_atoms_hard()

        legacy_data = _build_phl_fixture()
        legacy_data['locked_weeks'] = {1}
        engine_b = _engine(legacy_data)
        legacy_count = engine_b._phl_times_hard()

        assert atom_count - legacy_count in (0, 1), (
            f"locked-week parity: atoms={atom_count} legacy={legacy_count}"
        )

    def test_atoms_full_pipeline_solves_identically(self, phl_data):
        """End-to-end: stage 1 + stage 2 with atoms vs legacy. Same status."""
        from tests.atoms.conftest import _build_phl_fixture

        engine_a = _engine(phl_data)
        engine_a.apply_stage_1_hard()
        engine_a.apply_stage_2_soft()
        status_a, _ = solve_with_timeout(engine_a.model, seconds=5.0)

        legacy_data = _build_phl_fixture()
        engine_b = _engine(legacy_data)
        # Run engine but force legacy methods for PHLAndSecondGradeTimes
        engine_b.skip_constraints = {'PHLAndSecondGradeTimes'}
        engine_b.apply_stage_1_hard()
        engine_b.apply_stage_2_soft()
        engine_b._phl_times_hard()
        engine_b._phl_times_soft()
        status_b, _ = solve_with_timeout(engine_b.model, seconds=5.0)

        assert status_a == status_b
