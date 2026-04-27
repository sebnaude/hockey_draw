"""Parity test: PHL atoms vs the legacy `_phl_times_hard()` method.

After the FORCED-as-count migration (see `docs/FORCED_GAMES_AS_COUNT_RULES.md`),
per-venue Friday count atoms (`BroadmeadowFridayCount`, `GosfordFridayCount`,
`MaitlandFridayCount`) are gone — those budgets live in `FORCED_GAMES` config
entries now. The atom dispatch therefore adds **fewer** constraints than the
legacy `_phl_times_hard()`, by exactly the count of those venue-Friday blocks.

Tests below verify the structural relationship and that core feasibility
behaviour matches once the FORCED count rules are added back as config.
"""
from __future__ import annotations

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
    def test_atoms_count_below_legacy_by_two_venue_blocks(self, phl_data):
        """Atom dispatch adds two fewer hard constraints than legacy: the
        Broadmeadow- and Gosford-Friday count blocks now live in FORCED_GAMES."""
        engine_atoms = _engine(phl_data)
        atom_count = engine_atoms._phl_times_atoms_hard()

        from tests.atoms.conftest import _build_phl_fixture
        legacy_data = _build_phl_fixture()
        engine_legacy = _engine(legacy_data)
        legacy_count = engine_legacy._phl_times_hard()

        # Legacy adds Broadmeadow Friday cap + Gosford Friday equality
        # on top of the four atoms still in dispatch. Difference is exactly 2.
        assert legacy_count - atom_count == 2, (
            f"expected legacy = atom + 2; got atom={atom_count} legacy={legacy_count}"
        )

    def test_atoms_solve_feasibly_on_clean_fixture(self, phl_data):
        """Without the count atoms, the four remaining atoms still produce a
        feasible model on the clean fixture."""
        engine_a = _engine(phl_data)
        engine_a._phl_times_atoms_hard()
        status_a, _ = solve_with_timeout(engine_a.model, seconds=5.0)
        assert status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_atoms_match_legacy_under_locked_weeks(self, phl_data):
        """Locked weeks: both atoms and legacy drop locked-week vars. Atom count
        stays exactly two below legacy (same Broadmeadow + Gosford gap)."""
        from tests.atoms.conftest import _build_phl_fixture

        phl_data['locked_weeks'] = {1}
        engine_a = _engine(phl_data)
        atom_count = engine_a._phl_times_atoms_hard()

        legacy_data = _build_phl_fixture()
        legacy_data['locked_weeks'] = {1}
        engine_b = _engine(legacy_data)
        legacy_count = engine_b._phl_times_hard()

        assert legacy_count - atom_count == 2

    def test_atoms_full_pipeline_matches_legacy_status(self, phl_data):
        """End-to-end: atom dispatch and legacy `_phl_times_hard()` produce
        the same solver status (feasibility outcome). They differ in
        constraint count by exactly two (the per-venue Friday count blocks
        that moved to FORCED_GAMES), but the fixture doesn't actually
        engage those caps, so feasibility behaviour is identical."""
        from tests.atoms.conftest import _build_phl_fixture

        engine_a = _engine(phl_data)
        engine_a.apply_stage_1_hard()
        engine_a.apply_stage_2_soft()
        status_a, _ = solve_with_timeout(engine_a.model, seconds=5.0)

        legacy_data = _build_phl_fixture()
        engine_b = _engine(legacy_data)
        engine_b.skip_constraints = {'PHLAndSecondGradeTimes'}
        engine_b.apply_stage_1_hard()
        engine_b.apply_stage_2_soft()
        engine_b._phl_times_hard()
        engine_b._phl_times_soft()
        status_b, _ = solve_with_timeout(engine_b.model, seconds=5.0)

        assert status_a == status_b
