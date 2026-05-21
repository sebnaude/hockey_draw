"""Parity test: PHL atoms vs the legacy `_phl_times_hard()` method.

After the FORCED-as-count migration (see `docs/FORCED_GAMES_AS_COUNT_RULES.md`),
per-venue Friday count atoms (`BroadmeadowFridayCount`, `GosfordFridayCount`,
`MaitlandFridayCount`) are gone — those budgets live in `FORCED_GAMES` config
entries now.

After spec-010, `PHLRoundOnePlay` is also removed from `_PHL_HARD_ATOMS`. The
legacy `_phl_times_hard()` still includes it, so the gap between legacy and atom
dispatch grows by 1 per PHL team (5 teams in the fixture → +5 constraints).

After spec-015, `GosfordFridayRoundsForced` is removed from `_PHL_HARD_ATOMS`
too (the per-round `sum == 1` rule is now a FORCED_GAMES count entry). The
legacy `_phl_times_hard()` still enforces it, reading the default round set
`{2,4,5,9,10}`; the fixture has Friday-Gosford vars in weeks 2, 4 and 5 → +3
constraints in legacy that the atom path no longer adds.

Baseline gap: 2 (Broadmeadow + Gosford Friday count blocks, now FORCED_GAMES).
spec-010 additional gap: 5 (one per-team round-1 constraint in PHLRoundOnePlay).
spec-015 additional gap: 3 (Gosford-Friday-rounds {2,4,5} present in fixture).
Total expected gap (no locked weeks): 2 + 5 + 3 = 10.

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
    def test_atoms_count_below_legacy_by_venue_blocks_and_r1(self, phl_data):
        """Atom dispatch adds fewer hard constraints than legacy. Four gaps:
        1. Broadmeadow Friday cap (1 constraint) — lives in FORCED_GAMES.
        2. Gosford Friday equality (1 constraint) — lives in FORCED_GAMES.
        3. PHLRoundOnePlay: 1 constraint per PHL team (5 teams) — removed by
           spec-010. Convenor uses FORCED_GAMES for round-1 intent.
        4. GosfordFridayRoundsForced: 1 constraint per applicable round —
           removed by spec-015 (now FORCED_GAMES). Legacy reads the default
           round set {2,4,5,9,10}; the fixture has Friday-Gosford vars in
           weeks 2, 4 and 5 → 3 constraints.

        Hand-computed oracle:
          - Fixture has 5 PHL teams (Tigers, Wests, Norths, Maitland, Gosford).
          - Legacy adds 1 (Broadmeadow) + 1 (Gosford count) + 5 (round-1 per
            team) + 3 (Gosford rounds 2/4/5) = 10 more constraints than the
            current atom dispatch.
          - Total expected gap: 10.
        """
        engine_atoms = _engine(phl_data)
        atom_count = engine_atoms._phl_times_atoms_hard()

        from tests.atoms.conftest import _build_phl_fixture
        legacy_data = _build_phl_fixture()
        engine_legacy = _engine(legacy_data)
        legacy_count = engine_legacy._phl_times_hard()

        # Hand-computed: 2 (Friday-count blocks via FORCED) + 5 (PHLRoundOnePlay
        # spec-010) + 3 (GosfordFridayRoundsForced spec-015, rounds {2,4,5}) = 10.
        assert legacy_count - atom_count == 10, (
            f"expected legacy = atom + 10; got atom={atom_count} legacy={legacy_count}"
        )

    def test_atoms_solve_feasibly_on_clean_fixture(self, phl_data):
        """Without the count atoms, the four remaining atoms still produce a
        feasible model on the clean fixture."""
        engine_a = _engine(phl_data)
        engine_a._phl_times_atoms_hard()
        status_a, _ = solve_with_timeout(engine_a.model, seconds=5.0)
        assert status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_atoms_match_legacy_under_locked_weeks(self, phl_data):
        """Locked weeks {1}: both atoms and legacy drop locked-week vars.

        Hand-computed gap = 2 (Broadmeadow + Gosford Friday count blocks) + 0
        (round-1 constraints are all in week 1, which is locked → dropped) + 3
        (Gosford-Friday rounds {2,4,5} are not in week 1, so still enforced by
        legacy) = 5."""
        from tests.atoms.conftest import _build_phl_fixture

        phl_data['locked_weeks'] = {1}
        engine_a = _engine(phl_data)
        atom_count = engine_a._phl_times_atoms_hard()

        legacy_data = _build_phl_fixture()
        legacy_data['locked_weeks'] = {1}
        engine_b = _engine(legacy_data)
        legacy_count = engine_b._phl_times_hard()

        assert legacy_count - atom_count == 5

    def test_atoms_full_pipeline_matches_legacy_status(self, phl_data):
        """End-to-end: atom dispatch and legacy `_phl_times_hard()` produce
        the same solver status (feasibility outcome). They differ in
        constraint count (Friday count blocks, round-1, and Gosford rounds
        that moved to FORCED_GAMES — see the count test), but the clean
        fixture doesn't engage those caps, so feasibility is identical."""
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
