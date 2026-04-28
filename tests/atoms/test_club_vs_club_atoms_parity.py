"""Parity tests: ClubVsClub atoms vs the legacy `_club_alignment_hard` /
`_club_alignment_soft` methods.

The atom set is strictly stronger than the pre-atomization legacy methods:
- The legacy unified engine never enforced the PHL/2nd Sunday back-to-back
  same-field rule from `original.py:ClubVsClubAlignment` (lines 1096–1198).
  The new `PHLAnd2ndBackToBackSameField` atom adds it back. Verified via the
  per-atom violation test in `test_club_vs_club_atoms.py`.
- For the lower-grade alignment block, atoms reproduce the legacy behaviour
  exactly. This file verifies status parity on a clean fixture.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.unified import UnifiedConstraintEngine

from tests.atoms.club_vs_club_fixture import (
    build_cvc_fixture, build_model_X, solve_with_timeout,
)


def _engine_for(data):
    model, X = build_model_X(data)
    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()
    return engine, model, X


class TestClubVsClubAtomParity:
    def test_atoms_match_legacy_status_on_clean_fixture(self):
        """Atoms produce a feasible model on the clean fixture, just like
        the legacy methods (which only enforced the lower-grade block)."""
        data_a = build_cvc_fixture()
        engine_a, _, _ = _engine_for(data_a)
        engine_a._club_vs_club_atoms_hard()
        status_a, _ = solve_with_timeout(engine_a.model)

        data_l = build_cvc_fixture()
        engine_l, _, _ = _engine_for(data_l)
        engine_l._club_alignment_hard()
        status_l, _ = solve_with_timeout(engine_l.model)

        assert status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert status_l in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_atoms_add_back_to_back_block(self):
        """Atom hard count > legacy hard count because PHLAnd2ndBackToBackSameField
        adds constraints the legacy unified engine silently omitted."""
        data_a = build_cvc_fixture()
        engine_a, _, _ = _engine_for(data_a)
        atom_count = engine_a._club_vs_club_atoms_hard()

        data_l = build_cvc_fixture()
        engine_l, _, _ = _engine_for(data_l)
        legacy_count = engine_l._club_alignment_hard()

        assert atom_count > legacy_count, (
            f'atoms should add the missing PHL/2nd back-to-back block; '
            f'atoms={atom_count}, legacy={legacy_count}'
        )
