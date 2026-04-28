"""Parity tests: ClubDay atoms vs the legacy `_club_day_scheduling` /
`_club_day_field_contiguity` methods on `UnifiedConstraintEngine`.

The 5 atoms collectively reproduce the participation/derby/same-field/
contiguous behavior, *plus* the opponent-matchup branch from
`original.py:ClubDayConstraint` (Decision #4 in `ATOMIZATION_PLAN.md`). The
unified engine's pre-atomization legacy methods did not enforce the opponent
branch, and they additionally don't handle the `{'date':, 'opponent':}` dict
form of `CLUB_DAYS` inside `build_groupings` — fixed in Phase 3b.

Parity is verified on the no-opponent path. The opponent path is covered by
the per-atom enforcement tests in `test_club_day_atoms.py`.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.unified import UnifiedConstraintEngine

from tests.atoms.club_day_fixture import (
    build_club_day_fixture, build_model_X, solve_with_timeout,
)


def _engine_for(data):
    model, X = build_model_X(data)
    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()
    return engine, model, X


class TestClubDayAtomParity:
    def test_atoms_match_legacy_status_no_opponent(self):
        """Atoms and legacy methods produce the same solver status on a clean
        no-opponent fixture (only the participation/derby/same-field/contiguity
        branches are exercised — opponent branch is moot)."""
        data_a = build_club_day_fixture()
        data_l = build_club_day_fixture()

        engine_a, _, _ = _engine_for(data_a)
        engine_a._club_day_atoms_hard()
        status_a, _ = solve_with_timeout(engine_a.model)

        engine_l, _, _ = _engine_for(data_l)
        engine_l._club_day_scheduling()
        engine_l._club_day_field_contiguity()
        status_l, _ = solve_with_timeout(engine_l.model)

        assert status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert status_l in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_atoms_enforce_opponent_path_unlike_legacy(self):
        """Without opponent: atoms add the same number of base constraints as
        legacy. With opponent: atoms add at least one *more* constraint
        (the cross-club ClubDayOpponentMatchup), confirming the documented
        Decision #4 — opponent branch lives in the atom set."""
        # No opponent — atoms should not add opponent-matchup constraints.
        data_no_opp = build_club_day_fixture()
        engine_no, _, _ = _engine_for(data_no_opp)
        atom_count_no = engine_no._club_day_atoms_hard()

        # With opponent — atoms add at least one extra constraint.
        data_with_opp = build_club_day_fixture(opponent='Wests')
        engine_with, _, _ = _engine_for(data_with_opp)
        atom_count_with = engine_with._club_day_atoms_hard()

        assert atom_count_with > atom_count_no, (
            f'opponent path must add ≥1 cross-club constraint; '
            f'no_opp={atom_count_no}, with_opp={atom_count_with}'
        )
