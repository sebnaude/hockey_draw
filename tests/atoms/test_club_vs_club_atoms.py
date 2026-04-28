"""Solo-clean + solo-violation tests for the 4 ClubVsClubAlignment atoms.

Each atom is exercised against the small `cvc_fixture`. ClubVsClubFieldLimit
and ClubVsClubDeficitPenalty depend on `coincide` BoolVars registered by
ClubVsClubCoincidence — the tests respect that ordering.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import (
    ClubVsClubCoincidence,
    ClubVsClubDeficitPenalty,
    ClubVsClubFieldLimit,
    PHLAnd2ndBackToBackSameField,
)
from constraints.atoms._club_vs_club_shared import (
    COINCIDE_KEY_PREFIX,
    PHL_BTB_COINCIDE_PREFIX,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_vs_club_fixture import (
    build_cvc_fixture, build_model_X, solve_with_timeout,
)


def _registry(model):
    return HelperVarRegistry(model)


# ----------------------------------------------------------------------
# ClubVsClubCoincidence
# ----------------------------------------------------------------------


class TestClubVsClubCoincidence:
    def test_solo_clean_feasible(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        n = ClubVsClubCoincidence().apply(model, X, data, registry)
        assert n >= 1, 'expect at least one min-coincidence constraint'
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_registers_coincide_var_for_other_atoms(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        ClubVsClubCoincidence().apply(model, X, data, registry)
        # Tigers/Wests pair, grades 3rd vs 4th, must have a coincide var
        # registered for at least round 1.
        coincide = registry.get(
            (COINCIDE_KEY_PREFIX, '3rd', '4th', ('Tigers', 'Wests'), 1)
        )
        assert coincide is not None


# ----------------------------------------------------------------------
# ClubVsClubFieldLimit
# ----------------------------------------------------------------------


class TestClubVsClubFieldLimit:
    def test_clean_runs_after_coincidence(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        ClubVsClubCoincidence().apply(model, X, data, registry)
        n = ClubVsClubFieldLimit().apply(model, X, data, registry)
        assert n >= 1
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_creates_field_excess_penalty_bucket(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        ClubVsClubCoincidence().apply(model, X, data, registry)
        ClubVsClubFieldLimit().apply(model, X, data, registry)
        assert 'ClubVsClubAlignmentField' in data['penalties']
        assert data['penalties']['ClubVsClubAlignmentField']['weight'] == 50000


# ----------------------------------------------------------------------
# ClubVsClubDeficitPenalty
# ----------------------------------------------------------------------


class TestClubVsClubDeficitPenalty:
    def test_clean_creates_penalty_bucket(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        ClubVsClubCoincidence().apply(model, X, data, registry)
        n = ClubVsClubDeficitPenalty().apply(model, X, data, registry)
        assert n >= 1
        assert 'ClubVsClubAlignment' in data['penalties']
        assert data['penalties']['ClubVsClubAlignment']['weight'] == 100000
        assert len(data['penalties']['ClubVsClubAlignment']['penalties']) >= 1

    def test_penalty_added_to_existing_bucket_not_overwritten(self):
        """If PHLAnd2ndBackToBackSameField runs first and pre-populates the
        bucket, DeficitPenalty must append, not overwrite."""
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        ClubVsClubCoincidence().apply(model, X, data, registry)
        PHLAnd2ndBackToBackSameField().apply(model, X, data, registry)
        bucket = data['penalties'].get('ClubVsClubAlignment')
        assert bucket is not None
        before = list(bucket['penalties'])
        ClubVsClubDeficitPenalty().apply(model, X, data, registry)
        after = data['penalties']['ClubVsClubAlignment']['penalties']
        assert len(after) >= len(before), 'PHL btb penalties were overwritten'


# ----------------------------------------------------------------------
# PHLAnd2ndBackToBackSameField
# ----------------------------------------------------------------------


class TestPHLAnd2ndBackToBackSameField:
    def test_solo_clean_feasible(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        n = PHLAnd2ndBackToBackSameField().apply(model, X, data, registry)
        assert n >= 1
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_registers_phl_btb_coincide_var(self):
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        PHLAnd2ndBackToBackSameField().apply(model, X, data, registry)
        coincide = registry.get(
            (PHL_BTB_COINCIDE_PREFIX, 'PHL', '2nd', ('Tigers', 'Wests'), 1)
        )
        assert coincide is not None

    def test_violation_phl_2nd_coincide_without_back_to_back(self):
        """Lock round 1 PHL Tigers/Wests to (slot 1, EF) and 2nd Tigers/Wests
        to (slot 4, WF) — every other round-1 PHL/2nd var off. They coincide
        (both occur in round 1) but cannot be back-to-back same-field —
        INFEASIBLE."""
        data = build_cvc_fixture()
        model, X = build_model_X(data)
        registry = _registry(model)
        phl_pin = [
            k for k in X
            if k[2] == 'PHL' and k[8] == 1 and k[9] == 'EF' and k[4] == 1
        ][0]
        snd_pin = [
            k for k in X
            if k[2] == '2nd' and k[8] == 1 and k[9] == 'WF' and k[4] == 4
        ][0]
        model.Add(X[phl_pin] == 1)
        model.Add(X[snd_pin] == 1)
        # All OTHER PHL/2nd vars in round 1 off -> only the pinned slot/field
        # combinations are eligible, ruling out any back-to-back same-field.
        for k, v in X.items():
            if k[8] != 1:
                continue
            if k[2] not in ('PHL', '2nd'):
                continue
            if k == phl_pin or k == snd_pin:
                continue
            model.Add(v == 0)
        PHLAnd2ndBackToBackSameField().apply(model, X, data, registry)
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE
