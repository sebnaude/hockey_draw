"""GWT tests for ClubDaySameFieldRegenSoft.

No mocks.  Two scenarios with hand-computed oracles:

Scenario 1 — violation (penalty == 1):
  Force one Tigers game onto EF and a *different* Tigers game onto WF on the
  club-day date.  The matchup spans 2 fields → penalty = 2 - 1 = 1.
  Expected: FEASIBLE, total penalty == 1.

Scenario 2 — clean (penalty == 0):
  Force both the EF game AND the WF game onto EF only (block every WF Tigers var).
  All games land on a single field → penalty = 1 - 1 = 0.
  Expected: FEASIBLE, total penalty == 0.

Oracle derivation for Scenario 1
---------------------------------
Let F = {EF, WF} be the fields that have Tigers variables on CLUB_DAY_DATE_STR.
We force:
  - X[ef_key] = 1   (a Tigers game on EF)
  - X[wf_key] = 1   (a *different* Tigers game on WF)
After solving, ef_used = max(EF-vars) = 1 and wf_used = max(WF-vars) = 1.
  sum(field_used_indicators) = ef_used + wf_used = 2
  pen >= 2 - 1 = 1  →  pen == 1  (pen ∈ [0, 1], pushed to 1 by the constraint)
Total penalty == 1.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.club_day_same_field_regen_soft import (
    ClubDaySameFieldRegenSoft,
    REGEN_CLUB_DAY_SAME_FIELD_DEFAULT_WEIGHT,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_day_fixture import (
    CLUB_DAY_DATE_STR,
    build_club_day_fixture,
    build_model_X,
    solve_with_timeout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atom():
    return ClubDaySameFieldRegenSoft()


def _registry(model):
    return HelperVarRegistry(model)


def _total_penalty(solver, data) -> int:
    """Sum up all penalty variable values from the bucket."""
    bucket = data.get('penalties', {}).get('regen_club_day_same_field', {})
    return sum(solver.Value(v) for v in bucket.get('penalties', []))


def _pick_tigers_key(X, date_str: str, field_name: str,
                     exclude_pair=None) -> tuple:
    """Return the first Tigers key on `date_str`/`field_name`, optionally
    excluding a specific (team1, team2) pair."""
    for key in X:
        if len(key) < 11 or not key[3]:
            continue
        if key[7] != date_str or key[9] != field_name:
            continue
        if 'Tigers' not in key[0] + key[1]:
            continue
        if exclude_pair is not None and (key[0], key[1]) == exclude_pair:
            continue
        return key
    raise AssertionError(
        f'No Tigers key found for date={date_str} field={field_name}'
        + (f' excluding {exclude_pair}' if exclude_pair else '')
    )


# ---------------------------------------------------------------------------
# Scenario 1: violation — two different fields used → penalty == 1
# ---------------------------------------------------------------------------

class TestScenario1Violation:
    """
    Given: A Tigers club-day with games available on both EF and WF.
    When:  One Tigers game is forced onto EF and a *different* Tigers game is
           forced onto WF.
    Then:  The model is FEASIBLE and the total regen_club_day_same_field penalty
           equals 1 (two fields used: 2 - 1 = 1).
    """

    def test_feasible_with_penalty_one(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        # Pick one Tigers game on EF.
        ef_key = _pick_tigers_key(X, CLUB_DAY_DATE_STR, 'EF')
        # Pick a DIFFERENT Tigers game on WF.
        wf_key = _pick_tigers_key(
            X, CLUB_DAY_DATE_STR, 'WF',
            exclude_pair=(ef_key[0], ef_key[1]),
        )

        # Hand oracle: ef_key is on EF, wf_key is on WF, they are different
        # matchups.  After forcing both to 1:
        #   ef_used = max(all EF Tigers vars) = 1
        #   wf_used = max(all WF Tigers vars) = 1
        #   sum(indicators) = 2 → pen ≥ 1 → pen == 1.
        model.Add(X[ef_key] == 1)
        model.Add(X[wf_key] == 1)

        n = _atom().apply(model, X, data, _registry(model))
        assert n == 1, f'expected 1 penalty term emitted, got {n}'

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            'model must remain FEASIBLE even when games are split across fields'
        )

        penalty = _total_penalty(solver, data)
        assert penalty == 1, (
            f'Hand oracle: 2 fields used → penalty = 2-1 = 1, got {penalty}'
        )

    def test_bucket_registered(self):
        """Bucket is created in data['penalties'] with correct weight."""
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        _atom().apply(model, X, data, _registry(model))
        bucket = data.get('penalties', {}).get('regen_club_day_same_field')
        assert bucket is not None, 'bucket must be registered in data[penalties]'
        assert bucket['weight'] == REGEN_CLUB_DAY_SAME_FIELD_DEFAULT_WEIGHT
        assert len(bucket['penalties']) == 1, 'one penalty IntVar per matchup'

    def test_weight_zero_returns_zero_and_no_bucket(self):
        """When weight == 0, atom is a no-op and no bucket is created."""
        data = build_club_day_fixture()
        data['penalty_weights']['regen_club_day_same_field'] = 0
        model, X = build_model_X(data)
        n = _atom().apply(model, X, data, _registry(model))
        assert n == 0
        assert 'regen_club_day_same_field' not in data.get('penalties', {})


# ---------------------------------------------------------------------------
# Scenario 2: clean — all games on the same field → penalty == 0
# ---------------------------------------------------------------------------

class TestScenario2Clean:
    """
    Given: A Tigers club-day with games available on both EF and WF.
    When:  All WF Tigers variables are forced to 0 (cannot schedule on WF) and
           one EF Tigers game is forced to 1.
    Then:  The model is FEASIBLE and the total penalty is 0 (only one field
           used: 1 - 1 = 0).
    """

    def test_feasible_with_penalty_zero(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        # Block every Tigers WF game on the club day → forces single-field use.
        for key, var in X.items():
            if (
                key[7] == CLUB_DAY_DATE_STR
                and key[9] == 'WF'
                and 'Tigers' in key[0] + key[1]
            ):
                model.Add(var == 0)

        # Ensure at least one EF game is scheduled so the atom sees activity.
        ef_key = _pick_tigers_key(X, CLUB_DAY_DATE_STR, 'EF')
        model.Add(X[ef_key] == 1)

        n = _atom().apply(model, X, data, _registry(model))
        assert n == 1, f'expected 1 penalty term emitted, got {n}'

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            'model must be FEASIBLE when games are on a single field'
        )

        penalty = _total_penalty(solver, data)
        assert penalty == 0, (
            f'Hand oracle: 1 field used → penalty = 1-1 = 0, got {penalty}'
        )
