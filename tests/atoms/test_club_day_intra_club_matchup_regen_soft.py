"""GWT tests for ClubDayIntraClubMatchupRegenSoft (spec-027 regen-soft).

No mocks, real CP-SAT models, hand-computed oracles.

The HARD atom forces same-club same-grade derbies ONTO the club-day. This SOFT
analogue stays feasible for any X and instead emits ONE penalty unit per
intra-club matchup of a governed grade scheduled OFF the club-day.

Fixture (`club_day_fixture.build_club_day_fixture`, opponent=None):
- Tigers has exactly two 3rd-grade teams: 'Tigers-1 3rd', 'Tigers-2 3rd'.
- The only intra-club 3rd matchup is the pair {Tigers-1 3rd, Tigers-2 3rd}.
- Club-day date = CLUB_DAY_DATE_STR (2026-06-14); the other available date is
  OTHER_DATE_STR (2026-06-21).
- Tigers 4th has a single team → no intra-club matchup → never penalised.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.club_day_intra_club_matchup_regen_soft import (
    ClubDayIntraClubMatchupRegenSoft,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_day_fixture import (
    CLUB_DAY_DATE_STR, OTHER_DATE_STR,
    build_club_day_fixture, build_model_X, solve_with_timeout,
)


def _registry(model):
    return HelperVarRegistry(model)


def _total_penalty_expr(data):
    """Sum of all vars across the regen bucket (each var = one off-day derby)."""
    bucket = data['penalties']['regen_club_day_intra_club_matchup']
    return sum(bucket['penalties'])


def _intra_pair_key(X, date_str):
    """The single Tigers-1 vs Tigers-2 3rd matchup key on `date_str` (slot 1, EF)."""
    cands = [
        k for k in X
        if k[7] == date_str and k[2] == '3rd'
        and {k[0], k[1]} == {'Tigers-1 3rd', 'Tigers-2 3rd'}
    ]
    assert cands, f'no intra-club 3rd matchup var on {date_str}'
    return cands[0]


class TestClubDayIntraClubMatchupRegenSoft:
    def test_violation_off_day_derby_penalised(self):
        # Given: the Tigers 3rd derby forced onto the OTHER date (not club-day),
        # and blocked on the club-day so the only place it can live is off-day.
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        off_day_key = _intra_pair_key(X, OTHER_DATE_STR)
        model.Add(X[off_day_key] == 1)
        for k, v in X.items():
            if (
                k[7] == CLUB_DAY_DATE_STR and k[2] == '3rd'
                and {k[0], k[1]} == {'Tigers-1 3rd', 'Tigers-2 3rd'}
            ):
                model.Add(v == 0)

        # When: the soft atom runs.
        n = ClubDayIntraClubMatchupRegenSoft().apply(
            model, X, data, _registry(model)
        )
        assert n >= 1, 'should register penalty vars for off-day derby candidates'

        # Then: minimising the bucket leaves exactly ONE off-day derby scheduled.
        # Hand oracle: one pair {Tigers-1 3rd, Tigers-2 3rd}; it is forced ON the
        # OTHER date and forbidden on the club-day, so exactly one off-day
        # intra-club matchup var is 1 → total penalty == 1.
        model.Minimize(_total_penalty_expr(data))
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        total = solver.Value(_total_penalty_expr(data))
        assert total == 1, f'expected total penalty 1, got {total}'

    def test_clean_on_day_derby_zero_penalty(self):
        # Given: the Tigers 3rd derby forced ONTO the club-day, and forbidden on
        # every other date so it cannot land off-day.
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        on_day_key = _intra_pair_key(X, CLUB_DAY_DATE_STR)
        model.Add(X[on_day_key] == 1)
        for k, v in X.items():
            if (
                k[7] != CLUB_DAY_DATE_STR and k[2] == '3rd'
                and {k[0], k[1]} == {'Tigers-1 3rd', 'Tigers-2 3rd'}
            ):
                model.Add(v == 0)

        # When: the soft atom runs and we minimise the penalty bucket.
        ClubDayIntraClubMatchupRegenSoft().apply(model, X, data, _registry(model))
        model.Minimize(_total_penalty_expr(data))
        status, solver = solve_with_timeout(model)

        # Then: the derby is on the club-day; no off-day derby is scheduled →
        # total penalty == 0.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        total = solver.Value(_total_penalty_expr(data))
        assert total == 0, f'expected total penalty 0, got {total}'
