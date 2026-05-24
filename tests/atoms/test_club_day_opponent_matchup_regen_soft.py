"""GWT tests for ClubDayOpponentMatchupRegenSoft (spec-027 regen-soft).

No mocks, real CP-SAT models, hand-computed oracles.

The HARD atom forces cross-club host-vs-opponent matchups ONTO the club-day.
This SOFT analogue stays feasible for any X and instead emits ONE penalty unit
per cross-club opponent matchup of a shared grade scheduled OFF the club-day.

Fixture (`club_day_fixture.build_club_day_fixture`, opponent='Wests'):
- Tigers 3rd: {Tigers-1 3rd, Tigers-2 3rd}; Wests 3rd: {Wests-1 3rd, Wests-2 3rd}.
  → 4 cross-club 3rd pairs.
- Tigers 4th: {Tigers 4th}; Wests 4th: {Wests 4th} → 1 cross-club 4th pair.
- Club-day date = CLUB_DAY_DATE_STR (2026-06-14); other date = OTHER_DATE_STR.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.club_day_opponent_matchup_regen_soft import (
    ClubDayOpponentMatchupRegenSoft,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_day_fixture import (
    CLUB_DAY_DATE_STR, OTHER_DATE_STR,
    build_club_day_fixture, build_model_X, solve_with_timeout,
)


def _registry(model):
    return HelperVarRegistry(model)


def _total_penalty_expr(data):
    """Sum of all vars in the regen bucket (each var = one off-day matchup)."""
    bucket = data['penalties']['regen_club_day_opponent_matchup']
    return sum(bucket['penalties'])


def _is_tigers_wests_cross(k):
    """True iff key k is a Tigers-vs-Wests cross-club matchup (any orientation)."""
    def club_of(name):
        if name.startswith('Tigers'):
            return 'Tigers'
        if name.startswith('Wests'):
            return 'Wests'
        return None
    return {club_of(k[0]), club_of(k[1])} == {'Tigers', 'Wests'}


class TestClubDayOpponentMatchupRegenSoft:
    def test_violation_off_day_matchup_penalised(self):
        # Given: one specific cross-club 3rd matchup forced onto the OTHER date
        # (off the club-day) and blocked on the club-day, while every OTHER
        # cross-club matchup is left free to land on the club-day at penalty 0.
        data = build_club_day_fixture(opponent='Wests')
        model, X = build_model_X(data)

        # Pick the single matchup {Tigers-1 3rd, Wests-1 3rd}.
        pair = {'Tigers-1 3rd', 'Wests-1 3rd'}
        off_day_cands = [
            k for k in X
            if k[7] == OTHER_DATE_STR and k[2] == '3rd' and {k[0], k[1]} == pair
        ]
        assert off_day_cands, 'no off-day Tigers-1 vs Wests-1 3rd var'
        model.Add(X[off_day_cands[0]] == 1)
        # Forbid that pair on the club-day so it MUST live off-day.
        for k, v in X.items():
            if k[7] == CLUB_DAY_DATE_STR and k[2] == '3rd' and {k[0], k[1]} == pair:
                model.Add(v == 0)

        # When: the soft atom runs.
        n = ClubDayOpponentMatchupRegenSoft().apply(
            model, X, data, _registry(model)
        )
        assert n >= 1, 'should register penalty vars for off-day matchup candidates'

        # Then: minimising the bucket leaves exactly ONE off-day cross-club
        # matchup scheduled.
        # Hand oracle: the pair {Tigers-1 3rd, Wests-1 3rd} is pinned ON the
        # OTHER date and forbidden on the club-day → exactly one off-day cross-
        # club matchup var is 1. Every other cross-club matchup (the remaining
        # three 3rd pairs + the 4th pair) is free, so the minimiser schedules
        # them ON the club-day (or not at all) at zero penalty. Total == 1.
        model.Minimize(_total_penalty_expr(data))
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        total = solver.Value(_total_penalty_expr(data))
        assert total == 1, f'expected total penalty 1, got {total}'

    def test_clean_on_day_matchups_zero_penalty(self):
        # Given: every Tigers-vs-Wests cross-club matchup forbidden on every
        # date OTHER than the club-day, so none can land off-day.
        data = build_club_day_fixture(opponent='Wests')
        model, X = build_model_X(data)

        for k, v in X.items():
            if k[7] != CLUB_DAY_DATE_STR and _is_tigers_wests_cross(k):
                model.Add(v == 0)

        # When: the soft atom runs and we minimise the penalty bucket.
        ClubDayOpponentMatchupRegenSoft().apply(model, X, data, _registry(model))
        model.Minimize(_total_penalty_expr(data))
        status, solver = solve_with_timeout(model)

        # Then: no cross-club opponent matchup can be off the club-day →
        # total penalty == 0.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        total = solver.Value(_total_penalty_expr(data))
        assert total == 0, f'expected total penalty 0, got {total}'
