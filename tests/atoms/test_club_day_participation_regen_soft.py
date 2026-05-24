"""Solo tests for ClubDayParticipationRegenSoft (spec-027 regen-soft).

The SOFT analogue of ClubDayParticipation. Instead of forbidding a club-day
team from sitting out its club-day, it emits one penalty BoolVar per such team
that is 1 exactly when that team plays NO game on its club-day date. The model
stays FEASIBLE for any X.

No mocks — real CP-SAT models against the shared `club_day_fixture`.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms.club_day_participation_regen_soft import (
    ClubDayParticipationRegenSoft,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_day_fixture import (
    CLUB_DAY_DATE_STR,
    build_club_day_fixture, build_model_X, solve_with_timeout,
)


def _registry(model):
    return HelperVarRegistry(model)


# The three Tigers (host club) teams with candidate games on the club day.
TIGERS_TEAMS = ['Tigers-1 3rd', 'Tigers-2 3rd', 'Tigers 4th']


def _total_penalty(data, solver):
    """Sum the solved values of every penalty var in the regen bucket."""
    bucket = data['penalties']['regen_club_day_participation']
    return sum(solver.Value(v) for v in bucket['penalties'])


class TestClubDayParticipationRegenSoft:
    def test_violation_all_tigers_absent_penalty_equals_count(self):
        # GIVEN the default fixture: Tigers' club-day is 2026-06-14, and Tigers
        # field 3 teams (Tigers-1 3rd, Tigers-2 3rd, Tigers 4th) each having
        # candidate games on that date.
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        # WHEN we force EVERY Tigers var on the club-day OFF, so all 3 Tigers
        # teams are absent on their club-day. (Wests/Norths games are free.)
        for k, v in X.items():
            if k[7] == CLUB_DAY_DATE_STR and 'Tigers' in k[0] + k[1]:
                model.Add(v == 0)

        n = ClubDayParticipationRegenSoft().apply(model, X, data, _registry(model))
        # One penalty var per Tigers team that has candidate club-day games.
        assert n == 3, 'one penalty var per Tigers team'

        status, solver = solve_with_timeout(model)

        # THEN the model is still FEASIBLE (soft, never forbids), and the total
        # penalty equals the number of absent teams.
        #
        # Hand oracle: all club-day games involving any Tigers team are pinned
        # to 0. So for each of the 3 Tigers teams, sum(team_vars) == 0, which
        # pins its penalty var v == 1 (v >= 1 - 0). The other two teams' vars
        # being 0 doesn't matter — every Tigers team is absent. Total = 3.
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert _total_penalty(data, solver) == 3

    def test_clean_all_tigers_present_zero_penalty(self):
        # GIVEN the default fixture.
        data = build_club_day_fixture()
        model, X = build_model_X(data)

        # WHEN we force each Tigers team to play at least one game on the club
        # day. The two 3rd-grade Tigers play each other (covers both at once);
        # Tigers 4th plays Norths 4th on the club day.
        derby = [
            k for k in X
            if k[7] == CLUB_DAY_DATE_STR and k[2] == '3rd'
            and {k[0], k[1]} == {'Tigers-1 3rd', 'Tigers-2 3rd'}
        ]
        tigers4th = [
            k for k in X
            if k[7] == CLUB_DAY_DATE_STR and k[2] == '4th'
            and 'Tigers 4th' in (k[0], k[1])
        ]
        assert derby and tigers4th
        model.Add(X[derby[0]] == 1)
        model.Add(X[tigers4th[0]] == 1)

        ClubDayParticipationRegenSoft().apply(model, X, data, _registry(model))
        status, solver = solve_with_timeout(model)

        # THEN feasible and total penalty is 0 — every Tigers team plays on its
        # club-day, so every penalty var is pinned to 0 (v <= 1 - 1 = 0).
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert _total_penalty(data, solver) == 0
