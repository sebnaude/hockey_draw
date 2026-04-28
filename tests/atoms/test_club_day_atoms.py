"""Solo-clean + solo-violation tests for the 5 ClubDay atoms.

Mirrors the pattern of `test_phl_atoms.py`. Atoms are exercised against a
small Broadmeadow-only fixture in `club_day_fixture.py`.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import (
    ClubDayContiguousSlots,
    ClubDayIntraClubMatchup,
    ClubDayOpponentMatchup,
    ClubDayParticipation,
    ClubDaySameField,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_day_fixture import (
    CLUB_DAY_DATE_STR, OTHER_DATE_STR,
    build_club_day_fixture, build_model_X, solve_with_timeout,
)


def _registry(model):
    return HelperVarRegistry(model)


# ----------------------------------------------------------------------
# ClubDayParticipation
# ----------------------------------------------------------------------


class TestClubDayParticipation:
    def test_solo_clean_feasible(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        n = ClubDayParticipation().apply(model, X, data, _registry(model))
        assert n >= 3, 'should add ≥1 constraint per Tigers team'
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_team_absent_on_club_day(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        # Force every Tigers-1 3rd var on the club day off → team can't play.
        team = 'Tigers-1 3rd'
        for k, v in X.items():
            if k[7] == CLUB_DAY_DATE_STR and team in (k[0], k[1]):
                model.Add(v == 0)
        ClubDayParticipation().apply(model, X, data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE

    def test_no_games_on_date_raises(self):
        """If the club day date has no candidate games at all, atom raises."""
        data = build_club_day_fixture()
        # Strip every Tigers var on the club day so club_day_game_keys empties.
        data['games'] = [
            (t1, t2, g) for (t1, t2, g) in data['games']
            if 'Tigers' not in t1 and 'Tigers' not in t2
        ]
        model, X = build_model_X(data)
        try:
            ClubDayParticipation().apply(model, X, data, _registry(model))
        except ValueError as e:
            assert 'Tigers' in str(e)
        else:
            raise AssertionError('expected ValueError when no club games found')


# ----------------------------------------------------------------------
# ClubDayIntraClubMatchup
# ----------------------------------------------------------------------


class TestClubDayIntraClubMatchup:
    def test_solo_clean_forces_3rd_grade_derby(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        n = ClubDayIntraClubMatchup().apply(model, X, data, _registry(model))
        assert n >= 1, 'should add intra-club derby constraint for Tigers 3rd'
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_no_intra_club_3rd_pairing_on_day(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        # Block the Tigers-1 vs Tigers-2 3rd pair on the club day.
        for k, v in X.items():
            if (
                k[7] == CLUB_DAY_DATE_STR and k[2] == '3rd'
                and {k[0], k[1]} == {'Tigers-1 3rd', 'Tigers-2 3rd'}
            ):
                model.Add(v == 0)
        ClubDayIntraClubMatchup().apply(model, X, data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE

    def test_skipped_when_opponent_covers_grade(self):
        """When opponent has teams in the grade, no derby is forced."""
        data = build_club_day_fixture(opponent='Wests')
        model, X = build_model_X(data)
        # No constraints from this atom for 3rd (opp covers it) and only
        # if 4th has >1 host team (it doesn't), so n == 0.
        n = ClubDayIntraClubMatchup().apply(model, X, data, _registry(model))
        assert n == 0


# ----------------------------------------------------------------------
# ClubDayOpponentMatchup
# ----------------------------------------------------------------------


class TestClubDayOpponentMatchup:
    def test_no_opponent_no_constraints(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        n = ClubDayOpponentMatchup().apply(model, X, data, _registry(model))
        assert n == 0

    def test_with_opponent_forces_cross_club_games(self):
        data = build_club_day_fixture(opponent='Wests')
        model, X = build_model_X(data)
        n = ClubDayOpponentMatchup().apply(model, X, data, _registry(model))
        assert n >= 2, 'expect 3rd + 4th cross-club constraints'
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_no_cross_club_games_on_date(self):
        data = build_club_day_fixture(opponent='Wests')
        model, X = build_model_X(data)
        # Block every Tigers-vs-Wests game on the club day.
        for k, v in X.items():
            if k[7] != CLUB_DAY_DATE_STR:
                continue
            t1_club = 'Tigers' if 'Tigers' in k[0] else (
                'Wests' if 'Wests' in k[0] else None
            )
            t2_club = 'Tigers' if 'Tigers' in k[1] else (
                'Wests' if 'Wests' in k[1] else None
            )
            if {t1_club, t2_club} == {'Tigers', 'Wests'}:
                model.Add(v == 0)
        ClubDayOpponentMatchup().apply(model, X, data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE


# ----------------------------------------------------------------------
# ClubDaySameField
# ----------------------------------------------------------------------


class TestClubDaySameField:
    def test_solo_clean_feasible(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        n = ClubDaySameField().apply(model, X, data, _registry(model))
        assert n == 1
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_games_split_across_two_fields(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        # Pin one Tigers game to EF and another to WF on the club day.
        candidates_ef = [
            k for k in X if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and 'Tigers' in k[0] + k[1]
        ]
        candidates_wf = [
            k for k in X if k[7] == CLUB_DAY_DATE_STR and k[9] == 'WF'
            and 'Tigers' in k[0] + k[1] and (k[0], k[1]) != (candidates_ef[0][0], candidates_ef[0][1])
        ]
        assert candidates_ef and candidates_wf
        model.Add(X[candidates_ef[0]] == 1)
        model.Add(X[candidates_wf[0]] == 1)
        ClubDaySameField().apply(model, X, data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE


# ----------------------------------------------------------------------
# ClubDayContiguousSlots
# ----------------------------------------------------------------------


class TestClubDayContiguousSlots:
    def test_solo_clean_feasible(self):
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        n = ClubDayContiguousSlots().apply(model, X, data, _registry(model))
        # 4 sunday slots → 2 middle slots → at least 2 constraints
        assert n >= 1
        status, _ = solve_with_timeout(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_violation_gap_between_used_slots(self):
        """Pin Tigers games into slot 1 and slot 3 (slot 2 empty) — infeasible."""
        data = build_club_day_fixture()
        model, X = build_model_X(data)
        ef_slot1 = [
            k for k in X if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and k[4] == 1 and 'Tigers' in k[0] + k[1]
        ]
        ef_slot3 = [
            k for k in X if k[7] == CLUB_DAY_DATE_STR and k[9] == 'EF'
            and k[4] == 3 and 'Tigers' in k[0] + k[1]
            and (k[0], k[1]) != (ef_slot1[0][0], ef_slot1[0][1])
        ]
        assert ef_slot1 and ef_slot3
        model.Add(X[ef_slot1[0]] == 1)
        model.Add(X[ef_slot3[0]] == 1)
        # Force every Tigers var on slot 2 (any field) off → middle slot empty.
        for k, v in X.items():
            if k[7] == CLUB_DAY_DATE_STR and k[4] == 2 and 'Tigers' in k[0] + k[1]:
                model.Add(v == 0)
        ClubDayContiguousSlots().apply(model, X, data, _registry(model))
        status, _ = solve_with_timeout(model)
        assert status == cp_model.INFEASIBLE
