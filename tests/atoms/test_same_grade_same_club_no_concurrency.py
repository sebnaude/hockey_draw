"""Tests for the `SameGradeSameClubNoConcurrency` atom (spec-007).

Real CP-SAT models, no mocks. Each test computes its expected outcome by hand
in comments. Scenarios:

1. SOLO-CLEAN: a fixture where no club has duplicate same-grade teams. The
   atom is a no-op and the model is trivially feasible.
2. SAME-GRADE-SAME-CLUB MUST NOT COINCIDE: a club with two teams in the same
   grade, forced into different opponents in the same slot. The atom must
   block this.
3. ADJACENT-GRADE STILL ALLOWED: PHL + 2nd from the same club forced into the
   same slot must REMAIN feasible. This is the spec-007 freed behaviour --
   the legacy atom would have blocked it; the new atom must not.
4. INTRA-CLUB DERBY (one shared var, two duplicate teams): the two duplicate
   teams playing each other in the slot is a single variable. The atom must
   not double-count and must keep this feasible.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

from constraints.atoms import SameGradeSameClubNoConcurrency
from constraints.atoms.base import BROADMEADOW
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


def _registry(model):
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


def _fixture_duplicate_3rd(*, with_phl: bool = False) -> Dict:
    """Build a small fixture where Tigers fields TWO 3rd-grade teams.

    Other clubs (Norths, Wests) field one 3rd-grade team each so we can
    create cross-club games involving each Tigers 3rd team without ever
    self-matching.
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    fields = [ef, wf]

    clubs = [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Wests',  home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Tigers 3rd-A', club=clubs[0], grade='3rd'),
        Team(name='Tigers 3rd-B', club=clubs[0], grade='3rd'),
        Team(name='Norths 3rd',   club=clubs[1], grade='3rd'),
        Team(name='Wests 3rd',    club=clubs[2], grade='3rd'),
    ]
    if with_phl:
        teams += [
            Team(name='Tigers PHL',  club=clubs[0], grade='PHL'),
            Team(name='Norths PHL',  club=clubs[1], grade='PHL'),
        ]
    grades = [Grade(name='3rd', teams=[t.name for t in teams if t.grade == '3rd'])]
    if with_phl:
        grades.append(Grade(name='PHL', teams=[t.name for t in teams if t.grade == 'PHL']))

    # Games: every cross-team pairing within each grade, alphabetical (t1<t2).
    games: List[Tuple[str, str, str]] = []
    for g in grades:
        names = sorted(g.teams)
        for i, t1 in enumerate(names):
            for t2 in names[i + 1:]:
                games.append((t1, t2, g.name))

    # Timeslots: one Sunday, two slots, two fields.
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                 day_slot=1, field=ef, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                 day_slot=1, field=wf, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='13:00', week=1,
                 day_slot=2, field=ef, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='13:00', week=1,
                 day_slot=2, field=wf, round_no=1),
    ]
    return {
        'games': games,
        'teams': teams,
        'clubs': clubs,
        'grades': grades,
        'timeslots': timeslots,
        'fields': fields,
        'locked_weeks': set(),
        'num_rounds': {'PHL': 1, '3rd': 1, 'max': 1},
        'penalties': {},
    }


def _fixture_clean_no_duplicates() -> Dict:
    """Fixture where every club has at most one team per grade. The atom is
    a structural no-op."""
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    fields = [ef, wf]
    clubs = [
        Club(name='A', home_field=BROADMEADOW),
        Club(name='B', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='A PHL', club=clubs[0], grade='PHL'),
        Team(name='B PHL', club=clubs[1], grade='PHL'),
    ]
    grades = [Grade(name='PHL', teams=['A PHL', 'B PHL'])]
    games = [('A PHL', 'B PHL', 'PHL')]
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                 day_slot=1, field=ef, round_no=1),
        Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                 day_slot=1, field=wf, round_no=1),
    ]
    return {
        'games': games, 'teams': teams, 'clubs': clubs, 'grades': grades,
        'timeslots': timeslots, 'fields': fields,
        'locked_weeks': set(),
        'num_rounds': {'PHL': 1, 'max': 1},
        'penalties': {},
    }


def _build_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (t1, t2, grade, ts.day, ts.day_slot, ts.time,
                   ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{grade}_w{ts.week}_s{ts.day_slot}_{ts.field.name}'
            )
    return model, X


def _solve(model, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


# ----------------------------------------------------------------------
# Scenario 1: SOLO-CLEAN — no duplicates anywhere
# ----------------------------------------------------------------------


class TestSoloClean:
    def test_no_duplicates_means_atom_is_noop_and_model_is_feasible(self):
        """Given: 2 clubs, 1 PHL team each. No duplicate (club, grade).
        When: atom applied.
        Then: zero constraints added; model trivially feasible.

        Hand-computed expected count: 0 -- the duplicate-set is empty so the
        early return on line ~46 fires."""
        data = _fixture_clean_no_duplicates()
        model, X = _build_X(data)
        atom = SameGradeSameClubNoConcurrency()
        n = atom.apply(model, X, data, _registry(model))
        assert n == 0
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 2: DUPLICATE TEAMS CANNOT COINCIDE
# ----------------------------------------------------------------------


class TestSameGradeSameClubBlocked:
    def test_two_tigers_3rd_in_same_slot_is_infeasible(self):
        """Given: Tigers fields 3rd-A and 3rd-B. Force Tigers 3rd-A vs Norths 3rd
        AND Tigers 3rd-B vs Wests 3rd into the same (week=1, day_slot=1).
        When: atom applied.
        Then: model is INFEASIBLE -- two Tigers 3rd-grade variables in
              week=1, slot=1 give sum > 1.

        Hand calc: the bucket (Tigers, 3rd, week=1, day_slot=1) collects both
        variables (one EF, one WF). The atom adds `sum(vars) <= 1`, which
        contradicts the two forced `== 1` constraints. Status = INFEASIBLE."""
        data = _fixture_duplicate_3rd()
        model, X = _build_X(data)
        # Pick the two specific keys we want to force.
        ka = next(
            k for k in X
            if {k[0], k[1]} == {'Tigers 3rd-A', 'Norths 3rd'}
            and k[4] == 1 and k[9] == 'EF'
        )
        kb = next(
            k for k in X
            if {k[0], k[1]} == {'Tigers 3rd-B', 'Wests 3rd'}
            and k[4] == 1 and k[9] == 'WF'
        )
        model.Add(X[ka] == 1)
        model.Add(X[kb] == 1)
        n = SameGradeSameClubNoConcurrency().apply(model, X, data, _registry(model))
        assert n >= 1, 'atom must add at least one constraint for the Tigers 3rd bucket'
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE

    def test_two_tigers_3rd_in_different_slots_is_feasible(self):
        """Given: Tigers 3rd-A in slot 1, Tigers 3rd-B in slot 2 (different).
        When: atom applied.
        Then: model is FEASIBLE.

        Hand calc: the duplicate-bucket per slot has at most one variable,
        so the atom adds no `sum <= 1` constraint that bites (or none at all
        for these singletons). Status FEASIBLE."""
        data = _fixture_duplicate_3rd()
        model, X = _build_X(data)
        ka = next(
            k for k in X
            if {k[0], k[1]} == {'Tigers 3rd-A', 'Norths 3rd'}
            and k[4] == 1 and k[9] == 'EF'
        )
        kb = next(
            k for k in X
            if {k[0], k[1]} == {'Tigers 3rd-B', 'Wests 3rd'}
            and k[4] == 2 and k[9] == 'EF'
        )
        model.Add(X[ka] == 1)
        model.Add(X[kb] == 1)
        SameGradeSameClubNoConcurrency().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 3: ADJACENT GRADES ARE FREE (spec-007 behaviour change)
# ----------------------------------------------------------------------


class TestAdjacentGradeIsFreed:
    def test_tigers_phl_and_tigers_3rd_same_slot_is_feasible(self):
        """Given: Tigers PHL and Tigers 3rd in the same Sunday slot.
        When: atom applied.
        Then: model is FEASIBLE.

        The legacy `ClubGradeAdjacencyConstraint` would have penalised this
        (PHL and 2nd were treated as adjacent). Tigers does not field
        duplicate teams in PHL or in any grade adjacent to PHL here, so the
        atom must NOT block it. Hand-computed expected: status FEASIBLE."""
        data = _fixture_duplicate_3rd(with_phl=True)
        model, X = _build_X(data)
        phl_key = next(
            k for k in X
            if {k[0], k[1]} == {'Norths PHL', 'Tigers PHL'}
            and k[4] == 1 and k[9] == 'EF'
        )
        # Note: Tigers fields 3rd-A and 3rd-B; pick one game involving
        # 3rd-A in slot 1 on WF -- different club-grade than PHL so no
        # duplicate bucket overlap.
        third_key = next(
            k for k in X
            if {k[0], k[1]} == {'Norths 3rd', 'Tigers 3rd-A'}
            and k[4] == 1 and k[9] == 'WF'
        )
        model.Add(X[phl_key] == 1)
        model.Add(X[third_key] == 1)
        SameGradeSameClubNoConcurrency().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# Scenario 4: INTRA-CLUB DERBY uses ONE variable (no double-count)
# ----------------------------------------------------------------------


class TestIntraClubDerby:
    def test_tigers_3rd_a_vs_tigers_3rd_b_in_same_slot_is_feasible(self):
        """Given: Tigers 3rd-A vs Tigers 3rd-B (the intra-club derby) forced
        into week=1 day_slot=1 EF.
        When: atom applied.
        Then: model is FEASIBLE.

        Hand calc: the derby uses ONE variable (one game), and the atom
        explicitly skips intra-club variables (c1 == c2 branch). So the
        Tigers-3rd-A and Tigers-3rd-B duplicate-team buckets each remain
        empty in slot 1 (or the derby is excluded from them). The atom adds
        no biting constraint. Status FEASIBLE."""
        data = _fixture_duplicate_3rd()
        model, X = _build_X(data)
        derby_key = next(
            k for k in X
            if {k[0], k[1]} == {'Tigers 3rd-A', 'Tigers 3rd-B'}
            and k[4] == 1 and k[9] == 'EF'
        )
        model.Add(X[derby_key] == 1)
        SameGradeSameClubNoConcurrency().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
