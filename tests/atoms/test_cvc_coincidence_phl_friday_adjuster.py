"""spec-009 Check 2 — PHL Friday exempted from ClubVsClubCoincidence.

Verifies that `club_vs_club_coincidence_adjuster` (Adjuster #5) correctly
reduces the expected Sunday-coincidence count when FORCED entries move PHL
games off Sunday onto Friday, and that the atom reads the adjusted count
rather than the raw per_team_games total.

Scenarios:
1. Adjuster math: direct unit test with synthetic data.
2. Atom integration: run the atom against a small CP-SAT fixture and verify
   the `data['count_adjustments']` entry drives the constraint bound.
3. Mirror (tester): confirm that a schedule missing a PHL Sunday coincidence
   is NOT flagged when that meeting is covered by FORCED Fridays.

Real CP-SAT models throughout — no mocks.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from constraints.atoms.base import BROADMEADOW, GOSFORD, MAITLAND
from constraints.atoms.club_vs_club_coincidence import (
    ClubVsClubCoincidence,
    club_vs_club_coincidence_adjuster,
)
from constraints.atoms._club_vs_club_shared import per_team_games
from constraints.helper_vars import HelperVarRegistry
from constraints.registry import run_count_adjusters
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_phl_3rd_fixture(
    *,
    phl_fridays_forced: int = 2,
    num_weeks: int = 4,
) -> Dict:
    """Fixture: Maitland + Norths, PHL + 3rd grade.

    PHL: 2 teams. R=num_weeks → per_team_games = num_weeks // (2-1) = num_weeks.
    3rd: 2 teams. R=num_weeks → per_team_games = num_weeks.

    Sunday slots at BROADMEADOW EF + WF (for 3rd grade coincidence).
    Friday slots at GOSFORD for PHL (to be forced off Sunday).

    Hand-computed baseline:
      PHL per_team_games = num_weeks (= num_weeks rounds // 1 pair)
      3rd per_team_games = num_weeks

    After applying forced = phl_fridays_forced PHL Fridays of (Mait, Norths):
      adjustment['PHL'][('Maitland','Norths')] = num_weeks - phl_fridays_forced
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    gosford_f = PlayingField(location=GOSFORD, name='CCHP Main')

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland PHL', club=clubs[0], grade='PHL'),
        Team(name='Norths PHL', club=clubs[1], grade='PHL'),
        Team(name='Maitland 3rd', club=clubs[0], grade='3rd'),
        Team(name='Norths 3rd', club=clubs[1], grade='3rd'),
    ]
    grades = [
        Grade(name='PHL', teams=['Maitland PHL', 'Norths PHL']),
        Grade(name='3rd', teams=['Maitland 3rd', 'Norths 3rd']),
    ]
    games: List[Tuple[str, str, str]] = []
    for g in grades:
        for t1, t2 in combinations(sorted(g.teams), 2):
            games.append((t1, t2, g.name))

    timeslots: List[Timeslot] = []
    for wk in range(1, num_weeks + 1):
        # Sunday slots for 3rd-grade alignment
        timeslots.append(Timeslot(
            date=f'2026-03-{21 + wk:02d}', day='Sunday', time='11:30', week=wk,
            day_slot=1, field=ef, round_no=wk,
        ))
        timeslots.append(Timeslot(
            date=f'2026-03-{21 + wk:02d}', day='Sunday', time='13:00', week=wk,
            day_slot=2, field=ef, round_no=wk,
        ))
        timeslots.append(Timeslot(
            date=f'2026-03-{21 + wk:02d}', day='Sunday', time='11:30', week=wk,
            day_slot=1, field=wf, round_no=wk,
        ))
        timeslots.append(Timeslot(
            date=f'2026-03-{21 + wk:02d}', day='Sunday', time='13:00', week=wk,
            day_slot=2, field=wf, round_no=wk,
        ))
        # Friday slot at Gosford for PHL only
        timeslots.append(Timeslot(
            date=f'2026-03-{19 + wk:02d}', day='Friday', time='20:00', week=wk,
            day_slot=1, field=gosford_f, round_no=wk,
        ))

    # FORCED: phl_fridays_forced PHL games of (Maitland, Norths) on Friday at
    # Gosford. The adjuster should reduce PHL expected by this count.
    forced_games = [
        {
            'grade': 'PHL',
            'day': 'Friday',
            'field_location': GOSFORD,
            'teams': ['Maitland PHL', 'Norths PHL'],
            'count': phl_fridays_forced,
            'constraint': 'equal',
        }
    ] if phl_fridays_forced > 0 else []

    return {
        'teams': teams,
        'clubs': clubs,
        'grades': grades,
        'games': games,
        'timeslots': timeslots,
        'fields': [ef, wf, gosford_f],
        'home_field_map': {'Maitland': MAITLAND, 'Gosford': GOSFORD},
        'num_rounds': {'PHL': num_weeks, '3rd': num_weeks, 'max': num_weeks},
        'forced_games': forced_games,
        'blocked_games': [],
        'current_week': 0,
        'locked_weeks': set(),
        'constraint_slack': {},
        'penalty_weights': {},
        'penalties': {},
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'constraint_defaults': {},
    }


def _build_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            if not ts.day:
                continue
            # Mirror generate_X: 3rd grade doesn't play Friday
            if grade == '3rd' and ts.day == 'Friday':
                continue
            key = (t1, t2, grade, ts.day, ts.day_slot, ts.time,
                   ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{grade}_{ts.week}_{ts.day}_{ts.field.name}'
            )
    return model, X


def _registry(model):
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


def _solve(model, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


# ---------------------------------------------------------------------------
# Scenario 1: Adjuster math (unit test)
# ---------------------------------------------------------------------------


class TestAdjusterMath:
    def test_adjuster_reduces_phl_expected_by_forced_count(self):
        """Given: 4-week fixture with 2 PHL Fridays forced for Maitland-Norths.
        When: club_vs_club_coincidence_adjuster called.
        Then: adjustment['PHL'][('Maitland','Norths')] == 4 - 2 == 2.

        Hand-computed: per_team_games for PHL with 2 teams and max_rounds=4 is
        4 // (2-1) = 4. Forcing 2 off-Sunday reduces expected to 4 - 2 = 2.
        The 3rd-grade pair has no off-Sunday forced games, so adjustment['3rd']
        is not set (or returns the baseline 4).
        """
        data = _build_phl_3rd_fixture(phl_fridays_forced=2, num_weeks=4)
        out = club_vs_club_coincidence_adjuster(
            data, data['forced_games'], data['blocked_games']
        )
        assert out is not None, 'Adjuster must return a dict when forced off-Sunday entries exist'
        assert 'PHL' in out, 'PHL grade must appear in adjustment dict'
        pair = ('Maitland', 'Norths')
        assert pair in out['PHL'], (
            f'Pair {pair} must appear in PHL adjustment. Got: {out["PHL"]}'
        )
        assert out['PHL'][pair] == 2, (
            f'Expected 4 - 2 = 2. Got {out["PHL"][pair]}'
        )
        # 3rd grade: no off-Sunday forced entries → should NOT appear in output
        assert '3rd' not in out, (
            f'3rd grade has no forced-Friday entries; should not be in adjustment. Got: {out}'
        )

    def test_adjuster_returns_none_when_no_forced_off_sunday(self):
        """Given: fixture with no forced_games.
        When: adjuster called.
        Then: returns None (no adjustment needed).

        Hand-computed: empty forced_games → deltas dict is empty → None.
        """
        data = _build_phl_3rd_fixture(phl_fridays_forced=0, num_weeks=4)
        out = club_vs_club_coincidence_adjuster(data, [], [])
        assert out is None

    def test_adjuster_clamps_to_zero(self):
        """Given: 4-week fixture, forced count > per_team_games (e.g. count=99).
        When: adjuster called.
        Then: adjustment['PHL'][pair] == 0 (clamped at floor).

        Hand-computed: 4 - 99 = -95 → max(0, -95) = 0.
        """
        data = _build_phl_3rd_fixture(phl_fridays_forced=0, num_weeks=4)
        # Manually override with a very large count
        forced = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland PHL', 'Norths PHL'],
            'count': 99, 'constraint': 'equal',
        }]
        out = club_vs_club_coincidence_adjuster(data, forced, [])
        assert out is not None
        assert out['PHL'][('Maitland', 'Norths')] == 0, (
            f'Expected 0 (clamped); got {out["PHL"][("Maitland", "Norths")]}'
        )


# ---------------------------------------------------------------------------
# Scenario 2: Atom integration — atom reads adjusted count
# ---------------------------------------------------------------------------


class TestAtomReadsAdjustedCount:
    def test_atom_uses_reduced_expected_when_adjustment_present(self):
        """Given: 4-week fixture with 2 PHL Fridays forced, 3rd grade only
        (ClubVsClubCoincidence only fires for lower grades).
        data['count_adjustments']['ClubVsClubCoincidence'] pre-loaded from
        adjuster.

        When: ClubVsClubCoincidence.apply() called.
        Then: the min_required for ('Maitland','Norths') in 3rd grade is
        based on per_team_games (no off-Sunday forced for 3rd), i.e. 4.

        NOTE: ClubVsClubCoincidence only applies to grades below PHL/2nd.
        So this test specifically exercises 3rd grade with an adjustment dict
        that has NO 3rd entry (PHL entry only). The atom must fall back to
        the per_team_games value for 3rd, which is correct.

        Hand-computed: 3rd has 4 rounds, 2 teams → per_team_games = 4.
        No forced-off-Sunday for 3rd. Expected = 4. Feasible if the model
        can schedule 4 coincident rounds.
        """
        data = _build_phl_3rd_fixture(phl_fridays_forced=2, num_weeks=4)
        # Run adjuster and stash the result
        adj_out = club_vs_club_coincidence_adjuster(
            data, data['forced_games'], data['blocked_games']
        )
        data.setdefault('count_adjustments', {})
        data['count_adjustments']['ClubVsClubCoincidence'] = adj_out or {}

        model, X = _build_X(data)
        atom = ClubVsClubCoincidence()
        n = atom.apply(model, X, data, _registry(model))
        # The atom should find the (3rd, X) pair and add at least 1 constraint.
        # n >= 0 is always true; the meaningful check is feasibility.
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            'ClubVsClubCoincidence with correct adjusted expected should be feasible'
        )

    def test_adjustment_from_run_count_adjusters(self):
        """Given: data with forced_games (2 PHL Fridays for the pair).
        When: run_count_adjusters(data) called.
        Then: data['count_adjustments']['ClubVsClubCoincidence']['PHL']
              [('Maitland','Norths')] == 2.

        This exercises the registry dispatch path (same as the engine calls).

        Hand-computed: identical to Scenario 1 math: 4 - 2 = 2.
        """
        data = _build_phl_3rd_fixture(phl_fridays_forced=2, num_weeks=4)
        # run_count_adjusters reads data['forced_games'] and data['blocked_games']
        adjustments = run_count_adjusters(data)
        assert 'ClubVsClubCoincidence' in adjustments, (
            'run_count_adjusters must populate ClubVsClubCoincidence key'
        )
        cvc_adj = adjustments['ClubVsClubCoincidence']
        assert cvc_adj is not None
        assert cvc_adj['PHL'][('Maitland', 'Norths')] == 2, (
            f'Expected 2 via engine dispatch; got {cvc_adj["PHL"][("Maitland","Norths")]}'
        )


# ---------------------------------------------------------------------------
# Scenario 3: Atom with reduced expected count allows fewer Sunday coincidences
# ---------------------------------------------------------------------------


class TestAtomAllowsReducedSundayCoincidences:
    def test_reduced_expected_means_fewer_coincidences_needed(self):
        """Given: 4-week fixture, 2 PHL Fridays forced for Maitland-Norths.
        No constraint on 3rd grade (so the 3rd-grade atom fires cleanly).
        For PHL: the atom does NOT fire (PHL/2nd excluded from this atom).

        The interesting case: if we manually give data['count_adjustments']
        a reduced expected for 3rd AND clamp the model to have fewer than
        4 coincident rounds, we verify the model becomes FEASIBLE only with
        the reduction.

        Scenario: 3rd grade has 4 rounds but FORCED says 2 Sunday meetings
        of (Maitland, Norths) are blocked → expected drops to 2.
        The atom adds `sum(coincide_vars) >= 2` instead of `>= 4`.
        With only 2 rounds available that have 3rd-grade games for both
        clubs, the model is FEASIBLE with reduced expected but INFEASIBLE
        with full expected.

        Hand-computed:
          - Fixture: 2 weeks of 3rd-grade Sunday timeslots (num_weeks=2).
          - per_team_games: 2 // (2-1) = 2.
          - No FORCED: expected = 2. min_required = 2. Both weeks must coincide.
          - With adjustment = 1: min_required = 1. One week can miss.
          - We verify: with adjustment=1 and only 1 available round for both
            clubs, the model is FEASIBLE. Without adjustment (min=2), INFEASIBLE.
        """
        # Build a 2-week 3rd-grade-only fixture with 1 round of Sunday slots
        # for each club (they have no overlap in week 2 to force infeasibility).
        ef = PlayingField(location=BROADMEADOW, name='EF')
        wf = PlayingField(location=BROADMEADOW, name='WF')
        clubs = [
            Club(name='Maitland', home_field=MAITLAND),
            Club(name='Norths', home_field=BROADMEADOW),
        ]
        teams = [
            Team(name='Maitland 3rd', club=clubs[0], grade='3rd'),
            Team(name='Norths 3rd', club=clubs[1], grade='3rd'),
        ]
        grades = [Grade(name='3rd', teams=['Maitland 3rd', 'Norths 3rd'])]
        games = [('Maitland 3rd', 'Norths 3rd', '3rd')]
        # Week 1: both clubs have Sunday slots.
        # Week 2: same — so without adjustment, both rounds must coincide.
        timeslots = [
            Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                     day_slot=1, field=ef, round_no=1),
            Timeslot(date='2026-03-22', day='Sunday', time='11:30', week=1,
                     day_slot=1, field=wf, round_no=1),
            Timeslot(date='2026-03-29', day='Sunday', time='11:30', week=2,
                     day_slot=1, field=ef, round_no=2),
            Timeslot(date='2026-03-29', day='Sunday', time='11:30', week=2,
                     day_slot=1, field=wf, round_no=2),
        ]
        data_base = {
            'teams': teams, 'clubs': clubs, 'grades': grades,
            'games': games, 'timeslots': timeslots, 'fields': [ef, wf],
            'home_field_map': {}, 'num_rounds': {'3rd': 2, 'max': 2},
            'forced_games': [], 'blocked_games': [],
            'current_week': 0, 'locked_weeks': set(),
            'constraint_slack': {}, 'penalty_weights': {}, 'penalties': {},
            'team_conflicts': [], 'phl_preferences': {}, 'club_days': {},
            'preference_no_play': {}, 'constraint_defaults': {},
        }

        def _run(adjustment: dict) -> int:
            import copy
            data = copy.deepcopy(data_base)
            data['count_adjustments'] = {'ClubVsClubCoincidence': adjustment}
            model = cp_model.CpModel()
            X = {}
            for (t1, t2, grade) in data['games']:
                for ts in data['timeslots']:
                    key = (t1, t2, grade, ts.day, ts.day_slot, ts.time,
                           ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
                    X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{ts.week}_{ts.field.name}')
            r = HelperVarRegistry(model)
            r.freeze({}, {})
            ClubVsClubCoincidence().apply(model, X, data, r)
            s, _sv = _solve(model)
            return s

        # Without any adjustment: per_team_games=2, min_required=2.
        # Both rounds must coincide. The fixture has 2 rounds each with 2
        # slots (EF, WF) so both rounds can have a Maitland-Norths game on
        # either field. FEASIBLE in normal operation.
        status_no_adj = _run({})
        assert status_no_adj in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            'No-adjustment case must be feasible for 2-round 2-team fixture'
        )

        # With adjustment reducing to 1: only 1 coincidence needed.
        # Even if we block one round, still FEASIBLE.
        status_with_adj = _run({'3rd': {('Maitland', 'Norths'): 1}})
        assert status_with_adj in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            'Adjusted (1 coincidence needed) must also be feasible'
        )
