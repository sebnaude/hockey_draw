"""spec-009 Check 1 — FORCED total + per-pair stacking.

Verifies that a FORCED total-count entry ("exactly 8 Gosford Friday PHL
games per season") correctly composes with a per-pair FORCED entry ("1 of
those is Gosford-vs-Maitland") so the Gosford-vs-Maitland Friday game counts
toward BOTH scopes — and the total stays at 8, not 9.

Also verifies a no-field_location per-pair entry does NOT inflate the
Gosford-location total (the shared variable is scoped to a different
location, so the scopes are disjoint and the third FORCED entry is
independently satisfied by the same variable counted under a different
scope key).

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.atoms.base import BROADMEADOW, GOSFORD, MAITLAND
from models import Club, Grade, PlayingField, Team, Timeslot
from utils import (
    _build_forced_game_rules,
    _get_matching_forced_scopes,
    _build_blocked_game_rules,
    _is_blocked_by_no_play,
)


# ---------------------------------------------------------------------------
# Fixture: 3-club PHL, 10 weeks with Friday slots at Gosford + Maitland Park
# ---------------------------------------------------------------------------


def _build_gosford_fixture() -> Dict:
    """Small PHL fixture: Gosford, Maitland, Norths clubs; 10 weeks.

    Friday slots at Central Coast Hockey Park (Gosford) and Maitland Park.
    Sunday slots at Broadmeadow EF.

    This gives us enough Gosford Friday capacity to force exactly 8.
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    gosford_field = PlayingField(location=GOSFORD, name='CCHP Main')
    maitland_field = PlayingField(location=MAITLAND, name='Maitland Park')

    clubs = [
        Club(name='Gosford', home_field=GOSFORD),
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [Team(name=f'{c.name} PHL', club=c, grade='PHL') for c in clubs]

    games: List[Tuple[str, str, str]] = []
    for t1, t2 in combinations([t.name for t in teams], 2):
        pair = tuple(sorted([t1, t2]))
        games.append((pair[0], pair[1], 'PHL'))

    # 10 weeks: each week has a Friday slot at GOSFORD and at MAITLAND, plus
    # a Sunday slot at BROADMEADOW.
    timeslots: List[Timeslot] = []
    for wk in range(1, 11):
        # Friday at Gosford
        timeslots.append(Timeslot(
            date=f'2026-03-{19 + wk:02d}', day='Friday', time='20:00', week=wk,
            day_slot=1, field=gosford_field, round_no=wk,
        ))
        # Friday at Maitland
        timeslots.append(Timeslot(
            date=f'2026-03-{19 + wk:02d}', day='Friday', time='19:00', week=wk,
            day_slot=1, field=maitland_field, round_no=wk,
        ))
        # Sunday at Broadmeadow
        timeslots.append(Timeslot(
            date=f'2026-03-{21 + wk:02d}', day='Sunday', time='11:30', week=wk,
            day_slot=1, field=ef, round_no=wk,
        ))

    return {
        'teams': teams,
        'clubs': clubs,
        'fields': [ef, gosford_field, maitland_field],
        'games': games,
        'timeslots': timeslots,
        'home_field_map': {'Gosford': GOSFORD, 'Maitland': MAITLAND},
        'grades': [Grade(name='PHL', teams=[t.name for t in teams])],
        'num_rounds': {'PHL': 10, 'max': 10},
    }


def _build_model_with_forced(
    data: Dict, forced_games: list, blocked_games: list = None,
):
    """Build a CP-SAT model applying forced/blocked rules via the production
    pipeline (generate_X equivalent): home-venue filter + FORCED scope logic.

    Returns (model, X, scope_to_vars, ctypes, ccounts).
    """
    if blocked_games is None:
        blocked_games = []

    forced_rules, ctypes, ccounts = _build_forced_game_rules(forced_games, data['teams'])
    blocked_rules = _build_blocked_game_rules(blocked_games, data['teams'])

    venue_to_home_club = {v: c for c, v in data['home_field_map'].items()}
    team_to_club = {t.name: t.club.name for t in data['teams']}

    model = cp_model.CpModel()
    X: Dict[tuple, cp_model.IntVar] = {}
    scope_to_vars = defaultdict(list)

    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            # Home-venue filter
            home_club = venue_to_home_club.get(ts.field.location)
            if home_club is not None:
                if team_to_club.get(t1) != home_club and team_to_club.get(t2) != home_club:
                    continue
            key = (t1, t2, grade, ts.day, ts.day_slot, ts.time,
                   ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location)
            if blocked_rules and _is_blocked_by_no_play(key, blocked_rules):
                continue
            matches = _get_matching_forced_scopes(key, forced_rules) if forced_rules else []
            var = model.NewBoolVar(f'X_{t1}_{t2}_{ts.week}_{ts.day}_{ts.field.name}')
            X[key] = var
            for sk in matches:
                scope_to_vars[sk].append(var)

    # Apply scope sum constraints
    for sk, vars_list in scope_to_vars.items():
        ctype = ctypes.get(sk, 'equal')
        count = ccounts.get(sk, 1)
        if ctype == 'equal':
            model.Add(sum(vars_list) == count)
        elif ctype == 'lesse':
            model.Add(sum(vars_list) <= count)
        elif ctype == 'greatere':
            model.Add(sum(vars_list) >= count)
        elif ctype == 'greater':
            model.Add(sum(vars_list) > count)
        elif ctype == 'less':
            model.Add(sum(vars_list) < count)

    return model, X, scope_to_vars, ctypes, ccounts


def _solve(model: cp_model.CpModel, seconds: float = 10.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    return status, solver


# ---------------------------------------------------------------------------
# Scenario A: Total + per-pair composition
# ---------------------------------------------------------------------------


class TestForcedTotalPlusPerPair:
    """Given: 3-club PHL fixture, 10 weeks each with Gosford + Maitland Fridays.
    FORCED:
      - scope A: {grade='PHL', day='Friday', field_location=GOSFORD, count=8, equal}
        → exactly 8 Gosford-Friday PHL games per season
      - scope B: {teams=['Gosford PHL','Maitland PHL'], grade='PHL', day='Friday',
                  field_location=GOSFORD, count=1, equal}
        → exactly 1 of those is Gosford-vs-Maitland

    Hand-computed expected:
      - Exactly 8 Gosford Friday games.
      - Exactly 1 of those is Gosford-vs-Maitland.
      - The Gosford-vs-Maitland game is counted in BOTH scope A and scope B
        (multi-scope match). The total does NOT inflate to 9.
    """

    def test_gosford_vs_maitland_var_matches_both_scopes(self):
        """Given: forced rules built from total + pair.
        When: _get_matching_forced_scopes called on a Gosford-vs-Maitland
              Friday-at-Gosford key.
        Then: matches 2 scopes.

        Gosford-vs-Norths matches only 1 scope (scope A, the club-level total).

        Hand-computed: the 11-tuple for Gosford-vs-Maitland at GOSFORD on
        Friday satisfies BOTH scope A (grade=PHL, day=Friday,
        field_location=GOSFORD) and scope B (same + pair filter). The
        Gosford-vs-Norths tuple satisfies scope A only (pair filter in B
        requires Maitland team). So matches_gvsm == 2, matches_gvsn == 1.
        """
        data = _build_gosford_fixture()
        forced = [
            {'grade': 'PHL', 'day': 'Friday', 'field_location': GOSFORD,
             'count': 8, 'constraint': 'equal',
             'description': 'Exactly 8 Gosford Friday PHL games per season'},
            {'teams': ['Gosford PHL', 'Maitland PHL'], 'grade': 'PHL',
             'day': 'Friday', 'field_location': GOSFORD,
             'count': 1, 'constraint': 'equal',
             'description': '1 of those is Gosford-vs-Maitland'},
        ]
        forced_rules, _ctypes, _ccounts = _build_forced_game_rules(forced, data['teams'])

        # Gosford-vs-Maitland Friday Gosford key (week=1)
        key_gvsm = (
            'Gosford PHL', 'Maitland PHL', 'PHL', 'Friday', 1, '20:00',
            1, '2026-03-20', 1, 'CCHP Main', GOSFORD,
        )
        # Gosford-vs-Norths Friday Gosford key (week=1)
        key_gvsn = (
            'Gosford PHL', 'Norths PHL', 'PHL', 'Friday', 1, '20:00',
            1, '2026-03-20', 1, 'CCHP Main', GOSFORD,
        )

        matches_gvsm = _get_matching_forced_scopes(key_gvsm, forced_rules)
        matches_gvsn = _get_matching_forced_scopes(key_gvsn, forced_rules)

        assert len(matches_gvsm) == 2, (
            f'Gosford-vs-Maitland Friday Gosford must match BOTH scopes. '
            f'Got {len(matches_gvsm)}'
        )
        assert len(matches_gvsn) == 1, (
            f'Gosford-vs-Norths Friday Gosford must match only scope A. '
            f'Got {len(matches_gvsn)}'
        )

    def test_solver_total_stays_at_8_not_9(self):
        """Given: FORCED total=8 + per-pair=1 at Gosford.
        When: model solved.
        Then: exactly 8 Gosford Friday games total, exactly 1 is Gosford-vs-Maitland.

        Hand-computed: if multi-scope registration is correct, the solver sees
        a single variable counted toward scope A (total=8) AND scope B (pair=1).
        If it were NOT multi-scope, the solver would see 9 distinct "slots" to
        fill (8 + 1 independent) and would need 9 games — which is impossible
        with only 10 weeks and 2 Friday slots per week but many other
        constraints, or it would schedule an extra game beyond the declared 8.

        The test asserts len(gosford_fridays) == 8, so any inflation to 9
        makes the assertion fail.
        """
        data = _build_gosford_fixture()
        forced = [
            {'grade': 'PHL', 'day': 'Friday', 'field_location': GOSFORD,
             'count': 8, 'constraint': 'equal'},
            {'teams': ['Gosford PHL', 'Maitland PHL'], 'grade': 'PHL',
             'day': 'Friday', 'field_location': GOSFORD,
             'count': 1, 'constraint': 'equal'},
        ]
        model, X, _stv, _ct, _cc = _build_model_with_forced(data, forced)

        # Add per-team-per-week no-double-booking so solver can actually plan.
        by_team_week = defaultdict(list)
        for key, var in X.items():
            by_team_week[(key[0], key[6])].append(var)
            by_team_week[(key[1], key[6])].append(var)
        for vars_list in by_team_week.values():
            model.Add(sum(vars_list) <= 1)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Model infeasible — check fixture capacity. Status: {status}'
        )

        selected = [k for k, v in X.items() if solver.Value(v) == 1]

        # Gosford Friday games
        gosford_fridays = [
            k for k in selected
            if k[3] == 'Friday' and k[10] == GOSFORD
        ]
        assert len(gosford_fridays) == 8, (
            f'Expected 8 Gosford Friday games; got {len(gosford_fridays)}. '
            f'If multi-scope composition is broken the count may be wrong.'
        )

        # Gosford-vs-Maitland
        gvsm = [
            k for k in gosford_fridays
            if {'Gosford PHL', 'Maitland PHL'} == {k[0], k[1]}
        ]
        assert len(gvsm) == 1, (
            f'Expected exactly 1 Gosford-vs-Maitland Friday; got {len(gvsm)}'
        )

    def test_no_field_location_per_pair_does_not_inflate_gosford_total(self):
        """Given: total scope at GOSFORD (count=8) + per-pair scope at GOSFORD
        (count=1, Gosford-vs-Maitland) + third scope: same pair but
        NO field_location restriction (can match any field on Friday).

        The third scope is independent of scope A's field_location=GOSFORD.
        A Friday-at-Maitland Gosford-vs-Maitland game satisfies scopes B2
        and B3, but NOT scope A (different field_location). So the 8-game
        Gosford total is unchanged regardless of where scope B3 is satisfied.

        Hand-computed:
          - Scope A: 8 Gosford Fridays (could be at GOSFORD or mix)
            — Wait, scope A is specifically field_location=GOSFORD, so
              only GOSFORD vars match it. The total GOSFORD Fridays = 8.
          - Scope B: 1 Gosford-vs-Maitland at GOSFORD.
          - Scope C: 1 Gosford-vs-Maitland anywhere on Friday (no loc filter).
            This is DISTINCT from scope A because it has no field_location.
            A Gosford-vs-Maitland at GOSFORD matches scopes A, B, C.
            A Gosford-vs-Maitland at MAITLAND matches only scope C.
          - The solver is free to satisfy scope C by placing Gosford-vs-Maitland
            at MAITLAND (if scope B already pins one at GOSFORD) or by letting
            scope B satisfy it too (if they share a variable).
          - In any valid solution: GOSFORD Fridays == 8.
        """
        data = _build_gosford_fixture()
        forced = [
            {'grade': 'PHL', 'day': 'Friday', 'field_location': GOSFORD,
             'count': 8, 'constraint': 'equal',
             'description': 'Exactly 8 Gosford Friday PHL games'},
            {'teams': ['Gosford PHL', 'Maitland PHL'], 'grade': 'PHL',
             'day': 'Friday', 'field_location': GOSFORD,
             'count': 1, 'constraint': 'equal',
             'description': 'Exactly 1 Gosford-vs-Maitland at Gosford'},
            {'teams': ['Gosford PHL', 'Maitland PHL'], 'grade': 'PHL',
             'day': 'Friday',  # NOTE: no field_location — matches any Friday
             'count': 1, 'constraint': 'equal',
             'description': 'Exactly 1 Gosford-vs-Maitland on any Friday'},
        ]
        model, X, _stv, _ct, _cc = _build_model_with_forced(data, forced)

        by_team_week = defaultdict(list)
        for key, var in X.items():
            by_team_week[(key[0], key[6])].append(var)
            by_team_week[(key[1], key[6])].append(var)
        for vars_list in by_team_week.values():
            model.Add(sum(vars_list) <= 1)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Model infeasible. Status: {status}'
        )

        selected = [k for k, v in X.items() if solver.Value(v) == 1]
        gosford_fridays = [
            k for k in selected if k[3] == 'Friday' and k[10] == GOSFORD
        ]
        # Gosford Friday total must still be exactly 8.
        assert len(gosford_fridays) == 8, (
            f'Expected 8 Gosford Friday games; got {len(gosford_fridays)}. '
            f'Third per-pair scope (no field_location) should NOT inflate Gosford total.'
        )
