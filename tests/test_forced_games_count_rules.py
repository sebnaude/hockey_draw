# spec-015: pins the generic FORCED_GAMES count-rule capability so the deleted
# GosfordFridayRoundsForced atom can't be re-introduced as a code workaround.
"""Generic FORCED_GAMES count-rule tests (spec-015).

`GosfordFridayRoundsForced` was a bespoke atom that hard-coded a season-specific
`sum == 1` PHL Gosford-Friday rule per round. spec-015 deleted it: the season
config expresses such rules generically as FORCED_GAMES entries carrying a
**scope** (e.g. `field_location` + `day`), a **`count`**, and an equality-type
**`constraint`** (`equal | lesse | greatere | greater | less`). The engine
(`utils._build_forced_game_rules` + `_get_matching_forced_scopes`) sums over all
variables matching the scope and applies the requested relation.

These tests prove that capability end-to-end on **synthetic** data — no season
config, no mocks, real CP-SAT models, hand-computed oracles. They are the
regression guard that replaces the deleted atom's own tests.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Club, Grade, PlayingField, Team, Timeslot  # noqa: E402
from utils import (  # noqa: E402
    _build_forced_game_rules,
    _get_matching_forced_scopes,
)

# A synthetic venue + grade — deliberately NOT a real season venue, to make
# clear this is engine-level engineering, not season curation.
SYNTH_VENUE = 'Synthetic Friday Arena'


def _build_fixture() -> Dict:
    """4 PHL teams; each week has exactly ONE Friday slot at SYNTH_VENUE plus a
    Sunday slot elsewhere (out of scope). 3 weeks -> the C(4,2)=6 pairings each
    get a candidate var per Friday week, so SYNTH_VENUE Fridays have 6*3 = 18
    candidate variables across the fixture (before any per-week dedup)."""
    arena = PlayingField(location=SYNTH_VENUE, name='Arena 1')
    other = PlayingField(location='Elsewhere', name='Other 1')

    clubs = [Club(name=n, home_field='Elsewhere') for n in
             ('Alpha', 'Bravo', 'Charlie', 'Delta')]
    teams = [Team(name=f'{c.name} PHL', club=c, grade='PHL') for c in clubs]
    grades = [Grade(name='PHL', teams=[t.name for t in teams])]

    games: List[Tuple[str, str, str]] = [
        (t1, t2, 'PHL') for t1, t2 in combinations([t.name for t in teams], 2)
    ]

    timeslots: List[Timeslot] = []
    week_dates = [
        (1, '2026-03-22', '2026-03-20'),
        (2, '2026-03-29', '2026-03-27'),
        (3, '2026-04-05', '2026-04-03'),
    ]
    for wk, sun, fri in week_dates:
        timeslots.append(Timeslot(date=sun, day='Sunday', time='11:30', week=wk,
                                  day_slot=1, field=other, round_no=wk))
        timeslots.append(Timeslot(date=fri, day='Friday', time='19:00', week=wk,
                                  day_slot=1, field=arena, round_no=wk))

    return {
        'teams': teams, 'clubs': clubs, 'fields': [arena, other],
        'games': games, 'grades': grades, 'timeslots': timeslots,
        'home_field_map': {},  # no home-venue filter for the synthetic venue
    }


def _all_keys(data: Dict) -> List[tuple]:
    keys = []
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            keys.append((t1, t2, grade, ts.day, ts.day_slot, ts.time,
                         ts.week, ts.date, ts.round_no, ts.field.name,
                         ts.field.location))
    return keys


def _build_model(data: Dict, forced: list):
    """Mirror the FORCED loop inside generate_X (no home filter here). Returns
    (model, X, scope_to_vars, ctypes, ccounts)."""
    rules, ctypes, ccounts = _build_forced_game_rules(forced, data['teams'])
    model = cp_model.CpModel()
    X: Dict[tuple, cp_model.IntVar] = {}
    scope_to_vars = defaultdict(list)
    for key in _all_keys(data):
        var = model.NewBoolVar(f'X_{key[0]}_{key[1]}_{key[6]}_{key[3]}')
        X[key] = var
        for sk in (_get_matching_forced_scopes(key, rules) if rules else []):
            scope_to_vars[sk].append(var)
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


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    return solver.Solve(model), solver


def _arena_friday_vars(X):
    return [v for k, v in X.items()
            if k[3] == 'Friday' and k[10] == SYNTH_VENUE]


# ----------------------------------------------------------------------
# No-team scope matches exactly the Friday/venue variables (hand oracle)
# ----------------------------------------------------------------------

def test_no_team_scope_matches_all_matching_vars():
    """Given a FORCED entry with only {field_location, day} (no teams). Then the
    matched scope-var set is EXACTLY every Friday var at SYNTH_VENUE.

    Hand oracle: 6 pairings × 3 Friday weeks = 18 Friday/venue vars; the 18
    Sunday/Elsewhere vars must NOT match."""
    data = _build_fixture()
    forced = [{'grade': 'PHL', 'day': 'Friday',
               'field_location': SYNTH_VENUE, 'count': 2, 'constraint': 'equal'}]
    _model, X, scope_to_vars, _ctypes, _counts = _build_model(data, forced)

    assert len(scope_to_vars) == 1
    matched = set(map(id, next(iter(scope_to_vars.values()))))
    expected = {id(v) for v in _arena_friday_vars(X)}
    assert matched == expected
    assert len(expected) == 18  # 6 pairings × 3 Friday weeks


# ----------------------------------------------------------------------
# Equality types each constrain the matching-var sum (hand oracles)
# ----------------------------------------------------------------------

def test_equal_forces_exact_count():
    """Given {SYNTH_VENUE, Friday, equal, count:2}. Then a solved schedule has
    EXACTLY 2 Friday games at SYNTH_VENUE."""
    data = _build_fixture()
    forced = [{'grade': 'PHL', 'day': 'Friday',
               'field_location': SYNTH_VENUE, 'count': 2, 'constraint': 'equal'}]
    model, X, *_ = _build_model(data, forced)
    # Per-team-per-week no-double-booking so the solver schedules realistically.
    by_team_week = defaultdict(list)
    for key, var in X.items():
        by_team_week[(key[0], key[6])].append(var)
        by_team_week[(key[1], key[6])].append(var)
    for vl in by_team_week.values():
        model.Add(sum(vl) <= 1)
    status, solver = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    chosen = sum(solver.Value(v) for v in _arena_friday_vars(X))
    assert chosen == 2  # exactly N

    # And: forcing all Friday/venue vars to 0 makes equal:2 INFEASIBLE.
    model2, X2, *_ = _build_model(data, forced)
    for v in _arena_friday_vars(X2):
        model2.Add(v == 0)
    status2, _ = _solve(model2)
    assert status2 == cp_model.INFEASIBLE


def test_lesse_caps_count():
    """Given {SYNTH_VENUE, Friday, lesse, count:1}. Then no schedule may place
    more than 1 Friday game at SYNTH_VENUE, even when maximised."""
    data = _build_fixture()
    forced = [{'grade': 'PHL', 'day': 'Friday',
               'field_location': SYNTH_VENUE, 'count': 1, 'constraint': 'lesse'}]
    model, X, *_ = _build_model(data, forced)
    fri = _arena_friday_vars(X)
    assert len(fri) > 1
    model.Maximize(sum(fri))  # push the solver to use as many as allowed
    status, solver = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert sum(solver.Value(v) for v in fri) <= 1


def test_greatere_floors_count():
    """Given {SYNTH_VENUE, Friday, greatere, count:2}. Then any feasible
    schedule has AT LEAST 2 Friday games at SYNTH_VENUE, even when minimised."""
    data = _build_fixture()
    forced = [{'grade': 'PHL', 'day': 'Friday',
               'field_location': SYNTH_VENUE, 'count': 2, 'constraint': 'greatere'}]
    model, X, *_ = _build_model(data, forced)
    fri = _arena_friday_vars(X)
    model.Minimize(sum(fri))  # push the solver to use as few as allowed
    status, solver = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert sum(solver.Value(v) for v in fri) >= 2
