"""Multi-scope FORCED_GAMES tests.

Verifies the design described in `docs/FORCED_GAMES_AS_COUNT_RULES.md`:

  - A single decision variable can satisfy multiple FORCED scopes simultaneously
    (the cd8a338 fix), so a Norths-vs-Maitland Friday game counts toward
    BOTH a "Maitland-club Friday count" rule AND a "Norths-vs-Maitland once"
    rule.

  - The new `club` team-filter on FORCED entries resolves to all teams of the
    given club at the entry's grade, mirroring BLOCKED_GAMES.

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
    _is_blocked_by_no_play,
    _build_blocked_game_rules,
)


def _build_fixture() -> Dict:
    """Small PHL fixture: 4 teams, 2 weeks, Friday slots at Maitland Park
    plus Sunday slots at Broadmeadow."""
    ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Maitland Main Field')

    clubs = [
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
    ]
    teams = [Team(name=f'{c.name} PHL', club=c, grade='PHL') for c in clubs]

    games: List[Tuple[str, str, str]] = []
    for t1, t2 in combinations([t.name for t in teams], 2):
        games.append((t1, t2, 'PHL'))

    timeslots: List[Timeslot] = []
    week_dates = [
        (1, '2026-03-22', '2026-03-20'),
        (2, '2026-03-29', '2026-03-27'),
        (3, '2026-04-05', '2026-04-03'),
    ]
    for wk, sun, fri in week_dates:
        # Sunday slot at Broadmeadow EF (general PHL slot, not constrained)
        timeslots.append(Timeslot(
            date=sun, day='Sunday', time='11:30', week=wk,
            day_slot=1, field=ef, round_no=wk,
        ))
        # Friday slot at Maitland Park (the scope of the FORCED rules)
        timeslots.append(Timeslot(
            date=fri, day='Friday', time='19:00', week=wk,
            day_slot=1, field=mp, round_no=wk,
        ))

    return {
        'teams': teams,
        'clubs': clubs,
        'fields': [ef, mp],
        'games': games,
        'timeslots': timeslots,
        'home_field_map': {'Maitland': MAITLAND},
    }


def _build_keys_with_home_filter(data: Dict) -> List[tuple]:
    """Mirror the home-venue filter inside generate_X — only Maitland teams
    play at Maitland Park."""
    venue_to_home_club = {v: c for c, v in data['home_field_map'].items()}
    team_to_club = {t.name: t.club.name for t in data['teams']}
    keys = []
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            home_club = venue_to_home_club.get(ts.field.location)
            if home_club is not None:
                if team_to_club.get(t1) != home_club and team_to_club.get(t2) != home_club:
                    continue
            keys.append((t1, t2, grade, ts.day, ts.day_slot, ts.time,
                         ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location))
    return keys


def _build_model_with_forced(data: Dict, forced_games: list, blocked_games: list = None):
    """Mirror the FORCED/BLOCKED loop inside generate_X — production-equivalent
    pipeline minus the venue-format detection. Returns (model, X, scope_to_vars,
    forced_constraint_types, forced_constraint_counts).
    """
    if blocked_games is None:
        blocked_games = []
    forced_rules, ctypes, ccounts = _build_forced_game_rules(forced_games, data['teams'])
    blocked_rules = _build_blocked_game_rules(blocked_games, data['teams'])

    model = cp_model.CpModel()
    X: Dict[tuple, cp_model.IntVar] = {}
    scope_to_vars = defaultdict(list)

    for key in _build_keys_with_home_filter(data):
        if blocked_rules and _is_blocked_by_no_play(key, blocked_rules):
            continue
        matches = _get_matching_forced_scopes(key, forced_rules) if forced_rules else []
        if forced_rules and matches:
            var = model.NewBoolVar(f'X_{key[0]}_{key[1]}_{key[6]}_{key[3]}')
            X[key] = var
            for sk in matches:
                scope_to_vars[sk].append(var)
        else:
            X[key] = model.NewBoolVar(f'X_{key[0]}_{key[1]}_{key[6]}_{key[3]}')

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


# ----------------------------------------------------------------------
# Unit tests on the rule-building / matching primitives
# ----------------------------------------------------------------------

def test_club_filter_resolves_to_team_matchers():
    data = _build_fixture()
    forced = [{
        'club': 'Maitland', 'grade': 'PHL', 'day': 'Friday',
        'field_location': MAITLAND, 'count': 2, 'constraint': 'equal',
    }]
    rules, ctypes, counts = _build_forced_game_rules(forced, data['teams'])

    assert len(rules) == 1
    scope_key = next(iter(rules))
    matchers = rules[scope_key]
    # Maitland only has a single 'PHL' team in the fixture
    assert matchers == [('any', 'Maitland PHL')]
    assert ctypes[scope_key] == 'equal'
    assert counts[scope_key] == 2


def test_club_and_pair_scopes_coexist_independently():
    """Two FORCED entries with overlapping scopes (one club-filtered, one
    pair-filtered) get distinct scope_keys so each carries its own count."""
    data = _build_fixture()
    forced = [
        {'club': 'Maitland', 'grade': 'PHL', 'day': 'Friday',
         'field_location': MAITLAND, 'count': 2, 'constraint': 'equal'},
        {'teams': ['Norths', 'Maitland'], 'grade': 'PHL', 'day': 'Friday',
         'field_location': MAITLAND, 'count': 1, 'constraint': 'equal'},
    ]
    rules, _ctypes, counts = _build_forced_game_rules(forced, data['teams'])
    assert len(rules) == 2  # two distinct scope_keys via _entry_idx
    assert sorted(counts.values()) == [1, 2]


def test_norths_vs_maitland_var_matches_both_scopes():
    """The cd8a338 multi-scope fix: a single var can satisfy both a club-scope
    rule AND a pair-scope rule simultaneously."""
    data = _build_fixture()
    forced = [
        {'club': 'Maitland', 'grade': 'PHL', 'day': 'Friday',
         'field_location': MAITLAND, 'count': 2, 'constraint': 'equal'},
        {'teams': ['Norths', 'Maitland'], 'grade': 'PHL', 'day': 'Friday',
         'field_location': MAITLAND, 'count': 1, 'constraint': 'equal'},
    ]
    rules, _ctypes, _counts = _build_forced_game_rules(forced, data['teams'])

    # A Norths-vs-Maitland Friday Maitland-Park var
    norths_maitland_key = (
        'Maitland PHL', 'Norths PHL', 'PHL', 'Friday', 1, '19:00',
        1, '2026-03-20', 1, 'Maitland Main Field', MAITLAND,
    )
    matches = _get_matching_forced_scopes(norths_maitland_key, rules)
    assert len(matches) == 2  # matches both scopes

    # A Tigers-vs-Maitland Friday var: matches only the club scope
    tigers_maitland_key = (
        'Maitland PHL', 'Tigers PHL', 'PHL', 'Friday', 1, '19:00',
        1, '2026-03-20', 1, 'Maitland Main Field', MAITLAND,
    )
    matches = _get_matching_forced_scopes(tigers_maitland_key, rules)
    assert len(matches) == 1


# ----------------------------------------------------------------------
# Solver-level test — build a real CP-SAT model and check the schedule
# ----------------------------------------------------------------------

def test_solver_satisfies_combined_club_and_pair_count_rules():
    """Force exactly 2 PHL Friday games at Maitland Park involving Maitland,
    of which exactly 1 is Norths-vs-Maitland. Solver must find a schedule that
    satisfies both rules together."""
    data = _build_fixture()
    forced = [
        {'club': 'Maitland', 'grade': 'PHL', 'day': 'Friday',
         'field_location': MAITLAND, 'count': 2, 'constraint': 'equal'},
        {'teams': ['Norths', 'Maitland'], 'grade': 'PHL', 'day': 'Friday',
         'field_location': MAITLAND, 'count': 1, 'constraint': 'equal'},
    ]
    model, X, scope_to_vars, _, _ = _build_model_with_forced(data, forced)

    # Add a no-double-booking-per-week per-team constraint so the solver doesn't
    # double-book a team in a single week.
    by_team_week = defaultdict(list)
    for key, var in X.items():
        by_team_week[(key[0], key[6])].append(var)
        by_team_week[(key[1], key[6])].append(var)
    for vars_list in by_team_week.values():
        model.Add(sum(vars_list) <= 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    selected = [k for k, v in X.items() if solver.Value(v) == 1]

    friday_at_maitland = [
        k for k in selected
        if k[3] == 'Friday' and k[10] == MAITLAND
    ]
    assert len(friday_at_maitland) == 2

    norths_v_maitland = [
        k for k in friday_at_maitland
        if {'Norths PHL', 'Maitland PHL'} == {k[0], k[1]}
    ]
    assert len(norths_v_maitland) == 1

    # The other Friday Maitland game involves Maitland but is NOT vs Norths
    other = [k for k in friday_at_maitland if k not in norths_v_maitland]
    assert len(other) == 1
    assert 'Maitland PHL' in (other[0][0], other[0][1])
    assert 'Norths PHL' not in (other[0][0], other[0][1])


def test_lesse_count_supports_friday_broadmeadow_pattern():
    """The per-venue Friday count pattern from `docs/FORCED_GAMES_AS_COUNT_RULES.md`:
    `lesse` + `count=N` enforces sum <= N over a no-team-filter scope.
    Models 'max 3 PHL Fridays at Broadmeadow per season' on a 3-week fixture
    where there's only 1 BR Friday slot per week."""
    data = _build_fixture()
    # Move the Sunday slot out of scope — we only want the Friday counts to matter.
    # Use a 3-week fixture; confirm the sum cap is honoured (cap=2 so solver
    # must NOT schedule all 3 possible Maitland Fridays).
    forced = [
        {'grade': 'PHL', 'day': 'Friday', 'field_location': MAITLAND,
         'count': 2, 'constraint': 'lesse'},
    ]
    model, X, scope_to_vars, _, _ = _build_model_with_forced(data, forced)

    # Force EVERY available Maitland-Park Friday var to 1 if possible — the
    # solver should reject (or instead pick at most 2). Add no extra constraints
    # so the solver is free to maximise; force objective = sum of those vars.
    fri_mp_vars = [
        v for k, v in X.items() if k[3] == 'Friday' and k[10] == MAITLAND
    ]
    assert len(fri_mp_vars) > 2  # fixture has at least 3 candidate vars
    model.Maximize(sum(fri_mp_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    selected = sum(solver.Value(v) for v in fri_mp_vars)
    assert selected <= 2
