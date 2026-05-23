"""Tests for AwayClubPerOpponentAndAggregateHomeBalance atom (spec-004).

Real CP-SAT models, no mocks. Each test:
  - constructs a tiny scenario;
  - hand-computes the expected per-pair and aggregate home counts;
  - applies the atom AND minimum supporting hard constraints (pair sums, field
    capacity), then SOLVES;
  - asserts the actual home/away counts fall in the expected
    [floor(total/2), ceil(total/2)] window.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms.away_club_home_balance import (
    AwayClubPerOpponentAndAggregateHomeBalance,
)
from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


def _build_3opponent_fixture(
    *, num_weeks: int, meetings_per_pair: int,
) -> Dict:
    """Maitland PHL vs three different opponents; each pair plays exactly
    `meetings_per_pair` games over `num_weeks` weeks.

    Each week has 1 slot at Broadmeadow + 1 slot at Maitland Park, both Sunday.
    """
    bm_ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Main')

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Tigers', home_field=BROADMEADOW),
    ]

    teams = [
        Team(name=f'{c.name} PHL', club=c, grade='PHL') for c in clubs
    ]
    grade_obj = Grade(name='PHL', teams=[t.name for t in teams])

    # Only include Maitland-vs-X pairs (skip Wests-vs-Tigers etc. to keep the
    # fixture focused on Maitland's home/away balance).
    games: List[Tuple[str, str, str]] = []
    for opp in ('Norths PHL', 'Wests PHL', 'Tigers PHL'):
        pair = tuple(sorted(['Maitland PHL', opp]))
        games.append((pair[0], pair[1], 'PHL'))

    # Maitland plays meetings_per_pair × 3 opponents in total.
    total_per_team = meetings_per_pair * 3

    timeslots: List[Timeslot] = []
    base_n = 22
    for week in range(1, num_weeks + 1):
        sun_d = f'2026-03-{base_n + 7 * (week - 1):02d}'
        timeslots.append(Timeslot(
            date=sun_d, day='Sunday', time='11:30', week=week,
            day_slot=1, field=bm_ef, round_no=week,
        ))
        timeslots.append(Timeslot(
            date=sun_d, day='Sunday', time='13:00', week=week,
            day_slot=2, field=bm_ef, round_no=week,
        ))
        timeslots.append(Timeslot(
            date=sun_d, day='Sunday', time='11:30', week=week,
            day_slot=1, field=mp, round_no=week,
        ))
        timeslots.append(Timeslot(
            date=sun_d, day='Sunday', time='13:00', week=week,
            day_slot=2, field=mp, round_no=week,
        ))

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': [grade_obj],
        'clubs': clubs,
        'fields': [bm_ef, mp],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'PHL': total_per_team, 'max': total_per_team},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
        '_meetings_per_pair': meetings_per_pair,
    }


def _build_X(model, data: Dict) -> Dict:
    X = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{ts.week}_{ts.field.name}_{ts.day_slot}'
            )
    return X


def _add_basic_hard(model, X: Dict, data: Dict) -> None:
    pair_vars = defaultdict(list)
    field_slot_vars = defaultdict(list)
    for key, var in X.items():
        pair_vars[(key[0], key[1], key[2])].append(var)
        # one game per (field_name, date, slot)
        field_slot_vars[(key[7], key[9], key[4])].append(var)
    meetings = data['_meetings_per_pair']
    for vars_list in pair_vars.values():
        # Each pair plays exactly `meetings_per_pair` games over the season.
        model.Add(sum(vars_list) == meetings)
    for vars_list in field_slot_vars.values():
        model.Add(sum(vars_list) <= 1)


def _solve_pair_home_away(model, X, data) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Solve and return {(maitland_team, opponent): (home_games, away_games)}."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
        f'Expected feasible solve, got {solver.status_name(status)}'
    )

    home_venue = data['home_field_map']['Maitland']
    counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for key, var in X.items():
        if solver.Value(var) != 1:
            continue
        t1, t2 = key[0], key[1]
        # Find which team is Maitland.
        if 'Maitland' in t1:
            maitland_team, opponent = t1, t2
        elif 'Maitland' in t2:
            maitland_team, opponent = t2, t1
        else:
            continue
        if key[10] == home_venue:
            counts[(maitland_team, opponent)][0] += 1
        else:
            counts[(maitland_team, opponent)][1] += 1
    return {k: tuple(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Scenario A: each pair plays 3 times → 2H/1A or 1H/2A; aggregate 9 games → 5H/4A or 4H/5A.
# ---------------------------------------------------------------------------


class TestAwayClubHomeBalance:

    def test_given_each_pair_plays_three_then_pair_within_one(self):
        """Per-pair: with 3 meetings, home ∈ {floor(3/2), ceil(3/2)} = {1, 2}.

        Spec-004 DoD: "each pair lands 2H/1A or 1H/2A (within ±1) — no pair
        stuck at 3H/0A."
        """
        data = _build_3opponent_fixture(num_weeks=12, meetings_per_pair=3)

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard(model, X, data)

        registry = HelperVarRegistry(model)
        AwayClubPerOpponentAndAggregateHomeBalance().apply(model, X, data, registry)

        per_pair = _solve_pair_home_away(model, X, data)
        # Hand-computed: for each pair, home_games ∈ {1, 2}.
        # Aggregate: 9 total games, home ∈ {4, 5}.
        for (mait, opp), (home, away) in per_pair.items():
            total = home + away
            assert total == 3, f'{mait} vs {opp}: expected 3 meetings, got {total}'
            assert home in (1, 2), (
                f'{mait} vs {opp}: home={home} not in {{1,2}} for 3 meetings'
            )

        agg_home = sum(h for h, a in per_pair.values())
        agg_total = sum(h + a for h, a in per_pair.values())
        # 9 total games → home ∈ {4, 5}.
        assert agg_total == 9, f'expected 9 total Maitland games, got {agg_total}'
        assert agg_home in (4, 5), (
            f'aggregate home_games={agg_home} not in {{4,5}} for 9 games'
        )

    def test_given_each_pair_plays_six_then_per_pair_exactly_three(self):
        """Per-pair: 6 meetings → home == 3 (exact half). Aggregate: 18 → 9."""
        data = _build_3opponent_fixture(num_weeks=24, meetings_per_pair=6)

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard(model, X, data)

        registry = HelperVarRegistry(model)
        AwayClubPerOpponentAndAggregateHomeBalance().apply(model, X, data, registry)

        per_pair = _solve_pair_home_away(model, X, data)
        for (mait, opp), (home, away) in per_pair.items():
            assert home == 3, f'{mait} vs {opp}: expected exact 3H, got {home}H/{away}A'

        agg_home = sum(h for h, a in per_pair.values())
        # Hand-computed: 18 total → home ∈ {9} (exact).
        assert agg_home == 9, f'aggregate home_games={agg_home}, expected 9'

    def test_given_each_pair_plays_one_then_pair_zero_or_one(self):
        """Per-pair: 1 meeting → home ∈ {0, 1}. Aggregate: 3 games → home ∈ {1, 2}."""
        data = _build_3opponent_fixture(num_weeks=4, meetings_per_pair=1)

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard(model, X, data)

        registry = HelperVarRegistry(model)
        AwayClubPerOpponentAndAggregateHomeBalance().apply(model, X, data, registry)

        per_pair = _solve_pair_home_away(model, X, data)
        for (mait, opp), (home, away) in per_pair.items():
            assert home in (0, 1), (
                f'{mait} vs {opp}: home={home} not in {{0,1}} for 1 meeting'
            )

        agg_home = sum(h for h, a in per_pair.values())
        # Hand-computed: 3 total → home ∈ {1, 2}.
        assert agg_home in (1, 2), (
            f'aggregate home_games={agg_home} not in {{1,2}} for 3 games'
        )

    def test_no_away_clubs_then_no_constraints(self):
        """Empty home_field_map → no-op."""
        data = _build_3opponent_fixture(num_weeks=4, meetings_per_pair=1)
        data['home_field_map'] = {}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        registry = HelperVarRegistry(model)
        count = AwayClubPerOpponentAndAggregateHomeBalance().apply(
            model, X, data, registry,
        )
        assert count == 0
