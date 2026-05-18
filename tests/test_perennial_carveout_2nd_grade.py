"""spec-010 Part B: Verify FORCED overrides PERENNIAL R1/R2 block for 2nd grade.

Background
----------
spec-001 shipped "FORCED overrides PERENNIAL R1/R2-Broadmeadow-only" for PHL
(tested in `test_perennial_blocked_forced_exemption.py`). The implementation
in `generate_X` is grade-agnostic — the perennial/FORCED resolution applies to
every variable regardless of grade. This module locks that behaviour explicitly
for the 2nd-grade case so any future refactor that accidentally makes it
grade-specific will be caught.

Target behaviour (spec-010 Part B DoD):
    1. A 2nd-grade variable that matches both a PERENNIAL BLOCKED R1/R2 scope
       AND a FORCED_GAMES scope is **kept** (FORCED wins).
    2. A 2nd-grade variable matching PERENNIAL BLOCKED with no FORCED match is
       eliminated (current behaviour preserved).
    3. After `generate_X`, solving a model that forces the FORCED 2nd-grade
       game to 1 is FEASIBLE — the variable exists and is satisfiable.

Fixture: two Broadmeadow clubs + one Maitland club, 2nd grade only, 3 rounds.
Timeslots: one NIHC EF slot + one Maitland Park slot per round (Sunday only;
2nd grade doesn't play Fridays and doesn't play at Gosford).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.atoms.base import BROADMEADOW, MAITLAND
from models import Club, Grade, PlayingField, Team, Timeslot
from utils import generate_X


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _build_2nd_grade_fixture() -> Dict:
    """2nd-grade fixture for rounds 1-3.

    Teams:
      - Norths 2nd (home: Broadmeadow)
      - Tigers 2nd  (home: Broadmeadow)
      - Maitland 2nd (home: Maitland Park)

    Venues:
      - NIHC EF (Broadmeadow)
      - Maitland Park (away venue; only Maitland-involving games scheduled here
        due to the home-venue filter in `generate_X`)

    Timeslots: one EF Sunday slot + one Maitland Park Sunday slot per round,
    3 rounds. (2nd grade is Sunday-only and never plays at Gosford.)

    Candidate Maitland Park 2nd-grade vars before BLOCKED/FORCED:
      Per round: Maitland-vs-Norths + Maitland-vs-Tigers = 2 vars.
      3 rounds × 2 vars = 6 vars total.
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Maitland Main Field')

    clubs = [
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
    ]
    teams = [Team(name=f'{c.name} 2nd', club=c, grade='2nd') for c in clubs]

    week_dates = [
        (1, '2026-03-22'),
        (2, '2026-03-29'),
        (3, '2026-04-05'),
    ]
    timeslots: List[Timeslot] = []
    for wk, sun in week_dates:
        timeslots.append(Timeslot(
            date=sun, day='Sunday', time='11:30', week=wk,
            day_slot=1, field=ef, round_no=wk,
        ))
        timeslots.append(Timeslot(
            date=sun, day='Sunday', time='13:00', week=wk,
            day_slot=2, field=mp, round_no=wk,
        ))

    grade = Grade(name='2nd', teams=[t.name for t in teams])

    data: Dict = {
        'year': 2026,
        'teams': teams,
        'clubs': clubs,
        'grades': [grade],
        'fields': [ef, mp],
        'timeslots': timeslots,
        'num_rounds': {'2nd': 3},
        'home_field_map': {'Maitland': MAITLAND},
        'locked_weeks': set(),
        # 2nd grade filtering: generate_X uses `second_grade_times` like PHL
        # uses `phl_game_times`. An empty dict means no extra slot filtering.
        'phl_game_times': {},
        'second_grade_times': {},
        'phl_preferences': {'preferred_dates': []},
        'club_days': {},
        'preference_no_play': {},
        'penalties': {},
        'forced_games': [],
        'blocked_games': [],
        'penalty_weights': {},
        'constraint_defaults': {},
        'away_venue_rules': {},
    }
    return data


# ---------------------------------------------------------------------------
# DoD #1: FORCED 2nd-grade R1 Maitland-Park var survives perennial block
# ---------------------------------------------------------------------------

def test_forced_round1_2nd_grade_maitland_park_survives_perennial_block():
    """spec-010 Part B DoD #1.

    Setup:
      - PERENNIAL BLOCKED: round_no=1 at Maitland Park (perennial=True).
      - PERENNIAL BLOCKED: round_no=2 at Maitland Park (perennial=True).
      - FORCED: Maitland-vs-Norths 2nd on 2026-03-22 at Maitland Park (round 1).

    Hand-computed expected outcome:
      - Round 1 MP: only Maitland-vs-Norths survives (FORCED override).
        Maitland-vs-Tigers has no FORCED match → eliminated.
        ⇒ exactly 1 round-1 MP variable.
      - Round 2 MP: no FORCED ⇒ all 2 vars eliminated by perennial block.
        ⇒ 0 round-2 MP variables.
      - Round 3 MP: no block at all ⇒ both candidate vars (Maitland-vs-Norths,
        Maitland-vs-Tigers) survive.
        ⇒ 2 round-3 MP variables.

    Total Maitland-Park 2nd-grade variables expected: 1 + 0 + 2 = 3.
    """
    model = cp_model.CpModel()
    data = _build_2nd_grade_fixture()
    data['blocked_games'] = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Rounds 1-2 at Broadmeadow only (perennial)',
         'perennial': True},
        {'round_no': 2, 'field_location': MAITLAND,
         'description': 'Rounds 1-2 at Broadmeadow only (perennial)',
         'perennial': True},
    ]
    data['forced_games'] = [
        {'teams': ['Maitland', 'Norths'], 'grade': '2nd',
         'date': '2026-03-22', 'field_location': MAITLAND,
         'count': 1, 'constraint': 'equal',
         'description': 'Test: opening round Maitland-vs-Norths 2nd at Maitland Park'},
    ]

    X, _conflicts = generate_X(model, data)

    mp_r1_keys = [k for k in X if k[8] == 1 and k[10] == MAITLAND and k[2] == '2nd']
    mp_r2_keys = [k for k in X if k[8] == 2 and k[10] == MAITLAND and k[2] == '2nd']
    mp_r3_keys = [k for k in X if k[8] == 3 and k[10] == MAITLAND and k[2] == '2nd']

    # FORCED override: exactly 1 round-1 Maitland Park 2nd-grade var survives.
    assert len(mp_r1_keys) == 1, (
        f"Expected exactly 1 round-1 Maitland-Park 2nd-grade var (the FORCED "
        f"Maitland-vs-Norths), got {len(mp_r1_keys)}: {mp_r1_keys}"
    )
    r1_pair = tuple(sorted((mp_r1_keys[0][0], mp_r1_keys[0][1])))
    assert r1_pair == ('Maitland 2nd', 'Norths 2nd'), (
        f"The surviving round-1 var should be Maitland-vs-Norths 2nd, got {r1_pair}"
    )

    # Maitland-vs-Tigers 2nd has no FORCED override → eliminated by perennial block.
    assert all('Tigers 2nd' not in (k[0], k[1]) for k in mp_r1_keys)

    # Round 2 fully eliminated (no FORCED override for round 2).
    assert mp_r2_keys == [], (
        f"Round-2 Maitland-Park 2nd-grade vars should be fully eliminated by "
        f"the perennial block (no FORCED override), got {mp_r2_keys}"
    )

    # Round 3 unblocked — both Maitland-involving vars survive.
    assert len(mp_r3_keys) == 2, (
        f"Round 3 should have 2 Maitland-Park 2nd-grade vars (no block), "
        f"got {len(mp_r3_keys)}: {mp_r3_keys}"
    )


# ---------------------------------------------------------------------------
# DoD #2: Perennial R1 block with no FORCED still eliminates all 2nd-grade vars
# ---------------------------------------------------------------------------

def test_perennial_block_with_no_forced_eliminates_all_2nd_grade_vars_in_scope():
    """spec-010 Part B DoD #2.

    Setup:
      - PERENNIAL BLOCKED: round_no=1 at Maitland Park (perennial=True).
      - No FORCED entries.

    Hand-computed expected outcome:
      - Round 1 MP: both 2nd-grade candidate vars eliminated ⇒ 0.
      - Rounds 2 + 3 MP: not blocked ⇒ 2 vars each ⇒ 4 vars total.
    """
    model = cp_model.CpModel()
    data = _build_2nd_grade_fixture()
    data['blocked_games'] = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Perennial r1 MP block', 'perennial': True},
    ]
    data['forced_games'] = []

    X, _conflicts = generate_X(model, data)

    mp_r1 = [k for k in X if k[8] == 1 and k[10] == MAITLAND and k[2] == '2nd']
    mp_r2 = [k for k in X if k[8] == 2 and k[10] == MAITLAND and k[2] == '2nd']
    mp_r3 = [k for k in X if k[8] == 3 and k[10] == MAITLAND and k[2] == '2nd']

    assert mp_r1 == [], (
        f"Round-1 Maitland-Park 2nd-grade vars should all be eliminated by the "
        f"perennial block (no FORCED), got {mp_r1}"
    )
    assert len(mp_r2) == 2
    assert len(mp_r3) == 2


# ---------------------------------------------------------------------------
# DoD #3: Solving with FORCED 2nd-grade var active is FEASIBLE
# ---------------------------------------------------------------------------

def test_forced_2nd_grade_r1_maitland_is_solvable():
    """spec-010 Part B DoD #3.

    After `generate_X` keeps the FORCED 2nd-grade Maitland-vs-Norths round-1
    Maitland Park variable, we pin it to 1 and assert the CP-SAT model is still
    FEASIBLE.

    Hand-computed oracle: pinning one variable to 1 is compatible with the
    model having a trivial solution where only that variable is 1 (no other
    hard constraints are applied in this minimal fixture).

    Given: FORCED + PERENNIAL block as in DoD #1.
    When:  the surviving FORCED variable is pinned to 1 in the model.
    Then:  model is FEASIBLE.
    """
    model = cp_model.CpModel()
    data = _build_2nd_grade_fixture()
    data['blocked_games'] = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Rounds 1-2 at Broadmeadow only (perennial)',
         'perennial': True},
        {'round_no': 2, 'field_location': MAITLAND,
         'description': 'Rounds 1-2 at Broadmeadow only (perennial)',
         'perennial': True},
    ]
    data['forced_games'] = [
        {'teams': ['Maitland', 'Norths'], 'grade': '2nd',
         'date': '2026-03-22', 'field_location': MAITLAND,
         'count': 1, 'constraint': 'equal',
         'description': 'Test: opening round Maitland-vs-Norths 2nd at Maitland Park'},
    ]

    X, _conflicts = generate_X(model, data)

    mp_r1_keys = [k for k in X if k[8] == 1 and k[10] == MAITLAND and k[2] == '2nd']
    assert len(mp_r1_keys) == 1, (
        "Precondition: exactly 1 FORCED round-1 2nd-grade MP var must exist"
    )
    # Pin the FORCED variable to 1.
    model.Add(X[mp_r1_keys[0]] == 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        "Pinning the FORCED 2nd-grade round-1 Maitland Park var to 1 should "
        "be FEASIBLE — the variable was kept by spec-001 grade-agnostic logic"
    )
