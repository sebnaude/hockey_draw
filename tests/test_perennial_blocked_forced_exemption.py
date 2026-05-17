"""Spec-001 regression tests: PERENNIAL_BLOCKED_GAMES must be overridable by FORCED_GAMES.

Background
----------
`PERENNIAL_BLOCKED_GAMES` in `config/defaults.py` ships a "rounds 1-2 at
Broadmeadow only" rule. Without spec-001 this rule eliminated *every* non-NIHC
variable in rounds 1-2 before FORCED_GAMES could match — so a convenor
forcing an opening-round Maitland-vs-Norths game at Maitland Park would
silently lose the variable, then `generate_X` would FATAL on the missing
forced rule.

Target behaviour (spec-001 DoD):
    1. A variable that matches both a PERENNIAL BLOCKED scope **and** any
       FORCED_GAMES scope is **kept** (FORCED wins).
    2. A variable matching PERENNIAL BLOCKED with no FORCED match is
       eliminated (current behaviour preserved).
    3. A variable matching a NON-perennial BLOCKED scope is always
       eliminated, even when a FORCED scope also matches — perennial-only
       fix is targeted (spec-001 out-of-scope §1).

All tests build the same small fixture (PHL, 3 weeks, two venues) and walk
the real `generate_X` / `_build_blocked_game_rules_with_perennial` /
`_get_matching_forced_scopes` machinery. No mocks anywhere. The expected
variable counts are computed by hand in each test's docstring.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ortools.sat.python import cp_model

from constraints.atoms.base import BROADMEADOW, MAITLAND
from models import Club, Grade, PlayingField, Team, Timeslot
from utils import (
    _build_blocked_game_rules,
    _build_blocked_game_rules_with_perennial,
    _matching_blocked_scope_keys,
    generate_X,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _build_fixture() -> Dict:
    """PHL fixture for rounds 1-3:

      Teams: Norths PHL, Tigers PHL, Maitland PHL
      Venues: Broadmeadow EF, Maitland Park (away venue, home club = Maitland)
      Timeslots: one EF slot + one Maitland-Park slot per round, 3 rounds.

    The home-venue filter inside `generate_X` keeps only games involving
    Maitland at Maitland Park, so at Maitland Park the candidate matchups
    each week are {Maitland-vs-Norths, Maitland-vs-Tigers} (2 vars per
    week × 3 weeks = 6 vars at Maitland Park before BLOCKED/FORCED).
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Maitland Main Field')

    clubs = [
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
    ]
    teams = [Team(name=f'{c.name} PHL', club=c, grade='PHL') for c in clubs]

    week_dates = [
        (1, '2026-03-22'),
        (2, '2026-03-29'),
        (3, '2026-04-05'),
    ]
    timeslots: List[Timeslot] = []
    for wk, sun in week_dates:
        # Broadmeadow EF slot — used for non-blocked vars
        timeslots.append(Timeslot(
            date=sun, day='Sunday', time='11:30', week=wk,
            day_slot=1, field=ef, round_no=wk,
        ))
        # Maitland Park slot — perennially blocked in rounds 1-2
        timeslots.append(Timeslot(
            date=sun, day='Sunday', time='13:00', week=wk,
            day_slot=2, field=mp, round_no=wk,
        ))

    grade = Grade(name='PHL', teams=[t.name for t in teams])

    data = {
        'year': 2026,
        'teams': teams,
        'clubs': clubs,
        'grades': [grade],
        'fields': [ef, mp],
        'timeslots': timeslots,
        'num_rounds': {'PHL': 3},
        'home_field_map': {'Maitland': MAITLAND},
        'locked_weeks': set(),
        'phl_game_times': {},     # empty => 2026 nested format, no PHL filtering on phl_valid_slots
        'phl_preferences': {'preferred_dates': []},
        'second_grade_times': {},
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


# But generate_X needs phl_valid_slots populated for PHL. Build a helper that
# pre-loads phl_game_times with the two slots actually present in the fixture
# so the PHL filter inside generate_X lets them through.
def _make_phl_aware_data(data: Dict) -> Dict:
    """Wire up phl_game_times so generate_X's PHL filter accepts the fixture slots."""
    from datetime import time as tm
    data['phl_game_times'] = {
        BROADMEADOW: {
            'EF': {'Sunday': [tm(11, 30)]},
        },
        MAITLAND: {
            'Maitland Main Field': {'Sunday': [tm(13, 0)]},
        },
    }
    return data


# ---------------------------------------------------------------------------
# Unit-level coverage of the helper primitives
# ---------------------------------------------------------------------------

def test_perennial_flag_round_trips_into_scope_key_set():
    """A BLOCKED entry with `'perennial': True` must surface in the
    `perennial_scope_keys` set returned by
    `_build_blocked_game_rules_with_perennial`."""
    data = _make_phl_aware_data(_build_fixture())
    blocked = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Perennial r1 Maitland block',
         'perennial': True},
        {'round_no': 2, 'field_location': MAITLAND,
         'description': 'Perennial r2 Maitland block',
         'perennial': True},
        # Non-perennial: a season-specific block
        {'round_no': 3, 'field_location': MAITLAND,
         'description': 'Season-specific r3 Maitland block'},
    ]
    rules, perennial = _build_blocked_game_rules_with_perennial(blocked, data['teams'])
    # Hand-computed: 3 distinct scope_keys (different round_no per entry).
    assert len(rules) == 3
    # Hand-computed: exactly 2 of those scope_keys are flagged perennial.
    assert len(perennial) == 2


def test_back_compat_build_blocked_game_rules_still_returns_dict():
    """The legacy `_build_blocked_game_rules(...) -> dict` API must continue to
    work for the 5 production + test callers that pre-date spec-001."""
    data = _make_phl_aware_data(_build_fixture())
    blocked = [
        {'round_no': 1, 'field_location': MAITLAND, 'perennial': True},
    ]
    rules = _build_blocked_game_rules(blocked, data['teams'])
    assert isinstance(rules, dict)
    assert len(rules) == 1


def test_matching_blocked_scope_keys_returns_all_matches():
    """`_matching_blocked_scope_keys` must return every blocking scope_key, not
    just the first. Needed because perennial-vs-non-perennial decisions in
    `generate_X` look at the FULL set of matches."""
    data = _make_phl_aware_data(_build_fixture())
    blocked = [
        {'round_no': 1, 'field_location': MAITLAND, 'perennial': True},
        {'date': '2026-03-22', 'field_location': MAITLAND},  # non-perennial overlap
    ]
    rules, _per = _build_blocked_game_rules_with_perennial(blocked, data['teams'])
    # A round-1 Maitland-Park var matches BOTH scopes (round_no=1 + date=2026-03-22).
    key = (
        'Maitland PHL', 'Norths PHL', 'PHL', 'Sunday', 2, '13:00',
        1, '2026-03-22', 1, 'Maitland Main Field', MAITLAND,
    )
    matches = _matching_blocked_scope_keys(key, rules)
    # Hand-computed: exactly 2 scope_keys match this var.
    assert len(matches) == 2


# ---------------------------------------------------------------------------
# generate_X end-to-end: FORCED overrides perennial-BLOCKED
# ---------------------------------------------------------------------------

def test_forced_round1_maitland_park_survives_perennial_block():
    """Spec-001 DoD #1 + #3.

    Setup:
      - PERENNIAL_BLOCKED for r1 Maitland Park (perennial flag set).
      - PERENNIAL_BLOCKED for r2 Maitland Park (perennial flag set).
      - FORCED: Maitland-vs-Norths PHL on 2026-03-22 at Maitland Park (round 1).
      - 3 rounds, EF slot per week, Maitland Park slot per week.

    Hand-computed expected outcome:
      - Round 1 Maitland Park: only Maitland-vs-Norths survives the perennial
        block (FORCED override). Maitland-vs-Tigers eliminated (no FORCED).
        ⇒ exactly 1 round-1 MP variable, and it is Maitland-vs-Norths.
      - Round 2 Maitland Park: NO FORCED entries ⇒ all candidate vars
        eliminated. ⇒ 0 round-2 MP variables.
      - Round 3 Maitland Park: no perennial block ⇒ all candidate vars (2)
        survive (home-venue filter already trimmed to Maitland-involving pairs).
        ⇒ 2 round-3 MP variables.

    Total Maitland-Park variables expected: 1 + 0 + 2 = 3.
    """
    model = cp_model.CpModel()
    data = _make_phl_aware_data(_build_fixture())
    data['blocked_games'] = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Rounds 1-2 at Broadmeadow only (perennial)',
         'perennial': True},
        {'round_no': 2, 'field_location': MAITLAND,
         'description': 'Rounds 1-2 at Broadmeadow only (perennial)',
         'perennial': True},
    ]
    data['forced_games'] = [
        {'teams': ['Maitland', 'Norths'], 'grade': 'PHL',
         'date': '2026-03-22', 'field_location': MAITLAND,
         'count': 1, 'constraint': 'equal',
         'description': 'Opening round Maitland-vs-Norths at MP'},
    ]

    X, _conflicts = generate_X(model, data)

    mp_r1_keys = [k for k in X if k[8] == 1 and k[10] == MAITLAND]
    mp_r2_keys = [k for k in X if k[8] == 2 and k[10] == MAITLAND]
    mp_r3_keys = [k for k in X if k[8] == 3 and k[10] == MAITLAND]

    # DoD #1: FORCED-overridden var survives.
    assert len(mp_r1_keys) == 1, (
        f"Expected exactly 1 round-1 Maitland-Park var (the FORCED M-vs-N), "
        f"got {len(mp_r1_keys)}: {mp_r1_keys}"
    )
    r1_pair = tuple(sorted((mp_r1_keys[0][0], mp_r1_keys[0][1])))
    assert r1_pair == ('Maitland PHL', 'Norths PHL')

    # DoD #2: perennial-only block with no FORCED still eliminates.
    assert mp_r2_keys == [], (
        f"Round-2 Maitland-Park vars should be fully eliminated by perennial "
        f"block (no FORCED override), got {mp_r2_keys}"
    )

    # DoD #1 negative side: Maitland-vs-Tigers in round 1 has no FORCED
    # override and must be eliminated. Already implied by len == 1 check
    # above; assert explicitly for clarity.
    assert all('Tigers PHL' not in (k[0], k[1]) for k in mp_r1_keys), (
        "Round-1 Maitland-Park var should NOT include Tigers (no FORCED override)"
    )

    # Round 3 unaffected — sanity check that nothing else regressed.
    assert len(mp_r3_keys) == 2


def test_non_perennial_block_is_not_overridable_by_forced():
    """Spec-001 out-of-scope §1: NON-perennial BLOCKED scopes still beat FORCED.

    Setup:
      - BLOCKED for r1 Maitland Park (NO perennial flag → season-specific).
      - FORCED: Maitland-vs-Norths PHL on 2026-03-22 at MP (round 1).

    Hand-computed expected outcome:
      - Round 1 MP: the FORCED var is BLOCKED by the season-specific entry.
        ⇒ 0 round-1 MP variables.
      - Because the FORCED rule then has 0 matching vars, `generate_X` raises
        SystemExit (its FATAL path for missing FORCED). We assert that.
    """
    model = cp_model.CpModel()
    data = _make_phl_aware_data(_build_fixture())
    data['blocked_games'] = [
        # No 'perennial' key → defaults to False.
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Season-specific r1 MP block'},
    ]
    data['forced_games'] = [
        {'teams': ['Maitland', 'Norths'], 'grade': 'PHL',
         'date': '2026-03-22', 'field_location': MAITLAND,
         'count': 1, 'constraint': 'equal'},
    ]

    with pytest.raises(SystemExit):
        generate_X(model, data)


def test_perennial_block_with_no_forced_eliminates_all_vars_in_scope():
    """Spec-001 DoD #2.

    Setup:
      - PERENNIAL block for r1 Maitland Park (perennial=True).
      - No FORCED entries at all.

    Hand-computed expected outcome:
      - Round 1 MP: 2 candidate vars (Maitland-vs-Norths, Maitland-vs-Tigers),
        both eliminated ⇒ 0 round-1 MP variables.
      - Round 2 + 3 MP: not blocked ⇒ 2 vars each ⇒ 4 vars total.
    """
    model = cp_model.CpModel()
    data = _make_phl_aware_data(_build_fixture())
    data['blocked_games'] = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Perennial r1 MP block', 'perennial': True},
    ]
    data['forced_games'] = []

    X, _conflicts = generate_X(model, data)

    mp_r1 = [k for k in X if k[8] == 1 and k[10] == MAITLAND]
    mp_r2 = [k for k in X if k[8] == 2 and k[10] == MAITLAND]
    mp_r3 = [k for k in X if k[8] == 3 and k[10] == MAITLAND]

    assert mp_r1 == []
    assert len(mp_r2) == 2
    assert len(mp_r3) == 2


def test_mixed_perennial_and_non_perennial_block_eliminates_var():
    """If a var matches BOTH a perennial scope AND a non-perennial scope, the
    non-perennial block wins (FORCED can only override an *exclusively*
    perennial set of blocks).

    Setup:
      - Perennial block: round_no=1 + field_location=MP
      - Non-perennial block: date=2026-03-22 + field_location=MP
        (also matches round 1, so the round-1 var hits both scopes)
      - FORCED: M-vs-N on 2026-03-22 at MP

    Hand-computed expected outcome:
      - The FORCED Maitland-vs-Norths round-1 var matches BOTH blocking
        scopes. Because one is non-perennial, the FORCED override does NOT
        apply. ⇒ var eliminated ⇒ FORCED has 0 matches ⇒ SystemExit.
    """
    model = cp_model.CpModel()
    data = _make_phl_aware_data(_build_fixture())
    data['blocked_games'] = [
        {'round_no': 1, 'field_location': MAITLAND,
         'description': 'Perennial r1 MP', 'perennial': True},
        {'date': '2026-03-22', 'field_location': MAITLAND,
         'description': 'Season-specific date block'},
    ]
    data['forced_games'] = [
        {'teams': ['Maitland', 'Norths'], 'grade': 'PHL',
         'date': '2026-03-22', 'field_location': MAITLAND,
         'count': 1, 'constraint': 'equal'},
    ]

    with pytest.raises(SystemExit):
        generate_X(model, data)
