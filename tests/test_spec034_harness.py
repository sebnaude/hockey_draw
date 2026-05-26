"""spec-034 Unit A — harness + real-data fixtures smoke (no mocks).

Proves the shared fixtures every downstream assurance unit (B/C/D) consumes are
REAL data, not stubs, and that the batched green-suite runner builds a sane plan.

Oracles are derived from documented facts, not echoed from the code under test:
  - CLAUDE.md documents the competition as 6 grades: PHL, 2nd, 3rd, 4th, 5th, 6th.
  - The committed fixture `draw_2026_first6weeks.json` declares num_games = 108.
  - `season_test` is, by definition, the forced-FREE config (zero forced games).
"""
from __future__ import annotations

import importlib

# The exact, documented grade set for the competition (CLAUDE.md: "6 grades
# (PHL, 2nd-6th)"). Used as the hand oracle for both real configs.
EXPECTED_GRADES = {'PHL', '2nd', '3rd', '4th', '5th', '6th'}


def test_real_2026_data_is_real(real_2026_data):
    """Given the 2026 fixture, then it is the real season dict with the documented
    6-grade structure and self-consistent team membership (not a stub)."""
    data = real_2026_data
    assert isinstance(data, dict)
    grade_names = {g.name for g in data['grades']}
    # Oracle: exactly the six documented grades.
    assert grade_names == EXPECTED_GRADES, grade_names
    # Oracle: every team's grade is one of the six, and the flat team list size
    # equals the sum of the per-grade rosters (internal consistency of real data).
    teams = data['teams']
    assert all(t.grade in EXPECTED_GRADES for t in teams)
    roster_total = sum(len(g.teams) for g in data['grades'])
    assert len(teams) == roster_total, (len(teams), roster_total)
    # Real season carries real timeslots and (production) forced games.
    assert len(data['timeslots']) > 0


def test_test_season_data_is_forced_free(test_season_data):
    """Given season_test, then it is real AND carries zero forced games (its defining
    property — it is the forced-free config used by the e2e plan)."""
    data = test_season_data
    assert isinstance(data, dict)
    assert {g.name for g in data['grades']} == EXPECTED_GRADES
    # Oracle: forced-free by definition.
    assert data.get('forced_games', []) == []


def test_fixtures_are_mutation_isolated(real_2026_data):
    """Given a deepcopy fixture, mutating it must not leak into a re-load (proves
    each test gets a fresh copy — atoms mutate data['penalties'] in place)."""
    from config import load_season_data
    real_2026_data['penalties'] = {'SENTINEL': 'mutated'}
    fresh = load_season_data(2026)
    assert 'SENTINEL' not in fresh.get('penalties', {})


def test_clean_real_draw_is_real_committed_draw(clean_real_draw):
    """Given the committed clean draw, then it loads as a DrawStorage with the
    declared 108 games (oracle: the fixture's own num_games header)."""
    draw = clean_real_draw
    # Oracle: 108 games (the committed fixture's declared count).
    assert len(draw.games) == 108
    # Every game references a real grade and a real venue (not a stub draw).
    assert all(g.grade in EXPECTED_GRADES for g in draw.games)
    assert all(g.field_location for g in draw.games)


def test_green_suite_runner_builds_a_plan():
    """Given the batched runner module, then it builds a non-trivial batch plan
    whose first batch is the atoms package (the documented segfault-safe ordering)."""
    mod = importlib.import_module('scripts.run_green_suite')
    batches = mod.build_batches()
    # Oracle: at least 2 batches, first is the atoms package.
    assert len(batches) >= 2
    assert batches[0] == ['tests/atoms']
    # Every later batch is a non-empty list of test files.
    assert all(b and all(p.endswith('.py') for p in b) for b in batches[1:])
