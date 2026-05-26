"""spec-025 Unit C — locked_pairing_outcomes metadata + tester check + registry.

GWT, no mocks. Uses real DrawStorage / StoredGame / DrawTester / DrawVersionManager
objects. Oracles hand-derived in comments below.

Hand oracle (metadata, satisfied pin)
-------------------------------------
Pin: {teams: ['Tigers', 'Wests'], grade: 'PHL', date: '2026-03-22'}
Solution keys (only those with value 1 count as scheduled):
  K1 = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 3, '11:30', 1, '2026-03-22',
        1, 'EF', 'Newcastle International Hockey Centre')  -> value 1
The pin matcher requires: date == k[7] ('2026-03-22'), grade == k[2] ('PHL'),
and the unordered pair {Tigers, Wests} via _club_matches prefix logic:
  _club_matches('Tigers PHL', 'Tigers') is True  (startswith 'Tigers ')
  _club_matches('Wests PHL',  'Wests')  is True  (startswith 'Wests ')
=> K1 matches. No other scheduled key matches.
=> matched_count == 1, satisfied == True,
   resolved_time == '11:30', resolved_day_slot == 3,
   resolved_field_name == 'EF',
   resolved_field_location == 'Newcastle International Hockey Centre'.

Hand oracle (tester check)
--------------------------
Same pin. The check requires EXACTLY one game matching the pairing+grade+date.
- ON its date  -> exactly 1 match  -> count == 1 -> NO violation.
- MOVED off its date (game's date '2026-03-29' != pin date '2026-03-22')
  -> 0 matches -> count == 0 != 1 -> ONE violation, constraint 'LockedPairings',
  message naming the pin.
"""

import pytest

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from analytics.versioning import DrawVersionManager
from constraints.registry import CONSTRAINT_REGISTRY, get_all_tester_methods
from models import Club, Team, Grade


NIHC = 'Newcastle International Hockey Centre'


@pytest.fixture
def clubs():
    return [
        Club(name='Tigers', home_field=NIHC),
        Club(name='Wests', home_field=NIHC),
    ]


@pytest.fixture
def teams(clubs):
    tigers, wests = clubs
    return [
        Team(name='Tigers PHL', club=tigers, grade='PHL'),
        Team(name='Wests PHL', club=wests, grade='PHL'),
    ]


@pytest.fixture
def grades():
    return [Grade(name='PHL', teams=['Tigers PHL', 'Wests PHL'])]


@pytest.fixture
def data(clubs, teams, grades):
    return {
        'clubs': clubs,
        'teams': teams,
        'grades': grades,
        'locked_pairings': [
            {
                'teams': ['Tigers', 'Wests'],
                'grade': 'PHL',
                'date': '2026-03-22',
                'description': 'Locked wk1 Tigers v Wests PHL',
            }
        ],
    }


def _game_on(date):
    return StoredGame(
        game_id='G00001',
        team1='Tigers PHL',
        team2='Wests PHL',
        grade='PHL',
        week=1,
        round_no=1,
        date=date,
        day='Sunday',
        time='11:30',
        day_slot=3,
        field_name='EF',
        field_location=NIHC,
    )


# ----------------------------------------------------------------------------
# DoD 7 — metadata: locked_pairing_outcomes
# ----------------------------------------------------------------------------

def test_metadata_records_satisfied_pin_with_resolved_slot(tmp_path, data):
    """GIVEN a solution with the pinned pairing scheduled on its date,
    WHEN _build_draw_metadata runs, THEN locked_pairing_outcomes records the pin
    as satisfied with matched_count 1 and the resolved time/slot/field."""
    mgr = DrawVersionManager(str(tmp_path))

    # Full 11-tuple key matching the documented variable layout.
    key = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 3, '11:30', 1,
           '2026-03-22', 1, 'EF', NIHC)
    solution = {key: 1}

    meta = mgr._build_draw_metadata(
        draw=None, solution=solution, data=data, mode='simple',
        timestamp='2026-05-23T00:00:00')

    outcomes = meta['locked_pairing_outcomes']
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o['satisfied'] is True
    assert o['matched_count'] == 1
    assert o['resolved_time'] == '11:30'
    assert o['resolved_day_slot'] == 3
    assert o['resolved_field_name'] == 'EF'
    assert o['resolved_field_location'] == NIHC
    # config fields are preserved (copied from the entry)
    assert o['teams'] == ['Tigers', 'Wests']
    assert o['date'] == '2026-03-22'


def test_metadata_records_unsatisfied_pin(tmp_path, data):
    """GIVEN a solution where the pinned pairing is NOT on its date,
    THEN the outcome is satisfied=False with matched_count 0 and null resolved
    fields."""
    mgr = DrawVersionManager(str(tmp_path))

    # Scheduled on a different date — does not satisfy the pin.
    key = ('Tigers PHL', 'Wests PHL', 'PHL', 'Sunday', 3, '11:30', 2,
           '2026-03-29', 2, 'EF', NIHC)
    solution = {key: 1}

    meta = mgr._build_draw_metadata(
        draw=None, solution=solution, data=data, mode='simple',
        timestamp='2026-05-23T00:00:00')

    o = meta['locked_pairing_outcomes'][0]
    assert o['satisfied'] is False
    assert o['matched_count'] == 0
    assert o['resolved_time'] is None
    assert o['resolved_day_slot'] is None
    assert o['resolved_field_name'] is None
    assert o['resolved_field_location'] is None


# ----------------------------------------------------------------------------
# DoD 8 — tester check: _check_locked_pairings
# ----------------------------------------------------------------------------

def test_tester_no_violation_when_pin_on_date(data):
    """GIVEN a draw with the pinned pairing on its date, THEN no violation."""
    draw = DrawStorage(description='on-date', num_weeks=1, num_games=1,
                       games=[_game_on('2026-03-22')])
    tester = DrawTester(draw, data)
    violations = tester._check_locked_pairings()
    assert violations == []


def test_tester_flags_pin_moved_off_date(data):
    """GIVEN a draw where the pinned pairing was moved off its date,
    THEN exactly one LockedPairings violation naming the pairing."""
    draw = DrawStorage(description='moved', num_weeks=2, num_games=1,
                       games=[_game_on('2026-03-29')])  # wrong date
    tester = DrawTester(draw, data)
    violations = tester._check_locked_pairings()
    assert len(violations) == 1
    v = violations[0]
    assert v.constraint == 'LockedPairings'
    assert 'Locked wk1 Tigers v Wests PHL' in v.message


def test_tester_empty_locked_pairings_no_violation():
    """GIVEN no locked_pairings in data, THEN the check is a no-op."""
    draw = DrawStorage(description='none', num_weeks=1, num_games=1,
                       games=[_game_on('2026-03-22')])
    tester = DrawTester(draw, {'teams': [], 'locked_pairings': []})
    assert tester._check_locked_pairings() == []


def test_full_violation_check_runs_locked_pairings(data):
    """GIVEN a date-moved pin, WHEN the full run_violation_check executes,
    THEN a LockedPairings violation surfaces (the check is registered in the
    ordered check-list and runs as tester_only)."""
    draw = DrawStorage(description='moved', num_weeks=2, num_games=1,
                       games=[_game_on('2026-03-29')])
    tester = DrawTester(draw, data)
    report = tester.run_violation_check()
    names = {v.constraint for v in report.violations}
    assert 'LockedPairings' in names


# ----------------------------------------------------------------------------
# DoD 8 — registry entry + count
# ----------------------------------------------------------------------------

def test_registry_has_locked_pairings_entry():
    info = CONSTRAINT_REGISTRY['LockedPairings']
    assert info.tester_only is True
    assert info.tester_check_methods == ['_check_locked_pairings']
    # tester_violation_names MUST match what _check_locked_pairings emits.
    assert info.tester_violation_names == ['LockedPairings']
    assert info.solver_class_names == []


def test_registry_count_is_49():
    """spec-025 adds LockedPairings: 37 -> 38.
    spec-027 adds 13 regeneration soft-analogue (`*RegenSoft`) atoms: 38 -> 51.
    spec-030 deletes PHLAnd2ndConcurrencyAtBroadmeadow: 51 -> 50.
    spec-031 removes the ClubFieldConcentration tester-only diagnostic: 50 -> 49.
    spec-032 retags only (49); spec-036 retains the ClubVsClubAlignment anchor (49)."""
    assert len(CONSTRAINT_REGISTRY) == 49


def test_check_method_is_registered():
    """_check_locked_pairings is covered by the registry (mirrors
    test_every_drawtester_check_in_registry)."""
    assert '_check_locked_pairings' in set(get_all_tester_methods())
