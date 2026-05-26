"""spec-034 Unit D — Assurance C: soft constraints are measured in a draw.

Proves that bending a SOFT atom shows up in the `soft_pressure` rollup with the
right hand-computed `total_penalty`, and that honouring it leaves the atom absent
from `soft_pressure`. No mocks/patches.

`soft_pressure` is populated by `ViolationBreakdown.from_violations`, which is
EXACTLY what the `ViolationReport.breakdown` property invokes
(`analytics/tester.py` — `breakdown` returns `ViolationBreakdown.from_violations(
self.violations)`). We build the breakdown from a single check's real violations
so each atom's rollup is asserted in isolation (robust on small fixtures, identical
code path to `report.breakdown.soft_pressure`).

SCOPE — which soft atoms are measured HERE: the soft atoms whose DrawTester check
emits a `metric_value` (the only thing `soft_pressure` rolls up): PreferredGames,
ClubNoConcurrentSlot, TeamConflict (and the symmetry NIHC-fill pair, already
detection-tested in test_tester_nihc_field_fill_order.py). The remaining soft atoms
(`*RegenSoft` analogues, SoftLexMatchupOrdering) have NO tester check — they are
measured at the SOLVER level via their `data['penalties'][...]` buckets reaching the
objective (proven for the live engine in tests/atoms/test_spec034_assurance_a_realdata
and the spec-027/033 regen tests). See docs/system/TESTING.md for the full map.
Note: `over_limit` counts metric violations; `at_limit` is unused (always 0) — never
asserted here.
"""
from __future__ import annotations

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester, ViolationBreakdown
from models import Team, Club, Grade

NIHC = 'Newcastle International Hockey Centre'


def _soft_pressure(tester, check_name):
    """The exact production rollup (= ViolationReport.breakdown.soft_pressure),
    isolated to one check's violations."""
    viols = getattr(tester, check_name)()
    return ViolationBreakdown.from_violations(viols).soft_pressure


# ====================================================================
# PreferredGames  (soft group)
# ====================================================================

def _slot_scope(g):
    return {'grade': g.grade, 'date': g.date, 'time': g.time, 'day_slot': g.day_slot,
            'field_name': g.field_name, 'field_location': g.field_location}


def test_preferred_games_soft_pressure_bent_then_honoured(clean_real_draw, real_2026_data):
    draw = clean_real_draw
    g = draw.games[0]
    entry = {**_slot_scope(g), 'constraint': 'equal'}

    # BENT: prefer 3 in a slot holding 1 -> equal-penalty |1-3| = 2.
    real_2026_data['preferred_games'] = [{**entry, 'count': 3}]
    sp = _soft_pressure(DrawTester(draw, real_2026_data), '_check_preferred_games')
    assert 'PreferredGames' in sp
    assert sp['PreferredGames']['total_penalty'] == 2.0   # oracle: |1-3|
    assert sp['PreferredGames']['over_limit'] == 1
    assert sp['PreferredGames']['worst_value'] == 2.0

    # HONOURED: prefer exactly the 1 game present -> penalty 0 -> atom absent.
    real_2026_data['preferred_games'] = [{**entry, 'count': 1}]
    sp2 = _soft_pressure(DrawTester(draw, real_2026_data), '_check_preferred_games')
    assert 'PreferredGames' not in sp2


# ====================================================================
# ClubNoConcurrentSlot  (soft, hard cap 1+slack with soft overage)
# ====================================================================

def _club_data():
    tigers = Club(name='Tigers', home_field=NIHC)
    wests = Club(name='Wests', home_field=NIHC)
    teams = [Team(name='Tigers 3rd', club=tigers, grade='3rd'),
             Team(name='Tigers 4th', club=tigers, grade='4th'),
             Team(name='Wests 3rd', club=wests, grade='3rd'),
             Team(name='Wests 4th', club=wests, grade='4th')]
    grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd']),
              Grade(name='4th', teams=['Tigers 4th', 'Wests 4th'])]
    return {'teams': teams, 'clubs': [tigers, wests], 'grades': grades,
            'num_rounds': {'max': 1}, 'constraint_slack': {}}


def _game(gid, t1, t2, grade, slot, field):
    return StoredGame(game_id=gid, team1=t1, team2=t2, grade=grade, week=1,
                      round_no=1, date='2026-03-22', day='Sunday', time='10:00',
                      day_slot=slot, field_name=field, field_location=NIHC)


def test_club_no_concurrent_soft_pressure_bent_then_honoured():
    # BENT: Tigers AND Wests each have 2 games in (wk1, slot1, NIHC); cap=1.
    bent = DrawStorage(description='bent', num_weeks=1, num_games=2, games=[
        _game('G1', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 'EF'),
        _game('G2', 'Tigers 4th', 'Wests 4th', '4th', 1, 'WF')])
    sp = _soft_pressure(DrawTester(bent, _club_data()), '_check_club_no_concurrent_slot')
    assert 'ClubNoConcurrentSlot' in sp
    # Oracle: two clubs each over by (2 - cap=1) = 1 -> total_penalty 2, over_limit 2.
    assert sp['ClubNoConcurrentSlot']['over_limit'] == 2
    assert sp['ClubNoConcurrentSlot']['total_penalty'] == 2.0
    assert sp['ClubNoConcurrentSlot']['worst_value'] == 1.0

    # HONOURED: second game in slot 2 -> each club <=1 per slot -> atom absent.
    honoured = DrawStorage(description='ok', num_weeks=1, num_games=2, games=[
        _game('G1', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 'EF'),
        _game('G2', 'Tigers 4th', 'Wests 4th', '4th', 2, 'WF')])
    sp2 = _soft_pressure(DrawTester(honoured, _club_data()), '_check_club_no_concurrent_slot')
    assert 'ClubNoConcurrentSlot' not in sp2


# ====================================================================
# TeamConflict  (soft group — spec-033 made it soft-only)
# ====================================================================

def _conflict_data():
    c = Club(name='Tigers', home_field=NIHC)
    w = Club(name='Wests', home_field=NIHC)
    teams = [Team(name='Tigers 3rd', club=c, grade='3rd'),
             Team(name='Tigers 4th', club=c, grade='4th'),
             Team(name='Wests 3rd', club=w, grade='3rd'),
             Team(name='Wests 4th', club=w, grade='4th')]
    grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd']),
              Grade(name='4th', teams=['Tigers 4th', 'Wests 4th'])]
    return {'teams': teams, 'clubs': [c, w], 'grades': grades,
            'num_rounds': {'max': 1},
            'team_conflicts': [('Tigers 3rd', 'Tigers 4th')]}


def test_team_conflict_soft_pressure_bent_then_honoured():
    # BENT: the two conflicting teams both play week 1, slot 1 -> 1 clash, metric 1.0.
    bent = DrawStorage(description='clash', num_weeks=1, num_games=2, games=[
        _game('G1', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 'EF'),
        _game('G2', 'Tigers 4th', 'Wests 4th', '4th', 1, 'WF')])
    sp = _soft_pressure(DrawTester(bent, _conflict_data()), '_check_team_conflict')
    assert 'TeamConflict' in sp
    # Oracle: exactly one concurrent appearance -> total_penalty 1.0, over_limit 1.
    assert sp['TeamConflict']['over_limit'] == 1
    assert sp['TeamConflict']['total_penalty'] == 1.0

    # HONOURED: conflicting teams in different slots -> no clash -> atom absent.
    honoured = DrawStorage(description='ok', num_weeks=1, num_games=2, games=[
        _game('G1', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 'EF'),
        _game('G2', 'Tigers 4th', 'Wests 4th', '4th', 2, 'WF')])
    sp2 = _soft_pressure(DrawTester(honoured, _conflict_data()), '_check_team_conflict')
    assert 'TeamConflict' not in sp2
