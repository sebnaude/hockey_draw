"""spec-034 Unit C — Assurance B: DrawTester detects a failed constraint.

For every DrawTester check that lacked a detection test, prove the check FLAGS the
right rule when a draw breaks it and reports ZERO when it doesn't. No mocks/patches.

The other ~18 checks already have real-data detection tests — chiefly
`test_constraints_realdata.py` (clean-pass + injected-violation per check) plus
`test_tester_nihc_field_fill_order.py` and `test_locked_pairings_unit_c.py`. This
file covers ONLY the six checks with no prior coverage (see the spec-034
assurance-B mapping in docs/system/TESTING.md):
    _check_forced_games, _check_blocked_games, _check_preferred_games,
    _check_team_pair_no_concurrency, _check_club_no_concurrent_slot,
    _check_balanced_bye_spacing

WHY injected config: the committed `clean_real_draw` is a 5-week PARTIAL draw, so
the real full-season FORCED/BLOCKED config naturally shows deviations against it
(forced=14, blocked=2). To get an honest clean-vs-broken pair we therefore OVERWRITE
the relevant config key with a controlled entry the partial draw satisfies, then
mutate exactly that entry to break it. The two slot/round-shape checks
(`club_no_concurrent_slot`, `balanced_bye_spacing`) instead use small CONTROLLED
real DrawStorage fixtures with hand-computed oracles.
"""
from __future__ import annotations

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from models import Team, Club, Grade

NIHC = 'Newcastle International Hockey Centre'


# ====================================================================
# _check_forced_games  (hard: matched count must equal target)
# ====================================================================

def _slot_unique_scope(g):
    """A scope pinning EXACTLY one game: one game exists per (date, time, slot,
    field) in a clean draw, so this matches `g` and nothing else. No team matcher
    (team1/team2 are 'any'-team matchers — they'd match every game a team plays)."""
    return {'grade': g.grade, 'date': g.date, 'time': g.time, 'day_slot': g.day_slot,
            'field_name': g.field_name, 'field_location': g.field_location}


def test_forced_games_clean_then_flagged(clean_real_draw, real_2026_data):
    draw = clean_real_draw
    g = draw.games[0]
    entry = {**_slot_unique_scope(g), 'constraint': 'equal'}

    # CLEAN: force exactly the one game that sits in that slot -> satisfied -> zero.
    real_2026_data['forced_games'] = [{**entry, 'count': 1}]
    t = DrawTester(draw, real_2026_data)
    assert len(t._check_forced_games()) == 0

    # BROKEN: demand 2 games in a slot that holds 1 -> exactly one unmet entry.
    real_2026_data['forced_games'] = [{**entry, 'count': 2}]
    t2 = DrawTester(draw, real_2026_data)
    viols = t2._check_forced_games()
    assert len(viols) == 1
    assert viols[0].constraint == 'ForcedGames'


# ====================================================================
# _check_blocked_games  (hard: matched count must be zero)
# ====================================================================

def test_blocked_games_clean_then_flagged(clean_real_draw, real_2026_data):
    draw = clean_real_draw
    g = draw.games[0]

    # CLEAN: block a slot scope with a bogus field -> matches nothing -> zero.
    clean_scope = {**_slot_unique_scope(g), 'field_name': 'ZZ_NO_SUCH_FIELD'}
    real_2026_data['blocked_games'] = [clean_scope]
    t = DrawTester(draw, real_2026_data)
    assert len(t._check_blocked_games()) == 0

    # BROKEN: block the exact slot the game sits in -> exactly one violation.
    real_2026_data['blocked_games'] = [_slot_unique_scope(g)]
    t2 = DrawTester(draw, real_2026_data)
    viols = t2._check_blocked_games()
    assert len(viols) == 1
    assert viols[0].constraint == 'BlockedGames'
    assert viols[0].affected_games == [g.game_id]


# ====================================================================
# _check_preferred_games  (soft: deviation -> soft_pressure, not hard fail)
# ====================================================================

def test_preferred_games_clean_then_flagged(clean_real_draw, real_2026_data):
    draw = clean_real_draw
    g = draw.games[0]
    entry = {**_slot_unique_scope(g), 'constraint': 'equal'}

    # CLEAN: prefer exactly the one game in that slot -> penalty 0 -> zero.
    real_2026_data['preferred_games'] = [{**entry, 'count': 1}]
    t = DrawTester(draw, real_2026_data)
    assert len(t._check_preferred_games()) == 0

    # BROKEN: prefer 3 in a slot that holds 1 -> penalty = |1-3| = 2.
    real_2026_data['preferred_games'] = [{**entry, 'count': 3}]
    t2 = DrawTester(draw, real_2026_data)
    viols = t2._check_preferred_games()
    assert len(viols) == 1
    assert viols[0].constraint == 'PreferredGames'
    assert viols[0].metric_value == 2  # oracle: equal-deviation = |1-3|


# ====================================================================
# _check_team_pair_no_concurrency  (soft: pair sharing a (week, slot))
# ====================================================================

def test_team_pair_no_concurrency_clean_then_flagged(clean_real_draw, real_2026_data):
    draw = clean_real_draw
    # Find a real co-occurring pair (two teams in the same week+day_slot).
    from collections import defaultdict
    slot_teams = defaultdict(set)
    for x in draw.games:
        slot_teams[(x.week, x.day_slot)].add(x.team1)
        slot_teams[(x.week, x.day_slot)].add(x.team2)
    cooccur = None
    for teams in slot_teams.values():
        ts = sorted(teams)
        if len(ts) >= 2:
            cooccur = (ts[0], ts[1])
            break
    assert cooccur, 'fixture should have at least one shared slot'

    # Find a NON-co-occurring pair: two teams whose appearance-slots never overlap.
    team_slots = defaultdict(set)
    for x in draw.games:
        team_slots[x.team1].add((x.week, x.day_slot))
        team_slots[x.team2].add((x.week, x.day_slot))
    all_teams = sorted(team_slots)
    noncooccur = None
    for i in range(len(all_teams)):
        for j in range(i + 1, len(all_teams)):
            a, b = all_teams[i], all_teams[j]
            if not (team_slots[a] & team_slots[b]):
                noncooccur = (a, b)
                break
        if noncooccur:
            break
    assert noncooccur, 'expected at least one never-concurrent pair'

    # CLEAN: declare the never-concurrent pair -> zero.
    real_2026_data.setdefault('constraint_defaults', {})['TEAM_PAIR_NO_CONCURRENCY'] = [noncooccur]
    t = DrawTester(draw, real_2026_data)
    assert len(t._check_team_pair_no_concurrency()) == 0

    # BROKEN: declare the co-occurring pair -> at least one flagged slot.
    real_2026_data['constraint_defaults']['TEAM_PAIR_NO_CONCURRENCY'] = [cooccur]
    t2 = DrawTester(draw, real_2026_data)
    viols = t2._check_team_pair_no_concurrency()
    assert len(viols) >= 1
    assert all(v.constraint == 'TeamPairNoConcurrency' for v in viols)


# ====================================================================
# _check_club_no_concurrent_slot  (controlled real draw)
# ====================================================================
#
# Cap = 1 + slack(=0). A club with 2 games in the SAME (week, slot, location)
# (even on different NIHC fields) exceeds the cap.
# Oracle (broken): both Tigers games at (wk1, slot1, NIHC) -> Tigers over by 1;
#   Wests too (both their games share the slot) -> exactly 2 violations.
# Oracle (clean): second game moved to slot 2 -> each club <=1 per slot -> zero.

def _club_slot_data():
    tigers = Club(name='Tigers', home_field=NIHC)
    wests = Club(name='Wests', home_field=NIHC)
    teams = [Team(name='Tigers 3rd', club=tigers, grade='3rd'),
             Team(name='Tigers 4th', club=tigers, grade='4th'),
             Team(name='Wests 3rd', club=wests, grade='3rd'),
             Team(name='Wests 4th', club=wests, grade='4th')]
    grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd']),
              Grade(name='4th', teams=['Tigers 4th', 'Wests 4th'])]
    return {'teams': teams, 'clubs': [tigers, wests], 'grades': grades,
            'num_rounds': {'max': 1, '3rd': 1, '4th': 1}, 'constraint_slack': {}}


def _club_slot_game(gid, t1, t2, grade, slot, field):
    return StoredGame(game_id=gid, team1=t1, team2=t2, grade=grade, week=1,
                      round_no=1, date='2026-03-22', day='Sunday', time='10:00',
                      day_slot=slot, field_name=field, field_location=NIHC)


def test_club_no_concurrent_slot_flagged_when_two_games_one_slot():
    data = _club_slot_data()
    draw = DrawStorage(description='collide', num_weeks=1, num_games=2, games=[
        _club_slot_game('G1', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 'EF'),
        _club_slot_game('G2', 'Tigers 4th', 'Wests 4th', '4th', 1, 'WF'),
    ])
    viols = DrawTester(draw, data)._check_club_no_concurrent_slot()
    # Oracle: Tigers AND Wests each have 2 games in (wk1, slot1, NIHC) -> 2 violations.
    assert len(viols) == 2
    assert all(v.constraint == 'ClubNoConcurrentSlot' and v.metric_value == 1 for v in viols)


def test_club_no_concurrent_slot_clean_when_different_slots():
    data = _club_slot_data()
    draw = DrawStorage(description='ok', num_weeks=1, num_games=2, games=[
        _club_slot_game('G1', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 'EF'),
        _club_slot_game('G2', 'Tigers 4th', 'Wests 4th', '4th', 2, 'WF'),
    ])
    viols = DrawTester(draw, data)._check_club_no_concurrent_slot()
    assert len(viols) == 0


# ====================================================================
# _check_balanced_bye_spacing  (controlled real draw)
# ====================================================================
#
# max_r=6, grade 'X', games_t=4 -> byes_t=2 -> S = ideal_bye_gap(6,2) = 6//2-1 = 2.
# Each team plays the rounds where a game exists; byes = the rounds it doesn't.
# Oracle (broken): games in rounds 1-4 -> both teams bye in {5,6}, gap 1 <= 2
#   -> flagged for BOTH teams = 2 violations.
# Oracle (clean): games in rounds {1,2,4,5} -> byes {3,6}, gap 3 > 2 -> zero.

def _bye_data():
    c = Club(name='Tigers', home_field=NIHC)
    teams = [Team(name='A', club=c, grade='X'), Team(name='B', club=c, grade='X')]
    grades = [Grade(name='X', teams=['A', 'B'])]
    return {'teams': teams, 'clubs': [c], 'grades': grades,
            'num_rounds': {'max': 6, 'X': 4}, 'constraint_slack': {}}


def _bye_game(gid, rnd):
    return StoredGame(game_id=gid, team1='A', team2='B', grade='X', week=rnd,
                      round_no=rnd, date='2026-03-22', day='Sunday', time='10:00',
                      day_slot=1, field_name='EF', field_location=NIHC)


def test_balanced_bye_spacing_flagged_when_byes_bunched():
    data = _bye_data()
    draw = DrawStorage(description='bunched', num_weeks=6, num_games=4,
                       games=[_bye_game(f'G{r}', r) for r in (1, 2, 3, 4)])
    viols = DrawTester(draw, data)._check_balanced_bye_spacing()
    # Oracle: byes {5,6}, gap 1 <= S=2, flagged for A and B -> 2 violations.
    assert len(viols) == 2
    assert all(v.constraint == 'BalancedByeSpacing' for v in viols)


def test_balanced_bye_spacing_clean_when_byes_spread():
    data = _bye_data()
    draw = DrawStorage(description='spread', num_weeks=6, num_games=4,
                       games=[_bye_game(f'G{r}', r) for r in (1, 2, 4, 5)])
    viols = DrawTester(draw, data)._check_balanced_bye_spacing()
    # Oracle: byes {3,6}, gap 3 > S=2 -> zero.
    assert len(viols) == 0
