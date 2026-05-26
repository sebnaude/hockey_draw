"""spec-034 Unit B — Assurance A gap-fill: atoms enforce on real data.

These cover the registered atoms that lacked a dedicated `tests/atoms/` file:
the four legacy *solver* atoms (EqualGamesAndBalanceMatchUps, FiftyFiftyHomeandAway,
TeamConflict, PreferredTimes) and the config-filter atom BlockedGames.

NO mocks / patches / monkeypatch. Each test builds a REAL CP-SAT model from REAL
domain objects (Club/Team/Grade/Timeslot/PlayingField) and drives the LIVE
enforcement path — the `UnifiedConstraintEngine` methods the production solver
calls, and the real `generate_X` variable filter — then asserts a hand-computed
oracle for both a satisfying ("enforce") and a violating ("bite") assignment.

(ForcedGames and LockedPairings — the other two tester-only config filters — are
covered by the existing real-data tests `test_forced_games_count_rules.py`,
`test_forced_games_multi_scope.py` and `test_locked_pairings_generate.py`; see the
spec-034 assurance-A mapping in docs/system/TESTING.md. Not duplicated here.)
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from models import PlayingField, Team, Club, Grade, Timeslot
from constraints.unified import UnifiedConstraintEngine

NIHC = 'Newcastle International Hockey Centre'
MAITLAND = 'Maitland Park'


def _build_X(model, games, timeslots):
    """Real decision vars over (games x timeslots), keyed by the production 11-tuple."""
    X = {}
    for (t1, t2, grade) in games:
        for t in timeslots:
            key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date,
                   t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}')
    return X


# ====================================================================
# EqualGamesAndBalanceMatchUps  (live: _equal_games_balanced_matchups)
# ====================================================================
#
# Scenario: one grade '3rd', three teams A,B,C, num_rounds=2.
# Live rule: each team's scheduled games == R (=2); each pair sum in [base, base+1]
# with base = R//T = 2//3 = 0  -> each of the 3 pairs scheduled 0 or 1 time.
# Oracle (enforce): total games = 3 teams * 2 / 2 = 3; the only way every team
#   reaches 2 with each pair <=1 is each pair exactly once -> FEASIBLE.
# Oracle (bite): forcing team A to play 3 games contradicts ==2 -> INFEASIBLE.

def _equal_games_setup():
    club = Club(name='Tigers', home_field=NIHC)
    clubB = Club(name='Wests', home_field=NIHC)
    clubC = Club(name='Norths', home_field=NIHC)
    teams = [Team(name='Tigers 3rd', club=club, grade='3rd'),
             Team(name='Wests 3rd', club=clubB, grade='3rd'),
             Team(name='Norths 3rd', club=clubC, grade='3rd')]
    grades = [Grade(name='3rd', teams=[t.name for t in teams])]
    ef = PlayingField(location=NIHC, name='EF')
    wf = PlayingField(location=NIHC, name='WF')
    # 3 weeks x 2 fields = 6 slots: ample room for 3 games, 1 per team-week max.
    timeslots = []
    for wk, date in enumerate(['2026-03-22', '2026-03-29', '2026-04-05'], 1):
        for f in (ef, wf):
            timeslots.append(Timeslot(date=date, day='Sunday', time='10:00',
                                      week=wk, day_slot=1, field=f, round_no=wk))
    games = [('Norths 3rd', 'Tigers 3rd', '3rd'),
             ('Norths 3rd', 'Wests 3rd', '3rd'),
             ('Tigers 3rd', 'Wests 3rd', '3rd')]
    data = {'teams': teams, 'clubs': [club, clubB, clubC], 'grades': grades,
            'games': games, 'timeslots': timeslots, 'fields': [ef, wf],
            'num_rounds': {'3rd': 2}}
    return data, games, timeslots


def test_equal_games_enforces_each_team_plays_R():
    data, games, timeslots = _equal_games_setup()
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    n = eng._equal_games_balanced_matchups()
    assert n > 0
    # No team in two slots of the same week (physical sanity, so "==2" means 2 distinct weeks).
    for wk in (1, 2, 3):
        for team in ('Tigers 3rd', 'Wests 3rd', 'Norths 3rd'):
            wk_vars = [v for k, v in X.items() if k[6] == wk and team in (k[0], k[1])]
            model.Add(sum(wk_vars) <= 1)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # Oracle: each team exactly 2 games; total exactly 3.
    for team in ('Tigers 3rd', 'Wests 3rd', 'Norths 3rd'):
        played = sum(solver.Value(v) for k, v in X.items() if team in (k[0], k[1]))
        assert played == 2, (team, played)
    total = sum(solver.Value(v) for v in X.values())
    assert total == 3


def test_equal_games_bites_when_team_overplays():
    data, games, timeslots = _equal_games_setup()
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    eng._equal_games_balanced_matchups()
    # Force Tigers 3rd to play 3 games — contradicts the live "== 2" rule.
    tigers_vars = [v for k, v in X.items() if 'Tigers 3rd' in (k[0], k[1])]
    model.Add(sum(tigers_vars) == 3)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.INFEASIBLE


# ====================================================================
# FiftyFiftyHomeandAway  (live: _fifty_fifty_home_away)
# ====================================================================
#
# Scenario: Maitland 3rd vs Tigers 3rd meet twice. Maitland's home = Maitland Park.
# Live rule per pair: hc*2 within [tc-1, tc+1], hc=#home(Maitland), tc=total.
# Oracle (enforce): with both games scheduled (tc=2), hc must be 1 (hc=0 -> 0>=1 false;
#   hc=2 -> 4<=3 false) -> exactly one game at Maitland Park.
# Oracle (bite): forcing BOTH meetings at Maitland Park (hc=2, tc=2) -> 4<=3 -> INFEASIBLE.

def _fifty_fifty_setup():
    maitland = Club(name='Maitland', home_field=MAITLAND)
    tigers = Club(name='Tigers', home_field=NIHC)
    teams = [Team(name='Maitland 3rd', club=maitland, grade='3rd'),
             Team(name='Tigers 3rd', club=tigers, grade='3rd')]
    grades = [Grade(name='3rd', teams=[t.name for t in teams])]
    mp = PlayingField(location=MAITLAND, name='Main')
    ef = PlayingField(location=NIHC, name='EF')
    # Always offer BOTH a home (Maitland Park) and an away (Broadmeadow) option,
    # otherwise `_fifty_fifty_home_away` no-ops the pair (it skips pairs lacking
    # either a home or an away candidate). week1+week2 at MP, week3 at EF.
    slots = [('2026-03-22', mp), ('2026-03-29', mp), ('2026-04-05', ef)]
    timeslots = []
    for wk, (date, f) in enumerate(slots, 1):
        timeslots.append(Timeslot(date=date, day='Sunday', time='10:00',
                                  week=wk, day_slot=1, field=f, round_no=wk))
    games = [('Maitland 3rd', 'Tigers 3rd', '3rd')]
    data = {'teams': teams, 'clubs': [maitland, tigers], 'grades': grades,
            'games': games, 'timeslots': timeslots, 'fields': [mp, ef],
            'num_rounds': {'3rd': 2}}
    return data, games, timeslots


def test_fifty_fifty_enforces_one_home_of_two():
    data, games, timeslots = _fifty_fifty_setup()
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    n = eng._fifty_fifty_home_away()
    assert n > 0
    model.Add(sum(X.values()) == 2)  # both meetings played
    solver = cp_model.CpSolver()
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    home = sum(solver.Value(v) for k, v in X.items() if k[10] == MAITLAND)
    assert home == 1  # oracle: exactly one of the two at Maitland Park


def test_fifty_fifty_bites_when_both_home():
    data, games, timeslots = _fifty_fifty_setup()
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    n = eng._fifty_fifty_home_away()
    assert n > 0  # the pair IS constrained (it has both home and away options)
    # Force both scheduled games to be at Maitland Park (hc=2) with no away game
    # (tc=2): hc*2=4 <= tc+1=3 is violated -> INFEASIBLE.
    model.Add(sum(v for k, v in X.items() if k[10] == MAITLAND) == 2)
    model.Add(sum(X.values()) == 2)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.INFEASIBLE


# ====================================================================
# TeamConflict  (live soft: _team_conflict_soft)
# ====================================================================
#
# Scenario: two teams declared a conflict pair; both must play in week 1.
# Live rule: per (week, day_slot) where both could appear, a penalty BoolVar
#   p >= sum(v1)+sum(v2)-1. Soft (never blocks feasibility).
# Oracle (bite): only ONE day_slot in week 1 -> both forced into it -> p>=1, and
#   minimising the penalty cannot avoid it -> conflict penalty == 1.
# Oracle (enforce/honoured): TWO day_slots -> minimiser separates them -> penalty 0.

def _team_conflict_setup(n_slots):
    club = Club(name='Tigers', home_field=NIHC)
    teams = [Team(name='Tigers 3rd', club=club, grade='3rd'),
             Team(name='Tigers 4th', club=club, grade='4th')]
    grades = [Grade(name='3rd', teams=['Tigers 3rd']),
              Grade(name='4th', teams=['Tigers 4th'])]
    ef = PlayingField(location=NIHC, name='EF')
    timeslots = [Timeslot(date='2026-03-22', day='Sunday', time=f'1{s}:00',
                          week=1, day_slot=s, field=ef, round_no=1)
                 for s in range(1, n_slots + 1)]
    # Each team needs an opponent to form a game; add filler same-grade teams.
    teams += [Team(name='Wests 3rd', club=Club(name='Wests', home_field=NIHC), grade='3rd'),
              Team(name='Wests 4th', club=Club(name='Wests', home_field=NIHC), grade='4th')]
    grades[0].teams.append('Wests 3rd')
    grades[1].teams.append('Wests 4th')
    games = [('Tigers 3rd', 'Wests 3rd', '3rd'), ('Tigers 4th', 'Wests 4th', '4th')]
    data = {'teams': teams, 'clubs': [club, teams[2].club, teams[3].club], 'grades': grades,
            'games': games, 'timeslots': timeslots, 'fields': [ef], 'num_rounds': {},
            'team_conflicts': [('Tigers 3rd', 'Tigers 4th')], 'penalties': {},
            'penalty_weights': {'TeamConflict': 200000}}
    return data, games, timeslots


def _solve_min_penalty(model, X, data):
    pen = data['penalties']['TeamConflict']['penalties']
    model.Minimize(sum(pen))
    # Force both conflict teams to play their single game.
    model.Add(sum(v for k, v in X.items() if 'Tigers 3rd' in (k[0], k[1])) == 1)
    model.Add(sum(v for k, v in X.items() if 'Tigers 4th' in (k[0], k[1])) == 1)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status, solver, pen


def test_team_conflict_bites_with_single_slot():
    data, games, timeslots = _team_conflict_setup(n_slots=1)
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    n = eng._team_conflict_soft()
    assert n >= 1  # at least one (week,slot) where both could appear
    status, solver, pen = _solve_min_penalty(model, X, data)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # Oracle: both teams squeezed into the one slot -> penalty cannot be avoided.
    assert sum(solver.Value(p) for p in pen) == 1


def test_team_conflict_honoured_with_two_slots():
    data, games, timeslots = _team_conflict_setup(n_slots=2)
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    eng._team_conflict_soft()
    status, solver, pen = _solve_min_penalty(model, X, data)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # Oracle: minimiser separates the two teams into different slots -> zero penalty.
    assert sum(solver.Value(p) for p in pen) == 0


# ====================================================================
# PreferredTimes  (live soft: _preferred_times — no-play penalty)
# ====================================================================
#
# Scenario: a no-play preference on a specific time for a club; one matching game.
# Live rule: each matching X-var gets a penalty IntVar pv == X[key]. Soft.
# Oracle (bite): force the matching game on -> pv == 1.
# Oracle (honoured): an alternative non-matching slot lets the minimiser pick pv == 0.

def _preferred_times_setup():
    club = Club(name='Tigers', home_field=NIHC)
    wests = Club(name='Wests', home_field=NIHC)
    teams = [Team(name='Tigers 3rd', club=club, grade='3rd'),
             Team(name='Wests 3rd', club=wests, grade='3rd')]
    grades = [Grade(name='3rd', teams=['Tigers 3rd', 'Wests 3rd'])]
    ef = PlayingField(location=NIHC, name='EF')
    # Two candidate slots in week 1: 19:00 (the disfavoured time) and 10:00.
    timeslots = [Timeslot(date='2026-03-22', day='Sunday', time='19:00', week=1, day_slot=5, field=ef, round_no=1),
                 Timeslot(date='2026-03-22', day='Sunday', time='10:00', week=1, day_slot=1, field=ef, round_no=1)]
    games = [('Tigers 3rd', 'Wests 3rd', '3rd')]
    data = {'teams': teams, 'clubs': [club, wests], 'grades': grades,
            'games': games, 'timeslots': timeslots, 'fields': [ef], 'num_rounds': {},
            'preference_no_play': {
                'tigers_no_1900': {'club': 'Tigers', 'time': '19:00',
                                   'description': 'spec-034 time-only no-play test'},
            },
            'penalties': {}, 'penalty_weights': {'PreferredTimesConstraint': 10000000}}
    return data, games, timeslots


def test_preferred_times_penalises_matching_slot():
    data, games, timeslots = _preferred_times_setup()
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    n = eng._preferred_times()
    pen = data['penalties']['PreferredTimesConstraint']['penalties']
    assert n >= 1 and len(pen) >= 1  # the 19:00 var got a penalty
    # Bite: force the 19:00 game on -> its penalty var == 1.
    nineteen = [v for k, v in X.items() if k[5] == '19:00']
    assert len(nineteen) == 1
    model.Add(nineteen[0] == 1)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert sum(solver.Value(p) for p in pen) == 1


def test_preferred_times_honoured_when_alternative_slot_chosen():
    data, games, timeslots = _preferred_times_setup()
    model = cp_model.CpModel()
    X = _build_X(model, games, timeslots)
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    eng._preferred_times()
    pen = data['penalties']['PreferredTimesConstraint']['penalties']
    # Honoured: exactly one game, minimise penalty -> picks the 10:00 slot, pv == 0.
    model.Add(sum(X.values()) == 1)
    model.Minimize(sum(pen))
    solver = cp_model.CpSolver()
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert sum(solver.Value(p) for p in pen) == 0
    # And the chosen game is the 10:00 one.
    assert sum(solver.Value(v) for k, v in X.items() if k[5] == '10:00') == 1


# ====================================================================
# BlockedGames  (config filter in generate_X — variable elimination)
# ====================================================================
#
# Live rule: a BLOCKED_GAMES entry removes matching decision vars entirely.
# Oracle: with a club-vs-club block in place, generate_X produces ZERO vars for
#   that matchup, while the same build with no block produces > 0.

def test_blocked_games_eliminates_matching_variables(test_season_data):
    from copy import deepcopy
    from utils import generate_X

    base = test_season_data
    # Pick a real 3rd-grade matchup that exists unblocked.
    g3 = next(g for g in base['grades'] if g.name == '3rd')
    t1, t2 = sorted(g3.teams)[:2]

    model0 = cp_model.CpModel()
    X0, _ = generate_X(model0, deepcopy(base))
    unblocked = [k for k in X0 if len(k) >= 11 and {k[0], k[1]} == {t1, t2}]
    assert len(unblocked) > 0, 'expected the matchup to exist when unblocked'
    # Choose one real round_no this matchup plays in, avoiding rounds 1-2 (the
    # perennial Broadmeadow-only block) so the narrow block can't gridlock validation.
    rounds = sorted({k[8] for k in unblocked if k[8] not in (1, 2)})
    assert rounds, 'matchup should have candidate vars outside rounds 1-2'
    target_round = rounds[0]

    blocked_data = deepcopy(base)
    # Pairwise block scoped to a SINGLE round: removes only this matchup in that
    # round (every other round/opponent untouched -> no team loses all dates).
    blocked_data['blocked_games'] = [
        {'team1': t1, 'team2': t2, 'round_no': target_round,
         'description': 'spec-034 pairwise block test'},
    ]
    model1 = cp_model.CpModel()
    X1, _ = generate_X(model1, blocked_data)
    in_round = [k for k in X1 if len(k) >= 11 and {k[0], k[1]} == {t1, t2} and k[8] == target_round]
    other_rounds = [k for k in X1 if len(k) >= 11 and {k[0], k[1]} == {t1, t2} and k[8] != target_round]
    # Oracle: zero vars for the matchup in the blocked round; it survives elsewhere.
    assert len(in_round) == 0, f'block left {len(in_round)} vars in round {target_round}'
    assert len(other_rounds) > 0, 'block wrongly removed the matchup from other rounds'
