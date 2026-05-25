"""spec-033 Unit A — verification that the KEPT soft constraints still work.

Unit A removes the dead `ClubVsClubAlignment` slack. It must NOT disturb the two
deviation-tolerant constraints the audit table marks "verify only":

  * `EqualMatchUpSpacing` — hard floor honours `spacing_base_slack` (=2 for 2026)
    AND a sliding-window soft term emits `'EqualMatchUpSpacing'` penalties.
  * `ClubGameSpread` (contiguity) — hard cap permits <=1 hole per field for fields
    with >=4 games, <=0 holes for <=3 games, and the soft term emits per-hole
    `'ClubGameSpread'` penalties.

NO mocks / patches / monkeypatch. We use the real `load_season_data(2026)` config
and the real `UnifiedConstraintEngine`. Every expected value is hand-computed in a
comment so the oracle is independent of the implementation.

Spacing hand-oracle (from `constraints/atoms/_spacing.py`):
  _legacy_min_gap(10) = 6 ; ideal_gap(10) = 6 - 1 = 5 ;
  effective_spacing(10, base_slack=2, config_slack=0) = max(0, 5 - 2 - 0) = 3.
  => with spacing_base_slack=2 the hard rule forbids a repeat meeting whose
     gap = r2 - r1 is <= 3 (i.e. gaps 1,2,3 forbidden; gap 4 allowed).
  If base_slack were 0 (the old default) S would be 5 and gap 4 would ALSO be
  forbidden — so a feasible gap-4 / infeasible gap-3 pair pins base_slack==2.

Contiguity hand-oracle (from `tests/test_club_game_spread_contiguity.py`):
  holes = (max_used - min_used + 1) - num_distinct_used_slots
  gap_cap = max(0, min(1, n_games - 3))   (<=3 games -> 0 holes; >=4 -> 1 hole)
"""
from __future__ import annotations

from typing import List, Tuple

from ortools.sat.python import cp_model

from config import load_season_data
from constraints.atoms._spacing import (effective_spacing, ideal_gap,
                                         _legacy_min_gap)
from constraints.atoms.base import BROADMEADOW
from constraints.unified import UnifiedConstraintEngine
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars


# --------------------------------------------------------------------------
# (a) EqualMatchUpSpacing — base slack + sliding-window soft penalties.
# --------------------------------------------------------------------------

def test_2026_spacing_base_slack_is_two_and_drives_S():
    """spacing_base_slack is 2 in the real 2026 config and feeds effective_spacing.

    Hand oracle: _legacy_min_gap(10)=6 ; ideal_gap(10)=5 ;
    effective_spacing(10, base_slack=2) = 5 - 2 = 3.
    """
    data = load_season_data(2026)
    base_slack = int(data['constraint_defaults']['spacing_base_slack'])
    assert base_slack == 2, f"2026 spacing_base_slack must be 2, got {base_slack}"

    assert _legacy_min_gap(10) == 6
    assert ideal_gap(10) == 5
    assert effective_spacing(10, base_slack=base_slack, config_slack=0) == 3


def _spacing_fixture(num_teams: int, num_rounds: int):
    """A single grade of `num_teams` teams over `num_rounds` Sunday rounds, one
    NIHC field with one slot per round. A single matchup (the first pair) gets a
    var in every round so the engine's spacing rule can constrain it. EqualGames
    is NOT applied here, so we are free to pin the pair into specific rounds."""
    field = PlayingField(location=BROADMEADOW, name='EF')
    club = Club(name='C', home_field=BROADMEADOW)
    teams = [Team(name=f'T{i}', club=club, grade='3rd') for i in range(num_teams)]
    grade_objs = [Grade(name='3rd', teams=[t.name for t in teams])]
    # Only the (T0, T1) matchup is needed for the spacing oracle.
    games: List[Tuple[str, str, str]] = [('T0', 'T1', '3rd')]
    timeslots = [
        Timeslot(date=f'2026-03-{20 + r}', day='Sunday', time='10:00',
                 week=r, day_slot=1, field=field, round_no=r)
        for r in range(1, num_rounds + 1)
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': grade_objs, 'clubs': [club], 'fields': [field],
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {'3rd': num_rounds, 'max': num_rounds},
        'constraint_slack': {}, 'penalty_weights': {}, 'penalties': {},
        'forced_games': [], 'blocked_games': [], 'team_conflicts': [],
        'phl_preferences': {}, 'club_days': {}, 'preference_no_play': {},
        'home_field_map': {},
        # The real 2026 spacing_base_slack so the engine reads the live value.
        'constraint_defaults': {
            'spacing_base_slack': int(
                load_season_data(2026)['constraint_defaults']['spacing_base_slack']
            )
        },
    }
    return model, X, data


def _engine(model, X, data):
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    return eng


def _pin_round(model, X, t1, t2, round_no):
    """Force matchup (t1,t2) to play in round `round_no`."""
    for k, v in X.items():
        if {k[0], k[1]} == {t1, t2}:
            model.Add(v == (1 if k[8] == round_no else 0))


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    return solver.Solve(model)


def test_spacing_hard_forbids_gap_three_with_base_slack_two():
    """T=10 -> S=3 (base_slack=2). The pair pinned to rounds 1 and 4 has gap 3
    which is <= S => the hard rule forbids both meetings simultaneously =>
    INFEASIBLE once we force BOTH games on."""
    model, X, data = _spacing_fixture(num_teams=10, num_rounds=8)
    n = _engine(model, X, data)._matchup_spacing_hard()
    assert n > 0, "no hard spacing constraints emitted"
    # Force the SAME pair to play in BOTH round 1 and round 4 (gap 3).
    for k, v in X.items():
        if {k[0], k[1]} == {'T0', 'T1'} and k[8] in (1, 4):
            model.Add(v == 1)
        elif {k[0], k[1]} == {'T0', 'T1'}:
            model.Add(v == 0)
    assert _solve(model) == cp_model.INFEASIBLE


def test_spacing_hard_permits_gap_four_with_base_slack_two():
    """Same fixture. Rounds 1 and 5 => gap 4 > S(=3) => allowed => FEASIBLE.

    This is the discriminating case: with the OLD base_slack=0 the floor would
    be S=5 and gap 4 would ALSO be forbidden. Feasibility here proves the engine
    read base_slack=2."""
    model, X, data = _spacing_fixture(num_teams=10, num_rounds=8)
    _engine(model, X, data)._matchup_spacing_hard()
    for k, v in X.items():
        if {k[0], k[1]} == {'T0', 'T1'} and k[8] in (1, 5):
            model.Add(v == 1)
        elif {k[0], k[1]} == {'T0', 'T1'}:
            model.Add(v == 0)
    assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_spacing_soft_emits_sliding_window_penalties():
    """The soft term registers `'EqualMatchUpSpacing'` penalties. The window size
    is `space = T - 1 = 9`; the soft method skips a grade when `space >= R`, so
    we need R > 9 -> use R=12. A window of 9 rounds covers rounds [1..9]; the pair
    pinned into rounds 1 and 5 (gap 4, HARD-allowed) puts 2 vars in that window =>
    pen >= 2 - 1 = 1. Assert the bucket sums to at least one penalty unit."""
    model, X, data = _spacing_fixture(num_teams=10, num_rounds=12)
    eng = _engine(model, X, data)
    eng._matchup_spacing_hard()
    n_soft = eng._matchup_spacing_soft()
    assert n_soft > 0, "no soft spacing windows emitted"
    assert 'EqualMatchUpSpacing' in data['penalties']
    for k, v in X.items():
        if {k[0], k[1]} == {'T0', 'T1'} and k[8] in (1, 5):
            model.Add(v == 1)
        elif {k[0], k[1]} == {'T0', 'T1'}:
            model.Add(v == 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    penalties = data['penalties']['EqualMatchUpSpacing']['penalties']
    total = sum(solver.Value(p) for p in penalties)
    # Two meetings both inside the single size-9 window => at least one penalty.
    assert total >= 1, f"expected >=1 sliding-window penalty unit, got {total}"


# --------------------------------------------------------------------------
# (b) ClubGameSpread contiguity — hole cap + per-hole soft penalties.
# --------------------------------------------------------------------------

GRADES = ['PHL', '2nd', '3rd', '4th']
SLOTS = list(range(1, 7))


def _spread_fixture(n_club_teams: int):
    """Club 'C' fields `n_club_teams` teams (distinct grades); each plays one
    opponent from its own 1-team club, on ONE NIHC field with 6 slots, week 1."""
    field = PlayingField(location=BROADMEADOW, name='EF')
    grades = GRADES[:n_club_teams]
    c = Club(name='C', home_field=BROADMEADOW)
    teams = [Team(name=f'C {g}', club=c, grade=g) for g in grades]
    opp_clubs = [Club(name=f'O{i}', home_field=BROADMEADOW) for i in range(n_club_teams)]
    teams += [Team(name=f'O{i} {g}', club=opp_clubs[i], grade=g)
              for i, g in enumerate(grades)]
    grade_objs = [Grade(name=g, teams=[f'C {g}', f'O{i} {g}'])
                  for i, g in enumerate(grades)]
    games: List[Tuple[str, str, str]] = [
        (f'C {g}', f'O{i} {g}', g) for i, g in enumerate(grades)
    ]
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time=f'{8 + s}:00',
                 week=1, day_slot=s, field=field, round_no=1)
        for s in SLOTS
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': grade_objs, 'clubs': [c] + opp_clubs, 'fields': [field],
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {g: 1 for g in grades}, 'constraint_slack': {},
        'penalty_weights': {}, 'penalties': {}, 'forced_games': [],
        'blocked_games': [], 'team_conflicts': [], 'phl_preferences': {},
        'club_days': {}, 'preference_no_play': {}, 'home_field_map': {},
        'constraint_defaults': {},
    }
    return model, X, data


def _pin_slot(model, X, t1, t2, slot):
    for k, v in X.items():
        if {k[0], k[1]} == {t1, t2}:
            model.Add(v == (1 if k[4] == slot else 0))


def test_spread_three_games_zero_holes_required():
    """n=3 -> gap_cap 0. Slots {1,2,4} -> hole at 3 -> INFEASIBLE."""
    model, X, data = _spread_fixture(3)
    _engine(model, X, data)._club_game_spread_hard()
    _pin_slot(model, X, 'C PHL', 'O0 PHL', 1)
    _pin_slot(model, X, 'C 2nd', 'O1 2nd', 2)
    _pin_slot(model, X, 'C 3rd', 'O2 3rd', 4)
    assert _solve(model) == cp_model.INFEASIBLE


def test_spread_four_games_one_hole_allowed():
    """n=4 -> gap_cap 1. Slots {1,2,4,5} -> exactly 1 hole (slot 3) -> FEASIBLE."""
    model, X, data = _spread_fixture(4)
    _engine(model, X, data)._club_game_spread_hard()
    _pin_slot(model, X, 'C PHL', 'O0 PHL', 1)
    _pin_slot(model, X, 'C 2nd', 'O1 2nd', 2)
    _pin_slot(model, X, 'C 3rd', 'O2 3rd', 4)
    _pin_slot(model, X, 'C 4th', 'O3 4th', 5)
    assert _solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_spread_soft_emits_per_hole_penalty():
    """n=4, slots {1,2,4,5}: exactly 1 interior hole at slot 3. The soft term
    registers a per-hole penalty in the `'ClubGameSpread'` bucket; off-primary
    is 0 (single field) so the bucket sums to exactly 1 (the single hole)."""
    model, X, data = _spread_fixture(4)
    eng = _engine(model, X, data)
    eng._club_game_spread_hard()
    eng._club_game_spread_soft()
    assert 'ClubGameSpread' in data['penalties']
    _pin_slot(model, X, 'C PHL', 'O0 PHL', 1)
    _pin_slot(model, X, 'C 2nd', 'O1 2nd', 2)
    _pin_slot(model, X, 'C 3rd', 'O2 3rd', 4)
    _pin_slot(model, X, 'C 4th', 'O3 4th', 5)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    penalties = data['penalties']['ClubGameSpread']['penalties']
    total = sum(solver.Value(p) for p in penalties)
    # 1 interior hole, 0 off-primary (single field) -> exactly 1.
    assert total == 1, f"expected exactly 1 hole penalty unit, got {total}"


def test_spread_contiguous_zero_penalty():
    """n=4, slots {1,2,3,4}: contiguous -> 0 holes, single field -> 0 off-primary
    -> ClubGameSpread penalty bucket sums to 0."""
    model, X, data = _spread_fixture(4)
    eng = _engine(model, X, data)
    eng._club_game_spread_hard()
    eng._club_game_spread_soft()
    _pin_slot(model, X, 'C PHL', 'O0 PHL', 1)
    _pin_slot(model, X, 'C 2nd', 'O1 2nd', 2)
    _pin_slot(model, X, 'C 3rd', 'O2 3rd', 3)
    _pin_slot(model, X, 'C 4th', 'O3 4th', 4)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    penalties = data['penalties']['ClubGameSpread']['penalties']
    total = sum(solver.Value(p) for p in penalties)
    assert total == 0, f"expected 0 penalty units, got {total}"
