"""GWT tests for ClubNoConcurrentSlot (spec-033 Unit E), hand oracles.

No mocks: real cp_model, real registry, hand-built X (11-tuples) + real Team/Club
models.

spec-033 Unit E changed the rule from a capacity-aware `ceil(n_loc/S)` cap to a
HARD ceiling of `1 + slack` overlaps per (club, week, day, location, slot), with
a SOFT penalty per game beyond the first pushing overlaps -> 0. The aggregation
across a location's fields covers cross-field (EF/WF/SF) overlap at NIHC.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import ClubNoConcurrentSlot
from constraints.helper_vars import HelperVarRegistry
from models import Club, Team

LOC = 'Central Coast Hockey Park'
DATE = '2026-03-22'


def _key(t1, t2, grade, slot):
    return (t1, t2, grade, 'Sunday', slot, '12:00', 1, DATE, 1, 'Wyong Main Field', LOC)


def _data(home_teams, slack=0, weight=200_000):
    """Club 'G' fields `home_teams` (a list of grades). Each opponent is its own
    1-team club so only G's cap binds."""
    g = Club(name='G', home_field=LOC)
    teams = [Team(name=f'G {gr}', club=g, grade=gr) for gr in home_teams]
    for i, gr in enumerate(home_teams):
        oc = Club(name=f'O{i}', home_field=LOC)
        teams.append(Team(name=f'O{i} {gr}', club=oc, grade=gr))
    return {
        'teams': teams,
        'locked_weeks': set(),
        'constraint_slack': {'ClubNoConcurrentSlot': slack},
        'penalty_weights': {'ClubNoConcurrentSlot': weight},
        'penalties': {},
    }


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    return solver.Solve(model)


def _apply(model, X, data):
    return ClubNoConcurrentSlot().apply(model, X, data, HelperVarRegistry(model))


def _pin(model, X, assignment):
    """Fully pin every var: each (grade -> chosen slot) plays in that slot only."""
    for key, v in X.items():
        grade = key[2]
        chosen = assignment.get(grade)
        model.Add(v == (1 if key[4] == chosen else 0))


class TestClubNoConcurrentSlot:
    def test_two_games_one_slot_slack0_infeasible(self):
        # Hand oracle (DoD 35): 2 club games forced into ONE (location, slot) at
        # slack 0 -> hard cap sum(slot_vars) <= 1+0 = 1. 2 > 1 -> INFEASIBLE.
        data = _data(['PHL', '2nd'], slack=0)
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        _apply(model, X, data)
        _pin(model, X, {'PHL': 1, '2nd': 1})  # both in slot 1
        assert _solve(model) == cp_model.INFEASIBLE

    def test_two_games_one_slot_slack1_feasible_one_penalty(self):
        # Hand oracle (DoD 35): at slack 1 the cap is <= 2, so both games fit.
        # The atom emits one `over` IntVar PER slot that has >=2 candidate club
        # vars. Here each grade has a candidate var in BOTH slots, so both slot 1
        # and slot 2 get a cap + an `over` var (2 vars). But both games are pinned
        # into slot 1, so the TOTAL penalty value is over(slot1)+over(slot2) =
        # (2-1) + (0) = 1. The minimised objective is the meaningful oracle: 1 unit.
        data = _data(['PHL', '2nd'], slack=1)
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        _apply(model, X, data)
        _pin(model, X, {'PHL': 1, '2nd': 1})  # both in slot 1

        bucket = data['penalties']['ClubNoConcurrentSlot']
        assert bucket['weight'] == 200_000
        # One `over` var per slot with >=2 candidate club vars (slot1 + slot2).
        assert len(bucket['penalties']) == 2
        total = sum(bucket['penalties'])

        # Minimise the total penalty so the solver reports the true overage.
        model.Minimize(total)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        # over(slot1)=2-1=1, over(slot2)=0 -> total = 1 penalty unit.
        assert sum(solver.Value(p) for p in bucket['penalties']) == 1

    def test_three_games_two_slots_slack0_infeasible(self):
        # Hand oracle (DoD 35): 3 club games, 2 distinct slots, slack 0 -> cap 1
        # per slot. Pigeonhole: 3 games into 2 slots at <=1 each is impossible
        # -> INFEASIBLE. (We pin all 3 to the 2 available slots; the cap binds.)
        grades = ['PHL', '2nd', '3rd']
        data = _data(grades, slack=0)
        model = cp_model.CpModel()
        # Each grade has candidate vars in both slots; force each to play exactly
        # once (one of its two slot vars). The hard cap then forbids any slot
        # holding 2 -> with 3 games and 2 slots one slot must hold 2 -> infeasible.
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(grades) for s in (1, 2)}
        _apply(model, X, data)
        for i, gr in enumerate(grades):
            model.Add(sum(X[_key(f'G {gr}', f'O{i} {gr}', gr, s)] for s in (1, 2)) == 1)
        assert _solve(model) == cp_model.INFEASIBLE

    def test_three_games_two_slots_slack1_feasible_one_penalty(self):
        # Hand oracle (DoD 35): at slack 1 the cap is 2 per slot. One slot holds 2
        # club games, the other holds 1 -> feasible. The over for the 2-game slot
        # = 2 - 1 = 1; the 1-game slot is skipped (len(slot_vars) < 2). Total
        # penalty = 1 unit.
        grades = ['PHL', '2nd', '3rd']
        data = _data(grades, slack=1)
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(grades) for s in (1, 2)}
        _apply(model, X, data)
        # Pin: PHL + 2nd in slot 1 (2 games), 3rd in slot 2 (1 game).
        _pin(model, X, {'PHL': 1, '2nd': 1, '3rd': 2})

        bucket = data['penalties']['ClubNoConcurrentSlot']
        # Both slots have >=2 candidate club vars (each grade has a var in each
        # slot) -> 2 `over` vars. With PHL+2nd pinned to slot 1 and 3rd to slot 2,
        # over(slot1)=2-1=1 and over(slot2)=1-... but slot2 holds only 1 game so
        # over(slot2)=0. Total penalty = 1 unit.
        assert len(bucket['penalties']) == 2
        total = sum(bucket['penalties'])
        model.Minimize(total)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(solver.Value(p) for p in bucket['penalties']) == 1

    def test_staggered_zero_penalty_value(self):
        # Hand oracle: caps are built per slot with >=2 CANDIDATE club vars. Each
        # grade here has a candidate var in both slots, so both slots get a cap +
        # `over` var (2 caps). But when the games are staggered (PHL slot1, 2nd
        # slot2 — each slot holds exactly 1 game), every `over` value is 0:
        # over(slot1)=1-1=0, over(slot2)=1-1=0. Total penalty = 0 — staggering is
        # free, overlap is penalised.
        data = _data(['PHL', '2nd'], slack=0)
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        n = _apply(model, X, data)
        assert n == 2  # one cap per slot with >=2 candidate club vars
        _pin(model, X, {'PHL': 1, '2nd': 2})  # staggered: one game per slot
        bucket = data['penalties']['ClubNoConcurrentSlot']
        total = sum(bucket['penalties'])
        model.Minimize(total)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(solver.Value(p) for p in bucket['penalties']) == 0

    def test_single_candidate_per_slot_no_constraint(self):
        # Hand oracle: if a slot has only ONE candidate club var it is skipped
        # (len(slot_vars) < 2) -> no cap, no penalty. Build PHL with a candidate
        # only in slot 1 and 2nd only in slot 2 (distinct slots, 1 var each).
        data = _data(['PHL', '2nd'], slack=0)
        model = cp_model.CpModel()
        X = {
            _key('G PHL', 'O0 PHL', 'PHL', 1): model.NewBoolVar('phl_s1'),
            _key('G 2nd', 'O1 2nd', '2nd', 2): model.NewBoolVar('2nd_s2'),
        }
        n = _apply(model, X, data)
        assert n == 0
        assert data['penalties'].get('ClubNoConcurrentSlot', {}).get('penalties', []) == []

    def test_locked_week_skipped(self):
        data = _data(['PHL', '2nd'], slack=0)
        data['locked_weeks'] = {1}
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        assert _apply(model, X, data) == 0

    def test_weight_zero_disables_soft_but_keeps_hard(self):
        # weight 0 -> no soft bucket, but the hard cap still binds.
        data = _data(['PHL', '2nd'], slack=0, weight=0)
        model = cp_model.CpModel()
        X = {_key(f'G {gr}', f'O{i} {gr}', gr, s): model.NewBoolVar(f'g{i}s{s}')
             for i, gr in enumerate(['PHL', '2nd']) for s in (1, 2)}
        _apply(model, X, data)
        _pin(model, X, {'PHL': 1, '2nd': 1})  # both in slot 1, cap 1
        assert _solve(model) == cp_model.INFEASIBLE
        assert 'ClubNoConcurrentSlot' not in data['penalties']


class TestComputeNoFieldSlots:
    """compute_no_field_slots still exists (no longer gates the cap, but the
    helper and data['no_field_slots'] remain for other consumers)."""

    def test_derives_distinct_time_count_per_location(self):
        from config.defaults import compute_no_field_slots
        dtm = {
            'NIHC': {'Sunday': ['8:30', '10:00', '11:30'], 'Friday': ['19:00']},
            'Maitland': {'Sunday': ['9:00', '10:30']},
            'Empty': {},
        }
        out = compute_no_field_slots(dtm)
        assert out == {'NIHC': 3, 'Maitland': 2, 'Empty': 0}

    def test_real_2026_config(self):
        from config import load_season_data
        nfs = load_season_data(2026)['no_field_slots']
        assert nfs['Newcastle International Hockey Centre'] == 8
        assert nfs['Maitland Park'] == 6
        assert nfs['Central Coast Hockey Park'] == 2
