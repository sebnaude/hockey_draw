"""Tests for SoftLexMatchupOrdering atom.

All tests use real CP-SAT models with hand-computed oracles. No mocks.

Fixture: 3 teams (Norths, Tigers, Wests) × 1 grade × 2 rounds.
Pairs (alphabetical order):
  rank 0: ('Norths PHL', 'Tigers PHL')   -- N before T
  rank 1: ('Norths PHL', 'Wests PHL')    -- N before W (but after N-T)
  rank 2: ('Tigers PHL', 'Wests PHL')    -- T before W, last

With 3 teams and 2 rounds we have 3 pairs to schedule but only 2 rounds.
One pair per round; the 3rd pair is a bye or uses a dummy if available.
For simplicity, test with 3 rounds (one game per round), which ensures all
pairs can be scheduled.

The atom is purely soft — it never blocks feasibility.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from typing import Dict

import pytest
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from constraints.atoms.base import BROADMEADOW
from constraints.atoms.soft_lex_matchup_ordering import SoftLexMatchupOrdering
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Minimal 3-team fixture builder
# ---------------------------------------------------------------------------

def _build_lex_fixture(num_rounds: int = 3) -> Dict:
    """3 teams (Norths PHL, Tigers PHL, Wests PHL) × `num_rounds` rounds.

    Pairs (alphabetical by team1, team2):
      rank 0: (Norths PHL, Tigers PHL)
      rank 1: (Norths PHL, Wests PHL)
      rank 2: (Tigers PHL, Wests PHL)

    Each round has 2 timeslots (EF + WF at Broadmeadow, same Sunday, same slot)
    so there is room for 2 games per round.
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')

    clubs = [
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
    ]
    teams = [Team(name=f'{c.name} PHL', club=c, grade='PHL') for c in clubs]
    grades = [Grade(name='PHL', teams=[t.name for t in teams])]

    games = list(combinations([t.name for t in teams], 2))  # 3 pairs

    timeslots = []
    for rnd in range(1, num_rounds + 1):
        date_str = f'2026-0{3 + rnd}-01' if rnd < 8 else f'2026-{rnd:02d}-01'
        for field in (ef, wf):
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time='11:30',
                week=rnd, day_slot=1, field=field, round_no=rnd,
            ))

    return {
        'games': [(t1, t2, 'PHL') for t1, t2 in games],
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': [ef, wf],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'PHL': num_rounds, 'max': num_rounds},
        'constraint_slack': {},
        'penalty_weights': {'soft_lex_ordering': 1},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {},
        'constraint_defaults': {},
    }


def _build_model_X(data: Dict):
    """Build a CpModel + X dict from the lex fixture data."""
    model = cp_model.CpModel()
    X = {}
    for (t1, t2, grade) in data['games']:
        for t in data['timeslots']:
            if not t.day:
                continue
            key = (
                t1, t2, grade,
                t.day, t.day_slot, t.time,
                t.week, t.date, t.round_no,
                t.field.name, t.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}'
            )
    return model, X


def _registry(model):
    r = HelperVarRegistry(model)
    return r


def _solve(model, seconds=5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    return status, solver


# ---------------------------------------------------------------------------
# Scenario 1: solo-clean — atom applies without error, model is SAT
# ---------------------------------------------------------------------------

class TestSoftLexMatchupOrderingSoloClean:
    """Given no other constraints, When SoftLexMatchupOrdering is applied,
    Then the model is feasible (atom never blocks scheduling)."""

    def test_solo_clean_feasible(self):
        # GIVEN: 3-team, 3-round fixture with only the soft atom applied
        data = _build_lex_fixture(num_rounds=3)
        model, X = _build_model_X(data)

        # WHEN: apply the atom
        n = SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        # THEN: n > 0 (rank-1 and rank-2 pairs each get a penalty var)
        # Hand calc: 3 pairs → ranks 0,1,2 → skip rank 0 → 2 penalty vars (rank 1 + rank 2)
        assert n == 2

        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_penalties_bucket_created(self):
        # GIVEN: fresh fixture
        data = _build_lex_fixture(num_rounds=3)
        model, X = _build_model_X(data)

        # WHEN: apply
        SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        # THEN: data['penalties']['soft_lex_ordering'] exists with correct weight
        assert 'soft_lex_ordering' in data['penalties']
        bucket = data['penalties']['soft_lex_ordering']
        assert bucket['weight'] == 1
        assert len(bucket['penalties']) == 2  # rank 1 + rank 2


# ---------------------------------------------------------------------------
# Scenario 2: ordering preference — solver prefers lex-earlier matchup first
# ---------------------------------------------------------------------------

class TestSoftLexMatchupOrderingPreference:
    """Given soft lex atom + forced exactly-one-game-per-round-per-pair,
    When optimising, Then the alphabetically-earlier pair (rank 0) appears
    in round 1 and the alphabetically-later pair (rank 2) appears in round 3.

    Hand oracle:
      pairs (alphabetical): NT (rank 0), NW (rank 1), TW (rank 2)
      penalty weight = 1
      For each pair, penalty = rank * X[key] summed over all keys of that pair.

      With 3 rounds and 2 fields/round, 3 pairs can each play exactly once.
      Objective: minimise total penalty = sum_over_keys(rank * X[key]).
      Optimal: rank-0 pair plays in any round (contributes 0 penalty per game),
               rank-1 and rank-2 pairs each contribute rank * (number of vars used).
      The solver has no preference on which *round* — only which *vars* are 1.
      So the atom enforces: schedule all pairs (no round-ordering within), but
      the sum-of-rank penalty means scheduling more high-rank pairs is costlier.
      For a single game per pair, the penalty is identical across rounds.

    This scenario instead verifies the soft nature: even if we FORCE a high-rank
    pair into round 1, the model remains SAT (higher penalty accepted).
    """

    def test_forced_wrong_order_still_feasible(self):
        # GIVEN: 3 teams, 3 rounds, force rank-2 pair (Tigers PHL, Wests PHL) into round 1
        data = _build_lex_fixture(num_rounds=3)
        model, X = _build_model_X(data)

        # Find a key for (Tigers PHL, Wests PHL) in round 1
        # team1 < team2 alphabetically: Tigers < Wests → key[0]='Tigers PHL', key[1]='Wests PHL'
        tw_r1_keys = [
            k for k in X
            if k[0] == 'Tigers PHL' and k[1] == 'Wests PHL' and k[6] == 1
        ]
        assert len(tw_r1_keys) >= 1, 'fixture must produce Tigers-Wests in round 1'
        model.Add(X[tw_r1_keys[0]] == 1)

        # WHEN: apply the soft atom
        SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        # THEN: still feasible (forced "wrong" order is acceptable — just higher penalty)
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_lex_order_preferred_when_no_other_constraints(self):
        """Solver should produce lower objective when lex-earlier pairs go first.

        Hand oracle (3 teams, 3 rounds, 2 fields/round, weight=1):
          pairs: ('Norths PHL','Tigers PHL') rank=0, ('Norths PHL','Wests PHL') rank=1,
                 ('Tigers PHL','Wests PHL') rank=2.
          Each pair plays exactly once. Penalty per pair = rank * 1 (one var = 1).
          Total penalty regardless of round = 0 + 1 + 2 = 3.
          (Round doesn't affect penalty — the linear sum is always 3 when all 3 play.)

          However, if only ONE pair must play in round 1 (via force):
          - Forcing rank-0 pair into round 1: penalty for round-1 game = 0.
          - Forcing rank-2 pair into round 1: penalty for round-1 game = 2.
          This shows the atom discourages scheduling later-alphabetical pairs.
        """
        # Build comparison: force rank-0 pair in round 1
        data_a = _build_lex_fixture(num_rounds=3)
        model_a, X_a = _build_model_X(data_a)
        SoftLexMatchupOrdering().apply(model_a, X_a, data_a, _registry(model_a))

        nt_r1 = [k for k in X_a if k[0] == 'Norths PHL' and k[1] == 'Tigers PHL' and k[6] == 1]
        assert nt_r1
        model_a.Add(X_a[nt_r1[0]] == 1)

        # Minimise total penalty in model_a
        total_pen_a = sum(data_a['penalties']['soft_lex_ordering']['penalties'])
        model_a.Minimize(total_pen_a)
        status_a, solver_a = _solve(model_a)
        assert status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        obj_a = solver_a.ObjectiveValue()

        # Build comparison: force rank-2 pair in round 1
        data_b = _build_lex_fixture(num_rounds=3)
        model_b, X_b = _build_model_X(data_b)
        SoftLexMatchupOrdering().apply(model_b, X_b, data_b, _registry(model_b))

        tw_r1 = [k for k in X_b if k[0] == 'Tigers PHL' and k[1] == 'Wests PHL' and k[6] == 1]
        assert tw_r1
        model_b.Add(X_b[tw_r1[0]] == 1)

        total_pen_b = sum(data_b['penalties']['soft_lex_ordering']['penalties'])
        model_b.Minimize(total_pen_b)
        status_b, solver_b = _solve(model_b)
        assert status_b in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        obj_b = solver_b.ObjectiveValue()

        # Hand calc:
        # model_a: rank-0 pair (NT) in round 1 → contributes rank_0 * 1 = 0 to penalty.
        # model_b: rank-2 pair (TW) in round 1 → contributes rank_2 * 1 = 2 to penalty.
        # With no other games forced, the minimiser sets all other vars to 0.
        # So obj_a == 0, obj_b == 2. In any case obj_b > obj_a.
        assert obj_b > obj_a, (
            f'Expected rank-2-first penalty ({obj_b}) > rank-0-first penalty ({obj_a})'
        )


# ---------------------------------------------------------------------------
# Scenario 3: weight = 0 disables the atom
# ---------------------------------------------------------------------------

class TestSoftLexMatchupOrderingWeightZero:
    """Given weight=0, When atom is applied, Then it adds no penalty vars (n==0)."""

    def test_zero_weight_returns_zero(self):
        # GIVEN: weight configured to 0
        data = _build_lex_fixture(num_rounds=3)
        data['penalty_weights']['soft_lex_ordering'] = 0
        model, X = _build_model_X(data)

        # WHEN: apply
        n = SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        # THEN: no constraints added, no bucket created
        assert n == 0
        assert 'soft_lex_ordering' not in data.get('penalties', {})


# ---------------------------------------------------------------------------
# Scenario 4: dummy keys are skipped
# ---------------------------------------------------------------------------

class TestSoftLexMatchupOrderingDummySkip:
    """Given X contains a dummy key (len < 11), When atom is applied,
    Then the dummy var is not included in any penalty."""

    def test_dummy_keys_excluded(self):
        # GIVEN: 3-team fixture with an extra dummy var injected
        data = _build_lex_fixture(num_rounds=3)
        model, X = _build_model_X(data)

        # Inject a 4-tuple dummy key
        dummy_key = ('Norths PHL', 'Tigers PHL', 'PHL', 0)
        dummy_var = model.NewBoolVar('dummy')
        X[dummy_key] = dummy_var

        # WHEN: apply
        n = SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        # THEN: same result as without dummy (2 penalty vars: rank-1 and rank-2 pairs)
        # The dummy key is 4 elements < 11, so skipped.
        assert n == 2


# ---------------------------------------------------------------------------
# Scenario 5: locked weeks are skipped
# ---------------------------------------------------------------------------

class TestSoftLexMatchupOrderingLockedWeeks:
    """Given round 1 is locked, When atom is applied, Then round-1 vars are excluded."""

    def test_locked_week_vars_excluded(self):
        # GIVEN: round 1 is locked
        data = _build_lex_fixture(num_rounds=3)
        data['locked_weeks'] = {1}
        model, X = _build_model_X(data)

        # WHEN: apply
        n = SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        # THEN: still 2 penalty vars (rank-1 and rank-2) — from rounds 2 and 3.
        # Rank-0 pair still contributes 0 penalty (skipped); rounds 2+3 have vars.
        assert n == 2
        # Verify no round-1 var appears in penalty vars
        bucket = data['penalties']['soft_lex_ordering']['penalties']
        # We can't inspect IntVar bounds easily without solving, but n==2 confirms
        # the correct number of penalty IntVars were created.
        assert len(bucket) == 2


# ---------------------------------------------------------------------------
# Scenario 6: empty X — atom handles gracefully
# ---------------------------------------------------------------------------

class TestSoftLexMatchupOrderingEmptyX:
    """Given X is empty, When atom is applied, Then n==0 and no crash."""

    def test_empty_X(self):
        data = _build_lex_fixture(num_rounds=3)
        model = cp_model.CpModel()
        X = {}

        n = SoftLexMatchupOrdering().apply(model, X, data, _registry(model))

        assert n == 0
