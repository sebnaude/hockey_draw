# spec-017: matchup spacing promoted to HARD (critical_feasibility).
"""spec-017 — `EqualMatchUpSpacing` is now a HARD production constraint.

Before spec-017 the atom sat ONLY in the `soft_optimisation` stage, which is
`soft_only=True`; `apply_solver_stage` skips `apply_stage_1_hard()` for a
soft_only stage, so the hard pairwise-forbidden-gap clause never applied in
production. spec-017 moves the atom to `critical_feasibility` (a non-soft_only
stage), so BOTH its hard and soft parts now apply.

Real CP-SAT models, no mocks. The grade size is T=10 so the spacing threshold
is hand-computable: `ideal_gap(10) == 5`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from config.defaults import DEFAULT_STAGES
from constraints.atoms._spacing import effective_spacing, ideal_gap
from constraints.registry import CONSTRAINT_REGISTRY
from constraints.stages import (
    ALL_ENGINE_KEYS,
    apply_solver_stage,
    validate_solver_stages,
)
from constraints.unified import UnifiedConstraintEngine
from constraints.atoms.base import BROADMEADOW
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars


T = 10                       # teams in the test grade
# Rounds must exceed the soft sliding-window size (T-1 = 9) for the soft
# density penalty to fire, and be wide enough to exercise gaps up to S+1 = 6.
NUM_ROUNDS = 11
S0 = ideal_gap(T)            # = 5 at slack 0 (hand oracle)


def _fixture(num_rounds: int = NUM_ROUNDS, slack: int = 0) -> Tuple[cp_model.CpModel, Dict, Dict]:
    """One grade of T=10 teams; one NIHC slot per week across `num_rounds`
    weeks. Every pair therefore has exactly one candidate var per week."""
    field = PlayingField(location=BROADMEADOW, name='EF')
    clubs = [Club(name=f'C{i}', home_field=BROADMEADOW) for i in range(T)]
    teams = [Team(name=f'T{i}', club=clubs[i], grade='Test') for i in range(T)]
    grade = Grade(name='Test', teams=[t.name for t in teams])
    games: List[Tuple[str, str, str]] = [
        (a, b, 'Test') for a, b in combinations([t.name for t in teams], 2)
    ]
    timeslots = [
        Timeslot(date=f'2026-03-{20 + r:02d}', day='Sunday', time='11:30',
                 week=r, day_slot=1, field=field, round_no=r)
        for r in range(1, num_rounds + 1)
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': [grade], 'clubs': clubs, 'fields': [field],
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {'Test': num_rounds, 'max': num_rounds},
        'constraint_slack': {'EqualMatchUpSpacingConstraint': slack} if slack else {},
        'penalty_weights': {}, 'penalties': {},
        'forced_games': [], 'blocked_games': [], 'team_conflicts': [],
        'phl_preferences': {}, 'club_days': {}, 'preference_no_play': {},
        'home_field_map': {}, 'constraint_defaults': {},
    }
    return model, X, data


def _engine(model, X, data) -> UnifiedConstraintEngine:
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    return eng


def _pair_vars(X, a: str, b: str, round_no: int):
    return [v for k, v in X.items()
            if {k[0], k[1]} == {a, b} and k[2] == 'Test' and k[8] == round_no]


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    return solver.Solve(model), solver


# ----------------------------------------------------------------------
# DoD 1 + 5: stage placement and the deliberate "keep byes separate" decision
# ----------------------------------------------------------------------


class TestStagePlacement:
    def test_in_critical_feasibility_after_byes_only(self):
        cf = next(s for s in DEFAULT_STAGES if s['name'] == 'critical_feasibility')
        so = next(s for s in DEFAULT_STAGES if s['name'] == 'soft_optimisation')
        assert 'EqualMatchUpSpacing' in cf['atoms']
        assert 'EqualMatchUpSpacing' not in so['atoms']  # moved, not duplicated
        # immediately after BalancedByeSpacing
        assert (cf['atoms'].index('EqualMatchUpSpacing')
                == cf['atoms'].index('BalancedByeSpacing') + 1)
        assert validate_solver_stages(DEFAULT_STAGES) == []

    def test_byes_remain_separate_atom_with_own_slack_key(self):
        """DoD 5: BalancedByeSpacing untouched — its own slack key, not merged."""
        assert CONSTRAINT_REGISTRY['BalancedByeSpacing'].slack_key == 'BalancedByeSpacing'
        assert (CONSTRAINT_REGISTRY['EqualMatchUpSpacing'].slack_key
                == 'EqualMatchUpSpacingConstraint')
        # They are distinct registry entries (not merged into one).
        assert 'BalancedByeSpacing' in CONSTRAINT_REGISTRY
        assert 'EqualMatchUpSpacing' in CONSTRAINT_REGISTRY


# ----------------------------------------------------------------------
# DoD 2: a critical_feasibility (non-soft_only) dispatch applies BOTH parts
# ----------------------------------------------------------------------


class TestDispatchAppliesBothParts:
    def test_non_soft_only_dispatch_applies_hard_and_soft(self):
        """Given a non-soft_only stage containing EqualMatchUpSpacing, the
        dispatcher runs apply_stage_1_hard (hard pairwise gaps) AND
        apply_stage_2_soft (penalty bucket). Proven by: soft bucket populated,
        AND a forced gap<=S pair is now INFEASIBLE (hard bit)."""
        model, X, data = _fixture()
        engine = _engine(model, X, data)
        stage = {'name': 'critical_feasibility', 'atoms': ['EqualMatchUpSpacing']}
        added, applied = apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )
        assert 'EqualMatchUpSpacing' in applied
        # Soft part populated the penalty bucket.
        assert data['penalties']['EqualMatchUpSpacing']['penalties']
        # Hard part bites: force T0 vs T1 at gap == S0 (=5) -> INFEASIBLE.
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1)) >= 1)
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1 + S0)) >= 1)  # rounds 1 and 6
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE

    def test_soft_only_dispatch_skips_hard(self):
        """Contrast (the pre-spec-017 bug): a soft_only stage skips the hard
        part. Forcing a gap=1 repeat is still FEASIBLE because only the soft
        penalty applies."""
        model, X, data = _fixture()
        engine = _engine(model, X, data)
        stage = {'name': 'soft_optimisation', 'atoms': ['EqualMatchUpSpacing'],
                 'soft_only': True}
        apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )
        # Soft bucket still populated.
        assert data['penalties']['EqualMatchUpSpacing']['penalties']
        # No hard clause: force T0 vs T1 in consecutive rounds (gap=1) -> FEASIBLE.
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1)) >= 1)
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 2)) >= 1)
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# DoD 3: regression — forced gap <= S INFEASIBLE; gap == S+1 FEASIBLE
# ----------------------------------------------------------------------


class TestHardRegression:
    def test_gap_equal_S_is_infeasible(self):
        """S0 = ideal_gap(10) = 5. Force T0 vs T1 at gap exactly 5 (rounds 1,6)
        -> forbidden (gap <= S) -> INFEASIBLE."""
        assert S0 == 5  # hand oracle
        model, X, data = _fixture()
        engine = _engine(model, X, data)
        engine.skip_constraints = ALL_ENGINE_KEYS - {'EqualMatchUpSpacing'}
        engine.apply_stage_1_hard()
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1)) >= 1)
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 6)) >= 1)  # gap = 5 == S0
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE

    def test_gap_S_plus_one_is_feasible(self):
        """Force T0 vs T1 at gap S+1 = 6 (rounds 1,7) -> allowed -> FEASIBLE."""
        model, X, data = _fixture()
        engine = _engine(model, X, data)
        engine.skip_constraints = ALL_ENGINE_KEYS - {'EqualMatchUpSpacing'}
        engine.apply_stage_1_hard()
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1)) >= 1)
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 7)) >= 1)  # gap = 6 == S0 + 1
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ----------------------------------------------------------------------
# DoD 4: --slack EqualMatchUpSpacingConstraint N loosens S by N
# ----------------------------------------------------------------------


class TestSlackLoosensThreshold:
    def test_slack_values_hand_oracle(self):
        """effective_spacing(10, slack) — S=5 at slack 0, S=3 at slack 2."""
        assert effective_spacing(T, base_slack=0, config_slack=0) == 5
        assert effective_spacing(T, base_slack=0, config_slack=2) == 3

    def test_gap4_infeasible_at_slack0_feasible_at_slack2(self):
        """Force T0 vs T1 at gap 4 (rounds 1,5).
        slack 0: S=5, gap 4 <= 5 -> INFEASIBLE.
        slack 2: S=3, gap 4 >  3 -> FEASIBLE."""
        # slack 0
        model, X, data = _fixture(slack=0)
        engine = _engine(model, X, data)
        engine.skip_constraints = ALL_ENGINE_KEYS - {'EqualMatchUpSpacing'}
        engine.apply_stage_1_hard()
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1)) >= 1)
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 5)) >= 1)  # gap = 4
        status0, _ = _solve(model)
        assert status0 == cp_model.INFEASIBLE

        # slack 2
        model2, X2, data2 = _fixture(slack=2)
        engine2 = _engine(model2, X2, data2)
        engine2.skip_constraints = ALL_ENGINE_KEYS - {'EqualMatchUpSpacing'}
        engine2.apply_stage_1_hard()
        model2.Add(sum(_pair_vars(X2, 'T0', 'T1', 1)) >= 1)
        model2.Add(sum(_pair_vars(X2, 'T0', 'T1', 5)) >= 1)  # gap = 4
        status2, _ = _solve(model2)
        assert status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE)
