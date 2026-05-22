"""spec-021 DoD 11 — VenueEarliestSlotFill is wired into a HARD production stage.

Before spec-021 the earliest-fill rule (`EnsureBestTimeslotChoices`) sat ONLY in
the `soft_only` `soft_optimisation` stage, so `apply_solver_stage` skipped its
hard part — the earliest-packing guarantee never applied in production. spec-021
replaces it with the `VenueEarliestSlotFill` atom in `critical_feasibility`
(non-soft_only). This test proves the atom (a) lives in a non-soft_only stage in
`DEFAULT_STAGES` and (b) actually emits hard constraints when dispatched through
the real `apply_solver_stage` path.

Real CP-SAT model + real engine, no mocks.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from config.defaults import DEFAULT_STAGES
from constraints.atoms.base import BROADMEADOW
from constraints.stages import apply_solver_stage, validate_solver_stages
from constraints.unified import UnifiedConstraintEngine
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars


def _fixture() -> Tuple[cp_model.CpModel, Dict, Dict]:
    """3 teams in one grade; 3 NIHC slots in one week (date 2026-03-22).

    Each of the 3 matchups gets a candidate var in each of slots 1,2,3 →
    every slot is occupiable, so the venue grouping has 3 slots and the atom
    adds exactly 2 monotone-fill implications.
    """
    field = PlayingField(location=BROADMEADOW, name='EF')
    clubs = [Club(name=f'C{i}', home_field=BROADMEADOW) for i in range(3)]
    teams = [Team(name=f'T{i}', club=clubs[i], grade='Test') for i in range(3)]
    grade = Grade(name='Test', teams=[t.name for t in teams])
    games: List[Tuple[str, str, str]] = [
        (a, b, 'Test') for a, b in combinations([t.name for t in teams], 2)
    ]
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time=f'1{s}:00',
                 week=1, day_slot=s, field=field, round_no=1)
        for s in (1, 2, 3)
    ]
    model, X = create_model_and_vars(games, timeslots)
    data = {
        'games': games, 'timeslots': timeslots, 'teams': teams,
        'grades': [grade], 'clubs': clubs, 'fields': [field],
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {'Test': 1, 'max': 1},
        'constraint_slack': {}, 'penalty_weights': {}, 'penalties': {},
        'forced_games': [], 'blocked_games': [], 'team_conflicts': [],
        'phl_preferences': {}, 'club_days': {}, 'preference_no_play': {},
        'home_field_map': {}, 'constraint_defaults': {},
    }
    return model, X, data


class TestVenueFillStagePlacement:
    def test_in_a_non_soft_only_stage_only(self):
        # Lives in critical_feasibility (hard); never in a soft_only stage.
        hard_stages = [s for s in DEFAULT_STAGES
                       if 'VenueEarliestSlotFill' in s['atoms'] and not s.get('soft_only')]
        soft_stages = [s for s in DEFAULT_STAGES
                       if 'VenueEarliestSlotFill' in s['atoms'] and s.get('soft_only')]
        assert len(hard_stages) == 1
        assert hard_stages[0]['name'] == 'critical_feasibility'
        assert soft_stages == []
        assert validate_solver_stages(DEFAULT_STAGES) == []


class TestVenueFillProductionDispatch:
    def test_dispatch_emits_hard_constraints(self):
        """Dispatching VenueEarliestSlotFill through apply_solver_stage on a
        non-soft_only stage adds > 0 constraints, and the emitted chain bites:
        forcing a game into slot 3 with slot 1 empty is INFEASIBLE."""
        model, X, data = _fixture()
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        stage = {'name': 'critical_feasibility', 'atoms': ['VenueEarliestSlotFill']}
        added, applied = apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )
        assert 'VenueEarliestSlotFill' in applied
        assert added > 0  # hard constraints really emitted in the production path

        # Hard chain bites: any game in slot 3, none in slot 1 -> INFEASIBLE.
        slot3 = [v for k, v in X.items() if k[4] == 3]
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot3) >= 1)
        model.Add(sum(slot1) == 0)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.Solve(model) == cp_model.INFEASIBLE
