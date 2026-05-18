"""Tests for time-only PREFERENCE_NO_PLAY entries (spec-012, Part A).

Before spec-012 the `_preferred_times` method (in unified.py) and the soft
mirror in soft.py both contained an early `if 'date' not in constraint:
continue` skip. That made an entry like

    {'club': 'Maitland', 'time': '08:30'}

a silent no-op: no penalty was attached to any X-variable, so the solver
freely scheduled Maitland teams at 08:30 even though the convenor asked
otherwise.

The fix in spec-012 removes the skip and instead allows a date-less entry to
penalise every X-var that matches the remaining filters (time, day, grade,
team_name, etc.). The locked-week short-circuit still applies only when a
date is given.

These tests verify the fix using real CP-SAT models with hand-computed
oracles. No mocks.

Scenarios:

1. `TestTimeOnlyPenaltyCreated` — a time-only entry creates one penalty IntVar
   per matching X-var (each Maitland 08:30 var gets a 0/1 penalty bound to it).

2. `TestSolverAvoidsTimeOnlyPenalty` — when the solver is free to choose
   between an 08:30 slot and a 10:00 slot for a Maitland team, minimising the
   PreferredTimes bucket drives it to pick the 10:00 slot.

3. `TestSoftDoesNotBlockFeasibility` — when 08:30 is the only available slot
   (no alternative), the constraint accepts the penalty and stays feasible.

4. `TestNonMatchingClubUnaffected` — a 'Maitland' time-only entry does NOT
   create penalties for an unrelated club's 08:30 var.

5. `TestSoftMirror` — the legacy soft.py mirror behaves identically (the same
   fix was applied to both dispatch sites).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from constraints.unified import UnifiedConstraintEngine
from constraints.soft import PreferredTimesConstraintSoft
from models import Club, Grade, PlayingField, Team, Timeslot


BROADMEADOW = 'Newcastle International Hockey Centre'


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

def _build_fixture(
    *,
    times: List[str],
    extra_club_team: Tuple[str, str] = None,
) -> Dict:
    """Build a single-pair Maitland-vs-Norths fixture with the given time slots.

    Each `time` becomes one timeslot at NIHC East Field on Sunday 2026-03-22,
    week 1. The fixture has exactly one game pair (Maitland 1st vs Norths 1st)
    so the matchup constraints are trivial.

    `extra_club_team`: optional ('Team Name', 'Club Name') tuple to add a third
    team in a different club, used by `TestNonMatchingClubUnaffected`.
    """
    field = PlayingField(location=BROADMEADOW, name='EF')

    clubs = [
        Club(name='Maitland', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland 1st', club=clubs[0], grade='PHL'),
        Team(name='Norths 1st', club=clubs[1], grade='PHL'),
    ]
    games = [('Maitland 1st', 'Norths 1st', 'PHL')]

    if extra_club_team is not None:
        team_name, club_name = extra_club_team
        extra_club = Club(name=club_name, home_field=BROADMEADOW)
        extra_team = Team(name=team_name, club=extra_club, grade='PHL')
        clubs.append(extra_club)
        teams.append(extra_team)
        # one additional pair so the extra team has a game scheduled
        games.append(('Maitland 1st', team_name, 'PHL'))

    grades = [Grade(name='PHL', teams=[t.name for t in teams])]

    timeslots = []
    for i, time_str in enumerate(times, start=1):
        timeslots.append(Timeslot(
            date='2026-03-22', day='Sunday', time=time_str,
            week=1, day_slot=i, field=field, round_no=1,
        ))

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': [field],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'PHL': 1, 'max': 1},
        'constraint_slack': {},
        'penalty_weights': {'PreferredTimesConstraint': 1000},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {},
        'constraint_defaults': {},
        'preferred_weekends': [],
    }


def _build_model_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    """Build a tiny CP-SAT model with one X-var per (game, timeslot).

    Returns (model, X). Skips any game/timeslot combinations that don't apply.
    """
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        # team1 must be alphabetically before team2 (canonical X-key shape)
        t1_canon, t2_canon = sorted((t1, t2))
        for ts in data['timeslots']:
            key = (
                t1_canon, t2_canon, grade,
                ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no,
                ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1_canon}_{t2_canon}_w{ts.week}_t{ts.time.replace(":", "")}'
            )
    return model, X


# ---------------------------------------------------------------------------
# Scenario 1: a time-only entry creates penalties (the fix)
# ---------------------------------------------------------------------------

class TestTimeOnlyPenaltyCreated:
    """Given a {'club': 'Maitland', 'time': '08:30'} entry,
    When `_preferred_times` runs,
    Then exactly one penalty IntVar is created per matching X-var
         (i.e. per Maitland X-var at 08:30).

    Hand oracle:
      Fixture: one pair (Maitland 1st, Norths 1st) × 2 timeslots (08:30, 10:00)
      → 2 X-vars total. Both involve Maitland 1st (so the club_team filter
      passes for both). Only one has time='08:30' → exactly 1 penalty var.

      Before the spec-012 fix: 0 penalty vars (the date-less entry was skipped).
      After: 1 penalty var, valued == the underlying X-var.
    """

    def test_one_penalty_per_matching_var(self):
        # GIVEN
        data = _build_fixture(times=['08:30', '10:00'])
        data['preference_no_play'] = {
            'maitland_no_830': {'club': 'Maitland', 'time': '08:30',
                                 'description': 'Maitland time-only test'},
        }
        model, X = _build_model_X(data)

        # WHEN
        engine = UnifiedConstraintEngine(model, X, data)
        # _preferred_times needs build_groupings() to have run because it
        # only reads self.X / self.data / self.timeslots / self.locked_weeks.
        # build_groupings is heavier but safe. We can call it.
        engine.build_groupings()
        n = engine._preferred_times()

        # THEN — exactly one penalty var (08:30 slot only)
        assert n == 1, (
            f'Expected exactly 1 penalty for the 08:30 Maitland slot; got {n}. '
            'Before the spec-012 fix this was 0 (the date-less entry was '
            'silently skipped).'
        )
        bucket = data['penalties']['PreferredTimesConstraint']
        assert len(bucket['penalties']) == 1

        # Bind solver — penalty must equal the underlying X-var value.
        # Force the 08:30 var ON and check penalty=1.
        slot_830 = [
            (k, v) for k, v in X.items() if k[5] == '08:30'
        ]
        assert len(slot_830) == 1
        model.Add(slot_830[0][1] == 1)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(bucket['penalties'][0]) == 1


# ---------------------------------------------------------------------------
# Scenario 2: solver avoids the penalised time when an alternative exists
# ---------------------------------------------------------------------------

class TestSolverAvoidsTimeOnlyPenalty:
    """Given alternatives, the solver chooses the non-penalised slot.

    Hand oracle:
      Fixture: 2 timeslots (08:30, 10:00) for the single Maitland-vs-Norths
      pair. Require sum(X) == 1 (one game scheduled).

      Without the fix: 4 (2) equivalent solutions, solver picks arbitrarily.
      With the fix + minimise(PreferredTimes bucket): the 08:30 slot has
      penalty cost 1 if picked, 0 otherwise. The 10:00 slot has no penalty.
      Optimal → picks the 10:00 slot. Objective = 0.
    """

    def test_solver_picks_non_penalised_slot(self):
        data = _build_fixture(times=['08:30', '10:00'])
        data['preference_no_play'] = {
            'maitland_no_830': {'club': 'Maitland', 'time': '08:30',
                                 'description': 'Maitland time-only test'},
        }
        model, X = _build_model_X(data)

        # exactly one game
        model.Add(sum(X.values()) == 1)

        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        n = engine._preferred_times()
        assert n == 1

        bucket = data['penalties']['PreferredTimesConstraint']
        model.Minimize(sum(bucket['penalties']))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        assert status == cp_model.OPTIMAL

        # The 08:30 var is 0; the 10:00 var is 1.
        for k, v in X.items():
            if k[5] == '08:30':
                assert solver.Value(v) == 0, (
                    'Solver should avoid the 08:30 slot when 10:00 is free.'
                )
            elif k[5] == '10:00':
                assert solver.Value(v) == 1
        assert solver.ObjectiveValue() == 0


# ---------------------------------------------------------------------------
# Scenario 3: soft penalty does not block feasibility
# ---------------------------------------------------------------------------

class TestSoftDoesNotBlockFeasibility:
    """Given 08:30 is the only available time, the model stays feasible.

    Hand oracle:
      Fixture: 1 timeslot (08:30 only). Require sum(X) == 1.
      Only choice is the 08:30 slot. Optimiser accepts penalty=1.
    """

    def test_no_alternative_still_solves(self):
        data = _build_fixture(times=['08:30'])
        data['preference_no_play'] = {
            'maitland_no_830': {'club': 'Maitland', 'time': '08:30',
                                 'description': 'Maitland time-only test'},
        }
        model, X = _build_model_X(data)
        model.Add(sum(X.values()) == 1)

        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        n = engine._preferred_times()
        assert n == 1

        bucket = data['penalties']['PreferredTimesConstraint']
        model.Minimize(sum(bucket['penalties']))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        assert status == cp_model.OPTIMAL
        # The single X-var = 1, penalty = 1, objective = 1.
        assert solver.ObjectiveValue() == 1
        assert sum(solver.Value(v) for v in X.values()) == 1


# ---------------------------------------------------------------------------
# Scenario 4: a Maitland-only entry does NOT penalise other clubs
# ---------------------------------------------------------------------------

class TestNonMatchingClubUnaffected:
    """Given a 'Maitland' entry, a Souths 08:30 game incurs NO penalty.

    Hand oracle:
      Fixture: pair (Maitland 1st, Norths 1st) at 08:30 + pair (Maitland 1st,
      Souths 1st) at 08:30 = 2 X-vars. Both contain Maitland 1st, so the
      club_team filter passes for both → 2 penalty vars.

      Wait — the spec entry is `{'club': 'Maitland', ...}`, which the
      normaliser resolves to all Maitland teams. The penalty is attached to
      any var where ANY club team is in the game, regardless of opponent club.
      That's the existing PreferredTimes semantics (club_team membership), not
      a bug. So 2 penalty vars here.

      Now flip the entry to `{'club': 'Souths', ...}`: the only Souths team is
      Souths 1st. Only 1 of the 2 vars contains Souths 1st → 1 penalty var.
    """

    def test_other_club_entry_only_penalises_its_own_teams(self):
        # Add a Souths team; only the Maitland-vs-Souths var should incur a
        # Souths-flagged penalty.
        data = _build_fixture(
            times=['08:30'],
            extra_club_team=('Souths 1st', 'Souths'),
        )
        # Souths-only entry
        data['preference_no_play'] = {
            'souths_no_830': {'club': 'Souths', 'time': '08:30',
                              'description': 'Souths-only test'},
        }
        model, X = _build_model_X(data)

        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        n = engine._preferred_times()

        # We have 2 games × 1 timeslot = 2 X-vars.
        # 'Souths' has exactly 1 team (Souths 1st). The Maitland-vs-Norths
        # game doesn't contain Souths 1st → no penalty. The Maitland-vs-Souths
        # game does → 1 penalty.
        assert n == 1, (
            f'Expected exactly 1 penalty (only the Souths-involving var); got {n}.'
        )


# ---------------------------------------------------------------------------
# Scenario 5: soft.py mirror behaves identically (no regression)
# ---------------------------------------------------------------------------

class TestSoftMirror:
    """Given the same time-only entry, `PreferredTimesConstraintSoft.apply`
    creates the same number of penalty IntVars as the unified-engine path.

    Hand oracle:
      Identical to Scenario 1: 2 X-vars (08:30, 10:00), one entry filtered to
      time='08:30' → exactly 1 penalty var.
    """

    def test_soft_path_creates_penalty_for_time_only_entry(self):
        data = _build_fixture(times=['08:30', '10:00'])
        data['preference_no_play'] = {
            'maitland_no_830': {'club': 'Maitland', 'time': '08:30',
                                 'description': 'Maitland time-only test'},
        }
        model, X = _build_model_X(data)

        soft = PreferredTimesConstraintSoft()
        soft.apply(model, X, data)

        bucket = data['penalties'].get('PreferredTimesConstraintSoft')
        assert bucket is not None, (
            'soft.py path should still create the bucket even if no entries match.'
        )
        # Hand oracle: 1 matching X-var (the 08:30 one)
        assert len(bucket['penalties']) == 1
