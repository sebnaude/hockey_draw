"""Tests for the PreferredWeekendsAwayGround atom.

All tests use real CP-SAT models with hand-computed oracles. No mocks.

Fixture: 2 teams ('Maitland 1st', 'Norths 1st') on grade 'PHL', 2 dates each
with 1 timeslot at Maitland Park and 1 at Broadmeadow.  This gives us 4 X
variables per round per pair — enough to test that the solver can choose
between Maitland Park and Broadmeadow for a given pair.

Pairs: only one pair (Maitland 1st, Norths 1st), so balanced-matchup math is
trivial. We use single-pair fixtures to isolate the atom's behaviour.

The atom is purely soft — it never blocks feasibility. Tests confirm:
  1. avoid mode: solver picks an alternative date when feasible.
  2. prefer mode: solver schedules at the venue on the preferred date
     when feasible.
  3. conflicting prefer+avoid on same date/venue: no crash; penalties stack.
  4. weight 0 disables.
  5. missing config: atom returns 0, no bucket created.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.atoms.preferred_weekends_away_ground import (
    PreferredWeekendsAwayGround,
)
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

def _build_fixture(
    dates: List[str],
    *,
    include_maitland: bool = True,
    include_broadmeadow: bool = True,
    extra_maitland_field: str = None,
) -> Dict:
    """Build a 2-team / single-pair fixture across `dates`.

    Each date has at most one Maitland Park timeslot and one Broadmeadow
    timeslot (controlled by include_maitland/include_broadmeadow). Optionally
    add a second Maitland field (`extra_maitland_field` = field name).
    """
    bm_ef = PlayingField(location=BROADMEADOW, name='EF')
    mp_main = PlayingField(location=MAITLAND, name='Main')
    mp_extra = (
        PlayingField(location=MAITLAND, name=extra_maitland_field)
        if extra_maitland_field
        else None
    )

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland 1st', club=clubs[0], grade='PHL'),
        Team(name='Norths 1st', club=clubs[1], grade='PHL'),
    ]
    grades = [Grade(name='PHL', teams=[t.name for t in teams])]

    timeslots = []
    for week, date_str in enumerate(dates, start=1):
        if include_broadmeadow:
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time='11:30',
                week=week, day_slot=1, field=bm_ef, round_no=week,
            ))
        if include_maitland:
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time='11:30',
                week=week, day_slot=1, field=mp_main, round_no=week,
            ))
            if mp_extra is not None:
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time='13:00',
                    week=week, day_slot=2, field=mp_extra, round_no=week,
                ))

    fields = [bm_ef]
    if include_maitland:
        fields.append(mp_main)
        if mp_extra is not None:
            fields.append(mp_extra)

    return {
        'games': [('Maitland 1st', 'Norths 1st', 'PHL')],
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'PHL': len(dates), 'max': len(dates)},
        'constraint_slack': {},
        'penalty_weights': {'preferred_weekends_away_ground': 1000},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND, 'Norths': BROADMEADOW},
        'constraint_defaults': {},
        'preferred_weekends': [],
    }


def _build_model_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            if not ts.day:
                continue
            key = (
                t1, t2, grade,
                ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no,
                ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_w{ts.week}_{ts.field.name}_{ts.field.location[:2]}'
            )
    return model, X


def _registry(model) -> HelperVarRegistry:
    r = HelperVarRegistry(model)
    r.freeze({}, {})
    return r


def _solve(model, seconds: float = 5.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    status = solver.Solve(model)
    return status, solver


def _vars_at(X: Dict, *, date: str, field_location: str) -> List:
    return [v for k, v in X.items() if k[7] == date and k[10] == field_location]


# ---------------------------------------------------------------------------
# Scenario 1: avoid mode discourages scheduling at the venue on that date
# ---------------------------------------------------------------------------

class TestAvoidModeSuppresses:
    """Given an avoid entry for Maitland Park on 2026-04-05,
    When the solver minimises objective with exactly one game required,
    Then it picks the alternative date / venue.

    Hand oracle:
      Fixture: 2 dates × 2 venues each = 4 X vars for the single pair.
        date_1 = 2026-04-05 (Maitland Park EF — avoided; or Broadmeadow EF)
        date_2 = 2026-04-12 (Maitland Park EF; or Broadmeadow EF)
      Avoid entry: date 2026-04-05, Maitland Park (venue-level).
      We force exactly one game = 1 across all 4 vars.

      Without the atom: 4 equivalent ways to schedule.
      With atom: scheduling at (Maitland Park, 2026-04-05) raises penalty by
                 `multiplier=1` per game (and the bucket normalises).
      Optimiser minimises penalty → picks any of the 3 non-Maitland-on-04-05
      vars.

      Expected solver outcome: var at (date=2026-04-05, field_loc=Maitland Park)
      is 0; one of the other three vars = 1.
    """

    def test_avoid_picks_alternative(self):
        # GIVEN: 2 dates, both venues available, avoid Maitland Park on date 1.
        dates = ['2026-04-05', '2026-04-12']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'avoid',
                'description': 'avoid Maitland on date 1',
            },
        ]
        model, X = _build_model_X(data)

        # Require exactly one game scheduled.
        model.Add(sum(X.values()) == 1)

        # WHEN: apply atom and minimise objective (penalty only — sum vars unchanged at 1)
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 1, f'expected 1 penalty var, got {n}'

        bucket = data['penalties']['preferred_weekends_away_ground']
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # THEN: the var at (date=2026-04-05, Maitland Park) is 0.
        maitland_d1_vars = _vars_at(X, date='2026-04-05', field_location=MAITLAND)
        assert maitland_d1_vars, 'fixture must have a Maitland-on-d1 var'
        for v in maitland_d1_vars:
            assert solver.Value(v) == 0, 'avoid date+venue should not be picked'

        # And exactly one other var is 1.
        total_picked = sum(solver.Value(v) for v in X.values())
        assert total_picked == 1

        # Objective is 0 (avoided successfully).
        assert solver.ObjectiveValue() == 0

    def test_avoid_only_no_alternative_still_feasible(self):
        # GIVEN: only Maitland Park exists on date 1 (no alternative venue/date)
        dates = ['2026-04-05']
        data = _build_fixture(dates, include_broadmeadow=False)
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'avoid',
                'description': 'avoid but only option',
            },
        ]
        model, X = _build_model_X(data)
        model.Add(sum(X.values()) == 1)

        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 1

        bucket = data['penalties']['preferred_weekends_away_ground']
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # Soft constraint — solver accepts the penalty since no alternative exists.
        # Hand calc: 1 game forced at Maitland on avoid date → raw penalty = 1.
        # multiplier = max(1, 1000//1000) = 1 → bucket has 1 IntVar valued 1.
        # Objective = sum(bucket) = 1.
        assert solver.ObjectiveValue() == 1


# ---------------------------------------------------------------------------
# Scenario 2: prefer mode encourages scheduling at the venue on that date
# ---------------------------------------------------------------------------

class TestPreferModeEncourages:
    """Given a prefer entry for Maitland Park on 2026-04-05,
    When the solver minimises the penalty (no game forced),
    Then it schedules a game at Maitland Park on that date.

    Hand oracle:
      Fixture: 2 dates × 2 venues each = 4 X vars.
      Prefer entry: date=2026-04-05, Maitland Park, mode=prefer, target=1 (default).
      shortage = max(0, 1 - sum(vars_at_maitland_on_d1)).
      If 0 games at (Maitland Park, 2026-04-05): shortage = 1, penalty = 1.
      If ≥1 game there: shortage = 0, penalty = 0.

      We do NOT force a game total. Maximising X.sum (objective minus penalty)
      isn't part of this isolated test — we only minimise the bucket.

      Minimising the penalty alone → solver sets at least 1 of the Maitland-on-d1
      vars to 1 (penalty 0). Without the atom, the solver could leave everything
      at 0.

      Expected: at least 1 var at (Maitland Park, 2026-04-05) is 1.
    """

    def test_prefer_schedules_at_venue(self):
        dates = ['2026-04-05', '2026-04-12']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'prefer',
                'description': 'prefer game at Maitland on d1',
            },
        ]
        model, X = _build_model_X(data)

        # No game count forced. Only the soft penalty drives the objective.
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 1

        bucket = data['penalties']['preferred_weekends_away_ground']
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # THEN: at least one Maitland-Park-on-d1 var is 1.
        m_d1_vars = _vars_at(X, date='2026-04-05', field_location=MAITLAND)
        chosen = sum(solver.Value(v) for v in m_d1_vars)
        assert chosen >= 1, (
            'prefer mode should drive solver to schedule at least one game '
            'at the venue on the preferred date'
        )
        # Objective = 0 (shortage filled).
        assert solver.ObjectiveValue() == 0

    def test_prefer_no_vars_yields_constant_penalty(self):
        # GIVEN: prefer entry but the fixture has NO vars at that date+venue
        # (the date the entry references is not in the fixture at all).
        dates = ['2026-04-12']  # only this date, no '2026-04-05' vars.
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',  # date not present in any X key
                'field_location': MAITLAND,
                'mode': 'prefer',
                'description': 'prefer game on unsatisfiable date',
            },
        ]
        model, X = _build_model_X(data)

        # WHEN: apply the atom
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 1, 'unsatisfiable prefer still adds a constant penalty var'

        bucket = data['penalties']['preferred_weekends_away_ground']
        assert len(bucket['penalties']) == 1
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # Hand calc: target=1, multiplier=1 → constant value 1.
        assert solver.ObjectiveValue() == 1


# ---------------------------------------------------------------------------
# Scenario 3: conflicting prefer + avoid on same date+venue
# ---------------------------------------------------------------------------

class TestConflictingPreferAvoid:
    """Given a 'prefer' and an 'avoid' entry on the same date+venue,
    When the atom applies, Then no crash and both penalties accumulate."""

    def test_conflict_does_not_crash(self):
        # GIVEN: prefer AND avoid on (2026-04-05, Maitland Park).
        dates = ['2026-04-05', '2026-04-12']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'prefer',
                'description': 'prefer (conflicts with avoid)',
            },
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'avoid',
                'description': 'avoid (conflicts with prefer)',
            },
        ]
        model, X = _build_model_X(data)

        # WHEN: apply atom
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))

        # THEN: 2 penalty vars (one per entry), no exception
        assert n == 2
        bucket = data['penalties']['preferred_weekends_away_ground']
        assert len(bucket['penalties']) == 2

        # Solve with minimised penalty to confirm no crash and both terms count.
        # Hand oracle for the bucket sum:
        #   Let G = number of games scheduled at (Maitland Park, 2026-04-05) in {0,1}
        #   (only one timeslot exists at that venue on that date).
        #   avoid raw   = G                       → contributes G
        #   prefer raw  = max(0, 1 - G)           → contributes 1-G
        #   total       = G + (1 - G) = 1        regardless of G.
        # So the optimal objective for this bucket alone is exactly 1.
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.ObjectiveValue() == 1


# ---------------------------------------------------------------------------
# Scenario 4: weight 0 disables the atom
# ---------------------------------------------------------------------------

class TestWeightZeroDisables:
    """Given default weight 0, When the atom applies, Then no penalties added."""

    def test_zero_default_weight_disables(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        data['penalty_weights']['preferred_weekends_away_ground'] = 0
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'avoid',
            },
        ]
        model, X = _build_model_X(data)
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 0
        assert 'preferred_weekends_away_ground' not in data.get('penalties', {})


# ---------------------------------------------------------------------------
# Scenario 5: empty preferred_weekends → no-op
# ---------------------------------------------------------------------------

class TestEmptyConfigNoOp:
    """Given no preferred_weekends entries, When atom applies, Then n==0."""

    def test_empty_list(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        data['preferred_weekends'] = []
        model, X = _build_model_X(data)
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 0
        assert 'preferred_weekends_away_ground' not in data.get('penalties', {})

    def test_missing_key(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        del data['preferred_weekends']
        model, X = _build_model_X(data)
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 0


# ---------------------------------------------------------------------------
# Scenario 6: locked weeks excluded
# ---------------------------------------------------------------------------

class TestLockedWeeksExcluded:
    """Given the only week containing a venue+date is locked,
    When the atom applies, Then no real-X penalty is generated for that entry
    (avoid → 0 penalty since no vars; prefer → constant penalty since target unsatisfiable).
    """

    def test_locked_week_avoid_zero_penalty(self):
        dates = ['2026-04-05']  # week=1
        data = _build_fixture(dates)
        data['locked_weeks'] = {1}
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'avoid',
            },
        ]
        model, X = _build_model_X(data)
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        # avoid + no live vars = no penalty term
        assert n == 0
        assert 'preferred_weekends_away_ground' not in data.get('penalties', {}) or \
            len(data['penalties']['preferred_weekends_away_ground']['penalties']) == 0


# ---------------------------------------------------------------------------
# Scenario 7: field-level scoping
# ---------------------------------------------------------------------------

class TestFieldNameScoping:
    """Given an avoid entry with field_name set,
    When the fixture has two Maitland fields, Then only the named field is penalised.

    Hand oracle:
      Two fields at Maitland Park on date 1: 'Main' and 'Extra'.
      Avoid: (date=2026-04-05, field_location=Maitland Park, field_name='Main').
      Vars at lookup_key (date_1, Maitland Park, 'Main'): 1 var.
      Vars at (date_1, Maitland Park, 'Extra'): 1 var — NOT penalised.
      Solver picks 'Extra' to avoid penalty when forced to play once.
    """

    def test_field_name_isolates_penalty(self):
        dates = ['2026-04-05']
        data = _build_fixture(
            dates,
            include_broadmeadow=False,
            extra_maitland_field='Extra',
        )
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'field_name': 'Main',
                'mode': 'avoid',
            },
        ]
        model, X = _build_model_X(data)
        # force exactly 1 game
        model.Add(sum(X.values()) == 1)

        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 1
        bucket = data['penalties']['preferred_weekends_away_ground']
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # Find Main and Extra vars
        main_vars = [v for k, v in X.items() if k[9] == 'Main']
        extra_vars = [v for k, v in X.items() if k[9] == 'Extra']
        assert len(main_vars) == 1 and len(extra_vars) == 1
        # Solver should pick Extra (no penalty), not Main (penalty 1).
        assert solver.Value(main_vars[0]) == 0
        assert solver.Value(extra_vars[0]) == 1
        assert solver.ObjectiveValue() == 0


# ---------------------------------------------------------------------------
# Scenario 8: multi-date entry expansion ('dates' list)
# ---------------------------------------------------------------------------

class TestMultiDateExpansion:
    """Given an entry with 'dates': [d1, d2], When the atom applies,
    Then it creates one penalty term per date."""

    def test_dates_list_expands(self):
        dates = ['2026-04-05', '2026-04-12']
        data = _build_fixture(dates, include_broadmeadow=False)  # only Maitland
        data['preferred_weekends'] = [
            {
                'dates': ['2026-04-05', '2026-04-12'],
                'field_location': MAITLAND,
                'mode': 'avoid',
            },
        ]
        model, X = _build_model_X(data)
        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        # 2 dates → 2 penalty vars
        assert n == 2
        bucket = data['penalties']['preferred_weekends_away_ground']
        assert len(bucket['penalties']) == 2


# ---------------------------------------------------------------------------
# Scenario 9: per-entry weight override → multiplier applied
# ---------------------------------------------------------------------------

class TestPerEntryWeightMultiplier:
    """Given an entry weight = 3 × default, When the atom applies,
    Then the penalty term is multiplied by 3 (multiplier semantics)."""

    def test_higher_entry_weight_multiplies(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates, include_broadmeadow=False)
        # default weight = 1000; entry weight = 3000 → multiplier = 3
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'field_location': MAITLAND,
                'mode': 'avoid',
                'weight': 3000,
            },
        ]
        model, X = _build_model_X(data)
        # Force one game (the only var) so penalty raw = 1.
        model.Add(sum(X.values()) == 1)

        n = PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
        assert n == 1

        bucket = data['penalties']['preferred_weekends_away_ground']
        total_pen = sum(bucket['penalties'])
        model.Minimize(total_pen)

        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # Hand calc: raw=1, multiplier=3 → scaled var = 3.
        assert solver.ObjectiveValue() == 3


# ---------------------------------------------------------------------------
# Scenario 10: invalid mode raises
# ---------------------------------------------------------------------------

class TestInvalidEntryRaises:
    """Given malformed entries, When the atom applies, Then ValueError is raised."""

    def test_invalid_mode_raises(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {'date': '2026-04-05', 'field_location': MAITLAND, 'mode': 'bogus'},
        ]
        model, X = _build_model_X(data)
        with pytest.raises(ValueError, match='invalid mode'):
            PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))

    def test_missing_field_location_raises(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [{'date': '2026-04-05', 'mode': 'avoid'}]
        model, X = _build_model_X(data)
        with pytest.raises(ValueError, match="field_location"):
            PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))

    def test_both_date_and_dates_raises(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {
                'date': '2026-04-05',
                'dates': ['2026-04-12'],
                'field_location': MAITLAND,
                'mode': 'avoid',
            },
        ]
        model, X = _build_model_X(data)
        with pytest.raises(ValueError, match="both 'date' and 'dates'"):
            PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))

    def test_missing_both_date_keys_raises(self):
        dates = ['2026-04-05']
        data = _build_fixture(dates)
        data['preferred_weekends'] = [
            {'field_location': MAITLAND, 'mode': 'avoid'},
        ]
        model, X = _build_model_X(data)
        with pytest.raises(ValueError, match="'date' or 'dates'"):
            PreferredWeekendsAwayGround().apply(model, X, data, _registry(model))
