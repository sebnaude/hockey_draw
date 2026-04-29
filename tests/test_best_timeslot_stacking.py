"""
Tests for EnsureBestTimeslotChoices stacking constraint (both original and AI).

Verifies the core rule: at a given location, you cannot use slot N on ANY field
until slot N-1 is filled on ALL fields at that location. Each location is independent.

All tests are parametrized to run against BOTH the original and AI constraint
implementations, since they must achieve the same thing.

Test categories:
1. FEASIBILITY - constraint doesn't break valid schedules
2. PER-FIELD CONTIGUITY - no gaps on a single field
3. CROSS-FIELD STACKING - all fields must fill before moving to next slot
4. EARLY-SLOT PUSH - games must start from slot 1
5. MULTI-LOCATION INDEPENDENCE - locations don't affect each other
6. 7PM PENALTY - 19:00 games are penalised
7. SOLUTION INSPECTION - verify actual solutions obey the stacking rule
"""

import pytest
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlayingField, Team, Club, Grade, Timeslot
from constraints.archived.ai import EnsureBestTimeslotChoicesAI
from constraints.archived.original import EnsureBestTimeslotChoices

BROADMEADOW = 'Newcastle International Hockey Centre'
MAITLAND = 'Maitland Park'

# Both implementations must pass every test
BOTH_VERSIONS = pytest.mark.parametrize(
    "constraint_cls",
    [EnsureBestTimeslotChoices, EnsureBestTimeslotChoicesAI],
    ids=["original", "AI"],
)


# ============== Helpers ==============

def make_fields_at(location, field_names):
    """Create fields at a single location."""
    return [PlayingField(location=location, name=n) for n in field_names]


def make_timeslots(fields, num_weeks, slots_per_field, day='Sunday',
                   start_date=datetime(2025, 3, 23), times=None):
    """Generate timeslots. If times is provided, use those time strings per slot index."""
    timeslots = []
    for week in range(1, num_weeks + 1):
        week_date = start_date + timedelta(weeks=week - 1)
        date_str = week_date.strftime('%Y-%m-%d')
        for field in fields:
            for slot in range(1, slots_per_field + 1):
                if times:
                    time_str = times[slot - 1]
                else:
                    hour = 8 + (slot - 1) * 2  # 8:00, 10:00, 12:00, ...
                    time_str = f'{hour}:00'
                timeslots.append(Timeslot(
                    date=date_str, day=day, time=time_str,
                    week=week, day_slot=slot, field=field, round_no=week
                ))
    return timeslots


def make_games(team_names, grade='3rd'):
    """Create all pairwise games for a list of team names."""
    from itertools import combinations
    return [(t1, t2, grade) for t1, t2 in combinations(sorted(team_names), 2)]


def make_teams_and_clubs(num_clubs, grade='3rd'):
    """Create N clubs with one team each."""
    clubs = []
    teams = []
    for i in range(num_clubs):
        name = chr(65 + i)  # A, B, C, ...
        club = Club(name=name, home_field=BROADMEADOW)
        clubs.append(club)
        teams.append(Team(name=f'{name} {grade}', club=club, grade=grade))
    return clubs, teams


def create_model_and_vars(games, timeslots):
    """Create model and decision variables."""
    model = cp_model.CpModel()
    X = {}
    for (t1, t2, grade) in games:
        for t in timeslots:
            if not t.day:
                continue
            key = (t1, t2, grade, t.day, t.day_slot, t.time,
                   t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1}_{t2}_{t.week}_{t.day_slot}_{t.field.name}')
    return model, X


def solve(model, timeout=10.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = 4
    return solver.Solve(model), solver


def is_feasible(status):
    return status in [cp_model.OPTIMAL, cp_model.FEASIBLE]


def build_data(games, timeslots, teams, fields, **extra):
    """Build minimal data dict for constraint."""
    data = {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'fields': fields,
        'locked_weeks': set(),
        'penalties': {},
    }
    data.update(extra)
    return data


def get_solution_games(X, solver):
    """Extract scheduled games from solution as list of keys."""
    return [k for k, v in X.items() if solver.Value(v) == 1]


def get_field_slot_usage(scheduled_keys):
    """From scheduled game keys, return {(week, location, field): set of day_slots used}."""
    usage = defaultdict(set)
    for k in scheduled_keys:
        # key: (t1, t2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
        week, field_name, field_location, day_slot = k[6], k[9], k[10], k[4]
        usage[(week, field_location, field_name)].add(day_slot)
    return usage


# ============== 1. FEASIBILITY ==============

class TestStackingFeasibility:
    """The constraint must not make valid data infeasible."""

    @BOTH_VERSIONS
    def test_basic_feasibility_single_field(self, constraint_cls):
        """Single field, few games — must be feasible."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs, teams = make_teams_and_clubs(3)
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 4, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    @BOTH_VERSIONS
    def test_basic_feasibility_multi_field(self, constraint_cls):
        """Multiple fields at same location — must be feasible."""
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF', 'SF'])
        clubs, teams = make_teams_and_clubs(6)
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 4, slots_per_field=6)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)

    @BOTH_VERSIONS
    def test_feasibility_multi_location(self, constraint_cls):
        """Fields at two different locations — must be feasible."""
        fields = (make_fields_at(BROADMEADOW, ['EF', 'WF']) +
                  make_fields_at(MAITLAND, ['Main']))
        clubs, teams = make_teams_and_clubs(4)
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 4, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)
        status, _ = solve(model)
        assert is_feasible(status)


# ============== 2. PER-FIELD CONTIGUITY ==============

class TestPerFieldContiguity:
    """Games on a single field must be contiguous — no gaps."""

    @BOTH_VERSIONS
    def test_rejects_gap_on_single_field(self, constraint_cls):
        """Force games at slot 1 and 3 but NOT slot 2 on same field — INFEASIBLE."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B', 'C', 'D', 'E', 'F']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd'), ('C 3rd', 'D 3rd', '3rd'), ('E 3rd', 'F 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force game at slot 1 on EF
        slot1_ef = [v for k, v in X.items() if k[4] == 1 and k[9] == 'EF' and k[0] == 'A 3rd']
        model.Add(sum(slot1_ef) == 1)

        # Force game at slot 3 on EF
        slot3_ef = [v for k, v in X.items() if k[4] == 3 and k[9] == 'EF' and k[0] == 'E 3rd']
        model.Add(sum(slot3_ef) == 1)

        # Force NO game at slot 2 on EF
        slot2_ef = [v for k, v in X.items() if k[4] == 2 and k[9] == 'EF']
        model.Add(sum(slot2_ef) == 0)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE, "Gap on single field should be rejected"

    @BOTH_VERSIONS
    def test_allows_contiguous_on_single_field(self, constraint_cls):
        """Games at slots 1, 2, 3 contiguously on same field — FEASIBLE."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B', 'C', 'D', 'E', 'F']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd'), ('C 3rd', 'D 3rd', '3rd'), ('E 3rd', 'F 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force one game per slot 1, 2, 3
        for slot in [1, 2, 3]:
            slot_vars = [v for k, v in X.items() if k[4] == slot and k[6] == 1]
            model.Add(sum(slot_vars) == 1)
        # No game at slot 4
        slot4_vars = [v for k, v in X.items() if k[4] == 4 and k[6] == 1]
        model.Add(sum(slot4_vars) == 0)

        status, _ = solve(model)
        assert is_feasible(status), "Contiguous games on single field should be feasible"


# ============== 3. CROSS-FIELD STACKING ==============

class TestCrossFieldStacking:
    """All fields at a location must fill slot N before any field uses slot N+1."""

    @BOTH_VERSIONS
    def test_rejects_next_slot_without_all_fields_filled(self, constraint_cls):
        """
        2 fields (EF, WF), force a game at slot 2 on EF but NO game at slot 1 on WF.
        This violates stacking: can't use slot 2 on EF unless slot 1 on WF is filled.
        """
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B', 'C', 'D']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd'), ('C 3rd', 'D 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force a game at slot 2 on EF
        slot2_ef = [v for k, v in X.items() if k[4] == 2 and k[9] == 'EF']
        model.Add(sum(slot2_ef) >= 1)

        # Force NO games on WF at slot 1
        slot1_wf = [v for k, v in X.items() if k[4] == 1 and k[9] == 'WF']
        model.Add(sum(slot1_wf) == 0)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE, (
            "Using slot 2 on EF without slot 1 filled on WF should be rejected"
        )

    @BOTH_VERSIONS
    def test_allows_stacked_usage(self, constraint_cls):
        """
        2 fields, 3 games: slot 1 on both fields, slot 2 on one field.
        This is valid stacking.
        """
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF'])
        clubs = [Club(name=n, home_field=BROADMEADOW)
                 for n in ['A', 'B', 'C', 'D', 'E', 'F']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd'), ('C 3rd', 'D 3rd', '3rd'),
                 ('E 3rd', 'F 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force slot 1 on both fields
        for field_name in ['EF', 'WF']:
            slot1 = [v for k, v in X.items() if k[4] == 1 and k[9] == field_name]
            model.Add(sum(slot1) >= 1)

        # Force slot 2 on EF only
        slot2_ef = [v for k, v in X.items() if k[4] == 2 and k[9] == 'EF']
        model.Add(sum(slot2_ef) >= 1)

        # Exactly 3 games total
        model.Add(sum(X.values()) == 3)

        status, _ = solve(model)
        assert is_feasible(status), "Properly stacked games should be feasible"

    @BOTH_VERSIONS
    def test_rejects_slot3_when_slot2_not_full(self, constraint_cls):
        """
        3 fields (EF, WF, SF). Force game at slot 3 on EF, but leave slot 2 empty
        on SF. Stacking says: to use slot 3 on EF, slot 2 must be used on ALL fields.
        """
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF', 'SF'])
        num_teams = 8
        clubs = [Club(name=chr(65 + i), home_field=BROADMEADOW) for i in range(num_teams)]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 1, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force game at slot 3 on EF
        slot3_ef = [v for k, v in X.items() if k[4] == 3 and k[9] == 'EF' and k[6] == 1]
        model.Add(sum(slot3_ef) >= 1)

        # Force NO game at slot 2 on SF
        slot2_sf = [v for k, v in X.items() if k[4] == 2 and k[9] == 'SF' and k[6] == 1]
        model.Add(sum(slot2_sf) == 0)

        # Slot 1 must be filled on all (otherwise the cascade would block slot 2 too)
        for f_name in ['EF', 'WF', 'SF']:
            slot1 = [v for k, v in X.items() if k[4] == 1 and k[9] == f_name and k[6] == 1]
            model.Add(sum(slot1) >= 1)
        # Slot 2 on EF and WF filled
        for f_name in ['EF', 'WF']:
            slot2 = [v for k, v in X.items() if k[4] == 2 and k[9] == f_name and k[6] == 1]
            model.Add(sum(slot2) >= 1)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE, (
            "Using slot 3 on EF when slot 2 on SF is empty should be rejected"
        )


# ============== 4. EARLY-SLOT PUSH ==============

class TestEarlySlotPush:
    """Games must start from slot 1 — can't skip to late slots."""

    @BOTH_VERSIONS
    def test_rejects_game_at_slot3_with_empty_slot1(self, constraint_cls):
        """Force a game at slot 3 but nothing at slot 1 — INFEASIBLE."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force game at slot 3
        slot3 = [v for k, v in X.items() if k[4] == 3]
        model.Add(sum(slot3) == 1)

        # No game at slot 1
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot1) == 0)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE, (
            "Game at slot 3 with empty slot 1 should be rejected"
        )

    @BOTH_VERSIONS
    def test_allows_game_at_slot1(self, constraint_cls):
        """Single game at slot 1 — always valid."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force exactly 1 game at slot 1
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot1) == 1)
        model.Add(sum(X.values()) == 1)

        status, _ = solve(model)
        assert is_feasible(status)

    @BOTH_VERSIONS
    def test_rejects_late_slot_skipping_early(self, constraint_cls):
        """
        2 fields. Force games at slot 2 on both fields, but slot 1 empty on one.
        Stacking: can't use slot 2 on any field unless slot 1 on ALL fields is filled.
        """
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B', 'C', 'D']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd'), ('C 3rd', 'D 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force both games at slot 2 (one per field)
        slot2_ef = [v for k, v in X.items() if k[4] == 2 and k[9] == 'EF']
        slot2_wf = [v for k, v in X.items() if k[4] == 2 and k[9] == 'WF']
        model.Add(sum(slot2_ef) >= 1)
        model.Add(sum(slot2_wf) >= 1)

        # No games at slot 1 on EF
        slot1_ef = [v for k, v in X.items() if k[4] == 1 and k[9] == 'EF']
        model.Add(sum(slot1_ef) == 0)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE


# ============== 5. MULTI-LOCATION INDEPENDENCE ==============

class TestMultiLocationIndependence:
    """Stacking at one location must not affect another location."""

    @BOTH_VERSIONS
    def test_locations_are_independent(self, constraint_cls):
        """
        Broadmeadow has 2 fields, Maitland has 1 field.
        Force game at slot 2 on Maitland (with slot 1 filled at Maitland).
        Force NO game at slot 1 on WF at Broadmeadow.
        If locations are independent, this should be FEASIBLE because
        Maitland's stacking only checks Maitland fields.
        """
        ef = PlayingField(location=BROADMEADOW, name='EF')
        wf = PlayingField(location=BROADMEADOW, name='WF')
        mf = PlayingField(location=MAITLAND, name='Main')
        fields = [ef, wf, mf]

        clubs = [Club(name=n, home_field=BROADMEADOW)
                 for n in ['A', 'B', 'C', 'D', 'E', 'F']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # At Maitland: game at slot 1 and slot 2 (valid stacking on 1 field)
        slot1_mait = [v for k, v in X.items() if k[4] == 1 and k[9] == 'Main']
        slot2_mait = [v for k, v in X.items() if k[4] == 2 and k[9] == 'Main']
        model.Add(sum(slot1_mait) >= 1)
        model.Add(sum(slot2_mait) >= 1)

        # At Broadmeadow: game at slot 1 on EF only, nothing on WF at all
        slot1_ef = [v for k, v in X.items() if k[4] == 1 and k[9] == 'EF']
        model.Add(sum(slot1_ef) >= 1)

        all_wf = [v for k, v in X.items() if k[9] == 'WF']
        model.Add(sum(all_wf) == 0)

        # Only need enough games to fill the forced slots
        # Maitland stacking is fine (1 field, slots 1+2 contiguous)
        # Broadmeadow: only EF slot 1 used, WF empty — still valid
        # (WF having no games is fine, stacking only checks: if f2 uses next_slot,
        #  f must use curr_slot. WF doesn't use any slot, so no constraint triggered.)

        status, _ = solve(model)
        assert is_feasible(status), "Locations should be stacked independently"

    @BOTH_VERSIONS
    def test_cross_location_does_not_bleed(self, constraint_cls):
        """
        Force Maitland at slot 2 with slot 1 empty. This should be INFEASIBLE
        at Maitland (per-field contiguity). Meanwhile Broadmeadow is fine.
        Confirms Maitland is checked independently.
        """
        ef = PlayingField(location=BROADMEADOW, name='EF')
        mf = PlayingField(location=MAITLAND, name='Main')
        fields = [ef, mf]

        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B', 'C', 'D']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Maitland: force game at slot 2 with NO game at slot 1
        slot2_mait = [v for k, v in X.items() if k[4] == 2 and k[9] == 'Main']
        slot1_mait = [v for k, v in X.items() if k[4] == 1 and k[9] == 'Main']
        model.Add(sum(slot2_mait) >= 1)
        model.Add(sum(slot1_mait) == 0)

        status, _ = solve(model)
        assert status == cp_model.INFEASIBLE, (
            "Maitland should enforce its own stacking independently"
        )


# ============== 6. 7PM PENALTY ==============

class TestSevenPmPenalty:
    """19:00 games should be penalised."""

    @BOTH_VERSIONS
    def test_7pm_penalty_vars_created(self, constraint_cls):
        """Constraint must create penalty variables for 19:00 timeslots."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd')]
        times = ['10:00', '14:00', '19:00']
        timeslots = make_timeslots(fields, 1, slots_per_field=3, times=times)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        assert 'EnsureBestTimeslotChoices_7pm' in data['penalties'], (
            "7pm penalty key must exist in penalties dict"
        )
        penalties = data['penalties']['EnsureBestTimeslotChoices_7pm']['penalties']
        assert len(penalties) > 0, "Must have at least one 7pm penalty variable"

    @BOTH_VERSIONS
    def test_7pm_penalty_not_for_other_times(self, constraint_cls):
        """Penalty vars should ONLY be for 19:00 games, not 10:00 or 14:00."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd')]
        # Only non-7pm times
        times = ['10:00', '14:00', '16:00']
        timeslots = make_timeslots(fields, 1, slots_per_field=3, times=times)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        penalties = data['penalties']['EnsureBestTimeslotChoices_7pm']['penalties']
        assert len(penalties) == 0, "No 7pm penalties when no 19:00 timeslots exist"

    @BOTH_VERSIONS
    def test_solver_prefers_non_7pm(self, constraint_cls):
        """Given a choice between 10:00 and 19:00 with equal stacking, solver should avoid 7pm."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd')]
        # Only 2 slots: 10:00 (slot 1) and 19:00 (slot 2) — stacking allows slot 1
        times = ['10:00', '19:00']
        timeslots = make_timeslots(fields, 1, slots_per_field=2, times=times)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force exactly 1 game
        model.Add(sum(X.values()) == 1)

        # Maximise (games scheduled - penalties), same as production solver
        penalty_vars = data['penalties']['EnsureBestTimeslotChoices_7pm']['penalties']
        weight = data['penalties']['EnsureBestTimeslotChoices_7pm']['weight']
        model.Maximize(sum(X.values()) * weight * 2 - sum(pv * weight for pv in penalty_vars))

        status, solver = solve(model)
        assert is_feasible(status)

        # Check that the game is at 10:00, not 19:00
        scheduled = get_solution_games(X, solver)
        assert len(scheduled) == 1
        assert scheduled[0][5] == '10:00', (
            f"Solver should prefer 10:00 over 19:00, got {scheduled[0][5]}"
        )


# ============== 7. SOLUTION INSPECTION ==============

class TestSolutionInspection:
    """Solve and verify that actual solutions obey stacking rules."""

    @BOTH_VERSIONS
    def test_solution_obeys_stacking_2_fields(self, constraint_cls):
        """
        Solve with 2 fields, inspect solution to verify stacking:
        for each week, if any field uses slot N+1, all fields must use slot N.
        """
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF'])
        clubs, teams = make_teams_and_clubs(6)
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 3, slots_per_field=5)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)

        # Need at least NoDoubleBooking to get a realistic solution
        from constraints.archived.ai import NoDoubleBookingTeamsConstraintAI, NoDoubleBookingFieldsConstraintAI
        NoDoubleBookingTeamsConstraintAI().apply(model, X, data)
        NoDoubleBookingFieldsConstraintAI().apply(model, X, data)
        constraint_cls().apply(model, X, data)

        # Schedule some games
        model.Add(sum(X.values()) >= 6)

        status, solver = solve(model, timeout=15)
        assert is_feasible(status), "Should find a feasible solution"

        scheduled = get_solution_games(X, solver)
        usage = get_field_slot_usage(scheduled)

        # Verify stacking rule per location
        # Group by (week, location)
        loc_weeks = defaultdict(lambda: defaultdict(set))
        for (week, location, field_name), slots in usage.items():
            loc_weeks[(week, location)][field_name] = slots

        for (week, location), field_slots in loc_weeks.items():
            field_names = list(field_slots.keys())
            all_slots = set()
            for s in field_slots.values():
                all_slots.update(s)

            sorted_slots = sorted(all_slots)
            for i in range(len(sorted_slots) - 1):
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]
                # If any field uses next_slot, all fields must use curr_slot
                for f2 in field_names:
                    if next_slot in field_slots[f2]:
                        for f in field_names:
                            assert curr_slot in field_slots[f], (
                                f"Week {week}, {location}: field {f2} uses slot {next_slot} "
                                f"but field {f} doesn't use slot {curr_slot}. "
                                f"Field usage: {dict(field_slots)}"
                            )

    @BOTH_VERSIONS
    def test_solution_obeys_stacking_3_fields(self, constraint_cls):
        """Same as above but with 3 fields — more thorough."""
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF', 'SF'])
        clubs, teams = make_teams_and_clubs(8)
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 4, slots_per_field=6)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)

        from constraints.archived.ai import NoDoubleBookingTeamsConstraintAI, NoDoubleBookingFieldsConstraintAI
        NoDoubleBookingTeamsConstraintAI().apply(model, X, data)
        NoDoubleBookingFieldsConstraintAI().apply(model, X, data)
        constraint_cls().apply(model, X, data)

        model.Add(sum(X.values()) >= 8)

        status, solver = solve(model, timeout=15)
        assert is_feasible(status)

        scheduled = get_solution_games(X, solver)
        usage = get_field_slot_usage(scheduled)

        loc_weeks = defaultdict(lambda: defaultdict(set))
        for (week, location, field_name), slots in usage.items():
            loc_weeks[(week, location)][field_name] = slots

        for (week, location), field_slots in loc_weeks.items():
            field_names = list(field_slots.keys())
            all_slots = set()
            for s in field_slots.values():
                all_slots.update(s)
            sorted_slots = sorted(all_slots)

            for i in range(len(sorted_slots) - 1):
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]
                for f2 in field_names:
                    if next_slot in field_slots[f2]:
                        for f in field_names:
                            assert curr_slot in field_slots[f], (
                                f"Week {week}, {location}: field {f2} uses slot {next_slot} "
                                f"but field {f} doesn't use slot {curr_slot}. "
                                f"Field usage: {dict(field_slots)}"
                            )

    @BOTH_VERSIONS
    def test_solution_starts_from_slot1(self, constraint_cls):
        """Verify that if any games are scheduled at a location, slot 1 is always used."""
        fields = make_fields_at(BROADMEADOW, ['EF', 'WF'])
        clubs, teams = make_teams_and_clubs(4)
        games = make_games([t.name for t in teams])
        timeslots = make_timeslots(fields, 3, slots_per_field=4)
        data = build_data(games, timeslots, teams, fields)

        model, X = create_model_and_vars(games, timeslots)

        from constraints.archived.ai import NoDoubleBookingTeamsConstraintAI, NoDoubleBookingFieldsConstraintAI
        NoDoubleBookingTeamsConstraintAI().apply(model, X, data)
        NoDoubleBookingFieldsConstraintAI().apply(model, X, data)
        constraint_cls().apply(model, X, data)

        model.Add(sum(X.values()) >= 3)

        status, solver = solve(model, timeout=15)
        assert is_feasible(status)

        scheduled = get_solution_games(X, solver)
        usage = get_field_slot_usage(scheduled)

        # Group by (week, location)
        loc_weeks = defaultdict(lambda: defaultdict(set))
        for (week, location, field_name), slots in usage.items():
            loc_weeks[(week, location)][field_name] = slots

        for (week, location), field_slots in loc_weeks.items():
            all_slots = set()
            for s in field_slots.values():
                all_slots.update(s)
            if all_slots:
                assert 1 in all_slots, (
                    f"Week {week}, {location}: games scheduled at slots {sorted(all_slots)} "
                    f"but slot 1 is not used. Stacking requires starting from slot 1."
                )


# ============== 8. LOCKED WEEKS ==============

class TestLockedWeeks:
    """Locked weeks should be skipped by the constraint."""

    @BOTH_VERSIONS
    def test_locked_week_allows_gap(self, constraint_cls):
        """In a locked week, stacking should NOT be enforced."""
        fields = make_fields_at(BROADMEADOW, ['EF'])
        clubs = [Club(name=n, home_field=BROADMEADOW) for n in ['A', 'B']]
        teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]
        games = [('A 3rd', 'B 3rd', '3rd')]
        timeslots = make_timeslots(fields, 1, slots_per_field=3)
        data = build_data(games, timeslots, teams, fields, locked_weeks={1})

        model, X = create_model_and_vars(games, timeslots)
        constraint_cls().apply(model, X, data)

        # Force game at slot 3 with nothing at slot 1 — should be fine since week 1 is locked
        slot3 = [v for k, v in X.items() if k[4] == 3]
        slot1 = [v for k, v in X.items() if k[4] == 1]
        model.Add(sum(slot3) == 1)
        model.Add(sum(slot1) == 0)

        status, _ = solve(model)
        assert is_feasible(status), "Locked weeks should not enforce stacking"
