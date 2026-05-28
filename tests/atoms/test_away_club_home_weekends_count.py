"""Tests for AwayClubHomeWeekendsCount atom (spec-004, redesigned in spec-037).

Real CP-SAT models, no mocks. The atom now enforces a SINGLE two-sided range
on the number of Sunday-home weekends for each away-based club:

    min_sundays_home(data, club) <= sum(sunday_home_indicators) <= max_sundays_home(data, club)

where the bounds are derived from per-grade home-game demands (see
``constraints/atoms/_phl_forced_friday_helper.py``). Friday-home and total-
weekend counts are NOT pinned by the atom; FORCED_GAMES config carries the
PHL Friday count instead.

Hand-computed oracles in each test. GIVEN/WHEN/THEN structure.

Six scenarios per spec-037 DoD #6:
  1. Maitland-shape bounds: min=9, max=10.
  2. Gosford bounds: min=0, max=10.
  3. 3rd-dominant bounds: min=11, max=11.
  4. CP-SAT solve on a Maitland-shaped fixture (no FORCED Fridays): feasible
     AND 9 <= sundays <= 10.
  5. Same fixture + 3 PHL Friday FORCED: feasible AND 9 <= sundays <= 10
     (DoD #6 calls out "specifically 9" as one valid landing, but the atom
     enforces the range bound only; the exact landing is solver-dependent).
  6. All PHL home games (10) forced to Friday extreme: feasible AND
     sundays == 9 (PHL contributes 0 home Sundays; 3rd contributes 9; floor
     is binding by construction).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms._phl_forced_friday_helper import (
    away_club_max_sundays_home,
    away_club_min_sundays_home,
)
from constraints.atoms.away_club_home_weekends_count import (
    AwayClubHomeWeekendsCount,
)
from constraints.atoms.base import BROADMEADOW, MAITLAND
from constraints.helper_vars import HelperVarRegistry
from models import Club, Grade, PlayingField, Team, Timeslot


# ---------------------------------------------------------------------------
# Pure bounds builder — for scenarios 1-3 (no CP-SAT).
# ---------------------------------------------------------------------------


def _build_minimal_data(clubs_grades, num_rounds):
    """Build a minimal `data` dict with only what the bounds helpers read."""
    clubs = []
    teams = []
    for club_name, grades in clubs_grades.items():
        club = Club(name=club_name, home_field='dummy')
        clubs.append(club)
        for g in grades:
            teams.append(Team(name=f'{club_name} {g}', club=club, grade=g))
    return {
        'teams': teams,
        'clubs': clubs,
        'num_rounds': dict(num_rounds),
        'grades': [],
        'forced_games': [],
    }


# ---------------------------------------------------------------------------
# Multi-grade Maitland-vs-Norths fixture builder — for scenarios 4-6 (CP-SAT).
# Single Maitland-vs-Norths pair per grade. Each week has Friday + Sunday
# slots at BM (Norths home) and MP (Maitland home). 3rd grade has no Friday
# slots — non-PHL grades have no Friday alternative, which is what drives the
# floor.
# ---------------------------------------------------------------------------


def _build_multi_grade_fixture(
    *,
    num_weeks: int,
    phl_required: int,
    third_required: int,
) -> Dict:
    bm_ef = PlayingField(location=BROADMEADOW, name='EF')
    mp = PlayingField(location=MAITLAND, name='Main')

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]
    teams = [
        Team(name='Maitland PHL', club=clubs[0], grade='PHL'),
        Team(name='Norths PHL', club=clubs[1], grade='PHL'),
        Team(name='Maitland 3rd', club=clubs[0], grade='3rd'),
        Team(name='Norths 3rd', club=clubs[1], grade='3rd'),
    ]
    grade_objs = [
        Grade(name='PHL', teams=['Maitland PHL', 'Norths PHL']),
        Grade(name='3rd', teams=['Maitland 3rd', 'Norths 3rd']),
    ]
    num_rounds = {
        'PHL': phl_required,
        '3rd': third_required,
        'max': max(phl_required, third_required),
    }

    games: List[Tuple[str, str, str]] = [
        ('Maitland PHL', 'Norths PHL', 'PHL'),
        ('Maitland 3rd', 'Norths 3rd', '3rd'),
    ]

    timeslots: List[Timeslot] = []
    # Two Sunday slots per week per venue (so each grade has somewhere to
    # land independently), plus one Friday slot per venue (PHL only). Friday
    # vars for 3rd grade are filtered out by `_build_X` below.
    base_date_n = 22  # 2026-03-22 = week 1 Sunday
    for week in range(1, num_weeks + 1):
        sun_d = f'2026-03-{base_date_n + 7 * (week - 1):02d}'
        fri_d = f'2026-03-{base_date_n + 7 * (week - 1) - 2:02d}'
        for slot, time in enumerate(['11:30', '13:30'], 1):
            timeslots.append(Timeslot(
                date=sun_d, day='Sunday', time=time, week=week,
                day_slot=slot, field=bm_ef, round_no=week,
            ))
            timeslots.append(Timeslot(
                date=sun_d, day='Sunday', time=time, week=week,
                day_slot=slot, field=mp, round_no=week,
            ))
        # Friday slot per venue (PHL only).
        timeslots.append(Timeslot(
            date=fri_d, day='Friday', time='19:00', week=week,
            day_slot=1, field=bm_ef, round_no=week,
        ))
        timeslots.append(Timeslot(
            date=fri_d, day='Friday', time='19:00', week=week,
            day_slot=1, field=mp, round_no=week,
        ))

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grade_objs,
        'clubs': clubs,
        'fields': [bm_ef, mp],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': num_rounds,
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
    }


def _build_X(model: cp_model.CpModel, data: Dict) -> Dict:
    """Build X vars for every (game, timeslot). Skip Friday slots for non-PHL
    grades — production behaviour and what makes the floor meaningful."""
    X = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            if grade != 'PHL' and ts.day == 'Friday':
                continue  # non-PHL grades don't play Friday
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name, ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{ts.week}_{ts.day}_{ts.field.name}_{ts.day_slot}'
            )
    return X


def _add_basic_hard_constraints(
    model: cp_model.CpModel, X: Dict, data: Dict,
) -> None:
    """Per-pair sum == num_rounds[grade]; per slot-bucket <= 1.

    Per-PAIR home/away balance (5050) is also enforced here so the floor/ceiling
    interact with realistic placement — without it the solver could put every
    Maitland-pair game at Maitland Park (or every at Broadmeadow), which makes
    the bounds vacuous.
    """
    from collections import defaultdict
    pair_vars = defaultdict(list)
    field_slot_vars = defaultdict(list)
    home_vars_by_pair_grade = defaultdict(list)
    for key, var in X.items():
        t1, t2, grade = key[0], key[1], key[2]
        pair_vars[(t1, t2, grade)].append(var)
        # Slot bucket: at most one game per (week, day, day_slot, field, location).
        field_slot_vars[
            (key[6], key[3], key[4], key[9], key[10])
        ].append(var)
        # Home at Maitland Park for Maitland-vs-Norths pairs (any grade).
        if key[10] == MAITLAND:
            home_vars_by_pair_grade[(t1, t2, grade)].append(var)
    num_rounds = data['num_rounds']
    for (t1, t2, grade), vars_list in pair_vars.items():
        model.Add(sum(vars_list) == num_rounds[grade])
    for vars_list in field_slot_vars.values():
        model.Add(sum(vars_list) <= 1)
    # 50/50 per pair-grade: home_count must equal floor or ceil of N/2.
    for (t1, t2, grade), home_list in home_vars_by_pair_grade.items():
        n = num_rounds[grade]
        lo = n // 2
        hi = (n + 1) // 2
        model.Add(sum(home_list) >= lo)
        model.Add(sum(home_list) <= hi)


def _solve(model: cp_model.CpModel, *, time_s: float = 15.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_s
    return solver, solver.Solve(model)


def _count_sunday_home_weeks(
    solver: cp_model.CpSolver, X: Dict, data: Dict, club: str,
) -> int:
    """Count DISTINCT weeks the club has a Sunday home game at its home venue."""
    home_venue = data['home_field_map'][club]
    team_to_club = {t.name: t.club.name for t in data['teams']}
    weeks_seen = set()
    for key, var in X.items():
        if solver.Value(var) != 1:
            continue
        if key[3] != 'Sunday':
            continue
        if key[10] != home_venue:
            continue
        t1, t2 = key[0], key[1]
        if team_to_club.get(t1) != club and team_to_club.get(t2) != club:
            continue
        weeks_seen.add(key[6])
    return len(weeks_seen)


def _count_friday_home_weeks(
    solver: cp_model.CpSolver, X: Dict, data: Dict, club: str,
) -> int:
    home_venue = data['home_field_map'][club]
    team_to_club = {t.name: t.club.name for t in data['teams']}
    weeks_seen = set()
    for key, var in X.items():
        if solver.Value(var) != 1:
            continue
        if key[3] != 'Friday':
            continue
        if key[10] != home_venue:
            continue
        t1, t2 = key[0], key[1]
        if team_to_club.get(t1) != club and team_to_club.get(t2) != club:
            continue
        weeks_seen.add(key[6])
    return len(weeks_seen)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBoundsOnly:
    """Scenarios 1-3 of DoD #6 — pure bounds-helper assertions, no CP-SAT."""

    def test_scenario_1_maitland_shape_floor9_ceil10(self):
        """GIVEN Maitland fields PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18,
        WHEN computing bounds via the helpers,
        THEN min=9, max=10.

        Hand-computed:
          non-PHL home-games per grade: 3rd 9, 4th 9, 5th 8, 6th 9
            -> floor = max(9, 9, 8, 9) = 9
          all-grade incl PHL: 10 (PHL ceil), 9, 9, 8, 9
            -> ceiling = max(10, 9, 9, 8, 9) = 10
        """
        data = _build_minimal_data(
            {'Maitland': ['PHL', '3rd', '4th', '5th', '6th']},
            {'PHL': 20, '3rd': 18, '4th': 18, '5th': 16, '6th': 18},
        )
        assert away_club_min_sundays_home(data, 'Maitland') == 9
        assert away_club_max_sundays_home(data, 'Maitland') == 10

    def test_scenario_2_gosford_phl_only_floor0_ceil10(self):
        """GIVEN Gosford fields only PHL=20,
        WHEN computing bounds,
        THEN min=0 (no non-PHL grades), max=10.

        Hand-computed: non-PHL list empty -> 0; ceil(20/2) = 10.
        """
        data = _build_minimal_data(
            {'Gosford': ['PHL']},
            {'PHL': 20},
        )
        assert away_club_min_sundays_home(data, 'Gosford') == 0
        assert away_club_max_sundays_home(data, 'Gosford') == 10

    def test_scenario_3_third_dominant_floor11_ceil11(self):
        """GIVEN a club fields PHL=18, 3rd=22 (3rd plays MORE than PHL),
        WHEN computing bounds,
        THEN min=11, max=11 (3rd drives both; range collapses to equality).

        Hand-computed:
          non-PHL home games: 3rd 11 -> floor = 11
          all-grade incl PHL: 11 (3rd), 9 (PHL ceil 18/2) -> ceiling = 11
        """
        data = _build_minimal_data(
            {'TestClub': ['PHL', '3rd']},
            {'PHL': 18, '3rd': 22},
        )
        assert away_club_min_sundays_home(data, 'TestClub') == 11
        assert away_club_max_sundays_home(data, 'TestClub') == 11


class TestSolverWithAtom:
    """Scenarios 4-6 of DoD #6 — CP-SAT solves with the atom applied."""

    def test_scenario_4_no_forced_then_solver_sundays_in_range(self):
        """GIVEN Maitland-shape: PHL=20, 3rd=18, no FORCED Fridays,
        WHEN solver finds any feasible assignment,
        THEN 9 <= count(Sunday home indicators true) <= 10.

        Hand-computed bounds (helper, this fixture):
          non-PHL grades fielded by Maitland = ['3rd']; floor = 18 // 2 = 9.
          all grades incl PHL = ['PHL', '3rd']; ceiling = max(20//2 ceil, 18//2)
            = max(10, 9) = 10.

        Sunday home count must be in [9, 10] for any feasible solution.
        """
        data = _build_multi_grade_fixture(
            num_weeks=22, phl_required=20, third_required=18,
        )
        assert away_club_min_sundays_home(data, 'Maitland') == 9
        assert away_club_max_sundays_home(data, 'Maitland') == 10

        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)
        registry = HelperVarRegistry(model)
        AwayClubHomeWeekendsCount().apply(model, X, data, registry)

        solver, status = _solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )
        sun_count = _count_sunday_home_weeks(solver, X, data, 'Maitland')
        assert 9 <= sun_count <= 10, (
            f'Expected Sunday home count in [9, 10], got {sun_count}'
        )

    def test_scenario_5_three_forced_fridays_then_sundays_in_range(self):
        """GIVEN Maitland-shape + 3 PHL Friday games FORCED at Maitland Park,
        WHEN solver runs,
        THEN model feasible AND Sunday-home count is in [9, 10].

        Hand-computed:
          PHL home games for Maitland-vs-Norths = 10 (half of 20).
          With 3 forced to Friday at MP, 7 PHL home games remain on Sunday.
          3rd-grade home games at MP = 9 (half of 18), all on Sunday (3rd has
          no Friday alternative).
          The Sunday-home INDICATOR per week OR's any Maitland home Sunday
          var (PHL or 3rd). PHL contributes 7 weeks, 3rd contributes 9 weeks.
          Their union W satisfies max(7, 9) = 9 <= W <= 7 + 9 = 16. The atom
          clamps W to [9, 10]. Whichever value the solver picks (9 if it
          overlaps PHL into 3rd-weeks, 10 if exactly one PHL week is
          disjoint), feasibility holds and the count sits inside the range.

        Note: spec-037 DoD #6 calls out 9 as a possible "specifically"
        landing, but achieving the floor requires overlap that the atom does
        not enforce structurally — only the bound. The range assertion is
        the rule the atom guarantees; the exact landing depends on the
        objective (here unforced).
        """
        data = _build_multi_grade_fixture(
            num_weeks=22, phl_required=20, third_required=18,
        )
        # Force exactly 3 PHL Friday games at Maitland Park via the same
        # variable-sum constraint that the FORCED_GAMES pipeline produces.
        # (The atom does NOT read FORCED_GAMES — this `model.Add` is the
        # config-side guarantee.)
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)
        forced_vars = [
            v for k, v in X.items()
            if k[2] == 'PHL' and k[3] == 'Friday' and k[10] == MAITLAND
        ]
        model.Add(sum(forced_vars) == 3)

        registry = HelperVarRegistry(model)
        AwayClubHomeWeekendsCount().apply(model, X, data, registry)

        solver, status = _solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )
        sun_count = _count_sunday_home_weeks(solver, X, data, 'Maitland')
        fri_count = _count_friday_home_weeks(solver, X, data, 'Maitland')
        # 3 PHL Fridays were forced.
        assert fri_count == 3, f'Expected 3 Friday-home weeks, got {fri_count}'
        # Atom guarantees 9 <= sun_count <= 10 regardless of solver's choice.
        assert 9 <= sun_count <= 10, (
            f'Expected Sunday count in [9, 10], got {sun_count}'
        )

    def test_scenario_6_all_phl_home_forced_friday_then_sundays_equal_nine(self):
        """GIVEN Maitland-shape + ALL 10 PHL home games FORCED to Friday at MP,
        WHEN solver runs,
        THEN feasible AND Sunday-home count = max(9, 0) = 9.

        Hand-computed:
          PHL contributes 0 home Sundays (all 10 forced to Friday).
          3rd-grade home Sundays = 9.
          Union W = 9 (only 3rd contributes).
          Atom bounds [9, 10]; W = 9 sits at the floor.
        """
        data = _build_multi_grade_fixture(
            num_weeks=22, phl_required=20, third_required=18,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _add_basic_hard_constraints(model, X, data)
        forced_vars = [
            v for k, v in X.items()
            if k[2] == 'PHL' and k[3] == 'Friday' and k[10] == MAITLAND
        ]
        # All 10 PHL home games to Friday.
        model.Add(sum(forced_vars) == 10)

        registry = HelperVarRegistry(model)
        AwayClubHomeWeekendsCount().apply(model, X, data, registry)

        solver, status = _solve(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible, got {solver.status_name(status)}'
        )
        sun_count = _count_sunday_home_weeks(solver, X, data, 'Maitland')
        fri_count = _count_friday_home_weeks(solver, X, data, 'Maitland')
        assert fri_count == 10, f'Expected 10 Friday-home weeks, got {fri_count}'
        # Per hand oracle: Sunday-home = 9 exactly (only 3rd grade contributes).
        assert sun_count == 9, (
            f'Expected Sunday count == 9, got {sun_count}'
        )


class TestAtomEdgeCases:
    """Defensive scenarios — empty map, raise-on-inversion."""

    def test_given_no_away_clubs_then_zero_constraints(self):
        """GIVEN empty home_field_map, WHEN applied, THEN no-op (returns 0).

        Hand-computed: home_field_map is the iteration surface; empty -> 0.
        """
        data = _build_multi_grade_fixture(
            num_weeks=4, phl_required=4, third_required=2,
        )
        data['home_field_map'] = {}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        registry = HelperVarRegistry(model)
        count = AwayClubHomeWeekendsCount().apply(model, X, data, registry)
        assert count == 0
