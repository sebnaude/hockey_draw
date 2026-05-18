"""Tests for spec-005 `ClubVsClubStackedAlignment` cluster.

Walks every DoD scenario from `docs/todo/spec-005-clubvsclub-stacking.md`:

  1. Stacking math: pin a pair to (PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1), solve,
     assert the per-week active grade set matches the peel-off layout.
  2. PHL Sunday budget: pin 2 FORCED PHL Fridays for the pair, assert the
     stacking math uses 2 (not 4) for PHL.
  3. Multi-team-per-club-per-grade: assert per-pair meeting count = matchups
     × per_matchup, not 1.
  4. Co-location: assert all stacked-week games for the pair are on the
     same field with contiguous day_slots (|slot_i - slot_j| == 1 across
     adjacent active grades).

Every oracle is hand-computed in the test/docstring; no mocks anywhere.
"""
from __future__ import annotations

import os
import sys

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from constraints.atoms import (
    ClubVsClubStackedCoLocation,
    ClubVsClubStackedWeekends,
)
from constraints.atoms._club_vs_club_stacked_shared import (
    STACK_PLAY_PREFIX,
    enumerate_club_pairs,
    pair_grade_sunday_meetings,
    per_pair_grade_matchup_counts,
    per_pair_grade_meeting_counts,
)
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.club_vs_club_stacked_fixture import (
    DEFAULT_GRADE_COUNTS,
    build_model_X,
    build_stacked_fixture,
    solve_with_timeout,
)


PAIR = ('Maitland', 'Norths')


def _registry(model):
    return HelperVarRegistry(model)


def _add_no_double_booking_fields(model, X):
    """Inline NoDoubleBookingFields: at most one game per (date, day_slot,
    field_name). Mirrors `original.py:NoDoubleBookingFieldsConstraint` for
    test fixtures that don't load the full constraint stack.

    Production runs always apply this in `critical_feasibility` stage; the
    stacking tests need it so two grades' games can't share a single field
    + slot (which would be a real-world impossibility).
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11:
            continue
        # (date, day_slot, field_name) is the field uniqueness bucket.
        buckets[(key[7], key[4], key[9])].append(var)
    for vars_list in buckets.values():
        if len(vars_list) > 1:
            model.Add(sum(vars_list) <= 1)


def _solve_with_stacking(data):
    """Apply both stacked atoms + field-uniqueness and solve.

    Field-uniqueness (NoDoubleBookingFields) is added so the co-location
    contiguity check is testing real solutions: without it the solver could
    stack 2+ games on the same (slot, field) which the production model
    would never do.

    Returns (status, X, solver).
    """
    model, X = build_model_X(data)
    reg = _registry(model)
    _add_no_double_booking_fields(model, X)
    ClubVsClubStackedWeekends().apply(model, X, data, reg)
    ClubVsClubStackedCoLocation().apply(model, X, data, reg)
    status, solver = solve_with_timeout(model)
    return status, X, solver


def _active_grades_per_week(data, X, solver):
    """Return `{week: set(grades_with_at_least_one_game_for_the_pair)}` from solution."""
    teams = data['teams']
    team_club = {t.name: t.club.name for t in teams}
    out: dict = {}
    for key, var in X.items():
        if len(key) < 11 or key[3] != 'Sunday':
            continue
        c1 = team_club.get(key[0])
        c2 = team_club.get(key[1])
        if not c1 or not c2 or {c1, c2} != set(PAIR):
            continue
        if solver.Value(var) == 1:
            out.setdefault(key[6], set()).add(key[2])
    return out


# ---------------------------------------------------------------------------
# Scenario 1 — Stacking math (DoD #7 bullet 1)
# ---------------------------------------------------------------------------


class TestScenarioOneStackingMath:
    """Given (Maitland, Norths) with PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1, no
    FORCED Fridays, When solved with both stacked atoms, Then the per-week
    active grade set follows the strict-nested-superset structure.

    Hand-computed oracle (peel-off layout):
        1 weekend: {PHL, 2nd, 3rd, 4th, 5th}   ← 5th plays + everything higher
        1 weekend: {PHL, 2nd, 3rd, 4th}        ← 3rd/4th play but not 5th
        1 weekend: {PHL, 2nd}                  ← gap of 1 (3rd, 4th drop together)
        1 weekend: {PHL}                       ← only PHL

    Per-grade counts: PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1 — match the input.
    """

    def _data(self):
        """5 slots/field so the all-5-grades week fits in one contiguous run."""
        return build_stacked_fixture(slots_per_field=5)

    def test_meeting_counts_match_spec(self):
        """Sanity: the fixture really does produce the spec's per-grade counts."""
        data = self._data()
        counts = per_pair_grade_meeting_counts(data, PAIR)
        # PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1.
        assert counts == DEFAULT_GRADE_COUNTS

    def test_solver_status_feasible(self):
        data = self._data()
        status, _X, _solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f'Expected feasible stacking on a clean fixture, got {status}'
        )

    def test_per_grade_counts_pinned_exactly(self):
        """Assert sum(play[g, w]) over w == grade_counts[g] in the solution."""
        data = self._data()
        status, X, solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        active = _active_grades_per_week(data, X, solver)
        per_grade: dict = {g: 0 for g in DEFAULT_GRADE_COUNTS}
        for grades_set in active.values():
            for g in grades_set:
                per_grade[g] += 1
        assert per_grade == DEFAULT_GRADE_COUNTS

    def test_nested_superset_structure(self):
        """Every active week's grade set is a prefix of the descending-count
        chain — never a "hole" where a lower-count grade plays without a
        higher-count one.
        """
        data = self._data()
        status, X, solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        active = _active_grades_per_week(data, X, solver)

        # Sort grades by descending count (PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1).
        # 3rd and 4th tie — implication-chain order is alphabetical for the
        # tie break, but for the "prefix" check we accept either {3rd} alone
        # or {4th} alone is wrong (both must coincide).
        sorted_grades = sorted(
            DEFAULT_GRADE_COUNTS.items(), key=lambda gc: (-gc[1], gc[0]),
        )
        sorted_names = [g for g, _ in sorted_grades]
        for week, grades_set in active.items():
            # The active set must be a prefix (under the desc-count chain).
            idx_of_last = max(sorted_names.index(g) for g in grades_set)
            expected_prefix = set(sorted_names[: idx_of_last + 1])
            assert grades_set == expected_prefix, (
                f'week {week}: active={grades_set}, expected prefix '
                f'{expected_prefix} (last grade idx={idx_of_last})'
            )

    def test_peel_off_layout_exactly(self):
        """The 4 weeks with games have grade-set sizes [5, 4, 2, 1] in some
        order — matching the hand-computed peel-off layout. (Sizes uniquely
        determine the structure because grades sort by count.)"""
        data = self._data()
        status, X, solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        active = _active_grades_per_week(data, X, solver)
        sizes = sorted([len(s) for s in active.values()], reverse=True)
        # 4 weeks of activity (matches max(grade_counts) = 4 = PHL count).
        # Sizes: 5 (with 5th), 4 (without 5th), 2 (only PHL+2nd), 1 (only PHL).
        assert sizes == [5, 4, 2, 1], (
            f'Peel-off layout sizes mismatch: got {sizes}, expected [5,4,2,1]'
        )


# ---------------------------------------------------------------------------
# Scenario 2 — PHL Sunday budget with FORCED Fridays (DoD #7 bullet 2)
# ---------------------------------------------------------------------------


class TestScenarioTwoForcedFridayBudget:
    """Given PHL=4 total meetings + 2 FORCED PHL Fridays for the (Maitland,
    Norths) pair, When solved, Then stacking uses PHL=2 (not 4) so the PHL
    Sunday count drops to 2, while other grades unchanged.

    Hand-computed oracle:
      - phl_forced_friday_meetings(data, 'Maitland', 'Norths') == 2
      - pair_grade_sunday_meetings(data, PAIR, 'PHL') == 4 - 2 == 2
      - In the solution, Sundays with PHL active for the pair == 2.
    """

    def _data_with_two_forced_fridays(self):
        # 5 slots/field — the all-5-grades stacked week needs 5 contiguous
        # slots on one field. Same as scenario 1.
        data = build_stacked_fixture(slots_per_field=5)
        # FORCED entry: 2 Maitland-vs-Norths PHL Friday games at Maitland Park.
        # Using `teams` matcher so both candidate Friday slots match. count=2
        # means the convenor pinned exactly 2 of these Friday games. The
        # atom subtracts that from the PHL Sunday budget; the model itself
        # doesn't enforce the FORCED entry (those constraints live in
        # `generate_X` / a separate validator), so the helper purely shifts
        # the stacking budget.
        data['forced_games'] = [
            {
                'grade': 'PHL',
                'day': 'Friday',
                'teams': ['Maitland PHL', 'Norths PHL'],
                'count': 2,
                'constraint': 'equal',
                'description': '2 Maitland-vs-Norths PHL Fridays (per-pair FORCED)',
            },
        ]
        return data

    def test_per_pair_friday_helper_counts_two(self):
        """Sanity: helper returns 2 for the (Maitland, Norths) pair."""
        from constraints.atoms._phl_forced_friday_helper import (
            phl_forced_friday_meetings,
        )
        data = self._data_with_two_forced_fridays()
        assert phl_forced_friday_meetings(data, 'Maitland', 'Norths') == 2

    def test_phl_sunday_budget_is_two_not_four(self):
        """The Sunday meeting count for PHL drops from 4 to 2."""
        data = self._data_with_two_forced_fridays()
        # Total per-pair PHL meetings: 4 (unchanged by FORCED).
        assert per_pair_grade_meeting_counts(data, PAIR)['PHL'] == 4
        # Sunday-available budget: 4 - 2 = 2.
        assert pair_grade_sunday_meetings(data, PAIR, 'PHL') == 2

    def test_solver_pins_two_phl_sunday_weeks_for_pair(self):
        """With Sunday budget==2, the stacking atom forces sum(play[PHL, w])
        over weeks == 2. Solve and count PHL Sundays in the solution."""
        data = self._data_with_two_forced_fridays()
        status, X, solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        active = _active_grades_per_week(data, X, solver)
        phl_sundays = sum(1 for s in active.values() if 'PHL' in s)
        assert phl_sundays == 2, (
            f'Expected exactly 2 PHL Sunday weekends after FORCED-Friday '
            f'subtraction, got {phl_sundays}. Active per week: {active}'
        )

    def test_other_grades_unchanged(self):
        """2nd, 3rd, 4th, 5th counts are NOT affected by FORCED PHL Fridays."""
        data = self._data_with_two_forced_fridays()
        for grade, expected in {'2nd': 3, '3rd': 2, '4th': 2, '5th': 1}.items():
            assert pair_grade_sunday_meetings(data, PAIR, grade) == expected


# ---------------------------------------------------------------------------
# Scenario 3 — Multiple teams per club per grade (DoD #7 bullet 3)
# ---------------------------------------------------------------------------


class TestScenarioThreeMultipleTeamsPerClubPerGrade:
    """Given Norths fields 2 PHL teams (Maitland still 1), When computing
    meeting counts, Then count = matchups * per_matchup, not 1.

    Hand-computed oracle:
      Teams in PHL: Maitland (1), Norths-1, Norths-2  → T = 3, odd.
      Matchups for (Maitland, Norths) pair = 2
          (Maitland vs Norths-1, Maitland vs Norths-2; Norths-1 vs Norths-2 is
           intra-club and doesn't count for the pair).
      per_matchup = R // T = num_rounds['PHL'] // 3.
      With fixture R=2 (computed: per_matchup=1 → R=per_matchup*T=3 → R=3? No,
      our fixture uses R=1*T=3 for odd T). So per_matchup = 3//3 = 1.
      meetings = 2 matchups * 1 per_matchup = 2.
    """

    def test_matchup_count_is_two_for_phl(self):
        data = build_stacked_fixture(
            extra_teams_in_grade={'PHL': {'Norths': 1}},
        )
        matchups = per_pair_grade_matchup_counts(data, PAIR)
        # Maitland (1 team) vs Norths (2 teams) = 2 distinct matchups.
        assert matchups['PHL'] == 2, (
            f'Expected 2 distinct PHL matchups for the pair, got {matchups}'
        )

    def test_meeting_count_aggregates_matchups(self):
        """meetings = matchups × per_matchup; not 1 (would be the buggy
        "club-pair coincides on a week" interpretation)."""
        data = build_stacked_fixture(
            extra_teams_in_grade={'PHL': {'Norths': 1}},
        )
        meetings = per_pair_grade_meeting_counts(data, PAIR)
        # T = 3 PHL teams (1 Maitland + 2 Norths). R per fixture = 1 * T = 3.
        # per_matchup = R // T = 3 // 3 = 1. meetings = 2 * 1 = 2.
        assert meetings['PHL'] == 2, (
            f'Expected meetings[PHL] = 2 matchups × 1 per_matchup = 2, '
            f'got {meetings["PHL"]}. num_rounds={data["num_rounds"]}'
        )

    def test_pair_grade_sunday_meetings_passes_through(self):
        """No FORCED Fridays here, so Sunday budget == total meetings."""
        data = build_stacked_fixture(
            extra_teams_in_grade={'PHL': {'Norths': 1}},
        )
        assert pair_grade_sunday_meetings(data, PAIR, 'PHL') == 2


# ---------------------------------------------------------------------------
# Scenario 4 — Co-location on stacked weekends (DoD #7 bullet 4)
# ---------------------------------------------------------------------------


class TestScenarioFourCoLocation:
    """Given a stacked weekend (≥ 2 grades for the pair active on the same
    Sunday), When inspecting the solution, Then all participating games are
    on the same field with contiguous day_slots (adjacent grade games on
    consecutive slots).

    Hand-computed oracle for the default fixture:
      The peel-off layout has weeks with grade-sets of size {1, 2, 4, 5}.
      The size-5 week (5 grades all play) needs 5 contiguous slots on one
      field. Our fixture has 4 slots per field → infeasible. So we use a
      smaller fixture with 3 grades active max.
    """

    def _smaller_fixture(self):
        """3-grade fixture: PHL=2, 2nd=2, 3rd=1. Peel-off → 1 week with
        {PHL,2nd,3rd}, 1 week with {PHL,2nd}. The first needs 3 contiguous
        slots — fits in 4-slot fields.
        """
        return build_stacked_fixture(
            grade_counts={'PHL': 2, '2nd': 2, '3rd': 1},
            num_weeks=4,
            slots_per_field=4,
        )

    def test_stacked_week_all_same_field(self):
        data = self._smaller_fixture()
        status, X, solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        teams = data['teams']
        team_club = {t.name: t.club.name for t in teams}
        # Gather (week -> [(grade, slot, field)]) for the pair.
        per_week: dict = {}
        for key, var in X.items():
            if len(key) < 11 or key[3] != 'Sunday':
                continue
            c1 = team_club.get(key[0])
            c2 = team_club.get(key[1])
            if not c1 or not c2 or {c1, c2} != set(PAIR):
                continue
            if solver.Value(var) != 1:
                continue
            per_week.setdefault(key[6], []).append((key[2], key[4], key[9]))

        # Any week with ≥ 2 grades must use only 1 field.
        for week, games in per_week.items():
            grades = {g for g, _s, _f in games}
            if len(grades) < 2:
                continue
            fields = {f for _g, _s, f in games}
            assert len(fields) == 1, (
                f'Stacked week {week} uses multiple fields {fields}; '
                f'expected 1. Games: {games}'
            )

    def test_stacked_week_contiguous_slots(self):
        data = self._smaller_fixture()
        status, X, solver = _solve_with_stacking(data)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        teams = data['teams']
        team_club = {t.name: t.club.name for t in teams}
        per_week: dict = {}
        for key, var in X.items():
            if len(key) < 11 or key[3] != 'Sunday':
                continue
            c1 = team_club.get(key[0])
            c2 = team_club.get(key[1])
            if not c1 or not c2 or {c1, c2} != set(PAIR):
                continue
            if solver.Value(var) != 1:
                continue
            per_week.setdefault(key[6], []).append((key[2], key[4], key[9]))

        # Each stacked week's slots must be contiguous (max - min + 1 == count).
        for week, games in per_week.items():
            grades = {g for g, _s, _f in games}
            if len(grades) < 2:
                continue
            slots = sorted([s for _g, s, _f in games])
            assert slots[-1] - slots[0] + 1 == len(slots), (
                f'Stacked week {week} slots {slots} are NOT contiguous '
                f'(span={slots[-1] - slots[0] + 1}, count={len(slots)}). '
                f'Games: {games}'
            )
            # Adjacent-grade games (sorted by slot) have |slot_i - slot_j| == 1.
            for i in range(len(slots) - 1):
                assert slots[i + 1] - slots[i] == 1, (
                    f'Stacked week {week}: gap between slots {slots[i]} and '
                    f'{slots[i + 1]} != 1; full slot list {slots}'
                )


# ---------------------------------------------------------------------------
# Standalone smoke + helper-coverage tests
# ---------------------------------------------------------------------------


class TestStackedAtomSmoke:
    """Quick smoke coverage: registry helpers + atom apply mechanics."""

    def test_enumerate_club_pairs_returns_maitland_norths(self):
        data = build_stacked_fixture()
        pairs = enumerate_club_pairs(data)
        assert PAIR in pairs

    def test_weekends_atom_registers_play_indicator(self):
        data = build_stacked_fixture()
        from ortools.sat.python import cp_model as _cp
        model = _cp.CpModel()
        from tests.atoms.club_vs_club_stacked_fixture import build_model_X as _bmx
        _, X = _bmx(data)
        # Build a fresh model with same X structure to register vars onto it.
        model, X = _bmx(data)
        reg = _registry(model)
        n = ClubVsClubStackedWeekends().apply(model, X, data, reg)
        assert n > 0
        # The play indicator for PHL at week 1 should be registered.
        ind = reg.get((STACK_PLAY_PREFIX, PAIR, 'PHL', 1))
        assert ind is not None

    def test_co_location_raises_if_weekends_not_run_first(self):
        """Programming-error guard: co-location must come AFTER stacked-weekends
        so the `play` indicators exist in the registry."""
        data = build_stacked_fixture(
            grade_counts={'PHL': 2, '2nd': 2}, num_weeks=3, slots_per_field=4,
        )
        from ortools.sat.python import cp_model as _cp
        from tests.atoms.club_vs_club_stacked_fixture import build_model_X as _bmx
        model, X = _bmx(data)
        reg = _registry(model)
        # Skip stacked-weekends; call co-location directly.
        with pytest.raises(RuntimeError, match='missing play indicator'):
            ClubVsClubStackedCoLocation().apply(model, X, data, reg)

    def test_weekends_raises_on_overbudget_grade(self):
        """If the Sunday budget exceeds available weeks, the atom raises
        ValueError up front rather than silently producing an infeasible
        model."""
        data = build_stacked_fixture(
            grade_counts={'PHL': 10}, num_weeks=3, slots_per_field=4,
        )
        from tests.atoms.club_vs_club_stacked_fixture import build_model_X as _bmx
        model, X = _bmx(data)
        reg = _registry(model)
        with pytest.raises(ValueError, match='exceeds available Sunday weeks'):
            ClubVsClubStackedWeekends().apply(model, X, data, reg)
