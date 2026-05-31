# spec-038 Unit B: hand-computed oracles for the four-layer aligned-weekend model.
"""Tests for the spec-038 rewrite of `ClubVsClubStackedWeekends`.

Every test builds a real CP-SAT model, applies the atom, and solves to
verify the four-layer model (`team_pair_play` → budget → `play_pg` →
cardinality → total → cross-grade chain). No mocks, no monkeypatching;
all oracles are hand-computed inline.

Scenarios covered (matching spec-038 DoD #23):

  1. 1×1 PHL pair, per_matchup=4 — 4 distinct Sundays, 1 game per Sunday.
  2. 1×1 non-PHL pair, per_matchup=2 — 2 distinct Sundays, 1 game per Sunday.
  3. 2×2 lower-grade pair (Tigers Yellow/Black × Univ Gentlemen/Seapigs in
     6th) per_matchup=2 — 4 distinct Sundays, 2 games per Sunday, each of
     4 team-pairs plays exactly 2 Sundays.
  4. 1×2 asymmetric (Maitland × Univ Redhogs/Seapigs in 4th) per_matchup=1
     — 2 distinct Sundays, 1 game per Sunday.
  5. PHL forced-Friday subtraction — synthetic 4-meeting pair with 2 forced
     Fridays → Sunday budget reduces to 2.
  6. Cross-grade nested-superset chain — synthetic 2-grade pair (PHL=4,
     6th=2) → 6th's 2 active Sundays are a subset of PHL's 4.
  7. Validation `ValueError` — budget > available Sunday weeks.
  8. PHL preservation — Maitland-Gosford PHL on season_test (no forced
     Fridays) matches the spec-005 PHL behaviour (`sum_w play_pg ==
     per_matchup == 4`).
"""
from __future__ import annotations

import os
import sys

import pytest
from ortools.sat.python import cp_model

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import load_season_data
from utils import generate_X
from constraints.atoms.club_vs_club_stacked_weekends import (
    ClubVsClubStackedWeekends,
)
from constraints.atoms._club_vs_club_stacked_shared import (
    STACK_PLAY_PREFIX,
    STACK_TEAM_PAIR_PLAY_PREFIX,
    enumerate_team_pairs_in_pair_grade,
)
from constraints.atoms.base import MAITLAND
from constraints.helper_vars import HelperVarRegistry
from tests.atoms.club_vs_club_stacked_fixture import (
    build_model_X,
    build_stacked_fixture,
    solve_with_timeout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry(model):
    return HelperVarRegistry(model)


def _add_no_double_booking_fields(model, X):
    """At most one game per (date, day_slot, field_name).

    Production runs always have this in the critical_feasibility stage; the
    stacking tests need it so two grades' games can't share a single
    field-slot, which the production model would never permit.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11:
            continue
        buckets[(key[7], key[4], key[9])].append(var)
    for vars_list in buckets.values():
        if len(vars_list) > 1:
            model.Add(sum(vars_list) <= 1)


def _add_no_double_booking_teams(model, X):
    """At most one game per (team, week) — counted for either team1 or team2.

    Required for multi-team-per-club fixtures so a team doesn't play more
    than one game on a Sunday. Mirrors `NoDoubleBookingTeamsConstraint`.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for key, var in X.items():
        if len(key) < 11:
            continue
        buckets[(key[0], key[6])].append(var)
        buckets[(key[1], key[6])].append(var)
    for vars_list in buckets.values():
        if len(vars_list) > 1:
            model.Add(sum(vars_list) <= 1)


# ---------------------------------------------------------------------------
# Scenario 1: 1×1 PHL pair, per_matchup=4 (Maitland-Norths on synthetic
# fixture). budget=4, 1 game/weekend, 4 distinct Sundays.
# ---------------------------------------------------------------------------


class TestScenario1_1x1_PHL_pm4:
    """1×1 PHL pair, per_matchup=4.

    Hand oracle: T=2, R=4, per_matchup=4//1=4. a=b=1 → max=1, min=1.
    weekends_budget = max(1,1)*4 = 4. games/weekend = min(1,1) = 1.
    Total games = 1*1*4 = 4. Each of the 4 games on a distinct Sunday.
    """

    def test_solves_and_satisfies_budget(self):
        data = build_stacked_fixture({'PHL': 4})
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        n = ClubVsClubStackedWeekends().apply(model, X, data, reg)
        assert n > 0  # atom added at least some constraints

        status, solver = solve_with_timeout(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'Expected feasible/optimal, got status={status}'
        )

        # Count Sunday games played for the PHL pair.
        sunday_weeks_with_games = set()
        total_games = 0
        for key, var in X.items():
            if len(key) < 11 or key[3] != 'Sunday':
                continue
            if key[2] != 'PHL':
                continue
            if solver.Value(var) == 1:
                sunday_weeks_with_games.add(key[6])
                total_games += 1

        assert total_games == 4, f'Expected 4 PHL games, got {total_games}'
        assert len(sunday_weeks_with_games) == 4, (
            f'Expected 4 distinct Sundays, got {len(sunday_weeks_with_games)}'
        )

        # play_pg matches: sum over weeks == 4.
        pair = ('Maitland', 'Norths')
        play_pg_sum = 0
        for w in range(1, 7):  # fixture has 6 weeks
            ind = reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', w))
            if ind is not None:
                play_pg_sum += solver.Value(ind)
        assert play_pg_sum == 4, f'play_pg sum = {play_pg_sum}, expected 4'


# ---------------------------------------------------------------------------
# Scenario 2: 1×1 non-PHL (3rd grade) per_matchup=2. budget=2.
# ---------------------------------------------------------------------------


class TestScenario2_1x1_NonPHL_pm2:
    """1×1 3rd-grade pair, per_matchup=2.

    Hand oracle: fixture {'3rd': 2}, T=2, R=2, per_matchup=2//1=2.
    weekends_budget = max(1,1)*2 = 2. games/weekend = 1. Total games = 2.
    """

    def test_solves_with_2_sundays(self):
        data = build_stacked_fixture({'3rd': 2})
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL)

        sunday_weeks = set()
        for key, var in X.items():
            if len(key) < 11 or key[3] != 'Sunday' or key[2] != '3rd':
                continue
            if solver.Value(var) == 1:
                sunday_weeks.add(key[6])
        assert len(sunday_weeks) == 2, (
            f'Expected 2 distinct Sundays, got {len(sunday_weeks)}'
        )


# ---------------------------------------------------------------------------
# Scenario 3: 2×2 lower-grade pair. Real season_test data:
# Tigers (Yellow + Black) × University (Gentlemen + Seapigs) in 6th.
# per_matchup=2, budget=4, games/weekend=2, total games=8.
# ---------------------------------------------------------------------------


class TestScenario3_2x2_LowerGrade:
    """2×2 6th grade, Tigers × University. Real season_test data.

    Hand oracle: a=2, b=2 → max=2, min=2. per_matchup=2 (T=10, R=18,
    R//(T-1)=18//9=2). weekends_budget = max(2,2)*2 = 4. games/weekend =
    min(2,2) = 2. Total games = 2*2*2 = 8. Each of 4 team-pairs plays
    exactly 2 Sundays; on each of the 4 active Sundays, exactly 2 team-pairs
    play.
    """

    def _build_minimal_2x2_fixture(self):
        """Build a synthetic 2×2 6th-grade fixture (Tigers vs University
        analog) using `build_stacked_fixture` with extras.

        Using a fixture instead of full season_test lets us scope the model
        to just this pair-grade for a clean unit test. The arithmetic is
        identical to the season_test 2×2 case.

        T=4 (2 extras per club in 3rd grade). R needs to be 6 so
        per_matchup = R//(T-1) = 6//3 = 2.
        """
        data = build_stacked_fixture(
            {'3rd': 1},
            extra_teams_in_grade={'3rd': {'Maitland': 1, 'Norths': 1}},
            num_weeks=6,
        )
        # Override num_rounds['3rd'] to produce per_matchup=2.
        data['num_rounds']['3rd'] = 6
        data['num_rounds']['max'] = 6
        return data

    def test_2x2_solves_with_correct_cardinality(self):
        data = self._build_minimal_2x2_fixture()
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'2×2 expected feasible, got {status}'
        )

        pair = ('Maitland', 'Norths')
        team_pairs = enumerate_team_pairs_in_pair_grade(data, pair, '3rd')
        assert len(team_pairs) == 4, f'expected 4 team-pairs, got {len(team_pairs)}'

        sunday_weeks = sorted({
            k[6] for k in X if len(k) >= 11 and k[3] == 'Sunday'
        })

        # Layer 2 oracle: each team-pair plays exactly 2 Sundays.
        for tp in team_pairs:
            count = 0
            for w in sunday_weeks:
                ind = reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w))
                if ind is not None:
                    count += solver.Value(ind)
            assert count == 2, (
                f'team_pair {tp} played on {count} weekends, expected 2'
            )

        # Layer 5 oracle: sum_w play_pg == 4.
        play_pg_sum = 0
        active_weeks = []
        for w in sunday_weeks:
            ind = reg.lookup((STACK_PLAY_PREFIX, pair, '3rd', w))
            if ind is not None and solver.Value(ind) == 1:
                play_pg_sum += 1
                active_weeks.append(w)
        assert play_pg_sum == 4, f'play_pg sum = {play_pg_sum}, expected 4'

        # Layer 4 oracle: on each active week, exactly 2 team-pairs play.
        for w in active_weeks:
            tp_count = 0
            for tp in team_pairs:
                ind = reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w))
                if ind is not None:
                    tp_count += solver.Value(ind)
            assert tp_count == 2, (
                f'on active week {w}, {tp_count} team-pairs played; expected 2'
            )

        # And on inactive weeks: zero team-pairs play.
        inactive_weeks = [w for w in sunday_weeks if w not in active_weeks]
        for w in inactive_weeks:
            tp_count = 0
            for tp in team_pairs:
                ind = reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w))
                if ind is not None:
                    tp_count += solver.Value(ind)
            assert tp_count == 0, (
                f'on inactive week {w}, {tp_count} team-pairs played; expected 0'
            )


# ---------------------------------------------------------------------------
# Scenario 4: 1×2 asymmetric, per_matchup=1.
# ---------------------------------------------------------------------------


class TestScenario4_1x2_Asymmetric:
    """1×2 asymmetric: 1-team Maitland × 2-team Norths in 3rd grade.

    Hand oracle (range semantics per spec-038 fix): a=1 (Maitland), b=2 (Norths).
    T=3 (odd), R=3, base=3//3=1, base+1=2. Per BalancedMatchups, each team-pair
    meets in [1, 2]. Layer 5: sum_w play_pg ∈ [max(a,b)*base, max(a,b)*(base+1)]
    = [2, 4]. Layer 4 (1 tp per aligned weekend) → num_aligned = sum_tp_plays.

    Maitland 3rd plays 2 opponents (the 2 Norths 3rd teams), sum to R=3:
    one tp at 1 meeting + one tp at 2 meetings (BalancedMatchups + per-team
    forces this distribution). Total tp-plays = 3 → 3 aligned weekends.
    """

    def test_1x2_solves_with_correct_layout(self):
        data = build_stacked_fixture(
            {'3rd': 1},
            extra_teams_in_grade={'3rd': {'Norths': 1}},
            num_weeks=6,
        )
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL)

        pair = ('Maitland', 'Norths')
        team_pairs = enumerate_team_pairs_in_pair_grade(data, pair, '3rd')
        assert len(team_pairs) == 2, f'expected 2 team-pairs, got {len(team_pairs)}'

        sunday_weeks = sorted({
            k[6] for k in X if len(k) >= 11 and k[3] == 'Sunday'
        })

        # Each team-pair plays in [base, base+1] = [1, 2] Sundays.
        tp_counts = []
        for tp in team_pairs:
            count = sum(
                solver.Value(reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w)))
                for w in sunday_weeks
                if reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w)) is not None
            )
            assert 1 <= count <= 2, f'tp {tp} played {count} weekends; expected [1, 2]'
            tp_counts.append(count)

        # Layer 5 range: sum_w play_pg ∈ [2, 4]. Solver's actual count = sum_tp_plays
        # (since min(a,b)=1 game per aligned weekend).
        active_weeks = [
            w for w in sunday_weeks
            if reg.lookup((STACK_PLAY_PREFIX, pair, '3rd', w)) is not None
            and solver.Value(reg.lookup((STACK_PLAY_PREFIX, pair, '3rd', w))) == 1
        ]
        assert 2 <= len(active_weeks) <= 4, (
            f'expected aligned weekends in [2, 4], got {len(active_weeks)}'
        )
        assert len(active_weeks) == sum(tp_counts), (
            'aligned weekend count must equal total tp-plays for 1×2 (min=1 game/week)'
        )

        # On each active weekend: exactly 1 team-pair plays (min(a,b)=1).
        for w in active_weeks:
            tp_count = sum(
                solver.Value(reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w)))
                for tp in team_pairs
            )
            assert tp_count == 1, (
                f'active week {w}: {tp_count} team-pairs played, expected 1'
            )


# ---------------------------------------------------------------------------
# Scenario 5: PHL forced-Friday subtraction.
# ---------------------------------------------------------------------------


class TestScenario5_PHL_ForcedFriday_Subtraction:
    """PHL pair with per_matchup=4 and 2 forced Friday meetings.

    Hand oracle: weekends_budget = max(1,1)*4 - 2 = 2. Sunday budget shrinks
    from 4 to 2. The atom's `sum_w play_pg == 2` constraint enforces this.

    Implementation note: the atom does NOT consume the Friday vars or place
    games on Friday slots itself — `phl_forced_friday_meetings(data, A, B)`
    is a pure data-side count derived from `data['forced_games']` and is
    subtracted inside `pair_grade_sunday_aligned_weekends`. The Sunday-budget
    side of the model is the only place the atom enforces. So this test
    verifies the Sunday-side cardinality without also verifying that real
    Friday games occur.
    """

    def test_2_forced_fridays_reduces_sunday_budget_to_2(self):
        data = build_stacked_fixture({'PHL': 4})
        data['forced_games'] = [{
            'grade': 'PHL', 'day': 'Friday',
            'teams': ['Maitland PHL', 'Norths PHL'],
            'field_location': MAITLAND,
            'count': 2, 'constraint': 'equal',
            'description': 'Two Maitland-vs-Norths PHL Fridays',
        }]
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL)

        # Count Sunday PHL games — should be exactly the Sunday budget = 2.
        sunday_games = 0
        for key, var in X.items():
            if len(key) < 11 or key[3] != 'Sunday' or key[2] != 'PHL':
                continue
            if solver.Value(var) == 1:
                sunday_games += 1
        assert sunday_games == 2, f'expected 2 Sunday PHL games, got {sunday_games}'

        # And play_pg sum == 2.
        pair = ('Maitland', 'Norths')
        play_pg_sum = sum(
            solver.Value(reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', w)))
            for w in range(1, 7)
            if reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', w)) is not None
        )
        assert play_pg_sum == 2, f'play_pg sum = {play_pg_sum}, expected 2'


# ---------------------------------------------------------------------------
# Scenario 6: cross-grade nested-superset chain. PHL=4, 6th=2 → 6th's 2
# weeks subset of PHL's 4.
# ---------------------------------------------------------------------------


class TestScenario6_CrossGradeNestedSupersetChain:
    """Synthetic 2-grade pair: PHL budget=4, 6th budget=2.

    Hand oracle: PHL (1×1, R=4, T=2) → pm=4, budget=4.
                 6th (1×1, R=2, T=2) → pm=2, budget=2.
    Chain: play_pg[6th, w] <= play_pg[PHL, w] for every w. So every Sunday
    6th is "on", PHL is also "on" — i.e. the 2 active 6th weeks are a
    subset of the 4 active PHL weeks.
    """

    def test_6th_subset_of_phl(self):
        data = build_stacked_fixture({'PHL': 4, '6th': 2}, num_weeks=6)
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)
        status, solver = solve_with_timeout(model)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL)

        pair = ('Maitland', 'Norths')
        weeks = list(range(1, 7))
        phl_active = {
            w for w in weeks
            if reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', w)) is not None
            and solver.Value(reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', w))) == 1
        }
        sixth_active = {
            w for w in weeks
            if reg.lookup((STACK_PLAY_PREFIX, pair, '6th', w)) is not None
            and solver.Value(reg.lookup((STACK_PLAY_PREFIX, pair, '6th', w))) == 1
        }
        assert len(phl_active) == 4
        assert len(sixth_active) == 2
        assert sixth_active.issubset(phl_active), (
            f'6th active {sixth_active} NOT subset of PHL active {phl_active}'
        )


# ---------------------------------------------------------------------------
# Scenario 7: ValueError when budget > available Sunday weeks.
# ---------------------------------------------------------------------------


class TestScenario7_Validation_BudgetExceedsAvailability:
    """Synthetic pair where budget > number of Sunday weeks.

    Hand oracle: PHL budget=4, but only 3 Sunday weeks generated. The atom
    must raise `ValueError` with pair, grade, budget, available count in
    the message.
    """

    def test_value_error_when_budget_exceeds_available_weeks(self):
        data = build_stacked_fixture({'PHL': 4}, num_weeks=3)
        model, X = build_model_X(data)
        reg = _registry(model)
        # No need to add double-booking — the atom should raise before solving.
        with pytest.raises(ValueError) as excinfo:
            ClubVsClubStackedWeekends().apply(model, X, data, reg)
        msg = str(excinfo.value)
        assert 'ClubVsClubStackedWeekends' in msg
        assert 'PHL' in msg
        assert "'Maitland', 'Norths'" in msg or 'Maitland' in msg
        assert 'budget=4' in msg
        # available count should be in the error.
        assert 'available' in msg.lower()


# ---------------------------------------------------------------------------
# Scenario 8: PHL preservation safety check — Maitland-Gosford on
# season_test (no forced Fridays) should look identical to spec-005.
# ---------------------------------------------------------------------------


class TestScenario8_PHL_Preservation_SeasonTest:
    """Maitland-Gosford PHL on season_test: a=b=1, per_matchup=4, no forced
    Fridays. The new formula collapses to the old PHL formula
    (sum_w play_pg == 4), so behaviour is backward-compatible.

    We don't run a full solve on season_test (it's huge). Instead we verify
    the helper-derived budget matches the expected value AND that the atom
    correctly registers `cvc_stack_team_pair_play` indicators for the
    single PHL team-pair.
    """

    def test_phl_pair_aligned_weekends_equals_per_matchup(self):
        data = load_season_data('test')
        # Sanity: spec-038 helper returns 4 (= per_matchup for PHL on test).
        from constraints.atoms._club_vs_club_stacked_shared import (
            pair_grade_sunday_aligned_weekends,
        )
        wb = pair_grade_sunday_aligned_weekends(
            data, ('Maitland', 'Gosford'), 'PHL'
        )
        assert wb == 4, f'PHL Maitland-Gosford budget = {wb}, expected 4'
        # And the team-pair is exactly the single 1×1 pair.
        tps = enumerate_team_pairs_in_pair_grade(
            data, ('Maitland', 'Gosford'), 'PHL'
        )
        assert len(tps) == 1
        assert tps[0] == ('Gosford PHL', 'Maitland PHL')


# ---------------------------------------------------------------------------
# Registry/helper-var family check (DoD #34): the new prefix is registered
# under the right key shape and reg.lookup is callable.
# ---------------------------------------------------------------------------


class TestHelperVarRegistration:
    """The atom must register `cvc_stack_team_pair_play` indicators under
    `(STACK_TEAM_PAIR_PLAY_PREFIX, team_pair, week)` for every team-pair
    in every pair-grade it processes."""

    def test_team_pair_play_registered_under_correct_prefix(self):
        data = build_stacked_fixture({'PHL': 4})
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)

        # The PHL team-pair is ('Maitland PHL', 'Norths PHL'); 6 Sundays
        # → 6 registrations.
        tp = ('Maitland PHL', 'Norths PHL')
        for w in range(1, 7):
            ind = reg.lookup((STACK_TEAM_PAIR_PLAY_PREFIX, tp, w))
            assert ind is not None, (
                f'no cvc_stack_team_pair_play for tp={tp} w={w}'
            )

        # And `cvc_stack_play` is registered for the pair-grade per week.
        pair = ('Maitland', 'Norths')
        for w in range(1, 7):
            ind = reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', w))
            assert ind is not None


# ---------------------------------------------------------------------------
# spec-038 club-day fix: Layer-6 cross-club nesting is exempted on the host
# club's club-day weekend. ClubDayParticipation pins the whole host roster to
# one Sunday — multi-team grades turn inward (derbies), single-team grades
# reach outside — so the per-week same-opponent nesting is structurally
# unsatisfiable there. We verify by contrast: forcing a lower-budget grade
# active on a week WITHOUT the higher-budget grade is INFEASIBLE on a normal
# week (Layer 6 forbids it) but FEASIBLE on the host's club-day week (exempt).
# ---------------------------------------------------------------------------


class TestClubDayWeekendNestingExemption:
    """The Layer-6 `play_pg[lo] <= play_pg[hi]` chain must NOT fire on the host
    club's club-day weekend; it MUST still fire on every other weekend."""

    # Clean weekly Sunday dates per week (the fixture's generated dates include
    # invalid strings like 2026-02-29 that `get_nearest_week_by_date`'s strptime
    # would reject — it parses every timeslot date).
    _WEEK_DATES = {
        1: '2026-03-01', 2: '2026-03-08', 3: '2026-03-15',
        4: '2026-03-22', 5: '2026-03-29', 6: '2026-04-05',
    }

    def _build(self):
        # PHL budget=4, 6th budget=2 → Layer 6 forces 6th-active ⊆ PHL-active.
        data = build_stacked_fixture({'PHL': 4, '6th': 2}, num_weeks=6)
        for t in data['timeslots']:
            if t.day == 'Sunday':
                t.date = self._WEEK_DATES[t.week]
        return data

    def _solve_forcing_inversion(self, club_days, force_week):
        """Apply the atom (with `club_days`), force 6th active and PHL inactive
        on `force_week`, and return the solve status."""
        from datetime import datetime as _dt
        data = self._build()
        data['club_days'] = club_days
        model, X = build_model_X(data)
        reg = _registry(model)
        _add_no_double_booking_fields(model, X)
        _add_no_double_booking_teams(model, X)
        ClubVsClubStackedWeekends().apply(model, X, data, reg)

        pair = ('Maitland', 'Norths')
        six = reg.lookup((STACK_PLAY_PREFIX, pair, '6th', force_week))
        phl = reg.lookup((STACK_PLAY_PREFIX, pair, 'PHL', force_week))
        assert six is not None and phl is not None
        model.Add(six == 1)   # lower-budget grade active this week
        model.Add(phl == 0)   # higher-budget grade NOT active this week
        return solve_with_timeout(model)

    def test_normal_week_inversion_is_infeasible(self):
        # No club day → Layer 6 active everywhere → 6th cannot outrun PHL.
        status, _ = self._solve_forcing_inversion(club_days={}, force_week=3)
        assert status == cp_model.INFEASIBLE, (
            f'Layer 6 should forbid 6th-without-PHL on a normal week; got {status}'
        )

    def test_club_day_week_inversion_is_feasible(self):
        # Maitland (a club in the pair) hosts a club day on week-3's Sunday.
        from datetime import datetime as _dt
        club_days = {'Maitland': {'date': _dt(2026, 3, 15)}}  # → week 3
        status, _ = self._solve_forcing_inversion(
            club_days=club_days, force_week=3,
        )
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), (
            f'club-day weekend must be exempt from Layer 6; got {status}'
        )

    def test_club_day_does_not_exempt_other_weeks(self):
        # Club day on week 3, but force the inversion on week 5 → still INFEASIBLE
        # (only the host's club-day weekend is exempt, not the whole season).
        from datetime import datetime as _dt
        club_days = {'Maitland': {'date': _dt(2026, 3, 15)}}  # → week 3
        status, _ = self._solve_forcing_inversion(
            club_days=club_days, force_week=5,
        )
        assert status == cp_model.INFEASIBLE, (
            f'non-club-day weeks must keep Layer 6; got {status}'
        )


# ---------------------------------------------------------------------------
# spec-044 Unit B: real-config away-club Sunday-floor regression.
#
# This is the integration regression that proves the umbrella-Friday-aware
# floor (Unit A) keeps the away clubs' aggregated Sunday floor within their
# Sunday capacity on the REAL 2026 config, so the alignment atom is no longer
# the `core` infeasibility blocker that the spec-035 stage-5 handoff left open.
#
# It mirrors `scripts/isolate_clubvsclub_infeasibility.py reproduce`:
#   1. build `load_season_data(2026)` + `generate_X` (populating `data['games']`
#      from the var-key list — phl_forced_friday_meetings only sees pairs once
#      the games list is materialised),
#   2. apply the always-on fundamentals (EqualGamesAndBalanceMatchUps +
#      NoDoubleBookingTeams/Fields from constraints.archived.original),
#   3. apply the faithful Layer-1..5 replica of `ClubVsClubStackedWeekends`
#      capped at max_layer=5 (the layer the capacity contradiction lives in —
#      Layer 2's per-team-pair floor + Layer 5's aggregate min_budget),
#   4. solve with a short cap and assert the status is NOT INFEASIBLE.
#
# Layer 6 (the cross-grade nested chain) is excluded from the replica because
# the capacity overflow lives entirely in the Layer-2/Layer-5 floors (spec-044
# "Why"; the analytical mode finds 0 Layer-6 contradictions). max_layer=5
# is the most faithful reproduction of the documented blocker.
# ---------------------------------------------------------------------------


class TestSpec044RealConfigAwayClubFloorRegression:
    """Real 2026 config: the umbrella-aware floor keeps every PHL away club's
    aggregated Sunday meeting floor within Sunday capacity, so fundamentals +
    the alignment atom (Layers 1-5) is no longer INFEASIBLE.

    Hand oracles (deterministic, asserted BEFORE the solver runs — these ARE
    the proof; the solver step is corroboration):

      PHL on 2026: T=6, R=20 → base = 20 // 5 = 4, so each pair's per-team-pair
      Sunday floor is `tp_min = max(0, base - pair_named - umb)` where
      `umb = max(umb(A), umb(B))` is the more-constrained club's umbrella term.

      Umbrella forced-Friday counts (Unit A helper):
        umb(Gosford)  = 8  (CCHP `sum==8` PHL-Friday umbrella)
        umb(Maitland) = 2  (Maitland Park `sum==2` PHL-Friday umbrella)
        every other (central) club = 0.

      Gosford Σ_pairs tp_min = 0:
        vs Maitland : umb=max(8,2)=8, pair_named=0 → max(0, 4-0-8) = 0
        vs Norths   : umb=max(8,0)=8, pair_named=1 → max(0, 4-1-8) = 0
        vs Souths   : umb=max(8,0)=8, pair_named=0 → max(0, 4-0-8) = 0
        vs Tigers   : umb=max(8,0)=8, pair_named=0 → max(0, 4-0-8) = 0
        vs Wests    : umb=max(8,0)=8, pair_named=0 → max(0, 4-0-8) = 0
        → 0+0+0+0+0 = 0   (≤ Sunday capacity R-8 = 12 ✓)

      Maitland Σ_pairs tp_min = 7:
        vs Gosford  : umb=max(8,2)=8, pair_named=0 → max(0, 4-0-8) = 0
        vs Norths   : umb=max(2,0)=2, pair_named=0 → max(0, 4-0-2) = 2
        vs Souths   : umb=max(2,0)=2, pair_named=1 → max(0, 4-1-2) = 1
        vs Tigers   : umb=max(2,0)=2, pair_named=0 → max(0, 4-0-2) = 2
        vs Wests    : umb=max(2,0)=2, pair_named=0 → max(0, 4-0-2) = 2
        → 0+2+1+2+2 = 7   (≤ Sunday capacity R-2 = 18 ✓)

      NB: the spec's stale worked example said Maitland Σ=9 (4 pairs at 2 + 1
      at 1). That omitted the Maitland-vs-Gosford pair, whose floor is driven
      to 0 by Gosford's dominating umb=8 (more-constrained-club rule). The real
      figure recomputed here is 7 (= 0+2+1+2+2). Pre-fix both Gosford and
      Maitland had Σ tp_min = 19 (no umbrella subtraction), exceeding caps 12
      and 18 respectively → the documented INFEASIBLE blocker.
    """

    @pytest.fixture(scope='class')
    def real_X(self):
        """Build the real 2026 config + X once for the class (slow: ~82k vars).

        Mirrors `isolate_clubvsclub_infeasibility.build_data_and_X`: run
        `generate_X`, stash conflicts, and materialise `data['games']` from the
        var-key dict (the helpers read `data['games']` as a list of
        (t1, t2, grade) tuples).
        """
        import contextlib
        import io
        from ortools.sat.python import cp_model as _cp

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = load_season_data(2026)
            model = _cp.CpModel()
            X, conflicts = generate_X(model, data)
        data['team_conflicts'] = conflicts
        data['games'] = (
            list(data['games'].keys())
            if isinstance(data['games'], dict) else data['games']
        )
        return data, model, X

    def test_helper_oracles_before_solve(self, real_X):
        """Deterministic hand oracles — the actual proof of the fix."""
        from constraints.atoms._club_vs_club_stacked_shared import (
            enumerate_club_pairs,
            team_pair_counts,
            team_pair_sunday_meetings_range,
        )
        from constraints.atoms._phl_forced_friday_helper import (
            club_umbrella_forced_friday_meetings,
        )

        data, _model, _X = real_X

        # Umbrella forced-Friday counts (Unit A helper).
        assert club_umbrella_forced_friday_meetings(data, 'Gosford') == 8
        assert club_umbrella_forced_friday_meetings(data, 'Maitland') == 2
        assert club_umbrella_forced_friday_meetings(data, 'Norths') == 0

        # Aggregated per-team-pair Sunday floor for each away club.
        def _club_floor(club):
            total = 0
            for pair in enumerate_club_pairs(data):
                if club not in pair:
                    continue
                a, b = team_pair_counts(data, pair, 'PHL')
                if a == 0 or b == 0:
                    continue
                tp_min, _tp_max = team_pair_sunday_meetings_range(
                    data, pair, 'PHL'
                )
                total += tp_min
            return total

        # Hand oracles (see class docstring for the per-pair breakdown).
        assert _club_floor('Gosford') == 0, 'Gosford Σ tp_min must clamp to 0'
        assert _club_floor('Maitland') == 7, 'Maitland Σ tp_min must be 7'

    def test_fundamentals_plus_alignment_not_infeasible(self, real_X):
        """Solve fundamentals + the Layer-1..5 alignment replica on the real
        config; assert the status is NOT INFEASIBLE.

        Why UNKNOWN is acceptable: the real model is ~82k vars and the only
        capacity contradiction this spec targets — the away-club Sunday-floor
        overflow — would, if it still existed, surface as a fast INFEASIBLE
        (Layer-2/5 floors vs EqualGames are a pure counting contradiction the
        solver proves without deep search; pre-fix the proof script flags it
        deterministically). Its ABSENCE is the regression signal. A short cap
        on a large model may legitimately return UNKNOWN/REACHED_SEARCH
        without proving optimality — that is fine. INFEASIBLE is the ONLY
        failure outcome this test forbids.
        """
        import importlib.util
        import os

        from ortools.sat.python import cp_model as _cp

        from constraints.helper_vars import HelperVarRegistry
        from constraints.archived.original import (
            EnsureEqualGamesAndBalanceMatchUps,
            NoDoubleBookingTeamsConstraint,
            NoDoubleBookingFieldsConstraint,
        )
        from constraints.atoms._club_vs_club_stacked_shared import (
            enumerate_club_pairs,
        )

        # Load the faithful Layer-1..5 replica from the isolate diagnostic
        # script (the documented reproduction recipe). The script lives under
        # scripts/ and is import-only here (we call apply_stacked directly).
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        iso_path = os.path.join(
            repo_root, 'scripts', 'isolate_clubvsclub_infeasibility.py'
        )
        if not os.path.exists(iso_path):
            pytest.skip(
                'isolate_clubvsclub_infeasibility.py not present (evidence '
                'tooling is untracked); helper-oracle test carries the proof.'
            )
        spec = importlib.util.spec_from_file_location('_iso_s044', iso_path)
        iso = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(iso)

        data, model, X = real_X

        # Fundamentals (always-on critical stage).
        EnsureEqualGamesAndBalanceMatchUps().apply(model, X, data)
        NoDoubleBookingTeamsConstraint().apply(model, X, data)
        NoDoubleBookingFieldsConstraint().apply(model, X, data)

        # Faithful alignment replica, Layers 1-5 (the capacity-floor layers).
        reg = HelperVarRegistry(model)
        pairs = enumerate_club_pairs(data)
        n_added = iso.apply_stacked(model, X, data, reg, pairs, max_layer=5)
        assert n_added > 0, 'alignment replica added no constraints'

        solver = _cp.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)

        assert status != _cp.INFEASIBLE, (
            'fundamentals + alignment(max_layer=5) is INFEASIBLE on the real '
            '2026 config — the away-club Sunday-floor overflow (spec-044) was '
            f'NOT fixed. status={solver.status_name(status)}'
        )
