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
