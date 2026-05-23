# spec-020: GWT tests for the generic `PreferredGames` soft atom.
"""Given/When/Then tests for `PreferredGames` (constraints/atoms/preferred_games.py).

`PreferredGames` is the soft, weighted analogue of the whole FORCED_GAMES
grammar: it reuses the shared scope/team parser and adds a penalty-on-deviation
from `count` per `constraint` type, into a single shared
`data['penalties']['preferred_games']` bucket.

All tests use real CP-SAT models on the small `phl_data` fixture (conftest.py).
No mocks. Each penalty value is read back from a solved model and compared to a
hand-computed oracle.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from constraints.atoms import PreferredGames
from constraints.helper_vars import HelperVarRegistry

from tests.atoms.conftest import build_model_X, solve_with_timeout


def _registry(model):
    reg = HelperVarRegistry(model)
    return reg


def _phl_vars_on_date(X, date_str):
    """All PHL vars on a given date (the candidate set the atom buckets)."""
    return [v for k, v in X.items() if k[2] == 'PHL' and k[7] == date_str]


def _solve_penalty(model, pen_var):
    """Solve and return the integer value the solver assigned to pen_var."""
    status, solver = solve_with_timeout(model, seconds=5.0)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"status={status}"
    return solver.Value(pen_var)


# ----------------------------------------------------------------------
# equal — two-sided |sum − N|
# ----------------------------------------------------------------------

class TestEqualConstraint:
    def test_no_entries_is_no_op(self, phl_data):
        # Given: no preferred_games configured.
        phl_data['preferred_games'] = []
        model, X = build_model_X(phl_data, allow_2nd=False)
        # When/Then: the atom returns 0 and creates no bucket.
        n = PreferredGames().apply(model, X, phl_data, _registry(model))
        assert n == 0
        assert 'preferred_games' not in phl_data.get('penalties', {})

    def test_equal_count1_pins_sum_zero_one_two(self, phl_data):
        """Scenario: equal/count=1 on a PHL date → penalty = |sum − 1|.

        Hand oracle: deviation for sum∈{0,1,2} is {1,0,1} (two-sided).
        We pin sum to each value and read the penalty back.
        """
        date = '2026-03-22'
        # Build three independent models so pinning sums doesn't interfere.
        for pinned_sum, expected_penalty in [(0, 1), (1, 0), (2, 1)]:
            phl_data['preferred_games'] = [
                {'grade': 'PHL', 'date': date, 'constraint': 'equal',
                 'count': 1, 'description': 'eq1'}
            ]
            phl_data['penalties'] = {}
            model, X = build_model_X(phl_data, allow_2nd=False)
            cand = _phl_vars_on_date(X, date)
            assert len(cand) >= 3, 'fixture must have >=3 PHL vars on date'
            n = PreferredGames().apply(model, X, phl_data, _registry(model))
            assert n == 1
            # Pin sum(candidates) to the scenario value.
            model.Add(sum(cand) == pinned_sum)
            pen = phl_data['penalties']['preferred_games']['penalties'][0]
            assert _solve_penalty(model, pen) == expected_penalty

    def test_bucket_weight_is_default(self, phl_data):
        # Given: penalty_weights has no preferred_games key.
        phl_data['penalty_weights'] = {}
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': '2026-03-22', 'constraint': 'equal', 'count': 1}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        PreferredGames().apply(model, X, phl_data, _registry(model))
        # Then: bucket carries the atom default weight 10_000.
        assert phl_data['penalties']['preferred_games']['weight'] == 10_000


# ----------------------------------------------------------------------
# lesse — max(0, sum − N)
# ----------------------------------------------------------------------

class TestLesseConstraint:
    def test_lesse_count2(self, phl_data):
        """Scenario: lesse/count=2 → penalty = max(0, sum − 2).

        Hand oracle: sum=3 → 1; sum=2 → 0.
        """
        date = '2026-03-22'
        for pinned_sum, expected in [(3, 1), (2, 0)]:
            phl_data['preferred_games'] = [
                {'grade': 'PHL', 'date': date, 'constraint': 'lesse', 'count': 2}
            ]
            phl_data['penalties'] = {}
            model, X = build_model_X(phl_data, allow_2nd=False)
            cand = _phl_vars_on_date(X, date)
            PreferredGames().apply(model, X, phl_data, _registry(model))
            model.Add(sum(cand) == pinned_sum)
            pen = phl_data['penalties']['preferred_games']['penalties'][0]
            assert _solve_penalty(model, pen) == expected


# ----------------------------------------------------------------------
# greatere — max(0, N − sum)
# ----------------------------------------------------------------------

class TestGreatereConstraint:
    def test_greatere_count1(self, phl_data):
        """Scenario: greatere/count=1 → penalty = max(0, 1 − sum).

        Hand oracle: sum=0 → 1; sum=1 → 0.
        """
        date = '2026-03-22'
        for pinned_sum, expected in [(0, 1), (1, 0)]:
            phl_data['preferred_games'] = [
                {'grade': 'PHL', 'date': date, 'constraint': 'greatere', 'count': 1}
            ]
            phl_data['penalties'] = {}
            model, X = build_model_X(phl_data, allow_2nd=False)
            cand = _phl_vars_on_date(X, date)
            PreferredGames().apply(model, X, phl_data, _registry(model))
            model.Add(sum(cand) == pinned_sum)
            pen = phl_data['penalties']['preferred_games']['penalties'][0]
            assert _solve_penalty(model, pen) == expected


# ----------------------------------------------------------------------
# greater / less — strict, +1/−1 shift
# ----------------------------------------------------------------------

class TestGreaterLessConstraint:
    def test_greater_count1(self, phl_data):
        """greater/count=1 → sum > 1 == sum >= 2 → penalty = max(0, 2 − sum).

        Hand oracle: sum=1 → 1; sum=2 → 0.
        """
        date = '2026-03-22'
        for pinned_sum, expected in [(1, 1), (2, 0)]:
            phl_data['preferred_games'] = [
                {'grade': 'PHL', 'date': date, 'constraint': 'greater', 'count': 1}
            ]
            phl_data['penalties'] = {}
            model, X = build_model_X(phl_data, allow_2nd=False)
            cand = _phl_vars_on_date(X, date)
            PreferredGames().apply(model, X, phl_data, _registry(model))
            model.Add(sum(cand) == pinned_sum)
            pen = phl_data['penalties']['preferred_games']['penalties'][0]
            assert _solve_penalty(model, pen) == expected

    def test_less_count2(self, phl_data):
        """less/count=2 → sum < 2 == sum <= 1 → penalty = max(0, sum − 1).

        Hand oracle: sum=2 → 1; sum=1 → 0.
        """
        date = '2026-03-22'
        for pinned_sum, expected in [(2, 1), (1, 0)]:
            phl_data['preferred_games'] = [
                {'grade': 'PHL', 'date': date, 'constraint': 'less', 'count': 2}
            ]
            phl_data['penalties'] = {}
            model, X = build_model_X(phl_data, allow_2nd=False)
            cand = _phl_vars_on_date(X, date)
            PreferredGames().apply(model, X, phl_data, _registry(model))
            model.Add(sum(cand) == pinned_sum)
            pen = phl_data['penalties']['preferred_games']['penalties'][0]
            assert _solve_penalty(model, pen) == expected


# ----------------------------------------------------------------------
# Weight multiplier — per-entry weight scales the raw penalty
# ----------------------------------------------------------------------

class TestWeightMultiplier:
    def test_per_entry_weight_scales_penalty(self, phl_data):
        """Scenario: weight=30000 with default 10000 → multiplier 3.

        Hand oracle: raw |sum−1| = 1 (sum pinned to 0) → scaled = 3.
        """
        date = '2026-03-22'
        phl_data['penalty_weights'] = {'preferred_games': 10_000}
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': date, 'constraint': 'equal',
             'count': 1, 'weight': 30_000}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        cand = _phl_vars_on_date(X, date)
        PreferredGames().apply(model, X, phl_data, _registry(model))
        model.Add(sum(cand) == 0)  # raw deviation = |0 − 1| = 1
        pen = phl_data['penalties']['preferred_games']['penalties'][0]
        # multiplier = max(1, 30000 // 10000) = 3 → scaled penalty = 3 * 1 = 3.
        assert _solve_penalty(model, pen) == 3


# ----------------------------------------------------------------------
# Scope bucketing — team-pair, club, ('all',) venue+day
# ----------------------------------------------------------------------

class TestScopeBucketing:
    def test_team_pair_scope(self, phl_data):
        """A team-pair entry buckets only that pair's vars on the date.

        Hand oracle: per PHL pairing per date the fixture creates Sunday vars at
        EF×4 + WF×4 (8 NIHC) + Central Coast×2 + Maitland Park×4 = 14
        (build_model_X applies no home-venue filter to PHL). So 'Tigers PHL' vs
        'Wests PHL' on 2026-03-22 has exactly 14 candidate vars. The atom builds
        a |sum − 1| penalty over precisely those vars.
        """
        date = '2026-03-22'
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': date,
             'teams': ['Tigers', 'Wests'],
             'constraint': 'equal', 'count': 1}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        # Hand-identify the matched set: that pairing's PHL vars on the date.
        expected = [
            v for k, v in X.items()
            if k[2] == 'PHL' and k[7] == date
            and {k[0], k[1]} == {'Tigers PHL', 'Wests PHL'}
        ]
        assert len(expected) == 14, f'expected 14 Sunday vars, got {len(expected)}'
        n = PreferredGames().apply(model, X, phl_data, _registry(model))
        assert n == 1
        # Pin the pair's sum to 0 → penalty 1; other pairs irrelevant.
        model.Add(sum(expected) == 0)
        pen = phl_data['penalties']['preferred_games']['penalties'][0]
        assert _solve_penalty(model, pen) == 1

    def test_club_scope_expands(self, phl_data):
        """A club entry expands to all that club's teams (any game involving it).

        Hand oracle: club 'Tigers' at PHL on the date matches every PHL var
        where Tigers PHL is team1 or team2 — 4 opponents × 14 Sunday vars = 56.
        """
        date = '2026-03-22'
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': date, 'club': 'Tigers',
             'constraint': 'greatere', 'count': 1}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        expected = [
            v for k, v in X.items()
            if k[2] == 'PHL' and k[7] == date
            and ('Tigers PHL' in (k[0], k[1]))
        ]
        # 4 opponents (Wests, Norths, Maitland, Gosford) × 14 Sunday vars.
        assert len(expected) == 56, f'expected 56, got {len(expected)}'
        PreferredGames().apply(model, X, phl_data, _registry(model))
        # greatere count 1: pin sum to 0 → penalty max(0, 1 − 0) = 1.
        model.Add(sum(expected) == 0)
        pen = phl_data['penalties']['preferred_games']['penalties'][0]
        assert _solve_penalty(model, pen) == 1

    def test_all_matcher_venue_day_scope(self, phl_data):
        """A team-less entry ('all',) buckets every var matching the scope.

        Hand oracle: {day:'Sunday', field_location:GOSFORD} on PHL — every PHL
        Sunday var at Central Coast across all dates/pairs. The fixture has 2
        Gosford Sunday slots per date × 5 dates × C(5,2)=10 PHL pairs = 100.
        """
        from constraints.atoms.base import GOSFORD
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'day': 'Sunday', 'field_location': GOSFORD,
             'constraint': 'lesse', 'count': 0}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        expected = [
            v for k, v in X.items()
            if k[2] == 'PHL' and k[3] == 'Sunday' and k[10] == GOSFORD
        ]
        assert len(expected) == 100, f'expected 100, got {len(expected)}'
        PreferredGames().apply(model, X, phl_data, _registry(model))
        # lesse count 0: penalty = max(0, sum − 0) = sum. Pin sum to 2 → 2.
        model.Add(sum(expected) == 2)
        pen = phl_data['penalties']['preferred_games']['penalties'][0]
        assert _solve_penalty(model, pen) == 2


# ----------------------------------------------------------------------
# Non-fatal edge cases — empty scope, locked weeks
# ----------------------------------------------------------------------

class TestNonFatalEdges:
    def test_empty_scope_no_penalty_no_crash(self, phl_data):
        """Zero-candidate scope → no penalty, no sys.exit, warning logged."""
        # Date with no timeslots in the fixture → no candidate vars.
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': '2099-01-01', 'constraint': 'equal', 'count': 1}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PreferredGames().apply(model, X, phl_data, _registry(model))
        # No candidates → no penalty entry created for this scope.
        assert n == 0
        bucket = phl_data['penalties'].get('preferred_games', {'penalties': []})
        assert len(bucket['penalties']) == 0

    def test_count_exceeds_candidates_no_crash(self, phl_data):
        """equal/count=99 on a date with few candidates must not crash CP-SAT.

        review C2: the penalty IntVar bound must cover |sum − 99|. With sum
        pinned to 0 the deviation is 99; the bound max(N, len) = 99 holds.
        """
        date = '2026-03-22'
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': date, 'constraint': 'equal', 'count': 99}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        cand = _phl_vars_on_date(X, date)
        PreferredGames().apply(model, X, phl_data, _registry(model))
        model.Add(sum(cand) == 0)
        pen = phl_data['penalties']['preferred_games']['penalties'][0]
        # |0 − 99| = 99.
        assert _solve_penalty(model, pen) == 99

    def test_locked_week_scope_skipped(self, phl_data):
        """Vars in locked weeks are skipped per-variable (review M4).

        Hand oracle: lock week 1 (date 2026-03-22). An equal/count=1 entry on
        that date now has zero non-locked candidates → no penalty entry.
        """
        date = '2026-03-22'
        phl_data['locked_weeks'] = {1}
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': date, 'constraint': 'equal', 'count': 1}
        ]
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        n = PreferredGames().apply(model, X, phl_data, _registry(model))
        assert n == 0


# ----------------------------------------------------------------------
# DoD 4 — equivalence with the deleted PreferredDates
# ----------------------------------------------------------------------

class TestPreferredDatesEquivalence:
    def test_migrated_entry_matches_preferred_dates_penalty(self, phl_data):
        """The migrated PREFERRED_GAMES entry yields the same |sum − 1| penalty.

        The deleted `PreferredDates` atom produced, per preferred date with PHL
        candidate vars, exactly one penalty IntVar = `|sum(PHL vars on date) − 1|`
        (constraints/atoms/preferred_dates.py:47-48). The migration expresses
        that as `{grade:'PHL', date:X, constraint:'equal', count:1}`.

        Hand oracle: on 2026-03-22 the PHL candidate set is the full set of PHL
        vars on that date. With sum pinned to 0, both old and new penalty = 1;
        with sum pinned to 1, both = 0. We assert the new atom reproduces that.
        """
        date = '2026-03-22'
        # New mechanism: the marquee-PHL-date migration entry.
        phl_data['preferred_games'] = [
            {'grade': 'PHL', 'date': date, 'constraint': 'equal',
             'count': 1, 'weight': 10_000, 'description': 'marquee PHL date'}
        ]
        phl_data['penalty_weights'] = {'preferred_games': 10_000}
        phl_data['penalties'] = {}
        model, X = build_model_X(phl_data, allow_2nd=False)
        cand = _phl_vars_on_date(X, date)
        n = PreferredGames().apply(model, X, phl_data, _registry(model))
        # Old PreferredDates emitted exactly 1 penalty IntVar for this date.
        assert n == 1
        pen = phl_data['penalties']['preferred_games']['penalties'][0]
        # sum=0 → |0 − 1| = 1 (multiplier 1 since weight == default).
        model.Add(sum(cand) == 0)
        assert _solve_penalty(model, pen) == 1
