"""Tests for the FORCED/BLOCKED count-adjuster framework (Phase 4).

The framework lets each `ConstraintInfo` register a `forced_blocked_adjuster`
callable. `run_count_adjusters(data)` invokes every registered adjuster and
stashes its return value under `data['count_adjustments'][canonical_name]`.
Atoms read their entry during `apply()`.

The tests below verify the framework, not any specific adjuster — actual
adjusters land in subsequent commits with per-formula sign-off.
"""
from __future__ import annotations

import pytest

from constraints.registry import (
    CONSTRAINT_REGISTRY,
    ConstraintInfo,
    run_count_adjusters,
)


@pytest.fixture
def temp_adjuster():
    """Fixture that registers an adjuster on an existing entry, then cleans up."""
    target = 'EqualGamesAndBalanceMatchUps'
    saved = CONSTRAINT_REGISTRY[target].forced_blocked_adjuster
    yield target
    CONSTRAINT_REGISTRY[target].forced_blocked_adjuster = saved


class TestRunCountAdjusters:
    def test_no_adjusters_registered_returns_empty(self):
        data = {'forced_games': [], 'blocked_games': []}
        out = run_count_adjusters(data)
        assert isinstance(out, dict)
        assert data.get('count_adjustments') is out

    def test_adjuster_result_stored_under_canonical_name(self, temp_adjuster):
        target = temp_adjuster
        CONSTRAINT_REGISTRY[target].forced_blocked_adjuster = (
            lambda d, f, b: {'foo': 1, 'bar': 2}
        )
        data = {'forced_games': [], 'blocked_games': []}
        out = run_count_adjusters(data)
        assert out[target] == {'foo': 1, 'bar': 2}
        assert data['count_adjustments'][target] == {'foo': 1, 'bar': 2}

    def test_adjuster_returning_none_is_skipped(self, temp_adjuster):
        target = temp_adjuster
        CONSTRAINT_REGISTRY[target].forced_blocked_adjuster = (
            lambda d, f, b: None
        )
        data = {'forced_games': [], 'blocked_games': []}
        out = run_count_adjusters(data)
        assert target not in out

    def test_adjuster_receives_forced_and_blocked_lists(self, temp_adjuster):
        target = temp_adjuster
        seen = {}

        def adj(d, f, b):
            seen['forced'] = f
            seen['blocked'] = b
            return {}

        CONSTRAINT_REGISTRY[target].forced_blocked_adjuster = adj
        data = {
            'forced_games': [{'reason': 'x'}],
            'blocked_games': [{'reason': 'y'}],
        }
        run_count_adjusters(data)
        assert seen['forced'] == [{'reason': 'x'}]
        assert seen['blocked'] == [{'reason': 'y'}]

    def test_adjuster_handles_missing_forced_blocked_keys(self, temp_adjuster):
        target = temp_adjuster
        CONSTRAINT_REGISTRY[target].forced_blocked_adjuster = (
            lambda d, f, b: {'forced_count': len(f), 'blocked_count': len(b)}
        )
        data = {}
        out = run_count_adjusters(data)
        assert out[target] == {'forced_count': 0, 'blocked_count': 0}

    def test_buggy_adjuster_wraps_error_with_context(self, temp_adjuster):
        target = temp_adjuster

        def boom(d, f, b):
            raise ValueError('synthetic')

        CONSTRAINT_REGISTRY[target].forced_blocked_adjuster = boom
        data = {'forced_games': [], 'blocked_games': []}
        with pytest.raises(RuntimeError, match=target):
            run_count_adjusters(data)


class TestEngineDispatch:
    """`UnifiedConstraintEngine.build_groupings` must populate count_adjustments."""

    def test_build_groupings_populates_count_adjustments(self):
        from ortools.sat.python import cp_model
        from constraints.unified import UnifiedConstraintEngine
        from tests.atoms.club_day_fixture import (
            build_club_day_fixture, build_model_X,
        )

        data = build_club_day_fixture()
        model, X = build_model_X(data)
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        assert 'count_adjustments' in data
        # No adjusters are registered yet, so the dict is empty — but it
        # exists, which is the contract atoms rely on.
        assert isinstance(data['count_adjustments'], dict)
