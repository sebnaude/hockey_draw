"""Phase 4 FORCED/BLOCKED count adjusters — math + integration tests.

Tests cover the adjuster output (synthetic FORCED/BLOCKED inputs → expected
adjustment dict) and the integration point on the atom / legacy unified.py
method (the constraint reads the adjustment and behaves accordingly).
"""
from __future__ import annotations

from typing import Dict, List

import pytest

from constraints.atoms._adjusters import (
    equal_matchup_spacing_adjuster,
)
from constraints.registry import CONSTRAINT_REGISTRY, run_count_adjusters
from models import Club, Grade, PlayingField, Team, Timeslot


# ----------------------------------------------------------------------
# Tiny shared fixture: 2 clubs (Maitland + Norths), PHL teams + 3rd-grade
# teams. Enough to exercise all adjusters.
# ----------------------------------------------------------------------


def _build_data() -> Dict:
    clubs = [
        Club(name='Maitland', home_field='Maitland Park'),
        Club(name='Norths', home_field='Newcastle International Hockey Centre'),
    ]
    teams = [
        Team(name='Maitland PHL', club=clubs[0], grade='PHL'),
        Team(name='Norths PHL', club=clubs[1], grade='PHL'),
        Team(name='Maitland 3rd', club=clubs[0], grade='3rd'),
        Team(name='Norths 3rd', club=clubs[1], grade='3rd'),
    ]
    grades = [
        Grade(name='PHL', teams=['Maitland PHL', 'Norths PHL']),
        Grade(name='3rd', teams=['Maitland 3rd', 'Norths 3rd']),
    ]
    return {
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'home_field_map': {'Maitland': 'Maitland Park'},
        'forced_games': [],
        'blocked_games': [],
        'timeslots': [],
        'num_rounds': {'PHL': 6, '3rd': 6, 'max': 6},
    }


# ----------------------------------------------------------------------
# #2 EqualMatchUpSpacing adjuster
# ----------------------------------------------------------------------


class TestEqualMatchUpSpacingAdjuster:
    def test_no_forced_returns_none(self):
        data = _build_data()
        out = equal_matchup_spacing_adjuster(data, [], [])
        assert out is None

    def test_forced_pair_records_weeks(self):
        data = _build_data()
        forced = [
            {'grade': 'PHL', 'teams': ['Maitland', 'Norths'], 'week': 3},
            {'grade': 'PHL', 'teams': ['Maitland', 'Norths'], 'week': 7},
        ]
        out = equal_matchup_spacing_adjuster(data, forced, [])
        # Pairs are alphabetical; check both orderings safely.
        key = ('Maitland PHL', 'Norths PHL', 'PHL')
        assert key in out
        assert out[key] == {3, 7}

    def test_forced_with_round_no(self):
        data = _build_data()
        forced = [
            {'grade': 'PHL', 'teams': ['Maitland', 'Norths'], 'round_no': 5},
        ]
        out = equal_matchup_spacing_adjuster(data, forced, [])
        assert out[('Maitland PHL', 'Norths PHL', 'PHL')] == {5}

    def test_entry_without_pin_skipped(self):
        data = _build_data()
        forced = [
            # No week/round_no/date — can't pin, so adjuster ignores.
            {'grade': 'PHL', 'teams': ['Maitland', 'Norths']},
        ]
        out = equal_matchup_spacing_adjuster(data, forced, [])
        assert out is None


# spec-018: TestMaitlandHomeGroupingAdjuster and
# TestAwayAtMaitlandGroupingAdjuster removed — the
# `maitland_home_grouping_adjuster` / `away_at_maitland_grouping_adjuster`
# adjusters were deleted along with the venue-sequencing rules they fed.


# ----------------------------------------------------------------------
# Engine-level integration: registry dispatches every adjuster.
# ----------------------------------------------------------------------


class TestRegistryDispatch:
    def test_all_adjusters_registered(self):
        # spec-018: MaitlandHomeGrouping / AwayAtMaitlandGrouping adjusters
        # removed; only EqualMatchUpSpacing remains.
        for name in (
            'EqualMatchUpSpacing',
        ):
            assert CONSTRAINT_REGISTRY[name].forced_blocked_adjuster is not None, (
                f'Adjuster missing for {name}'
            )

    def test_run_count_adjusters_populates_spacing(self):
        data = _build_data()
        # A FORCED PHL Maitland-vs-Norths entry pinned to round 5 feeds the
        # EqualMatchUpSpacing adjuster (per-pair forced rounds).
        data['forced_games'] = [
            {
                'grade': 'PHL',
                'teams': ['Maitland', 'Norths'],
                'round_no': 5,
            },
        ]
        adjustments = run_count_adjusters(data)
        # spec-018: MaitlandHomeGrouping / AwayAtMaitlandGrouping entries gone.
        assert 'MaitlandHomeGrouping' not in adjustments
        assert 'AwayAtMaitlandGrouping' not in adjustments
        # EqualMatchUpSpacing recorded round 5 for the Maitland/Norths PHL pair.
        spacing = adjustments.get('EqualMatchUpSpacing', {})
        assert any(5 in rounds for rounds in spacing.values())
