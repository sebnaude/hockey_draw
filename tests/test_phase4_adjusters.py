"""Phase 4 FORCED/BLOCKED count adjusters — math + integration tests.

Tests cover the adjuster output (synthetic FORCED/BLOCKED inputs → expected
adjustment dict) and the integration point on the atom / legacy unified.py
method (the constraint reads the adjustment and behaves accordingly).
"""
from __future__ import annotations

from typing import Dict, List

import pytest

from constraints.atoms._adjusters import (
    away_at_maitland_grouping_adjuster,
    equal_matchup_spacing_adjuster,
    maitland_home_grouping_adjuster,
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


# ----------------------------------------------------------------------
# #3 MaitlandHomeGrouping adjuster
# ----------------------------------------------------------------------


class TestMaitlandHomeGroupingAdjuster:
    def test_no_forced_returns_none(self):
        data = _build_data()
        out = maitland_home_grouping_adjuster(data, [], [])
        assert out is None

    def test_forced_home_weekend_recorded(self):
        data = _build_data()
        forced = [{
            'grade': 'PHL',
            'teams': ['Maitland', 'Norths'],
            'field_location': 'Maitland Park',
            'week': 5,
        }]
        out = maitland_home_grouping_adjuster(data, forced, [])
        assert out == {'Maitland': {5}}

    def test_no_team_match_skipped(self):
        data = _build_data()
        # Forced entry at Maitland Park but neither team is from Maitland.
        forced = [{
            'grade': '3rd',
            'teams': ['Norths', 'Norths'],
            'field_location': 'Maitland Park',
            'week': 5,
        }]
        out = maitland_home_grouping_adjuster(data, forced, [])
        # No Maitland team in any pair → no home weekend recorded.
        assert out is None

    def test_non_default_venue_ignored(self):
        data = _build_data()
        forced = [{
            'grade': 'PHL',
            'teams': ['Maitland', 'Norths'],
            'field_location': 'Newcastle International Hockey Centre',
            'week': 5,
        }]
        out = maitland_home_grouping_adjuster(data, forced, [])
        assert out is None


# ----------------------------------------------------------------------
# #4 AwayAtMaitlandGrouping adjuster
# ----------------------------------------------------------------------


class TestAwayAtMaitlandGroupingAdjuster:
    def test_no_forced_returns_none(self):
        data = _build_data()
        out = away_at_maitland_grouping_adjuster(data, [], [])
        assert out is None

    def test_forced_away_recorded(self):
        data = _build_data()
        forced = [{
            'grade': 'PHL',
            'teams': ['Maitland', 'Norths'],
            'field_location': 'Maitland Park',
            'week': 7,
        }]
        out = away_at_maitland_grouping_adjuster(data, forced, [])
        assert out == {(7, 'Maitland Park'): {'Norths'}}

    def test_multiple_away_clubs_aggregated(self):
        data = _build_data()
        # Add a third club so we have two distinct away clubs.
        data['clubs'].append(Club(name='Souths', home_field='Newcastle International Hockey Centre'))
        data['teams'].append(Team(name='Souths PHL', club=data['clubs'][-1], grade='PHL'))
        forced = [
            {'grade': 'PHL', 'teams': ['Maitland', 'Norths'],
             'field_location': 'Maitland Park', 'week': 7},
            {'grade': 'PHL', 'teams': ['Maitland', 'Souths'],
             'field_location': 'Maitland Park', 'week': 7},
        ]
        out = away_at_maitland_grouping_adjuster(data, forced, [])
        assert out == {(7, 'Maitland Park'): {'Norths', 'Souths'}}


# ----------------------------------------------------------------------
# Engine-level integration: registry dispatches every adjuster.
# ----------------------------------------------------------------------


class TestRegistryDispatch:
    def test_all_adjusters_registered(self):
        for name in (
            'EqualMatchUpSpacing',
            'MaitlandHomeGrouping',
            'AwayAtMaitlandGrouping',
        ):
            assert CONSTRAINT_REGISTRY[name].forced_blocked_adjuster is not None, (
                f'Adjuster missing for {name}'
            )

    def test_run_count_adjusters_populates_all(self):
        data = _build_data()
        data['forced_games'] = [
            {
                'grade': 'PHL',
                'teams': ['Maitland', 'Norths'],
                'field_location': 'Maitland Park',
                'week': 5,
            },
            {
                'grade': 'PHL',
                'teams': ['Maitland', 'Norths'],
                'field_location': 'Central Coast Hockey Park',
                'day': 'Friday',
                'count': 2,
            },
        ]
        adjustments = run_count_adjusters(data)
        # MaitlandHomeGrouping registered week 5 from entry 1.
        assert adjustments['MaitlandHomeGrouping'] == {'Maitland': {5}}
        # AwayAtMaitlandGrouping recorded Norths in week 5.
        assert adjustments['AwayAtMaitlandGrouping'][(5, 'Maitland Park')] == {'Norths'}
