"""Tests for `_merge_constraint_defaults` and the perennial CONSTRAINT_DEFAULTS."""

import pytest

from utils import _merge_constraint_defaults
from config.defaults import CONSTRAINT_DEFAULTS


def test_perennial_defaults_have_required_keys():
    expected_keys = {
        'spacing_base_slack',
        'max_friday_broadmeadow',
        'gosford_friday_games',
        'maitland_friday_games',
        # spec-018: maitland_max_consecutive_home / away_maitland_max_clubs
        # removed (venue-sequencing rules deleted).
        # spec-024: max_clubs_per_field removed (MinimiseClubsOnAFieldBroadmeadow deleted).
        'club_game_spread_max_gap',
        'club_game_spread_max_overlap',
        # spec-033 Unit D: hard field-concentration cap for ClubGameSpread.
        'club_game_spread_max_fields',
        # spec-033 Unit B: base slack below the raw ideal bye gap.
        'bye_spacing_base_slack',
        # spec-033 Unit A: club_vs_club_alignment_base_slack removed — alignment
        # is a fixed hard rule with no slack.
        'phl_2nd_cross_venue_min_minutes',
        'worst_timeslot_time',
    }
    missing = expected_keys - set(CONSTRAINT_DEFAULTS.keys())
    assert not missing, f"Missing keys in CONSTRAINT_DEFAULTS: {missing}"


def test_merge_with_empty_overrides_returns_defaults():
    merged = _merge_constraint_defaults({})
    assert merged == CONSTRAINT_DEFAULTS
    # Confirm it's a copy, not the same object
    assert merged is not CONSTRAINT_DEFAULTS


def test_merge_with_none_returns_defaults():
    merged = _merge_constraint_defaults(None)
    assert merged == CONSTRAINT_DEFAULTS


def test_merge_overrides_take_precedence():
    overrides = {'max_friday_broadmeadow': 99, 'phl_2nd_cross_venue_min_minutes': 240}
    merged = _merge_constraint_defaults(overrides)
    assert merged['max_friday_broadmeadow'] == 99
    assert merged['phl_2nd_cross_venue_min_minutes'] == 240


def test_merge_keeps_unrelated_defaults():
    overrides = {'max_friday_broadmeadow': 99}
    merged = _merge_constraint_defaults(overrides)
    assert merged['gosford_friday_games'] == CONSTRAINT_DEFAULTS['gosford_friday_games']
    assert merged['maitland_friday_games'] == CONSTRAINT_DEFAULTS['maitland_friday_games']


def test_merge_does_not_mutate_input():
    overrides = {'max_friday_broadmeadow': 99}
    snapshot = dict(overrides)
    _merge_constraint_defaults(overrides)
    assert overrides == snapshot


def test_season_2026_inherits_perennials_for_unset_keys():
    """The 2026 season config doesn't set every key — those must fall back to perennials."""
    from config import load_season_data
    data = load_season_data(2026)
    cd = data['constraint_defaults']
    # Keys 2026 sets explicitly should match the season config
    assert cd['max_friday_broadmeadow'] == 3
    # Keys 2026 doesn't set should come from defaults
    assert cd['phl_2nd_cross_venue_min_minutes'] == CONSTRAINT_DEFAULTS['phl_2nd_cross_venue_min_minutes']
    assert cd['worst_timeslot_time'] == CONSTRAINT_DEFAULTS['worst_timeslot_time']

# spec-015: removed test_gosford_friday_rounds_drives_unified_engine_behavior —
# the 'gosford_friday_rounds' default was deleted (it only fed the deleted
# GosfordFridayRoundsForced atom). Gosford Friday rounds are now FORCED_GAMES
# count entries; the generic capability is covered by
# tests/test_forced_games_count_rules.py.
