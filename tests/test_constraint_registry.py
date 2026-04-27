"""Tests for constraints/registry.py -- constraint name mapping integrity."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.registry import (
    CONSTRAINT_REGISTRY,
    ConstraintInfo,
    normalize_constraint_name,
    get_canonical_for_solver_name,
    get_checks_for_applied_constraints,
    get_all_tester_methods,
    get_slack_key,
    get_all_canonical_names,
    get_tester_only_constraints,
)
from constraints.severity import CONSTRAINT_TO_SEVERITY


class TestRegistryCompleteness:
    """Verify every solver constraint and tester method is covered."""

    def test_every_severity_entry_maps_to_registry(self):
        """Every entry in CONSTRAINT_TO_SEVERITY should resolve to a registry entry."""
        unmapped = []
        for solver_name in CONSTRAINT_TO_SEVERITY:
            canonical = get_canonical_for_solver_name(solver_name)
            if canonical is None:
                # Try normalization fallback
                canonical = normalize_constraint_name(solver_name)
            if canonical not in CONSTRAINT_REGISTRY:
                unmapped.append(solver_name)
        assert unmapped == [], f"Solver names not in registry: {unmapped}"

    def test_every_tester_check_method_covered(self):
        """Every _check_* method in the registry should be a real method on DrawTester."""
        from analytics.tester import DrawTester
        all_methods = get_all_tester_methods()
        for method_name in all_methods:
            assert hasattr(DrawTester, method_name), \
                f"Registry lists '{method_name}' but DrawTester has no such method"

    def test_every_drawtester_check_in_registry(self):
        """Every _check_* method on DrawTester should be in the registry."""
        from analytics.tester import DrawTester
        registry_methods = set(get_all_tester_methods())
        for attr_name in dir(DrawTester):
            if attr_name.startswith('_check_') and callable(getattr(DrawTester, attr_name)):
                assert attr_name in registry_methods, \
                    f"DrawTester.{attr_name} not covered by registry"

    def test_registry_has_expected_entry_count(self):
        """Registry should have 19 entries (18 solver + 1 tester-only).
        MaitlandHomeGrouping and MaxMaitlandHomeWeekends are merged."""
        assert len(CONSTRAINT_REGISTRY) == 21

    def test_all_entries_have_required_fields(self):
        """Every ConstraintInfo must have canonical_name and at least one tester method."""
        for name, info in CONSTRAINT_REGISTRY.items():
            assert info.canonical_name == name
            assert len(info.tester_check_methods) >= 1, \
                f"{name}: must have at least one tester_check_method"
            assert len(info.tester_violation_names) >= 1, \
                f"{name}: must have at least one tester_violation_name"
            assert 1 <= info.severity_level <= 5, \
                f"{name}: severity_level must be 1-5, got {info.severity_level}"


class TestNormalization:
    """Test normalize_constraint_name with various suffix patterns."""

    @pytest.mark.parametrize("input_name,expected", [
        ('NoDoubleBookingTeamsConstraint', 'NoDoubleBookingTeams'),
        ('NoDoubleBookingTeamsConstraintAI', 'NoDoubleBookingTeams'),
        ('EnsureEqualGamesAndBalanceMatchUps', 'EqualGamesAndBalanceMatchUps'),
        ('EnsureEqualGamesAndBalanceMatchUpsAI', 'EqualGamesAndBalanceMatchUps'),
        ('ClubDayConstraint', 'ClubDay'),
        ('ClubDayConstraintAI', 'ClubDay'),
        ('MaitlandHomeGrouping', 'MaitlandHomeGrouping'),
        ('MaitlandHomeGroupingAI', 'MaitlandHomeGrouping'),
        ('EqualMatchUpSpacingConstraint', 'EqualMatchUpSpacing'),
        ('EqualMatchUpSpacingConstraintAI', 'EqualMatchUpSpacing'),
        ('PreferredTimesConstraint', 'PreferredTimes'),
        ('PreferredTimesConstraintAI', 'PreferredTimes'),
    ])
    def test_normalize_suffix_patterns(self, input_name, expected):
        result = normalize_constraint_name(input_name)
        assert result == expected, f"normalize('{input_name}') = '{result}', expected '{expected}'"

    def test_normalize_canonical_passthrough(self):
        """Canonical names should pass through unchanged."""
        for name in CONSTRAINT_REGISTRY:
            assert normalize_constraint_name(name) == name

    def test_normalize_unknown_returns_stripped(self):
        """Unknown names get suffixes stripped but aren't forced to a registry entry."""
        result = normalize_constraint_name('SomeRandomConstraint')
        assert result == 'SomeRandom'


class TestSolverNameLookup:
    """Test get_canonical_for_solver_name."""

    def test_all_solver_names_resolve(self):
        """Every solver_class_name in the registry should resolve back."""
        for canonical, info in CONSTRAINT_REGISTRY.items():
            for solver_name in info.solver_class_names:
                result = get_canonical_for_solver_name(solver_name)
                assert result == canonical, \
                    f"get_canonical_for_solver_name('{solver_name}') = '{result}', expected '{canonical}'"

    def test_unknown_solver_name_returns_none(self):
        assert get_canonical_for_solver_name('NonExistentConstraint') is None

    def test_round_trip(self):
        """solver name -> canonical -> solver names should contain original."""
        for canonical, info in CONSTRAINT_REGISTRY.items():
            for solver_name in info.solver_class_names:
                resolved_canonical = get_canonical_for_solver_name(solver_name)
                resolved_info = CONSTRAINT_REGISTRY[resolved_canonical]
                assert solver_name in resolved_info.solver_class_names


class TestChecksForApplied:
    """Test get_checks_for_applied_constraints."""

    def test_returns_correct_methods(self):
        methods = get_checks_for_applied_constraints(
            ['NoDoubleBookingTeamsConstraint', 'EnsureEqualGamesAndBalanceMatchUps']
        )
        assert '_check_no_double_booking_teams' in methods
        assert '_check_equal_games' in methods
        assert '_check_balanced_matchups' in methods

    def test_empty_input(self):
        methods = get_checks_for_applied_constraints([])
        assert len(methods) == 0

    def test_unknown_name_ignored(self):
        methods = get_checks_for_applied_constraints(['NonExistent'])
        assert len(methods) == 0

    def test_ai_variant_maps_same_methods(self):
        m1 = get_checks_for_applied_constraints(['NoDoubleBookingTeamsConstraint'])
        m2 = get_checks_for_applied_constraints(['NoDoubleBookingTeamsConstraintAI'])
        assert m1 == m2


class TestSlackKeys:
    """Test get_slack_key."""

    def test_known_slack_keys(self):
        assert get_slack_key('MaitlandHomeGrouping') == 'MaitlandHomeGrouping'
        assert get_slack_key('EqualMatchUpSpacing') == 'EqualMatchUpSpacingConstraint'
        assert get_slack_key('AwayAtMaitlandGrouping') == 'AwayAtMaitlandGrouping'
        assert get_slack_key('ClubGameSpread') == 'ClubGameSpread'
        assert get_slack_key('ClubVsClubAlignment') == 'ClubVsClubAlignment'

    def test_no_slack_returns_none(self):
        assert get_slack_key('NoDoubleBookingTeams') is None
        assert get_slack_key('PHLAndSecondGradeAdjacency') is None

    def test_all_slack_keys_exist_in_known_dicts(self):
        """All slack keys in the registry should be recognizable slack dict keys."""
        known_slack_keys = {
            'EqualMatchUpSpacingConstraint',
            'AwayAtMaitlandGrouping',
            'MaitlandHomeGrouping',
            'ClubVsClubAlignment',
            'MaximiseClubsPerTimeslotBroadmeadow',
            'MinimiseClubsOnAFieldBroadmeadow',
            'ClubGameSpread',
        }
        for info in CONSTRAINT_REGISTRY.values():
            if info.slack_key:
                assert info.slack_key in known_slack_keys, \
                    f"Registry slack_key '{info.slack_key}' for {info.canonical_name} not in known set"


class TestTesterOnlyConstraints:
    """Test tester-only diagnostic constraints."""

    def test_club_field_concentration_is_tester_only(self):
        info = CONSTRAINT_REGISTRY['ClubFieldConcentration']
        assert info.tester_only is True
        assert info.solver_class_names == []

    def test_get_tester_only_returns_expected(self):
        tester_only = get_tester_only_constraints()
        assert 'ClubFieldConcentration' in tester_only

    def test_non_tester_only_have_solver_names(self):
        for name, info in CONSTRAINT_REGISTRY.items():
            if not info.tester_only:
                assert len(info.solver_class_names) >= 1, \
                    f"{name}: non-tester-only must have solver_class_names"
