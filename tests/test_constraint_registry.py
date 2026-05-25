"""Tests for constraints/registry.py -- constraint name mapping integrity."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.registry import (
    CONSTRAINT_REGISTRY,
    ConstraintInfo,
    HELPER_VAR_CATALOG,
    normalize_constraint_name,
    get_canonical_for_solver_name,
    get_checks_for_applied_constraints,
    get_all_tester_methods,
    get_slack_key,
    get_all_canonical_names,
    get_tester_only_constraints,
    get_atoms_in_group,
    get_adjuster,
    validate_required_helpers,
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
        """Registry contains 21 originals + 5 PHL atoms (Phase 3a; spec-010
        later DELETED PHLRoundOnePlay → 4) + 5 ClubDay atoms (Phase 3b) +
        2 Phase-6 generic aliases (NonDefaultHomeGrouping,
        AwayAtNonDefaultGrouping) +
        1 spec-002 soft penalty atom (SoftLexMatchupOrdering) +
        2 spec-007 atoms (SameGradeSameClubNoConcurrency, TeamPairNoConcurrency) +
        2 spec-003 atoms (NIHCFillWFBeforeEF, NIHCFillEFBeforeSF) +
        1 spec-006 soft penalty atom (PreferredWeekendsAwayGround) +
        2 spec-004 atoms (AwayClubHomeWeekendsCount,
        AwayClubPerOpponentAndAggregateHomeBalance) +
        2 spec-005 atoms (ClubVsClubStackedWeekends,
        ClubVsClubStackedCoLocation — the 4 obsolete Phase-3c ClubVsClub atoms
        they replaced were DELETED, leaving only the legacy ClubVsClubAlignment
        slack-key/parity entry) +
        1 spec-012 soft penalty atom (MaitlandAlternateHomeAway) +
        1 spec-008 atom (BalancedByeSpacing — byes-as-first-class
          spacing, HARD severity 2, own slack key)
        = 44.

        spec-014 renamed the `PHLAndSecondGradeAdjacency` entry to the
        `PHLAnd2ndAdjacency` atom (1:1 replacement) — count unchanged.
        spec-015 DELETED `GosfordFridayRoundsForced` (its per-round sum==1 rule
        is now a generic FORCED_GAMES count entry): 44 - 1 = 43.
        spec-018 DELETED the venue-sequencing entries `NonDefaultHomeGrouping`,
        `MaitlandHomeGrouping`, `AwayAtNonDefaultGrouping`,
        `AwayAtMaitlandGrouping` and `MaitlandAlternateHomeAway`: 43 - 5 = 38.
        spec-020 DELETED `PreferredDates` and ADDED `PreferredGames` (the generic
        soft FORCED analogue) — net 0: 38 - 1 + 1 = 38.
        spec-021 replaced `EnsureBestTimeslotChoices` with `VenueEarliestSlotFill`
        (net 0) and ADDED `ClubNoConcurrentSlot` (extracted from ClubGameSpread's
        lower no-double-up bound): 38 + 1 = 39.
        spec-024 DELETED `MaximiseClubsPerTimeslotBroadmeadow` and
        `MinimiseClubsOnAFieldBroadmeadow` (their club-spread intent is now the
        field-aware `ClubGameSpread`): 39 - 2 = 37.
        spec-025 ADDED `LockedPairings` (tester_only date-pin check, sister to
        `ForcedGames`/`BlockedGames`): 37 + 1 = 38.
        spec-027 ADDED 13 regeneration soft-analogue atoms (`*RegenSoft`):
        PHLAnd2ndAdjacency, AwayClubHomeWeekendsCount, ClubVsClubStackedWeekends,
        ClubVsClubStackedCoLocation, EqualMatchUpSpacing, BalancedByeSpacing,
        the 5 ClubDay sub-atoms (Participation/IntraClubMatchup/OpponentMatchup/
        SameField/ContiguousSlots), ClubGameSpread, VenueEarliestSlotFill:
        38 + 13 = 51. spec-030 DELETED PHLAnd2ndConcurrencyAtBroadmeadow: 50.
        spec-031 DELETED 1 tester-only diagnostic: 49."""
        assert len(CONSTRAINT_REGISTRY) == 49

    def test_all_entries_have_required_fields(self):
        """Every ConstraintInfo must have canonical_name and at least one tester method.

        Exception: pure-soft penalty atoms (has_soft_component=True, empty
        tester_check_methods, no tester_only) contribute only to the objective
        and have no violation semantics — they may have empty tester lists.
        """
        for name, info in CONSTRAINT_REGISTRY.items():
            assert info.canonical_name == name
            assert 1 <= info.severity_level <= 5, \
                f"{name}: severity_level must be 1-5, got {info.severity_level}"
            # Pure-soft penalty atoms (objective-only, no violation checker) are exempt.
            is_pure_soft = (
                info.has_soft_component
                and not info.tester_only
                and len(info.tester_check_methods) == 0
                and len(info.tester_violation_names) == 0
            )
            if not is_pure_soft:
                assert len(info.tester_check_methods) >= 1, \
                    f"{name}: must have at least one tester_check_method"
                assert len(info.tester_violation_names) >= 1, \
                    f"{name}: must have at least one tester_violation_name"


class TestNormalization:
    """Test normalize_constraint_name with various suffix patterns."""

    @pytest.mark.parametrize("input_name,expected", [
        ('NoDoubleBookingTeamsConstraint', 'NoDoubleBookingTeams'),
        ('NoDoubleBookingTeamsConstraintAI', 'NoDoubleBookingTeams'),
        ('EnsureEqualGamesAndBalanceMatchUps', 'EqualGamesAndBalanceMatchUps'),
        ('EnsureEqualGamesAndBalanceMatchUpsAI', 'EqualGamesAndBalanceMatchUps'),
        ('ClubDayConstraint', 'ClubDay'),
        ('ClubDayConstraintAI', 'ClubDay'),
        # spec-018: MaitlandHomeGrouping* normalization cases removed — the
        # entries and their solver classes were deleted.
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
        # spec-018: MaitlandHomeGrouping / AwayAtMaitlandGrouping slack keys
        # removed along with their rules.
        assert get_slack_key('EqualMatchUpSpacing') == 'EqualMatchUpSpacingConstraint'
        assert get_slack_key('ClubGameSpread') == 'ClubGameSpread'
        # spec-033 Unit A: ClubVsClubAlignment slack key removed — alignment is
        # now a fixed hard rule with no slack.

    def test_no_slack_returns_none(self):
        assert get_slack_key('NoDoubleBookingTeams') is None
        assert get_slack_key('PHLAnd2ndAdjacency') is None
        # spec-033 Unit A: ClubVsClubAlignment + its two stacked atoms are slack-free.
        assert get_slack_key('ClubVsClubAlignment') is None
        assert get_slack_key('ClubVsClubStackedWeekends') is None
        assert get_slack_key('ClubVsClubStackedCoLocation') is None

    def test_all_slack_keys_exist_in_known_dicts(self):
        """All slack keys in the registry should be recognizable slack dict keys."""
        known_slack_keys = {
            'EqualMatchUpSpacingConstraint',
            # spec-018: AwayAtMaitlandGrouping / MaitlandHomeGrouping removed.
            # spec-033 Unit A: ClubVsClubAlignment slack key removed (fixed hard rule).
            'MaximiseClubsPerTimeslotBroadmeadow',
            'MinimiseClubsOnAFieldBroadmeadow',
            'ClubGameSpread',
            # spec-008 Part B: separate slack for byes spacing — convenor
            # can loosen one without touching matchup spacing.
            'BalancedByeSpacing',
            # spec-033 Unit E: ClubNoConcurrentSlot is now soft + slack — its
            # slack raises the per-slot overlap ceiling (1 + slack).
            'ClubNoConcurrentSlot',
        }
        for info in CONSTRAINT_REGISTRY.values():
            if info.slack_key:
                assert info.slack_key in known_slack_keys, \
                    f"Registry slack_key '{info.slack_key}' for {info.canonical_name} not in known set"


class TestTesterOnlyConstraints:
    """Test tester-only diagnostic constraints."""

    def test_get_tester_only_returns_expected(self):
        tester_only = get_tester_only_constraints()
        # spec-031 removed one tester-only diagnostic; ForcedGames is a remaining one
        assert 'ForcedGames' in tester_only

    def test_non_tester_only_have_solver_names(self):
        # spec-018: the Phase-6 Maitland-named alias entries
        # (`MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`) were deleted along
        # with their rules.
        # spec-010: `PHLRoundOnePlay` is obsolete — atom file kept on disk as
        # parity reference, `solver_class_names` emptied so legacy CLI flag
        # lookups don't accidentally redispatch it.
        ALIAS_NAMES = {
            'PHLRoundOnePlay',  # spec-010 obsolete; parity reference only
        }
        for name, info in CONSTRAINT_REGISTRY.items():
            if not info.tester_only and name not in ALIAS_NAMES:
                assert len(info.solver_class_names) >= 1, \
                    f"{name}: non-tester-only must have solver_class_names"


class TestConstraintInfoExtensions:
    """Phase 2: atom_group, required_helpers, forced_blocked_adjuster."""

    def test_default_atom_group_is_none(self):
        for info in CONSTRAINT_REGISTRY.values():
            # Pre-Phase-3 entries have atom_group = None
            if info.atom_group is not None:
                assert isinstance(info.atom_group, str)

    def test_default_required_helpers_is_empty_list(self):
        for info in CONSTRAINT_REGISTRY.values():
            assert isinstance(info.required_helpers, list)

    def test_default_adjuster_is_none(self):
        for info in CONSTRAINT_REGISTRY.values():
            if info.forced_blocked_adjuster is not None:
                assert callable(info.forced_blocked_adjuster)

    def test_required_helpers_are_in_catalog(self):
        bad = validate_required_helpers()
        assert bad == [], f"Constraints reference unknown helper-var kinds: {bad}"

    def test_get_adjuster_for_unknown_returns_none(self):
        assert get_adjuster('NonExistentConstraint') is None

    def test_get_atoms_in_group_returns_list(self):
        # No atoms in groups yet (Phase 3 will add them); function must still return a list
        result = get_atoms_in_group('SomeUnknownGroup')
        assert result == []

    def test_helper_var_catalog_is_populated(self):
        assert len(HELPER_VAR_CATALOG) > 0
        for kind in HELPER_VAR_CATALOG:
            assert isinstance(kind, str) and kind
