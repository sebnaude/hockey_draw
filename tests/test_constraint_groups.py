"""spec-023 Unit A — constraint-group resolver tests.

Given/When/Then scenarios with hand-computed oracles. No mocks/patches: every
assertion runs against the REAL `constraints.registry` (or, for the
deliberately-broken order case, a small fabricated registry-shaped fixture).

Oracles are computed by hand from the §1 tagging table reconciled against
`config/defaults.py::DEFAULT_STAGES` and written inline as literals.
"""
from dataclasses import dataclass, field
from typing import FrozenSet, List

import constraints.registry as reg
from constraints.registry import (
    CONSTRAINT_REGISTRY,
    DERIVED_GROUPS,
    resolve_group,
    resolve_groups,
    list_group_names,
    validate_group_order,
)


# ---------------------------------------------------------------------------
# Hand-computed oracle for resolve_groups(['core', 'soft']).
#
# `core` (19 members) and `soft` (4 members) are DISJOINT, so the union is
# exactly 23 names. Listed below in CONSTRAINT_REGISTRY insertion order (the
# canonical apply order) — derived by hand from the reconciled tags:
#   (spec-030: PHLAnd2ndConcurrencyAtBroadmeadow deleted -> 28 down to 27.)
#   (spec-032: EqualMatchUpSpacing peeled core->{spacing}; the three symmetry
#    breakers peeled soft->{symmetry_breakers}. core 20->19, soft 7->4, so the
#    core∪soft union drops from 27 to 23. The four atoms still reach `default`
#    via the widened _is_fresh_build predicate, but are NOT in core or soft.)
#   critical_feasibility (core): NoDoubleBookingTeams, NoDoubleBookingFields,
#       EqualGamesAndBalanceMatchUps, PHLAnd2ndAdjacency,
#       PHLConcurrencyAtBroadmeadow,
#       BalancedByeSpacing, SameGradeSameClubNoConcurrency,
#       ClubNoConcurrentSlot, VenueEarliestSlotFill
#   home_away_balance (core): AwayClubHomeWeekendsCount,
#       AwayClubPerOpponentAndAggregateHomeBalance
#   club_alignment (core): ClubVsClubStackedWeekends, ClubVsClubStackedCoLocation
#   club_day (core): ClubDayParticipation, ClubDayIntraClubMatchup,
#       ClubDayOpponentMatchup, ClubDaySameField, ClubDayContiguousSlots,
#       ClubGameSpread
#   soft_optimisation (soft): TeamPairNoConcurrency, PreferredTimes,
#       PreferredWeekendsAwayGround, PreferredGames
# Then sorted into registry insertion order:
# ---------------------------------------------------------------------------
ORACLE_CORE_PLUS_SOFT: List[str] = [
    'NoDoubleBookingTeams',
    'NoDoubleBookingFields',
    'EqualGamesAndBalanceMatchUps',
    'AwayClubHomeWeekendsCount',
    'AwayClubPerOpponentAndAggregateHomeBalance',
    'PHLAnd2ndAdjacency',
    'PHLConcurrencyAtBroadmeadow',
    'BalancedByeSpacing',
    'ClubDayParticipation',
    'ClubDayIntraClubMatchup',
    'ClubDayOpponentMatchup',
    'ClubDaySameField',
    'ClubDayContiguousSlots',
    'SameGradeSameClubNoConcurrency',
    'TeamPairNoConcurrency',
    'ClubVsClubStackedWeekends',
    'ClubVsClubStackedCoLocation',
    'ClubGameSpread',
    'ClubNoConcurrentSlot',
    'VenueEarliestSlotFill',
    'PreferredTimes',
    'PreferredWeekendsAwayGround',
    'PreferredGames',
]


def test_resolve_groups_core_soft_equals_hand_oracle():
    """Given the real registry tagged per the spec-030/spec-032 retags,
    When resolve_groups(['core','soft']) is called,
    Then it equals the hand-computed union, with NO duplicates, in registry order."""
    result = resolve_groups(['core', 'soft'])
    # Exact list match (order + membership) against the hand oracle.
    assert result == ORACLE_CORE_PLUS_SOFT
    # No duplicates.
    assert len(result) == len(set(result))
    # 19 core + 4 soft = 23 (spec-032: was 27 before peeling EqualMatchUpSpacing
    # to {spacing} and the three symmetry breakers to {symmetry_breakers}).
    assert len(result) == 23


# ---------------------------------------------------------------------------
# spec-032 hand oracles: the two new explicit groups + the post-retag core set.
# ---------------------------------------------------------------------------
ORACLE_SYMMETRY_BREAKERS = {
    'NIHCFillWFBeforeEF',
    'NIHCFillEFBeforeSF',
    'SoftLexMatchupOrdering',
}

# core after spec-030 (removed PHLAnd2ndConcurrencyAtBroadmeadow) and spec-032
# (removed EqualMatchUpSpacing -> {spacing}): 19 members, hand-listed.
ORACLE_CORE_AFTER_RETAG = {
    'NoDoubleBookingTeams',
    'NoDoubleBookingFields',
    'EqualGamesAndBalanceMatchUps',
    'AwayClubHomeWeekendsCount',
    'AwayClubPerOpponentAndAggregateHomeBalance',
    'PHLAnd2ndAdjacency',
    'PHLConcurrencyAtBroadmeadow',
    'BalancedByeSpacing',
    'ClubDayParticipation',
    'ClubDayIntraClubMatchup',
    'ClubDayOpponentMatchup',
    'ClubDaySameField',
    'ClubDayContiguousSlots',
    'SameGradeSameClubNoConcurrency',
    'ClubVsClubStackedWeekends',
    'ClubVsClubStackedCoLocation',
    'ClubGameSpread',
    'ClubNoConcurrentSlot',
    'VenueEarliestSlotFill',
}


def test_resolve_group_symmetry_breakers():
    """Given the spec-032 retag,
    When resolve_group('symmetry_breakers') is called,
    Then it equals exactly the three always-on tie-breaker atoms,
    and none of them remain in `soft`."""
    assert resolve_group('symmetry_breakers') == ORACLE_SYMMETRY_BREAKERS
    soft = resolve_group('soft')
    assert not (ORACLE_SYMMETRY_BREAKERS & soft)


def test_resolve_group_spacing():
    """Given the spec-032 retag,
    When resolve_group('spacing') is called,
    Then it equals exactly {'EqualMatchUpSpacing'}, which is no longer in `core`,
    but is still selected by the derived `severity_1` group (severity unchanged)."""
    assert resolve_group('spacing') == {'EqualMatchUpSpacing'}
    assert 'EqualMatchUpSpacing' not in resolve_group('core')
    assert 'EqualMatchUpSpacing' in resolve_group('severity_1')


def test_core_after_retag_equals_hand_oracle():
    """Given the spec-032 retag,
    When resolve_group('core') is called,
    Then it equals the hand-listed 19-member post-retag core set
    (no EqualMatchUpSpacing, no PHLAnd2ndConcurrencyAtBroadmeadow)."""
    assert resolve_group('core') == ORACLE_CORE_AFTER_RETAG
    assert len(ORACLE_CORE_AFTER_RETAG) == 19


def test_overlapping_groups_yield_one_copy_each():
    """Given two requested groups that BOTH contain ClubDayParticipation
    (core and club_day both tag it),
    When resolve_groups unions them,
    Then ClubDayParticipation appears exactly once."""
    # ClubDayParticipation is tagged {'core', 'club_day'} — reachable via both.
    assert 'club_day' in CONSTRAINT_REGISTRY['ClubDayParticipation'].groups
    assert 'core' in CONSTRAINT_REGISTRY['ClubDayParticipation'].groups
    result = resolve_groups(['core', 'club_day'])
    assert result.count('ClubDayParticipation') == 1
    # And the union has no duplicates at all.
    assert len(result) == len(set(result))
    # core is a superset of the production club_day members here, so the union
    # equals core exactly (club_day adds nothing new).
    assert set(result) == resolve_group('core')


def test_resolve_groups_output_is_registry_insertion_order():
    """Given any group selection,
    When resolved,
    Then the returned canonical names are in strictly-increasing registry index."""
    index = {name: i for i, name in enumerate(CONSTRAINT_REGISTRY)}
    for selection in (['core', 'soft'], ['soft', 'core'], ['club_day', 'core'],
                      ['severity_1', 'soft']):
        result = resolve_groups(selection)
        indices = [index[n] for n in result]
        assert indices == sorted(indices), (selection, indices)
        # Strictly monotonic (no dupes -> strictly increasing).
        assert all(a < b for a, b in zip(indices, indices[1:])), selection


def test_request_order_does_not_change_output():
    """Given the same two groups requested in either order,
    When resolved,
    Then the output is identical (canonical order, not request order)."""
    assert resolve_groups(['core', 'soft']) == resolve_groups(['soft', 'core'])


def test_derived_severity_1_matches_severity_metadata():
    """Given the derived group severity_1,
    When resolved,
    Then it equals exactly the set of registry entries with severity_level == 1."""
    expected = {
        name for name, info in CONSTRAINT_REGISTRY.items()
        if info.severity_level == 1
    }
    assert resolve_group('severity_1') == expected
    # The predicate itself selects on severity_level == 1.
    pred = DERIVED_GROUPS['severity_1']
    for name, info in CONSTRAINT_REGISTRY.items():
        assert pred(info) == (info.severity_level == 1)


def test_default_group_is_every_production_constraint():
    """Given the derived 'default'/'all'/'production' groups,
    When resolved,
    Then each equals every FRESH-SEASON-BUILD constraint
    = exactly core ∪ soft ∪ spacing ∪ symmetry_breakers.

    spec-027 changed default/all/production from `bool(info.groups)` to
    `core ∪ soft`, keeping the `core_hard`-only freeze pins + TeamConflict and the
    `regen_soft` atoms OUT of a fresh build. spec-032 then widened the predicate
    to also include `spacing` and `symmetry_breakers`, because EqualMatchUpSpacing
    moved core->spacing and the three symmetry breakers moved soft->symmetry_breakers
    — without the widening they would silently drop from a fresh build. The
    membership is unchanged from before the retag (still 27 atoms); only the tag
    routing moved."""
    production = {
        name for name, info in CONSTRAINT_REGISTRY.items()
        if {'core', 'soft', 'spacing', 'symmetry_breakers'} & info.groups
    }
    assert resolve_group('default') == production
    assert resolve_group('all') == production
    assert resolve_group('production') == production
    # And it is exactly core ∪ soft ∪ spacing ∪ symmetry_breakers.
    assert production == (
        resolve_group('core') | resolve_group('soft')
        | resolve_group('spacing') | resolve_group('symmetry_breakers')
    )
    # spec-032: the retag is membership-preserving — still 27 fresh-build atoms.
    assert len(production) == 27


def test_list_group_names_includes_explicit_and_derived():
    """Given the registry,
    When listing group names,
    Then both explicit tags and every derived name are present, deduped."""
    names = list_group_names()
    assert len(names) == len(set(names))
    for explicit in ('core', 'soft', 'critical_feasibility', 'club_day',
                     'home_away_balance', 'club_alignment', 'soft_optimisation',
                     'spacing', 'symmetry_breakers'):
        assert explicit in names
    for derived in ('severity_1', 'severity_5', 'default', 'all', 'production'):
        assert derived in names


def test_order_validator_passes_on_real_registry():
    """Given the real registry,
    When validate_group_order runs,
    Then there are no producer-after-consumer violations."""
    assert validate_group_order() == []


def test_producer_precedes_consumer_for_cvc_stack():
    """Given the cvc_stack_play producer/consumer pair,
    When comparing registry indices,
    Then ClubVsClubStackedWeekends precedes ClubVsClubStackedCoLocation."""
    index = {name: i for i, name in enumerate(CONSTRAINT_REGISTRY)}
    assert index['ClubVsClubStackedWeekends'] < index['ClubVsClubStackedCoLocation']


# ---------------------------------------------------------------------------
# Deliberately-swapped fixture: a consumer placed BEFORE its producer must be
# flagged. We rebuild a tiny registry-shaped dict and run the SAME validation
# logic against it by monkey-free dependency injection: a local clone of the
# validator that reads a supplied registry. To avoid duplicating logic, we
# temporarily swap the module-level registry + index, run the real validator,
# and restore — no mocks/patches library, just attribute save/restore.
# ---------------------------------------------------------------------------
@dataclass
class _FakeInfo:
    required_helpers: List[str] = field(default_factory=list)
    groups: FrozenSet[str] = frozenset()
    severity_level: int = 1


def test_order_validator_flags_consumer_before_producer():
    """Given a fabricated registry where the consumer of a helper var is
    inserted BEFORE its producer,
    When validate_group_order runs,
    Then it returns a non-empty list of violations."""
    # Consumer 'BadConsumer' (declares helper 'shared_kind') appears first;
    # 'GoodProducer' (also declares 'shared_kind') appears second. With the
    # first-declarer-is-producer rule, BadConsumer becomes the de-facto producer,
    # so this fixture instead triggers the EXPLICIT producer/consumer check.
    fake_registry = {
        'BadConsumer': _FakeInfo(required_helpers=['shared_kind']),
        'GoodProducer': _FakeInfo(required_helpers=['shared_kind']),
    }
    fake_index = {name: i for i, name in enumerate(fake_registry)}
    fake_pairs = [('GoodProducer', 'BadConsumer', ['shared_kind'])]

    saved_registry = reg.CONSTRAINT_REGISTRY
    saved_index = reg._REGISTRY_INDEX
    saved_pairs = reg.HELPER_VAR_PRODUCER_CONSUMER
    try:
        reg.CONSTRAINT_REGISTRY = fake_registry
        reg._REGISTRY_INDEX = fake_index
        reg.HELPER_VAR_PRODUCER_CONSUMER = fake_pairs
        violations = validate_group_order()
    finally:
        reg.CONSTRAINT_REGISTRY = saved_registry
        reg._REGISTRY_INDEX = saved_index
        reg.HELPER_VAR_PRODUCER_CONSUMER = saved_pairs

    assert violations, "expected a violation: GoodProducer (idx 1) after BadConsumer (idx 0)"
    assert any('GoodProducer' in v and 'BadConsumer' in v for v in violations)
    # Sanity: the real registry is restored and still clean.
    assert validate_group_order() == []
