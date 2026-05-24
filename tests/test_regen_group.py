"""spec-027 Unit A — `core_hard` tags + `regen` derived group.

Given/When/Then scenarios with hand-computed oracles. No mocks/patches: every
assertion runs against the REAL `constraints.registry`.

Oracles are computed by hand from spec-027 DoD-1 (the core-hard set) reconciled
against the registry, and written inline as literals.

Unit A defines the tags + group; the `regen_soft` atoms land in Unit B. So the
inclusion side of DoD-5 ("every regen_soft member present") is asserted in
`tests/atoms/` once those atoms exist. This file pins the two halves that are
already decidable in Unit A: the exact `core_hard` membership, that the `regen`
group contains every `core_hard` + every `soft` member, and that it EXCLUDES the
normal hard atoms of the soon-to-be-softened constraints.
"""
from constraints.registry import (
    CONSTRAINT_REGISTRY,
    resolve_group,
    resolve_groups,
    list_group_names,
    validate_group_order,
)


# ---------------------------------------------------------------------------
# Hand-computed oracle for the `core_hard` set (spec-027 DoD-1). 12 members:
#   8 grouped-core physical/feasibility atoms that stay hard in regen, plus
#   TeamConflict (data-driven hard pair rule) and the 3 freeze pins
#   (ForcedGames, BlockedGames, LockedPairings), all enforced at generate_X.
# ---------------------------------------------------------------------------
ORACLE_CORE_HARD = {
    'NoDoubleBookingTeams',
    'NoDoubleBookingFields',
    'EqualGamesAndBalanceMatchUps',
    'AwayClubPerOpponentAndAggregateHomeBalance',
    'PHLConcurrencyAtBroadmeadow',
    'SameGradeSameClubNoConcurrency',
    'ClubNoConcurrentSlot',
    'TeamConflict',
    'ForcedGames',
    'BlockedGames',
    'LockedPairings',
}

# The normal hard atoms of the constraints regen SOFTENS. Each is a production
# constraint carrying only `{core, ...}` (or, for the two engine keys, no
# `core_hard`/`regen_soft`/`soft` tag), so the `regen` predicate must EXCLUDE
# all of them — regen reaches their behaviour via a RegenSoft analogue instead.
ORACLE_SOFTENED_HARD_ATOMS = {
    'PHLAnd2ndAdjacency',
    'EqualMatchUpSpacing',
    'BalancedByeSpacing',
    'ClubDayParticipation',
    'ClubDayIntraClubMatchup',
    'ClubDayOpponentMatchup',
    'ClubDaySameField',
    'ClubDayContiguousSlots',
    'ClubVsClubStackedWeekends',
    'ClubVsClubStackedCoLocation',
    'ClubGameSpread',
    'AwayClubHomeWeekendsCount',
    'VenueEarliestSlotFill',
}

# The always-soft group: stays soft in regen exactly as in a normal build.
ORACLE_SOFT_MEMBERS = {
    'TeamPairNoConcurrency',
    'NIHCFillWFBeforeEF',
    'NIHCFillEFBeforeSF',
    'PreferredTimes',
    'SoftLexMatchupOrdering',
    'PreferredWeekendsAwayGround',
    'PreferredGames',
}

# The 13 spec-027 regen soft-analogue atoms (Unit B). The `regen` group selects
# these IN PLACE OF the hard atoms in ORACLE_SOFTENED_HARD_ATOMS.
ORACLE_REGEN_SOFT = {
    'PHLAnd2ndAdjacencyRegenSoft',
    'AwayClubHomeWeekendsCountRegenSoft',
    'ClubVsClubStackedWeekendsRegenSoft',
    'ClubVsClubStackedCoLocationRegenSoft',
    'EqualMatchUpSpacingRegenSoft',
    'BalancedByeSpacingRegenSoft',
    'ClubDayParticipationRegenSoft',
    'ClubDayIntraClubMatchupRegenSoft',
    'ClubDayOpponentMatchupRegenSoft',
    'ClubDaySameFieldRegenSoft',
    'ClubDayContiguousSlotsRegenSoft',
    'ClubGameSpreadRegenSoft',
    'VenueEarliestSlotFillRegenSoft',
}


def test_core_hard_membership_equals_hand_oracle():
    """Given the registry tagged per spec-027 DoD-1,
    When resolve_group('core_hard') is called,
    Then it equals exactly the hand-listed 12-member core-hard set."""
    assert resolve_group('core_hard') == ORACLE_CORE_HARD
    assert len(ORACLE_CORE_HARD) == 11  # spec-030: was 12 before deleting PHLAnd2ndConcurrencyAtBroadmeadow


def test_regen_includes_every_core_hard_regen_soft_and_soft_member():
    """Given the regen derived group (DoD-5),
    When resolved,
    Then every core_hard member, every regen_soft member, and every soft member
    is present."""
    regen = set(resolve_groups(['regen']))
    missing_hard = ORACLE_CORE_HARD - regen
    missing_regen_soft = ORACLE_REGEN_SOFT - regen
    missing_soft = ORACLE_SOFT_MEMBERS - regen
    assert not missing_hard, f"core_hard members missing from regen: {missing_hard}"
    assert not missing_regen_soft, f"regen_soft members missing: {missing_regen_soft}"
    assert not missing_soft, f"soft members missing from regen: {missing_soft}"


def test_regen_soft_group_has_exactly_13_members():
    """Given the regen_soft tag,
    When resolved,
    Then it is exactly the 13 spec-027 soft-analogue atoms (DoD-10 count)."""
    assert resolve_group('regen_soft') == ORACLE_REGEN_SOFT
    assert len(ORACLE_REGEN_SOFT) == 13


def test_regen_excludes_softened_hard_atoms():
    """Given the regen derived group,
    When resolved,
    Then NONE of the soon-to-be-softened normal hard atoms appear — regen reaches
    their behaviour via a RegenSoft analogue, never the hard original."""
    regen = set(resolve_groups(['regen']))
    leaked = ORACLE_SOFTENED_HARD_ATOMS & regen
    assert not leaked, (
        f"these hard atoms must be softened (absent) in regen but leaked in: {leaked}"
    )


def test_regen_equals_core_hard_plus_regen_soft_plus_soft():
    """Given the full registry (Unit A tags + Unit B atoms),
    When resolve_groups(['regen']) is called,
    Then it equals EXACTLY core_hard ∪ regen_soft ∪ soft — no more, no less. In
    particular it contains none of the softened hard atoms (those are replaced by
    their RegenSoft analogue) and nothing outside the three tagged dimensions."""
    assert set(resolve_groups(['regen'])) == (
        ORACLE_CORE_HARD | ORACLE_REGEN_SOFT | ORACLE_SOFT_MEMBERS
    )


def test_regen_output_is_registry_insertion_order():
    """Given the regen selection,
    When resolved,
    Then the returned names are in strictly-increasing registry index (no dupes)."""
    index = {name: i for i, name in enumerate(CONSTRAINT_REGISTRY)}
    result = resolve_groups(['regen'])
    indices = [index[n] for n in result]
    assert indices == sorted(indices)
    assert all(a < b for a, b in zip(indices, indices[1:]))


def test_default_group_excludes_core_hard_only_and_regen_soft():
    """Given the fresh-build default group,
    When resolved,
    Then the core_hard-ONLY freeze pins + TeamConflict are absent (they carry no
    `core`/`soft` tag), preserving fresh-season behaviour."""
    default = set(resolve_groups(['default']))
    # TeamConflict + the three pins are core_hard-only → never in a fresh build.
    for name in ('TeamConflict', 'ForcedGames', 'BlockedGames', 'LockedPairings'):
        assert name not in default, f"{name} must not be in the fresh-build default group"
    # default is still exactly core ∪ soft.
    assert default == resolve_group('core') | resolve_group('soft')


def test_regen_is_a_listed_group_name():
    """Given the registry,
    When listing group names,
    Then both the explicit `core_hard` tag and the derived `regen` group appear."""
    names = list_group_names()
    assert 'core_hard' in names
    assert 'regen' in names
    assert len(names) == len(set(names))


def test_order_validator_still_clean_after_spec_027_tags():
    """Given the spec-027 tag additions,
    When validate_group_order runs,
    Then there are still no producer-after-consumer violations."""
    assert validate_group_order() == []
