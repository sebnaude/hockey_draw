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
# Hand-computed oracle for the `core_hard` set (spec-027 DoD-1).
# spec-033 Unit C: TeamConflict was REMOVED from core_hard — it is now a
# soft-only preference (see ORACLE_SOFT_MEMBERS), so it no longer reaches regen
# via the `core_hard` branch but via the `soft` branch instead.
# spec-033 Unit E: ClubNoConcurrentSlot was REMOVED from core_hard — it is now
# soft + slack (hard <=1 overlap +slack, push->0) carrying {core,
# critical_feasibility}. Unlike TeamConflict it gains NO soft/regen_soft tag, so
# it LEAVES regen entirely (no regen analogue). That drops the hand-listed
# core_hard set from 11 (post-spec-030) to 10 (post-Unit-C) to 9 members:
#   6 grouped-core physical/feasibility atoms that stay hard in regen, plus the
#   3 freeze pins (ForcedGames, BlockedGames, LockedPairings), all enforced at
#   generate_X. (Verified against the live registry: resolve_group('core_hard')
#   == this set, count 9.)
# ---------------------------------------------------------------------------
ORACLE_CORE_HARD = {
    'NoDoubleBookingTeams',
    'NoDoubleBookingFields',
    'EqualGamesAndBalanceMatchUps',
    'AwayClubPerOpponentAndAggregateHomeBalance',
    'PHLConcurrencyAtBroadmeadow',
    'SameGradeSameClubNoConcurrency',
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
# spec-032: the three symmetry breakers (NIHCFill*, SoftLexMatchupOrdering) left
# `soft` for the new `symmetry_breakers` group, so they are no longer here. They
# STILL reach regen — via the widened regen predicate's `symmetry_breakers`
# branch — see ORACLE_SYMMETRY_BREAKERS below.
ORACLE_SOFT_MEMBERS = {
    'TeamPairNoConcurrency',
    'PreferredTimes',
    'PreferredWeekendsAwayGround',
    'PreferredGames',
    # spec-033 Unit C: TeamConflict moved here from core_hard. It now carries
    # {'soft','soft_optimisation'} (soft-only preference, no hard component) and
    # so reaches regen via the `soft` branch instead of `core_hard`.
    'TeamConflict',
}

# spec-032: the always-on tie-breaker bundle. Tagged {symmetry_breakers} only,
# but the regen predicate was widened to include `symmetry_breakers`, so all
# three remain in the regen set (the convenor's regen output keeps the
# tie-breakers unless --no-symmetry-breakers).
ORACLE_SYMMETRY_BREAKERS = {
    'NIHCFillWFBeforeEF',
    'NIHCFillEFBeforeSF',
    'SoftLexMatchupOrdering',
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
    # spec-030: 12 -> 11 (deleted PHLAnd2ndConcurrencyAtBroadmeadow).
    # spec-033 Unit C: 11 -> 10 (TeamConflict softened out of core_hard).
    # spec-033 Unit E: 10 -> 9 (ClubNoConcurrentSlot softened out of core_hard;
    # it gains no regen analogue, so it leaves regen entirely).
    assert len(ORACLE_CORE_HARD) == 9


def test_regen_includes_every_core_hard_regen_soft_and_soft_member():
    """Given the regen derived group (DoD-5),
    When resolved,
    Then every core_hard member, every regen_soft member, every soft member, and
    (spec-032) every symmetry-breaker is present."""
    regen = set(resolve_groups(['regen']))
    missing_hard = ORACLE_CORE_HARD - regen
    missing_regen_soft = ORACLE_REGEN_SOFT - regen
    missing_soft = ORACLE_SOFT_MEMBERS - regen
    missing_symmetry = ORACLE_SYMMETRY_BREAKERS - regen
    assert not missing_hard, f"core_hard members missing from regen: {missing_hard}"
    assert not missing_regen_soft, f"regen_soft members missing: {missing_regen_soft}"
    assert not missing_soft, f"soft members missing from regen: {missing_soft}"
    assert not missing_symmetry, f"symmetry breakers missing from regen: {missing_symmetry}"


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
        | ORACLE_SYMMETRY_BREAKERS  # spec-032: widened regen predicate keeps them
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
    Then the core_hard-ONLY freeze pins are absent (they carry no `core`/`soft`
    tag), preserving fresh-season behaviour.

    spec-033 Unit C: TeamConflict was REMOVED from this exclusion list. It is no
    longer core_hard-only — it is now a soft-only preference carrying
    {'soft','soft_optimisation'}, so it IS pulled into the fresh-build default
    group via the `soft` branch (intended: declared conflict pairs should be
    discouraged in a fresh draw, not ignored). The three freeze pins remain
    core_hard-only and stay excluded."""
    default = set(resolve_groups(['default']))
    # The three pins are core_hard-only → never in a fresh build.
    for name in ('ForcedGames', 'BlockedGames', 'LockedPairings'):
        assert name not in default, f"{name} must not be in the fresh-build default group"
    # spec-033 Unit C: TeamConflict (now soft) IS in the fresh-build default group.
    assert 'TeamConflict' in default, "TeamConflict (now soft) must be in the fresh-build default group"
    # default is still exactly the fresh-build set. spec-032 widened
    # _is_fresh_build to {core, soft, spacing, symmetry_breakers}; spec-033
    # Unit B appended `bye_spacing` (BalancedByeSpacing peeled core->bye_spacing),
    # so the union must include all five dimensions. spec-033 Unit C softened
    # TeamConflict (core_hard-only -> {soft,soft_optimisation}), which ADDS it to
    # the default set (27 -> 28 atoms — a new member, not just a moved tag).
    assert default == (
        resolve_group('core') | resolve_group('soft')
        | resolve_group('spacing') | resolve_group('symmetry_breakers')
        | resolve_group('bye_spacing')
    )


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
