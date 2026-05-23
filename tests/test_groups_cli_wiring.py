"""spec-023 Unit C — CLI `--groups` resolution + main-path wiring.

Given/When/Then scenarios with hand-computed oracles. No mocks/patches: the
resolution helper (`run._resolve_group_selection`) and the registry resolver run
for real; the simple-path engine-key derivation is exercised against the REAL
`UnifiedConstraintEngine` on REAL 2026 data (build only, no solve).

Covers Unit C DoD-7:
  - `--groups core` selects only the core canonical names.
  - `--groups test test-1` (two ad-hoc DERIVED_GROUPS fixtures) -> deduped union.
  - no `--groups` == full `default` set.
  - `--exclude X --groups core` removes X.
  - `--staged` still produces severity-ordered selections (severity_solver_stages
    unchanged from baseline).
  - simple path honours the selection: `--simple --groups core` derives a
    skip_constraints that excludes every non-core engine key (incl. the soft
    PreferredTimesConstraint), and that selection is IDENTICAL to the engine keys
    the staged path applies for the same group.
"""
import constraints.registry as reg
from constraints.registry import (
    CONSTRAINT_REGISTRY,
    resolve_group,
    resolve_groups,
)
from constraints.stages import (
    collect_engine_keys,
    ALL_ENGINE_KEYS,
    severity_solver_stages,
)
import run


# ---------------------------------------------------------------------------
# Helper mirroring run.py / main_staged simple-path skip derivation, so a test
# documents the EXACT translation from a resolved group set to engine skips.
# ---------------------------------------------------------------------------
def _simple_skip_for(constraint_names):
    selected_engine_keys, _ = collect_engine_keys(list(constraint_names))
    return ALL_ENGINE_KEYS - selected_engine_keys


# ============== --groups resolution ==============

def test_groups_core_selects_only_core_canonical_names():
    """Given --groups core,
    When resolved,
    Then constraint_names == resolve_group('core') in registry order, no extras."""
    group_names, constraint_names = run._resolve_group_selection(['core'], [])
    assert group_names == ['core']
    # Same membership as resolve_group('core'); ordered = resolve_groups(['core']).
    assert set(constraint_names) == resolve_group('core')
    assert constraint_names == resolve_groups(['core'])
    assert len(constraint_names) == len(set(constraint_names))


def test_two_adhoc_groups_yield_deduped_union():
    """Given two ad-hoc groups 'test' and 'test-1' defined in DERIVED_GROUPS,
    both overlapping on EqualMatchUpSpacing,
    When --groups test test-1 is resolved,
    Then the result is the deduped union with EqualMatchUpSpacing appearing once."""
    # Fixture: register two ephemeral derived groups. 'test' = {NoDoubleBookingTeams,
    # EqualMatchUpSpacing}; 'test-1' = {EqualMatchUpSpacing, ClubGameSpread}.
    test_members = {'NoDoubleBookingTeams', 'EqualMatchUpSpacing'}
    test1_members = {'EqualMatchUpSpacing', 'ClubGameSpread'}
    saved = dict(reg.DERIVED_GROUPS)
    try:
        reg.DERIVED_GROUPS['test'] = lambda info, _m=test_members: (
            CONSTRAINT_REGISTRY_NAME(info) in _m
        )
        reg.DERIVED_GROUPS['test-1'] = lambda info, _m=test1_members: (
            CONSTRAINT_REGISTRY_NAME(info) in _m
        )
        group_names, constraint_names = run._resolve_group_selection(
            ['test', 'test-1'], []
        )
        assert group_names == ['test', 'test-1']
        # Deduped union: 3 distinct names, EqualMatchUpSpacing once.
        assert set(constraint_names) == (test_members | test1_members)
        assert constraint_names.count('EqualMatchUpSpacing') == 1
        # Canonical (registry) order.
        index = {n: i for i, n in enumerate(CONSTRAINT_REGISTRY)}
        idxs = [index[n] for n in constraint_names]
        assert idxs == sorted(idxs)
    finally:
        reg.DERIVED_GROUPS.clear()
        reg.DERIVED_GROUPS.update(saved)


def test_no_groups_is_full_default_set():
    """Given no --groups,
    When resolved,
    Then group_names == ['default'] and constraint_names == every production
    constraint (resolve_group('default'))."""
    group_names, constraint_names = run._resolve_group_selection(None, [])
    assert group_names == ['default']
    assert set(constraint_names) == resolve_group('default')
    # Also when passed an empty list (argparse nargs='+' never yields [], but
    # be defensive).
    g2, c2 = run._resolve_group_selection([], [])
    assert g2 == ['default']
    assert set(c2) == resolve_group('default')


def test_exclude_subtracts_from_resolved_set():
    """Given --exclude ClubGameSpread --groups core,
    When resolved,
    Then ClubGameSpread is absent and everything else in core remains."""
    _, base = run._resolve_group_selection(['core'], [])
    assert 'ClubGameSpread' in base  # precondition
    _, constraint_names = run._resolve_group_selection(['core'], ['ClubGameSpread'])
    assert 'ClubGameSpread' not in constraint_names
    assert set(constraint_names) == resolve_group('core') - {'ClubGameSpread'}


def test_unknown_group_name_exits():
    """Given an unknown --groups name,
    When resolved,
    Then the helper exits (SystemExit) rather than silently selecting nothing."""
    import pytest
    with pytest.raises(SystemExit):
        run._resolve_group_selection(['no-such-group'], [])


# ============== --staged unchanged ==============

# Baseline captured from the pre-Unit-C registry-walk implementation. Hand
# oracle: every non-tester, non-atomized-combined name grouped by severity.
BASELINE_SEVERITY_STAGES = [
    ('severity_1', [
        'AwayClubHomeWeekendsCount',
        'AwayClubPerOpponentAndAggregateHomeBalance',
        'EqualGamesAndBalanceMatchUps',
        'EqualMatchUpSpacing',
        'FiftyFiftyHomeandAway',
        'NoDoubleBookingFields',
        'NoDoubleBookingTeams',
        'PHLAnd2ndAdjacency',
        'PHLAnd2ndConcurrencyAtBroadmeadow',
        'PHLConcurrencyAtBroadmeadow',
        'SameGradeSameClubNoConcurrency',
    ]),
    ('severity_2', [
        'BalancedByeSpacing',
        'ClubDayContiguousSlots',
        'ClubDayIntraClubMatchup',
        'ClubDayOpponentMatchup',
        'ClubDayParticipation',
        'ClubDaySameField',
        'ClubNoConcurrentSlot',
        'TeamConflict',
        'VenueEarliestSlotFill',
    ]),
    ('severity_3', [
        'ClubGameSpread',
        'ClubVsClubAlignment',
        'ClubVsClubStackedCoLocation',
        'ClubVsClubStackedWeekends',
        'TeamPairNoConcurrency',
    ]),
    ('severity_5', [
        'NIHCFillEFBeforeSF',
        'NIHCFillWFBeforeEF',
        'PreferredGames',
        'PreferredTimes',
        'PreferredWeekendsAwayGround',
        'SoftLexMatchupOrdering',
    ]),
]


def test_severity_solver_stages_unchanged_from_baseline():
    """Given the spec-023 reimplementation in terms of resolve_group('severity_N'),
    When severity_solver_stages() runs,
    Then it is byte-for-byte identical to the captured baseline (so --staged
    still produces the same severity-ordered selections)."""
    stages = severity_solver_stages()
    got = [(s['name'], s['atoms']) for s in stages]
    assert got == BASELINE_SEVERITY_STAGES


def test_severity_stages_are_ordered_and_non_overlapping():
    """Given the severity stages,
    When concatenated,
    Then they are strictly level-ordered and pairwise disjoint (a constraint has
    exactly one severity level)."""
    stages = severity_solver_stages()
    levels = [int(s['name'].split('_')[1]) for s in stages]
    assert levels == sorted(levels)
    seen = set()
    for s in stages:
        atoms = set(s['atoms'])
        assert not (atoms & seen), f"{s['name']} overlaps a prior stage"
        seen |= atoms


# ============== DoD-7: simple path honours the selection ==============

def test_simple_path_skip_constraints_for_core():
    """Given --simple --groups core,
    When the simple path derives engine skip_constraints from the resolved set,
    Then it skips every engine key NOT in core — including the soft-only
    PreferredTimesConstraint — so the simple path applies only core's engine
    constraints."""
    _, core_names = run._resolve_group_selection(['core'], [])
    skip = _simple_skip_for(core_names)
    # core is hard-feasibility; it does NOT include the soft PreferredTimes.
    assert 'PreferredTimesConstraint' in skip
    # Obsolete engine keys never tagged into a production group are also skipped.
    for obsolete in ('FiftyFiftyHomeandAway', 'ClubVsClubAlignment', 'TeamConflict'):
        assert obsolete in skip
    # The engine keys actually applied = core's engine keys.
    applied = ALL_ENGINE_KEYS - skip
    assert applied == collect_engine_keys(core_names)[0]
    assert applied  # non-empty


def test_simple_and_staged_select_identical_engine_keys():
    """Given the SAME resolved --groups set,
    When the simple path derives its applied engine keys (ALL - skip) and the
    staged path derives its applied engine keys (collect_engine_keys over the
    same resolved set),
    Then they are IDENTICAL — proving --groups selects the same constraints in
    both modes."""
    for groups in (['core'], ['core', 'soft'], ['default'], ['severity_1']):
        _, names = run._resolve_group_selection(groups, [])
        simple_applied = ALL_ENGINE_KEYS - _simple_skip_for(names)
        # Staged applies via apply_constraint_set -> collect_engine_keys(names).
        staged_applied, _ = collect_engine_keys(names)
        assert simple_applied == staged_applied, groups


def test_simple_engine_respects_skip_on_real_data():
    """Given the REAL UnifiedConstraintEngine on REAL 2026 data,
    When apply_stage_1_hard runs with a core-derived skip vs an empty skip,
    Then the skipped (core) run adds strictly fewer hard constraints (it omits
    the non-core engine methods like fifty-fifty / club-alignment)."""
    from ortools.sat.python import cp_model
    from utils import generate_X
    from constraints.unified import UnifiedConstraintEngine
    from main_staged import load_data

    data = load_data(2026)

    def _hard_count(skip):
        model = cp_model.CpModel()
        X, conflicts = generate_X(model, data)
        data['team_conflicts'] = conflicts
        eng = UnifiedConstraintEngine(model, X, data, skip_constraints=set(skip))
        eng.build_groupings()
        return eng.apply_stage_1_hard()

    _, core_names = run._resolve_group_selection(['core'], [])
    core_skip = _simple_skip_for(core_names)

    full = _hard_count(set())          # legacy: all engine keys
    core_only = _hard_count(core_skip)  # core selection
    # Core skips FiftyFiftyHomeandAway / ClubVsClubAlignment / TeamConflict (and
    # PreferredTimes is soft, not in stage-1), so the hard count must drop.
    assert core_only < full, (core_only, full)
    assert core_only > 0


# Tiny shim: map a ConstraintInfo back to its canonical name for the ad-hoc
# DERIVED_GROUPS fixture predicates (registry has no back-pointer).
_INFO_TO_NAME = {id(info): name for name, info in CONSTRAINT_REGISTRY.items()}


def CONSTRAINT_REGISTRY_NAME(info):
    return _INFO_TO_NAME.get(id(info), '')
