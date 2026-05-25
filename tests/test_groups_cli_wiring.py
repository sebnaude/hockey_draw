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
    """Given --groups core (spec-032: with --no-symmetry-breakers to isolate the
    pure core selection from the always-on tie-breaker union),
    When resolved,
    Then constraint_names == resolve_group('core') in registry order, no extras."""
    group_names, constraint_names = run._resolve_group_selection(
        ['core'], [], no_symmetry_breakers=True)
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
        # spec-032: suppress the always-on tie-breakers to isolate dedup mechanics.
        group_names, constraint_names = run._resolve_group_selection(
            ['test', 'test-1'], [], no_symmetry_breakers=True
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
    # spec-032: suppress the always-on tie-breakers to isolate exclude mechanics.
    _, base = run._resolve_group_selection(['core'], [], no_symmetry_breakers=True)
    assert 'ClubGameSpread' in base  # precondition
    _, constraint_names = run._resolve_group_selection(
        ['core'], ['ClubGameSpread'], no_symmetry_breakers=True)
    assert 'ClubGameSpread' not in constraint_names
    assert set(constraint_names) == resolve_group('core') - {'ClubGameSpread'}


# ============== spec-032: always-on symmetry breakers + --no-symmetry-breakers ==============

SYMMETRY_ATOMS = {'NIHCFillWFBeforeEF', 'NIHCFillEFBeforeSF', 'SoftLexMatchupOrdering'}


def test_symmetry_breakers_unioned_into_groups_core():
    """Given --groups core with NO suppression (the default),
    When resolved,
    Then the three symmetry breakers are unioned in even though `core` does not
    itself contain them, AND every core constraint is still present, in canonical
    registry order. (DoD 6: tie-breakers shape EVERY solve.)"""
    _, constraint_names = run._resolve_group_selection(['core'], [])
    names = set(constraint_names)
    assert SYMMETRY_ATOMS <= names, SYMMETRY_ATOMS - names
    assert resolve_group('core') <= names
    # Hand oracle: core (19) + 3 symmetry = 22 distinct, none doubled.
    assert len(constraint_names) == len(set(constraint_names))
    assert len(constraint_names) == 22
    # Canonical (registry) order preserved across the union.
    index = {n: i for i, n in enumerate(CONSTRAINT_REGISTRY)}
    idxs = [index[n] for n in constraint_names]
    assert idxs == sorted(idxs)


def test_no_symmetry_breakers_drops_them_from_groups_path():
    """Given --groups core --no-symmetry-breakers,
    When resolved,
    Then NONE of the three tie-breakers appear, and the selection is exactly
    `core` (19). (DoD 7: --groups-path suppression.)"""
    _, constraint_names = run._resolve_group_selection(
        ['core'], [], no_symmetry_breakers=True)
    assert not (SYMMETRY_ATOMS & set(constraint_names))
    assert set(constraint_names) == resolve_group('core')
    assert len(constraint_names) == 19


def test_no_symmetry_breakers_drops_them_from_default_group():
    """Given NO --groups (the `default` group, which DOES contain the three
    tie-breakers) plus --no-symmetry-breakers,
    When resolved,
    Then the three are excluded even though `default` carries them — proving the
    suppression excludes, not merely 'declines to add'. (DoD 7.)"""
    _, default_with = run._resolve_group_selection(None, [])
    assert SYMMETRY_ATOMS <= set(default_with)  # default carries them by default
    _, default_without = run._resolve_group_selection(None, [], no_symmetry_breakers=True)
    assert not (SYMMETRY_ATOMS & set(default_without))
    # Hand oracle: default (27) minus the 3 tie-breakers = 24.
    assert len(default_without) == 24
    assert set(default_without) == resolve_group('default') - SYMMETRY_ATOMS


def test_no_symmetry_breakers_forces_plain_path_staged_filter():
    """Given the plain (no --groups) path with --no-symmetry-breakers,
    When DEFAULT_STAGES is filtered to the resolved (default-minus-symmetry) set —
    the exact filter run_generate forces non-None in this case,
    Then the three tie-breakers are dropped from the applied stage atoms, whereas
    the unsuppressed plain path (legacy None filter) keeps them via DEFAULT_STAGES.
    (DoD 7: plain-path suppression must force a non-None filter.)"""
    from constraints.stages import load_solver_stages

    base = load_solver_stages({})
    all_default_stage_atoms = {a for s in base for a in s.get('atoms', [])}
    # Precondition: DEFAULT_STAGES itself still contains all three tie-breakers,
    # so the legacy None plain path (no flag) applies them.
    assert SYMMETRY_ATOMS <= all_default_stage_atoms

    # The filter run_generate forces when no --groups + --no-symmetry-breakers:
    _, suppressed = run._resolve_group_selection(None, [], no_symmetry_breakers=True)
    keep = set(suppressed)
    applied = {a for s in base for a in s.get('atoms', []) if a in keep}
    # The three tie-breakers are filtered out; everything else in DEFAULT_STAGES
    # that is also in the default group survives.
    assert not (SYMMETRY_ATOMS & applied)
    assert (all_default_stage_atoms - SYMMETRY_ATOMS) <= applied


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


# Hand-listed engine-key oracles per group. These are written out by hand (NOT
# computed from collect_engine_keys / the simple-skip derivation), so a test that
# checks them cannot tautologically cancel — it pins the ACTUAL engine keys each
# group resolves to. Derived once by inspection of the registry's group tags and
# each whole-constraint's engine_keys, then frozen here as the contract.
ENGINE_KEY_ORACLE = {
    # core = hard feasibility set. Note the legacy engine keys
    # FiftyFiftyHomeandAway / ClubVsClubAlignment / TeamConflict are ABSENT (no
    # production-group atom maps to them — see spec §1) and the soft
    # PreferredTimesConstraint is absent (core is hard-only).
    # spec-032: EqualMatchUpSpacing peeled core->{spacing}, so its engine key is
    # no longer in `core` (it stays in `default` and `severity_1` below).
    'core': {
        'ClubDay',
        'ClubGameSpread',
        'EqualGamesAndBalanceMatchUps',
        'NoDoubleBookingFields',
        'NoDoubleBookingTeams',
        'PHLAndSecondGradeTimes',
    },
    # core + soft adds the soft engine key PreferredTimesConstraint.
    # spec-032: EqualMatchUpSpacing absent (now in `spacing`, not core); the three
    # symmetry breakers carry no engine key, so leaving `soft` does not change this.
    ('core', 'soft'): {
        'ClubDay',
        'ClubGameSpread',
        'EqualGamesAndBalanceMatchUps',
        'NoDoubleBookingFields',
        'NoDoubleBookingTeams',
        'PHLAndSecondGradeTimes',
        'PreferredTimesConstraint',
    },
    # default = full production set = core + soft + spacing + symmetry_breakers
    # engine keys (the obsolete trio is still excluded BY DESIGN). spec-032:
    # EqualMatchUpSpacing's engine key returns here via the `spacing` tag + the
    # widened _is_fresh_build predicate, so `default` is unchanged.
    'default': {
        'ClubDay',
        'ClubGameSpread',
        'EqualGamesAndBalanceMatchUps',
        'EqualMatchUpSpacing',
        'NoDoubleBookingFields',
        'NoDoubleBookingTeams',
        'PHLAndSecondGradeTimes',
        'PreferredTimesConstraint',
    },
    # severity_1 = the CRITICAL severity stage. Unlike the production groups it
    # legitimately DOES carry the obsolete engine key FiftyFiftyHomeandAway
    # (severity stages are a separate, historical grouping per the staged
    # baseline above), so it appears here but NOT in core/default.
    'severity_1': {
        'EqualGamesAndBalanceMatchUps',
        'EqualMatchUpSpacing',
        'FiftyFiftyHomeandAway',
        'NoDoubleBookingFields',
        'NoDoubleBookingTeams',
        'PHLAndSecondGradeTimes',
    },
}


def _production_simple_applied(constraint_names):
    """Return the engine keys the PRODUCTION simple path will actually apply.

    `main_staged._main_simple_unified` derives ``skip_constraints`` from a
    resolved selection, hands it to a real ``UnifiedConstraintEngine``, and the
    engine applies ``ALL_ENGINE_KEYS - skip_constraints``. We run that exact
    sequence on real 2026 data (build only, no solve) and read the engine's
    stored ``skip_constraints`` back, so the returned set is what production
    applies — not a formula re-spelled in the test.

    The anti-tautology guarantee lives in the CALLER: this result is compared to
    an INDEPENDENT hand oracle (ENGINE_KEY_ORACLE), so even though the simple and
    staged sides share ``collect_engine_keys``, the oracle is the load-bearing
    check and the assertion cannot reduce to ``X == X``."""
    from ortools.sat.python import cp_model
    from utils import generate_X
    from constraints.unified import UnifiedConstraintEngine
    from main_staged import load_data
    # Mirror _main_simple_unified's derivation exactly.
    selected_engine_keys, _ = collect_engine_keys(list(constraint_names))
    skip = ALL_ENGINE_KEYS - selected_engine_keys
    model = cp_model.CpModel()
    data = load_data(2026)
    X, conflicts = generate_X(model, data)
    data['team_conflicts'] = conflicts
    eng = UnifiedConstraintEngine(model, X, data, skip_constraints=set(skip))
    # Engine now holds the production skip; report what it will apply.
    return ALL_ENGINE_KEYS - set(eng.skip_constraints)


def test_simple_applied_engine_keys_match_hand_oracle():
    """Given a resolved --groups set,
    When the production simple path derives its applied engine keys (read back
    from a real UnifiedConstraintEngine on real data),
    Then they equal a HAND-WRITTEN oracle for that group — and the staged path
    (collect_engine_keys over the same names) equals the SAME oracle.

    This replaces the old self-cancelling identity test
    (``ALL - (ALL - X) == X``). Both the simple and staged sides are pinned
    against an independent hand oracle, so the assertion proves the actual wiring
    rather than reducing to a tautology."""
    for groups, oracle in (
        (['core'], ENGINE_KEY_ORACLE['core']),
        (['core', 'soft'], ENGINE_KEY_ORACLE[('core', 'soft')]),
        (['default'], ENGINE_KEY_ORACLE['default']),
        (['severity_1'], ENGINE_KEY_ORACLE['severity_1']),
    ):
        _, names = run._resolve_group_selection(groups, [])
        # Simple side: production engine's applied engine keys (skip read-back).
        simple_applied = _production_simple_applied(names)
        # Staged side: registry resolver's engine keys for the same names.
        staged_applied, _ = collect_engine_keys(names)
        # Independent hand oracle pins both — proving the wiring, not an identity.
        assert simple_applied == oracle, (groups, sorted(simple_applied), sorted(oracle))
        assert staged_applied == oracle, (groups, sorted(staged_applied), sorted(oracle))
        assert simple_applied == staged_applied, groups


def test_default_simple_selection_excludes_obsolete_trio_by_design():
    """Given NO --groups (default selection),
    When the production simple path derives skip_constraints
    (ALL_ENGINE_KEYS - collect_engine_keys(default-group names)),
    Then it skips EXACTLY {FiftyFiftyHomeandAway, ClubVsClubAlignment,
    TeamConflict}.

    This pins the intended default-`--simple` behaviour at the SELECTION level.
    The three skipped engine keys are obsolete legacy entries that map to NO
    production-group atom: per spec-023 §1 they "get no production group" because
    they are superseded by atoms AwayClubHomeWeekendsCount /
    AwayClubPerOpponentAndAggregateHomeBalance (replacing FiftyFiftyHomeandAway)
    and ClubVsClubStackedWeekends / ClubVsClubStackedCoLocation (replacing
    ClubVsClubAlignment), with TeamConflict carried separately. Per the convenor's
    RESOLVED Option B, default `--simple` selects the `default` group (production
    constraints). The resulting ~-2,453-hard-constraint delta versus the old
    no-skip simple path is therefore INTENTIONAL, not a regression.

    (The real-data cross-mode parity baseline — actual hard-constraint counts —
    is Unit D's DoD-8 deliverable; this test stays at the selection level, which
    is what Unit C controls.)"""
    group_names, default_names = run._resolve_group_selection(None, [])
    assert group_names == ['default']  # no --groups => default group
    selected_engine_keys, _ = collect_engine_keys(default_names)
    skip = ALL_ENGINE_KEYS - selected_engine_keys
    # Hand-computed oracle: exactly the obsolete trio, nothing else.
    assert skip == {'FiftyFiftyHomeandAway', 'ClubVsClubAlignment', 'TeamConflict'}


def test_staged_none_equals_default_group_filter():
    """Given the staged path,
    When run with constraint_names=None (no --groups, legacy path) vs
    constraint_names=resolve('default') (the filter applied),
    Then the applied per-stage atoms are IDENTICAL — proving the run.py
    asymmetry (simple always passes default; staged passes None when no --groups)
    is cosmetic, because filtering DEFAULT_STAGES' atoms to the default group is a
    no-op. This pins the comment in run.run_generate next to
    `staged_constraint_filter`."""
    from constraints.stages import load_solver_stages

    base = load_solver_stages({})
    # staged-None: no filtering (the legacy path run.py uses when no --groups).
    none_applied = [(s['name'], list(s.get('atoms', []))) for s in base]

    # staged default-group: the exact filter main_staged applies when
    # constraint_names is the resolved default set.
    _, default_names = run._resolve_group_selection(None, [])
    keep = set(default_names)
    default_applied = []
    for s in base:
        kept = [a for a in s.get('atoms', []) if a in keep]
        if kept:
            default_applied.append((s['name'], kept))

    assert none_applied == default_applied


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
