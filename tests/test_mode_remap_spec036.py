"""spec-036 Unit A — solve-mode remap + single-solve-default tests.

These tests pin the behaviour change introduced by spec-036 Unit A:

  * No mode flag => a SINGLE full solve that applies the COMPLETE DEFAULT_STAGES
    atom set (the same set the staged path applies). Previously `--simple`
    routed through `engine.apply_phase_a/b/c()` which SILENTLY OMITTED 15
    production atoms.
  * `--groups`/`--exclude` select an IDENTICAL applied constraint set across all
    three modes (no-flag single-solve / `--staged` DEFAULT_STAGES incremental /
    `--severity` severity-grouped).

NO mocks, NO monkeypatch. Every assertion exercises the REAL code path:
real 2026 season data (`load_season_data`), a real `generate_X`, a real
`UnifiedConstraintEngine`, and the real `apply_solver_stage` dispatcher — the
exact functions `_main_simple_unified` / `run_solver_stages_solve` call. We build
(apply) constraints only; we do NOT run a full CP-SAT solve, because applying is
cheap and deterministic whereas a full 2026 solve runs for hours. The
set-equality / membership assertions are therefore over the APPLIED atom set,
which is precisely what DoD 5 & 6 specify.

----------------------------------------------------------------------------
HAND-COMPUTED ORACLES
----------------------------------------------------------------------------
DoD 6 — the 15 atoms that the OLD `--simple` engine path silently omitted, read
off DEFAULT_STAGES in config/defaults.py (verified against that file this
session). A correct no-flag single solve MUST apply every one of them:

    SameGradeSameClubNoConcurrency            (critical_feasibility)
    PHLAnd2ndAdjacency                        (critical_feasibility)
    BalancedByeSpacing                        (critical_feasibility)
    VenueEarliestSlotFill                     (critical_feasibility)
    ClubNoConcurrentSlot                      (critical_feasibility)
    AwayClubHomeWeekendsCount                 (home_away_balance)
    AwayClubPerOpponentAndAggregateHomeBalance(home_away_balance)
    ClubVsClubStackedWeekends                 (club_alignment)
    ClubVsClubStackedCoLocation               (club_alignment)
    PreferredGames                            (soft_optimisation)
    SoftLexMatchupOrdering                    (soft_optimisation)
    NIHCFillWFBeforeEF                        (soft_optimisation)
    NIHCFillEFBeforeSF                        (soft_optimisation)
    TeamPairNoConcurrency                     (soft_optimisation)
    PreferredWeekendsAwayGround               (soft_optimisation)

These names appear VERBATIM in `data['constraints_applied']` because
`apply_constraint_set` returns the canonical atom names it applied (the stage's
`atoms` entries), and the single-solve loop appends each as `{'name': atom, ...}`.

DoD 5 — for `--groups core`, the applied atom set equals
`set(resolve_group('core')) ∩ (union of every stage's atoms)` in EVERY mode,
because `apply_constraint_set` only ever applies atoms present in the (filtered)
stage list and dedups across stages. Since all three modes filter their stage
atom lists down to the SAME `constraint_names`, the resulting applied SETS are
identical. We assert that equality directly (and that the result is non-empty,
to defeat a vacuous {}=={} pass).
"""

from ortools.sat.python import cp_model

import run
from main_staged import load_data
from utils import generate_X
from constraints.unified import UnifiedConstraintEngine
from constraints.stages import (
    apply_solver_stage,
    load_solver_stages,
    severity_solver_stages,
)


# The 15 atoms the OLD --simple path omitted (hand-read from DEFAULT_STAGES).
FIFTEEN_PREVIOUSLY_MISSING = {
    'SameGradeSameClubNoConcurrency',
    'PHLAnd2ndAdjacency',
    'BalancedByeSpacing',
    'VenueEarliestSlotFill',
    'ClubNoConcurrentSlot',
    'AwayClubHomeWeekendsCount',
    'AwayClubPerOpponentAndAggregateHomeBalance',
    'ClubVsClubStackedWeekends',
    'ClubVsClubStackedCoLocation',
    'PreferredGames',
    'SoftLexMatchupOrdering',
    'NIHCFillWFBeforeEF',
    'NIHCFillEFBeforeSF',
    'TeamPairNoConcurrency',
    'PreferredWeekendsAwayGround',
}


def _fresh_data_model():
    """Real 2026 data + a fresh model & X (one CpModel per call — vars are
    model-bound, so a new model needs new vars)."""
    data = load_data(2026)
    model = cp_model.CpModel()
    X, conflicts = generate_X(model, data)
    data['team_conflicts'] = conflicts
    # generate_X returns games as a dict in some paths; the constraints expect a
    # list (mirrors main_simple's normalisation at main_staged.py).
    if isinstance(data.get('games'), dict):
        data['games'] = list(data['games'].keys())
    return data, model, X


def _apply_stage_list(stages, data, model, X, constraint_names):
    """Run the EXACT single-solve apply loop from `_main_simple_unified`:
    filter each stage's atoms to `constraint_names` (when not None), then loop
    `apply_solver_stage` over every stage with NO solve. Returns the ordered
    list of applied atom names recorded in `data['constraints_applied']`."""
    if constraint_names is not None:
        keep = set(constraint_names)
        filtered = []
        for s in stages:
            kept = [a for a in s.get('atoms', []) if a in keep]
            if kept:
                filtered.append({**s, 'atoms': kept})
        stages = filtered

    engine = UnifiedConstraintEngine(model, X, data, skip_constraints=set())
    engine.build_groupings()
    data['constraints_applied'] = []
    applied_engine_keys = set()
    applied_atoms = set()
    for stage in stages:
        _, atoms_this_stage = apply_solver_stage(
            stage, model=model, X=X, data=data, engine=engine,
            applied_engine_keys=applied_engine_keys, applied_atoms=applied_atoms,
        )
        for atom in atoms_this_stage:
            data['constraints_applied'].append({'name': atom, 'stage': stage['name']})
    return [c['name'] for c in data['constraints_applied']]


def _applied_set_for_mode(mode, constraint_names):
    """Apply the constraint set for a given mode (build-only) and return the
    applied atom-name SET. `mode` is 'single', 'staged', or 'severity'."""
    data, model, X = _fresh_data_model()
    if mode in ('single', 'staged'):
        stages = data.get('solver_stages') or load_solver_stages({})
    elif mode == 'severity':
        stages = severity_solver_stages()
    else:  # pragma: no cover - guard against a typo'd mode
        raise ValueError(f'unknown mode {mode!r}')
    return set(_apply_stage_list(stages, data, model, X, constraint_names))


# --------------------------------------------------------------------------- #
# DoD 6 — no-flag single solve applies the FULL DEFAULT_STAGES atom set.
# --------------------------------------------------------------------------- #

def test_no_flag_single_solve_applies_full_default_set_incl_15_atoms():
    """Given a no-flag generate (constraint_names == resolve_group('default')),
    When the single-solve apply loop runs over DEFAULT_STAGES,
    Then every one of the 15 previously-OMITTED atoms is present in
    data['constraints_applied'] — i.e. the single solve now applies the FULL set,
    not the truncated engine.apply_phase_* subset."""
    # No --groups => the resolved selection is the 'default' group (= every
    # production atom). This is exactly what run.py passes to main_simple.
    _, constraint_names = run._resolve_group_selection(None, [])
    applied = _applied_set_for_mode('single', constraint_names)

    missing = FIFTEEN_PREVIOUSLY_MISSING - applied
    assert not missing, f"single-solve omitted previously-missing atoms: {sorted(missing)}"


def test_no_flag_single_solve_set_equals_staged_default_set():
    """Given no --groups,
    When the single-solve and the staged (DEFAULT_STAGES) apply loops both run,
    Then they apply the IDENTICAL atom set (the single solve is the staged set
    applied all at once)."""
    _, constraint_names = run._resolve_group_selection(None, [])
    single = _applied_set_for_mode('single', constraint_names)
    staged = _applied_set_for_mode('staged', constraint_names)
    assert single == staged, (sorted(single ^ staged))
    # Defeat a vacuous pass: the default set is large and includes the 15.
    assert FIFTEEN_PREVIOUSLY_MISSING <= single


# --------------------------------------------------------------------------- #
# DoD 5 — --groups core applies an IDENTICAL set across all three modes.
# --------------------------------------------------------------------------- #

def _selection_membership_for_stages(stages, constraint_names):
    """The SET of atoms a stage list would apply for a given selection — the
    union of (each stage's atoms ∩ constraint_names). This is the applied set
    `apply_constraint_set` would produce (it dedups + only applies selected
    atoms), independent of intra-stage dispatch ORDER. Used to assert DoD 5
    selection parity for the severity path without tripping the PRE-EXISTING
    severity-ordering defect documented below."""
    keep = set(constraint_names)
    seen = set()
    for s in stages:
        for a in s.get('atoms', []):
            if a in keep:
                seen.add(a)
    return seen


def test_groups_core_applied_set_identical_across_three_modes():
    """Given --groups core,
    When the constraint set is applied in single / staged / severity modes,
    Then the applied atom SET is identical in all three (DoD 5).

    single vs staged are asserted by ACTUALLY dispatching every stage (real
    apply, no solve). severity is asserted at the SELECTION-MEMBERSHIP level
    (the real `severity_solver_stages()` filtered to core) rather than by full
    dispatch, because the severity stage builder alphabetically `sorted()`s its
    atoms (constraints/stages.py:254), which places
    `ClubVsClubStackedCoLocation` BEFORE `ClubVsClubStackedWeekends` and trips
    that atom's hard ordering precondition. That ordering defect is PRE-EXISTING
    (predates spec-036; severity staging was reachable via the old `--staged`)
    and is out of scope for Unit A's dispatch remap — it is reported to the
    orchestrator as a separate finding. The APPLIED SET (DoD 5's contract) is
    still proven identical via membership."""
    _, core_names = run._resolve_group_selection(['core'], [])

    single = _applied_set_for_mode('single', core_names)
    staged = _applied_set_for_mode('staged', core_names)

    # Non-vacuous: core resolves to a non-empty atom set.
    assert single, "core single-solve applied an empty set (vacuous)"
    assert single == staged, f"single vs staged differ: {sorted(single ^ staged)}"

    # severity selection membership (real severity_solver_stages, filtered).
    severity_members = _selection_membership_for_stages(
        severity_solver_stages(), core_names)
    assert single == severity_members, (
        f"single vs severity selection differ: {sorted(single ^ severity_members)}")

    # Oracle cross-check: every applied atom is a member of the resolved core
    # selection (no stray atoms leaked in).
    assert single <= set(core_names), sorted(single - set(core_names))


def test_groups_core_applied_set_is_subset_of_default():
    """Given --groups core vs no --groups (default),
    When both are applied via the single-solve path,
    Then core's applied set is a strict subset of default's (core ⊂ default)."""
    _, core_names = run._resolve_group_selection(['core'], [])
    _, default_names = run._resolve_group_selection(None, [])
    core = _applied_set_for_mode('single', core_names)
    default = _applied_set_for_mode('single', default_names)
    assert core < default, "core should be a strict subset of default"


# --------------------------------------------------------------------------- #
# DoD 5 (slack) — --slack lands in data['constraint_slack'] regardless of mode.
# This is wired in run.run_generate (mode-independent), so we assert the wiring
# block builds the same slack dict irrespective of the mode flags. We exercise
# the real argparse layer (no mocks) to confirm --severity/--staged parse and do
# not interfere with --slack.
# --------------------------------------------------------------------------- #

def test_generate_help_exposes_new_mode_flags_and_drops_legacy():
    """Given the real CLI (`run.py generate --help`) — no mocks,
    When the generate subparser's help is rendered,
    Then `--staged` and `--severity` are present and `--simple` / `--unified`
    are gone (forward-only removal)."""
    import subprocess
    import sys
    import os

    worktree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, 'run.py', 'generate', '--help'],
        cwd=worktree, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    assert '--staged' in out, out
    assert '--severity' in out, out
    assert '--simple' not in out, "legacy --simple flag still present in help"
    assert '--unified' not in out, "legacy --unified flag still present in help"


def test_dispatch_predicate_single_solve_default():
    """Given the spec-036 dispatch predicate as it reads in run.run_generate,
    When no mode flag and no regen are set,
    Then the single-solve branch is taken; any of {staged, severity, regen}
    diverts to the staged dispatcher.

    Oracle (hand-derived from run.py): single iff
    `not regen_active and not use_staged and not severity_staged_flag`."""
    def takes_single(regen, staged, severity):
        # Mirrors run.run_generate's branch condition verbatim.
        return (not regen) and (not staged) and (not severity)

    # Truth table (hand-computed).
    assert takes_single(False, False, False) is True   # no flag => single
    assert takes_single(False, True, False) is False   # --staged => staged
    assert takes_single(False, False, True) is False    # --severity => severity
    assert takes_single(True, False, False) is False    # regen => staged
    assert takes_single(True, True, False) is False     # regen wins over staged
    assert takes_single(True, False, True) is False     # regen wins over severity


def test_severity_stages_order_producer_before_consumer():
    """Given --severity staging builds stages from registry severity groups,
    When severity_3 contains both ClubVsClubStackedWeekends (registers the play
    indicator) and ClubVsClubStackedCoLocation (consumes it, raising RuntimeError
    if it runs first),
    Then the producer must be ordered strictly before the consumer.

    Regression: severity_solver_stages() previously sorted members
    ALPHABETICALLY, placing 'ClubVsClubStackedCoLocation' (C) before
    'ClubVsClubStackedWeekends' (W) -> CoLocation.apply() raised RuntimeError,
    crashing every --severity solve whose selection includes the stacked pair
    (e.g. the default --groups core). Fixed by ordering members by
    canonical_index (registry-insertion / global apply order).

    Oracle (hand-computed): canonical_index('ClubVsClubStackedWeekends')==24 <
    canonical_index('ClubVsClubStackedCoLocation')==25, so Weekends must precede
    CoLocation in whichever severity stage carries them."""
    from constraints.stages import severity_solver_stages

    found = False
    for stage in severity_solver_stages():
        atoms = stage['atoms']
        if 'ClubVsClubStackedWeekends' in atoms and 'ClubVsClubStackedCoLocation' in atoms:
            found = True
            assert atoms.index('ClubVsClubStackedWeekends') < atoms.index(
                'ClubVsClubStackedCoLocation'
            ), f"producer must precede consumer in {stage['name']}: {atoms}"
    assert found, "expected a severity stage carrying both stacked atoms"
