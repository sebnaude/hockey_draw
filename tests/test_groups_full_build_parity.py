"""spec-023 Unit D — DoD 8: full-config 2026 STAGED build behaviour parity.

The canonical "full build" parity target is the STAGED DEFAULT_STAGES build
(NOT the old simple path). We build the real 2026 model two ways through the
SAME production dispatch (`apply_solver_stage` -> `apply_constraint_set`) and
assert they emit an IDENTICAL hard-constraint count (and identical total model
constraints):

  (a) staged-legacy  — iterate DEFAULT_STAGES with NO --groups filter (the
      pre-refactor full-build selection: the deduped union of every stage's
      atoms).
  (b) staged-default — iterate DEFAULT_STAGES with each stage's atoms filtered
      to `resolve_group('default')` (every production constraint), exactly the
      way main_staged.py applies a `--groups`-resolved set (lines ~1122-1129).

ORACLE (hand-computed, zero delta expected)
-------------------------------------------
The expected delta between (a) and (b) is **ZERO**. Why (spec §Why + DoD 8):

  * `resolve_group('default')` == the deduped union of all DEFAULT_STAGES atoms.
    Verified independently: both are the SAME 28 canonical atoms / 8 engine
    keys (see `test_default_group_equals_default_stages_union` below). So the
    filter in (b) drops nothing — every stage's atoms survive.
  * Removing `soft_only` is behaviour-neutral: spec-021 (done) moved
    `ClubGameSpread` to the non-`soft_only` `club_day` stage, so its HARD part
    already runs in both builds; and the only other constraints in
    `soft_optimisation` are pure-soft atoms with no live hard engine method.
    There is therefore no hard constraint that (a) suppresses-and-(b)-doesn't or
    vice-versa.
  * The obsolete trio (`FiftyFiftyHomeandAway`, `ClubVsClubAlignment`, and the
    legacy `ClubDay`/etc. duplicates) is NOT in DEFAULT_STAGES' atom lists, so
    neither build selects them. (That −2453 simple-path delta is an INTENTIONAL
    Unit-C change pinned elsewhere — see
    `test_groups_cli_wiring::test_default_simple_selection_excludes_obsolete_trio_by_design`
    — and is explicitly NOT what this test measures.)

Hence: hard(a) == hard(b), and total(a) == total(b).

NO MOCKS: real `load_data(2026)` + real `generate_X` + the real production
dispatch functions. The only instrumentation is a thin wrapper around the
engine's own `apply_stage_1_hard` to accumulate its returned hard count — it
does not alter behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.defaults import DEFAULT_STAGES
from constraints.registry import resolve_group
from constraints.stages import apply_solver_stage, load_solver_stages
from constraints.unified import UnifiedConstraintEngine


# ----------------------------------------------------------------------
# Selection-level oracle: default group == DEFAULT_STAGES atom union.
# ----------------------------------------------------------------------


def test_default_group_equals_default_stages_union():
    """Given the registry `default` group and the DEFAULT_STAGES partition,
    When we take the union of every stage's atoms,
    Then it equals `resolve_group('default')` exactly (no atom added/dropped) —
    this is the hand-oracle that makes the build-level zero-delta inevitable."""
    stage_union = set()
    for s in DEFAULT_STAGES:
        stage_union.update(s.get('atoms', []))
    default_set = resolve_group('default')
    assert stage_union == default_set, (
        f"in stages not default: {sorted(stage_union - default_set)}; "
        f"in default not stages: {sorted(default_set - stage_union)}"
    )


# ----------------------------------------------------------------------
# Build harness — mirrors main_staged.run_solver_stages_solve (no solve).
# ----------------------------------------------------------------------


def _build_2026_data_and_X():
    from main_staged import load_data
    from utils import generate_X

    data = load_data(2026)
    model = cp_model.CpModel()
    X, _Y = generate_X(model, data)
    return data, model, X


def _instrumented_engine(model, X, data):
    """Real engine, with `apply_stage_1_hard` wrapped to accumulate the HARD
    count it returns. The wrapper changes no behaviour — it calls the real
    method and records the integer it returns."""
    engine = UnifiedConstraintEngine(model, X, data, skip_constraints=set())
    engine.build_groupings()
    engine._hard_total = 0
    _orig_hard = engine.apply_stage_1_hard

    def _wrapped_hard():
        c = _orig_hard()
        engine._hard_total += c
        return c

    engine.apply_stage_1_hard = _wrapped_hard
    # apply_phase_a is the alias the engine itself routes through; keep it
    # pointing at the wrapped version so any internal call is also counted.
    engine.apply_phase_a = _wrapped_hard
    return engine


def _apply_stages(stages, data, model, X):
    """Mirror main_staged.run_solver_stages_solve's stage loop WITHOUT solving:
    one engine, build_groupings, then apply each stage via the production
    `apply_solver_stage` dispatcher carrying applied_* sets across stages.
    Returns (engine_hard_count, total_model_constraints)."""
    engine = _instrumented_engine(model, X, data)
    applied_engine_keys: set = set()
    applied_atoms: set = set()
    for stage in stages:
        apply_solver_stage(
            stage,
            model=model, X=X, data=data, engine=engine,
            applied_engine_keys=applied_engine_keys,
            applied_atoms=applied_atoms,
        )
    return engine._hard_total, len(model.Proto().constraints)


def _legacy_stages():
    """staged-legacy: DEFAULT_STAGES, no --groups filter."""
    return load_solver_stages({})


def _default_group_stages():
    """staged-default: DEFAULT_STAGES with each stage's atoms filtered to
    resolve_group('default'), exactly as main_staged.py does for a resolved
    --groups set (drop emptied stages)."""
    keep = resolve_group('default')
    filtered = []
    for s in load_solver_stages({}):
        kept = [a for a in s.get('atoms', []) if a in keep]
        if kept:
            filtered.append({**s, 'atoms': kept})
    return filtered


# ----------------------------------------------------------------------
# DoD 8 — the parity assertion.
# ----------------------------------------------------------------------


def test_staged_legacy_vs_staged_default_hard_count_parity():
    """Given the real 2026 model built (a) through legacy DEFAULT_STAGES and
    (b) through the same stages filtered to the `default` group,
    When both run the real production dispatch (no solve),
    Then the HARD-constraint count is IDENTICAL (zero delta) and the TOTAL
    model-constraint count is IDENTICAL — the spec-023 refactor is
    behaviour-neutral for the canonical full build."""
    # Each build needs its own data/model/X (penalty buckets + model are
    # mutated during application).
    data_a, model_a, X_a = _build_2026_data_and_X()
    hard_a, total_a = _apply_stages(_legacy_stages(), data_a, model_a, X_a)

    data_b, model_b, X_b = _build_2026_data_and_X()
    hard_b, total_b = _apply_stages(_default_group_stages(), data_b, model_b, X_b)

    # Sanity: a real full build emits a non-trivial number of hard constraints.
    assert hard_a > 0
    assert total_a > 0

    # The DoD-8 parity assertions: zero delta.
    assert hard_a == hard_b, (
        f"hard-constraint count differs: staged-legacy={hard_a} "
        f"vs staged-default={hard_b} (delta {hard_b - hard_a}, expected 0)"
    )
    assert total_a == total_b, (
        f"total model-constraint count differs: staged-legacy={total_a} "
        f"vs staged-default={total_b} (delta {total_b - total_a}, expected 0)"
    )
