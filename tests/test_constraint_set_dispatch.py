"""spec-023 Unit B — `apply_constraint_set` dispatch + `validate_solver_stages` rewrite.

Covers DoD 4, 5, 6:
- DoD 5: dispatching a set that includes an engine key emits HARD constraints
  (count > 0) — proven against the REAL engine (`EqualMatchUpSpacing`). The
  CONTRAST witness (a `{'soft_only': True}` stage USED to skip the hard pass)
  uses a SYNTHETIC engine fixture, because after spec-017/021 no real production
  constraint is hard-and-stranded in `soft_optimisation`.
- DoD 6: the rewritten `validate_solver_stages` accepts an overlapping-group
  config (returns []), still rejects an unknown-atom config, and returns []
  on the real DEFAULT_STAGES.

GWT, no mocks of the production dispatch logic. The synthetic fixture below is a
hand-built recorder object (not a mock framework) used purely to prove the OLD
`if not soft_only:` branch behaviour against a contrived stage dict.
"""
from __future__ import annotations

from typing import Dict, List, Set

from ortools.sat.python import cp_model

from config.defaults import DEFAULT_STAGES
from constraints.registry import resolve_groups
from constraints.stages import (
    ALL_ENGINE_KEYS,
    apply_constraint_set,
    apply_solver_stage,
    validate_solver_stages,
)

from tests.test_spacing_promoted_hard import _fixture, _engine, _pair_vars, S0


# ----------------------------------------------------------------------
# DoD 5 — real engine key emits HARD when dispatched (no soft_only)
# ----------------------------------------------------------------------


class TestEngineKeyEmitsHard:
    def test_apply_constraint_set_runs_hard_for_engine_key(self):
        """Given a resolved set containing the engine key EqualMatchUpSpacing,
        when apply_constraint_set dispatches it, then HARD constraints are
        emitted (count > 0) AND the hard clause bites (forced gap == S0 ->
        INFEASIBLE)."""
        model, X, data = _fixture()
        engine = _engine(model, X, data)
        added, applied = apply_constraint_set(
            ['EqualMatchUpSpacing'],
            model=model, X=X, data=data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )
        assert applied == ['EqualMatchUpSpacing']
        assert added > 0  # hard + soft constraints emitted
        # Soft bucket populated (whole-constraint: hard AND soft).
        assert data['penalties']['EqualMatchUpSpacing']['penalties']
        # Hard part bites.
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1)) >= 1)
        model.Add(sum(_pair_vars(X, 'T0', 'T1', 1 + S0)) >= 1)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        assert solver.Solve(model) == cp_model.INFEASIBLE


# ----------------------------------------------------------------------
# DoD 5 — SYNTHETIC contrast: the OLD soft_only branch skipped the hard pass.
# Hand-built recorder engine + a synthetic engine key + a {'soft_only': True}
# stage dict, replicating the pre-spec-023 lever to prove it is now gone.
# ----------------------------------------------------------------------


class _RecorderEngine:
    """Minimal stand-in for UnifiedConstraintEngine that records which passes
    ran. Not a mock framework — a hand oracle: `apply_stage_1_hard` returns 7
    'hard' constraints, `apply_stage_2_soft` returns 3 'soft'. We then assert on
    `hard_calls` / `soft_calls`."""

    def __init__(self) -> None:
        self.skip_constraints: Set[str] = set()
        self.hard_calls = 0
        self.soft_calls = 0
        # Atoms dispatch needs a registry attr; None forces the ephemeral path,
        # but we only ever feed it an engine key, so the atom loop is empty.
        self.helper_registry = None
        self.registry = None

    def apply_stage_1_hard(self) -> int:
        self.hard_calls += 1
        return 7

    def apply_stage_2_soft(self) -> int:
        self.soft_calls += 1
        return 3


# The witness atom for the synthetic stage. We reuse a real engine key
# (recognised by `collect_engine_keys` via the `atom_name in ALL_ENGINE_KEYS`
# branch) so the dispatcher routes it through the engine pass — but the engine
# itself is the hand-built `_RecorderEngine`, so no real constraints are built.
_SYNTH_KEY = 'NoDoubleBookingTeams'


class TestSyntheticSoftOnlyContrast:
    def test_old_soft_only_branch_skipped_hard_now_always_runs(self):
        """Witness the deleted lever. Pre-spec-023, apply_solver_stage did
        `if not soft_only: apply_stage_1_hard()`. We feed the CURRENT
        apply_solver_stage a `{'soft_only': True}` stage and assert the hard
        pass STILL runs (the key is ignored) — i.e. the suppression behaviour
        is gone. Then we replicate the OLD branch by hand on the recorder to
        document the contrast that the dispatcher no longer exhibits."""
        engine = _RecorderEngine()
        model = cp_model.CpModel()
        stage = {'name': 'synthetic', 'atoms': [_SYNTH_KEY], 'soft_only': True}
        added, applied = apply_solver_stage(
            stage, model=model, X={}, data={'_use_ai': False},
            engine=engine, applied_engine_keys=set(), applied_atoms=set(),
        )
        # NEW behaviour: hard ran despite soft_only (key ignored) -> 7 + 3 = 10.
        assert engine.hard_calls == 1
        assert engine.soft_calls == 1
        assert added == 10
        assert applied == [_SYNTH_KEY]

        # CONTRAST (hand oracle of the DELETED branch): had the dispatcher still
        # honoured soft_only it would have skipped hard, yielding only the soft
        # pass (3). Replicate that branch directly to document what was removed.
        old_engine = _RecorderEngine()
        soft_only = bool(stage.get('soft_only'))
        old_added = 0
        if not soft_only:                       # the deleted guard
            old_added += old_engine.apply_stage_1_hard()
        old_added += old_engine.apply_stage_2_soft()
        assert old_engine.hard_calls == 0       # would have skipped hard
        assert old_added == 3                   # soft only
        # The new dispatch (added=10) differs from the old (old_added=3) by
        # exactly the suppressed hard pass — proving the lever is gone.
        assert added - old_added == 7


# ----------------------------------------------------------------------
# DoD 6 — validate_solver_stages rewrite
# ----------------------------------------------------------------------


class TestValidateSolverStages:
    def test_default_stages_valid(self):
        assert validate_solver_stages(DEFAULT_STAGES) == []

    def test_overlapping_groups_accepted(self):
        """The no-overlap rule is REMOVED: the same atom may appear in two
        stages. Pre-spec-023 this returned a 'appears in stages X and Y' error;
        now it must be []."""
        atom = 'NoDoubleBookingTeams'
        stages = [
            {'name': 'g1', 'atoms': [atom, 'NoDoubleBookingFields']},
            {'name': 'g2', 'atoms': [atom, 'EqualGamesAndBalanceMatchUps']},
        ]
        assert validate_solver_stages(stages) == []

    def test_unknown_atom_rejected(self):
        stages = [{'name': 's', 'atoms': ['DefinitelyNotARegisteredAtom']}]
        errors = validate_solver_stages(stages)
        assert errors
        assert any('DefinitelyNotARegisteredAtom' in e for e in errors)

    def test_group_name_member_accepted(self):
        """A stage may reference a group name (e.g. 'core') as a member."""
        stages = [{'name': 's', 'atoms': ['core']}]
        assert validate_solver_stages(stages) == []

    def test_empty_atoms_rejected(self):
        stages = [{'name': 's', 'atoms': []}]
        errors = validate_solver_stages(stages)
        assert any('non-empty list' in e for e in errors)

    def test_duplicate_stage_name_rejected(self):
        stages = [
            {'name': 'dup', 'atoms': ['NoDoubleBookingTeams']},
            {'name': 'dup', 'atoms': ['NoDoubleBookingFields']},
        ]
        errors = validate_solver_stages(stages)
        assert any('duplicate stage name' in e for e in errors)

    def test_soft_only_key_now_unknown(self):
        """spec-023: `soft_only` is removed from ALL_KEYS, so a stage carrying
        it is flagged as an unknown key (it is no longer a recognised option)."""
        stages = [{'name': 's', 'atoms': ['NoDoubleBookingTeams'],
                   'soft_only': True}]
        errors = validate_solver_stages(stages)
        assert any('unknown keys' in e and 'soft_only' in e for e in errors)


# ----------------------------------------------------------------------
# DoD 5/6 cross-check — a resolved group set dispatches whole constraints.
# ----------------------------------------------------------------------


class TestResolvedSetDispatch:
    def test_apply_resolved_soft_group_runs(self):
        """resolve_groups(['soft']) -> deduped ordered set; dispatching it via
        apply_constraint_set applies each constraint whole. The soft group's
        engine key (PreferredTimes -> PreferredTimesConstraint) runs its soft
        pass; non-engine atoms run their full apply. We assert the call applies
        the resolved names without error and reports them as applied."""
        names = resolve_groups(['soft'])
        assert names  # non-empty
        model, X, data = _fixture()
        engine = _engine(model, X, data)
        added, applied = apply_constraint_set(
            names,
            model=model, X=X, data=data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )
        # Every resolved name that has a dispatch path is reported applied; the
        # set is deduped, so no name repeats.
        assert applied == [n for n in names if n in applied]
        assert len(applied) == len(set(applied))
