<!-- status: delayed -->
<!-- owner: session=none claimed=none -->
<!-- BLOCKED: delayed pending user approval before review or implementation (orchestrator hold, 2026-05-22). Do not pick up. -->
<!-- depends_on: spec-021 (removes the `_best_timeslot_choices_*` and `_club_game_spread_*` engine methods and adds the key-level soft_only guard). Shares constraints/stages.py, constraints/unified.py, constraints/atoms/base.py, config/defaults.py with spec-018/021/022 — rebase + re-run validate_solver_stages before merge. -->
<!-- tier: complex -->

# spec-023 — Atoms expose hard/soft *phases*; a stage is a uniform collection (no dedicated hard-vs-soft machinery)

## Why

There are currently **two ways** a rule's "hardness" is decided, and they disagree:

- **Legacy engine constraints** (in `UnifiedConstraintEngine`) split into per-constraint
  `_x_hard()` / `_x_soft()` methods, run by two monolithic passes `apply_stage_1_hard()` and
  `apply_stage_2_soft()`, gated per-constraint by `skip_constraints` and per-stage by the
  `soft_only` flag.
- **Atoms** (`constraints/atoms/*.py`) have a single `apply()` that does *both* its hard
  `model.Add(...)` and its soft penalty registration in one call. The atom-dispatch branch in
  `constraints/stages.py::apply_solver_stage` runs `apply()` **unconditionally — it does not
  respect `soft_only` at all.**

Consequences of the asymmetry:
1. You cannot make an atom soft by staging — its hard half fires regardless of `soft_only`
   (which is *why* spec-016 had to hand-rewrite the NIHC atoms' `apply` to emit penalties
   instead of relying on the stage). Each "make this atom soft/hard" becomes a bespoke code
   edit rather than a stage declaration.
2. `critical_feasibility` and `soft_optimisation` look like dedicated subsystems (a "hard pass"
   and a "soft pass") when conceptually a stage is *just a named collection of rules to apply*.
   The `soft_only` flag + two passes are machinery papering over the fact that an atom can't say
   "here is my hard half, here is my soft half."
3. "Is constraint X hard or soft?" is answerable three different ways (engine hard-key
   membership, engine soft-key membership, or buried inside an atom's `apply`). The violation
   reporter, the tester, and `--relax`/severity logic all have to special-case this.

The convenor-facing principle (this spec): **every rule a constraint enforces is exposed as a
phase — hard or soft — and a stage is simply a collection of (atom, phase) selections.**
`critical_feasibility` is "the collection whose hard phases must hold"; `soft_optimisation` is
"the collection whose soft phases we optimise." Same dispatch mechanism, no dedicated hard pass
vs soft pass. This makes spec-021's `soft_only`-trap guard structurally impossible to violate
rather than something a test has to police.

## Definition of Done

1. **`Atom` base API (`constraints/atoms/base.py`)** gains two phase methods with an explicit
   contract:
   - `apply_hard(self, model, X, data, registry) -> int` — adds only hard constraints.
   - `apply_soft(self, model, X, data, registry) -> int` — adds only soft penalty terms.
   - Default implementations: both return 0 (a pure-hard atom overrides only `apply_hard`; a
     pure-soft atom only `apply_soft`). A `phases()` introspection returns the set of phases the
     subclass actually implements (detected via method override), so the registry/inventory can
     report each atom's phases without running it.
   - `apply()` is retained as a thin back-compat shim = `apply_hard(...) + apply_soft(...)`, so
     any caller still calling `apply()` gets both halves (no silent behaviour change).
2. **Every atom that bundles both halves is split** into `apply_hard` / `apply_soft`. Audit at
   plan time (post-spec-021) — the bundled atoms are: `ClubGameSpread` (if it survives spec-021
   as an atom with hard no-gap + soft spread), and any other atom that registers a penalty AND
   adds a hard constraint in one `apply`. Pure-hard atoms (NoDoubleBooking*, ClubDay*,
   PHLAnd2ndAdjacency, SameGradeSameClubNoConcurrency, BalancedByeSpacing, ClubVsClubStacked*,
   AwayClub*, NIHC*, VenueEarliestSlotFill) implement only `apply_hard`; pure-soft atoms
   (PreferredDates/PreferredGames, SoftLexMatchupOrdering, TeamPairNoConcurrency,
   PreferredWeekendsAwayGround) implement only `apply_soft`. The audit result is recorded in the
   plan before Unit B runs.
3. **`apply_solver_stage` is phase-aware and symmetric** for both dispatch worlds. A stage
   declares its phase(s) — replace the boolean `soft_only` with an explicit
   `phases: ['hard','soft']` list (default `['hard','soft']`; `soft_optimisation` →
   `['soft']`). The atom branch calls `apply_hard`/`apply_soft` according to the stage's phases
   — the **same** selection the engine branch already applies via `apply_stage_1_hard` /
   `apply_stage_2_soft`. `soft_only: True` is accepted as a deprecated alias for
   `phases: ['soft']` (one season's grace), with a migration note.
4. **The `soft_only`-only behaviour gap is closed:** an atom placed in a `phases:['soft']` stage
   runs ONLY its `apply_soft`; placed in a default stage runs both. A regression test proves a
   hard-bearing atom in a soft stage adds **zero** hard constraints (the exact bug spec-016 had
   to hand-patch).
5. **`validate_solver_stages`** additionally checks: every atom named in a stage implements at
   least one of the phases that stage runs (no atom contributing nothing); and an atom's hard
   phase is never the sole content of a `['soft']` stage (subsumes spec-021 DoD-12's guard at
   the atom level, kept alongside the key-level guard).
6. **Registry** records each entry's phases (derived from the atom class's `phases()`), so
   `CONSTRAINT_INVENTORY.md` generation and the severity/`--relax` logic read phase from one
   place instead of inferring from `has_soft_component` + engine-key membership. `has_soft_component`
   becomes a derived convenience (`'soft' in phases`).
7. **Engine-method constraints are reconciled, not orphaned.** For World-1 constraints still
   living as `_x_hard`/`_x_soft` engine methods at execution time (post-spec-021 remainder, e.g.
   `EqualGamesAndBalanceMatchUps`, the two Broadmeadow atoms, `PreferredTimes`), the engine
   branch keeps working unchanged — `apply_stage_1_hard`/`_soft` already honour the same phase
   selection. This spec does NOT require atomising them (see Out of scope), only that the stage
   `phases` selection drives both branches identically.
8. Full suite green; `validate_solver_stages(DEFAULT_STAGES) == []`; an integration test builds
   the real 2026 model through the new phase-aware dispatch and asserts the produced
   hard-constraint and penalty counts match the pre-refactor build **exactly** (behaviour
   parity — this is a refactor, not a behaviour change).
9. Grep-clean: no remaining unconditional atom `apply()` call inside `apply_solver_stage` (it
   must go through `apply_hard`/`apply_soft`); `soft_only` references either removed or routed
   through the alias shim.

## Open decisions (recommendations baked in)

- **A. Replace `soft_only` with `phases` list, or keep `soft_only` + add `hard_only`?**
  Recommendation: **`phases: [...]` list** — it's the honest model ("a stage runs these phases of
  its atoms"), extends to a future third phase if ever needed, and makes "both" the explicit
  default. Keep `soft_only: True` as a parsed alias for one season so existing season configs and
  `--stages-config` JSON don't break.
- **B. Retire `apply_stage_1_hard` / `apply_stage_2_soft` monoliths now?** Recommendation: **no,
  not in this spec.** They still serve the un-atomised World-1 constraints. This spec makes the
  *dispatch* uniform and phase-driven; fully deleting the two passes happens once the last
  engine-method constraint becomes an atom (separate spec, depends on the atomization finishing).
  Deleting them now would strand World-1.
- **C. How to detect which phases an atom implements?** Recommendation: method-override detection
  in `phases()` (compare `type(self).apply_hard is not Atom.apply_hard`), not a hand-maintained
  class attribute — keeps it impossible to mis-declare.

## Implementation units

> Heavily shared files (`base.py`, `stages.py`, `unified.py`, `defaults.py`, `registry.py`).
> Sequence the units in one worktree; rebase on spec-021 (and spec-018) before merge.

### Unit A — `Atom` phase API + `phases()` introspection
- Files: `constraints/atoms/base.py`, `tests/atoms/test_atom_phase_api.py` (new).
- Add `apply_hard`/`apply_soft` (default 0), `phases()` via override-detection, and the
  `apply()` back-compat shim. No atom migrated yet.
- Test (GWT, hand oracle): a stub pure-hard atom reports `phases()=={'hard'}` and its
  `apply_soft` adds nothing; a pure-soft atom the reverse; a both-atom reports both and
  `apply()==apply_hard()+apply_soft()` count.

### Unit B — Migrate bundled atoms to the split
- Files: the atom files identified in the DoD-2 audit (e.g. `club_game_spread` if atomised by
  spec-021), each `tests/atoms/test_*` updated to call the phase methods.
- Depends on Unit A. Behaviour-identical; existing atom tests stay green; add a per-atom test
  asserting `apply_hard` alone adds the hard constraints and `apply_soft` alone the penalties.

### Unit C — Phase-aware `apply_solver_stage` + stage `phases` + validation
- Files: `constraints/stages.py` (dispatch + `phases` parsing + `soft_only` alias +
  `validate_solver_stages` checks), `config/defaults.py` (DEFAULT_STAGES: `soft_optimisation`
  → `phases:['soft']`), `tests/test_solver_stages_dispatch.py`.
- Depends on Units A+B. Test per DoD 3, 4, 5; the DoD-8 parity integration test.

### Unit D — Registry phase metadata + inventory/severity read-through
- Files: `constraints/registry.py` (phase field derived from atom `phases()`),
  `constraints/severity.py` (read phase from registry where it inferred before),
  `analytics/tester.py` if it special-cased soft, `tests/test_constraint_registry.py`.
- Depends on Units A–C. Test: every registry entry's phase set matches its atom class's
  `phases()`; `has_soft_component == ('soft' in phases)` for all.

## Doc registry

- `docs/system/STAGES.md` — document stages as `(atoms, phases)` collections; the `phases` field;
  `soft_only` deprecation/alias.
- `docs/system/CONSTRAINT_INVENTORY.md` — add a per-atom "phases" column; note phase is the
  single source of truth for hard/soft.
- `docs/operator-ai/CONSTRAINT_APPLICATION.md` — update "how to add a constraint": implement
  `apply_hard` and/or `apply_soft`, not a bundled `apply`.
- `CLAUDE.md` — update the constraint-system / severity section: hardness is an atom phase, a
  stage selects phases; `--relax`/severity read phase from the registry.
- `docs/todo/GOALS.md` — add spec-023 row; record the "stage = collection of (atom, phase),
  no dedicated hard/soft machinery" principle under §2.

## Out of scope

- Atomising the remaining World-1 engine-method constraints (`EqualGamesAndBalanceMatchUps`, the
  Broadmeadow min/max, `PreferredTimes`, legacy `ClubVsClubAlignment`) — separate atomization
  specs. This spec only makes dispatch phase-uniform across both worlds.
- Deleting the `apply_stage_1_hard` / `apply_stage_2_soft` monoliths (Open decision B — gated on
  World-1 atomization finishing).
- The helper-var pathway unification (spec-022) and the contiguity primitive (spec-021) — this
  spec rebases on them but does not change them.
- Any change to *which* rules are hard or soft (that's spec-016/017/021's job) — spec-023 is a
  pure structural refactor with behaviour parity (DoD 8).
