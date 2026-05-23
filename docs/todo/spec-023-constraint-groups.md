<!-- status: building -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- owner: session=opus-c9436a1-20260523T091537Z claimed=2026-05-23T09:15:37Z -->
<!-- reviewed: adversarial Sonnet review 2026-05-23 (re-review — corrected stale spec-021 dependency claim) — fixes applied inline -->
<!-- depends_on: spec-021 (done — `_best_timeslot_choices_hard/_soft` removed, `EnsureBestTimeslotChoices` engine-key removed, `ClubGameSpread` moved to `club_day` hard stage, `VenueEarliestSlotFill` + `ClubNoConcurrentSlot` atoms registered) and spec-024 (done — `ClubGameSpread` re-scoped to per-field). Both are fully landed; `validate_solver_stages(DEFAULT_STAGES)==[]`. Deleting `soft_only` is behaviour-neutral: `ClubGameSpread` is already in `club_day` (non-`soft_only`), so its hard part already runs; the `soft_optimisation` stage no longer carries any hard-bearing constraint. Shared files `constraints/registry.py`, `constraints/stages.py`, `config/defaults.py`, `run.py`, `main_staged.py` — rebase + re-run validate before merge. -->
<!-- tier: complex -->

# spec-023 — Constraint groups: composable, deduped, flag-selected; delete the hard/soft (`soft_only`) machinery

> **Supersedes the previous spec-023** ("atoms expose hard/soft *phases*; a stage selects phases").
> That design is **rejected**. A constraint — whether one atom or a bundle of atoms — is applied
> as a **whole**: its hard part sets the cap, its soft part improves within the cap, and the two
> are never applied separately. The honest model is not "peel the hard half off a constraint by
> staging" (that is what `soft_only` does today, misleadingly) but "**select which whole
> constraints to apply**, by composing named groups." If we ever genuinely need to apply one
> idea of a constraint without another, that is the trigger to **split it into two atoms** — not
> to give every atom a hard/soft phase switch.

## Why

There are three tangled mechanisms deciding *which* constraints apply and *how much* of each:

1. **`DEFAULT_STAGES`** (`config/defaults.py`) is a strict **partition** — `validate_solver_stages`
   (`constraints/stages.py:132`, no-overlap rule enforced at ~175) **forbids any atom appearing in
   more than one stage**. So stages cannot overlap and cannot be composed. There is no way to say
   "apply the union of these two groupings."
   (review fix — Medium: `validate_solver_stages` function starts at line 132, not 171-175;
   the duplicate-atom check is at ~175-180.)
2. **`soft_only: True`** on the `soft_optimisation` stage tells the engine to skip
   `apply_stage_1_hard()` for engine-key constraints (`apply_solver_stage`,
   `stages.py:272-279` — `soft_only = bool(stage.get('soft_only'))` at 272, `if not soft_only:` at 277).
   (review fix — Medium: was `stages.py:268-275`; actual lines are 272-279.)
   This is the misleading lever: it silently suppresses the *hard half* of a whole constraint.
   It is also **asymmetric** — the atom-dispatch branch ignores `soft_only` entirely and always
   runs the atom's full `apply()` — which is *why* spec-016 had to hand-rewrite the NIHC atoms and
   spec-021 had to rescue `ClubGameSpread`/`EnsureBestTimeslotChoices` from being dead-hard.
3. **Flags** (`--staged`, `--stages`, `--stage-only`, `--skip-stage`, `--exclude`) select stages
   or drop constraints, but there is **no general "apply this named selection of constraints"**
   mechanism and no `--core`/`--soft`/ad-hoc selector. Adding a new grouping today means adding a
   new stage to the partition (and fighting the no-overlap rule).

The convenor-facing principle (this spec): **a "group" is a named, possibly-overlapping set of
whole constraints; a solve applies the deduped union of the selected groups, in one canonical
order; selecting the same constraint via two groups applies it exactly once.** `core`, `soft`,
`severity_1`, an ad-hoc `test-2` — all are just groups. This is the general machinery the project
wants for "different rankings / different orderings" (GOALS §2, §4), and it makes the `soft_only`
trap **structurally impossible** because there is no longer any "hard pass / soft pass" — there is
only "which whole constraints are in the set."

### What this is NOT

- **NOT** a behaviour change to which rules are hard or soft (that's spec-016/017/021). Removing
  `soft_only` is behaviour-neutral because spec-021 (done) moved `ClubGameSpread` to the
  `club_day` hard stage and spec-024 (done) re-scoped it to per-field — the `soft_optimisation`
  stage no longer contains any constraint with a live hard part. See Precondition for the full
  verification checklist.
  (review fix — C1: original said "given spec-021 has landed — see Precondition" but the
  Precondition was stale; inlined the actual reason so the claim stands on its own.)
- **NOT** a per-constraint hard/soft phase API. A group tag selects *whole* constraints; it never
  changes how much of a constraint applies.
- **NOT** a rework of `--slack` (orthogonal: slack loosens *within* a constraint) or `--relax`
  (dropped from scope — see Out of scope).

## Design

### 1. Group membership is registry metadata (single source of truth)

`ConstraintInfo` (`constraints/registry.py`) gains:

```python
groups: frozenset[str] = frozenset()   # named groups this constraint belongs to
```

Every production constraint is tagged. Overlap is allowed and expected. Tags are the **only**
hand-maintained membership store; derived groups (below) are computed, never stored, so there is
no second source of truth. Initial tag assignment mirrors today's stages **plus** the two new
coarse dimensions the convenor asked for:

| Constraint (canonical) | groups |
|---|---|
| NoDoubleBookingTeams, NoDoubleBookingFields, EqualGamesAndBalanceMatchUps, PHLConcurrencyAtBroadmeadow, PHLAnd2ndConcurrencyAtBroadmeadow, PHLAnd2ndAdjacency, SameGradeSameClubNoConcurrency, BalancedByeSpacing, EqualMatchUpSpacing | `{core, critical_feasibility}` |
| AwayClubHomeWeekendsCount, AwayClubPerOpponentAndAggregateHomeBalance | `{core, home_away_balance}` |
| ClubVsClubStackedWeekends, ClubVsClubStackedCoLocation | `{core, club_alignment}` |
| ClubDayParticipation, ClubDayIntraClubMatchup, ClubDayOpponentMatchup, ClubDaySameField, ClubDayContiguousSlots | `{core, club_day}` |
| VenueEarliestSlotFill, ClubGameSpread, ClubNoConcurrentSlot | `{core}` |
| PreferredTimes, PreferredGames, SoftLexMatchupOrdering, NIHCFillWFBeforeEF, NIHCFillEFBeforeSF, TeamPairNoConcurrency, PreferredWeekendsAwayGround | `{soft, soft_optimisation}` |

> (review fix — Medium: `MaximiseClubsPerTimeslotBroadmeadow` and
> `MinimiseClubsOnAFieldBroadmeadow` were deleted by spec-024 and are NOT in the registry —
> listing them here would cause Unit A to fail at tagging time. Removed from the table.
> Their spread intent is now owned by `ClubGameSpread` in `club_day`.)

> The spec-021/024 atoms are tagged `{core}` because both specs make them **hard**:
> `VenueEarliestSlotFill` (DoD 2 — replaces `EnsureBestTimeslotChoices` as a hard earliest-pack
> rule; wired into `critical_feasibility`), `ClubGameSpread` (DoD 6 + Open-decision A — the UPPER
> contiguity HARD rule encoded with `slot_used` indicators; spec-024 re-scoped to per-field; both
> hard and soft engine methods remain in `unified.py` — the completion signal is the non-`soft_only`
> staging in `club_day`, not method deletion), and `ClubNoConcurrentSlot` (DoD 6b — extracted hard
> concurrency atom). They carry no legacy-stage group tag (`critical_feasibility` etc.) in this
> initial tagging pass; Unit A should reconcile against the actual `DEFAULT_STAGES` and add the
> appropriate stage tag (e.g. `{core, critical_feasibility}` for `VenueEarliestSlotFill`,
> `{core, club_day}` for `ClubGameSpread`/`ClubNoConcurrentSlot`) if the convenor wants `--stages`
> to include them via legacy-stage group name. The three `{core}` assignments above are the minimum;
> reconcile against the actual code at start time.
> (review fix — C1/C3: original said "no legacy-stage tag because they are newly-hard and belong
> to no original hard stage" which was wrong for `ClubGameSpread` — it has been in `club_day`
> since spec-021 landed; clarified the note and corrected the implication that `_club_game_spread_*`
> methods were deleted — they are intentionally kept/refactored.)
>
> `tester_only` entries (`ForcedGames`, `BlockedGames`, `ClubFieldConcentration`)
> and obsolete legacy-only entries (`FiftyFiftyHomeandAway`, `ClubDay`, `ClubGradeAdjacency`,
> `ClubVsClubAlignment`, `PHLAndSecondGradeTimes`, `EnsureBestTimeslotChoices`,
> `PreferredTimes`-legacy-class) get **no production group** — they keep resolving for parity/legacy
> lookups but are not selected by any group.

### 2. Group definitions: explicit tags + derived predicates

`constraints/registry.py` gains a resolver:

```python
# A group is either an explicit tag (membership lives on each ConstraintInfo.groups)
# or a derived predicate over ConstraintInfo. Derived groups never store membership.
DERIVED_GROUPS: Dict[str, Callable[[ConstraintInfo], bool]] = {
    f'severity_{n}': (lambda info, n=n: info.severity_level == n) for n in range(1, 6)
}
# plus 'default' / 'all' = every constraint with a non-empty groups set (i.e. every
# production constraint), and 'production' as an alias.

def resolve_group(name: str) -> set[str]: ...           # one group -> canonical names
def resolve_groups(names: Iterable[str]) -> list[str]:   # DEDUPED UNION, canonical order
    """Union the membership of every named group, dedupe, return in canonical order."""
def list_group_names() -> list[str]: ...                 # explicit tags ∪ derived names
```

`resolve_groups` is the heart of the dedup-union guarantee: it builds a `set` union of canonical
names across all requested groups, then returns them sorted by **canonical order** (below). Asking
for `['test', 'test-1']` where both contain `ClubDayParticipation` yields it **once**.

### 3. Canonical apply order = registry insertion order, validated for helper-var deps

When a deduped union is applied to one model, the only correctness-relevant ordering is
**producer-before-consumer for shared helper vars** (CP-SAT is declarative; the order of `Add*`
calls otherwise doesn't change the feasible region). The single global canonical order is the
**`CONSTRAINT_REGISTRY` dict insertion order** — which already places producers before consumers
(e.g. `ClubVsClubStackedWeekends` at registry line 326 precedes `ClubVsClubStackedCoLocation` at
336; `ClubVsClubStackedCoLocation` reads the `cvc_stack_play` helpers Weekends registers). No
per-constraint `order` int is added (deferred — see Open decision A). A **validation check**
asserts that for every known producer/consumer helper relationship, the producer's registry index
is lower than the consumer's, so a future reorder that breaks the dependency trips a red test.

### 4. Delete `soft_only`; dispatch applies whole constraints

- `soft_only` removed from `OPTIONAL_KEYS`, from `apply_solver_stage` (no more
  `if not soft_only: apply_stage_1_hard()` — **always** run both hard and soft for engine keys),
  from `list_stages`, and from `DEFAULT_STAGES`/season configs.
- A new entry point `apply_constraint_set(canonical_names, *, model, X, data, engine, applied_engine_keys, applied_atoms)`
  applies a resolved, ordered, deduped list of **whole** constraints: engine keys run
  `apply_stage_1_hard()` **and** `apply_stage_2_soft()`; non-engine atoms run their full
  `apply(model, X, data, registry)`. `apply_solver_stage` is reimplemented as a thin wrapper that
  resolves the stage's atoms and calls `apply_constraint_set` (so the staged solver keeps working),
  or is replaced outright by `apply_constraint_set` at its call sites — whichever is cleaner at
  implementation time. Bundles co-apply naturally: both bundle atoms carry the same group tag and
  the canonical order keeps Weekends before CoLocation.

### 5. CLI: one general `--groups` selector

- New `run.py generate --groups NAME [NAME ...]` (varargs). Resolves to
  `resolve_groups(names)`, minus `--exclude`d constraints. **Adding a new grouping requires no new
  flag and no code change** — define the group (tag constraints, or add a derived predicate) and
  pass its name. `core`, `soft`, `test`, `test-2` are all just names.
- No `--groups` given → default to the `default` group (every production constraint) — identical
  selection to today's full `DEFAULT_STAGES` union.
- `--exclude X` subtracts `X` from the resolved set (unchanged semantics, now applied to the union).
- `--staged` / severity staging is **kept working**: `severity_solver_stages()` is reimplemented in
  terms of `resolve_group('severity_N')` and remains a naturally non-overlapping ordered selection.
  Extending `--staged` to arbitrary ordered group-lists is a small follow-on, explicitly out of scope.
- `--stages` / `--stage-only` / `--skip-stage` keep working by treating the legacy stage names as
  group names (they are the same strings — `critical_feasibility`, `club_day`, …).

## Definition of Done

1. **Registry** (`constraints/registry.py`): `ConstraintInfo.groups: frozenset[str]` field added;
   every production constraint tagged per §1 (the spec-021 survivors' tags read from the landed
   spec-021 state, not guessed). `resolve_group`, `resolve_groups`, `list_group_names`, and
   `DERIVED_GROUPS` (incl. `severity_1..5`, `default`/`all`/`production`) implemented.
   `resolve_groups` returns the **deduped union in canonical (registry insertion) order**.
2. **Dedup guarantee, tested:** `resolve_groups(['core','soft'])` equals the hand-computed union
   with **no duplicates**; a constraint tagged into two requested groups appears exactly once; the
   returned list is in registry order.
3. **Canonical-order validation:** a test (`validate_group_order()` or equivalent) asserts every
   known helper-var producer precedes its consumer in registry order — concretely
   `idx(ClubVsClubStackedWeekends) < idx(ClubVsClubStackedCoLocation)`, plus any other
   producer/consumer pair surfaced by `required_helpers`. Reordering the registry to break this
   fails the test.
4. **`soft_only` deleted everywhere:** removed from `stages.py` (`OPTIONAL_KEYS`,
   `apply_solver_stage`, `list_stages`), from `config/defaults.py` `DEFAULT_STAGES`, and from any
   season config. `grep -r soft_only constraints/ config/ run.py main_staged.py` returns nothing
   except this spec's note. Engine keys in any selected set now always run hard **and** soft.
5. **Whole-constraint dispatch:** `apply_constraint_set` (or the rewritten `apply_solver_stage`)
   applies a resolved/ordered/deduped list, running each engine key's hard+soft and each atom's
   full `apply`. A regression test proves that calling the dispatch function on a group set that
   includes an engine key (e.g. `ClubGameSpread` or `EqualMatchUpSpacing`) without `soft_only`
   emits hard constraints (count > 0), whereas the same call **with** a synthetic
   `{'soft_only': True}` stage skips them. **Note (review note — Low):** after spec-016/017/021
   land, *no real production constraint remains hard-and-stranded in `soft_optimisation`* — the
   "formerly suppressed" gap the spec fixed was `EnsureBestTimeslotChoices` (→ `VenueEarliestSlotFill`)
   and `ClubGameSpread` (moved to `club_day`), both fixed before spec-023 runs. The witness
   in this test must therefore be a **synthetic** fixture (a mini engine key + a
   `{'soft_only': True}` stage dict), not a real production stage. Use the existing
   `test_spacing_promoted_hard.py::test_soft_only_dispatch_skips_hard` pattern as a template.
   (review fix — Low: clarified that the "hard-bearing atom witness" must be a synthetic fixture
   since spec-016/017/021 have already evacuated all hard parts from `soft_optimisation`.)
6. **`validate_solver_stages` rewritten:** the **no-atom-in-two-stages rule is removed** (overlap
   is now legal); validation instead checks every group member is a registered canonical name,
   group names resolve, and the canonical-order helper-dep check (DoD 3) passes. The existing
   "unknown atom" / "well-typed keys" checks are kept.
7. **CLI:** `run.py generate --groups NAME...` implemented; resolves to the deduped union minus
   `--exclude`; no flag → `default` group. `main_staged.py` / `main_simple` consume the resolved
   set. `--staged`, `--stages`, `--stage-only`, `--skip-stage` still function (stage names treated
   as group names; severity staging via `resolve_group('severity_N')`). `--relax` left untouched
   (out of scope). `--list-stages` updated to also list group names (`--list-groups` or extend it).
8. **Behaviour parity:** a full-config 2026 build through the new group dispatch produces
   **exactly** the same hard-constraint count, penalty buckets, and tester violations as the
   pre-refactor build. The expected delta from removing `soft_only` is **zero** — spec-021 (done)
   moved `ClubGameSpread` to the non-`soft_only` `club_day` stage, so its hard part already runs;
   and the only other constraints in `soft_optimisation` are pure-soft atoms with no hard engine
   method. An integration test asserts this parity against a recorded baseline.
   (review fix — C4: old wording said "nil once spec-021 has landed" implying spec-021 was not
   yet done. spec-021 is done; the zero-delta is already the case, not a future condition.)
9. **Metadata:** draw metadata records the **resolved group selection** (`groups_selected`, the
   list of group names) and the **applied constraint set** (canonical names), alongside the
   existing `constraints_applied`. The tester/`save_solver_output` read group membership from the
   registry, not from a re-derived stage list.
10. `len(CONSTRAINT_REGISTRY)` unchanged (no entries added/removed — this is pure machinery).
    Full suite green; import smoke (`import constraints.registry, constraints.stages,
    constraints.unified, config.defaults`).

## Implementation units

> Heavily-shared files (`registry.py`, `stages.py`, `defaults.py`, `run.py`, `main_staged.py`).
> Sequence as one worktree, commit per unit; spec-021 and spec-024 are done — rebase on
> tip-of-final-form before starting, then re-run `validate_solver_stages` to confirm `[]`.

### Unit A — Registry: `groups` field, tagging, resolver, canonical-order validation
- Files: `constraints/registry.py`.
- Add `groups: frozenset[str]`; tag **every** production constraint per the §1 table — including
  the spec-021/024 survivors (`VenueEarliestSlotFill`, `ClubGameSpread`, `ClubNoConcurrentSlot`),
  already registered in `constraints/registry.py` (confirmed landed). §1 gives the expected
  `{core}` tags; reconcile against the actual registry before tagging. Implement `resolve_group`,
  `resolve_groups` (deduped union, registry order), `list_group_names`, `DERIVED_GROUPS`, and the
  producer-before-consumer order validator.
- Test (GWT, no mocks, hand oracle): `resolve_groups(['core','soft'])` == hand-listed union, no
  dupes, registry order; a synthetic two-group overlap yields one copy; the order validator passes
  on the real registry and fails on a deliberately-swapped fixture.

### Unit B — Dispatch: delete `soft_only`, `apply_constraint_set`, rewrite `validate_solver_stages`
- Files: `constraints/stages.py`, `config/defaults.py` (remove `soft_only`; keep `DEFAULT_STAGES`
  atom lists as the legacy-named groups), any season config carrying `soft_only`.
- Depends on Unit A.
- Test (GWT, hand oracle): an engine key + a hard-bearing atom selected via a `soft`-tagged set
  both emit their hard constraints (count > 0); the rewritten `validate_solver_stages` accepts an
  overlapping-group config and still rejects an unknown-atom config; `validate_solver_stages(...)`
  on the real default config == `[]`.

### Unit C — CLI `--groups` + main wiring; preserve `--staged`/`--stages`/`--exclude`
- Files: `run.py` (`--groups`, `--list-groups`/extend `--list-stages`), `main_staged.py`
  (`run_solver_stages` loop, receives resolved set), `constraints/stages.py::severity_solver_stages`
  (reimplement via `resolve_group`).
- **`main_simple` / `_main_simple_unified` (critical wiring note):** `main_simple` in
  `main_staged.py` does NOT route through `apply_solver_stage` — it calls
  `_main_simple_unified` which calls `UnifiedConstraintEngine.apply_phase_a()` /
  `apply_phase_b()` / `apply_phase_c()` directly (i.e. `apply_stage_1_hard()` /
  `apply_stage_2_soft()` with no stage filter). The `--groups` / `--stages`-resolved set is
  therefore NOT threaded through `main_simple` today, and that gap exists independently of
  spec-023.
  - **RESOLVED 2026-05-23 (convenor): Option B (full).** `--simple` is intended to be THE mode
    where the operator selects exactly which constraint groups to apply, and that capability must
    be kept. So `--groups` MUST be wired into the simple/unified path, not scoped out of it.
    Implement: add a `constraint_names` (resolved deduped union from `resolve_groups`) kwarg to
    `main_simple` / `_main_simple_unified`, and propagate it as an `engine.skip_constraints` filter
    so the `UnifiedConstraintEngine.apply_phase_a/b/c()` calls apply only the selected whole
    constraints. `--groups` must produce identical selection in both `--simple` and `--staged`.
    (Option A — scoping `--groups` out of `--simple` — is REJECTED per this decision.)
  - Add `main_staged.py`, `main_staged.py::main_simple`, and `main_staged.py::_main_simple_unified`
    to this unit's file list (the original unit list omitted them). The staged path
    (`run_solver_stages`) in `main_staged.py` must ALSO be updated to accept the resolved set, so
    both dispatch paths share one resolution.
  - DoD addition: a test asserts `--simple --groups core` applies exactly the `core` canonical set
    (and NOT soft-only constraints), proving the simple path honours the selection.
  (review fix — High: `main_simple` bypass of `apply_solver_stage` was not mentioned in the
  plan; the dispatch path routes through `_main_simple_unified` → `apply_phase_a/b/c` directly,
  and `--groups` would have no effect on `--simple` runs without explicit wiring.)
- Depends on Units A+B.
- Test: `--groups core` selects only core canonical names; `--groups test test-1` (two ad-hoc
  groups defined in a test fixture) yields the deduped union; no `--groups` == full `default` set;
  `--exclude X --groups core` removes X; `--staged` still produces severity-ordered selections.

### Unit D — Metadata + tester read-through + docs
- Files: `analytics/storage.py` / `analytics/versioning.py` (`groups_selected` +
  applied-constraint-set in metadata), `analytics/tester.py` (read group membership from registry),
  `constraints/severity.py` (read severity-as-group through the registry where it inferred before).
- Depends on Units A–C.
- Test: a generated draw records `groups_selected`; `test_every_drawtester_check_in_registry` /
  registry-count tests pass; the DoD-8 parity integration test green.

## Open decisions (recommendations baked in)

- **A. Explicit `order: int` per constraint vs registry insertion order?** Recommendation: **use
  registry insertion order** + the producer-before-consumer validator (DoD 3). It needs no
  per-entry bookkeeping and the registry is already in dependency order. Add an explicit `order`
  field only if a future grouping needs an apply order that diverges from registry order — note it,
  don't build it now.
- **B. `--groups` varargs vs convenience flags (`--core`, `--soft`)?** Recommendation: **single
  `--groups NAME...`** as the general mechanism (so a brand-new `--test-2` grouping is a config
  edit, never a code change). `--core`/`--soft` may be added later as thin aliases for
  `--groups core` / `--groups soft` if the convenor wants the shorthand — out of scope here.
- **C. Keep legacy stage names as groups, or rename?** Recommendation: **keep them as group tags**
  (`critical_feasibility`, `home_away_balance`, `club_alignment`, `club_day`, `soft_optimisation`)
  so `--stages`/`--stage-only`/`--skip-stage` keep working with zero churn; layer `core`/`soft` on
  top as additional, overlapping tags.

## Precondition (review)

spec-021 and spec-024 are **both fully landed** as of 2026-05-23 — this spec is startable now.
The checklist below is left as a verification guide for the implementer (re-confirm by reading
the repo before starting Unit A):

- `VenueEarliestSlotFill` atom exists in `constraints/atoms/venue_earliest_slot_fill.py` and is
  wired into `critical_feasibility` (non-`soft_only`) in `DEFAULT_STAGES`. ✓ VERIFIED
- `_best_timeslot_choices_hard` and `_best_timeslot_choices_soft` methods are **removed** from
  `constraints/unified.py`, and `EnsureBestTimeslotChoices` is removed from `ENGINE_HARD_KEYS`
  and `ENGINE_SOFT_KEYS` in `constraints/stages.py`. ✓ VERIFIED
- `ClubGameSpread` is wired into a **non-`soft_only`** stage (`club_day`) in `DEFAULT_STAGES`.
  `_club_game_spread_hard` and `_club_game_spread_soft` engine methods **remain in
  `unified.py`** — spec-021 DoD 6 says "refactored," not deleted; spec-024 re-scoped them to
  per-field. Their presence is correct and intentional. The completion signal is NOT method
  removal but the non-`soft_only` staging. ✓ VERIFIED (`club_day` has `soft_only: False`)
- `ClubNoConcurrentSlot` atom exists in `constraints/atoms/club_no_concurrent_slot.py` and is
  registered in `constraints/registry.py`. ✓ VERIFIED
- `validate_solver_stages(DEFAULT_STAGES) == []`. ✓ VERIFIED

**Why `soft_only` deletion is behaviour-neutral:** the `soft_optimisation` stage carries only
atoms (`SoftLexMatchupOrdering`, `NIHCFillWFBeforeEF`, `NIHCFillEFBeforeSF`,
`TeamPairNoConcurrency`, `PreferredWeekendsAwayGround`, `PreferredTimes`, `PreferredGames`)
that are pure soft/penalty constraints. `ClubGameSpread` (the only atom that had a live hard
engine method) was moved to `club_day` by spec-021; `EnsureBestTimeslotChoices` was replaced by
`VenueEarliestSlotFill` in `critical_feasibility`. Removing `soft_only` from `soft_optimisation`
therefore changes nothing: `apply_stage_1_hard()` will be called for `soft_optimisation` but
those atoms' engine-key hard methods are either already absent (no dispatch path) or trivially
return 0 for pure-soft engine keys like `PreferredTimesConstraint`. Verify this by checking the
`soft_optimisation` atom list against `ENGINE_HARD_KEYS` before starting Unit B.

(review fix — C2/C3/C4: prior Precondition said "spec-021 is `in_progress`", "NOT YET
STARTABLE", and required `_club_game_spread_hard/_soft` to be deleted — all three were wrong.
spec-021 and spec-024 are both `done`; the engine methods are intentionally kept (refactored);
the staging is correct; the spec is startable. Root cause: the old review was written before
spec-021 landed, and neither that review nor the subsequent review-fix caught that spec-021 had
completed and that its DoD #6 said "refactored" not "deleted.")

## Doc registry

- `docs/system/STAGES.md` — rewrite: stages → **groups** (named, overlapping, deduped-union,
  flag-selected); document `--groups`, the `core`/`soft`/`severity_N`/`default` groups, canonical
  (registry) apply order, and the `soft_only` removal. Record the "select whole constraints; split
  the atom if you must select sub-parts" principle.
- `docs/system/CONSTRAINT_INVENTORY.md` — add a per-constraint **groups** column; note group
  membership is the SSoT for selection; update §3 if counts/sections shift.
- `docs/system/HELPER_VARS.md` — note the producer-before-consumer canonical-order guarantee that
  the order validator enforces (Weekends → CoLocation).
- `CLAUDE.md` — replace the Constraint-System severity/stage wording: a solve applies the deduped
  union of selected groups (whole constraints, hard+soft together); `soft_only` is gone; `--groups`
  is the selector; `--slack` still loosens within a constraint. Update the Quick-Commands block
  with a `--groups` example.
- `docs/operator-ai/CONSTRAINT_APPLICATION.md` — "how to add a constraint": tag it into groups in
  the registry instead of slotting it into a single stage; selection is by group, not partition.
- `docs/todo/GOALS.md` — update the spec-023 row (status `ready`, new title); add a §2 worked
  example: "groups replace the stage partition; a constraint is whole; split only to select
  sub-parts." Keep §2 step 5 ("wire into a stage") accurate by rewording to "tag into groups."

## Out of scope

- **`--relax`** — dropped from this spec entirely (per convenor: it's about slack, which is the
  separate `--slack` lever; leave `--relax` untouched, neither reworked nor removed here).
- **Re-implementing `--staged` on arbitrary ordered group-lists** — the machinery makes it a small
  follow-on; this spec only keeps the existing severity-based `--staged` working. If the convenor
  wants ordered-group staging, spawn a new spec then.
- **Convenience alias flags `--core` / `--soft`** — `--groups core` / `--groups soft` cover it;
  aliases are a later cosmetic add (Open decision B).
- **Any hard/soft phase API on atoms** — explicitly rejected; this spec deletes the hard/soft
  machinery rather than formalising it. A constraint is whole.
- **Changing which rules are hard or soft** — that is spec-016/017/021's domain; spec-023 is a
  pure machinery refactor with behaviour parity (DoD 8).
- **Atomising the remaining World-1 engine-method constraints** (`EqualGamesAndBalanceMatchUps`,
  the Broadmeadow min/max, `PreferredTimes`) — separate atomization specs; they dispatch through
  the engine unchanged here.
- **Explicit per-constraint `order` field** — registry insertion order suffices (Open decision A);
  build only if a future grouping needs divergent ordering.
