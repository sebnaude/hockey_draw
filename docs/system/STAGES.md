# Constraint groups — named, overlapping, deduped-union, flag-selected

> **spec-023 reframe.** What used to be a strict *stage partition* is now a set
> of **named constraint groups**. A group is a named, possibly-**overlapping**
> set of WHOLE constraints. A solve applies the **deduped union** of the
> selected groups, in one canonical order; selecting the same constraint via two
> groups applies it **exactly once**. The legacy stage names
> (`critical_feasibility`, `home_away_balance`, `club_alignment`, `club_day`,
> `soft_optimisation`) are still valid group names — they keep `--stages` /
> `--stage-only` / `--skip-stage` working — but they are no longer a partition:
> the coarse `core` / `soft` / `severity_N` / `default` groups overlap them
> freely.

## The whole-constraint principle

A constraint — whether one atom or a bundle of atoms — is applied as a **whole**:
its hard part sets the cap, its soft part improves within the cap, and the two
are never applied separately. A group tag selects *whole* constraints; it never
changes how much of a constraint applies. **If you genuinely need to apply one
idea of a constraint without another, split it into two atoms** — do not give an
atom a hard/soft phase switch. (This is exactly why the old `soft_only` lever was
deleted — see below.)

`--slack` is orthogonal: it loosens *within* a constraint and is unaffected by
group selection.

## Group membership is registry metadata (single source of truth)

Each `ConstraintInfo` in `constraints/registry.py` carries
`groups: frozenset[str]`. Tags are the only hand-maintained membership store.
Derived groups are **computed, never stored**, so there is no second source of
truth:

- **Explicit tags** — `core`, `soft`, plus the legacy stage names
  (`critical_feasibility`, `home_away_balance`, `club_alignment`, `club_day`,
  `soft_optimisation`) and the coarse dimensions (`home_away_balance`,
  `club_alignment`, `critical_feasibility`, `soft_optimisation`).
- **Derived predicates** — `DERIVED_GROUPS` in the registry:
  - `severity_1` … `severity_5` = every constraint whose
    `ConstraintInfo.severity_level == n`.
  - `default` / `all` / `production` (alias) = every constraint with a non-empty
    `groups` set (i.e. every production constraint). This is the selection a run
    with no `--groups` flag uses — identical to today's full build.

> **Groups and severity levels are orthogonal axes.** A constraint's
> `severity_N` membership comes from its `ConstraintInfo.severity_level`, which is
> independent of its explicit stage/`core`/`soft` group tags. So a constraint can
> live in the `critical_feasibility` group yet have `severity_level == 2` (e.g.
> `BalancedByeSpacing`, `ClubNoConcurrentSlot`, `VenueEarliestSlotFill`) — it
> appears in `severity_2`, NOT `severity_1`. `--groups severity_1` therefore does
> NOT select every `critical_feasibility` constraint; use `--groups
> critical_feasibility` (or `core`) for that. Don't assume the stage group and the
> severity group coincide.

### Resolver API

```python
from constraints.registry import (
    resolve_group,        # one group name -> set[str] of canonical names
    resolve_groups,        # iterable of group names -> DEDUPED UNION, canonical order
    list_group_names,      # explicit tags ∪ derived names
    validate_group_order,  # producer-before-consumer helper-var dep check
)

names = resolve_groups(['core', 'soft'])   # deduped, canonical (registry) order
```

`resolve_groups` is the heart of the dedup-union guarantee: it builds a `set`
union of canonical names across all requested groups, then returns them sorted by
**canonical order** (registry insertion order). Asking for two groups that both
contain `ClubDayParticipation` yields it **once**.

## Canonical apply order = registry insertion order

When a deduped union is applied to one model, the only correctness-relevant
ordering is **producer-before-consumer for shared helper vars** (CP-SAT is
declarative; otherwise `Add*` order does not change the feasible region). The
single global canonical order is the **`CONSTRAINT_REGISTRY` dict insertion
order**, which already places producers before consumers (e.g.
`ClubVsClubStackedWeekends` precedes `ClubVsClubStackedCoLocation`, which reads
the helpers Weekends registers). `validate_group_order()` asserts every known
producer/consumer helper relationship has the producer at a lower registry index;
a reorder that breaks a dependency trips a red test. See
`docs/system/HELPER_VARS.md`.

## `soft_only` is gone

The old `soft_only: True` stage key told the engine to skip
`apply_stage_1_hard()` for engine-key constraints — silently suppressing the
*hard half* of a whole constraint. It was deleted (spec-023): there is no longer
any "hard pass / soft pass" toggle. Engine keys in **any** selected set always
run hard **and** soft. Removing it was behaviour-neutral (spec-021 had already
moved `ClubGameSpread` to the non-`soft_only` `club_day` group, and the
remaining `soft_optimisation` members are pure-soft atoms) — verified by the
DoD-8 parity test (`tests/test_groups_full_build_parity.py`): the staged-legacy
and staged-default builds emit an identical hard-constraint count.

## The default groups (intent)

The live group memberships are defined on each `ConstraintInfo.groups` in
`constraints/registry.py` — read the registry for the current canonical
membership rather than trusting a duplicated table here. The legacy-named groups'
*intent*:

| Group name | Intent |
|---|---|
| `critical_feasibility` | Hard feasibility prerequisites for any valid draw — no team/field double-booking, every team plays its required games against the right opponents, season-wide PHL/2nd-grade concurrency at Broadmeadow, same-club PHL/2nd back-to-back at one venue or ≥3 h apart across venues, byes spread evenly, repeat-meeting spacing (hard since spec-017, `--slack` relief). spec-021 added `VenueEarliestSlotFill` (games pack into earliest slots — no gaps, earliest start, structurally avoids 7pm) and `ClubNoConcurrentSlot` (a club's games per timeslot/venue capped). Non-negotiable. |
| `home_away_balance` | Home/away fairness for clubs whose home venue is not Broadmeadow — per-pair and per-team aggregate balance, per-club season home-weekend totals. |
| `club_alignment` | Cross-grade same-club / same-opponent alignment at Broadmeadow — which weekends two clubs' grades stack, and how they co-locate. Order-sensitive: stacking decision before the co-location decision that reads its helpers. |
| `club_day` | Per-club day-of-week preferences — participation, intra-club matchups, opponent routing, field consistency, slot contiguity. spec-021 moved `ClubGameSpread` here (its hard part was dead in `soft_optimisation`). |
| `soft_optimisation` | Soft penalties / optimisation only — preferred & avoided dates/weekends, preferred times, multiple clubs per Broadmeadow timeslot without monopolising a field, alphabetical matchup tie-break, NIHC fill order WF→EF→SF (soft symmetry-breaker), per-team-pair no-concurrency requests. |
| `core` | The deduped union of every constraint a normal full build needs hard+soft (overlaps the four feasibility/balance/alignment/day groups). |
| `soft` | The soft-optimisation constraints (overlaps `soft_optimisation`). |
| `default` / `all` / `production` | Every production constraint — the no-`--groups` selection. |
| `severity_1` … `severity_5` | Derived: constraints at each severity level. |

Season configs may still override the legacy stage list by setting
`'solver_stages'` in `SEASON_CONFIG` (full replace, no merge) — those names are
treated as group names.

## Dispatch

`constraints/stages.py::apply_constraint_set(canonical_names, *, model, X, data,
engine, applied_engine_keys, applied_atoms)` is the single dispatch entry point.
Given a resolved/ordered/deduped list of WHOLE constraints it:

- runs each **engine key**'s `apply_stage_1_hard()` **and** `apply_stage_2_soft()`
  (no `soft_only` toggle); and
- runs each **non-engine atom**'s full `apply(model, X, data, registry)` (the
  legacy solver class resolved via `solver_class_names` in the registry, e.g.
  `PreferredTimes`).

`applied_engine_keys` / `applied_atoms` are threaded across calls so a constraint
is never double-applied — this is what makes overlapping groups safe.
`apply_solver_stage(stage, …)` is a thin wrapper that resolves a stage's `atoms`
and calls `apply_constraint_set`, so the staged solver
(`StagedScheduleSolver.run_solver_stages_solve`) keeps working unchanged.

## CLI

```bash
# Select by group (varargs). Deduped union, minus --exclude.
run.py generate --year 2026 --groups core soft
run.py generate --year 2026 --groups core --exclude ClubGameSpread
run.py generate --year 2026 --simple --groups core   # simple path honours it too

# No --groups -> the `default` group (every production constraint), identical
# selection to the legacy full DEFAULT_STAGES union.
run.py generate --year 2026

# List the available group names (and the legacy stage list).
run.py generate --list-groups
```

**Adding a new grouping requires no new flag and no code change** — tag
constraints (or add a `DERIVED_GROUPS` predicate) and pass the name. `core`,
`soft`, `test`, `test-2` are all just names.

Legacy stage flags still function by treating the legacy stage names as group
names:

| Flag | Purpose |
|---|---|
| `--groups NAME...` | Select the deduped union of these groups (minus `--exclude`). |
| `--list-groups` | Print available group names and exit. |
| `--stages-config FILE` | Load a custom stage/group list from JSON. |
| `--stage-only NAME` | Restrict to a single stage/group by name. |
| `--skip-stage NAME` | Skip a stage/group by name (repeatable). |
| `--list-stages` | Print the resolved stage list and exit. |

`--staged` / severity staging is kept working: `severity_solver_stages()` is
reimplemented over `resolve_group('severity_N')` and remains a naturally
non-overlapping ordered selection. `--slack` and `--relax` are untouched.

## Validation

`validate_solver_stages(stages)` (spec-023 rewrite):

- Every stage/group has a unique `name` and a non-empty `atoms` list.
- Every member is a registered canonical name **or** a resolvable group name.
- The canonical-order helper-var producer/consumer check (`validate_group_order`)
  passes.
- Optional keys are well-typed.

The **no-atom-in-two-stages (no-overlap) rule is removed** — overlap is now
legal because a solve applies the deduped union. `utils.py::validate_game_config`
still runs `_validate_stages` (Phase 22) when `data['solver_stages']` is present;
errors become FATAL config errors.

## Config validation phase

`utils.py::validate_game_config` Phase 22 `_validate_stages` runs
`validate_solver_stages(data['solver_stages'])` when the key is present; errors
become FATAL.
