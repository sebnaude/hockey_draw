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
| `critical_feasibility` | Hard feasibility prerequisites for any valid draw — no team/field double-booking, every team plays its required games against the right opponents, season-wide PHL/2nd-grade concurrency at Broadmeadow, same-club PHL/2nd back-to-back at one venue or ≥3 h apart across venues, byes spread evenly. spec-021 added `VenueEarliestSlotFill` (games pack into earliest slots — no gaps, earliest start, structurally avoids 7pm) and `ClubNoConcurrentSlot` (a club's games per timeslot/venue capped). Non-negotiable. (spec-032: repeat-meeting spacing, `EqualMatchUpSpacing`, was peeled out of `critical_feasibility`/`core` into its own lonesome `spacing` group — see below.) |
| `home_away_balance` | Home/away fairness for clubs whose home venue is not Broadmeadow — per-pair and per-team aggregate balance, per-club season home-weekend totals. |
| `club_alignment` | Cross-grade same-club / same-opponent alignment at Broadmeadow — which weekends two clubs' grades stack, and how they co-locate. Order-sensitive: stacking decision before the co-location decision that reads its helpers. |
| `club_day` | Per-club day-of-week preferences — participation, intra-club matchups, opponent routing, field consistency, slot contiguity. spec-021 moved `ClubGameSpread` here (its hard part was dead in `soft_optimisation`). |
| `soft_optimisation` | Soft penalties / optimisation only — preferred & avoided dates/weekends, preferred times, multiple clubs per Broadmeadow timeslot without monopolising a field, per-team-pair no-concurrency requests. (spec-032: the alphabetical matchup tie-break and the NIHC WF→EF→SF fill-order symmetry-breakers moved out into the dedicated `symmetry_breakers` group.) |
| `core` | The deduped union of every constraint a normal full build needs hard+soft (overlaps the feasibility/balance/alignment/day groups). spec-032: no longer includes `EqualMatchUpSpacing` (now in `spacing`). |
| `soft` | The soft-optimisation constraints (overlaps `soft_optimisation`). spec-032: no longer includes the three symmetry-breakers (now in `symmetry_breakers`). |
| `spacing` | **spec-032 — explicit, lonesome.** Just `EqualMatchUpSpacing` (repeat-meeting spacing; hard since spec-017, `--slack EqualMatchUpSpacingConstraint N` relief). Peeled out of `core` so the convenor can select or drop it independently. Still in `severity_1` (severity unchanged) and in `default` (via the widened `_is_fresh_build` predicate). |
| `symmetry_breakers` | **spec-032 — explicit, always-on.** The three objective-shaping tie-breakers `NIHCFillWFBeforeEF`, `NIHCFillEFBeforeSF`, `SoftLexMatchupOrdering`. The CLI unions this group into **every** solve regardless of `--groups` (so `--groups core` still gets them), unless `--no-symmetry-breakers` is passed. Also reaches `regen` (widened regen predicate) and `default`. |
| `default` / `all` / `production` | Every production constraint — the no-`--groups` selection. spec-032: still the full set (`core ∪ soft ∪ spacing ∪ symmetry_breakers`); the retag is membership-preserving via the widened `_is_fresh_build` predicate. |
| `severity_1` … `severity_5` | Derived: constraints at each severity level. |

Season configs may still override the legacy stage list by setting
`'solver_stages'` in `SEASON_CONFIG` (full replace, no merge) — those names are
treated as group names.

### Always-on symmetry breakers + `--no-symmetry-breakers` (spec-032)

The `symmetry_breakers` group is a pure objective-shaping bundle (three
tie-breakers, no feasibility effect), so it should shape **every** solve. Rather
than rely on it being in whatever `--groups` you picked, `run.py`'s
`_resolve_group_selection` **always unions `resolve_group('symmetry_breakers')`
into the selection** — even `--groups core` (which does not itself contain them)
comes out with the three tie-breakers applied. The metadata `groups_selected` /
`constraints_applied` reflect them.

`--no-symmetry-breakers` reverses this in **both** dispatch paths:

- **`--groups` path:** the three names are added to the exclude set, so they are
  dropped even from groups that carry them (e.g. `default`).
- **Plain (no `--groups`) path:** that path normally runs `DEFAULT_STAGES`
  untouched (staged filter `None`), and `DEFAULT_STAGES` still contains all three
  tie-breakers — so suppression must **force** the staged filter non-`None` to
  `default`-minus-symmetry. `run.py` does this when `--no-symmetry-breakers` is
  set with no `--groups`. Without the flag, the plain path is byte-for-byte the
  legacy behaviour (symmetry breakers applied via `DEFAULT_STAGES`).

Note `DEFAULT_STAGES` itself is unchanged — spacing and the symmetry atoms still
live in it, so a plain solve still applies them; the only lever on the plain path
is `--no-symmetry-breakers`.

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
run.py generate --year 2026 --staged --groups core   # honoured identically in every mode

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

## Regeneration group (`regen`) — spec-027

A regeneration run (`--regen-from`, spec-026) automatically selects the **`regen`
constraint group** via `run.py::_select_regen_group()` →
`resolve_groups(['regen'])`. The caller does not need to pass `--groups regen`
explicitly.

`regen` is a DERIVED group (predicate in `DERIVED_GROUPS`): a constraint is in
`regen` iff its `groups` set intersects `{'core_hard', 'regen_soft', 'soft'}`.
This resolves to 32 constraints — 12 `core_hard` entries that stay hard
(genuine physical constraints), 13 `regen_soft` penalty atoms that replace the
hard rules that assumed full solver freedom (adjacency, spacing, club-day
clustering, co-location, venue fill), and the 7 `soft` entries from the normal
soft-optimisation stage.

**Dispatch:** regen runs are always routed through `apply_constraint_set` with a
single synthetic `'regen'` stage. Regen ignores the solve-mode flags
(`--staged`/`--severity`) and always uses the staged dispatcher, because the
non-engine `regen_soft` atoms can only be applied through that path.

**Normal (non-regen) runs never select `regen`.** The `regen_soft` atoms carry
only `{'regen_soft'}` as their groups tag — no `core`, `club_day`,
`critical_feasibility`, or any other production tag. They are invisible to
`default`, `all`, `production`, and every named stage group.

The old `WARNING: spec-027 regen group not available …` fallback in `run.py` is
gone; `_select_regen_group()` returns the resolved set directly.

Full reference (core-hard table, regen-soft atom table, engine-key design note
for `EqualMatchUpSpacing`/`ClubGameSpread`): `docs/system/REGEN_CONSTRAINTS.md`.

## Config validation phase

`utils.py::validate_game_config` Phase 22 `_validate_stages` runs
`validate_solver_stages(data['solver_stages'])` when the key is present; errors
become FATAL.
