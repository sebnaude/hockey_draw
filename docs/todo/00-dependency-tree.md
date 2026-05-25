<!-- status: index -->

# Spec dependency tree

Aggregated `depends_on` view of every live (`ready`/`not_ready`/`in_progress`/`delayed`) spec in
`docs/todo/`. Source of truth for *ordering* is each spec's own `depends_on` header; this file is
the cross-spec view so independent work runs in parallel rather than accidentally serialised.
Keep it current: update an edge whenever a spec is created, completed, re-scoped, or unblocked.
**No cycles** — if you find one, two specs are really one unit; collapse or re-cut them.

## Live specs

Authored **2026-05-24** (this session) — a PHL/2nd constraint cleanup and a constraint-group
restructure, cut into three serialised specs:

- **spec-030** — PHL/2nd cleanup: 2.5 h cross-venue gap, delete the redundant
  `PHLAnd2ndConcurrencyAtBroadmeadow` atom, locked-week skip on `PHLConcurrencyAtBroadmeadow`.
  `depends_on: none`. Single unit, S2. Status: `review_pending`.
- **spec-031** — remove the `ClubFieldConcentration` tester-only diagnostic. `depends_on:
  spec-030` (shares `registry.py`/`CONSTRAINT_INVENTORY.md`; the registry count test moves
  50→49 after spec-030's 51→50). spec-029 — which had edited the same `analytics/tester.py` —
  is now **done** (`ccc3d07`), so that contention is resolved; branch from current source.
  Single unit, S2. Status: `review_pending`.
- **spec-032** — constraint group restructure: new `symmetry_breakers` group (always-on, with a
  `--no-symmetry-breakers` escape) + `core` minus `EqualMatchUpSpacing` + lonesome `spacing`
  group. `depends_on: spec-031` (shares `registry.py`/`CONSTRAINT_INVENTORY.md`/`CLAUDE.md`).
  Two units (A registry → B run.py), S3. Status: `done` (merged into final-form
  2026-05-25; A `9efee82`, B `2b39008`; both passed /adversarial Mode B).
- **spec-033** — uniform soft+slack treatment of deviation-tolerant constraints. Five serial units:
  (A) remove `ClubVsClubAlignment` slack (dead in solver, live+divergent in tester) + verify
  EqMatchup/ClubGameSpread-contiguity soft; (B) `BalancedByeSpacing` — add normal-mode soft analogue +
  set base slack 0→2 (raw-ideal floor risks infeasibility) + retag out of `core` into its own
  `{bye_spacing}` group (widening `_is_fresh_build`, after spec-032) + no-play-weekend bye-count check;
  (C) `TeamConflict` → soft-only (drop hard feasibility); (D) `ClubGameSpread` hard ≤2-field cap
  (+slack) closing the field-concentration structure (soft push→1 already exists); (E)
  `ClubNoConcurrentSlot` → soft+slack (hard ≤1 overlap, push→0; drops capacity-aware floor +
  `core_hard`; covers cross-field non-overlap). `depends_on: spec-032` (shares `registry.py`/
  `run.py`/`config/season_2026.py`/`analytics/tester.py`/`CONSTRAINT_INVENTORY.md`/`CLAUDE.md`).
  Five units A→B→C→D→E (strictly serial, shared files), S3. Status: **`done`** (2026-05-25 —
  all 5 units merged into final-form + /adversarial Mode B verified; archived to `done/`). Each
  unit also got a Mode-B-driven fix applied inline (Unit A stale slack-dict tests; Unit C
  TeamConflict soft-pressure tester test + a default-stages parity repair `89fdfd6` for a Unit-C
  regression caught during Unit D; Unit D perennial-defaults guard keys; Unit E registry-count
  narrative 38→49 + stale ClubNoConcurrentSlot comments). Surfaced for the convenor: TeamConflict &
  ClubNoConcurrentSlot are no longer feasibility blockers (intended), ClubNoConcurrentSlot left the
  `regen` group entirely (no regen-soft analogue), and the hard ≤1+slack overlap cap makes a venue
  with fewer slots than a club's games infeasible at slack 0 — slack is the release; first real
  full solve is spec-035.

- **spec-036** — default solve = single-solve over the FULL modern set (no-flag), and remap the
  solve-mode flags: no-flag → single solve (was DEFAULT_STAGES staged), `--staged` → DEFAULT_STAGES
  incremental (was severity), new `--severity` → severity staging; removed `--simple`/`--unified`
  (forward-only). Rerouted the single-solve builder through `apply_solver_stage` so it applies the
  same complete set as staged (closed the 15-atom gap the old `--simple` had), then deleted the
  now-dead legacy `_club_alignment_hard/_soft` + orphaned groupings + ENGINE_HARD/SOFT_KEYS membership
  (superseded by the spec-005 stacked atoms). `depends_on: spec-033`. Three units A→B→C, S3.
  Status: **`done`** (2026-05-25 — A `f4205a9`/`1c87d62`/`ed35167`, B `6cbbdf0`/`312bdf0`,
  C `3323270`/`f353c3f`/`e378d8f`; all /adversarial Mode B verified; archived to `done/`). In-flight
  S1 fix: `severity_solver_stages()` alphabetical ordering crashed `--severity` on the
  `ClubVsClubStackedWeekends`/`CoLocation` pair → now ordered by `canonical_index`. Registry count
  unchanged (49 — `ClubVsClubAlignment` entry retained as tester/name anchor). Convenor note: the
  no-flag default now applies the COMPLETE set and is INFEASIBLE at slack 0 on the 2026 production
  config (same documented state as spec-033; slack releases it; forced-free real solve = spec-035).

Two **special end-of-line plans** authored **2026-05-24** (this session). They are gated behind
*everything else* and run last, in order. **spec-035 now also sequences after spec-036** — its raw
single-solve e2e depends on the no-flag/single-solve path being correct (its `depends_on` should add
`spec-036` when next edited):

- **spec-034** — PENULTIMATE: green test suite + honest coverage (≥85% on atoms/registry/stages/
  tester) + three real-data, no-mock assurances (atoms enforce on real data; `DrawTester` detects a
  failed constraint in a draw; soft constraints are measured via `soft_pressure`). `depends_on:
  spec-030, spec-031, spec-032, spec-033` — the suite shape (atom set, registry counts, group
  membership, tester checks) is only final once the whole chain lands. Five units (A fixtures/cov →
  B/C/D assurances in parallel → E green-up). S3. Status: `review_pending`.
- **spec-035** — ULTIMATE: raw `--core` e2e solve on the forced-free `season_test` config
  (2026 base teams/fields, no forced games, **week 1 NOT fixed** — no `--fix-round-1`/locks,
  `--workers 10`). Goal: get through presolve + survive ≥30 min of search (killed at 30 min
  regardless of solution), and read out how much symmetry the model has left (CP-SAT presolve stats
  — **not currently captured to the log**, so Unit B wires that up; **no historical baseline exists**,
  confirmed by scan, so we record the first baseline). `depends_on: spec-030…033 + spec-034`. Three
  units (A launcher ∥ B symmetry-capture → C run+readout). S3. Status: `review_pending`.

```
spec-030  ──depends_on──▶  (none)                              [done — merged 5362d41]
spec-031  ──depends_on──▶  spec-030                             [done — merged into final-form 2026-05-24]
spec-032  ──depends_on──▶  spec-031                             [done — merged into final-form 2026-05-25 (A 9efee82, B 2b39008)]
spec-033  ──depends_on──▶  spec-032                             [done — merged into final-form 2026-05-25 (A 6037235, B c4af289, C b3a15c1, D f760896, E 8316911)]
spec-036  ──depends_on──▶  spec-033                             [done — merged into final-form 2026-05-25 (A f4205a9, B 312bdf0, C e378d8f)]
spec-034  ──depends_on──▶  spec-030, spec-031, spec-032, spec-033          [review_pending]   (PENULTIMATE — UNBLOCKED: 030-033 all done)
spec-035  ──depends_on──▶  spec-030…033, spec-034, spec-036                [review_pending]   (ULTIMATE — last; spec-036 now done, still blocked on spec-034)
```

The 030→031→032 chain is a deliberate serialisation: all three edit `constraints/registry.py`
and `docs/system/CONSTRAINT_INVENTORY.md` (032 also touches `CLAUDE.md`), so they merge one at a
time to avoid registry/count/doc conflicts. The registry count test moves 51 → 50 (spec-030) →
49 (spec-031); spec-032 retags only (count stays 49). When each lands, mark it done, move it to
`docs/todo/done/`, drop its node, and note the next-unblocked spec. **spec-034 unblocks only when
030-033 are all `done`; spec-035 unblocks only when 030-034 are all `done` — it is the final plan,
after which the `final-form` plan line is fully drained.**

Most recently completed:

- **spec-036** — single-solve default + solve-mode flag remap + legacy-alignment deletion. **Done**
  (2026-05-25, Units A `f4205a9`/`1c87d62`/`ed35167`, B `6cbbdf0`/`312bdf0`, C `3323270`/`f353c3f`/`e378d8f`
  merged into final-form, each /adversarial Mode B verified). No-flag now = single full solve of the
  complete modern set (closed the old `--simple` 15-atom gap); `--staged` = DEFAULT_STAGES incremental;
  new `--severity` = severity-staged; `--simple`/`--unified` removed. Deleted the dead legacy
  `_club_alignment_*` engine path (superseded by spec-005 stacked cluster; registry stays 49). In-flight
  S1 fix: severity-stage atom ordering (alphabetical → `canonical_index`) so `--severity` no longer
  crashes on the stacked pair. `depends_on: spec-033`.

- **spec-033** — uniform soft+slack treatment of deviation-tolerant constraints. **Done**
  (2026-05-25, all 5 units A–E merged into final-form, each /adversarial Mode B verified).
  Unit A `6037235` (ClubVsClubAlignment slack removed → fixed hard rule); Unit B `c4af289`
  (BalancedByeSpacing normal-mode soft analogue + base slack 0→2 + own `bye_spacing` group);
  Unit C `b3a15c1` (TeamConflict → soft-only) + parity repair `89fdfd6`; Unit D `f760896`
  (ClubGameSpread hard ≤2-field concentration cap); Unit E `8316911` (ClubNoConcurrentSlot →
  soft + slack, hard ≤1 overlap push→0). Convenor-facing consequences (intended): TeamConflict
  & ClubNoConcurrentSlot stop being feasibility blockers; ClubNoConcurrentSlot leaves the `regen`
  group; the ≤1+slack overlap cap makes an under-slotted venue infeasible at slack 0 (slack
  releases it; first real solve = spec-035). `depends_on: spec-032`.

- **spec-032** — constraint group restructure. **Done** (2026-05-25, Units A `9efee82`
  + B `2b39008` merged into final-form). Peeled `EqualMatchUpSpacing` into a lonesome
  `spacing` group and the three tie-breakers into an always-on `symmetry_breakers`
  group (CLI unions them into every solve unless `--no-symmetry-breakers`); widened
  `_is_fresh_build` + the `regen` predicate so `default` (27) and `regen` (31) are
  membership-unchanged. Both units passed /adversarial Mode B. `depends_on: spec-031`.

- **spec-029** — club-day weekends in the published-draw Notes column. **Done** (2026-05-24,
  commit `ccc3d07`). Added a sixth `Club Day` notes category auto-derived from `CLUB_DAYS`
  (opt-in `'note'` field), converted the 2026 `CLUB_DAYS` to dict form, and routed six
  direct-access `club_days` callsites through `normalize_club_day`. `depends_on: none`
  (extended spec-028).

- **spec-027** — regeneration soft-constraint group. **Done** (2026-05-24, merged `c851f25`).
  Delivered the `regen` constraint group (`core_hard` ∪ `regen_soft` ∪ `soft`), 13 new
  `*RegenSoft` soft-analogue atoms, the `core_hard` tags + `regen` derived group in the registry,
  the `--regen-from` → staged-dispatch wiring (the engine-only `--simple` path can't dispatch the
  non-engine RegenSoft atoms), and the DoD-7 infeasible→feasible witness. Depended on spec-023 +
  spec-025 + spec-026 (all `done` before it started). See
  `docs/system/REGEN_CONSTRAINTS.md` for the full reference.

Earlier (all `done`, in `docs/todo/done/`): spec-001…022, **spec-023** (constraint-groups
machinery, merged `083bf5a`), **spec-024** (field-spread), **spec-025** (`LOCKED_PAIRINGS`,
`7afc656`), **spec-026** (unified regeneration mode, `6f39b83`), **spec-028** (per-weekend notes
export column, `c077c28`).

## Ready to start in parallel right now

- **spec-034 — PENULTIMATE** (`depends_on: spec-030, spec-031, spec-032, spec-033` — ALL `done`,
  so it is UNBLOCKED). Green test suite + honest coverage + real-data assurances. Status
  `review_pending`, unowned. It is the penultimate plan; per the run instruction it should only be
  picked up once it "makes sense" (its deps satisfied — now true). **This is the only startable plan.**
- **spec-035 — ULTIMATE** is still blocked: `depends_on` includes spec-034 (not yet `done`).
  spec-036 — its other former blocker — is now `done`. spec-035 runs last; after it lands the
  `final-form` plan line is fully drained.
- Ordering note: with spec-036 `done`, spec-034 is the sole remaining startable plan; spec-035
  then unblocks once spec-034 lands. One plan per session; stamp `owner` before building.
