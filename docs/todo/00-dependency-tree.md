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

Two **end-of-line plans** authored **2026-05-24** (this session). They are gated behind
*everything else* and run last, in order. **spec-035 now also sequences after spec-036** — its raw
single-solve e2e depends on the no-flag/single-solve path being correct (its `depends_on` should add
`spec-036` when next edited):

- **spec-034** — green test suite + honest coverage (≥85% on atoms/registry/stages/
  tester) + three real-data, no-mock assurances (atoms enforce on real data; `DrawTester` detects a
  failed constraint in a draw; soft constraints are measured via `soft_pressure`). `depends_on:
  spec-030, spec-031, spec-032, spec-033` — the suite shape (atom set, registry counts, group
  membership, tester checks) is only final once the whole chain lands. Five units (A fixtures/cov →
  B/C/D assurances in parallel → E green-up). S3. Status: **`done`** (2026-05-26 — Unit A coverage
  infra + real-data fixtures + `scripts/run_green_suite.py`; B enforce+violate for the 7 uncovered
  atoms via the live engine; C detection for the 6 uncovered tester checks; D soft_pressure rollup;
  E green-up + docs). Verified registry count 49 / default set 28 at build time (forward-note
  honoured). Green-up also fixed 5 PRE-EXISTING failures left by the LOCKED_PAIRINGS removal
  (commit 5761e88) + registry recounts: 2 stale-oracle bumps (registry 50→49, default 27→28) and
  3 locked-pairings tests redirected to a frozen pre-removal artefact fixture
  (`tests/fixtures/locked_pairings_premigration_2026.json`) since the convenor edits live
  `season_2026.LOCKED_PAIRINGS` for manual testing. Coverage: honest 82.2% on the DoD-2 floor
  surfaces (registry 92%, stages 93%, atoms strong; `analytics/tester.py` 73.1% — its report/export
  surface isn't exercised by constraint tests; documented sub-floor per DoD-2, no padding).
- **spec-035** — raw `--core` e2e solve on the forced-free `season_test` config
  (2026 base teams/fields, no forced games, **week 1 NOT fixed** — no `--fix-round-1`/locks,
  `--workers 10`). Goal: get through presolve + survive ≥30 min of search (killed at 30 min
  regardless of solution), and read out how much symmetry the model has left (CP-SAT presolve stats
  — **not currently captured to the log**, so Unit B wires that up; **no historical baseline exists**,
  confirmed by scan, so we record the first baseline). `depends_on: spec-030…033 + spec-034`. Three
  units (A launcher ∥ B symmetry-capture → C run+readout). S3. Status: `in_progress` —
  Units A and B merged into `final-form`; Unit C (the actual solve) is BLOCKED on a presolve
  infeasibility surfaced in the run (`docs/todo/spec-035-e2e-infeasibility-handoff.md`). Two atoms
  identified as culprits: `AwayClubHomeWeekendsCount` (handled by spec-037) and
  `ClubVsClubStackedWeekends` (handled by spec-038). Resumes when both ship.

Authored **2026-05-28** — one of two atom redesigns surfaced by the spec-035 infeasibility handoff
still in flight (spec-037 the other half — already `done` 2026-05-28; see "Most recently completed"):

- **spec-038** — `ClubVsClubStackedWeekends` granularity rework: replaces `matchups × per_matchup`
  budget (game-count) with `max(team_count_A, team_count_B) × per_matchup` (aligned-weekend count);
  introduces a per-team-pair sub-budget + per-aligned-weekend cardinality (`min(a, b)` games per
  aligned weekend) so multi-team-per-club-per-grade cases stack correctly. PHL collapses to the old
  formula by construction (`a=b=1`). New helper-var key family `cvc_stack_team_pair_play`; existing
  `cvc_stack_play` key shape preserved (semantics shift transparently for the co-location consumer).
  `depends_on: none`. Four units A→B→C→D (A shared helpers, B atom rewrite, C co-location/regen-soft
  parity, D bisect-harness acceptance + docs). S3. Status: **`done`** (all code units merged
  2026-05-28: A `5391334`, B `2ebe31e`, club-day Layer-6 exemption `6f4c0e5`; status closeout
  2026-05-31, archived to `done/`).

Authored **2026-05-30** (this session) — the **draw-analysis / reporting engine** plan-set, cut into
four dependency-wired specs. Goal: every draw can be *verifiably* analysed (atom↔test alignment is a
gate, not a vibe), soft-constraint outcomes are measured + scored, the whole thing runs automatically
at end of draw creation into an HTML+JSON+Plotly report with per-team/club/rule breakdowns, and the
analysis always runs at the draw's own effective slack so it never throws false violations.

- **spec-039** — machine-checked **registry↔tester alignment gate**: a standing test asserts every
  atom's `tester_check_methods`/`slack_key`/`has_soft_component` stays truthful against the live
  `DrawTester`, the `run.py` slack-key set, and `PENALTY_WEIGHTS`; new `no_tester_check_reason`
  registry field labels deliberately-unchecked soft atoms; shared `AlignmentReport` (`analytics/registry_audit.py`)
  feeds both the gate and the report. `depends_on: none`. Two units (A registry+audit → B gate+doc), S2.
  Status: `ready` (Mode-A hardened 2026-05-30).
- **spec-040** — **slack & limit provenance**: persist the effective constraint config (CLI
  `constraint_slack` + base `constraint_defaults`) into draw/checkpoint metadata; `analytics/slack_provenance.py`
  reconstructs the slack a draw was solved at (stored provenance, else current-config fallback +
  WARNING); `DrawTester` auto-replays faithfully. Closes the "throws errors when it shouldn't" gap.
  `depends_on: none` (parallel-safe with spec-039 — no shared edited file). Two units (A persist → B
  resolver+tester), S3. Status: `ready` (Mode-A hardened 2026-05-30).
- **spec-041** — **soft-constraint outcome measurement**: per-rule 0–100 satisfaction score + raw
  domain metric recomputed from draw geometry (penalty values are never persisted), filling the
  unmeasured atoms (incl. the Maitland away-on-NRL-weekends `PreferredWeekendsAwayGround` flagship and
  soft `AwayClubHomeWeekendsCount`); per-team + per-club rollups; scored at the draw's effective slack.
  `depends_on: spec-039, spec-040`. Two units (A layer+scoring+per-team → B gap-filling measurements), S3.
  Status: `ready` (Mode-A hardened 2026-05-30).
- **spec-042** — **auto-generated analysis report (HTML + JSON + Plotly)**: assembles
  alignment + slack-provenance + hard violations + soft scores into per-team/club/rule breakdowns with
  smart dense layouts (heatmaps + collapsibles); auto-runs at end of `save_solver_output` (non-fatal) +
  `report --analysis` CLI; JSON `schema_version` is the forward contract for the eventual UI. Adds
  `plotly` (only new dep). `depends_on: spec-039, spec-040, spec-041`. Three units (A assembly/JSON →
  B HTML/Plotly/layout → C hook+CLI+docs), S3. Status: `ready` (Mode-A hardened 2026-05-30).

- **spec-043** — **pre-draw venue-capacity feasibility precheck (UI-callable)**: standalone pure
  `analyze_capacity` (config primitives in / structured dict out) + `run.py precheck` + auto-run gate
  before `generate`. New diagnostics = away-venue **per-day floor** (slots ≥ home teams) and
  **Broadmeadow per-Sunday slot-need** with a headroom-aware "can the 19:00 slot be dropped?"
  recommendation; reuses `generate_timeslots` + the `(rounds+1)//2` convention and leaves Phase-20
  (`_check_scheduling_feasibility`) and `PreSeasonReport` untouched. `depends_on: none` (branches off
  final-form). Units A→B→C (A core blocks B CLI + C generate-gate; B/C both edit `run.py` →
  serialise B before C). S2. Status: `ready` (authored 2026-05-30; Mode-A hardened).

- **spec-044** — **ClubVsClubStackedWeekends PHL Sunday FLOOR umbrella-Friday-aware**: fixes the
  real-2026 `core` infeasibility (spec-035 stage-5 handoff's OPEN blocker). Exact cause = Layer 2
  per-team-pair Sunday floor over-demands Gosford/Maitland Sunday capacity because the umbrella
  `gosford_friday_games=8` / `maitland_friday_games=2` forced Fridays are subtracted from no pair
  (`phl_forced_friday_meetings` credits only pair-named scopes). Fix = new helper
  `club_umbrella_forced_friday_meetings` + subtract it from the LOWER bound of the two spec-038
  `_range` helpers (ceiling untouched). `depends_on: spec-038` (shares `_club_vs_club_stacked_shared.py`
  + `_phl_forced_friday_helper.py`). Units A (production+helper tests) → B (integration regression). S2.
  Status: **`building`** (Mode-A hardened to `ready` 2026-05-31; user-authorised this session; Unit A
  already committed on branch `spec044-unitA`, unmerged — picking up the WIP).

```
spec-030  ──depends_on──▶  (none)                              [done — merged 5362d41]
spec-031  ──depends_on──▶  spec-030                             [done — merged into final-form 2026-05-24]
spec-032  ──depends_on──▶  spec-031                             [done — merged into final-form 2026-05-25 (A 9efee82, B 2b39008)]
spec-033  ──depends_on──▶  spec-032                             [done — merged into final-form 2026-05-25 (A 6037235, B c4af289, C b3a15c1, D f760896, E 8316911)]
spec-036  ──depends_on──▶  spec-033                             [done — merged into final-form 2026-05-25 (A f4205a9, B 312bdf0, C e378d8f)]
spec-034  ──depends_on──▶  spec-030, spec-031, spec-032, spec-033          [done — merged into final-form 2026-05-26]
spec-035  ──depends_on──▶  spec-030…033, spec-034, spec-036                [in_progress — Units A+B done; C blocked on presolve infeasibility; resumes after 037+038]
spec-037  ──depends_on──▶  (none)                                          [done — merged into final-form 2026-05-28 (Unit A a5f1685+976dcd4+abc91b9+b96a6aa; Unit B spec035-flense torn down)]
spec-038  ──depends_on──▶  (none)                                          [done — merged into final-form 2026-05-28 (5391334, 2ebe31e, 6f4c0e5); status closeout 2026-05-31]

spec-039  ──depends_on──▶  (none)                                          [ready — registry↔tester alignment gate (Mode-A hardened 2026-05-30)]
spec-040  ──depends_on──▶  (none)                                          [ready — slack/limit provenance + faithful replay]
spec-041  ──depends_on──▶  spec-039, spec-040                              [ready — soft-outcome scores + raw metrics]
spec-042  ──depends_on──▶  spec-039, spec-040, spec-041                    [ready — auto HTML+JSON+Plotly analysis report]
spec-043  ──depends_on──▶  (none)                                          [ready — pre-draw venue-capacity precheck (away floor + BM drop-slot)]
spec-044  ──depends_on──▶  spec-038 ✅                                      [building — ClubVsClub PHL Sunday floor umbrella-Friday-aware (real-config core blocker); 2026-05-31]
```

The 039→…→042 chain is the **analysis-engine plan-set** (authored 2026-05-30, not yet implemented —
awaiting user go-ahead). spec-039 and spec-040 are independent of each other and of the in-flight
spec-038 (different files), so both are startable in parallel the moment the user authorises
implementation. spec-041 gates on both 039+040; spec-042 gates on all three. The set does **not**
depend on the spec-035 e2e line and can proceed independently of it.

The 030→031→032 chain is a deliberate serialisation: all three edit `constraints/registry.py`
and `docs/system/CONSTRAINT_INVENTORY.md` (032 also touches `CLAUDE.md`), so they merge one at a
time to avoid registry/count/doc conflicts. The registry count test moves 51 → 50 (spec-030) →
49 (spec-031); spec-032 retags only (count stays 49). When each lands, mark it done, move it to
`docs/todo/done/`, drop its node, and note the next-unblocked spec. **spec-034 unblocks only when
030-033 are all `done`; spec-035 unblocks only when 030-034 are all `done` — it is the final plan,
after which the `final-form` plan line is fully drained.**

Most recently completed:

- **spec-037** — `AwayClubHomeWeekendsCount` redesigned as a two-sided derived Sunday-home range
  bound (floor = max non-PHL home games; ceiling = max all-grade home games with PHL ceil-rounded).
  **Done** (2026-05-28, Unit A merged into final-form: `a5f1685` atom+helpers, `976dcd4` tests,
  `abc91b9` docs, `b96a6aa` Mode-B-driven module-docstring fix; Unit B `spec035-flense` worktree
  and local branch torn down). Forced-Friday awareness removed from the atom (handled by
  `FORCED_GAMES` config); orphan helpers `phl_forced_friday_count` / `away_club_required_sundays` /
  `away_club_total_weekends` (+ two private helpers) deleted forward-only; `_friday_phl_forced_entries`
  kept (consumed by surviving `phl_forced_friday_meetings`). Regen-soft twin redesigned in parallel
  (single deviation IntVar). Bisect probe DoD#7 hit `FEASIBLE/OPTIMAL` for both the alone and the
  canonical pair (AwayClub + AwayClubBalance), confirming the joint feasibility that was the
  spec-035 §6 blocker is resolved on the forced-free `season_test`. 28 atom-tests + 24 spec-038
  shared-helper tests all green post-merge. /adversarial Mode B passed (1 Low docstring inconsistency
  routed inline, no Critical/High/Medium). `depends_on: none`.

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

- **spec-038** is `building` (session `2026-05-28-spec038-build`); Unit A already merged into
  final-form. Once it lands `done`, spec-035 Unit C is unblocked (spec-037 — the other half of the
  spec-035 §6 culprit pair — is now `done`).
- **spec-035** is `in_progress` but its Unit C is **BLOCKED** on the presolve
  infeasibility documented in `docs/todo/spec-035-e2e-infeasibility-handoff.md`. Resumes when
  spec-038 is `done` (spec-037 ✅ already done 2026-05-28).
- **spec-037** is **`done`** (2026-05-28 — merged into final-form; archived to `done/`).
- **spec-034** is **`done`** (2026-05-26 — merged into final-form).
- **spec-039** and **spec-040** (analysis-engine plan-set) are `review_pending` → once `/adversarial`
  Mode A hardens them to `ready` and the user authorises implementation, BOTH are startable in
  parallel (independent files, independent of spec-038). spec-041 then spec-042 follow in dependency
  order. Not yet authorised to build.
