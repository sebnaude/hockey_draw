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
  Two units (A registry → B run.py), S3. Status: `review_pending`.
- **spec-033** — remove `ClubVsClubAlignment` slack (dead in solver, live+divergent in tester) +
  audit soft-analogue coverage of slack-tolerant constraints; the audit's one gap —
  `BalancedByeSpacing` has no normal-mode soft analogue — is closed by adding one. `depends_on:
  spec-032` (shares `registry.py`/`run.py`/`CONSTRAINT_INVENTORY.md`/`CLAUDE.md`). Two units
  (A removal+verify → B bye-soft), S3. Status: `review_pending`.

Two **special end-of-line plans** authored **2026-05-24** (this session). They are gated behind
*everything else* and run last, in order:

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
spec-030  ──depends_on──▶  (none)                              [review_pending]
spec-031  ──depends_on──▶  spec-030                             [review_pending]   (spec-029 dep satisfied — done)
spec-032  ──depends_on──▶  spec-031                             [review_pending]
spec-033  ──depends_on──▶  spec-032                             [review_pending]
spec-034  ──depends_on──▶  spec-030, spec-031, spec-032, spec-033          [review_pending]   (PENULTIMATE)
spec-035  ──depends_on──▶  spec-030…033, spec-034                          [review_pending]   (ULTIMATE — last)
```

The 030→031→032 chain is a deliberate serialisation: all three edit `constraints/registry.py`
and `docs/system/CONSTRAINT_INVENTORY.md` (032 also touches `CLAUDE.md`), so they merge one at a
time to avoid registry/count/doc conflicts. The registry count test moves 51 → 50 (spec-030) →
49 (spec-031); spec-032 retags only (count stays 49). When each lands, mark it done, move it to
`docs/todo/done/`, drop its node, and note the next-unblocked spec. **spec-034 unblocks only when
030-033 are all `done`; spec-035 unblocks only when 030-034 are all `done` — it is the final plan,
after which the `final-form` plan line is fully drained.**

Most recently completed:

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

- **spec-030** (once `/adversarial` Mode A stamps it `ready`; `depends_on: none`).
- spec-031 unblocks when spec-030 is `done`; spec-032 unblocks when spec-031 is `done`. The chain
  is serial, so at most one of the three is startable at a time.
- **spec-034 and spec-035 are NOT startable yet** — they are the deliberately end-of-line special
  plans. spec-034 (penultimate) unblocks only when 030-033 are all `done`; spec-035 (ultimate)
  unblocks only when 030-034 are all `done`. They run last, in that order.
