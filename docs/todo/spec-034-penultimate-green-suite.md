<!-- status: ready -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-030, spec-031, spec-032, spec-033 -->
<!-- owner: session=none claimed=none -->

# spec-034 — PENULTIMATE: green test suite, honest coverage, three real-data assurances

> **This is one of two "special" end-of-line plans.** It is the *penultimate* gate: it runs only
> after **every other live spec is `done`** (spec-030, spec-031, spec-032, spec-033 and anything
> filed after them). The *ultimate* plan (spec-035) runs after this one. Nothing in `docs/todo/`
> may be the last thing standing except spec-035.
>
> **Forward-looking note for reviewers/executors:** this plan describes the test suite *as it will
> exist once spec-030…033 have landed* (49 atoms, the `symmetry_breakers`/`spacing` groups, the
> `PHLAnd2ndConcurrencyAtBroadmeadow` and `ClubFieldConcentration` removals). Where a claim names a
> count or symbol, **verify it against the registry at build time** — the dependency chain may have
> shifted exact numbers. The *shape* of the assurances does not change; the literals might.

**Spec source:** convenor request (this session) — "set the test suite as green with coverage met
with real data tests, no patches or mocks. The true things we need to test are: are the constraints
doing what they should on real data; can we take a draw and tell when the constraints have failed;
and measure the soft constraints to ensure they are working correctly in the draw too."

## Why

The atomization rework (specs 001→033) decomposed every scheduling rule into a registered atom and
rewired the dispatch, registry, groups, tester and exports around it. Each spec proved *its own*
unit, but no spec proves the **whole assembled suite is green on real data with honest coverage**
once they have all landed — and the per-spec churn (atom counts moving 51→50→49, group retags,
tester checks added/removed) means the suite shape is only final at the *end* of the chain. Worse,
"the tests pass" is not the same as "the tests prove the right things." This plan closes that gap by
making the suite affirmatively answer the convenor's three questions:

- **A — enforcement:** do the atoms actually constrain what they claim, on the *real 2026 model*
  (not just hand-built mini-fixtures)?
- **B — detection:** given a finished draw, can `DrawTester` correctly tell *when a constraint has
  failed* — flag the right rule when it's broken, and report clean when it isn't?
- **C — soft measurement:** are the soft constraints *measured* correctly in a draw — does
  `ViolationReport.breakdown.soft_pressure` (the `soft_pressure` dict lives on `ViolationBreakdown`,
  accessed as `report.breakdown.soft_pressure`; the rollup logic is `analytics/tester.py:195-219`
  inside `ViolationBreakdown.from_violations`) roll up each soft atom's deviation with the right
  `metric_value`, non-zero when bent and zero when honoured?
  (review fix — C1: `ViolationReport` has no direct `soft_pressure` attribute. The field is on
  `ViolationBreakdown` (line 188), accessed via the `breakdown` property on `ViolationReport` (line
  243). The cited line range 181-219 covers the `ViolationBreakdown` class, not `ViolationReport`.)

Cost of not doing this: "done" across 030-033 stays unverified at the system level; a green-looking
suite that is green because of mocks, skips, or assertions-on-own-output gives false confidence that
gets trusted at publish time. The hard rule for this plan is the `/basic` no-cheating rule:
**no mocks, no patches, no monkeypatching, no skip/xfail-to-go-green, no coverage gaming.** A
truthful "this module sits at 71%, here is the uncovered branch and why" beats a fabricated 85%.

## Definition of Done

1. **Suite green, in batches.** The full suite runs with **zero failures and zero errors**. Per the
   known ortools/Windows constraint (full `pytest tests/` segfaults — this is documented in the
   user's Claude Code global memory as `memory/reference_full_test_suite_segfaults.md`, which is
   NOT an in-repo file; there is no `memory/` directory in this repository), the suite is executed
   in documented batches;
   (review fix — M1: the citation `memory/reference_full_test_suite_segfaults.md` is a path in the
   user's global Claude Code memory store, not a file in this repo. An executor looking for it in
   the repo will get a file-not-found. Clarified in-place.)
   the batch invocation is captured as a repeatable command (DoD 7). Any `skip`/`xfail` that remains
   carries an inline justification naming *why* (e.g. platform-segfault batch boundary); **no test is
   skipped, xfailed, deleted, or weakened to make the suite green.**
2. **Coverage tooling added and honest.** `coverage`/`pytest-cov` is wired (config in `pytest.ini`
   or a `.coveragerc`/`pyproject` block — match whatever the repo lands on; `pytest.ini` currently
   has no `--cov`). Coverage is reported per run. The **floor is ≥85% on the new/changed surface
   that matters**: `constraints/atoms/`, `constraints/registry.py`, `constraints/stages.py`, and
   `analytics/tester.py`. Where 85% genuinely can't be reached honestly, the **real number plus the
   specific uncovered branches and the reason** are written into the coverage doc (DoD 7). The 85%
   is an honest floor, never a number to fake.
3. **Assurance A — atoms enforce on real data.** Every atom in `CONSTRAINT_REGISTRY` (expected 49
   post-spec-031; **verify the live count at build time**) has at least one **no-mock** test that
   builds a *real* model — the real 2026 model via `load_season_data(2026)` where tractable, else the
   real forced-free `season_test` model via `load_season_data('test')` (works at runtime via
   `f".season_{year}"` string formatting despite the `year: int` type hint in `config/__init__.py:50`
   — the hint is wrong, the duck-typed call is valid; see `config/season_test.py` docstring) — and
   asserts the atom's rule
   (review note — Low: `load_season_data` is typed `year: int` but `'test'` works via `importlib`.
   If a type-checker is run, this will flag. The executor should suppress with `# type: ignore` or
   note the discrepancy. Not a runtime bug.)
   **(i)** holds under a satisfying assignment and **(ii)** is rejected/penalised under a hand-built
   violating assignment. Each test carries a **hand-computed oracle** in a comment (the expected
   satisfying/violating count or the expected infeasibility), and asserts against that oracle, never
   against the code's own output. Atoms that already have such a test in `tests/atoms/` satisfy this
   as-is; only genuine gaps get new tests.
4. **Assurance B — tester detects a failed constraint in a draw.** For every check `DrawTester`
   performs (`analytics/tester.py`), there is a Given/When/Then test that takes a **real clean draw**,
   **deliberately corrupts exactly one game** to break that one rule, and asserts the report names
   **that** constraint with the **correct violation count**; and asserts the **same check reports
   zero** on the un-corrupted draw. Corruption is done on real draw data, no mocks.
5. **Assurance C — soft constraints are measured.** For every **soft** atom (the `soft` group, the
   always-on `symmetry_breakers` trio, the `regen_soft` analogues, and any soft analogue of
   `spacing`), a no-mock real-data test asserts `report.breakdown.soft_pressure[canonical_name]`
   rolls up the deviation with a **hand-computed `metric_value`/`total_penalty`** — non-zero (and
   equal to the hand oracle) when the soft rule is bent, and zero/absent when it is honoured.
   (review fix — C1: access path is `report.breakdown.soft_pressure`, not `report.soft_pressure`;
   `ViolationReport` exposes `breakdown` as a property that returns a `ViolationBreakdown` instance.)
6. **Static-analysis gates green.** Type-check clean on touched code; changed-file lint clean; AST
   dead-code/dark-path sweep clean over new test helpers and any touched non-test code (no unused
   fixtures, no swallowed assertions). `tests/test_no_legacy_imports.py` still green.
7. **Repeatable command + coverage doc.** A documented, copy-pasteable command (or `scripts/`
   helper) runs the batched green suite **with coverage** end-to-end on Windows without segfault, and
   the resulting honest coverage numbers (per the floor in DoD 2, including any sub-floor module with
   its reason) are recorded in the doc registry below.

## Implementation units

Decomposed so units touch disjoint files and run in parallel. The collision rule: each gap-fill unit
**adds new test files** (one new file per concern) rather than editing shared existing test files,
so two units never edit the same file. Unit E is the only cross-file unit and runs last.

### Unit A — Test harness + coverage infra + shared real-data fixtures
- **Files touched:** `pytest.ini` (or new `.coveragerc`/`pyproject` cov block — pick one, don't
  split config across two), `tests/conftest.py` (add session-scoped real-model fixtures:
  `real_2026_data = load_season_data(2026)` and `test_season_data = load_season_data('test')`, plus a
  `clean_real_draw` fixture that **loads a pre-committed test draw JSON file** (or builds one by
  running a short bounded solve on `season_test` and committing the result as
  `tests/fixtures/clean_draw_test.json`; the draw must be committed so CI is deterministic and
  does not require a solver run every test invocation), for Assurance B), a new
  (review fix — H1: "solves-or-loads" is ambiguous. `draws/2026/` is empty in the repo — no
  `current.json` exists. An executor who "solves" in a fixture will make Unit C non-deterministic
  and slow. The fixture must resolve to a committed static file or the plan must explicitly say
  to commit a bootstrap draw. Clarified: either commit `tests/fixtures/clean_draw_test.json` or
  document the bootstrap step as an explicit pre-condition before Unit A may start.)
  `scripts/run_green_suite.(ps1|py)` that runs the documented batches with coverage and merges the
  `.coverage` data files across batches (coverage `combine`).
- **Change summary:** stand up coverage measurement + batched-run harness + the shared no-mock
  real-data fixtures every other unit consumes. **No edits to any existing test's assertions.**
- **Depends on:** spec-030…033 done (so the model/registry shape is final).
- **Executor model:** Opus (the batch/segfault + coverage-combine wiring on Windows is fiddly and
  every downstream unit builds on these fixtures).
- **No-mock test outline:** *Given* the harness, *when* `scripts/run_green_suite` runs, *then* it
  exits 0, produces a combined coverage report, and never segfaults (oracle: all batches complete;
  combined `coverage report` prints a total). *Given* `real_2026_data`, *then* it is a real dict with
  the expected team/grade counts (oracle: hand-listed from `season_2026.py`), proving the fixture is
  real data, not a stub.

### Unit B — Assurance A gap-fill (atoms enforce on real data)
- **Files touched:** `tests/atoms/` — **new files only**, one per atom found to be missing a
  real-data enforce+violate test. First action: enumerate `CONSTRAINT_REGISTRY`, cross-reference
  existing `tests/atoms/test_*.py`, and produce the gap list; fill each gap with
  `tests/atoms/test_<atom>_realdata.py`.
- **Change summary:** guarantee every registered atom has a no-mock real-data test proving it
  enforces (satisfying assignment passes) and bites (violating assignment is rejected/penalised),
  with a hand-computed oracle.
- **Depends on:** Unit A (fixtures).
- **Executor model:** Sonnet per atom for mechanical atoms; **Opus** for the subtle ones
  (`PHLAnd2ndAdjacency` 2.5 h cross-venue window, `EqualMatchUpSpacing`, the ClubVsClub stacked
  co-location atoms, `VenueEarliestSlotFill`).
- **No-mock test outline:** per atom — *Given* the real model with a hand-built assignment that
  satisfies the rule, *when* the atom is applied + solved, *then* feasible / zero violations (oracle:
  hand count). *Given* a hand-built assignment that breaks the rule by exactly one game, *then*
  infeasible or exactly-one penalised (oracle: the specific game + expected count).

### Unit C — Assurance B (DrawTester detects a failed constraint)
- **Files touched:** a new `tests/test_tester_detects_failures.py` (do **not** edit the existing
  `tests/test_violation_*.py` — avoids collision and keeps the new detection matrix in one place;
  note: `tests/test_analytics_tester.py` does NOT currently exist in the repo — if it is created
  by an upstream unit before Unit C runs, do not edit it either).
  (review fix — M2: `tests/test_analytics_tester.py` is cited as "existing" but it does not exist
  at current HEAD. The collision-avoidance instruction is harmlessly vacuous for that file but is
  misleading about the repo's current state. Corrected to reflect reality.)
  First action: enumerate every check `DrawTester` runs.
- **Change summary:** one corrupt-one-game Given/When/Then per tester check proving the right rule is
  flagged with the right count when broken and zero when clean.
- **Depends on:** Unit A (`clean_real_draw` fixture).
- **Executor model:** Opus (must reason about *which* single-game mutation isolates *each* check —
  e.g. moving a game to double-book a field vs a team vs breaking the 2.5 h adjacency window).
- **No-mock test outline:** *Given* a real clean draw, *when* one game is moved to violate check X
  (and only X), *then* `run_violation_check()` reports check X with count == hand oracle and the
  other checks unchanged. *Given* the un-corrupted draw, *then* check X reports zero.

### Unit D — Assurance C (soft constraints measured in the draw)
- **Files touched:** a new `tests/test_soft_pressure_realdata.py`.
- **Change summary:** per soft atom, prove `soft_pressure[atom]` rolls up the hand-computed
  deviation (non-zero when bent, zero/absent when honoured).
- **Depends on:** Unit A (fixtures), and the soft-atom set fixed by spec-032 (`symmetry_breakers`)
  and spec-027/033 (`regen_soft`, bye-soft analogue).
- **Executor model:** Opus (the `metric_value` → `total_penalty` rollup math in
  `tester.py:195-219` must be reproduced by hand per atom).
- **No-mock test outline:** *Given* a real draw bent against soft atom S by a known amount, *then*
  `report.breakdown.soft_pressure[S]['total_penalty'] == hand_oracle` and `> 0`. *Given* a draw
  that honours S, *then* `S not in report.breakdown.soft_pressure` or `total_penalty == 0`.
  Note: `over_limit` is the correct field that counts violations (incremented per violation with
  `metric_value is not None`); `at_limit` is initialised to 0 in `ViolationBreakdown` and is
  never incremented by `from_violations` — do NOT assert on `at_limit`.
  (review fix — C1 continued: access-path fix applied here too. Also: low-severity note that
  `at_limit` is dead/zero in current rollup — do not write oracles against it.)

### Unit E — Green-up, coverage close, docs (cross-file, last)
- **Files touched:** whatever genuine failures the assembled post-033 suite surfaces (could be any
  test or, if a real bug, a small fix re-graded per `/basic`); the coverage doc; `CLAUDE.md` test
  section. This is the only unit permitted to edit pre-existing files broadly.
- **Change summary:** run the full batched suite + coverage, drive it honestly to green and to the
  ≥85% floor (or documented sub-floor), finalise docs.
- **Depends on:** Units A, B, C, D.
- **Executor model:** Opus.
- **No-mock test outline:** *Given* the whole suite via `scripts/run_green_suite`, *then* zero
  failures/errors and the combined coverage total meets the floor (or the shortfall is documented
  with the exact uncovered branches).

## Doc registry

- `CLAUDE.md` (repo root) — add/extend the "Run tests" section with the batched green-suite +
  coverage command and the Windows-segfault batching note. [Unit A defines command; Unit E finalises]
- `docs/system/TESTING.md` *(new if absent — confirmed absent at current HEAD; `docs/system/`
  exists and contains other system docs, so this file simply needs to be created there)* —
  the three assurances (enforcement / detection / soft-measurement), the no-mock policy, the
  coverage floor, and the honest coverage numbers (incl. any sub-floor module + reason).
  **Registration:** add an entry row to `docs/README.md`'s "At-a-glance file tree" under
  `system/` (this is the canonical doc-index, not `CLAUDE.md`). Optionally add a reference in
  `GOALS.md`'s doc-map paragraph if it points to specific system docs. The `CLAUDE.md` doc-index
  section lists doc *categories*, not individual files — do NOT add a per-file row there.
  (review fix — M3: the plan said "register it in the CLAUDE.md doc-index table" but the CLAUDE.md
  table lists categories (`docs/system/` as one row), not individual files. The correct index is
  `docs/README.md`'s file-tree block. `docs/system/README.md` (the per-category README) should
  also get a one-line entry.)
  [Unit E]
- `docs/todo/GOALS.md` — add the spec-034 row to the `## Specifications` table. [Unit E]
- `docs/todo/00-dependency-tree.md` — add the spec-034 node and mark it unblocked when 030-033 are
  done. [Unit E]

## Out of scope

- **The e2e solver run.** Proving the model builds, presolves, and survives 30 min of search on the
  forced-free config is **spec-035** (the ultimate plan), which depends on this one. Not duplicated
  here.
- **Changing any constraint behaviour.** This plan *tests* the atoms; it does not re-scope them. If a
  test surfaces a genuine semantic bug in an atom, that is a discovered S2/S3 → file its own spec
  (per `/basic` no-deferral), do not silently patch it into a test file.
- **Solver performance / optimality.** Not a concern of the suite.
- **Mock-based or coverage-padding tests of any kind** — explicitly forbidden by the convenor's
  "no patches or mocks" and the `/basic` no-cheating rule.

## Dependencies

- **Other plans:** `depends_on: spec-030, spec-031, spec-032, spec-033`. The suite shape (atom set,
  registry counts, group membership, tester checks) is only final once all four land — starting
  earlier means writing oracles against numbers that still move. Per `/basic`, unsatisfied
  dependency ⇒ **not startable** until each of those is `done` and merged onto the source branch.
- **Within this plan:** B, C, D depend on A (shared fixtures). E depends on A, B, C, D.

## Risks & blast radius

- **Windows/ortools full-suite segfault** — building the real 2026 model repeatedly and running the
  whole suite in one process can segfault. Mitigation: batched runs + coverage `combine`. Awareness
  item; surfaced, not insured against beyond batching.
- **Real-2026-model tests are slow/heavy** — session-scoped fixtures and `season_test` (smaller,
  forced-free) where a full-model build isn't needed keep it tractable.
- **Some atoms may be awkward to violate in isolation on the full model** — fall back to a real
  `season_test` model or a minimal real CP-SAT fixture (still no mocks) and document the choice.
- **A surfaced semantic bug** could balloon scope — handle per `/basic`: file a new spec, don't
  inline-patch.

## Open Questions

0 — none. Coverage floor is set at ≥85% per `/basic`; if the convenor later wants a different
number, that is a one-line tweak to DoD 2, not a blocker.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Autonomous: run end-to-end without waiting for the user, except where this hits `blocked`. -->
1. Status must be `ready` (carries a `reviewed:` stamp from /adversarial Mode A). If `review_pending`
   / `under_review`, let review finish — do not implement. Confirm **spec-030, 031, 032, 033 are all
   `done` and merged** onto the source branch; if not, STOP — this plan is not startable.
2. Stamp `in_progress`, claim `owner`. You are the orchestrator (Opus).
3. Dispatch Unit A first (fixtures + coverage infra). Then dispatch B, C, D in parallel (own
   worktree+branch each, one unit per agent). Run the gates for S3 per `/basic`.
4. After each unit, launch `/adversarial` Mode B to verify the diff against this plan's DoD —
   re-deriving every hand oracle and confirming **no mocks/patches/skips**. Route fixes, re-verify.
   NEVER merge an unverified unit.
5. Run Unit E last: full batched suite + coverage to green/floor, finalise docs.
6. Merge → push origin → post-merge verify (`scripts/run_green_suite` green on the merged branch) →
   remove worktrees. Tick each checkbox.
7. When all units pass: stamp `done`, move file to `docs/todo/done/`, update the dependency tree
   (drop satisfied edges; spec-035 becomes the only remaining startable plan), and continue into
   spec-035.
