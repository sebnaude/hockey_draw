<!-- status: in_progress -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-030, spec-031, spec-032, spec-033, spec-034, spec-036 -->
<!-- owner: session=opus-spec035-20260526T000000Z claimed=2026-05-26T00:00:00Z -->
<!-- 2026-05-29 STATUS: Unit C still BLOCKED. After spec-037+038 landed, core−ClubGameSpread is STILL
     infeasible on forced-free season_test, with a NEW signature (exactly_one: empty or all false).
     Re-bisected to a minimal HARD pair: ClubVsClubStackedWeekends × ClubDayParticipation (both club-day
     dates are Sundays; neither atom is slackable). See spec-035-e2e-infeasibility-handoff.md §11. This is
     a constraint-semantic conflict (DoD-6 OUT of scope) owned by spec-038 (building). A follow-on
     spacing-family phase (Units D/E: add bye_spacing, then swap to spacing) was appended per convenor
     request 2026-05-29 — it is `drafting` and blocked behind Unit C liveness + Open Question FP. -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline. 2026-05-25: added intermediate ClubGameSpread-excluded run + cross-run symmetry comparison (Unit A gains an exclude param, Unit C runs both solves) — incremental scope, no re-review. 2026-05-25 (convenor): RUN ORDER SWAPPED — the `--exclude ClubGameSpread` run is now Run 1, the full core run is Run 2. Baseline framing unchanged (full core remains the recorded reference); only execution order moved. No re-review (operational ordering only). -->

# spec-035 — raw `--core` e2e solve on the forced-free test config + remaining-symmetry readout

> **This is an end-of-line e2e validation plan.** It runs only after **every other
> live spec is `done`, including spec-034.** It is a real end-to-end solver run, not
> a unit test: it proves the assembled, post-atomization model actually builds, gets through CP-SAT
> presolve, and survives a real search on the real 2026 teams/fields with nothing forced and week 1
> *not* fixed.
>
> **Forward-looking note for reviewers/executors:** this plan assumes spec-032 has shipped (so
> `--groups core` exists and `symmetry_breakers` is always-on) and spec-034 has the suite green.
> Verify those at build time.

**Spec source:** convenor request (this session) — "perform a test run of the `--core` ones against
the test set up (base fields availability and teams for 2026, no forced games). The goal is to ensure
the model gets through presolve and at least 30 minutes into the solution. Another goal is to
[measure] the symmetry of the model … we are NOT going to fix the first week, it's a raw run … Once
the model gets to 30 minutes the bash should be killed regardless of if a solution was found … only
use `--workers 10` … a true e2e test." Clarified this session: **no historical symmetry baseline
exists** (confirmed by filesystem + checkpoint scan), so the comparison-to-earlier-runs goal is
dropped — we instead **measure and report how much symmetry the model has left, absolutely.**

## Why

Every other spec validates pieces; nothing validates that the *whole* atomized model, selected as the
production `core` group (plus the always-on `symmetry_breakers`), with **no forced games** and
**week 1 unfixed**, will (a) construct, (b) get through CP-SAT presolve without blowing up, and
(c) actually enter and survive a real search. Presolve blowups, helper-var explosions, and
model-construction regressions are exactly the failures unit tests miss and that only a real solve
surfaces. This is the final acceptance gate for the `final-form` line.

Two concrete goals:

1. **Liveness:** the raw core build gets *through presolve* and at least *30 minutes into search*
   on `--workers 10`. We do **not** need a feasible draw — we need proof the model is solvable-shaped
   and doesn't die in presolve or immediately go infeasible.
2. **Remaining symmetry:** quantify how much symmetry the model still has after all the
   symmetry-breaking work (spec-002 lex ordering, spec-003/016 NIHC fill order, etc.). The convenor
   expected a stored baseline to compare against; **there is none** (filesystem scan found no
   `[Symmetry]` data in any log, checkpoint metadata, or model-init artifact — the closest hit is an
   unrelated `scripts/_search_results.txt`). So the deliverable is the **current** model's symmetry
   readout, recorded as the *first* baseline for any future comparison — not a delta.

3. **Cross-run symmetry comparison (intermediate phase — convenor request 2026-05-25):** run the raw
   `--core` solve in two variants — **Run 1 with the entire `ClubGameSpread` method excluded**
   (`--exclude ClubGameSpread`) and **Run 2 the full core** (no exclude) — capture each run's
   `[Symmetry]` stats, and **diff the two readouts**.
   The hypothesis being tested: whether the `ClubGameSpread` atoms strip exploitable symmetry from the
   model (i.e. whether their presence changes the generator/orbit counts CP-SAT's presolve detects).
   Because the full-core run (Goal 2) and this excluded run differ by exactly the ClubGameSpread method
   and nothing else, any change in the symmetry readout is attributable to those atoms. This gives us a
   *within-session* comparison ("across the runs") even though no historical baseline exists.
   **Scope of the exclusion:** the `ClubGameSpread` canonical name covers **both** atoms of the method —
   `UnifiedConstraintEngine._club_game_spread_hard()` (`unified.py:382`, guarded by
   `'ClubGameSpread' not in _skip`) **and** `_club_game_spread_soft()` (`unified.py:782`, same guard).
   A single `--exclude ClubGameSpread` skips both (verified: `--exclude` → `_resolve_group_selection`
   → `skip_constraints` → both unified-engine guards). **Do NOT also exclude `ClubNoConcurrentSlot`:**
   although it was historically *extracted from* ClubGameSpread's lower no-double-up bound (spec-021),
   it is now a **separate `core_hard` canonical constraint** (`registry.py:397`,
   `groups={core, critical_feasibility, core_hard}`) and is **not** part of "the ClubGameSpread method."
   Dropping it would relax physical slot-concurrency feasibility and confound the comparison — both runs
   must keep it. The only delta between the two runs is `ClubGameSpread` (hard + soft).

A subtlety this plan must fix first: **CP-SAT's presolve symmetry stats are not currently captured.**
The solver sets `log_search_progress=True` (`main_staged.py:1294`, `solver_diagnostics.py:288`), but
the saved `logs/*.log` files contain only the Python-side `solver` logger output (MONITOR / MODEL
STATISTICS / INFO lines) — CP-SAT's own stdout, where the `[Symmetry]` section lives, is **not
redirected into the log**. So Goal 2 is impossible until that stdout is captured (Unit B).

A second wiring fact: the forced-free test config **already exists** — `config/season_test.py`
(`forced_games = []`, built via `load_season_data('test')`). But its own docstring
(`season_test.py:25`) warns that `run.py --year` is **int-typed and cannot select it** as
`--year test`. So the run needs a small launcher (or a `--config`/string-year path), not a bare
`run.py generate --year test` (Unit A).

## Definition of Done

1. **Forced-free test-config launch path.** There is a repeatable way to launch a real solve against
   `config/season_test.py` (the existing forced-free config — **DoD asserts `forced_games == []`** and
   that it carries the **2026 base teams and field availability**, not some reduced toy set;
   reconcile with `season_2026.py` at build time and fix `season_test.py` if it has drifted). Chosen
   mechanism: a `scripts/run_core_e2e.(py|ps1)` launcher that loads `load_season_data('test')`
   programmatically (matching the documented usage at `season_test.py:22`) and drives the **same**
   solve entry point `run.py`/`main_staged` uses — *not* a bare `--year test` (the int-typed `--year`
   can't select it). The launcher fixes the solve flags below in one place.
   **Type-annotation note:** `config/__init__.py::load_season_data` is typed `year: int`; calling it
   with `'test'` works at runtime (Python ignores annotations; the f-string `f".season_{year}"` resolves
   to `.season_test`) but will be flagged by a strict type checker. Unit A must either: (a) widen the
   annotation to `Union[int, str]` in `config/__init__.py`, or (b) call
   `from config.season_test import get_season_data; data = get_season_data()` directly, bypassing
   `load_season_data` entirely. Option (b) is simpler and avoids a cross-cutting signature change. Either
   is acceptable; the executor must choose and ensure DoD-8 type-check passes. (review fix — C2: type
   annotation mismatch not previously called out.)
2. **Raw core flag profile.** The run uses exactly: the **`core` group** (`--groups core`, which per
   spec-032 also applies the always-on `symmetry_breakers` trio unless suppressed — they are **not**
   suppressed here), **`--workers 10`** (and only 10), **no `--fix-round-1`** (week 1 is *not*
   fixed — `run.py:111` is the actual `--fix-round-1` flag, destination `fix_round_1`; confirmed by
   code review — there is NO `--round1-symmetry` flag; the "fix the first week" lever is
   `--fix-round-1`) (review fix — C1: wrong flag name corrected), **no `--locked` /
   `--lock-weeks`** (raw), and **no forced games** (inherent to `season_test`). The launcher records
   the exact resolved flag set into the run metadata/log so the profile is auditable. The launcher
   takes an **optional exclude list** so the identical profile can be re-run with
   `--exclude ClubGameSpread` for the intermediate phase (DoD-2b) — exclusion is the *only* permitted
   delta between the two runs.
2b. **Intermediate ClubGameSpread-excluded profile.** A second profile, identical to DoD-2 in every
   respect **except** `--exclude ClubGameSpread` (which skips both `_club_game_spread_hard` and
   `_club_game_spread_soft` — DoD asserts both engine guards are skipped and that `ClubNoConcurrentSlot`
   is **still applied**). `--workers 10`, `--groups core`, `symmetry_breakers` on, no `--fix-round-1`,
   no locks, no forced games — same as DoD-2. The launcher's recorded flag set proves the two runs
   differ only by the exclude.
3. **CP-SAT presolve output captured.** The run's log file contains CP-SAT's own presolve section,
   including the `[Symmetry]` lines (generators / orbits / variable+constraint reduction), by
   redirecting the CP-SAT solver stdout/log stream into the run log (it currently is not — only the
   Python `solver` logger is). Observable: `grep -i symmetry <logfile>` returns the presolve symmetry
   block.
4. **Liveness reached (both runs).** Each run **gets through presolve** (model built; presolve
   completes; search starts — evidenced by a CP-SAT `#1`/`best:`/objective progress line or an
   explicit search-start marker in the captured log) **and survives ≥30 minutes of search**. Each
   process is **killed at the 30-minute mark regardless of whether a feasible solution was found**
   (wall-clock; clean kill of the background ortools process on Windows). Success criterion per run =
   *reached search* **and** *30 min elapsed without crash or presolve-time infeasibility*. (Finding a
   solution is a bonus, not required; not finding one is not a failure.) The ClubGameSpread-excluded
   run (DoD-2b, **Run 1**) and the full-core run (DoD-2, **Run 2**) are each held to this bar. They run
   **sequentially, not concurrently** — two simultaneous `--workers 10` ortools solves would contend
   for memory and confound the liveness/OOM signal; run the excluded one (Run 1), kill at 30 min, then
   run the full core (Run 2).
5. **Cross-run remaining-symmetry readout + comparison.** A single readout artifact (e.g.
   `scripts/e2e_symmetry_readout.md`) parses the captured `[Symmetry]` stats from **both** runs and
   reports, side by side: number of generators, orbit count/sizes, and the var/constraint reduction
   presolve reports for (a) the full `core` run and (b) the `core` − `ClubGameSpread` run. It states a
   **conclusion on Goal 3**: whether excluding the ClubGameSpread atoms changed the model's exploitable
   symmetry (more generators/larger orbits when excluded ⇒ ClubGameSpread was suppressing symmetry; no
   change ⇒ it is symmetry-neutral). It explicitly records: "**no historical baseline exists**
   (filesystem + checkpoint scan, spec-035 build) — the full-core numbers are recorded as the first
   baseline, and the excluded run is compared against it within this session." If a historical baseline
   *is* discovered during the build-time re-scan, the readout diffs against it too.
6. **Debugging to the liveness state.** If the model fails to reach search (presolve blowup, OOM at
   `--workers 10`, immediate infeasibility), it is **debugged to the DoD-4 state on the assumption
   that the core constraints are correct** — i.e. fixes target model construction / presolve /
   resourcing, **not** constraint semantics. Whatever was changed to get there is documented in the
   readout. If the obstruction turns out to be a genuine *constraint-semantic* bug (core constraint
   actually wrong), that is **out of scope** → file a new spec per `/basic`, do not patch semantics
   here.
7. **Runs executed in background, killed at 30 min, artifacts saved.** **Both** solves
   (core−ClubGameSpread first, then full core) run via Bash `run_in_background`, each capped/killed at
   30 minutes (Monitor or a timeout wrapper), **sequentially** (DoD-4). `--workers 10` only. Each run gets its own
   captured log; both logs and the single comparison readout are saved under `logs/`/`scripts/` and
   referenced from the readout.
8. **Gates:** launcher + log-capture code type-checks clean; changed-file lint clean; AST sweep clean
   (no dead launcher branches, presolve-fail path logged at INFO via the project logger, never
   silently swallowed); spec-034's suite stays green after the Unit A/B code changes.

## Implementation units

Three units. A and B touch disjoint code (A: a new launcher script — now with an optional exclude
param — + possibly a `--config`/string path in `run.py`'s arg layer; B: the CP-SAT stdout/log capture
in `solver_diagnostics.py` / `main_staged.py`) and can run in parallel. C is the actual **two** runs
(the intermediate `core`−`ClubGameSpread` run first, then full core) + the comparison readout +
debugging, and depends on both.

### Unit A — Forced-free test-config launcher + raw core flag profile
- **Files touched:** new `scripts/run_core_e2e.(py|ps1)`; verification of `config/season_test.py`
  (and a fix there only if it has drifted from 2026 base teams/fields or isn't `forced_games == []`);
  *optionally* a string-accepting `--config`/year path in `run.py` if that is cleaner than a pure
  script (executor's call — prefer the script to avoid widening CLI surface, since `season_test`'s
  own docs already prescribe the programmatic path).
- **Change summary:** one-command launch of the raw core solve on the forced-free config with the
  exact DoD-2 flag profile, flags recorded to metadata. The launcher accepts an **optional exclude
  list** (default empty) threaded straight to the solve entry point's `--exclude`, so the same command
  produces both the DoD-2 (no exclude) and DoD-2b (`--exclude ClubGameSpread`) runs with no other
  difference.
- **Depends on:** spec-032 done (the `core` group + always-on `symmetry_breakers` must exist); spec-034
  done (don't e2e a red suite).
- **Executor model:** Opus (must thread the exact flag profile through the real solve entry point and
  guarantee `--fix-round-1` is absent / `fix_round_1=False` and locks are *off*; the flag name is
  `--fix-round-1`, NOT `--round1-symmetry` — review fix C1).
- **No-mock test outline:** *Given* data loaded via the chosen path (`get_season_data()` or
  `load_season_data('test')`), *then* `forced_games == []` and team set == the 2026 base team set
  (oracle: hand-listed from `season_2026.py`). *Given* the launcher's resolved config with no exclude,
  *then* the recorded flag set contains `groups=['core']`, `workers==10`, `fix_round_1` falsy (NOT
  `round1_symmetry` — the actual solver attribute is `fix_round_1`; review fix C1), no locked weeks,
  `exclude == []` (oracle: the DoD-2 profile, asserted field-by-field). *Given* the launcher invoked
  with `exclude=['ClubGameSpread']`, *then* the recorded flag set is identical except
  `exclude == ['ClubGameSpread']`, and the constructed engine's `skip_constraints` contains
  `'ClubGameSpread'` while NOT containing `'ClubNoConcurrentSlot'` (oracle: the DoD-2b profile — the
  exclude is the only delta and ClubNoConcurrentSlot stays in).

### Unit B — Capture CP-SAT presolve stdout (incl. `[Symmetry]`) into the run log
- **Files touched:** `solver_diagnostics.py` and/or `main_staged.py` — redirect/duplicate the CP-SAT
  solver log stream (e.g. via the OR-Tools log callback / `solver.log_callback`, or capturing the
  stdout the C++ layer writes under `log_search_progress=True`) into the run's log file alongside the
  existing Python `solver` logger; ensure the presolve `[Symmetry]` block lands in the file. A small
  `parse_symmetry_stats(logpath)` helper (generators/orbits/reduction) for Unit C's readout.
- **Change summary:** the run log gains CP-SAT's presolve output so symmetry is measurable; a parser
  exposes the numbers.
- **Depends on:** none (independent of Unit A).
- **Executor model:** Opus (OR-Tools log redirection on Windows can interact with the existing
  solution-callback logging; getting the C++ stdout into the file without breaking the monitor is the
  subtle part).
- **No-mock test outline:** *Given* a tiny real CP-SAT model solved with the capture wired and
  `log_search_progress=True`, *then* the log file contains the presolve section and
  `parse_symmetry_stats` returns a dict with integer `num_generators`/`num_orbits` (oracle: a known
  small symmetric model whose generator count is hand-derivable, e.g. an all-interchangeable-variable
  toy where the expected generator count is computable). *Given* a log with no symmetry block, *then*
  the parser returns an explicit "not present" sentinel, logged at INFO — never a silent `None`.

### Unit C — The two raw e2e runs + 30-min kills + cross-run symmetry comparison readout + debugging
- **Files touched:** new `scripts/e2e_symmetry_readout.md` (the artifact), plus any
  model-construction/presolve/resourcing fix that DoD-6 turns out to require (re-graded per `/basic`
  if it grows).
- **Change summary:** execute the launcher in the background at `--workers 10` **twice, sequentially**
  — (1) `core` with `--exclude ClubGameSpread`, (2) full `core` — killing each at 30 min, parse the
  `[Symmetry]` stats from both captured logs, write the side-by-side comparison readout with the Goal-3
  conclusion, re-scan for any historical baseline, and debug to the liveness state if presolve doesn't
  clear (construction/resourcing only). Run 2 only starts after Run 1's process is confirmed killed (no
  concurrent solves — DoD-4).
- **Depends on:** Unit A (launcher, incl. the exclude param) and Unit B (symmetry capture + parser).
- **Executor model:** Opus.
- **No-mock test outline:** this unit's "test" is the two e2e runs themselves (real solves, no mocks):
  *Given* each launcher run in background with `--workers 10`, *when* 30 minutes elapse, *then* that
  run's captured log shows presolve completed + at least one search/progress line before the kill
  (oracle: presence of a CP-SAT search-start/`#1` marker), and the kill is clean (no orphaned ortools
  process). *Given* both captured logs, *then* `parse_symmetry_stats` yields numbers for each and the
  readout records the side-by-side comparison + the Goal-3 conclusion + "no baseline / first baseline."
  *Given* the two runs' resolved flag sets, *then* they are byte-identical except Run 1 carries
  `exclude=['ClubGameSpread']` (oracle: the only-delta invariant — guards against an accidental second
  difference that would confound the comparison).

## Doc registry

- `docs/system/SOLVER_E2E.md` *(new — register per the canonical convention: a one-line entry in
  `docs/README.md`'s master file-tree under `docs/system/` AND in the per-category
  `docs/system/README.md`. Both are mandatory/exhaustive. A row in the CLAUDE.md "Additional
  Documentation" per-file table is OPTIONAL — that table is curated, not exhaustive, and niche docs
  like REGEN_CONSTRAINTS are intentionally absent; skip it for this operational e2e doc.)* — how to
  run the raw core e2e on the forced-free test config, the 30-min-kill protocol, `--workers 10`, the
  symmetry-capture mechanism, and where the readout lands. [Unit C, with the capture mechanism from
  Unit B] (review fix — M1: `docs/README.md` added. Global cross-plan review 2026-05-24: aligned to
  spec-034's convention — `docs/README.md` + `docs/system/README.md` mandatory, CLAUDE.md per-file
  row optional — resolving the prior 034/035 contradiction over whether a CLAUDE.md row is required.)
- `CLAUDE.md` (repo root) — (a) a "Quick Commands" entry for the e2e core run via
  `scripts/run_core_e2e`; (b) a note that CP-SAT presolve symmetry is now captured to the log.
  (c) DROPPED: a per-file row in the doc-index table is no longer required — see the convention note
  above; the canonical index is `docs/README.md` + `docs/system/README.md`. [Unit C]
- `config/season_test.py` — docstring touch *only if* Unit A changes how it's launched (e.g. to
  reference `get_season_data()` as the preferred path after the type-annotation fix). [Unit A]
- `docs/todo/GOALS.md` — **spec-035 row already present** (line 171 as of plan authoring); update
  status column from `review_pending` → `done` when work lands. Do NOT add a duplicate row.
  (review fix — H1: "add the spec-035 row" was wrong — the row already exists.)
- `docs/todo/00-dependency-tree.md` — **spec-035 node already present** (lines 43-48 and 57);
  update its status from `review_pending` → `done` when work lands; drop its node from the "Live
  specs" list and add it to "Most recently completed". Do NOT add a duplicate entry. (review fix —
  H2: "add the spec-035 node" was wrong — the node already exists.)

## Out of scope

- **Producing a publishable draw.** The run is killed at 30 minutes; optimality/feasibility-to-publish
  is explicitly *not* a goal.
- **Comparing to an earlier main-branch / historical run.** Dropped: no historical baseline exists
  (confirmed). We record the first baseline instead. NOTE: this is distinct from the **within-session**
  comparison added 2026-05-25 (Goal 3 / DoD-5) — comparing the full-core run against the
  `core`−`ClubGameSpread` run *is* in scope; only comparison against a *prior, stored* run is not.
- **Tuning the solver** (workers beyond 10, parameter sweeps, warm-start hints) — `--workers 10` only,
  raw.
- **Fixing constraint *semantics*.** If a core constraint is genuinely wrong, that is a new spec, not
  this run's job (this run assumes the core constraints are correct and debugs only construction /
  presolve / resourcing).
- **The suite-green + coverage work** — that is spec-034 (the prerequisite green-suite plan).

## Dependencies

- **Other plans:** `depends_on: spec-030, spec-031, spec-032, spec-033, spec-034, spec-036`. spec-032
  supplies the `core` group and always-on `symmetry_breakers`; spec-034 supplies a green suite (don't
  e2e a red codebase); **spec-036 supplies the correct no-flag single-solve over the full set** (this
  e2e is a raw single-solve, so it must run on the fixed default path, not the old incomplete
  `--simple`). Per `/basic`, **not startable** until all six are `done` and merged.
- **Within this plan:** Unit C depends on Units A and B. A and B are mutually independent (disjoint
  files) and run in parallel.

## Risks & blast radius

- **Presolve memory blowup at `--workers 10`** — historical logs show the real ~83 k-var model at
  ~80% memory with many workers; `--workers 10` is the cap precisely to contain this. If it still
  OOMs in presolve, that is DoD-6 debugging territory (resourcing, not semantics). Awareness item.
- **CP-SAT stdout capture interfering with the solution callback / monitor** — Unit B touches live
  solver logging; the spec-034 suite must stay green after it. Verify post-merge.
- **Clean process kill on Windows** — a background ortools solve must be terminated without orphans
  at the 30-min mark; the launcher/kill wrapper must handle the worker subprocesses. On Windows,
  `taskkill /F /T /PID <pid>` kills the process tree (required — `kill <pid>` alone leaves ortools
  worker subprocesses running as orphans). The launcher's kill wrapper must use the `/T` (tree) flag
  or equivalent (`subprocess.Popen.kill()` on a process group). (review note — Low L2: execution
  protocol step 5 says "kill at 30 minutes (Monitor/timeout)" without specifying the Windows
  mechanism; executor must use process-tree kill.)
- **`season_test` drift** — if it isn't actually the 2026 base teams/fields or isn't forced-free, the
  run wouldn't be the intended test; Unit A asserts and fixes this before the run.

## Open Questions

0 — none. The convenor answered this session: no baseline exists → report current absolute symmetry;
use the existing `config/season_test.py`; the filesystem scan for a baseline has been done (none
found). The launch mechanism (script vs CLI flag) is an executor implementation choice, defaulting to
the script per `season_test.py`'s own documented programmatic usage.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Autonomous: run end-to-end without waiting for the user, except where this hits `blocked`. -->
1. Status must be `ready` (carries a `reviewed:` stamp). Confirm **spec-030, 031, 032, 033 AND
   spec-034 are all `done` and merged** — this is the genuinely last plan; if any is open, STOP.
2. Stamp `in_progress`, claim `owner`. Orchestrator = Opus.
3. Dispatch Units A and B in parallel (own worktree+branch each). Run S3 gates per `/basic`. Verify
   each with `/adversarial` Mode B before merge (re-derive the toy symmetry oracle; confirm the raw
   flag profile has `--fix-round-1` absent / `fix_round_1=False` and no locks — the flag is
   `--fix-round-1`, NOT `--round1-symmetry`; review fix C1). NEVER merge unverified.
4. Merge A and B, push, post-merge-verify the spec-034 suite is still green.
5. Run Unit C **as two sequential runs**: (5a) launch `scripts/run_core_e2e` with
   `--exclude ClubGameSpread` via Bash `run_in_background` at `--workers 10`; **kill at 30 minutes**
   (Monitor/timeout) regardless of solution state; confirm the process tree is dead. (5b) launch the
   same launcher with **no exclude** (full core — the only delta), again `--workers 10`,
   **kill at 30 minutes**. Do not run 5a and 5b concurrently (memory contention confounds liveness —
   DoD-4). Then parse both captured
   logs, write the side-by-side comparison readout with the Goal-3 conclusion (does ClubGameSpread
   strip symmetry?), and debug to the DoD-4 liveness state if presolve doesn't clear
   (construction/resourcing only). If a constraint-semantic bug is the blocker, file a new spec and
   surface to the user.
6. Tick checkboxes, stamp the plan `done`, move to `docs/todo/done/`, update the dependency tree —
   the `final-form` plan line is now fully drained.

---

## FOLLOW-ON PHASE — spacing-family e2e validation (convenor request 2026-05-29)
<!-- status: drafting — NOT yet adversarially reviewed; depends on Unit C liveness (core reaching search) -->
<!-- 2026-05-29 finding: core−ClubGameSpread is STILL infeasible at HEAD (2ebe31e) — minimal pair
     ClubVsClubStackedWeekends × ClubDayParticipation (see spec-035-e2e-infeasibility-handoff.md §11).
     Units C, D, E are all BLOCKED until that hard-vs-hard conflict is resolved (a spec-038 / new-spec
     decision — see Open Questions FP). This section is recorded per the user's explicit instruction to
     "add to the plan"; it is `drafting` and must pass /adversarial Mode A before implementation. -->

> **Convenor instruction (verbatim intent):** once the core run reaches the DoD-4 liveness bar
> (Run 1 = core−ClubGameSpread surviving a 30-min run, Run 2 = full core), **then** extend the same
> raw e2e harness to the two spacing-family groups that spec-032/033 peeled out of `core`:
> first **add `bye_spacing`** (`BalancedByeSpacing`), then **remove `bye_spacing` and add `spacing`**
> (`EqualMatchUpSpacing`) — and **get both debugged and working** (each reaching the same liveness bar).
>
> Group/atom mapping verified against `constraints/registry.py` (2026-05-29):
> `bye_spacing` group ⇒ the single atom `BalancedByeSpacing`; `spacing` group ⇒ the single atom
> `EqualMatchUpSpacing`. Both were peeled out of `core` (spec-032 → `spacing`; spec-033 Unit B →
> `bye_spacing`), so the full set is `--groups core,<group>` (symmetry_breakers stay on; ClubGameSpread
> handling unchanged from Run 2).

### Additional Definition of Done (DoD-9 … DoD-12)

9. **Launcher generalised to add a group.** `scripts/run_core_e2e.py` accepts the run's **group set**
   (default `['core']`, the existing behaviour) so the identical raw profile (`--workers 10`,
   `fix_round_1=False`, no locks, forced-free, symmetry_breakers on) can be launched as
   `core`, `core,bye_spacing`, or `core,spacing`. The recorded profile sidecar shows the group set, and
   the group set is the **only** delta between the spacing-family runs and Run 2. (No change to the
   exclude param.)
10. **Run 3 — `core,bye_spacing` reaches liveness.** A raw e2e run with `--groups core,bye_spacing`
    (symmetry_breakers on, ClubGameSpread present, same as Run 2 otherwise) **gets through presolve and
    survives ≥30 min of search**, killed at 30 min (process-tree kill). `BalancedByeSpacing` is confirmed
    APPLIED in the resolved constraint set (assert it is present, not silently dropped). If it fails to
    reach search, it is **debugged to working** — construction/resourcing per DoD-6; a genuine
    `BalancedByeSpacing` *semantic* conflict is a new spec, not patched here.
11. **Run 4 — `core,spacing` (bye_spacing removed) reaches liveness.** A raw e2e run with
    `--groups core,spacing` and **`bye_spacing` NOT selected** (assert `BalancedByeSpacing` is ABSENT and
    `EqualMatchUpSpacing` is PRESENT in the resolved set) **gets through presolve and survives ≥30 min of
    search**, killed at 30 min. Debugged to working on the same DoD-6 terms. Run 3 and Run 4 are
    **sequential, not concurrent** (memory — same rule as DoD-4), and run AFTER Runs 1+2.
12. **Spacing-family readout.** `scripts/e2e_symmetry_readout.md` (or a sibling) is extended with the
    Run 3 + Run 4 liveness outcomes and their captured `[Symmetry]` stats, side by side with Run 2
    (full core), so the marginal effect of `bye_spacing` and of `spacing` on the model's remaining
    symmetry is recorded. Any debugging done to reach liveness for either run is documented there.

### Additional Implementation units

#### Unit D — Generalise the launcher to add a group + run `core,bye_spacing` to liveness
- **Files touched:** `scripts/run_core_e2e.py` (add a `groups` param, default `['core']`, threaded into
  `build_run_config`/`_resolve_group_selection`; preserve the existing single-arg behaviour and tests);
  `scripts/e2e_symmetry_readout.md` (append Run 3). Plus any construction/resourcing fix DoD-10 needs.
- **Change summary:** launcher learns `--groups`; execute the `core,bye_spacing` raw run in the
  background at `--workers 10`, kill at 30 min, capture symmetry, debug to liveness.
- **Depends on:** Unit C reaching liveness (i.e. the §11 blocker resolved) + Unit A/B merged.
- **Executor model:** Opus.
- **No-mock test outline:** *Given* `build_run_config(groups=['core','bye_spacing'])`, *then* the
  resolved set CONTAINS `BalancedByeSpacing` and the profile records `groups==['core','bye_spacing']`,
  everything else byte-identical to Run 2 (oracle: only-delta invariant). *Given* the background run,
  *when* 30 min elapse, *then* the captured log shows presolve cleared + a search/`#1` marker
  (oracle: liveness).

#### Unit E — Run `core,spacing` (bye_spacing removed) to liveness
- **Files touched:** `scripts/e2e_symmetry_readout.md` (append Run 4); any construction/resourcing fix
  DoD-11 needs. (No further launcher change — Unit D already generalised it.)
- **Change summary:** execute `--groups core,spacing` raw run in the background at `--workers 10` AFTER
  Run 3, kill at 30 min, capture symmetry, debug to liveness.
- **Depends on:** Unit D (launcher generalisation) + Unit C liveness.
- **Executor model:** Opus.
- **No-mock test outline:** *Given* `build_run_config(groups=['core','spacing'])`, *then* the resolved
  set CONTAINS `EqualMatchUpSpacing` and does NOT contain `BalancedByeSpacing` (oracle: the swap is
  exact — bye_spacing out, spacing in). *Given* the background run, *when* 30 min elapse, *then* the log
  shows presolve cleared + a search marker (oracle: liveness).

### Additional Open Questions (block the whole follow-on phase AND Unit C)

FP. **How should the `ClubVsClubStackedWeekends` × `ClubDayParticipation` hard conflict (handoff §11) be
   resolved?** It blocks Unit C (core liveness) and therefore Units D/E. This is a constraint-semantic
   decision, NOT spec-035 construction work — options the convenor must choose between are captured in
   the session report (e.g. fix/loosen the spec-038 stacking model; exempt club-day Sundays from
   stacking; make one atom soft; or run the e2e with `ClubVsClubStackedWeekends` excluded as a
   documented partial-core liveness+symmetry run). Until answered, this whole phase stays `drafting`.

### Execution protocol addendum (after step 6, only once DoD-9…12 are ready & the §11 blocker is resolved)
7. Implement Unit D: generalise the launcher (`--groups`), re-run its Unit-A tests green, then run
   `core,bye_spacing` in the background at `--workers 10`, kill at 30 min (process-tree kill), capture
   symmetry, debug to liveness (construction/resourcing only — semantic conflict ⇒ new spec).
8. Implement Unit E: run `core,spacing` (bye_spacing out, spacing in) sequentially AFTER Run 3, same
   kill + capture + debug.
9. Extend `scripts/e2e_symmetry_readout.md` with Runs 3 + 4 (liveness + symmetry, side by side with
   Run 2). Tick the new checkboxes only when both runs hit the liveness bar.

---

## EXECUTION OUTCOME (2026-05-30) — Unit C + spacing follow-on + full sweep

**Unblocked:** the §11 blocker (`ClubVsClubStackedWeekends × ClubDayParticipation`) was cleared by
spec-038 `6f4c0e5`; a bisect probe then reached search, so Unit C + Units D/E + the appended full
sweep were all executed this session. Open Question FP is therefore **resolved** (by the spec-038 fix).

**Convenor decisions (2026-05-30):** after Run 1 OOM'd at `--workers 10` on a RAM-constrained box
(15.6 GB total, ~4.5 GB free; the 110k-var solve grows past that), the convenor (a) **accepted Run 1
as a liveness success** (it reached search), and (b) authorised **`--workers 8`** for all remaining
runs (a DoD-6 resourcing override — worker count does not affect the model or its `[Symmetry]`
readout). The launcher gained `--workers` and `--groups`; Unit C's 30-min kill is `scripts/run_e2e_30min.py`.

**Full results** (forced-free `season_test`, raw, week-1 unfixed; converged-pass symmetry):

| Run | Profile | Workers | Liveness (30-min) | Converged symmetry | Feasible soln |
|---|---|---|---|---|---|
| 1 | core − ClubGameSpread | 10 | reached search; OOM ~6.8 min (convenor: ✅) | 5 gen / 40,594 vars | none |
| 2 | full core | 8 | ✅ survived (1892s) | 1 gen / 48 vars | none (`best:-inf`) |
| 3 | core,bye_spacing | 8 | ✅ survived (1802s) | 1 gen / 48 vars | none |
| 4 | core,spacing | 8 | ✅ survived (1809s) | **0 gen / 0 vars** | none |
| 5 | core,bye_spacing,spacing (FULL SWEEP) | 8 | ✅ survived (1801s) | **0 gen / 0 vars** | none |

Atom invariants asserted from logs: Run 3 BalancedByeSpacing present / EqualMatchUpSpacing absent;
Run 4 the exact swap; Run 5 BOTH spacing atoms present. Full per-run detail + caveats in
`scripts/e2e_symmetry_readout.md`.

### DoD status
- DoD-1…3 (launch path, raw profile, CP-SAT symmetry captured): ✅ (Units A+B, verified live).
- DoD-4 (liveness, both core runs): ✅ — Run 2 full 30 min; Run 1 reached search, accepted by convenor.
- DoD-5 (cross-run symmetry readout + Goal-3 conclusion): ✅ — `scripts/e2e_symmetry_readout.md`.
- DoD-6 (debug to liveness via resourcing): ✅ — OOM resolved by workers 10→8 (no semantic change).
- DoD-7 (background, killed at 30 min, artifacts saved): ✅ — `run_e2e_30min.py` (psutil tree-kill).
- DoD-8 (gates): launcher/wrapper compile clean; changes are scripts only.
- DoD-9…12 (spacing-family): ✅ — Runs 3/4 liveness + invariants + readout extended.

### Findings (Goals 2/3 + the appended full sweep)
- **`ClubGameSpread` strips ~all exploitable symmetry**: core−CGS 40,594 symmetric vars → full core 48
  (~99.9% reduction). `bye_spacing` is symmetry-neutral (transient first-pass bump only).
  **`EqualMatchUpSpacing` removes the remainder → 0 generators.** The **full sweep (Run 5) also has 0
  symmetry** — the maximal model is fully symmetry-broken.
- **Feasibility is hard, and it is NOT a symmetry problem** (symmetry is 0 with spacing on). No run
  with the full constraint set found a feasible incumbent in 30 min (Run 2's escaped orphan ran ~66 min,
  still `best:-inf`); bounds stay very loose (`next:[-4.1M, …]`). This matches the documented state that
  the complete set is **infeasible at slack 0** — **slack is the release lever**, not week-1 fixing
  (which here only removed symmetry that no longer exists and would risk over-constraining) and not more
  workers (a feasibility-hardness, not throughput, issue; 16 workers would also OOM on this box).

### Remaining closeout (NOT done this session — left per convenor instruction)
- Plan status stamp → `done`, dep-tree + `GOALS.md` update.
- Doc registry: new `docs/system/SOLVER_E2E.md`, CLAUDE.md quick-command + symmetry-capture note,
  `docs/README.md` / `docs/system/README.md` registration.
- Final `done`-stamp should follow **spec-038 going `done`** (spec-035 `depends_on` it; still `building`).
