<!-- status: ready -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-030, spec-031, spec-032, spec-033, spec-034 -->
<!-- owner: session=none claimed=none -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline -->

# spec-035 — ULTIMATE: raw `--core` e2e solve on the forced-free test config + remaining-symmetry readout

> **This is the *ultimate* plan — truly the last thing done.** It runs only after **every other
> live spec is `done`, including the penultimate spec-034.** It is a real end-to-end solver run, not
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
   the exact resolved flag set into the run metadata/log so the profile is auditable.
3. **CP-SAT presolve output captured.** The run's log file contains CP-SAT's own presolve section,
   including the `[Symmetry]` lines (generators / orbits / variable+constraint reduction), by
   redirecting the CP-SAT solver stdout/log stream into the run log (it currently is not — only the
   Python `solver` logger is). Observable: `grep -i symmetry <logfile>` returns the presolve symmetry
   block.
4. **Liveness reached.** The run **gets through presolve** (model built; presolve completes; search
   starts — evidenced by a CP-SAT `#1`/`best:`/objective progress line or an explicit search-start
   marker in the captured log) **and survives ≥30 minutes of search**. The process is **killed at the
   30-minute mark regardless of whether a feasible solution was found** (wall-clock; clean kill of
   the background ortools process on Windows). Success criterion = *reached search* **and** *30 min
   elapsed without crash or presolve-time infeasibility*. (Finding a solution is a bonus, not
   required; not finding one is not a failure.)
5. **Remaining-symmetry readout.** A short readout artifact (e.g. `scripts/e2e_symmetry_readout.md`
   or a printed+saved summary) parses the captured `[Symmetry]` stats and states the **current**
   model's remaining symmetry (number of generators, orbit count/sizes, and the var/constraint
   reduction presolve reports). It explicitly records: "**no historical baseline exists** (filesystem
   + checkpoint scan, spec-035 build) — these numbers are recorded as the first baseline." If a
   baseline *is* discovered during the build-time re-scan, the readout diffs against it instead.
6. **Debugging to the liveness state.** If the model fails to reach search (presolve blowup, OOM at
   `--workers 10`, immediate infeasibility), it is **debugged to the DoD-4 state on the assumption
   that the core constraints are correct** — i.e. fixes target model construction / presolve /
   resourcing, **not** constraint semantics. Whatever was changed to get there is documented in the
   readout. If the obstruction turns out to be a genuine *constraint-semantic* bug (core constraint
   actually wrong), that is **out of scope** → file a new spec per `/basic`, do not patch semantics
   here.
7. **Run executed in background, killed at 30 min, artifacts saved.** The solve runs via Bash
   `run_in_background`, capped/killed at 30 minutes (Monitor or a timeout wrapper). `--workers 10`
   only. The log and the symmetry readout are saved under `logs/`/`scripts/` and referenced from the
   readout.
8. **Gates:** launcher + log-capture code type-checks clean; changed-file lint clean; AST sweep clean
   (no dead launcher branches, presolve-fail path logged at INFO via the project logger, never
   silently swallowed); spec-034's suite stays green after the Unit A/B code changes.

## Implementation units

Three units. A and B touch disjoint code (A: a new launcher script + possibly a `--config`/string
path in `run.py`'s arg layer; B: the CP-SAT stdout/log capture in `solver_diagnostics.py` /
`main_staged.py`) and can run in parallel. C is the actual run + readout + debugging and depends on
both.

### Unit A — Forced-free test-config launcher + raw core flag profile
- **Files touched:** new `scripts/run_core_e2e.(py|ps1)`; verification of `config/season_test.py`
  (and a fix there only if it has drifted from 2026 base teams/fields or isn't `forced_games == []`);
  *optionally* a string-accepting `--config`/year path in `run.py` if that is cleaner than a pure
  script (executor's call — prefer the script to avoid widening CLI surface, since `season_test`'s
  own docs already prescribe the programmatic path).
- **Change summary:** one-command launch of the raw core solve on the forced-free config with the
  exact DoD-2 flag profile, flags recorded to metadata.
- **Depends on:** spec-032 done (the `core` group + always-on `symmetry_breakers` must exist); spec-034
  done (don't e2e a red suite).
- **Executor model:** Opus (must thread the exact flag profile through the real solve entry point and
  guarantee `--fix-round-1` is absent / `fix_round_1=False` and locks are *off*; the flag name is
  `--fix-round-1`, NOT `--round1-symmetry` — review fix C1).
- **No-mock test outline:** *Given* data loaded via the chosen path (`get_season_data()` or
  `load_season_data('test')`), *then* `forced_games == []` and team set == the 2026 base team set
  (oracle: hand-listed from `season_2026.py`). *Given* the launcher's resolved config, *then* the
  recorded flag set contains `groups=['core']`, `workers==10`, `fix_round_1` falsy (NOT
  `round1_symmetry` — the actual solver attribute is `fix_round_1`; review fix C1), no locked weeks
  (oracle: the DoD-2 profile, asserted field-by-field).

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

### Unit C — The raw e2e run + 30-min kill + remaining-symmetry readout + debugging
- **Files touched:** new `scripts/e2e_symmetry_readout.md` (the artifact), plus any
  model-construction/presolve/resourcing fix that DoD-6 turns out to require (re-graded per `/basic`
  if it grows).
- **Change summary:** execute the launcher in the background at `--workers 10`, kill at 30 min, parse
  + record remaining symmetry, re-scan for any baseline, and debug to the liveness state if presolve
  doesn't clear.
- **Depends on:** Unit A (launcher) and Unit B (symmetry capture + parser).
- **Executor model:** Opus.
- **No-mock test outline:** this unit's "test" is the e2e run itself (real solve, no mocks):
  *Given* the launcher run in background with `--workers 10`, *when* 30 minutes elapse, *then* the
  captured log shows presolve completed + at least one search/progress line before the kill (oracle:
  presence of a CP-SAT search-start/`#1` marker), and the kill is clean (no orphaned ortools
  process). *Given* the captured log, *then* `parse_symmetry_stats` yields the readout numbers and
  the readout records "no baseline / first baseline."

## Doc registry

- `docs/system/SOLVER_E2E.md` *(new — register in the CLAUDE.md doc-index table AND in
  `docs/README.md` master doc-index under `docs/system/`)* — how to run the raw core e2e on the
  forced-free test config, the 30-min-kill protocol, `--workers 10`, the symmetry-capture mechanism,
  and where the readout lands. [Unit C, with the capture mechanism from Unit B] (review fix — M1:
  `docs/README.md` added — the master doc index must also list new `docs/system/` docs.)
- `CLAUDE.md` (repo root) — (a) a "Quick Commands" entry for the e2e core run via
  `scripts/run_core_e2e`; (b) a note that CP-SAT presolve symmetry is now captured to the log;
  (c) add `docs/system/SOLVER_E2E.md` to the doc-index table (row: "how to run the raw core e2e + 30-
  min-kill protocol; read when operating or reproducing a core e2e run"). [Unit C]
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
- **Comparing to an earlier main-branch run.** Dropped: no baseline exists (confirmed). We record the
  first baseline instead.
- **Tuning the solver** (workers beyond 10, parameter sweeps, warm-start hints) — `--workers 10` only,
  raw.
- **Fixing constraint *semantics*.** If a core constraint is genuinely wrong, that is a new spec, not
  this run's job (this run assumes the core constraints are correct and debugs only construction /
  presolve / resourcing).
- **The suite-green + coverage work** — that is spec-034 (the prerequisite penultimate plan).

## Dependencies

- **Other plans:** `depends_on: spec-030, spec-031, spec-032, spec-033, spec-034`. spec-032 supplies
  the `core` group and always-on `symmetry_breakers`; spec-034 supplies a green suite (don't e2e a red
  codebase). Per `/basic`, **not startable** until all five are `done` and merged.
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
5. Run Unit C: launch `scripts/run_core_e2e` via Bash `run_in_background` at `--workers 10`; **kill at
   30 minutes** (Monitor/timeout) regardless of solution state; parse + write the remaining-symmetry
   readout; debug to the DoD-4 liveness state if presolve doesn't clear (construction/resourcing
   only). If a constraint-semantic bug is the blocker, file a new spec and surface to the user.
6. Tick checkboxes, stamp the plan `done`, move to `docs/todo/done/`, update the dependency tree —
   the `final-form` plan line is now fully drained.
