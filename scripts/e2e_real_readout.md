# REAL-2026-config e2e progression — readout

Generated 2026-05-30 · year=**2026 (real production config)** · workers=8 · cap=6 min/run
(~5 min into search after presolve) · raw: **no slack**, week-1 unfixed, no locks.

Emulates the spec-035 stage progression, but on the **full real 2026 config**
(`load_season_data(2026)`: 18 FORCED_GAMES, 69 BLOCKED_GAMES incl. the PHL premiership-weekend
exemptions, field unavailabilities, LOCKED_PAIRINGS) instead of the forced-free `season_test`.
Same solve entry point (`main_staged.main_simple` via `run._resolve_group_selection`). Runs
sequential; each child process-tree-killed at the cap. Decision variables built: **82,053**
(after the home-venue filter eliminated 18,610 and blocked rules eliminated 14,132); the CP-SAT
optimization model expands to ~129k internal vars / ~335k constraints.

> **Correction note.** Two earlier drafts of this file were wrong and have been replaced:
> (1) a version copied from the 2-min *smoke* test ("1 gen / 48 vars, all survived"), and
> (2) a version that guessed runs 3 & 5 died of OOM. The table below is **verified line-by-line
> against each run's CP-SAT solver log** (final `CpSolverResponse status` + the presolve trace).

## Results (verified from solver logs)

| # | run | groups | exclude | reached search | **final CP-SAT status** | when | symmetry (first detection) |
|---|-----|--------|---------|----------------|-------------------------|------|----------------------------|
| 1 | run1_core_noCGS   | core                     | ClubGameSpread | ✅ (#Bound @75s) | **INFEASIBLE** (proven) | ~94 s | 7 gen / 53,839 vars / 16,151 orbits |
| 2 | run2_core_full    | core                     | —              | ✅ (#Bound @144s) | **INFEASIBLE** (proven) | ~164 s | 7 gen / 64,194 vars / 19,257 orbits |
| 3 | run3_core_bye     | core,bye_spacing         | —              | ❌ | **INFEASIBLE — proven during initial copy** | ~2 s | (died before symmetry phase) |
| 4 | run4_core_spacing | core,spacing             | —              | ✅ (#Bound @324s) | **UNKNOWN** (`best:-inf`, ran to 6-min cap) | killed @360 s | 35 gen / 70 vars / 35 orbits |
| 5 | run5_full_sweep   | core,bye_spacing,spacing | —              | ❌ | **INFEASIBLE — proven during initial copy** | ~2 s | (died before symmetry phase) |

All five **built the real model and entered CP-SAT.** None were OOM (process memory stayed
~370 MB; box at ~80 % but the solve never hit an allocation failure). No run found a feasible
incumbent.

## Findings

1. **The real 2026 model at slack 0 is INFEASIBLE — confirming the documented state.** This is
   the expected "complete set is infeasible at slack 0; **slack is the release lever**" result
   (spec-033/036 convenor notes), now demonstrated on the real config end-to-end. Runs 1 & 2
   prove it the hard way (presolve + search → INFEASIBLE in 94 s / 164 s); run 4 (core+spacing)
   couldn't prove it either way inside the 6-min probe (stayed `best:-inf`).

2. **The `bye_spacing` group makes it *trivially, immediately* infeasible.** Runs 3 and 5 — the
   two that include `bye_spacing` — are **proven infeasible during CP-SAT's initial copy** (~2 s,
   before presolve/symmetry/search), on constraint #94840: a linear term `2·x ≤ 1` where the
   variable `x` is already pinned to `{1}` (so `2 ≤ 1`, a direct contradiction). This is a
   `BalancedByeSpacing` constraint at base slack 0 against a forced game. It matches the spec-033
   note that BalancedByeSpacing's raw-ideal floor "risks infeasibility" (which is why its base
   slack was raised 0→2) — at slack 0 on the real forced config it is an outright contradiction,
   not just tight. **This is a constraint-semantic / slack finding, not a resourcing one.**

3. **`EqualMatchUpSpacing` (`spacing`) is the dominant symmetry-breaker on the real config.**
   core alone → 7 generators / 64,194 symmetric vars; core−ClubGameSpread → 7 gen / 53,839 (so on
   *production* data ClubGameSpread is **not** the lever it was on forced-free `season_test`);
   **core+spacing → 35 gen / 70 vars** — almost fully broken. Consistent with the spec-035
   `season_test` finding that EqualMatchUpSpacing removes essentially all remaining symmetry.

## What this run did and did NOT establish

- **Did:** the real-config solve path is live (build → CP-SAT) with the real forced/blocked games
  and PHL exemptions in effect; symmetry is captured per group; and the real model's slack-0
  infeasibility (and bye_spacing's role in it) is now demonstrated, not just asserted.
- **Did NOT:** produce a feasible/publishable draw (not the goal), and did not run the formal
  30-min liveness bar (this was the convenor's "~5 min into solve" probe — 6-min wall cap each).
  To get feasibility, re-run with `--slack N` (the documented release lever); to get the 30-min
  bar, raise the cap.

## Per-run logs

- run1_core_noCGS: `logs/solver_20260530_151054_real_run1_core_noCGS.log` · stdout `logs/e2e_real_run1_core_noCGS.stdout.log`
- run2_core_full: `logs/solver_20260530_151302_real_run2_core_full.log` · stdout `logs/e2e_real_run2_core_full.stdout.log`
- run3_core_bye: `logs/solver_20260530_151623_real_run3_core_bye.log` (INFEASIBLE @ initial copy, const #94840) · stdout `logs/e2e_real_run3_core_bye.stdout.log`
- run4_core_spacing: `logs/solver_20260530_151708_real_run4_core_spacing.log` · stdout `logs/e2e_real_run4_core_spacing.stdout.log`
- run5_full_sweep: `logs/solver_20260530_152310_real_run5_full_sweep.log` (INFEASIBLE @ initial copy) · stdout `logs/e2e_real_run5_full_sweep.stdout.log`

## Caveats

- **~5-min-into-search probe, not the formal 30-min bar.** 6-min wall cap per run.
- "reached search" / "final status" are read from each run's CP-SAT solver log
  (`CpSolverResponse status` + post-presolve `#Bound` lines), not the driver's looser keyword
  scan (whose interim "INFEASIBLE" matches were subsolver chatter).
