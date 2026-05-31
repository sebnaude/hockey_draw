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

---

# spec-044 acceptance evidence — umbrella-aware PHL Sunday floor (Unit B)

Generated 2026-05-31 · year=**2026 (real production config)** · venv
`C:\Users\c3205\Documents\Code\python\draw\.venv` · run from worktree
`draw-s044-B` (branch `spec044-unitB`, with the Unit A fix landed).

Records DoD-5 (deterministic capacity proof), DoD-8 (30 s quick re-probes) and
DoD-9 (genuine 5-min solves) for the two `core` groups after the
umbrella-Friday-aware floor fix. All four solver probes were run with
`scripts/bisect_realconfig_feasibility.py --probe` (verdict from the script's own
`_classify` on each child's CP-SAT log).

## DoD-5 — deterministic capacity proof (post-fix)

`scripts/gosford_sunday_capacity_proof.py` (builds X first, so the per-pair
helpers see materialised `data['games']`). Every PHL club's atom Sunday floor
(`Σ tp_min`) is now within its Sunday capacity (`R − forced Friday`):

| club | Σ tp_min (atom floor) | forced Fri (min) | Sunday cap = R−Fri | verdict |
|------|-----------------------|------------------|--------------------|---------|
| **Gosford**  | **0**  | 8 | 12 | ok |
| **Maitland** | **7**  | 2 | 18 | ok |
| Norths   | 13 | 0 | 20 | ok |
| Souths   | 13 | 0 | 20 | ok |
| Tigers   | 13 | 0 | 20 | ok |
| Wests    | 12 | 0 | 20 | ok |

PHL: T=6, R=20 → base=4 (each pair meets in [4,5]). Pre-fix Gosford and Maitland
both had Σ tp_min = 19 (no umbrella subtraction) → 19 > 12 and 19 > 18, the
documented INFEASIBLE blocker. Post-fix the more-constrained-club umbrella term
(`umb(Gosford)=8`, `umb(Maitland)=2`) is subtracted from the LOWER bound only,
clamping the away clubs' floors to 0 / 7.

Note on the Maitland Σ = 7 (the spec's stale worked example said 9): the
per-pair breakdown is
`Gosford 0, Norths 2, Souths 1, Tigers 2, Wests 2 → 7`.
The Maitland-vs-Gosford pair's floor is driven to 0 by Gosford's dominating
`umb=8` under the max-over-both-clubs rule (`max(0, 4−0−8)=0`), and the
Maitland-vs-Souths pair has a pair-named NIHC Friday so its floor is 1 not 2
(`max(0, 4−1−2)=1`). These two corrections (the spec missed the
Gosford-dominated pair and the Souths pair-named entry) take 9 down to 7. This
figure is recomputed by hand in the new regression test's class docstring and
asserted there.

## DoD-8 — quick 30 s re-probes (8 workers)

`scripts/bisect_realconfig_feasibility.py --probe --groups core [--exclude
ClubGameSpread] --max-time 30 --workers 8`. Verdict from the script's `_classify`.

| probe | CLI flags | verdict |
|-------|-----------|---------|
| core (with ClubGameSpread)   | `--probe --groups core --max-time 30 --workers 8 --run-id s044_core30` | **UNKNOWN** |
| core − ClubGameSpread        | `--probe --groups core --exclude ClubGameSpread --max-time 30 --workers 8 --run-id s044_coreNoCGS30` | **UNKNOWN** |

**Honest deviation from the spec's DoD-8 expectation.** The spec predicted a
clean `INFEASIBLE_PRESOLVE → REACHED_SEARCH` flip at 30 s. In practice **both
30 s probes return UNKNOWN, both pre- and post-fix** — the 30 s cap is hit
*mid-presolve* on the ~82k-var model (no `#Bound`/`Starting search` line is
emitted before the cap), so the script cannot classify them as REACHED_SEARCH
and there is no INFEASIBLE proof either. The capacity contradiction this spec
fixes is **not** caught in fast presolve on the full model — it needs search the
30 s cap can't reach. So the 30 s probe is inconclusive in BOTH directions; the
honest post-fix signal is "no presolve-stage INFEASIBLE," which holds. The
unambiguous before/after evidence is the deterministic DoD-5 capacity proof and
the DoD-9 5-min runs below (which DO reach search). This matches the brief's
finding (c).

## DoD-9 — genuine 5-min solves (300 s, 8 workers)

`scripts/bisect_realconfig_feasibility.py --probe --groups core [--exclude
ClubGameSpread] --max-time 300 --workers 8`. (The `e2e_real_config_solve.py`
launcher's fixed 5-profile wiring doesn't take ad-hoc `--exclude`; per the spec's
own fallback note the bisect `--probe ... --max-time 300` gives the same real
solve + verdict, so it was used for both runs. **Deviation from the spec's
literal `e2e_real_config_solve.py` command, using the documented fallback.**)

| run | CLI flags | reached search | best objective | final CP-SAT status |
|-----|-----------|----------------|----------------|---------------------|
| core (with ClubGameSpread) | `--probe --groups core --max-time 300 --workers 8 --run-id s044_core300` | ✅ `#Bound @74.51 s` | `best:-inf` (no incumbent) | **UNKNOWN** |
| core − ClubGameSpread | `--probe --groups core --exclude ClubGameSpread --max-time 300 --workers 8 --run-id s044_coreNoCGS300` | ✅ `#Bound @68.24 s` | `best:-inf` (no incumbent) | **UNKNOWN** |

## spec-044 findings

1. **The alignment atom is no longer the `core` presolve/search blocker.** Both
   5-min runs get past presolve into search (`#Bound` at 74.51 s / 68.24 s) and
   run the full 300 s cap without proving INFEASIBLE. Contrast the **pre-fix**
   `e2e_real_readout` runs 1 & 2 above, which both reached a **proven
   INFEASIBLE** at ~94 s / ~164 s. The deterministic DoD-5 proof shows exactly
   why: the away clubs' Sunday floor (`Σ tp_min`) is now within capacity for
   every PHL club, where pre-fix Gosford (19 > 12) and Maitland (19 > 18)
   overflowed.

2. **No feasible incumbent in 5 minutes — recorded honestly, NOT a flip.** Both
   runs stay `best:-inf` and end UNKNOWN. This is *not* a clean
   INFEASIBLE→FEASIBLE flip and is not claimed as one. The remaining slack-0
   tightness of the full real config (symmetry, spacing, and the separate
   out-of-scope BalancedByeSpacing contradiction documented in the pre-fix
   findings above) means a feasible incumbent needs `--slack` and/or longer
   search than the 5-min probe allows. The DoD-9 evidence is "past presolve into
   genuine search with the floor overflow gone," which both runs clear; the
   deterministic DoD-5 capacity proof is the unambiguous before/after signal.

3. **ClubGameSpread is irrelevant to this blocker.** With- and
   without-ClubGameSpread runs behave identically (UNKNOWN at 30 s; reached
   search + `best:-inf` + UNKNOWN at 300 s), confirming the spec's
   "ClubGameSpread is a red herring" claim for the alignment floor.

## spec-044 per-run solver logs

- core 30 s: `logs/solver_20260531_160812_s044_core30.log` (UNKNOWN, mid-presolve)
- core − ClubGameSpread 30 s: `logs/solver_20260531_160912_s044_coreNoCGS30.log` (UNKNOWN, mid-presolve)
- core 300 s: `logs/solver_20260531_161025_s044_core300.log` (#Bound @74.51 s, best:-inf, UNKNOWN)
- core − ClubGameSpread 300 s: `logs/solver_20260531_172143_s044_coreNoCGS300.log` (#Bound @68.24 s, best:-inf, UNKNOWN)

## DoD-9 RE-VERIFICATION on merged final-form (2026-05-31, this session)

The original DoD-9 logs were written in a concurrent session's torn-down worktree and
were NOT preserved into final-form (only the summary above survived). Re-ran both 5-min
solves on the MERGED final-form (`cea9aca`) so the artefacts persist. Setup identical:
`scripts/bisect_realconfig_feasibility.py --probe --groups core [--exclude ClubGameSpread]
--max-time 300 --workers 8`.

| run | reached search | final status | best_bound | incumbent | verdict | log |
|-----|----------------|--------------|------------|-----------|---------|-----|
| core (with ClubGameSpread) | ✅ Starting search @73.65s | UNKNOWN | −2328 | none (`best:-inf`) | **PASS** | `logs/solver_20260531_200348_s044_core300_verify.log` |
| core − ClubGameSpread | ✅ Starting search @40.57s | UNKNOWN | −5044 | none (`best:-inf`) | **PASS** | `logs/solver_20260531_200916_s044_coreNoCGS300_verify.log` |

**Both PASS**: each cleared presolve and entered genuine search within the 5-min cap; neither
returned INFEASIBLE. Pre-fix, both reached a proven INFEASIBLE at presolve (~94s/164s) — so the
away-club Sunday-floor overflow is gone, independently reconfirmed against the merged code with
saved logs. As before, no feasible incumbent in 5 min (`best:-inf`, UNKNOWN) — expected on the
slack-0 real config; NOT claimed as a feasible flip. ClubGameSpread remains irrelevant to the
blocker (both behave the same).
