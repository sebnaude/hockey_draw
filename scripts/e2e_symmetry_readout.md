# spec-035 Unit C — raw `--core` e2e liveness + cross-run symmetry readout

<!-- Artifact for spec-035 DoD-5. Generated 2026-05-30 from two real CP-SAT solves
     on the forced-free `season_test` config (2026 base teams/fields, no forced
     games, week 1 NOT fixed). Logs captured via Unit B (`attach_cpsat_log_capture`);
     symmetry parsed via `parse_symmetry_stats`. -->

**Date:** 2026-05-30
**Config:** `config/season_test` (forced-free; `forced_games == []`; 2026 base teams + field availability; week 1 NOT fixed)
**Launcher:** `scripts/run_core_e2e.py` (Unit A) driven by `scripts/run_e2e_30min.py` (Unit C 30-min timeout wrapper, clean Windows process-tree kill)
**Blocker history:** Unit C was BLOCKED on a hard-vs-hard presolve infeasibility
(`ClubVsClubStackedWeekends × ClubDayParticipation` on forced-free `season_test`,
see `spec-035-e2e-infeasibility-handoff.md §11`). **Resolved** by spec-038 commit
`6f4c0e5` ("exempt club-day weekends from ClubVsClub Layer-6 nesting"). A fresh
bisect probe (2026-05-30, `bisect_core_feasibility.py --max-time 150 --workers 10`)
then reached search, confirming the model is solvable-shaped.

---

## The two runs

The runs differ by exactly **one model atom-group** — `ClubGameSpread` (both its
hard and soft atoms) — which is the variable under test. A second, non-model
difference (worker count) was forced by a hardware resourcing limit; see Caveats.

| | **Run 1** | **Run 2** |
|---|---|---|
| Profile | `--groups core --exclude ClubGameSpread` | `--groups core` (full core) |
| `exclude` (sidecar) | `["ClubGameSpread"]` | `[]` |
| Workers | 10 (DoD-2) | 8 (convenor-authorised resourcing, see Caveats) |
| Symmetry breakers | on | on |
| Forced games / locks / fix-round-1 | none / none / no | none / none / no |
| Solver log | `logs/solver_20260530_110409_run1_noCGS.log` | `logs/solver_20260530_115929_run2_fullcore_w8.log` |
| Profile sidecar | `logs/core_e2e_profile_run1_noCGS.json` | `logs/core_e2e_profile_run2_fullcore_w8.json` |

---

## Goal 1 — Liveness (DoD-4)

Both runs **built, cleared presolve, and entered real CP-SAT search.**

| | Run 1 (core − ClubGameSpread) | Run 2 (full core) |
|---|---|---|
| Raw `#Variables` | 121,262 (96,537 bools) | 147,100 (100,987 bools) |
| Presolved `#Variables` | 110,327 | 131,743 |
| Search start | **`Starting search at 59.90s with 10 workers`** | **`Starting search at 114.76s with 8 workers`** |
| Search duration before stop | reached `#Model 379.21s`, then OOM-crashed at ~406s (~6.8 min) | reached `#Model 1549.40s`; **killed cleanly at the 30-min cap** (elapsed 1892s, `killed_at_cap=True`) |
| Liveness verdict | **reached search** — recorded as success by the convenor (2026-05-30) | **reached search + survived ≥30 min** — full DoD-4 pass |

**Run 1 OOM (DoD-6 resourcing, not a model defect):** at workers=10 the
~110k-var model grew past available RAM at ~6.8 min (monitor logged
`CRITICAL Memory: 96.8%, 513MB available` immediately before the abort
`rc=0xC0000409`). The machine has 15.6 GB total with only ~4.5 GB free at idle
(the remainder is the user's everyday apps). The model is solvable-shaped — it
reaches search and runs normally — so this is purely a resourcing ceiling, which
DoD-6 scopes as "debug via resourcing, not semantics." The convenor's resolution
(2026-05-30) was to **accept Run 1 as a liveness success** (search reached) and
run the remaining solves at **workers=8** for headroom. Run 2 at workers=8
confirmed the fix: it ran the full 30 minutes at ~96% memory without crashing.

---

## Goal 2 / Goal 3 — Remaining symmetry, and the effect of `ClubGameSpread`

CP-SAT runs presolve symmetry detection one or more times. The **full-model pass**
(first detection, on the freshly-built model) is the inherent-symmetry headline;
later passes run on the partially-presolved model. All passes are recorded below.

### Run 1 — core − ClubGameSpread (the symmetry-rich model)

| Pass | #generators | orbits | variables in orbits | orbit sizes |
|---|---|---|---|---|
| 1 (full model) | **16** | **10,804** | **97,530** | 16 |
| 2 | 9 | 20,274 | 65,890 | 4 |
| 3 | 11 | 20,274 | 76,026 | 6 |
| 4 (nearest search) | 5 | 20,297 | 40,594 | 2 |

### Run 2 — full core, +ClubGameSpread (the symmetry-poor model)

| Pass | #generators | orbits | variables in orbits | orbit sizes |
|---|---|---|---|---|
| 1 (only pass) | **1** | **24** | **48** | 2 |

### Conclusion (Goal 3): `ClubGameSpread` is a powerful symmetry-breaker

Adding the `ClubGameSpread` atoms **collapses the model's exploitable symmetry to
near-zero**:

- full-model generators: **16 → 1**
- orbits: **10,804 → 24**
- variables sitting in symmetric orbits: **97,530 → 48** (≈ 99.95% reduction)
- CP-SAT stops re-running symmetry detection entirely (4 passes → 1) — there is
  essentially nothing left to exploit once `ClubGameSpread` is present.

This is the answer to the spec-035 Goal-3 hypothesis: **`ClubGameSpread` does NOT
leave symmetry on the table — it strips almost all of it.** Mechanistically this is
expected: `ClubGameSpread` pins how each club's games spread/concentrate across
weeks and fields, which destroys the week/game interchangeability that produces the
large size-16 and size-2…6 orbits seen in the core−ClubGameSpread model.

**Practical implication:** the dedicated symmetry-breaking atoms (lex ordering, NIHC
fill order) matter most when `ClubGameSpread` is *absent* or relaxed. On the full
production `core`, `ClubGameSpread` already removes the symmetry those breakers
target — so symmetry-driven search blow-up is not a concern for the full set.

---

## Baseline note (DoD-5)

**No historical symmetry baseline exists** — confirmed by the spec-035 build-time
filesystem + checkpoint scan, and re-confirmed this session (no stored `[Symmetry]`
artifact predates these runs). These two runs are therefore recorded as the **first
baseline**. The full-core numbers (1 generator / 24 orbits / 48 vars) are the
reference for any future comparison; Run 1 is the within-session
`−ClubGameSpread` comparator.

---

## Caveats

### C001: worker-count delta between the two runs (10 vs 8)
- **dimensions**: functional_impact=0.2, silent_failure=0.1
- **timing**: before-release
- **proposed_disposition**: dismissed-with-evidence
- Run 1 ran at workers=10 (DoD-2); Run 2 at workers=8 (convenor-authorised DoD-6
  resourcing override after the Run-1 OOM). **This does not confound the symmetry
  comparison:** CP-SAT detects symmetry on the *model* during main-thread presolve,
  independently of the number of search worker threads. The only *model* delta
  between the runs is `ClubGameSpread` (confirmed byte-for-byte by the two profile
  sidecars: identical except `exclude` and `workers`). Worker count affects only
  search throughput, not the presolve `[Symmetry]` graph. Evidence: the symmetry
  blocks are emitted at 11:04:28 (Run 1) / 12:01:50 (Run 2), both *before* the
  "Starting search at … with N workers" line.

### C002: Run 1 did not complete the full 30-minute window
- **dimensions**: functional_impact=0.3
- **timing**: before-release
- **proposed_disposition**: dismissed-with-evidence
- Run 1 OOM-crashed at ~6.8 min at workers=10; it reached search and was accepted
  as a liveness success by the convenor. The symmetry readout for Run 1 is
  unaffected — the `[Symmetry]` block lands during presolve (~within the first
  minute), well before the crash. A clean full-30-min Run-1 equivalent is available
  on demand by re-running `--exclude ClubGameSpread --workers 8` if a complete
  window is later required.
