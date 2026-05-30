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

**Methodology note (important):** CP-SAT runs symmetry detection *several times*
during presolve, and the count changes between passes as presolve simplifies the
model. The **first** pass is on a barely-presolved model and is NOT a fair
cross-run number (different constraint sets leave different amounts of *transient*
helper-variable symmetry that later presolve rounds then break). The fair,
apples-to-apples figure is the **last / converged pass — closest to the start of
search** — which is exactly what `parse_symmetry_stats` returns. All passes are
listed per run below; the **converged pass is the one compared in the conclusion.**

### Run 1 — core − ClubGameSpread (the symmetry-rich model)

| Pass | #generators | orbits | variables in orbits | orbit sizes |
|---|---|---|---|---|
| 1 (first) | 16 | 10,804 | 97,530 | 16 |
| 2 | 9 | 20,274 | 65,890 | 4 |
| 3 | 11 | 20,274 | 76,026 | 6 |
| **4 (converged)** | **5** | **20,297** | **40,594** | 2 |

### Run 2 — full core, +ClubGameSpread (the symmetry-poor model)

| Pass | #generators | orbits | variables in orbits | orbit sizes |
|---|---|---|---|---|
| **1 (only / converged)** | **1** | **24** | **48** | 2 |

### Run 3 — full core + bye_spacing (DoD-10; see spacing-family section)

| Pass | #generators | orbits | variables in orbits | orbit sizes |
|---|---|---|---|---|
| 1 (first, transient) | 1 | 6,491 | 12,982 | 2 |
| **2 (converged)** | **1** | **24** | **48** | 2 |

### Conclusion (Goal 3): `ClubGameSpread` is a powerful symmetry-breaker

Compared at the **converged pass** (the only fair comparison):

| Model | converged symmetric vars | converged orbits | generators |
|---|---|---|---|
| core − ClubGameSpread | **40,594** | 20,297 | 5 |
| full core (+ClubGameSpread) | **48** | 24 | 1 |
| full core + bye_spacing | **48** | 24 | 1 |

Adding `ClubGameSpread` collapses the model's exploitable symmetry from ~40k
variables-in-orbits down to **48** — a ~99.9% reduction — and drops the generator
count from 5 to 1. This answers the Goal-3 hypothesis: **`ClubGameSpread` does NOT
leave symmetry on the table — it strips almost all of it.** Mechanistically,
`ClubGameSpread` pins how each club's games spread/concentrate across weeks and
fields, destroying the week/game interchangeability that produces the large
size-16 / size-2…6 orbits seen in the core−ClubGameSpread model.

**On the apparent bye_spacing anomaly (first-pass 12,982 vars):** adding
`bye_spacing` on top of full core looks, at the *first* symmetry pass, like it adds
symmetry (1 gen / 6,491 orbits / 12,982 vars). It does not. That is a transient
presolve artifact: `bye_spacing` adds barely any *raw* variables (+1,056 bools) but
**prevents presolve from eliminating ~25k booleans early** (presolved bools
51,140 → 76,436) that full-core-alone fixes outright — those surviving game-bools
are interchangeable in pairs at the first pass. CP-SAT's later presolve rounds then
break that symmetry, and Run 3's **converged pass returns to exactly 24 orbits / 48
vars — identical to full core.** So a strict superset of full core does *not* end up
with more residual symmetry, as expected; `ClubGameSpread`'s symmetry-stripping is
intact and dominant.

**Practical implication:** the dedicated symmetry-breaking atoms (lex ordering, NIHC
fill order) matter most when `ClubGameSpread` is *absent* or relaxed. On the full
production `core`, `ClubGameSpread` already removes the symmetry those breakers
target — so symmetry-driven search blow-up is not a concern for the full set.

---

## Spacing-family follow-on (Units D/E — DoD-9…12)

Per convenor request (2026-05-29), once core reached liveness the same raw harness
was extended to the two spacing-family groups that spec-032/033 peeled out of
`core`: first **add `bye_spacing`** (`BalancedByeSpacing`), then **swap to `spacing`**
(`EqualMatchUpSpacing`). The launcher gained a `--groups` parameter (Unit D); the
group set is the only delta from Run 2 (full core). The follow-on phase was gated
behind Open Question FP (the `ClubVsClubStackedWeekends × ClubDayParticipation`
hard conflict) — **resolved by spec-038 `6f4c0e5`**, the same fix that unblocked
Unit C, so the phase proceeded.

| | Run 2 (full core) | Run 3 (core,bye_spacing) | Run 4 (core,spacing) |
|---|---|---|---|
| Group set | `core` | `core,bye_spacing` | `core,spacing` |
| Spacing atom asserted | — | `BalancedByeSpacing` PRESENT; `EqualMatchUpSpacing` absent ✅ | `EqualMatchUpSpacing` PRESENT; `BalancedByeSpacing` absent ✅ |
| Workers | 8 | 8 | 8 |
| Raw → presolved vars | 147,100 → 131,743 | 148,996 → 133,961 | 149,453 → 140,219 |
| Search start | @114.8s | @182.2s | @112.3s |
| Liveness (30-min) | ✅ survived (1892s, killed @cap) | ✅ survived (1802s, killed @cap) | ✅ survived (1809s, killed @cap) |
| **Converged symmetry** | 1 gen / 24 orbits / **48 vars** | 1 gen / 24 orbits / **48 vars** | **0 gen / 0 orbits / 0 vars** |

### Marginal symmetry effect of the spacing groups

- **`bye_spacing`**: no net effect on residual symmetry (converged 48 vars, same as
  full core). It adds a transient first-pass bump (6,491 orbits / 12,982 vars) by
  keeping ~25k more bools alive through early presolve, which later rounds clear —
  see the bye_spacing-anomaly note above. Liveness fine.
- **`spacing` (`EqualMatchUpSpacing`)**: **removes the last residual symmetry.**
  CP-SAT ran symmetry detection (built the symmetry graph: 534k nodes, 6.5M arcs,
  ~4 passes) but found **zero generators** every pass — `parse_symmetry_stats`
  returns `{present: False}`. `EqualMatchUpSpacing` constrains the gap between a
  pair's repeated meetings, which breaks the final week-interchangeability that left
  full core with 48 symmetric vars. With it on, the model has **no exploitable
  symmetry at all**.

### Full progression (converged-pass, the headline)

| Model | symmetric vars | generators |
|---|---|---|
| core − ClubGameSpread | 40,594 | 5 |
| full core | 48 | 1 |
| full core + bye_spacing | 48 | 1 |
| full core + spacing | **0** | **0** |

`ClubGameSpread` does the heavy lifting (40,594 → 48); `bye_spacing` is symmetry-
neutral; `EqualMatchUpSpacing` eliminates the remainder. All four models reach
search and (at workers 8) sustain the 30-minute liveness bar.

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
