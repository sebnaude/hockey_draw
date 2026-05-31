<!-- status: handoff -->
<!-- spec: spec-035 follow-on — get the full atom set (stage 5) feasible on the REAL 2026 config -->
<!-- author: opus session "second draw run" 2026-05-30 -->
<!-- goal: a feasible draw with ALL atoms applied (core + bye_spacing + spacing) on load_season_data(2026) -->

# spec-035 — REAL-CONFIG "stage 5" feasibility handoff

> **Read this as evidence + a plan, not a fix.** Goal: get **stage 5** — the full
> atom set (`core` + `bye_spacing` + `spacing`, symmetry_breakers on) — to a
> **feasible** solve on the **real 2026 production config** (`load_season_data(2026)`:
> 18 forced / 69 blocked games incl. PHL premiership exemptions, field
> unavailabilities, LOCKED_PAIRINGS). The working assumption (convenor): the
> remaining work is **loosening over-tight hard atoms**, not fixing bugs. This
> handoff says *which* atoms, with probe evidence, and flags the one that is **not
> currently loosenable**.

---

## 0. TL;DR — what needs loosening (and the catch)

| Atom | Is it a real-config blocker? | Slackable today? | Action |
|------|------------------------------|------------------|--------|
| **`ClubVsClubStackedWeekends`** | **YES — the sole `core` blocker** (probe P1) | **NO — `slack_key` removed by spec-033 ("fixed-hard")** | **Add a slack mechanism (or semantic relax). This is the critical path.** Owned by spec-038. |
| **`BalancedByeSpacing`** | YES — separate trivial contradiction (probe P5) | YES (`slack_key='BalancedByeSpacing'`, base_slack=2) | Loosen via `--slack BalancedByeSpacing N`, and/or fix the forced-bye structural case (below). |
| `EqualMatchUpSpacing` | Not isolated as a blocker in this trace | YES (`slack_key='EqualMatchUpSpacingConstraint'`, base_slack=2) | Loosen with `--slack` *if* it blocks once the two above are cleared. |
| `ClubDayParticipation` | No (dropping `club_day` alone did NOT relieve — P2) | No slack_key | Leave; not implicated on real config. |
| `AwayClubHomeWeekendsCount` + `AwayClub…HomeBalance` | No (dropping `home_away` alone did NOT relieve — P3) | n/a | Leave; not implicated on real config. |

**Headline:** the convenor's "it's the spacing ones" intuition is **half right**.
`BalancedByeSpacing` genuinely needs loosening. But the **dominant** blocker is
`ClubVsClubStackedWeekends`, a HARD atom that spec-033 deliberately made
**non-slackable** — so `--slack` alone cannot reach stage-5 feasibility today.
Something has to give that atom a release valve first.

---

## 1. EVIDENCE — the real-config bisection

Harness: `scripts/bisect_realconfig_feasibility.py` (NEW this session — the
real-2026 analogue of `scripts/bisect_core_feasibility.py`, which only does
forced-free `season_test`). Driver runs each probe as its own child process,
classifies its CP-SAT log, writes `scripts/realconfig_infeasibility_trace.md`.

Run: `year=2026`, `--workers 8`, `--max-time 200` per probe, raw (no slack,
week-1 unfixed). Reproduce:

```powershell
C:\Users\c3205\Documents\Code\python\draw\.venv\Scripts\python.exe `
  C:\Users\c3205\Documents\Code\python\draw-final-form\scripts\bisect_realconfig_feasibility.py `
  --max-time 200 --workers 8
```

| Probe | Groups | Dropped atoms | Verdict |
|-------|--------|---------------|---------|
| P0 | core | — (full core) | **INFEASIBLE** (presolve) |
| **P1** | core | ClubVsClubStackedWeekends, ClubVsClubStackedCoLocation | **REACHED_SEARCH ✅** |
| P2 | core | club_day (5 atoms) | **INFEASIBLE** |
| P3 | core | home_away (2 atoms) | **INFEASIBLE** |
| P4 | core | club_alignment + club_day (7 atoms) | **REACHED_SEARCH ✅** |
| P5 | core,bye_spacing | — | **INFEASIBLE_INITIAL_COPY** |

**Reading:** the only probes that reach search are P1 and P4 — **both drop
`ClubVsClubStackedWeekends`.** Dropping `club_day` alone (P2) or `home_away` alone
(P3) does *not* help. So `ClubVsClubStackedWeekends` is the **sole** `core`-level
infeasibility source on the real config. P5 shows `bye_spacing` adds a *separate,
independent* contradiction on top.

P0 signature (presolve): `rule 'linear + amo: infeasible linear constraint' was
applied 221 times` → `INFEASIBLE: proven during presolve`.
P5 signature (initial copy, ~2s): a `2·x ≤ 1` linear constraint with `x` pinned to
`{1}` — i.e. `2 ≤ 1` (the BalancedByeSpacing forced-bye case in §3).

---

## 2. IMPORTANT — real config ≠ season_test

The earlier `season_test` handoff (`spec-035-e2e-infeasibility-handoff.md` §11)
localised the forced-free blocker to the **pair** `ClubVsClubStackedWeekends ×
ClubDayParticipation` (both club-day dates were Sundays). **On the real config that
is NOT the pair:** dropping `club_day` (P2) leaves it INFEASIBLE, so here
`ClubVsClubStackedWeekends` conflicts with the always-on fundamentals / the real
forced+blocked structure, **independently of club_day**. The two configs fail for
related-but-different reasons — do not assume the season_test root-cause carries
over. The real config is the one that matters for a publishable draw.

---

## 3. MECHANISM — why each blocker is infeasible (evidence + open part)

### 3a. `BalancedByeSpacing` (P5, initial-copy `2x≤1`)
`constraints/atoms/balanced_bye_spacing.py`. For a round where a team has **zero
candidate game vars** (all removed by blocked games / venue filter), the atom sets
that round's bye indicator to `model.NewConstant(1)` — a *forced bye* (line ~192).
The pairwise rule forbids two byes within the spacing window `S` (`S = max(0,
ideal_bye_gap(R, byes) - bye_spacing_base_slack - config_slack)`). If a team has
**two forced byes within `S` rounds**, the pairwise clause becomes `1 + 1 ≤ 1` →
proven infeasible at initial copy. The real config's premiership-weekend blocks
(which null whole grades on specific Sundays) create exactly these adjacent forced
byes. `bye_spacing_base_slack` is already 2 in `season_2026.py`, and it is still
infeasible → needs MORE release, or a structural carve-out.
- **Loosen:** `--slack BalancedByeSpacing N` large enough to drive `S → 0` for the
  affected grades (when `S=0` the pairwise loop is skipped entirely — `if gap >
  max_gap: continue`, `max_gap=0`). OR (structural) exclude forced byes that come
  from config-nulled rounds from the pairwise floor — arguably the *correct* fix,
  since those byes aren't schedulable choices. **Decide which before sizing slack.**

### 3b. `ClubVsClubStackedWeekends` (P0/P1, the core blocker)
`constraints/atoms/club_vs_club_stacked_weekends.py` (spec-038 four-layer model).
Hard layers: per-team-pair Sunday budget `== per_matchup`; per-aligned-weekend
cardinality `== min(a,b)·play_pg`; per-pair-grade total `== weekends_budget`;
cross-grade nested-superset chain. On the real config one (or the interaction) of
these exact `==` budgets is unsatisfiable given the real forced/blocked Sunday
availability — `linear + amo: infeasible linear constraint` ×221 in presolve.
- **OPEN (not line-traced):** *which* layer over-constrains was NOT isolated. Next
  diagnostic step = an assumptions/IIS-style probe, or per-layer instrumentation,
  to name the specific `==` that should become a range. This tells you both the fix
  shape and the slack size.
- **THE CATCH:** this atom has **no `slack_key`** — `constraints/registry.py:~385`
  says *"spec-033 Unit A: slack_key removed — alignment is fixed-hard, no slack."*
  So `--slack` does nothing for it today. **To loosen it you must first re-introduce
  a slack mechanism** (a `slack_key` + slack-aware budget: e.g. relax the
  per-pair-grade `== weekends_budget` to `[weekends_budget - slack, weekends_budget]`,
  or the per-aligned-weekend `== min(a,b)` to `<= min(a,b)`). This is an edit to the
  atom (spec-038 owns it), not a config change — so strictly it's a small spec, but
  the *intent* is loosening, not a bug fix, consistent with the convenor's framing.

---

## 4. SLACK / REGISTRY FACTS (verified in `constraints/registry.py`)

| Atom | slack_key | base slack (season_2026) |
|------|-----------|--------------------------|
| EqualMatchUpSpacing | `EqualMatchUpSpacingConstraint` | `spacing_base_slack = 2` |
| BalancedByeSpacing | `BalancedByeSpacing` | `bye_spacing_base_slack = 2` |
| ClubGameSpread | `ClubGameSpread` | — |
| ClubNoConcurrentSlot | `ClubNoConcurrentSlot` | — |
| **ClubVsClubStackedWeekends** | **none (removed, spec-033)** | — |
| ClubDayParticipation (+ other club_day) | none | — |
| AwayClubHomeWeekendsCount / …HomeBalance | none | — |

`--slack` is passed as `data['constraint_slack'][<slack_key>] = N`; an atom with no
`slack_key` cannot be reached by it. `get_slack_key()` is the lookup
(`registry.py:761`).

---

## 5. THE PLAN to get stage 5 feasible

Ordered; each step gated on the previous because they stack:

1. **Trace `ClubVsClubStackedWeekends`'s specific over-tight layer** (assumptions/IIS
   probe or per-layer instrument). Output: the exact `==` budget to relax + by how
   much. *(Diagnostic — no code change yet.)*
2. **Add a slack mechanism to `ClubVsClubStackedWeekends`** (spec-038 owns it): a
   `slack_key`, and turn the implicated `==` into a slack-parameterised range.
   Re-run probe P0-equivalent with `--slack <newkey> N`; raise N until REACHED_SEARCH.
3. **Resolve `BalancedByeSpacing` forced-bye contradiction** — pick: (a)
   `--slack BalancedByeSpacing N` to zero the window for affected grades, or (b)
   structurally exempt config-nulled forced byes from the pairwise floor. Re-run
   probe P5-equivalent → must clear INITIAL_COPY.
4. **Add `EqualMatchUpSpacing`** (the `spacing` group). If it blocks, `--slack
   EqualMatchUpSpacingConstraint N`. (Not yet shown to block, but it's the last atom
   into stage 5.)
5. **Full stage-5 run:** `scripts/e2e_real_config_solve.py` (default groups =
   `core,bye_spacing,spacing`) with the assembled `--slack` profile, real config,
   to the 30-min bar (raise the cap from the 6-min probe). Success = a feasible
   incumbent, not just REACHED_SEARCH.
6. **Record the working slack profile** in `season_2026.py` `CONSTRAINT_DEFAULTS` /
   the run recipe, and update `scripts/e2e_real_readout.md`.

> Per spec-035 DoD-6, genuine constraint-*semantic* changes are out-of-scope for the
> e2e run itself → step 2 (and 3b) should be a small filed spec under spec-038's
> ownership. Steps 1, 3a, 4, 5 are loosening/operational and can run under this
> handoff.

---

## 6. ARTIFACTS / HOW TO REPRODUCE

- **Bisection harness:** `scripts/bisect_realconfig_feasibility.py` (real config).
  `--probe` runs one solve; default runs all 6 probes + writes the trace.
- **Trace doc:** `scripts/realconfig_infeasibility_trace.md` (auto-generated table).
- **5-stage e2e launcher:** `scripts/e2e_real_config_solve.py` (`--groups`,
  `--workers`, `--minutes`, `--only N`); readout `scripts/e2e_real_readout.md`.
- **Per-probe logs:** `logs/solver_*_realbisect_P{0..5}_*.log` (CP-SAT) +
  `logs/realbisect_P{0..5}_*.stdout.log`.
- **season_test analogue + prior root-cause:** `scripts/bisect_core_feasibility.py`,
  `docs/todo/spec-035-e2e-infeasibility-handoff.md`.
- **Venv:** only `draw\.venv`; scripts resolve their own repo root, so run that
  python against this worktree's scripts. Use **workers 8** (10 risks OOM).

---

## 7. STATUS

- Renumber/retitle task (separate ask): done — spec-034/035 retitled, superlatives
  stripped, numbers kept.
- This trace fills the spec-035 **DoD-6 debugging** gap for the real config
  (previously only the conclusion "infeasible at slack 0" was recorded; now the
  blocking atoms are localised with probe evidence).
- **NOT done (next owner):** steps 1–6 above. Nothing here is committed yet —
  harness scripts + this doc are untracked in the working tree.
