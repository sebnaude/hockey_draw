<!-- status: handoff -->
<!-- spec: spec-035 (ULTIMATE e2e) — execution blocked on a model-infeasibility discovery -->
<!-- author: opus session 2026-05-26/27 -->

# spec-035 e2e — INFEASIBILITY HANDOFF

> **Read this as evidence, not conclusions.** The "Interpretation" sections are
> one agent's reading and the convenor has said that reading is likely wrong.
> The "Evidence" sections are reproducible facts (probe commands + verdicts).
> Trust the evidence; re-derive the interpretation.

---

## 1. What spec-035 is trying to do

spec-035 (`docs/todo/spec-035-ultimate-e2e-core-run.md`, status `in_progress`,
owned this session) is the ULTIMATE plan: a raw end-to-end CP-SAT solve of the
`core` constraint group (minus `ClubGameSpread` for one of the two runs) on the
**forced-free `season_test` config** (`config/season_test.py` — 2026 base teams +
field availability, but `forced_games == []`, `blocked_games == []`,
`locked_pairings == []`). Goals: (1) get through presolve + survive 30 min of
search ("liveness"); (2) read out CP-SAT's remaining `[Symmetry]`; (3) compare
full-core vs core−ClubGameSpread symmetry. Week 1 is NOT fixed; `--workers 10`.

**The run never reaches search.** The model is proven INFEASIBLE during CP-SAT's
*initial constraint copy* (~2s), so goals 1–3 are all blocked. This handoff is
about that infeasibility.

---

## 2. Branch / worktree state (all local; nothing pushed to origin)

| Worktree | Branch | Head | Contents |
|---|---|---|---|
| `draw-final-form` | `final-form` | `bfaadb8` | Integration branch. Has Units A+B merged + the bisect harness. |
| `draw-s035-A` | `spec035-unitA` | `112ca1c` | Unit A source (already merged to final-form). Can be torn down. |
| `draw-s035-B` | `spec035-unitB` | `06d0a4f` | Unit B source (already merged to final-form). Can be torn down. |
| `draw-s035-flense` | `spec035-flense` | `b6b7186` | **WIP** — the `AwayClubHomeWeekendsCount` flense (see §6). NOT merged. |

**Merged to `final-form` (both /adversarial Mode B verified):**
- **Unit A** (`d7356d4`,`112ca1c` → merge `81231bd`): `scripts/run_core_e2e.py` —
  launcher that drives `main_staged.main_simple(year='test', …)` with the raw
  core profile (`groups=['core']`, `workers=10`, `fix_round_1=False`, no locks,
  optional `exclude`). Writes a profile sidecar to `logs/core_e2e_profile_<run_id>.json`.
- **Unit B** (`06d0a4f` → merge `9a845b1`): CP-SAT log capture. `attach_cpsat_log_capture`
  + `parse_symmetry_stats` in `solver_diagnostics.py`; wired at `main_staged.py`
  `_main_simple_unified` (the single-solve path Unit C uses) and `solve_stage`.
  Confirmed working — `[Symmetry]` lands in the run log on a solve that reaches presolve.
- **M1/L1 review fixes** (`832e66e`): launcher no longer double-inits logging →
  exactly one `logs/solver_*_<run_id>.log` per run; sidecar carries `solve_log_glob`.
- **Bisect harness** (`bfaadb8`): `scripts/bisect_core_feasibility.py` (see §4).

Post-merge test state: full batched suite (`scripts/run_green_suite.py --no-cov`)
green except batch 1, which hits the **known nondeterministic ortools/Windows heap
-corruption segfault** (`0xc0000374`) — every file in batch 1 passes when run in
smaller chunks. Not a regression (pre-flight had the same).

---

## 3. EVIDENCE — the raw symptom

Running the launcher (or the bisect harness with no extra excludes) on forced-free
`season_test` with `--groups core --exclude ClubGameSpread`:

```
#Variables: 143'242 (#bools: 100'987 ...)   <- model BUILDS fine
Starting presolve at 2.02s
INFEASIBLE: 'proven during initial copy of constraint #124504:
Problem proven infeasible during initial copy.
status: INFEASIBLE
```

So: model construction succeeds (143k vars), then CP-SAT proves infeasibility
during the initial copy at constraint #124504, in ~2 seconds, before presolve
symmetry detection or search. (`#124504` is a CP-SAT-internal constraint index,
not an atom name — it does not directly map to a named atom.)

Full core (WITH ClubGameSpread) shows the identical signature.

`season_test` week structure (relevant to the count targets below): **22 distinct
Sunday weeks, 21 distinct Friday weeks**; PHL `num_rounds` = 20.

---

## 4. EVIDENCE — the bisection harness

`scripts/bisect_core_feasibility.py` (committed `bfaadb8`). Runs `main_simple`
against `season_test` with an arbitrary `--exclude` set and a short `--max-time`
cap, then classifies the run log:
- `INFEASIBLE_PRESOLVE` — "proven infeasible" / "status: INFEASIBLE" in the log.
- `REACHED_SEARCH` — got past presolve into search (search/bound/objective lines).
- `FEASIBLE/OPTIMAL` — found a solution within the cap.

`ClubGameSpread` is always excluded by the harness (the spec-035 target is
`core − ClubGameSpread`). Run it with the venv python from whichever worktree's
code you want to test:

```
C:\Users\c3205\Documents\Code\python\draw\.venv\Scripts\python.exe \
  <worktree>\scripts\bisect_core_feasibility.py --max-time 120 --workers 10 --exclude <names...>
```

Infeasibility is proven in ~2s, so probes are fast; a feasible chunk runs to the cap.

---

## 5. EVIDENCE — bisection results

The 20 constraints in `core − ClubGameSpread` (3 are soft symmetry-breakers:
`SoftLexMatchupOrdering`, `NIHCFillEFBeforeSF`, `NIHCFillWFBeforeEF`). Probes are
stated as "kept on top of the `critical_feasibility` fundamentals". Fundamentals =
`NoDoubleBookingTeams/Fields`, `EqualGamesAndBalanceMatchUps`,
`PHLConcurrencyAtBroadmeadow`, `SameGradeSameClubNoConcurrency`,
`ClubNoConcurrentSlot`, `PHLAnd2ndAdjacency`, `VenueEarliestSlotFill`.

| # | Probe (kept atoms, on top of fundamentals) | Verdict |
|---|---|---|
| 1 | fundamentals only | ✅ REACHED_SEARCH |
| 2a | + all `club_day` (5) | ✅ REACHED_SEARCH |
| 2b | + `club_alignment`(2) + `home_away`(2), club_day excluded | ❌ INFEASIBLE |
| 3a | + `home_away`(2) only [AwayClubHomeWeekendsCount + AwayClubPerOpponentAndAggregateHomeBalance] | ❌ INFEASIBLE |
| 3b | + `club_alignment`(2) only [ClubVsClubStackedWeekends + StackedCoLocation] | ❌ INFEASIBLE |
| 4c | + `AwayClubHomeWeekendsCount` ALONE | ❌ INFEASIBLE |
| 4d | + `AwayClubPerOpponentAndAggregateHomeBalance` ALONE | ✅ REACHED_SEARCH |
| 4b | + `ClubVsClubStackedWeekends` ALONE | ❌ INFEASIBLE |
| — | target set MINUS {AwayClubHomeWeekendsCount, ClubVsClubStackedWeekends, StackedCoLocation} | ✅ REACHED_SEARCH |

Notes:
- `ClubVsClubStackedCoLocation` CANNOT run without `ClubVsClubStackedWeekends`
  (helper-var producer/consumer: CoLocation reads `play[pair,grade,week]`
  indicators that StackedWeekends registers). RuntimeError if you exclude only
  StackedWeekends. So that pair is tested together / excluded together.

**Conclusion from evidence (not disputed):** the *complete* set of presolve-
infeasibility sources in `core − ClubGameSpread` is exactly **two hard atoms**:
`AwayClubHomeWeekendsCount` and `ClubVsClubStackedWeekends`. Remove both (+ the
CoLocation that depends on the latter) and the remaining 17 atoms reach search.

---

## 6. EVIDENCE — culprit #1: `AwayClubHomeWeekendsCount`

File: `constraints/atoms/away_club_home_weekends_count.py`. Helper:
`constraints/atoms/_phl_forced_friday_helper.py`. Regen-soft twin (NOT touched):
`constraints/atoms/away_club_home_weekends_count_regen_soft.py`.

**As-was (head `bfaadb8`):** for each away-based club (Maitland@Maitland Park,
Gosford@Central Coast — from `home_field_map`) it added THREE hard equalities:
1. `sum(friday_home_indicators) == phl_forced_friday_count(data, club)`
2. `sum(sunday_home_indicators) == away_club_required_sundays(data, club)`
   `= max(PHL_required − forced_fridays, max_other_grade)`
3. `sum(all_home_indicators)    == away_club_total_weekends(data, club)`
   `= max(PHL_required, max_other_grade)`

**Computed targets on forced-free season_test** (forced_games==[] ⇒
`phl_forced_friday_count == 0`): for BOTH Maitland and Gosford →
`friday_target=0, sunday_target=20, total_target=20`. I.e. the atom hard-forces
"zero home Fridays AND 20 distinct Sunday-home weekends AND 20 total".

**Flense done (committed `b6b7186` on `spec035-flense`):** dropped equalities (1)
and (2) and their helper calls (`phl_forced_friday_count`,
`away_club_required_sundays`); kept ONLY (3) `sum(all_home) == max(PHL, max_other)`
(no forced term). Removed the now-dead `_build_week_indicators` helper. Updated
docstring. Compiles clean, pyflakes clean.

**Probe results after the flense:**
- flensed `AwayClubHomeWeekendsCount` ALONE (on fundamentals) → ✅ FEASIBLE/OPTIMAL.
- flensed `AwayClubHomeWeekendsCount` + `AwayClubPerOpponentAndAggregateHomeBalance`
  (on fundamentals) → ❌ INFEASIBLE. (Each feasible alone; together infeasible.)

**Interpretation (DISPUTED — treat as a hypothesis):** I read this as the two
home/away atoms (`AwayClubHomeWeekendsCount` total-weekends `==` vs
`AwayClubPerOpponentAndAggregateHomeBalance` per-opponent + aggregate) carrying
mutually inconsistent hard requirements / overlapping responsibility. The convenor
indicates this interpretation is likely wrong, so DO NOT act on it without re-deriving.

---

## 7. EVIDENCE — culprit #2: `ClubVsClubStackedWeekends`

Files: `constraints/atoms/club_vs_club_stacked_weekends.py`,
`constraints/atoms/_club_vs_club_stacked_shared.py` (the budget math, ~line 187:
`forced_fri = phl_forced_friday_meetings(data, a, b)`), and
`constraints/atoms/club_vs_club_stacked_co_location.py` (consumer).
Regen-soft twin (NOT touched): `clubvsclub_stacked_weekends_regen_soft.py`.

**What it does:** per unordered club pair (A,B), pins each grade's Sunday meeting
count and a strict nested-superset stacking implication across weeks
(`play[g_{k+1},w] <= play[g_k,w]`, `sum_w play[g_k,w] == c[g_k]`). The PHL Sunday
budget `c[PHL] = total_phl_meetings(A,B) − phl_forced_friday_meetings(A,B)`.

**Key fact:** on forced-free season_test `phl_forced_friday_meetings == 0`, so the
budget is already just `total_phl_meetings` — i.e. removing the forced-Friday
subtraction would change NOTHING on the forced-free config. NOT yet flensed or
modified. The atom alone (on fundamentals) is INFEASIBLE (probe 4b).

**Interpretation (DISPUTED — hypothesis only):** I read the infeasibility as
structural — the hard exact-count all-Sunday stacking requires all PHL pair-
meetings on Sundays and only stays feasible in production because forced Fridays
reduce the Sunday budget. NOT verified. The convenor indicates the framing is
likely wrong. Re-derive.

---

## 8. EVIDENCE — FORCED_GAMES count rules work (verified)

`tests/test_forced_games_count_rules.py` → 4 passed. `config/season_2026.py`
`FORCED_GAMES` contains, e.g.:
```python
{"grade":"PHL","day":"Friday","field_location":"Central Coast Hockey Park",
 "count":8,"constraint":"equal","description":"Exactly 8 PHL Friday games at Gosford per season (AGM 2026)"}
```
with the in-config comment: *"Per-venue counts are budgets, not structural
constraints; they belong here in season config, not in hardcoded constraint atoms."*
This is the basis for the convenor's instruction that forced-Friday counts should
NOT be baked into the constraint atoms.

---

## 8b. The Friday-game interpretation + the proposed fix — BOTH UNVERIFIED

> This section records, verbatim, (a) the agent's interpretation of the
> Friday-game problem and (b) the convenor's instructed fix. **The convenor has
> explicitly said NEITHER may be correct.** Do not treat either as established.
> They are here so the next person can confirm or refute them, not adopt them.

### (a) The agent's interpretation of the Friday-game problem (UNVERIFIED)

Both culprit atoms compute their hard count targets by subtracting a *forced-Friday
count read from the config*:

- `AwayClubHomeWeekendsCount` pinned, per away-based club:
  - `sum(friday_home_indicators) == phl_forced_friday_count(data, club)`
  - `sum(sunday_home_indicators) == max(PHL_required − phl_forced_friday_count, max_other)`
  - `sum(all_home_indicators)    == max(PHL_required, max_other)`
- `ClubVsClubStackedWeekends` set its PHL Sunday stacking budget to
  `total_phl_meetings(A,B) − phl_forced_friday_meetings(A,B)`.

The agent's claim: on forced-free `season_test`, `forced_games == []` so
`phl_forced_friday_count == 0` and `phl_forced_friday_meetings == 0`. That makes
`AwayClubHomeWeekendsCount` hard-demand **`friday_home == 0` AND `sunday_home == 20`
AND `total == 20`** for both Maitland and Gosford simultaneously — which the agent
argued is physically unsatisfiable — and makes `ClubVsClubStackedWeekends` force
ALL PHL pair-meetings onto Sundays. **Root-cause story (agent):** *these atoms
bake production's forced-Friday structure into hard `==` constraints and therefore
go infeasible whenever the config forces nothing.*

**Why this may be WRONG (open):** the agent did NOT prove the `friday==0 +
sunday==20` combination is actually unsatisfiable from first principles (capacity /
EqualGames geometry) — it inferred it from the infeasibility verdict. It also did
NOT explain why `ClubVsClubStackedWeekends` is infeasible forced-free *given* that
the forced term is already 0 there (so "forced-Friday coupling" cannot be the
forced-free cause for #2). The real mechanism behind constraint #124504's
infeasibility was never traced to a specific contradiction.

### (b) The convenor's instructed fix (UNVERIFIED / possibly wrong)

Verbatim intent: forcing of a specific number of Friday games (e.g. "exactly 8
PHL Friday games at Gosford/Central Coast") must be expressed in `FORCED_GAMES`
as a partial-key COUNT rule — `{grade:'PHL', day:'Friday', field_location:'Central
Coast Hockey Park', count:8, constraint:'equal'}` — and the constraint atoms
should **no longer contain any forced-Friday awareness**. Instruction was:
(1) verify FORCED_GAMES does the forcing (done — §8), (2) "flense the offending
[forced-Friday] code from the constraints", (3) try again.

**What the agent did with that instruction:** flensed `AwayClubHomeWeekendsCount`
(removed equalities 1+2 and the `phl_forced_friday_count` /
`away_club_required_sundays` calls; kept only `total == max(PHL, max_other)`).
Committed `b6b7186` on `spec035-flense`.

**Why the fix may be WRONG / incomplete (open):**
- After flensing, `AwayClubHomeWeekendsCount` is feasible alone but still
  infeasible together with `AwayClubPerOpponentAndAggregateHomeBalance` — so
  removing the forced-Friday code did NOT by itself make the home/away pair
  feasible. Something else is going on between those two atoms.
- For `ClubVsClubStackedWeekends`, the forced-Friday term is already 0 on
  forced-free, so "flensing the forced-Friday code" cannot change the forced-free
  outcome at all — the instruction, taken literally, is a no-op for culprit #2.
- Whether keeping `total == max(PHL, max_other)` as a hard `==` is even the right
  surviving semantics is unconfirmed (it may need to be a bound, or the atom may
  be redundant, or the forced-free config may simply be the wrong test input for
  these atoms — none of these has been established).

**Bottom line:** the *mechanical* facts (§3–§7) are solid; the *causal story* for
why #124504 is infeasible — and therefore the correct fix — is NOT yet established
by either party. Next step should be to actually trace the specific contradiction
(e.g. dump the conflicting constraints / use an assumptions/IIS-style probe), not
to act on (a) or (b).

## 9. What is DONE vs NOT DONE

Done:
- Units A + B built, Mode B verified, merged to `final-form`; M1/L1 fixes applied.
- Bisect harness built + committed.
- Two culprits isolated definitively (§5).
- `AwayClubHomeWeekendsCount` forced-Friday coupling flensed on `spec035-flense`
  (`b6b7186`) — feasible alone, but interacts with `AwayClubPerOpponentAndAggregateHomeBalance`.
- FORCED_GAMES count mechanism verified.
- Memory written: `feedback_forcing_belongs_in_config`.

NOT done / open:
- The flense is NOT merged (WIP on `spec035-flense`); its regen-soft twin
  (`away_club_home_weekends_count_regen_soft.py`) is UNCHANGED — if the flense is
  kept, forward-only requires the twin + the now-orphaned helper functions
  (`phl_forced_friday_count`, `away_club_required_sundays`, and possibly
  `phl_forced_friday_meetings`) + tests + registry/inventory docs to be reconciled.
- `ClubVsClubStackedWeekends` is untouched and still infeasible forced-free.
- Root-cause INTERPRETATION for BOTH atoms is unresolved/disputed — the convenor
  believes the issue has been mis-read. The mechanical facts in §3–§7 stand; the
  "why" does not.
- spec-035's own DoD-2 ("raw, no slack") vs the documented "complete set is
  INFEASIBLE at slack 0" (spec-033/036 notes in `docs/todo/00-dependency-tree.md`)
  is an unreconciled contradiction in the plan that Mode A review missed.

---

## 10. How to reproduce / continue

```powershell
# Reproduce the raw infeasibility (full target set):
C:\Users\c3205\Documents\Code\python\draw\.venv\Scripts\python.exe `
  C:\Users\c3205\Documents\Code\python\draw-final-form\scripts\bisect_core_feasibility.py --max-time 60

# Any chunk (run from the worktree whose atom code you want to test, e.g. the flense):
... draw-s035-flense\scripts\bisect_core_feasibility.py --max-time 120 --workers 10 --exclude <names>

# Compute the AwayClubHomeWeekendsCount targets on season_test:
python -c "from config.season_test import get_season_data; from constraints.atoms._phl_forced_friday_helper import *; d=get_season_data(); [print(c, phl_forced_friday_count(d,c), away_club_required_sundays(d,c), away_club_total_weekends(d,c)) for c in d['home_field_map']]"
```

Venv lives ONLY in `draw\.venv`; scripts resolve their own repo root from
`__file__`, so run that python against any worktree's script.
