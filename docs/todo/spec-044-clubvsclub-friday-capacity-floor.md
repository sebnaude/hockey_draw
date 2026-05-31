<!-- status: ready -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-038 -->
<!-- owner: none -->
<!-- reviewed: adversarial Sonnet review 2026-05-31 — fixes applied inline -->

# spec-044 — ClubVsClubStackedWeekends: make the PHL Sunday FLOOR umbrella-Friday-aware

> Fixes the real-2026 `core` infeasibility that the `spec-035` stage-5 handoff
> left OPEN. The blocker is **ClubVsClubStackedWeekends Layer 2** (the per-team-pair
> Sunday-meeting floor), NOT Layer 5/cardinality (handoff §3b guess) and NOT the
> Layer-6 nested chain. Exact-isolation evidence below.

## Why

On `load_season_data(2026)`, `core` is INFEASIBLE in presolve (P0:
`linear + amo: infeasible linear constraint` ×221). Bisection localised it to the
alignment atom (P1/P4 reach search only when it is dropped). This spec pins the
exact cause and fixes it.

**Mechanism (proven — solver-free counting + an empirical layer bisect):** PHL has
T=6, R=20 → `base = 4`, so `EnsureEqualGamesAndBalanceMatchUps` forces every PHL
pair to meet in `[4, 5]`. The atom's per-team-pair Sunday FLOOR (`tp_min`, Layer 2,
from `team_pair_sunday_meetings_range`) subtracts only **pair-named** forced Fridays,
because `phl_forced_friday_meetings()` deliberately under-counts umbrella scopes
(`_phl_forced_friday_helper.py` module docstring lines 56-61; `_entry_targets_pair_phl_friday` at line 186 enforces the both-clubs rule). But the bulk of the away clubs'
forced Fridays come from **umbrella `FORCED_GAMES` scopes that name no opponent**:

- `gosford_friday_games = 8` — `{grade:'PHL', field_location:'Central Coast Hockey
  Park', day:'Friday'}`, `sum == 8` (all 8 involve Gosford via the home-venue filter).
- `maitland_friday_games = 2` — the Maitland Park Friday `sum == 2`.

`EqualGames` still forces those games onto Fridays, cutting the club's Sunday
capacity, yet the atom subtracts them from **no pair's** floor. Result:

| club | Σ tp_min (atom Sunday floor) | forced Fri (min) | Sunday capacity R−Fri | verdict |
|------|------------------------------|------------------|-----------------------|---------|
| **Gosford** | 19 (3 vs Norths + 4×4) | 8 (CCHP umbrella) | **12** | 19 > 12 → INFEASIBLE |
| **Maitland** | 19 (3 vs Souths + 4×4) | 2 (Maitland umbrella) | **18** | 19 > 18 → INFEASIBLE |
| Norths | 18 | 0 | 20 | ok |
| Souths | 19 | 0 | 20 | ok |
| Tigers | 19 | 0 | 20 | ok |
| Wests | 18 | 0 | 20 | ok |

(review fix — M1: "Norths/Souths/Tigers/Wests 18–19" is confirmed by running the proof
script: Norths=18, Souths=19, Tigers=19, Wests=18, cap=20 for all four. The compact
"18-19 ok" claim was correct in substance. The table now makes each value explicit.
"Maitland 19" row now matches: 4 pairs at floor=4 + 1 pair (Souths) at floor=3 = 19.)

The cost of not fixing: stage-5 (full atom set) can never reach a feasible incumbent
on the real config; the published-draw pipeline stays blocked on this atom. `--slack`
cannot reach it (the atom lost its `slack_key` in spec-033). ClubGameSpread is a red
herring — excluding it does not touch this; the conflict is fundamentals + Layer 2.

**Isolation evidence (diagnostics already in the worktree under `scripts/`):**
- `scripts/isolate_clubvsclub_infeasibility.py reproduce` → fundamentals-only OPTIMAL;
  fundamentals + alignment capped at `max_layer=2` → **INFEASIBLE** (358 constraints).
  So Layers 1+2 alone are the blocker. Its `analytical` mode finds **0** Layer-6
  prefix-intersection contradictions, ruling out the nested chain independently.
- `scripts/gosford_sunday_capacity_proof.py` → the solver-free counting table above.

(review fix — L1: plan said "untracked" but both scripts exist and are tracked in the repo
under `scripts/`. Updated wording. The scripts are real and the modes/commands are correct.)

## The fix (design)

Subtract a club's **umbrella-forced PHL Fridays** from the **LOWER bound only** of the
two spec-038 range helpers; leave the UPPER bound untouched (a pair *could* still play
`base+1` Sundays if the club's umbrella Fridays all land on other opponents — so the
ceiling stays sound). For a PHL pair `(A, B)`:

```
umb        = max(club_umbrella_forced_friday_meetings(data, A),
                 club_umbrella_forced_friday_meetings(data, B))   # more-constrained club drives the floor
pair_named = phl_forced_friday_meetings(data, A, B)               # unchanged
tp_min     = max(0, base       - pair_named - umb)                # LOWER bound: now umbrella-aware
tp_max     = max(0, base + 1   - pair_named)                      # UPPER bound: unchanged
```

Layer 5's aggregate `min_budget` (`pair_grade_sunday_aligned_weekend_range`) gets the
same `- umb` on its lower bound. The **current** code at lines 406-410 subtracts
`forced_fri` (pair-named) from **both** bounds; the fix changes this to:

```
min_budget = max(0, max_ab * base     - forced_fri - umb)   # umb on lower only
max_budget = max(0, max_ab * (base+1) - forced_fri)         # max_budget unchanged
```

For PHL `a=b=1` (max_ab=1), `min_budget` reduces to the same expression as `tp_min`.
Non-PHL grades have no Fridays → `umb = 0` → unchanged.

(review fix — H1: the original vague formula `max(0, max(a,b)*base - pair_named*... -
max(a,b)*umb_per…)` was ambiguous and inconsistent with the current code structure.
Replaced with an explicit two-line diff that matches the actual function body. The old
formula suggested scaling umb by max(a,b) per-team which is wrong; umb is a club-level
count and for 1×1 PHL it happens to equal the correct result, but the code subtracts the
scalar umb from min_budget directly, not scaled. This matters conceptually even though it
doesn't affect the 1×1 PHL production case.)

Worked check after fix (verified by running `gosford_sunday_capacity_proof.py` logic
against real config):
- Gosford pairs: 4 pairs have `pair_named=0`, 1 pair (vs Norths) has `pair_named=1`.
  `tp_min = max(0, 4−{0|1}−8) = 0` for all 5 → Σ = 0 ≤ 12 ✓.
- Maitland pairs: 4 pairs have `pair_named=0` → `max(0,4−0−2)=2`; 1 pair (vs Souths,
  NIHC pair entry) has `pair_named=1` → `max(0,4−1−2)=1`. Σ = 4×2 + 1×1 = 9 ≤ 18 ✓.
- Norths, Souths, Tigers, Wests: `umb=0` → ranges unchanged, all ok.

The atom no longer over-demands away-club Sunday capacity, while `tp_max` and the
Layer-5 ceiling keep the alignment ceiling intact.

The new helper `club_umbrella_forced_friday_meetings(data, club)` counts, per club, the
**max forced count per away-venue** over Friday-PHL forced entries that have
`constraint == 'equal'` and do NOT name both clubs (umbrella scopes), attributing each
away-venue umbrella to its home club (CCHP→Gosford, Maitland Park→Maitland; NIHC
umbrellas name no single home club → skipped, keeping the count a sound lower bound).
Max-per-venue de-dupes subset scopes: the 7 specific CCHP date entries (count=1 each)
are dominated by the `sum==8` CCHP umbrella; the Norths–Gosford pair entry names both
clubs so is excluded by the "does not name both clubs" check. `lesse`-constraint entries
(e.g. the NIHC `max 3` cap) are skipped because they do not guarantee a hard floor. This
is exactly the deduped logic already validated in `scripts/gosford_sunday_capacity_proof.py`.

(review fix — M2: added the `constraint == 'equal'` filter and `lesse`-skipping rationale
to the helper description. The proof script already implements this correctly but the plan
description omitted it, leaving the implementer uncertain whether to skip lesse entries.
Also clarified that the Norths-Gosford pair entry is excluded by the "both clubs" check —
the original text was ambiguous about whether it counts as a "subset" or "pair entry".)

## Definition of Done

1. New helper `club_umbrella_forced_friday_meetings(data, club)` exists in
   `constraints/atoms/_phl_forced_friday_helper.py`, exported in `__all__`, with a
   docstring stating the max-per-away-venue, home-club-attribution, NIHC-skipped rule.
2. `team_pair_sunday_meetings_range` subtracts `umb` (more-constrained-club umbrella
   count) from `tp_min` only; `tp_max` unchanged; both clamped ≥ 0.
3. `pair_grade_sunday_aligned_weekend_range` subtracts the umbrella term from
   `min_budget` only; `max_budget` unchanged; both clamped ≥ 0.
4. Non-PHL grades are provably unaffected (helper returns 0; ranges identical to pre-fix).
   A regression test asserts a non-PHL pair's range is byte-identical before/after.
5. `scripts/gosford_sunday_capacity_proof.py` shows every PHL club `ok` (Σ tp_min ≤
   Sunday capacity) after the fix.
6. `scripts/isolate_clubvsclub_infeasibility.py reproduce` shows
   `fundamentals + alignment(max_layer=2)` flips from INFEASIBLE to OPTIMAL/FEASIBLE,
   and `max_layer=5` likewise no longer INFEASIBLE.
7. The existing PHL-preservation oracle test in
   `tests/atoms/test_club_vs_club_stacked_weekends.py` and the Friday-budget regression
   in `tests/atoms/test_cvc_stacked_friday_aware.py` still pass (update expected values
   ONLY where the new floor legitimately changes them, with a new hand-computed oracle
   and a comment explaining the change).
8. **Quick infeasibility re-probe (the 30s runs that flagged this).** Both real-config
   `core` probes that previously died at presolve now REACH_SEARCH at **8 workers, 30s cap**:
   (a) `core` (WITH ClubGameSpread) and (b) `core --exclude ClubGameSpread`. Verdict must
   flip INFEASIBLE_PRESOLVE → REACHED_SEARCH for **both** — i.e. the atom is no longer the
   `core` blocker, and ClubGameSpread is confirmed irrelevant to it. Logs captured. (Scope is
   deliberately `core` only — `bye_spacing`/`spacing` are knowingly excluded here since those
   spacing atoms still need their own loosening; see Out of scope.)
9. **Genuine 5-minute solves (the runs that don't die at presolve).** With the fix in, run
   the real solver at **8 workers, 300s (5 min) cap** for the same two groups —
   `core` and `core --exclude ClubGameSpread`. Each must (i) get past presolve into search
   and (ii) record its best objective / incumbent status. A feasible incumbent (`best ≠ -inf`)
   is the success signal and is expected for a `core`-only solve (the alignment atom was the
   sole `core` blocker); a REACHED_SEARCH with `best:-inf` at 5 min is acceptable evidence the
   presolve infeasibility is gone but must be recorded as such (NOT silently passed). Both
   runs' CP-SAT logs captured under `logs/`.
10. Type-check clean; changed-file lint clean; new tests are no-mock Given/When/Then with
    hand-computed oracles; ≥85% coverage on the new/changed helper code.

### Solver re-run verification (commands)
Run on the merged source branch after both units land (Execution protocol step 5). Venv:
`C:\Users\c3205\Documents\Code\python\draw\.venv\Scripts\python.exe`.

```powershell
# (8) Quick 30s re-probes — both must report REACHED_SEARCH (were INFEASIBLE pre-fix)
$py = "C:\Users\c3205\Documents\Code\python\draw\.venv\Scripts\python.exe"
& $py scripts\bisect_realconfig_feasibility.py --probe --groups core --max-time 30 --workers 8 --run-id s044_core30
& $py scripts\bisect_realconfig_feasibility.py --probe --groups core --exclude ClubGameSpread --max-time 30 --workers 8 --run-id s044_coreNoCGS30

# (9) Genuine 5-min solves at 8 workers — capture best objective / incumbent
& $py scripts\e2e_real_config_solve.py --groups core --minutes 5 --workers 8
& $py scripts\e2e_real_config_solve.py --groups core --exclude ClubGameSpread --minutes 5 --workers 8
# (fallback if the e2e launcher's fixed-profile wiring doesn't take the group args:
#  bisect_realconfig_feasibility.py --probe ... --max-time 300 --workers 8 gives the same real solve + verdict)
```
Record each verdict + best objective in `scripts/e2e_real_readout.md` (the existing readout).

## Implementation units

### Unit A — umbrella-Friday-aware floor (production + helper tests)
- **Files:** `constraints/atoms/_phl_forced_friday_helper.py` (new helper + `__all__` +
  docstring), `constraints/atoms/_club_vs_club_stacked_shared.py`
  (`team_pair_sunday_meetings_range`, `pair_grade_sunday_aligned_weekend_range`),
  `tests/atoms/test_club_vs_club_stacked_shared.py` (helper + range oracle tests).
- **Change summary:** implement the fix design above. The two range helpers call the new
  helper for the `umb` term. No change to `phl_forced_friday_meetings`,
  `pair_grade_sunday_meetings`, or `pair_grade_sunday_aligned_weekends` (the scalar
  helpers) — only the spec-038 `_range` variants.
- **Dependency:** none within this plan (it is the root unit).
- **No-mock test outline (hand oracles, real fixtures via `build_stacked_fixture`):**
  - *Given* a fixture with a Gosford-style club carrying an 8-count CCHP Friday umbrella
    (`constraint='equal'`, no teams) + a 1-count Norths-Gosford pair entry at CCHP
    (teams named), + 7 date-specific CCHP entries (count=1, no teams each), *When*
    `club_umbrella_forced_friday_meetings(data, 'Gosford')`, *Then* `== 8` (max-per-venue,
    pair entry excluded by both-clubs check; 7 date entries yield max(1,…)=1 < 8, dominated
    by umbrella). Oracle: max(8, 1, 1, 1, 1, 1, 1, 1) = 8.

  (review fix — M3: original oracle said "max(8, 1×specific)=8" which was imprecise about
  why the 7 date-entries also return 1 each and why the pair entry is excluded. Expanded
  to match the real config structure for a more verifiable fixture.)
  - *Given* the same, *When* `team_pair_sunday_meetings_range(data, ('Gosford','Souths'),
    'PHL')` with base=4, *Then* `(0, 5)`. Oracle: `tp_min=max(0,4−0−8)=0`, `tp_max=4+1−0=5`.
  - *Given* `('Gosford','Norths')` (pair_named=1), *Then* `(0, 4)`. Oracle: `max(0,4−1−8)=0`, `5−1=4`.
  - *Given* a club with NO forced Fridays (`umb=0`), *Then* range unchanged `(4, 5)`.
  - *Given* a non-PHL pair, *Then* range identical before/after (helper returns 0 for non-PHL).
- **Executor:** Opus (subtle: forced-scope attribution + asymmetric bound math; being wrong reintroduces the infeasibility).

### Unit B — integration regression + reproduce-flip proof
- **Files:** `tests/atoms/test_club_vs_club_stacked_weekends.py` (extend with a real-config
  away-club regression), and adjust any now-stale expected values there and in
  `tests/atoms/test_cvc_stacked_friday_aware.py` with fresh oracles + explanatory comments.
- **Change summary:** add a Given/When/Then test that builds `load_season_data(2026)`,
  applies fundamentals + the alignment atom (mirroring
  `isolate_clubvsclub_infeasibility.py reproduce` at `max_layer in {2,5,6}`) and asserts
  the model is now feasible (`status in {OPTIMAL, FEASIBLE}`). Hand oracle for helper
  values (BEFORE the solver runs): Gosford Σ tp_min = 0 (all 5 pairs → 0 each);
  Maitland Σ tp_min = 9 (4 pairs → 2 each, vs Souths → 1).
  `tests/atoms/test_club_vs_club_stacked_shared.py` also needs new helper-oracle tests
  for `club_umbrella_forced_friday_meetings` (DoD-1/Unit A test requirement but Unit B
  must update the file). See Unit A test outline for the exact G/W/T cases.

  (review fix — H2: original text only said "Hand oracle: Gosford Σ tp_min == 0 ≤ 12,
  Maitland Σ tp_min computed and ≤ 18" — "computed" is not a hand oracle and doesn't help
  the implementer verify correctness. Added the actual Maitland figure of 9. Also clarified
  that test_club_vs_club_stacked_shared.py needs the new helper oracle tests from Unit A.)
- **Dependency:** `depends_on: Unit A` (same atom behaviour; tests the post-fix state).
- **Solver re-runs (DoD-8 + DoD-9) are this unit's post-merge acceptance evidence.** After
  Unit B merges, run the four solver invocations in the "Solver re-run verification (commands)"
  block — the 30s quick re-probes (both `core` and `core --exclude ClubGameSpread`, must flip to
  REACHED_SEARCH) and the 5-min genuine solves (8 workers, capture best objective) — and record
  the verdicts/objectives in `scripts/e2e_real_readout.md`. These are run-and-record evidence
  steps, not new code.
- **Executor:** Sonnet (mechanical once Unit A lands), but escalate to Opus if the real-config
  build in-test proves fiddly.

## Doc registry
- `constraints/atoms/_phl_forced_friday_helper.py` — module docstring (section 1 "Per-pair
  PHL Friday meetings"): add the new helper's contract (umbrella, max-per-venue, equal-only,
  home-club attribution, NIHC-skipped → sound lower bound); update `__all__` to export
  `club_umbrella_forced_friday_meetings`.
- `constraints/atoms/_club_vs_club_stacked_shared.py` — docstrings of the two `_range`
  helpers (`team_pair_sunday_meetings_range`, `pair_grade_sunday_aligned_weekend_range`):
  note the umbrella-Friday subtraction on the LOWER bound only and why the ceiling is
  left intact; update module docstring list of building blocks to mention the new helper.
- `constraints/atoms/club_vs_club_stacked_weekends.py` — module docstring (Layer 2/5
  paragraphs): record that PHL Sunday floors are away-venue-umbrella-aware.
- `docs/todo/spec-035-realconfig-stage5-handoff.md` — update §0/§3b: the real `core` blocker
  was Layer 2 capacity (not Layer 5); handoff Step 2 is fulfilled by spec-044.
- `scripts/e2e_real_readout.md` — append the DoD-8/DoD-9 re-run verdicts + best objectives
  (30s probes + 5-min genuine solves, both `core` and `core --exclude ClubGameSpread`).
- `docs/todo/00-dependency-tree.md` — spec-044 node already present (`review_pending`);
  update status to `done` and note stage-5 still awaits BalancedByeSpacing fix.

(review fix — M4: original said "add spec-044 node" but the node already exists in the
dep tree (confirmed in the file). The real action is updating its status. Also added
`__all__` as an explicit registry update target — DoD-1 requires it but Doc Registry
didn't list it. Added module-docstring building-block list update for the shared module.)

## Out of scope
- **BalancedByeSpacing forced-bye contradiction** (handoff §3a, P5 `2x≤1`): a *separate,
  independent* stage-5 blocker in a different atom, already documented and tracked in
  `docs/todo/spec-035-realconfig-stage5-handoff.md`. Not introduced or touched here; this
  plan's DoD-8 deliberately scopes to `core`, not full stage-5.
- **Re-adding a `slack_key` to the atom** (removed in spec-033). The capacity fix makes the
  atom feasible at slack 0, so no slack mechanism is needed for this blocker; adding one is a
  distinct concern, not done here.
- **EqualMatchUpSpacing** (the `spacing` group) — handoff Step 4; only relevant once core +
  bye_spacing are clear. Not this plan.

## Dependencies
- `depends_on: spec-038` — spec-038 (`status: building` in its plan header, but all code
  units already merged to `final-form` as of 2026-05-28 commit `2ebe31e`; the `building`
  status reflects incomplete plan-closeout not incomplete code). In practice the files are
  in their post-spec-038 state NOW. The executor must re-read both helper files before
  editing (as always), and must confirm the spec-038 plan is marked `done` before starting
  this work — or obtain explicit user authorisation to proceed if spec-038 closeout is
  merely administrative. Unit B `depends_on` Unit A within this plan.

(review fix — H3: original said spec-038 is "building" without noting that its code is
fully merged. A future executor reading this plan could misidentify spec-038 as having
unmerged code changes that would conflict. Clarified the actual state: code merged, plan
closeout pending. Also added the "re-read before editing" reminder from the Risks section
to the Dependencies section where the executor looks.)

## Risks & blast radius
- **Over-loosening the away-club floor.** The fix can drive Gosford pair floors to 0, so the
  atom stops *forcing* any Sunday stacking for Gosford (it plays mostly Fridays anyway). The
  `tp_max` ceiling and Layer-5 max still bound it, and the solver may still stack — but the
  hard floor is gone for that club. Awareness item; surfaced for the convenor, not insured
  against. Touches: alignment quality for Gosford/Maitland Sundays only.
- **min_budget > max_budget after fix?** Cannot happen: `min_budget` loses `umb` but
  `max_budget` retains `base+1` over `base`, so `max_budget >= min_budget` in all cases
  (formally: `max(0, max_ab*base - f - umb) <= max(0, max_ab*(base+1) - f)` since
  `base <= base+1` and `umb >= 0`). Same argument holds for `tp_min <= tp_max`.
  Verified analytically; no IntVar/bound violation is possible.
- **Shared-file race with spec-038.** Both touch the two helper files. Mitigated by the
  `depends_on: spec-038` gate; re-read both files before editing.
- **Stale oracle drift.** Existing atom tests may encode the pre-fix floor for away clubs;
  Unit B must re-derive those oracles by hand, never by reading the new code output. Note
  that `test_cvc_stacked_friday_aware.py` and `test_club_vs_club_stacked_weekends.py` do
  NOT currently encode away-club umbrella floors (they use pair-named entries only), so
  their existing tests are unlikely to break — but the executor MUST verify before assuming.

(review fix — M5: added explicit bound-safety analysis confirming min <= max after fix,
answering the "no IntVar/bound violation" claim from DoD description. Added note on which
existing tests are at risk to help the executor scope the oracle-update work.)

## Open Questions
None. (The away-club floor-loosening tradeoff is recorded as a Risk, not a blocking decision —
the ceiling + Layer-5 max preserve the alignment intent that is achievable given forced Fridays.)

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->
0. **Do NOT start without an explicit user instruction to implement this plan.** `ready` = authorised-when-asked, not build-now. If you arrived straight off authoring/review with no go-ahead, STOP and ask.
1. Status must be `ready` (carries a `reviewed:` stamp). If `review_pending`/`under_review`, let review finish. If `blocked`, STOP — Open Questions need the user. **If spec-038 code is not yet fully merged to `final-form`, STOP — not startable.** (As of 2026-05-31, spec-038 code is merged; the plan header still says `building` but all code units are landed. Confirm with `git log --oneline -- constraints/atoms/_club_vs_club_stacked_shared.py` before starting.)
2. Only after the user says to implement: stamp `building`, claim `owner`. You are the orchestrator (Opus).
3. Unit A first (own worktree+branch, Opus). Then Unit B (own worktree+branch; `depends_on` Unit A landed). Run the S2 gate set: type-check, changed-file lint, AST dead-code/dark-path sweep, no-mock G/W/T tests with hand oracles, ≥85% on changed code.
4. After each unit, launch `/adversarial` Mode B to verify the diff against this DoD (it re-derives the oracles and re-runs the gates). Route fixes, re-verify. NEVER merge unverified.
5. Merge → push origin → post-merge verify on the merged branch: (a) `scripts/gosford_sunday_capacity_proof.py` (DoD-5) + `scripts/isolate_clubvsclub_infeasibility.py reproduce` (DoD-6); (b) the **Solver re-run verification (commands)** block — the two 30s quick re-probes `core` and `core --exclude ClubGameSpread` must flip to REACHED_SEARCH (DoD-8), then the two 5-min @8-worker genuine solves, recording verdict + best objective in `scripts/e2e_real_readout.md` (DoD-9). → remove worktree. Tick the unit's checkbox.
6. When both units pass: stamp `done`, update `docs/todo/00-dependency-tree.md` (drop the spec-044 edge; note stage-5 still awaits the BalancedByeSpacing fix).
