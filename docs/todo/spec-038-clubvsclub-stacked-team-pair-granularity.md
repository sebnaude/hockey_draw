<!-- status: building -->
<!-- reviewed: adversarial Sonnet review 2026-05-28 — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->
<!-- owner: session=2026-05-28-spec038-build claimed=2026-05-28 -->

# spec-038 — `ClubVsClubStackedWeekends` granularity: per-team-pair budget, `max(team_count) × per_matchup` aligned-weekend formula

**Spec source:** convenor + spec-035 handoff investigation (this session). Surfaced as the second
of two structurally-broken atoms in `docs/todo/spec-035-e2e-infeasibility-handoff.md` (§7).
Formula and stacking model clarified by the convenor in two rounds of this session's discussion
(see "Why" below for the precise example walkthrough).

## Why

`ClubVsClubStackedWeekends` is the second presolve-infeasibility source in the `core` group on
forced-free `season_test`. The atom is INFEASIBLE on fundamentals alone (handoff §5, probe 4b)
even with zero forced Friday meetings — so the "flense the forced-Friday code" instruction is a
no-op for it (`phl_forced_friday_meetings == 0` on forced-free). The real root cause is **wrong
budget and wrong granularity** for the play indicator.

### What it's supposed to do (convenor's intent, verbatim)

For each unordered pair of clubs `(A, B)`, and for each grade both clubs field:

- Each grade meets the other club some number of times per season (derived, not configured).
- Stack those meetings on the **same aligned weekends across grades** so a family with kids in
  4ths and 5ths sees Tigers-vs-Colts on the same Sunday for both kids. Nested-superset: the
  smaller meeting-count grades only play on weekends a larger meeting-count grade also plays.
- **Caveat 1 (PHL):** PHL forced-Friday meetings consume the pair's matchup count but cannot
  satisfy Sunday stacking, so they are subtracted from PHL's Sunday-stacking budget.
- **Caveat 2 (multi-team-per-club-per-grade):** when both clubs field 2 teams in a grade (e.g.
  Tigers Black + Tigers Yellow vs Colts Gold + Colts Green in 5ths), BOTH cross-club games for
  that grade should play on each aligned weekend. When one side fields 2 teams and the other 1,
  the model is free to pick which of the two teams plays the single opposing team on each
  aligned weekend (so the asymmetric case naturally produces more aligned weekends).

### What the code does (verified against `_club_vs_club_stacked_shared.py`)

- `per_pair_grade_meeting_counts(data, pair)[grade] = matchups × per_matchup` — **counts games**.
- `play[club_pair, grade, w] = OR over all Sunday vars of that (pair, grade, week)` — **indicates "≥ 1 game this weekend"**.
- Constraint: `sum_w play == matchups × per_matchup`.

Two compounding bugs:

1. **Wrong budget unit.** `matchups × per_matchup` is a count of *games*; `play` is a count of
   *weekends*. With multi-team-per-club-per-grade (matchups > 1 from `a × b` cross-team-pairs),
   the budget is structurally larger than the number of weekends actually needed — and the OR
   semantics means stacking two games on one weekend only contributes 1 to the sum, so the only
   way to reach the budget is to **spread games into distinct weekends** — the OPPOSITE of the
   convenor's "stack on the aligned weekend" intent.
2. **Wrong granularity.** Even with the right budget, OR-over-all-team-pairs doesn't enforce that
   on an aligned weekend, **all** the in-grade cross-team matchups play together. The solver can
   satisfy `sum_w play == N` by putting one matchup per weekend across N weekends and leaving
   the other matchups to find other weekends — breaking alignment.

### The corrected formula (this spec)

For each `(club_pair, grade)`, define:

- `a = teams_in_grade[club_A_in_grade]`, `b = teams_in_grade[club_B_in_grade]` (typically 1 or 2).
- `per_matchup = num_rounds[grade] // (T-1) if T%2==0 else num_rounds[grade] // T` (existing formula in `_club_vs_club_stacked_shared.per_pair_grade_meeting_counts`; T = grade's `num_teams`).
- `weekends_budget = max(a, b) × per_matchup` — number of distinct aligned weekends for this `(pair, grade)`.
- `games_per_aligned_weekend = min(a, b)` — how many in-grade cross-team games happen on each aligned weekend.
- Total games for the pair-grade = `a × b × per_matchup = max(a,b) × min(a,b) × per_matchup = weekends_budget × games_per_aligned_weekend`. ✓ (consistent.)

Case table (sanity check):

| Layout | `a × b` | matchups | weekends (`max·per_matchup`) | games/weekend (`min`) | total games |
|---|---|---|---|---|---|
| 1 vs 1, per_matchup=1 (e.g. lower-grade single team each) | 1×1 | 1 | 1 | 1 | 1 |
| 1 vs 1, per_matchup=2 | 1×1 | 1 | 2 | 1 | 2 |
| 2 vs 2, per_matchup=1 (synthetic — see fixture below; the real season_test 2×2 pair is Tigers-University 6th at per_matchup=2, not 5th — Tigers and University each field 1 team in 5th) | 2×2 | 4 | 2 | 2 | 4 |
| 2 vs 2, per_matchup=2 | 2×2 | 4 | 4 | 2 | 8 |
| 1 vs 2, per_matchup=1 (single team plays each of 2 opposing teams) | 1×2 | 2 | 2 | 1 | 2 |
| 1 vs 2, per_matchup=2 | 1×2 | 2 | 4 | 1 | 4 |
| PHL (always 1×1 in the current league) | 1×1 | 1 | per_matchup | 1 | per_matchup |

The PHL row is structurally identical to the old formula (since `a=b=1` → `max·per_matchup = matchups·per_matchup`), so the corrected formula is BACKWARD-COMPATIBLE for PHL pairs and only changes behaviour for multi-team-per-club lower grades.

### Why the convenor's "free to choose" comment works for free

In the asymmetric 1×2 case (e.g. Maitland fields 1 team in 4th, University fields 2 teams in 4th), the model has `a=1, b=2, weekends=2, games/weekend=1`. On each of the 2 aligned weekends, the single Maitland 4th team plays ONE of the two University 4th teams. Which University team plays on which weekend is a free variable — no extra constraint needed; symmetry breaking elsewhere or the solver's tie-break chooses. The convenor's "free to choose" intent is the natural model behaviour without explicit rules.

## Definition of Done

### Semantic model

1. **Per-team-pair play indicators (new).** For each `(team_pair, week)` where `team_pair` is an unordered (team_A, team_B) with team_A in club_A_in_grade and team_B in club_B_in_grade:
   - `team_pair_play[tp, w] = OR over all Sunday X-vars with (team1, team2) == tp and week == w`.
   - Registered under helper-var key `(STACK_TEAM_PAIR_PLAY_PREFIX, tp, w)` (new prefix `cvc_stack_team_pair_play`).
   - Fixed to 0 if no Sunday vars exist for `(tp, w)`.
2. **Per-team-pair budget (new hard constraint).** For each team_pair in a `(club_pair, grade)`:
   - `sum_w team_pair_play[tp, w] == per_matchup`.
3. **Per-pair-grade aligned-weekend indicator.** `play_pg[club_pair, grade, w] = OR over team_pair_play[tp, w] for all tp in (club_pair, grade)`. Registered under EXISTING `STACK_PLAY_PREFIX` key shape `(club_pair, grade, w)` — **semantics changed (now an OR over team-pair indicators, not over raw vars), key shape unchanged**, so the `ClubVsClubStackedCoLocation` consumer continues to read it without internal modification.
4. **Per-aligned-weekend cardinality (new hard constraint).** For each `(club_pair, grade, w)`:
   - `sum_{tp in (club_pair, grade)} team_pair_play[tp, w] == min(a, b) × play_pg[club_pair, grade, w]`
   - i.e. on each aligned weekend (`play_pg == 1`), exactly `min(a, b)` team-pairs play; on non-aligned weekends (`play_pg == 0`), zero team-pairs play.
   - Implemented as `model.Add(sum == min_ab).OnlyEnforceIf(play_pg)` + `model.Add(sum == 0).OnlyEnforceIf(play_pg.Not())`.
5. **Total aligned-weekend budget (replaces existing `sum_w play == budget`).** For each `(club_pair, grade)`:
   - `sum_w play_pg[club_pair, grade, w] == weekends_budget`
   - where `weekends_budget = max(a, b) × per_matchup` for non-PHL grades, and `max(0, max(a, b) × per_matchup - phl_forced_friday_meetings(data, A, B))` for PHL.
   - Note PHL is always `a=b=1` in the current league, so the PHL formula reduces to `per_matchup - phl_forced_friday_meetings` — identical to today's PHL Sunday budget. No PHL behaviour change.
6. **Cross-grade nested-superset chain unchanged in shape.** For each `(club_pair, w)`, with grades sorted by `weekends_budget` descending:
   - `play_pg[club_pair, g_{k+1}, w] <= play_pg[club_pair, g_k, w]`.
   - Same implication chain as today; runs over the new `play_pg` indicator (semantics-changed but key-shape-preserved).
7. **Early validation.** If `weekends_budget > number_of_available_Sunday_weeks_with_vars` for any `(pair, grade)`, raise `ValueError` at apply-time with the pair+grade+budget+available counts (same pattern as the current atom's `ValueError`).

### Shared helpers (`_club_vs_club_stacked_shared.py`)

8. **New helper `team_pair_counts(data, club_pair, grade) -> tuple[int, int]`** returning `(a, b)` = (# teams of club_A in grade, # teams of club_B in grade), sorted so that `a <= b` is NOT enforced (return in `club_pair` order: `(a_for_first_club, b_for_second_club)`).
9. **New helper `enumerate_team_pairs_in_pair_grade(data, club_pair, grade) -> list[tuple[str, str]]`** returning every cross-club unordered team-pair `(t1, t2)` with t1's club + t2's club == `{A, B}` and both teams in `grade`. Deterministic ordering (sorted by `(t1, t2)` strings).
10. **`per_pair_grade_aligned_weekends(data, club_pair, grade) -> int`** returning `max(a, b) × per_matchup` (using `team_pair_counts` and the existing `per_matchup` derivation).
11. **`pair_grade_sunday_aligned_weekends(data, club_pair, grade) -> int`** (REPLACES `pair_grade_sunday_meetings`): returns `per_pair_grade_aligned_weekends(data, pair, grade)` for non-PHL; for PHL, `max(0, per_pair_grade_aligned_weekends - phl_forced_friday_meetings(data, A, B))`. Returns 0 if a or b is 0 or if budget is exhausted.
12. **Keep `per_pair_grade_matchup_counts`** unchanged — still useful for the `(a, b)` derivation and as a sanity check (matchup_count should equal `a × b`).
13. **Keep `per_pair_grade_meeting_counts`** unchanged for now but mark its docstring "use `per_pair_grade_aligned_weekends` for the stacking budget; this function returns the game-count which is a different quantity". After Units B and C land, callers will be: `club_vs_club_stacked_co_location.py` (line 60 — for grade enumeration outer loop), `clubvsclub_stacked_colocation_regen_soft.py` (line 41), and `clubvsclub_stacked_weekends_regen_soft.py` (line 39) — all for the "which grades have meetings?" outer loop, NOT for the budget. These callers use `per_pair_grade_meeting_counts` solely to enumerate grades (checking `grade_meetings_total.items()`) and then call the budget function separately. They MUST be audited at Unit D time — if they still call `per_pair_grade_meeting_counts` for grade enumeration (which is correct — it returns which grades have cross-club matchups), the function must be KEPT and only its budget-usage consumers removed. DELETE only if truly zero consumers remain (grep first). — (review note — verified: at time of review, `per_pair_grade_meeting_counts` is called for grade enumeration in all four atoms including both hard and soft co-location; this is correct usage independent of the budget fix; deletion is NOT safe unless all four switch to `per_pair_grade_matchup_counts` for enumeration.) —
14. **Keep `phl_forced_friday_meetings`** unchanged — used as-is for PHL budget subtraction.
15. **Keep `collect_pair_grade_week_vars`** unchanged — used to construct `team_pair_play` (filter further by team_pair inside the atom).
16. **Keep `collect_pair_week_sunday_vars`** unchanged — `ClubVsClubStackedCoLocation` reads it.
17. **New constant `STACK_TEAM_PAIR_PLAY_PREFIX = 'cvc_stack_team_pair_play'`** added to the shared helper module; exported.

### Helper-var registry

18. **`docs/system/HELPER_VARS.md` updated** to add the `cvc_stack_team_pair_play` family with key shape `(team_pair, week)`, channeling rule `AddMaxEquality(ind, vars_for_team_pair_week)` (or fixed-0 when no vars), and consumer list (just `ClubVsClubStackedWeekends` itself — `play_pg` is computed from these).

### Co-location atom (minimal code change required)

19. **`ClubVsClubStackedCoLocation` must switch its budget helper call.** The co-location atom calls `pair_grade_sunday_meetings` (imported at `club_vs_club_stacked_co_location.py:59,88`) to compute `sunday_budget` for the second-ranked grade selection. After Unit A deletes `pair_grade_sunday_meetings`, this import breaks. The atom must switch its `pair_grade_sunday_meetings` calls to `pair_grade_sunday_aligned_weekends` — the gate logic ("which grade is second-ranked?") uses the same descending-count sort, so swapping the budget function is sufficient.

    — (review note — verified: `club_vs_club_stacked_co_location.py` lines 59 and 88 import and call `pair_grade_sunday_meetings`; both must be updated; no other internal logic changes) —

    The `play_pg` indicator read via `registry.get(STACK_PLAY_PREFIX, pair, second_grade, w)` (line 119) is unchanged in key shape — semantics shift (OR-of-team-pair-ORs rather than OR-of-raw-vars) but the indicator value is identical for any given game placement, so the gate logic is transparent.

    **Unit C adds `club_vs_club_stacked_co_location.py` to its file-touched list.**

20. **Co-location tests RECALIBRATED** to use the new aligned-weekend counts. Where a co-location test sets up a 2×2 lower-grade scenario, the expected number of co-location-active weekends changes from the old `matchups × per_matchup` to the new `max × per_matchup` — tests' hand-computed oracles updated accordingly.

### Regen-soft twin

21. **`ClubVsClubStackedWeekendsRegenSoft`** (file `constraints/atoms/clubvsclub_stacked_weekends_regen_soft.py`) updated to mirror the new model: deviation penalty against the per-team-pair budget + the per-aligned-weekend cardinality + the total aligned-weekend budget. Penalty-weight key reused. The exact penalty composition is documented in the regen-soft atom's docstring and tests. Also switches its `pair_grade_sunday_meetings` import (line 38) to `pair_grade_sunday_aligned_weekends` (required once Unit A removes the old function).
22. **`ClubVsClubStackedCoLocationRegenSoft`** (file `constraints/atoms/clubvsclub_stacked_colocation_regen_soft.py`) must switch its `pair_grade_sunday_meetings` import (line 40) to `pair_grade_sunday_aligned_weekends`. The co-location regen-soft atom uses this only for the "does this pair have >= 2 grades with non-zero budget?" gate (lines 89-96) — the semantics of that check with the new aligned-weekend count are identical for the skip/no-op decision. The budget function swap is the only code change; Unit C adds it to the file-touched list.

    — (review note — verified: `clubvsclub_stacked_colocation_regen_soft.py` line 40 imports `pair_grade_sunday_meetings`, line 92 calls it; both must switch to `pair_grade_sunday_aligned_weekends` when Unit A removes the old function) —

### Tests

23. **`tests/atoms/test_club_vs_club_stacked_weekends.py`** rewritten with hand-computed oracles, no mocks:
    - 1×1 PHL pair, per_matchup=4 (Maitland-Gosford on season_test): weekends_budget=4, games/weekend=1, expected 4 distinct aligned Sundays with 1 game each.
    - 1×1 non-PHL pair, per_matchup=2 (3rd-grade single-team pair): budget=2.
    - 2×2 lower-grade pair (Tigers 6th × University 6th — VERIFIED in Unit A as the real season_test 2×2 pair: Tigers Yellow + Tigers Black vs University Gentlemen + University Seapigs, per_matchup=2 (R=18, T=10, R//(T-1)=2)): budget=max(2,2)×2=4, games/weekend=2. Hand oracle: 8 cross-team games land on exactly 4 distinct Sundays, 2 games per Sunday, all 4 team-pairs covered exactly twice across the 4 weekends. — (review note — Unit A's `test_club_vs_club_stacked_shared.py:131` ground-truthed this: Tigers and University each field only 1 team in 5th; the real 2×2 pair is in 6th grade. The earlier spec text claiming "Tigers-University 5th, per_matchup=1" was wrong. For the case-table `2x2_pm1` parametrised row, use a synthetic fixture — Unit A's tests do this via `_build_case_table_fixture` — labelled as synthetic.) —
    - 1×2 asymmetric lower-grade pair: use University 4th (University + University 2) × Maitland 4th (1 team) OR Norths 2nd (Norths Light + Norths Dark) × University 2nd (1 team) — both are 1×2 on season_test. For per_matchup=1: budget=2, games/weekend=1. Hand oracle: 2 games on 2 distinct Sundays, the single opposing team plays each of the 2 teams exactly once.
    - PHL forced-Friday subtraction: 2 forced PHL Friday meetings for Maitland-Gosford → Sunday budget = 4 - 2 = 2. Hand oracle.
    - Cross-grade nested chain: synthetic pair with PHL=4 + 6th=2 aligned weekends → 6th's 2 weekends are a subset of PHL's 4. Hand oracle.
    - Validation: pair with budget > available Sunday weeks raises `ValueError`. Hand oracle.
24. **`tests/atoms/test_club_vs_club_stacked_weekends_regen_soft.py`** rewritten with hand-computed oracles per the new penalty composition.
25. **`tests/atoms/test_club_vs_club_stacked_co_location.py`** recalibrated for the new aligned-weekend counts in multi-team-per-grade scenarios (no code change to the atom, but oracle counts shift).
26. **`tests/atoms/test_club_vs_club_stacked_co_location_regen_soft.py`** recalibrated similarly.
27. **New unit tests for the shared helpers** (`tests/atoms/test_club_vs_club_stacked_shared.py` — create if missing):
    - `team_pair_counts` for representative club pairs.
    - `enumerate_team_pairs_in_pair_grade` ordering + completeness.
    - `per_pair_grade_aligned_weekends` matches the case table above.
    - `pair_grade_sunday_aligned_weekends` PHL subtraction edge cases.

### Acceptance probes (bisect harness)

28. **`ClubVsClubStackedWeekends` ALONE on fundamentals → ✅ `REACHED_SEARCH`** (was: ❌ INFEASIBLE pre-redesign, probe 4b in handoff §5). This is the canonical acceptance probe for spec-038. Run:
    ```
    .venv\Scripts\python.exe scripts\bisect_core_feasibility.py --max-time 120 --workers 10 \
        --exclude <every-core-atom-except-ClubVsClubStackedWeekends-and-fundamentals>
    ```
29. **`ClubVsClubStackedWeekends + ClubVsClubStackedCoLocation` on fundamentals → ✅ `REACHED_SEARCH`** (both atoms together remain feasible; co-location reads the new `play_pg` semantics correctly).
30. **Full `core - ClubGameSpread` on fundamentals → ✅ `REACHED_SEARCH`** (spec-035's actual unblock condition — with both spec-037 and spec-038 landed, the previously-blocking pair is gone).

### Docs

31. **`docs/system/HELPER_VARS.md`** — new `cvc_stack_team_pair_play` family entry + updated `cvc_stack_play` entry noting the indicator now sits on top of team-pair-play.
32. **`docs/system/CONSTRAINT_INVENTORY.md`** — `ClubVsClubStackedWeekends` and `ClubVsClubStackedWeekendsRegenSoft` row descriptions updated to reflect aligned-weekend semantics (not game-count semantics).
33. **`docs/todo/00-dependency-tree.md`** — add spec-038 as a live entry.

### Registry / wiring

34. **Registry changes in Unit B.** The `HELPER_VAR_CATALOG` set in `constraints/registry.py` (lines 781-808) must be updated to add `'cvc_stack_team_pair_play'` alongside the existing four `cvc_stack_*` entries. The `HELPER_VAR_PRODUCER_CONSUMER` list at line 1021 keeps the existing `('ClubVsClubStackedWeekends', 'ClubVsClubStackedCoLocation', ['cvc_stack_play'])` triple and gains a second triple `('ClubVsClubStackedWeekends', 'ClubVsClubStackedWeekends', ['cvc_stack_team_pair_play'])`. The `ConstraintInfo` entry for `ClubVsClubStackedWeekends` at line 379 should add `'cvc_stack_team_pair_play'` to its `required_helpers` list (currently only has `['cvc_stack_play']`) since this atom is its own producer. Registry count of `ConstraintInfo` entries (49, verified by grep) is unchanged — no new constraints are registered.

    — (review note — verified: `HELPER_VAR_CATALOG` at `registry.py:781-808` lists all known helper-var kinds; `cvc_stack_team_pair_play` is absent and must be added in Unit B. Without this, `validate_group_order` layer-2 checks may flag it as unknown.) —

## Implementation units

### Unit A — Shared helper rewrite (foundational, pure data)

- **Files touched:** `constraints/atoms/_club_vs_club_stacked_shared.py`, `tests/atoms/test_club_vs_club_stacked_shared.py` (new).
- **Scope:** add `team_pair_counts`, `enumerate_team_pairs_in_pair_grade`, `per_pair_grade_aligned_weekends`, `pair_grade_sunday_aligned_weekends`; add `STACK_TEAM_PAIR_PLAY_PREFIX`; update `__all__` + module docstring. **DO NOT delete `pair_grade_sunday_meetings` in Unit A** — it is still imported by `club_vs_club_stacked_weekends.py`, `club_vs_club_stacked_co_location.py`, `clubvsclub_stacked_weekends_regen_soft.py`, `clubvsclub_stacked_colocation_regen_soft.py`, and several test files; removing it in Unit A would break the build before Units B/C have switched their callers. Deletion moves to Unit D after all consumers have switched. `per_pair_grade_meeting_counts` is left untouched in Unit A. Tests cover hand-computed oracles for the new functions.

    — (review note — verified: at time of review `pair_grade_sunday_meetings` is imported by 4 production files and referenced by 3+ test files; premature deletion in Unit A would break the existing test suite before Units B/C repair it.) —
- **Suggested executor:** Sonnet. Pure data functions, mechanical math, no model-touching.
- **Dependencies:** none within plan.

### Unit B — `ClubVsClubStackedWeekends` atom rewrite

- **Files touched:** `constraints/atoms/club_vs_club_stacked_weekends.py`, `constraints/registry.py` (add `cvc_stack_team_pair_play` to `HELPER_VAR_CATALOG` + `HELPER_VAR_PRODUCER_CONSUMER` + `ClubVsClubStackedWeekends` `required_helpers`), `tests/atoms/test_club_vs_club_stacked_weekends.py`.
- **Depends on:** Unit A merged.
- **Scope:** rewrite `apply` to use the new four-layer model (per-team-pair vars → team-pair-play → play_pg → cross-grade chain). Register team-pair-play under the new prefix; register `play_pg` under the existing prefix with new semantics. Update `constraints/registry.py` per DoD #34. Rewrite the test file per DoD #23.
- **Suggested executor:** Opus. Cross-cutting helper-var registry change, subtle indicator semantics, oracle computation needs care — erring to Opus on the line per `basic`.

### Unit C — Co-location parity + regen-soft parity (parallel-safe with B's commit)

- **Files touched:** `constraints/atoms/club_vs_club_stacked_co_location.py` (switch `pair_grade_sunday_meetings` → `pair_grade_sunday_aligned_weekends`), `tests/atoms/test_club_vs_club_stacked_co_location.py`, `constraints/atoms/clubvsclub_stacked_weekends_regen_soft.py` (switch budget helper + mirror new four-layer model), `tests/atoms/test_club_vs_club_stacked_weekends_regen_soft.py`, `constraints/atoms/clubvsclub_stacked_colocation_regen_soft.py` (switch `pair_grade_sunday_meetings` → `pair_grade_sunday_aligned_weekends`), `tests/atoms/test_club_vs_club_stacked_co_location_regen_soft.py`.
- **Depends on:** Unit B merged (regen-soft and co-location tests need the new semantics established).
- **Scope:**
  - `club_vs_club_stacked_co_location.py`: switch the `pair_grade_sunday_meetings` import and call (lines 59, 88) to `pair_grade_sunday_aligned_weekends`. No other logic changes — the grade-ranking sort and `registry.get(STACK_PLAY_PREFIX, ...)` lookup are unchanged.
  - `clubvsclub_stacked_colocation_regen_soft.py`: switch the `pair_grade_sunday_meetings` import and call (lines 40, 92) to `pair_grade_sunday_aligned_weekends`. The >= 2 grades gate uses the same boolean outcome.
  - `clubvsclub_stacked_weekends_regen_soft.py`: switch budget helper AND rewrite the apply logic to mirror the new four-layer model with deviation penalty.
  - All associated test files recalibrated per DoD #24, #25, #26.

    — (review note — co-location atom is NOT code-free in this spec; it requires the budget helper switch. Previous DoD #19 claim "co-location CODE is NOT modified" was wrong. Without the switch, Unit A's new function exists alongside the old one with no deletion; Unit D would delete `pair_grade_sunday_meetings` and then the co-location import would break at runtime. The switch must happen in Unit C, not Unit D.) —
- **Suggested executor:** Sonnet. Mechanical mirroring once Unit B's API is settled.

### Unit D — Cleanup of dead helpers + bisect-harness acceptance + docs

- **Files touched:** `constraints/atoms/_club_vs_club_stacked_shared.py` (DELETE `pair_grade_sunday_meetings` after all consumers have switched; DELETE `per_pair_grade_meeting_counts` ONLY if no consumers remain for grade enumeration — see DoD #13 note), `docs/system/HELPER_VARS.md`, `docs/system/CONSTRAINT_INVENTORY.md`, `docs/todo/00-dependency-tree.md`.
- **Depends on:** Units A, B, C all merged.
- **Scope:**
  - Grep for `pair_grade_sunday_meetings` — must be zero callers after Unit C; delete from `_club_vs_club_stacked_shared.py` and its `__all__`.
  - Grep for `per_pair_grade_meeting_counts` — if callers remain for grade enumeration in co-location atoms, keep the function (grade enumeration is a valid use); delete only if genuinely zero callers.
  - Also grep for remaining test-file references to `pair_grade_sunday_meetings` (e.g. `test_club_vs_club_stacked_alignment.py` line 37, `test_cvc_stacked_friday_aware.py` line 31, `test_clubvsclub_stacked_weekends_regen_soft.py` line 35) — these test files must also switch their imports to `pair_grade_sunday_aligned_weekends`.
  - Update docs per DoD #31–33. Run the three bisect-harness probes per DoD #28–30 and record the verdicts in the merge commit message.
- **Suggested executor:** Opus (the probe verdict is the canonical "did we ship?" check; needs orchestrator-level judgement).

## Doc registry

- `docs/system/HELPER_VARS.md` — new `cvc_stack_team_pair_play` family, updated `cvc_stack_play` entry.
- `docs/system/CONSTRAINT_INVENTORY.md` — `ClubVsClubStackedWeekends*` row updates.
- `docs/todo/00-dependency-tree.md` — add spec-038 entry; mark as ready-to-start in parallel with spec-037.

## Out of scope

- **`AwayClubHomeWeekendsCount`** — spec-037 (parallel).
- **`ClubVsClubStackedCoLocation` co-location atom internals** — only the budget helper call is switched (DoD #19). The game-placement logic (same-field, contiguous-slots, gate via `registry.get`) is NOT modified. Its test oracles shift to match new aligned-weekend counts in multi-team scenarios.
- **Symmetry-breaking of team-pair selection on aligned weekends** — when both clubs field 2 teams (2×2 case) and `min(a,b) = 2` team-pairs play on each aligned weekend, *which* permutation of team-pairs occurs on which weekend is left free (the convenor's "free to choose" note covers the asymmetric case; the symmetric case naturally permits any pairing). Adding a symmetry-breaking rule (e.g. lex-order on team names) is a separate concern, not in this spec.
- **`per_matchup` semantics changes** — left as-is from `per_pair_grade_matchup_counts`/`per_pair_grade_meeting_counts`. If the convenor later wants per-grade scheduling-method tweaks (`grade_scheduling_method` already exists in `num_rounds`), that's a separate spec.
- **The spec-035 e2e run** — resumes after both spec-037 and spec-038 land. Not in scope here.
- **Production-config validation.** This spec validates against `season_test` (forced-free) via the bisect harness. Validating the full 2026 production config with real `FORCED_GAMES` happens via spec-035's eventual run.

## Dependencies

- `depends_on: none`. spec-038 is independent of spec-037 — they touch different atoms, different shared helpers, and the only shared module (`_phl_forced_friday_helper.py`) sees spec-037 *remove* per-club functions while spec-038 *keeps* the per-pair `phl_forced_friday_meetings` it uses; no merge contention.
- spec-035  implicitly depends on both this spec and spec-037 landing before its e2e probe will pass — but spec-035's `depends_on` should be updated when its next edit happens, not by us. spec-035 is `in_progress` and not implementable until its handoff is acted on.

## Risks & blast radius

- **Helper-var pool key collision.** New prefix `cvc_stack_team_pair_play` must not collide with existing prefixes. Plan-time grep confirms no current use of this prefix. Implementation must re-grep at start of Unit B.
- **`play_pg` semantics shift.** The key shape `(STACK_PLAY_PREFIX, club_pair, grade, week)` is preserved, but `play_pg` is now an OR-of-team-pair-ORs instead of an OR-of-raw-vars. Functionally identical ("≥ 1 game this weekend"), but worth verifying with an integration test where `ClubVsClubStackedCoLocation` reads the indicator and behaves correctly. Unit C's co-location test recalibration is the catch point.
- **PHL behaviour preservation.** PHL is always `a=b=1`, so the new formula collapses to the old PHL behaviour (`per_matchup - phl_forced_friday_meetings`). A test with the Maitland-Gosford PHL pair on real `season_2026` config (where `FORCED_GAMES` includes PHL Fridays) should confirm no PHL behaviour change. Added as one of the Unit B test scenarios.
- **Aligned-weekend cardinality `OnlyEnforceIf` correctness.** The construction `sum == min_ab.OnlyEnforceIf(play_pg)` + `sum == 0.OnlyEnforceIf(play_pg.Not())` must be tested for both directions (the AddMaxEquality channeling on `play_pg` already implies the right direction, but `OnlyEnforceIf` requires both branches). Hand-computed oracles in Unit B cover this. Verified: `HelperVarRegistry.get_or_create_bool` (`constraints/helper_vars.py:49-66`) uses `AddMaxEquality` for non-empty lists and `model.Add(ind == 0)` for empty, which is correct bidirectional BoolVar channeling (`max` of BoolVars == OR). The plan's DoD #3 claim that `AddMaxEquality(play_pg, [team_pair_play[tp, w] for tp in tps])` gives `play_pg == 1 iff any team_pair_play is 1` is confirmed correct.
- **Co-location atom transparently uses `play_pg` — but the new `play_pg` is computed from team-pair-play, not directly from raw vars.** If the channeling on `play_pg` doesn't hold tight (e.g. if `team_pair_play[tp, w]` is fixed-to-0 by `AddMaxEquality` when no vars exist, but `play_pg` is OR over those — both should hold), the co-location atom would gate on a slightly different signal than today. Unit B's test scenarios include a "no Sunday vars exist for this `(pair, grade, week)`" edge case to confirm both indicators correctly fix to 0.
- **Test-oracle drift on `season_test`.** The `num_rounds` / team-layout config can shift. Tests use inline-constructed fixtures for the case-table scenarios and `load_season_data('test')` for the realistic-shape scenarios, recomputing oracles from current config values at test runtime.
- **Symmetric 2×2 case: multiple valid partitions.** For Tigers-University 6th (the real season_test 2×2 pair, per_matchup=2 — Unit A verified via real data): 4 team-pairs, 4 aligned weekends, 2 games/weekend. Each team-pair plays on exactly 2 of the 4 weekends; on each aligned weekend, 2 of the 4 team-pairs play. There are multiple valid assignments; the solver picks one. The convenor's "free to choose" intent explicitly covers this case — no symmetry-breaking is needed or required. (review note — `sum_w team_pair_play[tp, w] == per_matchup == 2` for each of 4 team-pairs + `sum_{tp} team_pair_play[tp, w] == min(a,b) == 2` on each of 4 aligned weekends. 8 indicator-on cells in a 4×4 grid with row sums 2 and column sums 2 — many valid configurations exist; any one satisfies. Correct and sufficient.)

## Open Questions

0 — the formula was settled by the convenor in this session's discussion; the symmetric/asymmetric/PHL cases are all covered by the case table above.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->

0. **Do NOT start without an explicit user instruction to implement this plan.** A `ready` status means "authorised to be built when asked", not "build now". If you arrived here straight off authoring/review with no user go-ahead, STOP and ask.
1. Status must be `ready` (carries a `reviewed:` stamp). If `review_pending`/`under_review`, let review finish. If `blocked`, STOP: Open Questions need the user.
2. Only after the user says to implement: stamp `building`, claim `owner`. You are the orchestrator (Opus).
3. **Unit A** runs first on its own worktree+branch off `final-form` (`spec038-unitA`):
   - Delegate to a Sonnet subagent.
   - Gates: type-check + `pytest tests/atoms/test_club_vs_club_stacked_shared.py -v`; AST dead-code sweep on the touched file; `/adversarial` Mode B on the diff.
   - Merge → push → tear down worktree.
4. **Unit B** runs after A on `spec038-unitB` worktree:
   - Delegate to an Opus subagent.
   - Gates: type-check + `pytest tests/atoms/test_club_vs_club_stacked_weekends.py -v`; bisect-harness probe DoD #28; `/adversarial` Mode B on the diff.
   - Merge → push → tear down worktree.
5. **Unit C** runs after B on `spec038-unitC` worktree:
   - Delegate to a Sonnet subagent.
   - Gates: type-check + `pytest tests/atoms/test_club_vs_club_stacked_co_location.py tests/atoms/test_club_vs_club_stacked_weekends_regen_soft.py tests/atoms/test_club_vs_club_stacked_co_location_regen_soft.py -v`; `/adversarial` Mode B on the diff.
   - Merge → push → tear down worktree.
6. **Unit D** runs after A, B, C all merged, on `spec038-unitD` worktree:
   - Delegate to an Opus subagent.
   - Run bisect-harness probes DoD #28, #29, #30 — capture verdicts in commit message.
   - Update docs. `/adversarial` Mode B on the diff (light — mostly docs).
   - Merge → push → tear down worktree.
7. When all units pass: stamp the plan `done`, archive to `docs/todo/done/spec-038-clubvsclub-stacked-team-pair-granularity.md`, update `docs/todo/00-dependency-tree.md`.
