<!-- status: building -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-030 -->
<!-- note: spec-029 (which edited the same analytics/tester.py) is now done (ccc3d07); that contention is resolved. Original depends_on listed spec-029 too. -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline -->

<!-- owner: session=opus-aiupd-031 claimed=2026-05-24 -->

# spec-031 — Remove the `ClubFieldConcentration` diagnostic

**Spec source:** convenor request (this session) — after explaining what `ClubFieldConcentration` does, the convenor judged it "NOT a problem as far as I am aware" and asked to remove it.

## Why

`ClubFieldConcentration` is a **tester-only diagnostic** (`solver_class_names=[]`, `tester_only=True`, `constraints/registry.py:410-417`). Verified against `analytics/tester.py:2217` (`_check_club_field_concentration`): per `(club, week, day)` at Broadmeadow only, it computes `field_spread = num_games − max_games_on_any_single_field` and flags when a club's games for that day are *spread across too many fields* rather than clustered on one (hard cap `num_games // 2 − 1 + slack`; any `field_spread > 0` also emitted as a `[soft]` warning). (review fix — M1: corrected tester.py line reference from 2215 to 2217, the real current line.)

It is removable for two concrete reasons:

1. **No solver constraint enforces it.** It is a pure post-hoc check with no production rule behind it, so it can only ever *report* — it cannot shape a draw. Its `[soft]` branch fires on *any* multi-field club-day, producing routine noise in the violation report for a condition the convenor does not consider a problem.
2. **Its intent is superseded.** The "keep a club's games together" idea is owned by `ClubGameSpread` (spec-024), which enforces per-field contiguity + an off-primary-field soft penalty as a real solver constraint. `ClubFieldConcentration` is a stale echo of the pre-spec-024 "concentrate clubs on a field" thinking and even reuses `ClubGameSpread`'s slack key (`tester.py:2225`). (review fix — M2: corrected slack line reference from 2223 to 2225.)

Cost of not doing this: every violation report keeps surfacing a metric the convenor has explicitly disowned, obscuring the violations that matter.

## Definition of Done

1. `_check_club_field_concentration` is deleted from `analytics/tester.py` (the method body at line 2217 and its checks-list entry `('ClubFieldConcentration', self._check_club_field_concentration)` at line 1203). Also remove the inline docstring comment at line 1162 that names `ClubFieldConcentration` as an example of a tester-only diagnostic (update or generalise that sentence). (review fix — M3: added missing cleanup of line 1162 comment.)
2. The `ClubFieldConcentration` `ConstraintInfo` entry is deleted from `constraints/registry.py` (lines 410-417).
3. `len(CONSTRAINT_REGISTRY) == 49` (was 50 after spec-030); `tests/test_constraint_registry.py` count assertion updated to `49`. (Oracle: 51 − spec-030's 1 − this spec's 1 = 49; chain is correct but spec-030 must be done first.)
4. Every live test reference to `ClubFieldConcentration` / `_check_club_field_concentration` is removed or re-pointed across these **seven** files (review fix — C1: plan said 5 files; the full grep reveals 7; `config/season_2026.py` and `analytics/tester.py` inline comment are additional sites):
   - `analytics/tester.py` — method + checks-list entry + line-1162 comment (per DoD 1).
   - `constraints/registry.py` — `ConstraintInfo` entry (per DoD 2).
   - `config/season_2026.py` — delete the `'ClubFieldConcentration': 80_000` entry from `PENALTY_WEIGHTS` (line 876), and update/remove the adjacent comment on line 878 (`'ClubVsClubAlignmentField': 0,  # Superseded by ClubFieldConcentration`) since `ClubFieldConcentration` itself is now gone. The comment should read e.g. `# (key retained for legacy checkpoint compat; was superseded by ClubFieldConcentration, itself now removed)` or just be deleted if `ClubVsClubAlignmentField` is never used.
   - `tests/test_constraint_registry.py` — count assertion → `49`; delete `TestTesterOnlyConstraints.test_club_field_concentration_is_tester_only` and `test_get_tester_only_returns_expected`'s `'ClubFieldConcentration' in tester_only` assertion (or delete the assert line; keep the test if it still checks other tester-only entries).
   - `tests/test_tester_coverage.py` — delete the entire `TestCheckClubFieldConcentration` class (lines 883–917); it calls `_check_club_field_concentration` directly.
   - `tests/test_tester_metadata.py` — delete the entire `TestTesterOnlyConstraint` class (lines 453–476); also fix line 145's `assert 'ClubFieldConcentration' in run_names` (delete or replace with a remaining tester-only diagnostic name such as `ForcedGames`).
   - `tests/test_constraints_realdata.py` — delete `TestClubFieldConcentration` class (lines 399–404); update the docstring comment at line 32 that counts `ClubFieldConcentration: 37`.
   - `tests/test_constraints_ai.py` — delete the `# ---------- Field concentration ----------` section (lines 617–702) from `TestClubGameSpreadAI`. **Important nuance (review fix — H1):** these tests import `ClubGameSpreadAI` from `constraints.archived.ai` and test the *archived* solver's internal `data['penalties']['ClubFieldConcentration']` penalty dict — they test different behaviour (the old archived solver's field-concentration sub-feature) than the tester diagnostic being removed here. Deleting them loses those archived-solver tests, but since that code is archived and not production, deletion is safe. Do NOT confuse these with any live solver test.
   - `docs/system/CONSTRAINT_INVENTORY.md` — delete the `ClubFieldConcentration` row (appears at lines 54, 135, and 219; all three must go).
5. Running `DrawTester.run_violation_check()` on the current `draws/2026/current.json` produces **no** `ConstraintResult` or `Violation` with `constraint == 'ClubFieldConcentration'`, and does not raise. A draw that previously produced only `ClubFieldConcentration` `[soft]` warnings now reports clean for that category.
6. Docs updated per the doc registry. Type-check clean; changed-file lint clean; AST sweep clean (no orphan reference to the removed method/entry, no now-unreachable helper left behind); full suite green. Confirm `'ClubFieldConcentration'` does not appear in any non-archived, non-checkpoint `.py` or `.md` file after the changes.

> Note on checkpoint JSON files: `checkpoints/run_1/run_metadata.json` and `checkpoints/run_2/run_metadata.json` contain `"ClubFieldConcentration": 80000` in their `penalty_weights` snapshot — these are historical artefacts of past solver runs and do **not** need to be edited (checkpoints are write-once records).

## Implementation units

Single unit. A self-contained removal across `tester.py` + `registry.py` + `season_2026.py` + 5 test files + 1 doc (~1 deleted method, 1 deleted registry entry, 1 deleted config key, count/coverage test edits). Graded S2 (removes public-ish surface across modules + changes the violation-report contract). One worktree.

### Unit A — Delete the diagnostic + tests + docs

- **Files touched:**
  - `analytics/tester.py` — delete the `('ClubFieldConcentration', self._check_club_field_concentration)` checks-list entry (confirmed at line 1203) and the `_check_club_field_concentration` method (confirmed at line 2217 to its end, line ~2267). Also update the inline comment at line 1162. Confirm via AST that no other method calls it and no helper is left dangling. (review fix — M1/M3: corrected line numbers; added line 1162 comment.)
  - `constraints/registry.py` — delete the `ClubFieldConcentration` entry (lines 410-417).
  - `config/season_2026.py` — delete `'ClubFieldConcentration': 80_000` from `PENALTY_WEIGHTS` (line 876); update or delete adjacent line 878 comment referencing it. (review fix — C1: this file was entirely absent from the plan's file list.)
  - `tests/test_constraint_registry.py` — count assertion → `49`; drop the `ClubFieldConcentration` assertions from `TestTesterOnlyConstraints`.
  - `tests/test_tester_coverage.py` — delete the entire `TestCheckClubFieldConcentration` class (lines ~883–917).
  - `tests/test_tester_metadata.py` — delete `TestTesterOnlyConstraint` class (lines ~453–476); fix line 145 in the test that asserts `'ClubFieldConcentration' in run_names`.
  - `tests/test_constraints_realdata.py` — delete `TestClubFieldConcentration` class and update line-32 comment.
  - `tests/test_constraints_ai.py` — delete the `# ---------- Field concentration ----------` section (lines ~617–702) from `TestClubGameSpreadAI`. These test the archived solver's internal penalty dict, not the tester diagnostic; deletion is safe. (review fix — H1: plan gave insufficient per-site guidance; clarified what these tests actually test and why deleting is safe.)
  - `docs/system/CONSTRAINT_INVENTORY.md` — delete all three `ClubFieldConcentration` rows (lines 54, 135, 219). (review fix — M4: plan said "the row" implying one; there are three.)
- **Change summary:** pure deletion of a tester-only diagnostic and all its references, including a config penalty-weight entry and three inventory rows.
- **Depends on:** spec-030 done (shares `constraints/registry.py` and `docs/system/CONSTRAINT_INVENTORY.md`; the count test moves 51→50→49 across the two specs); spec-029 done (shares `analytics/tester.py`).
- **Executor model:** Sonnet (mechanical, well-scoped deletion).
- **No-mock test outline (Given/When/Then, hand-computed oracle):**
  - *Given* the real `load_season_data(2026)` and `DrawStorage.load('draws/2026/current.json')`, *when* `DrawTester.from_file(...).run_violation_check()` runs, *then* no `ConstraintResult` or `Violation` has `constraint == 'ClubFieldConcentration'`, and the run completes without error. (Oracle: the check is removed from the run list, so count == 0.)
  - *Given* `len(CONSTRAINT_REGISTRY)`, *then* it equals 49 and `'ClubFieldConcentration' not in CONSTRAINT_REGISTRY`. (Oracle: 51 − spec-030's 1 − this spec's 1 = 49.)
  - *Given* `'ClubFieldConcentration' in load_season_data(2026)['penalty_weights']`, *then* it is `False` (the key is deleted from `PENALTY_WEIGHTS`).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — remove all three `ClubFieldConcentration` rows (lines 54, 135, 219). (review fix — M4: was "the row".)
- `docs/todo/GOALS.md` — spec-031 row is already present (status `review_pending` → update to `done` when done).
- `docs/todo/00-dependency-tree.md` — spec-031 node already present; update status when done.

## Out of scope

- **Touching `ClubGameSpread`** (the real solver constraint that supersedes this diagnostic) — it is correct and unchanged; this spec only removes the redundant *report*.
- **Removing any other tester-only diagnostic** — only `ClubFieldConcentration` was disowned.
- **The group restructure** (spec-032) and **the PHL/2nd cleanup** (spec-030) — separate specs.
- **Editing checkpoint JSON files** — `checkpoints/run_*/run_metadata.json` are historical write-once records; leave them as-is.

## Dependencies

- **Other plans:**
  - `spec-030` — must land first: shared `constraints/registry.py` and `docs/system/CONSTRAINT_INVENTORY.md`, and the `tests/test_constraint_registry.py` count assertion is updated by both (51→50 in spec-030, 50→49 here). Serialising avoids a registry/count merge conflict.
  - `spec-029` — **now `done`** (merged `ccc3d07`); it had edited the same `analytics/tester.py` and `CLAUDE.md`, so it was originally a blocker. With it merged, the contention is resolved — branch from current source and re-read `tester.py` before editing.
- **Within this plan:** single unit; no internal dependencies.

## Risks & blast radius

- **`analytics/tester.py` line drift.** Line numbers quoted in this plan (1203, 2217) match the current state post-spec-029 (confirmed by adversarial review). Still: the implementer must re-read the file immediately before editing and locate targets by name, not just by line number.
- **Tester-coverage test is a tripwire.** `test_tester_coverage.py` asserts a bijection between registry tester-methods and tester `_check_*` methods. Removing both the registry entry and the method keeps them in sync; removing only one would trip it. DoD 4 requires both go together.
- **`config/season_2026.py` penalty weight is easy to miss.** It does not follow from a grep of `test` files and was absent from the original plan. The `PENALTY_WEIGHTS` dict key serves no purpose once the tester diagnostic is gone (the archived solver used it, but that path is no longer production). Deleting it cleanly avoids confusion if `PENALTY_WEIGHTS` is ever iterated.
- **`test_constraints_ai.py` archived-solver tests.** Lines 617–702 of `TestClubGameSpreadAI` test `constraints.archived.ai.ClubGameSpreadAI`'s internal `ClubFieldConcentration` penalty dict — not the diagnostic being deleted. Deleting them is safe (archived code, not production) but note the deliberate loss of archived-solver coverage.
- **Three rows in CONSTRAINT_INVENTORY.md.** `ClubFieldConcentration` appears at lines 54, 135, and 219 in three separate tables. The implementer must delete all three, not just the first.
- **Incidental string references.** Several test files mention `ClubFieldConcentration` only inside assertion messages or docstrings. Grep each site before deleting the whole test method/class to ensure no nearby assertion tests something unrelated.

## Open Questions

None.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Autonomous: run end-to-end without waiting for the user, except where this hits `blocked`. -->
1. Status must be `ready` (carries a `reviewed:` stamp). Verify `depends_on: spec-030` is `done` and merged on the source branch (spec-029 is already `done`); if not, do NOT start (basic-skill: unsatisfied dependency = cannot start). If `blocked`, STOP.
2. Stamp `building`, claim `owner`. Orchestrator = Opus; the deletion unit may go to Sonnet.
3. Single unit (Unit A): own worktree+branch off the up-to-date source (post spec-029, post spec-030). Re-read `analytics/tester.py` before editing. Run the S2 gates: type-check, changed-file lint, AST sweep (confirm no orphaned helper, no remaining `ClubFieldConcentration` string in non-archived non-checkpoint files), no-mock tests (DoD 5/6), docs updated.
4. After implementing, launch `/adversarial` Mode B to verify the diff against this plan's DoD (deletion completeness across all 9 sites; tester-coverage test sync; config key removed; all three inventory rows gone). Route fixes, re-verify. NEVER merge unverified.
5. Merge → push origin → post-merge verify (`run_violation_check` on current.json shows no `ClubFieldConcentration`; `len(CONSTRAINT_REGISTRY) == 49`; `'ClubFieldConcentration' not in load_season_data(2026)['penalty_weights']`) → remove worktree. Tick the checkbox.
6. Stamp the plan `done`, move it to `docs/todo/done/`, update `docs/todo/00-dependency-tree.md` (drop spec-031; note spec-032 now unblocked).
