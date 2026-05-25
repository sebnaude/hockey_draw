<!-- status: done -->
<!-- reviewed: adversarial Sonnet review 2026-05-25 — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-033 -->
<!-- owner: session=opus-5989004-20260525T083142Z claimed=2026-05-25T08:31:42Z -->
<!-- note 2026-05-25: prior 'session=slack' owner stamp cleared — orphaned claim, no worktree/branch/claim-commit existed. -->
<!-- done: 2026-05-25 — all 3 units merged into final-form + /adversarial Mode B verified.
     Unit A f4205a9 (single-solve default + flag remap) + severity-ordering fix 1c87d62 + baseline ed35167;
     Unit B 6cbbdf0/312bdf0 (deleted legacy _club_alignment_* engine path, superseded by spec-005 stacked cluster; registry 49 unchanged);
     Unit C 3323270/f353c3f/e378d8f (docs + skill + CLI-test cleanup to the three-mode model).
     Discovered+fixed in-flight (S1): severity_solver_stages() ordered atoms alphabetically, crashing --severity on the stacked pair — now ordered by canonical_index.
     Post-merge: no-flag single solve applies the full 5-stage/15-atom set + enters solver; --severity dispatches clean (severity_3 stacked pair applies, no RuntimeError); 252 passed/1 skipped across changed areas.
     Convenor note: the no-flag default now applies the COMPLETE modern set and is INFEASIBLE at slack 0 on the 2026 production config (forced games + spec-033 hard caps) — same documented state as spec-033; slack is the release; the forced-free real solve is spec-035. --> 

# spec-036 — Default = single-solve full modern set; remap solve-mode flags; delete legacy alignment path

## Why

The solve-mode CLI is both **incomplete** and **mislabelled**, and it carries a legacy constraint path that diverges from the production atoms. Verified against the worktree this session:

1. **`--simple` applies an incomplete constraint set.** `--simple`/`--unified` route to `main_simple → _main_simple_unified` (`main_staged.py:1545`), which runs only `engine.apply_phase_a/b/c()` (`1266-1270`). That dispatches engine-native methods + the PHL-times and ClubDay atom clusters + the **legacy `_club_alignment_hard/_soft`** — and nothing else. The full production set lives in `DEFAULT_STAGES` and is applied only by the staged dispatcher (`apply_solver_stage`, `stages.py`). So `--simple` **silently omits 15 atoms**: `SameGradeSameClubNoConcurrency`, `PHLAnd2ndAdjacency`, `BalancedByeSpacing`, `VenueEarliestSlotFill`, `ClubNoConcurrentSlot`, `AwayClubHomeWeekendsCount`, `AwayClubPerOpponentAndAggregateHomeBalance`, `ClubVsClubStackedWeekends`, `ClubVsClubStackedCoLocation`, `PreferredGames`, `SoftLexMatchupOrdering`, `NIHCFillWFBeforeEF`, `NIHCFillEFBeforeSF`, `TeamPairNoConcurrency`, `PreferredWeekendsAwayGround`. (review fix — counted 15 non-engine DEFAULT_STAGES atoms, not 14; all 15 verified against `apply_stage_1_hard`/`apply_stage_2_soft` and `ALL_ENGINE_KEYS` in `stages.py`; `PHLConcurrencyAtBroadmeadow`/`PHLAnd2ndConcurrencyAtBroadmeadow` have `atom_group='PHLAndSecondGradeTimes'` and ARE engine-dispatched.) Because CLAUDE.md documents `--simple` as the generate command, draws produced that way have been missing these rules and using the legacy alignment instead of the spec-005 stacked cluster.

2. **The mode flags are mislabelled vs the convenor's intent.** Today: no-flag → `DEFAULT_STAGES` incremental staged solve (`main_staged`, `severity_staged=False`); `--staged` → **severity** staging (`run.py:938`); `--simple`/`--unified` → single solve. The convenor wants: **no-flag = everything applied at once (single solve)**, `--staged` = the `DEFAULT_STAGES` incremental staged solver, `--severity` = the severity-grouped staged solver. `--groups` (selection) and `--slack` are orthogonal and work in every mode.

3. **The legacy `_club_alignment_hard/_soft` are an un-migrated `--simple`-only path.** They are the alignment implementation for the engine `apply_phase_*` path only; the staged path uses the spec-005 stacked atoms and skips them (the `ClubVsClubAlignment` engine key is in no group, always in `skip_constraints`). Once `--simple`/default routes through the same stage dispatcher (Unit A), these methods are dispatched by **no** path → genuinely dead (superseded by `ClubVsClubStackedWeekends`/`CoLocation`, spec-005) → deletable.

Cost of not solving: a documented command that produces under-constrained draws; three solve modes whose names don't match what they do; and a legacy alignment rule shadowing the production one.

## Definition of Done

**Unit A — Single-solve default + mode-flag remap:**

1. The single-solve builder (`_main_simple_unified`, `main_staged.py:1237`) is rewritten to apply the **full selected constraint set** by looping every resolved stage through `apply_solver_stage` (apply-only, no per-stage solve), then `build_objective()` once, then a single `solver.Solve`. It no longer relies on `engine.apply_phase_a/b/c` for the constraint set; it uses the same dispatcher as the staged path, so its applied set is **identical** to the staged path's for the same `--groups` selection. `skip_constraints` continues to honour `--groups`/`--exclude` exactly as today.
2. `run.py` mode dispatch (`~916`) is remapped: **no mode flag → the single-solve path** (DoD 1); `--staged` → `main_staged` with the `DEFAULT_STAGES` incremental dispatch (the *current* no-flag behaviour); **new `--severity`** flag → `main_staged` with `severity_staged=True` (the *current* `--staged` behaviour).
3. The `--simple` and `--unified` flags are **removed** from the `generate` subparser (`run.py:65,92`) and every reference in `run.py` (`899`, `916`). Forward-only: no alias kept. `getattr(args,'severity',False)` and `getattr(args,'staged',False)` drive the dispatch; both false → single solve.
4. Regen (`regen_active`) routing is preserved: a regen run still dispatches via the staged `apply_constraint_set` path (it cannot run as a single engine solve); the `args.simple` reference in the regen branch (`run.py:899`) is removed and the branch keyed off the absence of a staged/severity flag instead. `--regen-from` behaviour is otherwise unchanged.
5. `--groups` selection and `--slack` work identically across no-flag / `--staged` / `--severity` (a test asserts the *applied constraint set* for `--groups core` is the same in all three modes; slack value lands in `data['constraint_slack']` regardless of mode).
6. No-mock test: a no-flag generate on a small synthetic season applies the **full** `DEFAULT_STAGES` atom set (assert each of the 15 previously-missing atoms is present in `data['constraints_applied']`), solves once (one `solver.Solve` call / one objective build), and is feasible. `--staged` applies the same set across multiple stage-solves; `--severity` applies it across severity groups. (review fix — "14" corrected to "15")

**Unit B — Delete the now-dead legacy alignment path:**

7. `constraints/unified.py`: delete `_club_alignment_hard` (`620-681`) and `_club_alignment_soft` (`831-~914`) and their dispatch sites (`377`, `772-773`) plus the `if 'ClubVsClubAlignment' not in _skip` guards. Delete the now-orphaned groupings `by_grade_clubpair_round` and `by_sunday_clubpair_round_field` (declarations `167-168`, build `272-274`) — confirmed used ONLY by those two methods (no other reader in `constraints/`).
8. `constraints/stages.py`: remove `'ClubVsClubAlignment'` from `ENGINE_HARD_KEYS` (`58`) and `ENGINE_SOFT_KEYS` (`72`). The `ClubVsClubAlignment` **registry entry stays** — it is the tester anchor (`_check_club_vs_club_alignment`, violation name `ClubVsClubAlignment`, shared by the stacked atoms) and the legacy class-name resolver (`resolver.py:472-473`, `soft.py:984-985`, `severity.py:79-80`). `len(CONSTRAINT_REGISTRY)` is therefore unchanged.
9. `PENALTY_WEIGHTS['ClubVsClubAlignment']` (`season_2026.py:873`) and `['ClubVsClubAlignmentField']` (`878`) are removed (their only live readers were the deleted soft method; archived parity classes use `weights.get(..., default)` fallbacks, so they still import). (Supersedes spec-033 Unit A's "LEFT" note — coordinate at merge: spec-033 lands first.)
10. Deletion is justified as **superseded** (basic skill): the replacement is the spec-005 `ClubVsClubStackedWeekends`/`CoLocation` cluster, which covers alignment in every solve path after Unit A. No wire-in needed.
11. No-mock test: after Unit A+B, a no-flag solve still enforces club-vs-club alignment (via the stacked atoms) — assert the stacked atoms appear in `constraints_applied` and a deliberately mis-aligned draw is rejected; assert `_club_alignment_hard`/`_soft` no longer exist (AST/`getattr` check) and the engine builds clean.

**Unit C — Documentation + skill cleanup:**

12. `CLAUDE.md` (worktree): every `run.py generate … --simple` example becomes the no-flag form; document the three modes (no-flag single-solve = default, `--staged` = DEFAULT_STAGES incremental, `--severity` = severity-staged) and that `--groups`/`--slack` are orthogonal. Remove `--simple`/`--unified` from any flag list.
13. `.claude/commands/generate-draw` skill: update its command template to the no-flag default (drop `--simple`). Note: this file does NOT exist in the final-form worktree; it lives in the parent project at `draw/.claude/commands/generate-draw`. (review fix — path clarified; absence in worktree confirmed.)
14. Docs updated to the new mode model and to drop `--simple`/`--unified`: `docs/operator-ai/AI_OPERATIONS_MANUAL.md`, `docs/operator-ai/SYSTEM_OPERATION.md`, `docs/operator-human/CAPABILITIES.md`, `docs/operator-human/USER_GUIDE.md`, `docs/system/STAGES.md`, `docs/system/SYSTEM_OVERVIEW.md`, `docs/system/REGEN_CONSTRAINTS.md` (note: regen ignores mode flags; the old "`--simple` ignored for regen" wording is removed). The `docs/todo/done/spec-023`/`spec-027` references are historical — left as-is.
15. `--list-groups`/help text and any `--simple`-mentioning help strings updated; `run.py --help` shows `--staged`, `--severity`, no `--simple`/`--unified`.
16. **Additional files with `--simple`/`--unified` references that Unit C must update** (review fix — these were missing from the original scope): (a) `README.md` (line 22: generate example uses `--simple`); (b) `.github/copilot-skills/hockey-draw-scheduler.md` (lines 33, 36, 74, 182: multiple `--simple` examples + description); (c) `constraints/severity.py` (line 15: a comment example uses `--simple --relax` — update comment only); (d) `docs/todo/GOALS.md` (line 301: `--simple` description in comment — update); (e) `docs/todo/spec-026-regeneration-mode.md` (DoD line 169: `--simple` in a generate command — update); (f) `docs/todo/spec-033-slack-cleanup-soft-analogue-audit.md` (execution protocol lines 57, 172: use `--simple` — update to no-flag). Tests with `--simple` references (`tests/test_run_cli.py`, `tests/test_run_coverage.py`, `tests/test_groups_cli_wiring.py`, `tests/test_regen_end_to_end.py`) **must be updated** as part of Unit C (not Unit B) since they test CLI flag handling, not the legacy alignment path.
17. Full suite green; `python run.py generate --year 2026` (no flag) builds + solves the full set; `--staged` and `--severity` build; AST sweep shows no dead `_club_alignment_*`; changed-file lint clean. Mode B reviewer greps `--simple` and `--unified` across the entire worktree (including `.github/`) and verifies zero hits outside `docs/todo/done/` historical records.

## Implementation units

> Serial **A→B→C**. A touches `run.py` + `main_staged.py`; B touches `constraints/unified.py`/`stages.py`/`config/season_2026.py`; C touches docs + the skill + help text. B depends on A (the legacy methods are only provably dead once A reroutes the single-solve path off `apply_phase_*`). C depends on A+B (docs reflect the final modes + deletions).

### Unit A — Single-solve default + flag remap  (S3; executor: Opus — dispatch rewrite, behaviour change)
**Files:** `run.py` (mode args + dispatch `~65,92,101,899,916,938`), `main_staged.py` (`_main_simple_unified` rewrite `1237-1284`; reuse `apply_solver_stage` + `build_objective`), new test under `tests/`. **Dep:** none within plan (depends_on spec-033 at plan level). **Oracle:** no-flag generate → `set(data['constraints_applied names'])` ⊇ the 15 atoms; exactly one objective build + one Solve; `--groups core` applied set equal across the three modes. (review fix — "14" corrected to "15")

### Unit B — Delete dead legacy alignment  (S2; executor: Opus — deletion with archaeology)
**Files:** `constraints/unified.py` (methods + dispatch + groupings), `constraints/stages.py` (ENGINE key sets), `config/season_2026.py` (two PENALTY_WEIGHTS keys), tests asserting engine-subset alignment behaviour (remove/update — grep `_club_alignment`, `by_grade_clubpair_round` in `tests/`: `test_unified_engine.py`, `test_ai_constraints_comprehensive.py`, `test_constraints_realdata.py`, `test_tester_coverage.py`, `test_violation_metadata.py`, and the parity tests `test_constraints_equivalence.py`/`test_groups_full_build_parity.py`). **Dep:** A. **Oracle:** stacked atoms still enforce alignment post-deletion; `_club_alignment_*` gone; registry count unchanged.

### Unit C — Docs + skill  (S1→S2; executor: Sonnet — mechanical, wide)
**Files:** `CLAUDE.md`, `draw/.claude/commands/generate-draw` (in parent project, not worktree), the six docs in DoD 14, help strings in `run.py`, `README.md`, `.github/copilot-skills/hockey-draw-scheduler.md`, `constraints/severity.py` (comment only), `docs/todo/GOALS.md`, `docs/todo/spec-026-regeneration-mode.md`, `docs/todo/spec-033-slack-cleanup-soft-analogue-audit.md`, tests `tests/test_run_cli.py` / `tests/test_run_coverage.py` / `tests/test_groups_cli_wiring.py` / `tests/test_regen_end_to_end.py`. **Dep:** A+B. **Oracle:** no doc/skill/help references `--simple`/`--unified` across the entire worktree (grep `.github/` included); mode model documented consistently. (review fix — file list expanded from grep sweep; 11 additional files confirmed.)

## Doc registry

- `CLAUDE.md` — **(C)** mode model + drop `--simple`; the generate examples become no-flag.
- `draw/.claude/commands/generate-draw` (parent project, not worktree) — **(C)** command template → no-flag default. (review fix — confirmed this path does not exist in the final-form worktree.)
- `docs/operator-ai/AI_OPERATIONS_MANUAL.md`, `docs/operator-ai/SYSTEM_OPERATION.md`, `docs/operator-human/CAPABILITIES.md`, `docs/operator-human/USER_GUIDE.md`, `docs/system/STAGES.md`, `docs/system/SYSTEM_OVERVIEW.md`, `docs/system/REGEN_CONSTRAINTS.md` — **(C)** new three-mode model; remove `--simple`/`--unified`.
- `README.md` — **(C)** generate example at line 22 uses `--simple` → no-flag. (review fix — was missing.)
- `.github/copilot-skills/hockey-draw-scheduler.md` — **(C)** four `--simple` references; update to no-flag + new mode descriptions. (review fix — was missing.)
- `constraints/severity.py` — **(C)** comment on line 15 uses `--simple --relax`; update comment only. (review fix — was missing.)
- `docs/todo/GOALS.md` — **(C)** reference to `--simple` in comment (line 301); update. (review fix — was missing.)
- `docs/todo/spec-026-regeneration-mode.md` — **(C)** DoD line 169 uses `--simple` in a generate command; update. (review fix — was missing; active plan, not done.)
- `docs/todo/spec-033-slack-cleanup-soft-analogue-audit.md` — **(C)** execution protocol references `--simple` at lines 57, 172; update. (review fix — was missing; other active plan.)
- `tests/test_run_cli.py`, `tests/test_run_coverage.py`, `tests/test_groups_cli_wiring.py`, `tests/test_regen_end_to_end.py` — **(C)** update tests that reference `--simple`/`--unified` CLI flag (argparse mock, CLI dispatch, comments). (review fix — all missed; confirmed by grep.)
- `docs/system/CONSTRAINT_INVENTORY.md` — **(B)** note the legacy `_club_alignment_*` engine path deleted; `ClubVsClubAlignment` registry entry retained as tester/name anchor only.
- `docs/todo/00-dependency-tree.md` — **(this plan)** add spec-036; record that spec-035 (e2e single-solve) now sequences after spec-036.

## Out of scope

- **`--core`/`--soft` convenience flags + collapsing `default`/`all`/`production`.** Convenor is happy with `--groups core`/`--groups soft`; not doing the sugar now. Excluded by design.
- **Splitting the `groups` field into stage/selection/regen sub-fields.** A clarity-only refactor; not pursued (convenor chose minimal tidy).
- **Removing the `ClubVsClubAlignment` registry entry / legacy class-name resolution / archived parity classes.** Still needed by the tester + resolver; only the *engine methods* are deleted.
- **spec-033's constraint reworks** — separate plan; spec-036 picks up its final shapes (hence `depends_on`).

## Dependencies

- `depends_on: spec-033`. spec-036's single-solve default must apply the *final* constraint set (TeamConflict→soft, ClubNoConcurrentSlot→soft, ClubGameSpread field cap, bye soft + `{bye_spacing}` group). It also edits `config/season_2026.py` PENALTY_WEIGHTS (shared with spec-033) and `CONSTRAINT_INVENTORY.md`. Start only after spec-033 is `done`.
- **spec-035 (e2e `--groups core` single-solve) must sequence AFTER spec-036** — its raw single-solve depends on the no-flag/single-solve path being correct and complete. Recorded in the dependency tree; spec-035's `depends_on` should add `spec-036` when next edited.
- Within plan: A→B→C serial.

## Risks & blast radius

- **Behaviour change to the default and to `--staged` (intended).** Anyone running no-flag now gets a single full solve (not incremental staged); anyone running `--staged` now gets `DEFAULT_STAGES` incremental (not severity — that moves to `--severity`). Habitual `--simple` users must drop the flag. Surface prominently in CLAUDE.md + the operator docs.
- **`--simple`/`--unified` removal ripples widely** (≈20 files: 7 operator docs + README + copilot skill + CLAUDE.md + severity.py comment + 2 active specs + GOALS.md + 4 test files + generate-draw skill + help). Unit C must catch them all; the Mode B reviewer greps for stragglers (including `.github/`). (review fix — updated count from ≈10 to ≈20 after full grep sweep.)
- **Single-solve completeness depends on `apply_solver_stage` being solve-agnostic.** Confirm `apply_solver_stage` only *applies* constraints (the solve is the caller's), so looping it across all stages without solving is valid; `build_objective` then runs once. (Verified: `run_solver_stages_solve` calls `build_objective`/solve itself after each `apply_solver_stage`.)
- **Deleting `_club_alignment_*` while spec-033 Unit A edits the same methods.** spec-033 lands first (dependency); by the time spec-036 Unit B runs, those methods still exist (spec-033 only stripped their slack). Unit B deletes them outright — confirm no merge straggler re-introduces a slack read.
- **Parity tests** that compared the engine `_club_alignment_*` output to archived classes will break by design — Unit B removes/repoints them; they were testing a path that no longer exists.

## Open Questions

None. The convenor specified the mode model explicitly (no-flag = single full solve; `--staged` = DEFAULT_STAGES staged; `--severity` = severity staged; `--groups`/`--slack` orthogonal). The `--simple`/`--unified` removal follows the forward-only rule (no back-compat alias); flagged to the convenor in the session summary.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Semi-automated: idea→write→review→ready unattended; ready→build requires the user's explicit go-ahead. -->
1. Status must be `ready` (carries a `reviewed:` stamp). Confirm **spec-033 is `done` and merged** (unsatisfied `depends_on` otherwise). Do NOT implement without the user's explicit go-ahead even when `ready`.
2. Stamp `building`, claim `owner`. Orchestrator = Opus.
3. Units serial A→B→C, each own worktree+branch branched from source after the prior merges. Delegate per the unit's executor note; run the unit's severity gates.
4. After each unit, launch `/adversarial` Mode B against this plan's DoD. Route fixes, re-verify. NEVER merge unverified.
5. Per unit: merge → push origin → post-merge verify (no-flag `run.py generate --year 2026` builds + solves the full set; `--staged`/`--severity` build; no `--simple` anywhere) → remove worktree → tick checkbox.
6. When all pass: stamp `done`, move to `docs/todo/done/`, update `docs/todo/00-dependency-tree.md` (drop this node; confirm spec-035 unblocks only after this is done).
