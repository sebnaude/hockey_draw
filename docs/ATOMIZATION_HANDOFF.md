# Hand-off prompt — Atomization, FINAL-FINAL PUSH (status: ALL ✅)

> **Update (commits `dd76a79` + `4599f01` + `0140495` on `final-form`):**
> Every remaining phase is now shipped. Test bar: **1285 passed, 1
> skipped** (down from the 1383 mid-push checkpoint because two test
> files dedicated to deleted STAGES infrastructure were removed in the
> 7c-completion commit). See the "What landed in this push" appendix at
> the bottom of this document for the full breakdown. The only
> remaining open follow-up is rewiring `run.py diagnose` to drive
> `InfeasibilityResolver` from atom canonical names — the rest of the
> spec is fulfilled.
>
> Original handoff text follows — preserved for reference.

> Paste this entire document into a fresh Claude Code session as the first
> user message. It is self-contained.
>
> **This is the genuinely-last push.** Phases 0, 1, 2, 3a, 3b, 3c, 4, 5, 6,
> 7a (initial delivery), 7b (foundation), and 7d (initial sweep) have all
> shipped on `final-form`. Three things remain:
>
> 1. **Phase 7b full pipeline rewire** — wire `main_staged.py` to dispatch
>    from `data['solver_stages']`, delete the legacy `STAGES` and
>    `STAGES_AI` dicts, and add the `--stages-config` / `--stage-only` /
>    `--skip-stage` / `--list-stages` CLI flags.
> 2. **Phase 7c** — move `constraints/{original,ai,archived_equalspacing_original}.py`
>    into `constraints/archived/`, update every prod import site to go via
>    the registry, add `tests/test_no_legacy_imports.py`, and remove the
>    `--ai` CLI flag.
> 3. **Phase 7a expansion** — add the remaining ≥4 violation fixtures so
>    the suite has at least 8 (per the original spec) and add atom
>    population of `Violation.affected_clubs` / `Violation.metric_value` so
>    `ViolationReport.breakdown.soft_pressure` has real data to roll up.
>
> The infrastructure for all three is already in place. Everything left is
> mechanical wiring + careful import surgery + a few more JSON fixtures.
> Do all three. Don't pause for sign-off — the spec is locked. If you find
> a real bug in the spec, flag it in your final summary, not mid-work.

---

## Repo + branch + worktree

- **Repo:** `C:/Users/c3205/Documents/Code/python/draw` (main worktree —
  DO NOT touch; on `feat/ai-updates`)
- **Branch you must work on:** `final-form`
- **Worktree to use:** `C:/Users/c3205/Documents/Code/python/draw-final-form`
  — **always `cd` into here**
- **Push permission requires user approval.** Don't push yourself; ask
  the user when everything is green.
- The main project doc `CLAUDE.md` (in the worktree root) lists project
  rules — read it before any code change.

## Read these first (MANDATORY)

1. `CLAUDE.md` — project rules (atomization-aware as of Phase 6/7a/7b)
2. `docs/ATOMIZATION_PLAN.md` — phase status table; everything except 7b
   full / 7c shows ✅
3. `docs/CONSTRAINT_INVENTORY.md` — single source of truth for constraint
   → atom mapping. Now lists Phase-6 alias entries.
4. `docs/HELPER_VARS.md` — `HelperVarRegistry` API
5. `docs/COUNT_ADJUSTERS.md` — Phase 4 adjuster formulas + statuses (all ✅)
6. `docs/STAGES.md` — `SOLVER_STAGES` schema + validation API (foundation)
7. `docs/HARNESS.md` — end-to-end pipeline reference
8. `config/season_2026.py` — actual season config

## Current state of `final-form`

Quick test bar (no slow integration tests):
```
cd /c/Users/c3205/Documents/Code/python/draw-final-form
timeout 240 /c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe -m pytest tests/ \
  --ignore=tests/test_solver_integration.py \
  --ignore=tests/test_spacing_integration.py -q
```

**Baseline at start: 1352 passed, 1 skipped.** Don't ship a phase that drops
below this; every phase below should add tests and lift it further.

(The venv lives in the *main* worktree `draw/.venv`, not in
`draw-final-form/.venv`. Use the absolute path above.)

### Commits in the chain so far

| Hash | Phase | Summary |
|---|---|---|
| `6e16d14` | 0 | `docs: add constraint inventory` |
| `244f8cd` | 1 | `feat(constraints): extract HelperVarRegistry from SharedVariablePool` |
| `c64c1d4` | 2 | `feat(constraints): extend ConstraintInfo for atomization` |
| `535cac3` | 5 | `feat(config): migrate hardcoded constraint constants to CONSTRAINT_DEFAULTS` |
| `48f5222` | 6 prep | `feat(config): add AWAY_VENUE_RULES skeleton for generic home-ground` |
| `1956608` | 3a | `feat(constraints): atomize PHLAndSecondGradeTimes into 8 atoms` |
| `e9bf5a7` + `5cfae6c` | 3a retraction | per-venue Friday counts moved to FORCED_GAMES |
| `4f0777c` + `4ed0abf` | docs | sweep |
| `0cf78e6` | 3b | `feat(constraints): atomize ClubDayConstraint into 5 atoms` |
| `8d2934d` | 3c | `feat(constraints): atomize ClubVsClubAlignment into 4 atoms` |
| `08f11be` | docs | hash backfill |
| `1521c9b` | 4 framework | `feat(constraints): wire FORCED/BLOCKED count-adjuster framework` |
| `8c0e351` | docs | hash backfill |
| `d450ba5` | docs | full sweep + final-push handoff prompt |
| `4ad6add` | 4 (full) | `feat(constraints): implement Phase 4 FORCED/BLOCKED count adjusters` (4 implementations + EqualGames no-op note + 18 tests) |
| `67474f4` | 6 + 7a + 7b found. + 7d (initial) | Phase 6 generic non-default-home rename, `ViolationBreakdown` + 4 violation fixtures, `SOLVER_STAGES` config + validation API, docs sweep |
| `0e16306` | docs | full sweep + final-final push handoff prompt |
| `dd76a79` | 7b full | `feat(solver): config-driven SOLVER_STAGES dispatch + CLI flags` |
| `4599f01` | 7a expansion + 7c partial | `test(analytics): expand violation fixtures + populate atom metadata + lockdown skeleton` |

### What's already in place

- `constraints/atoms/` — 14 atoms across 3 groups: PHL (5), ClubDay (5),
  ClubVsClub (4) + shared helpers + `base.py` + `__init__.py` (+ Phase-4
  adjusters in `_adjusters.py`).
- `constraints/registry.py` — 37 entries (21 originals + 14 atoms + 2
  Phase-6 generic aliases), `run_count_adjusters(data)` plumbing,
  `HELPER_VAR_CATALOG`. Adjusters wired for `EqualMatchUpSpacing`,
  `MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`,
  `ClubVsClubCoincidence`.
- `constraints/unified.py` — atom dispatch wired for all 3 groups; legacy
  `_phl_times_*`, `_club_day_*`, `_club_alignment_*` retained as
  parity-reference (not invoked in stage 1/2). `_maitland_grouping_*` /
  `_away_maitland_*` iterate `home_field_map` and read `AWAY_VENUE_RULES`.
- `constraints/helper_vars.py` — `HelperVarRegistry` (declarative + pool API)
- `constraints/stages.py` — Phase 7b foundation: `load_solver_stages` /
  `validate_solver_stages` / `list_stages`.
- `config/defaults.py` — perennial `CONSTRAINT_DEFAULTS`, `AWAY_VENUE_RULES`
  (Maitland inherits from defaults, Gosford has explicit overrides),
  `DEFAULT_STAGES` (Phase 7b).
- `tests/atoms/` — atom-level tests + 3 fixtures (PHL, ClubDay, CvC)
- `tests/test_count_adjusters.py` — 7 framework tests
- `tests/test_phase4_adjusters.py` — 18 adjuster math + integration tests
- `tests/test_phase6_generic_home.py` — 6 Phase-6 tests
- `tests/test_violation_breakdown.py` — 7 ViolationBreakdown tests
- `tests/test_violation_fixtures.py` — walks `tests/fixtures/violations/`
- `tests/test_solver_stages.py` — 10 SOLVER_STAGES validation tests
- `tests/fixtures/violations/` — 4 starter fixtures (NoDoubleBookingTeams,
  NoDoubleBookingFields, ClubGradeAdjacency, MaxMaitlandHomeWeekends)
- `docs/` — inventory, plan, helper-vars, count-adjusters, stages, harness,
  forced-games-as-count-rules, perennial rules

### What's still in place that needs to go

- `constraints/original.py` (~1733 lines) and `constraints/ai.py` (~2040
  lines) — legacy combined classes. Pipeline must NOT import them in prod
  by end of Phase 7c.
- `constraints/archived_equalspacing_original.py` at the top level — also
  moves to `constraints/archived/` in 7c.
- Hardcoded `STAGES` and `STAGES_AI` dicts in `main_staged.py` — Phase 7b
  full replaces with config-driven `SOLVER_STAGES`.
- `--ai` CLI flag in `run.py` — Phase 7c removes it.

## Decisions already locked (do NOT re-litigate)

| # | Decision |
|---|---|
| 1 | Atom names use `PHLConcurrencyAtBroadmeadow`-style descriptive names. No `Atom` suffix. |
| 2/3 | Legacy `original.py` + `ai.py` move to `constraints/archived/`. Pipeline imports forbidden. Test enforces. |
| 4 | `ClubDayOpponentMatchup` atom matches `original.py` behaviour. DONE in 3b. |
| 5 | Per-club home-venue config key is `AWAY_VENUE_RULES`. DONE in 6. |
| 6 | Tests use both static fixtures (one per atom-violation) AND programmatic per-test construction. Plus per-club / per-type breakdown in `ViolationReport`. DONE in 7a (initial). |
| 7 | `SOLVER_STAGES` is a config-driven list of `{name, description, atoms, ...}` dicts. CLI flags `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`. Foundation DONE in 7b; full wire-up THIS SESSION. |
| 8 | Phase 4 adjuster formulas in `docs/COUNT_ADJUSTERS.md` are **approved spec** — implement them as written. DONE in 4. |

## Critical project facts you MUST remember

These are common pitfalls — they're in `CLAUDE.md` too but easy to get wrong:

1. **Variable key is an 11-tuple** `(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)`. Home/away by `field_location`, NOT by team1/team2 position.
2. **Dummy keys are 4-tuples** `(team1, team2, grade, index)`. Skip them via `len(key) < 11` or `not key[3]`.
3. **`generate_X` signature: `(X, conflicts)`**.
4. **A FORCED variable can match multiple scopes.** Use `_get_matching_forced_scopes()` (returns list).
5. **Solver runs are LONG (hours/days).** Never run `python run.py generate ...` synchronously. Use small fixtures for unit tests.
6. **`HelperVarRegistry.get(key)`** returns the cached pool var (or `None`). Use `registry.get_declared(kind, key)` for the declarative API.
7. **`coincide` BoolVars** for ClubVsClub registered under prefixes `cvc_coincide` (lower-grade) and `cvc_phl_btb_coincide` (PHL/2nd).
8. **Phase-4 count adjustments** live at `data['count_adjustments'][canonical_name]`. Engine populates them once in `build_groupings()` before atom `apply()`. Atoms read by canonical name.
9. **Phase-6 generic non-default-home** — `home_field_map.keys()` drives iteration. `AWAY_VENUE_RULES[club]` per-club tuning with `None` meaning "skip". Falls back to `CONSTRAINT_DEFAULTS`.

## How to work

**Per-phase workflow:**
1. Re-read the relevant phase section in `ATOMIZATION_PLAN.md` and the
   detail below.
2. Plan with `TaskCreate` to track sub-tasks within the phase.
3. Implement with `Edit`/`Write`. NEVER touch `constraints/archived/` (once
   that exists) — those files are reference-only.
4. Run the test bar:
   ```
   cd /c/Users/c3205/Documents/Code/python/draw-final-form
   timeout 240 /c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe -m pytest tests/ \
     --ignore=tests/test_solver_integration.py --ignore=tests/test_spacing_integration.py -q
   ```
   Must pass at the running baseline (start: 1352) before you commit each phase.
5. Commit on `final-form` with a descriptive message. Do NOT use
   `--no-verify` unless explicitly authorized.
6. Update **all** of these docs at end of each phase:
   - `docs/ATOMIZATION_HANDOFF.md` — phase status + commit table
   - `docs/ATOMIZATION_PLAN.md` — phase status table
   - `docs/CONSTRAINT_INVENTORY.md` — atom annotations if relevant
   - `docs/COUNT_ADJUSTERS.md` — adjuster status table (if relevant)
   - `docs/STAGES.md` — fill in CLI flags + dispatch wiring section after 7b
   - `CLAUDE.md` — file structure, pitfalls, anything that drifts
   - `README.md` — if dir structure changes
7. Move on. Don't pause for sign-off. Final report at the very end.

**Do NOT:**
- Push without user approval.
- Modify `feat/ai-updates` (work only on `final-form`).
- Modify the main worktree at `C:/Users/c3205/Documents/Code/python/draw`.
- Invent new constraint behavior.
- Use `unittest.mock`, `monkeypatch`, or any mocking on solver/data dicts
  in tests. Real CP-SAT models with sampled data only.
- Run the full solver. Use small fixtures.

---

# THE WORK

## Phase 7b — Full pipeline rewire + CLI flags

Foundation already shipped (`67474f4`):
- `config/defaults.py::DEFAULT_STAGES` — 5 stages keyed by canonical atom name.
- `constraints/stages.py` — `load_solver_stages` / `validate_solver_stages`
  / `list_stages`.
- 10 tests in `tests/test_solver_stages.py`.

### What's left

#### 1. Wire `main_staged.py` to read `data['solver_stages']`

`main_staged.py` currently has hardcoded `STAGES` and `STAGES_AI` dicts
mapping stage names to lists of constraint *classes*. The new path:

- At the top of `main_staged()` (and any other entry point), call
  `data['solver_stages'] = load_solver_stages(season_config)` if not set.
- Replace the `STAGES[stage_name]` lookup with a loop over
  `data['solver_stages']`. Per stage:
  - Look up each canonical atom name via `CONSTRAINT_REGISTRY[name]`.
  - For atomised clusters (PHL_TIMES_ATOMS, CLUB_DAY_ATOMS, CLUB_VS_CLUB_ATOMS,
    `_maitland_grouping_*`, `_away_maitland_*`, etc.) the dispatch goes
    through `UnifiedConstraintEngine`. The simplest mapping: keep using the
    `_skip` set on the engine — if an atom isn't in any stage being
    applied, add it to `_skip`. Then `apply_stage_1_hard` / `apply_stage_2_soft`
    naturally do the right thing.
  - For non-atomised constraints that still go through legacy classes:
    instantiate the class via the canonical name → solver class lookup.
- After 7c moves the legacy classes, this dispatch becomes "atom names
  only — fail fast on missing".

#### 2. Add CLI flags in `run.py`

```
--stages-config <path>   # JSON file with a custom stage list (replaces in-config)
--stage-only NAME        # run only the named stage
--skip-stage NAME        # skip the named stage (can be passed multiple times)
--list-stages            # print the configured stages and exit
```

`--list-stages` calls `list_stages(load_solver_stages(season_config))` and
exits cleanly. The other three modify `data['solver_stages']` before
solving.

#### 3. Add `_validate_stages` phase to `validate_game_config`

A new phase in the 20-phase config validation harness:

```python
def _validate_stages(data, warnings, fatals):
    stages = data.get('solver_stages')
    if stages is None:
        return  # default will be loaded later
    errors = validate_solver_stages(stages)
    for err in errors:
        fatals.append(f"solver_stages: {err}")
```

#### 4. Delete the hardcoded `STAGES` and `STAGES_AI` dicts in `main_staged.py`

Once the loop reads from `data['solver_stages']`, those dicts have no
callers. Drop them along with their imports.

### Tests

Add to `tests/test_solver_stages.py` (or a new `tests/test_main_staged_stages.py`):
- `--list-stages` prints the expected stage names without solving.
- `--stage-only NAME` reduces the stage list to one stage.
- `--skip-stage NAME` removes a stage.
- `--stages-config path/to/custom.json` overrides the in-config stages.
- `_validate_stages` rejects unknown atom names.
- A tiny end-to-end test: tiny fixture, run all stages, assert at least one
  CP-SAT constraint added per stage. (Use the existing `tests/atoms/` fixtures.)

### Phase 7b commit

`feat(solver): config-driven SOLVER_STAGES + CLI flags (Phase 7b full)`

---

## Phase 7c — Move legacy + remove `--ai`

### Move

- `constraints/original.py` → `constraints/archived/original.py`
- `constraints/ai.py` → `constraints/archived/ai.py`
- `constraints/archived_equalspacing_original.py` → `constraints/archived/equalspacing_original.py`
- New `constraints/archived/__init__.py` (empty or with `__all__ = []`)
- New `constraints/archived/README.md` explaining: "These are
  pre-atomization implementations kept for historical reference. Do NOT
  import in production code; the pipeline is locked against this. Use the
  atoms in `constraints/atoms/` instead."

### Update import sites (~20 files)

Run this grep to find them:
```
cd /c/Users/c3205/Documents/Code/python/draw-final-form
git grep -l "from constraints.original\|from constraints.ai\|from constraints.archived_equalspacing_original\|from constraints import.*original\|from constraints import.*ai"
```

Approximate list (verify before editing):
- `main_staged.py` — biggest one. After Phase 7b dispatch is config-driven,
  most legacy imports here should disappear naturally. Anything that
  remains needs to come from the registry.
- `constraints/__init__.py` — keep ONLY: atoms, `UnifiedConstraintEngine`,
  registry. No re-export of archived modules.
- `constraints/soft.py`, `constraints/resolver.py` — port to use atoms or
  the registry.
- `constraints/unified.py` — has one or two reference imports
  (`_normalize_preference_no_play`); inline-copy or move that helper.
- `analytics/*` — registry-only.
- `run.py` — registry-only.
- `scripts/*.py` — most can be deleted (they target legacy). Keep only
  ones still useful; update those to use atoms.
- Tests that target legacy classes (`test_constraints_comprehensive.py`,
  `test_ai_constraints_comprehensive.py`, `test_constraints_equivalence.py`,
  `test_constraints_ai.py`, `test_best_timeslot_stacking.py`,
  `test_severity_relaxation.py`, `test_infeasibility_resolver.py`,
  `test_spacing_integration.py`) — either:
  - Update to import from `constraints.archived.*` (allowed for tests; the
    no-legacy-imports test only fires on prod code).
  - Mark `pytest.skip("legacy class — archived in Phase 7c")` if the test
    has no atom-level equivalent.
  - Delete if obviously obsolete (parity tests against atoms make it
    redundant).

### Remove `--ai` CLI flag

- `run.py`: drop the `--ai` flag.
- `main_staged.py`: drop `args.ai` usage and the `STAGES_AI` dict (already
  scheduled for delete in 7b).
- Update tests that reference `--ai`.

### Add `tests/test_no_legacy_imports.py`

```python
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALLOWED = (REPO / 'constraints' / 'archived', REPO / 'tests')
PATTERN = re.compile(
    r'^\s*from\s+constraints\.(original|ai|archived(_equalspacing_original)?|archived\.\w+)\s+import|'
    r'^\s*import\s+constraints\.(original|ai|archived(_equalspacing_original)?|archived\.\w+)',
    re.MULTILINE,
)


def _iter_py_files():
    for path in REPO.rglob('*.py'):
        if any(str(path).startswith(str(allowed)) for allowed in ALLOWED):
            continue
        if 'site-packages' in str(path) or '.venv' in str(path):
            continue
        yield path


def test_no_prod_module_imports_archived():
    offenders = []
    for path in _iter_py_files():
        text = path.read_text(encoding='utf-8', errors='ignore')
        if PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f'These prod modules still import archived constraints: {offenders}'
```

### Phase 7c commit

`refactor: archive legacy constraint classes; remove --ai (Phase 7c)`

---

## Phase 7a expansion — more violation fixtures + populate atom metadata

### Goal

The Phase 7a initial delivery has 4 fixtures + the breakdown framework. The
spec asks for ≥8 fixtures covering distinct atom violation paths, plus
atoms populating `Violation.affected_clubs` and `Violation.metric_value` so
the soft-pressure rollups have real data.

### Fixtures to add (target ≥8 total)

Already in `tests/fixtures/violations/`:
- `no_double_booking_teams.json`
- `no_double_booking_fields.json`
- `club_grade_adjacency.json`
- `maitland_back_to_back.json`

Add at least 4 more. Each is a tiny hand-crafted JSON — use the existing
fixtures as templates. Required `_violations` field at top.

Suggested set:
- `club_game_spread_overlap.json` — clubs with overlapping games beyond
  limit. `_violations: ["ClubGameSpread"]`
- `phl_friday_overcount.json` — too many PHL Friday Broadmeadow games
  (FORCED_GAMES check). `_violations: ["ForcedGames"]`
- `home_away_imbalance.json` — Maitland team plays one opponent 3x at home
  / 0x away. `_violations: ["FiftyFiftyHomeAway"]`
- `club_day_split_field.json` — club day games on two fields.
  `_violations: ["ClubDayConstraint"]`
- `club_vs_club_field_excess.json` — coinciding club-pair on 3 fields.
  `_violations: ["ClubVsClubAlignment"]`
- `phl_2nd_no_back_to_back.json` — PHL/2nd Sunday games not back-to-back
  same field. `_violations: ["ClubVsClubAlignment"]`

`tests/test_violation_fixtures.py::test_violation_fixtures_present` already
asserts ≥4. Update to ≥8 once the fixtures land.

### Populate `Violation.affected_clubs` / `metric_value`

Update the violations created by these tester methods to fill in the new
fields:

- `_check_club_game_spread` → `affected_clubs=[club]`, `metric_value=overlap_count`
- `_check_maitland_back_to_back` → `affected_clubs=[club]`, `metric_value=window_sum`
- `_check_maitland_away_clubs_limit` → `affected_clubs=list(clubs)`, `metric_value=len(clubs)`
- `_check_club_grade_adjacency` → `affected_clubs=[club]`
- `_check_fifty_fifty_home_away` → `affected_clubs=[club_of_team]`,
  `metric_value=abs(home - away)`
- `_check_club_vs_club_alignment` → `affected_clubs=[c1, c2]`,
  `metric_value=field_count_or_deficit`

Add tests in `tests/test_violation_breakdown.py` that load the new
fixtures, run `report.breakdown`, and assert the rollups have the right
shape.

### Phase 7a expansion commit

`test(analytics): expand violation fixtures + populate atom metadata (Phase 7a expansion)`

---

# WHEN YOU'RE DONE

Run the test bar one final time. Confirm:
- All Phase 7c imports clean (`tests/test_no_legacy_imports.py` green).
- Test count substantially higher than 1352 (expect 1380+ after 7b/7c/7a
  expansion).
- No skipped tests other than the original 1 (or a small handful explicitly
  marked legacy/archived in Phase 7c).

Then summarise to the user:

```
## Atomization complete — every phase ✅

**Final commit chain (this session):** <hashes>

**Test bar:** 1352 → <final> passed, <N> skipped

**Files touched (totals):**
- New atoms: <N>
- New tests: <N>
- Renamed constraints: <list>
- Files moved to constraints/archived/: 3
- Files deleted: <list>

**Phases 7b full / 7c / 7a expansion — all done.** Ready for push approval.

Open questions / things flagged during the work:
- ...
```

Then ask the user for push approval before `git push`.

Work carefully. Commit small per-phase. Verify often. Don't pause for
sign-off — the spec is locked. If you find a real bug in the spec, flag it
in your final summary, not mid-work.

---

# What landed in this push (commits `dd76a79`, `4599f01`)

## Phase 7b — full pipeline rewire ✅

**Commit `dd76a79`** — `feat(solver): config-driven SOLVER_STAGES dispatch + CLI flags`

- New `apply_solver_stage()` dispatcher in `constraints/stages.py`. Maps
  each atom canonical name to either an engine skip-key
  (`atom_to_engine_key()`) or a legacy solver class (`_resolve_solver_class()`).
  Threads `applied_engine_keys` + `applied_atoms` across stages so a
  cluster's atoms don't get re-added.
- New `StagedScheduleSolver.run_solver_stages_solve()` drives the
  per-stage loop using the dispatcher. Applies hints between stages,
  saves checkpoints per stage.
- `main_staged()` defaults to the new path. `severity_staged=True` keeps
  using the legacy `STAGES_SEVERITY[_AI]` dict (unchanged).
- `solver_stages` parameter added to `main_staged()`. The CLI passes
  `--stages-config` / `--stage-only` / `--skip-stage` through this
  parameter.
- `run.py generate` flags: `--stages-config FILE`, `--stage-only NAME`,
  `--skip-stage NAME` (repeatable), `--list-stages`. Resolution helper
  `_resolve_solver_stages` validates via `validate_solver_stages` and
  exits non-zero on bad input.
- Phase 22 `_validate_stages` step in `utils.py::validate_game_config`.
- 18 new tests in `tests/test_solver_stages_dispatch.py`.

## Phase 7a expansion ✅

**Commit `4599f01`** — `test(analytics): expand violation fixtures + populate atom metadata + lockdown skeleton`

- 4 new violation fixtures in `tests/fixtures/violations/`
  (8 total — meets the spec's ≥8 bar): `away_at_maitland_overflow`,
  `club_game_spread_overlap`, `club_vs_club_non_coincident`,
  `home_away_imbalance`. The threshold in
  `test_violation_fixtures_present` is now ≥8.
- Tester populates `Violation.affected_clubs` + `metric_value` for:
  `_check_club_game_spread`, `_check_maitland_back_to_back`,
  `_check_maitland_away_clubs_limit`, `_check_club_grade_adjacency`,
  `_check_fifty_fifty_home_away`, `_check_club_vs_club_alignment`.
- 9 new tests in `tests/test_violation_metadata.py` covering the
  populated metadata + breakdown rollups (`by_club`, `by_type`,
  `soft_pressure`).

## Phase 7c — completion ✅

**Commit `0140495`** — `refactor: archive legacy constraint classes; remove --ai (Phase 7c complete)`

- `constraints/{original,ai,archived_equalspacing_original}.py` moved
  to `constraints/archived/{original,ai,equalspacing_original}.py` via
  `git mv`.
- `_normalize_preference_no_play` lifted out of `original.py` into
  `utils.py` (as `normalize_preference_no_play`). The 3 prod consumers
  (`constraints/{soft,unified}.py` + the archived AI variant) now pull
  from `utils`, so no prod module depends on `constraints.archived.*`.
- `constraints/__init__.py` slimmed to severity + resolver helpers.
  Legacy class re-exports are gone.
- `main_staged.py`: `STAGES` / `STAGES_AI` / `STAGES_UNIFIED` /
  `STAGES_SEVERITY[_AI]` dicts deleted. `run_staged_solve` is now a
  thin shim over `run_solver_stages_solve`. Severity dispatch goes
  through `severity_solver_stages()` (built from the registry).
  `main_simple` legacy path (which iterated those dicts) routes
  through `_main_simple_unified`. `--unified` is now a no-op.
- `run.py`: `--ai` flag removed from `generate`, `list-constraints`,
  and `diagnose`. `run_list_constraints` rewritten to print
  SOLVER_STAGES atoms via the registry. `run_diagnose` is deprecated
  with a clear stub message — see "Open follow-up" below.
- `constraints/stages.py`: new `severity_solver_stages()` helper.
- All tests that relied on `from constraints.original` /
  `from constraints.ai` / legacy class re-exports updated to use
  `constraints.archived.*`.
- `tests/test_no_legacy_imports.py` tightened: pattern now forbids
  `constraints.original`, `constraints.ai`,
  `constraints.archived_equalspacing_original`, AND
  `constraints.archived.*` in prod modules. Tests + the
  archived package itself remain exempt.
- Two legacy test files deleted (entirely about removed STAGES dicts):
  `tests/test_main_staged_coverage.py`,
  `tests/test_severity_staged.py`. Four obsolete diagnostic scripts
  deleted.

## Open follow-up

`run.py diagnose` is currently a stub that prints workaround
instructions and exits non-zero. The `InfeasibilityResolver` still
operates on constraint *classes*, but the diagnose CLI was wired to the
deleted `STAGES`/`STAGES_AI` dicts. Re-porting it to drive the resolver
from atom canonical names (probably via a small adapter that resolves
canonical names through the registry) is the only spec item left. The
shape of the rewrite:

```python
from constraints.stages import load_solver_stages
from constraints.registry import CONSTRAINT_REGISTRY

stages = load_solver_stages({})
stage = next(s for s in stages if s['name'] == args.stage)
atoms = [CONSTRAINT_REGISTRY[a] for a in stage['atoms']]
# ... feed `atoms` into a registry-aware InfeasibilityResolver variant ...
```

**Test bar across the chain: 1216 → 1352 → 1383 → 1285 passed, 1 skipped.**
The dip at the end is the ~120 tests removed alongside the deleted
STAGES dict test files; those tests were entirely about infrastructure
that no longer exists.
