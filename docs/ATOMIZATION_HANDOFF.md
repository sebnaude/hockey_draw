# Hand-off prompt — Atomization implementation

> Paste this entire document into a fresh Claude Code session as the first user message. It is self-contained.

---

You are taking over an in-flight refactor of a hockey-draw constraint solver. All planning is done; you are executing. Work carefully, ship one phase at a time, verify with tests, and report after each phase.

## Repo + branch + worktree

- **Repo:** `C:/Users/c3205/Documents/Code/python/draw` (main worktree)
- **Branch you must work on:** `final-form`
- **Worktree to use:** `C:/Users/c3205/Documents/Code/python/draw-final-form` — **always `cd` into here**, do NOT modify the main worktree (it's on `feat/ai-updates`)
- Push permission requires user approval. Don't push yourself; ask the user.
- The main project doc `CLAUDE.md` (in the worktree root) lists project rules — read it before any code change.

## Read these first (MANDATORY)

1. `C:/Users/c3205/Documents/Code/python/draw-final-form/CLAUDE.md` — project rules
2. `C:/Users/c3205/Documents/Code/python/draw-final-form/docs/ATOMIZATION_PLAN.md` — the full plan with decisions, atom catalog, helper-var catalog, configurable-stages design, constants punch list, generic home-ground design, test patterns, definition of done
3. `C:/Users/c3205/Documents/Code/python/draw-final-form/config/season_2026.py` — actual season config (lock this in your head before editing constraints)

## Current state of the branch (commit `cd8a338`)

- 1211/1212 tests passing (1 skipped). Two slow integration tests (`tests/test_solver_integration.py`, `tests/test_spacing_integration.py`) excluded from quick runs.
- `constraints/unified.py` exists with the 2-stage hard/soft architecture and `SharedVariablePool` (lazy helper-var cache, ~1404 lines). This is your **starting point** for the unified engine; you will replace `SharedVariablePool` with `HelperVarRegistry` in Phase 1 and replace inline constraint application with atom dispatch in Phase 3.
- `constraints/registry.py` exists with 21 entries (constraint name-mapping). Phase 2 extends `ConstraintInfo`.
- `constraints/original.py` (1733 lines) and `constraints/ai.py` (2040 lines) hold legacy combined classes. **Reference-only.** Move to `constraints/archived/` in Phase 7c. The pipeline must be locked against importing them.
- `constraints/archived_equalspacing_original.py` exists at the top level — move it to `constraints/archived/equalspacing_original.py` in Phase 7c.
- `utils.py` (3571 lines) has the 20-phase `validate_game_config` harness, `_get_matching_forced_scopes` (multi-scope FORCED match — the recent bug fix), `validate_draw_keys`, `repair_locked_keys`. `generate_X` returns `(X, conflicts)` — NOT `(X, Y, conflicts)`. Don't reintroduce the Y-dict path.
- `analytics/tester.py` (2495 lines) has `_check_forced_games`, `_check_blocked_games`, dynamic per-club ClubGameSpread overlap bound. Phase 7a-bis adds the per-club/per-type breakdown.
- `tests/fixtures/draw_2026_first6weeks.json` exists. Phase 7a creates `tests/fixtures/violations/`.

## Decisions already locked (do NOT re-litigate)

| # | Decision |
|---|---|
| 1 | Atom names use `PHLConcurrencyAtBroadmeadow`-style descriptive names. No `Atom` suffix. |
| 2/3 | Legacy `original.py` + `ai.py` move to `constraints/archived/`. Pipeline imports forbidden. Test enforces. |
| 4 | `ClubDayOpponentMatchup` atom matches `original.py` behavior (supports `{'date': ..., 'opponent': 'OppClub'}`). |
| 5 | Per-club home-venue config key is `AWAY_VENUE_RULES`. |
| 6 | Tests use both static fixtures (one per atom-violation) AND programmatic per-test construction. Plus per-club / per-type breakdown in `ViolationReport`. |
| 7 | `SOLVER_STAGES` is a config-driven list of `{name, description, atoms, ...}` dicts. User picks how many stages and which atoms in each. CLI flags `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`. |

## Critical project facts you MUST remember

These are common pitfalls — they're in CLAUDE.md too but are easy to get wrong:

1. **Variable key is an 11-tuple** `(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)`. `team1` is alphabetically first. Home/away is determined by `field_location`, NOT by team1/team2 position.
2. **Dummy keys are 4-tuples** `(team1, team2, grade, index)`. Skip them via `len(key) < 11` or `not key[3]`.
3. **`generate_X` signature: `(X, conflicts)`** — final-form does NOT use the `(X, Y, conflicts)` 3-tuple; dummies are handled via `SharedVariablePool` (and will be by `HelperVarRegistry` after Phase 1).
4. **A FORCED variable can match multiple scopes.** Use `_get_matching_forced_scopes()` (returns list), not `_check_forced_game_status()` (returns first match only — back-compat wrapper only).
5. **Aggregate per-team home/away constraint was removed deliberately.** Per-pair balance only. CLAUDE.md notes "by design".
6. **PHL locked-week HACKs in `constraints/original.py`** (lines ~242-301) are explicit user-flagged tech debt. They will be REPLACED by proper atoms (`BroadmeadowFridayCount`, `GosfordFridayCount`, `MaitlandFridayCount`, `PHLRoundOnePlay`) with FORCED-aware adjusters that handle locked-week math cleanly.
7. **Maitland Sunday slot is 13:30 (not 13:00)** — `season_2026.py` uses 13:30. If you see 13:00 anywhere, it's stale.
8. **`home_field_map` already exists** at `data['home_field_map']` (e.g. `{'Maitland': 'Maitland Park', 'Gosford': 'Central Coast Hockey Park'}`). Defaults to Newcastle International Hockey Centre (Broadmeadow) for unlisted clubs.
9. **Solver runs are LONG (hours/days).** Never run `python run.py generate ...` synchronously. Use background jobs for solver tests; for unit tests build small fixtures and run for ≤30s.
10. **The `--ai` CLI flag is being deprecated.** After atomization there's no original/AI distinction. Don't add new code paths that use `args.ai`.

## How to work

Iterate one phase at a time. Order from `ATOMIZATION_PLAN.md`:

```
0 (inventory) ──┐
                ├─→ 1 (helper-var registry) ─→ 3 (atomize) ─→ 7c (move legacy)
2 (constraint registry extend) ─┘                      │
                                                        ↓
                            4 (FORCED/BLOCKED adjusters)
                                                        ↓
5 (constants migration) ────────────────────────────────┤
6 (generic home-ground) ────────────────────────────────┤
                                                        ↓
7a (tests) ─ 7b (configurable stages) ─ 7d (docs)
```

**Per-phase workflow:**
1. Re-read the relevant phase section in `ATOMIZATION_PLAN.md`.
2. Plan the work (use `TaskCreate` to track sub-tasks within the phase).
3. Implement with `Edit`/`Write`. NEVER touch `constraints/archived/` (once that exists) — those files are reference-only.
4. Run the test bar for that phase. The full quick test command is:
   ```
   cd /c/Users/c3205/Documents/Code/python/draw-final-form
   timeout 180 python -m pytest tests/ --ignore=tests/test_solver_integration.py --ignore=tests/test_spacing_integration.py -q
   ```
   Must pass at ≥1212 (current baseline) before you commit.
5. Commit on `final-form` with a descriptive message. Use `git commit --no-verify` ONLY if the user explicitly authorizes (don't skip pre-commit hooks proactively).
6. Report to the user: what changed (file:line refs), any FORCED/BLOCKED adjuster formulas added (let user sanity-check), any constants that resisted migration (with reason), any atom that doesn't have full parity with its legacy version (with the gap).
7. Wait for user approval before starting the next phase.

**Do NOT:**
- Push without user approval (push permission is denied; ask).
- Modify `feat/ai-updates` (work only on `final-form`).
- Modify the main worktree at `C:/Users/c3205/Documents/Code/python/draw` (use the `final-form` worktree).
- Invent new constraint behavior. You're re-shaping existing logic into atoms; behavioral changes get raised separately for sign-off first.
- Use `unittest.mock`, `monkeypatch`, or any mocking on solver/data dicts in tests. Real CP-SAT models with sampled data only.
- Add docstrings beyond what was there. Don't over-comment. CLAUDE.md is strict on this.
- Run the full solver (it takes hours). Use small fixtures for tests.
- Touch the open Excel files at the repo root (`2026 Season Draw V*.xlsx` etc.) — they're gitignored as drafts.

## Phase 0 starter

Begin with Phase 0 — pure documentation, no code:

1. Open `constraints/registry.py` and read all 21 `ConstraintInfo` entries.
2. Open `constraints/original.py`, scan every class, extract its actual behavior (not docstring).
3. Open `constraints/ai.py`, do the same.
4. Read `tests/test_constraint_registry.py` for what's expected.
5. Write `docs/CONSTRAINT_INVENTORY.md` — one table:
   - Columns: canonical name | source file:class | one-line actual behavior | severity | slack key | atom-target name(s) when split
   - For multi-idea constraints, one row per atom showing its target name from the plan.
6. No code changes. Commit `docs: add constraint inventory (Phase 0)`.
7. Report back with: file path, table line count, anything you found in the code that didn't match the plan's assumptions.

After Phase 0 is approved, proceed to Phase 1 (Helper-Var Registry).

## Useful commands

```bash
# always work from the final-form worktree
cd /c/Users/c3205/Documents/Code/python/draw-final-form

# quick test run
timeout 180 python -m pytest tests/ --ignore=tests/test_solver_integration.py --ignore=tests/test_spacing_integration.py -q

# check what's in unified.py around helper vars
grep -n "SharedVariablePool\|class.*VarPool\|self.pool" constraints/unified.py

# inspect the registry
python -c "from constraints.registry import CONSTRAINT_REGISTRY; [print(n, '->', i.severity_level, i.tester_check_methods) for n, i in CONSTRAINT_REGISTRY.items()]"

# load season data and inspect
python -c "from config import load_season_data; d = load_season_data(2026); print('teams:', len(d['teams']), 'timeslots:', len(d['timeslots']), 'forced:', len(d['forced_games']), 'blocked:', len(d['blocked_games']))"

# count constraint classes in original.py / ai.py
grep -c "^class.*Constraint" constraints/original.py constraints/ai.py
```

## When you finish a phase

Report in this shape (copy/paste template):

```
## Phase N — <name> — DONE

**Commits:** <hash> "<msg>"

**Files touched:**
- path/to/file.py (LOC change: +X / -Y) — what changed in one line
- ...

**Tests:** <N passing / M total>

**FORCED/BLOCKED adjusters added (for user sanity-check):**
- AtomName: <formula in plain English> — example: "expected_meetings = total_meetings - forced_off_sunday - blocked_on_sunday"
- ...

**Parity gaps (atom vs legacy):**
- <none> OR
- AtomName: <gap explained, why it's intentional>
- ...

**Constants migrated:**
- ConstantName: from `path/to/file.py:line` to `config_key.subkey`
- ...

**Open questions for next phase:**
- ...

Ready for Phase N+1?
```

Work carefully. Ask before doing anything risky. Commit small. Verify often.
