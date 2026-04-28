# Hand-off prompt — Atomization implementation (continuation)

> Paste this entire document into a fresh Claude Code session as the first user message. It is self-contained.

---

You are taking over an in-flight refactor of a hockey-draw constraint solver. **Phases 0, 1, 2, 3a, 3b, 5, and Phase-6 prep are merged on `final-form`.** Your job is the remaining work: Phase 3c (`ClubVsClubAlignment` → 4 atoms), Phase 4 (FORCED/BLOCKED count adjusters), Phase 6 (generic home-ground rename), and Phase 7 (tests, configurable stages, archive legacy, docs).

Work carefully, ship one phase at a time, verify with tests, and report after each phase.

## Repo + branch + worktree

- **Repo:** `C:/Users/c3205/Documents/Code/python/draw` (main worktree — DO NOT touch; on `feat/ai-updates`)
- **Branch you must work on:** `final-form`
- **Worktree to use:** `C:/Users/c3205/Documents/Code/python/draw-final-form` — **always `cd` into here**
- **Push permission requires user approval.** Don't push yourself; ask the user.
- The main project doc `CLAUDE.md` (in the worktree root) lists project rules — read it before any code change.

## Read these first (MANDATORY)

1. `C:/Users/c3205/Documents/Code/python/draw-final-form/CLAUDE.md` — project rules
2. `C:/Users/c3205/Documents/Code/python/draw-final-form/docs/ATOMIZATION_PLAN.md` — full plan (now annotated with phase status)
3. `C:/Users/c3205/Documents/Code/python/draw-final-form/docs/CONSTRAINT_INVENTORY.md` — Phase 0 deliverable; the table you'll re-read for atom names + behavior
4. `C:/Users/c3205/Documents/Code/python/draw-final-form/docs/HELPER_VARS.md` — Phase 1 deliverable; HelperVarRegistry API atoms must use
5. `C:/Users/c3205/Documents/Code/python/draw-final-form/config/season_2026.py` — actual season config (lock this in your head before editing constraints)

## Current state of the branch (commit `<phase-3b>`)

Quick test bar (no slow integration tests):
```
cd /c/Users/c3205/Documents/Code/python/draw-final-form
timeout 240 /c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe -m pytest tests/ \
  --ignore=tests/test_solver_integration.py \
  --ignore=tests/test_spacing_integration.py -q
```

**Baseline: 1287 passed, 1 skipped** (was 1272 before Phase 3b shipped 15 new atom tests). Don't ship a phase that drops below 1287.

(Note: the venv lives in the *main* worktree `draw/.venv`, not in `draw-final-form/.venv`. Use the absolute path above.)

### Commits in the chain so far (master pickup point = `cd8a338`)

| Hash | Phase | Summary |
|---|---|---|
| `6e16d14` | 0 | `docs: add constraint inventory (Phase 0)` |
| `244f8cd` | 1 | `feat(constraints): extract HelperVarRegistry from SharedVariablePool (Phase 1)` |
| `c64c1d4` | 2 | `feat(constraints): extend ConstraintInfo for atomization (Phase 2)` |
| `535cac3` | 5 | `feat(config): migrate hardcoded constraint constants to CONSTRAINT_DEFAULTS (Phase 5)` |
| `48f5222` | 6 prep | `feat(config): add AWAY_VENUE_RULES skeleton for generic home-ground` |
| `1956608` | 3a | `feat(constraints): atomize PHLAndSecondGradeTimes into 8 atoms (Phase 3a)` |
| `3a4ab4d` | 3a retraction | `docs: hand-off — FORCED_GAMES supersedes per-venue Friday count atoms` |
| `e9bf5a7` | 3a retraction | `feat(forced-games): support 'club' filter + multi-scope subset consistency` |
| `5cfae6c` | 3a retraction | `refactor(constraints): retire per-venue Friday-count atoms for FORCED_GAMES` |
| `4f0777c` | 3a retraction | `docs: document FORCED-as-count-budget as a perennial rule` |
| `<phase-3b>` | 3b | `feat(constraints): atomize ClubDayConstraint into 5 atoms (Phase 3b)` |

### Phase 3a retraction — DONE (FORCED-as-count-rules)

`docs/FORCED_GAMES_AS_COUNT_RULES.md` is fully implemented. Cluster atom count is **5**: `PHLConcurrencyAtBroadmeadow`, `PHLAnd2ndConcurrencyAtBroadmeadow`, `GosfordFridayRoundsForced`, `PHLRoundOnePlay`, `PreferredDates`. (`GosfordFridayRoundsForced` is RETAINED — its per-round `sum == 1` enforcement isn't yet covered by individual per-round FORCED entries; verify before retiring in a future pass.)

What shipped:
1. `'club'` team-filter for `FORCED_GAMES` entries (mirrors `BLOCKED_GAMES`).
2. New `validate_game_config` Phase 21: subset-consistency for overlapping FORCED scopes (catches `equal N` ⊂ `equal M` when `N > M`, etc.).
3. `season_2026.py` FORCED_GAMES gained three per-venue Friday count entries (Broadmeadow ≤3, Gosford ==8, Maitland ==2).
4. `BroadmeadowFridayCount` / `GosfordFridayCount` / `MaitlandFridayCount` atoms + their registry entries + atom tests removed.
5. `docs/PERENNIAL_RULES.md` Rule 3 documents the count-budget-as-FORCED pattern for new seasons.
6. Test bar: **1272 passed, 1 skipped** (was 1268).

CONSTRAINT_DEFAULTS keys (`max_friday_broadmeadow`, `gosford_friday_games`, `maitland_friday_games`) are RETAINED for now — `original.py`/`ai.py` and the legacy `_phl_times_hard()` parity reference still consume them. They retire when legacy code archives in Phase 7c.

### What changed structurally (prior session)

- `constraints/helper_vars.py` (new) — `HelperVarRegistry` with declarative `declare`/`freeze`/`get_declared` API atoms must use, plus pool-style `get_or_create_bool`/`get_or_create_presence`/`register`/`lookup`/`get` (legacy compat). `SharedVariablePool` is an alias for `HelperVarRegistry`.
- `constraints/unified.py` — engine exposes `self.registry` (declarative entry point for atoms) and `self.pool` (legacy alias). Class-level `PHL_ADJACENCY_MINUTES`, `MAITLAND_AWAY_HARD_LIMIT`, `CLUBS_ON_FIELD_HARD_LIMIT`, `CLUB_GAME_SPREAD_HARD_LIMIT` removed (first migrated to config; the other three were dead code). `BROADMEADOW_MAX_SLOTS = 6` retained — it's a tuning threshold, not derivable. Hardcoded Gosford-Friday rounds `[2, 4, 5, 9, 10]` now read from `data['constraint_defaults']['gosford_friday_rounds']`.
- `constraints/registry.py` — `ConstraintInfo` extended with `atom_group` (str|None), `required_helpers` (list[str]), `forced_blocked_adjuster` (callable|None). `HELPER_VAR_CATALOG` set lists allowed helper kinds. New helpers `get_atoms_in_group`, `get_adjuster`, `validate_required_helpers`.
- `config/defaults.py` — adds perennial `CONSTRAINT_DEFAULTS` dict (every constraint param has a default; seasons override only what they want changed). New keys: `phl_adjacency_window_minutes`, `gosford_friday_rounds`, `worst_timeslot_time`. Adds `AWAY_VENUE_RULES` skeleton (Maitland + Gosford `max_consecutive_home`/`friday_games`/`max_away_clubs`) — no constraint reads from it yet.
- `utils.py` — `_merge_constraint_defaults()` merges season overrides over perennial defaults inside `build_season_data`.

### What's still in place

- `constraints/unified.py` (~1410 lines) — 2-stage hard/soft engine, 27+ pre-computed groupings, lookup caches. Atom dispatch is wired for `PHLAndSecondGradeTimes` (Phase 3a) AND `ClubDayConstraint` (Phase 3b — see `_club_day_atoms_hard`). The legacy `_club_day_scheduling` / `_club_day_field_contiguity` methods are retained as parity-reference only (not invoked by stage 1). Internal methods for `ClubVsClubAlignment` still hold legacy combined logic — Phase 3c.
- `constraints/registry.py` (~430 lines) — 31 entries (21 original + 5 PHL atoms + 5 ClubDay atoms). Phase 3c will add 4 more atom entries (with `atom_group='ClubVsClubAlignment'`).
- `constraints/atoms/` — 12 atoms shipped: 5 PHL (`phl_concurrency.py`, `phl_2nd_concurrency.py`, `gosford_friday_rounds.py`, `phl_round_one_play.py`, `preferred_dates.py`) + 5 ClubDay (`club_day_participation.py`, `club_day_intra_club_matchup.py`, `club_day_opponent_matchup.py`, `club_day_same_field.py`, `club_day_contiguous_slots.py`) + `_club_day_shared.py` helpers + `base.py` + `__init__.py`.
- `constraints/original.py` (1733 lines) and `constraints/ai.py` (2040 lines) — legacy combined classes. Reference-only. Move to `constraints/archived/` in Phase 7c.
- `constraints/archived_equalspacing_original.py` exists at the top level — move it to `constraints/archived/equalspacing_original.py` in Phase 7c.
- `utils.py` (~3813 lines) — 21-phase `validate_game_config` harness (Phase 21 added in `e9bf5a7` for FORCED subset consistency). `_build_forced_game_rules` accepts `'club'` filter (also `e9bf5a7`). `_get_matching_forced_scopes` does multi-scope FORCED match. `generate_X` returns `(X, conflicts)` — NOT `(X, Y, conflicts)`.
- `analytics/tester.py` (~2495 lines) — `_check_forced_games`, `_check_blocked_games`, dynamic per-club `ClubGameSpread` overlap bound. Phase 7a-bis adds the per-club / per-type breakdown.
- `tests/fixtures/draw_2026_first6weeks.json` exists. `tests/atoms/conftest.py` has the small PHL+2nd fixture used by atom tests; Phase 7a creates `tests/fixtures/violations/`.

## Decisions already locked (do NOT re-litigate)

| # | Decision |
|---|---|
| 1 | Atom names use `PHLConcurrencyAtBroadmeadow`-style descriptive names. No `Atom` suffix. |
| 2/3 | Legacy `original.py` + `ai.py` move to `constraints/archived/`. Pipeline imports forbidden. Test enforces. |
| 4 | `ClubDayOpponentMatchup` atom matches `original.py` behavior (supports `{'date': ..., 'opponent': 'OppClub'}`). |
| 5 | Per-club home-venue config key is `AWAY_VENUE_RULES` — already added in `config/defaults.py`. |
| 6 | Tests use both static fixtures (one per atom-violation) AND programmatic per-test construction. Plus per-club / per-type breakdown in `ViolationReport`. |
| 7 | `SOLVER_STAGES` is a config-driven list of `{name, description, atoms, ...}` dicts. CLI flags `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`. |

## Critical project facts you MUST remember

These are common pitfalls — they're in `CLAUDE.md` too but are easy to get wrong:

1. **Variable key is an 11-tuple** `(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)`. `team1` is alphabetically first. Home/away is determined by `field_location`, NOT by team1/team2 position.
2. **Dummy keys are 4-tuples** `(team1, team2, grade, index)`. Skip them via `len(key) < 11` or `not key[3]`.
3. **`generate_X` signature: `(X, conflicts)`** — final-form does NOT use the `(X, Y, conflicts)` 3-tuple; dummies are handled via `HelperVarRegistry`.
4. **A FORCED variable can match multiple scopes.** Use `_get_matching_forced_scopes()` (returns list), not `_check_forced_game_status()` (returns first match only — back-compat wrapper only).
5. **Aggregate per-team home/away constraint was removed deliberately.** Per-pair balance only. CLAUDE.md notes "by design". *Phase 0 inventory found the aggregate block is still in `original.py:426-447`* — the atom must drop it; flag for user sign-off.
6. **PHL locked-week HACKs in `constraints/original.py`** (lines ~242-301) are explicit user-flagged tech debt. They become obsolete when (a) per-venue Friday count budgets moved to `FORCED_GAMES` (DONE — `e9bf5a7`/`5cfae6c`); (b) `original.py`/`ai.py` archive in Phase 7c. Until then, the legacy classes still consume `CONSTRAINT_DEFAULTS['max_friday_broadmeadow']` etc. and the HACK still adjusts for locked weeks. Don't write new code that depends on the HACK's substring matching — use `home_field_map` lookups.
7. **Maitland Sunday slot is 13:30 (not 13:00)** — `season_2026.py` uses 13:30. If you see 13:00 anywhere, it's stale.
8. **`home_field_map` already exists** at `data['home_field_map']` (e.g. `{'Maitland': 'Maitland Park', 'Gosford': 'Central Coast Hockey Park'}`). Defaults to Newcastle International Hockey Centre (Broadmeadow) for unlisted clubs.
9. **Solver runs are LONG (hours/days).** Never run `python run.py generate ...` synchronously. Use background jobs for solver tests; for unit tests build small fixtures and run for ≤30s.
10. **The `--ai` CLI flag is being deprecated.** After atomization there's no original/AI distinction. Don't add new code paths that use `args.ai`.
11. **`HelperVarRegistry.get(key)`** returns the cached pool var (or `None`) — NOT the declarative var. Use `registry.get_declared(kind, key)` for the declarative API.

## How to work

Iterate one phase at a time. Order from `ATOMIZATION_PLAN.md`:

```
[0✅ inventory] ──┐
                 ├─→ [1✅ helper-var registry] ─→ [3a✅] ─→ [3b✅ ClubDay] ─→ 3c (ClubVsClub) ─→ 7c move legacy
[2✅ constraint registry extend] ─┘                                                       │
                                                                                           ↓
                                                                     4 (FORCED/BLOCKED adjusters)
                                                                                           ↓
[5✅ constants migration] ────────────────────────────────────────────────────────────────┤
[6 prep ✅] 6 (generic home-ground rename) ───────────────────────────────────────────────┤
                                                                                           ↓
7a (tests) ─ 7b (configurable stages) ─ 7d (docs)
```

**Per-phase workflow:**
1. Re-read the relevant phase section in `ATOMIZATION_PLAN.md` (now annotated with status).
2. Plan the work (use `TaskCreate` to track sub-tasks within the phase).
3. Implement with `Edit`/`Write`. NEVER touch `constraints/archived/` (once that exists) — those files are reference-only.
4. Run the test bar:
   ```
   cd /c/Users/c3205/Documents/Code/python/draw-final-form
   timeout 240 /c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe -m pytest tests/ \
     --ignore=tests/test_solver_integration.py --ignore=tests/test_spacing_integration.py -q
   ```
   Must pass at ≥1287 (current baseline) before you commit.
5. Commit on `final-form` with a descriptive message. Use `git commit --no-verify` ONLY if the user explicitly authorizes (don't skip pre-commit hooks proactively).
6. Report to the user using the template at the end of this doc.
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

## Phase-by-phase pickup notes

### Phase 3 — Atomize the 3 multi-idea constraints

#### Phase 3a — `PHLAndSecondGradeTimes` → 5 atoms — ✅ DONE
Atoms shipped in `1956608` (8 originally) then trimmed to 5 in `e9bf5a7`+`5cfae6c`:
`PHLConcurrencyAtBroadmeadow`, `PHLAnd2ndConcurrencyAtBroadmeadow`,
`GosfordFridayRoundsForced`, `PHLRoundOnePlay`, `PreferredDates`. Per-venue
Friday count atoms became `FORCED_GAMES` entries (see
`docs/FORCED_GAMES_AS_COUNT_RULES.md`).

#### Phase 3b — `ClubDayConstraint` → 5 atoms — ✅ DONE

Atoms shipped: `ClubDayParticipation`, `ClubDayIntraClubMatchup`,
`ClubDayOpponentMatchup`, `ClubDaySameField`, `ClubDayContiguousSlots`. Each
in its own file under `constraints/atoms/`. Shared parsing/lookup helpers in
`constraints/atoms/_club_day_shared.py`.

Wired into `constraints/unified.py:_club_day_atoms_hard` — replaces the legacy
`_club_day_scheduling` + `_club_day_field_contiguity` pair (kept callable for
parity reference; not dispatched). `unified.py:build_groupings` now calls
`normalize_club_day` so the `{'date': ..., 'opponent': ...}` dict form parses
correctly.

Per Decision #4 the atoms additionally enforce the opponent-matchup branch
from `original.py:ClubDayConstraint` that the pre-atomization unified engine
silently dropped — the legacy unified methods only ever ran the derby branch.

Tests:
- `tests/atoms/test_club_day_atoms.py` — 13 solo-clean / solo-violation tests
- `tests/atoms/test_club_day_atoms_parity.py` — 2 parity tests vs legacy methods
- `tests/atoms/club_day_fixture.py` — 3-club Broadmeadow fixture with
  duplicate 3rd-grade teams for derby + opponent scenarios.

Test bar lifted to **1287 passed, 1 skipped** (was 1272 before Phase 3b).

Atom helpers used: `club_day_field_used` (SameField), `club_day_slot_used`
(ContiguousSlots) — both already in `HELPER_VAR_CATALOG`.

#### Phase 3c — `ClubVsClubAlignment` → 4 atoms

3 atoms (`ClubVsClubCoincidence`, `ClubVsClubFieldLimit`, `ClubVsClubDeficitPenalty`) + extracted `PHLAnd2ndBackToBackSameField`. See `CONSTRAINT_INVENTORY.md` for the behaviour breakdown.

### Phase 4 — FORCED/BLOCKED count adjusters

Once Phase 3 atoms exist, register a `forced_blocked_adjuster` callable on each count-sensitive atom's `ConstraintInfo`. Catalog from the plan:

| Atom | Adjuster behavior |
|---|---|
| `ClubVsClubCoincidence` | `expected_meetings = total_meetings - forced_off_sunday - blocked_on_sunday` per (club_pair, grade) |
| `EqualMatchUpSpacing` | If pair forced into N specific weeks, reduce flexibility budget by N |
| `MaitlandHomeGrouping` (→ `NonDefaultHomeGrouping`) | Forced home weekend changes consecutive-window calc |
| `AwayAtMaitlandGrouping` (→ `AwayAtNonDefaultGrouping`) | Forced away match at venue X changes away-clubs-per-week count |
| `EqualGames` | If team is FORCED into N games, per-team budget = num_rounds - N |

(Per-venue Friday count atoms are gone — those budgets are FORCED_GAMES entries directly, no adjuster needed.)

Each adjuster has signature:
```python
def adjuster(constraint_data: dict, forced_games: list, blocked_games: list) -> dict:
    """Return count overrides keyed by whatever the constraint cares about."""
```

Engine runs every adjuster after FORCED/BLOCKED parsing, before atom `apply()`. Output is stored at `data['count_adjustments'][canonical_name]`. Atoms read it.

**Report each adjuster formula to the user before committing.** They want to sanity-check.

### Phase 6 — Generic home-ground rename

Renames + per-club iteration (no longer hardcode "Maitland" / "Gosford"):

| Today | Generic |
|---|---|
| `MaitlandHomeGrouping` | `NonDefaultHomeGrouping` (per non-default-home club) |
| `MaxMaitlandHomeWeekends` | folded into `NonDefaultHomeGrouping` (was redundant) |
| `AwayAtMaitlandGrouping` | `AwayAtNonDefaultGrouping` per club |

- Constraint logic iterates over `data['home_field_map'].keys()` instead of hardcoded `'Maitland'` / `'Gosford'` strings.
- Per-club tuning comes from `AWAY_VENUE_RULES` (already added to `config/defaults.py`).
- Keep registry aliases: old canonical names (`MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`, `MaxMaitlandHomeWeekends`) point to the new entries via `solver_class_names` so existing tests + severity/slack lookups keep working.
- Touches ~20 files / ~100 references — go carefully.

### Phase 7 — Tests, configurable stages, archive, docs

- **7a:** static fixtures in `tests/fixtures/violations/` (one per atom that benefits) + programmatic per-test construction. `ViolationReport.breakdown` (by_club, by_type, by_severity, soft_pressure) — see plan section 7a-bis.
- **7b:** `SOLVER_STAGES` list in config (default in `config/defaults.py`, season override allowed). CLI flags `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`. Validation runs in `validate_game_config`. Drop the hardcoded `STAGES` / `STAGES_AI` dicts in `main_staged.py`.
- **7c:** move `constraints/original.py`, `constraints/ai.py`, `constraints/archived_equalspacing_original.py` to `constraints/archived/`. `tests/test_no_legacy_imports.py` greps prod code (everything outside `constraints/archived/` and `tests/`) and fails if any module imports those. Remove `--ai` CLI flag.
- **7d:** rewrite `CLAUDE.md` constraint sections; add `docs/HELPER_VARS.md` already (Phase 1 deliverable), add `docs/HARNESS.md` and `docs/STAGES.md`; move full constraint table out of `CLAUDE.md` into `docs/CONSTRAINT_INVENTORY.md` (single source of truth).

## Useful commands

```bash
# always work from the final-form worktree
cd /c/Users/c3205/Documents/Code/python/draw-final-form

# venv is in the MAIN worktree (final-form has no .venv of its own)
PY=/c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe

# quick test run
timeout 240 $PY -m pytest tests/ \
  --ignore=tests/test_solver_integration.py \
  --ignore=tests/test_spacing_integration.py -q

# helper-var registry — verify Phase 1 wiring
$PY -c "from constraints.helper_vars import HelperVarRegistry; r = HelperVarRegistry(None); print(r.diagnostics())"

# inspect the registry (now with atom_group / required_helpers / adjuster)
$PY -c "from constraints.registry import CONSTRAINT_REGISTRY; \
[print(n, '->', i.severity_level, 'atoms:', i.atom_group, 'helpers:', i.required_helpers) \
 for n, i in CONSTRAINT_REGISTRY.items()]"

# check that perennial CONSTRAINT_DEFAULTS merged into season data correctly
$PY -c "from config import load_season_data; d = load_season_data(2026); \
print({k: d['constraint_defaults'][k] for k in ('phl_adjacency_window_minutes', 'gosford_friday_rounds', 'worst_timeslot_time')})"

# count constraint classes in original.py / ai.py (for archive verification)
grep -c "^class.*Constraint\b" constraints/original.py constraints/ai.py

# inspect the helper-var catalog
$PY -c "from constraints.registry import HELPER_VAR_CATALOG; print(sorted(HELPER_VAR_CATALOG))"
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
