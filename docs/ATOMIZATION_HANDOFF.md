# Hand-off prompt — Atomization, FINAL PUSH

> Paste this entire document into a fresh Claude Code session as the first
> user message. It is self-contained.
>
> **This is the last push.** Phases 0, 1, 2, 3a, 3b, 3c, 5, Phase-6 prep, and
> Phase-4 framework are merged on `final-form`. You are doing the rest in
> one continuous session: Phase 4 per-adjuster implementations, Phase 6
> generic home-ground rename, Phase 7a tests + violation breakdowns, Phase
> 7b configurable stages, Phase 7c archive legacy, Phase 7d docs. Do not
> stop and ask between phases unless you find a real blocker — the
> formulas, decisions, and architecture are all locked in. Run the test bar
> after every phase, fix what's broken, commit, move on.

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

1. `CLAUDE.md` — project rules (atomization-aware as of Phase 3c)
2. `docs/ATOMIZATION_PLAN.md` — full plan, annotated with phase status
3. `docs/CONSTRAINT_INVENTORY.md` — single source of truth for constraint → atom mapping
4. `docs/HELPER_VARS.md` — `HelperVarRegistry` API atoms must use
5. `docs/COUNT_ADJUSTERS.md` — Phase 4 framework + the 5 approved adjuster formulas
6. `docs/FORCED_GAMES_AS_COUNT_RULES.md` — perennial rule for count budgets
7. `config/season_2026.py` — actual season config (lock this in your head before editing constraints)

## Current state of `final-form`

Quick test bar (no slow integration tests):
```
cd /c/Users/c3205/Documents/Code/python/draw-final-form
timeout 240 /c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe -m pytest tests/ \
  --ignore=tests/test_solver_integration.py \
  --ignore=tests/test_spacing_integration.py -q
```

**Baseline at start: 1305 passed, 1 skipped.** Don't ship a phase that drops
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

### What's already in place

- `constraints/atoms/` — 14 atoms across 3 groups: PHL (5), ClubDay (5),
  ClubVsClub (4) + shared helpers + `base.py` + `__init__.py`
- `constraints/registry.py` — 35 entries (21 originals + 14 atoms),
  `run_count_adjusters(data)` plumbing, `HELPER_VAR_CATALOG`
- `constraints/unified.py` — atom dispatch wired for all 3 groups; legacy
  `_phl_times_*`, `_club_day_*`, `_club_alignment_*` retained as
  parity-reference (not invoked in stage 1/2)
- `constraints/helper_vars.py` — `HelperVarRegistry` (declarative + pool API)
- `config/defaults.py` — perennial `CONSTRAINT_DEFAULTS`,
  `AWAY_VENUE_RULES` skeleton (per-club home-venue rules; no constraint
  reads from it yet — that's Phase 6)
- `tests/atoms/` — atom-level tests + 3 fixtures (PHL, ClubDay, CvC)
- `tests/test_count_adjusters.py` — 7 framework tests for Phase 4
- `docs/` — inventory, plan, helper-vars, count-adjusters,
  forced-games-as-count-rules, perennial rules

### What's still in place that needs to go

- `constraints/original.py` (~1733 lines) and `constraints/ai.py` (~2040
  lines) — legacy combined classes. Pipeline must NOT import them in prod
  by end of Phase 7c.
- `constraints/archived_equalspacing_original.py` at the top level — also
  moves to `constraints/archived/` in 7c.
- Hardcoded `STAGES` and `STAGES_AI` dicts in `main_staged.py` — Phase 7b
  replaces with config-driven `SOLVER_STAGES`.
- `--ai` CLI flag in `run.py` — Phase 7c removes it.
- Hardcoded `'Maitland'` / `'Gosford'` strings in constraint logic —
  Phase 6 replaces with iteration over `home_field_map.keys()` and
  `AWAY_VENUE_RULES`.

## Decisions already locked (do NOT re-litigate)

| # | Decision |
|---|---|
| 1 | Atom names use `PHLConcurrencyAtBroadmeadow`-style descriptive names. No `Atom` suffix. |
| 2/3 | Legacy `original.py` + `ai.py` move to `constraints/archived/`. Pipeline imports forbidden. Test enforces. |
| 4 | `ClubDayOpponentMatchup` atom matches `original.py` behaviour (supports `{'date': ..., 'opponent': 'OppClub'}` form). DONE in 3b. |
| 5 | Per-club home-venue config key is `AWAY_VENUE_RULES` — already in `config/defaults.py`. |
| 6 | Tests use both static fixtures (one per atom-violation) AND programmatic per-test construction. Plus per-club / per-type breakdown in `ViolationReport`. |
| 7 | `SOLVER_STAGES` is a config-driven list of `{name, description, atoms, ...}` dicts. CLI flags `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`. |
| 8 | Phase 4 adjuster formulas in `docs/COUNT_ADJUSTERS.md` are **approved spec** — implement them as written. |

## Critical project facts you MUST remember

These are common pitfalls — they're in `CLAUDE.md` too but easy to get wrong:

1. **Variable key is an 11-tuple** `(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)`. `team1` is alphabetically first. Home/away is determined by `field_location`, NOT by team1/team2 position.
2. **Dummy keys are 4-tuples** `(team1, team2, grade, index)`. Skip them via `len(key) < 11` or `not key[3]`.
3. **`generate_X` signature: `(X, conflicts)`** — final-form does NOT use the `(X, Y, conflicts)` 3-tuple; dummies are handled via `HelperVarRegistry`.
4. **A FORCED variable can match multiple scopes.** Use `_get_matching_forced_scopes()` (returns list), not `_check_forced_game_status()` (returns first match only — back-compat wrapper only).
5. **Aggregate per-team home/away constraint was removed deliberately.** Per-pair balance only. The aggregate block is still in `original.py:426-447` — when archiving the legacy classes (7c), this drops away naturally.
6. **PHL locked-week HACKs in `constraints/original.py`** are explicit user-flagged tech debt. They become obsolete when `original.py`/`ai.py` archive in 7c.
7. **Maitland Sunday slot is 13:30 (not 13:00)** — `season_2026.py` uses 13:30. If you see 13:00 anywhere, it's stale.
8. **`home_field_map` already exists** at `data['home_field_map']` (e.g. `{'Maitland': 'Maitland Park', 'Gosford': 'Central Coast Hockey Park'}`). Defaults to Newcastle International Hockey Centre (Broadmeadow) for unlisted clubs.
9. **Solver runs are LONG (hours/days).** Never run `python run.py generate ...` synchronously. Use small fixtures for unit tests; never run the full solver from a Phase 7 task.
10. **`HelperVarRegistry.get(key)`** returns the cached pool var (or `None`) — NOT the declarative var. Use `registry.get_declared(kind, key)` for the declarative API.
11. **`coincide` BoolVars** for ClubVsClub are registered under pool prefixes
    `cvc_coincide` (lower-grade) and `cvc_phl_btb_coincide` (PHL/2nd) — atoms
    after `Coincidence` / `BackToBack` read them by these prefix keys.
12. **Phase-4 count adjustments** live at `data['count_adjustments'][canonical_name]`.
    `UnifiedConstraintEngine.build_groupings()` calls `run_count_adjusters(data)`
    once before any atom `apply()`. Atoms read by canonical name.

## How to work

**Per-phase workflow:**
1. Re-read the relevant phase section in `ATOMIZATION_PLAN.md` and the
   "What I'll do per phase" detail below.
2. Plan with `TaskCreate` to track sub-tasks within the phase.
3. Implement with `Edit`/`Write`. NEVER touch `constraints/archived/` (once
   that exists) — those files are reference-only.
4. Run the test bar:
   ```
   cd /c/Users/c3205/Documents/Code/python/draw-final-form
   timeout 240 /c/Users/c3205/Documents/Code/python/draw/.venv/Scripts/python.exe -m pytest tests/ \
     --ignore=tests/test_solver_integration.py --ignore=tests/test_spacing_integration.py -q
   ```
   Must pass at the running baseline (start: 1305) before you commit each phase.
5. Commit on `final-form` with a descriptive message. Do NOT use
   `--no-verify` unless explicitly authorized.
6. Update **all** of these docs at end of each phase:
   - `docs/ATOMIZATION_HANDOFF.md` — phase status + commit table
   - `docs/ATOMIZATION_PLAN.md` — phase status table
   - `docs/CONSTRAINT_INVENTORY.md` — atom annotations if relevant
   - `docs/COUNT_ADJUSTERS.md` — adjuster status table (Phase 4)
   - `CLAUDE.md` — file structure, pitfalls, anything that drifts
   - `README.md` — if dir structure changes
7. Move on. Don't pause for sign-off. Final report at the very end.

**Do NOT:**
- Push without user approval (push permission is denied; ask at the end).
- Modify `feat/ai-updates` (work only on `final-form`).
- Modify the main worktree at `C:/Users/c3205/Documents/Code/python/draw`.
- Invent new constraint behavior. You're re-shaping existing logic into
  atoms; behavioral changes get raised separately for sign-off first.
- Use `unittest.mock`, `monkeypatch`, or any mocking on solver/data dicts
  in tests. Real CP-SAT models with sampled data only.
- Add docstrings beyond what was there. Don't over-comment.
- Run the full solver (it takes hours). Use small fixtures.
- Touch the open Excel files at the repo root (`2026 Season Draw V*.xlsx`
  etc.) — they're gitignored as drafts.

---

# THE WORK

## Phase 4 — FORCED/BLOCKED count adjusters (5 implementations)

Framework already shipped in `1521c9b`. The 5 formulas in
`docs/COUNT_ADJUSTERS.md` are **approved spec** — implement them as written.

### Where to put each adjuster

Each adjuster is a free function (or staticmethod) that lives next to its
atom. Convention:

```python
# constraints/atoms/<atom_file>.py — bottom of the file

def <atom_name>_adjuster(data, forced_games, blocked_games):
    """One-line description.

    Returns dict keyed by whatever the atom needs.
    """
    ...

# Wire into the registry on import:
from constraints.registry import CONSTRAINT_REGISTRY
CONSTRAINT_REGISTRY['<atom_canonical_name>'].forced_blocked_adjuster = <atom_name>_adjuster
```

(The `CONSTRAINT_REGISTRY` mutation pattern is OK because each atom file is
imported exactly once via `constraints/atoms/__init__.py`.)

### Per-adjuster implementations

#### #1 `EqualGames` — no adjuster needed

`docs/COUNT_ADJUSTERS.md` formula #1 explains why: FORCED entries pin
specific X vars to 1, but `sum(team_vars) == num_rounds` is unchanged
(FORCED just makes some terms equal 1; the sum still has to add up to
num_rounds). No code change needed.

**Action:** add a one-paragraph note in `docs/COUNT_ADJUSTERS.md` confirming
"no adjuster needed; analysis verified" and update the status table.

Note: the `EqualGames` atom doesn't actually exist yet on final-form — it's
still inside `EqualGamesAndBalanceMatchUps`. Don't split it just for this;
the analysis applies regardless.

#### #2 `EqualMatchUpSpacing` adjuster

**Atom:** `EqualMatchUpSpacing` (canonical name; not yet split into its own
atom file — the legacy classes still own this. Wire the adjuster against
the registry entry `'EqualMatchUpSpacing'`. The atom file you create or
edit reads `data['count_adjustments']['EqualMatchUpSpacing']`.)

**Adjuster spec:**
```
forced_rounds_per_pair: Dict[(t1, t2, grade), Set[int]]
  for each FORCED entry that pins (t1, t2, grade) into specific weeks/rounds:
    accumulate the rounds.
return forced_rounds_per_pair
```

The atom uses this to compute `effective_T = T - len(forced_rounds_per_pair[pair])`
and clamp the spacing window down. If the pair has 0 forced rounds, behaviour is
unchanged.

**Tests:**
- Adjuster math: synthesise FORCED entries pinning (t1, t2, grade) into weeks
  W1, W2; assert `forced_rounds_per_pair[(t1,t2,grade)] == {W1, W2}`.
- Atom integration: clean fixture passes; same fixture + adjuster output
  showing 2 forced rounds → atom uses tighter min_gap.

#### #3 `MaitlandHomeGrouping` adjuster (will rename to `NonDefaultHomeGrouping` in Phase 6)

**Atom:** `MaitlandHomeGrouping` for now; rename in Phase 6.

**Adjuster spec:**
```
forced_home_weeks_per_club: Dict[club_name, Set[int]]
  for each FORCED entry that pins a (team_of_club_C, _, ..., field_location=home_field_map[C]):
    accumulate the week.
return forced_home_weeks_per_club
```

The atom uses this to clamp the per-week home indicator to 1 for those
weeks (so the sliding-window math accounts for them as definite home weekends).

**Tests:** synthesise FORCED entry pinning Maitland home weekend in week W;
assert adjuster returns `{'Maitland': {W}}`; assert atom no longer permits
W to be 0 in the home indicator.

#### #4 `AwayAtMaitlandGrouping` adjuster (will rename to `AwayAtNonDefaultGrouping` in Phase 6)

**Atom:** `AwayAtMaitlandGrouping` for now; rename in Phase 6.

**Adjuster spec:**
```
forced_away_clubs_per_week: Dict[(week, venue), Set[away_club]]
  for each FORCED entry that pins a game at field_location=V where V != home of either team:
    away_club = the team's club whose home is NOT V
    accumulate (week, V) -> away_club
return forced_away_clubs_per_week
```

The atom uses `len(forced_away_clubs_per_week[(week, V)])` as a baseline
floor for the per-week away-clubs count.

**Tests:** synthesise FORCED Norths-vs-Maitland Friday at Maitland Park,
week W; assert adjuster returns `{(W, 'Maitland Park'): {'Norths'}}`.

#### #5 `ClubVsClubCoincidence` adjuster (the user's worked example)

**Atom:** `ClubVsClubCoincidence` (already exists, Phase 3c).

**Adjuster spec:**
```
expected_meetings: Dict[grade, Dict[club_pair, int]]
  for each grade, club_pair in by_grade_clubpair_round:
    total = num_meetings(club_pair, grade) per the season  # see below
    forced_off_sunday = count of FORCED entries pinning that pair on a non-Sunday day
    blocked_on_sunday = count of BLOCKED entries removing Sunday vars for that pair
    expected_meetings[grade][club_pair] = max(0, total - forced_off_sunday - blocked_on_sunday)
return expected_meetings
```

`total` is the BalancedMatchups expected meeting count: for grade with R rounds
and T teams, base = R // (T-1) for even T, R // T for odd. Each pair gets
either base or base+1 meetings. The adjuster can use `base` as a conservative
lower bound, or compute exact per-pair if available.

The atom uses `expected_meetings[grade][club_pair]` instead of the constant
`num_games` when computing
`min_required = max(0, expected_meetings - slack)`.

**Tests:**
- Synthesised FORCED on Friday Gosford for (Maitland PHL, Norths PHL) round 5,
  count=2 → assert `expected_meetings['PHL'][('Maitland', 'Norths')] == total - 2`.
- Atom integration: with adjuster output, the lower-grade alignment block
  doesn't penalize "missing" coincidences for those forced rounds.

### Phase 4 sweep at end

- All 5 adjuster statuses in `docs/COUNT_ADJUSTERS.md` flipped from 🟡 → ✅
- Test bar should rise by ~15-25 tests (3-5 per adjuster).
- Commit message: `feat(constraints): implement Phase 4 FORCED/BLOCKED count adjusters`

---

## Phase 6 — Generic home-ground rename

**Goal:** stop hardcoding `'Maitland'` / `'Gosford'` in constraint logic.
Iterate over `home_field_map.keys()` instead. Per-club tuning comes from
`AWAY_VENUE_RULES` (already in `config/defaults.py`).

### Renames

| Today | Generic | Behaviour |
|---|---|---|
| `MaitlandHomeGrouping` | `NonDefaultHomeGrouping` | Per non-default-home club, no back-to-back home weekends (sliding window). Slack from `AWAY_VENUE_RULES[club]['max_consecutive_home']`. |
| `MaxMaitlandHomeWeekends` | folded into `NonDefaultHomeGrouping` | Was redundant once `MaitlandHomeGrouping` switched to sliding window. |
| `AwayAtMaitlandGrouping` | `AwayAtNonDefaultGrouping` | Per non-default-home club, max away clubs per week. Limit from `AWAY_VENUE_RULES[club]['max_away_clubs']`. |

### Files touched (~20 files / ~100 references — go carefully)

- `constraints/unified.py` — `_maitland_grouping_hard`, `_maitland_grouping_soft`, `_max_venue_weekends`, `_away_maitland_hard`, `_away_maitland_soft`. Rename methods, replace string literals, iterate `home_field_map`.
- `constraints/registry.py` — add new canonical entries `NonDefaultHomeGrouping`, `AwayAtNonDefaultGrouping`. Keep old entries (`MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`, `MaxMaitlandHomeWeekends`) as aliases pointing at the new entries (preserve `solver_class_names`, `slack_key`, `tester_check_methods` so existing tests + severity/slack lookups keep working).
- `analytics/tester.py` — `_check_maitland_back_to_back`, `_check_maitland_away_clubs_limit` rename + iterate non-default clubs. Keep old method names as thin aliases that call the new ones (so registry's `tester_check_methods` lookups don't break).
- `constraints/severity.py` — update `CONSTRAINT_TO_SEVERITY` map.
- `constraints/original.py` / `ai.py` — leave alone (legacy, will archive in 7c).
- `tests/` — update assertions referencing old names; add a fictional 3rd
  non-default-home club to a fixture and assert constraints scope to it.

### Behaviour rules

- "Non-default home" = `home_field_map.get(club_name)` exists AND is not Broadmeadow.
- For each non-default-home club, the new constraints scope by that club's
  `home_field_map` value.
- `AWAY_VENUE_RULES[club]` provides per-club tuning. Default values live in
  `config/defaults.py`; missing keys fall back to historical Maitland values.
- Removing a club from `home_field_map` makes its constraints go silent
  rather than crash — verify with a test.

### Phase 6 sweep at end

- Adjusters #3 and #4 (Phase 4) updated to register against the new
  `NonDefaultHomeGrouping` / `AwayAtNonDefaultGrouping` entries.
- `docs/CONSTRAINT_INVENTORY.md` rows for `MaitlandHomeGrouping` /
  `AwayAtMaitlandGrouping` annotated with the rename.
- `CLAUDE.md` "Severity Levels" table updated.
- Commit message: `refactor(constraints): generic home-ground (Phase 6)`

---

## Phase 7a — Tests + violation breakdowns

### Static violation fixtures

Create `tests/fixtures/violations/` with one JSON per atom that benefits
from a static fixture:

```
tests/fixtures/violations/
  club_game_spread_overlap.json     # _violations: ["ClubGameSpread"]
  club_grade_adjacency.json         # _violations: ["ClubGradeAdjacency"]
  phl_friday_overcount.json         # _violations: ["ForcedGames"]  (count budget)
  home_away_imbalance.json          # _violations: ["FiftyFiftyHomeAway"]
  club_day_split_field.json         # _violations: ["ClubDaySameField"]
  club_day_opponent_missing.json    # _violations: ["ClubDayOpponentMatchup"]
  cvc_field_excess.json             # _violations: ["ClubVsClubFieldLimit"]
  phl_2nd_no_back_to_back.json      # _violations: ["PHLAnd2ndBackToBackSameField"]
```

Each fixture has top-level metadata:
```json
{
  "_violations": ["AtomCanonicalName1", "..."],
  "_description": "Why this fixture exists.",
  "...rest is normal DrawStorage shape..."
}
```

A new test in `tests/atoms/test_violation_fixtures.py` walks the directory,
loads each fixture, runs `DrawTester.run_violation_check()`, and asserts
that **at least** the listed violations are flagged. Fixtures may flag more
than they list (other latent issues) — that's fine.

### `ViolationReport.breakdown`

Extend `analytics/tester.py`:

```python
@dataclass
class ViolationBreakdown:
    by_club: Dict[str, List[Violation]]
    by_type: Dict[str, List[Violation]]   # canonical name -> violations
    by_severity: Dict[str, List[Violation]]  # 'CRITICAL'/.../'VERY LOW' -> violations
    soft_pressure: Dict[str, Dict[str, Any]]
    # ^ canonical_name -> { 'at_limit': N_clubs, 'over_limit': N_clubs,
    #                       'total_penalty': X, 'worst_club': 'ClubName',
    #                       'worst_value': X }

@dataclass
class ViolationReport:
    ...existing fields...
    breakdown: ViolationBreakdown
```

Each `Violation` gains `affected_clubs: List[str]` (already there via
existing fields if not, add it) and `metric_value: Optional[float]`.
`ViolationReport.breakdown` aggregates without re-deriving.

**Test pattern:** load a violations fixture, run check, assert
`report.breakdown.by_type` has the expected keys; assert
`soft_pressure[constraint_name]['at_limit']` is the right club count.

### Phase 7a sweep at end

- ≥8 fixtures in `tests/fixtures/violations/`
- ≥10 new tests in `tests/atoms/test_violation_fixtures.py` and
  `tests/test_violation_breakdown.py`
- Commit message: `test(analytics): static violation fixtures + ViolationReport.breakdown (Phase 7a)`

---

## Phase 7b — Configurable stages

### `SOLVER_STAGES` config

Add to `config/defaults.py`:
```python
DEFAULT_STAGES = [
    {
        'name': 'critical_feasibility',
        'description': 'Hard feasibility — every constraint that must hold for a valid draw',
        'atoms': [
            'NoDoubleBookingTeams', 'NoDoubleBookingFields',
            'EqualGamesAndBalanceMatchUps',
            'PHLConcurrencyAtBroadmeadow', 'PHLAnd2ndConcurrencyAtBroadmeadow',
            'GosfordFridayRoundsForced', 'PHLRoundOnePlay',
        ],
    },
    {
        'name': 'home_away_balance',
        'atoms': [
            'FiftyFiftyHomeandAway',
            'NonDefaultHomeGrouping', 'AwayAtNonDefaultGrouping',
        ],
    },
    {
        'name': 'club_alignment',
        'atoms': [
            'ClubGradeAdjacency',
            'ClubVsClubCoincidence', 'ClubVsClubFieldLimit',
            'PHLAnd2ndBackToBackSameField',
        ],
    },
    {
        'name': 'club_day',
        'atoms': [
            'ClubDayParticipation', 'ClubDayIntraClubMatchup',
            'ClubDayOpponentMatchup', 'ClubDaySameField', 'ClubDayContiguousSlots',
        ],
    },
    {
        'name': 'soft_optimisation',
        'atoms': [
            'EqualMatchUpSpacing', 'ClubGameSpread',
            'ClubVsClubDeficitPenalty', 'PreferredDates',
            'EnsureBestTimeslotChoices', 'PreferredTimes',
            'MaximiseClubsPerTimeslotBroadmeadow', 'MinimiseClubsOnAFieldBroadmeadow',
        ],
    },
]
```

Per-stage optional fields: `time_limit_seconds`, `use_prior_solution_as_hint`,
`soft_only`, `requires_complete_solution`. Defaults documented in `docs/STAGES.md`.

### CLI

`run.py`: add `--stages-config <path>` (override config-defined stages),
`--stage-only NAME`, `--skip-stage NAME`, `--list-stages`.

### Pipeline rewire

- `main_staged.py::main_staged()` reads `data['solver_stages']`, validates,
  iterates stages.
- For each stage, instantiate the listed atoms via the registry, dispatch
  through the unified engine.
- Delete the hardcoded `STAGES` and `STAGES_AI` dicts in `main_staged.py`.

### Validation

New `_validate_stages` phase in `validate_game_config`:
- Every atom listed in any stage must exist in `CONSTRAINT_REGISTRY`.
- Every registered atom (excluding `tester_only=True`) must appear in
  exactly one stage (warn if missing; fail if duplicate).

### Phase 7b sweep at end

- `--list-stages` works on `season_2026`.
- Round-trip: edit `SOLVER_STAGES` to put `EqualGames` in stage 2 → solver
  validates and produces a draw (test on a tiny fixture, not full solve).
- ≥8 new tests in `tests/test_configurable_stages.py`.
- Commit message: `feat(solver): config-driven SOLVER_STAGES + CLI flags (Phase 7b)`

---

## Phase 7c — Move legacy classes + remove `--ai`

### Move

- `constraints/original.py` → `constraints/archived/original.py`
- `constraints/ai.py` → `constraints/archived/ai.py`
- `constraints/archived_equalspacing_original.py` → `constraints/archived/equalspacing_original.py`
- New `constraints/archived/__init__.py` (empty or with `__all__ = []`)
- New `constraints/archived/README.md` explaining: "These are
  pre-atomization implementations kept for historical reference. Do NOT
  import in production code; the pipeline is locked against this. Use the
  atoms in `constraints/atoms/` instead."

### Lock down imports

- `constraints/__init__.py` exports ONLY atoms + `UnifiedConstraintEngine` +
  `registry`. No re-export of archived modules.
- `tests/test_no_legacy_imports.py` (new): greps the prod codebase
  (everything outside `constraints/archived/` and `tests/`) for
  `from constraints.original`, `from constraints.ai`, `from constraints.archived`
  and fails if any prod module imports them.
- Update `main_staged.py`, `analytics/*`, `run.py`, every script in
  `scripts/` to import only via the registry.

### Remove `--ai` flag

- `run.py`: drop the `--ai` CLI flag.
- `main_staged.py`: drop `args.ai` usage and the `STAGES_AI` dict (already
  scheduled for delete in 7b).
- Update tests that reference `--ai`.

### Update parity tests

The atom-level parity tests in `tests/atoms/test_*_parity.py` reference
legacy methods on `UnifiedConstraintEngine` (`_phl_times_hard`,
`_club_day_scheduling`, `_club_alignment_hard`, etc.). These methods are
defined inline in `unified.py`, NOT imported from `original.py` / `ai.py`,
so they survive the archive move. Verify the parity tests still pass.

### Phase 7c sweep at end

- `tests/test_no_legacy_imports.py` passes.
- All atom parity tests still pass.
- Full test suite passes at the running baseline + Phase 7a/7b additions.
- Commit message: `refactor: archive legacy constraint classes; remove --ai (Phase 7c)`

---

## Phase 7d — Documentation

### Rewrite

- `CLAUDE.md` — re-do the constraint architecture sections: atoms +
  registry replace the "ai vs original" model. Move the constraint table
  to `docs/CONSTRAINT_INVENTORY.md` (single source) and reference it from
  CLAUDE.md.
- `docs/HARNESS.md` (new) — describe the
  `generate_X` → `validate_game_config` → `UnifiedConstraintEngine`
  pipeline, including the count-adjuster step.
- `docs/STAGES.md` (new) — `SOLVER_STAGES` config schema, per-stage options,
  CLI flags, examples.
- `docs/CONSTRAINT_INVENTORY.md` — sweep all rows to mark Phase 6 renames
  and confirm atom-shipped status.
- `README.md` — final pass; remove Phase-3 "in-flight" caveats, point at
  the docs index.

### Final sweep

- `docs/ATOMIZATION_HANDOFF.md` (this file) — mark all phases ✅ DONE,
  remove "next phase" pointers.
- `docs/ATOMIZATION_PLAN.md` — phase status table all ✅; "Definition of
  done" checklist all ticked.

Commit message: `docs: Phase 7d sweep — atomization complete`

---

# WHEN YOU'RE DONE

Run the test bar one final time. Confirm:
- All Phase 7c imports clean (`tests/test_no_legacy_imports.py` green).
- Test count substantially higher than 1305 (expect 1400+ after all
  Phase 7 additions).
- No skipped tests other than the original 1.

Then summarise to the user:

```
## Atomization complete — all phases ✅

**Final commit chain (Phase 4 onwards):** <hashes>

**Test bar:** 1305 → <final> passed, 1 skipped

**Files touched (totals):**
- New atoms: <N>
- New tests: <N>
- Renamed constraints: <list>
- Files moved to constraints/archived/: 3

**Phases 4 / 6 / 7a / 7b / 7c / 7d — all done.** Ready for push approval.

Open questions / things flagged during the work:
- ...
```

Then ask the user for push approval before `git push`.

Work carefully. Commit small per-phase. Verify often. Don't pause for
sign-off — the spec is locked. If you find a real bug in the spec, flag it
in your final summary, not mid-work.
