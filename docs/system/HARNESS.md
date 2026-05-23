# Solver Harness — pipeline reference

End-to-end shape of the solver run on `final-form`. Covers what
`generate_X` produces, how `validate_game_config` shapes the data dict, what
the unified engine does in each phase, and where the count-adjuster step
slots in.

## Pipeline overview

```
SEASON_CONFIG (config/season_2026.py)
        │
        ▼
build_season_data(config)         # config/__init__.py → utils.py
        │   ├─ _merge_constraint_defaults  # perennial defaults + season overrides
        │   ├─ _merge_away_venue_rules     # AWAY_VENUE_RULES + season overrides
        │   └─ load teams / clubs / grades / timeslots
        ▼
data: dict
        │
        ▼
validate_game_config(data)        # 20-phase config validation harness
        │
        ▼
generate_X(model, data)           # decision-variable construction (utils.py)
        │   ├─ home_field_map filter        — eliminate cross-club venue games
        │   ├─ PHL_GAME_TIMES filter        — restrict PHL to configured slots
        │   ├─ SECOND_GRADE_TIMES filter    — same for 2nd grade
        │   ├─ FORCED_GAMES rules           — register against ALL matching scopes
        │   └─ BLOCKED_GAMES elimination    — drop variables matching scope+team
        │   returns (X, conflicts)
        ▼
UnifiedConstraintEngine(model, X, data)
        │
        ▼
engine.build_groupings()
        │   ├─ single pass over X populates ~30 grouping dicts
        │   ├─ run_count_adjusters(data)    — Phase 4 adjusters fire here
        │   │   stashes results under data['count_adjustments'][canonical_name]
        │   └─ HelperVarRegistry materialised lazily on first .get()
        ▼
engine.apply_stage_1_hard()       # CP-SAT hard constraints
        │   └─ atoms / legacy methods read data['count_adjustments']
        ▼
engine.apply_stage_2_soft()       # soft penalties + optimisation
        │
        ▼
solver.Solve(model)
        │
        ▼
convert_X_to_roster(...)          # solution → DrawStorage
        │
        ▼
DrawStorage.export_schedule_xlsx(...) / DrawTester.run_violation_check(...)
```

## Data dict — keys consumed by the engine

Output of `build_season_data()`. Constraints can read any key but typically
use:

| Key | Producer | Consumed by |
|---|---|---|
| `teams`, `grades`, `clubs`, `fields`, `timeslots` | `build_season_data` | every constraint |
| `home_field_map` | season config | AwayClubHomeWeekendsCount, AwayClubPerOpponentAndAggregateHomeBalance, generate_X home filter |
| `away_venue_rules` | `_merge_away_venue_rules` | (no active consumers — the home-grouping constraints that read it were removed in spec-018) |
| `constraint_defaults` | `_merge_constraint_defaults` | every constraint with a tunable threshold |
| `constraint_slack` | CLI `--slack` | spacing, alignment, broadmeadow constraints |
| `forced_games` | season config | generate_X, ForcedGames tester check, Phase 4 adjusters |
| `blocked_games` | season config | generate_X, BlockedGames tester check, Phase 4 adjusters |
| `count_adjustments` | `run_count_adjusters` (during build_groupings) | atoms / legacy methods that need FORCED-aware counts |
| `locked_weeks` | `--locked` + `--lock-weeks` CLI | engine groupings skip these |
| `penalties` | engine populates per soft constraint | objective function |
| `solver_stages` | season config (Phase 7b foundation) | `constraints/stages.py` (not yet wired into main_staged.py) |

## Decision variables

Variable key is an 11-tuple:

```
(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
```

- `team1` is alphabetically first (sorted on creation).
- Home/away is determined by `field_location`, NOT by team1/team2 position.
- Dummy keys are 4-tuples `(team1, team2, grade, index)`. Atoms skip them
  via `len(key) < 11` or `not key[3]`.
- A FORCED variable can match multiple scopes; `_get_matching_forced_scopes`
  returns ALL matches so each scope's `sum == N` constraint sees the variable.

## Helper variable registry

`HelperVarRegistry` (pool-style API) + `SharedVariablePool` alias for
back-compat live in `constraints/helper_vars.py`. The engine assigns
`self.registry = HelperVarRegistry(model)` and aliases `self.pool = self.registry`
so legacy methods keep working.

`HELPER_VAR_CATALOG` in `constraints/registry.py` lists allowed kinds. Atoms
declaring a `required_helpers` kind not in the catalog fail registry
validation.

## Count-adjuster step

Per-constraint adjusters registered on `ConstraintInfo.forced_blocked_adjuster`
fire once during `build_groupings()`. See `docs/COUNT_ADJUSTERS.md` for the
shipped adjusters and their formulas.

## Unified engine — atom dispatch on `final-form`

The engine dispatches via atoms, not the legacy combined classes:

| Stage | Method | Atom dispatch |
|---|---|---|
| 1 hard | `_phl_times_atoms_hard` | `PHL_HARD_ATOMS` (PHLConcurrencyAtBroadmeadow, PHLAnd2ndConcurrencyAtBroadmeadow — **PHLRoundOnePlay removed by spec-010; GosfordFridayRoundsForced removed by spec-015**, now FORCED_GAMES count entries) |
| 1 hard | `_club_day_atoms_hard` | `CLUB_DAY_HARD_ATOMS` (5 atoms) |
| 1 hard | `_club_vs_club_atoms_hard` | `CLUB_VS_CLUB_HARD_ATOMS` (Coincidence, FieldLimit, PHLAnd2ndBackToBackSameField) |
| 2 soft | `_phl_times_atoms_soft` | `PHL_SOFT_ATOMS` (PreferredDates) |
| 2 soft | `_club_vs_club_atoms_soft` | `CLUB_VS_CLUB_SOFT_ATOMS` (DeficitPenalty) |

Legacy `_phl_times_*`, `_club_day_*`, `_club_alignment_*` methods in
`unified.py` are retained as parity reference but **not invoked**. Phase 7c
will move them out to `constraints/archived/` once import sites are migrated.

## Tester

`analytics.tester.DrawTester` runs post-hoc constraint checks against a
`DrawStorage`. Key entry points:

```python
DrawTester.from_X_solution(X_solution, data, description='...')
DrawTester.from_file('draws/2026/current.json', data)
report = tester.run_violation_check()
report.breakdown   # Phase 7a — by_club / by_type / by_severity / soft_pressure
```

Slack-aware: tester reads `data['constraint_slack'][...]` and applies the
same window math the engine uses.

## What's NOT in this doc

- `docs/STAGES.md` — `SOLVER_STAGES` config + validation API.
- `docs/COUNT_ADJUSTERS.md` — Phase 4 adjuster formulas + status.
- `docs/HELPER_VARS.md` — `HelperVarRegistry` declarative API.
- `docs/CONSTRAINT_INVENTORY.md` — single source of truth for the
  constraint → atom mapping.
- `docs/ATOMIZATION_HANDOFF.md` — remaining work for the next session.
