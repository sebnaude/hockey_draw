# Solver Stages — `SOLVER_STAGES` config

Phase 7b foundation. Lists how solver stages are configured by canonical atom
name, validated, and inspected. This document covers the foundation that
shipped (`config/defaults.py::DEFAULT_STAGES`, `constraints/stages.py`); the
`main_staged.py` rewire and CLI flags are tracked in
`docs/ATOMIZATION_HANDOFF.md` as the "final-final" remaining work.

## Schema

A stage is a dict with these keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | str | yes | Short identifier; unique within the stage list. Used by `--stage-only` / `--skip-stage` CLI. |
| `description` | str | no | Human-readable summary printed by `--list-stages`. |
| `atoms` | list[str] | yes | Canonical atom names from `CONSTRAINT_REGISTRY`. Non-empty, no duplicates across stages. |
| `time_limit_seconds` | int/float | no | Per-stage solver time limit. Defaults to `SEASON_CONFIG['max_time_per_stage']`. |
| `use_prior_solution_as_hint` | bool | no | Default True. If False, the stage starts from scratch. |
| `soft_only` | bool | no | Default False. If True, only soft penalties; no hard constraints added. |
| `requires_complete_solution` | bool | no | Default True. If False, the stage can run on partial solutions. |

## Default stages (`config/defaults.py::DEFAULT_STAGES`)

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
        'description': 'Per-pair home/away + non-default-home grouping',
        'atoms': [
            'FiftyFiftyHomeandAway',
            'NonDefaultHomeGrouping', 'AwayAtNonDefaultGrouping',
        ],
    },
    {
        'name': 'club_alignment',
        'description': 'Cross-grade coincidence + field limits',
        'atoms': [
            'ClubGradeAdjacency',
            'ClubVsClubCoincidence', 'ClubVsClubFieldLimit',
            'PHLAnd2ndBackToBackSameField',
        ],
    },
    {
        'name': 'club_day',
        'description': 'Per-club day-of-week constraints',
        'atoms': [
            'ClubDayParticipation', 'ClubDayIntraClubMatchup',
            'ClubDayOpponentMatchup', 'ClubDaySameField', 'ClubDayContiguousSlots',
        ],
    },
    {
        'name': 'soft_optimisation',
        'description': 'Soft penalties and optimisation',
        'soft_only': True,
        'atoms': [
            'EqualMatchUpSpacing', 'ClubGameSpread',
            'ClubVsClubDeficitPenalty', 'PreferredDates',
            'EnsureBestTimeslotChoices', 'PreferredTimes',
            'MaximiseClubsPerTimeslotBroadmeadow', 'MinimiseClubsOnAFieldBroadmeadow',
        ],
    },
]
```

Season configs may override by setting `'solver_stages'` in their
`SEASON_CONFIG` dict. The override fully replaces `DEFAULT_STAGES` (no merge).

## API

```python
from constraints.stages import (
    load_solver_stages, validate_solver_stages, list_stages,
)

stages = load_solver_stages(season_config)  # falls back to DEFAULT_STAGES
errors = validate_solver_stages(stages)     # list[str], empty = valid
print(list_stages(stages))                  # human-readable
```

`validate_solver_stages` enforces:
- Every stage has `name` (string, unique) and `atoms` (non-empty list).
- Every atom name exists in `CONSTRAINT_REGISTRY`.
- No atom appears in more than one stage.
- Optional keys are spelled correctly.

## Dispatcher

`constraints/stages.py::apply_solver_stage` is the single entry point that
the solver uses to apply a stage's atoms to the model. It splits each stage's
atoms into:

- **engine atoms** — handled by `UnifiedConstraintEngine` via its
  `skip_constraints` set. The dispatcher computes the set of engine
  skip-keys for the stage and runs `apply_stage_1_hard` / `apply_stage_2_soft`
  with the inverse skip mask. `applied_engine_keys` is threaded across
  stages so a cluster's hard atoms never get re-added.
- **non-engine atoms** — atoms whose canonical names don't map to an
  engine method (e.g. `MaximiseClubsPerTimeslotBroadmeadow`). The
  dispatcher resolves the legacy solver class via `solver_class_names` in
  the registry and instantiates it directly.

The mapping atom canonical name → engine skip-key lives in
`atom_to_engine_key()`. Atoms with `atom_group` set (atomized clusters)
return their parent's name; Phase-6 aliases (`NonDefaultHomeGrouping`,
`AwayAtNonDefaultGrouping`, `PreferredTimes`) return their legacy
counterparts. Atoms whose canonical name is itself a legacy combined name
return that name. Anything else returns `None` (legacy class fallback).

`StagedScheduleSolver.run_solver_stages_solve` drives the loop: builds
engine + groupings once, then per-stage applies the dispatcher, builds
the objective, solves, saves a checkpoint, and carries the solution as
hints into the next stage.

## CLI flags

`run.py generate` accepts:

| Flag | Purpose |
|---|---|
| `--stages-config FILE` | Load a custom stage list from a JSON file (replaces season config + defaults). |
| `--stage-only NAME` | Restrict the run to a single stage by name. |
| `--skip-stage NAME` | Skip a stage by name. May be passed multiple times. |
| `--list-stages` | Print the resolved stage list and exit. Honours the other three flags. |

`_resolve_solver_stages` in `run.py` wires the flags into the stage list.
Validation runs before solving; an unknown atom or invalid stage name
exits non-zero.

## Config validation phase

`utils.py::validate_game_config` includes a Phase 22 step
`_validate_stages` that runs `validate_solver_stages(data['solver_stages'])`
when the key is present. Any errors become FATAL config errors via the
existing fatals/warnings collection.

## Status (as of `final-form` after Phase 7c completion commit `0140495`)

- ✅ `DEFAULT_STAGES` config + `constraints/stages.py` API + 10 tests in
  `tests/test_solver_stages.py`.
- ✅ `main_staged.py` dispatch rewire (`run_solver_stages_solve`).
  Default `main_staged()` path is now SOLVER_STAGES; `severity_staged`
  builds its stage list from the registry via
  `severity_solver_stages()`.
- ✅ CLI flags `--stages-config`, `--stage-only`, `--skip-stage`,
  `--list-stages`. 18 dispatch tests in `tests/test_solver_stages_dispatch.py`.
- ✅ Config-validation Phase 22 `_validate_stages`.
- ✅ Legacy `STAGES` / `STAGES_AI` / `STAGES_UNIFIED` /
  `STAGES_SEVERITY[_AI]` dicts deleted (Phase 7c).
- ✅ `--ai` CLI flag removed; `run_list_constraints` reads the
  registry; `run_diagnose` ported to drive the unified engine via
  `apply_solver_stage` with cluster-level removal testing.
