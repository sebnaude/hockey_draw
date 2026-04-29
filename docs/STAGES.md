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

## Status (as of `final-form` after the Phase-6/7a/7b commit `67474f4`)

- ✅ `DEFAULT_STAGES` config + `constraints/stages.py` API + 10 tests in
  `tests/test_solver_stages.py`.
- ⏳ `main_staged.py` dispatch rewire — still uses legacy `STAGES` /
  `STAGES_AI` dicts. Tracked in `docs/ATOMIZATION_HANDOFF.md`.
- ⏳ CLI flags `--stages-config`, `--stage-only`, `--skip-stage`,
  `--list-stages` — pending in same follow-up commit.
