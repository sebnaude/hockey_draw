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

The live default stage list is defined in `config/defaults.py::DEFAULT_STAGES`
(starting at line 144); this document deliberately does not duplicate the atom
membership of each stage to avoid drift. Read the config file directly when you
need the current canonical atom names for a stage.

The default pipeline ships five stages, in the order they run. Each stage's
*intent* — what the stage exists to enforce in the solve, independent of which
atoms currently implement it — is summarised below.

| Stage name | Intent |
|---|---|
| `critical_feasibility` | Hard feasibility prerequisites for any valid draw — no team or field is double-booked, every team plays its required games against the right opponents, season-wide PHL / 2nd-grade concurrency rules at Broadmeadow hold, same-club PHL/2nd games are back-to-back at one venue or ≥3 h apart across venues, byes are spread evenly across the season, and repeat meetings of any pair are spaced out (matchup spacing, hard since spec-017, with `--slack` relief). spec-021 added two hard structural rules here: `VenueEarliestSlotFill` (games at a venue pack into the earliest timeslots — no gaps + earliest start, which structurally avoids 7pm) and `ClubNoConcurrentSlot` (a club's games per timeslot/venue capped capacity-aware via `no_field_slots`). Gosford-Friday-round placement is no longer here — it's expressed via `FORCED_GAMES` count entries (spec-015). Anything in this stage is non-negotiable; failure here means the draw cannot be published. |
| `home_away_balance` | Home/away fairness for clubs whose home venue is not Broadmeadow. Enforces the per-pair and per-team aggregate balance of home vs away games and pins per-club season totals for home weekends (Friday / Sunday / overall). (The consecutive-home-week cap and away-clubs-per-weekend cap were removed in spec-018 — there is no longer any weekend *sequencing* dimension.) |
| `club_alignment` | Cross-grade, same-club / same-opponent alignment at Broadmeadow. Decides which weekends two clubs' grades stack together at the venue and, when they do stack, how their games co-locate on fields and timeslots. Order-sensitive within the stage: the stacking decision must be made before the co-location decision that depends on it. |
| `club_day` | Per-club day-of-week scheduling preferences. For each club that has a preferred playing day, governs which of its teams participate that day, how same-club matchups slot in, how opponents are routed onto that day, field consistency across the day, and contiguity of the club's slots so a club's day reads as a single block. spec-021 moved `ClubGameSpread` here (from `soft_optimisation`, where its hard part was dead): a club's games on a day form a near-contiguous block — ≤3 games allow no holes, ≥4 allow one — with a soft penalty driving residual holes to zero. |
| `soft_optimisation` | Soft penalties and optimisation only — no new hard constraints. Honours preferred / avoided dates and weekends, biases the solver toward the convenor's preferred times, encourages multiple clubs per Broadmeadow timeslot while keeping any one field from being monopolised by a single club, applies a stable alphabetical tie-break for matchups, prefers the canonical NIHC field-fill order WF→EF→SF (soft symmetry-breaker, spec-016), and honours convenor-supplied per-team-pair no-concurrency requests. (The matchup-spacing *density* penalty still runs too — its atom now lives in `critical_feasibility`, whose dispatch always applies the soft pass alongside the hard one.) |

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
  engine method (e.g. `PreferredTimes`). The
  dispatcher resolves the legacy solver class via `solver_class_names` in
  the registry and instantiates it directly.

The mapping atom canonical name → engine skip-key lives in
`atom_to_engine_key()`. Atoms with `atom_group` set (atomized clusters)
return their parent's name; Phase-6 aliases (e.g. `PreferredTimes`) return
their legacy counterparts. (The `NonDefaultHomeGrouping` /
`AwayAtNonDefaultGrouping` aliases were removed in spec-018.) Atoms whose
canonical name is itself a legacy combined name
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

## Regeneration group (spec-026 → spec-027)

A regeneration run (`--regen-from`, spec-026) selects a dedicated **`regen`
constraint group** via `run.py::_select_regen_group()` →
`resolve_groups(['regen'])` (the groups machinery is spec-023; the `regen`
group's soft atoms are spec-027). The regen group softens the hard rules that
assume the solver controls game times (adjacency/spacing/co-location) so that
frozen-but-retimed pins stay feasible — see `spec-027-regen-soft-constraint-group.md`.

Until spec-023 and spec-027 land, `_select_regen_group()` returns `None`, prints
`WARNING: spec-027 regen group not available …`, and the regen run falls back to
the full hard constraint set (feasible only for small scopes such as
`--regen-grades 6th`). A normal (non-regen) run never selects the `regen` group.

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
